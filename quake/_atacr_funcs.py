# -*- coding: utf-8 -*-
"""

"""

from obstools.atacr import StaNoise, DayNoise, TFNoise, EventStream
# from obspy.clients.fdsn import Client
import obspy
import numpy as np
import shutil
import stdb
import os
monthdict   = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

class atacr_event_sta(object):
    
    def __init__(self, inv, datadir, outdir, noisedir, otime, overlap = 0.3, chan_rank = ['L', 'H', 'B'], sps = 1.):
        network     = inv.networks[0]
        station     = network[0]
        channel     = station[0]
        self.stdb_inv   = stdb.StDbElement(network = network.code, station = station.code, channel = channel.code[:2],\
            location = channel.location_code, latitude = station.latitude, longitude = station.longitude,\
                elevation = station.elevation, polarity=1., azcorr=0., startdate = station.start_date,\
                    enddate = station.end_date, restricted_status = station.restricted_status)
        self.datadir    = datadir
        self.outdir     = outdir
        self.noisedir   = noisedir
        self.otime      = otime
        self.overlap    = overlap
        self.staid      = network.code+'.'+station.code
        self.chan_rank  = chan_rank
        self.stlo       = station.longitude
        self.stla       = station.latitude
        self.monthdir   = self.noisedir + '/%04d.%s' %(self.otime.year, monthdict[self.otime.month])
        self.daydir     = self.monthdir+'/%d.%s.%d' %(self.otime.year, monthdict[self.otime.month], self.otime.day)
        self.sps        = sps
        return
    
    
    def transfer_func(self):
        """compute daily transfer function
        """
        targetdt        = 1./self.sps
        oyear           = self.otime.year
        omonth          = self.otime.month
        oday            = self.otime.day
        ohour           = self.otime.hour
        omin            = self.otime.minute
        osec            = self.otime.second
        label           = '%d_%s_%d_%d_%d_%d' %(oyear, monthdict[omonth], oday, ohour, omin, osec)
        self.eventdir   = self.datadir +'/'+label
        self.outeventdir= self.outdir +'/'+label
        self.label      = label
        chan_type       = None
        # load SAC data
        for chtype in self.chan_rank:
            fname1  = self.eventdir + '/%s_%sH1.SAC' %(self.staid, chtype)
            fname2  = self.eventdir + '/%s_%sH2.SAC' %(self.staid, chtype)
            fnamez  = self.eventdir + '/%s_%sHZ.SAC' %(self.staid, chtype)
            fnamep  = self.eventdir + '/%s_%sDH.SAC' %(self.staid, chtype)
            if os.path.isfile(fname1) and os.path.isfile(fname1) and \
                os.path.isfile(fnamez) and os.path.isfile(fnamep):
                chan_type   = chtype
                break
        if chan_type is None:
            return 0
        self.chan_type  = chan_type
        fname1          = self.eventdir + '/%s_%sH1.SAC' %(self.staid, chan_type)
        fname2          = self.eventdir + '/%s_%sH2.SAC' %(self.staid, chan_type)
        fnamez          = self.eventdir + '/%s_%sHZ.SAC' %(self.staid, chan_type)
        fnamep          = self.eventdir + '/%s_%sDH.SAC' %(self.staid, chan_type)
        self.sth        = obspy.read(fname1)
        self.sth        += obspy.read(fname2)
        self.sth        += obspy.read(fnamez)
        self.stp        = obspy.read(fnamep)
        #
        
        if abs(self.sth[0].stats.delta - targetdt) > 1e-3 or abs(self.sth[1].stats.delta - targetdt) > 1e-3 or \
            abs(self.sth[2].stats.delta - targetdt) > 1e-3 or abs(self.stp[0].stats.delta - targetdt) > 1e-3:
            raise ValueError('!!! CHECK fs :'+ self.staid)
        else:
            self.sth[0].stats.delta = targetdt
            self.sth[1].stats.delta = targetdt
            self.sth[2].stats.delta = targetdt
            self.stp[0].stats.delta = targetdt
        stime_event     = self.sth[-1].stats.starttime
        etime_event     = self.sth[-1].stats.endtime
        self.sth.trim(starttime = stime_event, endtime = etime_event, pad = True, nearest_sample = True, fill_value = 0.)
        self.stp.trim(starttime = stime_event, endtime = etime_event, pad = True, nearest_sample = True, fill_value = 0.)
        self.window     = self.sth[-1].stats.npts / self.sps
        # trim data
        
        # load daily noise data
        daystr          = '%d.%s.%d.%s' %(self.otime.year, monthdict[self.otime.month], self.otime.day, self.staid)
        dfname1         = self.daydir + '/ft_%s.%sH1.SAC' %(daystr, chan_type)
        dfname2         = self.daydir + '/ft_%s.%sH2.SAC' %(daystr, chan_type)
        dfnamez         = self.daydir + '/ft_%s.%sHZ.SAC' %(daystr, chan_type)
        dfnamep         = self.daydir + '/ft_%s.%sDH.SAC' %(daystr, chan_type)
        if not( os.path.isfile(dfname1) and os.path.isfile(dfname2) and \
                os.path.isfile(dfnamez) and os.path.isfile(dfnamep)):
            return 0
        tr1             = obspy.read(dfname1)[0]
        tr2             = obspy.read(dfname2)[0]
        trZ             = obspy.read(dfnamez)[0]
        trP             = obspy.read(dfnamep)[0]
        
        if abs(tr1.stats.delta - targetdt) > 1e-3 or abs(tr2.stats.delta - targetdt) > 1e-3 or \
                abs(trZ.stats.delta - targetdt) > 1e-3 or abs(trP.stats.delta - targetdt) > 1e-3:
                raise ValueError('!!! CHECK fs :'+ self.staid)
        else:
            tr1.stats.delta     = targetdt
            tr2.stats.delta     = targetdt
            trP.stats.delta     = targetdt
            trZ.stats.delta     = targetdt
                
        # trim data
        slidind_wlength = self.window - int(self.overlap*self.window)*tr1.stats.delta
        stime_noise     = tr1.stats.starttime
        newtime         = np.floor((tr1.stats.endtime - stime_noise)/slidind_wlength) * slidind_wlength
        tr1.trim(starttime = stime_noise, endtime = stime_noise + newtime)
        tr2.trim(starttime = stime_noise, endtime = stime_noise + newtime)
        trZ.trim(starttime = stime_noise, endtime = stime_noise + newtime)
        trP.trim(starttime = stime_noise, endtime = stime_noise + newtime)
        
        if np.all(trP.data == 0.) and not (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)):
            self.daynoise   = DayNoise(tr1=tr1, tr2=tr2, trZ=trZ, trP=obspy.Trace(), overlap=self.overlap, window = self.window)
            self.out_dtype  = 'Z2-1'
        elif (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)) and (not np.all(trP.data == 0.)):
            self.daynoise   = DayNoise(tr1=obspy.Trace(), tr2=obspy.Trace(), trZ=trZ, trP=trP, overlap=self.overlap, window = self.window)
            self.out_dtype  = 'ZP'
        elif (not (np.all(tr1.data == 0.) or np.all(tr2.data == 0.))) and (not np.all(trP.data == 0.)):
            self.daynoise   = DayNoise(tr1=tr1, tr2=tr2, trZ=trZ, trP=trP, overlap=self.overlap, window = self.window)
            self.out_dtype  = 'ZP-21'
        else:
            return 0
        # self.daynoise.QC_daily_spectra()
        # self.daynoise.average_daily_spectra()
        # self.tfnoise    = TFNoise(self.daynoise)
        # self.tfnoise.transfer_func()
        try:
            self.daynoise.QC_daily_spectra()
            self.daynoise.average_daily_spectra()
            self.tfnoise    = TFNoise(self.daynoise)
            self.tfnoise.transfer_func()
        except:
            return -1
        return 1
    
    def correct(self):
        """compute monthly transfer function
        """
        tmptime = self.sth[-1].stats.starttime
        tstamp  = str(tmptime.year).zfill(4)+'.' + \
            str(tmptime.julday).zfill(3)+'.'
        tstamp  = tstamp + str(tmptime.hour).zfill(2) + \
                    '.'+str(tmptime.minute).zfill(2)
        eventstream = EventStream( sta = self.stdb_inv, sth = self.sth, stp = self.stp,\
                    tstamp = tstamp, lat = self.stla, lon = self.stlo, time = tmptime,\
                    window = self.window, sampling_rate = 1., ncomp = 4)
        eventstream.correct_data(self.tfnoise)
        # save data
        if not os.path.isdir(self.outeventdir):
            os.makedirs(self.outeventdir)
        outTrZ      = (self.sth.select(channel = '??Z')[0]).copy()
        outTrZ.data = eventstream.correct[self.out_dtype].copy()
        outfnameZ   = self.outeventdir + '/%s_%sHZ.SAC' %(self.staid, self.chan_type)
        if os.path.isfile(outfnameZ):
            shutil.copyfile(src = outfnameZ, dst = outfnameZ+'_old')
            os.remove(outfnameZ)
        outTrZ.write(outfnameZ, format = 'SAC')
        
        
            
            
def atacr_for_mp(in_atacr_sta, verbose = False):
    if verbose:
        print('=== station :'+self.staid)
    if not in_atacr_sta.transfer_func():
        return
    in_atacr_sta.correct()
    staid   = in_atacr_sta.staid
    logfname= self.outdir+'/logs_atacr/'+self.label+'/'+staid+'.log'
    with open(logfname, 'w') as fid:
        fid.writelines('SUCCESS\n')
    return
    