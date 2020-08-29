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

class atacr_monthly_sta(object):
    
    def __init__(self, inv, datadir, outdir, year, month, overlap = 0.5, window = 21000., chan_rank = ['L', 'H', 'B'], sps = 1.):
        network     = inv.networks[0]
        station     = network[0]
        channel     = station[0]
        self.stdb_inv   = stdb.StDbElement(network = network.code, station = station.code, channel = channel.code[:2],\
            location = channel.location_code, latitude = station.latitude, longitude = station.longitude,\
                elevation = station.elevation, polarity=1., azcorr=0., startdate = station.start_date,\
                    enddate = station.end_date, restricted_status = station.restricted_status)
        self.datadir    = datadir
        self.outdir     = outdir
        self.year       = year
        self.month      = month
        self.overlap    = overlap
        self.window     = window
        self.staid      = network.code+'.'+station.code
        self.chan_rank  = chan_rank
        self.stanoise   = StaNoise()
        self.stlo       = station.longitude
        self.stla       = station.latitude
        self.monthdir   = self.datadir + '/%04d.%s' %(self.year, monthdict[self.month])
        self.sps        = sps
        return
    
    def transfer_func(self):
        """compute monthly transfer function
        """
        targetdt    = 1./self.sps
        stime       = obspy.UTCDateTime('%04d%02d01' %(self.year, self.month))
        monthdir    = self.monthdir
        if not os.path.isdir(monthdir):
            return False
        chan_type   = None
        Nday        = 0
        while( (stime.year == self.year) and (stime.month == self.month)):
            daydir  = monthdir+'/%d.%s.%d' %(self.year, monthdict[self.month], stime.day)
            if not os.path.isdir(daydir):
                stime   += 86400.
                continue 
            fpattern= daydir+'/ft_%d.%s.%d.%s' %(self.year, monthdict[self.month], stime.day, self.staid)
            if chan_type is None:
                for chtype in self.chan_rank:
                    fname1  = fpattern+'.%sH1.SAC' %chtype
                    fname2  = fpattern+'.%sH2.SAC' %chtype
                    fnamez  = fpattern+'.%sHZ.SAC' %chtype
                    fnamep  = fpattern+'.%sDH.SAC' %chtype
                    if os.path.isfile(fname1) and os.path.isfile(fname1) and \
                        os.path.isfile(fnamez) and os.path.isfile(fnamep):
                        chan_type   = chtype
                        break
            if chan_type is None:
                stime   += 86400.
                continue
            fname1  = fpattern+'.%sH1.SAC' %chan_type
            fname2  = fpattern+'.%sH2.SAC' %chan_type
            fnamez  = fpattern+'.%sHZ.SAC' %chan_type
            fnamep  = fpattern+'.%sDH.SAC' %chan_type
            if not (os.path.isfile(fname1) and os.path.isfile(fname1) and \
                    os.path.isfile(fnamez) and os.path.isfile(fnamep)):
                stime   += 86400.
                continue
            tr1     = obspy.read(fname1)[0]
            stimetr = tr1.stats.starttime
            tr2     = obspy.read(fname2)[0]
            trZ     = obspy.read(fnamez)[0]
            trP     = obspy.read(fnamep)[0]
            
            if abs(tr1.stats.delta - targetdt) > 1e-3 or abs(tr2.stats.delta - targetdt) > 1e-3 or \
                abs(trZ.stats.delta - targetdt) > 1e-3 or abs(trP.stats.delta - targetdt) > 1e-3:
                raise ValueError('!!! CHECK fs :'+ self.staid)
            else:
                tr1.stats.delta     = targetdt
                tr2.stats.delta     = targetdt
                trP.stats.delta     = targetdt
                trZ.stats.delta     = targetdt
            # # # if np.all(tr1.data == 0.) or np.all(tr2.data == 0.) or np.all(trZ.data == 0.) or np.all(trP.data == 0.):
            # # #     stime   += 86400.
            # # #     continue
            tr1.trim(starttime = stimetr, endtime = stimetr + 8400.*10-1)
            tr2.trim(starttime = stimetr, endtime = stimetr + 8400.*10-1)
            trZ.trim(starttime = stimetr, endtime = stimetr + 8400.*10-1)
            trP.trim(starttime = stimetr, endtime = stimetr + 8400.*10-1)
            
            # # # print (self.staid)
            if np.all(trP.data == 0.) and not (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)):
                self.stanoise   += DayNoise(tr1=tr1, tr2=tr2, trZ=trZ, trP=obspy.Trace(), overlap=self.overlap, window = self.window)
                self.out_dtype  = 'Z2-1'
            elif (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)) and (not np.all(trP.data == 0.)):
                self.stanoise   += DayNoise(tr1=obspy.Trace(), tr2=obspy.Trace(), trZ=trZ, trP=trP, overlap=self.overlap, window = self.window)
                self.out_dtype  = 'ZP'
            elif (not (np.all(tr1.data == 0.) or np.all(tr2.data == 0.))) and (not np.all(trP.data == 0.)):
                self.stanoise   += DayNoise(tr1=tr1, tr2=tr2, trZ=trZ, trP=trP, overlap=self.overlap, window = self.window)
                self.out_dtype  = 'ZP-21'
            else:
                stime   += 86400.
                continue
            stime   += 86400.
            Nday    += 1
        if Nday <= 1:
            return False
        self.stanoise.QC_sta_spectra()
        self.stanoise.average_sta_spectra()
        self.tfnoise= TFNoise(self.stanoise)
        self.tfnoise.transfer_func()
        return True
    
    def correct(self):
        """compute monthly transfer function
        """
        targetdt    = 1./self.sps
        stime       = obspy.UTCDateTime('%04d%02d01' %(self.year, self.month))
        monthdir    = self.monthdir
        omonthdir   = self.outdir + '/%04d.%s' %(self.year, monthdict[self.month])
        if not os.path.isdir(monthdir):
            return False
        chan_type   = None
        while( (stime.year == self.year) and (stime.month == self.month)):
            daydir      = monthdir+'/%d.%s.%d' %(self.year, monthdict[self.month], stime.day)
            fpattern    = daydir+'/ft_%d.%s.%d.%s' %(self.year, monthdict[self.month], stime.day, self.staid)
            odaydir     = omonthdir+'/%d.%s.%d' %(self.year, monthdict[self.month], stime.day)
            if not os.path.isdir(odaydir):
                os.makedirs(odaydir)
            if chan_type is None:
                for chtype in self.chan_rank:
                    fname1  = fpattern+'.%sH1.SAC' %chtype
                    fname2  = fpattern+'.%sH2.SAC' %chtype
                    fnamez  = fpattern+'.%sHZ.SAC' %chtype
                    fnamep  = fpattern+'.%sDH.SAC' %chtype
                    if os.path.isfile(fname1) and os.path.isfile(fname2) and \
                        os.path.isfile(fnamez) and os.path.isfile(fnamep):
                        chan_type   = chtype
                        break
            if chan_type is None:
                stime   += 86400.
                continue
            fname1  = fpattern+'.%sH1.SAC' %chan_type
            fname2  = fpattern+'.%sH2.SAC' %chan_type
            fnamez  = fpattern+'.%sHZ.SAC' %chan_type
            fnamep  = fpattern+'.%sDH.SAC' %chan_type
            if not (os.path.isfile(fname1) and os.path.isfile(fname2) and \
                    os.path.isfile(fnamez) and os.path.isfile(fnamep)):
                stime   += 86400.
                continue
            tr1     = obspy.read(fname1)[0]
            stimetr = tr1.stats.starttime
            tr2     = obspy.read(fname2)[0]
            trZ     = obspy.read(fnamez)[0]
            trP     = obspy.read(fnamep)[0]
            
            if (np.all(tr1.data == 0.) or np.all(tr2.data == 0.)) and np.all(trP.data == 0.):
                stime   += 86400.
                continue
            
            if abs(tr1.stats.delta - targetdt) > 1e-3 or abs(tr2.stats.delta - targetdt) > 1e-3 or \
                abs(trZ.stats.delta - targetdt) > 1e-3 or abs(trP.stats.delta - targetdt) > 1e-3:
                raise ValueError('!!! CHECK fs :'+ self.staid)
            else:
                tr1.stats.delta     = targetdt
                tr2.stats.delta     = targetdt
                trP.stats.delta     = targetdt
                trZ.stats.delta     = targetdt
            
            outfnameZ   = odaydir+'/ft_%d.%s.%d.%s.%sHZ.SAC' %(self.year, monthdict[self.month], stime.day, self.staid, chan_type)
            # sliding window raw data
            StreamZ = obspy.Stream()
            overlap = 0.99
            for tmptr in trZ.slide(window_length = self.window-1, step = int((self.window-1)*overlap)):
                # print (tmptr.stats.npts)
                StreamZ += tmptr.copy()
            Stream1 = obspy.Stream()
            for tmptr in tr1.slide(window_length = self.window-1, step = int((self.window-1)*overlap)):
                Stream1 += tmptr.copy()
            Stream2 = obspy.Stream()
            for tmptr in tr2.slide(window_length = self.window-1, step = int((self.window-1)*overlap)):
                Stream2 += tmptr.copy()
            StreamP = obspy.Stream()
            for tmptr in trP.slide(window_length = self.window-1, step = int((self.window-1)*overlap)):
                StreamP += tmptr.copy()
            # remove tilt and compliance
            outStreamZ  = obspy.Stream()
            Ntraces     = len(StreamZ)
            for itr in range(Ntraces):
                sth     = obspy.Stream()
                if not (np.all(Stream1[itr].data == 0.) or np.all(Stream1[itr].data == 0.)):
                    sth     += Stream1[itr]
                    sth     += Stream2[itr]
                sth     += StreamZ[itr]
                stp     = obspy.Stream()
                if not np.all(StreamP[itr].data == 0.):
                    stp     += StreamP[itr]
                tmptime = StreamZ[itr].stats.starttime
                tstamp  = str(tmptime.year).zfill(4)+'.' + \
                    str(tmptime.julday).zfill(3)+'.'
                tstamp  = tstamp + str(tmptime.hour).zfill(2) + \
                            '.'+str(tmptime.minute).zfill(2)
                if self.out_dtype == 'ZP':
                    ncomp   = 2
                elif self.out_dtype == 'Z2-1':
                    ncomp   = 3
                elif self.out_dtype  == 'ZP-21':
                    ncomp   = 4
                
                eventstream = EventStream( sta = self.stdb_inv, sth = sth, stp = stp,\
                    tstamp = tstamp, lat = self.stla, lon = self.stlo, time = tmptime,\
                    window = self.window,
                    sampling_rate = 1., ncomp = ncomp)
                eventstream.correct_data(self.tfnoise)
                tmptr       = StreamZ[itr].copy()
                tmptr.data  = eventstream.correct[self.out_dtype].copy()
                outStreamZ  += tmptr
                # outStreamZ.data = 
            # merge data
            outStreamZ.merge(method = 1, fill_value = 'interpolate', interpolation_samples=2)
            outTrZ  = outStreamZ[0]
            if os.path.isfile(outfnameZ):
                shutil.copyfile(src = outfnameZ, dst = outfnameZ+'_old')
                os.remove(outfnameZ)
            outTrZ.write(outfnameZ, format = 'SAC')
            stime   += 86400.
            
            
def atacr_for_mp(in_atacr_sta, verbose = False):
    if verbose:
        print('=== station :'+self.staid)
    if not in_atacr_sta.transfer_func():
        return
    in_atacr_sta.correct()
    staid   = in_atacr_sta.staid
    logfname= self.outdir+'/logs_atacr/'+self.monthdir+'/'+staid+'.log'
    with open(logfname, 'w') as fid:
        fid.writelines('SUCCESS\n')
    return
    