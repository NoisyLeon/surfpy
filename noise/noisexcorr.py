# -*- coding: utf-8 -*-
"""
ASDF for cross-correlation
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
try:
    import surfpy.noise.noisebase as noisebase
except:
    import noisebase

try:
    import surfpy.noise._xcorr_funcs as _xcorr_funcs
except:
    import _xcorr_funcs 

import numpy as np
from functools import partial
import multiprocessing
import obspy
import obspy.io.sac
import obspy.io.xseed
from datetime import datetime
import warnings
import tarfile
import shutil
import glob
import sys
import copy
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'


monthdict               = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
xcorr_header_default    = {'netcode1': '', 'stacode1': '', 'netcode2': '', 'stacode2': '', 'chan1': '', 'chan2': '',
        'npts': 12345, 'b': 12345, 'e': 12345, 'delta': 12345, 'dist': 12345, 'az': 12345, 'baz': 12345, 'stackday': 0}
xcorr_sacheader_default = {'knetwk': '', 'kstnm': '', 'kcmpnm': '', 'stla': 12345, 'stlo': 12345, 
            'kuser0': '', 'kevnm': '', 'evla': 12345, 'evlo': 12345, 'evdp': 0., 'dist': 0., 'az': 12345, 'baz': 12345, 
                'delta': 12345, 'npts': 12345, 'user0': 0, 'b': 12345, 'e': 12345}
# ------------- xcorr specific exceptions ---------------------------------------
class xcorrError(Exception):
    pass

class xcorrIOError(xcorrError, IOError):
    pass

class xcorrHeaderError(xcorrError):
    """
    Raised if header has issues.
    """
    pass

class xcorrDataError(xcorrError):
    """
    Raised if header has issues.
    """
    pass

class xcorrASDF(noisebase.baseASDF):
    """ Class for xcorr process
    =================================================================================================================
    version history:
        2020/07/09
    =================================================================================================================
    """
    def tar_mseed_to_sac(self, datadir, outdir, start_date, end_date, unit_nm=True, sps=1., outtype=0, rmresp=True, hvflag=False,
            chtype='LH', channels='ENZ', ntaper=2, halfw=100, tb = 1., tlen = 86398., tb2 = 1000., tlen2 = 84000.,
            perl = 5., perh = 200., pfx='LF_', delete_tar=False, delete_extract=True, verbose=False, verbose2 = False):
        """extract tared mseed files to SAC
        """
        if channels != 'EN' and channels != 'ENZ' and channels != 'Z':
            raise xcorrError('Unexpected channels = '+channels)
        starttime   = obspy.core.utcdatetime.UTCDateTime(start_date)
        endtime     = obspy.core.utcdatetime.UTCDateTime(end_date)
        curtime     = starttime
        Nnodataday  = 0
        Nday        = 0
        # frequencies for response removal 
        f2          = 1./(perh*1.3)
        f1          = f2*0.8
        f3          = 1./(perl*0.8)
        f4          = f3*1.2
        targetdt    = 1./sps
        if ((np.ceil(tb/targetdt)*targetdt - tb) > (targetdt/10.)) or ((np.ceil(tlen/targetdt) -tlen) > (targetdt/10.)) or\
            ((np.ceil(tb2/targetdt)*targetdt - tb2) > (targetdt/10.)) or ((np.ceil(tlen2/targetdt) -tlen2) > (targetdt/10.)):
            print ('WARNING: tb = %g tlen = %g tb2 = %g tlen2 = %g' %(tb, tlen, tb2, tlen2))
            # # # raise xcorrError('tb and tlen must both be multiplilier of target dt!')
        
        print ('[%s] [TARMSEED2SAC] Extracting tar mseed from: ' %datetime.now().isoformat().split('.')[0]+datadir+' to '+outdir)
        while (curtime <= endtime):
            print ('[%s] [TARMSEED2SAC] Date: ' %datetime.now().isoformat().split('.')[0]+curtime.date.isoformat())
            Nday        +=1
            Ndata       = 0
            Nnodata     = 0
            tarwildcard = datadir+'/'+pfx+str(curtime.year)+'.'+monthdict[curtime.month]+'.'+str(curtime.day)+'.*.tar.mseed'
            tarlst      = glob.glob(tarwildcard)
            if len(tarlst) == 0:
                print ('!!! NO DATA DATE: '+curtime.date.isoformat())
                curtime     += 86400
                Nnodataday  += 1
                continue
            elif len(tarlst) > 1:
                print ('!!! MORE DATA DATE: '+curtime.date.isoformat())
            # time stamps for user specified tb and te (tb + tlen)
            tbtime  = curtime + tb
            tetime  = tbtime + tlen
            tbtime2 = curtime + tb2
            tetime2 = tbtime2 + tlen2
            if tbtime2 < tbtime or tetime2 > tetime:
                raise xcorrError('removed resp should be in the range of raw data ')
            # extract tar files
            tmptar  = tarfile.open(tarlst[0])
            tmptar.extractall(path = outdir)
            tmptar.close()
            datedir     = outdir+'/'+(tarlst[0].split('/')[-1])[:-10]
            outdatedir  = outdir+'/'+str(curtime.year)+'.'+ monthdict[curtime.month] + '/' \
                        +str(curtime.year)+'.'+monthdict[curtime.month]+'.'+str(curtime.day)
            # loop over stations
            for staid in self.waveforms.list():
                netcode     = staid.split('.')[0]
                stacode     = staid.split('.')[1]
                skip_this_station   = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    staxml  = self.waveforms[staid].StationXML
                mseedfname      = datedir + '/' + stacode+'.'+netcode+'.mseed'
                xmlfname        = datedir + '/IRISDMC-' + stacode+'.'+netcode+'.xml'
                datalessfname   = datedir + '/IRISDMC-' + stacode+'.'+netcode+'.dataless'
                # load data
                if not os.path.isfile(mseedfname):
                    if curtime >= staxml[0][0].creation_date and curtime <= staxml[0][0].end_date:
                        if verbose:
                            print ('*** NO DATA STATION: '+staid)
                        Nnodata     += 1
                    continue
                #out SAC file names
                fnameZ      = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'Z.SAC'
                fnameE      = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'E.SAC'
                fnameN      = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'N.SAC'
                if outtype != 0 and channels=='Z':
                    fnameZ      = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+stacode+'.'+chtype+'Z.SAC'
                # load data
                st      = obspy.read(mseedfname)
                st.sort(keys=['location', 'channel', 'starttime', 'endtime']) # sort the stream
                #=============================
                # get response information
                # rmresp = True, from XML
                # rmresp = False, from dataless
                #=============================
                if rmresp:
                    if not os.path.isfile(xmlfname):
                        print ('*** NO RESPXML FILE STATION: '+staid)
                        resp_inv = staxml.copy()
                        try:
                            for tr in st:
                                seed_id     = tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.location+'.'+tr.stats.channel
                                resp_inv.get_response(seed_id = seed_id, datetime = curtime)
                        except:
                            print ('*** NO RESP STATION: '+staid)
                            Nnodata     += 1
                            continue
                    else:
                        try:
                            resp_inv = obspy.read_inventory(xmlfname)
                        except:
                            Nnodata     += 1
                            continue
                else:
                    if not os.path.isfile(datalessfname):
                        print ('*** NO DATALESS FILE STATION: '+staid)
                #===========================================
                # resample the data and perform time shift 
                #===========================================
                ipoplst = []
                for i in range(len(st)):
                    # time shift
                    if (abs(st[i].stats.delta - targetdt)/targetdt) < (1e-4) :
                        st[i].stats.delta   = targetdt
                        dt                  = st[i].stats.delta
                        tmpstime            = st[i].stats.starttime
                        st[i].data          = st[i].data.astype(np.float64) # convert int in gains to float64
                        tdiff               = tmpstime - curtime
                        Nt                  = np.floor(tdiff/dt)
                        tshift              = tdiff - Nt*dt
                        if tshift < 0.:
                            raise xcorrError('UNEXPECTED tshift = '+str(tshift)+' STATION:'+staid)
                        # apply the time shift
                        if tshift < dt*0.5:
                            st[i].data              = _xcorr_funcs._tshift_fft(st[i].data, dt=dt, tshift = tshift) 
                            st[i].stats.starttime   -= tshift
                        else:
                            st[i].data              = _xcorr_funcs._tshift_fft(st[i].data, dt=dt, tshift = tshift-dt ) 
                            st[i].stats.starttime   += dt - tshift
                        if tdiff < 0.:
                            print ('!!! STARTTIME IN PREVIOUS DAY STATION: '+staid)
                            st[i].trim(starttime=curtime)
                    # resample and time "shift"
                    else:
                        # print ('!!! RESAMPLING DATA STATION: '+staid)
                        # detrend the data to prevent edge effect when perform prefiltering before decimate
                        st[i].detrend()
                        dt          = st[i].stats.delta
                        # change dt
                        factor      = np.round(targetdt/dt)
                        if abs(factor*dt - targetdt) < min(dt, targetdt/50.):
                            dt                  = targetdt/factor
                            st[i].stats.delta   = dt
                        else:
                            print('Unexpected dt: ', targetdt, dt)
                            skip_this_station   = True
                            # raise ValueError('CHECK!' + staid)
                            break
                        # "shift" the data by changing the start timestamp
                        tmpstime    = st[i].stats.starttime
                        tdiff       = tmpstime - curtime
                        Nt          = np.floor(tdiff/dt)
                        tshift_s    = tdiff - Nt*dt
                        if tshift_s < dt*0.5:
                            st[i].stats.starttime   -= tshift_s
                        else:
                            st[i].stats.starttime   += dt - tshift_s
                        # new start time for trim
                        tmpstime    = st[i].stats.starttime
                        tdiff       = tmpstime - curtime
                        Nt          = np.floor(tdiff/targetdt)
                        tshift_s    = tdiff - Nt*targetdt
                        newstime    = tmpstime + (targetdt - tshift_s)
                        # new end time for trim
                        tmpetime    = st[i].stats.endtime
                        tdiff       = tmpetime - curtime
                        Nt          = np.floor(tdiff/targetdt)
                        tshift_e    = tdiff - Nt*targetdt
                        newetime    = tmpetime - tshift_e
                        if newetime < newstime:
                            if tmpetime - tmpstime > targetdt:
                                print (st[i].stats.starttime)
                                print (newstime)
                                print (st[i].stats.endtime)
                                print (newetime)
                                raise ValueError('CHECK!')
                            else:
                                ipoplst.append(i)
                                continue
                        # trim the data
                        st[i].trim(starttime = newstime, endtime = newetime)
                        # decimate
                        try:
                            st[i].filter(type = 'lowpass', freq = sps/2., zerophase = True) # prefilter
                            st[i].decimate(factor = int(factor), no_filter = True)
                        except:
                            skip_this_station = True
                            break
                        # check the time stamp again, for debug purposes
                        if st[i].stats.starttime != newstime or st[i].stats.endtime != newetime:
                            print (st[i].stats.starttime)
                            print (newstime)
                            print (st[i].stats.endtime)
                            print (newetime)
                            raise ValueError('CHECK start/end time' + staid)
                        if (int((newstime - curtime)/targetdt) * targetdt != (newstime - curtime))\
                            or (int((newetime - curtime)/targetdt) * targetdt != (newetime - curtime)):
                            print (newstime)
                            print (newetime)
                            raise ValueError('CHECK start/end time' + staid)
                if skip_this_station:
                    continue
                if len(ipoplst) > 0:
                    print ('!!! poping traces!'+staid)
                    npop        = 0
                    for ipop in ipoplst:
                        st.pop(index = ipop - npop)
                        npop    += 1
                #====================================================
                # merge the data: taper merge overlaps or fill gaps
                #====================================================
                if hvflag:
                    raise xcorrError('hvflag = True not yet supported!')
                st2     = obspy.Stream()
                isZ     = False
                isEN    = False
                locZ    = None
                locEN   = None
                # Z component
                if channels[-1] == 'Z':
                    StreamZ    = st.select(channel=chtype+'Z')
                    StreamZ.sort(keys=['starttime', 'endtime'])
                    StreamZ.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                    if len(StreamZ) == 0:
                        if verbose:
                            print ('!!! NO Z COMPONENT STATION: '+staid)
                        Nrec            = 0
                        Nrec2           = 0
                    else:
                        trZ             = StreamZ[0].copy()
                        gapT            = max(0, trZ.stats.starttime - tbtime) + max(0, tetime - trZ.stats.endtime)
                        # more than two traces with different locations, choose the longer one
                        if len(StreamZ) > 1:
                            for tmptr in StreamZ:
                                tmpgapT = max(0, tmptr.stats.starttime - tbtime) + max(0, tetime - tmptr.stats.endtime)
                                if tmpgapT < gapT:
                                    gapT= tmpgapT
                                    trZ = tmptr.copy()
                            if verbose2:
                                print ('!!! MORE Z LOCS STATION: '+staid+', CHOOSE: '+trZ.stats.location)
                            locZ    = trZ.stats.location
                        if trZ.stats.starttime > tetime or trZ.stats.endtime < tbtime:
                            if verbose:
                                print ('!!! NO Z COMPONENT STATION: '+staid)
                            Nrec        = 0
                            Nrec2       = 0
                        else:
                            # trim the data for tb and tb + tlen
                            trZ.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                            if isinstance(trZ.data, np.ma.masked_array):
                                maskZ   = trZ.data.mask
                                dataZ   = trZ.data.data
                                sigstd  = trZ.data.std()
                                sigmean = trZ.data.mean()
                                if np.isnan(sigstd) or np.isnan(sigmean):
                                    raise xcorrDataError('NaN Z SIG/MEAN STATION: '+staid)
                                dataZ[maskZ]    = 0.
                                # gap list
                                gaparr, Ngap    = _xcorr_funcs._gap_lst(maskZ)
                                gaplst          = gaparr[:Ngap, :]
                                # get the rec list
                                Nrecarr, Nrec   = _xcorr_funcs._rec_lst(maskZ)
                                Nreclst         = Nrecarr[:Nrec, :]
                                if np.any(Nreclst<0) or np.any(gaplst<0):
                                    raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                # values for gap filling
                                fillvals        = _xcorr_funcs._fill_gap_vals(gaplst, Nreclst, dataZ, Ngap, halfw)
                                trZ.data        = fillvals * maskZ + dataZ
                                if np.any(np.isnan(trZ.data)):
                                    raise xcorrDataError('NaN Z DATA STATION: '+staid)
                                # rec lst for tb2 and tlen2
                                im0             = int((tb2 - tb)/targetdt)
                                im1             = int((tb2 + tlen2 - tb)/targetdt) + 1
                                maskZ2          = maskZ[im0:im1]
                                Nrecarr2, Nrec2 = _xcorr_funcs._rec_lst(maskZ2)
                                Nreclst2        = Nrecarr2[:Nrec2, :]
                            else:
                                Nrec    = 0
                                Nrec2   = 0
                            st2.append(trZ)
                            isZ     = True
                    if Nrec > 0:
                        if not os.path.isdir(outdatedir):
                            os.makedirs(outdatedir)
                        with open(fnameZ+'_rec', 'w') as fid:
                            for i in range(Nrec):
                                fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                    if Nrec2 > 0:
                        if not os.path.isdir(outdatedir):
                            os.makedirs(outdatedir)
                        if verbose2:
                            print ('!!! GAP Z  STATION: '+staid)
                        with open(fnameZ+'_rec2', 'w') as fid:
                            for i in range(Nrec2):
                                fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                # EN component
                if len(channels)>= 2:
                    if channels[:2] == 'EN':
                        StreamE    = st.select(channel=chtype+'E')
                        StreamE.sort(keys=['starttime', 'endtime'])
                        StreamE.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                        StreamN    = st.select(channel=chtype+'N')
                        StreamN.sort(keys=['starttime', 'endtime'])
                        StreamN.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                        Nrec        = 0
                        Nrec2       = 0
                        if len(StreamE) == 0 or (len(StreamN) != len(StreamE)):
                            if verbose2:
                                print ('!!! NO E or N COMPONENT STATION: '+staid)
                            Nrec    = 0
                            Nrec2   = 0
                        else:
                            trE             = StreamE[0].copy()
                            trN             = StreamN[0].copy()
                            gapT            = max(0, trE.stats.starttime - tbtime) + max(0, tetime - trE.stats.endtime)
                            # more than two traces with different locations, choose the longer one
                            if len(StreamE) > 1:
                                for tmptr in StreamE:
                                    tmpgapT = max(0, tmptr.stats.starttime - tbtime) + max(0, tetime - tmptr.stats.endtime)
                                    if tmpgapT < gapT:
                                        gapT= tmpgapT
                                        trE = tmptr.copy()
                                if verbose2:
                                    print ('!!! MORE E LOCS STATION: '+staid+', CHOOSE: '+trE.stats.location)
                                locEN   = trE.stats.location
                                trN     = StreamN.select(location=locEN)[0]
                            if trE.stats.starttime > tetime or trE.stats.endtime < tbtime or\
                                    trN.stats.starttime > tetime or trN.stats.endtime < tbtime:
                                if verbose:
                                    print ('!!! NO E or N COMPONENT STATION: '+staid)
                                Nrec        = 0
                                Nrec2       = 0
                            else:
                                # trim the data for tb and tb+tlen
                                trE.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                                trN.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                                ismask      = False
                                if isinstance(trE.data, np.ma.masked_array):
                                    mask    = trE.data.mask.copy()
                                    dataE   = trE.data.data.copy()
                                    ismask  = True
                                else:
                                    dataE   = trE.data.copy()
                                if isinstance(trN.data, np.ma.masked_array):
                                    if ismask:
                                        mask    += trN.data.mask.copy()
                                    else:
                                        mask    = trN.data.mask.copy()
                                        ismask  = True
                                    dataN   = trN.data.data.copy()
                                else:
                                    dataN   = trN.data.copy()
                                allmasked   = False
                                if ismask:
                                    allmasked   = np.all(mask)
                                if ismask and (not allmasked) :
                                    sigstdE     = trE.data.std()
                                    sigmeanE    = trE.data.mean()
                                    sigstdN     = trN.data.std()
                                    sigmeanN    = trN.data.mean()
                                    if np.isnan(sigstdE) or np.isnan(sigmeanE) or \
                                        np.isnan(sigstdN) or np.isnan(sigmeanN):
                                        raise xcorrDataError('NaN EN SIG/MEAN STATION: '+staid)
                                    dataE[mask] = 0.
                                    dataN[mask] = 0.
                                    # gap list
                                    gaparr, Ngap    = _xcorr_funcs._gap_lst(mask)
                                    gaplst          = gaparr[:Ngap, :]
                                    # get the rec list
                                    Nrecarr, Nrec   = _xcorr_funcs._rec_lst(mask)
                                    Nreclst         = Nrecarr[:Nrec, :]
                                    if np.any(Nreclst<0) or np.any(gaplst<0):
                                        raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                    # values for gap filling
                                    fillvalsE   = _xcorr_funcs._fill_gap_vals(gaplst, Nreclst, dataE, Ngap, halfw)
                                    fillvalsN   = _xcorr_funcs._fill_gap_vals(gaplst, Nreclst, dataN, Ngap, halfw)
                                    trE.data    = fillvalsE * mask + dataE
                                    trN.data    = fillvalsN * mask + dataN
                                    if np.any(np.isnan(trE.data)) or np.any(np.isnan(trN.data)):
                                        raise xcorrDataError('NaN EN DATA STATION: '+staid)
                                    if np.any(Nreclst<0):
                                        raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                    # rec lst for tb2 and tlen2
                                    im0             = int((tb2 - tb)/targetdt)
                                    im1             = int((tb2 + tlen2 - tb)/targetdt) + 1
                                    mask            = mask[im0:im1]
                                    Nrecarr2, Nrec2 = _xcorr_funcs._rec_lst(mask)
                                    Nreclst2        = Nrecarr2[:Nrec2, :]
                                else:
                                    Nrec    = 0
                                    Nrec2   = 0
                                if not allmasked:
                                    st2.append(trE)
                                    st2.append(trN)
                                    isEN     = True
                            if Nrec > 0:
                                if not os.path.isdir(outdatedir):
                                    os.makedirs(outdatedir)
                                with open(fnameE+'_rec', 'w') as fid:
                                    for i in range(Nrec):
                                        fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                                with open(fnameN+'_rec', 'w') as fid:
                                    for i in range(Nrec):
                                        fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                            if Nrec2 > 0:
                                if not os.path.isdir(outdatedir):
                                    os.makedirs(outdatedir)
                                if verbose2:
                                    print ('!!! GAP EN STATION: '+staid)
                                with open(fnameE+'_rec2', 'w') as fid:
                                    for i in range(Nrec2):
                                        fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                                with open(fnameN+'_rec2', 'w') as fid:
                                    for i in range(Nrec2):
                                        fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                if (not isZ) and (not isEN):
                    continue
                if not os.path.isdir(outdatedir):
                    os.makedirs(outdatedir)
                # remove trend, response
                if rmresp:
                    if tbtime2 < tbtime or tetime2 > tetime:
                        raise xcorrError('removed resp should be in the range of raw data ')
                    st2.detrend()
                    try:
                        st2.remove_response(inventory = resp_inv, pre_filt = [f1, f2, f3, f4])
                    except:
                        continue
                    if unit_nm: # convert unit from m/sec to nm/sec
                        for i in range(len(st2)):
                            st2[i].data *= 1e9
                    st2.trim(starttime = tbtime2, endtime = tetime2, pad = True, fill_value=0)
                else:
                    fnameZ          = outdatedir+'/'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'Z.SAC'
                    fnameE          = outdatedir+'/'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'E.SAC'
                    fnameN          = outdatedir+'/'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+chtype+'N.SAC'
                    if outtype != 0 and channels=='Z':
                        fnameZ      = outdatedir+'/'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+stacode+'.'+chtype+'Z.SAC'
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sp              = obspy.io.xseed.Parser(datalessfname)
                    sp.write_resp(folder = outdatedir)
                    if locZ is not None:
                        respzlst    = glob.glob(outdatedir+'/RESP.'+staid+'*'+chtype+'Z')
                        keepfname   = outdatedir+'/RESP.'+staid+'.'+locZ+'.'+chtype+'Z'
                        for respfname in respzlst:
                            if keepfname != respfname:
                                os.remove(respfname)
                    if locEN is not None:
                        respelst    = glob.glob(outdatedir+'/RESP.'+staid+'*'+chtype+'E')
                        keepfname   = outdatedir+'/RESP.'+staid+'.'+locEN+'.'+chtype+'E'
                        for respfname in respelst:
                            if keepfname != respfname:
                                os.remove(respfname)
                        respnlst    = glob.glob(outdatedir+'/RESP.'+staid+'*'+chtype+'N')
                        keepfname   = outdatedir+'/RESP.'+staid+'.'+locEN+'.'+chtype+'N'
                        for respfname in respnlst:
                            if keepfname != respfname:
                                os.remove(respfname)
                # save data to SAC
                if isZ:
                    sactrZ  = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=chtype+'Z')[0])
                    sactrZ.write(fnameZ)
                if isEN:
                    sactrE  = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=chtype+'E')[0])
                    sactrE.write(fnameE)
                    sactrN  = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=chtype+'N')[0])
                    sactrN.write(fnameN)
                Ndata   += 1
            # End loop over stations
            curtime     += 86400
            if verbose:
                print ('[%s] [TARMSEED2SAC] %d/%d (data/no_data) groups of traces extracted!'\
                       %(datetime.now().isoformat().split('.')[0], Ndata, Nnodata))
            # delete raw data
            if delete_extract:
                shutil.rmtree(datedir)
            if delete_tar:
                os.remove(tarlst[0])
        # End loop over dates
        print ('[%s] [TARMSEED2SAC] Extracted %d/%d (days_with)data/total_days) days of data'\
               %(datetime.now().isoformat().split('.')[0], Nday - Nnodataday, Nday))
        return
    
    def mseed_to_sac(self, datadir, outdir, start_date, end_date, staxmldir = None, unit_nm = True, sps=1., \
            hvflag=False, chan_rank=['LH', 'BH', 'HH'], channels='ENZ', ntaper=2, halfw=100,\
            tb = 1., tlen = 86398., tb2 = 1000., tlen2 = 84000., perl = 5., perh = 200., delete_mseed=False, verbose=True, verbose2 = False):
        """extract mseed files to SAC
        """
        if channels != 'EN' and channels != 'ENZ' and channels != 'Z':
            raise xcorrError('Unexpected channels = '+channels)
        starttime   = obspy.core.utcdatetime.UTCDateTime(start_date)
        endtime     = obspy.core.utcdatetime.UTCDateTime(end_date)
        curtime     = starttime
        Nnodataday  = 0
        Nday        = 0
        # frequencies for response removal 
        f2          = 1./(perh*1.3)
        f1          = f2*0.8
        f3          = 1./(perl*0.8)
        f4          = f3*1.2
        targetdt    = 1./sps
        if ((np.ceil(tb/targetdt)*targetdt - tb) > (targetdt/100.)) or ((np.ceil(tlen/targetdt) -tlen) > (targetdt/100.)) or\
            ((np.ceil(tb2/targetdt)*targetdt - tb2) > (targetdt/100.)) or ((np.ceil(tlen2/targetdt) -tlen2) > (targetdt/100.)):
            raise xcorrError('tb and tlen must both be multiplilier of target dt!')
        
        print ('[%s] [MSEED2SAC] Extracting mseed from: ' %datetime.now().isoformat().split('.')[0]+datadir+' to '+outdir)
        while (curtime <= endtime):
            if verbose:
                print ('[%s] [MSEED2SAC] Date: ' %datetime.now().isoformat().split('.')[0]+curtime.date.isoformat())
            Nday        +=1
            Ndata       = 0
            Nnodata     = 0
            # time stamps for user specified tb and te (tb + tlen)
            tbtime      = curtime + tb
            tetime      = tbtime + tlen
            tbtime2     = curtime + tb2
            tetime2     = tbtime2 + tlen2
            if tbtime2 < tbtime or tetime2 > tetime:
                raise xcorrError('removed resp should be in the range of raw data ')
            # time label
            day0        = '%d%02d%02d' %(curtime.year, curtime.month, curtime.day)
            tmptime     = curtime + 86400
            day1        = '%d%02d%02d' %(tmptime.year, tmptime.month, tmptime.day)
            time_label  = '%sT000000Z.%sT000000Z' %(day0, day1)
            # loop over stations
            for staid in self.waveforms.list():
                netcode     = staid.split('.')[0]
                stacode     = staid.split('.')[1]
                skip_this_station   = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    staxml  = self.waveforms[staid].StationXML
                # determine type of channels
                channel_type= None
                for tmpchtype in chan_rank:
                    ich     = 0
                    for chan in channels:
                        mseedpattern    = datadir + '/%s/%s/%s%s*%s.mseed' %(netcode, stacode, tmpchtype, chan, time_label)
                        if len(glob.glob(mseedpattern)) == 0:
                            break
                        ich += 1
                    if ich == len(channels):
                        channel_type= tmpchtype
                        location    = glob.glob(mseedpattern)[0].split('%s/%s/%s%s' \
                                        %(netcode, stacode, tmpchtype, chan))[1].split('.')[1]
                        break
                if channel_type is None:
                    if curtime >= staxml[0][0].creation_date and curtime <= staxml[0][0].end_date:
                        print ('*** NO DATA STATION: '+staid)
                        Nnodata     += 1
                    continue
                #out SAC file names
                outdatedir  = outdir+'/'+str(curtime.year)+'.'+ monthdict[curtime.month] + '/' \
                        +str(curtime.year)+'.'+monthdict[curtime.month]+'.'+str(curtime.day)
                fnameZ  = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+channel_type+'Z.SAC'
                fnameE  = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+channel_type+'E.SAC'
                fnameN  = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+'.'+staid+'.'+channel_type+'N.SAC'
                # load data
                st      = obspy.Stream()
                for chan in channels:
                    mseedfname  = datadir + '/%s/%s/%s%s.%s.%s.mseed' %(netcode, stacode, channel_type, chan, location, time_label)
                    st          +=obspy.read(mseedfname)
                    if delete_mseed:
                        os.remove(mseedfname)
                #=============================
                # get response information
                #=============================
                if staxmldir is not None:
                    xmlfname    = staxmldir + '/%s/%s.xml' %(netcode, stacode)
                if not os.path.isfile(xmlfname) or staxmldir is None:
                    print ('*** NO RESPXML FILE STATION: '+staid)
                    resp_inv    = staxml.copy()
                    try:
                        for tr in st:
                            seed_id     = tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.location+'.'+tr.stats.channel
                            resp_inv.get_response(seed_id = seed_id, datetime = curtime)
                    except:
                        print ('*** NO RESP STATION: '+staid)
                        Nnodata     += 1
                        continue
                else:
                    try:
                        resp_inv = obspy.read_inventory(xmlfname)
                    except:
                        print ('*** NO RESPXML FILE STATION: '+staid)
                        resp_inv    = staxml.copy()
                        try:
                            for tr in st:
                                seed_id     = tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.location+'.'+tr.stats.channel
                                resp_inv.get_response(seed_id = seed_id, datetime = curtime)
                        except:
                            print ('*** NO RESP STATION: '+staid)
                            Nnodata     += 1
                            continue
                #===========================================
                # resample the data and perform time shift 
                #===========================================
                ipoplst = []
                for i in range(len(st)):
                    # time shift
                    if (abs(st[i].stats.delta - targetdt)/targetdt) < (1e-4) :
                        st[i].stats.delta   = targetdt
                        dt                  = st[i].stats.delta
                        tmpstime            = st[i].stats.starttime
                        st[i].data          = st[i].data.astype(np.float64) # convert int in gains to float64
                        tdiff               = tmpstime - curtime
                        Nt                  = np.floor(tdiff/dt)
                        tshift              = tdiff - Nt*dt
                        if tshift < 0.:
                            raise xcorrError('UNEXPECTED tshift = '+str(tshift)+' STATION:'+staid)
                        # apply the time shift
                        if tshift < dt*0.5:
                            st[i].data              = _xcorr_funcs._tshift_fft(st[i].data, dt=dt, tshift = tshift) 
                            st[i].stats.starttime   -= tshift
                        else:
                            st[i].data              = _xcorr_funcs._tshift_fft(st[i].data, dt=dt, tshift = tshift-dt ) 
                            st[i].stats.starttime   += dt - tshift
                        if tdiff < 0.:
                            print ('!!! STARTTIME IN PREVIOUS DAY STATION: '+staid)
                            st[i].trim(starttime=curtime)
                    # resample and time "shift"
                    else:
                        # detrend the data to prevent edge effect when perform prefiltering before decimate
                        st[i].detrend()
                        dt          = st[i].stats.delta
                        # change dt
                        factor      = np.round(targetdt/dt)
                        if abs(factor*dt - targetdt) < min(dt, targetdt/50.):
                            dt                  = targetdt/factor
                            st[i].stats.delta   = dt
                        else:
                            print('Unexpected dt: ', targetdt, dt)
                            skip_this_station   = True
                            # raise ValueError('CHECK!' + staid)
                            break
                        # "shift" the data by changing the start timestamp
                        tmpstime    = st[i].stats.starttime
                        tdiff       = tmpstime - curtime
                        Nt          = np.floor(tdiff/dt)
                        tshift_s    = tdiff - Nt*dt
                        if tshift_s < dt*0.5:
                            st[i].stats.starttime   -= tshift_s
                        else:
                            st[i].stats.starttime   += dt - tshift_s
                        # new start time for trim
                        tmpstime    = st[i].stats.starttime
                        tdiff       = tmpstime - curtime
                        Nt          = np.floor(tdiff/targetdt)
                        tshift_s    = tdiff - Nt*targetdt
                        newstime    = tmpstime + (targetdt - tshift_s)
                        # new end time for trim
                        tmpetime    = st[i].stats.endtime
                        tdiff       = tmpetime - curtime
                        Nt          = np.floor(tdiff/targetdt)
                        tshift_e    = tdiff - Nt*targetdt
                        newetime    = tmpetime - tshift_e
                        if newetime < newstime:
                            if tmpetime - tmpstime > targetdt:
                                print (st[i].stats.starttime)
                                print (newstime)
                                print (st[i].stats.endtime)
                                print (newetime)
                                raise ValueError('CHECK!')
                            else:
                                ipoplst.append(i)
                                continue
                        # trim the data
                        st[i].trim(starttime = newstime, endtime = newetime)
                        # decimate
                        try:
                            st[i].filter(type = 'lowpass', freq = sps/2., zerophase = True) # prefilter
                            st[i].decimate(factor = int(factor), no_filter = True)
                        except:
                            skip_this_station = True
                            break
                        # check the time stamp again, for debug purposes
                        if st[i].stats.starttime != newstime or st[i].stats.endtime != newetime:
                            print (st[i].stats.starttime)
                            print (newstime)
                            print (st[i].stats.endtime)
                            print (newetime)
                            raise ValueError('CHECK start/end time' + staid)
                        if (int((newstime - curtime)/targetdt) * targetdt != (newstime - curtime))\
                            or (int((newetime - curtime)/targetdt) * targetdt != (newetime - curtime)):
                            print (newstime)
                            print (newetime)
                            raise ValueError('CHECK start/end time' + staid)
                if skip_this_station:
                    continue
                if len(ipoplst) > 0:
                    print ('!!! poping traces!'+staid)
                    npop        = 0
                    for ipop in ipoplst:
                        st.pop(index = ipop - npop)
                        npop    += 1
                #====================================================
                # merge the data: taper merge overlaps or fill gaps
                #====================================================
                if hvflag:
                    raise xcorrError('hvflag = True not yet supported!')
                st2     = obspy.Stream()
                isZ     = False
                isEN    = False
                locZ    = None
                locEN   = None
                # Z component
                if channels[-1] == 'Z':
                    StreamZ    = st.select(channel=channel_type+'Z')
                    StreamZ.sort(keys=['starttime', 'endtime'])
                    StreamZ.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                    if len(StreamZ) == 0:
                        print ('!!! NO Z COMPONENT STATION: '+staid)
                        Nrec            = 0
                        Nrec2           = 0
                    else:
                        trZ             = StreamZ[0].copy()
                        gapT            = max(0, trZ.stats.starttime - tbtime) + max(0, tetime - trZ.stats.endtime)
                        # more than two traces with different locations, choose the longer one
                        if len(StreamZ) > 1:
                            for tmptr in StreamZ:
                                tmpgapT = max(0, tmptr.stats.starttime - tbtime) + max(0, tetime - tmptr.stats.endtime)
                                if tmpgapT < gapT:
                                    gapT= tmpgapT
                                    trZ = tmptr.copy()
                            if verbose2:
                                print ('!!! MORE Z LOCS STATION: '+staid+', CHOOSE: '+trZ.stats.location)
                            locZ    = trZ.stats.location
                        if trZ.stats.starttime > tetime or trZ.stats.endtime < tbtime:
                            print ('!!! NO Z COMPONENT STATION: '+staid)
                            Nrec        = 0
                            Nrec2       = 0
                        else:
                            # trim the data for tb and tb + tlen
                            trZ.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                            if isinstance(trZ.data, np.ma.masked_array):
                                maskZ   = trZ.data.mask
                                dataZ   = trZ.data.data
                                sigstd  = trZ.data.std()
                                sigmean = trZ.data.mean()
                                if np.isnan(sigstd) or np.isnan(sigmean):
                                    raise xcorrDataError('NaN Z SIG/MEAN STATION: '+staid)
                                dataZ[maskZ]    = 0.
                                # gap list
                                gaparr, Ngap    = _xcorr_funcs._gap_lst(maskZ)
                                gaplst          = gaparr[:Ngap, :]
                                # get the rec list
                                Nrecarr, Nrec   = _xcorr_funcs._rec_lst(maskZ)
                                Nreclst         = Nrecarr[:Nrec, :]
                                if np.any(Nreclst<0) or np.any(gaplst<0):
                                    raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                # values for gap filling
                                try:
                                    fillvals        = _xcorr_funcs._fill_gap_vals(gaplst, Nreclst, dataZ, Ngap, halfw)
                                except:
                                    skip_this_station   = True
                                trZ.data        = fillvals * maskZ + dataZ
                                if np.any(np.isnan(trZ.data)):
                                    raise xcorrDataError('NaN Z DATA STATION: '+staid)
                                # rec lst for tb2 and tlen2
                                im0             = int((tb2 - tb)/targetdt)
                                im1             = int((tb2 + tlen2 - tb)/targetdt) + 1
                                maskZ2          = maskZ[im0:im1]
                                Nrecarr2, Nrec2 = _xcorr_funcs._rec_lst(maskZ2)
                                Nreclst2        = Nrecarr2[:Nrec2, :]
                            else:
                                Nrec    = 0
                                Nrec2   = 0
                            st2.append(trZ)
                            isZ     = True
                    if Nrec > 0:
                        if not os.path.isdir(outdatedir):
                            os.makedirs(outdatedir)
                        with open(fnameZ+'_rec', 'w') as fid:
                            for i in range(Nrec):
                                fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                    if Nrec2 > 0:
                        if not os.path.isdir(outdatedir):
                            os.makedirs(outdatedir)
                        print ('!!! GAP Z  STATION: '+staid)
                        with open(fnameZ+'_rec2', 'w') as fid:
                            for i in range(Nrec2):
                                fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                if skip_this_station:
                    continue
                # EN component
                if len(channels)>= 2:
                    if channels[:2] == 'EN':
                        StreamE    = st.select(channel=channel_type+'E')
                        StreamE.sort(keys=['starttime', 'endtime'])
                        StreamE.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                        StreamN    = st.select(channel=channel_type+'N')
                        StreamN.sort(keys=['starttime', 'endtime'])
                        StreamN.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                        Nrec        = 0
                        Nrec2       = 0
                        if len(StreamE) == 0 or (len(StreamN) != len(StreamE)):
                            if verbose2:
                                print ('!!! NO E or N COMPONENT STATION: '+staid)
                            Nrec    = 0
                            Nrec2   = 0
                        else:
                            trE             = StreamE[0].copy()
                            trN             = StreamN[0].copy()
                            gapT            = max(0, trE.stats.starttime - tbtime) + max(0, tetime - trE.stats.endtime)
                            # more than two traces with different locations, choose the longer one
                            if len(StreamE) > 1:
                                for tmptr in StreamE:
                                    tmpgapT = max(0, tmptr.stats.starttime - tbtime) + max(0, tetime - tmptr.stats.endtime)
                                    if tmpgapT < gapT:
                                        gapT= tmpgapT
                                        trE = tmptr.copy()
                                if verbose2:
                                    print ('!!! MORE E LOCS STATION: '+staid+', CHOOSE: '+trE.stats.location)
                                locEN   = trE.stats.location
                                trN     = StreamN.select(location=locEN)[0]
                            if trE.stats.starttime > tetime or trE.stats.endtime < tbtime or\
                                    trN.stats.starttime > tetime or trN.stats.endtime < tbtime:
                                print ('!!! NO E or N COMPONENT STATION: '+staid)
                                Nrec        = 0
                                Nrec2       = 0
                            else:
                                # trim the data for tb and tb+tlen
                                trE.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                                trN.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                                ismask      = False
                                if isinstance(trE.data, np.ma.masked_array):
                                    mask    = trE.data.mask.copy()
                                    dataE   = trE.data.data.copy()
                                    ismask  = True
                                else:
                                    dataE   = trE.data.copy()
                                if isinstance(trN.data, np.ma.masked_array):
                                    if ismask:
                                        mask    += trN.data.mask.copy()
                                    else:
                                        mask    = trN.data.mask.copy()
                                        ismask  = True
                                    dataN   = trN.data.data.copy()
                                else:
                                    dataN   = trN.data.copy()
                                allmasked   = False
                                if ismask:
                                    allmasked   = np.all(mask)
                                if ismask and (not allmasked) :
                                    sigstdE     = trE.data.std()
                                    sigmeanE    = trE.data.mean()
                                    sigstdN     = trN.data.std()
                                    sigmeanN    = trN.data.mean()
                                    if np.isnan(sigstdE) or np.isnan(sigmeanE) or \
                                        np.isnan(sigstdN) or np.isnan(sigmeanN):
                                        raise xcorrDataError('NaN EN SIG/MEAN STATION: '+staid)
                                    dataE[mask] = 0.
                                    dataN[mask] = 0.
                                    # gap list
                                    gaparr, Ngap    = _xcorr_funcs._gap_lst(mask)
                                    gaplst          = gaparr[:Ngap, :]
                                    # get the rec list
                                    Nrecarr, Nrec   = _xcorr_funcs._rec_lst(mask)
                                    Nreclst         = Nrecarr[:Nrec, :]
                                    if np.any(Nreclst<0) or np.any(gaplst<0):
                                        raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                    # values for gap filling
                                    try:
                                        fillvalsE   = _xcorr_funcs._fill_gap_vals(gaplst, Nreclst, dataE, Ngap, halfw)
                                        fillvalsN   = _xcorr_funcs._fill_gap_vals(gaplst, Nreclst, dataN, Ngap, halfw)
                                    except:
                                        skip_this_station = True
                                    trE.data    = fillvalsE * mask + dataE
                                    trN.data    = fillvalsN * mask + dataN
                                    if np.any(np.isnan(trE.data)) or np.any(np.isnan(trN.data)):
                                        raise xcorrDataError('NaN EN DATA STATION: '+staid)
                                    if np.any(Nreclst<0):
                                        raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                    # rec lst for tb2 and tlen2
                                    im0             = int((tb2 - tb)/targetdt)
                                    im1             = int((tb2 + tlen2 - tb)/targetdt) + 1
                                    mask            = mask[im0:im1]
                                    Nrecarr2, Nrec2 = _xcorr_funcs._rec_lst(mask)
                                    Nreclst2        = Nrecarr2[:Nrec2, :]
                                else:
                                    Nrec    = 0
                                    Nrec2   = 0
                                if not allmasked:
                                    st2.append(trE)
                                    st2.append(trN)
                                    isEN     = True
                            if Nrec > 0:
                                if not os.path.isdir(outdatedir):
                                    os.makedirs(outdatedir)
                                with open(fnameE+'_rec', 'w') as fid:
                                    for i in range(Nrec):
                                        fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                                with open(fnameN+'_rec', 'w') as fid:
                                    for i in range(Nrec):
                                        fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                            if Nrec2 > 0:
                                if not os.path.isdir(outdatedir):
                                    os.makedirs(outdatedir)
                                print ('!!! GAP EN STATION: '+staid)
                                with open(fnameE+'_rec2', 'w') as fid:
                                    for i in range(Nrec2):
                                        fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                                with open(fnameN+'_rec2', 'w') as fid:
                                    for i in range(Nrec2):
                                        fid.writelines(str(Nreclst2[i, 0])+' '+str(Nreclst2[i, 1])+'\n')
                if (not isZ) and (not isEN):
                    continue
                if skip_this_station:
                    continue
                if not os.path.isdir(outdatedir):
                    os.makedirs(outdatedir)
                #=======================
                # remove trend, response
                #=======================
                if tbtime2 < tbtime or tetime2 > tetime:
                    raise xcorrError('removed resp should be in the range of raw data ')
                st2.detrend()
                try:
                    st2.remove_response(inventory = resp_inv, pre_filt = [f1, f2, f3, f4])
                except:
                    continue
                if unit_nm: # convert unit from m/sec to nm/sec
                    for i in range(len(st2)):
                        st2[i].data *= 1e9
                st2.trim(starttime = tbtime2, endtime = tetime2, pad = True, fill_value=0)
                # save data to SAC
                if isZ:
                    sactrZ  = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=channel_type+'Z')[0])
                    sactrZ.write(fnameZ)
                if isEN:
                    sactrE  = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=channel_type+'E')[0])
                    sactrE.write(fnameE)
                    sactrN  = obspy.io.sac.SACTrace.from_obspy_trace(st2.select(channel=channel_type+'N')[0])
                    sactrN.write(fnameN)
                Ndata   += 1
            # End loop over stations
            curtime     += 86400
            if verbose:
                print ('[%s] [MSEED2SAC] %d/%d (data/no_data) groups of traces extracted!'\
                       %(datetime.now().isoformat().split('.')[0], Ndata, Nnodata))
        # End loop over dates
        print ('[%s] [MSEED2SAC] Extracted %d/%d (days_with)data/total_days) days of data'\
               %(datetime.now().isoformat().split('.')[0], Nday - Nnodataday, Nday))
        return
    
    def remove_mseed(self, datadir, start_date, end_date, chan_rank=['LH', 'BH', 'HH'], channels='ENZ', verbose = True):
        """remove mseed files to SAC
        """
        if channels != 'EN' and channels != 'ENZ' and channels != 'Z':
            raise xcorrError('Unexpected channels = '+channels)
        starttime   = obspy.core.utcdatetime.UTCDateTime(start_date)
        endtime     = obspy.core.utcdatetime.UTCDateTime(end_date)
        curtime     = starttime
        Nnodataday  = 0
        Nday        = 0
        print ('[%s] [REMOVEMSEED] Removing mseed from: ' %datetime.now().isoformat().split('.')[0]+datadir)
        while (curtime <= endtime):
            if verbose:
                print ('[%s] [REMOVEMSEED] Date: ' %datetime.now().isoformat().split('.')[0]+curtime.date.isoformat())
            Nday        +=1
            Ndata       = 0
            Nnodata     = 0
            # time label
            day0        = '%d%02d%02d' %(curtime.year, curtime.month, curtime.day)
            tmptime     = curtime + 86400
            day1        = '%d%02d%02d' %(tmptime.year, tmptime.month, tmptime.day)
            time_label  = '%sT000000Z.%sT000000Z' %(day0, day1)
            # loop over stations
            for staid in self.waveforms.list():
                netcode     = staid.split('.')[0]
                stacode     = staid.split('.')[1]
                skip_this_station   = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    staxml  = self.waveforms[staid].StationXML
                # determine type of channels
                channel_type= None
                for tmpchtype in chan_rank:
                    ich     = 0
                    for chan in channels:
                        mseedpattern    = datadir + '/%s/%s/%s%s*%s.mseed' %(netcode, stacode, tmpchtype, chan, time_label)
                        if len(glob.glob(mseedpattern)) == 0:
                            break
                        ich += 1
                    if ich == len(channels):
                        channel_type= tmpchtype
                        location    = glob.glob(mseedpattern)[0].split('%s/%s/%s%s' \
                                        %(netcode, stacode, tmpchtype, chan))[1].split('.')[1]
                        break
                if channel_type is None:
                    if curtime >= staxml[0][0].creation_date and curtime <= staxml[0][0].end_date:
                        # print ('*** NO DATA STATION: '+staid)
                        Nnodata     += 1
                    continue
                # remove data
                for chan in channels:
                    mseedfname  = datadir + '/%s/%s/%s%s.%s.%s.mseed' %(netcode, stacode, channel_type, chan, location, time_label)
                    os.remove(mseedfname)
            # End loop over stations
            curtime     += 86400
        return

    def xcorr(self, datadir, start_date, end_date, runtype=0, skipinv=True, chans=['LHZ', 'LHE', 'LHN'], \
            chan_types=[], logsfx='ZZ', ftlen = True, tlen = 84000., mintlen = 20000., sps = 1., lagtime = 3000., CorOutflag = 0, \
                fprcs = False, fastfft=True, parallel=True, nprocess=None, subsize=1000, verbose=False, verbose2=False):
        """
        compute ambient noise cross-correlation given preprocessed amplitude and phase files
        =================================================================================================================
        ::: input parameters :::
        datadir             - directory including data and output
        startdate/enddate   - start/end date for computation
        runtype             - type of runs
                                -1  - run the xcorr after deleting all the log files
                                0   - first run, run the xcorr by creating new log files
                                1   - skip if log file indicates SUCCESS & SKIPPED & NODATA
                                2   - skip if log file indicates SUCCESS
                                3   - skip if log file exists
                                4   - skip if montly/staid1 log directory exists
                                5   - skip if monthly log directory exists
        skipinv             - skip the month if not within the start/end date of the station inventory
        chans               - channel list
        chan_types          - types (also used as priorities) of channels, used for hybrid channel xcorr
        ftlen               - turn (on/off) cross-correlation-time-length correction for amplitude
        tlen                - time length of daily records (in sec)
        mintlen             - allowed minimum time length for cross-correlation (takes effect only when ftlen = True)
        sps                 - target sampling rate
        lagtime             - cross-correlation signal half length in sec
        CorOutflag          - 0 = only output monthly xcorr data, 1 = only daily, 2 or others = output both
        fprcs               - turn on/off (1/0) precursor signal checking
        fastfft             - speeding up the computation by using precomputed fftw_plan or not
        parallel            - run the xcorr parallelly or not
        nprocess            - number of processes
        subsize             - subsize of processing list, use to prevent lock in multiprocessing process
        =================================================================================================================
        """
        #---------------------------------
        # check the channel related input
        #---------------------------------
        if len(chan_types) > 0:
            print ('[%s] [XCORR] Hybrid channel xcorr!' %datetime.now().isoformat().split('.')[0])
            if not parallel:
                raise xcorrError('Hybrid channel run is supposed to run ONLY in parallel')
            for tmpch in chans:
                for chtype in chan_types:
                    if len(chtype + tmpch) != 3:
                        raise xcorrError('Invalid Hybrid channel: '+ chtype + tmpch)
        stime   = obspy.UTCDateTime(start_date)
        etime   = obspy.UTCDateTime(end_date)
        # check log directory and initialize log indices
        if os.path.isdir(datadir+'/log_xcorr'):
            if runtype == 0:
                raise xcorrError('!!! Log directory exists, runtype should NOT be 0')
            elif runtype == -1:
                shutil.rmtree(datadir+'/log_xcorr')
                print ('!!! WARNING: log files are DELETED, all the results will be overwritten!')
        Nsuccess    = 0
        Nskipped    = 0
        Nfailed     = 0
        Nnodata     = 0
        successstr  = ''
        skipstr     = ''
        failstr     = ''
        nodatastr   = ''
        #-------------------------
        # Loop over month
        #-------------------------
        print ('[%s] [XCORR] computation START' %datetime.now().isoformat().split('.')[0])
        while(stime < etime):
            print ('[%s] [XCORR] data preparing: ' %datetime.now().isoformat().split('.')[0] +str(stime.year)+'.'+monthdict[stime.month])
            month_dir   = datadir+'/'+str(stime.year)+'.'+monthdict[stime.month]
            logmondir   = datadir+'/log_xcorr_'+logsfx+'/'+str(stime.year)+'.'+monthdict[stime.month]
            if not os.path.isdir(month_dir):
                print ('--- Xcorr dir NOT exists : '+str(stime.year)+'.'+monthdict[stime.month])
                if stime.month == 12:
                    stime       = obspy.UTCDateTime(str(stime.year + 1)+'0101')
                else:
                    stime       = obspy.UTCDateTime(str(stime.year)+'%02d01' %(stime.month+1))
                continue
            # skip upon existences of monthly log folder
            if os.path.isdir(logmondir) and runtype == 5:
                print ('!!! SKIPPED upon log dir existence : '+str(stime.year)+'.'+monthdict[stime.month])
                if stime.month == 12:
                    stime       = obspy.UTCDateTime(str(stime.year + 1)+'0101')
                else:
                    stime       = obspy.UTCDateTime(str(stime.year)+'%02d01' %(stime.month+1))
                continue
            # xcorr list
            xcorr_lst   = []
            # define the first day and last day of the current month
            c_stime     = obspy.UTCDateTime(str(stime.year)+'-'+str(stime.month)+'-1')
            try:
                c_etime = obspy.UTCDateTime(str(stime.year)+'-'+str(stime.month+1)+'-1')
            except ValueError:
                c_etime = obspy.UTCDateTime(str(stime.year+1)+'-1-1')
            #-------------------------
            # Loop over station 1
            #-------------------------
            for staid1 in self.waveforms.list():
                # determine if the range of the station 1 matches current month
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stainv1     = self.waveforms[staid1].StationXML.networks[0].stations[0]
                    st_date1    = stainv1.start_date
                    ed_date1    = stainv1.end_date
                if skipinv and (st_date1 > c_etime or ed_date1 < c_stime):
                    continue
                # create log folder for staid1
                logstadir       = logmondir+'/'+staid1
                if os.path.isdir(logstadir):
                    # skip upon existences of monthly log-sta folder
                    if runtype == 4:
                        print ('!!! SKIPPED upon log-sta dir existence : '+str(stime.year)+'.'+monthdict[stime.month]+'.'+staid1)
                        continue
                else:
                    os.makedirs(logstadir)
                netcode1, stacode1  = staid1.split('.')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos1 = self.waveforms[staid1].coordinates
                stla1       = tmppos1['latitude']
                stlo1       = tmppos1['longitude']
                stz1        = tmppos1['elevation_in_m']
                #-------------------------
                # Loop over station 2
                #-------------------------
                for staid2 in self.waveforms.list():
                    if staid1 >= staid2:
                        continue
                    # check log files
                    logfname    = logstadir+'/'+staid1+'_'+staid2+'.log'
                    if os.path.isfile(logfname):
                        if runtype == 3:
                            continue
                        with open(logfname, 'r') as fid:
                            logflag = fid.readlines()[0].split()[0]
                        if logflag == 'SUCCESS':
                            continue
                        elif logflag == 'SKIPPED':
                            if runtype == 1:
                                continue
                        elif logflag == 'NODATA':
                            if runtype == 1:
                                continue
                        elif logflag == 'FAILED':
                            pass
                        else:
                            raise xcorrError('!!! UNEXPECTED log flag = '+logflag)
                    # end checking log files
                    netcode2, stacode2  = staid2.split('.')
                    # determine if the range of the station 2 matches current month
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        stainv2     = self.waveforms[staid2].StationXML.networks[0].stations[0]
                        st_date2    = stainv2.start_date
                        ed_date2    = stainv2.end_date
                    if skipinv and (st_date2 > c_etime or ed_date2 < c_stime) :
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tmppos2     = self.waveforms[staid2].coordinates
                    stla2       = tmppos2['latitude']
                    stlo2       = tmppos2['longitude']
                    stz2        = tmppos2['elevation_in_m']
                    #-------------------------------------------------------------------------------------
                    # append the station pair to xcorr list directly if the code will be run in parallel
                    #-------------------------------------------------------------------------------------
                    if parallel:
                        xcorr_lst.append(_xcorr_funcs.xcorr_pair(stacode1 = stacode1, netcode1=netcode1, stla1=stla1, stlo1=stlo1, \
                            stacode2=stacode2, netcode2=netcode2, stla2 = stla2, stlo2=stlo2, chan_types=chan_types, \
                            monthdir=str(stime.year)+'.'+monthdict[stime.month], daylst=[], year=stime.year, month=stime.month) )
                        continue
                    #--------------------------------------------------------------
                    # otherwise, get the station pair by checking file existence
                    #--------------------------------------------------------------
                    ctime       = obspy.UTCDateTime(str(stime.year)+'-'+str(stime.month)+'-1')
                    daylst      = []
                    # Loop over days
                    while(True):
                        daydir  = month_dir+'/'+str(stime.year)+'.'+monthdict[stime.month]+'.'+str(ctime.day)
                        skipday = False
                        if os.path.isdir(daydir):
                            for chan in chans:
                                infname1    = daydir+'/ft_'+str(stime.year)+'.'+monthdict[stime.month]+'.'+str(ctime.day)+\
                                               '.'+staid1+'.'+chan+ '.SAC'
                                infname2    = daydir+'/ft_'+str(stime.year)+'.'+monthdict[stime.month]+'.'+str(ctime.day)+\
                                               '.'+staid2+'.'+chan+ '.SAC'
                                if os.path.isfile(infname1+'.am') and os.path.isfile(infname1+'.ph')\
                                        and os.path.isfile(infname2+'.am') and os.path.isfile(infname2+'.ph'):
                                    continue
                                else:
                                    skipday = True
                                    break
                            if not skipday:
                                daylst.append(ctime.day)
                        tmpmonth= ctime.month
                        ctime   += 86400.
                        if tmpmonth != ctime.month:
                            break
                    if len(daylst) != 0:
                        xcorr_lst.append(_xcorr_funcs.xcorr_pair(stacode1 = stacode1, netcode1=netcode1, stla1=stla1, stlo1=stlo1, \
                            stacode2=stacode2, netcode2=netcode2, stla2 = stla2, stlo2=stlo2, \
                                monthdir=str(stime.year)+'.'+monthdict[stime.month], daylst=daylst) )
            # End loop over station1/station2
            if len(xcorr_lst) == 0:
                print ('!!! NO DATA: '+str(stime.year)+'.'+monthdict[stime.month])
                if stime.month == 12:
                    stime       = obspy.UTCDateTime(str(stime.year + 1)+'0101')
                else:
                    stime       = obspy.UTCDateTime(str(stime.year)+'%02d01' %(stime.month+1))
                # delete empty log-sta folders
                for staid1 in self.waveforms.list():
                    logstadir   = logmondir+'/'+staid1
                    if os.path.isdir(logstadir):
                        numlogs = len(os.listdir(logstadir))
                        if numlogs == 0:
                            os.rmdir(logstadir)
                continue
            #===============================
            # Cross-correlation computation
            #===============================
            print ('[%s] [XCORR] computating: ' %datetime.now().isoformat().split('.')[0] +str(stime.year)+'.'+monthdict[stime.month]+' : '+ str(len(xcorr_lst)) + ' pairs')
            # parallelized run
            if parallel:
                #-----------------------------------------
                # Computing xcorr with multiprocessing
                #-----------------------------------------
                if len(xcorr_lst) > subsize:
                    Nsub            = int(len(xcorr_lst)/subsize)
                    for isub in range(Nsub):
                        print ('[%s] [XCORR] subset:' %datetime.now().isoformat().split('.')[0], isub, 'in', Nsub, 'sets')
                        cxcorrLst   = xcorr_lst[isub*subsize:(isub+1)*subsize]
                        XCORR       = partial(_xcorr_funcs.amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft, runtype = runtype, verbose=verbose, verbose2=verbose2)
                        pool        = multiprocessing.Pool(processes=nprocess)
                        pool.map(XCORR, cxcorrLst) #make our results with a map call
                        pool.close() #we are not adding any more processes
                        pool.join() #tell it to wait until all threads are done before going on
                    cxcorrLst       = xcorr_lst[(isub+1)*subsize:]
                    XCORR           = partial(_xcorr_funcs.amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft, runtype = runtype, verbose=verbose, verbose2=verbose2)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(XCORR, cxcorrLst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                else:
                    XCORR           = partial(_xcorr_funcs.amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft, runtype = runtype, verbose=verbose, verbose2=verbose2)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(XCORR, xcorr_lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
            else:
                for ilst in range(len(xcorr_lst)):
                    try:
                        xcorr_lst[ilst].convert_amph_to_xcorr(datadir=datadir, chans=chans, ftlen = ftlen,\
                            tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                fprcs = fprcs, fastfft=fastfft, runtype = runtype, verbose=verbose, verbose2=verbose2)
                    except:
                        staid1  = xcorr_lst[ilst].netcode1 + '.' + xcorr_lst[ilst].stacode1
                        staid2  = xcorr_lst[ilst].netcode2 + '.' + xcorr_lst[ilst].stacode2
                        logfname= datadir+'/log_xcorr/'+xcorr_lst[ilst].monthdir+'/'+staid1+'/'+staid1+'_'+staid2+'.log'
                        with open(logfname, 'w') as fid:
                            fid.writelines('FAILED\n')
            #==================================
            # End cross-correlation computation
            #==================================
            # delete empty log-sta folders
            for staid1 in self.waveforms.list():
                logstadir   = logmondir+'/'+staid1
                if os.path.isdir(logstadir):
                    numlogs = len(os.listdir(logstadir))
                    if numlogs == 0:
                        os.rmdir(logstadir)
            #==========================
            # check all the log files
            #==========================
            Msuccess    = 0
            Mskipped    = 0
            Mfailed     = 0
            Mnodata     = 0
            for staid1 in self.waveforms.list():
                logstadir   = logmondir+'/'+staid1
                if not os.path.isdir(logstadir):
                    continue
                # Loop over station 2
                for staid2 in self.waveforms.list():
                    logfname= logstadir+'/'+staid1+'_'+staid2+'.log'
                    if not os.path.isfile(logfname):
                        continue
                    with open(logfname, 'r') as fid:
                        logflag = fid.readlines()[0].split()[0]
                    if logflag == 'SUCCESS':
                        Nsuccess    += 1
                        Msuccess    += 1
                        successstr  += (str(stime.year)+'.'+monthdict[stime.month]+'.'+staid1+'_'+staid2+'\n')
                    elif logflag == 'SKIPPED':
                        Nskipped    += 1
                        Mskipped    += 1
                        skipstr     += (str(stime.year)+'.'+monthdict[stime.month]+'.'+staid1+'_'+staid2+'\n')
                    elif logflag == 'FAILED':
                        Nfailed     += 1
                        Mfailed     += 1
                        failstr     += (str(stime.year)+'.'+monthdict[stime.month]+'.'+staid1+'_'+staid2+'\n')
                    elif logflag == 'NODATA':
                        Nnodata     += 1
                        Mnodata     += 1
                        nodatastr   += (str(stime.year)+'.'+monthdict[stime.month]+'.'+staid1+'_'+staid2+'\n')
                    else:
                        raise xcorrError('!!! UNEXPECTED log flag = '+logflag)
            print ('[%s] [XCORR] computation done: ' %datetime.now().isoformat().split('.')[0]+str(stime.year)+'.'+monthdict[stime.month] +\
                   ' success/nodata/skip/fail: %d/%d/%d/%d' %(Msuccess, Mnodata, Mskipped, Mfailed))
            if stime.month == 12:
                stime       = obspy.UTCDateTime(str(stime.year + 1)+'0101')
            else:
                stime       = obspy.UTCDateTime(str(stime.year)+'%02d01' %(stime.month+1))
        # summarize the log information
        if Nsuccess>0:
            successstr  = 'Total pairs = %d\n' %Nsuccess + successstr
            logsuccess  = datadir+'/log_xcorr/success.log'
            with open(logsuccess, 'w') as fid:
                fid.writelines(successstr)
        if Nskipped>0:
            skipstr     = 'Total pairs = %d\n' %Nskipped+ skipstr
            logskip     = datadir+'/log_xcorr/skipped.log'
            with open(logskip, 'w') as fid:
                fid.writelines(skipstr)
        if Nfailed>0:
            failstr     = 'Total pairs = %d\n' %Nfailed+ failstr
            logfail     = datadir+'/log_xcorr/failed.log'
            with open(logfail, 'w') as fid:
                fid.writelines(failstr)
        if Nnodata>0:
            nodatastr   = 'Total pairs = %d\n' %Nnodata+ nodatastr
            lognodata   = datadir+'/log_xcorr/nodata.log'
            with open(lognodata, 'w') as fid:
                fid.writelines(nodatastr)
        print ('[%s] [XCORR] computation ALL done: success/nodata/skip/fail: %d/%d/%d/%d' %(datetime.now().isoformat().split('.')[0], Nsuccess, Nnodata, Nskipped, Nfailed))
        return
     
    def stack(self, datadir, startyear, startmonth, endyear, endmonth, pfx='COR', skipinv=True, outdir=None, \
                chan_types = [], channels=['LHZ'], fnametype=1, verbose=False):
        """Stack cross-correlation data from monthly-stacked sac files
        ===========================================================================================================
        ::: input parameters :::
        datadir                 - data directory
        startyear, startmonth   - start date for stacking
        endyear, endmonth       - end date for stacking
        pfx                     - prefix
        outdir                  - output directory (None is not to save sac files)
        chan_types              - types (also used as priorities) of channels, used for hybrid channel xcorr
        channels                - input channels
        fnametype               - input sac file name type
                                    =1: datadir/2011.JAN/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
                                    =2: datadir/2011.JAN/COR/G12A/COR_G12A_R21A.SAC
                                    =3: datadir/2011.JAN/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC, deprecated
        -----------------------------------------------------------------------------------------------------------
        ::: output :::
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ===========================================================================================================
        """
        #---------------------------------
        # check the channel related input
        #---------------------------------
        if len(chan_types) > 0:
            if fnametype != 1:
                raise xcorrError('Hybrid channel stack only works for fnametyoe = 1')
            print ('[%s] [STACK] Hybrid channel stack!' %datetime.now().isoformat().split('.')[0])
            for tmpch in channels:
                for chtype in chan_types:
                    if len(chtype + tmpch) != 3:
                        raise xcorrError('Invalid Hybrid channel: '+ chtype + tmpch)
        #----------------------------------------
        # prepare year/month list for stacking
        #----------------------------------------
        print('[%s] [STACK] preparing month list for stacking' %datetime.now().isoformat().split('.')[0])
        utcdate         = obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst            = np.array([], dtype=int)
        mlst            = np.array([], dtype=int)
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst        = np.append(ylst, utcdate.year)
            mlst        = np.append(mlst, utcdate.month)
            if utcdate.month == 12:
                utcdate = obspy.UTCDateTime(str(utcdate.year + 1)+'0101')
            else:
                utcdate = obspy.UTCDateTime(str(utcdate.year)+'%02d01' %(utcdate.month+1))
        mnumb           = mlst.size
        #--------------------------------------------------
        # main loop for station pairs
        #--------------------------------------------------
        staLst                  = self.waveforms.list()
        Nsta                    = len(staLst)
        Ntotal_traces           = Nsta*(Nsta-1)/2
        itrstack                = 0
        Ntr_one_percent         = int(Ntotal_traces/100.)
        ipercent                = 0
        print ('[%s] [STACK] start stacking: '%datetime.now().isoformat().split('.')[0] +str(Ntotal_traces)+' pairs')
        for staid1 in staLst:
            netcode1, stacode1  = staid1.split('.')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stainv1         = self.waveforms[staid1].StationXML.networks[0].stations[0]
                st_date1        = stainv1.start_date
                ed_date1        = stainv1.end_date
                lon1            = stainv1.longitude
                lat1            = stainv1.latitude
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stainv2         = self.waveforms[staid2].StationXML.networks[0].stations[0]
                    st_date2        = stainv2.start_date
                    ed_date2        = stainv2.end_date
                    lon2            = stainv2.longitude
                    lat2            = stainv2.latitude
                if fnametype == 1:
                    if staid1 >= staid2:
                        continue
                else:
                    if stacode1 >= stacode2:
                        continue
                itrstack            += 1
                # print the status of stacking
                ipercent            = float(itrstack)/float(Ntotal_traces)*100.
                if np.fmod(itrstack, 500) == 0 or np.fmod(itrstack, Ntr_one_percent) ==0:
                    # percent_str     = '%0.2f' %ipercent
                    print ('[%s] [STACK] Number of traces finished: %d/%d   %0.2f'
                           %(datetime.now().isoformat().split('.')[0], itrstack, Ntotal_traces, ipercent)+' %')
                # skip if no overlaped time
                if (st_date1 > ed_date2 or st_date2 > ed_date1) and skipinv:
                    continue
                stackedST           = []
                init_stack_flag     = False
                #-------------------------------
                # used for hybrid channel types
                #-------------------------------
                channels1           = []
                channels2           = []
                #--------------------------------
                # Loop over month for stacking
                #--------------------------------
                for im in range(mnumb):
                    month           = monthdict[mlst[im]]
                    yrmonth         = str(ylst[im])+'.'+month
                    if fnametype == 1:
                        subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1
                    else:
                        subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1
                    if not os.path.isdir(subdir):
                        continue
                    # define the first day and last day of the current month
                    c_stime     = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im])+'-1')
                    try:
                        c_etime = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im]+1)+'-1')
                    except ValueError:
                        c_etime = obspy.UTCDateTime(str(ylst[im]+1)+'-1-1')
                    # skip if either of the stations out of time range
                    if skipinv and (st_date1 > c_etime or ed_date1 < c_stime or st_date2 > c_etime or ed_date2 < c_stime):
                        continue 
                    skip_this_month = False
                    cST             = []
                    # only one type of channels
                    if len(chan_types) == 0:
                        for chan1 in channels:
                            if skip_this_month:
                                break
                            for chan2 in channels:
                                if fnametype    == 1:
                                    fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'.SAC'
                                elif fnametype  == 2:
                                    fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                                #----------------------------------------------------------
                                elif fnametype  == 3:
                                    fname   = ''
                                #----------------------------------------------------------
                                if not os.path.isfile(fname):
                                    skip_this_month = True
                                    break
                                try:
                                    # I/O through obspy.io.sac.SACTrace.read() is ~ 10 times faster than obspy.read()
                                    tr              = obspy.io.sac.SACTrace.read(fname)
                                except TypeError:
                                    warnings.warn('Unable to read SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                    skip_this_month = True
                                    break
                                # added on 2018-02-27
                                if (abs(tr.evlo - lon1) > 0.001) or (abs(tr.evla - lat1) > 0.001) \
                                        or (abs(tr.stlo - lon2) > 0.001) or (abs(tr.stla - lat2) > 0.001):
                                    print ('WARNING: Same station id but different locations detected ' + staid1 +'_'+ staid2)
                                    print ('FILENAME: '+ fname)
                                    skip_this_month = True
                                    break
                                if (np.isnan(tr.data)).any() or abs(tr.data.max())>1e20:
                                    warnings.warn('NaN monthly SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                    skip_this_month = True
                                    break
                                cST.append(tr)
                    # hybrid channels
                    else:
                        itrace      = 0
                        for chan1 in channels:
                            if skip_this_month:
                                break
                            for chan2 in channels:
                                is_data     = False
                                # choose channel types if channels1 and channels2 not initilized
                                if not init_stack_flag:
                                    for chtype1 in chan_types:
                                        if is_data:
                                            break
                                        for chtype2 in chan_types:
                                            fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+ chtype1 + chan1\
                                                      + '_' + staid2 + '_' + chtype2 + chan2 + '.SAC'
                                            if os.path.isfile(fname):
                                                is_data = True
                                                channels1.append(chtype1 + chan1)
                                                channels2.append(chtype2 + chan2)
                                                break
                                else:
                                    channel1= channels1[itrace]
                                    channel2= channels2[itrace]
                                    fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+ channel1\
                                                      + '_' + staid2 + '_' + channel2 + '.SAC'
                                    is_data = os.path.isfile(fname)
                                if not is_data:
                                    skip_this_month = True
                                    break
                                try:
                                    # I/O through obspy.io.sac.SACTrace.read() is ~ 10 times faster than obspy.read()
                                    tr              = obspy.io.sac.SACTrace.read(fname)
                                except TypeError:
                                    warnings.warn('Unable to read SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                    skip_this_month = True
                                    break
                                # added on 2018-02-27
                                if (abs(tr.evlo - lon1) > 0.001) or (abs(tr.evla - lat1) > 0.001) \
                                        or (abs(tr.stlo - lon2) > 0.001) or (abs(tr.stla - lat2) > 0.001):
                                    print ('WARNING: Same station id but different locations detected ' + staid1 +'_'+ staid2)
                                    print ('FILENAME: '+ fname)
                                    skip_this_month = True
                                    break
                                if (np.isnan(tr.data)).any() or abs(tr.data.max())>1e20:
                                    warnings.warn('NaN monthly SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                    skip_this_month = True
                                    break
                                cST.append(tr)
                                itrace  += 1
                        # debug purpose
                        if (len(channels1) != (len(channels)*len(channels)) or len(channels2) != (len(channels)*len(channels))) \
                            and (not skip_this_month):
                            raise xcorrError('CHECK!'+staid1+' '+staid2+ ' '+yrmonth)
                    if len(cST) != (len(channels)*len(channels)) or skip_this_month:
                        continue
                    # stacking
                    if init_stack_flag:
                        for itr in range(len(cST)):
                            mtr                             = cST[itr]
                            stackedST[itr].data             += mtr.data
                            stackedST[itr].user0            += mtr.user0
                    else:
                        stackedST                           = copy.deepcopy(cST)
                        init_stack_flag                     = True
                #------------------------------------------------------------
                # finish stacking for a statin pair, save data
                #------------------------------------------------------------
                if len(stackedST) == (len(channels)*len(channels)):
                    if verbose:
                        print('Finished stacking for:'+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2)
                    # create sac output directory 
                    if outdir is not None:
                        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
                            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
                    # write cross-correlation header information
                    xcorr_header            = xcorr_header_default.copy()
                    xcorr_header['b']       = stackedST[0].b
                    xcorr_header['e']       = stackedST[0].e
                    xcorr_header['netcode1']= netcode1
                    xcorr_header['netcode2']= netcode2
                    xcorr_header['stacode1']= stacode1
                    xcorr_header['stacode2']= stacode2
                    xcorr_header['npts']    = stackedST[0].npts
                    xcorr_header['delta']   = stackedST[0].delta
                    xcorr_header['stackday']= stackedST[0].user0
                    dist, az, baz           = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                    dist                    = dist/1000.
                    xcorr_header['dist']    = dist
                    xcorr_header['az']      = az
                    xcorr_header['baz']     = baz
                    if staid1 > staid2:
                        staid_aux           = netcode2+'/'+stacode2+'/'+netcode1+'/'+stacode1
                    else:
                        staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                    itrace                  = 0
                    for ich1 in range(len(channels)):
                        for ich2 in range(len(channels)):
                            if len(chan_types) == 0: # single channel type
                                chan1       = channels[ich1]
                                chan2       = channels[ich2]
                            else: # multiple channel types
                                try:
                                    chan1       = channels1[itrace]
                                    chan2       = channels2[itrace]
                                except IndexError:
                                    print (channels1)
                                    print (channels2)
                                    print (itrace)
                                    chan1       = channels1[itrace]
                                    chan2       = channels2[itrace]
                            stackedTr       = stackedST[itrace]
                            if outdir is not None:
                                outfname    = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ pfx+'_'+netcode1+'.'+stacode1+\
                                                        '_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
                                stackedTr.write(outfname)
                            xcorr_header['chan1']   = chan1
                            xcorr_header['chan2']   = chan2
                            # check channels
                            if stackedST[itrace].kcmpnm is not None:
                                if stackedST[itrace].kcmpnm != xcorr_header['chan1'] + xcorr_header['chan2']:
                                    raise xcorrHeaderError('Inconsistent channels: '+ stackedST[itrace].kcmpnm+' '+\
                                                xcorr_header['chan1']+' '+ xcorr_header['chan2'])
                            self.add_auxiliary_data(data=stackedTr.data, data_type='NoiseXcorr',\
                                                    path=staid_aux+'/'+chan1+'/'+chan2, parameters=xcorr_header)
                            itrace                  += 1
        print ('[%s] [STACK] All stacking done: '%datetime.now().isoformat().split('.')[0] +str(Ntotal_traces)+' pairs')
        return
    
    def append(self, input_asdf, datadir, startyear, startmonth, endyear, endmonth, \
                    skipinv=True, pfx='COR', outdir=None, channels=['LHZ'], fnametype=1, verbose=False):
        """Append cross-correlation data from monthly-stacked sac files to an existing ASDF database
        ===========================================================================================================
        ::: input parameters :::
        input_asdf              - input ASDF file name
        datadir                 - data directory
        startyear, startmonth   - start date for stacking
        endyear, endmonth       - end date for stacking
        pfx                     - prefix
        outdir                  - output directory (None is not to save sac files)
        inchannels              - input channels, if None, will read channel information from obspy inventory
        fnametype               - input sac file name type
                                    =1: datadir/2011.JAN/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
                                    =2: datadir/2011.JAN/COR/G12A/COR_G12A_R21A.SAC
                                    =3: datadir/2011.JAN/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC
        -----------------------------------------------------------------------------------------------------------
        ::: output :::
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
        ===========================================================================================================
        """
        #----------------------------------------
        # copy station inventory
        #----------------------------------------
        indset                  = noiseASDF(input_asdf)
        stalst                  = indset.waveforms.list()
        for staid in stalst:
            self.add_stationxml(indset.waveforms[staid].StationXML)
        #----------------------------------------
        # prepare year/month list for stacking
        #----------------------------------------
        print('[%s] [APPEND] preparing month list for appending' %datetime.now().isoformat().split('.')[0])
        utcdate                 = obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst                    = np.array([], dtype=int)
        mlst                    = np.array([], dtype=int)
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst                = np.append(ylst, utcdate.year)
            mlst                = np.append(mlst, utcdate.month)
            if utcdate.month == 12:
                utcdate = obspy.UTCDateTime(str(utcdate.year + 1)+'0101')
            else:
                utcdate = obspy.UTCDateTime(str(utcdate.year)+'%02d01' %(utcdate.month+1))
        mnumb                   = mlst.size
        #--------------------------------------------------
        # main loop for station pairs
        #--------------------------------------------------
        staLst                  = self.waveforms.list()
        Nsta                    = len(staLst)
        Ntotal_traces           = Nsta*(Nsta-1)/2
        itrstack                = 0
        Ntr_one_percent         = int(Ntotal_traces/100.)
        ipercent                = 0
        print ('[%s] [STACK] start stacking: '%datetime.now().isoformat().split('.')[0] +str(Ntotal_traces)+' pairs')
        for staid1 in staLst:
            netcode1, stacode1  = staid1.split('.')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stainv1             = self.waveforms[staid1].StationXML.networks[0].stations[0]
                lon1                = stainv1.longitude
                lat1                = stainv1.latitude
                st_date1            = stainv1.start_date
                ed_date1            = stainv1.end_date
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stainv2         = self.waveforms[staid2].StationXML.networks[0].stations[0]
                    lon2            = stainv2.longitude
                    lat2            = stainv2.latitude
                    st_date2        = stainv2.start_date
                    ed_date2        = stainv2.end_date
                if fnametype == 1:
                    if staid1 >= staid2:
                        continue
                else:
                    if stacode1 >= stacode2:
                        continue
                itrstack            += 1
                # print the status of stacking
                ipercent            = float(itrstack)/float(Ntotal_traces)*100.
                if np.fmod(itrstack, 500) == 0 or np.fmod(itrstack, Ntr_one_percent) ==0:
                    print ('[%s] [STACK] Number of traces finished stacking: %d/%d   %0.2f'
                           %(datetime.now().isoformat().split('.')[0], itrstack, Ntotal_traces, ipercent)+' %')
                # if no overlaped time
                if skipinv and (st_date1 > ed_date2 or st_date2 > ed_date1):
                    continue
                stackedST           = []
                init_stack_flag     = False
                # Loop over months
                for im in range(mnumb):
                    month           = monthdict[mlst[im]]
                    yrmonth         = str(ylst[im])+'.'+month
                    # skip if directory not exist
                    if fnametype == 1:
                        subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1
                    else:
                        subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1
                    if not os.path.isdir(subdir):
                        continue
                    # define the first day and last day of the current month
                    c_stime     = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im])+'-1')
                    try:
                        c_etime = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im]+1)+'-1')
                    except ValueError:
                        c_etime = obspy.UTCDateTime(str(ylst[im]+1)+'-1-1')
                    # skip if either of the stations out of time range
                    if skipinv and (st_date1 > c_etime or ed_date1 < c_stime or st_date2 > c_etime or ed_date2 < c_stime):
                        continue 
                    skip_this_month = False
                    cST             = []
                    for chan1 in channels:
                        if skip_this_month:
                            break
                        for chan2 in channels:
                            month       = monthdict[mlst[im]]
                            yrmonth     = str(ylst[im])+'.'+month
                            if fnametype    == 1:
                                fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'.SAC'
                            elif fnametype  == 2:
                                fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                            #----------------------------------------------------------
                            elif fnametype  == 3:
                                fname   = ''
                                # fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'\
                                #             +stacode2+'_'+chan2.code+'.SAC'
                            #----------------------------------------------------------
                            if not os.path.isfile(fname):
                                skip_this_month = True
                                break
                            try:
                                # I/O through obspy.io.sac.SACTrace.read() is ~ 10 times faster than obspy.read()
                                tr              = obspy.io.sac.SACTrace.read(fname)
                            except TypeError:
                                warnings.warn('Unable to read SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skip_this_month = True
                                break
                            if (np.isnan(tr.data)).any() or abs(tr.data.max())>1e20:
                                warnings.warn('NaN monthly SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skip_this_month = True
                                break
                            cST.append(tr)
                    if len(cST) != (len(channels)*len(channels)) or skip_this_month:
                        continue
                    if init_stack_flag:
                        for itr in range(len(cST)):
                            mtr                     = cST[itr]
                            stackedST[itr].data     += mtr.data
                            stackedST[itr].user0    += mtr.user0
                    else:
                        stackedST                   = copy.deepcopy(cST)
                        init_stack_flag             = True
                # end of loop over month
                if len(stackedST)== (len(channels)*len(channels)):
                    if verbose:
                        print('Finished stacking for:'+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2)
                    # create sac output directory 
                    if outdir is not None:
                        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
                            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
                    #----------------------------------------------
                    # write cross-correlation header information
                    #----------------------------------------------
                    xcorr_header            = xcorr_header_default.copy()
                    xcorr_header['b']       = stackedST[0].b
                    xcorr_header['e']       = stackedST[0].e
                    xcorr_header['netcode1']= netcode1
                    xcorr_header['netcode2']= netcode2
                    xcorr_header['stacode1']= stacode1
                    xcorr_header['stacode2']= stacode2
                    xcorr_header['npts']    = stackedST[0].npts
                    xcorr_header['delta']   = stackedST[0].delta
                    xcorr_header['stackday']= stackedST[0].user0
                    dist, az, baz           = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                    dist                    = dist/1000.
                    xcorr_header['dist']    = dist
                    xcorr_header['az']      = az
                    xcorr_header['baz']     = baz
                    # determine whether append data or not
                    is_append               = True
                    if staid1 > staid2:
                        staid_aux                   = netcode2+'/'+stacode2+'/'+netcode1+'/'+stacode1
                        try:
                            instapair               = indset.auxiliary_data['NoiseXcorr'][netcode2][stacode2][netcode1][stacode1]
                        except KeyError:
                            is_append               = False
                    else:
                        staid_aux                   = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                        try:
                            instapair               = indset.auxiliary_data['NoiseXcorr'][netcode1][stacode1][netcode2][stacode2]
                        except KeyError:
                            is_append               = False
                    itrace                          = 0
                    for chan1 in channels:
                        for chan2 in channels:
                            stackedTr               = stackedST[itrace]
                            if outdir is not None:
                                outfname            = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
                                stackedTr.write(outfname)
                            xcorr_header['chan1']   = chan1
                            xcorr_header['chan2']   = chan2
                            if is_append:
                                # get data from original database
                                indata              = instapair[chan1][chan2]
                                orig_sdays          = indata.parameters['stackday']
                                orig_data           = indata.data.value
                                xcorr_header['stackday']\
                                                    += orig_sdays
                                stackedTr.data      += orig_data
                                self.add_auxiliary_data(data = stackedTr.data, data_type='NoiseXcorr', \
                                                    path = staid_aux+'/'+chan1+'/'+chan2, parameters=xcorr_header)
                            else:
                                self.add_auxiliary_data(data = stackedTr.data, data_type='NoiseXcorr', \
                                                    path = staid_aux+'/'+chan1+'/'+chan2, parameters=xcorr_header)
                            itrace                  += 1
                else:
                    #==========================
                    # copy existing xcorr data
                    #==========================
                    is_copy                         = True
                    if staid1 > staid2:
                        staid_aux                   = netcode2+'/'+stacode2+'/'+netcode1+'/'+stacode1
                        try:
                            instapair               = indset.auxiliary_data['NoiseXcorr'][netcode2][stacode2][netcode1][stacode1]
                        except KeyError:
                            is_copy                 = False
                    else:
                        staid_aux                   = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                        try:
                            instapair               = indset.auxiliary_data['NoiseXcorr'][netcode1][stacode1][netcode2][stacode2]
                        except KeyError:
                            is_copy                 = False
                    if is_copy:
                        for chan1 in channels:
                            for chan2 in channels:
                                indata              = instapair[chan1][chan2]
                                xcorr_header        = indata.parameters
                                xcorr_data          = indata.data.value
                                self.add_auxiliary_data(data=xcorr_data, data_type='NoiseXcorr', \
                                            path=staid_aux+'/'+chan1+'/'+chan2, parameters=xcorr_header)
        return
    
    def rotation(self, outdir = None, pfx = 'COR', rotatetype='RT', chan_types = ['LH', 'BH', 'HH'], verbose=False):
        """Rotate cross-correlation data 
        ===========================================================================================================
        ::: input parameters :::
        outdir              - output directory for sac files (None is not to write)
        pfx                 - prefix
        rotatetype          - type of rotation ('RT' or 'RTZ')
        chantype            - type of channel
        -----------------------------------------------------------------------------------------------------------
        ::: output :::
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ===========================================================================================================
        """
        if not rotatetype in ['RT', 'RTZ']:
            raise xcorrError('Unexpected rotation type: '+rotatetype)
        staLst                  = self.waveforms.list()
        Nsta                    = len(staLst)
        Ntotal_traces           = int(Nsta*(Nsta-1)/2)
        itrstack                = 0
        Ntr_one_percent         = int(Ntotal_traces/100.)
        irotate                 = 0
        print ('[%s] [ROTATION] start rotation: ' %datetime.now().isoformat().split('.')[0] +str(Ntotal_traces)+' pairs')
        for staid1 in staLst:
            netcode1, stacode1      = staid1.split('.')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos1         = self.waveforms[staid1].coordinates
                lat1            = tmppos1['latitude']
                lon1            = tmppos1['longitude']
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos2         = self.waveforms[staid2].coordinates
                    lat2            = tmppos2['latitude']
                    lon2            = tmppos2['longitude']
                irotate             += 1
                # print the status of rotation
                ipercent            = float(irotate)/float(Ntotal_traces)*100.
                if np.fmod(irotate, 500) == 0 or np.fmod(irotate, Ntr_one_percent) == 0:
                    percent_str     = '%0.2f' %ipercent
                    print ('[%s] [ROTATION] Number of traces finished: ' %datetime.now().isoformat().split('.')[0] \
                           +str(irotate)+'/'+str(Ntotal_traces)+' '+percent_str+'%')
                try:
                    subdset = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2]
                except KeyError:
                    continue
                #=================
                # EN->RT rotation
                #=================
                chantype1   = None
                for chtype1 in chan_types:
                    chan1E  = chtype1 + 'E'
                    chan1N  = chtype1 + 'N'
                    if chan1E in subdset.list() and chan1N in subdset.list():
                        chantype1   = chtype1
                        break
                if chantype1 is None:
                    continue
                chantype2   = None
                for chtype2 in chan_types:
                    chan2E  = chtype2 + 'E'
                    chan2N  = chtype2 + 'N'
                    if chan2E in subdset[chan1E].list() and chan2N in subdset[chan1E].list():
                        chantype2   = chtype2
                        break
                if chantype2 is None:
                    continue
                dsetEE      = subdset[chan1E][chan2E]
                dsetEN      = subdset[chan1E][chan2N]
                dsetNE      = subdset[chan1N][chan2E]
                dsetNN      = subdset[chan1N][chan2N]
                temp_header = dsetEE.parameters.copy()
                chan1R      = chantype1 + 'R'
                chan1T      = chantype1 + 'T'
                chan2R      = chantype2 + 'R'
                chan2T      = chantype2 + 'T'
                # define azimuth/back-azimuth
                theta           = temp_header['az']
                psi             = temp_header['baz']
                # check az/baz
                dist, az, baz   = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                dist            = dist/1000.
                if abs(az - theta) > 0.01 and abs(abs(az - theta) - 360.) > 0.01:
                    raise ValueError('computed az = '+str(az)+' stored az = '+str(theta)+' '+staid1+'_'+staid2)
                if abs(baz - psi) > 0.01 and abs(abs(baz - psi) - 360.) > 0.01:
                    raise ValueError('computed baz = '+str(baz)+' stored baz = '+str(psi)+' '+staid1+'_'+staid2)
                Ctheta      = np.cos(np.pi*theta/180.)
                Stheta      = np.sin(np.pi*theta/180.)
                Cpsi        = np.cos(np.pi*psi/180.)
                Spsi        = np.sin(np.pi*psi/180.)
                #------------------------------- perform EN -> RT rotation ------------------------------
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tempTT      = -Ctheta*Cpsi* dsetEE.data[()] + Ctheta*Spsi* dsetEN.data[()] - \
                                        Stheta*Spsi* dsetNN.data[()] + Stheta*Cpsi* dsetNE.data[()]
                    
                    tempRR      = - Stheta*Spsi* dsetEE.data[()] - Stheta*Cpsi* dsetEN.data[()] \
                                        -Ctheta*Cpsi*dsetNN.data[()] - Ctheta*Spsi*dsetNE.data[()]
                    
                    tempTR      = -Ctheta*Spsi* dsetEE.data[()] - Ctheta*Cpsi* dsetEN.data[()]  \
                                        + Stheta*Cpsi*dsetNN.data[()] + Stheta*Spsi*dsetNE.data[()]
                    
                    tempRT      = -Stheta*Cpsi* dsetEE.data[()] + Stheta*Spsi* dsetEN.data[()] \
                                        + Ctheta*Spsi* dsetNN.data[()] - Ctheta*Cpsi* dsetNE.data[()]
                #----------------------------------------------------------------------------------------
                # save horizontal components
                staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                temp_header['chan1']= chan1T
                temp_header['chan2']= chan2T
                self.add_auxiliary_data(data=tempTT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2T, parameters=temp_header)
                
                temp_header['chan1']= chan1R
                temp_header['chan2']= chan2R
                self.add_auxiliary_data(data=tempRR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2R, parameters=temp_header)
                
                temp_header['chan1']= chan1T
                temp_header['chan2']= chan2R
                self.add_auxiliary_data(data=tempTR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2R, parameters=temp_header)
                
                temp_header['chan1']= chan1R
                temp_header['chan2']= chan2T
                self.add_auxiliary_data(data=tempRT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2T, parameters=temp_header)
                # write to sac files
                if outdir is not None:
                    outstadir   = outdir +'/COR_ROTATE/'+staid1
                    if not os.path.isdir(outstadir):
                        os.makedirs(outstadir)
                    self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chanT, chan2=chanT, outdir=outstadir, pfx=pfx)
                    self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chanR, chan2=chanR, outdir=outstadir, pfx=pfx)
                    self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chanT, chan2=chanR, outdir=outstadir, pfx=pfx)
                    self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chanR, chan2=chanT, outdir=outstadir, pfx=pfx)
                #==================
                # ENZ->RTZ rotation
                #==================
                if rotatetype == 'RTZ':
                    chan1Z  = chantype1 + 'Z'
                    chan2Z  = chantype2 + 'Z'
                    # get data
                    dsetEZ  = subdset[chan1E][chan2Z]
                    dsetZE  = subdset[chan1Z][chan2E]
                    dsetNZ  = subdset[chan1N][chan2Z]
                    dsetZN  = subdset[chan1Z][chan2N]
                    # ----------------------- perform ENZ -> RTZ rotation ---------------------
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tempRZ  = Ctheta* dsetNZ.data[()] + Stheta* dsetEZ.data[()]
                        tempZR  = -Cpsi* dsetZN.data[()] - Spsi* dsetZE.data[()]
                        tempTZ  = -Stheta* dsetNZ.data[()] + Ctheta* dsetEZ.data[()]
                        tempZT  = Spsi* dsetZN.data[()] - Cpsi* dsetZE.data[()]
                    #--------------------------------------------------------------------------
                    temp_header['chan1']    = chan1R
                    temp_header['chan2']    = chan2Z
                    self.add_auxiliary_data(data=tempRZ, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2Z, parameters=temp_header)
                    temp_header['chan1']    = chan1Z
                    temp_header['chan2']    = chan2R
                    self.add_auxiliary_data(data=tempZR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1Z+'/'+chan2R, parameters=temp_header)
                    temp_header['chan1']    = chan1T
                    temp_header['chan2']    = chan2Z
                    self.add_auxiliary_data(data=tempTZ, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2Z, parameters=temp_header)
                    temp_header['chan1']    = chan1Z
                    temp_header['chan2']    = chan2T
                    self.add_auxiliary_data(data=tempZT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1Z+'/'+chan2T, parameters=temp_header)
                    # write to sac files
                    if outdir is not None:
                        self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1R, chan2=chan2Z, outdir=outstadir, pfx=pfx)                        
                        self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1Z, chan2=chan2R, outdir=outstadir, pfx=pfx)
                        self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1T, chan2=chan2Z, outdir=outstadir, pfx=pfx)
                        self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1Z, chan2=chan2T, outdir=outstadir, pfx=pfx)
                if verbose:
                    print('Finished rotation for:'+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2)
        print ('[%s] [ROTATION] All rotation done: '%datetime.now().isoformat().split('.')[0] +str(Ntotal_traces)+' pairs')
        return
    
    
    
    
    
