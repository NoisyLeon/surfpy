# -*- coding: utf-8 -*-
"""
ASDF for cross-correlation
    
:Copyright:
    Author: Lili Feng
    Research Geophysicist
    CGG
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
import warnings
import tarfile
import shutil
import glob
import sys
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'


monthdict               = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
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
    def tar_mseed_to_sac(self, datadir, outdir, start_date, end_date, sps=1., outtype=0, rmresp=False, hvflag=False,
            chtype='LH', channels='ENZ', ntaper=2, halfw=100, tb = 1., tlen = 86398., tb2 = 1000., tlen2 = 84000.,
            perl = 5., perh = 200., pfx='LF_', delete_tar=False, delete_extract=True, verbose=True, verbose2 = False):
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
        print ('=== Extracting tar mseed from: '+datadir+' to '+outdir)
        while (curtime <= endtime):
            if verbose:
                print ('--- Date: '+curtime.date.isoformat())
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    staxml  = self.waveforms[staid].StationXML
                mseedfname      = datedir + '/' + stacode+'.'+netcode+'.mseed'
                xmlfname        = datedir + '/IRISDMC-' + stacode+'.'+netcode+'.xml'
                datalessfname   = datedir + '/IRISDMC-' + stacode+'.'+netcode+'.dataless'
                # load data
                if not os.path.isfile(mseedfname):
                    if curtime >= staxml[0][0].creation_date and curtime <= staxml[0][0].end_date:
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
                                resp_inv.get_response(seed_id = seed_id, datatime = curtime)
                        except:
                            print ('*** NO RESP STATION: '+staid)
                            Nnodata     += 1
                            continue
                    else:
                        resp_inv = obspy.read_inventory(xmlfname)
                else:
                    if not os.path.isfile(datalessfname):
                        print ('*** NO DATALESS FILE STATION: '+staid)
                #===========================================
                # resample the data and perform time shift 
                #===========================================
                for i in range(len(st)):
                    # resample
                    if (abs(st[i].stats.delta - targetdt)/targetdt) < (1e-4) :
                        st[i].stats.delta           = targetdt
                    else:
                        print ('!!! RESAMPLING DATA STATION: '+staid)
                        st[i].resample(sampling_rate= sps)
                    # time shift
                    dt          = st[i].stats.delta
                    tmpstime    = st[i].stats.starttime
                    st[i].data  = st[i].data.astype(np.float64) # convert int in gains to float64
                    tdiff       = tmpstime - curtime
                    Nt          = np.floor(tdiff/dt)
                    tshift      = tdiff - Nt*dt
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
                            # trim the data for tb and tb+tlen
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
                    st2.remove_response(inventory = resp_inv, pre_filt = [f1, f2, f3, f4])
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
                print ('+++ %d/%d groups of traces extracted!' %(Ndata, Nnodata))
            # delete raw data
            if delete_extract:
                shutil.rmtree(datedir)
            if delete_tar:
                os.remove(tarlst[0])
        # End loop over dates
        print ('=== Extracted %d/%d days of data' %(Nday - Nnodataday, Nday))
        return
    
    def compute_xcorr(self, datadir, start_date, end_date, runtype=0, skipinv=False, chans=['LHZ', 'LHE', 'LHN'], \
            ftlen = True, tlen = 84000., mintlen = 20000., sps = 1., lagtime = 3000., CorOutflag = 0, \
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
                                1   - skip if log file indicates SUCCESS & SKIPPED
                                2   - skip if log file indicates SUCCESS
                                3   - skip if log file exists
                                4   - skip if montly/staid1 log directory exists
                                5   - skip if monthly log directory exists
        skipinv             - skip the month if not within the start/end date of the station inventory
        chans               - channel list
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
        successstr  = ''
        skipstr     = ''
        failstr     = ''
        #-------------------------
        # Loop over month
        #-------------------------
        print ('*** Xcorr computation START!')
        while(stime < etime):
            print ('=== Xcorr data preparing: '+str(stime.year)+'.'+monthdict[stime.month])
            month_dir   = datadir+'/'+str(stime.year)+'.'+monthdict[stime.month]
            logmondir   = datadir+'/log_xcorr/'+str(stime.year)+'.'+monthdict[stime.month]
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
                    st_date1    = self.waveforms[staid1].StationXML.networks[0].stations[0].start_date
                    ed_date1    = self.waveforms[staid1].StationXML.networks[0].stations[0].end_date
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
                        elif logflag == 'FAILED':
                            pass
                        else:
                            raise xcorrError('!!! UNEXPECTED log flag = '+logflag)
                    # end checking log files
                    netcode2, stacode2  = staid2.split('.')
                    # determine if the range of the station 2 matches current month
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        st_date2    = self.waveforms[staid2].StationXML.networks[0].stations[0].start_date
                        ed_date2    = self.waveforms[staid2].StationXML.networks[0].stations[0].end_date
                    if skipinv and (st_date2 > c_etime or ed_date2 < c_stime) :
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tmppos2     = self.waveforms[staid2].coordinates
                    stla2       = tmppos2['latitude']
                    stlo2       = tmppos2['longitude']
                    stz2        = tmppos2['elevation_in_m']
                    ctime       = obspy.UTCDateTime(str(stime.year)+'-'+str(stime.month)+'-1')
                    # day list
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
            # End loop over station1/station2/days
            if len(xcorr_lst) == 0:
                print ('!!! XCORR NO DATA: '+str(stime.year)+'.'+monthdict[stime.month])
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
            print ('--- Xcorr computating: '+str(stime.year)+'.'+monthdict[stime.month]+' : '+ str(len(xcorr_lst)) + ' pairs')
            # parallelized run
            if parallel:
                #-----------------------------------------
                # Computing xcorr with multiprocessing
                #-----------------------------------------
                if len(xcorr_lst) > subsize:
                    Nsub            = int(len(xcorr_lst)/subsize)
                    for isub in range(Nsub):
                        print ('xcorr : subset:', isub, 'in', Nsub, 'sets')
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
                    else:
                        raise xcorrError('!!! UNEXPECTED log flag = '+logflag)
            print ('=== Xcorr computation done: '+str(stime.year)+'.'+monthdict[stime.month] +\
                   ' success/skip/fail: %d/%d/%d' %(Msuccess, Mskipped, Mfailed))
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
        print ('*** Xcorr computation ALL done: success/skip/fail: %d/%d/%d' %(Nsuccess, Nskipped, Nfailed))
        return
    
    # def stack(self, datadir, startyear, startmonth, endyear, endmonth, pfx='COR', outdir=None, \
    #             inchannels=None, fnametype=1, verbose=False):
    #     """Stack cross-correlation data from monthly-stacked sac files
    #     ===========================================================================================================
    #     ::: input parameters :::
    #     datadir                 - data directory
    #     startyear, startmonth   - start date for stacking
    #     endyear, endmonth       - end date for stacking
    #     pfx                     - prefix
    #     outdir                  - output directory (None is not to save sac files)
    #     inchannels              - input channels, if None, will read channel information from obspy inventory
    #     fnametype               - input sac file name type
    #                                 =1: datadir/2011.JAN/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
    #                                 =2: datadir/2011.JAN/COR/G12A/COR_G12A_R21A.SAC
    #                                 =3: datadir/2011.JAN/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC, deprecated
    #     -----------------------------------------------------------------------------------------------------------
    #     ::: output :::
    #     ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
    #     sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
    #     ===========================================================================================================
    #     """
    #     #----------------------------------------
    #     # prepare year/month list for stacking
    #     #----------------------------------------
    #     print('=== preparing month list for stacking')
    #     utcdate                 = obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
    #     ylst                    = np.array([], dtype=int)
    #     mlst                    = np.array([], dtype=int)
    #     while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
    #         ylst                = np.append(ylst, utcdate.year)
    #         mlst                = np.append(mlst, utcdate.month)
    #         try:
    #             utcdate.month   +=1
    #         except ValueError:
    #             utcdate.year    +=1
    #             utcdate.month   = 1
    #     mnumb                   = mlst.size
    #     #--------------------------------------------------
    #     # determine channels if inchannels is specified
    #     #--------------------------------------------------
    #     if inchannels != None:
    #         try:
    #             if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
    #                 channels    = []
    #                 for inchan in inchannels:
    #                     channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='',
    #                                     latitude=0, longitude=0, elevation=0, depth=0) )
    #             else:
    #                 channels    = inchannels
    #         except:
    #             inchannels      = None
    #     if inchannels != None:
    #         chan_str_for_print      = ''
    #         for chan in channels:
    #             chan_str_for_print  += chan.code+' '
    #         print ('--- channels for stacking : '+ chan_str_for_print)
    #     #--------------------------------------------------
    #     # main loop for station pairs
    #     #--------------------------------------------------
    #     staLst                  = self.waveforms.list()
    #     Nsta                    = len(staLst)
    #     Ntotal_traces           = Nsta*(Nsta-1)/2
    #     itrstack                = 0
    #     Ntr_one_percent         = int(Ntotal_traces/100.)
    #     ipercent                = 0
    #     print ('--- start stacking: '+str(Ntotal_traces)+' pairs')
    #     for staid1 in staLst:
    #         netcode1, stacode1  = staid1.split('.')
    #         st_date1            = self.waveforms[staid1].StationXML.networks[0].stations[0].start_date
    #         ed_date1            = self.waveforms[staid1].StationXML.networks[0].stations[0].end_date
    #         lon1                = self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
    #         lat1                = self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
    #         for staid2 in staLst:
    #             netcode2, stacode2  = staid2.split('.')
    #             st_date2            = self.waveforms[staid2].StationXML.networks[0].stations[0].start_date
    #             ed_date2            = self.waveforms[staid2].StationXML.networks[0].stations[0].end_date
    #             lon2                = self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
    #             lat2                = self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
    #             if fnametype == 1:
    #                 if staid1 >= staid2:
    #                     continue
    #             else:
    #                 if stacode1 >= stacode2:
    #                     continue
    #             itrstack            += 1
    #             # print the status of stacking
    #             ipercent            = float(itrstack)/float(Ntotal_traces)*100.
    #             if np.fmod(itrstack, 500) == 0 or np.fmod(itrstack, Ntr_one_percent) ==0:
    #                 percent_str     = '%0.2f' %ipercent
    #                 print ('*** Number of traces finished stacking: '+str(itrstack)+'/'+str(Ntotal_traces)+' '+percent_str+'%')
    #             # skip if no overlaped time
    #             if st_date1 > ed_date2 or st_date2 > ed_date1:
    #                 continue
    #             stackedST           = []
    #             init_stack_flag     = False
    #             #-------------------------------------------------------------
    #             # determin channels for stacking if not specified beforehand
    #             #-------------------------------------------------------------
    #             if inchannels == None:
    #                 channels1       = []
    #                 channels2       = []
    #                 tempchans1      = self.waveforms[staid1].StationXML.networks[0].stations[0].channels
    #                 tempchans2      = self.waveforms[staid2].StationXML.networks[0].stations[0].channels
    #                 # get non-repeated component channel list
    #                 isZ             = False
    #                 isN             = False
    #                 isE             = False
    #                 for tempchan in tempchans1:
    #                     if tempchan.code[-1] == 'Z':
    #                         if isZ:
    #                             continue
    #                         else:
    #                             isZ         = True
    #                     if tempchan.code[-1] == 'N':
    #                         if isN:
    #                             continue
    #                         else:
    #                             isN         = True
    #                     if tempchan.code[-1] == 'E':
    #                         if isE:
    #                             continue
    #                         else:
    #                             isE         = True
    #                     channels1.append(tempchan)
    #                 isZ             = False
    #                 isN             = False
    #                 isE             = False
    #                 for tempchan in tempchans2:
    #                     if tempchan.code[-1] == 'Z':
    #                         if isZ:
    #                             continue
    #                         else:
    #                             isZ         = True
    #                     if tempchan.code[-1] == 'N':
    #                         if isN:
    #                             continue
    #                         else:
    #                             isN         = True
    #                     if tempchan.code[-1] == 'E':
    #                         if isE:
    #                             continue
    #                         else:
    #                             isE         = True
    #                     channels2.append(tempchan)
    #             else:
    #                 channels1       = channels
    #                 channels2       = channels
    #             #--------------------------------
    #             # Loop over month for stacking
    #             #--------------------------------
    #             for im in range(mnumb):
    #                 month           = monthdict[mlst[im]]
    #                 yrmonth         = str(ylst[im])+'.'+month
    #                 if fnametype == 1:
    #                     subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1
    #                 else:
    #                     subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1
    #                 if not os.path.isdir(subdir):
    #                     continue
    #                 # define the first day and last day of the current month
    #                 c_stime     = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im])+'-1')
    #                 try:
    #                     c_etime = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im]+1)+'-1')
    #                 except ValueError:
    #                     c_etime = obspy.UTCDateTime(str(ylst[im]+1)+'-1-1')
    #                 # skip if either of the stations out of time range
    #                 if st_date1 > c_etime or ed_date1 < c_stime or \
    #                     st_date2 > c_etime or ed_date2 < c_stime:
    #                     continue 
    #                 skip_this_month = False
    #                 cST             = []
    #                 for chan1 in channels1:
    #                     if skip_this_month:
    #                         break
    #                     for chan2 in channels2:
    #                         if fnametype    == 1:
    #                             fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+chan1.code+'_'\
    #                                         +staid2+'_'+chan2.code+'.SAC'
    #                         elif fnametype  == 2:
    #                             fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
    #                         #----------------------------------------------------------
    #                         elif fnametype  == 3:
    #                             fname   = ''
    #                             # fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'\
    #                             #             +stacode2+'_'+chan2.code+'.SAC'
    #                         #----------------------------------------------------------
    #                         if not os.path.isfile(fname):
    #                             skip_this_month = True
    #                             break
    #                         try:
    #                             # I/O through obspy.io.sac.SACTrace.read() is ~ 10 times faster than obspy.read()
    #                             tr              = obspy.io.sac.SACTrace.read(fname)
    #                         except TypeError:
    #                             warnings.warn('Unable to read SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
    #                             skip_this_month = True
    #                             break
    #                         # added on 2018-02-27
    #                         # # # if (abs(tr.stats.sac.evlo - lon1) > 0.001)\
    #                         # # #         or (abs(tr.stats.sac.evla - lat1) > 0.001) \
    #                         # # #         or (abs(tr.stats.sac.stlo - lon2) > 0.001) \
    #                         # # #         or (abs(tr.stats.sac.stla - lat2) > 0.001):
    #                         # # #     print 'WARNING: Same station code but different locations detected ' + staid1 +'_'+ staid2
    #                         # # #     print 'FILENAME: '+ fname
    #                         # # #     skipflag= True
    #                         # # #     break
    #                         if (np.isnan(tr.data)).any() or abs(tr.data.max())>1e20:
    #                             warnings.warn('NaN monthly SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
    #                             skip_this_month = True
    #                             break
    #                         cST.append(tr)
    #                 if len(cST) != (len(channels1)*len(channels2)) or skip_this_month:
    #                     continue
    #                 # stacking
    #                 if init_stack_flag:
    #                     for itr in range(len(cST)):
    #                         mtr                             = cST[itr]
    #                         stackedST[itr].data             += mtr.data
    #                         stackedST[itr].user0            += mtr.user0
    #                 else:
    #                     stackedST                           = copy.deepcopy(cST)
    #                     init_stack_flag                     = True
    #             #------------------------------------------------------------
    #             # finish stacking for a statin pair, save data
    #             #------------------------------------------------------------
    #             if len(stackedST) == (len(channels1)*len(channels2)):
    #                 if verbose:
    #                     print('Finished stacking for:'+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2)
    #                 # create sac output directory 
    #                 if outdir != None:
    #                     if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
    #                         os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
    #                 # write cross-correlation header information
    #                 xcorr_header            = xcorr_header_default.copy()
    #                 xcorr_header['b']       = stackedST[0].b
    #                 xcorr_header['e']       = stackedST[0].e
    #                 xcorr_header['netcode1']= netcode1
    #                 xcorr_header['netcode2']= netcode2
    #                 xcorr_header['stacode1']= stacode1
    #                 xcorr_header['stacode2']= stacode2
    #                 xcorr_header['npts']    = stackedST[0].npts
    #                 xcorr_header['delta']   = stackedST[0].delta
    #                 xcorr_header['stackday']= stackedST[0].user0
    #                 dist, az, baz           = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
    #                 dist                    = dist/1000.
    #                 xcorr_header['dist']    = dist
    #                 xcorr_header['az']      = az
    #                 xcorr_header['baz']     = baz
    #                 if staid1 > staid2:
    #                     staid_aux           = netcode2+'/'+stacode2+'/'+netcode1+'/'+stacode1
    #                 else:
    #                     staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
    #                 itrace                  = 0
    #                 for chan1 in channels1:
    #                     for chan2 in channels2:
    #                         stackedTr       = stackedST[itrace]
    #                         if outdir != None:
    #                             outfname            = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ pfx+'_'+netcode1+'.'+stacode1+\
    #                                                     '_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
    #                             stackedTr.write(outfname)
    #                         xcorr_header['chan1']   = chan1.code
    #                         xcorr_header['chan2']   = chan2.code
    #                         # check channels
    #                         if stackedST[itrace].kcmpnm != None:
    #                             if stackedST[itrace].kcmpnm != xcorr_header['chan1'] + xcorr_header['chan2']:
    #                                 raise ValueError('Inconsistent channels: '+ stackedST[itrace].kcmpnm+' '+\
    #                                             xcorr_header['chan1']+' '+ xcorr_header['chan2'])
    #                         self.add_auxiliary_data(data=stackedTr.data, data_type='NoiseXcorr',\
    #                                                 path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
    #                         itrace                  += 1
    #     return
    
    
    
    
    
