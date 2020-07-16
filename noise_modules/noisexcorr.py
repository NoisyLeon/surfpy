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
    import surfpy.noise_modules.noisebase as noisebase
except:
    import noisebase

try:
    import surfpy.noise_modules._xcorr_funcs as _xcorr_funcs
except:
    import _xcorr_funcs 

import numpy as np
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
    
    def compute_xcorr(self, datadir, start_date, end_date, chans=['LHZ', 'LHE', 'LHN'], \
            fskipxcorr = 0, ftlen = True, tlen = 84000., mintlen = 20000., sps = 1., lagtime = 3000., CorOutflag = 0, \
                fprcs = False, fastfft=True, parallel=True, nprocess=None, subsize=1000):
        """
        compute ambient noise cross-correlation given preprocessed amplitude and phase files
        =================================================================================================================
        ::: input parameters :::
        datadir             - directory including data and output
        startdate/enddate   - start/end date for computation           
        chans               - channel list
        fskipxcorr          - skip flags: 1 = skip upon existence of target file, 0 = overwrites
        ftlen               - turn (on/off) cross-correlation-time-length correction for amplitude
        tlen                - time length of daily records (in sec)
        mintlen             - allowed minimum time length for cross-correlation (takes effect only when ftlen = True)
        sps                 - target sampling rate
        lagtime             - cross-correlation signal half length in sec
        CorOutflag          - 0 = only output monthly xcorr data, 1 = only daily, 2 or others = output both
        fprcs               - turn on/off (1/0) precursor signal checking, NOT implemented yet
        fastfft             - speeding up the computation by using precomputed fftw_plan or not
        parallel            - run the xcorr parallelly or not
        nprocess            - number of processes
        subsize             - subsize of processing list, use to prevent lock in multiprocessing process
        =================================================================================================================
        """
        stime   = obspy.UTCDateTime(start_date)
        etime   = obspy.UTCDateTime(end_date)
        #-------------------------
        # Loop over month
        #-------------------------
        while(stime < etime):
            print ('=== Xcorr data preparing: '+str(stime.year)+'.'+monthdict[stime.month])
            month_dir   = datadir+'/'+str(stime.year)+'.'+monthdict[stime.month]
            if not os.path.isdir(month_dir):
                print ('--- Xcorr dir NOT exists : '+str(stime.year)+'.'+monthdict[stime.month])
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
                st_date1    = self.waveforms[staid1].StationXML.networks[0].stations[0].start_date
                ed_date1    = self.waveforms[staid1].StationXML.networks[0].stations[0].end_date
                if st_date1 > c_etime or ed_date1 < c_stime:
                    continue
                netcode1, stacode1  = staid1.split('.')
                #-------------------------
                # Loop over station 2
                #-------------------------
                for staid2 in self.waveforms.list():
                    if staid1 >= staid2:
                        continue
                    netcode2, stacode2  = staid2.split('.')
                    ###
                    # if staid1 != 'IU.COLA' or staid2 != 'XE.DH3':
                    #     continue
                    ###
                    # determine if the range of the station 2 matches current month
                    st_date2    = self.waveforms[staid2].StationXML.networks[0].stations[0].start_date
                    ed_date2    = self.waveforms[staid2].StationXML.networks[0].stations[0].end_date
                    if st_date2 > c_etime or ed_date2 < c_stime:
                        continue
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
                        try:
                            ctime.day   += 1
                        except ValueError:
                            break
                    if len(daylst) != 0:
                        xcorr_lst.append( xcorr_pair(stacode1 = stacode1, netcode1=netcode1,\
                            stacode2=stacode2, netcode2=netcode2, monthdir=str(stime.year)+'.'+monthdict[stime.month], daylst=daylst) )
                        ###
                        # # # return xcorr_lst
                        ###
            # End loop over station1/station2/days
            if len(xcorr_lst) == 0:
                print ('--- Xcorr NO data: '+str(stime.year)+'.'+monthdict[stime.month]+' : '+ str(len(xcorr_lst)) + ' pairs')
                if stime.month == 12:
                    stime       = obspy.UTCDateTime(str(stime.year + 1)+'0101')
                else:
                    stime.month += 1
                continue
            #--------------------------------
            # Cross-correlation computation
            #--------------------------------
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
                        XCORR       = partial(amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft)
                        pool        = multiprocessing.Pool(processes=nprocess)
                        pool.map(XCORR, cxcorrLst) #make our results with a map call
                        pool.close() #we are not adding any more processes
                        pool.join() #tell it to wait until all threads are done before going on
                    cxcorrLst       = xcorr_lst[(isub+1)*subsize:]
                    XCORR           = partial(amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(XCORR, cxcorrLst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                else:
                    XCORR           = partial(amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(XCORR, xcorr_lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
            else:
                for ilst in range(len(xcorr_lst)):
                    xcorr_lst[ilst].convert_amph_to_xcorr(datadir=datadir, chans=chans, ftlen = ftlen,\
                            tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                fprcs = fprcs, fastfft=fastfft, verbose=False)
            print ('=== Xcorr computation done: '+str(stime.year)+'.'+monthdict[stime.month])
            if stime.month == 12:
                stime       = obspy.UTCDateTime(str(stime.year + 1)+'0101')
            else:
                stime.month += 1
        return
    
    
    
    
    
    
