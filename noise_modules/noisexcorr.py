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
import numpy as np
import obspy
import obspy.io.sac
import obspy.io.xseed 
from numba import jit, float32, int32, boolean, float64, int64
import numba
import pyfftw
import warnings
import tarfile
import shutil
import glob
import sys
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'

# ------------- aftan specific exceptions ---------------------------------------
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

monthdict   = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def _tshift_fft(data, dt, tshift):
    """positive means delaying the waveform
    """
    npts    = data.size
    Np2     = int(max(1<<(npts-1).bit_length(), 2**12))
    Xf      = np.fft.rfft(data, n=Np2)
    freq    = 1./dt/Np2*np.arange((Np2/2+1), dtype = float)
    ph_shift= np.exp(-2j*np.pi*freq*tshift)
    Xf2     = Xf*ph_shift
    return np.real(np.fft.irfft(Xf2)[:npts])

@jit(numba.types.Tuple((int64[:, :], int64))(boolean[:]), nopython=True)
def _rec_lst(mask):
    reclst  = -np.ones((mask.size, 2), dtype = np.int64)
    isrec   = False
    irec    = 0
    for i in range(mask.size):
        if mask[i]:
            if isrec:
                reclst[irec, 1] = i-1
                irec            += 1
                isrec           = False
            else:
                continue
        else:
            if isrec:
                # last element
                if i == (mask.size - 1):
                    reclst[irec, 1] = i
                    irec            += 1
                continue
            else:
                isrec           = True
                reclst[irec, 0] = i
    return reclst, irec

@jit(numba.types.Tuple((int64[:, :], int64))(boolean[:]), nopython=True)
def _gap_lst(mask):
    gaplst  = -np.ones((mask.size, 2), dtype = np.int64)
    isgap   = False
    igap    = 0
    for i in range(mask.size):
        if mask[i]:
            if isgap:
                # last element
                if i == (mask.size - 1):
                    gaplst[igap, 1] = i
                    igap            += 1
                continue
            else:
                isgap           = True
                gaplst[igap, 0] = i
        else:
            if isgap:
                gaplst[igap, 1] = i-1
                igap            += 1
                isgap           = False
            else:
                continue
    return gaplst, igap
# 
@jit(float64[:](int64[:, :], int64[:, :], float64[:], int64, int64), nopython=True)
def _fill_gap_vals(gaplst, reclst, data, Ngap, halfw):
    """
    """
    alpha   = -0.5 / (halfw * halfw)
    gaparr  = np.zeros(data.size, dtype = np.float64)
    if gaplst[0, 0] < reclst[0, 0]:
        gaphead = True
    else:
        gaphead = False
    if gaplst[-1, 1] > reclst[-1, 1]:
        gaptail = True
    else:
        gaptail = False
    for igap in range(Ngap):
        igp0    = gaplst[igap, 0]
        igp1    = gaplst[igap, 1]
        tmp_npts= igp1 - igp0 + 1
        if gaphead and igap == 0:
            ilrec   = 0
            irrec   = 0
        elif gaptail and igap == (Ngap - 1):
            ilrec   = -1
            irrec   = -1
        elif gaphead:
            ilrec   = igap - 1
            irrec   = igap
        else:
            ilrec   = igap
            irrec   = igap + 1
        il0     = reclst[ilrec, 0]
        il1     = reclst[ilrec, 1]
        ir0     = reclst[irrec, 0]
        ir1     = reclst[irrec, 1]
        if (il1 - il0 + 1 ) < halfw:
            lmean   = data[il0:(il1+1)].mean()
            lstd    = data[il0:(il1+1)].std()
        else:
            lmean   = data[(il1-halfw+1):(il1+1)].mean()
            lstd    = data[(il1-halfw+1):(il1+1)].std()
        if (ir1 - ir0 + 1 ) < halfw:
            rmean   = data[ir0:(ir1+1)].mean()
            rstd    = data[ir0:(ir1+1)].std()
        else:
            rmean   = data[ir0:(ir0+halfw)].mean()
            rstd    = data[ir0:(ir0+halfw)].std()
        if gaphead and igap == 0:
            lstd= 0
        elif gaptail and igap == (Ngap - 1):
            rstd= 0
        if tmp_npts == 1:
            gaparr[igp0]    = (lmean+rmean)/2. + np.random.uniform(-(lstd+rstd)/2, (lstd+rstd)/2)
        else:
            imid    = int(np.floor(tmp_npts/2))
            for i in range(tmp_npts):
                j           = i + igp0
                slope       = (rmean - lmean)/tmp_npts * i + lmean
                if i < imid:
                    gsamp   = np.exp(alpha * i * i)
                    tmpstd  = lstd
                else:
                    gsamp   = np.exp(alpha * (tmp_npts - i - 1) * (tmp_npts - i - 1))
                    tmpstd  = rstd
                gaparr[j]   = gsamp * np.random.uniform(-tmpstd, tmpstd) + slope
    return gaparr
        
class xcorrASDF(noisebase.baseASDF):
    """ Class for xcorr process
    =================================================================================================================
    version history:
        2020/07/09
    =================================================================================================================
    """
    def tar_mseed_to_sac(self, datadir, outdir, start_date, end_date, targetdt=1., outtype=0, rmresp=False, hvflag=False, chtype='LH', channels='ENZ',
            ntaper=2, halfw=100, tb = 1., tlen = 86398., perl = 5., perh = 200., pfx='LF_', tshift_thresh = 0.01,
                delete_tar=False, delete_extract=True, verbose=True, verbose2 = False):
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
                        st[i].resample(sampling_rate= 1./targetdt)
                    # time shift
                    dt          = st[i].stats.delta
                    if ((np.ceil(tb/dt)*dt - tb) > (dt/100.)) or ((np.ceil(tlen/dt) -tlen) > (dt/100.)):
                        raise xcorrError('tb and tlen must both be multiplilier of dt!')
                    tmpstime    = st[i].stats.starttime
                    st[i].data  = st[i].data.astype(np.float64) # convert int in gains to float64
                    tdiff       = tmpstime - curtime
                    Nt          = np.floor(tdiff/dt)
                    tshift      = tdiff - Nt*dt
                    if tshift < 0.:
                        raise xcorrError('UNEXPECTED tshift = '+str(tshift)+' STATION:'+staid)
                    # apply the time shift
                    if tshift < dt*0.5:
                        st[i].data              = _tshift_fft(st[i].data, dt=dt, tshift = tshift) 
                        st[i].stats.starttime   -= tshift
                    else:
                        st[i].data              = _tshift_fft(st[i].data, dt=dt, tshift = tshift-dt ) 
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
                                gaparr, Ngap    = _gap_lst(maskZ)
                                gaplst          = gaparr[:Ngap, :]
                                # get the rec list
                                Nrecarr, Nrec   = _rec_lst(maskZ)
                                Nreclst         = Nrecarr[:Nrec, :]
                                if np.any(Nreclst<0) or np.any(gaplst<0):
                                    raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                # values for gap filling
                                fillvals        = _fill_gap_vals(gaplst, Nreclst, dataZ, Ngap, halfw)
                                trZ.data        = fillvals * maskZ + dataZ
                                if np.any(np.isnan(trZ.data)):
                                    raise xcorrDataError('NaN Z DATA STATION: '+staid)
                            else:
                                Nrec    = 0
                            st2.append(trZ)
                            isZ     = True
                    if Nrec > 0:
                        if not os.path.isdir(outdatedir):
                            os.makedirs(outdatedir)
                        print ('!!! GAP Z  STATION: '+staid)
                        with open(fnameZ+'_rec', 'w') as fid:
                            for i in range(Nrec):
                                fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
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
                        if len(StreamE) == 0 or (len(StreamN) != len(StreamE)):
                            if verbose2:
                                print ('!!! NO E or N COMPONENT STATION: '+staid)
                            Nrec    = 0
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
                                if ismask:
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
                                    gaparr, Ngap    = _gap_lst(mask)
                                    gaplst          = gaparr[:Ngap, :]
                                    # get the rec list
                                    Nrecarr, Nrec   = _rec_lst(mask)
                                    Nreclst         = Nrecarr[:Nrec, :]
                                    if np.any(Nreclst<0) or np.any(gaplst<0):
                                        raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                    # values for gap filling
                                    fillvalsE   = _fill_gap_vals(gaplst, Nreclst, dataE, Ngap, halfw)
                                    fillvalsN   = _fill_gap_vals(gaplst, Nreclst, dataN, Ngap, halfw)
                                    trE.data    = fillvalsE * mask + dataE
                                    trN.data    = fillvalsN * mask + dataN
                                    if np.any(np.isnan(trE.data)) or np.any(np.isnan(trN.data)):
                                        raise xcorrDataError('NaN EN DATA STATION: '+staid)
                                    if np.any(Nreclst<0):
                                        raise xcorrDataError('WRONG RECLST STATION: '+staid)
                                else:
                                    Nrec    = 0
                                st2.append(trE)
                                st2.append(trN)
                                isEN     = True
                            if Nrec > 0:
                                if not os.path.isdir(outdatedir):
                                    os.makedirs(outdatedir)
                                print ('!!! GAP EN STATION: '+staid)
                                with open(fnameE+'_rec', 'w') as fid:
                                    for i in range(Nrec):
                                        fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                                with open(fnameN+'_rec', 'w') as fid:
                                    for i in range(Nrec):
                                        fid.writelines(str(Nreclst[i, 0])+' '+str(Nreclst[i, 1])+'\n')
                if (not isZ) and (not isEN):
                    continue
                
                if not os.path.isdir(outdatedir):
                    os.makedirs(outdatedir)
                # remove trend, response
                if rmresp:
                    st2.detrend()
                    st2.remove_response(inventory = resp_inv, pre_filt = [f1, f2, f3, f4])
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
    
    
    
    
    
    
    
