# -*- coding: utf-8 -*-
"""
ASDF for obs data
    
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


class obsASDF(noisebase.baseASDF):
    """ Class for obs data 
    =================================================================================================================
    version history:
        2020/07/09
    =================================================================================================================
    """
    def tar_mseed_to_sac(self, datadir, outdir, start_date, end_date, sps=1., rmresp=False, 
            chan_rank=['H', 'B', 'L'], chanz = 'HZ', in_auxchan=['H1', 'H2', 'DH'], ntaper=2, halfw=100,\
            tb = 1., tlen = 86398., tb2 = 1000., tlen2 = 84000., perl = 5., perh = 200., pfx='LF_', \
            delete_tar=False, delete_extract=True, verbose=True, verbose2 = False):
        """Extract tared mseed files to SAC, designed for OBS data
        """
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
            raise ValueError('tb and tlen must both be multiplilier of target dt!')
        # Loop start
        print ('[%s] [TARMSEED2SAC] Extracting tar mseed from: ' %datetime.now().isoformat().split('.')[0]+datadir+' to '+outdir)
        while (curtime <= endtime):
            if verbose:
                print ('[%s] [TARMSEED2SAC] Date: ' %datetime.now().isoformat().split('.')[0]+curtime.date.isoformat())
            Nday        +=1
            Ndata       = 0
            Nobsdata    = 0
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
                raise ValueError('removed resp data should be in the range of raw data ')
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
                    # resample and "shift"
                    else:
                        # detrend the data to prevent edge effect when perform prefiltering before decimate
                        st[i].detrend()
                        dt          = st[i].stats.delta
                        # change dt
                        factor      = np.round(targetdt/dt)
                        if abs(factor*dt - targetdt) < min(dt, targetdt/1000.):
                            dt                  = targetdt/factor
                            st[i].stats.delta   = dt
                        else:
                            print(targetdt, dt)
                            raise ValueError('CHECK!' + staid)
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
                        st[i].filter(type = 'lowpass_cheby_2', freq = sps/2.) # prefilter
                        st[i].decimate(factor = int(factor), no_filter = True)
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
                if len(ipoplst) > 0:
                    print ('!!! poping traces!'+staid)
                    npop        = 0
                    for ipop in ipoplst:
                        st.pop(index = ipop - npop)
                        npop    += 1
                #====================================
                # Z component
                #====================================
                StreamZ     = st.select(channel='?'+chanz)
                channelZ    = None
                if len(StreamZ) == 0:
                    print ('*** NO Z COMPONENT STATION: '+staid)
                    Nnodata     += 1
                    continue
                for chtype in chan_rank:
                    tmpSTZ  = StreamZ.select(channel=chtype+chanz)
                    if len(tmpSTZ) > 0:
                        channelZ    = chtype + chanz
                        StreamZ     = tmpSTZ
                        break
                if channelZ is None:
                    raise ValueError('No expected channel type')
                StreamZ.sort(keys=['starttime', 'endtime'])
                StreamZ.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                # more than two traces with different locations, choose the longer one
                trZ             = StreamZ[0].copy()
                gapT            = max(0, trZ.stats.starttime - tbtime) + max(0, tetime - trZ.stats.endtime)
                if len(StreamZ) > 1:
                    for tmptr in StreamZ:
                        tmpgapT = max(0, tmptr.stats.starttime - tbtime) + max(0, tetime - tmptr.stats.endtime)
                        if tmpgapT < gapT:
                            gapT= tmpgapT
                            trZ = tmptr.copy()
                    if verbose2:
                        print ('!!! MORE Z LOCS STATION: '+staid+', CHOOSE: '+trZ.stats.location)
                if trZ.stats.starttime > tetime or trZ.stats.endtime < tbtime:
                    print ('!!! NO Z COMPONENT STATION: '+staid)
                    continue 
                else:
                    # trim the data for tb and tb+tlen
                    trZ.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                    if isinstance(trZ.data, np.ma.masked_array):
                        maskZ   = trZ.data.mask
                        dataZ   = trZ.data.data
                        sigstd  = trZ.data.std()
                        sigmean = trZ.data.mean()
                        if np.isnan(sigstd) or np.isnan(sigmean):
                            raise ValueError('NaN Z SIG/MEAN STATION: '+staid)
                        dataZ[maskZ]    = 0.
                        # gap list
                        gaparr, Ngap    = _xcorr_funcs._gap_lst(maskZ)
                        gaplst          = gaparr[:Ngap, :]
                        # get the rec list
                        Nrecarr, Nrec   = _xcorr_funcs._rec_lst(maskZ)
                        Nreclst         = Nrecarr[:Nrec, :]
                        if np.any(Nreclst<0) or np.any(gaplst<0):
                            raise ValueError('WRONG RECLST STATION: '+staid)
                        # values for gap filling
                        fillvals        = _xcorr_funcs._fill_gap_vals(gaplst, Nreclst, dataZ, Ngap, halfw)
                        trZ.data        = fillvals * maskZ + dataZ
                        if np.any(np.isnan(trZ.data)):
                            raise ValueError('NaN Z DATA STATION: '+staid)
                        # rec lst for tb2 and tlen2
                        im0             = int((tb2 - tb)/targetdt)
                        im1             = int((tb2 + tlen2 - tb)/targetdt) + 1
                        maskZ2          = maskZ[im0:im1]
                        Nrecarr2, Nrec2 = _xcorr_funcs._rec_lst(maskZ2)
                        Nreclst2        = Nrecarr2[:Nrec2, :]
                    else:
                        Nrec    = 0
                        Nrec2   = 0
                    fnameZ  = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)\
                            +'.'+staid+'.'+channelZ+'.SAC'
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
                # save Z component data
                if not os.path.isdir(outdatedir):
                    os.makedirs(outdatedir)
                if rmresp:
                    if tbtime2 < tbtime or tetime2 > tetime:
                        raise ValueError('removed resp should be in the range of raw data ')
                    trZ.detrend()
                    trZ.remove_response(inventory = resp_inv, pre_filt = [f1, f2, f3, f4])
                    trZ.trim(starttime = tbtime2, endtime = tetime2, pad = True, fill_value=0)
                    fnameZ  = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)\
                            +'.'+staid+'.'+channelZ+'.SAC'
                    sactrZ  = obspy.io.sac.SACTrace.from_obspy_trace(trZ)
                    sactrZ.write(fnameZ)
                else:
                    fnameZ  = outdatedir+'/'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)\
                            +'.'+staid+'.'+channelZ+'.SAC'
                    sactrZ  = obspy.io.sac.SACTrace.from_obspy_trace(trZ)
                    sactrZ.write(fnameZ)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sp              = obspy.io.xseed.Parser(datalessfname)
                    sp.write_resp(folder = outdatedir) 
                    locZ        = trZ.stats.location
                    respzlst    = glob.glob(outdatedir+'/RESP.'+staid+'*'+channelZ)
                    keepfname   = outdatedir+'/RESP.'+staid+'.'+locZ+'.'+channelZ
                    for respfname in respzlst:
                        if keepfname != respfname:
                            os.remove(respfname)
                Ndata   += 1
                #====================================
                # auxiliary components
                #====================================
                st_aux      = obspy.Stream()
                is_aux      = False
                is_all_exist= True
                auxchannels = []
                for auxch in in_auxchan:
                    auxST   = st.select(channel='?'+auxch)
                    if len(auxST) == 0:
                        if is_aux:
                            print ('!!! WARNING: not all auxilliary component exist, station: '+ staid)
                            is_all_exist    = False
                            break
                        continue
                    is_aux  = True
                    for chtype in chan_rank:
                        tmpST   = auxST.select(channel=chtype+auxch)
                        if len(tmpST) > 0:
                            auxchannels.append(chtype + auxch)
                            auxST       = tmpST
                            break
                    auxST.sort(keys=['starttime', 'endtime'])
                    auxST.merge(method = 1, interpolation_samples = ntaper, fill_value=None)
                    # # # auxST.merge(method = 1, interpolation_samples = ntaper, fill_value='interpolate')
                    # more than two traces with different locations, choose the longer one
                    tr_aux          = auxST[0].copy()
                    gapT            = max(0, tr_aux.stats.starttime - tbtime) + max(0, tetime - tr_aux.stats.endtime)
                    if len(auxST) > 1:
                        for tmptr in auxST:
                            tmpgapT = max(0, tmptr.stats.starttime - tbtime) + max(0, tetime - tmptr.stats.endtime)
                            if tmpgapT < gapT:
                                gapT    = tmpgapT
                                tr_aux  = tmptr.copy()
                        if verbose2:
                            print ('!!! MORE Z LOCS STATION: '+staid+', CHOOSE: '+tr_aux.stats.location)
                    if tr_aux.stats.starttime > tetime or tr_aux.stats.endtime < tbtime:
                        print ('!!! WARNING: not all auxilliary component exist, station: '+ staid)
                        is_all_exist    = False
                        break 
                    else:
                        # trim the data for tb and tb+tlen
                        tr_aux.trim(starttime = tbtime, endtime = tetime, pad = True, fill_value=None)
                        if isinstance(tr_aux.data, np.ma.masked_array):
                            mask    = tr_aux.data.mask
                            data    = tr_aux.data.data
                            sigstd  = tr_aux.data.std()
                            sigmean = tr_aux.data.mean()
                            if np.isnan(sigstd) or np.isnan(sigmean):
                                raise ValueError('NaN '+chtype+auxch+' SIG/MEAN STATION: '+staid)
                            data[mask]      = 0.
                            # gap list
                            gaparr, Ngap    = _xcorr_funcs._gap_lst(mask)
                            gaplst          = gaparr[:Ngap, :]
                            # get the rec list
                            Nrecarr, Nrec   = _xcorr_funcs._rec_lst(mask)
                            Nreclst         = Nrecarr[:Nrec, :]
                            if np.any(Nreclst<0) or np.any(gaplst<0):
                                raise ValueError('WRONG RECLST STATION: '+staid)
                            # values for gap filling
                            fillvals        = _xcorr_funcs._fill_gap_vals(gaplst, Nreclst, data, Ngap, halfw)
                            tr_aux.data     = fillvals * mask + data
                            if np.any(np.isnan(tr_aux.data)):
                                raise ValueError('NaN '+chtype+auxch+' DATA STATION: '+staid)
                    st_aux.append(tr_aux)
                if not (is_aux and is_all_exist):
                    continue # land stations
                if rmresp:
                    if tbtime2 < tbtime or tetime2 > tetime:
                        raise ValueError('removed resp should be in the range of raw data ')
                    st_aux.detrend()
                    st_aux.remove_response(inventory = resp_inv, pre_filt = [f1, f2, f3, f4])
                    st_aux.trim(starttime = tbtime2, endtime = tetime2, pad = True, fill_value=0)
                    for auxch in auxchannels:
                        fname   = outdatedir+'/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+\
                                    '.'+staid+'.'+auxch+'.SAC'
                        sactr   = obspy.io.sac.SACTrace.from_obspy_trace(st_aux.select(channel=auxch)[0])
                        sactr.write(fname)
                else:
                    for auxch in auxchannels:
                        fname   = outdatedir+'/'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'+str(curtime.day)+\
                                    '.'+staid+'.'+auxch+'.SAC'
                        tr_aux  = st_aux.select(channel=auxch)[0]
                        sactr   = obspy.io.sac.SACTrace.from_obspy_trace(tr_aux)
                        sactr.write(fname)
                        loc_aux     = tr_aux.stats.location
                        resplst     = glob.glob(outdatedir+'/RESP.'+staid+'*'+auxch)
                        keepfname   = outdatedir+'/RESP.'+staid+'.'+loc_aux+'.'+auxch
                        for respfname in resplst:
                            if keepfname != respfname:
                                os.remove(respfname)
                Nobsdata    += 1
            # End loop over stations
            curtime     += 86400
            if verbose:
                print ('[%s] [TARMSEED2SAC] %d/%d/%d groups of traces extracted!' %(datetime.now().isoformat().split('.')[0], Ndata, Nobsdata, Nnodata))
            # delete raw data
            if delete_extract:
                shutil.rmtree(datedir)
            if delete_tar:
                os.remove(tarlst[0])
        # End loop over dates
        print ('[%s] [TARMSEED2SAC] Extracted %d/%d days of data' %(datetime.now().isoformat().split('.')[0], Nday - Nnodataday, Nday))
        return
    
    def prep_tiltcomp_removal(self, datadir, outdir, start_date, end_date, sac_type = 1, copy_land = False,
            chan_rank=['H', 'B', 'L'], chanz = 'HZ', in_auxchan=['H1', 'H2', 'DH'], verbose=True):
        """prepare sac file list for tilt/compliance noise removal
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        fid_saclst  = open(outdir+'/tilt_compliance_sac.lst', 'w')
        starttime   = obspy.core.UTCDateTime(start_date)
        endtime     = obspy.core.UTCDateTime(end_date)
        curtime     = starttime
        Nnodataday  = 0
        Nday        = 0
        # Loop start
        print ('[%s] [PREP_TILT_COMPLIANCE] generate sac list: ' \
               %datetime.now().isoformat().split('.')[0]+ datadir +' to '+outdir)
        while (curtime <= endtime):
            if verbose:
                print ('[%s] [PREP_TILT_COMPLIANCE] Date: ' %datetime.now().isoformat().split('.')[0] + curtime.date.isoformat())
            Nday        +=1
            Nlanddata   = 0
            Nobsdata    = 0
            Nnodata     = 0
            daydir      = datadir + '/'+str(curtime.year)+'.'+ monthdict[curtime.month] + '/' \
                            +str(curtime.year)+'.'+monthdict[curtime.month]+'.'+str(curtime.day)
            if not os.path.isdir(daydir):
                print ('!!! NO DATA DATE: '+curtime.date.isoformat())
                curtime     += 86400
                Nnodataday  += 1
                continue
            outdaydir   = outdir + '/'+str(curtime.year)+'.'+ monthdict[curtime.month] + '/' \
                            +str(curtime.year)+'.'+monthdict[curtime.month]+'.'+str(curtime.day)
            if not os.path.isdir(outdaydir):
                os.makedirs(outdaydir)
            # loop over stations
            for staid in self.waveforms.list():
                netcode     = staid.split('.')[0]
                stacode     = staid.split('.')[1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos  = self.waveforms[staid].coordinates
                stla        = tmppos['latitude']
                stlo        = tmppos['longitude']
                water_depth = -tmppos['elevation_in_m']
                is_Z        = False
                # Z component
                for chtype in chan_rank:
                    fnameZ  = daydir + '/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'\
                                + str(curtime.day) +'.'+staid+'.'+ chtype + chanz +'.SAC'
                    if os.path.isfile(fnameZ):
                        channelZ= chtype + chanz
                        is_Z    = True
                        break
                if not is_Z:
                    Nnodata += 1
                    continue
                outfnameZ   = outdaydir + '/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'\
                                + str(curtime.day)+'.'+staid+'.'+ channelZ +'.SAC'
                # copy rec, rec2 files
                if os.path.isfile(fnameZ+'_rec'):
                    shutil.copyfile(src = fnameZ+'_rec', dst = outfnameZ+'_rec')
                if os.path.isfile(fnameZ+'_rec2'):
                    shutil.copyfile(src = fnameZ+'_rec2', dst = outfnameZ+'_rec2')
                # check H1, H2, DH components
                auxfilestr  = ''
                Naux        = 0
                for auxchan in in_auxchan:
                    for chtype in chan_rank:
                        fname   = daydir + '/ft_'+str(curtime.year)+'.'+ monthdict[curtime.month]+'.'\
                                    + str(curtime.day) +'.'+staid+'.'+ chtype + auxchan +'.SAC'
                        if os.path.isfile(fname):
                            auxfilestr  += '%s ' %fname
                            Naux        += 1
                            break
                if Naux == 3:
                    is_obs  = True
                elif Naux < 3:
                    is_obs  = False
                else:
                    raise ValueError('CHECK number of auxchan!')
                if is_obs:
                    outstr      = '%s ' %fnameZ
                    outstr      += auxfilestr
                    outstr      += '%d %g 0 ' %(sac_type, water_depth)
                    outstr      += '%s\n' %outfnameZ
                    fid_saclst.writelines(outstr)
                    Nobsdata    += 1
                else: # copy Z component file if not obs
                    if copy_land:
                        shutil.copyfile(src = fnameZ, dst = outfnameZ)
                        Nlanddata   += 1
                    else:
                        Nnodata     += 1
            # End loop over stations
            curtime     += 86400
            if verbose:
                print ('[%s] [PREP_TILT_COMPLIANCE] %d/%d/%d (obs/land/no) groups of traces extracted!'\
                       %(datetime.now().isoformat().split('.')[0], Nobsdata, Nlanddata, Nnodata))
        # End loop over dates
        fid_saclst.close()
        print ('[%s] [PREP_TILT_COMPLIANCE] Prepare %d/%d days of data'\
               %(datetime.now().isoformat().split('.')[0], Nday - Nnodataday, Nday))
        return

    def compute_xcorr(self, datadir, start_date, end_date, runtype=0, skipinv=True, chan_type=['H', 'B', 'L'], \
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
                                1   - skip if log file indicates SUCCESS & SKIPPED & NODATA
                                2   - skip if log file indicates SUCCESS
                                3   - skip if log file exists
                                4   - skip if montly/staid1 log directory exists
                                5   - skip if monthly log directory exists
        skipinv             - skip the month if not within the start/end date of the station inventory
        chan_type           - type of channels
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
                    #-------------------------------------------------------------------------------------
                    # append the station pair to xcorr list directly if the code will be run in parallel
                    #-------------------------------------------------------------------------------------
                    if parallel:
                        xcorr_lst.append(_xcorr_funcs.xcorr_pair(stacode1 = stacode1, netcode1=netcode1, stla1=stla1, stlo1=stlo1, \
                            stacode2=stacode2, netcode2=netcode2, stla2 = stla2, stlo2=stlo2, \
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
                        print ('[XCORR] subset:', isub, 'in', Nsub, 'sets')
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
    