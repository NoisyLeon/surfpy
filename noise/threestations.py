# -*- coding: utf-8 -*-
"""
ASDF for three station interferometry
    
"""
try:
    import surfpy.noise.noisebase as noisebase
except:
    import noisebase

import surfpy.noise._c3_funcs as _c3_funcs
import surfpy.aftan.pyaftan as pyaftan

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
    

class tripleASDF(noisebase.baseASDF):
    
    def dw_interfere(self, datadir, outdir = None, channel='ZZ', chan_types=['LH', 'BH', 'HH'], \
            alpha = 0.01, vmin = 1., vmax = 5., Tmin = 5., Tmax = 150., bfact_dw = 1., efact_dw = 1., dthresh = 5., \
            parallel=False, nprocess=None, subsize=1000, verbose=True, verbose2=False):
        """
        compute three station direct wave interferometry
        =================================================================================================================
        ::: input parameters :::
        datadir             - directory including data and output
        startdate/enddate   - start/end date for computation
        
        channel             - channel 
        chan_types          - types (also used as priorities) of channels, used for hybrid channel xcorr
        
        parallel            - run the xcorr parallelly or not
        nprocess            - number of processes
        subsize             - subsize of processing list, use to prevent lock in multiprocessing process
        =================================================================================================================
        """
        StationInv          = obspy.Inventory()        
        for staid in self.waveforms.list():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                StationInv  += self.waveforms[staid].StationXML
        if outdir is None:
            outdir  = datadir 
        #---------------------------------
        # prepare data
        #---------------------------------
        print ('[%s] [DW_INTERFERE] preparing for three station direct wave interferometry' %datetime.now().isoformat().split('.')[0])
        c3_lst  = []
        for staid1 in self.waveforms.list():
            netcode1, stacode1      = staid1.split('.')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos1         = self.waveforms[staid1].coordinates
                stla1           = tmppos1['latitude']
                stlo1           = tmppos1['longitude']
            for staid2 in self.waveforms.list():
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos2         = self.waveforms[staid2].coordinates
                    stla2           = tmppos2['latitude']
                    stlo2           = tmppos2['longitude']
                c3_lst.append(_c3_funcs.c3_pair(datadir = datadir, outdir = outdir, stacode1 = stacode1, netcode1 = netcode1,\
                    stla1 = stla1, stlo1 = stlo1,  stacode2 = stacode2, netcode2 = netcode2, stla2 = stla2, stlo2 = stlo2,\
                    channel = channel, chan_types = chan_types, StationInv = StationInv, alpha = alpha, vmin = vmin, vmax = vmax, Tmin = Tmin, Tmax = Tmax, \
                    bfact_dw = bfact_dw, efact_dw = efact_dw, dthresh = dthresh))
        #===============================
        # direct wave interferometry
        #===============================
        print ('[%s] [DW_INTERFERE] computating... ' %datetime.now().isoformat().split('.')[0] )
        # parallelized run
        if parallel:
            #-----------------------------------------
            # Computing xcorr with multiprocessing
            #-----------------------------------------
            if len(c3_lst) > subsize:
                Nsub            = int(len(c3_lst)/subsize)
                for isub in range(Nsub):
                    print ('[%s] [DW_INTERFERE] subset:' %datetime.now().isoformat().split('.')[0], isub, 'in', Nsub, 'sets')
                    cur_c3Lst   = c3_lst[isub*subsize:(isub+1)*subsize]
                    CCUBE       = partial(_c3_funcs.direct_wave_interfere_for_mp, verbose = verbose, verbose2 = verbose2)
                    pool        = multiprocessing.Pool(processes=nprocess)
                    pool.map(CCUBE, cur_c3Lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cur_c3Lst       = c3_lst[(isub+1)*subsize:]
                CCUBE           = partial(_c3_funcs.direct_wave_interfere_for_mp, verbose = verbose, verbose2 = verbose2)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(CCUBE, cur_c3Lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                CCUBE           = partial(_c3_funcs.direct_wave_interfere_for_mp, verbose = verbose, verbose2 = verbose2)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(CCUBE, c3_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
        else:
            Nsuccess    = 0
            Nnodata     = 0
            for ilst in range(len(c3_lst)):
                if c3_lst[ilst].direct_wave_interfere(verbose = verbose, verbose2 = verbose2) > 0:
                    Nsuccess+= 1
                else:
                    Nnodata += 1
            print ('[%s] [DW_INTERFERE] computation ALL done: success/nodata: %d/%d' %(datetime.now().isoformat().split('.')[0], Nsuccess, Nnodata))
        return
    
    def dw_aftan(self, datadir, prephdir, fskip = 0, channel='ZZ', tb=0., outdir = None, inftan = pyaftan.InputFtanParam(),\
            basic1=True, basic2=True, pmf1=True, pmf2=True, verbose = True, f77=True, pfx='DISP', parallel = False, \
            nprocess=None, subsize=1000):
        """direct wave interferometry aftan
        =================================================================================================================
            fskip   -   0: overwrite
                        1: skip if success/nodata
                        2: skip upon log file existence
                        -1: debug purpose
        =================================================================================================================
        """
        if outdir is None:
            outdir  = datadir
        #---------------------------------
        # prepare data
        #---------------------------------
        print ('[%s] [DW_AFTAN] preparing for three station direct wave aftan' %datetime.now().isoformat().split('.')[0])
        c3_lst  = []
        for staid1 in self.waveforms.list():
            netcode1, stacode1      = staid1.split('.')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos1         = self.waveforms[staid1].coordinates
                stla1           = tmppos1['latitude']
                stlo1           = tmppos1['longitude']
            for staid2 in self.waveforms.list():
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                # # # # if stacode1 != 'MONP' or stacode2 != 'R12A':
                # # # #     continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos2         = self.waveforms[staid2].coordinates
                    stla2           = tmppos2['latitude']
                    stlo2           = tmppos2['longitude']
                # skip or not
                logfname    = datadir + '/logs_dw_aftan/'+staid1+'/'+staid1+'_'+staid2+'.log'
                if os.path.isfile(logfname):
                    if fskip == 2:
                        continue
                    with open(logfname, 'r') as fid:
                        logflag = fid.readlines()[0].split()[0]
                    if (logflag == 'SUCCESS' or logflag == 'NODATA') and fskip == 1:
                        continue
                    if (logflag != 'FAILED') and fskip == -1: # debug
                        continue
                else:
                    if fskip == -1:
                        continue
                c3_lst.append(_c3_funcs.c3_pair(datadir = datadir, outdir = outdir, stacode1 = stacode1, netcode1 = netcode1,\
                    stla1 = stla1, stlo1 = stlo1,  stacode2 = stacode2, netcode2 = netcode2, stla2 = stla2, stlo2 = stlo2,\
                    channel = channel, inftan = inftan, basic1=basic1, basic2=basic2, pmf1=pmf1, pmf2=pmf2, f77=f77, prephdir = prephdir))
        #===============================
        # direct wave interferometry
        #===============================
        print ('[%s] [DW_AFTAN] computating... ' %datetime.now().isoformat().split('.')[0] )
        # parallelized run
        if parallel:
            #-----------------------------------------
            # Computing xcorr with multiprocessing
            #-----------------------------------------
            if len(c3_lst) > subsize:
                Nsub            = int(len(c3_lst)/subsize)
                for isub in range(Nsub):
                    print ('[%s] [DW_AFTAN] subset:' %datetime.now().isoformat().split('.')[0], isub, 'in', Nsub, 'sets')
                    cur_c3Lst   = c3_lst[isub*subsize:(isub+1)*subsize]
                    AFTAN       = partial(_c3_funcs.direct_wave_aftan_for_mp, verbose = verbose)
                    pool        = multiprocessing.Pool(processes=nprocess)
                    pool.map(AFTAN, cur_c3Lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cur_c3Lst       = c3_lst[(isub+1)*subsize:]
                AFTAN           = partial(_c3_funcs.direct_wave_aftan_for_mp, verbose = verbose)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN, cur_c3Lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                AFTAN           = partial(_c3_funcs.direct_wave_aftan_for_mp, verbose = verbose)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN, c3_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
        else:
            Nsuccess    = 0
            Nnodata     = 0
            for ilst in range(len(c3_lst)):
                c3_lst[ilst].direct_wave_aftan(verbose = verbose) 
            # print ('[%s] [DW_AFTAN] computation ALL done: success/nodata: %d/%d' %(datetime.now().isoformat().split('.')[0], Nsuccess, Nnodata))
        return
    
    def dw_interp_disp(self, datadir, prephdir, fskip = 0, channel='ZZ', tb=0., outdir = None, inftan = pyaftan.InputFtanParam(),\
            basic1=True, basic2=True, pmf1=True, pmf2=True, verbose = True, f77=True, pfx='DISP', parallel = False, \
            nprocess=None, subsize=1000):
        """direct wave interferometry aftan
        =================================================================================================================
            fskip   -   0: overwrite
                        1: skip if success/nodata
                        2: skip upon log file existence
                        -1: debug purpose
        =================================================================================================================
        """
        if outdir is None:
            outdir  = datadir
        #---------------------------------
        # prepare data
        #---------------------------------
        print ('[%s] [DW_AFTAN] preparing for three station direct wave aftan' %datetime.now().isoformat().split('.')[0])
        c3_lst  = []
        for staid1 in self.waveforms.list():
            netcode1, stacode1      = staid1.split('.')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos1         = self.waveforms[staid1].coordinates
                stla1           = tmppos1['latitude']
                stlo1           = tmppos1['longitude']
            for staid2 in self.waveforms.list():
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos2         = self.waveforms[staid2].coordinates
                    stla2           = tmppos2['latitude']
                    stlo2           = tmppos2['longitude']
                # skip or not
                logfname    = datadir + '/logs_dw_aftan/'+staid1+'/'+staid1+'_'+staid2+'.log'
                if os.path.isfile(logfname):
                    if fskip == 2:
                        continue
                    with open(logfname, 'r') as fid:
                        logflag = fid.readlines()[0].split()[0]
                    if (logflag == 'SUCCESS' or logflag == 'NODATA') and fskip == 1:
                        continue
                    if (logflag != 'FAILED') and fskip == -1: # debug
                        continue
                else:
                    if fskip == -1:
                        continue
                c3_lst.append(_c3_funcs.c3_pair(datadir = datadir, outdir = outdir, stacode1 = stacode1, netcode1 = netcode1,\
                    stla1 = stla1, stlo1 = stlo1,  stacode2 = stacode2, netcode2 = netcode2, stla2 = stla2, stlo2 = stlo2,\
                    channel = channel, inftan = inftan, basic1=basic1, basic2=basic2, pmf1=pmf1, pmf2=pmf2, f77=f77, prephdir = prephdir))
        #===============================
        # direct wave interferometry
        #===============================
        print ('[%s] [DW_AFTAN] computating... ' %datetime.now().isoformat().split('.')[0] )
        # parallelized run
        if parallel:
            #-----------------------------------------
            # Computing xcorr with multiprocessing
            #-----------------------------------------
            if len(c3_lst) > subsize:
                Nsub            = int(len(c3_lst)/subsize)
                for isub in range(Nsub):
                    print ('[%s] [DW_AFTAN] subset:' %datetime.now().isoformat().split('.')[0], isub, 'in', Nsub, 'sets')
                    cur_c3Lst   = c3_lst[isub*subsize:(isub+1)*subsize]
                    AFTAN       = partial(_c3_funcs.direct_wave_aftan_for_mp, verbose = verbose)
                    pool        = multiprocessing.Pool(processes=nprocess)
                    pool.map(AFTAN, cur_c3Lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cur_c3Lst       = c3_lst[(isub+1)*subsize:]
                AFTAN           = partial(_c3_funcs.direct_wave_aftan_for_mp, verbose = verbose)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN, cur_c3Lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                AFTAN           = partial(_c3_funcs.direct_wave_aftan_for_mp, verbose = verbose)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN, c3_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
        else:
            Nsuccess    = 0
            Nnodata     = 0
            for ilst in range(len(c3_lst)):
                c3_lst[ilst].direct_wave_aftan(verbose = verbose) 
            # print ('[%s] [DW_AFTAN] computation ALL done: success/nodata: %d/%d' %(datetime.now().isoformat().split('.')[0], Nsuccess, Nnodata))
        return 
    
    def dw_stack_disp(self, datadir, outdir = None, fskip = 0, data_type = 'C3DISPpmf2', channel = 'ZZ', pers = [],\
                      snr_thresh = 10., Ntrace_min = 5, nfmin = 5, jump_thresh = 3., isrun = True, verbose = False):
        """ stack dispersion results
        =======================================================================================================
        ::: input parameters :::
        data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        pers        - period array
        
        ::: output :::
        self.auxiliary_data.C3DISPpmf2
        =======================================================================================================
        """
        if outdir is None:
            outdir  = datadir
        print ('[%s] [DW_STACK_DISP] start stacking direct c3 aftan results' %datetime.now().isoformat().split('.')[0])
        if len(pers) == 0:
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        if isrun:
            c3_lst  = []
            for staid1 in self.waveforms.list():
                netcode1, stacode1      = staid1.split('.')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos1         = self.waveforms[staid1].coordinates
                    stla1           = tmppos1['latitude']
                    stlo1           = tmppos1['longitude']
                for staid2 in self.waveforms.list():
                    netcode2, stacode2  = staid2.split('.')
                    if staid1 >= staid2:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tmppos2         = self.waveforms[staid2].coordinates
                        stla2           = tmppos2['latitude']
                        stlo2           = tmppos2['longitude']
                    # skip or not
                    logfname    = datadir + '/logs_dw_stack_disp/'+staid1+'/'+staid1+'_'+staid2+'.log'
                    if os.path.isfile(logfname):
                        if fskip == 2:
                            continue
                        with open(logfname, 'r') as fid:
                            logflag = fid.readlines()[0].split()[0]
                        if (logflag == 'SUCCESS' or logflag == 'NODATA') and fskip == 1:
                            continue
                        if (logflag != 'FAILED') and fskip == -1: # debug
                            continue
                    else:
                        if fskip == -1:
                            continue
                    c3_lst.append(_c3_funcs.c3_pair(datadir = datadir, outdir = outdir, stacode1 = stacode1, netcode1 = netcode1,\
                        stla1 = stla1, stlo1 = stlo1,  stacode2 = stacode2, netcode2 = netcode2, stla2 = stla2, stlo2 = stlo2,\
                        channel = channel, snr_thresh = snr_thresh, Ntrace_min = Ntrace_min, nfmin = nfmin, jump_thresh = jump_thresh))
            #===============================
            # direct wave interferometry
            #===============================
            print ('[%s] [DW_STACK_DISP] computating... ' %datetime.now().isoformat().split('.')[0] )
            # parallelized run
            if parallel:
                #-----------------------------------------
                # Computing xcorr with multiprocessing
                #-----------------------------------------
                if len(c3_lst) > subsize:
                    Nsub            = int(len(c3_lst)/subsize)
                    for isub in range(Nsub):
                        print ('[%s] [DW_STACK_DISP] subset:' %datetime.now().isoformat().split('.')[0], isub, 'in', Nsub, 'sets')
                        cur_c3Lst   = c3_lst[isub*subsize:(isub+1)*subsize]
                        AFTAN       = partial(_c3_funcs.direct_wave_stack_disp_for_mp, verbose = verbose)
                        pool        = multiprocessing.Pool(processes=nprocess)
                        pool.map(AFTAN, cur_c3Lst) #make our results with a map call
                        pool.close() #we are not adding any more processes
                        pool.join() #tell it to wait until all threads are done before going on
                    cur_c3Lst       = c3_lst[(isub+1)*subsize:]
                    AFTAN           = partial(_c3_funcs.direct_wave_stack_disp_for_mp, verbose = verbose)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(AFTAN, cur_c3Lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                else:
                    AFTAN           = partial(_c3_funcs.direct_wave_stack_disp_for_mp, verbose = verbose)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(AFTAN, c3_lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
            else:
                for ilst in range(len(c3_lst)):
                    c3_lst[ilst].direct_wave_stack_disp(verbose = verbose)     
        # load data 
        for staid1 in self.waveforms.list():
            netcode1, stacode1  = staid1.split('.')
            for staid2 in self.waveforms.list():    
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                infname     = self.outdir + '/DW_DISP/'+staid1 + '/DISP_'+staid1+'_'+channel[0]+'_'+staid2+'_'+channel[1]+'.npz'
                if not os.path.isfile(infname):
                    continue
                inarr       = np.load(infname)
                pers        = inarr['arr_0']
                C           = inarr['arr_1']
                un          = inarr['arr_2']
                snr         = inarr['arr_3']
                Nm          = inarr['arr_4']
                index       = inarr['arr_5']
                #
                outindex    = { 'To': 0, 'C': 1,  'un': 2, 'snr': 3, 'Nm': 4, 'index': 5, 'Np': pers.size }
                if pers.size < nfmin:
                    print ('*** WARNING: Not enough datapoints for: '+ staid1+'_'+staid2+'_'+channel)
                    continue
                # store interpolated data to interpdata array
                outdata     = np.append(pers, C)
                outdata     = np.append(outdata, un)
                outdata     = np.append(outdata, snr)
                outdata     = np.append(outdata, Nm)
                outdata     = np.append(outdata, index)
                outdata     = outdata.reshape(6, pers.size)
                staid_aux   = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                self.add_auxiliary_data(data = outdata, data_type = data_type, path=staid_aux, parameters = outindex)
        print ('[%s] [DW_STACK_DISP] all done' %datetime.now().isoformat().split('.')[0])
        return
    
    