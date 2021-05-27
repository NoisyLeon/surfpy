# -*- coding: utf-8 -*-
"""
ASDF for three station interferometry
    
"""
import surfpy.noise.noisebase as noisebase
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
    
c3_header_default       = {'netcode1': '', 'stacode1': '', 'netcode2': '', 'stacode2': '', 'chan1': '', 'chan2': '',
        'npts': 12345, 'b': 12345, 'e': 12345, 'delta': 12345, 'dist': 12345, 'az': 12345, 'baz': 12345, 'stacktrace': 0}

class tripleASDF(noisebase.baseASDF):
    
    def dw_interfere(self, datadir, outdir = None, fskip = 0, networks= [], channel='ZZ', chan_types=['LH', 'BH', 'HH'], \
            alpha = 0.01, dthresh = 5., parallel=False, nprocess=None, subsize=1000, verbose=True, verbose2=False):
        """
        compute three station direct wave interferometry
        =================================================================================================================
        ::: input parameters :::
        datadir     - directory including data
        outdir      - output directory
        channel     - channel 
        chan_types  - types (also used as priorities) of channels (NOT implemented yet)
        alpha       - threshhold percentage to determine stationary phase zone
        dtresh      - threshhold difference in distance ( abs( abs(del_d_hyp) - del_d_ell) > dtresh )
        parallel    - run the xcorr parallelly or not
        nprocess    - number of processes
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
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
                if len(networks) > 0:
                    if (not (netcode1 in networks)) and (not (netcode2 in networks)):
                        continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos2         = self.waveforms[staid2].coordinates
                    stla2           = tmppos2['latitude']
                    stlo2           = tmppos2['longitude']
                # skip or not
                logfname    = datadir + '/logs_dw_interfere/'+staid1+'/'+staid1+'_'+staid2+'.log'
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
                    channel = channel, chan_types = chan_types, StationInv = StationInv, alpha = alpha, \
                    dthresh = dthresh))
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
    
    def dw_aftan(self, datadir, prephdir, fskip = 0, networks= [], channel='ZZ', outdir = None, inftan = pyaftan.InputFtanParam(),\
            basic1=True, basic2=True, pmf1=True, pmf2=True, verbose = True, f77=True, pfx='DISP', parallel = False, \
            nprocess=None, subsize=1000):
        """direct wave interferometry aftan
        =================================================================================================================
        datadir     - directory including data
        prephdir    - directory for predicted phase velocity dispersion curve
        fskip       - skip upon dispersion output existence
                        0: overwrite
                        1: skip if success/nodata
                        2: skip upon log file existence
                        -1: debug purpose
        channel     - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        outdir      - directory for output disp txt files (default = None, no txt output)
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        parallel    - run the xcorr parallelly or not
        nprocess    - number of processes
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
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
                if len(networks) > 0:
                    if (not (netcode1 in networks)) and (not (netcode2 in networks)):
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
        print ('[%s] [DW_AFTAN] computation ALL done' %datetime.now().isoformat().split('.')[0])
        return
    
    def dw_stack_disp(self, datadir, outdir = None, fskip = 0, networks= [], channel = 'ZZ', pers = [], vmin = 1.0, vmax = 4.5,
            snr_thresh = 10., Ntrace_min = 5, nfmin = 5, jump_thresh = 3., parallel = False, \
            nprocess=None, subsize=1000, verbose = False):
        """ stack dispersion results
        =======================================================================================================
        ::: input parameters :::
        datadir     - directory including data
        outdir      - output directory
        fskip       - skip upon dispersion output existence
                        0: overwrite
                        1: skip if success/nodata
                        2: skip upon log file existence
                        -1: debug purpose
        channel     - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        pers        - periods for stacking
        snr_thresh  - threshold SNR
        Ntrace_min  - minimum number of traces
        nfmin       - minimum frequency (period) data points for each trace
        jump_thresh - threshold value for jump detection
        
        parallel    - run the xcorr parallelly or not
        nprocess    - number of processes
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
        =======================================================================================================
        """
        if outdir is None:
            outdir  = datadir
        print ('[%s] [DW_STACK_DISP] start stacking direct c3 aftan results' %datetime.now().isoformat().split('.')[0])
        if len(pers) == 0:
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
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
                if len(networks) > 0:
                    if (not (netcode1 in networks)) and (not (netcode2 in networks)):
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
                    stla1 = stla1, stlo1 = stlo1,  stacode2 = stacode2, netcode2 = netcode2, stla2 = stla2, stlo2 = stlo2, \
                    vmin = vmin, vmax = vmax, channel = channel, snr_thresh = snr_thresh, Ntrace_min = Ntrace_min,\
                    nfmin = nfmin, jump_thresh = jump_thresh))
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
                    AFTAN_STACK = partial(_c3_funcs.direct_wave_stack_disp_for_mp, verbose = verbose)
                    pool        = multiprocessing.Pool(processes=nprocess)
                    pool.map(AFTAN_STACK, cur_c3Lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cur_c3Lst       = c3_lst[(isub+1)*subsize:]
                AFTAN_STACK     = partial(_c3_funcs.direct_wave_stack_disp_for_mp, verbose = verbose)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN_STACK, cur_c3Lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                AFTAN_STACK     = partial(_c3_funcs.direct_wave_stack_disp_for_mp, verbose = verbose)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN_STACK, c3_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
        else:
            for ilst in range(len(c3_lst)):
                c3_lst[ilst].direct_wave_stack_disp(verbose = verbose)     
        print ('[%s] [DW_STACK_DISP] all done' %datetime.now().isoformat().split('.')[0])
        return
    
    def dw_stack(self, datadir, outdir = None, fskip=0, networks= [], channel='ZZ', vmin = 1., vmax = 5., Tmin = 5., Tmax = 150.,\
            prefer_c3_disp = True,  bfact_dw = 1., efact_dw = 1., snr_thresh = 10., ftan_type = 'DISPpmf2',\
            parallel = False,  nprocess=None, subsize=1000, verbose = True):
        """ stack direct wave interferogram
        =======================================================================================================
        ::: input parameters :::
        datadir         - directory including data
        outdir          - output directory
        channel         - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        vmin/vmax       - min/max group velocity to find max amplitude
        Tmin/Tmax       - min/max periods
        pers            - periods for stacking
        snr_thresh      - threshold SNR
        use_xcorr_aftan - use aftan results from xcorr as reference curve or not
        ftan_type       - ftan type of xcorr
        =======================================================================================================
        """
        if outdir is None:
            outdir  = datadir
        #---------------------------------
        # prepare data
        #---------------------------------
        print ('[%s] [DW_STACK] preparing C3 stack' %datetime.now().isoformat().split('.')[0])
        chan1                   = 'C3'+channel[0]
        chan2                   = 'C3'+channel[1]
        c3_lst                  = []
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
                if len(networks) > 0:
                    if (not (netcode1 in networks)) and (not (netcode2 in networks)):
                        continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos2         = self.waveforms[staid2].coordinates
                    stla2           = tmppos2['latitude']
                    stlo2           = tmppos2['longitude']
                # skip or not
                logfname    = datadir + '/logs_dw_stack/'+staid1+'/'+staid1+'_'+staid2+'.log'
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
                try:
                    subdset     = self.auxiliary_data[ftan_type][netcode1][stacode1][netcode2][stacode2][channel]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data        = subdset.data[()]
                        index       = subdset.parameters
                    Np              = int(index['Np'])
                    # reference curves
                    pers_ref        = data[index['To']][:Np]
                    phvel_ref       = data[index['C']][:Np]
                    if Np < 5:
                        print ('*** WARNING: Not enough I2 datapoints for: '+ staid1+'_'+staid2+'_'+channel)
                        phvel_ref   = []
                        pers_ref    = []
                except KeyError:
                    phvel_ref   = []
                    pers_ref    = []
                    # print ('!!! no reference phase velocity from I2: '+staid1+'_'+staid2)
                temp_c3_pair        = _c3_funcs.c3_pair(datadir = datadir, outdir = outdir, stacode1 = stacode1, netcode1 = netcode1,\
                    stla1 = stla1, stlo1 = stlo1,  stacode2 = stacode2, netcode2 = netcode2, stla2 = stla2, stlo2 = stlo2,\
                    channel = channel, vmin = vmin, vmax = vmax, Tmin = Tmin, Tmax = Tmax,  bfact_dw = bfact_dw, efact_dw = efact_dw,\
                    phvel_ref = phvel_ref, pers_ref = pers_ref, prefer_c3_disp = prefer_c3_disp)
                c3_lst.append(temp_c3_pair)
        #===============================
        # phase shift stack
        #===============================
        print ('[%s] [DW_STACK] computating... ' %datetime.now().isoformat().split('.')[0] )
        # parallelized run
        if parallel:
            #-----------------------------------------
            # Computing xcorr with multiprocessing
            #-----------------------------------------
            if len(c3_lst) > subsize:
                Nsub            = int(len(c3_lst)/subsize)
                for isub in range(Nsub):
                    print ('[%s] [DW_STACK] subset:' %datetime.now().isoformat().split('.')[0], isub, 'in', Nsub, 'sets')
                    cur_c3Lst   = c3_lst[isub*subsize:(isub+1)*subsize]
                    DW_STACK    = partial(_c3_funcs.direct_wave_phase_shift_stack_for_mp, verbose = verbose)
                    pool        = multiprocessing.Pool(processes=nprocess)
                    pool.map(DW_STACK, cur_c3Lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cur_c3Lst       = c3_lst[(isub+1)*subsize:]
                DW_STACK        = partial(_c3_funcs.direct_wave_phase_shift_stack_for_mp, verbose = verbose)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(DW_STACK, cur_c3Lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                DW_STACK        = partial(_c3_funcs.direct_wave_phase_shift_stack_for_mp, verbose = verbose)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(DW_STACK, c3_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
        else:
            for ilst in range(len(c3_lst)):
                c3_lst[ilst].direct_wave_phase_shift_stack(verbose = verbose)     
        print ('[%s] [DW_STACK] ALL done' %datetime.now().isoformat().split('.')[0])
        return 
         
                
