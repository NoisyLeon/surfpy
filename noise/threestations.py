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
    
    def dw_aftan(self, datadir, prephdir, fskip, channel='ZZ', tb=0., outdir = None, inftan = pyaftan.InputFtanParam(),\
            basic1=True, basic2=True, pmf1=True, pmf2=True, verbose = True, f77=True, pfx='DISP', parallel = False, \
            nprocess=None, subsize=1000):
        """direct wave interferometry aftan
            fskip   -   0: overwrite
                        1: skip if success/nodata
                        2: skip upon log file existence
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
    
    # def interp_disp(self, data_type='C3DISPpmf2', channel='ZZ', pers=np.array([]), verbose=False):
    #     """ Interpolate dispersion curve for a given period array.
    #     =======================================================================================================
    #     ::: input parameters :::
    #     data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
    #     pers        - period array
    #     
    #     ::: output :::
    #     self.auxiliary_data.DISPbasic1interp, self.auxiliary_data.DISPbasic2interp,
    #     self.auxiliary_data.DISPpmf1interp, self.auxiliary_data.DISPpmf2interp
    #     =======================================================================================================
    #     """
    #     print ('[%s] [DW_FTAN_INTERP] start interpolating direct c3 aftan results' %datetime.now().isoformat().split('.')[0])
    #     if data_type=='C3DISPpmf2':
    #         ntype   = 6
    #     else:
    #         ntype   = 5
    #     if pers.size==0:
    #         pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
    #     staLst                      = self.waveforms.list()
    #     Nsta                        = len(staLst)
    #     Ntotal_traces               = Nsta*(Nsta-1)/2
    #     iinterp                     = 0
    #     Ntr_one_percent             = int(Ntotal_traces/100.)
    #     ipercent                    = 0
    #     for staid1 in staLst:
    #         for staid2 in staLst:
    #             netcode1, stacode1  = staid1.split('.')
    #             netcode2, stacode2  = staid2.split('.')
    #             if staid1 >= staid2:
    #                 continue
    #             iinterp             += 1
    #             if np.fmod(iinterp, Ntr_one_percent) ==0:
    #                 ipercent        += 1
    #                 print ('[%s] [FTAN_INTERP] Number of traces finished: ' %datetime.now().isoformat().split('.')[0]+\
    #                                 str(iinterp)+'/'+str(Ntotal_traces)+' '+str(ipercent)+'%')
    #             # get the data
    #             try:
    #                 subdset         = self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
    #             except KeyError:
    #                 continue
    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("ignore")
    #                 data            = subdset.data.value
    #                 index           = subdset.parameters
    #             if verbose:
    #                 print ('--- interpolating dispersion curve for '+ staid1+'_'+staid2+'_'+channel)
    #             outindex            = { 'To': 0, 'U': 1, 'C': 2,  'amp': 3, 'snr': 4, 'inbound': 5, 'Np': pers.size }
    #             Np                  = int(index['Np'])
    #             if Np < 5:
    #                 # if verbose:
    #                     # warnings.warn('Not enough datapoints for: '+ staid1+'_'+staid2+'_'+channel, UserWarning, stacklevel=1)
    #                 print ('*** WARNING: Not enough datapoints for: '+ staid1+'_'+staid2+'_'+channel)
    #                 continue
    #             # interpolation
    #             obsT                = data[index['To']][:Np]
    #             U                   = np.interp(pers, obsT, data[index['U']][:Np] )
    #             C                   = np.interp(pers, obsT, data[index['C']][:Np] )
    #             amp                 = np.interp(pers, obsT, data[index['amp']][:Np] )
    #             inbound             = (pers > obsT[0])*(pers < obsT[-1])*1
    #             # store interpolated data to interpdata array
    #             interpdata          = np.append(pers, U)
    #             interpdata          = np.append(interpdata, C)
    #             interpdata          = np.append(interpdata, amp)
    #             if data_type is 'DISPpmf2':
    #                 snr             = np.interp(pers, obsT, data[index['snr']][:Np] )
    #                 interpdata      = np.append(interpdata, snr)
    #             interpdata          = np.append(interpdata, inbound)
    #             interpdata          = interpdata.reshape(ntype, pers.size)
    #             staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
    #             self.add_auxiliary_data(data=interpdata, data_type=data_type+'interp', path=staid_aux, parameters=outindex)
    #     print ('[%s] [FTAN_INTERP] aftan interpolation all done' %datetime.now().isoformat().split('.')[0])
    #     return
    
    