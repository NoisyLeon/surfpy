# -*- coding: utf-8 -*-
"""
ASDF for three station interferometry
    
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
    

class tripleASDF(noisebase.baseASDF):
    
    def makesym(self, channel='ZZ'):
        """ aftan analysis of cross-correlation data 
        =======================================================================================
        ::: input parameters :::
        channel     - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        tb          - begin time (default = 0.0)
        outdir      - directory for output disp txt files (default = None, no txt output)
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        prephdir    - directory for predicted phase velocity dispersion curve
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        ---------------------------------------------------------------------------------------
        ::: output :::
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        # print ('[%s] [AFTAN] start aftan analysis' %datetime.now().isoformat().split('.')[0])
        StationInv          = obspy.Inventory()
        for staid in self.waveforms.list():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                StationInv  += self.waveforms[staid].StationXML
        for network1 in StationInv:
            for station1 in network1:
                staid1  = network1.code+'.'+station1.code
                lat1    = station1.latitude
                lon1    = station1.longitude
                print (staid1)
                for network2 in StationInv:
                    for station2 in network2:
                        staid2  = network2.code+'.'+station2.code
                        lat2    = station2.latitude
                        lon2    = station2.longitude
        # 
        # 
        # Nsta                        = len(staLst)
        # Ntotal_traces               = Nsta*(Nsta-1)/2
        # iaftan                      = 0
        # Ntr_one_percent             = int(Ntotal_traces/100.)
        # ipercent                    = 0
        # for staid1 in self.waveforms.list():
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")
        #         tmppos1     = self.waveforms[staid1].coordinates
        #         lat1        = tmppos1['latitude']
        #         lon1        = tmppos1['longitude']
        #         elv1        = tmppos1['elevation_in_m']
        #     netcode1, stacode1  = staid1.split('.')
        #     print (staid1)
        #     for staid2 in self.waveforms.list():
        #         netcode2, stacode2  = staid2.split('.')
        #         if staid1 >= staid2:
        #             continue
        #         with warnings.catch_warnings():
        #             warnings.simplefilter("ignore")
        #             tmppos2     = self.waveforms[staid2].coordinates
        #             lat2        = tmppos2['latitude']
        #             lon2        = tmppos2['longitude']
        #             elv2        = tmppos2['elevation_in_m']
                    
                # print how many traces has been processed
        #         iaftan              += 1
        #         if np.fmod(iaftan, Ntr_one_percent) ==0:
        #             ipercent        += 1
        #             print ('[%s] [AFTAN] Number of traces finished : ' %datetime.now().isoformat().split('.')[0] \
        #                    +str(iaftan)+'/'+str(Ntotal_traces)+' '+str(ipercent)+'%')
        #         # determine channels
        #         try:
        #             channels1       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
        #             for chan in channels1:
        #                 if chan[-1] == channel[0]:
        #                     chan1   = chan
        #                     break
        #             channels2       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1].list()
        #             for chan in channels2:
        #                 if chan[-1] == channel[1]:
        #                     chan2   = chan
        #                     break
        #         except KeyError:
        #             continue
        #         # get data
        #         tr                  = self.get_xcorr_trace(netcode1, stacode1, netcode2, stacode2, chan1, chan2)
        #         if tr is None:
        #             print ('*** WARNING: '+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+' not exists!')
        #             continue
        #         aftanTr             = pyaftan.aftantrace(tr.data, tr.stats)
        #         if abs(aftanTr.stats.sac.b + aftanTr.stats.sac.e) < aftanTr.stats.delta:
        #             aftanTr.makesym()
        #         else:
        #             print ('*** WARNING: '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel+' NOT symmetric')
        #             continue
        #         phvelname           = prephdir + "/%s.%s.pre" %(netcode1+'.'+stacode1, netcode2+'.'+stacode2)
        #         if not os.path.isfile(phvelname):
        #             print ('*** WARNING: '+ phvelname+' not exists!')
        #             continue
        #         # aftan analysis
        #         if f77:
        #             aftanTr.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
        #                 tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
        #                     npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
        #         else:
        #             aftanTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
        #                 tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
        #                     npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
        #         if verbose:
        #             print ('--- aftan analysis for: ' + netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel)
        #         # SNR
        #         aftanTr.get_snr(ffact = inftan.ffact) 
        #         staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
        #         #=====================================
        #         # save aftan results to ASDF dataset
        #         #=====================================
        #         if basic1:
        #             parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6,\
        #                                     'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_1}
        #             self.add_auxiliary_data(data=aftanTr.ftanparam.arr1_1, data_type='DISPbasic1', path=staid_aux,\
        #                                     parameters=parameters)
        #         if basic2:
        #             parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6,\
        #                                     'amp': 7, 'Np': aftanTr.ftanparam.nfout2_1}
        #             self.add_auxiliary_data(data=aftanTr.ftanparam.arr2_1, data_type='DISPbasic2', path=staid_aux,\
        #                                     parameters=parameters)
        #         if inftan.pmf:
        #             if pmf1:
        #                 parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6,\
        #                                     'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_2}
        #                 self.add_auxiliary_data(data=aftanTr.ftanparam.arr1_2, data_type='DISPpmf1', path=staid_aux,\
        #                                         parameters=parameters)
        #             if pmf2:
        #                 parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6,\
        #                                     'amp': 7, 'snr':8, 'Np': aftanTr.ftanparam.nfout2_2}
        #                 self.add_auxiliary_data(data=aftanTr.ftanparam.arr2_2, data_type='DISPpmf2', path=staid_aux,\
        #                                         parameters=parameters)
        #         if outdir != None:
        #             if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
        #                 os.makedirs(outdir+'/'+pfx+'/'+staid1)
        #             foutPR          = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
        #                                 pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
        #             aftanTr.ftanparam.writeDISP(foutPR)
        # print ('[%s] [AFTAN] aftan analysis done' %datetime.now().isoformat().split('.')[0])
        # return