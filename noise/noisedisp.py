# -*- coding: utf-8 -*-
"""
ASDF for noise data dispersion analysis
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.noise.noisebase as noisebase

try:
    import surfpy.aftan.pyaftan as pyaftan
    is_aftan    = True
except:
    is_aftan    = False
    
import surfpy.map_dat.glb_ph_vel_maps as MAPS
global_map_path    = MAPS.__path__._path[0]

import numpy as np
from functools import partial
import multiprocessing
import obspy
import obspy.io.sac
import obspy.io.xseed
from datetime import datetime
import warnings
import shutil
import glob
import sys
import copy
from subprocess import call
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
class dispError(Exception):
    pass

class dispIOError(dispError, IOError):
    pass

class dispHeaderError(dispError):
    """
    Raised if header has issues.
    """
    pass

class dispDataError(dispError):
    """
    Raised if header has issues.
    """
    pass

class dispASDF(noisebase.baseASDF):
    """ Class for dispersion analysis
    =================================================================================================================
    version history:
        2020/07/09
    =================================================================================================================
    """
    def prephp(self, outdir, map_path = None):
        """
        Generate predicted phase velocity dispersion curves for cross-correlation pairs
        ====================================================================================
        ::: input parameters :::
        outdir  - output directory
        mapfile - phase velocity maps
        ------------------------------------------------------------------------------------
        Input format:
        prephaseEXE pathfname mapfile perlst staname
        
        if ERROR like this shows up:
        Internal Error: get_unit(): Bad internal unit KIND
        
        try to change gfortran version, then recompile the code
        it is related to gfortran version and libgfortran.so.3 (or maybe libgfortran.so.3)

        Output format:
        outdirL(outdirR)/evid.staid.pre
        ====================================================================================
        """
        if map_path is None:
            map_path= global_map_path
        prephaseEXE = map_path+'/mhr_grvel_predict/lf_mhr_predict_earth'
        perlst      = map_path+'/mhr_grvel_predict/perlist_phase'
        if not os.path.isfile(prephaseEXE):
            raise dispError('lf_mhr_predict_earth executable does not exist!')
        if not os.path.isfile(perlst):
            raise dispError('period list does not exist!')
        mapfile     = map_path+'/smpkolya_phv'
        outdirL     = outdir+'_L'
        outdirR     = outdir+'_R'
        if not os.path.isdir(outdirL):
            os.makedirs(outdirL)
        if not os.path.isdir(outdirR):
            os.makedirs(outdirR)
        staLst      = self.waveforms.list()
        for evid in staLst:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos1 = self.waveforms[evid].coordinates
            evla        = tmppos1['latitude']
            evlo        = tmppos1['longitude']
            evz         = tmppos1['elevation_in_m']
            pathfname   = evid+'_pathfile'
            with open(pathfname,'w') as f:
                ista                = 0
                for station_id in staLst:
                    if evid >= station_id:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tmppos2 = self.waveforms[station_id].coordinates
                    stla        = tmppos2['latitude']
                    stlo        = tmppos2['longitude']
                    stz         = tmppos2['elevation_in_m']
                    if ( abs(stlo-evlo) < 0.1 and abs(stla-evla)<0.1 ):
                        continue
                    ista        = ista+1
                    f.writelines('%5d%5d %15s %15s %10.5f %10.5f %10.5f %10.5f \n'
                            %(1, ista, evid, station_id, evla, evlo, stla, stlo ))
            call([prephaseEXE, pathfname, mapfile, perlst, evid])
            os.remove(pathfname)
            fout            = open(evid+'_temp','w')
            for l1 in open('PREDICTION_L'+'_'+evid):
                l2          = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[3],l2[4])
                    fout    = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[2],l2[3])
                    fout    = open(outname,"w")                
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            for l1 in open('PREDICTION_R'+'_'+evid):
                l2          = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[3],l2[4])
                    fout    = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[2],l2[3])
                    fout    = open(outname,"w")         
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            fout.close()
            os.remove(evid+'_temp')
            os.remove('PREDICTION_L'+'_'+evid)
            os.remove('PREDICTION_R'+'_'+evid)
        return
    
    def aftan(self, prephdir, ic2c3 = 1, channel = 'ZZ', outdir = None, inftan = pyaftan.InputFtanParam(),\
        basic1 = True, basic2 = True, pmf1 = True, pmf2 = True, verbose = False, f77 = True):
        """ aftan analysis of cross-correlation data 
        =======================================================================================
        ::: input parameters :::
        prephdir    - directory for predicted phase velocity dispersion curve
        ic2c3       - index for xcorr or C3 ( 1 - xcorr; 2 - C3)
        channel     - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        outdir      - directory for output disp txt files (default = None, no txt output)
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        f77         - use aftanf77 or not
        ---------------------------------------------------------------------------------------
        ::: output :::
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        if ic2c3 == 1:
            pfx     = 'DISP'
        elif ic2c3 == 2:
            pfx     = 'C3DISP'
        else:
            raise ValueError('Unexpected ic2c3 = %d' %ic2c3)
        print ('[%s] [AFTAN] start aftan analysis' %datetime.now().isoformat().split('.')[0])
        staLst                      = self.waveforms.list()
        Nsta                        = len(staLst)
        Ntotal_traces               = int(Nsta*(Nsta-1)/2)
        iaftan                      = 0
        Ntr_one_percent             = int(Ntotal_traces/100.)
        ipercent                    = 0
        for staid1 in staLst:
            netcode1, stacode1      = staid1.split('.')
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                # print how many traces has been processed
                iaftan              += 1
                if np.fmod(iaftan, Ntr_one_percent) ==0:
                    ipercent        += 1
                    print ('[%s] [AFTAN] Number of traces finished : ' %datetime.now().isoformat().split('.')[0] \
                           +str(iaftan)+'/'+str(Ntotal_traces)+' '+str(ipercent)+'%')
                # determine channels and get data
                if ic2c3 == 1:
                    try:
                        channels1       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                        for chan in channels1:
                            if chan[-1] == channel[0]:
                                chan1   = chan
                                break
                        channels2       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1].list()
                        for chan in channels2:
                            if chan[-1] == channel[1]:
                                chan2   = chan
                                break
                    except KeyError:
                        continue
                    tr                  = self.get_xcorr_trace(netcode1, stacode1, netcode2, stacode2, chan1, chan2)
                elif ic2c3 == 2:
                    tr                  = self.get_c3_trace(netcode1, stacode1, netcode2, stacode2, channel[0], channel[1])
                if tr is None:
                    # print ('*** WARNING: '+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+' not exists!')
                    continue
                #================
                # aftan analysis
                #================
                aftanTr             = pyaftan.aftantrace(tr.data, tr.stats)
                if abs(aftanTr.stats.sac.b + aftanTr.stats.sac.e) < aftanTr.stats.delta :
                    aftanTr.makesym()
                phvelname           = prephdir + "/%s.%s.pre" %(netcode1+'.'+stacode1, netcode2+'.'+stacode2)
                if not os.path.isfile(phvelname):
                    print ('*** WARNING: '+ phvelname+' not exists!')
                    continue
                if f77:
                    aftanTr.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                            npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
                else:
                    aftanTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                            npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
                if verbose:
                    print ('--- aftan analysis for: ' + netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel)
                # SNR
                aftanTr.get_snr(ffact = inftan.ffact) 
                staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                #=====================================
                # save aftan results to ASDF dataset
                #=====================================
                if basic1:
                    parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6,\
                                            'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_1}
                    self.add_auxiliary_data(data = aftanTr.ftanparam.arr1_1, data_type = pfx + 'basic1',\
                                            path = staid_aux, parameters = parameters)
                if basic2:
                    parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6,\
                                            'amp': 7, 'Np': aftanTr.ftanparam.nfout2_1}
                    self.add_auxiliary_data(data = aftanTr.ftanparam.arr2_1, data_type = pfx + 'basic2',\
                                            path = staid_aux, parameters = parameters)
                if inftan.pmf:
                    if pmf1:
                        parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6,\
                                            'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_2}
                        self.add_auxiliary_data(data = aftanTr.ftanparam.arr1_2, data_type = pfx + 'pmf1',\
                                            path = staid_aux, parameters = parameters)
                    if pmf2:
                        parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6,\
                                            'amp': 7, 'snr':8, 'Np': aftanTr.ftanparam.nfout2_2}
                        self.add_auxiliary_data(data = aftanTr.ftanparam.arr2_2, data_type = pfx + 'pmf2',\
                                            path = staid_aux, parameters = parameters)
                if outdir is not None:
                    if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
                        os.makedirs(outdir+'/'+pfx+'/'+staid1)
                    foutPR  = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
                    aftanTr.ftanparam.writeDISP(foutPR)
        print ('[%s] [AFTAN] aftan analysis done' %datetime.now().isoformat().split('.')[0])
        return
    
    def interp_disp(self, data_type = 'DISPpmf2', channel = 'ZZ', pers = np.array([]), verbose = False):
        """ Interpolate dispersion curve for a given period array.
        =======================================================================================================
        ::: input parameters :::
        data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        pers        - period array
        
        ::: output :::
        self.auxiliary_data.DISPbasic1interp, self.auxiliary_data.DISPbasic2interp,
        self.auxiliary_data.DISPpmf1interp, self.auxiliary_data.DISPpmf2interp
        =======================================================================================================
        """
        print ('[%s] [FTAN_INTERP] start interpolating aftan results' %datetime.now().isoformat().split('.')[0])
        if data_type=='DISPpmf2' or data_type == 'C3DISPpmf2':
            ntype   = 6
        else:
            ntype   = 5
        if pers.size==0:
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        staLst                      = self.waveforms.list()
        Nsta                        = len(staLst)
        Ntotal_traces               = int(Nsta*(Nsta-1)/2)
        iinterp                     = 0
        Ntr_one_percent             = int(Ntotal_traces/100.)
        ipercent                    = 0
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                iinterp             += 1
                if np.fmod(iinterp, Ntr_one_percent) ==0:
                    ipercent        += 1
                    print ('[%s] [FTAN_INTERP] Number of traces finished: ' %datetime.now().isoformat().split('.')[0]+\
                                    str(iinterp)+'/'+str(Ntotal_traces)+' '+str(ipercent)+'%')
                # get the data
                try:
                    subdset         = self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except KeyError:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data            = subdset.data.value
                    index           = subdset.parameters
                if verbose:
                    print ('--- interpolating dispersion curve for '+ staid1+'_'+staid2+'_'+channel)
                outindex            = { 'To': 0, 'U': 1, 'C': 2,  'amp': 3, 'snr': 4, 'inbound': 5, 'Np': pers.size }
                Np                  = int(index['Np'])
                if Np < 5:
                    if verbose:
                        # warnings.warn('Not enough datapoints for: '+ staid1+'_'+staid2+'_'+channel, UserWarning, stacklevel=1)
                        print ('*** WARNING: Not enough datapoints for: '+ staid1+'_'+staid2+'_'+channel)
                    continue
                # interpolation
                obsT                = data[index['To']][:Np]
                U                   = np.interp(pers, obsT, data[index['U']][:Np] )
                C                   = np.interp(pers, obsT, data[index['C']][:Np] )
                amp                 = np.interp(pers, obsT, data[index['amp']][:Np] )
                inbound             = (pers > obsT[0])*(pers < obsT[-1])*1
                # store interpolated data to interpdata array
                interpdata          = np.append(pers, U)
                interpdata          = np.append(interpdata, C)
                interpdata          = np.append(interpdata, amp)
                if data_type is 'DISPpmf2' or data_type is 'C3DISPpmf2':
                    snr             = np.interp(pers, obsT, data[index['snr']][:Np] )
                    interpdata      = np.append(interpdata, snr)
                interpdata          = np.append(interpdata, inbound)
                interpdata          = interpdata.reshape(ntype, pers.size)
                staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                self.add_auxiliary_data(data = interpdata, data_type = data_type+'interp', path = staid_aux, parameters = outindex)
        print ('[%s] [FTAN_INTERP] aftan interpolation all done' %datetime.now().isoformat().split('.')[0])
        return
    
    def raytomo_input(self, outdir, staxml=None, netcodelst=[], lambda_factor=3., snr_thresh=15., channel='ZZ',\
                           pers=[], outpfx='raytomo_in_', data_type='DISPpmf2interp', verbose=True):
        """generate input files for Barmine's straight ray surface wave tomography code.
        =======================================================================================================
        ::: input parameters :::
        outdir          - output directory
        staxml          - input StationXML for data selection
        netcodelst      - network list for data selection
        lambda_factor   - wavelength factor for data selection (default = 3.)
        snr_thresh      - threshold SNR (default = 15.)
        channel         - channel for tomography
        pers            - period array
        outpfx          - prefix for output files, default is 'raytomo_in_'
        data_type       - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        -------------------------------------------------------------------------------------------------------
        Output format:
        outdir/outpfx+per_channel_ph.lst
        =======================================================================================================
        """
        print ('[%s] [RAYTOMO_INPUT] Start generating straight ray tomography input files' %datetime.now().isoformat().split('.')[0])
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if len(pers) ==0:
            pers        = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        else:
            pers        = np.asarray(pers)
        # open output files
        fph_lst         = []
        fgr_lst         = []
        for per in pers:
            fname_ph    = outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_ph.lst' %( per )
            fname_gr    = outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_gr.lst' %( per )
            fph         = open(fname_ph, 'w')
            fgr         = open(fname_gr, 'w')
            fph_lst.append(fph)
            fgr_lst.append(fgr)
        #------------------------------------------------
        # using stations from an input StationXML file
        #------------------------------------------------
        if staxml is not None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            staLst          = waveformLst
            print ('--- Load stations from input StationXML file')
        else:
            print ('--- Load all the stations from database')
            staLst          = self.waveforms.list()
        # network selection
        if len(netcodelst) != 0:
            staLst_ALL      = copy.deepcopy(staLst)
            staLst          = []
            for staid in staLst_ALL:
                netcode, stacode    = staid.split('.')
                if not (netcode in netcodelst):
                    continue
                staLst.append(staid)
            print ('--- Select stations according to network code: '+str(len(staLst))+'/'+str(len(staLst_ALL))+' (selected/all)')
        # Loop over stations
        Nsta            = len(staLst)
        Ntotal_traces   = int(Nsta*(Nsta-1)/2)
        Ntr_one_percent = int(Ntotal_traces/100.)
        ipercent        = 0
        iray            = -1
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                # print how many rays has been processed 
                iray                += 1
                if np.fmod(iray+1, Ntr_one_percent) ==0:
                    ipercent        += 1
                    print ('[%s] [RAYTOMO_INPUT] Number of traces finished: ' \
                        %datetime.now().isoformat().split('.')[0] +str(iray)+'/'+str(Ntotal_traces)+' '+str(ipercent)+'%')
                # get data
                try:
                    subdset         = self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos1 = self.waveforms[staid1].coordinates
                    tmppos2 = self.waveforms[staid2].coordinates
                lat1            = tmppos1['latitude']
                lon1            = tmppos1['longitude']
                elv1            = tmppos1['elevation_in_m']
                lat2            = tmppos2['latitude']
                lon2            = tmppos2['longitude']
                elv2            = tmppos2['elevation_in_m']
                dist, az, baz   = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist            = dist/1000.
                if lon1 < 0:
                    lon1        += 360.
                if lon2 < 0:
                    lon2        += 360.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data        = subdset.data.value
                    index       = subdset.parameters
                for iper in range(pers.size):
                    per         = pers[iper]
                    # wavelength criteria
                    if dist < lambda_factor * per * 3.5:
                        continue
                    ind_per         = np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise AttributeError('*** WARNING: No interpolated dispersion curve data for period = '+str(per)+' sec!')
                    pvel            = data[index['C']][ind_per]
                    gvel            = data[index['U']][ind_per]
                    snr             = data[index['snr']][ind_per]
                    inbound         = data[index['inbound']][ind_per]
                    # quality control
                    if inbound != 1.:
                        continue
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10:
                        continue
                    if snr < snr_thresh: # SNR larger than threshold value
                        continue
                    fph             = fph_lst[iper]
                    fgr             = fgr_lst[iper]
                    fph.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(iray, lat1, lon1, lat2, lon2, pvel, staid1, staid2))
                    fgr.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(iray, lat1, lon1, lat2, lon2, gvel, staid1, staid2))
        for iper in range(pers.size):
            fph     = fph_lst[iper]
            fgr     = fgr_lst[iper]
            fph.close()
            fgr.close()
        print ('[%s] [RAYTOMO_INPUT] all done!' %datetime.now().isoformat().split('.')[0])
        return
    
    def get_field(self, outdir= None, channel='ZZ', pers=[], data_type = 'DISPpmf2interp', use_all=True, verbose=True):
        """ get the field data for eikonal tomography
        ============================================================================================================================
        ::: input parameters :::
        outdir          - directory for txt output (default is not to generate txt output)
        channel         - channel name
        pers            - period array
        datatype        - dispersion data type (default = DISPpmf2interp, interpolated pmf aftan results after jump detection)
        ::: output :::
        self.auxiliary_data.FieldDISPpmf2interp
        ============================================================================================================================
        """
        print ('[%s] [GET_FIELD] Get field data for eikonal tomography' %datetime.now().isoformat().split('.')[0])
        if len(pers) == 0:
            pers        = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        else:
            pers        = np.asarray(pers)
        outindex        = { 'longitude': 0, 'latitude': 1, 'C': 2,  'U':3, 'snr': 4, 'dist': 5 }
        #===================
        # Loop over stations
        #===================
        for staid1 in self.waveforms.list():
            netcode1, stacode1  = staid1.split('.')
            field_lst   = []
            Nfplst      = []
            for per in pers:
                field_lst.append(np.array([]))
                Nfplst.append(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos1         = self.waveforms[staid1].coordinates
            lat1                = tmppos1['latitude']
            lon1                = tmppos1['longitude']
            elv1                = tmppos1['elevation_in_m']
            Ndata       = 0
            for staid2 in self.waveforms.list():
                if staid1 == staid2:
                    continue
                netcode2, stacode2  = staid2.split('.')
                #============
                # get data
                #============
                no_data         = False
                ind_borrow      = 0
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        subdset     = self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            subdset = self.auxiliary_data[data_type][netcode2][stacode2][netcode1][stacode1][channel]
                    except:
                        no_data     = True
                if no_data and use_all:
                    no_data         = False
                    ind_borrow      = 1
                    if data_type[:2] == 'C3':
                        tmp_data_type   = data_type[2:]
                    else:
                        tmp_data_type   = 'C3'+data_type
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            subdset     = self.auxiliary_data[tmp_data_type][netcode1][stacode1][netcode2][stacode2][channel]
                    except:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                subdset = self.auxiliary_data[tmp_data_type][netcode2][stacode2][netcode1][stacode1][channel]
                        except:
                            no_data     = True
                if no_data:
                    continue
                Ndata               +=1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos2         = self.waveforms[staid2].coordinates
                lat2                = tmppos2['latitude']
                lon2                = tmppos2['longitude']
                elv2                = tmppos2['elevation_in_m']
                dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist                = dist/1000.
                if lon1<0:
                    lon1    += 360.
                if lon2<0:
                    lon2    += 360.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data            = subdset.data.value
                    index           = subdset.parameters
                # loop over periods
                for iper in range(pers.size):
                    per     = pers[iper]
                    ind_per = np.where(data[index['To']][:] == per)[0]
                    if ind_per.size == 0:
                        raise KeyError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    try:
                        pvel    = data[index['C']][ind_per]
                        gvel    = data[index['U']][ind_per]
                    except:
                        pvel    = data[index['Vph']][ind_per]
                        gvel    = data[index['Vgr']][ind_per]
                    snr     = data[index['snr']][ind_per]
                    inbound = data[index['inbound']][ind_per]
                    # quality control
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10:
                        continue
                    if not bool(inbound):
                        continue
                    field_lst[iper] = np.append(field_lst[iper], lon2)
                    field_lst[iper] = np.append(field_lst[iper], lat2)
                    field_lst[iper] = np.append(field_lst[iper], pvel)
                    field_lst[iper] = np.append(field_lst[iper], gvel)
                    field_lst[iper] = np.append(field_lst[iper], snr)
                    field_lst[iper] = np.append(field_lst[iper], dist)
                    # index indicating if the path is borrowed from C2/C3
                    field_lst[iper] = np.append(field_lst[iper], ind_borrow)
                    Nfplst[iper]    += 1
            if verbose:
                print ('=== Getting field data for: '+staid1+', '+str(Ndata)+' paths')
            # end of reading data from all receivers, taking staid1 as virtual source
            if outdir is not None:
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
            #===========
            # save data
            #===========
            staid_aux   = netcode1+'/'+stacode1+'/'+channel
            for iper in range(pers.size):
                per                 = pers[iper]
                del_per             = per-int(per)
                if field_lst[iper].size==0:
                    continue
                field_lst[iper]     = field_lst[iper].reshape(Nfplst[iper], 7)
                if del_per == 0.:
                    staid_aux_per   = staid_aux+'/'+str(int(per))+'sec'
                else:
                    dper            = str(del_per)
                    staid_aux_per   = staid_aux+'/'+str(int(per))+'sec'+dper.split('.')[1]
                self.add_auxiliary_data(data = field_lst[iper], data_type = 'Field'+data_type,\
                                        path = staid_aux_per, parameters = outindex)
                if outdir is not None:
                    if not os.path.isdir(outdir+'/'+str(per)+'sec'):
                        os.makedirs(outdir+'/'+str(per)+'sec')
                    txtfname        = outdir+'/'+str(per)+'sec'+'/'+staid1+'_'+str(per)+'.txt'
                    header          = 'evlo='+str(lon1)+' evla='+str(lat1)
                    np.savetxt( txtfname, field_lst[iper], fmt='%g', header=header )
        print ('[%s] [GET_FIELD] ALL done' %datetime.now().isoformat().split('.')[0])
        return
    
    
