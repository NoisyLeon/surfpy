# -*- coding: utf-8 -*-
"""
ASDF for earthquake data dispersion analysis
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.quake.quakebase as quakebase
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

monthdict   = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
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

class dispASDF(quakebase.baseASDF):
    """ Class for dispersion analysis
    =================================================================================================================
    version history:
        2020/07/09
    =================================================================================================================
    """
    
    def prephp(self, outdir, map_path = None, verbose = True):
        """
        generate predicted phase velocity dispersion curves for earthquake data
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
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        if len(self.cat) >= 10000:
            raise ValueError ('number of events is larger than 10000')
        # loop over stations
        for station_id in self.waveforms.list():
            if verbose:
                print ('*** Station ID: '+station_id)
            netcode     = station_id.split('.')[0]
            stacode     = station_id.split('.')[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos  = self.waveforms[station_id].coordinates
            stla        = tmppos['latitude']
            stlo        = tmppos['longitude']
            stz         = tmppos['elevation_in_m']
            pathfname   = station_id+'_pathfile'
            ievent      = 0
            Ndata       = 0
            taglst      = self.waveforms[station_id].get_waveform_tags()
            with open(pathfname,'w') as f:
                # loop over events 
                for event in self.cat:
                    ievent          += 1
                    evid            = 'E%04d' % (ievent) # evid, 
                    pmag            = event.preferred_magnitude()
                    magnitude       = pmag.mag
                    Mtype           = pmag.magnitude_type
                    event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                    porigin         = event.preferred_origin()
                    otime           = porigin.time
                    timestr         = otime.isoformat()
                    evlo            = porigin.longitude
                    evla            = porigin.latitude
                    try:
                        evdp        = porigin.depth/1000.
                    except:
                        continue
                    event_id        = event.resource_id.id.split('=')[-1]
                    timestr         = otime.isoformat()
                    oyear           = otime.year
                    omonth          = otime.month
                    oday            = otime.day
                    ohour           = otime.hour
                    omin            = otime.minute
                    osec            = otime.second
                    label           = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
                    tag             = 'surf_'+label
                    if not tag in taglst:
                        continue
                    if ( abs(stlo-evlo) < 0.1 and abs(stla-evla)<0.1 ):
                        continue
                    f.writelines('%5d%5d %15s %15s %10.5f %10.5f %10.5f %10.5f \n'
                            %(1, ievent, station_id, evid, stla, stlo, evla, evlo ))
                    Ndata           += 1
            call([prephaseEXE, pathfname, mapfile, perlst, station_id])
            os.remove(pathfname)
            outdirL         = outdir+'_L'
            outdirR         = outdir+'_R'
            if not os.path.isdir(outdirL):
                os.makedirs(outdirL)
            if not os.path.isdir(outdirR):
                os.makedirs(outdirR)
            fout            = open(station_id+'_temp','w')
            for l1 in open('PREDICTION_L'+'_'+station_id):
                l2          = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[4],l2[3])
                    fout    = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[3],l2[2])
                    fout    = open(outname,"w")                
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            for l1 in open('PREDICTION_R'+'_'+station_id):
                l2          = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[4],l2[3])
                    fout    = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[3],l2[2])
                    fout    = open(outname,"w")         
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            fout.close()
            os.remove(station_id+'_temp')
            os.remove('PREDICTION_L'+'_'+station_id)
            os.remove('PREDICTION_R'+'_'+station_id)
        return
    
    def aftan(self, prephdir, sps = 1., channel = 'Z', outdir = None, inftan = pyaftan.InputFtanParam(),\
            basic1 = True, basic2 = True, pmf1 = True, pmf2 = True, verbose = False, f77 = True, pfx = 'DISP'):
        """ aftan analysis of earthquake data 
        =======================================================================================
        ::: input parameters :::
        prephdir    - directory for predicted phase velocity dispersion curve
        sps         - target sampling rate
        channel     - channel pair for aftan analysis('Z', 'R', 'T')
        outdir      - directory for output disp txt files (default = None, no txt output)
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        ---------------------------------------------------------------------------------------
        ::: output :::
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        print ('[%s] [AFTAN] start aftan analysis' %datetime.now().isoformat().split('.')[0])
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        if len(self.cat) >= 10000:
            raise ValueError ('number of events is larger than 10000')
        # Loop over stations
        Nsta            = len(self.waveforms.list())
        ista            = 0
        for staid in self.waveforms.list():
            ista                += 1
            print ('[%s] [AFTAN] Station ID: %s %d/%d' %(datetime.now().isoformat().split('.')[0], \
                           staid, ista, Nsta))
            netcode, stacode    = staid.split('.')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos  = self.waveforms[staid].coordinates
            stla        = tmppos['latitude']
            stlo        = tmppos['longitude']
            stz         = tmppos['elevation_in_m']
            outstr      = ''
            taglst      = self.waveforms[staid].get_waveform_tags()
            if len(taglst) == 0:
                print ('!!! No data for station: '+ staid)
                continue
            # Loop over tags(events)
            ievent      = 0
            Ndata       = 0
            for event in self.cat:
                ievent          += 1
                evid            = 'E%04d' % (ievent) # evid, 
                pmag            = event.preferred_magnitude()
                magnitude       = pmag.mag
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                porigin         = event.preferred_origin()
                otime           = porigin.time
                timestr         = otime.isoformat()
                evlo            = porigin.longitude
                evla            = porigin.latitude
                try:
                    evdp        = porigin.depth/1000.
                except:
                    continue
                event_id        = event.resource_id.id.split('=')[-1]
                timestr         = otime.isoformat()
                oyear   = otime.year
                omonth  = otime.month
                oday    = otime.day
                ohour   = otime.hour
                omin    = otime.minute
                osec    = otime.second
                label   = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
                tag     = 'surf_' + label
                if not tag in taglst:
                    continue
                staid_aux   = netcode+'_'+stacode+'_'+channel
                try:
                    if tag in self.auxiliary_data['DISPbasic1'].list():
                        if staid_aux in self.auxiliary_data['DISPbasic1'][tag].list():
                            continue
                except:
                    pass
                dist, az, baz   = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                dist            = dist/1000.
                if baz < 0:
                    baz         += 360.
                #-------------------
                # get waveform data
                #-------------------
                try:
                    inST        = self.waveforms[staid][tag].select(component = channel)
                except KeyError:
                    continue
                if len(inST) == 0:
                    continue
                else:
                    if len(inST) > 1:
                        print ('!!! WARNING: more traces stored: '+tag+' station: ' + staid )
                    tr              = inST[0]
                # resample
                target_dt       = 1./sps
                dt              = tr.stats.delta
                if abs(dt - target_dt)> (min(dt, target_dt))/100.:
                    factor      = np.round(target_dt/dt)
                    if abs(factor*dt - target_dt) < min(dt, target_dt/1000.):
                        dt                  = target_dt/factor
                        tr.stats.delta      = dt
                    else:
                        print(target_dt, dt)
                        raise ValueError('CHECK!' + staid)
                    tr.filter(type = 'lowpass', freq = sps/2., zerophase = True) # prefilter
                    tr.decimate(factor = int(factor), no_filter = True)
                    # # # try:
                    # # #     tr.filter(type = 'lowpass', freq = sps/2., zerophase = True) # prefilter
                    # # #     tr.decimate(factor = int(factor), no_filter = True)
                    # # # except:
                    # # #     continue
                else:
                    tr.stats.delta  = target_dt
                stime               = tr.stats.starttime
                etime               = tr.stats.endtime
                tr.stats.sac        = {}
                tr.stats.sac['dist']= dist
                tr.stats.sac['b']   = stime - otime
                tr.stats.sac['e']   = etime - otime
                aftanTr             = pyaftan.aftantrace(tr.data, tr.stats)
                phvelname           = prephdir + "/%s.%s.pre" %(evid, staid)
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
                aftanTr.get_snr(ffact = inftan.ffact) # SNR analysis
                staid_aux           = tag+'/'+netcode+'_'+stacode+'_'+channel
                #-----------------------------------
                # save aftan results to ASDF dataset
                #-----------------------------------
                if basic1:
                    parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_1}
                    self.add_auxiliary_data(data = aftanTr.ftanparam.arr1_1, data_type = 'DISPbasic1',\
                            path = staid_aux, parameters = parameters)
                if basic2:
                    parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': aftanTr.ftanparam.nfout2_1}
                    self.add_auxiliary_data(data = aftanTr.ftanparam.arr2_1, data_type = 'DISPbasic2',\
                            path = staid_aux, parameters = parameters)
                if inftan.pmf:
                    if pmf1:
                        parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_2}
                        self.add_auxiliary_data(data = aftanTr.ftanparam.arr1_2, data_type = 'DISPpmf1',\
                            path = staid_aux, parameters = parameters)
                    if pmf2:
                        parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'snr':8, 'Np': aftanTr.ftanparam.nfout2_2}
                        self.add_auxiliary_data(data = aftanTr.ftanparam.arr2_2, data_type = 'DISPpmf2',\
                            path = staid_aux, parameters = parameters)
                if outdir is not None:
                    if not os.path.isdir(outdir+'/'+pfx+'/'+tag):
                        os.makedirs(outdir+'/'+pfx+'/'+tag)
                    foutPR          = outdir+'/'+pfx+'/'+tag+'/'+ staid+'_'+channel+'.SAC'
                    aftanTr.ftanparam.writeDISP(foutPR)
                Ndata               += 1
                outstr              += otime.date.isoformat()
                outstr              += ' '
            print('--- %4d traces processed' %Ndata)
            if verbose:
                print('EVENT DATE: '+outstr)
            print('-----------------------------------------------------------------------------------------------------------')
        print ('[%s] [AFTAN] all done' %datetime.now().isoformat().split('.')[0])
        return
    
    def interp_disp(self, data_type = 'DISPpmf2', channel = 'Z', pers = [], verbose = True):
        """ Interpolate dispersion curve for a given period array.
        =======================================================================================================
        ::: input parameters :::
        data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        pers        - period array
        
        Output:
        self.auxiliary_data.DISPbasic1interp, self.auxiliary_data.DISPbasic2interp,
        self.auxiliary_data.DISPpmf1interp, self.auxiliary_data.DISPpmf2interp
        =======================================================================================================
        """
        print ('[%s] [INTERP_DISP] start interpolating dispersion curves' %datetime.now().isoformat().split('.')[0])
        if data_type=='DISPpmf2':
            ntype   = 6
        else:
            ntype   = 5
        if len(pers) == 0:
            pers    = np.append( np.arange(16.)*2.+10., np.arange(10.)*5.+45.)
        else:
            pers    = np.asarray(pers)
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        staLst      = self.waveforms.list()
        ievent      = 0
        Nevent      = len(self.cat)
        taglst      = self.auxiliary_data[data_type].list()
        # Loop over events
        for event in self.cat:
            ievent          += 1
            Ndata           = 0
            outstr          = ''
            pmag            = event.preferred_magnitude()
            magnitude       = pmag.mag
            Mtype           = pmag.magnitude_type
            event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin         = event.preferred_origin()
            otime           = porigin.time
            timestr         = otime.isoformat()
            evlo            = porigin.longitude
            evla            = porigin.latitude
            try:
                evdp        = porigin.depth/1000.
            except:
                continue
            event_id        = event.resource_id.id.split('=')[-1]
            oyear           = otime.year
            omonth          = otime.month
            oday            = otime.day
            ohour           = otime.hour
            omin            = otime.minute
            osec            = otime.second
            label           = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
            event_tag       = 'surf_'+label
            if not event_tag in taglst:
                # print(event_tag + ' not in the event list')
                continue
            if magnitude is None:
                continue
            print ('[%s] [INTERP_DISP] ' %(datetime.now().isoformat().split('.')[0]) + \
                   'Event ' + str(ievent)+'/'+str(Nevent)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
            datalst         = self.auxiliary_data[data_type][event_tag].list()
            # Loop over stations
            for staid in self.waveforms.list():
                netcode, stacode    = staid.split('.')
                dataid              = netcode+'_'+stacode+'_'+channel
                if not dataid in datalst:
                    continue
                try:
                    subdset         = self.auxiliary_data[data_type][event_tag][dataid]
                except KeyError:
                    continue
                # skip upon existence
                staid_aux_temp      = netcode+'_'+stacode+'_'+channel
                if data_type+'interp' in self.auxiliary_data.list():
                    if event_tag in self.auxiliary_data[data_type+'interp'].list():
                        if staid_aux_temp in self.auxiliary_data[data_type+'interp'][event_tag].list():
                            continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data            = subdset.data[()]
                    index           = subdset.parameters
                outindex            = { 'To': 0, 'U': 1, 'C': 2,  'amp': 3, 'snr': 4, 'inbound': 5, 'Np': pers.size }
                Np                  = int(index['Np'])
                if Np < 5:
                    continue
                obsT                = data[index['To']][:Np]
                U                   = np.interp(pers, obsT, data[index['U']][:Np] )
                C                   = np.interp(pers, obsT, data[index['C']][:Np] )
                amp                 = np.interp(pers, obsT, data[index['amp']][:Np] )
                inbound             = (pers > obsT[0])*(pers < obsT[-1])*1
                interpdata          = np.append(pers, U)
                interpdata          = np.append(interpdata, C)
                interpdata          = np.append(interpdata, amp)
                if data_type == 'DISPpmf2':
                    snr             = np.interp(pers, obsT, data[index['snr']][:Np] )
                    interpdata      = np.append(interpdata, snr)
                interpdata          = np.append(interpdata, inbound)
                interpdata          = interpdata.reshape(ntype, pers.size)
                staid_aux           = event_tag+'/'+netcode+'_'+stacode+'_'+channel
                self.add_auxiliary_data(data = interpdata, data_type = data_type+'interp', path = staid_aux, parameters = outindex)
                Ndata               += 1
                outstr              += staid
                outstr              += ' '
            print('--- ' +str(Ndata)+' traces processed')
            if verbose:
                print('STATION CODE: '+outstr)
            print('-----------------------------------------------------------------------------------------------------------')
        return
    
    def get_field(self, outdir = None, channel = 'Z', pers = [], data_type = 'DISPpmf2interp', verbose = True):
        """ Get the field data for Eikonal/Helmholtz tomography
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
        if len(pers) == 0:
            pers    = np.append( np.arange(16.)*2.+10., np.arange(10.)*5.+45.)
        else:
            pers    = np.asarray(pers)
        outindex    = { 'longitude': 0, 'latitude': 1, 'C': 2,  'U':3, 'amp': 4, 'snr': 5, 'dist': 6 }
        staLst      = self.waveforms.list()
        ievent      = 0
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        Nevent      = len(self.cat)
        taglst      = self.auxiliary_data[data_type].list()
        for event in self.cat:
            ievent          += 1
            evid            = 'E%05d' % ievent # evid, e.g. E01011 corresponds to cat[1010]
            Ndata           = 0
            outstr          = ''
            pmag            = event.preferred_magnitude()
            magnitude       = pmag.mag
            Mtype           = pmag.magnitude_type
            event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin         = event.preferred_origin()
            otime           = porigin.time
            timestr         = otime.isoformat()
            evlo            = porigin.longitude
            evla            = porigin.latitude
            try:
                evdp        = porigin.depth/1000.
            except:
                continue
            event_id        = event.resource_id.id.split('=')[-1]
            oyear           = otime.year
            omonth          = otime.month
            oday            = otime.day
            ohour           = otime.hour
            omin            = otime.minute
            osec            = otime.second
            label           = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
            event_tag       = 'surf_'+label
            if not event_tag in taglst:
                # print(event_tag + ' not in the event list')
                continue
            if magnitude is None:
                continue
            print ('[%s] [GET_FIELD] ' %(datetime.now().isoformat().split('.')[0]) + \
                   'Event ' + str(ievent)+'/'+str(Nevent)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
            field_lst       = []
            Nfplst          = []
            for per in pers:
                field_lst.append(np.array([]))
                Nfplst.append(0)
            datalst         = self.auxiliary_data[data_type][event_tag].list()
            # skip upon existence
            if 'Field'+data_type in self.auxiliary_data.list():
                if event_tag+'_'+channel in self.auxiliary_data['Field'+data_type].list():
                    print ('--- Skip upon existence!')
                    continue
            # Loop over stations
            for staid in self.waveforms.list():
                netcode, stacode    = staid.split('.')
                dataid              = netcode+'_'+stacode+'_'+channel
                if not dataid in datalst:
                    continue
                subdset         = self.auxiliary_data[data_type][event_tag][dataid]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos      = self.waveforms[staid].coordinates
                stla            = tmppos['latitude']
                stlo            = tmppos['longitude']
                stz             = tmppos['elevation_in_m']
                dist, az, baz   = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                dist            = dist/1000.
                if stlo<0:
                    stlo        += 360.
                if evlo<0:
                    evlo        += 360.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data            = subdset.data[()]
                    index           = subdset.parameters
                for iper in range(pers.size):
                    per             = pers[iper]
                    ind_per         = np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise AttributeError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    pvel            = data[index['C']][ind_per]
                    gvel            = data[index['U']][ind_per]
                    snr             = data[index['snr']][ind_per]
                    amp             = data[index['amp']][ind_per]
                    inbound         = data[index['inbound']][ind_per]
                    # quality control
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10:
                        continue
                    if inbound != 1.:
                        continue
                    field_lst[iper] = np.append(field_lst[iper], stlo)
                    field_lst[iper] = np.append(field_lst[iper], stla)
                    field_lst[iper] = np.append(field_lst[iper], pvel)
                    field_lst[iper] = np.append(field_lst[iper], gvel)
                    field_lst[iper] = np.append(field_lst[iper], snr)
                    field_lst[iper] = np.append(field_lst[iper], dist)
                    field_lst[iper] = np.append(field_lst[iper], amp)
                    Nfplst[iper]    += 1                    
                Ndata               += 1
                outstr              += staid
                outstr              += ' '
            print('--- '+str(Ndata)+' traces processed for field data')
            if verbose:
                print('STATION CODE: '+outstr)
            print('-----------------------------------------------------------------------------------------------------------')
            if outdir is not None:
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
            staid_aux               = event_tag+'_'+channel
            for iper in range(pers.size):
                per                 = pers[iper]
                del_per             = per-int(per)
                if field_lst[iper].size==0:
                    continue
                field_lst[iper]     = field_lst[iper].reshape(Nfplst[iper], 7)
                if del_per==0.:
                    staid_aux_per   = staid_aux+'/'+str(int(per))+'sec'
                else:
                    dper            = str(del_per)
                    staid_aux_per   = staid_aux+'/'+str(int(per))+'sec'+dper.split('.')[1]
                self.add_auxiliary_data(data = field_lst[iper], data_type = 'Field'+data_type,\
                                        path = staid_aux_per, parameters = outindex)
                if outdir is not None:
                    if not os.path.isdir(outdir+'/'+str(per)+'sec'):
                        os.makedirs(outdir+'/'+str(per)+'sec')
                    txtfname        = outdir+'/'+str(per)+'sec'+'/'+evid+'_'+str(per)+'.txt'
                    header          = 'evlo='+str(evlo)+' evla='+str(evla)
                    np.savetxt( txtfname, field_lst[iper], fmt='%g', header=header )
        return
    
    
    
