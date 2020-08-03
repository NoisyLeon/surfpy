# -*- coding: utf-8 -*-
"""
hdf5 for noise eikonal tomography
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
try:
    import surfpy.eikonal.tomobase as tomobase
except:
    import tomobase

try:
    import surfpy.eikonal._eikonal_funcs as _eikonal_funcs
except:
    import _eikonal_funcs
    
try:
    import surfpy.eikonal._grid_class as _grid_class
except:
    import _grid_class

import numpy as np
from functools import partial
import multiprocessing
import obspy
import obspy.io.sac
import obspy.io.xseed
import pyasdf 
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


class noiseh5(tomobase.baseh5):
    
    def load_ASDF(self, in_asdf_fname, channel='ZZ', data_type='FieldDISPpmf2interp',\
                  staxml = None, netcodelst=[], verbose = True):
        """load travel time field data from ASDF
        =================================================================================================================
        ::: input parameters :::
        in_asdf_fname   - input ASDF data file
        channel         - channel for analysis (default = ZZ )
        data_type       - data type
                            default = 'FieldDISPpmf2interp'
                                aftan measurements with phase-matched filtering and jump correction
        =================================================================================================================
        """
        #---------------------------------
        # get stations (virtual events)
        #---------------------------------
        # input xcorr database
        indbase             = pyasdf.ASDFDataSet(in_asdf_fname)
        if staxml is not None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            event_lst       = waveformLst
            print ('--- Load stations from input StationXML file')
        else:
            print ('--- Load all the stations from database')
            event_lst       = indbase.waveforms.list()
        # network selection
        if len(netcodelst) != 0:
            staLst_ALL      = copy.deepcopy(event_lst)
            event_lst       = []
            for staid in staLst_ALL:
                netcode, stacode    = staid.split('.')
                if not (netcode in netcodelst):
                    continue
                event_lst.append(staid)
            print ('--- Select stations according to network code: '+str(len(event_lst))+'/'+str(len(staLst_ALL))+' (selected/all)')
        # create group for input data
        group               = self.create_group( name = 'input_field_data')
        group.attrs.create(name = 'channel', data = channel)
        # loop over periods
        for per in self.pers:
            print ('--- loading data for: '+str(per)+' sec')
            del_per         = per - int(per)
            if del_per==0.:
                per_name    = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                per_name    = str(int(per))+'sec'+dper.split('.')[1]
            per_group       = group.create_group(name = '%g_sec' %per)
            for evid in event_lst:
                netcode1, stacode1  = evid.split('.')
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        subdset     = indbase.auxiliary_data[data_type][netcode1][stacode1][channel][per_name]
                except KeyError:
                    print ('!!! No travel time field for: '+evid)
                    continue
                if verbose:
                    print ('--- virtual event: '+evid)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos1         = indbase.waveforms[evid].coordinates
                lat1                = tmppos1['latitude']
                lon1                = tmppos1['longitude']
                elv1                = tmppos1['elevation_in_m']
                if lon1 < 0.:
                    lon1            += 360.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data            = subdset.data.value
                # save data to hdf5 dataset
                event_group         = per_group.create_group(name = evid)
                event_group.attrs.create(name = 'evlo', data = lon1)
                event_group.attrs.create(name = 'evla', data = lat1)
                event_group.attrs.create(name = 'num_data_points', data = data.shape[0])
                event_group.create_dataset(name='lons', data = data[:, 0])
                event_group.create_dataset(name='lats', data = data[:, 1])
                event_group.create_dataset(name='phase_velocity', data = data[:, 2])
                event_group.create_dataset(name='group_velocity', data = data[:, 3])
                event_group.create_dataset(name='snr', data = data[:, 4])
                event_group.create_dataset(name='distance', data = data[:, 5])
        return
    
    def run(self, workingdir = None, lambda_factor = 3., snr_thresh = 15., runid = 0, cdist = 250., mindp = 10, deletetxt = True, verbose = False):
        """perform eikonal computing
        =================================================================================================================
        ::: input parameters :::
        workingdir      - working directory
        lambda_factor   - wavelength factor for data selection (default = 3.)
        snr_thresh      - threshold SNR (default = 15.)
        runid           - run id
        cdist           - distance for nearneighbor station criteria
        mindp           - minnimum required number of data points for eikonal operator
        deletetxt       - delete output txt files in working directory
        =================================================================================================================
        """
        # create new eikonal group
        create_group        = False
        while (not create_group):
            try:
                group       = self.create_group( name = 'Eikonal_run_'+str(runid) )
                create_group= True
            except:
                runid       += 1
                continue
        if workingdir is None:
            workingdir      = os.path.dirname(self.filename)+'/eikonal_run_%g' %runid
        datagrp             = self['input_field_data']
        channel             = datagrp.attrs['channel']
        minlon              = self.minlon
        maxlon              = self.maxlon
        minlat              = self.minlat
        maxlat              = self.maxlat
        dlon                = self.dlon
        dlat                = self.dlat
        print ('[%s] [EIKONAL_TOMO] eikonal tomography START' %datetime.now().isoformat().split('.')[0])
        for per in self.pers:
            print ('[%s] [EIKONAL_TOMO] Computing gradients for T = %g sec' %(datetime.now().isoformat().split('.')[0], per))
            dat_per_grp     = datagrp['%g_sec' %per] 
            event_lst       = dat_per_grp.keys()
            working_per     = workingdir+'/'+str(per)+'sec'
            per_group       = group.create_group( name='%g_sec'%( per ) )
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            for evid in event_lst:
                dat_ev_grp  = dat_per_grp[evid]
                numb_points = dat_ev_grp.attrs['num_data_points']
                if numb_points <= mindp:
                    # # print (numb_points)
                    continue
                evlo        = dat_ev_grp.attrs['evlo']
                evla        = dat_ev_grp.attrs['evla']
                lons        = dat_ev_grp['lons'][()]
                lats        = dat_ev_grp['lats'][()]
                dist        = dat_ev_grp['distance'][()]
                C           = dat_ev_grp['phase_velocity'][()]
                if verbose:
                    print ('=== event: '+evid)
                gridder     = _grid_class.SphereGridder(minlon = minlon, maxlon = maxlon, dlon = dlon, \
                                minlat = minlat, maxlat = maxlat, dlat = dlat, period = per, \
                                evlo = evlo, evla = evla, fieldtype = 'Tph', evid = evid)
                gridder.read_array(inlons = np.append(evlo, lons), inlats = np.append(evla, lats), inzarr = np.append(0., dist/C))
                outfname    = evid+'_Tph_'+channel+'.lst'
                prefix      = evid+'_'+channel+'_'
                gridder.interp_surface(workingdir = working_per, outfname = outfname)
                gridder.check_curvature(workingdir = working_per, outpfx = prefix)
                gridder.eikonal(workingdir = working_per, inpfx = prefix, nearneighbor = True, cdist = cdist)
                #==========================
                # save data to hdf5 dataset
                #==========================
                event_group = per_group.create_group(name = evid)
                event_group.attrs.create(name = 'evlo', data = evlo)
                event_group.attrs.create(name = 'evla', data = evla)
                event_group.attrs.create(name = 'Ntotal_grd', data = gridder.Ntotal_grd)
                event_group.attrs.create(name = 'Nvalid_grd', data = gridder.Nvalid_grd)
                # output datasets
                event_group.create_dataset(name = 'apparent_velocity', data = gridder.app_vel)
                event_group.create_dataset(name = 'reason_n', data = gridder.reason_n)
                event_group.create_dataset(name = 'propagation_angle', data = gridder.pro_angle)
                event_group.create_dataset(name = 'azimuth', data = gridder.az)
                event_group.create_dataset(name = 'back_azimuth', data = gridder.baz)
                event_group.create_dataset(name = 'travel_time', data = gridder.Zarr)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def runMP(self, workingdir = None, lambda_factor = 3., snr_thresh = 15., runid = 0, cdist = 250., mindp = 10,\
            subsize = 1000, nprocess = None, deletetxt = True, verbose = False):
        """perform eikonal computing with multiprocessing
        =================================================================================================================
        ::: input parameters :::
        workingdir      - working directory
        lambda_factor   - wavelength factor for data selection (default = 3.)
        snr_thresh      - threshold SNR (default = 15.)
        runid           - run id
        cdist           - distance for nearneighbor station criteria
        mindp           - minnimum required number of data points for eikonal operator
        subsize         - subsize of processing list, use to prevent lock in multiprocessing process
        nprocess        - number of processes
        deletetxt       - delete output txt files in working directory
        =================================================================================================================
        """
        # create new eikonal group
        create_group        = False
        while (not create_group):
            try:
                group       = self.create_group( name = 'Eikonal_run_'+str(runid) )
                create_group= True
            except:
                runid       += 1
                continue
        if workingdir is None:
            workingdir      = os.path.dirname(self.filename)+'/eikonal_run_%g' %runid
        datagrp             = self['input_field_data']
        channel             = datagrp.attrs['channel']
        minlon              = self.minlon
        maxlon              = self.maxlon
        minlat              = self.minlat
        maxlat              = self.maxlat
        dlon                = self.dlon
        dlat                = self.dlat
        print ('[%s] [EIKONAL_TOMO] eikonal tomography START' %datetime.now().isoformat().split('.')[0])
        for per in self.pers:
            print ('[%s] [EIKONAL_TOMO] Computing gradients for T = %g sec' %(datetime.now().isoformat().split('.')[0], per))
            grdlst          = []
            dat_per_grp     = datagrp['%g_sec' %per] 
            event_lst       = dat_per_grp.keys()
            working_per     = workingdir+'/'+str(per)+'sec'
            per_group       = group.create_group( name='%g_sec'%( per ) )
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            for evid in event_lst:
                dat_ev_grp  = dat_per_grp[evid]
                numb_points = dat_ev_grp.attrs['num_data_points']
                if numb_points <= mindp:
                    continue
                evlo        = dat_ev_grp.attrs['evlo']
                evla        = dat_ev_grp.attrs['evla']
                lons        = dat_ev_grp['lons'][()]
                lats        = dat_ev_grp['lats'][()]
                dist        = dat_ev_grp['distance'][()]
                C           = dat_ev_grp['phase_velocity'][()]
                gridder     = _grid_class.SphereGridder(minlon = minlon, maxlon = maxlon, dlon = dlon, \
                                minlat = minlat, maxlat = maxlat, dlat = dlat, period = per, \
                                evlo = evlo, evla = evla, fieldtype = 'Tph', evid = evid)
                gridder.read_array(inlons = np.append(evlo, lons), inlats = np.append(evla, lats), inzarr = np.append(0., dist/C))
                grdlst.append(gridder)
            #-----------------------------------------
            # Computing gradient with multiprocessing
            #-----------------------------------------
            if len(grdlst) > subsize:
                Nsub                    = int(len(grdlst)/subsize)
                for isub in range(Nsub):
                    print ('[%s] [EIKONAL_TOMO] subset:' %datetime.now().isoformat().split('.')[0], isub,'in',Nsub,'sets')
                    tmpgrdlst           = grdlst[isub*subsize:(isub+1)*subsize]
                    EIKONAL             = partial(_eikonal_funcs.eikonal_multithread, workingdir = workingdir,\
                                            channel = channel, cdist = cdist)
                    pool                = multiprocessing.Pool(processes = nprocess)
                    pool.map(EIKONAL, tmpgrdlst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                tmpgrdlst               = grdlst[(isub+1)*subsize:]
                EIKONAL                 = partial(_eikonal_funcs.eikonal_multithread, workingdir = workingdir,\
                                                  channel = channel, cdist = cdist)
                pool                    = multiprocessing.Pool(processes = nprocess)
                pool.map(EIKONAL, tmpgrdlst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                print ('[%s] [EIKONAL_TOMO] one set' %datetime.now().isoformat().split('.')[0])
                EIKONAL                 = partial(_eikonal_funcs.eikonal_multithread, workingdir = workingdir,\
                                                  channel = channel, cdist = cdist)
                pool                    = multiprocessing.Pool(processes = nprocess)
                pool.map(EIKONAL, grdlst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            #=============================
            # Read data into hdf5 dataset
            #=============================
            print ('[%s] [EIKONAL_TOMO] loading data' %datetime.now().isoformat().split('.')[0])
            for evid in event_lst:
                infname = working_per+'/'+evid+'_eikonal.npz'
                if not os.path.isfile(infname):
                    if verbose:
                        print ('!!! NO DATA :', evid)
                    continue
                evlo        = dat_ev_grp.attrs['evlo']
                evla        = dat_ev_grp.attrs['evla']
                inarr       = np.load(infname)
                app_vel     = inarr['arr_0']
                reason_n    = inarr['arr_1']
                pro_angle   = inarr['arr_2']
                az          = inarr['arr_3']
                baz         = inarr['arr_4']
                Zarr        = inarr['arr_5']
                Ngrd        = inarr['arr_6']
                event_group = per_group.create_group(name = evid)
                event_group.attrs.create(name = 'evlo', data = evlo)
                event_group.attrs.create(name = 'evla', data = evla)
                event_group.attrs.create(name = 'Ntotal_grd', data = Ngrd[0])
                event_group.attrs.create(name = 'Nvalid_grd', data = Ngrd[1])
                # output datasets
                event_group.create_dataset(name = 'apparent_velocity', data = app_vel)
                event_group.create_dataset(name = 'reason_n', data = reason_n)
                event_group.create_dataset(name = 'propagation_angle', data = pro_angle)
                event_group.create_dataset(name = 'azimuth', data = az)
                event_group.create_dataset(name = 'back_azimuth', data = baz)
                event_group.create_dataset(name = 'travel_time', data = Zarr)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def stack(self, runid = 0, minazi = -180, maxazi = 180, N_bin = 20, threshmeasure = 80, anisotropic = False, \
                spacing_ani = 0.3, coverage = 0.1, azi_amp_tresh = 0.05):
        """stack gradient results to perform Eikonal tomography
        =================================================================================================================
        ::: input parameters :::
        runid           - run id
        minazi/maxazi   - min/max azimuth for anisotropic parameters determination
        N_bin           - number of bins for anisotropic parameters determination
        threshmeasure   - minimum number of measurements required to perform stacking
        spacing_ani     - grid spacing for anisotropic stacking
        anisotropic     - perform anisotropic parameters determination or not
        coverage        - required coverage rate ({Number of valid grid points}/{Number of total grid points})
        =================================================================================================================
        """
        # read attribute information
        minlon          = self.minlon
        maxlon          = self.maxlon
        minlat          = self.minlat
        maxlat          = self.maxlat
        dlon            = self.dlon
        dlat            = self.dlat
        Nlon            = self.Nlon
        Nlat            = self.Nlat
        group           = self['Eikonal_run_'+str(runid)]
        try:
            group_out   = self.create_group( name = 'Eikonal_stack_'+str(runid) )
        except ValueError:
            warnings.warn('Eikonal_stack_'+str(runid)+' exists! Will be recomputed!', UserWarning, stacklevel=1)
            del self['Eikonal_stack_'+str(runid)]
            group_out   = self.create_group( name = 'Eikonal_stack_'+str(runid) )
        # determine anisotropic stacking grid spacing
        if anisotropic:
            grid_factor                 = int(np.ceil(spacing_ani/dlat))
            gridx                       = grid_factor
            gridy                       = grid_factor
            if gridx % 2 == 0:
                gridx                   += 1
            if gridy % 2 == 0:
                gridy                   += 1
            print '--- anisotropic grid factor = '+ str(gridx)+'/'+str(gridy)
            group_out.attrs.create(name = 'gridx', data = gridx)
            group_out.attrs.create(name = 'gridy', data = gridy)
        # attributes for output group
        group_out.attrs.create(name = 'anisotropic', data = anisotropic)
        group_out.attrs.create(name = 'N_bin', data = N_bin)
        group_out.attrs.create(name = 'minazi', data = minazi)
        group_out.attrs.create(name = 'maxazi', data = maxazi)
        for per in self.pers:
            print ('[%s] [EIKONAL_STACK] T = %g sec' %(datetime.now().isoformat().split('.')[0], per))
            per_group   = group['%g_sec'%( per )]
            Nevent      = len(per_group.keys())
            # initialize data arrays 
            Nmeasure    = np.zeros((Nlat, Nlon), dtype = np.int32)
            weightALL   = np.zeros((Nevent, Nlat, Nlon))
            slownessALL = np.zeros((Nevent, Nlat, Nlon))
            aziALL      = np.zeros((Nevent, Nlat, Nlon), dtype='float32')
            reason_nALL = np.zeros((Nevent, Nlat, Nlon))
            validALL    = np.zeros((Nevent, Nlat, Nlon), dtype='float32')
            #-----------------------------------------------------
            # Loop over events to get eikonal maps for each event
            #-----------------------------------------------------
            print ('[%s] [EIKONAL_STACK] reading data' %datetime.now().isoformat().split('.')[0])
            for iev in range(Nevent):
                evid                        = per_group.keys()[iev]
                event_group                 = per_group[evid]
                az                          = event_group['az'][()]
                #-------------------------------------------------
                # get apparent velocities for individual event
                #-------------------------------------------------
                velocity                    = event_group['apparent_velocity'][()]
                reason_n                    = event_group['reason_n'][()]
                oneArr                      = np.ones((Nlat, Nlon), dtype = np.int32)
                oneArr[reason_n!=0]         = 0
                slowness                    = np.zeros((Nlat, Nlon), dtype = np.float32)
                slowness[velocity!=0]       = 1./velocity[velocity!=0]                
                slownessALL[iev, :, :]      = slowness
                reason_nALL[iev, :, :]      = reason_n
                aziALL[iev, :, :]           = az
                Nmeasure                    += oneArr
                # quality control of coverage
                try:
                    Ntotal_grd              = event_group.attrs['Ntotal_grd']
                    Nvalid_grd              = event_group.attrs['Nvalid_grd']
                    if float(Nvalid_grd)/float(Ntotal_grd)< coverage:
                        reason_nALL[iev, :, :]  = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                except:
                    pass
                
            #----------------------------
            # isotropic stacking
            #----------------------------
            print '*** Stacking data'
            if Nmeasure.max()< threshmeasure:
                print ('--- No enough measurements for: '+str(per)+' sec')
                continue
            # discard grid points where number of raw measurements is low, added Sep 26th, 2018
            index_discard                   = Nmeasure < 50
            reason_nALL[:, index_discard]   = 10
            #-----------------------------------------------
            # Get weight for each grid point per event
            #-----------------------------------------------
            if use_numba:
                validALL[reason_nALL==0]    = 1
                weightALL                   = _get_azi_weight(aziALL, validALL)
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            else:
                azi_event1                  = np.broadcast_to(aziALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                azi_event2                  = np.swapaxes(azi_event1, 0, 1)
                validALL[reason_nALL==0]    = 1
                validALL4                   = np.broadcast_to(validALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                # use numexpr for very large array manipulations
                del_aziALL                  = numexpr.evaluate('abs(azi_event1-azi_event2)')
                index_azi                   = numexpr.evaluate('(1*(del_aziALL<20)+1*(del_aziALL>340))*validALL4')
                weightALL                   = numexpr.evaluate('sum(index_azi, 0)')
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            #-----------------------------------------------
            # reduce large weight to some value.
            #-----------------------------------------------
            avgArr                          = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            avgArr[Nmeasure!=0]             = weightsum[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            # bug fixed, 02/07/2018
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            stdArr                          = np.sum( signALL*(weightALL-avgArr)**2, axis=0)
            stdArr[Nmeasure!=0]             = stdArr[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            stdArr                          = np.sqrt(stdArr)
            threshhold                      = np.broadcast_to(avgArr+3.*stdArr, weightALL.shape)
            weightALL[weightALL>threshhold] = threshhold[weightALL>threshhold] # threshhold truncated weightALL
            # recompute weight arrays after large weight value reduction
            weightsum                       = np.sum(weightALL, axis=0)
            weightsumALL                    = np.broadcast_to(weightsum, weightALL.shape)
            # weight over all events, note that before this, weightALL is weight over events in azimuth bin
            weightALL[weightsumALL!=0]      = weightALL[weightsumALL!=0]/weightsumALL[weightsumALL!=0] 
            ###
            weightALL[weightALL==1.]        = 0. # data will be discarded if no other data within 20 degree
            #-----------------------------------------------
            # Compute mean/std of slowness
            #-----------------------------------------------
            slownessALL2                    = slownessALL*weightALL
            slowness_sum                    = np.sum(slownessALL2, axis=0)
            slowness_sumALL                 = np.broadcast_to(slowness_sum, weightALL.shape)
            # weighted standard deviation
            # formula: https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            MArr                            = np.sum(signALL, axis=0)
            temp                            = weightALL*(slownessALL-slowness_sumALL)**2
            temp                            = np.sum(temp, axis=0)
            slowness_std                    = np.zeros(temp.shape)
            tind                            = (weightsum!=0)*(MArr!=1)*(MArr!=0)
            slowness_std[tind]              = np.sqrt(temp[tind]/ ( weightsum[tind]*(MArr[tind]-1)/MArr[tind] ) )
            slowness_stdALL                 = np.broadcast_to(slowness_std, weightALL.shape)
            #-----------------------------------------------
            # discard outliers of slowness
            #-----------------------------------------------
            weightALLQC                     = weightALL.copy()
            index_outlier                   = (np.abs(slownessALL-slowness_sumALL))>2.*slowness_stdALL
            index_outlier                   += reason_nALL != 0
            weightALLQC[index_outlier]      = 0
            weightsumQC                     = np.sum(weightALLQC, axis=0)
            NmALL                           = np.sign(weightALLQC)
            NmeasureQC                      = np.sum(NmALL, axis=0)
            weightsumQCALL                  = np.broadcast_to(weightsumQC, weightALL.shape)
            weightALLQC[weightsumQCALL!=0]  = weightALLQC[weightsumQCALL!=0]/weightsumQCALL[weightsumQCALL!=0]
            temp                            = weightALLQC*slownessALL
            slowness_sumQC                  = np.sum(temp, axis=0)
            # new
            signALLQC                       = weightALLQC.copy()
            signALLQC[signALLQC!=0]         = 1.
            MArrQC                          = np.sum(signALLQC, axis=0)
            temp                            = weightALLQC*(slownessALL-slowness_sumQC)**2
            temp                            = np.sum(temp, axis=0)
            slowness_stdQC                  = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            slowness_stdQC[tind]            = np.sqrt(temp[tind]/ ( weightsumQC[tind]*(MArrQC[tind]-1)/MArrQC[tind] ))
            #---------------------------------------------------------------
            # mask, velocity, and sem arrays of shape Nlat, Nlon
            #---------------------------------------------------------------
            mask                            = np.ones((Nlat, Nlon), dtype=np.bool)
            tempmask                        = (weightsumQC == 0)
            mask[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                                            = tempmask
            vel_iso                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            tempvel                         = slowness_sumQC.copy()
            tempvel[tempvel!=0]             = 1./ tempvel[tempvel!=0]
            vel_iso[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempvel
            #----------------------------------------------------------------------------------------
            # standard error of the mean, updated on 09/20/2018
            # formula: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Statistical_properties
            #----------------------------------------------------------------------------------------
            slownessALL_temp                = slownessALL.copy()
            slownessALL_temp[slownessALL_temp==0.]\
                                            = 0.3
            if np.any(weightALLQC[slownessALL==0.]> 0.):
                raise ValueError('Check weight array!')
            temp                            = (weightALLQC*(1./slownessALL_temp-tempvel))**2
            temp                            = np.sum(temp, axis=0)
            tempsem                         = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            tempsem[tind]                   = np.sqrt( temp[tind] * ( MArrQC[tind]/(weightsumQC[tind])**2/(MArrQC[tind]-1) ) ) 
            vel_sem                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            vel_sem[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempsem
            #---------------------------------------
            # save isotropic velocity to database
            #---------------------------------------
            per_group_out                   = group_out.create_group( name='%g_sec'%( per ) )
            sdset                           = per_group_out.create_dataset(name='slowness', data=slowness_sumQC)
            s_stddset                       = per_group_out.create_dataset(name='slowness_std', data=slowness_stdQC)
            Nmdset                          = per_group_out.create_dataset(name='Nmeasure', data=Nmeasure)
            NmQCdset                        = per_group_out.create_dataset(name='NmeasureQC', data=NmeasureQC)
            maskdset                        = per_group_out.create_dataset(name='mask', data=mask)
            visodset                        = per_group_out.create_dataset(name='vel_iso', data=vel_iso)
            vsemdset                        = per_group_out.create_dataset(name='vel_sem', data=vel_sem)
            #----------------------------------------------------------------------------
            # determine anisotropic parameters, need benchmark and further verification
            #----------------------------------------------------------------------------
            if anisotropic:
                # quality control
                slowness_sumQC_ALL          = np.broadcast_to(slowness_sumQC, slownessALL.shape)
                # # # slowness_stdQC_ALL          = np.broadcast_to(slowness_stdQC, slownessALL.shape)
                # # # index_outlier               = (np.abs(slownessALL-slowness_sumQC_ALL))>2.*slowness_stdQC_ALL
                diff_slowness               = np.abs(slownessALL-slowness_sumQC_ALL)
                ind_nonzero                 = slowness_sumQC_ALL!= 0.
                diff_slowness[ind_nonzero]  = diff_slowness[ind_nonzero]/slowness_sumQC_ALL[ind_nonzero]
                index_outlier               += diff_slowness > azi_amp_tresh
                # stacking to get anisotropic parameters
                dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni    \
                                            = _anisotropic_stacking_parallel(np.int32(gridx), np.int32(gridy), np.float32(maxazi), np.float32(minazi),\
                                                np.int32(N_bin), np.float64(Nmeasure), np.float64(aziALL),\
                                                np.float64(slowness_sumQC), np.float64(slownessALL), index_outlier.astype(bool))
                #----------------------------
                # save data to database
                #----------------------------
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = dslow_sum_ani
                s_anidset       = per_group_out.create_dataset(name='slownessAni', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = dslow_un
                s_anisemdset    = per_group_out.create_dataset(name='slownessAni_sem', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = vel_un
                v_anisemdset    = per_group_out.create_dataset(name='velAni_sem', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = histArr
                histdset        = per_group_out.create_dataset(name='histArr', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr         = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[(gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = NmeasureAni
                NmAnidset       = per_group_out.create_dataset(name='NmeasureAni', data=out_arr)
        return
    
    
    