# -*- coding: utf-8 -*-
"""
hdf5 for noise eikonal tomography
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.eikonal.tomobase as tomobase
import surfpy.eikonal._eikonal_funcs as _eikonal_funcs
import surfpy.eikonal._grid_class as _grid_class

import numpy as np
from uncertainties import ufloat
import uncertainties.umath
from functools import partial
import multiprocessing
import obspy
from datetime import datetime
import warnings
import shutil
import glob
import sys
import copy
import os


class runh5(tomobase.baseh5):
    
    def run(self, workingdir = None, lambda_factor = 3., snr_thresh = 15., runid = 0, nearneighbor = 1, cdist = 250.,\
            mindp = 10, c2_use_c3 = True, c3_use_c2 = False, thresh_borrow = 0.8, deletetxt = True, verbose = False):
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
                group       = self.create_group( name = 'tomo_run_'+str(runid) )
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
        group.attrs.create(name = 'tomography_type', data = 'eikonal')
        print ('[%s] [EIKONAL_TOMO] eikonal tomography START' %datetime.now().isoformat().split('.')[0])
        for per in self.pers:
            print ('[%s] [EIKONAL_TOMO] Computing gradients for T = %g sec' %(datetime.now().isoformat().split('.')[0], per))
            dat_per_grp     = datagrp['%g_sec' %per] 
            event_lst       = list(dat_per_grp.keys())
            working_per     = workingdir+'/'+str(per)+'sec'
            per_group       = group.create_group( name='%g_sec'%( per ) )
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            for evid in event_lst:
                if evid[-3:] == '_C3':
                    ic2c3   = 2
                else:
                    ic2c3   = 1
                dat_ev_grp  = dat_per_grp[evid]
                numb_points = dat_ev_grp.attrs['num_data_points']
                if numb_points <= mindp:
                    continue
                evlo        = dat_ev_grp.attrs['evlo']
                evla        = dat_ev_grp.attrs['evla']
                lons        = dat_ev_grp['lons'][()]
                lats        = dat_ev_grp['lats'][()]
                ind_inbound = (lats >= self.minlat)*(lats <= self.maxlat)*(lons >= self.minlon)*(lons <= self.maxlon)
                #=================================================================
                # check number of data points borrowed from xcorr/C3 to C3/xcorr
                #=================================================================
                ind_borrow  = dat_ev_grp['index_borrow'][()]
                if len(ind_borrow[ind_inbound]) == 0:
                    continue
                borrow_percentage   = (ind_borrow[ind_inbound]).sum()/(ind_borrow[ind_inbound]).size
                # use borrowed data or not
                if borrow_percentage > thresh_borrow or (ic2c3 == 1 and (not c2_use_c3))\
                    or (ic2c3 == 2 and (not c3_use_c2)):
                    ind_dat = np.logical_not(ind_borrow.astype(bool))
                    use_all = False
                else:
                    use_all = True
                if use_all:
                    numb_points = np.where(ind_inbound)[0].size
                else:
                    numb_points = np.where(ind_inbound*ind_dat)[0].size
                if numb_points <= mindp:
                    continue
                dist        = dat_ev_grp['distance'][()]
                C           = dat_ev_grp['phase_velocity'][()]
                if not use_all:
                    lons    = lons[ind_dat]
                    lats    = lats[ind_dat]
                    dist    = dist[ind_dat]
                    C       = C[ind_dat]
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
                gridder.eikonal(workingdir = working_per, inpfx = prefix, nearneighbor = nearneighbor, cdist = cdist)
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
    
    def runMP(self, workingdir = None, lambda_factor = 3., snr_thresh = 10., runid = 0, nearneighbor = 1, cdist = 250.,
            mindp = 10, c2_use_c3 = True, c3_use_c2 = False, thresh_borrow = 0.8, subsize = 1000, nprocess = None,\
            deletetxt = True, verbose = False):
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
                group       = self.create_group( name = 'tomo_run_'+str(runid) )
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
        group.attrs.create(name = 'tomography_type', data = 'eikonal')
        print ('[%s] [EIKONAL_TOMO] eikonal tomography START' %datetime.now().isoformat().split('.')[0])
        for per in self.pers:
            print ('[%s] [EIKONAL_TOMO] Computing gradients for T = %g sec' %(datetime.now().isoformat().split('.')[0], per))
            grdlst          = []
            dat_per_grp     = datagrp['%g_sec' %per] 
            event_lst       = list(dat_per_grp.keys())
            working_per     = workingdir+'/'+str(per)+'sec'
            per_group       = group.create_group( name='%g_sec'%( per ) )
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            for evid in event_lst:
                if evid[-3:] == '_C3':
                    ic2c3   = 2
                else:
                    ic2c3   = 1
                dat_ev_grp  = dat_per_grp[evid]
                numb_points = dat_ev_grp.attrs['num_data_points']
                if numb_points <= mindp:
                    continue
                evlo        = dat_ev_grp.attrs['evlo']
                evla        = dat_ev_grp.attrs['evla']
                lons        = dat_ev_grp['lons'][()]
                lats        = dat_ev_grp['lats'][()]
                ind_inbound = (lats >= self.minlat)*(lats <= self.maxlat)*(lons >= self.minlon)*(lons <= self.maxlon)
                #=================================================================
                # check number of data points borrowed from xcorr/C3 to C3/xcorr
                #=================================================================
                ind_borrow  = dat_ev_grp['index_borrow'][()]
                if len(ind_borrow[ind_inbound]) == 0:
                    continue
                borrow_percentage   = (ind_borrow[ind_inbound]).sum()/(ind_borrow[ind_inbound]).size
                # use borrowed data or not
                if borrow_percentage > thresh_borrow or (ic2c3 == 1 and (not c2_use_c3))\
                    or (ic2c3 == 2 and (not c3_use_c2)):
                    ind_dat = np.logical_not(ind_borrow.astype(bool))
                    use_all = False
                else:
                    use_all = True
                if use_all:
                    numb_points = np.where(ind_inbound)[0].size
                else:
                    numb_points = np.where(ind_inbound*ind_dat)[0].size
                if numb_points <= mindp:
                    continue
                dist        = dat_ev_grp['distance'][()]
                C           = dat_ev_grp['phase_velocity'][()]
                if not use_all:
                    lons    = lons[ind_dat]
                    lats    = lats[ind_dat]
                    dist    = dist[ind_dat]
                    C       = C[ind_dat]
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
                                            channel = channel, nearneighbor = nearneighbor, cdist = cdist)
                    pool                = multiprocessing.Pool(processes = nprocess)
                    pool.map(EIKONAL, tmpgrdlst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                tmpgrdlst               = grdlst[(isub+1)*subsize:]
                EIKONAL                 = partial(_eikonal_funcs.eikonal_multithread, workingdir = workingdir,\
                                            channel = channel, nearneighbor = nearneighbor, cdist = cdist)
                pool                    = multiprocessing.Pool(processes = nprocess)
                pool.map(EIKONAL, tmpgrdlst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                print ('[%s] [EIKONAL_TOMO] one set' %datetime.now().isoformat().split('.')[0])
                EIKONAL                 = partial(_eikonal_funcs.eikonal_multithread, workingdir = workingdir,\
                                            channel = channel, nearneighbor = nearneighbor, cdist = cdist)
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
    
    def stack(self, runid = 0, minazi = -180, maxazi = 180, N_bin = 20, threshmeasure = 50, anisotropic = False, \
                spacing_ani = 0.3, coverage = 0.1, azi_amp_tresh = 0.05, parallel = True):
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
        azi_amp_tresh   - threshhold value of azimuthal anisotropic amplitude, dimentionless (0.05 = 5 %)
        parallel        - run the anisotropic stacking in parallel or not, ONLY takes effect when anisotropic = True
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
        group           = self['tomo_run_'+str(runid)]
        try:
            group_out   = self.create_group( name = 'tomo_stack_'+str(runid) )
        except ValueError:
            warnings.warn('tomo_stack_'+str(runid)+' exists! Will be recomputed!', UserWarning, stacklevel=1)
            del self['tomo_stack_'+str(runid)]
            group_out   = self.create_group( name = 'tomo_stack_'+str(runid) )
        # determine anisotropic stacking grid spacing
        if anisotropic:
            grid_factor                 = int(np.ceil(spacing_ani/dlat))
            grid_lat                    = grid_factor
            grid_lon                    = grid_factor
            if grid_lat % 2 == 0:
                grid_lat                += 1
            if grid_lon % 2 == 0:
                grid_lon                += 1
            grid_lon                    = np.int32(grid_lon)
            grid_lat                    = np.int32(grid_lat)
            print ('--- anisotropic grid factor = '+ str(grid_lon)+'/'+str(grid_lat))
            group_out.attrs.create(name = 'anisotropic_grid_lon', data = grid_lon)
            group_out.attrs.create(name = 'anisotropic_grid_lat', data = grid_lat)
        # attributes for output group
        group_out.attrs.create(name = 'anisotropic', data = anisotropic)
        group_out.attrs.create(name = 'N_bin', data = N_bin)
        group_out.attrs.create(name = 'minazi', data = minazi)
        group_out.attrs.create(name = 'maxazi', data = maxazi)
        for per in self.pers:
            print ('[%s] [EIKONAL_STACK] T = %g sec' %(datetime.now().isoformat().split('.')[0], per))
            per_group   = group['%g_sec'%( per )]
            Nevent      = len(list(per_group.keys()))
            # initialize data arrays 
            Nmeasure    = np.zeros((Nlat, Nlon), dtype = np.int32)
            weightALL   = np.zeros((Nevent, Nlat, Nlon))
            slownessALL = np.zeros((Nevent, Nlat, Nlon))
            aziALL      = np.zeros((Nevent, Nlat, Nlon), dtype = 'float32')
            reason_nALL = np.zeros((Nevent, Nlat, Nlon), dtype = np.int32)
            validALL    = np.zeros((Nevent, Nlat, Nlon), dtype = 'float32')
            event_lst   = list(per_group.keys())
            #-----------------------------------------------------
            # Loop over events to get eikonal maps for each event
            #-----------------------------------------------------
            print ('[%s] [EIKONAL_STACK] reading data' %datetime.now().isoformat().split('.')[0])
            iev         = 0
            for evid in event_lst:
                event_group             = per_group[evid]
                az                      = event_group['azimuth'][()]
                #-------------------------------------------------
                # get apparent velocities for individual event
                #-------------------------------------------------
                velocity                = event_group['apparent_velocity'][()]
                reason_n                = event_group['reason_n'][()]
                oneArr                  = np.ones((Nlat, Nlon), dtype = np.int32)
                oneArr[reason_n!=0]     = 0
                slowness                = np.zeros((Nlat, Nlon), dtype = np.float32)
                slowness[velocity!=0]   = 1./velocity[velocity!=0]                
                slownessALL[iev, :, :]  = slowness
                reason_nALL[iev, :, :]  = reason_n
                aziALL[iev, :, :]       = az
                Nmeasure                += oneArr
                # quality control of coverage
                Ntotal_grd              = event_group.attrs['Ntotal_grd']
                Nvalid_grd              = event_group.attrs['Nvalid_grd']
                if float(Nvalid_grd)/float(Ntotal_grd) < coverage:
                    reason_nALL[iev, :, :]  = np.ones((Nlat, Nlon))
                iev                     += 1
            #====================
            # isotropic stacking
            #====================
            print ('[%s] [EIKONAL_STACK] isotropic stack data' %datetime.now().isoformat().split('.')[0])
            if Nmeasure.max()< threshmeasure:
                print ('!!! NO ENOUGH MEASUREMENTS T = '+str(per)+' sec')
                continue
            # discard grid points where number of raw measurements is low
            index_discard                   = Nmeasure < 50
            reason_nALL[:, index_discard]   = 10
            #-----------------------------------------------
            # Get weight for each grid point per event
            #-----------------------------------------------
            validALL[reason_nALL == 0]      = 1
            weightALL                       = _eikonal_funcs._get_azi_weight(aziALL, validALL)
            weightALL[reason_nALL != 0]     = 0
            weightALL[weightALL != 0]       = 1./weightALL[weightALL != 0]
            weightsum                       = np.sum(weightALL, axis=0)
            #-----------------------------------------------
            # reduce large weight to some value.
            #-----------------------------------------------
            avgArr                          = np.zeros((Nlat, Nlon))
            avgArr[Nmeasure!=0]             = weightsum[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
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
            index_outlier                   = (np.abs(slownessALL - slowness_sumALL))>2.*slowness_stdALL
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
            mask                            = (weightsumQC == 0)
            tempvel                         = slowness_sumQC.copy()
            tempvel[tempvel!=0]             = 1./ tempvel[tempvel!=0]
            vel_iso                         = tempvel.copy()
            #----------------------------------------------------------------------------------------
            # standard error of the mean
            # formula: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Statistical_properties
            #----------------------------------------------------------------------------------------
            slownessALL_temp                = slownessALL.copy()
            slownessALL_temp[slownessALL_temp==0.]\
                                            = 0.3
            if np.any(weightALLQC[slownessALL==0.]> 0.): # debug
                raise ValueError('Check weight array!')
            temp                            = (weightALLQC*(1./slownessALL_temp - vel_iso))**2
            temp                            = np.sum(temp, axis=0)
            tempsem                         = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            tempsem[tind]                   = np.sqrt( temp[tind] * ( MArrQC[tind]/(weightsumQC[tind])**2/(MArrQC[tind]-1) ) ) 
            vel_sem                         = tempsem.copy()
            #=======================================
            # save isotropic velocity to database
            #=======================================
            per_group_out                   = group_out.create_group( name='%g_sec'%( per ) )
            per_group_out.create_dataset(name = 'slowness', data = slowness_sumQC)
            per_group_out.create_dataset(name = 'slowness_std', data = slowness_stdQC)
            per_group_out.create_dataset(name = 'Nmeasure', data = Nmeasure)
            per_group_out.create_dataset(name = 'NmeasureQC', data = NmeasureQC)
            per_group_out.create_dataset(name = 'mask', data = mask)
            per_group_out.create_dataset(name = 'vel_iso', data = vel_iso)
            per_group_out.create_dataset(name = 'vel_sem', data = vel_sem)
            #=====================================
            # determine anisotropic parameters
            #=====================================
            if anisotropic:
                print ('[%s] [EIKONAL_STACK] anisotropic stack data' %datetime.now().isoformat().split('.')[0])
                # quality control
                slowness_sumQC_ALL          = np.broadcast_to(slowness_sumQC, slownessALL.shape)
                # # # slowness_stdQC_ALL          = np.broadcast_to(slowness_stdQC, slownessALL.shape)
                # # # index_outlier               = (np.abs(slownessALL-slowness_sumQC_ALL))>2.*slowness_stdQC_ALL
                diff_slowness               = np.abs(slownessALL-slowness_sumQC_ALL)
                ind_nonzero                 = slowness_sumQC_ALL!= 0.
                diff_slowness[ind_nonzero]  = diff_slowness[ind_nonzero]/slowness_sumQC_ALL[ind_nonzero]
                index_outlier               += diff_slowness > azi_amp_tresh
                # stacking to get anisotropic parameters
                # NOTE: grid_lat and grid_lon are considerred as gridx and gridy instead in _anisotropic_stacking
                if parallel:
                    dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni    \
                                            = _eikonal_funcs._anisotropic_stacking_parallel(np.int32(grid_lat), np.int32(grid_lon),\
                                                np.float32(maxazi), np.float32(minazi), np.int32(N_bin), np.float64(Nmeasure),\
                                                np.float64(aziALL), np.float64(slowness_sumQC), np.float64(slownessALL), index_outlier.astype(bool))
                    # benchmark
                    # # # a1, a2, a3, a4, a5    \
                    # # #                         = _eikonal_funcs._anisotropic_stacking(np.int32(grid_lat), np.int32(grid_lon),\
                    # # #                             np.float32(maxazi), np.float32(minazi), np.int32(N_bin), np.float64(Nmeasure),\
                    # # #                             np.float64(aziALL), np.float64(slowness_sumQC), np.float64(slownessALL), index_outlier.astype(bool))
                    # # # print (np.allclose(a1, dslow_sum_ani))
                    # # # print (np.allclose(a2, dslow_un))
                    # # # print (np.allclose(a3, vel_un))
                    # # # print (np.allclose(a4, histArr))
                    # # # print (np.allclose(a5, NmeasureAni))
                else:
                    dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni    \
                                            = _eikonal_funcs._anisotropic_stacking(np.int32(grid_lat), np.int32(grid_lon),\
                                                np.float32(maxazi), np.float32(minazi), np.int32(N_bin), np.float64(Nmeasure),\
                                                np.float64(aziALL), np.float64(slowness_sumQC), np.float64(slownessALL), index_outlier.astype(bool))
                #=======================
                # save data to database
                #=======================
                ilat            = np.int32((grid_lat - 1)/2)
                ilon            = np.int32((grid_lon - 1)/2)
                # anisotropic slowness
                out_arr         = np.zeros((N_bin, Nlat, Nlon))
                out_arr[:, ilat:-ilat, ilon:-ilon]  = dslow_sum_ani
                per_group_out.create_dataset(name = 'slowness_aniso', data = out_arr)
                # standard error of the mean, anisotropic slowness
                out_arr         = np.zeros((N_bin, Nlat, Nlon))
                out_arr[:, ilat:-ilat, ilon:-ilon]  = dslow_un
                per_group_out.create_dataset(name = 'slowness_aniso_sem', data = out_arr)
                # standard error of the mean, anisotropic velocity
                out_arr         = np.zeros((N_bin, Nlat, Nlon))
                out_arr[:, ilat:-ilat, ilon:-ilon]  = vel_un
                per_group_out.create_dataset(name = 'vel_aniso_sem', data = out_arr)
                # azimuthal binned histogram
                out_arr         = np.zeros((N_bin, Nlat, Nlon))
                out_arr[:, ilat:-ilat, ilon:-ilon]  = histArr
                per_group_out.create_dataset(name = 'hist_aniso', data = out_arr)
                # total number of anisotropic measurements
                out_arr         = np.zeros((Nlat, Nlon))
                out_arr[ilat:-ilat, ilon:-ilon]     = NmeasureAni
                per_group_out.create_dataset(name = 'Nmeasure_aniso', data = out_arr)
        print ('[%s] [EIKONAL_STACK] eikonal stacking ALL DONE' %datetime.now().isoformat().split('.')[0])
        return
    
    def azi_aniso(self, runid = 0,  Ntotal_thresh = None, N_thresh = 5, Nbin_thresh = 5, semfactor = 5.):
        """compute azimuthal anisotropic parameters based on stacked results
        =================================================================================================================
        ::: input parameters :::
        runid           - run id
        Ntotal_thresh   - threshold total number of measurements (all the bins)
        N_thresh        - threshold single bin number of measurements
        Nbin_thresh     - threshold number of bins
        semfactor       - factors for scaling up/down uncertainties (standard error of the mean)
        =================================================================================================================
        """
        dataid          = 'tomo_stack_'+str(runid)
        ingroup         = self[dataid]
        if not ingroup.attrs['anisotropic']:
            raise AttributeError('The stacked result is NOT anisotropic!')
        grid_lon        = ingroup.attrs['anisotropic_grid_lon']
        grid_lat        = ingroup.attrs['anisotropic_grid_lat']
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin_default    = ingroup.attrs['N_bin']
        d_bin           = float((maxazi-minazi)/Nbin_default)
        # # # azArr           = np.arange(Nbin_default)*d_bin + minazi
        azArr           = np.arange(Nbin_default)*d_bin + minazi + d_bin/2.
        if Ntotal_thresh is None:
            Ntotal_thresh   = N_thresh*grid_lon*grid_lat*Nbin_default/2.
        self._get_lon_lat_arr()
        for period in self.pers:
            print ('=== Fitting azimuthal angle and amplitude '+str(period)+' sec')
            pergrp      = ingroup['%g_sec'%( period )]
            mask        = pergrp['mask'][()]
            slowAni     = pergrp['slowness_aniso'][()] + pergrp['slowness'][()]
            velAnisem   = pergrp['vel_aniso_sem'][()] * semfactor
            slowness    = pergrp['slowness'][()]
            histArr     = pergrp['hist_aniso'][()]
            psiarr      = np.zeros((self.Nlat, self.Nlon))
            amparr      = np.zeros((self.Nlat, self.Nlon))
            misfitarr   = np.zeros((self.Nlat, self.Nlon))
            Nbinarr     = np.zeros((self.Nlat, self.Nlon))
            Nmarr       = np.zeros((self.Nlat, self.Nlon))
            mask_aniso  = np.ones((self.Nlat, self.Nlon), dtype = bool)
            #------------------------
            # uncertainty arrays
            #------------------------
            un_psiarr   = np.zeros((self.Nlat, self.Nlon))
            un_amparr   = np.zeros((self.Nlat, self.Nlon))
            for ilat in range(self.Nlat):
                for ilon in range(self.Nlon):
                    if mask[ilat, ilon]:
                        continue
                    outslowness = slowAni[:, ilat, ilon]
                    outvel_sem  = velAnisem[:, ilat, ilon]
                    avg_slowness= slowness[ilat, ilon]
                    out_hist    = histArr[:, ilat, ilon]
                    if out_hist.sum() < Ntotal_thresh:
                        continue
                    # get the valid binned data
                    # quality control
                    index       = np.where((outvel_sem != 0)*(out_hist > N_thresh ))[0]
                    outslowness = outslowness[index]
                    az_grd      = azArr[index]
                    outvel_sem  = outvel_sem[index]
                    Nbin        = index.size
                    # skip if not enough bins
                    if Nbin < Nbin_thresh:
                        continue
                    # normalized number of measurements
                    Nmarr[ilat, ilon]   = int(out_hist.sum()/(grid_lon*grid_lat))
                    Nbinarr[ilat, ilon] = Nbin
                    #=======================================================
                    # fitting the binned data with psi-2 sinusoidal function
                    # TODO: add fit psi-1 functionality
                    #=======================================================
                    # construct forward operator matrix
                    tG                  = np.ones((Nbin, 1), dtype=np.float64)
                    G                   = tG.copy()
                    # convert azimuth to the 'real' azimuth coordinate
                    az_grd              += 180.
                    az_grd              = 360. - az_grd
                    az_grd              -= 90.
                    az_grd[az_grd<0.]   += 360.
                    #--------------------
                    # 2-psi terms
                    #--------------------
                    tbaz                = np.pi*(az_grd)/180.
                    tGsin2              = np.sin(tbaz*2)
                    tGcos2              = np.cos(tbaz*2)
                    G                   = np.append(G, tGsin2)
                    G                   = np.append(G, tGcos2)
                    G                   = G.reshape((3, Nbin))
                    G                   = G.T
                    # data
                    indat               = (1./outslowness).reshape(1, Nbin)
                    d                   = indat.T
                    # data covariance matrix
                    Cd                  = np.zeros((Nbin, Nbin), dtype=np.float64)
                    np.fill_diagonal(Cd, outvel_sem**2)
                    #------------------------------------------
                    # Tarantola's solution, page 67
                    # Example 3.4: eq. 3.40, 3.41
                    #------------------------------------------
                    Ginv1               = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cd)), G) )
                    Ginv2               = np.dot( np.dot(G.T, np.linalg.inv(Cd)), d)
                    model               = np.dot(Ginv1, Ginv2)
                    Cm                  = Ginv1 # model covariance matrix
                    pcov                = np.sqrt(np.absolute(Cm))
                    m0                  = ufloat(model[0][0], pcov[0][0])
                    m1                  = ufloat(model[1][0], pcov[1][1])
                    m2                  = ufloat(model[2][0], pcov[2][2])
                    A0                  = model[0][0]
                    A2                  = np.sqrt(model[1][0]**2 + model[2][0]**2)
                    psi2                = np.arctan2(model[1][0], model[2][0])/2.
                    if psi2 < 0.:
                        psi2            += np.pi # -90 ~ 90 -> 0 ~ 180.
                    # compute misfit
                    predat                  = A0 + A2*np.cos(2.*(np.pi/180.*(az_grd+180.)-psi2) )
                    misfit                  = np.sqrt( ((predat - 1./outslowness)**2 / outvel_sem**2).sum()/ az_grd.size )
                    amparr[ilat, ilon]      = A2/A0*100.  # convert to percentage
                    psiarr[ilat, ilon]      = psi2/np.pi*180.
                    mask_aniso[ilat, ilon]  = False
                    misfitarr[ilat, ilon]   = misfit
                    # uncertainties
                    unA2                    = (uncertainties.umath.sqrt(m1**2+m2**2)/m0*100.).std_dev
                    unpsi2                  = (uncertainties.umath.atan2(m1, m2)/np.pi*180./2.).std_dev
                    unamp                   = unA2
                    unpsi2                  = min(unpsi2, 90.)
                    un_psiarr[ilat, ilon]   = unpsi2
                    un_amparr[ilat, ilon]   = unamp
            #=============================
            # save data to hdf5 datasets
            #=============================
            try:
                pergrp.create_dataset(name = 'amparr', data = amparr)
                pergrp.create_dataset(name = 'psiarr', data = psiarr)
                pergrp.create_dataset(name = 'mask_aniso', data = mask_aniso)
                pergrp.create_dataset(name = 'misfit', data = misfitarr)
                pergrp.create_dataset(name = 'num_total_measurements', data = Nmarr)
                pergrp.create_dataset(name = 'num_bins', data = Nbinarr)
                pergrp.create_dataset(name = 'uncertainty_psi', data = un_psiarr)
                pergrp.create_dataset(name = 'uncertainty_amp', data = un_amparr)
            except:
                del pergrp['amparr']
                del pergrp['psiarr']
                del pergrp['mask_aniso']
                del pergrp['misfit']
                del pergrp['num_total_measurements']
                del pergrp['num_bins']
                del pergrp['uncertainty_psi']
                del pergrp['uncertainty_amp']
                pergrp.create_dataset(name = 'amparr', data = amparr)
                pergrp.create_dataset(name = 'psiarr', data = psiarr)
                pergrp.create_dataset(name = 'mask_aniso', data = mask_aniso)
                pergrp.create_dataset(name = 'misfit', data = misfitarr)
                pergrp.create_dataset(name = 'num_total_measurements', data = Nmarr)
                pergrp.create_dataset(name = 'num_bins', data = Nbinarr)
                pergrp.create_dataset(name = 'uncertainty_psi', data = un_psiarr)
                pergrp.create_dataset(name = 'uncertainty_amp', data = un_amparr)
        return
    
    
    