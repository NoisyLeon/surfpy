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
        group               = self.create_group( name = 'input_field_data' )
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
                event_group.attrs.create(name = 'num_data_points', data = data.shape[1])
                event_group.create_dataset(name='lons', data = data[:, 0])
                event_group.create_dataset(name='lats', data = data[:, 1])
                event_group.create_dataset(name='phase_velocity', data = data[:, 2])
                event_group.create_dataset(name='group_velocity', data = data[:, 3])
                event_group.create_dataset(name='snr', data = data[:, 4])
                event_group.create_dataset(name='distance', data = data[:, 5])
        return
    
    def run(self, workingdir, lambda_factor=3., snr_thresh=15., runid=0, cdist=150., mindp=10, deletetxt=True, verbose=False):
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
        # # # # input xcorr database
        # # # inDbase             = pyasdf.ASDFDataSet(inasdffname)
        # # # # get header 
        # # # pers                = self.attrs['period_array']
        # # # minlon              = self.attrs['minlon']
        # # # maxlon              = self.attrs['maxlon']
        # # # minlat              = self.attrs['minlat']
        # # # maxlat              = self.attrs['maxlat']
        # # # dlon                = self.attrs['dlon']
        # # # dlat                = self.attrs['dlat']
        # # # nlat_grad           = self.attrs['nlat_grad']
        # # # nlon_grad           = self.attrs['nlon_grad']
        # # # nlat_lplc           = self.attrs['nlat_lplc']
        # # # nlon_lplc           = self.attrs['nlon_lplc']
        # # # fdict               = { 'Tph': 2, 'Tgr': 3}
        # # # evLst               = inDbase.waveforms.list()
        print ('[%s] [EIKONAL_TOMO] eikonal tomography start!' %datetime.now().isoformat().split('.')[0])
        for per in self.pers:
            print ('[%s] [EIKONAL_TOMO] Computing gradients for T = %g sec' %(datetime.now().isoformat().split('.')[0], per))
            
            
            del_per         = per-int(per)
            if del_per==0.:
                persfx      = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                persfx      = str(int(per))+'sec'+dper.split('.')[1]
            working_per     = workingdir+'/'+str(per)+'sec'
            per_group       = group.create_group( name='%g_sec'%( per ) )
            for evid in evLst:
                netcode1, stacode1  = evid.split('.')
                try:
                    subdset         = inDbase.auxiliary_data[data_type][netcode1][stacode1][channel][persfx]
                except KeyError:
                    print ('No travel time field for: '+evid)
                    continue
                if verbose:
                    print ('Event: '+evid)
                lat1, elv1, lon1    = inDbase.waveforms[evid].coordinates.values()
                if lon1<0.:
                    lon1            += 360.
                dataArr             = subdset.data.value
                field2d             = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                        minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=lon1, evla=lat1, fieldtype=fieldtype, \
                                        nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
                Zarr                = dataArr[:, fdict[fieldtype]]
                # skip if not enough data points
                if Zarr.size <= mindp:
                    continue
                distArr             = dataArr[:, 5]
                field2d.read_array(lonArr=np.append(lon1, dataArr[:,0]), latArr=np.append(lat1, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                outfname            = evid+'_'+fieldtype+'_'+channel+'.lst'
                field2d.interp_surface(workingdir=working_per, outfname=outfname)
                field2d.check_curvature(workingdir=working_per, outpfx=evid+'_'+channel+'_')
                field2d.eikonal_operator(workingdir=working_per, inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)
                # save data to hdf5 dataset
                event_group         = per_group.create_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=lon1)
                event_group.attrs.create(name = 'evla', data=lat1)
                # added 04/05/2018
                event_group.attrs.create(name = 'Ntotal_grd', data=field2d.Ntotal_grd)
                event_group.attrs.create(name = 'Nvalid_grd', data=field2d.Nvalid_grd)
                #
                appVdset            = event_group.create_dataset(name='appV', data=field2d.appV)
                reason_ndset        = event_group.create_dataset(name='reason_n', data=field2d.reason_n)
                proAngledset        = event_group.create_dataset(name='proAngle', data=field2d.proAngle)
                azdset              = event_group.create_dataset(name='az', data=field2d.az)
                bazdset             = event_group.create_dataset(name='baz', data=field2d.baz)
                Tdset               = event_group.create_dataset(name='travelT', data=field2d.Zarr)
        if deletetxt:
            shutil.rmtree(workingdir)
        return

    
    