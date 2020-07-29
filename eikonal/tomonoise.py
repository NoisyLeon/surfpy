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
    
    def load_ASDF(self, in_asdf_fname, channel='ZZ', data_type='FieldDISPpmf2interp', verbose = True):
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
        # create group for input data
        group               = self.create_group( name = 'input_field_data' )
        # input xcorr database
        indbase             = pyasdf.ASDFDataSet(in_asdf_fname)
        event_lst           = indbase.waveforms.list()
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
                    print ('--- Virtual event: '+evid)
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
        # self.update_dat()
        return
    
    # def run(self, staxml = None, netcodelst=[], lambda_factor=3., snr_thresh=15.):
    #     """load travel time field data from ASDF
    #     =================================================================================================================
    #     ::: input parameters :::
    #     in_asdf_fname   - input ASDF data file
    #     staxml          - input StationXML for data selection
    #     netcodelst      - network list for data selection
    #     channel         - channel for analysis (default = ZZ )
    #     data_type       - data type
    #                         default = 'FieldDISPpmf2interp'
    #                             aftan measurements with phase-matched filtering and jump correction
    #     =================================================================================================================
    #     """
    #     if staxml is not None:
    #         inv             = obspy.read_inventory(staxml)
    #         waveformLst     = []
    #         for network in inv:
    #             netcode     = network.code
    #             for station in network:
    #                 stacode = station.code
    #                 waveformLst.append(netcode+'.'+stacode)
    #         staLst          = waveformLst
    #         print ('--- Load stations from input StationXML file')
    #     else:
    #         print ('--- Load all the stations from database')
    #         staLst          = self.waveforms.list()
    #     # network selection
    #     if len(netcodelst) != 0:
    #         staLst_ALL      = copy.deepcopy(staLst)
    #         staLst          = []
    #         for staid in staLst_ALL:
    #             netcode, stacode    = staid.split('.')
    #             if not (netcode in netcodelst):
    #                 continue
    #             staLst.append(staid)
    #         print ('--- Select stations according to network code: '+str(len(staLst))+'/'+str(len(staLst_ALL))+' (selected/all)')
    #     return 
    # 
    
    