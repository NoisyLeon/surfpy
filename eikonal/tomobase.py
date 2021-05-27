# -*- coding: utf-8 -*-
"""
hdf5 for eikonal tomography base
    
:copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
    
:references:
    Feng, L., & Ritzwoller, M. H. (2017). The effect of sedimentary basins on surface waves that pass through them.
        Geophysical Journal International, 211(1), 572-592.
    Feng, L., & Ritzwoller, M. H. (2019). A 3-D shear velocity model of the crust and uppermost mantle beneath Alaska including apparent radial anisotropy.
        Journal of Geophysical Research: Solid Earth, 124(10), 10468-10497.
    Lin, Fan-Chi, Michael H. Ritzwoller, and Roel Snieder. "Eikonal tomography: surface wave tomography by phase front tracking across a regional broad-band seismic array."
        Geophysical Journal International 177.3 (2009): 1091-1110.
    Lin, Fan-Chi, and Michael H. Ritzwoller. "Helmholtz surface wave tomography for isotropic and azimuthally anisotropic structure."
        Geophysical Journal International 186.3 (2011): 1104-1120.
"""


import surfpy.eikonal._eikonal_funcs as _eikonal_funcs
import surfpy.eikonal._grid_class as _grid_class
from uncertainties import ufloat
import uncertainties
import numpy as np
import numpy.ma as ma
import h5py
import pyasdf
import warnings
import obspy
import shutil
from subprocess import call
import os

import surfpy.cpt_files as cpt_files
cpt_path    = cpt_files.__path__._path[0]

import surfpy.map_dat.glb_ph_vel_maps as MAPS
global_map_path    = MAPS.__path__._path[0]


import shapefile
    
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt

class baseh5(h5py.File):
    """
    """
    def __init__(self, name, mode='a', driver=None, libver=None, userblock_size=None, swmr=False,\
            rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None, **kwds):
        super(baseh5, self).__init__( name, mode, driver, libver, userblock_size,\
            swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order)
        #======================================
        # initializations of attributes
        #======================================
        if self.update_attrs():
            self._get_lon_lat_arr()
        # self.update_dat()
        # try:
        #     self.datapfx    = self.attrs['data_pfx']
            
        # self.inv        = obspy.Inventory()
        # self.start_date = obspy.UTCDateTime('2599-01-01')
        # self.end_date   = obspy.UTCDateTime('1900-01-01')
        # self.update_inv_info()
        return
    
    def _get_lon_lat_arr(self, ncut=0):
        """Get longitude/latitude array
        """
        self.lons   = np.arange((self.maxlon-self.minlon)/self.dlon+1-2*ncut)*self.dlon + self.minlon + ncut*self.dlon
        self.lats   = np.arange((self.maxlat-self.minlat)/self.dlat+1-2*ncut)*self.dlat + self.minlat + ncut*self.dlat
        self.Nlon   = self.lons.size
        self.Nlat   = self.lats.size
        self.lonArr, self.latArr = np.meshgrid(self.lons, self.lats)
        if self.lons[0] != self.minlon or self.lons[-1] != self.maxlon \
            or self.lats[0] != self.minlat or self.lats[-1] != self.maxlat:
            raise ValueError('!!! longitude/latitude arrays not consistent with bounds')
        try:
            self.ilontype   = self.attrs['ilontype']
        except:
            self.ilontype   = 1
        return
    
    def update_attrs(self):
        try:
            self.pers       = self.attrs['period_array']
            self.minlon     = self.attrs['minlon']
            self.maxlon     = self.attrs['maxlon']
            self.minlat     = self.attrs['minlat']
            self.maxlat     = self.attrs['maxlat']
            self.Nlon       = int(self.attrs['Nlon'])
            self.dlon       = self.attrs['dlon']
            self.Nlat       = int(self.attrs['Nlat'])
            self.dlat       = self.attrs['dlat']
            self.proj_name  = self.attrs['proj_name']
            return True
        except:
            return False
        ###
        try:
            self.ilontype   = self.attrs['ilontype']
        except:
            self.ilontype   = 1
    
    def set_input_parameters(self, minlon, maxlon, minlat, maxlat, pers=[], dlon=0.2, dlat=0.2, optimize_spacing=True, proj_name = ''):
        """set input parameters for tomographic inversion.
        =================================================================================================================
        ::: input parameters :::
        minlon, maxlon  - minimum/maximum longitude
        minlat, maxlat  - minimum/maximum latitude
        pers            - period array, default = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        dlon, dlat      - longitude/latitude interval
        optimize_spacing- optimize the grid spacing or not
                            if True, the distance for input dlat/dlon will be calculated and dlat may be changed to
                                make the distance of dlat as close to the distance of dlon as possible
        =================================================================================================================
        """
        if len(pers) == 0:
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        else:
            pers    = np.asarray(pers)
        if minlon < 0.:
            self.attrs.create(name = 'ilontype', data = 0, dtype = np.int32)
            print ('Longitude type will be -180/180 !')
        else:
            self.attrs.create(name = 'ilontype', data = 1, dtype = np.int32)
        self.attrs.create(name = 'period_array', data = pers, dtype = np.float64)
        self.attrs.create(name = 'minlon', data = minlon, dtype = np.float64)
        self.attrs.create(name = 'maxlon', data = maxlon, dtype = np.float64)
        self.attrs.create(name = 'minlat', data = minlat, dtype = np.float64)
        self.attrs.create(name = 'maxlat', data = maxlat, dtype = np.float64)
        
        if optimize_spacing:
            ratio   = _eikonal_funcs.determine_interval(minlat=minlat, maxlat=maxlat, dlon=dlon, dlat = dlat)
            if ratio != 1.:
                print ('----------------------------------------------------------')
                print ('Changed dlat from dlat =',dlat,'to dlat =',dlat/ratio)
                print ('----------------------------------------------------------')
                dlat    = dlat/ratio
        self.attrs.create(name = 'dlon', data = dlon, dtype = np.float64)
        self.attrs.create(name = 'dlat', data = dlat, dtype = np.float64)
        Nlon        = int((maxlon-minlon)/dlon+1)
        Nlat        = int((maxlat-minlat)/dlat+1)
        self.attrs.create(name = 'Nlon', data = Nlon, dtype = np.int64)
        self.attrs.create(name = 'Nlat', data = Nlat, dtype = np.int64)
        self.attrs.create(name = 'proj_name', data = proj_name)
        self.update_attrs()
        self._get_lon_lat_arr()
        return
    #==================================================
    # functions print the information of database
    #==================================================
    def print_attrs(self, print_to_screen=True):
        """print the attrsbute information of the dataset.
        """
        if not self.update_attrs():
            print ('Empty Database!')
            return 
        outstr  = '============================= Surface wave eikonal/Helmholtz tomography database ===========================\n'
        outstr  += 'Project name:                           - '+str(self.proj_name)+'\n'        
        outstr  += 'period(s):                              - '+str(self.pers)+'\n'
        outstr  += 'longitude range                         - '+str(self.minlon)+' ~ '+str(self.maxlon)+'\n'
        outstr  += 'longitude spacing/npts                  - '+str(self.dlon)+'/'+str(self.Nlon)+'\n'
        outstr  += 'latitude range                          - '+str(self.minlat)+' ~ '+str(self.maxlat)+'\n'
        outstr  += 'latitude spacing/npts                   - '+str(self.dlat)+'/'+str(self.Nlat)+'\n'
        if print_to_screen:
            print (outstr)
        else:
            return outstr
        return
    
    def print_info(self, runid=0):
        """print the information of given eikonal/Helmholz run
        """
        outstr      = self.print_attrs(print_to_screen=False)
        if outstr is None:
            return
        outstr      += '============================================ tomo_run_%d ====================================================\n'  %runid 
        subgroup    = self['tomo_run_%d' %runid]
        pers        = self.pers
        perid       = '%d_sec' %pers[0]
        Nevent      = len(list(subgroup[perid].keys()))
        outstr      += '--- tomography type                             - %s\n' %subgroup.attrs['tomography_type']
        outstr      += '--- number of (virtual) events                  - '+str(Nevent)+'\n'
        evid        = list(subgroup[perid].keys())[0]
        evgrp       = subgroup[perid][evid]
        outstr      += '--- attributes for each event                   - evlo, evla, Nvalid_grd, Ntotal_grd \n'
        outstr      += '--- apparent_velocity                           - '+str(evgrp['apparent_velocity'].shape)+'\n'
        try:    
            outstr  += '--- corrected_velocity                          - '+str(evgrp['corrected_velocity'].shape)+'\n'
        except KeyError:
            outstr  += '*** NO corrected velocity \n'
        try:    
            outstr  += '--- lplc_amp (amplitude Laplacian)              - '+str(evgrp['lplc_amp'].shape)+'\n'
        except KeyError:
            outstr  += '*** NO corrected lplc_amp \n'
        outstr      += '--- azimuth                                     - '+str(evgrp['azimuth'].shape)+'\n'
        outstr      += '--- back_azimuth                                - '+str(evgrp['back_azimuth'].shape)+'\n'
        outstr      += '--- propagation_angle                           - '+str(evgrp['propagation_angle'].shape)+'\n'
        outstr      += '--- travel_time                                 - '+str(evgrp['travel_time'].shape)+'\n'
        outstr      += '--- reason_n (index array)                      - '+str(evgrp['reason_n'].shape)+'\n'
        outstr      += '        0: accepted point \n' + \
                       '        1: data point the has large difference between v1HD and v1HD02 \n' + \
                       '        2: data point that does not have near neighbor points at all E/W/N/S directions\n' + \
                       '        3: slowness is too large/small \n' + \
                       '        4: near a zero field data point \n' + \
                       '        5: epicentral distance is too small \n' + \
                       '        6: large curvature              \n'
        try:
            outstr  += '--- reason_n_helm (index array for Helmoltz)    - '+str(evgrp['reason_n_helm'].shape)+'\n'
            outstr  += '        0 ~ 6: same as above \n' + \
                       '        7: reason_n of amplitude field is non-zero (invalid) \n' + \
                       '        8: negative phase slowness after correction \n'
        except KeyError:
            outstr  += '*** NO reason_n_helm \n'
        
        try:
            subgroup= self['tomo_stack_%d' %runid]
            outstr  += '============================================ tomo_stack_%d ==================================================\n'  %runid 
        except KeyError:
            outstr  += '======================================== NO corresponding stacked results ===================================\n'
            return
        if subgroup.attrs['anisotropic']:
            tempstr = 'anisotropic'
            outstr  += '--- isotropic/anisotropic                           - '+tempstr+'\n'
            outstr  += '--- anisotropic grid factor in lon/lat              - %g/%g\n' %(subgroup.attrs['anisotropic_grid_lon'], subgroup.attrs['anisotropic_grid_lat'])
            outstr  += '--- N_bin (number of bins, for ani run)             - '+str(subgroup.attrs['N_bin'])+'\n'
            outstr  += '--- minazi/maxazi (min/max azi, for ani run)        - '+str(subgroup.attrs['minazi'])+'/'+str(subgroup.attrs['maxazi'])+'\n'
        else:
            tempstr = 'isotropic'
        pergrp      = subgroup[perid]
        outstr      += '--- Nmeasure (number of raw measurements)           - '+str(pergrp['Nmeasure'].shape)+'\n'
        outstr      += '--- NmeasureQC (number of qc measurements)          - '+str(pergrp['NmeasureQC'].shape)+'\n'
        outstr      += '--- slowness                                        - '+str(pergrp['slowness'].shape)+'\n'
        outstr      += '--- slowness_std                                    - '+str(pergrp['slowness_std'].shape)+'\n'
        outstr      += '--- mask                                            - '+str(pergrp['mask'].shape)+'\n'
        outstr      += '--- vel_iso (isotropic velocity)                    - '+str(pergrp['vel_iso'].shape)+'\n'
        outstr      += '--- vel_sem (uncertainties for velocity)            - '+str(pergrp['vel_sem'].shape)+'\n'
        if subgroup.attrs['anisotropic']:
            outstr  += '--- Nmeasure_aniso (number of aniso measurements)   - '+str(pergrp['Nmeasure_aniso'].shape)+'\n'
            outstr  += '--- hist_aniso (number of binned measurements)      - '+str(pergrp['hist_aniso'].shape)+'\n'
            outstr  += '--- slowness_aniso (aniso perturbation in slowness) - '+str(pergrp['slowness_aniso'].shape)+'\n'
            outstr  += '--- slowness_aniso_sem (sem in slownessAni)         - '+str(pergrp['slowness_aniso_sem'].shape)+'\n'
            outstr  += '--- vel_aniso_sem (sem in binned velocity)          - '+str(pergrp['vel_aniso_sem'].shape)+'\n'
        print (outstr)
        return
    
    def load_ASDF_noise(self, in_asdf_fname, channel='ZZ', data_type='FieldDISPpmf2interp', \
            staxml = None, netcodelst=[], verbose = False):
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
        if 'C3' in data_type:
            ic2c3   = 2
        else:
            ic2c3   = 1
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
        group               = self.require_group( name = 'input_field_data')
        group.attrs.create(name = 'channel', data = channel[1])
        # loop over periods
        for per in self.pers:
            print ('--- loading data for: '+str(per)+' sec')
            del_per         = per - int(per)
            if del_per==0.:
                per_name    = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                per_name    = str(int(per))+'sec'+dper.split('.')[1]
            per_group       = group.require_group(name = '%g_sec' %per)
            for evid in event_lst:
                if ic2c3 == 1:
                    evid_out= evid
                else:
                    evid_out= evid + '_C3'
                # check existence:
                if evid_out in per_group.keys():
                    if verbose:
                        print ('--- virtual event exists: '+evid_out)
                    continue
                netcode1, stacode1  = evid.split('.')
                ###
                if len(netcode1) == 1:
                    netcode1    += '0'
                ###
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        subdset     = indbase.auxiliary_data[data_type][netcode1][stacode1][channel][per_name]
                except KeyError:
                    print ('!!! No travel time field for: '+evid_out)
                    continue
                if verbose:
                    print ('--- virtual event: '+evid_out)
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
                    data            = subdset.data[()]
                # save data to hdf5 dataset
                event_group         = per_group.create_group(name = evid_out)
                event_group.attrs.create(name = 'evlo', data = lon1)
                event_group.attrs.create(name = 'evla', data = lat1)
                event_group.attrs.create(name = 'num_data_points', data = data.shape[0])
                event_group.create_dataset(name='lons', data = data[:, 0])
                event_group.create_dataset(name='lats', data = data[:, 1])
                event_group.create_dataset(name='phase_velocity', data = data[:, 2])
                event_group.create_dataset(name='group_velocity', data = data[:, 3])
                event_group.create_dataset(name='snr', data = data[:, 4])
                event_group.create_dataset(name='distance', data = data[:, 5])
                event_group.create_dataset(name='index_borrow', data = data[:, 6])
        return
    
    def load_ASDF_quake(self, in_asdf_fname, channel='Z', data_type='FieldDISPpmf2interp', verbose = False):
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
        indbase             = pyasdf.ASDFDataSet(in_asdf_fname)
        cat                 = indbase.events
        # create group for input data
        group               = self.require_group( name = 'input_field_data')
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
            per_group       = group.require_group(name = '%g_sec' %per)
            for event in cat:
                Ndata           = 0
                outstr          = ''
                otime           = event.origins[0].time
                event_id        = event.resource_id.id.split('=')[-1]
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                magnitude       = event.magnitudes[0].mag
                Mtype           = event.magnitudes[0].magnitude_type
                timestr         = otime.isoformat()
                evlo            = event.origins[0].longitude
                evla            = event.origins[0].latitude
                evdp            = event.origins[0].depth
                oyear           = otime.year
                omonth          = otime.month
                oday            = otime.day
                ohour           = otime.hour
                omin            = otime.minute
                osec            = otime.second
                label           = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
                event_tag       = 'surf_'+label
                dataid          = event_tag + '_'+channel
                # check existence:event_tag
                if event_tag in per_group.keys():
                    if verbose:
                        print ('--- virtual event exists: ' + dataid)
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        subdset     = indbase.auxiliary_data[data_type][dataid][per_name]
                except KeyError:
                    print ('!!! No travel time field for: '+ dataid)
                    continue
                if verbose:
                    print ('--- virtual event: '+dataid)
                if evlo < 0.:
                    evlo            += 360.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data            = subdset.data[()]
                # save data to hdf5 dataset
                event_group         = per_group.create_group(name = event_tag)
                event_group.attrs.create(name = 'evlo', data = evlo)
                event_group.attrs.create(name = 'evla', data = evla)
                event_group.attrs.create(name = 'num_data_points', data = data.shape[0])
                event_group.create_dataset(name='lons', data = data[:, 0])
                event_group.create_dataset(name='lats', data = data[:, 1])
                event_group.create_dataset(name='phase_velocity', data = data[:, 2])
                event_group.create_dataset(name='group_velocity', data = data[:, 3])
                event_group.create_dataset(name='snr', data = data[:, 4])
                event_group.create_dataset(name='distance', data = data[:, 5])
                event_group.create_dataset(name='amplitude', data = data[:, 6])
        return
    
    
    def get_mask(self, runid = 0, Tmin = -999, Tmax = 999, ntrim = 0, nmeasure = 0, nmeasure_aniso = 0):
        """get mask for all periods
        """
        dataid          = 'tomo_stack_'+str(runid)
        ingroup         = self[dataid]
        mask            = np.zeros((self.Nlat, self.Nlon), dtype = bool)
        if ingroup.attrs['anisotropic']:
            is_aniso    = True
            mask_aniso  = np.zeros((self.Nlat, self.Nlon), dtype = bool)
        else:
            is_aniso    = False
        print ('Anisotropic mask')
        self._get_lon_lat_arr()
        for period in self.pers:
            if period < Tmin or period > Tmax:
                continue
            try:
                pergrp      = ingroup['%g_sec'%( period )]
            except KeyError:
                continue    
            mask            += pergrp['mask'][()]
            if nmeasure > 0:
                narr        = pergrp['NmeasureQC'][()]
                mask        += (narr < nmeasure)
            if is_aniso:
                mask_aniso  += pergrp['mask_aniso'][()]
                if nmeasure_aniso > 0:
                    narr_aniso  = pergrp['Nmeasure_aniso'][()]
                    mask_aniso  += (narr_aniso < nmeasure_aniso)
        tmp = _grid_class._check_neighbor_val_2(np.int32(mask), np.float64(mask), np.int32(1), np.float64(1))
        mask= tmp.astype(np.bool)
        tmpntrim    = ntrim
        while(tmpntrim > 0):
            mask            = _grid_class._trim_mask(mask)
            tmpntrim        -= 1
        self.attrs.create(name = 'mask', data = mask)
        if is_aniso:
            while(ntrim > 0):
                mask_aniso  = _grid_class._trim_mask(mask_aniso)
                ntrim       -= 1
            self.attrs.create(name = 'mask_aniso', data = mask_aniso)
        self.attrs.create(name = 'mask_runid', data = runid)
        return
    
    def copy_mask(self, fname):
        """get mask for all periods
        """
        indset      = h5py.File(fname, mode = 'r')
        mask        = indset.attrs['mask']
        self.attrs.create(name = 'mask', data = mask)
        return
    
    def merge_mask(self, fname):
        """merge mask
        """
        indset      = h5py.File(fname, mode = 'r')
        maskin      = indset.attrs['mask']
        mask        = self.attrs['mask']
        self.attrs.create(name = 'mask', data = mask + maskin)
        try:
            mask_aniso = self.attrs['mask_aniso']
            self.attrs.create(name = 'mask_aniso', data = mask_aniso + maskin)
        except Exception:
            pass
        return
    
    def set_spain_mask(self):
        """merge mask
        """
        mask    = self.attrs['mask_aniso']
        import h5py
        dset    = h5py.File('/raid/lili/data_spain/azi_run_dir/azi_files_sta_75km_crt100p/0427_mod_w10.h5')
        topo    = dset['topo'][()]
        ind     = (topo < 0.)*(self.latArr > 43.1)*(self.lonArr < 0.)
        mask[ind] = True
        self.attrs.create(name = 'mask_aniso', data = mask)
        
        return
    
    
    def compare_dset(self, in_h5fname, runid = 0):
        """compare two datasets, for debug purpose
        """
        indset  = baseh5(in_h5fname)
        datagrp = self['tomo_run_'+str(runid)]
        indatgrp= indset['tomo_run_'+str(runid)]
        for per in self.pers:
            dat_per_grp     = datagrp['%g_sec' %per] 
            event_lst       = dat_per_grp.keys()
            indat_per_grp   = indatgrp['%g_sec' %per] 
            for evid in event_lst:
                try:
                    dat_ev_grp  = dat_per_grp[evid]
                    indat_ev_grp= indat_per_grp[evid]
                except:
                    print ('No data:'+evid)
                app_vel1    = dat_ev_grp['apparent_velocity'][()]
                app_vel2    = indat_ev_grp['apparent_velocity'][()]
                diff_vel    = app_vel1 - app_vel2
                if np.allclose(app_vel1, app_vel2):
                    print ('--- Apparent velocity ALL equal: '+evid)
                else:
                    print ('--- Apparent velocity NOT equal: '+evid)
                    print ('--- min/max difference: %g/%g' %(diff_vel.min(), diff_vel.max()))
        return
    
    def generate_corrected_map(self, outdir, pers = [], runid = 0, wavetype = 'R'):
        """generate corrected global phase velocity map using a regional phase velocity map.
        =================================================================================================================
        ::: input parameters :::
        outdir              - output directory
        pers                - period array for correction (default is 4)
        -----------------------------------------------------------------------------------------------------------------
        ::: output format ::::
        outdir/outpfx+str(int(per))
        =================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if len(pers) == 0:
            pers            = np.arange(7.)*10.+40.
        dataid          = 'tomo_stack_'+str(runid)
        ingroup         = self[dataid]
        self._get_lon_lat_arr()
        minlon          = self.minlon
        maxlon          = self.maxlon
        minlat          = self.minlat
        maxlat          = self.maxlat
        for per in pers:
            if not per in self.pers:
                print ('!!! period = %g sec NOT in database!' %per)
                continue
            pergrp      = ingroup['%g_sec'%( per )]
            velocity    = pergrp['vel_iso'][()]
            mask        = pergrp['mask'][()]
            mapfile     = global_map_path + '/smpkolya_phv_'+wavetype+'_%d' %per
            out_mapfile = outdir + '/smpkolya_phv_'+wavetype+'_%d' %per
            if not os.path.isfile(mapfile):
                print ('!!! period = %g sec global reference map NOT exists!' %per)
                continue
            outarr      = np.loadtxt(mapfile)
            # interpolate to 1 deg x 1 deg
            lons        = self.lonArr[np.logical_not(mask)]
            lats        = self.latArr[np.logical_not(mask)]
            C           = velocity[np.logical_not(mask)]
            gridder     = _grid_class.SphereGridder(minlon = minlon, maxlon = maxlon, dlon = 1., \
                            minlat = minlat, maxlat = maxlat, dlat = 1., period = per, \
                            evlo = -1., evla = -1., fieldtype = 'phvel', evid = 'REF')
            gridder.read_array(inlons = lons, inlats = lats, inzarr = C)
            # outfname    = 'REF_phvel_'+wavetype+'.lst'
            # prefix      = 'REF_'+wavetype+'_'
            # working_per = outdir + '/%g_sec' %per
            # if not os.path.isdir(working_per):
            #     os.makedirs(working_per)
            # gridder.interp_surface(workingdir = working_per, outfname = outfname)
            gridder.interp_surface()   
            for ig in range(outarr.shape[0]):
                glb_lon = outarr[ig,0]
                glb_lat = outarr[ig,1]
                glb_C   = outarr[ig,2]
                for ilat in range(gridder.Nlat):
                    for ilon in range(gridder.Nlon):
                        reg_lon     = gridder.lons[ilon]
                        if reg_lon < 0.:
                            reg_lon += 360.
                        reg_lat     = gridder.lats[ilat]
                        reg_C       = gridder.Zarr[ilat, ilon]
                    if abs(reg_lon-glb_lon)<0.05 and abs(reg_lat-glb_lat)<0.05 and reg_C != 0 :
                        if abs(glb_C - reg_C) < 0.5:
                            outarr[ig, 2]     = reg_C
                        else:
                            print ('Large changes in regional map: \
                                    vel_glb = '+str(glb_C)+' km/s'+' vel_reg = '+str(reg_C)+' km/sec, '+str(reg_lon)+' '+str(reg_lat))
            np.savetxt(out_mapfile, outarr, fmt='%g %g %.4f')
            # return gridder
        return
    
    def _get_basemap(self, projection='lambert', resolution='i', blon=0., blat=0.):
        """Get basemap for plotting results
        """
        fig=plt.figure(num=None, figsize=(12, 12), dpi=100, facecolor='w', edgecolor='k')
        try:
            minlon  = self.minlon-blon
            maxlon  = self.maxlon+blon
            minlat  = self.minlat-blat
            maxlat  = self.maxlat+blat
        except AttributeError:
            self.get_limits_lonlat()
            minlon  = self.minlon-blon; maxlon=self.maxlon+blon; minlat=self.minlat-blat; maxlat=self.maxlat+blat
        
        # minlon=-165.+360.
        # maxlon=-147+360.
        # minlat=51.
        # maxlat=62.
        
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection == 'merc':
            minlon=-10.
            maxlon=5.
            minlat=31.
            maxlat=45.
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                      urcrnrlon=maxlon, lat_ts=0, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.), labels=[1,1,1,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[0,0,1,0], fontsize=15)
        elif projection == 'merc2':
            minlon=5.
            maxlon=20.
            minlat=35.
            maxlat=50.
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                      urcrnrlon=maxlon, lat_ts=0, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.), labels=[1,1,1,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[0,0,1,0], fontsize=15)
        elif projection == 'global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
        elif projection == 'regional_ortho':
            mapfactor = 2.
            m1      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                        llcrnrx = 0., llcrnry = 0., urcrnrx = m1.urcrnrx/mapfactor, urcrnry = m1.urcrnry/2.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            minlon=-165.+360.
            maxlon=-147+360.
            minlat=51.
            maxlat=62.
            lat_centre  = (maxlat+minlat)/2.0
            lon_centre  = (maxlon+minlon)/2.0
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m

            m       = Basemap(width=1100000, height=1100000, rsphere=(6378137.00,6356752.3142), resolution='h', projection='lcc',\
                        lat_1 = minlat, lat_2 = maxlat, lon_0 = lon_centre, lat_0 = lat_centre + 0.5)
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1, dashes=[2,2], labels=[1,1,1,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,5.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        elif projection=='lambert2':


            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m

            m       = Basemap(width=900000, height=900000, rsphere=(6378137.00,6356752.3142), resolution='h', projection='lcc',\
                        lat_1 = minlat, lat_2 = maxlat, lon_0 = lon_centre, lat_0 = lat_centre + 0.25)
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1, dashes=[2,2], labels=[1,1,1,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,5.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        elif projection == 'ortho':
            m       = Basemap(projection = 'ortho', lon_0 = -170., lat_0 = 40., resolution='l')
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=1,  fontsize=20)
            m.drawmeridians(np.arange(-180.0,180.0,10.0),  linewidth=1)
        elif projection == 'aeqd':
            width = 10000000
            m = Basemap(width = width/1.6,height=width/2.2,projection='aeqd', resolution='h',
                 lon_0 = -153., lat_0 = 62.)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1., dashes=[2,2], labels=[1,1,0,0], fontsize = 15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,0,1], fontsize = 15)
            
        # # m.drawcoastlines(linewidth=0.2)
        try:
            coasts = m.drawcoastlines(zorder=100,color= 'k',linewidth=0.0000)
            # # Exact the paths from coasts
            
            coasts_paths = coasts.get_paths()
            poly_stop = 50
            for ipoly in range(len(coasts_paths)):
                print (ipoly)
                if ipoly > poly_stop:
                    break
                r = coasts_paths[ipoly]
                # Convert into lon/lat vertices
                polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in
                                    r.iter_segments(simplify=False)]
                px = [polygon_vertices[i][0] for i in range(len(polygon_vertices))]
                py = [polygon_vertices[i][1] for i in range(len(polygon_vertices))]
                
                m.plot(px,py,'k-',linewidth=1.)
        except:
            pass
            
        # m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
        # m.fillcontinents(color='coral',lake_color='aqua')
        # m.drawcountries(linewidth=1.)
        return m
    
    def plot(self, runid, datatype, period, width=-1., use_mask_all = False, semfactor=2., Nthresh=None, clabel='', cmap='surf',\
             projection='lambert', hillshade = False, vmin = None, vmax = None, showfig = True, v_rel = None, figname=None,
            instafname = None, tlon=None, tlat=None):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        datatype        - datatype for plotting
        period          - period of data
        sem_factor      - factor multiplied to get the finalized uncertainties
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        vmin, vmax      - min/max value of plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        dataid          = 'tomo_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        self._get_lon_lat_arr()
        if not period in pers:
            raise KeyError('!!! period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        if datatype == 'vel' or datatype=='velocity' or datatype == 'v':
            datatype    = 'vel_iso'
        elif datatype == 'sem' or datatype == 'un' or datatype == 'uncertainty':
            datatype    = 'vel_sem'
        elif datatype == 'std':
            datatype    = 'slowness_std'
        try:
            data        = pergrp[datatype][()]
        except:
            outstr      = ''
            for key in pergrp.keys():
                outstr  +=key
                outstr  +=', '
            outstr      = outstr[:-1]
            raise KeyError('Unexpected datatype: '+datatype+ ', available datatypes are: '+outstr)
        
        if datatype=='Nmeasure_aniso':
            factor      = ingroup.attrs['grid_lon'] * ingroup.attrs['grid_lat']
        else:
            factor      = 1
        if datatype=='Nmeasure_aniso' or datatype=='unpsi' or datatype=='unamp' or datatype=='amparr':
            mask        = pergrp['mask_aniso'][()] + pergrp['mask'][()]
        else:
            try:
                mask    = pergrp['mask'][()]
            except:
                print ('NO mask!!!')
                mask    = np.zeros(data.shape)
        if use_mask_all:
            mask        = self.attrs['mask']            
        if not (Nthresh is None):
            Narr        = pergrp['NmeasureQC'][()]
            mask        += Narr < Nthresh
        if datatype == 'vel_sem':
            data        *= 1000.*semfactor
        
        # smoothing
        if width > 0.:
            gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = period, \
                            evlo = 0., evla = 0., fieldtype = 'Tph', evid = 'plt')
            gridder.read_array(inlons = self.lonArr[np.logical_not(mask)], inlats = self.latArr[np.logical_not(mask)], inzarr = data[np.logical_not(mask)])
            outfname    = 'plt_Tph.lst'
            prefix      = 'plt_Tph_'
            gridder.gauss_smoothing(workingdir = './temp_plt', outfname = outfname, width = width)
            data[:]     = gridder.Zarr
        
        mdata           = ma.masked_array(data/factor, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection = projection)
        x, y            = m(self.lonArr, self.latArr)
        try:
            import pycpt
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
                # cmap    = cmap.reversed()
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
            cmap.set_bad('silver', alpha = 0.)
        except:
            pass
        ###################################################################
        # shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'

        import surfpy.map_dat.shapefile_faults as fault_maps
        fault_map_path    = fault_maps.__path__._path[0]
        shapefname  = fault_map_path+'/gem_active_faults'
        # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
        if projection=='lambert':
            shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
            if not os.path.isfile(shapefname+'.shp'):
                shapefname  = '/raid/lili/data_marin/map_data/geological_maps/qfaults'
            m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
        # elif projection == 'merc':
        #     m.readshapefile(shapefname, 'faultline', linewidth = 2.2, color='black', default_encoding='windows-1252')
        #     m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='yellow', default_encoding='windows-1252')
        elif projection != 'merc':
            m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        
        if projection == 'merc' and os.path.isdir('../geo_maps'):
            shapefname  = '../geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='grey')
            
        
        # shapefname  = '/home/lili/data_mongo/fault_shp/doc-line'
        # # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black')
        # m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey')
        
        # # # if instafname is  None:
        # # #     import surfpy.map_dat.volcano_locs as volc_maps
        # # #     volc_map_path    = volc_maps.__path__._path[0]
        # # #     shapefname  = volc_map_path+'/SDE_GLB_VOLC.shp'
        # # #     shplst      = shapefile.Reader(shapefname)
        # # #     for rec in shplst.records():
        # # #         lon_vol = rec[4]
        # # #         lat_vol = rec[3]
        # # #         xvol, yvol            = m(lon_vol, lat_vol)
        # # #         m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        ###################################################################
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            import pycpt
            etopodata   = Dataset('/home/lili/gebco_mongo.nc')
            etopo       = (etopodata.variables['elevation'][:]).data
            lons        = (etopodata.variables['lon'][:]).data
            lons[lons>180.] = lons[lons>180.] - 360.
            lats        = (etopodata.variables['lat'][:]).data
            ls          = LightSource(azdeg=315, altdeg=45)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            
        
        
        if v_rel is not None:
            mdata       = (mdata - v_rel)/v_rel * 100.
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap = cmap, shading = 'gouraud', vmin = vmin, vmax = vmax, alpha=.5)
        else:
            im          = m.pcolormesh(x, y, mdata, cmap = cmap, shading = 'gouraud', vmin = vmin, vmax = vmax)
            
        try:
            tx, ty = m(tlon, tlat)
            m.plot(tx, ty, 'ro', ms=5, zorder = 100)
        except Exception:
            pass
            
            
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        # m.drawcountries(linewidth=1.)
        # m.fillcontinents(color='none', lake_color='#99ffff',zorder=100., alpha=1.)
        
            # m.contour(x, y, mask_eik, colors='white', lw=1)
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.])
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
        # cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.0, 4.1, 4.2, 4.3, 4.4])
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        
        cb.set_label(clabel, fontsize=40, rotation=0)
        # cb.outline.set_linewidth(2)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize = 20)
        print ('=== plotting data from '+dataid)
        
        ##############
        if instafname is not None:
            dset = pyasdf.ASDFDataSet(instafname)
            for staid in dset.waveforms.list():
                netcode, stacode  = staid.split('.')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stainv             = dset.waveforms[staid].StationXML.networks[0].stations[0]
                    lon                = stainv.longitude
                    lat                = stainv.latitude
                stax, stay          = m(lon, lat)
                if netcode != 'XL':
                    continue
                m.plot(stax, stay, '^', markerfacecolor="None", mec='k', markersize=12)
        ##############
        if showfig:
            plt.show()
        if figname is not None:
            plt.savefig(figname)
        return
    
    def plot_psi(self, runid, period, factor=5, normv=5., width=0.005, ampref=0.02, datatype='', gwidth=-1., use_mask_all = False, 
            scaled=False, masked=True, clabel='', cmap='surf', projection='merc', vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        datatype        - type of data to plot as background color
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        vmin, vmax      - min/max value of plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        dataid  = 'tomo_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup = self[dataid]
        # period array
        pers    = self.pers
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi     = pergrp['psiarr'][()]
        amp     = pergrp['amparr'][()]
        if use_mask_all:
            mask= self.attrs['mask_aniso']
        else:    
            mask= pergrp['mask_aniso'][()] + pergrp['mask'][()]
        # get velocity
        try:
            data= pergrp[datatype][()]
            plot_data   = True
        except:
            plot_data   = False
        #-----------
        # plot data
        #-----------
        m       = self._get_basemap(projection = projection)
        if scaled:
            U       = np.sin(psi/180.*np.pi)*amp/ampref/normv
            V       = np.cos(psi/180.*np.pi)*amp/ampref/normv
            Uref    = np.ones(self.lonArr.shape)*1./normv
            Vref    = np.zeros(self.lonArr.shape)
        else:
            U       = np.sin(psi/180.*np.pi)/normv
            V       = np.cos(psi/180.*np.pi)/normv
        # rotate vectors to map projection coordinates
        U, V, x, y  = m.rotate_vector(U, V, self.lonArr, self.latArr, returnxy=True)
        if scaled:
            Uref, Vref, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr, self.latArr, returnxy=True)
        #--------------------------------------
        # plot background data
        #--------------------------------------
        if plot_data:
            print ('=== plotting data: '+datatype)
            try:
                import pycpt
                if os.path.isfile(cmap):
                    cmap    = pycpt.load.gmtColormap(cmap)
                elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                    cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
                cmap.set_bad('silver', alpha = 0.)
            except:
                pass
            # smoothing
            if gwidth > 0.:
                gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                                minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = period, \
                                evlo = 0., evla = 0., fieldtype = 'Tph', evid = 'plt')
                gridder.read_array(inlons = self.lonArr[np.logical_not(mask)], inlats = self.latArr[np.logical_not(mask)], inzarr = data[np.logical_not(mask)])
                outfname    = 'plt_Tph.lst'
                prefix      = 'plt_Tph_'
                gridder.gauss_smoothing(workingdir = './temp_plt', outfname = outfname, width = gwidth)
                data[:]     = gridder.Zarr
            
            
            if masked:
                data    = ma.masked_array(data, mask = mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=40, rotation=0)
            # cb.outline.set_linewidth(2)
            plt.suptitle(str(period)+' sec', fontsize=20)
            cb.ax.tick_params(labelsize=20)
            # cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")

        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)
        ###
        import surfpy.map_dat.shapefile_faults as fault_maps
        fault_map_path    = fault_maps.__path__._path[0]
        shapefname  = fault_map_path+'/gem_active_faults'
        # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
        if projection=='lambert':
            shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
            m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
        elif projection != 'merc':
            m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        if projection == 'merc' and os.path.isdir('/home/lili/spain_proj/geo_maps'):
            shapefname  = '/home/lili/spain_proj/geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='grey')
        ###
        #--------------------------------------
        # plot fast axis
        #--------------------------------------
        x_psi       = x.copy()
        y_psi       = y.copy()
        mask_psi    = mask.copy()
        if factor!=None:
            x_psi   = x_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
            y_psi   = y_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
            U       = U[0:self.Nlat:factor, 0:self.Nlon:factor]
            V       = V[0:self.Nlat:factor, 0:self.Nlon:factor]
            mask_psi= mask_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
        # Q       = m.quiver(x, y, U, V, scale=30, width=0.001, headaxislength=0)
        if masked:
            U   = ma.masked_array(U, mask=mask_psi )
            V   = ma.masked_array(V, mask=mask_psi )
        Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        # if scaled:
        #     mask_ref        = np.ones(self.lonArr.shape)
        #     Uref            = ma.masked_array(Uref, mask=mask_ref )
        #     Vref            = ma.masked_array(Vref, mask=mask_ref )
        #     m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        #     m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        
        x_ref, y_ref = m(2.5, 32.)
        Uref    = 2./ampref/normv
        Vref    = 0.
        Q1      = m.quiver(x_ref, y_ref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        Q2      = m.quiver(x_ref, y_ref, -Uref, -Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        Q1      = m.quiver(x_ref, y_ref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        Q2      = m.quiver(x_ref, y_ref, -Uref, -Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        print (amp.max(), U.max(), V.max(), Uref)
        
        plt.suptitle(str(period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        return
        
    def plot_diff(self, infname, period, runid=0, datatype='v', width=-1., use_mask_all = False, semfactor=2., Nthresh=None, clabel='', cmap='surf',\
             projection='lambert', hillshade = False, vmin = None, vmax = None, showfig = True, v_rel = None, figname=None,
                instafname = None):
        dataid          = 'tomo_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        self._get_lon_lat_arr()
        if not period in pers:
            raise KeyError('!!! period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        if datatype == 'vel' or datatype=='velocity' or datatype == 'v':
            datatype    = 'vel_iso'
        elif datatype == 'sem' or datatype == 'un' or datatype == 'uncertainty':
            datatype    = 'vel_sem'
        elif datatype == 'std':
            datatype    = 'slowness_std'
        try:
            data        = pergrp[datatype][()]
        except:
            outstr      = ''
            for key in pergrp.keys():
                outstr  +=key
                outstr  +=', '
            outstr      = outstr[:-1]
            raise KeyError('Unexpected datatype: '+datatype+ ', available datatypes are: '+outstr)
        
        if datatype=='Nmeasure_aniso':
            factor      = ingroup.attrs['grid_lon'] * ingroup.attrs['grid_lat']
        else:
            factor      = 1
        if datatype=='Nmeasure_aniso' or datatype=='unpsi' or datatype=='unamp' or datatype=='amparr':
            mask        = pergrp['mask_aniso'][()] + pergrp['mask'][()]
        else:
            try:
                mask    = pergrp['mask'][()]
            except:
                print ('NO mask!!!')
                mask    = np.zeros(data.shape)
        if use_mask_all:
            mask        = self.attrs['mask']            
        if not (Nthresh is None):
            Narr        = pergrp['NmeasureQC'][()]
            mask        += Narr < Nthresh
        if datatype == 'vel_sem':
            data        *= 1000.*semfactor
        ###
        dset            = baseh5(infname)
        dataid          = 'tomo_stack_'+str(runid)
        grp2            = dset[dataid]
        pergrp2         = grp2['%g_sec'%( period )]
        data2           = pergrp2[datatype][()]
        
        data            = data - data2
        factor = 0.001
        
        # data[data>0.18] -= 0.1
        # data[data<-0.18] += 0.1
        ###
        
        # smoothing
        if width > 0.:
            gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = period, \
                            evlo = 0., evla = 0., fieldtype = 'Tph', evid = 'plt')
            gridder.read_array(inlons = self.lonArr[np.logical_not(mask)], inlats = self.latArr[np.logical_not(mask)], inzarr = data[np.logical_not(mask)])
            outfname    = 'plt_Tph.lst'
            prefix      = 'plt_Tph_'
            gridder.gauss_smoothing(workingdir = './temp_plt', outfname = outfname, width = width)
            data[:]     = gridder.Zarr
        
        mdata           = ma.masked_array(data/factor, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection = projection)
        x, y            = m(self.lonArr, self.latArr)
        try:
            import pycpt
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
                # cmap    = cmap.reversed()
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
            cmap.set_bad('silver', alpha = 0.)
        except:
            pass
        ###################################################################
        # shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'

        import surfpy.map_dat.shapefile_faults as fault_maps
        fault_map_path    = fault_maps.__path__._path[0]
        shapefname  = fault_map_path+'/gem_active_faults'
        # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
        if projection=='lambert':
            shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
            m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
        # elif projection == 'merc':
        #     m.readshapefile(shapefname, 'faultline', linewidth = 2.2, color='black', default_encoding='windows-1252')
        #     m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='yellow', default_encoding='windows-1252')
        # elif projection != 'merc':
        #     m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        
        if projection == 'merc' and os.path.isdir('/raid/lili/geo_map_europe'):
            shapefname  = '/raid/lili/geo_map_europe/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='grey')
            
        
        # shapefname  = '/home/lili/data_mongo/fault_shp/doc-line'
        # # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black')
        # m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey')
        # # # if instafname is  None:
        # # #     import surfpy.map_dat.volcano_locs as volc_maps
        # # #     volc_map_path    = volc_maps.__path__._path[0]
        # # #     shapefname  = volc_map_path+'/SDE_GLB_VOLC.shp'
        # # #     shplst      = shapefile.Reader(shapefname)
        # # #     for rec in shplst.records():
        # # #         lon_vol = rec[4]
        # # #         lat_vol = rec[3]
        # # #         xvol, yvol            = m(lon_vol, lat_vol)
        # # #         m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        
        if v_rel is not None:
            mdata       = (mdata - v_rel)/v_rel * 100.
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap = cmap, shading = 'gouraud', vmin = vmin, vmax = vmax, alpha=.5)
        else:
            im          = m.pcolormesh(x, y, mdata, cmap = cmap, shading = 'gouraud', vmin = vmin, vmax = vmax)
        
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        
        cb.set_label(clabel, fontsize=40, rotation=0)
        # cb.outline.set_linewidth(2)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize = 20)
        print ('=== plotting data from '+dataid)
        
        # # # # if np.any(data < 0.):
        # # # #     negative       = data < 0.
        # # # #     # negative       = ma.masked_array(negative, mask = mask )
        # # # #     m.contour(x, y, negative, colors='w', lw=0.01)
        
        ##############
        if instafname is not None:
            dset = pyasdf.ASDFDataSet(instafname)
            for staid in dset.waveforms.list():
                netcode, stacode  = staid.split('.')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stainv             = dset.waveforms[staid].StationXML.networks[0].stations[0]
                    lon                = stainv.longitude
                    lat                = stainv.latitude
                stax, stay          = m(lon, lat)
                if netcode != 'XL':
                    continue
                m.plot(stax, stay, '^', markerfacecolor="None", mec='k', markersize=12)
            
                    
        ##############
        if showfig:
            plt.show()
        if figname is not None:
            plt.savefig(figname)
        
        from matplotlib.ticker import FuncFormatter
        import matplotlib
        def to_percent(y, position):
            # Ignore the passed in position. This has the effect of scaling the default
            # tick locations.
            s = '%.0f' %(100. * y)
            # The percent symbol needs escaping in latex
            if matplotlib.rcParams['text.usetex'] is True:
                return s + r'$\%$'
            else:
                return s + '%'
        
        plt.figure(figsize=[18, 9.6])
        xlabel  = 'Phase Velocity Difference (m/s)'
        data = data[np.logical_not(mask)]/factor
        weights = np.ones_like(data)/float(data.size)
        dbin    = 10.
        bins    = np.arange(min(data), max(data) + dbin, dbin)
        ax          = plt.subplot()
        plt.hist(data, bins=bins, weights=weights, alpha=1., color='r')
        # plt.hist(paraval2, bins=bins2, weights=weights2, alpha=.5, color='b')
        # plt.hist(paraval3, bins=bins3, weights=weights3, alpha=1., edgecolor='k', facecolor='None')
        
        
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel('Percentage (%)', fontsize=30)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlim([vmin, vmax])
        plt.suptitle('mean %g, std %g' %(data.mean(), data.std()))
        plt.show()
        
        
        return
    
    def plot_rays(self, period,  clabel='', cmap='surf', projection='lambert', vmin = None, vmax = None,
            showfig = True, v_rel = None, figname=None, instafname = None, snr_thresh = 10.):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        datatype        - datatype for plotting
        period          - period of data
        sem_factor      - factor multiplied to get the finalized uncertainties
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        vmin, vmax      - min/max value of plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        dataid          = 'input_field_data'
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        self._get_lon_lat_arr()
        m               = self._get_basemap(projection = projection)
        
                
        if not period in pers:
            raise KeyError('!!! period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        event_lst       = list(pergrp.keys())
        
        NrayC2          = 0
        NrayC3          = 0
        for evid in event_lst:
            # determine type of data
            if evid[:4] == 'surf': # earthquake
                idat_type   = 3
            elif evid[-3:] == '_C3': # C3
                idat_type   = 2
            else:
                idat_type   = 1 # C2
            dat_ev_grp  = pergrp[evid]
            evlo        = dat_ev_grp.attrs['evlo']
            evla        = dat_ev_grp.attrs['evla']
            lons        = dat_ev_grp['lons'][()]
            if self.ilontype == 0:
                lons[lons>180.] -= 360.
            else:
                lons[lons<0.]   += 360.
            lats        = dat_ev_grp['lats'][()]
            dist        = dat_ev_grp['distance'][()]
            snr         = dat_ev_grp['snr'][()]
            ind_borrow  = dat_ev_grp['index_borrow'][()]
            
            ind         = snr >= snr_thresh
            ind2        = (lats <= 62.)*(lats >= 51.)*(lons <= 213.)*(lons >= 195.)
            ind         = ind*ind2*np.logical_not(ind_borrow.astype(bool))
            
            snr         = snr[ind]
            lons        = lons[ind]
            lats        = lats[ind]
            if snr.size == 0:
                continue
            if idat_type == 1:
                NrayC2  += snr.size
            elif idat_type == 2:
                NrayC3  += snr.size
            xev, yev    = m(evlo, evla)
            xsta, ysta  = m(lons, lats)
            for i in range(snr.size):
                x       = xsta[i]
                y       = ysta[i]
                m.plot([xev, x], [yev, y], '-', color = 'grey', lw=1, alpha = 0.06)
            
    
        ###################################################################
        # shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'

        # import surfpy.map_dat.shapefile_faults as fault_maps
        # fault_map_path    = fault_maps.__path__._path[0]
        # shapefname  = fault_map_path+'/gem_active_faults'
        # # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
        # if projection=='lambert':
        #     shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
        #     if not os.path.isfile(shapefname+'.shp'):
        #         shapefname  = '/raid/lili/data_marin/map_data/geological_maps/qfaults'
        #     m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
        #     m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
        # # elif projection == 'merc':
        # #     m.readshapefile(shapefname, 'faultline', linewidth = 2.2, color='black', default_encoding='windows-1252')
        # #     m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='yellow', default_encoding='windows-1252')
        # elif projection != 'merc':
        #     m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        # 
        # if projection == 'merc' and os.path.isdir('../geo_maps'):
        #     shapefname  = '../geo_maps/prv4_2l-polygon'
        #     m.readshapefile(shapefname, 'faultline', linewidth = 2, color='grey')
            
        
        # shapefname  = '/home/lili/data_mongo/fault_shp/doc-line'
        # # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black')
        # m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey')
        if instafname is  None:
            import surfpy.map_dat.volcano_locs as volc_maps
            volc_map_path    = volc_maps.__path__._path[0]
            shapefname  = volc_map_path+'/SDE_GLB_VOLC.shp'
            shplst      = shapefile.Reader(shapefname)
            for rec in shplst.records():
                lon_vol = rec[4]
                lat_vol = rec[3]
                xvol, yvol            = m(lon_vol, lat_vol)
                m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)

        m.fillcontinents(color='white', lake_color='none',zorder=0.2, alpha=1.)
        # m.drawcountries(linewidth=1.)
        # m.fillcontinents(color='none', lake_color='#99ffff',zorder=100., alpha=1.)

        ##############
        if instafname is not None:
            dset = pyasdf.ASDFDataSet(instafname)
            for staid in dset.waveforms.list():
                netcode, stacode  = staid.split('.')
                if stacode == 'WD61':
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stainv             = dset.waveforms[staid].StationXML.networks[0].stations[0]
                    lon                = stainv.longitude
                    lat                = stainv.latitude
                stax, stay          = m(lon, lat)
                # # # if netcode != 'XL':
                # # #     continue
                m.plot(stax, stay, '^', markerfacecolor="None", mec='k', markersize=12)
            
                    
        ##############
        print ('c2 %d c3 %d' %(NrayC2, NrayC3))
        plt.suptitle('c2 %d c3 %d' %(NrayC2, NrayC3))
        if showfig:
            plt.show()
        if figname is not None:
            plt.savefig(figname)
        return
    
    def plot_azi_fit(self, inlat, inlon, period, runid=0,  ampfactor=1., shift=0., A0factor=1., autofit = -1, unthresh = 0.04,\
                fitpsi1=False, fitpsi2=True, getdata=False, showfig=True, outfname = None, semfactor=2., inpredat=[], N_thresh = 5):
        
        dataid          = 'tomo_stack_'+str(runid)
        ingroup         = self[dataid]
        self._get_lon_lat_arr()
        
        ilat            = self.lats == inlat
        ilon            = self.lons == inlon
        # if index[0].size == 0 or index[1].size == 0:
        #     print ('No data at lon = '+str(inlon)+' lat = '+str(inlat))
        #     return
        if not period in self.pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        mask            = pergrp['mask_aniso'][()]
        if mask[ilat, ilon]:
            print ('No valid data at lon = '+str(inlon)+' lat = '+str(inlat))
            return
        slowAni         = pergrp['slowness_aniso'][()] + pergrp['slowness'][()]
        velAnisem       = pergrp['vel_aniso_sem'][()]
        outslowness     = slowAni[:, ilat, ilon]
        outvel_sem      = velAnisem[:, ilat, ilon]
        avg_slowness    = pergrp['slowness'][()][ilat, ilon]
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin            = ingroup.attrs['N_bin']
        d_bin           = float((maxazi-minazi)/Nbin)
        azArr           = np.arange(Nbin)*d_bin + minazi + d_bin/2.
        # # azArr           = np.arange(Nbin)*d_bin + minazi
        
        out_hist        = pergrp['hist_aniso'][()][:, ilat, ilon]

        if N_thresh > 0:
            ind         = np.where((outvel_sem != 0)*(out_hist > N_thresh ))[0]
        else:
            ind         = np.where((outvel_sem != 0))[0]
        outslowness     = outslowness[ind]
        azArr           = azArr[ind]
        outvel_sem      = outvel_sem[ind]*semfactor
        Nbin            = ind.size
        if getdata:
            return azArr, 1./outslowness, outvel_sem, 1./avg_slowness
        if fitpsi1 or fitpsi2:
            indat           = (1./outslowness).reshape(1, Nbin)
            # construct forward operator matrix
            tG              = np.ones((Nbin, 1), dtype=np.float64)
            G               = tG.copy()
            azArr           += 180.
            azArr           = 360. - azArr
            azArr           -= 90.
            azArr[azArr<0.] += 360.  
            tbaz            = np.pi*(azArr)/180.
            if fitpsi2:
                tGsin2      = np.sin(tbaz*2)
                tGcos2      = np.cos(tbaz*2)
                G           = np.append(G, tGsin2)
                G           = np.append(G, tGcos2)
            if fitpsi1:
                tGsin       = np.sin(tbaz)
                tGcos       = np.cos(tbaz)
                G           = np.append(G, tGsin)
                G           = np.append(G, tGcos)
            if fitpsi1 and fitpsi2:
                G           = G.reshape((5, Nbin))
            else:
                G           = G.reshape((3, Nbin))
            d   = indat.T
            G   = G.T
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
            # uncertainties
            unA2                    = (uncertainties.umath.sqrt(m1**2+m2**2)/m0*100.).std_dev
            unpsi2                  = (uncertainties.umath.atan2(m1, m2)/np.pi*180./2.).std_dev
            unamp                   = unA2
            unpsi2                  = min(unpsi2, 90.)
            if fitpsi1:
                m1s         = ufloat(model[-2][0], pcov[-2][1])
                m1c         = ufloat(model[-1][0], pcov[-1][2])
                unA1        = (uncertainties.umath.sqrt(m1s**2+m1c**2)/m0*100.).std_dev
                A1          = (uncertainties.umath.sqrt(m1s**2+m1c**2)/m0*100.).n
                unpsi1      = (uncertainties.umath.atan2(m1s, m1c)/np.pi*180.).std_dev
                psi1        = (uncertainties.umath.atan2(m1s, m1c)/np.pi*180.).n
                
        psi2  = psi2/np.pi * 180.
            
        
        plt.figure(figsize=[18, 9.6])
        ax      = plt.subplot()
        az_plt  = []
        vel_plt = []
        sem_plt = []
        for i in range(azArr.size):
            if azArr[i] in az_plt:
                continue
            if outvel_sem[i] > unthresh:
                continue
            az_plt.append(azArr[i])
            vel_plt.append(1./outslowness[i])
            sem_plt.append(outvel_sem[i])
        #
        A2      /= ampfactor
        dvel    = (vel_plt - A0)/ampfactor
        A0      /= A0factor
        vel_plt = A0 + dvel
        #
        Nroll   = np.round(shift/d_bin)
        vel_plt = np.roll(vel_plt, -int(Nroll))
        psi2    += shift
        #
        # print (az_plt)
        # print (vel_plt)
        # print (sem_plt)
        az_plt = np.array(az_plt)
        az_plt = az_plt.reshape(az_plt.size)
        vel_plt = np.array(vel_plt)
        vel_plt = vel_plt.reshape(vel_plt.size)
        sem_plt = np.array(sem_plt)
        sem_plt =sem_plt.reshape(sem_plt.size)
        
        if fitpsi1 or fitpsi2:
            az_fit          = np.mgrid[minazi:maxazi:100*1j]
            if fitpsi1:
                predat      = A0 + A1/100.*A0*np.cos( np.pi/180.*(az_fit+180.-psi1) )
                predat1     = A0 + A1/100.*A0*np.cos( np.pi/180.*(az_plt+180.-psi1) )
                fitlabel    = 'A1: %g %%, psi1: %g deg \n' %(A1, psi1)
                fitlabel    += '\n unA1: %g %%, unphi1: %g deg \n' %(unA1, unpsi1)
                if fitpsi2:
                     predat     += A2*np.cos(2.*(np.pi/180.*(az_fit+180.-psi2)) )
                     predat1    += A2*np.cos(2.*(np.pi/180.*(az_plt+180.-psi2)) )
                     fitlabel   += 'A2: %g %%, phi2: %g deg' %(A2/A0*100., psi2)
            else:
                predat      = A0 + A2*np.cos(2.*(np.pi/180.*(az_fit+180.-psi2)) )
                predat1     = A0 + A2*np.cos(2.*(np.pi/180.*(az_plt+180.-psi2)) )
                fitlabel    = 'A2: %g %%, phi2: %g deg' %(A2/A0*100., psi2)
            fitlabel    += '\n unA2: %g %%, unphi2: %g deg' %(unA2, unpsi2)
            plt.plot(az_fit+180., predat, 'b-', lw=5, label='fit \n'+fitlabel )
            if len(inpredat)==len(az_fit):
                plt.plot(az_fit+180., inpredat, 'r--', lw=3, label='fit \n'+fitlabel )
        while (autofit > 0):
            # vel_plt[(predat1 - vel_plt) > sem_plt]   += sem_plt[(predat1 - vel_plt) > sem_plt]
            # vel_plt[(predat1 - vel_plt) <-sem_plt]   -= sem_plt[(predat1 - vel_plt) < -sem_plt]
            try:
                vel_plt[(predat1 - vel_plt) > sem_plt]   += sem_plt[(predat1 - vel_plt) > sem_plt]
                vel_plt[(predat1 - vel_plt) <-sem_plt]   -= sem_plt[(predat1 - vel_plt) < -sem_plt]
            except Exception:
                break
            autofit -= 1
        
        ind = abs(predat1 - vel_plt) < 3.*sem_plt
        
        plt.errorbar(az_plt[ind], vel_plt[ind], yerr=sem_plt[ind], fmt='o', color='k',  ms=10, capsize=3, capthick=3)
        
        plt.ylabel('Phase velocity(km/sec)', fontsize=50)
        plt.xlabel('Azimuth (degree)', fontsize=50)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        # plt.legend()
        if fitpsi1:
            plt.suptitle('A0 %g, A2: %g %%, phi2: %g deg, unA2: %g, unpsi2: %g, A1: %g, psi1: %g, unA1: %g, unpsi1: %g'\
                         %(A0, A2/A0*100., psi2, unA2, unpsi2, A1, psi1, unA1, unpsi1))
        else:
            plt.suptitle('A0 %g, A2: %g %%, phi2: %g deg, unA2: %g, unpsi2: %g' %(A0, A2/A0*100., psi2, unA2, unpsi2))
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return predat
    
    def plot_azi_fit_old(self, inlat, inlon, period, runid=0,  ampfactor=1., shift=0., A0factor=1.,\
                fitpsi1=False, fitpsi2=True, getdata=False, showfig=True, outfname = None, semfactor=5., inpredat=[], N_thresh = 5):
        
        dataid          = 'tomo_stack_'+str(runid)
        ingroup         = self[dataid]
        self._get_lon_lat_arr()
        
        index   = np.where((self.latArr==inlat)*(self.lonArr==inlon))
        if index[0].size == 0 or index[1].size == 0:
            print ('No data at lon = '+str(inlon)+' lat = '+str(inlat))
            return
        if not period in self.pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        mask            = pergrp['mask_aniso'][()]
        if mask[index[0], index[1]]:
            print ('No valid data at lon = '+str(inlon)+' lat = '+str(inlat))
            return
        slowAni         = pergrp['slowness_aniso'][()] + pergrp['slowness'][()]
        velAnisem       = pergrp['vel_aniso_sem'][()]
        outslowness     = slowAni[:, index[0], index[1]]
        outvel_sem      = velAnisem[:, index[0], index[1]]
        avg_slowness    = pergrp['slowness'][()][index[0], index[1]]
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin            = ingroup.attrs['N_bin']
        d_bin           = float((maxazi-minazi)/Nbin)
        azArr           = np.arange(Nbin)*d_bin + minazi
        
        out_hist        = pergrp['hist_aniso'][()][:, index[0], index[1]]

        if N_thresh > 0:
            ind         = np.where((outvel_sem != 0)*(out_hist > N_thresh ))[0]
        else:
            ind         = np.where((outvel_sem != 0))[0]
        outslowness     = outslowness[ind]
        azArr           = azArr[ind]
        outvel_sem      = outvel_sem[ind]*semfactor
        Nbin            = ind.size
        if getdata:
            return azArr, 1./outslowness, outvel_sem, 1./avg_slowness
        if fitpsi1 or fitpsi2:
            indat           = (1./outslowness).reshape(1, Nbin)
            # construct forward operator matrix
            tG              = np.ones((Nbin, 1), dtype=np.float64)
            G               = tG.copy()
            azArr           += 180.
            azArr           = 360. - azArr
            azArr           -= 90.
            azArr[azArr<0.] += 360.  
            tbaz            = np.pi*(azArr)/180.
            if fitpsi1:
                tGsin       = np.sin(tbaz)
                tGcos       = np.cos(tbaz)
                G           = np.append(G, tGsin)
                G           = np.append(G, tGcos)
            if fitpsi2:
                tGsin2      = np.sin(tbaz*2)
                tGcos2      = np.cos(tbaz*2)
                G           = np.append(G, tGsin2)
                G           = np.append(G, tGcos2)
            if fitpsi1 and fitpsi2:
                G           = G.reshape((5, Nbin))
            else:
                G           = G.reshape((3, Nbin))
            G               = G.T
            d               = indat.T
            # linear inversion
            Cd              = np.zeros((Nbin, Nbin), dtype=np.float64)
            np.fill_diagonal(Cd, outvel_sem**2)
            Ginv1           = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cd)), G) )
            Ginv2           = np.dot( np.dot(G.T, np.linalg.inv(Cd)), d)
            m               = np.dot(Ginv1, Ginv2)
            Cm              = Ginv1
            pcov            = np.sqrt(np.absolute(Cm))
            m0              = ufloat(m[0][0], pcov[0][0])
            A0              = m[0][0]
            if fitpsi2:
                # psi-2
                m2s             = ufloat(m[-2][0], pcov[-2][-2])
                m2c             = ufloat(m[-1][0], pcov[-1][-1])
                unA2            = (uncertainties.umath.sqrt(m2s**2+m2c**2)/m0*100.).std_dev
                A2              = (uncertainties.umath.sqrt(m2s**2+m2c**2)/m0*100.).n
                unpsi2          = (uncertainties.umath.atan2(m2s, m2c)/np.pi*180./2.).std_dev
                psi2            = (uncertainties.umath.atan2(m2s, m2c)/np.pi*180./2.).n
            # psi-1
            if fitpsi1:
                m1s         = ufloat(m[1][0], pcov[1][1])
                m1c         = ufloat(m[2][0], pcov[2][2])
                unA1        = (uncertainties.umath.sqrt(m1s**2+m1c**2)/m0*100.).std_dev
                A1          = (uncertainties.umath.sqrt(m1s**2+m1c**2)/m0*100.).n
                unpsi1      = (uncertainties.umath.atan2(m1s, m1c)/np.pi*180.).std_dev
                psi1        = (uncertainties.umath.atan2(m1s, m1c)/np.pi*180.).n
        
        plt.figure(figsize=[18, 9.6])
        ax      = plt.subplot()
        az_plt  = []
        vel_plt = []
        sem_plt = []
        for i in range(azArr.size):
            if azArr[i] in az_plt:
                continue
            az_plt.append(azArr[i])
            vel_plt.append(1./outslowness[i])
            sem_plt.append(outvel_sem[i])
        #
        A2      /= ampfactor
        dvel    = (vel_plt - A0)/ampfactor
        A0      /= A0factor
        vel_plt = A0 + dvel
        #
        Nroll   = np.round(shift/d_bin)
        vel_plt = np.roll(vel_plt, -int(Nroll))
        psi2    += shift
        #
        # print (az_plt)
        # print (vel_plt)
        # print (sem_plt)
        az_plt = np.array(az_plt)
        az_plt = az_plt.reshape(az_plt.size)
        vel_plt = np.array(vel_plt)
        vel_plt = vel_plt.reshape(vel_plt.size)
        sem_plt = np.array(sem_plt)
        sem_plt =sem_plt.reshape(sem_plt.size)
        plt.errorbar(az_plt, vel_plt, yerr=sem_plt, fmt='o', color='k',  ms=10, capsize=3, capthick=3)
        if fitpsi1 or fitpsi2:
            az_fit          = np.mgrid[minazi:maxazi:100*1j]
            if fitpsi1:
                predat      = A0 + A1*A0/100.*np.cos( np.pi/180.*(az_fit+180.-psi1) )
                fitlabel    = 'A1: %g %%, psi1: %g deg \n' %(A1, psi1)
                fitlabel    += '\n unA1: %g %%, unphi1: %g deg \n' %(unA1, unpsi1)
                if fitpsi2:
                     predat     += A2*A0/100.*np.cos(2.*(np.pi/180.*(az_fit+180.-psi2)) )
                     fitlabel   += 'A2: %g %%, phi2: %g deg' %(A2, psi2)
            else:
                predat      = A0 + A2*A0/100.*np.cos(2.*(np.pi/180.*(az_fit+180.-psi2)) )
                fitlabel    = 'A2: %g %%, phi2: %g deg' %(A2, psi2)
            fitlabel    += '\n unA2: %g %%, unphi2: %g deg' %(unA2, unpsi2)
            plt.plot(az_fit+180., predat, 'b-', lw=3, label='fit \n'+fitlabel )
            if len(inpredat)==len(az_fit):
                plt.plot(az_fit+180., inpredat, 'r--', lw=3, label='fit \n'+fitlabel )
        plt.ylabel('Phase velocity(km/sec)', fontsize=50)
        plt.xlabel('Azimuth (degree)', fontsize=50)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        plt.legend()
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return predat
    
    
    
    
