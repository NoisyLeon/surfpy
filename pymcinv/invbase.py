# -*- coding: utf-8 -*-
"""
base hdf5 for inversion

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""

import surfpy.pymcinv._model_funcs as _model_funcs
import surfpy.eikonal._grid_class as _grid_class
import surfpy.eikonal.tomobase as eikonal_tomobase

import surfpy.map_dat as map_dat
map_path    = map_dat.__path__._path[0]

import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy
import copy
import os
import shutil
import obspy
import numpy.ma as ma
import matplotlib

import surfpy.cpt_files as cpt_files
cpt_path    = cpt_files.__path__._path[0]

    
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt


class baseh5(h5py.File):
    """ base hdf5 Markov Chain Monte Carlo inversion based on HDF5 database
    ===================================================================================================================
    version history:
           - first version
    
    --- NOTES: mask data ---
    self[grd_id].attrs['mask_ph']   - existence of phase dispersion data, bool
    self[grd_id].attrs['mask_gr']   - existence of group dispersion data, bool
    self[grd_id].attrs['mask']      - existence of inversion, bool
    self.attrs['mask_inv']          - mask array for inversion, bool array
                                        this array is typically the mask_LD in the original ray tomography database
                                                    or mask_ray in the original hybrid tomography database
    self.attrs['mask_interp']       - mask array for interpolated finalized results, bool array
                                        this array is typically the "mask_inv" in the original ray tomography database
    ===================================================================================================================
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
    
    def update_attrs(self):
        try:
            self.minlon     = self.attrs['minlon']
            self.maxlon     = self.attrs['maxlon']
            self.minlat     = self.attrs['minlat']
            self.maxlat     = self.attrs['maxlat']
            # lontitude attributes
            # # # self.Nlon       = int(self.attrs['Nlon'])
            # # # self.Nlon_inv   = int(self.attrs['Nlon_inv'])
            self.Nlon_eik   = int(self.attrs['Nlon_eik'])
            self.dlon       = self.attrs['dlon']
            self.dlon_inv   = self.attrs['dlon_inv']
            self.dlon_eik   = self.attrs['dlon_eik']
            # latitude attributes
            # # # self.Nlat       = int(self.attrs['Nlat'])
            # # # self.Nlat_inv   = int(self.attrs['Nlat_inv'])
            self.Nlat_eik   = int(self.attrs['Nlat_eik'])
            self.dlat       = self.attrs['dlat']
            self.dlat_eik   = self.attrs['dlat_eik']
            self.dlat_inv   = self.attrs['dlat_inv']
            self.proj_name  = self.attrs['proj_name']
            return True
        except:
            return False
        
    def _get_lon_lat_arr(self):
        """get longitude/latitude array
        """
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        # model grid
        self.lons       = np.arange(int((maxlon-minlon)/self.dlon)+1)*self.dlon+minlon
        self.lats       = np.arange(int((maxlat-minlat)/self.dlat)+1)*self.dlat+minlat
        self.Nlon       = self.lons.size
        self.Nlat       = self.lats.size
        self.lonArr, self.latArr            = np.meshgrid(self.lons, self.lats)
        # inversion grid
        self.lons_inv   = np.arange(int((maxlon-minlon)/self.dlon_inv)+1)*self.dlon_inv+minlon
        self.lats_inv   = np.arange(int((maxlat-minlat)/self.dlat_inv)+1)*self.dlat_inv+minlat
        self.Nlon_inv   = self.lons_inv.size
        self.Nlat_inv   = self.lats_inv.size
        self.lonArr_inv, self.latArr_inv    = np.meshgrid(self.lons_inv, self.lats_inv)
        return
    
    
    def print_info(self):
        """print information of the database
        """
        outstr  = '================================================= Marcov Chain Monte Carlo Inversion Database ===============================================\n'
        outstr  += self.__str__()+'\n'
        outstr  += '-------------------------------------------------------------- headers ---------------------------------------------------------------------\n'
        if not self.update_attrs():
            print ('Empty Database!')
            return
        outstr      += '--- minlon/maxlon                                       - '+str(self.minlon)+'/'+str(self.maxlon)+'\n'
        outstr      += '--- minlat/maxlat                                       - '+str(self.minlat)+'/'+str(self.maxlat)+'\n'
        outstr      += '--- dlon/dlat                                           - '+str(self.dlon)+'/'+str(self.dlat)+'\n'
        outstr      += '--- dlon_inv/dlat_inv                                   - '+str(self.dlon_inv)+'/'+str(self.dlat_inv)+'\n'
        try:
            outstr  += '--- mask_inv (mask_ray_interp - hybridtomo(later updated after read_inv); mask_inv/mask_LD/mask_HD - raytomo) \n' + \
                       '                                                        - shape = ' +str(self.attrs['mask_inv'].shape)+'\n'
        except:
            
            outstr  += '--- mask_inv array NOT initialized  \n'
        if is_interp:   
            outstr  += '--- dlon_interp/dlat_interp (initialized in get_raytomo_mask/get_hybrid_mask) \n'+ \
                       '                                                        - '+str(dlon_interp)+'/'+str(dlat_interp)+'\n'
        try:
            outstr  += '--- mask_interp (mask_ray - hybridtomo( could be a combination of ray/lov database); mask_inv - raytomo) \n' + \
                       '                                                        - shape = '+str(self.attrs['mask_interp'].shape)+'\n'
        except:
            outstr  += '--- mask_interp array NOT initialized  \n'
        outstr      += '---------------------------------------------------------- grid point data -----------------------------------------------------------------\n'
        grd_grp     = self['grd_pts']
        Ngrid       = len(grd_grp.keys())
        outstr      += '--- number of grid points                               - ' +str(Ngrid)+'\n'
        grdid       = grd_grp.keys()[0]
        grdgrp      = grd_grp[grdid]
        outstr      += '--- attributes (data) \n'
        try:
            topo    = grdgrp.attrs['topo']
            outstr  += '    etopo_source                                        - '+grdgrp.attrs['etopo_source']+'\n'
        except:
            outstr  += '    etopo_source                                        - NO \n'
        try:
            sedthk  = grdgrp.attrs['sedi_thk']
            outstr  += '    sedi_thk_source                                     - '+grdgrp.attrs['sedi_thk_source']+'\n'
        except:
            outstr  += '    sedi_thk_source                                     - NO \n'
        try:
            sedthk  = grdgrp.attrs['crust_thk']
            outstr  += '    crust_thk_source                                    - '+grdgrp.attrs['crust_thk_source']+'\n'
        except:
            outstr  += '    crust_thk_source                                    - NO \n'
        outstr      += '--- attributes (inversion results) \n'
        try:
            avg_misfit  = grdgrp.attrs['avg_misfit']
            min_misfit  = grdgrp.attrs['min_misfit']
            mean_misfit = grdgrp.attrs['mean_misfit']
            outstr  += '    avg_misfit/min_misfit/mean_misfit                   - detected    \n'
        except:
            outstr  += '    avg_misfit/min_misfit/mean_misfit                   - NO    \n'
        #----------------------
        outstr      += '--- arrays (data) \n'
        try:
            disp_gr_ray     = grdgrp['disp_gr_ray']
            outstr  += '    disp_gr_ray (Rayleigh wave group dispersion)        - shape = '+str(disp_gr_ray.shape)+'\n'
        except:
            outstr  += '    disp_gr_ray (Rayleigh wave group dispersion)        - NO \n'
        try:
            disp_ph_ray     = grdgrp['disp_ph_ray']
            outstr  += '    disp_ph_ray (Rayleigh wave phase dispersion)        - shape = '+str(disp_ph_ray.shape)+'\n'
        except:
            outstr  += '    disp_ph_ray (Rayleigh wave phase dispersion)        - NO \n'
        try:
            disp_gr_lov     = grdgrp['disp_gr_lov']
            outstr  += '    disp_gr_lov (Love wave group dispersion)            - shape = '+str(disp_gr_lov.shape)+'\n'
        except:
            outstr  += '    disp_gr_lov (Love wave group dispersion)            - NO \n'
        try:
            disp_ph_lov     = grdgrp['disp_ph_lov']
            outstr  += '    disp_ph_lov (Love wave phase dispersion)            - shape = '+str(disp_ph_lov.shape)+'\n'
        except:
            outstr  += '    disp_ph_lov (Love wave phase dispersion)            - NO \n'
        #----------------------
        outstr      += '--- arrays (inversion results, avg model) \n'
        try:
            avg_gr_ray      = grdgrp['avg_gr_ray']
            outstr  += '    avg_gr_ray (Rayleigh group disperion from avg model)- shape = '+str(avg_gr_ray.shape)+'\n'
        except:
            outstr  += '    avg_gr_ray (Rayleigh group disperion from avg model)- NO \n'
        try:
            avg_ph_ray      = grdgrp['avg_ph_ray']
            outstr  += '    avg_ph_ray (Rayleigh phase disperion from avg model)- shape = '+str(avg_ph_ray.shape)+'\n'
        except:
            outstr  += '    avg_ph_ray (Rayleigh phase disperion from avg model)- NO \n'
        try:
            avg_paraval     = grdgrp['avg_paraval']
            outstr  += '    avg_paraval (model parameter array of avg model)    - shape = '+str(avg_paraval.shape)+'\n'
        except:
            outstr  += '    avg_paraval (model parameter array of avg model)    - NO \n'
        # min
        outstr      += '--- arrays (inversion results, min model) \n'
        try:
            min_gr_ray      = grdgrp['min_gr_ray']
            outstr  += '    min_gr_ray (Rayleigh group disperion from min model)- shape = '+str(min_gr_ray.shape)+'\n'
        except:
            outstr  += '    min_gr_ray (Rayleigh group disperion from min model)- NO \n'
        try:
            min_ph_ray      = grdgrp['min_ph_ray']
            outstr  += '    min_ph_ray (Rayleigh phase disperion from min model)- shape = '+str(min_ph_ray.shape)+'\n'
        except:
            outstr  += '    avg_ph_ray (Rayleigh phase disperion from min model)- NO \n'
        try:
            min_paraval     = grdgrp['min_paraval']
            outstr  += '    min_paraval (model parameter array of min model)    - shape = '+str(min_paraval.shape)+'\n'
        except:
            outstr  += '    min_paraval (model parameter array of min model)    - NO \n'
        ################
        outstr      += '--- arrays (inversion results, statistical) \n'
        try:
            sem_paraval     = grdgrp['sem_paraval']
            outstr  += '    sem_paraval (SEM of model parameter array)          - shape = '+str(sem_paraval.shape)+'\n'
        except:
            outstr  += '    sem_paraval (SEM of model parameter array)          - NO \n'
        try:
            std_paraval     = grdgrp['std_paraval']
            outstr  += '    std_paraval (STD of model parameter array)          - shape = '+str(std_paraval.shape)+'\n'
        except:
            outstr  += '    std_paraval (STD of model parameter array)          - NO \n'
        try:
            zArr_ensemble   = grdgrp['zArr_ensemble']
            outstr  += '    zArr_ensemble (depth array for ensemble of models)  - shape = '+str(zArr_ensemble.shape)+'\n'
        except:
            outstr  += '    zArr_ensemble (depth array for ensemble of models)    - NO \n'
        try:
            vs_lower_bound  = grdgrp['vs_lower_bound']
            vs_upper_bound  = grdgrp['vs_upper_bound']
            vs_mean         = grdgrp['vs_mean']
            vs_std          = grdgrp['vs_std']
            outstr  += '    vs arrays (upper/lower bounds, std, mean)           - shape = '+str(vs_mean.shape)+'\n'
        except:
            outstr  += '    vs arrays (upper/lower bounds, std, mean)           - NO \n'
        outstr  += '--------------------------------------------------------------- Models ---------------------------------------------------------------------\n'
        subgrps     = self.keys()
        if 'mask' in subgrps:
            outstr  += '--- mask array detected    \n'
        if 'topo' in subgrps:
            outstr  += '--- topo array (topography data for dlon/dlat)    \n' +\
                       '                                                        - shape = '+str(self['topo'].shape)+'\n'
        if 'topo_interp' in subgrps:
            outstr  += '--- topo_interp array (topography data for dlon_interp/dlat_interp)     \n'+\
                       '                                                        - shape = '+str(self['topo_interp'].shape)+'\n'
        # average model
        if 'avg_paraval' in subgrps:
            outstr  += '!!! average model \n'
            subgrp  = self['avg_paraval']
            if '0_org' in subgrp.keys():
                outstr\
                    += '--- original model parameters (2D arrays)               - shape = '+str(subgrp['0_org'].shape)+'\n'
            if 'vs_org' in subgrp.keys():
                outstr\
                    += '--- original 3D model  (3D arrays)                      - shape = '+str(subgrp['vs_org'].shape)+'\n'
            if '0_smooth' in subgrp.keys():
                outstr\
                    += '--- smooth model parameters (2D arrays)                 - shape = '+str(subgrp['0_smooth'].shape)+'\n'
            if 'vs_smooth' in subgrp.keys():
                outstr\
                    += '--- smooth 3D model  (3D arrays)                        - shape = '+str(subgrp['vs_smooth'].shape)+'\n'
        # minimum misfit model
        if 'min_paraval' in subgrps:
            outstr  += '!!! minimum misfit model \n'
            subgrp  = self['min_paraval']
            if '0_org' in subgrp.keys():
                outstr\
                    += '--- original model parameters (2D arrays)               - shape = '+str(subgrp['0_org'].shape)+'\n'
            if 'vs_org' in subgrp.keys():
                outstr\
                    += '--- original 3D model  (3D arrays)                      - shape = '+str(subgrp['vs_org'].shape)+'\n'
            if '0_smooth' in subgrp.keys():
                outstr\
                    += '--- smooth model parameters (2D arrays)                 - shape = '+str(subgrp['0_smooth'].shape)+'\n'
            if 'vs_smooth' in subgrp.keys():
                outstr\
                    += '--- smooth 3D model  (3D arrays)                        - shape = '+str(subgrp['vs_smooth'].shape)+'\n'
        outstr += '============================================================================================================================================\n'
        print (outstr)
        return
 
    def set_spacing(self, dlon = 0.2, dlat = 0.2, dlon_inv = 0.5, dlat_inv = 0.5):
        self.attrs.create(name = 'dlon', data = dlon)
        self.attrs.create(name = 'dlat', data = dlat)
        self.attrs.create(name = 'dlon_inv', data = dlon_inv)
        self.attrs.create(name = 'dlat_inv', data = dlat_inv)
        return
    
    #==================================================================
    # I/O functions
    #==================================================================
    
    def load_eikonal(self, inh5fname, runid = 0, dtype = 'ph', wtype = 'ray', Tmin = -999, Tmax = 999, semfactor = 2.):
        """read eikonal tomography results
        =================================================================================
        ::: input :::
        inh5fname   - input hdf5 file name
        runid       - id of run for the ray tomography
        dtype       - data type (ph or gr)
        wtype       - wave type (ray or lov)
        Tmin, Tmax  - minimum and maximum period to extract from the tomographic results
        semfactor   - factor to multiply for standard error of the mean (sem)
                        suggested by Lin et al. (2009)
        =================================================================================
        """
        if dtype is not 'ph' and dtype is not 'gr':
            raise ValueError('data type can only be ph or gr!')
        if wtype is not 'ray' and wtype is not 'lov':
            raise ValueError('wave type can only be ray or lov!')
        dset            = eikonal_tomobase.baseh5(inh5fname)
        #--------------------------------------------
        # header information from input hdf5 file
        #--------------------------------------------
        dset.get_mask(runid = runid, Tmin = Tmin, Tmax = Tmax)
        pers            = dset.pers
        minlon          = dset.minlon
        maxlon          = dset.maxlon
        minlat          = dset.minlat
        maxlat          = dset.maxlat
        dlon_eik        = dset.dlon
        Nlon_eik        = dset.Nlon
        dlat_eik        = dset.dlat
        Nlat_eik        = dset.Nlat
        mask_eik        = dset.attrs['mask']
        proj_name       = dset.proj_name
        outdir          = os.path.dirname(self.filename) + '/in_eikonal_interp'
        # save attributes
        self.attrs.create(name = 'minlon', data = minlon, dtype = 'f')
        self.attrs.create(name = 'maxlon', data = maxlon, dtype = 'f')
        self.attrs.create(name = 'minlat', data = minlat, dtype = 'f')
        self.attrs.create(name = 'maxlat', data = maxlat, dtype = 'f')
        self.attrs.create(name = 'proj_name', data = proj_name)
        self.attrs.create(name = 'dlon_eik', data = dlon_eik)
        self.attrs.create(name = 'dlat_eik', data = dlat_eik)
        self.attrs.create(name = 'Nlon_eik', data = Nlon_eik)
        self.attrs.create(name = 'Nlat_eik', data = Nlat_eik)
        self.attrs.create(name = 'mask_eik', data = mask_eik)
        self.update_attrs()
        self._get_lon_lat_arr()
        # mask of the model
        mask        = _model_funcs.mask_interp(dlon = dlon_eik, dlat = dlat_eik, minlon = self.minlon, \
                    minlat = self.minlat, maxlon = self.maxlon, maxlat = self.maxlat, mask_in = mask_eik,\
                    dlon_out = self.dlon, dlat_out = self.dlat, inear_true_false = False)
        self.attrs.create(name = 'mask', data = mask)
        # mask of inversion
        mask_inv    = _model_funcs.mask_interp(dlon = dlon_eik, dlat = dlat_eik, minlon = self.minlon, \
                    minlat = self.minlat, maxlon = self.maxlon, maxlat = self.maxlat, mask_in = mask_eik,\
                    dlon_out = self.dlon_inv, dlat_out = self.dlat_inv, inear_true_false = False)
        self.attrs.create(name = 'mask_inv', data = mask_inv)
        #====================================================
        # store interpolate eikonal maps to inversion grid
        #====================================================
        dat_grp         = self.create_group(name = 'interp_eikonal_data_' + wtype +'_'+dtype)
        period_arr      = []
        dataid          = 'tomo_stack_'+str(runid)
        eik_grp         = dset[dataid]
        for per in pers:
            try:
                pergrp      = eik_grp['%d_sec' %per]
            except KeyError:
                continue
            period_arr.append(per)
            dat_per_grp = dat_grp.create_group(name = '%d_sec' %per)
            mask_per    = pergrp['mask'][()]
            vel_per     = pergrp['vel_iso'][()]
            un_per      = pergrp['vel_sem'][()]
            lons        = dset.lonArr[np.logical_not(mask_per)]
            lats        = dset.latArr[np.logical_not(mask_per)]
            # interpolate velocity
            C           = vel_per[np.logical_not(mask_per)]
            gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon_inv, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat_inv, period = per, \
                            evlo = -1., evla = -1., fieldtype = 'phvel', evid = 'INEIK')
            gridder.read_array(inlons = lons, inlats = lats, inzarr = C)
            outfname    = 'INEIK_phvel.lst'
            prefix      = 'INEIK_'
            working_dir = outdir + '/%g_sec' %per
            if not os.path.isdir(working_dir):
                os.makedirs(working_dir)
            gridder.interp_surface(workingdir = working_dir, outfname = outfname)
            dat_per_grp.create_dataset(name = 'vel_iso', data = gridder.Zarr )
            # interpolate uncertainties
            un          = un_per[np.logical_not(mask_per)]
            gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon_inv, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat_inv, period = per, \
                            evlo = -1., evla = -1., fieldtype = 'phvelun', evid = 'INEIK')
            gridder.read_array(inlons = lons, inlats = lats, inzarr = un)
            # outfname    = 'INEIK_phvelun.lst'
            # prefix      = 'INEIK_'
            # working_dir = outdir + '/%g_sec' %per
            # if not os.path.isdir(working_dir):
            #     os.makedirs(working_dir)
            # gridder.interp_surface(workingdir = working_dir, outfname = outfname)
            gridder.interp_surface()            
            dat_per_grp.create_dataset(name = 'vel_sem', data = gridder.Zarr )
        #remove working directory
        shutil.rmtree(outdir)
        grd_grp             = self.require_group('grd_pts')
        for ilat in range(self.Nlat_inv):
            for ilon in range(self.Nlon_inv):
                if mask_inv[ilat, ilon]:
                    continue
                data_str    = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                group       = grd_grp.require_group( name = data_str )
                disp_v      = np.array([])
                disp_un     = np.array([])
                T           = np.array([])
                for per in period_arr:
                    if per < Tmin or per > Tmax:
                        continue
                    try:
                        dat_per_grp = dat_grp['%g_sec'%( per )]
                        vel         = dat_per_grp['vel_iso'][()]
                        vel_sem     = dat_per_grp['vel_sem'][()]
                    except KeyError:
                        print ('No data for T = '+str(per)+' sec')
                        continue
                    T               = np.append(T, per)
                    disp_v          = np.append(disp_v, vel[ilat, ilon])
                    disp_un         = np.append(disp_un, vel_sem[ilat, ilon])
                data                = np.zeros((3, T.size))
                data[0, :]          = T[:]
                data[1, :]          = disp_v[:]
                data[2, :]          = disp_un[:] * semfactor
                group.create_dataset(name = 'disp_'+dtype+'_'+wtype, data = data)
        self.attrs.create(name = 'period_array', data = np.asarray(period_arr), dtype = 'f')
        self.attrs.create(name = 'sem_factor', data = semfactor, dtype = 'f')
        dset.close()
        return
    
    def load_crust_thickness(self, fname = None, source = 'crust_1.0', overwrite = False):
        """read crust thickness from a txt file (crust 1.0 model)
        """
        try:
            if self.attrs['is_crust_thk']:
                if overwrite:
                    print ('!!! reference crustal thickness data exists, OVERWRITE!')
                else:
                    print ('!!! reference crustal thickness data exist!')
                    return
        except:
            pass
        if fname is None:
            fname   = map_path+'/crsthk.xyz'
        if not os.path.isfile(fname):
            raise ValueError('!!! reference crust thickness file not exists!')
        inarr       = np.loadtxt(fname)
        lonArr      = inarr[:, 0]
        lonArr      = lonArr.reshape(int(lonArr.size/360), 360)
        latArr      = inarr[:, 1]
        latArr      = latArr.reshape(int(latArr.size/360), 360)
        depthArr    = inarr[:, 2]
        depthArr    = depthArr.reshape(int(depthArr.size/360), 360)
        grd_grp     = self.require_group('grd_pts')
        for grp_id in grd_grp.keys():
            grp     = grd_grp[grp_id]
            split_id= grp_id.split('_')
            grd_lon = float(split_id[0])
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            whereArr= np.where((lonArr>=grd_lon)*(latArr>=grd_lat))
            ind_lat = whereArr[0][-1]
            ind_lon = whereArr[1][0]
            # check
            lon     = lonArr[ind_lat, ind_lon]
            lat     = latArr[ind_lat, ind_lon]
            if abs(lon-grd_lon) > 1. or abs(lat - grd_lat) > 1.:
                print ('ERROR!', lon, lat, grd_lon, grd_lat)
            depth   = depthArr[ind_lat, ind_lon]
            grp.attrs.create(name = 'crust_thk', data = depth)
            grp.attrs.create(name = 'crust_thk_source', data = source)
        self.attrs.create(name = 'is_crust_thk', data = True)
        return
    
    def read_sediment_thickness(self, fname = None, source='crust_1.0', overwrite = False):
        """read sediment thickness from a txt file (crust 1.0 model)
        """
        try:
            if self.attrs['is_sediment_thk']:
                if overwrite:
                    print ('!!! reference sedimentary thickness data exists, OVERWRITE!')
                else:
                    print ('!!! reference sedimentary thickness data exists!')
                    return
        except:
            pass
        if fname is None:
            fname   = map_path+'/sedthk.xyz'
        if not os.path.isfile(fname):
            raise ValueError('!!! reference crust thickness file not exists!')
        inarr       = np.loadtxt(fname)
        lonArr      = inarr[:, 0]
        lonArr      = lonArr.reshape(int(lonArr.size/360), 360)
        latArr      = inarr[:, 1]
        latArr      = latArr.reshape(int(latArr.size/360), 360)
        depthArr    = inarr[:, 2]
        depthArr    = depthArr.reshape(int(depthArr.size/360), 360)
        grd_grp     = self.require_group('grd_pts')
        for grp_id in grd_grp.keys():
            grp     = grd_grp[grp_id]
            split_id= grp_id.split('_')
            grd_lon = float(split_id[0])
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            whereArr= np.where((lonArr>=grd_lon)*(latArr>=grd_lat))
            ind_lat = whereArr[0][-1]
            ind_lon = whereArr[1][0]
            # check
            lon     = lonArr[ind_lat, ind_lon]
            lat     = latArr[ind_lat, ind_lon]
            if abs(lon-grd_lon) > 1. or abs(lat - grd_lat) > 1.:
                print ('ERROR!', lon, lat, grd_lon, grd_lat)
            depth   = depthArr[ind_lat, ind_lon]
            grp.attrs.create(name='sediment_thk', data=depth)
            grp.attrs.create(name='sediment_thk_source', data=source)
        self.attrs.create(name = 'is_sediment_thk', data = True)
        return
    
    def load_CU_model(self, fname = None):
        """read reference model from a hdf5 file (CU Global Vs model)
        """
        try:
            if self.attrs['is_reference_vs']:
                print ('!!! reference Vs exists!')
                return
        except:
            pass
        if fname is None:
            fname   = map_path+'/CU_SDT1.0.mod.h5'
        if not os.path.isfile(fname):
            raise ValueError('!!! reference crust thickness file not exists!')
        indset      = h5py.File(fname, mode = 'r')
        lons        = np.mgrid[0.:359.:2.]
        lats        = np.mgrid[-88.:89.:2.]
        grd_grp     = self.require_group('grd_pts')
        for grp_id in grd_grp.keys():
            grp         = grd_grp[grp_id]
            split_id    = grp_id.split('_')
            try:
                grd_lon = float(split_id[0])
            except ValueError:
                continue
            if grd_lon < 0.:
                grd_lon += 360.
            grd_lat = float(split_id[1])
            try:
                ind_lon         = np.where(lons>=grd_lon)[0][0]
            except:
                ind_lon         = lons.size - 1
            try:
                ind_lat         = np.where(lats>=grd_lat)[0][0]
            except:
                ind_lat         = lats.size - 1
            if lons[ind_lon] - grd_lon > 1. and ind_lon > 0:
                ind_lon         -= 1
            if lats[ind_lat] - grd_lat > 1. and ind_lat > 0:
                ind_lat         -= 1
            if abs(lons[ind_lon] - grd_lon) > 1. or abs(lats[ind_lat] - grd_lat) > 1.:
                print ('ERROR!', lons[ind_lon], lats[ind_lat] , grd_lon, grd_lat)
            data        = indset[str(lons[ind_lon])+'_'+str(lats[ind_lat])][()]
            grp.create_dataset(name='reference_vs', data = data)
        indset.close()
        self.attrs.create(name = 'is_reference_vs', data = True)
        return
    
    def load_etopo(self, fname = None, source='etopo2', overwrite = False):
        """read topography data from etopo2
        ============================================================================
        ::: input :::
        infname     - input file name
        source      - source name (default - etopo2)
        ============================================================================
        """
        try:
            if self.attrs['is_topo']:
                if overwrite:
                    print ('!!! topography data exists, OVERWRITE!')
                else:
                    print ('!!! topography data exists!')
                    return
        except:
            pass
        if fname is None:
            fname   = map_path+'/etopo2.h5'
        if not os.path.isfile(fname):
            raise ValueError('!!! topography file not exists!')
        indset      = h5py.File(fname, mode = 'r')
        etopo       = indset['etopo'][()]
        lons        = indset['longitudes'][()]
        lats        = indset['latitudes'][()]
        grd_grp     = self['grd_pts']
        for grp_id in grd_grp.keys():
            grp     = grd_grp[grp_id]
            split_id= grp_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat         = float(split_id[1])
            try:
                ind_lon     = np.where(lons>=grd_lon)[0][0]
            except:
                ind_lon     = lons.size - 1
            try:
                ind_lat     = np.where(lats>=grd_lat)[0][0]
            except:
                ind_lat     = lats.size - 1
            if lons[ind_lon] - grd_lon > (1./30.):
                ind_lon     -= 1
            if lats[ind_lat] - grd_lat > (1./30.):
                ind_lat     -= 1
            if abs(lons[ind_lon] - grd_lon) > 1./30. or abs(lats[ind_lat] - grd_lat) > 1./30.:
                print ('ERROR!', lons[ind_lon], lats[ind_lat] , grd_lon, grd_lat)
            z               = etopo[ind_lat, ind_lon]/1000. # convert to km
            grp.attrs.create(name = 'topo', data = z)
            grp.attrs.create(name = 'etopo_source', data = source)
        self.attrs.create(name = 'is_topo', data = True)
        return
    
    #==================================================================
    # function inspection of the input data
    #==================================================================
#     
    def get_disp(self, lon, lat, wtype='ray'):
        if lon < 0.:
            lon     += 360.
        data_str    = str(lon)+'_'+str(lat)
        grd_grp     = self['grd_pts']
        try:
            grp     = grd_grp[data_str]
        except:
            print ('No data at longitude =',lon,' lattitude =',lat)
            return
        try:
            disp_ph = grp['disp_ph_'+wtype]
        except:
            disp_ph = None
            pass
        try:
            disp_gr = grp['disp_gr_'+wtype]
        except:
            disp_gr = None
            pass
        return disp_ph, disp_gr
    
    def plot_disp(self, lon, lat, wtype='ray', derive_group = False, ploterror = False, showfig = True):
        """
        plot dispersion data given location of the grid point
        ==========================================================================================
        ::: input :::
        lon/lat     - location of the grid point
        wtype       - type of waves (ray or lov)
        derivegr    - compute and plot the group velocities derived from phase velocities or not
        ploterror   - plot uncertainties or not
        showfig     - show the figure or not
        ==========================================================================================
        """
        if lon < 0.:
            lon     += 360.
        data_str    = str(lon)+'_'+str(lat)
        grd_grp     = self['grd_pts']
        try:
            grp     = grd_grp[data_str]
        except:
            print ('No data at longitude =',lon,' lattitude =',lat)
            return
        plt.figure()
        ax      = plt.subplot()
        try:
            disp_ph = grp['disp_ph_'+wtype]
            if ploterror:
                plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], color='b', lw=3, label='phase')
            else:
                plt.plot(disp_ph[0, :], disp_ph[1, :], 'bo-', lw=3, ms=10, label='phase')
        except:
            pass
        # compute and plot the derived group velocities
        if derivegr:
            import scipy.interpolate
            CubicSpl    = scipy.interpolate.CubicSpline(disp_ph[0, :], disp_ph[1, :])
            Tmin        = disp_ph[0, 0]
            Tmax        = disp_ph[0, -1]
            Tinterp     = np.mgrid[Tmin:Tmax:0.1]
            Cinterp     = CubicSpl(Tinterp)
            diffC       = Cinterp[2:] - Cinterp[:-2]
            dCdTinterp  = diffC/0.2
            sU          = 1./Cinterp[1:-1] + (Tinterp[1:-1]/(Cinterp[1:-1])**2)*dCdTinterp
            derived_U   = 1./sU
            plt.plot(Tinterp[1:-1], derived_U, 'k--', lw=3, label='derived group')
        try:
            disp_gr = grp['disp_gr_'+wtype]
            if ploterror:
                plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], color='r', lw=3, label='group')
            else:
                plt.plot(disp_gr[0, :], disp_gr[1, :], 'ro-', lw=3, ms=10, label='group')
        except:
            pass
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('Period (sec)', fontsize=30)
        plt.ylabel('Velocity (km/sec)', fontsize=30)
        if lon > 180.:
            lon     -= 360.
        plt.title('longitude = '+str(lon)+' latitude = '+str(lat), fontsize=30)
        plt.legend(loc=0, fontsize=20)
        if showfig:
            plt.show()
        return

    
