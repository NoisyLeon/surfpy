# -*- coding: utf-8 -*-
"""
A python module for inversion with only dispersion data based on hdf5 database

:Methods:


:Dependencies:
    pyasdf and its dependencies
    ObsPy  and its dependencies
    pyproj
    Basemap
    pyfftw 0.10.3 (optional)
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy
import warnings
import copy
import os, shutil
from functools import partial
import multiprocessing
from subprocess import call
import obspy
import vprofile, mcpost, mcpost_vti, vmodel
import time
import numpy.ma as ma
import scipy.signal
import matplotlib

from pyproj import Geod

import colormaps, pycpt
#import field2d_earth
from mpl_toolkits.basemap import Basemap, shiftgrid, cm, interp

import numba
import time

def _get_vs_2d(z0, z1, zArr, vs_3d):
    Nlat, Nlon, Nz  = vs_3d.shape
    vs_out          = np.zeros((Nlat, Nlon))
    for ilat in range(Nlat):
        for ilon in range(Nlon):
            ind     = np.where((zArr > z0[ilat, ilon])*(zArr < z1[ilat, ilon]))[0]
            vs_temp = vs_3d[ilat, ilon, ind].mean()
            vs_out[ilat, ilon]\
                    = vs_temp
    return vs_out

@numba.jit(numba.float64[:, :, :](numba.float64[:, ], numba.float64[:, :, :], numba.float64))
def _get_avg_vs3d(zArr, vs3d, depthavg):
    tvs3d           = vs3d.copy()
    Nlat, Nlon, Nz  = vs3d.shape
    Nz              = zArr.size
    # # vs_out          = np.zeros((Nlat, Nlon))
    # for ilat in range(Nlat):
    #     for ilon in range(Nlon):
    #         
    for i in range(Nz):
        z       = zArr[i]
        print i
        if z < depthavg:
            tvs3d[:, :, i]  = (vs3d[:, :, zArr <= 2.*depthavg]).mean(axis=2)
            continue
        index   = (zArr <= z + depthavg) + (zArr >= z - depthavg)
        tvs3d[:, :, i]  = (vs3d[:, :, index]).mean(axis=2)
    return tvs3d

def to_percent(y, position):
     # Ignore the passed in position. This has the effect of scaling the default
     # tick locations.
     s = '%.0f' %(100. * y)
     # The percent symbol needs escaping in latex
     if matplotlib.rcParams['text.usetex'] is True:
         return s + r'$\%$'
     else:
         return s + '%'
    
def read_slab_contour(infname, depth):
    ctrlst  = []
    lonlst  = []
    latlst  = []
    with open(infname, 'rb') as fio:
        newctr  = False
        for line in fio.readlines():
            if line.split()[0] is '>':
                newctr  = True
                if len(lonlst) != 0:
                    ctrlst.append([lonlst, latlst])
                lonlst  = []
                latlst  = []
                z       = -float(line.split()[1])
                if z == depth:
                    skipflag    = False
                else:
                    skipflag    = True
                continue
            if skipflag:
                continue
            lonlst.append(float(line.split()[0]))
            latlst.append(float(line.split()[1]))
    return ctrlst
                    
    
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    if os.path.isfile(base_cmap):
        import pycpt
        base    = pycpt.load.gmtColormap(base_cmap)
    else:
        base    = plt.cm.get_cmap(base_cmap)
    color_list  = base(np.linspace(0, 1, N))
    cmap_name   = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_fault_lines(mapobj, infname, lw=2, color='red'):
    with open(infname, 'rb') as fio:
        is_new  = False
        lonlst  = []
        latlst  = []
        for line in fio.readlines():
            if line.split()[0] == '>':
                x, y  = mapobj(lonlst, latlst)
                mapobj.plot(x, y,  lw = lw, color=color)
                # # # m.plot(xslb, yslb,  lw = 3, color='white')
                lonlst  = []
                latlst  = []
                continue
            lonlst.append(float(line.split()[0]))
            latlst.append(float(line.split()[1]))
        x, y  = mapobj(lonlst, latlst)
        mapobj.plot(x, y,  lw = lw, color=color, zorder=0)

class invhdf5(h5py.File):
    """ An object to for Markov Chain Monte Carlo inversion based on HDF5 database
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
    def print_info(self):
        """
        print information of the database
        """
        outstr  = '================================================= Marcov Chain Monte Carlo Inversion Database ===============================================\n'
        outstr  += self.__str__()+'\n'
        outstr  += '-------------------------------------------------------------- headers ---------------------------------------------------------------------\n'
        try:
            minlon          = self.attrs['minlon']
            maxlon          = self.attrs['maxlon']
            minlat          = self.attrs['minlat']
            maxlat          = self.attrs['maxlat']
            dlon            = self.attrs['dlon']
            dlat            = self.attrs['dlat']
            is_interp       = self.attrs['is_interp']
            if is_interp:
                dlon_interp = self.attrs['dlon_interp']
                dlat_interp = self.attrs['dlat_interp']
        except:
            print 'Empty Database!'
            return
        outstr      += '--- minlon/maxlon                                       - '+str(minlon)+'/'+str(maxlon)+'\n'
        outstr      += '--- minlat/maxlat                                       - '+str(minlat)+'/'+str(maxlat)+'\n'
        outstr      += '--- dlon/dlat                                           - '+str(dlon)+'/'+str(dlat)+'\n'
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
        print outstr
        return
#     
    def _get_lon_lat_arr(self, is_interp=False):
        """Get longitude/latitude array
        """
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        if is_interp:
            dlon        = self.attrs['dlon_interp']
            dlat        = self.attrs['dlat_interp']
        else:
            dlon        = self.attrs['dlon']
            dlat        = self.attrs['dlat']
        self.lons       = np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon
        self.lats       = np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat
        self.Nlon       = self.lons.size
        self.Nlat       = self.lats.size
        self.lonArr, self.latArr \
                        = np.meshgrid(self.lons, self.lats)
        return
    
    def _set_interp(self, dlon, dlat):
        self.attrs.create(name = 'dlon_interp', data=dlon)
        self.attrs.create(name = 'dlat_interp', data=dlat)
        return
    
    #==================================================================
    # functions before MC inversion runs
    #==================================================================
    
    def read_hybridtomo_dbase(self, inh5fname, runid, dtype='ph', wtype='ray', create_header=True, \
                Tmin=-999, Tmax=999, verbose=False, semfactor=2.):
        """
        read hybrid tomography database
        =================================================================================
        ::: input :::
        inh5fname   - input hdf5 file name
        runid       - id of run for the ray tomography
        dtype       - data type (ph or gr)
        wtype       - wave type (ray or lov)
        Tmin, Tmax  - minimum and maximum period to extract from the tomographic results
        res         - resolution for grid points, default is LD, low-definition
        semfactor   - factor to multiply for standard error of the mean (sem)
                        suggested by Fan-chi Lin
        =================================================================================
        """
        if dtype is not 'ph' and dtype is not 'gr':
            raise ValueError('data type can only be ph or gr!')
        if wtype is not 'ray' and wtype is not 'lov':
            raise ValueError('wave type can only be ray or lov!')
        indset          = h5py.File(inh5fname)
        #--------------------------------------------
        # header information from input hdf5 file
        #--------------------------------------------
        dataid          = 'merged_tomo_'+str(runid)
        pers            = indset.attrs['period_array']
        grp             = indset[dataid]
        minlon          = indset.attrs['minlon']
        maxlon          = indset.attrs['maxlon']
        minlat          = indset.attrs['minlat']
        maxlat          = indset.attrs['maxlat']
        dlon_interp     = indset.attrs['dlon_interp']
        dlat_interp     = indset.attrs['dlat_interp']
        dlon            = indset.attrs['dlon']
        dlat            = indset.attrs['dlat']
        mask_ray        = indset[dataid+'/mask_ray_interp']
        if create_header:
            self.attrs.create(name = 'minlon', data=minlon, dtype='f')
            self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
            self.attrs.create(name = 'minlat', data=minlat, dtype='f')
            self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
            self.attrs.create(name = 'dlon', data=dlon_interp)
            self.attrs.create(name = 'dlat', data=dlat_interp)
        self._get_lon_lat_arr()
        # create mask_inv array indicating inversions
        self.attrs.create(name='mask_inv', data = mask_ray)
        Traymax             = grp.attrs['T_ray_max']
        # added on 2019-03-17
        grd_grp             = self.require_group('grd_pts')
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if mask_ray[ilat, ilon]:
                    continue
                data_str    = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                group       = grd_grp.require_group( name = data_str )
                disp_v      = np.array([])
                disp_un     = np.array([])
                T           = np.array([])
                for per in pers:
                    if per < Tmin or per > Tmax:
                        continue
                    try:
                        pergrp      = grp['%g_sec'%( per )]
                        vel         = pergrp['vel_iso_interp'].value
                        vel_sem     = pergrp['vel_sem_interp'].value
                    except KeyError:
                        if verbose:
                            print 'No data for T = '+str(per)+' sec'
                        continue
                    if per > Traymax:
                        mask_interp = pergrp['mask_interp'].value
                        if mask_interp[ilat, ilon]:
                            continue
                    T               = np.append(T, per)
                    disp_v          = np.append(disp_v, vel[ilat, ilon])
                    disp_un         = np.append(disp_un, vel_sem[ilat, ilon])
                data                = np.zeros((3, T.size))
                data[0, :]          = T[:]
                data[1, :]          = disp_v[:]
                data[2, :]          = disp_un[:] * semfactor
                group.create_dataset(name='disp_'+dtype+'_'+wtype, data=data)
        indset.close()
        return
    
    def read_raytomo_dbase(self, inh5fname, runid, dtype='ph', wtype='ray', create_header=True, \
                           Tmin=-999, Tmax=999, verbose=False, res='LD', semfactor=2.):
        """
        read ray tomography database
        =================================================================================
        ::: input :::
        inh5fname   - input hdf5 file name
        runid       - id of run for the ray tomography
        dtype       - data type (ph or gr)
        wtype       - wave type (ray or lov)
        Tmin, Tmax  - minimum and maximum period to extract from the tomographic results
        res         - resolution for grid points, default is LD, low-definition
        semfactor   - factor to multiply for standard error of the mean (sem)
                        suggested by Fan-chi Lin
        =================================================================================
        """
        if dtype is not 'ph' and dtype is not 'gr':
            raise ValueError('data type can only be ph or gr!')
        if wtype is not 'ray' and wtype is not 'lov':
            raise ValueError('wave type can only be ray or lov!')
        indset          = h5py.File(inh5fname)
        #--------------------------------------------
        # header information from input hdf5 file
        #--------------------------------------------
        dataid          = 'reshaped_qc_run_'+str(runid)
        pers            = indset.attrs['period_array']
        grp             = indset[dataid]
        isotropic       = grp.attrs['isotropic']
        org_grp         = indset['qc_run_'+str(runid)]
        if org_grp.attrs['datatype'] != dtype:
            raise ValueError('incompatible data type, input = '+dtype+', database = '+org_grp.attrs['datatype'])
        minlon          = indset.attrs['minlon']
        maxlon          = indset.attrs['maxlon']
        minlat          = indset.attrs['minlat']
        maxlat          = indset.attrs['maxlat']
        if isotropic:
            print 'isotropic inversion results do not output gaussian std!'
            return
        if res == 'LD':
            sfx         = '_LD'
        elif res == 'HD':
            sfx         = '_HD'
        else:
            sfx         = ''
        dlon_interp     = org_grp.attrs['dlon'+sfx]
        dlat_interp     = org_grp.attrs['dlat'+sfx]
        dlon            = org_grp.attrs['dlon']
        dlat            = org_grp.attrs['dlat']
        if sfx == '':
            mask        = indset[dataid+'/mask_inv']
        else:
            mask        = indset[dataid+'/mask'+sfx]
        if create_header:
            self.attrs.create(name = 'minlon', data=minlon, dtype='f')
            self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
            self.attrs.create(name = 'minlat', data=minlat, dtype='f')
            self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
            self.attrs.create(name = 'dlon', data=dlon_interp)
            self.attrs.create(name = 'dlat', data=dlat_interp)
        self._get_lon_lat_arr()
        # create mask_ray array indicating inversions
        if wtype == 'lov':
            mask_inv    = mask + self.attrs['mask_inv']
            self.attrs.create(name='mask_inv', data = mask_inv)
        else:
            self.attrs.create(name='mask_inv', data = mask)
        grd_grp             = self.require_group('grd_pts')
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if mask[ilat, ilon]:
                    continue
                data_str    = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                group       = grd_grp.create_group( name = data_str )
                disp_v      = np.array([])
                disp_un     = np.array([])
                T           = np.array([])
                for per in pers:
                    if per < Tmin or per > Tmax:
                        continue
                    try:
                        pergrp      = grp['%g_sec'%( per )]
                        vel         = pergrp['vel_iso'+sfx].value
                        vel_sem     = pergrp['vel_sem'+sfx].value
                    except KeyError:
                        if verbose:
                            print 'No data for T = '+str(per)+' sec'
                        continue
                    T               = np.append(T, per)
                    disp_v          = np.append(disp_v, vel[ilat, ilon])
                    disp_un         = np.append(disp_un, vel_sem[ilat, ilon])
                data                = np.zeros((3, T.size))
                data[0, :]          = T[:]
                data[1, :]          = disp_v[:]
                data[2, :]          = disp_un[:] * semfactor
                group.create_dataset(name='disp_'+dtype+'_'+wtype, data=data)
        indset.close()
        return
    
    def read_raytomo_dbase_group(self, inh5fname, runid, wtype='ray',\
            create_grp = False, Tmin=-999, Tmax=999, verbose=False, res='LD', semfactor=2.):
        """
        read group dispersion data from ray tomography database
        =================================================================================
        ::: input :::
        inh5fname   - input hdf5 file name
        runid       - id of run for the ray tomography
        wtype       - wave type (ray or lov)
        Tmin, Tmax  - minimum and maximum period to extract from the tomographic results
        res         - resolution for grid points, default is LD, low-definition
        semfactor   - factor to multiply for standard error of the mean (sem)
        =================================================================================
        """
        if wtype is not 'ray' and wtype is not 'lov':
            raise ValueError('wave type can only be ray or lov!')
        indset          = h5py.File(inh5fname)
        #--------------------------------------------
        # header information from input hdf5 file
        #--------------------------------------------
        dataid          = 'reshaped_qc_run_'+str(runid)
        pers            = indset.attrs['period_array']
        grp             = indset[dataid]
        isotropic       = grp.attrs['isotropic']
        org_grp         = indset['qc_run_'+str(runid)]
        if isotropic:
            print 'isotropic inversion results do not output gaussian std!'
            return
        if res == 'LD':
            sfx         = '_LD'
        elif res == 'HD':
            sfx         = '_HD'
        else:
            sfx         = ''
        if sfx == '':
            mask_new    = indset[dataid+'/mask_inv']
        else:
            mask_new    = indset[dataid+'/mask'+sfx]
        mask_org        = self.attrs['mask_inv']
        mask            = mask_org + mask_new
        self.attrs.create(name='mask_inv', data = mask)
        self._get_lon_lat_arr()
        grd_grp         = self.require_group('grd_pts')
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if mask[ilat, ilon]:
                    continue
                data_str    = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                try:
                    group   = grd_grp[data_str]
                except:
                    if create_grp:
                        group  \
                            = grd_grp.create_group( name = data_str )
                    else:
                        continue
                disp_v      = np.array([])
                T           = np.array([])
                for per in pers:
                    if per < Tmin or per > Tmax:
                        continue
                    try:
                        pergrp      = grp['%g_sec'%( per )]
                        vel         = pergrp['vel_iso'+sfx].value
                    except KeyError:
                        if verbose:
                            print 'No data for T = '+str(per)+' sec'
                        continue
                    T               = np.append(T, per)
                    disp_v          = np.append(disp_v, vel[ilat, ilon])
                # get sem from phase for group
                per_phase           = group['disp_ph_ray'].value[0, :]
                disp_un             = group['disp_ph_ray'].value[2, :]
                index               = per_phase<=Tmax
                per_phase           = per_phase[index]
                disp_un             = disp_un[index]
                if not np.allclose(T, per_phase):
                    raise ValueError('Incompatible period for phase and group dispersion data!')
                data                = np.zeros((3, T.size))
                data[0, :]          = T[:]
                data[1, :]          = disp_v[:]
                data[2, :]          = disp_un[:] * semfactor
                group.create_dataset(name='disp_gr_'+wtype, data=data)
        indset.close()
        return
    
    
    def read_eik_azi_aniso(self, inh5fname, runid=0, Tmin=-999, Tmax=999, semfactor=2., psisemfactor=3.5, ampsemfactor=4., wtype='ray'):
        indset          = h5py.File(inh5fname)
        #--------------------------------------------
        # header information from input hdf5 file
        #--------------------------------------------
        dataid          = 'Eikonal_stack_'+str(runid)
        pers            = indset.attrs['period_array']
        grp             = indset[dataid]
        dlon            = indset.attrs['dlon']
        dlat            = indset.attrs['dlat']
        mask            = indset[dataid+'/mask_allT']
        self.attrs.create(name='mask_azi', data = mask)
        if dlon!= self.attrs['dlon_interp'] or dlat != self.attrs['dlat_interp']:
            raise ValueError('Incompatible grid spacing!')
        azi_grp         = self.require_group('azi_grd_pts')
        self._get_lon_lat_arr(is_interp=True)
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if mask[ilat, ilon]:
                    continue
                data_str    = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                group       = azi_grp.require_group( name = data_str )
                disp_v      = np.array([])
                disp_un     = np.array([])
                psi2        = np.array([])
                unpsi2      = np.array([])
                amp         = np.array([])
                unamp       = np.array([])
                T           = np.array([])
                for per in pers:
                    if per < Tmin or per > Tmax:
                        continue
                    try:
                        pergrp      = grp['%g_sec'%( per )]
                        vel         = pergrp['vel_iso'].value
                        vel_sem     = pergrp['vel_sem'].value
                        psiarr      = pergrp['psiarr'].value
                        unpsiarr    = pergrp['unpsi'].value
                        amparr      = pergrp['amparr'].value
                        unamparr    = pergrp['unamp'].value
                    except KeyError:
                        if verbose:
                            print 'No data for T = '+str(per)+' sec'
                        continue
                    mask_per        = pergrp['mask'].value + pergrp['mask_aniso'].value
                    if mask_per[ilat, ilon]:
                        continue
                    T               = np.append(T, per)
                    disp_v          = np.append(disp_v, vel[ilat, ilon])
                    disp_un         = np.append(disp_un, vel_sem[ilat, ilon])
                    psi2            = np.append(psi2, psiarr[ilat, ilon])
                    unpsi2          = np.append(unpsi2, unpsiarr[ilat, ilon])
                    amp             = np.append(amp, amparr[ilat, ilon])
                    unamp           = np.append(unamp, unamparr[ilat, ilon])
                data                = np.zeros((7, T.size))
                data[0, :]          = T[:]
                data[1, :]          = disp_v[:]
                data[2, :]          = disp_un[:] * semfactor
                data[3, :]          = psi2[:]
                unpsi2              *= psisemfactor
                unpsi2[unpsi2>90.]  = 90.
                data[4, :]          = unpsi2[:]
                data[5, :]          = amp[:]
                unamp               *= ampsemfactor
                # # # unamp[unamp>amp]    = amp[unamp>amp]
                data[6, :]          = unamp[:] 
                group.create_dataset(name='disp_azi_'+wtype, data=data)
        indset.close()
        return
    
    def read_eik_azi_aniso_2(self, inh5fname, runid=0, Tmin=-999, Tmax=999, semfactor=2., psisemfactor=3.5, ampsemfactor=4., wtype='ray'):
        indset          = h5py.File(inh5fname)
        #--------------------------------------------
        # header information from input hdf5 file
        #--------------------------------------------
        dataid          = 'Eikonal_stack_'+str(runid)
        pers            = indset.attrs['period_array']
        grp             = indset[dataid]
        dlon            = indset.attrs['dlon']
        dlat            = indset.attrs['dlat']
        mask            = indset[dataid+'/mask_allT']
        self.attrs.create(name='mask_azi', data = mask)
        if dlon!= self.attrs['dlon_interp'] or dlat != self.attrs['dlat_interp']:
            raise ValueError('Incompatible grid spacing!')
        azi_grp         = self.require_group('azi_grd_pts')
        self._get_lon_lat_arr(is_interp=True)
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if mask[ilat, ilon]:
                    continue
                data_str    = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                group       = azi_grp.require_group( name = data_str )
                disp_v      = np.array([])
                disp_un     = np.array([])
                psi2        = np.array([])
                unpsi2      = np.array([])
                amp         = np.array([])
                unamp       = np.array([])
                T           = np.array([])
                for per in pers:
                    if per < Tmin or per > Tmax:
                        continue
                    try:
                        pergrp      = grp['%g_sec'%( per )]
                        vel         = pergrp['vel_iso'].value
                        vel_sem     = pergrp['vel_sem'].value
                        psiarr      = pergrp['psiarr'].value
                        unpsiarr    = pergrp['unpsi'].value
                        amparr      = pergrp['amparr'].value
                        unamparr    = pergrp['unamp'].value
                    except KeyError:
                        if verbose:
                            print 'No data for T = '+str(per)+' sec'
                        continue
                    mask_per        = pergrp['mask'].value + pergrp['mask_aniso'].value
                    if mask_per[ilat, ilon]:
                        continue
                    T               = np.append(T, per)
                    disp_v          = np.append(disp_v, vel[ilat, ilon])
                    disp_un         = np.append(disp_un, vel_sem[ilat, ilon])
                    psi2            = np.append(psi2, psiarr[ilat, ilon])
                    unpsi2          = np.append(unpsi2, unpsiarr[ilat, ilon])
                    amp             = np.append(amp, amparr[ilat, ilon])
                    unamp           = np.append(unamp, unamparr[ilat, ilon])
                data                = np.zeros((7, T.size))
                data[0, :]          = T[:]
                data[1, :]          = disp_v[:]
                data[2, :]          = disp_un[:] * semfactor
                data[3, :]          = psi2[:]
                unpsi2              *= psisemfactor
                unpsi2[unpsi2>90.]  = 90.
                data[4, :]          = unpsi2[:]
                data[5, :]          = amp[:]
                unamp               *= ampsemfactor
                # # # unamp[unamp>amp]    = amp[unamp>amp]
                data[6, :]          = unamp[:]
                try:
                    group.create_dataset(name='disp_azi_'+wtype, data=data)
                except:
                    del group['disp_azi_'+wtype]
                    group.create_dataset(name='disp_azi_'+wtype, data=data)
        indset.close()
        return 
    
    def read_crust_thickness(self, infname='crsthk.xyz', source='crust_1.0', replace_moho=None, infname_refine=''):
        """
        read crust thickness from a txt file (crust 1.0 model)
        """
        inArr       = np.loadtxt(infname)
        lonArr      = inArr[:, 0]
        lonArr      = lonArr.reshape(lonArr.size/360, 360)
        latArr      = inArr[:, 1]
        latArr      = latArr.reshape(latArr.size/360, 360)
        depthArr    = inArr[:, 2]
        depthArr    = depthArr.reshape(depthArr.size/360, 360)
        if os.path.isfile(infname_refine):
            inarr   = np.load(infname_refine)
            mohoarr = inarr['gridded_data_1']
            lonarr  = np.degrees(inarr['gridlons'])
            latarr  = np.degrees(inarr['gridlats'])
            lonarr[lonarr>180.] \
                    -= 360.
        else:
            replace_moho    = None
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
                print 'ERROR!', lon, lat, grd_lon, grd_lat
            depth   = depthArr[ind_lat, ind_lon]
            # replace crust 1.0 with crustal thickness from other datasets
            if replace_moho is not None:
                topo= grp.attrs['topo']
                if topo>0.:
                    index1  = np.where((abs(lonarr-grd_lon) < 1.)*(abs(latarr-grd_lat) < 1.))[0]
                    if index1.size > 0:
                        index2  = (abs(lonarr[index1]-grd_lon)+abs(latarr[index1]-grd_lat)).argmin()
                        index   = index1[index2]
                        if abs(lonarr[index] - grd_lon)>0.5 or abs(latarr[index] - grd_lat)>0.5:
                            print lonarr[index] , grd_lon, latarr[index] , grd_lat
                        new_moho= mohoarr[index]
                        if new_moho > replace_moho and new_moho > depth:
                            depth   = new_moho
                            # if abs(new_moho-depth)/new_moho > 0.5:
                            #     print new_moho, depth, abs(new_moho-depth)/new_moho 
            grp.attrs.create(name='crust_thk', data=depth)
            grp.attrs.create(name='crust_thk_source', data=source)
        return
    
    def read_sediment_thickness(self, infname='sedthk.xyz', source='crust_1.0'):
        """
        read sediment thickness from a txt file (crust 1.0 model)
        """
        inArr       = np.loadtxt(infname)
        lonArr      = inArr[:, 0]
        lonArr      = lonArr.reshape(lonArr.size/360, 360)
        latArr      = inArr[:, 1]
        latArr      = latArr.reshape(latArr.size/360, 360)
        depthArr    = inArr[:, 2]
        depthArr    = depthArr.reshape(depthArr.size/360, 360)
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
                print 'ERROR!', lon, lat, grd_lon, grd_lat
            depth   = depthArr[ind_lat, ind_lon]
            grp.attrs.create(name='sedi_thk', data=depth)
            grp.attrs.create(name='sedi_thk_source', data=source)
        return
    
    def read_CU_model(self, infname='CU_SDT1.0.mod.h5'):
        """
        read reference model from a hdf5 file (CU Global Vs model)
        """
        indset      = h5py.File(infname)
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
                print 'ERROR!', lons[ind_lon], lats[ind_lat] , grd_lon, grd_lat
            data        = indset[str(lons[ind_lon])+'_'+str(lats[ind_lat])].value
            grp.create_dataset(name='reference_vs', data=data)
        indset.close()
        return
    
    def read_etopo(self, infname='../ETOPO2v2g_f4.nc', download=True, delete=True, source='etopo2'):
        """
        read topography data from etopo2
        ============================================================================
        ::: input :::
        infname     - input file name
        download    - download the data or not, if the etopo file does not exist
        delete      - delete the downloaded etopo file or not
        source      - source name (default - etopo2)
        ============================================================================
        """
        from netCDF4 import Dataset
        try:
            etopodbase      = Dataset(infname)
        except IOError:
            if download:
                url         = 'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2g/netCDF/ETOPO2v2g_f4_netCDF.zip'
                os.system('wget '+url)
                os.system('unzip ETOPO2v2g_f4_netCDF.zip')
                if delete:
                    os.remove('ETOPO2v2g_f4_netCDF.zip')
                etopodbase  = Dataset('./ETOPO2v2g_f4.nc')
            else:
                print 'No etopo data!'
                return
        etopo       = etopodbase.variables['z'][:]
        lons        = etopodbase.variables['x'][:]
        lats        = etopodbase.variables['y'][:]
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
            grd_lat = float(split_id[1])
            try:
                ind_lon         = np.where(lons>=grd_lon)[0][0]
            except:
                ind_lon         = lons.size - 1
            try:
                ind_lat         = np.where(lats>=grd_lat)[0][0]
            except:
                ind_lat         = lats.size - 1
            if lons[ind_lon] - grd_lon > (1./60.):
                ind_lon         -= 1
            if lats[ind_lat] - grd_lat > (1./60.):
                ind_lat         -= 1
            if abs(lons[ind_lon] - grd_lon) > 1./60. or abs(lats[ind_lat] - grd_lat) > 1./60.:
                print 'ERROR!', lons[ind_lon], lats[ind_lat] , grd_lon, grd_lat
            z                   = etopo[ind_lat, ind_lon]/1000. # convert to km
            grp.attrs.create(name='topo', data=z)
            grp.attrs.create(name='etopo_source', data=source)
        if delete and os.path.isfile('./ETOPO2v2g_f4.nc'):
            os.remove('./ETOPO2v2g_f4.nc')
        return
    
    def get_raytomo_mask(self, inh5fname, runid):
        """
        get the mask array from ray tomography database
        """
        # mask array
        dataid      = 'reshaped_qc_run_'+str(runid)
        indset      = h5py.File(inh5fname)
        grp         = indset[dataid]
        mask        = grp['mask_inv'].value
        # dlon/dlat
        dataid      = 'qc_run_'+str(runid)
        grp         = indset[dataid]
        dlon        = grp.attrs['dlon']
        dlat        = grp.attrs['dlat']
        self._set_interp(dlon=dlon, dlat=dlat)
        self._get_lon_lat_arr(is_interp=True)
        if mask.shape == self.lonArr.shape:
            try:
                mask_org\
                    = self.attrs['mask_interp']
                mask+= mask_org
            except KeyError:
                self.attrs.create(name = 'mask_interp', data = mask)
        else:
            raise ValueError('Incompatible dlon/dlat with input mask array from ray tomography database')
        self.attrs.create(name = 'is_interp', data=True, dtype=bool)
        return
    
    def get_hybrid_mask(self, inh5fname, runid):
        """
        get the mask array from hybrid database
        """
        # mask array
        dataid      = 'merged_tomo_'+str(runid)
        indset      = h5py.File(inh5fname)
        grp         = indset[dataid]
        mask        = grp['mask_ray'].value
        # dlon/dlat
        dlon        = indset.attrs['dlon']
        dlat        = indset.attrs['dlat']
        self._set_interp(dlon=dlon, dlat=dlat)
        self._get_lon_lat_arr(is_interp=True)
        if mask.shape == self.lonArr.shape:
            try:
                mask_org    = self.attrs['mask_interp']
                mask        += mask_org
                self.attrs.create(name = 'mask_interp', data = mask)
            except KeyError:
                self.attrs.create(name = 'mask_interp', data = mask)
        else:
            raise ValueError('Incompatible dlon/dlat with input mask array from ray tomography database')
        self.attrs.create(name = 'is_interp', data=True, dtype=bool)
        return
    
    def get_basin_mask(self, inh5fname, runid=1):
        """
        get the mask array from hybrid database
        """
        dataid      = 'qc_run_'+str(runid)
        indset      = h5py.File(inh5fname)
        ingroup     = indset['reshaped_'+dataid]
        period      = 10.
        pergrp      = ingroup['%g_sec'%( period )]
        datatype    = 'vel_iso'
        data        = pergrp[datatype].value
        mask        = ingroup['mask1']
        self._get_lon_lat_arr(is_interp=True)
        #
        mask        += data > 2.5
        mask        += self.latArr < 68.
        #
        if mask.shape == self.lonArr.shape:
            try:
                mask_org    = self.attrs['mask_interp']
                mask        += mask_org
                self.attrs.create(name = 'mask_interp', data = mask)
            except KeyError:
                self.attrs.create(name = 'mask_interp', data = mask)
        else:
            raise ValueError('Incompatible dlon/dlat with input mask array from ray tomography database')
        self.attrs.create(name = 'is_interp', data=True, dtype=bool)
        return
    
    def get_basin_mask_inv(self, datadir, isoutput=False):
        """
        get the mask array from hybrid database
        """
        grd_grp     = self['grd_pts']
        grdlst      = grd_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        temp_mask   = self.attrs['mask_inv']
        temp_mask[:]= True
        self._get_lon_lat_arr(is_interp=False)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            # if grd_lon > 180.:
            #     grd_lon     -= 360.
            grd_lat     = float(split_id[1])
            igrd        += 1
            grp         = grd_grp[grd_id]
            ilat        = np.where(grd_lat == self.lats)[0]
            ilon        = np.where(grd_lon == self.lons)[0]
            invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
            datapfx     = datadir+'/'+grd_id
            if not (os.path.isfile(invfname)):
                # print invfname
                # print '--- No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                grp.attrs.create(name='mask', data = True)
                temp_mask[ilat, ilon]   = True
            else:
                # print '--- Reading inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                # print invfname, ilat, ilon
                temp_mask[ilat, ilon]   = False
        if isoutput:
            return temp_mask
        else:
            # set the is_interp as False (default)
            self.attrs.create(name = 'is_interp', data=False, dtype=bool)
            self.attrs.create(name='mask_inv', data = temp_mask)
            # return temp_mask
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
            print 'No data at longitude =',lon,' lattitude =',lat
            return
        try:
            disp_ph = grp['disp_ph_'+wtype]
        except:
            pass
        try:
            disp_gr = grp['disp_gr_'+wtype]
        except:
            pass
        return disp_ph, disp_gr
    
    def plot_disp(self, lon, lat, wtype='ray', derivegr=False, ploterror=False, showfig=True):
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
            print 'No data at longitude =',lon,' lattitude =',lat
            return
        plt.figure()
        ax  = plt.subplot()
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
            CubicSpl= scipy.interpolate.CubicSpline(disp_ph[0, :], disp_ph[1, :])
            Tmin    = disp_ph[0, 0]
            Tmax    = disp_ph[0, -1]
            Tinterp = np.mgrid[Tmin:Tmax:0.1]
            Cinterp = CubicSpl(Tinterp)
            diffC   = Cinterp[2:] - Cinterp[:-2]
            dCdTinterp    \
                    = diffC/0.2
            # dCdT    = np.zeros(disp_ph[0, :].size)
            # for i in range(dCdT.size):
            #     if i == 0:
            #         dCdT[i] = dCdTinterp[0]
            #         continue
            #     if i == dCdT.size-1:
            #         dCdT[i] = dCdTinterp[-1]
            #         continue
            #     ind = np.where(abs(Tinterp[1:-1] - disp_ph[0, i])<0.01)[0]
            #     # print Tinterp[1:-1], disp_ph[0, i]
            #     dCdT[i]\
            #         = dCdTinterp[ind]
            # sU      = 1./disp_ph[1, :] + (disp_ph[0, :]/(disp_ph[1, :])**2)*dCdT
            # derived_U\
            #         = 1./sU
            # plt.plot(disp_ph[0, :], derived_U, 'k--', lw=1, ms=10, label='derived group')
            
            sU      = 1./Cinterp[1:-1] + (Tinterp[1:-1]/(Cinterp[1:-1])**2)*dCdTinterp
            derived_U\
                    = 1./sU
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
    
    def plot_disp_vti(self, lon, lat, plot_group=False, ploterror=False, showfig=True):
        """
        plot dispersion data for inversion of VTI model given location of the grid point
        ==========================================================================================
        ::: input :::
        lon/lat     - location of the grid point
        plot_group  - plot the group velocities or not
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
            print 'No data at longitude =',lon,' lattitude =',lat
            return
        plt.figure()
        ax  = plt.subplot()
        try:
            disp_ph_ray = grp['disp_ph_ray']
            if ploterror:
                plt.errorbar(disp_ph_ray[0, :], disp_ph_ray[1, :], yerr=disp_ph_ray[2, :], color='b', lw=3, label='phase')
            else:
                plt.plot(disp_ph_ray[0, :], disp_ph_ray[1, :], 'bo-', lw=3, ms=10, label='phase')
        except:
            pass
        try:
            disp_ph_lov = grp['disp_ph_lov']
            if ploterror:
                plt.errorbar(disp_ph_lov[0, :], disp_ph_lov[1, :], yerr=disp_ph_lov[2, :], color='k', lw=3, label='phase')
            else:
                plt.plot(disp_ph_lov[0, :], disp_ph_lov[1, :], 'ko-', lw=3, ms=10, label='phase')
        except:
            pass
        if plot_group:
            try:
                disp_gr_ray = grp['disp_gr_ray']
                if ploterror:
                    plt.errorbar(disp_gr_ray[0, :], disp_gr_ray[1, :], yerr=disp_gr_ray[2, :], color='r', lw=3, label='group')
                else:
                    plt.plot(disp_gr_ray[0, :], disp_gr_ray[1, :], 'ro-', lw=3, ms=10, label='group')
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
    #==================================================================
    # function for MC inversion runs
    #==================================================================
    
    def mc_inv_iso(self, use_ref=False, ingrdfname=None, phase=True, group=False, outdir='./workingdir', vp_water=1.5, isconstrt=True,
            verbose=False, step4uwalk=1500, numbrun=15000, subsize=1000, nprocess=None, parallel=True, skipmask=True,\
            Ntotalruns=10, misfit_thresh=1.0, Nmodelthresh=200, outlon=None, outlat=None):
        """
        Bayesian Monte Carlo inversion of surface wave data for an isotropic model
        ==================================================================================================================
        ::: input :::
        use_ref         - use reference input model or not(default = False, use ak135 instead)
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        phase/group     - include phase/group velocity dispersion data or not
        outdir          - output directory
        vp_water        - P wave velocity in water layer (default - 1.5 km/s)
        isconstrt       - require monotonical increase in the crust or not
        step4uwalk      - step interval for uniform random walk in the parameter space
        numbrun         - total number of runs
        subsize         - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess        - number of process
        parallel        - run the inversion in parallel or not
        skipmask        - skip masked grid points or not
        Ntotalruns      - number of times of total runs, the code would run at most numbrun*Ntotalruns iterations
        misfit_thresh   - threshold misfit value to determine "good" models
        Nmodelthresh    - required number of "good" models
        outlon/outlat   - output a vprofile object given longitude and latitude
        ---
        version history:
                    - Added the functionality of adding addtional runs if not enough good models found, Sep 28th, 2018
                    - Added the functionality of using ak135 model as intial model, Sep 28th, 2018
        ==================================================================================================================
        """
        start_time_total    = time.time()
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        grd_grp     = self['grd_pts']
        # get the list for inversion
        if ingrdfname is None:
            grdlst  = grd_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        if phase and group:
            dispdtype   = 'both'
        elif phase and not group:
            dispdtype   = 'ph'
        else:
            dispdtype   = 'gr'
        igrd        = 0
        Ngrd        = len(grdlst)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            igrd    += 1
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = vprofile.vprofile1d()
            if phase:
                try:
                    indisp      = grd_grp[grd_id+'/disp_ph_ray'].value
                    vpr.get_disp(indata=indisp, dtype='ph', wtype='ray')
                except KeyError:
                    print 'WARNING: No phase dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
            if group:
                try:
                    indisp      = grd_grp[grd_id+'/disp_gr_ray'].value
                    vpr.get_disp(indata=indisp, dtype='gr', wtype='ray')
                except KeyError:
                    print 'WARNING: No group dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
            if vpr.data.dispR.npper == 0 and vpr.data.dispR.ngper == 0:
                print 'WARNING: No dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
                continue
            #-----------------------------
            # initial model parameters
            #-----------------------------
            crtthk              = grd_grp[grd_id].attrs['crust_thk']
            sedthk              = grd_grp[grd_id].attrs['sedi_thk']
            topovalue           = grd_grp[grd_id].attrs['topo']
            if use_ref:
                vsdata          = grd_grp[grd_id+'/reference_vs'].value
                vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], crtthk=crtthk, sedthk=sedthk,\
                            topovalue=topovalue, maxdepth=200., vp_water=vp_water)
            else:
                vpr.model.isomod.parameterize_ak135(crtthk=crtthk, sedthk=sedthk, topovalue=topovalue, \
                        maxdepth=200., vp_water=vp_water)
            vpr.getpara()
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:    
                    return vpr
            start_time_grd  = time.time()
            print '=== MC inversion for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=1., Ntotalruns=Ntotalruns, \
                    misfit_thresh=misfit_thresh, Nmodelthresh=Nmodelthresh, isconstrt=isconstrt, pfx=grd_id, verbose=verbose,\
                        step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=1., \
                   isconstrt=isconstrt, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            # end_time_grd    = time.time()
            end_time    = time.time()
            print '--- Elasped time = '+str(end_time - start_time_grd) + ' sec; total elasped time = '+str(end_time - start_time_total)
        return
    
    def mc_inv_vti(self, solver_type=1, use_ref=True, ingrdfname=None, phase=True, group=False, outdir='./workingdir', vp_water=1.5,\
            isconstrt=True, verbose=False, step4uwalk=1500, numbrun=15000, subsize=1000, nprocess=None, parallel=True, skipmask=True,\
            Ntotalruns=10, misfit_thresh=1.0, Nmodelthresh=200, outlon=None, outlat=None, depth_mid_crt=-1., iulcrt=2):
        """
        Bayesian Monte Carlo inversion of VTI model
        ==================================================================================================================
        ::: input :::
        solver_type     - type of solver
                            0   - fast_surf
                            1   - tcps
        use_ref         - use reference input model or not(default = True, use Vsv model from isotropic inversion)
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        phase/group     - include phase/group velocity dispersion data or not
        outdir          - output directory
        vp_water        - P wave velocity in water layer (default - 1.5 km/s)
        isconstrt       - require monotonical increase in the crust or not
        step4uwalk      - step interval for uniform random walk in the parameter space
        numbrun         - total number of runs
        subsize         - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess        - number of process
        parallel        - run the inversion in parallel or not
        skipmask        - skip masked grid points or not
        Ntotalruns      - number of times of total runs, the code would run at most numbrun*Ntotalruns iterations
        misfit_thresh   - threshold misfit value to determine "good" models
        Nmodelthresh    - required number of "good" models
        outlon/outlat   - output a vprofile object given longitude and latitude
        ---
        version history:
                    - first version (2019-03-28)
        ==================================================================================================================
        """
        start_time_total    = time.time()
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        grd_grp     = self['grd_pts']
        # get the list for inversion
        if ingrdfname is None:
            grdlst  = grd_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        if phase and group:
            dispdtype   = 'both'
        elif phase and not group:
            dispdtype   = 'ph'
        else:
            dispdtype   = 'gr'
        igrd        = 0
        Ngrd        = len(grdlst)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            igrd    += 1
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = vprofile.vprofile1d()
            if phase:
                try:
                    disp_ph_lov = grd_grp[grd_id+'/disp_ph_lov'].value
                    vpr.get_disp(indata=disp_ph_lov, dtype='ph', wtype='lov')
                except KeyError:
                    pass
                    # print 'WARNING: No Love wave phase dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
                try:
                    disp_ph_ray = grd_grp[grd_id+'/disp_ph_ray'].value
                    vpr.get_disp(indata=disp_ph_ray, dtype='ph', wtype='ray')
                except KeyError:
                    pass
                    # print 'WARNING: No Rayleigh wave phase dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
            if group:
                try:
                    disp_gr_lov = grd_grp[grd_id+'/disp_gr_lov'].value
                    vpr.get_disp(indata=disp_gr_lov, dtype='gr', wtype='lov')
                except KeyError:
                    print 'WARNING: No Love wave group dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
                try:
                    disp_gr_ray = grd_grp[grd_id+'/disp_gr_ray'].value
                    vpr.get_disp(indata=disp_gr_ray, dtype='gr', wtype='ray')
                except KeyError:
                    print 'WARNING: No Rayleigh wave  group dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
            if vpr.data.dispL.npper == 0 or vpr.data.dispR.npper  == 0:
                print 'WARNING: No dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                continue
            #-----------------------------
            # initial model parameters
            #-----------------------------
            topovalue               = grd_grp[grd_id].attrs['topo']
            if use_ref:
                try:
                    avg_paraval_ray = grd_grp[grd_id+'/avg_paraval_ray'].value
                    std_paraval_ray = grd_grp[grd_id+'/std_paraval_ray'].value
                except KeyError:
                    continue
                vpr.model.vtimod.parameterize_ray(paraval = avg_paraval_ray, topovalue = topovalue, maxdepth=200., vp_water=vp_water)
                vpr.model.vtimod.get_paraind_gamma(std_paraval = std_paraval_ray)
            else:
                crtthk              = grd_grp[grd_id].attrs['crust_thk']
                sedthk              = grd_grp[grd_id].attrs['sedi_thk']
                vpr.model.vtimod.parameterize_ak135(crtthk=crtthk, sedthk=sedthk, topovalue=topovalue, \
                        maxdepth=200., vp_water=vp_water)
                vpr.model.vtimod.get_paraind_gamma()
            # return vpr
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:
                    return vpr
            ###
            # # # if grd_lat < 68.:
            # # #     continue
            # # # try:
            # # #     disp_gr_ray = grd_grp[grd_id+'/disp_gr_ray'].value
            # # #     temp_pers   = disp_gr_ray[0, :]
            # # #     temp_U      = disp_gr_ray[1, temp_pers==10.]
            # # #     if temp_U > 2.5:
            # # #         continue
            # # # except:
            # # #     continue
            
            ###
            # if grd_lon != -163. or grd_lat != 69.:
            #     continue
            # else:
            #     return vpr
            ###
            start_time_grd  = time.time()
            print '=== MC VTI inversion for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            if parallel:
                vpr.mc_joint_inv_vti_mp(outdir=outdir, run_inv=True, solver_type=solver_type, isconstrt=isconstrt, pfx=grd_id,\
                        verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun, savedata=True, subsize=subsize, \
                        nprocess=nprocess, merge=True, Ntotalruns=Ntotalruns, misfit_thresh=misfit_thresh, Nmodelthresh=Nmodelthresh, \
                        depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
            else:
                vpr.mc_joint_inv_vti(outdir=outdir, run_inv=True, solver_type=solver_type, numbcheck=None, misfit_thresh=misfit_thresh, \
                    isconstrt=isconstrt, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun, init_run=True, savedata=True, \
                    depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
            # end_time_grd    = time.time()
            end_time    = time.time()
            print '--- Elasped time = '+str(end_time - start_time_grd) + ' sec; total elasped time = '+str(end_time - start_time_total)
        return
    
    def compute_kernels_hti(self, ingrdfname=None, outdir='./workingdir', vp_water=1.5, misfit_thresh=1.5,\
                outlon=None, outlat=None, outlog='error.log'):
        """
        Bayesian Monte Carlo inversion of VTI model
        ==================================================================================================================
        ::: input :::
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        outdir          - output directory
        vp_water        - P wave velocity in water layer (default - 1.5 km/s)
        outlon/outlat   - output a vprofile object given longitude and latitude
        ---
        version history:
                    - first version (2019-03-28)
        ==================================================================================================================
        """
        start_time_total    = time.time()
        self._get_lon_lat_arr(is_interp=True)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        azi_grp     = self['azi_grd_pts']
        # get the list for inversion
        if ingrdfname is None:
            grdlst  = azi_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        ipercent    = 0
        topoarr     = self['topo_interp'].value
        fid         = open(outlog, 'wb')
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            end_time= time.time()
            if float(igrd)/float(Ngrd)*100. > ipercent: 
                print '--- Computing sensitivity kernels: '+str(ipercent)+' % finished' + \
                    ', elasped time = %g' %(end_time - start_time_total)
                ipercent            += 1
            #-----------------------------
            # get data
            #-----------------------------
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
            vpr                 = vprofile.vprofile1d()
            disp_azi_ray        = azi_grp[grd_id+'/disp_azi_ray'].value
            vpr.get_azi_disp(indata = disp_azi_ray)
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (self.lonArr == grd_lon)*(self.latArr == grd_lat)
            paraval_ref         = np.zeros(13, np.float64)
            paraval_ref[0]      = self['avg_paraval/0_smooth'].value[index]
            paraval_ref[1]      = self['avg_paraval/1_smooth'].value[index]
            paraval_ref[2]      = self['avg_paraval/2_smooth'].value[index]
            paraval_ref[3]      = self['avg_paraval/3_smooth'].value[index]
            paraval_ref[4]      = self['avg_paraval/4_smooth'].value[index]
            paraval_ref[5]      = self['avg_paraval/5_smooth'].value[index]
            paraval_ref[6]      = self['avg_paraval/6_smooth'].value[index]
            paraval_ref[7]      = self['avg_paraval/7_smooth'].value[index]
            paraval_ref[8]      = self['avg_paraval/8_smooth'].value[index]
            paraval_ref[9]      = self['avg_paraval/9_smooth'].value[index]
            paraval_ref[10]     = self['avg_paraval/10_smooth'].value[index]
            paraval_ref[11]     = self['avg_paraval/11_smooth'].value[index]
            paraval_ref[12]     = self['avg_paraval/12_smooth'].value[index]
            topovalue           = topoarr[index]
            vpr.model.vtimod.parameterize_ray(paraval = paraval_ref, topovalue = topovalue, maxdepth=200., vp_water=vp_water)
            vpr.model.vtimod.get_paraind_gamma()
            vpr.update_mod(mtype = 'vti')
            vpr.get_vmodel(mtype = 'vti')
            vpr.get_period()
            cmin                = vpr.data.dispR.pvelo.min()-0.5
            cmax                = vpr.data.dispR.pvelo.max()+0.5
            vpr.compute_reference_vti(wtype='ray', cmin=cmin, cmax=cmax)
            vpr.get_misfit()
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:
                    return vpr
            if vpr.data.dispR.check_disp() or vpr.data.misfit > misfit_thresh:
                print '??? Unstable disp: '+grd_id+', misfit = '+str(vpr.data.misfit)
                cmin                = vpr.data.dispR.pvelo.min()-0.6
                cmax                = vpr.data.dispR.pvelo.max()+0.6
                Ntry                = 0
                while ( (cmin > 0. and cmax < 5.) and vpr.data.dispR.check_disp()):
                    vpr.compute_reference_vti(wtype='ray', cmin=cmin, cmax=cmax)
                    cmin    -= 0.2
                    cmax    += 0.2
                    Ntry    += 1
                    if Ntry > 100:
                        break
                vpr.get_misfit()
                if vpr.data.dispR.check_disp():
                    fid.writelines('%g %g %g %g %g\n' %(grd_lon, grd_lat, vpr.data.misfit, cmin, cmax))
                    continue
                else:
                    print '!!! Stable disp found: '+grd_id+', misfit = '+str(vpr.data.misfit)
            #----------
            # store sensitivity kernels
            #----------
            azi_grp[grd_id].create_dataset(name='dcdA', data=vpr.eigkR.dcdA)
            azi_grp[grd_id].create_dataset(name='dcdC', data=vpr.eigkR.dcdC)
            azi_grp[grd_id].create_dataset(name='dcdF', data=vpr.eigkR.dcdF)
            azi_grp[grd_id].create_dataset(name='dcdL', data=vpr.eigkR.dcdL)
            azi_grp[grd_id].create_dataset(name='iso_misfit', data=vpr.data.misfit)
            azi_grp[grd_id].create_dataset(name='pvel_ref', data=vpr.data.dispR.pvelref)
        end_time    = time.time()
        fid.close()
        print '--- Elasped time = '+str(end_time - start_time_total)
        return
    
    def linear_inv_hti(self, ingrdfname=None, outdir='./workingdir', vp_water=1.5, misfit_thresh=5.0,\
                       verbose=False, outlon=None, outlat=None, depth_mid_crust=15., depth_mid_mantle=-1.):
        """
        Linear inversion of HTI model
        ==================================================================================================================
        ::: input :::
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        outdir          - output directory
        vp_water        - P wave velocity in water layer (default - 1.5 km/s)
        misfit_thresh   - threshold misfit value to determine "good" models
        outlon/outlat   - output a vprofile object given longitude and latitude
        ---
        version history:
                    - first version (2019-03-28)
        ==================================================================================================================
        """
        start_time_total    = time.time()
        self._get_lon_lat_arr(is_interp=True)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        azi_grp     = self['azi_grd_pts']
        # get the list for inversion
        if ingrdfname is None:
            grdlst  = azi_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        topoarr     = self['topo_interp'].value
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            #-----------------------------
            # get data
            #-----------------------------
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
            vpr                 = vprofile.vprofile1d()
            disp_azi_ray        = azi_grp[grd_id+'/disp_azi_ray'].value
            vpr.get_azi_disp(indata = disp_azi_ray)
            ###
            # # # ampsemfactor    = 2.
            # # # vpr.data.dispR.unamp    *= ampsemfactor
            # # # vpr.data.dispR.unamp[vpr.data.dispR.unamp>vpr.data.dispR.amp]   = vpr.data.dispR.amp[vpr.data.dispR.unamp>vpr.data.dispR.amp]
            ###
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (self.lonArr == grd_lon)*(self.latArr == grd_lat)
            paraval_ref         = np.zeros(13, np.float64)
            paraval_ref[0]      = self['avg_paraval/0_smooth'].value[index]
            paraval_ref[1]      = self['avg_paraval/1_smooth'].value[index]
            paraval_ref[2]      = self['avg_paraval/2_smooth'].value[index]
            paraval_ref[3]      = self['avg_paraval/3_smooth'].value[index]
            paraval_ref[4]      = self['avg_paraval/4_smooth'].value[index]
            paraval_ref[5]      = self['avg_paraval/5_smooth'].value[index]
            paraval_ref[6]      = self['avg_paraval/6_smooth'].value[index]
            paraval_ref[7]      = self['avg_paraval/7_smooth'].value[index]
            paraval_ref[8]      = self['avg_paraval/8_smooth'].value[index]
            paraval_ref[9]      = self['avg_paraval/9_smooth'].value[index]
            paraval_ref[10]     = self['avg_paraval/10_smooth'].value[index]
            paraval_ref[11]     = self['avg_paraval/11_smooth'].value[index]
            paraval_ref[12]     = self['avg_paraval/12_smooth'].value[index]
            topovalue           = topoarr[index]
            vpr.model.vtimod.parameterize_ray(paraval = paraval_ref, topovalue = topovalue, maxdepth=200., vp_water=vp_water)
            vpr.model.vtimod.get_paraind_gamma()
            vpr.update_mod(mtype = 'vti')
            vpr.get_vmodel(mtype = 'vti')
            vpr.get_period()
            if not 'dcdL' in azi_grp[grd_id].keys():   
                # cmin                = vpr.data.dispR.pvelo.min()-0.5
                # cmax                = vpr.data.dispR.pvelo.max()+0.5
                cmin                = 1.5
                cmax                = 6.
                vpr.compute_reference_vti(wtype='ray', cmin=cmin, cmax=cmax)
                vpr.get_misfit()
                if vpr.data.dispR.check_disp(thresh=0.4):
                    print 'Unstable disp value: '+grd_id+', misfit = '+str(vpr.data.misfit)
                    continue
                #----------
                # store sensitivity kernels
                #----------
                azi_grp[grd_id].create_dataset(name='dcdA', data=vpr.eigkR.dcdA)
                azi_grp[grd_id].create_dataset(name='dcdC', data=vpr.eigkR.dcdC)
                azi_grp[grd_id].create_dataset(name='dcdF', data=vpr.eigkR.dcdF)
                azi_grp[grd_id].create_dataset(name='dcdL', data=vpr.eigkR.dcdL)
                azi_grp[grd_id].create_dataset(name='iso_misfit', data=vpr.data.misfit)
                iso_misfit      = vpr.data.misfit
                azi_grp[grd_id].create_dataset(name='pvel_ref', data=vpr.data.dispR.pvelref)
            else:
                iso_misfit      = azi_grp[grd_id+'/iso_misfit'].value
            dcdA                = azi_grp[grd_id+'/dcdA'].value
            dcdC                = azi_grp[grd_id+'/dcdC'].value
            dcdF                = azi_grp[grd_id+'/dcdF'].value
            dcdL                = azi_grp[grd_id+'/dcdL'].value
            pvelref             = azi_grp[grd_id+'/pvel_ref'].value
            vpr.get_reference_hti(pvelref=pvelref, dcdA=dcdA, dcdC=dcdC, dcdF=dcdF, dcdL=dcdL)
            if iso_misfit > misfit_thresh:
                print 'Large misfit value: '+grd_id+', misfit = '+str(iso_misfit)
            #------------
            # inversion
            #------------
            vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=depth_mid_crust, depth_mid_mantle=depth_mid_mantle)
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:
                    return vpr
            #-------------------------
            # save inversion results
            #-------------------------
            azi_grp[grd_id].create_dataset(name='azi_misfit', data=vpr.data.misfit)
            azi_grp[grd_id].create_dataset(name='psi2', data=vpr.model.htimod.psi2)
            azi_grp[grd_id].create_dataset(name='unpsi2', data=vpr.model.htimod.unpsi2)
            azi_grp[grd_id].create_dataset(name='amp', data=vpr.model.htimod.amp)
            azi_grp[grd_id].create_dataset(name='unamp', data=vpr.model.htimod.unamp)
            
        return
    # function 
    
    
    def linear_inv_hti_transdimensional(self, ingrdfname=None, outdir='./workingdir', vp_water=1.5, misfit_thresh=5.0,\
            improve_thresh=0.4, verbose=False, outlon=None, outlat=None, depth_mid_crust=15., depth_mid_mantle=80.):
        """
        Linear inversion of HTI model
        ==================================================================================================================
        ::: input :::
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        outdir          - output directory
        vp_water        - P wave velocity in water layer (default - 1.5 km/s)
        misfit_thresh   - threshold misfit value to determine "good" models
        outlon/outlat   - output a vprofile object given longitude and latitude
        ---
        version history:
                    - first version (2019-03-28)
        ==================================================================================================================
        """
        start_time_total    = time.time()
        self._get_lon_lat_arr(is_interp=True)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        azi_grp     = self['azi_grd_pts']
        # get the list for inversion
        if ingrdfname is None:
            grdlst  = azi_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        topoarr     = self['topo_interp'].value
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            #-----------------------------
            # get data
            #-----------------------------
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
            vpr                 = vprofile.vprofile1d()
            disp_azi_ray        = azi_grp[grd_id+'/disp_azi_ray'].value
            vpr.get_azi_disp(indata = disp_azi_ray)
            ###
            # # # ampsemfactor    = 2.
            # # # vpr.data.dispR.unamp    *= ampsemfactor
            # # # vpr.data.dispR.unamp[vpr.data.dispR.unamp>vpr.data.dispR.amp]   = vpr.data.dispR.amp[vpr.data.dispR.unamp>vpr.data.dispR.amp]
            ###
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (self.lonArr == grd_lon)*(self.latArr == grd_lat)
            paraval_ref         = np.zeros(13, np.float64)
            paraval_ref[0]      = self['avg_paraval/0_smooth'].value[index]
            paraval_ref[1]      = self['avg_paraval/1_smooth'].value[index]
            paraval_ref[2]      = self['avg_paraval/2_smooth'].value[index]
            paraval_ref[3]      = self['avg_paraval/3_smooth'].value[index]
            paraval_ref[4]      = self['avg_paraval/4_smooth'].value[index]
            paraval_ref[5]      = self['avg_paraval/5_smooth'].value[index]
            paraval_ref[6]      = self['avg_paraval/6_smooth'].value[index]
            paraval_ref[7]      = self['avg_paraval/7_smooth'].value[index]
            paraval_ref[8]      = self['avg_paraval/8_smooth'].value[index]
            paraval_ref[9]      = self['avg_paraval/9_smooth'].value[index]
            paraval_ref[10]     = self['avg_paraval/10_smooth'].value[index]
            paraval_ref[11]     = self['avg_paraval/11_smooth'].value[index]
            paraval_ref[12]     = self['avg_paraval/12_smooth'].value[index]
            topovalue           = topoarr[index]
            vpr.model.vtimod.parameterize_ray(paraval = paraval_ref, topovalue = topovalue, maxdepth=200., vp_water=vp_water)
            vpr.model.vtimod.get_paraind_gamma()
            vpr.update_mod(mtype = 'vti')
            vpr.get_vmodel(mtype = 'vti')
            vpr.get_period()
            if not 'dcdL' in azi_grp[grd_id].keys():   
                # cmin                = vpr.data.dispR.pvelo.min()-0.5
                # cmax                = vpr.data.dispR.pvelo.max()+0.5
                cmin                = 1.5
                cmax                = 6.
                vpr.compute_reference_vti(wtype='ray', cmin=cmin, cmax=cmax)
                vpr.get_misfit()
                if vpr.data.dispR.check_disp(thresh=0.4):
                    print 'Unstable disp value: '+grd_id+', misfit = '+str(vpr.data.misfit)
                    continue
                #----------
                # store sensitivity kernels
                #----------
                azi_grp[grd_id].create_dataset(name='dcdA', data=vpr.eigkR.dcdA)
                azi_grp[grd_id].create_dataset(name='dcdC', data=vpr.eigkR.dcdC)
                azi_grp[grd_id].create_dataset(name='dcdF', data=vpr.eigkR.dcdF)
                azi_grp[grd_id].create_dataset(name='dcdL', data=vpr.eigkR.dcdL)
                azi_grp[grd_id].create_dataset(name='iso_misfit', data=vpr.data.misfit)
                iso_misfit      = vpr.data.misfit
                azi_grp[grd_id].create_dataset(name='pvel_ref', data=vpr.data.dispR.pvelref)
            else:
                iso_misfit      = azi_grp[grd_id+'/iso_misfit'].value
            dcdA                = azi_grp[grd_id+'/dcdA'].value
            dcdC                = azi_grp[grd_id+'/dcdC'].value
            dcdF                = azi_grp[grd_id+'/dcdF'].value
            dcdL                = azi_grp[grd_id+'/dcdL'].value
            pvelref             = azi_grp[grd_id+'/pvel_ref'].value
            vpr.get_reference_hti(pvelref=pvelref, dcdA=dcdA, dcdC=dcdC, dcdF=dcdF, dcdL=dcdL)
            if iso_misfit > misfit_thresh:
                print 'Large misfit value: '+grd_id+', misfit = '+str(iso_misfit)
            #------------
            # inversion
            #------------
            vpr2                = copy.deepcopy(vpr)
            vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=depth_mid_crust, depth_mid_mantle=-1.)
            vpr2.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=depth_mid_crust, depth_mid_mantle=depth_mid_mantle)
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:
                    return vpr
            #-------------------------
            # save inversion results
            #-------------------------
            if ((vpr.data.misfit - vpr2.data.misfit) >= improve_thresh) and (vpr.data.misfit >= 0.75):
                azi_grp[grd_id].create_dataset(name='azi_misfit', data=vpr2.data.misfit)
                azi_grp[grd_id].create_dataset(name='psi2', data=vpr2.model.htimod.psi2)
                azi_grp[grd_id].create_dataset(name='unpsi2', data=vpr2.model.htimod.unpsi2)
                azi_grp[grd_id].create_dataset(name='amp', data=vpr2.model.htimod.amp)
                azi_grp[grd_id].create_dataset(name='unamp', data=vpr2.model.htimod.unamp)
            else:
                azi_grp[grd_id].create_dataset(name='azi_misfit', data=vpr.data.misfit)
                azi_grp[grd_id].create_dataset(name='psi2', data=vpr.model.htimod.psi2)
                azi_grp[grd_id].create_dataset(name='unpsi2', data=vpr.model.htimod.unpsi2)
                azi_grp[grd_id].create_dataset(name='amp', data=vpr.model.htimod.amp)
                azi_grp[grd_id].create_dataset(name='unamp', data=vpr.model.htimod.unamp)
        return
    
    def linear_inv_hti_adaptive(self, ingrdfname=None, outdir='./workingdir', vp_water=1.5, misfit_thresh=5.0,\
            verbose=False, outlon=None, outlat=None, depth_mid_crust=-1., depth_mid_mantle=-1., \
            labthresh=60., imoho=True, ilab=False, noasth=False, depth2d=np.array([]), Tmin=10., Tmax=80.):
        """
        Linear inversion of HTI model
        ==================================================================================================================
        ::: input :::
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        outdir          - output directory
        vp_water        - P wave velocity in water layer (default - 1.5 km/s)
        misfit_thresh   - threshold misfit value to determine "good" models
        outlon/outlat   - output a vprofile object given longitude and latitude
        ---
        version history:
                    - first version (2019-03-28)
        ==================================================================================================================
        """
        start_time_total    = time.time()
        self._get_lon_lat_arr(is_interp=True)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        azi_grp     = self['azi_grd_pts']
        # get the list for inversion
        if ingrdfname is None:
            grdlst  = azi_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        topoarr     = self['topo_interp'].value
        # number of anisotropic layers
        nlay                = 1
        if depth_mid_crust > 0.:
            nlay            += 1
        if imoho:
            nlay            += 1
        if ilab:
            nlay            += 1
        self.attrs.create(name='imoho', data = imoho)
        self.attrs.create(name='depth_mid_crust', data = depth_mid_crust)
        self.attrs.create(name='ilab', data = ilab)
        self.attrs.create(name='nlay_azi', data = nlay)
        print "number of layers = "+str(nlay)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            #-----------------------------
            # get data
            #-----------------------------
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
            vpr                 = vprofile.vprofile1d()
            disp_azi_ray        = azi_grp[grd_id+'/disp_azi_ray'].value
            periods             = disp_azi_ray[0, :]
            index_per           = (periods>=Tmin)*(periods<=Tmax)
            vpr.get_azi_disp(indata = disp_azi_ray[:, index_per])
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (self.lonArr == grd_lon)*(self.latArr == grd_lat)
            paraval_ref         = np.zeros(13, np.float64)
            paraval_ref[0]      = self['avg_paraval/0_smooth'].value[index]
            paraval_ref[1]      = self['avg_paraval/1_smooth'].value[index]
            paraval_ref[2]      = self['avg_paraval/2_smooth'].value[index]
            paraval_ref[3]      = self['avg_paraval/3_smooth'].value[index]
            paraval_ref[4]      = self['avg_paraval/4_smooth'].value[index]
            paraval_ref[5]      = self['avg_paraval/5_smooth'].value[index]
            paraval_ref[6]      = self['avg_paraval/6_smooth'].value[index]
            paraval_ref[7]      = self['avg_paraval/7_smooth'].value[index]
            paraval_ref[8]      = self['avg_paraval/8_smooth'].value[index]
            paraval_ref[9]      = self['avg_paraval/9_smooth'].value[index]
            paraval_ref[10]     = self['avg_paraval/10_smooth'].value[index]
            paraval_ref[11]     = self['avg_paraval/11_smooth'].value[index]
            paraval_ref[12]     = self['avg_paraval/12_smooth'].value[index]
            topovalue           = topoarr[index]
            vpr.model.vtimod.parameterize_ray(paraval = paraval_ref, topovalue = topovalue, maxdepth=200., vp_water=vp_water)
            vpr.model.vtimod.get_paraind_gamma()
            vpr.update_mod(mtype = 'vti')
            vpr.get_vmodel(mtype = 'vti')
            vpr.get_period()
            rerun_kernel            = False
            if not 'dcdL' in azi_grp[grd_id].keys():   
                cmin                = 1.5
                cmax                = 6.
                vpr.compute_reference_vti(wtype='ray', cmin=cmin, cmax=cmax)
                vpr.get_misfit()
                if vpr.data.dispR.check_disp(thresh=0.4):
                    print 'Unstable disp value: '+grd_id+', misfit = '+str(vpr.data.misfit)
                    continue
                #----------
                # store sensitivity kernels
                #----------
                azi_grp[grd_id].create_dataset(name='dcdA', data=vpr.eigkR.dcdA)
                azi_grp[grd_id].create_dataset(name='dcdC', data=vpr.eigkR.dcdC)
                azi_grp[grd_id].create_dataset(name='dcdF', data=vpr.eigkR.dcdF)
                azi_grp[grd_id].create_dataset(name='dcdL', data=vpr.eigkR.dcdL)
                azi_grp[grd_id].create_dataset(name='iso_misfit', data=vpr.data.misfit)
                iso_misfit      = vpr.data.misfit
                azi_grp[grd_id].create_dataset(name='pvel_ref', data=vpr.data.dispR.pvelref)
                rerun_kernel    = True
            else:
                iso_misfit      = azi_grp[grd_id+'/iso_misfit'].value
            try:
                dcdA                = azi_grp[grd_id+'/dcdA'].value[index_per, :]
                dcdC                = azi_grp[grd_id+'/dcdC'].value[index_per, :]
                dcdF                = azi_grp[grd_id+'/dcdF'].value[index_per, :]
                dcdL                = azi_grp[grd_id+'/dcdL'].value[index_per, :]
                pvelref             = azi_grp[grd_id+'/pvel_ref'].value[index_per]
            except:
                if dcdA.shape[0] == periods[index_per].size:
                    pass
                else:
                    raise ValueError('Check array '+grd_id)
            # print
            if not rerun_kernel:
                vpr.get_reference_hti(pvelref=pvelref, dcdA=dcdA, dcdC=dcdC, dcdF=dcdF, dcdL=dcdL)
            ###
            vpr.eigkR.bottom_padding()
            ###
            if iso_misfit > misfit_thresh:
                print 'Large misfit value: '+grd_id+', misfit = '+str(iso_misfit)
            #------------
            # inversion
            #------------
            if depth_mid_mantle>0.:
                lab_depth       = max(depth_mid_mantle, labthresh)
            else:
                lab_depth       = max(azi_grp[grd_id].attrs['LAB'], labthresh)
            
            # # # slab_depth          = max(azi_grp[grd_id].attrs['slab'], labthresh)
            # # # crtthk              = vpr.model.vtimod.thickness[:-1].sum()
            # # # if slab_depth - crtthk >= 5. and slab_depth <= 195.:
            # # #     lab_depth       = slab_depth
            vpr.lab_depth       = lab_depth
                
            if noasth and ilab:
                raise ValueError('ilab and noasth can not be both True!')
            # two layer
            if nlay == 2:
                if imoho:
                    if noasth:
                        vpr.linear_inv_hti_twolayer(isBcs=True, useref=False, depth=-2., maxdepth=lab_depth, depth2d=depth2d)
                    else:
                        # # # print depth2d
                        # print 'aa'
                        vpr.linear_inv_hti_twolayer(isBcs=True, useref=False, depth=-2., depth2d=depth2d)
                elif ilab:
                    if lab_depth <= 195.:
                        vpr.linear_inv_hti_twolayer(isBcs=True, useref=False, depth=lab_depth)
                    else:
                        vpr.linear_inv_hti_twolayer(isBcs=True, useref=False, depth=195.)
                else:
                    vpr.linear_inv_hti_twolayer(isBcs=True, useref=False, depth=depth_mid_crust)
                    # # # print 'aa'
                    
            elif nlay == 3:
                if imoho and ilab:
                    if lab_depth <= 195.:
                        vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=-1., depth_mid_mantle=lab_depth)
                    else:
                        vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=-1., depth_mid_mantle=-1.)
                else:
                    vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=depth_mid_crust, depth_mid_mantle=-1.)
            else:
                vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=depth_mid_crust, depth_mid_mantle=lab_depth)
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:
                    return vpr
            #-------------------------
            # save inversion results
            #-------------------------            
            azi_grp[grd_id].create_dataset(name='azi_misfit', data=vpr.data.misfit)
            azi_grp[grd_id].create_dataset(name='psi2', data=vpr.model.htimod.psi2)
            azi_grp[grd_id].create_dataset(name='unpsi2', data=vpr.model.htimod.unpsi2)
            azi_grp[grd_id].create_dataset(name='amp', data=vpr.model.htimod.amp)
            azi_grp[grd_id].create_dataset(name='unamp', data=vpr.model.htimod.unamp)
            # newly added
            azi_grp[grd_id].create_dataset(name='amp_misfit', data=vpr.data.dispR.pmisfit_amp)
            azi_grp[grd_id].create_dataset(name='psi_misfit', data=vpr.data.dispR.pmisfit_psi)
            # # # print vpr.data.dispR.pmisfit_psi
            # CGG,
            vpr.data.dispR.get_misfit_hti_select(t1=20., t2=40.)
            azi_grp[grd_id].create_dataset(name='azi_misfit_crt', data=vpr.data.dispR.pmisfit_crt)
            azi_grp[grd_id].create_dataset(name='azi_misfit_man', data=vpr.data.dispR.pmisfit_man)
            # JGR revision
            azi_grp[grd_id].create_dataset(name='azi_misfit_med', data=vpr.data.dispR.pmisfit_med)
        return
    
    
    
    #==================================================================
    # function to read MC inversion results
    #==================================================================
    def read_inv(self, datadir, ingrdfname=None, factor=1., thresh=0.5, stdfactor=2, avgqc=True, \
                 Nmax=None, Nmin=500, wtype='ray'):
        """
        read the inversion results in to data base
        ==================================================================================================================
        ::: input :::
        datadir     - data directory
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        factor      - factor to determine the threshhold value for selectingthe finalized model
        thresh      - threshhold value for selecting the finalized model
                        misfit < min_misfit*factor + thresh
        avgqc       - turn on quality control for average model or not
        Nmax        - required maximum number of accepted model
        Nmin        - required minimum number of accepted model
        ::: NOTE :::
        mask_inv array will be updated according to the existence of inversion results
        ==================================================================================================================
        """
        grd_grp     = self['grd_pts']
        if ingrdfname is None:
            grdlst  = grd_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        temp_mask   = self.attrs['mask_inv']
        self._get_lon_lat_arr(is_interp=False)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat     = float(split_id[1])
            igrd        += 1
            grp         = grd_grp[grd_id]
            ilat        = np.where(grd_lat == self.lats)[0]
            ilon        = np.where(grd_lon == self.lons)[0]
            invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
            datafname   = datadir+'/mc_data.'+grd_id+'.npz'
            if not (os.path.isfile(invfname) and os.path.isfile(datafname)):
                print '--- No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                grp.attrs.create(name='mask', data = True)
                temp_mask[ilat, ilon]\
                        = True
                continue
            print '--- Reading inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            temp_mask[ilat, ilon]\
                        = False
            topovalue   = grp.attrs['topo']
            vpr         = mcpost.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh, stdfactor=stdfactor)
            vpr.read_data(infname = datafname)
            vpr.read_inv_data(infname = invfname, verbose=False, Nmax=Nmax, Nmin=Nmin)
            # --- added Sep 7th, 2018
            vpr.get_paraval()
            vpr.run_avg_fwrd(wdisp=1.)
            # # # return vpr
            # --- added 2019/01/16
            vpr.get_ensemble()
            vpr.get_vs_std()
            if avgqc:
                if vpr.avg_misfit > (vpr.min_misfit*vpr.factor + vpr.thresh)*3.:
                    print '--- Unstable inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                    continue
            #------------------------------------------
            # store inversion results in the database
            #------------------------------------------
            grp.create_dataset(name = 'avg_paraval_'+wtype, data = vpr.avg_paraval)
            grp.create_dataset(name = 'min_paraval_'+wtype, data = vpr.min_paraval)
            grp.create_dataset(name = 'sem_paraval_'+wtype, data = vpr.sem_paraval)
            grp.create_dataset(name = 'std_paraval_'+wtype, data = vpr.std_paraval)
            # --- added 2019/01/16
            grp.create_dataset(name = 'zArr_ensemble_'+wtype, data = vpr.zArr_ensemble)
            grp.create_dataset(name = 'vs_upper_bound_'+wtype, data = vpr.vs_upper_bound)
            grp.create_dataset(name = 'vs_lower_bound_'+wtype, data = vpr.vs_lower_bound)
            grp.create_dataset(name = 'vs_std_'+wtype, data = vpr.vs_std)
            grp.create_dataset(name = 'vs_mean_'+wtype, data = vpr.vs_mean)
            if ('disp_ph_'+wtype) in grp.keys():
                grp.create_dataset(name = 'avg_ph_'+wtype, data = vpr.vprfwrd.data.dispR.pvelp)
                disp_min                = vpr.disppre_ph[vpr.ind_min, :]
                grp.create_dataset(name = 'min_ph_'+wtype, data = disp_min)
            if ('disp_gr_'+wtype) in grp.keys():
                grp.create_dataset(name = 'avg_gr_'+wtype, data = vpr.vprfwrd.data.dispR.gvelp)
                disp_min                = vpr.disppre_gr[vpr.ind_min, :]
                grp.create_dataset(name = 'min_gr_'+wtype, data = disp_min)
            # grp.create_dataset(name = 'min_paraval', data = vpr.sem_paraval)
            grp.attrs.create(name = 'avg_misfit_'+wtype, data = vpr.vprfwrd.data.misfit)
            grp.attrs.create(name = 'min_misfit_'+wtype, data = vpr.min_misfit)
            grp.attrs.create(name = 'mean_misfit_'+wtype, data = vpr.mean_misfit)
        # set the is_interp as False (default)
        self.attrs.create(name = 'is_interp', data=False, dtype=bool)
        self.attrs.create(name='mask_inv', data = temp_mask)
        return
    
    def read_inv_vti(self, datadir, ingrdfname=None, factor=1., thresh=0.5, stdfactor=2, avgqc=True, \
                 Nmax=None, Nmin=500):
        """
        read the inversion results in to data base
        ==================================================================================================================
        ::: input :::
        datadir     - data directory
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        factor      - factor to determine the threshhold value for selectingthe finalized model
        thresh      - threshhold value for selecting the finalized model
                        misfit < min_misfit*factor + thresh
        avgqc       - turn on quality control for average model or not
        Nmax        - required maximum number of accepted model
        Nmin        - required minimum number of accepted model
        ::: NOTE :::
        mask_inv array will be updated according to the existence of inversion results
        ==================================================================================================================
        """
        grd_grp     = self['grd_pts']
        if ingrdfname is None:
            grdlst  = grd_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        temp_mask   = self.attrs['mask_inv']
        self._get_lon_lat_arr(is_interp=False)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat     = float(split_id[1])
            igrd        += 1
            grp         = grd_grp[grd_id]
            ilat        = np.where(grd_lat == self.lats)[0]
            ilon        = np.where(grd_lon == self.lons)[0]
            invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
            datapfx     = datadir+'/'+grd_id
            if not (os.path.isfile(invfname)):
                print '--- No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                grp.attrs.create(name='mask', data = True)
                temp_mask[ilat, ilon]\
                        = True
                continue
            print '--- Reading inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            temp_mask[ilat, ilon]\
                        = False
            topovalue   = grp.attrs['topo']
            vpr         = mcpost_vti.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh, stdfactor=stdfactor)
            vpr.read_data(pfx = datapfx)
            vpr.read_inv_data(infname = invfname, verbose=False, Nmax=Nmax, Nmin=Nmin)
            vpr.get_paraval()
            vpr.get_vmodel()
            vpr.run_avg_fwrd()
            # # # return vpr
            # --- added 2019/01/16
            # # # vpr.get_ensemble()
            # # # vpr.get_vs_std()
            if avgqc:
                if vpr.avg_misfit > (vpr.min_misfit*vpr.factor + vpr.thresh)*3.:
                    print '--- Unstable inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                    temp_mask[ilat, ilon]\
                        = True
                    continue
            #------------------------------------------
            # store inversion results in the database
            #------------------------------------------
            # continue here
            grp.create_dataset(name = 'avg_paraval_vti', data = vpr.avg_paraval)
            grp.create_dataset(name = 'min_paraval_vti', data = vpr.min_paraval)
            grp.create_dataset(name = 'sem_paraval_vti', data = vpr.sem_paraval)
            grp.create_dataset(name = 'std_paraval_vti', data = vpr.std_paraval)
            # --- added 2019/01/16
            # # # grp.create_dataset(name = 'zArr_ensemble_'+wtype, data = vpr.zArr_ensemble)
            # # # grp.create_dataset(name = 'vs_upper_bound_'+wtype, data = vpr.vs_upper_bound)
            # # # grp.create_dataset(name = 'vs_lower_bound_'+wtype, data = vpr.vs_lower_bound)
            # # # grp.create_dataset(name = 'vs_std_'+wtype, data = vpr.vs_std)
            # # # grp.create_dataset(name = 'vs_mean_'+wtype, data = vpr.vs_mean)
            # store Rayleigh wave average and minimum dispersion curves
            grp.create_dataset(name = 'avg_ph_ray_vti', data = vpr.vprfwrd.data.dispR.pvelp)
            disp_min                = vpr.disppre_ray[vpr.ind_min, :]
            grp.create_dataset(name = 'min_ph_ray_vti', data = disp_min)
            # store Love wave average and minimum dispersion curves
            grp.create_dataset(name = 'avg_ph_lov_vti', data = vpr.vprfwrd.data.dispL.pvelp)
            disp_min                = vpr.disppre_lov[vpr.ind_min, :]
            grp.create_dataset(name = 'min_ph_lov_vti', data = disp_min)
            # store misfit
            grp.attrs.create(name = 'avg_misfit_vti', data = vpr.vprfwrd.data.misfit)
            grp.attrs.create(name = 'min_misfit_vti', data = vpr.min_misfit)
            grp.attrs.create(name = 'mean_misfit_vti', data = vpr.mean_misfit)
            grp.attrs.create(name = 'init_misfit_vti', data = vpr.init_misfit)
        # set the is_interp as False (default)
        self.attrs.create(name = 'is_interp', data=False, dtype=bool)
        self.attrs.create(name='mask_inv', data = temp_mask)
        return
    
    def read_inv_vti_2(self, datadir, ingrdfname=None, factor=1., thresh=0.5, stdfactor=2, avgqc=True, \
                 Nmax=None, Nmin=500):
        """
        read the inversion results in to data base, append group speed data
        ==================================================================================================================
        ::: input :::
        datadir     - data directory
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        factor      - factor to determine the threshhold value for selectingthe finalized model
        thresh      - threshhold value for selecting the finalized model
                        misfit < min_misfit*factor + thresh
        avgqc       - turn on quality control for average model or not
        Nmax        - required maximum number of accepted model
        Nmin        - required minimum number of accepted model
        ::: NOTE :::
        mask_inv array will be updated according to the existence of inversion results
        ==================================================================================================================
        """
        grd_grp     = self['grd_pts']
        if ingrdfname is None:
            grdlst  = grd_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        temp_mask   = self.attrs['mask_inv']
        self._get_lon_lat_arr(is_interp=False)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat     = float(split_id[1])
            igrd        += 1
            grp         = grd_grp[grd_id]
            ilat        = np.where(grd_lat == self.lats)[0]
            ilon        = np.where(grd_lon == self.lons)[0]
            invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
            datapfx     = datadir+'/'+grd_id
            if not (os.path.isfile(invfname)):
                print '--- No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                grp.attrs.create(name='mask', data = True)
                temp_mask[ilat, ilon]\
                        = True
                continue
            print '--- Reading inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            temp_mask[ilat, ilon]\
                        = False
            topovalue   = grp.attrs['topo']            
            post_vpr    = mcpost_vti.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh, stdfactor=stdfactor)
            #---------------------
            # initial model
            #---------------------
            avg_paraval_ray = grp['avg_paraval_ray'].value
            init_vpr        = vprofile.vprofile1d()
            init_vpr.model.vtimod.parameterize_ray(paraval = avg_paraval_ray, topovalue = topovalue, maxdepth=200., vp_water=1.5)
            init_vpr.model.vtimod.get_paraind_gamma()
            try:
                disp_gr_ray     = grp['disp_gr_ray'].value
                init_vpr.get_disp(indata=disp_gr_ray, dtype='gr', wtype='ray')
                is_group        = True
            except:
                is_group        = False
            disp_ph_ray     = grp['disp_ph_ray'].value
            init_vpr.get_disp(indata=disp_ph_ray, dtype='ph', wtype='ray')
            disp_ph_lov     = grp['disp_ph_lov'].value
            init_vpr.get_disp(indata=disp_ph_lov, dtype='ph', wtype='lov')
            init_vpr.get_period()
            init_vpr.update_mod(mtype = 'vti')
            init_vpr.get_vmodel(mtype = 'vti')
            init_vpr.compute_disp_vti(solver_type = 0)
            if is_group:
                init_vpr.data.get_misfit_vti_2()
            else:
                init_vpr.data.get_misfit_vti()
            #---------------------
            # post vpr
            #---------------------
            post_vpr         = mcpost_vti.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh, stdfactor=stdfactor)
            post_vpr.read_data(pfx = datapfx)
            post_vpr.read_inv_data(infname = invfname, verbose=False, Nmax=Nmax, Nmin=Nmin)
            post_vpr.get_paraval()
            post_vpr.get_vmodel()
            if is_group:
                post_vpr.data.dispR.gper        = disp_gr_ray[0, :]
                post_vpr.data.dispR.gvelo       = disp_gr_ray[1, :]
                post_vpr.data.dispR.stdgvelo    = disp_gr_ray[2, :]
                post_vpr.data.dispR.ngper       = post_vpr.data.dispR.gper.size
                post_vpr.data.dispR.isgroup     = True
            post_vpr.run_avg_fwrd()
            if is_group:
                post_vpr.vprfwrd.data.get_misfit_vti_2()
            else:
                post_vpr.vprfwrd.data.get_misfit_vti()
            
            #------------------------------------------
            # store inversion results in the database
            #------------------------------------------
            grp.create_dataset(name = 'avg_gr_ray_vti', data = post_vpr.vprfwrd.data.dispR.gvelp)
            # store misfit
            grp.attrs.create(name = 'init_misfit_vti_gr', data = init_vpr.data.misfit)
            grp.attrs.create(name = 'avg_misfit_vti_gr', data = post_vpr.vprfwrd.data.misfit)
            grp.attrs.create(name = 'min_misfit_vti_gr', data = post_vpr.min_misfit)
        # set the is_interp as False (default)
        self.attrs.create(name = 'is_interp', data=False, dtype=bool)
        self.attrs.create(name='mask_inv', data = temp_mask)
        return
    
    def get_vpr(self, datadir, lon, lat, factor=1., thresh=0.5, Nmax=None, Nmin=None):
        """
        Get the postvpr (postprocessing vertical profile)
        """
        if lon < 0.:
            lon     += 360.
        grd_id      = str(lon)+'_'+str(lat)
        grd_grp     = self['grd_pts']
        try:
            grp     = grd_grp[grd_id]
        except:
            print 'No data at longitude =',lon,' lattitude =',lat
            return 
        invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
        datafname   = datadir+'/mc_data.'+grd_id+'.npz'
        topovalue   = grp.attrs['topo']
        vpr         = mcpost.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh)
        vpr.read_inv_data(infname = invfname, verbose=True, Nmax=Nmax, Nmin=Nmin)
        vpr.read_data(infname = datafname)
        vpr.get_paraval()
        vpr.run_avg_fwrd(wdisp=1.)
        if vpr.avg_misfit > (vpr.min_misfit*vpr.factor + vpr.thresh)*2.:
            print '--- Unstable inversion results for grid: lon = '+str(lon)+', lat = '+str(lat)
        if lon > 0.:
            lon     -= 360.
        vpr.code    = str(lon)+'_'+str(lat)
        return vpr
    
    def get_vpr_vti(self, datadir, lon, lat, factor=1., thresh=0.5, stdfactor=2., Nmax=None, Nmin=None):
        """
        Get the postvpr (postprocessing vertical profile)
        """
        if lon < 0.:
            lon     += 360.
        grd_id      = str(lon)+'_'+str(lat)
        grd_grp     = self['grd_pts']
        try:
            grp     = grd_grp[grd_id]
        except:
            print 'No data at longitude =',lon,' lattitude =',lat
            return 
        invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
        datapfx     = datadir+'/'+grd_id
        topovalue   = grp.attrs['topo']
        vpr         = mcpost_vti.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh)
        vpr.read_inv_data(infname = invfname, verbose=True, Nmax=Nmax, Nmin=Nmin)
        vpr.read_data(pfx = datapfx)
        # group speed
        vpr.data.dispR.gper     = grd_grp[grd_id+'/disp_gr_ray'].value[0, :]
        vpr.data.dispR.gvelo    = grd_grp[grd_id+'/disp_gr_ray'].value[1, :]
        vpr.data.dispR.stdgvelo = grd_grp[grd_id+'/disp_gr_ray'].value[2, :]
        vpr.data.dispR.ngper    = vpr.data.dispR.gper.size
        #--------------------------------
        avg_paraval_ray         = grd_grp[grd_id+'/avg_paraval_ray'].value
        std_paraval_ray         = grd_grp[grd_id+'/std_paraval_ray'].value
        vpr.prior_paraval       = avg_paraval_ray
        vpr.std_prior           = std_paraval_ray
        vpr.get_paraval()
        vpr.get_vmodel()
        vpr.run_avg_fwrd()
        if vpr.avg_misfit > (vpr.min_misfit*vpr.factor + vpr.thresh)*2.:
            print '--- Unstable inversion results for grid: lon = '+str(lon)+', lat = '+str(lat)
        if lon > 0.:
            lon     -= 360.
        vpr.code    = str(lon)+'_'+str(lat)
        return vpr
        
    #==================================================================
    # postprocessing, functions maniplulating paraval arrays
    #==================================================================
    
    def get_paraval(self, pindex, dtype='min', itype='ray', ingrdfname=None, isthk=False, depth=5., depthavg=0.):
        """
        get the data for the model parameter
        ==================================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        ==================================================================================================================
        """
        self._get_lon_lat_arr(is_interp=False)
        data        = np.ones(self.latArr.shape)
        grd_grp     = self['grd_pts']
        if ingrdfname is None:
            grdlst  = grd_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd            = 0
        Ngrd            = len(grdlst)
        for grd_id in grdlst:
            split_id    = grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat     = float(split_id[1])
            igrd        += 1
            grp         = grd_grp[grd_id]
            try:
                ind_lon = np.where(grd_lon==self.lons)[0][0]
                ind_lat = np.where(grd_lat==self.lats)[0][0]
            except IndexError:
                # print 'WARNING: grid data N/A at: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                continue
            try:
                paraval                 = grp[dtype+'_paraval_'+itype].value
            except KeyError:
                # print 'WARNING: no data at grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                continue
            # # # if pindex == 'fitratio':
            # # #     Nin             = 0
            # # #     Nall            = 0
            # # #     if 'disp_ph_ray' in grp.keys():
            # # #         obs_ph_ray  = grp['disp_ph_ray'].value[1, :]
            # # #         un_ph_ray   = grp['disp_ph_ray'].value[2, :]
            # # #         pre_ph_ray  = grp['min_ph_ray'].value
            # # #         upper_bound = obs_ph_ray + un_ph_ray
            # # #         lower_bound = obs_ph_ray - un_ph_ray
            # # #         Nin         = np.where( (pre_ph_ray <= upper_bound)*(pre_ph_ray >= lower_bound))[0].size
            # # #         Nall        = obs_ph_ray.size
            # # #     if 'disp_gr_ray' in grp.keys():
            # # #         obs_gr_ray  = grp['disp_gr_ray'].value[1, :]
            # # #         un_gr_ray   = grp['disp_gr_ray'].value[2, :]
            # # #         pre_gr_ray  = grp['min_gr_ray'].value
            # # #         upper_bound = obs_gr_ray + un_gr_ray
            # # #         lower_bound = obs_gr_ray - un_gr_ray
            # # #         Nin         += np.where( (pre_gr_ray <= upper_bound)*(pre_gr_ray >= lower_bound))[0].size
            # # #         Nall        += obs_gr_ray.size
            # # #     data[ind_lat, ind_lon]\
            # # #                     = float(Nin)/float(Nall)
            #  20181203
            if pindex =='moho':
                # get crustal thickness (including sediments)
                if dtype != 'std' and dtype != 'sem':
                    data[ind_lat, ind_lon]  = paraval[-1] + paraval[-2]
                else:
                    data[ind_lat, ind_lon]  = paraval[-1] * 1.5  #  
            elif pindex == 'vs_std_ray':
                unArr                       = grp['vs_std_ray'].value
                zArr                        = grp['zArr_ensemble_ray'].value
                ind_un                      = (zArr <= (depth + depthavg))*(zArr >= (depth - depthavg))
                data[ind_lat, ind_lon]      = unArr[ind_un].mean() 
            else:
                try:
                    float(pindex)
                    data[ind_lat, ind_lon]  = paraval[pindex]
                except ValueError:
                    try:
                        data[ind_lat, ind_lon]  = grp.attrs[pindex]
                    except:
                        pass
            # convert thickness to depth
            if isthk:
                topovalue                   = grp.attrs['topo']
                data[ind_lat, ind_lon]      = data[ind_lat, ind_lon] - topovalue
        return data
    
    def get_filled_paraval(self, pindex, dtype='min', itype='ray', ingrdfname=None, isthk=False, do_interp=False, \
                           workingdir='working_interpolation', depth=5., depthavg=0.):
        """
        get the filled data array for the model parameter
        ==================================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        do_interp   - perform interpolation or not
        workingdir  - working directory for interpolation
        ==================================================================================================================
        """
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        data        = self.get_paraval(pindex=pindex, dtype=dtype, itype=itype, ingrdfname=ingrdfname, isthk=isthk, depth=depth, depthavg=depthavg)
        mask_inv    = self.attrs['mask_inv']
        ind_valid   = np.logical_not(mask_inv)
        data_out    = data.copy()
        g           = Geod(ellps='WGS84')
        vlonArr     = self.lonArr[ind_valid]
        vlatArr     = self.latArr[ind_valid]
        vdata       = data[ind_valid]
        L           = vlonArr.size
        #------------------------------
        # filling the data_out array
        #------------------------------
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if not mask_inv[ilat, ilon]:
                    continue
                clonArr         = np.ones(L, dtype=float)*self.lons[ilon]
                clatArr         = np.ones(L, dtype=float)*self.lats[ilat]
                az, baz, dist   = g.inv(clonArr, clatArr, vlonArr, vlatArr)
                ind_min         = dist.argmin()
                data_out[ilat, ilon] \
                                = vdata[ind_min]
        if do_interp:
            #----------------------------------------------------
            # interpolation for data to dlon_interp/dlat_interp
            #----------------------------------------------------
            dlon                = self.attrs['dlon_interp']
            dlat                = self.attrs['dlat_interp']
            field2d             = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                    minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d.read_array(lonArr = vlonArr, latArr = vlatArr, ZarrIn = vdata)
            outfname            = 'interp_data.lst'
            field2d.interp_surface(workingdir=workingdir, outfname=outfname)
            data_out            = field2d.Zarr
        return data_out
    
    def get_smooth_paraval(self, pindex, sigma=1., smooth_type = 'gauss', dtype='min', itype='ray', \
            workingdir = 'working_gauss_smooth', gsigma=50., ingrdfname=None, isthk=False, do_interp=False,\
            depth=5., depthavg=0.):
        """
        get smooth data array for the model parameter
        ==================================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        sigma       - total number of smooth iterations
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        gsigma      - sigma for Gaussian smoothing (unit - km)
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        ==================================================================================================================
        """
        data            = self.get_filled_paraval(pindex=pindex, dtype=dtype, itype=itype, ingrdfname=ingrdfname, isthk=isthk, do_interp=do_interp, \
                                depth=depth, depthavg=depthavg)
        if smooth_type is 'nearneighbor':
            data_smooth = data.copy()
            #- Smoothing by averaging over neighbouring cells. ----------------------
            for iteration in range(int(sigma)):
                for ilat in range(1, self.Nlat-1):
                    for ilon in range(1, self.Nlon-1):
                        data_smooth[ilat, ilon] = (data[ilat, ilon] + data[ilat+1, ilon] \
                                                   + data[ilat-1, ilon] + data[ilat, ilon+1] + data[ilat, ilon-1])/5.0
        elif smooth_type is 'gauss':
            minlon          = self.attrs['minlon']
            maxlon          = self.attrs['maxlon']
            minlat          = self.attrs['minlat']
            maxlat          = self.attrs['maxlat']
            if do_interp:
                dlon        = self.attrs['dlon_interp']
                dlat        = self.attrs['dlat_interp']
                self._get_lon_lat_arr(is_interp=True)
                # change mask array if interpolation is performed
                mask        = self.attrs['mask_interp']
            else:
                dlon        = self.attrs['dlon']
                dlat        = self.attrs['dlat']
                mask        = self.attrs['mask_inv']
            field           = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                    minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            index           = np.logical_not(mask)
            field.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = data[index])
            outfname        = 'smooth_paraval.lst'
            field.gauss_smoothing(workingdir=workingdir, outfname=outfname, sigma=gsigma)
            data_smooth     = field.Zarr
        return data, data_smooth
    
    def paraval_arrays(self, dtype='min', itype='ray', sigma=1, gsigma = 50., verbose=False, depth=5., depthavg=0.):
        """
        get the paraval arrays and store them in the database
        =============================================================================
        ::: input :::
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        sigma       - total number of smooth iterations
        gsigma      - sigma for Gaussian smoothing (unit - km)
        dlon/dlat   - longitude/latitude interval for interpolation
        -----------------------------------------------------------------------------
        ::: procedures :::
        1.  get_paraval
                    - get the paraval for each grid point in the inversion
        2.  get_filled_paraval
                    - a. fill the grid points that are NOT included in the inversion
                      b. perform interpolation if needed
        3.  get_smooth_paraval
                    - perform spatial smoothing of the paraval in each grid point
        
        =============================================================================
        """
        grp                 = self.require_group( name = dtype+'_paraval' )
        do_interp           = self.attrs['is_interp']
        if do_interp:
            topo            = self['topo_interp'].value
        else:
            topo            = self['topo'].value
        #  20181203
        for pindex in range(13):
            if pindex == 11:
                data, data_smooth   = self.get_smooth_paraval(pindex=pindex, dtype=dtype, itype=itype, \
                        sigma=sigma, gsigma = gsigma, isthk=True, do_interp=do_interp, depth=depth, depthavg=depthavg)
                # convert sediment depth to sediment thickness
                data        += topo
                data_smooth += topo
                sedi        = data.copy()
                sedi_smooth = data_smooth.copy()
            elif pindex == 12:
                data, data_smooth   = self.get_smooth_paraval(pindex='moho', dtype=dtype, itype=itype, \
                        sigma=sigma, gsigma = gsigma, isthk=True, do_interp=do_interp, depth=depth, depthavg=depthavg)
                # convert moho depth to crustal thickness (excluding sediments)
                data        += topo
                data_smooth += topo
                data        -= sedi
                data_smooth -= sedi_smooth
            else:
                data, data_smooth   = self.get_smooth_paraval(pindex=pindex, dtype=dtype, itype=itype, \
                        sigma=sigma, gsigma = gsigma, isthk=False, do_interp=do_interp, depth=depth, depthavg=depthavg)
            grp.create_dataset(name = str(pindex)+'_org', data = data)
            grp.create_dataset(name = str(pindex)+'_smooth', data = data_smooth)
        return 
    
    #==================================================================
    # postprocessing, functions for 3D model
    #==================================================================
    
    def construct_3d(self, dtype='min', is_smooth=False, maxdepth=200., dz=0.1):
        """
        construct 3D vs array
        =================================================================
        ::: input :::
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the smoothed array or not
        maxdepth    - maximum depth (default - 200 km)
        dz          - depth interval (default - 0.1 km)
        =================================================================
        """
        is_interp   = self.attrs['is_interp']
        grp         = self[dtype+'_paraval']
        self._get_lon_lat_arr(is_interp=is_interp)
        if self.latArr.shape != grp['0_org'].value.shape:
            raise ValueError('incompatible paraval data with lonArr/latArr !')
        Nz          = int(maxdepth/dz) + 1
        zArr        = np.arange(Nz)*dz
        vs3d        = np.zeros((self.latArr.shape[0], self.latArr.shape[1], Nz))
        Ntotal      = self.Nlat*self.Nlon
        N0          = int(Ntotal/100.)
        i           = 0
        j           = 0
        mask_interp = self.attrs['mask_interp']
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                i                   += 1
                if np.floor(i/N0) > j:
                    print 'Constructing 3d model:',j,' % finished'
                    j               += 1
                paraval             = np.zeros(13, dtype=np.float64)
                if is_interp:
                    topovalue       = self['topo_interp'].value[ilat, ilon]
                else:
                    grd_id          = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                    topovalue       = self[grd_id].attrs['topo']
                for pindex in range(13):
                    if is_smooth:
                        data        = grp[str(pindex)+'_smooth'].value
                    else:
                        data        = grp[str(pindex)+'_org'].value
                    paraval[pindex] = data[ilat, ilon]
                vel_mod             = vmodel.model1d()
                if mask_interp[ilat, ilon]:
                    continue
                if topovalue < 0.:
                    vel_mod.get_para_model(paraval = paraval, waterdepth=-topovalue, vpwater=1.5, nmod=4, \
                        numbp=np.array([1, 2, 4, 5]), mtype = np.array([5, 4, 2, 2]), vpvs = np.array([0, 2., 1.75, 1.75]), maxdepth=200.)
                else:
                    vel_mod.get_para_model(paraval = paraval)
                zArr_in, VsvArr_in  = vel_mod.get_grid_mod()
                if topovalue > 0.:
                    zArr_in         = zArr_in - topovalue
                # # interpolation
                vs_interp           = np.interp(zArr, xp = zArr_in, fp = VsvArr_in)
                vs3d[ilat, ilon, :] = vs_interp[:]                
        if is_smooth:
            grp.create_dataset(name = 'vs_smooth', data = vs3d)
            grp.create_dataset(name = 'z_smooth', data = zArr)
        else:
            grp.create_dataset(name = 'vs_org', data = vs3d)
            grp.create_dataset(name = 'z_org', data = zArr)
        return
        
    def get_topo_arr(self, infname='../ETOPO2v2g_f4.nc'):
        """
        get the topography array
        """
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        topoarr     = np.zeros(self.lonArr.shape)
        if is_interp:
            from netCDF4 import Dataset
            try:
                etopodbase  = Dataset(infname)
            except IOError:
                if download:
                    url     = 'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2g/netCDF/ETOPO2v2g_f4_netCDF.zip'
                    os.system('wget '+url)
                    os.system('unzip ETOPO2v2g_f4_netCDF.zip')
                    if delete:
                        os.remove('ETOPO2v2g_f4_netCDF.zip')
                    etopodbase  = Dataset('./ETOPO2v2g_f4.nc')
                else:
                    print 'No etopo data!'
                    return
            etopo       = etopodbase.variables['z'][:]
            lons        = etopodbase.variables['x'][:]
            lats        = etopodbase.variables['y'][:]
            for ilat in range(self.Nlat):
                for ilon in range(self.Nlon):
                    grd_lon             = self.lons[ilon]
                    grd_lat             = self.lats[ilat]
                    if grd_lon > 180.:
                        grd_lon         -= 360.
                    try:
                        ind_lon         = np.where(lons>=grd_lon)[0][0]
                    except:
                        ind_lon         = lons.size - 1
                    try:
                        ind_lat         = np.where(lats>=grd_lat)[0][0]
                    except:
                        ind_lat         = lats.size - 1
                    if lons[ind_lon] - grd_lon > (1./60.):
                        ind_lon         -= 1
                    if lats[ind_lat] - grd_lat > (1./60.):
                        ind_lat         -= 1
                    if abs(lons[ind_lon] - grd_lon) > 1./60. or abs(lats[ind_lat] - grd_lat) > 1./60.:
                        print 'ERROR!', lons[ind_lon], lats[ind_lat] , grd_lon, grd_lat
                    z                   = etopo[ind_lat, ind_lon]/1000. # convert to km
                    topoarr[ilat, ilon] = z
            self.create_dataset(name='topo_interp', data = topoarr)
        else:
            for ilat in range(self.Nlat):
                for ilon in range(self.Nlon):
                    grd_id              = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                    topovalue           = self[grd_id].attrs['topo']
                    topoarr[ilat, ilon] = topovalue
            self.create_dataset(name='topo', data = topoarr)
        return
    
    def convert_to_vts(self, outdir, dtype='avg', is_smooth=True, pfx='', verbose=False, unit=True, depthavg=3., dz=1.):
        """ Convert Vs model to vts format for plotting with Paraview, VisIt
        ========================================================================================
        ::: input :::
        outdir      - output directory
        modelname   - modelname ('dvsv', 'dvsh', 'dvp', 'drho')
        pfx         - prefix of output files
        unit        - output unit sphere(radius=1) or not
        ========================================================================================
        """
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
            data_str= dtype + '_smooth'
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
            data_str= dtype + '_org'
        
        if depthavg>0.:
            vs3d    = _get_avg_vs3d(zArr, vs3d, depthavg)
            # tvs3d   = vs3d.copy()
            # Nz      = zArr.size
            # for i in range(Nz):
            #     z       = zArr[i]
            #     print i
            #     if z < depthavg:
            #         tvs3d[:, :, i]  = (vs3d[:, :, zArr <= 2.*depthavg]).mean(axis=2)
            #         continue
            #     index   = (zArr <= z + depthavg) + (zArr >= z - depthavg)
            #     tvs3d[:, :, i]  = (vs3d[:, :, index]).mean(axis=2)
            # vs3d        = tvs3d
        print 'End depth averaging'
        
        if dz != zArr[1] - zArr[0]:
            Nz      = int(zArr[-1]/dz) + 1
            tzArr   = dz*np.arange(Nz)
            tvs3d   = np.zeros((vs3d.shape[0], vs3d.shape[1], Nz))
            for i in range(Nz):
                z               = tzArr[i]
                indz            = zArr == z
                tvs3d[:, :, i]  = vs3d[:, :, indz][:, :, 0]
            vs3d        = tvs3d
            zArr        = tzArr
        print 'End downsampling'

        ###
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        from tvtk.api import tvtk, write_data
        if unit:
            Rref=6471.
        else:
            Rref=1.
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        # convert geographycal coordinate to spherichal coordinate
        theta       = (90.0 - self.lats)*np.pi/180.
        phi         = self.lons*np.pi/180.
        radius      = Rref - zArr
        theta, phi, radius \
                    = np.meshgrid(theta, phi, radius, indexing='ij')
        # convert spherichal coordinate to 3D Cartesian coordinate
        x           = radius * np.sin(theta) * np.cos(phi)/Rref
        y           = radius * np.sin(theta) * np.sin(phi)/Rref
        z           = radius * np.cos(theta)/Rref
        dims        = vs3d.shape
        pts         = np.empty(z.shape + (3,), dtype=float)
        pts[..., 0] = x
        pts[..., 1] = y
        pts[..., 2] = z
        pts         = pts.transpose(2, 1, 0, 3).copy()
        pts.shape   = pts.size / 3, 3
        sgrid       = tvtk.StructuredGrid(dimensions=dims, points=pts)
        sgrid.point_data.scalars \
                    = (vs3d).ravel(order='F')
        sgrid.point_data.scalars.name \
                    = 'Vs'
        outfname    = outdir+'/'+pfx+'Vs_'+data_str+'.vts'
        write_data(sgrid, outfname)
        return
    
    def convert_to_slab_vts(self, outdir='outvts', dtype='avg', is_smooth=True, pfx='slab_', verbose=False, unit=True, depthavg=-1., dz=1.):
        """ Convert Vs model to vts format for plotting with Paraview, VisIt
        ========================================================================================
        ::: input :::
        outdir      - output directory
        modelname   - modelname ('dvsv', 'dvsh', 'dvp', 'drho')
        pfx         - prefix of output files
        unit        - output unit sphere(radius=1) or not
        ========================================================================================
        """
        from sympy.ntheory import primefactors
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
            data_str= dtype + '_smooth'
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
            data_str= dtype + '_org'
        
        if depthavg>0.:
            vs3d    = _get_avg_vs3d(zArr, vs3d, depthavg)
            # tvs3d   = vs3d.copy()
            # Nz      = zArr.size
            # for i in range(Nz):
            #     z       = zArr[i]
            #     print i
            #     if z < depthavg:
            #         tvs3d[:, :, i]  = (vs3d[:, :, zArr <= 2.*depthavg]).mean(axis=2)
            #         continue
            #     index   = (zArr <= z + depthavg) + (zArr >= z - depthavg)
            #     tvs3d[:, :, i]  = (vs3d[:, :, index]).mean(axis=2)
            # vs3d        = tvs3d
        print 'End depth averaging'
        
        if dz != zArr[1] - zArr[0]:
            Nz      = int(zArr[-1]/dz) + 1
            tzArr   = dz*np.arange(Nz)
            tvs3d   = np.zeros((vs3d.shape[0], vs3d.shape[1], Nz))
            for i in range(Nz):
                z               = tzArr[i]
                indz            = zArr == z
                tvs3d[:, :, i]  = vs3d[:, :, indz][:, :, 0]
            vs3d        = tvs3d
            zArr        = tzArr
        print 'End downsampling'

        ###
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        from tvtk.api import tvtk, write_data
        if unit:
            Rref=6471.
        else:
            Rref=1.
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        # convert geographycal coordinate to spherichal coordinate
        theta       = (90.0 - self.lats)*np.pi/180.
        phi         = self.lons*np.pi/180.
        radius      = Rref - zArr
        theta, phi, radius \
                    = np.meshgrid(theta, phi, radius, indexing='ij')
        # convert spherichal coordinate to 3D Cartesian coordinate
        x           = radius * np.sin(theta) * np.cos(phi)/Rref
        y           = radius * np.sin(theta) * np.sin(phi)/Rref
        z           = radius * np.cos(theta)/Rref
        
        index   = vs3d>4.35
        x       = x[index]
        y       = y[index]
        z       = z[index]
        vs3d    = vs3d[index]
        least_prime = primefactors(vs3d.size)[0]
        dims        = (vs3d.size/least_prime, least_prime, 1)
        pts         = np.empty(z.shape + (3,), dtype=float)
        pts[..., 0] = x; pts[..., 1] = y; pts[..., 2] = z
        sgrid = tvtk.StructuredGrid(dimensions=dims, points=pts)
        sgrid.point_data.scalars = (vs3d).ravel(order='F')
        sgrid.point_data.scalars.name = 'Vs'
        outfname    = outdir+'/'+pfx+'Vs_'+data_str+'.vts'
        write_data(sgrid, outfname)
        
        return
    
    def convert_to_vtk(self, outdir, filename='Vsv.vtk', dtype='avg', is_smooth=True, pfx='', verbose=False, unit=True, depthavg=3., dz=1.):
        """ convert ses3d model to vtk format for plotting with Paraview, VisIt, ... .
        """
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
            data_str= dtype + '_smooth'
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
            data_str= dtype + '_org'
        if dz != zArr[1] - zArr[0]:
            Nz      = int(zArr[-1]/dz) + 1
            tzArr   = dz*np.arange(Nz)
            tvs3d   = np.zeros((vs3d.shape[0], vs3d.shape[1], Nz))
            for i in range(Nz):
                z               = tzArr[i]
                indz            = zArr == z
                tvs3d[:, :, i]  = vs3d[:, :, indz][:, :, 0]
            vs3d        = tvs3d
            zArr        = tzArr
        print 'End downsampling'
        if depthavg>0.:
            tvs3d   = vs3d.copy()
            Nz      = zArr.size
            for i in range(Nz):
                z       = zArr[i]
                print i
                if z < depthavg:
                    tvs3d[:, :, i]  = (vs3d[:, :, zArr <= 2.*depthavg]).mean(axis=2)
                    continue
                index   = (zArr <= z + depthavg) + (zArr >= z - depthavg)
                tvs3d[:, :, i]  = (vs3d[:, :, index]).mean(axis=2)
            vs3d        = tvs3d
        print 'End depth averaging'
        ###
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if unit:
            Rref=6471.
        else:
            Rref=1.
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        
        #- open file and write header
        fid = open(outdir+'/'+filename,'w')
        if verbose==True:
            print 'write to file '+directory+filename
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        #- write grid points
        N           = vs3d.size
        nx, ny, nz  = vs3d.shape
        radius      = Rref - zArr
        fid.write('POINTS '+str(N)+' float\n')
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    theta   = (90.0 - self.lats[i])*np.pi/180.
                    phi     = self.lons[j]*np.pi/180.
                    r       = radius[k]/Rref ### !!!
                    x       = r*np.sin(theta)*np.cos(phi)
                    y       = r*np.sin(theta)*np.sin(phi)
                    z       = r*np.cos(theta)
                    fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')
        #- write connectivity
        n_cells     = (nx-1)*(ny-1)*(nz-1)
        fid.write('\n')
        fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')
        for i in range(1, nx):
            for j in range(1, ny):
                for k in range(1, nz):
                    a   = k+(j-1)*nz+(i-1)*ny*nz-1
                    b   = k+(j-1)*nz+(i-1)*ny*nz
                    c   = k+(j)*nz+(i-1)*ny*nz-1
                    d   = k+(j)*nz+(i-1)*ny*nz
                    e   = k+(j-1)*nz+(i)*ny*nz-1
                    f   = k+(j-1)*nz+(i)*ny*nz
                    g   = k+(j)*nz+(i)*ny*nz-1
                    h   = k+(j)*nz+(i)*ny*nz
                    fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
        #- write cell types
        fid.write('\n')
        fid.write('CELL_TYPES '+str(n_cells)+'\n')
        for i in range(nx-1):
            for j in range(ny-1):
                for k in range(nz-1):
                    fid.write('11\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS scalars float\n')
        fid.write('LOOKUP_TABLE mytable\n')

        idx         = np.arange(nx)
        idx[nx-1]   = nx-2
        idy         = np.arange(ny)
        idy[ny-1]   = ny-2
        idz         = np.arange(nz)
        idz[nz-1]   = nz-2
        for i in idx:
            for j in idy:
                for k in idz:
                    fid.write(str(vs3d[i,j,k])+'\n')
        #- clean up
        fid.close()
        return
    
    def construct_LAB(self, outlon=None, outlat=None, vp_water=1.5):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        grdlst      = azi_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        out_grp     = self.require_group('hti_model')
        labarr      = np.zeros(self.lonArr.shape, dtype=np.float64)
        mask        = np.ones(self.lonArr.shape, dtype=bool)
        topoarr     = self['topo_interp'].value
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            ind_lon = np.where(self.lons == grd_lon)[0]
            ind_lat = np.where(self.lats == grd_lat)[0]
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
            azi_grp[grd_id].attrs.create(name='LAB', data=-1.)
            vpr                 = vprofile.vprofile1d()
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (self.lonArr == grd_lon)*(self.latArr == grd_lat)
            paraval_ref         = np.zeros(13, np.float64)
            paraval_ref[0]      = self['avg_paraval/0_smooth'].value[index]
            paraval_ref[1]      = self['avg_paraval/1_smooth'].value[index]
            paraval_ref[2]      = self['avg_paraval/2_smooth'].value[index]
            paraval_ref[3]      = self['avg_paraval/3_smooth'].value[index]
            paraval_ref[4]      = self['avg_paraval/4_smooth'].value[index]
            paraval_ref[5]      = self['avg_paraval/5_smooth'].value[index]
            paraval_ref[6]      = self['avg_paraval/6_smooth'].value[index]
            paraval_ref[7]      = self['avg_paraval/7_smooth'].value[index]
            paraval_ref[8]      = self['avg_paraval/8_smooth'].value[index]
            paraval_ref[9]      = self['avg_paraval/9_smooth'].value[index]
            paraval_ref[10]     = self['avg_paraval/10_smooth'].value[index]
            paraval_ref[11]     = self['avg_paraval/11_smooth'].value[index]
            paraval_ref[12]     = self['avg_paraval/12_smooth'].value[index]
            topovalue           = topoarr[index]
            vpr.model.vtimod.parameterize_ray(paraval = paraval_ref, topovalue = topovalue, maxdepth=200., vp_water=vp_water)
            vpr.model.vtimod.get_paraind_gamma()
            vpr.update_mod(mtype = 'vti')
            vpr.get_vmodel(mtype = 'vti')
            #-----------------------
            # determine LAB
            #-----------------------
            nlay_mantle         = vpr.model.vtimod.nlay[-1]
            vsv_mantle          = vpr.model.vsv[-nlay_mantle:]
            ind                 = scipy.signal.argrelmin(vsv_mantle)[0]
            if ind.size == 0:
                continue
            ind_min             = ind[(ind>1)*(ind<vsv_mantle.size-2)]
            if ind_min.size != 1:
                continue
            if vsv_mantle[ind_min[0]] > 4.4:
                continue
            nlay_above_man      = vpr.model.vtimod.nlay[:-1].sum()
            z                   = vpr.model.h.cumsum()
            lab_depth           = z[nlay_above_man+ind_min[0]]
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:
                    return vpr
            azi_grp[grd_id].attrs.create(name='LAB', data=lab_depth)
            labarr[ind_lat, ind_lon]    = lab_depth
            mask[ind_lat, ind_lon]      = False
        #--------------
        # save data
        #--------------
        # lab
        out_grp.create_dataset(name='labarr', data=labarr)
        # mask
        out_grp.create_dataset(name='mask_lab', data=mask)
        return
    
    def construct_LAB_miller(self):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        grdlst      = azi_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        out_grp     = self.require_group('hti_model')
        labarr      = np.zeros(self.lonArr.shape, dtype=np.float64)
        mask        = np.ones(self.lonArr.shape, dtype=bool)
        # determin avg vs 75 ~ 125km
        vs3d        = self['avg_paraval/vs_smooth'].value
        zarr        = self['avg_paraval/z_smooth'].value
        indz        = (zarr<=125.)*(zarr>=75.)
        vsavgarr    = (vs3d[:, :, indz]).mean(axis=2)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            ind_lon = np.where(self.lons == grd_lon)[0]
            ind_lat = np.where(self.lats == grd_lat)[0]
            
            vsavg   = vsavgarr[ind_lat, ind_lon]
            if vsavg > 4.5:
                lab_depth   = 200.
            else:
                k           = (150.-80.)/(4.5-4.2)
                lab_depth   = k*(vsavg - 4.2) + 80.
            azi_grp[grd_id].attrs.create(name='LAB', data=lab_depth)
            labarr[ind_lat, ind_lon]    = lab_depth
            mask[ind_lat, ind_lon]      = False
        #--------------
        # save data
        #--------------
        # lab
        out_grp.create_dataset(name='labarr', data=labarr)
        # mask
        out_grp.create_dataset(name='mask_lab', data=mask)
        return
    
    
    def read_LAB(self):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        grdlst      = azi_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        out_grp     = self.require_group('hti_model')
        labarr      = np.zeros(self.lonArr.shape, dtype=np.float64)
        mask        = np.ones(self.lonArr.shape, dtype=bool)
        topoarr     = self['topo_interp'].value
        # input LAB file
        inarr       = np.loadtxt('./Torne_etal_ALASKA_DATA.xyz')
        inlon       = inarr[:, 0]
        inlat       = inarr[:, 1]
        inlab       = inarr[:, 4]
        
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            ind_lon = np.where(self.lons == grd_lon)[0]
            ind_lat = np.where(self.lats == grd_lat)[0]
            ind     = np.where( (abs(inlon-grd_lon) < .2)*(abs(inlat-grd_lat) < .2))[0]
            if ind.size == 0:
                azi_grp[grd_id].attrs.create(name='LAB', data=-1.)
                continue
            lab_depth                   = inlab[ind].mean()     
            azi_grp[grd_id].attrs.create(name='LAB', data=lab_depth)
            labarr[ind_lat, ind_lon]    = lab_depth
            mask[ind_lat, ind_lon]      = False
        #--------------
        # save data
        #--------------
        # lab
        out_grp.create_dataset(name='labarr', data=labarr)
        # mask
        out_grp.create_dataset(name='mask_lab', data=mask)
        return
    
    def read_LAB_interp(self, extrapolate = False):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        grdlst      = azi_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        out_grp     = self.require_group('hti_model')
        labarr      = np.zeros(self.lonArr.shape, dtype=np.float64)
        mask        = np.ones(self.lonArr.shape, dtype=bool)
        topoarr     = self['topo_interp'].value
        # input LAB file
        inarr       = np.loadtxt('./Torne_etal_ALASKA_DATA.xyz')
        inlon       = inarr[:, 0]
        inlat       = inarr[:, 1]
        inlab       = inarr[:, 4]
        # interpolation
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        dlon        = self.attrs['dlon_interp']
        dlat        = self.attrs['dlat_interp']
        field       = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                    minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
        field.read_array(lonArr = inlon, latArr = inlat, ZarrIn = inlab)
        outfname    = 'interp_LAB.lst'
        # # # field.gauss_smoothing(workingdir='./temp_smooth', outfname=outfname, width=15.)
        field.interp_surface(workingdir='temp_interp_LAB', outfname=outfname)
        data        = field.Zarr
        if data.shape != labarr.shape:
            raise ValueError('Incompatible shape of arrays!')
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            ind_lon = np.where(self.lons == grd_lon)[0]
            ind_lat = np.where(self.lats == grd_lat)[0]
            if not extrapolate:
                ind     = np.where( (abs(inlon-grd_lon) < .2)*(abs(inlat-grd_lat) < .2))[0]
                if ind.size == 0:
                    azi_grp[grd_id].attrs.create(name='LAB', data=-1.)
                    continue
            lab_depth                   = data[ind_lat, ind_lon]
            azi_grp[grd_id].attrs.create(name='LAB', data=lab_depth)
            labarr[ind_lat, ind_lon]    = lab_depth
            mask[ind_lat, ind_lon]      = False
        #--------------
        # save data
        #--------------
        # lab
        out_grp.create_dataset(name='labarr', data=labarr)
        # mask
        out_grp.create_dataset(name='mask_lab', data=mask)
        return
    
    def construct_slab_edge(self, infname='SlabE325_5_200.dat'):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        grdlst      = azi_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        out_grp     = self.require_group('hti_model')
        slabarr     = np.zeros(self.lonArr.shape, dtype=np.float64)
        mask        = np.ones(self.lonArr.shape, dtype=bool)
        topoarr     = self['topo_interp'].value
        # slab data
        inarr       = np.loadtxt(infname)
        lonarr      = inarr[:, 0]
        latarr      = inarr[:, 1]
        zarr        = -inarr[:, 2]
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            ind_lon = np.where(self.lons == grd_lon)[0]
            ind_lat = np.where(self.lats == grd_lat)[0]
            
            grd_lon -= 360.
            ind     = np.where( (abs(lonarr-grd_lon) < .1)*(abs(latarr-grd_lat) < .1))[0]
            if ind.size == 0:
                azi_grp[grd_id].attrs.create(name='slab', data=-1.)
                continue
            slab_depth                  = zarr[ind].mean()     
            azi_grp[grd_id].attrs.create(name='slab', data=slab_depth)
            slabarr[ind_lat, ind_lon]   = slab_depth
            mask[ind_lat, ind_lon]      = False
        #--------------
        # save data
        #--------------
        # slab
        out_grp.create_dataset(name='slabarr', data=slabarr)
        # mask
        out_grp.create_dataset(name='mask_slab', data=mask)
        return
    
    def construct_dvs(self, outlon=None, outlat=None, vp_water=1.5):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        grdlst      = azi_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        out_grp     = self.require_group('hti_model')
        dvsarr      = np.zeros(self.lonArr.shape, dtype=np.float64)
        mask        = np.ones(self.lonArr.shape, dtype=bool)
        topoarr     = self['topo_interp'].value
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            ind_lon = np.where(self.lons == grd_lon)[0]
            ind_lat = np.where(self.lats == grd_lat)[0]
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
            azi_grp[grd_id].attrs.create(name='dvs', data=-1.)
            vpr                 = vprofile.vprofile1d()
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (self.lonArr == grd_lon)*(self.latArr == grd_lat)
            paraval_ref         = np.zeros(13, np.float64)
            paraval_ref[0]      = self['avg_paraval/0_smooth'].value[index]
            paraval_ref[1]      = self['avg_paraval/1_smooth'].value[index]
            paraval_ref[2]      = self['avg_paraval/2_smooth'].value[index]
            paraval_ref[3]      = self['avg_paraval/3_smooth'].value[index]
            paraval_ref[4]      = self['avg_paraval/4_smooth'].value[index]
            paraval_ref[5]      = self['avg_paraval/5_smooth'].value[index]
            paraval_ref[6]      = self['avg_paraval/6_smooth'].value[index]
            paraval_ref[7]      = self['avg_paraval/7_smooth'].value[index]
            paraval_ref[8]      = self['avg_paraval/8_smooth'].value[index]
            paraval_ref[9]      = self['avg_paraval/9_smooth'].value[index]
            paraval_ref[10]     = self['avg_paraval/10_smooth'].value[index]
            paraval_ref[11]     = self['avg_paraval/11_smooth'].value[index]
            paraval_ref[12]     = self['avg_paraval/12_smooth'].value[index]
            topovalue           = topoarr[index]
            vpr.model.vtimod.parameterize_ray(paraval = paraval_ref, topovalue = topovalue, maxdepth=200., vp_water=vp_water)
            vpr.model.vtimod.get_paraind_gamma()
            vpr.update_mod(mtype = 'vti')
            vpr.get_vmodel(mtype = 'vti')
            #-----------------------
            # determine LAB
            #-----------------------
            nlay_mantle         = vpr.model.vtimod.nlay[-1]
            vsv_mantle          = vpr.model.vsv[-nlay_mantle:]
            z                   = vpr.model.h.cumsum()
            z_mantle            = z[-nlay_mantle:]
            vsv0                = vsv_mantle[z_mantle<80.].mean()
            vsv1                = vsv_mantle[z_mantle>=80.].mean()
            dvs                 = vsv1 - vsv0
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:
                    return vpr
            azi_grp[grd_id].attrs.create(name='dvs', data=dvs)
            dvsarr[ind_lat, ind_lon]    = dvs
            mask[ind_lat, ind_lon]      = False
        #--------------
        # save data
        #--------------
        # lab
        out_grp.create_dataset(name='dvsarr', data=dvsarr)
        # mask
        out_grp.create_dataset(name='mask_dvs', data=mask)
        return
    
    def construct_hti_model(self):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        grdlst      = azi_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        out_grp     = self.require_group('hti_model')
        # six arrays of pis2
        psiarr0     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unpsiarr0   = np.zeros(self.lonArr.shape, dtype=np.float64)
        psiarr1     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unpsiarr1   = np.zeros(self.lonArr.shape, dtype=np.float64)
        psiarr2     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unpsiarr2   = np.zeros(self.lonArr.shape, dtype=np.float64)
        # six arrays of amp
        amparr0     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unamparr0   = np.zeros(self.lonArr.shape, dtype=np.float64)
        amparr1     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unamparr1   = np.zeros(self.lonArr.shape, dtype=np.float64)
        amparr2     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unamparr2   = np.zeros(self.lonArr.shape, dtype=np.float64)
        # one array of misfit
        misfitarr   = np.zeros(self.lonArr.shape, dtype=np.float64)
        ampmisfitarr= np.zeros(self.lonArr.shape, dtype=np.float64)
        psimisfitarr= np.zeros(self.lonArr.shape, dtype=np.float64)
        #
        psimisfitarr_crt= np.zeros(self.lonArr.shape, dtype=np.float64)
        psimisfitarr_man= np.zeros(self.lonArr.shape, dtype=np.float64)
        psimisfitarr_med= np.zeros(self.lonArr.shape, dtype=np.float64)
        # one array of mask
        mask        = np.ones(self.lonArr.shape, dtype=bool)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            ind_lon = np.where(self.lons == grd_lon)[0]
            ind_lat = np.where(self.lats == grd_lat)[0]
            #-----------------------------
            # get data
            #-----------------------------
            try:
                psi2                    = azi_grp[grd_id+'/psi2'].value
                unpsi2                  = azi_grp[grd_id+'/unpsi2'].value
                amp                     = azi_grp[grd_id+'/amp'].value
                unamp                   = azi_grp[grd_id+'/unamp'].value
                misfit                  = azi_grp[grd_id+'/azi_misfit'].value
                ampmisfit               = azi_grp[grd_id+'/amp_misfit'].value
                psimisfit               = azi_grp[grd_id+'/psi_misfit'].value
                psimisfit_crt               = azi_grp[grd_id+'/azi_misfit_crt'].value
                psimisfit_man               = azi_grp[grd_id+'/azi_misfit_man'].value
                psimisfit_med               = azi_grp[grd_id+'/azi_misfit_med'].value
            except:
                temp_grd_id             = grdlst[igrd]
                split_id= grd_id.split('_')
                try:
                    tmp_grd_lon         = float(split_id[0])
                except ValueError:
                    continue
                tmp_grd_lat             = float(split_id[1])
                if not (grd_lon == tmp_grd_lon and abs(tmp_grd_lat - grd_lat)<self.attrs['dlat_interp']/100. ):
                    print temp_grd_id, grd_id
                    raise ValueError('ERROR!')
                psi2                    = azi_grp[temp_grd_id+'/psi2'].value
                unpsi2                  = azi_grp[temp_grd_id+'/unpsi2'].value
                amp                     = azi_grp[temp_grd_id+'/amp'].value
                unamp                   = azi_grp[temp_grd_id+'/unamp'].value
                misfit                  = azi_grp[temp_grd_id+'/azi_misfit'].value
                ampmisfit               = azi_grp[temp_grd_id+'/amp_misfit'].value
                psimisfit               = azi_grp[temp_grd_id+'/psi_misfit'].value
                psimisfit_crt               = azi_grp[temp_grd_id+'/azi_misfit_crt'].value
                psimisfit_man               = azi_grp[temp_grd_id+'/azi_misfit_man'].value
                psimisfit_med               = azi_grp[temp_grd_id+'/azi_misfit_med'].value
            # fast azimuth
            psiarr0[ind_lat, ind_lon]   = psi2[0]
            unpsiarr0[ind_lat, ind_lon] = unpsi2[0]
            psiarr1[ind_lat, ind_lon]   = psi2[1]
            unpsiarr1[ind_lat, ind_lon] = unpsi2[1]
            psiarr2[ind_lat, ind_lon]   = psi2[-1]
            unpsiarr2[ind_lat, ind_lon] = unpsi2[-1]
            # amplitude
            amparr0[ind_lat, ind_lon]   = amp[0]
            unamparr0[ind_lat, ind_lon] = unamp[0]
            amparr1[ind_lat, ind_lon]   = amp[1]
            unamparr1[ind_lat, ind_lon] = unamp[1]
            amparr2[ind_lat, ind_lon]   = amp[-1]
            unamparr2[ind_lat, ind_lon] = unamp[-1]
            # misfit
            misfitarr[ind_lat, ind_lon]     = misfit
            ampmisfitarr[ind_lat, ind_lon]  = ampmisfit
            psimisfitarr[ind_lat, ind_lon]  = psimisfit
            
            psimisfitarr_crt[ind_lat, ind_lon]  = psimisfit_crt
            psimisfitarr_man[ind_lat, ind_lon]  = psimisfit_man
            psimisfitarr_med[ind_lat, ind_lon]  = psimisfit_med
            # mask
            mask[ind_lat, ind_lon]          = False
        #--------------
        # save data
        #--------------
        # fast azimuth
        out_grp.create_dataset(name='psi2_0', data=psiarr0)
        out_grp.create_dataset(name='unpsi2_0', data=unpsiarr0)
        out_grp.create_dataset(name='psi2_1', data=psiarr1)
        out_grp.create_dataset(name='unpsi2_1', data=unpsiarr1)
        out_grp.create_dataset(name='psi2_2', data=psiarr2)
        out_grp.create_dataset(name='unpsi2_2', data=unpsiarr2)
        # amplitude
        out_grp.create_dataset(name='amp_0', data=amparr0)
        out_grp.create_dataset(name='unamp_0', data=unamparr0)
        out_grp.create_dataset(name='amp_1', data=amparr1)
        out_grp.create_dataset(name='unamp_1', data=unamparr1)
        out_grp.create_dataset(name='amp_2', data=amparr2)
        out_grp.create_dataset(name='unamp_2', data=unamparr2)
        # misfit
        out_grp.create_dataset(name='misfit', data=misfitarr)
        out_grp.create_dataset(name='amp_misfit', data=ampmisfitarr)
        out_grp.create_dataset(name='psi_misfit', data=psimisfitarr)
        #
        out_grp.create_dataset(name='psi_misfit_crt', data=psimisfitarr_crt)
        out_grp.create_dataset(name='psi_misfit_man', data=psimisfitarr_man)
        out_grp.create_dataset(name='psi_misfit_med', data=psimisfitarr_med)
        # mask
        out_grp.create_dataset(name='mask', data=mask)
        return
    
    def construct_hti_model_four_lay(self):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        grdlst      = azi_grp.keys()
        igrd        = 0
        Ngrd        = len(grdlst)
        out_grp     = self.require_group('hti_model')
        # six arrays of pis2
        psiarr0     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unpsiarr0   = np.zeros(self.lonArr.shape, dtype=np.float64)
        psiarr1     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unpsiarr1   = np.zeros(self.lonArr.shape, dtype=np.float64)
        psiarr2     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unpsiarr2   = np.zeros(self.lonArr.shape, dtype=np.float64)
        psiarr3     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unpsiarr3   = np.zeros(self.lonArr.shape, dtype=np.float64)
        # six arrays of amp
        amparr0     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unamparr0   = np.zeros(self.lonArr.shape, dtype=np.float64)
        amparr1     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unamparr1   = np.zeros(self.lonArr.shape, dtype=np.float64)
        amparr2     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unamparr2   = np.zeros(self.lonArr.shape, dtype=np.float64)
        amparr3     = np.zeros(self.lonArr.shape, dtype=np.float64)
        unamparr3   = np.zeros(self.lonArr.shape, dtype=np.float64)
        # one array of misfit
        misfitarr   = np.zeros(self.lonArr.shape, dtype=np.float64)
        # one array of mask
        mask        = np.ones(self.lonArr.shape, dtype=bool)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            ind_lon = np.where(self.lons == grd_lon)[0]
            ind_lat = np.where(self.lats == grd_lat)[0]
            #-----------------------------
            # get data
            #-----------------------------
            try:
                psi2                    = azi_grp[grd_id+'/psi2'].value
                unpsi2                  = azi_grp[grd_id+'/unpsi2'].value
                amp                     = azi_grp[grd_id+'/amp'].value
                unamp                   = azi_grp[grd_id+'/unamp'].value
                misfit                  = azi_grp[grd_id+'/azi_misfit'].value
            except:
                temp_grd_id             = grdlst[igrd]
                split_id= grd_id.split('_')
                try:
                    tmp_grd_lon         = float(split_id[0])
                except ValueError:
                    continue
                tmp_grd_lat             = float(split_id[1])
                if not (grd_lon == tmp_grd_lon and abs(tmp_grd_lat - grd_lat)<self.attrs['dlat_interp']/100. ):
                    print temp_grd_id, grd_id
                    raise ValueError('ERROR!')
                psi2                    = azi_grp[temp_grd_id+'/psi2'].value
                unpsi2                  = azi_grp[temp_grd_id+'/unpsi2'].value
                amp                     = azi_grp[temp_grd_id+'/amp'].value
                unamp                   = azi_grp[temp_grd_id+'/unamp'].value
                misfit                  = azi_grp[temp_grd_id+'/azi_misfit'].value
            # fast azimuth
            psiarr0[ind_lat, ind_lon]   = psi2[0]
            unpsiarr0[ind_lat, ind_lon] = unpsi2[0]
            psiarr1[ind_lat, ind_lon]   = psi2[1]
            unpsiarr1[ind_lat, ind_lon] = unpsi2[1]
            psiarr2[ind_lat, ind_lon]   = psi2[2]
            unpsiarr2[ind_lat, ind_lon] = unpsi2[2]
            psiarr3[ind_lat, ind_lon]   = psi2[-1]
            unpsiarr3[ind_lat, ind_lon] = unpsi2[-1]
            # amplitude
            amparr0[ind_lat, ind_lon]   = amp[0]
            unamparr0[ind_lat, ind_lon] = unamp[0]
            amparr1[ind_lat, ind_lon]   = amp[1]
            unamparr1[ind_lat, ind_lon] = unamp[1]
            amparr2[ind_lat, ind_lon]   = amp[2]
            unamparr2[ind_lat, ind_lon] = unamp[2]
            amparr3[ind_lat, ind_lon]   = amp[-1]
            unamparr3[ind_lat, ind_lon] = unamp[-1]
            # misfit
            misfitarr[ind_lat, ind_lon] = misfit
            # mask
            mask[ind_lat, ind_lon]      = False
        #--------------
        # save data
        #--------------
        # fast azimuth
        out_grp.create_dataset(name='psi2_0', data=psiarr0)
        out_grp.create_dataset(name='unpsi2_0', data=unpsiarr0)
        out_grp.create_dataset(name='psi2_1', data=psiarr1)
        out_grp.create_dataset(name='unpsi2_1', data=unpsiarr1)
        out_grp.create_dataset(name='psi2_2', data=psiarr2)
        out_grp.create_dataset(name='unpsi2_2', data=unpsiarr2)
        out_grp.create_dataset(name='psi2_3', data=psiarr3)
        out_grp.create_dataset(name='unpsi2_3', data=unpsiarr3)
        # amplitude
        out_grp.create_dataset(name='amp_0', data=amparr0)
        out_grp.create_dataset(name='unamp_0', data=unamparr0)
        out_grp.create_dataset(name='amp_1', data=amparr1)
        out_grp.create_dataset(name='unamp_1', data=unamparr1)
        out_grp.create_dataset(name='amp_2', data=amparr2)
        out_grp.create_dataset(name='unamp_2', data=unamparr2)
        out_grp.create_dataset(name='amp_3', data=amparr3)
        out_grp.create_dataset(name='unamp_3', data=unamparr3)
        # misfit
        out_grp.create_dataset(name='misfit', data=misfitarr)
        # mask
        out_grp.create_dataset(name='mask', data=mask)
        return
    
    #==================================================================
    # functions for inspection of the database 
    #==================================================================
    def misfit_check(self, mtype='min', misfit_thresh=1.):
        if mtype is 'min':
            pindex      = 'min_misfit'
        elif mtype is 'avg':
            pindex      = 'avg_misfit'
        data, data_smooth\
                        = self.get_smooth_paraval(pindex=pindex, dtype='min',\
                            sigma=1, gsigma = 50., isthk=False, do_interp=False)
        mask            = self.attrs['mask_inv']
        data[mask]      = -1.
        index           = np.where(data > misfit_thresh)
        lons            = self.lonArr[index[0], index[1]]
        lats            = self.latArr[index[0], index[1]]
        return lons, lats
    
    def generate_disp_vs_figs(self, datadir, outdir, dlon=4., dlat=2.,projection='lambert',\
                            Nmax=None, Nmin=None, hillshade=True):
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        lons            = np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon
        lats            = np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat
        lon_plt         = []
        lat_plt         = []
        id_lst          = []
        i               = 0
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for lon in lons:
            for lat in lats:
                vpr         = self.get_vpr(datadir=datadir, lon=lon, lat=lat, factor=1., thresh=0.2, Nmax=Nmax, Nmin=Nmin)
                if vpr is None:
                    continue
                try:
                    gper    = vpr.data.dispR.gper
                except AttributeError:
                    continue
                return vpr
                lon_plt.append(lon)
                lat_plt.append(lat)
                id_lst.append(i)
                # 
                grd_id      = str(lon)+'_'+str(lat)
                fname_disp  = outdir+'/disp_'+str(i)+'_'+grd_id+'.jpg'
                fname_vs    = outdir+'/vs_'+str(i)+'_'+grd_id+'.jpg'
                title       = 'id = '+str(i)+' min_misfit = %2.4f '%vpr.min_misfit
                vpr.expected_misfit()
                title       += 'exp_misfit = %2.4f' %vpr.data.dispR.exp_misfit+','
                title       += ' Nacc = '+str(vpr.ind_thresh.size)+','
                vpr.plot_disp(fname=fname_disp, title=title, savefig=True, showfig=False, disptype='both')
                vpr.plot_profile(fname=fname_vs, title='Vs profile', savefig=True, showfig=False)
                #
                i           += 1
                if i > 2:
                    break
        return  
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        shapefname      = '/home/leon/geological_maps/qfaults'
        m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/home/leon/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/home/leon/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        xc, yc      = m(lon_plt, lat_plt)
        # print lon_plt, lat_plt
        m.plot(xc, yc,'o', ms = 5, mfc='cyan', mec='k')
        for i, txt in enumerate(id_lst):
            plt.annotate(txt, (xc[i], yc[i]), fontsize=15, color='red')
        plt.show()
        return 
        
    
    #==================================================================
    # plotting functions 
    #==================================================================
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.figure()
        plt.figure(figsize=[18, 9.6])
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        
        minlon      = 188 - 360.
        maxlon      = 238. - 360.
        minlat      = 52.
        maxlat      = 72.
        
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                        urcrnrlon=maxlon, lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,1,1,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,1,0])
            # m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
        elif projection=='regional_ortho':
            m1      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                        llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m
            m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                        lat_1=minlat, lat_2=maxlat, lon_0=lon_centre-2., lat_0=lat_centre+2.4)
            # m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[1,1,0,1], fontsize=15)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,1,0], fontsize=15)
            
            # m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[0,0,0,0], fontsize=15)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,0,0], fontsize=15)
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[0,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,0,0], fontsize=15)
            # m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[1,1,0,1], fontsize=15)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,1,1], fontsize=15)
            # # # 
            # # # distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            # # # distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m
            # # # m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
            # # #             lat_1=minlat, lat_2=maxlat, lon_0=lon_centre-2., lat_0=lat_centre+2.4)
            # # # m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            # # # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        
        m.drawcountries(linewidth=1.)
                #################
        coasts = m.drawcoastlines(zorder=100,color= '0.9',linewidth=0.001)
        
        # Exact the paths from coasts
        coasts_paths = coasts.get_paths()
        
        # In order to see which paths you want to retain or discard you'll need to plot them one
        # at a time noting those that you want etc.
        poly_stop = 10
        for ipoly in xrange(len(coasts_paths)):
            print ipoly
            if ipoly > poly_stop:
                break
            r = coasts_paths[ipoly]
            # Convert into lon/lat vertices
            polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in
                                r.iter_segments(simplify=False)]
            px = [polygon_vertices[i][0] for i in xrange(len(polygon_vertices))]
            py = [polygon_vertices[i][1] for i in xrange(len(polygon_vertices))]
            m.plot(px,py,'k-',linewidth=2.)
        ######################
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    def _get_basemap_2(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        # plt.figure()
        plt.figure(figsize=[18, 9.6])
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        
        minlon      = 188 - 360.
        maxlon      = 227. - 360.
        minlat      = 52.
        maxlat      = 72.
        
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                        urcrnrlon=maxlon, lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,1,1,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,1,0])
            # m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
        elif projection=='regional_ortho':
            m1      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                        llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m
            m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                        lat_1=minlat, lat_2=maxlat, lon_0=lon_centre-2., lat_0=lat_centre+2.4)
            # m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[1,1,0,1], fontsize=15)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,1,0], fontsize=15)
            
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[1,1,0,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,1,1], fontsize=15)
        
        m.drawcountries(linewidth=1.)
                #################
        m.drawcoastlines(linewidth=2)
        #coasts = m.drawcoastlines(zorder=100,color= '0.9',linewidth=0.001)
        #
        ## Exact the paths from coasts
        #coasts_paths = coasts.get_paths()
        #
        ## In order to see which paths you want to retain or discard you'll need to plot them one
        ## at a time noting those that you want etc.
        #poly_stop = 10
        #for ipoly in xrange(len(coasts_paths)):
        #    print ipoly
        #    if ipoly > poly_stop:
        #        break
        #    r = coasts_paths[ipoly]
        #    # Convert into lon/lat vertices
        #    polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in
        #                        r.iter_segments(simplify=False)]
        #    px = [polygon_vertices[i][0] for i in xrange(len(polygon_vertices))]
        #    py = [polygon_vertices[i][1] for i in xrange(len(polygon_vertices))]
        #    m.plot(px,py,'k-',linewidth=2.)
        #######################
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
         
    def _get_basemap_3(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        plt.figure(figsize=[18, 9.6])
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        
        minlon      = 195 - 360.
        maxlon      = 232. - 360.
        minlat      = 52.
        maxlat      = 66.
        
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                        urcrnrlon=maxlon, lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,1,1,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,1,0])
            # m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
        elif projection=='regional_ortho':
            m1      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                        llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m
            m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='h', projection='lcc',\
                        lat_1=minlat, lat_2=maxlat, lon_0=lon_centre-2., lat_0=lat_centre+2.4)
            # m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[1,1,0,1], fontsize=15)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,1,0], fontsize=15)
            
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[1,1,0,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        
        m.drawcountries(linewidth=1.)
                #################
        # m.drawcoastlines(linewidth=2)
        coasts = m.drawcoastlines(zorder=100,color= '0.9',linewidth=0.001)
        
        # Exact the paths from coasts
        coasts_paths = coasts.get_paths()
        
        # In order to see which paths you want to retain or discard you'll need to plot them one
        # at a time noting those that you want etc.
        poly_stop = 25
        for ipoly in xrange(len(coasts_paths)):
            print ipoly
            if ipoly > poly_stop:
                break
            r = coasts_paths[ipoly]
            # Convert into lon/lat vertices
            polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in
                                r.iter_segments(simplify=False)]
            px = [polygon_vertices[i][0] for i in xrange(len(polygon_vertices))]
            py = [polygon_vertices[i][1] for i in xrange(len(polygon_vertices))]
            m.plot(px,py,'k-',linewidth=2.)
        ######################
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
         
    def plot_paraval(self, pindex, is_smooth=True, dtype='avg', itype='ray', sigma=1, gsigma = 50., \
            ingrdfname=None, isthk=False, shpfx=None, outfname=None, outimg=None, clabel='', title='', cmap='cv', \
                projection='lambert', lonplt=[], latplt=[], hillshade=False, geopolygons=None,\
                    vmin=None, vmax=None, showfig=True, depth = 5., depthavg = 0.):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        org_mask    - use the original mask in the database or not
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        is_interp       = self.attrs['is_interp']
        if pindex is 'min_misfit' or pindex is 'avg_misfit' or pindex is 'fitratio' or pindex is 'mean_misfit':
            is_interp   = False
        if is_interp:
            mask        = self.attrs['mask_interp']
        else:
            mask        = self.attrs['mask_inv']
        if pindex =='rel_moho_std':
            data, data_smooth\
                        = self.get_smooth_paraval(pindex='moho', dtype='avg', itype=itype, \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            # print 'mean = ', data[np.logical_not(mask)].mean()
            undata, undata_smooth\
                        = self.get_smooth_paraval(pindex='moho', dtype='std', itype=itype, \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            # print 'mean = ', undata[np.logical_not(mask)].mean()
            data = undata/data
            data_smooth = undata_smooth/data_smooth
        else:
            data, data_smooth\
                        = self.get_smooth_paraval(pindex=pindex, dtype=dtype, itype=itype, \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        # return data
        if pindex is 'min_misfit' or pindex is 'avg_misfit':
            indmin      = np.where(data==data.min())
            print indmin
            print 'minimum overall misfit = '+str(data.min())+' longitude/latitude ='\
                        + str(self.lonArr[indmin[0], indmin[1]])+'/'+str(self.latArr[indmin[0], indmin[1]])
            indmax      = np.where(data==data.max())
            print 'maximum overall misfit = '+str(data.max())+' longitude/latitude ='\
                        + str(self.lonArr[indmax[0], indmax[1]])+'/'+str(self.latArr[indmax[0], indmax[1]])
            #
            ind         = (self.latArr == 62.)*(self.lonArr==-149.+360.)
            data[ind]   = 0.645
            #
        
        if is_smooth:
            mdata       = ma.masked_array(data_smooth, mask=mask )
        else:
            mdata       = ma.masked_array(data, mask=mask )
        print 'mean = ', data[np.logical_not(mask)].mean()
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        # m           = self._get_basemap_3(projection=projection, geopolygons=geopolygons)
        x, y            = m(self.lonArr, self.latArr)
        # shapefname      = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        # slb_ctrlst      = read_slab_contour('alu_contours.in', depth=100.)
        # if len(slb_ctrlst) == 0:
        #     print 'No contour at this depth =',depth
        # else:
        #     for slbctr in slb_ctrlst:
        #         xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
        #         m.plot(xslb, yslb,  '--', lw = 5, color='black')
        #         m.plot(xslb, yslb,  '--', lw = 3, color='white')
        ### slab edge
        arr             = np.loadtxt('SlabE325.dat')
        lonslb          = arr[:, 0]
        latslb          = arr[:, 1]
        depthslb        = -arr[:, 2]
        index           = (depthslb > (depth - .05))*(depthslb < (depth + .05))
        lonslb          = lonslb[index]
        latslb          = latslb[index]
        indsort         = lonslb.argsort()
        lonslb          = lonslb[indsort]
        latslb          = latslb[indsort]
        xslb, yslb      = m(lonslb, latslb)
        m.plot(xslb, yslb,  '-', lw = 5, color='black')
        m.plot(xslb, yslb,  '-', lw = 3, color='cyan')
        
        

        # m.plot(xslb, yslb,  '--', lw = 3, color='cyan')
        ### 
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
                    cmap    = cmap.reversed()
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/home/leon/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/home/leon/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=.5)
        else:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        if pindex == 'moho' and dtype == 'avg':
            cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[25., 29., 33., 37., 41., 45.])
        elif pindex == 'moho' and dtype == 'std':
            cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        else:
            cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)

        # # cb.solids.set_rasterized(True)
        # ###
        # xc, yc      = m(np.array([-156]), np.array([67.5]))
        # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
        # xc, yc      = m(np.array([-153]), np.array([61.]))
        # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
        # xc, yc      = m(np.array([-149]), np.array([64.]))
        # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
        # # xc, yc      = m(np.array([-143]), np.array([61.5]))
        # # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
        # 
        # xc, yc      = m(np.array([-152]), np.array([60.]))
        # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
        # xc, yc      = m(np.array([-155]), np.array([69]))
        # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
        ###
        #############################
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
        plt.suptitle(title, fontsize=30)
        
        cb.solids.set_edgecolor("face")
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'go', lw = 3)
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            ind_valid   = np.logical_not(mask)
            outlon      = self.lonArr[ind_valid]
            outlat      = self.latArr[ind_valid]
            outZ        = data[ind_valid]
            OutArr      = np.append(outlon, outlat)
            OutArr      = np.append(OutArr, outZ)
            OutArr      = OutArr.reshape(3, outZ.size)
            OutArr      = OutArr.T
            np.savetxt(outfname, OutArr, '%g')
        if outimg is not None:
            plt.savefig(outimg)
        return
    
    def plot_paraval_merged(self, pindex, is_smooth=True, dtype='avg', itype='ray', sigma=1, gsigma = 50., \
            ingrdfname=None, isthk=False, shpfx=None, outfname=None, outimg=None, clabel='', title='', cmap='cv', \
                projection='lambert', lonplt=[], latplt=[], hillshade=False, geopolygons=None,\
                    vmin=None, vmax=None, showfig=True, depth = 5., depthavg = 0.):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        org_mask    - use the original mask in the database or not
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        is_interp       = False
        if pindex is 'min_misfit' or pindex is 'avg_misfit' or pindex is 'fitratio' or pindex is 'mean_misfit':
            is_interp   = False
        data, data_smooth\
                        = self.get_smooth_paraval(pindex=pindex, dtype=dtype, itype=itype, \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        indset          = invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190501_no_osci_vti_sed_25_crt_10_mantle_10_col.h5')
        
        data2, data_smooth2\
                        = indset.get_smooth_paraval(pindex='min_misfit_vti_gr', dtype=dtype, itype=itype, \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        indset          = invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190501_no_osci_vti_sed_25_crt_10_mantle_0_col.h5')
        if is_interp:
            mask2       = indset.attrs['mask_interp']
        else:
            mask2       = indset.attrs['mask_inv']
        if is_smooth:
            data_smooth[np.logical_not(mask2)]  = data_smooth2[np.logical_not(mask2)]
        else:
            data[np.logical_not(mask2)]         = data2[np.logical_not(mask2)]
            
        if pindex is 'min_misfit' or pindex is 'avg_misfit':
            indmin      = np.where(data==data.min())
            print indmin
            print 'minimum overall misfit = '+str(data.min())+' longitude/latitude ='\
                        + str(self.lonArr[indmin[0], indmin[1]])+'/'+str(self.latArr[indmin[0], indmin[1]])
            indmax      = np.where(data==data.max())
            print 'maximum overall misfit = '+str(data.max())+' longitude/latitude ='\
                        + str(self.lonArr[indmax[0], indmax[1]])+'/'+str(self.latArr[indmax[0], indmax[1]])
            #
            ind         = (self.latArr == 62.)*(self.lonArr==-149.+360.)
            data[ind]   = 0.645
            #
        if is_interp:
            mask        = self.attrs['mask_interp']
        else:
            mask        = self.attrs['mask_inv']
        if is_smooth:
            mdata       = ma.masked_array(data_smooth, mask=mask )
        else:
            mdata       = ma.masked_array(data, mask=mask )
        print 'mean = ', data[np.logical_not(mask)].mean()
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
                
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
                    cmap    = cmap.reversed()
            except:
                pass
        im              = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        if pindex == 'moho' and dtype == 'avg':
            cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[25., 29., 33., 37., 41., 45.])
        elif pindex == 'moho' and dtype == 'std':
            cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        else:
            cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)        
        cb.solids.set_edgecolor("face")
        
        mask2           = indset.attrs['mask_interp']
        self._get_lon_lat_arr(True)
        x, y            = m(self.lonArr, self.latArr)
        m.contour(x, y, mask2, colors='blue', lw=1., levels=[0.])
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'go', lw = 3)
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            ind_valid   = np.logical_not(mask)
            outlon      = self.lonArr[ind_valid]
            outlat      = self.latArr[ind_valid]
            outZ        = data[ind_valid]
            OutArr      = np.append(outlon, outlat)
            OutArr      = np.append(OutArr, outZ)
            OutArr      = OutArr.reshape(3, outZ.size)
            OutArr      = OutArr.T
            np.savetxt(outfname, OutArr, '%g')
        if outimg is not None:
            plt.savefig(outimg)
        return
    
    def plot_rel_jump(self, is_smooth=True, dtype='avg', itype='ray', sigma=1, gsigma = 50., \
            ingrdfname=None, isthk=False, shpfx=None, outfname=None, outimg=None, clabel='', title='', cmap='cv', \
                projection='lambert', lonplt=[], latplt=[], hillshade=False, geopolygons=None,\
                    vmin=None, vmax=None, showfig=True, depth = 5., depthavg = 0.):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        org_mask    - use the original mask in the database or not
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        is_interp       = self.attrs['is_interp']
        vc, vc_smooth\
                        = self.get_smooth_paraval(pindex=5, dtype=dtype, itype=itype, \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        vm, vm_smooth\
                        = self.get_smooth_paraval(pindex=6, dtype=dtype, itype=itype, \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        r, r_smooth\
                        = self.get_smooth_paraval(pindex=-2, dtype='avg', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        if is_interp:
            mask        = self.attrs['mask_interp']
        else:
            mask        = self.attrs['mask_inv']
            
        if is_smooth:
            mdata       = ma.masked_array(2.*(vm - vc)/(vm+vc)*100., mask=mask )
        else:
            mdata       = ma.masked_array(2.*(vm_smooth - vc_smooth)/(vm_smooth+vc_smooth)*100., mask=mask )
            
        # if is_smooth:
        #     mask[(2.*(vm - vc)/(vm+vc)*100. - r)>=0.]   = True
        #     mask[self.latArr>65.]                       = True
        #     mdata       = ma.masked_array(2.*(vm - vc)/(vm+vc)*100. - r, mask=mask )
        # else:
        #     mask[(2.*(vm_smooth - vc_smooth)/(vm_smooth+vc_smooth)*100. - r_smooth)>=0.]   = True
        #     mask[self.latArr>65.]                       = True
        #     mdata       = ma.masked_array(2.*(vm_smooth - vc_smooth)/(vm_smooth+vc_smooth)*100. - r_smooth, mask=mask )
        print 'min = ', mdata.min()
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
                    cmap    = cmap.reversed()
            except:
                pass
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)
        cb.solids.set_edgecolor("face")
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'go', lw = 3)
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            ind_valid   = np.logical_not(mask)
            outlon      = self.lonArr[ind_valid]
            outlat      = self.latArr[ind_valid]
            outZ        = data[ind_valid]
            OutArr      = np.append(outlon, outlat)
            OutArr      = np.append(OutArr, outZ)
            OutArr      = OutArr.reshape(3, outZ.size)
            OutArr      = OutArr.T
            np.savetxt(outfname, OutArr, '%g')
        if outimg is not None:
            plt.savefig(outimg)
        
        if is_smooth:
            data       = 2.*(vm - vc)/(vm+vc)*100.
        else:
            data       = 2.*(vm_smooth - vc_smooth)/(vm_smooth+vc_smooth)*100.
        data            = data[np.logical_not(mask)]
        from statsmodels import robust
        mad     = robust.mad(data)
        outmean = data.mean()
        outstd  = data.std()
        import matplotlib
        def to_percent(y, position):
            # Ignore the passed in position. This has the effect of scaling the default
            # tick locations.
            s = '%.0f' %( 100.*y)
            # The percent symbol needs escaping in latex
            if matplotlib.rcParams['text.usetex'] is True:
                return s + r'$\%$'
            else:
                return s + '%'
        ax      = plt.subplot()
        dbin    = 0.5
        bins    = np.arange(min(data), max(data) + dbin, dbin)
        weights = np.ones_like(data)/float(data.size)
        plt.hist(data, bins=bins, weights = weights)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.title('mean = %g , std = %g , mad = %g ' %(outmean, outstd, mad), fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([vmin, vmax])
        # data2
        if showfig:
            plt.show()
        return
    
    def plot_aniso(self, icrtmtl=1, unthresh = 1., is_smooth=True, sigma=1, gsigma = 50., \
            ingrdfname=None, isthk=False, shpfx=None, outfname=None, title='', cmap='cv', \
                projection='lambert', lonplt=[], latplt=[], hillshade=False, geopolygons=None,\
                    vmin=None, vmax=None, showfig=True, depth = 5., depthavg = 0.):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        org_mask    - use the original mask in the database or not
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        is_interp       = self.attrs['is_interp']
        if icrtmtl == 1:
            data, data_smooth\
                        = self.get_smooth_paraval(pindex=-2, dtype='avg', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            un, un_smooth\
                        = self.get_smooth_paraval(pindex=-2, dtype='std', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        else:
            data, data_smooth\
                        = self.get_smooth_paraval(pindex=-1, dtype='avg', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            un, un_smooth\
                        = self.get_smooth_paraval(pindex=-1, dtype='std', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            
            ###
            dset = invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190501_no_osci_vti_sed_25_crt_10_mantle_10_col.h5')
            data2, data_smooth2\
                        = dset.get_smooth_paraval(pindex=-1, dtype='avg', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            un2, un_smooth2\
                        = dset.get_smooth_paraval(pindex=-1, dtype='std', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            mask2       = dset.attrs['mask_inv']
            data_smooth[np.logical_not(mask2)]  = data_smooth2[np.logical_not(mask2)]
            un[np.logical_not(mask2)]           = un2[np.logical_not(mask2)]
            ###
        if is_interp:
            mask        = self.attrs['mask_interp']
        else:
            mask        = self.attrs['mask_inv']
        if is_smooth:
            mdata       = ma.masked_array(data_smooth, mask=mask )
        else:
            mdata       = ma.masked_array(data, mask=mask )
        print 'mean = ', un[np.logical_not(mask)].mean()
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        # # # slb_ctrlst      = read_slab_contour('alu_contours.in', depth=100.)
        # # # if len(slb_ctrlst) == 0:
        # # #     print 'No contour at this depth =',depth
        # # # else:
        # # #     for slbctr in slb_ctrlst:
        # # #         xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
        # # #         m.plot(xslb, yslb,  '--', lw = 5, color='black')
        # # #         m.plot(xslb, yslb,  '--', lw = 3, color='white')
                
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
                    cmap    = cmap.reversed()
            except:
                pass
        # # # return data_smooth, un_smooth, unthresh
        # ind         = (abs(data_smooth) > un)
        # ind[(un < unthresh)] = True
        
        ind         = un < unthresh
        # ind[(un < unthresh)] = True
        ind[mask]   = False
        indno       = np.logical_not(ind)
        indno[mask] = False
        
        sbmask      = self.get_basin_mask_inv('/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190501_150000_sed_25_crust_0_mantle_10_vti_col',\
                                    isoutput=True)
        ind[np.logical_not(sbmask)]     = False
        indno[np.logical_not(sbmask)]   = True
        
        data2       = data_smooth[indno]
        x2          = x[indno]
        y2          = y[indno]
        im          = plt.scatter(x2, y2, s=200,  c='grey', edgecolors='k', alpha=0.8, marker='s')
        
        
        data1       = data_smooth[ind]
        x1          = x[ind]
        y1          = y[ind]
        im          = plt.scatter(x1, y1, s=200,  c=data1, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='k', alpha=0.8)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')#, ticks=[-10., -5., 0., 5., 10.])
        #
        if icrtmtl == 1:
            cb.set_label('Crustal anisotropy(%)', fontsize=60, rotation=0)
        else:
            cb.set_label('Mantle anisotropy(%)', fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        plt.suptitle(title, fontsize=30)
        
        print data1.max(), data1.mean()
        ###
        # # # depth = 100.
        # # # slb_ctrlst      = read_slab_contour('alu_contours.in', depth=depth)
        # # # # slb_ctrlst      = read_slab_contour('/home/leon/Slab2Distribute_Mar2018/Slab2_CONTOURS/alu_slab2_dep_02.23.18_contours.in', depth=depth)
        # # # if len(slb_ctrlst) == 0:
        # # #     print 'No contour at this depth =',depth
        # # # else:
        # # #     for slbctr in slb_ctrlst:
        # # #         xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
        # # #         # m.plot(xslb, yslb,  '', lw = 5, color='black')
        # # #         factor      = 20
        # # #         # N           = xslb.size
        # # #         # xslb        = xslb[0:N:factor]
        # # #         # yslb        = yslb[0:N:factor]
        # # #         m.plot(xslb, yslb,  '--', lw = 5, color='red', ms=8, markeredgecolor='k')
        # # #                                              
        # # # #############################
        # # # yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        # # # yatlons             = yakutat_slb_dat[:, 0]
        # # # yatlats             = yakutat_slb_dat[:, 1]
        # # # xyat, yyat          = m(yatlons, yatlats)
        # # # m.plot(xyat, yyat, lw = 5, color='black')
        # # # m.plot(xyat, yyat, lw = 3, color='white')
        # # # #############################
        # # # import shapefile
        # # # shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        # # # shplst      = shapefile.Reader(shapefname)
        # # # for rec in shplst.records():
        # # #     lon_vol = rec[4]
        # # #     lat_vol = rec[3]
        # # #     xvol, yvol            = m(lon_vol, lat_vol)
        # # #     m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
        ####
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        #
        lon     = self.lonArr[ind]
        lat     = self.latArr[ind]
        N       = lon.size
        areas   = np.zeros(N)
        dlon        = self.attrs['dlon']
        dlat        = self.attrs['dlat']
        data        = data_smooth[ind]
        for i in range(N):
            distEW, az, baz     = obspy.geodetics.gps2dist_azimuth(lat[i], lon[i]-dlon, lat[i], lon[i]+dlon)
            distNS, az, baz     = obspy.geodetics.gps2dist_azimuth(lat[i]-dlat, lon[i], lat[i]+dlat, lon[i])
            areas[i]   = distEW*distNS/1000.**2
        ### 
        from statsmodels import robust
        mad     = robust.mad(data)
        outmean = data.mean()
        outstd  = data.std()
        import matplotlib
        def to_percent(y, position):
            # Ignore the passed in position. This has the effect of scaling the default
            # tick locations.
            s = '%.0f' %( 100.*y)
            # The percent symbol needs escaping in latex
            if matplotlib.rcParams['text.usetex'] is True:
                return s + r'$\%$'
            else:
                return s + '%'
        ax      = plt.subplot()
        dbin    = 0.1
        bins    = np.arange(min(data), max(data) + dbin, dbin)
        weights = np.ones_like(data)/float(data.size)
        # # # data[data>3.] = 3.
        plt.hist(data, bins=bins, weights = weights)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=60)
        if icrtmtl == 1:
            plt.xlabel('Crustal anisotropy(%)', fontsize=60, rotation=0)
        else:
            plt.xlabel('Mantle anisotropy(%)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g , mad = %g ' %(outmean, outstd, np.median(data)), fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([vmin, vmax])
        # data2
        if showfig:
            plt.show()
        return
    
    def plot_aniso_sb(self, unthresh = 1., is_smooth=True, sigma=1, gsigma = 50., \
            ingrdfname=None, isthk=False, shpfx=None, outfname=None, title='', cmap='cv', \
                projection='lambert', lonplt=[], latplt=[], hillshade=False, geopolygons=None,\
                    vmin=None, vmax=None, showfig=True, depth = 5., depthavg = 0.):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        org_mask    - use the original mask in the database or not
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        is_interp       = False
        data, data_smooth\
                    = self.get_smooth_paraval(pindex=-3, dtype='avg', itype='vti', \
                        sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        un, un_smooth\
                    = self.get_smooth_paraval(pindex=-3, dtype='std', itype='vti', \
                        sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)

        mask        = self.attrs['mask_inv']
        if is_smooth:
            mdata       = ma.masked_array(data_smooth, mask=mask )
        else:
            mdata       = ma.masked_array(data, mask=mask )
        print 'mean = ', un[np.logical_not(mask)].mean()
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
                
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
                    cmap    = cmap.reversed()
            except:
                pass
        
        ind         = un < unthresh
        # ind[(un < unthresh)] = True
        ind[mask]   = False
        indno       = np.logical_not(ind)
        indno[mask] = False
        
        
        data2       = data_smooth[indno]
        x2          = x[indno]
        y2          = y[indno]
        im          = plt.scatter(x2, y2, s=200,  c='grey', edgecolors='k', alpha=0.8, marker='s')
        
        
        # data1       = data_smooth[ind]
        data1       = un[ind]
        x1          = x[ind]
        y1          = y[ind]
        im          = plt.scatter(x1, y1, s=200,  c=data1, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='k', alpha=0.8)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')#, ticks=[-10., -5., 0., 5., 10.])
        #
        cb.set_label('Sediment anisotropy(%)', fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        #
        lon     = self.lonArr[ind]
        lat     = self.latArr[ind]
        N       = lon.size
        areas   = np.zeros(N)
        dlon        = self.attrs['dlon']
        dlat        = self.attrs['dlat']
        data        = data_smooth[ind]
        for i in range(N):
            distEW, az, baz     = obspy.geodetics.gps2dist_azimuth(lat[i], lon[i]-dlon, lat[i], lon[i]+dlon)
            distNS, az, baz     = obspy.geodetics.gps2dist_azimuth(lat[i]-dlat, lon[i], lat[i]+dlat, lon[i])
            areas[i]   = distEW*distNS/1000.**2
        ### 
        from statsmodels import robust
        mad     = robust.mad(data)
        outmean = data.mean()
        outstd  = data.std()
        import matplotlib
        def to_percent(y, position):
            # Ignore the passed in position. This has the effect of scaling the default
            # tick locations.
            s = '%.0f' %( 100.*y)
            # The percent symbol needs escaping in latex
            if matplotlib.rcParams['text.usetex'] is True:
                return s + r'$\%$'
            else:
                return s + '%'
        ax      = plt.subplot()
        dbin    = 0.1
        bins    = np.arange(min(data), max(data) + dbin, dbin)
        weights = np.ones_like(data)/float(data.size)
        # # # data[data>3.] = 3.
        plt.hist(data, bins=bins, weights = weights)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Sediment anisotropy(%)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g , mad = %g ' %(outmean, outstd, mad), fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([vmin, vmax])
        # data2
        if showfig:
            plt.show()
        return
    
    def plot_aniso_ctr(self, icrtmtl=1, unthresh = 1., is_smooth=True, sigma=1, gsigma = 50., \
            ingrdfname=None, isthk=False, shpfx=None, outfname=None, title='', cmap='cv', \
                projection='lambert', lonplt=[], latplt=[], hillshade=False, geopolygons=None,\
                    vmin=None, vmax=None, showfig=True, depth = 5., depthavg = 0.):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
                        0 ~ 13, moho: model parameters from paraval arrays
                        vs_std      : vs_std from the model ensemble, dtype does NOT take effect
        org_mask    - use the original mask in the database or not
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'vti'   - VTI intersion using Rayleigh and Love waves
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        is_interp       = True
        if icrtmtl == 1:
            data, data_smooth\
                        = self.get_smooth_paraval(pindex=-2, dtype='avg', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            un, un_smooth\
                        = self.get_smooth_paraval(pindex=-2, dtype='std', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            # dset = invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190501_no_osci_vti_sed_25_crt_10_mantle_10_col.h5')
            # data2, data_smooth2\
            #             = dset.get_smooth_paraval(pindex=-1, dtype='avg', itype='vti', \
            #                 sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            # un2, un_smooth2\
            #             = dset.get_smooth_paraval(pindex=-1, dtype='std', itype='vti', \
            #                 sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            # mask2       = dset.attrs['mask_inv']
        else:
            data, data_smooth\
                        = self.get_smooth_paraval(pindex=-1, dtype='avg', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
            un, un_smooth\
                        = self.get_smooth_paraval(pindex=-1, dtype='std', itype='vti', \
                            sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        if is_interp:
            mask        = self.attrs['mask_interp']
        else:
            mask        = self.attrs['mask_inv']
        if is_smooth:
            mdata       = ma.masked_array(data_smooth, mask=mask )
        else:
            mdata       = ma.masked_array(data, mask=mask )
        print 'mean = ', un[np.logical_not(mask)].mean()
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap_2(projection=projection)
        #################
        from netCDF4 import Dataset
        from matplotlib.colors import LightSource
        import pycpt
        etopodata   = Dataset('/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
        etopo       = etopodata.variables['z'][:]
        lons        = etopodata.variables['x'][:]
        lats        = etopodata.variables['y'][:]
        ls          = LightSource(azdeg=315, altdeg=45)
        # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
        etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
        # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
        ny, nx      = etopo.shape
        topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
        m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
        mycm1       = pycpt.load.gmtColormap('/home/leon/station_map/etopo1.cpt')
        mycm2       = pycpt.load.gmtColormap('/home/leon/station_map/bathy1.cpt')
        mycm2.set_over('w',0)
        m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
        m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        #################
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='black')

        
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
                    cmap    = cmap.reversed()
            except:
                pass
        ind         = un < unthresh
        # ind[(un < unthresh)] = True
        ind[mask]   = False
        indno       = np.logical_not(ind)
        indno[mask] = False
        
        # sbmask      = self.get_basin_mask_inv('/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190501_150000_sed_25_crust_0_mantle_10_vti_col',\
        #                             isoutput=True)
        ###
        dataid      = 'qc_run_'+str(1)
        inh5fname   = '/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20190318_gr.h5'
        indset      = h5py.File(inh5fname)
        ingroup     = indset['reshaped_'+dataid]
        period      = 10.
        pergrp      = ingroup['%g_sec'%( period )]
        datatype    = 'vel_iso'
        vel_iso        = pergrp[datatype].value
        sbmask      = ingroup['mask1']
        self._get_lon_lat_arr(is_interp=True)
        #
        sbmask        += vel_iso > 2.5
        sbmask        += self.latArr < 68.
        #
        # if mask.shape == self.lonArr.shape:
        #     try:
        #         mask_org    = self.attrs['mask_interp']
        #         mask        += mask_org
        #         self.attrs.create(name = 'mask_interp', data = mask)
        #     except KeyError:
        #         self.attrs.create(name = 'mask_interp', data = mask)
        # else:
        #     raise ValueError('Incompatible dlon/dlat with input mask array from ray tomography database')
        ###
        
        # ind[np.logical_not(sbmask)]     = False
        # indno[np.logical_not(sbmask)]   = True
        data_smooth[np.logical_not(sbmask)] = 0
        mask_final  = np.logical_not(ind)
        # r   = 3.0
        data_smooth[data_smooth>=2.6]    = 3.1
        data_smooth[data_smooth<2.6]        = 0.
        mask_final[data_smooth==0.]     = True
        data        = ma.masked_array(data_smooth, mask=mask_final )
        
        # 
        # data[np.logical_not(sbmask)] = 0.
        # mask_final  = np.logical_not(ind)
        # data[data>=2.8]    = 3.1
        # data[data<2.8]        = 0.
        # data        = ma.masked_array(data, mask=mask_final )
        # m.contour(x, y, data, levels=[3., 4., 5.], colors=['blue', 'red', 'green'])
        # m.contour(x, y, data, levels=[3.], colors=['black'])
        
        m.pcolormesh(x, y, data, cmap='jet_r', alpha=0.2, shading='gouraud')
        # data2
        if showfig:
            plt.show()
        return 
    
    def plot_hti(self, datatype='amp_0', gindex=0, plot_axis=True, plot_data=True, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, lon_plt=[], lat_plt=[], ticks=[], msfactor=1.):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex].value
            unpsi2      = grp['unpsi2_%d' %gindex].value
            amp         = grp['amp_%d' %gindex].value
            unamp       = grp['unamp_%d' %gindex].value
        else:
            plot_axis   = False
        
        data        = grp[datatype].value
        if datatype=='unamp_0' or datatype=='unamp_1' or datatype=='unamp_2':
            data    *= msfactor
        if datatype == 'labarr':
            mask        = grp['mask_lab'].value
        elif datatype == 'slabarr':
            mask        = grp['mask_slab'].value
        elif datatype == 'dvsarr':
            mask        = grp['mask_dvs'].value
        else:
            mask        = grp['mask'].value
        #
        if datatype=='misfit' or datatype=='psi_misfit' or datatype=='amp_misfit' or datatype=='psi_misfit_crt' \
                    or datatype=='psi_misfit_man' or datatype=='psi_misfit_med':
            data    /= msfactor
        #
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        # # 
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        # 
        # import shapefile
        # shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        # shplst      = shapefile.Reader(shapefname)
        # for rec in shplst.records():
        #     lon_vol = rec[4]
        #     lat_vol = rec[3]
        #     xvol, yvol            = m(lon_vol, lat_vol)
        #     m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        #--------------------------
        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(data, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            
            if len(ticks)>0:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=ticks)
            else:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=40, rotation=0)
            cb.ax.tick_params(labelsize=40)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        if plot_axis:
            if scaled:
                # print ampref
                U       = np.sin(psi2/180.*np.pi)*amp/ampref/normv
                V       = np.cos(psi2/180.*np.pi)*amp/ampref/normv
                Uref    = np.ones(self.lonArr.shape)*1./normv
                Vref    = np.zeros(self.lonArr.shape)
            else:
                U       = np.sin(psi2/180.*np.pi)/normv
                V       = np.cos(psi2/180.*np.pi)/normv
            # rotate vectors to map projection coordinates
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
            if scaled:
                Uref, Vref, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr-360., self.latArr, returnxy=True)
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
            if masked:
                U   = ma.masked_array(U, mask=mask_psi )
                V   = ma.masked_array(V, mask=mask_psi )
            # # # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            if scaled:
                mask_ref        = np.ones(self.lonArr.shape)
                ind_lat         = np.where(self.lats==58.)[0]
                ind_lon         = np.where(self.lons==-145.+360.)[0]
                mask_ref[ind_lat, ind_lon] = False
                Uref            = ma.masked_array(Uref, mask=mask_ref )
                Vref            = ma.masked_array(Vref, mask=mask_ref )
                # m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
                # m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
                m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                m.quiver(xref, yref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
                m.quiver(xref, yref, -Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
            xref, yref = m(-145.9, 57.5)
            plt.text(xref, yref, '%g' %ampref + '%', fontsize = 20)

        plt.suptitle(title, fontsize=20)
        ###
        if len(lon_plt) == len(lat_plt) and len(lon_plt) >0:
            xc, yc      = m(lon_plt, lat_plt)
            m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        # xc, yc      = m(np.array([-155]), np.array([64]))
        # m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        # xc, yc      = m(np.array([-150]), np.array([60.5]))
        # m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        # xc, yc      = m(np.array([-155]), np.array([68.4]))
        # m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')        
        # xc, yc      = m(np.array([-144]), np.array([65.]))
        # m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        
        # xc, yc      = m(np.array([-154]), np.array([61.3]))
        # m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        
        
        
        if showfig:
            plt.show()
        return
    
    def plot_hti_vel(self, depth, depthavg=3., gindex=0, plot_axis=True, plot_data=True, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex].value
            unpsi2      = grp['unpsi2_%d' %gindex].value
            amp         = grp['amp_%d' %gindex].value
            unamp       = grp['unamp_%d' %gindex].value
        else:
            plot_axis   = False
        mask        = grp['mask'].value
        #
        #
        #
        grp         = self['avg_paraval']
        vs3d        = grp['vs_smooth'].value
        zArr        = grp['z_smooth'].value
        if depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            data    = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            data        = vs3d[:, :, index]
        
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='purple')
        
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        # 
        # import shapefile
        # shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        # shplst      = shapefile.Reader(shapefname)
        # for rec in shplst.records():
        #     lon_vol = rec[4]
        #     lat_vol = rec[3]
        #     xvol, yvol            = m(lon_vol, lat_vol)
        #     m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        #--------------------------
        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(data, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            if len(ticks)>0:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=ticks)
            else:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=35, rotation=0)
            cb.ax.tick_params(labelsize=35)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        if plot_axis:
            if scaled:
                # print ampref
                U       = np.sin(psi2/180.*np.pi)*amp/ampref/normv
                V       = np.cos(psi2/180.*np.pi)*amp/ampref/normv
                Uref    = np.ones(self.lonArr.shape)*1./normv
                Vref    = np.zeros(self.lonArr.shape)
            else:
                U       = np.sin(psi2/180.*np.pi)/normv
                V       = np.cos(psi2/180.*np.pi)/normv
            # rotate vectors to map projection coordinates
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
            if scaled:
                Uref1, Vref1, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr-360., self.latArr, returnxy=True)
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
            if masked:
                U   = ma.masked_array(U, mask=mask_psi )
                V   = ma.masked_array(V, mask=mask_psi )
            # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            if scaled:
                mask_ref        = np.ones(self.lonArr.shape)
                ind_lat         = np.where(self.lats==58.)[0]
                ind_lon         = np.where(self.lons==-145.+360.)[0]
                mask_ref[ind_lat, ind_lon] = False
                Uref            = ma.masked_array(Uref, mask=mask_ref )
                Vref            = ma.masked_array(Vref, mask=mask_ref )
                # m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                # m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                m.quiver(xref, yref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
                m.quiver(xref, yref, -Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        ##
        # reference
        xref, yref = m(-145.9, 57.5)
        plt.text(xref, yref, '%g' %ampref + '%', fontsize = 20)
        
        if depth >= 50.:
            dlst=[40., 60., 80., 100.]
            for d in dlst:
                arr             = np.loadtxt('SlabE325.dat')
                lonslb          = arr[:, 0]
                latslb          = arr[:, 1]
                depthslb        = -arr[:, 2]
                index           = (depthslb > (d - .05))*(depthslb < (d + .05))
                lonslb          = lonslb[index]
                latslb          = latslb[index]
                indsort         = lonslb.argsort()
                lonslb          = lonslb[indsort]
                latslb          = latslb[indsort]
                xslb, yslb      = m(lonslb, latslb)
                m.plot(xslb, yslb,  '-', lw = 3, color='black')
                m.plot(xslb, yslb,  '-', lw = 1, color='cyan')
        
        plt.suptitle(title, fontsize=20)
        
        
        # xc, yc      = m(np.array([-153.]), np.array([66.1]))
        # m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        # azarr       = np.arange(36.)*10.
        
        # radius      = 100.
        # g               = Geod(ellps='WGS84')
        # lonlst = []
        # latlst=[]
        # for az in azarr:
        #     
        #     outx, outy, outz = g.fwd(-153., 66.1, az, radius*1000.)
        #     lonlst.append(outx)
        #     latlst.append(outy)
        # xc, yc      = m(lonlst, latlst)
        # m.plot(xc, yc,'-', lw=3)
        
        # radius      = 3.5*35. 
        # g               = Geod(ellps='WGS84')
        # lonlst = []
        # latlst=[]
        # for az in azarr:
        #     
        #     outx, outy, outz = g.fwd(-153., 66.1, az, radius*1000.)
        #     lonlst.append(outx)
        #     latlst.append(outy)
        # xc, yc      = m(lonlst, latlst)
        # m.plot(xc, yc,'-', lw = 3)
        # 
        # radius      = 3.5*65. 
        # g               = Geod(ellps='WGS84')
        # lonlst = []
        # latlst=[]
        # for az in azarr:
        #     
        #     outx, outy, outz = g.fwd(-153., 66.1, az, radius*1000.)
        #     lonlst.append(outx)
        #     latlst.append(outy)
        # xc, yc      = m(lonlst, latlst)
        # m.plot(xc, yc,'-', lw=3.)
        
        if showfig:
            plt.show()
        return
    
    def plot_hti_vel_sb(self, gindex=0, plot_axis=True, plot_data=True, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex].value
            unpsi2      = grp['unpsi2_%d' %gindex].value
            amp         = grp['amp_%d' %gindex].value
            unamp       = grp['unamp_%d' %gindex].value
        else:
            plot_axis   = False
        mask        = grp['mask'].value
        #
        # vs data
        #
        depth = 100.
        is_smooth = True
        dtype = 'avg'
        is_interp       = self.attrs['is_interp']
        if is_interp:
            topoArr     = self['topo_interp'].value
        else:
            topoArr     = self['topo'].value
        distype = 'sedi'
        if distype is 'moho':
            if is_smooth:
                disArr  = self[dtype+'_paraval/12_smooth'].value + self[dtype+'_paraval/11_smooth'].value - topoArr
            else:
                disArr  = self[dtype+'_paraval/12_org'].value + self[dtype+'_paraval/11_org'].value - topoArr
        elif distype is 'sedi':
            if is_smooth:
                disArr  = self[dtype+'_paraval/11_smooth'].value - topoArr
            else:
                disArr  = self[dtype+'_paraval/11_org'].value - topoArr
        else:
            raise ValueError('Unexpected type of discontinuity:'+distype)
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        
        if distype is 'moho':
            depth0  = disArr
            depth1 = 200.*np.ones(depth0.shape)
        else:
            depth0  = disArr
            depth1 = 15.*np.ones(depth0.shape)
        vs_plt      = _get_vs_2d(z0=depth0, z1=depth1, zArr=zArr, vs_3d=vs3d)
        
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='purple')
        
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        # 
        #--------------------------
        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(vs_plt, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            if len(ticks)>0:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=ticks)
            else:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=35, rotation=0)
            cb.ax.tick_params(labelsize=35)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        if plot_axis:
            if scaled:
                # print ampref
                U       = np.sin(psi2/180.*np.pi)*amp/ampref/normv
                V       = np.cos(psi2/180.*np.pi)*amp/ampref/normv
                Uref    = np.ones(self.lonArr.shape)*1./normv
                Vref    = np.zeros(self.lonArr.shape)
            else:
                U       = np.sin(psi2/180.*np.pi)/normv
                V       = np.cos(psi2/180.*np.pi)/normv
            # rotate vectors to map projection coordinates
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
            if scaled:
                Uref1, Vref1, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr-360., self.latArr, returnxy=True)
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
            if masked:
                U   = ma.masked_array(U, mask=mask_psi )
                V   = ma.masked_array(V, mask=mask_psi )
            # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            if scaled:
                mask_ref        = np.ones(self.lonArr.shape)
                ind_lat         = np.where(self.lats==58.)[0]
                ind_lon         = np.where(self.lons==-145.+360.)[0]
                mask_ref[ind_lat, ind_lon] = False
                Uref            = ma.masked_array(Uref, mask=mask_ref )
                Vref            = ma.masked_array(Vref, mask=mask_ref )
                # m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                # m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                m.quiver(xref, yref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
                m.quiver(xref, yref, -Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        ##
        # reference
        xref, yref = m(-145.9, 57.5)
        plt.text(xref, yref, '%g' %ampref + '%', fontsize = 20)
        
        if depth >= 50.:
            dlst=[40., 60., 80., 100.]
            for d in dlst:
                arr             = np.loadtxt('SlabE325.dat')
                lonslb          = arr[:, 0]
                latslb          = arr[:, 1]
                depthslb        = -arr[:, 2]
                index           = (depthslb > (d - .05))*(depthslb < (d + .05))
                lonslb          = lonslb[index]
                latslb          = latslb[index]
                indsort         = lonslb.argsort()
                lonslb          = lonslb[indsort]
                latslb          = latslb[indsort]
                xslb, yslb      = m(lonslb, latslb)
                m.plot(xslb, yslb,  '-', lw = 3, color='black')
                m.plot(xslb, yslb,  '-', lw = 1, color='cyan')
        
        plt.suptitle(title, fontsize=20)
        
        if showfig:
            plt.show()
        return
    
    def plot_hti_sks(self, depth, depthavg=3., gindex=1, plot_axis=True, plot_data=False, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex].value
            unpsi2      = grp['unpsi2_%d' %gindex].value
            amp         = grp['amp_%d' %gindex].value
            unamp       = grp['unamp_%d' %gindex].value
        else:
            plot_axis   = False
        mask        = grp['mask'].value
        grp         = self['avg_paraval']
        vs3d        = grp['vs_smooth'].value
        zArr        = grp['z_smooth'].value
        if depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            data    = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            data        = vs3d[:, :, index]
        
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='purple')
        
        # yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        # yatlons             = yakutat_slb_dat[:, 0]
        # yatlats             = yakutat_slb_dat[:, 1]
        # xyat, yyat          = m(yatlons, yatlats)
        # m.plot(xyat, yyat, lw = 5, color='black', zorder=0)
        # m.plot(xyat, yyat, lw = 3, color='white', zorder=0)
        # 
        import shapefile
        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        #--------------------------
        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(data, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            if len(ticks)>0:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=ticks)
            else:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=35, rotation=0)
            cb.ax.tick_params(labelsize=35)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        if plot_axis:
            if scaled:
                # print ampref
                U       = np.sin(psi2/180.*np.pi)*amp/ampref/normv
                V       = np.cos(psi2/180.*np.pi)*amp/ampref/normv
                Uref    = np.ones(self.lonArr.shape)*1./normv
                Vref    = np.zeros(self.lonArr.shape)
            else:
                U       = np.sin(psi2/180.*np.pi)/normv
                V       = np.cos(psi2/180.*np.pi)/normv
            # rotate vectors to map projection coordinates
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
            # # # if scaled:
            # # #     Uref1, Vref1, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr-360., self.latArr, returnxy=True)
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
            if masked:
                U   = ma.masked_array(U, mask=mask_psi )
                V   = ma.masked_array(V, mask=mask_psi )

            # # # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            # # # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            
            # # # if scaled:
            # # #     mask_ref        = np.ones(self.lonArr.shape)
            # # #     ind_lat         = np.where(self.lats==58.)[0]
            # # #     ind_lon         = np.where(self.lons==-145.+360.)[0]
            # # #     mask_ref[ind_lat, ind_lon] = False
            # # #     Uref            = ma.masked_array(Uref, mask=mask_ref )
            # # #     Vref            = ma.masked_array(Vref, mask=mask_ref )
            # # #     m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # #     m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # #     m.quiver(xref, yref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
            # # #     m.quiver(xref, yref, -Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        ##
        # reference
        # # # xref, yref = m(-145.9, 57.5)
        # # # plt.text(xref, yref, '%g' %ampref + '%', fontsize = 20)
        
        # # # if depth >= 50.:
        # # #     dlst=[40., 60., 80., 100.]
        # # #     for d in dlst:
        # # #         arr             = np.loadtxt('SlabE325.dat')
        # # #         lonslb          = arr[:, 0]
        # # #         latslb          = arr[:, 1]
        # # #         depthslb        = -arr[:, 2]
        # # #         index           = (depthslb > (d - .05))*(depthslb < (d + .05))
        # # #         lonslb          = lonslb[index]
        # # #         latslb          = latslb[index]
        # # #         indsort         = lonslb.argsort()
        # # #         lonslb          = lonslb[indsort]
        # # #         latslb          = latslb[indsort]
        # # #         xslb, yslb      = m(lonslb, latslb)
        # # #         m.plot(xslb, yslb,  '-', lw = 3, color='black', zorder=0)
        # # #         m.plot(xslb, yslb,  '-', lw = 1, color='cyan', zorder=0)
        #
        #
        fname       = 'Venereau.txt'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        dtlst       = []
        lonlst      = []
        latlst      = []
        amplst      = []
        misfit      = self['hti_model/misfit'].value
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        
        with open(fname, 'rb') as fid:
            for line in fid.readlines():
                lst = line.split()
                lonsks      = float(lst[4])
                lonsks      += 360.
                latsks      = float(lst[2])
                ind_lon     = np.where(abs(self.lons - lonsks)<.2)[0][0]
                ind_lat     = np.where(abs(self.lats - latsks)<.1)[0][0]
                if mask[ind_lat, ind_lon]:
                    continue
                
                stalst.append(lst[0])
                philst.append(float(lst[5]))
                unphilst.append(float(lst[6]))
                dtlst.append(float(lst[7]))
                lonlst.append(float(lst[4]))
                latlst.append(float(lst[2]))
                psilst.append(psi2[ind_lat, ind_lon])
                unpsilst.append(unpsi2[ind_lat, ind_lon])
                amplst.append(amp[ind_lat, ind_lon])
                
                temp_misfit = misfit[ind_lat, ind_lon]
                temp_dpsi   = abs(psi2[ind_lat, ind_lon] - float(lst[5]))
                if temp_dpsi > 90.:
                    temp_dpsi   = 180. - temp_dpsi
                    
                # if self.lons[ind_lon] < -140.+360.:
                #     continue
                
                # if amp[ind_lat, ind_lon] < .3:
                #     continue
                # if self.lats[ind_lat] > 61.:
                #     continue
                if temp_misfit > 1. and temp_dpsi > 30. or temp_dpsi > 30. and self.lons[ind_lon] > -140.+360.:
                    vpr = self.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=True,\
                                outlon=self.lons[ind_lon], outlat=self.lats[ind_lat])
                    vpr.linear_inv_hti(depth_mid_crust=-1., depth_mid_mantle=100.)
                    psilst1.append(vpr.model.htimod.psi2[1])
                    psilst2.append(vpr.model.htimod.psi2[2])
                    lonlst2.append(float(lst[4]))
                    latlst2.append(float(lst[2]))
        phiarr  = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        unphiarr= np.array(unphilst)
        unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        dtarr   = np.array(dtlst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        dtref   = 1.
        normv   = 2.
        
        # # # Usks    = np.sin(phiarr/180.*np.pi)*dtarr/dtref/normv
        # # # Vsks    = np.cos(phiarr/180.*np.pi)*dtarr/dtref/normv
        
        Usks    = np.sin(phiarr/180.*np.pi)/dtref/normv
        Vsks    = np.cos(phiarr/180.*np.pi)/dtref/normv
        
        Upsi    = np.sin(psiarr/180.*np.pi)/dtref/normv
        Vpsi    = np.cos(psiarr/180.*np.pi)/dtref/normv
        
        Uref    = np.ones(self.lonArr.shape)*1./normv
        Vref    = np.zeros(self.lonArr.shape)
        Usks, Vsks, xsks, ysks  = m.rotate_vector(Usks, Vsks, lonarr, latarr, returnxy=True)
        mask    = np.zeros(Usks.size, dtype=bool)
        
        dpsi            = abs(psiarr - phiarr)
        dpsi2 = (psiarr - phiarr)
        dpsi2[dpsi2>90.] = dpsi2[dpsi2>90.] - 180.
        dpsi2[dpsi2<-90.] = dpsi2[dpsi2<-90.] + 180.
        # dpsi
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        print 'mean damp', amparr.mean() 
        undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        undpsi2 = undpsi.copy()
        # print 'un:', unpsiarr.mean(), unphiarr.mean(), undpsi.mean()
        # return unpsiarr, unphiarr
        # # # ind_outline         = amparr < .2
        
        # 81 % comparisonH
        mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        mask[(amparr<.32)*(dpsi>=30.)]   = True # 2 p
        # # mask[(amparr<.4)*(dpsi>=30.)]   = True # 3 p
        
        # mask[(amparr<.4)]   = True
        # return amparr, mask
        print 'mean damp', amparr[np.logical_not(mask)].mean() 
        ###
        
        # mask[(amparr<.2)*(dpsi>=30.)]   = True
        # mask[(amparr<.3)*(dpsi>=30.)*(lonarr<-140.)]   = True
        
        
        xsks    = xsks[np.logical_not(mask)]
        ysks    = ysks[np.logical_not(mask)]
        Usks    = Usks[np.logical_not(mask)]
        Vsks    = Vsks[np.logical_not(mask)]
        Upsi    = Upsi[np.logical_not(mask)]
        Vpsi    = Vpsi[np.logical_not(mask)]
        dpsi    = dpsi[np.logical_not(mask)]
        undpsi = undpsi[np.logical_not(mask)]
        # dpsi2 = dpsi2[np.logical_not(mask)]

        # # # Q1      = m.quiver(xsks, ysks, Usks, Vsks, scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b')
        # # # Q2      = m.quiver(xsks, ysks, -Usks, -Vsks, scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b')
        Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Usks[dpsi<=30.], Vsks[dpsi<=30.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Usks[dpsi<=30.], -Vsks[dpsi<=30.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q1      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], Usks[(dpsi>30.)*(dpsi<=60.)], Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q2      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], -Usks[(dpsi>30.)*(dpsi<=60.)], -Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q1      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], Usks[dpsi>60.], Vsks[dpsi>60.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        Q2      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], -Usks[dpsi>60.], -Vsks[dpsi>60.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        
        # # # Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Upsi[dpsi<=30.], Vpsi[dpsi<=30.], scale=20, width=width-0.001, headaxislength=0, headlength=0, headwidth=0.5, color='r')
        # # # Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Upsi[dpsi<=30.], -Vpsi[dpsi<=30.], scale=20, width=width-0.001, headaxislength=0, headlength=0, headwidth=0.5, color='r')
        # # # 
        # # # Q1      = m.quiver(xsks[dpsi>30.], ysks[dpsi>30.], Upsi[dpsi>30.], Vpsi[dpsi>30.], scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        # # # Q2      = m.quiver(xsks[dpsi>30.], ysks[dpsi>30.], -Upsi[dpsi>30.], -Vpsi[dpsi>30.], scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        
            #         Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        # # # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        # # # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        
        
        # if len(psilst1) > 0.:
        #     Upsi2   = np.sin(np.array(psilst2)/180.*np.pi)/dtref/normv
        #     Vpsi2   = np.cos(np.array(psilst2)/180.*np.pi)/dtref/normv
        #     # print np.array(psilst2)
        #     # ind = np.array(lonlst2).argmax()
        #     Upsi2[0] = Upsi2[1]
        #     Vpsi2[0] = Vpsi2[1]
        #     Upsi2, Vpsi2, xsks2, ysks2  = m.rotate_vector(Upsi2, Vpsi2, np.array(lonlst2), np.array(latlst2), returnxy=True)
        #     Q1      = m.quiver(xsks2, ysks2, Upsi2, Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        #     Q2      = m.quiver(xsks2, ysks2, -Upsi2, -Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            
            
        plt.suptitle(title, fontsize=20)
        plt.show()
        
        ax      = plt.subplot()
        dbin    = 10.
        bins    = np.arange(min(dpsi), max(dpsi) + dbin, dbin)
        
        weights = np.ones_like(dpsi)/float(dpsi.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(dpsi)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)

        p.patches[3].set_color('lime')
        p.patches[4].set_color('lime')
        p.patches[5].set_color('lime')
        p.patches[6].set_color('r')
        p.patches[7].set_color('r')
        p.patches[8].set_color('r')
        
        # # # print dpsi.size
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        good_per= float(dpsi[dpsi<30.].size)/float(dpsi.size)
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()
        
        #
        # # # # print dpsi.size
        # ax      = plt.subplot()
        # import matplotlib.mlab as mlab
        # from matplotlib.ticker import FuncFormatter
        # # good_per= float(dpsi[dpsi<30.].size)/float(dpsi.size)
        # # plt.ylabel('Percentage (%)', fontsize=60)
        # # print dpsi2.shape, undpsi2.shape
        # weights = np.ones_like(dpsi2)/float(dpsi2.size)
        # plt.hist(dpsi2/undpsi2, weights=weights, bins=25)
        # print (dpsi2/undpsi2).std()
        # plt.xlabel('Normalized Angle difference (deg)', fontsize=60, rotation=0)
        # # plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        # ax.tick_params(axis='x', labelsize=40)
        # # plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        # ax.tick_params(axis='y', labelsize=40)
        # formatter = FuncFormatter(to_percent)
        # # Set the formatter
        # plt.gca().yaxis.set_major_formatter(formatter)
        # plt.xlim([-10, 10.])
        # plt.show()
            
        return
    
    def plot_amp_sks(self, gindex=1, plot_axis=True, plot_data=True, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        psi2        = grp['psi2_%d' %gindex].value
        unpsi2      = grp['unpsi2_%d' %gindex].value
        amp         = grp['amp_%d' %gindex].value
        unamp       = grp['unamp_%d' %gindex].value
        mask        = grp['mask'].value

        #
        #
        fname       = 'Venereau.txt'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        dtlst       = []
        dtlst2      = []
        undtlst2    = []
        undtlst     = []
        lonlst      = []
        latlst      = []
        amplst      = []
        misfit      = self['hti_model/misfit'].value
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        
        with open(fname, 'rb') as fid:
            for line in fid.readlines():
                lst = line.split()
                lonsks      = float(lst[4])
                lonsks      += 360.
                latsks      = float(lst[2])
                ind_lon     = np.where(abs(self.lons - lonsks)<.2)[0][0]
                ind_lat     = np.where(abs(self.lats - latsks)<.1)[0][0]
                if mask[ind_lat, ind_lon]:
                    continue
                
                stalst.append(lst[0])
                philst.append(float(lst[5]))
                unphilst.append(float(lst[6]))
                
                lonlst.append(float(lst[4]))
                latlst.append(float(lst[2]))
                psilst.append(psi2[ind_lat, ind_lon])
                unpsilst.append(unpsi2[ind_lat, ind_lon])
                amplst.append(amp[ind_lat, ind_lon])
                
                d2d = np.zeros((2,2))
                # # # # upper crust
                d2d[0, 0] = -1
                d2d[0, 1] = 15.
                d2d[1, 0] = -2
                d2d[1, 1] = 150.
                
                vpr     = self.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=False,\
                                outlon=self.lons[ind_lon], outlat=self.lats[ind_lat], depth2d=d2d)
                # print vpr.model.htimod.layer_ind
                harr    = vpr.model.h[vpr.model.htimod.layer_ind[1, 0]:vpr.model.htimod.layer_ind[1, 1]]
                vsarr   = vpr.model.vsv[vpr.model.htimod.layer_ind[1, 0]:vpr.model.htimod.layer_ind[1, 1]]
                tamp    = amp[ind_lat, ind_lon]
                temp_dt = ((harr/vsarr/(1.-tamp/100.)).sum() - (harr/vsarr/(1+tamp/100.)).sum())*2.
                tunamp  = unamp[ind_lat, ind_lon]
                temp_undt = ((harr/vsarr/(1.-tunamp/100.)).sum() - (harr/vsarr/(1+tunamp/100.)).sum())*2.
                
                if float(lst[7]) > 2.5:
                    continue
                dphi = abs(float(lst[5]) - psi2[ind_lat, ind_lon])
                if dphi>90.:
                    dphi    = 180. -dphi
                if dphi > 30.:
                    continue
                dtlst2.append(temp_dt)
                dtlst.append(float(lst[7]))
                undtlst2.append(temp_undt)
                undtlst.append(float(lst[8]))
        # print 
        phiarr  = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        unphiarr= np.array(unphilst)
        unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        dtarr   = np.array(dtlst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        dtref   = 1.
        print amparr.max(), amparr.mean()
        plt.figure(figsize=[10, 10])
        ax      = plt.subplot()
        # plt.plot(dtlst, dtlst2, 'o', ms=10)
        plt.errorbar(dtlst, dtlst2, yerr=undtlst2, xerr=undtlst, fmt='ko', ms=8)
        plt.plot([0., 2.5], [0., 2.5], 'b--', lw=3)
        # plt.ylabel('Predicted delay time', fontsize=60)
        # plt.xlabel('Observed delay time', fontsize=60, rotation=0)yerr
        # plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        ax.tick_params(axis='x', labelsize=30)
        # plt.xticks([0., 0.5, 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=30)

        plt.axis(option='equal', ymin=0., ymax=2.5, xmin=0., xmax = 2.5)
        
        import pandas as pd
        # s = pd.Series(dpsi)
        # p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        
        diffdata= np.array(dtlst2)- np.array(dtlst)
        print "mean dt: ",np.array(dtlst2).mean(), np.array(dtlst).mean()
        dbin    = .15
        bins    = np.arange(min(diffdata), max(diffdata) + dbin, dbin)
        
        weights = np.ones_like(diffdata)/float(diffdata.size)
        # print bins.size
        import pandas as pd

        plt.figure()
        ax      = plt.subplot()
        plt.hist(diffdata, bins=bins, weights = weights)
        plt.title('mean = %g , std = %g' %(diffdata.mean(), diffdata.std()) , fontsize=30)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        # # # good_per= float(dpsi[dpsi<30.].size)/float(dpsi.size)
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Delay time difference (s)', fontsize=60, rotation=0)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        # plt.xlim([-2., 0.])
        plt.show()
            
        return
    
    def plot_hti_flow(self, depth, depthavg=3., gindex=1, plot_axis=True, plot_data=False, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex].value
            unpsi2      = grp['unpsi2_%d' %gindex].value
            amp         = grp['amp_%d' %gindex].value
            unamp       = grp['unamp_%d' %gindex].value
        else:
            plot_axis   = False
        mask        = grp['mask'].value
        grp         = self['avg_paraval']
        vs3d        = grp['vs_smooth'].value
        zArr        = grp['z_smooth'].value
        if depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            data    = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            data        = vs3d[:, :, index]
        
        mdata       = ma.masked_array(data, mask=mask )
        
        
        
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='purple')
        
        # yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        # yatlons             = yakutat_slb_dat[:, 0]
        # yatlats             = yakutat_slb_dat[:, 1]
        # xyat, yyat          = m(yatlons, yatlats)
        # m.plot(xyat, yyat, lw = 5, color='black', zorder=0)
        # m.plot(xyat, yyat, lw = 3, color='white', zorder=0)
        # 
        import shapefile
        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        #--------------------------
        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(data, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            if len(ticks)>0:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=ticks)
            else:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=35, rotation=0)
            cb.ax.tick_params(labelsize=35)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        if plot_axis:
            if scaled:
                # print ampref
                U       = np.sin(psi2/180.*np.pi)*amp/ampref/normv
                V       = np.cos(psi2/180.*np.pi)*amp/ampref/normv
                Uref    = np.ones(self.lonArr.shape)*1./normv
                Vref    = np.zeros(self.lonArr.shape)
            else:
                U       = np.sin(psi2/180.*np.pi)/normv
                V       = np.cos(psi2/180.*np.pi)/normv
            # rotate vectors to map projection coordinates
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
            # # # if scaled:
            # # #     Uref1, Vref1, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr-360., self.latArr, returnxy=True)
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
            if masked:
                U   = ma.masked_array(U, mask=mask_psi )
                V   = ma.masked_array(V, mask=mask_psi )

            # # # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            # # # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            
            # # # if scaled:
            # # #     mask_ref        = np.ones(self.lonArr.shape)
            # # #     ind_lat         = np.where(self.lats==58.)[0]
            # # #     ind_lon         = np.where(self.lons==-145.+360.)[0]
            # # #     mask_ref[ind_lat, ind_lon] = False
            # # #     Uref            = ma.masked_array(Uref, mask=mask_ref )
            # # #     Vref            = ma.masked_array(Vref, mask=mask_ref )
            # # #     m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # #     m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # # #     m.quiver(xref, yref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
            # # #     m.quiver(xref, yref, -Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        ##
        # reference
        # # # xref, yref = m(-145.9, 57.5)
        # # # plt.text(xref, yref, '%g' %ampref + '%', fontsize = 20)
        
        # # # if depth >= 50.:
        # # #     dlst=[40., 60., 80., 100.]
        # # #     for d in dlst:
        # # #         arr             = np.loadtxt('SlabE325.dat')
        # # #         lonslb          = arr[:, 0]
        # # #         latslb          = arr[:, 1]
        # # #         depthslb        = -arr[:, 2]
        # # #         index           = (depthslb > (d - .05))*(depthslb < (d + .05))
        # # #         lonslb          = lonslb[index]
        # # #         latslb          = latslb[index]
        # # #         indsort         = lonslb.argsort()
        # # #         lonslb          = lonslb[indsort]
        # # #         latslb          = latslb[indsort]
        # # #         xslb, yslb      = m(lonslb, latslb)
        # # #         m.plot(xslb, yslb,  '-', lw = 3, color='black', zorder=0)
        # # #         m.plot(xslb, yslb,  '-', lw = 1, color='cyan', zorder=0)
        #
        #
        fname       = 'Venereau.txt'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        dtlst       = []
        lonlst      = []
        latlst      = []
        amplst      = []
        misfit      = self['hti_model/misfit'].value
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        
        with open(fname, 'rb') as fid:
            for line in fid.readlines():
                lst = line.split()
                lonsks      = float(lst[4])
                lonsks      += 360.
                latsks      = float(lst[2])
                ind_lon     = np.where(abs(self.lons - lonsks)<.2)[0][0]
                ind_lat     = np.where(abs(self.lats - latsks)<.1)[0][0]
                if mask[ind_lat, ind_lon]:
                    continue
                
                stalst.append(lst[0])
                philst.append(float(lst[5]))
                unphilst.append(float(lst[6]))
                dtlst.append(float(lst[7]))
                lonlst.append(float(lst[4]))
                latlst.append(float(lst[2]))
                psilst.append(psi2[ind_lat, ind_lon])
                unpsilst.append(unpsi2[ind_lat, ind_lon])
                amplst.append(amp[ind_lat, ind_lon])
                
                temp_misfit = misfit[ind_lat, ind_lon]
                temp_dpsi   = abs(psi2[ind_lat, ind_lon] - float(lst[5]))
                if temp_dpsi > 90.:
                    temp_dpsi   = 180. - temp_dpsi
                    
                # if self.lons[ind_lon] < -140.+360.:
                #     continue
                
                # if amp[ind_lat, ind_lon] < .3:
                #     continue
                # if self.lats[ind_lat] > 61.:
                #     continue
                if temp_misfit > 1. and temp_dpsi > 30. or temp_dpsi > 30. and self.lons[ind_lon] > -140.+360.:
                    vpr = self.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=True,\
                                outlon=self.lons[ind_lon], outlat=self.lats[ind_lat])
                    vpr.linear_inv_hti(depth_mid_crust=-1., depth_mid_mantle=100.)
                    psilst1.append(vpr.model.htimod.psi2[1])
                    psilst2.append(vpr.model.htimod.psi2[2])
                    lonlst2.append(float(lst[4]))
                    latlst2.append(float(lst[2]))
        phiarr  = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        unphiarr= np.array(unphilst)
        unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        dtarr   = np.array(dtlst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        dtref   = 1.
        normv   = 2.
        
        # # # Usks    = np.sin(phiarr/180.*np.pi)*dtarr/dtref/normv
        # # # Vsks    = np.cos(phiarr/180.*np.pi)*dtarr/dtref/normv
        
        Usks    = np.sin(phiarr/180.*np.pi)/dtref/normv
        Vsks    = np.cos(phiarr/180.*np.pi)/dtref/normv
        
        Upsi    = np.sin(psiarr/180.*np.pi)/dtref/normv
        Vpsi    = np.cos(psiarr/180.*np.pi)/dtref/normv
        
        Uref    = np.ones(self.lonArr.shape)*1./normv
        Vref    = np.zeros(self.lonArr.shape)
        Usks, Vsks, xsks, ysks  = m.rotate_vector(Usks, Vsks, lonarr, latarr, returnxy=True)
        mask    = np.zeros(Usks.size, dtype=bool)
        
        dpsi            = abs(psiarr - phiarr)
        dpsi2 = (psiarr - phiarr)
        dpsi2[dpsi2>90.] = dpsi2[dpsi2>90.] - 180.
        dpsi2[dpsi2<-90.] = dpsi2[dpsi2<-90.] + 180.
        # dpsi
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        print 'mean damp', amparr.mean() 
        undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        undpsi2 = undpsi.copy()
        # print 'un:', unpsiarr.mean(), unphiarr.mean(), undpsi.mean()
        # return unpsiarr, unphiarr
        # # # ind_outline         = amparr < .2
        
        # 81 % comparisonH
        mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        mask[(amparr<.32)*(dpsi>=30.)]   = True # 2 p
        # # mask[(amparr<.4)*(dpsi>=30.)]   = True # 3 p
        
        # mask[(amparr<.4)]   = True
        # return amparr, mask
        print 'mean damp', amparr[np.logical_not(mask)].mean() 
        ###
        
        # mask[(amparr<.2)*(dpsi>=30.)]   = True
        # mask[(amparr<.3)*(dpsi>=30.)*(lonarr<-140.)]   = True
        
        
        xsks    = xsks[np.logical_not(mask)]
        ysks    = ysks[np.logical_not(mask)]
        Usks    = Usks[np.logical_not(mask)]
        Vsks    = Vsks[np.logical_not(mask)]
        Upsi    = Upsi[np.logical_not(mask)]
        Vpsi    = Vpsi[np.logical_not(mask)]
        dpsi    = dpsi[np.logical_not(mask)]
        undpsi = undpsi[np.logical_not(mask)]
        # dpsi2 = dpsi2[np.logical_not(mask)]

        # # # Q1      = m.quiver(xsks, ysks, Usks, Vsks, scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b')
        # # # Q2      = m.quiver(xsks, ysks, -Usks, -Vsks, scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b')
        Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Usks[dpsi<=30.], Vsks[dpsi<=30.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Usks[dpsi<=30.], -Vsks[dpsi<=30.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q1      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], Usks[(dpsi>30.)*(dpsi<=60.)], Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q2      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], -Usks[(dpsi>30.)*(dpsi<=60.)], -Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q1      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], Usks[dpsi>60.], Vsks[dpsi>60.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        Q2      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], -Usks[dpsi>60.], -Vsks[dpsi>60.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        
        # # # Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Upsi[dpsi<=30.], Vpsi[dpsi<=30.], scale=20, width=width-0.001, headaxislength=0, headlength=0, headwidth=0.5, color='r')
        # # # Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Upsi[dpsi<=30.], -Vpsi[dpsi<=30.], scale=20, width=width-0.001, headaxislength=0, headlength=0, headwidth=0.5, color='r')
        # # # 
        # # # Q1      = m.quiver(xsks[dpsi>30.], ysks[dpsi>30.], Upsi[dpsi>30.], Vpsi[dpsi>30.], scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        # # # Q2      = m.quiver(xsks[dpsi>30.], ysks[dpsi>30.], -Upsi[dpsi>30.], -Vpsi[dpsi>30.], scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        
            #         Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        # # # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        # # # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        
        
        # if len(psilst1) > 0.:
        #     Upsi2   = np.sin(np.array(psilst2)/180.*np.pi)/dtref/normv
        #     Vpsi2   = np.cos(np.array(psilst2)/180.*np.pi)/dtref/normv
        #     # print np.array(psilst2)
        #     # ind = np.array(lonlst2).argmax()
        #     Upsi2[0] = Upsi2[1]
        #     Vpsi2[0] = Vpsi2[1]
        #     Upsi2, Vpsi2, xsks2, ysks2  = m.rotate_vector(Upsi2, Vpsi2, np.array(lonlst2), np.array(latlst2), returnxy=True)
        #     Q1      = m.quiver(xsks2, ysks2, Upsi2, Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        #     Q2      = m.quiver(xsks2, ysks2, -Upsi2, -Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            
            
        plt.suptitle(title, fontsize=20)
        plt.show()
        
        ax      = plt.subplot()
        dbin    = 10.
        bins    = np.arange(min(dpsi), max(dpsi) + dbin, dbin)
        
        weights = np.ones_like(dpsi)/float(dpsi.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(dpsi)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)

        p.patches[3].set_color('lime')
        p.patches[4].set_color('lime')
        p.patches[5].set_color('lime')
        p.patches[6].set_color('r')
        p.patches[7].set_color('r')
        p.patches[8].set_color('r')
        
        # # # print dpsi.size
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        good_per= float(dpsi[dpsi<30.].size)/float(dpsi.size)
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()
        
        #
        # # # # print dpsi.size
        # ax      = plt.subplot()
        # import matplotlib.mlab as mlab
        # from matplotlib.ticker import FuncFormatter
        # # good_per= float(dpsi[dpsi<30.].size)/float(dpsi.size)
        # # plt.ylabel('Percentage (%)', fontsize=60)
        # # print dpsi2.shape, undpsi2.shape
        # weights = np.ones_like(dpsi2)/float(dpsi2.size)
        # plt.hist(dpsi2/undpsi2, weights=weights, bins=25)
        # print (dpsi2/undpsi2).std()
        # plt.xlabel('Normalized Angle difference (deg)', fontsize=60, rotation=0)
        # # plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        # ax.tick_params(axis='x', labelsize=40)
        # # plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        # ax.tick_params(axis='y', labelsize=40)
        # formatter = FuncFormatter(to_percent)
        # # Set the formatter
        # plt.gca().yaxis.set_major_formatter(formatter)
        # plt.xlim([-10, 10.])
        # plt.show()
            
        return
    
    
    def plot_hti_doublelay(self, depth, depthavg=3., gindex=0, plot_axis=True, plot_data=True, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex].value
            unpsi2      = grp['unpsi2_%d' %gindex].value
            amp         = grp['amp_%d' %gindex].value
            unamp       = grp['unamp_%d' %gindex].value
        else:
            plot_axis   = False
        mask        = grp['mask'].value
        grp         = self['avg_paraval']
        vs3d        = grp['vs_smooth'].value
        zArr        = grp['z_smooth'].value
        if depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            data    = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            data        = vs3d[:, :, index]
        
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='purple')
        
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black', zorder=0)
        m.plot(xyat, yyat, lw = 3, color='white', zorder=0)
        # 
        import shapefile
        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        #--------------------------
        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(data, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            if len(ticks)>0:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=ticks)
            else:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=35, rotation=0)
            cb.ax.tick_params(labelsize=35)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        if plot_axis:
            if scaled:
                # print ampref
                U       = np.sin(psi2/180.*np.pi)*amp/ampref/normv
                V       = np.cos(psi2/180.*np.pi)*amp/ampref/normv
                Uref    = np.ones(self.lonArr.shape)*1./normv
                Vref    = np.zeros(self.lonArr.shape)
            else:
                U       = np.sin(psi2/180.*np.pi)/normv
                V       = np.cos(psi2/180.*np.pi)/normv
            # rotate vectors to map projection coordinates
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
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
            if masked:
                U   = ma.masked_array(U, mask=mask_psi )
                V   = ma.masked_array(V, mask=mask_psi )
        
        if depth >= 50.:
            dlst=[40., 60., 80., 95.]
            for d in dlst:
                arr             = np.loadtxt('SlabE325.dat')
                lonslb          = arr[:, 0]
                latslb          = arr[:, 1]
                depthslb        = -arr[:, 2]
                index           = (depthslb > (d - .05))*(depthslb < (d + .05))
                lonslb          = lonslb[index]
                latslb          = latslb[index]
                indsort         = lonslb.argsort()
                lonslb          = lonslb[indsort]
                latslb          = latslb[indsort]
                xslb, yslb      = m(lonslb, latslb)
                # m.plot(xslb, yslb,  '-', lw = 3, color='black', zorder=0)
                m.plot(xslb, yslb,  '-', lw = 1, color='cyan', zorder=0)
            
            
                arr             = np.loadtxt('SlabE115.dat')
                lonslb          = arr[:, 0]
                latslb          = arr[:, 1]
                depthslb        = -arr[:, 2]
                index           = (depthslb > (d - .05))*(depthslb < (d + .05))
                lonslb          = lonslb[index]
                latslb          = latslb[index]
                indsort         = lonslb.argsort()
                lonslb          = lonslb[indsort]
                latslb          = latslb[indsort]
                xslb, yslb      = m(lonslb, latslb)
                m.plot(xslb, yslb,  '--', lw = 1, color='red', zorder=0)
        #
        #
        fname       = 'Venereau.txt'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        dtlst       = []
        lonlst      = []
        latlst      = []
        amplst      = []
        misfit      = self['hti_model/misfit'].value
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        
        with open(fname, 'rb') as fid:
            for line in fid.readlines():
                lst = line.split()
                lonsks      = float(lst[4])
                lonsks      += 360.
                latsks      = float(lst[2])
                ind_lon     = np.where(abs(self.lons - lonsks)<.2)[0][0]
                ind_lat     = np.where(abs(self.lats - latsks)<.1)[0][0]
                if mask[ind_lat, ind_lon]:
                    continue
                temp_misfit = misfit[ind_lat, ind_lon]
                temp_dpsi   = abs(psi2[ind_lat, ind_lon] - float(lst[5]))
                if temp_dpsi > 90.:
                    temp_dpsi   = 180. - temp_dpsi
                if self.lons[ind_lon] < -140.+360.:
                    continue
                if temp_misfit > 1. and temp_dpsi > 30. or temp_dpsi > 30. and self.lons[ind_lon] > -140.+360.:
                    vpr = self.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=True,\
                                outlon=self.lons[ind_lon], outlat=self.lats[ind_lat])
                    vpr.linear_inv_hti(depth_mid_crust=-1., depth_mid_mantle=100.)
                    psilst1.append(vpr.model.htimod.psi2[1])
                    psilst2.append(vpr.model.htimod.psi2[2])
                    lonlst2.append(float(lst[4]))
                    latlst2.append(float(lst[2]))
                    #
                    print lonsks-360., latsks
                    stalst.append(lst[0])
                    philst.append(float(lst[5]))
                    unphilst.append(float(lst[6]))
                    dtlst.append(float(lst[7]))
                    lonlst.append(float(lst[4]))
                    latlst.append(float(lst[2]))
                    psilst.append(psi2[ind_lat, ind_lon])
                    unpsilst.append(unpsi2[ind_lat, ind_lon])
                    amplst.append(amp[ind_lat, ind_lon])
                
        phiarr  = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        unphiarr= np.array(unphilst)
        unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        dtarr   = np.array(dtlst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        dtref   = 1.
        normv   = 1.5

        Usks    = np.sin(phiarr/180.*np.pi)/dtref/normv
        Vsks    = np.cos(phiarr/180.*np.pi)/dtref/normv
        
        Upsi    = np.sin(psiarr/180.*np.pi)/dtref/normv
        Vpsi    = np.cos(psiarr/180.*np.pi)/dtref/normv
        
        Uref    = np.ones(self.lonArr.shape)*1./normv
        Vref    = np.zeros(self.lonArr.shape)
        Usks, Vsks, xsks, ysks  = m.rotate_vector(Usks, Vsks, lonarr, latarr, returnxy=True)
        mask    = np.zeros(Usks.size, dtype=bool)
        
        dpsi            = abs(psiarr - phiarr)
        # dpsi
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        
        undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        # return unpsiarr, unphiarr
        # # # ind_outline         = amparr < .2
        # mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        # mask[(amparr<.2)*(dpsi>=20.)]   = True
        # mask[(amparr<.3)*(dpsi>=30.)*(lonarr<-140.)]   = True
        
        xsks    = xsks[np.logical_not(mask)]
        ysks    = ysks[np.logical_not(mask)]
        Usks    = Usks[np.logical_not(mask)]
        Vsks    = Vsks[np.logical_not(mask)]
        Upsi    = Upsi[np.logical_not(mask)]
        Vpsi    = Vpsi[np.logical_not(mask)]
        dpsi    = dpsi[np.logical_not(mask)]

        Q1      = m.quiver(xsks, ysks, Usks, Vsks, scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Usks, -Vsks, scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        
        
        
        if len(psilst1) > 0.:
            Upsi2   = np.sin(np.array(psilst2)/180.*np.pi)/dtref/normv
            Vpsi2   = np.cos(np.array(psilst2)/180.*np.pi)/dtref/normv
            Upsi2[0] = Upsi2[1]
            Vpsi2[0] = Vpsi2[1]
            Upsi2, Vpsi2, xsks2, ysks2  = m.rotate_vector(Upsi2, Vpsi2, np.array(lonlst2), np.array(latlst2), returnxy=True)
            Q1      = m.quiver(xsks2, ysks2, Upsi2, Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='red', zorder=3)
            Q2      = m.quiver(xsks2, ysks2, -Upsi2, -Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='red', zorder=3)
            
            Upsi2   = np.sin(np.array(psilst1)/180.*np.pi)/dtref/normv
            Vpsi2   = np.cos(np.array(psilst1)/180.*np.pi)/dtref/normv
            Upsi2[0] = Upsi2[1]
            Vpsi2[0] = Vpsi2[1]
            Upsi2, Vpsi2, xsks2, ysks2  = m.rotate_vector(Upsi2, Vpsi2, np.array(lonlst2), np.array(latlst2), returnxy=True)
            Q1      = m.quiver(xsks2, ysks2, Upsi2, Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='blue', zorder=4)
            Q2      = m.quiver(xsks2, ysks2, -Upsi2, -Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='blue', zorder=4)
        
        # Koyukuk
        vpr = self.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=True,\
                                outlon=-153.+360., outlat=66.1)
        vpr.linear_inv_hti(depth_mid_crust=-1., depth_mid_mantle=100.)
        Upsi   = np.sin(np.array([vpr.model.htimod.psi2[2]])/180.*np.pi)/dtref/normv
        Vpsi   = np.cos(np.array([vpr.model.htimod.psi2[2]])/180.*np.pi)/dtref/normv
        Upsi, Vpsi, xpsi, ypsi  = m.rotate_vector(Upsi, Vpsi, np.array([-153.]), np.array([66.1]), returnxy=True)
        Q1      = m.quiver(xpsi, ypsi, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='red', zorder=3)
        Q2      = m.quiver(xpsi, ypsi, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='red', zorder=3)
        
        Upsi   = np.sin(np.array([vpr.model.htimod.psi2[1]])/180.*np.pi)/dtref/normv
        Vpsi   = np.cos(np.array([vpr.model.htimod.psi2[1]])/180.*np.pi)/dtref/normv
        Upsi, Vpsi, xpsi, ypsi  = m.rotate_vector(Upsi, Vpsi, np.array([-153.]), np.array([66.1]), returnxy=True)
        Q1      = m.quiver(xpsi, ypsi, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='blue', zorder=4)
        Q2      = m.quiver(xpsi, ypsi, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='blue', zorder=4)
        
        # WVF
        vpr = self.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=True,\
                                outlon=-144.+360., outlat=62.)
        vpr.linear_inv_hti(depth_mid_crust=-1., depth_mid_mantle=70.)
        Upsi   = np.sin(np.array([vpr.model.htimod.psi2[2]])/180.*np.pi)/dtref/normv
        Vpsi   = np.cos(np.array([vpr.model.htimod.psi2[2]])/180.*np.pi)/dtref/normv
        Upsi, Vpsi, xpsi, ypsi  = m.rotate_vector(Upsi, Vpsi, np.array([-144.]), np.array([62.]), returnxy=True)
        Q1      = m.quiver(xpsi, ypsi, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='red', zorder=3)
        Q2      = m.quiver(xpsi, ypsi, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='red', zorder=3)
        
        Upsi   = np.sin(np.array([vpr.model.htimod.psi2[1]])/180.*np.pi)/dtref/normv
        Vpsi   = np.cos(np.array([vpr.model.htimod.psi2[1]])/180.*np.pi)/dtref/normv
        Upsi, Vpsi, xpsi, ypsi  = m.rotate_vector(Upsi, Vpsi, np.array([-144.]), np.array([62.]), returnxy=True)
        Q1      = m.quiver(xpsi, ypsi, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='blue', zorder=4)
        Q2      = m.quiver(xpsi, ypsi, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='blue', zorder=4) 
            
            
            
        plt.show()
            
        return
    
    def plot_hti_diff_misfit(self, inh5fname, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, lon_plt=[], lat_plt=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        data        = grp['misfit'].value
        mask        = grp['mask'].value
        indset      = h5py.File(inh5fname)
        grp2        = indset['hti_model']
        data2       = grp2['misfit'].value
        mask2       = grp2['mask'].value
        
        diffdata    = data - data2
        # return diffdata
        mdata       = ma.masked_array(diffdata, mask=mask + mask2 )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        # # 
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')

        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=40, rotation=0)
        cb.ax.tick_params(labelsize=40)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")

        plt.suptitle(title, fontsize=20)
        ###
        if len(lon_plt) == len(lat_plt) and len(lon_plt) >0:
            xc, yc      = m(lon_plt, lat_plt)
            m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        if showfig:
            plt.show()
        return
    
    def plot_hti_diff_psi(self, inh5fname, gindex, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, lon_plt=[], lat_plt=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        data        = grp['psi2_%d' %gindex].value
        mask        = grp['mask'].value
        
        indset      = h5py.File(inh5fname)
        grp2        = indset['hti_model']
        data2       = grp2['psi2_%d' %gindex].value
        mask2       = grp2['mask'].value
        
        diffdata    = abs(data - data2)
        diffdata[diffdata>90.]  -= 90. 
        # return diffdata
        mdata       = ma.masked_array(diffdata, mask=mask + mask2 )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        # # 
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')

        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=40, rotation=0)
        cb.ax.tick_params(labelsize=40)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")

        plt.suptitle(title, fontsize=20)
        ###
        if len(lon_plt) == len(lat_plt) and len(lon_plt) >0:
            xc, yc      = m(lon_plt, lat_plt)
            m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        if showfig:
            plt.show()
        
        
        ax      = plt.subplot()
        # # # ###
        # # # data /= 1.4
        # # # ###
        # # # data[data>90]   = 180. - data[data>90]
        diffdata= diffdata[np.logical_not(mask)]
        dbin    = 10.
        bins    = np.arange(min(diffdata), max(diffdata) + dbin, dbin)
        
        weights = np.ones_like(diffdata)/float(diffdata.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(diffdata)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        # return p
        p.patches[3].set_color('r')
        p.patches[4].set_color('r')
        p.patches[5].set_color('r')
        p.patches[6].set_color('k')
        p.patches[7].set_color('k')
        p.patches[8].set_color('k')
        
        # # # plt.hist(data, bins=bins, weights = weights)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        good_per= float(diffdata[diffdata<30.].size)/float(diffdata.size)
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(diffdata.mean(), diffdata.std(), good_per*100.) + '%', fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()

        if showfig:
            plt.show()
        return
    
    def plot_hti_diff_psi_umvslm(self, gindex, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, lon_plt=[], lat_plt=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        data        = grp['psi2_%d' %gindex].value
        mask        = grp['mask'].value
        data2       = grp['psi2_%d' %(gindex+1)].value
        mask2       = grp['mask'].value
        
        LAB         = grp['labarr'].value
        # # # mask3       = grp['mask_lab'].value + LAB > 130.
        mask3       = mask2.copy()
        
        diffdata    = abs(data - data2)
        diffdata[diffdata>90.]  = 180. - diffdata[diffdata>90.]
        # return diffdata
        mdata       = ma.masked_array(diffdata, mask=mask + mask2 + mask3 )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        # # 
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')

        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=40, rotation=0)
        cb.ax.tick_params(labelsize=40)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")

        plt.suptitle(title, fontsize=20)
        ###
        if len(lon_plt) == len(lat_plt) and len(lon_plt) >0:
            xc, yc      = m(lon_plt, lat_plt)
            m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        if showfig:
            plt.show()
        
        
        ax      = plt.subplot()
        diffdata= diffdata[np.logical_not(mask+mask2+mask3)]
        dbin    = 10.
        bins    = np.arange(min(diffdata), max(diffdata) + dbin, dbin)
        
        weights = np.ones_like(diffdata)/float(diffdata.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(diffdata)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        # return p
        p.patches[3].set_color('r')
        p.patches[4].set_color('r')
        p.patches[5].set_color('r')
        p.patches[6].set_color('k')
        p.patches[7].set_color('k')
        p.patches[8].set_color('k')
        
        # # # plt.hist(data, bins=bins, weights = weights)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        per1    = float(diffdata[diffdata<30.].size)/float(diffdata.size)
        per2    = float(diffdata[(diffdata>=30.)*(diffdata<60.)].size)/float(diffdata.size)
        per3    = float(diffdata[(diffdata>=60.)].size)/float(diffdata.size)
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('0~30 = %g , 30~60 = %g, 6-~90 = %g' %(per1*100., per2*100., per3*100.), fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()

        if showfig:
            plt.show()
        return
    
    def plot_horizontal(self, depth, depthb=None, depthavg=None, dtype='avg', is_smooth=True, shpfx=None, clabel='', title='',\
            cmap='cv', projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, \
            lonplt=[], latplt=[], incat=None, plotevents=False, showfig=True, outfname=None):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        depth       - depth of the slice for plotting
        depthb      - depth of bottom grid for plotting (default: None)
        depthavg    - depth range for average, vs will be averaged for depth +/- depthavg
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the data that has been smoothed or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        =================================================================================================================
        """
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if depthb is not None:
            if depthb < depth:
                raise ValueError('depthb should be larger than depth!')
            index   = np.where((zArr >= depth)*(zArr <= depthb) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        elif depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            vs_plt      = vs3d[:, :, index]
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        mvs         = ma.masked_array(vs_plt, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr-360., self.latArr)
        # shapefname  = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        # shapefname  = '/home/leon/sediments_US/Sedimentary_Basins_of_the_United_States'
        # m.readshapefile(shapefname, 'sediments', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AK_sediments/AK_Sedimentary_Basins'
        # m.readshapefile(shapefname, 'sediments', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # if depth < 
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[4.05, 4.15, 4.25, 4.35, 4.45, 4.55, 4.65])
        # cb.set_label(clabel, fontsize=20, rotation=0)
        # cb.ax.tick_params(labelsize=15)
        
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)
        cb.set_alpha(1)
        cb.draw_all()
        #
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'go', lw = 3)
        ############################################################
        if plotevents or incat is not None:
            evlons  = np.array([])
            evlats  = np.array([])
            values  = np.array([])
            valuetype = 'depth'
            if incat is None:
                print 'Loading catalog'
                cat     = obspy.read_events('alaska_events.xml')
                print 'Catalog loaded!'
            else:
                cat     = incat
            for event in cat:
                event_id    = event.resource_id.id.split('=')[-1]
                porigin     = event.preferred_origin()
                pmag        = event.preferred_magnitude()
                magnitude   = pmag.mag
                Mtype       = pmag.magnitude_type
                otime       = porigin.time
                try:
                    evlo        = porigin.longitude
                    evla        = porigin.latitude
                    evdp        = porigin.depth/1000.
                except:
                    continue
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla);
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            ind             = (values >= depth - 5.)*(values<=depth+5.)
            x, y            = m(evlons[ind], evlats[ind])
            m.plot(x, y, 'o', mfc='white', mec='k', ms=3, alpha=0.5)
        # # # 
        # # # if vmax==None and vmin==None:
        # # #     vmax        = values.max()
        # # #     vmin        = values.min()
        # # # if gcmt:
        # # #     for i in xrange(len(focmecs)):
        # # #         value   = values[i]
        # # #         rgbcolor= cmap( (value-vmin)/(vmax-vmin) )
        # # #         b       = beach(focmecs[i], xy=(x[i], y[i]), width=100000, linewidth=1, facecolor=rgbcolor)
        # # #         b.set_zorder(10)
        # # #         ax.add_collection(b)
        # # #         # ax.annotate(str(i), (x[i]+50000, y[i]+50000))
        # # #     im          = m.scatter(x, y, marker='o', s=1, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        # # #     cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # # #     cb.set_label(valuetype, fontsize=20)
        # # # else:
        # # #     if values.size!=0:
        # # #         im      = m.scatter(x, y, marker='o', s=300, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        # # #         cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        # # #     else:
        # # #         m.plot(x,y,'o')
        # # # if gcmt:
        # # #     stime       = self.events[0].origins[0].time
        # # #     etime       = self.events[-1].origins[0].time
        # # # else:
        # # #     etime       = self.events[0].origins[0].time
        # # #     stime       = self.events[-1].origins[0].time
        # # # plt.suptitle('Number of event: '+str(len(self.events))+' time range: '+str(stime)+' - '+str(etime), fontsize=20 )
        # # # if showfig:
        # # #     plt.show()

        ############################
        # slb_ctrlst      = read_slab_contour('alu_contours.in', depth=depth)
        # if len(slb_ctrlst) == 0:
        #     print 'No contour at this depth =',depth
        # else:
        #     for slbctr in slb_ctrlst:
        #         xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
        #         m.plot(xslb, yslb,  '-', lw = 5, color='black')
        #         m.plot(xslb, yslb,  '-', lw = 3, color='cyan')
        ####    
        arr             = np.loadtxt('SlabE325.dat')
        lonslb          = arr[:, 0]
        latslb          = arr[:, 1]
        depthslb        = -arr[:, 2]
        index           = (depthslb > (depth - .05))*(depthslb < (depth + .05))
        lonslb          = lonslb[index]
        latslb          = latslb[index]
        indsort         = lonslb.argsort()
        lonslb          = lonslb[indsort]
        latslb          = latslb[indsort]
        xslb, yslb      = m(lonslb, latslb)
        m.plot(xslb, yslb,  '-', lw = 5, color='black')
        m.plot(xslb, yslb,  '-', lw = 3, color='cyan')
                                                     
        #############################
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return
    
    def plot_horizontal_cross(self, depth, depthb=None, depthavg=None, dtype='avg', is_smooth=True, shpfx=None, clabel='', title='',\
            cmap='cv', projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, \
            lonplt=[], latplt=[], incat=None, plotevents=False, showfig=True, outfname=None):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        depth       - depth of the slice for plotting
        depthb      - depth of bottom grid for plotting (default: None)
        depthavg    - depth range for average, vs will be averaged for depth +/- depthavg
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the data that has been smoothed or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        =================================================================================================================
        """
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if depthb is not None:
            if depthb < depth:
                raise ValueError('depthb should be larger than depth!')
            index   = np.where((zArr >= depth)*(zArr <= depthb) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        elif depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            vs_plt      = vs3d[:, :, index]
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        mvs         = ma.masked_array(vs_plt, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr-360., self.latArr)
        # shapefname  = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        # shapefname  = '/home/leon/sediments_US/Sedimentary_Basins_of_the_United_States'
        # m.readshapefile(shapefname, 'sediments', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AK_sediments/AK_Sedimentary_Basins'
        # m.readshapefile(shapefname, 'sediments', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # if depth < 
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[4.05, 4.15, 4.25, 4.35, 4.45, 4.55, 4.65])
        # cb.set_label(clabel, fontsize=20, rotation=0)
        # cb.ax.tick_params(labelsize=15)
        
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)
        cb.set_alpha(1)
        cb.draw_all()
        #
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'go', lw = 3)
        ############################################################
        if plotevents or incat is not None:
            evlons  = np.array([])
            evlats  = np.array([])
            values  = np.array([])
            valuetype = 'depth'
            if incat is None:
                print 'Loading catalog'
                cat     = obspy.read_events('alaska_events.xml')
                print 'Catalog loaded!'
            else:
                cat     = incat
            for event in cat:
                event_id    = event.resource_id.id.split('=')[-1]
                porigin     = event.preferred_origin()
                pmag        = event.preferred_magnitude()
                magnitude   = pmag.mag
                Mtype       = pmag.magnitude_type
                otime       = porigin.time
                try:
                    evlo        = porigin.longitude
                    evla        = porigin.latitude
                    evdp        = porigin.depth/1000.
                except:
                    continue
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla);
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            ind             = (values >= depth - 5.)*(values<=depth+5.)
            x, y            = m(evlons[ind], evlats[ind])
            m.plot(x, y, 'o', mfc='yellow', mec='k', ms=6, alpha=1.)
            # m.plot(x, y, 'o', mfc='white', mec='k', ms=3, alpha=0.5)
        # # # 
        # # # if vmax==None and vmin==None:
        # # #     vmax        = values.max()
        # # #     vmin        = values.min()
        # # # if gcmt:
        # # #     for i in xrange(len(focmecs)):
        # # #         value   = values[i]
        # # #         rgbcolor= cmap( (value-vmin)/(vmax-vmin) )
        # # #         b       = beach(focmecs[i], xy=(x[i], y[i]), width=100000, linewidth=1, facecolor=rgbcolor)
        # # #         b.set_zorder(10)
        # # #         ax.add_collection(b)
        # # #         # ax.annotate(str(i), (x[i]+50000, y[i]+50000))
        # # #     im          = m.scatter(x, y, marker='o', s=1, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        # # #     cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # # #     cb.set_label(valuetype, fontsize=20)
        # # # else:
        # # #     if values.size!=0:
        # # #         im      = m.scatter(x, y, marker='o', s=300, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        # # #         cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        # # #     else:
        # # #         m.plot(x,y,'o')
        # # # if gcmt:
        # # #     stime       = self.events[0].origins[0].time
        # # #     etime       = self.events[-1].origins[0].time
        # # # else:
        # # #     etime       = self.events[0].origins[0].time
        # # #     stime       = self.events[-1].origins[0].time
        # # # plt.suptitle('Number of event: '+str(len(self.events))+' time range: '+str(stime)+' - '+str(etime), fontsize=20 )
        # # # if showfig:
        # # #     plt.show()
        #########################################################################

        
        ###
        # xc, yc      = m(np.array([-146, -142]), np.array([59, 64]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='yellow')
        # 
        # xc, yc      = m(np.array([-146, -159]), np.array([59, 62]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='yellow')
        
        # xc, yc      = m(np.array([-150, -150]), np.array([58, 70]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='yellow')
        
        # xc, yc      = m(np.array([-150, -159]), np.array([58.5, 60.5]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='yellow')
        # 
        # xc, yc      = m(np.array([-149, -140]), np.array([59, 64]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='yellow')
        # 
        # xc, yc      = m(np.array([-145, -138]), np.array([59, 64]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='yellow')
        # 
        # xc, yc      = m(np.array([-160, -136]), np.array([60, 60]))
        # g               = Geod(ellps='WGS84')
        # az, baz, dist   = g.inv(lon1, lat1, lon2, lat2)
        # dist            = dist/1000.
        # d               = dist/float(int(dist/d))
        # Nd              = int(dist/d)
        # lonlats         = g.npts(lon1, lat1, lon2, lat2, npts=Nd-1)
        # lonlats         = [(lon1, lat1)] + lonlats
        # lonlats.append((lon2, lat2))
        # xc, yc      = m(np.array([-153., -153.]), np.array([65., 68.]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='white')
        
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='yellow')
        ############################
        # slb_ctrlst      = read_slab_contour('alu_contours.in', depth=depth)
        # if len(slb_ctrlst) == 0:
        #     print 'No contour at this depth =',depth
        # else:
        #     for slbctr in slb_ctrlst:
        #         xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
        #         m.plot(xslb, yslb,  '-', lw = 5, color='black')
        #         m.plot(xslb, yslb,  '-', lw = 3, color='cyan')
        #########################
        # arr             = np.loadtxt('SlabE325.dat')
        # lonslb          = arr[:, 0]
        # latslb          = arr[:, 1]
        # depthslb        = -arr[:, 2]
        # index           = (depthslb > (depth - .05))*(depthslb < (depth + .05))
        # lonslb          = lonslb[index]
        # latslb          = latslb[index]
        # indsort         = lonslb.argsort()
        # lonslb          = lonslb[indsort]
        # latslb          = latslb[indsort]
        # xslb, yslb      = m(lonslb, latslb)
        # m.plot(xslb, yslb,  '-', lw = 5, color='black')
        # m.plot(xslb, yslb,  '-', lw = 3, color='cyan')
        #############################
        # yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        # yatlons             = yakutat_slb_dat[:, 0]
        # yatlats             = yakutat_slb_dat[:, 1]
        # xyat, yyat          = m(yatlons, yatlats)
        # m.plot(xyat, yyat, lw = 5, color='black')
        # m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        #
        # print 'plotting data from '+dataid
        # # cb.solids.set_rasterized(True)
        # cb.solids.set_edgecolor("face")
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return
    
    def plot_horizontal_zoomin(self, depth, depthb=None, depthavg=None, dtype='avg', is_smooth=True, shpfx=None, clabel='', title='',\
            cmap='cv', projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, \
            lonplt=[], latplt=[], incat=None, plotevents=False, showfig=True, outfname=None):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        depth       - depth of the slice for plotting
        depthb      - depth of bottom grid for plotting (default: None)
        depthavg    - depth range for average, vs will be averaged for depth +/- depthavg
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the data that has been smoothed or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        =================================================================================================================
        """
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if depthb is not None:
            if depthb < depth:
                raise ValueError('depthb should be larger than depth!')
            index   = np.where((zArr >= depth)*(zArr <= depthb) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        elif depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            vs_plt      = vs3d[:, :, index]
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        mvs         = ma.masked_array(vs_plt, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap_3(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr-360., self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # if depth < 
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[4.05, 4.15, 4.25, 4.35, 4.45, 4.55, 4.65])
        # cb.set_label(clabel, fontsize=20, rotation=0)
        # cb.ax.tick_params(labelsize=15)
        
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)
        cb.set_alpha(1)
        cb.draw_all()
        #
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'go', lw = 3)
        ############################################################
        if plotevents or incat is not None:
            evlons  = np.array([])
            evlats  = np.array([])
            values  = np.array([])
            valuetype = 'depth'
            if incat is None:
                print 'Loading catalog'
                cat     = obspy.read_events('alaska_events.xml')
                print 'Catalog loaded!'
            else:
                cat     = incat
            for event in cat:
                event_id    = event.resource_id.id.split('=')[-1]
                porigin     = event.preferred_origin()
                pmag        = event.preferred_magnitude()
                magnitude   = pmag.mag
                Mtype       = pmag.magnitude_type
                otime       = porigin.time
                try:
                    evlo        = porigin.longitude
                    evla        = porigin.latitude
                    evdp        = porigin.depth/1000.
                except:
                    continue
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla);
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            ind             = (values >= depth - 5.)*(values<=depth+5.)
            x, y            = m(evlons[ind], evlats[ind])
            m.plot(x, y, 'o', mfc='yellow', mec='k', ms=6, alpha=1.)
        # # # 
        # # # if vmax==None and vmin==None:
        # # #     vmax        = values.max()
        # # #     vmin        = values.min()
        # # # if gcmt:
        # # #     for i in xrange(len(focmecs)):
        # # #         value   = values[i]
        # # #         rgbcolor= cmap( (value-vmin)/(vmax-vmin) )
        # # #         b       = beach(focmecs[i], xy=(x[i], y[i]), width=100000, linewidth=1, facecolor=rgbcolor)
        # # #         b.set_zorder(10)
        # # #         ax.add_collection(b)
        # # #         # ax.annotate(str(i), (x[i]+50000, y[i]+50000))
        # # #     im          = m.scatter(x, y, marker='o', s=1, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        # # #     cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # # #     cb.set_label(valuetype, fontsize=20)
        # # # else:
        # # #     if values.size!=0:
        # # #         im      = m.scatter(x, y, marker='o', s=300, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        # # #         cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        # # #     else:
        # # #         m.plot(x,y,'o')
        # # # if gcmt:
        # # #     stime       = self.events[0].origins[0].time
        # # #     etime       = self.events[-1].origins[0].time
        # # # else:
        # # #     etime       = self.events[0].origins[0].time
        # # #     stime       = self.events[-1].origins[0].time
        # # # plt.suptitle('Number of event: '+str(len(self.events))+' time range: '+str(stime)+' - '+str(etime), fontsize=20 )
        # # # if showfig:
        # # #     plt.show()

        ############################
        # xc, yc      = m(np.array([-146, -142]), np.array([59, 64]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='yellow')
        # 
        # xc, yc      = m(np.array([-146, -159]), np.array([59, 62]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='green')
        # 
        # # # xc, yc      = m(np.array([-150, -150]), np.array([58, 70]))
        # # # m.plot(xc, yc,'k', lw = 5, color='black')
        # # # m.plot(xc, yc,'k', lw = 3, color='yellow')
        # 
        # xc, yc      = m(np.array([-150, -159]), np.array([58.5, 60.5]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='green')
        # 
        # xc, yc      = m(np.array([-149, -140]), np.array([59, 64]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='green')
        # 
        # xc, yc      = m(np.array([-145, -138]), np.array([59, 64]))
        # m.plot(xc, yc,'k', lw = 5, color='black')
        # m.plot(xc, yc,'k', lw = 3, color='green')
        
        
        ###
        xc, yc      = m(np.array([-149, -160.]), np.array([58, 61.2]))
        m.plot(xc, yc,'k', lw = 5, color='black')
        m.plot(xc, yc,'k', lw = 3, color='green')
        
        xc, yc      = m(np.array([-146, -157.5]), np.array([59, 62]))
        m.plot(xc, yc,'k', lw = 5, color='black')
        m.plot(xc, yc,'k', lw = 3, color='green')
        
        xc, yc      = m(np.array([-145, -137.3]), np.array([59, 64.3]))
        m.plot(xc, yc,'k', lw = 5, color='black')
        m.plot(xc, yc,'k', lw = 3, color='green')
        
        xc, yc      = m(np.array([-149., -140.5]), np.array([59, 64]))
        m.plot(xc, yc,'k', lw = 5, color='black')
        m.plot(xc, yc,'k', lw = 3, color='green')
        
        xc, yc      = m(np.array([-156., -143.]), np.array([64, 60]))
        m.plot(xc, yc,'k', lw = 5, color='black')
        m.plot(xc, yc,'k', lw = 3, color='green')
        
        # # # xc, yc      = m(np.array([-153., -153.]), np.array([65., 68.]))
        # # # m.plot(xc, yc,'k', lw = 5, color='black')
        # # # m.plot(xc, yc,'k', lw = 3, color='white')
        
        ####    
        arr             = np.loadtxt('SlabE325.dat')
        lonslb          = arr[:, 0]
        latslb          = arr[:, 1]
        depthslb        = -arr[:, 2]
        index           = (depthslb > (depth - .05))*(depthslb < (depth + .05))
        lonslb          = lonslb[index]
        latslb          = latslb[index]
        indsort         = lonslb.argsort()
        lonslb          = lonslb[indsort]
        latslb          = latslb[indsort]
        xslb, yslb      = m(lonslb, latslb)
        m.plot(xslb, yslb,  '-', lw = 7, color='black')
        m.plot(xslb, yslb,  '-', lw = 5, color='cyan')
        ###
        slb_ctrlst      = read_slab_contour('alu_contours.in', depth=depth)
        # slb_ctrlst      = read_slab_contour('/home/leon/Slab2Distribute_Mar2018/Slab2_CONTOURS/alu_slab2_dep_02.23.18_contours.in', depth=depth)
        if len(slb_ctrlst) == 0:
            print 'No contour at this depth =',depth
        else:
            for slbctr in slb_ctrlst:
                xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
                # m.plot(xslb, yslb,  '', lw = 5, color='black')
                factor      = 20
                # N           = xslb.size
                # xslb        = xslb[0:N:factor]
                # yslb        = yslb[0:N:factor]
                m.plot(xslb, yslb,  '--', lw = 5, color='red', ms=8, markeredgecolor='k')
                                                     
        #############################
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return
    
    def plot_horizontal_zoomin_vsh(self, depth, depthb=None, depthavg=None, dtype='avg', is_smooth=True, shpfx=None, clabel='', title='',\
            cmap='cv', projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, \
            lonplt=[], latplt=[], incat=None, plotevents=False, showfig=True, outfname=None):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        depth       - depth of the slice for plotting
        depthb      - depth of bottom grid for plotting (default: None)
        depthavg    - depth range for average, vs will be averaged for depth +/- depthavg
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the data that has been smoothed or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        =================================================================================================================
        """
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if depthb is not None:
            if depthb < depth:
                raise ValueError('depthb should be larger than depth!')
            index   = np.where((zArr >= depth)*(zArr <= depthb) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        elif depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            vs_plt      = vs3d[:, :, index]
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        mvs         = ma.masked_array(vs_plt, mask=mask )
        ###
        dset = invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190327_no_ocsi_crust_15_mantle_10_vti_gr.h5')
        data2, data_smooth2\
                    = dset.get_smooth_paraval(pindex=-1, dtype='avg', itype='vti', \
                        sigma=1, gsigma = 50., isthk=False, do_interp=True, depth = 5., depthavg = 0.)
        # un2, un_smooth2\
        #             = dset.get_smooth_paraval(pindex=-1, dtype='std', itype='vti', \
        #                 sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
        # mask2       = dset.attrs['mask_inv']
        # data_smooth[np.logical_not(mask2)]  = data_smooth2[np.logical_not(mask2)]
        # un[np.logical_not(mask2)]           = un2[np.logical_not(mask2)]
        hv_ratio    = (1. + data_smooth2/200.)/(1 - data_smooth2/200.)
        mvs         *= hv_ratio
        
        ###
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap_3(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr-360., self.latArr)
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # if depth < 
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[4.05, 4.15, 4.25, 4.35, 4.45, 4.55, 4.65])
        # cb.set_label(clabel, fontsize=20, rotation=0)
        # cb.ax.tick_params(labelsize=15)
        
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)
        cb.set_alpha(1)
        cb.draw_all()
        #
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'go', lw = 3)

        ####    
        arr             = np.loadtxt('SlabE325.dat')
        lonslb          = arr[:, 0]
        latslb          = arr[:, 1]
        depthslb        = -arr[:, 2]
        index           = (depthslb > (depth - .05))*(depthslb < (depth + .05))
        lonslb          = lonslb[index]
        latslb          = latslb[index]
        indsort         = lonslb.argsort()
        lonslb          = lonslb[indsort]
        latslb          = latslb[indsort]
        xslb, yslb      = m(lonslb, latslb)
        m.plot(xslb, yslb,  '-', lw = 7, color='black')
        m.plot(xslb, yslb,  '-', lw = 5, color='cyan')
        ###
        slb_ctrlst      = read_slab_contour('alu_contours.in', depth=depth)
        if len(slb_ctrlst) == 0:
            print 'No contour at this depth =',depth
        else:
            for slbctr in slb_ctrlst:
                xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
                # m.plot(xslb, yslb,  '', lw = 5, color='black')
                factor      = 20
                N           = xslb.size
                xslb        = xslb[0:N:factor]
                yslb        = yslb[0:N:factor]
                m.plot(xslb, yslb,  'o', lw = 1, color='red', ms=8, markeredgecolor='k')
                                                     
        #############################
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return
    
    def plot_horizontal_discontinuity(self, depthrange, distype='moho', dtype='avg', is_smooth=True, shpfx=None, clabel='', title='',\
            cmap='cv', projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, \
            lonplt=[], latplt=[], showfig=True):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        depthrange  - depth range for average
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the data that has been smoothed or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        =================================================================================================================
        """
        is_interp       = self.attrs['is_interp']
        if is_interp:
            topoArr     = self['topo_interp'].value
        else:
            topoArr     = self['topo'].value
        if distype is 'moho':
            if is_smooth:
                disArr  = self[dtype+'_paraval/12_smooth'].value + self[dtype+'_paraval/11_smooth'].value - topoArr
            else:
                disArr  = self[dtype+'_paraval/12_org'].value + self[dtype+'_paraval/11_org'].value - topoArr
        elif distype is 'sedi':
            if is_smooth:
                disArr  = self[dtype+'_paraval/11_smooth'].value - topoArr
            else:
                disArr  = self[dtype+'_paraval/11_org'].value - topoArr
        else:
            raise ValueError('Unexpected type of discontinuity:'+distype)
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if depthrange < 0.:
            depth0  = disArr + depthrange
            depth1  = disArr.copy()
        else:
            depth0  = disArr 
            depth1  = disArr + depthrange
        vs_plt      = _get_vs_2d(z0=depth0, z1=depth1, zArr=zArr, vs_3d=vs3d)
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        mvs         = ma.masked_array(vs_plt, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr, self.latArr)
        # shapefname  = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=30)
        cb.set_alpha(1)
        cb.draw_all()
        #
        # xc, yc      = m(np.array([-150, -170]), np.array([57, 64]))
        # m.plot(xc, yc,'k', lw = 3)
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'ko', lw = 3)
        #############################
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
            
        cb.solids.set_edgecolor("face")
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        return
    
    def plot_vertical_rel(self, lon1, lat1, lon2, lat2, maxdepth, vs_mantle=4.4, plottype = 0, d = 10., dtype='avg', is_smooth=True,\
                      clabel='', cmap='cv', vmin1=3.0, vmax1=4.2, vmin2=-10., vmax2=10., incat=None, dist_thresh=20., showfig=True):
        is_interp   = self.attrs['is_interp']
        if is_interp:
            topoArr = self['topo_interp'].value
        else:
            topoArr = self['topo'].value
        if is_smooth:
            mohoArr = self[dtype+'_paraval/12_smooth'].value + self[dtype+'_paraval/11_smooth'].value - topoArr
        else:
            mohoArr = self[dtype+'_paraval/12_org'].value + self[dtype+'_paraval/11_org'].value - topoArr
        if lon1 == lon2 and lat1 == lat2:
            raise ValueError('The start and end points are the same!')
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        ind_z       = np.where(zArr <= maxdepth )[0]
        zplot       = zArr[ind_z]
        ###
        # if lon1 == lon2 or lat1 == lat2:
        #     if lon1 == lon2:    
        #         ind_lon = np.where(self.lons == lon1)[0]
        #         ind_lat = np.where((self.lats<=max(lat1, lat2))*(self.lats>=min(lat1, lat2)))[0]
        #         # data    = np.zeros((len(ind_lat), ind_z.size))
        #     else:
        #         ind_lon = np.where((self.lons<=max(lon1, lon2))*(self.lons>=min(lon1, lon2)))[0]
        #         ind_lat = np.where(self.lats == lat1)[0]
        #         # data    = np.zeros((len(ind_lon), ind_z.size))
        #     data_temp   = vs3d[ind_lat, ind_lon, :]
        #     data        = data_temp[:, ind_z]
        #     if lon1 == lon2:
        #         xplot       = self.lats[ind_lat]
        #         xlabel      = 'latitude (deg)'
        #     if lat1 == lat2:
        #         xplot       = self.lons[ind_lon]
        #         xlabel      = 'longitude (deg)'
        #     # 
        #     topo1d          = topoArr[ind_lat, ind_lon]
        #     moho1d          = mohoArr[ind_lat, ind_lon]
        #     #
        #     data_moho       = data.copy()
        #     mask_moho       = np.ones(data.shape, dtype=bool)
        #     data_mantle     = data.copy()
        #     mask_mantle     = np.ones(data.shape, dtype=bool)
        #     for ix in range(data.shape[0]):
        #         ind_moho    = zplot <= moho1d[ix]
        #         ind_mantle  = np.logical_not(ind_moho)
        #         mask_moho[ix, ind_moho] \
        #                     = False
        #         mask_mantle[ix, ind_mantle] \
        #                     = False
        #         data_mantle[ix, :] \
        #                     = (data_mantle[ix, :] - vs_mantle)/vs_mantle*100.
        # else:
        g               = Geod(ellps='WGS84')
        az, baz, dist   = g.inv(lon1, lat1, lon2, lat2)
        dist            = dist/1000.
        d               = dist/float(int(dist/d))
        Nd              = int(dist/d)
        lonlats         = g.npts(lon1, lat1, lon2, lat2, npts=Nd-1)
        lonlats         = [(lon1, lat1)] + lonlats
        lonlats.append((lon2, lat2))
        data            = np.zeros((len(lonlats), ind_z.size))
        mask1d          = np.ones((len(lonlats), ind_z.size), dtype=bool)
        L               = self.lonArr.size
        vlonArr         = self.lonArr.reshape(L)
        vlatArr         = self.latArr.reshape(L)
        ind_data        = 0
        plons           = np.zeros(len(lonlats))
        plats           = np.zeros(len(lonlats))
        topo1d          = np.zeros(len(lonlats))
        moho1d          = np.zeros(len(lonlats))
        for lon,lat in lonlats:
            if lon < 0.:
                lon     += 360.
            clonArr         = np.ones(L, dtype=float)*lon
            clatArr         = np.ones(L, dtype=float)*lat
            az, baz, dist   = g.inv(clonArr, clatArr, vlonArr, vlatArr)
            ind_min         = dist.argmin()
            ind_lat         = int(np.floor(ind_min/self.Nlon))
            ind_lon         = ind_min - self.Nlon*ind_lat
            azmin, bazmin, distmin = g.inv(lon, lat, self.lons[ind_lon], self.lats[ind_lat])
            if distmin != dist[ind_min]:
                raise ValueError('DEBUG!')
            data[ind_data, :]   \
                            = vs3d[ind_lat, ind_lon, ind_z]
            plons[ind_data] = lon
            plats[ind_data] = lat
            topo1d[ind_data]= topoArr[ind_lat, ind_lon]
            moho1d[ind_data]= mohoArr[ind_lat, ind_lon]
            mask1d[ind_data, :]\
                            = mask[ind_lat, ind_lon]
            ind_data        += 1
        data_moho           = data.copy()
        mask_moho           = np.ones(data.shape, dtype=bool)
        data_mantle         = data.copy()
        mask_mantle         = np.ones(data.shape, dtype=bool)
        for ix in range(data.shape[0]):
            ind_moho        = zplot <= moho1d[ix]
            ind_mantle      = np.logical_not(ind_moho)
            mask_moho[ix, ind_moho] \
                            = False
            mask_mantle[ix, ind_mantle] \
                            = False
            data_mantle[ix, :] \
                            = (data_mantle[ix, :] - vs_mantle)/vs_mantle*100.
        mask_moho           += mask1d
        mask_mantle         += mask1d
        if plottype == 0:
            xplot   = plons
            xlabel  = 'longitude (deg)'
        else:
            xplot   = plats
            xlabel  = 'latitude (deg)'
        ########################
        cmap1           = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        cmap2           = pycpt.load.gmtColormap('./cv.cpt')
        f, (ax1, ax2)   = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios':[1,4]})
        topo1d[topo1d<0.]   \
                        = 0.
        ax1.plot(xplot, topo1d*1000., 'k', lw=3)
        ax1.fill_between(xplot, 0, topo1d*1000., facecolor='grey')
        ax1.set_ylabel('Elevation (m)', fontsize=20)
        ax1.set_ylim(0, topo1d.max()*1000.+10.)
        mdata_moho      = ma.masked_array(data_moho, mask=mask_moho )
        mdata_mantle    = ma.masked_array(data_mantle, mask=mask_mantle )
        m1              = ax2.pcolormesh(xplot, zplot, mdata_mantle.T, shading='gouraud', vmax=vmax2, vmin=vmin2, cmap=cmap2)
        cb1             = f.colorbar(m1, orientation='horizontal', fraction=0.05)
        cb1.set_label('Mantle Vsv perturbation relative to '+str(vs_mantle)+' km/s (%)', fontsize=20)
        cb1.ax.tick_params(labelsize=20) 
        m2              = ax2.pcolormesh(xplot, zplot, mdata_moho.T, shading='gouraud', vmax=vmax1, vmin=vmin1, cmap=cmap2)
        cb2             = f.colorbar(m2, orientation='horizontal', fraction=0.06)
        cb2.set_label('Crustal Vsv (km/s)', fontsize=20)
        cb2.ax.tick_params(labelsize=20) 
        #
        ax2.plot(xplot, moho1d, 'r', lw=3)
        #
        ax2.set_xlabel(xlabel, fontsize=20)
        ax2.set_ylabel('Depth (km)', fontsize=20)
        f.subplots_adjust(hspace=0)
        ############################################################
        lonlats_arr \
                = np.asarray(lonlats)
        lons_arr= lonlats_arr[:, 0]
        lats_arr= lonlats_arr[:, 1]
        evlons  = np.array([])
        evlats  = np.array([])
        values  = np.array([])
        valuetype = 'depth'
        if incat != -1:
            if incat is None:
                print 'Loading catalog'
                cat     = obspy.read_events('alaska_events.xml')
                print 'Catalog loaded!'
            else:
                cat     = incat
            Nevent      = 0
            for event in cat:
                event_id    = event.resource_id.id.split('=')[-1]
                porigin     = event.preferred_origin()
                pmag        = event.preferred_magnitude()
                magnitude   = pmag.mag
                Mtype       = pmag.magnitude_type
                otime       = porigin.time
                try:
                    evlo        = porigin.longitude
                    evla        = porigin.latitude
                    evdp        = porigin.depth/1000.
                except:
                    continue
                az, baz, dist \
                                = g.inv(lons_arr, lats_arr, np.ones(lons_arr.size)*evlo, np.ones(lons_arr.size)*evla)
                # print dist.min()/1000.
                if evlo < 0.:
                    evlo        += 360.
                if dist.min()/1000. < dist_thresh:
                    evlons      = np.append(evlons, evlo)
                    evlats      = np.append(evlats, evla)
                    if valuetype=='depth':
                        values  = np.append(values, evdp)
                    elif valuetype=='mag':
                        values  = np.append(values, magnitude)
            # 
            # for lon,lat in lonlats:
            #     if lon < 0.:
            #         lon     += 360.
            #     dist, az, baz \
            #                 = obspy.geodetics.gps2dist_azimuth(lat, lon, evla, evlo)
            #     # az, baz, dist \
            #     #             = g.inv(lon, lat, evlo, evla)
            #     if dist/1000. < 10.:
            #         evlons      = np.append(evlons, evlo)
            #         evlats      = np.append(evlats, evla)
            #     if valuetype=='depth':
            #         values  = np.append(values, evdp)
            #     elif valuetype=='mag':
            #         values  = np.append(values, magnitude)
            #         break
            
        ####
        # arr             = np.loadtxt('SlabE325.dat')
        # # index           = np.logical_not(np.isnan(arr[:, 2]))
        # # lonslb          = arr[index, 0]
        # # latslb          = arr[index, 1]
        # # depthslb        = arr[index, 2]
        # 
        # lonslb          = arr[:, 0]
        # latslb          = arr[:, 1]
        # depthslb        = arr[:, 2]
        # L               = lonslb.size
        # ind_data        = 0
        # plons           = np.zeros(len(lonlats))
        # plats           = np.zeros(len(lonlats))
        # slbdepth        = np.zeros(len(lonlats))
        # for lon,lat in lonlats:
        #     if lon < 0.:
        #         lon     += 360.
        #     clonArr             = np.ones(L, dtype=float)*lon
        #     clatArr             = np.ones(L, dtype=float)*lat
        #     az, baz, dist       = g.inv(clonArr, clatArr, lonslb, latslb)
        #     ind_min             = dist.argmin()
        #     plons[ind_data]     = lon
        #     plats[ind_data]     = lat
        #     slbdepth[ind_data]  = -depthslb[ind_min]
        #     if lon > 222.:
        #         slbdepth[ind_data]  = 200.
        #     ind_data            += 1
        # ax2.plot(xplot, slbdepth, 'k', lw=5)
        # ax2.plot(xplot, slbdepth, 'w', lw=3)
        ####
        
        # # # for lon,lat in lonlats:
        # # #     if lon < 0.:
        # # #         lon     += 360.
        # # #     for event in cat:
        # # #         event_id    = event.resource_id.id.split('=')[-1]
        # # #         porigin     = event.preferred_origin()
        # # #         pmag        = event.preferred_magnitude()
        # # #         magnitude   = pmag.mag
        # # #         Mtype       = pmag.magnitude_type
        # # #         otime       = porigin.time
        # # #         try:
        # # #             evlo        = porigin.longitude
        # # #             evla        = porigin.latitude
        # # #             evdp        = porigin.depth/1000.
        # # #         except:
        # # #             continue
        # # #         if evlo < 0.:
        # # #             evlo    += 360.
        # # #         if abs(evlo-lon)<0.1 and abs(evla-lat)<0.1:
        # # #             evlons      = np.append(evlons, evlo)
        # # #             evlats      = np.append(evlats, evla)
        # # #             if valuetype=='depth':
        # # #                 values  = np.append(values, evdp)
        # # #             elif valuetype=='mag':
        # # #                 values  = np.append(values, magnitude)
        # # print evlons.size
        if plottype == 0:
            # evlons  -=
            ax2.plot(evlons, values, 'o', mfc='white', mec='k', ms=5, alpha=0.8)
        else:
            ax2.plot(evlats, values, 'o', mfc='white', mec='k', ms=5, alpha=0.8)
            
        #########################################################################
        ax1.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylim([zplot[0], zplot[-1]])
        ax2.set_xlim([xplot[0], xplot[-1]])
        plt.gca().invert_yaxis()
        if showfig:
            plt.show()
        return
    
    def plot_vertical_rel_2(self, lon1, lat1, lon2, lat2, maxdepth, vs_mantle=4.4, plottype = 0, d = 10., dtype='avg', is_smooth=True,\
                      clabel='', cmap='cv', vmin1=3.0, vmax1=4.2, vmin2=4.1, vmax2=4.6, incat=None, dist_thresh=20., showfig=True):
        is_interp   = self.attrs['is_interp']
        if is_interp:
            topoArr = self['topo_interp'].value
        else:
            topoArr = self['topo'].value
        if is_smooth:
            mohoArr = self[dtype+'_paraval/12_smooth'].value + self[dtype+'_paraval/11_smooth'].value - topoArr
        else:
            mohoArr = self[dtype+'_paraval/12_org'].value + self[dtype+'_paraval/11_org'].value - topoArr
        if lon1 == lon2 and lat1 == lat2:
            raise ValueError('The start and end points are the same!')
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        ind_z       = np.where(zArr <= maxdepth )[0]
        zplot       = zArr[ind_z]
        g               = Geod(ellps='WGS84')
        az, baz, dist   = g.inv(lon1, lat1, lon2, lat2)
        dist            = dist/1000.
        d               = dist/float(int(dist/d))
        Nd              = int(dist/d)
        lonlats         = g.npts(lon1, lat1, lon2, lat2, npts=Nd-1)
        lonlats         = [(lon1, lat1)] + lonlats
        lonlats.append((lon2, lat2))
        data            = np.zeros((len(lonlats), ind_z.size))
        mask1d          = np.ones((len(lonlats), ind_z.size), dtype=bool)
        L               = self.lonArr.size
        vlonArr         = self.lonArr.reshape(L)
        vlatArr         = self.latArr.reshape(L)
        ind_data        = 0
        plons           = np.zeros(len(lonlats))
        plats           = np.zeros(len(lonlats))
        topo1d          = np.zeros(len(lonlats))
        moho1d          = np.zeros(len(lonlats))
        for lon,lat in lonlats:
            if lon < 0.:
                lon     += 360.
            clonArr         = np.ones(L, dtype=float)*lon
            clatArr         = np.ones(L, dtype=float)*lat
            az, baz, dist   = g.inv(clonArr, clatArr, vlonArr, vlatArr)
            ind_min         = dist.argmin()
            ind_lat         = int(np.floor(ind_min/self.Nlon))
            ind_lon         = ind_min - self.Nlon*ind_lat
            azmin, bazmin, distmin = g.inv(lon, lat, self.lons[ind_lon], self.lats[ind_lat])
            if distmin != dist[ind_min]:
                raise ValueError('DEBUG!')
            data[ind_data, :]   \
                            = vs3d[ind_lat, ind_lon, ind_z]
            plons[ind_data] = lon
            plats[ind_data] = lat
            topo1d[ind_data]= topoArr[ind_lat, ind_lon]
            moho1d[ind_data]= mohoArr[ind_lat, ind_lon]
            mask1d[ind_data, :]\
                            = mask[ind_lat, ind_lon]
            ind_data        += 1
        data_moho           = data.copy()
        mask_moho           = np.ones(data.shape, dtype=bool)
        data_mantle         = data.copy()
        mask_mantle         = np.ones(data.shape, dtype=bool)
        for ix in range(data.shape[0]):
            ind_moho        = zplot <= moho1d[ix]
            ind_mantle      = np.logical_not(ind_moho)
            mask_moho[ix, ind_moho] \
                            = False
            mask_mantle[ix, ind_mantle] \
                            = False
            data_mantle[ix, :] \
                            = (data_mantle[ix, :] - vs_mantle)/vs_mantle*100.
        # # # for ix in range(data.shape[0]):
        # # #     ind_moho        = zplot <= moho1d[ix]
        # # #     ind_mantle      = np.logical_not(ind_moho)
        # # #     mask_moho[ix, ind_moho] \
        # # #                     = False
        # # #     mask_mantle[ix, ind_mantle] \
        # # #                     = False
        # # #     data_mantle[ix, :] \
        # # #                     = (data_mantle[ix, :] - vs_mantle)/vs_mantle*100.
        mask_moho           += mask1d
        mask_mantle         += mask1d
        if plottype == 0:
            xplot   = plons
            xlabel  = 'longitude (deg)'
        else:
            xplot   = plats
            xlabel  = 'latitude (deg)'
        ########################
        cmap1           = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        cmap2           = pycpt.load.gmtColormap('./cv.cpt')
        f, (ax1, ax2)   = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios':[1,4]})
        topo1d[topo1d<0.]   \
                        = 0.
        ax1.plot(xplot, topo1d*1000., 'k', lw=3)
        ax1.fill_between(xplot, 0, topo1d*1000., facecolor='grey')
        ax1.set_ylabel('Elevation (m)', fontsize=20)
        ax1.set_ylim(0, topo1d.max()*1000.+10.)
        mdata_moho      = ma.masked_array(data_moho, mask=mask_moho )
        mdata_mantle    = ma.masked_array(data_mantle, mask=mask_mantle )
        m1              = ax2.pcolormesh(xplot, zplot, mdata_mantle.T, shading='gouraud', vmax=vmax2, vmin=vmin2, cmap=cmap2)
        cb1             = f.colorbar(m1, orientation='horizontal', fraction=0.05)
        cb1.set_label('Mantle Vsv (km/s)', fontsize=20)
        cb1.ax.tick_params(labelsize=20) 
        m2              = ax2.pcolormesh(xplot, zplot, mdata_moho.T, shading='gouraud', vmax=vmax1, vmin=vmin1, cmap=cmap2)
        cb2             = f.colorbar(m2, orientation='horizontal', fraction=0.06)
        cb2.set_label('Crustal Vsv (km/s)', fontsize=20)
        cb2.ax.tick_params(labelsize=20) 
        #
        ax2.plot(xplot, moho1d, 'r', lw=3)
        #
        ax2.set_xlabel(xlabel, fontsize=20)
        ax2.set_ylabel('Depth (km)', fontsize=20)
        f.subplots_adjust(hspace=0)
        ############################################################
        lonlats_arr \
                = np.asarray(lonlats)
        lons_arr= lonlats_arr[:, 0]
        lats_arr= lonlats_arr[:, 1]
        evlons  = np.array([])
        evlats  = np.array([])
        values  = np.array([])
        valuetype = 'depth'
        if incat != -1:
            if incat is None:
                print 'Loading catalog'
                cat     = obspy.read_events('alaska_events.xml')
                print 'Catalog loaded!'
            else:
                cat     = incat
            Nevent      = 0
            for event in cat:
                event_id    = event.resource_id.id.split('=')[-1]
                porigin     = event.preferred_origin()
                pmag        = event.preferred_magnitude()
                magnitude   = pmag.mag
                Mtype       = pmag.magnitude_type
                otime       = porigin.time
                try:
                    evlo        = porigin.longitude
                    evla        = porigin.latitude
                    evdp        = porigin.depth/1000.
                except:
                    continue
                az, baz, dist \
                                = g.inv(lons_arr, lats_arr, np.ones(lons_arr.size)*evlo, np.ones(lons_arr.size)*evla)
                # print dist.min()/1000.
                if evlo < 0.:
                    evlo        += 360.
                if dist.min()/1000. < dist_thresh:
                    evlons      = np.append(evlons, evlo)
                    evlats      = np.append(evlats, evla)
                    if valuetype=='depth':
                        values  = np.append(values, evdp)
                    elif valuetype=='mag':
                        values  = np.append(values, magnitude)
        ####
        # arr             = np.loadtxt('SlabE325.dat')
        # # index           = np.logical_not(np.isnan(arr[:, 2]))
        # # lonslb          = arr[index, 0]
        # # latslb          = arr[index, 1]
        # # depthslb        = arr[index, 2]
        # 
        # lonslb          = arr[:, 0]
        # latslb          = arr[:, 1]
        # depthslb        = arr[:, 2]
        # L               = lonslb.size
        # ind_data        = 0
        # plons           = np.zeros(len(lonlats))
        # plats           = np.zeros(len(lonlats))
        # slbdepth        = np.zeros(len(lonlats))
        # for lon,lat in lonlats:
        #     if lon < 0.:
        #         lon     += 360.
        #     clonArr             = np.ones(L, dtype=float)*lon
        #     clatArr             = np.ones(L, dtype=float)*lat
        #     az, baz, dist       = g.inv(clonArr, clatArr, lonslb, latslb)
        #     ind_min             = dist.argmin()
        #     plons[ind_data]     = lon
        #     plats[ind_data]     = lat
        #     slbdepth[ind_data]  = -depthslb[ind_min]
        #     if lon > 222.:
        #         slbdepth[ind_data]  = 200.
        #     ind_data            += 1
        # ax2.plot(xplot, slbdepth, 'k', lw=5)
        # ax2.plot(xplot, slbdepth, 'w', lw=3)
        ####
        if plottype == 0:
            # evlons  -=
            ax2.plot(evlons, values, 'o', mfc='yellow', mec='k', ms=8, alpha=1)
        else:
            ax2.plot(evlats, values, 'o', mfc='yellow', mec='k', ms=8, alpha=1)
            
        #########################################################################
        ax1.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylim([zplot[0], zplot[-1]])
        ax2.set_xlim([xplot[0], xplot[-1]])
        plt.gca().invert_yaxis()
        if showfig:
            plt.show()
        return
                    
    def plot_vertical_abs(self, lon1, lat1, lon2, lat2, maxdepth, plottype = 0, d = 10., dtype='min', is_smooth=False,\
                      clabel='', cmap='cv', vmin=None, vmax=None, showfig=True):        
        if lon1 == lon2 and lat1 == lat2:
            raise ValueError('The start and end points are the same!')
        self._get_lon_lat_arr()
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        ind_z       = np.where(zArr <= maxdepth )[0]
        zplot       = zArr[ind_z]
        if lon1 == lon2 or lat1 == lat2:
            if lon1 == lon2:    
                ind_lon = np.where(self.lons == lon1)[0]
                ind_lat = np.where((self.lats<=max(lat1, lat2))*(self.lats>=min(lat1, lat2)))[0]
                # data    = np.zeros((len(ind_lat), ind_z.size))
            else:
                ind_lon = np.where((self.lons<=max(lon1, lon2))*(self.lons>=min(lon1, lon2)))[0]
                ind_lat = np.where(self.lats == lat1)[0]
                # data    = np.zeros((len(ind_lon), ind_z.size))
            data_temp   = vs3d[ind_lat, ind_lon, :]
            data        = data_temp[:, ind_z]
            # return data, data_temp
            if lon1 == lon2:
                xplot       = self.lats[ind_lat]
                xlabel      = 'latitude (deg)'
            if lat1 == lat2:
                xplot       = self.lons[ind_lon]
                xlabel      = 'longitude (deg)'            
        else:
            g               = Geod(ellps='WGS84')
            az, baz, dist   = g.inv(lon1, lat1, lon2, lat2)
            dist            = dist/1000.
            d               = dist/float(int(dist/d))
            Nd              = int(dist/d)
            lonlats         = g.npts(lon1, lat1, lon2, lat2, npts=Nd-1)
            lonlats         = [(lon1, lat1)] + lonlats
            lonlats.append((lon2, lat2))
            data            = np.zeros((len(lonlats), ind_z.size))
            L               = self.lonArr.size
            vlonArr         = self.lonArr.reshape(L)
            vlatArr         = self.latArr.reshape(L)
            ind_data        = 0
            plons           = np.zeros(len(lonlats))
            plats           = np.zeros(len(lonlats))
            for lon,lat in lonlats:
                if lon < 0.:
                    lon     += 360.
                # if lat <
                # print lon, lat
                clonArr         = np.ones(L, dtype=float)*lon
                clatArr         = np.ones(L, dtype=float)*lat
                az, baz, dist   = g.inv(clonArr, clatArr, vlonArr, vlatArr)
                ind_min         = dist.argmin()
                ind_lat         = int(np.floor(ind_min/self.Nlon))
                ind_lon         = ind_min - self.Nlon*ind_lat
                # 
                azmin, bazmin, distmin = g.inv(lon, lat, self.lons[ind_lon], self.lats[ind_lat])
                if distmin != dist[ind_min]:
                    raise ValueError('DEBUG!')
                #
                data[ind_data, :]   \
                                = vs3d[ind_lat, ind_lon, ind_z]
                plons[ind_data] = lon
                plats[ind_data] = lat
                ind_data        += 1
            # data[0, :]          = 
            if plottype == 0:
                xplot   = plons
                xlabel  = 'longitude (deg)'
            else:
                xplot   = plats
                xlabel  = 'latitude (deg)'
                
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ax      = plt.subplot()
        plt.pcolormesh(xplot, zplot, data.T, shading='gouraud', vmax=vmax, vmin=vmin, cmap=cmap)
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel('depth (km)', fontsize=30)
        plt.gca().invert_yaxis()
        # plt.axis([self.xgrid[0], self.xgrid[-1], self.ygrid[0], self.ygrid[-1]], 'scaled')
        cb=plt.colorbar()
        cb.set_label('Vs (km/s)', fontsize=30)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        if showfig:
            plt.show()
# 

# quick and dirty functions
    def plot_miller_moho(self, vmin=20., vmax=60., clabel='Crustal thickness (km)', cmap='gist_ncar',showfig=True, projection='lambert', \
                         infname='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoho.npz'):
        inarr   = np.load(infname)['alaska_moho']
        mohoarr = []
        lonarr  = []
        latarr  = []
        for data in inarr:
            lonarr.append(data[0])
            latarr.append(data[1])
            mohoarr.append(data[2])
        lonarr  = np.array(lonarr)
        latarr  = np.array(latarr)
        mohoarr = np.array(mohoarr)
        print mohoarr.min(), mohoarr.max()
        m               = self._get_basemap(projection=projection)
        shapefname      = '/home/leon/geological_maps/qfaults'
        m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap= pycpt.load.gmtColormap(cmap)
            except:
                pass
        x, y            = m(lonarr, latarr)
        import matplotlib
        # cmap            = matplotlib.cm.get_cmap(cmap)
        # normalize       = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        # colors          = [cmap(normalize(value)) for value in mohoarr]

        im              = m.scatter(x, y, c=mohoarr, s=100, edgecolors='k', cmap=cmap, vmin=vmin, vmax=vmax)
        cb              = m.colorbar(im, location='bottom', size="3%", pad='2%')
        # cb              = plt.colorbar()
        cb.set_label(clabel, fontsize=20, rotation=0)
        cb.ax.tick_params(labelsize=15)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        ###
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
        if showfig:
            plt.show()
            
    def plot_miller_moho_finer_scatter(self, vmin=20., vmax=60., clabel='Crustal thickness (km)', cmap='gist_ncar',showfig=True, projection='lambert', \
                         infname='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoHiErrs-AlaskaMohoFineGrid.npz'):
        inarr   = np.load(infname)
        mohoarr = inarr['gridded_data_1']
        lonarr  = np.degrees(inarr['gridlons'])
        latarr  = np.degrees(inarr['gridlats'])
        print mohoarr.min(), mohoarr.max()
        m               = self._get_basemap(projection=projection)
        shapefname      = '/home/leon/geological_maps/qfaults'
        m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap= pycpt.load.gmtColormap(cmap)
            except:
                pass
        x, y            = m(lonarr, latarr)
        import matplotlib
        # cmap            = matplotlib.cm.get_cmap(cmap)
        # normalize       = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        # colors          = [cmap(normalize(value)) for value in mohoarr]

        im              = m.scatter(x, y, c=mohoarr, s=20, cmap=cmap, vmin=vmin, vmax=vmax)
        cb              = m.colorbar(im, location='bottom', size="3%", pad='2%')
        # cb              = plt.colorbar()
        cb.set_label(clabel, fontsize=20, rotation=0)
        cb.ax.tick_params(labelsize=15)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        
        if showfig:
            plt.show()
            
    def plot_miller_moho_finer(self, vmin=20., vmax=60., clabel='Crustal thickness (km)', cmap='gist_ncar',showfig=True, projection='lambert', \
                         infname='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoHiErrs-AlaskaMohoFineGrid.npz'):
        inarr   = np.load(infname)
        mohoarr = inarr['gridded_data_1']
        lonarr  = np.degrees(inarr['gridlons'])
        latarr  = np.degrees(inarr['gridlats'])
        qual    = inarr['quality']
        print mohoarr.min(), mohoarr.max()
        # m               = self._get_basemap(projection=projection)
        # shapefname      = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap= pycpt.load.gmtColormap(cmap)
            except:
                pass
        m               = self._get_basemap(projection=projection)
        self._get_lon_lat_arr(is_interp=True)
        x, y            = m(self.lonArr, self.latArr)
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon        = self.attrs['dlon_interp']
        dlat        = self.attrs['dlat_interp']
        field2d     = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
        field2d.read_array(lonArr = lonarr, latArr = latarr, ZarrIn = mohoarr)
        outfname    = 'interp_moho.lst'
        field2d.interp_surface(workingdir='./miller_moho_interp', outfname=outfname)
        # field2d.Zarr
        mask        = self.attrs['mask_interp']
        print field2d.Zarr.shape, mask.shape
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                tlat = self.lats[ilat]
                tlon = self.lons[ilon]
                ind      = np.where((abs(lonarr-tlon) < 0.6) * (abs(latarr-tlat) < 0.6))[0]
                # print ind
                if ind.size == 0:
                    mask[ilat, ilon] = True
                if np.any(qual[ind] == 0.):
                    mask[ilat, ilon] = True

                
        mdata       = ma.masked_array(field2d.Zarr, mask=mask )
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        
        cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[25., 29., 33., 37., 41., 45.])
        # cb              = plt.colorbar()
        
        cb.set_label(clabel, fontsize=20, rotation=0)
        cb.ax.tick_params(labelsize=40)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        
        ###
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
        if showfig:
            plt.show()
            
    def plot_diff_miller_moho_finer(self, vmin=20., vmax=60., clabel='Crustal thickness (km)', cmap='gist_ncar',showfig=True, projection='lambert', \
                         infname='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoHiErrs-AlaskaMohoFineGrid.npz'):
        inarr   = np.load(infname)
        mohoarr = inarr['gridded_data_1']
        lonarr  = np.degrees(inarr['gridlons'])
        latarr  = np.degrees(inarr['gridlats'])
        qual    = inarr['quality']
        print mohoarr.min(), mohoarr.max()
        
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap= pycpt.load.gmtColormap(cmap)
            except:
                pass
        self._get_lon_lat_arr(is_interp=True)

        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon        = self.attrs['dlon_interp']
        dlat        = self.attrs['dlat_interp']
        field2d     = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
        field2d.read_array(lonArr = lonarr, latArr = latarr, ZarrIn = mohoarr)
        outfname    = 'interp_moho.lst'
        field2d.interp_surface(workingdir='./miller_moho_interp', outfname=outfname)
        
        mask        = self.attrs['mask_interp']
        data, data_smooth\
                    = self.get_smooth_paraval(pindex='moho', dtype='avg', itype='ray', sigma=1, gsigma = 50., do_interp=True)
        diffdata    = field2d.Zarr - data_smooth
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                tlat = self.lats[ilat]
                tlon = self.lons[ilon]
                ind      = np.where((abs(lonarr-tlon) < 0.6) * (abs(latarr-tlat) < 0.6))[0]
                # print ind
                if ind.size == 0:
                    mask[ilat, ilon] = True
                if np.any(qual[ind] == 0.):
                    mask[ilat, ilon] = True
        diffdata    = diffdata[np.logical_not(mask)]
        
        from statsmodels import robust
        mad     = robust.mad(diffdata)
        outmean = diffdata.mean()
        outstd  = diffdata.std()
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
        ax      = plt.subplot()
        dbin    = 1.
        bins    = np.arange(min(diffdata), max(diffdata) + dbin, dbin)
        plt.hist(diffdata, bins=bins, normed=True)#, weights = areas)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Thickness difference (km)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g , mad = %g ' %(outmean, outstd, mad), fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([-15, 15])
        
        if showfig:
            plt.show()
    
    def plot_diff_miller_moho_finer(self, vmin=20., vmax=60., clabel='Crustal thickness (km)', cmap='gist_ncar',showfig=True, projection='lambert', \
                         infname='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoHiErrs-AlaskaMohoFineGrid.npz'):
        inarr   = np.load(infname)
        mohoarr = inarr['gridded_data_1']
        lonarr  = np.degrees(inarr['gridlons'])
        latarr  = np.degrees(inarr['gridlats'])
        qual    = inarr['quality']
        print mohoarr.min(), mohoarr.max()
        
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap= pycpt.load.gmtColormap(cmap)
            except:
                pass
        self._get_lon_lat_arr(is_interp=True)

        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon        = self.attrs['dlon_interp']
        dlat        = self.attrs['dlat_interp']
        field2d     = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
        field2d.read_array(lonArr = lonarr, latArr = latarr, ZarrIn = mohoarr)
        outfname    = 'interp_moho.lst'
        field2d.interp_surface(workingdir='./miller_moho_interp', outfname=outfname)
        
        mask        = self.attrs['mask_interp']
        data, data_smooth\
                    = self.get_smooth_paraval(pindex='moho', dtype='avg', itype='ray', sigma=1, gsigma = 50., do_interp=True)
        diffdata    = field2d.Zarr - data_smooth
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                tlat = self.lats[ilat]
                tlon = self.lons[ilon]
                ind      = np.where((abs(lonarr-tlon) < 0.6) * (abs(latarr-tlat) < 0.6))[0]
                # print ind
                if ind.size == 0:
                    mask[ilat, ilon] = True
                if np.any(qual[ind] == 0.):
                    mask[ilat, ilon] = True
        diffdata    = diffdata[np.logical_not(mask)]
        
        from statsmodels import robust
        mad     = robust.mad(diffdata)
        outmean = diffdata.mean()
        outstd  = diffdata.std()
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
        ax      = plt.subplot()
        dbin    = 1.
        bins    = np.arange(min(diffdata), max(diffdata) + dbin, dbin)
        plt.hist(diffdata, bins=bins, normed=True)#, weights = areas)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Thickness difference (km)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g , mad = %g ' %(outmean, outstd, mad), fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([-15, 15])
        
        data, data_smooth\
                    = self.get_smooth_paraval(pindex='moho', dtype='std', itype='ray', sigma=1, gsigma = 50., do_interp=True)
        diffdata    = diffdata/data_smooth[np.logical_not(mask)]
        mad     = robust.mad(diffdata)
        outmean = diffdata.mean()
        outstd  = diffdata.std()
        plt.figure()
        ax      = plt.subplot()
        dbin    = 0.2
        bins    = np.arange(min(diffdata), max(diffdata) + dbin, dbin)
        plt.hist(diffdata, bins=bins, normed=True)#, weights = areas)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Thickness difference (km)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g , mad = %g ' %(outmean, outstd, mad), fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([-3, 3])
        
        
        if showfig:
            plt.show()
            
    def plot_crust1(self, infname='crsthk.xyz', vmin=20., vmax=60., clabel='Crustal thickness (km)',
                    cmap='gist_ncar',showfig=True, projection='lambert'):
        inArr       = np.loadtxt(infname)
        lonArr      = inArr[:, 0]
        lonArr      = lonArr.reshape(lonArr.size/360, 360)
        latArr      = inArr[:, 1]
        latArr      = latArr.reshape(latArr.size/360, 360)
        depthArr    = inArr[:, 2]
        depthArr    = depthArr.reshape(depthArr.size/360, 360)
        m               = self._get_basemap(projection=projection)
        # shapefname      = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap= pycpt.load.gmtColormap(cmap)
            except:
                pass
        x, y            = m(lonArr, latArr)

        im              = m.pcolormesh(x, y, depthArr, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[25., 29., 33., 37., 41., 45.])
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=40)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        ###
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #############################
        import shapefile
        shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)

        if showfig:
            plt.show()
    
    def plot_diff_crust1(self, infname='crsthk.xyz', vmin=20., vmax=60., clabel='Crustal thickness (km)',
                    cmap='gist_ncar',showfig=True, projection='lambert'):
        inArr       = np.loadtxt(infname)
        lonArr      = inArr[:, 0] + 360.
        # lonArr      = lonArr.reshape(lonArr.size/360, 360)
        latArr      = inArr[:, 1]
        # latArr      = latArr.reshape(latArr.size/360, 360)
        depthArr    = inArr[:, 2]
        # depthArr    = depthArr.reshape(depthArr.size/360, 360)
        ###
       
        self._get_lon_lat_arr(is_interp=True)

        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon        = self.attrs['dlon_interp']
        dlat        = self.attrs['dlat_interp']
        field2d     = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
        field2d.read_array(lonArr = lonArr, latArr = latArr, ZarrIn = depthArr)
        outfname    = 'interp_moho.lst'
        field2d.interp_surface(workingdir='./miller_moho_interp', outfname=outfname)
        
        mask        = self.attrs['mask_interp']
        data, data_smooth\
                    = self.get_smooth_paraval(pindex='moho', dtype='avg', itype='ray', sigma=1, gsigma = 50., do_interp=True)
        diffdata    = field2d.Zarr - data_smooth
        ###
        infname='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoHiErrs-AlaskaMohoFineGrid.npz'
        inarr   = np.load(infname)
        mohoarr = inarr['gridded_data_1']
        lonarr  = np.degrees(inarr['gridlons'])
        latarr  = np.degrees(inarr['gridlats'])
        qual    = inarr['quality']
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                tlat = self.lats[ilat]
                tlon = self.lons[ilon]
                ind      = np.where((abs(lonarr-tlon) < 0.6) * (abs(latarr-tlat) < 0.6))[0]
                # print ind
                if ind.size == 0:
                    mask[ilat, ilon] = True
                if np.any(qual[ind] == 0.):
                    mask[ilat, ilon] = True
        ###
        diffdata    = diffdata[np.logical_not(mask)]
        
        from statsmodels import robust
        mad     = robust.mad(diffdata)
        outmean = diffdata.mean()
        outstd  = diffdata.std()
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
        ax      = plt.subplot()
        dbin    = 1.
        bins    = np.arange(min(diffdata), max(diffdata) + dbin, dbin)
        plt.hist(diffdata, bins=bins, normed=True)#, weights = areas)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Thickness difference (km)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g , mad = %g ' %(outmean, outstd, mad), fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([-15, 15])
        
        
        if showfig:
            plt.show()
    
    def plot_sed1(self, infname='sedthk.xyz', vmin=0., vmax=7., clabel='Sediment thickness (km)',
                    cmap='gist_ncar',showfig=True, projection='lambert'):
        inArr       = np.loadtxt(infname)
        lonArr      = inArr[:, 0]
        lonArr      = lonArr.reshape(lonArr.size/360, 360)
        latArr      = inArr[:, 1]
        latArr      = latArr.reshape(latArr.size/360, 360)
        depthArr    = inArr[:, 2]
        depthArr    = depthArr.reshape(depthArr.size/360, 360)
        m               = self._get_basemap(projection=projection)
        # shapefname      = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap= pycpt.load.gmtColormap(cmap)
            except:
                pass
        x, y            = m(lonArr, latArr)

        im              = m.pcolormesh(x, y, depthArr, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb              = m.colorbar(im, location='bottom', size="3%", pad='2%')
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=40)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        
        if showfig:
            plt.show()
            
            
    def get_azi_data(self, fname, lon, lat):
        if lon < 0.:
            lon     += 360.
        grd_id  = str(float(lon))+'_'+str(float(lat))
        grp     = self['azi_grd_pts']
        if grd_id in grp.keys():
            data= grp[grd_id+'/disp_azi_ray'].value
            np.savetxt(fname, data.T, fmt='%g', header='T C_ray unC_ray psi unpsi amp unamp')
        else:
            print 'No data for this point!'
        return
    
    def get_lov_data(self, fname, lon, lat):
        if lon < 0.:
            lon     += 360.
        grd_id  = str(float(lon))+'_'+str(float(lat))
        grp     = self['grd_pts']
        if grd_id in grp.keys():
            data= grp[grd_id+'/disp_ph_lov'].value
            np.savetxt(fname, data.T, fmt='%g', header='T C_lov unC_lov')
        else:
            print 'No data for this point!'
        return
    
    def get_refmod(self, fname, lon, lat, dtype='avg'):
        if lon < 0.:
            lon     += 360.
        grd_id  = str(float(lon))+'_'+str(float(lat))
        grp     = self['grd_pts']
        if grd_id in grp.keys():
            data= grp[grd_id+'/'+dtype+'_paraval_ray'].value
            np.savetxt(fname, data.T, fmt='%g')
        else:
            print 'No data for this point!'
        return
    
    def save_vsv(self, outdir):
        grd_lst = self['grd_pts'].keys()
        self._get_lon_lat_arr(is_interp=True)
        z       = self['avg_paraval/z_smooth']
        vs3d    = self['avg_paraval/vs_smooth']
        for grd_id in grd_lst:
            try:
                unvs        = self['grd_pts/'+grd_id+'/vs_std_ray'].value
            except:
                continue
            try:
                avg_paraval     = self['grd_pts/'+grd_id+'/avg_paraval_vti'].value
                std_paraval     = self['grd_pts/'+grd_id+'/std_paraval_vti'].value
            except:
                continue
            # Vsv
            vsvfname    = outdir + '/'+grd_id+'_vsv.mod'
            split_id    = grd_id.split('_')
            grd_lon     = float(split_id[0])
            grd_lat     = float(split_id[1])
            ind_lon     = self.lons==grd_lon
            ind_lat     = self.lats==grd_lat
            vs          = np.zeros(z.size)
            tvs         = vs3d[ind_lat, :, :]
            vs[:]       = tvs[0, ind_lon, :]
            vs[0]       = vs[1]
            unvs[0]     = unvs[1]
            outarr      = np.append(z, vs)
            outarr      = np.append(outarr, unvs)
            outarr      = outarr.reshape(3, z.size)
            outarr      = outarr.T

            np.savetxt(vsvfname, outarr, fmt='%g', header='depth(km) Vsv(km/s) Error_Vsv(km/s)')
            
    def save_gamma(self, outdir):
        grd_lst = self['grd_pts'].keys()
        for grd_id in grd_lst:
            try:
                avg_paraval     = self['grd_pts/'+grd_id+'/avg_paraval_vti'].value
                std_paraval     = self['grd_pts/'+grd_id+'/std_paraval_vti'].value
            except:
                continue
            gamma       = avg_paraval[-3:]
            ungamma     = std_paraval[-3:]
            
            gammafname    = outdir + '/'+grd_id+'_gamma.mod'
            outarr      = np.append(gamma, ungamma)
            outarr      = outarr.reshape(2, 3)
            outarr      = outarr.T
            np.savetxt(gammafname, outarr, fmt='%g', header='gamma(%) Error_gamma(%) ')
            
    def save_vsv_vsh(self, outdir):
        grd_lst = self['grd_pts'].keys()
        self._get_lon_lat_arr(is_interp=True)
        z       = self['avg_paraval/z_smooth']
        vs3d    = self['avg_paraval/vs_smooth']
        for grd_id in grd_lst:
            try:
                unvs        = self['grd_pts/'+grd_id+'/vs_std_ray'].value
            except:
                continue
            try:
                avg_paraval     = self['grd_pts/'+grd_id+'/avg_paraval_vti'].value
                std_paraval     = self['grd_pts/'+grd_id+'/std_paraval_vti'].value
            except:
                continue
            # Vsv
            fname    = outdir + '/'+grd_id+'_vsv_vsh.mod'
            split_id    = grd_id.split('_')
            grd_lon     = float(split_id[0])
            grd_lat     = float(split_id[1])
            ind_lon     = self.lons==grd_lon
            ind_lat     = self.lats==grd_lat
            vs          = np.zeros(z.size)
            tvs         = vs3d[ind_lat, :, :]
            vs[:]       = tvs[0, ind_lon, :]
            vs[0]       = vs[1]
            unvs[0]     = unvs[1]
            outarr      = np.append(z, vs)
            outarr      = np.append(outarr, unvs)
            # vsv done
            gamma       = avg_paraval[-3:]
            ungamma     = std_paraval[-3:]
            hv_ratio    = (1. + gamma/200.)/(1 - gamma/200.)
            import uncertainties
            from uncertainties import unumpy
            gamma_un    = unumpy.uarray(gamma, ungamma)
            hv_ratio_un = (1. + gamma_un/200.)/(1 - gamma_un/200.)
            # crust & sedi
            crtthk      = avg_paraval[-4]
            sedthk      = avg_paraval[-5]
            # vsh
            vsh         = vs.copy()
            unvsh       = unvs.copy()
            if gamma[0] == 0.:
                vsh[(z>sedthk)*(z<=crtthk)] *= hv_ratio[1]
                # un
                tempvsv         = vs[(z>sedthk)*(z<=crtthk)]
                tempunvsv       = unvs[(z>sedthk)*(z<=crtthk)]
                temp_vsv_un     = unumpy.uarray(tempvsv, tempunvsv)
                temp_vsh_un     = temp_vsv_un*hv_ratio_un[1]
                unvsh[(z>sedthk)*(z<=crtthk)] = uncertainties.unumpy.std_devs(temp_vsh_un)
            else:
                vsh[(z<=sedthk)] *= hv_ratio[0]
                # un
                tempvsv         = vs[(z<=sedthk)]
                tempunvsv       = unvs[(z<=sedthk)]
                temp_vsv_un     = unumpy.uarray(tempvsv, tempunvsv)
                temp_vsh_un     = temp_vsv_un*hv_ratio_un[0]
                unvsh[(z<=sedthk)] = uncertainties.unumpy.std_devs(temp_vsh_un)
                
            vsh[(z>crtthk)] *= hv_ratio[2]
            # un
            tempvsv         = vs[(z>crtthk)]
            tempunvsv       = unvs[(z>crtthk)]
            temp_vsv_un     = unumpy.uarray(tempvsv, tempunvsv)
            temp_vsh_un     = temp_vsv_un*hv_ratio_un[2]
            unvsh[(z>crtthk)] = uncertainties.unumpy.std_devs(temp_vsh_un)
            
            # outarr      = np.append(z, vs)
            outarr      = np.append(outarr, vsh)
            outarr      = np.append(outarr, unvsh)
            outarr      = outarr.reshape(5, z.size)
            outarr      = outarr.T
        # 
            np.savetxt(fname, outarr, fmt='%g', header='depth(km) Vsv(km/s) Error_Vsv(km/s) Vsh(km/s) Error_Vsh(km/s)')
    
    def save_vsv_gamma(self, outdir):
        grd_lst = self['grd_pts'].keys()
        self._get_lon_lat_arr(is_interp=True)
        z       = self['avg_paraval/z_smooth']
        vs3d    = self['avg_paraval/vs_smooth']
        for grd_id in grd_lst:
            try:
                unvs        = self['grd_pts/'+grd_id+'/vs_std_ray'].value
            except:
                continue
            try:
                avg_paraval     = self['grd_pts/'+grd_id+'/avg_paraval_vti'].value
                std_paraval     = self['grd_pts/'+grd_id+'/std_paraval_vti'].value
            except:
                continue
            # Vsv
            fname    = outdir + '/'+grd_id+'_vsv_gamma.mod'
            split_id    = grd_id.split('_')
            grd_lon     = float(split_id[0])
            grd_lat     = float(split_id[1])
            ind_lon     = self.lons==grd_lon
            ind_lat     = self.lats==grd_lat
            vs          = np.zeros(z.size)
            tvs         = vs3d[ind_lat, :, :]
            vs[:]       = tvs[0, ind_lon, :]
            vs[0]       = vs[1]
            unvs[0]     = unvs[1]
            outarr      = np.append(z, vs)
            outarr      = np.append(outarr, unvs)
            # vsv done
            gamma       = avg_paraval[-3:]
            ungamma     = std_paraval[-3:]
            # vsh
            gamma_arr   = np.zeros(vs.size)     
            ungamma_arr = np.zeros(vs.size)
            # crust & sedi
            crtthk      = avg_paraval[-4]
            sedthk      = avg_paraval[-5]
            if gamma[0] == 0.:
                gamma_arr[(z>sedthk)*(z<=crtthk)] = gamma[1]
                ungamma_arr[(z>sedthk)*(z<=crtthk)] = ungamma[1]
            else:
                gamma_arr[(z<=sedthk)] = gamma[0]
                ungamma_arr[(z<=sedthk)] = ungamma[0]
                
            gamma_arr[(z>crtthk)] = gamma[2]
            ungamma_arr[(z>crtthk)] = ungamma[2]

            outarr      = np.append(outarr, gamma_arr)
            outarr      = np.append(outarr, ungamma_arr)
            outarr      = outarr.reshape(5, z.size)
            outarr      = outarr.T
        # 
            np.savetxt(fname, outarr, fmt='%g', header='depth(km) Vsv(km/s) Error_Vsv(km/s) gamma(%) Error_gamma(%)')

    def save_group_vel(self, outdir, Tmax=24.):
        grd_lst = self['grd_pts'].keys()
        for grd_id in grd_lst:
            try:
                data            = self['grd_pts/'+grd_id+'/disp_gr_ray'].value
            except:
                continue
            # Vsv
            grpfname    = outdir + '/'+grd_id+'_U.txt'
            
            # outarr      = np.append(z, vs)
            # outarr      = np.append(outarr, unvs)
            # outarr      = outarr.reshape(3, z.size)
            # outarr      = outarr.T
            
            ind         = data[0, :] <= Tmax
            data        = data[:, ind]
            np.savetxt(grpfname, data.T, fmt='%g', header='Period(sec) U(km/s) Error_U(km/s)')
            
    def dump_azi_uppcrst(self, outfname = './azi_uppcrst.txt', ingrdfname=None):
        self._get_lon_lat_arr(is_interp=True)
        azi_grp     = self['azi_grd_pts']
        # get the list for inversion
        if ingrdfname is None:
            grdlst  = azi_grp.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        fid = open(outfname, 'wb')
        igrd        = 0
        Ngrd        = len(grdlst)
        topoarr     = self['topo_interp'].value
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat = float(split_id[1])
            igrd    += 1
            #-----------------------------
            # get data
            #-----------------------------
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (self.lonArr == grd_lon)*(self.latArr == grd_lat)
            topovalue           = topoarr[index]
            uppcrt_thk          = 15. + topovalue - self['avg_paraval/11_smooth'].value[index]
            try:
                psi2                = azi_grp[grd_id]['psi2'].value[0]
                unpsi2              = azi_grp[grd_id]['unpsi2'].value[0]
                amp                 = azi_grp[grd_id]['amp'].value[0]
                unamp               = azi_grp[grd_id]['unamp'].value[0]
                fid.writelines('%g %g %g %g %g %g %g\n' %(grd_lon, grd_lat, uppcrt_thk, psi2, unpsi2, amp, unamp))
            except KeyError:
                continue
            
            
        fid.close()
        
    def plot_hti_stress(self, depth, depthavg=3., gindex=0, plot_axis=True, plot_data=False, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='cv', projection='lambert', geopolygons=None, \
                    vmin=None, vmax=None, showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::

        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr(is_interp=True)
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex].value
            unpsi2      = grp['unpsi2_%d' %gindex].value
            amp         = grp['amp_%d' %gindex].value
            unamp       = grp['unamp_%d' %gindex].value
        else:
            plot_axis   = False
        mask        = grp['mask'].value
        grp         = self['avg_paraval']
        vs3d        = grp['vs_smooth'].value
        zArr        = grp['z_smooth'].value
        if depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            data    = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            data        = vs3d[:, :, index]
        
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='purple')

        # 
        import shapefile
        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        #--------------------------
        
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(data, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            if len(ticks)>0:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=ticks)
            else:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=35, rotation=0)
            cb.ax.tick_params(labelsize=35)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        if plot_axis:
            if scaled:
                # print ampref
                U       = np.sin(psi2/180.*np.pi)*amp/ampref/normv
                V       = np.cos(psi2/180.*np.pi)*amp/ampref/normv
                Uref    = np.ones(self.lonArr.shape)*1./normv
                Vref    = np.zeros(self.lonArr.shape)
            else:
                U       = np.sin(psi2/180.*np.pi)/normv
                V       = np.cos(psi2/180.*np.pi)/normv
            # rotate vectors to map projection coordinates
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
            # # # if scaled:
            # # #     Uref1, Vref1, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr-360., self.latArr, returnxy=True)
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
            if masked:
                U   = ma.masked_array(U, mask=mask_psi )
                V   = ma.masked_array(V, mask=mask_psi )

         
        fname       = 'wsm2016.csv'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        dtlst       = []
        lonlst      = []
        latlst      = []
        amplst      = []
        misfit      = self['hti_model/misfit'].value
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        
        with open(fname, 'rb') as fid:
            firstline = True
            for line in fid.readlines():
                if firstline:
                    firstline   = False
                    continue
                lst = line.split(',')
                lonsks      = float(lst[3])
                lonsks      += 360.
                latsks      = float(lst[2])
                try:
                    ind_lon     = np.where(abs(self.lons - lonsks)<.2)[0][0]
                    ind_lat     = np.where(abs(self.lats - latsks)<.1)[0][0]
                except:
                    continue
                if mask[ind_lat, ind_lon]:
                    continue
                if unpsi2[ind_lat, ind_lon] > 30. or amp[ind_lat, ind_lon] < 0.6:
                    continue
                if lst[7]=='D' :
                    continue
                
                stalst.append(lst[0])
                philst.append(float(lst[4]))

                lonlst.append(float(lst[3]))
                latlst.append(float(lst[2]))
                psilst.append(psi2[ind_lat, ind_lon])
                unpsilst.append(unpsi2[ind_lat, ind_lon])
                amplst.append(amp[ind_lat, ind_lon])
                
                # temp_misfit = misfit[ind_lat, ind_lon]
                # temp_dpsi   = abs(psi2[ind_lat, ind_lon] - float(lst[5]))
                # if temp_dpsi > 90.:
                #     temp_dpsi   = 180. - temp_dpsi
                # # # 
                # # # if temp_misfit > 1. and temp_dpsi > 30. or temp_dpsi > 30. and self.lons[ind_lon] > -140.+360.:
                # # #     vpr = self.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=True,\
                # # #                 outlon=self.lons[ind_lon], outlat=self.lats[ind_lat])
                # # #     vpr.linear_inv_hti(depth_mid_crust=-1., depth_mid_mantle=100.)
                # # #     psilst1.append(vpr.model.htimod.psi2[1])
                # # #     psilst2.append(vpr.model.htimod.psi2[2])
                # # #     lonlst2.append(float(lst[4]))
                # # #     latlst2.append(float(lst[2]))
        phiarr              = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        # unphiarr= np.array(unphilst)
        # unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        # print amparr.mean()
        # dtarr   = np.array(dtlst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        dtref   = 1.
        normv   = 2.
        
        # # # Usks    = np.sin(phiarr/180.*np.pi)*dtarr/dtref/normv
        # # # Vsks    = np.cos(phiarr/180.*np.pi)*dtarr/dtref/normv
        
        Usks    = np.sin(phiarr/180.*np.pi)/dtref/normv
        Vsks    = np.cos(phiarr/180.*np.pi)/dtref/normv
        
        Upsi    = np.sin(psiarr/180.*np.pi)/dtref/normv
        Vpsi    = np.cos(psiarr/180.*np.pi)/dtref/normv
        
        Uref    = np.ones(self.lonArr.shape)*1./normv
        Vref    = np.zeros(self.lonArr.shape)
        Usks, Vsks, xsks, ysks  = m.rotate_vector(Usks, Vsks, lonarr, latarr, returnxy=True)
        mask    = np.zeros(Usks.size, dtype=bool)
        # 
        dpsi            = abs(psiarr - phiarr)
        
        # dpsi2 = (psiarr - phiarr)
        # dpsi2[dpsi2>90.] = dpsi2[dpsi2>90.] - 180.
        # dpsi2[dpsi2<-90.] = dpsi2[dpsi2<-90.] + 180.
        # # dpsi
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        # print 'mean damp', amparr.mean() 
        # undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        # undpsi2 = undpsi.copy()
        # # print 'un:', unpsiarr.mean(), unphiarr.mean(), undpsi.mean()
        # # return unpsiarr, unphiarr
        # # # # ind_outline         = amparr < .2
        # 
        # # 81 % comparisonH
        # mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        # mask[(amparr<.32)*(dpsi>=30.)]   = True # 2 p
        # # # mask[(amparr<.4)*(dpsi>=30.)]   = True # 3 p
        # 
        # # mask[(amparr<.4)]   = True
        # # return amparr, mask
        # print 'mean damp', amparr[np.logical_not(mask)].mean() 
        ###
        
        # mask[(amparr<.2)*(dpsi>=30.)]   = True
        # mask[(amparr<.3)*(dpsi>=30.)*(lonarr<-140.)]   = True
        
        
        xsks    = xsks[np.logical_not(mask)]
        ysks    = ysks[np.logical_not(mask)]
        Usks    = Usks[np.logical_not(mask)]
        Vsks    = Vsks[np.logical_not(mask)]
        Upsi    = Upsi[np.logical_not(mask)]
        Vpsi    = Vpsi[np.logical_not(mask)]
        # dpsi    = dpsi[np.logical_not(mask)]
        # undpsi = undpsi[np.logical_not(mask)]
        # dpsi2 = dpsi2[np.logical_not(mask)]

        # # # Q1      = m.quiver(xsks, ysks, Usks, Vsks, scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b')
        # # # Q2      = m.quiver(xsks, ysks, -Usks, -Vsks, scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b')
        Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Usks[dpsi<=30.], Vsks[dpsi<=30.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Usks[dpsi<=30.], -Vsks[dpsi<=30.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q1      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], Usks[(dpsi>30.)*(dpsi<=60.)], Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q2      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], -Usks[(dpsi>30.)*(dpsi<=60.)], -Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q1      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], Usks[dpsi>60.], Vsks[dpsi>60.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        Q2      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], -Usks[dpsi>60.], -Vsks[dpsi>60.],\
                           scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        
        # Q1      = m.quiver(xsks, ysks, Usks, Vsks,\
        #                    scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        # Q2      = m.quiver(xsks, ysks, -Usks, -Vsks,\
        #                    scale=20, width=width+0.003, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        # # # Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Upsi[dpsi<=30.], Vpsi[dpsi<=30.], scale=20, width=width-0.001, headaxislength=0, headlength=0, headwidth=0.5, color='r')
        # # # Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Upsi[dpsi<=30.], -Vpsi[dpsi<=30.], scale=20, width=width-0.001, headaxislength=0, headlength=0, headwidth=0.5, color='r')
        # # # 
        # # # Q1      = m.quiver(xsks[dpsi>30.], ysks[dpsi>30.], Upsi[dpsi>30.], Vpsi[dpsi>30.], scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        # # # Q2      = m.quiver(xsks[dpsi>30.], ysks[dpsi>30.], -Upsi[dpsi>30.], -Vpsi[dpsi>30.], scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        
            #         Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        # # # Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        # # # Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
        
        
        # if len(psilst1) > 0.:
        #     Upsi2   = np.sin(np.array(psilst2)/180.*np.pi)/dtref/normv
        #     Vpsi2   = np.cos(np.array(psilst2)/180.*np.pi)/dtref/normv
        #     # print np.array(psilst2)
        #     # ind = np.array(lonlst2).argmax()
        #     Upsi2[0] = Upsi2[1]
        #     Vpsi2[0] = Vpsi2[1]
        #     Upsi2, Vpsi2, xsks2, ysks2  = m.rotate_vector(Upsi2, Vpsi2, np.array(lonlst2), np.array(latlst2), returnxy=True)
        #     Q1      = m.quiver(xsks2, ysks2, Upsi2, Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        #     Q2      = m.quiver(xsks2, ysks2, -Upsi2, -Vpsi2, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            
            
        plt.suptitle(title, fontsize=20)
        plt.show()
        # 
        ax      = plt.subplot()
        dbin    = 10.
        bins    = np.arange(0, 90.+ dbin, dbin)
        
        weights = np.ones_like(dpsi)/float(dpsi.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(dpsi)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        
        p.patches[3].set_color('lime')
        p.patches[4].set_color('lime')
        p.patches[5].set_color('lime')
        p.patches[6].set_color('r')
        p.patches[7].set_color('r')
        p.patches[8].set_color('r')
        # 
        # # # # print dpsi.size
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        good_per= float(dpsi[dpsi>60.].size)/float(dpsi.size)
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()
        # 
        #
        # # # # print dpsi.size
        # ax      = plt.subplot()
        # import matplotlib.mlab as mlab
        # from matplotlib.ticker import FuncFormatter
        # # good_per= float(dpsi[dpsi<30.].size)/float(dpsi.size)
        # # plt.ylabel('Percentage (%)', fontsize=60)
        # # print dpsi2.shape, undpsi2.shape
        # weights = np.ones_like(dpsi2)/float(dpsi2.size)
        # plt.hist(dpsi2/undpsi2, weights=weights, bins=25)
        # print (dpsi2/undpsi2).std()
        # plt.xlabel('Normalized Angle difference (deg)', fontsize=60, rotation=0)
        # # plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        # ax.tick_params(axis='x', labelsize=40)
        # # plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        # ax.tick_params(axis='y', labelsize=40)
        # formatter = FuncFormatter(to_percent)
        # # Set the formatter
        # plt.gca().yaxis.set_major_formatter(formatter)
        # plt.xlim([-10, 10.])
        # plt.show()
            
        return amp
    
    
    
    
