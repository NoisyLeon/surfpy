# -*- coding: utf-8 -*-
"""
hdf5 for noise eikonal tomography
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.pymcinv.invbase as invbase
import surfpy.pymcinv.inviso as inviso
import surfpy.pymcinv.inverse_solver as inverse_solver
# # # import surfpy.pymcinv.isopost as isopost
import surfpy.pymcinv.vmodel as vmodel
import surfpy.pymcinv._model_funcs as _model_funcs
from surfpy.pymcinv._modparam_vti import NOANISO, LAYERGAMMA, GAMMASPLINE, VSHSPLINE, LAYER, BSPLINE, GRADIENT, WATER
import surfpy.eikonal._grid_class as _grid_class

import surfpy.cpt_files as cpt_files
cpt_path    = cpt_files.__path__._path[0]
import surfpy.map_dat as map_dat
map_path    = map_dat.__path__._path[0]

import numpy as np
import numpy.ma as ma
import numba
from pyproj import Geod
import obspy
import time
from datetime import datetime
import warnings
import glob
import sys
import copy
import os

if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import shapefile
import matplotlib.pyplot as plt

geodist             = Geod(ellps='WGS84')

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

# @numba.jit(numba.float64[:, :, :](numba.float64[:, ], numba.float64[:, :, :], numba.float64), nopython = True)
def _get_avg_vs3d(zArr, vs3d, depthavg):
    tvs3d           = vs3d.copy()
    Nlat, Nlon, Nz  = vs3d.shape
    Nz              = zArr.size
    for i in range(Nz):
        z       = zArr[i]
        if z < depthavg:
            tvs3d[:, :, i]  = (vs3d[:, :, zArr <= 2.*depthavg]).mean(axis=2)
            continue
        index   = (zArr <= z + depthavg) + (zArr >= z - depthavg)
        tvs3d[:, :, i]  = (vs3d[:, :, index]).mean(axis=2)
    return tvs3d

class vtih5(inviso.isoh5):
    
    def mc_inv_vti(self, use_ref=False, ingrdfname=None, solver_type = 0, crtstd = None, wdisp = 1., cdist = 75.,rffactor = 40., ray=True, lov=True,\
        outdir = None, restart = False, vp_water=1.5, isconstrt=True, sedani = False, crtani=True, manani=True,
        crt_depth = -1., mantle_depth = -1., step4uwalk=1500, numbrun=15000, subsize=1000, nprocess=None, parallel=True,\
        skipmask=True, Ntotalruns=10, misfit_thresh=1.0, Nmodelthresh=200, outlon=None, outlat=None, verbose = False):
        """
        Bayesian Monte Carlo inversion of geographical grid points
        ==================================================================================================================
        ::: input :::
        use_ref         - use reference input model or not(default = False, use ak135 instead)
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        wdisp           - weight of dispersion data (default = 1.0, only use dispersion data for inversion)
        cdist           - threshhold distance for loading rf data, only takes effect when wdisp < 1.0
        rffactor        - factor of increasing receiver function uncertainty
        
        ray/lov         - include Rayleigh/Love dispersion data
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
        self._get_lon_lat_arr()
        if (outlon is None) or (outlat is None):
            print ('[%s] [MC_VTI_INVERSION] inversion START' %datetime.now().isoformat().split('.')[0])
            if outdir is None:
                if restart:
                    try:
                        outdir  = self.attrs['mc_inv_run_path']
                    except:
                        outdir  = os.path.dirname(self.filename)+'/mc_inv_run_%s' %datetime.now().isoformat().split('.')[0]
                else:
                    outdir  = os.path.dirname(self.filename)+'/mc_inv_run_%s' %datetime.now().isoformat().split('.')[0]
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            self.attrs.create(name = 'mc_inv_run_path', data = outdir)
        else:
            restart     = False
            if outlon > 180. and self.ilontype == 0:
                outlon  -= 360.
            if outlon < 0. and self.ilontype == 1:
                outlon  += 360.
        start_time_total= time.time()
        grd_grp         = self['grd_pts']
        # get the list for inversion
        if ingrdfname is None:
            grdlst  = list(grd_grp.keys())
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    # # # if lon < 0.:
                    # # #     lon += 360.
                    if lon > 180. and self.ilontype == 0:
                        lon  -= 360.
                    if lon < 0. and self.ilontype == 1:
                        lon  += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        if wdisp != 1. and wdisp > 0.:
            try:
                sta_grp = self['sta_pts']
                stlos   = self.attrs['stlos']
                stlas   = self.attrs['stlas']
                Nsta    = stlos.size
            except:
                print ('!!! Error ! wdisp must be 1.0 if station group NOT exists')
                return
        igrd        = 0
        Ngrd        = len(grdlst)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            # # # if grd_lon > 180.:
            # # #     grd_lon     -= 360.
            if grd_lon > 180. and self.ilontype == 0:
                grd_lon -= 360.
            if grd_lon < 0. and self.ilontype == 1:
                grd_lon += 360.
            grd_lat = float(split_id[1])
            igrd    += 1
            # check if result exists
            if restart:
                outfname    = outdir+'/mc_inv.'+grd_id+'.npz'
                if os.path.isfile(outfname):
                    print ('[%s] [MC_VTI_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'SKIP upon exitence, grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
                    continue
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = inverse_solver.inverse_vprofile()
            if ray:
                try:
                    indisp      = grd_grp[grd_id+'/disp_ph_ray'][()]
                    vpr.get_disp(indata = indisp, dtype='ph', wtype='ray')
                except KeyError:
                    print ('!!! WARNING: No Rayleigh phase dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat))
            if lov:
                try:
                    indisp      = grd_grp[grd_id+'/disp_ph_lov'][()]
                    vpr.get_disp(indata = indisp, dtype='ph', wtype='lov')
                except KeyError:
                    print ('!!! WARNING: No Love phase dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat))
            if vpr.data.dispR.npper == 0 and vpr.data.dispL.npper == 0:
                print ('!!! WARNING: No dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat))
                continue
            # receiver functions
            if wdisp != 1.0 and wdisp > 0.:
                az, baz, dist   = geodist.inv(grd_lon*np.ones(Nsta), grd_lat*np.ones(Nsta), stlos, stlas) 
                dist            /= 1000.
                if dist.min() >= cdist:
                    grd_wdisp   = 1.0
                else:
                    ind_min     = dist.argmin()
                    staid       = list(sta_grp.keys())[ind_min]
                    delta       = sta_grp[staid].attrs['delta']
                    stla        = sta_grp[staid].attrs['stla']
                    stlo        = sta_grp[staid].attrs['stlo']
                    distmin, az, baz    = obspy.geodetics.gps2dist_azimuth(stla, stlo, grd_lat, grd_lon)
                    distmin     /= 1000.
                    indata      = sta_grp[staid+'/rf_data'][()]
                    vpr.get_rf(indata = indata, delta = delta)
                    # weight
                    grd_wdisp   = wdisp + (dist.min()/cdist)*(1. - wdisp)
            elif wdisp < 0.:
                grd_wdisp   = wdisp
            else:
                grd_wdisp   = 1.0
            #-----------------------------
            # initial model parameters
            #-----------------------------
            crtthk              = grd_grp[grd_id].attrs['crust_thk']
            sedthk              = grd_grp[grd_id].attrs['sediment_thk']
            topovalue           = grd_grp[grd_id].attrs['topo']
            vti_numbp           = np.array([sedani, crtani, manani], dtype = int)
            vti_modtype         = [NOANISO, NOANISO, NOANISO]
            if sedani:
                vti_modtype[0]  = LAYERGAMMA
            if crtani:
                vti_modtype[1]  = LAYERGAMMA
            if manani:
                vti_modtype[2]  = LAYERGAMMA
            if use_ref:
                vsdata          = sta_grp[staid+'/reference_vs'][()]
                vpr.model.vtimod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], crtthk=crtthk, sedthk=sedthk,\
                            topovalue=topovalue, maxdepth=200., vp_water=vp_water)
            else:
                vpr.model.vtimod.parameterize_ak135(crtthk=crtthk, sedthk=sedthk, topovalue=topovalue, \
                    maxdepth = 200., vp_water=vp_water, crt_depth = crt_depth, mantle_depth = mantle_depth,\
                    vti_numbp = vti_numbp, vti_modtype = vti_modtype)
            if crtstd is None:
                vpr.get_paraind(mtype='vti', crtthk = None)
            else:
                vpr.get_paraind(mtype='vti', crtthk = crtthk, crtstd = crtstd)
            if (not outlon is None) and (not outlat is None):
                # # # # # if ( vpr.model.vtimod.para.paraindex[2, -1] >70.):
                # # # # #     return vpr
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:    
                    return vpr
            start_time_grd  = time.time()
            print ('[%s] [MC_VTI_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
            if grd_wdisp != 1.0 and grd_wdisp >= 0.:
                print ('=== using rf data, station id: %s, stla = %g, stlo = %g, distance = %g km, wdisp = %g' %(staid, stla, stlo, distmin, grd_wdisp))
                grd_grp[grd_id].attrs.create(name = 'is_rf', data = True)
                grd_grp[grd_id].attrs.create(name = 'distance_rf', data = dist.min())
            else:
                grd_grp[grd_id].attrs.create(name = 'is_rf', data = False)
            if parallel:
                vpr.mc_joint_inv_vti_mp(outdir=outdir, wdisp=wdisp, rffactor=rffactor, solver_type=solver_type, Ntotalruns=Ntotalruns, \
                    misfit_thresh=misfit_thresh, Nmodelthresh=Nmodelthresh, isconstrt=isconstrt, pfx=grd_id, verbose=verbose,\
                    step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_vti(outdir=outdir, wdisp=wdisp, rffactor=rffactor, solver_type=solver_type,\
                   isconstrt=isconstrt, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            end_time    = time.time()
            print ('[%s] [MC_VTI_INVERSION] inversion DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', elasped time = %g'%(end_time - start_time_grd) + ' sec; total elasped time = %g' %(end_time - start_time_total))
        print ('[%s] [MC_VTI_INVERSION] inversion ALL DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', total elasped time = %g' %(end_time - start_time_total))
        return
    
    
    def mc_inv_sta_vti(self, use_ref=False, instafname = None, ray = True, lov=True, outdir = None, restart = False,
        crtstd = None, wdisp = 0.2, rffactor = 40., solver_type = 0, vp_water=1.5, isconstrt=True, sedani = False, crtani=True, manani=True,
        crt_depth = -1., mantle_depth = -1., step4uwalk=1500, numbrun=15000, subsize=1000, nprocess=None, parallel=True,
        Ntotalruns=10, misfit_thresh=1.0, Nmodelthresh=200, outstaid=None, verbose = False, verbose2 = False):
        """
        Bayesian Monte Carlo inversion of surface wave/receiver functions at station location
        ==================================================================================================================
        ::: input :::
        use_ref         - use reference input model or not(default = False, use ak135 instead)
        instafname      - input station id  list file indicating the grid points for station based inversion
        phase/group     - include phase/group velocity dispersion data or not
        outdir          - output directory
        restart         - continue to run previous inversion or not
        wdisp           - weight of dispersion data
        rffactor        - factor of increasing receiver function uncertainty
        
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
        if outstaid is None:
            print ('[%s] [MC_ISO_STA_INVERSION] inversion START' %datetime.now().isoformat().split('.')[0])
            if outdir is None:
                if restart:
                    try:
                        outdir  = self.attrs['mc_inv_run_sta_path']
                    except:
                        outdir  = os.path.dirname(self.filename)+'/mc_inv_run_sta_%s' %datetime.now().isoformat().split('.')[0]
                else:
                    outdir  = os.path.dirname(self.filename)+'/mc_inv_run_sta_%s' %datetime.now().isoformat().split('.')[0]
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            self.attrs.create(name = 'mc_inv_run_sta_path', data = outdir)
        else:
            restart     = False
        start_time_total= time.time()
        sta_grp         = self['sta_pts']
        # get the list for inversion
        if instafname is None:
            stalst  = list(sta_grp.keys())
        else:
            stalst  = []
            with open(instafname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    if sline[1] == '1':
                        stalst.append(sline[0])
        ista        = 0
        Nsta        = len(stalst)
        for staid in stalst:
            ista    += 1
            # check if result exists
            if restart:
                outfname    = outdir+'/mc_inv.'+staid+'.npz'
                if os.path.isfile(outfname):
                    print ('[%s] [MC_VTI_STA_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'SKIP upon exitence, station id: '+staid+' '+str(ista)+'/'+str(Nsta))
                    continue
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = inverse_solver.inverse_vprofile()
            # surface waves
            if ray:
                try:
                    indisp      = sta_grp[staid+'/disp_ph_ray'][()]
                    vpr.get_disp(indata = indisp, dtype='ph', wtype='ray')
                except KeyError:
                    print ('!!! WARNING: No Rayleigh phase dispersion data for , station id: '+ staid)
            # # # # if group:
            # # # #     try:
            # # # #         indisp      = sta_grp[staid+'/disp_gr_ray'].value
            # # # #         vpr.get_disp(indata=indisp, dtype='gr', wtype='ray')
            # # # #     except KeyError:
            # # # #         print ('!!! WARNING: No Rayleigh group dispersion data for , station id: '+ staid)
            if lov:
                try:
                    indisp      = sta_grp[staid+'/disp_ph_lov'][()]
                    vpr.get_disp(indata = indisp, dtype='ph', wtype='lov')
                except KeyError:
                    print ('!!! WARNING: No Love phase dispersion data for , station id: '+ staid)
            if (vpr.data.dispR.npper == 0 and vpr.data.dispR.ngper == 0) or vpr.data.dispL.npper == 0:
                print ('!!! WARNING: No dispersion data for , station id: '+ staid)
                continue
            
            # receiver functions
            delta   = sta_grp[staid].attrs['delta']
            indata  = sta_grp[staid+'/rf_data'][()]
            vpr.get_rf(indata = indata, delta = delta)
            #-----------------------------
            # initial model parameters
            #-----------------------------
            crtthk              = sta_grp[staid].attrs['crust_thk']
            sedthk              = sta_grp[staid].attrs['sediment_thk']
            topovalue           = sta_grp[staid].attrs['elevation_in_km']
            vpr.topo            = topovalue
            vti_numbp           = np.array([sedani, crtani, manani], dtype = int)
            vti_modtype         = [NOANISO, NOANISO, NOANISO]
            if sedani:
                vti_modtype[0]  = LAYERGAMMA
            if crtani:
                vti_modtype[1]  = LAYERGAMMA
            if manani:
                vti_modtype[2]  = LAYERGAMMA
            if use_ref:
                vsdata          = sta_grp[staid+'/reference_vs'][()]
                vpr.model.vtimod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], crtthk=crtthk, sedthk=sedthk,\
                            topovalue=topovalue, maxdepth=200., vp_water=vp_water)
            else:
                vpr.model.vtimod.parameterize_ak135(crtthk=crtthk, sedthk=sedthk, topovalue=topovalue, \
                    maxdepth=200., vp_water=vp_water, crt_depth = crt_depth, mantle_depth = mantle_depth,\
                    vti_numbp = vti_numbp, vti_modtype = vti_modtype)
            if crtstd is None:
                vpr.get_paraind(mtype='vti', crtthk = None)
            else:
                vpr.get_paraind(mtype='vti', crtthk = crtthk, crtstd = crtstd)
            if outstaid is not None:
                if staid != outstaid:
                    continue
                else:    
                    return vpr
            start_time_grd  = time.time()
            print ('[%s] [MC_VTI_STA_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'station id: '+staid+', '+str(ista)+'/'+str(Nsta))
            if parallel:
                vpr.mc_joint_inv_vti_mp(outdir=outdir, wdisp=wdisp, rffactor=rffactor, solver_type=solver_type, Ntotalruns=Ntotalruns, \
                    misfit_thresh=misfit_thresh, Nmodelthresh=Nmodelthresh, isconstrt=isconstrt, pfx=staid, verbose=verbose,\
                    step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_vti(outdir=outdir, wdisp=wdisp, rffactor=rffactor, solver_type=solver_type,\
                   isconstrt=isconstrt, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            end_time    = time.time()
            print ('[%s] [MC_VTI_STA_INVERSION] inversion DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', elasped time = %g'%(end_time - start_time_grd) + ' sec; total elasped time = %g' %(end_time - start_time_total))
        print ('[%s] [MC_VTI_STA_INVERSION] inversion ALL DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', total elasped time = %g' %(end_time - start_time_total))
        return