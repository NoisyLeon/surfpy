# -*- coding: utf-8 -*-
"""
hdf5 for noise eikonal tomography
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.pymcinv.invbase as invbase
import surfpy.pymcinv.inverse_solver as inverse_solver
import surfpy.pymcinv.isopost as isopost
import surfpy.pymcinv.vmodel as vmodel
import surfpy.pymcinv._model_funcs as _model_funcs

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

class isoh5(invbase.baseh5):
    
    def mc_inv(self, use_ref=False, ingrdfname=None, crtstd = None, wdisp = 1., cdist = 75.,rffactor = 40., phase=True, group=False,\
        outdir = None, restart = False, vp_water=1.5, isconstrt=True, step4uwalk=1500, numbrun=15000, subsize=1000, nprocess=None, parallel=True,\
        skipmask=True, Ntotalruns=10, misfit_thresh=1.0, Nmodelthresh=200, outlon=None, outlat=None, verbose = False, verbose2 = False):
        """
        Bayesian Monte Carlo inversion of geographical grid points
        ==================================================================================================================
        ::: input :::
        use_ref         - use reference input model or not(default = False, use ak135 instead)
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        wdisp           - weight of dispersion data (default = 1.0, only use dispersion data for inversion)
        cdist           - threshhold distance for loading rf data, only takes effect when wdisp < 1.0
        rffactor        - factor of increasing receiver function uncertainty
        
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
        if (outlon is None) or (outlat is None):
            print ('[%s] [MC_ISO_INVERSION] inversion START' %datetime.now().isoformat().split('.')[0])
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
            if outlon > 180.:
                outlon  -= 360.
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
        if wdisp != 1.:
            try:
                sta_grp = self['sta_pts']
                stlos   = self.attrs['stlos']
                stlas   = self.attrs['stlas']
                Nsta    = stlos.size
            except:
                print ('!!! Error ! wdisp must be 1.0 if station group NOT exists')
                return
        self.attrs.create(name = 'dispersion_dtype', data = dispdtype)
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
            # check if result exists
            if restart:
                outfname    = outdir+'/mc_inv.'+grd_id+'.npz'
                if os.path.isfile(outfname):
                    print ('[%s] [MC_ISO_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'SKIP upon exitence, grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
                    continue
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = inverse_solver.inverse_vprofile()
            if phase:
                try:
                    indisp      = grd_grp[grd_id+'/disp_ph_ray'][()]
                    vpr.get_disp(indata = indisp, dtype='ph', wtype='ray')
                except KeyError:
                    print ('!!! WARNING: No phase dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat))
            if group:
                try:
                    indisp      = grd_grp[grd_id+'/disp_gr_ray'].value
                    vpr.get_disp(indata=indisp, dtype='gr', wtype='ray')
                except KeyError:
                    print ('!!! WARNING: No group dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat))
            if vpr.data.dispR.npper == 0 and vpr.data.dispR.ngper == 0:
                print ('!!! WARNING: No dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat))
                continue
            # receiver functions
            if wdisp != 1.0:
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
            else:
                grd_wdisp   = 1.0
            #-----------------------------
            # initial model parameters
            #-----------------------------
            crtthk              = grd_grp[grd_id].attrs['crust_thk']
            sedthk              = grd_grp[grd_id].attrs['sediment_thk']
            topovalue           = grd_grp[grd_id].attrs['topo']
            if use_ref:
                vsdata          = grd_grp[grd_id+'/reference_vs'][()]
                vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], crtthk=crtthk, sedthk=sedthk,\
                            topovalue=topovalue, maxdepth=200., vp_water=vp_water)
            else:
                vpr.model.isomod.parameterize_ak135(crtthk=crtthk, sedthk=sedthk, topovalue=topovalue, maxdepth=200., vp_water=vp_water)
            if crtstd is None:
                vpr.get_paraind(crtthk = None)
            else:
                vpr.get_paraind(crtthk = crtthk, crtstd = crtstd)
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:    
                    return vpr
            start_time_grd  = time.time()
            print ('[%s] [MC_ISO_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
            if grd_wdisp != 1.0:
                print ('=== using rf data, station id: %s, stla = %g, stlo = %g, distance = %g km, wdisp = %g' %(staid, stla, stlo, distmin, grd_wdisp))
                grd_grp[grd_id].attrs.create(name = 'is_rf', data = True)
                grd_grp[grd_id].attrs.create(name = 'distance_rf', data = dist.min())
            else:
                grd_grp[grd_id].attrs.create(name = 'is_rf', data = False)
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=grd_wdisp, Ntotalruns=Ntotalruns, \
                    misfit_thresh=misfit_thresh, Nmodelthresh=Nmodelthresh, isconstrt=isconstrt, pfx=grd_id, verbose=verbose,\
                        verbose2 = verbose2, step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=grd_wdisp, \
                   isconstrt=isconstrt, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            end_time    = time.time()
            print ('[%s] [MC_ISO_INVERSION] inversion DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', elasped time = %g'%(end_time - start_time_grd) + ' sec; total elasped time = %g' %(end_time - start_time_total))
        print ('[%s] [MC_ISO_INVERSION] inversion ALL DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', total elasped time = %g' %(end_time - start_time_total))
        return
    
    def mc_inv_sta(self, use_ref=False, instafname = None, phase = True, group=False, outdir = None, restart = False,
        crtstd = None, wdisp = 0.2, rffactor = 40., vp_water=1.5, isconstrt=True, step4uwalk=1500, numbrun=15000, subsize=1000, \
        nprocess=None, parallel=True, Ntotalruns=10, misfit_thresh=1.0, Nmodelthresh=200, outstaid=None, verbose = False, verbose2 = False):
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
        if phase and group:
            dispdtype   = 'both'
        elif phase and not group:
            dispdtype   = 'ph'
        else:
            dispdtype   = 'gr'
        self.attrs.create(name = 'dispersion_dtype', data = dispdtype)
        ista        = 0
        Nsta        = len(stalst)
        for staid in stalst:
            ista    += 1
            # check if result exists
            if restart:
                outfname    = outdir+'/mc_inv.'+staid+'.npz'
                if os.path.isfile(outfname):
                    print ('[%s] [MC_ISO_STA_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'SKIP upon exitence, station id: '+staid+' '+str(ista)+'/'+str(Nsta))
                    continue
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = inverse_solver.inverse_vprofile()
            # surface waves
            if phase:
                try:
                    indisp      = sta_grp[staid+'/disp_ph_ray'][()]
                    vpr.get_disp(indata = indisp, dtype='ph', wtype='ray')
                except KeyError:
                    print ('!!! WARNING: No phase dispersion data for , station id: '+ staid)
            if group:
                try:
                    indisp      = sta_grp[staid+'/disp_gr_ray'].value
                    vpr.get_disp(indata=indisp, dtype='gr', wtype='ray')
                except KeyError:
                    print ('!!! WARNING: No group dispersion data for , station id: '+ staid)
            if vpr.data.dispR.npper == 0 and vpr.data.dispR.ngper == 0:
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
            if use_ref:
                vsdata          = sta_grp[staid+'/reference_vs'][()]
                vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], crtthk=crtthk, sedthk=sedthk,\
                            topovalue=topovalue, maxdepth=200., vp_water=vp_water)
            else:
                vpr.model.isomod.parameterize_ak135(crtthk=crtthk, sedthk=sedthk, topovalue=topovalue, \
                        maxdepth=200., vp_water=vp_water)
            if crtstd is None:
                vpr.get_paraind(crtthk = None)
            else:
                vpr.get_paraind(crtthk = crtthk, crtstd = crtstd)
            if outstaid is not None:
                if staid != outstaid:
                    continue
                else:    
                    return vpr
            start_time_grd  = time.time()
            print ('[%s] [MC_ISO_STA_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'station id: '+staid+', '+str(ista)+'/'+str(Nsta))
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, Ntotalruns=Ntotalruns, \
                    misfit_thresh=misfit_thresh, Nmodelthresh=Nmodelthresh, isconstrt=isconstrt, pfx=staid, verbose=verbose,\
                        verbose2 = verbose2, step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, \
                   isconstrt=isconstrt, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            end_time    = time.time()
            print ('[%s] [MC_ISO_STA_INVERSION] inversion DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', elasped time = %g'%(end_time - start_time_grd) + ' sec; total elasped time = %g' %(end_time - start_time_total))
        print ('[%s] [MC_ISO_STA_INVERSION] inversion ALL DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', total elasped time = %g' %(end_time - start_time_total))
        return
    
    def read_inv(self, datadir = None, ingrdfname=None, factor=1., thresh=0.5, stdfactor=2, avgqc=True, Nmax=None, Nmin=500, wtype='ray'):
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
        if datadir is None:
            datadir = self.attrs['mc_inv_run_path']
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
        mask_inv    = self.attrs['mask_inv']
        self._get_lon_lat_arr()
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
            ilat        = np.where(grd_lat == self.lats_inv)[0]
            ilon        = np.where(grd_lon == self.lons_inv)[0]
            invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
            datafname   = datadir+'/mc_data.'+grd_id+'.npz'
            if not (os.path.isfile(invfname) and os.path.isfile(datafname)):
                print ('!!! No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
                grp.attrs.create(name='mask', data = True)
                mask_inv[ilat, ilon]= True
                continue
            print ('=== Reading inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
            mask_inv[ilat, ilon]    = False
            #------------------------------------------
            # load inversion results
            #------------------------------------------
            topovalue               = grp.attrs['topo']
            postvpr                 = isopost.postvprofile(waterdepth = - topovalue, factor = factor, thresh = thresh, stdfactor = stdfactor)
            postvpr.read_data(infname = datafname)
            postvpr.read_inv(infname = invfname, verbose=False, Nmax=Nmax, Nmin=Nmin)
            postvpr.get_paraval()
            postvpr.run_avg_fwrd(wdisp=1.)
            postvpr.get_ensemble()
            postvpr.get_vs_std()
            if avgqc:
                if postvpr.avg_misfit > (postvpr.min_misfit*postvpr.factor + postvpr.thresh)*3.:
                    print ('--- Unstable inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
                    continue
            #------------------------------------------
            # store inversion results in the database
            #------------------------------------------
            grp.create_dataset(name = 'avg_paraval_'+wtype, data = postvpr.avg_paraval)
            grp.create_dataset(name = 'min_paraval_'+wtype, data = postvpr.min_paraval)
            grp.create_dataset(name = 'sem_paraval_'+wtype, data = postvpr.sem_paraval)
            grp.create_dataset(name = 'std_paraval_'+wtype, data = postvpr.std_paraval)
            # --- added 2019/01/16
            grp.create_dataset(name = 'z_ensemble_'+wtype, data = postvpr.z_ensemble)
            grp.create_dataset(name = 'vs_upper_bound_'+wtype, data = postvpr.vs_upper_bound)
            grp.create_dataset(name = 'vs_lower_bound_'+wtype, data = postvpr.vs_lower_bound)
            grp.create_dataset(name = 'vs_std_'+wtype, data = postvpr.vs_std)
            grp.create_dataset(name = 'vs_mean_'+wtype, data = postvpr.vs_mean)
            if ('disp_ph_'+wtype) in grp.keys():
                grp.create_dataset(name = 'avg_ph_'+wtype, data = postvpr.vprfwrd.data.dispR.pvelp)
                disp_min                = postvpr.disppre_ph[postvpr.ind_min, :]
                grp.create_dataset(name = 'min_ph_'+wtype, data = disp_min)
            if ('disp_gr_'+wtype) in grp.keys():
                grp.create_dataset(name = 'avg_gr_'+wtype, data = postvpr.vprfwrd.data.dispR.gvelp)
                disp_min                = postvpr.disppre_gr[postvpr.ind_min, :]
                grp.create_dataset(name = 'min_gr_'+wtype, data = disp_min)
            grp.attrs.create(name = 'average_mod_misfit_'+wtype, data = postvpr.vprfwrd.data.misfit)
            grp.attrs.create(name = 'min_misfit_'+wtype, data = postvpr.min_misfit)
            grp.attrs.create(name = 'mean_misfit_'+wtype, data = postvpr.mean_misfit)
        self.attrs.create(name='mask_inv_result', data = mask_inv)
        return
    
    def read_inv_sta(self, datadir = None, instafname=None, factor=1., thresh=0.5, stdfactor=2, avgqc=True,\
                     Nmax=None, Nmin=500, itype='ray_rf'):
        """
        read the inversion results in to data base
        ==================================================================================================================
        ::: input :::
        datadir     - data directory
        instafname  - input station id list file indicating the grid points for station based inversion
        factor      - factor to determine the threshhold value for selectingthe finalized model
        thresh      - threshhold value for selecting the finalized model
                        misfit < min_misfit*factor + thresh
        avgqc       - turn on quality control for average model or not
        Nmax        - required maximum number of accepted model
        Nmin        - required minimum number of accepted model
        ::: NOTE :::
        mask_sta array will be updated according to the existence of inversion results
        ==================================================================================================================
        """
        if datadir is None:
            datadir = self.attrs['mc_inv_run_sta_path']
        sta_grp     = self['sta_pts']
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
        mask_sta    = np.ones(Nsta, dtype = bool)
        stlos       = np.array([])
        stlas       = np.array([])
        self._get_lon_lat_arr()
        for staid in stalst:
            ista        += 1
            grp         = sta_grp[staid]
            stlas       = np.append(stlas, grp.attrs['stla'])
            stlos       = np.append(stlos, grp.attrs['stlo'])
            invfname    = datadir+'/mc_inv.'+staid+'.npz'
            datafname   = datadir+'/mc_data.'+staid+'.npz'
            if not (os.path.isfile(invfname) and os.path.isfile(datafname)):
                print ('!!! No inversion results for station id: '+staid+', '+str(ista)+'/'+str(Nsta))
                grp.attrs.create(name='mask', data = True)
                continue
            print ('=== Reading inversion results for station id: '+staid+', '+str(ista)+'/'+str(Nsta))
            #------------------------------------------
            # load inversion results
            #------------------------------------------
            topovalue               = grp.attrs['elevation_in_km']
            postvpr                 = isopost.postvprofile(waterdepth = - topovalue, factor = factor, thresh = thresh, stdfactor = stdfactor)
            postvpr.read_data(infname = datafname)
            postvpr.read_inv(infname = invfname, verbose=False, Nmax=Nmax, Nmin=Nmin)
            postvpr.get_paraval()
            postvpr.run_avg_fwrd(wdisp=1.)
            postvpr.get_ensemble()
            postvpr.get_vs_std()
            if avgqc:
                if postvpr.avg_misfit > (postvpr.min_misfit*postvpr.factor + postvpr.thresh)*3.:
                    print ('--- Unstable inversion results for station id: '+staid+', '+str(ista)+'/'+str(Nsta))
                    continue
            mask_sta[ista-1]        = False
            grp.attrs.create(name='mask', data = False)
            #------------------------------------------
            # store inversion results in the database
            #------------------------------------------
            grp.create_dataset(name = 'avg_paraval_'+itype, data = postvpr.avg_paraval)
            grp.create_dataset(name = 'min_paraval_'+itype, data = postvpr.min_paraval)
            grp.create_dataset(name = 'sem_paraval_'+itype, data = postvpr.sem_paraval)
            grp.create_dataset(name = 'std_paraval_'+itype, data = postvpr.std_paraval)
            # --- added 2019/01/16
            grp.create_dataset(name = 'z_ensemble_'+itype, data = postvpr.z_ensemble)
            grp.create_dataset(name = 'vs_upper_bound_'+itype, data = postvpr.vs_upper_bound)
            grp.create_dataset(name = 'vs_lower_bound_'+itype, data = postvpr.vs_lower_bound)
            grp.create_dataset(name = 'vs_std_'+itype, data = postvpr.vs_std)
            grp.create_dataset(name = 'vs_mean_'+itype, data = postvpr.vs_mean)
            if ('disp_ph_'+itype) in grp.keys():
                grp.create_dataset(name = 'avg_ph_'+itype, data = postvpr.vprfwrd.data.dispR.pvelp)
                disp_min                = postvpr.disppre_ph[postvpr.ind_min, :]
                grp.create_dataset(name = 'min_ph_'+itype, data = disp_min)
            if ('disp_gr_'+itype) in grp.keys():
                grp.create_dataset(name = 'avg_gr_'+itype, data = postvpr.vprfwrd.data.dispR.gvelp)
                disp_min                = postvpr.disppre_gr[postvpr.ind_min, :]
                grp.create_dataset(name = 'min_gr_'+itype, data = disp_min)
            grp.attrs.create(name = 'average_mod_misfit_'+itype, data = postvpr.vprfwrd.data.misfit)
            grp.attrs.create(name = 'min_misfit_'+itype, data = postvpr.min_misfit)
            grp.attrs.create(name = 'mean_misfit_'+itype, data = postvpr.mean_misfit)
        self.attrs.create(name = 'stlos', data = stlos)
        self.attrs.create(name = 'stlas', data = stlas)
        self.attrs.create(name = 'mask_sta', data = mask_sta)
        return
    
    def merge_sta_grd(self, cdist = 75., itype_grd = 'ray', itype_sta = 'ray_rf', itype = 'ray_rf'):
        igrd        = 0
        grd_grps    = self['grd_pts']
        grdlst      = list(grd_grps.keys())
        Ngrd        = len(grdlst)
        mask_inv    = self.attrs['mask_inv']
        sta_grps    = self['sta_pts']
        stlos       = self.attrs['stlos']
        stlas       = self.attrs['stlas']
        Nsta        = stlos.size
        self._get_lon_lat_arr()
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat         = float(split_id[1])
            igrd            += 1
            grp             = grd_grps[grd_id]
            topo_grd        = grp.attrs['topo']
            is_merge        = True
            # station group
            az, baz, dist   = geodist.inv(grd_lon*np.ones(Nsta), grd_lat*np.ones(Nsta), stlos, stlas)
            dist            /= 1000.
            if dist.min() >= cdist:
                is_merge    = False
            else:
                ind_min     = dist.argmin()
                staid       = list(sta_grps.keys())[ind_min]
                grp_sta     = sta_grps[staid]
                delta       = grp_sta.attrs['delta']
                stla        = grp_sta.attrs['stla']
                stlo        = grp_sta.attrs['stlo']
                topo_sta    = grp_sta.attrs['elevation_in_km']
                distmin     = dist.min()
            #============
            # merge
            #============
            # grid point results
            min_paraval_grd     = grp['min_paraval_'+itype_grd][()]
            avg_paraval_grd     = grp['avg_paraval_'+itype_grd][()]
            sem_paraval_grd     = grp['sem_paraval_'+itype_grd][()]
            std_paraval_grd     = grp['std_paraval_'+itype_grd][()]
            vs_upper_bound_grd  = grp['vs_upper_bound_'+itype_grd][()]
            vs_lower_bound_grd  = grp['vs_lower_bound_'+itype_grd][()]
            vs_std_grd          = grp['vs_std_'+itype_grd][()]
            vs_mean_grd         = grp['vs_mean_'+itype_grd][()]
            if is_merge:
                # print (grd_lat, stla, grd_lon, stlo, distmin)
                # station based results
                min_paraval_sta     = grp_sta['min_paraval_'+itype_sta][()]
                avg_paraval_sta     = grp_sta['avg_paraval_'+itype_sta][()]
                sem_paraval_sta     = grp_sta['sem_paraval_'+itype_sta][()]
                std_paraval_sta     = grp_sta['std_paraval_'+itype_sta][()]
                vs_upper_bound_sta  = grp_sta['vs_upper_bound_'+itype_sta][()]
                vs_lower_bound_sta  = grp_sta['vs_lower_bound_'+itype_sta][()]
                vs_std_sta          = grp_sta['vs_std_'+itype_sta][()]
                vs_mean_sta         = grp_sta['vs_mean_'+itype_sta][()]
                # merged results
                weight_grd          = distmin/cdist
                min_paraval_out     = min_paraval_grd *weight_grd + (1. - weight_grd) * min_paraval_sta
                avg_paraval_out     = avg_paraval_grd *weight_grd + (1. - weight_grd) * avg_paraval_sta
                #=================
                # min depth arrays
                #=================
                # min moho
                moho_depth_grd      = min_paraval_grd[-1] + min_paraval_grd[-2] - topo_grd
                moho_depth_sta      = min_paraval_sta[-1] + min_paraval_sta[-2] - topo_sta
                moho_depth_out      = moho_depth_grd *weight_grd + (1. - weight_grd) * moho_depth_sta
                # min sediment
                sed_depth_grd       = min_paraval_grd[-2] - topo_grd
                sed_depth_sta       = min_paraval_sta[-2] - topo_sta
                sed_depth_out       = sed_depth_grd *weight_grd + (1. - weight_grd) * sed_depth_sta
                # # # tmp1= min_paraval_out[-2]
                # # # tmp2= min_paraval_out[-1]
                if (sed_depth_out + topo_grd) > 0.:
                    min_paraval_out[-2] = sed_depth_out + topo_grd
                min_paraval_out[-1] = moho_depth_out - sed_depth_out
                #=================
                # avg depth arrays
                #=================
                # min moho
                moho_depth_grd      = avg_paraval_grd[-1] + avg_paraval_grd[-2] - topo_grd
                moho_depth_sta      = avg_paraval_sta[-1] + avg_paraval_sta[-2] - topo_sta
                moho_depth_out      = moho_depth_grd *weight_grd + (1. - weight_grd) * moho_depth_sta
                # min sediment
                sed_depth_grd       = avg_paraval_grd[-2] - topo_grd
                sed_depth_sta       = avg_paraval_sta[-2] - topo_sta
                sed_depth_out       = sed_depth_grd *weight_grd + (1. - weight_grd) * sed_depth_sta
                # # # tmp3= avg_paraval_out[-2]
                # # # tmp4= avg_paraval_out[-1]
                if (sed_depth_out + topo_grd) > 0.:
                    avg_paraval_out[-2] = sed_depth_out + topo_grd
                avg_paraval_out[-1] = moho_depth_out - sed_depth_out
                # print (tmp1, min_paraval_out[-2], tmp2, min_paraval_out[-1], tmp3, avg_paraval_out[-2], tmp4, avg_paraval_out[-1])
                # other arrays
                sem_paraval_out     = sem_paraval_grd *weight_grd + (1. - weight_grd) * sem_paraval_sta
                std_paraval_out     = std_paraval_grd *weight_grd + (1. - weight_grd) * std_paraval_sta
                vs_upper_bound_out  = vs_upper_bound_grd *weight_grd + (1. - weight_grd) * vs_upper_bound_sta
                vs_lower_bound_out  = vs_lower_bound_grd *weight_grd + (1. - weight_grd) * vs_lower_bound_sta
                vs_std_out          = vs_std_grd *weight_grd + (1. - weight_grd) * vs_std_sta
                vs_mean_out         = vs_mean_grd *weight_grd + (1. - weight_grd) * vs_mean_sta
            else:
                min_paraval_out     = min_paraval_grd
                avg_paraval_out     = avg_paraval_grd
                sem_paraval_out     = sem_paraval_grd
                std_paraval_out     = std_paraval_grd
                vs_upper_bound_out  = vs_upper_bound_grd
                vs_lower_bound_out  = vs_lower_bound_grd
                vs_std_out          = vs_std_grd
                vs_mean_out         = vs_mean_grd
            # store merged data
            grp.create_dataset(name = 'avg_paraval_'+itype, data = avg_paraval_out)
            grp.create_dataset(name = 'min_paraval_'+itype, data = min_paraval_out)
            grp.create_dataset(name = 'sem_paraval_'+itype, data = sem_paraval_out)
            grp.create_dataset(name = 'std_paraval_'+itype, data = std_paraval_out)
            # --- added 2019/01/16
            grp.create_dataset(name = 'vs_upper_bound_'+itype, data = vs_upper_bound_out)
            grp.create_dataset(name = 'vs_lower_bound_'+itype, data = vs_lower_bound_out)
            grp.create_dataset(name = 'vs_std_'+itype, data = vs_std_out)
            grp.create_dataset(name = 'vs_mean_'+itype, data = vs_mean_out)
        return
    
    def get_paraval(self, pindex, dtype = 'min', itype = 'ray', ingrdfname = None, isthk = False, depth = 5., depthavg = 0.):
        """
        get the data for the model parameter
        ==================================================================================================================
        ::: input :::
        pindex          - parameter index in the paraval array
                            0 ~ 13      : 
                            moho        : model parameters from paraval arrays
                            vs_std_ray  : vs_std from the model ensemble, dtype does NOT take effect
                            other attrs : topo, crust_thk, sediment_thk, mean_misfit_ray, min_misfit_ray
        dtype           - data type:
                            avg - average model
                            min - minimum misfit model
                            sem - uncertainties (standard error of the mean)
        itype           - inversion type
                            'ray'   - isotropic inversion using Rayleigh wave
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        isthk           - flag indicating if the parameter is thickness or not
                            if so, the returned values are actually depth by subtracting topography
                                this is useful for smoothing
        depth, depthavg - only takes effect when pindex == 'vs_std_ray'
        ==================================================================================================================
        """
        self._get_lon_lat_arr()
        data        = -999. * np.ones((self.Nlat_inv, self.Nlon_inv), dtype = np.float64)
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
                print ('ERROR ', grd_id)
                continue
            grd_lat     = float(split_id[1])
            igrd        += 1
            grp         = grd_grp[grd_id]
            try:
                ind_lon = np.where(grd_lon==self.lons_inv)[0][0]
                ind_lat = np.where(grd_lat==self.lats_inv)[0][0]
            except IndexError:
                continue
            try:
                paraval                     = grp[dtype+'_paraval_'+itype][()]
            except KeyError:
                print ('WARNING: no data at grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
                continue
            if pindex =='moho':
                # get crustal thickness (including sediments)
                if dtype != 'std' and dtype != 'sem':
                    data[ind_lat, ind_lon]  = paraval[-1] + paraval[-2]
                else:
                    data[ind_lat, ind_lon]  = paraval[-1] * 1.5  #  
            elif pindex == 'vs_std_ray':
                unArr                       = grp['vs_std_ray'][()]
                zArr                        = grp['z_ensemble_ray'][()]
                ind_un                      = (zArr <= (depth + depthavg))*(zArr >= (depth - depthavg))
                data[ind_lat, ind_lon]      = unArr[ind_un].mean() 
            else:
                if isinstance(pindex, int):
                    data[ind_lat, ind_lon]  = paraval[pindex]
                else:
                    try:
                        data[ind_lat, ind_lon]  = grp.attrs[pindex]
                    except:
                        pass
            # convert thickness to depth
            if isthk:
                topovalue                   = grp.attrs['topo']
                data[ind_lat, ind_lon]      -= topovalue
        return data
    
    def get_paraval_sta(self, pindex, dtype = 'min', itype = 'ray_rf', instafname = None, isthk = False, depth = 5., depthavg = 0.):
        """
        get the data for the model parameter
        ==================================================================================================================
        ::: input :::
        pindex          - parameter index in the paraval array
                            0 ~ 13      : 
                            moho        : model parameters from paraval arrays
                            vs_std_ray  : vs_std from the model ensemble, dtype does NOT take effect
                            other attrs : topo, crust_thk, sediment_thk, mean_misfit_ray, min_misfit_ray
        dtype           - data type:
                            avg - average model
                            min - minimum misfit model
                            sem - uncertainties (standard error of the mean)
        itype           - inversion type
                            'ray'   - isotropic inversion using Rayleigh wave
        instafname      - input station id list file indicating the grid points for station based inversion
        isthk           - flag indicating if the parameter is thickness or not
                            if so, the returned values are actually depth by subtracting topography
                                this is useful for smoothing
        depth, depthavg - only takes effect when pindex == 'vs_std_ray'
        ==================================================================================================================
        """
        self._get_lon_lat_arr()
        index_sta   = np.logical_not(self.mask_sta)
        stlos       = self.stlos[index_sta]
        data        = -999. * np.ones(stlos.size, dtype = np.float64)
        sta_grp     = self['sta_pts']
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
            grp         = sta_grp[staid]
            if grp.attrs['mask']:
                continue
            ista        += 1
            paraval     = grp[dtype+'_paraval_'+itype][()]
            if pindex =='moho':
                # get crustal thickness (including sediments)
                if dtype != 'std' and dtype != 'sem':
                    data[ista-1]    = paraval[-1] + paraval[-2]
                else:
                    data[ista-1]    = paraval[-1] * 1.5  #  
            else:
                if isinstance(pindex, int):
                    data[ista-1]    = paraval[pindex]
                else:
                    try:
                        data[ista-1]= grp.attrs[pindex]
                    except:
                        pass
            # convert thickness to depth
            if isthk:
                topovalue           = grp.attrs['topo']
                data[ista-1]        -= topovalue
        return data
    
    def update_mask(self, cdist = 100.):
        self._get_lon_lat_arr()
        index_sta       = np.logical_not(self.mask_sta)
        stlos           = self.stlos[index_sta]
        stlas           = self.stlas[index_sta]
        Nsta            = stlos.size
        # update mask
        mask            = np.ones((self.Nlat, self.Nlon), dtype = bool)
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                tmplon                  = self.lons[ilon]
                tmplat                  = self.lats[ilat]
                az, baz, dist           = geodist.inv(tmplon*np.ones(Nsta), tmplat*np.ones(Nsta), stlos, stlas) 
                dist                    /= 1000.
                if dist.min() < cdist:
                    mask[ilat, ilon]    = False
        self.attrs.create(name = 'mask', data = mask)
    
    def update_mask_inv(self, ingrdfname=None, dtype = 'min', itype = 'ray'):
        self._get_lon_lat_arr()
        mask_inv    = np.ones((self.Nlat_inv, self.Nlon_inv), dtype = bool)
        # # # mask_inv_old= self.attrs['mask_inv_result']
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
                print ('ERROR ', grd_id)
                continue
            grd_lat     = float(split_id[1])
            igrd        += 1
            grp         = grd_grp[grd_id]
            try:
                ind_lon = np.where(grd_lon==self.lons_inv)[0][0]
                ind_lat = np.where(grd_lat==self.lats_inv)[0][0]
            except IndexError:
                continue
            try:
                paraval                     = grp[dtype+'_paraval_'+itype][()]
            except KeyError:
                print ('WARNING: no data at grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
                continue
            mask_inv[ind_lat, ind_lon]      = False
        self.attrs.create(name = 'mask_inv_result', data = mask_inv)
        return
        
        
    
    def paraval_arrays(self, igrdsta = 1, cdist = 100., dtype = 'min', itype = 'ray', sigma=1,\
            gsigma = 50., depth = 5., depthavg = 0., verbose=False):
        """
        get the paraval arrays and store them in the database
        ==========================================================================================
        ::: input :::
        igrdsta     - flag indicating 1: grid point inversion, 2: station point inversion
        cdist       - distance for mask of the model, only takes effects for igrdsta = 2
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        itype       - inversion type
                        'ray'   - isotropic inversion using Rayleigh wave
                        'ray_rf'- isotropic inversion using Rayleigh wave and receiver functions
                        'vti'   - VTI intersion using Rayleigh and Love waves
        sigma       - total number of smooth iterations
        gsigma      - sigma for Gaussian smoothing (unit - km)
        dlon/dlat   - longitude/latitude interval for interpolation
        ==========================================================================================
        """
        self._get_lon_lat_arr()
        topo                = self['topo'][()]
        if igrdsta == 1:
            index_inv       = np.logical_not(self.attrs['mask_inv_result'])
            lons_inv        = self.lonArr_inv[index_inv]
            lats_inv        = self.latArr_inv[index_inv]
            if not np.all(self.attrs['mask_inv'] == self.attrs['mask_inv_result']):
                print ('!!! WARNING: mask_inv not equal to mask_inv_result')
        else:
            if itype == 'ray':
                print ('!!! Station based inversion changed from %s to ray_rf' %itype)
                itype       = 'ray_rf'
            index_sta       = np.logical_not(self.mask_sta)
            stlos           = self.stlos[index_sta]
            stlas           = self.stlas[index_sta]
        index               = np.logical_not(self.attrs['mask'])
        lons                = self.lonArr[index]
        lats                = self.latArr[index]
        grp                 = self.require_group( name = dtype+'_paraval_'+itype )
        for pindex in range(13):
            if pindex == 11:
                if igrdsta == 1:
                    data_inv= self.get_paraval(pindex = pindex, dtype = dtype, itype = itype,\
                                    isthk = True, depth = depth, depthavg = depthavg)
                    if np.any(data_inv[index_inv] < -100.):
                        print (data_inv[index_inv].min(), data_inv[index_inv].max(), data_inv[index_inv].mean())
                        raise ValueError('!!! Error in inverted data!')
                else:
                    data_inv= self.get_paraval_sta(pindex = pindex, dtype = dtype, itype = itype, \
                                    isthk = True, depth = depth, depthavg = depthavg)
                # interpolation
                gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'sedi_thk', evid = 'MCINV')
                if igrdsta == 1:
                    gridder.read_array(inlons = lons_inv, inlats = lats_inv, inzarr = data_inv[index_inv])
                else:
                    gridder.read_array(inlons = stlos, inlats = stlas, inzarr = data_inv)
                gridder.interp_surface(do_blockmedian = True)
                data        = gridder.Zarr.copy()
                # smoothing
                smoothgrder = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'sedi_thk', evid = 'MCINV')
                smoothgrder.read_array(inlons = lons, inlats = lats, inzarr = data[index])
                outfname    = 'smooth_paraval.lst'
                smoothgrder.gauss_smoothing(workingdir = 'mc_gauss_smooth', outfname = outfname, width = gsigma)
                data_smooth = smoothgrder.Zarr.copy()
                #=================================================
                # convert sediment depth to sediment thickness
                #=================================================
                data        += topo
                data_smooth += topo
                sedi        = data.copy()
                sedi_smooth = data_smooth.copy()
            elif pindex == 12:
                if igrdsta == 1:
                    data_inv= self.get_paraval(pindex = 'moho', dtype = dtype, itype = itype,\
                                    isthk = True, depth = depth, depthavg = depthavg)
                    if np.any(data_inv[index_inv] < -100.):
                        raise ValueError('!!! Error in inverted data!')
                else:
                    data_inv= self.get_paraval_sta(pindex = 'moho', dtype = dtype, itype = itype, \
                                    isthk = True, depth = depth, depthavg = depthavg)
                # interpolation
                gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'crst_thk', evid = 'MCINV')
                if igrdsta == 1:
                    gridder.read_array(inlons = lons_inv, inlats = lats_inv, inzarr = data_inv[index_inv])
                else:
                    gridder.read_array(inlons = stlos, inlats = stlas, inzarr = data_inv)
                gridder.interp_surface(do_blockmedian = True)
                data        = gridder.Zarr.copy()
                # smoothing
                smoothgrder = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'crst_thk', evid = 'MCINV')
                smoothgrder.read_array(inlons = lons, inlats = lats, inzarr = data[index])
                outfname    = 'smooth_paraval.lst'
                smoothgrder.gauss_smoothing(workingdir = 'mc_gauss_smooth', outfname = outfname, width = gsigma)
                data_smooth = smoothgrder.Zarr.copy()
                #===============================================================
                # convert moho depth to crustal thickness (excluding sediments)
                #===============================================================
                data        += topo
                data_smooth += topo
                data        -= sedi
                data_smooth -= sedi_smooth
            else:
                if igrdsta == 1:
                    data_inv= self.get_paraval(pindex = pindex, dtype = dtype, itype = itype,\
                                    isthk = False, depth = depth, depthavg = depthavg)
                    if np.any(data_inv[index_inv] < -100.):
                        print (data_inv[index_inv].min(), data_inv[index_inv].max(), data_inv[index_inv].mean())
                        # print 
                        raise ValueError('!!! Error in inverted data!')
                else:
                    data_inv= self.get_paraval_sta(pindex = pindex, dtype = dtype, itype = itype, \
                                    isthk = False, depth = depth, depthavg = depthavg)
                # interpolation
                gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'paraval_'+str(pindex), evid = 'MCINV')
                if igrdsta == 1:
                    gridder.read_array(inlons = lons_inv, inlats = lats_inv, inzarr = data_inv[index_inv])
                else:
                    gridder.read_array(inlons = stlos, inlats = stlas, inzarr = data_inv)
                gridder.interp_surface(do_blockmedian = True)
                data        = gridder.Zarr.copy()
                # smoothing
                smoothgrder = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'paraval_'+str(pindex), evid = 'MCINV')
                smoothgrder.read_array(inlons = lons, inlats = lats, inzarr = data[index])
                outfname    = 'smooth_paraval.lst'
                smoothgrder.gauss_smoothing(workingdir = 'mc_gauss_smooth', outfname = outfname, width = gsigma)
                data_smooth = smoothgrder.Zarr.copy()
            # store data
            grp.create_dataset(name = str(pindex)+'_org', data = data)
            grp.create_dataset(name = str(pindex)+'_smooth', data = data_smooth)
        return 
    
    def construct_3d(self, dtype = 'avg', itype = 'ray', is_smooth = True, maxdepth = 200., dz = 0.1):
        """construct 3D vs array
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
        grp         = self[dtype+'_paraval_'+itype]
        topo        = self['topo'][()]
        self._get_lon_lat_arr()
        if self.latArr.shape != grp['0_org'][()].shape:
            raise ValueError('incompatible paraval data with lonArr/latArr !')
        Nz          = int(maxdepth/dz) + 1
        zArr        = np.arange(Nz)*dz
        vs3d        = np.zeros((self.latArr.shape[0], self.latArr.shape[1], Nz))
        Ntotal      = self.Nlat*self.Nlon
        N0          = int(Ntotal/100.)
        i           = 0
        j           = 0
        mask        = self.attrs['mask']
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                i                   += 1
                if np.floor(i/N0) > j:
                    print ('=== Constructing 3d model:',j,' % finished')
                    j               += 1
                paraval             = np.zeros(13, dtype = np.float64)
                topovalue           = topo[ilat, ilon]
                for pindex in range(13):
                    if is_smooth:
                        data        = grp[str(pindex)+'_smooth'][()]
                    else:
                        data        = grp[str(pindex)+'_org'][()]
                    paraval[pindex] = data[ilat, ilon]
                vel_mod             = vmodel.model1d()
                if mask[ilat, ilon]:
                    continue
                if topovalue < 0.:
                    vel_mod.get_para_model(paraval = paraval, waterdepth = -topovalue, vpwater = 1.5, nmod = 4, \
                        numbp = np.array([1, 2, 4, 5]), mtype = np.array([5, 4, 2, 2]), vpvs = np.array([0, 2., 1.75, 1.75]), maxdepth = 200.)
                else:
                    vel_mod.get_para_model(paraval = paraval)
                zArr_in, VsvArr_in  = vel_mod.get_grid_mod()
                if topovalue > 0.:
                    zArr_in         = zArr_in - topovalue
                # interpolation
                vs_interp           = np.interp(zArr, xp = zArr_in, fp = VsvArr_in)
                vs3d[ilat, ilon, :] = vs_interp[:]                
        if is_smooth:
            grp.create_dataset(name = 'vs_smooth', data = vs3d)
            grp.create_dataset(name = 'z_smooth', data = zArr)
        else:
            grp.create_dataset(name = 'vs_org', data = vs3d)
            grp.create_dataset(name = 'z_org', data = zArr)
        return
    
    def convert_to_vts(self, outdir, dtype='avg', itype='ray_rf', is_smooth=True, pfx='',\
                unit=True, depthavg=3., dz=1., zfactor =1.):
        """ Convert Vs model to vts format for plotting with Paraview, VisIt
        ========================================================================================
        ::: input :::
        outdir      - output directory
        modelname   - modelname ('dvsv', 'dvsh', 'dvp', 'drho')
        pfx         - prefix of output files
        unit        - output unit sphere(radius=1) or not
        ========================================================================================
        """
        grp         = self[dtype+'_paraval_'+itype]
        if is_smooth:
            vs3d    = grp['vs_smooth'][()]
            zArr    = grp['z_smooth'][()]
            data_str= dtype + '_smooth'
        else:
            vs3d    = grp['vs_org'][()]
            zArr    = grp['z_org'][()]
            data_str= dtype + '_org'
        
        if depthavg>0.:
            vs3d    = _get_avg_vs3d(zArr, vs3d, depthavg)
        print ('End depth averaging')
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
        print ('End downsampling')

        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        from tvtk.api import tvtk, write_data
        if unit:
            Rref    = 6471.
        else:
            Rref    = 1.
        self._get_lon_lat_arr()
        # convert geographycal coordinate to spherichal coordinate
        theta       = (90.0 - self.lats)*np.pi/180.
        phi         = self.lons*np.pi/180.
        radius      = Rref - zArr/zfactor
        theta, phi, radius  = np.meshgrid(theta, phi, radius, indexing='ij')
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
        pts.shape   = int(pts.size / 3), 3
        sgrid       = tvtk.StructuredGrid(dimensions=dims, points=pts)
        sgrid.point_data.scalars        = (vs3d).ravel(order='F')
        sgrid.point_data.scalars.name   = 'Vs'
        outfname    = outdir+'/'+pfx+'Vs_'+data_str+'.vts'
        write_data(sgrid, outfname)
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
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection == 'merc':
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                      urcrnrlon=maxlon, lat_ts=0, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,1,1,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,1,1,1], fontsize=15)
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
            
        try:
            coasts = m.drawcoastlines(zorder=100,color= 'k',linewidth=0.0000)
            # Exact the paths from coasts
            coasts_paths = coasts.get_paths()
            poly_stop = 50
            for ipoly in range(len(coasts_paths)):
                # print (ipoly)
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
        # m.fillcontinents(color='grey',lake_color='aqua')
        # m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
        # m.fillcontinents(color='coral',lake_color='aqua')
        # m.drawcountries(linewidth=1.)
        return m
    
    def plot_paraval(self, pindex, is_smooth=True, dtype='avg', itype='ray', sigma=1, gsigma = 50., \
            ingrdfname=None, isthk=False, shpfx=None, outfname=None, outimg=None, clabel='', title='', cmap='surf', \
                projection='lambert', lonplt=[], latplt=[], plotfault = True,\
                    vmin=None, vmax=None, showfig=True, depth = 5., depthavg = 0., width=-1.):
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
        mask        = self.attrs['mask']
        if pindex == 'moho':
            topoArr     = self['topo'][()]
            if is_smooth:
                data    = self[dtype+'_paraval_'+itype+'/12_smooth'][()]\
                            + self[dtype+'_paraval_'+itype+'/11_smooth'][()] - topoArr
            else:
                data    = self[dtype+'_paraval_'+itype+'/12_org'][()]\
                            + self[dtype+'_paraval_'+itype+'/11_org'][()] - topoArr
        elif pindex == 'crust_thk':
            if is_smooth:
                data    = self[dtype+'_paraval_'+itype+'/12_smooth'][()]\
                            + self[dtype+'_paraval_'+itype+'/11_smooth'][()] 
            else:
                data    = self[dtype+'_paraval_'+itype+'/12_org'][()]\
                            + self[dtype+'_paraval_'+itype+'/11_org'][()] 
        else:
            if is_smooth:
                data    =  self[dtype+'_paraval_'+itype+'/%d_smooth' %pindex][()]
            else:
                data    =  self[dtype+'_paraval_'+itype+'/%d_org' %pindex][()]
        
        
        # smoothing
        if width > 0.:
            gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = 0., evla = 0., fieldtype = 'paraval', evid = 'plt')
            gridder.read_array(inlons = self.lonArr[np.logical_not(mask)], inlats = self.latArr[np.logical_not(mask)], inzarr = data[np.logical_not(mask)])
            outfname    = 'plt_paraval.lst'
            prefix      = 'plt_paraval_'
            gridder.gauss_smoothing(workingdir = './temp_plt', outfname = outfname, width = width)
            data[:]     = gridder.Zarr
        
        mdata       = ma.masked_array(data, mask=mask )
        try:
            import pycpt
            if cmap == 'panoply':
                is_reverse = True
            else:
                is_reverse = False
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
            # cmap.set_bad('silver', alpha = 0.)
            if is_reverse:
                cmap = cmap.reversed()
            cmap.set_bad('silver', alpha = 0.)
        except:
            pass
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        if vmin ==45. and vmax == 55.:
            cb              = m.colorbar(im, location='bottom', size="5%", pad='2%', ticks=[45, 47, 49, 51, 53, 55.])
        elif vmin==42. and vmax == 54.:
            cb              = m.colorbar(im, location='bottom', size="5%", pad='2%', ticks=[42, 44, 46, 48, 50, 52, 54.])
        elif vmin==42. and vmax == 56.:
            cb              = m.colorbar(im, location='bottom', size="5%", pad='2%', ticks=[42, 44, 46, 48, 50, 52, 54., 56.])
        else:
            if projection == 'lambert':
                cb              = m.colorbar(im, location='bottom', size="5%", pad='2%', ticks=[10., 15, 20, 25, 30, 35, 40, 45])
            else:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
                # cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            # cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[10., 15, 20, 25, 30, 35, 40])
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=25)
        cb.set_alpha(1)
        cb.draw_all()
        
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)
        
        if plotfault:
            if projection == 'lambert':
                shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
                m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
                m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
            else:
                shapefname  = '/home/lili/code/gem-global-active-faults/shapefile/gem_active_faults'
                # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
                m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
            
            # shapefname  = '/home/lili/data_mongo/fault_shp/doc-line'
            # # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black')
            # m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey')
            
        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        return
    
    def plot_crust1(self, fname = None, vmin=20., vmax=60., clabel='Crustal thickness (km)',
                    cmap='RdYlBu',showfig=True, projection='lambert', plotfault=True):
        if fname is None:
            fname   = map_path+'/crsthk.xyz'
        if not os.path.isfile(fname):
            raise ValueError('!!! reference crust thickness file not exists!')
        
        inArr       = np.loadtxt(fname)
        lonArr      = inArr[:, 0]
        lonArr      = lonArr.reshape(int(lonArr.size/360), 360)
        latArr      = inArr[:, 1]
        latArr      = latArr.reshape(int(latArr.size/360), 360)
        depthArr    = inArr[:, 2]
        depthArr    = depthArr.reshape(int(depthArr.size/360), 360)
        
        try:
            import pycpt
            if cmap == 'panoply':
                is_reverse = True
            else:
                is_reverse = False
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
            
            if is_reverse:
                cmap = cmap.reversed()
            cmap.set_bad('silver', alpha = 0.)
        except:
            pass
        m               = self._get_basemap(projection=projection)
        x, y            = m(lonArr, latArr)
        im              = m.pcolormesh(x, y, depthArr, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)

        if vmin ==45. and vmax == 55.:
            cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[45, 47, 49, 51, 53, 55.])
        elif vmin==42. and vmax == 54.:
            cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[42, 44, 46, 48, 50, 52, 54.])
        elif vmin==42. and vmax == 56.:
            cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[42, 44, 46, 48, 50, 52, 54., 56.])
        else:
            
            cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
            # cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[10., 15, 20, 25, 30, 35, 40])
            
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=25)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        
        #############################
        if plotfault:
            # shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
            # m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
            # m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')

            shapefname  = '/home/lili/code/gem-global-active-faults/shapefile/gem_active_faults'
            # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
            m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        
        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)

        if showfig:
            plt.show()
    
    def plot_horizontal(self, depth, evdepavg = 5., verlats = [], verlons = [], depthb=None, depthavg=None, dtype='avg', itype = 'ray',
        is_smooth=True, shpfx=None, clabel='', title='', cmap='surf', projection='lambert',  vmin=None, vmax=None, \
            lonplt=[], latplt=[], incat=None, plotevents=False, showfig=True, outfname=None, plotfault=True, plotslab=False,
        plottecto = False, vprlons = [], vprlats = [], plotcontour=False):
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
        self._get_lon_lat_arr()
        topoArr     = self['topo'][()]
        if is_smooth:
            mohoArr = self[dtype+'_paraval_'+itype+'/12_smooth'][()]\
                        + self[dtype+'_paraval_'+itype+'/11_smooth'][()] - topoArr
        else:
            mohoArr = self[dtype+'_paraval_'+itype+'/12_org'][()]\
                        + self[dtype+'_paraval_'+itype+'/11_org'][()] - topoArr
        grp         = self[dtype+'_paraval_'+itype]
        if is_smooth:
            vs3d    = grp['vs_smooth'][()]
            zArr    = grp['z_smooth'][()]
        else:
            vs3d    = grp['vs_org'][()]
            zArr    = grp['z_org'][()]
        if depthb is not None:
            if depthb < depth:
                raise ValueError('depthb should be larger than depth!')
            index   = np.where((zArr >= depth)*(zArr <= depthb) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        elif depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth + depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis = 2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print ('depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km')
                return
            depth       = zArr[index]
            vs_plt      = vs3d[:, :, index]
        mask        = self.attrs['mask']
        #
        mask        += (vs3d[:, :, zArr == depth - 1.0] == 0.).reshape((self.Nlat, self.Nlon))
        #
        mvs         = ma.masked_array(vs_plt, mask = mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection = projection)
        x, y        = m(self.lonArr-360., self.latArr)
        if plotfault:
            if projection == 'lambert':
                shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
                m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
                m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
            else:
                shapefname  = '/home/lili/code/gem-global-active-faults/shapefile/gem_active_faults'
                # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
                m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        if plottecto:
            shapefname  = '/home/lili/mongolia_proj/Tectono_WGS84_map/TectonoMapCAOB'
            m.readshapefile(shapefname, 'tecto', linewidth = 1, color='black')
            # m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
 
        # # sedi = '/home/lili/data_marin/map_data/AKgeol_web_shp/AKStategeolpoly_generalized_WGS84'
        # # m.readshapefile(sedi, 'faultline', linewidth = 3, color='black')

        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        
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
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # if depth < 
        
        if vmin == 4.1 and vmax == 4.6:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6])
        elif vmin == 4.15 and vmax == 4.55:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.15, 4.25, 4.35, 4.45, 4.55])
        elif vmin == 3.5 and vmax == 4.5:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[3.5, 3.75, 4.0, 4.25, 4.5])
        else:
            cb          = m.colorbar(im,  "bottom", size="5%", pad='2%')
            
        if plotcontour:
            mc          = m.contour(x, y, mvs, colors=['r', 'g', 'b'], levels = [4.2, 4.25, 4.3], linewidths=[1,1,1])
            plt.clabel(mc, inline=True, fmt='%g', fontsize=15)
        # cb.set_label(clabel, fontsize=20, rotation=0)
        # cb.ax.tick_params(labelsize=15)
        print ('mean Vs: %g min: %g, max %g' %(mvs.mean(), mvs.min(), mvs.max()))
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=20)
        cb.set_alpha(1)
        cb.draw_all()
        #
        if len(verlons) > 0 and len(verlons) == len(verlats): 
            xv, yv      = m(verlons, verlats)
            m.plot(xv, yv,'k-', lw = 3)
        if len(vprlons) > 0 and len(vprlons) == len(vprlats):
            for i in range(len(vprlons)):
                xv, yv      = m(vprlons[i], vprlats[i])
                m.plot(xv, yv,'*', color = 'red', ms = 5, mec='k')
        ############
        
        # xv, yv      = m([-160., -152.2], [57.5, 54.5])
        # m.plot(xv, yv,'k-', lw = 4)
        # m.plot(xv, yv,color = 'lime', lw = 3)
        # xv, yv      = m([-160., -154.5], [57.5, 53.3])
        # m.plot(xv, yv,'k-', lw = 4)
        # m.plot(xv, yv,color = 'lime', lw = 3)
        # xv, yv      = m([-160., -159], [57.5, 52.3])
        # m.plot(xv, yv,'k-', lw = 4)
        # m.plot(xv, yv,color = 'lime', lw = 3)
        # xv, yv      = m([-160., -149.8], [57.5, 55.7])
        # m.plot(xv, yv,'k-', lw = 4)
        # m.plot(xv, yv,color = 'lime', lw = 3)
        ############
        
        
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
                print ('Loading catalog')
                cat     = obspy.read_events('alaska_events.xml')
                print ('Catalog loaded!')
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
                ind_lon     = abs(self.lons - evlo).argmin()
                ind_lat     = abs(self.lats - evla).argmin()
                moho_depth  = mohoArr[ind_lat, ind_lon]
                # if evdp < moho_depth:
                #     continue
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla);
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            ind             = (values >= depth - evdepavg)*(values<=depth+evdepavg)
            x, y            = m(evlons[ind], evlats[ind])
            m.plot(x, y, 'o', mfc='yellow', mec='k', ms=5, alpha=.8)
        ############################
        if plotslab:
            from netCDF4 import Dataset
            slab2       = Dataset('/home/lili/data_marin/map_data/Slab2Distribute_Mar2018/alu_slab2_dep_02.23.18.grd')
            depthz       = (slab2.variables['z'][:]).data
            lons        = (slab2.variables['x'][:])
            lats        = (slab2.variables['y'][:])
            mask        = (slab2.variables['z'][:]).mask
            
            lonslb,latslb   = np.meshgrid(lons, lats)

            lonslb  = lonslb[np.logical_not(mask)]
            latslb  = latslb[np.logical_not(mask)]
            depthslb  = -depthz[np.logical_not(mask)]
            ind = abs(depthslb - depth)<1.0
            xslb, yslb = m(lonslb[ind]-360., latslb[ind])
                                                         
            m.plot(xslb, yslb, 'k-', lw=4, mec='k')
            m.plot(xslb, yslb, color = 'cyan', lw=2.5, mec='k')
        ############################
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return
    
    def plot_vertical(self, lon1, lat1, lon2, lat2, maxdepth, vs_mantle=4.4, plottype = 0, d = 10., dtype='avg', is_smooth=True,\
            itype = 'ray', clabel='', cmap='surf', vmin1=3.0, vmax1=4.2, vmin2=4.15, vmax2=4.55, incat=None, dist_thresh=20., showfig=True):
        topoArr     = self['topo'][()]
        if is_smooth:
            mohoArr = self[dtype+'_paraval_'+itype+'/12_smooth'][()]\
                        + self[dtype+'_paraval_'+itype+'/11_smooth'][()] - topoArr
        else:
            mohoArr = self[dtype+'_paraval_'+itype+'/12_org'][()]\
                        + self[dtype+'_paraval_'+itype+'/11_org'][()] - topoArr
        if lon1 == lon2 and lat1 == lat2:
            raise ValueError('The start and end points are the same!')
        self._get_lon_lat_arr()
        grp         = self[dtype+'_paraval_'+itype]
        if is_smooth:
            vs3d    = grp['vs_smooth'][()]
            zArr    = grp['z_smooth'][()]
        else:
            vs3d    = grp['vs_org'][()]
            zArr    = grp['z_org'][()]
        mask        = self.attrs['mask']
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
            ###
            topo1d[ind_data]= topoArr[ind_lat, ind_lon]
            
            # from netCDF4 import Dataset
            # from matplotlib.colors import LightSource
            # import pycpt
            # # # # etopodata   = Dataset('/home/lili/gebco_mongo.nc')
            # etopodata   = Dataset('/home/lili/gebco_aacse.nc')
            # 
            # etopo       = (etopodata.variables['elevation'][:]).data
            # lons_etopo  = (etopodata.variables['lon'][:]).data
            # lats_etopo  = (etopodata.variables['lat'][:]).data
            # 
            # tmp_ilon    = (abs(lons_etopo - lon)).argmin()
            # tmp_ilat    = (abs(lats_etopo - lat)).argmin()
            # 
            # topo1d[ind_data]= etopo[tmp_ilat, tmp_ilon]/1000.
            
            ###
            moho1d[ind_data]= mohoArr[ind_lat, ind_lon]
            mask1d[ind_data, :]\
                            = mask[ind_lat, ind_lon]
            ind_data        += 1
        data_moho           = data.copy()
        mask_moho           = np.ones(data.shape, dtype=bool)
        data_mantle         = data.copy()
        mask_mantle         = np.ones(data.shape, dtype=bool)
        for ix in range(data.shape[0]):
            ind_moho                    = zplot <= moho1d[ix]
            ind_mantle                  = np.logical_not(ind_moho)
            mask_moho[ix, ind_moho]     = False
            mask_mantle[ix, ind_mantle] = False
            # data_mantle[ix, :] \
            #                 = (data_mantle[ix, :] - vs_mantle)/vs_mantle*100.
            data_mantle[ix, :]          = data_mantle[ix, :] 
        mask_moho           += mask1d
        mask_mantle         += mask1d
        if plottype == 0:
            xplot   = plons
            xlabel  = 'longitude (deg)'
        else:
            xplot   = plats
            xlabel  = 'latitude (deg)'
        ########################

        try:
            import pycpt
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
        except:
            pass
        f, (ax1, ax2)   = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios':[1,4]})
        topo1d[topo1d<0.]   \
                        = 0.
        ax1.plot(xplot, topo1d*1000., 'k', lw=3)
        ax1.fill_between(xplot, 0, topo1d*1000., facecolor='grey')
        ax1.set_ylabel('Elevation (m)', fontsize=20)
        ax1.set_ylim(0, topo1d.max()*1000.+10.)
        mdata_moho      = ma.masked_array(data_moho, mask=mask_moho )
        mdata_mantle    = ma.masked_array(data_mantle, mask=mask_mantle )
        m1              = ax2.pcolormesh(xplot, zplot, mdata_mantle.T, shading='gouraud', vmax=vmax2, vmin=vmin2, cmap=cmap)
        
        # mc              = ax2.contour(xplot, zplot, mdata_mantle.T, colors=['r', 'g', 'b'], levels = [4.2, 4.25, 4.3], linewidths=[3,3,3])
        # ax2.clabel(mc, inline=True, fmt='%g', fontsize=15)
        
        if vmin2 == 4.15 and vmax2 == 4.55:
            cb1             = f.colorbar(m1, orientation='horizontal', fraction=0.05, ticks=[4.15, 4.25, 4.35, 4.45, 4.55])
        else:
            cb1             = f.colorbar(m1, orientation='horizontal', fraction=0.05)
        
        cb1.set_label('Mantle Vs (km/s)', fontsize=20)
        cb1.ax.tick_params(labelsize=15) 
        m2              = ax2.pcolormesh(xplot, zplot, mdata_moho.T, shading='gouraud', vmax=vmax1, vmin=vmin1, cmap=cmap)
        cb2             = f.colorbar(m2, orientation='horizontal', fraction=0.06, ticks=[3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
        cb2.set_label('Crustal Vs (km/s)', fontsize=20)
        cb2.ax.tick_params(labelsize=10) 
        #
        ax2.plot(xplot, moho1d, 'r', lw=3)
        #
        # ax2.set_xlabel(xlabel, fontsize=20)
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
                print ('Loading catalog')
                cat     = obspy.read_events('../alaska_events.xml')
                print ('Catalog loaded!')
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
                az, baz, dist   = g.inv(lons_arr, lats_arr, np.ones(lons_arr.size)*evlo, np.ones(lons_arr.size)*evla)
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
                
        ############
        from netCDF4 import Dataset
        
        slab2       = Dataset('/home/lili/data_marin/map_data/Slab2Distribute_Mar2018/alu_slab2_dep_02.23.18.grd')
        depth       = (slab2.variables['z'][:]).data
        lons        = (slab2.variables['x'][:])
        lats        = (slab2.variables['y'][:])
        mask        = (slab2.variables['z'][:]).mask
        
        lonslb,latslb   = np.meshgrid(lons, lats)
        lonslb  = lonslb[np.logical_not(mask)]
        latslb  = latslb[np.logical_not(mask)]
        depthslb  = depth[np.logical_not(mask)]
        
        L               = lonslb.size
        ind_data        = 0
        plons           = np.zeros(len(lonlats))
        plats           = np.zeros(len(lonlats))
        slbdepth        = np.zeros(len(lonlats))
        for lon,lat in lonlats:
            if lon < 0.:
                lon     += 360.
            clonArr             = np.ones(L, dtype=float)*lon
            clatArr             = np.ones(L, dtype=float)*lat
            az, baz, dist       = g.inv(clonArr, clatArr, lonslb, latslb)
            ind_min             = dist.argmin()
            plons[ind_data]     = lon
            plats[ind_data]     = lat
            slbdepth[ind_data]  = -depthslb[ind_min]
            ind_data            += 1
        ax2.plot(xplot, slbdepth, 'k', lw=5)
        ax2.plot(xplot, slbdepth, 'cyan', lw=3)
        # # # # 
        ########
        if plottype == 0:
            ax2.plot(evlons, values, 'o', mfc='yellow', mec='k', ms=5, alpha=0.8)
        else:
            ax2.plot(evlats, values, 'o', mfc='yellow', mec='k', ms=5, alpha=0.8)
            
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
    
    def plot_horizontal_discontinuity(self, depthrange, distype='moho', dtype='avg', itype='ray', is_smooth=True, shpfx=None, clabel='', title='',\
            cmap='surf', projection='lambert', plottecto=True, plotfault=True, vmin=None, vmax=None, \
            lonplt=[], latplt=[], showfig=True, incat=None, val=4.405):
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
        
        topoArr     = self['topo'][()]
        if distype is 'moho':
            if is_smooth:
                disArr  = self[dtype+'_paraval_%s/12_smooth' %itype][()] + self[dtype+'_paraval_%s/11_smooth'%itype][()] - topoArr
            else:
                disArr  = self[dtype+'_paraval_%s/12_org' %itype][()] + self[dtype+'_paraval_%s/11_org'%itype][()] - topoArr
        elif distype is 'sedi':
            if is_smooth:
                disArr  = self[dtype+'_paraval_%s/11_smooth'%itype][()] - topoArr
            else:
                disArr  = self[dtype+'_paraval_%s/11_org'%itype][()] - topoArr
        else:
            raise ValueError('Unexpected type of discontinuity:'+distype)
        self._get_lon_lat_arr()
        grp         = self[dtype+'_paraval_%s'%itype]
        if is_smooth:
            vs3d    = grp['vs_smooth'][()]
            zArr    = grp['z_smooth'][()]
        else:
            vs3d    = grp['vs_org'][()]
            zArr    = grp['z_org'][()]
        if depthrange < 0.:
            depth0  = disArr + depthrange
            depth1  = disArr.copy()
        else:
            depth0  = disArr
            # depth0  = -topoArr
            # depth1  = disArr + depthrange
            depth1  = 60.*np.ones(disArr.shape)
        vs_plt      = _get_vs_2d(z0=depth0, z1=depth1, zArr=zArr, vs_3d=vs3d)
        mask        = self.attrs['mask']
        mvs         = ma.masked_array(vs_plt, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        if plotfault:
            shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
            m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black', zorder=7.)
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white', zorder=8.)
            
            
            # shapefname  = '/home/lili/code/gem-global-active-faults/shapefile/gem_active_faults'
            # # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
            # m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        if plottecto:
            shapefname  = '/home/lili/mongolia_proj/Tectono_WGS84_map/TectonoMapCAOB'
            m.readshapefile(shapefname, 'tecto', linewidth = 1, color='black')
            # m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        
        
        
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
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        
        if vmin == 4.1 and vmax == 4.6:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6])
        elif vmin == 4.15 and vmax == 4.55:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.15, 4.25, 4.35, 4.45, 4.55])
        else:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%')

        if incat is not None:
            evlons  = np.array([])
            evlats  = np.array([])
            values  = np.array([])
            valuetype = 'depth'
            
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
                ind_lon     = abs(self.lons - evlo).argmin()
                ind_lat     = abs(self.lats - evla).argmin()
                # moho_depth  = mohoArr[ind_lat, ind_lon]
                # if evdp < moho_depth:
                #     continue
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla);
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            ind = (values <= 60.)
            x, y            = m(evlons[ind], evlats[ind])

            # x, y            = m(evlons, evlats)
            m.plot(x, y, 'o', mfc='yellow', mec='k', ms=5, alpha=.8, zorder=5.)
        
        x, y        = m(self.lonArr, self.latArr)
        mask[(x>717650)] = True
        mask[(x<443493)] = True
        mask[y>433346] = True
        mvs         = ma.masked_array(vs_plt, mask=mask )
        mc          = m.contour(x, y, mvs, colors=['k'], levels = [val], linewidths=[3], zorder=10)
        
        
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=20)
        cb.set_alpha(1)
        cb.draw_all()
        if showfig:
            plt.show()
        return
    
    
    def plot_slip_aacse(self, depth, val=4.405, evdepavg = 5., verlats = [], verlons = [], depthb=None, depthavg=None, dtype='avg', itype = 'ray',
        is_smooth=True, shpfx=None, clabel='', title='', cmap='surf', projection='lambert',  vmin=None, vmax=None, \
            lonplt=[], latplt=[], incat=None, plotevents=False, showfig=True, outfname=None, plotfault=True, plotslab=False,
        plottecto = False, vprlons = [], vprlats = [], plotcontour=True):
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
        self._get_lon_lat_arr()
        topoArr     = self['topo'][()]
        if is_smooth:
            mohoArr = self[dtype+'_paraval_'+itype+'/12_smooth'][()]\
                        + self[dtype+'_paraval_'+itype+'/11_smooth'][()] - topoArr
        else:
            mohoArr = self[dtype+'_paraval_'+itype+'/12_org'][()]\
                        + self[dtype+'_paraval_'+itype+'/11_org'][()] - topoArr
        grp         = self[dtype+'_paraval_'+itype]
        if is_smooth:
            vs3d    = grp['vs_smooth'][()]
            zArr    = grp['z_smooth'][()]
        else:
            vs3d    = grp['vs_org'][()]
            zArr    = grp['z_org'][()]
        if depthb is not None:
            if depthb < depth:
                raise ValueError('depthb should be larger than depth!')
            index   = np.where((zArr >= depth)*(zArr <= depthb) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        elif depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth + depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis = 2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print ('depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km')
                return
            depth       = zArr[index]
            vs_plt      = vs3d[:, :, index]
        mask        = self.attrs['mask']
        #
        mask        += (vs3d[:, :, zArr == depth - 1.0] == 0.).reshape((self.Nlat, self.Nlon))
        #
        mvs         = ma.masked_array(vs_plt, mask = mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection = projection)
        x, y        = m(self.lonArr-360., self.latArr)
        if plotfault:
            if projection == 'lambert':
                shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
                m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
                m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
            else:
                shapefname  = '/home/lili/code/gem-global-active-faults/shapefile/gem_active_faults'
                # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
                m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        if plottecto:
            shapefname  = '/home/lili/mongolia_proj/Tectono_WGS84_map/TectonoMapCAOB'
            m.readshapefile(shapefname, 'tecto', linewidth = 1, color='black')


        shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        shplst      = shapefile.Reader(shapefname)
        for rec in shplst.records():
            lon_vol = rec[4]
            lat_vol = rec[3]
            xvol, yvol            = m(lon_vol, lat_vol)
            m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        
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
        ##############################################################################################
        inarr = np.loadtxt('slip.txt')
        lons = inarr[:, 0] - 360.
        lats = inarr[:, 1]
        zlock= inarr[:, 2]
        minlon = 194.6-360.
        minlat = 52.7
        maxlon = 207.5-360.
        maxlat = 58.6
        gridder     = _grid_class.SphereGridder(minlon = minlon, maxlon = maxlon, dlon = 0.1, \
                            minlat = minlat, maxlat = maxlat, dlat = 0.1, period = 10., \
                            evlo = 0., evla = 0., fieldtype = 'Tph', evid = 'plt')
        gridder.read_array(inlons = lons, inlats = lats, inzarr = zlock)
        gridder.interp_surface( do_blockmedian = True)
        x, y        = m(gridder.lon2d-360., gridder.lat2d)
        
        zlockarr    = gridder.Zarr
        mask = np.ones(zlockarr.shape, dtype=bool)
        for ix in range(gridder.Nlat):
            for iy in range(gridder.Nlon):
                ilat = gridder.lats[ix]
                ilon = gridder.lons[iy]
                ind = (abs(lons-ilon)<0.1)*(abs(lats-ilat)<0.1)
                if len(lons[ind]) > 0:
                    mask[ix, iy] = False

        # # # # return gridder
        # # # 
        mzlock         = ma.masked_array(zlockarr, mask = mask )
        im          = m.pcolormesh(x, y, mzlock, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        
        mc          = m.contour(x, y, mzlock, colors=['g'], levels = [0.5], linewidths=[3])
        # plt.clabel(mc, inline=True, fmt='%g', fontsize=15)
        # # # x, y        = m(lons, lats)
        # # # im          = m.scatter(x, y, zlock, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='None', markersize =10)
        
        if vmin == 4.1 and vmax == 4.6:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6])
        elif vmin == 4.15 and vmax == 4.55:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.15, 4.25, 4.35, 4.45, 4.55])
        elif vmin == 3.5 and vmax == 4.5:
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[3.5, 3.75, 4.0, 4.25, 4.5])
        else:
            cb          = m.colorbar(im,  "bottom", size="5%", pad='2%')
        
        ########################################################
        distype = 'moho'
        topoArr     = self['topo'][()]
        if distype is 'moho':
            if is_smooth:
                disArr  = self[dtype+'_paraval_%s/12_smooth' %itype][()] + self[dtype+'_paraval_%s/11_smooth'%itype][()] - topoArr
            else:
                disArr  = self[dtype+'_paraval_%s/12_org' %itype][()] + self[dtype+'_paraval_%s/11_org'%itype][()] - topoArr
        elif distype is 'sedi':
            if is_smooth:
                disArr  = self[dtype+'_paraval_%s/11_smooth'%itype][()] - topoArr
            else:
                disArr  = self[dtype+'_paraval_%s/11_org'%itype][()] - topoArr
        else:
            raise ValueError('Unexpected type of discontinuity:'+distype)
        self._get_lon_lat_arr()
        grp         = self[dtype+'_paraval_%s'%itype]
        if is_smooth:
            vs3d    = grp['vs_smooth'][()]
            zArr    = grp['z_smooth'][()]
        else:
            vs3d    = grp['vs_org'][()]
            zArr    = grp['z_org'][()]

        depth0  = disArr 
        # depth1  = disArr + depthrange
        depth1  = 60.*np.ones(disArr.shape)
        vs_plt      = _get_vs_2d(z0=depth0, z1=depth1, zArr=zArr, vs_3d=vs3d)
        mask        = self.attrs['mask']
        
        
        ########################################################
        x, y        = m(self.lonArr-360., self.latArr)
        mask[(x>717650)] = True
        mask[(x<443493)] = True
        mask[y>433346] = True
        mvs         = ma.masked_array(vs_plt, mask=mask )
        if plotcontour:
            # mc          = m.contour(x, y, mvs, colors=['r', 'g', 'b'], levels = [4.2, 4.25, 4.3], linewidths=[1,1,1])
            # mc          = m.contour(x, y, mvs, colors=['r', 'g', 'b'], levels = [4.35, 4.4, 4.405], linewidths=[1,1,1])
            mc          = m.contour(x, y, mvs, colors=['k'], levels = [val], linewidths=[3])
            # plt.clabel(mc, inline=True, fmt='%g', fontsize=15)
        # cb.set_label(clabel, fontsize=20, rotation=0)
        # cb.ax.tick_params(labelsize=15)
        print ('mean Vs: %g min: %g, max %g' %(mvs.mean(), mvs.min(), mvs.max()))
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=20)
        cb.set_alpha(1)
        cb.draw_all()
        #
        if len(verlons) > 0 and len(verlons) == len(verlats): 
            xv, yv      = m(verlons, verlats)
            m.plot(xv, yv,'k-', lw = 3)
        if len(vprlons) > 0 and len(vprlons) == len(vprlats):
            for i in range(len(vprlons)):
                xv, yv      = m(vprlons[i], vprlats[i])
                m.plot(xv, yv,'*', color = 'red', ms = 5, mec='k')
        
        
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
                print ('Loading catalog')
                cat     = obspy.read_events('alaska_events.xml')
                print ('Catalog loaded!')
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
                ind_lon     = abs(self.lons - evlo).argmin()
                ind_lat     = abs(self.lats - evla).argmin()
                moho_depth  = mohoArr[ind_lat, ind_lon]
                # if evdp < moho_depth:
                #     continue
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla);
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            ind             = (values >= depth - evdepavg)*(values<=depth+evdepavg)
            x, y            = m(evlons[ind], evlats[ind])
            m.plot(x, y, 'o', mfc='yellow', mec='k', ms=5, alpha=.8)
        ############################
        if plotslab:
            from netCDF4 import Dataset
            slab2       = Dataset('/home/lili/data_marin/map_data/Slab2Distribute_Mar2018/alu_slab2_dep_02.23.18.grd')
            depthz       = (slab2.variables['z'][:]).data
            lons        = (slab2.variables['x'][:])
            lats        = (slab2.variables['y'][:])
            mask        = (slab2.variables['z'][:]).mask
            
            lonslb,latslb   = np.meshgrid(lons, lats)

            lonslb  = lonslb[np.logical_not(mask)]
            latslb  = latslb[np.logical_not(mask)]
            depthslb  = -depthz[np.logical_not(mask)]
            ind = abs(depthslb - depth)<1.0
            xslb, yslb = m(lonslb[ind]-360., latslb[ind])
                                                         
            m.plot(xslb, yslb, 'k-', lw=4, mec='k')
            m.plot(xslb, yslb, color = 'cyan', lw=2.5, mec='k')
        ############################
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return
    