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
import surfpy.eikonal._grid_class as _grid_class
import surfpy.cpt_files as cpt_files
cpt_path    = cpt_files.__path__._path[0]

import numpy as np
import numpy.ma as ma

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
import matplotlib.pyplot as plt

class isoh5(invbase.baseh5):
    
    def mc_inv_disp(self, use_ref=False, ingrdfname=None, phase=True, group=False, outdir = None, restart = False, \
        vp_water=1.5, isconstrt=True, step4uwalk=1500, numbrun=15000, subsize=1000, nprocess=None, parallel=True, skipmask=True,\
            Ntotalruns=10, misfit_thresh=1.0, Nmodelthresh=200, outlon=None, outlat=None, verbose = False, verbose2 = False):
        """
        Bayesian Monte Carlo inversion of surface wave data 
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
        if (outlon is None) or (outlat is None):
            print ('[%s] [MC_ISO_INVERSION] inversion START' %datetime.now().isoformat().split('.')[0])
            if outdir is None:
                if restart and 'mc_inv_run_path' in self.attrs.keys():
                    outdir  = self.attrs['mc_inv_run_path']
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
                vpr.model.isomod.parameterize_ak135(crtthk=crtthk, sedthk=sedthk, topovalue=topovalue, \
                        maxdepth=200., vp_water=vp_water)
            vpr.get_paraind()
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:    
                    return vpr
            start_time_grd  = time.time()
            print ('[%s] [MC_ISO_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=1., Ntotalruns=Ntotalruns, \
                    misfit_thresh=misfit_thresh, Nmodelthresh=Nmodelthresh, isconstrt=isconstrt, pfx=grd_id, verbose=verbose,\
                        verbose2 = verbose2, step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=1., \
                   isconstrt=isconstrt, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            end_time    = time.time()
            print ('[%s] [MC_ISO_INVERSION] inversion DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', elasped time = %g'%(end_time - start_time_grd) + ' sec; total elasped time = %g' %(end_time - start_time_total))
        print ('[%s] [MC_ISO_INVERSION] inversion ALL DONE' %datetime.now().isoformat().split('.')[0] + \
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
            ilat        = np.where(grd_lat == self.lats)[0]
            ilon        = np.where(grd_lon == self.lons)[0]
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
            return postvpr
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
    
    # def get_vpr(self, datadir, lon, lat, factor=1., thresh=0.5, Nmax=None, Nmin=None):
    #     """
    #     Get the postvpr (postprocessing vertical profile)
    #     """
    #     if lon < 0.:
    #         lon     += 360.
    #     grd_id      = str(lon)+'_'+str(lat)
    #     grd_grp     = self['grd_pts']
    #     try:
    #         grp     = grd_grp[grd_id]
    #     except:
    #         print ('!!! No data at longitude =',lon,' lattitude =',lat)
    #         return 
    #     invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
    #     datafname   = datadir+'/mc_data.'+grd_id+'.npz'
    #     topovalue   = grp.attrs['topo']
    #     vpr         = mcpost.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh)
    #     vpr.read_inv_data(infname = invfname, verbose=True, Nmax=Nmax, Nmin=Nmin)
    #     vpr.read_data(infname = datafname)
    #     vpr.get_paraval()
    #     vpr.run_avg_fwrd(wdisp=1.)
    #     if vpr.avg_misfit > (vpr.min_misfit*vpr.factor + vpr.thresh)*2.:
    #         print '--- Unstable inversion results for grid: lon = '+str(lon)+', lat = '+str(lat)
    #     if lon > 0.:
    #         lon     -= 360.
    #     vpr.code    = str(lon)+'_'+str(lat)
    #     return vpr
    
    
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
                paraval                 = grp[dtype+'_paraval_'+itype][()]
            except KeyError:
                # print 'WARNING: no data at grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
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
                data[ind_lat, ind_lon]      = data[ind_lat, ind_lon] - topovalue
        return data
    
    def paraval_arrays(self, dtype = 'min', itype = 'ray', sigma=1, gsigma = 50., depth = 5., depthavg = 0., verbose=False):
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
        if not np.all(self.attrs['mask_inv'] == self.attrs['mask_inv_result']):
            print ('!!! WARNING: mask_inv not equal to mask_inv_result')
        grp                 = self.require_group( name = dtype+'_paraval_'+itype )
        topo                = self['topo'][()]
        index_inv           = np.logical_not(self.attrs['mask_inv_result'])
        index               = np.logical_not(self.attrs['mask'])
        lons_inv            = self.lonArr_inv[index_inv]
        lats_inv            = self.latArr_inv[index_inv]
        lons                = self.lonArr[index]
        lats                = self.latArr[index]
        for pindex in range(13):
            if pindex == 11:
                data_inv    = self.get_paraval(pindex = pindex, dtype = dtype, itype = itype, isthk = True, depth = depth, depthavg = depthavg)
                if np.any(data_inv[index_inv] < -100.):
                    raise ValueError('!!! Error in inverted data!')
                # interpolation
                gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'sedi_thk', evid = 'MCINV')
                gridder.read_array(inlons = lons_inv, inlats = lats_inv, inzarr = data_inv[index_inv])
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
                # convert sediment depth to sediment thickness
                data        += topo
                data_smooth += topo
                sedi        = data.copy()
                sedi_smooth = data_smooth.copy()
            elif pindex == 12:
                data_inv    = self.get_paraval(pindex = 'moho', dtype = dtype, itype = itype, isthk = True, depth = depth, depthavg = depthavg)
                if np.any(data_inv[index_inv] < -100.):
                    raise ValueError('!!! Error in inverted data!')
                # interpolation
                gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'crst_thk', evid = 'MCINV')
                gridder.read_array(inlons = lons_inv, inlats = lats_inv, inzarr = data_inv[index_inv])
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
                # convert moho depth to crustal thickness (excluding sediments)
                data        += topo
                data_smooth += topo
                data        -= sedi
                data_smooth -= sedi_smooth
            else:
                data_inv    = self.get_paraval(pindex = pindex, dtype = dtype, itype = itype, isthk = False, depth = depth, depthavg = depthavg)
                if np.any(data_inv[index_inv] < -100.):
                    raise ValueError('!!! Error in inverted data!')
                # interpolation
                gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = -1., evla = -1., fieldtype = 'paraval_'+str(pindex), evid = 'MCINV')
                gridder.read_array(inlons = lons_inv, inlats = lats_inv, inzarr = data_inv[index_inv])
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
    
    
    def _get_basemap(self, projection='lambert', resolution='i'):
        """get basemap for plotting results
        """
        plt.figure()
        minlon          = self.minlon
        maxlon          = self.maxlon
        minlat          = self.minlat
        maxlat          = self.maxlat
        lat_centre      = (maxlat+minlat)/2.0
        lon_centre      = (maxlon+minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat = minlat, urcrnrlat = maxlat, llcrnrlon=minlon,
                      urcrnrlon=maxlon, lat_ts=0, resolution=resolution)
            # m.drawparallels(np.arange(minlat,maxlat,dlat), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(minlon,maxlon,dlon), labels=[1,0,0,1])
            m.drawparallels(np.arange(-80.0, 80.0, 1.0), labels=[1,1,1,1])
            m.drawmeridians(np.arange(-170.0, 170.0, 1.0), labels=[1,1,1,0], fontsize=5)
            # m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            # m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        elif projection=='regional_ortho':
            m      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            # m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
            #             llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/2., urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m
            
            # # distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            # # distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m
            # # # m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
            # # #             lat_1=minlat, lat_2=maxlat, lon_0=lon_centre-2., lat_0=lat_centre+2.4)
            m       = Basemap(width=1100000, height=1000000, rsphere=(6378137.00,6356752.3142), resolution='h', projection='lcc',\
                        lat_1 = minlat, lat_2 = maxlat, lon_0 = lon_centre, lat_0 = lat_centre+1.)
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1, dashes=[2,2], labels=[0,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,0,0], fontsize=15)
        m.drawcountries(linewidth=1.)
        #################
        try:
            coasts = m.drawcoastlines(zorder = 100, color = '0.9',linewidth = 0.0001)
            # 
            # # Exact the paths from coasts
            coasts_paths = coasts.get_paths()
            
            # In order to see which paths you want to retain or discard you'll need to plot them one
            # at a time noting those that you want etc.
            poly_stop = 10
            for ipoly in range(len(coasts_paths)):
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
        ######################
        m.drawstates(linewidth=1.)
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        return m
    
    # def plot_paraval(self, pindex, is_smooth=True, dtype='avg', itype='ray', sigma=1, gsigma = 50., \
    #         ingrdfname=None, isthk=False, shpfx=None, outfname=None, outimg=None, clabel='', title='', cmap='cv', \
    #             projection='lambert', lonplt=[], latplt=[],\
    #                 vmin=None, vmax=None, showfig=True, depth = 5., depthavg = 0.):
    #     """
    #     plot the one given parameter in the paraval array
    #     ===================================================================================================
    #     ::: input :::
    #     pindex      - parameter index in the paraval array
    #                     0 ~ 13, moho: model parameters from paraval arrays
    #                     vs_std      : vs_std from the model ensemble, dtype does NOT take effect
    #     org_mask    - use the original mask in the database or not
    #     dtype       - data type:
    #                     avg - average model
    #                     min - minimum misfit model
    #                     sem - uncertainties (standard error of the mean)
    #     itype       - inversion type
    #                     'ray'   - isotropic inversion using Rayleigh wave
    #                     'vti'   - VTI intersion using Rayleigh and Love waves
    #     ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
    #     isthk       - flag indicating if the parameter is thickness or not
    #     clabel      - label of colorbar
    #     cmap        - colormap
    #     projection  - projection type
    #     geopolygons - geological polygons for plotting
    #     vmin, vmax  - min/max value of plotting
    #     showfig     - show figure or not
    #     ===================================================================================================
    #     """
    #     if pindex is 'min_misfit' or pindex is 'avg_misfit' or pindex is 'fitratio' or pindex is 'mean_misfit':
    #         is_interp   = False
    #     if is_interp:
    #         mask        = self.attrs['mask_interp']
    #     else:
    #         mask        = self.attrs['mask_inv']
    #     if pindex =='rel_moho_std':
    #         data, data_smooth\
    #                     = self.get_smooth_paraval(pindex='moho', dtype='avg', itype=itype, \
    #                         sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
    #         # print 'mean = ', data[np.logical_not(mask)].mean()
    #         undata, undata_smooth\
    #                     = self.get_smooth_paraval(pindex='moho', dtype='std', itype=itype, \
    #                         sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
    #         # print 'mean = ', undata[np.logical_not(mask)].mean()
    #         data = undata/data
    #         data_smooth = undata_smooth/data_smooth
    #     else:
    #         data, data_smooth\
    #                     = self.get_smooth_paraval(pindex=pindex, dtype=dtype, itype=itype, \
    #                         sigma=sigma, gsigma = gsigma, isthk=isthk, do_interp=is_interp, depth=depth, depthavg=depthavg)
    #     # return data
    #     if pindex is 'min_misfit' or pindex is 'avg_misfit':
    #         indmin      = np.where(data==data.min())
    #         print indmin
    #         print 'minimum overall misfit = '+str(data.min())+' longitude/latitude ='\
    #                     + str(self.lonArr[indmin[0], indmin[1]])+'/'+str(self.latArr[indmin[0], indmin[1]])
    #         indmax      = np.where(data==data.max())
    #         print 'maximum overall misfit = '+str(data.max())+' longitude/latitude ='\
    #                     + str(self.lonArr[indmax[0], indmax[1]])+'/'+str(self.latArr[indmax[0], indmax[1]])
    #         #
    #         ind         = (self.latArr == 62.)*(self.lonArr==-149.+360.)
    #         data[ind]   = 0.645
    #         #
    #     
    #     if is_smooth:
    #         mdata       = ma.masked_array(data_smooth, mask=mask )
    #     else:
    #         mdata       = ma.masked_array(data, mask=mask )
    #     print 'mean = ', data[np.logical_not(mask)].mean()
    #     #-----------
    #     # plot data
    #     #-----------
    #     m               = self._get_basemap(projection=projection)
    #     # m           = self._get_basemap_3(projection=projection, geopolygons=geopolygons)
    #     x, y            = m(self.lonArr, self.latArr)
    #     # shapefname      = '/home/leon/geological_maps/qfaults'
    #     # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
    #     # shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
    #     # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
    #     plot_fault_lines(m, 'AK_Faults.txt', color='grey')
    #     # slb_ctrlst      = read_slab_contour('alu_contours.in', depth=100.)
    #     # if len(slb_ctrlst) == 0:
    #     #     print 'No contour at this depth =',depth
    #     # else:
    #     #     for slbctr in slb_ctrlst:
    #     #         xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
    #     #         m.plot(xslb, yslb,  '--', lw = 5, color='black')
    #     #         m.plot(xslb, yslb,  '--', lw = 3, color='white')
    #     ### slab edge
    #     arr             = np.loadtxt('SlabE325.dat')
    #     lonslb          = arr[:, 0]
    #     latslb          = arr[:, 1]
    #     depthslb        = -arr[:, 2]
    #     index           = (depthslb > (depth - .05))*(depthslb < (depth + .05))
    #     lonslb          = lonslb[index]
    #     latslb          = latslb[index]
    #     indsort         = lonslb.argsort()
    #     lonslb          = lonslb[indsort]
    #     latslb          = latslb[indsort]
    #     xslb, yslb      = m(lonslb, latslb)
    #     m.plot(xslb, yslb,  '-', lw = 5, color='black')
    #     m.plot(xslb, yslb,  '-', lw = 3, color='cyan')
    #     
    #     
    # 
    #     # m.plot(xslb, yslb,  '--', lw = 3, color='cyan')
    #     ### 
    #     if cmap == 'ses3d':
    #         cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
    #                         0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
    #     elif cmap == 'cv':
    #         import pycpt
    #         cmap        = pycpt.load.gmtColormap('./cv.cpt')
    #     elif cmap == 'gmtseis':
    #         import pycpt
    #         cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
    #     else:
    #         try:
    #             if os.path.isfile(cmap):
    #                 import pycpt
    #                 cmap    = pycpt.load.gmtColormap(cmap)
    #                 cmap    = cmap.reversed()
    #         except:
    #             pass
    #     ###################################################################
    #     # if hillshade:
    #     #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
    #     # else:
    #     #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
    #     if hillshade:
    #         im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=.5)
    #     else:
    #         im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
    #     if pindex == 'moho' and dtype == 'avg':
    #         cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[25., 29., 33., 37., 41., 45.])
    #     elif pindex == 'moho' and dtype == 'std':
    #         cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    #     else:
    #         cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
    #     # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    #     cb.set_label(clabel, fontsize=60, rotation=0)
    #     cb.ax.tick_params(labelsize=30)
    # 
    #     # # cb.solids.set_rasterized(True)
    #     # ###
    #     # xc, yc      = m(np.array([-156]), np.array([67.5]))
    #     # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
    #     # xc, yc      = m(np.array([-153]), np.array([61.]))
    #     # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
    #     # xc, yc      = m(np.array([-149]), np.array([64.]))
    #     # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
    #     # # xc, yc      = m(np.array([-143]), np.array([61.5]))
    #     # # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
    #     # 
    #     # xc, yc      = m(np.array([-152]), np.array([60.]))
    #     # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
    #     # xc, yc      = m(np.array([-155]), np.array([69]))
    #     # m.plot(xc, yc,'*', ms = 15, markeredgecolor='black', markerfacecolor='yellow')
    #     ###
    #     #############################
    #     yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
    #     yatlons             = yakutat_slb_dat[:, 0]
    #     yatlats             = yakutat_slb_dat[:, 1]
    #     xyat, yyat          = m(yatlons, yatlats)
    #     m.plot(xyat, yyat, lw = 5, color='black')
    #     m.plot(xyat, yyat, lw = 3, color='white')
    #     #############################
    #     import shapefile
    #     shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
    #     shplst      = shapefile.Reader(shapefname)
    #     for rec in shplst.records():
    #         lon_vol = rec[4]
    #         lat_vol = rec[3]
    #         xvol, yvol            = m(lon_vol, lat_vol)
    #         m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
    #     plt.suptitle(title, fontsize=30)
    #     
    #     cb.solids.set_edgecolor("face")
    #     if len(lonplt) > 0 and len(lonplt) == len(latplt): 
    #         xc, yc      = m(lonplt, latplt)
    #         m.plot(xc, yc,'go', lw = 3)
    #     plt.suptitle(title, fontsize=30)
    #     # m.shadedrelief(scale=1., origin='lower')
    #     if showfig:
    #         plt.show()
    #     if outfname is not None:
    #         ind_valid   = np.logical_not(mask)
    #         outlon      = self.lonArr[ind_valid]
    #         outlat      = self.latArr[ind_valid]
    #         outZ        = data[ind_valid]
    #         OutArr      = np.append(outlon, outlat)
    #         OutArr      = np.append(OutArr, outZ)
    #         OutArr      = OutArr.reshape(3, outZ.size)
    #         OutArr      = OutArr.T
    #         np.savetxt(outfname, OutArr, '%g')
    #     if outimg is not None:
    #         plt.savefig(outimg)
    #     return
    
    def plot_horizontal(self, depth, evdepavg = 5., depthb=None, depthavg=None, dtype='avg', itype = 'ray', is_smooth=True, shpfx=None,\
            clabel='', title='', cmap='surf', projection='lambert',  vmin=None, vmax=None, \
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
        self._get_lon_lat_arr()
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
        mvs         = ma.masked_array(vs_plt, mask = mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection = projection)
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
        
        try:
            import pycpt
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
                # cmap    = cmap.reversed()
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
        except:
            pass
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
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla);
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            ind             = (values >= depth - evdepavg)*(values<=depth+evdepavg)
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
        # import shapefile
        # shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        # shplst      = shapefile.Reader(shapefname)
        # for rec in shplst.records():
        #     lon_vol = rec[4]
        #     lat_vol = rec[3]
        #     xvol, yvol            = m(lon_vol, lat_vol)
        #     m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        # plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return