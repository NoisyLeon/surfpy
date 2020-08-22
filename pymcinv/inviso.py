# -*- coding: utf-8 -*-
"""
hdf5 for noise eikonal tomography
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.pymcinv.invbase as invbase
import surfpy.pymcinv.inverse_solver as inverse_solver

import numpy as np

import obspy
from datetime import datetime
import warnings
import glob
import sys
import copy
import os


class isoh5(invbase.baseh5):
    
    def mc_inv_disp(self, use_ref=False, ingrdfname=None, phase=True, group=False, outdir = None, vp_water=1.5, isconstrt=True,
            verbose=False, step4uwalk=1500, numbrun=15000, subsize=1000, nprocess=None, parallel=True, skipmask=True,\
            Ntotalruns=10, misfit_thresh=1.0, Nmodelthresh=200, outlon=None, outlat=None):
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
            outdir  = os.path.dirname(self.filename)+'/mc_inv_run_%s' %datetime.now().isoformat().split('.')[0]
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        self.attrs.create(name = 'mc_inv_run_path', data = outdir)
        start_time_total= time.time()
        grd_grp         = self['grd_pts']
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
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = inverse_solver.inverse_vprofile()
            if phase:
                try:
                    indisp      = grd_grp[grd_id+'/disp_ph_ray'].value
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
            print ('[%s] [MC_ISO_INVERSION] ' %datetime.now().isoformat().split('.')[0] + \
                    'grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=1., Ntotalruns=Ntotalruns, \
                    misfit_thresh=misfit_thresh, Nmodelthresh=Nmodelthresh, isconstrt=isconstrt, pfx=grd_id, verbose=verbose,\
                        step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=1., \
                   isconstrt=isconstrt, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            end_time    = time.time()
            print ('[%s] [MC_ISO_INVERSION] inversion DONE' %datetime.now().isoformat().split('.')[0] + \
                    ', elasped time = '+str(end_time - start_time_grd) + ' sec; total elasped time = '+str(end_time - start_time_total))
        return
    
    def read_inv(self, datadir = None, ingrdfname=None, factor=1., thresh=0.5, stdfactor=2, avgqc=True, \
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
                print ('!!! No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
                grp.attrs.create(name='mask', data = True)
                temp_mask[ilat, ilon]\
                        = True
                continue
            print ('=== Reading inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
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
                    print ('--- Unstable inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
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
    