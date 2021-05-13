# -*- coding: utf-8 -*-
"""
hdf5 for 
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.pymcinv.invbase as invbase
import surfpy.pymcinv.inverse_solver as inverse_solver
import surfpy.pymcinv.vmodel as vmodel
import surfpy.pymcinv._model_funcs as _model_funcs
# # # from surfpy.pymcinv._modparam_vti import NOANISO, LAYERGAMMA, GAMMASPLINE, VSHSPLINE, LAYER, BSPLINE, GRADIENT, WATER
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


class htih5(invbase.baseh5):
    
    def set_spain_mask(self):
        """merge mask
        """
        grp         = self['hti_model']
        mask        = grp.attrs['mask_hti'][()]
        import h5py
        dset    = h5py.File('/raid/lili/data_spain/azi_run_dir/azi_files_sta_75km_crt100p/0427_mod_w10.h5')
        topo    = dset['topo'][()]
        ind     = (topo < 0.)*(self.latArr > 43.1)*(self.lonArr < 0.)
        mask[ind] = True
        grp.attrs.create(name = 'mask_hti', data = mask)
        
        return
    
    def compute_kernels_hti(self, ingrdfname=None, vp_water=1.5, misfit_thresh=1.5, outlon=None, outlat=None, outlog='error.log'):
        """
        Bayesian Monte Carlo inversion of VTI model
        ==================================================================================================================
        ::: input :::
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        outdir          - output directory
        vp_water        - P wave velocity in water layer (default - 1.5 km/s)
        outlon/outlat   - output a vprofile object given longitude and latitude
        ==================================================================================================================
        """
        start_time_total    = time.time()
        self._get_lon_lat_arr()
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
                    if lon < 0. and self.ilontype == 1:
                        lon += 360.
                    elif lon > 180. and self.ilontype == 0:
                        lon -= 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        ipercent    = 0
        topoarr     = self['topo'][()]
        print ('[%s] [HTI_KERNELS] computing START' %datetime.now().isoformat().split('.')[0])
        fid         = open(outlog, 'w')
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
                print ('[%s] [HTI_KERNELS] ' %datetime.now().isoformat().split('.')[0] + '%g'%ipercent+ '% finished')
                ipercent            += 1
            
            #-----------------------------
            # get data
            #-----------------------------
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
            vpr                 = inverse_solver.inverse_vprofile()
            disp_azi_ray        = azi_grp[grd_id+'/disp_azi_ray'][()]
            vpr.get_azi_disp(indata = disp_azi_ray)
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (abs(self.lonArr - grd_lon) < self.dlon/10.) * (abs(self.latArr - grd_lat) < self.dlat/10.)
            # # # *(self.latArr == grd_lat)
            paraval_ref         = np.zeros(13, np.float64)
            try:
                paraval_ref[0]      = self['avg_paraval/0_smooth'][()][index]
                paraval_ref[1]      = self['avg_paraval/1_smooth'][()][index]
                paraval_ref[2]      = self['avg_paraval/2_smooth'][()][index]
                paraval_ref[3]      = self['avg_paraval/3_smooth'][()][index]
                paraval_ref[4]      = self['avg_paraval/4_smooth'][()][index]
                paraval_ref[5]      = self['avg_paraval/5_smooth'][()][index]
                paraval_ref[6]      = self['avg_paraval/6_smooth'][()][index]
                paraval_ref[7]      = self['avg_paraval/7_smooth'][()][index]
                paraval_ref[8]      = self['avg_paraval/8_smooth'][()][index]
                paraval_ref[9]      = self['avg_paraval/9_smooth'][()][index]
                paraval_ref[10]     = self['avg_paraval/10_smooth'][()][index]
                paraval_ref[11]     = self['avg_paraval/11_smooth'][()][index]
                paraval_ref[12]     = self['avg_paraval/12_smooth'][()][index]
            except Exception:
                try:
                    paraval_ref[0]      = self['avg_paraval/0_smooth_iso'][()][index]
                    paraval_ref[1]      = self['avg_paraval/1_smooth_iso'][()][index]
                    paraval_ref[2]      = self['avg_paraval/2_smooth_iso'][()][index]
                    paraval_ref[3]      = self['avg_paraval/3_smooth_iso'][()][index]
                    paraval_ref[4]      = self['avg_paraval/4_smooth_iso'][()][index]
                    paraval_ref[5]      = self['avg_paraval/5_smooth_iso'][()][index]
                    paraval_ref[6]      = self['avg_paraval/6_smooth_iso'][()][index]
                    paraval_ref[7]      = self['avg_paraval/7_smooth_iso'][()][index]
                    paraval_ref[8]      = self['avg_paraval/8_smooth_iso'][()][index]
                    paraval_ref[9]      = self['avg_paraval/9_smooth_iso'][()][index]
                    paraval_ref[10]     = self['avg_paraval/10_smooth_iso'][()][index]
                    paraval_ref[11]     = self['avg_paraval/11_smooth_iso'][()][index]
                    paraval_ref[12]     = self['avg_paraval/12_smooth_iso'][()][index]
                except Exception:
                    print ('!!! ERROR %s' %grd_id)
                    return
            topovalue           = topoarr[index]
            if topovalue >= 0.:
                vpr.model.get_para_model(paraval = paraval_ref)
            else:
                vpr.model.get_para_model(paraval = paraval_ref, waterdepth = -topovalue, nmod=4, \
                    numbp=np.array([1, 2, 4, 5]), mtype = np.array([5, 4, 2, 2]), vpvs = np.array([0, 2., 1.75, 1.75]), maxdepth=200.)
            vpr.model.isomod.mod2para()
            vpr.get_period()
            vpr.update_mod(mtype = 'iso')
            vpr.get_vmodel(mtype = 'iso')
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
                print ('!!! Unstable disp: '+grd_id+', misfit = '+str(vpr.data.misfit))
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
                else:
                    print ('!!! Stable disp found: '+grd_id+', misfit = '+str(vpr.data.misfit))
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
        # print '--- Elasped time = '+str(end_time - start_time_total)
        return
    
    def linear_inv_hti(self, ingrdfname=None,  vp_water=1.5, misfit_thresh=5.0, invtype = 0, tquake = 999., quakestdfactor = 2.,\
            quakestdback = False, outlon=None, outlat=None, depth_mid_crust=-1., depth_mid_mantle=-1., depth_mid_mantle2=-1.,verbose=False):
        """
        Linear inversion of HTI model
        ==================================================================================================================
        ::: input :::
        ingrdfname      - input grid point list file indicating the grid points for surface wave inversion
        vp_water        - P wave velocity in water layer (default - 1.5 km/s)
        misfit_thresh   - threshold misfit value to determine "good" models
        outlon/outlat   - output a vprofile object given longitude and latitude
        ---
        version history:
                    - first version (2019-03-28)
        ==================================================================================================================
        """
        start_time_total    = time.time()
        self._get_lon_lat_arr()
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
                    if lon < 0. and self.ilontype == 1:
                        lon += 360.
                    elif lon > 180. and self.ilontype == 0:
                        lon -= 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        igrd        = 0
        Ngrd        = len(grdlst)
        topoarr     = self['topo'][()]
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
            vpr                 = inverse_solver.inverse_vprofile()
            disp_azi_ray        = azi_grp[grd_id+'/disp_azi_ray'][()]
            vpr.get_azi_disp(indata = disp_azi_ray)
            pers                = vpr.data.dispR.pper
            if pers[-1] >= tquake:
                ind                         = pers>= tquake
                vpr.data.dispR.unamp[ind]   /= quakestdfactor
                vpr.data.dispR.unpsi2[ind]  /= quakestdfactor
            #-----------------------------------------------------------------
            # initialize reference model and computing sensitivity kernels
            #-----------------------------------------------------------------
            index               = (abs(self.lonArr - grd_lon) < self.dlon/10.) * (abs(self.latArr - grd_lat) < self.dlat/10.)
            paraval_ref         = np.zeros(13, np.float64)
            try:
                paraval_ref[0]      = self['avg_paraval/0_smooth'][()][index]
                paraval_ref[1]      = self['avg_paraval/1_smooth'][()][index]
                paraval_ref[2]      = self['avg_paraval/2_smooth'][()][index]
                paraval_ref[3]      = self['avg_paraval/3_smooth'][()][index]
                paraval_ref[4]      = self['avg_paraval/4_smooth'][()][index]
                paraval_ref[5]      = self['avg_paraval/5_smooth'][()][index]
                paraval_ref[6]      = self['avg_paraval/6_smooth'][()][index]
                paraval_ref[7]      = self['avg_paraval/7_smooth'][()][index]
                paraval_ref[8]      = self['avg_paraval/8_smooth'][()][index]
                paraval_ref[9]      = self['avg_paraval/9_smooth'][()][index]
                paraval_ref[10]     = self['avg_paraval/10_smooth'][()][index]
                paraval_ref[11]     = self['avg_paraval/11_smooth'][()][index]
                paraval_ref[12]     = self['avg_paraval/12_smooth'][()][index]
            except Exception:
                try:
                    paraval_ref[0]      = self['avg_paraval/0_smooth_iso'][()][index]
                    paraval_ref[1]      = self['avg_paraval/1_smooth_iso'][()][index]
                    paraval_ref[2]      = self['avg_paraval/2_smooth_iso'][()][index]
                    paraval_ref[3]      = self['avg_paraval/3_smooth_iso'][()][index]
                    paraval_ref[4]      = self['avg_paraval/4_smooth_iso'][()][index]
                    paraval_ref[5]      = self['avg_paraval/5_smooth_iso'][()][index]
                    paraval_ref[6]      = self['avg_paraval/6_smooth_iso'][()][index]
                    paraval_ref[7]      = self['avg_paraval/7_smooth_iso'][()][index]
                    paraval_ref[8]      = self['avg_paraval/8_smooth_iso'][()][index]
                    paraval_ref[9]      = self['avg_paraval/9_smooth_iso'][()][index]
                    paraval_ref[10]     = self['avg_paraval/10_smooth_iso'][()][index]
                    paraval_ref[11]     = self['avg_paraval/11_smooth_iso'][()][index]
                    paraval_ref[12]     = self['avg_paraval/12_smooth_iso'][()][index]
                except Exception:
                    print ('!!! ERROR %s' %grd_id)
                    return
            topovalue           = topoarr[index]
            if topovalue >= 0.:
                vpr.model.get_para_model(paraval = paraval_ref)
            else:
                vpr.model.get_para_model(paraval = paraval_ref, waterdepth = -topovalue, nmod=4, \
                    numbp=np.array([1, 2, 4, 5]), mtype = np.array([5, 4, 2, 2]), vpvs = np.array([0, 2., 1.75, 1.75]), maxdepth=200.)
            vpr.model.isomod.mod2para()
            vpr.get_period()
            vpr.update_mod(mtype = 'iso')
            vpr.get_vmodel(mtype = 'iso')
            if not 'dcdL' in azi_grp[grd_id].keys():   
                # cmin                = vpr.data.dispR.pvelo.min()-0.5
                # cmax                = vpr.data.dispR.pvelo.max()+0.5
                cmin                = 1.5
                cmax                = 6.
                vpr.compute_reference_vti(wtype='ray', cmin=cmin, cmax=cmax)
                vpr.get_misfit()
                if vpr.data.dispR.check_disp(thresh=0.4):
                    print ('Unstable disp value: '+grd_id+', misfit = '+str(vpr.data.misfit))
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
                iso_misfit      = azi_grp[grd_id+'/iso_misfit'][()]
            dcdA                = azi_grp[grd_id+'/dcdA'][()]
            dcdC                = azi_grp[grd_id+'/dcdC'][()]
            dcdF                = azi_grp[grd_id+'/dcdF'][()]
            dcdL                = azi_grp[grd_id+'/dcdL'][()]
            pvelref             = azi_grp[grd_id+'/pvel_ref'][()]
            vpr.get_reference_hti(pvelref=pvelref, dcdA=dcdA, dcdC=dcdC, dcdF=dcdF, dcdL=dcdL)
            if iso_misfit > misfit_thresh:
                print ('Large misfit value: '+grd_id+', misfit = '+str(iso_misfit))
            #------------
            # inversion
            #------------
            if (not outlon is None) and (not outlat is None):
                if grd_lon != outlon or grd_lat != outlat:
                    continue
                else:
                    return vpr
            crtthk              = vpr.model.isomod.thickness[:-1].sum()
            if vpr.model.isomod.mtype[0] == 5:
                crtthk          -= vpr.model.isomod.thickness[0]
            # else:
            #     crtthk          = vpr.model.isomod.thickness[1]
            # try:
            if depth_mid_mantle > 0. and depth_mid_mantle2 > 0:
                vpr.linear_inv_hti_three_mantle(isBcs=True, useref=False, depth1 = depth_mid_mantle, depth2 = depth_mid_mantle2)
            else:
                if crtthk >= (depth_mid_crust + 5):
                    vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=depth_mid_crust, depth_mid_mantle=depth_mid_mantle)
                else:
                    vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=-1, depth_mid_mantle=depth_mid_mantle)
            # except Exception:
            #     continue
            
            # # # if quakestdback and pers[-1] >= tquake:
            # # #     ind                         = pers>= tquake
            # # #     vpr.data.dispR.unamp[ind]   *= quakestdfactor
            # # #     vpr.data.dispR.unpsi2[ind]  *= quakestdfactor
            if invtype == -1 and vpr.data.dispR.pmisfit_psi > .5 and depth_mid_mantle < 0.\
                    and grd_lat > 41.6 and grd_lat < 43.4 and grd_lon > -4.75 and grd_lon < -0.75:
                t0 = 30.
                t1 = 80.
                pers = vpr.data.dispR.pper
                psi0 = vpr.data.dispR.psi2[pers == t0]
                psi1 = vpr.data.dispR.psi2[pers == t1]
                dpsi = abs(psi0 - psi1)
                if dpsi > 90.:
                    dpsi = 180. - dpsi
                dpsi2 = abs(vpr.model.htimod.psi2[1] - psi1)
                if dpsi2 > 90.:
                    dpsi2 = 180. - dpsi2
                if dpsi > 60. and dpsi2 > 30.:
                    vpr1                = inverse_solver.inverse_vprofile()
                    vpr1.get_azi_disp(indata = disp_azi_ray)
                    if topovalue >= 0.:
                        vpr1.model.get_para_model(paraval = paraval_ref)
                    else:
                        vpr1.model.get_para_model(paraval = paraval_ref, waterdepth = -topovalue, nmod=4, \
                            numbp=np.array([1, 2, 4, 5]), mtype = np.array([5, 4, 2, 2]), vpvs = np.array([0, 2., 1.75, 1.75]), maxdepth=200.)
                    vpr1.model.isomod.mod2para()
                    vpr1.get_period()
                    vpr1.update_mod(mtype = 'iso')
                    vpr1.get_vmodel(mtype = 'iso')
                    vpr1.get_reference_hti(pvelref=pvelref, dcdA=dcdA, dcdC=dcdC, dcdF=dcdF, dcdL=dcdL)
                    vpr1.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=depth_mid_crust, depth_mid_mantle = 60.)
                    
                    vpr.model.htimod.psi2[-1] = vpr1.model.htimod.psi2[-1]
                    vpr.model.htimod.unpsi2[-1] = vpr1.model.htimod.unpsi2[-1]
                    vpr.model.htimod.amp[-1] = vpr1.model.htimod.amp[-1]
                    vpr.model.htimod.unamp[-1] = vpr1.model.htimod.unamp[-1]
                    
                    azi_grp[grd_id].create_dataset(name='azi_misfit', data=vpr1.data.misfit)
                    azi_grp[grd_id].create_dataset(name='psi_misfit', data=vpr1.data.dispR.pmisfit_psi)
                    azi_grp[grd_id].create_dataset(name='amp_misfit', data=vpr1.data.dispR.pmisfit_amp)
                    azi_grp[grd_id].create_dataset(name='psi2', data=vpr.model.htimod.psi2)
                    azi_grp[grd_id].create_dataset(name='unpsi2', data=vpr.model.htimod.unpsi2)
                    azi_grp[grd_id].create_dataset(name='amp', data=vpr.model.htimod.amp)
                    azi_grp[grd_id].create_dataset(name='unamp', data=vpr.model.htimod.unamp)
                    
                    
                    #==============================================
                    # ind = (pers >= 22.)*(pers <= 40.)
                    # ind = (pers <= 50.)
                    # vpr.data.dispR.psi2[ind] = psi1
                    # vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=depth_mid_crust, depth_mid_mantle = -1.)

                    # azi_grp[grd_id].create_dataset(name='azi_misfit', data=vpr.data.misfit)
                    # azi_grp[grd_id].create_dataset(name='psi_misfit', data=vpr.data.dispR.pmisfit_psi)
                    # azi_grp[grd_id].create_dataset(name='amp_misfit', data=vpr.data.dispR.pmisfit_amp)
                    # azi_grp[grd_id].create_dataset(name='psi2', data=vpr.model.htimod.psi2)
                    # azi_grp[grd_id].create_dataset(name='unpsi2', data=vpr.model.htimod.unpsi2)
                    # azi_grp[grd_id].create_dataset(name='amp', data=vpr.model.htimod.amp)
                    # azi_grp[grd_id].create_dataset(name='unamp', data=vpr.model.htimod.unamp)
                    
                    azi_grp[grd_id].create_dataset(name='adaptive', data=1)
                    
                    continue
            #-------------------------
            # save inversion results
            #-------------------------
            if depth_mid_mantle > 0. and depth_mid_mantle2 > 0:
                azi_grp[grd_id].create_dataset(name='azi_misfit', data=vpr.data.misfit)
                azi_grp[grd_id].create_dataset(name='psi_misfit', data=vpr.data.dispR.pmisfit_psi)
                azi_grp[grd_id].create_dataset(name='amp_misfit', data=vpr.data.dispR.pmisfit_amp)
                azi_grp[grd_id].create_dataset(name='psi2', data=vpr.model.htimod.psi2[1:])
                azi_grp[grd_id].create_dataset(name='unpsi2', data=vpr.model.htimod.unpsi2[1:])
                azi_grp[grd_id].create_dataset(name='amp', data=vpr.model.htimod.amp[1:])
                azi_grp[grd_id].create_dataset(name='unamp', data=vpr.model.htimod.unamp[1:])

                
                azi_grp[grd_id].create_dataset(name='adaptive', data=0)
            else:
                azi_grp[grd_id].create_dataset(name='azi_misfit', data=vpr.data.misfit)
                azi_grp[grd_id].create_dataset(name='psi_misfit', data=vpr.data.dispR.pmisfit_psi)
                azi_grp[grd_id].create_dataset(name='amp_misfit', data=vpr.data.dispR.pmisfit_amp)
                azi_grp[grd_id].create_dataset(name='psi2', data=vpr.model.htimod.psi2)
                azi_grp[grd_id].create_dataset(name='unpsi2', data=vpr.model.htimod.unpsi2)
                azi_grp[grd_id].create_dataset(name='amp', data=vpr.model.htimod.amp)
                azi_grp[grd_id].create_dataset(name='unamp', data=vpr.model.htimod.unamp)
                
                azi_grp[grd_id].create_dataset(name='adaptive', data=0)
        return
    
    
    def construct_hti_model(self):
        self._get_lon_lat_arr()
        azi_grp     = self['azi_grd_pts']
        grdlst      = list(azi_grp.keys())
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
        adparr      = np.zeros(self.lonArr.shape, dtype=np.int32)
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
            ind_lon = np.where( abs(self.lons - grd_lon)< self.dlon/10. )[0]
            ind_lat = np.where( abs(self.lats - grd_lat) < self.dlat/10. )[0]
            #-----------------------------
            # get data
            #-----------------------------
            try:
                psi2                    = azi_grp[grd_id+'/psi2'][()]
                unpsi2                  = azi_grp[grd_id+'/unpsi2'][()]
                amp                     = azi_grp[grd_id+'/amp'][()]
                unamp                   = azi_grp[grd_id+'/unamp'][()]
                misfit                  = azi_grp[grd_id+'/azi_misfit'][()]
                ampmisfit               = azi_grp[grd_id+'/amp_misfit'][()]
                psimisfit               = azi_grp[grd_id+'/psi_misfit'][()]
            except:
                continue
                # temp_grd_id             = grdlst[igrd]
                # split_id= grd_id.split('_')
                # try:
                #     tmp_grd_lon         = float(split_id[0])
                # except ValueError:
                #     continue
                # tmp_grd_lat             = float(split_id[1])
                # if not (grd_lon == tmp_grd_lon and abs(tmp_grd_lat - grd_lat)<self.dlat/100. ):
                #     print (temp_grd_id, grd_id)
                #     raise ValueError('ERROR!')
                # psi2                    = azi_grp[temp_grd_id+'/psi2'][()]
                # unpsi2                  = azi_grp[temp_grd_id+'/unpsi2'][()]
                # amp                     = azi_grp[temp_grd_id+'/amp'][()]
                # unamp                   = azi_grp[temp_grd_id+'/unamp'][()]
                # misfit                  = azi_grp[temp_grd_id+'/azi_misfit'][()]
                # ampmisfit               = azi_grp[temp_grd_id+'/amp_misfit'][()]
                # psimisfit               = azi_grp[temp_grd_id+'/psi_misfit'][()]
            try:
                adaptive    = azi_grp[grd_id+'/adaptive'][()]
            except Exception:
                adaptive    = -1
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
            # mask
            mask[ind_lat, ind_lon]          = False
            adparr[ind_lat, ind_lon]        = adaptive
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
        # mask
        out_grp.attrs.create(name='mask_hti', data=mask)
        # aux
        out_grp.create_dataset(name='adaptive', data=adparr)
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
            minlon=-10.
            maxlon=5.
            minlat=31.
            maxlat=45.
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
    
    #==================================================================
    # plotting functions 
    #==================================================================

#     
    def plot_hti_vel(self, depth, depthavg=3., gindex=1, plot_axis=True, plot_data=True, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='surf', projection='merc', \
                    vmin=None, vmax=None, showfig=True, ticks=[], lon_plt=[], lat_plt=[], gwidth = -1.):
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
        self._get_lon_lat_arr()
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex][()]
            unpsi2      = grp['unpsi2_%d' %gindex][()]
            amp         = grp['amp_%d' %gindex][()]
            unamp       = grp['unamp_%d' %gindex][()]
        else:
            plot_axis   = False
        mask        = grp.attrs['mask_hti'][()]

        grp         = self['avg_paraval']
        vs3d        = grp['vsv_smooth'][()]
        zArr        = grp['z_smooth'][()]
        if depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            data    = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print ('depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km')
                return
            depth       = zArr[index]
            data        = vs3d[:, :, index]
        
        if gwidth > 0.:
            gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = 0., evla = 0., fieldtype = 'Tph', evid = 'plt')
            gridder.read_array(inlons = self.lonArr[np.logical_not(mask)], inlats = self.latArr[np.logical_not(mask)], inzarr = data[np.logical_not(mask)])
            outfname    = 'plt_Tph.lst'
            prefix      = 'plt_Tph_'
            gridder.gauss_smoothing(workingdir = './temp_plt', outfname = outfname, width = gwidth)
            data[:]     = gridder.Zarr
        
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        if plot_data:
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
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)

            if vmin == 4.1 and vmax == 4.6:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6])
            elif vmin == 4.15 and vmax == 4.55:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.15, 4.25, 4.35, 4.45, 4.55])
            elif vmin == 3.5 and vmax == 4.5:
                cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[3.5, 3.75, 4.0, 4.25, 4.5])
            else:
                cb          = m.colorbar(im,  "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=60, rotation=0)
            cb.ax.tick_params(labelsize=20)
            cb.set_alpha(1)
            cb.draw_all()
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)
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
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr, self.latArr, returnxy=True)
            
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
                # mask_ref        = np.ones(self.lonArr.shape)
                # ind_lat         = np.where(self.lats==58.)[0]
                # ind_lon         = np.where(self.lons==-145.+360.)[0]
                # mask_ref[ind_lat, ind_lon] = False
                # Uref            = ma.masked_array(Uref, mask=mask_ref )
                # Vref            = ma.masked_array(Vref, mask=mask_ref )
                # # m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                # # m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                # m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                # m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                # m.quiver(xref, yref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
                # m.quiver(xref, yref, -Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
                
                x_ref, y_ref = m(2.5, 32.)
                Uref    = 2./ampref/normv
                Vref    = 0.
                Q1      = m.quiver(x_ref, y_ref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                Q2      = m.quiver(x_ref, y_ref, -Uref, -Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                Q1      = m.quiver(x_ref, y_ref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
                Q2      = m.quiver(x_ref, y_ref, -Uref, -Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        ##
        if projection == 'merc' and os.path.isdir('/home/lili/spain_proj/geo_maps'):
            shapefname  = '/home/lili/spain_proj/geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='grey')
        plt.suptitle(title, fontsize=20)
        
        
        # xc, yc      = m(np.array([-153.]), np.array([66.1]))
        # m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        # azarr       = np.arange(36.)*10.
        
        if len(lon_plt) == len(lat_plt) and len(lon_plt) >0:
            xc, yc      = m(lon_plt, lat_plt)
            m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')

        
        if showfig:
            plt.show()
        return
#     
    def plot_hti_sks(self,  gindex=1, masked=True, clabel='', title='', projection='merc', showfig=True, ticks=[]):
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
        self._get_lon_lat_arr()
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex][()]
            unpsi2      = grp['unpsi2_%d' %gindex][()]
            amp         = grp['amp_%d' %gindex][()]
            unamp       = grp['unamp_%d' %gindex][()]
        else:
            plot_axis   = False
        
        mask        = grp.attrs['mask_hti'][()]
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        fname       = 'splittingDB.ascii'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        dtlst       = []
        lonlst      = []
        latlst      = []
        amplst      = []
        misfit      = self['hti_model/misfit'][()]
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        
        with open(fname, 'r') as fid:
            for line in fid.readlines():
                lst = line.split('|')
                if lst[0] == 'id':
                    continue
                try:
                    #code
                
                    lonsks      = float(lst[3])
                    if lonsks < 0. and self.ilontype == 1:
                        lonsks += 360.
                    elif lonsks > 180. and self.ilontype == 0:
                        lonsks -= 360.
                    latsks      = float(lst[2])
                    if self.minlat > latsks or self.maxlat < latsks or\
                        self.minlon > lonsks or self.maxlon < lonsks:
                        continue
                    
                    ind_lon     = np.where(abs(self.lons - lonsks)<self.dlon)[0][0]
                    ind_lat     = np.where(abs(self.lats - latsks)<self.dlat)[0][0]
                    if mask[ind_lat, ind_lon]:
                        continue
                    tmpphi = float(lst[4])
                    tmpdt  = float(lst[5])
                except Exception:
                    continue   
                stalst.append(lst[0])
                philst.append(tmpphi)
                dtlst.append(tmpdt)
                lonlst.append(lonsks)
                latlst.append(latsks)
                
                psilst.append(psi2[ind_lat, ind_lon])
                unpsilst.append(unpsi2[ind_lat, ind_lon])
                amplst.append(amp[ind_lat, ind_lon])
                
                temp_misfit = misfit[ind_lat, ind_lon]
                temp_dpsi   = abs(psi2[ind_lat, ind_lon] - float(lst[4]))
                if temp_dpsi > 180.:
                    temp_dpsi   -= 180.
                if temp_dpsi > 90.:
                    temp_dpsi   = 180. - temp_dpsi

        phiarr  = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        # unphiarr= np.array(unphilst)
        unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        dtarr   = np.array(dtlst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        dtref   = 1.
        normv   = 2.
        width = 0.006
        
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
        # dpsi
        dpsi[dpsi>180.] -= 180.
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        
        # undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        # return unpsiarr, unphiarr
        # # # ind_outline         = amparr < .2
        
        # 81 % comparison
        # mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        mask[(unpsiarr>=30.)*(dpsi>=30.)]   = True
        mask[(amparr<.3)*(dpsi>=30.)*(lonarr < -5.2)]   = True
        mask[(amparr<.5)*(dpsi>=30.)*(latarr<35.)]   = True
        
        # ind = (lonarr>=-6.4)*(lonarr<=-5.484)*(latarr>=36.)*(latarr<=36.637)*np.logical_not(mask)
        # mask[np.logical_not(ind)] = True
        ###
        # return lonarr[ind], latarr[ind], psiarr[ind], phiarr[ind]
        # mask[(amparr<.2)*(dpsi>=30.)]   = True
        # mask[(amparr<.3)*(dpsi>=30.)*(lonarr<-140.)]   = True
        
        
        xsks    = xsks[np.logical_not(mask)]
        ysks    = ysks[np.logical_not(mask)]
        Usks    = Usks[np.logical_not(mask)]
        Vsks    = Vsks[np.logical_not(mask)]
        Upsi    = Upsi[np.logical_not(mask)]
        Vpsi    = Vpsi[np.logical_not(mask)]
        dpsi    = dpsi[np.logical_not(mask)]

        
        Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Usks[dpsi<=30.], Vsks[dpsi<=30.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Usks[dpsi<=30.], -Vsks[dpsi<=30.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q1      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], Usks[(dpsi>30.)*(dpsi<=60.)], Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q2      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], -Usks[(dpsi>30.)*(dpsi<=60.)], -Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q1      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], Usks[dpsi>60.], Vsks[dpsi>60.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        Q2      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], -Usks[dpsi>60.], -Vsks[dpsi>60.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        
        
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        
            
        plt.suptitle(title, fontsize=20)
        plt.show()
        
        ax      = plt.subplot()
        dbin    = 10.
        # bins    = np.arange(min(dpsi), max(dpsi) + dbin, dbin)
        bins    = np.arange(0., 100., dbin)
        
        weights = np.ones_like(dpsi)/float(dpsi.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(dpsi)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        
        
        p.patches[0].set_color('blue')
        p.patches[1].set_color('blue')
        p.patches[2].set_color('blue')
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
        plt.ylabel('Percentage (%)', fontsize=50)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=10)
        ax.tick_params(axis='x', labelsize=30)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=30)
        formatter = FuncFormatter(_model_funcs.to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()
            
        return
    
    def plot_hti_sks_misfit(self, infname, thresh = 0.8, dattype='psi',  gindex=1, masked=True, clabel='', title='', projection='merc', showfig=True, ticks=[]):
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
        import h5py
        self._get_lon_lat_arr()
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex][()]
            unpsi2      = grp['unpsi2_%d' %gindex][()]
            amp         = grp['amp_%d' %gindex][()]
            unamp       = grp['unamp_%d' %gindex][()]
        else:
            plot_axis   = False

        dset        = h5py.File(infname)
        grp2        = dset['hti_model']
        if dattype == 'all':
            misfit    = grp2['misfit'][()]
        elif dattype == 'psi':
            misfit    = grp2['psi_misfit'][()]
        elif dattype == 'amp':
            misfit    = grp2['amp_misfit'][()]
        elif dattype == 'adp':
            misfit    = grp2['adaptive'][()]
        
        
        dset        = h5py.File('/raid/lili/data_spain/azi_run_dir/azi_files_0506/0506_tquake60_std2.h5')
        psi2_onelay = dset['hti_model/psi2_1'][()]
        
        
        print (misfit.mean(), misfit.max())
        mask        = grp.attrs['mask_hti'][()]
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        fname       = 'splittingDB.ascii'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        dtlst       = []
        lonlst      = []
        latlst      = []
        amplst      = []
        # # misfit      = self['hti_model/misfit'][()]
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        
        with open(fname, 'r') as fid:
            for line in fid.readlines():
                lst = line.split('|')
                if lst[0] == 'id':
                    continue
                try:
                    #code
                
                    lonsks      = float(lst[3])
                    if lonsks < 0. and self.ilontype == 1:
                        lonsks += 360.
                    elif lonsks > 180. and self.ilontype == 0:
                        lonsks -= 360.
                    latsks      = float(lst[2])
                    #########################
                    if not (latsks > 35.1 and latsks < 43.2 and lonsks < 0. and lonsks > -5.1):
                        continue
                    if latsks > 36.5 and latsks < 37.5:
                        continue
                    if latsks > 36.7:
                        continue
                    #########################
                    if self.minlat > latsks or self.maxlat < latsks or\
                        self.minlon > lonsks or self.maxlon < lonsks:
                        continue
                    
                    ind_lon     = np.where(abs(self.lons - lonsks)<self.dlon)[0][0]
                    ind_lat     = np.where(abs(self.lats - latsks)<self.dlat)[0][0]
                    if mask[ind_lat, ind_lon]:
                        continue
                    if misfit[ind_lat, ind_lon] < thresh:
                        continue
                    # # # print (misfit[ind_lat, ind_lon])
                    tmpphi = float(lst[4])
                    tmpdt  = float(lst[5])
                    
                    tmp     = abs(psi2_onelay[ind_lat, ind_lon] - tmpphi)
                    if tmp > 180.:
                        tmp -= 180.
                    if tmp > 90.:
                        tmp   = 180. - tmp
                    if tmp < 30.:
                        continue
                    
                except Exception:
                    continue   
                stalst.append(lst[0])
                philst.append(tmpphi)
                dtlst.append(tmpdt)
                lonlst.append(lonsks)
                latlst.append(latsks)
                
                psilst.append(psi2[ind_lat, ind_lon])
                unpsilst.append(unpsi2[ind_lat, ind_lon])
                amplst.append(amp[ind_lat, ind_lon])
                
                temp_misfit = misfit[ind_lat, ind_lon]
                temp_dpsi   = abs(psi2[ind_lat, ind_lon] - float(lst[4]))
                if temp_dpsi > 180.:
                    temp_dpsi   -= 180.
                if temp_dpsi > 90.:
                    temp_dpsi   = 180. - temp_dpsi
        # return 
        phiarr  = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        # unphiarr= np.array(unphilst)
        unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        dtarr   = np.array(dtlst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        dtref   = 1.
        normv   = 2.
        width = 0.006
        
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
        # dpsi
        dpsi[dpsi>180.] -= 180.
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        
        # undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        # return unpsiarr, unphiarr
        # # # ind_outline         = amparr < .2
        
        # 81 % comparison
        # mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        
        # mask[(unpsiarr>=30.)*(dpsi>=30.)]   = True
        # mask[(amparr<.3)*(dpsi>=30.)*(lonarr < -5.2)]   = True
        # mask[(amparr<.5)*(dpsi>=30.)*(latarr<35.)]   = True
        

        xsks    = xsks[np.logical_not(mask)]
        ysks    = ysks[np.logical_not(mask)]
        Usks    = Usks[np.logical_not(mask)]
        Vsks    = Vsks[np.logical_not(mask)]
        Upsi    = Upsi[np.logical_not(mask)]
        Vpsi    = Vpsi[np.logical_not(mask)]
        dpsi    = dpsi[np.logical_not(mask)]

        
        Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Usks[dpsi<=30.], Vsks[dpsi<=30.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Usks[dpsi<=30.], -Vsks[dpsi<=30.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q1      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], Usks[(dpsi>30.)*(dpsi<=60.)], Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q2      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], -Usks[(dpsi>30.)*(dpsi<=60.)], -Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q1      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], Usks[dpsi>60.], Vsks[dpsi>60.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        Q2      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], -Usks[dpsi>60.], -Vsks[dpsi>60.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        
        
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        
            
        plt.suptitle(title, fontsize=20)
        plt.show()
        
        ax      = plt.subplot()
        dbin    = 10.
        # bins    = np.arange(min(dpsi), max(dpsi) + dbin, dbin)
        bins    = np.arange(0., 100., dbin)
        
        weights = np.ones_like(dpsi)/float(dpsi.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(dpsi)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        
        
        p.patches[0].set_color('blue')
        p.patches[1].set_color('blue')
        p.patches[2].set_color('blue')
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
        plt.ylabel('Percentage (%)', fontsize=50)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=10)
        ax.tick_params(axis='x', labelsize=30)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=30)
        formatter = FuncFormatter(_model_funcs.to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()
            
        return

    
    def plot_hti_stress(self,  gindex=0, masked=True, clabel='', title='', projection='merc', showfig=True, ticks=[]):
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
        self._get_lon_lat_arr()
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex][()]
            unpsi2      = grp['unpsi2_%d' %gindex][()]
            amp         = grp['amp_%d' %gindex][()]
            unamp       = grp['unamp_%d' %gindex][()]
        else:
            plot_axis   = False
        mask        = grp.attrs['mask_hti'][()]
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        fname       = 'wsm2016.csv'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        qclst       = []
        # dtlst       = []
        lonlst      = []
        latlst      = []
        amplst      = []
        misfit      = self['hti_model/misfit'][()]
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        import csv
        with open(fname, 'r', encoding="utf8", errors='ignore') as fid:
            reader = csv.reader(fid)
            for lst in reader:
                # lst = line.split(',')
                if lst[0] == 'ID':
                    continue
                try:
                    #code
                
                    lonsks      = float(lst[3])
                    if lonsks < 0. and self.ilontype == 1:
                        lonsks += 360.
                    elif lonsks > 180. and self.ilontype == 0:
                        lonsks -= 360.
                    latsks      = float(lst[2])
                    if self.minlat > latsks or self.maxlat < latsks or\
                        self.minlon > lonsks or self.maxlon < lonsks:
                        continue
                    
                    ind_lon     = np.where(abs(self.lons - lonsks)<self.dlon)[0][0]
                    ind_lat     = np.where(abs(self.lats - latsks)<self.dlat)[0][0]
                    if mask[ind_lat, ind_lon]:
                        continue
                    tmpphi = float(lst[4])
                    quality  = lst[7]
                except Exception:
                    continue
                
                if not (quality == 'A' or quality == 'B' or quality == 'C'):
                    continue
                
                stalst.append(lst[0])
                philst.append(tmpphi)
                # dtlst.append(tmpdt)
                qclst.append(quality)
                lonlst.append(lonsks)
                latlst.append(latsks)
                
                psilst.append(psi2[ind_lat, ind_lon])
                unpsilst.append(unpsi2[ind_lat, ind_lon])
                amplst.append(amp[ind_lat, ind_lon])
                
                temp_misfit = misfit[ind_lat, ind_lon]
                temp_dpsi   = abs(psi2[ind_lat, ind_lon] - float(lst[4]))
                if temp_dpsi > 90.:
                    temp_dpsi   = 180. - temp_dpsi
                

        phiarr  = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        # unphiarr= np.array(unphilst)
        unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        qcarr   = np.array(qclst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        dtref   = 1.
        normv   = 2.
        width = 0.006
        
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
        # dpsi
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        
        # undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        # return unpsiarr, unphiarr
        # # # ind_outline         = amparr < .2
        
        # 81 % comparison
        # mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        
        # # # mask[(unpsiarr>=30.)*(dpsi>=30.)]   = True
        # # # mask[(amparr<.3)*(dpsi>=30.)]   = True
        # # # mask[(amparr<.5)*(dpsi>=30.)*(latarr<35.)]   = True
        
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


        Q1      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], Usks[dpsi<=30.], Vsks[dpsi<=30.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q2      = m.quiver(xsks[dpsi<=30.], ysks[dpsi<=30.], -Usks[dpsi<=30.], -Vsks[dpsi<=30.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q1      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], Usks[(dpsi>30.)*(dpsi<=60.)], Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q2      = m.quiver(xsks[(dpsi>30.)*(dpsi<=60.)], ysks[(dpsi>30.)*(dpsi<=60.)], -Usks[(dpsi>30.)*(dpsi<=60.)], -Vsks[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q1      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], Usks[dpsi>60.], Vsks[dpsi>60.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        Q2      = m.quiver(xsks[dpsi>60.], ysks[dpsi>60.], -Usks[dpsi>60.], -Vsks[dpsi>60.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        

        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        
        Q1      = m.quiver(xsks, ysks, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        Q2      = m.quiver(xsks, ysks, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        
            
        plt.suptitle(title, fontsize=20)
        plt.show()
        
        ax      = plt.subplot()
        dbin    = 10.
        # bins    = np.arange(min(dpsi), max(dpsi) + dbin, dbin)
        bins    = np.arange(0., 100., dbin)
        
        weights = np.ones_like(dpsi)/float(dpsi.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(dpsi)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        
        
        p.patches[0].set_color('blue')
        p.patches[1].set_color('blue')
        p.patches[2].set_color('blue')
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
        plt.ylabel('Percentage (%)', fontsize=50)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=10)
        ax.tick_params(axis='x', labelsize=30)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=30)
        formatter = FuncFormatter(_model_funcs.to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()
            
        return
    
    def plot_hti_flow(self,  gindex=1, depth = 200., clabel='', title='', projection='merc', showfig=True, ticks=[], normv = 20., width = 0.005):
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
        self._get_lon_lat_arr()
        grp         = self['hti_model']

        psi2        = grp['psi2_%d' %gindex][()]
        unpsi2      = grp['unpsi2_%d' %gindex][()]
        amp         = grp['amp_%d' %gindex][()]
        unamp       = grp['unamp_%d' %gindex][()]
        mask        = grp.attrs['mask_hti'][()]
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        fxyz        = 's20rtsb.inexhs3v.comb.0.xyzt'
        fv          = 's20rtsb.inexhs3v.comb.0.v'
        xyzarr      = np.loadtxt(fxyz)
        varr        = np.loadtxt(fv)
        colatarr    = xyzarr[:, 0]/np.pi*180.
        latarr      = 90. - colatarr
        lonarr      = xyzarr[:, 1]/np.pi*180.
        if self.ilontype == 0:
            lonarr[lonarr > 180.]   -= 360.
        ratioarr    = xyzarr[:, 2]
        radius      = 6371.
        deptharr    = (1. - ratioarr)*6371.
        zminind     = np.argmin( abs(deptharr - depth) )
        depth       = deptharr[zminind]
        print ('depth updated to %g' %depth)
        depind      = np.where( abs(deptharr - depth) < 0.1)[0]
        
        #
        lons        = lonarr[depind]
        lats        = latarr[depind]
        Vflow       = varr[depind, 0]
        Uflow       = varr[depind, 1]
        
        psiflow_in  = np.arcsin(Uflow/np.sqrt(Vflow**2 +Uflow**2))
        
        # psiflow_in  = np.arcsin(Uflow/np.sqrt(Vflow**2 +Uflow**2))
        psiflow_in  = 90. - psiflow_in
        
        psiflow     = np.zeros(self.lonArr.shape)
        
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if mask[ilat, ilon]:
                    continue
                tmplon  = self.lons[ilon]
                tmplat  = self.lats[ilat]
                tind    = (abs(tmplon - lons) < 1.)*(abs(tmplat - lats) < 1.)
                psiflow[ilat, ilon] = psiflow_in[tind][0]

        Uflow   = np.sin(psiflow/180.*np.pi)/normv
        Vflow   = np.cos(psiflow/180.*np.pi)/normv
        Uflow, Vflow, xflow, yflow  = m.rotate_vector(Uflow, Vflow, self.lonArr, self.latArr, returnxy=True)
        
        U       = np.sin(psi2/180.*np.pi)/normv
        V       = np.cos(psi2/180.*np.pi)/normv
        # rotate vectors to map projection coordinates
        Upsi, Vpsi, xpsi, ypsi  = m.rotate_vector(U, V, self.lonArr, self.latArr, returnxy=True)
        

        
        dpsi    = abs(psi2 - psiflow)
        # dpsi
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        
        # undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        # return unpsiarr, unphiarr
        # # # ind_outline         = amparr < .2
        
        # 81 % comparison
        # mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        # mask[(unpsiarr>=30.)*(dpsi>=30.)]   = True
        # mask[(amparr<.5)*(dpsi>=30.)]   = True
        # mask[(amparr<.5)*(dpsi>=30.)*(latarr<35.)]   = True
        ###
        
        # mask[(amparr<.2)*(dpsi>=30.)]   = True
        # mask[(amparr<.3)*(dpsi>=30.)*(lonarr<-140.)]   = True
        
        
        xflow   = xflow[np.logical_not(mask)]
        yflow   = yflow[np.logical_not(mask)]
        Uflow   = Uflow[np.logical_not(mask)]
        Vflow   = Vflow[np.logical_not(mask)]
        Upsi    = Upsi[np.logical_not(mask)]
        Vpsi    = Vpsi[np.logical_not(mask)]
        dpsi    = dpsi[np.logical_not(mask)]


        
        Q1      = m.quiver(xflow[dpsi<=30.], yflow[dpsi<=30.], Uflow[dpsi<=30.], Vflow[dpsi<=30.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q2      = m.quiver(xflow[dpsi<=30.], yflow[dpsi<=30.], -Uflow[dpsi<=30.], -Vflow[dpsi<=30.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='b', zorder=1)
        Q1      = m.quiver(xflow[(dpsi>30.)*(dpsi<=60.)], yflow[(dpsi>30.)*(dpsi<=60.)], Uflow[(dpsi>30.)*(dpsi<=60.)], Vflow[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q2      = m.quiver(xflow[(dpsi>30.)*(dpsi<=60.)], yflow[(dpsi>30.)*(dpsi<=60.)], -Uflow[(dpsi>30.)*(dpsi<=60.)], -Vflow[(dpsi>30.)*(dpsi<=60.)],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='lime', zorder=1)
        Q1      = m.quiver(xflow[dpsi>60.], yflow[dpsi>60.], Uflow[dpsi>60.], Vflow[dpsi>60.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)
        Q2      = m.quiver(xflow[dpsi>60.], yflow[dpsi>60.], -Uflow[dpsi>60.], -Vflow[dpsi>60.],\
                           scale=20, width=width+0.002, headaxislength=0, headlength=0, headwidth=0.5, color='r', zorder=1)

        Q1      = m.quiver(xflow, yflow, Upsi, Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        Q2      = m.quiver(xflow, yflow, -Upsi, -Vpsi, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k', zorder=2)
        
        Q1      = m.quiver(xflow, yflow, Upsi, Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        Q2      = m.quiver(xflow, yflow, -Upsi, -Vpsi, scale=20, width=width-0.002, headaxislength=0, headlength=0, headwidth=0.5, color='gold', zorder=2)
        

        plt.suptitle(title, fontsize=20)
        plt.show()
        
        ax      = plt.subplot()
        dbin    = 10.
        # bins    = np.arange(min(dpsi), max(dpsi) + dbin, dbin)
        bins    = np.arange(0., 100., dbin)
        
        weights = np.ones_like(dpsi)/float(dpsi.size)
        # print bins.size
        import pandas as pd
        s = pd.Series(dpsi)
        p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        
        
        p.patches[0].set_color('blue')
        p.patches[1].set_color('blue')
        p.patches[2].set_color('blue')
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
        plt.ylabel('Percentage (%)', fontsize=50)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=10)
        ax.tick_params(axis='x', labelsize=30)
        plt.xticks([0., 10., 20, 30, 40, 50, 60, 70, 80, 90])
        ax.tick_params(axis='y', labelsize=30)
        formatter = FuncFormatter(_model_funcs.to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()
            
        return
    
    
    def plot_amp_sks(self, gindex=1, clabel='', title='', showfig=True, ticks=[]):
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
        self._get_lon_lat_arr()
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex][()]
            unpsi2      = grp['unpsi2_%d' %gindex][()]
            amp         = grp['amp_%d' %gindex][()]
            unamp       = grp['unamp_%d' %gindex][()]
        else:
            plot_axis   = False
        
        mask        = grp.attrs['mask_hti'][()]
        #-----------
        # plot data
        #-----------

        fname       = 'splittingDB.ascii'
        stalst      = []
        philst      = []
        unphilst    = []
        psilst      = []
        unpsilst    = []
        dtlst       = []
        dtlst2       = []
        undtlst2       = []
        lonlst      = []
        latlst      = []
        amplst      = []
        unamplst    = []
        misfit      = self['hti_model/misfit'][()]
        lonlst2     = []
        latlst2     = []
        psilst1     = []
        psilst2     = []
        
        with open(fname, 'r') as fid:
            for line in fid.readlines():
                lst = line.split('|')
                if lst[0] == 'id':
                    continue
                try:
                    #code
                
                    lonsks      = float(lst[3])
                    if lonsks < 0. and self.ilontype == 1:
                        lonsks += 360.
                    elif lonsks > 180. and self.ilontype == 0:
                        lonsks -= 360.
                    latsks      = float(lst[2])
                    if self.minlat > latsks or self.maxlat < latsks or\
                        self.minlon > lonsks or self.maxlon < lonsks:
                        continue
                    
                    ind_lon     = np.where(abs(self.lons - lonsks)<self.dlon)[0][0]
                    ind_lat     = np.where(abs(self.lats - latsks)<self.dlat)[0][0]
                    if mask[ind_lat, ind_lon]:
                        continue
                    tmpphi = float(lst[4])
                    tmpdt  = float(lst[5])
                except Exception:
                    continue
                
                vpr     = self.linear_inv_hti(outlon=self.lons[ind_lon], outlat=self.lats[ind_lat])
                if vpr is None:
                    continue
                vpr.linear_inv_hti(isBcs=True, useref=False, depth_mid_crust=-1, depth_mid_mantle=-1.)
                # vpr.model.htimod.layer_ind
                harr    = vpr.model.h[vpr.model.htimod.layer_ind[1, 0]:vpr.model.htimod.layer_ind[1, 1]]
                vsarr   = vpr.model.vsv[vpr.model.htimod.layer_ind[1, 0]:vpr.model.htimod.layer_ind[1, 1]]
                tamp    = amp[ind_lat, ind_lon]
                temp_dt = ((harr/vsarr/(1.-tamp/100.)).sum() - (harr/vsarr/(1+tamp/100.)).sum())
                tunamp  = unamp[ind_lat, ind_lon]
                temp_undt = ((harr/vsarr/(1.-tunamp/100.)).sum() - (harr/vsarr/(1+tunamp/100.)).sum())
                
                stalst.append(lst[0])
                philst.append(tmpphi)
                dtlst.append(tmpdt)
                lonlst.append(lonsks)
                latlst.append(latsks)
                
                psilst.append(psi2[ind_lat, ind_lon])
                unpsilst.append(unpsi2[ind_lat, ind_lon])
                amplst.append(amp[ind_lat, ind_lon])
                unamplst.append(unamp[ind_lat, ind_lon])
                dtlst2.append(temp_dt)
                undtlst2.append(temp_undt)
                
                temp_misfit = misfit[ind_lat, ind_lon]
                temp_dpsi   = abs(psi2[ind_lat, ind_lon] - float(lst[4]))
                if temp_dpsi > 180.:
                    temp_dpsi   -= 180.
                if temp_dpsi > 90.:
                    temp_dpsi   = 180. - temp_dpsi

        phiarr  = np.array(philst)
        phiarr[phiarr<0.]   += 180.
        psiarr  = np.array(psilst)
        # unphiarr= np.array(unphilst)
        unpsiarr= np.array(unpsilst)
        amparr  = np.array(amplst)
        unamparr= np.array(unamplst)
        dtarr   = np.array(dtlst)
        lonarr  = np.array(lonlst)
        latarr  = np.array(latlst)
        
        dtarr2  = np.array(dtlst2)
        undtarr2  = np.array(undtlst2)
        
        
        mask    = np.zeros(dtarr.size, dtype=bool)
        
        dpsi            = abs(psiarr - phiarr)
        # dpsi
        dpsi[dpsi>180.] -= 180.
        dpsi[dpsi>90.]  = 180.-dpsi[dpsi>90.]
        
        # undpsi          = np.sqrt(unpsiarr**2 + unphiarr**2)
        # return unpsiarr, unphiarr
        # # # ind_outline         = amparr < .2
        
        # 81 % comparison
        # mask[(undpsi>=30.)*(dpsi>=30.)]   = True
        # mask[(unpsiarr>=30.)*(dpsi>=30.)]   = True
        # mask[(amparr<.3)*(dpsi>=30.)*(lonarr < -5.2)]   = True
        # mask[(amparr<.5)*(dpsi>=30.)*(latarr<35.)]   = True
        
        mask[dpsi>30.] = True
        
        ind = np.logical_not(mask)
        # final results
        dtarr = dtarr[ind]
        dtarr2=dtarr2[ind]
        undtarr2=undtarr2[ind]
        

        plt.figure(figsize=[10, 10])
        ax      = plt.subplot()
        # plt.plot(dtlst, dtlst2, 'o', ms=10)
        plt.errorbar(dtarr, dtarr2, yerr=undtarr2, fmt='ko', ms=8)
        # plt.plot([0., 2.5], [0., 2.5], 'b--', lw=3)
        # plt.ylabel('Predicted delay time', fontsize=30)
        # plt.xlabel('Observed delay time', fontsize=60, rotation=0)yerr
        # plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

        # plt.axis(option='equal', ymin=0., ymax=2.5, xmin=0., xmax = 2.5)
        
        import pandas as pd
        # s = pd.Series(dpsi)
        # p = s.plot(kind='hist', bins=bins, color='blue', weights=weights)
        
        diffdata= dtarr - dtarr2
        dbin    = .15
        bins    = np.arange(min(diffdata), max(diffdata) + dbin, dbin)
        
        weights = np.ones_like(diffdata)/float(diffdata.size)
        # print bins.size
        import pandas as pd

        plt.figure()
        ax      = plt.subplot()
        plt.hist(diffdata, bins=bins, weights = weights)
        plt.title('mean = %g , std = %g, mean dt1 = %g, mean dt2 = %g' %(diffdata.mean(), diffdata.std(), dtarr.mean(), dtarr2.mean()) , fontsize=10)

        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        # # # good_per= float(dpsi[dpsi<30.].size)/float(dpsi.size)
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Delay time difference (s)', fontsize=60, rotation=0)
        # plt.title('mean = %g , std = %g, good = %g' %(dpsi.mean(), dpsi.std(), good_per*100.) + '%', fontsize=30)
        ax.tick_params(axis='x', labelsize=30)
        # plt.xticks([-2., -1.5, -1., -.5, 0.])
        ax.tick_params(axis='y', labelsize=30)
        formatter = FuncFormatter(_model_funcs.to_percent)
        # # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        # plt.xlim([-2., 0.])
        plt.show()
            
        return

    def plot_misfit(self, dattype = 'all', cmap='hot_r', factor=1., vmin=None, vmax=None, masked=True, clabel='', title='', projection='merc', showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr()
        grp         = self['hti_model']
        
        mask        = grp.attrs['mask_hti'][()]
        if dattype == 'all':
            data    = grp['misfit'][()]
        elif dattype == 'psi':
            data    = grp['psi_misfit'][()]
        elif dattype == 'amp':
            data    = grp['amp_misfit'][()]
        elif dattype == 'adp':
            data    = grp['adaptive'][()]
        mdata       = ma.masked_array(data/factor, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        
            # cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[10., 15, 20, 25, 30, 35, 40])
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=20)
        cb.set_alpha(1)
        cb.draw_all()
        
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)
        
        # if plotfault:
        #     if projection == 'lambert':
        #         shapefname  = '/raid/lili/data_marin/map_data/geological_maps/qfaults'
        #         m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
        #         m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
            # # # else:
            # # #     shapefname  = '/home/lili/code/gem-global-active-faults/shapefile/gem_active_faults'
            # # #     # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
            # # #     m.readshapefile(shapefname, 'faultline', linewidth = 2., color='grey', default_encoding='windows-1252')
        if projection == 'merc' and os.path.isdir('/home/lili/spain_proj/geo_maps'):
            shapefname  = '/home/lili/spain_proj/geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='grey')
            
            
        # # # shapefname  = '/raid/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
        # # # shplst      = shapefile.Reader(shapefname)
        # # # for rec in shplst.records():
        # # #     lon_vol = rec[4]
        # # #     lat_vol = rec[3]
        # # #     xvol, yvol            = m(lon_vol, lat_vol)
        # # #     m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        # m.shadedrelief(scale=1., origin='lower')
        
        print (mdata.mean(), mdata.max())
        if showfig:
            plt.show()
        return
    
    def plot_misfit_exclude(self, dattype = 'all', cmap='hot_r', factor=1., vmin=None, vmax=None, masked=True,\
            lon_plt = [], lat_plt = [], clabel='', title='', projection='merc', showfig=True, ticks=[]):

        self._get_lon_lat_arr()
        grp         = self['hti_model']
        
        mask        = grp.attrs['mask_hti'][()]
        if dattype == 'all':
            data    = grp['misfit'][()]
        elif dattype == 'psi':
            data    = grp['psi_misfit'][()]
        elif dattype == 'amp':
            data    = grp['amp_misfit'][()]
        elif dattype == 'adp':
            data    = grp['adaptive'][()]
            
            
        # if not (latsks > 34.8 and latsks < 43.2 and lonsks < 0. and lonsks > -5.1):
                        
        ind0 = (self.lonArr < -1.32) * (self.latArr > 43.2)
        data[ind0] = 0.75
        ind0 = (self.lonArr > -5.1) * (self.latArr < 36.5) *(self.latArr >34.8) *(self.lonArr < 0.)
        data[ind0] /= factor
        
        ind1 = (self.latArr > 36.5) *(self.latArr < 38.)*data > 1.5
        data[ind1] = 1.1
        
        ind0 = (self.lonArr > -5.1) * (self.latArr < 43.2) *(self.latArr >38.) *(self.lonArr < 0.)
        data[ind0] /= factor
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        
            # cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[10., 15, 20, 25, 30, 35, 40])
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=20)
        cb.set_alpha(1)
        cb.draw_all()
        
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)

        if projection == 'merc' and os.path.isdir('/home/lili/spain_proj/geo_maps'):
            shapefname  = '/home/lili/spain_proj/geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='grey')
            
        if len(lon_plt) == len(lat_plt) and len(lon_plt) >0:
            xc, yc      = m(lon_plt, lat_plt)
            m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
  
        print (mdata.mean(), mdata.max())
        if showfig:
            plt.show()
        return
    
    def plot_diff_psi(self, gindex1 = 1, gindex2 = 2, cmap='hot_r', factor=1., vmin=None, vmax=None, masked=True, clabel='', title='', projection='merc', showfig=True, ticks=[]):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        self._get_lon_lat_arr()
        grp         = self['hti_model']
        psi1        = grp['psi2_%d' %gindex1][()]
        unpsi1      = grp['unpsi2_%d' %gindex1][()]
        amp1        = grp['amp_%d' %gindex1][()]
        unamp1      = grp['unamp_%d' %gindex1][()]
        
        psi2        = grp['psi2_%d' %gindex2][()]
        unpsi2      = grp['unpsi2_%d' %gindex2][()]
        amp2        = grp['amp_%d' %gindex2][()]
        unamp2      = grp['unamp_%d' %gindex2][()]
        mask        = grp.attrs['mask_hti'][()]
        
        data            = abs(psi1 - psi2)
        data[data>90.]  = 180. - data[data>90.]
        
        
        mdata       = ma.masked_array(data/factor, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        
            # cb              = m.colorbar(im, location='bottom', size="3%", pad='2%', ticks=[10., 15, 20, 25, 30, 35, 40])
        cb.set_label(clabel, fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=20)
        cb.set_alpha(1)
        cb.draw_all()
        
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)
        
        
        if projection == 'merc' and os.path.isdir('/home/lili/spain_proj/geo_maps'):
            shapefname  = '/home/lili/spain_proj/geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='grey')
            
        
        print (mdata.mean(), mdata.max())
        if showfig:
            plt.show()
        return
        

    def plot_hti_vel_misfit(self, infname, depth, thresh = 0.8, depthavg=3., gindex=1, dattype='psi', plot_axis=True, plot_data=True, factor=10, normv=5., width=0.006, ampref=0.5, \
                 scaled=True, masked=True, clabel='', title='', cmap='surf', projection='merc', \
                    vmin=None, vmax=None, showfig=True, ticks=[], lon_plt=[], lat_plt=[], gwidth = -1.):
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
        import h5py
        self._get_lon_lat_arr()
        grp         = self['hti_model']
        if gindex >=0:
            psi2        = grp['psi2_%d' %gindex][()]
            unpsi2      = grp['unpsi2_%d' %gindex][()]
            amp         = grp['amp_%d' %gindex][()]
            unamp       = grp['unamp_%d' %gindex][()]
        else:
            plot_axis   = False
            
        dset        = h5py.File(infname)
        grp2        = dset['hti_model']
        if dattype == 'all':
            misfit    = grp2['misfit'][()]
        elif dattype == 'psi':
            misfit    = grp2['psi_misfit'][()]
        elif dattype == 'amp':
            misfit    = grp2['amp_misfit'][()]
        elif dattype == 'adp':
            misfit    = grp2['adaptive'][()]
        
        mask        = grp.attrs['mask_hti'][()]

        grp         = self['avg_paraval']
        vs3d        = grp['vsv_smooth'][()]
        zArr        = grp['z_smooth'][()]
        if depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            data    = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print ('depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km')
                return
            depth       = zArr[index]
            data        = vs3d[:, :, index]
        
        if gwidth > 0.:
            gridder     = _grid_class.SphereGridder(minlon = self.minlon, maxlon = self.maxlon, dlon = self.dlon, \
                            minlat = self.minlat, maxlat = self.maxlat, dlat = self.dlat, period = 10., \
                            evlo = 0., evla = 0., fieldtype = 'Tph', evid = 'plt')
            gridder.read_array(inlons = self.lonArr[np.logical_not(mask)], inlats = self.latArr[np.logical_not(mask)], inzarr = data[np.logical_not(mask)])
            outfname    = 'plt_Tph.lst'
            prefix      = 'plt_Tph_'
            gridder.gauss_smoothing(workingdir = './temp_plt', outfname = outfname, width = gwidth)
            data[:]     = gridder.Zarr
        
        ind1 = (self.latArr > 35.1) * (self.latArr < 43.2) * (self.lonArr > -5.1) * (self.lonArr < 0.)
        ind2 = (self.latArr > 36.5) * (self.latArr < 38.5)
        ind3 = (self.latArr > 36.7)
        
        mask1 = mask.copy()
        mask1[np.logical_not(ind1)] = True
        mask1[ind2] = True
        mask1[ind3] = True
        mask1[misfit < thresh] = True
        mask1[self.lonArr>-2.] = True
        mask1[(self.latArr < 36.8) * (self.lonArr >-2.3)] = True
        

        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        if plot_data:
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
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            cb          = m.colorbar(im,  "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=60, rotation=0)
            cb.ax.tick_params(labelsize=20)
            cb.set_alpha(1)
            cb.draw_all()
        m.fillcontinents(color='silver', lake_color='none',zorder=0.2, alpha=1.)
        m.drawcountries(linewidth=1.)
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
            U, V, x, y  = m.rotate_vector(U, V, self.lonArr, self.latArr, returnxy=True)
            
            #--------------------------------------
            # plot fast axis
            #--------------------------------------
            x_psi       = x.copy()
            y_psi       = y.copy()
            mask_psi    = mask1
            if factor!=None:
                x_psi   = x_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
                y_psi   = y_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
                U       = U[0:self.Nlat:factor, 0:self.Nlon:factor]
                V       = V[0:self.Nlat:factor, 0:self.Nlon:factor]
                mask_psi= mask_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
            if masked:
                U   = ma.masked_array(U, mask=mask_psi )
                V   = ma.masked_array(V, mask=mask_psi )

            Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
            Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, facecolor='y')
            if scaled:

                x_ref, y_ref = m(2.5, 32.)
                Uref    = 2./ampref/normv
                Vref    = 0.
                Q1      = m.quiver(x_ref, y_ref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                Q2      = m.quiver(x_ref, y_ref, -Uref, -Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
                Q1      = m.quiver(x_ref, y_ref, Uref, Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
                Q2      = m.quiver(x_ref, y_ref, -Uref, -Vref, scale=20, width=width-0.003, headaxislength=0, headlength=0, headwidth=0.5, color='y')
        ##
        if projection == 'merc' and os.path.isdir('/home/lili/spain_proj/geo_maps'):
            shapefname  = '/home/lili/spain_proj/geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='grey')
        plt.suptitle(title, fontsize=20)
        
        
        # xc, yc      = m(np.array([-153.]), np.array([66.1]))
        # m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')
        # azarr       = np.arange(36.)*10.
        
        if len(lon_plt) == len(lat_plt) and len(lon_plt) >0:
            xc, yc      = m(lon_plt, lat_plt)
            m.plot(xc, yc,'*', ms = 20, markeredgecolor='black', markerfacecolor='yellow')

        
        if showfig:
            plt.show()
        return
    
    def plot_aniso_etopo(self,  thresh=0.8, projection='merc', lonplt=[], latplt=[], plotfault = True, showfig=True, dattype='psi'):
        
        self._get_lon_lat_arr()

        grp        = self['hti_model']
        if dattype == 'all':
            misfit    = grp['misfit'][()]
        elif dattype == 'psi':
            misfit    = grp['psi_misfit'][()]
        elif dattype == 'amp':
            misfit    = grp['amp_misfit'][()]
        elif dattype == 'adp':
            misfit    = grp['adaptive'][()]
        
        mask        = grp.attrs['mask_hti'][()]

        ind1        = (self.latArr > 38.5) * (self.latArr < 43.2) * (self.lonArr > -5.1) * (self.lonArr < -2.)*(misfit >= thresh)
        ind2        = (self.latArr > 34.8) * (self.latArr < 36.5) * (self.lonArr > -5.1) * (self.lonArr < -2.5)*(misfit >= thresh)
        
        ind3        = np.logical_not(mask)*np.logical_not(ind1) *np.logical_not(ind2)
        
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)

        if projection == 'merc' and os.path.isdir('/home/lili/spain_proj/geo_maps'):
            shapefname  = '/home/lili/spain_proj/geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
        
        
        from netCDF4 import Dataset
        from matplotlib.colors import LightSource
        import pycpt
        etopodata   = Dataset('/raid/lili/data_spain/GEBCO_2020_30_Mar_2021_30cd972b6f07/gebco_2020_n47.0_s27.0_w-12.0_e8.0.nc')
        etopo       = (etopodata.variables['elevation'][:]).data
        lons        = (etopodata.variables['lon'][:]).data
        lons[lons>180.] = lons[lons>180.] - 360.
        lats        = (etopodata.variables['lat'][:]).data


        ls          = LightSource(azdeg=315, altdeg=45)
        # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
        # etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
        # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
        ny, nx      = etopo.shape
        topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
        m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
        mycm1       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/etopo1.cpt_land')

        mycm2       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/bathy1.cpt')
        mycm2.set_over('w',0)

        m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0., vmax=5000.))
        m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000., vmax=-0.5))
        
        im          = plt.scatter(x[ind1], y[ind1], s=70,  c='blue', edgecolors='none', alpha=0.5, marker='s')
        im          = plt.scatter(x[ind2], y[ind2], s=70,  c='red', edgecolors='none', alpha=0.5, marker='s')
        im          = plt.scatter(x[ind3], y[ind3], s=70,  c='green', edgecolors='none', alpha=0.5, marker='s')
         
        
  
        if showfig:
            plt.show()
        return
    
    def plot_etopo(self,  thresh=0.8, projection='merc', lon_plt=[], lat_plt=[], plotfault = True, showfig=True, dattype='psi'):
        
        self._get_lon_lat_arr()

        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)

        if projection == 'merc' and os.path.isdir('/home/lili/spain_proj/geo_maps'):
            shapefname  = '/home/lili/spain_proj/geo_maps/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 3, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 1.5, color='white')
        
        
        from netCDF4 import Dataset
        from matplotlib.colors import LightSource
        import pycpt
        etopodata   = Dataset('/raid/lili/data_spain/GEBCO_2020_30_Mar_2021_30cd972b6f07/gebco_2020_n47.0_s27.0_w-12.0_e8.0.nc')
        etopo       = (etopodata.variables['elevation'][:]).data
        lons        = (etopodata.variables['lon'][:]).data
        lons[lons>180.] = lons[lons>180.] - 360.
        lats        = (etopodata.variables['lat'][:]).data


        ls          = LightSource(azdeg=315, altdeg=45)
        # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
        # etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
        # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
        ny, nx      = etopo.shape
        topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
        m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
        mycm1       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/etopo1.cpt_land')

        mycm2       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/bathy1.cpt')
        mycm2.set_over('w',0)

        m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0., vmax=5000.))
        m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000., vmax=-0.5))
        if len(lon_plt) == len(lat_plt) and len(lon_plt) >0:
            xc, yc      = m(lon_plt, lat_plt)
            m.plot(xc, yc,'s', ms = 10, markeredgecolor='black', markerfacecolor='blue', zorder = 100)
  
        if showfig:
            plt.show()
        return
