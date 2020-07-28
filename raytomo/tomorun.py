# -*- coding: utf-8 -*-
"""
HDF5 database for ray tomography, running part
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""

import surfpy.raytomo.tomobase as tomobase
import surfpy.raytomo.bin as raytomo_bin
bin_path    = raytomo_bin.__path__._path[0]


import numpy as np
import multiprocessing
from datetime import datetime
import shutil
import glob
import sys
import copy
from subprocess import call
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
    

class runh5(tomobase.baseh5):
    """ Class for performing ray tomography
    =================================================================================================================
    version history:
        2020/07/27
    =================================================================================================================
    """
    
    def creat_reshape_data(self, runtype = 0, runid = 0):
        """convert data to Nlat * Nlon shape and store the mask
        =================================================================================================================
        ::: input parameters :::
        runtype         - type of run
                            0 - smooth run
                            1 - quality controlled run
        runid           - id of run
        =================================================================================================================
        """
        rundict     = {0: 'smooth_run', 1: 'qc_run'}
        dataid      = rundict[runtype]+'_'+str(runid)
        ingroup     = self[dataid]
        pers        = self.attrs['period_array']
        self._get_lon_lat_arr(dataid=dataid)
        ingrp       = self[dataid]
        outgrp      = self.create_group( name = 'reshaped_'+dataid)
        if runtype == 1:
            isotropic   = ingrp.attrs['isotropic']
            outgrp.attrs.create(name = 'isotropic', data=isotropic)
        else:
            isotropic   = True
        #-----------------
        # mask array
        #-----------------
        if not isotropic:
            mask1       = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
            mask2       = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
            tempgrp     = ingrp['%g_sec'%( pers[0] )]
            # get value for mask1 array
            lonlat_arr1 = tempgrp['lons_lats'].value
            inlon       = lonlat_arr1[:,0]
            inlat       = lonlat_arr1[:,1]
            for i in range(inlon.size):
                lon                         = inlon[i]
                lat                         = inlat[i]
                # index                       = np.where((self.lonArr==lon)*(self.latArr==lat))
                index                       = np.where((abs(self.lonArr-lon)<0.001)*(abs(self.latArr-lat)<0.001))
                mask1[index[0], index[1]]   = False
            # get value for mask2 array
            lonlat_arr2 = tempgrp['lons_lats_rea'].value
            inlon       = lonlat_arr2[:,0]
            inlat       = lonlat_arr2[:,1]
            for i in range(inlon.size):
                lon                         = inlon[i]
                lat                         = inlat[i]
                # index                       = np.where((self.lonArr==lon)*(self.latArr==lat))
                index                       = np.where((abs(self.lonArr-lon)<0.001)*(abs(self.latArr-lat)<0.001))
                mask2[index[0], index[1]]   = False
            outgrp.create_dataset(name='mask1', data=mask1)
            outgrp.create_dataset(name='mask2', data=mask2)
            index1      = np.logical_not(mask1)
            index2      = np.logical_not(mask2)
            anipara     = ingroup.attrs['anipara']
        # loop over periods
        for per in pers:
            # get data
            pergrp  = ingrp['%g_sec'%( per )]
            try:
                velocity        = pergrp['velocity'].value
                dv              = pergrp['Dvelocity'].value
                azicov          = pergrp['azi_coverage'].value
                pathden         = pergrp['path_density'].value
                if not isotropic:
                    resol       = pergrp['resolution'].value
            except:
                raise AttributeError(str(per)+ ' sec data does not exist!')
            # save data
            opergrp         = outgrp.create_group(name='%g_sec'%( per ))
            if isotropic:
                # velocity
                outv        = velocity.reshape(self.Nlat, self.Nlon)
                v0dset      = opergrp.create_dataset(name='velocity', data=outv)
                v0dset.attrs.create(name='Nlat', data=self.Nlat)
                v0dset.attrs.create(name='Nlon', data=self.Nlon)
                # relative velocity perturbation
                outdv       = dv.reshape(self.Nlat, self.Nlon)
                dvdset      = opergrp.create_dataset(name='Dvelocity', data=outdv)
                dvdset.attrs.create(name='Nlat', data=self.Nlat)
                dvdset.attrs.create(name='Nlon', data=self.Nlon)
                # azimuthal coverage, squared sum
                outazicov   = (azicov[:, 0]).reshape(self.Nlat, self.Nlon)
                azidset     = opergrp.create_dataset(name='azi_coverage1', data=outazicov)
                azidset.attrs.create(name='Nlat', data=self.Nlat)
                azidset.attrs.create(name='Nlon', data=self.Nlon)
                # azimuthal coverage, max value
                outazicov   = (azicov[:, 1]).reshape(self.Nlat, self.Nlon)
                azidset     = opergrp.create_dataset(name='azi_coverage2', data=outazicov)
                azidset.attrs.create(name='Nlat', data=self.Nlat)
                azidset.attrs.create(name='Nlon', data=self.Nlon)
                # path density
                outpathden  = pathden.reshape(self.Nlat, self.Nlon)
                pddset      = opergrp.create_dataset(name='path_density', data=outpathden)
                pddset.attrs.create(name='Nlat', data=self.Nlat)
                pddset.attrs.create(name='Nlon', data=self.Nlon)
            else:
                # isotropic velocity
                outv_iso        = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outv_iso[index1]= velocity[:, 0]
                v0dset          = opergrp.create_dataset(name='vel_iso', data=outv_iso)
                v0dset.attrs.create(name='Nlat', data=self.Nlat)
                v0dset.attrs.create(name='Nlon', data=self.Nlon)
                # relative velocity perturbation
                outdv           = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outdv[index1]   = dv
                dvdset          = opergrp.create_dataset(name='dv', data=outdv)
                dvdset.attrs.create(name='Nlat', data=self.Nlat)
                dvdset.attrs.create(name='Nlon', data=self.Nlon)
                if anipara != 0:
                    # azimuthal amplitude for 2psi
                    outamp2         = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                    outamp2[index1] = velocity[:, 3]
                    amp2dset        = opergrp.create_dataset(name='amp2', data=outamp2)
                    amp2dset.attrs.create(name='Nlat', data=self.Nlat)
                    amp2dset.attrs.create(name='Nlon', data=self.Nlon)
                    # psi2
                    outpsi2         = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                    outpsi2[index1] = velocity[:, 4]
                    psi2dset        = opergrp.create_dataset(name='psi2', data=outpsi2)
                    psi2dset.attrs.create(name='Nlat', data=self.Nlat)
                    psi2dset.attrs.create(name='Nlon', data=self.Nlon)
                if anipara == 2:
                    # azimuthal amplitude for 4psi
                    outamp4         = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                    outamp4[index1] = velocity[:, 7]
                    amp4dset        = opergrp.create_dataset(name='amp4', data=outamp4)
                    amp4dset.attrs.create(name='Nlat', data=self.Nlat)
                    amp4dset.attrs.create(name='Nlon', data=self.Nlon)
                    # psi4
                    outpsi4         = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                    outpsi4[index1] = velocity[:, 8]
                    psi4dset        = opergrp.create_dataset(name='psi4', data=outpsi4)
                    psi4dset.attrs.create(name='Nlat', data=self.Nlat)
                    psi4dset.attrs.create(name='Nlon', data=self.Nlon)
                # azimuthal coverage, squared sum
                outazicov           = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outazicov[index1]   = azicov[:, 0]
                azidset             = opergrp.create_dataset(name='azi_coverage1', data=outazicov)
                azidset.attrs.create(name='Nlat', data=self.Nlat)
                azidset.attrs.create(name='Nlon', data=self.Nlon)
                # azimuthal coverage, max value
                outazicov           = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outazicov[index1]   = azicov[:, 1]
                azidset             = opergrp.create_dataset(name='azi_coverage2', data=outazicov)
                azidset.attrs.create(name='Nlat', data=self.Nlat)
                azidset.attrs.create(name='Nlon', data=self.Nlon)
                # path density, all orbits
                outpathden          = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outpathden[index1]  = pathden[:, 0]
                pddset              = opergrp.create_dataset(name='path_density', data=outpathden)
                pddset.attrs.create(name='Nlat', data=self.Nlat)
                pddset.attrs.create(name='Nlon', data=self.Nlon)
                # path density, first orbit
                outpathden          = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outpathden[index1]  = pathden[:, 1]
                pddset              = opergrp.create_dataset(name='path_density1', data=outpathden)
                pddset.attrs.create(name='Nlat', data=self.Nlat)
                pddset.attrs.create(name='Nlon', data=self.Nlon)
                # path density, second orbit
                outpathden          = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outpathden[index1]  = pathden[:, 2]
                pddset              = opergrp.create_dataset(name='path_density2', data=outpathden)
                pddset.attrs.create(name='Nlat', data=self.Nlat)
                pddset.attrs.create(name='Nlon', data=self.Nlon)
                # resolution analysis, cone radius
                outrea              = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outrea[index2]      = resol[:, 0]
                readset             = opergrp.create_dataset(name='cone_radius', data=outrea)
                readset.attrs.create(name='Nlat', data=self.Nlat)
                readset.attrs.create(name='Nlon', data=self.Nlon)
                # resolution analysis, Gaussian std
                outrea              = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outrea[index2]      = resol[:, 1]
                readset             = opergrp.create_dataset(name='gauss_std', data=outrea)
                readset.attrs.create(name='Nlat', data=self.Nlat)
                readset.attrs.create(name='Nlon', data=self.Nlon)
                # resolution analysis, maximum response value
                outrea              = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outrea[index2]      = resol[:, 2]
                readset             = opergrp.create_dataset(name='max_resp', data=outrea)
                readset.attrs.create(name='Nlat', data=self.Nlat)
                readset.attrs.create(name='Nlon', data=self.Nlon)
                # resolution analysis, number of cells involved in cone base
                outrea              = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outrea[index2]      = resol[:, 3]
                readset             = opergrp.create_dataset(name='ncone', data=outrea)
                readset.attrs.create(name='Nlat', data=self.Nlat)
                readset.attrs.create(name='Nlon', data=self.Nlon)
                # resolution analysis, number of cells involved in Gaussian construction
                outrea              = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
                outrea[index2]      = resol[:, 4]
                readset             = opergrp.create_dataset(name='ngauss', data=outrea)
                readset.attrs.create(name='Nlat', data=self.Nlat)
                readset.attrs.create(name='Nlon', data=self.Nlon)
        return
    
    def run_smooth(self, datadir, outdir, datatype='ph', channel='ZZ', dlon=0.5, dlat=0.5, \
            stepinte=0.2, lengthcell=1.0, alpha1=3000, alpha2=100, sigma=500,
            runid=0, comments='', deletetxt=False, contourfname='./contour.ctr', reshape=True):
        """
        run Misha's tomography code with large regularization parameters.
        This function is designed to do an inital test run, the output can be used to discard outliers in aftan results.
        =================================================================================================================
        ::: input parameters :::
        datadir/outdir      - data/output directory
        datatype            - ph: phase velocity inversion, gr: group velocity inversion
        channel             - channel for analysis (default: ZZ, xcorr ZZ component)
        dlon/dlat           - longitude/latitude interval
        stepinte            - step of integration (degree), works only for Gaussian method
        lengthcell          - size of main cell (degree)
        alpha1,alpha2,sigma - regularization parameters for isotropic tomography
                                alpha1  : smoothing coefficient
                                alpha2  : path density damping
                                sigma   : Gaussian smoothing (radius of correlation)
        runid               - id number for the run
        comments            - comments for the run
        deletetxt           - delete txt output or not
        contourfname        - path to contour file (see the manual for detailed description)
        IsoMishaexe         - path to Misha's Tomography code executable (isotropic version)
        ------------------------------------------------------------------------------------------------------------------
        input format:
        datadir/data_pfx+'%g'%( per ) +'_'+channel+'_'+datatype+'.lst' (e.g. datadir/raytomo_10_ZZ_ph.lst)
        e.g. datadir/MISHA_in_20.0_BHZ_BHZ_ph.lst
        
        output format:
        e.g. 
        prefix: outdir/10_ph/N_INIT_3000_500_100
        output file: outdir/10.0_ph/N_INIT_3000_500_100_10.0.1 etc. 
        =================================================================================================================
        """
        IsoMishaexe     = bin_path+'/itomo_sp_cu_shn'
        if not os.path.isfile(IsoMishaexe):
            raise AttributeError('IsoMishaexe does not exist!')
        if not os.path.isfile(contourfname):
            raise AttributeError('Contour file does not exist!')
        self.update_attrs()
        pers            = self.pers
        minlon          = self.minlon
        maxlon          = self.maxlon
        minlat          = self.minlat
        maxlat          = self.maxlat
        data_pfx        = self.data_pfx
        smooth_pfx      = self.smooth_pfx
        if not os.path.isdir(outdir):
            deleteall   = True
        #-----------------------------------------
        # run the tomography code for each period
        #-----------------------------------------
        print('================================= Smooth run of surface wave tomography ==================================')
        for per in pers:
            print('----------------------------------------------------------------------------------------------------------')
            print('----------------------------------------- T = %3d sec ----------------------------------------------------' %per)
            print('----------------------------------------------------------------------------------------------------------')
            infname     = datadir+'/'+data_pfx+'%g'%( per ) +'_'+channel+'_'+datatype+'.lst'
            outper      = outdir+'/'+'%g'%( per ) +'_'+datatype
            if not os.path.isdir(outper):
                os.makedirs(outper)
            outpfx      = outper+'/'+smooth_pfx+str(alpha1)+'_'+str(sigma)+'_'+str(alpha2)
            temprunsh   = 'temp_'+'%g_Smooth.sh' %(per)
            with open(temprunsh,'w') as f:
                f.writelines('%s %s %s %g <<-EOF\n' %(IsoMishaexe, infname, outpfx, per ))
                f.writelines('me \n4 \n5 \n%g \n6 \n%g \n%g \n%g \n' %( alpha2, alpha1, sigma, sigma) )
                f.writelines('7 \n%g %g %g \n8 \n%g %g %g \n12 \n%g \n%g \n16 \n' %(minlat, maxlat, dlat, minlon, maxlon, dlon, stepinte, lengthcell) )
                f.writelines('v \nq \ngo \nEOF \n' )
            call(['bash', temprunsh])
            os.remove(temprunsh)
        #================================
        # save results to hdf5 dataset
        #================================
        create_group        = False
        while (not create_group):
            try:
                group       = self.create_group( name = 'smooth_run_'+str(runid) )
                create_group= True
            except:
                runid       += 1
                continue
        group.attrs.create(name = 'comments', data=comments)
        group.attrs.create(name = 'dlon', data=dlon)
        group.attrs.create(name = 'dlat', data=dlat)
        group.attrs.create(name = 'step_of_integration', data=stepinte)
        group.attrs.create(name = 'datatype', data=datatype)
        group.attrs.create(name = 'channel', data=channel)
        group.attrs.create(name = 'alpha1', data=alpha1)
        group.attrs.create(name = 'alpha2', data=alpha2)
        group.attrs.create(name = 'sigma', data=sigma)
        for per in pers:
            subgroup    = group.create_group(name='%g_sec'%( per ))
            outper      = outdir+'/'+'%g'%( per ) +'_'+datatype
            outpfx      = outper+'/'+smooth_pfx+str(alpha1)+'_'+str(sigma)+'_'+str(alpha2)
            # absolute velocity
            v0fname     = outpfx+'_%g.1' %(per)
            inArr       = np.loadtxt(v0fname)
            v0Arr       = inArr[:,2]
            v0dset      = subgroup.create_dataset(name='velocity', data=v0Arr)
            # relative velocity perturbation
            dvfname     = outpfx+'_%g.1' %(per)+'_%_'
            inArr       = np.loadtxt(dvfname)
            dvArr       = inArr[:,2]
            dvdset      = subgroup.create_dataset(name='Dvelocity', data=dvArr)
            # azimuthal coverage
            azifname    = outpfx+'_%g.azi' %(per)
            inArr       = np.loadtxt(azifname)
            aziArr      = inArr[:,2:4]
            azidset     = subgroup.create_dataset(name='azi_coverage', data=aziArr)
            # residual file
            # id fi0 lam0 f1 lam1 vel_obs weight res_tomo res_mod delta
            residfname  = outpfx+'_%g.resid' %(per)
            inArr       = np.loadtxt(residfname)
            residdset   = subgroup.create_dataset(name='residual', data=inArr)
            # path density file
            resfname    = outpfx+'_%g.res' %(per)
            inArr       = np.loadtxt(resfname)
            resArr      = inArr[:,2:]
            resdset     = subgroup.create_dataset(name='path_density', data=resArr)
            if deletetxt:
                shutil.rmtree(outper)
        if deletetxt and deleteall:
            shutil.rmtree(outdir)
        if reshape:
            self.creat_reshape_data(runtype=0, runid=runid)
        print('================================= End smooth run of surface wave tomography ===============================')
        return
    
    
    