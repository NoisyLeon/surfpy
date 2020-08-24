# -*- coding: utf-8 -*-
"""
HDF5 database for ray tomography, I/O part
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import obspy
import numpy as np
import numpy.ma as ma
import h5py
import os
import surfpy.cpt_files as cpt_files
cpt_path    = cpt_files.__path__._path[0]
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt
import surfpy.raytomo._tomo_funcs as _tomo_funcs

class baseh5(h5py.File):
    """
    =================================================================================================================
    version history:
        
    =================================================================================================================
    """
    def __init__(self, name, mode='a', driver=None, libver=None, userblock_size=None, swmr=False,\
            rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None, **kwds):
        super(baseh5, self).__init__( name, mode, driver, libver, userblock_size,\
            swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order)
        #======================================
        # initializations of attributes
        #======================================
        self.update_attrs()
        # try:
        #     self.datapfx    = self.attrs['data_pfx']
            
        # self.inv        = obspy.Inventory()
        # self.start_date = obspy.UTCDateTime('2599-01-01')
        # self.end_date   = obspy.UTCDateTime('1900-01-01')
        # self.update_inv_info()
        return 
        
    # def __del__(self):
    #     """
    #     Cleanup. Force flushing and close the file.
    #     If called with MPI this will also enable MPI to cleanly shutdown.
    #     """
    #     try:
    #         self.flush()
    #         self.close()
    #     except (ValueError, TypeError, AttributeError):
    #         pass
    
    def update_attrs(self):
        try:
            self.data_pfx   = self.attrs['data_prefix']
            self.smooth_pfx = self.attrs['smooth_prefix']
            self.qc_pfx     = self.attrs['qc_prefix']
            self.pers       = self.attrs['period_array']
            self.minlon     = self.attrs['minlon']
            self.maxlon     = self.attrs['maxlon']
            self.minlat     = self.attrs['minlat']
            self.maxlat     = self.attrs['maxlat']
        except:
            pass
        return
    
    def update_data(self):
        return
    
    def _get_lon_lat_arr(self, dataid, sfx=''):
        """Get longitude/latitude array
        """
        self.update_attrs()
        if sfx == '':
            dlon                = self[dataid].attrs['dlon']
            dlat                = self[dataid].attrs['dlat']
        else:
            dlon                = self[dataid].attrs['dlon_'+sfx]
            dlat                = self[dataid].attrs['dlat_'+sfx]
        self.lons               = np.arange(int((self.maxlon-self.minlon)/dlon)+1)*dlon+self.minlon
        self.lats               = np.arange(int((self.maxlat-self.minlat)/dlat)+1)*dlat+self.minlat
        self.Nlon               = self.lons.size
        self.Nlat               = self.lats.size
        self.lonArr, self.latArr= np.meshgrid(self.lons, self.lats)
        if self.lons[0] != self.minlon or self.lons[-1] != self.maxlon \
            or self.lats[0] != self.minlat or self.lats[-1] != self.maxlat:
            raise ValueError('!!! longitude/latitude arrays not consistent with bounds')
        return
    #==================================================
    # functions print the information of database
    #==================================================
    def print_attrs(self, print_to_screen=True):
        """
        Print the attrsbute information of the dataset.
        """
        outstr          = '================================= Surface wave ray tomography Database ==================================\n'
        try:
            outstr      += 'Input data prefix       - '+self.attrs['data_prefix']+'\n'
            outstr      += 'Smooth run prefix       - '+self.attrs['smooth_prefix']+'\n'
            outstr      += 'QC run prefix           - '+self.attrs['qc_prefix']+'\n'
            outstr      += 'Period(s):              - '+str(self.attrs['period_array'])+'\n'
            outstr      += 'Longitude range         - '+str(self.attrs['minlon'])+' ~ '+str(self.attrs['maxlon'])+'\n'
            outstr      += 'Latitude range          - '+str(self.attrs['minlat'])+' ~ '+str(self.attrs['maxlat'])+'\n'
            self.update_attrs()
        except:
            print ('Empty Database!')
            return None
        if print_to_screen:
            print (outstr)
        else:
            return (outstr)
        return
    
    def print_smooth_info(self, runid = 0):
        """print all the data stored in the smooth run
        """
        outstr      = '----------------------------------------- Smooth run data : id = '+str(runid)+' -----------------------------------------------\n'
        try:
            subgroup1   = self['smooth_run_%d' %runid]
            subgroup2   = self['reshaped_smooth_run_%d' %runid]
        except KeyError:
            print ('*** No data for smooth run id = '+str(runid))
            return
        outstr          += 'Channel                             - '+str(subgroup1.attrs['channel'])+'\n'
        outstr          += 'datatype(ph: phase; gr: group)      - '+str(subgroup1.attrs['datatype'])+'\n'
        outstr          += 'dlon, dlat                          - '+str(subgroup1.attrs['dlon'])+', '+str(subgroup1.attrs['dlat'])+'\n'
        outstr          += 'Step of integration                 - '+str(subgroup1.attrs['step_of_integration'])+'\n'
        outstr          += 'Smoothing coefficient (alpha1)      - '+str(subgroup1.attrs['alpha1'])+'\n'
        outstr          += 'Path density damping (alpha2)       - '+str(subgroup1.attrs['alpha2'])+'\n'
        outstr          += 'radius of correlation (sigma)       - '+str(subgroup1.attrs['sigma'])+'\n'
        outstr          += 'Comments                            - '+str(subgroup1.attrs['comments'])+'\n'
        outstr          += '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n'
        perid           = list(subgroup1.keys())[0]
        outstr          += '*** Dvelocity:                      size    = '+str(subgroup1[perid]['Dvelocity'][()].shape)+ '; shape = '+\
                            str(subgroup2[perid]['Dvelocity'][()].shape)+'\n'
        outstr          += '*** velocity:                       size    = '+str(subgroup1[perid]['velocity'][()].shape)+ '; shape = '+\
                            str(subgroup2[perid]['velocity'][()].shape)+'\n'
        outstr          += '*** azi_coverage:                   size    = '+str(subgroup1[perid]['azi_coverage'][()].shape)+'\n'
        outstr          += '*** azi_coverage1 (squared sum):    shape   = '+str(subgroup2[perid]['azi_coverage1'][()].shape)+'\n'
        outstr          += '*** azi_coverage2 (max value):      shape   = '+str(subgroup2[perid]['azi_coverage2'][()].shape)+'\n'
        outstr          += '*** path_density:                   size    = '+str(subgroup1[perid]['path_density'][()].shape)+ '; shape = '+\
                            str(subgroup2[perid]['path_density'][()].shape)+'\n'
        outstr          += '*** residual:                       size    = '+str(subgroup1[perid]['residual'][()].shape)+'\n'
        outstr          += '    id fi0 lam0 f1 lam1 vel_obs weight res_tomo[:, 7] res_mod delta '+ '\n'
        print (outstr)
        return
    
    def print_qc_info(self, runid=0):
        """print all the data stored in the qc run
        """
        outstr      = '------------------------------------ Quality controlled run data : id = '+str(runid)+' -------------------------------------\n'
        try:
            subgroup1   = self['qc_run_%d' %runid]
            subgroup2   = self['reshaped_qc_run_%d' %runid]
        except KeyError:
            print ('*** No data for qc run id = '+str(runid))
            return
        pers        = self.attrs['period_array']
        if subgroup1.attrs['isotropic']:
            tempstr = 'isotropic'
        else:
            tempstr = 'anisotropic'
        outstr      += '--- smooth run id                       - '+str(subgroup1.attrs['smoothid'])+'\n'
        outstr      += '--- isotropic/anisotropic               - '+tempstr+'\n'
        outstr      += '--- datatype(ph: phase; gr: group)      - '+str(subgroup1.attrs['datatype'])+'\n'
        outstr      += '--- wavetype(R: Rayleigh; L: Love)      - '+str(subgroup1.attrs['wavetype'])+'\n'
        outstr      += '--- Criteria factor/limit               - '+str(subgroup1.attrs['crifactor'])+'/'+str(subgroup1.attrs['crilimit'])+'\n'
        outstr      += '--- dlon, dlat                          - '+str(subgroup1.attrs['dlon'])+', '+str(subgroup1.attrs['dlat'])+'\n'
        try:
            outstr  += '!!! dlon_LD, dlat_LD                    - '+str(subgroup1.attrs['dlon_LD'])+', '+str(subgroup1.attrs['dlat_LD'])+'\n'
        except:
            try:
                outstr  += '!!! dlon_HD, dlat_HD                    - '+str(subgroup1.attrs['dlon_HD'])+', '+str(subgroup1.attrs['dlat_HD'])+'\n'
            except:
                try:
                    outstr  += '!!! dlon_interp, dlat_interp            - '+str(subgroup1.attrs['dlon_interp'])+', '+str(subgroup1.attrs['dlat_interp'])+'\n'
                except:
                    outstr  += '!!! No interpolation for for MC inversion \n'
        outstr      += '--- Step of integration                 - '+str(subgroup1.attrs['step_of_integration'])+'\n'
        outstr      += '--- Size of main cell (degree)          - '+str(subgroup1.attrs['lengthcell'])+'\n'
        if subgroup1.attrs['isotropic']:
            outstr      += '--- Smoothing coefficient (alpha)       - '+str(subgroup1.attrs['alpha'])+'\n'
            outstr      += '--- Path density damping (beta)         - '+str(subgroup1.attrs['beta'])+'\n'
            outstr      += '--- Gaussian damping (sigma)            - '+str(subgroup1.attrs['sigma'])+'\n'
        if not subgroup1.attrs['isotropic']:
            outstr      += '--- Size of anisotropic cell (degree)   - '+str(subgroup1.attrs['lengthcellAni'])+'\n'
            outstr      += '--- Anisotropic parameter               - '+str(subgroup1.attrs['anipara'])+'\n'
            outstr      += '    0: isotropic'+'\n'
            outstr      += '    1: 2 psi anisotropic'+'\n'
            outstr      += '    2: 2&4 psi anisotropic '+'\n'
            outstr      += '--- xZone                               - '+str(subgroup1.attrs['xZone'])+'\n'
            outstr      += '--- 0th smoothing coefficient(alphaAni0)- '+str(subgroup1.attrs['alphaAni0'])+'\n'
            outstr      += '--- 0th path density damping (betaAni0) - '+str(subgroup1.attrs['betaAni0'])+'\n'
            outstr      += '--- 0th Gaussian damping (sigmaAni0)    - '+str(subgroup1.attrs['sigmaAni0'])+'\n'
            outstr      += '--- 2rd smoothing coefficient(alphaAni2)- '+str(subgroup1.attrs['alphaAni2'])+'\n'
            outstr      += '--- 2rd Gaussian damping (sigmaAni2)    - '+str(subgroup1.attrs['sigmaAni2'])+'\n'
            outstr      += '--- 4th smoothing coefficient(alphaAni4)- '+str(subgroup1.attrs['alphaAni4'])+'\n'
            outstr      += '--- 4th Gaussian damping (sigmaAni4)    - '+str(subgroup1.attrs['sigmaAni4'])+'\n'
        outstr          += '--- Comments                            - '+str(subgroup1.attrs['comments'])+'\n'
        perid           = '%d_sec' %pers[0]
        if subgroup1.attrs['isotropic']:
            outstr          += '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n'
            outstr          += '*** Dvelocity:                      size    = '+str(subgroup1[perid]['Dvelocity'][()].shape)+ '; shape = '+\
                                str(subgroup2[perid]['Dvelocity'][()].shape)+'\n'
            outstr          += '*** velocity:                       size    = '+str(subgroup1[perid]['velocity'][()].shape)+ '; shape = '+\
                                str(subgroup2[perid]['velocity'][()].shape)+'\n'
            outstr          += '*** azi_coverage:                   size    = '+str(subgroup1[perid]['azi_coverage'][()].shape)+'\n'
            outstr          += '*** azi_coverage1 (squared sum):    shape   = '+str(subgroup2[perid]['azi_coverage1'][()].shape)+'\n'
            outstr          += '*** azi_coverage2 (max value):      shape   = '+str(subgroup2[perid]['azi_coverage2'][()].shape)+'\n'
            outstr          += '*** path_density:                   size    = '+str(subgroup1[perid]['path_density'][()].shape)+ '; shape = '+\
                                str(subgroup2[perid]['path_density'][()].shape)+'\n'
            outstr          += '*** residual:                       size    = '+str(subgroup1[perid]['residual'][()].shape)+'\n'
            outstr          += '    id fi0 lam0 f1 lam1 vel_obs weight res_tomo[:, 7] res_mod delta '+ '\n'
        else:
            # velocity
            outstr          += '=================================================================================================\n'
            outstr          += '*** Dvelocity:                      size    = '+str(subgroup1[perid]['Dvelocity'][()].shape)+ '\n'
            outstr          += '*** dv (reshaped):                  shape   = '+str(subgroup2[perid]['dv'][()].shape)+ '\n'
            outstr          += '=================================================================================================\n'
            outstr          += '*** velocity:                       size    = '+str(subgroup1[perid]['velocity'][()].shape)+ '\n'
            outstr          += '!!! 0. vel_iso (reshaped):          shape   = '+str(subgroup2[perid]['vel_iso'][()].shape)+ '\n'
            try:
                outstr      += '    3. amp2 (reshaped):             shape   = '+str(subgroup2[perid]['amp2'][()].shape)+ '\n'
                outstr      += '    4. psi2 (reshaped):             shape   = '+str(subgroup2[perid]['psi2'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- No psi2 inversion results \n'
            try:
                outstr      += '    7. amp4 (reshaped):             shape   = '+str(subgroup2[perid]['amp4'][()].shape)+ '\n'
                outstr      += '    8. psi4 (reshaped):             shape   = '+str(subgroup2[perid]['psi4'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- No psi4 inversion results \n'
            try:
                outstr      += '!!! vel_iso_interp (reshaped):      shape   = '+str(subgroup2[perid]['vel_iso_interp'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- NO vel_iso_interp \n'
            try:
                outstr      += '!!! vel_iso_LD (reshaped):          shape   = '+str(subgroup2[perid]['vel_iso_LD'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- NO vel_iso_LD \n'
            try:
                outstr      += '!!! vel_iso_HD (reshaped):          shape   = '+str(subgroup2[perid]['vel_iso_HD'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- NO vel_iso_HD \n'
            outstr          += '*** lons_lats(loc of velcotiy):     size    = '+str(subgroup1[perid]['lons_lats'][()].shape)+ '\n'
            outstr          += '$$$ mask1 (NOT in per sub-directory)size    = '+str(subgroup2['mask1'][()].shape)+'\n'
            outstr          += '=================================================================================================\n'
            # resolution
            outstr          += '!!! resolution:                     size    = '+str(subgroup1[perid]['resolution'][()].shape)+ '\n'
            outstr          += '    0. cone_radius (reshaped):      shape   = '+str(subgroup2[perid]['cone_radius'][()].shape)+ '\n'
            outstr          += '    1. gauss_std (reshaped):        shape   = '+str(subgroup2[perid]['gauss_std'][()].shape)+ '\n'
            outstr          += '    2. max_resp (max response value, reshaped):   \n'+\
                               '                                    shape   = '+str(subgroup2[perid]['max_resp'][()].shape)+ '\n'
            outstr          += '    3. ncone (number of cells involved in cone base, reshaped): \n'+\
                               '                                    shape   = '+str(subgroup2[perid]['ncone'][()].shape)+ '\n'
            outstr          += '    4. ngauss (number of cells involved in Gaussian construction, reshaped):  \n'+\
                               '                                    shape   = '+str(subgroup2[perid]['ngauss'][()].shape)+ '\n'
            try:
                outstr      += '!!! vel_sem (reshaped):             shape   = '+str(subgroup2[perid]['vel_sem'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- NO uncertainties estimated from eikonal tomography. No need for hybrid / group database! \n'
            try:
                outstr      += '!!! vel_sem_interp (reshaped):      shape   = '+str(subgroup2[perid]['vel_sem_interp'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- NO vel_sem_interp \n'
            try:
                outstr      += '!!! vel_sem_LD (reshaped):          shape   = '+str(subgroup2[perid]['vel_sem_LD'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- NO vel_sem_LD \n'
            try:
                outstr      += '!!! vel_sem_HD (reshaped):          shape   = '+str(subgroup2[perid]['vel_sem_HD'][()].shape)+ '\n'
            except KeyError:
                outstr      += '--- NO vel_sem_HD \n'
            outstr          += '*** lons_lats_rea(loc of reso):     size    = '+str(subgroup1[perid]['lons_lats_rea'][()].shape)+ '\n'
            outstr          += '$$$ mask2 (NOT in per sub-directory)size    = '+str(subgroup2['mask2'][()].shape)+'\n'
            try:
                outstr      += '!!! mask_inv(NOT in per sub-directory, determined by get_mask_inv. used for MC inversion): \n'+\
                               '                                    size    = '+str(subgroup2['mask_inv'][()].shape)+'\n'
            except:
                outstr      += '--- NO mask_inv array. No need for hybrid / group database! \n'
            try:
                outstr      += '!!! mask_LD(NOT in per sub-directory, determined by interp_surface. used for MC inversion): \n'+\
                               '                                    size    = '+str(subgroup2['mask_LD'][()].shape)+'\n'
            except:
                outstr      += '--- NO mask_LD array \n'
            try:
                outstr      += '!!! mask_HD(NOT in per sub-directory, determined by interp_surface. used for MC inversion): \n'+\
                               '                                    size    = '+str(subgroup2['mask_HD'][()].shape)+'\n'
            except:
                outstr      += '--- NO mask_HD array \n'
            try:
                outstr      += '!!! mask_interp(NOT in per sub-directory, determined by interp_surface. used for MC inversion): \n'+\
                               '                                    size    = '+str(subgroup2['mask_interp'][()].shape)+'\n'
            except:
                outstr      += '--- NO mask_interp array \n'
            outstr          += '=================================================================================================\n'
            outstr          += '!!! azi_coverage:                   size    = '+str(subgroup1[perid]['azi_coverage'][()].shape)+'\n'
            outstr          += '    azi_coverage1 (squared sum):    shape   = '+str(subgroup2[perid]['azi_coverage1'][()].shape)+'\n'
            outstr          += '    azi_coverage2 (max value):      shape   = '+str(subgroup2[perid]['azi_coverage2'][()].shape)+'\n'
            outstr          += '=================================================================================================\n'
            outstr          += '!!! path_density :                  size    = '+str(subgroup1[perid]['path_density'][()].shape)+ '\n'
            outstr          += '    0. path_density(all orbits):    shape   = '+str(subgroup2[perid]['path_density1'][()].shape)+'\n'
            outstr          += '    1. path_density1(first orbits): shape   = '+str(subgroup2[perid]['path_density1'][()].shape)+'\n'
            outstr          += '    2. path_density2(second orbits):shape   = '+str(subgroup2[perid]['path_density2'][()].shape)+'\n'
            outstr          += '=================================================================================================\n'
            outstr          += '!!! residual :                      size    = (:, '+str(subgroup1[perid]['residual'][()].shape[1])+')\n'
            outstr          += '    id fi0 lam0 f1 lam1 vel_obs weight orb res_tomo[:, 8] res_mod delta '+ '\n'
        print (outstr)
        return
    
    def print_info(self):
        """print general information of the dataset.
        """
        outstr          = self.print_attrs(print_to_screen=False)
        if outstr is None:
            return
        per_arr         = self.attrs['period_array']
        outstr          += '----------------------------------------- Smooth run data -----------------------------------------------\n'
        nid             = 0
        while True:
            key         =  'smooth_run_%d' %nid
            if not key in self.keys():
                break
            nid         += 1
            subgroup    = self[key]
            outstr      += '$$$$$$$$$$$$$$$$$$$$$$$$$$$ Run id: '+key+' $$$$$$$$$$$$$$$$$$$$$$$$$$$\n'
            # check data of different periods
            for per in per_arr:
                per_key = '%g_sec' %per
                if not per_key in subgroup.keys():
                    outstr  += '%g sec NOT in the database !\n' %per
            outstr          += 'Channel                             - '+str(subgroup.attrs['channel'])+'\n'
            outstr          += 'datatype(ph: phase; gr: group)      - '+str(subgroup.attrs['datatype'])+'\n'
            outstr          += 'dlon, dlat                          - '+str(subgroup.attrs['dlon'])+', '+str(subgroup.attrs['dlat'])+'\n'
            outstr          += 'Step of integration                 - '+str(subgroup.attrs['step_of_integration'])+'\n'
            outstr          += 'Smoothing coefficient (alpha1)      - '+str(subgroup.attrs['alpha1'])+'\n'
            outstr          += 'Path density damping (alpha2)       - '+str(subgroup.attrs['alpha2'])+'\n'
            outstr          += 'radius of correlation (sigma)       - '+str(subgroup.attrs['sigma'])+'\n'
            outstr          += 'Comments                            - '+str(subgroup.attrs['comments'])+'\n'
        outstr  += '------------------------------------ Quality controlled run data ----------------------------------------\n'
        nid     = 0
        while True:
            key =  'qc_run_%d' %nid
            if not key in self.keys():
                break
            nid +=1
            subgroup    =  self[key]
            outstr      += '$$$$$$$$$$$$$$$$$$$$$$$$$$$ Run id: '+key+' $$$$$$$$$$$$$$$$$$$$$$$$$$$\n'
            # check data of different periods
            for per in per_arr:
                per_key = '%g_sec' %per
                if not per_key in subgroup.keys():
                    outstr  += '%g sec NOT in the database !\n' %per
            if subgroup.attrs['isotropic']:
                tempstr = 'isotropic'
            else:
                tempstr = 'anisotropic'
            outstr      += 'Smooth run id                       - '+str(subgroup.attrs['smoothid'])+'\n'
            outstr      += 'isotropic/anisotropic               - '+tempstr+'\n'
            outstr      += 'datatype(ph: phase; gr: group)      - '+str(subgroup.attrs['datatype'])+'\n'
            outstr      += 'wavetype(R: Rayleigh; L: Love)      - '+str(subgroup.attrs['wavetype'])+'\n'
            outstr      += 'Criteria factor/limit               - '+str(subgroup.attrs['crifactor'])+'/'+str(subgroup.attrs['crilimit'])+'\n'
            outstr      += 'dlon, dlat                          - '+str(subgroup.attrs['dlon'])+', '+str(subgroup.attrs['dlat'])+'\n'
            outstr      += 'Step of integration                 - '+str(subgroup.attrs['step_of_integration'])+'\n'
            outstr      += 'Size of main cell (degree)          - '+str(subgroup.attrs['lengthcell'])+'\n'
            if subgroup.attrs['isotropic']:
                outstr      += 'Smoothing coefficient (alpha)       - '+str(subgroup.attrs['alpha'])+'\n'
                outstr      += 'Path density damping (beta)         - '+str(subgroup.attrs['beta'])+'\n'
                outstr      += 'Gaussian damping (sigma)            - '+str(subgroup.attrs['sigma'])+'\n'
            if not subgroup.attrs['isotropic']:
                outstr      += 'Size of anisotropic cell (degree)   - '+str(subgroup.attrs['lengthcellAni'])+'\n'
                outstr      += 'Anisotropic parameter               - '+str(subgroup.attrs['anipara'])+'\n'
                outstr      += '0: isotropic'+'\n'
                outstr      += '1: 2 psi anisotropic'+'\n'
                outstr      += '2: 2&4 psi anisotropic '+'\n'
                outstr      += 'xZone                               - '+str(subgroup.attrs['xZone'])+'\n'
                outstr      += '0th smoothing coefficient(alphaAni0)- '+str(subgroup.attrs['alphaAni0'])+'\n'
                outstr      += '0th path density damping (betaAni0) - '+str(subgroup.attrs['betaAni0'])+'\n'
                outstr      += '0th Gaussian damping (sigmaAni0)    - '+str(subgroup.attrs['sigmaAni0'])+'\n'
                outstr      += '2rd smoothing coefficient(alphaAni2)- '+str(subgroup.attrs['alphaAni2'])+'\n'
                outstr      += '2rd Gaussian damping (sigmaAni2)    - '+str(subgroup.attrs['sigmaAni2'])+'\n'
                outstr      += '4th smoothing coefficient(alphaAni4)- '+str(subgroup.attrs['alphaAni4'])+'\n'
                outstr      += '4th Gaussian damping (sigmaAni4)    - '+str(subgroup.attrs['sigmaAni4'])+'\n'
            outstr      += 'Comments                            - '+str(subgroup.attrs['comments'])+'\n'
        outstr += '=========================================================================================================\n'
        print (outstr)
        return
    
    def set_input_parameters(self, minlon, maxlon, minlat, maxlat, pers=[], \
            data_pfx='raytomo_in_', smooth_pfx='N_INIT_', qc_pfx='QC_'):
        """
        Set input parameters for tomographic inversion.
        =================================================================================================================
        ::: input parameters :::
        minlon, maxlon  - minimum/maximum longitude
        minlat, maxlat  - minimum/maximum latitude
        pers            - period array, default = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        data_pfx        - input data file prefix
        smoothpfx       - prefix for smooth run files
        smoothpfx       - prefix for qc(quanlity controlled) run files
        =================================================================================================================
        """
        if len(pers) == 0:
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        else:
            pers    = np.asarray(pers)
        self.attrs.create(name = 'period_array', data = pers, dtype = np.float64)
        self.attrs.create(name = 'minlon', data = minlon, dtype = np.float64)
        self.attrs.create(name = 'maxlon', data = maxlon, dtype = np.float64)
        self.attrs.create(name = 'minlat', data = minlat, dtype = np.float64)
        self.attrs.create(name = 'maxlat', data = maxlat, dtype = np.float64)
        self.attrs.create(name = 'data_prefix', data = data_pfx)
        self.attrs.create(name = 'smooth_prefix', data = smooth_pfx)
        self.attrs.create(name = 'qc_prefix', data = qc_pfx)
        self.update_attrs()
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
        ######################
        m.drawstates(linewidth=1.)
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        return m
    
    def plot(self, runtype, runid, datatype, period, clabel = '', cmap = 'surf', projection = 'lambert', \
             hillshade = False, vmin = None, vmax = None, thresh = 100., semfactor = 2., showfig = True):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runtype         - type of run (0 - smooth run, 1 - quality controlled run)
        runid           - id of run
        datatype        - datatype for plotting
        period          - period of data
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        thresh          - threhold value for Gaussian deviation to determine the mask for plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        # vdict       = {'ph': 'C', 'gr': 'U'}
        # datatype    = datatype.lower()
        rundict     = {0: 'smooth_run', 1: 'qc_run'}
        dataid      = rundict[runtype]+'_'+str(runid)
        self._get_lon_lat_arr(dataid)
        try:
            ingroup     = self['reshaped_'+dataid]
        except KeyError:
            try:
                self.creat_reshape_data(runtype=runtype, runid=runid)
                ingroup = self['reshaped_'+dataid]
            except KeyError:
                raise KeyError(dataid+ ' not exists!')
        if not period in self.pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        if runtype == 1:
            isotropic   = ingroup.attrs['isotropic']
        else:
            isotropic   = True
        factor              = 1.
        if datatype == 'vel' or datatype=='velocity' or datatype == 'v':
            if isotropic:
                datatype    = 'velocity'
            else:
                datatype    = 'vel_iso'
        if datatype == 'un' or datatype=='sem' or datatype == 'vel_sem':
            datatype        = 'vel_sem'
            factor          = 2.
        if datatype == 'resolution':
            datatype        = 'gauss_std'
            factor          = 2.
        try:
            data    = pergrp[datatype][()]*factor
        except:
            outstr      = ''
            for key in pergrp.keys():
                outstr  +=key
                outstr  +=', '
            outstr      = outstr[:-1]
            raise KeyError('Unexpected datatype: '+datatype+\
                           ', available datatypes are: '+outstr)
        if datatype == 'amp2':
            data    = data*100.
        if datatype == 'vel_sem':
            data        = data*1000.*semfactor
        
        if not isotropic:
            if datatype == 'cone_radius' or datatype == 'gauss_std' or datatype == 'max_resp' or datatype == 'ncone' or \
                         datatype == 'ngauss' or datatype == 'vel_sem':
                mask    = ingroup['mask2']
            else:
                mask    = ingroup['mask1']
            if thresh is not None:
                gauss_std   = pergrp['gauss_std'][()]
                mask_gstd   = gauss_std > thresh
                mask        = mask + mask_gstd
            mdata       = ma.masked_array(data, mask=mask )
        else:
            mdata       = data.copy()
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection = projection)
        x, y        = m(self.lonArr, self.latArr)
        # shapefname  = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        
        # plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        # shapefname  = '../AKfaults/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname  = '../AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        # shapefname  = '/projects/life9360/AK_sediments/Cook_Inlet_sediments_WGS84'
        # m.readshapefile(shapefname, 'faultline', linewidth=1, color='blue')
        try:
            import pycpt
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
                # cmap    = cmap.reversed()
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
        except:
            pass
        ################################3
        # if hillshade:
        #     from netCDF4 import Dataset
        #     from matplotlib.colors import LightSource
        # 
        #     etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
        #     etopo       = etopodata.variables['z'][:]
        #     lons        = etopodata.variables['x'][:]
        #     lats        = etopodata.variables['y'][:]
        #     ls          = LightSource(azdeg=315, altdeg=45)
        #     # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
        #     etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
        #     # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
        #     ny, nx      = etopo.shape
        #     topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
        #     m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
        #     mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
        #     mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
        #     mycm2.set_over('w',0)
        #     m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
        #     m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=.5)
        else:
            if datatype is 'path_density':
                import matplotlib.colors as colors
                im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', norm=colors.LogNorm(vmin=vmin, vmax=vmax),)
            else:
                im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.])
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')#, ticks=[20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
        cb.set_label(clabel, fontsize=20, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        
        ###
        # consb       = mask.copy()
        # consb       += self.latArr<68.
        # consb       += data>2.5
        # m.contour(x, y, consb, linestyles='dashed', colors='blue', lw=1.)
        ###
        
        cb.set_alpha(1)
        cb.draw_all()
        print ('plotting data from '+dataid)
        # # cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")
        if datatype is 'path_density':
            cb.set_ticks([1, 10, 100, 1000, 10000])
            cb.set_ticklabels([1, 10, 100, 1000, 10000])

        
        if showfig:
            plt.show()
        return
    
    def check_station_residual(self, instaxml, period, runid = 0, discard = False, usemad = True, madfactor = 3., crifactor = 0.5, crilimit = 10.,\
            plot = True, projection = 'merc', cmap = 'surf', vmin = None, vmax = None, clabel = 'average absolute'):
        stainv  = obspy.read_inventory(instaxml)
        lats    = []
        lons    = []
        staids  = []
        for network in stainv:
            for station in network:
                stlo    = float(station.longitude)
                if stlo < 0.:
                    stlo    += 360.
                if station.latitude <= self.maxlat and station.latitude >= self.minlat\
                    and stlo <= self.maxlon and stlo >= self.minlon:
                    lats.append(station.latitude)
                    lons.append(stlo)
                    staids.append(network.code+'.'+station.code)
        smoothgroup     = self['smooth_run_'+str(runid)]       
        try:
            residdset   = smoothgroup['%g_sec'%( period )+'/residual']
            # id fi0 lam0 f1 lam1 vel_obs weight res_tomo res_mod delta
            residual    = residdset[()]
        except:
            raise AttributeError('Residual data: '+ str(period)+ ' sec does not exist!')
        if discard:
            res_tomo        = residual[:,7]
            # quality control to discard data with large misfit
            if usemad:
                from statsmodels import robust
                mad         = robust.mad(res_tomo)
                cri_res     = madfactor * mad
            else:
                cri_res     = min(crifactor * per, crilimit)
            residual        = residual[np.abs(res_tomo)<cri_res, :]
            
        lats    = np.asarray(lats, dtype = np.float64)
        lons    = np.asarray(lons, dtype = np.float64)
        Ncounts, absres, res    = _tomo_funcs._station_residual(np.float64(lats), np.float64(lons), np.float64(residual))
        
        # plot
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection = projection)
        x, y        = m(lons, lats)
        try:
            import pycpt
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
                # cmap    = cmap.reversed()
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
        except:
            pass
        values      = res/Ncounts
        im          = m.scatter(x, y, marker='^', s = 50, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')#, ticks=[20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
        cb.set_label(clabel, fontsize=20, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        

        
        cb.set_alpha(1)
        cb.draw_all()

        # # cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")


        plt.show()
        
        return Ncounts, absres, res, staids
    
    
    

