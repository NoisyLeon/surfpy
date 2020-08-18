# -*- coding: utf-8 -*-
"""
HDF5 database for ray tomography, I/O part
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import h5py

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
        self.attrs.create(name = 'period_array', data = pers, dtype='f')
        self.attrs.create(name = 'minlon', data=minlon, dtype='f')
        self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
        self.attrs.create(name = 'minlat', data=minlat, dtype='f')
        self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
        self.attrs.create(name = 'data_prefix', data=data_pfx)
        self.attrs.create(name = 'smooth_prefix', data=smooth_pfx)
        self.attrs.create(name = 'qc_prefix', data=qc_pfx)
        self.update_attrs()
        return
    
    
    
    
    
    

