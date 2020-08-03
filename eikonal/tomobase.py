# -*- coding: utf-8 -*-
"""
hdf5 for eikonal tomography base
    
:copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
    
:references:
    Feng, L., & Ritzwoller, M. H. (2017). The effect of sedimentary basins on surface waves that pass through them.
        Geophysical Journal International, 211(1), 572-592.
    Feng, L., & Ritzwoller, M. H. (2019). A 3-D shear velocity model of the crust and uppermost mantle beneath Alaska including apparent radial anisotropy.
        Journal of Geophysical Research: Solid Earth, 124(10), 10468-10497.
    Lin, Fan-Chi, Michael H. Ritzwoller, and Roel Snieder. "Eikonal tomography: surface wave tomography by phase front tracking across a regional broad-band seismic array."
        Geophysical Journal International 177.3 (2009): 1091-1110.
    Lin, Fan-Chi, and Michael H. Ritzwoller. "Helmholtz surface wave tomography for isotropic and azimuthally anisotropic structure."
        Geophysical Journal International 186.3 (2011): 1104-1120.
"""

try:
    import surfpy.eikonal._eikonal_funcs as _eikonal_funcs
except:
    import _eikonal_funcs
    
import numpy as np
import h5py
import shutil
from subprocess import call
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt


class baseh5(h5py.File):
    """
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
    
    def _get_lon_lat_arr(self, ncut=0):
        """Get longitude/latitude array
        """
        self.lons   = np.arange((self.maxlon-self.minlon)/self.dlon+1-2*ncut)*self.dlon + self.minlon + ncut*self.dlon
        self.lats   = np.arange((self.maxlat-self.minlat)/self.dlat+1-2*ncut)*self.dlat + self.minlat + ncut*self.dlat
        self.Nlon   = self.lons.size
        self.Nlat   = self.lats.size
        self.lonArr, self.latArr = np.meshgrid(self.lons, self.lats)
        return
    
    def update_attrs(self):
        try:
            self.pers       = self.attrs['period_array']
            self.minlon     = self.attrs['minlon']
            self.maxlon     = self.attrs['maxlon']
            self.minlat     = self.attrs['minlat']
            self.maxlat     = self.attrs['maxlat']
            self.Nlon       = int(self.attrs['Nlon'])
            self.dlon       = self.attrs['dlon']
            self.Nlat       = int(self.attrs['Nlat'])
            self.dlat       = self.attrs['dlat']
            self.proj_name  = self.attrs['proj_name']
            return True
        except:
            return False
    
    # def update_dat(self):
    #     try:
    #         self.events     = self['input_field_data'].keys()
    #         # self.minlon     = self.attrs['minlon']
    #         # self.maxlon     = self.attrs['maxlon']
    #         # self.minlat     = self.attrs['minlat']
    #         # self.maxlat     = self.attrs['maxlat']
    #         # self.Nlon       = self.attrs['Nlon']
    #         # self.dlon       = self.attrs['dlon']
    #         # self.nlon_grad  = self.attrs['nlon_grad']
    #         # self.nlon_lplc  = self.attrs['nlon_lplc']
    #         # self.Nlat       = self.attrs['Nlat']
    #         # self.dlat       = self.attrs['dlat']
    #         # self.nlat_grad  = self.attrs['nlat_grad']
    #         # self.nlat_lplc  = self.attrs['nlat_lplc']
    #         # self.proj_name  = self.attrs['proj_name']
    #         return True
    #     except:
    #         return False
    
    def set_input_parameters(self, minlon, maxlon, minlat, maxlat, pers=[], dlon=0.2, dlat=0.2, optimize_spacing=True, proj_name = ''):
        """set input parameters for tomographic inversion.
        =================================================================================================================
        ::: input parameters :::
        minlon, maxlon  - minimum/maximum longitude
        minlat, maxlat  - minimum/maximum latitude
        pers            - period array, default = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        dlon, dlat      - longitude/latitude interval
        optimize_spacing- optimize the grid spacing or not
                            if True, the distance for input dlat/dlon will be calculated and dlat may be changed to
                                make the distance of dlat as close to the distance of dlon as possible
        =================================================================================================================
        """
        if len(pers) == 0:
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        else:
            pers    = np.asarray(pers)
        self.attrs.create(name = 'period_array', data = pers, dtype='f')
        self.attrs.create(name = 'minlon', data = minlon, dtype='f')
        self.attrs.create(name = 'maxlon', data = maxlon, dtype='f')
        self.attrs.create(name = 'minlat', data = minlat, dtype='f')
        self.attrs.create(name = 'maxlat', data =maxlat, dtype='f')
        if optimize_spacing:
            ratio   = _eikonal_funcs.determine_interval(minlat=minlat, maxlat=maxlat, dlon=dlon, dlat = dlat)
            if ratio != 1.:
                print ('----------------------------------------------------------')
                print ('Changed dlat from dlat =',dlat,'to dlat =',dlat/ratio)
                print ('----------------------------------------------------------')
                dlat    = dlat/ratio
        self.attrs.create(name = 'dlon', data = dlon)
        self.attrs.create(name = 'dlat', data = dlat)
        Nlon        = int((maxlon-minlon)/dlon+1)
        Nlat        = int((maxlat-minlat)/dlat+1)
        self.attrs.create(name = 'Nlon', data = Nlon)
        self.attrs.create(name = 'Nlat', data = Nlat)
        self.attrs.create(name = 'proj_name', data = proj_name)
        self.update_attrs()
        return
    #==================================================
    # functions print the information of database
    #==================================================
    def print_attrs(self, print_to_screen=True):
        """print the attrsbute information of the dataset.
        """
        if not self.update_attrs():
            print ('Empty Database!')
            return 
        outstr  = '================================= Surface wave eikonal/Helmholtz tomography database ==================================\n'
        outstr  += 'Project name:                           - '+str(self.proj_name)+'\n'        
        outstr  += 'period(s):                              - '+str(self.pers)+'\n'
        outstr  += 'longitude range                         - '+str(self.minlon)+' ~ '+str(self.maxlon)+'\n'
        outstr  += 'longitude spacing/npts                  - '+str(self.dlon)+'/'+str(self.Nlon)+'\n'
        outstr  += 'latitude range                          - '+str(self.minlat)+' ~ '+str(self.maxlat)+'\n'
        outstr  += 'latitude spacing/npts                   - '+str(self.dlat)+'/'+str(self.Nlat)+'\n'
        if print_to_screen:
            print (outstr)
        else:
            return outstr
        return
    
    def print_info(self, runid=0):
        """print the information of given eikonal/Helmholz run
        """
        outstr      = self.print_attrs(print_to_screen=False)
        if outstr is None:
            return
        outstr      += '========================================== Eikonal_run_%d' %runid +' ====================================================\n'
        subgroup    = self['Eikonal_run_%d' %runid]
        pers        = self.attrs['period_array']
        perid       = '%d_sec' %pers[0]
        Nevent      = len(subgroup[perid].keys())
        outstr      += '--- number of (virtual) events                  - '+str(Nevent)+'\n'
        evid        = subgroup[perid].keys()[0]
        evgrp       = subgroup[perid][evid]
        outstr      += '--- attributes for each event                   - evlo, evla, Nvalid_grd, Ntotal_grd \n'
        outstr      += '--- appV (apparent velocity)                    - '+str(evgrp['appV'].shape)+'\n'
        try:    
            outstr  += '--- corV (corrected velocity)                   - '+str(evgrp['corV'].shape)+'\n'
        except KeyError:
            outstr  += '*** NO corrected velocity \n'
        try:    
            outstr  += '--- lplc_amp (amplitude Laplacian)              - '+str(evgrp['lplc_amp'].shape)+'\n'
        except KeyError:
            outstr  += '*** NO corrected lplc_amp \n'
        outstr      += '--- az (azimuth)                                - '+str(evgrp['az'].shape)+'\n'
        outstr      += '--- baz (back-azimuth)                          - '+str(evgrp['baz'].shape)+'\n'
        outstr      += '--- proAngle (propagation angle)                - '+str(evgrp['proAngle'].shape)+'\n'
        outstr      += '--- travelT (travel time)                       - '+str(evgrp['travelT'].shape)+'\n'
        outstr      += '--- reason_n (index array)                      - '+str(evgrp['reason_n'].shape)+'\n'
        outstr      += '        0: accepted point \n' + \
                       '        1: data point the has large difference between v1HD and v1HD02 \n' + \
                       '        2: data point that does not have near neighbor points at all E/W/N/S directions\n' + \
                       '        3: slowness is too large/small \n' + \
                       '        4: near a zero field data point \n' + \
                       '        5: epicentral distance is too small \n' + \
                       '        6: large curvature              \n'
        try:
            outstr  += '--- reason_n_helm (index array for Helmoltz)    - '+str(evgrp['reason_n_helm'].shape)+'\n'
            outstr  += '        0 ~ 6: same as above \n' + \
                       '        7: reason_n of amplitude field is non-zero (invalid) \n' + \
                       '        8: negative phase slowness after correction \n'
        except KeyError:
            outstr  += '*** NO reason_n_helm \n'
        
        try:
            subgroup= self['Eikonal_stack_%d' %runid]
            outstr  += '=============================================================================================================\n'
        except KeyError:
            outstr  += '========================================== NO corresponding stacked results =================================\n'
            return
        if subgroup.attrs['anisotropic']:
            tempstr = 'anisotropic'
            outstr  += '--- isotropic/anisotropic                           - '+tempstr+'\n'
            outstr  += '--- N_bin (number of bins, for ani run)             - '+str(subgroup.attrs['N_bin'])+'\n'
            outstr  += '--- minazi/maxazi (min/max azi, for ani run)        - '+str(subgroup.attrs['minazi'])+'/'+str(subgroup.attrs['maxazi'])+'\n'
        else:
            tempstr = 'isotropic'
        pergrp      = subgroup[perid]
        outstr      += '--- Nmeasure (number of raw measurements)           - '+str(pergrp['Nmeasure'].shape)+'\n'
        outstr      += '--- NmeasureQC (number of qc measurements)          - '+str(pergrp['NmeasureQC'].shape)+'\n'
        outstr      += '--- slowness                                        - '+str(pergrp['slowness'].shape)+'\n'
        outstr      += '--- slowness_std                                    - '+str(pergrp['slowness_std'].shape)+'\n'
        outstr      += '--- mask                                            - '+str(pergrp['mask'].shape)+'\n'
        outstr      += '--- vel_iso (isotropic velocity)                    - '+str(pergrp['vel_iso'].shape)+'\n'
        outstr      += '--- vel_sem (uncertainties for velocity)            - '+str(pergrp['vel_sem'].shape)+'\n'
        if subgroup.attrs['anisotropic']:
            outstr  += '--- NmeasureAni (number of aniso measurements)      - '+str(pergrp['NmeasureAni'].shape)+'\n'
            outstr  += '--- histArr (number of binned measurements)         - '+str(pergrp['histArr'].shape)+'\n'
            outstr  += '--- slownessAni (aniso perturbation in slowness)    - '+str(pergrp['slownessAni'].shape)+'\n'
            outstr  += '--- slownessAni_sem (uncertainties in slownessAni)  - '+str(pergrp['slownessAni_sem'].shape)+'\n'
            outstr  += '--- velAni_sem (uncertainties in binned velocity)   - '+str(pergrp['velAni_sem'].shape)+'\n'
        print (outstr)
        return
    
    def compare_dset(self, in_h5fname, runid = 0):
        """compare two datasets, for debug purpose
        """
        indset  = baseh5(in_h5fname)
        datagrp = self['Eikonal_run_'+str(runid)]
        indatgrp= indset['Eikonal_run_'+str(runid)]
        for per in self.pers:
            dat_per_grp     = datagrp['%g_sec' %per] 
            event_lst       = dat_per_grp.keys()
            indat_per_grp   = indatgrp['%g_sec' %per] 
            for evid in event_lst:
                try:
                    dat_ev_grp  = dat_per_grp[evid]
                    indat_ev_grp= indat_per_grp[evid]
                except:
                    print ('No data:'+evid)
                app_vel1    = dat_ev_grp['apparent_velocity'][()]
                app_vel2    = indat_ev_grp['apparent_velocity'][()]
                diff_vel    = app_vel1 - app_vel2
                if np.allclose(app_vel1, app_vel2):
                    print ('--- Apparent velocity ALL equal: '+evid)
                else:
                    print ('--- Apparent velocity NOT equal: '+evid)
                    print ('--- min/max difference: %g/%g' %(diff_vel.min(), diff_vel.max()))
        return
        





