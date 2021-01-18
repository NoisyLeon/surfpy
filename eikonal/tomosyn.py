
import pykonal
import obspy
import numpy as np

import h5py
# import pyproj
from pyproj import Geod
from scipy.ndimage import gaussian_filter

geodist             = Geod(ellps='WGS84')
class synh5(h5py.File):
    """
    """
    def __init__(self, name, mode='a', driver=None, libver=None, userblock_size=None, swmr=False,\
            rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None, **kwds):
        super(synh5, self).__init__( name, mode, driver, libver, userblock_size,\
            swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order)
        #======================================
        # initializations of attributes
        #======================================
        # if self.update_attrs():
        #     self._get_lon_lat_arr()
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
        if self.lons[0] != self.minlon or self.lons[-1] != self.maxlon \
            or self.lats[0] != self.minlat or self.lats[-1] != self.maxlat:
            raise ValueError('!!! longitude/latitude arrays not consistent with bounds')
        try:
            self.ilontype   = self.attrs['ilontype']
        except:
            self.ilontype   = 1
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
        ###
        try:
            self.ilontype   = self.attrs['ilontype']
        except:
            self.ilontype   = 1
    
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
        if minlon < 0.:
            self.attrs.create(name = 'ilontype', data = 0, dtype = np.int32)
            print ('Longitude type will be -180/180 !')
        else:
            self.attrs.create(name = 'ilontype', data = 1, dtype = np.int32)
        self.attrs.create(name = 'period_array', data = pers, dtype = np.float64)
        self.attrs.create(name = 'minlon', data = minlon, dtype = np.float64)
        self.attrs.create(name = 'maxlon', data = maxlon, dtype = np.float64)
        self.attrs.create(name = 'minlat', data = minlat, dtype = np.float64)
        self.attrs.create(name = 'maxlat', data = maxlat, dtype = np.float64)
        
        
        self.attrs.create(name = 'dlon', data = dlon, dtype = np.float64)
        self.attrs.create(name = 'dlat', data = dlat, dtype = np.float64)
        Nlon        = int((maxlon-minlon)/dlon+1)
        Nlat        = int((maxlat-minlat)/dlat+1)
        self.attrs.create(name = 'Nlon', data = Nlon, dtype = np.int64)
        self.attrs.create(name = 'Nlat', data = Nlat, dtype = np.int64)
        self.attrs.create(name = 'proj_name', data = proj_name)
        self.update_attrs()
        self._get_lon_lat_arr()
        return
    
    def get_input_vel(self, outfname, wlon = 1., wlat = 1., v0 = 3.25, dv = 0.25, sigma = 5., stadist=50.):
        outdset     = h5py.File(outfname, mode='a')
        #=======================
        # synthetic input model
        #=======================
        lon_npts    = int(180/(self.dlon))
        lat_npts    = int(180/(self.dlat))
        
        solver = pykonal.EikonalSolver(coord_sys="spherical")
        solver.velocity.min_coords = 6371., 0, 0
        
        solver.velocity.node_intervals = 1, np.pi/lat_npts, np.pi/lon_npts
        solver.velocity.npts = 1, (lat_npts+1), 2*lon_npts
        
        lonlats = solver.velocity.nodes/np.pi*180.
        lonlats[:, :, :, 1] = 90. - lonlats[:, :, :, 1]
        vlats   = lonlats[0, :, 1, 1]
        vlons   = lonlats[0, 1, :, 2]
        v0_arr  = np.ones((1, lat_npts+1, 2*lon_npts))*v0
        vel_arr = v0_arr.copy()
        for ilon in range(2*lon_npts):
            for ilat in range(lat_npts+1):
                lat = lonlats[0, ilat, ilon, 1]
                lon = lonlats[0, ilat, ilon, 2]
                if lat <=self.maxlat and lat >= self.minlat \
                    and lon <=self.maxlon and lon >= self.minlon:
                        tmplat  = (int(np.floor((lat - self.minlat)/wlat))%2)
                        tmplon  = (int(np.floor((lon - self.minlon)/wlon))%2)
                        clat    = np.floor((lat - self.minlat)/wlat) * wlat + self.minlat + wlat/2.
                        clon    = np.floor((lon - self.minlon)/wlon) * wlon + self.minlon + wlon/2.
                        ##########
                        if tmplon == tmplat:
                            vel_arr[0, ilat, ilon] -= dv
                        else:
                            vel_arr[0, ilat, ilon] += dv
        vel_arr[0, :, :] = gaussian_filter(vel_arr[0, :, :], sigma=sigma)
        # save input model
        vel_grp     = outdset.require_group( name = 'tomo_stack_0' )
        vel_per_grp = vel_grp.create_group( name='10_sec')
        ind_vel_lat = np.where((vlats >= self.minlat-0.001)*(vlats<=self.maxlat+0.001))[0]
        ind_vel_lon = np.where((vlons >= self.minlon-0.001)*(vlons<=self.maxlon+0.001))[0]
        # # # # print (vlons[ind_vel_lon][0], vlons[ind_vel_lon][-1])
        input_vel   = ((vel_arr[0, ind_vel_lat, :])[:, ind_vel_lon])
        ###
        ind1        = input_vel > v0
        ind2        = input_vel < v0
        input_vel[ind1] = v0 - (input_vel[ind1] - v0)
        input_vel[ind2] = v0 + (v0 - input_vel[ind2])
        ###
        vel_per_grp.create_dataset(name = 'input_velocity', data = input_vel)
        
        
    def checker_board_data(self, outfname, wlon = 1., wlat = 1., v0 = 3.25, dv = 0.25, cdist = 50.):
        outdset     = h5py.File(outfname, mode='a')
        #=======================
        # synthetic input model
        #=======================
        solver = pykonal.EikonalSolver(coord_sys="spherical")
        solver.velocity.min_coords = 6371., 0, 0
        
        solver.velocity.node_intervals = 1, np.pi/1800, np.pi/1800
        solver.velocity.npts = 1, 1801, 3600
        
        lonlats = solver.velocity.nodes/np.pi*180.
        lonlats[:, :, :, 1] = 90. - lonlats[:, :, :, 1]
        vlats   = lonlats[0, :, 1, 1]
        vlons   = lonlats[0, 1, :, 2]
        v0_arr  = np.ones((1, 1801, 3600))*v0
        vel_arr = v0_arr.copy()
        for ilon in range(3600):
            for ilat in range(1801):
                lat = lonlats[0, ilat, ilon, 1]
                lon = lonlats[0, ilat, ilon, 2]
                if lat <=self.maxlat and lat >= self.minlat \
                    and lon <=self.maxlon and lon >= self.minlon:
                        tmplat  = (int(np.floor((lat - self.minlat)/wlat))%2)
                        tmplon  = (int(np.floor((lon - self.minlon)/wlon))%2)
                        clat    = np.floor((lat - self.minlat)/wlat) * wlat + self.minlat + wlat/2.
                        clon    = np.floor((lon - self.minlon)/wlon) * wlon + self.minlon + wlon/2.
                        dist, az, baz           = obspy.geodetics.gps2dist_azimuth(lat, lon, clat, clon)
                        dist /= 1000.
                        if dist >= cdist:
                            continue
                        if tmplon == tmplat:
                            vel_arr[0, ilat, ilon] -= dv* (1. + np.cos(np.pi/cdist*dist))/2.
                        else:
                            vel_arr[0, ilat, ilon] += dv* (1. + np.cos(np.pi/cdist*dist))/2.
        # save input model
        vel_grp     = outdset.require_group( name = 'tomo_stack_0' )
        vel_per_grp = vel_grp.create_group( name='10_sec')
        ind_vel_lat = np.where((vlats >= self.minlat-0.001)*(vlats<=self.maxlat+0.001))[0]
        ind_vel_lon = np.where((vlons >= self.minlon-0.001)*(vlons<=self.maxlon+0.001))[0]
        # # # # print (vlons[ind_vel_lon][0], vlons[ind_vel_lon][-1])
        input_vel   = ((vel_arr[0, ind_vel_lat, :])[:, ind_vel_lon])
        ###
        ind1        = input_vel > v0
        ind2        = input_vel < v0
        input_vel[ind1] = v0 - (input_vel[ind1] - v0)
        input_vel[ind2] = v0 + (v0 - input_vel[ind2])
        ###
        vel_per_grp.create_dataset(name = 'input_velocity', data = input_vel)
        self.update_attrs()
        # create group for input data
        group       = outdset.require_group( name = 'input_field_data')
        ingrp       = self['input_field_data']
        group.attrs.create(name = 'channel', data = ingrp.attrs['channel'])
        #---------------------------------
        # get stations (virtual events)
        #---------------------------------
        # loop over periods
        for per in self.pers:
            print ('--- generating data for: '+str(per)+' sec')
            del_per         = per - int(per)
            if del_per==0.:
                per_name    = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                per_name    = str(int(per))+'sec'+dper.split('.')[1]
            in_per_grp      = ingrp['%g_sec' %per]
            per_group       = group.require_group(name = '%g_sec' %per)
            # loop over events
            for evid in in_per_grp.keys():
                in_evgrp    = in_per_grp[evid]
                in_evla     = in_evgrp.attrs['evla']
                in_evlo     = in_evgrp.attrs['evlo']
                in_lats     = in_evgrp['lats'][()]
                in_lons     = in_evgrp['lons'][()]
                snr         = in_evgrp['snr'][()]
                #
                if in_evlo < 0.:
                    in_evlo += 360.
                evlo    = np.round(in_evlo/self.dlon)*self.dlon
                evla    = np.round(in_evla/self.dlat)*self.dlat
                print ('Event '+ evid, in_evla, evla, in_evlo, evlo)
                ###
                # eikonal solver
                ###
                solver = pykonal.EikonalSolver(coord_sys="spherical")
                solver.velocity.min_coords = 6371., 0, 0
                
                solver.velocity.node_intervals = 1, np.pi/1800, np.pi/1800
                solver.velocity.npts = 1, 1801, 3600
                
                lonlats = solver.velocity.nodes/np.pi*180.
                lonlats[:, :, :, 1] = 90. - lonlats[:, :, :, 1]
                solver.velocity.values = vel_arr
                # solve
                ind_evlo    = int(np.round(in_evlo/self.dlon))
                ind_evla    = int(np.round(in_evla/self.dlat))
                src_idx     = (0, ind_evla, ind_evlo)
                solver.traveltime.values[src_idx] = 0
                solver.unknown[src_idx] = False
                solver.trial.push(*src_idx)
                solver.solve()
                ###
                Nsize       = in_lats.size
                travel_t    = []
                lons        = []
                lats        = []
                for i in range(Nsize):
                    inlat   = in_lats[i]
                    inlon   = in_lons[i]
                    outlats = lonlats[0, :, 1, 1]
                    outlons = lonlats[0, 1, :, 2]
                    ind_lat = (abs(inlat - outlats)).argmin()
                    ind_lon = (abs(inlon - outlons)).argmin()
                    outlat  = outlats[ind_lat]
                    outlon  = outlons[ind_lon]
                    travel_t.append(solver.traveltime.values[0, ind_lat, ind_lon])
                    lons.append(outlon)
                    lats.append(outlat)
                lons            = np.array(lons)
                lats            = np.array(lats)
                travel_t        = np.array(travel_t)
                az, baz, dist   = geodist.inv(np.ones(Nsize)*evlo, np.ones(Nsize)*evla, lons, lats)
                distance        = dist/1000.
                phase_velocity  = distance/travel_t
                # save data to hdf5 dataset
                event_group = per_group.create_group(name = evid)
                event_group.attrs.create(name = 'evlo', data = evlo)
                event_group.attrs.create(name = 'evla', data = evla)
                event_group.attrs.create(name = 'num_data_points', data = in_evgrp.attrs['num_data_points'])
                event_group.create_dataset(name='lons', data = lons)
                event_group.create_dataset(name='lats', data = lats)
                event_group.create_dataset(name='phase_velocity', data = phase_velocity)
                event_group.create_dataset(name='snr', data = snr)
                event_group.create_dataset(name='distance', data = distance)
                try:
                    index_borrow = in_evgrp['index_borrow'][()]
                    event_group.create_dataset(name='index_borrow', data = index_borrow)
                except:
                    pass
        return
    
    def checker_board_data_2(self, outfname, wlon = 1., wlat = 1., v0 = 3.25, dv = 0.25, cdist = 50.):
        outdset     = h5py.File(outfname, mode='a')
        #=======================
        # synthetic input model
        #=======================
        solver = pykonal.EikonalSolver(coord_sys="spherical")
        solver.velocity.min_coords = 6371., 0, 0
        
        solver.velocity.node_intervals = 1, np.pi/1800, np.pi/1800
        solver.velocity.npts = 1, 1801, 3600
        
        lonlats = solver.velocity.nodes/np.pi*180.
        lonlats[:, :, :, 1] = 90. - lonlats[:, :, :, 1]
        vlats   = lonlats[0, :, 1, 1]
        vlons   = lonlats[0, 1, :, 2]
        v0_arr  = np.ones((1, 1801, 3600))*v0
        vel_arr = v0_arr.copy()
        for ilon in range(3600):
            for ilat in range(1801):
                lat = lonlats[0, ilat, ilon, 1]
                lon = lonlats[0, ilat, ilon, 2]
                if lat <=self.maxlat and lat >= self.minlat \
                    and lon <=self.maxlon and lon >= self.minlon:
                        tmplat  = (int(np.floor((lat - self.minlat)/wlat))%2)
                        tmplon  = (int(np.floor((lon - self.minlon)/wlon))%2)
                        clat    = np.floor((lat - self.minlat)/wlat) * wlat + self.minlat + wlat/2.
                        clon    = np.floor((lon - self.minlon)/wlon) * wlon + self.minlon + wlon/2.
                        dist, az, baz           = obspy.geodetics.gps2dist_azimuth(lat, lon, clat, clon)
                        dist /= 1000.
                        if dist >= cdist:
                            continue
                        if tmplon == tmplat:
                            vel_arr[0, ilat, ilon] -= dv* (1. + np.cos(np.pi/cdist*dist))/2.
                        else:
                            vel_arr[0, ilat, ilon] += dv* (1. + np.cos(np.pi/cdist*dist))/2.
        # save input model
        vel_grp     = outdset.require_group( name = 'tomo_stack_0' )
        vel_per_grp = vel_grp.create_group( name='10_sec')
        ind_vel_lat = np.where((vlats >= self.minlat-0.001)*(vlats<=self.maxlat+0.001))[0]
        ind_vel_lon = np.where((vlons >= self.minlon-0.001)*(vlons<=self.maxlon+0.001))[0]
        # # # # print (vlons[ind_vel_lon][0], vlons[ind_vel_lon][-1])
        input_vel   = ((vel_arr[0, ind_vel_lat, :])[:, ind_vel_lon])
        vel_per_grp.create_dataset(name = 'input_velocity', data = input_vel)
        self.update_attrs()
        # create group for input data
        group       = outdset.require_group( name = 'input_field_data')
        ingrp       = self['input_field_data']
        group.attrs.create(name = 'channel', data = ingrp.attrs['channel'])
        #---------------------------------
        # get stations (virtual events)
        #---------------------------------
        # loop over periods
        for per in self.pers:
            print ('--- generating data for: '+str(per)+' sec')
            del_per         = per - int(per)
            if del_per==0.:
                per_name    = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                per_name    = str(int(per))+'sec'+dper.split('.')[1]
            in_per_grp      = ingrp['%g_sec' %per]
            per_group       = group.require_group(name = '%g_sec' %per)
            # loop over events
            for evid in in_per_grp.keys():
                in_evgrp    = in_per_grp[evid]
                in_evla     = in_evgrp.attrs['evla']
                in_evlo     = in_evgrp.attrs['evlo']
                #
                if in_evlo < 0.:
                    in_evlo += 360.
                evlo    = np.round(in_evlo/self.dlon)*self.dlon
                evla    = np.round(in_evla/self.dlat)*self.dlat
                print ('Event '+ evid, in_evla, evla, in_evlo, evlo)
                ###
                # eikonal solver
                ###
                solver = pykonal.EikonalSolver(coord_sys="spherical")
                solver.velocity.min_coords = 6371., 0, 0
                
                solver.velocity.node_intervals = 1, np.pi/1800, np.pi/1800
                solver.velocity.npts = 1, 1801, 3600
                
                lonlats = solver.velocity.nodes/np.pi*180.
                lonlats[:, :, :, 1] = 90. - lonlats[:, :, :, 1]
                solver.velocity.values = vel_arr
                # solve
                ind_evlo    = int(np.round(in_evlo/self.dlon))
                ind_evla    = int(np.round(in_evla/self.dlat))
                src_idx     = (0, ind_evla, ind_evlo)
                solver.traveltime.values[src_idx] = 0
                solver.unknown[src_idx] = False
                solver.trial.push(*src_idx)
                solver.solve()
                ###
                tmplats     = lonlats[0, :, 1, 1]
                tmplons     = lonlats[0, 1, :, 2]
                ind_out_lat = np.where((tmplats >= self.minlat)*(tmplats<=self.maxlat))[0]
                ind_out_lon = np.where((tmplons >= self.minlon)*(tmplons<=self.maxlon))[0]
                Nsize       = ind_out_lon.size*ind_out_lat.size
                lats        = ((lonlats[0, ind_out_lat, :, 1])[:, ind_out_lon]).reshape(Nsize)
                lons        = ((lonlats[0, ind_out_lat, :, 2])[:, ind_out_lon]).reshape(Nsize)
                travel_t    = ((solver.traveltime.values[0, ind_out_lat, :])[:, ind_out_lon]).reshape(Nsize)
                #
                index_valid = travel_t != 0.
                travel_t    = travel_t[index_valid]
                lons        = lons[index_valid]
                lats        = lats[index_valid]
                Nsize       = lons.size
                az, baz, dist   = geodist.inv(np.ones(Nsize)*evlo, np.ones(Nsize)*evla, lons, lats)
                distance        = dist/1000.
                phase_velocity  = distance/travel_t
                # save data to hdf5 dataset
                event_group = per_group.create_group(name = evid)
                event_group.attrs.create(name = 'evlo', data = evlo)
                event_group.attrs.create(name = 'evla', data = evla)
                event_group.attrs.create(name = 'num_data_points', data = Nsize)
                event_group.create_dataset(name='lons', data = lons)
                event_group.create_dataset(name='lats', data = lats)
                event_group.create_dataset(name='phase_velocity', data = phase_velocity)
                event_group.create_dataset(name='snr', data = np.ones(Nsize) * 20.)
                event_group.create_dataset(name='distance', data = distance)
                try:
                    index_borrow = in_evgrp['index_borrow'][()]
                    event_group.create_dataset(name='index_borrow', data = index_borrow)
                except:
                    pass
        return
    
    def checker_board_raw_data(self, outfname, wlon = 1., wlat = 1., v0 = 3.25, dv = 0.25, sigma = 5.):
        outdset     = h5py.File(outfname, mode='a')
        #=======================
        # synthetic input model
        #=======================
        lon_npts    = int(180/(self.dlon))
        lat_npts    = int(180/(self.dlat))
        
        solver = pykonal.EikonalSolver(coord_sys="spherical")
        solver.velocity.min_coords = 6371., 0, 0
        
        solver.velocity.node_intervals = 1, np.pi/lat_npts, np.pi/lon_npts
        solver.velocity.npts = 1, (lat_npts+1), 2*lon_npts
        
        lonlats = solver.velocity.nodes/np.pi*180.
        lonlats[:, :, :, 1] = 90. - lonlats[:, :, :, 1]
        vlats   = lonlats[0, :, 1, 1]
        vlons   = lonlats[0, 1, :, 2]
        v0_arr  = np.ones((1, lat_npts+1, 2*lon_npts))*v0
        vel_arr = v0_arr.copy()
        for ilon in range(2*lon_npts):
            for ilat in range(lat_npts+1):
                lat = lonlats[0, ilat, ilon, 1]
                lon = lonlats[0, ilat, ilon, 2]
                if lat <=self.maxlat and lat >= self.minlat \
                    and lon <=self.maxlon and lon >= self.minlon:
                        tmplat  = (int(np.floor((lat - self.minlat)/wlat))%2)
                        tmplon  = (int(np.floor((lon - self.minlon)/wlon))%2)
                        clat    = np.floor((lat - self.minlat)/wlat) * wlat + self.minlat + wlat/2.
                        clon    = np.floor((lon - self.minlon)/wlon) * wlon + self.minlon + wlon/2.
                        if tmplon == tmplat:
                            vel_arr[0, ilat, ilon] -= dv
                        else:
                            vel_arr[0, ilat, ilon] += dv
        vel_arr[0, :, :] = gaussian_filter(vel_arr[0, :, :], sigma=sigma)
        # save input model
        vel_grp     = outdset.require_group( name = 'tomo_stack_0' )
        vel_per_grp = vel_grp.create_group( name='10_sec')
        ind_vel_lat = np.where((vlats >= self.minlat-0.001)*(vlats<=self.maxlat+0.001))[0]
        ind_vel_lon = np.where((vlons >= self.minlon-0.001)*(vlons<=self.maxlon+0.001))[0]
        input_vel   = ((vel_arr[0, ind_vel_lat, :])[:, ind_vel_lon])
        ###
        # ind1        = input_vel > v0
        # ind2        = input_vel < v0
        # input_vel[ind1] = v0 - (input_vel[ind1] - v0)
        # input_vel[ind2] = v0 + (v0 - input_vel[ind2])
        ###
        # # # vel_per_grp.create_dataset(name = 'input_velocity', data = input_vel)
        self.update_attrs()
        # create group for input data
        group       = outdset.require_group( name = 'input_field_data')
        ingrp       = self['input_field_data']
        group.attrs.create(name = 'channel', data = ingrp.attrs['channel'])
        #---------------------------------
        # get stations (virtual events)
        #---------------------------------
        # loop over periods
        for per in self.pers:
            print ('--- generating data for: '+str(per)+' sec')
            del_per         = per - int(per)
            if del_per==0.:
                per_name    = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                per_name    = str(int(per))+'sec'+dper.split('.')[1]
            in_per_grp      = ingrp['%g_sec' %per]
            per_group       = group.require_group(name = '%g_sec' %per)
            # loop over events
            for evid in in_per_grp.keys():
                in_evgrp    = in_per_grp[evid]
                in_evla     = in_evgrp.attrs['evla']
                in_evlo     = in_evgrp.attrs['evlo']
                if in_evlo < 0.:
                    in_evlo += 360.
                evlo    = np.round(in_evlo/self.dlon)*self.dlon
                evla    = np.round(in_evla/self.dlat)*self.dlat
                print ('Event '+ evid, in_evla, evla, in_evlo, evlo)
                ###
                # eikonal solver
                ###
                solver = pykonal.EikonalSolver(coord_sys="spherical")
                solver.velocity.min_coords = 6371., 0, 0
                solver.velocity.node_intervals = 1, np.pi/lat_npts, np.pi/lon_npts
                solver.velocity.npts = 1, (lat_npts+1), 2*lon_npts
                lonlats = solver.velocity.nodes/np.pi*180.
                lonlats[:, :, :, 1] = 90. - lonlats[:, :, :, 1]
                solver.velocity.values = vel_arr
                # solve
                ind_evlo    = int(np.round(in_evlo/self.dlon))
                ind_evla    = int(np.round(in_evla/self.dlat))
                src_idx     = (0, ind_evla, ind_evlo)
                solver.traveltime.values[src_idx] = 0
                solver.unknown[src_idx] = False
                solver.trial.push(*src_idx)
                solver.solve()
                ###
                tmplats     = lonlats[0, :, 1, 1]
                tmplons     = lonlats[0, 1, :, 2]
                ind_out_lat = np.where((tmplats >= self.minlat)*(tmplats<=self.maxlat))[0]
                ind_out_lon = np.where((tmplons >= self.minlon)*(tmplons<=self.maxlon))[0]
                Nsize       = ind_out_lon.size*ind_out_lat.size
                lats        = ((lonlats[0, ind_out_lat, :, 1])[:, ind_out_lon]).reshape(Nsize)
                lons        = ((lonlats[0, ind_out_lat, :, 2])[:, ind_out_lon]).reshape(Nsize)
                travel_t    = ((solver.traveltime.values[0, ind_out_lat, :])[:, ind_out_lon]).reshape(Nsize)
                #
                index_valid = travel_t != 0.
                travel_t    = travel_t[index_valid]
                lons        = lons[index_valid]
                lats        = lats[index_valid]
                Nsize       = lons.size
                # save data to hdf5 dataset
                event_group = per_group.create_group(name = evid)
                event_group.attrs.create(name = 'evlo', data = evlo)
                event_group.attrs.create(name = 'evla', data = evla)
                event_group.attrs.create(name = 'raw_num_data_points', data = Nsize)
                event_group.create_dataset(name='raw_lons', data = lons)
                event_group.create_dataset(name='raw_lats', data = lats)
                event_group.create_dataset(name='raw_travel_time', data = travel_t)
                event_group.create_dataset(name='raw_snr', data = np.ones(Nsize) * 20.)
                try:
                    index_borrow = in_evgrp['index_borrow'][()]
                    event_group.create_dataset(name='index_borrow', data = index_borrow)
                except:
                    pass
        return
    
    def get_syn_dat(self, outfname, dlat=None, dlon=None, stadist=50.):
        outdset     = h5py.File(outfname, mode='a')
        #=======================
        # synthetic input model
        #=======================
        self.update_attrs()
        # create group for input data
        group       = outdset.require_group( name = 'input_field_data')
        ingrp       = self['input_field_data']
        group.attrs.create(name = 'channel', data = ingrp.attrs['channel'])
        #---------------------------------
        # get stations (virtual events)
        #---------------------------------
        # loop over periods
        for per in self.pers:
            print ('--- generating data for: '+str(per)+' sec')
            del_per         = per - int(per)
            if del_per==0.:
                per_name    = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                per_name    = str(int(per))+'sec'+dper.split('.')[1]
            in_per_grp      = ingrp['%g_sec' %per]
            per_group       = group.require_group(name = '%g_sec' %per)
            # loop over events
            for evid in in_per_grp.keys():
                in_evgrp    = in_per_grp[evid]
                in_lats     = in_evgrp['lats'][()]
                in_lons     = in_evgrp['lons'][()]
                event_group = per_group.require_group(name = evid)
                evlo        = event_group.attrs['evlo']
                evla        = event_group.attrs['evla']
                tlats       = event_group['raw_lats'][()]
                tlons       = event_group['raw_lons'][()]
                ttravel_t   = event_group['raw_travel_time'][()]
                lons        = []
                lats        = []
                travel_t    = []
                Nin         = in_lats.size
                for i in range(ttravel_t.size):
                    tmplon  = tlons[i]
                    tmplat  = tlats[i]
                    dellon  = tmplon - self.minlon
                    dellat  = tmplat - self.minlat
                    if dlon is not None and dlat is not None:
                        if (dellon - np.floor(dellon/dlon)*dlon ) > dlon/2. or \
                            (dellat - np.floor(dellat/dlat)*dlat )  > dlat/2.:
                                continue
                    az, baz, dist   = geodist.inv(np.ones(Nin)*tmplon, np.ones(Nin)*tmplat, in_lons, in_lats)
                    dist    /= 1000.
                    if dist.min() < stadist:
                        lons.append(tmplon)
                        lats.append(tmplat)
                        travel_t.append(ttravel_t[i])
                
                lons            = np.array(lons)
                lats            = np.array(lats)
                travel_t        = np.array(travel_t)
                Nsize           = lons.size
                az, baz, dist   = geodist.inv(np.ones(Nsize)*evlo, np.ones(Nsize)*evla, lons, lats)
                distance        = dist/1000.
                phase_velocity  = distance/travel_t
                # save data to hdf5 dataset
                print ('Event '+ evid + ', grid: '+str(Nsize)+'/'+str(ttravel_t.size)+', sta: '+str(Nin))
                event_group.attrs.create(name = 'num_data_points', data = Nsize)
                event_group.create_dataset(name='lons', data = lons)
                event_group.create_dataset(name='lats', data = lats)
                event_group.create_dataset(name='phase_velocity', data = phase_velocity)
                event_group.create_dataset(name='snr', data = np.ones(Nsize) * 20.)
                event_group.create_dataset(name='distance', data = distance)
                try:
                    index_borrow = in_evgrp['index_borrow'][()]
                    event_group.create_dataset(name='index_borrow', data = index_borrow)
                except:
                    pass
        return
    
    
    
    
    
    
    