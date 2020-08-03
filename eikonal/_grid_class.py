# -*- coding: utf-8 -*-
"""
Perform data interpolation/computation on the surface of the Earth

    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import numpy.ma as ma
import numba
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from subprocess import call
import obspy.geodetics
from pyproj import Geod
import random
import copy
import metpy.calc 
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm

#--------------------------------------------------
# weight arrays for finite difference computation
#--------------------------------------------------

# first derivatives
lon_diff_weight_2   = np.array([[1., 0., -1.]])/2.
lat_diff_weight_2   = lon_diff_weight_2.T
lon_diff_weight_4   = np.array([[-1., 8., 0., -8., 1.]])/12.
lat_diff_weight_4   = lon_diff_weight_4.T
lon_diff_weight_6   = np.array([[1./60., 	-3./20.,  3./4.,  0., -3./4., 3./20.,  -1./60.]])
lat_diff_weight_6   = lon_diff_weight_6.T
# second derivatives
lon_diff2_weight_2  = np.array([[1., -2., 1.]])
lat_diff2_weight_2  = lon_diff2_weight_2.T
lon_diff2_weight_4  = np.array([[-1., 16., -30., 16., -1.]])/12.
lat_diff2_weight_4  = lon_diff2_weight_4.T
lon_diff2_weight_6  = np.array([[1./90., 	-3./20.,  3./2.,  -49./18., 3./2., -3./20.,  1./90.]])
lat_diff2_weight_6  = lon_diff2_weight_6.T

geodist             = Geod(ellps='WGS84')

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base        = plt.cm.get_cmap(base_cmap)
    color_list  = base(np.linspace(0, 1, N))
    cmap_name   = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def _write_txt(fname, outlon, outlat, outZ):
    outArr      = np.append(outlon, outlat)
    outArr      = np.append(outArr, outZ)
    outArr      = outArr.reshape((3,outZ.size))
    outArr      = outArr.T
    np.savetxt(fname, outArr, fmt='%g')
    return

def _green_integral(Nx, Ny, gradx, grady, dx, dy):
    lplc    = np.zeros((Ny, Nx), dtype=np.float64)
    for ix in range(Nx):
        # edge
        if ix == 0 or ix == (Nx-1):
            continue
        for iy in range(Ny):
            # edge
            if iy == 0 or iy == (Ny-1):
                continue
            # loopsum     = (gradx[iy-1, ix-1] + gradx[iy-1, ix])/2.*dx[iy-1, ix-1] + \
            #                 (gradx[iy-1, ix] + gradx[iy-1, ix+1])/2.*dx[iy-1, ix]
            # 
            # loopsum     += (grady[iy-1, ix+1] + grady[iy, ix+1])/2.*dy[iy-1, ix+1] + \
            #                 (grady[iy, ix+1] + grady[iy+1, ix+1])/2.*dy[iy, ix+1]
            # 
            # loopsum     += -(gradx[iy+1, ix+1] + gradx[iy+1, ix])/2.*dx[iy+1, ix] - \
            #                 (gradx[iy+1, ix] + gradx[iy+1, ix])/2.*dx[iy+1, ix-1]
            # 
            # loopsum     += -(grady[iy+1, ix-1] + grady[iy, ix-1])/2.*dy[iy, ix-1] - \
            #                 (grady[iy, ix-1] + grady[iy-1, ix-1])/2.*dy[iy-1, ix-1]
            # area        = dx[iy-1, ix-1] * dy[iy-1, ix] + dx[iy-1, ix] * dy[iy-1, ix] +\
            #                 dx[iy, ix-1] * dy[iy, ix] + dx[iy, ix] * dy[iy, ix]
            
            loopsum     = gradx[iy-1, ix]*dx[iy-1, ix] - gradx[iy+1, ix]*dx[iy+1, ix] \
                            + grady[iy, ix+1]*dy[iy, ix+1] - grady[iy, ix-1]* dy[iy, ix-1]
            area        = (dx[iy-1, ix] + dx[iy+1, ix]) * (dy[iy, ix+1] + dy[iy, ix-1])/4.
            lplc[iy, ix]= loopsum/area
    return lplc

@numba.jit(numba.int32[:, :](numba.int32[:, :], numba.float64[:, :], numba.int32, numba.float64), nopython = True)
def _check_neighbor_val(reason_n, zarr, indval, zval):
    Ny, Nx  = reason_n.shape
    for iy in range(Ny):
        for ix in range(Nx):
            if zarr[iy, ix] == zval:
                if ix > 0 and reason_n[iy, ix - 1] == 0:
                    reason_n[iy, ix - 1] = indval
                if iy > 0 and reason_n[iy - 1, ix] == 0:
                    reason_n[iy - 1, ix] = indval
                if ix < Nx-1 and reason_n[iy, ix + 1] == 0:
                    reason_n[iy, ix + 1] = indval
                if iy < Ny-1 and reason_n[iy + 1, ix] == 0:
                    reason_n[iy + 1, ix] = indval
    return reason_n


def _repeat_check(lons, lats, zarr):
    N       = lons.size
    index   = np.ones(N, dtype = bool)
    outz    = zarr.copy()
    for i in range(N):
        for j in range(N):
            if abs(lats[i] - lats[j]) > 0.001:
                continue
            if np.cos(lats[j]*np.pi/180.) * abs(lons[i] - lons[j]) > 0.001:
                continue
            
    return 

class SphereGridder(object):
    """a class to analyze 2D spherical grid data on Earth
    ===============================================================================================
    ::: parameters :::
    dlon, dlat              - grid interval
    Nlon, Nlat              - grid number in longitude, latitude
    lons, lats              - 1D arrays for grid locations
    lon2d, lat2d            - 2D arrays for grid locations
    minlon/maxlon           - minimum/maximum longitude
    minlat/maxlat           - minimum/maximum latitude
    dlon_km/dlat_km         - 1D arrays for grid interval in km
    dlon_km2d/dlat_km2d     - 2D arrays for grid interval in km
    dlon_km2d_metpy/dlat_km2d_metpy
                            - 2D arrays for grid interval in km, used for metpy
    period                  - period
    evid                    - id for the event
    fieldtype               - field type (Tph, Tgr, Amp)
    Zarr                    - 2D data array (shape: Nlat, Nlon)
    evlo/evla               - longitude/latitue of the event
    -----------------------------------------------------------------------------------------------
    ::: derived parameters :::
    --- gradient related, shape
    grad[0]/grad[1]         - gradient arrays
    pro_angle               - propagation angle arrays 
    app_vel                 - apparent velocity
    corr_vel                - corrected velocity
    reason_n                - index array indicating validity of data
    reason_n_helm           - index array indicating validity of data, used for Helmholtz 
    az/baz                  - azimuth/back-azimuth array
    diff_angle              - differences between propagation angle and azimuth
                                indicating off-great-circle deflection
    --- Laplacian related, shape: 
    lplc                    - Laplacian array
    reason_n                - index array indicating validity of data
    --- others
    mask                    - mask array, shape: Nlat, Nlon
    mask_helm               - mask for Helmholtz tomography, shape: Nlat, Nlon
    lplc_amp                - amplitude correction terms for phase speed
    Nvalid_grd/Ntotal_grd   - number of valid/total grid points, validity means reason_n == 0.
    -----------------------------------------------------------------------------------------------
    Note: meshgrid's default indexing is 'xy', which means:
    lons, lats = np.meshgrid[lon, lat]
    in lons[i, j] or lats[i, j],  i->lat, j->lon
    ===============================================================================================
    """
    def __init__(self, minlon, maxlon, dlon, minlat, maxlat, dlat,
            period = 10., evlo = float('inf'), evla = float('inf'), fieldtype='Tph', evid=''):
        self.dlon               = dlon
        self.dlat               = dlat
        self.Nlon               = int(round((maxlon-minlon)/dlon)+1)
        self.Nlat               = int(round((maxlat-minlat)/dlat)+1)
        self.lons               = np.arange(self.Nlon)*self.dlon+minlon
        self.minlon             = minlon
        self.maxlon             = self.lons[-1]
        self.lats               = np.arange(self.Nlat)*self.dlat+minlat
        self.minlat             = minlat
        self.maxlat             = self.lats[-1]
        self.fieldtype          = fieldtype
        # grid arrays
        self.lon2d, self.lat2d  = np.meshgrid(self.lons, self.lats)
        self._get_dlon_dlat_km()
        # data arrays
        self.Zarr               = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.az                 = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.baz                = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.grad               = []
        self.grad.append(np.zeros((self.Nlat, self.Nlon), dtype=np.float64))
        self.grad.append(np.zeros((self.Nlat, self.Nlon), dtype=np.float64))
        self.lplc               = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.lplc_amp           = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.app_vel            = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.pro_angle          = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.diff_angle         = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.corr_vel           = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        # flag arrays
        self.reason_n           = np.ones((self.Nlat, self.Nlon), dtype=np.int32)
        self.reason_n_helm      = np.ones((self.Nlat, self.Nlon), dtype=np.int32)
        self.mask               = np.zeros((self.Nlat, self.Nlon), dtype=bool)
        self.mask_helm          = np.zeros((self.Nlat, self.Nlon), dtype=bool)
        #-------------------------
        # surface wave attributes
        #-------------------------
        self.period             = period
        self.evid               = evid
        self.evlo               = evlo
        self.evla               = evla
        return
    
    def copy(self):
        return copy.deepcopy(self)
    
    def  _get_dlon_dlat_km(self):
        """get longitude and latitude interval in km
        """
        az, baz, dist_lon       = geodist.inv(np.zeros(self.lats.size), self.lats, np.ones(self.lats.size)*self.dlon, self.lats) 
        az, baz, dist_lat       = geodist.inv(np.zeros(self.lats.size), self.lats, np.zeros(self.lats.size), self.lats + self.dlat) 
        self.dlon_km            = dist_lon/1000.
        self.dlat_km            = dist_lat/1000.
        self.dlon_km2d          = (np.tile(self.dlon_km, self.Nlon).reshape(self.Nlon, self.Nlat)).T
        self.dlat_km2d          = (np.tile(self.dlat_km, self.Nlon).reshape(self.Nlon, self.Nlat)).T
        # spacing arrays for gradient/Laplacian using metpy
        self.dlon_km2d_metpy    = (np.tile(self.dlon_km, self.Nlon - 1).reshape(self.Nlon - 1, self.Nlat)).T
        self.dlat_km2d_metpy    = (np.tile(self.dlat_km[:-1], self.Nlon).reshape(self.Nlon, self.Nlat - 1)).T
        return
    
    #--------------------------------------------------
    # functions for I/O
    #--------------------------------------------------
    
    def read(self, fname):
        """read field file
        """
        try:
            Inarray         = np.loadtxt(fname)
            with open(fname) as f:
                inline      = f.readline()
                if inline.split()[0] =='#':
                    evlostr = inline.split()[1]
                    evlastr = inline.split()[2]
                    if evlostr.split('=')[0] =='evlo':
                        self.evlo   = float(evlostr.split('=')[1])
                    if evlastr.split('=')[0] =='evla':
                        self.evla   = float(evlastr.split('=')[1])
        except:
            Inarray     = np.load(fname)
        self.lonsIn     = Inarray[:,0]
        self.latsIn     = Inarray[:,1]
        self.ZarrIn     = Inarray[:,2]
        return
    
    def read_ind(self, fname, zindex = 2, dindex=None):
        """read field file
        """
        try:
            Inarray                 = np.loadtxt(fname)
            with open(fname) as f:
                inline              = f.readline()
                if inline.split()[0] =='#':
                    evlostr         = inline.split()[1]
                    evlastr         = inline.split()[2]
                    if evlostr.split('=')[0] =='evlo':
                        self.evlo   = float(evlostr.split('=')[1])
                    if evlastr.split('=')[0] =='evla':
                        self.evla   = float(evlastr.split('=')[1])
        except:
            Inarray     = np.load(fname)
        self.lonsIn     = Inarray[:,0]
        self.latsIn     = Inarray[:,1]
        self.ZarrIn     = Inarray[:,zindex]*1e9
        if dindex is not None:
            darrIn      = Inarray[:,dindex]
            self.ZarrIn = darrIn/Inarray[:,zindex]
        return
    
    def read_array(self, inlons, inlats, inzarr):
        """read field file
        """
        self.lonsIn     = inlons
        self.latsIn     = inlats
        self.ZarrIn     = inzarr
        return
    
    def load_field(self, ingridder):
        """load field data from an input object
        """
        self.lonsIn     = ingridder.lonsIn
        self.latsIn     = ingridder.latsIn
        self.ZarrIn     = ingridder.ZarrIn
        return
    
    def write(self, fname, fmt='npy'):
        """Save field file
        """
        OutArr      = np.append(self.lon2d, self.lat2d)
        OutArr      = np.append(OutArr, self.Zarr)
        OutArr      = OutArr.reshape(3, self.Nlon*self.Nlat)
        OutArr      = OutArr.T
        if fmt is 'npy':
            np.save(fname, OutArr)
        elif fmt is 'txt':
            np.savetxt(fname, OutArr)
        else:
            raise TypeError('Wrong output format!')
        return
    
    def write_binary(self, outfname, amplplc=False):
        """write data arrays to a binary npy file
        """
        if amplplc:
            np.savez( outfname, self.app_vel, self.reason_n, self.pro_angle, self.az, self.baz, self.Zarr,\
                     self.lplc_amp, self.corr_vel, self.reason_n_helm, np.array([self.Ntotal_grd, self.Nvalid_grd]))
        else:
            np.savez( outfname, self.app_vel, self.reason_n, self.pro_angle, self.az, self.baz, self.Zarr,\
                        np.array([self.Ntotal_grd, self.Nvalid_grd]))
        return
    
    def cut_edge(self, nlon, nlat):
        """cut edge
        =======================================================================================
        ::: input parameters :::
        nlon, nlon  - number of edge point in longitude/latitude to be cutted
        =======================================================================================
        """
        self.Nlon               = self.Nlon - 2*nlon
        self.Nlat               = self.Nlat - 2*nlat
        self.minlon             = self.minlon + nlon*self.dlon
        self.maxlon             = self.maxlon - nlon*self.dlon
        self.minlat             = self.minlat + nlat*self.dlat
        self.maxlat             = self.maxlat - nlat*self.dlat
        self.lons               = np.arange(self.Nlon)*self.dlon+self.minlon
        self.lats               = np.arange(self.Nlat)*self.dlat+self.minlat
        self.lon2d,self.lat2d   = np.meshgrid(self.lons, self.lats)
        self.Zarr               = self.Zarr[nlat:-nlat, nlon:-nlon]
        self.az               = self.az[nlat:-nlat, nlon:-nlon]
        self.baz               = self.baz[nlat:-nlat, nlon:-nlon]
        self.grad[0]            = self.grad[0][nlat:-nlat, nlon:-nlon]
        self.grad[1]            = self.grad[1][nlat:-nlat, nlon:-nlon]
        self.lplc               = self.lplc[nlat:-nlat, nlon:-nlon]
        self.lplc_amp           = self.lplc_amp[nlat:-nlat, nlon:-nlon]
        self.app_vel            = self.app_vel[nlat:-nlat, nlon:-nlon]
        self.pro_angle          = self.pro_angle[nlat:-nlat, nlon:-nlon]
        self.corr_vel           = self.corr_vel[nlat:-nlat, nlon:-nlon]
        # flag arrays
        self.reason_n           = self.reason_n[nlat:-nlat, nlon:-nlon]
        self.reason_n_helm      = self.reason_n_helm[nlat:-nlat, nlon:-nlon]
        self.mask               = self.mask[nlat:-nlat, nlon:-nlon]
        self.mask_helm          = self.mask_helm[nlat:-nlat, nlon:-nlon]
        self._get_dlon_dlat_km()
        return
    
    # synthetic tests 
    def synthetic_field(self, lat0, lon0, v=3.0):
        """generate synthetic field data
        """
        az, baz, distevent  = geodist.inv( np.ones(self.lonsIn.size)*lon0, np.ones(self.lonsIn.size)*lat0, self.lonsIn, self.latsIn)
        self.ZarrIn         = distevent/v/1000.
        return
    
    def add_noise(self, sigma=0.5):
        """Add Gaussian noise with standard deviation = sigma to the input data
            used for synthetic test
        """
        for i in xrange(self.ZarrIn.size):
            self.ZarrIn[i]  = self.ZarrIn[i] + random.gauss(0, sigma)
        return
    
    #==================================================
    # functions for interpolation/gradient/Laplacian 
    #==================================================
    
    def interp_surface(self, workingdir, outfname, tension = 0.0):
        """interpolate input data to grid point with GMT surface command
        =======================================================================================
        ::: input parameters :::
        workingdir  - working directory
        outfname    - output file name for interpolation
        tension     - input tension for gmt surface(0.0-1.0)
        ---------------------------------------------------------------------------------------
        ::: output :::
        self.Zarr   - interpolated field data
        ---------------------------------------------------------------------------------------
        version history
            - 2018/07/06    : added the capability of interpolation for dlon != dlat
        =======================================================================================
        """
        if not os.path.isdir(workingdir):
            os.makedirs(workingdir)
        raw_fname   = workingdir+'/raw_'+outfname
        qc_fname    = workingdir+'/'+outfname
        outarr      = np.append(self.lonsIn, self.latsIn)
        outarr      = np.append(outarr, self.ZarrIn)
        outarr      = outarr.reshape(3, self.lonsIn.size)
        outarr      = outarr.T
        np.savetxt(raw_fname, outarr, fmt='%g')
        fnameHD     = workingdir+'/'+outfname+'.HD'
        tempGMT     = workingdir+'/'+outfname+'_GMT.sh'
        grdfile     = workingdir+'/'+outfname+'.grd'
        with open(tempGMT,'w') as f:
            REG     = '-R'+str(self.minlon)+'/'+str(self.maxlon)+'/'+str(self.minlat)+'/'+str(self.maxlat)
            f.writelines('gmt gmtset MAP_FRAME_TYPE fancy \n')
            # interpolation
            if self.dlon == self.dlat:
                # # # f.writelines('gmt blockmean %s -I%g %s > %s \n' %( raw_fname, 0.1, REG, qc_fname))
                f.writelines('gmt blockmean %s -I%g %s > %s \n' %( raw_fname, self.dlon, REG, qc_fname))
                f.writelines('gmt surface %s -T%g -G%s -I%g %s \n' %( qc_fname, tension, grdfile, self.dlon, REG ))
            else:
                # # # f.writelines('gmt blockmean %s -I%g/%g %s > %s \n' %( raw_fname, (0.1*self.dlon/self.dlat), 0.1, REG, qc_fname))
                f.writelines('gmt blockmean %s -I%g/%g %s > %s \n' %( raw_fname, self.dlon, self.dlat, REG, qc_fname))
                f.writelines('gmt surface %s -T%g -G%s -I%g/%g %s \n' %( qc_fname, tension, grdfile, self.dlon, self.dlat, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile, REG, fnameHD ))
        call(['bash', tempGMT])
        os.remove(grdfile)
        os.remove(tempGMT)
        inArr       = np.loadtxt(fnameHD)
        ZarrIn      = inArr[:, 2]
        self.Zarr[:]= (ZarrIn.reshape(self.Nlat, self.Nlon))[::-1, :]
        return
    
    def gauss_smoothing(self, workingdir, outfname, tension=0.0, width=50.):
        """perform a Gaussian smoothing
        =======================================================================================
        ::: input parameters :::
        workingdir  - working directory
        outfname    - output file name for interpolation
        tension     - input tension for gmt surface(0.0-1.0)
        width       - Gaussian width in km
        ---------------------------------------------------------------------------------------
        ::: output :::
        self.Zarr   - smoothed field data
        =======================================================================================
        """
        if not os.path.isdir(workingdir):
            os.makedirs(workingdir)
        raw_fname   = workingdir+'/raw_'+outfname
        qc_fname    = workingdir+'/'+outfname
        outarr      = np.append(self.lonsIn, self.latsIn)
        outarr      = np.append(outarr, self.ZarrIn)
        outarr      = outarr.reshape(3, self.lonsIn.size)
        outarr      = outarr.T
        np.savetxt(raw_fname, outarr, fmt='%g')
        fnameHD     = workingdir+'/'+outfname+'.HD'
        tempGMT     = workingdir+'/'+outfname+'_GMT.sh'
        grdfile     = workingdir+'/'+outfname+'.grd'
        outgrd      = workingdir+'/'+outfname+'_filtered.grd'
        # http://gmt.soest.hawaii.edu/doc/5.3.2/grdfilter.html
        # (g) Gaussian: Weights are given by the Gaussian function,
        # where width is 6 times the conventional Gaussian sigma.
        width       = 6.*width
        with open(tempGMT,'wb') as f:
            REG     = '-R'+str(self.minlon)+'/'+str(self.maxlon)+'/'+str(self.minlat)+'/'+str(self.maxlat)
            f.writelines('gmt gmtset MAP_FRAME_TYPE fancy \n')
            if self.dlon == self.dlat:
                # # # f.writelines('gmt blockmean %s -I%g %s > %s \n' %( raw_fname, 0.01, REG, qc_fname))
                f.writelines('gmt blockmean %s -I%g %s > %s \n' %( raw_fname, self.dlon, REG, qc_fname))
                f.writelines('gmt surface %s -T%g -G%s -I%g %s \n' %( qc_fname, tension, grdfile, self.dlon, REG ))
            else:
                # # # f.writelines('gmt blockmean %s -I%g/%g %s > %s \n' %( raw_fname, (0.01*self.dlon/self.dlat), 0.01, REG, qc_fname))
                f.writelines('gmt blockmean %s -I%g/%g %s > %s \n' %( raw_fname, self.dlon, self.dlat, REG, qc_fname))
                f.writelines('gmt surface %s -T%g -G%s -I%g/%g %s \n' %( qc_fname, tension, grdfile, self.dlon, self.dlat, REG ))
            f.writelines('gmt grdfilter %s -D4 -Fg%g -G%s %s \n' %( grdfile, width, outgrd, REG))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( outgrd, REG, fnameHD ))
        call(['bash', tempGMT])
        os.remove(grdfile)
        os.remove(outgrd)
        os.remove(tempGMT)
        inArr       = np.loadtxt(fnameHD)
        ZarrIn      = inArr[:, 2]
        self.Zarr[:]= (ZarrIn.reshape(self.Nlat, self.Nlon))[::-1, :]
        return
    
    def gradient(self, method='metpy', order=2):
        """compute gradient of the field
        =============================================================================================
        ::: input parameters :::
        method      - method: 'metpy' : use metpy; 'convolve': use convolution
        order       - order of finite difference scheme, only takes effect when method='convolve'
        =============================================================================================
        """
        if method=='metpy':
            self.grad[0][:], self.grad[1][:] \
                            = metpy.calc.gradient(self.Zarr, deltas=(self.dlat_km2d_metpy, self.dlon_km2d_metpy))
        elif method == 'convolve':
            if order==2:
                diff_lon    = convolve(self.Zarr, lon_diff_weight_2)/self.dlon_km2d
                diff_lat    = convolve(self.Zarr, lat_diff_weight_2)/self.dlat_km2d
            elif order==4:
                diff_lon    = convolve(self.Zarr, lon_diff_weight_4)/self.dlon_km2d
                diff_lat    = convolve(self.Zarr, lat_diff_weight_4)/self.dlat_km2d
            elif order==6:
                diff_lon    = convolve(self.Zarr, lon_diff_weight_6)/self.dlon_km2d
                diff_lat    = convolve(self.Zarr, lat_diff_weight_6)/self.dlat_km2d
            else:
                raise ValueError('Unexpected order = %d'+ order)
            self.grad[0][:] = diff_lat
            self.grad[1][:] = diff_lon
        else:
            raise ValueError('Unexpected method = '+ method)
        # propagation direction angle
        self.pro_angle[:]   = np.arctan2(self.grad[0], self.grad[1])/np.pi*180.
        return
    
    def laplacian(self, method='metpy', order=4, verbose=False):
        """compute Laplacian of the field
        =============================================================================================================
        ::: input parameters :::
        method      - method: 'metpy'   : use metpy
                              'convolve': use convolution
                              'green'   : use Green's theorem( 2D Gauss's theorem )
        order       - order of finite difference scheme, only has effect when method='convolve'
        =============================================================================================================
        """
        if method == 'metpy':
            self.lplc[:]    = metpy.calc.laplacian(self.Zarr, deltas=(self.dlat_km2d_metpy, self.dlon_km2d_metpy)) 
        elif method == 'convolve':
            if order==2:
                diff2_lon   = convolve(self.Zarr, lon_diff2_weight_2)/self.dlon_km2d/self.dlon_km2d
                diff2_lat   = convolve(self.Zarr, lat_diff2_weight_2)/self.dlat_km2d/self.dlat_km2d
            elif order==4:
                diff2_lon   = convolve(self.Zarr, lon_diff2_weight_4)/self.dlon_km2d/self.dlon_km2d
                diff2_lat   = convolve(self.Zarr, lat_diff2_weight_4)/self.dlat_km2d/self.dlat_km2d
            elif order==6:
                diff2_lon   = convolve(self.Zarr, lon_diff2_weight_6)/self.dlon_km2d/self.dlon_km2d
                diff2_lat   = convolve(self.Zarr, lat_diff2_weight_6)/self.dlat_km2d/self.dlat_km2d
            self.lplc[:]    = diff2_lon + diff2_lat
        else:
            raise ValueError('Unexpected method = '+ method)
        # # elif method=='green':
        # #     #----------------
        # #     # gradient arrays
        # #     #----------------
        # #     self.gradient(method = 'metpy')
        # #     # self.lplc[:]        = _green_integral(self.Nlon, self.Nlat, self.grad[0], self.grad[1],\
        # #     #                             self.dlon_km2d_metpy, self.dlat_km2d_metpy)
        # #     
        # #     self.lplc[:]        = _green_integral(self.Nlon, self.Nlat, self.grad[1], self.grad[0],\
        # #                                 self.dlon_km2d, self.dlat_km2d)
        if verbose:
            print ('max lplc:',self.lplc.max(), 'min lplc:',self.lplc.min())
        return
    
    def get_apparent_vel(self):
        """Get the apparent velocity from gradient
        """
        slowness                = np.sqrt ( self.grad[0] ** 2 + self.grad[1] ** 2)
        slowness[slowness==0]   = 0.3
        self.app_vel[:]         = 1./slowness
        return
      
    #--------------------------------------------------
    # functions for data quality controls
    #--------------------------------------------------
    
    def check_curvature(self, workingdir, outpfx='', threshold=0.005):
        """
        Check and discard data points with large curvatures.
        Points at boundaries will be discarded.
        Two interpolation schemes with different tension (0, 0.2) will be applied to the quality controlled field data file. 
        =====================================================================================================================
        ::: input parameters :::
        workingdir  - working directory
        outpfx      - prefix for output files
        threshold   - threshold value for Laplacian, default - 0.005, the value is suggested in Lin et al.(2009)
        ---------------------------------------------------------------------------------------------------------------------
        ::: output :::
        workingdir/outpfx+fieldtype_per_v1.lst         - output field file with data point passing curvature checking
        workingdir/outpfx+fieldtype_per_v1.lst.HD      - interpolated travel time file 
        workingdir/outpfx+fieldtype_per_v1.lst.HD_0.2  - interpolated travel time file with tension=0.2
        ---------------------------------------------------------------------------------------------------------------------
        version history
            - 07/06/2018    : added the capability of dealing with dlon != dlat
        =====================================================================================================================
        """
        # if outpfx == '' and self.evid != '':
        #     outpfx  = self.evid + '_'
        # Compute Laplacian
        self.laplacian(method='metpy')
        tmpgrder    = self.copy()
        #--------------------
        # quality control
        #--------------------
        LonLst      = tmpgrder.lon2d.reshape(tmpgrder.lon2d.size)
        LatLst      = tmpgrder.lat2d.reshape(tmpgrder.lat2d.size)
        TLst        = tmpgrder.Zarr.reshape(tmpgrder.Zarr.size)
        lplc        = self.lplc.reshape(self.lplc.size)
        index       = np.where((lplc>-threshold)*(lplc<threshold))[0]
        # 09/24/2018, if no data 
        if index.size == 0:
            return False
        LonLst      = LonLst[index]
        LatLst      = LatLst[index]
        TLst        = TLst[index]
        # output to txt file
        outfname    = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1.lst'
        TfnameHD    = outfname+'.HD'
        _write_txt(fname=outfname, outlon=LonLst, outlat=LatLst, outZ=TLst)
        # interpolate with gmt surface
        tempGMT     = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1_GMT.sh'
        grdfile     = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1.grd'
        with open(tempGMT,'w') as f:
            REG     = '-R'+str(self.minlon)+'/'+str(self.maxlon)+'/'+str(self.minlat)+'/'+str(self.maxlat)
            f.writelines('gmt gmtset MAP_FRAME_TYPE fancy \n')
            if self.dlon == self.dlat:
                f.writelines('gmt surface %s -T0.0 -G%s -I%g %s \n' %( outfname, grdfile, self.dlon, REG ))
            else:
                f.writelines('gmt surface %s -T0.0 -G%s -I%g/%g %s \n' %( outfname, grdfile, self.dlon, self.dlat, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile, REG, TfnameHD ))
            if self.dlon == self.dlat:
                f.writelines('gmt surface %s -T0.2 -G%s -I%g %s \n' %( outfname, grdfile+'.T0.2', self.dlon, REG ))
            else:
                f.writelines('gmt surface %s -T0.2 -G%s -I%g/%g %s \n' %( outfname, grdfile+'.T0.2', self.dlon, self.dlat, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile+'.T0.2', REG, TfnameHD+'_0.2' ))
        call(['bash', tempGMT])
        os.remove(grdfile+'.T0.2')
        os.remove(grdfile)
        os.remove(tempGMT)
        return True
    
    def check_curvature_amp(self, workingdir, outpfx='', threshold=0.2):
        """
        Check and discard data points with large curvatures, designed for amplitude field
        Points at boundaries will be discarded.
        Two interpolation schemes with different tension (0, 0.2) will be applied to the quality controlled field data file. 
        =====================================================================================================================
        ::: input parameters :::
        workingdir  - working directory
        threshold   - threshold value for Laplacian
        ---------------------------------------------------------------------------------------------------------------------
        ::: output :::
        workingdir/outpfx+fieldtype_per_v1.lst         - output field file with data point passing curvature checking
        workingdir/outpfx+fieldtype_per_v1.lst.HD      - interpolated travel time file 
        workingdir/outpfx+fieldtype_per_v1.lst.HD_0.2  - interpolated travel time file with tension=0.2
        ---------------------------------------------------------------------------------------------------------------------
        version history
            - 2018/07/06    : added the capability of dealing with dlon != dlat
        =====================================================================================================================
        """
        # if outpfx == '' and self.evid != '':
        #     outpfx  = self.evid + '_'
        # Compute Laplacian
        self.laplacian(method='metpy')
        tmpgrder    = self.copy()
        threshold   = threshold*2./(3.**2)
        #--------------------
        # quality control
        #--------------------
        LonLst      = tmpgrder.lon2d.reshape(tmpgrder.lon2d.size)
        LatLst      = tmpgrder.lat2d.reshape(tmpgrder.lat2d.size)
        ampLst      = tmpgrder.Zarr.reshape(tmpgrder.Zarr.size)
        # # # lplc        = self.lplc.reshape(self.lplc.size)
        # # # lplc_corr   = lplc.copy()
        # # # lplc_corr   = self.lplc.copy()
        # # # lplc_corr[ampLst!=0.]\
        # # #             = lplc[ampLst!=0.]/ampLst[ampLst!=0.]
        # # # lplc_corr[ampLst==0.]\
        # # #             = 0.
        lplc_corr   = self.lplc.copy()
        #
        if lplc_corr.shape != tmpgrder.Zarr.shape:
            raise ValueError('001: '+tmpgrder.evid)
        #
        lplc_corr[tmpgrder.Zarr==0.]\
                    = 0.
        lplc_corr   = lplc_corr.reshape(lplc_corr.size)
        omega       = 2.*np.pi/self.period
        # original
        # lplc_corr   = lplc_corr/(omega**2)
        # index       = np.where((lplc_corr>-threshold)*(lplc_corr<threshold))[0]
        # new
        c0          = 4.
        threshold   = (ampLst*omega*omega/c0/c0)
        index       = np.where((lplc_corr>-threshold)*(lplc_corr<threshold))[0]
        if index.size == 0:
            return False
        LonLst      = LonLst[index]
        LatLst      = LatLst[index]
        ampLst      = ampLst[index]
        # output to txt file
        outfname    = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1.lst'
        AfnameHD    = outfname+'.HD'
        _write_txt(fname=outfname, outlon=LonLst, outlat=LatLst, outZ=ampLst)
        # interpolate with gmt surface
        tempGMT     = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1_GMT.sh'
        grdfile     = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1.grd'
        with open(tempGMT,'w') as f:
            REG     = '-R'+str(self.minlon)+'/'+str(self.maxlon)+'/'+str(self.minlat)+'/'+str(self.maxlat)            
            f.writelines('gmt gmtset MAP_FRAME_TYPE fancy \n')
            if self.dlon == self.dlat:
                f.writelines('gmt surface %s -T0.0 -G%s -I%g %s \n' %( outfname, grdfile, self.dlon, REG ))
            else:
                f.writelines('gmt surface %s -T0.0 -G%s -I%g/%g %s \n' %( outfname, grdfile, self.dlon, self.dlat, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile, REG, AfnameHD ))
            if self.dlon == self.dlat:
                f.writelines('gmt surface %s -T0.2 -G%s -I%g %s \n' %( outfname, grdfile+'.T0.2', self.dlon, REG ))
            else:
                f.writelines('gmt surface %s -T0.2 -G%s -I%g/%g %s \n' %( outfname, grdfile+'.T0.2', self.dlon, self.dlat, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile+'.T0.2', REG, AfnameHD+'_0.2' ))
        call(['bash', tempGMT])
        os.remove(grdfile+'.T0.2')
        os.remove(grdfile)
        os.remove(tempGMT)
        return True
        
    def eikonal(self, workingdir, inpfx='', nearneighbor=True, cdist=150., lplcthresh=0.005, lplcnearneighbor=False):
        """
        Generate slowness maps from travel time maps using eikonal equation
        Two interpolated travel time file with different tension will be used for quality control.
        =====================================================================================================================
        ::: input parameters :::
        workingdir      - working directory
        inpfx           - prefix for input files
        nearneighbor    - do near neighbor quality control or not
        cdist           - distance for quality control, default is 12*period
        lplcthresh      - threshold value for Laplacian
        lplcnearneighbor- also discard near neighbor points for a grid point with large Laplacian
        ::: output format :::
        outdir/slow_azi_stacode.pflag.txt.HD.2.v2 - Slowness map
        =====================================================================================================================
        """
        if cdist is None:
            cdist   = max(12.*self.period/3., 150.)
        evlo        = self.evlo
        evla        = self.evla
        #===============================================================================
        # Read data
        # v1: data that passes check_curvature criterion
        # v1HD and v1HD02: interpolated v1 data with tension = 0. and 0.2
        #===============================================================================
        fnamev1     = workingdir+'/'+inpfx+self.fieldtype+'_'+str(self.period)+'_v1.lst'
        fnamev1HD   = fnamev1+'.HD'
        fnamev1HD02 = fnamev1HD+'_0.2'
        InarrayV1   = np.loadtxt(fnamev1)
        loninV1     = InarrayV1[:,0]
        latinV1     = InarrayV1[:,1]
        fieldin     = InarrayV1[:,2]
        Inv1HD      = np.loadtxt(fnamev1HD)
        lonv1HD     = Inv1HD[:,0]
        latv1HD     = Inv1HD[:,1]
        fieldv1HD   = Inv1HD[:,2]
        Inv1HD02    = np.loadtxt(fnamev1HD02)
        lonv1HD02   = Inv1HD02[:,0]
        latv1HD02   = Inv1HD02[:,1]
        fieldv1HD02 = Inv1HD02[:,2]
        # Set field value to be zero if there is large difference between v1HD and v1HD02
        diffArr     = fieldv1HD - fieldv1HD02
        fieldArr    = fieldv1HD*((diffArr<2.)*(diffArr>-2.))
        fieldArr    = (fieldArr.reshape(self.Nlat, self.Nlon))[::-1, :]
        #===================================================================================
        # reason_n array
        #   0: accepted point
        #   1: data point the has large difference between v1HD and v1HD02
        #   2: data point that does not have near neighbor points at all E/W/N/S directions
        #   3: slowness is too large/small
        #   4: near a zero field data point
        #   5: epicentral distance is too small
        #   6: large curvature
        #===================================================================================
        reason_n    = np.ones(fieldArr.size, dtype=np.int32)
        reason_n    = np.int32(reason_n*(diffArr>2.)) + np.int32(reason_n*(diffArr<-2.))
        reason_n    = (reason_n.reshape(self.Nlat, self.Nlon))[::-1,:]
        #-------------------------------------------------------------------------------------------------------
        # check each data point if there are close-by four stations located at E/W/N/S directions respectively
        #-------------------------------------------------------------------------------------------------------
        if nearneighbor:
            for ilat in range(self.Nlat):
                for ilon in range(self.Nlon):
                    if reason_n[ilat, ilon]==1:
                        continue
                    lon         = self.lons[ilon]
                    lat         = self.lats[ilat]
                    dlon_km     = self.dlon_km[ilat]
                    dlat_km     = self.dlat_km[ilat]
                    difflon     = abs(self.lonsIn-lon)/self.dlon*dlon_km
                    difflat     = abs(self.latsIn-lat)/self.dlat*dlat_km
                    index       = np.where((difflon<cdist)*(difflat<cdist))[0]
                    marker_EN   = np.zeros((2,2), dtype=bool)
                    marker_nn   = 4
                    tflag       = False
                    for iv1 in index:
                        lon2    = self.lonsIn[iv1]
                        lat2    = self.latsIn[iv1]
                        if lon2-lon<0:
                            marker_E    = 0
                        else:
                            marker_E    = 1
                        if lat2-lat<0:
                            marker_N    = 0
                        else:
                            marker_N    = 1
                        if marker_EN[marker_E , marker_N]:
                            continue
                        az, baz, dist   = geodist.inv(lon, lat, lon2, lat2) # loninArr/latinArr are initial points
                        dist            = dist/1000.
                        if dist< cdist*2 and dist >= 1:
                            marker_nn   = marker_nn - 1
                            if marker_nn == 0:
                                tflag   = True
                                break
                            marker_EN[marker_E, marker_N]   = True
                    if not tflag:
                        fieldArr[ilat, ilon]    = 0
                        reason_n[ilat, ilon]    = 2
        # Start to Compute Gradient
        tfield                      = self.copy()
        tfield.Zarr                 = fieldArr
        tfield.gradient('metpy')
        # if one field point has zero value, reason_n for four near neighbor points will all be set to 4
        reason_n                    = _check_neighbor_val(reason_n, tfield.Zarr, np.int32(4), np.float64(0.))
        # if slowness is too large/small, reason_n will be set to 3
        slowness                    = np.sqrt(tfield.grad[0]**2 + tfield.grad[1]**2)
        if self.fieldtype=='Tph' or self.fieldtype=='Tgr':
            reason_n[(slowness>0.5)*(reason_n==0)]  = 3
            reason_n[(slowness<0.2)*(reason_n==0)]  = 3
        #-------------------------------------
        # computing propagation deflection
        #-------------------------------------
        ind0                        = np.where(reason_n==0)
        diff_angle                  = np.zeros(reason_n.shape, dtype = np.float64)
        latsin                      = self.lats[ind0[0]]
        lonsin                      = self.lons[ind0[1]]
        evloarr                     = np.ones(latsin.size, dtype=np.float64)*evlo
        evlaarr                     = np.ones(lonsin.size, dtype=np.float64)*evla
        az, baz, distevent          = geodist.inv(lonsin, latsin, evloarr, evlaarr) # loninArr/latinArr are initial points
        distevent                   = distevent/1000.        
        az                          = az + 180.
        az                          = 90.-az
        baz                         = 90.-baz
        az[az>180.]                 = az[az>180.] - 360.
        az[az<-180.]                = az[az<-180.] + 360.
        baz[baz>180.]               = baz[baz>180.] - 360.
        baz[baz<-180.]              = baz[baz<-180.] + 360.
        # az azimuth receiver -> source 
        diff_angle[ind0[0], ind0[1]]= tfield.pro_angle[ind0[0], ind0[1]] - az
        self.gradient('metpy')
        self.az[ind0[0], ind0[1]]   = az
        self.baz[ind0[0], ind0[1]]  = baz
        #---------------------------
        # three wavelength criteria
        #---------------------------
        # if epicentral distance is too small, reason_n will be set to 5, and diff_angle will be 0.
        dist_per                    = 4.*self.period*3.
        indnear                     = distevent<dist_per
        tmparr                      = diff_angle[ind0[0], ind0[1]]
        tmparr[indnear]             = 0.
        diff_angle[ind0[0], ind0[1]]= tmparr
        diff_angle[diff_angle>180.] = diff_angle[diff_angle>180.] - 360.
        diff_angle[diff_angle<-180.]= diff_angle[diff_angle<-180.] + 360.
        tmparr                      = reason_n[ind0[0], ind0[1]]
        tmparr[indnear]             = 5
        reason_n[ind0[0], ind0[1]]  = tmparr
        #------------------------------------------------------------------------
        # final check of curvature, discard grid points with large curvature
        #------------------------------------------------------------------------
        self.laplacian(method='metpy')
        tempind                     = (self.lplc > lplcthresh) + (self.lplc < -lplcthresh)
        reason_n[tempind]           = 6
        # near neighbor discard for large curvature
        if lplcnearneighbor:
            reason_n                = _check_neighbor_val(reason_n, np.float64(reason_n), np.int32(6), np.float64(6))
        # store final data
        self.diff_angle[:]          = diff_angle
        self.grad[0][:]             = tfield.grad[0]
        self.grad[1][:]             = tfield.grad[1]
        self.get_apparent_vel()
        self.reason_n[:]            = reason_n
        self.mask[:]                = reason_n != 0
        self.Nvalid_grd             = (np.where(reason_n == 0)[0]).size
        self.Ntotal_grd             = reason_n.size
        return
    
    # 
    # def helmholtz_operator(self, workingdir, inpfx='', lplcthresh=0.2):
    #     """
    #     Generate amplitude Laplacian maps for helmholtz tomography
    #     Two interpolated amplitude file with different tension will be used for quality control.
    #     =====================================================================================================================
    #     ::: input parameters :::
    #     workingdir      - working directory
    #     inpfx           - prefix for input files
    #     lplcthresh      - threshold value for Laplacian
    #     =====================================================================================================================
    #     """
    #     # Read data,
    #     # v1: data that pass check_curvature criterion
    #     # v1HD and v1HD02: interpolated v1 data with tension = 0. and 0.2
    #     fnamev1     = workingdir+'/'+inpfx+self.fieldtype+'_'+str(self.period)+'_v1.lst'
    #     fnamev1HD   = fnamev1+'.HD'
    #     fnamev1HD02 = fnamev1HD+'_0.2'
    #     InarrayV1   = np.loadtxt(fnamev1)
    #     loninV1     = InarrayV1[:,0]
    #     latinV1     = InarrayV1[:,1]
    #     fieldin     = InarrayV1[:,2]
    #     Inv1HD      = np.loadtxt(fnamev1HD)
    #     lonv1HD     = Inv1HD[:,0]
    #     latv1HD     = Inv1HD[:,1]
    #     fieldv1HD   = Inv1HD[:,2]
    #     Inv1HD02    = np.loadtxt(fnamev1HD02)
    #     lonv1HD02   = Inv1HD02[:,0]
    #     latv1HD02   = Inv1HD02[:,1]
    #     fieldv1HD02 = Inv1HD02[:,2]
    #     # Set field value to be zero if there is large difference between v1HD and v1HD02
    #     diffArr     = fieldv1HD - fieldv1HD02
    #     # new
    #     fieldArr    = fieldv1HD*(abs(diffArr)<abs(0.01*fieldv1HD))
    #     #old
    #     #med_amp     = np.median(self.Zarr)
    #     #fieldArr    = fieldv1HD*((diffArr<0.01*med_amp)*(diffArr>-0.01*med_amp))
    #     fieldArr    = (fieldArr.reshape(self.Nlat, self.Nlon))[::-1, :]
    #     # reason_n 
    #     #   0: accepted point
    #     #   1: data point the has large difference between v1HD and v1HD02
    #     #   3: large curvature
    #     #   4: near a zero field data point
    #     reason_n    = np.ones(fieldArr.size, dtype=np.int32)
    #     # new
    #     reason_n1   = np.int32(reason_n*(diffArr>abs(0.01*fieldv1HD)))
    #     reason_n2   = np.int32(reason_n*(diffArr<-abs(0.01*fieldv1HD)))
    #     # old
    #     # reason_n1   = np.int32(reason_n*(diffArr>0.01*med_amp))
    #     # reason_n2   = np.int32(reason_n*(diffArr<-0.01*med_amp))
    #     reason_n    = reason_n1 + reason_n2
    #     reason_n    = (reason_n.reshape(self.Nlat, self.Nlon))[::-1,:]
    #     #-------------------------------
    #     # Start to compute Laplacian
    #     #-------------------------------
    #     tfield                      = self.copy()
    #     tfield.Zarr                 = fieldArr
    #     tfield.Laplacian(method='green') ## schemes
    #     # if one field point has zero value, reason_n for four near neighbor points will all be set to 4
    #     tempZarr                    = tfield.Zarr[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
    #     index0                      = np.where(tempZarr==0.)
    #     ilat2d                     = index0[0] + 1
    #     ilon2d                     = index0[1] + 1
    #     reason_n[ilat2d+1, ilon2d]= 4
    #     reason_n[ilat2d-1, ilon2d]= 4
    #     reason_n[ilat2d, ilon2d+1]= 4
    #     reason_n[ilat2d, ilon2d-1]= 4
    #     # reduce size of reason_n to be the same shape as Laplacian arrays
    #     reason_n                    = reason_n[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
    #     # if Laplacian is too large/small, reason_n will be set to 3
    #     lplc_corr                   = tfield.lplc.copy()
    #     # lplc_corr[tempZarr!=0.]     = \
    #     #             tfield.lplc[tempZarr!=0.]/tempZarr[tempZarr!=0.]
    #     lplc_corr[tempZarr==0.]     = 0.
    #     omega                       = 2.*np.pi/self.period
    #     # old
    #     # # # lplc_corr                   = lplc_corr/(omega**2)*(3.**2)/2.
    #     # # # reason_n[(lplc_corr>lplcthresh)*(reason_n==0.)]  \
    #     # # #                             = 3
    #     # # # reason_n[(lplc_corr<-lplcthresh)*(reason_n==0.)]  \
    #     # # #                             = 3
    #     # new
    #     c0                          = 4.
    #     lplcthresh                  = (tempZarr*omega*omega/c0/c0)
    #     # if lplc_corr.shape != lplcthresh.shape or lplc_corr.shape != reason_n.shape:
    #     #     print '002: '
    #     #     print lplc_corr.shape
    #     #     print lplcthresh.shape
    #     #     print reason_n.shape
    #     #     raise ValueError('002')
    #     #
    #     reason_n[(lplc_corr>lplcthresh)*(reason_n==0.)]  \
    #                                 = 3
    #     reason_n[(lplc_corr<-lplcthresh)*(reason_n==0.)]  \
    #                                 = 3
    #     self.reason_n               = reason_n
    #     self.lplc                   = tfield.lplc.copy()
    #     self.mask                   = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
    #     tempmask                    = reason_n != 0
    #     self.mask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
    #                                 = tempmask
    #     return
    # 
    # def get_lplc_amp(self, fieldamp):
    #     """
    #     get the amplitude Laplacian correction terms from input field
    #     """
    #     if fieldamp.fieldtype!='amp':
    #         raise ValueError('No amplitude field!')
    #     # get data
    #     lplc                        = fieldamp.lplc
    #     # reason_n array from amplitude field
    #     reason_n_amp                = fieldamp.reason_n
    #     reason_n                    = self.reason_n.copy()
    #     dnlat                       = fieldamp.nlat_lplc - self.nlat_grad
    #     dnlon                       = fieldamp.nlon_lplc - self.nlon_grad
    #     # reason_n
    #     # 7: reason_n for amplitude field is not valid
    #     # 8: negative phase slowness after correction
    #     if dnlat == 0 and dnlon == 0:
    #         reason_n[reason_n_amp!=0]\
    #                                 = 7
    #         appV                    = self.appV.copy()
    #         self.reason_n_helm      = reason_n.copy()
    #     elif dnlat == 0 and dnlon != 0:
    #         (reason_n[:, dnlon:-dnlon])[reason_n_amp!=0]\
    #                                 = 7
    #         appV                    = self.appV[:, dnlon:-dnlon]
    #         self.reason_n_helm      = reason_n[:, dnlon:-dnlon]
    #     elif dnlat != 0 and dnlon == 0:
    #         (reason_n[dnlat:-dnlat, :])[reason_n_amp!=0]\
    #                                 = 7
    #         appV                    = self.appV[dnlat:-dnlat, :]
    #         self.reason_n_helm      = reason_n[dnlat:-dnlat, :]
    #     else:
    #         (reason_n[dnlat:-dnlat, dnlon:-dnlon])[reason_n_amp!=0]\
    #                                 = 7
    #         appV                    = self.appV[dnlat:-dnlat, dnlon:-dnlon]
    #         self.reason_n_helm      = reason_n[dnlat:-dnlat, dnlon:-dnlon]
    #     # compute amplitude Laplacian terms and corrected velocities
    #     omega                       = 2.*np.pi/self.period
    #     tamp                        = fieldamp.Zarr[fieldamp.nlat_lplc:-fieldamp.nlat_lplc, fieldamp.nlon_lplc:-fieldamp.nlon_lplc]
    #     self.lplc_amp               = fieldamp.lplc.copy()
    #     self.lplc_amp[tamp!=0.]     = self.lplc_amp[tamp!=0.]/(tamp[tamp!=0.]*omega**2)
    #     temp                        = 1./appV**2 - self.lplc_amp
    #     ind                         = temp<0.
    #     temp[ind]                   = 1./3**2.
    #     self.reason_n_helm[ind]     = 8
    #     self.corV                   = np.sqrt(1./temp)
    #     self.nlat_lplc              = fieldamp.nlat_lplc
    #     self.nlon_lplc              = fieldamp.nlon_lplc
    #     # mask array
    #     self.mask_helm              = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
    #     tempmask                    = self.reason_n_helm != 0
    #     self.mask_helm[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
    #                                 = tempmask
    #     return
    # 
    # #--------------------------------------------------
    # # functions for plotting
    # #--------------------------------------------------
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.figure()      
        minlon      = self.minlon
        maxlon      = self.maxlon
        minlat      = self.minlat
        maxlat      = self.maxlat
        
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                      urcrnrlon=maxlon, lat_ts=0, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,1,1,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,1,1,0])
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
            m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                        lat_1=minlat, lat_2=maxlat, lon_0=lon_centre-2., lat_0=lat_centre+2.4)
            # m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1., dashes=[2,2], labels=[1,1,0,1], fontsize=15)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,1,0], fontsize=15)
            # # # 
            # # # distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon) # distance is in m
            # # # distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-2, minlon) # distance is in m
            # # # m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
            # # #             lat_1=minlat, lat_2=maxlat, lon_0=lon_centre, lat_0=lat_centre+1.5)
            # # # m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            # # # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1, dashes=[2,2], labels=[0,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,0,0], fontsize=15)
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=1.)
        #################
        coasts = m.drawcoastlines(zorder=100,color= '0.9',linewidth=0.001)
        # m.drawstates(linewidth=1.)
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        return m
    
    def plot(self, datatype, title='', projection='lambert', cmap='surf', contour=False, showfig=True, vmin=None, vmax=None, stations=False, event=False):
        """Plot data with contour
        """
        m       = self._get_basemap(projection=projection)
        x, y    = m(self.lon2d, self.lat2d)
        datatype= datatype.lower()
        if event:
            try:
                evx, evy    = m(self.evlo, self.evla)
                m.plot(evx, evy, '^', markerfacecolor='yellow', markersize=15, markeredgecolor='k')
            except:
                pass
        if stations:
            try:
                stx, sty    = m(self.lonsIn, self.latsIn)
                m.plot(stx, sty, 'y^', markersize=6)
            except:
                pass
        try:
            stx, sty        = m(self.stalons, self.stalats)
            m.plot(stx, sty, 'b^', markersize=6)
        except:
            pass
        if datatype == 'v' or datatype == 'appv':
            data        = self.app_vel
            mdata       = ma.masked_array(data, mask=self.mask )
        elif datatype == 'corv':
            data        = self.corr_vel
            mdata       = ma.masked_array(data, mask=self.mask_helm )
        elif datatype == 'lplc':
            data        = self.lplc
            mdata       = ma.masked_array(data, mask=self.mask )
        elif datatype == 'z':
            data        = self.Zarr
            try:
                mdata   = ma.masked_array(data, mask=self.mask )
            except:
                mdata   = data.copy()
        elif datatype == 'reason_n':
            data        = self.reason_n
            mdata       = data.copy()
        if cmap == 'surf':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./surf.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
        im      = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb      = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.ax.tick_params(labelsize=40)
        if self.fieldtype=='Tph' or self.fieldtype=='Tgr':
            if datatype == 'z':
                cb.set_label('Travel time (sec)', fontsize=30, rotation=0)
            else:    
                cb.set_label('C (km/s)', fontsize=30, rotation=0)
        if self.fieldtype=='amp':
            cb.set_label('meters', fontsize=30, rotation=0)
        if contour:
            # levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 20)
            levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 60)
            m.contour(x, y, mdata, colors='k', levels=levels, linewidths=0.5)
        plt.suptitle(title, fontsize=50)
        if showfig:
            plt.show()
        return m
    