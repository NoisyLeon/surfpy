# -*- coding: utf-8 -*-
"""
Perform data interpolation/computation on the surface of the Earth

    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
import matplotlib
import multiprocessing
from functools import partial
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt

from subprocess import call
import obspy.geodetics
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from pyproj import Geod
import random
import copy
import colormaps
import math
import numba

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

class Field2d(object):
    """a class to analyze 2D spherical field data on Earth
    ===============================================================================================
    ::: parameters :::
    dlon, dlat              - grid interval
    Nlon, Nlat              - grid number in longitude, latitude
    lon, lat                - 1D arrays for grid locations
    lonArr, latArr          - 2D arrays for grid locations
    minlon/maxlon           - minimum/maximum longitude
    minlat/maxlat           - minimum/maximum latitude
    dlon_km/dlat_km         - 1D arrays for grid interval in km
    dlon_kmArr/dlon_kmArr   - 2D arrays for grid interval in km
    period                  - period
    evid                    - id for the event
    fieldtype               - field type (Tph, Tgr, Amp)
    Zarr                    - 2D data array (shape: Nlat, Nlon)
    evlo/evla               - longitude/latitue of the event
    nlon_grad/nlat_grad     - number of grids for cutting in gradient array
    nlon_lplc/nlat_lplc     - number of grids for cutting in Laplacian array
    -----------------------------------------------------------------------------------------------
    ::: derived parameters :::
    --- gradient related, shape: Nlat-2*nlat_grad, Nlon-2*nlon_grad
    grad[0]/grad[1]         - gradient arrays 
    proAngle                - propagation angle arrays 
    appV                    - apparent velocity
    reason_n                - index array indicating validity of data
    az/baz                  - azimuth/back-azimuth array
    diffaArr                - differences between propagation angle and azimuth
                                indicating off-great-circle deflection
    --- Laplacian related, shape: Nlat-2*nlat_lplc, Nlon-2*nlon_lplc
    lplc                    - Laplacian array
    reason_n                - index array indicating validity of data
    --- others
    mask                    - mask array, shape: Nlat, Nlon
    mask_helm               - mask for Helmholtz tomography, shape: Nlat, Nlon
    lplc_amp                - amplitude correction terms for phase speed
                                shape: Nlat-2*nlat_lplc, Nlon-2*nlon_lplc
    corV                    - corrected velocity, shape: Nlat-2*nlat_lplc, Nlon-2*nlon_lplc
    Nvalid_grd/Ntotal_grd   - number of valid/total grid points, validity means reason_n == 0.
    -----------------------------------------------------------------------------------------------
    Note: meshgrid's default indexing is 'xy', which means:
    lons, lats = np.meshgrid[lon, lat]
    in lons[i, j] or lats[i, j],  i->lat, j->lon
    ===============================================================================================
    """
    def __init__(self, minlon, maxlon, dlon, minlat, maxlat, dlat,
                 period=10., evlo=float('inf'), evla=float('inf'), fieldtype='Tph',\
                 evid='', nlat_grad=1, nlon_grad=1, nlat_lplc=2, nlon_lplc=2):
        self.dlon               = dlon
        self.dlat               = dlat
        self.Nlon               = int(round((maxlon-minlon)/dlon)+1)
        self.Nlat               = int(round((maxlat-minlat)/dlat)+1)
        self.lon                = np.arange(self.Nlon)*self.dlon+minlon
        self.lat                = np.arange(self.Nlat)*self.dlat+minlat
        self.lonArr, self.latArr= np.meshgrid(self.lon, self.lat)
        self.minlon             = minlon
        self.maxlon             = self.lon.max()
        self.minlat             = minlat
        self.maxlat             = self.lat.max()
        self._get_dlon_dlat_km()
        self.period             = period
        self.evid               = evid
        self.fieldtype          = fieldtype
        self.Zarr               = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.evlo               = evlo
        self.evla               = evla
        #-----------------------------------------------------------
        # parameters indicate edge cutting for gradient/lplc arrays
        #-----------------------------------------------------------
        self.nlon_grad          = nlon_grad
        self.nlat_grad          = nlat_grad
        self.nlon_lplc          = nlon_lplc
        self.nlat_lplc          = nlat_lplc
        return
    
    def copy(self):
        return copy.deepcopy(self)
    
    def  _get_dlon_dlat_km(self):
        """Get longitude and latitude interval in km
        """
        az, baz, dist_lon       = geodist.inv(np.zeros(self.lat.size), self.lat, np.ones(self.lat.size)*self.dlon, self.lat) 
        az, baz, dist_lat       = geodist.inv(np.zeros(self.lat.size), self.lat, np.zeros(self.lat.size), self.lat+self.dlat) 
        self.dlon_km            = dist_lon/1000.
        self.dlat_km            = dist_lat/1000.
        self.dlon_kmArr         = (np.tile(self.dlon_km, self.Nlon).reshape(self.Nlon, self.Nlat)).T
        self.dlat_kmArr         = (np.tile(self.dlat_km, self.Nlon).reshape(self.Nlon, self.Nlat)).T
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
        self.lonArrIn   = Inarray[:,0]
        self.latArrIn   = Inarray[:,1]
        self.ZarrIn     = Inarray[:,2]
        return
    
    def read_HD(self, fname):
        """
        """
        Inarray     = np.loadtxt(fname)
        data        = Inarray[:, 2].reshape((self.lonArr.shape[1]-2*self.nlon_grad, self.lonArr.shape[0]-2*self.nlat_grad))
        data        = data.T
        self.appV   = data.copy()
        self.mask   = np.zeros(self.lonArr.shape, dtype=bool)
        mask        = data == 0.
        self.mask[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad] \
                    = mask.copy()
        self.appV[np.logical_not(mask)]   \
                    = 1./self.appV[np.logical_not(mask)]
        return
    
    def synthetic_field(self, lat0, lon0, v=3.0):
        """generate synthetic field data
        """
        az, baz, distevent  = geodist.inv( np.ones(self.lonArrIn.size)*lon0, np.ones(self.lonArrIn.size)*lat0, self.lonArrIn, self.latArrIn)
        self.ZarrIn         = distevent/v/1000.
        return
    
    def read_ind(self, fname, zindex=2, dindex=None):
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
        self.lonArrIn   = Inarray[:,0]
        self.latArrIn   = Inarray[:,1]
        self.ZarrIn     = Inarray[:,zindex]*1e9
        if dindex is not None:
            darrIn      = Inarray[:,dindex]
            self.ZarrIn = darrIn/Inarray[:,zindex]
        return
    
    def read_array(self, lonArr, latArr, ZarrIn):
        """read field file
        """
        self.lonArrIn   = lonArr
        self.latArrIn   = latArr
        self.ZarrIn     = ZarrIn
        return
    
    def load_field(self, inField):
        """Load field data from an input object
        """
        self.lonArrIn   = inField.lonArr
        self.latArrIn   = inField.latArr
        self.ZarrIn     = inField.Zarr
        return
    
    def write(self, fname, fmt='npy'):
        """Save field file
        """
        OutArr      = np.append(self.lonArr, self.latArr)
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
            np.savez( outfname, self.appV, self.reason_n, self.proAngle, self.az, self.baz, self.Zarr,\
                     self.lplc_amp, self.corV, self.reason_n_helm, np.array([self.Ntotal_grd, self.Nvalid_grd]))
        else:
            np.savez( outfname, self.appV, self.reason_n, self.proAngle, self.az, self.baz, self.Zarr,\
                        np.array([self.Ntotal_grd, self.Nvalid_grd]))
        return
    
    def np2ma(self):
        """Convert all the data array to masked array according to reason_n array.
        """
        try:
            reason_n                = self.reason_n
        except:
            raise AttrictError('No reason_n array!')
        self.Zarr                   = ma.masked_array(self.Zarr, mask=np.zeros(reason_n.shape) )
        self.Zarr.mask[reason_n!=0] = 1 
        try:
            self.diffaArr                   = ma.masked_array(self.diffaArr, mask=np.zeros(reason_n.shape) )
            self.diffaArr.mask[reason_n!=0] = 1
        except:
            pass
        try:
            self.appV                       = ma.masked_array(self.appV, mask=np.zeros(reason_n.shape) )
            self.appV.mask[reason_n!=0]     = 1
        except:
            pass
        try:
            self.grad[0]                    = ma.masked_array(self.grad[0], mask=np.zeros(reason_n.shape) )
            self.grad[0].mask[reason_n!=0]  = 1
            self.grad[1]                    = ma.masked_array(self.grad[1], mask=np.zeros(reason_n.shape) )
            self.grad[1].mask[reason_n!=0]  = 1
        except:
            pass
        try:
            self.lplc                       = ma.masked_array(self.lplc, mask=np.zeros(reason_n.shape) )
            self.lplc.mask[reason_n!=0]     = 1
        except:
            print 'No Laplacian array!'
            pass
        return
    
    def ma2np(self):
        """Convert all the maksed data array to numpy array
        """
        self.Zarr           = ma.getdata(self.Zarr)
        try:
            self.diffaArr   = ma.getdata(self.diffaArr)
        except:
            pass
        try:
            self.appV       = ma.getdata(self.appV)
        except:
            pass
        try:
            self.lplc       = ma.getdata(self.lplc)
        except:
            pass
        return
    
    def add_noise(self, sigma=0.5):
        """Add Gaussian noise with standard deviation = sigma to the input data
            used for synthetic test
        """
        for i in xrange(self.ZarrIn.size):
            self.ZarrIn[i]  = self.ZarrIn[i] + random.gauss(0, sigma)
        return
    
    def cut_edge(self, nlon, nlat):
        """Cut edge
        =======================================================================================
        ::: input parameters :::
        nlon, nlon  - number of edge point in longitude/latitude to be cutted
        =======================================================================================
        """
        self.Nlon               = self.Nlon-2*nlon
        self.Nlat               = self.Nlat-2*nlat
        self.minlon             = self.minlon + nlon*self.dlon
        self.maxlon             = self.maxlon - nlon*self.dlon
        self.minlat             = self.minlat + nlat*self.dlat
        self.maxlat             = self.maxlat - nlat*self.dlat
        self.lon                = np.arange(self.Nlon)*self.dlon+self.minlon
        self.lat                = np.arange(self.Nlat)*self.dlat+self.minlat
        self.lonArr,self.latArr = np.meshgrid(self.lon, self.lat)
        self.Zarr               = self.Zarr[nlat:-nlat, nlon:-nlon]
        try:
            self.reason_n       = self.reason_n[nlat:-nlat, nlon:-nlon]
        except:
            pass
        self._get_dlon_dlat_km()
        return
    
    #--------------------------------------------------
    # functions for interpolation/gradient/Laplacian 
    #--------------------------------------------------
    
    def interp_surface(self, workingdir, outfname, tension=0.0):
        """interpolate input data to grid point with gmt surface command
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
        OutArr      = np.append(self.lonArrIn, self.latArrIn)
        OutArr      = np.append(OutArr, self.ZarrIn)
        OutArr      = OutArr.reshape(3, self.lonArrIn.size)
        OutArr      = OutArr.T
        np.savetxt(workingdir+'/'+outfname, OutArr, fmt='%g')
        fnameHD     = workingdir+'/'+outfname+'.HD'
        tempGMT     = workingdir+'/'+outfname+'_GMT.sh'
        grdfile     = workingdir+'/'+outfname+'.grd'
        with open(tempGMT,'wb') as f:
            REG     = '-R'+str(self.minlon)+'/'+str(self.maxlon)+'/'+str(self.minlat)+'/'+str(self.maxlat)
            f.writelines('gmt gmtset MAP_FRAME_TYPE fancy \n')
            if self.dlon == self.dlat:
                f.writelines('gmt surface %s -T%g -G%s -I%g %s \n' %( workingdir+'/'+outfname, tension, grdfile, self.dlon, REG ))
            else:
                f.writelines('gmt surface %s -T%g -G%s -I%g/%g %s \n' %( workingdir+'/'+outfname, tension, grdfile, self.dlon, self.dlat, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile, REG, fnameHD ))
        call(['bash', tempGMT])
        os.remove(grdfile)
        os.remove(tempGMT)
        inArr       = np.loadtxt(fnameHD)
        ZarrIn      = inArr[:, 2]
        self.Zarr   = (ZarrIn.reshape(self.Nlat, self.Nlon))[::-1, :]
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
        OutArr      = np.append(self.lonArrIn, self.latArrIn)
        OutArr      = np.append(OutArr, self.ZarrIn)
        OutArr      = OutArr.reshape(3, self.lonArrIn.size)
        OutArr      = OutArr.T
        np.savetxt(workingdir+'/'+outfname, OutArr, fmt='%g')
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
                f.writelines('gmt surface %s -T%g -G%s -I%g %s \n' %( workingdir+'/'+outfname, tension, grdfile, self.dlon, REG ))
            else:
                f.writelines('gmt surface %s -T%g -G%s -I%g/%g %s \n' %( workingdir+'/'+outfname, tension, grdfile, self.dlon, self.dlat, REG ))
            f.writelines('gmt grdfilter %s -D4 -Fg%g -G%s %s \n' %( grdfile, width, outgrd, REG))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( outgrd, REG, fnameHD ))
        call(['bash', tempGMT])
        os.remove(grdfile)
        os.remove(outgrd)
        os.remove(tempGMT)
        inArr       = np.loadtxt(fnameHD)
        ZarrIn      = inArr[:, 2]
        self.Zarr   = (ZarrIn.reshape(self.Nlat, self.Nlon))[::-1, :]
        return
    
    def gradient(self, method='diff', edge_order=1, order=2):
        """Compute gradient of the field
        =============================================================================================
        ::: input parameters :::
        edge_order  - edge_order : {1, 2}, optional, only has effect when method='diff'
                        Gradient is calculated using Nth order accurate differences at the boundaries
        method      - method: 'diff' : use numpy.gradient; 'convolve': use convolution
        order       - order of finite difference scheme, only has effect when method='convolve'
        ::: note :::
        gradient arrays are of shape Nlat-2*nlat_grad, Nlon-2*nlon_grad
        =============================================================================================
        """
        Zarr            = self.Zarr
        if method=='diff':
            # self.dlat_kmArr : dx here in numpy gradient since Zarr is Z[ilat, ilon]
            self.grad   = np.gradient( self.Zarr, self.dlat_kmArr, self.dlon_kmArr, edge_order=edge_order)
            self.grad[0]= self.grad[0][self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
            self.grad[1]= self.grad[1][self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        elif method == 'convolve':
            dlat_km     = self.dlat_kmArr
            dlon_km     = self.dlon_kmArr
            if order==2:
                diff_lon= convolve(Zarr, lon_diff_weight_2)/dlon_km
                diff_lat= convolve(Zarr, lat_diff_weight_2)/dlat_km
            elif order==4:
                diff_lon= convolve(Zarr, lon_diff_weight_4)/dlon_km
                diff_lat= convolve(Zarr, lat_diff_weight_4)/dlat_km
            elif order==6:
                diff_lon= convolve(Zarr, lon_diff_weight_6)/dlon_km
                diff_lat= convolve(Zarr, lat_diff_weight_6)/dlat_km
            self.grad   = []
            self.grad.append(diff_lat[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad])
            self.grad.append(diff_lon[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad])
        # propagation direction angle
        self.proAngle   = np.arctan2(self.grad[0], self.grad[1])/np.pi*180.
        return
    
    def Laplacian(self, method='green', order=4, verbose=False):
        """Compute Laplacian of the field
        =============================================================================================================
        ::: input parameters :::
        method      - method: 'diff'    : use central finite difference scheme, similar to convolve with order =2
                              'convolve': use convolution
                              'green'   : use Green's theorem( 2D Gauss's theorem )
        order       - order of finite difference scheme, only has effect when method='convolve'
        ::: note :::
        Laplacian arrays are of shape Nlat-2*nlat_lplc, Nlon-2*nlon_lplc
        =============================================================================================================
        """
        Zarr                = self.Zarr
        if method == 'diff':
            dlat_km         = self.dlat_kmArr[1:-1, 1:-1]
            dlon_km         = self.dlon_kmArr[1:-1, 1:-1]
            Zarr_latp       = Zarr[2:, 1:-1]
            Zarr_latn       = Zarr[:-2, 1:-1]
            Zarr_lonp       = Zarr[1:-1, 2:]
            Zarr_lonn       = Zarr[1:-1, :-2]
            Zarr            = Zarr[1:-1, 1:-1]
            lplc            = (Zarr_latp+Zarr_latn-2*Zarr) / (dlat_km**2) + (Zarr_lonp+Zarr_lonn-2*Zarr) / (dlon_km**2)
            dnlat           = self.nlat_lplc - 1
            dnlon           = self.nlon_lplc - 1
            if dnlat == 0 and dnlon == 0:
                self.lplc   = lplc
            elif dnlat == 0 and dnlon != 0:
                self.lplc   = lplc[:, dnlon:-dnlon]
            elif dnlat != 0 and dnlon == 0:
                self.lplc   = lplc[dnlat:-dnlat, :]
            else:
                self.lplc   = lplc[dnlat:-dnlat, dnlon:-dnlon]
        elif method == 'convolve':
            dlat_km         = self.dlat_kmArr
            dlon_km         = self.dlon_kmArr
            if order==2:
                diff2_lon   = convolve(Zarr, lon_diff2_weight_2)/dlon_km/dlon_km
                diff2_lat   = convolve(Zarr, lat_diff2_weight_2)/dlat_km/dlat_km
            elif order==4:
                diff2_lon   = convolve(Zarr, lon_diff2_weight_4)/dlon_km/dlon_km
                diff2_lat   = convolve(Zarr, lat_diff2_weight_4)/dlat_km/dlat_km
            elif order==6:
                diff2_lon   = convolve(Zarr, lon_diff2_weight_6)/dlon_km/dlon_km
                diff2_lat   = convolve(Zarr, lat_diff2_weight_6)/dlat_km/dlat_km
            self.lplc       = diff2_lon+diff2_lat
            self.lplc       = self.lplc[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        elif method=='green':
            #---------------
            # gradient arrays
            #---------------
            try:
                grad_y          = self.grad[0]
                grad_x          = self.grad[1]
            except:
                self.gradient('diff')
                grad_y          = self.grad[0]
                grad_x          = self.grad[1]
            grad_xp             = grad_x[1:-1, 2:]
            grad_xn             = grad_x[1:-1, :-2]
            grad_yp             = grad_y[2:, 1:-1]
            grad_yn             = grad_y[:-2, 1:-1]
            dlat_km             = self.dlat_kmArr[self.nlat_grad+1:-self.nlat_grad-1, self.nlon_grad+1:-self.nlon_grad-1]
            dlon_km             = self.dlon_kmArr[self.nlat_grad+1:-self.nlat_grad-1, self.nlon_grad+1:-self.nlon_grad-1]
            #------------------
            # Green's theorem
            #------------------
            loopsum             = (grad_xp - grad_xn)*dlat_km + (grad_yp - grad_yn)*dlon_km
            area                = dlat_km*dlon_km
            lplc                = loopsum/area
            #-----------------------------------------------
            # cut edges according to nlat_lplc, nlon_lplc
            #-----------------------------------------------
            dnlat               = self.nlat_lplc - self.nlat_grad - 1
            if dnlat < 0:
                self.nlat_lplc  = self.nlat_grad + 1
            dnlon               = self.nlon_lplc - self.nlon_grad - 1
            if dnlon < 0:
                self.nlon_lplc  = self.nlon_grad + 1
            if dnlat == 0 and dnlon == 0:
                self.lplc       = lplc
            elif dnlat == 0 and dnlon != 0:
                self.lplc       = lplc[:, dnlon:-dnlon]
            elif dnlat != 0 and dnlon == 0:
                self.lplc       = lplc[dnlat:-dnlat, :]
            else:
                self.lplc       = lplc[dnlat:-dnlat, dnlon:-dnlon]
        if verbose:
            print 'max lplc:',self.lplc.max(), 'min lplc:',self.lplc.min()
        return
    
    def get_appV(self):
        """Get the apparent velocity from gradient
        """
        slowness                = np.sqrt ( self.grad[0] ** 2 + self.grad[1] ** 2)
        slowness[slowness==0]   = 0.3
        self.appV               = 1./slowness
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
        # Compute Laplacian
        self.Laplacian(method='green')
        tfield      = self.copy()
        tfield.cut_edge(nlon=self.nlon_lplc, nlat=self.nlat_lplc)
        #--------------------
        # quality control
        #--------------------
        LonLst      = tfield.lonArr.reshape(tfield.lonArr.size)
        LatLst      = tfield.latArr.reshape(tfield.latArr.size)
        TLst        = tfield.Zarr.reshape(tfield.Zarr.size)
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
        with open(tempGMT,'wb') as f:
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
        # Compute Laplacian
        self.Laplacian(method='green')
        tfield      = self.copy()
        tfield.cut_edge(nlon=self.nlon_lplc, nlat=self.nlat_lplc)
        threshold   = threshold*2./(3.**2)
        #--------------------
        # quality control
        #--------------------
        LonLst      = tfield.lonArr.reshape(tfield.lonArr.size)
        LatLst      = tfield.latArr.reshape(tfield.latArr.size)
        ampLst      = tfield.Zarr.reshape(tfield.Zarr.size)
        # # # lplc        = self.lplc.reshape(self.lplc.size)
        # # # lplc_corr   = lplc.copy()
        # # # lplc_corr   = self.lplc.copy()
        # # # lplc_corr[ampLst!=0.]\
        # # #             = lplc[ampLst!=0.]/ampLst[ampLst!=0.]
        # # # lplc_corr[ampLst==0.]\
        # # #             = 0.
        lplc_corr   = self.lplc.copy()
        #
        if lplc_corr.shape != tfield.Zarr.shape:
            raise ValueError('001: '+tfield.evid)
        #
        lplc_corr[tfield.Zarr==0.]\
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
        with open(tempGMT,'wb') as f:
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
        
    def eikonal_operator(self, workingdir, inpfx='', nearneighbor=True, cdist=150., lplcthresh=0.005, lplcnearneighbor=False):
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
        # Read data
        # v1: data that passes check_curvature criterion
        # v1HD and v1HD02: interpolated v1 data with tension = 0. and 0.2
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
        diffArr     = fieldv1HD-fieldv1HD02
        fieldArr    = fieldv1HD*((diffArr<2.)*(diffArr>-2.)) 
        fieldArr    = (fieldArr.reshape(self.Nlat, self.Nlon))[::-1, :]
        #-------------------------------------------------------------------------------------
        # reason_n array
        #   0: accepted point
        #   1: data point the has large difference between v1HD and v1HD02
        #   2: data point that does not have near neighbor points at all E/W/N/S directions
        #   3: slowness is too large/small
        #   4: near a zero field data point
        #   5: epicentral distance is too small
        #   6: large curvature
        #-------------------------------------------------------------------------------------
        reason_n    = np.ones(fieldArr.size, dtype=np.int32)
        reason_n1   = np.int32(reason_n*(diffArr>2.))
        reason_n2   = np.int32(reason_n*(diffArr<-2.))
        reason_n    = reason_n1+reason_n2
        reason_n    = (reason_n.reshape(self.Nlat, self.Nlon))[::-1,:]
        #-------------------------------------------------------------------------------------------------------
        # check each data point if there are close-by four stations located at E/W/N/S directions respectively
        #-------------------------------------------------------------------------------------------------------
        if nearneighbor:
            for ilat in range(self.Nlat):
                for ilon in range(self.Nlon):
                    if reason_n[ilat, ilon]==1:
                        continue
                    lon         = self.lon[ilon]
                    lat         = self.lat[ilat]
                    dlon_km     = self.dlon_km[ilat]
                    dlat_km     = self.dlat_km[ilat]
                    difflon     = abs(self.lonArrIn-lon)/self.dlon*dlon_km
                    difflat     = abs(self.latArrIn-lat)/self.dlat*dlat_km
                    index       = np.where((difflon<cdist)*(difflat<cdist))[0]
                    marker_EN   = np.zeros((2,2), dtype=np.bool)
                    marker_nn   = 4
                    tflag       = False
                    for iv1 in index:
                        lon2    = self.lonArrIn[iv1]
                        lat2    = self.latArrIn[iv1]
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
                            marker_nn   = marker_nn-1
                            if marker_nn==0:
                                tflag   = True
                                break
                            marker_EN[marker_E, marker_N]   = True
                    if not tflag:
                        fieldArr[ilat, ilon]    = 0
                        reason_n[ilat, ilon]    = 2
        # Start to Compute Gradient
        tfield                      = self.copy()
        tfield.Zarr                 = fieldArr
        tfield.gradient('diff')
        # if one field point has zero value, reason_n for four near neighbor points will all be set to 4
        tempZarr                    = tfield.Zarr[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        index0                      = np.where(tempZarr==0.)
        ilatArr                     = index0[0] + 1
        ilonArr                     = index0[1] + 1
        reason_n[ilatArr+1, ilonArr]= 4
        reason_n[ilatArr-1, ilonArr]= 4
        reason_n[ilatArr, ilonArr+1]= 4
        reason_n[ilatArr, ilonArr-1]= 4
        # reduce size of reason_n to be the same shape as gradient arrays
        reason_n                    = reason_n[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        # if slowness is too large/small, reason_n will be set to 3
        slowness                    = np.sqrt(tfield.grad[0]**2 + tfield.grad[1]**2)
        if self.fieldtype=='Tph' or self.fieldtype=='Tgr':
            reason_n[(slowness>0.5)*(reason_n==0)]  = 3
            reason_n[(slowness<0.2)*(reason_n==0)]  = 3
        #-------------------------------------
        # computing propagation deflection
        #-------------------------------------
        indexvalid                              = np.where(reason_n==0)
        diffaArr                                = np.zeros(reason_n.shape, dtype = np.float64)
        latinArr                                = self.lat[indexvalid[0] + self.nlat_grad]
        loninArr                                = self.lon[indexvalid[1] + self.nlon_grad]
        evloArr                                 = np.ones(loninArr.size, dtype=np.float64)*evlo
        evlaArr                                 = np.ones(latinArr.size, dtype=np.float64)*evla
        az, baz, distevent                      = geodist.inv(loninArr, latinArr, evloArr, evlaArr) # loninArr/latinArr are initial points
        distevent                               = distevent/1000.        
        az                                      = az + 180.
        az                                      = 90.-az
        baz                                     = 90.-baz
        az[az>180.]                             = az[az>180.] - 360.
        az[az<-180.]                            = az[az<-180.] + 360.
        baz[baz>180.]                           = baz[baz>180.] - 360.
        baz[baz<-180.]                          = baz[baz<-180.] + 360.
        # az azimuth receiver -> source 
        diffaArr[indexvalid[0], indexvalid[1]]  = tfield.proAngle[indexvalid[0], indexvalid[1]] - az
        self.gradient('diff')
        self.az                                 = np.zeros(self.proAngle.shape, dtype=np.float64)
        self.az[indexvalid[0], indexvalid[1]]   = az
        self.baz                                = np.zeros(self.proAngle.shape, dtype=np.float64)
        self.baz[indexvalid[0], indexvalid[1]]  = baz
        # three wavelength criteria
        # if epicentral distance is too small, reason_n will be set to 5, and diffaArr will be 0.
        dist_per                                = 4.*self.period*3.
        tempArr                                 = diffaArr[indexvalid[0], indexvalid[1]]
        tempArr[distevent<dist_per]             = 0.
        diffaArr[indexvalid[0], indexvalid[1]]  = tempArr
        diffaArr[diffaArr>180.]                 = diffaArr[diffaArr>180.]-360.
        diffaArr[diffaArr<-180.]                = diffaArr[diffaArr<-180.]+360.
        tempArr                                 = reason_n[indexvalid[0], indexvalid[1]]
        tempArr[distevent<dist_per]             = 5
        reason_n[indexvalid[0], indexvalid[1]]  = tempArr
        #------------------------------------------------------------------------
        # final check of curvature, discard grid points with large curvature
        #------------------------------------------------------------------------
        self.Laplacian(method='green')
        dnlat                                   = self.nlat_lplc - self.nlat_grad
        dnlon                                   = self.nlon_lplc - self.nlon_grad
        tempind                                 = (self.lplc > lplcthresh) + (self.lplc < -lplcthresh)
        if dnlat == 0 and dnlon == 0:
            reason_n[tempind]                   = 6
        elif dnlat == 0 and dnlon != 0:
            (reason_n[:, dnlon:-dnlon])[tempind]= 6
        elif dnlat != 0 and dnlon == 0:
            (reason_n[dnlat:-dnlat, :])[tempind]= 6
        else:
            (reason_n[dnlat:-dnlat, dnlon:-dnlon])[tempind]\
                                                = 6
        # near neighbor discard for large curvature
        if lplcnearneighbor:
            indexlplc                               = np.where(reason_n==6.)
            ilatArr                                 = indexlplc[0] 
            ilonArr                                 = indexlplc[1]
            reason_n_temp                           = np.zeros(self.lonArr.shape)
            reason_n_temp[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad] \
                                                    = reason_n.copy()
            reason_n_temp[ilatArr+1, ilonArr]       = 6
            reason_n_temp[ilatArr-1, ilonArr]       = 6
            reason_n_temp[ilatArr, ilonArr+1]       = 6
            reason_n_temp[ilatArr, ilonArr-1]       = 6
            reason_n                                = reason_n_temp[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        # store final data
        self.diffaArr                           = diffaArr
        self.grad                               = tfield.grad
        self.get_appV()
        ###
        # cgg on site
        # # # reason_n[reason_n==2.] = 0.
        # # # reason_n[reason_n==1.] = 0.
        # # # reason_n[reason_n==3.] = 0.
        ###
        self.reason_n                           = reason_n
        self.mask                               = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
        tempmask                                = reason_n != 0
        self.mask[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]\
                                                = tempmask
        # added 04/05/2018
        self.Nvalid_grd                         = (np.where(reason_n==0.)[0]).size
        self.Ntotal_grd                         = reason_n.size
        return
    
    def helmholtz_operator(self, workingdir, inpfx='', lplcthresh=0.2):
        """
        Generate amplitude Laplacian maps for helmholtz tomography
        Two interpolated amplitude file with different tension will be used for quality control.
        =====================================================================================================================
        ::: input parameters :::
        workingdir      - working directory
        inpfx           - prefix for input files
        lplcthresh      - threshold value for Laplacian
        =====================================================================================================================
        """
        # Read data,
        # v1: data that pass check_curvature criterion
        # v1HD and v1HD02: interpolated v1 data with tension = 0. and 0.2
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
        # new
        fieldArr    = fieldv1HD*(abs(diffArr)<abs(0.01*fieldv1HD))
        #old
        #med_amp     = np.median(self.Zarr)
        #fieldArr    = fieldv1HD*((diffArr<0.01*med_amp)*(diffArr>-0.01*med_amp))
        fieldArr    = (fieldArr.reshape(self.Nlat, self.Nlon))[::-1, :]
        # reason_n 
        #   0: accepted point
        #   1: data point the has large difference between v1HD and v1HD02
        #   3: large curvature
        #   4: near a zero field data point
        reason_n    = np.ones(fieldArr.size, dtype=np.int32)
        # new
        reason_n1   = np.int32(reason_n*(diffArr>abs(0.01*fieldv1HD)))
        reason_n2   = np.int32(reason_n*(diffArr<-abs(0.01*fieldv1HD)))
        # old
        # reason_n1   = np.int32(reason_n*(diffArr>0.01*med_amp))
        # reason_n2   = np.int32(reason_n*(diffArr<-0.01*med_amp))
        reason_n    = reason_n1 + reason_n2
        reason_n    = (reason_n.reshape(self.Nlat, self.Nlon))[::-1,:]
        #-------------------------------
        # Start to compute Laplacian
        #-------------------------------
        tfield                      = self.copy()
        tfield.Zarr                 = fieldArr
        tfield.Laplacian(method='green') ## schemes
        # if one field point has zero value, reason_n for four near neighbor points will all be set to 4
        tempZarr                    = tfield.Zarr[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        index0                      = np.where(tempZarr==0.)
        ilatArr                     = index0[0] + 1
        ilonArr                     = index0[1] + 1
        reason_n[ilatArr+1, ilonArr]= 4
        reason_n[ilatArr-1, ilonArr]= 4
        reason_n[ilatArr, ilonArr+1]= 4
        reason_n[ilatArr, ilonArr-1]= 4
        # reduce size of reason_n to be the same shape as Laplacian arrays
        reason_n                    = reason_n[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # if Laplacian is too large/small, reason_n will be set to 3
        lplc_corr                   = tfield.lplc.copy()
        # lplc_corr[tempZarr!=0.]     = \
        #             tfield.lplc[tempZarr!=0.]/tempZarr[tempZarr!=0.]
        lplc_corr[tempZarr==0.]     = 0.
        omega                       = 2.*np.pi/self.period
        # old
        # # # lplc_corr                   = lplc_corr/(omega**2)*(3.**2)/2.
        # # # reason_n[(lplc_corr>lplcthresh)*(reason_n==0.)]  \
        # # #                             = 3
        # # # reason_n[(lplc_corr<-lplcthresh)*(reason_n==0.)]  \
        # # #                             = 3
        # new
        c0                          = 4.
        lplcthresh                  = (tempZarr*omega*omega/c0/c0)
        # if lplc_corr.shape != lplcthresh.shape or lplc_corr.shape != reason_n.shape:
        #     print '002: '
        #     print lplc_corr.shape
        #     print lplcthresh.shape
        #     print reason_n.shape
        #     raise ValueError('002')
        #
        reason_n[(lplc_corr>lplcthresh)*(reason_n==0.)]  \
                                    = 3
        reason_n[(lplc_corr<-lplcthresh)*(reason_n==0.)]  \
                                    = 3
        self.reason_n               = reason_n
        self.lplc                   = tfield.lplc.copy()
        self.mask                   = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
        tempmask                    = reason_n != 0
        self.mask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                                    = tempmask
        return
    
    def get_lplc_amp(self, fieldamp):
        """
        get the amplitude Laplacian correction terms from input field
        """
        if fieldamp.fieldtype!='amp':
            raise ValueError('No amplitude field!')
        # get data
        lplc                        = fieldamp.lplc
        # reason_n array from amplitude field
        reason_n_amp                = fieldamp.reason_n
        reason_n                    = self.reason_n.copy()
        dnlat                       = fieldamp.nlat_lplc - self.nlat_grad
        dnlon                       = fieldamp.nlon_lplc - self.nlon_grad
        # reason_n
        # 7: reason_n for amplitude field is not valid
        # 8: negative phase slowness after correction
        if dnlat == 0 and dnlon == 0:
            reason_n[reason_n_amp!=0]\
                                    = 7
            appV                    = self.appV.copy()
            self.reason_n_helm      = reason_n.copy()
        elif dnlat == 0 and dnlon != 0:
            (reason_n[:, dnlon:-dnlon])[reason_n_amp!=0]\
                                    = 7
            appV                    = self.appV[:, dnlon:-dnlon]
            self.reason_n_helm      = reason_n[:, dnlon:-dnlon]
        elif dnlat != 0 and dnlon == 0:
            (reason_n[dnlat:-dnlat, :])[reason_n_amp!=0]\
                                    = 7
            appV                    = self.appV[dnlat:-dnlat, :]
            self.reason_n_helm      = reason_n[dnlat:-dnlat, :]
        else:
            (reason_n[dnlat:-dnlat, dnlon:-dnlon])[reason_n_amp!=0]\
                                    = 7
            appV                    = self.appV[dnlat:-dnlat, dnlon:-dnlon]
            self.reason_n_helm      = reason_n[dnlat:-dnlat, dnlon:-dnlon]
        # compute amplitude Laplacian terms and corrected velocities
        omega                       = 2.*np.pi/self.period
        tamp                        = fieldamp.Zarr[fieldamp.nlat_lplc:-fieldamp.nlat_lplc, fieldamp.nlon_lplc:-fieldamp.nlon_lplc]
        self.lplc_amp               = fieldamp.lplc.copy()
        self.lplc_amp[tamp!=0.]     = self.lplc_amp[tamp!=0.]/(tamp[tamp!=0.]*omega**2)
        temp                        = 1./appV**2 - self.lplc_amp
        ind                         = temp<0.
        temp[ind]                   = 1./3**2.
        self.reason_n_helm[ind]     = 8
        self.corV                   = np.sqrt(1./temp)
        self.nlat_lplc              = fieldamp.nlat_lplc
        self.nlon_lplc              = fieldamp.nlon_lplc
        # mask array
        self.mask_helm              = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
        tempmask                    = self.reason_n_helm != 0
        self.mask_helm[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                                    = tempmask
        return
    
    #--------------------------------------------------
    # functions for plotting
    #--------------------------------------------------
    
    def _get_basemap_default(self, projection='lambert', geopolygons=None, resolution='i'):
        """
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        lat_centre  = (self.maxlat+self.minlat)/2.0
        lon_centre  = (self.maxlon+self.minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=self.minlat-5., urcrnrlat=self.maxlat+5., llcrnrlon=self.minlon-5.,
                        urcrnrlon=self.maxlon+5., lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        
        elif projection=='regional_ortho':
            m1      = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution=resolution,\
                        llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon,
                                self.minlat, self.maxlon) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon,
                                self.maxlat+2., self.minlon) # distance is in m
            m               = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                                lat_1=self.minlat, lat_2=self.maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.figure()      
        
        minlon      = 188 - 360.
        maxlon      = 238. - 360.
        minlat      = 52.
        maxlat      = 72.
        
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection=='merc':
            minlon      = -165.
            maxlon      = -135.
            minlat      = 56.
            maxlat      = 70.
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                      urcrnrlon=maxlon, lat_ts=0, resolution=resolution)
            # m.drawparallels(np.arange(minlat,maxlat,dlat), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(minlon,maxlon,dlon), labels=[1,0,0,1])
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
        # m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=1.)
        #################
        coasts = m.drawcoastlines(zorder=100,color= '0.9',linewidth=0.001)
        
        # Exact the paths from coasts
        coasts_paths = coasts.get_paths()
        
        # In order to see which paths you want to retain or discard you'll need to plot them one
        # at a time noting those that you want etc.
        poly_stop = 10
        for ipoly in xrange(len(coasts_paths)):
            print ipoly
            if ipoly > poly_stop:
                break
            r = coasts_paths[ipoly]
            # Convert into lon/lat vertices
            polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in
                                r.iter_segments(simplify=False)]
            px = [polygon_vertices[i][0] for i in xrange(len(polygon_vertices))]
            py = [polygon_vertices[i][1] for i in xrange(len(polygon_vertices))]
            m.plot(px,py,'k-',linewidth=2.)
        ######################
        # m.drawstates(linewidth=1.)
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    
    
    def plot(self, datatype, title='', projection='lambert', cmap='cv', contour=False, geopolygons=None, showfig=True, vmin=None, vmax=None, stations=False, event=False):
        """Plot data with contour
        """
        m       = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y    = m(self.lonArr, self.latArr)
        datatype= datatype.lower()
        if event:
            try:
                evx, evy    = m(self.evlo, self.evla)
                m.plot(evx, evy, '^', markerfacecolor='yellow', markersize=15, markeredgecolor='k')
            except:
                pass
        if stations:
            try:
                stx, sty    = m(self.lonArrIn, self.latArrIn)
                m.plot(stx, sty, 'y^', markersize=6)
            except:
                pass
        try:
            stx, sty        = m(self.stalons, self.stalats)
            m.plot(stx, sty, 'b^', markersize=6)
        except:
            pass
        if datatype == 'v' or datatype == 'appv':
            data        = np.zeros(self.lonArr.shape)
            data[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]\
                        = self.appV
            mdata       = ma.masked_array(data, mask=self.mask )
        elif datatype == 'corv':
            data        = np.zeros(self.lonArr.shape)
            data[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                        = self.corV
            mdata       = ma.masked_array(data, mask=self.mask_helm )
        elif datatype == 'z':
            data        = self.Zarr
            # # # data[1:-1, 1:-1]        = self.az
            try:
                mdata   = ma.masked_array(data, mask=self.mask )
            except:
                mdata   = data.copy()
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
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
            cb.set_label('nm', fontsize=30, rotation=0)
        
        if contour:
            # levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 20)
            levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 60)
            m.contour(x, y, mdata, colors='k', levels=levels, linewidths=0.5)
        plt.suptitle(title, fontsize=50)
        

        if showfig:
            plt.show()
        return m
    
    #--------------------------------------------------
    # functions to be deprecated
    #--------------------------------------------------
    
    def get_fit(self, lon=None, lat=None, dlon = 0.2, dlat = 0.2, plotfig=False):
        if lon is None and lat is None:
            raise ValueError('longitude or lattitude should be specified!')
            return
        if lon is None:
            ind_in      = (self.latArrIn < lat + dlat)*((self.latArrIn > lat - dlat))
            datain      = self.ZarrIn[ind_in]
            lonin       = self.lonArrIn[ind_in]
            #
            ind_interp  = (self.latArr < lat + self.dlat/2.)*((self.latArr > lat - self.dlat/2.))
            data        = self.Zarr[ind_interp]
            lon         = self.lonArr[ind_interp]
            if plotfig:
                plt.plot(lonin, datain, 'o', ms=5)
                plt.plot(lon, data, '-', lw=3)
                # plt.yabel()
                plt.xlabel('longitude(deg)', fontsize=30)
                plt.show()
            return lonin, datain, lon, data
        if lat is None:
            ind_in      = (self.lonArrIn < lon + dlon)*((self.lonArrIn > lon - dlon))
            datain      = self.ZarrIn[ind_in]
            latin       = self.latArrIn[ind_in]
            #
            ind_interp  = (self.lonArr < lon + self.dlon/2.)*((self.lonArr > lon - self.dlon/2.))
            data        = self.Zarr[ind_interp]
            lat         = self.latArr[ind_interp]
            if plotfig:
                plt.plot(latin, datain, 'o', ms=5)
                plt.plot(lat, data, '-', lw=3)
                # plt.yabel()
                plt.xlabel('latitude(deg)', fontsize=30)
                plt.show()
            return latin, datain, lat, data
        
    def get_line_lplc(self, lon=None, lat=None, dlon = 0.2, dlat = 0.2, plotfig=False):
        if lon is None and lat is None:
            raise ValueError('longitude or lattitude should be specified!')
            return
        latArr      = self.latArr[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        lonArr      = self.lonArr[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        if lon is None:
            #
            ind_interp  = (latArr < lat + self.dlat/2.)*(latArr > lat - self.dlat/2.)
            data        = self.lplc[ind_interp]
            lon         = lonArr[ind_interp]
            if plotfig:
                plt.plot(lonin, datain, 'o', ms=5)
                plt.plot(lon, data, '-', lw=3)
                # plt.yabel()
                plt.xlabel('longitude(deg)', fontsize=30)
                plt.show()
            return lon, data
        if lat is None:
            #
            ind_interp  = (lonArr < lon + self.dlon/2.)*(lonArr > lon - self.dlon/2.)
            data        = self.lplc[ind_interp]
            lat         = latArr[ind_interp]
            if plotfig:
                plt.plot(latin, datain, 'o', ms=5)
                plt.plot(lat, data, '-', lw=3)
                # plt.yabel()
                plt.xlabel('latitude(deg)', fontsize=30)
                plt.show()
            return lat, data
        
    def coarse_data(self, dlon=1., dlat=1.):
        minlon          = self.minlon - (self.minlon % 1.)
        minlat          = self.minlat - (self.minlat % 1.)
        maxlon          = self.maxlon - (self.maxlon % 1.)
        maxlat          = self.maxlat - (self.maxlat % 1.)
        Nlon            = int(round((maxlon-minlon)/dlon)+1)
        Nlat            = int(round((maxlat-minlat)/dlat)+1)
        lons            = np.arange(Nlon)*dlon + minlon
        lats            = np.arange(Nlat)*dlat+minlat
        lonArr, latArr  = np.meshgrid(lons, lats)
        L               = lonArr.size
        self.lonArrIn   = lonArr.reshape(L)
        self.latArrIn   = latArr.reshape(L)
        self.ZarrIn     = np.zeros(L, dtype=np.float64)
        for i in range(L):
            lon             = self.lonArrIn[i]
            lat             = self.latArrIn[i]
            ind             = np.where((self.lonArr==lon)*(self.latArr==lat))
            self.ZarrIn[i]  = self.Zarr[ind[0], ind[1]]
            
    # # # 
    # # # def plot_field(self, projection='lambert', contour=True, geopolygons=None, showfig=True, vmin=None, vmax=None, stations=False, event=False):
    # # #     """Plot data with contour
    # # #     """
    # # #     m=self._get_basemap(projection=projection, geopolygons=geopolygons)
    # # #     x, y=m(self.lonArr, self.latArr)
    # # #     if event:
    # # #         try:
    # # #             evx, evy=m(self.evlo, self.evla)
    # # #             m.plot(evx, evy, 'yo', markersize=10)
    # # #         except: pass
    # # #     if stations:
    # # #         try:
    # # #             stx, sty=m(self.lonArrIn, self.latArrIn)
    # # #             m.plot(stx, sty, 'y^', markersize=6)
    # # #         except: pass
    # # #     try:
    # # #         stx, sty = m(self.stalons, self.stalats)
    # # #         m.plot(stx, sty, 'b^', markersize=6)
    # # #     except: pass
    # # #     im=m.pcolormesh(x, y, self.Zarr, cmap='gist_ncar_r', shading='gouraud', vmin=vmin, vmax=vmax)
    # # #     cb = m.colorbar(im, "bottom", size="3%", pad='2%')
    # # #     cb.ax.tick_params(labelsize=10)
    # # #     if self.fieldtype=='Tph' or self.fieldtype=='Tgr':
    # # #         cb.set_label('sec', fontsize=12, rotation=0)
    # # #     if self.fieldtype=='Amp':
    # # #         cb.set_label('nm', fontsize=12, rotation=0)
    # # #     if contour:
    # # #         # levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 20)
    # # #         levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 60)
    # # #         m.contour(x, y, self.Zarr, colors='k', levels=levels, linewidths=0.5)
    # # #     if showfig:
    # # #         plt.show()
    # # #     return m
    # # # 
    def plot_lplc(self, cmap='seismic', projection='lambert', contour=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
        """Plot data with contour
        """
        plt.figure()
        m       = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y    = m(self.lonArr, self.latArr)
        # x       = x[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # y       = y[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # cmap =discrete_cmap(int(vmax-vmin)/2+1, 'seismic')
        data        = np.zeros(self.lonArr.shape)
        data[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                        = self.lplc
        tempmask    = np.ones(self.lonArr.shape, dtype=np.bool)
        tempmask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                    = self.mask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        mdata       = ma.masked_array(data, mask=tempmask)
        im          =m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # cb      = m.colorbar()
        # cb.ax.tick_params(labelsize=15)
        cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=20)
        cb.set_label('Travel time Laplacian (s/km^2)', fontsize=25, rotation=0)
        # # levels  = np.linspace(self.lplc.min(), self.lplc.max(), 100)
        # # if contour:
        # #     plt.contour(x, y, self.lplc, colors='k', levels=levels)
        if showfig:
            plt.show()
        return
    
    
    def plot_diff_lplc(self, cmap='seismic', projection='lambert', contour=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
        """Plot data with contour
        """
        plt.figure()
        m       = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y    = m(self.lonArr, self.latArr)
        # x       = x[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # y       = y[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # cmap =discrete_cmap(int(vmax-vmin)/2+1, 'seismic')
        data        = np.zeros(self.lonArr.shape)
        data[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                        = self.lplc_diff
        tempmask    = np.ones(self.lonArr.shape, dtype=np.bool)
        tempmask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                    = self.mask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        mdata       = ma.masked_array(data, mask=tempmask)
        im          =m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # cb      = m.colorbar()
        # cb.ax.tick_params(labelsize=15)
        cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=20)
        cb.set_label('Travel time Laplacian (s/km^2)', fontsize=25, rotation=0)
        # # levels  = np.linspace(self.lplc.min(), self.lplc.max(), 100)
        # # if contour:
        # #     plt.contour(x, y, self.lplc, colors='k', levels=levels)
        if showfig:
            plt.show()
        return
    
    def plot_theo_lplc(self, cmap='seismic', projection='lambert', contour=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
        """Plot data with contour
        """
        plt.figure()
        m       = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y    = m(self.lonArr, self.latArr)
        # x       = x[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # y       = y[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # cmap =discrete_cmap(int(vmax-vmin)/2+1, 'seismic')
        data        = np.zeros(self.lonArr.shape)
        data[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                        = self.lplc_theo
        tempmask    = np.ones(self.lonArr.shape, dtype=np.bool)
        tempmask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                    = self.mask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        mdata       = ma.masked_array(data, mask=tempmask)
        im          =m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # cb      = m.colorbar()
        # cb.ax.tick_params(labelsize=15)
        cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=20)
        cb.set_label('Travel time Laplacian (s/km^2)', fontsize=25, rotation=0)
        # # levels  = np.linspace(self.lplc.min(), self.lplc.max(), 100)
        # # if contour:
        # #     plt.contour(x, y, self.lplc, colors='k', levels=levels)
        if showfig:
            plt.show()
        return
    
    def plot_gmt_lplc(self, cmap='seismic', projection='lambert', contour=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
        """Plot data with contour
        """
        plt.figure()
        m       = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y    = m(self.lonArr, self.latArr)
        # x       = x[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # y       = y[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # cmap =discrete_cmap(int(vmax-vmin)/2+1, 'seismic')
        data        = np.zeros(self.lonArr.shape)
        data[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                        = self.lplc_gmt
        tempmask    = np.ones(self.lonArr.shape, dtype=np.bool)
        tempmask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]\
                    = self.mask[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        mdata       = ma.masked_array(data, mask=tempmask)
        im          =m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # cb      = m.colorbar()
        # cb.ax.tick_params(labelsize=15)
        cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=20)
        cb.set_label('Travel time Laplacian (s/km^2)', fontsize=25, rotation=0)
        # # levels  = np.linspace(self.lplc.min(), self.lplc.max(), 100)
        # # if contour:
        # #     plt.contour(x, y, self.lplc, colors='k', levels=levels)
        if showfig:
            plt.show()
        return
    
    
                
                    
    

    

