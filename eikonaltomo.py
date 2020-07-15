# -*- coding: utf-8 -*-
"""
A python module to run surface wave eikonal/Helmholtz tomography
The code creates a datadbase based on hdf5 data format

:Dependencies:
    pyasdf and its dependencies
    GMT 5.x.x (for interpolation on Earth surface)
    numba
    numexpr
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
    
:References:
    Lin, Fan-Chi, Michael H. Ritzwoller, and Roel Snieder. "Eikonal tomography: surface wave tomography by phase front tracking across a regional broad-band seismic array."
        Geophysical Journal International 177.3 (2009): 1091-1110.
    Lin, Fan-Chi, and Michael H. Ritzwoller. "Helmholtz surface wave tomography for isotropic and azimuthally anisotropic structure."
        Geophysical Journal International 186.3 (2011): 1104-1120.
"""
import numpy as np
import numpy.ma as ma
import scipy.stats
import h5py, pyasdf
import os, shutil
from subprocess import call
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import matplotlib
import colormaps
import obspy
import field2d_earth
import numexpr
import warnings
from functools import partial
import multiprocessing
from numba import jit, float32, int32, boolean, float64, int64
import numba
from numba import njit, prange
import time
from scipy import optimize
from uncertainties import ufloat
import uncertainties.umath

# compiled function to get weight for each event and each grid point
@jit(float32[:,:,:](float32[:,:,:], float32[:,:,:]))
def _get_azi_weight(aziALL, validALL):
    Nevent, Nlon, Nlat  = aziALL.shape
    weightALL           = np.zeros((Nevent, Nlon, Nlat), dtype=np.float32)
    for ilon in xrange(Nlon):
        for ilat in xrange(Nlat):
            for i in xrange(Nevent):
                for j in xrange(Nevent):
                    delAzi                      = abs(aziALL[i, ilon, ilat] - aziALL[j, ilon, ilat])
                    if delAzi < 20. or delAzi > 340.:
                        weightALL[i, ilon, ilat]+= validALL[i, ilon, ilat]    
    return weightALL

@jit(boolean[:](boolean[:], float64[:], float64[:], float64[:], float64[:]))
def _get_mask_interp(mask_in, lons_in, lats_in, lons, lats):
    Nlat            = lats.size
    Nlon            = lons.size
    mask_out        = np.ones((Nlat, Nlon), dtype=np.bool)
    for i in range(Nlat):
        for j in range(Nlon):
            clat    = lats[i]
            clon    = lons[j]
            ind_lon = np.where(clon<=lons_in)[0][0]      
            ind_lat = np.where(clat<=lats_in)[0][0]
            if (clon - lons_in[ind_lon])< 0.001 and (clat - lats_in[ind_lat]) < 0.001:
                mask_out[i, j]      = mask_in[ind_lat, ind_lon]
                continue
            mask_out[i, j]          = mask_out[i, j]*mask_in[ind_lat, ind_lon]
            if ind_lat > 0:
                mask_out[i, j]      = mask_out[i, j]*mask_in[ind_lat-1, ind_lon]
                if ind_lon > 0:
                    mask_out[i, j]  = mask_out[i, j]*mask_in[ind_lat-1, ind_lon-1]
            if ind_lon > 0:
                mask_out[i, j]      = mask_out[i, j]*mask_in[ind_lat, ind_lon-1]
                if ind_lat > 0:
                    mask_out[i, j]  = mask_out[i, j]*mask_in[ind_lat-1, ind_lon-1]
    return mask_out

# compiled function to evaluate station distribution 
@jit(boolean(float64[:], float64[:], int32))
def _check_station_distribution_old(lons, lats, Nvalid_min):
    N       = lons.size
    Nvalid  = 0
    for i in range(N):
        lon1            = lons[i]
        lat1            = lats[i]
        NnearE          = 0
        NnearW          = 0
        NnearN          = 0
        NnearS          = 0
        for j in range(N):
            lon2        = lons[j]
            lat2        = lats[j]
            if i == j:
                continue
            if abs(lat1 - lat2) < 1.5:
                colat           = 90. - (lat1+lat2)/2.
                temp_R          = 6371. * np.sin(np.pi * colat/180.)
                dlon            = abs(lon1 - lon2)
                dist_lon        = temp_R * np.sin(dlon*np.pi/180.)
                if dist_lon < 150.:
                    if lon2 >= lon1:
                        NnearW  += 1
                    else:
                        NnearE  += 1
                    if lat2 >= lat1:
                        NnearN  += 1
                    else:
                        NnearS  += 1
        if NnearE > 0 and NnearW > 0 and NnearN > 0 and NnearS > 0:
            Nvalid  += 1
    if Nvalid >= Nvalid_min:
        return True
    else:
        return False
    
@jit(boolean(float64[:], float64[:], int32))
def _check_station_distribution(lons, lats, Nvalid_min):
    """check the station distribution
        Step 1. a station is counted as valid if there are at least four stations nearby
        Step 2. check if the number of valid stations is larger than Nvalid_min 
    """
    N       = lons.size
    Nvalid  = 0
    for i in range(N):
        lon1            = lons[i]
        lat1            = lats[i]
        Nnear           = 0
        for j in range(N):
            lon2        = lons[j]
            lat2        = lats[j]
            if i == j:
                continue
            if abs(lat1 - lat2) < 1.5:
                colat           = 90. - (lat1+lat2)/2.
                temp_R          = 6371. * np.sin(np.pi * colat/180.)
                dlon            = abs(lon1 - lon2)
                dist_lon        = temp_R * np.sin(dlon*np.pi/180.)
                if dist_lon < 150.:
                    Nnear       += 1
        if Nnear >= 4:
            Nvalid  += 1
    if Nvalid >= Nvalid_min:
        return True
    else:
        return False

def to_percent(y, position):
     # Ignore the passed in position. This has the effect of scaling the default
     # tick locations.
     s = '%.0f' %(100. * y)
     # The percent symbol needs escaping in latex
     if matplotlib.rcParams['text.usetex'] is True:
         return s + r'$\%$'
     else:
         return s + '%'

@jit(boolean[:](boolean[:], float64[:], float64[:], float64[:], float64[:]))
def _get_mask_interp(mask_in, lons_in, lats_in, lons, lats):
    Nlat            = lats.size
    Nlon            = lons.size
    mask_out        = np.ones((Nlat, Nlon), dtype=np.bool)
    for i in range(Nlat):
        for j in range(Nlon):
            clat    = lats[i]
            clon    = lons[j]
            ind_lon = np.where(clon<=lons_in)[0][0]      
            ind_lat = np.where(clat<=lats_in)[0][0]
            if (clon - lons_in[ind_lon])< 0.001 and (clat - lats_in[ind_lat]) < 0.001:
                mask_out[i, j]      = mask_in[ind_lat, ind_lon]
                continue
            mask_out[i, j]          = mask_out[i, j]*mask_in[ind_lat, ind_lon]
            if ind_lat > 0:
                mask_out[i, j]      = mask_out[i, j]*mask_in[ind_lat-1, ind_lon]
                if ind_lon > 0:
                    mask_out[i, j]  = mask_out[i, j]*mask_in[ind_lat-1, ind_lon-1]
            if ind_lon > 0:
                mask_out[i, j]      = mask_out[i, j]*mask_in[ind_lat, ind_lon-1]
                if ind_lat > 0:
                    mask_out[i, j]  = mask_out[i, j]*mask_in[ind_lat-1, ind_lon-1]
    return mask_out

def plot_fault_lines(mapobj, infname, lw=2, color='red'):
    with open(infname, 'rb') as fio:
        is_new  = False
        lonlst  = []
        latlst  = []
        for line in fio.readlines():
            if line.split()[0] == '>':
                x, y  = mapobj(lonlst, latlst)
                mapobj.plot(x, y,  lw = lw, color=color)
                # # # m.plot(xslb, yslb,  lw = 3, color='white')
                lonlst  = []
                latlst  = []
                continue
            lonlst.append(float(line.split()[0]))
            latlst.append(float(line.split()[1]))
        x, y  = mapobj(lonlst, latlst)
        mapobj.plot(x, y,  lw = lw, color=color)

@njit(numba.types.Tuple((float64[:, :, :], float64[:, :, :], float64[:, :, :], int64[:, :, :], int64[:, :]))\
     (int64, int64, float32, float32, int64, float64[:, :], float64[:, :, :], float64[:, :], float64[:, :, :], numba.boolean[:, :, :]))
def _anisotropic_stacking(gridx, gridy, maxazi, minazi, N_bin, Nmeasure, aziALL,\
        slowness_sumQC, slownessALL, index_outlier):
    Nevent, Nx, Ny  = aziALL.shape
    Nx_trim         = Nx - (gridx - 1)
    Ny_trim         = Ny - (gridy - 1)
    NmeasureAni     = np.zeros((Nx_trim, Ny_trim), dtype=np.int64) # for quality control
    # initialization of anisotropic parameters
    d_bin           = float((maxazi-minazi)/N_bin)
    # number of measurements in each bin
    histArr         = np.zeros((N_bin, Nx_trim, Ny_trim), dtype=np.int64)
    # slowness in each bin
    dslow_sum_ani   = np.zeros((N_bin, Nx_trim, Ny_trim))
    # slowness uncertainties for each bin
    dslow_un        = np.zeros((N_bin, Nx_trim, Ny_trim))
    # velocity uncertainties for each bin
    vel_un          = np.zeros((N_bin, Nx_trim, Ny_trim))
    #----------------------------------------------------------------------------------
    # Loop over azimuth bins to get slowness, velocity and number of measurements
    #----------------------------------------------------------------------------------
    for ibin in range(N_bin):
        sumNbin                     = np.zeros((Nx_trim, Ny_trim))
        # slowness arrays
        dslowbin                    = np.zeros((Nx_trim, Ny_trim))
        dslow_un_ibin               = np.zeros((Nx_trim, Ny_trim))
        dslow_mean                  = np.zeros((Nx_trim, Ny_trim))
        # velocity arrays
        velbin                      = np.zeros((Nx_trim, Ny_trim))
        vel_un_ibin                 = np.zeros((Nx_trim, Ny_trim))
        vel_mean                    = np.zeros((Nx_trim, Ny_trim))
        for ix in range(Nx_trim):
            for iy in range(Ny_trim):
                for ishift_x in range(gridx):
                    for ishift_y in range(gridy):
                        for iev in range(Nevent):
                            azi         = aziALL[iev, ix + ishift_x, iy + ishift_y]
                            ibin_temp   = np.floor((azi - minazi)/d_bin)
                            if ibin_temp != ibin:
                                continue
                            is_outlier  = index_outlier[iev, ix + ishift_x, iy + ishift_y]
                            if is_outlier:
                                continue
                            temp_dslow  = slownessALL[iev, ix + ishift_x, iy + ishift_y] - slowness_sumQC[ix + ishift_x, iy + ishift_y]
                            if slownessALL[iev, ix + ishift_x, iy + ishift_y] != 0.:
                                temp_vel= 1./slownessALL[iev, ix + ishift_x, iy + ishift_y]
                            else:
                                temp_vel= 0.
                            sumNbin[ix, iy]     += 1
                            dslowbin[ix, iy]    += temp_dslow
                            velbin[ix, iy]      += temp_vel
                            NmeasureAni[ix, iy] += 1 # 2019-06-06
                # end nested loop of grid shifting
                if sumNbin[ix, iy] >= 2:
                    vel_mean[ix, iy]            = velbin[ix, iy] / sumNbin[ix, iy]
                    dslow_mean[ix, iy]          = dslowbin[ix, iy] / sumNbin[ix, iy]
                else:
                    sumNbin[ix, iy]             = 0
        # compute uncertainties
        for ix in range(Nx_trim):
            for iy in range(Ny_trim):
                for ishift_x in range(gridx):
                    for ishift_y in range(gridy):
                        for iev in range(Nevent):
                            azi                     = aziALL[iev, ix + ishift_x, iy + ishift_y]
                            ibin_temp               = np.floor((azi - minazi)/d_bin)
                            if ibin_temp != ibin:
                                continue
                            is_outlier              = index_outlier[iev, ix + ishift_x, iy + ishift_y]
                            if is_outlier:
                                continue
                            if slownessALL[iev, ix + ishift_x, iy + ishift_y] != 0.:
                                temp_vel            = 1./slownessALL[iev, ix + ishift_x, iy + ishift_y]
                            else:
                                temp_vel            = 0.
                            temp_vel_mean           = vel_mean[ix, iy]
                            vel_un_ibin[ix, iy]     += (temp_vel - temp_vel_mean)**2
                            temp_dslow              = slownessALL[iev, ix + ishift_x, iy + ishift_y] - slowness_sumQC[ix + ishift_x, iy + ishift_y]
                            temp_dslow_mean         = dslow_mean[ix, iy]
                            dslow_un_ibin[ix, iy]   += (temp_dslow - temp_dslow_mean)**2
        for ix in range(Nx_trim):
            for iy in range(Ny_trim):
                if sumNbin[ix, iy] < 2:
                    continue
                vel_un_ibin[ix, iy]             = np.sqrt(vel_un_ibin[ix, iy]/(sumNbin[ix, iy] - 1)/sumNbin[ix, iy])
                vel_un[ibin, ix, iy]            = vel_un_ibin[ix, iy]
                dslow_un_ibin[ix, iy]           = np.sqrt(dslow_un_ibin[ix, iy]/(sumNbin[ix, iy] - 1)/sumNbin[ix, iy])
                dslow_un[ibin, ix, iy]          = dslow_un_ibin[ix, iy]
                histArr[ibin, ix, iy]           = sumNbin[ix, iy]
                dslow_sum_ani[ibin, ix, iy]     = dslow_mean[ix, iy]
    return dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni



@njit(numba.types.Tuple((float64[:, :, :], float64[:, :, :], float64[:, :, :], int64[:, :, :], int64[:, :]))\
     (int64, int64, float32, float32, int64, float64[:, :], float64[:, :, :], float64[:, :], float64[:, :, :], numba.boolean[:, :, :]), parallel=True)
def _anisotropic_stacking_parallel(gridx, gridy, maxazi, minazi, N_bin, Nmeasure, aziALL,\
        slowness_sumQC, slownessALL, index_outlier):
    Nevent, Nx, Ny  = aziALL.shape
    Nx_trim         = Nx - (gridx - 1)
    Ny_trim         = Ny - (gridy - 1)
    NmeasureAni     = np.zeros((Nx_trim, Ny_trim), dtype=np.int64) # for quality control
    # initialization of anisotropic parameters
    d_bin           = float((maxazi-minazi)/N_bin)
    # number of measurements in each bin
    histArr         = np.zeros((N_bin, Nx_trim, Ny_trim), dtype=np.int64)
    # slowness in each bin
    dslow_sum_ani   = np.zeros((N_bin, Nx_trim, Ny_trim))
    # slowness uncertainties for each bin
    dslow_un        = np.zeros((N_bin, Nx_trim, Ny_trim))
    # velocity uncertainties for each bin
    vel_un          = np.zeros((N_bin, Nx_trim, Ny_trim))
    #----------------------------------------------------------------------------------
    # Loop over azimuth bins to get slowness, velocity and number of measurements
    #----------------------------------------------------------------------------------
    for ibin in range(N_bin):
        sumNbin                     = np.zeros((Nx_trim, Ny_trim))
        # slowness arrays
        dslowbin                    = np.zeros((Nx_trim, Ny_trim))
        dslow_un_ibin               = np.zeros((Nx_trim, Ny_trim))
        dslow_mean                  = np.zeros((Nx_trim, Ny_trim))
        # velocity arrays
        velbin                      = np.zeros((Nx_trim, Ny_trim))
        vel_un_ibin                 = np.zeros((Nx_trim, Ny_trim))
        vel_mean                    = np.zeros((Nx_trim, Ny_trim))
        for ix in prange(Nx_trim):
            for iy in prange(Ny_trim):
                for ishift_x in range(gridx):
                    for ishift_y in range(gridy):
                        for iev in range(Nevent):
                            azi         = aziALL[iev, ix + ishift_x, iy + ishift_y]
                            ibin_temp   = np.floor((azi - minazi)/d_bin)
                            if ibin_temp != ibin:
                                continue
                            is_outlier  = index_outlier[iev, ix + ishift_x, iy + ishift_y]
                            if is_outlier:
                                continue
                            temp_dslow  = slownessALL[iev, ix + ishift_x, iy + ishift_y] - slowness_sumQC[ix + ishift_x, iy + ishift_y]
                            if slownessALL[iev, ix + ishift_x, iy + ishift_y] != 0.:
                                temp_vel= 1./slownessALL[iev, ix + ishift_x, iy + ishift_y]
                            else:
                                temp_vel= 0.
                            # changing values
                            sumNbin[ix, iy]     += 1
                            dslowbin[ix, iy]    += temp_dslow
                            velbin[ix, iy]      += temp_vel
                            NmeasureAni[ix, iy] += 1 # 2019-06-06
                # end nested loop of grid shifting
                if sumNbin[ix, iy] >= 2:
                    vel_mean[ix, iy]            = velbin[ix, iy] / sumNbin[ix, iy]
                    dslow_mean[ix, iy]          = dslowbin[ix, iy] / sumNbin[ix, iy]
                else:
                    sumNbin[ix, iy]             = 0
        # compute uncertainties
        for ix in prange(Nx_trim):
            for iy in prange(Ny_trim):
                for ishift_x in range(gridx):
                    for ishift_y in range(gridy):
                        for iev in range(Nevent):
                            azi                     = aziALL[iev, ix + ishift_x, iy + ishift_y]
                            ibin_temp               = np.floor((azi - minazi)/d_bin)
                            if ibin_temp != ibin:
                                continue
                            is_outlier              = index_outlier[iev, ix + ishift_x, iy + ishift_y]
                            if is_outlier:
                                continue
                            if slownessALL[iev, ix + ishift_x, iy + ishift_y] != 0.:
                                temp_vel            = 1./slownessALL[iev, ix + ishift_x, iy + ishift_y]
                            else:
                                temp_vel            = 0.
                            # changing values
                            temp_vel_mean           = vel_mean[ix, iy]
                            vel_un_ibin[ix, iy]     += (temp_vel - temp_vel_mean)**2
                            temp_dslow              = slownessALL[iev, ix + ishift_x, iy + ishift_y] - slowness_sumQC[ix + ishift_x, iy + ishift_y]
                            temp_dslow_mean         = dslow_mean[ix, iy]
                            dslow_un_ibin[ix, iy]   += (temp_dslow - temp_dslow_mean)**2
        for ix in prange(Nx_trim):
            for iy in prange(Ny_trim):
                if sumNbin[ix, iy] < 2:
                    continue
                vel_un_ibin[ix, iy]             = np.sqrt(vel_un_ibin[ix, iy]/(sumNbin[ix, iy] - 1)/sumNbin[ix, iy])
                vel_un[ibin, ix, iy]            = vel_un_ibin[ix, iy]
                dslow_un_ibin[ix, iy]           = np.sqrt(dslow_un_ibin[ix, iy]/(sumNbin[ix, iy] - 1)/sumNbin[ix, iy])
                dslow_un[ibin, ix, iy]          = dslow_un_ibin[ix, iy]
                histArr[ibin, ix, iy]           = sumNbin[ix, iy]
                dslow_sum_ani[ibin, ix, iy]     = dslow_mean[ix, iy]
    return dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni

# @jit(numba.types.Tuple((float32[:, :], float32[:, :]))(numba.boolean[:, :]), parallel=True)
def _get_gaps(indarr):
    Nbin, Nlat, Nlon    = indarr.shape
    gap1                = np.zeros((Nlat, Nlon), dtype=np.float32)
    gap2                = np.zeros((Nlat, Nlon), dtype=np.float32)
    for ilat in range(Nlat):
        for ilon in range(Nlon):
            index       = indarr[:, ilat, ilon]
            isfirst     = True
            isnew       = True
            temp_Ngap   = 0
            gaparr      = np.array([], dtype=np.float32)
            for i in range(Nbin):
                if index[i]:
                    if temp_Ngap> 0:
                        gaparr  = np.append(gaparr, temp_Ngap)
                    temp_Ngap   = 0
                    continue
                if isfirst:
                    firstind    = i
                    isfirst     = False
                temp_Ngap   += 1
            if temp_Ngap>0:
                gaparr  = np.append(gaparr, temp_Ngap)
            if gaparr.size == 0:
                gap1[ilat, ilon]   \
                        = 0
                gap2[ilat, ilon]   \
                        = 0
                continue
            if firstind == 0 and gaparr.size>1:
                outgap  = gaparr[1:-1]
                outgap  = np.append(outgap, gaparr[0]+gaparr[-1])
            else:
                outgap  = gaparr.copy()
            if outgap.size>=2:
                outgap      = np.sort(outgap)
                gap1[ilat, ilon]   \
                            = outgap[-1]
                gap2[ilat, ilon]   \
                        = outgap[-2]
            else:
                gap1[ilat, ilon]   \
                            = outgap[0]
                gap2[ilat, ilon]\
                        = 0
    return gap1, gap2
            

Npsi_glb    = 360
dpsi_glb    = 180./np.float64(Npsi_glb)
Namp_glb    = 500
ampmax_glb  = 5.
damp_glb    = ampmax_glb/np.float64(Namp_glb)

@njit(float64[:, :](float64[:], float64[:], float64[:], float64), parallel=True)
def _get_azi_misfit(obsdat, obssem, az_grd, A0):
    psiarr      = np.arange(Npsi_glb)*dpsi_glb
    amparr      = np.arange(Namp_glb)*damp_glb
    misfitarr   = np.zeros((Npsi_glb, Namp_glb), dtype=np.float64)
    Naz         = np.float64(az_grd.size)
    for ipsi in prange(Npsi_glb):
        for iamp in prange(Namp_glb):
            psi         = psiarr[ipsi]*np.pi/180.
            amp         = amparr[iamp]
            A2          = A0*amp/100.
            predat      = A0 + A2*np.cos(2.*(np.pi/180.*(az_grd+180.)-psi) )
            misfitarr[ipsi, iamp]\
                        = np.sqrt( ((predat - obsdat)**2 / obssem**2).sum()/ Naz )
    return misfitarr

def _get_azi_sem(obsdat, obssem, az_grd, A0):
    misfitarr   = _get_azi_misfit(obsdat, obssem, az_grd, A0)
    minmisfit   = misfitarr.min()
    thresh      = minmisfit + 0.5
    index       = misfitarr <= thresh
    psiarr      = np.arange(Npsi_glb)*dpsi_glb
    amparr      = np.arange(Namp_glb)*damp_glb
    psi2d, amp2d= np.meshgrid(psiarr, amparr, indexing='ij')
    psi_acc     = psi2d[index]
    amp_acc     = amp2d[index]
    unpsi       = scipy.stats.circstd(psi_acc, high=180., low=0.)
    unamp       = amp_acc.std()
    return unpsi, unamp

def _get_azi_sem_2(obsdat, obssem, az_grd, A0):
    misfitarr   = _get_azi_misfit(obsdat, obssem, az_grd, A0)
    minmisfit   = misfitarr.min()
    thresh      = minmisfit + 0.5
    index       = misfitarr <= thresh
    psiarr      = np.arange(Npsi_glb)*dpsi_glb
    amparr      = np.arange(Namp_glb)*damp_glb
    psi2d, amp2d= np.meshgrid(psiarr, amparr, indexing='ij')
    psi_acc     = psi2d[index]
    amp_acc     = amp2d[index]
    unpsi       = scipy.stats.circstd(psi_acc, high=180.01, low=0.)
    unamp       = amp_acc.std()
    return unpsi, unamp
                
def pre_azi_aniso(m, theta):
    return m[0] + m[1]*np.sin(2.*theta/180.*np.pi) + m[2]*np.cos(2.*theta/180.*np.pi)


def azi_errfunc(m, azarr, tvel, tsem):
    return (pre_azi_aniso(m, azarr) - tvel) / tsem
            
            
class data4stack(object):
    """data object storing array for eikonal stacking
    """
    def __init__(self, slownessALL, reason_nALL, aziALL, Nmeasure, Nevent, period, Nlon, Nlat, nlon_grad, nlat_grad):
        self.slownessALL    = slownessALL
        self.reason_nALL    = reason_nALL
        self.aziALL         = aziALL
        self.Nmeasure       = Nmeasure
        self.Nevent         = Nevent
        self.period         = period
        self.Nlon           = Nlon
        self.Nlat           = Nlat
        self.nlat_grad      = nlat_grad
        self.nlon_grad      = nlon_grad

class EikonalTomoDataSet(h5py.File):
    """
    Object for eikonal/Helmholtz tomography, builded upon hdf5 data file.
    """
    #==================================================
    # functions print the information of database
    #==================================================
    def print_attrs(self, print_to_screen=True):
        """
        Print the attrsbute information of the dataset.
        """
        outstr          = '================================= Surface wave eikonal/Helmholtz tomography database ==================================\n'
        try:
            outstr      += 'period(s):                              - '+str(self.attrs['period_array'])+'\n'
            outstr      += 'longitude range                         - '+str(self.attrs['minlon'])+' ~ '+str(self.attrs['maxlon'])+'\n'
            outstr      += 'longitude spacing/npts                  - '+str(self.attrs['dlon'])+'/'+str(self.attrs['Nlon'])+'\n'
            outstr      += 'nlon_grad/nlon_lplc                     - '+str(self.attrs['nlon_grad'])+'/'+str(self.attrs['nlon_lplc'])+'\n'
            outstr      += 'latitude range                          - '+str(self.attrs['minlat'])+' ~ '+str(self.attrs['maxlat'])+'\n'
            outstr      += 'latitude spacing/npts                   - '+str(self.attrs['dlat'])+'/'+str(self.attrs['Nlat'])+'\n'
            outstr      += 'nlat_grad/nlat_lplc                     - '+str(self.attrs['nlat_grad'])+'/'+str(self.attrs['nlat_lplc'])+'\n'
            per_arr     = self.attrs['period_array']
        except:
            print 'Empty Database!'
            return None
        if print_to_screen:
            print outstr
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
        print outstr
        return
    
    def set_input_parameters(self, minlon, maxlon, minlat, maxlat, pers=np.array([]), dlon=0.2, dlat=0.2, \
                             nlat_grad=1, nlon_grad=1, nlat_lplc=2, nlon_lplc=2, optimize_spacing=True):
        """
        Set input parameters for tomographic inversion.
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
        if pers.size==0:
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        self.attrs.create(name = 'period_array', data=pers, dtype='f')
        self.attrs.create(name = 'minlon', data=minlon, dtype='f')
        self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
        self.attrs.create(name = 'minlat', data=minlat, dtype='f')
        self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
        if optimize_spacing:
            ratio   = field2d_earth.determine_interval(minlat=minlat, maxlat=maxlat, dlon=dlon, dlat = dlat)
            print '----------------------------------------------------------'
            print 'Changed dlat from dlat =',dlat,'to dlat =',dlat/ratio
            print '----------------------------------------------------------'
            dlat    = dlat/ratio
        self.attrs.create(name = 'dlon', data=dlon)
        self.attrs.create(name = 'dlat', data=dlat)
        Nlon        = int((maxlon-minlon)/dlon+1)
        Nlat        = int((maxlat-minlat)/dlat+1)
        self.attrs.create(name = 'Nlon', data=Nlon)
        self.attrs.create(name = 'Nlat', data=Nlat)
        self.attrs.create(name = 'nlat_grad', data=nlat_grad)
        self.attrs.create(name = 'nlon_grad', data=nlon_grad)
        self.attrs.create(name = 'nlat_lplc', data=nlat_lplc)
        self.attrs.create(name = 'nlon_lplc', data=nlon_lplc)
        return
    
    def xcorr_eikonal(self, inasdffname, workingdir, fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', runid=0,\
                      deletetxt=True, verbose=False, cdist=150., mindp=10):
        """
        Compute gradient of travel time for cross-correlation data
        =================================================================================================================
        ::: input parameters :::
        inasdffname - input ASDF data file
        workingdir  - working directory
        fieldtype   - fieldtype (Tph or Tgr)
        channel     - channel for analysis (default = ZZ )
        data_type   - data type
                     (default='FieldDISPpmf2interp', aftan measurements with phase-matched filtering and jump correction)
        runid       - run id
        deletetxt   - delete output txt files in working directory
        cdist       - distance for nearneighbor station criteria
        mindp       - minnimum required number of data points for eikonal operator
        =================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        # create new eikonal group
        create_group        = False
        while (not create_group):
            try:
                group       = self.create_group( name = 'Eikonal_run_'+str(runid) )
                create_group= True
            except:
                runid       += 1
                continue
        group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        # input xcorr database
        inDbase             = pyasdf.ASDFDataSet(inasdffname)
        # get header 
        pers                = self.attrs['period_array']
        minlon              = self.attrs['minlon']
        maxlon              = self.attrs['maxlon']
        minlat              = self.attrs['minlat']
        maxlat              = self.attrs['maxlat']
        dlon                = self.attrs['dlon']
        dlat                = self.attrs['dlat']
        nlat_grad           = self.attrs['nlat_grad']
        nlon_grad           = self.attrs['nlon_grad']
        nlat_lplc           = self.attrs['nlat_lplc']
        nlon_lplc           = self.attrs['nlon_lplc']
        fdict               = { 'Tph': 2, 'Tgr': 3}
        evLst               = inDbase.waveforms.list()
        for per in pers:
            print '--- computing gradient for: '+str(per)+' sec'
            del_per         = per-int(per)
            if del_per==0.:
                persfx      = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                persfx      = str(int(per))+'sec'+dper.split('.')[1]
            working_per     = workingdir+'/'+str(per)+'sec'
            per_group       = group.create_group( name='%g_sec'%( per ) )
            for evid in evLst:
                netcode1, stacode1  = evid.split('.')
                try:
                    subdset         = inDbase.auxiliary_data[data_type][netcode1][stacode1][channel][persfx]
                except KeyError:
                    print ('No travel time field for: '+evid)
                    continue
                if verbose:
                    print ('Event: '+evid)
                lat1, elv1, lon1    = inDbase.waveforms[evid].coordinates.values()
                if lon1<0.:
                    lon1            += 360.
                dataArr             = subdset.data.value
                field2d             = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                        minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=lon1, evla=lat1, fieldtype=fieldtype, \
                                        nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
                Zarr                = dataArr[:, fdict[fieldtype]]
                # skip if not enough data points
                if Zarr.size <= mindp:
                    continue
                distArr             = dataArr[:, 5]
                field2d.read_array(lonArr=np.append(lon1, dataArr[:,0]), latArr=np.append(lat1, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                outfname            = evid+'_'+fieldtype+'_'+channel+'.lst'
                field2d.interp_surface(workingdir=working_per, outfname=outfname)
                field2d.check_curvature(workingdir=working_per, outpfx=evid+'_'+channel+'_')
                field2d.eikonal_operator(workingdir=working_per, inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)
                # save data to hdf5 dataset
                event_group         = per_group.create_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=lon1)
                event_group.attrs.create(name = 'evla', data=lat1)
                # added 04/05/2018
                event_group.attrs.create(name = 'Ntotal_grd', data=field2d.Ntotal_grd)
                event_group.attrs.create(name = 'Nvalid_grd', data=field2d.Nvalid_grd)
                #
                appVdset            = event_group.create_dataset(name='appV', data=field2d.appV)
                reason_ndset        = event_group.create_dataset(name='reason_n', data=field2d.reason_n)
                proAngledset        = event_group.create_dataset(name='proAngle', data=field2d.proAngle)
                azdset              = event_group.create_dataset(name='az', data=field2d.az)
                bazdset             = event_group.create_dataset(name='baz', data=field2d.baz)
                Tdset               = event_group.create_dataset(name='travelT', data=field2d.Zarr)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def xcorr_eikonal_raydbase(self, inh5fname, workingdir, rayruntype=0, rayrunid=0, period=None, crifactor=0.5, crilimit=10.,\
            fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', runid=0, deletetxt=True, verbose=False, cdist=150., mindp=10):
        """
        Compute gradient of travel time for cross-correlation data according to ray tomography database
        =================================================================================================================
        ::: input parameters :::
        inasdffname - input ASDF data file
        workingdir  - working directory
        fieldtype   - fieldtype (Tph or Tgr)
        channel     - channel for analysis
        data_type   - data type
                     (default='FieldDISPpmf2interp', aftan measurements with phase-matched filtering and jump correction)
        runid       - run id
        deletetxt   - delete output txt files in working directory
        cdist       - distance for nearneighbor station criteria
        mindp       - minnimum required number of data points for eikonal operator
        =================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        create_group        = False
        while (not create_group):
            try:
                group       = self.create_group( name = 'Eikonal_run_'+str(runid) )
                create_group= True
            except:
                runid       += 1
                continue
        group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        pers                = self.attrs['period_array']
        minlon              = self.attrs['minlon']
        maxlon              = self.attrs['maxlon']
        minlat              = self.attrs['minlat']
        maxlat              = self.attrs['maxlat']
        dlon                = self.attrs['dlon']
        dlat                = self.attrs['dlat']
        nlat_grad           = self.attrs['nlat_grad']
        nlon_grad           = self.attrs['nlon_grad']
        nlat_lplc           = self.attrs['nlat_lplc']
        nlon_lplc           = self.attrs['nlon_lplc']
        fdict               = { 'Tph': 2, 'Tgr': 3}
        if period is not None:
            pers            = np.array([period])
        inDbase             = h5py.File(inh5fname)
        rundict             = {0: 'smooth_run', 1: 'qc_run'}
        data_id             = rundict[rayruntype]+'_'+str(rayrunid)
        ingroup             = inDbase[data_id]
        ind_flag            = 1
        if rayruntype == 0:
            ind_flag        = 0
        else:
            if ingroup.attrs['isotropic']:
                ind_flag    = 0
        for per in pers:
            print 'Computing gradient for: '+str(per)+' sec'
            del_per         = per-int(per)
            if del_per==0.:
                persfx      = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                persfx      = str(int(per))+'sec'+dper.split('.')[1]
            working_per     = workingdir+'/'+str(per)+'sec'
            per_group       = group.create_group( name='%g_sec'%( per ) )
            # get data array from ray tomography database
            ray_per_id      = '%g_sec'%( per )
            data            = ingroup[ray_per_id+'/residual'].value
            res_tomo        = data[:,7+ind_flag]
            cri_res         = min(crifactor*per, crilimit)
            data            = data[ np.abs(res_tomo)<cri_res , :]
            evlo            = 0.
            evla            = 0.
            Ndata           = data.shape[0]
            i_event         = 0
            for i in range(Ndata):
                if evla != data[i, 1] or evlo != data[i, 2]:
                    # compute
                    if i != 0:
                        field2d.read_array(lonArr   = np.append(evlo, stlos), latArr=np.append(evla, stlas), ZarrIn=np.append(0., Zarr) )
                        outfname        = evid+'_'+fieldtype+'_'+channel+'.lst'
                        print outfname, Zarr.size, stlos.size, stlas.size
                        field2d.interp_surface(workingdir=working_per, outfname=outfname)
                        field2d.check_curvature(workingdir=working_per, outpfx=evid+'_'+channel+'_')
                        field2d.eikonal_operator(workingdir=working_per, inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)
                        # save data to hdf5 dataset
                        event_group     = per_group.create_group(name=evid)
                        event_group.attrs.create(name = 'evlo', data=evlo)
                        event_group.attrs.create(name = 'evla', data=evla)
                        # added 04/05/2018
                        event_group.attrs.create(name = 'Ntotal_grd', data=field2d.Ntotal_grd)
                        event_group.attrs.create(name = 'Nvalid_grd', data=field2d.Nvalid_grd)
                        #
                        appVdset        = event_group.create_dataset(name='appV', data=field2d.appV)
                        reason_ndset    = event_group.create_dataset(name='reason_n', data=field2d.reason_n)
                        proAngledset    = event_group.create_dataset(name='proAngle', data=field2d.proAngle)
                        azdset          = event_group.create_dataset(name='az', data=field2d.az)
                        bazdset         = event_group.create_dataset(name='baz', data=field2d.baz)
                        Tdset           = event_group.create_dataset(name='travelT', data=field2d.Zarr)
                    evla    = data[i, 1]
                    evlo    = data[i, 2]
                    field2d = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype=fieldtype, \
                                    nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
                    stlas   = np.array([])
                    stlos   = np.array([])
                    Zarr    = np.array([])
                    i_event += 1
                    evid    = 'ALK'+str(i_event)
                stla        = data[i, 3]
                stlo        = data[i, 4]
                stlas       = np.append(stlas, stla)
                stlos       = np.append(stlos, stlo)
                dist, az, baz   \
                            = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo)
                travelT     = dist/data[i, 5]/1000.
                Zarr        = np.append(Zarr, travelT)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def xcorr_eikonal_mp(self, inasdffname, workingdir, fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp',\
                runid=0, new_group=True, deletetxt=True, verbose=False, subsize=1000, nprocess=None, cdist=150., mindp=10, pers=None):
        """
        Compute gradient of travel time for cross-correlation data with multiprocessing
        =================================================================================================================
        ::: input parameters :::
        inh5fname   - input hdf5 data file
        workingdir  - working directory
        fieldtype   - fieldtype (Tph or Tgr)
        channel     - channel for analysis (default = ZZ )
        data_type   - data type
                     (default='FieldDISPpmf2interp', aftan measurements with phase-matched filtering and jump correction)
        runid       - run id
        deletetxt   - delete output txt files in working directory
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
        nprocess    - number of processes
        cdist       - distance for nearneighbor station criteria
        mindp       - minnimum required number of data points for eikonal operator
        =================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        if new_group:
            create_group        = False
            while (not create_group):
                try:
                    group       = self.create_group( name = 'Eikonal_run_'+str(runid) )
                    create_group= True
                except:
                    runid       += 1
                    continue
        else:
            group   = self.require_group( name = 'Eikonal_run_'+str(runid) )
        group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        # input xcorr database
        inDbase             = pyasdf.ASDFDataSet(inasdffname)
        # header information
        if isinstance(pers, np.ndarray):
            pers_dbase      = self.attrs['period_array']
            for per in pers:
                if not (per in pers_dbase):
                    raise KeyError('Period '+str(per)+' s in the database attributes')
        else:
            pers            = self.attrs['period_array']
        minlon              = self.attrs['minlon']
        maxlon              = self.attrs['maxlon']
        minlat              = self.attrs['minlat']
        maxlat              = self.attrs['maxlat']
        dlon                = self.attrs['dlon']
        dlat                = self.attrs['dlat']
        nlat_grad           = self.attrs['nlat_grad']
        nlon_grad           = self.attrs['nlon_grad']
        nlat_lplc           = self.attrs['nlat_lplc']
        nlon_lplc           = self.attrs['nlon_lplc']
        fdict               = { 'Tph': 2, 'Tgr': 3}
        evLst               = inDbase.waveforms.list()
        fieldLst            = []
        #------------------------
        # prepare data
        #------------------------
        for per in pers:
            print '--- preparing data for gradient computation of '+str(per)+' sec'
            del_per         = per-int(per)
            if del_per==0.:
                persfx      = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                persfx      = str(int(per))+'sec'+dper.split('.')[1]
            working_per     = workingdir+'/'+str(per)+'sec'
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            for evid in evLst:
                netcode1, stacode1  = evid.split('.')
                try:
                    subdset         = inDbase.auxiliary_data[data_type][netcode1][stacode1][channel][persfx]
                except KeyError:
                    if verbose:
                        print 'No travel time field for: '+evid
                    continue
                lat1, elv1, lon1    = inDbase.waveforms[evid].coordinates.values()
                if lon1<0.:
                    lon1            += 360.
                dataArr             = subdset.data.value
                field2d             = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon, minlat=minlat, maxlat=maxlat, dlat=dlat,
                                        period=per, evlo=lon1, evla=lat1, fieldtype=fieldtype, evid=evid, \
                                               nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
                Zarr                = dataArr[:, fdict[fieldtype]]
                if Zarr.size <= mindp:
                    continue
                distArr             = dataArr[:, 5]
                field2d.read_array(lonArr=np.append(lon1, dataArr[:,0]), latArr=np.append(lat1, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                fieldLst.append(field2d)
        #-----------------------------------------
        # Computing gradient with multiprocessing
        #-----------------------------------------
        if len(fieldLst) > subsize:
            Nsub                    = int(len(fieldLst)/subsize)
            for isub in range(Nsub):
                print '--- eikonal computation subset:', isub,'in',Nsub,'sets'
                cfieldLst           = fieldLst[isub*subsize:(isub+1)*subsize]
                EIKONAL             = partial(eikonal4mp, workingdir=workingdir, channel=channel, cdist=cdist)
                pool                = multiprocessing.Pool(processes=nprocess)
                pool.map(EIKONAL, cfieldLst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cfieldLst               = fieldLst[(isub+1)*subsize:]
            EIKONAL                 = partial(eikonal4mp, workingdir=workingdir, channel=channel, cdist=cdist)
            pool                    = multiprocessing.Pool(processes=nprocess)
            pool.map(EIKONAL, cfieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            print '--- eikonal computation, one set'
            EIKONAL                 = partial(eikonal4mp, workingdir=workingdir, channel=channel, cdist=cdist)
            pool                    = multiprocessing.Pool(processes=nprocess)
            pool.map(EIKONAL, fieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        #-----------------------------------------
        # Read data into hdf5 dataset
        #-----------------------------------------
        for per in pers:
            print '*** reading gradient data for: '+str(per)+' sec'
            working_per = workingdir+'/'+str(per)+'sec'
            per_group   = group.create_group( name='%g_sec'%( per ) )
            for evid in evLst:
                infname = working_per+'/'+evid+'_field2d.npz'
                if not os.path.isfile(infname):
                    if verbose:
                        print 'No data for:', evid
                    continue
                InArr           = np.load(infname)
                appV            = InArr['arr_0']
                reason_n        = InArr['arr_1']
                proAngle        = InArr['arr_2']
                az              = InArr['arr_3']
                baz             = InArr['arr_4']
                Zarr            = InArr['arr_5']
                Ngrd            = InArr['arr_6']
                lat1, elv1, lon1= inDbase.waveforms[evid].coordinates.values()
                # save data to hdf5 dataset
                event_group     = per_group.create_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=lon1)
                event_group.attrs.create(name = 'evla', data=lat1)
                # added 04/05/2018
                event_group.attrs.create(name = 'Ntotal_grd', data=Ngrd[0])
                event_group.attrs.create(name = 'Nvalid_grd', data=Ngrd[1])
                #
                appVdset        = event_group.create_dataset(name='appV', data=appV)
                reason_ndset    = event_group.create_dataset(name='reason_n', data=reason_n)
                proAngledset    = event_group.create_dataset(name='proAngle', data=proAngle)
                azdset          = event_group.create_dataset(name='az', data=az)
                bazdset         = event_group.create_dataset(name='baz', data=baz)
                Tdset           = event_group.create_dataset(name='travelT', data=Zarr)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def xcorr_eikonal_raydbase_mp(self, inh5fname, workingdir, rayruntype=0, rayrunid=0, period=None, crifactor=0.5, crilimit=10.,\
            fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', runid=0, new_group=True, \
                deletetxt=True, verbose=False, subsize=1000, nprocess=None, cdist=150., mindp=10, pers=None):
        """
        Compute gradient of travel time for cross-correlation data according to ray tomography database,
            with multiprocessing
        =================================================================================================================
        ::: input parameters :::
        inh5fname   - input hdf5 data file
        workingdir  - working directory
        fieldtype   - fieldtype (Tph or Tgr)
        channel     - channel for analysis
        data_type   - data type
                     (default='FieldDISPpmf2interp', aftan measurements with phase-matched filtering and jump correction)
        runid       - run id
        deletetxt   - delete output txt files in working directory
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
        nprocess    - number of processes
        cdist       - distance for nearneighbor station criteria
        mindp       - minnimum required number of data points for eikonal operator
        =================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        create_group        = False
        while (not create_group):
            try:
                group       = self.create_group( name = 'Eikonal_run_'+str(runid) )
                create_group= True
            except:
                runid       += 1
                continue
        group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        pers                = self.attrs['period_array']
        minlon              = self.attrs['minlon']
        maxlon              = self.attrs['maxlon']
        minlat              = self.attrs['minlat']
        maxlat              = self.attrs['maxlat']
        dlon                = self.attrs['dlon']
        dlat                = self.attrs['dlat']
        nlat_grad           = self.attrs['nlat_grad']
        nlon_grad           = self.attrs['nlon_grad']
        nlat_lplc           = self.attrs['nlat_lplc']
        nlon_lplc           = self.attrs['nlon_lplc']
        fdict               = { 'Tph': 2, 'Tgr': 3}
        if period is not None:
            pers            = np.array([period])
        inDbase             = h5py.File(inh5fname)
        rundict             = {0: 'smooth_run', 1: 'qc_run'}
        data_id             = rundict[rayruntype]+'_'+str(rayrunid)
        ingroup             = inDbase[data_id]
        ind_flag            = 1
        if rayruntype == 0:
            ind_flag        = 0
        else:
            if ingroup.attrs['isotropic']:
                ind_flag    = 0
        fieldLst            = []
        evlst               = []
        for per in pers:
            print 'Computing gradient for: '+str(per)+' sec'
            del_per         = per-int(per)
            if del_per==0.:
                persfx      = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                persfx      = str(int(per))+'sec'+dper.split('.')[1]
            working_per     = workingdir+'/'+str(per)+'sec'
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            # get data array from ray tomography database
            ray_per_id      = '%g_sec'%( per )
            data            = ingroup[ray_per_id+'/residual'].value
            res_tomo        = data[:,7+ind_flag]
            cri_res         = min(crifactor*per, crilimit)
            
            # # # data            = data[ np.abs(res_tomo)<cri_res , :]
            ind             = (res_tomo > -cri_res)*(res_tomo < 20.)
            data            = data[ind, :]
            
            evlo            = 0.
            evla            = 0.
            Ndata           = data.shape[0]
            i_event         = 0
            for i in range(Ndata):
                if evla != data[i, 1] or evlo != data[i, 2]:
                    # compute
                    if i != 0:
                        field2d.read_array(lonArr   = np.append(evlo, stlos), latArr=np.append(evla, stlas), ZarrIn=np.append(0., Zarr))
                        fieldLst.append(field2d)
                    evla    = data[i, 1]
                    evlo    = data[i, 2]
                    evlst.append(np.array([evla, evlo]))
                    i_event += 1
                    evid    = 'ALK'+str(i_event)
                    field2d = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon, 
                                minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evid=evid, evlo=evlo, evla=evla, fieldtype=fieldtype, \
                                    nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
                    stlas   = np.array([])
                    stlos   = np.array([])
                    Zarr    = np.array([])
                stla        = data[i, 3]
                stlo        = data[i, 4]
                stlas       = np.append(stlas, stla)
                stlos       = np.append(stlos, stlo)
                dist, az, baz   \
                            = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo)
                travelT     = dist/data[i, 5]/1000.
                Zarr        = np.append(Zarr, travelT)
                
        #-----------------------------------------
        # Computing gradient with multiprocessing
        #-----------------------------------------
        if len(fieldLst) > subsize:
            Nsub                    = int(len(fieldLst)/subsize)
            for isub in range(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cfieldLst           = fieldLst[isub*subsize:(isub+1)*subsize]
                EIKONAL             = partial(eikonal4mp, workingdir=workingdir, channel=channel, cdist=cdist)
                pool                = multiprocessing.Pool(processes=nprocess)
                pool.map(EIKONAL, cfieldLst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cfieldLst               = fieldLst[(isub+1)*subsize:]
            EIKONAL                 = partial(eikonal4mp, workingdir=workingdir, channel=channel, cdist=cdist)
            pool                    = multiprocessing.Pool(processes=nprocess)
            pool.map(EIKONAL, cfieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            print 'Computing eikonal tomography'
            EIKONAL                 = partial(eikonal4mp, workingdir=workingdir, channel=channel, cdist=cdist)
            pool                    = multiprocessing.Pool(processes=nprocess)
            pool.map(EIKONAL, fieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        #-----------------------------------------
        # Read data into hdf5 dataset
        #-----------------------------------------
        iper    = 0
        for per in pers:
            print 'Reading gradient data for: '+str(per)+' sec'
            working_per         = workingdir+'/'+str(per)+'sec'
            per_group           = group.create_group( name='%g_sec'%( per ) )
            for ievent in range(len(evlst)):
                evid            = 'ALK'+str(ievent)
                infname         = working_per+'/'+evid+'_field2d.npz'
                if not os.path.isfile(infname):
                    if verbose:
                        print 'No data for:', evid
                    continue
                InArr           = np.load(infname)
                appV            = InArr['arr_0']
                reason_n        = InArr['arr_1']
                proAngle        = InArr['arr_2']
                az              = InArr['arr_3']
                baz             = InArr['arr_4']
                Zarr            = InArr['arr_5']
                Ngrd            = InArr['arr_6']
                evla            = evlst[ievent][0]
                evlo            = evlst[ievent][1]
                # save data to hdf5 dataset
                event_group     = per_group.create_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=evlo)
                event_group.attrs.create(name = 'evla', data=evla)
                # added 04/05/2018
                event_group.attrs.create(name = 'Ntotal_grd', data=Ngrd[0])
                event_group.attrs.create(name = 'Nvalid_grd', data=Ngrd[1])
                #
                appVdset        = event_group.create_dataset(name='appV', data=appV)
                reason_ndset    = event_group.create_dataset(name='reason_n', data=reason_n)
                proAngledset    = event_group.create_dataset(name='proAngle', data=proAngle)
                azdset          = event_group.create_dataset(name='az', data=az)
                bazdset         = event_group.create_dataset(name='baz', data=baz)
                Tdset           = event_group.create_dataset(name='travelT', data=Zarr)
            iper    += 1
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def xcorr_eikonal_mp_lowmem(self, inasdffname, workingdir, fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', runid=0,
                deletetxt=True, verbose=False, subsize=1000, nprocess=None, cdist=150., mindp=10):
        """
        Low memory version of xcorr_eikonal_mp
        """
        pers_dbase      = self.attrs['period_array']
        for per in pers_dbase:
            print '=== eikonal tomography for T = '+str(per)+' sec'
            pers        = np.array([per])
            self.xcorr_eikonal_mp(inasdffname=inasdffname, workingdir=workingdir, fieldtype=fieldtype, channel=channel,\
                    data_type=data_type, runid=runid, new_group=False, deletetxt=deletetxt, verbose=verbose, subsize=subsize, nprocess=nprocess,\
                        cdist=cdist, mindp=mindp, pers=pers)
        return
        
    def quake_eikonal(self, inasdffname, workingdir, fieldtype='Tph', channel='Z', data_type='FieldDISPpmf2interp',
                pre_qual_ctrl=True, btime_qc=None, etime_qc = None, runid=0, merge=True, deletetxt=False,
                    verbose=True, amplplc=False, cdist=150., mindp=50, Tmin=-1., Tmax=999.):
        """
        Compute gradient of travel time for earthquake data
        =======================================================================================================================
        ::: input parameters :::
        inasdffname     - input ASDF data file
        workingdir      - working directory
        fieldtype       - fieldtype (Tph or Tgr)
        channel         - channel for analysis
        data_type       - data type
                            default='FieldDISPpmf2interp': 
                                interpolated aftan measurements with phase-matched filtering and jump correction
        pre_qual_ctrl   - perform pre-tomography quality control or not
        btime_qc        - begin time for quality control
        etime_qc        - end time for quality control
        runid           - run id
        deletetxt       - delete output txt files in working directory
        amplplc         - compute amplitude Laplacian term or not
        cdist           - distance for nearneighbor station criteria
        mindp           - minimum required number of data points for eikonal operator
        =======================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        # merge data to existing group or not
        if merge:
            try:
                group           = self.create_group( name = 'Eikonal_run_'+str(runid) )
                group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
            except ValueError:
                print 'Merging Eikonal run id: ',runid
                group           = self.require_group( name = 'Eikonal_run_'+str(runid) )
        else:
            create_group        = False
            while (not create_group):
                try:
                    group       = self.create_group( name = 'Eikonal_run_'+str(runid) )
                    create_group= True
                except:
                    runid       +=1
                    continue
            group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        pers                = self.attrs['period_array']
        minlon              = self.attrs['minlon']
        maxlon              = self.attrs['maxlon']
        minlat              = self.attrs['minlat']
        maxlat              = self.attrs['maxlat']
        dlon                = self.attrs['dlon']
        dlat                = self.attrs['dlat']
        nlat_grad           = self.attrs['nlat_grad']
        nlon_grad           = self.attrs['nlon_grad']
        nlat_lplc           = self.attrs['nlat_lplc']
        nlon_lplc           = self.attrs['nlon_lplc']
        fdict               = { 'Tph': 2, 'Tgr': 3, 'amp': 4}
        # load catalog from input ASDF file
        inDbase             = pyasdf.ASDFDataSet(inasdffname)
        print '--- loading catalog'
        cat                 = inDbase.events
        print '--- end loading catalog'
        L                   = len(cat)
        datalst             = inDbase.auxiliary_data[data_type].list()
        #-------------------------------------------------------------------------------------------------
        # quality control for the data before performing eikonal/Helmholtz operation, added 10/08/2018
        #-------------------------------------------------------------------------------------------------
        if pre_qual_ctrl:
            print '--- quality control for events'
            qc_cat              = obspy.Catalog()
            evnumb              = 0
            qc_evnumb           = 0
            evid_lst            = []
            if btime_qc is not None:
                btime_qc        = obspy.UTCDateTime(btime_qc)
            else:
                btime_qc        = obspy.UTCDateTime('1900-01-01')
            if etime_qc is not None:
                etime_qc        = obspy.UTCDateTime(etime_qc)
            else:
                etime_qc        = obspy.UTCDateTime('2599-01-01')
            for event in cat:
                evnumb          += 1
                evid            = 'E%05d' % evnumb
                outstr          = ''
                porigin         = event.preferred_origin()
                evlo            = porigin.longitude
                evla            = porigin.latitude
                evdp            = porigin.depth
                otime           = porigin.time
                pmag            = event.preferred_magnitude()
                magnitude       = pmag.mag
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                dataid          = evid+'_'+channel
                if not dataid in datalst:
                    continue
                if otime < btime_qc or otime > etime_qc:
                    print('SKIP: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                    continue
                # loop over periods
                skip_this_event     = True
                for iper in range(pers.size):
                    per             = pers[iper]
                    del_per         = per-int(per)
                    if del_per == 0.:
                        persfx      = str(int(per))+'sec'
                    else:
                        dper        = str(del_per)
                        persfx      = str(int(per))+'sec'+dper.split('.')[1]
                    try:
                        subdset     = inDbase.auxiliary_data[data_type][evid+'_'+channel][persfx]
                    except KeyError:
                        continue
                    dataArr         = subdset.data.value
                    if dataArr.shape[0] < mindp:
                        continue
                    lons            = dataArr[:, 0]
                    lats            = dataArr[:, 1]
                    if _check_station_distribution(lons, lats, np.int32(mindp/2.5)):
                        skip_this_event \
                                    = False
                        break
                if skip_this_event:
                    print('SKIP: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                    continue
                print('ACCEPT: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                qc_evnumb           += 1
                qc_cat              += event
                evid_lst.append(evid)
            Lqc                     = len(qc_cat)
            print '--- end quality control, events number = '+str(Lqc)+'/'+str(L)
            cat                     = qc_cat
        #--------------------------------------
        # eikonal/Helmholtz computation
        #--------------------------------------
        for per in pers:
            if per < Tmin or per > Tmax:
                continue
            print 'Computing gradient for: '+str(per)+' sec'
            del_per         = per-int(per)
            if del_per == 0.:
                persfx      = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                persfx      = str(int(per))+'sec'+dper.split('.')[1]
            working_per     = workingdir+'/'+str(per)+'sec'
            per_group       = group.require_group( name='%g_sec'%( per ) )
            # loop over events
            evnumb          = 0
            for event in cat:
                evnumb          += 1
                # added on 2018/10/08
                if pre_qual_ctrl:
                    evid        = evid_lst[evnumb-1]
                    qc_evid     = 'E%05d' % evnumb
                else:
                    evid        = 'E%05d' % evnumb
                ###
                if evid != 'E10811':
                    continue
                ###
                porigin         = event.preferred_origin()
                evlo            = porigin.longitude
                evla            = porigin.latitude
                evdp            = porigin.depth
                otime           = porigin.time
                pmag            = event.preferred_magnitude()
                magnitude       = pmag.mag
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                dataid          = evid+'_'+channel
                if not dataid in datalst:
                    # print('No field data for eikonal/Helmholtz tomography')
                    continue
                try:
                    subdset     = inDbase.auxiliary_data[data_type][evid+'_'+channel][persfx]
                except KeyError:
                    # print('No field data for eikonal/Helmholtz tomography')
                    continue
                if evlo<0.:
                    evlo        +=360.
                dataArr         = subdset.data.value
                field2d         = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                    minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype=fieldtype,\
                                        nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
                Zarr            = dataArr[:, fdict[fieldtype]]
                # added on 03/06/2018
                if Zarr.size <= mindp:
                    continue
                # added on 10/08/2018
                inlons          = dataArr[:, 0]
                inlats          = dataArr[:, 1]
                if not _check_station_distribution(inlons, inlats, np.int32(mindp/2.)):
                    continue
                distArr         = dataArr[:, 6] # Note amplitude is added!!!
                field2d.read_array(lonArr = inlons, latArr = inlats, ZarrIn = distArr/Zarr )
                # # # #
                # # # field2d.evid    = evid
                # # # helmhotz4mp([field2d], workingdir=working_per, channel='Z', amplplc=False, cdist=cdist)
                # # # #
                outfname        = evid+'_'+fieldtype+'_'+channel+'.lst'
                field2d.interp_surface(workingdir=working_per, outfname=outfname)
                if not field2d.check_curvature(workingdir=working_per, outpfx=evid+'_'+channel+'_'):
                    continue
                field2d.eikonal_operator(workingdir=working_per, inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)
                #-----------------------------
                # save data to hdf5 dataset
                #-----------------------------
                event_group     = per_group.create_group(name=evid) # evid is not the qc_evid
                event_group.attrs.create(name = 'evlo', data=evlo)
                event_group.attrs.create(name = 'evla', data=evla)
                # added 04/05/2018
                event_group.attrs.create(name = 'Ntotal_grd', data=field2d.Ntotal_grd)
                event_group.attrs.create(name = 'Nvalid_grd', data=field2d.Nvalid_grd)
                # save computed data arrays
                appVdset        = event_group.create_dataset(name='appV', data=field2d.appV)
                reason_ndset    = event_group.create_dataset(name='reason_n', data=field2d.reason_n)
                proAngledset    = event_group.create_dataset(name='proAngle', data=field2d.proAngle)
                azdset          = event_group.create_dataset(name='az', data=field2d.az)
                bazdset         = event_group.create_dataset(name='baz', data=field2d.baz)
                Tdset           = event_group.create_dataset(name='travelT', data=field2d.Zarr)
                #--------------------------------------
                # perform Helmholtz computation
                #--------------------------------------
                if amplplc:
                    # computation
                    field2dAmp      = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon, minlat=minlat, maxlat=maxlat,\
                                        dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype='amp', nlat_grad=nlat_grad, \
                                            nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
                    field2dAmp.read_array(lonArr=dataArr[:,0], latArr=dataArr[:,1], ZarrIn=dataArr[:, fdict['amp']] )
                    outfnameAmp     = evid+'_Amp_'+channel+'.lst'
                    field2dAmp.interp_surface(workingdir=working_per, outfname=outfnameAmp)
                    field2dAmp.check_curvature_amp(workingdir=working_per, outpfx=evid+'_Amp_'+channel+'_',  threshold=0.1)
                    field2dAmp.helmholtz_operator(workingdir=working_per, inpfx=evid+'_Amp_'+channel+'_', lplcthresh=0.1)
                    field2d.get_lplc_amp(fieldamp=field2dAmp)
                    # save data
                    lplc_ampdset    = event_group.create_dataset(name='lplc_amp', data=field2d.lplc_amp)
                    corV_ampdset    = event_group.create_dataset(name='corV', data=field2d.corV)
                    reason_nhelmdset= event_group.create_dataset(name='reason_n_helm', data=field2d.reason_n_helm)
                    ampdset         = event_group.create_dataset(name='amp', data=field2dAmp.Zarr)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def quake_eikonal_mp(self, inasdffname, workingdir, fieldtype='Tph', channel='Z', data_type='FieldDISPpmf2interp',
                pre_qual_ctrl=True, btime_qc=None, etime_qc = None, incat=None, evid_lst=None,  runid=0, merge=True,
                    deletetxt=True, verbose=True, subsize=1000, nprocess=None, amplplc=False, cdist=150., mindp=50, pers=None):
        """
        Compute gradient of travel time for cross-correlation data with multiprocessing
        =======================================================================================================================
        ::: input parameters :::
        inasdffname     - input ASDF data file
        workingdir      - working directory
        fieldtype       - fieldtype (Tph or Tgr)
        channel         - channel for analysis
        data_type       - data type
                            default='FieldDISPpmf2interp': 
                                interpolated aftan measurements with phase-matched filtering and jump correction
        --- pre-tomography quality control ---
        pre_qual_ctrl   - perform pre-tomography quality control or not
        btime_qc        - begin time for quality control
        etime_qc        - end time for quality control
        incat           - input (quality-controlled) catalog
        evid_lst        - event id list corresponding to incat
        --------------------------------------
        runid           - run id
        deletetxt       - delete output txt files in working directory
        subsize         - subsize of processing list, use to prevent lock in multiprocessing process
        nprocess        - number of processes
        amplplc         - compute amplitude Laplacian term or not
        cdist           - distance for nearneighbor station criteria
        mindp           - minnimum required number of data points for eikonal operator
        =======================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        # merge data to existing group or not
        if merge:
            try:
                group           = self.create_group( name = 'Eikonal_run_'+str(runid) )
                group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
            except ValueError:
                print 'Merging Eikonal run id: ',runid
                group           = self.require_group( name = 'Eikonal_run_'+str(runid) )
        else:
            create_group        = False
            while (not create_group):
                try:
                    group       = self.create_group( name = 'Eikonal_run_'+str(runid) )
                    create_group= True
                except:
                    runid       += 1
                    continue
            group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        # if period is specified, check if it is in the header 
        if isinstance(pers, np.ndarray):
            pers_dbase      = self.attrs['period_array']
            for per in pers:
                if not (per in pers_dbase):
                    raise KeyError('Period '+str(per)+' s in the database attributes')
        else:
            pers            = self.attrs['period_array']
        minlon              = self.attrs['minlon']
        maxlon              = self.attrs['maxlon']
        minlat              = self.attrs['minlat']
        maxlat              = self.attrs['maxlat']
        dlon                = self.attrs['dlon']
        dlat                = self.attrs['dlat']
        nlat_grad           = self.attrs['nlat_grad']
        nlon_grad           = self.attrs['nlon_grad']
        nlat_lplc           = self.attrs['nlat_lplc']
        nlon_lplc           = self.attrs['nlon_lplc']
        fdict               = { 'Tph': 2, 'Tgr': 3, 'amp': 4}
        fieldLst            = []
        # load catalog from input ASDF file
        inDbase             = pyasdf.ASDFDataSet(inasdffname)
        # if incat and evid_lst is specified, skip quality control
        if incat is not None and evid_lst is not None:
            cat             = incat
            pre_qual_ctrl   = False
        else:
            print '--- loading catalog'
            cat             = inDbase.events
            print '--- end loading catalog'
            L               = len(cat)
        datalst             = inDbase.auxiliary_data[data_type].list()
        #-------------------------------------------------------------------------------------------------
        # quality control for the data before performing eikonal/Helmholtz operation, added 2018-10-10
        #-------------------------------------------------------------------------------------------------
        if pre_qual_ctrl:
            print '--- quality control for events'
            qc_cat              = obspy.Catalog()
            evnumb              = 0
            qc_evnumb           = 0
            evid_lst            = []
            if btime_qc is not None:
                btime_qc        = obspy.UTCDateTime(btime_qc)
            else:
                btime_qc        = obspy.UTCDateTime('1900-01-01')
            if etime_qc is not None:
                etime_qc        = obspy.UTCDateTime(etime_qc)
            else:
                etime_qc        = obspy.UTCDateTime('2599-01-01')
            for event in cat:
                evnumb          += 1
                evid            = 'E%05d' % evnumb
                outstr          = ''
                porigin         = event.preferred_origin()
                evlo            = porigin.longitude
                evla            = porigin.latitude
                evdp            = porigin.depth
                otime           = porigin.time
                pmag            = event.preferred_magnitude()
                magnitude       = pmag.mag
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                dataid          = evid+'_'+channel
                if not dataid in datalst:
                    continue
                if otime < btime_qc or otime > etime_qc:
                    print('--- SKIP: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                    continue
                # loop over periods
                skip_this_event     = True
                for iper in range(pers.size):
                    per             = pers[iper]
                    del_per         = per-int(per)
                    if del_per == 0.:
                        persfx      = str(int(per))+'sec'
                    else:
                        dper        = str(del_per)
                        persfx      = str(int(per))+'sec'+dper.split('.')[1]
                    try:
                        subdset     = inDbase.auxiliary_data[data_type][evid+'_'+channel][persfx]
                    except KeyError:
                        continue
                    dataArr         = subdset.data.value
                    if dataArr.shape[0] < mindp:
                        continue
                    lons            = dataArr[:, 0]
                    lats            = dataArr[:, 1]
                    if _check_station_distribution(lons, lats, np.int32(mindp/2.5)):
                        skip_this_event     = False
                        break
                if skip_this_event:
                    if verbose:
                        print('--- SKIP: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                    continue
                if verbose:
                    print('--- ACCEPT: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                qc_evnumb           += 1
                qc_cat              += event
                evid_lst.append(evid)
            Lqc                     = len(qc_cat)
            print '--- end quality control, events number = '+str(Lqc)+'/'+str(L)
            cat                     = qc_cat
        if incat is not None and evid_lst is not None:
            pre_qual_ctrl       = True
        #-------------------------
        # prepare data
        #-------------------------
        for per in pers:
            print 'preparing for: '+str(per)+' sec'
            del_per         = per-int(per)
            if del_per==0.:
                persfx      = str(int(per))+'sec'
            else:
                dper        = str(del_per)
                persfx      = str(int(per))+'sec'+dper.split('.')[1]
            working_per     = workingdir+'/'+str(per)+'sec'
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            per_group       = group.require_group( name='%g_sec'%( per ) )
            evnumb          = 0
            for event in cat:
                evnumb      +=1
                # added on 2018-10-10
                if pre_qual_ctrl:
                    evid        = evid_lst[evnumb-1]
                    qc_evid     = 'E%05d' % evnumb
                else:
                    evid        = 'E%05d' % evnumb
                try:
                    subdset = inDbase.auxiliary_data[data_type][evid+'_'+channel][persfx]
                except KeyError:
                    # print 'No travel time field for: '+evid
                    continue
                porigin         = event.preferred_origin()
                evlo            = porigin.longitude
                evla            = porigin.latitude
                evdp            = porigin.depth
                otime           = porigin.time
                pmag            = event.preferred_magnitude()
                magnitude       = pmag.mag
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                if verbose:
                    print 'Event '+str(evnumb)+' :'+event_descrip+', '+Mtype+' = '+str(magnitude) 
                if evlo < 0.:
                    evlo        += 360.
                dataArr         = subdset.data.value
                fieldpair       = []
                field2d         = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                    minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype=fieldtype, evid=evid)
                Zarr            = dataArr[:, fdict[fieldtype]]
                # added on 2018/03/06
                if Zarr.size <= mindp:
                    continue
                # added on 2018/10/10
                inlons          = dataArr[:, 0]
                inlats          = dataArr[:, 1]
                if not _check_station_distribution(inlons, inlats, np.int32(mindp/2.)):
                    continue
                distArr         = dataArr[:, 6] # Note amplitude in added!!!
                # field2d.read_array(lonArr=np.append(evlo, dataArr[:,0]), latArr=np.append(evla, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                field2d.read_array(lonArr = inlons, latArr = inlats, ZarrIn = distArr/Zarr )
                fieldpair.append(field2d)
                if amplplc:
                    field2dAmp  = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype='amp', evid=evid)
                    field2dAmp.read_array(lonArr = dataArr[:,0], latArr = dataArr[:,1], ZarrIn = dataArr[:, fdict['amp']] )
                    fieldpair.append(field2dAmp)
                fieldLst.append(fieldpair)
            # return fieldLst
        #----------------------------------------
        # Computing gradient with multiprocessing
        #----------------------------------------
        if len(fieldLst) > subsize:
            Nsub                = int(len(fieldLst)/subsize)
            for isub in range(Nsub):
                print '--- eikonal/helmholtz computation: subset:', isub,'in',Nsub,'sets'
                cfieldLst       = fieldLst[isub*subsize:(isub+1)*subsize]
                HELMHOTZ        = partial(helmhotz4mp, workingdir=workingdir, channel=channel, amplplc=amplplc, cdist=cdist)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(HELMHOTZ, cfieldLst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cfieldLst           = fieldLst[(isub+1)*subsize:]
            HELMHOTZ            = partial(helmhotz4mp, workingdir=workingdir, channel=channel, amplplc=amplplc, cdist=cdist)
            pool                = multiprocessing.Pool(processes=nprocess)
            pool.map(HELMHOTZ, cfieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            print '--- eikonal/helmholtz computation: one set'
            HELMHOTZ            = partial(helmhotz4mp, workingdir=workingdir, channel=channel, amplplc=amplplc, cdist=cdist)
            pool                = multiprocessing.Pool(processes=nprocess)
            pool.map(HELMHOTZ, fieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        #-----------------------------------
        # read data into hdf5 dataset
        #-----------------------------------
        for per in pers:
            print '--- reading gradient data for: '+str(per)+' sec'
            working_per         = workingdir+'/'+str(per)+'sec'
            per_group           = group.require_group( name='%g_sec'%( per ) )
            evnumb              = 0
            for event in cat:
                evnumb          += 1
                # added on 2018-10-10
                if pre_qual_ctrl:
                    evid        = evid_lst[evnumb-1]
                    qc_evid     = 'E%05d' % evnumb
                else:
                    evid        = 'E%05d' % evnumb
                infname         = working_per+'/'+evid+'_field2d.npz'
                if not os.path.isfile(infname):
                    print '--- No data for:', evid
                    continue
                InArr           = np.load(infname)
                appV            = InArr['arr_0']
                reason_n        = InArr['arr_1']
                proAngle        = InArr['arr_2']
                az              = InArr['arr_3']
                baz             = InArr['arr_4']
                Zarr            = InArr['arr_5']
                if amplplc:
                    lplc_amp        = InArr['arr_6']
                    corV            = InArr['arr_7']
                    reason_n_helm   = InArr['arr_8']
                    Ngrd            = InArr['arr_9']
                else:
                    Ngrd            = InArr['arr_6']
                porigin         = event.preferred_origin()
                evlo            = porigin.longitude
                evla            = porigin.latitude
                evdp            = porigin.depth
                # save data to hdf5 dataset
                event_group     = per_group.require_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=evlo)
                event_group.attrs.create(name = 'evla', data=evla)
                # added 04/05/2018
                event_group.attrs.create(name = 'Ntotal_grd', data=Ngrd[0])
                event_group.attrs.create(name = 'Nvalid_grd', data=Ngrd[1])
                # eikonal results
                appVdset        = event_group.create_dataset(name='appV', data=appV)
                reason_ndset    = event_group.create_dataset(name='reason_n', data=reason_n)
                proAngledset    = event_group.create_dataset(name='proAngle', data=proAngle)
                azdset          = event_group.create_dataset(name='az', data=az)
                bazdset         = event_group.create_dataset(name='baz', data=baz)
                Tdset           = event_group.create_dataset(name='travelT', data=Zarr)
                if amplplc:
                    lplc_ampdset    = event_group.create_dataset(name='lplc_amp', data=lplc_amp)
                    corV_dset       = event_group.create_dataset(name='corV', data=corV)
                    reason_nhelmdset= event_group.create_dataset(name='reason_n_helm', data=reason_n_helm)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def quake_eikonal_mp_lowmem(self, inasdffname, workingdir, fieldtype='Tph', channel='Z', data_type='FieldDISPpmf2interp',
                    pre_qual_ctrl=True, btime_qc = None, etime_qc = None, runid=0, deletetxt=True, verbose=False,
                        subsize=1000, nprocess=None, amplplc=False, cdist=150., mindp=50, Tmin=-999., Tmax=999.):
        """
        Low memory version of xcorr_eikonal_mp
        =======================================================================================================================
        ::: input parameters :::
        inasdffname     - input ASDF data file
        workingdir      - working directory
        fieldtype       - fieldtype (Tph or Tgr)
        channel         - channel for analysis
        data_type       - data type
                            default='FieldDISPpmf2interp': 
                                interpolated aftan measurements with phase-matched filtering and jump correction
        --- pre-tomography quality control ---
        pre_qual_ctrl   - perform pre-tomography quality control or not
        btime_qc        - begin time for quality control
        etime_qc        - end time for quality control
        --------------------------------------
        runid           - run id
        deletetxt       - delete output txt files in working directory
        subsize         - subsize of processing list, use to prevent lock in multiprocessing process
        nprocess        - number of processes
        amplplc         - compute amplitude Laplacian term or not
        cdist           - distance for nearneighbor station criteria
        mindp           - minnimum required number of data points for eikonal operator
        Tmin/Tmax       - minimum/maxsimum period for computation
        =======================================================================================================================
        """
        pers_dbase      = self.attrs['period_array']
        #-------------------------------------------------------------------------------------------------
        # quality control for the data before performing eikonal/Helmholtz operation, added 10/10/2018
        #-------------------------------------------------------------------------------------------------
        inDbase         = pyasdf.ASDFDataSet(inasdffname)
        print '--- loading catalog'
        cat             = inDbase.events
        print '--- end loading catalog'
        L               = len(cat)
        datalst         = inDbase.auxiliary_data[data_type].list()
        if pre_qual_ctrl:
            print '--- quality control for events'
            qc_cat              = obspy.Catalog()
            evnumb              = 0
            qc_evnumb           = 0
            evid_lst            = []
            if btime_qc is not None:
                btime_qc        = obspy.UTCDateTime(btime_qc)
            else:
                btime_qc        = obspy.UTCDateTime('1900-01-01')
            if etime_qc is not None:
                etime_qc        = obspy.UTCDateTime(etime_qc)
            else:
                etime_qc        = obspy.UTCDateTime('2599-01-01')
            for event in cat:
                evnumb          += 1
                evid            = 'E%05d' % evnumb
                outstr          = ''
                porigin         = event.preferred_origin()
                evlo            = porigin.longitude
                evla            = porigin.latitude
                evdp            = porigin.depth
                otime           = porigin.time
                pmag            = event.preferred_magnitude()
                magnitude       = pmag.mag
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                dataid          = evid+'_'+channel
                if not dataid in datalst:
                    continue
                if otime < btime_qc or otime > etime_qc:
                    print('--- SKIP: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                    continue
                # loop over periods
                skip_this_event     = True
                for iper in range(pers_dbase.size):
                    per             = pers_dbase[iper]
                    del_per         = per-int(per)
                    if del_per == 0.:
                        persfx      = str(int(per))+'sec'
                    else:
                        dper        = str(del_per)
                        persfx      = str(int(per))+'sec'+dper.split('.')[1]
                    try:
                        subdset     = inDbase.auxiliary_data[data_type][evid+'_'+channel][persfx]
                    except KeyError:
                        continue
                    dataArr         = subdset.data.value
                    if dataArr.shape[0] < mindp:
                        continue
                    lons            = dataArr[:, 0]
                    lats            = dataArr[:, 1]
                    if _check_station_distribution(lons, lats, np.int32(mindp/2.5)):
                        skip_this_event = False
                        break
                if skip_this_event:
                    print('--- SKIP: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                    continue
                print('--- ACCEPT: Event ' + str(evnumb)+'/'+str(L)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
                qc_evnumb           += 1
                qc_cat              += event
                evid_lst.append(evid)
            Lqc                     = len(qc_cat)
            print '--- end quality control, events number = '+str(Lqc)+'/'+str(L)
        else:
            qc_cat                  = None
            evid_lst                = None
        # Loop over periods
        for per in pers_dbase:
            if per < Tmin or per > Tmax:
                print '=== SKIP: eikonal tomography for T = '+str(per)+' sec'
                continue
            print '=== eikonal tomography for T = '+str(per)+' sec'
            start       = time.time()
            pers        = np.array([per])
            self.quake_eikonal_mp(inasdffname=inasdffname, workingdir=workingdir, fieldtype=fieldtype, channel=channel, data_type=data_type,
                pre_qual_ctrl=False, btime_qc=btime_qc, etime_qc=etime_qc, runid=runid, merge=True, deletetxt=deletetxt,
                    verbose=verbose, subsize=subsize, nprocess=nprocess, amplplc=amplplc, cdist=cdist, mindp=mindp, pers=pers,
                            incat = qc_cat, evid_lst=evid_lst)
            print '=== elasped time = '+str(time.time() - start)+' sec'
        return
    
    def eikonal_stack(self, runid=0, minazi=-180, maxazi=180, N_bin=20, threshmeasure=80, anisotropic=False, \
                spacing_ani=0.3, coverage=0.1, use_numba=True, azi_amp_tresh=0.05):
        """
        Stack gradient results to perform Eikonal Tomography
        =================================================================================================================
        ::: input parameters :::
        runid           - run id
        minazi/maxazi   - min/max azimuth for anisotropic parameters determination
        N_bin           - number of bins for anisotropic parameters determination
        threshmeasure   - minimum number of measurements required to perform stacking
        spacing_ani     - grid spacing for anisotropic stacking
        anisotropic     - perform anisotropic parameters determination or not
        coverage        - required coverage rate ({Number of valid grid points}/{Number of total grid points})
        use_numba       - use numba for large array manipulation or not, faster and much less memory requirement
        -----------------------------------------------------------------------------------------------------------------
        version history:
            Dec 6th, 2016   - add function to use numba, faster and much less memory consumption
            Feb 7th, 2018   - bug fixed by adding signALL,
                                originally stdArr = np.sum( (weightALL-avgArr)**2, axis=0), 2018-02-07
        =================================================================================================================
        """
        # read attribute information
        pers            = self.attrs['period_array']
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon            = self.attrs['dlon']
        dlat            = self.attrs['dlat']
        Nlon            = int(self.attrs['Nlon'])
        Nlat            = int(self.attrs['Nlat'])
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        nlat_lplc       = self.attrs['nlat_lplc']
        nlon_lplc       = self.attrs['nlon_lplc']
        group           = self['Eikonal_run_'+str(runid)]
        try:
            group_out   = self.create_group( name = 'Eikonal_stack_'+str(runid) )
        except ValueError:
            warnings.warn('Eikonal_stack_'+str(runid)+' exists! Will be recomputed!', UserWarning, stacklevel=1)
            del self['Eikonal_stack_'+str(runid)]
            group_out   = self.create_group( name = 'Eikonal_stack_'+str(runid) )
        #
        if anisotropic:
            grid_factor                 = int(np.ceil(spacing_ani/dlat))
            gridx                       = grid_factor
            gridy                       = grid_factor
            if gridx % 2 == 0:
                gridx                   += 1
            if gridy % 2 == 0:
                gridy                   += 1
            print '--- anisotropic grid factor = '+ str(gridx)+'/'+str(gridy)
            group_out.attrs.create(name = 'gridx', data = gridx)
            group_out.attrs.create(name = 'gridy', data = gridy)
        # attributes for output group
        group_out.attrs.create(name = 'anisotropic', data = anisotropic)
        group_out.attrs.create(name = 'N_bin', data = N_bin)
        group_out.attrs.create(name = 'minazi', data = minazi)
        group_out.attrs.create(name = 'maxazi', data = maxazi)
        group_out.attrs.create(name = 'fieldtype', data = group.attrs['fieldtype'])
        for per in pers:
            print '--- stacking eikonal results for: '+str(per)+' sec'
            per_group   = group['%g_sec'%( per )]
            Nevent      = len(per_group.keys())
            # initialize data arrays 
            Nmeasure    = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
            weightALL   = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            slownessALL = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            aziALL      = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            reason_nALL = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            validALL    = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            #-----------------------------------------------------
            # Loop over events to get eikonal maps for each event
            #-----------------------------------------------------
            print '*** reading data'
            for iev in range(Nevent):
                evid                        = per_group.keys()[iev]
                event_group                 = per_group[evid]
                az                          = event_group['az'].value
                #-------------------------------------------------
                # get apparent velocities for individual event
                #-------------------------------------------------
                velocity                    = event_group['appV'].value
                reason_n                    = event_group['reason_n'].value
                oneArr                      = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
                oneArr[reason_n!=0]         = 0
                slowness                    = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.float32)
                slowness[velocity!=0]       = 1./velocity[velocity!=0]                
                slownessALL[iev, :, :]      = slowness
                reason_nALL[iev, :, :]      = reason_n
                aziALL[iev, :, :]           = az
                Nmeasure                    += oneArr
                # quality control of coverage
                try:
                    Ntotal_grd              = event_group.attrs['Ntotal_grd']
                    Nvalid_grd              = event_group.attrs['Nvalid_grd']
                    if float(Nvalid_grd)/float(Ntotal_grd)< coverage:
                        reason_nALL[iev, :, :]  = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                except:
                    pass
                
            #----------------------------
            # isotropic stacking
            #----------------------------
            print '*** Stacking data'
            if Nmeasure.max()< threshmeasure:
                print ('--- No enough measurements for: '+str(per)+' sec')
                continue
            # discard grid points where number of raw measurements is low, added Sep 26th, 2018
            index_discard                   = Nmeasure < 50
            reason_nALL[:, index_discard]   = 10
            #-----------------------------------------------
            # Get weight for each grid point per event
            #-----------------------------------------------
            if use_numba:
                validALL[reason_nALL==0]    = 1
                weightALL                   = _get_azi_weight(aziALL, validALL)
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            else:
                azi_event1                  = np.broadcast_to(aziALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                azi_event2                  = np.swapaxes(azi_event1, 0, 1)
                validALL[reason_nALL==0]    = 1
                validALL4                   = np.broadcast_to(validALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                # use numexpr for very large array manipulations
                del_aziALL                  = numexpr.evaluate('abs(azi_event1-azi_event2)')
                index_azi                   = numexpr.evaluate('(1*(del_aziALL<20)+1*(del_aziALL>340))*validALL4')
                weightALL                   = numexpr.evaluate('sum(index_azi, 0)')
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            #-----------------------------------------------
            # reduce large weight to some value.
            #-----------------------------------------------
            avgArr                          = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            avgArr[Nmeasure!=0]             = weightsum[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            # bug fixed, 02/07/2018
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            stdArr                          = np.sum( signALL*(weightALL-avgArr)**2, axis=0)
            stdArr[Nmeasure!=0]             = stdArr[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            stdArr                          = np.sqrt(stdArr)
            threshhold                      = np.broadcast_to(avgArr+3.*stdArr, weightALL.shape)
            weightALL[weightALL>threshhold] = threshhold[weightALL>threshhold] # threshhold truncated weightALL
            # recompute weight arrays after large weight value reduction
            weightsum                       = np.sum(weightALL, axis=0)
            weightsumALL                    = np.broadcast_to(weightsum, weightALL.shape)
            # weight over all events, note that before this, weightALL is weight over events in azimuth bin
            weightALL[weightsumALL!=0]      = weightALL[weightsumALL!=0]/weightsumALL[weightsumALL!=0] 
            ###
            weightALL[weightALL==1.]        = 0. # data will be discarded if no other data within 20 degree
            #-----------------------------------------------
            # Compute mean/std of slowness
            #-----------------------------------------------
            slownessALL2                    = slownessALL*weightALL
            slowness_sum                    = np.sum(slownessALL2, axis=0)
            slowness_sumALL                 = np.broadcast_to(slowness_sum, weightALL.shape)
            # weighted standard deviation
            # formula: https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            MArr                            = np.sum(signALL, axis=0)
            temp                            = weightALL*(slownessALL-slowness_sumALL)**2
            temp                            = np.sum(temp, axis=0)
            slowness_std                    = np.zeros(temp.shape)
            tind                            = (weightsum!=0)*(MArr!=1)*(MArr!=0)
            slowness_std[tind]              = np.sqrt(temp[tind]/ ( weightsum[tind]*(MArr[tind]-1)/MArr[tind] ) )
            slowness_stdALL                 = np.broadcast_to(slowness_std, weightALL.shape)
            #-----------------------------------------------
            # discard outliers of slowness
            #-----------------------------------------------
            weightALLQC                     = weightALL.copy()
            index_outlier                   = (np.abs(slownessALL-slowness_sumALL))>2.*slowness_stdALL
            index_outlier                   += reason_nALL != 0
            weightALLQC[index_outlier]      = 0
            weightsumQC                     = np.sum(weightALLQC, axis=0)
            NmALL                           = np.sign(weightALLQC)
            NmeasureQC                      = np.sum(NmALL, axis=0)
            weightsumQCALL                  = np.broadcast_to(weightsumQC, weightALL.shape)
            weightALLQC[weightsumQCALL!=0]  = weightALLQC[weightsumQCALL!=0]/weightsumQCALL[weightsumQCALL!=0]
            temp                            = weightALLQC*slownessALL
            slowness_sumQC                  = np.sum(temp, axis=0)
            # new
            signALLQC                       = weightALLQC.copy()
            signALLQC[signALLQC!=0]         = 1.
            MArrQC                          = np.sum(signALLQC, axis=0)
            temp                            = weightALLQC*(slownessALL-slowness_sumQC)**2
            temp                            = np.sum(temp, axis=0)
            slowness_stdQC                  = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            slowness_stdQC[tind]            = np.sqrt(temp[tind]/ ( weightsumQC[tind]*(MArrQC[tind]-1)/MArrQC[tind] ))
            #---------------------------------------------------------------
            # mask, velocity, and sem arrays of shape Nlat, Nlon
            #---------------------------------------------------------------
            mask                            = np.ones((Nlat, Nlon), dtype=np.bool)
            tempmask                        = (weightsumQC == 0)
            mask[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                                            = tempmask
            vel_iso                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            tempvel                         = slowness_sumQC.copy()
            tempvel[tempvel!=0]             = 1./ tempvel[tempvel!=0]
            vel_iso[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempvel
            #----------------------------------------------------------------------------------------
            # standard error of the mean, updated on 09/20/2018
            # formula: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Statistical_properties
            #----------------------------------------------------------------------------------------
            slownessALL_temp                = slownessALL.copy()
            slownessALL_temp[slownessALL_temp==0.]\
                                            = 0.3
            if np.any(weightALLQC[slownessALL==0.]> 0.):
                raise ValueError('Check weight array!')
            temp                            = (weightALLQC*(1./slownessALL_temp-tempvel))**2
            temp                            = np.sum(temp, axis=0)
            tempsem                         = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            tempsem[tind]                   = np.sqrt( temp[tind] * ( MArrQC[tind]/(weightsumQC[tind])**2/(MArrQC[tind]-1) ) ) 
            vel_sem                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            vel_sem[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempsem
            #---------------------------------------
            # save isotropic velocity to database
            #---------------------------------------
            per_group_out                   = group_out.create_group( name='%g_sec'%( per ) )
            sdset                           = per_group_out.create_dataset(name='slowness', data=slowness_sumQC)
            s_stddset                       = per_group_out.create_dataset(name='slowness_std', data=slowness_stdQC)
            Nmdset                          = per_group_out.create_dataset(name='Nmeasure', data=Nmeasure)
            NmQCdset                        = per_group_out.create_dataset(name='NmeasureQC', data=NmeasureQC)
            maskdset                        = per_group_out.create_dataset(name='mask', data=mask)
            visodset                        = per_group_out.create_dataset(name='vel_iso', data=vel_iso)
            vsemdset                        = per_group_out.create_dataset(name='vel_sem', data=vel_sem)
            #----------------------------------------------------------------------------
            # determine anisotropic parameters, need benchmark and further verification
            #----------------------------------------------------------------------------
            if anisotropic:
                # quality control
                slowness_sumQC_ALL          = np.broadcast_to(slowness_sumQC, slownessALL.shape)
                # # # slowness_stdQC_ALL          = np.broadcast_to(slowness_stdQC, slownessALL.shape)
                # # # index_outlier               = (np.abs(slownessALL-slowness_sumQC_ALL))>2.*slowness_stdQC_ALL
                diff_slowness               = np.abs(slownessALL-slowness_sumQC_ALL)
                ind_nonzero                 = slowness_sumQC_ALL!= 0.
                diff_slowness[ind_nonzero]  = diff_slowness[ind_nonzero]/slowness_sumQC_ALL[ind_nonzero]
                index_outlier               += diff_slowness > azi_amp_tresh
                # stacking to get anisotropic parameters
                dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni    \
                                            = _anisotropic_stacking_parallel(np.int32(gridx), np.int32(gridy), np.float32(maxazi), np.float32(minazi),\
                                                np.int32(N_bin), np.float64(Nmeasure), np.float64(aziALL),\
                                                np.float64(slowness_sumQC), np.float64(slownessALL), index_outlier.astype(bool))
                #----------------------------
                # save data to database
                #----------------------------
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = dslow_sum_ani
                s_anidset       = per_group_out.create_dataset(name='slownessAni', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = dslow_un
                s_anisemdset    = per_group_out.create_dataset(name='slownessAni_sem', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = vel_un
                v_anisemdset    = per_group_out.create_dataset(name='velAni_sem', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = histArr
                histdset        = per_group_out.create_dataset(name='histArr', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr         = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[(gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = NmeasureAni
                NmAnidset       = per_group_out.create_dataset(name='NmeasureAni', data=out_arr)
        return
    
    def eikonal_stack_mp(self, workingdir='./eik_stack_dir', runid=0, minazi=-180, maxazi=180, N_bin=20, threshmeasure=80, anisotropic=False, \
                spacing_ani=0.3, coverage=0.1, use_numba=True, azi_amp_tresh=0.1, nprocess=None, enhanced=False):
        # read attribute information
        pers            = self.attrs['period_array']
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon            = self.attrs['dlon']
        dlat            = self.attrs['dlat']
        Nlon            = int(self.attrs['Nlon'])
        Nlat            = int(self.attrs['Nlat'])
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        nlat_lplc       = self.attrs['nlat_lplc']
        nlon_lplc       = self.attrs['nlon_lplc']
        group           = self['Eikonal_run_'+str(runid)]
        try:
            group_out   = self.create_group( name = 'Eikonal_stack_'+str(runid) )
        except ValueError:
            warnings.warn('Eikonal_stack_'+str(runid)+' exists! Will be recomputed!', UserWarning, stacklevel=1)
            del self['Eikonal_stack_'+str(runid)]
            group_out   = self.create_group( name = 'Eikonal_stack_'+str(runid) )
        # header for anisotropy
        if anisotropic:
            grid_factor                 = int(np.ceil(spacing_ani/dlat))
            gridx                       = grid_factor
            gridy                       = grid_factor
            if gridx % 2 == 0:
                gridx                   += 1
            if gridy % 2 == 0:
                gridy                   += 1
            print '--- anisotropic grid factor = '+ str(gridx)+'/'+str(gridy)
            group_out.attrs.create(name = 'gridx', data = gridx)
            group_out.attrs.create(name = 'gridy', data = gridy)
        else:
            gridx                       = 0
            gridy                       = 0
        # attributes for output group
        group_out.attrs.create(name = 'anisotropic', data = anisotropic)
        group_out.attrs.create(name = 'N_bin', data = N_bin)
        group_out.attrs.create(name = 'minazi', data = minazi)
        group_out.attrs.create(name = 'maxazi', data = maxazi)
        group_out.attrs.create(name = 'fieldtype', data = group.attrs['fieldtype'])
        #-----------------------
        # prepare data
        #-----------------------
        stack_lst       = []
        for per in pers:
            print '--- prepare eikonal results for stacking: '+str(per)+' sec'
            per_group   = group['%g_sec'%( per )]
            Nevent      = len(per_group.keys())
            # initialize data arrays 
            Nmeasure    = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
            weightALL   = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            slownessALL = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            aziALL      = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            reason_nALL = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            validALL    = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            #-----------------------------------------------------
            # Loop over events to get eikonal maps for each event
            #-----------------------------------------------------
            for iev in range(Nevent):
                evid                        = per_group.keys()[iev]
                event_group                 = per_group[evid]
                az                          = event_group['az'].value
                #-------------------------------------------------
                # get apparent velocities for individual event
                #-------------------------------------------------
                velocity                    = event_group['appV'].value
                reason_n                    = event_group['reason_n'].value
                oneArr                      = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
                oneArr[reason_n!=0]         = 0
                slowness                    = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.float32)
                slowness[velocity!=0]       = 1./velocity[velocity!=0]
                # data arrays 
                slownessALL[iev, :, :]      = slowness
                reason_nALL[iev, :, :]      = reason_n
                aziALL[iev, :, :]           = az
                Nmeasure                    += oneArr
                # quality control of coverage
                try:
                    Ntotal_grd              = event_group.attrs['Ntotal_grd']
                    Nvalid_grd              = event_group.attrs['Nvalid_grd']
                    if float(Nvalid_grd)/float(Ntotal_grd)< coverage:
                        reason_nALL[iev, :, :]  = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                except:
                    pass
            stack_lst.append(data4stack(slownessALL=slownessALL, reason_nALL=reason_nALL, aziALL=aziALL, Nmeasure=Nmeasure, \
                                        Nevent=Nevent, period=per, Nlon=Nlon, Nlat=Nlat, nlon_grad=nlon_grad, nlat_grad=nlat_grad))
        ###
        # mp stacking
        ###
        if not os.path.isdir(workingdir):
            os.makedirs(workingdir)
        print '=== eikonal stacking'
        STACK       = partial(stack4mp, workingdir=workingdir, minazi=minazi, maxazi=maxazi, N_bin=N_bin, threshmeasure=threshmeasure,\
                    anisotropic=anisotropic, spacing_ani=spacing_ani, coverage=coverage, use_numba=use_numba,\
                        azi_amp_tresh=azi_amp_tresh, gridx=gridx, gridy=gridy, Nlat=Nlat, Nlon=Nlon, nlat_grad=nlat_grad,\
                            nlon_grad=nlon_grad, enhanced=enhanced)
        pool        = multiprocessing.Pool(processes=nprocess)
        pool.map(STACK, stack_lst) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on
        ###
        # read stacked data
        ###
        for per in pers:
            print '--- reading eikonal stacked data : '+str(per)+' sec'
            pfx             = '%g_sec'%( per )
            infname         = workingdir+'/iso_'+pfx+'.npz'
            inarr           = np.load(infname)
            slowness_sumQC  = inarr['arr_0']
            slowness_stdQC  = inarr['arr_1']
            Nmeasure        = inarr['arr_2']
            NmeasureQC      = inarr['arr_3']
            mask            = inarr['arr_4']
            vel_iso         = inarr['arr_5']
            vel_sem         = inarr['arr_6']
            # save to database
            per_group_out                   = group_out.create_group( name='%g_sec'%( per ) )
            sdset                           = per_group_out.create_dataset(name='slowness', data=slowness_sumQC)
            s_stddset                       = per_group_out.create_dataset(name='slowness_std', data=slowness_stdQC)
            Nmdset                          = per_group_out.create_dataset(name='Nmeasure', data=Nmeasure)
            NmQCdset                        = per_group_out.create_dataset(name='NmeasureQC', data=NmeasureQC)
            maskdset                        = per_group_out.create_dataset(name='mask', data=mask)
            visodset                        = per_group_out.create_dataset(name='vel_iso', data=vel_iso)
            vsemdset                        = per_group_out.create_dataset(name='vel_sem', data=vel_sem)
            if anisotropic:
                infname         = workingdir+'/azi_'+pfx+'.npz'
                inarr           = np.load(infname)
                dslow_sum_ani   = inarr['arr_0']
                dslow_un        = inarr['arr_1']
                vel_un          = inarr['arr_2']
                histArr         = inarr['arr_3']
                NmeasureAni     = inarr['arr_4']
                # save database
                s_anidset       = per_group_out.create_dataset(name='slownessAni', data=dslow_sum_ani)
                s_anisemdset    = per_group_out.create_dataset(name='slownessAni_sem', data=dslow_un)
                v_anisemdset    = per_group_out.create_dataset(name='velAni_sem', data=vel_un)
                histdset        = per_group_out.create_dataset(name='histArr', data=histArr)
                NmAnidset       = per_group_out.create_dataset(name='NmeasureAni', data=NmeasureAni)    

    def helm_stack(self, runid=0, minazi=-180, maxazi=180, N_bin=20, threshmeasure=80, anisotropic=False, \
                spacing_ani=0.6, use_numba=True, coverage=0.1, dv_thresh=None):
        """
        Stack gradient results to perform Helmholtz Tomography
        =================================================================================================================
        ::: input parameters :::
        runid           - run id
        minazi/maxazi   - min/max azimuth for anisotropic parameters determination
        N_bin           - number of bins for anisotropic parameters determination
        anisotropic     - perform anisotropic parameters determination or not
        use_numba       - use numba for large array manipulation or not, faster and much less memory requirement
        -----------------------------------------------------------------------------------------------------------------
        version history:
            Dec 6th, 2016   - add function to use numba, faster and much less memory consumption
            Feb 7th, 2018   - bug fixed by adding signALL,
                                originally stdArr = np.sum( (weightALL-avgArr)**2, axis=0), 2018-02-07
        =================================================================================================================
        """
        pers            = self.attrs['period_array']
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon            = self.attrs['dlon']
        dlat            = self.attrs['dlat']
        Nlon            = int(self.attrs['Nlon'])
        Nlat            = int(self.attrs['Nlat'])
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        nlat_lplc       = self.attrs['nlat_lplc']
        nlon_lplc       = self.attrs['nlon_lplc']
        group           = self['Eikonal_run_'+str(runid)]
        try:
            group_out   = self.create_group( name = 'Helmholtz_stack_'+str(runid) )
        except ValueError:
            warnings.warn('Helmholtz_stack_'+str(runid)+' exists! Will be recomputed!', UserWarning, stacklevel=1)
            del self['Helmholtz_stack_'+str(runid)]
            group_out   = self.create_group( name = 'Helmholtz_stack_'+str(runid) )
        # attributes for output group
        group_out.attrs.create(name = 'anisotropic', data=anisotropic)
        group_out.attrs.create(name = 'N_bin', data=N_bin)
        group_out.attrs.create(name = 'minazi', data=minazi)
        group_out.attrs.create(name = 'maxazi', data=maxazi)
        group_out.attrs.create(name = 'fieldtype', data=group.attrs['fieldtype'])
        dnlat           = nlat_lplc - nlat_grad
        dnlon           = nlon_lplc - nlon_grad
        if dnlat < 0 or dnlon < 0:
            raise ValueError('nlat_lplc/nlon_lplc should not be smaller than nlat_grad/nlon_grad !')
        for per in pers:
            print 'Stacking Helmholtz results for: '+str(per)+' sec'
            per_group   = group['%g_sec'%( per )]
            Nevent      = len(per_group.keys())
            # initialize data arrays 
            Nmeasure    = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
            weightALL   = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            slownessALL = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            aziALL      = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            reason_nALL = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            validALL    = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            #-------------------------------------------------------
            # Loop over events to get Helmholtz maps for each event
            #-------------------------------------------------------
            print '--- Reading data'
            for iev in range(Nevent):
                evid                = per_group.keys()[iev]
                event_group         = per_group[evid]
                az                  = event_group['az'].value
                #-------------------------------------------------
                # get corrected velocities for individual event
                #-------------------------------------------------
                temp_vel            = event_group['corV'].value
                temp_reason_n       = event_group['reason_n_helm'].value
                velocity            = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.float32)
                reason_n            = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.float32)
                if dnlat == 0 and dnlon == 0:
                    reason_n        = temp_reason_n.copy()
                    velocity        = temp_vel.copy()
                elif dnlat == 0 and dnlon != 0:
                    reason_n[:, dnlon:-dnlon]\
                                    = temp_reason_n.copy()
                    velocity[:, dnlon:-dnlon]\
                                    = temp_vel.copy()
                elif dnlat != 0 and dnlon == 0:
                    reason_n[dnlat:-dnlat, :]\
                                    = temp_reason_n.copy()
                    velocity[dnlat:-dnlat, :]\
                                    = temp_vel.copy()
                else:
                    reason_n[dnlat:-dnlat, dnlon:-dnlon]\
                                    = temp_reason_n.copy()
                    velocity[dnlat:-dnlat, dnlon:-dnlon]\
                                    = temp_vel.copy()
                # quality control, compare with apparent velocity
                if dv_thresh is not None:
                    eikonal_grp         = self['Eikonal_stack_'+str(runid)]
                    per_eik_grp         = eikonal_grp['%g_sec'%( per )]
                    appV                = per_eik_grp['vel_iso']
                    appV                = appV[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]
                    ind                 = np.logical_not(((velocity - appV) <dv_thresh) * ((velocity - appV) >-dv_thresh))
                    reason_n[ind]       = 10.
                # 
                oneArr                  = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
                oneArr[reason_n!=0]     = 0
                Nmeasure                += oneArr
                slowness                = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.float32)
                slowness[velocity!=0]   = 1./velocity[velocity!=0]
                slownessALL[iev, :, :]  = slowness
                reason_nALL[iev, :, :]  = reason_n
                aziALL[iev, :, :]       = az
                # quality control of coverage
                try:
                    Ntotal_grd              = event_group.attrs['Ntotal_grd']
                    Nvalid_grd              = event_group.attrs['Nvalid_grd']
                    if float(Nvalid_grd)/float(Ntotal_grd)< coverage:
                        reason_nALL[iev, :, :]  = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                except:
                    pass
            print '--- Stacking data'
            if Nmeasure.max()< threshmeasure:
                print ('No enough measurements for: '+str(per)+' sec')
                continue
            #-----------------------------------------------
            # Get weight for each grid point per event
            #-----------------------------------------------
            if use_numba:
                validALL[reason_nALL==0]    = 1
                weightALL                   = _get_azi_weight(aziALL, validALL)
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            else:
                azi_event1                  = np.broadcast_to(aziALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                azi_event2                  = np.swapaxes(azi_event1, 0, 1)
                validALL[reason_nALL==0]    = 1
                validALL4                   = np.broadcast_to(validALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                # use numexpr for very large array manipulations
                del_aziALL                  = numexpr.evaluate('abs(azi_event1-azi_event2)')
                index_azi                   = numexpr.evaluate('(1*(del_aziALL<20)+1*(del_aziALL>340))*validALL4')
                weightALL                   = numexpr.evaluate('sum(index_azi, 0)')
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            #-----------------------------------------------
            # reduce large weight to some value.
            #-----------------------------------------------
            avgArr                          = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            avgArr[Nmeasure!=0]             = weightsum[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            # bug fixed, 02/07/2018
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            stdArr                          = np.sum( signALL*(weightALL-avgArr)**2, axis=0)
            stdArr[Nmeasure!=0]             = stdArr[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            stdArr                          = np.sqrt(stdArr)
            threshhold                      = np.broadcast_to(avgArr+3.*stdArr, weightALL.shape)
            weightALL[weightALL>threshhold] = threshhold[weightALL>threshhold] # threshhold truncated weightALL
            # recompute weight arrays after large weight value reduction
            weightsum                       = np.sum(weightALL, axis=0)
            weightsumALL                    = np.broadcast_to(weightsum, weightALL.shape)
            # weight over all events, note that before this, weightALL is weight over events in azimuth bin
            weightALL[weightsumALL!=0]      = weightALL[weightsumALL!=0]/weightsumALL[weightsumALL!=0] 
            weightALL[weightALL==1.]        = 0. # data will be discarded if no other data within 20 degree
            #-----------------------------------------------
            # Compute mean/std of slowness
            #-----------------------------------------------
            slownessALL2                    = slownessALL*weightALL
            slowness_sum                    = np.sum(slownessALL2, axis=0)
            slowness_sumALL                 = np.broadcast_to(slowness_sum, weightALL.shape)
            # weighted standard deviation
            # formula: https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            MArr                            = np.sum(signALL, axis=0)
            temp                            = weightALL*(slownessALL-slowness_sumALL)**2
            temp                            = np.sum(temp, axis=0)
            slowness_std                    = np.zeros(temp.shape)
            tind                            = (weightsum!=0)*(MArr!=1)*(MArr!=0)
            slowness_std[tind]              = np.sqrt(temp[tind]/ ( weightsum[tind]*(MArr[tind]-1)/MArr[tind] ) )
            slowness_stdALL                 = np.broadcast_to(slowness_std, weightALL.shape)
            #-----------------------------------------------
            # discard outliers of slowness
            #-----------------------------------------------
            weightALLQC                     = weightALL.copy()
            index_outlier                   = (np.abs(slownessALL-slowness_sumALL))>2.*slowness_stdALL
            index_outlier                   += reason_nALL != 0
            weightALLQC[index_outlier]      = 0
            weightsumQC                     = np.sum(weightALLQC, axis=0)
            NmALL                           = np.sign(weightALLQC)
            NmeasureQC                      = np.sum(NmALL, axis=0)
            weightsumQCALL                  = np.broadcast_to(weightsumQC, weightALL.shape)
            weightALLQC[weightsumQCALL!=0]  = weightALLQC[weightsumQCALL!=0]/weightsumQCALL[weightsumQCALL!=0]
            temp                            = weightALLQC*slownessALL
            slowness_sumQC                  = np.sum(temp, axis=0)
            # new
            signALLQC                       = weightALLQC.copy()
            signALLQC[signALLQC!=0]         = 1.
            MArrQC                          = np.sum(signALLQC, axis=0)
            temp                            = weightALLQC*(slownessALL-slowness_sumQC)**2
            temp                            = np.sum(temp, axis=0)
            slowness_stdQC                  = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)*(MArrQC!=0)
            slowness_stdQC[tind]            = np.sqrt(temp[tind]/ ( weightsumQC[tind]*(MArrQC[tind]-1)/MArrQC[tind] ))
            #---------------------------------------------------------------
            # mask, velocity, and sem arrays of shape Nlat, Nlon
            #---------------------------------------------------------------
            mask                            = np.ones((Nlat, Nlon), dtype=np.bool)
            tempmask                        = (weightsumQC == 0)
            mask[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                                            = tempmask
            vel_iso                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            tempvel                         = slowness_sumQC.copy()
            tempvel[tempvel!=0]             = 1./ tempvel[tempvel!=0]
            vel_iso[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempvel
            #----------------------------------------------------------------------------------------
            # standard error of the mean, updated on 09/20/2018
            # formula: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Statistical_properties
            #----------------------------------------------------------------------------------------
            slownessALL_temp                = slownessALL.copy()
            slownessALL_temp[slownessALL_temp==0.]\
                                            = 0.3
            if np.any(weightALLQC[slownessALL==0.]> 0.):
                raise ValueError('Check weight array!')
            temp                            = (weightALLQC*(1./slownessALL_temp-tempvel))**2
            temp                            = np.sum(temp, axis=0)
            tempsem                         = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            tempsem[tind]                   = np.sqrt( temp[tind] * ( MArrQC[tind]/(weightsumQC[tind])**2/(MArrQC[tind]-1) ) ) 
            vel_sem                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            vel_sem[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempsem
            #---------------------------------------
            # save isotropic velocity to database
            #---------------------------------------
            per_group_out                   = group_out.create_group( name='%g_sec'%( per ) )
            sdset                           = per_group_out.create_dataset(name='slowness', data=slowness_sumQC)
            s_stddset                       = per_group_out.create_dataset(name='slowness_std', data=slowness_stdQC)
            Nmdset                          = per_group_out.create_dataset(name='Nmeasure', data=Nmeasure)
            NmQCdset                        = per_group_out.create_dataset(name='NmeasureQC', data=NmeasureQC)
            maskdset                        = per_group_out.create_dataset(name='mask', data=mask)
            visodset                        = per_group_out.create_dataset(name='vel_iso', data=vel_iso)
            vsemdset                        = per_group_out.create_dataset(name='vel_sem', data=vel_sem)
            #----------------------------------------------------------------------------
            # determine anisotropic parameters, need benchmark and further verification
            #----------------------------------------------------------------------------
            # debug, synthetic anisotropy
            # phi             = 72.
            # A               = 0.01
            # phi             = phi/180.*np.pi
            # tempazi         = (aziALL+180.)/180.*np.pi
            # vALL            = np.broadcast_to(slowness_sumQC.copy(), slownessALL.shape)
            # vALL.setflags(write=1)
            # index           = vALL==0
            # vALL[vALL!=0]   = 1./vALL[vALL!=0]
            # # return slownessALL, slowness_sumQC
            # vALL            = vALL + A*np.cos(2*(tempazi-phi))
            # vALL[index]     = 0.
            # slownessALL     = vALL.copy()
            # slownessALL[slownessALL!=0] \
            #                 = 1./slownessALL[slownessALL!=0]
            
            if anisotropic:
                grid_factor                 = int(np.ceil(spacing_ani/dlat))
                gridx                       = grid_factor
                gridy                       = int(grid_factor*np.floor(dlon/dlat))
                Nx_size                     = Nlat-2*nlat_grad
                Ny_size                     = Nlon-2*nlon_grad
                NmeasureAni                 = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                total_near_neighbor         = Nmeasure[0:-2*gridx, 0:-2*gridy] + Nmeasure[0:-2*gridx, gridy:-gridy] + \
                                    Nmeasure[0:-2*gridx, 2*gridy:Ny_size] + Nmeasure[gridx:-gridx, 0:-2*gridy] +\
                                    Nmeasure[gridx:-gridx, gridy:-gridy] + Nmeasure[gridx:-gridx, 2*gridy:Ny_size] +\
                                    Nmeasure[2*gridx:Nx_size, 0:-2*gridy] + Nmeasure[2*gridx:Nx_size, gridy:-gridy] +\
                                    Nmeasure[2*gridx:Nx_size, 2*gridy:Ny_size]
                NmeasureAni[gridx:-gridx, gridy:-gridy]     \
                                            = total_near_neighbor # for quality control
                # initialization of anisotropic parameters
                d_bin                       = (maxazi-minazi)/N_bin
                print 'anisotropic grid factor = '+ str(gridx)+'/'+str(gridy)
                # number of measurements in each bin
                histArr                     = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                histArr_cutted              = histArr[:, gridx:-gridx, gridy:-gridy]
                # slowness in each bin
                slow_sum_ani                = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                slow_sum_ani_cutted         = slow_sum_ani[:, gridx:-gridx, gridy:-gridy]
                # slowness uncertainties for each bin
                slow_un                     = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                slow_un_cutted              = slow_un[:, gridx:-gridx, gridy:-gridy]
                # velocity uncertainties for each bin
                vel_un                      = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                vel_un_cutted               = vel_un[:, gridx:-gridx, gridy:-gridy]
                #
                index_dict                  = { 0: [0, -2*gridx, 0,         -2*gridy], \
                                                1: [0, -2*gridx, gridy,     -gridy],\
                                                2: [0, -2*gridx, 2*gridy,   Ny_size],\
                                                3: [gridx, -gridx, 0,       -2*gridy],\
                                                4: [gridx, -gridx, gridy, -gridy],\
                                                5: [gridx, -gridx, 2*gridy, Ny_size],\
                                                6: [2*gridx, Nx_size, 0,    -2*gridy],\
                                                7: [2*gridx, Nx_size, gridy,-gridy],\
                                                8: [2*gridx, Nx_size, 2*gridy, Ny_size]}
                nmin_bin                    = 2 # change
                #----------------------------------------------------------------------------------
                # Loop over azimuth bins to get slowness, velocity and number of measurements
                #----------------------------------------------------------------------------------
                for ibin in xrange(N_bin):
                    sumNbin                     = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    slowbin                     = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    slow_un_ibin                = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    velbin                      = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    vel_un_ibin                 = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    for i in range(9):
                        indarr                  = index_dict[i]
                        azi_arr                 = aziALL[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        ibinarr                 = np.floor((azi_arr - minazi)/d_bin)
                        weight_bin              = 1*(ibinarr==ibin)
                        index_outlier_cutted    = index_outlier[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        weight_bin[index_outlier_cutted] \
                                                = 0
                        slowsumQC_cutted        = slowness_sumQC[indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        slownessALL_cutted      = slownessALL[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        # differences in slowness numexpr.evaluate('sum(index_azi, 0)')
                        temp_dslow              = numexpr.evaluate('weight_bin*(slownessALL_cutted-slowsumQC_cutted)')
                        temp_dslow              = numexpr.evaluate('sum(temp_dslow, 0)')
                        # velocities
                        temp_vel                = slownessALL_cutted.copy()
                        temp_vel[temp_vel!=0]   = 1./temp_vel[temp_vel!=0]
                        temp_vel                = numexpr.evaluate('weight_bin*temp_vel')
                        temp_vel                = numexpr.evaluate('sum(temp_vel, 0)')
                        # number of measurements in this bin
                        N_ibin                  = numexpr.evaluate('sum(weight_bin, 0)')
                        # quality control
                        ind_valid               = N_ibin >= nmin_bin
                        sumNbin[ind_valid]      += N_ibin[ind_valid]
                        slowbin[ind_valid]      += temp_dslow[ind_valid]
                        velbin[ind_valid]       += temp_vel[ind_valid]
                    vel_mean                    = velbin.copy()
                    vel_mean[sumNbin!=0]        = velbin[sumNbin!=0]/sumNbin[sumNbin!=0]
                    dslow_mean                  = slowbin.copy()
                    dslow_mean[sumNbin!=0]      = dslow_mean[sumNbin!=0]/sumNbin[sumNbin!=0]
                    # compute uncertainties
                    for i in range(9):
                        indarr                  = index_dict[i]
                        azi_arr                 = aziALL[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        ibinarr                 = np.floor((azi_arr-minazi)/d_bin)
                        weight_bin              = 1*(ibinarr==ibin)
                        index_outlier_cutted    = index_outlier[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        weight_bin[index_outlier_cutted] \
                                                = 0
                        slowsumQC_cutted        = slowness_sumQC[indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        slownessALL_cutted      = slownessALL[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        temp_vel                = slownessALL_cutted.copy()
                        temp_vel[temp_vel!=0]   = 1./temp_vel[temp_vel!=0]
                        vel_un_ibin             = vel_un_ibin + numexpr.evaluate('sum( (weight_bin*(temp_vel-vel_mean))**2, 0)')
                        slow_un_ibin            = slow_un_ibin + numexpr.evaluate('sum( (weight_bin*(slownessALL_cutted-slowsumQC_cutted \
                                                                - dslow_mean))**2, 0)')
                    #------------------------------------
                    vel_un_ibin[sumNbin!=0]     = np.sqrt(vel_un_ibin[sumNbin!=0]/(sumNbin[sumNbin!=0]-1)/sumNbin[sumNbin!=0])
                    vel_un_cutted[ibin, :, :]   = vel_un_ibin
                    slow_un_ibin[sumNbin!=0]    = np.sqrt(slow_un_ibin[sumNbin!=0]/(sumNbin[sumNbin!=0]-1)/sumNbin[sumNbin!=0])
                    slow_un_cutted[ibin, :, :]  = slow_un_ibin
                    histArr_cutted[ibin, :, :]  = sumNbin
                    slow_sum_ani_cutted[ibin, :, :]  \
                                                = dslow_mean
                #-------------------------------------------
                N_thresh                                = 10 # change
                slow_sum_ani_cutted[histArr_cutted<N_thresh] \
                                                        = 0
                slow_sum_ani[:, gridx:-gridx, gridy:-gridy]\
                                                        = slow_sum_ani_cutted
                # uncertainties
                slow_un_cutted[histArr_cutted<N_thresh] = 0
                slow_un[:, gridx:-gridx, gridy:-gridy]  = slow_un_cutted
                # convert sem of slowness to sem of velocity
                vel_un_cutted[histArr_cutted<N_thresh]  = 0
                vel_un[:, gridx:-gridx, gridy:-gridy]   = vel_un_cutted
                # # # return vel_un
                # near neighbor quality control
                Ntotal_thresh                           = 45 # change
                slow_sum_ani[:, NmeasureAni<Ntotal_thresh]    \
                                                        = 0 
                slow_un[:, NmeasureAni<Ntotal_thresh]   = 0
                vel_un[:, NmeasureAni<Ntotal_thresh]    = 0
                histArr[:, gridx:-gridx, gridy:-gridy]  = histArr_cutted
                # save data to database
                s_anidset       = per_group_out.create_dataset(name='slownessAni', data=slow_sum_ani)
                s_anisemdset    = per_group_out.create_dataset(name='slownessAni_sem', data=slow_un)
                v_anisemdset    = per_group_out.create_dataset(name='velAni_sem', data=vel_un)
                histdset        = per_group_out.create_dataset(name='histArr', data=histArr)
                NmAnidset       = per_group_out.create_dataset(name='NmeasureAni', data=NmeasureAni)
        return
    
    def compute_azi_aniso(self, runid=0, helm=False, Ntotal_thresh=None, N_thresh=5, Nbin_thresh=5, semfactor=5.):
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingroup         = self[dataid]
        gridx           = ingroup.attrs['gridx']
        gridy           = ingroup.attrs['gridy']
        pers            = self.attrs['period_array']
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin_default    = ingroup.attrs['N_bin']
        d_bin           = float((maxazi-minazi)/Nbin_default)
        azArr           = np.arange(Nbin_default)*d_bin + minazi
        if Ntotal_thresh is None:
            Ntotal_thresh   = N_thresh*gridx*gridy*Nbin_default/2.
        self._get_lon_lat_arr()
        for period in pers:
            print '=== Fitting azimuthal angle and amp '+str(period)+' sec'
            pergrp      = ingroup['%g_sec'%( period )]
            mask        = pergrp['mask'].value
            slowAni     = pergrp['slownessAni'].value + pergrp['slowness'].value
            velAnisem   = pergrp['velAni_sem'].value * semfactor
            slowness    = pergrp['slowness'].value
            histArr     = pergrp['histArr'].value
            psiarr      = np.zeros((self.Nlat, self.Nlon))
            amparr      = np.zeros((self.Nlat, self.Nlon))
            misfitarr   = np.zeros((self.Nlat, self.Nlon))
            Nbinarr     = np.zeros((self.Nlat, self.Nlon))
            Nmarr       = np.zeros((self.Nlat, self.Nlon))
            mask_aniso  = np.ones((self.Nlat, self.Nlon), dtype=bool)
            #------------------------
            # uncertainty arrays
            #------------------------
            un_psiarr   = np.zeros((self.Nlat, self.Nlon))
            un_amparr   = np.zeros((self.Nlat, self.Nlon))
            for ilat in range(self.Nlat):
                if (ilat-nlat_grad) < 0 or (ilat+nlat_grad) > (self.Nlat - 1) :
                    continue
                for ilon in range(self.Nlon):
                    if (ilon-nlon_grad) < 0 or (ilon+nlon_grad) > (self.Nlon - 1) :
                        continue
                    if mask[ilat - nlat_grad, ilon - nlon_grad]:
                        continue
                    outslowness = slowAni[:, ilat - nlat_grad, ilon - nlon_grad]
                    outvel_sem  = velAnisem[:, ilat - nlat_grad, ilon - nlon_grad]
                    avg_slowness= slowness[ilat - nlat_grad, ilon - nlon_grad]
                    out_hist    = histArr[:, ilat - nlat_grad, ilon - nlon_grad]
                    if out_hist.sum() < Ntotal_thresh:
                        continue
                    # get the valid binned data
                    # quality control
                    index       = np.where((outvel_sem != 0)*(out_hist > N_thresh ))[0]
                    outslowness = outslowness[index]
                    az_grd      = azArr[index]
                    outvel_sem  = outvel_sem[index]
                    Nbin        = index.size
                    if Nbin < Nbin_thresh:
                        continue
                    Nmarr[ilat, ilon]   = int(out_hist.sum()/(gridx*gridy))
                    Nbinarr[ilat, ilon] = Nbin
                    # construct forward operator matrix
                    tG                  = np.ones((Nbin, 1), dtype=np.float64)
                    G                   = tG.copy()
                    # convert azimuth to the 'real' azimuth coordinate
                    az_grd              += 180.
                    az_grd              = 360. - az_grd
                    az_grd              -= 90.
                    az_grd[az_grd<0.]   += 360.
                    # 2-psi terms
                    tbaz                = np.pi*(az_grd)/180.
                    tGsin2              = np.sin(tbaz*2)
                    tGcos2              = np.cos(tbaz*2)
                    G                   = np.append(G, tGsin2)
                    G                   = np.append(G, tGcos2)
                    G                   = G.reshape((3, Nbin))
                    G                   = G.T
                    # data
                    indat               = (1./outslowness).reshape(1, Nbin)
                    d                   = indat.T
                    # data covariance matrix
                    Cd                  = np.zeros((Nbin, Nbin), dtype=np.float64)
                    np.fill_diagonal(Cd, outvel_sem**2)
                    # Tarantola's solution, p67 eq. 3.40, 3.41
                    Ginv1               = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cd)), G) )
                    Ginv2               = np.dot( np.dot(G.T, np.linalg.inv(Cd)), d)
                    model               = np.dot(Ginv1, Ginv2)
                    Cm                  = Ginv1 # model covariance matrix
                    pcov                = np.sqrt(np.absolute(Cm))
                    m0                  = ufloat(model[0][0], pcov[0][0])
                    m1                  = ufloat(model[1][0], pcov[1][1])
                    m2                  = ufloat(model[2][0], pcov[2][2])
                    #
                    A0                  = model[0][0]
                    A2                  = np.sqrt(model[1][0]**2 + model[2][0]**2)
                    psi2                = np.arctan2(model[1][0], model[2][0])/2.
                    if psi2 < 0.:
                        psi2            += np.pi # -90 ~ 90 -> 0 ~ 180.
                    # compute misfit
                    predat                  = A0 + A2*np.cos(2.*(np.pi/180.*(az_grd+180.)-psi2) )
                    misfit                  = np.sqrt( ((predat - 1./outslowness)**2 / outvel_sem**2).sum()/ az_grd.size )
                    amparr[ilat, ilon]      = A2/A0*100.  # convert to percentage
                    psiarr[ilat, ilon]      = psi2/np.pi*180.
                    mask_aniso[ilat, ilon]  = False
                    misfitarr[ilat, ilon]   = misfit
                    # uncertainties
                    unA2                    = (uncertainties.umath.sqrt(m1**2+m2**2)/m0*100.).std_dev
                    unpsi2                  = (uncertainties.umath.atan2(m1, m2)/np.pi*180./2.).std_dev
                    unamp                   = unA2
                    unpsi2                  = min(unpsi2, 90.)
                    un_psiarr[ilat, ilon]   = unpsi2
                    un_amparr[ilat, ilon]   = unamp
            try:
                pergrp.create_dataset(name='amparr', data=amparr)
                pergrp.create_dataset(name='psiarr', data=psiarr)
                pergrp.create_dataset(name='mask_aniso', data=mask_aniso)
                pergrp.create_dataset(name='misfit', data=misfitarr)
                pergrp.create_dataset(name='Nmtotal', data=Nmarr)
                pergrp.create_dataset(name='Nbin', data=Nbinarr)
                #
                pergrp.create_dataset(name='unpsi', data=un_psiarr)
                pergrp.create_dataset(name='unamp', data=un_amparr)
            except:
                del pergrp['amparr']
                del pergrp['psiarr']
                del pergrp['mask_aniso']
                del pergrp['misfit']
                del pergrp['Nmtotal']
                del pergrp['Nbin']
                # 
                del pergrp['unpsi']
                del pergrp['unamp']
                pergrp.create_dataset(name='amparr', data=amparr)
                pergrp.create_dataset(name='psiarr', data=psiarr)
                pergrp.create_dataset(name='mask_aniso', data=mask_aniso)
                pergrp.create_dataset(name='misfit', data=misfitarr)
                pergrp.create_dataset(name='Nmtotal', data=Nmarr)
                pergrp.create_dataset(name='Nbin', data=Nbinarr)
                # 
                pergrp.create_dataset(name='unpsi', data=un_psiarr)
                pergrp.create_dataset(name='unamp', data=un_amparr)
        return

    def compute_azi_aniso_enhanced_old(self, runid=0, fitpsi1=False, fitpsi2=True, helm=False,\
                Ntotal_thresh=None, N_thresh=5, Nbin_thresh=5, semfactor=5.):
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingroup         = self[dataid]
        gridx           = ingroup.attrs['gridx']
        gridy           = ingroup.attrs['gridy']
        pers            = self.attrs['period_array']
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin_default    = ingroup.attrs['N_bin']
        d_bin           = float((maxazi-minazi)/Nbin_default)
        azArr           = np.arange(Nbin_default)*d_bin + minazi
        if Ntotal_thresh is None:
            Ntotal_thresh   = N_thresh*gridx*gridy*Nbin_default/2.
        self._get_lon_lat_arr()
        for period in pers:
            print '=== Fitting azimuthal angle and amp '+str(period)+' sec'
            pergrp      = ingroup['%g_sec'%( period )]
            mask        = pergrp['mask'].value
            slowAni     = pergrp['slownessAni'].value + pergrp['slowness'].value
            velAnisem   = pergrp['velAni_sem'].value * semfactor
            slowness    = pergrp['slowness'].value
            histArr     = pergrp['histArr'].value
            psiarr      = np.zeros((self.Nlat, self.Nlon))
            amparr      = np.zeros((self.Nlat, self.Nlon))
            misfitarr   = np.zeros((self.Nlat, self.Nlon))
            Nbinarr     = np.zeros((self.Nlat, self.Nlon))
            Nmarr       = np.zeros((self.Nlat, self.Nlon))
            mask_aniso  = np.ones((self.Nlat, self.Nlon), dtype=bool)
            #---
            # uncertainties
            #---
            un_psiarr   = np.zeros((self.Nlat, self.Nlon))
            un_amparr   = np.zeros((self.Nlat, self.Nlon))
            for ilat in range(self.Nlat):
                if (ilat-nlat_grad) < 0 or (ilat+nlat_grad) > (self.Nlat - 1) :
                    continue
                for ilon in range(self.Nlon):
                    if (ilon-nlon_grad) < 0 or (ilon+nlon_grad) > (self.Nlon - 1) :
                        continue
                    if mask[ilat - nlat_grad, ilon - nlon_grad]:
                        continue
                    outslowness = slowAni[:, ilat - nlat_grad, ilon - nlon_grad]
                    outvel_sem  = velAnisem[:, ilat - nlat_grad, ilon - nlon_grad]
                    avg_slowness= slowness[ilat - nlat_grad, ilon - nlon_grad]
                    out_hist    = histArr[:, ilat - nlat_grad, ilon - nlon_grad]
                    if out_hist.sum() < Ntotal_thresh:
                        continue
                    # get the valid binned data
                    # quality control
                    index       = np.where((outvel_sem != 0)*(out_hist > N_thresh ))[0]
                    outslowness = outslowness[index]
                    az_grd      = azArr[index]
                    outvel_sem  = outvel_sem[index]
                    Nbin        = index.size
                    if Nbin < Nbin_thresh:
                        continue
                    ###
                    # # # if not (self.lons[ilon] == -150.+360. and self.lats[ilat] == 65.):
                    # # #     continue
                    ###
                    Nmarr[ilat, ilon]   = int(out_hist.sum()/(gridx*gridy))
                    Nbinarr[ilat, ilon] = Nbin
                    try:
                        if fitpsi1 or fitpsi2:
                            indat           = (1./outslowness).reshape(1, Nbin)
                            U               = np.zeros((Nbin, Nbin), dtype=np.float64)
                            np.fill_diagonal(U, 1./outvel_sem)
                            # construct forward operator matrix
                            tG              = np.ones((Nbin, 1), dtype=np.float64)
                            G               = tG.copy()
                            # convert azimuth to the 'real' azimuth coordinate
                            az_grd          += 180.
                            az_grd          = 360. - az_grd
                            az_grd          -= 90.
                            az_grd[az_grd<0.]\
                                            += 360.  
                            tbaz            = np.pi*(az_grd)/180.
                            if fitpsi1:
                                tGsin       = np.sin(tbaz)
                                tGcos       = np.cos(tbaz)
                                G           = np.append(G, tGsin)
                                G           = np.append(G, tGcos)
                            if fitpsi2:
                                tGsin2      = np.sin(tbaz*2)
                                tGcos2      = np.cos(tbaz*2)
                                G           = np.append(G, tGsin2)
                                G           = np.append(G, tGcos2)
                            if fitpsi1 and fitpsi2:
                                G           = G.reshape((5, Nbin))
                            else:
                                G           = G.reshape((3, Nbin))
                            G               = G.T
                            G               = np.dot(U, G)
                            # data
                            d               = indat.T
                            d               = np.dot(U, d)
                            # least square inversion
                            model           = np.linalg.lstsq(G, d)[0]
                            A0              = model[0]
                            if fitpsi1:
                                A1          = np.sqrt(model[1]**2 + model[2]**2)
                                phi1        = np.arctan2(model[1], model[2])/2.
                                if fitpsi2:
                                    A2      = np.sqrt(model[3]**2 + model[4]**2)
                                    phi2    = np.arctan2(model[3], model[4])/2.
                            else:
                                A2          = np.sqrt(model[1]**2 + model[2]**2)
                                phi2        = np.arctan2(model[1], model[2])/2.
                    except:
                        continue
                    predat                  = A0 + A2*np.cos(2.*(np.pi/180.*(az_grd+180.)-phi2) )
                    misfit                  = np.sqrt( ((predat - 1./outslowness)**2 / outvel_sem**2).sum()/ az_grd.size )
                    amparr[ilat, ilon]      = A2/A0*100.  # convert to percentage
                    if phi2 < 0.:
                        phi2                += np.pi
                    psiarr[ilat, ilon]      = phi2/np.pi*180.
                    mask_aniso[ilat, ilon]  = False
                    misfitarr[ilat, ilon]   = misfit
                    ###
                    # get uncertainties
                    ###
                    # print phi2, 
                    # # # return 1./outslowness, outvel_sem, az_grd, np.float64(A0[0])
                    unpsi, unamp            = _get_azi_sem(1./outslowness, outvel_sem, az_grd, np.float64(A0[0]))
                    if np.isinf(unpsi) or np.isinf(unamp):
                        # print 'ERROR: lon = ' + str(self.lons[ilon]) + ', lat ='+str(self.lats[ilat])
                        unpsi, unamp        = _get_azi_sem_2(1./outslowness, outvel_sem, az_grd, np.float64(A0[0]))
                        if np.isinf(unpsi) or np.isinf(unamp):
                            print 'ERROR: lon = ' + str(self.lons[ilon]) + ', lat ='+str(self.lats[ilat])
                            mask_aniso[ilat, ilon]  = True
                            un_psiarr[ilat, ilon]   = 0.
                            un_amparr[ilat, ilon]   = 0.
                        un_psiarr[ilat, ilon]   = unpsi
                        un_amparr[ilat, ilon]   = unamp
                        # # # return 1./outslowness, outvel_sem, az_grd, np.float64(A0[0])
                    else:
                        un_psiarr[ilat, ilon]   = unpsi
                        un_amparr[ilat, ilon]   = unamp
            try:
                pergrp.create_dataset(name='amparr', data=amparr)
                pergrp.create_dataset(name='psiarr', data=psiarr)
                pergrp.create_dataset(name='mask_aniso', data=mask_aniso)
                pergrp.create_dataset(name='misfit', data=misfitarr)
                pergrp.create_dataset(name='Nmtotal', data=Nmarr)
                pergrp.create_dataset(name='Nbin', data=Nbinarr)
                #
                pergrp.create_dataset(name='unpsi', data=un_psiarr)
                pergrp.create_dataset(name='unamp', data=un_amparr)
            except:
                del pergrp['amparr']
                del pergrp['psiarr']
                del pergrp['mask_aniso']
                del pergrp['misfit']
                del pergrp['Nmtotal']
                del pergrp['Nbin']
                # 
                del pergrp['unpsi']
                del pergrp['unamp']
                pergrp.create_dataset(name='amparr', data=amparr)
                pergrp.create_dataset(name='psiarr', data=psiarr)
                pergrp.create_dataset(name='mask_aniso', data=mask_aniso)
                pergrp.create_dataset(name='misfit', data=misfitarr)
                pergrp.create_dataset(name='Nmtotal', data=Nmarr)
                pergrp.create_dataset(name='Nbin', data=Nbinarr)
                # 
                pergrp.create_dataset(name='unpsi', data=un_psiarr)
                pergrp.create_dataset(name='unamp', data=un_amparr)
        return
    
    def compute_azi_aniso_enhanced(self, runid=0, fitpsi1=False, fitpsi2=True, helm=False,\
                Ntotal_thresh=None, N_thresh=5, Nbin_thresh=5, semfactor=5.):
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingroup         = self[dataid]
        gridx           = ingroup.attrs['gridx']
        gridy           = ingroup.attrs['gridy']
        pers            = self.attrs['period_array']
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin_default    = ingroup.attrs['N_bin']
        d_bin           = float((maxazi-minazi)/Nbin_default)
        azArr           = np.arange(Nbin_default)*d_bin + minazi
        if Ntotal_thresh is None:
            Ntotal_thresh   = N_thresh*gridx*gridy*Nbin_default/2.
        self._get_lon_lat_arr()
        for period in pers:
            print '=== Fitting azimuthal angle and amp '+str(period)+' sec'
            pergrp      = ingroup['%g_sec'%( period )]
            mask        = pergrp['mask'].value
            slowAni     = pergrp['slownessAni'].value + pergrp['slowness'].value
            velAnisem   = pergrp['velAni_sem'].value * semfactor
            slowness    = pergrp['slowness'].value
            histArr     = pergrp['histArr'].value
            psiarr      = np.zeros((self.Nlat, self.Nlon))
            amparr      = np.zeros((self.Nlat, self.Nlon))
            misfitarr   = np.zeros((self.Nlat, self.Nlon))
            Nbinarr     = np.zeros((self.Nlat, self.Nlon))
            Nmarr       = np.zeros((self.Nlat, self.Nlon))
            mask_aniso  = np.ones((self.Nlat, self.Nlon), dtype=bool)
            #---
            # uncertainties
            #---
            un_psiarr   = np.zeros((self.Nlat, self.Nlon))
            un_amparr   = np.zeros((self.Nlat, self.Nlon))
            for ilat in range(self.Nlat):
                if (ilat-nlat_grad) < 0 or (ilat+nlat_grad) > (self.Nlat - 1) :
                    continue
                for ilon in range(self.Nlon):
                    if (ilon-nlon_grad) < 0 or (ilon+nlon_grad) > (self.Nlon - 1) :
                        continue
                    if mask[ilat - nlat_grad, ilon - nlon_grad]:
                        continue
                    outslowness = slowAni[:, ilat - nlat_grad, ilon - nlon_grad]
                    outvel_sem  = velAnisem[:, ilat - nlat_grad, ilon - nlon_grad]
                    avg_slowness= slowness[ilat - nlat_grad, ilon - nlon_grad]
                    out_hist    = histArr[:, ilat - nlat_grad, ilon - nlon_grad]
                    if out_hist.sum() < Ntotal_thresh:
                        continue
                    # get the valid binned data
                    # quality control
                    index       = np.where((outvel_sem != 0)*(out_hist > N_thresh ))[0]
                    outslowness = outslowness[index]
                    az_grd      = azArr[index]
                    outvel_sem  = outvel_sem[index]
                    Nbin        = index.size
                    if Nbin < Nbin_thresh:
                        continue
                    Nmarr[ilat, ilon]   = int(out_hist.sum()/(gridx*gridy))
                    Nbinarr[ilat, ilon] = Nbin
                    try:
                        if fitpsi1 or fitpsi2:
                            indat           = (1./outslowness).reshape(1, Nbin)
                            U               = np.zeros((Nbin, Nbin), dtype=np.float64)
                            np.fill_diagonal(U, 1./outvel_sem)
                            # construct forward operator matrix
                            tG              = np.ones((Nbin, 1), dtype=np.float64)
                            G               = tG.copy()
                            # convert azimuth to the 'real' azimuth coordinate
                            az_grd          += 180.
                            az_grd          = 360. - az_grd
                            az_grd          -= 90.
                            az_grd[az_grd<0.]\
                                            += 360.  
                            tbaz            = np.pi*(az_grd)/180.
                            if fitpsi1:
                                tGsin       = np.sin(tbaz)
                                tGcos       = np.cos(tbaz)
                                G           = np.append(G, tGsin)
                                G           = np.append(G, tGcos)
                            if fitpsi2:
                                tGsin2      = np.sin(tbaz*2)
                                tGcos2      = np.cos(tbaz*2)
                                G           = np.append(G, tGsin2)
                                G           = np.append(G, tGcos2)
                            if fitpsi1 and fitpsi2:
                                G           = G.reshape((5, Nbin))
                            else:
                                G           = G.reshape((3, Nbin))
                            G               = G.T
                            Gta             = G.copy()
                            G               = np.dot(U, G)
                            # data
                            d               = indat.T
                            dta             = d.copy()
                            d               = np.dot(U, d)
                            # least square inversion
                            model           = np.linalg.lstsq(G, d)[0]
                            A0              = model[0]
                            if fitpsi1:
                                A1          = np.sqrt(model[1]**2 + model[2]**2)
                                phi1        = np.arctan2(model[1], model[2])/2.
                                if fitpsi2:
                                    A2      = np.sqrt(model[3]**2 + model[4]**2)
                                    phi2    = np.arctan2(model[3], model[4])/2.
                            else:
                                A2          = np.sqrt(model[1]**2 + model[2]**2)
                                phi2        = np.arctan2(model[1], model[2])/2.
                    except:
                        continue
                    predat                  = A0 + A2*np.cos(2.*(np.pi/180.*(az_grd+180.)-phi2) )
                    misfit                  = np.sqrt( ((predat - 1./outslowness)**2 / outvel_sem**2).sum()/ az_grd.size )
                    amparr[ilat, ilon]      = A2/A0*100.  # convert to percentage
                    if phi2 < 0.:
                        phi2                += np.pi
                    psiarr[ilat, ilon]      = phi2/np.pi*180.
                    mask_aniso[ilat, ilon]  = False
                    misfitarr[ilat, ilon]   = misfit
                    ###
                    # get uncertainties
                    ###
                    # estimate uncertainties
                    x0   = np.array([model[0][0], model[1][0], model[2][0]], dtype=float)
                    # Tarantola uncertainties, p67 eq. 3.40, 3.41
                    Cd              = np.zeros((Nbin, Nbin), dtype=np.float64)
                    np.fill_diagonal(Cd, outvel_sem**2)
                    Ginv1           = np.linalg.inv( np.dot( np.dot(Gta.T, np.linalg.inv(Cd)), Gta) )
                    Ginv2           = np.dot( np.dot(Gta.T, np.linalg.inv(Cd)), dta)
                    mta             = np.dot(Ginv1, Ginv2)
                    Cm              = Ginv1
                    pcov            = np.sqrt(np.absolute(Cm))
                    m0              = ufloat(model[0][0], pcov[0][0])
                    m1              = ufloat(model[1][0], pcov[1][1])
                    m2              = ufloat(model[2][0], pcov[2][2])
                    if not np.allclose(mta[:, 0], x0):
                        raise ValueError('incompatible '+str(mta)+' '+str(x0))
                    # scipy uncertainties
                    # # # pfit, pcov, infodict, errmsg, success = \
                    # # #     optimize.leastsq(azi_errfunc, x0=x0,\
                    # # #                      args=(azArr, 1./outslowness[:, 0], outvel_sem[:, 0]), full_output=1, epsfcn=0.0001)
                    # # # if (len(azArr) > len(x0)) and pcov is not None:
                    # # #     s_sq = (azi_errfunc(x0, azArr, 1./outslowness[:, 0], outvel_sem[:, 0])**2).sum()/(len(azArr)-len(x0))
                    # # #     pcov = pcov * s_sq
                    # # # pcov            = np.sqrt(np.absolute(pcov))
                    # # # m0              = ufloat(model[0][0], pcov[0][0])
                    # # # m1              = ufloat(model[1][0], pcov[1][1])
                    # # # m2              = ufloat(model[2][0], pcov[2][2])

                    unA2            = (uncertainties.umath.sqrt(m1**2+m2**2)/m0*100.).std_dev
                    unpsi2          = uncertainties.umath.atan2(m1, m2).std_dev
                    unamp           = unA2
                    unpsi           = unpsi2/np.pi*180.
                    if unpsi>90.:
                        unpsi       = 90.
                    un_psiarr[ilat, ilon]   = unpsi
                    un_amparr[ilat, ilon]   = unamp
            try:
                pergrp.create_dataset(name='amparr', data=amparr)
                pergrp.create_dataset(name='psiarr', data=psiarr)
                pergrp.create_dataset(name='mask_aniso', data=mask_aniso)
                pergrp.create_dataset(name='misfit', data=misfitarr)
                pergrp.create_dataset(name='Nmtotal', data=Nmarr)
                pergrp.create_dataset(name='Nbin', data=Nbinarr)
                #
                pergrp.create_dataset(name='unpsi', data=un_psiarr)
                pergrp.create_dataset(name='unamp', data=un_amparr)
            except:
                del pergrp['amparr']
                del pergrp['psiarr']
                del pergrp['mask_aniso']
                del pergrp['misfit']
                del pergrp['Nmtotal']
                del pergrp['Nbin']
                # 
                del pergrp['unpsi']
                del pergrp['unamp']
                pergrp.create_dataset(name='amparr', data=amparr)
                pergrp.create_dataset(name='psiarr', data=psiarr)
                pergrp.create_dataset(name='mask_aniso', data=mask_aniso)
                pergrp.create_dataset(name='misfit', data=misfitarr)
                pergrp.create_dataset(name='Nmtotal', data=Nmarr)
                pergrp.create_dataset(name='Nbin', data=Nbinarr)
                # 
                pergrp.create_dataset(name='unpsi', data=un_psiarr)
                pergrp.create_dataset(name='unamp', data=un_amparr)
        return
    
    def num_measure_info(self, runid=0, percentage=None, num_thresh=None, helm=False):
        pers            = self.attrs['period_array']
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingrp           = self[dataid]
        for per in pers:
            # get data
            pergrp      = ingrp['%g_sec'%( per )]
            mask        = pergrp['mask'].value
            Nmeasure    = np.zeros(mask.shape)
            Nmeasure[1:-1, 1:-1]\
                        = pergrp['NmeasureQC'].value
            index       = np.logical_not(mask)
            Nmeasure2   = Nmeasure[index]
            if Nmeasure2.size==0:
                print '--- T = '+str(per)+' sec ---'
                print 'No data'
                print '----------------------------'
                continue
            NMmin       = Nmeasure2.min()
            NMmax       = Nmeasure2.max()
            if percentage is not None and num_thresh is None:
                NMthresh    = NMmax*percentage
            elif percentage is None and num_thresh is not None:
                NMthresh    = num_thresh
            elif percentage is not None and num_thresh is not None:
                NMthresh    = min(NMmax*percentage, num_thresh)
            else:
                raise ValueError('at least one of percentage/num_thresh should be specified')
            Nthresh     = Nmeasure2[Nmeasure2>=NMthresh].size    
            print '--- T = '+str(per)+' sec ---'
            print 'min Nmeasure: '+str(NMmin)+', max Nmeasure: '+str(NMmax)
            print 'threshhold num of measurement:'+str(NMthresh)+', number of grids larger than threhhold: '+str(Nthresh)
            print '----------------------------'
        return
    
    def debug_plot_azimuth(self, inlat, inlon):
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        self._get_lon_lat_arr()
        index           = np.where((self.latArr==inlat)*(self.lonArr==inlon))
        index_outlier   = self.index_outlier[:, index[0] - nlat_grad, index[1] - nlon_grad]
        slowness        = self.slownessALL[:, index[0] - nlat_grad, index[1] - nlon_grad]
        azi             = self.aziALL[:, index[0] - nlat_grad, index[1] - nlon_grad]
        
        outaz           = azi[index_outlier==0]
        outslow         = slowness[index_outlier==0]
        return outaz, outslow
        
    def plot_azimuthal_single_point(self, inlat, inlon, runid, period, helm=False, \
                            fitpsi1=False, fitpsi2=True, getdata=False, showfig=True, outfname = None, semfactor=5.):
        if inlon < 0:
            inlon       += 360.
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        self._get_lon_lat_arr()
        index   = np.where((self.latArr==inlat)*(self.lonArr==inlon))
        if index[0].size == 0 or index[1].size == 0:
            print 'No data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        mask            = pergrp['mask'].value
        if mask[index[0], index[1]]:
            print 'No valid data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        slowAni         = pergrp['slownessAni'].value + pergrp['slowness'].value
        velAnisem       = pergrp['velAni_sem'].value
        outslowness     = slowAni[:, index[0] - nlat_grad, index[1] - nlon_grad]
        outvel_sem      = velAnisem[:, index[0] - nlat_grad, index[1] - nlon_grad]
        avg_slowness    = pergrp['slowness'].value[index[0] - nlat_grad, index[1] - nlon_grad]
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin            = ingroup.attrs['N_bin']
        d_bin           = float((maxazi-minazi)/Nbin)
        azArr           = np.arange(Nbin)*d_bin + minazi
        hist_temp       = pergrp['histArr'].value[:, index[0] - nlat_grad, index[1] - nlon_grad]
        # outvel_sem      *= np.sqrt(hist_temp)
        
        ind             = np.where(outvel_sem != 0)[0]
        outslowness     = outslowness[ind]
        azArr           = azArr[ind]
        outvel_sem      = outvel_sem[ind]*semfactor
        Nbin            = ind.size
        if getdata:
            return azArr, 1./outslowness, outvel_sem, 1./avg_slowness
        if fitpsi1 or fitpsi2:
            indat           = (1./outslowness).reshape(1, Nbin)
            U               = np.zeros((Nbin, Nbin), dtype=np.float64)
            np.fill_diagonal(U, 1./outvel_sem)
            # construct forward operator matrix
            tG              = np.ones((Nbin, 1), dtype=np.float64)
            G               = tG.copy()
            azArr           += 180.
            azArr           = 360. - azArr
            azArr           -= 90.
            azArr[azArr<0.] += 360.  
            tbaz            = np.pi*(azArr)/180.
            if fitpsi1:
                tGsin       = np.sin(tbaz)
                tGcos       = np.cos(tbaz)
                G           = np.append(G, tGsin)
                G           = np.append(G, tGcos)
            if fitpsi2:
                tGsin2      = np.sin(tbaz*2)
                tGcos2      = np.cos(tbaz*2)
                G           = np.append(G, tGsin2)
                G           = np.append(G, tGcos2)
            if fitpsi1 and fitpsi2:
                G           = G.reshape((5, Nbin))
            else:
                G           = G.reshape((3, Nbin))
            G               = G.T
            Gta             = G.copy()
            G               = np.dot(U, G)
            # data
            d               = indat.T
            dta             = d.copy()
            d               = np.dot(U, d)
            # least square inversion
            model           = np.linalg.lstsq(G, d)[0]
            A0              = model[0]
            if fitpsi1:
                A1          = np.sqrt(model[1]**2 + model[2]**2)
                phi1        = np.arctan2(model[1], model[2])/2.
                if fitpsi2:
                    A2      = np.sqrt(model[3]**2 + model[4]**2)
                    phi2    = np.arctan2(model[3], model[4])/2.
            else:
                A2          = np.sqrt(model[1]**2 + model[2]**2)
                phi2        = np.arctan2(model[1], model[2])/2.
            # estimate uncertainties
            x0   = np.array([model[0][0], model[1][0], model[2][0]], dtype=float)
            # Tarantola uncertainties
            Cd              = np.zeros((Nbin, Nbin), dtype=np.float64)
            np.fill_diagonal(Cd, outvel_sem**2)
            Ginv1           = np.linalg.inv( np.dot( np.dot(Gta.T, np.linalg.inv(Cd)), Gta) )
            Ginv2           = np.dot( np.dot(Gta.T, np.linalg.inv(Cd)), dta)
            mta             = np.dot(Ginv1, Ginv2)
            Cm              = Ginv1
            pcov            = np.sqrt(np.absolute(Cm))
            m0              = ufloat(model[0][0], pcov[0][0])
            m1              = ufloat(model[1][0], pcov[1][1])
            m2              = ufloat(model[2][0], pcov[2][2])
            # scipy uncertainties
            pfit, pcov, infodict, errmsg, success = \
                optimize.leastsq(azi_errfunc, x0=x0,\
                                 args=(azArr, 1./outslowness[:, 0], outvel_sem[:, 0]), full_output=1, epsfcn=0.0001)
            if (len(azArr) > len(x0)) and pcov is not None:
                s_sq = (azi_errfunc(x0, azArr, 1./outslowness[:, 0], outvel_sem[:, 0])**2).sum()/(len(azArr)-len(x0))
                pcov = pcov * s_sq
            pcov            = np.sqrt(np.absolute(pcov))
            m0              = ufloat(model[0][0], pcov[0][0])
            m1              = ufloat(model[1][0], pcov[1][1])
            m2              = ufloat(model[2][0], pcov[2][2])
            
            
            unA2            = (uncertainties.umath.sqrt(m1**2+m2**2)/m0*100.).std_dev
            unpsi2          = (uncertainties.umath.atan2(m1, m2)/np.pi*180./2.).std_dev
            
            # print unA2/A0*100., unpsi2

            if not np.allclose(mta[:, 0], x0):
                raise ValueError('incompatible '+str(mta)+' '+str(x0))
            # # # print np.sqrt(np.absolute(Cm))
            # # # print mta.shape
            # # # print x0
            return 
        
        plt.figure(figsize=[18, 9.6])
        ax      = plt.subplot()
        if helm:
            plt.errorbar(azArr, 1./outslowness, yerr=outvel_sem, fmt='o', label='Helmholtz observed')
        else:
            az_plt  = []
            vel_plt = []
            sem_plt = []
            for i in range(azArr.size):
                if azArr[i] in az_plt:
                    continue
                az_plt.append(azArr[i])
                vel_plt.append(1./outslowness[i])
                sem_plt.append(outvel_sem[i])
            plt.errorbar(az_plt, vel_plt, yerr=sem_plt, fmt='o', color='k', label='eikonal observed', ms=10)
        if fitpsi1 or fitpsi2:
            az_fit          = np.mgrid[minazi:maxazi:100*1j]
            if fitpsi1:
                predat      = A0 + A1*np.cos((np.pi/180.*(az_fit+180.)-phi1) )
                fitlabel    = 'A1: %g %%, phi1: %g deg \n' %(A1[0]/A0[0]*100., phi1/np.pi*180.)
                if fitpsi2:
                     predat     += A2*np.cos(2.*(np.pi/180.*(az_fit+180.)-phi2) )
                     fitlabel   += 'A2: %g %%, phi2: %g deg' %(A2[0]/A0[0]*100., phi2/np.pi*180.)
            else:
                predat      = A0 + A2*np.cos(2.*(np.pi/180.*(az_fit+180.)-phi2) )
                fitlabel    = 'A2: %g %%, phi2: %g deg' %(A2[0]/A0[0]*100., phi2/np.pi*180.)
            if helm:
                plt.plot(az_fit+180., predat, 'b-', lw=3, label='Helmholtz fit \n'+fitlabel )
            else:
                plt.plot(az_fit+180., predat, 'b-', lw=3, label='eikonal fit \n'+fitlabel )
            # print phi1/np.pi*180.
            # # plt.plot(azArr+180., predat, '-')
        plt.ylabel('Phase velocity(km/sec)', fontsize=50)
        plt.xlabel('Azimuth (degree)', fontsize=50)
        ax.tick_params(axis='x', labelsize=50)
        ax.tick_params(axis='y', labelsize=50)
        plt.legend()
        # plt.title('lon = '+str(inlon-360.)+', lat = '+str(inlat), fontsize=30.)
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
    
    def plot_azimuthal_single_point_cor(self, inlat, inlon, runid, period, helm=False, \
                            fitpsi1=False, fitpsi2=True, getdata=False, showfig=True, outfname = None):
        if inlon < 0:
            inlon       += 360.
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        self._get_lon_lat_arr()
        index   = np.where((self.latArr==inlat)*(self.lonArr==inlon))
        if index[0].size == 0 or index[1].size == 0:
            print 'No data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        mask            = pergrp['mask'].value
        if mask[index[0], index[1]]:
            print 'No valid data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        slowAni         = pergrp['slownessAni'].value + pergrp['slowness'].value
        velAnisem       = pergrp['velAni_sem'].value
        outslowness     = slowAni[:, index[0] - nlat_grad, index[1] - nlon_grad]
        outvel_sem      = velAnisem[:, index[0] - nlat_grad, index[1] - nlon_grad]
        avg_slowness    = pergrp['slowness'].value[index[0] - nlat_grad, index[1] - nlon_grad]
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin            = ingroup.attrs['N_bin']
        d_bin           = float((maxazi-minazi)/Nbin)
        azArr           = np.arange(Nbin)*d_bin + minazi
        hist_temp       = pergrp['histArr'].value[:, index[0] - nlat_grad, index[1] - nlon_grad]
        # outvel_sem      *= np.sqrt(hist_temp)
        
        ind             = np.where(outvel_sem != 0)[0]
        outslowness     = outslowness[ind]
        azArr           = azArr[ind]
        outvel_sem      = outvel_sem[ind]
        Nbin            = ind.size
        if getdata:
            return azArr, 1./outslowness, outvel_sem, 1./avg_slowness
        try:
            if fitpsi1 or fitpsi2:
                indat           = (1./outslowness).reshape(1, Nbin)
                U               = np.zeros((Nbin, Nbin), dtype=np.float64)
                np.fill_diagonal(U, 1./outvel_sem)
                # construct forward operator matrix
                tG              = np.ones((Nbin, 1), dtype=np.float64)
                G               = tG.copy()
                azArr           += 180.
                azArr           = 360. - azArr
                azArr           -= 90.
                azArr[azArr<0.] += 360.  
                tbaz            = np.pi*(azArr)/180.
                if fitpsi1:
                    tGsin       = np.sin(tbaz)
                    tGcos       = np.cos(tbaz)
                    G           = np.append(G, tGsin)
                    G           = np.append(G, tGcos)
                if fitpsi2:
                    tGsin2      = np.sin(tbaz*2)
                    tGcos2      = np.cos(tbaz*2)
                    G           = np.append(G, tGsin2)
                    G           = np.append(G, tGcos2)
                if fitpsi1 and fitpsi2:
                    G           = G.reshape((5, Nbin))
                else:
                    G           = G.reshape((3, Nbin))
                G               = G.T
                G               = np.dot(U, G)
                # data
                d               = indat.T
                d               = np.dot(U, d)
                # least square inversion
                model           = np.linalg.lstsq(G, d)[0]
                A0              = model[0]
                if fitpsi1:
                    A1          = np.sqrt(model[1]**2 + model[2]**2)
                    phi1        = np.arctan2(model[1], model[2])/2.
                    if fitpsi2:
                        A2      = np.sqrt(model[3]**2 + model[4]**2)
                        phi2    = np.arctan2(model[3], model[4])/2.
                else:
                    A2          = np.sqrt(model[1]**2 + model[2]**2)
                    phi2        = np.arctan2(model[1], model[2])/2.
            plt.figure(figsize=[18, 9.6])
            ax      = plt.subplot()
            if helm:
                plt.errorbar(azArr, 1./outslowness, yerr=outvel_sem, fmt='o', label='Helmholtz observed')
            else:
                az_plt  = []
                vel_plt = []
                sem_plt = []
                for i in range(azArr.size):
                    if azArr[i] in az_plt:
                        continue
                    az_plt.append(azArr[i])
                    vel_plt.append(1./outslowness[i])
                    sem_plt.append(outvel_sem[i])
                plt.errorbar(az_plt, vel_plt, yerr=sem_plt*5., fmt='o', color='k', label='eikonal observed', ms=10)
            if fitpsi1 or fitpsi2:
                az_fit          = np.mgrid[minazi:maxazi:100*1j]
                if fitpsi1:
                    predat      = A0 + A1*np.cos((np.pi/180.*(az_fit+180.)-phi1) )
                    fitlabel    = 'A1: %g %%, phi1: %g deg \n' %(A1[0]/A0[0]*100., phi1/np.pi*180.)
                    if fitpsi2:
                         predat     += A2*np.cos(2.*(np.pi/180.*(az_fit+180.)-phi2) )
                         fitlabel   += 'A2: %g %%, phi2: %g deg' %(A2[0]/A0[0]*100., phi2/np.pi*180.)
                else:
                    predat      = A0 + A2*np.cos(2.*(np.pi/180.*(az_fit+180.)-phi2) )
                    fitlabel    = 'A2: %g %%, phi2: %g deg' %(A2[0]/A0[0]*100., phi2/np.pi*180.)
                if helm:
                    plt.plot(az_fit+180., predat, 'b-', lw=3, label='Helmholtz fit \n'+fitlabel )
                else:
                    plt.plot(az_fit+180., predat, 'b-', lw=3, label='eikonal fit \n'+fitlabel )
                # print phi1/np.pi*180.
                # # plt.plot(azArr+180., predat, '-')
            plt.ylabel('Phase velocity(km/sec)', fontsize=50)
            plt.xlabel('Azimuth (degree)', fontsize=50)
            ax.tick_params(axis='x', labelsize=50)
            ax.tick_params(axis='y', labelsize=50)
            plt.legend()
            # plt.title('lon = '+str(inlon-360.)+', lat = '+str(inlat), fontsize=30.)
            if showfig:
                plt.show()
            if outfname is not None:
                plt.savefig(outfname)
        except:
            return
    
    def plot_azimuthal_single_point_4psi(self, inlat, inlon, runid, period, helm=False, \
                            fitpsi1=False, fitpsi4=True, getdata=False, showfig=True, outfname = None):
        if inlon < 0:
            inlon       += 360.
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        self._get_lon_lat_arr()
        index   = np.where((self.latArr==inlat)*(self.lonArr==inlon))
        if index[0].size == 0 or index[1].size == 0:
            print 'No data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        mask            = pergrp['mask'].value
        if mask[index[0], index[1]]:
            print 'No valid data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        slowAni         = pergrp['slownessAni'].value + pergrp['slowness'].value
        velAnisem       = pergrp['velAni_sem'].value
        outslowness     = slowAni[:, index[0] - nlat_grad, index[1] - nlon_grad]
        outvel_sem      = velAnisem[:, index[0] - nlat_grad, index[1] - nlon_grad]
        avg_slowness    = pergrp['slowness'].value[index[0] - nlat_grad, index[1] - nlon_grad]
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin            = ingroup.attrs['N_bin']
        azArr           = np.mgrid[minazi:maxazi:Nbin*1j]
        
        ind             = np.where(outvel_sem != 0)[0]
        outslowness     = outslowness[ind]
        azArr           = azArr[ind]
        outvel_sem      = outvel_sem[ind]
        Nbin            = ind.size
        if getdata:
            return azArr, 1./outslowness, outvel_sem, 1./avg_slowness
        try:
            if fitpsi1 or fitpsi4:
                indat           = (1./outslowness).reshape(1, Nbin)
                U               = np.zeros((Nbin, Nbin), dtype=np.float64)
                np.fill_diagonal(U, 1./outvel_sem)
                # construct forward operator matrix
                tG              = np.ones((Nbin, 1), dtype=np.float64)
                G               = tG.copy()
                azArr           += 180.
                azArr           = 360. - azArr
                azArr           -= 90.
                azArr[azArr<0.] += 360.  
                tbaz            = np.pi*(azArr)/180.
                if fitpsi1:
                    tGsin       = np.sin(tbaz)
                    tGcos       = np.cos(tbaz)
                    G           = np.append(G, tGsin)
                    G           = np.append(G, tGcos)
                if fitpsi4:
                    tGsin4      = np.sin(tbaz*4)
                    tGcos4      = np.cos(tbaz*4)
                    G           = np.append(G, tGsin4)
                    G           = np.append(G, tGcos4)
                if fitpsi1 and fitpsi4:
                    G           = G.reshape((5, Nbin))
                else:
                    G           = G.reshape((3, Nbin))
                G               = G.T
                G               = np.dot(U, G)
                # data
                d               = indat.T
                d               = np.dot(U, d)
                # least square inversion
                model           = np.linalg.lstsq(G, d)[0]
                A0              = model[0]
                if fitpsi1:
                    A1          = np.sqrt(model[1]**2 + model[2]**2)
                    phi1        = np.arctan2(model[1], model[2])/2.
                    if fitpsi4:
                        A4      = np.sqrt(model[3]**2 + model[4]**2)
                        phi4    = np.arctan2(model[3], model[4])/2.
                else:
                    A4          = np.sqrt(model[1]**2 + model[2]**2)
                    phi4        = np.arctan2(model[1], model[2])/2.
                # # # predat          = np.dot(G, model) * outvel_sem
                # # # az_fit          = np.mgrid[minazi:maxazi:100*1j]
                # # # predat          = A1*np.cos(np.pi*(az_fit+180.) - phi1)
            plt.figure(figsize=[18, 9.6])
            ax      = plt.subplot()
            if helm:
                plt.errorbar(azArr, 1./outslowness, yerr=outvel_sem, fmt='o', label='Helmholtz observed')
            else:
                # index   = (azArr>263.)*(azArr<273.)
                # v       = (1./outslowness)
                # index   = np.logical_not( index * (v[:, 0]< 3.52) )
                # plt.errorbar(azArr[index], 1./outslowness[index], yerr=outvel_sem[index], fmt='o', label='eikonal observed')
                ind       = np.logical_not( (abs(azArr - 270.) <5.) * (1./outslowness[:, 0]>3.847) + (1./outslowness[:, 0] < 3.835))
                # print azArr[ind]
                # print azArr.shape, outslowness.shape
                plt.errorbar(azArr[ind], 1./outslowness[ind], yerr=outvel_sem[ind]*5., fmt='o', color='k', ms=10, label='eikonal observed')
                
                # plt.errorbar(azArr, 1./outslowness, yerr=outvel_sem, fmt='o', color='k', label='eikonal observed')
                
            if fitpsi1 or fitpsi4:
                az_fit          = np.mgrid[minazi:maxazi:100*1j]
                if fitpsi1:
                    predat      = A0 + A1*np.cos((np.pi/180.*(az_fit+180.)-phi1) )
                    fitlabel    = 'A1: %g %%, phi1: %g deg \n' %(A1[0]/A0[0]*100., phi1/np.pi*180.)
                    if fitpsi4:
                         predat     += A4*np.cos(4.*(np.pi/180.*(az_fit+180.)-phi4) )
                         fitlabel   += 'A4: %g %%, phi2: %g deg' %(A4[0]/A0[0]*100., phi4/np.pi*180.)
                else:
                    predat      = A0 + A4*np.cos(4.*(np.pi/180.*(az_fit+180.)-phi4) )
                    fitlabel    = 'A4: %g %%, phi4: %g deg' %(A4[0]/A0[0]*100., phi4/np.pi*180.)
                if helm:
                    plt.plot(az_fit+180., predat, 'b-', lw=3, label='Helmholtz fit \n'+fitlabel )
                else:
                    plt.plot(az_fit+180., predat, 'b-', lw=3, label='eikonal fit \n'+fitlabel )
                # print phi1/np.pi*180.
                # # plt.plot(azArr+180., predat, '-')
            plt.ylabel('Phase velocity(km/sec)', fontsize=50)
            plt.xlabel('Azimuth (degree)', fontsize=50)
            ax.tick_params(axis='x', labelsize=50)
            ax.tick_params(axis='y', labelsize=50)
            plt.legend()
            plt.title('lon = '+str(inlon-360.)+', lat = '+str(inlat), fontsize=30.)
            if showfig:
                plt.show()
            if outfname is not None:
                plt.savefig(outfname)
        except:
            return
        
    def plot_azimuthal_single_point_all(self, inlat, inlon, runid, period, helm=False, \
                            fitpsi = True, getdata=False, showfig=True, outfname = None):
        if inlon < 0:
            inlon       += 360.
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        self._get_lon_lat_arr()
        index   = np.where((self.latArr==inlat)*(self.lonArr==inlon))
        if index[0].size == 0 or index[1].size == 0:
            print 'No data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        mask            = pergrp['mask'].value
        if mask[index[0], index[1]]:
            print 'No valid data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        slowAni         = pergrp['slownessAni'].value + pergrp['slowness'].value
        velAnisem       = pergrp['velAni_sem'].value
        outslowness     = slowAni[:, index[0] - nlat_grad, index[1] - nlon_grad]
        outvel_sem      = velAnisem[:, index[0] - nlat_grad, index[1] - nlon_grad]
        avg_slowness    = pergrp['slowness'].value[index[0] - nlat_grad, index[1] - nlon_grad]
        maxazi          = ingroup.attrs['maxazi']
        minazi          = ingroup.attrs['minazi']
        Nbin            = ingroup.attrs['N_bin']
        azArr           = np.mgrid[minazi:maxazi:Nbin*1j]
        
        ind             = np.where(outvel_sem != 0)[0]
        outslowness     = outslowness[ind]
        azArr           = azArr[ind]
        outvel_sem      = outvel_sem[ind]
        Nbin            = ind.size
        if getdata:
            return azArr, 1./outslowness, outvel_sem, 1./avg_slowness
        try:
            if fitpsi:
                indat           = (1./outslowness).reshape(1, Nbin)
                U               = np.zeros((Nbin, Nbin), dtype=np.float64)
                np.fill_diagonal(U, 1./outvel_sem)
                # construct forward operator matrix
                tG              = np.ones((Nbin, 1), dtype=np.float64)
                G               = tG.copy()
                azArr           += 180.
                azArr           = 360. - azArr
                azArr           -= 90.
                azArr[azArr<0.] += 360.  
                tbaz            = np.pi*(azArr)/180.
                # fit psi1
                tGsin           = np.sin(tbaz)
                tGcos           = np.cos(tbaz)
                G1              = np.append(G, tGsin)
                G1              = np.append(G1, tGcos)
                # fit psi2
                tGsin2          = np.sin(tbaz*2)
                tGcos2          = np.cos(tbaz*2)
                G2              = np.append(G, tGsin2)
                G2              = np.append(G2, tGcos2)
                G2              = G2.reshape((3, Nbin))
                # fit both
                G3              = np.append(G1, tGsin2)
                G3              = np.append(G3, tGcos2)
                G3              = G3.reshape((5, Nbin))
                #--------------------------
                # inversion, psi2
                #--------------------------
                G2              = G2.T
                G2              = np.dot(U, G2)
                # data
                d               = indat.T
                d               = np.dot(U, d)
                # least square inversion
                model2          = np.linalg.lstsq(G2, d)[0]
                #--------------------------
                # inversion, psi1 + psi2
                #--------------------------
                G3              = G3.T
                G3              = np.dot(U, G3)
                # data
                d               = indat.T
                d               = np.dot(U, d)
                # least square inversion
                model3          = np.linalg.lstsq(G3, d)[0]
                
                A0_2            = model2[0]
                A2_2            = np.sqrt(model2[1]**2 + model2[2]**2)
                phi2_2          = np.arctan2(model2[1], model2[2])/2.
                
                A0_3            = model3[0]
                A1_3            = np.sqrt(model3[1]**2 + model3[2]**2)
                phi1_3          = np.arctan2(model3[1], model3[2])/2.
                A2_3            = np.sqrt(model3[3]**2 + model3[4]**2)
                phi2_3          = np.arctan2(model3[3], model3[4])/2.
                    
                # # # predat          = np.dot(G, model) * outvel_sem
                # # # az_fit          = np.mgrid[minazi:maxazi:100*1j]
                # # # predat          = A1*np.cos(np.pi*(az_fit+180.) - phi1)
            plt.figure(figsize=[18, 9.6])
            ax      = plt.subplot()
            if helm:
                plt.errorbar(azArr, 1./outslowness, yerr=outvel_sem, fmt='o', label='Helmholtz observed')
            else:
                plt.errorbar(azArr, 1./outslowness, yerr=outvel_sem, fmt='o', label='eikonal observed')
            if fitpsi:
                az_fit          = np.mgrid[minazi:maxazi:100*1j]
                predat2         = A0_2 + A2_2*np.cos(2.*(np.pi/180.*(az_fit+180.)-phi2_2) )
                fitlabel2       = 'psi2 inversion: A2: %g %%, phi2: %g deg' %(A2_2[0]/A0_2[0]*100., phi2_2/np.pi*180.)
                predat3         = A0_3 + A1_3*np.cos((np.pi/180.*(az_fit+180.)-phi1_3) ) \
                                    + A2_3*np.cos(2.*(np.pi/180.*(az_fit+180.)-phi2_3) )
                fitlabel3       = 'psi1 + psi2 inversion:'+ 'A1: %g %%, phi1: %g deg; ' %(A1_3[0]/A0_3[0]*100., phi1_3/np.pi*180.) \
                                    + 'A2: %g %%, phi2: %g deg' %(A2_3[0]/A0_3[0]*100., phi2_3/np.pi*180.)
                plt.plot(az_fit+180., predat2, '-', label=fitlabel2 )
                plt.plot(az_fit+180., predat3, '-', label=fitlabel3 )
            plt.ylabel('Phase velocity(km/sec)', fontsize=20)
            plt.xlabel('Azimuth (degree)', fontsize=20)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            plt.legend()
            plt.title('lon = '+str(inlon-360.)+', lat = '+str(inlat), fontsize=30.)
            vmin    = min((1./outslowness).min(), predat2.min()) - 0.01
            vmax    = max((1./outslowness).max(), predat2.max()) + 0.01
            plt.ylim(vmin, vmax)
            if showfig:
                plt.show()
            if outfname is not None:
                plt.savefig(outfname)
        except:
            return

    def plot_azimuthal_eik_helm(self, inlat, inlon, runid, period, fitdata=True, getdata=False):
        self.plot_azimuthal_single_point(inlat=inlat, inlon=inlon, runid=runid,\
                    period=period, helm=False, fitdata=fitdata, getdata=getdata, showfig=False)
        self.plot_azimuthal_single_point(inlat=inlat, inlon=inlon, runid=runid,\
                    period=period, helm=True, fitdata=fitdata, getdata=getdata, showfig=True)
        return
       
    def plot_azi_fit(self, inlat, inlon, runid=0, helm=False, showfig=True, outfname = None):
        if inlon < 0:
            inlon       += 360.
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        self._get_lon_lat_arr()
        index   = np.where((self.latArr==inlat)*(self.lonArr==inlon))
        if index[0].size == 0 or index[1].size == 0:
            print 'No data at lon = '+str(inlon)+' lat = '+str(inlat)
            return
        amparr          = np.zeros(pers.size)
        psiarr          = np.zeros(pers.size)
        unamparr        = np.zeros(pers.size)
        unpsiarr        = np.zeros(pers.size)
        iper            = 0
        for per in pers:
            pergrp          = ingroup['%g_sec'%( per )]
            mask            = pergrp['mask'].value
            if mask[index[0], index[1]]:
                print 'No valid data at lon = '+str(inlon)+' lat = '+str(inlat)
                return
            amp         = pergrp['amparr'].value[index[0], index[1]]
            unamp       = pergrp['unamp'].value[index[0], index[1]]
            psi         = pergrp['psiarr'].value[index[0], index[1]]
            if psi < 0.:
                psi     += 180.
            unpsi       = pergrp['unpsi'].value[index[0], index[1]]
            vel_iso     = pergrp['vel_iso'].value[index[0], index[1]]
            
            # # # amp         = amp/vel_iso*100.
            
            unamp       = min(unamp, amp)
            #
            amparr[iper]    = amp
            unamparr[iper]  = unamp
            psiarr[iper]    = psi
            unpsiarr[iper]  = unpsi
            #
            iper        += 1
        # ---
        plt.figure()
        ax      = plt.subplot()
        plt.errorbar(pers, amparr, yerr=unamparr, fmt='o')
        plt.ylabel('Anisotropy amplitude (%)', fontsize=50)
        plt.xlabel('Period (sec)', fontsize=50)
        ax.tick_params(axis='x', labelsize=50)
        ax.tick_params(axis='y', labelsize=50)
        # --- 
        plt.figure()
        ax      = plt.subplot()
        plt.errorbar(pers, psiarr, yerr=unpsiarr, fmt='o')
        plt.ylabel('Fast azimuth (deg)', fontsize=50)
        plt.xlabel('Period (sec)', fontsize=50)
        ax.tick_params(axis='x', labelsize=50)
        ax.tick_params(axis='y', labelsize=50)
        if showfig:
            plt.show()
        if outfname is not None:
            plt.savefig(outfname)
        return
    
    def _get_lon_lat_arr(self, ncut=0):
        """Get longitude/latitude array
        """
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        dlon        = self.attrs['dlon']
        dlat        = self.attrs['dlat']
        self.lons   = np.arange((maxlon-minlon)/dlon+1-2*ncut)*dlon+minlon+ncut*dlon
        self.lats   = np.arange((maxlat-minlat)/dlat+1-2*ncut)*dlat+minlat+ncut*dlat
        self.Nlon   = self.lons.size
        self.Nlat   = self.lats.size
        self.lonArr, self.latArr = np.meshgrid(self.lons, self.lats)
        return
    
    def _get_lon_lat_arr_interp(self, ncut=0):
        """Get longitude/latitude array
        """
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        dlon        = self.attrs['dlon_interp']
        dlat        = self.attrs['dlat_interp']
        self.lons   = np.arange((maxlon-minlon)/dlon+1-2*ncut)*dlon+minlon+ncut*dlon
        self.lats   = np.arange((maxlat-minlat)/dlat+1-2*ncut)*dlat+minlat+ncut*dlat
        self.Nlon   = self.lons.size
        self.Nlat   = self.lats.size
        self.lonArr, self.latArr = np.meshgrid(self.lons, self.lats)
        return
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.figure()
        minlon      = self.attrs['minlon'] 
        maxlon      = self.attrs['maxlon'] 
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']        
        
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
        # m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    def plot(self, runid, datatype, period, semfactor=2., Nthresh=None, helm=False, merged=False, clabel='', cmap='cv', projection='lambert',\
                hillshade=False, geopolygons=None, vmin=None, vmax=None, showfig=True, mfault=True, v_rel=None):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runtype         - type of run (0 - smooth run, 1 - quality controlled run)
        runid           - id of run
        datatype        - datatype for plotting
        period          - period of data
        sem_factor      - factor multiplied to get the finalized uncertainties
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        ###
        # # # dset_in     = h5py.File('/work1/leon/ALASKA_work/hdf5_files/eikonal_quake_20181030.h5')
        # # # dataid      = 'Eikonal_stack_'+str(0)
        # # # ingroup     = dset_in[dataid]
        # # # pergrp      = ingroup['%g_sec'%( period )]
        # # # datatype    = 'vel_iso'
        # # # indata      = pergrp[datatype].value
        # # # inmask      = pergrp['mask'].value
        # # # Nm_in       = np.zeros(indata.shape)
        # # # Nm_in[1:-1, 1:-1] \
        # # #             = pergrp['NmeasureQC'].value
        ###
        
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        if merged:
            dataid      = 'merged_tomo_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        self._get_lon_lat_arr()
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        if datatype == 'vel' or datatype=='velocity' or datatype == 'v':
            datatype    = 'vel_iso'
        elif datatype == 'sem' or datatype == 'un' or datatype == 'uncertainty':
            datatype    = 'vel_sem'
        elif datatype=='std':
            datatype    = 'slowness_std'
        try:
            data        = pergrp[datatype].value
            if datatype=='slowness_std' or datatype=='Nmeasure' or datatype=='NmeasureQC' or datatype=='NmeasureAni':
                if self.lonArr.shape != data.shape:
                    data2   = data.copy()
                    data    = np.zeros(self.lonArr.shape)
                    data[1:-1, 1:-1] = data2
        except:
            outstr      = ''
            for key in pergrp.keys():
                outstr  +=key
                outstr  +=', '
            outstr      = outstr[:-1]
            raise KeyError('Unexpected datatype: '+datatype+\
                           ', available datatypes are: '+outstr)
        if datatype=='NmeasureAni':
            factor  = ingroup.attrs['gridx'] * ingroup.attrs['gridy']
        else:
            factor  = 1
        # mask1       = pergrp['mask'].value
        # mask2       = pergrp['mask_eik'].value
        if datatype=='NmeasureAni' or datatype=='unpsi' or datatype=='unamp' or datatype=='amparr':
            mask    = pergrp['mask_aniso'].value + pergrp['mask'].value
        else:
            mask    = pergrp['mask'].value
        # if datatype == 'amparr':
        #     data    *= 100.
        # mask2       += mask
        if not (Nthresh is None):
            Narr        = np.zeros(self.lonArr.shape)
            Narr[1:-1, 1:-1]        \
                        = pergrp['NmeasureQC'].value
            mask        += Narr < Nthresh
        if (datatype=='Nmeasure' or datatype=='NmeasureQC') and merged:
            mask    = pergrp['mask_eik'].value
        if datatype == 'vel_sem':
            data    *= 1000.*semfactor
        ###
        # # # minlon              = self.attrs['minlon']
        # # # maxlon              = self.attrs['maxlon']
        # # # minlat              = self.attrs['minlat']
        # # # maxlat              = self.attrs['maxlat']
        # # # dlon                = self.attrs['dlon']
        # # # dlat                = self.attrs['dlat']
        # # # nlat_grad           = self.attrs['nlat_grad']
        # # # nlon_grad           = self.attrs['nlon_grad']
        # # # nlat_lplc           = self.attrs['nlat_lplc']
        # # # nlon_lplc           = self.attrs['nlon_lplc']
        # # # field           = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
        # # #                             minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
        # # # index           = np.logical_not(mask)
        # # # field.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = data[index])
        # # # outfname        = 'smooth_paraval.lst'
        # # # field.gauss_smoothing(workingdir='./temp_smooth', outfname=outfname, width=15.)
        # # # data            = field.Zarr
        ###
        mdata       = ma.masked_array(data/factor, mask=mask )
        
        print data[np.logical_not(mask)].mean()
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr, self.latArr)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
                    cmap    = cmap.reversed()
            except:
                pass
        ###################################################################
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
        ###################################################################
        
        if v_rel is not None:
            mdata       = (mdata - v_rel)/v_rel * 100.
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=.5)
        else:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        ###
        # contour for eikonal
        # mask_eik    = pergrp['mask_eik'].value
        # if datatype == 'vel_iso':
        #     m.contour(x, y, mask_eik, linestyles='dashed', colors='black', lw=1.)
        #     # m.contour(x, y, mask_eik, colors='cyan', lw=1)
        # else:
        #     m.contour(x, y, mask_eik, linestyles='dashed', colors='black', lw=1.)
        ###
        
            # m.contour(x, y, mask_eik, colors='white', lw=1)
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.])
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
        # cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.0, 4.1, 4.2, 4.3, 4.4])
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=40, rotation=0)
        # cb.outline.set_linewidth(2)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        print 'plotting data from '+dataid
        if showfig:
            plt.show()
        return
    
    def plot_un_hist(self, runid, period, semfactor=2., Nthresh=None, helm=False, merged=False, showfig=True):
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        if merged:
            dataid      = 'merged_tomo_'+str(runid)
        ingroup         = self[dataid]
        # pers            = self.attrs['period_array']
        # self._get_lon_lat_arr()
        # if not period in pers:
        #     raise KeyError('period = '+str(period)+' not included in the database')
        # pergrp          = ingroup['%g_sec'%( period )]
        # if datatype == 'vel' or datatype=='velocity' or datatype == 'v':
        #     datatype    = 'vel_iso'
        # elif datatype == 'sem' or datatype == 'un' or datatype == 'uncertainty':
        #     datatype    = 'vel_sem'
        # elif datatype=='std':
        #     datatype    = 'slowness_std'
        # try:
        #     data        = pergrp[datatype].value
        #     if datatype=='slowness_std' or datatype=='Nmeasure' or datatype=='NmeasureQC':
        #         if self.lonArr.shape != data.shape:
        #             data2   = data.copy()
        #             data    = np.zeros(self.lonArr.shape)
        #             data[1:-1, 1:-1] = data2
        # except:
        #     outstr      = ''
        #     for key in pergrp.keys():
        #         outstr  +=key
        #         outstr  +=', '
        #     outstr      = outstr[:-1]
        #     raise KeyError('Unexpected datatype: '+datatype+\
        #                    ', available datatypes are: '+outstr)
        # 
        # mask        = pergrp['mask'].value
        # # mask2       += mask
        # if not (Nthresh is None):
        #     Narr        = np.zeros(self.lonArr.shape)
        #     Narr[1:-1, 1:-1]        \
        #                 = pergrp['NmeasureQC'].value
        #     mask        += Narr < Nthresh
        # if (datatype=='Nmeasure' or datatype=='NmeasureQC') and merged:
        #     mask    = pergrp['mask_eik'].value
        # if datatype == 'vel_sem':
        #     data    *= 1000.*semfactor
        # 
        # data       = data[mask==False]
        # 
        
        def to_percent(y, position):
            # Ignore the passed in position. This has the effect of scaling the default
            # tick locations.
            s = '%.0f' %(100. * y)
            # s = str(y)
            # The percent symbol needs escaping in latex
            if matplotlib.rcParams['text.usetex'] is True:
                return s + r'$\%$'
            else:
                return s + '%'
            
        ax      = plt.subplot()
        pers    = [period]
        N       = len(pers)
        pers2   = [period]
        # pers2   = [10., 30., 50.]
        
        # pers    = [10., 40., 70.]
        # N       = len(pers)
        # pers2   = [10., 40., 70.]
        for i in range(N):
            period  = pers[i]
            pergrp  = ingroup['%g_sec'%( period )]
            mask    = pergrp['mask'].value
            data    = pergrp['vel_sem'].value*1000.*semfactor
            data    = data[mask==False]
            weights = np.ones_like(data)/float(data.size)
            desired_bin_size    = 5.
            min_val = np.min(data)
            max_val = np.max(data)
            min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
            max_boundary = max_val - max_val % desired_bin_size + desired_bin_size

            n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
            plt.hist(data, bins=n_bins, weights=weights, label='Rayleigh wave, %g sec'%( pers2[i] ))
            # plt.hist(data, bins=n_bins, weights=weights, label='Love wave, %g sec'%( pers2[i] ))
            outstd  = data.std()
            outmean = data.mean()
        
        # compute mad
        from statsmodels import robust
        mad     = robust.mad(data)
        outmean = data.mean()
        outstd  = data.std()
        outmedian=np.median(data)
        plt.title('Misfit mean = %g , median = %g, std = %g , mad = %g ' %(outmean, outmedian, outstd, mad), fontsize=20)
        # plt.xlim(-.2, .2)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=80)
        plt.xlabel('Uncertainties (m/s)', fontsize=80)
        plt.xticks(np.arange(6)*20.)
        # plt.xticks(np.arange(8)*10.)
        plt.yticks(np.arange(6)*5./100.)
        ax.tick_params(axis='x', labelsize=50)
        ax.tick_params(axis='y', labelsize=50)
        from matplotlib.ticker import FormatStrFormatter
        
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.xlim([0., 100.])
        # plt.xlim([0., 20.])
        # plt.xlim([0., 60.])
        # plt.legend(loc=0, fontsize=20)
        # plt.xticks(np.arange(10.))
        # plt.show()
        if showfig:
            plt.show()
        return
        
    def plot_diff(self, runid, datatype, period, helm=False, clabel='', cmap='cv', projection='lambert', hillshade=False,\
                  geopolygons=None, vmin=None, vmax=None, showfig=True, mfault=True):
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
        showfig         - show figure or not
        =================================================================================================================
        """
        # vdict       = {'ph': 'C', 'gr': 'U'}
        self._get_lon_lat_arr()
        dataid          = 'merged_tomo_'+str(runid)
        # 
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        try:
            appV        = pergrp['vel_iso'].value
        except:
            outstr      = ''
            for key in pergrp.keys():
                outstr  +=key
                outstr  +=', '
            outstr      = outstr[:-1]
            raise KeyError('Unexpected datatype: '+datatype+\
                           ', available datatypes are: '+outstr)
        #
        dataid          = 'Helmholtz_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        try:
            corV        = pergrp['vel_iso'].value
        except:
            outstr      = ''
            for key in pergrp.keys():
                outstr  +=key
                outstr  +=', '
            outstr      = outstr[:-1]
            raise KeyError('Unexpected datatype: '+datatype+\
                           ', available datatypes are: '+outstr)
        data        = (appV - corV)
        mask        = pergrp['mask'].value
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr, self.latArr)
        if mfault:
            try:
                shapefname  = '/scratch/summit/life9360/ALASKA_work/fault_maps/qfaults'
                m.readshapefile(shapefname, 'faultline', linewidth=2, color='r')
            except:
                pass
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
                ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
        ###################################################################
        if hillshade:
            m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        else:
            m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=.5)
        else:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=12, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=15)
        if showfig:
            plt.show()
        return
    
    def plot_fast_axis(self, runid, period, helm=False, factor=10, normv=5., width=0.005, ampref=0.05, datatype='',
            scaled=False, masked=True, clabel='', cmap='cv', projection='lambert', hillshade=False,\
            geopolygons=None, vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        anipara         - anisotropic paramter
                            0   - isotropic
                            1   - 2 psi anisotropic
                            2   - 2&4 psi anisotropic
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        plot_vel        - plot velocity or not
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        thresh          - threhold value for Gaussian deviation to determine the mask for plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup         = self[dataid]
        # period array
        pers        = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi     = pergrp['psiarr'].value
        amp     = pergrp['amparr'].value
        mask    = pergrp['mask_aniso'].value + pergrp['mask'].value 
        # return 
        # get velocity
        try:
            data= pergrp[datatype].value
            plot_data   = True
        except:
            plot_data   = False
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        # # x, y        = m(self.lonArr-360., self.latArr)
        # shapefname  = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='red')
        
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        
        # for d in [20.,40., 60., 80., 100., 120.]:
        # for d in [100.]:
        #     slb_ctrlst      = read_slab_contour('alu_contours.in', depth=d)
        #     
        #     for slbctr in slb_ctrlst:
        #         xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
        #         m.plot(xslb, yslb,  '-', lw = 2, color='magenta', alpha=0.8)
        #     # m.plot(xslb, yslb,  '--', lw = 3, color='white')
        # 
        # # slab contour
        # # slb_ctrlst      = read_slab_contour('alu_contours.in', depth=100.)
        # # 
        # # for slbctr in slb_ctrlst:
        # #     xslb, yslb  = m(np.array(slbctr[0])-360., np.array(slbctr[1]))
        # #     m.plot(xslb, yslb,  '--', lw = 5, color='black')
        # #     m.plot(xslb, yslb,  '--', lw = 3, color='white')
        # # #############################
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        # #############################
        # import shapefile
        # shapefname  = '/home/leon/volcano_locs/SDE_GLB_VOLC.shp'
        # shplst      = shapefile.Reader(shapefname)
        # for rec in shplst.records():
        #     lon_vol = rec[4]
        #     lat_vol = rec[3]
        #     xvol, yvol            = m(lon_vol, lat_vol)
        #     m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=10)
        #--------------------------
        
        if scaled:
            # ampref  = amp.max()
            # print ampref
            U       = np.sin(psi/180.*np.pi)*amp/ampref/normv
            V       = np.cos(psi/180.*np.pi)*amp/ampref/normv
            Uref    = np.ones(self.lonArr.shape)*1./normv
            Vref    = np.zeros(self.lonArr.shape)
        else:
            U       = np.sin(psi/180.*np.pi)/normv
            V       = np.cos(psi/180.*np.pi)/normv
        # rotate vectors to map projection coordinates
        U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
        if scaled:
            Uref, Vref, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr-360., self.latArr, returnxy=True)
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(data, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=40, rotation=0)
            # cb.outline.set_linewidth(2)
            plt.suptitle(str(period)+' sec', fontsize=20)
            cb.ax.tick_params(labelsize=40)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        
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
        # Q       = m.quiver(x, y, U, V, scale=30, width=0.001, headaxislength=0)
        if masked:
            U   = ma.masked_array(U, mask=mask_psi )
            V   = ma.masked_array(V, mask=mask_psi )
        Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        if scaled:
            mask_ref        = np.ones(self.lonArr.shape)
            # ind             = (self.lonArr == -146.+360.)*(self.latArr == 56.5)
            # print ind
            # mask_ref[ind]   = False
            Uref            = ma.masked_array(Uref, mask=mask_ref )
            Vref            = ma.masked_array(Vref, mask=mask_ref )
            m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
            m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        plt.suptitle(str(period)+' sec', fontsize=20)
        # return amp
            
        if showfig:
            plt.show()
        return
    
    def plot_fast_axis_coverage(self, runid, period, helm=False, factor=10, normv=5., width=0.005, ampref=0.05, datatype='',
            scaled=False, masked=True, clabel='', cmap='cv', projection='lambert', hillshade=False,\
            geopolygons=None, vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        anipara         - anisotropic paramter
                            0   - isotropic
                            1   - 2 psi anisotropic
                            2   - 2&4 psi anisotropic
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        plot_vel        - plot velocity or not
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        thresh          - threhold value for Gaussian deviation to determine the mask for plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup         = self[dataid]
        # period array
        pers        = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi     = pergrp['psiarr'].value
        amp     = pergrp['amparr'].value
        mask    = pergrp['mask_aniso'].value + pergrp['mask'].value 
        # return 
        # get velocity
        try:
            data= pergrp[datatype].value
            plot_data   = True
        except:
            plot_data   = False
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        
        if scaled:
            # ampref  = amp.max()
            # print ampref
            U       = np.sin(psi/180.*np.pi)*amp/ampref/normv
            V       = np.cos(psi/180.*np.pi)*amp/ampref/normv
            Uref    = np.ones(self.lonArr.shape)*1./normv
            Vref    = np.zeros(self.lonArr.shape)
        else:
            U       = np.sin(psi/180.*np.pi)/normv
            V       = np.cos(psi/180.*np.pi)/normv
        # rotate vectors to map projection coordinates
        U, V, x, y  = m.rotate_vector(U, V, self.lonArr-360., self.latArr, returnxy=True)
        if scaled:
            Uref, Vref, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr-360., self.latArr, returnxy=True)
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if plot_data:
            if cmap == 'ses3d':
                cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                                0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
            elif cmap == 'cv':
                import pycpt
                cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
            else:
                try:
                    if os.path.isfile(cmap):
                        import pycpt
                        cmap    = pycpt.load.gmtColormap(cmap)
                except:
                    pass
            if masked:
                data     = ma.masked_array(data, mask=mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=40, rotation=0)
            # cb.outline.set_linewidth(2)
            plt.suptitle(str(period)+' sec', fontsize=20)
            cb.ax.tick_params(labelsize=40)
            cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        
        #--------------------------------------
        # plot fast axis
        #--------------------------------------
        x_psi       = x.copy()
        y_psi       = y.copy()
        mask_psi    = mask.copy()
        ##
        ##
        dset        = EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_xcorr_2deg_0.05.h5')
        covdata     = dset.plot_azi_coverage(period=period, gaptype=1, Nazi_thresh=600, showfig=False)
        mask_cover  = covdata >= 90.
        mask_psi    += mask_cover
        ##
        if factor!=None:
            x_psi   = x_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
            y_psi   = y_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
            U       = U[0:self.Nlat:factor, 0:self.Nlon:factor]
            V       = V[0:self.Nlat:factor, 0:self.Nlon:factor]
            mask_psi= mask_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
        # Q       = m.quiver(x, y, U, V, scale=30, width=0.001, headaxislength=0)
        if masked:
            U   = ma.masked_array(U, mask=mask_psi )
            V   = ma.masked_array(V, mask=mask_psi )
        Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        if scaled:
            mask_ref        = np.ones(self.lonArr.shape)
            # ind             = (self.lonArr == -146.+360.)*(self.latArr == 56.5)
            # print ind
            # mask_ref[ind]   = False
            Uref            = ma.masked_array(Uref, mask=mask_ref )
            Vref            = ma.masked_array(Vref, mask=mask_ref )
            m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
            m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        plt.suptitle(str(period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        return
    
    def plot_azi_coverage(self, period, runid=0, helm=False, Nazi_thresh=50, N_thresh=1, gaptype=0,
            scaled=False, masked=True, clabel='', cmap='cv', projection='lambert', hillshade=False,\
            geopolygons=None, vmin=None, vmax=None, showfig=True):
        
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup         = self[dataid]
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        N_bin           = ingroup.attrs['N_bin']
        minazi          = ingroup.attrs['minazi']
        maxazi          = ingroup.attrs['maxazi']
        gridx           = ingroup.attrs['gridx']
        gridy           = ingroup.attrs['gridy']
        
        d_bin           = float((maxazi-minazi)/N_bin)
        azArr           = np.arange(N_bin)*d_bin + minazi
        if Nazi_thresh is None:
            Nazi_thresh   = N_thresh*gridx*gridy
            # Nazi_thresh   = 50
        # period array
        pers        = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        histArr = pergrp['histArr'].value
        mask    = pergrp['mask_aniso'].value + pergrp['mask'].value 


        index_thresh        = histArr>=Nazi_thresh
        total_gap           = np.zeros(self.lonArr.shape)
        total_gap[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                            = 360. - index_thresh.sum(axis=0) * d_bin
        gap_1               = np.zeros(total_gap.shape)
        gap_2               = np.zeros(total_gap.shape)
        g1, g2              = _get_gaps(index_thresh)
        gap_1[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                            = g1* d_bin
        gap_2[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                            = g2* d_bin
        if gaptype == 0:
           data = total_gap
        elif gaptype == 1:
            data = gap_1
        else:
            data = gap_2
        if not showfig:
            return data
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)        
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        yakutat_slb_dat     = np.loadtxt('YAK_extent.txt')
        yatlons             = yakutat_slb_dat[:, 0]
        yatlats             = yakutat_slb_dat[:, 1]
        xyat, yyat          = m(yatlons, yatlats)
        m.plot(xyat, yyat, lw = 5, color='black')
        m.plot(xyat, yyat, lw = 3, color='white')
        #--------------------------------------
        # plot isotropic velocity
        #--------------------------------------
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        if masked:
            data     = ma.masked_array(data, mask=mask )
        x, y        = m(self.lonArr, self.latArr)
        im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=40, rotation=0)
        # cb.outline.set_linewidth(2)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        cb.set_alpha(1)
        cb.draw_all()
        cb.solids.set_edgecolor("face")
        plt.suptitle(str(gaptype)+' type', fontsize=20)
            
        if showfig:
            plt.show()
        return
    
    def hist_fast_axis(self, inh5fname, runid, period, helm=False, factor=10, normv=5., width=0.005, ampref=0.05, datatype='',
            scaled=False, masked=True, clabel='', cmap='cv', projection='lambert', hillshade=False,\
            geopolygons=None, vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        anipara         - anisotropic paramter
                            0   - isotropic
                            1   - 2 psi anisotropic
                            2   - 2&4 psi anisotropic
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        plot_vel        - plot velocity or not
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        thresh          - threhold value for Gaussian deviation to determine the mask for plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup         = self[dataid]
        # period array
        pers        = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi     = pergrp['psiarr'].value
        amp     = pergrp['amparr'].value
        mask    = pergrp['mask_aniso'].value + pergrp['mask'].value
        
        dset    = h5py.File(inh5fname)
        ingroup2= dset[dataid]
        # period array
        pers        = dset.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp2 = ingroup2['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi2    = pergrp2['psiarr'].value
        amp2    = pergrp2['amparr'].value
        mask2   = pergrp2['mask_aniso'].value + pergrp['mask'].value
        
        
        data    = abs(psi-psi2)
        # data    = data[np.logical_not(mask+mask2)]
        
        covdata = self.plot_azi_coverage(period=period, gaptype=1, Nazi_thresh=400, showfig=False) 
        mask3   = covdata >=90.
        data    = data[np.logical_not(mask + mask2+mask3)]
        
        
        ax      = plt.subplot()
        
        data[data>90]   = 180. - data[data>90]
        dbin    = 10.
        bins    = np.arange(min(data), max(data) + dbin, dbin)
        
        weights = np.ones_like(data)/float(data.size)
        plt.hist(data, bins=bins, weights = weights)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        good_per= float(data[data<30.].size)/float(data.size)
        plt.ylabel('Percentage (%)', fontsize=60)
        plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
        plt.title('mean = %g , std = %g, good = %g' %(data.mean(), data.std(), good_per*100.) + '%', fontsize=30)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim([0, 90.])
        plt.show()

        if showfig:
            plt.show()
        return
    
    def diff_fast_axis(self, inh5fname, runid, period, helm=False, factor=10, normv=5., width=0.005, ampref=0.05, datatype='',
            scaled=False, masked=True, clabel='', cmap='cv', projection='lambert', hillshade=False,\
            geopolygons=None, vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        anipara         - anisotropic paramter
                            0   - isotropic
                            1   - 2 psi anisotropic
                            2   - 2&4 psi anisotropic
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        plot_vel        - plot velocity or not
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        thresh          - threhold value for Gaussian deviation to determine the mask for plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup         = self[dataid]
        # period array
        pers        = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi     = pergrp['psiarr'].value
        amp     = pergrp['amparr'].value
        mask    = pergrp['mask_aniso'].value + pergrp['mask'].value
        
        dset    = h5py.File(inh5fname)
        ingroup2= dset[dataid]
        # period array
        pers        = dset.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp2 = ingroup2['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi2    = pergrp2['psiarr'].value
        amp2    = pergrp2['amparr'].value
        mask2   = pergrp2['mask_aniso'].value + pergrp['mask'].value
        
        
        diffdata    = abs(psi-psi2)
        diffdata[diffdata>90.]  = 180. - diffdata[diffdata>90.]
        mdata       = ma.masked_array(diffdata, mask=mask + mask2 )
        ###
        np.savez('diffazi.npz', diffdata, mask + mask2)
        ##
        # tempdset= EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_xcorr_2deg_0.05.h5')
        # # # covdata = self.plot_azi_coverage(period=period, gaptype=1, Nazi_thresh=600, showfig=False) 
        # # # mask3   = covdata >=90.
        # # # data    = diffdata[np.logical_not(mask + mask2+mask3)]
        ##
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
                ################################
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=30, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        cb.solids.set_edgecolor("face")
        ###
        if np.any(diffdata < 0.):
            negative        = diffdata < 0.
            negative       = ma.masked_array(negative, mask=mask + inmask )
            m.contour(x, y, negative, colors='w', lw=3)
        ###
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        plt.show()
        return
    
    def combine_fast_axis(self, inh5fname, runid, period, helm=False, factor=10, normv=5., width=0.005, ampref=0.05, datatype='',
            scaled=False, masked=True, clabel='', cmap='cv', projection='lambert', hillshade=False,\
            geopolygons=None, vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        anipara         - anisotropic paramter
                            0   - isotropic
                            1   - 2 psi anisotropic
                            2   - 2&4 psi anisotropic
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        plot_vel        - plot velocity or not
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        thresh          - threhold value for Gaussian deviation to determine the mask for plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup         = self[dataid]
        # period array
        pers        = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        unpsi   = pergrp['unpsi'].value
        amp     = pergrp['amparr'].value
        mask    = pergrp['mask_aniso'].value + pergrp['mask'].value
        
        dset    = h5py.File(inh5fname)
        ingroup2= dset[dataid]
        # period array
        pers        = dset.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp2 = ingroup2['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        unpsi2  = pergrp2['unpsi'].value
        amp2    = pergrp2['amparr'].value
        mask2   = pergrp2['mask_aniso'].value + pergrp['mask'].value
        
        semfactor = 3.5
        data    = np.sqrt(unpsi**2 + unpsi2**2)*semfactor
        data[data>90.]  = 90.
        
        ###
        np.savez('unazi.npz', data, mask + mask2)
        ###
        # data    = unpsi
        mdata   = ma.masked_array(data, mask=mask + mask2 )
        ##
        # tempdset= EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_xcorr_2deg_0.05.h5')
        # # # covdata = self.plot_azi_coverage(period=period, gaptype=1, Nazi_thresh=600, showfig=False) 
        # # # mask3   = covdata >=90.
        # # # data    = diffdata[np.logical_not(mask + mask2+mask3)]
        ##
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
                ################################
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=30, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        cb.solids.set_edgecolor("face")
        ###
        # # # if np.any(diffdata < 0.):
        # # #     negative        = diffdata < 0.
        # # #     negative       = ma.masked_array(negative, mask=mask + inmask )
        # # #     m.contour(x, y, negative, colors='w', lw=3)
        ###
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        plt.show()
        return
    
    def diff_amp(self, inh5fname, runid, period, helm=False, factor=10, normv=5., width=0.005, ampref=0.05, datatype='',
            scaled=False, masked=True, clabel='', cmap='cv', projection='lambert', hillshade=False,\
            geopolygons=None, vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        anipara         - anisotropic paramter
                            0   - isotropic
                            1   - 2 psi anisotropic
                            2   - 2&4 psi anisotropic
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        plot_vel        - plot velocity or not
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        thresh          - threhold value for Gaussian deviation to determine the mask for plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup         = self[dataid]
        # period array
        pers        = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi     = pergrp['psiarr'].value
        amp     = pergrp['amparr'].value
        mask    = pergrp['mask_aniso'].value + pergrp['mask'].value
        
        dset    = h5py.File(inh5fname)
        ingroup2= dset[dataid]
        # period array
        pers        = dset.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp2 = ingroup2['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi2    = pergrp2['psiarr'].value
        amp2    = pergrp2['amparr'].value
        mask2   = pergrp2['mask_aniso'].value + pergrp['mask'].value
        
        
        # diffdata    = abs(amp/amp2)
        
        diffdata    = abs(amp-amp2) # /(amp+amp2)
        mdata       = ma.masked_array(diffdata, mask=mask + mask2 )
        
        ###
        np.savez('diffamp.npz', diffdata, mask + mask2)
        ##
        # tempdset= EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_xcorr_2deg_0.05.h5')
        # # # covdata = self.plot_azi_coverage(period=period, gaptype=1, Nazi_thresh=600, showfig=False) 
        # # # mask3   = covdata >=90.
        # # # data    = diffdata[np.logical_not(mask + mask2+mask3)]
        ##
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
                ################################
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=30, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        cb.solids.set_edgecolor("face")
        ###
        # if np.any(diffdata < 0.):
        #     negative        = diffdata < 0.
        #     negative       = ma.masked_array(negative, mask=mask + inmask )
        #     m.contour(x, y, negative, colors='w', lw=3)
        ###
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        plt.show()
        return
    
    def combine_amp(self, inh5fname, runid, period, helm=False, factor=10, normv=5., width=0.005, ampref=0.05, datatype='',
            scaled=False, masked=True, clabel='', cmap='cv', projection='lambert', hillshade=False,\
            geopolygons=None, vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        anipara         - anisotropic paramter
                            0   - isotropic
                            1   - 2 psi anisotropic
                            2   - 2&4 psi anisotropic
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        plot_vel        - plot velocity or not
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        thresh          - threhold value for Gaussian deviation to determine the mask for plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup         = self[dataid]
        # period array
        pers        = self.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        amp   = pergrp['amparr'].value
        unamp     = pergrp['unamp'].value
        mask    = pergrp['mask_aniso'].value + pergrp['mask'].value
        
        dset    = h5py.File(inh5fname)
        ingroup2= dset[dataid]
        # period array
        pers        = dset.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp2 = ingroup2['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        amp2  = pergrp2['amparr'].value
        unamp2    = pergrp2['unamp'].value
        mask2   = pergrp2['mask_aniso'].value + pergrp['mask'].value
        
        unamp[(amp==0.)*(amp2==0.)]  = 0.
        unamp2[(amp==0.)*(amp2==0.)]  = 0.
        amp[(amp==0.)*(amp2==0.)]  = 1.
        
        from uncertainties import unumpy
        tamp    =  unumpy.uarray(amp, unamp)
        tamp2   = unumpy.uarray(amp2, unamp2)
        semfactor = 2.
        # data    = np.sqrt(unamp**2 + unamp2**2)*semfactor
        
        temp    = (tamp-tamp2)
        data    = unumpy.std_devs(temp) * semfactor
        
        ###
        np.savez('unamp.npz', data, mask + mask2)
        ###
        # data    = unpsi
        mdata   = ma.masked_array(data, mask=mask + mask2 )
        ##
        # tempdset= EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_xcorr_2deg_0.05.h5')
        # # # covdata = self.plot_azi_coverage(period=period, gaptype=1, Nazi_thresh=600, showfig=False) 
        # # # mask3   = covdata >=90.
        # # # data    = diffdata[np.logical_not(mask + mask2+mask3)]
        ##
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
                ################################
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=30, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        cb.solids.set_edgecolor("face")
        ###
        # # # if np.any(diffdata < 0.):
        # # #     negative        = diffdata < 0.
        # # #     negative       = ma.masked_array(negative, mask=mask + inmask )
        # # #     m.contour(x, y, negative, colors='w', lw=3)
        ###
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        plt.show()
        return
      
    def compare_raytomo(self, inraytomofname, rayruntype, rayrunid, runid, period, showfig=True, projection='lambert', cmap='cv', clabel='C (km/s)'):
        """
        compare the eikonal tomography results with the ray tomography
        """
        # raytomo data
        dset_ray    = h5py.File(inraytomofname)
        rundict     = {0: 'smooth_run', 1: 'qc_run'}
        dataid      = rundict[rayruntype]+'_'+str(rayrunid)
        ingroup     = dset_ray['reshaped_'+dataid]
        pers        = dset_ray.attrs['period_array']
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the raytomo database')
        if rayruntype == 1:
            isotropic   = ingroup.attrs['isotropic']
        else:
            isotropic   = True
        pergrp  = ingroup['%g_sec'%( period )]
        if isotropic:
            datatype    = 'velocity'
        else:
            datatype    = 'vel_iso'
        raydata     = pergrp[datatype].value
        raymask     = ingroup['mask1']
        # Eikonal data
        dataid      = 'Eikonal_stack_'+str(runid)
        ingroup     = self[dataid]
        pergrp      = ingroup['%g_sec'%( period )]
        datatype    = 'vel_iso'
        data        = pergrp[datatype].value
        mask        = pergrp['mask'].value
        #
        self._get_lon_lat_arr()
        diffdata    = raydata - data
        mdata       = ma.masked_array(diffdata, mask=mask + raymask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
                ################################
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=-0.2, vmax=0.2)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=30, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=20)
        cb.solids.set_edgecolor("face")
        
        plt.show()
        
        ax      = plt.subplot()
        data    = diffdata[np.logical_not(mask + raymask)]
        plt.hist(data, bins=100, normed=True)
        outstd  = data.std()
        outmean = data.mean()
        # compute mad
        from statsmodels import robust
        mad     = robust.mad(data)
        plt.xlim(-.2, .2)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=30)
        plt.xlabel('Differences (km/sec)', fontsize=30)
        plt.title(str(period)+' sec, mean = %g m/s, std = %g m/s, mad = %g m/s' %(outmean*1000., outstd*1000., mad*1000.), fontsize=30)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        if showfig:
            plt.show()
        
    def compare_eiktomo(self, ineiktomofname, inrunid, runid, period, Nmeasure=None, helm=False, \
                showfig=True, projection='lambert', cmap='cv', clabel='C (km/s)', vmin=None, vmax=None):
        """
        compare the eikonal tomography results with the another eikonal tomography
        """
        # input eikonal data
        dset_in     = h5py.File(ineiktomofname)
        dataid      = 'merged_tomo_'+str(inrunid)
        ingroup     = dset_in[dataid]
        pergrp      = ingroup['%g_sec'%( period )]
        datatype    = 'vel_iso'
        indata      = pergrp[datatype].value
        inmask      = pergrp['mask'].value
        # Nm_in       = np.zeros(indata.shape)
        # Nm_in[1:-1, 1:-1] \
        #             = pergrp['NmeasureQC'].value
        # Eikonal data
        if helm:
            dataid  = 'merged_tomo_'+str(runid)
        else:
            dataid  = 'merged_tomo_'+str(runid)
        ingroup     = self[dataid]
        pergrp      = ingroup['%g_sec'%( period )]
        datatype    = 'vel_iso'
        data        = pergrp[datatype].value
        mask        = pergrp['mask'].value
        Nm          = np.zeros(indata.shape)
        # Nm[1:-1, 1:-1] \
        #             = pergrp['NmeasureQC'].value
        #
        # # # dataid  = 'Eikonal_stack_'+str(runid)
        # # # ingroup     = self[dataid]
        # # # pergrp      = ingroup['%g_sec'%( period )]
        # # # datatype    = 'vel_iso'
        # # # data        = pergrp[datatype].value
        #
        self._get_lon_lat_arr()
        diffdata    = indata - data
        # Nm_mask     = np.zeros(data.shape, dtype=bool)
        # if Nmeasure is not None:
        #     Nm_mask += Nm_in < Nmeasure
        #     Nm_mask += Nm < Nmeasure
        # mdata       = ma.masked_array(diffdata, mask=mask + inmask + Nm_mask )
        
        
        mdata       = ma.masked_array(diffdata, mask=mask + inmask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
                ################################
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=30, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        cb.solids.set_edgecolor("face")
        ###
        if np.any(diffdata < 0.):
            negative        = diffdata < 0.
            negative       = ma.masked_array(negative, mask=mask + inmask )
            m.contour(x, y, negative, colors='w', lw=3)
        ###
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        plt.show()
        
    def compare_eik_helm(self, period, eikrunid=0, helmrunid=0, Nmeasure=None, showfig=True,\
                         projection='lambert', cmap='cv', clabel='C (km/s)'):
        """
        compare the eikonal tomography results with the Helmholtz eikonal resultz
        """
        # eikonal data
        dataid      = 'Eikonal_stack_'+str(eikrunid)
        group_eik   = self[dataid]
        pergrp      = group_eik['%g_sec'%( period )]
        datatype    = 'vel_iso'
        data_eik    = pergrp[datatype].value
        mask_eik    = pergrp['mask'].value
        Nm_eik      = np.zeros(data_eik.shape)
        Nm_eik[1:-1, 1:-1] \
                    = pergrp['NmeasureQC'].value
        # Helmholtz data
        dataid      = 'Helmholtz_stack_'+str(helmrunid)
        group_helm  = self[dataid]
        pergrp      = group_helm['%g_sec'%( period )]
        datatype    = 'vel_iso'
        data_helm   = pergrp[datatype].value
        mask_helm   = pergrp['mask'].value
        Nm_helm     = np.zeros(data_helm.shape)
        Nm_helm[1:-1, 1:-1] \
                    = pergrp['NmeasureQC'].value
        self._get_lon_lat_arr()
        diffdata    = data_eik - data_helm
        Nm_mask     = np.zeros(data_helm.shape, dtype=bool)
        if Nmeasure is not None:
            Nm_mask += Nm_eik < Nmeasure
            Nm_mask += Nm_helm < Nmeasure
        mdata       = ma.masked_array(diffdata, mask= mask_helm + mask_eik + Nm_mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
                ################################
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=-0.2, vmax=0.2)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=30, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=20)
        cb.solids.set_edgecolor("face")
        plt.show()
        
        ax      = plt.subplot()
        data    = diffdata[np.logical_not(mask_helm + mask_eik + Nm_mask)]
        plt.hist(data, bins=100, normed=True)
        outstd  = data.std()
        outmean = data.mean()
        # compute mad
        from statsmodels import robust
        mad     = robust.mad(data)
        plt.xlim(-.2, .2)
        import matplotlib.mlab as mlab
        from matplotlib.ticker import FuncFormatter
        plt.ylabel('Percentage (%)', fontsize=30)
        plt.xlabel('Differences (km/sec)', fontsize=30)
        plt.title(str(period)+' sec, mean = %g m/s, std = %g m/s, mad = %g m/s' %(outmean*1000., outstd*1000., mad*1000.), fontsize=30)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        if showfig:
            plt.show()
        
    def plot_az(self, runid, period, iev, clabel='azimuth (deg)', cmap='cv', projection='lambert',\
                hillshade=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
        
        group               = self['Eikonal_run_'+str(runid)]
        per_group           = group['%g_sec'%( period )]
        self._get_lon_lat_arr()
        # Nevent      = len(per_group.keys())
        # #-----------------------------------------------------
        # # Loop over events to get eikonal maps for each event
        # #-----------------------------------------------------
        # for iev in range(Nevent):
        #     evid                = per_group.keys()[iev]
        #     event_group         = per_group[evid]
        #     az                  = event_group['az'].value
        #     reason_n            = event_group['reason_n'].value
        #     valid               = np.where(reason_n != 0)[0]
        #     print evid, valid.size
        # return
        evid                = per_group.keys()[iev]
        event_group         = per_group[evid]
        az                  = event_group['az'].value
        reason_n            = event_group['reason_n'].value
        data                = np.zeros(self.lonArr.shape)
        mask                = np.ones(self.lonArr.shape, dtype=bool)
        data[1:-1, 1:-1]    = az
        mask[1:-1, 1:-1]    = reason_n != 0
        mdata               = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr, self.latArr)
        # shapefname  = '/home/leon/geological_maps/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=20, rotation=0)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=15)
        # print 'plotting data from '+dataid
        if showfig:
            plt.show()

    def plot_travel_time(self, inasdffname, netcode, stacode, period, channel='ZZ', mindp=50):
        minlon              = self.attrs['minlon']
        maxlon              = self.attrs['maxlon']
        minlat              = self.attrs['minlat']
        maxlat              = self.attrs['maxlat']
        dlon                = self.attrs['dlon']
        dlat                = self.attrs['dlat']
        nlat_grad           = self.attrs['nlat_grad']
        nlon_grad           = self.attrs['nlon_grad']
        nlat_lplc           = self.attrs['nlat_lplc']
        nlon_lplc           = self.attrs['nlon_lplc']
        inDbase             = pyasdf.ASDFDataSet(inasdffname)
        try:
            data            = inDbase.auxiliary_data['FieldDISPpmf2interp'][netcode][stacode][channel][str(int(period))+'sec'].data.value
        except KeyError:
            print 'No data!'
            return
        lons        = data[:, 0]
        lats        = data[:, 1]
        C           = data[:, 2]
        dist        = data[:, 5]
        T           = dist/C
        coordinates = inDbase.waveforms[netcode+'.'+stacode].coordinates
        evlo        = coordinates['longitude']
        evla        = coordinates['latitude']
        field2d     = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=period, evlo=evlo, evla=evla, fieldtype='Tph',\
                                nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
        if not _check_station_distribution(lons, lats, np.int32(mindp/2.)):
            print 'Travel time related to this station will be discarded!'
        field2d.read_array(lonArr = lons, latArr = lats, ZarrIn = T )
        outfname        = netcode+'.'+stacode+'_Tph_'+channel+'.lst'
        field2d.interp_surface(workingdir='temp_travel_time_dir', outfname=outfname)
        if not field2d.check_curvature(workingdir='temp_travel_time_dir', outpfx=netcode+'.'+stacode+'_'+channel+'_'):
            print 'Did not pass the curvature test!'
            return
        field2d.eikonal_operator(workingdir='temp_travel_time_dir', inpfx=netcode+'.'+stacode+'_'+channel+'_')
        field2d.plot(datatype='z', title='Travel time for station: '+ netcode+'.'+stacode, contour=True, event=True, vmin=50., vmax=350.)
        field2d.plot(datatype='appV', title='Phase Speed for station: '+ netcode+'.'+stacode, contour=True, event=True, vmin=2.4, vmax=3.4)
    
    def write_un_noise(self, outfname, runid=0):
        pers        = self.attrs['period_array']
        unarr       = np.zeros(pers.size)
        i           = 0
        dataid      = 'Eikonal_stack_'+str(runid)
        grp         = self[dataid]
        for per in pers:
            try:
                pergrp          = grp['%g_sec'%( per )]
                velocity        = pergrp['vel_iso'].value
                uncertainty     = pergrp['vel_sem'].value
                mask_eik        = pergrp['mask'].value
                unarr[i]        = uncertainty[np.logical_not(mask_eik)].mean() * 2000.
            except:
                unarr[i]        = 999.
            
            i               += 1
        outArr  = np.append(pers, unarr)
        outArr  = outArr.reshape((2, pers.size))
        outArr  = outArr.T
        np.savetxt(outfname, outArr, fmt='%g')
        
    def interp_surface_azi_eik(self, workingdir='./azi_eik_interp_surface', dlon=None, dlat=None, runid=0, deletetxt=True):
        """interpolate inverted velocity maps and uncertainties to a grid for inversion of Vs
        =================================================================================================================
        ::: input parameters :::
        workingdir  - working directory
        dlon/dlat   - grid interval for interpolation
        runid       - id of run
        =================================================================================================================
        """
        self._get_lon_lat_arr()
        dataid          = 'Eikonal_stack_'+str(runid)
        pers            = self.attrs['period_array']
        grp             = self[dataid]
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon_org        = self.attrs['dlon']
        dlat_org        = self.attrs['dlat']
        if dlon is None and dlat is None:
            print 'At least one of dlon/dlat needs to be specified!'
            return
        if dlon == dlon_org and dlat == dlat_org:
            print 'No need to perform interpolation!'
            return
        self.attrs.create(name = 'dlon_interp', data=dlon)
        self.attrs.create(name = 'dlat_interp', data=dlat)
        #--------------------------------------------------
        # get the mask array for the interpolated data
        #--------------------------------------------------
        mask        = np.zeros((self.Nlat, self.Nlon), dtype=bool)
        for per in pers:
            pergrp  = grp['%g_sec'%( per )]
            mask    += pergrp['mask'].value
            mask    += pergrp['mask_aniso'].value
        # --- 
        lons            = np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon
        lats            = np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat
        Nlon            = lons.size
        Nlat            = lats.size
        lonArr, latArr  = np.meshgrid(lons, lats)
        # fill the center hole --- 
        ind             = (self.lonArr >= -150.+360.)*(self.lonArr <= -145.+360.)*\
                            (self.latArr >= 64.)*(self.latArr <= 66.)
        # # # ind             = (dset.lonArr >= -150.+360.)*(dset.lonArr <= -145.+360.)*\
        # # #                     (dset.latArr >= 64.)*(dset.latArr <= 66.)
        mask[ind]       = False
        # ---- 
        mask_interp     = _get_mask_interp(mask, self.lons, self.lats, lons, lats)
        grp.create_dataset(name = 'mask_interp', data=mask_interp)
        grp.create_dataset(name = 'mask_allT', data=mask)
        for per in pers:
            working_per = workingdir+'/'+str(per)+'sec'
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            #-------------------------------
            # get data
            #-------------------------------
            try:
                pergrp      = grp['%g_sec'%( per )]
                vel_iso     = pergrp['vel_iso'].value
                vel_sem     = pergrp['vel_sem'].value
                psiarr      = pergrp['psiarr'].value
                unpsi       = pergrp['unpsi'].value
                amparr      = pergrp['amparr'].value
                unamp       = pergrp['unamp'].value
                mask_per    = pergrp['mask'].value + pergrp['mask_aniso'].value
            except KeyError:
                print 'No data for T = '+str(per)+' sec'
                continue
            index           = np.logical_not(mask_per)
            mask_per_out    = _get_mask_interp(mask_per, self.lons, self.lats, lons, lats)
            maskinterp_dset = pergrp.create_dataset(name='mask_interp', data=mask_per_out)
            #-------------------------------
            # interpolation for velocity
            #-------------------------------
            field2d_v       = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_v.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = vel_iso[index])
            outfname        = 'interp_vel.lst'
            field2d_v.interp_surface(workingdir=working_per, outfname=outfname)
            vinterp_dset    = pergrp.create_dataset(name='vel_iso_interp', data=field2d_v.Zarr)
            del field2d_v
            #---------------------------------
            # interpolation for uncertainties
            #---------------------------------
            field2d_un      = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_un.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = vel_sem[index])
            outfname        = 'interp_un.lst'
            field2d_un.interp_surface(workingdir=working_per, outfname=outfname)
            uninterp_dset   = pergrp.create_dataset(name='vel_sem_interp', data=field2d_un.Zarr)
            del field2d_un
            #-------------------------------
            # interpolation for amp
            #-------------------------------
            field2d_amp     = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_amp.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = amparr[index])
            outfname        = 'interp_amp.lst'
            field2d_amp.interp_surface(workingdir=working_per, outfname=outfname)
            ampinterp_dset  = pergrp.create_dataset(name='amp_interp', data=field2d_amp.Zarr)
            del field2d_amp
            #-------------------------------
            # interpolation for amp uncertainties
            #-------------------------------
            field2d_unamp   = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_unamp.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = unamp[index])
            outfname        = 'interp_unamp.lst'
            field2d_unamp.interp_surface(workingdir=working_per, outfname=outfname)
            unampinterp_dset= pergrp.create_dataset(name='unamp_interp', data=field2d_unamp.Zarr)
            del field2d_unamp
            #-------------------------------
            # interpolation for psi uncertainties
            #-------------------------------
            field2d_unpsi   = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_unpsi.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = unpsi[index])
            outfname        = 'interp_unpsi.lst'
            field2d_unpsi.interp_surface(workingdir=working_per, outfname=outfname)
            unpsiinterp_dset= pergrp.create_dataset(name='unpsi_interp', data=field2d_unpsi.Zarr)
            del field2d_unpsi
            #-------------------------------
            # interpolation for psi
            #-------------------------------
            low         = 0.
            high        = 180.
            temp_psi    = psiarr[index]
            temp_psi[temp_psi<0.] \
                        += 180.
            ang         = (temp_psi - low)*2.*np.pi / (high - low)
            # interpolating sine
            Sarr        = np.sin(ang)
            field2d_S   = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_S.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = Sarr)
            outfname        = 'interp_S.lst'
            field2d_S.interp_surface(workingdir=working_per, outfname=outfname)
            Sarr_interp = field2d_S.Zarr
            # interpolating cosine
            Carr        = np.cos(ang)
            field2d_C   = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_C.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = Carr)
            outfname        = 'interp_C.lst'
            field2d_C.interp_surface(workingdir=working_per, outfname=outfname)
            Carr_interp = field2d_C.Zarr
            # convert to interpolated psi
            res             = np.arctan2(Sarr_interp, Carr_interp)
            res[res<0.]     += 2*np.pi
            psi_interp      = res*(high - low)/2.0/np.pi + low
            psiinterp_dset  = pergrp.create_dataset(name='psi_interp', data=psi_interp)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def plot_interp(self, runid, datatype, period, semfactor=2., Nthresh=None, helm=False, merged=False, clabel='', cmap='cv', projection='lambert',\
                hillshade=False, geopolygons=None, vmin=None, vmax=None, showfig=True, mfault=True, v_rel=None):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runtype         - type of run (0 - smooth run, 1 - quality controlled run)
        runid           - id of run
        datatype        - datatype for plotting
        period          - period of data
        sem_factor      - factor multiplied to get the finalized uncertainties
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        if helm:
            dataid      = 'Helmholtz_stack_'+str(runid)
        else:
            dataid      = 'Eikonal_stack_'+str(runid)
        if merged:
            dataid      = 'merged_tomo_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        self._get_lon_lat_arr_interp()
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        try:
            data        = pergrp[datatype].value
            if data.shape != self.lonArr.shape:
                print 'Incompatible data shape: '+str(data.shape)+' lonArr/latArr:'+str(self.lonArr.shape)
                return
        except:
            outstr      = ''
            for key in pergrp.keys():
                outstr  +=key
                outstr  +=', '
            outstr      = outstr[:-1]
            raise KeyError('Unexpected datatype: '+datatype+\
                           ', available datatypes are: '+outstr)
        if datatype == 'vel_sem_interp':
            data    *= 1000.*semfactor
        mask        = ingroup['mask_interp'].value
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr, self.latArr)
        
        plot_fault_lines(m, 'AK_Faults.txt', color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
                    cmap    = cmap.reversed()
            except:
                pass
        im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.])
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
        # cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.0, 4.1, 4.2, 4.3, 4.4])
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=40, rotation=0)
        # cb.outline.set_linewidth(2)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize=40)
        print 'plotting data from '+dataid
        if showfig:
            plt.show()
        return
    
    
    
def eikonal4mp(infield, workingdir, channel, cdist):
    working_per     = workingdir+'/'+str(infield.period)+'sec'
    outfname        = infield.evid+'_'+infield.fieldtype+'_'+channel+'.lst'
    infield.interp_surface(workingdir=working_per, outfname=outfname)
    if not infield.check_curvature(workingdir=working_per, outpfx=infield.evid+'_'+channel+'_'):
        return
    infield.eikonal_operator(workingdir=working_per, inpfx=infield.evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)
    outfname_npz    = working_per+'/'+infield.evid+'_field2d'
    infield.write_binary(outfname=outfname_npz)
    return

def helmhotz4mp(infieldpair, workingdir, channel, amplplc, cdist):
    tfield          = infieldpair[0]
    working_per     = workingdir+'/'+str(tfield.period)+'sec'
    outfname        = tfield.evid+'_'+tfield.fieldtype+'_'+channel+'.lst'
    tfield.interp_surface(workingdir=working_per, outfname=outfname)
    if not tfield.check_curvature(workingdir=working_per, outpfx=tfield.evid+'_'+channel+'_'):
        return
    tfield.eikonal_operator(workingdir=working_per, inpfx=tfield.evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)
    outfname_npz    = working_per+'/'+tfield.evid+'_field2d'
    if amplplc:
        field2dAmp          = infieldpair[1]
        outfnameAmp         = field2dAmp.evid+'_Amp_'+channel+'.lst'
        field2dAmp.interp_surface(workingdir = working_per, outfname = outfnameAmp)
        if not field2dAmp.check_curvature_amp(workingdir = working_per, outpfx = field2dAmp.evid+'_Amp_'+channel+'_', threshold = 0.5):
            return
        field2dAmp.helmholtz_operator(workingdir = working_per, inpfx = field2dAmp.evid+'_Amp_'+channel+'_', lplcthresh = 0.5)
        tfield.get_lplc_amp(fieldamp = field2dAmp)
    tfield.write_binary(outfname = outfname_npz, amplplc = amplplc)
    return

def stack4mp(sdata, workingdir, minazi, maxazi, N_bin, threshmeasure, anisotropic, spacing_ani, coverage,\
        use_numba, azi_amp_tresh, gridx, gridy,  Nlat, Nlon, nlat_grad, nlon_grad, enhanced=True):
    # reading data
    #----------------------------
    # isotropic stacking
    #----------------------------
    print '*** Stacking data: '+str(sdata.period)
    slownessALL = sdata.slownessALL
    reason_nALL = sdata.reason_nALL
    aziALL      = sdata.aziALL 
    Nmeasure    = sdata.Nmeasure
    Nevent      = sdata.Nevent 
    period      = sdata.period
    ###
    weightALL   = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
    validALL    = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
    if Nmeasure.max()< threshmeasure:
        print ('--- No enough measurements for: '+str(sdata.period)+' sec')
        return
    # discard grid points where number of raw measurements is low, added Sep 26th, 2018
    index_discard                   = Nmeasure < 50
    reason_nALL[:, index_discard]   = 10
    #-----------------------------------------------
    # Get weight for each grid point per event
    #-----------------------------------------------
    if use_numba:
        validALL[reason_nALL==0]    = 1
        weightALL                   = _get_azi_weight(aziALL, validALL)
        weightALL[reason_nALL!=0]   = 0
        weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
        weightsum                   = np.sum(weightALL, axis=0)
    else:
        azi_event1                  = np.broadcast_to(aziALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
        azi_event2                  = np.swapaxes(azi_event1, 0, 1)
        validALL[reason_nALL==0]    = 1
        validALL4                   = np.broadcast_to(validALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
        # use numexpr for very large array manipulations
        del_aziALL                  = numexpr.evaluate('abs(azi_event1-azi_event2)')
        index_azi                   = numexpr.evaluate('(1*(del_aziALL<20)+1*(del_aziALL>340))*validALL4')
        weightALL                   = numexpr.evaluate('sum(index_azi, 0)')
        weightALL[reason_nALL!=0]   = 0
        weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
        weightsum                   = np.sum(weightALL, axis=0)
    #-----------------------------------------------
    # reduce large weight to some value.
    #-----------------------------------------------
    avgArr                          = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
    avgArr[Nmeasure!=0]             = weightsum[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
    # bug fixed, 02/07/2018
    signALL                         = weightALL.copy()
    signALL[signALL!=0]             = 1.
    stdArr                          = np.sum( signALL*(weightALL-avgArr)**2, axis=0)
    stdArr[Nmeasure!=0]             = stdArr[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
    stdArr                          = np.sqrt(stdArr)
    threshhold                      = np.broadcast_to(avgArr+3.*stdArr, weightALL.shape)
    weightALL[weightALL>threshhold] = threshhold[weightALL>threshhold] # threshhold truncated weightALL
    # recompute weight arrays after large weight value reduction
    weightsum                       = np.sum(weightALL, axis=0)
    weightsumALL                    = np.broadcast_to(weightsum, weightALL.shape)
    # weight over all events, note that before this, weightALL is weight over events in azimuth bin
    weightALL[weightsumALL!=0]      = weightALL[weightsumALL!=0]/weightsumALL[weightsumALL!=0] 
    ###
    weightALL[weightALL==1.]        = 0. # data will be discarded if no other data within 20 degree
    #-----------------------------------------------
    # Compute mean/std of slowness
    #-----------------------------------------------
    slownessALL2                    = slownessALL*weightALL
    slowness_sum                    = np.sum(slownessALL2, axis=0)
    slowness_sumALL                 = np.broadcast_to(slowness_sum, weightALL.shape)
    # weighted standard deviation
    # formula: https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    signALL                         = weightALL.copy()
    signALL[signALL!=0]             = 1.
    MArr                            = np.sum(signALL, axis=0)
    temp                            = weightALL*(slownessALL-slowness_sumALL)**2
    temp                            = np.sum(temp, axis=0)
    slowness_std                    = np.zeros(temp.shape)
    tind                            = (weightsum!=0)*(MArr!=1)*(MArr!=0)
    slowness_std[tind]              = np.sqrt(temp[tind]/ ( weightsum[tind]*(MArr[tind]-1)/MArr[tind] ) )
    slowness_stdALL                 = np.broadcast_to(slowness_std, weightALL.shape)
    #-----------------------------------------------
    # discard outliers of slowness
    #-----------------------------------------------
    weightALLQC                     = weightALL.copy()
    index_outlier                   = (np.abs(slownessALL-slowness_sumALL))>2.*slowness_stdALL
    index_outlier                   += reason_nALL != 0
    weightALLQC[index_outlier]      = 0
    weightsumQC                     = np.sum(weightALLQC, axis=0)
    NmALL                           = np.sign(weightALLQC)
    NmeasureQC                      = np.sum(NmALL, axis=0)
    weightsumQCALL                  = np.broadcast_to(weightsumQC, weightALL.shape)
    weightALLQC[weightsumQCALL!=0]  = weightALLQC[weightsumQCALL!=0]/weightsumQCALL[weightsumQCALL!=0]
    temp                            = weightALLQC*slownessALL
    slowness_sumQC                  = np.sum(temp, axis=0)
    # new
    signALLQC                       = weightALLQC.copy()
    signALLQC[signALLQC!=0]         = 1.
    MArrQC                          = np.sum(signALLQC, axis=0)
    temp                            = weightALLQC*(slownessALL-slowness_sumQC)**2
    temp                            = np.sum(temp, axis=0)
    slowness_stdQC                  = np.zeros(temp.shape)
    tind                            = (weightsumQC!=0)*(MArrQC!=1)
    slowness_stdQC[tind]            = np.sqrt(temp[tind]/ ( weightsumQC[tind]*(MArrQC[tind]-1)/MArrQC[tind] ))
    #---------------------------------------------------------------
    # mask, velocity, and sem arrays of shape Nlat, Nlon
    #---------------------------------------------------------------
    mask                            = np.ones((Nlat, Nlon), dtype=np.bool)
    tempmask                        = (weightsumQC == 0)
    mask[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                                    = tempmask
    vel_iso                         = np.zeros((Nlat, Nlon), dtype=np.float32)
    tempvel                         = slowness_sumQC.copy()
    tempvel[tempvel!=0]             = 1./ tempvel[tempvel!=0]
    vel_iso[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                    = tempvel
    #----------------------------------------------------------------------------------------
    # standard error of the mean, updated on 09/20/2018
    # formula: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Statistical_properties
    #----------------------------------------------------------------------------------------
    slownessALL_temp                = slownessALL.copy()
    slownessALL_temp[slownessALL_temp==0.]\
                                    = 0.3
    if np.any(weightALLQC[slownessALL==0.]> 0.):
        raise ValueError('Check weight array!')
    temp                            = (weightALLQC*(1./slownessALL_temp-tempvel))**2
    temp                            = np.sum(temp, axis=0)
    tempsem                         = np.zeros(temp.shape)
    tind                            = (weightsumQC!=0)*(MArrQC!=1)
    tempsem[tind]                   = np.sqrt( temp[tind] * ( MArrQC[tind]/(weightsumQC[tind])**2/(MArrQC[tind]-1) ) ) 
    vel_sem                         = np.zeros((Nlat, Nlon), dtype=np.float32)
    vel_sem[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                    = tempsem
    #---------------------------------------
    # save isotropic velocity to npy files
    #---------------------------------------
    pfx         = '%g_sec'%( sdata.period )
    outfname    = workingdir+'/iso_'+pfx+'.npz'
    np.savez(outfname, slowness_sumQC, slowness_stdQC, Nmeasure, NmeasureQC, mask, vel_iso, vel_sem)
    #----------------------------------------------------------------------------
    # determine anisotropic parameters, need benchmark and further verification
    #----------------------------------------------------------------------------
    if anisotropic:
        print '*** Anisotropic stacking data: '+str(sdata.period)
        # quality control
        slowness_sumQC_ALL          = np.broadcast_to(slowness_sumQC, slownessALL.shape)
        diff_slowness               = np.abs(slownessALL-slowness_sumQC_ALL)
        ind_nonzero                 = slowness_sumQC_ALL!= 0.
        diff_slowness[ind_nonzero]  = diff_slowness[ind_nonzero]/slowness_sumQC_ALL[ind_nonzero]
        index_outlier               += diff_slowness > azi_amp_tresh
        # stacking to get anisotropic parameters
        dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni    \
                                    = _anisotropic_stacking(np.int64(gridx), np.int64(gridy), np.float32(maxazi), np.float32(minazi),\
                                                np.int64(N_bin), np.float64(Nmeasure), np.float64(aziALL),\
                                                np.float64(slowness_sumQC), np.float64(slownessALL), index_outlier.astype(bool))
        if enhanced:
            print '*** Enhanced anisotropic stacking data: '+str(sdata.period) 
            Nbinarr     = np.zeros((sdata.Nlat, sdata.Nlon))
            Nmarr       = np.zeros((sdata.Nlat, sdata.Nlon))
            Nlat        = sdata.Nlat
            Nlon        = sdata.Nlon
            nlat_grad   = sdata.nlat_grad
            nlon_grad   = sdata.nlon_grad
            # arrays
            # --- 
            out_dslow_sum_ani   = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            out_dslow_sum_ani[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = dslow_sum_ani
            slowAni             = out_dslow_sum_ani + slowness_sumQC
            # ---            
            velAnisem           = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            velAnisem[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = vel_un
            # ---
            slowness            = slowness_sumQC
            # --- 
            out_histArr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            out_histArr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = histArr
            histArr             = out_histArr
            # 
            Ntotal_thresh       = 50.*gridx*gridy/2.
            N_thresh            = 10.
            azArr               = np.mgrid[minazi:maxazi:N_bin*1j]
            for ilat in range(Nlat):
                if (ilat-nlat_grad) < 0 or (ilat+nlat_grad) > (Nlat - 1) :
                    continue
                for ilon in range(Nlon):
                    if (ilon-nlon_grad) < 0 or (ilon+nlon_grad) > (Nlon - 1) :
                        continue
                    outslowness = slowAni[:, ilat - nlat_grad, ilon - nlon_grad]
                    outvel_sem  = velAnisem[:, ilat - nlat_grad, ilon - nlon_grad]
                    avg_slowness= slowness[ilat - nlat_grad, ilon - nlon_grad]
                    out_hist    = histArr[:, ilat - nlat_grad, ilon - nlon_grad]
                    if out_hist.sum() < Ntotal_thresh:
                        continue
                    # get the valid binned data
                    # quality control
                    index       = np.where((outvel_sem != 0)*(out_hist > N_thresh ))[0]
                    outslowness = outslowness[index]
                    az_grd      = azArr[index]
                    outvel_sem  = outvel_sem[index]
                    Nbin        = index.size
                    if Nbin < 5:
                        continue
                    Nmarr[ilat, ilon]   = int(out_hist.sum()/(gridx*gridy))
                    Nbinarr[ilat, ilon] = Nbin
                    #--------------------
                    # sinusoidal fitting
                    #--------------------
                    indat           = (1./outslowness).reshape(1, Nbin)
                    U               = np.zeros((Nbin, Nbin), dtype=np.float64)
                    np.fill_diagonal(U, 1./outvel_sem)
                    # construct forward operator matrix
                    tG              = np.ones((Nbin, 1), dtype=np.float64)
                    G               = tG.copy()
                    # convert azimuth to the 'real' azimuth coordinate
                    az_grd          += 180.
                    az_grd          = 360. - az_grd
                    az_grd          -= 90.
                    az_grd[az_grd<0.]\
                                    += 360.  
                    tbaz            = np.pi*(az_grd)/180.
                    tGsin2          = np.sin(tbaz*2)
                    tGcos2          = np.cos(tbaz*2)
                    G               = np.append(G, tGsin2)
                    G               = np.append(G, tGcos2)
                    G               = G.reshape((3, Nbin))
                    G               = G.T
                    G               = np.dot(U, G)
                    # data
                    d               = indat.T
                    d               = np.dot(U, d)
                    # least square inversion
                    model           = np.linalg.lstsq(G, d)[0]
                    A0              = model[0]
                    A2              = np.sqrt(model[1]**2 + model[2]**2)
                    phi2            = np.arctan2(model[1], model[2])/2.
                    ### ---
                    temp_azi                = aziALL[:, ilat - nlat_grad, ilon - nlon_grad]
                    temp_ind_out            = index_outlier[:, ilat - nlat_grad, ilon - nlon_grad].copy()
                    temp_snowness           = slownessALL[:, ilat - nlat_grad, ilon - nlon_grad]
                    temp_vel                = temp_snowness.copy()
                    temp_vel[temp_vel!=0.]  = 1./temp_snowness[temp_snowness!=0.]
                    predat                  = A0 + A2*np.cos(2.*(np.pi/180.*(temp_azi+180.)-phi2) )
                    diff_vel                = abs(predat-temp_vel)
                    #---
                    temp_thresh             = A2
                    N0                      = temp_ind_out.size - temp_ind_out.sum()
                    temp_ind_out[diff_vel>2.*temp_thresh]    \
                                            = True
                    N1                      = temp_ind_out.size - temp_ind_out.sum()
                    # ---
                    if N0 == 0:
                        continue
                    while( (float(N1)/float(N0))<0.5 ):
                        temp_thresh         += 0.005
                        temp_ind_out        = index_outlier[:, ilat - nlat_grad, ilon - nlon_grad].copy()
                        temp_ind_out[diff_vel>2.*temp_thresh]    \
                                            = True
                        N1                  = temp_ind_out.size - temp_ind_out.sum()
                    # ---
                    index_outlier[:, ilat - nlat_grad, ilon - nlon_grad]\
                                            = temp_ind_out.copy()
                    ### ---
            # stacking to get anisotropic parameters
            dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni    \
                                        = _anisotropic_stacking(np.int64(gridx), np.int64(gridy), np.float32(maxazi), np.float32(minazi),\
                                                np.int64(N_bin), np.float64(Nmeasure), np.float64(aziALL),\
                                                np.float64(slowness_sumQC), np.float64(slownessALL), index_outlier.astype(bool))
        #----------------------------
        # save data to database
        #----------------------------
        outfname            = workingdir+'/azi_'+pfx+'.npz'
        out_dslow_sum_ani   = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
        out_dslow_sum_ani[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                            = dslow_sum_ani
        
        out_dslow_un        = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
        out_dslow_un[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                            = dslow_un
        
        out_vel_un          = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
        out_vel_un[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                            = vel_un
        
        out_histArr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
        out_histArr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                            = histArr
        
        out_NmeasureAni     = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
        out_NmeasureAni[(gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                            = NmeasureAni
        
        np.savez(outfname, out_dslow_sum_ani, out_dslow_un, out_vel_un, out_histArr, out_NmeasureAni)
    return