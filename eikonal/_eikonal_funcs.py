# -*- coding: utf-8 -*-
"""
Perform data interpolation/computation on the surface of the Earth

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import obspy
import os
from numba import jit, float32, int32, boolean, float64, int64
from numba import njit, prange
import numba 


def determine_interval(minlat=None, maxlat=None, dlon=0.2,  dlat=0.2, verbose=True):
    # if (medlat is None) and (minlat is None and maxlat is None):
    #     raise ValueError('medlat or minlat/maxlat need to be specified!')
    # if minlat is not None and maxlat is not None:
    medlat              = (minlat + maxlat)/2.
    dist_lon_max,az,baz = obspy.geodetics.gps2dist_azimuth(minlat, 0., minlat, dlon)
    dist_lon_min,az,baz = obspy.geodetics.gps2dist_azimuth(maxlat, 0., maxlat, dlon)
    dist_lon_med,az,baz = obspy.geodetics.gps2dist_azimuth(medlat, 0., medlat, dlon)
    dist_lat, az, baz   = obspy.geodetics.gps2dist_azimuth(medlat, 0., medlat + dlat, 0.)
    ratio_min           = dist_lat / dist_lon_max
    ratio_max           = dist_lat / dist_lon_min
    index               = np.floor(np.log2((ratio_min+ratio_max)/2.))
    final_ratio         = 2**index
    if verbose:
        print ('ratio_min =',ratio_min,',ratio_max =',ratio_max,',final_ratio =',final_ratio)
    return final_ratio

def eikonal_multithread(in_grder, workingdir, channel, nearneighbor, cdist):
    working_per     = workingdir+'/'+str(in_grder.period)+'sec'
    if in_grder.interpolate_type == 'gmt':
        in_grder.interp_surface(do_blockmedian = True)
    else:
        in_grder.interp_verde()
    if not in_grder.check_curvature():
        return
    in_grder.eikonal( nearneighbor = nearneighbor, cdist = cdist)
    outfname_npz    = working_per+'/'+in_grder.evid+'_eikonal'
    in_grder.write_binary(outfname = outfname_npz)
    return

@jit(float32[:,:,:](float32[:,:,:], float32[:,:,:]), nopython = True)
def _get_azi_weight(aziALL, validALL):
    Nevent, Nlat, Nlon  = aziALL.shape
    weightALL           = np.zeros((Nevent, Nlat, Nlon), dtype = np.float32)
    for ilon in range(Nlon):
        for ilat in range(Nlat):
            for i in range(Nevent):
                for j in range(Nevent):
                    del_az                      = abs(aziALL[i, ilat, ilon] - aziALL[j, ilat, ilon])
                    if del_az < 20. or del_az > 340.:
                        weightALL[i, ilat, ilon]+= validALL[i, ilat, ilon]    
    return weightALL

@njit(numba.types.Tuple((float64[:, :, :], float64[:, :, :], float64[:, :, :], int64[:, :, :], int64[:, :]))\
     (int64, int64, float32, float32, int64, float64[:, :], float64[:, :, :], float64[:, :], float64[:, :, :], numba.boolean[:, :, :]))
def _anisotropic_stacking(gridx, gridy, maxazi, minazi, N_bin, Nmeasure, aziALL, slowness_sumQC, slownessALL, index_outlier):
    """anisotropic stacking in parallel using numba
    NOTE: grid_lat and grid_lon are considerred as gridx and gridy here
    """
    Nevent, Nx, Ny  = aziALL.shape
    Nx_trim         = Nx - (gridx - 1)
    Ny_trim         = Ny - (gridy - 1)
    NmeasureAni     = np.zeros((Nx_trim, Ny_trim), dtype = np.int64) # for quality control
    # initialization of anisotropic parameters
    d_bin           = float((maxazi-minazi)/N_bin)
    # number of measurements in each bin
    histArr         = np.zeros((N_bin, Nx_trim, Ny_trim), dtype = np.int64)
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
     (int64, int64, float32, float32, int64, float64[:, :], float64[:, :, :], float64[:, :], float64[:, :, :],\
      numba.boolean[:, :, :]), parallel = True)
def _anisotropic_stacking_parallel(gridx, gridy, maxazi, minazi, N_bin, Nmeasure, aziALL, slowness_sumQC, slownessALL, index_outlier):
    """anisotropic stacking in parallel using numba
    NOTE: grid_lat and grid_lon are considerred as gridx and gridy here
    """
    Nevent, Nx, Ny  = aziALL.shape
    Nx_trim         = Nx - (gridx - 1)
    Ny_trim         = Ny - (gridy - 1)
    NmeasureAni     = np.zeros((Nx_trim, Ny_trim), dtype = np.int64) # for quality control
    # initialization of anisotropic parameters
    d_bin           = float((maxazi-minazi)/N_bin)
    # number of measurements in each bin
    histArr         = np.zeros((N_bin, Nx_trim, Ny_trim), dtype = np.int64)
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


def eikonal_multithread_old(in_grder, workingdir, channel, nearneighbor, cdist):
    working_per     = workingdir+'/'+str(in_grder.period)+'sec'
    outfname        = in_grder.evid+'_'+in_grder.fieldtype+'_'+channel+'.lst'
    prefix          = in_grder.evid+'_'+channel+'_'
    if in_grder.interpolate_type == 'gmt':
        in_grder.interp_surface(workingdir = working_per, outfname = outfname)
    else:
        in_grder.interp_verde()
    if not in_grder.check_curvature(workingdir = working_per, outpfx = prefix):
        return
    in_grder.eikonal(workingdir = working_per, inpfx = prefix, nearneighbor = nearneighbor, cdist = cdist)
    outfname_npz    = working_per+'/'+in_grder.evid+'_eikonal'
    in_grder.write_binary(outfname = outfname_npz)
    return