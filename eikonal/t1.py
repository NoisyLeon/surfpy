def _anisotropic_stacking(gridx, gridy, maxazi, minazi, N_bin, Nmeasure, aziALL, slowness_sumQC, slownessALL, index_outlier):
    """anisotropic stacking in parallel using numba
    NOTE: grid_lat and grid_lon are considerred as gridx and gridy here
    """
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
