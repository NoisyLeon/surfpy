# -*- coding: utf-8 -*-
import numpy as np
import numba
import matplotlib.pyplot as plt
import os


def mask_interp(dlon, dlat, minlon, minlat, maxlon, maxlat, mask_in, dlon_out, dlat_out, inear_true_false = False):
    """interpolate mask array
    """
    if dlon == dlon_out and dlat == dlat_out:
        return mask_in
    Nlon    = int(round((maxlon-minlon)/dlon)+1)
    Nlat    = int(round((maxlat-minlat)/dlat)+1)
    if Nlat != mask_in.shape[0] or Nlon != mask_in.shape[1]:
        raise ValueError('Inconsistent shape in mask!')
    Nlon_out= int(round((maxlon-minlon)/dlon_out)+1)
    Nlat_out= int(round((maxlat-minlat)/dlat_out)+1)
    mask_out= np.ones((Nlat_out, Nlon_out), dtype = bool)
    return _mask_interp(dlon, dlat, minlon, minlat, Nlon, Nlat, mask_in,
            dlon_out, dlat_out, Nlon_out, Nlat_out, mask_out, inear_true_false)

@numba.jit(numba.boolean[:, :](numba.float64, numba.float64, numba.float64, numba.float64, numba.int64, numba.int64, numba.boolean[:, :],\
         numba.float64, numba.float64, numba.int64, numba.int64, numba.boolean[:, :], numba.boolean), nopython = True)
def _mask_interp(dlon, dlat, minlon, minlat, Nlon, Nlat, mask_in, dlon_out, dlat_out, Nlon_out, Nlat_out, mask_out, inear_true_false):
    
    for ilon_out in range(Nlon_out):
        for ilat_out in range(Nlat_out):
            lon_out     = minlon + ilon_out*dlon_out
            lat_out     = minlat + ilat_out*dlat_out
            val_assigned= False
            mask_near   = np.ones(4)
            imask_near  = 0
            #=====================
            # loop over input mask
            #=====================
            for ilon in range(Nlon):
                if val_assigned:
                    break
                if imask_near == 4: # uncomment after debug
                    break
                for ilat in range(Nlat):
                    lon = minlon + ilon*dlon
                    lat = minlat + ilat*dlat
                    if abs(lon - lon_out) < 0.01 and abs(lat - lat_out) < 0.01:
                        mask_out[ilat_out, ilon_out]= mask_in[ilat, ilon]
                        val_assigned                = True
                        break
                    if imask_near == 4: # uncomment after debug
                        break
                    # nearby
                    if abs(lon - lon_out) <= dlon and abs(lat - lat_out) <= dlat:
                        mask_near[imask_near]   = mask_in[ilat, ilon]
                        imask_near              += 1
            if not val_assigned:
                # debug
                if imask_near != 2 and imask_near != 4:
                    print (imask_near, lon_out, lat_out, dlon, dlat)
                    raise ValueError('check near neighbor mask values')
                # True if any nearby value is True
                if inear_true_false:
                    mask_out[ilat_out, ilon_out]    = bool(mask_near[:imask_near].sum())
                # False if any nearby value is False
                else:
                    mask_out[ilat_out, ilon_out]    = bool(np.prod(mask_near))
    return mask_out

# 
# 
# def _get_vs_2d(z0, z1, zArr, vs_3d):
#     Nlat, Nlon, Nz  = vs_3d.shape
#     vs_out          = np.zeros((Nlat, Nlon))
#     for ilat in range(Nlat):
#         for ilon in range(Nlon):
#             ind     = np.where((zArr > z0[ilat, ilon])*(zArr < z1[ilat, ilon]))[0]
#             vs_temp = vs_3d[ilat, ilon, ind].mean()
#             vs_out[ilat, ilon]\
#                     = vs_temp
#     return vs_out
# 
# @numba.jit(numba.float64[:, :, :](numba.float64[:, ], numba.float64[:, :, :], numba.float64))
# def _get_avg_vs3d(zArr, vs3d, depthavg):
#     tvs3d           = vs3d.copy()
#     Nlat, Nlon, Nz  = vs3d.shape
#     Nz              = zArr.size
#     # # vs_out          = np.zeros((Nlat, Nlon))
#     # for ilat in range(Nlat):
#     #     for ilon in range(Nlon):
#     #         
#     for i in range(Nz):
#         z       = zArr[i]
#         print (i)
#         if z < depthavg:
#             tvs3d[:, :, i]  = (vs3d[:, :, zArr <= 2.*depthavg]).mean(axis=2)
#             continue
#         index   = (zArr <= z + depthavg) + (zArr >= z - depthavg)
#         tvs3d[:, :, i]  = (vs3d[:, :, index]).mean(axis=2)
#     return tvs3d
# 
# def to_percent(y, position):
#      # Ignore the passed in position. This has the effect of scaling the default
#      # tick locations.
#      s = '%.0f' %(100. * y)
#      # The percent symbol needs escaping in latex
#      if matplotlib.rcParams['text.usetex'] is True:
#          return s + r'$\%$'
#      else:
#          return s + '%'
#     
# def read_slab_contour(infname, depth):
#     ctrlst  = []
#     lonlst  = []
#     latlst  = []
#     with open(infname, 'rb') as fio:
#         newctr  = False
#         for line in fio.readlines():
#             if line.split()[0] is '>':
#                 newctr  = True
#                 if len(lonlst) != 0:
#                     ctrlst.append([lonlst, latlst])
#                 lonlst  = []
#                 latlst  = []
#                 z       = -float(line.split()[1])
#                 if z == depth:
#                     skipflag    = False
#                 else:
#                     skipflag    = True
#                 continue
#             if skipflag:
#                 continue
#             lonlst.append(float(line.split()[0]))
#             latlst.append(float(line.split()[1]))
#     return ctrlst
#                     
#     
# def discrete_cmap(N, base_cmap=None):
#     """Create an N-bin discrete colormap from the specified input map"""
#     # Note that if base_cmap is a string or None, you can simply do
#     #    return plt.cm.get_cmap(base_cmap, N)
#     # The following works for string, None, or a colormap instance:
#     if os.path.isfile(base_cmap):
#         import pycpt
#         base    = pycpt.load.gmtColormap(base_cmap)
#     else:
#         base    = plt.cm.get_cmap(base_cmap)
#     color_list  = base(np.linspace(0, 1, N))
#     cmap_name   = base.name + str(N)
#     return base.from_list(cmap_name, color_list, N)
# 
# 
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    if os.path.isfile(base_cmap):
        import pycpt
        base    = pycpt.load.gmtColormap(base_cmap)
    else:
        base    = plt.cm.get_cmap(base_cmap)
    color_list  = base(np.linspace(0, 1, N))
    cmap_name   = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def read_slab_contour(infname, depth):
    ctrlst  = []
    lonlst  = []
    latlst  = []
    with open(infname, 'r') as fio:
        newctr  = False
        skipflag    = False
        for line in fio.readlines():
            if line.split()[0] is '>':
                newctr  = True
                if len(lonlst) != 0:
                    ctrlst.append([lonlst, latlst])
                lonlst  = []
                latlst  = []
                z       = -float(line.split()[1])
                if z == depth:
                    skipflag    = False
                else:
                    skipflag    = True
                continue
            if skipflag:
                continue
            lonlst.append(float(line.split()[0]))
            latlst.append(float(line.split()[1]))
    return ctrlst

