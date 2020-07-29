# -*- coding: utf-8 -*-
"""
Perform data interpolation/computation on the surface of the Earth

    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import obspy
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numba


def determine_interval(minlat=None, maxlat=None, dlon=0.2,  dlat=0.2, verbose=True):
    # if (medlat is None) and (minlat is None and maxlat is None):
    #     raise ValueError('medlat or minlat/maxlat need to be specified!')
    # if minlat is not None and maxlat is not None:
    medlat              = (minlat + maxlat)/2.
    dist_lon_max,az,baz = obspy.geodetics.gps2dist_azimuth(minlat, 0., minlat, dlon)
    dist_lon_min,az,baz = obspy.geodetics.gps2dist_azimuth(maxlat, 0., maxlat, dlon)
    dist_lon_med,az,baz = obspy.geodetics.gps2dist_azimuth(medlat, 0., medlat, dlon)
    dist_lat, az, baz   = obspy.geodetics.gps2dist_azimuth(medlat, 0., medlat+dlat, 0.)
    ratio_min           = dist_lat / dist_lon_max
    ratio_max           = dist_lat / dist_lon_min
    index               = np.floor(np.log2((ratio_min+ratio_max)/2.))
    final_ratio         = 2**index
    if verbose:
        print ('ratio_min =',ratio_min,',ratio_max =',ratio_max,',final_ratio =',final_ratio)
    return final_ratio