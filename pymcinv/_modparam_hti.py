# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the HTI model

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import numba
import math
import random
from scipy.optimize import lsq_linear
import scipy.interpolate
import scipy.signal
import copy
from uncertainties import unumpy

#====================
# global variables
#====================

SEDDEPTH    = -1
MOHODEPTH   = -2
MAXDEPTH    = -3


class htimod(object):
    """
    An object for handling parameterization of 1D Horizontal TI model for the inversion
    =====================================================================================================================
    ::: parameters :::
    :   numbers     :
    nmod        - number of model groups
    :   1D arrays   :
    depth       - size: nmod+1; depth array indicating depth distribution of anisotropic parameters
                    if depth < 0.
                    -1 - use sediment depth
                    -2 - use moho depth
                    -3 - use maxdepth
    =====================================================================================================================
    """
    
    def __init__(self):
        self.nmod           = 0
        return
    
    def init_arr(self, nmod):
        """
        initialization of arrays
        """
        self.nmod       = nmod
        # arrays of size nmod
        self.Gc         = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.unGc       = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.Gs         = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.unGs       = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.psi2       = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.unpsi2     = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.amp        = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.unamp      = np.zeros(np.int64(self.nmod), dtype=np.float64)
        # # # self.Bc         = np.zeros(np.int64(self.nmod), dtype=np.float64)
        # # # self.Bs         = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.depth      = np.zeros(np.int64(self.nmod+1), dtype = np.float64)
        self.depth2d    = np.zeros((np.int64(self.nmod), np.int64(2) ), dtype=np.int32)
        self.layer_ind  = np.zeros(( np.int64(self.nmod), np.int64(2) ), dtype=np.int32)
        return
    
    
    def set_two_layer_crust(self, depth_mid_crust=15.):
        self.init_arr(3)
        self.depth[0]   = SEDDEPTH
        self.depth[1]   = depth_mid_crust
        self.depth[2]   = MOHODEPTH
        self.depth[3]   = MAXDEPTH
    
    def set_intermediate_depth(self, depth_mid_crust=15., depth_mid_mantle=-1.):
        self.depth[0]       = SEDDEPTH
        i                   = 1
        if depth_mid_crust > 0.:
            self.depth[i]   = depth_mid_crust
            i               += 1
        self.depth[i]       = MOHODEPTH
        i                   += 1
        if depth_mid_mantle > 0.:
            self.depth[i]   = depth_mid_mantle
            i               += 1
        self.depth[i]       = MAXDEPTH
        return
    
    def set_three_mantle(self, depth1 = 40., depth2 = 100.):
        self.depth[0]       = SEDDEPTH
        self.depth[1]       = MOHODEPTH
        self.depth[2]       = depth1
        self.depth[3]       = depth2
        self.depth[4]       = MAXDEPTH
        return
        
    def GcGs_to_azi(self):
        self.psi2[:]                    = np.arctan2(self.Gs, self.Gc)/2./np.pi*180.
        self.psi2[self.psi2<0.]         += 180.
        self.amp[:]                     = np.sqrt(self.Gs**2 + self.Gc**2)/2.*100.
        Gc_with_un                      = unumpy.uarray(self.Gc, self.unGc)
        Gs_with_un                      = unumpy.uarray(self.Gs, self.unGs)
        self.unpsi2[:]                  = unumpy.std_devs( unumpy.arctan2(Gs_with_un, Gc_with_un)/2./np.pi*180.)
        self.unpsi2[self.unpsi2>90.]    = 90.
        self.unamp[:]                   = unumpy.std_devs( unumpy.sqrt(Gs_with_un**2 + Gc_with_un**2)/2.*100.)
        self.unamp[self.unamp>self.amp] = self.amp[self.unamp>self.amp]
        

