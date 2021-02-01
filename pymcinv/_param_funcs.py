# -*- coding: utf-8 -*-
"""
functions and classes for parameterization of the model

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import numba
import math
import random
import copy

@numba.jit(numba.types.Tuple((numba.float64[:, :], numba.float64[:]))\
    (numba.int64, numba.int64, numba.float64, numba.float64, numba.int64, numba.int64), nopython = True)
def bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts):
    """function that generate B spline basis
    """
    #-------------------------------- 
    # defining the knot vector
    #--------------------------------
    m           = nBs-1+degBs
    t           = np.zeros(m+1, dtype = np.float64)
    for i in range(degBs):
        t[i]    = zmin_Bs + i*(zmax_Bs-zmin_Bs)/10000.
    for i in range(degBs,m+1-degBs):
        n_temp  = m+1-degBs-degBs+1
        if (disfacBs !=1):
            temp= (zmax_Bs-zmin_Bs)*(disfacBs-1)/(math.pow(disfacBs,n_temp)-1)
        else:
            temp= (zmax_Bs-zmin_Bs)/n_temp
        t[i]    = temp*math.pow(disfacBs,(i-degBs)) + zmin_Bs
    for i in range(m+1-degBs,m+1):
        t[i]    = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-i)
    # depth array
    step        = (zmax_Bs-zmin_Bs)/(npts-1)
    depth       = np.zeros(npts, dtype=np.float64)
    for i in range(npts):
        depth[i]= np.float64(i) * np.float64(step) + np.float64(zmin_Bs)
    # arrays for storing B spline basis
    obasis      = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float64)
    nbasis      = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float64)
    #------------------------------------ 
    # computing B spline basis functions
    #------------------------------------
    for i in range (m):
        for j in range (npts):
            if (depth[j] >=t[i] and depth[j]<t[i+1]):
                obasis[i][j]= 1
            else:
                obasis[i][j]= 0
    for pp in range (1,degBs):
        for i in range (m-pp):
            for j in range (npts):
                nbasis[i][j]= (depth[j]-t[i])/(t[i+pp]-t[i])*obasis[i][j] + \
                        (t[i+pp+1]-depth[j])/(t[i+pp+1]-t[i+1])*obasis[i+1][j]
        for i in range (m-pp):
            for j in range (npts):
                obasis[i][j]= nbasis[i][j]
    nbasis[0][0]            = 1
    nbasis[nBs-1][npts-1]   = 1
    return nbasis, t

@numba.jit(numba.float64(numba.float64, numba.float64, numba.float64, numba.float64), nopython = True)
def _gauss_random(oldval, step, lbound, ubound):
    run 	= True
    j		= 0
    while (run and j < 10000): 
        newval      = random.gauss(oldval, step)
        if (newval >= lbound and newval <= ubound):
            run     = False
        j           += 1
    return newval

class para1d(object):
    """A class for handling parameter perturbations
    =====================================================================================================================
    ::: parameters :::
    :   values  :
    npara       - number of parameters for perturbations
    maxind      - maximum number of index for each parameters
    isspace     - if space array is computed or not
    :   arrays  :
    paraval     - parameter array for perturbation
    ---------------------------------------------------------------------------------------------------------------------
    paraindex   - index array indicating numerical setup for each parameter
                1.  isomod
                    paraindex[0, :] - type of parameters
                                        0   - velocity coefficient for splines
                                        1   - thickness
                                       -1   - vp/vs ratio
                    paraindex[1, :] - index for type of amplitude for parameter perturbation
                                        1   - absolute
                                        -1  - relative
                                        0   - fixed, do NOT perturb
                    paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
                    paraindex[3, :] - step for parameter space 
                    paraindex[4, :] - index for the parameter in the model group   
                    paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0
                2.  vtimod
                    paraindex[0, :] - type of parameters
                                        0   - vsh coefficient for splines
                                        1   - vsv coefficient for splines
                                        2   - thickness
                                        3   - gamma = 2(Vsh-Vsv)/(Vsh+Vsv)
                                        -1  - vp/vs ratio
                    paraindex[1, :] - index for type of amplitude for parameter perturbation
                                        1   - absolute
                                        -1  - relative
                                        0   - fixed, do NOT perturb, added on 2019-03-19
                    paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
                    paraindex[3, :] - step for parameter space 
                    paraindex[4, :] - index for the parameter in the model group   
                    paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0 or 1
                3.  ttimod
                    paraindex[0, :] - type of parameters
                                        0   - vph coefficient for splines
                                        1   - vpv coefficient for splines
                                        2   - vsh coefficient for splines
                                        3   - vsv coefficient for splines
                                        4   - eta coefficient for splines
                                        5   - dip
                                        6   - strike
                                        ---------------------------------
                                        below are currently not used yet
                                        7   - rho coefficient for splines
                                        8   - thickness
                                        -1  - vp/vs ratio
                    paraindex[1, :] - index for type of amplitude for parameter perturbation
                                        1   - absolute
                                        else- relative
                    paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
                    paraindex[3, :] - step for parameter space 
                    paraindex[4, :] - index for the parameter in the model group   
                    paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0
    ---------------------------------------------------------------------------------------------------------------------
    space       - space array for defining perturbation range
                    space[0, :]     - min value
                    space[1, :]     - max value
                    space[2, :]     - step, used as sigma in Gaussian random generator
    =====================================================================================================================
    """
    
    def __init__(self):
        self.npara          = 0
        self.maxind         = 6
        self.isspace        = False
        return
    
    def init_arr(self, npara):
        """initialize the arrays
        """
        self.npara          = npara
        self.paraval        = np.zeros(self.npara, dtype = np.float64)
        self.paraindex      = np.zeros((self.maxind, self.npara), dtype = np.float64)
        self.space          = np.zeros((3, self.npara), dtype = np.float64)
        return
   
    def read(self, infname):
        """read txt perturbation parameter file
        ==========================================================================
        ::: input :::
        infname - input file name
        ==========================================================================
        """
        npara       = 0
        i           = 0
        for l1 in open(infname,"r"):
            npara   += 1
        print ("Number of parameters for perturbation: %d " % npara)
        self.init_arr(npara)
        with open(infname, 'r') as fid:
            for line in fid.readlines():
                temp                        = np.array(line.split(), dtype=np.float64)
                ne                          = temp.size
                for j in range(ne):
                    self.paraindex[j, i]    = temp[j]
                i                           += 1
        return
        
    def write_txt(self, outfname):
        np.savetxt(outfname, self.paraval, fmt='%g')
        return
    
    def read_txt(self, infname):
        self.paraval  = np.loadtxt(infname, dtype=np.float64)
        return

    def new_paraval(self, ptype):
        """perturb parameters in paraval array
        ===============================================================================
        ::: input :::
        ptype   - perturbation type
                    0   - uniform random value generated from parameter space
                    1   - Gauss random number generator given mu = oldval, sigma=step
        ===============================================================================
        """
        if not self.isspace:
            print ('*** parameter space for perturbation NOT initialized!')
            return False
        if ptype == 0:
            for i in range(self.npara):
                if int(self.paraindex[1, i]) == 0:
                    continue
                tval            = random.random() # numpy.random will generate same values in multiprocessing!
                self.paraval[i] = tval * ( self.space[1, i] - self.space[0, i] ) + self.space[0, i]
        elif ptype == 1:
            for i in range(self.npara):
                # do NOT perturb fixed value
                if int(self.paraindex[1, i]) == 0:
                    continue
                oldval 	= self.paraval[i]
                step 	= self.space[2, i]
                #-----------------------
                # random Gaussian values
                #-----------------------
                newval  = _gauss_random(np.float64(oldval), np.float64(step),\
                                np.float64(self.space[0, i]), np.float64(self.space[1, i]))
                if (newval >= self.space[0, i] and newval <= self.space[1, i]):
                    self.paraval[i] = newval
                else:
                    # print ('!!! WARNING: Gaussian perturb FAILS! index = '+ str(i))
                    tval            = random.random()
                    self.paraval[i] = tval * ( self.space[1, i] - self.space[0, i] ) + self.space[0, i]
        else:
            raise ValueError('Unexpected perturbation type!')
        return True

