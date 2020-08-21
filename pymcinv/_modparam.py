# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

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

class para1d(object):
    """
    An object for handling parameter perturbations
    =====================================================================================================================
    ::: parameters :::
    :   values  :
    npara       - number of parameters for perturbations
    maxind      - maximum number of index for each parameters
    isspace     - if space array is computed or not
    :   arrays  :
    paraval     - parameter array for perturbation
    paraindex   - index array indicating numerical setup for each parameter
                1.  isomod
                    paraindex[0, :] - type of parameters
                                        0   - velocity coefficient for splines
                                        1   - thickness
                                       -1   - vp/vs ratio
                    paraindex[1, :] - index for type of amplitude for parameter perturbation
                                        1   - absolute
                                        -1  - relative
                                        0   - fixed, do NOT perturb, added on 2019-03-19
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
        """
        initialize the arrays
        """
        self.npara          = npara
        self.paraval        = np.zeros(self.npara, dtype=np.float64)
        self.paraindex      = np.zeros((self.maxind, self.npara), dtype = np.float64)
        self.space          = np.zeros((3, self.npara), dtype = np.float64)
        return
   
    def readparatxt(self, infname):
        """
        read txt perturbation parameter file
        ==========================================================================
        ::: input :::
        infname - input file name
        ==========================================================================
        """
        npara       = 0
        i           = 0
        for l1 in open(infname,"r"):
            npara   += 1
        print "Number of parameters for perturbation: %d " % npara
        self.init_arr(npara)
        with open(infname, 'r') as fid:
            for line in fid.readlines():
                temp                        = np.array(line.split(), dtype=np.float64)
                ne                          = temp.size
                for j in range(ne):
                    self.paraindex[j, i]    = temp[j]
                i                           += 1
        return
        
        
    def write_paraval_txt(self, outfname):
        np.savetxt(outfname, self.paraval, fmt='%g')
        return
    
    def read_paraval_txt(self, infname):
        self.paraval  = np.loadtxt(infname, dtype=np.float64)
        return

    def new_paraval(self, ptype):
        """
        perturb parameters in paraval array
        ===============================================================================
        ::: input :::
        ptype   - perturbation type
                    0   - uniform random value generated from parameter space
                    1   - Gauss random number generator given mu = oldval, sigma=step
        ===============================================================================
        """
        if not self.isspace:
            print('Parameter space for perturbation has not been initialized yet!')
            return False
        # # # if ptype == 0:
        # # #     newparaval      = np.random.uniform(self.space[0, :], self.space[1, :], size=self.npara)
        # # #     ind_perturb     = (self.paraindex[1, :]).astype(int)!= 0
        # # #     self.paraval[ind_perturb] \
        # # #                     = newparaval[ind_perturb]
        if ptype == 0:
            for i in range(self.npara):
                if int(self.paraindex[1, i]) == 0:
                    continue
                tval            = random.random()
                self.paraval[i] = tval * ( self.space[1, i] - self.space[0, i] ) + self.space[0, i]
        elif ptype == 1:
            for i in range(self.npara):
                #====================================
                # added on 2019/03/15, 
                # do NOT perturb fixed value
                if int(self.paraindex[1, i]) == 0:
                    continue
                #====================================
                oldval 	= self.paraval[i]
                step 	= self.space[2, i]
                run 	= True
                j		= 0
                while (run and j<1000): 
                    newval      = random.gauss(oldval, step)
                    if (newval >= self.space[0, i] and newval <= self.space[1, i]):
                        run     = False
                    j           += 1
                # bug fix on 2019/01/18
                if j >= 1000:
                    tval            = random.random()
                    self.paraval[i] = tval * ( self.space[1, i] - self.space[0, i] ) + self.space[0, i]
                    # # # self.paraval[i] = np.random.uniform(self.space[0, i], self.space[1, i], size=1)
                else:
                    self.paraval[i] = newval
        else:
            raise ValueError('Unexpected perturbation type!')
        return True
    
####################################################
# auxiliary functions
####################################################

@numba.jit(numba.float64[:, :](numba.int64, numba.int64, numba.float64, numba.float64, numba.int64, numba.int64))
def bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts):
    """
    function that generate B spline basis
    """
    #-------------------------------- 
    # defining the knot vector
    #--------------------------------
    m           = nBs-1+degBs
    t           = np.zeros(m+1, dtype=np.float64)
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
    #-------------------------------- 
    # computing B spline basis functions
    #--------------------------------
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


class isomod(object):
    """
    An object for handling parameterization of 1d isotropic model for the inversion
    =====================================================================================================================
    ::: parameters :::
    :   numbers     :
    nmod        - number of model groups
    maxlay      - maximum layers for each group (default - 100)
    maxspl      - maximum spline coefficients for each group (default - 20)
    :   1D arrays   :
    numbp       - number of control points/basis (1D int array with length nmod)
    mtype       - model parameterization types (1D int array with length nmod)
                    1   - layer         - nlay  = numbp, hArr = ratio*thickness, vs = cvel
                    2   - B-splines     - hArr  = thickness/nlay, vs    = (cvel*spl)_sum over numbp
                    4   - gradient layer- nlay is defined depends on thickness
                                            hArr  = thickness/nlay, vs  = from cvel[0, i] to cvel[1, i]
                    5   - water         - nlay  = 1, vs = 0., hArr = thickness
    thickness   - thickness of each group (1D float array with length nmod)
    nlay        - number of layres in each group (1D int array with length nmod)
    vpvs        - vp/vs ratio in each group (1D float array with length nmod)
    isspl       - flag array indicating the existence of basis B spline (1D int array with length nmod)
                    0 - spline basis has NOT been computed
                    1 - spline basis has been computed
    :   multi-dim arrays    :
    t           - knot vectors for B splines (2D array - [:(self.numb[i]+degBs), i]; i indicating group id)
    spl         - B spline basis array (3D array - [:self.numb[i], :self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 2
    ratio       - array for the ratio of each layer (2D array - [:self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 1
    cvel        - velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
                    layer mod   - input velocities for each layer
                    spline mod  - coefficients for B spline
                    gradient mod- top/bottom layer velocity
    :   model arrays        :
    vs          - vs velocity arrays (2D array - [:self.nlay[i], i]; i indicating group id)
    hArr        - layer arrays (2D array - [:self.nlay[i], i]; i indicating group id)
    :   para1d  :
    para        - object storing parameters for perturbation
    *********************************************************************************************************************
    initialization      : (1) self.parameterize_ak135(); (2) self.get_paraind(); (3) self.upate(); (4) self.get_vmodel();
                          (5) self.mod2para()
    paraval -> model    : (1) self.para2mod(); (2) self.update(); (3) self.get_vmodel()
    model   -> paraval  : (1) self.mod2para()
    =====================================================================================================================
    """
    
    def __init__(self):
        self.nmod       = 0
        self.maxlay     = 100
        self.maxspl     = 20
        self.para       = para1d()
        return
    
    def init_arr(self, nmod):
        """
        initialization of arrays
        """
        self.nmod       = nmod
        # arrays of size nmod
        self.numbp      = np.zeros(self.nmod, dtype=np.int64)
        self.mtype      = np.zeros(self.nmod, dtype=np.int64)
        self.thickness  = np.zeros(self.nmod, dtype=np.float64)
        self.nlay       = np.ones(self.nmod, dtype=np.int64)*20
        self.vpvs       = np.ones(self.nmod, dtype=np.float64)*1.75
        self.isspl      = np.zeros(self.nmod, dtype=np.int64)
        # arrays of size maxspl, nmod
        self.cvel       = np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        # arrays of size maxlay, nmod
        self.ratio      = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
        self.vs         = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
        self.hArr       = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
        # arrays of size maxspl, maxlay, nmod
        self.spl        = np.zeros((self.maxspl, self.maxlay, self.nmod), dtype = np.float64)
        # added Sep 14th, 2018
        self.knot_vector= np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        self.Nknot      = np.zeros((self.nmod), dtype = np.int64)
        return
    
    def readmodtxt(self, infname):
        """
        Read model parameterization from a txt file
        column 1: id
        column 2: flag  - layer(1)/B-splines(2/3)/gradient layer(4)/water(5)
        column 3: thickness
        column 4: number of control points for the group
        column 5 - (4+tnp): value
        column 4+tnp - 4+2*tnp: ratio
        column -1: vpvs ratio
        ==========================================================================
        ::: input :::
        infname - input file name
        ==========================================================================
        """
        nmod        = 0
        for l1 in open(infname,"r"):
            nmod    += 1
        print "Number of model parameter groups: %d " % nmod
        # step 1
        self.init_arr(nmod)
        
        for l1 in open(infname,"r"):
            l1 			            = l1.rstrip()
            l2 			            = l1.split()
            iid 		            = int(l2[0])
            flag		            = int(l2[1])
            thickness	            = float(l2[2])
            tnp 		            = int(l2[3]) # number of parameters
            # step 2
            self.mtype[iid]	        = flag
            self.thickness[iid]     = thickness
            self.numbp[iid]         = tnp 
            if (int(l2[1]) == 5):  # water layer			
                if (tnp != 1):
                    print('Water layer! Only one value for Vp')
                    return False
            if (int(l2[1]) == 4):
                if (tnp != 2):
                    print('Error: only two values needed for gradient type, and one value for vpvs')
                    print tnp
                    return False
            if ( (int(l2[1])==1 and len(l2) != 4+2*tnp + 1) or (int(l2[1]) == 2 and len(l2) != 4+tnp + 1) ): # tnp parameters (+ tnp ratio for layered model) + 1 vpvs parameter
                print('wrong input !!!')
                return False
            nr          = 0
            # step 3
            for i in xrange(tnp):
                self.cvel[i, iid]       = float(l2[4+i])
                if (int(l2[1]) ==1):  # type 1 layer
                    self.ratio[nr, iid] = float(l2[4+tnp+i])
                    nr  += 1
            # step 4
            self.vpvs[iid]         = (float(l2[-1]))-0.
        return True

    def bspline(self, i):
        """
        Compute B-spline basis given group id
        The actual degree is k = degBs - 1
        e.g. nBs = 5, degBs = 4, k = 3, cubic B spline
        ::: output :::
        self.spl    - (nBs+k, npts)
                        [:nBs, :] B spline basis for nBs control points
                        [nBs:, :] can be ignored
        """    
        if self.thickness[i] >= 150:
            self.nlay[i]    = 60
        elif self.thickness[i] < 10:
            self.nlay[i]    = 5
        elif self.thickness[i] < 20:
            self.nlay[i]    = 10
        else:
            self.nlay[i]    = 30
        if self.isspl[i]:
            print('spline basis already exists!')
            return
        if self.mtype[i] != 2:
            print('Not spline parameterization!')
            return 
        # initialize
        if i >= self.nmod:
            raise ValueError('index for spline group out of range!')
            return
        nBs         = self.numbp[i]
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness[i]
        disfacBs    = 2.
        npts        = self.nlay[i]
        # original
        # nbasis      = bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        # modified Sep 14th, 2018 
        nbasis, t   = bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        m           = nBs-1+degBs
        if m > self.maxspl:
            raise ValueError('number of splines is too large, change default maxspl!')
        self.spl[:nBs, :npts, i]= nbasis[:nBs, :]
        self.isspl[i]           = True
        # added Sep 14th, 2018
        self.knot_vector[:(nBs+degBs), i]\
                                = t
        self.Nknot[i]           = t.size
        return True

    def update(self):
        """
        Update model (vs and hArr arrays), from the thickness, cvel
        """
        for i in range(self.nmod):
            if self.nlay[i] > self.maxlay:
                print ('number of layers is too large, need change default maxlay!')
                return False
            # layered model
            if self.mtype[i] == 1:
                self.nlay[i]                = self.numbp[i]
                self.hArr[:self.nlay[i], i] = self.ratio[:self.nlay[i], i] * self.thickness[i]
                self.vs[:self.nlay[i], i]   = self.cvel[:self.nlay[i], i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i]               = False
                self.bspline(i)
                self.vs[:self.nlay[i], i]   = np.dot( (self.spl[:self.numbp[i], :self.nlay[i], i]).T, self.cvel[:self.numbp[i], i])
                self.hArr[:self.nlay[i], i] = self.thickness[i]/self.nlay[i]
                # # for ilay in range(self.nlay[i]):
                # #     tvalue 	= 0.
                # #     for ibs in xrange(self.numbp[i]):
                # #         tvalue = tvalue + self.spl[ibs, ilay, i] * self.cvel[ibs, i]
                # #     self.vs[ilay, i]    = tvalue
                # #     self.hArr[ilay, i]  = self.thickness[i]/self.nlay[i]
            # gradient layer
            elif self.mtype[i] == 4:
                nlay 	    = 4
                if self.thickness[i] >= 20.:
                    nlay    = 20
                if self.thickness[i] > 10. and self.thickness[i] < 20.:
                    nlay    = int(self.thickness[i]/1.)
                if self.thickness[i] > 2. and self.thickness[i] <= 10.:
                    nlay    = int(self.thickness[i]/0.5)
                if self.thickness[i] < 0.5:
                    nlay    = 2
                dh 	                    = self.thickness[i]/float(nlay)
                dcvel 		            = (self.cvel[1, i] - self.cvel[0, i])/(nlay - 1.)
                self.vs[:nlay, i]       = self.cvel[0, i] + dcvel*np.arange(nlay, dtype=np.float64)
                self.hArr[:nlay, i]     = dh
                self.nlay[i]            = nlay
            # water layer
            elif self.mtype[i] == 5:
                nlay                    = 1
                self.vs[0, i]           = 0.
                self.hArr[0, i]         = self.thickness[i]
                self.nlay[i]            = 1
        return True
     
    def update_depth(self):
        """
        update hArr arrays only, used for paramerization of a refernce input model
        """
        for i in range(self.nmod):
            if self.nlay[i] > self.maxlay:
                printf('number of layers is too large, need change default maxlay!')
                return False
            # layered model
            if self.mtype[i] == 1:
                self.nlay[i]                = self.numbp[i]
                self.hArr[:self.nlay[i], i] = self.ratio[:self.nlay[i], i] * self.thickness[i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i]   = False
                self.bspline(i)
                self.hArr[:self.nlay[i], i] = self.thickness[i]/self.nlay[i]
            # gradient layer
            elif self.mtype[i] == 4:
                nlay 	    = 4
                if self.thickness[i] >= 20.:
                    nlay    = 20
                if self.thickness[i] > 10. and self.thickness[i] < 20.:
                    nlay    = int(self.thickness[i]/1.)
                if self.thickness[i] > 2. and self.thickness[i] <= 10.:
                    nlay    = int(self.thickness[i]/0.5)
                if self.thickness[i] < 0.5:
                    nlay    = 2
                dh 	                    = self.thickness[i]/float(nlay)
                self.hArr[:nlay, i]     = dh
                self.nlay[i]            = nlay
            # water layer
            elif self.mtype[i] == 5:
                nlay                    = 1
                self.hArr[0, i]         = self.thickness[i]
                self.nlay[i]            = 1
        return True
    
    def parameterize_input(self, zarr, vsarr, crtthk, sedthk, topovalue=1., maxdepth=200., vp_water=1.5):
        """
        paramerization of a reference input model
        ===============================================================================
        ::: input :::
        zarr, vsarr - input depth/vs array, must be the same size (unit - km, km/s)
        crtthk      - input crustal thickness (unit - km)
        sedthk      - input sediment thickness (unit - km)
        maxdepth    - maximum depth for the 1-D profile (default - 200 km)
        ::: output :::
        self.thickness  
        self.numbp      - [2, 4, 5]
        self.mtype      - [4, 2, 2]
        self.vpvs       - [2., 1.75, 1.75]
        self.spl
        self.cvel       - determined from input vs profile
        ::: history :::
        05/19/2018  - added the capability of parameterize model with water layer
        ===============================================================================
        """
        if zarr.size != vsarr.size:
            raise ValueError('Inconsistent input 1-D profile depth and vs arrays!')
        if topovalue < 0.:
            self.init_arr(4)
            self.thickness[:]   = np.array([-topovalue, sedthk, crtthk - sedthk, maxdepth - crtthk + topovalue])
            self.numbp[:]       = np.array([1, 2, 4, 5])
            self.mtype[:]       = np.array([5, 4, 2, 2])
            self.vpvs[:]        = np.array([0., 2., 1.75, 1.75])
        else:
            self.init_arr(3)
            self.thickness[:]   = np.array([sedthk, crtthk - sedthk, maxdepth - crtthk])
            self.numbp[:]       = np.array([2, 4, 5])
            self.mtype[:]       = np.array([4, 2, 2])
            self.vpvs[:]        = np.array([2., 1.75, 1.75])
        self.update_depth()
        hArr                = np.append(self.hArr[:self.nlay[0], 0], self.hArr[:self.nlay[1], 1])
        hArr                = np.append(hArr, self.hArr[:self.nlay[2], 2])
        if topovalue < 0.:
            hArr            = np.append(hArr, self.hArr[:self.nlay[3], 3])
        #--------------------------------------------
        # interpolation input vs to the grid points
        #--------------------------------------------
        # get grid points
        nlay_total          = self.nlay.sum()
        indlay              = np.arange(nlay_total, dtype=np.int32)
        indgrid0            = indlay*2
        indgrid1            = indlay*2+1
        indlay2             = np.arange(nlay_total-1, dtype=np.int32)
        indgrid2            = indlay2*2+2
        depth               = hArr.cumsum()
        zinterp             = np.zeros(2*nlay_total)
        zinterp[indgrid1]   = depth
        zinterp[indgrid2]   = depth[:-1] # grid points
        ind_max             = np.where(zarr >= maxdepth)[0][0]
        zarr                = zarr[:(ind_max+1)] 
        vsarr               = vsarr[:(ind_max+1)]
        # remove water layer
        try:
            ind_zero        = np.where(vsarr == 0.)[0][-1]
            zarr            = zarr[(ind_zero+1):]
            vsarr           = vsarr[(ind_zero+1):]
        except IndexError:
            pass
        # make necessary change to the input vs profile to enforce a monotonical increase in the crust
        if vsarr[0] > vsarr[1]:
            vs_temp         = vsarr[0]
            vsarr[0]        = vsarr[1]
            vsarr[1]        = vs_temp
        ind_crust           = np.where(zarr >= crtthk)[0][0]
        vs_crust            = vsarr[:(ind_crust+1)]  
        if not np.all(vs_crust[1:] >= vs_crust[:-1]):
            print 'WARNING: sort the input vs array to make it monotonically increases with depth in the crust'
            vs_crust        = np.sort(vs_crust)
            vsarr           = np.append(vs_crust, vsarr[(ind_crust+1):])
        vsinterp            = np.interp(zinterp, zarr, vsarr)
        # convert to layerized model
        indz0               = np.arange(nlay_total, dtype = np.int32)*2
        indz1               = np.arange(nlay_total, dtype = np.int32)*2 + 1
        #
        # debug
        z0                  = zinterp[indz0]
        z1                  = zinterp[indz1]
        h                   = z1 - z0
        if not np.allclose(h, hArr):
            raise ValueError('Debug layer array!')
        # END debug
        indlay              = np.arange(nlay_total, dtype = np.int32)*2 + 1
        vsinterp            = vsinterp[indlay]
        #------------------------------------
        # determine self.cvel arrays
        #------------------------------------
        # water layer
        if topovalue < 0.:
            self.cvel[0, 0] = vp_water
            vsinterp[0]     = 0.
        # sediments
        if topovalue >= 0.:
            self.cvel[0, 0] = vsinterp[0]
            self.cvel[1, 0] = vsinterp[self.nlay[0]-1]
        else:
            self.cvel[0, 1] = vsinterp[self.nlay[0]]
            self.cvel[1, 1] = vsinterp[self.nlay[0] + self.nlay[1]-1]
        #---------------------------------
        # inversion with lsq_linear
        #---------------------------------
        if topovalue >= 0.:
            # crust
            A                   = (self.spl[:self.numbp[1], :self.nlay[1], 1]).T
            b                   = vsinterp[self.nlay[0]:(self.nlay[0]+self.nlay[1])]
            vs0                 = max(vsinterp[self.nlay[0]], 3.0)
            vs1                 = min(vsinterp[self.nlay[0]+self.nlay[1] - 1], 4.2)
            x                   = lsq_linear(A, b, bounds=(vs0, vs1)).x
            self.cvel[:4, 1]    = x[:]
            # mantle
            A                   = (self.spl[:self.numbp[2], :self.nlay[2], 2]).T
            b                   = vsinterp[(self.nlay[0]+self.nlay[1]):]
            vs0                 = max(vsinterp[(self.nlay[0]+self.nlay[1]):].min(), 4.0)
            vs1                 = min(vsinterp[(self.nlay[0]+self.nlay[1]):].max(), vsarr.max())
            x                   = lsq_linear(A, b, bounds=(vs0, vs1)).x
            self.cvel[:5, 2]    = x[:]
        else:
            # crust
            A                   = (self.spl[:self.numbp[2], :self.nlay[2], 2]).T
            b                   = vsinterp[(self.nlay[0]+self.nlay[1]):(self.nlay[0]+self.nlay[1]+self.nlay[2])]
            vs0                 = max(vsinterp[self.nlay[0]+self.nlay[1]], 3.0)
            vs1                 = min(vsinterp[self.nlay[0]+self.nlay[1]+self.nlay[2] - 1], 4.2)
            x                   = lsq_linear(A, b, bounds=(vs0, vs1)).x
            self.cvel[:4, 2]    = x[:]
            # mantle
            A                   = (self.spl[:self.numbp[3], :self.nlay[3], 3]).T
            b                   = vsinterp[(self.nlay[0]+self.nlay[1]+self.nlay[2]):]
            vs0                 = max(vsinterp[(self.nlay[0]+self.nlay[1]+self.nlay[2]):].min(), 4.0)
            vs1                 = min(vsinterp[(self.nlay[0]+self.nlay[1]+self.nlay[2]):].max(), vsarr.max())
            x                   = lsq_linear(A, b, bounds=(vs0, vs1)).x
            self.cvel[:5, 3]    = x[:]
        return
    
    def parameterize_ak135(self, crtthk, sedthk, topovalue=1., maxdepth=200., vp_water=1.5):
        """
        use paramerization from ak135
        ===============================================================================
        ::: input :::
        crtthk      - input crustal thickness (unit - km)
        sedthk      - input sediment thickness (unit - km)
        maxdepth    - maximum depth for the 1-D profile (default - 200 km)
        ::: output :::
        self.thickness  
        self.numbp      - [2, 4, 5]
        self.mtype      - [4, 2, 2]
        self.vpvs       - [2., 1.75, 1.75]
        self.spl
        self.cvel       - determined from ak135
        ===============================================================================
        """
        if topovalue < 0.:
            self.init_arr(4)
            self.thickness[:]   = np.array([-topovalue, sedthk, crtthk - sedthk, maxdepth - crtthk + topovalue])
            self.numbp[:]       = np.array([1, 2, 4, 5])
            self.mtype[:]       = np.array([5, 4, 2, 2])
            self.vpvs[:]        = np.array([0., 2., 1.75, 1.75])
        else:
            self.init_arr(3)
            self.thickness[:]   = np.array([sedthk, crtthk - sedthk, maxdepth - crtthk])
            self.numbp[:]       = np.array([2, 4, 5])
            self.mtype[:]       = np.array([4, 2, 2])
            self.vpvs[:]        = np.array([2., 1.75, 1.75])
        self.update_depth()
        # water layer
        if topovalue < 0.:
            self.cvel[0, 0] = vp_water
        # sediments
        if topovalue >= 0.:
            self.cvel[0, 0] = 1.0
            self.cvel[1, 0] = 1.5
        else:
            self.cvel[0, 1] = 1.0
            self.cvel[1, 1] = 1.5
        # crust and mantle
        if topovalue >= 0.:
            self.cvel[:4, 1]= np.array([3.2, 3.46, 3.85, 3.9])
            self.cvel[:5, 2]= np.array([4.48,4.49, 4.51, 4.52, 4.6])
        else:
            self.cvel[:4, 2]= np.array([3.2, 3.46, 3.85, 3.9])
            self.cvel[:5, 3]= np.array([4.48,4.49, 4.51, 4.52, 4.6])
        return
    
    def parameterize_ray(self, paraval, topovalue=1., maxdepth=200., vp_water=1.5):
        """
        use paramerization from vsv model inferred from Rayleigh wave inversion
        ===============================================================================
        ::: input :::
        crtthk      - input crustal thickness (unit - km)
        sedthk      - input sediment thickness (unit - km)
        maxdepth    - maximum depth for the 1-D profile (default - 200 km)
        ::: output :::
        self.thickness  
        self.numbp      - [2, 4, 5]
        self.mtype      - [4, 2, 2]
        self.vpvs       - [2., 1.75, 1.75]
        self.spl
        self.cvel       - determined from ak135
        ===============================================================================
        """
        if topovalue < 0.:
            self.init_arr(4)
            self.thickness[:]   = np.array([-topovalue, paraval[-2], paraval[-1],\
                                        maxdepth - paraval[-2] - paraval[-1] + topovalue])
            self.numbp[:]       = np.array([1, 2, 4, 5])
            self.mtype[:]       = np.array([5, 4, 2, 2])
            self.vpvs[:]        = np.array([0., 2., 1.75, 1.75])
        else:
            self.init_arr(3)
            self.thickness[:]   = np.array([paraval[-2], paraval[-1], maxdepth - paraval[-2] - paraval[-1]])
            self.numbp[:]       = np.array([2, 4, 5])
            self.mtype[:]       = np.array([4, 2, 2])
            self.vpvs[:]        = np.array([2., 1.75, 1.75])
        self.update_depth()
        # water layer
        if topovalue < 0.:
            self.cvel[0, 0] = vp_water
        # sediments
        if topovalue >= 0.:
            self.cvel[0, 0] = paraval[0]
            self.cvel[1, 0] = paraval[1]
        else:
            self.cvel[0, 1] = paraval[0]
            self.cvel[1, 1] = paraval[1]
        # crust and mantle
        if topovalue >= 0.:
            self.cvel[:4, 1]= paraval[2:6]
            self.cvel[:5, 2]= paraval[6:11]
        else:
            self.cvel[:4, 2]= paraval[2:6]
            self.cvel[:5, 3]= paraval[6:11]
        return

    def get_paraind(self):
        """
        get parameter index arrays for para
        Table 1 and 2 in Shen et al. 2012
        references:
        Shen, W., Ritzwoller, M.H., Schulte-Pelkum, V. and Lin, F.C., 2012.
            Joint inversion of surface wave dispersion and receiver functions: a Bayesian Monte-Carlo approach.
                Geophysical Journal International, 192(2), pp.807-836.
        """
        numbp_sum   = self.numbp.sum()
        npara       = numbp_sum  + self.nmod - 1
        # water layer, added May 15, 2018
        if self.mtype[0] == 5:
            npara   -= 2
        #
        self.para.init_arr(npara)
        ipara       = 0
        for i in range(self.nmod):
            # water layer, added May 15, 2018
            if self.mtype[i] == 5:
                continue
            #--------------------------------
            for j in range(self.numbp[i]):
                self.para.paraindex[0, ipara]   = 0
                if i == 0 or (i == 1 and self.mtype[0] == 5): # water layer, added May 15, 2018
                    # sediment, cvel space is +- 1 km/s, different from Shen et al. 2012
                    self.para.paraindex[1, ipara]   = 1
                    self.para.paraindex[2, ipara]   = 1.
                else:
                    # +- 20 %
                    self.para.paraindex[1, ipara]   = -1
                    self.para.paraindex[2, ipara]   = 20.
                # 0.05 km/s 
                self.para.paraindex[3, ipara]       = 0.05
                self.para.paraindex[4, ipara]       = i
                self.para.paraindex[5, ipara]       = j
                ipara                               +=1
        if self.nmod >= 3:
            # sediment thickness
            self.para.paraindex[0, ipara]   = 1
            self.para.paraindex[1, ipara]   = -1
            self.para.paraindex[2, ipara]   = 100.
            self.para.paraindex[3, ipara]   = 0.1
            if self.mtype[0] == 5: # water layer, added May 15, 2018
                self.para.paraindex[4, ipara]   = 1
            else:
                self.para.paraindex[4, ipara]   = 0
            ipara                           += 1
        # crustal thickness/ +- 20 %
        self.para.paraindex[0, ipara]       = 1
        self.para.paraindex[1, ipara]       = -1
        # # self.para.paraindex[2, ipara]       = 20.
        self.para.paraindex[2, ipara]       = 50.
        self.para.paraindex[3, ipara]       = 1.
        if self.nmod >= 3:
            if self.mtype[0] == 5: # water layer, added May 15, 2018
                self.para.paraindex[4, ipara]   = 2.
            else:
                self.para.paraindex[4, ipara]   = 1.
        else:
            self.para.paraindex[4, ipara]   = 0.
        return
    
    def mod2para(self):
        """
        convert model to parameter arrays for perturbation
        """
        for i in range(self.para.npara):
            ig      = int(self.para.paraindex[4, i])
            # velocity coefficient 
            if int(self.para.paraindex[0, i]) == 0:
                ip  = int(self.para.paraindex[5, i])
                val = self.cvel[ip][ig]
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == 1:
                val = self.thickness[ig]
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == -1:
                val = self.vpvs[ig]
            else:
                print 'Unexpected value in paraindex!'
            self.para.paraval[i] = val
            #-------------------------------------------
            # defining parameter space for perturbation
            #-------------------------------------------
            if not self.para.isspace:
                step        = self.para.paraindex[3, i]
                if int(self.para.paraindex[1, i]) == 1:
                    valmin  = val - self.para.paraindex[2, i]
                    valmax  = val + self.para.paraindex[2, i]
                else:
                    # bugfix on 2019/01/18
                    if val < 0.001 and int(self.para.paraindex[0, i]) == 1: # 0 value for sediment thickness
                        valmin  = 0. 
                        valmax  = 0.1
                    else:
                        valmin  = val - val*self.para.paraindex[2, i]/100.
                        valmax  = val + val*self.para.paraindex[2, i]/100.
                valmin          = max (0., valmin)
                valmax          = max (valmin + 0.0001, valmax)
                if (int(self.para.paraindex[0, i]) == 0 and i == 0 \
                    and int(self.para.paraindex[5, i]) == 0): # if it is the upper sedi:
                    valmin              = max (0.2, valmin)
                    valmax              = max (0.5, valmax) 
                self.para.space[0, i]   = valmin
                self.para.space[1, i]   = valmax
                self.para.space[2, i]   = step
        self.para.isspace               = True
        return
    
    def para2mod(self):
        """
        Convert paratemers (for perturbation) to model parameters
        """
        for i in range(self.para.npara):
            val = self.para.paraval[i]
            ig  = int(self.para.paraindex[4, i])
            # velocity coeficient for splines
            if int(self.para.paraindex[0, i]) == 0:
                ip                  = int(self.para.paraindex[5, i])
                self.cvel[ip][ig]   = val
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == 1:
                self.thickness[ig]  = val
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == -1:
                self.vpvs[ig]       = val
            else:
                print('Unexpected value in paraindex!')
        self.thickness[-1]          = 200. - self.thickness[:-1].sum()
        return
    
    def isgood_test(self, m0, m1, g0, g1, dvs_thresh=0.05):
        """
        check the model is good or not
        ==========================================================================
        ::: input   :::
        m0, m1  - index of group for monotonic change checking
        g0, g1  - index of group for gradient change checking
        ==========================================================================
        """
        # velocity constrast, contraint (5) in 4.2 of Shen et al., 2012
        for i in range (self.nmod-1):
            nlay        = self.nlay[i]
            if self.vs[0, i+1] < self.vs[nlay-1, i]:
                return False
        #
        # Vs < 4.9 km/sec , contraint (6) in 4.2 of Shen et al., 2012
        if np.any(self.vs > 4.9):
            return False
        #
        if m1 >= self.nmod:
            m1  = self.nmod -1
        if m0 < 0:
            m0  = 0
        if g1 >= self.nmod:
            g1  = self.nmod -1
        if g0 < 0:
            g0  = 0
        # monotonic change
        # contraint (3) and (4) in 4.2 of Shen et al., 2012
        if m0 <= m1:
            for j in range(m0, m1+1):
                vs0     = self.vs[:self.nlay[j]-1, j]
                vs1     = self.vs[1:self.nlay[j], j]
                if np.any(np.greater(vs0, vs1)):
                    return False
        # constrain the last layer Vs in crust
        nlay_crust      = self.nlay[self.nmod-2]
        if self.vs[nlay_crust-1, self.nmod-2] > 4.3:
            return False
        # constrain the first layer Vs in mantle
        if self.vs[0, self.nmod-1] > 4.6:
            return False
        if self.vs[0, self.nmod-1] < 4.0:
            return False
        #
        # constrain the bottom layer Vs in mantle
        nlay_mantle     = self.nlay[self.nmod-1]
        if self.vs[nlay_mantle-1, self.nmod-1] < 4.3:
            return False

        #-------------------------------------------------------------------
        # penalize oscillations with differences in local/maximum extrema 
        #-------------------------------------------------------------------
        dv_osci = 0.01
        hArr    = np.zeros(self.nlay.sum(), dtype = np.float64)
        vs      = np.zeros(self.nlay.sum(), dtype = np.float64)
        for i in range(self.nmod):
            if i == 0:
                hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
            elif i < self.nmod - 1:
                hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
            else:
                hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
            if self.mtype[i] == 5 and i == 0:
                vs[0]                   = 0.
            elif (i == 0 and self.mtype[i] != 5):
                vs[:self.nlay[0]]       = self.vs[:self.nlay[i], i]
            elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2:
                vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
            elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2:
                vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
            elif i < self.nmod - 1:
                vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
            # changed on 2019/01/17, Hacker & Abers, 2004
            else:
                vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
        depth           = hArr.cumsum()
        
        vs_mantle       = self.vs[:self.nlay[self.nmod-1], self.nmod-1]
        local_indmax    = scipy.signal.argrelmax(vs_mantle)[0]
        local_indmin    = scipy.signal.argrelmin(vs_mantle)[0]
        if local_indmin.size > 0 and local_indmax.size > 0:
            if local_indmin.size == local_indmax.size:
                vmin    = vs_mantle[local_indmin]
                vmax    = vs_mantle[local_indmax]
            else:
                Ndiff   = local_indmax.size - local_indmin.size
                if Ndiff > 0:
                    vmin    = vs_mantle[local_indmin]
                    vmax    = vs_mantle[local_indmax[:-Ndiff]]
                else:
                    vmin    = vs_mantle[local_indmin[:Ndiff]]
                    vmax    = vs_mantle[local_indmax]
            if (vmax-vmin).max() > dv_osci and (local_indmax.size + local_indmin.size) >= 3:
                return False
        vs_trim         = vs[depth > 60.]
        local_indmax    = scipy.signal.argrelmax(vs_trim)[0]
        local_indmin    = scipy.signal.argrelmin(vs_trim)[0]
        # if local_indmax.size >= 1 and local_indmin.size >= 1:
        #     if abs(vs_trim[local_indmax].max() - vs_trim[local_indmin].min())>= dv_osci:
        #         return False
        ###
        # vs_trim         = vs[depth > 80.]
        # if np.any(vs_trim<4.0):
        #     return False
        
        return True
    
    def isgood(self, m0, m1, g0, g1, dvs_thresh=0.05):
        """
        check the model is good or not
        ==========================================================================
        ::: input   :::
        m0, m1  - index of group for monotonic change checking
        g0, g1  - index of group for gradient change checking
        ==========================================================================
        """
        # velocity constrast, contraint (5) in 4.2 of Shen et al., 2012
        for i in range (self.nmod-1):
            nlay        = self.nlay[i]
            if self.vs[0, i+1] < self.vs[nlay-1, i]:
                return False
        #
        # Vs < 4.9 km/sec , contraint (6) in 4.2 of Shen et al., 2012
        if np.any(self.vs > 4.9):
            return False
        #
        if m1 >= self.nmod:
            m1  = self.nmod -1
        if m0 < 0:
            m0  = 0
        if g1 >= self.nmod:
            g1  = self.nmod -1
        if g0 < 0:
            g0  = 0
        # monotonic change
        # contraint (3) and (4) in 4.2 of Shen et al., 2012
        if m0 <= m1:
            for j in range(m0, m1+1):
                vs0     = self.vs[:self.nlay[j]-1, j]
                vs1     = self.vs[1:self.nlay[j], j]
                if np.any(np.greater(vs0, vs1)):
                    return False
        # gradient check, positive gradient
        # if g0<=g1:
        #     for j in range(g0, g1+1):
        #         if self.vs[0, j] > self.vs[1, j]:
        #             return False
        # gradient check, negative gradient
        # if g0<=g1:
        #     for j in range(g0, g1+1):
        #         if self.vs[0, j] < self.vs[1, j]:
        #             return False
        #--------------------------------------
        # new constraints added, Sep 7th, 2018
        #--------------------------------------
        # constrain the last layer Vs in crust
        nlay_crust      = self.nlay[self.nmod-2]
        if self.vs[nlay_crust-1, self.nmod-2] > 4.3:
            return False
        # constrain the first layer Vs in mantle
        if self.vs[0, self.nmod-1] > 4.6:
            return False
        if self.vs[0, self.nmod-1] < 4.0:
            return False
        #
        # constrain the bottom layer Vs in mantle
        nlay_mantle     = self.nlay[self.nmod-1]
        if self.vs[nlay_mantle-1, self.nmod-1] < 4.3:
            return False
        #
        #--------------------------------------
        # curvature constraints in the mantle
        #--------------------------------------
        # Nknot_mantle    = self.Nknot[-1]
        # t               = self.knot_vector[:Nknot_mantle, -1]
        # c               = self.para.paraval[6:11]
        # k               = t.size - c.size - 1
        # depth           = np.arange(self.nlay[-1])*self.thickness[-1]/float(self.nlay[-1]-1)
        # bspl            = scipy.interpolate.BSpline(t=t, c=c, k=k)
        # curvature       = bspl.derivative(2)(depth)
        # if curvature.max() - curvature.min() > thresh_cur or abs(curvature.max()) > thresh_cur or abs(curvature.min()) > thresh_cur:
        #     return False
        #-------------------------------------------------------------------
        # penalize oscillations with differences in local/maximum extrema 
        #-------------------------------------------------------------------
        dv_osci = 0.01
        hArr    = np.zeros(self.nlay.sum(), dtype = np.float64)
        vs      = np.zeros(self.nlay.sum(), dtype = np.float64)
        for i in range(self.nmod):
            if i == 0:
                hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
            elif i < self.nmod - 1:
                hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
            else:
                hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
            if self.mtype[i] == 5 and i == 0:
                vs[0]                   = 0.
            elif (i == 0 and self.mtype[i] != 5):
                vs[:self.nlay[0]]       = self.vs[:self.nlay[i], i]
            elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2:
                vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
            elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2:
                vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
            elif i < self.nmod - 1:
                vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
            # changed on 2019/01/17, Hacker & Abers, 2004
            else:
                vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
        depth           = hArr.cumsum()
        
        vs_mantle       = self.vs[:self.nlay[self.nmod-1], self.nmod-1]
        local_indmax    = scipy.signal.argrelmax(vs_mantle)[0]
        local_indmin    = scipy.signal.argrelmin(vs_mantle)[0]
        if local_indmin.size > 0 and local_indmax.size > 0:
            if local_indmin.size == local_indmax.size:
                vmin    = vs_mantle[local_indmin]
                vmax    = vs_mantle[local_indmax]
            else:
                Ndiff   = local_indmax.size - local_indmin.size
                if Ndiff > 0:
                    vmin    = vs_mantle[local_indmin]
                    vmax    = vs_mantle[local_indmax[:-Ndiff]]
                else:
                    vmin    = vs_mantle[local_indmin[:Ndiff]]
                    vmax    = vs_mantle[local_indmax]
            if (vmax-vmin).max() > dv_osci and (local_indmax.size + local_indmin.size) >= 3:
                return False
        vs_trim         = vs[depth > 60.]
        local_indmax    = scipy.signal.argrelmax(vs_trim)[0]
        local_indmin    = scipy.signal.argrelmin(vs_trim)[0]
        if local_indmax.size >= 1 and local_indmin.size >= 1:
            if abs(vs_trim[local_indmax].max() - vs_trim[local_indmin].min())>= dv_osci:
                return False
        ###
        # vs_trim         = vs[depth > 80.]
        # if np.any(vs_trim<4.0):
        #     return False
        
        #########################
        # vs_mantle       = self.vs[:self.nlay[self.nmod-1], self.nmod-1]
        # local_indmax    = scipy.signal.argrelmax(vs_mantle)[0]
        # local_indmin    = scipy.signal.argrelmin(vs_mantle)[0]
        # if local_indmin.size > 0 and local_indmax.size > 0:
        #     if local_indmin.size == local_indmax.size:
        #         vmin    = vs_mantle[local_indmin]
        #         vmax    = vs_mantle[local_indmax]
        #     else:
        #         Ndiff   = local_indmax.size - local_indmin.size
        #         if Ndiff > 0:
        #             vmin    = vs_mantle[local_indmin]
        #             vmax    = vs_mantle[local_indmax[:-Ndiff]]
        #         else:
        #             vmin    = vs_mantle[local_indmin[:Ndiff]]
        #             vmax    = vs_mantle[local_indmax]
        #     if (vmax-vmin).max() > dvs_thresh*vs_mantle.mean():
        #         return False
        # if (vs_mantle.max() - vs_mantle.min()) > 0.15*vs_mantle.mean():
        #     return False
        # ###
        # # # nlay    = self.nlay.sum()
        # # # hArr    = np.zeros(nlay, dtype = np.float64)
        # # # vs      = np.zeros(nlay, dtype = np.float64)
        # # # for i in range(self.nmod):
        # # #     if i == 0:
        # # #         hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
        # # #     elif i < self.nmod - 1:
        # # #         hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
        # # #     else:
        # # #         hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
        # # #     if self.mtype[i] == 5 and i == 0:
        # # #         vs[0]                   = 0.
        # # #     elif (i == 0 and self.mtype[i] != 5):
        # # #         vs[:self.nlay[0]]       = self.vs[:self.nlay[i], i]
        # # #     elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2:
        # # #         vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
        # # #     elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2:
        # # #         vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
        # # #     elif i < self.nmod - 1:
        # # #         vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
        # # #     # changed on 2019/01/17, Hacker & Abers, 2004
        # # #     else:
        # # #         vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
        # # # depth   = hArr.cumsum()
        # # # index   = (depth <= 120.)*(depth>=80.)
        # # # if np.any(vs[index] > 4.5):
        # # #     return False
        
        return True
    
    def get_vmodel(self):
        """
        get velocity models
        ==========================================================================
        ::: output :::
        hArr, vs, vp, rho, qs, qp
        ==========================================================================
        """
        nlay    = self.nlay.sum()
        hArr    = np.zeros(nlay, dtype = np.float64)
        vs      = np.zeros(nlay, dtype = np.float64)
        vp      = np.zeros(nlay, dtype = np.float64)
        rho     = np.zeros(nlay, dtype = np.float64)
        qs      = np.zeros(nlay, dtype = np.float64)
        qp      = np.zeros(nlay, dtype = np.float64)
        depth   = np.zeros(nlay, dtype = np.float64)
        for i in range(self.nmod):
            if i == 0:
                hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
            elif i < self.nmod - 1:
                hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
            else:
                hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
            if self.mtype[i] == 5 and i == 0:
                vs[0]                   = 0.
                vp[0]                   = self.cvel[0][i]
                rho[0]                  = 1.02
                qs[0]                   = 10000.
                qp[0]                   = 57822.
            elif (i == 0 and self.mtype[i] != 5):
                vs[:self.nlay[0]]       = self.vs[:self.nlay[i], i]
                vp[:self.nlay[0]]       = self.vs[:self.nlay[i], i]*self.vpvs[i]
                rho[:self.nlay[0]]      = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
                qs[:self.nlay[0]]       = 80.*np.ones(self.nlay[i], dtype=np.float64)
                qp[:self.nlay[0]]       = 160.*np.ones(self.nlay[i], dtype=np.float64)
            elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2:
                vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]*self.vpvs[i]
                rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
                qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 80.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 160.*np.ones(self.nlay[i], dtype=np.float64)
            elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2:
                vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
                rho[self.nlay[:i].sum():]   = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
                qs[self.nlay[:i].sum():]    = 80.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():]    = 160.*np.ones(self.nlay[i], dtype=np.float64)
            elif i < self.nmod - 1:
                vs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = self.vs[:self.nlay[i], i]*self.vpvs[i]
                rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
                qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 600.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 1400.*np.ones(self.nlay[i], dtype=np.float64)
            # # # else:
            # # #     vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
            # # #     vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
            # # #     rho[self.nlay[:i].sum():]   = 0.541 + 0.3601*self.vs[:self.nlay[i], i]*self.vpvs[i]
            # # #     qs[self.nlay[:i].sum():]    = 600.*np.ones(self.nlay[i], dtype=np.float64)
            # # #     qp[self.nlay[:i].sum():]    = 1400.*np.ones(self.nlay[i], dtype=np.float64)
            # changed on 2019/01/17, Hacker & Abers, 2004
            else:
                vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
                rho[self.nlay[:i].sum():]   = 3.4268 + (self.vs[:self.nlay[i], i] - 4.5)/4.5 
                qs[self.nlay[:i].sum():]    = 150.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():]    = 1400.*np.ones(self.nlay[i], dtype=np.float64)
        depth               = hArr.cumsum()
        # changed on 2019/01/17
        # # # rho[vp > 7.5]       = 3.35 
        return hArr, vs, vp, rho, qs, qp, nlay
    
    
class vtimod(object):
    """
    An object for handling parameterization of 1D Vertical TI model for the inversion
    =====================================================================================================================
    ::: parameters :::
    :   numbers     :
    nmod        - number of model groups
    maxlay      - maximum layers for each group (default - 100)
    maxspl      - maximum spline coefficients for each group (default - 20)
    use_gamma   - use gamma array to represent radial anisotropy or not
    :   1D arrays   :
    numbp       - number of control points/basis (1D int array with length nmod)
    mtype       - model parameterization types (1D int array with length nmod)
                    1   - layer         - nlay  = numbp, hArr = ratio*thickness, vs = cvel
                    2   - B-splines     - hArr  = thickness/nlay, vs    = (cvel*spl)_sum over numbp
                    4   - gradient layer- nlay is defined depends on thickness
                                            hArr  = thickness/nlay, vs  = from cvel[0, i] to cvel[1, i]
                    5   - water         - nlay  = 1, vs = 0., hArr = thickness
    thickness   - thickness of each group (1D float array with length nmod)
    nlay        - number of layres in each group (1D int array with length nmod)
    vpvs        - vp/vs ratio in each group (1D float array with length nmod)
    ----------------------------------------------------------------------------
    gamma       - radial anisotropy, defined as 2*(Vsh - Vsv) /(Vsh + Vsv)
    ----------------------------------------------------------------------------
    isspl       - flag array indicating the existence of basis B spline (1D int array with length nmod)
                    0 - spline basis has NOT been computed
                    1 - spline basis has been computed
    :   multi-dim arrays    :
    t           - knot vectors for B splines (2D array - [:(self.numb[i]+degBs), i]; i indicating group id)
    spl         - B spline basis array (3D array - [:self.numb[i], :self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 2
    ratio       - array for the ratio of each layer (2D array - [:self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 1
    cvph        - vph velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    cvpv        - vpv velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    cvsh        - vsh velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    cvsv        - vsv velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    ceta        - eta coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    crho        - density coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
                    layer mod   - input velocities for each layer
                    spline mod  - coefficients for B spline
                    gradient mod- top/bottom layer velocity             
    :   model arrays        :
    vph         - vph array         (2D array - [:self.nlay[i], i]; i indicating group id)
    vpv         - vpv array         (2D array - [:self.nlay[i], i]; i indicating group id)
    vsh         - vsh array         (2D array - [:self.nlay[i], i]; i indicating group id)
    vsv         - vsv array         (2D array - [:self.nlay[i], i]; i indicating group id)
    eta         - eta array         (2D array - [:self.nlay[i], i]; i indicating group id)
    rho         - rho array         (2D array - [:self.nlay[i], i]; i indicating group id)
    hArr        - layer arrays      (2D array - [:self.nlay[i], i]; i indicating group id)
    :   para1d  :
    para        - object storing parameters for perturbation
    *********************************************************************************************************************
    initialization      : (1) self.parameterize_ak135(); (2) self.get_paraind(); (3) self.upate(); (4) self.get_vmodel();
                          (5) self.mod2para()
    paraval -> model    : (1) self.para2mod(); (2) self.update(); (3) self.get_vmodel()
    model   -> paraval  : (1) self.mod2para()
    =====================================================================================================================
    """
    
    def __init__(self):
        self.nmod       = 0
        self.maxlay     = 100
        self.maxspl     = 20
        self.para       = para1d()
        self.use_gamma  = False
        return
    
    def init_arr(self, nmod):
        """
        initialization of arrays
        """
        self.nmod       = nmod
        # arrays of size nmod
        self.numbp      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        self.mtype      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        self.thickness  = np.zeros(np.int64(self.nmod), dtype=np.float32)
        self.nlay       = np.ones(np.int64(self.nmod),  dtype=np.int32)*np.int32(20) 
        self.vpvs       = np.ones(np.int64(self.nmod),  dtype=np.float32)*np.float32(1.75)
        # radial anisotropy: (Vsh - Vsv) / ((Vsh + Vsv)/2), added on 2019-03-26
        self.gamma      = np.zeros(np.int64(self.nmod),  dtype=np.float32)
        self.isspl      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        # arrays of size maxspl, nmod
        self.cvph       = np.zeros((np.int64(self.maxspl),  np.int64(self.nmod)), dtype = np.float32)
        self.cvpv       = np.zeros((np.int64(self.maxspl),  np.int64(self.nmod)), dtype = np.float32)
        self.cvsh       = np.zeros((np.int64(self.maxspl),  np.int64(self.nmod)), dtype = np.float32)
        self.cvsv       = np.zeros((np.int64(self.maxspl),  np.int64(self.nmod)), dtype = np.float32)
        # # self.ceta       = np.zeros((np.int64(self.maxspl),  np.int64(self.nmod)), dtype = np.float32)
        # # self.crho       = np.zeros((np.int64(self.maxspl),  np.int64(self.nmod)), dtype = np.float32)
        # arrays of size maxlay, nmod
        self.ratio      = np.zeros((np.int64(self.maxlay),  np.int64(self.nmod)), dtype = np.float32)
        # # # self.vph        = np.zeros((np.int64(self.maxlay),  np.int64(self.nmod)), dtype = np.float32)
        # # # self.vpv        = np.zeros((np.int64(self.maxlay),  np.int64(self.nmod)), dtype = np.float32)
        self.vsh        = np.zeros((np.int64(self.maxlay),  np.int64(self.nmod)), dtype = np.float32)
        self.vsv        = np.zeros((np.int64(self.maxlay),  np.int64(self.nmod)), dtype = np.float32)
        # # # self.eta        = np.zeros((np.int64(self.maxlay),  np.int64(self.nmod)), dtype = np.float32)
        # # # self.rho        = np.zeros((np.int64(self.maxlay),  np.int64(self.nmod)), dtype = np.float32)
        self.hArr       = np.zeros((np.int64(self.maxlay),  np.int64(self.nmod)), dtype = np.float32)
        # arrays of size maxspl, maxlay, nmod
        self.spl        = np.zeros((np.int64(self.maxspl),  np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.knot_vector= np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        self.Nknot      = np.zeros((self.nmod), dtype = np.int64)
        return
    
    def bspline(self, i):
        """
        Compute B-spline basis given group id
        The actual degree is k = degBs - 1
        e.g. nBs = 5, degBs = 4, k = 3, cubic B spline
        ::: output :::
        self.spl    - (nBs+k, npts)
                        [:nBs, :] B spline basis for nBs control points
                        [nBs:, :] can be ignored
        """    
        if self.thickness[i] >= 150:
            self.nlay[i]    = 60
        elif self.thickness[i] < 10:
            self.nlay[i]    = 5
        elif self.thickness[i] < 20:
            self.nlay[i]    = 10
        else:
            self.nlay[i]    = 30
        if self.isspl[i]:
            print('spline basis already exists!')
            return
        if self.mtype[i] != 2:
            print('Not spline parameterization!')
            return 
        # initialize
        if i >= self.nmod:
            raise ValueError('index for spline group out of range!')
            return
        nBs         = self.numbp[i]
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness[i]
        disfacBs    = 2.
        npts        = self.nlay[i]
        nbasis, t   = bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        m           = nBs-1+degBs
        if m > self.maxspl:
            raise ValueError('number of splines is too large, change default maxspl!')
        self.spl[:nBs, :npts, i]= nbasis[:nBs, :]
        self.isspl[i]           = True
        self.knot_vector[:(nBs+degBs), i]\
                                = t
        self.Nknot[i]           = t.size
        return True

    def update(self):
        """
        Update model (velocities and hArr arrays), from the thickness, cvel
        """
        # ratio of Vsh/Vsv
        hv_ratio    = (1. + self.gamma/200.)/(1 - self.gamma/200.)
        for i in xrange(self.nmod):
            if self.nlay[i] > self.maxlay:
                raise ValueError('number of layers is too large, need change default maxlay!')
            # layered model
            if self.mtype[i] == 1:
                self.nlay[i]                    = self.numbp[i]
                self.hArr[:, i]                 = self.ratio[:, i] * self.thickness[i]
                if self.use_gamma:
                    self.vsv[:self.nlay[i], i]  = hv_ratio[i] * self.cvsv[:self.nlay[i], i]
                else:
                    self.vsh[:self.nlay[i], i]  = self.cvsh[:self.nlay[i], i]
                self.vsv[:self.nlay[i], i]      = self.cvsv[:self.nlay[i], i]
                tnlay                           = self.nlay[i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i]   = False
                self.bspline(i)
                self.vsv[:self.nlay[i], i]      = np.dot( (self.spl[:self.numbp[i], :self.nlay[i], i]).T, self.cvsv[:self.numbp[i], i])
                if self.use_gamma:
                    self.vsh[:self.nlay[i], i]  = hv_ratio[i] * self.vsv[:self.nlay[i], i] 
                else:
                    self.vsh[:self.nlay[i], i]  = np.dot( (self.spl[:self.numbp[i], :self.nlay[i], i]).T, self.cvsh[:self.numbp[i], i])
                self.hArr[:self.nlay[i], i]     = self.thickness[i]/self.nlay[i]
            # gradient layer
            elif self.mtype[i] == 4:
                nlay 	    = 4
                if self.thickness[i] >= 20.:
                    nlay    = 20
                if self.thickness[i] > 10. and self.thickness[i] < 20.:
                    nlay    = int(self.thickness[i]/1.)
                if self.thickness[i] > 2. and self.thickness[i] <= 10.:
                    nlay    = int(self.thickness[i]/0.5)
                if self.thickness[i] < 0.5:
                    nlay    = 2
                dh 	                    = self.thickness[i]/float(nlay)
                dcvsv 		            = (self.cvsv[1, i] - self.cvsv[0, i])/(nlay - 1.)
                self.vsv[:nlay, i]      = self.cvsv[0, i] + dcvsv*np.arange(nlay, dtype=np.float64)
                if self.use_gamma:          
                    self.vsh[:nlay, i]  = hv_ratio[i] * self.vsv[:nlay, i]
                else:
                    dcvsh 		        = (self.cvsh[1, i] - self.cvsh[0, i])/(nlay - 1.)
                    self.vsh[:nlay, i]  = self.cvsh[0, i] + dcvsh*np.arange(nlay, dtype=np.float64)
                # eta should not be gradient changing
                self.hArr[:nlay, i]     = dh
                self.nlay[i]            = nlay
            # water layer
            elif self.mtype[i] == 5:
                nlay                = 1
                self.vsh[0, i]      = 0.
                self.vsv[0, i]      = 0.
                self.hArr[0, i]     = self.thickness[i]
                self.nlay[i]        = 1
        return

    def update_depth(self):
        """
        update hArr arrays only, used for paramerization of a refernce input model
        """
        for i in range(self.nmod):
            if self.nlay[i] > self.maxlay:
                printf('number of layers is too large, need change default maxlay!')
                return False
            # layered model
            if self.mtype[i] == 1:
                self.nlay[i]                = self.numbp[i]
                self.hArr[:self.nlay[i], i] = self.ratio[:self.nlay[i], i] * self.thickness[i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i]   = False
                self.bspline(i)
                self.hArr[:self.nlay[i], i] = self.thickness[i]/self.nlay[i]
            # gradient layer
            elif self.mtype[i] == 4:
                nlay 	    = 4
                if self.thickness[i] >= 20.:
                    nlay    = 20
                if self.thickness[i] > 10. and self.thickness[i] < 20.:
                    nlay    = int(self.thickness[i]/1.)
                if self.thickness[i] > 2. and self.thickness[i] <= 10.:
                    nlay    = int(self.thickness[i]/0.5)
                if self.thickness[i] < 0.5:
                    nlay    = 2
                dh 	                    = self.thickness[i]/float(nlay)
                self.hArr[:nlay, i]     = dh
                self.nlay[i]            = nlay
            # water layer
            elif self.mtype[i] == 5:
                nlay                    = 1
                self.hArr[0, i]         = self.thickness[i]
                self.nlay[i]            = 1
        return True
    
    def parameterize_ak135(self, crtthk, sedthk, topovalue=1., maxdepth=200., vp_water=1.5):
        """
        use paramerization from ak135
        ===============================================================================
        ::: input :::
        crtthk      - input crustal thickness (unit - km)
        sedthk      - input sediment thickness (unit - km)
        maxdepth    - maximum depth for the 1-D profile (default - 200 km)
        ::: output :::
        self.thickness  
        self.numbp      - [2, 4, 5]
        self.mtype      - [4, 2, 2]
        self.vpvs       - [2., 1.75, 1.75]
        self.spl
        self.cvel       - determined from ak135
        ===============================================================================
        """
        if topovalue < 0.:
            self.init_arr(4)
            self.thickness[:]   = np.array([-topovalue, sedthk, crtthk - sedthk, maxdepth - crtthk + topovalue])
            self.numbp[:]       = np.array([1, 2, 4, 5])
            self.mtype[:]       = np.array([5, 4, 2, 2])
            self.vpvs[:]        = np.array([0., 2., 1.75, 1.75])
        else:
            self.init_arr(3)
            self.thickness[:]   = np.array([sedthk, crtthk - sedthk, maxdepth - crtthk])
            self.numbp[:]       = np.array([2, 4, 5])
            self.mtype[:]       = np.array([4, 2, 2])
            self.vpvs[:]        = np.array([2., 1.75, 1.75])
        self.update_depth()
        # water layer
        if topovalue < 0.:
            self.cvph[0, 0] = vp_water
            self.cvpv[0, 0] = vp_water
        # sediments
        if topovalue >= 0.:
            self.cvsh[0, 0] = 1.0
            self.cvsv[0, 0] = 1.0
            self.cvsh[1, 0] = 1.5
            self.cvsv[1, 0] = 1.5
        else:
            self.cvsh[0, 1] = 1.0
            self.cvsv[0, 1] = 1.0
            self.cvsh[1, 1] = 1.5
            self.cvsv[1, 1] = 1.5
        # crust and mantle
        if topovalue >= 0.:
            self.cvsh[:4, 1]= np.array([3.2, 3.46, 3.85, 3.9])
            self.cvsh[:5, 2]= np.array([4.48,4.49, 4.51, 4.52, 4.6])
            self.cvsv[:4, 1]= np.array([3.2, 3.46, 3.85, 3.9])
            self.cvsv[:5, 2]= np.array([4.48,4.49, 4.51, 4.52, 4.6])
        else:
            self.cvsh[:4, 2]= np.array([3.2, 3.46, 3.85, 3.9])
            self.cvsh[:5, 3]= np.array([4.48,4.49, 4.51, 4.52, 4.6])
            self.cvsv[:4, 2]= np.array([3.2, 3.46, 3.85, 3.9])
            self.cvsv[:5, 3]= np.array([4.48,4.49, 4.51, 4.52, 4.6])
        return
    
    def parameterize_ray(self, paraval, topovalue=1., maxdepth=200., vp_water=1.5):
        """
        use paramerization from vsv model inferred from Rayleigh wave inversion
        ===============================================================================
        ::: input :::
        crtthk      - input crustal thickness (unit - km)
        sedthk      - input sediment thickness (unit - km)
        maxdepth    - maximum depth for the 1-D profile (default - 200 km)
        ::: output :::
        self.thickness  
        self.numbp      - [2, 4, 5]
        self.mtype      - [4, 2, 2]
        self.vpvs       - [2., 1.75, 1.75]
        self.spl
        self.cvel       - determined from ak135
        ===============================================================================
        """
        if topovalue < 0.:
            self.init_arr(4)
            self.thickness[:]   = np.array([-topovalue, paraval[-2], paraval[-1],\
                                        maxdepth - paraval[-2] - paraval[-1] + topovalue])
            self.numbp[:]       = np.array([1, 2, 4, 5])
            self.mtype[:]       = np.array([5, 4, 2, 2])
            self.vpvs[:]        = np.array([0., 2., 1.75, 1.75])
        else:
            self.init_arr(3)
            self.thickness[:]   = np.array([paraval[-2], paraval[-1], maxdepth - paraval[-2] - paraval[-1]])
            self.numbp[:]       = np.array([2, 4, 5])
            self.mtype[:]       = np.array([4, 2, 2])
            self.vpvs[:]        = np.array([2., 1.75, 1.75])
        self.update_depth()
        # water layer
        if topovalue < 0.:
            self.cvph[0, 0] = vp_water
            self.cvpv[0, 0] = vp_water
        # sediments
        if topovalue >= 0.:
            self.cvsh[0, 0] = paraval[0]
            self.cvsv[0, 0] = paraval[0]
            self.cvsh[1, 0] = paraval[1]
            self.cvsv[1, 0] = paraval[1]
        else:
            self.cvsh[0, 1] = paraval[0]
            self.cvsv[0, 1] = paraval[0]
            self.cvsh[1, 1] = paraval[1]
            self.cvsv[1, 1] = paraval[1]
        # crust and mantle
        if topovalue >= 0.:
            self.cvsh[:4, 1]= paraval[2:6]
            self.cvsh[:5, 2]= paraval[6:11]
            self.cvsv[:4, 1]= paraval[2:6]
            self.cvsv[:5, 2]= paraval[6:11]
        else:
            self.cvsh[:4, 2]= paraval[2:6]
            self.cvsh[:5, 3]= paraval[6:11]
            self.cvsv[:4, 2]= paraval[2:6]
            self.cvsv[:5, 3]= paraval[6:11]
        return

    def get_paraind(self, perturb_thk=False, perturb_vsv=True):
        """
        get parameter index arrays for para
        =============================================================================================================================
        paraindex[0, :] - type of parameters
                            0   - vsh coefficient for splines
                            1   - vsv coefficient for splines
                            2   - thickness
                            3   - gamma = 2(Vsh-Vsv)/(Vsh+Vsv)
        paraindex[1, :] - index for type of amplitude for parameter perturbation
                            1   - absolute
                            -1  - relative
                            0   - fixed, do NOT perturb, added on 2019-03-19              
        paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
        paraindex[3, :] - step for parameter space 
        paraindex[4, :] - index for the parameter in the model group   
        paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0
        =============================================================================================================================
        """
        # water layer
        if self.mtype[0] == 5:
            npara   = (self.numbp[2:]).sum()*2 + 3
        else:
            npara   = (self.numbp[1:]).sum()*2 + 3
        self.para.init_arr(npara)
        ipara       = 0
        for i in xrange(self.nmod):
            # water layer
            if self.mtype[i] == 5:
                continue
            for j in xrange(self.numbp[i]):
                if i == 0 or (i == 1 and self.mtype[0] == 5): # water layer, added May 15, 2018
                    self.para.paraindex[0, ipara]   = 1
                    # sediment, parameter space is +- 0.5 km/s, different from Shen et al. 2012
                    if perturb_vsv:
                        self.para.paraindex[1, ipara]   = 1
                    else:
                        self.para.paraindex[1, ipara]   = 0
                    self.para.paraindex[2, ipara]   = .5
                    # 0.05 km/s 
                    self.para.paraindex[3, ipara]   = 0.05
                    self.para.paraindex[4, ipara]   = i
                    self.para.paraindex[5, ipara]   = j
                    ipara                           +=1
                elif (i == 1 and self.mtype[0] == 4 and j == 0) or (i == 2 and self.mtype[0] == 5 and j == 0):
                    # vsv
                    self.para.paraindex[0, ipara]   = 1
                    if perturb_vsv:
                        self.para.paraindex[1, ipara]   = -1
                    else:
                        self.para.paraindex[1, ipara]   = 0
                    self.para.paraindex[2, ipara]   = 5. # +- 5 % for vsv
                    self.para.paraindex[3, ipara]   = 0.05
                    self.para.paraindex[4, ipara]   = i
                    self.para.paraindex[5, ipara]   = j
                    ipara   +=1
                else:
                    # vsh
                    self.para.paraindex[0, ipara]   = 0
                    self.para.paraindex[1, ipara]   = -1
                    self.para.paraindex[2, ipara]   = 15. # +- 15 %
                    self.para.paraindex[3, ipara]   = 0.05
                    self.para.paraindex[4, ipara]   = i
                    self.para.paraindex[5, ipara]   = j
                    ipara   +=1
                    # vsv
                    self.para.paraindex[0, ipara]   = 1
                    self.para.paraindex[1, ipara]   = -1
                    self.para.paraindex[2, ipara]   = 5. # +- 5 % for vsv
                    self.para.paraindex[3, ipara]   = 0.05
                    self.para.paraindex[4, ipara]   = i
                    self.para.paraindex[5, ipara]   = j
                    ipara   +=1
        if self.nmod >= 3:
            # sediment thickness/ +- 50 %
            self.para.paraindex[0, ipara]       = 2
            if perturb_thk:
                self.para.paraindex[1, ipara]   = -1 # perturb flag
            else:
                self.para.paraindex[1, ipara]   = 0 # perturb flag
            self.para.paraindex[2, ipara]       = 50.
            self.para.paraindex[3, ipara]       = 0.1
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 1
            else:
                self.para.paraindex[4, ipara]   = 0
            ipara                               += 1
        # crustal thickness/ +- 10 %
        self.para.paraindex[0, ipara]           = 2
        if perturb_thk:
            self.para.paraindex[1, ipara]       = -1 # perturb flag
        else:
            self.para.paraindex[1, ipara]       = 0 # perturb flag
        self.para.paraindex[2, ipara]           = 10.
        self.para.paraindex[3, ipara]           = 1.
        if self.nmod >= 3:
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 2.
            else:
                self.para.paraindex[4, ipara]   = 1.
        else:
            self.para.paraindex[4, ipara]       = 0.            
        return
        
    def get_paraind_gamma(self, perturb_thk=False, std_paraval=np.array([]), issedani=True):
        """
        get parameter index arrays for para
        =============================================================================================================================
        ::: input :::
        perturb_thk     - perturb thickness or not
        std_paraval     - std of prior distribution of model parameters
        -----------------------------------------------------------------------------------------------------------------------------
        ::: output :::
        paraindex[0, :] - type of parameters
                            0   - vsh coefficient for splines
                            1   - vsv coefficient for splines
                            2   - thickness
                            3   - gamma = 2(Vsh-Vsv)/(Vsh+Vsv)
        paraindex[1, :] - index for type of amplitude for parameter perturbation
                            1   - absolute
                            -1  - relative
                            0   - fixed, do NOT perturb, added on 2019-03-19              
        paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
        paraindex[3, :] - step for parameter space 
        paraindex[4, :] - index for the parameter in the model group   
        paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0
        =============================================================================================================================
        """
        self.use_gamma  = True
        numbp_sum       = self.numbp.sum()
        npara           = numbp_sum  + self.nmod + 2
        # water layer
        if self.mtype[0] == 5:
            npara       -= 2
        self.para.init_arr(npara)
        use_prior       = True
        if std_paraval.size == 0:
            std_paraval = np.array([0.5, 0.5, 5., 5., 5., 5.,\
                                    5., 5., 5., 5., 5., 50., 10.])
            # std_paraval = np.array([0.5, 0.5, 10., 10., 10., 10.,\
            #                         20., 20., 20., 20., 20., 50., 10.])
            use_prior   = False
        ipara           = 0
        for i in xrange(self.nmod):
            # water layer
            if self.mtype[i] == 5:
                continue
            #--------------------------------
            for j in range(self.numbp[i]):
                self.para.paraindex[0, ipara]           = 1
                if i == 0 or (i == 1 and self.mtype[0] == 5): # water layer, added May 15, 2018
                    # sediment, cvel space is +- 1 km/s, different from Shen et al. 2012
                    self.para.paraindex[1, ipara]       = 1
                    self.para.paraindex[2, ipara]       = std_paraval[ipara]
                else:
                    if use_prior:
                        self.para.paraindex[1, ipara]   = 1
                        self.para.paraindex[2, ipara]   = std_paraval[ipara]
                    else:
                        # +- 5 % if no std of prior specified
                        self.para.paraindex[1, ipara]   = -1
                        self.para.paraindex[2, ipara]   = std_paraval[ipara]
                # 0.05 km/s 
                self.para.paraindex[3, ipara]           = 0.05
                self.para.paraindex[4, ipara]           = i
                self.para.paraindex[5, ipara]           = j
                ipara   +=1
        # sediment thickness
        if self.nmod >= 3:
            self.para.paraindex[0, ipara]               = 2
            if use_prior:
                self.para.paraindex[1, ipara]           = 1
                self.para.paraindex[2, ipara]           = std_paraval[ipara]
            else:
                self.para.paraindex[1, ipara]           = -1
                self.para.paraindex[2, ipara]           = std_paraval[ipara]
            if not perturb_thk:
                # do NOT perturb
                self.para.paraindex[1, ipara]           = 0 
            self.para.paraindex[3, ipara]               = 0.1
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]           = 1
            else:
                self.para.paraindex[4, ipara]           = 0
            ipara   += 1
        # crustal thickness/ +- 10 %
        self.para.paraindex[0, ipara]                   = 2
        if use_prior:
            self.para.paraindex[1, ipara]               = 1
            self.para.paraindex[2, ipara]               = std_paraval[ipara]
        else:
            self.para.paraindex[1, ipara]               = -1
            self.para.paraindex[2, ipara]               = std_paraval[ipara]
        if not perturb_thk:
            self.para.paraindex[1, ipara]               = 0 # perturb flag
        self.para.paraindex[3, ipara]       = 1.
        if self.nmod >= 3:
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 2.
            else:
                self.para.paraindex[4, ipara]   = 1.
        else:
            self.para.paraindex[4, ipara]       = 0.
        #-------------------------------
        # gamma value in the sediment
        #-------------------------------
        ipara   += 1
        self.para.paraindex[0, ipara]           = 3
        self.para.paraindex[1, ipara]           = 0 ##  perturb flag
        self.para.paraindex[2, ipara]           = 25. # +- 10 % in radial anisotropy
        self.para.paraindex[3, ipara]           = 1.
        if self.nmod >= 3:
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 1.
            else:
                self.para.paraindex[4, ipara]   = 0.
        else:
            self.para.paraindex[4, ipara]       = 0.
        #-------------------------------
        # gamma value in the crust
        #-------------------------------
        ipara   += 1
        self.para.paraindex[0, ipara]           = 3
        self.para.paraindex[1, ipara]           = 1 ##  perturb flag
        self.para.paraindex[2, ipara]           = 10. # +- 10 % in radial anisotropy
        self.para.paraindex[3, ipara]           = 1.
        if self.nmod >= 3:
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 2.
            else:
                self.para.paraindex[4, ipara]   = 1.
        else:
            self.para.paraindex[4, ipara]       = 0.
        #-------------------------------
        # gamma value in the mantle
        #-------------------------------
        ipara   += 1
        self.para.paraindex[0, ipara]           = 3
        self.para.paraindex[1, ipara]           = 1 #  perturb flag
        self.para.paraindex[2, ipara]           = 10.
        self.para.paraindex[3, ipara]           = 1.
        if self.nmod >= 3:
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 3.
            else:
                self.para.paraindex[4, ipara]   = 2.
        else:
            self.para.paraindex[4, ipara]       = 1.
        return

    def mod2para(self):
        """
        convert model to parameter arrays for perturbation
        """
        for i in range(self.para.npara):
            ig      = int(self.para.paraindex[4, i])
            ip      = int(self.para.paraindex[5, i])
            # vsh coefficient 
            if int(self.para.paraindex[0, i]) == 0:
                val = self.cvsh[ip , ig]
            # vsv coefficient 
            elif int(self.para.paraindex[0, i]) == 1:
                val = self.cvsv[ip , ig]
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == 2:
                val = self.thickness[ig]
            # gamma
            elif int(self.para.paraindex[0, i]) == 3:
                val = self.gamma[ig]
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == -1:
                val = self.vpvs[ig]
            else:
                print 'Unexpected value in paraindex!'
            self.para.paraval[i]        = val
            #-------------------------------------------
            # defining parameter space for perturbation
            #-------------------------------------------
            if not self.para.isspace:
                step        = self.para.paraindex[3, i]
                if int(self.para.paraindex[1, i]) == 1:
                    valmin  = val - self.para.paraindex[2, i]
                    valmax  = val + self.para.paraindex[2, i]
                else:
                    # bugfix on 2019/01/18
                    if val < 0.001 and int(self.para.paraindex[0, i]) == 2: # 0 value for sediment thickness
                        valmin  = 0. 
                        valmax  = 0.1
                    else:
                        valmin  = val - val*self.para.paraindex[2, i]/100.
                        valmax  = val + val*self.para.paraindex[2, i]/100.
                ## no negative sedimentary anisotropy
                if i == 13:
                    valmin  = val
                # radial anisotropy
                if int(self.para.paraindex[0, i]) != 3:
                    valmin      = max (0., valmin)
                    valmax      = max (valmin + 0.0001, valmax)
                if (int(self.para.paraindex[0, i]) == 1 and i == 0 \
                    and int(self.para.paraindex[5, i]) == 0): # if it is the upper sedi:
                    valmin              = max (0.2, valmin)
                    valmax              = max (0.5, valmax)
                self.para.space[0, i]   = valmin
                self.para.space[1, i]   = valmax
                self.para.space[2, i]   = step
        self.para.isspace               = True
        return
    
    def para2mod(self):
        """
        convert paratemers (for perturbation) to model parameters
        """
        for i in xrange(self.para.npara):
            val     = self.para.paraval[i]
            ig      = int(self.para.paraindex[4, i])
            ip      = int(self.para.paraindex[5, i])
            # vsh coefficient 
            if int(self.para.paraindex[0, i]) == 0:
                self.cvsh[ip , ig]  = val
            # vsv coefficient 
            elif int(self.para.paraindex[0, i]) == 1:
                self.cvsv[ip , ig]  = val
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == 2:
                self.thickness[ig]  = val
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == 3:
                self.gamma[ig]      = val
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == -1:
                self.vpvs[ig]       = val
            else:
                raise ValueError('Unexpected value in paraindex!')
        self.thickness[-1]          = 200. - self.thickness[:-1].sum()
        return
    
    def isgood(self, m0, m1, g0, g1, dvs_thresh=0.05):
        """
        check the model is good or not
        ==========================================================================
        ::: input   :::
        m0, m1  - index of group for monotonic change checking
        g0, g1  - index of group for gradient change checking
        ==========================================================================
        """
        # velocity constrast, contraint (5) in 4.2 of Shen et al., 2012
        for i in range (self.nmod-1):
            nlay        = self.nlay[i]
            if self.vsh[0, i+1] < self.vsh[nlay-1, i]:
                return False
            if self.vsv[0, i+1] < self.vsv[nlay-1, i]:
                return False
        # upper limit of anisotropy (20 %)
        for i in xrange(self.nmod):
            if i == 0: # do not check sediments
                continue
            temp_vsh    = self.vsh[:self.nlay[i], i]
            temp_vsv    = self.vsv[:self.nlay[i], i]
            if np.any(abs(temp_vsv-temp_vsh)/((temp_vsv+temp_vsh)/2.) > 0.2):
                return False
        # Vs < 4.9 km/sec , contraint (6) in 4.2 of Shen et al., 2012
        if np.any(self.vsv > 4.9) or np.any(self.vsh > 4.9):
            return False
        #
        if m1 >= self.nmod:
            m1  = self.nmod -1
        if m0 < 0:
            m0  = 0
        if g1 >= self.nmod:
            g1  = self.nmod -1
        if g0 < 0:
            g0  = 0
        # monotonic change
        # contraint (3) and (4) in 4.2 of Shen et al., 2012
        if m0 <= m1:
            for j in range(m0, m1+1):
                vs0     = self.vsh[:self.nlay[j]-1, j]
                vs1     = self.vsh[1:self.nlay[j], j]
                if np.any(np.greater(vs0, vs1)):
                    return False
                vs0     = self.vsv[:self.nlay[j]-1, j]
                vs1     = self.vsv[1:self.nlay[j], j]
                if np.any(np.greater(vs0, vs1)):
                    return False
        # gradient check, positive gradient
        # if g0<=g1:
        #     for j in range(g0, g1+1):
        #         if self.vs[0, j] > self.vs[1, j]:
        #             return False
        # gradient check, negative gradient
        # if g0<=g1:
        #     for j in range(g0, g1+1):
        #         if self.vs[0, j] < self.vs[1, j]:
        #             return False
        #--------------------------------------
        # new constraints added, Sep 7th, 2018
        #--------------------------------------
        # constrain the last layer Vs in crust
        nlay_crust      = self.nlay[self.nmod-2]
        if self.vsh[nlay_crust-1, self.nmod-2] > 4.3 or self.vsv[nlay_crust-1, self.nmod-2] > 4.3:
            return False
        # constrain the first layer Vs in mantle
        if self.vsh[0, self.nmod-1] > 4.6 or self.vsv[0, self.nmod-1] > 4.6:
            return False
        if self.vsv[0, self.nmod-1]< 4.0 or self.vsv[0, self.nmod-1]< 4.0:
            return False
        # constrain the bottom layer Vs in mantle
        nlay_mantle     = self.nlay[self.nmod-1]
        if self.vsh[nlay_mantle-1, self.nmod-1] < 4.3 or self.vsv[nlay_mantle-1, self.nmod-1] < 4.3:
            return False
        #-------------------------------------------------------------------
        # penalize oscillations with differences in local/maximum extrema 
        #-------------------------------------------------------------------
        dv_osci = 0.05
        hArr    = np.zeros(self.nlay.sum(), dtype = np.float64)
        vsh     = np.zeros(self.nlay.sum(), dtype = np.float64)
        vsv     = np.zeros(self.nlay.sum(), dtype = np.float64)
        for i in range(self.nmod):
            if i == 0:
                hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
            elif i < self.nmod - 1:
                hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
            else:
                hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
            if self.mtype[i] == 5 and i == 0:
                vsh[0]                  = 0.
                vsv[0]                  = 0.
            elif (i == 0 and self.mtype[i] != 5):
                vsh[:self.nlay[0]]      = self.vsh[:self.nlay[i], i]
                vsv[:self.nlay[0]]      = self.vsv[:self.nlay[i], i]
            elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2:
                vsh[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]
            elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2:
                vsh[self.nlay[:i].sum():]   = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():]   = self.vsv[:self.nlay[i], i]
            elif i < self.nmod - 1:
                vsh[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]
            # changed on 2019/01/17, Hacker & Abers, 2004
            else:
                vsh[self.nlay[:i].sum():]   = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():]   = self.vsv[:self.nlay[i], i]
        depth           = hArr.cumsum()
        #-----------------------
        # check Vsh
        #------------------------
        if not self.use_gamma:
            vs_mantle       = self.vsh[:self.nlay[self.nmod-1], self.nmod-1]
            local_indmax    = scipy.signal.argrelmax(vs_mantle)[0]
            local_indmin    = scipy.signal.argrelmin(vs_mantle)[0]
            if local_indmin.size > 0 and local_indmax.size > 0:
                if local_indmin.size == local_indmax.size:
                    vmin    = vs_mantle[local_indmin]
                    vmax    = vs_mantle[local_indmax]
                else:
                    Ndiff   = local_indmax.size - local_indmin.size
                    if Ndiff > 0:
                        vmin    = vs_mantle[local_indmin]
                        vmax    = vs_mantle[local_indmax[:-Ndiff]]
                    else:
                        vmin    = vs_mantle[local_indmin[:Ndiff]]
                        vmax    = vs_mantle[local_indmax]
                if (vmax-vmin).max() > dv_osci and (local_indmax.size + local_indmin.size) >= 3:
                    return False
            vs_trim         = vsh[depth > 60.]
            local_indmax    = scipy.signal.argrelmax(vs_trim)[0]
            local_indmin    = scipy.signal.argrelmin(vs_trim)[0]
            if local_indmax.size >= 1 and local_indmin.size >= 1:
                if (vs_trim[local_indmax].max() - vs_trim[local_indmin].min())>= dv_osci:
                    return False
        #-----------------------
        # check Vsv
        #------------------------
        vs_mantle       = self.vsv[:self.nlay[self.nmod-1], self.nmod-1]
        local_indmax    = scipy.signal.argrelmax(vs_mantle)[0]
        local_indmin    = scipy.signal.argrelmin(vs_mantle)[0]
        if local_indmin.size > 0 and local_indmax.size > 0:
            if local_indmin.size == local_indmax.size:
                vmin    = vs_mantle[local_indmin]
                vmax    = vs_mantle[local_indmax]
            else:
                Ndiff   = local_indmax.size - local_indmin.size
                if Ndiff > 0:
                    vmin    = vs_mantle[local_indmin]
                    vmax    = vs_mantle[local_indmax[:-Ndiff]]
                else:
                    vmin    = vs_mantle[local_indmin[:Ndiff]]
                    vmax    = vs_mantle[local_indmax]
            if (vmax-vmin).max() > dv_osci and (local_indmax.size + local_indmin.size) >= 3:
                return False
        vs_trim         = vsv[depth > 60.]
        local_indmax    = scipy.signal.argrelmax(vs_trim)[0]
        local_indmin    = scipy.signal.argrelmin(vs_trim)[0]
        if local_indmax.size >= 1 and local_indmin.size >= 1:
            if (vs_trim[local_indmax].max() - vs_trim[local_indmin].min())>= dv_osci:
                return False
        return True

    def get_vmodel(self, depth_mid_crt=-1., iulcrt=2):
        """
        get velocity models
        ==========================================================================
        ::: output :::
        hArr, vph, vpv, vsh, vsv, eta, rho, qs, qp
        ==========================================================================
        """
        nlay    = self.nlay.sum()
        hArr    = np.zeros(nlay, dtype = np.float64)
        vph     = np.zeros(nlay, dtype = np.float64)
        vpv     = np.zeros(nlay, dtype = np.float64)
        vsh     = np.zeros(nlay, dtype = np.float64)
        vsv     = np.zeros(nlay, dtype = np.float64)
        rho     = np.zeros(nlay, dtype = np.float64)
        eta     = np.ones(nlay, dtype = np.float64) # eta = 1
        qs      = np.zeros(nlay, dtype = np.float64)
        qp      = np.zeros(nlay, dtype = np.float64)
        depth   = np.zeros(nlay, dtype = np.float64)
        for i in range(self.nmod):
            if i == 0:
                hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
            elif i < self.nmod - 1:
                hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
            else:
                hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
            if self.mtype[i] == 5 and i == 0:
                vph[0]                  = self.cvph[0][0]
                vpv[0]                  = self.cvpv[0][0]
                vsh[0]                  = 0.
                vsv[0]                  = 0.
                rho[0]                  = 1.02
                qs[0]                   = 10000.
                qp[0]                   = 57822.
            elif (i == 0 and self.mtype[i] != 5):
                # Vph = Vpv, Xie et al., 2013
                vph[:self.nlay[0]]      = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vpv[:self.nlay[0]]      = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vsh[:self.nlay[0]]      = self.vsh[:self.nlay[i], i]
                vsv[:self.nlay[0]]      = self.vsv[:self.nlay[i], i]
                rho[:self.nlay[0]]      = 0.541 + 0.3601*self.vsv[:self.nlay[i], i]*self.vpvs[i]
                qs[:self.nlay[0]]       = 80.*np.ones(self.nlay[i], dtype=np.float64)
                qp[:self.nlay[0]]       = 160.*np.ones(self.nlay[i], dtype=np.float64)
            elif (i == 1 and self.mtype[0] == 5) and self.nmod > 2:
                # Vph = Vpv, Xie et al., 2013
                vph[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vpv[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vsh[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]
                rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = 0.541 + 0.3601*self.vsv[:self.nlay[i], i]*self.vpvs[i]
                qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 80.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 160.*np.ones(self.nlay[i], dtype=np.float64)
            elif (i == 1 and self.mtype[0] == 5) and self.nmod == 2:
                vph[self.nlay[:i].sum():]   = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vpv[self.nlay[:i].sum():]   = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vsh[self.nlay[:i].sum():]   = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():]   = self.vsv[:self.nlay[i], i]
                rho[self.nlay[:i].sum():]   = 0.541 + 0.3601*self.vsv[:self.nlay[i], i]*self.vpvs[i]
                qs[self.nlay[:i].sum():]    = 80.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():]    = 160.*np.ones(self.nlay[i], dtype=np.float64)
            elif i < self.nmod - 1:
                vph[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vpv[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vsh[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]
                rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = 0.541 + 0.3601*self.vsv[:self.nlay[i], i]*self.vpvs[i]
                qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 600.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 1400.*np.ones(self.nlay[i], dtype=np.float64)
            # changed on 2019/01/17, Hacker & Abers, 2004
            else:
                vph[self.nlay[:i].sum():]   = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vpv[self.nlay[:i].sum():]   = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vsh[self.nlay[:i].sum():]   = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():]   = self.vsv[:self.nlay[i], i]
                rho[self.nlay[:i].sum():]   = 3.4268 + (self.vsv[:self.nlay[i], i] - 4.5)/4.5 
                qs[self.nlay[:i].sum():]    = 150.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():]    = 1400.*np.ones(self.nlay[i], dtype=np.float64)
        depth               = hArr.cumsum()
        # reset upper/lower crust Vsh
        if depth_mid_crt > 0.:
            if self.mtype[0] == 5:
                icrt    = 2
            else:
                icrt    = 1
            tvsh        = vsh[self.nlay[:icrt].sum():self.nlay[:icrt+1].sum()].copy()
            tvsv        = vsv[self.nlay[:icrt].sum():self.nlay[:icrt+1].sum()].copy()
            tdepth      = depth[self.nlay[:icrt].sum():self.nlay[:icrt+1].sum()].copy()
            if iulcrt == 1:
                ind         = tdepth > depth_mid_crt
            else:
                ind         = tdepth <= depth_mid_crt
            tvsh[ind]   = tvsv[ind]
            vsh[self.nlay[:icrt].sum():self.nlay[:icrt+1].sum()]    = tvsh
        #
        return hArr, vph, vpv, vsh, vsv, eta, rho, qs, qp, nlay
    
    def new_paraval(self, ptype, m0=0, m1=1, g0=1, g1=0, dvs_thresh=0.05, Nthresh=10000, isconstrt=True):
        """
        perturb parameters in paraval array
        ===============================================================================
        ::: input :::
        ptype   - perturbation type
                    0   - uniform random value generated from parameter space
                    1   - Gauss random number generator given mu = oldval, sigma=step
        m0, m1  - index of group for monotonic change checking
        g0, g1  - index of group for gradient change checking
        ===============================================================================
        """
        if self.mtype[0] == 5:
            m0      += 1
            m1      += 1
            g0      += 1
            g1      += 1
        temp_mod    = copy.deepcopy(self)
        temp_mod.para.new_paraval(ptype)
        temp_mod.para2mod()
        temp_mod.update()
        if isconstrt:
            i_try       = 0
            while (not temp_mod.isgood(m0 = m0, m1 = m1, g0 = g0, g1= g1)) and i_try <= Nthresh:
                temp_mod    = copy.deepcopy(self)
                temp_mod.para.new_paraval(ptype)
                temp_mod.para2mod()
                temp_mod.update()
                i_try       += 1
            if i_try > Nthresh:
                return False
        self.para.paraval[:]    = temp_mod.para.paraval[:]
        self.para2mod()
        self.update()
        return True
    
    
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
        self.depth      = np.zeros(np.int64(self.nmod+1), dtype=np.float64)
        self.depth2d    = np.zeros((np.int64(self.nmod), np.int64(2) ), dtype=np.int32)
        self.layer_ind  = np.zeros(( np.int64(self.nmod), np.int64(2) ), dtype=np.int32)
        return
    
    
    def set_two_layer_crust(self, depth_mid_crust=15.):
        self.init_arr(3)
        self.depth[0]   = -1
        self.depth[1]   = depth_mid_crust
        self.depth[2]   = -2
        self.depth[3]   = -3
    
    def set_depth_disontinuity(self, depth_mid_crust=15., depth_mid_mantle=-1.):
        self.depth[0]   = -1
        i                   = 1
        if depth_mid_crust > 0.:
            self.depth[i]   = depth_mid_crust
            i               += 1
        self.depth[i]       = -2
        i                   += 1
        if depth_mid_mantle > 0.:
            self.depth[i]   = depth_mid_mantle
            i               += 1
        self.depth[i]       = -3
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
        
    def bspline(self, nlay, thickness, nBs=2, degBs=2):
        """
        Compute B-spline basis given group id
        The actual degree is k = degBs - 1
        e.g. nBs = 5, degBs = 4, k = 3, cubic B spline
        ::: output :::
        self.spl    - (nBs+k, npts)
                        [:nBs, :] B spline basis for nBs control points
                        [nBs:, :] can be ignored
        """
        zmin_Bs     = 0.
        zmax_Bs     = thickness
        disfacBs    = 2.
        npts        = nlay
        nbasis, t   = bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        m           = nBs-1+degBs
        
        # self.spl[:nBs, :npts, i]= nbasis[:nBs, :]
        # self.isspl[i]           = True
        # self.knot_vector[:(nBs+degBs), i]\
        #                         = t
        # self.Nknot[i]           = t.size
        self.spl        = nbasis
        self.knot_vector= t
        return 
    
    
    
    
    
    
    
    
    
    


    
