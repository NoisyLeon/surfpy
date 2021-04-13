# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.pymcinv._param_funcs as _param_funcs
from surfpy.pymcinv._param_funcs import VELOCITY, THICKNESS, GAMMA, VPVS, ABSOLUTE, RELATIVE, FIXED

import numpy as np
from scipy.optimize import lsq_linear
import scipy.interpolate
import scipy.signal
import copy

#====================
# global variables
#====================
# mtype
LAYER       = 1
BSPLINE     = 2
GRADIENT    = 4
WATER       = 5
# mtype_vti
NOANISO     = -1
LAYERGAMMA  = 0
GAMMASPLINE = 1
VSHSPLINE   = 2
# # # WATER       

class vtimod(object):
    """
    An object for handling parameterization of 1D Vertical TI model for the inversion
    =====================================================================================================================
    ::: parameters :::
    :   numbers     :
    nmod        - number of model groups
    maxlay      - maximum layers for each group (default - 100)
    maxspl      - maximum spline coefficients for each group (default - 20)
    ---------------------------------------------------------------------------------------------------------------------
    :   1D arrays, isotropic   :
    mtype       - model parameterization types (1D int array with length nmod)
                    1   - layer         - nlay  = numbp, hArr = ratio*thickness, vs = cvel
                    2   - B-splines     - hArr  = thickness/nlay, vs    = (cvel*spl)_sum over numbp
                    4   - gradient layer- nlay is defined depends on thickness
                                            hArr  = thickness/nlay, vs  = from cvel[0, i] to cvel[1, i]
                    5   - water         - nlay  = 1, vs = 0., hArr = thickness
    numbp       - number of control points/basis (1D int array with length nmod)
    thickness   - thickness of each group (1D float array with length nmod)
    nlay        - number of layers in each group (1D int array with length nmod)
    ---------------------------------------------------------------------------------------------------------------------
    :   1D arrays, VTI   :
    mtype_vti   - model parameterization types (1D int array with length nmod)
                    0   - layerized homogeneous gamma
                    1   - gamma spline
                    2   - vsh spline
                    5   - water         
    numbp_vti   - number of control points/basis (1D int array with length nmod)
                  OR number of model parameters  
    ---------------------------------------------------------------------------------------------------------------------
    :   multi-dim arrays, isotropic:
    t           - knot vectors for B splines (2D array - [:(self.numb[i]+degBs), i]; i indicating group id)
    spl         - B spline basis array (3D array - [:self.numb[i], :self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 2
    ratio       - array for the ratio of each layer (2D array - [:self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 1
    cvpv        - vpv velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    cvsv        - vsv velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    ---------------------------------------------------------------------------------------------------------------------
    :   multi-dim arrays, VTI:
    cvph        - vph velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    cvsh        - vsh velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
    cgamma      - gamma coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
                    defined as 2*(Vsh - Vsv) /(Vsh + Vsv)
    gammarange  - range of depth for gamma layer
                    self.gammarange[i][j][k]
                    i: index for nmod
                    j: index for gamma layer
                    k: 0: ztop 1: zbottom 2: nmod index (should be the same as i, for i checking purpose)
                    ONLY used for mtype_vti == 0
                    
    spl_vti     - B spline basis array (3D array - [:self.numb[i], :self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype_vti == 1 or 2
    ---------------------------------------------------------------------------------------------------------------------
    :   model arrays        :
    vph         - vph array         (2D array - [:self.nlay[i], i]; i indicating group id)
    vpv         - vpv array         (2D array - [:self.nlay[i], i]; i indicating group id)
    vsh         - vsh array         (2D array - [:self.nlay[i], i]; i indicating group id)
    vsv         - vsv array         (2D array - [:self.nlay[i], i]; i indicating group id)
    eta         - eta array         (2D array - [:self.nlay[i], i]; i indicating group id)
    rho         - rho array         (2D array - [:self.nlay[i], i]; i indicating group id)
    hArr        - layer arrays      (2D array - [:self.nlay[i], i]; i indicating group id)
    ---------------------------------------------------------------------------------------------------------------------
    :   para1d  :
    para        - object storing isotropic parameters for perturbation
    para_vti    - object storing VTI parameters for perturbation
    *********************************************************************************************************************
    initialization      : (1) self.parameterize_ak135(); (2) self.get_paraind(); (3) self.upate(); (4) self.get_vmodel();
                          (5) self.mod2para()
    paraval -> model    : (1) self.para2mod(); (2) self.update(); (3) self.get_vmodel()
    model   -> paraval  : (1) self.mod2para()
    =====================================================================================================================
    """
    
    def __init__(self):
        self.nmod           = 0
        self.maxlay         = 100
        self.maxspl         = 20
        self.para           = _param_funcs.para1d()
        self.para_vti       = _param_funcs.para1d()
        self.init_paraind   = False
        return
    
    def init_arr(self, nmod, vpvs_ratio = 1.75, nlay_per_group = 20):
        """
        initialization of arrays
        """
        self.nmod       = nmod
        # arrays of size nmod
        self.numbp      = np.zeros(np.int64(self.nmod), dtype=np.int64)
        self.mtype      = np.zeros(np.int64(self.nmod), dtype=np.int64)
        self.thickness  = np.zeros(np.int64(self.nmod), dtype=np.float64)
        self.nlay       = np.ones(np.int64(self.nmod),  dtype=np.int64) * nlay_per_group
        self.vpvs       = np.ones(np.int64(self.nmod),  dtype=np.float64) * vpvs_ratio
        self.isspl      = np.zeros(self.nmod, dtype=np.int64)
        # arrays of size maxspl, nmod
        self.cvpv       = np.zeros((self.maxspl,  self.nmod), dtype = np.float64)
        self.cvsv       = np.zeros((self.maxspl,  self.nmod), dtype = np.float64)
        # arrays of size maxlay, nmod
        self.ratio      = np.zeros((self.maxlay,  self.nmod), dtype = np.float64)
        self.vsv        = np.zeros((self.maxlay,  self.nmod), dtype = np.float64)
        self.hArr       = np.zeros((self.maxlay,  self.nmod), dtype = np.float64)
        # arrays of size maxspl, maxlay, nmod
        self.spl        = np.zeros((self.maxspl,  self.maxlay, self.nmod), dtype = np.float64)
        self.knot_vector= np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        self.Nknot      = np.zeros((self.nmod), dtype = np.int64)
        #====================================================
        # radial anisotropy: (Vsh - Vsv) / ((Vsh + Vsv)/2)
        #====================================================
        # 0: homogeneous gamma; 1: spline gamma; 2: spline vsh
        self.mtype_vti          = np.zeros(np.int64(self.nmod), dtype = np.int64)
        self.numbp_vti          = np.zeros(np.int64(self.nmod), dtype = np.int64)
        # other parameters
        # arrays of size maxlay, nmod
        self.vsh                = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
        # arrays of size maxspl, nmod
        self.cvsh               = np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        self.cvph               = np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        self.cgamma             = np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        # depth ranges for each gamma
        # 0: top of sediments; -1: bottom of sediments; -2: Moho; -3: bottom of model
        # self.gammarange[i][j][k]
        # i nmod; j layer index; k: 0 ztop 1 zbottom 2 nmod index
        self.gammarange         = []
        # splines for VTI 
        self.spl_vti            = np.zeros((self.maxspl, self.maxlay, self.nmod), dtype = np.float64)
        self.knot_vector_vti    = np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        self.Nknot_vti          = np.zeros((self.nmod), dtype = np.int64)
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
        #==============================
        # splines for isotropic part
        #==============================
        nBs         = self.numbp[i]
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness[i]
        disfacBs    = 2.
        npts        = self.nlay[i]
        nbasis, t   = _param_funcs.bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        m           = nBs - 1 + degBs
        if m > self.maxspl:
            raise ValueError('number of splines is too large, change default maxspl!')
        self.spl[:nBs, :npts, i]            = nbasis[:nBs, :]
        self.knot_vector[:(nBs+degBs), i]   = t
        self.Nknot[i]                       = t.size
        #==============================
        # splines for vti
        #==============================
        nBs         = self.numbp_vti[i]
        if nBs > 0 and (self.mtype_vti[i] == 1 or self.mtype_vti[i] == 2):
            if nBs < 4:
                degBs   = 3
            else:
                degBs   = 4
            nbasis_vti, t_vti   = _param_funcs.bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
            m                   = nBs - 1 + degBs
            if m > self.maxspl:
                raise ValueError('number of splines is too large, change default maxspl!')
            self.spl_vti[:nBs, :npts, i]            = nbasis_vti[:nBs, :]
            self.knot_vector_vti[:(nBs+degBs), i]   = t_vti
            self.Nknot_vti[i]                       = t_vti.size
        # end spline computing
        self.isspl[i]   = True
        return True

    def update(self):
        """
        Update model (velocities and hArr arrays), from the thickness, cvel
        """
        for i in range(self.nmod):
            if self.nlay[i] > self.maxlay:
                raise ValueError('number of layers is too large, need change default maxlay!')
            #====================================================
            # isotropic update
            #====================================================
            # layered model
            if self.mtype[i] == 1:
                self.nlay[i]                    = self.numbp[i]
                self.hArr[:, i]                 = self.ratio[:, i] * self.thickness[i]
                self.vsv[:self.nlay[i], i]      = self.cvsv[:self.nlay[i], i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i]   = False
                self.bspline(i)
                self.vsv[:self.nlay[i], i]      = np.dot( (self.spl[:self.numbp[i], :self.nlay[i], i]).T, self.cvsv[:self.numbp[i], i])
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
        #====================================================
        # radial anisotropy: (Vsh - Vsv) / ((Vsh + Vsv)/2)
        #====================================================
        nlay    = self.nlay.sum()
        hArr    = np.zeros(nlay, dtype = np.float64)
        for i in range(self.nmod):
            if i == 0:
                hArr[:self.nlay[0]]                             = self.hArr[:self.nlay[0], 0]
            elif i < self.nmod - 1:
                hArr[self.nlay[:i].sum():self.nlay[:i+1].sum()] = self.hArr[:self.nlay[i], i]
            else:
                hArr[self.nlay[:i].sum():]                      = self.hArr[:self.nlay[i], i]
        depth   = hArr.cumsum()
        for i in range(self.nmod):
            if self.mtype[i] == 5:
                continue
            #===========================
            # homogeneous gamma layers
            #===========================
            if self.mtype_vti[i] == 0:
                tmpzarr = depth[self.nlay[:i].sum():self.nlay[:i+1].sum()]
                tvsh    = (self.vsv[:self.nlay[i], i]).copy()
                ind0    = -1
                ind1    = -1
                tdepth1 = self.gammarange[i]
                for j in range(self.numbp_vti[i]):
                    tdepth2 = tdepth1[j]
                    if tdepth2[2] != i:
                        raise ValueError('Group index error in homogeneous gamma layer update: group: %d, index: %d' %(i, tdepth2[2]))
                    hv_ratio= (1. + self.cgamma[j, i]/200.)/(1 - self.cgamma[j, i]/200.) # vsh/vsv ratio
                    ztop    = tdepth2[0]
                    zbottom = tdepth2[1]
                    if ztop <= 0:
                        tind0   = 0
                    else:
                        tind0   = np.where(tmpzarr >= ztop)[0][0]
                    if zbottom <= 0:
                        tind1   = self.nlay[i]
                    else:
                        tind1   = np.where(tmpzarr >= zbottom)[0][0]
                    # print ('CHECK %d, %d, %d' %(i, tind0, tind1))
                    if tind0 < ind0 or tind1 < ind1 or tind1 < tind0:
                        raise ValueError('Index error in homogeneous gamma layer update: group: %d, index: %d/%d, %d/%d' %(j, tind0, tind1, ind0, ind1))
                    ind0                    = tind0
                    ind1                    = tind1
                    tvsh[ind0:ind1]         *= hv_ratio
                self.vsh[:self.nlay[i], i]  = tvsh[:]
            # gamma spline
            elif self.mtype_vti[i] == 1:
                tgamma                      = np.dot( (self.spl_vti[:self.numbp_vti[i], :self.nlay[i], i]).T, self.cgamma[:self.numbp_vti[i], i])
                hv_ratio                    = (1. + tgamma/200.)/(1 - tgamma/200.)
                self.vsh[:self.nlay[i], i]  = hv_ratio * self.vsv[:self.nlay[i], i]
            # Vsh spline
            elif self.mtype_vti[i] == 2:
                self.vsh[:self.nlay[i], i]  = np.dot( (self.spl_vti[:self.numbp_vti[i], :self.nlay[i], i]).T, self.cvsh[:self.numbp_vti[i], i])
            # no anisotropy
            elif self.mtype_vti[i] == -1:
                self.vsh[:self.nlay[i], i]  = self.vsv[:self.nlay[i], i]
            else:
                raise ValueError('Unrecognized model type: %d for vti') 
        return

    def update_depth(self):
        """
        update hArr arrays only, used for paramerization of a refernce input model
        """
        for i in range(self.nmod):
            if self.nlay[i] > self.maxlay:
                print('number of layers is too large, need change default maxlay!')
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
            # check VTI model type
            if (self.mtype_vti[i] == 1 or self.mtype_vti[i] == 2) and self.mtype[i] != 2:
                raise ValueError('vti model type CANNOT be b-splines, when isotropic part is NOT b-spline!')
        return True
    
    def parameterize_ak135(self, crtthk, sedthk, crt_depth = -1., mantle_depth = -1., \
            vti_numbp= [0, 1, 1], vti_modtype = [NOANISO, LAYERGAMMA, LAYERGAMMA], topovalue=1., maxdepth=200., vp_water=1.5):
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
            self.mtype[:]       = np.array([WATER, GRADIENT, BSPLINE, BSPLINE])
            self.vpvs[:]        = np.array([0., 2., 1.75, 1.75])
            # radial anisotropy
            self.numbp_vti[0]   = 0
            self.numbp_vti[1:]  = vti_numbp[:]
            self.mtype_vti[0]   = NOANISO
            self.mtype_vti[1:]  = vti_modtype[:]
        else:
            self.init_arr(3)
            self.thickness[:]   = np.array([sedthk, crtthk - sedthk, maxdepth - crtthk])
            self.numbp[:]       = np.array([2, 4, 5])
            self.mtype[:]       = np.array([GRADIENT, BSPLINE, BSPLINE])
            self.vpvs[:]        = np.array([2., 1.75, 1.75])
            # radial anisotropy
            self.numbp_vti[:]   = vti_numbp[:]
            self.mtype_vti[:]   = vti_modtype[:]
        self.update_depth()
        self.ngamma         = self.numbp_vti.sum()
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
        #========================
        # update gammarange array
        #========================
        for i in range(self.nmod):
            if self.mtype_vti[i] != LAYERGAMMA:
                self.gammarange.append([])
                continue
            # sediment
            if (i == 0 and self.nmod == 3) or (i == 1 and self.nmod == 4) :
                self.gammarange.append([[0, -1, i]])
            # crust
            elif (i == 1 and self.nmod == 3) or (i == 2 and self.nmod == 4) :
                if crt_depth < 0.:
                    self.gammarange.append([[-1, -2, i]])
                else:
                    self.gammarange.append([[-1, crt_depth, i], [crt_depth, -2, i]])
                    self.numbp_vti[i]   += 1
            # mantle
            elif i == (self.nmod-1):
                if mantle_depth < 0.:
                    self.gammarange.append([[-2, -3, i]])
                else:
                    self.gammarange.append([[-2, mantle_depth, i], [mantle_depth, -3, i]])
                    self.numbp_vti[i]   += 1
        self.maxdepth   = maxdepth
        return
    
    # def parameterize_ray(self, paraval, topovalue=1., maxdepth=200., vp_water=1.5):
    #     """
    #     use paramerization from vsv model inferred from Rayleigh wave inversion
    #     ===============================================================================
    #     ::: input :::
    #     crtthk      - input crustal thickness (unit - km)
    #     sedthk      - input sediment thickness (unit - km)
    #     maxdepth    - maximum depth for the 1-D profile (default - 200 km)
    #     ::: output :::
    #     self.thickness  
    #     self.numbp      - [2, 4, 5]
    #     self.mtype      - [4, 2, 2]
    #     self.vpvs       - [2., 1.75, 1.75]
    #     self.spl
    #     self.cvel       - determined from ak135
    #     ===============================================================================
    #     """
    #     if topovalue < 0.:
    #         self.init_arr(4)
    #         self.thickness[:]   = np.array([-topovalue, paraval[-2], paraval[-1],\
    #                                     maxdepth - paraval[-2] - paraval[-1] + topovalue])
    #         self.numbp[:]       = np.array([1, 2, 4, 5])
    #         self.mtype[:]       = np.array([5, 4, 2, 2])
    #         self.vpvs[:]        = np.array([0., 2., 1.75, 1.75])
    #     else:
    #         self.init_arr(3)
    #         self.thickness[:]   = np.array([paraval[-2], paraval[-1], maxdepth - paraval[-2] - paraval[-1]])
    #         self.numbp[:]       = np.array([2, 4, 5])
    #         self.mtype[:]       = np.array([4, 2, 2])
    #         self.vpvs[:]        = np.array([2., 1.75, 1.75])
    #     self.update_depth()
    #     # water layer
    #     if topovalue < 0.:
    #         self.cvph[0, 0] = vp_water
    #         self.cvpv[0, 0] = vp_water
    #     # sediments
    #     if topovalue >= 0.:
    #         self.cvsh[0, 0] = paraval[0]
    #         self.cvsv[0, 0] = paraval[0]
    #         self.cvsh[1, 0] = paraval[1]
    #         self.cvsv[1, 0] = paraval[1]
    #     else:
    #         self.cvsh[0, 1] = paraval[0]
    #         self.cvsv[0, 1] = paraval[0]
    #         self.cvsh[1, 1] = paraval[1]
    #         self.cvsv[1, 1] = paraval[1]
    #     # crust and mantle
    #     if topovalue >= 0.:
    #         self.cvsh[:4, 1]= paraval[2:6]
    #         self.cvsh[:5, 2]= paraval[6:11]
    #         self.cvsv[:4, 1]= paraval[2:6]
    #         self.cvsv[:5, 2]= paraval[6:11]
    #     else:
    #         self.cvsh[:4, 2]= paraval[2:6]
    #         self.cvsh[:5, 3]= paraval[6:11]
    #         self.cvsv[:4, 2]= paraval[2:6]
    #         self.cvsv[:5, 3]= paraval[6:11]
    #     return
    # 

    def get_paraind(self, perturb_thk=True, iso_std=np.array([]), sed_aniso=25., crt_aniso=10., man_aniso=10., crtthk = None, crtstd = 10.):
        """
        get parameter index arrays for para
        =============================================================================================================================
        ::: input :::
        perturb_thk     - perturb thickness or not
        std_paraval     - std of prior distribution of model parameters
        -----------------------------------------------------------------------------------------------------------------------------
        ::: output :::
        paraindex[0, :] - type of parameters
                            0   - velocity coefficient for splines
                            1   - thickness
                            2   - gamma = 2(Vsh-Vsv)/(Vsh+Vsv)
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
        numbp_sum       = self.numbp.sum()
        npara           = numbp_sum  + self.nmod - 1
        # water layer
        if self.mtype[0] == 5:
            npara       -= 2 # self.nmod = 4 (3), self.numbp = [1, 2, 4, 5] ([2, 4, 5])
        self.para.init_arr(npara)
        if iso_std.size == npara:
            use_prior   = True
        else:
            use_prior   = False
        ipara           = 0
        #===========================
        # isotropic vs
        #===========================
        for i in range(self.nmod):
            # water layer
            if self.mtype[i] == WATER:
                continue
            #--------------------------------
            for j in range(self.numbp[i]):
                self.para.paraindex[0, ipara]           = VELOCITY
                if i == 0 or (i == 1 and self.mtype[0] == 5): # sediment 
                    # sediment, cvel space is +- 1 km/s, different from Shen et al. 2012
                    self.para.paraindex[1, ipara]       = ABSOLUTE
                    if use_prior:
                        self.para.paraindex[2, ipara]   = iso_std[ipara]
                    else:
                        self.para.paraindex[2, ipara]   = 1.
                else:
                    if use_prior:
                        self.para.paraindex[1, ipara]   = ABSOLUTE
                        self.para.paraindex[2, ipara]   = iso_std[ipara]
                    else:
                        # +- 20 % if no std of prior specified
                        self.para.paraindex[1, ipara]   = RELATIVE
                        self.para.paraindex[2, ipara]   = 20.
                # 0.05 km/s 
                self.para.paraindex[3, ipara]           = 0.05
                self.para.paraindex[4, ipara]           = i
                self.para.paraindex[5, ipara]           = j
                ipara   +=1
        #===========================
        # sediment thickness
        #===========================
        if self.nmod >= 3:
            self.para.paraindex[0, ipara]       = THICKNESS
            if not perturb_thk:
                # do NOT perturb
                self.para.paraindex[1, ipara]   = FIXED
            else:
                self.para.paraindex[1, ipara]   = RELATIVE
            if use_prior:
                self.para.paraindex[2, ipara]   = iso_std[ipara]
            else:
                self.para.paraindex[2, ipara]   = 100.
            self.para.paraindex[3, ipara]       = 0.1
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 1
            else:
                self.para.paraindex[4, ipara]   = 0
            ipara   += 1
        #===========================
        # crustal thickness/ +- 50 %
        #===========================
        self.para.paraindex[0, ipara]           = THICKNESS
        if not perturb_thk:
            self.para.paraindex[1, ipara]       = FIXED # perturb flag
        else:
            self.para.paraindex[1, ipara]       = RELATIVE
        if use_prior:
            self.para.paraindex[2, ipara]       = iso_std[ipara]
        else:
            print ('!!! Crustal thickness range changed to 100 percentage')
            self.para.paraindex[2, ipara]   = 100. # crustal thickness/ +- 50 %
            #####
            # # # if crtthk is None:
            # # #     self.para.paraindex[2, ipara]   = 50. # crustal thickness/ +- 50 %
            # # # else:
            # # #     if crtstd > 0.5*crtthk:
            # # #         tmpstd  = min(crtstd, crtthk)
            # # #         self.para.paraindex[2, ipara]   = tmpstd/crtthk * 100.
            # # #         print ('!!! Crustal thickness range changed to %g percentage' %self.para.paraindex[2, ipara])
            # # #     else:
            # # #         self.para.paraindex[2, ipara]   = 50. # crustal thickness/ +- 50 %
        self.para.paraindex[3, ipara]           = 1.
        if self.nmod >= 3:
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 2.
            else:
                self.para.paraindex[4, ipara]   = 1.
        else:
            self.para.paraindex[4, ipara]       = 0.
        #===========================
        # radial anisotropy
        #===========================
        ipara           = 0
        self.para_vti.init_arr(self.numbp_vti.sum())
        for i in range(self.nmod):
            if self.mtype_vti[i] == NOANISO: # no anisotropy
                continue
            for j in range(self.numbp_vti[i]):
                if self.mtype_vti[i] == LAYERGAMMA or self.mtype_vti[i] == GAMMASPLINE: # gamma parameterization
                    self.para_vti.paraindex[0, ipara]       = GAMMA
                    self.para_vti.paraindex[1, ipara]       = ABSOLUTE
                    if i == 0 or (i == 1 and self.mtype[0] == WATER): # sediment 
                        self.para_vti.paraindex[2, ipara]   = sed_aniso # sediment radial anisotropy
                    elif i == 1 or (i == 2 and self.mtype[0] == WATER): # crust
                        self.para_vti.paraindex[2, ipara]   = crt_aniso # crustal radial anisotropy
                    elif i == 2 or (i == 3 and self.mtype[0] == WATER): # mantle
                        self.para_vti.paraindex[2, ipara]   = man_aniso # mantle radial anisotropy
                    self.para_vti.paraindex[3, ipara]       = 1.
                else: # vsh parameterization
                    self.para_vti.paraindex[0, ipara]       = VELOCITY
                    self.para_vti.paraindex[1, ipara]       = RELATIVE
                    if i == 0 or (i == 1 and self.mtype[0] == WATER): # sediment
                        self.para_vti.paraindex[1, ipara]   = ABSOLUTE
                        self.para_vti.paraindex[2, ipara]   = 1.
                    elif i == 1 or (i == 2 and self.mtype[0] == WATER): # crust
                        self.para_vti.paraindex[2, ipara]   = 20. # crust vsh += 20
                    elif i == 2 or (i == 3 and self.mtype[0] == WATER): # mantle
                        self.para_vti.paraindex[2, ipara]   = 20. # mantle vsh += 20
                    self.para_vti.paraindex[3, ipara]   = 0.05
                self.para_vti.paraindex[4, ipara]       = i
                self.para_vti.paraindex[5, ipara]       = j
                ipara   += 1
        if ipara != self.numbp_vti.sum():
            raise ValueError('incompatiable radial anisotropic parameters!')
        self.init_paraind   = True
        return

    def mod2para(self):
        """convert model to parameter arrays for perturbation
        """
        if not self.init_paraind:
            raise ValueError('parameter index array not initialized!')
        for i in range(self.para.npara):
            ig      = int(self.para.paraindex[4, i])
            # velocity coefficient 
            if int(self.para.paraindex[0, i]) == VELOCITY:
                ip  = int(self.para.paraindex[5, i])
                val = self.cvsv[ip][ig]
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == THICKNESS:
                val = self.thickness[ig]
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == VPVS:
                val = self.vpvs[ig]
            else:
                print ('Unexpected value in paraindex!')
            self.para.paraval[i] = val
            #-------------------------------------------
            # defining parameter space for perturbation
            #-------------------------------------------
            if not self.para.isspace:
                step        = self.para.paraindex[3, i]
                if int(self.para.paraindex[1, i]) == ABSOLUTE:
                    valmin  = val - self.para.paraindex[2, i]
                    valmax  = val + self.para.paraindex[2, i]
                else:
                    if val < 0.001 and int(self.para.paraindex[0, i]) == THICKNESS: # 0 value for sediment thickness
                        valmin  = 0. 
                        valmax  = 0.1
                    else:
                        valmin  = val - val*self.para.paraindex[2, i]/100.
                        valmax  = val + val*self.para.paraindex[2, i]/100.
                valmin          = max (0., valmin)
                valmax          = max (valmin + 0.0001, valmax)
                if (int(self.para.paraindex[0, i]) == VELOCITY and i == 0 \
                    and int(self.para.paraindex[5, i]) == 0): # if it is the upper sedi:
                    valmin              = max (0.2, valmin)
                    valmax              = max (0.5, valmax) 
                self.para.space[0, i]   = valmin
                self.para.space[1, i]   = valmax
                self.para.space[2, i]   = step
        self.para.isspace               = True
        #===========================
        # radial anisotropy
        #===========================
        for i in range(self.para_vti.npara):
            ig  = int(self.para_vti.paraindex[4, i])
            ip  = int(self.para_vti.paraindex[5, i])
            # velocity coefficient 
            if int(self.para_vti.paraindex[0, i]) == 0:
                val = self.cvsh[ip][ig]
            # total thickness of the group
            elif int(self.para_vti.paraindex[0, i]) == 2:
                val = self.cgamma[ip][ig]
            else:
                raise ValueError('Unexpected type of parameters %d' %int(self.para_vti.paraindex[0, i]))
            self.para_vti.paraval[i] = val
            #-------------------------------------------
            # defining parameter space for perturbation
            #-------------------------------------------
            if not self.para_vti.isspace:
                step    = self.para_vti.paraindex[3, i]
                if int(self.para_vti.paraindex[1, i]) == ABSOLUTE:
                    valmin  = val - self.para_vti.paraindex[2, i]
                    valmax  = val + self.para_vti.paraindex[2, i]
                else:
                    valmin  = val - val*self.para_vti.paraindex[2, i]/100.
                    valmax  = val + val*self.para_vti.paraindex[2, i]/100.
                if int(self.para_vti.paraindex[0, i]) == VELOCITY: # set lower limit for velocity 
                    valmin          = max (0., valmin)
                    valmax          = max (valmin + 0.0001, valmax)
                self.para_vti.space[0, i]   = valmin
                self.para_vti.space[1, i]   = valmax
                self.para_vti.space[2, i]   = step
        self.para_vti.isspace               = True
        return
    
    def para2mod(self):
        """convert paratemers (for perturbation) to model parameters
        """
        if not self.init_paraind:
            raise ValueError('parameter index array not initialized!')
        for i in range(self.para.npara):
            val = self.para.paraval[i]
            ig  = int(self.para.paraindex[4, i])
            # velocity coeficient for splines
            if int(self.para.paraindex[0, i]) == VELOCITY:
                ip                  = int(self.para.paraindex[5, i])
                self.cvsv[ip][ig]   = val
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == THICKNESS:
                self.thickness[ig]  = val
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == VPVS:
                self.vpvs[ig]       = val
            else:
                print('Unexpected value in paraindex!')
        # need to change 
        self.thickness[-1]          = self.maxdepth - self.thickness[:-1].sum()
        #===========================
        # radial anisotropy
        #===========================
        for i in range(self.para_vti.npara):
            val = self.para_vti.paraval[i]
            ig  = int(self.para_vti.paraindex[4, i])
            ip  = int(self.para_vti.paraindex[5, i])
            # velocity coeficient for splines
            if int(self.para_vti.paraindex[0, i]) == VELOCITY:
                self.cvsh[ip][ig]   = val
            # gamma
            elif int(self.para_vti.paraindex[0, i]) == GAMMA:
                self.cgamma[ip][ig] = val
            else:
                print('Unexpected value in paraindex!')
        return
    
    def isgood(self, m0, m1, g0, g1, dvs_thresh=0.01):
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
            # positive jump, 2021/04/09
            #
            if self.vsh[0, i+1] < self.vsv[nlay-1, i]:
                return False
            if self.vsv[0, i+1] < self.vsh[nlay-1, i]:
                return False
        # upper limit of anisotropy (20 %)
        for i in range(self.nmod):
            if i == 0: # do not check sediments
                continue
            temp_vsh    = self.vsh[:self.nlay[i], i]
            temp_vsv    = self.vsv[:self.nlay[i], i]
            if np.any(abs(temp_vsv-temp_vsh)/((temp_vsv+temp_vsh)/2.) > 0.2):
                return False
        # Vs < 4.9 km/sec , contraint (6) in 4.2 of Shen et al., 2012
        if np.any(self.vsv > 4.9) or np.any(self.vsh > 4.9):
            return False
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
        # # # dvs_thresh  = 0.01
        hArr        = np.zeros(self.nlay.sum(), dtype = np.float64)
        vsh         = np.zeros(self.nlay.sum(), dtype = np.float64)
        vsv         = np.zeros(self.nlay.sum(), dtype = np.float64)
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
            if (vmax-vmin).max() > dvs_thresh and (local_indmax.size + local_indmin.size) >= 3:
                return False
        vs_trim         = vsh[depth > 60.]
        local_indmax    = scipy.signal.argrelmax(vs_trim)[0]
        local_indmin    = scipy.signal.argrelmin(vs_trim)[0]
        if local_indmax.size >= 1 and local_indmin.size >= 1:
            if (vs_trim[local_indmax].max() - vs_trim[local_indmin].min())>= dvs_thresh:
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
            if (vmax-vmin).max() > dvs_thresh and (local_indmax.size + local_indmin.size) >= 3:
                return False
        vs_trim         = vsv[depth > 60.]
        local_indmax    = scipy.signal.argrelmax(vs_trim)[0]
        local_indmin    = scipy.signal.argrelmin(vs_trim)[0]
        if local_indmax.size >= 1 and local_indmin.size >= 1:
            if (vs_trim[local_indmax].max() - vs_trim[local_indmin].min())>= dvs_thresh:
                return False
        return True
    # 
    def get_vmodel(self):
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
            if self.mtype[i] == WATER and i == 0:
                vph[0]                  = self.cvph[0][0]
                vpv[0]                  = self.cvpv[0][0]
                vsh[0]                  = 0.
                vsv[0]                  = 0.
                rho[0]                  = 1.02
                qs[0]                   = 10000.
                qp[0]                   = 57822.
            elif (i == 0 and self.mtype[i] != WATER):
                # Vph = Vpv, Xie et al., 2013
                vph[:self.nlay[0]]      = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vpv[:self.nlay[0]]      = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vsh[:self.nlay[0]]      = self.vsh[:self.nlay[i], i]
                vsv[:self.nlay[0]]      = self.vsv[:self.nlay[i], i]
                rho[:self.nlay[0]]      = 0.541 + 0.3601*self.vsv[:self.nlay[i], i]*self.vpvs[i]
                qs[:self.nlay[0]]       = 80.*np.ones(self.nlay[i], dtype=np.float64)
                qp[:self.nlay[0]]       = 160.*np.ones(self.nlay[i], dtype=np.float64)
            elif (i == 1 and self.mtype[0] == WATER) and self.nmod > 2:
                # Vph = Vpv, Xie et al., 2013
                vph[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vpv[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]*self.vpvs[i]
                vsh[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsh[:self.nlay[i], i]
                vsv[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = self.vsv[:self.nlay[i], i]
                rho[self.nlay[:i].sum():self.nlay[:i+1].sum()]  = 0.541 + 0.3601*self.vsv[:self.nlay[i], i]*self.vpvs[i]
                qs[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 80.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():self.nlay[:i+1].sum()]   = 160.*np.ones(self.nlay[i], dtype=np.float64)
            elif (i == 1 and self.mtype[0] == WATER) and self.nmod == 2:
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
        return hArr, vph, vpv, vsh, vsv, eta, rho, qs, qp, nlay
    # 
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
        temp_mod.para_vti.new_paraval(ptype)
        temp_mod.para2mod()
        temp_mod.update()
        if isconstrt:
            i_try       = 0
            while (not temp_mod.isgood(m0 = m0, m1 = m1, g0 = g0, g1= g1)) and i_try <= Nthresh:
                temp_mod    = copy.deepcopy(self)
                temp_mod.para.new_paraval(ptype)
                temp_mod.para_vti.new_paraval(ptype)
                temp_mod.para2mod()
                temp_mod.update()
                i_try       += 1
            if i_try > Nthresh:
                return False
        self.para.paraval[:]    = temp_mod.para.paraval[:]
        self.para_vti.paraval[:]= temp_mod.para_vti.paraval[:]
        self.para2mod()
        self.update()
        return True

