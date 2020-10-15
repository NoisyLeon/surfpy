# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""

import surfpy.pymcinv._param_funcs as _param_funcs

import numpy as np
from scipy.optimize import lsq_linear
import scipy.interpolate
import scipy.signal
import copy


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
    knot_vector - B spline knot vector
    Nknot       - number of knots
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
        self.nmod           = 0
        self.maxlay         = 100
        self.maxspl         = 20
        self.para           = _param_funcs.para1d()
        self.init_paraind   = False
        self.maxdepth       = 200.
        return
    
    def copy(self):
        return copy.deepcopy(self)
    
    def init_arr(self, nmod, vpvs_ratio = 1.75, nlay_per_group = 20):
        """
        initialization of arrays
        """
        self.nmod       = nmod
        # arrays of size nmod
        self.numbp      = np.zeros(self.nmod,   dtype = np.int64)
        self.mtype      = np.zeros(self.nmod,   dtype = np.int64)
        self.thickness  = np.zeros(self.nmod,   dtype = np.float64)
        self.nlay       = np.ones(self.nmod,    dtype = np.int64)*nlay_per_group
        self.vpvs       = np.ones(self.nmod,    dtype = np.float64)*vpvs_ratio
        self.isspl      = np.zeros(self.nmod,   dtype = np.int64)
        # arrays of size maxspl, nmod
        self.cvel       = np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        # arrays of size maxlay, nmod
        self.ratio      = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
        self.vs         = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
        self.hArr       = np.zeros((self.maxlay, self.nmod), dtype = np.float64)
        # arrays of size maxspl, maxlay, nmod
        self.spl        = np.zeros((self.maxspl, self.maxlay, self.nmod), dtype = np.float64)
        self.knot_vector= np.zeros((self.maxspl, self.nmod), dtype = np.float64)
        self.Nknot      = np.zeros((self.nmod), dtype = np.int64)
        return
    
    def read(self, infname):
        """
        read model parameterization from a txt file
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
        print ("Number of model parameter groups: %d " % nmod)
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
                    print (tnp)
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
        """compute B-spline basis given group id
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
            print('*** spline basis already exists!')
            return False
        if self.mtype[i] != 2:
            print('*** Not spline parameterization!')
            return False
        # initialize
        if i >= self.nmod:
            raise ValueError('index for spline group out of range!')
        nBs         = self.numbp[i]
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness[i]
        disfacBs    = 2.
        npts        = self.nlay[i]
        # get B spline basis
        nbasis, t   = _param_funcs.bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        m           = nBs - 1 + degBs
        if m > self.maxspl:
            raise ValueError('number of splines is too large, change default maxspl!')
        #--------------------------
        # store the basis functions
        #--------------------------
        self.spl[:nBs, :npts, i]            = nbasis[:nBs, :]
        self.isspl[i]                       = True
        self.knot_vector[:(nBs+degBs), i]   = t
        self.Nknot[i]                       = t.size
        return True

    def update(self):
        """update model (vs and hArr arrays), from the thickness, cvel
        """
        for i in range(self.nmod):
            if self.nlay[i] > self.maxlay:
                print ('*** number of layers is too large, need change default maxlay!')
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
        """update hArr arrays only, used for paramerization of a refernce input model
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
            print ('WARNING: sort the input vs array to make it monotonically increases with depth in the crust')
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
        self.maxdepth   = maxdepth
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
        self.maxdepth   = maxdepth
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
        self.maxdepth   = maxdepth
        return

    def get_paraind(self):
        """get parameter index arrays for para
        Table 1 and 2 in Shen et al. 2012
        references:
        Shen, W., Ritzwoller, M.H., Schulte-Pelkum, V. and Lin, F.C., 2012.
            Joint inversion of surface wave dispersion and receiver functions: a Bayesian Monte-Carlo approach.
                Geophysical Journal International, 192(2), pp.807-836.
        """
        numbp_sum   = self.numbp.sum()
        npara       = numbp_sum  + self.nmod - 1
        # water layer
        if self.mtype[0] == 5:
            npara   -= 2
        self.para.init_arr(npara)
        ipara       = 0
        for i in range(self.nmod):
            # water layer
            if self.mtype[i] == 5:
                continue
            #--------------------------------
            for j in range(self.numbp[i]):
                self.para.paraindex[0, ipara]   = 0
                if i == 0 or (i == 1 and self.mtype[0] == 5): # water layer, 
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
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 1
            else:
                self.para.paraindex[4, ipara]   = 0
            ipara                           += 1
        # crustal thickness/ +- 20 %
        self.para.paraindex[0, ipara]       = 1
        self.para.paraindex[1, ipara]       = -1
        # # self.para.paraindex[2, ipara]       = 20.
        self.para.paraindex[2, ipara]       = 50. # crustal thickness/ +- 50 %
        self.para.paraindex[3, ipara]       = 1.
        if self.nmod >= 3:
            if self.mtype[0] == 5: # water layer
                self.para.paraindex[4, ipara]   = 2.
            else:
                self.para.paraindex[4, ipara]   = 1.
        else:
            self.para.paraindex[4, ipara]   = 0.
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
                print ('Unexpected value in paraindex!')
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
        """convert paratemers (for perturbation) to model parameters
        """
        if not self.init_paraind:
            raise ValueError('parameter index array not initialized!')
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
        # need to change 
        self.thickness[-1]          = self.maxdepth - self.thickness[:-1].sum()
        return
    
    def isgood(self, m0, m1, g0, g1, dvs_thresh = 0.05):
        """check the model is good or not
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
        # Vs < 4.9 km/sec , contraint (6) in 4.2 of Shen et al., 2012
        if np.any(self.vs > 4.9):
            return False
        # define the index for monotonic/gradient check
        if m1 >= self.nmod:
            m1      = self.nmod -1
        if m0 < 0:
            m0      = 0
        if g1 >= self.nmod:
            g1      = self.nmod -1
        if g0 < 0:
            g0      = 0
        #----------------------------------------------------
        # (1-a) monotonic change
        # contraint (3) and (4) in 4.2 of Shen et al., 2012
        if m0 <= m1:
            for j in range(m0, m1+1):
                vs0     = self.vs[:self.nlay[j]-1, j]
                vs1     = self.vs[1:self.nlay[j], j]
                if np.any(np.greater(vs0, vs1)):
                    return False
        # (1-b) gradient check, positive gradient
        # may not be implemented
    
        # if g0<=g1:
        #     for j in range(g0, g1+1):
        #         if self.vs[0, j] > self.vs[1, j]:
        #             return False
        # gradient check, negative gradient
        # if g0<=g1:
        #     for j in range(g0, g1+1):
        #         if self.vs[0, j] < self.vs[1, j]:
        #             return False
        #----------------------------------------------------
        # (2-a) constrain the last layer Vs in crust
        nlay_crust      = self.nlay[self.nmod-2]
        if self.vs[nlay_crust-1, self.nmod-2] > 4.3:
            return False
        # (2-b) constrain the first layer Vs in mantle
        if self.vs[0, self.nmod-1] > 4.6:
            return False
        if self.vs[0, self.nmod-1] < 4.0:
            return False
        # (2-c) constrain the bottom layer Vs in mantle
        nlay_mantle     = self.nlay[self.nmod-1]
        
        if self.vs[nlay_mantle-1, self.nmod-1] < 4.3:
            return False
        
        #--------------------------------------
        # curvature constraints in the mantle, may NOT be implemented
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
        # (3) penalize oscillations with differences in local/maximum extrema 
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
        return True
    
    def get_vmodel(self):
        """get velocity models
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
            # changed on 2019/01/17, Hacker & Abers, 2004
            else:
                vs[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]
                vp[self.nlay[:i].sum():]    = self.vs[:self.nlay[i], i]*self.vpvs[i]
                rho[self.nlay[:i].sum():]   = 3.4268 + (self.vs[:self.nlay[i], i] - 4.5)/4.5 
                qs[self.nlay[:i].sum():]    = 150.*np.ones(self.nlay[i], dtype=np.float64)
                qp[self.nlay[:i].sum():]    = 1400.*np.ones(self.nlay[i], dtype=np.float64)
        depth               = hArr.cumsum()
        return hArr, vs, vp, rho, qs, qp, nlay

