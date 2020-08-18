# -*- coding: utf-8 -*-
"""
Module for inversion of 1d models

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import os
import vmodel, modparam, data, eigenkernel
import copy
import fast_surf, theo, tdisp96, tregn96, tlegn96
import multiprocessing
from functools import partial
import time
import random
from uncertainties import unumpy

class vprofile1d(object):
    """
    An object for 1D velocity profile inversion
    =====================================================================================================================
    ::: parameters :::
    data                - object storing input data
    model               - object storing 1D model
    eigkR, eigkL        - eigenkernel objects storing Rayleigh/Love eigenfunctions and sensitivity kernels
    disprefR, disprefL  - flags indicating existence of sensitivity kernels for reference model
    =====================================================================================================================
    """
    def __init__(self):
        self.model      = vmodel.model1d()
        self.data       = data.data1d()
        self.eigkR      = eigenkernel.eigkernel()
        self.eigkL      = eigenkernel.eigkernel()
        self.ref_hArr   = None
        self.disprefR   = False
        self.disprefL   = False
        self.fs         = 40.
        self.slowness   = 0.06
        self.gausswidth = 2.5
        self.amplevel   = 0.005
        self.t0         = 0.
        self.code       = ''
        return
    
    def readdisp(self, infname, dtype='ph', wtype='ray'):
        """
        read dispersion curve data from a txt file
        ===========================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (phase or group)
        wtype       - wave type (Rayleigh or Love)
        ===========================================================
        """
        dtype   = dtype.lower()
        wtype   = wtype.lower()
        if wtype=='ray' or wtype=='rayleigh' or wtype=='r':
            self.data.dispR.readdisptxt(infname=infname, dtype=dtype)
            if self.data.dispR.npper>0:
                self.data.dispR.pvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
                self.data.dispR.gvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
#            if self.data.dispR.ngper>0:
#                self.data.dispR.gvelp = np.zeros(self.data.dispR.ngper, dtype=np.float64)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            self.data.dispL.readdisptxt(infname=infname, dtype=dtype)
            if self.data.dispL.npper>0:
                self.data.dispL.pvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
                self.data.dispL.gvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
#            if self.data.dispL.ngper>0:
#                self.data.dispL.gvelp = np.zeros(self.data.dispL.ngper, dtype=np.float64)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return
    
    def get_disp(self, indata, dtype='ph', wtype='ray'):
        """
        read dispersion curve data from a txt file
        ===========================================================
        ::: input :::
        indata      - input array (3, N)
        dtype       - data type (phase or group)
        wtype       - wave type (Rayleigh or Love)
        ===========================================================
        """
        dtype   = dtype.lower()
        wtype   = wtype.lower()
        if wtype=='ray' or wtype=='rayleigh' or wtype=='r':
            self.data.dispR.get_disp(indata=indata, dtype=dtype)
            if self.data.dispR.npper>0:
                self.data.dispR.pvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
                self.data.dispR.gvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            self.data.dispL.get_disp(indata=indata, dtype=dtype)
            if self.data.dispL.npper>0:
                self.data.dispL.pvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
                self.data.dispL.gvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return
    
    def get_azi_disp(self, indata, wtype='ray'):
        """
        read dispersion curve data from a txt file
        ===========================================================
        ::: input :::
        indata      - input array (7, N)
        wtype       - wave type (Rayleigh or Love)
        ===========================================================
        """
        wtype   = wtype.lower()
        if wtype=='ray' or wtype=='rayleigh' or wtype=='r':
            self.data.dispR.get_azi_disp(indata=indata)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            self.data.dispL.get_azi_disp(indata=indata)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return

    def readrf(self, infname, dtype='r'):
        """
        read receiver function data from a txt file
        ===========================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (radial or trnasverse)
        ===========================================================
        """
        dtype=dtype.lower()
        if dtype=='r' or dtype == 'radial':
            self.data.rfr.readrftxt(infname)
            self.data.rfr.tp    = np.linspace(self.data.rfr.to[0], self.data.rfr.to[-1], \
                        self.data.rfr.npts, dtype=np.float64)
            self.data.rfr.rfp   = np.zeros(self.data.rfr.npts, dtype=np.float64)
            self.npts           = self.data.rfr.npts
            self.fs             = 1./(self.data.rfr.to[1] - self.data.rfr.to[0])
        elif dtype=='t' or dtype == 'transverse':
            self.data.rft.readrftxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+dtype)
        return
    
    def get_rf(self, indata, dtype='r'):
        """
        read receiver function data from a txt file
        ===========================================================
        ::: input :::
        indata      - input data array (3, N)
        dtype       - data type (radial or transverse)
        ===========================================================
        """
        dtype   = dtype.lower()
        if dtype=='r' or dtype == 'radial':
            self.data.rfr.get_rf(indata=indata)
            self.data.rfr.tp    = np.linspace(self.data.rfr.to[0], self.data.rfr.to[-1], \
                        self.data.rfr.npts, dtype=np.float64)
            self.data.rfr.rfp   = np.zeros(self.data.rfr.npts, dtype=np.float64)
            self.npts           = self.data.rfr.npts
            self.fs             = 1./(self.data.rfr.to[1] - self.data.rfr.to[0])
        # # elif dtype=='t' or dtype == 'transverse':
        # #     self.data.rft.readrftxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+dtype)
        return
    
    def readmod(self, infname, mtype='iso'):
        """
        read model from a txt file
        ===========================================================
        ::: input :::
        infname     - input file name
        mtype       - model type (isotropic or tti)
        ===========================================================
        """
        mtype   = mtype.lower()
        if mtype == 'iso' or mtype == 'isotropic':
            self.model.isomod.readmodtxt(infname)
        # elif mtype == 'tti':
        #     self.model.ttimod.readttimodtxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def readpara(self, infname, mtype='iso'):
        """
        read parameter index indicating model parameters for perturbation
        =====================================================================
        ::: input :::
        infname     - input file name
        mtype       - model type (isotropic or tti)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.isomod.para.readparatxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def getpara(self, mtype='iso'):
        """
        get parameter index indicating model parameters for perturbation
        =====================================================================
        ::: input :::
        mtype       - model type (isotropic or Love)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.isomod.get_paraind()
        elif mtype == 'vti':
            self.model.vti.get_paraind_gamma()
#        elif mtype=='tti':
#            self.model.ttimod.get_paraind()
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def update_mod(self, mtype='iso'):
        """
        update model from model parameters
        =====================================================================
        ::: input :::
        mtype       - model type (0 - isotropic or 1 - tti)
        =====================================================================
        """
        if mtype == 'iso' or mtype == 'isotropic':
            self.model.isomod.update()
        elif mtype=='vti':
            self.model.vtimod.update()
        else:
            raise ValueError('Unexpected wave type: '+ mtype)
        return 
    
    def get_vmodel(self, mtype='iso', depth_mid_crt=-1., iulcrt=2):
        """
        get the velocity model arrays
        =====================================================================
        ::: input :::
        mtype       - model type (0 - isotropic or 1 - tti)
        =====================================================================
        """
        if mtype == 'iso' or mtype == 'isotropic':
            self.model.get_iso_vmodel()
        elif mtype=='vti':
            self.model.get_vti_vmodel(depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
        else:
            raise ValueError('Unexpected wave type: '+ mtype)
        return 
    
    #==========================================
    # forward modelling for surface waves
    #==========================================
    
    def get_period(self):
        """
        get period array for forward modelling
        """
        if self.data.dispR.npper>0:
            self.TRp        = self.data.dispR.pper.copy()
        if self.data.dispR.ngper>0:
            self.TRg        = self.data.dispR.gper.copy()
        # added 11/05/2018
        if self.data.dispR.npper>0 and self.data.dispR.ngper>0:
            if not np.allclose(self.TRp[:self.data.dispR.ngper], self.TRg):
                raise ValueError('incompatible phase/group periods!')
        if self.data.dispL.npper>0:
            self.TLp        = self.data.dispL.pper.copy()
        if self.data.dispL.ngper>0:
            self.TLg        = self.data.dispL.gper.copy()
        # added 11/05/2018
        if self.data.dispL.npper>0 and self.data.dispL.ngper>0:
            if not np.allclose(self.TLp[:self.data.dispL.ngper], self.TLg):
                raise ValueError('incompatible phase/group periods!')
        return
    #-------------------------------------
    # forward solver for isotropic model
    #-------------------------------------

    def compute_fsurf(self, wtype='ray'):
        """
        compute surface wave dispersion of isotropic model using fast_surf
        =====================================================================
        ::: input :::
        wtype       - wave type (Rayleigh or Love)
        =====================================================================
        """
        wtype   = wtype.lower()
        if self.model.nlay == 0:
            raise ValueError('No layerized model stored!')
        if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
            ilvry                   = 2
            nper                    = self.TRp.size
            per                     = np.zeros(200, dtype=np.float64)
            per[:nper]              = self.TRp[:]
            qsinv                   = 1./self.model.qs
            (ur0,ul0,cr0,cl0)       = fast_surf.fast_surf(self.model.nlay, ilvry, \
                                        self.model.vpv, self.model.vsv, self.model.rho, self.model.h, qsinv, per, nper)
            self.data.dispR.pvelp   = cr0[:nper]
            # modified 11/05/2018
            self.data.dispR.gvelp   = ur0[:self.data.dispR.ngper]
            # replace NaN value with oberved value
            # added Aug 30th, 2018
            index_nan               = np.isnan(self.data.dispR.gvelp)
            if np.any(index_nan) and self.data.dispR.ngper > 0:
                self.data.dispR.gvelp[index_nan]\
                                    = self.data.dispR.gvelo[index_nan]
        elif wtype=='l' or wtype == 'love' or wtype=='lov':
            ilvry                   = 1
            nper                    = self.TLp.size
            per                     = np.zeros(200, dtype=np.float64)
            per[:nper]              = self.TLp[:]
            qsinv                   = 1./self.model.qs
            (ur0,ul0,cr0,cl0)       = fast_surf.fast_surf(self.model.nlay, ilvry, \
                                        self.model.vph, self.model.vsh, self.model.rho, self.model.h, qsinv, per, nper)
            self.data.dispL.pvelp   = cl0[:nper]
            self.data.dispL.gvelp   = ul0[:self.data.dispL.ngper]
        return
    
    
    
    #-------------------------------------
    # forward solver for VTI model
    #-------------------------------------
    def compute_reference_vti(self, wtype='ray', verbose=0, nmodes=1, cmin=-1., cmax=-1., egn96=True, checkdisp=True, tol=10.):
        """
        compute (reference) surface wave dispersion of Vertical TI model using tcps
        ====================================================================================
        ::: input :::
        wtype       - wave type (Rayleigh or Love)
        nmodes      - number of modes
        cmin, cmax  - minimum/maximum value for phase velocity root searching
        egn96       - computing eigenfunctions/kernels or not
        checkdisp   - check the reasonability of dispersion curves with fast_surf
        tol         - tolerence of maximum differences between tcps and fast_surf
        ====================================================================================
        """
        wtype           = wtype.lower()
        self.ref_hArr   = self.model.h.copy()
        if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
            nfval       = self.TRp.size
            freq        = 1./ self.TRp
            nl_in       = self.model.h.size
            ilvry       = 2
            iflsph_in   = 1 # 1 - spherical Earth, 0 - flat Earth
            # initialize eigenkernel for Rayleigh wave
            self.eigkR.init_arr(nfreq=nfval, nlay=nl_in, ilvry=ilvry)
            # solve for phase velocity
            c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out = tdisp96.disprs(ilvry, 1., nfval, 1, verbose, nfval, \
                    np.append(freq, np.zeros(2049-nfval)), cmin, cmax, \
                    self.model.h, self.model.A, self.model.C, self.model.F, self.model.L, self.model.N, self.model.rho,
                    nl_in, iflsph_in, 0., nmodes,  1., 1.)  ### used for VTI inversion 
            # # # c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out = tdisp96.disprs(ilvry, 1., nfval, 1, verbose, nfval, \
            # # #         np.append(freq, np.zeros(2049-nfval)), cmin, cmax, \
            # # #         self.model.h, self.model.A, self.model.C, self.model.F, self.model.L, self.model.N, self.model.rho,
            # # #         nl_in, iflsph_in, 0., nmodes,  .5, .5) # used for HTI inversion
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # store reference model and ET model
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.eigkR.get_ref_model(A = self.model.A, C = self.model.C, F = self.model.F,\
                                    L = self.model.L, N = self.model.N, rho = self.model.rho)
            self.eigkR.get_ref_model_vel(ah = self.model.vph, av = self.model.vpv, bh = self.model.vsh,\
                                    bv = self.model.vsv, n = self.model.eta, r = self.model.rho)
            self.eigkR.get_ETI(Aeti = self.model.A, Ceti = self.model.C, Feti = self.model.F,\
                                Leti = self.model.L, Neti = self.model.N, rhoeti = self.model.rho)
            self.eigkR.get_ETI_vel(aheti = self.model.vph, aveti = self.model.vpv, bheti = self.model.vsh,\
                                    bveti = self.model.vsv, neti = self.model.eta, reti = self.model.rho)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # store the reference dispersion curve
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.data.dispR.pvelref   = np.float32(c_out[:nfval])
            self.data.dispR.pvelp     = np.float32(c_out[:nfval])
            #- compute eigenfunction/kernels
            if egn96:
                hs_in       = 0.
                hr_in       = 0.
                ohr_in      = 0.
                ohs_in      = 0.
                refdep_in   = 0.
                dogam       = True # turn on attenuation
                k           = 2.*np.pi/c_out[:nfval]/self.TRp
                k2d         = np.tile(k, (nl_in, 1))
                k2d         = k2d.T
                omega       = 2.*np.pi/self.TRp
                omega2d     = np.tile(omega, (nl_in, 1))
                omega2d     = omega2d.T
                # use spherical transformed model parameters
                d_in        = d_out
                TA_in       = TA_out
                TC_in       = TC_out
                TF_in       = TF_out
                TL_in       = TL_out
                TN_in       = TN_out
                TRho_in     = TRho_out
                # original model paramters should be used
                # d_in        = self.model.h
                # TA_in       = self.model.A
                # TC_in       = self.model.C
                # TF_in       = self.model.F
                # TL_in       = self.model.L
                # TN_in       = self.model.N
                # TRho_in     = self.model.rho
                
                qai_in      = self.model.qp
                qbi_in      = self.model.qs
                etapi_in    = np.zeros(nl_in)
                etasi_in    = np.zeros(nl_in)
                frefpi_in   = np.ones(nl_in)
                frefsi_in   = np.ones(nl_in)
                # solve for group velocity, kernels and eigenfunctions
                u_out, ur, tur, uz, tuz, dcdh, dcdav, dcdah, dcdbv, dcdbh, dcdn, dcdr = tregn96.tregn96(hs_in, hr_in, ohr_in, ohs_in,\
                    refdep_in, dogam, nl_in, iflsph_in, d_in, TA_in, TC_in, TF_in, TL_in, TN_in, TRho_in, \
                    qai_in, qbi_in, etapi_in, etasi_in, frefpi_in, frefsi_in, self.TRp.size, self.TRp, c_out[:nfval])
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # store output
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.data.dispR.gvelp   = np.float32(u_out)[:self.data.dispR.ngper]
                # eigenfunctions
                self.eigkR.get_eigen_psv(uz = uz[:nfval,:nl_in], tuz = tuz[:nfval,:nl_in],\
                                         ur = ur[:nfval,:nl_in], tur = tur[:nfval,:nl_in])
                # sensitivity kernels for velocity parameters and density
                # dcdah, dcdav, dcdbh, dcdbv, dcdn, dcdr
                self.eigkR.get_vkernel_psv(dcdah = dcdah[:nfval,:nl_in], dcdav = dcdav[:nfval,:nl_in], dcdbh = dcdbh[:nfval,:nl_in],\
                        dcdbv = dcdbv[:nfval,:nl_in], dcdn = dcdn[:nfval,:nl_in], dcdr = dcdr[:nfval,:nl_in])
                # Love parameters and density in the shape of nfval, nl_in
                self.eigkR.compute_love_kernels()
                self.disprefR   = True
        elif wtype=='l' or wtype == 'love' or wtype == 'lov':
            nfval       = self.TLp.size
            freq        = 1./self.TLp
            nl_in       = self.model.h.size
            ilvry       = 1
            self.eigkL.init_arr(nfreq=nfval, nlay=nl_in, ilvry=ilvry)
            #- root-finding algorithm using tdisp96, compute phase velocities 
            iflsph_in   = 1 # 1 - spherical Earth, 0 - flat Earth
            # solve for phase velocity
            c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out = tdisp96.disprs(ilvry, 1., nfval, 1, verbose, nfval, \
                np.append(freq, np.zeros(2049-nfval)), cmin, cmax, \
                self.model.h, self.model.A, self.model.C, self.model.F, self.model.L, self.model.N, self.model.rho, nl_in,\
                iflsph_in, 0., nmodes,  1., 1.)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # store reference model and ET model
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.eigkL.get_ref_model(A = self.model.A, C = self.model.C, F = self.model.F,\
                                    L = self.model.L, N = self.model.N, rho = self.model.rho)
            self.eigkL.get_ref_model_vel(ah = self.model.vph, av = self.model.vpv, bh = self.model.vsh,\
                                    bv = self.model.vsv, n = self.model.eta, r = self.model.rho)
            self.eigkL.get_ETI(Aeti = self.model.A, Ceti = self.model.C, Feti = self.model.F,\
                                Leti = self.model.L, Neti = self.model.N, rhoeti = self.model.rho)
            self.eigkL.get_ETI_vel(aheti = self.model.vph, aveti = self.model.vpv, bheti = self.model.vsh,\
                                    bveti = self.model.vsv, neti = self.model.eta, reti = self.model.rho)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # store the reference dispersion curve
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.data.dispL.pvelref = np.float32(c_out[:nfval])
            self.data.dispL.pvelp   = np.float32(c_out[:nfval])
            if egn96:
                hs_in       = 0.
                hr_in       = 0.
                ohr_in      = 0.
                ohs_in      = 0.
                refdep_in   = 0.
                dogam       = True # turn on attenuation
                nl_in       = self.model.h.size
                k           = 2.*np.pi/c_out[:nfval]/self.TLp
                k2d         = np.tile(k, (nl_in, 1))
                k2d         = k2d.T
                omega       = 2.*np.pi/self.TLp
                omega2d     = np.tile(omega, (nl_in, 1))
                omega2d     = omega2d.T
                # use spherical transformed model parameters
                d_in        = d_out
                TA_in       = TA_out
                TC_in       = TC_out
                TF_in       = TF_out
                TL_in       = TL_out
                TN_in       = TN_out
                TRho_in     = TRho_out
                # original model paramters should be used
                # d_in        = self.model.h
                # TA_in       = self.model.A
                # TC_in       = self.model.C
                # TF_in       = self.model.F
                # TL_in       = self.model.L
                # TN_in       = self.model.N
                # TRho_in     = self.model.rho
                
                qai_in      = self.model.qp
                qbi_in      = self.model.qs
                etapi_in    = np.zeros(nl_in)
                etasi_in    = np.zeros(nl_in)
                frefpi_in   = np.ones(nl_in)
                frefsi_in   = np.ones(nl_in)
                # solve for group velocity, kernels and eigenfunctions
                u_out, ut, tut, dcdh, dcdav, dcdah, dcdbv, dcdbh, dcdn, dcdr = tlegn96.tlegn96(hs_in, hr_in, ohr_in, ohs_in,\
                    refdep_in, dogam, nl_in, iflsph_in, d_in, TA_in, TC_in, TF_in, TL_in, TN_in, TRho_in, \
                    qai_in,qbi_in,etapi_in,etasi_in, frefpi_in, frefsi_in, self.TLp.size, self.TLp, c_out[:nfval])
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # store output
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.data.dispL.gvelp       = np.float32(u_out)[:self.data.dispL.ngper]
                # eigenfunctions
                self.eigkL.get_eigen_sh(ut = ut[:nfval,:nl_in], tut = tut[:nfval,:nl_in] )
                # sensitivity kernels for velocity parameters and density
                self.eigkL.get_vkernel_sh(dcdbh = dcdbh[:nfval,:nl_in], dcdbv = dcdbv[:nfval,:nl_in], dcdr = dcdr[:nfval,:nl_in])
                # Love parameters and density in the shape of nfval, nl_in
                self.eigkL.compute_love_kernels()
                self.disprefL   = True
        #----------------------------------------
        # check the consistency with fast_surf
        #----------------------------------------
        if checkdisp:
            hArr        = d_out
            vsv         = np.sqrt(TL_out/TRho_out)
            vpv         = np.sqrt(TC_out/TRho_out)
            vsh         = np.sqrt(TN_out/TRho_out)
            vph         = np.sqrt(TA_out/TRho_out)
            rho         = TRho_out
            qsinv       = 1./self.model.qs
            if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
                ilvry               = 2
                nper                = self.TRp.size
                per                 = np.zeros(200, dtype=np.float32)
                per[:nper]          = self.TRp[:]
                (ur0,ul0,cr0,cl0)   = fast_surf.fast_surf(vsv.size, ilvry, \
                                        vpv, vsv, rho, hArr, qsinv, per, nper)
                pvelp               = cr0[:nper]
                gvelp               = ur0[:nper]
                if (abs(pvelp - self.data.dispR.pvelref)/pvelp*100.).max() > tol:
                    # print('WARNING: reference dispersion curves may be erroneous!')
                    return False
            elif wtype=='l' or wtype == 'love' or wtype=='lov':
                ilvry               = 1
                nper                = self.TLp.size
                per                 = np.zeros(200, dtype=np.float32)
                per[:nper]          = self.TLp[:]
                (ur0,ul0,cr0,cl0)   = fast_surf.fast_surf(vsh.size, ilvry, \
                                       vph, vsh, rho, hArr, qsinv, per, nper)
                pvelp               = cl0[:nper]
                gvelp               = ul0[:nper]
                if (abs(pvelp - self.data.dispL.pvelref)/pvelp*100.).max() > tol:
                    # print('WARNING: reference dispersion curves may be erroneous!')
                    return False
        return True

    def compute_reference_vti_2(self, wtype='ray', verbose=0, nmodes=1, cmin=-1., cmax=-1., egn96=True, checkdisp=True, tol=10.):
        """
        compute (reference) surface wave dispersion of Vertical TI model using tcps
        ====================================================================================
        ::: input :::
        wtype       - wave type (Rayleigh or Love)
        nmodes      - number of modes
        cmin, cmax  - minimum/maximum value for phase velocity root searching
        egn96       - computing eigenfunctions/kernels or not
        checkdisp   - check the reasonability of dispersion curves with fast_surf
        tol         - tolerence of maximum differences between tcps and fast_surf
        ====================================================================================
        """
        wtype           = wtype.lower()
        self.ref_hArr   = self.model.h.copy()
        if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
            nfval       = self.TRp.size
            freq        = 1./ self.TRp
            nl_in       = self.model.h.size
            ilvry       = 2
            iflsph_in   = 1 # 1 - spherical Earth, 0 - flat Earth
            # initialize eigenkernel for Rayleigh wave
            self.eigkR.init_arr(nfreq=nfval, nlay=nl_in, ilvry=ilvry)
            # solve for phase velocity
            c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out = tdisp96.disprs(ilvry, 1., nfval, 1, verbose, nfval, \
                    np.append(freq, np.zeros(2049-nfval)), cmin, cmax, \
                    self.model.h, self.model.A, self.model.C, self.model.F, self.model.L, self.model.N, self.model.rho,
                    nl_in, iflsph_in, 0., nmodes,  1., 1.)  ### used for VTI inversion 
            # # # c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out = tdisp96.disprs(ilvry, 1., nfval, 1, verbose, nfval, \
            # # #         np.append(freq, np.zeros(2049-nfval)), cmin, cmax, \
            # # #         self.model.h, self.model.A, self.model.C, self.model.F, self.model.L, self.model.N, self.model.rho,
            # # #         nl_in, iflsph_in, 0., nmodes,  .5, .5) # used for HTI inversion
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # store reference model and ET model
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.eigkR.get_ref_model(A = self.model.A, C = self.model.C, F = self.model.F,\
                                    L = self.model.L, N = self.model.N, rho = self.model.rho)
            self.eigkR.get_ref_model_vel(ah = self.model.vph, av = self.model.vpv, bh = self.model.vsh,\
                                    bv = self.model.vsv, n = self.model.eta, r = self.model.rho)
            self.eigkR.get_ETI(Aeti = self.model.A, Ceti = self.model.C, Feti = self.model.F,\
                                Leti = self.model.L, Neti = self.model.N, rhoeti = self.model.rho)
            self.eigkR.get_ETI_vel(aheti = self.model.vph, aveti = self.model.vpv, bheti = self.model.vsh,\
                                    bveti = self.model.vsv, neti = self.model.eta, reti = self.model.rho)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # store the reference dispersion curve
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.data.dispR.pvelref   = np.float32(c_out[:nfval])
            self.data.dispR.pvelp     = np.float32(c_out[:nfval])
            #- compute eigenfunction/kernels
            if egn96:
                hs_in       = 0.
                hr_in       = 0.
                ohr_in      = 0.
                ohs_in      = 0.
                refdep_in   = 0.
                dogam       = True # turn on attenuation
                k           = 2.*np.pi/c_out[:nfval]/self.TRp
                k2d         = np.tile(k, (nl_in, 1))
                k2d         = k2d.T
                omega       = 2.*np.pi/self.TRp
                omega2d     = np.tile(omega, (nl_in, 1))
                omega2d     = omega2d.T
                # use spherical transformed model parameters
                # d_in        = d_out
                # TA_in       = TA_out
                # TC_in       = TC_out
                # TF_in       = TF_out
                # TL_in       = TL_out
                # TN_in       = TN_out
                # TRho_in     = TRho_out
                # original model paramters should be used
                d_in        = self.model.h
                TA_in       = self.model.A
                TC_in       = self.model.C
                TF_in       = self.model.F
                TL_in       = self.model.L
                TN_in       = self.model.N
                TRho_in     = self.model.rho
                
                qai_in      = self.model.qp
                qbi_in      = self.model.qs
                etapi_in    = np.zeros(nl_in)
                etasi_in    = np.zeros(nl_in)
                frefpi_in   = np.ones(nl_in)
                frefsi_in   = np.ones(nl_in)
                # solve for group velocity, kernels and eigenfunctions
                u_out, ur, tur, uz, tuz, dcdh, dcdav, dcdah, dcdbv, dcdbh, dcdn, dcdr = tregn96.tregn96(hs_in, hr_in, ohr_in, ohs_in,\
                    refdep_in, dogam, nl_in, iflsph_in, d_in, TA_in, TC_in, TF_in, TL_in, TN_in, TRho_in, \
                    qai_in, qbi_in, etapi_in, etasi_in, frefpi_in, frefsi_in, self.TRp.size, self.TRp, c_out[:nfval])
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # store output
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.data.dispR.gvelp   = np.float32(u_out)[:self.data.dispR.ngper]
                # eigenfunctions
                self.eigkR.get_eigen_psv(uz = uz[:nfval,:nl_in], tuz = tuz[:nfval,:nl_in],\
                                         ur = ur[:nfval,:nl_in], tur = tur[:nfval,:nl_in])
                # sensitivity kernels for velocity parameters and density
                # dcdah, dcdav, dcdbh, dcdbv, dcdn, dcdr
                self.eigkR.get_vkernel_psv(dcdah = dcdah[:nfval,:nl_in], dcdav = dcdav[:nfval,:nl_in], dcdbh = dcdbh[:nfval,:nl_in],\
                        dcdbv = dcdbv[:nfval,:nl_in], dcdn = dcdn[:nfval,:nl_in], dcdr = dcdr[:nfval,:nl_in])
                # Love parameters and density in the shape of nfval, nl_in
                self.eigkR.compute_love_kernels()
                self.disprefR   = True
        elif wtype=='l' or wtype == 'love' or wtype == 'lov':
            nfval       = self.TLp.size
            freq        = 1./self.TLp
            nl_in       = self.model.h.size
            ilvry       = 1
            self.eigkL.init_arr(nfreq=nfval, nlay=nl_in, ilvry=ilvry)
            #- root-finding algorithm using tdisp96, compute phase velocities 
            iflsph_in   = 1 # 1 - spherical Earth, 0 - flat Earth
            # solve for phase velocity
            c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out = tdisp96.disprs(ilvry, 1., nfval, 1, verbose, nfval, \
                np.append(freq, np.zeros(2049-nfval)), cmin, cmax, \
                self.model.h, self.model.A, self.model.C, self.model.F, self.model.L, self.model.N, self.model.rho, nl_in,\
                iflsph_in, 0., nmodes,  1., 1.)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # store reference model and ET model
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.eigkL.get_ref_model(A = self.model.A, C = self.model.C, F = self.model.F,\
                                    L = self.model.L, N = self.model.N, rho = self.model.rho)
            self.eigkL.get_ref_model_vel(ah = self.model.vph, av = self.model.vpv, bh = self.model.vsh,\
                                    bv = self.model.vsv, n = self.model.eta, r = self.model.rho)
            self.eigkL.get_ETI(Aeti = self.model.A, Ceti = self.model.C, Feti = self.model.F,\
                                Leti = self.model.L, Neti = self.model.N, rhoeti = self.model.rho)
            self.eigkL.get_ETI_vel(aheti = self.model.vph, aveti = self.model.vpv, bheti = self.model.vsh,\
                                    bveti = self.model.vsv, neti = self.model.eta, reti = self.model.rho)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # store the reference dispersion curve
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.data.dispL.pvelref = np.float32(c_out[:nfval])
            self.data.dispL.pvelp   = np.float32(c_out[:nfval])
            if egn96:
                hs_in       = 0.
                hr_in       = 0.
                ohr_in      = 0.
                ohs_in      = 0.
                refdep_in   = 0.
                dogam       = True # turn on attenuation
                nl_in       = self.model.h.size
                k           = 2.*np.pi/c_out[:nfval]/self.TLp
                k2d         = np.tile(k, (nl_in, 1))
                k2d         = k2d.T
                omega       = 2.*np.pi/self.TLp
                omega2d     = np.tile(omega, (nl_in, 1))
                omega2d     = omega2d.T
                # use spherical transformed model parameters
                # d_in        = d_out
                # TA_in       = TA_out
                # TC_in       = TC_out
                # TF_in       = TF_out
                # TL_in       = TL_out
                # TN_in       = TN_out
                # TRho_in     = TRho_out
                # original model paramters should be used
                d_in        = self.model.h
                TA_in       = self.model.A
                TC_in       = self.model.C
                TF_in       = self.model.F
                TL_in       = self.model.L
                TN_in       = self.model.N
                TRho_in     = self.model.rho
                
                qai_in      = self.model.qp
                qbi_in      = self.model.qs
                etapi_in    = np.zeros(nl_in)
                etasi_in    = np.zeros(nl_in)
                frefpi_in   = np.ones(nl_in)
                frefsi_in   = np.ones(nl_in)
                # solve for group velocity, kernels and eigenfunctions
                u_out, ut, tut, dcdh, dcdav, dcdah, dcdbv, dcdbh, dcdn, dcdr = tlegn96.tlegn96(hs_in, hr_in, ohr_in, ohs_in,\
                    refdep_in, dogam, nl_in, iflsph_in, d_in, TA_in, TC_in, TF_in, TL_in, TN_in, TRho_in, \
                    qai_in,qbi_in,etapi_in,etasi_in, frefpi_in, frefsi_in, self.TLp.size, self.TLp, c_out[:nfval])
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # store output
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.data.dispL.gvelp       = np.float32(u_out)[:self.data.dispL.ngper]
                # eigenfunctions
                self.eigkL.get_eigen_sh(ut = ut[:nfval,:nl_in], tut = tut[:nfval,:nl_in] )
                # sensitivity kernels for velocity parameters and density
                self.eigkL.get_vkernel_sh(dcdbh = dcdbh[:nfval,:nl_in], dcdbv = dcdbv[:nfval,:nl_in], dcdr = dcdr[:nfval,:nl_in])
                # Love parameters and density in the shape of nfval, nl_in
                self.eigkL.compute_love_kernels()
                self.disprefL   = True
        #----------------------------------------
        # check the consistency with fast_surf
        #----------------------------------------
        if checkdisp:
            hArr        = d_out
            vsv         = np.sqrt(TL_out/TRho_out)
            vpv         = np.sqrt(TC_out/TRho_out)
            vsh         = np.sqrt(TN_out/TRho_out)
            vph         = np.sqrt(TA_out/TRho_out)
            rho         = TRho_out
            qsinv       = 1./self.model.qs
            if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
                ilvry               = 2
                nper                = self.TRp.size
                per                 = np.zeros(200, dtype=np.float32)
                per[:nper]          = self.TRp[:]
                (ur0,ul0,cr0,cl0)   = fast_surf.fast_surf(vsv.size, ilvry, \
                                        vpv, vsv, rho, hArr, qsinv, per, nper)
                pvelp               = cr0[:nper]
                gvelp               = ur0[:nper]
                if (abs(pvelp - self.data.dispR.pvelref)/pvelp*100.).max() > tol:
                    # print('WARNING: reference dispersion curves may be erroneous!')
                    return False
            elif wtype=='l' or wtype == 'love' or wtype=='lov':
                ilvry               = 1
                nper                = self.TLp.size
                per                 = np.zeros(200, dtype=np.float32)
                per[:nper]          = self.TLp[:]
                (ur0,ul0,cr0,cl0)   = fast_surf.fast_surf(vsh.size, ilvry, \
                                       vph, vsh, rho, hArr, qsinv, per, nper)
                pvelp               = cl0[:nper]
                gvelp               = ul0[:nper]
                if (abs(pvelp - self.data.dispL.pvelref)/pvelp*100.).max() > tol:
                    # print('WARNING: reference dispersion curves may be erroneous!')
                    return False
        return True
    
    def perturb_from_kernel_vti(self, wtype='ray', ivellove=1):
        """
        compute perturbation in dispersion from reference model using sensitivity kernels
        ====================================================================================
        ::: input :::
        wtype       - wave type (Rayleigh or Love)
        ivellove    - use velocity kernels or Love parameter kernels
                        1   - velocity kernels
                        2   - Love kernels
        ====================================================================================
        """
        wtype   = wtype.lower()
        nl_in       = self.model.h.size
        if nl_in == 0:
            raise ValueError('No layer arrays stored!')
        if not np.allclose(self.ref_hArr, self.model.h):
            raise ValueError('layer array changed!')
        # Rayleigh wave
        if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
            if not self.disprefR:
                raise ValueError('referennce dispersion and kernels for Rayleigh wave not computed!')
            self.eigkR.get_ETI_vel(aheti = self.model.vph, aveti = self.model.vpv, bheti = self.model.vsh,\
                                    bveti = self.model.vsv, neti = self.model.eta, reti = self.model.rho)
            self.eigkR.get_ETI(Aeti = self.model.A, Ceti = self.model.C, Feti = self.model.F,\
                                    Leti = self.model.L, Neti = self.model.N, rhoeti = self.model.rho)
            if ivellove == 1:
                dpvel                   = self.eigkR.eti_perturb_vel()
            else:         
                dpvel                   = self.eigkR.eti_perturb()
            self.data.dispR.pvelp       = self.data.dispR.pvelref + dpvel
        # Love wave
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            if not self.disprefL:
                raise ValueError('referennce dispersion and kernels for Love wave not computed!')
            self.eigkL.get_ETI_vel(aheti = self.model.vph, aveti = self.model.vpv, bheti = self.model.vsh,\
                                    bveti = self.model.vsv, neti = self.model.eta, reti = self.model.rho)
            self.eigkL.get_ETI(Aeti = self.model.A, Ceti = self.model.C, Feti = self.model.F,\
                                    Leti = self.model.L, Neti = self.model.N, rhoeti = self.model.rho)
            if ivellove == 1:
                dpvel                   = self.eigkL.eti_perturb_vel()
            else:
                dpvel                   = self.eigkL.eti_perturb()
            self.data.dispL.pvelp       = self.data.dispL.pvelref + dpvel
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def compute_disp_vti(self, wtype='both', solver_type=0, \
            verbose=0, nmodes=1, crmin=-1., crmax=-1., clmin=-1., clmax=-1., egn96=True, checkdisp=True, tol=10.):
        """
        compute surface wave dispersion of Vertical TI model 
        ====================================================================================
        ::: input :::
        wtype       - wave type (Rayleigh or Love)
        solver_type - type of forward solver
                        0       - fast_surf
                        1       - direct computation of tcps
                        others  - use kernels from tcps
        nmodes      - number of modes
        crmin, crmax- minimum/maximum value for Rayleigh wave phase velocity root searching
        clmin, clmax- minimum/maximum value for Love wave phase velocity root searching
        egn96       - computing eigenfunctions/kernels or not
        checkdisp   - check the reasonability of dispersion curves with fast_surf
        tol         - tolerence of maximum differences between tcps and fast_surf
        ====================================================================================
        """
        wtype   = wtype.lower()
        if solver_type == 0:
            if wtype == 'both':
                self.compute_fsurf(wtype = 'ray')
                self.compute_fsurf(wtype = 'lov')
            else:
                self.compute_fsurf(wtype = wtype)
            return True
        elif solver_type == 1:
            if (crmin <= 0. or crmax <= 0.)and wtype != 'lov':
                temp_vpr        = copy.deepcopy(self)
                temp_vpr.compute_fsurf(wtype = 'ray')
                crmin           = temp_vpr.data.dispR.pvelp.min() - 0.1
                crmax           = temp_vpr.data.dispR.pvelp.max() + 0.1
            if (clmin <= 0. or clmax <= 0.) and wtype != 'ray':
                temp_vpr        = copy.deepcopy(self)
                temp_vpr.compute_fsurf(wtype = 'lov')
                clmin           = temp_vpr.data.dispL.pvelp.min() - 0.1
                clmax           = temp_vpr.data.dispL.pvelp.max() + 0.1
            if wtype == 'both':
                # Rayleigh wave
                valid_ray       = self.compute_reference_vti(wtype='ray', verbose=verbose, nmodes=nmodes,\
                                        cmin=crmin, cmax=crmax, egn96=egn96, checkdisp=checkdisp, tol=tol)
                if not valid_ray:
                    valid_ray   = self.data.dispR.check_pdisp(dtype='ph', Tthresh = 50., mono_tol  = 0.001, dv_tol=0.2)
                # Love wave
                valid_lov       = self.compute_reference_vti(wtype='lov', verbose=verbose, nmodes=nmodes,\
                                        cmin=clmin, cmax=clmax, egn96=egn96, checkdisp=checkdisp, tol=tol)
                if not valid_lov:
                    valid_lov   = self.data.dispR.check_pdisp(dtype='ph', Tthresh = 50., mono_tol  = 0.001, dv_tol=0.2)
                return bool(valid_ray*valid_lov)
            else:
                if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
                    valid       = self.compute_reference_vti(wtype=wtype, verbose=verbose, nmodes=nmodes,\
                                        cmin=crmin, cmax=crmax, egn96=egn96, checkdisp=checkdisp, tol=tol)
                else:
                    valid       = self.compute_reference_vti(wtype=wtype, verbose=verbose, nmodes=nmodes,\
                                        cmin=clmin, cmax=clmax, egn96=egn96, checkdisp=checkdisp, tol=tol)
                if not valid:
                    if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
                        valid   = self.data.dispR.check_pdisp(dtype='ph', Tthresh = 50., mono_tol  = 0.001, dv_tol=0.2)
                    else:
                        valid   = self.data.dispL.check_pdisp(dtype='ph', Tthresh = 50., mono_tol  = 0.001, dv_tol=0.2)
                return valid
        else:
            if wtype == 'both':   
                if not (self.disprefL and self.disprefR):
                    raise ValueError('reference dispersion curves and initialzed!')
                self.perturb_from_kernel_vti(wtype='ray')
                self.perturb_from_kernel_vti(wtype='lov')
            else:
                self.perturb_from_kernel_vti(wtype=wtype)
            return True
    
    #-------------------------------------
    # solver for HTI model
    #-------------------------------------
    def get_reference_hti(self, pvelref, dcdA, dcdC, dcdF, dcdL):
        """
        get (reference) surface wave dispersion of Horizontal TI model 
        ====================================================================================
        ::: input :::

        ====================================================================================
        """
        self.ref_hArr   = self.model.h.copy()
        nfval           = self.TRp.size
        freq            = 1./ self.TRp
        nl_in           = self.model.h.size
        ilvry           = 2
        iflsph_in       = 1 # 1 - spherical Earth, 0 - flat Earth
        # initialize eigenkernel for Rayleigh wave
        self.eigkR.init_arr(nfreq=nfval, nlay=nl_in, ilvry=ilvry)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # store reference model and ET model
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.eigkR.get_ref_model(A = self.model.A, C = self.model.C, F = self.model.F,\
                                L = self.model.L, N = self.model.N, rho = self.model.rho)
        self.eigkR.get_ref_model_vel(ah = self.model.vph, av = self.model.vpv, bh = self.model.vsh,\
                                bv = self.model.vsv, n = self.model.eta, r = self.model.rho)
        self.eigkR.get_ETI(Aeti = self.model.A, Ceti = self.model.C, Feti = self.model.F,\
                            Leti = self.model.L, Neti = self.model.N, rhoeti = self.model.rho)
        self.eigkR.get_ETI_vel(aheti = self.model.vph, aveti = self.model.vpv, bheti = self.model.vsh,\
                                bveti = self.model.vsv, neti = self.model.eta, reti = self.model.rho)
        self.data.dispR.pvelref = pvelref
        self.eigkR.dcdA[:, :]   = dcdA
        self.eigkR.dcdC[:, :]   = dcdC
        self.eigkR.dcdF[:, :]   = dcdF
        self.eigkR.dcdL[:, :]   = dcdL
        return True
    #==========================================
    # forward modelling for receiver function
    #==========================================
    
    def compute_rftheo(self, slowness = 0.06, din=None, npts=None):
        """
        compute receiver function of isotropic model using theo
        =============================================================================================
        ::: input :::
        slowness- reference horizontal slowness (default - 0.06 s/km, 1./0.06=16.6667)
        din     - incident angle in degree      (default - None, din will be computed from slowness)
        =============================================================================================
        """
        if self.data.rfr.npts == 0:
            raise ValueError('npts of receiver function is 0!')
            return
        if self.model.isomod.mtype[0] == 5:
            raise ValueError('receiver function cannot be computed in water!')
        # initialize input model arrays
        hin         = np.zeros(100, dtype=np.float64)
        vsin        = np.zeros(100, dtype=np.float64)
        vpvs        = np.zeros(100, dtype=np.float64)
        qsin        = 600.*np.ones(100, dtype=np.float64)
        qpin        = 1400.*np.ones(100, dtype=np.float64)
        # assign model arrays to the input arrays
        if self.model.nlay<100:
            nl      = self.model.nlay
        else:
            nl      = 100
        hin[:nl]    = self.model.h[:nl]
        vsin[:nl]   = self.model.vsv[:nl]
        vpvs[:nl]   = self.model.vpv[:nl]/self.model.vsv[:nl]
        qsin[:nl]   = self.model.qs[:nl]
        qpin[:nl]   = self.model.qp[:nl]
        # fs/npts
        fs          = self.fs
        # # # ntimes      = 1000
        if npts is None:
            ntimes  = self.data.rfr.npts
        else:
            ntimes  = npts
        # incident angle
        if din is None:
            din     = 180.*np.arcsin(vsin[nl-1]*vpvs[nl-1]*slowness)/np.pi
        # solve for receiver function using theo
        rx 	        = theo.theo(nl, vsin, hin, vpvs, qpin, qsin, fs, din, 2.5, 0.005, 0, ntimes)
        # store the predicted receiver function (ONLY radial component) to the data object
        self.data.rfr.rfp   = rx[:self.data.rfr.npts]
        self.data.rfr.tp    = np.arange(self.data.rfr.npts, dtype=np.float64)*1./self.fs
        return
    #==========================================
    # computing misfit
    #==========================================
    def get_misfit(self, mtype='iso', wdisp=1., rffactor=40.):
        """
        compute data misfit
        =====================================================================
        ::: input :::
        wdisp       - weight for dispersion curves (0.~1., default - 1.)
        rffactor    - downweighting factor for receiver function
        =====================================================================
        """
        if mtype == 'iso' or mtype == 'isotropic':
            self.data.get_misfit(wdisp, rffactor)
        elif mtype == 'vti':
            self.data.get_misfit_vti()
        return
    
    #==========================================
    # functions for isotropic inversions
    #==========================================
    
    def mc_joint_inv_iso(self, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40., numbcheck=None, misfit_thresh=1., \
                   isconstrt=True, pfx='MC', verbose=False, step4uwalk=1500, numbrun=15000, init_run=True, savedata=True):
        """
        Bayesian Monte Carlo joint inversion of receiver function and surface wave data for an isotropic model
        =================================================================================================================
        ::: input :::
        outdir          - output directory
        disptype        - type of dispersion curves (ph/gr/both, default - ph)
        wdisp           - weight of dispersion curve data (0. ~ 1.)
        rffactor        - factor for downweighting the misfit for likelihood computation of rf
        numbcheck       - number of runs that a checking of misfit value should be performed
        misfit_thresh   - threshold misfit value for checking
        isconstrt       - require model constraints or not
        pfx             - prefix for output, typically station id
        step4uwalk      - step interval for uniform random walk in the parameter space
        numbrun         - total number of runs
        init_run        - run and output prediction for inital model or not
                        IMPORTANT NOTE: if False, no uniform random walk will perform !
        savedata        - save data to npz binary file or not
        ---
        version history:
                    - Added the functionality of stop running if a targe misfit value is not acheived after numbcheck runs
                        Sep 27th, 2018
        =================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if numbcheck is None:
            numbcheck   = int(np.ceil(step4uwalk/2.*0.8))
        #-------------------------------
        # initializations
        #-------------------------------
        self.get_period()
        self.update_mod(mtype = 'iso')
        self.get_vmodel(mtype = 'iso')
        # output arrays
        outmodarr       = np.zeros((numbrun, self.model.isomod.para.npara+9)) # original
        outdisparr_ph   = np.zeros((numbrun, self.data.dispR.npper))
        outdisparr_gr   = np.zeros((numbrun, self.data.dispR.ngper))
        outrfarr        = np.zeros((numbrun, self.data.rfr.npts))
        # initial run
        if init_run:
            if wdisp > 0. and wdisp <= 1.:
                self.compute_fsurf()
            if wdisp < 1. and wdisp >= 0.:
                self.compute_rftheo()
            self.get_misfit(wdisp=wdisp, rffactor=rffactor)
            # write initial model
            outmod      = outdir+'/'+pfx+'.mod'
            self.model.write_model(outfname=outmod, isotropic=True)
            # write initial predicted data
            if wdisp > 0. and wdisp <= 1.:
                if dispdtype != 'both':
                    outdisp = outdir+'/'+pfx+'.'+dispdtype+'.disp'
                    self.data.dispR.writedisptxt(outfname=outdisp, dtype=dispdtype)
                else:
                    outdisp = outdir+'/'+pfx+'.ph.disp'
                    self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
                    outdisp = outdir+'/'+pfx+'.gr.disp'
                    self.data.dispR.writedisptxt(outfname=outdisp, dtype='gr')
            if wdisp < 1. and wdisp >= 0.:
                outrf       = outdir+'/'+pfx+'.rf'
                self.data.rfr.writerftxt(outfname=outrf)
            # convert initial model to para
            self.model.isomod.mod2para()
        else:
            self.model.isomod.mod2para()
            newmod      = copy.deepcopy(self.model.isomod)
            newmod.para.new_paraval(0)
            newmod.para2mod()
            newmod.update()
            # loop to find the "good" model,
            # satisfying the constraint (3), (4) and (5) in Shen et al., 2012
            m0  = 0
            m1  = 1
            # satisfying the constraint (7) in Shen et al., 2012
            if wdisp >= 1.:
                g0  = 2
                g1  = 2
            else:
                g0  = 1
                g1  = 0
            if newmod.mtype[0] == 5: # water layer, added May 16th, 2018
                m0  += 1
                m1  += 1
                g0  += 1
                g1  += 1
            igood       = 0
            while ( not newmod.isgood(m0, m1, g0, g1)):
                igood   += igood + 1
                newmod  = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(0)
                newmod.para2mod()
                newmod.update()
            # assign new model to old ones
            self.model.isomod   = newmod
            self.get_vmodel(mtype = 'iso')
            # forward computation
            if wdisp > 0. and wdisp <= 1.:
                self.compute_fsurf()
            if wdisp < 1. and wdisp >= 0.:
                self.compute_rftheo()
            self.get_misfit(wdisp=wdisp, rffactor=rffactor)
            if verbose:
                print pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
            self.model.isomod.mod2para()
        # likelihood/misfit
        oldL        = self.data.L
        oldmisfit   = self.data.misfit
        run         = True     # the key that controls the sampling
        inew        = 0     # count step (or new paras)
        iacc        = 0     # count acceptance model
        start       = time.time()
        misfitchecked \
                    = False
        while ( run ):
            inew    += 1
            if ( inew > numbrun ):
                break
            #-----------------------------------------
            # checking misfit after numbcheck runs
            # added Sep 27th, 2018
            #-----------------------------------------
            if (wdisp >= 0. and wdisp <=1.):
                if np.fmod(inew, step4uwalk) > numbcheck and not misfitchecked:
                    ind0            = int(np.ceil(inew/step4uwalk)*step4uwalk)
                    ind1            = inew-1
                    temp_min_misfit = outmodarr[ind0:ind1, self.model.isomod.para.npara+3].min()
                    if temp_min_misfit == 0.:
                        raise ValueError('Error!')
                    if temp_min_misfit > misfit_thresh:
                        # # # print 'min_misfit ='+str(temp_min_misfit)
                        inew        = int(np.ceil(inew/step4uwalk)*step4uwalk) + step4uwalk
                        if inew > numbrun:
                            break
                    misfitchecked   = True
            if (np.fmod(inew, 500) == 0) and verbose:
                print pfx, 'step =',inew, 'elasped time =', time.time()-start,' sec'
            #------------------------------------------------------------------------------------------
            # every step4uwalk step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, step4uwalk+1) == step4uwalk and init_run ):
                newmod      = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(0)
                newmod.para2mod()
                newmod.update()
                # loop to find the "good" model,
                # satisfying the constraint (3), (4) and (5) in Shen et al., 2012
                m0      = 0
                m1      = 1
                # satisfying the constraint (7) in Shen et al., 2012
                if wdisp >= 1.:
                    g0  = 2
                    g1  = 2
                else:
                    g0  = 1
                    g1  = 0
                if newmod.mtype[0] == 5: # water layer, added May 16th, 2018
                    m0  += 1
                    m1  += 1
                    g0  += 1
                    g1  += 1
                igood       = 0
                while ( not newmod.isgood(m0, m1, g0, g1)):
                    igood   += igood + 1
                    newmod  = copy.deepcopy(self.model.isomod)
                    newmod.para.new_paraval(0)
                    newmod.para2mod()
                    newmod.update()
                # assign new model to old ones
                self.model.isomod   = newmod
                self.get_vmodel()
                # forward computation
                if wdisp > 0. and wdisp <= 1.:
                    self.compute_fsurf()
                if wdisp < 1. and wdisp >= 1.:
                    self.compute_rftheo()
                self.get_misfit(wdisp=wdisp, rffactor=rffactor)
                oldL                = self.data.L
                oldmisfit           = self.data.misfit
                if verbose:
                    print pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
            #==================================================
            # inversion part
            #==================================================
            #----------------------------------
            # sample the posterior distribution
            #----------------------------------
            if (wdisp >= 0. and wdisp <=1.):
                newmod      = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(1)
                newmod.para2mod()
                newmod.update()
                if isconstrt:
                    # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                    # loop to find the "good" model, added on May 3rd, 2018
                    m0  = 0
                    m1  = 1
                    # satisfying the constraint (7) in Shen et al., 2012
                    if wdisp >= 1.:
                        g0  = 2
                        g1  = 2
                    else:
                        g0  = 1
                        g1  = 0
                    if newmod.mtype[0] == 5: # water layer, added May 16th, 2018
                        m0  += 1
                        m1  += 1
                        g0  += 1
                        g1  += 1
                    itemp   = 0
                    while (not newmod.isgood(m0, m1, g0, g1)) and itemp < 5000:
                        itemp       += 1
                        newmod      = copy.deepcopy(self.model.isomod)
                        newmod.para.new_paraval(1)
                        newmod.para2mod()
                        newmod.update()
                    if not newmod.isgood(m0, m1, g0, g1):
                        print 'No good model found!'
                        continue
                # assign new model to old ones
                oldmod              = copy.deepcopy(self.model.isomod)
                self.model.isomod   = newmod
                self.get_vmodel()
                #--------------------------------
                # forward computation
                #--------------------------------
                if wdisp > 0.:
                    self.compute_fsurf()
                if wdisp < 1.:
                    self.compute_rftheo()
                self.get_misfit(wdisp=wdisp, rffactor=rffactor)
                newL                = self.data.L
                newmisfit           = self.data.misfit
                # reject model if NaN misfit 
                if np.isnan(newmisfit):
                    print 'WARNING: '+pfx+', NaN misfit!'
                    outmodarr[inew-1, 0]                        = -1 # index for acceptance
                    outmodarr[inew-1, 1]                        = iacc
                    outmodarr[inew-1, 2:(newmod.para.npara+2)]  = newmod.para.paraval[:]
                    outmodarr[inew-1, newmod.para.npara+2]      = 0.
                    outmodarr[inew-1, newmod.para.npara+3]      = 9999.
                    outmodarr[inew-1, newmod.para.npara+4]      = self.data.rfr.L
                    outmodarr[inew-1, newmod.para.npara+5]      = self.data.rfr.misfit
                    outmodarr[inew-1, newmod.para.npara+6]      = self.data.dispR.L
                    outmodarr[inew-1, newmod.para.npara+7]      = self.data.dispR.L
                    outmodarr[inew-1, newmod.para.npara+8]      = time.time()-start
                    self.model.isomod                           = oldmod
                    continue
                if newL < oldL:
                    prob    = (oldL-newL)/oldL
                    rnumb   = random.random()
                    # reject the model
                    if rnumb < prob:
                        outmodarr[inew-1, 0]                        = -1 # index for acceptance
                        outmodarr[inew-1, 1]                        = iacc
                        outmodarr[inew-1, 2:(newmod.para.npara+2)]  = newmod.para.paraval[:]
                        outmodarr[inew-1, newmod.para.npara+2]      = newL
                        outmodarr[inew-1, newmod.para.npara+3]      = newmisfit
                        outmodarr[inew-1, newmod.para.npara+4]      = self.data.rfr.L
                        outmodarr[inew-1, newmod.para.npara+5]      = self.data.rfr.misfit
                        outmodarr[inew-1, newmod.para.npara+6]      = self.data.dispR.L
                        outmodarr[inew-1, newmod.para.npara+7]      = self.data.dispR.misfit
                        outmodarr[inew-1, newmod.para.npara+8]      = time.time()-start
                        self.model.isomod                           = oldmod
                        continue
                # accept the new model
                outmodarr[inew-1, 0]                        = 1 # index for acceptance
                outmodarr[inew-1, 1]                        = iacc
                outmodarr[inew-1, 2:(newmod.para.npara+2)]  = newmod.para.paraval[:]
                outmodarr[inew-1, newmod.para.npara+2]      = newL
                outmodarr[inew-1, newmod.para.npara+3]      = newmisfit
                outmodarr[inew-1, newmod.para.npara+4]      = self.data.rfr.L
                outmodarr[inew-1, newmod.para.npara+5]      = self.data.rfr.misfit
                outmodarr[inew-1, newmod.para.npara+6]      = self.data.dispR.L
                outmodarr[inew-1, newmod.para.npara+7]      = self.data.dispR.misfit
                outmodarr[inew-1, newmod.para.npara+8]      = time.time()-start
                # predicted dispersion data
                if wdisp > 0.:
                    if dispdtype == 'ph' or dispdtype == 'both':
                        outdisparr_ph[inew-1, :]    = self.data.dispR.pvelp[:]
                    if dispdtype == 'gr' or dispdtype == 'both':
                        outdisparr_gr[inew-1, :]    = self.data.dispR.gvelp[:]
                # predicted receiver function data
                if wdisp < 1.:
                    outrfarr[inew-1, :]             = self.data.rfr.rfp[:]
                # assign likelihood/misfit
                oldL        = newL
                oldmisfit   = newmisfit
                iacc        += 1
                continue
            #----------------------------------
            # sample the prior distribution
            #----------------------------------
            else:
                newmod      = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(1)
                newmod.para2mod()
                newmod.update()
                if isconstrt:
                    # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                    # loop to find the "good" model, added on May 3rd, 2018
                    m0      = 0
                    m1      = 1
                    # satisfying the constraint (7) in Shen et al., 2012
                    if wdisp >= 1.:
                        g0  = 2
                        g1  = 2
                    else:
                        g0  = 1
                        g1  = 0
                    if newmod.mtype[0] == 5: # water layer, added May 16th, 2018
                        m0  += 1
                        m1  += 1
                        g0  += 1
                        g1  += 1
                    itemp   = 0
                    while (not newmod.isgood(m0, m1, g0, g1)) and itemp < 5000:
                        itemp       += 1
                        newmod      = copy.deepcopy(self.model.isomod)
                        newmod.para.new_paraval(1)
                        newmod.para2mod()
                        newmod.update()
                    if not newmod.isgood(m0, m1, g0, g1):
                        print 'No good model found!'
                        continue
                self.model.isomod   = newmod
                # accept the new model
                outmodarr[inew-1, 0]                        = 1 # index for acceptance
                outmodarr[inew-1, 1]                        = iacc
                outmodarr[inew-1, 2:(newmod.para.npara+2)]  = newmod.para.paraval[:]
                outmodarr[inew-1, newmod.para.npara+2]      = 1.
                outmodarr[inew-1, newmod.para.npara+3]      = 0
                outmodarr[inew-1, newmod.para.npara+4]      = self.data.rfr.L
                outmodarr[inew-1, newmod.para.npara+5]      = self.data.rfr.misfit
                outmodarr[inew-1, newmod.para.npara+6]      = self.data.dispR.L
                outmodarr[inew-1, newmod.para.npara+7]      = self.data.dispR.misfit
                outmodarr[inew-1, newmod.para.npara+8]      = time.time() - start
                continue
        #-----------------------------------
        # write results to binary npz files
        #-----------------------------------
        outfname    = outdir+'/mc_inv.'+pfx+'.npz'
        np.savez_compressed(outfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
        if savedata:
            outdatafname\
                    = outdir+'/mc_data.'+pfx+'.npz'
            if self.data.dispR.npper > 0 and self.data.dispR.ngper > 0 and self.data.rfr.npts > 0:
                np.savez_compressed(outdatafname, np.array([1, 1, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                        self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo, \
                        self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            if self.data.dispR.npper > 0 and self.data.dispR.ngper > 0 and self.data.rfr.npts == 0:
                np.savez_compressed(outdatafname, np.array([1, 1, 0]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                        self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo)
            if self.data.dispR.npper > 0 and self.data.dispR.ngper == 0 and self.data.rfr.npts == 0:
                np.savez_compressed(outdatafname, np.array([1, 0, 0]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo)
            if self.data.dispR.npper > 0 and self.data.dispR.ngper == 0 and self.data.rfr.npts > 0:
                np.savez_compressed(outdatafname, np.array([1, 0, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                            self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            if self.data.dispR.npper == 0 and self.data.dispR.ngper > 0 and self.data.rfr.npts == 0:
                np.savez_compressed(outdatafname, np.array([0, 1, 0]), self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo)
            if self.data.dispR.npper == 0 and self.data.dispR.ngper > 0 and self.data.rfr.npts > 0:
                np.savez_compressed(outdatafname, np.array([0, 1, 1]), self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo,\
                            self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            if self.data.dispR.npper == 0 and self.data.dispR.ngper == 0 and self.data.rfr.npts > 0:
                np.savez_compressed(outdatafname, np.array([0, 0, 1]), self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            # 
            # try:
            #     np.savez_compressed(outfname, np.array([1, 1, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
            #             self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo, \
            #             self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            # except AttributeError:
            #     try:
            #         np.savez_compressed(outfname, np.array([1, 0, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
            #                 self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            #     except AttributeError:
            #         np.savez_compressed(outfname, np.array([0, 1, 1]), self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo,\
            #             self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
        del outmodarr
        del outdisparr_ph
        del outdisparr_gr
        del outrfarr
        return
    
    def mc_joint_inv_iso_mp(self, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40., isconstrt=True, pfx='MC', \
            verbose=False, step4uwalk=1500, numbrun=15000, savedata=True, subsize=1000, nprocess=None, merge=True, \
                Ntotalruns=10, misfit_thresh=2.0, Nmodelthresh=200):
        """
        Parallelized version of mc_joint_inv_iso
        ==================================================================================================================
        ::: input :::
        outdir          - output directory
        disptype        - type of dispersion curves (ph/gr/both, default - ph)
        wdisp           - weight of dispersion curve data (0. ~ 1.)
        rffactor        - factor for downweighting the misfit for likelihood computation of rf
        isconstrt       - require monotonical increase in the crust or not
        pfx             - prefix for output, typically station id
        step4uwalk      - step interval for uniform random walk in the parameter space
        numbrun         - total number of runs
        savedata        - save data to npz binary file or not
        subsize         - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess        - number of process
        merge           - merge data into one single npz file or not
        Ntotalruns      - number of times of total runs, the code would run at most numbrun*Ntotalruns iterations
        misfit_thresh   - threshold misfit value to determine "good" models
        Nmodelthresh    - required number of "good" models
        ---
        version history:
                    - Added the functionality of adding addtional runs if not enough good models found, Sep 27th, 2018
        ==================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        #-------------------------
        # prepare data
        #-------------------------
        vpr_lst = []
        Nvpr    = int(numbrun/step4uwalk)
        if Nvpr*step4uwalk != numbrun:
            print 'WARNING: number of runs changes: '+str(numbrun)+' --> '+str(Nvpr*step4uwalk)
            numbrun     = Nvpr*step4uwalk
        for i in range(Nvpr):
            temp_vpr            = copy.deepcopy(self)
            temp_vpr.process_id = i
            vpr_lst.append(temp_vpr)
        #----------------------------------------
        # Joint inversion with multiprocessing
        #----------------------------------------
        if verbose:
            print 'Start MC inversion: '+pfx+' '+time.ctime()
            stime   = time.time()
        run         = True
        i_totalrun  = 0
        imodels     = 0
        while (run):
            i_totalrun              += 1
            if Nvpr > subsize:
                Nsub                = int(len(vpr_lst)/subsize)
                for isub in xrange(Nsub):
                    print 'Subset:', isub,'in',Nsub,'sets'
                    cvpr_lst        = vpr_lst[isub*subsize:(isub+1)*subsize]
                    MCINV           = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                        isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(MCINV, cvpr_lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cvpr_lst            = vpr_lst[(isub+1)*subsize:]
                MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                        isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk, misfit_thresh=misfit_thresh)
                pool                = multiprocessing.Pool(processes=nprocess)
                pool.map(MCINV, cvpr_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                        isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk, misfit_thresh=misfit_thresh)
                pool                = multiprocessing.Pool(processes=nprocess)
                pool.map(MCINV, vpr_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            #----------------------------------------
            # Merge inversion results for each process
            #----------------------------------------
            if merge:
                outmodarr           = np.array([])
                outdisparr_ph       = np.array([])
                outdisparr_gr       = np.array([])
                outrfarr            = np.array([])
                for i in range(Nvpr):
                    invfname        = outdir+'/mc_inv.'+pfx+'_'+str(i)+'.npz'
                    inarr           = np.load(invfname)
                    outmodarr       = np.append(outmodarr, inarr['arr_0'])
                    outdisparr_ph   = np.append(outdisparr_ph, inarr['arr_1'])
                    outdisparr_gr   = np.append(outdisparr_gr, inarr['arr_2'])
                    outrfarr        = np.append(outrfarr, inarr['arr_3'])
                    os.remove(invfname)
                outmodarr           = outmodarr.reshape(numbrun, outmodarr.size/numbrun)
                outdisparr_ph       = outdisparr_ph.reshape(numbrun, outdisparr_ph.size/numbrun)
                outdisparr_gr       = outdisparr_gr.reshape(numbrun, outdisparr_gr.size/numbrun)
                outrfarr            = outrfarr.reshape(numbrun, outrfarr.size/numbrun)
                # added Sep 27th, 2018
                ind_valid           = outmodarr[:, 0] == 1.
                imodels             += np.where(outmodarr[ind_valid, temp_vpr.model.isomod.para.npara+3] <= misfit_thresh )[0].size
                if imodels >= Nmodelthresh and i_totalrun == 1:
                    outinvfname     = outdir+'/mc_inv.'+pfx+'.npz'
                    np.savez_compressed(outinvfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
                else:
                    outinvfname     = outdir+'/mc_inv.merged.'+str(i_totalrun)+'.'+pfx+'.npz'
                    np.savez_compressed(outinvfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
                # stop the loop if enough good models are found OR, number of total-runs is equal to the given threhold number
                print '== Number of good models = '+str(imodels)+', number of total runs = '+str(i_totalrun)
                if imodels >= Nmodelthresh or i_totalrun >= Ntotalruns:
                    break
        #--------------------------------------------------------
        # Merge inversion results for each additional total runs
        #--------------------------------------------------------
        if i_totalrun > 1:
            outmodarr           = np.array([])
            outdisparr_ph       = np.array([])
            outdisparr_gr       = np.array([])
            outrfarr            = np.array([])
            for i in range(i_totalrun):
                invfname        = outdir+'/mc_inv.merged.'+str(i+1)+'.'+pfx+'.npz'
                inarr           = np.load(invfname)
                outmodarr       = np.append(outmodarr, inarr['arr_0'])
                outdisparr_ph   = np.append(outdisparr_ph, inarr['arr_1'])
                outdisparr_gr   = np.append(outdisparr_gr, inarr['arr_2'])
                outrfarr        = np.append(outrfarr, inarr['arr_3'])
                os.remove(invfname)
            Nfinal_total_runs   = i_totalrun*numbrun
            outmodarr           = outmodarr.reshape(Nfinal_total_runs, outmodarr.size/Nfinal_total_runs)
            outdisparr_ph       = outdisparr_ph.reshape(Nfinal_total_runs, outdisparr_ph.size/Nfinal_total_runs)
            outdisparr_gr       = outdisparr_gr.reshape(Nfinal_total_runs, outdisparr_gr.size/Nfinal_total_runs)
            outrfarr            = outrfarr.reshape(Nfinal_total_runs, outrfarr.size/Nfinal_total_runs)
            outinvfname         = outdir+'/mc_inv.'+pfx+'.npz'
            np.savez_compressed(outinvfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
        if imodels < Nmodelthresh:
            print 'WARNING: Not enough good models, '+str(imodels)
        #----------------------------------------
        # save data
        #----------------------------------------
        if savedata:
            outdatafname    = outdir+'/mc_data.'+pfx+'.npz'
            if self.data.dispR.npper > 0 and self.data.dispR.ngper > 0 and self.data.rfr.npts > 0:
                np.savez_compressed(outdatafname, np.array([1, 1, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                        self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo, \
                        self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            if self.data.dispR.npper > 0 and self.data.dispR.ngper > 0 and self.data.rfr.npts == 0:
                np.savez_compressed(outdatafname, np.array([1, 1, 0]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                        self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo)
            if self.data.dispR.npper > 0 and self.data.dispR.ngper == 0 and self.data.rfr.npts == 0:
                np.savez_compressed(outdatafname, np.array([1, 0, 0]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo)
            if self.data.dispR.npper > 0 and self.data.dispR.ngper == 0 and self.data.rfr.npts > 0:
                np.savez_compressed(outdatafname, np.array([1, 0, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                            self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            if self.data.dispR.npper == 0 and self.data.dispR.ngper > 0 and self.data.rfr.npts == 0:
                np.savez_compressed(outdatafname, np.array([0, 1, 0]), self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo)
            if self.data.dispR.npper == 0 and self.data.dispR.ngper > 0 and self.data.rfr.npts > 0:
                np.savez_compressed(outdatafname, np.array([0, 1, 1]), self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo,\
                            self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            if self.data.dispR.npper == 0 and self.data.dispR.ngper == 0 and self.data.rfr.npts > 0:
                np.savez_compressed(outdatafname, np.array([0, 0, 1]), self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
        if verbose:
            print 'End MC inversion: '+pfx+' '+time.ctime()
            etime   = time.time()
            print 'Elapsed time: '+str(etime-stime)+' secs'
        return
    
    #==========================================
    # functions for VTI inversions
    #==========================================
    def mc_joint_inv_vti(self, outdir='./workingdir', run_inv=True, solver_type=1, numbcheck=None, misfit_thresh=1., \
                isconstrt=True, pfx='MC', verbose=False, step4uwalk=1500, numbrun=15000, init_run=True, savedata=True, \
                depth_mid_crt=-1., iulcrt=2):
        """
        Bayesian Monte Carlo joint inversion of receiver function and surface wave data for an isotropic model
        =================================================================================================================
        ::: input :::
        outdir          - output directory
        run_inv         - run the inversion or not
        solver_type     - type of solver
                            0   - fast_surf
                            1   - tcps
        numbcheck       - number of runs that a checking of misfit value should be performed
        misfit_thresh   - threshold misfit value for checking
        isconstrt       - require model constraints or not
        pfx             - prefix for output, typically station id
        step4uwalk      - step interval for uniform random walk in the parameter space
        numbrun         - total number of runs
        init_run        - run and output prediction for inital model or not
                        IMPORTANT NOTE: if False, no uniform random walk will perform !
        savedata        - save data to npz binary file or not
        ---
        version history:
                    - Added the functionality of stop running if a targe misfit value is not acheived after numbcheck runs
                        Sep 27th, 2018
        =================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if numbcheck is None:
            numbcheck   = int(np.ceil(step4uwalk/2.*0.8))
        #-------------------------------
        # initializations
        #-------------------------------
        self.get_period()
        self.update_mod(mtype = 'vti')
        self.get_vmodel(mtype = 'vti', depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
        # output arrays
        npara           = self.model.vtimod.para.npara
        outmodarr       = np.zeros((numbrun, npara+9)) # original
        outdisparr_ray  = np.zeros((numbrun, self.data.dispR.npper))
        outdisparr_lov  = np.zeros((numbrun, self.data.dispL.npper))
        # initial run
        if init_run:
            self.model.vtimod.mod2para()
            self.compute_disp_vti(wtype='both', solver_type = 1)
            self.get_misfit(mtype='vti')
            # write initial model
            outmod      = outdir+'/'+pfx+'.mod'
            self.model.write_model(outfname=outmod, isotropic=False)
            # write initial predicted data
            outdisp = outdir+'/'+pfx+'.ph.ray.disp'
            self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
            outdisp = outdir+'/'+pfx+'.ph.lov.disp'
            self.data.dispL.writedisptxt(outfname=outdisp, dtype='ph')
            if solver_type != 0:
                while not self.compute_disp_vti(wtype='both', solver_type = 1):
                    # # # print 'computing reference'
                    self.model.vtimod.new_paraval(ptype = 0)
                    self.get_vmodel(mtype = 'vti')
                self.get_misfit(mtype='vti')
            # convert initial model to para
            
        else:
            self.model.vtimod.mod2para()
            self.model.vtimod.new_paraval(ptype = 0)
            self.get_vmodel(mtype = 'vti')
            # forward computation
            if solver_type == 0:
                self.compute_disp_vti(wtype='both', solver_type = 0)
            else:
                while not self.compute_disp_vti(wtype='both', solver_type = 1):
                    # # # print 'computing reference'
                    self.model.vtimod.new_paraval(ptype = 0)
                    self.get_vmodel(mtype = 'vti')
            self.get_misfit(mtype='vti')
            if verbose:
                print pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
            self.model.vtimod.mod2para()
        # likelihood/misfit
        oldL        = self.data.L
        oldmisfit   = self.data.misfit
        run         = True      # the key that controls the sampling
        inew        = 0         # count step (or new paras)
        iacc        = 0         # count acceptance model
        start       = time.time()
        misfitchecked \
                    = False
        while ( run ):
            inew    += 1
            if ( inew > numbrun ):
                break
            #-----------------------------------------
            # checking misfit after numbcheck runs
            # added Sep 27th, 2018
            #-----------------------------------------
            if run_inv:
                if np.fmod(inew, step4uwalk) > numbcheck and not misfitchecked:
                    ind0            = int(np.ceil(inew/step4uwalk)*step4uwalk)
                    ind1            = inew-1
                    temp_min_misfit = outmodarr[ind0:ind1, npara+3].min()
                    if temp_min_misfit == 0.:
                        raise ValueError('Error!')
                    if temp_min_misfit > misfit_thresh:
                        inew        = int(np.ceil(inew/step4uwalk)*step4uwalk) + step4uwalk
                        if inew > numbrun:
                            break
                    misfitchecked   = True
            if (np.fmod(inew, 500) == 0) and verbose:
                print pfx, 'step =',inew, 'elasped time =', time.time()-start,' sec'
            #------------------------------------------------------------------------------------------
            # every step4uwalk step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, step4uwalk+1) == step4uwalk and init_run ):
                self.model.vtimod.mod2para()
                self.model.vtimod.new_paraval(ptype = 0)
                self.get_vmodel(mtype = 'vti')
                # forward computation
                if solver_type == 0:
                    self.compute_disp_vti(wtype='both', solver_type = 0)
                else:
                    while not self.compute_disp_vti(wtype='both', solver_type = 1):
                        self.model.vtimod.new_paraval(ptype = 0)
                        self.get_vmodel(mtype = 'vti')
                self.get_misfit(mtype='vti')
                oldL                = self.data.L
                oldmisfit           = self.data.misfit
                if verbose:
                    print pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
            #==================================================
            # inversion part
            #==================================================
            #----------------------------------
            # sample the posterior distribution
            #----------------------------------
            if run_inv:
                self.model.vtimod.mod2para()
                oldmod      = copy.deepcopy(self.model.vtimod)
                if not self.model.vtimod.new_paraval(ptype = 1):
                    print 'No good model found!'
                    continue
                self.get_vmodel(mtype = 'vti')
                #--------------------------------
                # forward computation
                #--------------------------------
                is_large_perturb    = False
                if solver_type == 0:
                    self.compute_disp_vti(wtype='both', solver_type = 0)
                else:
                    # compute dispersion curves based on sensitivity kernels
                    self.compute_disp_vti(wtype='both', solver_type = 2)
                    is_large_perturb= (self.data.dispR.check_large_perturb() or self.data.dispL.check_large_perturb())
                self.get_misfit(mtype='vti')
                newL                = self.data.L
                newmisfit           = self.data.misfit
                # reject model if NaN misfit 
                if np.isnan(newmisfit):
                    print 'WARNING: '+pfx+', NaN misfit!'
                    outmodarr[inew-1, 0]                = -1 # index for acceptance
                    outmodarr[inew-1, 1]                = iacc
                    outmodarr[inew-1, 2:(npara+2)]      = self.model.vtimod.para.paraval[:]
                    outmodarr[inew-1, npara+2]          = 0.
                    outmodarr[inew-1, npara+3]          = 9999.
                    outmodarr[inew-1, npara+4]          = self.data.dispR.L
                    outmodarr[inew-1, npara+5]          = self.data.dispR.misfit
                    outmodarr[inew-1, npara+6]          = self.data.dispL.L
                    outmodarr[inew-1, npara+7]          = self.data.dispL.misfit
                    outmodarr[inew-1, npara+8]          = time.time()-start
                    self.model.vtimod                   = oldmod
                    continue
                if newL < oldL:
                    prob    = (oldL-newL)/oldL
                    rnumb   = random.random()
                    # reject the model
                    if rnumb < prob:
                        outmodarr[inew-1, 0]            = -1 # index for acceptance
                        outmodarr[inew-1, 1]            = iacc
                        outmodarr[inew-1, 2:(npara+2)]  = self.model.vtimod.para.paraval[:]
                        outmodarr[inew-1, npara+2]      = newL
                        outmodarr[inew-1, npara+3]      = newmisfit
                        outmodarr[inew-1, npara+4]      = self.data.dispR.L
                        outmodarr[inew-1, npara+5]      = self.data.dispR.misfit
                        outmodarr[inew-1, npara+6]      = self.data.dispL.L
                        outmodarr[inew-1, npara+7]      = self.data.dispL.misfit
                        outmodarr[inew-1, npara+8]      = time.time()-start
                        self.model.vtimod               = oldmod
                        continue
                # update the kernels for the new reference model
                if is_large_perturb and solver_type == 1:
                    # # # print 'Update reference!'
                    oldvpr                              = copy.deepcopy(self)
                    if not self.compute_disp_vti(wtype='both', solver_type = 1):
                        self                            = oldvpr # reverse to be original vpr with old kernels
                        outmodarr[inew-1, 0]            = -1 # index for acceptance
                        outmodarr[inew-1, 1]            = iacc
                        outmodarr[inew-1, 2:(npara+2)]  = self.model.vtimod.para.paraval[:]
                        outmodarr[inew-1, npara+2]      = newL
                        outmodarr[inew-1, npara+3]      = newmisfit
                        outmodarr[inew-1, npara+4]      = self.data.dispR.L
                        outmodarr[inew-1, npara+5]      = self.data.dispR.misfit
                        outmodarr[inew-1, npara+6]      = self.data.dispL.L
                        outmodarr[inew-1, npara+7]      = self.data.dispL.misfit
                        outmodarr[inew-1, npara+8]      = time.time()-start
                        self.model.vtimod               = oldmod
                        continue
                    self.get_misfit(mtype='vti')
                    newL                                = self.data.L
                    newmisfit                           = self.data.misfit
                # accept the new model
                outmodarr[inew-1, 0]                    = 1 # index for acceptance
                outmodarr[inew-1, 1]                    = iacc
                outmodarr[inew-1, 2:(npara+2)]          = self.model.vtimod.para.paraval[:]
                outmodarr[inew-1, npara+2]              = newL
                outmodarr[inew-1, npara+3]              = newmisfit
                outmodarr[inew-1, npara+4]              = self.data.dispR.L
                outmodarr[inew-1, npara+5]              = self.data.dispR.misfit
                outmodarr[inew-1, npara+6]              = self.data.dispL.L
                outmodarr[inew-1, npara+7]              = self.data.dispL.misfit
                outmodarr[inew-1, npara+8]              = time.time()-start
                # predicted dispersion data
                outdisparr_ray[inew-1, :]               = self.data.dispR.pvelp[:]
                outdisparr_lov[inew-1, :]               = self.data.dispL.pvelp[:]
                # assign likelihood/misfit
                oldL        = newL
                oldmisfit   = newmisfit
                iacc        += 1
                # # # print inew, oldmisfit
                continue
            #----------------------------------
            # sample the prior distribution
            #----------------------------------
            else:
                self.model.vtimod.new_paraval(ptype = 0, isconstrt=isconstrt)
                # accept the new model
                outmodarr[inew-1, 0]                    = 1 # index for acceptance
                outmodarr[inew-1, 1]                    = iacc
                outmodarr[inew-1, 2:(npara+2)]          = self.model.vtimod.para.paraval[:]
                outmodarr[inew-1, npara+2]              = 1.
                outmodarr[inew-1, npara+3]              = 0
                outmodarr[inew-1, npara+4]              = self.data.dispR.L
                outmodarr[inew-1, npara+5]              = self.data.dispR.misfit
                outmodarr[inew-1, npara+6]              = self.data.dispL.L
                outmodarr[inew-1, npara+7]              = self.data.dispL.misfit
                outmodarr[inew-1, npara+8]              = time.time() - start
                continue
        #-----------------------------------
        # write results to binary npz files
        #-----------------------------------
        outfname    = outdir+'/mc_inv.'+pfx+'.npz'
        np.savez_compressed(outfname, outmodarr, outdisparr_ray, outdisparr_lov)
        if savedata:
            outdatafname\
                    = outdir+'/mc_data.'+pfx+'.npz'
            np.savez_compressed(outdatafname, self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                        self.data.dispL.pper, self.data.dispL.pvelo, self.data.dispL.stdpvelo)
        del outmodarr
        del outdisparr_ray
        del outdisparr_lov
        return
    
    def mc_joint_inv_vti_mp(self, outdir='./workingdir', run_inv=True, solver_type=1, isconstrt=True, pfx='MC',\
                verbose=False, step4uwalk=1500, numbrun=15000, savedata=True, subsize=1000,
                nprocess=None, merge=True, Ntotalruns=2, misfit_thresh=2.0, Nmodelthresh=200, depth_mid_crt=-1., iulcrt=2):
        """
        Parallelized version of mc_joint_inv_iso
        ==================================================================================================================
        ::: input :::
        outdir          - output directory
        run_inv         - run the inversion or not
        solver_type     - type of solver
                            0   - fast_surf
                            1   - tcps
        isconstrt       - require monotonical increase in the crust or not
        pfx             - prefix for output, typically station id
        step4uwalk      - step interval for uniform random walk in the parameter space
        numbrun         - total number of runs
        savedata        - save data to npz binary file or not
        subsize         - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess        - number of process
        merge           - merge data into one single npz file or not
        Ntotalruns      - number of times of total runs, the code would run at most numbrun*Ntotalruns iterations
        misfit_thresh   - threshold misfit value to determine "good" models
        Nmodelthresh    - required number of "good" models
        ---
        version history:
                    - Added the functionality of adding addtional runs if not enough good models found, Sep 27th, 2018
        ==================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        #-------------------------
        # prepare data
        #-------------------------
        vpr_lst = []
        Nvpr    = int(numbrun/step4uwalk)
        npara   = self.model.vtimod.para.npara
        if Nvpr*step4uwalk != numbrun:
            print 'WARNING: number of runs changes: '+str(numbrun)+' --> '+str(Nvpr*step4uwalk)
            numbrun     = Nvpr*step4uwalk
        for i in range(Nvpr):
            temp_vpr            = copy.deepcopy(self)
            temp_vpr.process_id = i
            vpr_lst.append(temp_vpr)
        #----------------------------------------
        # Joint inversion with multiprocessing
        #----------------------------------------
        if verbose:
            print 'Start MC inversion: '+pfx+' '+time.ctime()
            stime       = time.time()
        run             = True
        i_totalrun      = 0
        imodels         = 0
        need_to_merge   = False
        while (run):
            i_totalrun              += 1
            if Nvpr > subsize:
                Nsub                = int(len(vpr_lst)/subsize)
                for isub in xrange(Nsub):
                    print 'Subset:', isub,'in',Nsub,'sets'
                    cvpr_lst        = vpr_lst[isub*subsize:(isub+1)*subsize]
                    MCINV           = partial(mc4mp_vti, outdir=outdir, run_inv=run_inv, solver_type=solver_type,
                                        isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk, misfit_thresh=misfit_thresh, \
                                        depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(MCINV, cvpr_lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cvpr_lst            = vpr_lst[(isub+1)*subsize:]
                MCINV               = partial(mc4mp_vti, outdir=outdir, run_inv=run_inv, solver_type=solver_type,
                                        isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk, misfit_thresh=misfit_thresh, \
                                        depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
                pool                = multiprocessing.Pool(processes=nprocess)
                pool.map(MCINV, cvpr_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                MCINV               = partial(mc4mp_vti, outdir=outdir, run_inv=run_inv, solver_type=solver_type,
                                        isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk, misfit_thresh=misfit_thresh, \
                                        depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
                pool                = multiprocessing.Pool(processes=nprocess)
                pool.map(MCINV, vpr_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            #----------------------------------------
            # Merge inversion results for each process
            #----------------------------------------
            if merge:
                outmodarr           = np.array([])
                outdisparr_ray      = np.array([])
                outdisparr_lov      = np.array([])
                for i in range(Nvpr):
                    invfname        = outdir+'/mc_inv.'+pfx+'_'+str(i)+'.npz'
                    inarr           = np.load(invfname)
                    outmodarr       = np.append(outmodarr, inarr['arr_0'])
                    outdisparr_ray  = np.append(outdisparr_ray, inarr['arr_1'])
                    outdisparr_lov  = np.append(outdisparr_lov, inarr['arr_2'])
                    os.remove(invfname)
                outmodarr           = outmodarr.reshape(numbrun, outmodarr.size/numbrun)
                outdisparr_ray      = outdisparr_ray.reshape(numbrun, outdisparr_ray.size/numbrun)
                outdisparr_lov      = outdisparr_lov.reshape(numbrun, outdisparr_lov.size/numbrun)
                # added Sep 27th, 2018
                ind_valid           = outmodarr[:, 0] == 1.
                imodels             += np.where(outmodarr[ind_valid, npara+3] <= misfit_thresh )[0].size
                if imodels >= Nmodelthresh and i_totalrun == 1:
                    outinvfname     = outdir+'/mc_inv.'+pfx+'.npz'
                    np.savez_compressed(outinvfname, outmodarr, outdisparr_ray, outdisparr_lov)
                else:
                    outinvfname     = outdir+'/mc_inv.merged.'+str(i_totalrun)+'.'+pfx+'.npz'
                    np.savez_compressed(outinvfname, outmodarr, outdisparr_ray, outdisparr_lov)
                    need_to_merge   = True
                # stop the loop if enough good models are found OR, number of total-runs is equal to the given threhold number
                print '== Number of good models = '+str(imodels)+', number of total runs = '+str(i_totalrun)
                if imodels >= Nmodelthresh or i_totalrun >= Ntotalruns:
                    break
        #--------------------------------------------------------
        # Merge inversion results for each additional total runs
        #--------------------------------------------------------
        if need_to_merge:
            outmodarr           = np.array([])
            outdisparr_ray      = np.array([])
            outdisparr_lov      = np.array([])
            for i in range(i_totalrun):
                invfname        = outdir+'/mc_inv.merged.'+str(i+1)+'.'+pfx+'.npz'
                inarr           = np.load(invfname)
                outmodarr       = np.append(outmodarr, inarr['arr_0'])
                outdisparr_ray  = np.append(outdisparr_ray, inarr['arr_1'])
                outdisparr_lov  = np.append(outdisparr_lov, inarr['arr_2'])
                os.remove(invfname)
            Nfinal_total_runs   = i_totalrun*numbrun
            outmodarr           = outmodarr.reshape(Nfinal_total_runs, outmodarr.size/Nfinal_total_runs)
            outdisparr_ray      = outdisparr_ray.reshape(Nfinal_total_runs, outdisparr_ray.size/Nfinal_total_runs)
            outdisparr_lov      = outdisparr_lov.reshape(Nfinal_total_runs, outdisparr_lov.size/Nfinal_total_runs)
            outinvfname         = outdir+'/mc_inv.'+pfx+'.npz'
            np.savez_compressed(outinvfname, outmodarr, outdisparr_ray, outdisparr_lov)
        if imodels < Nmodelthresh:
            print 'WARNING: Not enough good models, '+str(imodels)
        #----------------------------------------
        # save data
        #----------------------------------------
        if savedata:
            outdatafname\
                    = outdir+'/mc_data.'+pfx+'.npz'
            np.savez_compressed(outdatafname, self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                        self.data.dispL.pper, self.data.dispL.pvelo, self.data.dispL.stdpvelo)
        if verbose:
            print 'End MC inversion: '+pfx+' '+time.ctime()
            etime   = time.time()
            print 'Elapsed time: '+str(etime-stime)+' secs'
        return
    
    #==========================================
    # functions for HTI inversions
    #==========================================
    def linear_inv_hti(self, isBcs=True, useref=False, depth_mid_crust=15., depth_mid_mantle=-1., usespl_man=False):
        # construct data array
        dc      = np.zeros(self.data.dispR.npper, dtype=np.float64)
        ds      = np.zeros(self.data.dispR.npper, dtype=np.float64)
        if useref:
            try:
                A2      = self.data.dispR.amp/100.*self.data.dispR.pvelref
                unA2    = self.data.dispR.unamp/100.*self.data.dispR.pvelref
                vel_iso = self.data.dispR.pvelref
            except:
                raise ValueError('No refernce dispersion curve stored!')
        else:
            A2      = self.data.dispR.amp/100.*self.data.dispR.pvelo
            unA2    = self.data.dispR.unamp/100.*self.data.dispR.pvelo
            vel_iso = self.data.dispR.pvelo
        dc[:]       = A2*np.cos(2. * (self.data.dispR.psi2/180.*np.pi) )
        ds[:]       = A2*np.sin(2. * (self.data.dispR.psi2/180.*np.pi) )
        #--------------------------
        # data covariance matrix
        #--------------------------
        A2_with_un  = unumpy.uarray(A2, unA2)
        psi2_with_un= unumpy.uarray(self.data.dispR.psi2, self.data.dispR.unpsi2)
        # dc
        Cdc         = np.zeros((self.data.dispR.npper, self.data.dispR.npper), dtype=np.float64)
        undc        = unumpy.std_devs( A2_with_un * unumpy.cos(2. * (psi2_with_un/180.*np.pi)) )
        np.fill_diagonal(Cdc, undc**2)
        # ds
        Cds         = np.zeros((self.data.dispR.npper, self.data.dispR.npper), dtype=np.float64)
        unds        = unumpy.std_devs( A2_with_un * unumpy.sin(2. * (psi2_with_un/180.*np.pi)) )
        np.fill_diagonal(Cds, unds**2)
        #--------------------------
        # forward operator matrix
        #--------------------------
        nmod        = 2
        if depth_mid_crust > 0.:
            nmod    += 1
        if depth_mid_mantle > 0.:
            nmod    += 1
        self.model.htimod.init_arr(nmod)
        self.model.htimod.set_depth_disontinuity(depth_mid_crust=depth_mid_crust, depth_mid_mantle=depth_mid_mantle)
        self.model.get_hti_layer_ind()
        # if usespl_man:
            
        # forward matrix
        G           = np.zeros((self.data.dispR.npper, nmod), dtype=np.float64)
        for i in range(nmod):
            ind0    = self.model.htimod.layer_ind[i, 0]
            ind1    = self.model.htimod.layer_ind[i, 1]
            dcdX    = self.eigkR.dcdL[:, ind0:ind1]
            if isBcs:
                dcdX+= self.eigkR.dcdA[:, ind0:ind1] * self.eigkR.Aeti[ind0:ind1]/self.eigkR.Leti[ind0:ind1]
            dcdX    *= self.eigkR.Leti[ind0:ind1]
            G[:, i] = dcdX.sum(axis=1)
        #--------------------------
        # solve the inverse problem
        #--------------------------
        # cosine terms
        Ginv1                       = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cdc)), G) )
        Ginv2                       = np.dot( np.dot(G.T, np.linalg.inv(Cdc)), dc)
        modelC                      = np.dot(Ginv1, Ginv2)
        Cmc                         = Ginv1 # model covariance matrix
        pcovc                       = np.sqrt(np.absolute(Cmc))
        self.model.htimod.Gc[:]     = modelC[:]
        self.model.htimod.unGc[:]   = pcovc.diagonal()
        # sine terms
        Ginv1                       = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cds)), G) )
        Ginv2                       = np.dot( np.dot(G.T, np.linalg.inv(Cds)), ds)
        modelS                      = np.dot(Ginv1, Ginv2)
        Cms                         = Ginv1 # model covariance matrix
        pcovs                       = np.sqrt(np.absolute(Cms))
        self.model.htimod.Gs[:]     = modelS[:]
        self.model.htimod.unGs[:]   = pcovs.diagonal()
        self.model.htimod.GcGs_to_azi()
        #--------------------------
        # predictions
        #--------------------------
        pre_dc                  = np.dot(G, self.model.htimod.Gc)
        pre_ds                  = np.dot(G, self.model.htimod.Gs)
        pre_amp                 = np.sqrt(pre_dc**2 + pre_ds**2)
        pre_amp                 = pre_amp/vel_iso*100.
        self.data.dispR.pamp    = pre_amp
        pre_psi                 = np.arctan2(pre_ds, pre_dc)/2./np.pi*180.
        pre_psi[pre_psi<0.]     += 180.
        self.data.dispR.ppsi2   = pre_psi
        self.data.get_misfit_hti()
        return
    
    def linear_inv_hti_twolayer(self, depth=-2., isBcs=True, useref=False, maxdepth=-3.,\
                                depth2d = np.array([])):
        # construct data array
        dc      = np.zeros(self.data.dispR.npper, dtype=np.float64)
        ds      = np.zeros(self.data.dispR.npper, dtype=np.float64)
        if useref:
            try:
                A2      = self.data.dispR.amp/100.*self.data.dispR.pvelref
                unA2    = self.data.dispR.unamp/100.*self.data.dispR.pvelref
                vel_iso = self.data.dispR.pvelref
            except:
                raise ValueError('No refernce dispersion curve stored!')
        else:
            A2      = self.data.dispR.amp/100.*self.data.dispR.pvelo
            unA2    = self.data.dispR.unamp/100.*self.data.dispR.pvelo
            vel_iso = self.data.dispR.pvelo
        dc[:]       = A2*np.cos(2. * (self.data.dispR.psi2/180.*np.pi) )
        ds[:]       = A2*np.sin(2. * (self.data.dispR.psi2/180.*np.pi) )
        #--------------------------
        # data covariance matrix
        #--------------------------
        A2_with_un  = unumpy.uarray(A2, unA2)
        psi2_with_un= unumpy.uarray(self.data.dispR.psi2, self.data.dispR.unpsi2)
        # dc
        Cdc         = np.zeros((self.data.dispR.npper, self.data.dispR.npper), dtype=np.float64)
        undc        = unumpy.std_devs( A2_with_un * unumpy.cos(2. * (psi2_with_un/180.*np.pi)) )
        np.fill_diagonal(Cdc, undc**2)
        # ds
        Cds         = np.zeros((self.data.dispR.npper, self.data.dispR.npper), dtype=np.float64)
        unds        = unumpy.std_devs( A2_with_un * unumpy.sin(2. * (psi2_with_un/180.*np.pi)) )
        np.fill_diagonal(Cds, unds**2)
        #--------------------------
        # forward operator matrix
        #--------------------------
        nmod        = 2
        self.model.htimod.init_arr(nmod)
        if depth2d.shape[0] != nmod:       
            self.model.htimod.depth[0]  = -1
            self.model.htimod.depth[1]  = depth
            self.model.htimod.depth[2]  = maxdepth
            self.model.get_hti_layer_ind()
        else:
            self.model.htimod.depth2d[:, :] = depth2d.copy()
            if self.model.htimod.depth2d[1, 1]  == -3.:
                self.model.htimod.depth2d[1, 1] = maxdepth
            # # # self.model.htimod.depth2d[0, 0] = -1
            # # # self.model.htimod.depth2d[0, 1] = 15.
            # # # self.model.htimod.depth2d[1, 0] = depth
            # # # self.model.htimod.depth2d[1, 1] = maxdepth
            self.model.get_hti_layer_ind_2d()
        # forward matrix
        G           = np.zeros((self.data.dispR.npper, nmod), dtype=np.float64)
        for i in range(nmod):
            ind0    = self.model.htimod.layer_ind[i, 0]
            ind1    = self.model.htimod.layer_ind[i, 1]
            dcdX    = self.eigkR.dcdL[:, ind0:ind1]
            if isBcs:
                dcdX+= self.eigkR.dcdA[:, ind0:ind1] * self.eigkR.Aeti[ind0:ind1]/self.eigkR.Leti[ind0:ind1]
            dcdX    *= self.eigkR.Leti[ind0:ind1]
            G[:, i] = dcdX.sum(axis=1)
        #--------------------------
        # solve the inverse problem
        #--------------------------
        # cosine terms
        Ginv1                       = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cdc)), G) )
        Ginv2                       = np.dot( np.dot(G.T, np.linalg.inv(Cdc)), dc)
        modelC                      = np.dot(Ginv1, Ginv2)
        Cmc                         = Ginv1 # model covariance matrix
        pcovc                       = np.sqrt(np.absolute(Cmc))
        self.model.htimod.Gc[:]     = modelC[:]
        self.model.htimod.unGc[:]   = pcovc.diagonal()
        # sine terms
        Ginv1                       = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cds)), G) )
        Ginv2                       = np.dot( np.dot(G.T, np.linalg.inv(Cds)), ds)
        modelS                      = np.dot(Ginv1, Ginv2)
        Cms                         = Ginv1 # model covariance matrix
        pcovs                       = np.sqrt(np.absolute(Cms))
        self.model.htimod.Gs[:]     = modelS[:]
        self.model.htimod.unGs[:]   = pcovs.diagonal()
        self.model.htimod.GcGs_to_azi()
        #--------------------------
        # predictions
        #--------------------------
        pre_dc                  = np.dot(G, self.model.htimod.Gc)
        pre_ds                  = np.dot(G, self.model.htimod.Gs)
        pre_amp                 = np.sqrt(pre_dc**2 + pre_ds**2)
        pre_amp                 = pre_amp/vel_iso*100.
        self.data.dispR.pamp    = pre_amp
        pre_psi                 = np.arctan2(pre_ds, pre_dc)/2./np.pi*180.
        pre_psi[pre_psi<0.]     += 180.
        self.data.dispR.ppsi2   = pre_psi
        self.data.get_misfit_hti()
        return
    
def mc4mp(invpr, outdir, dispdtype, wdisp, rffactor, isconstrt, pfx, verbose, numbrun, misfit_thresh):
    # print '--- MC inversion for station/grid: '+pfx+', process id: '+str(invpr.process_id)
    pfx     = pfx +'_'+str(invpr.process_id)
    if invpr.process_id == 0 or wdisp < 0.:
        invpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, misfit_thresh=misfit_thresh, \
                       isconstrt=isconstrt, pfx=pfx, verbose=False, step4uwalk=numbrun, numbrun=numbrun, init_run=True, savedata=False)
    else:
        invpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, misfit_thresh=misfit_thresh, \
                       isconstrt=isconstrt, pfx=pfx, verbose=False, step4uwalk=numbrun, numbrun=numbrun, init_run=False, savedata=False)
    return

def mc4mp_vti(invpr, outdir, run_inv, solver_type, isconstrt, pfx, verbose, numbrun, misfit_thresh, \
              depth_mid_crt, iulcrt):
    # print '--- MC inversion for station/grid: '+pfx+', process id: '+str(invpr.process_id)
    pfx     = pfx +'_'+str(invpr.process_id)
    if invpr.process_id == 0:
        invpr.mc_joint_inv_vti(outdir=outdir, run_inv=run_inv, misfit_thresh=misfit_thresh, \
            isconstrt=isconstrt, pfx=pfx, verbose=False, step4uwalk=numbrun, numbrun=numbrun, init_run=True, savedata=False, \
            depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
    else:
        invpr.mc_joint_inv_vti(outdir=outdir, run_inv=run_inv, misfit_thresh=misfit_thresh, \
            isconstrt=isconstrt, pfx=pfx, verbose=False, step4uwalk=numbrun, numbrun=numbrun, init_run=False, savedata=False, \
            depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
    return