# -*- coding: utf-8 -*-
"""
base for inversion of 1d models

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""

import surfpy.pymcinv._data as _data
import surfpy.pymcinv.vmodel as vmodel
import surfpy.pymcinv.eigenkernel as eigenkernel

import numpy as np
import os
import copy
from uncertainties import unumpy

#
WATER   = 5

class base_vprofile(object):
    """base class for 1D velocity profile inversion, I/O part
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
        self.data       = _data.data1d()
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
    
    def read_disp(self, infname, dtype='ph', wtype='ray'):
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
            self.data.dispR.read(infname=infname, dtype=dtype)
            if self.data.dispR.npper>0:
                self.data.dispR.pvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
                self.data.dispR.gvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            self.data.dispL.read(infname=infname, dtype=dtype)
            if self.data.dispL.npper>0:
                self.data.dispL.pvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
                self.data.dispL.gvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return
    
    def get_disp(self, indata, dtype='ph', wtype='ray'):
        """read dispersion curve data from numpy array
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
        """read dispersion curve data from numpy array
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

    def read_rf(self, infname, dtype='r'):
        """read receiver function data from a txt file
        ===========================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (radial or trnasverse)
        ===========================================================
        """
        dtype=dtype.lower()
        if dtype=='r' or dtype == 'radial':
            self.data.rfr.read(infname)
            self.data.rfr.tp    = np.linspace(self.data.rfr.to[0], self.data.rfr.to[-1], \
                        self.data.rfr.npts, dtype=np.float64)
            self.data.rfr.rfp   = np.zeros(self.data.rfr.npts, dtype=np.float64)
            self.npts           = self.data.rfr.npts
            self.fs             = 1./(self.data.rfr.to[1] - self.data.rfr.to[0])
        elif dtype=='t' or dtype == 'transverse':
            self.data.rft.read(infname)
        else:
            raise ValueError('Unexpected ref type: '+dtype)
        return
    
    def get_rf_old(self, indata, dtype='r'):
        """read receiver function data from numpy array
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
            raise ValueError('Unexpected ref type: '+dtype)
        return
    
    def get_rf(self, indata, delta, dtype='r'):
        """read receiver function data from numpy array
        ===========================================================
        ::: input :::
        indata      - input data array (2, N)
        dtype       - data type (radial or transverse)
        ===========================================================
        """
        dtype       = dtype.lower()
        if dtype=='r' or dtype == 'radial':
            npts    = indata.shape[1]
            to      = np.arange(npts) * delta
            indata  = np.append(to, indata)
            indata  = indata.reshape(3, npts)
            self.data.rfr.get_rf(indata = indata)
            self.data.rfr.tp    = np.linspace(self.data.rfr.to[0], self.data.rfr.to[-1], \
                        self.data.rfr.npts, dtype=np.float64)
            self.data.rfr.rfp   = np.zeros(self.data.rfr.npts, dtype=np.float64)
            self.npts           = npts
            self.fs             = 1./delta
        # # elif dtype=='t' or dtype == 'transverse':
        # #     self.data.rft.readrftxt(infname)
        else:
            raise ValueError('Unexpected ref type: '+dtype)
        return
    
    def read_mod(self, infname, mtype='iso'):
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
            self.model.isomod.read(infname)
        # elif mtype == 'tti':
        #     self.model.ttimod.readttimodtxt(infname)
        else:
            raise ValueError('Unexpected model type: '+mtype)
        return
    
    def read_para(self, infname, mtype='iso'):
        """read parameter index indicating model parameters for perturbation
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.isomod.para.read(infname)
        else:
            raise ValueError('Unexpected model type: '+mtype)
        return
    
    def get_paraind(self, mtype='iso', crtthk = None, crtstd = 10.):
        """
        get parameter index indicating model parameters for perturbation
        =====================================================================
        ::: input :::
        mtype       - model type (isotropic or Love)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.isomod.get_paraind(crtthk = crtthk, crtstd = crtstd)
        elif mtype == 'vti':
            self.model.vtimod.get_paraind(crtthk = crtthk, crtstd = crtstd)
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
    
    def get_vmodel(self, mtype='iso'):
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
            self.model.get_vti_vmodel()
        else:
            raise ValueError('Unexpected model type: '+ mtype)
        return 
    
    def get_period(self):
        """get period array for forward modelling
        """
        if self.data.dispR.npper>0:
            self.TRp        = self.data.dispR.pper.copy()
        if self.data.dispR.ngper>0:
            self.TRg        = self.data.dispR.gper.copy()
        if self.data.dispR.npper>0 and self.data.dispR.ngper>0:
            if not np.allclose(self.TRp[:self.data.dispR.ngper], self.TRg):
                raise ValueError('incompatible phase/group periods!')
        if self.data.dispL.npper>0:
            self.TLp        = self.data.dispL.pper.copy()
        if self.data.dispL.ngper>0:
            self.TLg        = self.data.dispL.gper.copy()
        if self.data.dispL.npper>0 and self.data.dispL.ngper>0:
            if not np.allclose(self.TLp[:self.data.dispL.ngper], self.TLg):
                raise ValueError('incompatible phase/group periods!')
        return