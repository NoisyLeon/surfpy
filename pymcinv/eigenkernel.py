# -*- coding: utf-8 -*-
"""
Module for handling output eigenfunction and sensitivity kernels of surface waves in tilted TI model


references
    Montagner, J.P. and Nataf, H.C., 1986. A simple method for inverting the azimuthal anisotropy of surface waves.
            Journal of Geophysical Research: Solid Earth, 91(B1), pp.511-520.

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import math


class eigkernel(object):
    """a class for handling eigenfunctions and sensitivity kernels 
    =====================================================================================================================
    ::: parameters :::
    :   values  :
    nlay        - number of layers
    ilvry       - indicator for Love or Rayleigh waves (1 - Love, 2 - Rayleigh)
    :   model   :
    A, C, F, L, N, rho                          - layerized model
    If the model is a tilted hexagonal symmetric model:
    BcArr, BsArr, GcArr, GsArr, HcArr, HsArr    - 2-theta azimuthal terms 
    CcArr, CsArr                                - 4-theta azimuthal terms
    : eigenfunctions :
    uz, ur      - vertical/radial displacement functions
    tuz, tur    - vertical/radial stress functions
    duzdz, durdz- derivatives of vertical/radial displacement functions
    : velocity/density sensitivity kernels :
    dcdah/dcdav - vph/vpv kernel
    dcdbh/dcdbv - vsh/vsv kernel
    dcdn        - eta kernel
    dcdr        - density kernel
    : Love parameters/density sensitivity kernels, derived from the kernels above using chain rule :
    dcdA, dcdC, dcdF, dcdL, dcdN    - Love parameter kernels
    dcdrl                           - density kernel
    =====================================================================================================================
    """
    def __init__(self):
        self.nfreq      = 0
        self.nlay       = -1
        self.ilvry      = -1
        return
    
    def init_arr(self, nfreq, nlay, ilvry):
        """
        initialize arrays
        """
        if ilvry != 1 and ilvry != 2:
            raise ValueError('Unexpected ilvry value!')
        self.nfreq      = nfreq
        self.nlay       = nlay
        self.ilvry      = ilvry
        # reference Love parameters and density
        self.A          = np.zeros(np.int64(nlay), dtype=np.float64)
        self.C          = np.zeros(np.int64(nlay), dtype=np.float64)
        self.F          = np.zeros(np.int64(nlay), dtype=np.float64)
        self.L          = np.zeros(np.int64(nlay), dtype=np.float64)
        self.N          = np.zeros(np.int64(nlay), dtype=np.float64)
        self.rho        = np.zeros(np.int64(nlay), dtype=np.float64)
        # reference velocity parameters and density
        self.ah         = np.zeros(np.int64(nlay), dtype=np.float64)
        self.av         = np.zeros(np.int64(nlay), dtype=np.float64)
        self.bh         = np.zeros(np.int64(nlay), dtype=np.float64)
        self.bv         = np.zeros(np.int64(nlay), dtype=np.float64)
        self.n          = np.zeros(np.int64(nlay), dtype=np.float64)
        self.r          = np.zeros(np.int64(nlay), dtype=np.float64)
        # ETI Love parameters and density
        self.Aeti       = np.zeros(np.int64(nlay), dtype=np.float64)
        self.Ceti       = np.zeros(np.int64(nlay), dtype=np.float64)
        self.Feti       = np.zeros(np.int64(nlay), dtype=np.float64)
        self.Leti       = np.zeros(np.int64(nlay), dtype=np.float64)
        self.Neti       = np.zeros(np.int64(nlay), dtype=np.float64)
        self.rhoeti     = np.zeros(np.int64(nlay), dtype=np.float64)
        # ETI velocity parameters and density
        self.aheti      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.aveti      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.bheti      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.bveti      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.neti       = np.zeros(np.int64(nlay), dtype=np.float64)
        self.reti       = np.zeros(np.int64(nlay), dtype=np.float64)
        # azimuthal anisotropic terms
        self.BcArr      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.BsArr      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.GcArr      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.GsArr      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.HcArr      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.HsArr      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.CcArr      = np.zeros(np.int64(nlay), dtype=np.float64)
        self.CsArr      = np.zeros(np.int64(nlay), dtype=np.float64)
        # eigenfunctions
        if ilvry == 1:
            self.ut     = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.tut    = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        else:
            self.uz     = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.tuz    = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.ur     = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.tur    = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        # velocity kernels
        if ilvry == 2:
            self.Khti   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.dcdah  = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.dcdav  = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.dcdn   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        self.dcdbh      = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        self.dcdbv      = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        self.dcdr       = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        # Love kernels
        if ilvry == 2:
            self.dcdA   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.dcdC   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
            self.dcdF   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        self.dcdL       = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        self.dcdN       = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        # density kernel for Love parameter group
        self.dcdrl      = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float64)
        return
    
    def get_ref_model(self, A, C, F, L, N, rho):
        """
        get the Love parameter arrays for the reference model
        """
        self.A[:]       = A
        self.C[:]       = C
        self.F[:]       = F
        self.L[:]       = L
        self.N[:]       = N
        self.rho[:]     = rho
        return
    
    def get_ETI(self, Aeti, Ceti, Feti, Leti, Neti, rhoeti):
        """
        get the ETI(effective TI) Love parameter arrays, as perturbation
        """
        self.Aeti[:]    = Aeti
        self.Ceti[:]    = Ceti
        self.Feti[:]    = Feti
        self.Leti[:]    = Leti
        self.Neti[:]    = Neti
        self.rhoeti[:]  = rhoeti
        return
    
    def get_ref_model_vel(self, ah, av, bh, bv, n, r):
        """
        get the velocity parameter arrays for the reference model
        """
        self.ah[:]      = ah
        self.av[:]      = av
        self.bh[:]      = bh
        self.bv[:]      = bv
        self.n[:]       = n
        self.r[:]       = r
        return
    
    def get_ETI_vel(self, aheti, aveti, bheti, bveti, neti, reti):
        """
        get the ETI(effective TI) velocity parameter arrays, as perturbation
        """
        self.aheti[:]   = aheti
        self.aveti[:]   = aveti
        self.bheti[:]   = bheti
        self.bveti[:]   = bveti
        self.neti[:]    = neti
        self.reti[:]    = reti
        return
    
    def get_AA(self, BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr):
        """
        get the AA(azimuthally anisotropic) term arrays, as perturbation
        """
        self.BcArr[:]  = BcArr
        self.BsArr[:]  = BsArr
        self.GcArr[:]  = GcArr
        self.GsArr[:]  = GsArr
        self.HcArr[:]  = HcArr
        self.HsArr[:]  = HsArr
        self.CcArr[:]  = CcArr
        self.CsArr[:]  = CsArr
        return
    
    def get_eigen_psv(self, uz, tuz, ur, tur):
        """
        get the P-SV motion eigenfunctions
        """
        self.uz[:,:]    = uz
        self.tuz[:,:]   = tuz
        self.ur[:,:]    = ur
        self.tur[:,:]   = tur
        return
    
    def get_eigen_sh(self, ut, tut):
        """
        get the SH motion eigenfunctions
        """
        self.ut = ut
        self.tut= tut
        return
    
    def get_vkernel_psv(self, dcdah, dcdav, dcdbh, dcdbv, dcdn, dcdr):
        """
        get the velocity kernels for P-SV motion
        """
        self.dcdah[:,:]     = dcdah
        self.dcdav[:,:]     = dcdav
        self.dcdbh[:,:]     = dcdbh
        self.dcdbv[:,:]     = dcdbv
        self.dcdr[:,:]      = dcdr
        self.dcdn[:,:]      = dcdn
        return
    
    def get_vkernel_sh(self, dcdbh, dcdbv, dcdr):
        """
        get the velocity kernels for SH motion
        """
        self.dcdbh[:,:]     = dcdbh
        self.dcdbv[:,:]     = dcdbv
        self.dcdr[:,:]      = dcdr
        return
    
    def compute_love_kernels(self):
        """
        compute sensitivity kernels for Love paramters using chain rule
        """
        if self.ilvry == 2:
            for i in xrange(self.nfreq):
                for j in xrange(self.nlay):
                    self.dcdA[i, j] = 0.5/np.sqrt(self.A[j]*self.rho[j]) * self.dcdah[i,j] - self.F[j]/((self.A[j]-2.*self.L[j])**2)*self.dcdn[i, j]
                    self.dcdC[i, j] = 0.5/np.sqrt(self.C[j]*self.rho[j]) * self.dcdav[i,j]
                    self.dcdF[i, j] = 1./(self.A[j]-2.*self.L[j])*self.dcdn[i,j]
                    if self.L[j] == 0.:
                        self.dcdL[i, j] = 0. + 2.*self.F[j]/((self.A[j]-2.*self.L[j])**2)*self.dcdn[i, j]
                    else:
                        self.dcdL[i, j] = 0.5/np.sqrt(self.L[j]*self.rho[j])*self.dcdbv[i,j] + 2.*self.F[j]/((self.A[j]-2.*self.L[j])**2)*self.dcdn[i, j]
                    self.dcdrl[i, j]= -0.5*self.dcdah[i, j]*np.sqrt(self.A[j]/(self.rho[j]**3)) - 0.5*self.dcdav[i, j]*np.sqrt(self.C[j]/(self.rho[j]**3))\
                                        -0.5*self.dcdbh[i, j]*np.sqrt(self.N[j]/(self.rho[j]**3)) -0.5*self.dcdbv[i, j]*np.sqrt(self.L[j]/(self.rho[j]**3))\
                                            + self.dcdr[i, j]
                    self.Khti[i, j] = self.dcdA[i, j]*self.A[j]/self.L[j]+ self.dcdL[i, j]
                    
                    # self.Khti[i, j] = (0.5/np.sqrt(self.A[j]*self.rho[j]) * self.dcdah[i,j])/ (0.5/np.sqrt(self.L[j]*self.rho[j])*self.dcdbv[i,j])
                    # 0.5/np.sqrt(self.A[j]*self.rho[j]) * self.dcdah[i,j]*self.A[j]/self.L[j] #+ \
                                # 0.5/np.sqrt(self.L[j]*self.rho[j])*self.dcdbv[i,j]
                    
        else:
            for i in xrange(self.nfreq):
                for j in xrange(self.nlay):
                    if self.L[j] == 0.:
                        self.dcdL[i, j] = 0.
                    else:
                        self.dcdL[i, j] = 0.5/np.sqrt(self.L[j]*self.rho[j])*self.dcdbv[i,j]
                    if self.N[j] == 0.:
                        self.dcdN[i, j] = 0.
                    else:
                        self.dcdN[i, j] = 0.5/np.sqrt(self.N[j]*self.rho[j])*self.dcdbh[i,j]
                    self.dcdrl[i, j]= -0.5*self.dcdbh[i,j]*np.sqrt(self.N[j]/(self.rho[j]**3)) \
                                        -0.5*self.dcdbv[i,j]*np.sqrt(self.L[j]/(self.rho[j]**3)) + self.dcdr[i,j]
        return
    
    def eti_perturb(self):
        """
        Compute the phase velocity perturbation from reference to ETI model, use Love kernels
        """
        dA      = self.Aeti - self.A
        dC      = self.Ceti - self.C
        dF      = self.Feti - self.F
        dL      = self.Leti - self.L
        dN      = self.Neti - self.N
        dr      = self.rhoeti - self.rho
        if self.ilvry == 2:
            dpvel   = np.dot(self.dcdA, dA) + np.dot(self.dcdC, dC) + np.dot(self.dcdF, dF)+ np.dot(self.dcdL, dL) \
                        + np.dot(self.dcdrl, dr)
        else:
            dpvel   = np.dot(self.dcdL, dL) + np.dot(self.dcdN, dN)+ np.dot(self.dcdrl, dr)
        return dpvel
    
    def eti_perturb_vel(self):
        """
        Compute the phase velocity perturbation from reference to ETI model, use velocity kernels
        """
        dah     = self.aheti - self.ah
        dav     = self.aveti - self.av
        dbh     = self.bheti - self.bh
        dbv     = self.bveti - self.bv
        dr      = self.reti - self.r
        dn      = self.neti - self.n
        if self.ilvry == 2:
            dpvel   = np.dot(self.dcdah, dah) + np.dot(self.dcdav, dav) + np.dot(self.dcdbv, dbv)+ np.dot(self.dcdn, dn) \
                        + np.dot(self.dcdr, dr)
        else:
            dpvel   = np.dot(self.dcdbv, dbv) + np.dot(self.dcdbh, dbh) + np.dot(self.dcdr, dr) 
        return dpvel
    
    def eti_perturb_old(self):
        """
        Compute the phase velocity perturbation from reference to ETI model
        """
        dA      = self.Aeti - self.A
        dC      = self.Ceti - self.C
        dF      = self.Feti - self.F
        dL      = self.Leti - self.L
        dN      = self.Neti - self.N
        dr      = self.rhoeti - self.rho
        dpvel   = np.zeros(np.int64(self.nfreq), dtype = np.float64)
        if self.ilvry == 2:
            for i in xrange(self.nfreq):
                for j in xrange(self.nlay):
                    dpvel[i]    = dpvel[i] + self.dcdA[i, j] * dA[j] + self.dcdC[i, j] * dC[j] + self.dcdF[i, j] * dF[j]\
                                    + self.dcdL[i, j] * dL[j] 
        else:
            for i in xrange(self.nfreq):
                for j in xrange(self.nlay):
                    dpvel[i]    = dpvel[i] + self.dcdL[i, j] * dL[j] + self.dcdN[i, j] * dN[j] 
        return dpvel
    
    def bottom_padding(self):
        if self.ilvry == 2:
            for i in xrange(self.nfreq):
                self.dcdA[i, -1]    = self.dcdA[i, -2]
                self.dcdC[i, -1]    = self.dcdC[i, -2]
                self.dcdF[i, -1]    = self.dcdF[i, -2]
                self.dcdL[i, -1]    = self.dcdL[i, -2]
                self.dcdrl[i, -1]   = self.dcdrl[i, -2]
        else:
            for i in xrange(self.nfreq):
                self.dcdL[i, -1]    = self.dcdL[i, -2]
                self.dcdN[i, -1]    = self.dcdN[i, -2]
                self.dcdrl[i, -1]   = self.dcdrl[i, -2]
        return
        
    # def aa_perturb(self):
    #     """
    #     Compute the phase velocity perturbation from ETI to AA(azimuthally anisotropic) model
    #     """
    #     az          = np.zeros(360, dtype = np.float64)
    #     for i in xrange(360):
    #         az[i]   = np.float64(i+1)
    #     faz         = np.zeros(360, dtype = np.float64)
    #     Ac2az       = np.zeros(np.int64(self.nfreq), dtype = np.float64)
    #     As2az       = np.zeros(np.int64(self.nfreq), dtype = np.float64)
    #     amp         = np.zeros(np.int64(self.nfreq), dtype = np.float64)
    #     phi         = np.zeros(np.int64(self.nfreq), dtype = np.float64)
    #     if self.ilvry != 2:
    #         raise ValueError('Love wave AA terms computation not supported!')
    #     for i in xrange(self.nfreq):
    #         for j in xrange(self.nlay):
    #             Ac2az[i]    = Ac2az[i] + self.BcArr[j] * self.dcdA[i, j] + self.GcArr[j] * self.dcdL[i, j] + self.HcArr[j] * self.dcdF[i, j]
    #             As2az[i]    = As2az[i] + self.BsArr[j] * self.dcdA[i, j] + self.GsArr[j] * self.dcdL[i, j] + self.HsArr[j] * self.dcdF[i, j]
    #         for k in xrange(360):
    #             faz[k]      = Ac2az[i] * np.cos(2.*az[k]/180.*np.pi) + As2az[i] * np.sin(2.*az[k]/180.*np.pi)
    #         amp[i]      = (faz.max() - faz.min())/2.
    #         indmax      = faz.argmax()
    #         phi[i]      = az[indmax]
    #         if phi[i] >= 180.:
    #             phi[i]  = phi[i] - 180.
    #     return amp, phi

    