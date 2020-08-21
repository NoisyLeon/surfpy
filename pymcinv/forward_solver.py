# -*- coding: utf-8 -*-
"""
Module for forward modelling of 1d models

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""

import surfpy.pymcinv.fast_surf_src.fast_surf as fast_surf
import surfpy.pymcinv.rftheo_src.theo as theo
import surfpy.pymcinv.tdisp96_src.tdisp96 as tdisp96
import surfpy.pymcinv.tregn96_src.tregn96 as tregn96
import surfpy.pymcinv.tlegn96_src.tlegn96 as tlegn96

import surfpy.pymcinv.profilebase as profilebase
import numpy as np
import os
import copy


class forward_vprofile(profilebase.base_vprofile):
    """a class for 1D velocity profile forward modelling
    """
    #==========================================
    # solver for isotropic model
    #==========================================
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
            self.data.dispR.gvelp   = ur0[:self.data.dispR.ngper]
            # replace NaN value with oberved value
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
    
    def compute_rftheo(self, slowness = 0.06, din = None, npts = None):
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
    # solver for VTI model
    #==========================================
    def compute_reference_vti(self, wtype='ray', verbose=0, nmodes=1, cmin=-1., cmax=-1., egn96=True, checkdisp=True, tol=10.):
        """compute (reference) surface wave dispersion of Vertical TI model using tcps
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
                # # # d_in        = self.model.h
                # # # TA_in       = self.model.A
                # # # TC_in       = self.model.C
                # # # TF_in       = self.model.F
                # # # TL_in       = self.model.L
                # # # TN_in       = self.model.N
                # # # TRho_in     = self.model.rho
                
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
                # # # d_in        = self.model.h
                # # # TA_in       = self.model.A
                # # # TC_in       = self.model.C
                # # # TF_in       = self.model.F
                # # # TL_in       = self.model.L
                # # # TN_in       = self.model.N
                # # # TRho_in     = self.model.rho
                
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
        """compute perturbation in dispersion from reference model using sensitivity kernels
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
    
    def compute_disp_vti(self, wtype='both', solver_type=0, verbose=0, \
            nmodes=1, crmin=-1., crmax=-1., clmin=-1., clmax=-1., egn96=True, checkdisp=True, tol=10.):
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
    
    #==========================================
    # solver for HTI model
    #==========================================
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
    
    
    
    