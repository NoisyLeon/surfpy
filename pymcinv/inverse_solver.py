# -*- coding: utf-8 -*-
"""
Module for inversion of 1d models

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""

import surfpy.pymcinv.forward_solver as forward_solver

import numpy as np
from datetime import datetime
import os
import copy
import multiprocessing
from functools import partial
import time
import random
from uncertainties import unumpy

class inverse_vprofile(forward_solver.forward_vprofile):
    """a class for 1D velocity profile inversion
    """
    def copy(self):
        return copy.deepcopy(self)

    def get_misfit(self, mtype='iso', wdisp = 1., rffactor = 40.):
        """
        compute data misfit
        =====================================================================
        ::: input :::
        wdisp       - weight for dispersion curves (0.~1., default - 1.)
        rffactor    - downweighting factor for receiver function
        =====================================================================
        """
        if mtype == 'iso' or mtype == 'isotropic':
            self.data.get_misfit(wdisp = wdisp, rffactor = rffactor)
        elif mtype == 'vti':
            self.data.get_misfit_vti()
        return
    
    #==========================================
    # functions for isotropic inversions
    #==========================================
    
    def mc_joint_inv_iso(self, outdir = './workingdir', dispdtype = 'ph', wdisp = 0.2, rffactor = 40., numbcheck = None,\
            misfit_thresh = 1., isconstrt = True, pfx = 'MC', verbose = False, step4uwalk = 1500, numbrun = 15000,\
            init_run = True, savedata = True):
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
        outmodarr       = np.zeros((numbrun, self.model.isomod.para.npara + 9)) # original
        outdisparr_ph   = np.zeros((numbrun, self.data.dispR.npper))
        outdisparr_gr   = np.zeros((numbrun, self.data.dispR.ngper))
        outrfarr        = np.zeros((numbrun, self.data.rfr.npts))
        #==============
        # initial run
        #==============
        if init_run:
            if wdisp > 0. and wdisp <= 1.:
                self.compute_fsurf()
            if wdisp < 1. and wdisp >= 0.:
                self.compute_rftheo()
            self.get_misfit(wdisp = wdisp, rffactor = rffactor)
            # write initial model
            outmod      = outdir+'/'+pfx+'.mod'
            self.model.write(outfname = outmod, isotropic = True)
            # write initial predicted data
            if wdisp > 0. and wdisp <= 1.:
                if dispdtype is not 'both':
                    outdisp = outdir+'/'+pfx+'.'+dispdtype+'.disp'
                    self.data.dispR.write(outfname = outdisp, dtype = dispdtype)
                else:
                    outdisp = outdir+'/'+pfx+'.ph.disp'
                    self.data.dispR.write(outfname = outdisp, dtype = 'ph')
                    outdisp = outdir+'/'+pfx+'.gr.disp'
                    self.data.dispR.write(outfname = outdisp, dtype = 'gr')
            if wdisp < 1. and wdisp >= 0.:
                outrf       = outdir+'/'+pfx+'.rf'
                self.data.rfr.write(outfname = outrf)
            # convert initial model to para
            self.model.isomod.mod2para()
        else:
            self.model.isomod.mod2para()
            newmod      = self.model.isomod.copy()
            newmod.para.new_paraval(0)
            newmod.para2mod()
            newmod.update()
            # loop to find the "good" model,
            # satisfying the constraint (3), (4) and (5) in Shen et al., 2012
            m0  = 0
            m1  = 1
            # satisfying the constraint (7) in Shen et al., 2012
            if wdisp == 1.:
                g0  = 2
                g1  = 2
            else:
                g0  = 1
                g1  = 0
            if newmod.mtype[0] == 5: # water layer
                m0  += 1
                m1  += 1
                g0  += 1
                g1  += 1
            igood       = 0
            while ( not newmod.isgood(m0, m1, g0, g1)):
                igood   += igood + 1
                newmod  = self.model.isomod.copy()
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
            self.get_misfit(wdisp = wdisp, rffactor = rffactor)
            if verbose:
                print (pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit)
            self.model.isomod.mod2para()
        # likelihood/misfit
        oldL            = self.data.L
        oldmisfit       = self.data.misfit
        run             = True     # the key that controls the sampling
        inew            = 0     # count step (or new paras)
        iacc            = 0     # count acceptance model
        start           = time.time()
        misfitchecked   = False
        while ( run ):
            inew    += 1
            if ( inew > numbrun ):
                break
            #-----------------------------------------
            # checking misfit after numbcheck runs
            #-----------------------------------------
            if (wdisp >= 0. and wdisp <=1.):
                if np.fmod(inew, step4uwalk) > numbcheck and not misfitchecked:
                    ind0            = int(np.floor(inew/step4uwalk)*step4uwalk)
                    ind1            = inew-1
                    temp_min_misfit = outmodarr[ind0:ind1, self.model.isomod.para.npara+3].min()
                    if temp_min_misfit == 0.:
                        raise ValueError('Error!')
                    if temp_min_misfit > misfit_thresh:
                        inew        = int(np.floor(inew/step4uwalk)*step4uwalk) + step4uwalk
                        if inew > numbrun:
                            break
                    misfitchecked   = True
            if (np.fmod(inew, 500) == 0) and verbose:
                print (pfx, 'step =',inew, 'elasped time = %g' %(time.time()-start),' sec')
            #------------------------------------------------------------------------------------------
            # every step4uwalk step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, step4uwalk+1) == step4uwalk and init_run ):
                newmod      = self.model.isomod.copy()
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
                if newmod.mtype[0] == 5: # water layer
                    m0  += 1
                    m1  += 1
                    g0  += 1
                    g1  += 1
                igood       = 0
                while ( not newmod.isgood(m0, m1, g0, g1)):
                    igood   += igood + 1
                    newmod  = self.model.isomod.copy()
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
                self.get_misfit(wdisp = wdisp, rffactor = rffactor)
                oldL                = self.data.L
                oldmisfit           = self.data.misfit
                if verbose:
                    print (pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit)
            #==================================================
            # inversion part
            #==================================================
            #----------------------------------
            # sample the posterior distribution
            #----------------------------------
            if (wdisp >= 0. and wdisp <=1.):
                newmod      = self.model.isomod.copy()
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
                        newmod      = self.model.isomod.copy()
                        newmod.para.new_paraval(1)
                        newmod.para2mod()
                        newmod.update()
                    if not newmod.isgood(m0, m1, g0, g1):
                        print ('!!! No good model found!')
                        continue
                # assign new model to old ones
                oldmod              = self.model.isomod.copy()
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
                    print ('!!! WARNING: '+pfx+', NaN misfit!')
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
                    if dispdtype == 'ph' or dispdtype == 'both' or dispdtype == 'phase':
                        outdisparr_ph[inew-1, :]    = self.data.dispR.pvelp[:]
                    if dispdtype == 'gr' or dispdtype == 'both'or dispdtype == 'group':
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
                newmod      = self.model.isomod.copy()
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
                    if newmod.mtype[0] == 5: # water layer
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
                        print ('!!! No good model found!')
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
        outfname            = outdir+'/mc_inv.'+pfx+'.npz'
        np.savez_compressed(outfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
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
        return
    
    def mc_joint_inv_iso_mp(self, outdir = './workingdir', dispdtype = 'ph', wdisp = 0.2, rffactor = 40., isconstrt = True,
            pfx = 'MC', step4uwalk = 1500, numbrun = 15000, savedata = True, subsize = 1000, nprocess = None, merge = True, \
            Ntotalruns = 10, misfit_thresh = 2.0, Nmodelthresh = 200, verbose = False, verbose2 = False):
        """parallelized version of mc_joint_inv_iso
        ==================================================================================================================
        ::: input :::
        outdir          - output directory
        disptype        - type of dispersion curves (ph/gr/both, default - ph)
        wdisp           - weight of dispersion curve data (0. ~ 1.)
        rffactor        - factor for downweighting the misfit for likelihood computation of rf
        isconstrt       - require monotonical increase in the crust or not
        pfx             - prefix for output( station id OR grid id)
        step4uwalk      - step interval for uniform random walk in the parameter space
        numbrun         - total number of runs
        savedata        - save data to npz binary file or not
        subsize         - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess        - number of process
        merge           - merge data into one single npz file or not
        Ntotalruns      - number of times of total runs, the code would run at most numbrun*Ntotalruns iterations
        misfit_thresh   - threshold misfit value to determine "good" models
        Nmodelthresh    - required number of "good" models
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
            print ('!!! WARNING: number of runs changes: '+str(numbrun)+' --> '+str(Nvpr*step4uwalk))
            numbrun     = Nvpr*step4uwalk
        for i in range(Nvpr):
            temp_vpr            = self.copy()
            temp_vpr.process_id = i
            vpr_lst.append(temp_vpr)
        #----------------------------------------
        # Joint inversion with multiprocessing
        #----------------------------------------
        if verbose:
            print ('[%s] [MC_ISO_INVERSION] %s start' %(datetime.now().isoformat().split('.')[0], pfx))
            stime       = time.time()
        run         = True
        i_totalrun  = 0
        imodels     = 0
        while (run):
            i_totalrun              += 1
            if Nvpr > subsize:
                Nsub                = int(len(vpr_lst)/subsize)
                for isub in range(Nsub):
                    if verbose:
                        print ('[%s] [MC_ISO_INVERSION] subset:' %datetime.now().isoformat().split('.')[0], isub, 'in', Nsub, 'sets')
                    cvpr_lst        = vpr_lst[isub*subsize:(isub+1)*subsize]
                    MCINV           = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, isconstrt=isconstrt,\
                                        pfx=pfx, numbrun=step4uwalk, misfit_thresh=misfit_thresh, verbose=verbose2)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(MCINV, cvpr_lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cvpr_lst            = vpr_lst[(isub+1)*subsize:]
                MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, isconstrt=isconstrt,\
                                        pfx=pfx, numbrun=step4uwalk, misfit_thresh=misfit_thresh, verbose=verbose2)
                pool                = multiprocessing.Pool(processes=nprocess)
                pool.map(MCINV, cvpr_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            else:
                MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, isconstrt=isconstrt,\
                                        pfx=pfx, numbrun=step4uwalk, misfit_thresh=misfit_thresh, verbose=verbose2)
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
                outmodarr           = outmodarr.reshape(int(numbrun), int(outmodarr.size/numbrun))
                outdisparr_ph       = outdisparr_ph.reshape(int(numbrun), int(outdisparr_ph.size/numbrun))
                outdisparr_gr       = outdisparr_gr.reshape(int(numbrun), int(outdisparr_gr.size/numbrun))
                outrfarr            = outrfarr.reshape(int(numbrun), int(outrfarr.size/numbrun))
                ind_valid           = outmodarr[:, 0] == 1.
                imodels             += np.where(outmodarr[ind_valid, temp_vpr.model.isomod.para.npara+3] <= misfit_thresh )[0].size
                if imodels >= Nmodelthresh and i_totalrun == 1:
                    outinvfname     = outdir+'/mc_inv.'+pfx+'.npz'
                    np.savez_compressed(outinvfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
                else:
                    outinvfname     = outdir+'/mc_inv.merged.'+str(i_totalrun)+'.'+pfx+'.npz'
                    np.savez_compressed(outinvfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
                # stop the loop if enough good models are found OR, number of total-runs is equal to the given threhold number
                print ('--- Number of good models = '+str(imodels)+', number of total runs = '+str(i_totalrun))
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
            outmodarr           = outmodarr.reshape(int(Nfinal_total_runs), int(outmodarr.size/Nfinal_total_runs))
            outdisparr_ph       = outdisparr_ph.reshape(int(Nfinal_total_runs), int(outdisparr_ph.size/Nfinal_total_runs))
            outdisparr_gr       = outdisparr_gr.reshape(int(Nfinal_total_runs), int(outdisparr_gr.size/Nfinal_total_runs))
            outrfarr            = outrfarr.reshape(int(Nfinal_total_runs), int(outrfarr.size/Nfinal_total_runs))
            outinvfname         = outdir+'/mc_inv.'+pfx+'.npz'
            np.savez_compressed(outinvfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
        if imodels < Nmodelthresh:
            print ('!!! WARNING: Not enough good models, good models =  '+str(imodels))
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
            etime       = time.time()
            elapsed_time= etime - stime
            print ('[%s] [MC_ISO_INVERSION] %s inversion DONE, elapsed time = %g'\
                   %(datetime.now().isoformat().split('.')[0], pfx, elapsed_time))
        return
    
    #==========================================
    # functions for VTI inversions
    #==========================================
    
    # def mc_joint_inv_vti(self, outdir='./workingdir', run_inv=True, solver_type=1, numbcheck=None, misfit_thresh=1., \
    #             isconstrt=True, pfx='MC', verbose=False, step4uwalk=1500, numbrun=15000, init_run=True, savedata=True, \
    #             depth_mid_crt=-1., iulcrt=2):
    #     """
    #     Bayesian Monte Carlo joint inversion of receiver function and surface wave data for an isotropic model
    #     =================================================================================================================
    #     ::: input :::
    #     outdir          - output directory
    #     run_inv         - run the inversion or not
    #     solver_type     - type of solver
    #                         0   - fast_surf
    #                         1   - tcps
    #     numbcheck       - number of runs that a checking of misfit value should be performed
    #     misfit_thresh   - threshold misfit value for checking
    #     isconstrt       - require model constraints or not
    #     pfx             - prefix for output, typically station id
    #     step4uwalk      - step interval for uniform random walk in the parameter space
    #     numbrun         - total number of runs
    #     init_run        - run and output prediction for inital model or not
    #                     IMPORTANT NOTE: if False, no uniform random walk will perform !
    #     savedata        - save data to npz binary file or not
    #     ---
    #     version history:
    #                 - Added the functionality of stop running if a targe misfit value is not acheived after numbcheck runs
    #                     Sep 27th, 2018
    #     =================================================================================================================
    #     """
    #     if not os.path.isdir(outdir):
    #         os.makedirs(outdir)
    #     if numbcheck is None:
    #         numbcheck   = int(np.ceil(step4uwalk/2.*0.8))
    #     #-------------------------------
    #     # initializations
    #     #-------------------------------
    #     self.get_period()
    #     self.update_mod(mtype = 'vti')
    #     self.get_vmodel(mtype = 'vti', depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
    #     # output arrays
    #     npara           = self.model.vtimod.para.npara
    #     outmodarr       = np.zeros((numbrun, npara+9)) # original
    #     outdisparr_ray  = np.zeros((numbrun, self.data.dispR.npper))
    #     outdisparr_lov  = np.zeros((numbrun, self.data.dispL.npper))
    #     # initial run
    #     if init_run:
    #         self.model.vtimod.mod2para()
    #         self.compute_disp_vti(wtype='both', solver_type = 1)
    #         self.get_misfit(mtype='vti')
    #         # write initial model
    #         outmod      = outdir+'/'+pfx+'.mod'
    #         self.model.write_model(outfname=outmod, isotropic=False)
    #         # write initial predicted data
    #         outdisp = outdir+'/'+pfx+'.ph.ray.disp'
    #         self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
    #         outdisp = outdir+'/'+pfx+'.ph.lov.disp'
    #         self.data.dispL.writedisptxt(outfname=outdisp, dtype='ph')
    #         if solver_type != 0:
    #             while not self.compute_disp_vti(wtype='both', solver_type = 1):
    #                 # # # print 'computing reference'
    #                 self.model.vtimod.new_paraval(ptype = 0)
    #                 self.get_vmodel(mtype = 'vti')
    #             self.get_misfit(mtype='vti')
    #         # convert initial model to para
    #         
    #     else:
    #         self.model.vtimod.mod2para()
    #         self.model.vtimod.new_paraval(ptype = 0)
    #         self.get_vmodel(mtype = 'vti')
    #         # forward computation
    #         if solver_type == 0:
    #             self.compute_disp_vti(wtype='both', solver_type = 0)
    #         else:
    #             while not self.compute_disp_vti(wtype='both', solver_type = 1):
    #                 # # # print 'computing reference'
    #                 self.model.vtimod.new_paraval(ptype = 0)
    #                 self.get_vmodel(mtype = 'vti')
    #         self.get_misfit(mtype='vti')
    #         if verbose:
    #             print pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
    #         self.model.vtimod.mod2para()
    #     # likelihood/misfit
    #     oldL        = self.data.L
    #     oldmisfit   = self.data.misfit
    #     run         = True      # the key that controls the sampling
    #     inew        = 0         # count step (or new paras)
    #     iacc        = 0         # count acceptance model
    #     start       = time.time()
    #     misfitchecked \
    #                 = False
    #     while ( run ):
    #         inew    += 1
    #         if ( inew > numbrun ):
    #             break
    #         #-----------------------------------------
    #         # checking misfit after numbcheck runs
    #         # added Sep 27th, 2018
    #         #-----------------------------------------
    #         if run_inv:
    #             if np.fmod(inew, step4uwalk) > numbcheck and not misfitchecked:
    #                 ind0            = int(np.ceil(inew/step4uwalk)*step4uwalk)
    #                 ind1            = inew-1
    #                 temp_min_misfit = outmodarr[ind0:ind1, npara+3].min()
    #                 if temp_min_misfit == 0.:
    #                     raise ValueError('Error!')
    #                 if temp_min_misfit > misfit_thresh:
    #                     inew        = int(np.ceil(inew/step4uwalk)*step4uwalk) + step4uwalk
    #                     if inew > numbrun:
    #                         break
    #                 misfitchecked   = True
    #         if (np.fmod(inew, 500) == 0) and verbose:
    #             print pfx, 'step =',inew, 'elasped time =', time.time()-start,' sec'
    #         #------------------------------------------------------------------------------------------
    #         # every step4uwalk step, perform a random walk with uniform random value in the paramerter space
    #         #------------------------------------------------------------------------------------------
    #         if ( np.fmod(inew, step4uwalk+1) == step4uwalk and init_run ):
    #             self.model.vtimod.mod2para()
    #             self.model.vtimod.new_paraval(ptype = 0)
    #             self.get_vmodel(mtype = 'vti')
    #             # forward computation
    #             if solver_type == 0:
    #                 self.compute_disp_vti(wtype='both', solver_type = 0)
    #             else:
    #                 while not self.compute_disp_vti(wtype='both', solver_type = 1):
    #                     self.model.vtimod.new_paraval(ptype = 0)
    #                     self.get_vmodel(mtype = 'vti')
    #             self.get_misfit(mtype='vti')
    #             oldL                = self.data.L
    #             oldmisfit           = self.data.misfit
    #             if verbose:
    #                 print pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
    #         #==================================================
    #         # inversion part
    #         #==================================================
    #         #----------------------------------
    #         # sample the posterior distribution
    #         #----------------------------------
    #         if run_inv:
    #             self.model.vtimod.mod2para()
    #             oldmod      = copy.deepcopy(self.model.vtimod)
    #             if not self.model.vtimod.new_paraval(ptype = 1):
    #                 print 'No good model found!'
    #                 continue
    #             self.get_vmodel(mtype = 'vti')
    #             #--------------------------------
    #             # forward computation
    #             #--------------------------------
    #             is_large_perturb    = False
    #             if solver_type == 0:
    #                 self.compute_disp_vti(wtype='both', solver_type = 0)
    #             else:
    #                 # compute dispersion curves based on sensitivity kernels
    #                 self.compute_disp_vti(wtype='both', solver_type = 2)
    #                 is_large_perturb= (self.data.dispR.check_large_perturb() or self.data.dispL.check_large_perturb())
    #             self.get_misfit(mtype='vti')
    #             newL                = self.data.L
    #             newmisfit           = self.data.misfit
    #             # reject model if NaN misfit 
    #             if np.isnan(newmisfit):
    #                 print 'WARNING: '+pfx+', NaN misfit!'
    #                 outmodarr[inew-1, 0]                = -1 # index for acceptance
    #                 outmodarr[inew-1, 1]                = iacc
    #                 outmodarr[inew-1, 2:(npara+2)]      = self.model.vtimod.para.paraval[:]
    #                 outmodarr[inew-1, npara+2]          = 0.
    #                 outmodarr[inew-1, npara+3]          = 9999.
    #                 outmodarr[inew-1, npara+4]          = self.data.dispR.L
    #                 outmodarr[inew-1, npara+5]          = self.data.dispR.misfit
    #                 outmodarr[inew-1, npara+6]          = self.data.dispL.L
    #                 outmodarr[inew-1, npara+7]          = self.data.dispL.misfit
    #                 outmodarr[inew-1, npara+8]          = time.time()-start
    #                 self.model.vtimod                   = oldmod
    #                 continue
    #             if newL < oldL:
    #                 prob    = (oldL-newL)/oldL
    #                 rnumb   = random.random()
    #                 # reject the model
    #                 if rnumb < prob:
    #                     outmodarr[inew-1, 0]            = -1 # index for acceptance
    #                     outmodarr[inew-1, 1]            = iacc
    #                     outmodarr[inew-1, 2:(npara+2)]  = self.model.vtimod.para.paraval[:]
    #                     outmodarr[inew-1, npara+2]      = newL
    #                     outmodarr[inew-1, npara+3]      = newmisfit
    #                     outmodarr[inew-1, npara+4]      = self.data.dispR.L
    #                     outmodarr[inew-1, npara+5]      = self.data.dispR.misfit
    #                     outmodarr[inew-1, npara+6]      = self.data.dispL.L
    #                     outmodarr[inew-1, npara+7]      = self.data.dispL.misfit
    #                     outmodarr[inew-1, npara+8]      = time.time()-start
    #                     self.model.vtimod               = oldmod
    #                     continue
    #             # update the kernels for the new reference model
    #             if is_large_perturb and solver_type == 1:
    #                 # # # print 'Update reference!'
    #                 oldvpr                              = copy.deepcopy(self)
    #                 if not self.compute_disp_vti(wtype='both', solver_type = 1):
    #                     self                            = oldvpr # reverse to be original vpr with old kernels
    #                     outmodarr[inew-1, 0]            = -1 # index for acceptance
    #                     outmodarr[inew-1, 1]            = iacc
    #                     outmodarr[inew-1, 2:(npara+2)]  = self.model.vtimod.para.paraval[:]
    #                     outmodarr[inew-1, npara+2]      = newL
    #                     outmodarr[inew-1, npara+3]      = newmisfit
    #                     outmodarr[inew-1, npara+4]      = self.data.dispR.L
    #                     outmodarr[inew-1, npara+5]      = self.data.dispR.misfit
    #                     outmodarr[inew-1, npara+6]      = self.data.dispL.L
    #                     outmodarr[inew-1, npara+7]      = self.data.dispL.misfit
    #                     outmodarr[inew-1, npara+8]      = time.time()-start
    #                     self.model.vtimod               = oldmod
    #                     continue
    #                 self.get_misfit(mtype='vti')
    #                 newL                                = self.data.L
    #                 newmisfit                           = self.data.misfit
    #             # accept the new model
    #             outmodarr[inew-1, 0]                    = 1 # index for acceptance
    #             outmodarr[inew-1, 1]                    = iacc
    #             outmodarr[inew-1, 2:(npara+2)]          = self.model.vtimod.para.paraval[:]
    #             outmodarr[inew-1, npara+2]              = newL
    #             outmodarr[inew-1, npara+3]              = newmisfit
    #             outmodarr[inew-1, npara+4]              = self.data.dispR.L
    #             outmodarr[inew-1, npara+5]              = self.data.dispR.misfit
    #             outmodarr[inew-1, npara+6]              = self.data.dispL.L
    #             outmodarr[inew-1, npara+7]              = self.data.dispL.misfit
    #             outmodarr[inew-1, npara+8]              = time.time()-start
    #             # predicted dispersion data
    #             outdisparr_ray[inew-1, :]               = self.data.dispR.pvelp[:]
    #             outdisparr_lov[inew-1, :]               = self.data.dispL.pvelp[:]
    #             # assign likelihood/misfit
    #             oldL        = newL
    #             oldmisfit   = newmisfit
    #             iacc        += 1
    #             # # # print inew, oldmisfit
    #             continue
    #         #----------------------------------
    #         # sample the prior distribution
    #         #----------------------------------
    #         else:
    #             self.model.vtimod.new_paraval(ptype = 0, isconstrt=isconstrt)
    #             # accept the new model
    #             outmodarr[inew-1, 0]                    = 1 # index for acceptance
    #             outmodarr[inew-1, 1]                    = iacc
    #             outmodarr[inew-1, 2:(npara+2)]          = self.model.vtimod.para.paraval[:]
    #             outmodarr[inew-1, npara+2]              = 1.
    #             outmodarr[inew-1, npara+3]              = 0
    #             outmodarr[inew-1, npara+4]              = self.data.dispR.L
    #             outmodarr[inew-1, npara+5]              = self.data.dispR.misfit
    #             outmodarr[inew-1, npara+6]              = self.data.dispL.L
    #             outmodarr[inew-1, npara+7]              = self.data.dispL.misfit
    #             outmodarr[inew-1, npara+8]              = time.time() - start
    #             continue
    #     #-----------------------------------
    #     # write results to binary npz files
    #     #-----------------------------------
    #     outfname    = outdir+'/mc_inv.'+pfx+'.npz'
    #     np.savez_compressed(outfname, outmodarr, outdisparr_ray, outdisparr_lov)
    #     if savedata:
    #         outdatafname\
    #                 = outdir+'/mc_data.'+pfx+'.npz'
    #         np.savez_compressed(outdatafname, self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
    #                     self.data.dispL.pper, self.data.dispL.pvelo, self.data.dispL.stdpvelo)
    #     del outmodarr
    #     del outdisparr_ray
    #     del outdisparr_lov
    #     return
    # 
    # def mc_joint_inv_vti_mp(self, outdir='./workingdir', run_inv=True, solver_type=1, isconstrt=True, pfx='MC',\
    #             verbose=False, step4uwalk=1500, numbrun=15000, savedata=True, subsize=1000,
    #             nprocess=None, merge=True, Ntotalruns=2, misfit_thresh=2.0, Nmodelthresh=200, depth_mid_crt=-1., iulcrt=2):
    #     """
    #     Parallelized version of mc_joint_inv_iso
    #     ==================================================================================================================
    #     ::: input :::
    #     outdir          - output directory
    #     run_inv         - run the inversion or not
    #     solver_type     - type of solver
    #                         0   - fast_surf
    #                         1   - tcps
    #     isconstrt       - require monotonical increase in the crust or not
    #     pfx             - prefix for output, typically station id
    #     step4uwalk      - step interval for uniform random walk in the parameter space
    #     numbrun         - total number of runs
    #     savedata        - save data to npz binary file or not
    #     subsize         - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
    #     nprocess        - number of process
    #     merge           - merge data into one single npz file or not
    #     Ntotalruns      - number of times of total runs, the code would run at most numbrun*Ntotalruns iterations
    #     misfit_thresh   - threshold misfit value to determine "good" models
    #     Nmodelthresh    - required number of "good" models
    #     ---
    #     version history:
    #                 - Added the functionality of adding addtional runs if not enough good models found, Sep 27th, 2018
    #     ==================================================================================================================
    #     """
    #     if not os.path.isdir(outdir):
    #         os.makedirs(outdir)
    #     #-------------------------
    #     # prepare data
    #     #-------------------------
    #     vpr_lst = []
    #     Nvpr    = int(numbrun/step4uwalk)
    #     npara   = self.model.vtimod.para.npara
    #     if Nvpr*step4uwalk != numbrun:
    #         print 'WARNING: number of runs changes: '+str(numbrun)+' --> '+str(Nvpr*step4uwalk)
    #         numbrun     = Nvpr*step4uwalk
    #     for i in range(Nvpr):
    #         temp_vpr            = copy.deepcopy(self)
    #         temp_vpr.process_id = i
    #         vpr_lst.append(temp_vpr)
    #     #----------------------------------------
    #     # Joint inversion with multiprocessing
    #     #----------------------------------------
    #     if verbose:
    #         print 'Start MC inversion: '+pfx+' '+time.ctime()
    #         stime       = time.time()
    #     run             = True
    #     i_totalrun      = 0
    #     imodels         = 0
    #     need_to_merge   = False
    #     while (run):
    #         i_totalrun              += 1
    #         if Nvpr > subsize:
    #             Nsub                = int(len(vpr_lst)/subsize)
    #             for isub in xrange(Nsub):
    #                 print 'Subset:', isub,'in',Nsub,'sets'
    #                 cvpr_lst        = vpr_lst[isub*subsize:(isub+1)*subsize]
    #                 MCINV           = partial(mc4mp_vti, outdir=outdir, run_inv=run_inv, solver_type=solver_type,
    #                                     isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk, misfit_thresh=misfit_thresh, \
    #                                     depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
    #                 pool            = multiprocessing.Pool(processes=nprocess)
    #                 pool.map(MCINV, cvpr_lst) #make our results with a map call
    #                 pool.close() #we are not adding any more processes
    #                 pool.join() #tell it to wait until all threads are done before going on
    #             cvpr_lst            = vpr_lst[(isub+1)*subsize:]
    #             MCINV               = partial(mc4mp_vti, outdir=outdir, run_inv=run_inv, solver_type=solver_type,
    #                                     isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk, misfit_thresh=misfit_thresh, \
    #                                     depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
    #             pool                = multiprocessing.Pool(processes=nprocess)
    #             pool.map(MCINV, cvpr_lst) #make our results with a map call
    #             pool.close() #we are not adding any more processes
    #             pool.join() #tell it to wait until all threads are done before going on
    #         else:
    #             MCINV               = partial(mc4mp_vti, outdir=outdir, run_inv=run_inv, solver_type=solver_type,
    #                                     isconstrt=isconstrt, pfx=pfx, verbose=verbose, numbrun=step4uwalk, misfit_thresh=misfit_thresh, \
    #                                     depth_mid_crt=depth_mid_crt, iulcrt=iulcrt)
    #             pool                = multiprocessing.Pool(processes=nprocess)
    #             pool.map(MCINV, vpr_lst) #make our results with a map call
    #             pool.close() #we are not adding any more processes
    #             pool.join() #tell it to wait until all threads are done before going on
    #         #----------------------------------------
    #         # Merge inversion results for each process
    #         #----------------------------------------
    #         if merge:
    #             outmodarr           = np.array([])
    #             outdisparr_ray      = np.array([])
    #             outdisparr_lov      = np.array([])
    #             for i in range(Nvpr):
    #                 invfname        = outdir+'/mc_inv.'+pfx+'_'+str(i)+'.npz'
    #                 inarr           = np.load(invfname)
    #                 outmodarr       = np.append(outmodarr, inarr['arr_0'])
    #                 outdisparr_ray  = np.append(outdisparr_ray, inarr['arr_1'])
    #                 outdisparr_lov  = np.append(outdisparr_lov, inarr['arr_2'])
    #                 os.remove(invfname)
    #             outmodarr           = outmodarr.reshape(numbrun, outmodarr.size/numbrun)
    #             outdisparr_ray      = outdisparr_ray.reshape(numbrun, outdisparr_ray.size/numbrun)
    #             outdisparr_lov      = outdisparr_lov.reshape(numbrun, outdisparr_lov.size/numbrun)
    #             # added Sep 27th, 2018
    #             ind_valid           = outmodarr[:, 0] == 1.
    #             imodels             += np.where(outmodarr[ind_valid, npara+3] <= misfit_thresh )[0].size
    #             if imodels >= Nmodelthresh and i_totalrun == 1:
    #                 outinvfname     = outdir+'/mc_inv.'+pfx+'.npz'
    #                 np.savez_compressed(outinvfname, outmodarr, outdisparr_ray, outdisparr_lov)
    #             else:
    #                 outinvfname     = outdir+'/mc_inv.merged.'+str(i_totalrun)+'.'+pfx+'.npz'
    #                 np.savez_compressed(outinvfname, outmodarr, outdisparr_ray, outdisparr_lov)
    #                 need_to_merge   = True
    #             # stop the loop if enough good models are found OR, number of total-runs is equal to the given threhold number
    #             print '== Number of good models = '+str(imodels)+', number of total runs = '+str(i_totalrun)
    #             if imodels >= Nmodelthresh or i_totalrun >= Ntotalruns:
    #                 break
    #     #--------------------------------------------------------
    #     # Merge inversion results for each additional total runs
    #     #--------------------------------------------------------
    #     if need_to_merge:
    #         outmodarr           = np.array([])
    #         outdisparr_ray      = np.array([])
    #         outdisparr_lov      = np.array([])
    #         for i in range(i_totalrun):
    #             invfname        = outdir+'/mc_inv.merged.'+str(i+1)+'.'+pfx+'.npz'
    #             inarr           = np.load(invfname)
    #             outmodarr       = np.append(outmodarr, inarr['arr_0'])
    #             outdisparr_ray  = np.append(outdisparr_ray, inarr['arr_1'])
    #             outdisparr_lov  = np.append(outdisparr_lov, inarr['arr_2'])
    #             os.remove(invfname)
    #         Nfinal_total_runs   = i_totalrun*numbrun
    #         outmodarr           = outmodarr.reshape(Nfinal_total_runs, outmodarr.size/Nfinal_total_runs)
    #         outdisparr_ray      = outdisparr_ray.reshape(Nfinal_total_runs, outdisparr_ray.size/Nfinal_total_runs)
    #         outdisparr_lov      = outdisparr_lov.reshape(Nfinal_total_runs, outdisparr_lov.size/Nfinal_total_runs)
    #         outinvfname         = outdir+'/mc_inv.'+pfx+'.npz'
    #         np.savez_compressed(outinvfname, outmodarr, outdisparr_ray, outdisparr_lov)
    #     if imodels < Nmodelthresh:
    #         print 'WARNING: Not enough good models, '+str(imodels)
    #     #----------------------------------------
    #     # save data
    #     #----------------------------------------
    #     if savedata:
    #         outdatafname\
    #                 = outdir+'/mc_data.'+pfx+'.npz'
    #         np.savez_compressed(outdatafname, self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
    #                     self.data.dispL.pper, self.data.dispL.pvelo, self.data.dispL.stdpvelo)
    #     if verbose:
    #         print 'End MC inversion: '+pfx+' '+time.ctime()
    #         etime   = time.time()
    #         print 'Elapsed time: '+str(etime-stime)+' secs'
    #     return
    # 
    #==========================================
    # functions for HTI inversions
    #==========================================
    # def linear_inv_hti(self, isBcs=True, useref=False, depth_mid_crust=15., depth_mid_mantle=-1., usespl_man=False):
    #     # construct data array
    #     dc      = np.zeros(self.data.dispR.npper, dtype=np.float64)
    #     ds      = np.zeros(self.data.dispR.npper, dtype=np.float64)
    #     if useref:
    #         try:
    #             A2      = self.data.dispR.amp/100.*self.data.dispR.pvelref
    #             unA2    = self.data.dispR.unamp/100.*self.data.dispR.pvelref
    #             vel_iso = self.data.dispR.pvelref
    #         except:
    #             raise ValueError('No refernce dispersion curve stored!')
    #     else:
    #         A2      = self.data.dispR.amp/100.*self.data.dispR.pvelo
    #         unA2    = self.data.dispR.unamp/100.*self.data.dispR.pvelo
    #         vel_iso = self.data.dispR.pvelo
    #     dc[:]       = A2*np.cos(2. * (self.data.dispR.psi2/180.*np.pi) )
    #     ds[:]       = A2*np.sin(2. * (self.data.dispR.psi2/180.*np.pi) )
    #     #--------------------------
    #     # data covariance matrix
    #     #--------------------------
    #     A2_with_un  = unumpy.uarray(A2, unA2)
    #     psi2_with_un= unumpy.uarray(self.data.dispR.psi2, self.data.dispR.unpsi2)
    #     # dc
    #     Cdc         = np.zeros((self.data.dispR.npper, self.data.dispR.npper), dtype=np.float64)
    #     undc        = unumpy.std_devs( A2_with_un * unumpy.cos(2. * (psi2_with_un/180.*np.pi)) )
    #     np.fill_diagonal(Cdc, undc**2)
    #     # ds
    #     Cds         = np.zeros((self.data.dispR.npper, self.data.dispR.npper), dtype=np.float64)
    #     unds        = unumpy.std_devs( A2_with_un * unumpy.sin(2. * (psi2_with_un/180.*np.pi)) )
    #     np.fill_diagonal(Cds, unds**2)
    #     #--------------------------
    #     # forward operator matrix
    #     #--------------------------
    #     nmod        = 2
    #     if depth_mid_crust > 0.:
    #         nmod    += 1
    #     if depth_mid_mantle > 0.:
    #         nmod    += 1
    #     self.model.htimod.init_arr(nmod)
    #     self.model.htimod.set_depth_disontinuity(depth_mid_crust=depth_mid_crust, depth_mid_mantle=depth_mid_mantle)
    #     self.model.get_hti_layer_ind()
    #     # if usespl_man:
    #         
    #     # forward matrix
    #     G           = np.zeros((self.data.dispR.npper, nmod), dtype=np.float64)
    #     for i in range(nmod):
    #         ind0    = self.model.htimod.layer_ind[i, 0]
    #         ind1    = self.model.htimod.layer_ind[i, 1]
    #         dcdX    = self.eigkR.dcdL[:, ind0:ind1]
    #         if isBcs:
    #             dcdX+= self.eigkR.dcdA[:, ind0:ind1] * self.eigkR.Aeti[ind0:ind1]/self.eigkR.Leti[ind0:ind1]
    #         dcdX    *= self.eigkR.Leti[ind0:ind1]
    #         G[:, i] = dcdX.sum(axis=1)
    #     #--------------------------
    #     # solve the inverse problem
    #     #--------------------------
    #     # cosine terms
    #     Ginv1                       = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cdc)), G) )
    #     Ginv2                       = np.dot( np.dot(G.T, np.linalg.inv(Cdc)), dc)
    #     modelC                      = np.dot(Ginv1, Ginv2)
    #     Cmc                         = Ginv1 # model covariance matrix
    #     pcovc                       = np.sqrt(np.absolute(Cmc))
    #     self.model.htimod.Gc[:]     = modelC[:]
    #     self.model.htimod.unGc[:]   = pcovc.diagonal()
    #     # sine terms
    #     Ginv1                       = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cds)), G) )
    #     Ginv2                       = np.dot( np.dot(G.T, np.linalg.inv(Cds)), ds)
    #     modelS                      = np.dot(Ginv1, Ginv2)
    #     Cms                         = Ginv1 # model covariance matrix
    #     pcovs                       = np.sqrt(np.absolute(Cms))
    #     self.model.htimod.Gs[:]     = modelS[:]
    #     self.model.htimod.unGs[:]   = pcovs.diagonal()
    #     self.model.htimod.GcGs_to_azi()
    #     #--------------------------
    #     # predictions
    #     #--------------------------
    #     pre_dc                  = np.dot(G, self.model.htimod.Gc)
    #     pre_ds                  = np.dot(G, self.model.htimod.Gs)
    #     pre_amp                 = np.sqrt(pre_dc**2 + pre_ds**2)
    #     pre_amp                 = pre_amp/vel_iso*100.
    #     self.data.dispR.pamp    = pre_amp
    #     pre_psi                 = np.arctan2(pre_ds, pre_dc)/2./np.pi*180.
    #     pre_psi[pre_psi<0.]     += 180.
    #     self.data.dispR.ppsi2   = pre_psi
    #     self.data.get_misfit_hti()
    #     return
    # 
    # def linear_inv_hti_twolayer(self, depth=-2., isBcs=True, useref=False, maxdepth=-3.,\
    #                             depth2d = np.array([])):
    #     # construct data array
    #     dc      = np.zeros(self.data.dispR.npper, dtype=np.float64)
    #     ds      = np.zeros(self.data.dispR.npper, dtype=np.float64)
    #     if useref:
    #         try:
    #             A2      = self.data.dispR.amp/100.*self.data.dispR.pvelref
    #             unA2    = self.data.dispR.unamp/100.*self.data.dispR.pvelref
    #             vel_iso = self.data.dispR.pvelref
    #         except:
    #             raise ValueError('No refernce dispersion curve stored!')
    #     else:
    #         A2      = self.data.dispR.amp/100.*self.data.dispR.pvelo
    #         unA2    = self.data.dispR.unamp/100.*self.data.dispR.pvelo
    #         vel_iso = self.data.dispR.pvelo
    #     dc[:]       = A2*np.cos(2. * (self.data.dispR.psi2/180.*np.pi) )
    #     ds[:]       = A2*np.sin(2. * (self.data.dispR.psi2/180.*np.pi) )
    #     #--------------------------
    #     # data covariance matrix
    #     #--------------------------
    #     A2_with_un  = unumpy.uarray(A2, unA2)
    #     psi2_with_un= unumpy.uarray(self.data.dispR.psi2, self.data.dispR.unpsi2)
    #     # dc
    #     Cdc         = np.zeros((self.data.dispR.npper, self.data.dispR.npper), dtype=np.float64)
    #     undc        = unumpy.std_devs( A2_with_un * unumpy.cos(2. * (psi2_with_un/180.*np.pi)) )
    #     np.fill_diagonal(Cdc, undc**2)
    #     # ds
    #     Cds         = np.zeros((self.data.dispR.npper, self.data.dispR.npper), dtype=np.float64)
    #     unds        = unumpy.std_devs( A2_with_un * unumpy.sin(2. * (psi2_with_un/180.*np.pi)) )
    #     np.fill_diagonal(Cds, unds**2)
    #     #--------------------------
    #     # forward operator matrix
    #     #--------------------------
    #     nmod        = 2
    #     self.model.htimod.init_arr(nmod)
    #     if depth2d.shape[0] != nmod:       
    #         self.model.htimod.depth[0]  = -1
    #         self.model.htimod.depth[1]  = depth
    #         self.model.htimod.depth[2]  = maxdepth
    #         self.model.get_hti_layer_ind()
    #     else:
    #         self.model.htimod.depth2d[:, :] = depth2d.copy()
    #         if self.model.htimod.depth2d[1, 1]  == -3.:
    #             self.model.htimod.depth2d[1, 1] = maxdepth
    #         # # # self.model.htimod.depth2d[0, 0] = -1
    #         # # # self.model.htimod.depth2d[0, 1] = 15.
    #         # # # self.model.htimod.depth2d[1, 0] = depth
    #         # # # self.model.htimod.depth2d[1, 1] = maxdepth
    #         self.model.get_hti_layer_ind_2d()
    #     # forward matrix
    #     G           = np.zeros((self.data.dispR.npper, nmod), dtype=np.float64)
    #     for i in range(nmod):
    #         ind0    = self.model.htimod.layer_ind[i, 0]
    #         ind1    = self.model.htimod.layer_ind[i, 1]
    #         dcdX    = self.eigkR.dcdL[:, ind0:ind1]
    #         if isBcs:
    #             dcdX+= self.eigkR.dcdA[:, ind0:ind1] * self.eigkR.Aeti[ind0:ind1]/self.eigkR.Leti[ind0:ind1]
    #         dcdX    *= self.eigkR.Leti[ind0:ind1]
    #         G[:, i] = dcdX.sum(axis=1)
    #     #--------------------------
    #     # solve the inverse problem
    #     #--------------------------
    #     # cosine terms
    #     Ginv1                       = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cdc)), G) )
    #     Ginv2                       = np.dot( np.dot(G.T, np.linalg.inv(Cdc)), dc)
    #     modelC                      = np.dot(Ginv1, Ginv2)
    #     Cmc                         = Ginv1 # model covariance matrix
    #     pcovc                       = np.sqrt(np.absolute(Cmc))
    #     self.model.htimod.Gc[:]     = modelC[:]
    #     self.model.htimod.unGc[:]   = pcovc.diagonal()
    #     # sine terms
    #     Ginv1                       = np.linalg.inv( np.dot( np.dot(G.T, np.linalg.inv(Cds)), G) )
    #     Ginv2                       = np.dot( np.dot(G.T, np.linalg.inv(Cds)), ds)
    #     modelS                      = np.dot(Ginv1, Ginv2)
    #     Cms                         = Ginv1 # model covariance matrix
    #     pcovs                       = np.sqrt(np.absolute(Cms))
    #     self.model.htimod.Gs[:]     = modelS[:]
    #     self.model.htimod.unGs[:]   = pcovs.diagonal()
    #     self.model.htimod.GcGs_to_azi()
    #     #--------------------------
    #     # predictions
    #     #--------------------------
    #     pre_dc                  = np.dot(G, self.model.htimod.Gc)
    #     pre_ds                  = np.dot(G, self.model.htimod.Gs)
    #     pre_amp                 = np.sqrt(pre_dc**2 + pre_ds**2)
    #     pre_amp                 = pre_amp/vel_iso*100.
    #     self.data.dispR.pamp    = pre_amp
    #     pre_psi                 = np.arctan2(pre_ds, pre_dc)/2./np.pi*180.
    #     pre_psi[pre_psi<0.]     += 180.
    #     self.data.dispR.ppsi2   = pre_psi
    #     self.data.get_misfit_hti()
    #     return
    # 
def mc4mp(invpr, outdir, dispdtype, wdisp, rffactor, isconstrt, pfx, numbrun, misfit_thresh, verbose):
    if verbose:
        print ('[%s] [MC_ISO_INVERSION] station/grid: ' %datetime.now().isoformat().split('.')[0]\
               +pfx+', process id: '+str(invpr.process_id))
    pfx     = pfx +'_'+str(invpr.process_id)
    if invpr.process_id == 0 or wdisp < 0.:
        invpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, misfit_thresh=misfit_thresh, \
            isconstrt=isconstrt, pfx=pfx, verbose=verbose, step4uwalk=numbrun, numbrun=numbrun, init_run=True, savedata=False)
    else:
        invpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor, misfit_thresh=misfit_thresh, \
            isconstrt=isconstrt, pfx=pfx, verbose=verbose, step4uwalk=numbrun, numbrun=numbrun, init_run=False, savedata=False)
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