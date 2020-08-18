# -*- coding: utf-8 -*-
# distutils: language=c++
"""
Module for handling input data for Bayesian Monte Carlo inversion

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.jit(numba.float64(numba.float64[:], numba.int64), nopython = True)
def _fast_compute_expect_misfit(stdarr, N):
    temp    = 0
    Ndata   = stdarr.size
    for i in range(N):
        temp_misfit = np.sqrt(((np.random.normal(scale=stdarr))**2/stdarr**2).sum()/Ndata)
        temp        +=temp_misfit
    expected_misfit = temp/N
    return expected_misfit

class rf(object):
    """
    A class for handling receiver function data and computing misfit
    ==========================================================================
    ::: parameters :::
    fs      - sampling rate
    npts    - number of data points
    rfo     - observed receiver function array
    to      - time array of observed receiver function
    stdrfo  - uncerntainties in observed receiver function
    tp      - time array of predicted receiver function
    rfp     - predicted receiver function array
    misfit  - misfit value
    L       - likelihood value
    ==========================================================================
    """
    def __init__(self):
        self.npts   = 0
        self.fs     = 0.
        return
    
    def read(self, infname):
        """
        read input txt file of receiver function
        ==========================================================================
        ::: input :::
        infname     - input file name
        ::: output :::
        receiver function data is stored in self
        ==========================================================================
        """
        if self.npts > 0:
            print ('*** receiver function data is already stored!')
            return False
        inarr 		    = np.loadtxt(infname, dtype=np.float64)
        self.npts       = inarr.shape[0]        
        self.to         = inarr[:,0]
        self.rfo        = inarr[:,1]
        try:
            self.stdrfo = inarr[:,2]
        except IndexError:
            self.stdrfo = np.ones(self.npts, dtype = np.float64)*0.1
        self.fs         = 1./(self.to[1] - self.to[0])
        return True
    
    def get_rf(self, indata):
        """
        get input receiver function data
        ==========================================================================
        ::: input :::
        indata      - input data array (3, N)
        ::: output :::
        receiver function data is stored in self
        ==========================================================================
        """
        if self.npts > 0:
            print ('*** receiver function data is already stored!')
            return False
        self.npts       = indata.shape[1]        
        self.to         = indata[0, :]
        self.rfo        = indata[1, :]
        try:
            self.stdrfo = indata[2, :]
        except IndexError:
            self.stdrfo = np.ones(self.npts, dtype = np.float64)*0.1
        self.fs         = 1./(self.to[1] - self.to[0])
        return True
  
    def write(self, outfname, tf = 10., prerf = True, obsrf = True):
        """
        Write receiver function data to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        tf          - end time point for trim
        prerf       - write predicted rf or not
        obsrf       - write observed rf or not
        ::: output :::
        a txt file contains predicted and observed receiver function data
        ==========================================================================
        """
        if self.npts == 0:
            print ('*** receiver function data is not stored!')
            return False
        nout    = int(self.fs*tf)+1
        nout    = min(nout, self.npts)
        if prerf:
            outarr  = np.append(self.tp[:nout], self.rfp[:nout])
        else:
            outarr  = np.array([])
        if obsrf:
            outarr  = np.append(outarr, self.to[:nout])
            outarr  = np.append(outarr, self.rfo[:nout])
            outarr  = np.append(outarr, self.stdrfo[:nout])
        Ncolumn     = 0
        if prerf:
            Ncolumn += 2
        if obsrf:
            Ncolumn += 3
        outarr  = outarr.reshape((Ncolumn, nout))    
        outarr  = outarr.T
        if Ncolumn == 5:
            header  = 'tp rfp to rfo stdrfo'
        elif Ncolumn == 2:
            header  = 'tp rfp'
        elif Ncolumn == 3:
            header  = 'to rfo stdrfo'
        np.savetxt(outfname, outarr, fmt='%g', header = header)
        return True

    def get_misfit(self, factor = 40.):
        """
        Compute misfit for receiver function
        ==============================================================================
        ::: input :::
        factor  - factor for downweighting the misfit for likelihood computation
        ==============================================================================
        """
        if self.npts == 0:
            self.misfit = 0.
            self.L      = 1.
            return False
        if not np.allclose(self.to, self.tp):
            raise ValueError('Incompatable time arrays for predicted and observed rf!')
        ind         = (self.to<10.)*(self.to>=0.)
        temp        = ((self.rfo[ind] - self.rfp[ind])**2 / (self.stdrfo[ind]**2)).sum()
        k           = (self.rfo[ind]).size
        self.misfit = np.sqrt(temp/k)
        tS          = temp/factor
        if tS > 50.:
            tS      = np.sqrt(tS*50.)
        self.L      = np.exp(-0.5 * tS)
        return True
    
    def plot(self, showfig = True, prediction = False):
        if self.npts == 0:
            print ('No data for plotting!')
            return
        plt.figure()
        ax  = plt.subplot()
        plt.errorbar(self.to, self.rfo, yerr = self.stdrfo, lw = 1)
        if prediction:
            plt.plot(self.tp, self.rfp, 'r-', lw=3)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('time (sec)', fontsize=30)
        plt.ylabel('amplitude', fontsize=30)
        plt.title('receiver function', fontsize=30)
        if showfig:
            plt.show()
    
    
class disp(object):
    """
    A class for handling dispersion data and computing misfit
    ==========================================================================
    ::: parameters :::
    --------------------------------------------------------------------------
    ::  phase   ::
    :   isotropic   :
    npper   - number of phase period
    pper    - phase period array
    pvelo   - observed phase velocities
    stdpvelo- uncertainties for observed phase velocities
    pvelp   - predicted phase velocities
    :   anisotropic :
    pphio   - observed phase velocity fast direction angle
    pampo   - observed phase velocity azimuthal anisotropic amplitude
    stdpphio- uncertainties for fast direction angle
    stdpampo- uncertainties for azimuthal anisotropic amplitude
    pphip   - predicted phase velocity fast direction angle
    pampp   - predicted phase velocity azimuthal anisotropic amplitude
    :   others  :
    isphase - phase dispersion data is stored or not
    pmisfit - phase dispersion misfit
    pL      - phase dispersion likelihood
    pS      - S function, L = exp(-0.5*S)
    --------------------------------------------------------------------------
    ::  group   ::
    ngper   - number of group period
    gper    - group period array
    gvelo   - observed group velocities
    stdgvelo- uncertainties for observed group velocities
    gvelp   - predicted group velocities
    :   others  :
    isgroup - group dispersion data is stored or not
    gmisfit - group dispersion misfit
    gL      - group dispersion likelihood
    --------------------------------------------------------------------------
    ::  others  ::
    misfit  - total misfit
    L       - total likelihood
    period  - common period array
    nper    - common number of periods
    ==========================================================================
    """
    def __init__(self):
        self.npper  = 0
        self.ngper  = 0
        self.nper   = 0
        self.isphase= False
        self.isgroup= False
        return
    #----------------------------------------------------
    # I/O functions
    #----------------------------------------------------
    
    def read(self, infname, dtype = 'ph'):
        """
        Read input txt file of dispersion curve
        ==========================================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (phase/group)
        ::: output :::
        dispersion curve is stored
        ==========================================================================
        """
        dtype                   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if self.isphase:
                print ('*** phase velocity data is already stored!')
                return False
            inarr 	            = np.loadtxt(infname, dtype = np.float64)
            self.pper           = inarr[:,0]
            self.pvelo          = inarr[:,1]
            self.npper          = self.pper.size
            try:
                self.stdpvelo   = inarr[:,2]
            except IndexError:
                self.stdpvelo   = np.ones(self.npper, dtype = np.float64)
            try:
                self.pvelp      = inarr[:,3]
            except IndexError:
                pass
            self.isphase        = True
        elif dtype == 'gr' or dtype == 'group':
            if self.isgroup:
                print ('*** group velocity data is already stored!')
                return False
            inarr 	            = np.loadtxt(infname, dtype = np.float64)
            self.gper           = inarr[:,0]
            self.gvelo          = inarr[:,1]
            self.ngper          = self.gper.size
            try:
                self.stdgvelo   = inarr[:,2]
            except IndexError:
                self.stdgvelo   = np.ones(self.ngper, dtype = np.float64)
            try:
                self.gvelp      = inarr[:,3]
            except IndexError:
                pass
            self.isgroup        = True
        else:
            raise ValueError('Unexpected dtype: '+ dtype)
        return True

    def get_disp(self, indata, dtype='ph'):
        """
        get dispersion curve data from a input numpy array
        ==========================================================================
        ::: input :::
        indata      - input array (3, N)
        dtype       - data type (phase/group)
        ::: output :::
        dispersion curve is stored
        ==========================================================================
        """
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if self.isphase:
                print 'phase velocity data is already stored!'
                return False
            self.pper           = indata[0, :]
            self.pvelo          = indata[1, :]
            self.npper          = self.pper.size
            try:
                self.stdpvelo   = indata[2, :]
            except IndexError:
                self.stdpvelo   = np.ones(self.npper, dtype=np.float64)
            self.isphase        = True
        elif dtype == 'gr' or dtype == 'group':
            if self.isgroup:
                print 'group velocity data is already stored!'
                return False
            self.gper           = indata[0, :]
            self.gvelo          = indata[1, :]
            self.ngper          = self.gper.size
            try:
                self.stdgvelo   = indata[2, :]
            except IndexError:
                self.stdgvelo   = np.ones(self.ngper, dtype=np.float64)
            self.isgroup        = True
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def get_azi_disp(self, indata):
        """
        get dispersion curve data from a input numpy array
        ==========================================================================
        ::: input :::
        indata      - input array (3, N)
        dtype       - data type (phase/group)
        ::: output :::
        dispersion curve is stored
        ==========================================================================
        """
        self.isphase        = True
        # isotropic phase vel
        self.pper           = indata[0, :]
        self.pvelo          = indata[1, :]
        self.npper          = self.pper.size
        self.stdpvelo       = indata[2, :]
        # azimuthal terms
        self.psi2           = indata[3, :]
        self.unpsi2         = indata[4, :]
        self.amp            = indata[5, :]
        self.unamp          = indata[6, :]
        return True
    
    def writedisptxt(self, outfname, dtype='ph', predisp=True, obsdisp=True):
        """
        Write dispersion curve to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        dtype       - data type (phase/group)
        ::: output :::
        a txt file contains predicted and observed dispersion data
        ==========================================================================
        """
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            if predisp:
                outArr  = np.append(self.pper, self.pvelp)
            else:
                outArr = self.pper.copy()
            if obsdisp:
                outArr  = np.append(outArr, self.pvelo)
                outArr  = np.append(outArr, self.stdpvelo)
            if predisp and not obsdisp:
                Ncolumn = 2
            elif not predisp and obsdisp:
                Ncolumn = 3
            elif predisp and obsdisp:
                Ncolumn = 4
            else:
                raise ValueError('At least one of predisp/obsdisp must be True!')
            outArr  = outArr.reshape((Ncolumn, self.npper))
            outArr  = outArr.T
            if Ncolumn == 4:
                header  = 'pper pvelp pvelo stdpvelo'
            elif Ncolumn == 3:
                header  = 'pper pvelo stdpvelo'
            elif Ncolumn == 2:
                header  = 'pper pvelp'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
        elif dtype == 'gr' or dtype == 'group':
            if not self.isgroup:
                print 'group velocity data is not stored!'
                return False
            outArr  = np.append(self.gper, self.gvelp)
            outArr  = np.append(outArr, self.gvelo)
            outArr  = np.append(outArr, self.stdgvelo)
            outArr  = outArr.reshape((4, self.ngper))
            outArr  = outArr.T 
            header  = 'gper gvelp gvelo stdgvelo'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    
    def readaziamptxt(self, infname, dtype='ph'):
        """
        Read input txt file of azimuthal amplitude
        ==========================================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (phase/group)
        ::: output :::
        azimuthal amplitude is stored
        ==========================================================================
        """
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            inArr 	    = np.loadtxt(infname, dtype=np.float64)
            if not np.allclose(self.pper , inArr[:,0]):
                print 'inconsistent period array !'
                return False
            self.pampo  = inArr[:,1]
            self.npper  = self.pper.size
            try:
                self.stdpampo   = inArr[:,2]
            except IndexError:
                self.stdpampo   = np.ones(self.npper, dtype=np.float64)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def writeaziamptxt(self, outfname, dtype='ph'):
        """
        Write azimuthal amplitude to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        dtype       - data type (phase/group)
        ::: output :::
        a txt file contains predicted and observed dispersion data
        ==========================================================================
        """  
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            outArr  = np.append(self.pper, self.pampp)
            outArr  = np.append(outArr, self.pampo)
            outArr  = np.append(outArr, self.stdpampo)
            outArr  = outArr.reshape((4, self.npper))
            outArr  = outArr.T
            header  = 'pper pampp pampo stdpampo'
            np.savetxt(outfname, outArr, fmt='%g')
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def readaziphitxt(self, infname, dtype='ph'):
        """
        Read input txt file of fast direction azimuth
        ==========================================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (phase/group)
        ::: output :::
        fast direction azimuth is stored 
        ==========================================================================
        """
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            inArr 		= np.loadtxt(infname, dtype=np.float64)
            if not np.allclose(self.pper , inArr[:, 0]):
                print 'inconsistent period array !'
                return False
            self.pphio  = inArr[:,1]
            self.npper  = self.pper.size
            try:
                self.stdpphio   = inArr[:,2]
            except IndexError:
                self.stdpphio   = np.ones(self.npper, dtype=np.float64)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def writeaziphitxt(self, outfname, dtype='ph'):
        """
        Write fast direction azimuth to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        dtype       - data type (phase/group)
        ::: output :::
        a txt file contains predicted and observed dispersion data
        ==========================================================================
        """ 
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            outArr  = np.append(self.pper, self.pphip)
            outArr  = np.append(outArr, self.pphio)
            outArr  = np.append(outArr, self.stdpphio)
            outArr  = outArr.reshape((4, self.npper))
            outArr  = outArr.T
            header  = 'pper pphip pphio stdpphio'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
        
    def writedispttitxt(self, outfname, dtype='ph'):
        """
        Write dispersion curve to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        dtype       - data type (phase/group)
        ::: output :::
        a txt file contains predicted and observed dispersion data
        ==========================================================================
        """
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            outArr  = np.append(self.pper, self.pvelp)
            outArr  = np.append(outArr, self.pvelo)
            outArr  = np.append(outArr, self.stdpvelo)
            # azimuthal amplitude
            outArr  = np.append(outArr, self.pampp)
            outArr  = np.append(outArr, self.pampo)
            outArr  = np.append(outArr, self.stdpampo)
            # fast-direction azimuth
            outArr  = np.append(outArr, self.pphip)
            outArr  = np.append(outArr, self.pphio)
            outArr  = np.append(outArr, self.stdpphio)
            outArr  = outArr.reshape((10, self.npper))
            outArr  = outArr.T
            header  = 'pper pvelp pvelo stdpvelo pampp pampo stdpampo pphip pphio stdpphio'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
        return True
    
    #----------------------------------------------------
    # functions computing misfit
    #----------------------------------------------------
    def get_pmisfit(self):
        """
        Compute the misfit for phase velocities
        """
        if not self.isphase :
            print('No phase velocity data stored')
            return False
        temp            = ((self.pvelo - self.pvelp)**2/self.stdpvelo**2).sum()
        self.pmisfit    = np.sqrt(temp/self.npper)
        self.pS         = temp
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        self.pL         = np.exp(-0.5 * temp)
        return True
    
    def get_gmisfit(self):
        """
        Compute the misfit for group velocities
        """
        if not self.isgroup:
            print('No group velocity data stored')
            return False
        temp            = ((self.gvelo - self.gvelp)**2/self.stdgvelo**2).sum()
        self.gmisfit    = np.sqrt(temp/self.ngper)
        self.gS         = temp
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        self.gL         = np.exp(-0.5 * temp)
        return True
    
    def get_misfit(self):
        """
        Compute combined misfit of both phase and group dispersion
        """
        # misfit for phase velocities
        temp1           = 0.
        temp2           = 0.
        if self.isphase: # isphase is determined when reading phase velocity data
            temp1       += ((self.pvelo - self.pvelp)**2/self.stdpvelo**2).sum()
            tS          = temp1
            self.pS     = tS
            misfit      = np.sqrt(temp1/self.npper)
            if tS > 50.:
                tS      = np.sqrt(tS*50.)
            L           = np.exp(-0.5 * tS)
            self.pmisfit= misfit
            self.pL     = L
        # misfit for group velocities
        if self.isgroup: # isgroup is determined when reading group velocity data
            temp2       += ((self.gvelo - self.gvelp)**2/self.stdgvelo**2).sum()
            tS          = temp2
            self.gS     = tS
            misfit      = np.sqrt(temp2/self.ngper)
            if tS > 50.:
                tS      = np.sqrt(tS*50.)
            L           = np.exp(-0.5 * tS)
            self.gmisfit= misfit
            self.gL     = L
        if (not self.isphase) and (not self.isgroup):
            print('No dispersion data stored!')
            self.misfit = 0.
            self.L      = 1.
            return False
        # misfit for both
        temp            = temp1 + temp2
        self.S          = temp
        self.misfit     = np.sqrt(temp/(self.npper+self.ngper))
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        self.L          = np.exp(-0.5 * temp)
        return True
    
    def get_misfit_tti(self):
        """
        compute misfit for inversion of tilted TI models, only applies to phase velocity dispersion
        """
        temp1                   = ((self.pvelo - self.pvelp)**2/self.stdpvelo**2).sum()
        temp2                   = ((self.pampo - self.pampp)**2/self.stdpampo**2).sum()
        phidiff                 = abs(self.pphio - self.pphip)
        phidiff[phidiff>90.]    = 180. - phidiff[phidiff>90.]
        temp3                   = (phidiff**2/self.stdpphio**2).sum()
        self.pS                 = temp1+temp2+temp3
        tS                      = temp1+temp2+temp3
        self.pmisfit            = np.sqrt(tS/3./self.npper)
        if tS > 50.:
            tS                  = np.sqrt(tS*50.)
        self.pL                 = np.exp(-0.5 * tS)
        return

    def expected_misfit(self):
        # misfit for phase velocities
        stdarr      = np.array([], dtype=np.float64)
        if self.isphase: # isphase is determined when reading phase velocity data
            stdarr  = np.append(stdarr, self.stdpvelo)
        if self.isgroup: # isgroup is determined when reading group velocity data
            stdarr  = np.append(stdarr, self.stdgvelo)
        
        # self.exp_misfit\
        #             = _fast_compute_expect_misfit(stdarr, 1000000)
        # temp1           = 0.
        # temp2           = 0.
        # if self.isphase: # isphase is determined when reading phase velocity data
        #     temp1       += ((np.random.normal(scale=self.stdpvelo))**2/self.stdpvelo**2).sum()
        # # misfit for group velocities
        # if self.isgroup: # isgroup is determined when reading group velocity data
        #     temp2       += ((np.random.normal(scale=self.stdgvelo))**2/self.stdgvelo**2).sum()
        # # misfit for both
        # temp            = temp1 + temp2
        # self.exp_misfit = np.sqrt(temp/(self.npper+self.ngper))
        return
    
    def check_pdisp(self, dtype='ph', Tthresh = 50., mono_tol  = 0.001, dv_tol=0.2):
        """ check the predicted phase velocity
        """
        if dtype == 'ph':
            pvel    = self.pvelp
            periods = self.pper
        elif dtype == 'gr':
            pvel    = self.gvelp
            periods = self.gper
        else:
            raise ValueError('Unexpected input dtype = '+dtype)
        # monotonical increase check
        ind         = periods > Tthresh
        if (periods[ind]).size >= 2:
            temp_pers   = periods[ind]
            temp_vel    = pvel[ind]
            vel_left    = temp_vel[:-1]
            vel_right   = temp_vel[1:]
            if np.any( (vel_left - vel_right) >= mono_tol ):
                return False
        # check the discontinuity in dispersion curves
        vel_left        = pvel[:-1]
        vel_right       = pvel[1:]
        if np.any( abs(vel_left - vel_right) >= dv_tol ):
            return False
        return True
    
    def check_large_perturb(self, thresh=10.):
        """check the differences between reference dispersion curve and predicted dispersion curve
        """
        return (abs(self.pvelref - self.pvelp)/self.pvelref).max()*100. > thresh
    
    def plot_azi_fit_old(self, inpsi=[], inamp=[]):
        plt.figure(figsize=[18, 9.6])
        ax      = plt.subplot()
        #self.psi2[self.psi2 - self.ppsi2 > 90.]     -= 180.
        #self.psi2[self.psi2 - self.ppsi2 < -90.]    += 180.
        self.ppsi2[self.ppsi2 - self.psi2 > 90.]     -= 180.
        self.ppsi2[self.ppsi2 - self.psi2 < -90.]     += 180.
        if len(inpsi) > 0:
            inpsi[inpsi - self.psi2 > 90.]     -= 180.
            inpsi[inpsi - self.psi2 < -90.]     += 180.
            
            plt.errorbar(self.pper, self.psi2, yerr=self.unpsi2, fmt='ko')
            plt.plot(self.pper, self.ppsi2, 'b-', lw=3, label='two-layer model')
            if len(inpsi) == self.npper:
                plt.plot(self.pper, inpsi, 'r--', lw=3, label='three-layer model')
            plt.ylabel('Fast azimuth (deg)', fontsize=50)
            plt.xlabel('Period (sec)', fontsize=50)
            ax.tick_params(axis='x', labelsize=30)
            ax.tick_params(axis='y', labelsize=30)
            if len(inamp) == self.npper:
                plt.legend(fontsize=30)
            #
            plt.figure(figsize=[18, 9.6])
            ax      = plt.subplot()
            plt.errorbar(self.pper, self.amp, yerr=self.unamp, fmt='ko')
            plt.plot(self.pper, self.pamp, 'b-', lw=3, label='two-layer model')
            if len(inamp) == self.npper:
                plt.plot(self.pper, inamp, 'r--', lw=3, label='three-layer model')
            plt.ylabel('Anisotropy amplitude (%)', fontsize=50)
            plt.xlabel('Period (sec)', fontsize=50)
            ax.tick_params(axis='x', labelsize=30)
            ax.tick_params(axis='y', labelsize=30)
            if len(inamp) == self.npper:
                plt.legend(fontsize=30)
            ymax    = (self.amp+self.unamp).max()
            plt.ylim([0., ymax])
            plt.show()
        
    def plot_azi_fit(self, psitype=0, title=''):
        # plt.figure(figsize=[18, 9.6])
        fig, axs = plt.subplots(2, 1)
        if psitype == 0:
            self.psi2[self.psi2 - self.ppsi2 > 90.] -= 180.
            self.psi2[self.psi2 - self.ppsi2 < -90.] += 180.
        elif psitype == 1:
            self.psi2[self.psi2<0.]     += 180.
            self.ppsi2[self.ppsi2<0.]     += 180.
        elif psitype == 2:
            self.psi2[self.psi2>90.]   -= 180.
            self.ppsi2[self.ppsi2>90.]   -= 180.
        elif psitype == 3:
            self.psi2[self.psi2>90.]    -= 180.
            self.ppsi2[self.ppsi2>90.]  -= 180.
            
        
        axs[0].errorbar(self.pper, self.psi2, yerr=self.unpsi2, fmt='ko')
        axs[0].plot(self.pper, self.ppsi2, 'b-', lw=3 )
        axs[0].set_ylabel('Fast azimuth (deg)', fontsize=20)
        # axs[0].set_xlabel('Period (sec)', fontsize=30)
        axs[0].tick_params(axis='x', labelsize=30)
        axs[0].tick_params(axis='y', labelsize=15)
        #
        # plt.figure(figsize=[18, 9.6])
        # ax      = plt.subplot()
        axs[1].errorbar(self.pper, self.amp, yerr=self.unamp, fmt='ko')
        axs[1].plot(self.pper, self.pamp, 'b-', lw=3 )
        axs[1].set_ylabel('Anisotropy amplitude (%)', fontsize=20)
        axs[1].set_xlabel('Period (sec)', fontsize=30)
        axs[1].tick_params(axis='x', labelsize=30)
        axs[1].tick_params(axis='y', labelsize=15)
        ymax    = (self.amp+self.unamp).max()
        axs[1].set_ylim([0., ymax])
        plt.suptitle('Misfit = %g' %self.pmisfit+title, fontsize=30)
        plt.show()
        
    def plot_azi_fit_2(self, psitype=0, title=''):
        # plt.figure(figsize=[18, 9.6])
        fig, axs = plt.subplots(2, 1)
        if psitype == 0:
            self.psi2[self.psi2 - self.ppsi2 > 90.] -= 180.
            self.psi2[self.psi2 - self.ppsi2 < -90.] += 180.
        elif psitype == 1:
            self.psi2[self.psi2<0.]     += 180.
            self.ppsi2[self.ppsi2<0.]     += 180.
        elif psitype == 2:
            self.psi2[self.psi2>90.]   -= 180.
            self.ppsi2[self.ppsi2>90.]   -= 180.
        elif psitype == 3:
            self.psi2[self.psi2>90.]    -= 180.
            self.ppsi2[self.ppsi2>90.]  -= 180.
        
        #
        min_psi2 = self.psi2 - self.unpsi2
        max_psi2 = self.psi2 + self.unpsi2
        if np.any(self.ppsi2 > max_psi2):
            self.unpsi2[self.ppsi2 > max_psi2] *= 1.5
        if np.any(self.ppsi2 < min_psi2):
            self.unpsi2[self.ppsi2 > min_psi2] *= 1.5
        #
        
        axs[0].errorbar(self.pper, self.psi2, yerr=self.unpsi2, fmt='ko')
        axs[0].plot(self.pper, self.ppsi2, 'b-', lw=3 )
        axs[0].set_ylabel('Fast azimuth (deg)', fontsize=20)
        # axs[0].set_xlabel('Period (sec)', fontsize=30)
        axs[0].tick_params(axis='x', labelsize=30)
        axs[0].tick_params(axis='y', labelsize=15)
        #
        # plt.figure(figsize=[18, 9.6])
        # ax      = plt.subplot()
        axs[1].errorbar(self.pper, self.amp, yerr=self.unamp, fmt='ko')
        axs[1].plot(self.pper, self.pamp, 'b-', lw=3 )
        axs[1].set_ylabel('Anisotropy amplitude (%)', fontsize=20)
        axs[1].set_xlabel('Period (sec)', fontsize=30)
        axs[1].tick_params(axis='x', labelsize=30)
        axs[1].tick_params(axis='y', labelsize=15)
        ymax    = (self.amp+self.unamp).max()
        axs[1].set_ylim([0., ymax])
        plt.suptitle('Misfit = %g' %self.pmisfit+title, fontsize=30)
        plt.show()
    
    def get_misfit_hti(self):
        temp1                   = ((self.amp - self.pamp)**2/self.unamp**2).sum()
        psidiff                 = abs(self.psi2 - self.ppsi2)
        psidiff[psidiff>90.]    = 180. - psidiff[psidiff>90.]
        temp2                   = (psidiff**2/self.unpsi2**2).sum()
        #
        self.pS                 = temp1+temp2
        tS                      = temp1+temp2
        self.pmisfit            = np.sqrt(tS/2./self.npper)
        #
        self.pS_amp             = temp1
        self.pmisfit_amp        = np.sqrt(temp1/self.npper)
        #
        self.pS_psi             = temp2
        self.pmisfit_psi        = np.sqrt(temp2/self.npper)
        
        if tS > 50.:
            tS                  = np.sqrt(tS*50.)
        self.pL                 = np.exp(-0.5 * tS)
    
    def get_misfit_hti_select(self, t1, t2):
        index1  = self.pper<=t1
        index2  = self.pper>=t2
        index3  = (self.pper>=t1) * (self.pper<=t2)
        
        psidiff                 = abs(self.psi2[index1] - self.ppsi2[index1])
        psidiff[psidiff>90.]    = 180. - psidiff[psidiff>90.]
        temp2                   = (psidiff**2/self.unpsi2[index1]**2).sum()

        self.pmisfit_crt        = np.sqrt(temp2/(np.where(index1)[0].size))
        
        psidiff                 = abs(self.psi2[index2] - self.ppsi2[index2])
        psidiff[psidiff>90.]    = 180. - psidiff[psidiff>90.]
        temp2                   = (psidiff**2/self.unpsi2[index2]**2).sum()

        self.pmisfit_man        = np.sqrt(temp2/(np.where(index2)[0].size))
        
        psidiff                 = abs(self.psi2[index3] - self.ppsi2[index3])
        psidiff[psidiff>90.]    = 180. - psidiff[psidiff>90.]
        temp2                   = (psidiff**2/self.unpsi2[index3]**2).sum()

        self.pmisfit_med        = np.sqrt(temp2/(np.where(index3)[0].size))
        
        
        
        
        
    def check_disp(self, thresh=0.4):
        diff_vel                = abs(self.pvelo - self.pvelp)
        return np.any(diff_vel>thresh)
        
class data1d(object):
    """
    An object for handling input data for inversion
    ==========================================================================
    ::: parameters :::
    dispR   - Rayleigh wave dispersion data
    dispL   - Love wave dispersion data
    rfr     - radial receiver function data
    rft     - transverse receiver function data
    misfit  - misfit value
    L       - likelihood value
    ==========================================================================
    """
    def __init__(self):
        self.dispR  = disp()
        self.dispL  = disp()
        self.rfr    = rf()
        self.rft    = rf()
        return
    
    def get_misfit(self, wdisp=0.2, rffactor=40.):
        """
        Compute combined misfit
        ==========================================================================================
        ::: input :::
        wdisp       - relative weigh for dispersion data ( 0.~1. )
                        wdisp == 0.: misfit of dispersion data is 0., likelihood is 1
                        wdisp == 1.: misfit of receiver function data is 0., likelihood is 1
                    
        rffactor    - factor for downweighting the misfit for likelihood computation of rf
        ==========================================================================================
        """
        if wdisp > 0.:
            self.dispR.get_misfit()
        else:
            self.dispR.misfit   = 0.
            self.dispR.L        = 1.
        if wdisp < 1.:
            self.rfr.get_misfit(rffactor = rffactor)
        else:
            self.rfr.misfit     = 0.
            self.rfr.L          = 1.
        # compute combined misfit and likelihood
        self.misfit = wdisp*self.dispR.misfit + (1.-wdisp)*self.rfr.misfit
        self.L      = ((self.dispR.L)**wdisp)*((self.rfr.L)**(1.-wdisp))
        return
   
    def get_misfit_vti(self):
       """
       compute misfit for inversion of Vertical TI models, only applies to phase velocity dispersion
       """
       self.dispR.get_misfit()
       self.dispL.get_misfit()
       self.misfit = np.sqrt((self.dispR.pS + self.dispL.pS)/(self.dispR.npper + self.dispL.npper) )
       tS          = (self.dispR.pS + self.dispL.pS)
       if tS > 50.:
           tS      = np.sqrt(tS*50.)
       if tS > 50.:
           tS      = np.sqrt(tS*50.)
       self.L      = np.exp(-0.5 * tS)
       return
    
    def get_misfit_vti_2(self):
       """
       compute misfit for inversion of Vertical TI models, only applies to phase velocity dispersion
       """
       self.dispR.get_misfit()
       self.dispL.get_misfit()
       self.misfit = np.sqrt((self.dispR.pS + self.dispR.gS + self.dispL.pS)/(self.dispR.npper +self.dispR.ngper + self.dispL.npper) )
       tS          = (self.dispR.pS + self.dispR.gS + self.dispL.pS)
       if tS > 50.:
           tS      = np.sqrt(tS*50.)
       if tS > 50.:
           tS      = np.sqrt(tS*50.)
       self.L      = np.exp(-0.5 * tS)
       return
    
    def get_misfit_hti(self):
       """
       compute misfit for inversion of Vertical TI models, only applies to phase velocity dispersion
       """
       self.dispR.get_misfit_hti()
       self.misfit = self.dispR.pmisfit
       self.L      = self.dispR.pL
       return
