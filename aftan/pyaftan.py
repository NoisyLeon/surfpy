# -*- coding: utf-8 -*-
"""
A python module for aftan (Automatic Frequency-Time ANalysis) analysis

This module include two major functions:
1. aftan written in pure python
2. a python wrapper for fortran77 version of aftan 

:Dependencies:
    numpy >=1.9.1
    scipy >=0.18.0
    matplotlib >=1.4.3
    ObsPy >=1.0.1
    pyfftw 0.10.3 (optional)
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
    
:Version Date:
    2020/07/07

:References:
    Levshin, A. L., & Ritzwoller, M. H. (2001). Automated detection, extraction, and measurement of regional surface waves.
        In Monitoring the comprehensive nuclear-test-ban treaty: Surface waves (pp. 1531-1545). Birkhäuser, Basel.
    Bensen, G. D., et al. (2007). Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements.
        Geophysical Journal International, 169(3), 1239-1260.
    Feng, L., & Ritzwoller, M. H. (2017). The effect of sedimentary basins on surface waves that pass through them.
        Geophysical Journal International, 211(1), 572-592.
    Feng, L., & Ritzwoller, M. H. (2019). A 3‐D shear velocity model of the crust and uppermost mantle beneath Alaska including apparent radial anisotropy.
        Journal of Geophysical Research: Solid Earth, 124(10), 10468-10497.
"""

import obspy
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import warnings
import numba
from scipy.signal import argrelmax, argrelmin, argrelextrema
import scipy.interpolate
import scipy.integrate

try:
    import pyfftw
    useFFTW = True
except:
    useFFTW = False
try:
    import surfpy.aftan.aftan as aftan
    isaftanf77  = True
except:
    isaftanf77  = False

# ------------- aftan specific exceptions ---------------------------------------
class ftanError(Exception):
    pass

class ftanIOError(ftanError, IOError):
    pass

class ftanHeaderError(ftanError):
    """
    Raised if header has issues.
    """
    pass

class ftanDataError(ftanError):
    """
    Raised if header has issues.
    """
    pass
#============================================================================
# Auxiliary functions
#============================================================================
@numba.jit(numba.float32[:](numba.float32[:], numba.int64, numba.int64, numba.int64, numba.int64, numba.int64), nopython=True)
def _taper(data, npts, nb, ne, ntapb, ntape):
    """Taper the input data with cosine window
    """
    omb         = np.pi/ntapb
    ome         = np.pi/ntape
    ncorr       = int(ne+ntape)
    if ncorr > npts:
        ncorr   = npts
    dataTapered = np.zeros(npts, dtype = np.float32)
    dataTapered[:ncorr]             = data[:ncorr]
    #=================================
    #zerp padding and cosine tapering
    #=================================
    # left end of the signal
    if nb-ntapb-1 > 0:
        dataTapered[:nb-ntapb-1]    = 0.
    if nb > ntapb:
        k                           = np.arange(ntapb+1)+nb-ntapb
        rwinb                       = (np.cos(omb*(nb-k))+1.)/2.
        dataTapered[nb-ntapb-1:nb]  = rwinb*dataTapered[nb-ntapb-1:nb]
        sums                        = 2.*np.sum(rwinb)
    else:
        k                           = np.arange(nb)
        rwinb                       = (np.cos(omb*(nb-k))+1.)/2.
        dataTapered[:nb]            = rwinb*dataTapered[:nb]
        sums                        = 2.*np.sum(rwinb)
    # right end of the signal
    if ne+ntape < npts:
        k                           = np.arange(ntape+1)+ne
        rwine                       = (np.cos(ome*(ne-k))+1.)/2.
        dataTapered[ne-1:ne+ntape]  = dataTapered[ne-1:ne+ntape]*rwine
    elif ne < npts:
        k                           = np.arange(npts-ne+1)+ne
        rwine                       = (np.cos(ome*(ne-k))+1.)/2.
        dataTapered[ne-1:]          = dataTapered[ne-1:]*rwine
    sums    = sums+ne-nb-1
    c       = np.sum(dataTapered[:ncorr])
    c       = -c/sums
    # detrend
    if nb > ntapb:
        dataTapered[nb-ntapb-1:nb]  = rwinb*c+dataTapered[nb-ntapb-1:nb]
    if ne+ntape<npts:
        dataTapered[ne-1:ne+ntape]  = dataTapered[ne-1:ne+ntape] + rwine*c
    elif ne < npts:
        dataTapered[ne-1:]          = dataTapered[ne-1:] + rwine*c
    dataTapered[nb:ne-1]            = dataTapered[nb:ne-1]+c
    return dataTapered

@numba.jit(numba.complex64[:](numba.float32, numba.float32, numba.int64, numba.complex64[:], numba.float32[:]), nopython=True)
def _aftan_gaussian_filter(alpha, omega0, ns, indata, omsArr):
    """Internal Gaussian filter function used for aftan
    """
    om2                         = -(omsArr-omega0)*(omsArr-omega0)*alpha/omega0/omega0
    b                           = np.exp(om2)
    b[np.abs(om2)>=40.]         = 0.
    filterred_data              = indata*b
    filterred_data[int(ns/2):]  = 0
    filterred_data[0]           /= 2
    filterred_data[int(ns/2)-1] = filterred_data[int(ns/2)-1].real+0.j
    return filterred_data

@numba.jit(numba.types.UniTuple(numba.float32, 4)(numba.float32, numba.int64, numba.float32[:], numba.float32[:],\
        numba.float32, numba.float32), nopython=True)
def _fmax(dt, ind, amp, pha, om, piover4):
    """parabolic interpolation of signal amplitude and phase, finding phase derivative
    """
    ind_l       = ind-1
    ind_r       = ind+1
    dd          = amp[ind_l]+amp[ind_r]-2.*amp[ind]
    if dd == 0.:
        t       = 0.
    else:
        t       = (amp[ind_l]-amp[ind_r])/dd/2.0
    a1          = pha[ind_l]
    a2          = pha[ind]
    a3          = pha[ind_r]
    k1          = (a2-a1-om*dt)/2./np.pi
    k1          = np.round(k1)
    a2          = a2-2.*k1*np.pi
    k2          = (a3-a2-om*dt)/2./np.pi
    k2          = np.round(k2)
    a3          = a3-2.*k2*np.pi
    dph         = t*(a1+a3-2.*a2)+(a3-a1)/2.
    tm          = t*t*(amp[ind_l]+amp[ind_r]-2.*amp[ind])/2.+t*(amp[ind_l]-amp[ind_r])/2.+amp[ind]
    ph          = t*t*(a1+a3-2.*a2)/2.+t*(a3-a1)/2.+a2+np.pi*piover4/4.
    return dph, tm, ph, t

@numba.jit(numba.types.Tuple((numba.float32[:], numba.int64[:], numba.int64))\
        (numba.int64, numba.float32[:], numba.float32[:], numba.float32), nopython=True)
def _trigger(nf, grvel, om , tresh):
    """Detect jumps in dispersion curve
    """
    hh1             = om[1:nf-1] - om[:nf-2]
    hh2             = om[2:] - om[1:nf-1]
    hh3             = hh1 + hh2
    r               = (grvel[:nf-2]/hh1 - (1./hh1+1/hh2)*grvel[1:nf-1] + grvel[2:]/hh2)*hh3/4.*100.
    ftrig           = np.zeros(nf, dtype=np.float32)
    ftrig[1:nf-1]   = r
    trig            = np.zeros(nf, dtype=np.int64)
    ierr            = 0
    for i in range(nf):
        if i == 0: continue
        if i == (nf-1): break
        if ftrig[i] > tresh:
            trig[i] = 1
            ierr    = 1
        elif ftrig[i] < -tresh:
            trig[i] = -1
            ierr    = 1
    return ftrig, trig, ierr

@numba.jit(numba.types.Tuple((numba.float32, numba.int64, numba.int64, numba.float32[:], numba.float32[:]))\
        (numba.float32, numba.float32, numba.float32, numba.float32, numba.int64), nopython=True)
def _tapers(omb, ome, dom, alpha, ns):
    """spectra tapering
    """        
    om2d                = omb/dom
    tresh               = 0.5
    wd                  = max(16., om2d*np.sqrt(tresh/alpha) )
    om1                 = int(round(max(1, om2d-wd/2)))
    om2                 = int(round(min(ns*1, om1+wd)))
    ampdom              = np.zeros(ns, dtype=np.float32)
    iArr1               = np.arange(float(om2-om1+1))+om1
    ampdom[om1-1:om2]   = (1.-np.cos(np.pi/(om2-om1)*(iArr1-om1)))/2.
    om3d                = ome/dom
    wd                  = max(16., om3d*np.sqrt(tresh/alpha))
    om4                 = int(round(min(ns*1, om3d+wd/2)))
    om3                 = int(round(max(1, om4-wd)))
    iArr2               = np.arange(float(om4-om3+1))+om3
    iArr2               = iArr2[::-1]
    ampdom[om3-1:om4]   = (1.-np.cos(np.pi/(om4-om3)*(iArr2-om3)))/2.
    ampdom[om2-1:om3]   = 1.
    omdom               = np.arange(ns)*dom
    omstart             = omb
    inds                = om1
    inde                = om4
    return omstart, inds, inde, omdom, ampdom


def _tgauss(dt, t0, fsnr, gt0, dw, n, fmatch, seis):
    """taper phase matched signal
    """        
    ss              = seis.copy()
    nc              = round(gt0/dt)+1
    smax            = np.abs(seis)
    ism             = smax.argmax()
    sm              = smax[ism]
    local_le        = argrelextrema(smax, np.less_equal)[0]
    local_e         = argrelextrema(smax, np.equal)[0]
    ind_localminima = np.setxor1d(local_le, local_e, assume_unique=True)
    ind_left        = ind_localminima[ind_localminima<ism]
    ind_right       = ind_localminima[ind_localminima>ism]
    val_left        = smax[ind_left]
    val_right       = smax[ind_right]
    nnnl            = 0
    if ind_left.size != 0:
        temp_nnnl   = ind_left[((ism-ind_left)*dt>5.)*(val_left<fsnr*sm)]
        if temp_nnnl.size != 0:
            nnnl    = temp_nnnl[-1]
            if temp_nnnl.size > 1:
                nnl = temp_nnnl[-2]
            else:
                nnl = 0
    nnnr    = 0
    if ind_right.size != 0:
        temp_nnnr   = ind_right[((ind_right-ism)*dt>5.)*(val_right<fsnr*sm)]
        if temp_nnnr.size != 0:
            nnnr    = temp_nnnr[0]
            if temp_nnnr.size > 1:
                nnr = temp_nnnr[1]
            else:
                nnr = n-1        
    if nnnr != 0 and nnnl != 0:
        nn      = max(abs(ism-nnnl), abs(ism-nnnr))
        nnn     = max(abs(nnnl-nnl), abs(nnnr-nnr))
        nnnl    = ism -nn
        nnl     = nnnl-nnn
        nnnr    = ism +nn
        nnr     = nnnr+nnn
    tresh = np.log(sm)-24.
    if nnnl != 0:
        nnl                 = int(round((nnl-ism)*fmatch))+ism
        nnnl                = int(round((nnnl-ism)*fmatch))+ism
        nnl                 = max(0, nnl)
        nnnl                = max(0, nnnl)
        freq                = (nnnl-nnl)+1
        iArr                = np.arange(nnnl+1.)
        tre                 = -(iArr-nnnl)/freq*(iArr-nnnl)/freq/2.
        temp_ss             = ss[:nnnl+1]
        temp_ss[tre>tresh]  = temp_ss[tre>tresh]*(np.exp(tre))[tre>tresh]
        temp_ss[tre<=tresh] = 0+0j
        ss[:nnnl+1]         = temp_ss
    if nnnr != 0:
        nnr                 = int(round((nnr-ism)*fmatch))+ism
        nnnr                = int(round((nnnr-ism)*fmatch))+ism
        nnr                 = min(n-1, nnr)
        nnnr                = min(n-1, nnnr)
        freq                = (nnr-nnnr)+1
        iArr                = np.arange(float(n-nnnr))+nnnr+1
        tre                 = -(iArr-nnnr-1)/freq*(iArr-nnnr-1)/freq/2.
        temp_ss             = ss[nnnr:]
        temp_ss[tre>tresh]  = temp_ss[tre>tresh]*(np.exp(tre))[tre>tresh]
        temp_ss[tre<=tresh] = 0+0j
        ss[nnnr:]           = temp_ss
    return ss

@numba.jit(numba.types.UniTuple(numba.float32[:], 14) \
        (numba.float32, numba.int64, numba.int64, numba.int64, numba.float32[:, :], numba.float32[:, :], numba.float32[:, :], \
         numba.float32[:], numba.float32[:], numba.float32, numba.float32, numba.float32, numba.float32, numba.int64), nopython=True)
def _freq_time_analysis(tb, nfin, npts, nb, amp, ampo, phaArr, omegaArr, perArr, dist, dt, piover4, tresh, npoints):
    """core auxiliary frequency time analysis function
    """
    tim1        = np.zeros(nfin, dtype=np.float32)
    tvis1       = np.zeros(nfin, dtype=np.float32)
    ampgr1      = np.zeros(nfin, dtype=np.float32)
    grvel1      = np.zeros(nfin, dtype=np.float32)
    snr1        = np.zeros(nfin, dtype=np.float32)
    wdth1       = np.zeros(nfin, dtype=np.float32)
    phgr1       = np.zeros(nfin, dtype=np.float32)
    
    ipar        = np.zeros((nfin, 6, npts), dtype=np.float32)
    nmax_freq   = np.zeros(nfin, dtype=np.int64)
    #======================================
    # loop over frequencies
    #======================================
    for k in range(nfin):
        j       = 0 # index for local max
        tmpmin  = ampo[0, k] # min value
        imin    = 0          # index of min value
        imax    = np.zeros(npts, dtype=np.int64)
        ilmin   = np.zeros(npts, dtype=np.int64)
        irmin   = np.zeros(npts, dtype=np.int64)
        #==========================================================================
        # loop over data to find local maximum and corresponding sidelobe minimums
        #==========================================================================
        for i in range(npts):
            if i == 0: continue
            if tmpmin > ampo[i, k]:
                tmpmin  = ampo[i, k]
                imin    = i
            if i == npts - 1:
                irmin[j-1]  = imin
                break
            if (amp[i-1, k] < amp[i, k]) and (amp[i+1, k] < amp[i, k]):
                dph, tm, ph, t  = _fmax(dt, i, amp[:,k], phaArr[:,k], omegaArr[k], piover4)
                ipar[k, 0, j]   = (nb+i-2+t)*dt 
                ipar[k, 1, j]   = 2.*np.pi*dt/dph
                ipar[k, 2, j]   = tm
                ipar[k, 5, j]   = ph
                imax[j]         = i
                ilmin[j]        = imin # left min value index
                if j > 0:
                    irmin[j-1]  = imin # right min value index
                j               += 1
                # update the min value and index
                tmpmin          = ampo[i, k]
                imin            = i
        # No local maximum
        if j == 0:
            dph, tm, ph, t  = _fmax(dt, (npts-2), amp[:,k], phaArr[:,k], omegaArr[k], piover4)
            ipar[k, 0, j]   = (nb+(npts-2)-2+t)*dt 
            ipar[k, 1, j]   = 2.*np.pi*dt/dph
            ipar[k, 2, j]   = tm
            ipar[k, 5, j]   = ph
            imax[j]         = (npts-2)
            ilmin[j]        = imin # left min value index
            irmin[j-1]      = npts - 1 # right min value index
            j               += 1
        #
        nlocalmax   = j
        nmax_freq[k]= nlocalmax
        imaxlmax    = 0 # index for the max of local maximums
        tmp_max     = ipar[k, 2, 0]
        # loop over local maximums to find the max of local maximums, along with snr(dB) and width
        for m in range(nlocalmax):
            lminval         = ampo[ilmin[m], k]
            rminval         = ampo[irmin[m], k]
            ipar[k, 3, m]   = 20.*np.log10(ampo[imax[m], k]/np.sqrt(lminval*rminval))
            ipar[k, 4, m]   = (abs(imax[m] - irmin[m]) + abs(imax[m] - ilmin[m]))/2.*dt
            if tmp_max<ipar[k, 2, m]:
                tmp_max     = ipar[k, 2, m]
                imaxlmax    = m
        # assign the values of 
        tim1[k]   = ipar[k, 0, imaxlmax]
        tvis1[k]  = ipar[k, 1, imaxlmax]
        ampgr1[k] = ipar[k, 2, imaxlmax]
        if tim1[k] +tb > 0.:
            grvel1[k] = dist/(tim1[k] +tb) 
        snr1[k]   = ipar[k, 3, imaxlmax]
        wdth1[k]  = ipar[k, 4, imaxlmax] 
        phgr1[k]  = ipar[k, 5, imaxlmax]
    #==========================================================================
    #       Check jumps in dispersion curve 
    #==========================================================================
    grvel2              = grvel1.copy()
    tvis2               = tvis1.copy()
    ampgr2              = ampgr1.copy()
    phgr2               = phgr1.copy()
    snr2                = snr1.copy()
    wdth2               = wdth1.copy()
    ftrig1, trig, ierr  = _trigger(nfin, grvel1, omegaArr, tresh)
    if ierr != 0:
        ijmp    = np.zeros(nfin, dtype=np.int64) # array storing the freq location of jumps
        Njmp    = 0 # number of jumps
        for k in range(nfin):
            if k == 0: continue
            if abs(trig[k] - trig[k-1])>1.5:
                ijmp[Njmp]  = k - 1
                Njmp        += 1
        if Njmp > 1:
            ii  =  np.zeros(Njmp, dtype=np.int64)
            Ncor= 0
            for m in range(Njmp):
                if m == 0: continue
                delijmp = ijmp[m] - ijmp[m - 1]
                if delijmp < npoints:
                    ii[Ncor]= m - 1
                    Ncor    += 1
            #==========================================================================
            # Try to correct jumps by finding another local maximum as group arrival
            #==========================================================================
            if Ncor > 0:
                grvelt  = grvel2.copy()
                tvist   = tvis2.copy()
                ampgrt  = ampgr2.copy()
                phgrt   = phgr2.copy()
                snrt    = snr2.copy()
                wdtht   = wdth2.copy()
                for kk in range(Ncor):
                    istrt   = ijmp[ii[kk]] # startng freq index for jump correction
                    ibeg    = istrt + 1
                    iend    = ijmp[ii[kk] + 1]
                    for i in range(iend - ibeg + 1):
                        k           = ibeg + i
                        ipartmp     = ipar[k, :, :nmax_freq[k]]
                        wor         = np.abs(dist/(ipartmp[0, :]+tb)-grvel2[k-1])
                        ima         = wor.argmin() # find group arrivals that has smallest 
                        grvel2[k]   = dist/(ipartmp[0, ima]+tb)
                        tvis2[k]    = ipartmp[1, ima]
                        ampgr2[k]   = ipartmp[2, ima]
                        phgr2[k]    = ipartmp[5, ima]
                        snr2[k]     = ipartmp[3, ima]
                        wdth2[k]    = ipartmp[4, ima]
                    ftrig2, trig, ierr      =  _trigger(nfin, grvel2, omegaArr, tresh)
                    tmptrig                 = trig[istrt:iend+2]
                    if not np.any(np.abs(tmptrig) > 0.5): 
                        # assign the corrected values
                        grvelt  = grvel2.copy()
                        tvist   = tvis2.copy()
                        ampgrt  = ampgr2.copy()
                        phgrt   = phgr2.copy()
                        snrt    = snr2.copy()
                        wdtht   = wdth2.copy()
                grvel2  = grvelt.copy()
                tvis2   = tvist.copy()
                ampgr2  = ampgrt.copy()
                phgr2   = phgrt.copy()
                snr2    = snrt.copy()
                wdth2   = wdtht.copy()
                ftrig2, trig, ierr      =  _trigger(nfin, grvel2, omegaArr, tresh)
        #==========================================================================
        # after correcting possible jumps, we cut frequency range to single
        # segment with maximum length
        #==========================================================================
        if ierr != 0:
            ist     = 0
            ibe     = 0
            for k in range(nfin):
                if trig[k] != 0:
                    if (k - ibe) > (ibe - ist):
                        ist     = ibe
                        ibe     = k
                else:
                    if k == (nfin - 1):
                        if (k - ibe) > (ibe - ist):
                            ist     = ibe
                            ibe     = k
            nfout2  = ibe - ist + 1
            per2    = perArr[ist:ibe+1]
            grvel2  = grvel2[ist:ibe+1]
            tvis2   = tvis2[ist:ibe+1]
            ampgr2  = ampgr2[ist:ibe+1]
            phgr2   = phgr2[ist:ibe+1]
            snr2    = snr2[ist:ibe+1]
            wdth2   = wdth2[ist:ibe+1]
        else:
            nfout2  = nfin
            per2    = perArr.copy()
    else:
        nfout2 = nfin
        per2   = perArr.copy() 
    return tvis1, ampgr1, grvel1, snr1, wdth1, phgr1, ftrig1, tvis2, ampgr2, grvel2, snr2, wdth2, phgr2, per2


@numba.jit(numba.float32[:]\
        (numba.float32, numba.float32, numba.float32, numba.int64,\
         numba.float32[:], numba.float32[:], numba.float32[:], numba.float32[:], numba.float32[:]), nopython=True)
def __get_phase_vel(dist, Vpred, phpred, n, omegaArr, T, phV, pha, sU):
    """ internal function for _phtovel
    """
    for i in range(n-1):
        m       = n-i-2
        Vpred   = 1/(((sU[m]+sU[m+1])*(omegaArr[m]-omegaArr[m+1])/2.+omegaArr[m+1]/phV[m+1])/omegaArr[m])
        phpred  = omegaArr[m]*(T[m]-dist/Vpred)
        k       = round((phpred -pha[m])/2.0/np.pi)
        phV[m]  = dist/(T[m]-(pha[m]+2.0*k*np.pi)/omegaArr[m])
    return phV
    
def _phtovel(dist, per, U, pha, npr, prper, prvel):
    """Convert observed phase to phase velocity
    """
    omegaArr= 2.*np.pi/per
    T       = dist/U
    sU      = 1./U
    spl     = scipy.interpolate.CubicSpline(prper, prvel)
    Vpred   = spl(per[-1])
    phpred  = omegaArr[-1]*(T[-1]-dist/Vpred)
    k       = round((phpred -pha[-1])/2.0/np.pi)
    phV     = np.zeros(U.size)
    phV[-1] = dist/(T[-1]-(pha[-1]+2.*k*np.pi)/omegaArr[-1])
    n       = omegaArr.size
    return __get_phase_vel(np.float32(dist), np.float32(Vpred), np.float32(phpred), np.int64(n), \
                np.float32(omegaArr), np.float32(T), np.float32(phV), np.float32(pha), np.float32(sU))

#============================================================================
# End of auxiliary functions
#============================================================================

class ftanParam(object):
    """ A class to handle ftan output parameters
    ===========================================================================
    Basic FTAN parameters:
    nfout1_1 - output number of frequencies for arr1, (integer*4)
    arr1_1   - preliminary results.
                Description: real*8 arr1(8,n), n >= nfin)
                arr1_1[0,:] -  central periods, s
                arr1_1[1,:] -  observed periods, s
                arr1_1[2,:] -  group velocities, km/s
                arr1_1[3,:] -  phase velocities, km/s or phase if nphpr=0, rad
                arr1_1[4,:] -  amplitudes, Db
                arr1_1[5,:] -  discrimination function
                arr1_1[6,:] -  signal/noise ratio, Db
                arr1_1[7,:] -  maximum half width, s
                arr1_1[8,:] -  amplitudes
    arr2_1   - final results with jump detection
    nfout2_1 - output number of frequencies for arr2, (integer*4)
                Description: real*8 arr2(7,n), n >= nfin)
                If nfout2 == 0, no final result.
                arr2_1[0,:] -  central periods, s
                arr2_1[1,:] -  observed periods, s
                arr2_1[2,:] -  group velocities, km/sor phase if nphpr=0, rad
                arr2_1[3,:] -  phase velocities, km/s
                arr2_1[4,:] -  amplitudes, Db
                arr2_1[5,:] -  signal/noise ratio, Db
                arr2_1[6,:] -  maximum half width, s
                arr2_1[7,:] -  amplitudes
    tamp_1   -  time to the beginning of ampo table, s (real*8)
    nrow_1   -  number of rows in array ampo, (integer*4)
    ncol_1   -  number of columns in array ampo, (integer*4)
    amp_1    -  Ftan amplitude array, Db, (real*8)
    ierr_1   - completion status, =0 - O.K.,           (integer*4)
                                 =1 - some problems occures
                                 =2 - no final results
    ----------------------------------------------------------------------------
    Phase-Matched-Filtered FTAN parameters:
    nfout1_2 - output number of frequencies for arr1, (integer*4)
    arr1_2   - preliminary results.
                Description: real*8 arr1(8,n), n >= nfin)
                arr1_2[0,:] -  central periods, s (real*8)
                arr1_2[1,:] -  apparent periods, s (real*8)
                arr1_2[2,:] -  group velocities, km/s (real*8)
                arr1_2[3,:] -  phase velocities, km/s (real*8)
                arr1_2[4,:] -  amplitudes, Db (real*8)
                arr1_2[5,:] -  discrimination function, (real*8)
                arr1_2[6,:] -  signal/noise ratio, Db (real*8)
                arr1_2[7,:] -  maximum half width, s (real*8)
                arr1_2[8,:] -  amplitudes 
    arr2_2   - final results with jump detection
    nfout2_2 - output number of frequencies for arr2, (integer*4)
                Description: real*8 arr2(7,n), n >= nfin)
                If nfout2 == 0, no final results.
                arr2_2[0,:] -  central periods, s (real*8)
                arr2_2[1,:] -  apparent periods, s (real*8)
                arr2_2[2,:] -  group velocities, km/s (real*8)
                arr2_2[3,:] -  phase velocities, km/s (real*8)
                arr2_2[4,:] -  amplitudes, Db (real*8)
                arr2_2[5,:] -  signal/noise ratio, Db (real*8)
                arr2_2[6,:] -  maximum half width, s (real*8)
                arr2_2[7,:] -  amplitudes
                arr2_2[8,:] -  signal/noise ratio (optional)
    tamp_2   -  time to the beginning of ampo table, s (real*8)
    nrow_2   -  number of rows in array ampo, (integer*4)
    ncol_2   -  number of columns in array ampo, (integer*4)
    amp_2    -  Ftan amplitude array, Db, (real*8)
    ierr_2   - completion status, =0 - O.K.,           (integer*4)
                                =1 - some problems occures
                                =2 - no final results
    ===========================================================================
    """
    def __init__(self):
        # Parameters for first iteration
        self.nfout1_1   = 0
        self.arr1_1     = np.array([])
        self.nfout2_1   = 0
        self.arr2_1     = np.array([])
        self.tamp_1     = 0.
        self.nrow_1     = 0
        self.ncol_1     = 0
        self.ampo_1     = np.array([],dtype='float32')
        self.ierr_1     = 0
        # Parameters for second iteration
        self.nfout1_2   = 0
        self.arr1_2     = np.array([])
        self.nfout2_2   = 0
        self.arr2_2     = np.array([])
        self.tamp_2     = 0.
        self.nrow_2     = 0
        self.ncol_2     = 0
        self.ampo_2     = np.array([])
        self.ierr_2     = 0
        # Flag for existence of predicted phase dispersion curve
        self.preflag    = False
        self.station_id = None

    def writeDISP(self, fnamePR):
        """
        Write FTAN parameters to DISP files given a prefix.
        fnamePR: file name prefix
        _1_DISP.0: arr1_1
        _1_DISP.1: arr2_1
        _2_DISP.0: arr1_2
        _2_DISP.1: arr2_2
        """
        if self.nfout1_1 != 0:
            f10         = fnamePR+'_1_DISP.0'
            Lf10        = self.nfout1_1
            outArrf10   = np.arange(Lf10)
            for i in range(9): outArrf10   = np.append(outArrf10, self.arr1_1[i,:Lf10])
            outArrf10   = outArrf10.reshape((10,Lf10))
            outArrf10   = outArrf10.T
            np.savetxt(f10, outArrf10, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf %12.4lf %12.4lf')
        if self.nfout2_1 != 0:
            f11         = fnamePR+'_1_DISP.1'
            Lf11        = self.nfout2_1
            outArrf11   = np.arange(Lf11)
            for i in range(8): outArrf11 = np.append(outArrf11, self.arr2_1[i,:Lf11])
            outArrf11   = outArrf11.reshape((9,Lf11))
            outArrf11   = outArrf11.T
            np.savetxt(f11, outArrf11, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf %12.4lf %12.4lf')
        if self.nfout1_2 != 0:
            f20         = fnamePR+'_2_DISP.0'
            Lf20        = self.nfout1_2
            outArrf20   = np.arange(Lf20)
            for i in range(9):
                outArrf20=np.append(outArrf20, self.arr1_2[i,:Lf20])
            outArrf20=outArrf20.reshape((10,Lf20))
            outArrf20=outArrf20.T
            np.savetxt(f20, outArrf20, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf %12.4lf %12.4lf')
        if self.nfout2_2 != 0:
            f21         = fnamePR+'_2_DISP.1'
            Lf21        = self.nfout2_2
            outArrf21   = np.arange(Lf21)
            for i in range(9): outArrf21   = np.append(outArrf21, self.arr2_2[i,:Lf21])
            outArrf21   = outArrf21.reshape((10,Lf21))
            outArrf21   = outArrf21.T
            np.savetxt(f21, outArrf21, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf %12.4lf %12.4lf')
        return
    
    def writeDISPbinary(self, fnamePR):
        """
        Write FTAN parameters to DISP files given a prefix.
        fnamePR: file name prefix
        _1_DISP.0: arr1_1
        _1_DISP.1: arr2_1
        _2_DISP.0: arr1_2
        _2_DISP.1: arr2_2
        """
        f10 = fnamePR+'_1_DISP.0'
        np.savez(f10, self.arr1_1, np.array([self.nfout1_1]) )
        f11 = fnamePR+'_1_DISP.1'
        np.savez(f11, self.arr2_1, np.array([self.nfout2_1]) )
        f20 = fnamePR+'_2_DISP.0'
        np.savez(f20, self.arr1_2, np.array([self.nfout1_2]) )
        f21 = fnamePR+'_2_DISP.1'
        np.savez(f21, self.arr2_2, np.array([self.nfout2_2]) )
        return
    
    def write_npy(self, outfname, outarr = np.array([])):
        attrs_arr   = np.array([self.nfout1_1, self.nfout2_1, self.nfout1_2, self.nfout2_2])
        np.savez( outfname, self.arr1_1, self.arr2_1, self.arr1_2, self.arr2_2, attrs_arr, outarr)
        return
    
    def load_npy(self, infname):
        inarr       = np.load(infname)
        self.arr1_1 = inarr['arr_0']
        self.arr2_1 = inarr['arr_1']
        self.arr1_2 = inarr['arr_2']
        self.arr2_2 = inarr['arr_3']
        attrs_arr   = inarr['arr_4']
        outarr      = inarr['arr_5']
        self.nfout1_1   = attrs_arr[0]
        self.nfout2_1   = attrs_arr[1]
        self.nfout1_2   = attrs_arr[2]
        self.nfout2_2   = attrs_arr[3]
        return outarr

    def FTANcomp(self, inftanparam, compflag=1):
        """
        Compare aftan results for two ftanParam objects.
        """
        if not isinstance(inftanparam, ftanParam):
            raise ValueError('Input inftanparam is not type of ftanParam!')
        fparam1     = self
        fparam2     = inftanparam
        if compflag == 1:
            obper1  = fparam1.arr1_1[1,:fparam1.nfout1_1]
            gvel1   = fparam1.arr1_1[2,:fparam1.nfout1_1]
            phvel1  = fparam1.arr1_1[3,:fparam1.nfout1_1]
            obper2  = fparam2.arr1_1[1,:fparam2.nfout1_1]
            gvel2   = fparam2.arr1_1[2,:fparam2.nfout1_1]
            phvel2  = fparam2.arr1_1[3,:fparam2.nfout1_1]
        elif compflag == 2:
            obper1  = fparam1.arr2_1[1,:fparam1.nfout2_1]
            gvel1   = fparam1.arr2_1[2,:fparam1.nfout2_1]
            phvel1  = fparam1.arr2_1[3,:fparam1.nfout2_1]
            obper2  = fparam2.arr2_1[1,:fparam2.nfout2_1]
            gvel2   = fparam2.arr2_1[2,:fparam2.nfout2_1]
            phvel2  = fparam2.arr2_1[3,:fparam2.nfout2_1]
        elif compflag == 3:
            obper1  = fparam1.arr1_2[1,:fparam1.nfout1_2]
            gvel1   = fparam1.arr1_2[2,:fparam1.nfout1_2]
            phvel1  = fparam1.arr1_2[3,:fparam1.nfout1_2]
            obper2  = fparam2.arr1_2[1,:fparam2.nfout1_2]
            gvel2   = fparam2.arr1_2[2,:fparam2.nfout1_2]
            phvel2  = fparam2.arr1_2[3,:fparam2.nfout1_2]
        else:
            obper1  = fparam1.arr2_2[1,:fparam1.nfout2_2]
            gvel1   = fparam1.arr2_2[2,:fparam1.nfout2_2]
            phvel1  = fparam1.arr2_2[3,:fparam1.nfout2_2]
            obper2  = fparam2.arr2_2[1,:fparam2.nfout2_2]
            gvel2   = fparam2.arr2_2[2,:fparam2.nfout2_2]
            phvel2  = fparam2.arr2_2[3,:fparam2.nfout2_2]
        plt.figure()
        ax          = plt.subplot()
        ax.plot(obper1, gvel1, '--k', lw=3) #
        ax.plot(obper2, gvel2, 'bo', markersize=5)
        plt.xlabel('Period(s)')
        plt.ylabel('Velocity(km/s)')
        plt.title('Group Velocity Comparison')
        if (fparam1.preflag and fparam2.preflag):
            plt.figure()
            ax      = plt.subplot()
            ax.plot(obper1, phvel1, '--k', lw=3) #
            ax.plot(obper2, phvel2, 'bo', markersize=5)
            plt.xlabel('Period(s)')
            plt.ylabel('Velocity(km/s)')
            plt.title('Phase Velocity Comparison')
        return

class InputFtanParam(object): ###
    """
    A class to store input parameters for aftan analysis and SNR Analysis
    ===============================================================================================================
    Parameters:
    pmf         - flag for Phase-Matched-Filtered output (default: Fasle)
    piover4     - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
    vmin        - minimal group velocity, km/s
    vmax        - maximal group velocity, km/s
    tmin        - minimal period, s
    tmax        - maximal period, s
    tresh       - treshold for jump detection, usualy = 10, need modifications
    ffact       - factor to automatic filter parameter, usualy =1
    taperl      - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
    snr         - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
    fmatch      - factor to length of phase matching window
    fhlen       - half length of Gaussian width
    nfin        - number of initial period points
    npoints     - number of continuous points in jump correction
    perc        - output segment
    predV       - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
    ===============================================================================================================
    """
    def __init__(self):
        self.pmf        = True
        self.piover4    = -1.0
        self.vmin       = 1.5
        self.vmax       = 5.0
        self.tmin       = 4.0
        self.tmax       = 70.0
        self.tresh      = 20.0
        self.ffact      = 1.0
        self.taperl     = 1.0
        self.snr        = 0.2
        self.fmatch     = 1.0
        self.fhlen      = 0.008
        self.nfin       = 64
        self.npoints    = 3
        self.perc       = 50
        self.predV      = np.array([])
    
class aftantrace(obspy.core.trace.Trace):
    """
    aftantrace:
    A derived class inherited from obspy.core.trace.Trace. To handle surface wave dispersion analysis
    """
    
    def _init_ftanparam(self):
        self.ftanparam   = ftanParam()
        return
    
    def reverse(self):
        """Reverse the trace
        """
        self.data       = self.data[::-1]
        return
    
    def makesym(self):
        """Turn the double lagged cross-correlation data to one single lag
        """
        if abs(self.stats.sac.b+self.stats.sac.e) > self.stats.delta:
            raise ftanDataError('Not neg-pos trace!')
        if self.stats.npts%2 != 1:
            raise ftanHeaderError('Incompatible begin and end time!')
        nhalf                   = int((self.stats.npts-1)/2+1)
        neg                     = self.data[:nhalf]
        pos                     = self.data[nhalf-1:self.stats.npts]
        neg                     = neg[::-1]
        self.data               = (pos+neg)/2 
        # self.stats.npts         = nhalf
        self.stats.starttime    = self.stats.starttime+self.stats.sac.e
        self.stats.sac.b        = 0.
        return

    def getneg(self):
        """Get the negative lag of a cross-correlation record
        """
        if abs(self.stats.sac.b+self.stats.sac.e) > self.stats.delta:
            raise ftanDataError('Not neg-pos trace!')
        negTr                   = self.copy()
        t                       = self.stats.starttime
        L                       = (int)((self.stats.npts-1)/2)+1
        negTr.data              = negTr.data[:L]
        negTr.data              = negTr.data[::-1]
        negTr.stats.npts        = L
        negTr.stats.sac.b       = 0.
        negTr.stats.starttime   = t-self.stats.sac.b
        return negTr

    def getpos(self):
        """Get the positive lag of a cross-correlation record
        """
        if abs(self.stats.sac.b+self.stats.sac.e)>self.stats.delta:
            raise ftanDataError('Not neg-pos trace!')
        posTr                   = self.copy()
        t                       = self.stats.starttime
        L                       = (int)((self.stats.npts-1)/2)+1
        posTr.data              = posTr.data[L-1:]
        posTr.stats.npts        = L
        posTr.stats.sac.b       = 0.
        posTr.stats.starttime   = t-self.stats.sac.b
        return posTr
    
    def aftan(self, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, tmax=30.0, tresh=20.0, ffact=1.0,
            taperl=1.0, snr=0.2, fmatch=1.0, nfin=64, npoints=3, perc=50., phvelname='', predV=np.array([])):
        """ (Automatic Frequency-Time ANalysis) aftan analysis:
        ===========================================================================================================
        Input Parameters:
        pmf        - flag for Phase-Matched-Filtered output (default: True)
        piover4    - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin       - minimal group velocity, km/s
        vmax       - maximal group velocity, km/s
        tmin       - minimal period, s
        tmax       - maximal period, s
        tresh      - treshold for jump detection, usualy = 10, need modifications
        ffact      - factor to automatic filter parameter, usualy =1
        taperl     - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        snr        - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
        fmatch     - factor to length of phase matching window
        nfin       - number of initial period points
        npoints    - number of continuous points in jump correction
        perc       - output segment
        phvelname  - predicted phase velocity file name
        predV      - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
        
        Output:
        self.ftanparam, a object of ftanParam class, to store output aftan results
        ===========================================================================================================
        """
        try:
            self.ftanparam
        except:
            self._init_ftanparam()
        # preparing for data
        # try:
        dist                        = self.stats.sac.dist
        # except:
        #     dist, az, baz           = obspy.geodetics.base.gps2dist_azimuth(self.stats.sac.evla, self.stats.sac.evlo,
        #                             self.stats.sac.stla, self.stats.sac.stlo) # distance is in m
        #     self.stats.sac.dist     = dist/1000.
        #     dist                    = dist/1000.
        if predV.size != 0:
            self.ftanparam.preflag  = True
        elif os.path.isfile(phvelname):
            predV                   = np.loadtxt(phvelname)
            self.ftanparam.preflag  = True
        else:
            warnings.warn('No predicted dispersion curve for:'+self.stats.network+'.'+self.stats.station, UserWarning, stacklevel=1)
        # basic aftan
        self._aftanpg(piover4=piover4, vmin=vmin, vmax=vmax, tmin=tmin, tmax=tmax, tresh=tresh, ffact=ffact, taperl=taperl,
                nfin=nfin, npoints=npoints, perc=perc, predV=predV)
        # phase matched filter aftan
        if pmf:
            if self.ftanparam.nfout2_1<3:
                return
            self._aftanipg(piover4=piover4, vmin=vmin, vmax=vmax, tresh=tresh, ffact=ffact, taperl=taperl,
                snr=snr, fmatch=fmatch, nfin=nfin, npoints=npoints, perc=perc, predV=predV)
        return
    
    def _aftanpg(self, piover4, vmin, vmax, tmin, tmax, tresh, ffact, taperl, nfin, npoints, perc, predV):
        """ Basic aftan analysis, internal function
        ===========================================================================================================
        Input Parameters:
        piover4    - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin       - minimal group velocity, km/s
        vmax       - maximal group velocity, km/s
        tmin       - minimal period, s
        tmax       - maximal period, s
        tresh      - treshold for jump detection, usualy = 10, need modifications
        ffact      - factor to automatic filter parameter, usualy =1
        taperl     - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        nfin       - number of initial period points
        npoints    - number of continuous points in jump correction
        perc       - output segment
        predV      - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
        ===========================================================================================================
        Output:
        self.ftanparam, a object of ftanParam class, to store output aftan results
        """
        if self.ftanparam.preflag:
            phprper = predV[:,0]
            phprvel = predV[:,1]
            nprpv   = predV[:,0].size
        else:
            nprpv   = 0
            phprper = np.array([])
            phprvel = np.array([])
        dt              = self.stats.delta
        tb              = self.stats.sac.b
        nsam            = self.stats.npts
        dist            = self.stats.sac.dist
        # alpha=ffact*20.*np.sqrt(dist/1000.)
        alpha           = ffact*20.
        # number of samples for tapering, left and right end
        ntapb           = int(round(taperl*tmax/dt))
        ntape           = int(round(tmax/dt))
        omb             = 2.0*np.pi/tmax
        ome             = 2.0*np.pi/tmin
        #===========================
        # tapering seismogram
        #===========================
        nb              = int(max(2, round((dist/vmax-tb)/dt)))
        tamp            = (nb-1)*dt+tb
        ne              = int(min(nsam, round((dist/vmin-tb)/dt)))
        nrow            = nfin
        ncol            = ne-nb+1
        tArr            = np.arange(ne-nb+1)*dt+tb
        tArr[tArr==0.]  = -1.
        vArr            = dist/tArr
        tdata           = _taper(self.data, self.stats.npts, max(nb, ntapb+1),\
                                min(ne, self.stats.npts-ntape), ntapb, ntape)
        ncorr           = min(ne, self.stats.npts)
        # prepare for FFT
        ns              = int(max(1<<(ncorr-1).bit_length(), 2**12))  # different !!!
        domega          = 2.*np.pi/ns/dt
        step            = (np.log(omb)-np.log(ome))/(nfin -1)
        omegaArr        = np.exp(np.log(ome)+np.arange(nfin)*step)
        perArr          = 2.*np.pi/omegaArr
        # FFT
        if useFFTW:
            fftdata     = np.complex64(pyfftw.interfaces.numpy_fft.fft(tdata, ns))
        else:
            fftdata     = np.complex64(np.fft.fft(tdata, ns))
        if useFFTW:
            temp_x_sp           = np.zeros(ns, dtype='complex64')
            temp_out            = np.zeros(ns, dtype='complex64')
            fftw_plan           = pyfftw.FFTW(input_array=temp_x_sp, output_array=temp_out, direction='FFTW_BACKWARD',\
                                flags=('FFTW_MEASURE', ))
            out_filter      = np.zeros(ns, dtype='complex64')
        omsArr          = np.arange(ns)*domega
        phaArr          = np.zeros((ne+3-nb, nfin))
        ampo            = np.zeros((ne+3-nb, nfin))
        amp             = np.zeros((ne+3-nb, nfin))
        #  main loop by frequency
        for k in range(nfin):
            # Gaussian filter
            filterS     = _aftan_gaussian_filter(alpha, omegaArr[k], ns, fftdata, np.float32(omsArr))
            if useFFTW:
                fftw_plan.update_arrays(filterS, out_filter)
                fftw_plan.execute()
                filterT = out_filter/ns
            else:
                filterT = np.fft.ifft(filterS, ns)
            # need to multiply by 2 due to zero padding of negative frequencies
            # but NO NEED to divide by ns due to the difference of numpy style and FFTW style
            filterT     = 2.*filterT 
            phaArr[:,k] = np.arctan2(np.imag(filterT[nb-2:ne+1]), np.real(filterT[nb-2:ne+1]))
            ampo[:,k]   = np.abs(filterT[nb-2:ne+1])
            amp[:,k]    = 20.*np.log10(ampo[:,k])
        # normalization amp diagram to 100 Db with three decade cutting
        amax        = amp.max()
        amp         = amp+100.-amax
        amp[amp<40.]= 40.
        # frequency time analysis
        tvis1, ampgr1, grvel1, snr1, wdth1, phgr1, ftrig1, tvis2, ampgr2, grvel2, snr2, wdth2, phgr2, per2 = \
            _freq_time_analysis(np.float32(tb), np.int64(nfin), np.int64(ne+3-nb), np.int64(nb), np.float32(amp), \
                np.float32(ampo), np.float32(phaArr), np.float32(omegaArr), np.float32(perArr), np.float32(dist), np.float32(dt),\
                    np.float32(piover4), np.float32(tresh), np.int64(npoints))
        per1    = perArr
        nfout1  = nfin
        nfout2  = per2.size
        if nfout2 != nfout1:
            ierr= 1
        else:
            ierr= 0
        #==============================
        # fill out output data arrays
        #==============================
        if nprpv != 0:
            phV1                    = _phtovel(dist = dist, per=tvis1, U=grvel1, pha=phgr1, npr=nprpv, prper=phprper, prvel=phprvel)
            amp1                    = 10.**( (ampgr1-100.+amax) /20.)
            self.ftanparam.nfout1_1 = nfout1
            arr1_1                  = np.concatenate((per1, tvis1, grvel1, phV1, ampgr1, ftrig1, snr1, wdth1, amp1))
            self.ftanparam.arr1_1   = arr1_1.reshape(9, per1.size)
            if nfout2 != 0:
                phV2                    = _phtovel(dist = dist, per=tvis2, U=grvel2, pha=phgr2, npr=nprpv, prper=phprper, prvel=phprvel)
                amp2                    = 10.**( (ampgr2-100.+amax) /20.)
                self.ftanparam.nfout2_1 = nfout2
                arr2_1                  = np.concatenate((per2, tvis2, grvel2, phV2, ampgr2, snr2, wdth2, amp2))
                self.ftanparam.arr2_1   = arr2_1.reshape(8, per2.size)
        else:
            amp1                    = 10.**( (ampgr1-100.+amax) /20.)
            self.ftanparam.nfout1_1 = nfout1
            arr1_1                  = np.concatenate((per1, tvis1, grvel1, phgr1, ampgr1, ftrig1, snr1, wdth1, amp1))
            self.ftanparam.arr1_1   = arr1_1.reshape(9, per1.size)
            if nfout2 != 0:
                amp2                    = 10.**( (ampgr2-100.+amax) /20.)
                self.ftanparam.nfout2_1 = nfout2
                arr2_1                  = np.concatenate((per2, tvis2, grvel2, phgr2, ampgr2, snr2, wdth2, amp2))
                self.ftanparam.arr2_1   = arr2_1.reshape(8, per2.size)
        self.ftanparam.ampo_1                       = amp
        self.ftanparam.ncol_1, self.ftanparam.nrow_1= amp.shape
        if nfout2 < perc*nfin/100:
            self.ftanparam.ierr_1   = 2
        else:
            self.ftanparam.ierr_1   = ierr
        self.ftanparam.tamp_1       = (nb-1)*dt+tb
        return
    
    def _aftanipg(self, piover4, vmin, vmax, tresh, ffact, taperl, snr, fmatch, nfin, npoints, perc, predV):
        """ (Automatic Frequency-Time ANalysis) aftan analysis:
        ===========================================================================================================
        Input Parameters:
        piover4    - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin       - minimal group velocity, km/s
        vmax       - maximal group velocity, km/s
        tmin       - minimal period, s
        tmax       - maximal period, s
        tresh      - treshold for jump detection, usualy = 10, need modifications
        ffact      - factor to automatic filter parameter, usualy =1
        taperl     - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        snr        - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
        fmatch     - factor to length of phase matching window
        nfin       - number of initial period points
        npoints    - number of continuous points in jump correction
        perc       - output segment
        predV      - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
        ===========================================================================================================
        Output:
        self.ftanparam, a object of ftanParam class, to store output aftan results
        """
        if self.ftanparam.preflag:
            phprper = predV[:,0]
            phprvel = predV[:,1]
            nprpv   = predV[:,0].size
        else:
            nprpv   = 0
            phprper = np.array([])
            phprvel = np.array([])
        tmin    = self.ftanparam.arr2_1[1,0]
        tmax    = self.ftanparam.arr2_1[1,-1]
        dt      = self.stats.delta
        tb      = self.stats.sac.b
        nsam    = self.stats.npts
        dist    = self.stats.sac.dist
        # end of preparing data
        # alpha=ffact*20.*np.sqrt(dist/1000.)
        alpha   = ffact*20.
        # number of samples for tapering, left and right end
        ntapb   = int(round(taperl*tmax/dt))
        ntape   = int(round(tmax/dt))
        omb     = 2.0*np.pi/tmax
        ome     = 2.0*np.pi/tmin
        #===========================
        # tapering seismogram
        #===========================
        nb      = int(max(2, round((dist/vmax-tb)/dt)))
        tamp    = (nb-1)*dt+tb
        ne      = int(min(nsam, round((dist/vmin-tb)/dt)))
        nrow    = nfin
        ncol    = ne-nb+1
        tArr    = np.arange(ne-nb+1)*dt+tb
        tArr[tArr==0.]  = -1.
        vArr            = dist/tArr
        tdata           = _taper(self.data, self.stats.npts, max(nb, ntapb+1),\
                                min(ne, self.stats.npts-ntape), ntapb, ntape)
        ncorr           = min(ne, self.stats.npts)
        ns              = int(max(1<<(ncorr-1).bit_length(), 2**12))  # different !!!
        domega          = 2.*np.pi/ns/dt
        step            = (np.log(omb)-np.log(ome))/(nfin -1)
        omegaArr        = np.exp(np.log(ome)+np.arange(nfin)*step)
        perArr          = 2.*np.pi/omegaArr
        #===========================
        # Phase Matched Filtering
        #===========================
        # FFT
        if useFFTW:
            fftdata     = np.complex64(pyfftw.interfaces.numpy_fft.fft(tdata, ns))
            temp_x_sp   = np.zeros(ns, dtype='complex64')
            temp_out    = np.zeros(ns, dtype='complex64')
            fftw_plan   = pyfftw.FFTW(input_array=temp_x_sp, output_array=temp_out, direction='FFTW_BACKWARD',\
                                flags=('FFTW_MEASURE', ))
            out_filter  = np.zeros(ns, dtype='complex64')
        else:
            fftdata     = np.complex64(np.fft.fft(tdata, ns))
        # spectral taperring
        omstart, inds, inde, omdom, ampdom  = _tapers(np.float32(omb), np.float32(ome), np.float32(domega), np.float32(alpha), np.int64(ns) )
        omstart                             = float(round(omstart/domega))*domega
        inde                                = min(inde, ns/2+2)
        pha_cor                             = np.zeros(ns, dtype='complex64')
        # integrate to get the phase
        ind                                 = np.where(omdom >= np.sqrt(omb*ome))[0][0]
        omega_avg                           = omdom[ind]
        tg0, cubicspline                    = self._pred_cur(omega_avg)
        yval                                = cubicspline(omdom)
        # trapezoid integral is applied, this is different from the original version
        Cintgrl                             = scipy.integrate.trapz(yval[:(ind+1)], x = omdom[:(ind+1)])
        pha_cor.real[1:]                    = scipy.integrate.cumtrapz(yval, x = omdom) - Cintgrl
        pha_cor.real[0]                     = -Cintgrl
        #=== OLD ===
        # # # tg0, cubicspline                    = self._pred_cur(np.sqrt(omb*ome))
        # # # for i in range(ns):
        # # #     pha_cor.real[i]     = cubicspline.integrate(a=np.sqrt(omb*ome), b=omdom[i])
        #===========
        pha_cor.real[:inds-1]   = 0.
        pha_cor.real[inde:]     = 0.
        dci                     = 0+1j
        pha_cor                 = np.exp(dci*pha_cor)
        fftdata                 = fftdata*pha_cor*ampdom
        if useFFTW:
            fftw_plan.update_arrays(fftdata, out_filter)
            fftw_plan.execute()
            env = out_filter/ns
        else:
            env = np.fft.ifft(fftdata, ns)
        env     = 2.*env
        dw      = omegaArr[0]-omegaArr[-1]
        t_env   = _tgauss(dt = dt, t0 = tb, fsnr=snr, gt0=tg0, dw=dw, n=ns, fmatch=fmatch, seis=env)
        #==============================
        # End of phase matched filter
        #==============================
        if useFFTW:
            fftdata = np.complex64(pyfftw.interfaces.numpy_fft.fft(t_env, ns))
        else:
            fftdata = np.complex64(np.fft.fft(t_env, ns))
        fftdata = fftdata/pha_cor
        omsArr  = np.arange(ns)*domega
        phaArr  = np.zeros((ne+3-nb, nfin), dtype=np.float32)
        ampo    = np.zeros((ne+3-nb, nfin), dtype=np.float32)
        amp     = np.zeros((ne+3-nb, nfin), dtype=np.float32)
        #  main loop by frequency
        for k in range(nfin):
            # Gaussian filter
            filterS     = _aftan_gaussian_filter(np.float32(alpha), np.float32(omegaArr[k]), np.int64(ns), np.complex64(fftdata), np.float32(omsArr))
            if useFFTW:
                fftw_plan.update_arrays(filterS, out_filter)
                fftw_plan.execute()
                filterT = out_filter/ns
            else:
                filterT = np.fft.ifft(filterS, ns)
            # need to multiply by 2 due to zero padding of negative frequencies
            # but NO NEED to divide by ns due to the difference of numpy style and FFTW style
            filterT     = 2.*filterT 
            phaArr[:,k] = np.arctan2(np.imag(filterT[nb-2:ne+1]), np.real(filterT[nb-2:ne+1]))
            ampo[:,k]   = np.abs(filterT[nb-2:ne+1])
            amp[:,k]    = 20.*np.log10(ampo[:,k])
        # normalization amp diagram to 100 Db with three decade cutting
        amax        = amp.max()
        amp         = amp+100.-amax
        amp[amp<40.]= 40.
        # frequency time analysis
        tvis1, ampgr1, grvel1, snr1, wdth1, phgr1, ftrig1, tvis2, ampgr2, grvel2, snr2, wdth2, phgr2, per2 = \
            _freq_time_analysis(np.float32(tb), np.int64(nfin), np.int64(ne+3-nb), np.int64(nb), np.float32(amp), \
                np.float32(ampo), np.float32(phaArr), np.float32(omegaArr), np.float32(perArr), np.float32(dist), np.float32(dt),\
                    np.float32(piover4), np.float32(tresh), np.int64(npoints))
        per1    = perArr
        nfout1  = nfin
        nfout2  = per2.size
        if nfout2 != nfout1:
            ierr= 1
        else:
            ierr= 0
        #==============================
        # fill out output data arrays
        #==============================
        if nprpv != 0:
            phV1                    = _phtovel(dist = dist, per=tvis1, U=grvel1, pha=phgr1, npr=nprpv, prper=phprper, prvel=phprvel)
            amp1                    = 10.**( (ampgr1-100.+amax) /20.)
            self.ftanparam.nfout1_2 = nfout1
            arr1_2                  = np.concatenate((per1, tvis1, grvel1, phV1, ampgr1, ftrig1, snr1, wdth1, amp1))
            self.ftanparam.arr1_2   = arr1_2.reshape(9, per1.size)
            if nfout2 != 0:
                phV2                    = _phtovel(dist = dist, per=tvis2, U=grvel2, pha=phgr2, npr=nprpv, prper=phprper, prvel=phprvel)
                amp2                    = 10.**( (ampgr2-100.+amax) /20.)
                self.ftanparam.nfout2_2 = nfout2
                arr2_2                  = np.concatenate((per2, tvis2, grvel2, phV2, ampgr2, snr2, wdth2, amp2))
                self.ftanparam.arr2_2   = arr2_2.reshape(8, per2.size)
        else:
            amp1                    = 10.**( (ampgr1-100.+amax) /20.)
            self.ftanparam.nfout1_2 = nfout1
            arr1_2                  = np.concatenate((per1, tvis1, grvel1, phgr1, ampgr1, ftrig1, snr1, wdth1, amp1))
            self.ftanparam.arr1_2   = arr1_2.reshape(9, per1.size)
            if nfout2 != 0:
                amp2                    = 10.**( (ampgr2-100.+amax) /20.)
                self.ftanparam.nfout2_2 = nfout2
                arr2_2                  = np.concatenate((per2, tvis2, grvel2, phgr2, ampgr2, snr2, wdth2, amp2))
                self.ftanparam.arr2_2   = arr2_2.reshape(8, per2.size)
        self.ftanparam.ampo_2                       = amp
        self.ftanparam.ncol_2, self.ftanparam.nrow_2= amp.shape
        if nfout2 < perc*nfin/100:
            self.ftanparam.ierr_2   = 2
        else:
            self.ftanparam.ierr_2   = ierr
        self.ftanparam.tamp_2       = (nb-1)*dt+tb
        return
    
    def _compare_arr(self, data1, data2):
        plt.plot(data1-data2, '-y')
        plt.plot(data1, '^r')
        plt.plot(data2, '*b')
        plt.show()
        return
    
    def _compare_arr2(self, d1, d2, number):
        data1   = d1[:, number]
        data2   = d2[:data1.size, number]
        plt.plot(data1-data2, '-y')
        plt.plot(data1, '^r')
        plt.plot(data2, '*b')
        plt.show()
        return

    def _pred_cur(self, om0):
        """create phase prediction curve by group velocity, will be used for phase matched filter
        """        
        pred    = self.ftanparam.arr2_1[1:3,:]
        dist    = self.stats.sac.dist
        x       = 2*np.pi/pred[0,::-1]
        y       = dist/pred[1,::-1]
        ind_x   = np.argsort(x)
        x       = x[ind_x]
        y       = y[ind_x]
        spl     = scipy.interpolate.CubicSpline(x, y)
        gt0     = spl(om0)
        y       = y - gt0
        spl     = scipy.interpolate.CubicSpline(x, y)
        return gt0, spl
        
    def aftanf77(self, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, tmax=30.0, tresh=20.0,
            ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0, nfin=64, npoints=3, perc=50., phvelname='', predV=np.array([])):
        """ (Automatic Frequency-Time ANalysis) aftan analysis:
        ===========================================================================================================
        Input Parameters:
        pmf        - flag for Phase-Matched-Filtered output (default: True)
        piover4    - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin       - minimal group velocity, km/s
        vmax       - maximal group velocity, km/s
        tmin       - minimal period, s
        tmax       - maximal period, s
        tresh      - treshold for jump detection, usualy = 10, need modifications
        ffact      - factor to automatic filter parameter, usualy =1
        taperl     - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        snr        - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
        fmatch     - factor to length of phase matching window
        nfin       - number of initial period points
        npoints    - number of continuous points in jump correction
        perc       - output segment
        phvelname  - predicted phase velocity file name
        predV      - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
        
        Output:
        self.ftanparam, a object of ftanParam class, to store output aftan results
        ===========================================================================================================
        """
        try:
            self.ftanparam
        except:
            self._init_ftanparam()
        if not isaftanf77:
            raise AttributeError('fortran77 aftan not imported correctly!')
        # preparing for data
        # try:
        dist                    = self.stats.sac.dist
        # except:
        #     dist, az, baz       = obspy.geodetics.base.gps2dist_azimuth(self.stats.sac.evla, self.stats.sac.evlo,
        #                             self.stats.sac.stla, self.stats.sac.stlo) # distance is in m
        #     self.stats.sac.dist = dist/1000.
        #     dist                = dist/1000.
        nprpv                   = 0
        phprper                 = np.zeros(300)
        phprvel                 = np.zeros(300)
        if predV.size != 0:
            phprper                 = predV[:,0]
            phprvel                 = predV[:,1]
            nprpv                   = predV[:,0].size
            phprper                 = np.append( phprper, np.zeros(300-phprper.size) )
            phprvel                 = np.append( phprvel, np.zeros(300-phprvel.size) )
            self.ftanparam.preflag  = True
        elif os.path.isfile(phvelname) :
            # print 'Using prefile:',phvelname
            php                     = np.loadtxt(phvelname)
            phprper                 = php[:,0]
            phprvel                 = php[:,1]
            nprpv                   = php[:,0].size
            phprper                 = np.append( phprper, np.zeros(300-phprper.size) )
            phprvel                 = np.append( phprvel, np.zeros(300-phprvel.size) )
            self.ftanparam.preflag  = True
        else:
            warnings.warn('No predicted dispersion curve for:'+self.stats.network+'.'+self.stats.station, UserWarning, stacklevel=1)
        tempsac                     = self.copy()
        tb                          = self.stats.sac.b
        length                      = len(tempsac.data)
        if length>32768:
            warnings.warn('Length of seismogram is larger than 32768!', UserWarning, stacklevel=1)
            nsam                    = 32768
            tempsac.data            = tempsac.data[:nsam]
            tempsac.stats.e         = (nsam-1)*tempsac.stats.delta+tb
            sig                     = tempsac.data
        else:
            sig                     = np.append(tempsac.data, np.zeros( 32768-tempsac.data.size, dtype='float64' ) )
            # # # nsam                    = int( float (tempsac.stats.npts) )### for unknown reasons, this has to be done, nsam=int(tempsac.stats.npts)  won't work as an input for aftan
            nsam                    = np.int32(tempsac.stats.npts)###  nsam=int(tempsac.stats.npts)  won't work as an input for aftan
        dt                          = tempsac.stats.delta
        # Start to do aftan utilizing fortran 77 aftan
        self.ftanparam.nfout1_1,self.ftanparam.arr1_1,self.ftanparam.nfout2_1,self.ftanparam.arr2_1,self.ftanparam.tamp_1, \
                self.ftanparam.nrow_1,self.ftanparam.ncol_1,self.ftanparam.ampo_1, self.ftanparam.ierr_1= aftan.aftanpg(piover4, nsam, \
                    sig, tb, dt, dist, vmin, vmax, tmin, tmax, tresh, ffact, perc, npoints, taperl, nfin, snr, nprpv, phprper, phprvel)
        if pmf:
            if self.ftanparam.nfout2_1<3:
                return
            npred           = self.ftanparam.nfout2_1
            tmin2           = self.ftanparam.arr2_1[1,0]
            tmax2           = self.ftanparam.arr2_1[1,self.ftanparam.nfout2_1-1]
            pred            = np.zeros((2,300))
            pred[:,0:100]   = self.ftanparam.arr2_1[1:3,:]
            pred            = pred.T
            self.ftanparam.nfout1_2,self.ftanparam.arr1_2,self.ftanparam.nfout2_2,self.ftanparam.arr2_2,self.ftanparam.tamp_2, \
                    self.ftanparam.nrow_2,self.ftanparam.ncol_2,self.ftanparam.ampo_2, self.ftanparam.ierr_2 = aftan.aftanipg(piover4,nsam, \
                        sig,tb,dt,dist,vmin,vmax,tmin2,tmax2,tresh,ffact,perc,npoints,taperl,nfin,snr,fmatch,npred,pred,nprpv,phprper,phprvel)
        return

    def plotftan(self, plotflag=3, sacname='', cmap='ftan'):
        """
        Plot ftan diagram:
        This function plot ftan diagram.
        ====================================================================
        Input Parameters:
        plotflag -
            0: only Basic FTAN
            1: only Phase Matched Filtered FTAN
            2: both
            3: both in one figure
        sacname - sac file name than can be used as the title of the figure
        ====================================================================
        """
        if cmap == 'ftan':
            try:
                import pycpt
                cmap    = pycpt.load.gmtColormap('./FTAN.cpt')
            except:
                cmap    = 'gist_rainbow'
        try:
            fparam  = self.ftanparam
            if fparam.nfout1_1 == 0:
                return "Error: No Basic FTAN parameters!"
            dt          = self.stats.delta
            dist        = self.stats.sac.dist
            if (plotflag!=1 and plotflag!=3):
                v1      = dist/(fparam.tamp_1+np.arange(fparam.ncol_1)*dt)
                ampo_1  = fparam.ampo_1[:fparam.ncol_1,:fparam.nrow_1]
                obper1_1= fparam.arr1_1[1,:fparam.nfout1_1]
                gvel1_1 = fparam.arr1_1[2,:fparam.nfout1_1]
                phvel1_1= fparam.arr1_1[3,:fparam.nfout1_1]
                plt.figure()
                ax      = plt.subplot()
                p       = plt.pcolormesh(obper1_1, v1, ampo_1, cmap=cmap,shading='gouraud')
                ax.plot(obper1_1, gvel1_1, '--k', lw=3) #
                if fparam.preflag:
                    ax.plot(obper1_1, phvel1_1, '--w', lw=3) #
                if (fparam.nfout2_1!=0):
                    obper2_1    = fparam.arr2_1[1,:fparam.nfout2_1]
                    gvel2_1     = fparam.arr2_1[2,:fparam.nfout2_1]
                    phvel2_1    = fparam.arr2_1[3,:fparam.nfout2_1]
                    ax.plot(obper2_1, gvel2_1, '-k', lw=3) #
                    if fparam.preflag:
                        ax.plot(obper2_1, phvel2_1, '-w', lw=3) #
                cb      = plt.colorbar(p, ax=ax)
                Tmin1   = obper1_1[0]
                Tmax1   = obper1_1[fparam.nfout1_1-1]
                vmin1   = v1[fparam.ncol_1-1]
                vmax1   = v1[0]
                plt.axis([Tmin1, Tmax1, vmin1, vmax1])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('Basic FTAN Diagram '+sacname,fontsize=15)
            if fparam.nfout1_2 == 0 and plotflag != 0:
                return "Error: No PMF FTAN parameters!"
            if (plotflag!=0 and plotflag!=3):
                v2      = dist/(fparam.tamp_2+np.arange(fparam.ncol_2)*dt)
                ampo_2  = fparam.ampo_2[:fparam.ncol_2,:fparam.nrow_2]
                obper1_2= fparam.arr1_2[1,:fparam.nfout1_2]
                gvel1_2 = fparam.arr1_2[2,:fparam.nfout1_2]
                phvel1_2= fparam.arr1_2[3,:fparam.nfout1_2]
                plt.figure()
                ax      = plt.subplot()
                p       = plt.pcolormesh(obper1_2, v2, ampo_2, cmap=cmap,shading='gouraud')
                ax.plot(obper1_2, gvel1_2, '--k', lw=3) #
                if fparam.preflag:
                    ax.plot(obper1_2, phvel1_2, '--w', lw=3) #
                if (fparam.nfout2_2!=0):
                    obper2_2    = fparam.arr2_2[1,:fparam.nfout2_2]
                    gvel2_2     = fparam.arr2_2[2,:fparam.nfout2_2]
                    phvel2_2    = fparam.arr2_2[3,:fparam.nfout2_2]
                    ax.plot(obper2_2, gvel2_2, '-k', lw=3) #
                    if fparam.preflag:
                        ax.plot(obper2_2, phvel2_2, '-w', lw=3) #
                cb      = plt.colorbar(p, ax=ax)
                Tmin2   = obper1_2[0]
                Tmax2   = obper1_2[fparam.nfout1_2-1]
                vmin2   = v2[fparam.ncol_2-1]
                vmax2   =v2[0]
                plt.axis([Tmin2, Tmax2, vmin2, vmax2])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('PMF FTAN Diagram '+sacname,fontsize=15)
            if ( plotflag==3 ):
                v1      = dist/(fparam.tamp_1+np.arange(fparam.ncol_1)*dt)
                ampo_1  = fparam.ampo_1[:fparam.ncol_1,:fparam.nrow_1]
                obper1_1= fparam.arr1_1[1,:fparam.nfout1_1]
                gvel1_1 = fparam.arr1_1[2,:fparam.nfout1_1]
                phvel1_1= fparam.arr1_1[3,:fparam.nfout1_1]
                plt.figure(num=None, figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                ax      = plt.subplot(2,1,1)
                p       = plt.pcolormesh(obper1_1, v1, ampo_1, cmap=cmap,shading='gouraud')
                ax.plot(obper1_1, gvel1_1, '--k', lw=3) #
                if fparam.preflag:
                    ax.plot(obper1_1, phvel1_1, '--w', lw=3) #
                if (fparam.nfout2_1!=0):
                    obper2_1    = fparam.arr2_1[1,:fparam.nfout2_1]
                    gvel2_1     = fparam.arr2_1[2,:fparam.nfout2_1]
                    phvel2_1    = fparam.arr2_1[3,:fparam.nfout2_1]
                    ax.plot(obper2_1, gvel2_1, '-k', lw=3) #
                    if fparam.preflag:
                        ax.plot(obper2_1, phvel2_1, '-w', lw=3) #
                cb      = plt.colorbar(p, ax=ax)
                Tmin1   = obper1_1[0]
                Tmax1   = obper1_1[fparam.nfout1_1-1]
                vmin1   = v1[fparam.ncol_1-1]
                vmax1   = v1[0]
                plt.axis([Tmin1, Tmax1, vmin1, vmax1])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('Basic FTAN Diagram '+sacname)
                v2      = dist/(fparam.tamp_2+np.arange(fparam.ncol_2)*dt)
                ampo_2  = fparam.ampo_2[:fparam.ncol_2,:fparam.nrow_2]
                obper1_2= fparam.arr1_2[1,:fparam.nfout1_2]
                gvel1_2 = fparam.arr1_2[2,:fparam.nfout1_2]
                phvel1_2= fparam.arr1_2[3,:fparam.nfout1_2]

                ax  = plt.subplot(2,1,2)
                p   = plt.pcolormesh(obper1_2, v2, ampo_2, cmap=cmap,shading='gouraud')
                ax.plot(obper1_2, gvel1_2, '--k', lw=3) #
                if fparam.preflag:
                    ax.plot(obper1_2, phvel1_2, '--w', lw=3) #

                if (fparam.nfout2_2!=0):
                    obper2_2= fparam.arr2_2[1,:fparam.nfout2_2]
                    gvel2_2 = fparam.arr2_2[2,:fparam.nfout2_2]
                    phvel2_2= fparam.arr2_2[3,:fparam.nfout2_2]
                    ax.plot(obper2_2, gvel2_2, '-k', lw=3) #
                    if fparam.preflag:
                        ax.plot(obper2_2, phvel2_2, '-w', lw=3) #
                cb      = plt.colorbar(p, ax=ax)
                Tmin2   = obper1_2[0]
                Tmax2   = obper1_2[fparam.nfout1_2-1]
                vmin2   = v2[fparam.ncol_2-1]
                vmax2   = v2[0]
                plt.axis([Tmin2, Tmax2, vmin2, vmax2])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('PMF FTAN Diagram '+sacname)
        except AttributeError:
            raise  ftanHeaderError('FTAN Parameters are not available!')
        return
    
    def get_snr(self, ffact=1.):
        """ compute snr of the waveform
        """
        fparam      = self.ftanparam
        dist        = self.stats.sac.dist
        begT        = self.stats.sac.b
        endT        = self.stats.sac.e
        dt          = self.stats.delta
        if fparam.nfout2_2!=0:
            o_per   = fparam.arr2_2[1,:]
            g_vel   = fparam.arr2_2[2,:]
            snrArr  = np.ones(o_per.size, dtype=np.float64)*-1.
            for i in range(fparam.nfout2_2):
                if g_vel[i]<0 or o_per[i]<0:
                    continue
                filtered_tr = self.gaussian_filter_aftan(1./o_per[i], ffact=ffact)
                minT        = dist/g_vel[i]-o_per[i]/2.
                maxT        = dist/g_vel[i]+o_per[i]/2.
                if(minT<begT):
                    minT    = begT
                if(maxT>endT):
                    maxT    = endT
                # Noise window
                minT        = maxT + o_per[i] * 5. + 500.
                skipflag    = False
                if( (endT - minT) < 50. ):
                    skipflag= True
                elif( (endT - minT) < 1100. ):
                    maxT    = endT - 10.
                else:
                    minT    = endT - 1100.
                    maxT    = endT - 100.
                if not skipflag:
                    ib              = (int)((minT-begT)/dt)
                    ie              = (int)((maxT-begT)/dt)+2
                    tempnoise       = filtered_tr[ib:ie]
                    noiserms        = np.sqrt(( np.sum(tempnoise**2))/(ie-ib-1.) )
                    amp             = self.ftanparam.arr2_2[7,i]
                    if noiserms != 0.:
                        snrArr[i]   = amp/noiserms
                    else:
                        snrArr[i]   = 1.
            self.ftanparam.arr2_2   = np.append(fparam.arr2_2, snrArr)
            self.ftanparam.arr2_2   = self.ftanparam.arr2_2.reshape(9, o_per.size)
        return 
                
    def gaussian_filter_snr(self, fcenter, fhlen=0.008):
        """
        Gaussian filter designed for SNR analysis
        exp( (-0.5/fhlen^2)*(f-fcenter)^2 )
        ====================================================================
        Input parameters:
        fcenter - central period
        fhlen   - half length of Gaussian width
        ====================================================================
        """
        npts    = self.stats.npts
        ns      = 1<<(npts-1).bit_length()
        df      = 1.0/self.stats.delta/ns
        nhalf   = ns/2+1
        fmax    = (nhalf-1)*df
        if fcenter > fmax:
            fcenter = fmax
        alpha   = -0.5/(fhlen*fhlen)
        F       = np.arange(ns)*df
        gauamp  = F - fcenter
        sf      = np.exp(alpha*gauamp**2)
        sp      = np.fft.fft(self.data, ns)
        filtered_sp         = sf*sp
        filtered_sp[ns/2:]  = 0
        filtered_sp[0]      /= 2
        filtered_sp[ns/2-1] = filtered_sp[ns/2-1].real+0.j
        filtered_seis       = np.fft.ifft(filtered_sp, ns)
        filtered_seis       = 2.*filtered_seis[:npts].real
        return filtered_seis
    
    def gaussian_filter_aftan(self, fcenter, ffact=1.):
        """
        Gaussian filter designed for SNR analysis
        exp( (-0.5/fhlen^2)*(f-fcenter)^2 )
        ====================================================================
        Input parameters:
        fcenter - central period
        ffact   - factor to automatic filter parameter, usualy =1
        ====================================================================
        """
        npts        = self.stats.npts
        ns          = 1<<(npts-1).bit_length()
        df          = 1.0/self.stats.delta/ns
        nhalf       = ns/2+1
        fmax        = (nhalf-1)*df
        alpha       = ffact*20.
        if fcenter > fmax:
            fcenter = fmax
        omega0      = 2.*np.pi*fcenter
        omsArr      = 2.*np.pi*np.arange(ns)*df
        sp          = np.fft.fft(self.data, ns)
        filtered_sp = _aftan_gaussian_filter(np.float32(alpha), np.float32(omega0), np.int64(ns), np.complex64(sp), np.float32(omsArr))
        if useFFTW:
            filtered_seis   = pyfftw.interfaces.numpy_fft.ifft(filtered_sp, ns)
        else:
            filtered_seis   = np.fft.ifft(filtered_sp, ns)
        filtered_seis       = 2.*filtered_seis[:npts].real
        return filtered_seis
     
    def compare_aftan(self, intr):
        return np.allclose(self.ftanparam.arr1_1, intr.ftanparam.arr1_1) * \
                np.allclose(self.ftanparam.arr1_2, intr.ftanparam.arr1_2) * \
                    np.allclose(self.ftanparam.arr2_1, intr.ftanparam.arr2_1) * \
                        np.allclose(self.ftanparam.arr2_2, intr.ftanparam.arr2_2) #*\
                        #     np.allclose(self.ftanparam.ampo_1, intr.ftanparam.ampo_1) *\
                        #         np.allclose(self.ftanparam.ampo_2, intr.ftanparam.ampo_2)
    
    
    