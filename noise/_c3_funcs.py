# -*- coding: utf-8 -*-
"""
internal functions and classes for three station interferometry

:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.aftan.pyaftan as pyaftan

import numpy as np
from numba import jit, float32, int32, boolean, float64, int64
import numba
import pyfftw
import obspy
import os
import multiprocessing
import obspy
import obspy.signal.filter
import scipy.signal
import glob
import time


@numba.jit(numba.types.Tuple((numba.float32[:], numba.int64[:], numba.int64, numba.int64, numba.int64))\
        (numba.int64, numba.float32[:], numba.float32[:], numba.float32), nopython=True)
def _trigger(nf, phvel, om , tresh):
    """Detect jumps in dispersion curve
    """
    hh1             = om[1:nf-1] - om[:nf-2]
    hh2             = om[2:] - om[1:nf-1]
    hh3             = hh1 + hh2
    r               = (phvel[:nf-2]/hh1 - (1./hh1+1/hh2)*phvel[1:nf-1] + phvel[2:]/hh2)*hh3/4.*100.
    ftrig           = np.zeros(nf, dtype=np.float32)
    ftrig[1:nf-1]   = r
    # second derivative
    ftrig[:-1]      = ftrig[:-1] - ftrig[1:]
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
    ist     = 0
    ibe     = 0
    # determine the longest length
    if ierr != 0:
        for k in range(nf):
            if trig[k] != 0:
                if (k - ibe) > (ibe - ist):
                    ist     = ibe
                    ibe     = k
            else:
                if k == (nf - 1):
                    if (k - ibe) > (ibe - ist):
                        ist     = ibe
                        ibe     = k
    return ftrig, trig, ierr, ist, ibe

@numba.jit(numba.types.Tuple((numba.int64, numba.int64))(numba.int64[:]), nopython=True)
def _get_pers_ind(Nm):
    """Detect jumps in dispersion curve
    """
    nf  = Nm.size
    ind0= np.where(Nm == 0)[0]
    ist = 0
    ibe = 0
    for i in range(ind0.size):
        if i == 0:
            ist     = 0
            ibe     = ind0[i] - 1
            continue
        if (ind0[i] - ind0[i-1]) > (ibe - ist):
            ist     = ind0[i-1] + 1
            ibe     = ind0[i] - 1 
    if (nf - 1 - ind0[-1]) > (ibe - ist):
        ist = ind0[-1] + 1
        ibe = nf - 1  
    return ist, ibe

def _tshift_fft(data, dt, pers, phvel, iphase, d):
    """positive means delaying the waveform
    """
    npts    = data.size
    Np2     = int(max(1<<(npts-1).bit_length(), 2**12))
    Xf      = np.fft.rfft(data, n=Np2)
    freq    = 1./dt/Np2*np.arange((Np2/2+1), dtype = float)
    infreq  = 1./pers[::-1]
    fitp    = scipy.interpolate.interp1d(infreq, phvel[::-1], kind='linear', fill_value='extrapolate', assume_sorted=True )
    C       = fitp(freq)
    # d < 0 : wave arrives earlier than expected, need to shift to right, tshift = -d/C > 0.
    # d > 0 : vice versa
    tshift  = -d/C 
    ph_shift= np.exp(-2j*np.pi*freq*tshift - 1j*iphase)
    Xf2     = Xf*ph_shift
    return np.real(np.fft.irfft(Xf2)[:npts])

class c3_pair(object):
    """ A class for ambient noise three station interferometry
    =================================================================================================================
    ::: parameters :::
    stacode1, netcode1  - station/network code for station 1
    stacode2, netcode2  - station/network code for station 2
    
    =================================================================================================================
    """
    def __init__(self, datadir, outdir, stacode1, netcode1, stla1, stlo1, stacode2, netcode2, stla2, stlo2,\
            channel, chan_types= [], StationInv = [], alpha = 0.01, vmin = 1., vmax = 5., Tmin = 5.,\
            Tmax = 150., bfact_dw = 1., efact_dw = 1., dthresh = 5., inftan = pyaftan.InputFtanParam(), \
            basic1=True, basic2=True, pmf1=True, pmf2=True, f77=True, prephdir='', pers = [],\
            snr_thresh = 10., Ntrace_min = 5, nfmin = 5, jump_thresh = 3., phvel_ref = [], pers_ref = [], prefer_c3_disp = True):
        self.datadir    = datadir
        self.outdir     = outdir
        self.stacode1   = stacode1
        self.netcode1   = netcode1
        self.stla1      = stla1
        self.stlo1      = stlo1
        self.stacode2   = stacode2
        self.netcode2   = netcode2
        self.stla2      = stla2
        self.stlo2      = stlo2
        self.channel    = channel
        self.chan_types = chan_types # not used for now
        self.StationInv = StationInv
        # parameters for interferometry data processing
        self.alpha      = alpha
        self.vmin       = vmin
        self.vmax       = vmax
        self.Tmin       = Tmin
        self.Tmax       = Tmax
        self.bfact_dw   = bfact_dw
        self.efact_dw   = efact_dw
        self.dthresh    = dthresh
        # parameters for aftan
        self.inftan     = inftan
        self.basic1     = basic1
        self.basic2     = basic2
        self.pmf1       = pmf1
        self.pmf2       = pmf2
        self.f77        = f77
        self.prephdir   = prephdir
        if len(pers) == 0:
            self.pers   = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        else:
            self.pers   = pers
        # aftan stack
        self.snr_thresh = snr_thresh 
        self.nfmin      = nfmin
        self.Ntrace_min = Ntrace_min
        self.jump_thresh= jump_thresh
        # reference dispersion curves, used for phase correction
        if len(phvel_ref) != len(pers_ref):
            raise ValueError('length of refernce phase speed and periods must be consistent')
        self.phvel_ref  = phvel_ref
        self.pers_ref   = pers_ref
        self.prefer_c3_disp = prefer_c3_disp
        return
    
    def print_info(self, process_id):
        """print the informations of this pair
        """
        staid1          = self.netcode1 + '.' + self.stacode1
        staid2          = self.netcode2 + '.' + self.stacode2
        print ('--- '+ staid1+'_'+staid2+' processID = '+str(process_id))
    
    def direct_wave_interfere(self, process_id= '', verbose = False, verbose2= False):
        """direct wave interferometry
        """
        if verbose:
            self.print_info(process_id=process_id)
        chan1           = self.channel[0]
        chan2           = self.channel[1]
        dist0, az0, baz0= obspy.geodetics.gps2dist_azimuth(self.stla1, self.stlo1, self.stla2, self.stlo2)
        dist0           /= 1000.
        staid1          = self.netcode1 + '.' + self.stacode1
        staid2          = self.netcode2 + '.' + self.stacode2
        xcorrpattern    = self.datadir + '/COR/'+staid1+'/COR_'+staid1+'_??'+chan1+'_'+staid2+'_??'+chan2+'.SAC'
        if len(glob.glob(xcorrpattern)) > 0:
            outdir      = self.outdir + '/SYNC_C3/'+staid1
        else:
            outdir      = self.outdir + '/ASYNC_C3/'+staid1
        Ntraces         = 0
        # loop over source stations
        for srcnet in self.StationInv:
            for srcsta in srcnet:
                sourceid        = srcnet.code+'.'+srcsta.code
                if sourceid == staid1 or sourceid == staid2:
                    continue
                evla            = srcsta.latitude
                evlo            = srcsta.longitude
                dist1, az1, baz1= obspy.geodetics.gps2dist_azimuth(evla, evlo, self.stla1, self.stlo1)
                dist2, az2, baz2= obspy.geodetics.gps2dist_azimuth(evla, evlo, self.stla2, self.stlo2)
                dist1           /= 1000.
                dist2           /= 1000.
                dhyp            = dist0 - abs(dist1 - dist2)
                dell            = dist1 + dist2 - dist0
                if abs(dhyp - dell) < self.dthresh:
                    if verbose2:
                        print ('!!! SKIP c3: %s_%s source: %s' %(staid1, staid2, sourceid))
                    continue
                # not in stationary phase zone
                if min(dhyp, dell) > dist0*self.alpha:
                    continue
                if dell < dhyp:
                    iellhyp = 1
                else:
                    iellhyp = 2
                # load xcorr data
                # station 1
                if staid1 < sourceid:
                    datadir1    = self.datadir + '/COR/'+staid1
                    fpattern1   = datadir1 + '/COR_'+staid1+'_??'+chan1+'_'+sourceid+'_??'+chan1+'.SAC'
                else:
                    datadir1    = self.datadir + '/COR/'+sourceid
                    fpattern1   = datadir1 + '/COR_'+sourceid+'_??'+chan1+'_'+staid1+'_??'+chan1+'.SAC'
                flst1   = glob.glob(fpattern1)
                if len(flst1) == 0:
                    continue
                fname1          = flst1[0]
                # station 2
                if staid2 < sourceid:
                    datadir2    = self.datadir + '/COR/'+staid2
                    fpattern2   = datadir2 + '/COR_'+staid2+'_??'+chan2+'_'+sourceid+'_??'+chan2+'.SAC'
                else:
                    datadir2    = self.datadir + '/COR/'+sourceid
                    fpattern2   = datadir2 + '/COR_'+sourceid+'_??'+chan2+'_'+staid2+'_??'+chan2+'.SAC'
                flst2   = glob.glob(fpattern2)
                if len(flst2) == 0:
                    continue
                fname2          = flst2[0]
                # load data, get symmetric components
                tr1             = obspy.read(fname1)[0]
                outsactr        = obspy.io.sac.SACTrace.from_obspy_trace(tr1.copy())
                tr1             = pyaftan.aftantrace(tr1.data, tr1.stats)
                tr1.makesym()
                tr2             = obspy.read(fname2)[0]
                tr2             = pyaftan.aftantrace(tr2.data, tr2.stats)
                tr2.makesym()
                if abs(tr1.stats.delta - tr2.stats.delta) > min(tr1.stats.delta/1000., tr2.stats.delta/1000.):
                    raise AttributeError('!!! xcorr must have the same sampling rate!')
                if iellhyp == 1:
                    outdata = scipy.signal.convolve(tr1.data, tr2.data, mode='full', method='fft')
                else:
                    outdata = scipy.signal.correlate(tr1.data, tr2.data, mode='full', method='fft')
                # data
                outsactr.data   = outdata
                #==================
                # update headers
                #==================
                if iellhyp == 1:
                    outsactr.b  += tr1.stats.sac.e
                outsactr.kuser0 = self.netcode1
                outsactr.kevnm  = self.stacode1
                outsactr.knetwk = self.netcode2
                outsactr.kstnm  = self.stacode2
                outsactr.evla   = self.stla1
                outsactr.evlo   = self.stlo1
                outsactr.stla   = self.stla2
                outsactr.stlo   = self.stlo2
                outsactr.dist   = dist0
                outsactr.az     = az0
                outsactr.baz    = baz0
                # source station
                outsactr.kuser1 = srcnet.code
                outsactr.kuser2 = srcsta.code
                if iellhyp == 1:
                    outsactr.user0  = dell
                else:
                    outsactr.user0  = -dhyp
                outsactr.user1  = srcsta.latitude
                outsactr.user2  = srcsta.longitude
                outsactr.kcmpnm = self.channel
                # save data
                if not os.path.isdir(outdir):
                    try:
                        os.makedirs(outdir)
                    except OSError:
                        i   = 0
                        while(i < 10):
                            sleep_time  = np.random.random()/10.
                            time.sleep(sleep_time)
                            if not os.path.isdir(outdir):
                                try:
                                    os.makedirs(outdir)
                                    break
                                except OSError:
                                    pass
                            i   += 1
                if iellhyp == 1:
                    outfname= outdir + '/C3_'+ staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_'+sourceid+'_ELL.SAC'
                else:
                    outfname= outdir + '/C3_'+ staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_'+sourceid+'_HYP.SAC'
                outsactr.write(outfname)
                Ntraces += 1
        return Ntraces
    
    def direct_wave_aftan(self, process_id= '', verbose = False):
        """direct wave aftan
        """
        inftan          = self.inftan
        if verbose:
            self.print_info(process_id = process_id)
        chan1           = self.channel[0]
        chan2           = self.channel[1]
        staid1          = self.netcode1 + '.' + self.stacode1
        staid2          = self.netcode2 + '.' + self.stacode2
        if not os.path.isdir(self.datadir + '/logs_dw_aftan/'+ staid1):
            try:
                os.makedirs(self.datadir + '/logs_dw_aftan/'+ staid1)
            except OSError:
                i   = 0
                while(i < 10):
                    sleep_time  = np.random.random()/10.
                    time.sleep(sleep_time)
                    if not os.path.isdir(self.datadir + '/logs_dw_aftan/'+ staid1):
                        try:
                            os.makedirs(self.datadir + '/logs_dw_aftan/'+ staid1)
                            break
                        except OSError:
                            pass
                    i   += 1
        logfname    = self.datadir + '/logs_dw_aftan/'+ staid1 + '/' + staid1 +'_'+staid2+'.log'
        with open(logfname, 'w') as fid:
            fid.writelines('RUNNING\n')
        if len(glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*ELL.SAC')) > 0 or \
            len(glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*HYP.SAC')) > 0:
            is_sync     = True
        elif len(glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*ELL.SAC')) > 0 or \
            len(glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*HYP.SAC')) > 0:
            is_sync     = False
        else:
            with open(logfname, 'w') as fid:
                fid.writelines('NODATA\n')
            return 
        dist0, az0, baz0= obspy.geodetics.gps2dist_azimuth(self.stla1, self.stlo1, self.stla2, self.stlo2)
        dist0           /= 1000.
        if is_sync:
            ellflst     = glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*ELL.SAC')
            hyplst      = glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*HYP.SAC')
        else:
            ellflst     = glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*ELL.SAC')
            hyplst      = glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*HYP.SAC')
        # source station in elliptical stationary phase zone
        ell_piover4     = -2.
        for ellfname in ellflst:
            elltr                   = obspy.read(ellfname)[0]
            ell_atr                 = pyaftan.aftantrace(elltr.data, elltr.stats)
            ell_atr.stats.sac.dist  = dist0 + ell_atr.stats.sac.user0 # distance correction
            phvelname               = self.prephdir + "/%s.%s.pre" %(staid1, staid2)
            if not os.path.isfile(phvelname):
                print ('*** WARNING: '+ phvelname+' not exists!')
                continue
            # aftan analysis
            if self.f77:
                ell_atr.aftanf77(pmf=inftan.pmf, piover4 = ell_piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                    tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                        npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
            else:
                ell_atr.aftan(pmf=inftan.pmf, piover4 = ell_piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                    tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                        npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
            # SNR
            ell_atr.get_snr(ffact = inftan.ffact)
            # save aftan
            outdispfname            = ellfname[:-4] + '.npz'
            outarr                  = np.array([dist0, ell_atr.stats.sac.user0])
            ell_atr.ftanparam.write_npy(outfname = outdispfname, outarr = outarr)
        # source station in hypobolic stationary phase zone
        hyp_piover4     = 0.
        for hypfname in hyplst:
            hyptr                   = obspy.read(hypfname)[0]
            hyp_atr                 = pyaftan.aftantrace(hyptr.data, hyptr.stats)
            hyp_atr.makesym()
            hyp_atr.stats.sac.dist  = dist0 + hyp_atr.stats.sac.user0 # distance correction
            phvelname               = self.prephdir + "/%s.%s.pre" %(staid1, staid2)
            if not os.path.isfile(phvelname):
                print ('*** WARNING: '+ phvelname+' not exists!')
                continue
            # aftan analysis
            if self.f77:
                hyp_atr.aftanf77(pmf=inftan.pmf, piover4 = hyp_piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                    tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                        npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
            else:
                hyp_atr.aftan(pmf=inftan.pmf, piover4 = hyp_piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                    tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                        npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
            # SNR
            hyp_atr.get_snr(ffact = inftan.ffact)
            # save aftan
            outdispfname            = hypfname[:-4] + '.npz'
            outarr                  = np.array([dist0, hyp_atr.stats.sac.user0])
            hyp_atr.ftanparam.write_npy(outfname = outdispfname, outarr = outarr)
        with open(logfname, 'w') as fid:
            fid.writelines('SUCCESS\n')
        # # # if len(ellflst) > 0 or len(hyplst) > 0:
        # # #     with open(logfname, 'w') as fid:
        # # #         fid.writelines('SUCCESS\n')
        # # # else:
        # # #     with open(logfname, 'w') as fid:
        # # #         fid.writelines('NODATA\n')
        return 
    
    def direct_wave_stack_disp(self, process_id= '', verbose = False):
        """stack dispersion results
        """
        if verbose:
            self.print_info(process_id = process_id)
        chan1           = self.channel[0]
        chan2           = self.channel[1]
        staid1          = self.netcode1 + '.' + self.stacode1
        staid2          = self.netcode2 + '.' + self.stacode2
        if len(glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*.npz')) > 0:
            is_sync     = True
            npzfilelst  = glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*.npz')
        elif len(glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*.npz')) > 0:
            is_sync     = False
            npzfilelst  = glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*.npz')
        else:
            return
        dist0, az0, baz0= obspy.geodetics.gps2dist_azimuth(self.stla1, self.stlo1, self.stla2, self.stlo2)
        dist0           /= 1000.
        #======================
        # load raw aftan data
        #======================
        raw_pers        = []
        raw_grvel       = []
        raw_phvel       = []
        raw_snr         = []
        distances       = []
        Tmin            = 999.
        Tmax            = -999.
        Naftan          = 0
        for npzfname in npzfilelst:
            fparam      = pyaftan.ftanParam()
            outarr      = fparam.load_npy(npzfname)
            if fparam.nfout2_2 == 0:
                continue
            if np.any(np.isnan(fparam.arr2_2[8, :fparam.nfout2_2])) or np.any(np.isnan(fparam.arr2_2[3, :fparam.nfout2_2])):
                print ('!!! NaN detected: '+staid1+'_'+staid2)
                continue
            if np.where(fparam.arr2_2[8, :fparam.nfout2_2] > self.snr_thresh)[0].size < self.nfmin:
                continue
            if np.any(fparam.arr2_2[3, :fparam.nfout2_2] < 0.):
                continue
            Naftan      += 1
            raw_pers.append(fparam.arr2_2[1, :fparam.nfout2_2])
            raw_grvel.append(fparam.arr2_2[2, :fparam.nfout2_2])
            raw_phvel.append(fparam.arr2_2[3, :fparam.nfout2_2])
            raw_snr.append(fparam.arr2_2[8, :fparam.nfout2_2])
            distances.append(outarr[0] + outarr[1])
            # get min/max periods
            if fparam.arr2_2[1, 0] < Tmin and fparam.arr2_2[8, 0] > self.snr_thresh:
                Tmin    = fparam.arr2_2[1, 0]
            if fparam.arr2_2[1, fparam.nfout2_2 - 1] > Tmax and fparam.arr2_2[8, fparam.nfout2_2 - 1] > self.snr_thresh:
                Tmax    = fparam.arr2_2[1, fparam.nfout2_2 - 1]
        if Naftan < self.Ntrace_min:
            return 
        #=======================================
        # 1st iteration statistical analysis
        #=======================================
        pers            = self.pers
        pers            = pers[(pers >= Tmin)*(pers <= Tmax)]
        phvelarr        = np.zeros((Naftan, pers.size))
        snrarr          = np.zeros((Naftan, pers.size))
        indarr          = np.zeros((Naftan, pers.size), dtype = bool)
        for i in range(Naftan):
            phvel_spl       = scipy.interpolate.CubicSpline(raw_pers[i], raw_phvel[i])
            snr_spl         = scipy.interpolate.CubicSpline(raw_pers[i], raw_snr[i])
            phvelarr[i, :]  = phvel_spl(pers)
            snrarr[i, :]    = snr_spl(pers)
            indarr[i, :]    = (pers <= raw_pers[i][-1])*(pers >= raw_pers[i][0])*(snrarr[i, :] >=self.snr_thresh)
        Nm              = indarr.sum(axis = 0)
        if np.any(Nm == 0):
            # # # print ('!!! GAP detected, 1st iteration : '+staid1+'_'+staid2)
            ist, ibe    = _get_pers_ind(Nm)
            pers        = pers[ist:ibe+1]
            phvelarr    = phvelarr[:,ist:ibe+1]
            snrarr      = snrarr[:,ist:ibe+1]
            indarr      = indarr[:,ist:ibe+1]
            Nm          = indarr.sum(axis = 0)
        if np.any(Nm == 0): # debug
            raise ValueError('CHECK number of measure: '+staid1+'_'+staid2)
        tmpphvel        = (phvelarr*indarr).sum(axis = 0)
        meanvel         = tmpphvel/Nm
        unarr           = np.sum( indarr*(phvelarr - meanvel)**2, axis = 0)
        unarr           = unarr/Nm/Nm
        unarr           = np.sqrt(unarr)
        # throw out outliers
        indout          = abs(phvelarr - meanvel) > 3*unarr
        indarr[indout]  = False
        Nm              = indarr.sum(axis = 0)
        if np.any(Nm == 0): # debug
            raise ValueError('CHECK number of measure: '+staid1+'_'+staid2)
        tmpphvel        = (phvelarr*indarr).sum(axis = 0)
        meanvel         = tmpphvel/Nm
        mean_phvel_spl  = scipy.interpolate.CubicSpline(pers, meanvel)
        #====================
        # correct cycle skip
        #====================
        for i in range(Naftan):
            tmppers     = raw_pers[i]
            tmpC        = raw_phvel[i]
            tmpU        = raw_grvel[i]
            meanC       = mean_phvel_spl(tmppers[-1])
            omega       = 2.*np.pi/tmppers
            phase       = omega*(distances[i]/tmpU - distances[i]/tmpC)
            if tmpC[-1] > meanC:
                phase   -= 2.*np.pi
            else:
                phase   += 2.*np.pi
            tmpC_new    = omega*distances[i]/(omega*distances[i]/tmpU - phase)
            meanCs      = mean_phvel_spl(tmppers)
            del_C1      = np.sqrt( np.sum((tmpC-meanCs)**2/tmpC.size) )
            del_C2      = np.sqrt( np.sum((tmpC_new-meanCs)**2/tmpC.size) )
            if del_C1 > del_C2:
                raw_phvel[i][:]    = tmpC_new
        #=======================================
        # 2nd iteration statistical analysis
        #======================================= 
        phvelarr        = np.zeros((Naftan, pers.size))
        indarr          = np.zeros((Naftan, pers.size), dtype = bool)
        for i in range(Naftan):
            phvel_spl       = scipy.interpolate.CubicSpline(raw_pers[i], raw_phvel[i])
            tmpphvel        = phvel_spl(pers)
            phvelarr[i, :]  = tmpphvel
            indarr[i, :]    = (pers <= raw_pers[i][-1])*(pers >= raw_pers[i][0])*(snrarr[i, :] >=self.snr_thresh)*\
                                (tmpphvel >= self.vmin)*(tmpphvel <= self.vmax)
        Nm              = indarr.sum(axis = 0)
        if np.any(Nm == 0):
            # # # print ('!!! GAP detected 2nd iteration : '+staid1+'_'+staid2)
            ist, ibe    = _get_pers_ind(Nm)
            pers        = pers[ist:ibe+1]
            phvelarr    = phvelarr[:,ist:ibe+1]
            snrarr      = snrarr[:,ist:ibe+1]
            indarr      = indarr[:,ist:ibe+1]
            Nm          = indarr.sum(axis = 0)
        if np.any(Nm == 0): # debug
            raise ValueError('CHECK number of measure: '+staid1+'_'+staid2)
        tmpphvel        = (phvelarr*indarr).sum(axis = 0)
        meanvel         = tmpphvel/Nm
        unarr           = np.sum( indarr*(phvelarr-meanvel)**2, axis = 0)
        unarr           = unarr/Nm/Nm
        unarr           = np.sqrt(unarr)
        # outliers
        indout          = abs(phvelarr - meanvel) > 2*unarr
        indarr[indout]  = False
        # discard the whole dispersion curve if not enough points kept
        Npoint          = indarr.sum(axis = 1)
        indout2         = Npoint < self.nfmin
        indout2         = np.tile(indout2, (pers.size, 1))
        indout2         = indout2.T
        indarr[indout2] = False
        # detect gaps and keep the longest no-gap periods
        Nm              = indarr.sum(axis = 0)
        if np.any(Nm == 0):
            # # # print ('!!! GAP detected 2nd iteration : '+staid1+'_'+staid2)
            ist, ibe    = _get_pers_ind(Nm)
            pers        = pers[ist:ibe+1]
            phvelarr    = phvelarr[:,ist:ibe+1]
            snrarr      = snrarr[:,ist:ibe+1]
            indarr      = indarr[:,ist:ibe+1]
            Nm          = indarr.sum(axis = 0)
        if np.any(Nm == 0): # debug
            raise ValueError('CHECK number of measure: '+staid1+'_'+staid2)
        # final stacked results 
        tmpphvel        = (phvelarr*indarr).sum(axis = 0)
        mean_phvel      = tmpphvel/Nm
        unarr           = np.sum( indarr*(phvelarr - mean_phvel)**2, axis = 0)
        unarr           = unarr/Nm/Nm
        unarr           = np.sqrt(unarr)
        ftrig, trig, ierr, ist, ibe = _trigger(mean_phvel.size, np.float32(mean_phvel),\
                                        np.float32(2*np.pi/pers), np.float32(self.jump_thresh) )
        index           = np.zeros(pers.size, dtype = bool)
        if ierr != 0:
            index[ist:ibe]  = True
        # save results
        outdir          = self.outdir + '/DW_DISP/'+staid1
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                i   = 0
                while(i < 10):
                    sleep_time  = np.random.random()/10.
                    time.sleep(sleep_time)
                    if not os.path.isdir(outdir):
                        try:
                            os.makedirs(outdir)
                            break
                        except OSError:
                            pass
                    i   += 1
        outfname    = outdir + '/DISP_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'.npz'
        np.savez( outfname, pers, mean_phvel, unarr, snrarr, Nm, index)
        # save log files
        logfname    = self.datadir + '/logs_dw_stack_disp/'+ staid1 + '/' + staid1 +'_'+staid2+'.log'
        if not os.path.isdir(self.datadir + '/logs_dw_stack_disp/'+ staid1):
            try:
                os.makedirs(self.datadir + '/logs_dw_stack_disp/'+ staid1)
            except OSError:
                i   = 0
                while(i < 10):
                    sleep_time  = np.random.random()/10.
                    time.sleep(sleep_time)
                    if not os.path.isdir(self.datadir + '/logs_dw_stack_disp/'+ staid1):
                        try:
                            os.makedirs(self.datadir + '/logs_dw_stack_disp/'+ staid1)
                            break
                        except OSError:
                            pass
                    i   += 1
        with open(logfname, 'w') as fid:
            fid.writelines('SUCCESS\n')
        return 
    
    def direct_wave_phase_shift_stack(self, process_id= '', verbose = False):
        """direct wave three station interferogram phase shift stack
        """
        if verbose:
            self.print_info(process_id = process_id)
        chan1           = self.channel[0]
        chan2           = self.channel[1]
        staid1          = self.netcode1 + '.' + self.stacode1
        staid2          = self.netcode2 + '.' + self.stacode2
        if len(glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*.SAC')) > 0:
            is_sync     = True
        elif len(glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*.SAC')) > 0:
            is_sync     = False
        else:
            return 
        dist0, az0, baz0= obspy.geodetics.gps2dist_azimuth(self.stla1, self.stlo1, self.stla2, self.stlo2)
        dist0           /= 1000.
        if is_sync:
            saclst      = glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*.SAC')
        else:
            saclst      = glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*.SAC')
        #==============================
        # reference dispersion curve
        #==============================
        if len(self.phvel_ref) == 0 or self.prefer_c3_disp:
            dispfname       = self.datadir + '/DW_DISP/'+staid1 + '/DISP_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'.npz'
            if (not os.path.isfile(dispfname)) and len(self.phvel_ref) == 0:
                return 
            inarr           = np.load(dispfname)
            pers            = inarr['arr_0']
            phvel           = inarr['arr_1']
            snr             = inarr['arr_3']
            if np.any(np.isnan(phvel)) or np.any(np.isnan(pers)) or np.any(np.isnan(snr)):
                pers        = self.pers_ref
                phvel       = self.phvel_ref
                if len(self.phvel_ref) == 0:
                    print ('!!! NaN detected: '+staid1+'_'+staid2)
                    return
        else: 
            pers        = self.pers_ref
            phvel       = self.phvel_ref
        # bound check
        if np.any(phvel < self.vmin) or np.any(phvel > self.vmax):
            pers        = self.pers_ref
            phvel       = self.phvel_ref
            if len(self.phvel_ref) == 0:
                print ('!!! phase velocity out of bound: '+staid1+'_'+staid2)
                return
            if np.any(phvel < self.vmin) or np.any(phvel > self.vmax):
                print ('!!! phase velocity out of bound: '+staid1+'_'+staid2)
                return
        # length check
        if len(phvel) == 0:
            pers        = self.pers_ref
            phvel       = self.phvel_ref
            if len(self.phvel_ref) == 0:
                print ('!!! no reference phase velocity: '+staid1+'_'+staid2)
                return
        init_trace      = False
        for sacfname in saclst:
            tr          = obspy.read(sacfname)[0]
            dt          = tr.stats.delta
            d           = tr.stats.sac.user0
            dist        = tr.stats.sac.user0 + tr.stats.sac.dist
            # get symmetric component
            if abs(tr.stats.sac.b+tr.stats.sac.e) < tr.stats.delta:
                nhalf                   = int((tr.stats.npts-1)/2+1)
                neg                     = tr.data[:nhalf]
                pos                     = tr.data[nhalf-1:tr.stats.npts]
                neg                     = neg[::-1]
                tr.data                 = (pos+neg)/2 
                tr.stats.starttime      = tr.stats.starttime+tr.stats.sac.e
                tr.stats.sac.b          = 0.
            else:
                etime                   = tr.stats.endtime - (tr.stats.sac.e - tr.stats.sac.b)/2.
                tr.trim(endtime = etime)
            #=========
            # get SNR
            #=========
            time            = tr.times()
            begT            = time[0]
            endT            = time[-1]
            data_envelope   = obspy.signal.filter.envelope(tr.data)
            minT            = dist/self.vmax 
            maxT            = dist/self.vmin 
            ind             = (time >= minT)*(time <= maxT)
            amp_max         = data_envelope[ind].max()
            # noise window
            minT            = maxT + self.Tmax + 500.
            if( (endT - minT) < 1100. ):
                maxT        = endT - 10.
            else:
                minT        = endT - 1100.
                maxT        = endT - 100.
            ib              = (int)((minT-begT)/dt)
            ie              = (int)((maxT-begT)/dt)+2
            tempnoise       = tr.data[ib:ie]
            if ie-ib-1<= 0:
                continue
            noiserms        = np.sqrt(( np.sum(tempnoise**2))/(ie-ib-1.) )
            if noiserms == 0 or np.isnan(noiserms):
                continue
            if amp_max/noiserms < self.snr_thresh:
                # # # print (amp_max, noiserms, sacfname)
                continue
            rms             = np.sqrt(( np.sum(tr.data**2))/(tr.data.size) )
            weight          = 1./rms
            if 'ELL.SAC' in sacfname:
                iphase      = np.pi/4
            elif 'HYP.SAC' in sacfname:
                iphase      = -np.pi/4
            else:
                raise ValueError('Unexpected type of C3')
            # perform phase shift
            tr.data         = _tshift_fft(tr.data, dt, pers, phvel, iphase, d)
            tr.data         *= weight
            # debug
            # outfname        =  sacfname[:-4] + '_shift.sac'
            # tr.write(outfname, format='SAC')
            
            if not init_trace:
                stack_trace                 = tr.copy()
                stack_trace.stats.sac.user3 = 1
                init_trace                  = True
                continue
            else:
                stack_trace.data            += tr.data
                stack_trace.stats.sac.user3 += 1
        if not init_trace:
            if verbose:
                print ('!!!NO C3 data for: '+ staid1+'_'+chan1+'_'+staid2+'_'+chan2)
            return
        # save data
        outdir  = self.outdir + '/STACK_C3/'+staid1
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                i   = 0
                while(i < 10):
                    sleep_time  = np.random.random()/10.
                    time.sleep(sleep_time)
                    if not os.path.isdir(outdir):
                        try:
                            os.makedirs(outdir)
                            break
                        except OSError:
                            pass
                    i   += 1
        outfname= outdir+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'.SAC'
        stack_trace.write(outfname, format='SAC')
        # save log files
        logfname    = self.datadir + '/logs_dw_stack/'+ staid1 + '/' + staid1 +'_'+staid2+'.log'
        if not os.path.isdir(self.datadir + '/logs_dw_stack/'+ staid1):
            try:
                os.makedirs(self.datadir + '/logs_dw_stack/'+ staid1)
            except OSError:
                i   = 0
                while(i < 10):
                    sleep_time  = np.random.random()/10.
                    time.sleep(sleep_time)
                    if not os.path.isdir(self.datadir + '/logs_dw_stack/'+ staid1):
                        try:
                            os.makedirs(self.datadir + '/logs_dw_stack/'+ staid1)
                            break
                        except OSError:
                            pass
                    i   += 1
        with open(logfname, 'w') as fid:
            fid.writelines('SUCCESS\n')
        return 
    
def direct_wave_interfere_for_mp(in_c3_pair, verbose=False, verbose2=False):
    process_id   = multiprocessing.current_process().pid
    in_c3_pair.direct_wave_interfere(verbose = verbose, verbose2 = verbose2, process_id = process_id)
    return

def direct_wave_aftan_for_mp(in_c3_pair, verbose=False, verbose2=False):
    process_id   = multiprocessing.current_process().pid
    try:
        in_c3_pair.direct_wave_aftan(verbose = verbose, process_id = process_id)
    except:
        # write log files
        outdir      = in_c3_pair.datadir + '/logs_dw_aftan/'+ in_c3_pair.netcode1 + '.' + in_c3_pair.stacode1
        logfname    =  outdir + '/' + in_c3_pair.netcode1 + '.' + in_c3_pair.stacode1 +\
                       '_'+in_c3_pair.netcode2 + '.' + in_c3_pair.stacode2+'.log'
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                i   = 0
                while(i < 10):
                    sleep_time  = np.random.random()/10.
                    time.sleep(sleep_time)
                    if not os.path.isdir(outdir):
                        try:
                            os.makedirs(outdir)
                            break
                        except OSError:
                            pass
                    i   += 1
        with open(logfname, 'w') as fid:
            fid.writelines('FAILED\n')
    return

def direct_wave_stack_disp_for_mp(in_c3_pair, verbose=False, verbose2=False):
    process_id   = multiprocessing.current_process().pid
    try:
        in_c3_pair.direct_wave_stack_disp(verbose = verbose, process_id = process_id)
    except:
        # write log files
        outdir      = in_c3_pair.datadir + '/logs_dw_stack_disp/'+ in_c3_pair.netcode1 + '.' + in_c3_pair.stacode1
        logfname    =  outdir + '/' + in_c3_pair.netcode1 + '.' + in_c3_pair.stacode1 +\
                       '_'+in_c3_pair.netcode2 + '.' + in_c3_pair.stacode2+'.log'
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                i   = 0
                while(i < 10):
                    sleep_time  = np.random.random()/10.
                    time.sleep(sleep_time)
                    if not os.path.isdir(outdir):
                        try:
                            os.makedirs(outdir)
                            break
                        except OSError:
                            pass
                    i   += 1
        with open(logfname, 'w') as fid:
            fid.writelines('FAILED\n')
    return

def direct_wave_phase_shift_stack_for_mp(in_c3_pair, verbose=False, verbose2=False):
    process_id   = multiprocessing.current_process().pid
    try:
        in_c3_pair.direct_wave_phase_shift_stack(verbose = verbose, process_id = process_id)
    except:
        # write log files
        outdir      = in_c3_pair.datadir + '/logs_dw_stack/'+ in_c3_pair.netcode1 + '.' + in_c3_pair.stacode1
        logfname    =  outdir + '/' + in_c3_pair.netcode1 + '.' + in_c3_pair.stacode1 +\
                       '_'+in_c3_pair.netcode2 + '.' + in_c3_pair.stacode2+'.log'
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                i   = 0
                while(i < 10):
                    sleep_time  = np.random.random()/10.
                    time.sleep(sleep_time)
                    if not os.path.isdir(outdir):
                        try:
                            os.makedirs(outdir)
                            break
                        except OSError:
                            pass
                    i   += 1
        with open(logfname, 'w') as fid:
            fid.writelines('FAILED\n')
    return

