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
import scipy.signal
import glob
import time


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
            basic1=True, basic2=True, pmf1=True, pmf2=True, f77=True, prephdir=''):
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
        return
    
    def print_info(self, process_id):
        """print the informations of this pair
        """
        staid1          = self.netcode1 + '.' + self.stacode1
        staid2          = self.netcode2 + '.' + self.stacode2
        print ('--- '+ staid1+'_'+staid2+' processID = '+str(process_id))
    
    def direct_wave_interfere(self, process_id= '', verbose = False, verbose2= False):
        """
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
        if len(glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*ELL.SAC')) > 0 or \
            len(glob.glob(self.datadir + '/SYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*HYP.SAC')) > 0:
            is_sync     = True
        elif len(glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*ELL.SAC')) > 0 or \
            len(glob.glob(self.datadir + '/ASYNC_C3/'+staid1+'/C3_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'_*HYP.SAC')) > 0:
            is_sync     = False
        else:
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
        return 
        
    
def direct_wave_interfere_for_mp(in_c3_pair, verbose=False, verbose2=False):
    process_id   = multiprocessing.current_process().pid
    in_c3_pair.direct_wave_interfere(verbose = verbose, verbose2 = verbose2, process_id = process_id)
    return

def direct_wave_aftan_for_mp(in_c3_pair, verbose=False, verbose2=False):
    process_id   = multiprocessing.current_process().pid
    in_c3_pair.direct_wave_aftan(verbose = verbose, process_id = process_id)
    return
    