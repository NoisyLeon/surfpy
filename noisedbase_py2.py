# -*- coding: utf-8 -*-
"""
A python module for ambient noise data analysis based on ASDF database

:Methods:
    aftan analysis (use pyaftan or aftanf77)
    preparing data for surface wave tomography (Barmin's method, Eikonal/Helmholtz tomography)
    stacking/rotation for cross-correlation results from ANXCorr and Seed2Cor

:Dependencies:
    pyasdf and its dependencies
    ObsPy  and its dependencies
    pyproj
    Basemap
    pyfftw 0.10.3 (optional)
    
:Copyright:
    Author: Lili Feng
    Research Geophysicist
    CGG
    email: lfeng1011@gmail.com
"""
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import obspy
import warnings
import copy
import os, shutil
from functools import partial
import multiprocessing
import pyaftan
from subprocess import call
from obspy.clients.fdsn.client import Client
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import obspy.signal.array_analysis
from obspy.imaging.cm import obspy_sequential
import glob
from numba import jit, float32, int32, boolean, float64
import pyfftw
import time


sta_info_default        = {'xcorr': 1, 'isnet': 0}
xcorr_header_default    = {'netcode1': '', 'stacode1': '', 'netcode2': '', 'stacode2': '', 'chan1': '', 'chan2': '',
        'npts': 12345, 'b': 12345, 'e': 12345, 'delta': 12345, 'dist': 12345, 'az': 12345, 'baz': 12345, 'stackday': 0}
xcorr_sacheader_default = {'knetwk': '', 'kstnm': '', 'kcmpnm': '', 'stla': 12345, 'stlo': 12345, 
            'kuser0': '', 'kevnm': '', 'evla': 12345, 'evlo': 12345, 'evdp': 0., 'dist': 0., 'az': 12345, 'baz': 12345, 
                'delta': 12345, 'npts': 12345, 'user0': 0, 'b': 12345, 'e': 12345}
monthdict               = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


@jit(float32[:](float32[:, :], float32[:, :], int32))
def _CalcRecCor(arr1, arr2, lagN):
    """compute the amplitude weight for the xcorr, used for amplitude correction
        optimized by numba
    ==============================================================================
    ::: input parameters :::
    arr1, arr2  - the input arrays from ft_*SAC_rec,
                    indicating holes in the original records
    lagN        - one-sided npts for xcorr
    ::: output :::
    cor_rec     - amplitude weight for the xcorr
    ==============================================================================
    """
    N1      = arr1.shape[0]
    N2      = arr2.shape[0]
    # array indicating number of data points for xcorr, used for amplitude correction
    cor_rec = np.zeros(int(2*lagN + 1), dtype=float)
    for i in range(lagN+1):
        cor_rec[lagN+i] = 0.
        cor_rec[lagN-i] = 0.
        for irec1 in range(N1):
            for irec2 in range(N2):
                if arr1[irec1, 0] >= arr2[irec2, 1] - i:
                    continue
                if arr1[irec1, 1] <= arr2[irec2, 0] - i:
                    break
                recB            = max(arr1[irec1, 0], arr2[irec2, 0] - i)
                recE            = min(arr1[irec1, 1], arr2[irec2, 1] - i)
                cor_rec[lagN+i] += recE - recB
            for irec2 in range(N2):
                if arr1[irec1, 0] >= arr2[irec2, 1] + i:
                    continue
                if arr1[irec1, 1] <= arr2[irec2, 0] + i:
                    break
                recB            = max(arr1[irec1, 0], arr2[irec2, 0] + i)
                recE            = min(arr1[irec1, 1], arr2[irec2, 1] + i)
                cor_rec[lagN-i] += recE - recB
    cor_rec[lagN]   /= 2.
    return cor_rec

def _amp_ph_to_xcorr(amp1, amp2, ph1, ph2, sps = 1., lagtime = 3000.):
    """Convert amplitude and phase arrays to xcorr
    ==============================================================================
    ::: input parameters :::
    amp1, ph1   - amplitude and phase data arrays for station(component) 1
    amp2, ph2   - amplitude and phase data arrays for station(component) 2
    sps         - target sampling rate
    lagtime     - lag time for xcorr
    ::: output :::
    out_data    - xcorr
    ==============================================================================
    """
    N           = amp1.size
    Ns          = int(2*N - 1)
    # cross-spectrum, conj(sac1)*(sac2)
    x_sp        = np.zeros(Ns, dtype=complex) 
    temp1       = np.zeros(N, dtype=complex)
    temp2       = np.zeros(N, dtype=complex)
    temp1.real  = amp1*np.cos(ph1)
    temp1.imag  = amp1*np.sin(ph1)
    temp2.real  = amp2*np.cos(ph2)
    temp2.imag  = amp2*np.sin(ph2)
    x_sp[:N]    = temp2 * np.conj(temp1)
    # perform inverse FFT with pyFFTW, much faster than numpy_fft, scipy.fftpack
    out         = pyfftw.interfaces.numpy_fft.ifft(x_sp)
    seis_out    = 2.*(out.real)
    lagN        = int(np.floor(lagtime*sps +0.5))
    if lagN > Ns:
        raise ValueError('Lagtime npts overflow!')
    out_data            = np.zeros(2*lagN+1, dtype=float)
    out_data[lagN]      = seis_out[0]
    out_data[:lagN]     = (seis_out[1:lagN+1])[::-1]
    out_data[(lagN+1):] = (seis_out[Ns-lagN:])[::-1]
    return out_data

def _amp_ph_to_xcorr_fast(amp1, amp2, ph1, ph2, fftw_plan, sps = 1., lagtime = 3000.):
    """Convert amplitude and phase arrays to xcorr
        This is the fast version of _amp_ph_to_xcorr, a precomputed fftw_plan needs
        to be prepared for speeding up
    ==============================================================================
    ::: input parameters :::
    amp1, ph1   - amplitude and phase data arrays for station(component) 1
    amp2, ph2   - amplitude and phase data arrays for station(component) 2
    fftw_plan   - precomputed fftw_plan
    sps         - target sampling rate
    lagtime     - lag time for xcorr
    ::: output :::
    out_data    - xcorr
    ==============================================================================
    """
    N           = amp1.size
    Ns          = int(2*N - 1)
    # cross-spectrum, conj(sac1)*(sac2)
    x_sp        = np.zeros(Ns, dtype=complex)
    out         = np.zeros(Ns, dtype=complex)
    temp1       = np.zeros(N, dtype=complex)
    temp2       = np.zeros(N, dtype=complex)
    temp1.real  = amp1*np.cos(ph1)
    temp1.imag  = amp1*np.sin(ph1)
    temp2.real  = amp2*np.cos(ph2)
    temp2.imag  = amp2*np.sin(ph2)
    x_sp[:N]    = temp2 * np.conj(temp1)
    # perform inverse FFT with pyFFTW, much faster than numpy_fft, scipy.fftpack
    # the precomputed fftw_plan is used
    fftw_plan.update_arrays(x_sp, out)
    fftw_plan.execute()
    seis_out    = 2.*(out.real)
    lagN        = int(np.floor(lagtime*sps +0.5))
    if lagN > Ns:
        raise ValueError('Lagtime npts overflow!')
    out_data            = np.zeros(2*lagN+1, dtype=float)
    out_data[lagN]      = seis_out[0]
    out_data[:lagN]     = (seis_out[1:lagN+1])[::-1]
    out_data[(lagN+1):] = (seis_out[Ns-lagN:])[::-1]
    return out_data/Ns

class xcorr_pair(object):
    """ An object to for ambient noise cross-correlation computation
    =================================================================================================================
    ::: parameters :::
    stacode1, netcode1  - station/network code for station 1
    stacode2, netcode2  - station/network code for station 2
    monthdir            - month directory (e.g. 2019.JAN)
    daylst              - list includes the days for xcorr
    =================================================================================================================
    """
    def __init__(self, stacode1, netcode1, stacode2, netcode2, monthdir, daylst):
        self.stacode1   = stacode1
        self.netcode1   = netcode1
        self.stacode2   = stacode2
        self.netcode2   = netcode2
        self.monthdir   = monthdir
        self.daylst     = daylst
        return
    
    def print_info(self):
        """print the informations of this pair
        """
        staid1          = self.netcode1 + '.' + self.stacode1
        staid2          = self.netcode2 + '.' + self.stacode2
        print '--- '+ staid1+'_'+staid2+' : '+self.monthdir+' '+str(len(self.daylst))+' days' 
    
    def convert_amph_to_xcorr(self, datadir, chans=['LHZ', 'LHE', 'LHN'], ftlen = True,\
            tlen = 84000., mintlen = 20000., sps = 1., lagtime = 3000., CorOutflag = 0, \
            fprcs = False, fastfft=True, verbose=False):
        """
        Convert amplitude and phase files to xcorr
        =================================================================================================================
        ::: input parameters :::
        datadir     - directory including data and output
        chans       - channel list
        ftlen       - turn (on/off) cross-correlation-time-length correction for amplitude
        tlen        - time length of daily records (in sec)
        mintlen     - allowed minimum time length for cross-correlation (takes effect only when ftlen = True)
        sps         - target sampling rate
        lagtime     - cross-correlation signal half length in sec
        CorOutflag  - 0 = only output monthly xcorr data, 1 = only daily, 2 or others = output both
        fprcs       - turn on/off (1/0) precursor signal checking, NOT implemented yet
        fastfft     - speeding up the computation by using precomputed fftw_plan or not
        =================================================================================================================
        """
        if verbose:
            self.print_info()
        staid1                  = self.netcode1 + '.' + self.stacode1
        staid2                  = self.netcode2 + '.' + self.stacode2
        month_dir               = datadir+'/'+self.monthdir
        monthly_xcorr           = []
        chan_size               = len(chans)
        init_common_header      = False
        xcorr_common_sacheader  = xcorr_sacheader_default.copy()
        lagN                    = np.floor(lagtime*sps +0.5) # npts for one-sided lag
        stacked_day             = 0
        #---------------------------------------
        # construct fftw_plan for speeding up
        #---------------------------------------
        if fastfft:
            temp_pfx        = month_dir+'/'+self.monthdir+'.'+str(self.daylst[0])+\
                                '/ft_'+self.monthdir+'.'+str(self.daylst[0])+'.'+staid1+'.'+chans[0]+'.SAC'
            amp_ref         = obspy.read(temp_pfx+'.am')[0]
            Nref            = amp_ref.data.size
            Ns              = int(2*Nref - 1)
            temp_x_sp       = np.zeros(Ns, dtype=complex)
            temp_out        = np.zeros(Ns, dtype=complex)
            fftw_plan       = pyfftw.FFTW(input_array=temp_x_sp, output_array=temp_out, direction='FFTW_BACKWARD',\
                                flags=('FFTW_MEASURE', ))
        else:
            Nref            = 0
        #-----------------
        # loop over days
        #-----------------
        for day in self.daylst:
            # input streams
            st_amp1     = obspy.Stream()
            st_ph1      = obspy.Stream()
            st_amp2     = obspy.Stream()
            st_ph2      = obspy.Stream()
            # daily output streams
            daily_xcorr = []
            daydir      = month_dir+'/'+self.monthdir+'.'+str(day)
            # read amp/ph files
            for chan in chans:
                pfx1    = daydir+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid1+'.'+chan+'.SAC'
                pfx2    = daydir+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid2+'.'+chan+'.SAC'
                st_amp1 += obspy.read(pfx1+'.am')
                st_ph1  += obspy.read(pfx1+'.ph')
                st_amp2 += obspy.read(pfx2+'.am')
                st_ph2  += obspy.read(pfx2+'.ph')
            #-----------------------------
            # define commone sac header
            #-----------------------------
            if not init_common_header:
                tr1                                 = st_amp1[0]
                tr2                                 = st_amp2[0]
                xcorr_common_sacheader['kuser0']    = self.netcode1
                xcorr_common_sacheader['kevnm']     = self.stacode1
                xcorr_common_sacheader['knetwk']    = self.netcode2
                xcorr_common_sacheader['kstnm']     = self.stacode2
                # # # xcorr_common_sacheader['kcmpnm']    = chan1+chan2
                xcorr_common_sacheader['evla']      = tr1.stats.sac.stla
                xcorr_common_sacheader['evlo']      = tr1.stats.sac.stlo
                xcorr_common_sacheader['stla']      = tr2.stats.sac.stla
                xcorr_common_sacheader['stlo']      = tr2.stats.sac.stlo
                dist, az, baz                       = obspy.geodetics.gps2dist_azimuth(tr1.stats.sac.stla, tr1.stats.sac.stlo,\
                                                        tr2.stats.sac.stla, tr2.stats.sac.stlo) # distance is in m
                xcorr_common_sacheader['dist']      = dist/1000.
                xcorr_common_sacheader['az']        = az
                xcorr_common_sacheader['baz']       = baz
                xcorr_common_sacheader['delta']     = 1./sps
                xcorr_common_sacheader['npts']      = int(2*lagN + 1)
                xcorr_common_sacheader['b']         = -float(lagN/sps)
                xcorr_common_sacheader['e']         = float(lagN/sps)
                xcorr_common_sacheader['user0']     = 1
                init_common_header                  = True
            skip_this_day   = False
            # compute cross-correlation
            for ich1 in range(chan_size):
                if skip_this_day:
                    break
                for ich2 in range(chan_size):
                    # get data arrays
                    amp1    = st_amp1[ich1].data
                    ph1     = st_ph1[ich1].data
                    amp2    = st_amp2[ich2].data
                    ph2     = st_ph2[ich2].data
                    # quality control
                    if (np.isnan(amp1)).any() or (np.isnan(amp2)).any() or \
                            (np.isnan(ph1)).any() or (np.isnan(ph2)).any():
                        skip_this_day   = True
                        break
                    if np.any(amp1 > 1e20) or np.any(amp2 > 1e20):
                        skip_this_day   = True
                        break
                    # get amplitude correction array
                    Namp        = amp1.size
                    if ftlen:
                        # npts for the length of the preprocessed daily record 
                        Nrec    = int(tlen*sps)
                        frec1   = daydir+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid1+'.'+chans[ich1]+'.SAC_rec'
                        frec2   = daydir+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid2+'.'+chans[ich1]+'.SAC_rec'
                        if os.path.isfile(frec1):
                            arr1= np.loadtxt(frec1)
                            if arr1.size == 2:
                                arr1    = arr1.reshape(1, 2) 
                        else:
                            arr1= (np.array([0, Nrec])).reshape(1, 2)                            
                        if os.path.isfile(frec2):
                            arr2= np.loadtxt(frec2)
                            if arr2.size == 2:
                                arr2    = arr2.reshape(1, 2)
                        else:
                            arr2= (np.array([0, Nrec])).reshape(1, 2)
                        cor_rec     = _CalcRecCor(arr1, arr2, np.int32(lagN))
                        # skip the day if the length of available data is too small
                        if cor_rec[0] < mintlen*sps or cor_rec[-1] < mintlen*sps:
                            skip_this_day   = True
                            break
                        # skip the day if any data point has a weight of zero
                        if np.any(cor_rec == 0.):
                            skip_this_day   = True
                            break
                    # comvert amp & ph files to xcorr
                    if fastfft and Namp == Nref:
                        out_data        = _amp_ph_to_xcorr_fast(amp1=amp1, ph1=ph1, amp2=amp2, ph2=ph2, sps=sps,\
                                                                lagtime=lagtime, fftw_plan=fftw_plan)
                    else:
                        out_data        = _amp_ph_to_xcorr(amp1=amp1, ph1=ph1, amp2=amp2, ph2=ph2, sps=sps, lagtime=lagtime)
                    # amplitude correction
                    if ftlen:
                        out_data    /= cor_rec
                        out_data    *= float(2*Namp - 1)
                    # end of computing individual xcorr
                    daily_xcorr.append(out_data)
            # end loop over channels
            if not skip_this_day:
                if verbose:
                    print 'xcorr finished '+ staid1+'_'+staid2+' : '+self.monthdir+'.'+str(day)
                # output daily xcorr
                if CorOutflag != 0:
                    out_daily_dir   = month_dir+'/COR_D/'+staid1
                    if not os.path.isdir(out_daily_dir):
                        os.makedirs(out_daily_dir)
                    for ich1 in range(chan_size):
                        for ich2 in range(chan_size):
                            i                       = 3*ich1 + ich2
                            out_daily_fname         = out_daily_dir+'/COR_'+staid1+'_'+chans[ich1]+\
                                                        '_'+staid2+'_'+chans[ich2]+'_'+str(day)+'.SAC'
                            daily_header            = xcorr_common_sacheader.copy()
                            daily_header['kcmpnm']  = chans[ich1]+chans[ich2]
                            sacTr                   = obspy.io.sac.sactrace.SACTrace(data = daily_xcorr[i], **daily_header)
                            sacTr.write(out_daily_fname)
                # append to monthly data
                if CorOutflag != 1:
                    # intilize
                    if stacked_day  == 0:
                        for ich1 in range(chan_size):
                            for ich2 in range(chan_size):
                                i                   = 3*ich1 + ich2
                                monthly_xcorr.append(daily_xcorr[i])
                    # stacking
                    else:
                        for ich1 in range(chan_size):
                            for ich2 in range(chan_size):
                                i                   = 3*ich1 + ich2
                                monthly_xcorr[i]    += daily_xcorr[i]
                    stacked_day += 1
        # end loop over days
        if CorOutflag != 1 and stacked_day != 0:
            out_monthly_dir     = month_dir+'/COR/'+staid1
            if not os.path.isdir(out_monthly_dir):
                try:
                    os.makedirs(out_monthly_dir)
                except OSError:
                    i   = 0
                    while(i < 10):
                        sleep_time  = np.random.random()/10.
                        time.sleep(sleep_time)
                        if not os.path.isdir(out_monthly_dir):
                            try:
                                os.makedirs(out_monthly_dir)
                                break
                            except OSError:
                                pass
                        i   += 1
            for ich1 in range(chan_size):
                for ich2 in range(chan_size):
                    i                           = 3*ich1 + ich2
                    out_monthly_fname           = out_monthly_dir+'/COR_'+staid1+'_'+chans[ich1]+\
                                                    '_'+staid2+'_'+chans[ich2]+'.SAC'
                    monthly_header              = xcorr_common_sacheader.copy()
                    monthly_header['kcmpnm']    = chans[ich1]+chans[ich2]
                    monthly_header['user0']     = stacked_day
                    sacTr                       = obspy.io.sac.sactrace.SACTrace(data = monthly_xcorr[i], **monthly_header)
                    sacTr.write(out_monthly_fname)
        return

def amph_to_xcorr_for_mp(in_xcorr_pair, datadir, chans=['LHZ', 'LHE', 'LHN'], ftlen = True,\
            tlen = 84000., mintlen = 20000., sps = 1., lagtime = 3000., CorOutflag = 0, \
            fprcs = False, fastfft=True):
    
    in_xcorr_pair.convert_amph_to_xcorr(datadir=datadir, chans=chans, ftlen = ftlen,\
            tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                    fprcs = fprcs, fastfft=fastfft)
    return


class beamforming_stream(obspy.Stream):
    """ An object to for ambient noise cross-correlation computation
    =================================================================================================================
    ::: parameters :::
    stacode1, netcode1  - station/network code for station 1
    stacode2, netcode2  - station/network code for station 2
    monthdir            - month directory (e.g. 2019.JAN)
    daylst              - list includes the days for xcorr
    =================================================================================================================
    """
    def get_addtional_info(self, datadir, bfUTCdate):
        self.datadir    = datadir
        self.bfUTCdate  = bfUTCdate
        return
    
class noiseASDF(pyasdf.ASDFDataSet):
    """ An object to for ambient noise cross-correlation analysis based on ASDF database
    =================================================================================================================
    version history:
        Dec 6th, 2016   - first version
    =================================================================================================================
    """
    def print_info(self):
        """
        print information of the dataset.
        """
        outstr  = '================================================= Ambient Noise Cross-correlation Database =================================================\n'
        outstr  += self.__str__()+'\n'
        outstr  += '--------------------------------------------------------------------------------------------------------------------------------------------\n'
        if 'NoiseXcorr' in self.auxiliary_data.list():
            outstr      += 'NoiseXcorr              - Cross-correlation seismogram\n'
        if 'StaInfo' in self.auxiliary_data.list():
            outstr      += 'StaInfo                 - Auxiliary station information\n'
        if 'DISPbasic1' in self.auxiliary_data.list():
            outstr      += 'DISPbasic1              - Basic dispersion curve, no jump correction\n'
        if 'DISPbasic2' in self.auxiliary_data.list():
            outstr      += 'DISPbasic2              - Basic dispersion curve, with jump correction\n'
        if 'DISPpmf1' in self.auxiliary_data.list():
            outstr      += 'DISPpmf1                - PMF dispersion curve, no jump correction\n'
        if 'DISPpmf2' in self.auxiliary_data.list():
            outstr      += 'DISPpmf2                - PMF dispersion curve, with jump correction\n'
        if 'DISPbasic1interp' in self.auxiliary_data.list():
            outstr      += 'DISPbasic1interp        - Interpolated DISPbasic1\n'
        if 'DISPbasic2interp' in self.auxiliary_data.list():
            outstr      += 'DISPbasic2interp        - Interpolated DISPbasic2\n'
        if 'DISPpmf1interp' in self.auxiliary_data.list():
            outstr      += 'DISPpmf1interp          - Interpolated DISPpmf1\n'
        if 'DISPpmf2interp' in self.auxiliary_data.list():
            outstr      += 'DISPpmf2interp          - Interpolated DISPpmf2\n'
        if 'FieldDISPbasic1interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPbasic1interp   - Field data of DISPbasic1\n'
        if 'FieldDISPbasic2interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPbasic2interp   - Field data of DISPbasic2\n'
        if 'FieldDISPpmf1interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPpmf1interp     - Field data of DISPpmf1\n'
        if 'FieldDISPpmf2interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPpmf2interp     - Field data of DISPpmf2\n'
        outstr += '============================================================================================================================================\n'
        print outstr
        return
    
    def write_stationxml(self, staxml, source='CIEI'):
        """write obspy inventory to StationXML data file
        """
        inv     = obspy.core.inventory.inventory.Inventory(networks=[], source=source)
        for staid in self.waveforms.list():
            inv += self.waveforms[staid].StationXML
        inv.write(staxml, format='stationxml')
        return
    
    def write_stationtxt(self, stafile):
        """write obspy inventory to txt station list(format used in ANXcorr and Seed2Cor)
        """
        try:
            auxiliary_info      = self.auxiliary_data.StaInfo
            isStaInfo           = True
        except:
            isStaInfo           = False
        with open(stafile, 'w') as f:
            for staid in self.waveforms.list():
                stainv          = self.waveforms[staid].StationXML
                netcode         = stainv.networks[0].code
                stacode         = stainv.networks[0].stations[0].code
                lon             = stainv.networks[0].stations[0].longitude
                lat             = stainv.networks[0].stations[0].latitude
                if isStaInfo:
                    staid_aux   = netcode+'/'+stacode
                    xcorrflag   = auxiliary_info[staid_aux].parameters['xcorr']
                    f.writelines('%s %3.4f %3.4f %d %s\n' %(stacode, lon, lat, xcorrflag, netcode) )
                else:
                    f.writelines('%s %3.4f %3.4f %s\n' %(stacode, lon, lat, netcode) )        
        return
    
    def read_stationtxt(self, stafile, source='CIEI', chans=['LHZ', 'LHE', 'LHN'], dnetcode=None):
        """read txt station list 
        """
        sta_info                        = sta_info_default.copy()
        with open(stafile, 'r') as f:
            Sta                         = []
            site                        = obspy.core.inventory.util.Site(name='01')
            creation_date               = obspy.core.utcdatetime.UTCDateTime(0)
            inv                         = obspy.core.inventory.inventory.Inventory(networks=[], source=source)
            total_number_of_channels    = len(chans)
            for lines in f.readlines():
                lines       = lines.split()
                stacode     = lines[0]
                lon         = float(lines[1])
                lat         = float(lines[2])
                netcode     = dnetcode
                xcorrflag   = None
                if len(lines)==5:
                    try:
                        xcorrflag   = int(lines[3])
                        netcode     = lines[4]
                    except ValueError:
                        xcorrflag   = int(lines[4])
                        netcode     = lines[3]
                if len(lines)==4:
                    try:
                        xcorrflag   = int(lines[3])
                    except ValueError:
                        netcode     = lines[3]
                if netcode is None:
                    netsta          = stacode
                else:
                    netsta          = netcode+'.'+stacode
                if Sta.__contains__(netsta):
                    index           = Sta.index(netsta)
                    if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                        raise ValueError('incompatible station location:' + netsta+' in station list!')
                    else:
                        print 'WARNING: repeated station:' +netsta+' in station list!'
                        continue
                channels    = []
                if lon>180.:
                    lon     -= 360.
                for chan in chans:
                    channel = obspy.core.inventory.channel.Channel(code=chan, location_code='01', latitude=lat, longitude=lon,
                                elevation=0.0, depth=0.0)
                    channels.append(channel)
                station     = obspy.core.inventory.station.Station(code=stacode, latitude=lat, longitude=lon, elevation=0.0,
                                site=site, channels=channels, total_number_of_channels = total_number_of_channels, creation_date = creation_date)
                network     = obspy.core.inventory.network.Network(code=netcode, stations=[station])
                networks    = [network]
                inv         += obspy.core.inventory.inventory.Inventory(networks=networks, source=source)
                if netcode is None:
                    staid_aux           = stacode
                else:
                    staid_aux           = netcode+'/'+stacode
                if xcorrflag != None:
                    sta_info['xcorr']   = xcorrflag
                self.add_auxiliary_data(data=np.array([]), data_type='StaInfo', path=staid_aux, parameters=sta_info)
        print('Writing obspy inventory to ASDF dataset')
        self.add_stationxml(inv)
        print('End writing obspy inventory to ASDF dataset')
        return 
    
    def read_stationtxt_ind(self, stafile, source='CIEI', chans=['LHZ', 'LHE', 'LHN'], s_ind=1, lon_ind=2, lat_ind=3, n_ind=0):
        """read txt station list, column index can be changed
        """
        sta_info                    = sta_info_default.copy()
        with open(stafile, 'r') as f:
            Sta                     = []
            site                    = obspy.core.inventory.util.Site(name='')
            creation_date           = obspy.core.utcdatetime.UTCDateTime(0)
            inv                     = obspy.core.inventory.inventory.Inventory(networks=[], source=source)
            total_number_of_channels= len(chans)
            for lines in f.readlines():
                lines           = lines.split()
                stacode         = lines[s_ind]
                lon             = float(lines[lon_ind])
                lat             = float(lines[lat_ind])
                netcode         = lines[n_ind]
                netsta          = netcode+'.'+stacode
                if Sta.__contains__(netsta):
                    index       = Sta.index(netsta)
                    if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                        raise ValueError('incompatible station location:' + netsta+' in station list!')
                    else:
                        print 'WARNING: repeated station:' +netsta+' in station list!'
                        continue
                channels        = []
                if lon>180.:
                    lon         -=360.
                for chan in chans:
                    channel     = obspy.core.inventory.channel.Channel(code=chan, location_code='01', latitude=lat, longitude=lon,
                                    elevation=0.0, depth=0.0)
                    channels.append(channel)
                station         = obspy.core.inventory.station.Station(code=stacode, latitude=lat, longitude=lon, elevation=0.0,
                                    site=site, channels=channels, total_number_of_channels = total_number_of_channels, creation_date = creation_date)
                network         = obspy.core.inventory.network.Network(code=netcode, stations=[station])
                networks        = [network]
                inv             += obspy.core.inventory.inventory.Inventory(networks=networks, source=source)
                staid_aux       = netcode+'/'+stacode
                self.add_auxiliary_data(data=np.array([]), data_type='StaInfo', path=staid_aux, parameters=sta_info)
        print('Writing obspy inventory to ASDF dataset')
        self.add_stationxml(inv)
        print('End writing obspy inventory to ASDF dataset')
        return 
    
    def copy_stations(self, inasdffname, startdate=None, enddate=None, location=None, channel=None, includerestricted=False,
            minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, latitude=None, longitude=None, minradius=None, maxradius=None):
        """copy and renew station inventory given an input ASDF file
            the function will copy the network and station names while renew other informations given new limitations
        =======================================================================================================
        ::: input parameters :::
        inasdffname         - input ASDF file name
        startdate, enddata  - start/end date for searching
        network             - Select one or more network codes.
                                Can be SEED network codes or data center defined codes.
                                    Multiple codes are comma-separated (e.g. "IU,TA").
        station             - Select one or more SEED station codes.
                                Multiple codes are comma-separated (e.g. "ANMO,PFO").
        location            - Select one or more SEED location identifiers.
                                Multiple identifiers are comma-separated (e.g. "00,01").
                                As a special case “--“ (two dashes) will be translated to a string of two space
                                characters to match blank location IDs.
        channel             - Select one or more SEED channel codes.
                                Multiple codes are comma-separated (e.g. "BHZ,HHZ").             
        minlatitude         - Limit to events with a latitude larger than the specified minimum.
        maxlatitude         - Limit to events with a latitude smaller than the specified maximum.
        minlongitude        - Limit to events with a longitude larger than the specified minimum.
        maxlongitude        - Limit to events with a longitude smaller than the specified maximum.
        latitude            - Specify the latitude to be used for a radius search.
        longitude           - Specify the longitude to the used for a radius search.
        minradius           - Limit to events within the specified minimum number of degrees from the
                                geographic point defined by the latitude and longitude parameters.
        maxradius           - Limit to events within the specified maximum number of degrees from the
                                geographic point defined by the latitude and longitude parameters.
        =======================================================================================================
        """
        try:
            starttime   = obspy.core.utcdatetime.UTCDateTime(startdate)
        except:
            starttime   = None
        try:
            endtime     = obspy.core.utcdatetime.UTCDateTime(enddate)
        except:
            endtime     = None
        client          = Client('IRIS')
        init_flag       = False
        indset          = pyasdf.ASDFDataSet(inasdffname)
        for staid in indset.waveforms.list():
            network     = staid.split('.')[0]
            station     = staid.split('.')[1]
            print 'Copying/renewing station inventory: '+ staid
            if init_flag:
                inv     += client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, channel=channel, 
                            minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                            latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, level='channel',
                            includerestricted=includerestricted)
            else:
                inv     = client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, channel=channel, 
                            minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                            latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, level='channel',
                            includerestricted=includerestricted)
                init_flag= True
        self.add_stationxml(inv)
        try:
            self.inv    +=inv
        except:
            self.inv    = inv
        return
    
    def get_limits_lonlat(self):
        """get the geographical limits of the stations
        """
        staLst      = self.waveforms.list()
        minlat      = 90.
        maxlat      = -90.
        minlon      = 360.
        maxlon      = 0.
        for staid in staLst:
            lat, elv, lon   = self.waveforms[staid].coordinates.values()
            if lon<0:
                lon         += 360.
            minlat  = min(lat, minlat)
            maxlat  = max(lat, maxlat)
            minlon  = min(lon, minlon)
            maxlon  = max(lon, maxlon)
        print 'latitude range: ', minlat, '-', maxlat, 'longitude range:', minlon, '-', maxlon
        self.minlat = minlat
        self.maxlat = maxlat
        self.minlon = minlon
        self.maxlon = maxlon
        return
            
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i', blon=0., blat=0.):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        try:
            minlon  = self.minlon-blon
            maxlon  = self.maxlon+blon
            minlat  = self.minlat-blat
            maxlat  = self.maxlat+blat
        except AttributeError:
            self.get_limits_lonlat()
            minlon  = self.minlon-blon
            maxlon  = self.maxlon+blon
            minlat  = self.minlat-blat
            maxlat  = self.maxlat+blat
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection == 'merc':
            m       = Basemap(projection='merc', llcrnrlat=minlat-5., urcrnrlat=maxlat+5., llcrnrlon=minlon-5.,
                        urcrnrlon=maxlon+5., lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection == 'global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
        elif projection == 'regional_ortho':
            m1      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                        llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, minlat, maxlon) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat+2., minlon) # distance is in m
            m               = Basemap(width = distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                                lat_1=minlat, lat_2=maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        # m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    def plot_stations(self, projection='lambert', geopolygons=None, showfig=True, blon=.5, blat=0.5):
        """plot station map
        ==============================================================================
        ::: input parameters :::
        projection      - type of geographical projection
        geopolygons     - geological polygons for plotting
        blon, blat      - extending boundaries in longitude/latitude
        showfig         - show figure or not
        ==============================================================================
        """
        staLst              = self.waveforms.list()
        stalons             = np.array([])
        stalats             = np.array([])
        for staid in staLst:
            stla, evz, stlo = self.waveforms[staid].coordinates.values()
            stalons         = np.append(stalons, stlo); stalats=np.append(stalats, stla)
        m                   = self._get_basemap(projection=projection, geopolygons=geopolygons, blon=blon, blat=blat)
        m.etopo()
        # m.shadedrelief()
        stax, stay          = m(stalons, stalats)
        m.plot(stax, stay, '^', markersize=5)
        # plt.title(str(self.period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        return
    
    def wsac_xcorr(self, netcode1, stacode1, netcode2, stacode2, chan1, chan2, outdir='.', pfx='COR'):
        """Write cross-correlation data from ASDF to sac file
        ==============================================================================
        ::: input parameters :::
        netcode1, stacode1, chan1   - network/station/channel name for station 1
        netcode2, stacode2, chan2   - network/station/channel name for station 2
        outdir                      - output directory
        pfx                         - prefix
        ::: output :::
        e.g. outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ==============================================================================
        """
        subdset                     = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sta1                        = self.waveforms[netcode1+'.'+stacode1].StationXML.networks[0].stations[0]
        sta2                        = self.waveforms[netcode2+'.'+stacode2].StationXML.networks[0].stations[0]
        xcorr_sacheader             = xcorr_sacheader_default.copy()
        xcorr_sacheader['kuser0']   = netcode1
        xcorr_sacheader['kevnm']    = stacode1
        xcorr_sacheader['knetwk']   = netcode2
        xcorr_sacheader['kstnm']    = stacode2
        xcorr_sacheader['kcmpnm']   = chan1+chan2
        xcorr_sacheader['evla']     = sta1.latitude
        xcorr_sacheader['evlo']     = sta1.longitude
        xcorr_sacheader['stla']     = sta2.latitude
        xcorr_sacheader['stlo']     = sta2.longitude
        xcorr_sacheader['dist']     = subdset.parameters['dist']
        xcorr_sacheader['az']       = subdset.parameters['az']
        xcorr_sacheader['baz']      = subdset.parameters['baz']
        xcorr_sacheader['b']        = subdset.parameters['b']
        xcorr_sacheader['e']        = subdset.parameters['e']
        xcorr_sacheader['delta']    = subdset.parameters['delta']
        xcorr_sacheader['npts']     = subdset.parameters['npts']
        xcorr_sacheader['user0']    = subdset.parameters['stackday']
        sacTr                       = obspy.io.sac.sactrace.SACTrace(data=subdset.data.value, **xcorr_sacheader)
        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
        sacfname                    = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
        sacTr.write(sacfname)
        return
    
    def wsac_xcorr_all(self, netcode1, stacode1, netcode2, stacode2, outdir='.', pfx='COR'):
        """Write all components of cross-correlation data from ASDF to sac file
        ==============================================================================
        ::: input parameters :::
        netcode1, stacode1  - network/station name for station 1
        netcode2, stacode2  - network/station name for station 2
        outdir              - output directory
        pfx                 - prefix
        ::: output :::
        e.g. outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ==============================================================================
        """
        subdset     = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2]
        channels1   = subdset.list()
        channels2   = subdset[channels1[0]].list()
        for chan1 in channels1:
            for chan2 in channels2:
                self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                    stacode2=stacode2, chan1=chan1, chan2=chan2, outdir=outdir, pfx=pfx)
        return
    
    def get_xcorr_trace(self, netcode1, stacode1, netcode2, stacode2, chan1, chan2):
        """Get one single cross-correlation trace
        ==============================================================================
        ::: input parameters :::
        netcode1, stacode1, chan1   - network/station/channel name for station 1
        netcode2, stacode2, chan2   - network/station/channel name for station 2
        ::: output :::
        obspy trace
        ==============================================================================
        """
        subdset             = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        evla, evz, evlo     = self.waveforms[netcode1+'.'+stacode1].coordinates.values()
        stla, stz, stlo     = self.waveforms[netcode2+'.'+stacode2].coordinates.values()
        tr                  = obspy.core.Trace()
        tr.data             = subdset.data.value
        tr.stats.sac        = {}
        tr.stats.sac.evla   = evla
        tr.stats.sac.evlo   = evlo
        tr.stats.sac.stla   = stla
        tr.stats.sac.stlo   = stlo
        tr.stats.sac.kuser0 = netcode1
        tr.stats.sac.kevnm  = stacode1
        tr.stats.network    = netcode2
        tr.stats.station    = stacode2
        tr.stats.sac.kcmpnm = chan1+chan2
        tr.stats.sac.dist   = subdset.parameters['dist']
        tr.stats.sac.az     = subdset.parameters['az']
        tr.stats.sac.baz    = subdset.parameters['baz']
        tr.stats.sac.b      = subdset.parameters['b']
        tr.stats.sac.e      = subdset.parameters['e']
        tr.stats.sac.user0  = subdset.parameters['stackday']
        tr.stats.delta      = subdset.parameters['delta']
        tr.stats.distance   = subdset.parameters['dist']*1000.
        return tr
    
    def get_xcorr_stream(self, netcode, stacode, chan1, chan2):
        st                  = obspy.Stream()
        stalst              = self.waveforms.list()
        for staid in stalst:
            netcode2, stacode2  \
                            = staid.split('.')
            try:
                st          += self.get_xcorr_trace(netcode1=netcode, stacode1=stacode, netcode2=netcode2, stacode2=stacode2, chan1=chan1, chan2=chan2)
            except KeyError:
                try:
                    st      += self.get_xcorr_trace(netcode1=netcode2, stacode1=stacode2, netcode2=netcode, stacode2=stacode, chan1=chan2, chan2=chan1)
                except KeyError:
                    pass
        
        return st
        
    
    def read_xcorr(self, datadir, pfx='COR', fnametype=1, inchannels=None, verbose=True):
        """Read cross-correlation data in ASDF database
        ===========================================================================================================
        ::: input parameters :::
        datadir                 - data directory
        pfx                     - prefix
        inchannels              - input channels, if None, will read channel information from obspy inventory
        fnametype               - input sac file name type
                                    =1: datadir/2011.JAN/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
                                    =2: datadir/2011.JAN/COR/G12A/COR_G12A_R21A.SAC
                                    =3: datadir/2011.JAN/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC
        -----------------------------------------------------------------------------------------------------------
        ::: output :::
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        ===========================================================================================================
        """
        staLst                  = self.waveforms.list()
        # main loop for station pairs
        if inchannels != None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels    = []
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels    = inchannels
            except:
                inchannels      = None
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                if fnametype==2 and not os.path.isfile(datadir+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+staid2+'.SAC'):
                    continue
                if inchannels==None:
                    channels1       = self.waveforms[staid1].StationXML.networks[0].stations[0].channels
                    channels2       = self.waveforms[staid2].StationXML.networks[0].stations[0].channels
                else:
                    channels1       = channels
                    channels2       = channels
                skipflag            = False
                for chan1 in channels1:
                    if skipflag:
                        break
                    for chan2 in channels2:
                        if fnametype    == 1:
                            fname   = datadir+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+chan1.code+'_'\
                                            +staid2+'_'+chan2.code+'.SAC'
                        elif fnametype  == 2:
                            fname   = datadir+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                        #----------------------------------------------------------
                        elif fnametype  == 3:
                            fname   = datadir+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'\
                                        +stacode2+'_'+chan2.code+'.SAC'
                        #----------------------------------------------------------
                        try:
                            tr      = obspy.core.read(fname)[0]
                        except IOError:
                            skipflag= True
                            break
                        # write cross-correlation header information
                        xcorr_header            = xcorr_header_default.copy()
                        xcorr_header['b']       = tr.stats.sac.b
                        xcorr_header['e']       = tr.stats.sac.e
                        xcorr_header['netcode1']= netcode1
                        xcorr_header['netcode2']= netcode2
                        xcorr_header['stacode1']= stacode1
                        xcorr_header['stacode2']= stacode2
                        xcorr_header['npts']    = tr.stats.npts
                        xcorr_header['delta']   = tr.stats.delta
                        xcorr_header['stackday']= tr.stats.sac.user0
                        try:
                            xcorr_header['dist']= tr.stats.sac.dist
                            xcorr_header['az']  = tr.stats.sac.az
                            xcorr_header['baz'] = tr.stats.sac.baz
                        except AttributeError:
                            lon1                = self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
                            lat1                = self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
                            lon2                = self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                            lat2                = self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                            dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                            dist                = dist/1000.
                            xcorr_header['dist']= dist
                            xcorr_header['az']  = az
                            xcorr_header['baz'] = baz
                        staid_aux               = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                        xcorr_header['chan1']   = chan1.code
                        xcorr_header['chan2']   = chan2.code
                        self.add_auxiliary_data(data=tr.data, data_type='NoiseXcorr', path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                if verbose and not skipflag:
                    print 'reading xcorr data: '+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2
        return
        
    def compute_xcorr(self, datadir, startdate, enddate, chans=['LHZ', 'LHE', 'LHN'], \
            fskipxcorr = 0, ftlen = True, tlen = 84000., mintlen = 20000., sps = 1., lagtime = 3000., CorOutflag = 0, \
                fprcs = False, fastfft=True, parallel=True, nprocess=None, subsize=1000):
        """
        compute ambient noise cross-correlation given preprocessed amplitude and phase files
        =================================================================================================================
        ::: input parameters :::
        datadir             - directory including data and output
        startdate/enddate   - start/end date for computation           
        chans               - channel list
        fskipxcorr          - skip flags: 1 = skip upon existence of target file, 0 = overwrites
        ftlen               - turn (on/off) cross-correlation-time-length correction for amplitude
        tlen                - time length of daily records (in sec)
        mintlen             - allowed minimum time length for cross-correlation (takes effect only when ftlen = True)
        sps                 - target sampling rate
        lagtime             - cross-correlation signal half length in sec
        CorOutflag          - 0 = only output monthly xcorr data, 1 = only daily, 2 or others = output both
        fprcs               - turn on/off (1/0) precursor signal checking, NOT implemented yet
        fastfft             - speeding up the computation by using precomputed fftw_plan or not
        parallel            - run the xcorr parallelly or not
        nprocess            - number of processes
        subsize             - subsize of processing list, use to prevent lock in multiprocessing process
        =================================================================================================================
        """
        stime   = obspy.UTCDateTime(startdate)
        etime   = obspy.UTCDateTime(enddate)
        #-------------------------
        # Loop over month
        #-------------------------
        while(stime < etime):
            print '=== Xcorr data preparing: '+str(stime.year)+'.'+monthdict[stime.month]
            month_dir   = datadir+'/'+str(stime.year)+'.'+monthdict[stime.month]
            if not os.path.isdir(month_dir):
                print '--- Xcorr dir NOT exists : '+str(stime.year)+'.'+monthdict[stime.month]
                continue
            # xcorr list
            xcorr_lst   = []
            # define the first day and last day of the current month
            c_stime     = obspy.UTCDateTime(str(stime.year)+'-'+str(stime.month)+'-1')
            try:
                c_etime = obspy.UTCDateTime(str(stime.year)+'-'+str(stime.month+1)+'-1')
            except ValueError:
                c_etime = obspy.UTCDateTime(str(stime.year+1)+'-1-1')
            #-------------------------
            # Loop over station 1
            #-------------------------
            for staid1 in self.waveforms.list():
                # determine if the range of the station 1 matches current month
                st_date1    = self.waveforms[staid1].StationXML.networks[0].stations[0].start_date
                ed_date1    = self.waveforms[staid1].StationXML.networks[0].stations[0].end_date
                if st_date1 > c_etime or ed_date1 < c_stime:
                    continue
                netcode1, stacode1  = staid1.split('.')
                #-------------------------
                # Loop over station 2
                #-------------------------
                for staid2 in self.waveforms.list():
                    if staid1 >= staid2:
                        continue
                    netcode2, stacode2  = staid2.split('.')
                    ###
                    # if staid1 != 'IU.COLA' or staid2 != 'XE.DH3':
                    #     continue
                    ###
                    # determine if the range of the station 2 matches current month
                    st_date2    = self.waveforms[staid2].StationXML.networks[0].stations[0].start_date
                    ed_date2    = self.waveforms[staid2].StationXML.networks[0].stations[0].end_date
                    if st_date2 > c_etime or ed_date2 < c_stime:
                        continue
                    ctime       = obspy.UTCDateTime(str(stime.year)+'-'+str(stime.month)+'-1')
                    # day list
                    daylst      = []
                    # Loop over days
                    while(True):
                        daydir  = month_dir+'/'+str(stime.year)+'.'+monthdict[stime.month]+'.'+str(ctime.day)
                        skipday = False
                        if os.path.isdir(daydir):
                            for chan in chans:
                                infname1    = daydir+'/ft_'+str(stime.year)+'.'+monthdict[stime.month]+'.'+str(ctime.day)+\
                                               '.'+staid1+'.'+chan+ '.SAC'
                                infname2    = daydir+'/ft_'+str(stime.year)+'.'+monthdict[stime.month]+'.'+str(ctime.day)+\
                                               '.'+staid2+'.'+chan+ '.SAC'
                                if os.path.isfile(infname1+'.am') and os.path.isfile(infname1+'.ph')\
                                        and os.path.isfile(infname2+'.am') and os.path.isfile(infname2+'.ph'):
                                    continue
                                else:
                                    skipday = True
                                    break
                            if not skipday:
                                daylst.append(ctime.day)
                        try:
                            ctime.day   += 1
                        except ValueError:
                            break
                    if len(daylst) != 0:
                        xcorr_lst.append( xcorr_pair(stacode1 = stacode1, netcode1=netcode1,\
                            stacode2=stacode2, netcode2=netcode2, monthdir=str(stime.year)+'.'+monthdict[stime.month], daylst=daylst) )
                        ###
                        # # # return xcorr_lst
                        ###
            # End loop over station1/station2/days
            if len(xcorr_lst) == 0:
                print '--- Xcorr NO data: '+str(stime.year)+'.'+monthdict[stime.month]+' : '+ str(len(xcorr_lst)) + ' pairs'
                if stime.month == 12:
                    stime       = obspy.UTCDateTime(str(stime.year + 1)+'0101')
                else:
                    stime.month += 1
                continue
            #--------------------------------
            # Cross-correlation computation
            #--------------------------------
            print '--- Xcorr computating: '+str(stime.year)+'.'+monthdict[stime.month]+' : '+ str(len(xcorr_lst)) + ' pairs'
            if not parallel:
                for ilst in range(len(xcorr_lst)):
                    xcorr_lst[ilst].convert_amph_to_xcorr(datadir=datadir, chans=chans, ftlen = ftlen,\
                            tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                fprcs = fprcs, fastfft=fastfft, verbose=False)
            # parallelized run
            else:
                #-----------------------------------------
                # Computing xcorr with multiprocessing
                #-----------------------------------------
                if len(xcorr_lst) > subsize:
                    Nsub            = int(len(xcorr_lst)/subsize)
                    for isub in range(Nsub):
                        print 'xcorr : subset:', isub, 'in', Nsub, 'sets'
                        cxcorrLst   = xcorr_lst[isub*subsize:(isub+1)*subsize]
                        XCORR       = partial(amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft)
                        pool        = multiprocessing.Pool(processes=nprocess)
                        pool.map(XCORR, cxcorrLst) #make our results with a map call
                        pool.close() #we are not adding any more processes
                        pool.join() #tell it to wait until all threads are done before going on
                    cxcorrLst       = xcorr_lst[(isub+1)*subsize:]
                    XCORR           = partial(amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(XCORR, cxcorrLst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                else:
                    XCORR           = partial(amph_to_xcorr_for_mp, datadir=datadir, chans=chans, ftlen = ftlen,\
                                        tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                                            fprcs = fprcs, fastfft=fastfft)
                    pool            = multiprocessing.Pool(processes=nprocess)
                    pool.map(XCORR, xcorr_lst) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
            print '=== Xcorr computation done: '+str(stime.year)+'.'+monthdict[stime.month]
            if stime.month == 12:
                stime       = obspy.UTCDateTime(str(stime.year + 1)+'0101')
            else:
                stime.month += 1
        return
    
    def xcorr_stack(self, datadir, startyear, startmonth, endyear, endmonth, pfx='COR', outdir=None, \
                inchannels=None, fnametype=1, verbose=False):
        """Stack cross-correlation data from monthly-stacked sac files
        ===========================================================================================================
        ::: input parameters :::
        datadir                 - data directory
        startyear, startmonth   - start date for stacking
        endyear, endmonth       - end date for stacking
        pfx                     - prefix
        outdir                  - output directory (None is not to save sac files)
        inchannels              - input channels, if None, will read channel information from obspy inventory
        fnametype               - input sac file name type
                                    =1: datadir/2011.JAN/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
                                    =2: datadir/2011.JAN/COR/G12A/COR_G12A_R21A.SAC
                                    =3: datadir/2011.JAN/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC, deprecated
        -----------------------------------------------------------------------------------------------------------
        ::: output :::
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ===========================================================================================================
        """
        #----------------------------------------
        # prepare year/month list for stacking
        #----------------------------------------
        print('=== preparing month list for stacking')
        utcdate                 = obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst                    = np.array([], dtype=int)
        mlst                    = np.array([], dtype=int)
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst                = np.append(ylst, utcdate.year)
            mlst                = np.append(mlst, utcdate.month)
            try:
                utcdate.month   +=1
            except ValueError:
                utcdate.year    +=1
                utcdate.month   = 1
        mnumb                   = mlst.size
        #--------------------------------------------------
        # determine channels if inchannels is specified
        #--------------------------------------------------
        if inchannels != None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels    = []
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels    = inchannels
            except:
                inchannels      = None
        if inchannels != None:
            chan_str_for_print      = ''
            for chan in channels:
                chan_str_for_print  += chan.code+' '
            print '--- channels for stacking : '+ chan_str_for_print
        #--------------------------------------------------
        # main loop for station pairs
        #--------------------------------------------------
        staLst                  = self.waveforms.list()
        Nsta                    = len(staLst)
        Ntotal_traces           = Nsta*(Nsta-1)/2
        itrstack                = 0
        Ntr_one_percent         = int(Ntotal_traces/100.)
        ipercent                = 0
        print '--- start stacking: '+str(Ntotal_traces)+' pairs'
        for staid1 in staLst:
            netcode1, stacode1  = staid1.split('.')
            st_date1            = self.waveforms[staid1].StationXML.networks[0].stations[0].start_date
            ed_date1            = self.waveforms[staid1].StationXML.networks[0].stations[0].end_date
            lon1                = self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
            lat1                = self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                st_date2            = self.waveforms[staid2].StationXML.networks[0].stations[0].start_date
                ed_date2            = self.waveforms[staid2].StationXML.networks[0].stations[0].end_date
                lon2                = self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                lat2                = self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                if fnametype == 1:
                    if staid1 >= staid2:
                        continue
                else:
                    if stacode1 >= stacode2:
                        continue
                itrstack            += 1
                # print the status of stacking
                ipercent            = float(itrstack)/float(Ntotal_traces)*100.
                if np.fmod(itrstack, 500) == 0 or np.fmod(itrstack, Ntr_one_percent) ==0:
                    percent_str     = '%0.2f' %ipercent
                    print '*** Number of traces finished stacking: '+str(itrstack)+'/'+str(Ntotal_traces)+' '+percent_str+'%'
                # skip if no overlaped time
                if st_date1 > ed_date2 or st_date2 > ed_date1:
                    continue
                stackedST           = []
                init_stack_flag     = False
                #-------------------------------------------------------------
                # determin channels for stacking if not specified beforehand
                #-------------------------------------------------------------
                if inchannels == None:
                    channels1       = []
                    channels2       = []
                    tempchans1      = self.waveforms[staid1].StationXML.networks[0].stations[0].channels
                    tempchans2      = self.waveforms[staid2].StationXML.networks[0].stations[0].channels
                    # get non-repeated component channel list
                    isZ             = False
                    isN             = False
                    isE             = False
                    for tempchan in tempchans1:
                        if tempchan.code[-1] == 'Z':
                            if isZ:
                                continue
                            else:
                                isZ         = True
                        if tempchan.code[-1] == 'N':
                            if isN:
                                continue
                            else:
                                isN         = True
                        if tempchan.code[-1] == 'E':
                            if isE:
                                continue
                            else:
                                isE         = True
                        channels1.append(tempchan)
                    isZ             = False
                    isN             = False
                    isE             = False
                    for tempchan in tempchans2:
                        if tempchan.code[-1] == 'Z':
                            if isZ:
                                continue
                            else:
                                isZ         = True
                        if tempchan.code[-1] == 'N':
                            if isN:
                                continue
                            else:
                                isN         = True
                        if tempchan.code[-1] == 'E':
                            if isE:
                                continue
                            else:
                                isE         = True
                        channels2.append(tempchan)
                else:
                    channels1       = channels
                    channels2       = channels
                #--------------------------------
                # Loop over month for stacking
                #--------------------------------
                for im in range(mnumb):
                    month           = monthdict[mlst[im]]
                    yrmonth         = str(ylst[im])+'.'+month
                    if fnametype == 1:
                        subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1
                    else:
                        subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1
                    if not os.path.isdir(subdir):
                        continue
                    # define the first day and last day of the current month
                    c_stime     = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im])+'-1')
                    try:
                        c_etime = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im]+1)+'-1')
                    except ValueError:
                        c_etime = obspy.UTCDateTime(str(ylst[im]+1)+'-1-1')
                    # skip if either of the stations out of time range
                    if st_date1 > c_etime or ed_date1 < c_stime or \
                        st_date2 > c_etime or ed_date2 < c_stime:
                        continue 
                    skip_this_month = False
                    cST             = []
                    for chan1 in channels1:
                        if skip_this_month:
                            break
                        for chan2 in channels2:
                            if fnametype    == 1:
                                fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+chan1.code+'_'\
                                            +staid2+'_'+chan2.code+'.SAC'
                            elif fnametype  == 2:
                                fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                            #----------------------------------------------------------
                            elif fnametype  == 3:
                                fname   = ''
                                # fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'\
                                #             +stacode2+'_'+chan2.code+'.SAC'
                            #----------------------------------------------------------
                            if not os.path.isfile(fname):
                                skip_this_month = True
                                break
                            try:
                                # I/O through obspy.io.sac.SACTrace.read() is ~ 10 times faster than obspy.read()
                                tr              = obspy.io.sac.SACTrace.read(fname)
                            except TypeError:
                                warnings.warn('Unable to read SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skip_this_month = True
                                break
                            # added on 2018-02-27
                            # # # if (abs(tr.stats.sac.evlo - lon1) > 0.001)\
                            # # #         or (abs(tr.stats.sac.evla - lat1) > 0.001) \
                            # # #         or (abs(tr.stats.sac.stlo - lon2) > 0.001) \
                            # # #         or (abs(tr.stats.sac.stla - lat2) > 0.001):
                            # # #     print 'WARNING: Same station code but different locations detected ' + staid1 +'_'+ staid2
                            # # #     print 'FILENAME: '+ fname
                            # # #     skipflag= True
                            # # #     break
                            if (np.isnan(tr.data)).any() or abs(tr.data.max())>1e20:
                                warnings.warn('NaN monthly SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skip_this_month = True
                                break
                            cST.append(tr)
                    if len(cST) != (len(channels1)*len(channels2)) or skip_this_month:
                        continue
                    # stacking
                    if init_stack_flag:
                        for itr in range(len(cST)):
                            mtr                             = cST[itr]
                            stackedST[itr].data             += mtr.data
                            stackedST[itr].user0            += mtr.user0
                    else:
                        stackedST                           = copy.deepcopy(cST)
                        init_stack_flag                     = True
                #------------------------------------------------------------
                # finish stacking for a statin pair, save data
                #------------------------------------------------------------
                if len(stackedST) == (len(channels1)*len(channels2)):
                    if verbose:
                        print('Finished stacking for:'+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2)
                    # create sac output directory 
                    if outdir != None:
                        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
                            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
                    # write cross-correlation header information
                    xcorr_header            = xcorr_header_default.copy()
                    xcorr_header['b']       = stackedST[0].b
                    xcorr_header['e']       = stackedST[0].e
                    xcorr_header['netcode1']= netcode1
                    xcorr_header['netcode2']= netcode2
                    xcorr_header['stacode1']= stacode1
                    xcorr_header['stacode2']= stacode2
                    xcorr_header['npts']    = stackedST[0].npts
                    xcorr_header['delta']   = stackedST[0].delta
                    xcorr_header['stackday']= stackedST[0].user0
                    dist, az, baz           = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                    dist                    = dist/1000.
                    xcorr_header['dist']    = dist
                    xcorr_header['az']      = az
                    xcorr_header['baz']     = baz
                    if staid1 > staid2:
                        staid_aux           = netcode2+'/'+stacode2+'/'+netcode1+'/'+stacode1
                    else:
                        staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                    itrace                  = 0
                    for chan1 in channels1:
                        for chan2 in channels2:
                            stackedTr       = stackedST[itrace]
                            if outdir != None:
                                outfname            = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ pfx+'_'+netcode1+'.'+stacode1+\
                                                        '_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                                stackedTr.write(outfname)
                            xcorr_header['chan1']   = chan1.code
                            xcorr_header['chan2']   = chan2.code
                            # check channels
                            if stackedST[itrace].kcmpnm != None:
                                if stackedST[itrace].kcmpnm != xcorr_header['chan1'] + xcorr_header['chan2']:
                                    raise ValueError('Inconsistent channels: '+ stackedST[itrace].kcmpnm+' '+\
                                                xcorr_header['chan1']+' '+ xcorr_header['chan2'])
                            self.add_auxiliary_data(data=stackedTr.data, data_type='NoiseXcorr',\
                                                    path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                            itrace                  += 1
        return
    
    def xcorr_stack_mp(self, datadir, outdir, startyear, startmonth, endyear, endmonth, pfx='COR', inchannels=None,\
                       fnametype=1, do_compute = True, subsize=1000, deletesac=False, nprocess=10):
        """Stack cross-correlation data from monthly-stacked sac files with multiprocessing
        ===========================================================================================================
        ::: input parameters :::
        datadir                 - data directory
        outdir                  - output directory 
        startyear, startmonth   - start date for stacking
        endyear, endmonth       - end date for stacking
        pfx                     - prefix
        inchannels              - input channels, if None, will read channel information from obspy inventory
        fnametype               - input sac file name type
                                    =1: datadir/2011.JAN/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
                                    =2: datadir/2011.JAN/COR/G12A/COR_G12A_R21A.SAC
                                    =3: datadir/2011.JAN/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC
        subsize                 - subsize of processing list, use to prevent lock in multiprocessing process
        deletesac               - delete output sac files
        nprocess                - number of processes
        -----------------------------------------------------------------------------------------------------------
        ::: output :::
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ===========================================================================================================
        """
        #----------------------------------------
        # prepare year/month list for stacking
        #----------------------------------------
        print('=== preparing month list for stacking')
        utcdate                 = obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst                    = np.array([], dtype=int)
        mlst                    = np.array([], dtype=int)
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst                = np.append(ylst, utcdate.year)
            mlst                = np.append(mlst, utcdate.month)
            try:
                utcdate.month   +=1
            except ValueError:
                utcdate.year    +=1
                utcdate.month   = 1
        mnumb                   = mlst.size
        #--------------------------------------------------
        # determine channels if inchannels is specified
        #--------------------------------------------------
        if inchannels != None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels    = []
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels    = inchannels
            except:
                inchannels      = None
        if inchannels != None:
            chan_str_for_print      = ''
            for chan in channels:
                chan_str_for_print  += chan.code+' '
            print '--- channels for stacking : '+ chan_str_for_print
        #--------------------------------------------------
        # main loop for station pairs
        #--------------------------------------------------
        print '--- Preparing station pair list for stacking'
        staLst                  = self.waveforms.list()
        stapairInvLst           = []
        Nsta                    = len(staLst)
        Ntotal_traces           = Nsta*(Nsta-1)/2
        itrstack                = 0
        Ntr_one_percent         = int(Ntotal_traces/100.)
        for staid1 in staLst:
            if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
                os.makedirs(outdir+'/'+pfx+'/'+staid1)
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if fnametype == 1:
                    if staid1 >= staid2:
                        continue
                else:
                    if stacode1 >= stacode2:
                        continue
                itrstack            += 1
                ipercent            = float(itrstack)/float(Ntotal_traces)*100.
                if np.fmod(itrstack, Ntr_one_percent) ==0:
                    percent_str     = '%1.0f' %ipercent
                    print '*** Number of traces finished preparing: '+str(itrstack)+'/'+str(Ntotal_traces)+' '+percent_str+'%'
                inv1                = self.waveforms[staid1].StationXML
                inv2                = self.waveforms[staid2].StationXML
                if inchannels != None:
                    inv1.networks[0].stations[0].channels   = channels
                    inv2.networks[0].stations[0].channels   = channels
                stapairInvLst.append([inv1, inv2]) 
        #------------------------------------------------------
        # Stacking with multiprocessing
        #------------------------------------------------------
        if do_compute:
            print('Start multiprocessing stacking !')
            if len(stapairInvLst) > subsize:
                Nsub            = int(len(stapairInvLst)/subsize)
                for isub in range(Nsub):
                    print 'xcorr stacking : ', isub,'in',Nsub
                    cstapairs   = stapairInvLst[isub*subsize:(isub+1)*subsize]
                    STACKING    = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
                    pool        = multiprocessing.Pool(processes=nprocess)
                    pool.map(STACKING, cstapairs) #make our results with a map call
                    pool.close() #we are not adding any more processes
                    pool.join() #tell it to wait until all threads are done before going on
                cstapairs       = stapairInvLst[(isub+1)*subsize:]
                STACKING        = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(STACKING, cstapairs) 
                pool.close() 
                pool.join() 
            else:
                STACKING        = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(STACKING, stapairInvLst) 
                pool.close() 
                pool.join() 
            print('End of multiprocessing stacking !')
        #------------------------------------------------------
        # read stacked data
        #------------------------------------------------------
        print('Reading data into ASDF database')
        itrstack                    = 0
        for invpair in stapairInvLst:
            # station 1
            channels1               = invpair[0].networks[0].stations[0].channels
            netcode1                = invpair[0].networks[0].code
            stacode1                = invpair[0].networks[0].stations[0].code
            # station 2
            channels2               = invpair[1].networks[0].stations[0].channels
            netcode2                = invpair[1].networks[0].code
            stacode2                = invpair[1].networks[0].stations[0].code
            skipflag                = False
            xcorr_header            = xcorr_header_default.copy()
            xcorr_header['netcode1']= netcode1
            xcorr_header['netcode2']= netcode2
            xcorr_header['stacode1']= stacode1
            xcorr_header['stacode2']= stacode2
            if staid1 > staid2:
                staid_aux           = netcode2+'/'+stacode2+'/'+netcode1+'/'+stacode1
            else:
                staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
            itrstack                += 1
            ipercent                = float(itrstack)/float(Ntotal_traces)*100.
            if np.fmod(itrstack, Ntr_one_percent) ==0:
                percent_str         = '%0.1f' %ipercent
                print '*** Number of traces finished preparing: '+str(itrstack)+'/'+str(Ntotal_traces)+' '+percent_str+'%'
            for chan1 in channels1:
                if skipflag:
                    break
                for chan2 in channels2:
                    sacfname        = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                    try:
                        tr                      = obspy.read(sacfname)[0]
                        # cross-correlation header 
                        xcorr_header['b']       = tr.stats.sac.b
                        xcorr_header['e']       = tr.stats.sac.e
                        xcorr_header['npts']    = tr.stats.npts
                        xcorr_header['delta']   = tr.stats.delta
                        xcorr_header['stackday']= tr.stats.sac.user0
                        try:
                            xcorr_header['dist']= tr.stats.sac.dist
                            xcorr_header['az']  = tr.stats.sac.az
                            xcorr_header['baz'] = tr.stats.sac.baz
                        except AttributeError:
                            lon1                = invpair[0].networks[0].stations[0].longitude
                            lat1                = invpair[0].networks[0].stations[0].latitude
                            lon2                = invpair[1].networks[0].stations[0].longitude
                            lat2                = invpair[1].networks[0].stations[0].latitude
                            dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                            dist                = dist/1000.
                            xcorr_header['dist']= dist
                            xcorr_header['az']  = az
                            xcorr_header['baz'] = baz
                        xcorr_header['chan1']   = chan1.code
                        xcorr_header['chan2']   = chan2.code
                        self.add_auxiliary_data(data=tr.data, data_type='NoiseXcorr', \
                                        path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                    except IOError:
                        skipflag                = True
                        break
        if deletesac:
            shutil.rmtree(outdir+'/'+pfx)
        print('End reading data into ASDF database')
        return
                    
    def xcorr_append(self, inasdffname, datadir, startyear, startmonth, endyear, endmonth,\
                        pfx='COR', outdir=None, inchannels=None, fnametype=1, verbose=False):
        """Append cross-correlation data from monthly-stacked sac files to an existing ASDF database
        ===========================================================================================================
        ::: input parameters :::
        inasdffname             - input ASDF file name
        datadir                 - data directory
        startyear, startmonth   - start date for stacking
        endyear, endmonth       - end date for stacking
        pfx                     - prefix
        outdir                  - output directory (None is not to save sac files)
        inchannels              - input channels, if None, will read channel information from obspy inventory
        fnametype               - input sac file name type
                                    =1: datadir/2011.JAN/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
                                    =2: datadir/2011.JAN/COR/G12A/COR_G12A_R21A.SAC
                                    =3: datadir/2011.JAN/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC
        -----------------------------------------------------------------------------------------------------------
        ::: output :::
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHZ_TA.R21A_BHZ.SAC
        ===========================================================================================================
        """
        #----------------------------------------
        # copy station inventory
        #----------------------------------------
        indset                  = noiseASDF(inasdffname)
        stalst                  = indset.waveforms.list()
        for staid in stalst:
            self.add_stationxml(indset.waveforms[staid].StationXML)
        #----------------------------------------
        # prepare year/month list for stacking
        #----------------------------------------
        print('=== preparing month list for appending')
        utcdate                 = obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst                    = np.array([], dtype=int)
        mlst                    = np.array([], dtype=int)
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst                = np.append(ylst, utcdate.year)
            mlst                = np.append(mlst, utcdate.month)
            try:
                utcdate.month   +=1
            except ValueError:
                utcdate.year    +=1
                utcdate.month   = 1
        mnumb                   = mlst.size
        # # print mlst, ylst
        #--------------------------------------------------
        # determine channels if inchannels is specified
        #--------------------------------------------------
        if inchannels != None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels    = []
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels    = inchannels
            except:
                inchannels      = None
        if inchannels != None:
            chan_str_for_print      = ''
            for chan in channels:
                chan_str_for_print  += chan.code+' '
            print '--- channels for stacking : '+ chan_str_for_print
        #--------------------------------------------------
        # main loop for station pairs
        #--------------------------------------------------
        staLst                  = self.waveforms.list()
        Nsta                    = len(staLst)
        Ntotal_traces           = Nsta*(Nsta-1)/2
        itrstack                = 0
        Ntr_one_percent         = int(Ntotal_traces/100.)
        ipercent                = 0
        print '--- start stacking: '+str(Ntotal_traces)+' pairs'
        for staid1 in staLst:
            netcode1, stacode1  = staid1.split('.')
            lon1                = self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
            lat1                = self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
            st_date1            = self.waveforms[staid1].StationXML.networks[0].stations[0].start_date
            ed_date1            = self.waveforms[staid1].StationXML.networks[0].stations[0].end_date
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                lon2                = self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                lat2                = self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                st_date2            = self.waveforms[staid2].StationXML.networks[0].stations[0].start_date
                ed_date2            = self.waveforms[staid2].StationXML.networks[0].stations[0].end_date
                if fnametype == 1:
                    if staid1 >= staid2:
                        continue
                else:
                    if stacode1 >= stacode2:
                        continue
                itrstack            += 1
                # print the status of stacking
                ipercent            = float(itrstack)/float(Ntotal_traces)*100.
                if np.fmod(itrstack, 500) == 0 or np.fmod(itrstack, Ntr_one_percent) ==0:
                    percent_str     = '%0.2f' %ipercent
                    print '*** Number of traces finished stacking: '+str(itrstack)+'/'+str(Ntotal_traces)+\
                                ' '+percent_str+'%'
                # if no overlaped time
                if st_date1 > ed_date2 or st_date2 > ed_date1:
                    continue
                stackedST           = []
                init_stack_flag     = False
                # determine channels if not specified
                if inchannels is None:
                    channels1       = []
                    channels2       = []
                    tempchans1      = self.waveforms[staid1].StationXML.networks[0].stations[0].channels
                    tempchans2      = self.waveforms[staid2].StationXML.networks[0].stations[0].channels
                    # get non-repeated component channel list
                    isZ             = False
                    isN             = False
                    isE             = False
                    for tempchan in tempchans1:
                        if tempchan.code[-1] == 'Z':
                            if isZ:
                                continue
                            else:
                                isZ         = True
                        if tempchan.code[-1] == 'N':
                            if isN:
                                continue
                            else:
                                isN         = True
                        if tempchan.code[-1] == 'E':
                            if isE:
                                continue
                            else:
                                isE         = True
                        channels1.append(tempchan)
                    isZ             = False
                    isN             = False
                    isE             = False
                    for tempchan in tempchans2:
                        if tempchan.code[-1] == 'Z':
                            if isZ:
                                continue
                            else:
                                isZ         = True
                        if tempchan.code[-1] == 'N':
                            if isN:
                                continue
                            else:
                                isN         = True
                        if tempchan.code[-1] == 'E':
                            if isE:
                                continue
                            else:
                                isE         = True
                        channels2.append(tempchan)
                else:
                    channels1       = channels
                    channels2       = channels
                # Loop over months
                for im in range(mnumb):
                    month           = monthdict[mlst[im]]
                    yrmonth         = str(ylst[im])+'.'+month
                    # skip if directory not exist
                    if fnametype == 1:
                        subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1
                    else:
                        subdir      = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1
                    if not os.path.isdir(subdir):
                        continue
                    # define the first day and last day of the current month
                    c_stime     = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im])+'-1')
                    try:
                        c_etime = obspy.UTCDateTime(str(ylst[im])+'-'+str(mlst[im]+1)+'-1')
                    except ValueError:
                        c_etime = obspy.UTCDateTime(str(ylst[im]+1)+'-1-1')
                    # skip if either of the stations out of time range
                    if st_date1 > c_etime or ed_date1 < c_stime or \
                        st_date2 > c_etime or ed_date2 < c_stime:
                        continue 
                    skip_this_month = False
                    cST             = []
                    for chan1 in channels1:
                        if skip_this_month:
                            break
                        for chan2 in channels2:
                            month       = monthdict[mlst[im]]
                            yrmonth     = str(ylst[im])+'.'+month
                            if fnametype    == 1:
                                fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+chan1.code+'_'\
                                            +staid2+'_'+chan2.code+'.SAC'
                            elif fnametype  == 2:
                                fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                            #----------------------------------------------------------
                            elif fnametype  == 3:
                                fname   = ''
                                # fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'\
                                #             +stacode2+'_'+chan2.code+'.SAC'
                            #----------------------------------------------------------
                            if not os.path.isfile(fname):
                                skip_this_month = True
                                break
                            try:
                                # I/O through obspy.io.sac.SACTrace.read() is ~ 10 times faster than obspy.read()
                                tr              = obspy.io.sac.SACTrace.read(fname)
                            except TypeError:
                                warnings.warn('Unable to read SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skip_this_month = True
                                break
                            if (np.isnan(tr.data)).any() or abs(tr.data.max())>1e20:
                                warnings.warn('NaN monthly SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skip_this_month = True
                                break
                            cST.append(tr)
                    if len(cST) != (len(channels1)*len(channels2)) or skip_this_month:
                        continue
                    if init_stack_flag:
                        for itr in range(len(cST)):
                            mtr                     = cST[itr]
                            stackedST[itr].data     += mtr.data
                            stackedST[itr].user0    += mtr.user0
                    else:
                        stackedST                   = copy.deepcopy(cST)
                        init_stack_flag             = True
                # end of loop over month
                if len(stackedST)== (len(channels1)*len(channels2)):
                    if verbose:
                        print('Finished stacking for:'+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2)
                    # create sac output directory 
                    if outdir is not None:
                        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
                            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
                    #----------------------------------------------
                    # write cross-correlation header information
                    #----------------------------------------------
                    xcorr_header            = xcorr_header_default.copy()
                    xcorr_header['b']       = stackedST[0].b
                    xcorr_header['e']       = stackedST[0].e
                    xcorr_header['netcode1']= netcode1
                    xcorr_header['netcode2']= netcode2
                    xcorr_header['stacode1']= stacode1
                    xcorr_header['stacode2']= stacode2
                    xcorr_header['npts']    = stackedST[0].npts
                    xcorr_header['delta']   = stackedST[0].delta
                    xcorr_header['stackday']= stackedST[0].user0
                    dist, az, baz           = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                    dist                    = dist/1000.
                    xcorr_header['dist']    = dist
                    xcorr_header['az']      = az
                    xcorr_header['baz']     = baz
                    # determine whether append data or not
                    is_append               = True
                    if staid1 > staid2:
                        staid_aux                   = netcode2+'/'+stacode2+'/'+netcode1+'/'+stacode1
                        try:
                            instapair               = indset.auxiliary_data['NoiseXcorr'][netcode2][stacode2][netcode1][stacode1]
                        except KeyError:
                            is_append               = False
                    else:
                        staid_aux                   = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                        try:
                            instapair               = indset.auxiliary_data['NoiseXcorr'][netcode1][stacode1][netcode2][stacode2]
                        except KeyError:
                            is_append               = False
                    itrace                          = 0
                    for chan1 in channels1:
                        for chan2 in channels2:
                            stackedTr               = stackedST[itrace]
                            if outdir is not None:
                                outfname            = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                                stackedTr.write(outfname)
                            xcorr_header['chan1']   = chan1.code
                            xcorr_header['chan2']   = chan2.code
                            if is_append:
                                #---------------------------------
                                # get data from original database
                                #---------------------------------
                                indata              = instapair[chan1.code][chan2.code]
                                orig_sdays          = indata.parameters['stackday']
                                orig_data           = indata.data.value
                                xcorr_header['stackday']\
                                                    += orig_sdays
                                stackedTr.data      += orig_data
                                self.add_auxiliary_data(data = stackedTr.data, data_type='NoiseXcorr', \
                                                    path = staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                            else:
                                self.add_auxiliary_data(data = stackedTr.data, data_type='NoiseXcorr', \
                                                    path = staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                            itrace                  += 1
                else:
                    #--------------------------
                    # copy existing xcorr data
                    #--------------------------
                    is_copy                         = True
                    if staid1 > staid2:
                        staid_aux                   = netcode2+'/'+stacode2+'/'+netcode1+'/'+stacode1
                        try:
                            instapair               = indset.auxiliary_data['NoiseXcorr'][netcode2][stacode2][netcode1][stacode1]
                        except KeyError:
                            is_copy                 = False
                    else:
                        staid_aux                   = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                        try:
                            instapair               = indset.auxiliary_data['NoiseXcorr'][netcode1][stacode1][netcode2][stacode2]
                        except KeyError:
                            is_copy                 = False
                    if is_copy:
                        for chan1 in channels1:
                            for chan2 in channels2:
                                indata              = instapair[chan1.code][chan2.code]
                                xcorr_header        = indata.parameters
                                xcorr_data          = indata.data.value
                                self.add_auxiliary_data(data=xcorr_data, data_type='NoiseXcorr', \
                                            path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
        return
    
    def xcorr_rotation(self, outdir = None, pfx = 'COR', verbose=False):
        """Rotate cross-correlation data 
        ===========================================================================================================
        ::: input parameters :::
        outdir                  - output directory for sac files (None is not to write)
        pfx                     - prefix
        -----------------------------------------------------------------------------------------------------------
        ::: output :::
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ===========================================================================================================
        """
        staLst                  = self.waveforms.list()
        Nsta                    = len(staLst)
        Ntotal_traces           = Nsta*(Nsta-1)/2
        itrstack                = 0
        Ntr_one_percent         = int(Ntotal_traces/100.)
        irotate                 = 0
        print '=== start rotation: '+str(Ntotal_traces)+' pairs'
        for staid1 in staLst:
            lon1                    = self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
            lat1                    = self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                lon2                = self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                lat2                = self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                irotate            += 1
                # print the status of rotation
                ipercent            = float(irotate)/float(Ntotal_traces)*100.
                if np.fmod(irotate, 500) == 0 or np.fmod(irotate, Ntr_one_percent) == 0:
                    percent_str     = '%0.2f' %ipercent
                    print '*** Number of traces finished rotation: '+str(irotate)+'/'+str(Ntotal_traces)+' '+percent_str+'%'
                #-------------------------
                # determine the channels
                #-------------------------
                chan1E  = None
                chan1N  = None
                chan1Z  = None
                chan2E  = None
                chan2N  = None
                chan2Z  = None
                try:
                    channels1   = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2   = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    cpfx1       = channels1[0][:2]
                    cpfx2       = channels2[0][:2]
                    for chan in channels1:
                        if chan[2]=='E':
                            chan1E  = chan
                        if chan[2]=='N':
                            chan1N  = chan
                        if chan[2]=='Z':
                            chan1Z  = chan
                    for chan in channels2:
                        if chan[2]=='E':
                            chan2E  = chan
                        if chan[2]=='N':
                            chan2N  = chan
                        if chan[2]=='Z':
                            chan2Z  = chan
                except KeyError:
                    continue
                subdset             = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2]
                if chan1E==None or chan1N==None or chan2E==None or chan2N==None:
                    continue
                if chan1Z==None or chan2Z==None:
                    if verbose:
                        print 'Do rotation(RT) for:'+staid1+' and '+staid2
                else:
                    if verbose:
                        print 'Do rotation(RTZ) for:'+staid1+' and '+staid2
                # get data
                dsetEE      = subdset[chan1E][chan2E]
                dsetEN      = subdset[chan1E][chan2N]
                dsetNE      = subdset[chan1N][chan2E]
                dsetNN      = subdset[chan1N][chan2N]
                temp_header = dsetEE.parameters.copy()
                chan1R      = cpfx1+'R'
                chan1T      = cpfx1+'T'
                chan2R      = cpfx2+'R'
                chan2T      = cpfx2+'T'
                # define azimuth/back-azimuth
                theta           = temp_header['az']
                psi             = temp_header['baz']
                # check az/baz
                dist, az, baz   = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                dist            = dist/1000.
                if abs(az - theta) > 0.01 and abs(abs(az - theta) - 360.) > 0.01:
                    raise ValueError('computed az = '+str(az)+' stored az = '+str(theta)+' '+staid1+'_'+staid2)
                if abs(baz - psi) > 0.01 and abs(abs(baz - psi) - 360.) > 0.01:
                    raise ValueError('computed baz = '+str(baz)+' stored baz = '+str(psi)+' '+staid1+'_'+staid2)
                Ctheta      = np.cos(np.pi*theta/180.)
                Stheta      = np.sin(np.pi*theta/180.)
                Cpsi        = np.cos(np.pi*psi/180.)
                Spsi        = np.sin(np.pi*psi/180.)
                #------------------------------- perform EN -> RT rotation ------------------------------
                tempTT      = -Ctheta*Cpsi* dsetEE.data.value + Ctheta*Spsi* dsetEN.data.value - \
                                    Stheta*Spsi* dsetNN.data.value + Stheta*Cpsi* dsetNE.data.value
                
                tempRR      = - Stheta*Spsi* dsetEE.data.value - Stheta*Cpsi* dsetEN.data.value \
                                    -Ctheta*Cpsi*dsetNN.data.value - Ctheta*Spsi*dsetNE.data.value
                
                tempTR      = -Ctheta*Spsi* dsetEE.data.value - Ctheta*Cpsi* dsetEN.data.value  \
                                    + Stheta*Cpsi*dsetNN.data.value + Stheta*Spsi*dsetNE.data.value
                
                tempRT      = -Stheta*Cpsi* dsetEE.data.value + Stheta*Spsi* dsetEN.data.value \
                                    + Ctheta*Spsi* dsetNN.data.value - Ctheta*Cpsi* dsetNE.data.value
                #----------------------------------------------------------------------------------------
                # save horizontal components
                staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                temp_header['chan1']= chan1T
                temp_header['chan2']= chan2T
                self.add_auxiliary_data(data=tempTT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2T, parameters=temp_header)
                
                temp_header['chan1']= chan1R
                temp_header['chan2']= chan2R
                self.add_auxiliary_data(data=tempRR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2R, parameters=temp_header)
                
                temp_header['chan1']= chan1T
                temp_header['chan2']= chan2R
                self.add_auxiliary_data(data=tempTR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2R, parameters=temp_header)
                
                temp_header['chan1']= chan1R
                temp_header['chan2']= chan2T
                self.add_auxiliary_data(data=tempRT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2T, parameters=temp_header)
                # write to sac files
                if outdir != None:
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1T, chan2=chan2T, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1R, chan2=chan2R, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1T, chan2=chan2R, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1R, chan2=chan2T, outdir=outdir, pfx=pfx)
                # RTZ rotation
                if chan1Z != None and chan2Z != None:
                    # get data
                    dsetEZ  = subdset[chan1E][chan2Z]
                    dsetZE  = subdset[chan1Z][chan2E]
                    dsetNZ  = subdset[chan1N][chan2Z]
                    dsetZN  = subdset[chan1Z][chan2N]
                    # ----------------------- perform ENZ -> RTZ rotation ---------------------
                    tempRZ  = Ctheta* dsetNZ.data.value + Stheta* dsetEZ.data.value
                    tempZR  = -Cpsi* dsetZN.data.value - Spsi* dsetZE.data.value
                    tempTZ  = -Stheta* dsetNZ.data.value + Ctheta* dsetEZ.data.value
                    tempZT  = Spsi* dsetZN.data.value - Cpsi* dsetZE.data.value
                    #--------------------------------------------------------------------------
                    temp_header['chan1']    = chan1R
                    temp_header['chan2']    = chan2Z
                    self.add_auxiliary_data(data=tempRZ, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2Z, parameters=temp_header)
                    temp_header['chan1']    = chan1Z
                    temp_header['chan2']    = chan2R
                    self.add_auxiliary_data(data=tempZR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1Z+'/'+chan2R, parameters=temp_header)
                    temp_header['chan1']    = chan1T
                    temp_header['chan2']    = chan2Z
                    self.add_auxiliary_data(data=tempTZ, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2Z, parameters=temp_header)
                    temp_header['chan1']    = chan1Z
                    temp_header['chan2']    = chan2T
                    self.add_auxiliary_data(data=tempZT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1Z+'/'+chan2T, parameters=temp_header)
                    # write to sac files
                    if outdir!=None:
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1R, chan2=chan2Z, outdir=outdir, pfx=pfx)                        
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1Z, chan2=chan2R, outdir=outdir, pfx=pfx)
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1T, chan2=chan2Z, outdir=outdir, pfx=pfx)
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1Z, chan2=chan2T, outdir=outdir, pfx=pfx)
        return
    
    def count_data(self, chan1='LHZ', chan2='LHZ', threshstackday=0):
        """count the number of available xcorr traces
        """
        staLst                      = self.waveforms.list()
        Nsta                        = len(staLst)
        Ntotal_traces               = Nsta*(Nsta-1)/2
        itrace                      = 0
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                try:
                    subdset         = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
                except KeyError:
                    continue
                if threshstackday > 0:
                    if subdset.parameters['stackday'] < threshstackday:
                        continue
                itrace              += 1
                # try:
                #     channels1       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                #     channels2       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                #     for chan in channels1:
                #         if chan[-1]==channel[0]:
                #             chan1   = chan
                #     for chan in channels2:
                #         if chan[-1]==channel[1]:
                #             chan2   = chan
                # except KeyError:
                #     continue
                
                # try:
                #     tr              = self.get_xcorr_trace(netcode1, stacode1, netcode2, stacode2, chan1, chan2)
                #     if tr.stats.sac.user0 < threshstackday:
                #         continue
                #     itrace          += 1
                # except NameError:
                #     pass
        print('Number of available xcorr traces: '+str(itrace)+'/'+str(Ntotal_traces))
        return
    
    def xcorr_prephp(self, outdir, mapfile='./MAPS/smpkolya_phv'):
        """
        Generate predicted phase velocity dispersion curves for cross-correlation pairs
        ====================================================================================
        ::: input parameters :::
        outdir  - output directory
        mapfile - phase velocity maps
        ------------------------------------------------------------------------------------
        Input format:
        prephaseEXE pathfname mapfile perlst staname
        
        Output format:
        outdirL(outdirR)/evid.staid.pre
        ====================================================================================
        """
        staLst                      = self.waveforms.list()
        for evid in staLst:
            evnetcode, evstacode    = evid.split('.')
            evla, evz, evlo         = self.waveforms[evid].coordinates.values()
            pathfname               = evid+'_pathfile'
            prephaseEXE             = './mhr_grvel_predict/lf_mhr_predict_earth'
            perlst                  = './mhr_grvel_predict/perlist_phase'
            if not os.path.isfile(prephaseEXE):
                print 'lf_mhr_predict_earth executable does not exist!'
                return
            if not os.path.isfile(perlst):
                print 'period list does not exist!'
                return
            with open(pathfname,'w') as f:
                ista                = 0
                for station_id in staLst:
                    stacode         = station_id.split('.')[1]
                    if evid >= station_id:
                        continue
                    stla, stz, stlo = self.waveforms[station_id].coordinates.values()
                    if ( abs(stlo-evlo) < 0.1 and abs(stla-evla)<0.1 ):
                        continue
                    ista            = ista+1
                    f.writelines('%5d%5d %15s %15s %10.5f %10.5f %10.5f %10.5f \n'
                            %(1, ista, evid, station_id, evla, evlo, stla, stlo ))
            call([prephaseEXE, pathfname, mapfile, perlst, evid])
            os.remove(pathfname)
            outdirL                 = outdir+'_L'
            outdirR                 = outdir+'_R'
            if not os.path.isdir(outdirL):
                os.makedirs(outdirL)
            if not os.path.isdir(outdirR):
                os.makedirs(outdirR)
            fout                    = open(evid+'_temp','wb')
            for l1 in open('PREDICTION_L'+'_'+evid):
                l2          = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[3],l2[4])
                    fout    = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[2],l2[3])
                    fout    = open(outname,"w")                
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            for l1 in open('PREDICTION_R'+'_'+evid):
                l2          = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[3],l2[4])
                    fout    = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[2],l2[3])
                    fout    = open(outname,"w")         
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            fout.close()
            os.remove(evid+'_temp')
            os.remove('PREDICTION_L'+'_'+evid)
            os.remove('PREDICTION_R'+'_'+evid)
        return
    
    def xcorr_aftan(self, channel='ZZ', tb=0., outdir=None, inftan=pyaftan.InputFtanParam(),\
            basic1=True, basic2=True, pmf1=True, pmf2=True, verbose=False, prephdir=None, f77=True, pfx='DISP'):
        """ aftan analysis of cross-correlation data 
        =======================================================================================
        ::: input parameters :::
        channel     - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        tb          - begin time (default = 0.0)
        outdir      - directory for output disp txt files (default = None, no txt output)
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        prephdir    - directory for predicted phase velocity dispersion curve
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        ---------------------------------------------------------------------------------------
        ::: output :::
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        print '=== start aftan analysis'
        staLst                      = self.waveforms.list()
        Nsta                        = len(staLst)
        Ntotal_traces               = Nsta*(Nsta-1)/2
        iaftan                      = 0
        Ntr_one_percent             = int(Ntotal_traces/100.)
        ipercent                    = 0
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                # print how many traces has been processed
                iaftan              += 1
                if np.fmod(iaftan, Ntr_one_percent) ==0:
                    ipercent        += 1
                    print ('*** Number of traces finished aftan analysis: '+str(iaftan)+'/'+\
                           str(Ntotal_traces)+' '+str(ipercent)+'%')
                # determine channels
                try:
                    channels1       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    for chan in channels1:
                        if chan[-1] == channel[0]:
                            chan1   = chan
                            break
                    channels2       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1].list()
                    for chan in channels2:
                        if chan[-1] == channel[1]:
                            chan2   = chan
                            break
                except KeyError:
                    continue
                # get data
                try:
                    tr              = self.get_xcorr_trace(netcode1, stacode1, netcode2, stacode2, chan1, chan2)
                except NameError:
                    print netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+' not exists!'
                    continue
                aftanTr             = pyaftan.aftantrace(tr.data, tr.stats)
                if abs(aftanTr.stats.sac.b+aftanTr.stats.sac.e) < aftanTr.stats.delta:
                    aftanTr.makesym()
                else:
                    print netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel+' NOT symmetric'
                    continue
                if prephdir != None:
                    phvelname       = prephdir + "/%s.%s.pre" %(netcode1+'.'+stacode1, netcode2+'.'+stacode2)
                else:
                    phvelname       = ''
                if not os.path.isfile(phvelname):
                    print phvelname+' not exists!'
                    continue
                if f77:
                    aftanTr.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                            npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
                else:
                    aftanTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                            npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
                if verbose:
                    print 'aftan analysis for: ' + netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                aftanTr.get_snr(ffact=inftan.ffact) # SNR analysis
                staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                # save aftan results to ASDF dataset
                if basic1:
                    parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6,\
                                            'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_1}
                    self.add_auxiliary_data(data=aftanTr.ftanparam.arr1_1, data_type='DISPbasic1', path=staid_aux,\
                                            parameters=parameters)
                if basic2:
                    parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6,\
                                            'amp': 7, 'Np': aftanTr.ftanparam.nfout2_1}
                    self.add_auxiliary_data(data=aftanTr.ftanparam.arr2_1, data_type='DISPbasic2', path=staid_aux,\
                                            parameters=parameters)
                if inftan.pmf:
                    if pmf1:
                        parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6,\
                                            'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_2}
                        self.add_auxiliary_data(data=aftanTr.ftanparam.arr1_2, data_type='DISPpmf1', path=staid_aux,\
                                                parameters=parameters)
                    if pmf2:
                        parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6,\
                                            'amp': 7, 'snr':8, 'Np': aftanTr.ftanparam.nfout2_2}
                        self.add_auxiliary_data(data=aftanTr.ftanparam.arr2_2, data_type='DISPpmf2', path=staid_aux,\
                                                parameters=parameters)
                if outdir != None:
                    if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
                        os.makedirs(outdir+'/'+pfx+'/'+staid1)
                    foutPR          = outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
                    aftanTr.ftanparam.writeDISP(foutPR)
        print '== end aftan analysis'
        return
               
    def xcorr_aftan_mp(self, outdir, channel='ZZ', tb=0., inftan=pyaftan.InputFtanParam(), basic1=True, basic2=True,
            pmf1=True, pmf2=True, verbose=True, prephdir=None, f77=True, pfx='DISP', subsize=1000, deletedisp=True, nprocess=None):
        """ aftan analysis of cross-correlation data with multiprocessing
        =======================================================================================
        ::: input parameters :::
        channel     - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        tb          - begin time (default = 0.0)
        outdir      - directory for output disp binary files
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        prephdir    - directory for predicted phase velocity dispersion curve
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
        deletedisp  - delete output dispersion files or not
        nprocess    - number of processes
        ---------------------------------------------------------------------------------------
        ::: output :::
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        print 'Preparing data for aftan analysis !'
        staLst                      = self.waveforms.list()
        inputStream                 = []
        for staid1 in staLst:
            if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
                os.makedirs(outdir+'/'+pfx+'/'+staid1)
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                try:
                    channels1       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    for chan in channels1:
                        if chan[2]==channel[0]:
                            chan1   = chan
                    for chan in channels2:
                        if chan[2]==channel[1]:
                            chan2   = chan
                except KeyError:
                    continue
                try:
                    tr              = self.get_xcorr_trace(netcode1, stacode1, netcode2, stacode2, chan1, chan2)
                except NameError:
                    print netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel+' not exists!'
                    continue
                if verbose:
                    print 'preparing aftan data: '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                aftanTr             = pyaftan.aftantrace(tr.data, tr.stats)
                inputStream.append(aftanTr)
        print 'Start multiprocessing aftan analysis !'
        if len(inputStream) > subsize:
            Nsub                    = int(len(inputStream)/subsize)
            for isub in range(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cstream             = inputStream[isub*subsize:(isub+1)*subsize]
                AFTAN               = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
                pool                = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN, cstream) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cstream                 = inputStream[(isub+1)*subsize:]
            AFTAN                   = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
            pool                    = multiprocessing.Pool(processes=nprocess)
            pool.map(AFTAN, cstream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            AFTAN                   = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
            pool                    = multiprocessing.Pool(processes=nprocess)
            pool.map(AFTAN, inputStream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print 'End of multiprocessing aftan analysis !'
        print 'Reading aftan results into ASDF Dataset !'
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if stacode1 >= stacode2:
                    continue
                try:
                    channels1       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    for chan in channels1:
                        if chan[2]==channel[0]:
                            chan1   = chan
                    for chan in channels2:
                        if chan[2]==channel[1]:
                            chan2   = chan
                except KeyError:
                    continue
                finPR               = pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
                try:
                    f10             = np.load(outdir+'/'+finPR+'_1_DISP.0.npz')
                    f11             = np.load(outdir+'/'+finPR+'_1_DISP.1.npz')
                    f20             = np.load(outdir+'/'+finPR+'_2_DISP.0.npz')
                    f21             = np.load(outdir+'/'+finPR+'_2_DISP.1.npz')
                except IOError:
                    print 'NO aftan results: '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                    continue
                print 'Reading aftan results '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                if deletedisp:
                    os.remove(outdir+'/'+finPR+'_1_DISP.0.npz')
                    os.remove(outdir+'/'+finPR+'_1_DISP.1.npz')
                    os.remove(outdir+'/'+finPR+'_2_DISP.0.npz')
                    os.remove(outdir+'/'+finPR+'_2_DISP.1.npz')
                arr1_1              = f10['arr_0']
                nfout1_1            = f10['arr_1']
                arr2_1              = f11['arr_0']
                nfout2_1            = f11['arr_1']
                arr1_2              = f20['arr_0']
                nfout1_2            = f20['arr_1']
                arr2_2              = f21['arr_0']
                nfout2_2            = f21['arr_1']
                staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                if basic1:
                    parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': nfout1_1}
                    self.add_auxiliary_data(data=arr1_1, data_type='DISPbasic1', path=staid_aux, parameters=parameters)
                if basic2:
                    parameters      = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': nfout2_1}
                    self.add_auxiliary_data(data=arr2_1, data_type='DISPbasic2', path=staid_aux, parameters=parameters)
                if inftan.pmf:
                    if pmf1:
                        parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': nfout1_2}
                        self.add_auxiliary_data(data=arr1_2, data_type='DISPpmf1', path=staid_aux, parameters=parameters)
                    if pmf2:
                        parameters  = {'Tc': 0, 'To': 1, 'U': 2, 'C': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'snr':8, 'Np': nfout2_2}
                        self.add_auxiliary_data(data=arr2_2, data_type='DISPpmf2', path=staid_aux, parameters=parameters)
        if deletedisp: shutil.rmtree(outdir+'/'+pfx)
        return
    
    def interp_disp(self, data_type='DISPpmf2', channel='ZZ', pers=np.array([]), verbose=False):
        """ Interpolate dispersion curve for a given period array.
        =======================================================================================================
        ::: input parameters :::
        data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        pers        - period array
        
        ::: output :::
        self.auxiliary_data.DISPbasic1interp, self.auxiliary_data.DISPbasic2interp,
        self.auxiliary_data.DISPpmf1interp, self.auxiliary_data.DISPpmf2interp
        =======================================================================================================
        """
        if data_type=='DISPpmf2':
            ntype   = 6
        else:
            ntype   = 5
        if pers.size==0:
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        staLst                      = self.waveforms.list()
        Nsta                        = len(staLst)
        Ntotal_traces               = Nsta*(Nsta-1)/2
        iinterp                     = 0
        Ntr_one_percent             = int(Ntotal_traces/100.)
        ipercent                    = 0
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                iinterp             += 1
                if np.fmod(iinterp, Ntr_one_percent) ==0:
                    ipercent        += 1
                    print ('*** Number of traces finished interpolating dispersion curve: '+\
                                    str(iinterp)+'/'+str(Ntotal_traces)+' '+str(ipercent)+'%')
                try:
                    subdset         = self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except KeyError:
                    continue
                data                = subdset.data.value
                index               = subdset.parameters
                if verbose:
                    print 'interpolating dispersion curve for '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                outindex            = { 'To': 0, 'U': 1, 'C': 2,  'amp': 3, 'snr': 4, 'inbound': 5, 'Np': pers.size }
                Np                  = int(index['Np'])
                if Np < 5:
                    if verbose:
                        warnings.warn('Not enough datapoints for: '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel, UserWarning, stacklevel=1)
                    continue
                # interpolation
                obsT                = data[index['To']][:Np]
                U                   = np.interp(pers, obsT, data[index['U']][:Np] )
                C                   = np.interp(pers, obsT, data[index['C']][:Np] )
                amp                 = np.interp(pers, obsT, data[index['amp']][:Np] )
                inbound             = (pers > obsT[0])*(pers < obsT[-1])*1
                # store interpolated data to interpdata array
                interpdata          = np.append(pers, U)
                interpdata          = np.append(interpdata, C)
                interpdata          = np.append(interpdata, amp)
                if data_type=='DISPpmf2':
                    snr             = np.interp(pers, obsT, data[index['snr']][:Np] )
                    interpdata      = np.append(interpdata, snr)
                interpdata          = np.append(interpdata, inbound)
                interpdata          = interpdata.reshape(ntype, pers.size)
                staid_aux           = netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                self.add_auxiliary_data(data=interpdata, data_type=data_type+'interp', path=staid_aux, parameters=outindex)
        return
    
    def xcorr_raytomoinput(self, outdir, staxml=None, netcodelst=[], lambda_factor=3., snr_thresh=15., channel='ZZ',\
                           pers=np.array([]), outpfx='raytomo_in_', data_type='DISPpmf2interp', verbose=True):
        """
        Generate input files for Barmine's straight ray surface wave tomography code.
        =======================================================================================================
        ::: input parameters :::
        outdir          - output directory
        staxml          - input StationXML for data selection
        netcodelst      - network list for data selection
        lambda_factor   - wavelength factor for data selection (default = 3.)
        snr_thresh      - threshold SNR (default = 15.)
        channel         - channel for tomography
        pers            - period array
        outpfx          - prefix for output files, default is 'raytomo_in_'
        data_type       - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        -------------------------------------------------------------------------------------------------------
        Output format:
        outdir/outpfx+per_channel_ph.lst
        =======================================================================================================
        """
        print ('=== Generating straight ray tomography input files')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if pers.size==0:
            pers        = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        # open output files
        fph_lst         = []
        fgr_lst         = []
        for per in pers:
            fname_ph    = outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_ph.lst' %( per )
            fname_gr    = outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_gr.lst' %( per )
            fph         = open(fname_ph, 'w')
            fgr         = open(fname_gr, 'w')
            fph_lst.append(fph)
            fgr_lst.append(fgr)
        #--------------------------------------------------------------------------
        # added the functionality of using stations from an input StationXML file
        #--------------------------------------------------------------------------
        if staxml != None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            staLst          = waveformLst
            print '--- Load stations from input StationXML file'
        else:
            print '--- Load all the stations from database'
            staLst          = self.waveforms.list()
        # network selection
        if len(netcodelst) != 0:
            staLst_ALL      = copy.deepcopy(staLst)
            staLst          = []
            for staid in staLst_ALL:
                netcode, stacode    = staid.split('.')
                if not (netcode in netcodelst):
                    continue
                staLst.append(staid)
            print '--- Select stations according to network code: '+str(len(staLst))+'/'+str(len(staLst_ALL))+' (selected/all)'
        # Loop over stations
        Nsta            = len(staLst)
        Ntotal_traces   = Nsta*(Nsta-1)/2
        Ntr_one_percent = int(Ntotal_traces/100.)
        ipercent        = 0
        iray            = -1
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                # print how many rays has been processed 
                iray                += 1
                if np.fmod(iray+1, Ntr_one_percent) ==0:
                    ipercent        += 1
                    print ('*** Number of traces finished generating raytomo input: '+str(iray)+'/'+str(Ntotal_traces)+' '+str(ipercent)+'%')
                # get data
                try:
                    subdset         = self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except:
                    continue
                lat1, elv1, lon1    = self.waveforms[staid1].coordinates.values()
                lat2, elv2, lon2    = self.waveforms[staid2].coordinates.values()
                dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist                = dist/1000.
                if lon1<0:
                    lon1            +=360.
                if lon2<0:
                    lon2            +=360.
                data                = subdset.data.value
                index               = subdset.parameters
                for iper in range(pers.size):
                    per             = pers[iper]
                    # wavelength criteria
                    if dist < lambda_factor*per*3.5:
                        continue
                    # if dist > 1200.:
                    #     continue 
                    ind_per         = np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise AttributeError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    pvel            = data[index['C']][ind_per]
                    gvel            = data[index['U']][ind_per]
                    snr             = data[index['snr']][ind_per]
                    inbound         = data[index['inbound']][ind_per]
                    # quality control
                    if inbound != 1.:
                        continue
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10:
                        continue
                    if snr < snr_thresh: # SNR larger than 15.
                        continue
                    fph             = fph_lst[iper]
                    fgr             = fgr_lst[iper]
                    fph.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(iray, lat1, lon1, lat2, lon2, pvel, staid1, staid2))
                    fgr.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(iray, lat1, lon1, lat2, lon2, gvel, staid1, staid2))
        for iper in range(pers.size):
            fph                     = fph_lst[iper]
            fgr                     = fgr_lst[iper]
            fph.close()
            fgr.close()
        print ('=== end of generating straight ray tomography input files')
        return
    
    def xcorr_raytomoinput_debug(self, outdir, exclude_stalst=None, staxml=None, netcodelst=[], stacodelst=[],  lambda_factor=3.,\
                snr_thresh=15., channel='ZZ', pers=np.array([]), outpfx='raytomo_in_', data_type='DISPpmf2interp', verbose=True):
        """
        Generate input files for Barmine's straight ray surface wave tomography code.
        =======================================================================================================
        ::: input parameters :::
        outdir          - output directory
        staxml          - input StationXML for data selection
        netcodelst      - network list for data selection
        lambda_factor   - wavelength factor for data selection (default = 3.)
        snr_thresh      - threshold SNR (default = 15.)
        channel         - channel for tomography
        pers            - period array
        outpfx          - prefix for output files, default is 'raytomo_in_'
        data_type       - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        -------------------------------------------------------------------------------------------------------
        Output format:
        outdir/outpfx+per_channel_ph.lst
        =======================================================================================================
        """
        print ('=== Generating straight ray tomography input files')
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if pers.size==0:
            pers        = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        # open output files
        fph_lst         = []
        fgr_lst         = []
        for per in pers:
            fname_ph    = outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_ph.lst' %( per )
            fname_gr    = outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_gr.lst' %( per )
            fph         = open(fname_ph, 'w')
            fgr         = open(fname_gr, 'w')
            fph_lst.append(fph)
            fgr_lst.append(fgr)
        #--------------------------------------------------------------------------
        # added the functionality of using stations from an input StationXML file
        #--------------------------------------------------------------------------
        if staxml != None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            staLst          = waveformLst
            print '--- Load stations from input StationXML file'
        else:
            print '--- Load all the stations from database'
            staLst          = self.waveforms.list()
        #
        ex_stalst           = []
        if exclude_stalst != None:
            with open(exclude_stalst, 'rb') as fio:
                for line in fio.readlines():
                    stacode = line.split()[0]
                    netcode = line.split()[-1]
                    exlon   = float(line.split()[1])
                    exlat   = float(line.split()[2])
                    dist, az, baz       = obspy.geodetics.gps2dist_azimuth(exlat, exlon, 64.5, -149.5) # distance is in m
                    if dist/1000. > 100.:
                        continue
                    # if not (abs(exlon - (-150)) < 1. and abs(exlat - 64.) < 1. ):
                        # continue
                    # print netcode+'.'+stacode
                    ex_stalst.append(netcode+'.'+stacode)
        # network selection
        if len(netcodelst) != 0 or len(stacodelst) != 0 or len(ex_stalst) != 0:
            staLst_ALL      = copy.deepcopy(staLst)
            staLst          = []
            for staid in staLst_ALL:
                netcode, stacode    = staid.split('.')
                if len(netcodelst) != 0:
                    if not (netcode in netcodelst):
                        continue
                if len(ex_stalst) != 0:
                    if staid in ex_stalst:
                        print staid
                        continue
                # print staid
                # if netcode == 'TA':
                #     continue
                if len(stacodelst) != 0:
                    if stacode in stacodelst:
                        continue
                staLst.append(staid)
            print '--- Select stations according to network code: '+str(len(staLst))+'/'+str(len(staLst_ALL))+' (selected/all)'
        # Loop over stations
        # return
        Nsta            = len(staLst)
        Ntotal_traces   = Nsta*(Nsta-1)/2
        Ntr_one_percent = int(Ntotal_traces/100.)
        ipercent        = 0
        iray            = -1
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                # print how many rays has been processed 
                iray                += 1
                if np.fmod(iray+1, Ntr_one_percent) ==0:
                    ipercent        += 1
                    print ('*** Number of traces finished generating raytomo input: '+str(iray)+'/'+str(Ntotal_traces)+' '+str(ipercent)+'%')
                # get data
                try:
                    subdset         = self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except:
                    continue
                lat1, elv1, lon1    = self.waveforms[staid1].coordinates.values()
                lat2, elv2, lon2    = self.waveforms[staid2].coordinates.values()
                dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist                = dist/1000.
                if lon1<0:
                    lon1            +=360.
                if lon2<0:
                    lon2            +=360.
                data                = subdset.data.value
                index               = subdset.parameters
                for iper in range(pers.size):
                    per             = pers[iper]
                    # wavelength criteria
                    if dist < lambda_factor*per*3.5:
                        continue
                    # if dist > 1200.:
                    #     continue 
                    ind_per         = np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise AttributeError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    pvel            = data[index['C']][ind_per]
                    gvel            = data[index['U']][ind_per]
                    snr             = data[index['snr']][ind_per]
                    inbound         = data[index['inbound']][ind_per]
                    # quality control
                    if inbound != 1.:
                        continue
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10:
                        continue
                    if snr < snr_thresh: # SNR larger than 15.
                        continue
                    fph             = fph_lst[iper]
                    fgr             = fgr_lst[iper]
                    fph.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(iray, lat1, lon1, lat2, lon2, pvel, staid1, staid2))
                    fgr.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(iray, lat1, lon1, lat2, lon2, gvel, staid1, staid2))
        for iper in range(pers.size):
            fph                     = fph_lst[iper]
            fgr                     = fgr_lst[iper]
            fph.close()
            fgr.close()
        print ('=== end of generating straight ray tomography input files')
        return
    
    def xcorr_get_field(self, outdir=None, staxml=None, netcodelst=[], lambda_factor=3., snr_thresh=15., channel='ZZ',\
                        pers=np.array([]), data_type='DISPpmf2interp', verbose=True):
        """ Get the field data for eikonal tomography
        ============================================================================================================================
        ::: input parameters :::
        outdir          - directory for txt output (default is not to generate txt output)
        staxml          - input StationXML for data selection
        netcodelst      - network list for data selection
        lambda_factor   - wavelength factor for data selection (default = 3.)
        snr_thresh      - threshold SNR (default = 15.)
        channel         - channel name
        pers            - period array
        datatype        - dispersion data type (default = DISPpmf2interp, interpolated pmf aftan results after jump detection)
        ::: output :::
        self.auxiliary_data.FieldDISPpmf2interp
        ============================================================================================================================
        """
        print ('=== generating arrays for eikonal tomography')
        if pers.size==0:
            pers        = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        outindex        = { 'longitude': 0, 'latitude': 1, 'C': 2,  'U':3, 'snr': 4, 'dist': 5 }
        #--------------------------------------------------------------------------
        # added the functionality of using stations from an input StationXML file
        #--------------------------------------------------------------------------
        if staxml != None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            staLst          = waveformLst
            print '--- Load stations from input StationXML file'
        else:
            print '--- Load all the stations from database'
            staLst          = self.waveforms.list()
        # network selection
        if len(netcodelst) != 0:
            staLst_ALL      = copy.deepcopy(staLst)
            staLst          = []
            for staid in staLst_ALL:
                netcode, stacode    = staid.split('.')
                if not (netcode in netcodelst):
                    continue
                staLst.append(staid)
            print '--- Select stations according to network code: '+str(len(staLst))+'/'+str(len(staLst_ALL))+' (selected/all)'
        # Loop over stations
        for staid1 in staLst:
            field_lst   = []
            Nfplst      = []
            for per in pers:
                field_lst.append(np.array([]))
                Nfplst.append(0)
            try:
                lat1, elv1, lon1    = self.waveforms[staid1].coordinates.values()
            except:
                print 'WARNING: No station:' +staid1+' in the database'
                continue
            Ndata       = 0
            for staid2 in staLst:
                if staid1 == staid2:
                    continue
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                try:
                    subdset         = self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except:
                    try:
                        subdset     = self.auxiliary_data[data_type][netcode2][stacode2][netcode1][stacode1][channel]
                    except:
                        continue
                Ndata               +=1
                lat2, elv2, lon2    = self.waveforms[staid2].coordinates.values()
                dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist                = dist/1000.
                if lon1<0:
                    lon1    += 360.
                if lon2<0:
                    lon2    += 360.
                data                = subdset.data.value
                index               = subdset.parameters
                # loop over periods
                for iper in range(pers.size):
                    per     = pers[iper]
                    # three wavelength, note that in eikonal_operator, another similar criteria will be applied
                    if dist < lambda_factor*per*3.5:
                        continue
                    ind_per = np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise KeyError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    pvel    = data[index['C']][ind_per]
                    gvel    = data[index['U']][ind_per]
                    # pvel    = data[index['Vph']][ind_per]
                    # gvel    = data[index['Vgr']][ind_per]
                    snr     = data[index['snr']][ind_per]
                    inbound = data[index['inbound']][ind_per]
                    # quality control
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10:
                        continue
                    if inbound!=1.:
                        continue
                    if snr < snr_thresh:
                        continue
                    field_lst[iper] = np.append(field_lst[iper], lon2)
                    field_lst[iper] = np.append(field_lst[iper], lat2)
                    field_lst[iper] = np.append(field_lst[iper], pvel)
                    field_lst[iper] = np.append(field_lst[iper], gvel)
                    field_lst[iper] = np.append(field_lst[iper], snr)
                    field_lst[iper] = np.append(field_lst[iper], dist)
                    Nfplst[iper]    += 1
            if verbose:
                print 'Getting field data for: '+staid1+', '+str(Ndata)+' paths'
            # end of reading data from all receivers, taking staid1 as virtual source
            if outdir is not None:
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
            # save data
            staid_aux   = netcode1+'/'+stacode1+'/'+channel
            for iper in range(pers.size):
                per                 = pers[iper]
                del_per             = per-int(per)
                if field_lst[iper].size==0:
                    continue
                field_lst[iper]     = field_lst[iper].reshape(Nfplst[iper], 6)
                if del_per == 0.:
                    staid_aux_per   = staid_aux+'/'+str(int(per))+'sec'
                else:
                    dper            = str(del_per)
                    staid_aux_per   = staid_aux+'/'+str(int(per))+'sec'+dper.split('.')[1]
                self.add_auxiliary_data(data=field_lst[iper], data_type='Field'+data_type,\
                                        path=staid_aux_per, parameters=outindex)
                if outdir is not None:
                    if not os.path.isdir(outdir+'/'+str(per)+'sec'):
                        os.makedirs(outdir+'/'+str(per)+'sec')
                    txtfname        = outdir+'/'+str(per)+'sec'+'/'+staid1+'_'+str(per)+'.txt'
                    header          = 'evlo='+str(lon1)+' evla='+str(lat1)
                    np.savetxt( txtfname, field_lst[iper], fmt='%g', header=header )
        print ('=== end generating arrays for eikonal tomography')
        return
    
    def plot_waveforms(self, staxml=None, channel='ZZ', ndata=100, dmin=1., dmax=1000.):
        if staxml != None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            staLst          = waveformLst
            print '--- Load stations from input StationXML file'
        else:
            print '--- Load all the stations from database'
            staLst          = self.waveforms.list()
        interval        = (dmax-dmin)/(ndata-1)
        dist_arr        = np.arange(ndata)*interval
        is_data         = np.zeros(ndata, dtype=bool)
        idata           = 0
        ax  = plt.subplot()
        for staid1 in staLst:
            netcode1, stacode1  = staid1.split('.')
            # lon1                = self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
            # lat1                = self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
            if idata >= ndata:
                break
            
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                # lon2                = self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                # lat2                = self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                if staid1 >= staid2:
                    continue
                # # # dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                # # # dist                = dist/1000.
                # # # if dist < dist_arr[0] or dist > dist_arr[-1]:
                # # #     continue
                # # # index               = (abs(dist_arr - dist)).argmin()
                # # # if is_data[index]:
                # # #     continue
                # determine channels
                try:
                    channels1       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    for chan in channels1:
                        if chan[-1] == channel[0]:
                            chan1   = chan
                            break
                    channels2       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1].list()
                    for chan in channels2:
                        if chan[-1] == channel[1]:
                            chan2   = chan
                            break
                except KeyError:
                    continue
                # get data
                try:
                    tr              = self.get_xcorr_trace(netcode1, stacode1, netcode2, stacode2, chan1, chan2)
                except NameError:
                    print netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+' not exists!'
                    continue
                # is_data[index]      = True
                idata               += 1
                print idata
                time    = tr.stats.sac.b + np.arange(tr.stats.npts)*tr.stats.delta
                plt.plot(time, tr.data/abs(tr.data.max())*10. + tr.stats.distance/1000., 'k-', lw= 0.1)
                if idata >= ndata:
                    break
        plt.xlim([-1000., 1000.])
        plt.ylim([-1., 1500.])
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        plt.ylabel('Distance (km)', fontsize=40)
        plt.xlabel('Time (sec)', fontsize=40)
        plt.show()
        
    def plot_waveforms_monthly(self, datadir, monthdir, staxml=None, chan1='LHZ', chan2='LHZ'):
        if staxml != None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            staLst          = waveformLst
            print '--- Load stations from input StationXML file'
        else:
            print '--- Load all the stations from database'
            staLst          = self.waveforms.list()

        ax              = plt.subplot()
        for staid1 in staLst:
            netcode1, stacode1  = staid1.split('.')
            try:
                lon1    = self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
                lat1    = self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
            except:
                continue
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                try:
                    lon2                = self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                    lat2                = self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                except:
                    continue
                if staid1 >= staid2:
                    continue
                dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist                = dist/1000.
                
                infname             = datadir+'/'+monthdir+'/COR/'+staid1+'/COR_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'.SAC'
                # print infname
                if not os.path.isfile(infname):
                    continue
                # get data
                tr                  = obspy.read(infname)[0]
                # is_data[index]      = True
                # idata               += 1
                # print idata
                time    = tr.stats.sac.b + np.arange(tr.stats.npts)*tr.stats.delta
                plt.plot(time, tr.data/abs(tr.data.max())*10. + dist, 'k-', lw= 0.1)

        plt.xlim([-1000., 1000.])
        plt.ylim([-1., 1000.])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.ylabel('Distance (km)', fontsize=30)
        plt.xlabel('Time (s)', fontsize=30)
        plt.title(monthdir, fontsize=40)
        plt.show()
        
    def compute_xcorr_coef(self, datadir, monthdir, staxml=None, chan1='LHZ', chan2='LHZ'):
        if staxml != None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            staLst          = waveformLst
            print '--- Load stations from input StationXML file'
        else:
            print '--- Load all the stations from database'
            staLst          = self.waveforms.list()

        ax              = plt.subplot()
        for staid1 in staLst:
            netcode1, stacode1  = staid1.split('.')
            try:
                lon1    = self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
                lat1    = self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
            except:
                continue
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                try:
                    lon2                = self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                    lat2                = self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                except:
                    continue
                if staid1 >= staid2:
                    continue
                dist, az, baz       = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist                = dist/1000.
                
                infname             = datadir+'/'+monthdir+'/COR/'+staid1+'/COR_'+staid1+'_'+chan1+'_'+staid2+'_'+chan2+'.SAC'
                # print infname
                if not os.path.isfile(infname):
                    continue
                # get data
                tr                  = obspy.read(infname)[0]
                # is_data[index]      = True
                # idata               += 1
                # print idata
                time    = tr.stats.sac.b + np.arange(tr.stats.npts)*tr.stats.delta
                plt.plot(time, tr.data/abs(tr.data.max())*10. + dist, 'k-', lw= 0.1)

        plt.xlim([-1000., 1000.])
        plt.ylim([-1., 1000.])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.ylabel('Distance (km)', fontsize=30)
        plt.xlabel('Time (s)', fontsize=30)
        plt.title(monthdir, fontsize=40)
        plt.show()
    
    
    # def plot_travel_time(self, netcode, stacode, period, channel='ZZ'):
    #     try:
    #         data    = self.auxiliary_data['FieldDISPpmf2interp'][netcode][stacode][channel][str(int(period)+'sec')].data.value
    #     except KeyError:
    #         print 'No data!'
    #         return
    #     lons        = data[:, 0]
    #     lats        = data[:, 1]
    #     C           = data[:, 2]
    #     dist        = data[:, 5]
    #     T           = dist/C
    #     field2d     = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
    #                                 minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype=fieldtype,\
    #                                     nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc)
    #             Zarr            = dataArr[:, fdict[fieldtype]]
    #             # added on 03/06/2018
    #             if Zarr.size <= mindp:
    #                 continue
    #             # added on 10/08/2018
    #             inlons          = dataArr[:, 0]
    #             inlats          = dataArr[:, 1]
    #             if not _check_station_distribution(inlons, inlats, np.int32(mindp/2.)):
    #                 continue
    #             distArr         = dataArr[:, 6] # Note amplitude is added!!!
    #             field2d.read_array(lonArr = inlons, latArr = inlats, ZarrIn = distArr/Zarr )
        
            
def stack4mp(invpair, datadir, outdir, ylst, mlst, pfx, fnametype):
    stackedST       = []
    init_stack_flag = False
    # station 1
    netcode1        = invpair[0].networks[0].code
    stacode1        = invpair[0].networks[0].stations[0].code
    channels1       = invpair[0].networks[0].stations[0].channels
    staid1          = netcode1+'.'+stacode1
    # station 2
    netcode2        = invpair[1].networks[0].code
    stacode2        = invpair[1].networks[0].stations[0].code
    channels2       = invpair[1].networks[0].stations[0].channels
    staid2          = netcode2+'.'+stacode2
    mnumb           = mlst.size
    for im in xrange(mnumb):
        skip_this_month = False
        cST     = []
        for chan1 in channels1:
            if skip_this_month:
                break
            for chan2 in channels2:
                month       = monthdict[mlst[im]]
                yrmonth     = str(ylst[im])+'.'+month
                if fnametype    == 1:
                    fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+chan1.code+'_'+staid2+'_'+chan2.code+'.SAC'
                elif fnametype  == 2:
                    fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                ############
                elif fnametype  == 3:
                    fname   = datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'+stacode2+'_'+chan2.code+'.SAC'
                ############
                if not os.path.isfile(fname):
                    skip_this_month = True
                    break
                try:
                    # # # tr              = obspy.core.read(fname)[0]
                    tr              = obspy.io.sac.SACTrace.read(fname)
                except TypeError:
                    warnings.warn('Unable to read SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                    skip_this_month = True
                if (np.isnan(tr.data)).any() or abs(tr.data.max())>1e20:
                    warnings.warn('NaN monthly SAC for: ' + staid1 +'_'+staid2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                    skip_this_month = True
                    break
                cST.append(tr)
        if len(cST) != (len(channels1)*len(channels2)) or skip_this_month:
            continue
        if init_stack_flag:
            for itr in xrange(len(cST)):
                mtr                             = cST[itr]
                stackedST[itr].data             += mtr.data
                # # # stackedST[itr].stats.sac.user0  += mtr.stats.sac.user0
                stackedST[itr].user0            += mtr.user0
        else:
            stackedST       = copy.deepcopy(cST)
            init_stack_flag = True
    if len(stackedST)== (len(channels1)*len(channels2)) :
        # print 'Finished Stacking for:'+staid1+'_'+staid2
        itrace              = 0
        for chan1 in channels1:
            for chan2 in channels2:
                stackedTr   = stackedST[itrace]
                outfname    = outdir+'/'+pfx+'/'+staid1+'/'+ \
                            pfx+'_'+staid1+'_'+chan1.code+'_'+staid2+'_'+chan2.code+'.SAC'
                if 'kcmpnm' in stackedTr.stats.sac:
                    if stackedTr.stats.sac.kcmpnm != chan1.code + chan2.code:
                        raise ValueError('Inconsistent channels: '+ stackedTr.stats.sac.kcmpnm+' '+\
                                    chan1.code+' '+ chan2.code)
                stackedTr.write(outfname, format='SAC')
                itrace      += 1
    return

def aftan4mp(aTr, outdir, inftan, prephdir, f77, pfx):
    # print 'aftan analysis for: '+ aTr.stats.sac.kuser0+'.'+aTr.stats.sac.kevnm+'_'+chan1+'_'+aTr.stats.network+'.'+aTr.stats.station+'_'+chan2
    if prephdir !=None:
        phvelname   = prephdir + "/%s.%s.pre" %(aTr.stats.sac.kuser0+'.'+aTr.stats.sac.kevnm, aTr.stats.network+'.'+aTr.stats.station)
    else:
        phvelname   = ''
    if abs(aTr.stats.sac.b+aTr.stats.sac.e)< aTr.stats.delta:
        aTr.makesym()
    if f77:
        aTr.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
            tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
    else:
        aTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
            tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
    aTr.get_snr(ffact=inftan.ffact) # SNR analysis
    chan1           = aTr.stats.sac.kcmpnm[:3]
    chan2           = aTr.stats.sac.kcmpnm[3:]
    foutPR          = outdir+'/'+pfx+'/'+aTr.stats.sac.kuser0+'.'+aTr.stats.sac.kevnm+'/'+ \
                        pfx+'_'+aTr.stats.sac.kuser0+'.'+aTr.stats.sac.kevnm+'_'+chan1+'_'+aTr.stats.network+'.'+aTr.stats.station+'_'+chan2+'.SAC'
    aTr.ftanparam.writeDISPbinary(foutPR)
    return
    
    
    


