# -*- coding: utf-8 -*-
"""
internal functions and classes for cross-correlations

:Copyright:
    Author: Lili Feng
    Research Geophysicist
    CGG
    email: lfeng1011@gmail.com
"""
import numpy as np
from numba import jit, float32, int32, boolean, float64, int64
import numba
import pyfftw
import obspy
import os
import multiprocessing
import obspy

# ------------- xcorr specific exceptions ---------------------------------------
class xcorrError(Exception):
    pass

class xcorrIOError(xcorrError, IOError):
    pass

class xcorrHeaderError(xcorrError):
    """
    Raised if header has issues.
    """
    pass

class xcorrDataError(xcorrError):
    """
    Raised if header has issues.
    """
    pass

xcorr_header_default    = {'netcode1': '', 'stacode1': '', 'netcode2': '', 'stacode2': '', 'chan1': '', 'chan2': '',
        'npts': 12345, 'b': 12345, 'e': 12345, 'delta': 12345, 'dist': 12345, 'az': 12345, 'baz': 12345, 'stackday': 0}
xcorr_sacheader_default = {'knetwk': '', 'kstnm': '', 'kcmpnm': '', 'stla': 12345, 'stlo': 12345, 
            'kuser0': '', 'kevnm': '', 'evla': 12345, 'evlo': 12345, 'evdp': 0., 'dist': 0., 'az': 12345, 'baz': 12345, 
                'delta': 12345, 'npts': 12345, 'user0': 0, 'b': 12345, 'e': 12345}

#===============================================
# functions for pre-processing
#===============================================
def _tshift_fft(data, dt, tshift):
    """positive means delaying the waveform
    """
    npts    = data.size
    Np2     = int(max(1<<(npts-1).bit_length(), 2**12))
    Xf      = np.fft.rfft(data, n=Np2)
    freq    = 1./dt/Np2*np.arange((Np2/2+1), dtype = float)
    ph_shift= np.exp(-2j*np.pi*freq*tshift)
    Xf2     = Xf*ph_shift
    return np.real(np.fft.irfft(Xf2)[:npts])

@jit(numba.types.Tuple((int64[:, :], int64))(boolean[:]), nopython=True)
def _rec_lst(mask):
    """Get rec list
    """
    reclst  = -np.ones((mask.size, 2), dtype = np.int64)
    isrec   = False
    irec    = 0
    for i in range(mask.size):
        if mask[i]:
            if isrec:
                reclst[irec, 1] = i-1
                irec            += 1
                isrec           = False
            else:
                continue
        else:
            if isrec:
                # last element
                if i == (mask.size - 1):
                    reclst[irec, 1] = i
                    irec            += 1
                continue
            else:
                isrec           = True
                reclst[irec, 0] = i
    return reclst, irec

@jit(numba.types.Tuple((int64[:, :], int64))(boolean[:]), nopython=True)
def _gap_lst(mask):
    """Get gap list
    """
    gaplst  = -np.ones((mask.size, 2), dtype = np.int64)
    isgap   = False
    igap    = 0
    for i in range(mask.size):
        if mask[i]:
            if isgap:
                # last element
                if i == (mask.size - 1):
                    gaplst[igap, 1] = i
                    igap            += 1
                continue
            else:
                isgap           = True
                gaplst[igap, 0] = i
        else:
            if isgap:
                gaplst[igap, 1] = i-1
                igap            += 1
                isgap           = False
            else:
                continue
    return gaplst, igap
# 
@jit(float64[:](int64[:, :], int64[:, :], float64[:], int64, int64), nopython=True)
def _fill_gap_vals(gaplst, reclst, data, Ngap, halfw):
    """Get the values for gap fill
    """
    alpha   = -0.5 / (halfw * halfw)
    gaparr  = np.zeros(data.size, dtype = np.float64)
    if gaplst[0, 0] < reclst[0, 0]:
        gaphead = True
    else:
        gaphead = False
    if gaplst[-1, 1] > reclst[-1, 1]:
        gaptail = True
    else:
        gaptail = False
    for igap in range(Ngap):
        igp0    = gaplst[igap, 0]
        igp1    = gaplst[igap, 1]
        tmp_npts= igp1 - igp0 + 1
        if gaphead and igap == 0:
            ilrec   = 0
            irrec   = 0
        elif gaptail and igap == (Ngap - 1):
            ilrec   = -1
            irrec   = -1
        elif gaphead:
            ilrec   = igap - 1
            irrec   = igap
        else:
            ilrec   = igap
            irrec   = igap + 1
        il0     = reclst[ilrec, 0]
        il1     = reclst[ilrec, 1]
        ir0     = reclst[irrec, 0]
        ir1     = reclst[irrec, 1]
        if (il1 - il0 + 1 ) < halfw:
            lmean   = data[il0:(il1+1)].mean()
            lstd    = data[il0:(il1+1)].std()
        else:
            lmean   = data[(il1-halfw+1):(il1+1)].mean()
            lstd    = data[(il1-halfw+1):(il1+1)].std()
        if (ir1 - ir0 + 1 ) < halfw:
            rmean   = data[ir0:(ir1+1)].mean()
            rstd    = data[ir0:(ir1+1)].std()
        else:
            rmean   = data[ir0:(ir0+halfw)].mean()
            rstd    = data[ir0:(ir0+halfw)].std()
        if gaphead and igap == 0:
            lstd= 0
        elif gaptail and igap == (Ngap - 1):
            rstd= 0
        if tmp_npts == 1:
            gaparr[igp0]    = (lmean+rmean)/2. + np.random.uniform(-(lstd+rstd)/2, (lstd+rstd)/2)
        else:
            imid    = int(np.floor(tmp_npts/2))
            for i in range(tmp_npts):
                j           = i + igp0
                slope       = (rmean - lmean)/tmp_npts * i + lmean
                if i < imid:
                    gsamp   = np.exp(alpha * i * i)
                    tmpstd  = lstd
                else:
                    gsamp   = np.exp(alpha * (tmp_npts - i - 1) * (tmp_npts - i - 1))
                    tmpstd  = rstd
                gaparr[j]   = gsamp * np.random.uniform(-tmpstd, tmpstd) + slope
    return gaparr

#===============================================
# functions for cross-correlations
#===============================================
@jit(int32[:](int32[:, :], int32[:, :], int32), nopython=True)
def _CalcRecCor(arr1, arr2, lagN):
    """compute the amplitude weight for the xcorr, used for amplitude correction
    ==============================================================================
    ::: input parameters :::
    arr1, arr2  - the input arrays from ft_*SAC_rec,
                    indicating holes in the original records
    lagN        - one-sided npts for xcorr
    ::: output :::
    cor_rec     - amplitude weight for the xcorr
    ==============================================================================
    """
    N1          = arr1.shape[0]
    N2          = arr2.shape[0]
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # arr1[:,1] +=1, arr2[:,1]+=1
    # changed for the new code
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    arr1[:,1]   += 1
    arr2[:,1]   += 1
    # array indicating number of data points for xcorr, used for amplitude correction
    cor_rec = np.zeros(int(2*lagN + 1), dtype=np.int32)
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
    cor_rec[lagN]   /= 2
    return cor_rec

@jit(boolean(int32, float32, float32[:], float32), nopython=True)
def _CheckPrecNoise(lagN, dt, data, dist):
    """check precursory noise
    """
    # trailing noise window
    Ndis    = np.int32(np.floor((dist/0.8+50.)/dt+0.5))
    if Ndis > lagN:
        return False
    Nb      = max(np.int32(lagN*4/5), Ndis)
    noiseSum= np.float32(0.)
    noiseN  = lagN - Nb + 1
    for i in range(noiseN):
        noiseSum    += (data[i + lagN + Nb])**2
        noiseSum    += (data[lagN - i - Nb])**2
    noiseSum    = np.sqrt(noiseSum/(2*(noiseN)))
    # precursory noise window
    Ndis    = np.int32(np.floor((dist/4.5-50.)/dt+0.5))
    if Ndis < 10./dt:
        return False
    Ne      = min(Ndis, np.int32(100./dt))
    precSum = np.float32(0.)
    precN   = 2*Ne + 1
    for i in range(precN):
        precSum += (data[lagN - Ne + i])**2
    precSum    = np.sqrt(precSum/precN)
    if precSum > 5.*noiseSum:
        return True
    else:
        return False

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
        raise xcorrDataError('Lagtime npts overflow!')
    out_data            = np.zeros(2*lagN+1, dtype=float)
    out_data[lagN]      = seis_out[0]
    out_data[:lagN]     = (seis_out[1:lagN+1])[::-1]
    out_data[(lagN+1):] = (seis_out[Ns-lagN:])[::-1]
    return out_data

def _amp_ph_to_xcorr_planned(amp1, amp2, ph1, ph2, fftw_plan, sps = 1., lagtime = 3000.):
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
    """ A class for ambient noise cross-correlation computation
    =================================================================================================================
    ::: parameters :::
    stacode1, netcode1  - station/network code for station 1
    stacode2, netcode2  - station/network code for station 2
    monthdir            - month directory (e.g. 2019.JAN)
    daylst              - list includes the days for xcorr
    =================================================================================================================
    """
    def __init__(self, stacode1, netcode1, stla1, stlo1, stacode2, netcode2, stla2, stlo2, monthdir, daylst, year=None, month=None):
        self.stacode1   = stacode1
        self.netcode1   = netcode1
        self.stla1      = stla1
        self.stlo1      = stlo1
        self.stacode2   = stacode2
        self.netcode2   = netcode2
        self.stla2      = stla2
        self.stlo2      = stlo2
        self.monthdir   = monthdir
        self.daylst     = daylst
        self.year       = year
        self.month      = month
        return
    
    def print_info(self, process_id):
        """print the informations of this pair
        """
        staid1          = self.netcode1 + '.' + self.stacode1
        staid2          = self.netcode2 + '.' + self.stacode2
        print ('--- '+ staid1+'_'+staid2+' : '+self.monthdir+' '+str(len(self.daylst))+' days; processID = '+str(process_id))
    
    def _get_daylst(self):
        """get the day list if not specified
        """
        if len(self.daylst) == 0 and (self.year is not None) and (self.month is not None) :
            stime   = obspy.UTCDateTime(str(self.year)+'%02d01' %(self.month))
            while( (stime.year == self.year) and (stime.month == self.month)):
                self.daylst.append(stime.day)
                stime   += 86400
        return 
    
    def convert_amph_to_xcorr(self, datadir, chans=['LHZ', 'LHE', 'LHN'], ftlen = True,\
            tlen = 84000., mintlen = 20000., sps = 1., lagtime = 3000., CorOutflag = 0, \
            fprcs = False, fastfft=True, runtype = 0, verbose=False, verbose2=False, process_id=''):
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
        fprcs       - turn on/off (1/0) precursor signal checking
        fastfft     - speeding up the computation by using precomputed fftw_plan or not
        runtype     - type of runs
                        0   - first run, run the xcorr by creating new log files
                        1   - skip if log file indicates SUCCESS & SKIPPED & NODATA
                        2   - skip if log file indicates SUCCESS
                        3   - skip if log file exists
                        4   - skip if montly/staid1 log directory exists
                        5   - skip if monthly log directory exists
        =================================================================================================================
        """
        if verbose:
            self.print_info(process_id=process_id)
        staid1                  = self.netcode1 + '.' + self.stacode1
        staid2                  = self.netcode2 + '.' + self.stacode2
        month_dir               = datadir+'/'+self.monthdir
        monthly_xcorr           = []
        chan_size               = len(chans)
        init_common_header      = False
        xcorr_common_sacheader  = xcorr_sacheader_default.copy()
        lagN                    = np.floor(lagtime*sps +0.5) # npts for one-sided lag
        stacked_day             = 0
        outlogstr               = ''
        Nvalid_day              = 0
        Nnodata                 = 0
        init_fft_plan           = False
        self._get_daylst()
        #-----------------
        # loop over days
        #-----------------
        for day in self.daylst:
            # input streams
            st_amp1         = obspy.Stream()
            st_ph1          = obspy.Stream()
            st_amp2         = obspy.Stream()
            st_ph2          = obspy.Stream()
            # daily output streams
            daily_xcorr     = []
            daydir          = month_dir+'/'+self.monthdir+'.'+str(day)
            skip_this_day   = False
            data_not_exist  = False
            # read amp/ph files
            for chan in chans:
                pfx1    = daydir+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid1+'.'+chan+'.SAC'
                pfx2    = daydir+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid2+'.'+chan+'.SAC'
                if not ( os.path.isfile(pfx1+'.am') and os.path.isfile(pfx1+'.ph') and\
                        os.path.isfile(pfx2+'.am') and os.path.isfile(pfx2+'.ph') ):
                    skip_this_day   = True
                    data_not_exist  = True
                    Nnodata         += 1
                    break
                st_amp1 += obspy.read(pfx1+'.am')
                st_ph1  += obspy.read(pfx1+'.ph')
                st_amp2 += obspy.read(pfx2+'.am')
                st_ph2  += obspy.read(pfx2+'.ph')
            #---------------------------------------
            # construct fftw_plan for speeding up
            #---------------------------------------
            if fastfft and (not init_fft_plan) and (not data_not_exist):
                temp_pfx    = month_dir+'/'+self.monthdir+'.'+str(day)+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid1+'.'+chans[0]+'.SAC'
                amp_ref     = obspy.read(temp_pfx+'.am')[0]
                Nref        = amp_ref.data.size
                Ns          = int(2*Nref - 1)
                temp_x_sp   = np.zeros(Ns, dtype=complex)
                temp_out    = np.zeros(Ns, dtype=complex)
                fftw_plan   = pyfftw.FFTW(input_array=temp_x_sp, output_array=temp_out, direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ))
                init_fft_plan   = True
            #-----------------------------
            # define common sac header
            #-----------------------------
            if (not init_common_header) and (not skip_this_day):
                tr1                                 = st_amp1[0]
                tr2                                 = st_amp2[0]
                xcorr_common_sacheader['kuser0']    = self.netcode1
                xcorr_common_sacheader['kevnm']     = self.stacode1
                xcorr_common_sacheader['knetwk']    = self.netcode2
                xcorr_common_sacheader['kstnm']     = self.stacode2
                xcorr_common_sacheader['evla']      = self.stla1
                xcorr_common_sacheader['evlo']      = self.stlo1
                xcorr_common_sacheader['stla']      = self.stla2
                xcorr_common_sacheader['stlo']      = self.stlo2
                dist, az, baz                       = obspy.geodetics.gps2dist_azimuth(self.stla1, self.stlo1, self.stla2, self.stlo2) # distance is in m
                xcorr_common_sacheader['dist']      = dist/1000.
                xcorr_common_sacheader['az']        = az
                xcorr_common_sacheader['baz']       = baz
                xcorr_common_sacheader['delta']     = 1./sps
                xcorr_common_sacheader['npts']      = int(2*lagN + 1)
                xcorr_common_sacheader['b']         = -float(lagN/sps)
                xcorr_common_sacheader['e']         = float(lagN/sps)
                xcorr_common_sacheader['user0']     = 1
                init_common_header                  = True
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
                    if (np.isnan(amp1)).any() or (np.isnan(amp2)).any() or (np.isnan(ph1)).any() or (np.isnan(ph2)).any():
                        skip_this_day   = True
                        break
                    if np.any(amp1 > 1e20) or np.any(amp2 > 1e20):
                        skip_this_day   = True
                        break
                    #-----------------------------------
                    # get amplitude correction array
                    #-----------------------------------
                    Namp        = amp1.size
                    if ftlen:
                        # npts for the length of the preprocessed daily record 
                        Nrec    = int(tlen*sps)
                        frec1   = daydir+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid1+'.'+chans[ich1]+'.SAC_rec2'
                        frec2   = daydir+'/ft_'+self.monthdir+'.'+str(day)+'.'+staid2+'.'+chans[ich2]+'.SAC_rec2'
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
                        cor_rec     = _CalcRecCor(np.int32(arr1), np.int32(arr2), np.int32(lagN))
                        # skip the day if the length of available data is too small
                        if cor_rec[0] < mintlen*sps or cor_rec[-1] < mintlen*sps:
                            skip_this_day   = True
                            break
                        # skip the day if any data point has a weight of zero
                        if np.any(cor_rec == 0):
                            skip_this_day   = True
                            break
                    # comvert amp & ph files to xcorr
                    if init_fft_plan:
                        out_data    = _amp_ph_to_xcorr_planned(amp1=amp1, ph1=ph1, amp2=amp2, ph2=ph2, sps=sps,\
                                                                lagtime=lagtime, fftw_plan=fftw_plan)
                    else:
                        out_data    = _amp_ph_to_xcorr(amp1=amp1, ph1=ph1, amp2=amp2, ph2=ph2, sps=sps, lagtime=lagtime)
                    # amplitude correction
                    if ftlen:
                        out_data    /= np.float64(cor_rec)
                        out_data    *= Namp  ## why?
                        # out_data    *= float(2*Namp -1)  ## why?
                    # precursory noise checking
                    if fprcs:
                        skip_this_day= _CheckPrecNoise(np.int32(lagN), np.float32(1./sps), np.float32(out_data), np.float32(dist))
                        if skip_this_day:
                            print ('!!! WARNING: LARGE PRECURSORY SIGNAL. SKIPPED: '+\
                                   staid1+'_'+staid2+' : '+self.monthdir+'.'+str(day))
                            break
                    # end of computing individual xcorr
                    daily_xcorr.append(out_data)
            # end loop over channels
            if not skip_this_day:
                if verbose2:
                    print ('xcorr finished '+ staid1+'_'+staid2+' : '+self.monthdir+'.'+str(day))
                outlogstr   += '%02d    1\n' %day
                Nvalid_day  += 1
                # output daily xcorr
                if CorOutflag != 0:
                    out_daily_dir   = month_dir+'/COR_D/'+staid1
                    if not os.path.isdir(out_daily_dir):
                        os.makedirs(out_daily_dir)
                    for ich1 in range(chan_size):
                        for ich2 in range(chan_size):
                            i                       = chan_size*ich1 + ich2
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
                                i                   = chan_size*ich1 + ich2
                                monthly_xcorr.append(daily_xcorr[i])
                    # stacking
                    else:
                        for ich1 in range(chan_size):
                            for ich2 in range(chan_size):
                                i                   = chan_size*ich1 + ich2
                                monthly_xcorr[i]    += daily_xcorr[i]
                    stacked_day += 1
            else:
                if data_not_exist:
                    outlogstr   += '%02d    -1\n' %day
                else:
                    outlogstr   += '%02d    0\n' %day
        #=====================
        # end loop over days
        #=====================
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
                    i                           = chan_size*ich1 + ich2
                    out_monthly_fname           = out_monthly_dir+'/COR_'+staid1+'_'+chans[ich1]+\
                                                    '_'+staid2+'_'+chans[ich2]+'.SAC'
                    monthly_header              = xcorr_common_sacheader.copy()
                    monthly_header['kcmpnm']    = chans[ich1]+chans[ich2]
                    monthly_header['user0']     = stacked_day
                    sacTr                       = obspy.io.sac.sactrace.SACTrace(data = monthly_xcorr[i], **monthly_header)
                    sacTr.write(out_monthly_fname)
        #================
        # write log file
        #================
        logfname    = datadir+'/log_xcorr/'+self.monthdir+'/'+staid1+'/'+staid1+'_'+staid2+'.log'
        outlogstr   = 'total: '+str(Nvalid_day) +' days\n' + outlogstr
        if Nnodata == len(self.daylst):
            outlogstr   = 'NODATA\n' + outlogstr
        elif Nvalid_day == 0:
            outlogstr   = 'SKIPPED\n' + outlogstr
        else:
            outlogstr   = 'SUCCESS\n' + outlogstr
        with open(logfname, 'w') as fid:
            fid.writelines(outlogstr)
        return

def amph_to_xcorr_for_mp(in_xcorr_pair, datadir, chans=['LHZ', 'LHE', 'LHN'], ftlen = True,\
            tlen = 84000., mintlen = 20000., sps = 1., lagtime = 3000., CorOutflag = 0, \
            fprcs = False, fastfft=True, runtype = 0, verbose=False, verbose2=False):
    process_id   = multiprocessing.current_process().pid
    try:
        in_xcorr_pair.convert_amph_to_xcorr(datadir=datadir, chans=chans, ftlen = ftlen,\
            tlen = tlen, mintlen = mintlen, sps = sps,  lagtime = lagtime, CorOutflag = CorOutflag,\
                    fprcs = fprcs, fastfft=fastfft, runtype = runtype, verbose=verbose, verbose2=verbose2, process_id = process_id)
    except:
        staid1  = in_xcorr_pair.netcode1 + '.' + in_xcorr_pair.stacode1
        staid2  = in_xcorr_pair.netcode2 + '.' + in_xcorr_pair.stacode2
        logfname= datadir+'/log_xcorr/'+in_xcorr_pair.monthdir+'/'+staid1+'/'+staid1+'_'+staid2+'.log'
        with open(logfname, 'w') as fid:
            fid.writelines('FAILED\n')
    return
