# -*- coding: utf-8 -*-
"""
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.map_dat as map_dat
map_path    = map_dat.__path__._path[0]

import matplotlib.pylab as plb
import matplotlib.pyplot as plt

import numpy as np
from scipy import fftpack
import obspy
from obspy.taup import TauPyModel


taupmodel       = TauPyModel(model="iasp91")
#---------------------------------------------------------
# preparing model arrays for stretching
#---------------------------------------------------------
dz4stretch      = 0.5

def _model4stretch_ak135():
    modelfname  = map_path+'/ak135_Q'
    ak135Arr    = np.loadtxt(modelfname)
    hak135      = ak135Arr[:, 0]
    vsak135     = ak135Arr[:, 1]
    vpak135     = ak135Arr[:, 2]
    zak135      = np.cumsum(hak135)
    zak135      = np.append(0., zak135)
    vsak135     = np.append(vsak135[0], vsak135)
    vpak135     = np.append(vpak135[0], vpak135)
    fak135vs    = interp1d(zak135, vsak135, kind='nearest')
    fak135vp    = interp1d(zak135, vpak135, kind='nearest')
    zmax        = 240.
    zarr        = np.arange(int(zmax/dz4stretch), dtype=np.float64)*dz4stretch
    
    vsarr       = fak135vs(zarr)
    vparr       = fak135vp(zarr)
    return vsarr, vparr

def _model4stretch_iasp91():
    modelfname  = map_path+'/IASP91.mod'
    iasp91Arr   = np.loadtxt(modelfname)
    ziasp91     = iasp91Arr[:, 0]
    vsiasp91    = iasp91Arr[:, 3]
    vpiasp91    = iasp91Arr[:, 2]
    zmax        = 240.
    zarr        = np.arange(int(zmax/dz4stretch), dtype=np.float64)*dz4stretch
    vsarr       = np.interp(zarr, ziasp91, vsiasp91)
    vparr       = np.interp(zarr, ziasp91, vpiasp91)
    nz          = zarr.size
    return vsarr, vparr, nz

vs4stretch, vp4stretch, nz4stretch  = _model4stretch_iasp91()

def _gaussFilter( dt, nft, f0 ):
    """
    Compute a gaussian filter in the freq domain which is unit area in time domain
    private function for IterDeconv
    ================================================================================
    :::input parameters:::
    dt      - sampling time interval
    nft     - number freq points
    f0      - width of filter
    
    Output:
    gauss   - Gaussian filter array (numpy)
    filter has the form: exp( - (0.5*w/f0)^2 ) the units of the filter are 1/s
    ================================================================================
    """
    df                  = 1.0/(nft*dt)
    nft21               = 0.5*nft + 1
    # get frequencies
    f                   = df*np.arange(nft21)
    w                   = 2*np.pi*f
    w                   = w/f0
    kernel              = w**2
    # compute the gaussian filter
    gauss               = np.zeros(nft)
    gauss[:int(nft21)]  = np.exp( -0.25*kernel )/dt
    gauss[int(nft21):]  = np.flipud(gauss[1:int(nft21)-1])
    return gauss

def _phaseshift( x, nfft, DT, TSHIFT ):
    """Add a shift to the data into the freq domain, private function for IterDeconv
    """
    Xf      = fftpack.fft(x)
    # Xf      = np.fft.fft(x)
    # phase shift in radians
    shift_i = round(TSHIFT/DT) # removed +1 from here.
    p       = np.arange(nfft)+1
    p       = 2*np.pi*shift_i/(nfft)*p
    # apply shift
    Xf      = Xf*(np.cos(p) - 1j*np.sin(p))
    # back into time
    x       = np.real( fftpack.ifft(Xf) )/np.cos(2*np.pi*shift_i/nfft)
    # x       = np.real( np.fft.ifft(Xf) )/np.cos(2*np.pi*shift_i/nfft)
    return x

def _FreFilter(inW, FilterW, dt ):
    """Filter input array in frequency domain, private function for IterDeconv
    """
    # # FinW    = np.fft.fft(inW)
    FinW    = fftpack.fft(inW)
    FinW    = FinW*FilterW*dt
    # # FilterdW= np.real(np.fft.ifft(FinW))
    FilterdW= np.real(fftpack.ifft(FinW))
    return FilterdW

def _stretch(tarr, data, slow, refslow=0.06, modeltype=0):
    """Stretch data to vertically incident receiver function given slowness, private function for move_out
    """
    dt          = tarr[1] - tarr[0]
    dz          = dz4stretch
    if modeltype == 0:
        vparr       = vp4stretch
        vsarr       = vs4stretch
        nz          = nz4stretch
    else:
        zmax        = 240.
        zarr        = np.arange(int(zmax/dz), dtype=np.float64)*dz
        nz          = zarr.size
        # layer array
        harr        = np.ones(nz, dtype=np.float64)*dz
        # velocity arrays
        vpvs        = 1.7
        vp          = 6.4
        vparr       = np.ones(nz, dtype=np.float64)*vp
        vparr       = vparr + (zarr>60.)*np.ones(nz, dtype = np.float64) * 1.4
        vsarr       = vparr/vpvs
    # 1/vsarr**2 and 1/vparr**2
    sv2         = vsarr**(-2)
    pv2         = vparr**(-2)
    # dz/vs - dz/vp, time array for vertically incident wave
    s1          = np.ones(nz, dtype=np.float64)*refslow*refslow
    difft       = np.zeros(nz+1, dtype=np.float64)
    difft[1:]   = (np.sqrt(sv2-s1) - np.sqrt(pv2-s1)) * dz
    cumdifft    = np.cumsum(difft)
    # # dz*(tan(a1p) - tan(a1s))*sin(a0p)/vp0 +dz / cos(a1s)/vs - dz / cos(a1p)/vp
    # # s = sin(a1s)/vs = sin(a1p)/vp = sin(a0p) / vp0
    # time array for wave with given slowness
    s2          = np.ones(nz, dtype=np.float64)*slow*slow
    difft2      = np.zeros(nz+1, dtype=np.float64)
    difft2[1:]  = (np.sqrt(sv2-s2)-np.sqrt(pv2-s2))*dz # dz/
    cumdifft2   = np.cumsum(difft2)
    # interpolate data to correspond to cumdifft2 array
    nseis       = np.interp(cumdifft2, tarr, data)
    # get new time array
    tf          = cumdifft[-1]
    ntf         = int(tf/dt)
    tarr2       = np.arange(ntf, dtype=np.float64)*dt
    data2       = np.interp(tarr2, cumdifft, nseis)
    return tarr2, data2

#------------------------------------------------
# functions for harmonic stripping
#------------------------------------------------

def _group ( inbaz, indat):
    """Group data according to back-azimuth, private function for harmonic stripping
    """
    binwidth    = 30
    nbin        = int((360+1)/binwidth)
    outbaz      = np.array([])
    outdat      = np.array([])
    outun       = np.array([])
    for i in range(nbin):
        bazmin  = i*binwidth
        bazmax  = (i+1)*binwidth
        tbaz    = i*binwidth + float(binwidth)/2
        tdat    = indat[(inbaz>=bazmin)*(inbaz<bazmax)]
        if (len(tdat) > 0):
            outbaz      = np.append(outbaz, tbaz)
            outdat      = np.append(outdat, tdat.mean())
            if (len(tdat)>1):
                if tdat.std() == 0.:
                    outun   = np.append(outun, 0.1 )
                else:
                    outun   = np.append(outun, tdat.std()/(np.sqrt(len(tdat))) )
            if (len(tdat)==1):
                outun   = np.append(outun, 0.1)
    return outbaz, outdat, outun

def _difference ( aa, bb, NN):
    """Compute difference between two input array, private function for harmonic stripping
    """
    if NN > 0:
        L   = min(len(aa), len(bb), NN)
    else:
        L   = min(len(aa), len(bb))
    aa      = aa[:L]
    bb      = bb[:L]
    diff    = np.sum((aa-bb)*(aa-bb))
    diff    = diff / L
    return np.sqrt(diff)

def _invert_A0 ( inbaz, indat, inun ):   #only invert for A0 part
    """invert by assuming only A0, private function for harmonic stripping
    """
    Nbaz    = inbaz.size 
    U       = np.zeros((Nbaz, Nbaz), dtype=np.float64)
    np.fill_diagonal(U, 1./inun)
    G       = np.ones((Nbaz, 1), dtype=np.float64)
    G       = np.dot(U, G)
    d       = indat.T
    d       = np.dot(U, d)
    model   = np.linalg.lstsq(G, d, rcond = -1)[0]
    A0      = model[0]
    predat  = np.dot(G, model)
    predat  = predat[:Nbaz]
    inun    = inun[:Nbaz]
    predat  = predat*inun
    return A0, predat

def _invert_A1 ( inbaz, indat, inun ):
    """invert by assuming only A0 and A1, private function for harmonic stripping
        indat   = A0 + A1*sin(theta + phi1)
                = A0 + A1*cos(phi1)*sin(theta) + A1*sin(phi1)*cos(theta)
    """
    Nbaz    = inbaz.size 
    U       = np.zeros((Nbaz, Nbaz), dtype=np.float64)
    np.fill_diagonal(U, 1./inun)
    # construct forward operator matrix
    tG      = np.ones((Nbaz, 1), dtype=np.float64)
    tbaz    = np.pi*inbaz/180
    tGsin   = np.sin(tbaz)
    tGcos   = np.cos(tbaz)
    G       = np.append(tG, tGsin)
    G       = np.append(G, tGcos)
    G       = G.reshape((3, Nbaz))
    G       = G.T
    G       = np.dot(U, G)
    # data
    d       = indat.T
    d       = np.dot(U, d)
    # least square inversion
    model   = np.linalg.lstsq(G,d, rcond = -1)[0]
    A0      = model[0]
    A1      = np.sqrt(model[1]**2 + model[2]**2)
    phi1    = np.arctan2(model[2], model[1])
    predat  = np.dot(G, model)
    predat  = predat*inun
    return A0, A1, phi1, predat

def _invert_A2 ( inbaz, indat, inun ):
    """invert by assuming only A0 and A2, private function for harmonic stripping
        indat   = A0 + A2*sin(2*theta + phi2)
                = A0 + A1*cos(phi1)*sin(theta) + A1*sin(phi1)*cos(theta)
    """
    Nbaz    = inbaz.size 
    U       = np.zeros((Nbaz, Nbaz), dtype=np.float64)
    np.fill_diagonal(U, 1./inun)
    # construct forward operator matrix
    tG      = np.ones((Nbaz, 1), dtype=np.float64)
    tbaz    = np.pi*inbaz/180.
    tGsin   = np.sin(tbaz*2.)
    tGcos   = np.cos(tbaz*2.)
    G       = np.append(tG, tGsin)
    G       = np.append(G, tGcos)
    G       = G.reshape((3, Nbaz))
    G       = G.T
    G       = np.dot(U, G)
    # data
    d       = indat.T
    d       = np.dot(U,d)
    # least square inversion
    model   = np.linalg.lstsq(G, d, rcond = -1)[0]
    A0      = model[0]
    A2      = np.sqrt(model[1]**2 + model[2]**2)
    phi2    = np.arctan2(model[2],model[1])
    predat  = np.dot(G, model)
    predat  = predat*inun
    return A0, A2, phi2, predat

def _invert_A0_A1_A2( inbaz, indat, inun):
    """invert for A0, A1, A2, private function for harmonic stripping
    """
    Nbaz    = inbaz.size 
    U       = np.zeros((Nbaz, Nbaz), dtype=np.float64)
    np.fill_diagonal(U, 1./inun)
    # construct forward operator matrix
    tG      = np.ones((Nbaz, 1), dtype=np.float64)
    tbaz    = np.pi*inbaz/180
    tGsin   = np.sin(tbaz)
    tGcos   = np.cos(tbaz)
    tGsin2  = np.sin(tbaz*2)
    tGcos2  = np.cos(tbaz*2)
    G       = np.append(tG, tGsin)
    G       = np.append(G, tGcos)
    G       = np.append(G, tGsin2)
    G       = np.append(G, tGcos2)
    G       = G.reshape((5, Nbaz))
    G       = G.T
    G       = np.dot(U, G)
    # data
    d       = indat.T
    d       = np.dot(U,d)
    # least square inversion
    model   = np.linalg.lstsq(G, d, rcond = -1)[0]
    A0      = model[0]
    A1      = np.sqrt(model[1]**2 + model[2]**2)
    phi1    = np.arctan2(model[2],model[1])                                           
    A2      = np.sqrt(model[3]**2 + model[4]**2)
    phi2    = np.arctan2(model[4], model[3])
    # compute forward
    predat  = np.dot(G, model)
    predat  = predat*inun
    return A0, A1, phi1, A2, phi2, predat

#------------------------------------------------
# Function for computing predictions
#------------------------------------------------

def A0_0pre ( inbaz, A0 ):
    return A0

def A01_1pre ( inbaz, A0, A1, phi1):
    return A0 + A1*np.sin(inbaz+phi1)

def A1_1pre ( inbaz, A1, phi1):
    return A1*np.sin(inbaz+phi1)

def A02_2pre ( inbaz, A0, A2, phi2):
    return A0 + A2*np.sin(2*inbaz+phi2)

def A2_2pre ( inbaz, A2, phi2):
    return A2*np.sin(2*inbaz+phi2)

def A1_3pre ( inbaz, A1, phi1):
    return A1*np.sin(inbaz + phi1)

def A2_3pre ( inbaz, A2, phi2):
    return A2*np.sin(2*inbaz + phi2)

def A12_3pre ( inbaz, A1, phi1, A2, phi2 ):
    return A1*np.sin(inbaz + phi1) + A2*np.sin(2*inbaz + phi2)

def A012_3pre ( inbaz, A0, A1, phi1, A2, phi2):
    return A0 + A1*np.sin(inbaz + phi1) + A2*np.sin(2*inbaz + phi2)

def _match( data1, data2 ):
    """compute matching of two input data
    """
    Nmin        = min(data1.size, data2.size)
    data1       = data1[:Nmin]
    data2       = data2[:Nmin]
    diffdat     = data1-data2
    diffdat_avg = diffdat.mean()
    normdiffdat = np.sum((diffdat-diffdat_avg)**2)
    return np.sqrt(normdiffdat/Nmin)

class InputRefparam(object):
    """class to store input parameters for receiver function analysis
    ===============================================================================================================
    Parameters:
    reftype     - type of receiver function('R' or 'T')
    tbeg, tend  - begin/end time for trim
    tdel        - phase delay
    f0          - Gaussian width factor
    niter       - number of maximum iteration
    minderr     - minimum misfit improvement, iteration will stop if improvement between two steps is smaller than minderr
    phase       - phase name, default is P, if set to '', also the possible phases will be included
    ===============================================================================================================
    """
    def __init__(self, reftype='R', tbeg=20.0, tend=-30.0, tdel=5., f0 = 2.5, niter=200, minderr=0.001, phase='P', refslow=0.06 ):
        self.reftype        = reftype
        self.tbeg           = tbeg
        self.tend           = tend
        self.tdel           = tdel
        self.f0             = f0
        self.niter          = niter
        self.minderr        = minderr
        self.phase          = phase
        self.refslow        = refslow

class RFTrace(obspy.Trace):
    """receiver function trace class, derived from obspy.Trace
    add-on parameters:
    Ztr, RTtr   - input data, numerator(R/T) and denominator(Z)
    """
    def get_data(self, Ztr, RTtr, tbeg = 20.0, tend = -30.0):
        """read raw R/T/Z data for receiver function analysis
        Arrival time will be read/computed for given phase, then data will be trimed according to tbeg and tend.
        """
        if isinstance (Ztr, str):
            self.Ztr    = obspy.read(Ztr)[0]
        elif isinstance(Ztr, obspy.Trace):
            self.Ztr    = Ztr
        else:
            raise TypeError('Unexpecetd type for Z component trace!')
        if isinstance (RTtr, str):
            self.RTtr   = obspy.read(RTtr)[0]
        elif isinstance(RTtr, obspy.Trace):
            self.RTtr   = RTtr
        else:
            raise TypeError('Unexpecetd type for RT component trace!')
        stime           = self.Ztr.stats.starttime
        etime           = self.Ztr.stats.endtime
        if stime+tbeg > etime+tend:
            return False
        self.Ztr.trim(starttime = stime + tbeg, endtime = etime + tend)
        self.RTtr.trim(starttime = stime + tbeg, endtime = etime + tend)
        if self.Ztr.stats.npts != self.RTtr.stats.npts:
            return False
        return True
    
    def iter_deconv(self, tdel = 5., f0 = 2.5, niter = 200, minderr = 0.001, phase = 'P', addhs = True ):
        """compute receiver function with iterative deconvolution algorithmn
        ========================================================================================================================
        ::: input parameters :::
        tdel       - time delay
        f0         - Gaussian width factor
        niter      - number of maximum iteration
        minderr    - minimum misfit improvement, iteration will stop if improvement between two steps is smaller than minderr
        phase      - phase name, default is P

        ::: input data  :::
        Ztr        - read from self.Ztr
        RTtr       - read from self.RTtr
        
        ::: output data :::
        self.data  - data array(numpy)
        ::: SAC header :::
        b          - begin time
        e          - end time
        user0      - Gaussian width factor
        user2      - variance reduction, (1-rms)*100
        user4      - horizontal slowness
        ========================================================================================================================
        """
        Ztr                     = self.Ztr
        RTtr                    = self.RTtr
        dt                      = Ztr.stats.delta
        npts                    = Ztr.stats.npts
        self.stats              = RTtr.stats
        self.stats.sac['b']     = -tdel
        self.stats.sac['e']     = -tdel+(npts-1)*dt
        self.stats.sac['user0'] = f0
        if addhs:
            if not self.slowness(phase = phase):
                return False
        # arrays for inversion
        RMS         = np.zeros(niter, dtype=np.float64)  # RMS errors
        nfft        = 2**(npts-1).bit_length() # number points in fourier transform
        P0          = np.zeros(nfft, dtype=np.float64) # predicted spikes
        # Resize and rename the numerator and denominator
        U0          = np.zeros(nfft, dtype=np.float64) #add zeros to the end
        W0          = np.zeros(nfft, dtype=np.float64)
        U0[:npts]   = RTtr.data 
        W0[:npts]   = Ztr.data 
        # get filter in Freq domain 
        gauss       = _gaussFilter( dt, nfft, f0 )
        # filter signals
        Wf0         = np.fft.fft(W0)
        FilteredU0  = _FreFilter(U0, gauss, dt )
        FilteredW0  = _FreFilter(W0, gauss, dt )
        R           = FilteredU0 #  residual numerator
        # Get power in numerator for error scaling
        powerU      = np.sum(FilteredU0**2)
        # Loop through iterations
        it          = 0
        sumsq_i     = 1
        d_error     = 100*powerU + minderr
        maxlag      = int(0.5*nfft)
        while( abs(d_error) > minderr  and  it < niter ):
            it          = it+1 # iteration advance
            #   Ligorria and Ammon method
            # # # RW          = np.real(np.fft.ifft(np.fft.fft(R)*np.conj(np.fft.fft(FilteredW0))))
            RW          = np.real(fftpack.ifft(fftpack.fft(R)*np.conj(fftpack.fft(FilteredW0))))
            sumW0       = np.sum(FilteredW0**2)
            RW          = RW/sumW0
            imax        = np.argmax(abs(RW[:maxlag]))
            amp         = RW[imax]/dt; # scale the max and get correct sign
            #   compute predicted deconvolution
            P0[imax]    = P0[imax] + amp  # get spike signal - predicted RF
            P           = _FreFilter(P0, gauss*Wf0, dt*dt ) # convolve with filter
            #   compute residual with filtered numerator
            R           = FilteredU0 - P
            sumsq       = np.sum(R**2)/powerU
            RMS[it-1]   = sumsq # scaled error
            d_error     = 100*(sumsq_i - sumsq)  # change in error 
            sumsq_i     = sumsq  # store rms for computing difference in next   
        # Compute final receiver function
        P                       = _FreFilter(P0, gauss, dt )
        # Phase shift
        P                       = _phaseshift(P, nfft, dt, tdel)
        # output first nt samples
        RFI                     = P[:npts]
        # output the rms values 
        RMS                     = RMS[:it]
        if np.any(np.isnan(RFI)) or it == 0:
            return False
        # store receiver function data
        self.data               = RFI
        self.stats.sac['user2'] = (1.0-RMS[it-1])*100.0
        return True
    
    def slowness(self, phase = 'P'):
        """add horizontal slowness to user4 SAC header, distance. az, baz will also be added
        Computed for a given phase using taup and iasp91 model
        """
        evla                    = self.stats.sac['evla']
        evlo                    = self.stats.sac['evlo']
        stla                    = self.stats.sac['stla']
        stlo                    = self.stats.sac['stlo']
        dist, az, baz           = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo)
        dist                    = dist/1000.  # distance is in km
        self.stats.sac['dist']  = dist
        self.stats.sac['az']    = az
        self.stats.sac['baz']   = baz
        evdp                    = self.stats.sac['evdp']/1000.
        Delta                   = obspy.geodetics.kilometer2degrees(dist)
        arrivals                = taupmodel.get_travel_times(source_depth_in_km=evdp,\
                                                distance_in_degree=Delta, phase_list=[phase])
        try:
            arr                 = arrivals[0]
        except IndexError:
            return False
        rayparam                = arr.ray_param_sec_degree
        arr_time                = arr.time
        self.stats.sac['user4'] = rayparam
        self.stats.sac['user5'] = arr_time
        return True
    
    def init_postdbase(self):
        """initialize post-processing database
        """
        self.postdbase  = PostDatabase()
        return
    
    def move_out(self, refslow = 0.06, modeltype = 0):
        """moveout for receiver function
        """
        self.init_postdbase()
        tslow       = self.stats.sac['user4']/111.12
        ratio       = self.stats.sac['user2']
        b           = self.stats.sac['b']
        e           = self.stats.sac['e']
        baz         = self.stats.sac['baz']
        dt          = self.stats.delta
        npts        = self.stats.npts
        fs          = 1./dt
        o           = 0.
        t           = np.arange(0, npts/fs, 1./fs)
        nb          = int(np.ceil((o-b)*fs))  # index for t = 0.
        nt          = np.arange(0+nb, 0+nb+20*fs, 1) # nt= nb ~ nb+ 20*fs, index array for data 
        if nt[-1]>npts:
            return False
        if len(nt)==1:
            data    = self.data[np.array([np.int_(nt)])]
        else:
            data    = self.data[np.int_(nt)]
        tarr1       = (nt - nb)/fs  # time array for move-outed data
        flag        = 0 # flag signifying whether postdatabase has been written or not
        #---------------------------------------------------------------------------------
        # Step 1: Discard data with too large or too small H slowness
        #---------------------------------------------------------------------------------
        if (tslow <= 0.04 or tslow > 0.1):
            self.postdbase.move_out_flag= -3
            self.postdbase.value        = tslow
            flag                        = 1
        refvp       = 6.0
        #-----------------------------------------------------------------------------------------------
        # Step 2: Discard data with too large Amplitude in receiver function after amplitude correction
        #-----------------------------------------------------------------------------------------------
        # correct amplitude to reference horizontal slowness
        reffactor   = np.arcsin(refslow*refvp)/np.arcsin(tslow*refvp)
        data        = data*reffactor
        absdata     = np.abs(data)
        maxdata     = absdata.max()
        if ( maxdata > 1 and flag == 0):
            self.postdbase.move_out_flag= -2
            self.postdbase.value1       = maxdata
            flag                        = 1
        #----------------------------------------
        # Step 3: Stretch Data
        #----------------------------------------
        tarr2, data2= _stretch(tarr1, data, tslow, refslow=refslow, modeltype=modeltype)
        #--------------------------------------------------------
        # Step 4: Discard data with negative value at zero time
        #--------------------------------------------------------
        if (data2[0] < 0 and flag == 0):
            self.postdbase.move_out_flag= -1
            self.postdbase.value1       = data2[0]
            flag                        = 1     
        if (flag == 0):
            self.postdbase.move_out_flag  = 1
            self.postdbase.value1       = None
        #--------------------------------------------------------
        # Step 5: Store the original and stretched data
        #--------------------------------------------------------
        DATA1                   = data/1.42
        L                       = DATA1.size
        self.postdbase.ampC     = np.append(tarr1,DATA1)
        self.postdbase.ampC     = self.postdbase.ampC.reshape((2, L))
        self.postdbase.ampC     = self.postdbase.ampC.T
        DATA2                   = data2/1.42
        L                       = DATA2.size
        self.postdbase.ampTC    = np.append(tarr2, DATA2)
        self.postdbase.ampTC    = self.postdbase.ampTC.reshape((2, L))
        self.postdbase.ampTC    = self.postdbase.ampTC.T
        self.stats.sac['user6'] = self.postdbase.move_out_flag
        return True
    
    def save_data(self, outdir):
        """Save receiver function and post processed (moveout) data to output directory
        """
        outfname                = outdir+'/'+self.stats.sac['kuser0']+'.sac'
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        self.write(outfname, format = 'sac')
        try:
            np.savez( outdir+'/'+self.stats.sac['kuser0']+'.post', self.postdbase.ampC, self.postdbase.ampTC)
        except:
            return
        return 

class HSStream(obspy.core.stream.Stream):
    """harmonic stripping stream, derived from obspy.Stream
    """
    def get_trace(self, network, station, indata, baz, dt, starttime):
        """get trace
        """
        tr                  = obspy.Trace()
        tr.stats.network    = network
        tr.stats.station    = station
        tr.stats.channel    = str(int(baz))
        tr.stats.delta      = dt
        tr.data             = indata
        tr.stats.starttime  = starttime
        self.append(tr)
        return
    
            
    def plot_hs(self, ampfactor = 40, title = '', ax = plt.subplot(), delta = 0.025):
        """plot harmonic stripping stream accoring to back-azimuth
        ===============================================================================================================
        ::: input parameters :::
        ampfactor   - amplication factor for visulization
        title       - title
        ax          - subplot object
        delta       - target dt for decimation
        ===============================================================================================================
        """
        ymax    = 361.
        ymin    = -1.
        for trace in self.traces:
            # # # downsamplefactor    = int(delta/trace.stats.delta)
            # # # if downsamplefactor!=1:
            # # #     trace.decimate(factor = downsamplefactor, no_filter = True)
            trace.resample(sampling_rate = 1./delta, no_filter = False)
            dt      = trace.stats.delta
            time    = dt*np.arange(trace.stats.npts)
            yvalue  = trace.data*ampfactor
            backazi = float(trace.stats.channel)
            ax.plot(time, yvalue + backazi, '-k', lw=0.05)
            ax.fill_between(time, y2=backazi, y1=yvalue+backazi, where=yvalue>0, color='red', lw=0.01, interpolate=True)
            ax.fill_between(time, y2=backazi, y1=yvalue+backazi, where=yvalue<0, color='blue', lw=0.01, interpolate=True)
        plt.axis([0., 10., ymin, ymax])
        plt.xlabel('Time(sec)')
        plt.title(title)
        return
    
    def save_HS(self, outdir, prefix):
        """Save harmonic stripping stream to MiniSEED
        """
        outfname    = outdir+'/'+prefix+'.mseed'
        self.write(outfname, format='mseed')
        return
    
    def load_HS(self, datadir, prefix):
        """Load harmonic stripping stream from MiniSEED
        """
        infname     = datadir+'/'+prefix+'.mseed'
        self.traces = obspy.read(infname)
        return

class PostDatabase(object):
    """
    A class to store post precessed receiver function
    ===============================================================================================================
    Parameters:
    move_out_flag - succeeded compute moveout or not
                    1   - valid move-outed receiver function
                    -1  - negative value at zero
                    -2  - too large amplitude, value1 = maximum amplitude 
                    -3  - too large or too small horizontal slowness, value1 = horizontal slowness
    ampC        - amplitude corrected receiver function
    ampTC       - moveout of receiver function (amplitude and time corrected)
    header      - receiver function header
    tdiff       - trace difference
    ===============================================================================================================
    """
    def __init__(self):
        self.move_out_flag  = None
        self.ampC           = np.array([]) 
        self.ampTC          = np.array([]) 
        self.header         = {}
        self.tdiff          = 10.
        
        
class PostRefLst(object):
    """a class to store as list of PostDatabase object
    """
    def __init__(self,PostDatas = None):
        self.PostDatas=[]
        if isinstance(PostDatas, PostDatabase):
            PostDatas = [PostDatas]
        if PostDatas:
            self.PostDatas.extend(PostDatas)
    
    def __add__(self, other):
        """
        Add two PostRefLst with self += other.
        """
        if isinstance(other, StaInfo):
            other = PostRefLst([other])
        if not isinstance(other, PostRefLst):
            raise TypeError
        PostDatas = self.PostDatas + other.PostDatas
        return self.__class__(PostDatas=PostDatas)

    def __len__(self):
        """
        Return the number of PostDatas in the PostRefLst object.
        """
        return len(self.PostDatas)

    def __getitem__(self, index):
        """
        __getitem__ method of PostRefLst objects.
        :return: PostDatabase objects
        """
        if isinstance(index, slice):
            return self.__class__(PostDatas=self.PostDatas.__getitem__(index))
        else:
            return self.PostDatas.__getitem__(index)

    def append(self, postdata):
        """
        Append a single PostDatabase object to the current PostRefLst object.
        """
        if isinstance(postdata, PostDatabase):
            self.PostDatas.append(postdata)
        else:
            msg = 'Append only supports a single PostDatabase object as an argument.'
            raise TypeError(msg)
        return self
    
    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of traces.
        """
        return self.PostDatas.__delitem__(index)
    
    def remove_bad(self, outdir = None, fs = 40., endtime = 10., savetxt = False):
        """Remove bad measurements and group data
        ===============================================================================================================
        ::: input parameters :::
        outdir      - output directory
        fs          - sampling rate
        endtime     - required endtime
        savetxt     - output txt results or not
        ::: output :::
        outdir/wmean.txt, outdir/bin_%d_txt
        ===============================================================================================================
        """
        if outdir is None:
            savetxt = False
        outlst      = PostRefLst()
        lens        = np.array([]) # array to store length for each moveout trace
        bazArr      = np.array([]) # array to store baz for each moveout trace
        delta       = 1./fs
        for PostData in self.PostDatas:
            time    = PostData.ampTC[:,0]
            data    = PostData.ampTC[:,1]
            L       = time.size
            # quality control
            if PostData.header['delta'] != delta:
                continue
            if abs(data).max()>1:
                continue
            if data[abs(time)<0.1].min()<0.02:
                continue
            if time[-1] < endtime:
                continue
            PostData.Len= L
            lens        = np.append(lens, L)
            outlst.append(PostData)
            bazArr      = np.append( bazArr, np.floor(PostData.header['baz']))
        if len(outlst) == 0:
            return outlst
        #------------------------------------------------------
        # group the data
        #------------------------------------------------------
        # grouped data array
        gbaz        = np.array([])
        gdata       = np.array([])
        gun         = np.array([])
        Lmin        = int(lens.min())
        dat_avg     = np.zeros(Lmin, dtype=np.float64)
        weight_avg  = np.zeros(Lmin, dtype=np.float64)
        time1       = outlst[0].ampTC[:,0]
        time1       = time1[:Lmin]
        NLst        = len(outlst)
        tdat        = np.zeros(NLst, dtype=np.float64)
        for i in range(Lmin):
            for j in range (NLst):
                tdat[j] = outlst[j].ampTC[i, 1]
            b1,d1,u1    = _group(bazArr, tdat)
            gbaz        = np.append(gbaz, b1)
            gdata       = np.append(gdata, d1)
            gun         = np.append(gun, u1)
            d1DIVu1     = d1/u1
            DIVu1       = 1./u1
            wmean       = np.sum(d1DIVu1)
            weight      = np.sum(DIVu1)
            if (weight > 0.):
                dat_avg[i]  = wmean/weight
            else:
                print ("weight is zero!!! ", len(d1), u1, d1)
                sys.exit()
            weight_avg[i]   = np.sum(u1)/len(u1)
        Ngbaz       = len(b1)
        gbaz        = gbaz.reshape((Lmin, Ngbaz))
        gdata       = gdata.reshape((Lmin, Ngbaz))
        gun         = gun.reshape((Lmin, Ngbaz))
        # compute and store trace difference, tdiff for quality control
        for i in range(len(outlst)):
            time            = outlst[i].ampTC[:,0]
            data            = outlst[i].ampTC[:,1]
            Lmin            = min( len(time) , len(time1) )
            tdiff           = _difference ( data[:Lmin], dat_avg[:Lmin], 0)
            outlst[i].tdiff = tdiff
        # Save data
        if savetxt:
            outname     = outdir+"/wmean.txt"
            outwmeanArr = np.append(time1, dat_avg)
            outwmeanArr = np.append(outwmeanArr, weight_avg)
            outwmeanArr = outwmeanArr.reshape((3, Lmin))
            outwmeanArr = outwmeanArr.T
            np.savetxt(outname, outwmeanArr, fmt='%g')
            # Save baz bin data
            for i in range (Ngbaz): # back -azimuth
                outname     = outdir+"/bin_%d_txt" % (int(gbaz[0][i]))
                outbinArr   = np.append(time1[:Lmin], gdata[:, i])
                outbinArr   = np.append(outbinArr, gun[:, i])
                outbinArr   = outbinArr.reshape((3, Lmin ))
                outbinArr   = outbinArr.T
                np.savetxt(outname, outbinArr, fmt='%g')
        return outlst
    
    def thresh_tdiff(self, tdiff=0.08):
        """remove data given threshold trace difference value
        """
        outlst      = PostRefLst()
        for PostData in self.PostDatas:
            if PostData.tdiff<tdiff:
                outlst.append(PostData)
        return outlst
    
    def harmonic_stripping(self, stacode, outdir=None, savetxt=False):
        """harmonic stripping analysis for quality controlled data.
        ===============================================================================================================
        ::: input parameters :::
        stacode     - station code( e.g. TA.R11A )
        outdir      - output directory
        savetxt     - output txt results or not
        
        ::: output :::
        outdir/bin_%d_rf.dat, outdir/A0.dat, outdir/A1.dat, outdir/A2.dat, outdir/A0_A1_A2.dat
        outdir/average_vr.dat, outdir/variance_reduction.dat
        outdir/prestre_*, outdir/repstre_*, outdir/obsstre_*, outdir/repstre_*
        outdir/0repstre_*, outdir/1repstre_*, outdir/2repstre_*, outdir/diffstre_*
        ===============================================================================================================
        """
        if outdir is None:
            savetxt = False
        NLst    = len(self.PostDatas)
        baz     = np.zeros(NLst, dtype=np.float64)
        lens    = np.zeros(NLst, dtype=np.float64)
        names   = []
        eventT  = []
        for i in range(NLst):
            PostData= self.PostDatas[i]
            time    = PostData.ampTC[:,0]
            lens[i] = time.size
            baz[i]  = np.floor(PostData.header['baz'])
            name    = 'moveout_'+str(int(PostData.header['baz']))+'_'+stacode+'_'+str(PostData.header['otime'])
            names.append(name)
            eventT.append(PostData.header['otime'])
        # store all time and data arrays
        # # Lmin    = int(min(lens.min(), (time[time<=endtime]).size))
        Lmin    = int(lens.min())
        atime   = np.zeros((Lmin, NLst), dtype=np.float64)
        adata   = np.zeros((Lmin, NLst), dtype=np.float64)
        for i in range(NLst):
            PostData        = self.PostDatas[i]
            time            = PostData.ampTC[:,0]
            data            = PostData.ampTC[:,1]
            adata[:, i]     = data[:Lmin]
            atime[:, i]     = time[:Lmin]
        # parameters in 3 different inversion
        # best fitting A0
        A0_0    = np.zeros(Lmin, dtype=np.float64)
        # best fitting A0 , A1 and phi1
        A0_1    = np.zeros(Lmin, dtype=np.float64)
        A1_1    = np.zeros(Lmin, dtype=np.float64)
        phi1_1  = np.zeros(Lmin, dtype=np.float64)
        # best fitting A0 , A2 and phi2
        A0_2    = np.zeros(Lmin, dtype=np.float64)
        A2_2    = np.zeros(Lmin, dtype=np.float64)
        phi2_2  = np.zeros(Lmin, dtype=np.float64)
        # best fitting A0, A1, phi1, A2 and phi2 
        A0      = np.zeros(Lmin, dtype=np.float64)
        A1      = np.zeros(Lmin, dtype=np.float64)
        A2      = np.zeros(Lmin, dtype=np.float64)
        phi1    = np.zeros(Lmin, dtype=np.float64)
        phi2    = np.zeros(Lmin, dtype=np.float64)
        # misfit arrays
        mfArr0  = np.zeros(Lmin, dtype=np.float64)  # misfit between A0 and R[i]
        mfArr1  = np.zeros(Lmin, dtype=np.float64)  # misfit between A0+A1+A2 and R[i]
        mfArr2  = np.zeros(Lmin, dtype=np.float64)  # misfit between A0+A1+A2 and binned data
        mfArr3  = np.zeros(Lmin, dtype=np.float64)  # weighted misfit between A0+A1+A2 and binned data
        Aavg    = np.zeros(Lmin, dtype=np.float64)  # average amplitude
        Astd    = np.zeros(Lmin, dtype=np.float64)
        # grouped data
        gbaz    = np.array([], dtype=np.float64)
        gdata   = np.array([], dtype=np.float64)
        gun     = np.array([], dtype=np.float64)
        tdat    = np.zeros(NLst, dtype=np.float64)
        for i in range (Lmin):
            for j in range(NLst):
                tdat[j]         = self.PostDatas[j].ampTC[i, 1]
            baz1,tdat1,udat1    = _group(baz, tdat)
            gbaz                = np.append(gbaz, baz1)
            gdata               = np.append(gdata, tdat1)
            gun                 = np.append(gun, udat1)
            #------------------------------------------------------
            # inversions
            #------------------------------------------------------
            # invert for best-fitting A0
            (tempA0, predat0)                   = _invert_A0(baz1, tdat1, udat1)
            A0_0[i]                             = tempA0
            # invert for best-fitting A0, A1 and phi1
            (tempA0, tempA1, tempphi1, predat1) = _invert_A1(baz1, tdat1, udat1)
            A0_1[i]                             = tempA0
            A1_1[i]                             = tempA1
            phi1_1[i]                           = tempphi1
            # invert for best-fitting A0, A2 and phi2
            (tempA0, tempA2, tempphi2, predat2) = _invert_A2(baz1, tdat1, udat1)
            A0_2[i]                             = tempA0
            A2_2[i]                             = tempA2
            phi2_2[i]                           = tempphi2
            # invert for best-fitting A0, A1 and A2
            (tempA0, tempA1, tempphi1, tempA2, tempphi2, predat) \
                                                = _invert_A0_A1_A2 (baz1,tdat1,udat1)
            A0[i]                               = tempA0
            A1[i]                               = tempA1
            phi1[i]                             = tempphi1
            A2[i]                               = tempA2
            phi2[i]                             = tempphi2
            # average amplitude and std
            Aavg[i]                             = tdat.mean()
            Astd[i]                             = tdat.std()
            # compute misfit for raw baz array
            misfit0         = np.sqrt(np.sum((A0[i] - adata[i, :])**2)/NLst)
            predatraw       = A012_3pre( baz*np.pi/180., A0=A0[i], A1=A1[i], phi1=phi1[i], A2=A2[i], phi2=phi2[i])
            misfit1         = np.sqrt(np.sum((predatraw - adata[i, :])**2)/NLst)
            if misfit0 < 0.005:
                misfit0     = 0.005
            if misfit1 < 0.005:
                misfit1     = 0.005
            mfArr0[i]       = misfit0
            mfArr1[i]       = misfit1
            # compute misfit for binned baz array
            Nbin            = baz1.size
            predatbin       = A012_3pre( baz1*np.pi/180., A0=A0[i], A1=A1[i], phi1=phi1[i], A2=A2[i], phi2=phi2[i])
            misfit2         = np.sqrt(np.sum((predatbin - tdat1)**2)/Nbin)
            wNbin           = np.sum(1./(udat1**2))
            misfit3         = np.sqrt(np.sum( (predatbin - tdat1)**2 /(udat1**2) )/wNbin)
            mfArr2[i]       = misfit2
            mfArr3[i]       = misfit3
        Nbin        = baz1.size
        gbaz        = gbaz.reshape((Lmin, Nbin))
        gdata       = gdata.reshape((Lmin, Nbin))
        gun         = gun.reshape((Lmin, Nbin))
        #-----------------------------------------
        # Output grouped data
        #-----------------------------------------
        if savetxt:
            # save binned ref data
            for i in range(Nbin): #baz
                binfname    = outdir+"/bin_%g_rf.dat" % (gbaz[0, i])
                outbinArr   = np.append(atime[:, 0], gdata[:, i])
                outbinArr   = np.append(outbinArr, gun[:, i])
                outbinArr   = outbinArr.reshape((3, Lmin ))
                outbinArr   = outbinArr.T
                np.savetxt(binfname, outbinArr, fmt='%g')
            # save A0 amplitude from A0 inversion
            time        = atime[:, 0]
            timeA0      = time[(A0_0>-2.)*(A0_0<2.)]
            A0_out      = A0_0[(A0_0>-2.)*(A0_0<2.)]
            LA0         = timeA0.size
            outArrf0    = np.append(timeA0, A0_out)
            outArrf0    = outArrf0.reshape((2, LA0))
            outArrf0    = outArrf0.T
            np.savetxt(outdir+"/A0.dat", outArrf0, fmt='%g', header='time A0; A0 inversion')
            # save A0, A1 and phi1 from A1 inversion
            timeA1      = time[(A0_1>-2.)*(A0_1<2.)]
            A0_out      = A0_1[(A0_1>-2.)*(A0_1<2.)]
            A1_out      = A1_1[(A0_1>-2.)*(A0_1<2.)]
            phi1_out    = phi1_1[(A0_1>-2.)*(A0_1<2.)]
            phi1_out    = phi1_out + (phi1_out<0.)*np.pi
            LA1         = timeA1.size
            outArrf1    = np.append(timeA1, A0_out)
            outArrf1    = np.append(outArrf1, A1_out)
            outArrf1    = np.append(outArrf1, phi1_out)
            outArrf1    = outArrf1.reshape((4, LA1))
            outArrf1    = outArrf1.T
            np.savetxt(outdir+"/A1.dat", outArrf1, fmt='%g', header='time A0 A1 phi1; A1 inversion')
            # save A0, A2 and phi2 from A2 inversion
            timeA2      = time[(A0_2>-2.)*(A0_2<2.)]
            A0_out      = A0_2[(A0_2>-2.)*(A0_2<2.)]
            A2_out      = A2_2[(A0_2>-2.)*(A0_2<2.)]
            phi2_out    = phi2_2[(A0_2>-2.)*(A0_2<2.)]
            phi2_out    = phi2_out + (phi2_out<0.)*np.pi
            LA2         = timeA2.size
            outArrf2    = np.append(timeA2, A0_out)
            outArrf2    = np.append(outArrf2, A2_out)
            outArrf2    = np.append(outArrf2, phi2_out)
            outArrf2    = outArrf2.reshape((4,LA2))
            outArrf2    = outArrf2.T
            np.savetxt(outdir+"/A2.dat", outArrf2, fmt='%g', header='time A0 A2 phi2; A2 inversion')
            # save A0, A1, phi1, A2, phi2 and misfit arrays from A0_A1_A2 inversion
            timeA3      = time[(A0>-200.)*(A0<200.)]
            A0_out      = A0[(A0>-200.)*(A0<200.)]
            A1_out      = A1[(A0>-200.)*(A0<200.)]
            phi1_out    = phi1[(A0>-200.)*(A0<200.)]*180./np.pi
            A2_out      = A2[(A0>-200.)*(A0<200.)]
            phi2_out    = phi2[(A0>-200.)*(A0<200.)]*180./np.pi
            mf0_out     = mfArr0[(A0>-200.)*(A0<200.)]
            mf1_out     = mfArr1[(A0>-200.)*(A0<200.)]
            mf2_out     = mfArr2[(A0>-200.)*(A0<200.)]
            mf3_out     = mfArr3[(A0>-200.)*(A0<200.)]
            Aavg_out    = Aavg[(A0>-200.)*(A0<200.)]
            Astd_out    = Astd[(A0>-200.)*(A0<200.)]
            LA3         = timeA3.size
            outArrf3    = np.append(timeA3, A0_out)
            outArrf3    = np.append(outArrf3, A1_out)
            outArrf3    = np.append(outArrf3, phi1_out)
            outArrf3    = np.append(outArrf3, A2_out)
            outArrf3    = np.append(outArrf3, phi2_out)
            outArrf3    = np.append(outArrf3, mf0_out)
            outArrf3    = np.append(outArrf3, mf1_out)
            outArrf3    = np.append(outArrf3, mf2_out)
            outArrf3    = np.append(outArrf3, mf3_out)
            outArrf3    = np.append(outArrf3, Aavg_out)
            outArrf3    = np.append(outArrf3, Astd_out)
            outArrf3    = outArrf3.reshape((12, LA3))
            outArrf3    = outArrf3.T
            np.savetxt(outdir+"/A0_A1_A2.dat", outArrf3, fmt='%g', \
            header      = 'time A0 A1 phi1 A2 phi2 misfit(A0-rawdata) misfit(A3-rawdata) misfit(A3-bindata) weighted_misfit(A3-bindata) Aavg Astd; A3 inversion')
            #---------------------------------------
            # save predicted data
            #---------------------------------------
            # broadcasting baz to 2d array of shape ( Lmin, NLst )
            baz2d       = np.repeat(baz, Lmin)
            baz2d       = baz2d.reshape(NLst, Lmin)
            baz2d       = baz2d.T
            # broadcasting inverted arrays to 2d array of shape ( Lmin, NLst )
            phi12d      = np.repeat(phi1, NLst)
            phi12d      = phi12d.reshape(Lmin, NLst)
            phi22d      = np.repeat(phi2, NLst)
            phi22d      = phi22d.reshape(Lmin, NLst)
            A12d        = np.repeat(A1, NLst)
            A12d        = A12d.reshape(Lmin, NLst)
            A22d        = np.repeat(A2, NLst)
            A22d        = A22d.reshape(Lmin, NLst)
            # predicted A0, A1, A2, A0+A1+A2 data arrays
            A0data      = np.repeat(A0, NLst)
            A0data      = A0data.reshape(Lmin, NLst)
            A1data      = A12d*np.sin(baz2d/180.*np.pi + phi12d)
            A2data      = A22d*np.sin(2.*baz2d/180.*np.pi + phi22d)
            A3data      = A0data + A1data + A2data
            diffdata    = adata - A3data
            
            indtime     = time<=10.
            mf0         = np.zeros(NLst, dtype=np.float64)
            mf1         = np.zeros(NLst, dtype=np.float64)
            mf2         = np.zeros(NLst, dtype=np.float64)
            mf3         = np.zeros(NLst, dtype=np.float64)
            with open(outdir+"/misfit.dat","w") as fidmf:
                for i in range(NLst):
                    tempbaz     = baz[i]
                    tempbaz1    = float(baz[i])*np.pi/180.
                    outname     = outdir+"/pre" + names[i]
                    time_out    = time[indtime]
                    Ntime       = time_out.size
                    obs         = adata[indtime, i]
                    # A0 inversion results
                    A0_0_out    = A0_0pre(tempbaz1, A0_0)[indtime]
                    # A1 inversion results
                    A01_1_out   = A01_1pre(tempbaz1, A0_1, A1_1, phi1_1)[indtime]
                    A1_1_out    = A1_1pre(tempbaz1, A1_1, phi1_1)[indtime]
                    # A2 inversion results
                    A02_2_out   = A02_2pre(tempbaz1, A0_2, A2_2, phi2_2)[indtime]
                    A2_2_out    = A2_2pre(tempbaz1, A2_2, phi2_2)[indtime]
                    # A0_A1_A2 inversion results
                    A012_3_out  = A012_3pre(tempbaz1, A0, A1, phi1, A2, phi2)[indtime]
                    A1_3_out    = A1_3pre(tempbaz1, A1, phi1)[indtime]
                    A2_3_out    = A2_3pre(tempbaz1, A2, phi2)[indtime]
                    # output array
                    outpreArr   = np.append(time_out, obs)
                    outpreArr   = np.append(outpreArr, A0_0_out)
                    outpreArr   = np.append(outpreArr, A01_1_out)
                    outpreArr   = np.append(outpreArr, A02_2_out)
                    outpreArr   = np.append(outpreArr, A012_3_out)
                    outpreArr   = np.append(outpreArr, A1_1_out)
                    outpreArr   = np.append(outpreArr, A2_2_out)
                    outpreArr   = np.append(outpreArr, A1_3_out)
                    outpreArr   = np.append(outpreArr, A2_3_out)
                    outpreArr   = outpreArr.reshape((10, Ntime))
                    outpreArr   = outpreArr.T
                    np.savetxt(outname, outpreArr, fmt='%g')
                    # variance reduction arrays
                    mf0[i]      = _match(A0_0_out, adata[indtime, i])
                    mf1[i]      = _match(A01_1_out, adata[indtime, i])
                    mf2[i]      = _match(A02_2_out, adata[indtime, i])
                    mf3[i]      = _match(A012_3_out, adata[indtime, i])
                    # write to variance reduction file
                    tempstr = "%d %g %g %g %g %s\n" %(baz[i], mf0[i], mf1[i], mf2[i], mf3[i], names[i])
                    fidmf.write(tempstr)
            with open(outdir+"/average_misfit.dat","w") as famf:
                tempstr = "%g %g %g %g\n" %(mf0.mean(), mf1.mean(), mf2.mean(), mf3.mean())
                famf.write(tempstr) 
            for i in range (NLst):
                outname = outdir+"/diff" + names[i]
                outArr  = np.append(time, diffdata[:,i])
                outArr  = outArr.reshape((2, Lmin))
                outArr  = outArr.T
                np.savetxt(outname, outArr, fmt='%g', header='time diffdata(A3-obs); A3 inversion')
            
            for i in range (NLst):
                outname = outdir+"/rep" + names[i]
                outArr  = np.append(time, A3data[:,i])
                outArr  = outArr.reshape((2,Lmin))
                outArr  = outArr.T
                np.savetxt(outname, outArr, fmt='%g', header='time A3; A3 inversion')
            
            for i in range (NLst):
                outname = outdir+"/0rep" + names[i]
                outArr  = np.append(time, A0)
                outArr  = outArr.reshape((2, Lmin))
                outArr  = outArr.T
                np.savetxt(outname, outArr, fmt='%g', header='time A0; A3 inversion')
            
            for i in range (NLst):
                outname = outdir+"/1rep" + names[i]
                outArr  = np.append(time, A1data[:,i])
                outArr  = outArr.reshape((2,Lmin))
                outArr  = outArr.T
                np.savetxt(outname, outArr, fmt='%g', header='time A1; A3 inversion')
                
            for i in range (NLst):
                outname = outdir+"/2rep" + names[i]
                outArr  = np.append(time, A2data[:,i])
                outArr  = outArr.reshape((2,Lmin))
                outArr  = outArr.T
                np.savetxt(outname, outArr, fmt='%g', header='time A2; A3 inversion')
                
            for i in range (NLst):
                outname = outdir+"/obs" + names[i]
                outArr  = np.append(time, adata[:,i])
                outArr  = outArr.reshape((2,Lmin))
                outArr  = outArr.T
                np.savetxt(outname, outArr, fmt='%g', header='time obs')
                
        return A0_0, A0_1, A1_1, phi1_1, A0_2, A2_2, phi2_2, A0, A1, A2, phi1, phi2,\
                mfArr0, mfArr1, mfArr2, mfArr3, Aavg, Astd, gbaz, gdata, gun
            

class hsdatabase(object):
    """harmonic stripping database, include 6 harmonic stripping streams
    """
    def __init__(self, obsST = HSStream(), diffST = HSStream(), repST = HSStream(),\
        repST0 = HSStream(), repST1 = HSStream(), repST2 = HSStream()):
        self.obsST  = obsST
        self.diffST = diffST
        self.repST  = repST
        self.repST0 = repST0
        self.repST1 = repST1
        self.repST2 = repST2
        return
    
    def plot(self, outdir='', stacode='', ampfactor=40, delta=0.025, longitude='', latitude='', browseflag=False, saveflag=True,\
            obsflag=True, diffflag=False, repflag=True, rep0flag=True, rep1flag=True, rep2flag=True):
        """Plot harmonic stripping streams accoring to back-azimuth
        ===============================================================================================================
        ::: input parameters :::
        outdir              - output directory for saving figure
        stacode             - station code
        ampfactor           - amplication factor for visulization
        delta               - target dt for decimation
        longitude/latitude  - station location
        browseflag          - browse figure or not
        saveflag            - save figure or not
        obsflag             - plot observed receiver function or not
        diffflag            - plot difference of observed and predicted receiver function or not
        repflag             - plot predicted receiver function or not
        rep0flag            - plot A0 of receiver function or not
        rep1flag            - plot A1 of receiver function or not
        rep2flag            - plot A2 of receiver function or not
        ===============================================================================================================
        """
        totalpn     = obsflag+diffflag+repflag+rep0flag+rep1flag+rep2flag
        cpn         = 1
        plt.close('all')
        fig         = plb.figure(num=1, figsize=(12.,8.), facecolor='w', edgecolor='k')
        ylabelflag  = False
        if obsflag:
            ax          = plt.subplot(1, totalpn, cpn)
            cpn         = cpn+1
            self.obsST.plot_hs(ampfactor=ampfactor, delta=delta, title='Observed', ax=ax)
            plt.ylabel('Backazimuth(°)', fontsize = 20)
            plt.xlabel('Time (s)', fontsize = 20)
            plt.title('Observed', fontsize = 20)
            ylabelflag  = True
            ax.tick_params(axis='y', labelsize=15)
            ax.tick_params(axis='x', labelsize=15)
        if diffflag:
            ax          = plt.subplot(1, totalpn, cpn)
            cpn         = cpn+1
            self.diffST.plot_hs(ampfactor=ampfactor, delta=delta, title='Residual', ax=ax)
            if not ylabelflag:
                plt.ylabel('Backazimuth(deg)')
            plt.xlabel('Time (s)', fontsize = 20)
            plt.title('Residual', fontsize = 20)
            ax.tick_params(axis='y', labelsize=0.1)
            ax.tick_params(axis='x', labelsize=15)
        if repflag:
            ax  = plt.subplot(1, totalpn,cpn)
            cpn = cpn+1
            self.repST.plot_hs(ampfactor=ampfactor, delta=delta, title='H', ax=ax)
            if not ylabelflag:
                plt.ylabel('Backazimuth(deg)')
            plt.xlabel('Time (s)', fontsize = 20)
            plt.title('H', fontsize = 20)
            ax.tick_params(axis='y', labelsize=0.1)
            ax.tick_params(axis='x', labelsize=15)
        if rep0flag:
            ax  = plt.subplot(1, totalpn,cpn)
            cpn = cpn+1
            self.repST0.plot_hs(ampfactor=ampfactor, delta=delta, title='A0', ax=ax)
            if not ylabelflag:
                plt.ylabel('Backazimuth(deg)')
            plt.xlabel('Time (s)', fontsize = 20)
            plt.title('A0', fontsize = 20)
            ax.tick_params(axis='y', labelsize=0.1)
            ax.tick_params(axis='x', labelsize=15)
        if rep1flag:
            ax  = plt.subplot(1, totalpn,cpn)
            cpn = cpn+1
            self.repST1.plot_hs(ampfactor=ampfactor, delta=delta, title='A1', ax=ax)
            if not ylabelflag:
                plt.ylabel('Backazimuth(deg)')
            plt.xlabel('Time (s)', fontsize = 20)
            plt.title('A1', fontsize = 20)
            ax.tick_params(axis='y', labelsize=0.1)
            ax.tick_params(axis='x', labelsize=15)
        if rep2flag:
            ax  = plt.subplot(1, totalpn,cpn)
            self.repST2.plot_hs(ampfactor=ampfactor, delta=delta, title='A2', ax=ax)
            if not ylabelflag:
                plt.ylabel('Backazimuth(deg)')
            plt.xlabel('Time (s)', fontsize = 20)
            plt.title('A2', fontsize = 20)
            ax.tick_params(axis='y', labelsize=0.1)
            ax.tick_params(axis='x', labelsize=15)
        fig.suptitle(stacode+' Longitude:'+str(longitude)+' Latitude:'+str(latitude), fontsize=15)
        if browseflag:
            plt.draw()
            plt.pause(1) # <-------
            raw_input("<Hit Enter To Close>")
            plt.close('all')
        if saveflag and outdir!='':
            fig.savefig(outdir+'/'+stacode+'_COM.pdf', orientation='landscape', format='pdf')
        return
            
    def save(self, outdir, stacode=''):
        """Save harmonic stripping streams to MiniSEED
        """
        prefix  = stacode+'_obs'
        self.obsST.save_HS(outdir, prefix)
        prefix  = stacode+'_diff'
        self.diffST.save_HS(outdir, prefix)
        prefix  = stacode+'_rep'
        self.repST.save_HS(outdir, prefix)
        prefix  = stacode+'_rep0'
        self.repST0.save_HS(outdir, prefix)
        prefix  = stacode+'_rep1'
        self.repST1.save_HS(outdir, prefix)
        prefix  = stacode+'_rep2'
        self.repST2.save_HS(outdir, prefix)
        return
    
    def load(self, datadir, stacode=''):
        """Load harmonic stripping streams from MiniSEED
        """
        prefix  = stacode+'_obs'
        self.obsST.load_HS(datadir, prefix)
        prefix  = stacode+'_diff'
        self.diffST.load_HS(datadir, prefix)
        prefix  = stacode+'_rep'
        self.repST.load_HS(datadir, prefix)
        prefix  = stacode+'_rep0'
        self.repST0.load_HS(datadir, prefix)
        prefix  = stacode+'_rep1'
        self.repST1.load_HS(datadir, prefix)
        prefix  = stacode+'_rep2'
        self.repST2.load_HS(datadir, prefix)
        return
    

