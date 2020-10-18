# -*- coding: utf-8 -*-
"""
ASDF for receiver function processing
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.rfuncs.rfbase as rfbase
import surfpy.rfuncs._rf_funcs as _rf_funcs

import numpy as np
import obspy
from datetime import datetime
import timeit
import warnings
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'

ref_header_default  = {'otime': '', 'network': '', 'station': '', 'stla': 12345, 'stlo': 12345, 'evla': 12345, 'evlo': 12345, 'evdp': 0.,
                        'dist': 0., 'az': 12345, 'baz': 12345, 'delta': 12345, 'npts': 12345, 'b': 12345, 'e': 12345, 'arrival': 12345, 'phase': '',
                        'tbeg': 12345, 'tend': 12345, 'hslowness': 12345, 'ghw': 12345, 'VR':  12345, 'moveout': -1}

class processASDF(rfbase.baseASDF):
    """ Class for receiver function processs
    =================================================================================================================
    version history:
        2020/09/08
    =================================================================================================================
    """
    def compute(self, inrefparam = _rf_funcs.InputRefparam(), refslow = 0.06, saveampc = True, verbose = False,
                    startdate = None, enddate = None, fs = 40., walltimeinhours = None, walltimetol = 2000., startind = 1):
        """Compute receiver function and post processed data(moveout)
        ====================================================================================================================
        ::: input parameters :::
        inrefparam  - input parameters for receiver function, refer to InputRefparam in CURefPy for details
        saveampc    - save amplitude corrected post processed data
        =====================================================================================================================
        """
        if walltimeinhours != None:
            walltime        = walltimeinhours*3600.
        else:
            walltime        = 1e10
        stime4compute       = timeit.default_timer()
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        try:
            stime4ref   = obspy.core.utcdatetime.UTCDateTime(startdate)
        except:
            stime4ref   = obspy.UTCDateTime(0)
        try:
            etime4ref   = obspy.core.utcdatetime.UTCDateTime(enddate)
        except:
            etime4ref   = obspy.UTCDateTime()
        delta           = 1./fs
        Nsta            = len(self.waveforms.list())
        ista            = startind-1
        print ('[%s] [RECEIVER FUNCS] start' %(datetime.now().isoformat().split('.')[0] ))
        for staid in (self.waveforms.list())[(startind-1):]:
            etime4compute       = timeit.default_timer()
            if etime4compute - stime4compute > walltime - walltimetol:
                print ('================================== End computation due to walltime ======================================')
                print ('start from '+str(ista+1)+' next run!')
                break
            netcode, stacode    = staid.split('.')
            ista                += 1
            print ('[%s] [RECEIVER FUNCS] station: %s %d/%d' %(datetime.now().isoformat().split('.')[0], staid, ista, Nsta))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos  = self.waveforms[staid].coordinates
            stla        = tmppos['latitude']
            stlo        = tmppos['longitude']
            elev        = tmppos['elevation_in_m']
            ievent      = 0
            Ndata       = 0              
            for event in self.cat:
                evid            = 'E%05d' %ievent
                event_id        = event.resource_id.id.split('=')[-1]
                pmag            = event.preferred_magnitude()
                magnitude       = pmag.mag
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                porigin         = event.preferred_origin()
                otime           = porigin.time
                evlo            = porigin.longitude
                evla            = porigin.latitude
                evdp            = porigin.depth
                # tag
                oyear           = otime.year
                omonth          = otime.month
                oday            = otime.day
                ohour           = otime.hour
                omin            = otime.minute
                osec            = otime.second
                label           = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
                tag             = 'body_'+label
                ievent          += 1
                try:
                    st          = self.waveforms[staid][tag]
                except KeyError:
                    continue
                if len(st) != 3:
                    continue
                phase           = st[0].stats.asdf.labels[0]
                if inrefparam.phase != '' and inrefparam.phase != phase:
                    continue
                if otime < stime4ref or otime > etime4ref:
                    continue
                for tr in st:
                    tr.stats.sac            = obspy.core.util.attribdict.AttribDict()
                    tr.stats.sac['evlo']    = evlo
                    tr.stats.sac['evla']    = evla
                    tr.stats.sac['evdp']    = evdp
                    tr.stats.sac['stlo']    = stlo
                    tr.stats.sac['stla']    = stla
                    tr.stats.sac['kuser0']  = evid
                    tr.stats.sac['kuser1']  = phase
                if verbose:
                    print('=== event ' + str(ievent)+' : '+event_descrip+', '+Mtype+' = '+str(magnitude))
                refTr                   = _rf_funcs.RFTrace()
                try:
                    if not refTr.get_data(Ztr = st.select(component='Z')[0], RTtr = st.select(component=inrefparam.reftype)[0],\
                            tbeg = inrefparam.tbeg, tend = inrefparam.tend):
                        continue
                except:
                    continue
                if not refTr.iter_deconv( tdel = inrefparam.tdel, f0 = inrefparam.f0, niter = inrefparam.niter,\
                        minderr = inrefparam.minderr, phase = phase ):
                    continue
                if refTr.stats.delta != delta:
                    if verbose:
                        print ('!!! WARNING: '+staid+' resampling fs = '+str(1./refTr.stats.delta) + ' --> '+str(fs))
                    try:
                        refTr.resample(sampling_rate = fs, no_filter = False)
                    except:
                        refTr.detrend()
                        refTr.filter(type = 'lowpass', freq = fs/2., zerophase = True) # prefilter
                        refTr.resample(sampling_rate = fs, no_filter = True)
                        continue
                #============================
                # store header attributes
                #============================
                ref_header              = ref_header_default.copy()
                ref_header['otime']     = str(otime)
                ref_header['network']   = netcode
                ref_header['station']   = stacode
                ref_header['stla']      = stla
                ref_header['stlo']      = stlo
                ref_header['evla']      = evla
                ref_header['evlo']      = evlo
                ref_header['evdp']      = evdp
                ref_header['dist']      = refTr.stats.sac['dist']
                ref_header['az']        = refTr.stats.sac['az']
                ref_header['baz']       = refTr.stats.sac['baz']
                ref_header['delta']     = refTr.stats.delta
                ref_header['npts']      = refTr.stats.npts
                ref_header['b']         = refTr.stats.sac['b']
                ref_header['e']         = refTr.stats.sac['e']
                ref_header['arrival']   = refTr.stats.sac['user5']
                ref_header['phase']     = phase
                ref_header['tbeg']      = inrefparam.tbeg
                ref_header['tend']      = inrefparam.tend
                ref_header['hslowness'] = refTr.stats.sac['user4']
                ref_header['ghw']       = inrefparam.f0
                ref_header['VR']        = refTr.stats.sac['user2']
                ref_header['evid']      = refTr.stats.sac['kuser0']
                staid_aux               = netcode+'_'+stacode+'_'+phase+'/'+label
                self.add_auxiliary_data(data = refTr.data, data_type = 'Ref'+inrefparam.reftype,\
                            path = staid_aux, parameters = ref_header)
                # move out to reference slowness receiver function
                if not refTr.move_out(refslow = refslow):
                    continue
                postdbase               = refTr.postdbase
                ref_header['moveout']   = postdbase.move_out_flag
                if saveampc:
                    self.add_auxiliary_data(data = postdbase.ampC, data_type = 'Ref'+inrefparam.reftype+'ampc',\
                                        path = staid_aux, parameters = ref_header)
                self.add_auxiliary_data(data = postdbase.ampTC, data_type = 'Ref'+inrefparam.reftype+'moveout',\
                                        path = staid_aux, parameters = ref_header)
                Ndata                   += 1 
            print('=== %d data streams processed' %Ndata)
        return
    
    def harmonic_stripping(self, outdir = None, data_type = 'RefRmoveout', VR = 80., tdiff = 0.08, phase = 'P',\
            reftype = 'R', fs = 40., endtime = 10., savetxt = False, savepredat = True):
        """Harmonic stripping analysis
        ====================================================================================================================
        ::: input parameters :::
        outdir          - output directory
        data_type       - datatype, default is 'RefRmoveout', moveouted radial receiver function
        VR              - threshold variance reduction for quality control
        tdiff           - threshold trace difference for quality control
        phase           - phase, default = 'P'
        reftype         - receiver function type, default = 'R'
        =====================================================================================================================
        """
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        if outdir is None:
            savetxt     = False
        if not savetxt:
            outsta      = None
        Nsta            = len(self.waveforms.list())
        ista            = 0
        print ('[%s] [HARMONIC STRIPPING] start' %(datetime.now().isoformat().split('.')[0]))
        for staid in self.waveforms.list():
            
            netcode, stacode    = staid.split('.')
            ista                += 1
            print ('[%s] [HARMONIC STRIPPING] station: %s %d/%d' %(datetime.now().isoformat().split('.')[0], staid, ista, Nsta))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tmppos          = self.waveforms[staid].coordinates
            stla                = tmppos['latitude']
            stlo                = tmppos['longitude']
            elev                = tmppos['elevation_in_m']
            ievent              = 0
            postLst             = _rf_funcs.PostRefLst()
            if savetxt:
                outsta          = outdir+'/'+staid
                if not os.path.isdir(outsta):
                    os.makedirs(outsta)
            Nraw                = 0
            for event in self.cat:
                evid            = 'E%05d' %ievent
                event_id        = event.resource_id.id.split('=')[-1]
                pmag            = event.preferred_magnitude()
                magnitude       = pmag.mag
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                porigin         = event.preferred_origin()
                otime           = porigin.time
                evlo            = porigin.longitude
                evla            = porigin.latitude
                evdp            = porigin.depth
                # tag
                oyear           = otime.year
                omonth          = otime.month
                oday            = otime.day
                ohour           = otime.hour
                omin            = otime.minute
                osec            = otime.second
                label           = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
                ievent          += 1
                try:
                    subdset     = self.auxiliary_data[data_type][netcode+'_'+stacode+'_'+phase][label]
                except KeyError:
                    continue
                Nraw            += 1
                ref_header      = subdset.parameters
                # quality control
                if ref_header['moveout'] <0 or ref_header['VR'] < VR:
                    continue
                if np.any(np.isnan(subdset.data[()])):
                    continue
                pdbase          = _rf_funcs.PostDatabase()
                pdbase.ampTC    = subdset.data[()]
                pdbase.header   = subdset.parameters
                pdbase.label    = label
                postLst.append(pdbase)
            #------------------------------------------
            # harmonic stripping
            #------------------------------------------
            qcLst               = postLst.remove_bad(outdir = outsta, fs = fs, endtime = endtime, savetxt = savetxt)
            staid_aux           = netcode+'_'+stacode+'_'+phase
            if len(qcLst) == 0:
                print('0/'+str(Nraw)+' receiver function traces ')
                count_header    = {'Nraw': Nraw, 'Nhs': 0}
                self.add_auxiliary_data(data = np.array([]), data_type = 'Ref'+reftype+'HScount',
                    path = staid_aux, parameters = count_header)
                continue
            qcLst               = qcLst.thresh_tdiff(tdiff=tdiff)
            Nhs                 = len(qcLst)
            if Nhs == 0:
                print('0/'+str(Nraw)+' receiver function traces ')
                count_header    = {'Nraw': Nraw, 'Nhs': 0}
                self.add_auxiliary_data(data = np.array([]), data_type = 'Ref'+reftype+'HScount',
                    path = staid_aux, parameters = count_header)
                continue
            else:
                count_header    = {'Nraw': Nraw, 'Nhs': Nhs}
                self.add_auxiliary_data(data = np.array([]), data_type = 'Ref'+reftype+'HScount',
                    path = staid_aux, parameters = count_header)
                print('=== receiver function traces: %d/%d ' %(Nhs, Nraw))
            A0_0, A0_1, A1_1, phi1_1, A0_2, A2_2, phi2_2, \
                A0, A1, A2, phi1, phi2, mfArr0, mfArr1, mfArr2, mfArr3,\
                    Aavg, Astd, gbaz, gdat, gun = qcLst.harmonic_stripping(outdir = outsta, stacode = staid)
            #------------------------------------------
            # store data
            #------------------------------------------
            # raw quality controlled data
            time                = qcLst[0].ampTC[:, 0]
            ind                 = time < endtime
            delta               = 1./fs
            for pdbase in qcLst:
                ref_header      = pdbase.header
                time            = pdbase.ampTC[:, 0]
                obsdata         = pdbase.ampTC[:, 1]
                time            = time[ind]
                obsdata         = obsdata[ind]
                label           = pdbase.label
                self.add_auxiliary_data(data = obsdata, data_type = 'Ref'+reftype+'HSdata',
                    path = staid_aux+'/obs/'+label, parameters = ref_header)
            # binned data
            gdat                = gdat[ind, :]
            gbaz                = gbaz[ind, :]
            gun                 = gun[ind, :]
            npts, Nbin          = gdat.shape
            binheader           = {'Nbin': Nbin, 'npts': npts, 'delta': delta}
            self.add_auxiliary_data(data = gdat, data_type = 'Ref'+reftype+'HSbindata',
                    path = staid_aux+'/data', parameters = binheader)
            self.add_auxiliary_data(data = gbaz, data_type = 'Ref'+reftype+'HSbindata',
                    path = staid_aux+'/baz', parameters = binheader)
            self.add_auxiliary_data(data = gun, data_type = 'Ref'+reftype+'HSbindata',
                    path = staid_aux+'/sem', parameters = binheader)
            # average data
            avgheader           = {'npts': npts, 'delta': delta}
            self.add_auxiliary_data(data = Aavg[ind], data_type = 'Ref'+reftype+'HSavgdata',
                    path = staid_aux+'/data', parameters = avgheader)
            self.add_auxiliary_data(data = Astd[ind], data_type = 'Ref'+reftype+'HSavgdata',
                    path = staid_aux+'/std', parameters = avgheader)
            if savepredat:
                for pdbase in qcLst:
                    ref_header      = pdbase.header
                    label           = pdbase.label
                    time            = pdbase.ampTC[:,0]
                    obsdata         = pdbase.ampTC[:,1]
                    time            = time[ind]
                    obsdata         = obsdata[ind]
                    evid            = ref_header['evid']
                    # compute predicted data
                    baz             = pdbase.header['baz']/180.*np.pi
                    A012data        = _rf_funcs.A012_3pre(baz, A0, A1, phi1, A2, phi2)[ind]
                    A1data          = _rf_funcs.A1_3pre(baz, A1, phi1)[ind]
                    A2data          = _rf_funcs.A2_3pre(baz, A2, phi2)[ind]
                    A0data          = A0[ind]
                    diffdata        = A012data-obsdata
                    # store data
                    self.add_auxiliary_data(data = diffdata, data_type = 'Ref'+reftype+'HSdata',
                        path = staid_aux+'/diff/'+label, parameters = {})
                    self.add_auxiliary_data(data = A012data, data_type = 'Ref'+reftype+'HSdata',
                        path = staid_aux+'/rep/'+label, parameters = {})
                    self.add_auxiliary_data(data = A0data, data_type = 'Ref'+reftype+'HSdata',
                        path = staid_aux+'/rep0/'+label, parameters = {})
                    self.add_auxiliary_data(data = A1data, data_type = 'Ref'+reftype+'HSdata',
                        path = staid_aux+'/rep1/'+label, parameters = {})
                    self.add_auxiliary_data(data = A2data, data_type = 'Ref'+reftype+'HSdata',
                        path = staid_aux+'/rep2/'+label, parameters = {})
            #-------------------------------------
            # store harmonic stripping results
            #-------------------------------------
            # A0 inversion
            self.add_auxiliary_data(data = A0_0[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0/A0', parameters = {})
            # A1 inversion
            self.add_auxiliary_data(data = A0_1[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A1/A0', parameters = {})
            self.add_auxiliary_data(data = A1_1[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A1/A1', parameters = {})
            self.add_auxiliary_data(data = phi1_1[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A1/phi1', parameters = {})
            # A2 inversion
            self.add_auxiliary_data(data=A0_2[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A2/A0', parameters = {})
            self.add_auxiliary_data(data=A2_2[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A2/A2', parameters = {})
            self.add_auxiliary_data(data = phi2_2[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A2/phi2', parameters = {})
            # A0_A1_A2 inversion
            A3header        = {'npts': npts, 'delta': delta}
            self.add_auxiliary_data(data = A0[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/A0', parameters = A3header)
            self.add_auxiliary_data(data = A1[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/A1', parameters = A3header)
            self.add_auxiliary_data(data = A2[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/A2', parameters = A3header)
            self.add_auxiliary_data(data = phi1[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/phi1', parameters = A3header)
            self.add_auxiliary_data(data = phi2[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/phi2', parameters = A3header)
            # misfit between A0 and R[i]
            self.add_auxiliary_data(data = mfArr0[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/mf_A0_obs', parameters = {})
            # misfit between A0+A1+A2 and R[i], can be used as uncertainties
            self.add_auxiliary_data(data = mfArr1[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/mf_A0_A1_A2_obs', parameters = {})
            # misfit between A0+A1+A2 and binned data, can be used as uncertainties
            self.add_auxiliary_data(data = mfArr2[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/mf_A0_A1_A2_bin', parameters = {})
            # weighted misfit between A0+A1+A2 and binned data, can be used as uncertainties
            self.add_auxiliary_data(data = mfArr3[ind], data_type = 'Ref'+reftype+'HSmodel',
                    path = staid_aux+'/A0_A1_A2/wmf_A0_A1_A2_bin', parameters = {})
        return
    
    