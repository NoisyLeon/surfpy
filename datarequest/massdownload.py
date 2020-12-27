# -*- coding: utf-8 -*-
"""
ASDF for mass downloader
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
try:
    import surfpy.datarequest.browsebase as browsebase
except:
    import browsebase
import warnings
import obspy
from obspy.taup import TauPyModel
import numpy as np
from datetime import datetime
from pyproj import Geod
from obspy.clients.fdsn.mass_downloader import RectangularDomain, \
    Restrictions, MassDownloader
import time
import os

geodist         = Geod(ellps='WGS84')
taupmodel       = TauPyModel(model="iasp91")

mondict = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


class massdownloadASDF(browsebase.baseASDF):
    

    def download_surf(self, datadir, commontime = True, fskip=True, chanrank=['LH', 'BH', 'HH'],\
            channels='ZNE', vmax = 8.0, vmin=.5, verbose=False, start_date=None, end_date=None, skipinv=True, threads_per_client = 3,\
            providers  = None, blon = 0.05, blat = 0.05):
        """request Rayleigh wave data from 
        ====================================================================================================================
        ::: input parameters :::
        lon0, lat0      - center of array. If specified, all waveform will have the same starttime and endtime
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channel         - Channel code, e.g. 'BHZ'.
                            Last character (i.e. component) can be a wildcard (‘?’ or ‘*’) to fetch Z, N and E component.
        vmin, vmax      - minimum/maximum velocity for surface wave window
        =====================================================================================================================
        """
        if providers is None:
            providers = ['BGR', 'ETH', 'GFZ', 'ICGC', 'INGV', 'IPGP',\
                'IRIS', 'KNMI', 'KOERI', 'LMU', 'NCEDC', 'NIEP', 'NOA', 'ODC', 'ORFEUS',\
                'RASPISHAKE', 'RESIF', 'SCEDC', 'TEXNET', 'USP']
        self.get_limits_lonlat()
        minlongitude= self.minlon
        maxlongitude= self.maxlon
        if minlongitude > 180.:
            minlongitude -= 360.
        if maxlongitude > 180.:
            maxlongitude -= 360.
        lon0        = (minlongitude + maxlongitude)/2.
        lat0        = (self.minlat + self.maxlat)/2.
        domain      = RectangularDomain(minlatitude=self.minlat - blat, maxlatitude=self.maxlat+blat,
                        minlongitude=minlongitude-blon, maxlongitude=maxlongitude+blon)
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        try:
            stime4down  = obspy.core.utcdatetime.UTCDateTime(start_date)
        except:
            stime4down  = obspy.UTCDateTime(0)
        try:
            etime4down  = obspy.core.utcdatetime.UTCDateTime(end_date)
        except:
            etime4down  = obspy.UTCDateTime()
        mdl                 = MassDownloader(providers = providers)
        chantype_list       = []
        for chantype in chanrank:
            chantype_list.append('%s[%s]' %(chantype, channels))
        channel_priorities  = tuple(chantype_list)
        t1 = time.time()
        # loop over events
        for event in self.cat:
            pmag            = event.preferred_magnitude()
            try:
                magnitude       = pmag.mag
                evdp            = porigin.depth/1000.
            except:
                pass
            try:
                Mtype           = pmag.magnitude_type
                event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                porigin         = event.preferred_origin()
                otime           = porigin.time
                timestr         = otime.isoformat()
                evlo            = porigin.longitude
                evla            = porigin.latitude
            except:
                pass
            if otime < stime4down or otime > etime4down:
                continue
            if commontime:
                dist, az, baz   = obspy.geodetics.gps2dist_azimuth(evla, evlo, lat0, lon0) # distance is in m
                dist            = dist/1000.
                starttime       = otime+dist/vmax
                endtime         = otime+dist/vmin
            oyear               = otime.year
            omonth              = otime.month
            oday                = otime.day
            ohour               = otime.hour
            omin                = otime.minute
            osec                = otime.second
            label               = '%d_%s_%d_%d_%d_%d' %(oyear, mondict[omonth], oday, ohour, omin, osec)
            eventdir            = datadir + '/' +label
            if not os.path.isdir(eventdir):
                os.makedirs(eventdir)
            event_logfname      = eventdir+'/download.log'
            if fskip and os.path.isfile(event_logfname):
                continue
            stationxml_storage  = "%s/{network}/{station}.xml" %eventdir
            if commontime:
                restrictions = Restrictions(
                    # starttime and endtime
                    starttime   = starttime,
                    endtime     = endtime,
                    # You might not want to deal with gaps in the data.
                    reject_channels_with_gaps=True,
                    # And you might only want waveforms that have data for at least
                    # 95 % of the requested time span.
                    minimum_length=0.95,
                    # No two stations should be closer than 10 km to each other.
                    minimum_interstation_distance_in_m=10E3,
                    # Only HH or BH channels. If a station has HH channels,
                    # those will be downloaded, otherwise the BH. Nothing will be
                    # downloaded if it has neither.
                    channel_priorities  = channel_priorities,
                    sanitize            = True)
                mseed_storage = eventdir
                # # # mseed_storage   = ("%s/{network}/{station}/{channel}.{location}.%s.mseed" %(datadir, label) )
                mdl.download(domain, restrictions, mseed_storage=mseed_storage,
                    stationxml_storage=stationxml_storage, threads_per_client=threads_per_client)       
            else:
                # loop over stations
                Nsta            = 0
                for network in self.inv:
                    for station in network:
                        netcode = network.code
                        stacode = station.code
                        staid   = netcode+'.'+stacode
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            st_date     = station.start_date
                            ed_date     = station.end_date
                        if skipinv and (otime < st_date or otime > ed_date):
                            continue
                        stlo            = station.longitude
                        stla            = station.latitude
                        dist, az, baz   = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                        dist            = dist/1000.
                        starttime       = otime+dist/vmax
                        endtime         = otime+dist/vmin
                        
                        restrictions = Restrictions(
                            network     = netcode,
                            station     = stacode,
                            # starttime and endtime
                            starttime   = starttime,
                            endtime     = endtime,
                            # You might not want to deal with gaps in the data.
                            reject_channels_with_gaps=True,
                            # And you might only want waveforms that have data for at least
                            # 95 % of the requested time span.
                            minimum_length=0.95,
                            # No two stations should be closer than 10 km to each other.
                            minimum_interstation_distance_in_m=10E3,
                            # Only HH or BH channels. If a station has HH channels,
                            # those will be downloaded, otherwise the BH. Nothing will be
                            # downloaded if it has neither.
                            channel_priorities  = channel_priorities,
                            sanitize            = True)
                        mseed_storage = eventdir
                        # mseed_storage   = ("%s/{network}/{station}/{channel}.{location}.%s.mseed" %(datadir, label) )
                        mdl.download(domain, restrictions, mseed_storage=mseed_storage,
                            stationxml_storage=stationxml_storage, threads_per_client=threads_per_client)
                        Nsta    += 1
            print ('--- [RAYLEIGH DATA DOWNLOAD] Event: %s %s' %(otime.isoformat(), event_descrip))
            with open(event_logfname, 'w') as fid:
                fid.writelines('evlo: %g, evla: %g\n' %(evlo, evla))
                if commontime:
                    fid.writelines('distance: %g km\n' %dist)
                fid.writelines('DONE\n')
        return

    def download_rf(self, datadir, minDelta=30, maxDelta=150, fskip=True, chanrank=['BH', 'HH'], channels = 'ZNE', phase='P',\
            startoffset=-30., endoffset=60.0, verbose=False, start_date=None, end_date=None, skipinv=True, threads_per_client = 3,\
            providers  = None, blon = 0.05, blat = 0.05):
        """request receiver function data from IRIS server
        ====================================================================================================================
        ::: input parameters :::
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channels        - Channel code, e.g. 'BHZ'.
                            Last character (i.e. component) can be a wildcard (‘?’ or ‘*’) to fetch Z, N and E component.
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channel_rank    - rank of channel types
        phase           - body wave phase to be downloaded, arrival time will be computed using taup
        start/endoffset - start and end offset for downloaded data
        =====================================================================================================================
        """
        if providers is None:
            providers = ['BGR', 'ETH', 'GFZ', 'ICGC', 'INGV', 'IPGP',\
                'IRIS', 'KNMI', 'KOERI', 'LMU', 'NCEDC', 'NIEP', 'NOA', 'ODC', 'ORFEUS',\
                'RASPISHAKE', 'RESIF', 'SCEDC', 'TEXNET', 'USP']
        self.get_limits_lonlat()
        minlongitude= self.minlon
        maxlongitude= self.maxlon
        if minlongitude > 180.:
            minlongitude -= 360.
        if maxlongitude > 180.:
            maxlongitude -= 360.
        lon0        = (minlongitude + maxlongitude)/2.
        lat0        = (self.minlat + self.maxlat)/2.
        domain      = RectangularDomain(minlatitude=self.minlat - blat, maxlatitude=self.maxlat+blat,
                        minlongitude=minlongitude-blon, maxlongitude=maxlongitude+blon)
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        try:
            stime4down  = obspy.core.utcdatetime.UTCDateTime(start_date)
        except:
            stime4down  = obspy.UTCDateTime(0)
        try:
            etime4down  = obspy.core.utcdatetime.UTCDateTime(end_date)
        except:
            etime4down  = obspy.UTCDateTime()
        mdl                 = MassDownloader(providers = providers)
        chantype_list       = []
        for chantype in chanrank:
            chantype_list.append('%s[%s]' %(chantype, channels))
        channel_priorities  = tuple(chantype_list)
        t1 = time.time()
        # loop over events
        ievent              = 0
        for event in self.cat:
            event_id        = event.resource_id.id.split('=')[-1]
            pmag            = event.preferred_magnitude()
            magnitude       = pmag.mag
            Mtype           = pmag.magnitude_type
            event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin         = event.preferred_origin()
            otime           = porigin.time
            if otime < stime4down or otime > etime4down:
                continue
            ievent          += 1
            try:
                print('[%s] [DOWNLOAD BODY WAVE] ' %datetime.now().isoformat().split('.')[0] + \
                            'Event ' + str(ievent)+': '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
            except:
                print('[%s] [DOWNLOAD BODY WAVE] ' %datetime.now().isoformat().split('.')[0] + \
                    'Event ' + str(ievent)+': '+ str(otime)+' '+ event_descrip+', M = '+str(magnitude))
            evlo            = porigin.longitude
            evla            = porigin.latitude
            try:
                evdp        = porigin.depth/1000.
            except:
                continue
            # log file existence
            oyear               = otime.year
            omonth              = otime.month
            oday                = otime.day
            ohour               = otime.hour
            omin                = otime.minute
            osec                = otime.second
            label               = '%d_%s_%d_%d_%d_%d' %(oyear, mondict[omonth], oday, ohour, omin, osec)
            eventdir            = datadir + '/' +label
            if not os.path.isdir(eventdir):
                os.makedirs(eventdir)
            event_logfname      = eventdir+'/download.log'
            if fskip and os.path.isfile(event_logfname):
                continue
            stationxml_storage  = "%s/{network}/{station}.xml" %eventdir
            # loop over stations
            Nsta            = 0
            for network in self.inv:
                for station in network:
                    netcode = network.code
                    stacode = station.code
                    staid   = netcode+'.'+stacode
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        st_date     = station.start_date
                        ed_date     = station.end_date
                    if skipinv and (otime < st_date or otime > ed_date):
                        continue
                    stlo            = station.longitude
                    stla            = station.latitude
                    dist, az, baz   = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                    dist            = dist/1000.
                    if baz<0.:
                        baz             += 360.
                    Delta               = obspy.geodetics.kilometer2degrees(dist)
                    if Delta<minDelta:
                        continue
                    if Delta>maxDelta:
                        continue
                    arrivals            = taupmodel.get_travel_times(source_depth_in_km=evdp, distance_in_degree=Delta, phase_list=[phase])#, receiver_depth_in_km=0)
                    try:
                        arr             = arrivals[0]
                        arrival_time    = arr.time
                        rayparam        = arr.ray_param_sec_degree
                    except IndexError:
                        continue
                    starttime       = otime + arrival_time + startoffset
                    endtime         = otime + arrival_time + endoffset
                    restrictions    = Restrictions(
                        network     = netcode,
                        station     = stacode,
                        # starttime and endtime
                        starttime   = starttime,
                        endtime     = endtime,
                        # You might not want to deal with gaps in the data.
                        reject_channels_with_gaps=True,
                        # And you might only want waveforms that have data for at least
                        # 95 % of the requested time span.
                        minimum_length=0.95,
                        # No two stations should be closer than 10 km to each other.
                        minimum_interstation_distance_in_m=10E3,
                        channel_priorities  = channel_priorities,
                        sanitize            = True)
                    mseed_storage = eventdir
                    mdl.download(domain, restrictions, mseed_storage=mseed_storage,
                        stationxml_storage=stationxml_storage, threads_per_client=threads_per_client)
                    Nsta    += 1
            print ('--- [DOWNLOAD BODY WAVE] Event: %s %s' %(otime.isoformat(), event_descrip))
            with open(event_logfname, 'w') as fid:
                fid.writelines('evlo: %g, evla: %g\n' %(evlo, evla))
                fid.writelines('DONE\n')
        return
   