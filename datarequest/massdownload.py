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
from pyproj import Geod
from obspy.clients.fdsn.mass_downloader import RectangularDomain, \
    Restrictions, MassDownloader
import time
import os

geodist         = Geod(ellps='WGS84')
taupmodel       = TauPyModel(model="iasp91")

mondict = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


class massdownloadASDF(browsebase.baseASDF):
    

    def download_surf(self, datadir, staxmldir, commontime = True, fskip=True, chanrank=['LH', 'BH', 'HH'],\
            channels='ZNE', vmax = 8.0, vmin=.5, verbose=False, start_date=None, end_date=None, skipinv=True, threads_per_client = 3,\
            providers  = None, blon = 0.05, blat = 0.05):
        """request Rayleigh wave data from IRIS server
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
        stationxml_storage  = "%s/{network}/{station}.xml" %staxmldir
        chantype_list       = []
        for chantype in chanrank:
            chantype_list.append('%s[%s]' %(chantype, channels))
        channel_priorities  = tuple(chantype_list)
        t1 = time.time()
        # loop over events
        for event in self.cat:
            pmag            = event.preferred_magnitude()
            magnitude       = pmag.mag
            Mtype           = pmag.magnitude_type
            event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin         = event.preferred_origin()
            otime           = porigin.time
            timestr         = otime.isoformat()
            evlo            = porigin.longitude
            evla            = porigin.latitude
            evdp            = porigin.depth/1000.
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
    
    

    def download_rf(self, minDelta=30, maxDelta=150, chanrank=['BH', 'HH'], channels = 'ENZ', phase='P',\
            startoffset=-30., endoffset=60.0, verbose=False, start_date=None, end_date=None, skipinv=True,\
            label='LF', quality = 'B', name = 'LiliFeng', email_address='lfengmac@gmail.com', iris_email='breq_fast@iris.washington.edu'):
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
        header_str1     = '.NAME %s\n' %name + '.INST CU\n'+'.MAIL University of Colorado Boulder\n'
        header_str1     += '.EMAIL %s\n' %email_address+'.PHONE\n'+'.FAX\n'+'.MEDIA: Electronic (FTP)\n'
        header_str1     += '.ALTERNATE MEDIA: Electronic (FTP)\n'
        FROM            = 'no_reply@surfpy.com'
        TO              = iris_email
        title           = 'Subject: Requesting Data\n\n'
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
        for event in self.cat:
            pmag            = event.preferred_magnitude()
            magnitude       = pmag.mag
            Mtype           = pmag.magnitude_type
            event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin         = event.preferred_origin()
            otime           = porigin.time
            timestr         = otime.isoformat()
            evlo            = porigin.longitude
            evla            = porigin.latitude
            evdp            = porigin.depth/1000.
            
            if otime < stime4down or otime > etime4down:
                continue
            oyear               = otime.year
            omonth              = otime.month
            oday                = otime.day
            ohour               = otime.hour
            omin                = otime.minute
            osec                = otime.second
            header_str2         = header_str1 +'.LABEL %s_%d_%s_%d_%d_%d_%d\n' %(label, oyear, mondict[omonth], oday, ohour, omin, osec)
            header_str2         += '.QUALITY %s\n' %quality +'.END\n'
            out_str             = ''
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
                    # determine channel type
                    channel_type    = None
                    for chantype in chanrank:
                        tmpchE      = station.select(channel = chantype+'E')
                        tmpchN      = station.select(channel = chantype+'N')
                        tmpchZ      = station.select(channel = chantype+'Z')
                        if len(tmpchE)>0 and len(tmpchN)>0 and len(tmpchZ)>0:
                            channel_type    = chantype
                            break
                    if channel_type is None:
                        if verbose:
                            print('!!! NO selected channel types: '+ staid)
                        continue
                    stlo            = station.longitude
                    stla            = station.latitude
                    az, baz, dist   = geodist.inv(evlo, evla, stlo, stla)
                    dist            = dist/1000.
                    if baz<0.:
                        baz         += 360.
                    Delta           = obspy.geodetics.kilometer2degrees(dist)
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
                    # start time stampe
                    year            = starttime.year
                    month           = starttime.month
                    day             = starttime.day
                    hour            = starttime.hour
                    minute          = starttime.minute
                    second          = starttime.second
                    # end time stampe
                    year2           = endtime.year
                    month2          = endtime.month
                    day2            = endtime.day
                    hour2           = endtime.hour
                    minute2         = endtime.minute
                    second2         = endtime.second
                    day_str         = '%d %d %d %d %d %d %d %d %d %d %d %d' %(year, month, day, hour, minute, second, \
                                        year2, month2, day2, hour2, minute2, second2)
                    for tmpch in channels:
                        chan        = channel_type + tmpch
                        chan_str    = '1 %s' %chan
                        sta_str     = '%s %s %s %s\n' %(stacode, netcode, day_str, chan_str)
                        out_str     += sta_str
                    Nsta    += 1
            out_str     = header_str2 + out_str
            if Nsta == 0:
                print ('--- [RF DATA REQUEST] No data available in inventory, Event: %s %s' %(otime.isoformat(), event_descrip))
                continue
            #========================
            # send email to IRIS
            #========================
            server  = smtplib.SMTP('localhost')
            MSG     = title + out_str
            server.sendmail(FROM, TO, MSG)
            server.quit()
            print ('--- [RF DATA REQUEST] email sent to IRIS, Event: %s %s' %(otime.isoformat(), event_descrip))
        return
   