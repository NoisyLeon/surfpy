# -*- coding: utf-8 -*-
"""
Base ASDF for receiver function analysis


:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import pyasdf
import surfpy.rfuncs._rf_funcs as _rf_funcs
import numpy as np
import matplotlib.pyplot as plt
import obspy
import obspy.io.sac
from obspy.clients.fdsn.client import Client
from obspy.taup import TauPyModel
import warnings
from datetime import datetime
import glob
import tarfile
import shutil
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from pyproj import Geod

geodist         = Geod(ellps='WGS84')
taupmodel       = TauPyModel(model="iasp91")

monthdict               = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

class baseASDF(pyasdf.ASDFDataSet):
    """ An object to for ambient noise cross-correlation analysis based on ASDF database
    =================================================================================================================
    version history:
        2020/07/09
    =================================================================================================================
    """
    def __init__(
        self,
        filename,
        compression="gzip-3",
        shuffle=True,
        debug=False,
        mpi=None,
        mode="a",
        single_item_read_limit_in_mb=4096.0,
        format_version=None,
        ):
        # initialize ASDF
        super(baseASDF, self).__init__( filename = filename, compression=compression, shuffle=shuffle, debug=debug,
            mpi=mpi, mode=mode, single_item_read_limit_in_mb=single_item_read_limit_in_mb, format_version=format_version)
        #======================================
        # initializations of other attributes
        #======================================
        # range of station coverage
        try:
            limits_lonlat_param = self.auxiliary_data.Rfuncs['limits_lonlat'].parameters
            self.minlat         = limits_lonlat_param['minlat']
            self.maxlat         = limits_lonlat_param['maxlat']
            self.minlon         = limits_lonlat_param['minlon']
            self.maxlon         = limits_lonlat_param['maxlon']
        except:
            pass
        # station inventory; start/end date of the stations
        self.inv        = obspy.Inventory()
        self.start_date = obspy.UTCDateTime('2599-01-01')
        self.end_date   = obspy.UTCDateTime('1900-01-01')
        self.update_inv_info()
        return
    
    def print_info(self):
        """
        Print information of the dataset.
        """                                                                  
        outstr  = '========================================================= Receiver function Database =======================================================\n'
        outstr  += self.__str__()+'\n'                                   
        outstr  += '--------------------------------------------------------------------------------------------------------------------------------------------\n'
        if 'RefR' in self.auxiliary_data.list():
            outstr      += 'RefR                    - Radial receiver function\n'
        if 'RefRHS' in self.auxiliary_data.list():
            outstr      += 'RefRHS                  - Harmonic stripping results of radial receiver function\n'
        if 'RefRmoveout' in self.auxiliary_data.list():
            outstr      += 'RefRmoveout             - Move out of radial receiver function\n'
        if 'RefRampc' in self.auxiliary_data.list():
            outstr      += 'RefRampc                - Scaled radial receiver function\n'
        if 'RefRstreback' in self.auxiliary_data.list():
            outstr      += 'RefRstreback            - Stretch back of radial receiver function\n'
        outstr  += '============================================================================================================================================\n'
        print (outstr)
        return
    
    def get_limits_lonlat(self):
        """get the geographical limits of the stations
        """
        try:
            del self.auxiliary_data.NoiseXcorr['limits_lonlat']
        except:
            pass
        staLst      = self.waveforms.list()
        minlat      = 90.
        maxlat      = -90.
        minlon      = 360.
        maxlon      = 0.
        for staid in staLst:
            tmppos  = self.waveforms[staid].coordinates
            lat     = tmppos['latitude']
            lon     = tmppos['longitude']
            elv     = tmppos['elevation_in_m']
            if lon<0:
                lon         += 360.
            minlat  = min(lat, minlat)
            maxlat  = max(lat, maxlat)
            minlon  = min(lon, minlon)
            maxlon  = max(lon, maxlon)
        print ('latitude range: ', minlat, '-', maxlat, 'longitude range:', minlon, '-', maxlon)
        paramters   = {'minlat': minlat, 'maxlat': maxlat, 'minlon': minlon, 'maxlon': maxlon}
        self.add_auxiliary_data(data=np.array([]), data_type='Rfuncs', path='limits_lonlat', parameters=paramters)
        self.minlat = minlat
        self.maxlat = maxlat
        self.minlon = minlon
        self.maxlon = maxlon
        return
    
    def update_inv_info(self):
        """update inventory information
        """
        start_date      = self.start_date
        end_date        = self.end_date
        for staid in self.waveforms.list():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.inv    += self.waveforms[staid].StationXML
                if start_date > self.waveforms[staid].StationXML[0][0].start_date:
                    start_date  = self.waveforms[staid].StationXML[0][0].start_date
                if end_date < self.waveforms[staid].StationXML[0][0].end_date:
                    end_date    = self.waveforms[staid].StationXML[0][0].end_date
        self.start_date = start_date
        self.end_date   = end_date
        if len(self.inv) > 0:
            self.get_limits_lonlat()
        return
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        lat_centre  = (self.maxlat+self.minlat)/2.0
        lon_centre  = (self.maxlon+self.minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=self.minlat-5., urcrnrlat=self.maxlat+5., llcrnrlon=self.minlon-5.,
                        urcrnrlon=self.maxlon+5., lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        elif projection=='regional_ortho':
            m1      = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution=resolution,\
                        llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon, self.minlat, self.maxlon) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon, self.maxlat+2., self.minlon) # distance is in m
            m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                        lat_1=self.minlat, lat_2=self.maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        m.drawstates()
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    #==================================================================
    # functions for manipulating earthquake catalog
    #==================================================================
    def get_events(self, startdate, enddate, add2dbase=True, gcmt=False, Mmin=5.5, Mmax=None,
            minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, latitude=None, longitude=None,\
            minradius=None, maxradius=None, mindepth=None, maxdepth=None, magnitudetype=None, outquakeml=None):
        """Get earthquake catalog from IRIS server
        =======================================================================================================
        ::: input parameters :::
        startdate, enddate  - start/end date for searching
        Mmin, Mmax          - minimum/maximum magnitude for searching                
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
        mindepth            - Limit to events with depth, in kilometers, larger than the specified minimum.
        maxdepth            - Limit to events with depth, in kilometers, smaller than the specified maximum.
        magnitudetype       - Specify a magnitude type to use for testing the minimum and maximum limits.
        =======================================================================================================
        """
        starttime   = obspy.core.utcdatetime.UTCDateTime(startdate)
        endtime     = obspy.core.utcdatetime.UTCDateTime(enddate)
        if not gcmt:
            client  = Client('IRIS')
            try:
                catISC      = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=Mmin, maxmagnitude=Mmax, catalog='ISC',
                                minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                                latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, mindepth=mindepth,
                                maxdepth=maxdepth, magnitudetype=magnitudetype)
                endtimeISC  = catISC[0].origins[0].time
            except:
                catISC      = obspy.core.event.Catalog()
                endtimeISC  = starttime
            if endtime.julday-endtimeISC.julday >1:
                try:
                    catPDE  = client.get_events(starttime=endtimeISC, endtime=endtime, minmagnitude=Mmin, maxmagnitude=Mmax, catalog='NEIC PDE',
                                minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                                latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, mindepth=mindepth,
                                maxdepth=maxdepth, magnitudetype=magnitudetype)
                    catalog = catISC+catPDE
                except:
                    catalog = catISC
            else:
                catalog     = catISC
            outcatalog      = obspy.core.event.Catalog()
            # check magnitude
            for event in catalog:
                if event.magnitudes[0].mag < Mmin:
                    continue
                outcatalog.append(event)
        else:
            # Updated the URL on Jul 25th, 2020
            gcmt_url_old    = 'http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec17.ndk'
            gcmt_new        = 'http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY'
            if starttime.year < 2005:
                print('--- Loading catalog: '+gcmt_url_old)
                cat_old     = obspy.read_events(gcmt_url_old)
                if Mmax != None:
                    cat_old = cat_old.filter("magnitude <= %g" %Mmax)
                if maxlongitude != None:
                    cat_old = cat_old.filter("longitude <= %g" %maxlongitude)
                if minlongitude != None:
                    cat_old = cat_old.filter("longitude >= %g" %minlongitude)
                if maxlatitude != None:
                    cat_old = cat_old.filter("latitude <= %g" %maxlatitude)
                if minlatitude != None:
                    cat_old = cat_old.filter("latitude >= %g" %minlatitude)
                if maxdepth != None:
                    cat_old = cat_old.filter("depth <= %g" %(maxdepth*1000.))
                if mindepth != None:
                    cat_old = cat_old.filter("depth >= %g" %(mindepth*1000.))
                temp_stime  = obspy.core.utcdatetime.UTCDateTime('2018-01-01')
                outcatalog  = cat_old.filter("magnitude >= %g" %Mmin, "time >= %s" %str(starttime), "time <= %s" %str(endtime) )
            else:
                outcatalog      = obspy.core.event.Catalog()
                temp_stime      = copy.deepcopy(starttime)
                temp_stime.day  = 1
            while (temp_stime < endtime):
                year            = temp_stime.year
                month           = temp_stime.month
                yearstr         = str(int(year))[2:]
                monstr          = monthdict[month]
                monstr          = monstr.lower()
                if year==2005 and month==6:
                    monstr      = 'june'
                if year==2005 and month==7:
                    monstr      = 'july'
                if year==2005 and month==9:
                    monstr      = 'sept'
                gcmt_url_new    = gcmt_new+'/'+str(int(year))+'/'+monstr+yearstr+'.ndk'
                try:
                    cat_new     = obspy.read_events(gcmt_url_new, format='ndk')
                    print('--- Loading catalog: '+gcmt_url_new)
                except:
                    print('--- Link not found: '+gcmt_url_new)
                    break
                cat_new         = cat_new.filter("magnitude >= %g" %Mmin, "time >= %s" %str(starttime), "time <= %s" %str(endtime) )
                if Mmax != None:
                    cat_new     = cat_new.filter("magnitude <= %g" %Mmax)
                if maxlongitude != None:
                    cat_new     = cat_new.filter("longitude <= %g" %maxlongitude)
                if minlongitude!=None:
                    cat_new     = cat_new.filter("longitude >= %g" %minlongitude)
                if maxlatitude!=None:
                    cat_new     = cat_new.filter("latitude <= %g" %maxlatitude)
                if minlatitude!=None:
                    cat_new     = cat_new.filter("latitude >= %g" %minlatitude)
                if maxdepth != None:
                    cat_new     = cat_new.filter("depth <= %g" %(maxdepth*1000.))
                if mindepth != None:
                    cat_new     = cat_new.filter("depth >= %g" %(mindepth*1000.))
                outcatalog      += cat_new
                try:
                    temp_stime.month    +=1
                except:
                    temp_stime.year     +=1
                    temp_stime.month    = 1
        try:
            self.cat    += outcatalog
        except:
            self.cat    = outcatalog
        if add2dbase:
            self.add_quakeml(outcatalog)
        if outquakeml is not None:
            self.cat.write(outquakeml, format='quakeml')
        return
    
    def read_quakeml(self, inquakeml, add2dbase=False):
        self.cat    = obspy.read_events(inquakeml)
        if add2dbase:
            self.add_quakeml(self.cat)
        return
    
    def copy_catalog(self):
        print('=== Copying catalog from ASDF to memory')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cat    = self.events
        return
    
    def copy_catalog_fromasdf(self, asdffname):
        print('Copying catalog from ASDF file')
        indset  = pyasdf.ASDFDataSet(asdffname)
        cat     = indset.events
        self.add_quakeml(cat)
        return

    #==================================================================
    # functions for manipulating station inventory
    #==================================================================
    def get_stations(self, client_name='IRIS', startdate=None, enddate=None,  startbefore=None, startafter=None, endbefore=None, endafter=None,\
            network=None, station=None, location=None, channel=None, includerestricted=False,\
            minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, latitude=None, longitude=None, minradius=None, maxradius=None):
        """Get station inventory from IRIS server
        =======================================================================================================
        Input Parameters:
        startdate, enddata  - start/end date for searching
        network             - Select one or more network codes.
                                Can be SEED network codes or data center defined codes.
                                    Multiple codes are comma-separated (e.g. "IU,TA").
        station             - Select one or more SEED station codes.
                                Multiple codes are comma-separated (e.g. "ANMO,PFO").
        location            - Select one or more SEED location identifiers.
                                Multiple identifiers are comma-separated (e.g. "00,01").
                                As a special case ?--? (two dashes) will be translated to a string of two space
                                characters to match blank location IDs.
        channel             - Select one or more SEED channel codes.
                                Multiple codes are comma-separated (e.g. "BHZ,HHZ").
        includerestricted   - default is False
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
        try:
            startbefore = obspy.core.utcdatetime.UTCDateTime(startbefore)
        except:
            startbefore = None
        try:
            startafter  = obspy.core.utcdatetime.UTCDateTime(startafter)
        except:
            startafter  = None
        try:
            endbefore   = obspy.core.utcdatetime.UTCDateTime(endbefore)
        except:
            endbefore   = None
        try:
            endafter    = obspy.core.utcdatetime.UTCDateTime(endafter)
        except:
            endafter    = None
        client          = Client(client_name)
        inv             = client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, startbefore=startbefore, startafter=startafter,\
                                endbefore=endbefore, endafter=endafter, channel=channel, minlatitude=minlatitude, maxlatitude=maxlatitude, \
                                minlongitude=minlongitude, maxlongitude=maxlongitude, latitude=latitude, longitude=longitude, minradius=minradius, \
                                    maxradius=maxradius, level='channel', includerestricted=includerestricted)
        self.add_stationxml(inv)
        self.update_inv_info()
        return
     
    def download_body_waveforms(self, outdir, fskip=False, client_name='IRIS', minDelta=30, maxDelta=150, channel_rank=['BH', 'HH'],\
            phase='P', startoffset=-30., endoffset=60.0, verbose=False, rotation=True, startdate=None, enddate=None):
        """Download body wave data from IRIS server
        ====================================================================================================================
        ::: input parameters :::
        outdir          - output directory
        fskip           - flag for downloa/overwrite
                            False   - overwrite
                            True    - skip upon existence
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channel_rank    - rank of channel types
        phase           - body wave phase to be downloaded, arrival time will be computed using taup
        start/endoffset - start and end offset for downloaded data
        rotation        - rotate the seismogram to RT or not
        =====================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        client          = Client(client_name)
        ievent          = 0
        Ntrace          = 0
        try:
            stime4down  = obspy.core.utcdatetime.UTCDateTime(startdate)
        except:
            stime4down  = obspy.UTCDateTime(0)
        try:
            etime4down  = obspy.core.utcdatetime.UTCDateTime(enddate)
        except:
            etime4down  = obspy.UTCDateTime()
        print('[%s] [DOWNLOAD BODY WAVE] Start downloading body wave data' %datetime.now().isoformat().split('.')[0])
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
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
            evstr           = '%s' %otime.isoformat()
            outfname        = outdir + '/' + evstr+'.mseed'
            logfname        = outdir + '/' + evstr+'.log'
            # check file existence
            if os.path.isfile(outfname):
                if fskip:
                    if os.path.isfile(logfname):
                        os.remove(logfname)
                        os.remove(outfname)
                    else:
                        continue
                else:
                    os.remove(outfname)
                    if os.path.isfile(logfname):
                        os.remove(logfname)
            elif os.path.isfile(logfname):
                try:
                    with open(logfname, 'r') as fid:
                        logflag     = fid.readline().split()[0][:4]
                    if logflag == 'DONE' and fskip:
                        continue
                except:
                    pass 
            # initialize log file
            with open(logfname, 'w') as fid:
                fid.writelines('DOWNLOADING\n')
            out_stream      = obspy.Stream()
            itrace          = 0
            for staid in self.waveforms.list():
                netcode, stacode    = staid.split('.')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos          = self.waveforms[staid].coordinates
                stla                = tmppos['latitude']
                stlo                = tmppos['longitude']
                elev                = tmppos['elevation_in_m']
                elev                = elev/1000.
                az, baz, dist       = geodist.inv(evlo, evla, stlo, stla)
                dist                = dist/1000.
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
                starttime           = otime + arrival_time + startoffset
                endtime             = otime + arrival_time + endoffset
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    location        = self.waveforms[staid].StationXML[0].stations[0].channels[0].location_code
                # determine type of channel
                channel_type        = None
                for tmpch_type in channel_rank:
                    channel         = '%sE,%sN,%sZ' %(tmpch_type, tmpch_type, tmpch_type)
                    try:
                        st          = client.get_waveforms(network=netcode, station=stacode, location=location, channel=channel,
                                            starttime=starttime, endtime=endtime, attach_response=True)
                        if len(st) >= 3:
                            channel_type= tmpch_type
                            break
                    except:
                        pass
                if channel_type is None:
                    if verbose:
                        print ('--- No data for:', staid)
                    continue
                pre_filt            = (0.04, 0.05, 20., 25.)
                st.detrend()
                try:
                    st.remove_response(pre_filt=pre_filt, taper_fraction=0.1)
                except ValueError:
                    print ('!!! ERROR with response removal for:', staid)
                    continue 
                if rotation:
                    try:
                        st.rotate('NE->RT', back_azimuth=baz)
                    except:
                        continue
                if verbose:
                    print ('--- Getting data for:', staid)
                # append stream
                out_stream  += st
                itrace      += 1
                Ntrace      += 1
            # save data to miniseed
            if itrace != 0:
                out_stream.write(outfname, format = 'mseed', encoding = 'FLOAT64')
                os.remove(logfname) # delete log file
            else:
                with open(logfname, 'w') as fid:
                    fid.writelines('DONE\n')
            print('[%s] [DOWNLOAD BODY WAVE] ' %datetime.now().isoformat().split('.')[0]+\
                  'Event ' + str(ievent)+': dowloaded %d traces' %itrace)
        print('[%s] [DOWNLOAD BODY WAVE] All done' %datetime.now().isoformat().split('.')[0] + ' %d events, %d traces' %(ievent, Ntrace))
        return
    
    def extract_tar_mseed(self, datadir, outdir, fskip = False, start_date = None, end_date = None, \
            rmresp = True, ninterp = 2, chanrank=['BH', 'HH'], channels='ENZ', rotate = True, pfx='LF_',\
            delete_tar = False, delete_extract = True, verbose = True, verbose2 = False):
        """load tarred mseed data
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        try:
            stime4load  = obspy.core.utcdatetime.UTCDateTime(start_date)
        except:
            stime4load  = obspy.UTCDateTime(0)
        try:
            etime4load  = obspy.core.utcdatetime.UTCDateTime(end_date)
        except:
            etime4load  = obspy.UTCDateTime()
        # frequencies for response removal 
        Nnodataev   = 0
        Nevent      = 0
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
            event_id        = event.resource_id.id.split('=')[-1]
            timestr         = otime.isoformat()
            if otime < stime4load or otime > etime4load:
                continue
            Nevent          += 1
            try:
                descrip     = event_descrip+', '+Mtype+' = '+str(magnitude)
            except:
                continue
            oyear           = otime.year
            omonth          = otime.month
            oday            = otime.day
            ohour           = otime.hour
            omin            = otime.minute
            osec            = otime.second
            label           = '%d_%s_%d_%d_%d_%d' %(oyear, monthdict[omonth], oday, ohour, omin, osec)
            tarwildcard     = datadir+'/'+pfx + label +'.*.tar.mseed'
            tarlst          = glob.glob(tarwildcard)
            if len(tarlst) == 0:
                print ('!!! NO DATA: %s %s' %(otime.isoformat(), descrip))
                Nnodataev  += 1
                continue
            elif len(tarlst) > 1:
                print ('!!! MORE DATA DATE: %s %s' %(otime.isoformat(), descrip))
            if verbose:
                print ('[%s] [EXTRACT_MSEED] loading: %s %s' %(datetime.now().isoformat().split('.')[0], \
                            otime.isoformat(), descrip))
            evstr           = '%s' %otime.isoformat()
            outfname        = outdir + '/' + evstr+'.mseed'
            if os.path.isdir(outfname) and fskip:
                continue
            event_stream    = obspy.Stream()
            # extract tar files
            tmptar          = tarfile.open(tarlst[0])
            tmptar.extractall(path = datadir)
            tmptar.close()
            eventdir        = datadir+'/'+(tarlst[0].split('/')[-1])[:-10]
            # loop over stations
            Ndata           = 0
            Nnodata         = 0
            for staid in self.waveforms.list():
                netcode     = staid.split('.')[0]
                stacode     = staid.split('.')[1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    staxml  = self.waveforms[staid].StationXML
                mseedfname  = eventdir + '/' + stacode+'.'+netcode+'.mseed'
                xmlfname    = eventdir + '/IRISDMC-' + stacode+'.'+netcode+'.xml'
                stla        = staxml[0][0].latitude
                stlo        = staxml[0][0].longitude
                # load data
                if not os.path.isfile(mseedfname):
                    if otime >= staxml[0][0].start_date and otime <= staxml[0][0].end_date:
                        if verbose2:
                            print ('*** NO DATA STATION: '+staid)
                        Nnodata     += 1
                    continue
                # load data
                st              = obspy.read(mseedfname)
                #=============================
                # get response information
                # rmresp = True, from XML
                #=============================
                if rmresp:
                    if not os.path.isfile(xmlfname):
                        print ('*** NO RESPXML FILE STATION: '+staid)
                        resp_inv = staxml.copy()
                        try:
                            for tr in st:
                                seed_id     = tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.location+'.'+tr.stats.channel
                                resp_inv.get_response(seed_id = seed_id, datetime = curtime)
                        except:
                            print ('*** NO RESP STATION: '+staid)
                            Nnodata     += 1
                            continue
                    else:
                        resp_inv = obspy.read_inventory(xmlfname)
                dist, az, baz   = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                dist            = dist/1000.
                # merge data, fill gaps
                try:
                    st.merge(method = 1, interpolation_samples = ninterp, fill_value = 'interpolate')
                except:
                    print ('*** NOT THE SAME SAMPLING RATE, STATION: '+staid)
                    Nnodata     += 1
                    continue
                # choose channel type
                chan_type   = None
                for tmpchtype in chanrank:
                    ich     = 0
                    for chan in channels:
                        if len(st.select(channel = tmpchtype + chan)) > 0:
                            ich += 1
                    if ich == len(channels):
                        chan_type   = tmpchtype
                        break
                if chan_type is None:
                    if verbose2:
                        print ('*** NO CHANNEL STATION: '+staid)
                    Nnodata     += 1
                    continue
                stream      = obspy.Stream()
                for chan in channels:
                    tmpst   = st.select(channel = chan_type + chan)
                    if len(tmpst) > 0 and verbose2:
                        print ('*** MORE THAN ONE LOCS STATION: '+staid)
                        Nvalid      = (tmpst[0].stats.npts)
                        outtr       = tmpst[0].copy()
                        for tmptr in tmpst:
                            tmp_n   = tmptr.stats.npts
                            if tmp_n > Nvalid:
                                Nvalid  = tmp_n
                                outtr   = tmptr.copy()
                        stream.append(outtr)
                    else:
                        stream.append(tmpst[0])
                if rmresp:
                    stream.detrend()
                    pre_filt    = (0.04, 0.05, 20., 25.)
                    try:
                        stream.remove_response(inventory = resp_inv, pre_filt = pre_filt, taper_fraction=0.1)
                    except:
                        print ('*** ERROR IN RESPONSE REMOVE STATION: '+staid)
                        Nnodata  += 1
                        continue
                if rotate:
                    try:
                        stream.rotate('NE->RT', back_azimuth = baz)
                    except:
                        print ('*** ERROR IN ROTATION STATION: '+staid)
                        Nnodata     += 1
                        continue
                event_stream    += stream
                Ndata           += 1
            if verbose:
                print ('[%s] [EXTRACT_MSEED] %d/%d (data/no_data) groups of traces extracted!'\
                       %(datetime.now().isoformat().split('.')[0], Ndata, Nnodata))
            # delete raw data
            if delete_extract:
                shutil.rmtree(eventdir)
            if delete_tar:
                os.remove(tarlst[0])
            if Ndata>0:
                event_stream.write(outfname, format = 'mseed', encoding = 'FLOAT64')
        # End loop over events
        print ('[%s] [EXTRACT_MSEED] Extracted %d/%d (events_with)data/total_events) events of data'\
               %(datetime.now().isoformat().split('.')[0], Nevent - Nnodataev, Nevent))
        return
    
    def extract_mass_mseed(self, datadir, outdir, rotate=True, chan_rank=['BH', 'HH'], fskip = True, delete_mseed = False,\
                ninterp = 2, channels = 'ZNE', outchannels = 'RTZ', startdate = None, enddate = None, phase = 'P', verbose=False):
        """extract massdownload mseed
        """
        Nnodataev       = 0
        Nevent          = 0
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        try:
            stime4load  = obspy.core.utcdatetime.UTCDateTime(startdate)
        except:
            stime4load  = obspy.UTCDateTime(0)
        try:
            etime4load  = obspy.core.utcdatetime.UTCDateTime(enddate)
        except:
            etime4load  = obspy.UTCDateTime()
        print('[%s] [EXTRACT_MASS_MSEED] Start extract mseed' %datetime.now().isoformat().split('.')[0])
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        for event in self.cat:
            event_id        = event.resource_id.id.split('=')[-1]
            pmag            = event.preferred_magnitude()
            magnitude       = pmag.mag
            Mtype           = pmag.magnitude_type
            event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin         = event.preferred_origin()
            otime           = porigin.time
            if otime < stime4load or otime > etime4load:
                continue
            Nevent          += 1
            try:
                print('[%s] [EXTRACT_MASS_MSEED] ' %datetime.now().isoformat().split('.')[0] + \
                            'Event ' + str(Nevent)+': '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
            except:
                print (otime)
                print (magnitude)
                # continue
            evlo            = porigin.longitude
            evla            = porigin.latitude
            oyear           = otime.year
            omonth          = otime.month
            oday            = otime.day
            ohour           = otime.hour
            omin            = otime.minute
            osec            = otime.second
            label           = '%d_%s_%d_%d_%d_%d' %(oyear, monthdict[omonth], oday, ohour, omin, osec)
            event_dir       = datadir + '/' +label
            if not os.path.isdir(event_dir):
                print ('!!! NO DATA!')
                Nnodataev   += 1
                continue
            evstr           = '%s' %otime.isoformat()
            outfname        = outdir + '/' + evstr+'.mseed'
            if os.path.isdir(outfname) and fskip:
                print ('!!! SKIP UPON DATA EXITENCE!')
            # loop over stations
            Ndata           = 0
            Nnodata         = 0
            event_stream    = obspy.Stream()
            for staid in self.waveforms.list():
                netcode     = staid.split('.')[0]
                stacode     = staid.split('.')[1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    staxml  = self.waveforms[staid].StationXML
                stla        = staxml[0][0].latitude
                stlo        = staxml[0][0].longitude
                # determine type of channels
                channel_type= None
                for tmpchtype in chan_rank:
                    ich     = 0
                    for chan in channels:
                        mseedpattern    = event_dir + '/%s.%s.*%s%s*.mseed' %(netcode, stacode, tmpchtype, chan)
                        if len(glob.glob(mseedpattern)) == 0:
                            break
                        ich += 1
                    if ich == len(channels):
                        channel_type= tmpchtype
                        location    = glob.glob(mseedpattern)[0].split('%s.%s' \
                                        %(netcode, stacode))[1].split('.')[1]
                        tmp_str     = glob.glob(mseedpattern)[0].split('%s.%s' \
                                        %(netcode, stacode))[1].split('.')[2].split('__')
                        time_label  = tmp_str[1]+'__'+tmp_str[2]
                        break
                if channel_type is None:
                    if otime >= staxml[0][0].creation_date and otime <= staxml[0][0].end_date:
                        if verbose:
                            print ('*** NO DATA STATION: '+staid)
                        Nnodata     += 1
                    continue
                # load data
                st                  = obspy.Stream()
                skip_this_station   = False
                for chan in channels:
                    mseedfname  = event_dir + '/%s.%s.%s.%s%s__%s.mseed' %(netcode, stacode, location, channel_type, chan, time_label)
                    try:
                        st      +=obspy.read(mseedfname)
                    except Exception:
                        skip_this_station   = True
                        break
                    if delete_mseed:
                        os.remove(mseedfname)
                if skip_this_station:
                    print ('*** ERROR LOADING DATA STATION: '+staid)
                    Nnodata     += 1
                    continue
                #=============================
                # get response information
                #=============================
                resp_from_download  = False
                xmlfname            = event_dir + '/%s/%s.xml' %(netcode, stacode)
                if not os.path.isfile(xmlfname):
                    print ('*** NO RESPXML FILE STATION: '+staid)
                    resp_inv    = staxml.copy()
                    try:
                        for tr in st:
                            resp_inv.get_response(seed_id = tr.id, datetime = otime)
                        resp_from_download  = True
                    except:
                        print ('*** NO RESP STATION: '+staid)
                        Nnodata     += 1
                        continue
                else:
                    resp_inv = obspy.read_inventory(xmlfname)
                dist, az, baz   = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                dist            = dist/1000.
                # merge data, fill gaps
                try:
                    st.merge(method = 1, interpolation_samples = ninterp, fill_value = 'interpolate')
                except:
                    print ('*** NOT THE SAME SAMPLING RATE, STATION: '+staid)
                    Nnodata     += 1
                    continue
                # remove response
                st.detrend()
                pre_filt        = (0.04, 0.05, 20., 25.)
                
                try:
                    st.remove_response(inventory = resp_inv, pre_filt = pre_filt)
                except:
                    if not resp_from_download:
                        resp_inv    = staxml.copy()
                        try:
                            for tr in st:
                                resp_inv.get_response(seed_id = tr.id, datetime = otime)
                            resp_from_download  = True
                            st.remove_response(inventory = resp_inv, pre_filt = pre_filt)
                        except:
                            print ('*** ERROR IN RESPONSE REMOVE STATION: '+staid)
                            Nnodata  += 1
                            continue
                if rotate:
                    try:
                        st.rotate('NE->RT', back_azimuth=baz)
                    except ValueError:
                        stime   = st[0].stats.starttime
                        etime   = st[0].stats.endtime
                        for tr in st:
                            if stime < tr.stats.starttime:
                                stime   = tr.stats.starttime
                            if etime > tr.stats.endtime:
                                etime   = tr.stats.endtime
                        st.trim(starttime = stime, endtime = etime)
                        try:
                            st.rotate('NE->RT', back_azimuth=baz)
                        except:
                            print ('*** ERROR IN ROTATION STATION: '+staid)
                            Nnodata  += 1
                            continue
                event_stream+= st
                Ndata       += 1
            if Ndata>0:
                event_stream.write(outfname, format = 'mseed', encoding = 'FLOAT64')
            print ('[%s] [EXTRACT_MASS_MSEED] %d/%d (data/no_data) groups of traces extracted!'\
                   %(datetime.now().isoformat().split('.')[0], Ndata, Nnodata))
        # End loop over events
        print ('[%s] [EXTRACT_MASS_MSEED] Extracted %d/%d (events_with)data/total_events) events of data'\
               %(datetime.now().isoformat().split('.')[0], Nevent - Nnodataev, Nevent))
        return
    
    def load_body_wave(self, datadir, startdate=None, enddate=None, phase = 'P'):
        """Load body wave data
        """
        ievent          = 0
        Ntrace          = 0
        try:
            stime4load  = obspy.core.utcdatetime.UTCDateTime(startdate)
        except:
            stime4load  = obspy.UTCDateTime(0)
        try:
            etime4load  = obspy.core.utcdatetime.UTCDateTime(enddate)
        except:
            etime4load  = obspy.UTCDateTime()
        print('[%s] [LOAD BODY WAVE] Start loading body wave data' %datetime.now().isoformat().split('.')[0])
        try:
            print (self.cat)
        except AttributeError:
            self.copy_catalog()
        for event in self.cat:
            event_id        = event.resource_id.id.split('=')[-1]
            pmag            = event.preferred_magnitude()
            magnitude       = pmag.mag
            Mtype           = pmag.magnitude_type
            event_descrip   = event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            porigin         = event.preferred_origin()
            otime           = porigin.time
            if otime < stime4load or otime > etime4load:
                continue
            ievent          += 1
            try:
                print('[%s] [LOAD BODY WAVE] ' %datetime.now().isoformat().split('.')[0] + \
                            'Event ' + str(ievent)+': '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude))
            except:
                print (otime)
                print (magnitude)
                # continue
            evlo            = porigin.longitude
            evla            = porigin.latitude
            evstr           = '%s' %otime.isoformat()
            infname         = datadir + '/' + evstr+'.mseed'
            oyear           = otime.year
            omonth          = otime.month
            oday            = otime.day
            ohour           = otime.hour
            omin            = otime.minute
            osec            = otime.second
            label           = '%d_%d_%d_%d_%d_%d' %(oyear, omonth, oday, ohour, omin, osec)
            if not os.path.isfile(infname):
                print ('!!! NO DATA!')
                continue
            stream          = obspy.read(infname)
            tag             = 'body_'+label
            # adding waveforms
            try:
                self.add_waveforms(stream, event_id = event_id, tag = tag, labels=phase)
            except Exception:
                print ('!!! ERROR DATA!')
                continue
            itrace          = len(stream)
            Nsta            = itrace/3
            Ntrace          += itrace
            print('[%s] [LOAD BODY WAVE] ' %datetime.now().isoformat().split('.')[0]+\
                  'Event ' + str(ievent)+': loaded %d traces, %d stations' %(itrace, Nsta))
        print('[%s] [LOAD BODY WAVE] All done' %datetime.now().isoformat().split('.')[0] + ' %d events, %d traces' %(ievent, Ntrace))
        return
    
    def plot_ref(self, network, station, phase = 'P', datatype = 'RefRHSdata', outdir = None, avgflag = True):
        """plot receiver function
        ====================================================================================================================
        ::: input parameters :::
        network, station    - specify station
        phase               - phase, default = 'P'
        datatype            - datatype, default = 'RefRHS', harmonic striped radial receiver function
        =====================================================================================================================
        """
        obsHSst         = _rf_funcs.HSStream()
        diffHSst        = _rf_funcs.HSStream()
        repHSst         = _rf_funcs.HSStream()
        rep0HSst        = _rf_funcs.HSStream()
        rep1HSst        = _rf_funcs.HSStream()
        rep2HSst        = _rf_funcs.HSStream()
        subgroup        = self.auxiliary_data[datatype][network+'_'+station+'_'+phase]
        if avgflag:
            avgHSst     = _rf_funcs.HSStream()
            subgroup2   = self.auxiliary_data['RefRHSavgdata'][network+'_'+station+'_'+phase]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmppos      = self.waveforms[network+'.'+station].coordinates
        stla            = tmppos['latitude']
        stlo            = tmppos['longitude']
        elev            = tmppos['elevation_in_m']
        for label in subgroup.obs.list():
            ref_header  = subgroup['obs'][label].parameters
            dt          = ref_header['delta']
            baz         = ref_header['baz']
            eventT      = ref_header['otime']
            obsArr      = subgroup['obs'][label].data[()]
            starttime   = obspy.core.utcdatetime.UTCDateTime(eventT) + ref_header['arrival'] - ref_header['tbeg'] + 30.
            obsHSst.get_trace(network=network, station=station, indata=obsArr, baz=baz, dt=dt, starttime=starttime)
            try:
                diffArr     = subgroup['diff'][label].data[()]
                diffHSst.get_trace(network=network, station=station, indata=diffArr, baz=baz, dt=dt, starttime=starttime)
                
                repArr      = subgroup['rep'][label].data[()]
                repHSst.get_trace(network=network, station=station, indata=repArr, baz=baz, dt=dt, starttime=starttime)
                
                rep0Arr     = subgroup['rep0'][label].data[()]
                rep0HSst.get_trace(network=network, station=station, indata=rep0Arr, baz=baz, dt=dt, starttime=starttime)
                
                rep1Arr     =subgroup['rep1'][label].data[()]
                rep1HSst.get_trace(network=network, station=station, indata=rep1Arr, baz=baz, dt=dt, starttime=starttime)
                
                rep2Arr     = subgroup['rep2'][label].data[()]
                rep2HSst.get_trace(network=network, station=station, indata=rep2Arr, baz=baz, dt=dt, starttime=starttime)
            except KeyError:
                print('No predicted data for plotting')
                return
            if avgflag:
                avgdat      = subgroup2['data'].data[()]
                avgHSst.get_trace(network=network, station=station, indata=avgdat, baz=baz, dt=dt, starttime=starttime)
        if avgflag:
            self.hsdbase    = _rf_funcs.hsdatabase(obsST=obsHSst, diffST=diffHSst, repST=repHSst,\
                            repST0=rep0HSst, repST1=rep1HSst, repST2=rep2HSst, avgST = avgHSst)
        else:
            self.hsdbase    = _rf_funcs.hsdatabase(obsST=obsHSst, diffST=diffHSst, repST=repHSst,\
                            repST0=rep0HSst, repST1=rep1HSst, repST2=rep2HSst)
        if outdir is None:
            self.hsdbase.plot(stacode=network+'.'+station, longitude=stlo, latitude=stla, avgflag = avgflag)
        else:
            self.hsdbase.plot(stacode=network+'.'+station, longitude=stlo, latitude=stla, outdir = outdir, avgflag = avgflag)
        return
    
    def plot_all_ref(self, outdir, phase = 'P', outtxt = None, avgflag = True):
        """
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        print ('Plotting ref results !')
        if outtxt is not None:
            fid     = open(outtxt, 'w')
        for staid in self.waveforms.list():
            netcode, stacode    = staid.split('.')
            try:
                Ndata           = len(self.auxiliary_data.RefRHSdata[netcode+'_'+stacode+'_'+phase]['obs'].list())
            except KeyError:
                Ndata           = 0
            print (staid+': '+str(Ndata))
            if outtxt is not None:
                fid.writelines(staid+'  '+str(Ndata)+'\n')
            if Ndata == 0:
                continue
            self.plot_ref(network=netcode, station=stacode, phase=phase, outdir=outdir, avgflag = avgflag)
        if outtxt is not None:
            fid.close()
    
    
    
    