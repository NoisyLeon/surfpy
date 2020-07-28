# -*- coding: utf-8 -*-
"""
Base ASDF for receiver function analysis


:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.clients.fdsn.client import Client
from obspy.taup import TauPyModel
import warnings
from datetime import datetime
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from pyproj import Geod

geodist         = Geod(ellps='WGS84')
taupmodel       = TauPyModel(model="iasp91")

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
    
    
    def download_body_waveforms(self, outdir, fskip=False, client_name='IRIS', minDelta=30, maxDelta=150, channel='BHE,BHN,BHZ', phase='P',
                        startoffset=-30., endoffset=60.0, verbose=False, rotation=True, startdate=None, enddate=None):
        """Download body wave data from IRIS server
        ====================================================================================================================
        ::: input parameters :::
        outdir          - output directory
        fskip           - flag for downloa/overwrite
                            False   - overwrite
                            True    - skip upon existence
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channel         - Channel code, e.g. 'BHZ'.
                            Last character (i.e. component) can be a wildcard (??? or ?*?) to fetch Z, N and E component.
        phase           - body wave phase to be downloaded, arrival time will be computed using taup
        start/endoffset - start and end offset for downloaded data
        vmin, vmax      - minimum/maximum velocity for surface wave window
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
            evdp            = porigin.depth/1000.
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
                with open(logfname, 'r') as fid:
                    logflag     = fid.readline().split()[0][:4]
                # # # print (logflag)
                if logflag == 'DONE' and fskip:
                    continue
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
                try:
                    st              = client.get_waveforms(network=netcode, station=stacode, location=location, channel=channel,
                                        starttime=starttime, endtime=endtime, attach_response=True)
                except:
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
    # 
    # def get_obspy_trace(self, network, station, evnumb, datatype='body'):
    #     """ Get obspy trace data from ASDF
    #     ====================================================================================================================
    #     input parameters:
    #     network, station    - specify station
    #     evnumb              - event id
    #     datatype            - data type ('body' - body wave, 'surf' - surface wave)
    #     =====================================================================================================================
    #     """
    #     event               = self.events[evnumb-1]
    #     tag                 = datatype+'_ev_%05d' %evnumb
    #     st                  = self.waveforms[network+'.'+station][tag]
    #     stla, elev, stlo    = self.waveforms[network+'.'+station].coordinates.values()
    #     evlo                = event.origins[0].longitude
    #     evla                = event.origins[0].latitude
    #     evdp                = event.origins[0].depth
    #     for tr in st:
    #         tr.stats.sac            = obspy.core.util.attribdict.AttribDict()
    #         tr.stats.sac['evlo']    = evlo
    #         tr.stats.sac['evla']    = evla
    #         tr.stats.sac['evdp']    = evdp
    #         tr.stats.sac['stlo']    = stlo
    #         tr.stats.sac['stla']    = stla    
    #     return st
    
    