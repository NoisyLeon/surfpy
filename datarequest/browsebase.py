# -*- coding: utf-8 -*-
"""
ASDF for seismic station browser
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import pyasdf
import warnings
import obspy
from obspy.clients.fdsn.client import Client
import obspy.clients.iris
import matplotlib.pyplot as plt
import copy
import numpy as np
import warnings
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import shapefile
from pyproj import Geod
geodist     = Geod(ellps='WGS84')

monthdict       = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL',\
                   8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

# base_url_list   = ['BGR', 'EMSC', 'ETH', 'GEONET', 'GFZ', 'ICGC', 'INGV', 'IPGP',\
#     'IRIS', 'ISC', 'KNMI', 'KOERI', 'LMU', 'NCEDC', 'NIEP', 'NOA', 'ODC', 'ORFEUS',\
#     'RASPISHAKE', 'RESIF', 'SCEDC', 'TEXNET', 'USGS', 'USP']

# base_url_list   = ['BGR', 'ETH', 'GFZ', 'ICGC', 'INGV', 'IPGP','IRIS', 'KNMI',  'LMU',\
#                    'ODC', 'ORFEUS', 'RESIF']
# base_url_list   = ['GFZ', 'ICGC', 'INGV', 'IPGP','IRIS', 'ODC', 'ORFEUS', 'RESIF', 'KOERI', 'NOA']

# base_url_list   = ['IRIS', 'ORFEUS', 'BGR', 'ETH', 'IPGP', 'GFZ', 'RESIF']

base_url_list   = ['BGR', 'ETH', 'GFZ', 'ICGC', 'INGV', 'IPGP',\
    'IRIS', 'KNMI', 'KOERI', 'LMU', 'NCEDC', 'NIEP', 'NOA', 'ODC', 'ORFEUS',\
    'RASPISHAKE', 'RESIF', 'SCEDC', 'TEXNET', 'USP']

class baseASDF(pyasdf.ASDFDataSet):
    
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
        # station inventory; start/end date of the stations
        self.inv        = obspy.Inventory()
        self.start_date = obspy.UTCDateTime('2599-01-01')
        self.end_date   = obspy.UTCDateTime('1900-01-01')
        self.update_inv_info()
        return
    
    def get_limits_lonlat(self, lontype = 0):
        """get the geographical limits of the stations
        """
        staLst      = self.waveforms.list()
        minlat      = 90.
        maxlat      = -90.
        if lontype == 0:
            minlon      = 180.
            maxlon      = -180.
        else:
            minlon      = 360.
            maxlon      = 0.
        for staid in staLst:
            tmppos  = self.waveforms[staid].coordinates
            lat     = tmppos['latitude']
            lon     = tmppos['longitude']
            elv     = tmppos['elevation_in_m']
            if lontype == 0:
                if lon > 180.:
                    lon     -= 360.
            else:
                if lon < 0.:
                    lon     += 360.
            minlat  = min(lat, minlat)
            maxlat  = max(lat, maxlat)
            minlon  = min(lon, minlon)
            maxlon  = max(lon, maxlon)
        print ('latitude range: ', minlat, '-', maxlat, 'longitude range:', minlon, '-', maxlon)
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
    
    def write_inv(self, outfname, format='stationxml'):
        self.inv.write(outfname, format=format)
        return
    
    def read_inv(self, infname):
        inv    = obspy.core.inventory.inventory.read_inventory(infname)
        self.add_stationxml(inv)
        self.update_inv_info()
        return
    
    def get_stations(self, client_name='IRIS', startdate=None, enddate=None, startbefore=None, startafter=None, endbefore=None, endafter=None,\
            network_reject = None, network=None, station=None, location=None, channel=None, includerestricted=False,\
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
        if client_name is None:
            inv         = obspy.Inventory()
            for base_url in base_url_list:
                # print (base_url)
                client  = Client(base_url)
                try:
                    inv += client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, startbefore=startbefore, startafter=startafter,\
                            endbefore=endbefore, endafter=endafter, channel=channel, minlatitude=minlatitude, maxlatitude=maxlatitude, \
                            minlongitude=minlongitude, maxlongitude=maxlongitude, latitude=latitude, longitude=longitude, minradius=minradius, \
                            maxradius=maxradius, level='channel', includerestricted=includerestricted)
                except:
                    pass
                # # # self.add_stationxml(inv)
        else:
            client      = Client(client_name)
            inv         = client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, startbefore=startbefore, startafter=startafter,\
                                endbefore=endbefore, endafter=endafter, channel=channel, minlatitude=minlatitude, maxlatitude=maxlatitude, \
                                minlongitude=minlongitude, maxlongitude=maxlongitude, latitude=latitude, longitude=longitude, minradius=minradius, \
                                    maxradius=maxradius, level='channel', includerestricted=includerestricted)
        if network_reject is not None:
            inv = inv.remove(network = network_reject)
    
        # inv2         = obspy.Inventory()
        # inv2 += inv.select(station = 'HD21')
        # inv2 += inv.select(station = 'HD35')
        # self.add_stationxml(inv2)
        # # # return inv
        self.add_stationxml(inv)
        self.update_inv_info()
        return
    
    def get_events(self, startdate, enddate, base_url='IRIS', add2dbase=True, gcmt=False, Mmin=5.5, Mmax=None,
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
            client  = Client(base_url)
            try:
                catISC      = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=Mmin, maxmagnitude=Mmax, catalog='ISC',
                                minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                                latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, mindepth=mindepth,
                                maxdepth=maxdepth, magnitudetype=magnitudetype)
                endtimeISC  = catISC[0].origins[0].time
            except:
                catISC      = obspy.core.event.Catalog()
                endtimeISC  = starttime
            if endtime>endtimeISC:
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
    
    def copy_catalog(self):
        print('Copying catalog from ASDF to memory')
        self.cat    = self.events
        return
    
    def read_sta_lst(self, infname, client_name='IRIS', startdate=None, enddate=None,  startbefore=None, startafter=None, endbefore=None, endafter=None,
            location=None, channel=None, includerestricted=False, minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, \
            latitude=None, longitude=None, minradius=None, maxradius=None):
        """read station list from txt file
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
        init_flag       = True
        with open(infname, 'rb') as fio:
            for line in fio.readlines():
                network = line.split()[1]
                station = line.split()[2]
                if network == 'NET':
                    continue
                # print network, station
                if init_flag:
                    try:
                        inv     = client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, startbefore=startbefore, startafter=startafter,\
                                    endbefore=endbefore, endafter=endafter, channel=channel, minlatitude=minlatitude, maxlatitude=maxlatitude, \
                                        minlongitude=minlongitude, maxlongitude=maxlongitude, latitude=latitude, longitude=longitude, minradius=minradius, \
                                            maxradius=maxradius, level='channel', includerestricted=includerestricted)
                    except:
                        print ('No station inv: ', line)
                        continue
                    init_flag   = False
                    continue
                try:
                    inv     += client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, startbefore=startbefore, startafter=startafter,\
                                endbefore=endbefore, endafter=endafter, channel=channel, minlatitude=minlatitude, maxlatitude=maxlatitude, \
                                    minlongitude=minlongitude, maxlongitude=maxlongitude, latitude=latitude, longitude=longitude, minradius=minradius, \
                                        maxradius=maxradius, level='channel', includerestricted=includerestricted)
                except:
                    print ('No station inv: ', line)
                    continue
        self.add_stationxml(inv)
        self.update_inv_info()
        return
    
    def min_dist(self, netcodelist=[]):
        """Get an array corresponding to the nearest station distance
        netcodelist     - list of network codes
        """
        inv         = self.inv
        inet        = 0
        stalons     = np.array([])
        stalats     = np.array([])
        for network in inv:
            if len(netcodelist) != 0:
                if network.code not in netcodelist:
                    continue
            inet        += 1
            for station in network:
                stalons         = np.append(stalons, station.longitude)
                stalats         = np.append(stalats, station.latitude)
        Nsta        = stalons.size
        g           = Geod(ellps='WGS84')
        distarr     = np.zeros(Nsta)
        for ista in range(Nsta):
            lon     = stalons[ista]
            lat     = stalats[ista]
            ind     = (abs(stalons - lon)>0.1) + (abs(stalats - lat)>0.1)
            tlons   = stalons[ind]
            tlats   = stalats[ind]
            L       = tlons.size
            clonArr         = np.ones(L, dtype=float)*lon
            clatArr         = np.ones(L, dtype=float)*lat
            az, baz, dist   = g.inv(clonArr, clatArr, tlons, tlats)
            distarr[ista]   = dist.min()/1000.
        return distarr

    def check_access(self):
        for net in self.inv:
            for sta in net:
                if sta.restricted_status is not 'open':
                    print ('!!! Restriced stations: %s' %sta)
        return
    
    def count_data(self, daylist  = np.array([1, 30, 60, 90, 180, 360, 720]), recompute=False):
        """count the number of available xcorr traces
        """
        daylist         = np.asarray(daylist)
        # check if data counts already exists
        try:
            dset_pair   = self.auxiliary_data.DataInfo['data_pairs']
            dset_sta    = self.auxiliary_data.DataInfo['data_stations']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data1   = dset_pair.data.value
                data2   = dset_sta.data.value
            tmpday1 = data1[:, 0]
            tmpday2 = data2[:, 0]
            prcount = data1[:, 1]
            stacount= data2[:, 1]
            if (np.alltrue(tmpday1 == daylist) and np.alltrue(tmpday2 == daylist)):
                for i in range(daylist.size):
                    print ('--- Operation days >= %5d:           %8d stations ' %(daylist[i], stacount[i]))
                for i in range(daylist.size):
                    print ('--- Overlap days >= %5d:                %8d pairs ' %(daylist[i], prcount[i]))
                if not recompute:
                    return
        except:
            pass
        print ('*** Recomputing data counts!')
        prcount         = np.zeros(daylist.size)
        stacount        = np.zeros(daylist.size)
        staLst          = self.waveforms.list()
        for staid1 in staLst:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st_date1        = self.waveforms[staid1].StationXML.networks[0].stations[0].start_date
                ed_date1        = self.waveforms[staid1].StationXML.networks[0].stations[0].end_date
            if ed_date1 is None:
                ed_date1    = obspy.UTCDateTime()
            Ndeployday      = int((ed_date1 - st_date1)/86400)
            stacount        += (Ndeployday >= daylist)
            for staid2 in staLst:
                if staid1 >= staid2:
                    continue
                # print (staid1)
                # print (staid2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    st_date2    = self.waveforms[staid2].StationXML.networks[0].stations[0].start_date
                    ed_date2    = self.waveforms[staid2].StationXML.networks[0].stations[0].end_date
                if ed_date2 is None:
                    ed_date2    = obspy.UTCDateTime()
                if st_date2 >= ed_date1 or st_date1 >= ed_date2:
                    Noverlapday = 0
                elif st_date1 <= st_date2 and ed_date2 <= ed_date1:
                    Noverlapday = int((ed_date2 - st_date2)/86400)
                elif st_date2 <= st_date1 and ed_date1 <= ed_date2:
                    Noverlapday = int((ed_date1 - st_date1)/86400)
                elif st_date2 >= st_date1 and ed_date2 >= ed_date1:
                    Noverlapday = int((ed_date1 - st_date2)/86400)
                elif st_date1 >= st_date2 and ed_date1 >= ed_date2:
                    Noverlapday = int((ed_date2 - st_date1)/86400)
                else:
                    print (st_date1)
                    print (ed_date1)
                    print (st_date2)
                    print (ed_date2)
                    raise ValueError('ERROR')
                
                prcount         += (Noverlapday >= daylist)
        data        = np.zeros((prcount.size, 2), dtype = np.int32)
        data[:, 0]  = daylist[:]
        data[:, 1]  = prcount[:]
        try:
            del self.auxiliary_data.DataInfo['data_pairs']
        except:
            pass
        self.add_auxiliary_data(data=data, data_type='DataInfo', path='data_pairs', parameters={})
        data[:, 1]  = stacount[:]
        try:
            del self.auxiliary_data.DataInfo['data_stations']
        except:
            pass
        self.add_auxiliary_data(data=data, data_type='DataInfo', path='data_stations', parameters={})
        for i in range(daylist.size):
            print ('--- Operation days >= %5d:           %8d stations ' %(daylist[i], stacount[i]))
        for i in range(daylist.size):
            print ('--- Overlap days >= %5d:                %8d pairs ' %(daylist[i], prcount[i]))
        return
        
    def write_txt(self, outfname):
        with open(outfname, 'w') as fid:
            for staid in self.waveforms.list():
                temp    = staid.split('.')
                network = temp[0]
                stacode = temp[1]
                fid.writelines(stacode+' '+network+'\n')
        return
    
    def write_seedprep_txt(self, outfname, start_date='19700101', end_date='25990101' ):
        start_date  = obspy.UTCDateTime(start_date)
        end_date    = obspy.UTCDateTime(end_date)
        with open(outfname, 'w') as fid:
            for staid in self.waveforms.list():
                st_date = self.waveforms[staid].StationXML.networks[0].stations[0].start_date
                ed_date = self.waveforms[staid].StationXML.networks[0].stations[0].end_date
                if st_date > end_date or ed_date < start_date:
                    continue
                temp    = staid.split('.')
                network = temp[0]
                stacode = temp[1]
                stlo    = self.waveforms[staid].StationXML.networks[0].stations[0].longitude
                stla    = self.waveforms[staid].StationXML.networks[0].stations[0].latitude
                fid.writelines(stacode+' '+str(stlo)+' '+str(stla)+' '+network+'\n')
        return
    
    def _get_basemap(self, projection='lambert', resolution='i', blon=0., blat=0.):
        """Get basemap for plotting results
        """
        fig=plt.figure(num=None, figsize=(12, 12), dpi=100, facecolor='w', edgecolor='k')
        try:
            minlon  = self.minlon-blon
            maxlon  = self.maxlon+blon
            minlat  = self.minlat-blat
            maxlat  = self.maxlat+blat
        except AttributeError:
            self.get_limits_lonlat()
            minlon  = self.minlon-blon; maxlon=self.maxlon+blon; minlat=self.minlat-blat; maxlat=self.maxlat+blat
        
       
        
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection == 'merc':
            minlon=-10.
            maxlon=5.
            minlat=31.
            maxlat=45.
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                      urcrnrlon=maxlon, lat_ts=0, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,1,1,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,1,1,1], fontsize=15)
        elif projection == 'global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
        elif projection == 'regional_ortho':
            mapfactor = 2.
            m1      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                        llcrnrx = 0., llcrnry = 0., urcrnrx = m1.urcrnrx/mapfactor, urcrnry = m1.urcrnry/2.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            minlon=-165.+360.
            maxlon=-147+360.
            minlat=51.
            maxlat=62.
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m

            m       = Basemap(width=1100000, height=1100000, rsphere=(6378137.00,6356752.3142), resolution='h', projection='lcc',\
                        lat_1 = minlat, lat_2 = maxlat, lon_0 = lon_centre, lat_0 = lat_centre + 0.5)
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1, dashes=[2,2], labels=[1,1,1,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,5.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        elif projection=='lambert2':
            minlon=93.+360.
            maxlon=105.+360.
            minlat=44.
            maxlat=52.
            
            lat_centre  = (maxlat+minlat)/2.0
            lon_centre  = (maxlon+minlon)/2.0
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m

            m       = Basemap(width=900000, height=900000, rsphere=(6378137.00,6356752.3142), resolution='h', projection='lcc',\
                        lat_1 = minlat, lat_2 = maxlat, lon_0 = lon_centre, lat_0 = lat_centre + 0.25)
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1, dashes=[2,2], labels=[1,1,1,1], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,5.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        elif projection == 'ortho':
            m       = Basemap(projection = 'ortho', lon_0 = -170., lat_0 = 40., resolution='l')
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=1,  fontsize=20)
            m.drawmeridians(np.arange(-180.0,180.0,10.0),  linewidth=1)
        elif projection == 'aeqd':
            width = 10000000
            m = Basemap(width = width/1.6,height=width/2.2,projection='aeqd', resolution='h',
                 lon_0 = -153., lat_0 = 62.)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1., dashes=[2,2], labels=[1,1,0,0], fontsize = 15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,0,1], fontsize = 15)
        elif projection == 'aeqd2':
            width = 10000000
            m = Basemap(width = width/1.85,height=width/2.6,projection='aeqd', resolution='h',
                 lon_0 = 98.2, lat_0 = 41.)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1., dashes=[2,2], labels=[1,1,0,0], fontsize = 15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,0,1], fontsize = 15)
        elif projection == 'aeqd3':
            width = 8000000
            m = Basemap(width = width/2.7, height=width/2.6,projection='aeqd', resolution='h',
                 lon_0 = -1.5, lat_0 = 42.2)
            # m = Basemap(height=width/2.6,projection='aeqd', resolution='h',
            #      lon_0 = 0., lat_0 = 42.2)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1., dashes=[2,2], labels=[1,1,0,0], fontsize = 20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1., dashes=[2,2], labels=[0,0,1,1], fontsize = 20)
            
        # m.drawcoastlines(linewidth=0.2)
        try:
            coasts = m.drawcoastlines(zorder=1,color= 'k',linewidth=0.000)
            # Exact the paths from coasts
            coasts_paths = coasts.get_paths()
            poly_stop = 50
            for ipoly in range(len(coasts_paths)):
                print (ipoly)
                if ipoly > poly_stop:
                    break
                r = coasts_paths[ipoly]
                # Convert into lon/lat vertices
                polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in
                                    r.iter_segments(simplify=False)]
                px = [polygon_vertices[i][0] for i in range(len(polygon_vertices))]
                py = [polygon_vertices[i][1] for i in range(len(polygon_vertices))]
                
                m.plot(px,py,'k-',linewidth=1., zorder=1)
        except:
            pass
        
        # m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
        
        # m.drawcountries(linewidth=1.)
        return m
    
    def plot_stations(self, projection='lambert', showfig=True, blon=.5, blat=0.5, plotetopo=False):
        """Plot station map
        ==============================================================================
        Input Parameters:
        projection      - type of geographical projection
        geopolygons     - geological polygons for plotting
        blon, blat      - extending boundaries in longitude/latitude
        showfig         - show figure or not
        ==============================================================================
        """
        staLst  = self.waveforms.list()
        stalons = np.array([])
        stalats = np.array([])
        sinlons = np.array([])
        sinlats = np.array([])
        sxolons = np.array([])
        sxolats = np.array([])
        
        exlons = np.array([])
        exlats = np.array([])
        ex2lons = np.array([])
        ex2lats = np.array([])
        ex3lons = np.array([])
        ex3lats = np.array([])
        minlon=-165.
        maxlon=-147
        minlat=51.
        maxlat=62.
        exclude_list = [ 'XO.EP23', 'XO.LT02', 'XO.LT08', 'XO.LT11', \
                  'XO.KT06', 'XO.KT09']
        exclude_list2 = [ 'XO.LA22', 'XO.LA27', 'XO.LA31', 'XO.LD24', \
                  'XO.LD42', 'XO.LD43', 'XO.LT19']
        exclude_list3 = ['XO.WD61']
        for staid in staLst:
            tmppos          = self.waveforms[staid].coordinates
            tmppos  = self.waveforms[staid].coordinates
            lat     = tmppos['latitude']
            lon     = tmppos['longitude']
            evz     = tmppos['elevation_in_m']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inv     = self.waveforms[staid].StationXML
            if staid in exclude_list:
                exlons         = np.append(exlons, lon)
                exlats         = np.append(exlats, lat)
                continue
            if staid in exclude_list2:
                print (staid)
                ex2lons         = np.append(ex2lons, lon)
                ex2lats         = np.append(ex2lats, lat)
                continue
            if staid in exclude_list3:
                print (staid)
                ex3lons         = np.append(ex3lons, lon)
                ex3lats         = np.append(ex3lats, lat)
                continue
            # minlon=-165.+360., maxlon=-147+360., minlat=51., maxlat=62.
            if (lon > minlon and lon < maxlon and lat > minlat and lat < maxlat) and inv[0].code != 'XO':
            # if inv[0].code == 'XO':
                sinlons         = np.append(sinlons, lon)
                sinlats         = np.append(sinlats, lat)
                continue
            elif inv[0].code == 'XO':
                sxolons         = np.append(sxolons, lon)
                sxolats         = np.append(sxolats, lat)
                continue
            stalons         = np.append(stalons, lon)
            stalats         = np.append(stalats, lat)
        m                   = self._get_basemap(projection=projection, blon=blon, blat=blat)
        if not plotetopo:
            m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
        
        m.drawcountries(linewidth=1.)
        if plotetopo:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            import pycpt
            etopodata   = Dataset('/home/lili/gebco_aacse.nc')
            etopo       = (etopodata.variables['elevation'][:]).data
            lons        = (etopodata.variables['lon'][:]).data
            lons[lons>180.] = lons[lons>180.] - 360.
            lats        = (etopodata.variables['lat'][:]).data

            ind_lon     = (lons <= -140.)*(lons>=-170.)
            ind_lat     = (lats <= 63.)*(lats>=50.)
            tetopo      = etopo[ind_lat, :]
            etopo       = tetopo[:, ind_lon]
            lons        = lons[ind_lon]
            lats        = lats[ind_lat]
            
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            # etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/etopo1.cpt_land')
            # mycm1       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0., vmax=5000.))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000., vmax=-0.5))
        
        if plotetopo:
            m.fillcontinents(color='none', lake_color='deepskyblue',zorder=0.2, alpha=1.)
            shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
            m.readshapefile(shapefname, 'faultline', linewidth = 5, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 3, color='white')
            # 
            shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
            shplst      = shapefile.Reader(shapefname)
            for rec in shplst.records():
                lon_vol = rec[4]
                lat_vol = rec[3]
                xvol, yvol            = m(lon_vol, lat_vol)
                m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=20)
        #     
        #     
        # #######
            from netCDF4 import Dataset
                
            slab2       = Dataset('/home/lili/data_marin/map_data/Slab2Distribute_Mar2018/alu_slab2_dep_02.23.18.grd')
            depthz       = (slab2.variables['z'][:]).data
            lons        = (slab2.variables['x'][:])
            lats        = (slab2.variables['y'][:])
            mask        = (slab2.variables['z'][:]).mask
            
            lonslb,latslb   = np.meshgrid(lons, lats)
            
            lonslb  = lonslb[np.logical_not(mask)]
            latslb  = latslb[np.logical_not(mask)]
            depthslb  = -depthz[np.logical_not(mask)]
            for depth in [40., 60., 80.]:
                ind = abs(depthslb - depth)<1.0
                xslb, yslb = m(lonslb[ind]-360., latslb[ind])
                                                             
                m.plot(xslb, yslb, 'k-', lw=5, mec='k')
                m.plot(xslb, yslb, color = 'yellow', lw=3., mec='k')
        ########
            
        #
        if not plotetopo:
            stax, stay          = m(stalons, stalats)
            m.plot(stax, stay, 'b^', mec='k',markersize=8)
        stax, stay          = m(sxolons, sxolats)
        m.plot(stax, stay, 'r^', mec='k', markersize=8)
        stax, stay          = m(sinlons, sinlats)
        m.plot(stax, stay, '^', color = 'yellow', mec='k', markersize=8)
        stax, stay          = m(exlons, exlats)
        m.plot(stax, stay, '^', color = 'lime', mec='k', markersize=8)
        stax, stay          = m(ex2lons, ex2lats)
        m.plot(stax, stay, 'k^', markersize=8)
        
        stax, stay          = m(ex3lons, ex3lats)
        m.plot(stax, stay, '^', color = 'cyan', mec='k', markersize=8)
        
        
        # stax, stay          = m(sxolons, sxolats)
        # m.plot(stax, stay, 'r^', mec='k', markersize=10)
        # stax, stay          = m(sinlons, sinlats)
        # m.plot(stax, stay, '^', color = 'yellow', mec='k', markersize=10)
        # 
        # stax, stay          = m(exlons, exlats)
        # m.plot(stax, stay, '^', color = 'lime', mec='k', markersize=15)
        # 
        # stax, stay          = m(ex2lons, ex2lats)
        # m.plot(stax, stay, 'k^', markersize=10)
        
        # plt.title(str(self.period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        # if showfig:
            # plt.savefig('aacse_sta.png')
        return
    
    def plot_events(self, cat, vmin=0., vmax=200., plotslab=True, plotfault=True, projection='lambert',\
                    showfig=True, blon=.5, blat=0.5, plotetopo=False):
        """Plot station map
        ==============================================================================
        Input Parameters:
        projection      - type of geographical projection
        geopolygons     - geological polygons for plotting
        blon, blat      - extending boundaries in longitude/latitude
        showfig         - show figure or not
        ==============================================================================
        """
        m                   = self._get_basemap(projection=projection, blon=blon, blat=blat)
        if not plotetopo:
            m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
        
        m.drawcountries(linewidth=1.)
        if plotetopo:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            import pycpt
            etopodata   = Dataset('/home/lili/gebco_aacse.nc')
            etopo       = (etopodata.variables['elevation'][:]).data
            lons        = (etopodata.variables['lon'][:]).data
            lons[lons>180.] = lons[lons>180.] - 360.
            lats        = (etopodata.variables['lat'][:]).data

            ind_lon     = (lons <= -140.)*(lons>=-170.)
            ind_lat     = (lats <= 63.)*(lats>=50.)
            tetopo      = etopo[ind_lat, :]
            etopo       = tetopo[:, ind_lon]
            lons        = lons[ind_lon]
            lats        = lats[ind_lat]
            
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            # etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/etopo1.cpt_land')
            # mycm1       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0., vmax=5000.))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000., vmax=-0.5))
        
        if plotfault:
            m.fillcontinents(color='none', lake_color='deepskyblue',zorder=0.2, alpha=1.)
            shapefname  = '/home/lili/data_marin/map_data/geological_maps/qfaults'
            m.readshapefile(shapefname, 'faultline', linewidth = 5, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 3, color='white')
            # 
            shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
            shplst      = shapefile.Reader(shapefname)
            for rec in shplst.records():
                lon_vol = rec[4]
                lat_vol = rec[3]
                xvol, yvol            = m(lon_vol, lat_vol)
                m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=20)
        #     
        #     
        # #######
        if plotslab:
            from netCDF4 import Dataset
                
            slab2       = Dataset('/home/lili/data_marin/map_data/Slab2Distribute_Mar2018/alu_slab2_dep_02.23.18.grd')
            depthz       = (slab2.variables['z'][:]).data
            lons        = (slab2.variables['x'][:])
            lats        = (slab2.variables['y'][:])
            mask        = (slab2.variables['z'][:]).mask
            
            lonslb,latslb   = np.meshgrid(lons, lats)
            
            lonslb  = lonslb[np.logical_not(mask)]
            latslb  = latslb[np.logical_not(mask)]
            depthslb  = -depthz[np.logical_not(mask)]
            for depth in [40., 60., 80.]:
                ind = abs(depthslb - depth)<1.0
                xslb, yslb = m(lonslb[ind]-360., latslb[ind])
                                                             
                m.plot(xslb, yslb, 'k-', lw=5, mec='k')
                m.plot(xslb, yslb, color = 'yellow', lw=3., mec='k')

        
        xevlst = []
        yevlst = []
        zlst   = []
        for event in cat:
            porigin         = event.preferred_origin()
            evlo            = porigin.longitude
            evla            = porigin.latitude
            try:
                evdp        = porigin.depth/1000.
            except:
                continue
            
            if evlo > 180.:
                xev, yev = m(evlo - 360., evla)
            else:
            # print (evlo)
                xev, yev = m(evlo, evla)

            xevlst.append(xev)
            yevlst.append(yev)
            zlst.append(evdp)
        im  = m.scatter(xevlst, yevlst, s=25, c=zlst, marker='o', cmap='jet', alpha=.8, vmin=vmin, vmax=vmax)
            
        cb  = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label('Depth (km)', fontsize=60, rotation=0)
        cb.ax.tick_params(labelsize=20)
        cb.set_alpha(1)
        cb.draw_all()
        # if showfig:
            # plt.savefig('aacse_sta.png')
        if showfig:
            plt.show()
        return
    
    def plot_stations_mongo(self, projection='lambert2', showfig=True, blon=.5, blat=0.5,vmin=None, vmax=None, plotetopo=False, plotgrav=False):
        """Plot station map
        ==============================================================================
        Input Parameters:
        projection      - type of geographical projection
        geopolygons     - geological polygons for plotting
        blon, blat      - extending boundaries in longitude/latitude
        showfig         - show figure or not
        ==============================================================================
        """
        staLst  = self.waveforms.list()
        stalons = np.array([])
        stalats = np.array([])
        satlons = np.array([])
        satlats = np.array([])
        shdlons = np.array([])
        shdlats = np.array([])
        shvlons = np.array([])
        shvlats = np.array([])
        
        ssplons = np.array([])
        ssplats = np.array([])
        nsyn = 0
        for staid in staLst:
            tmppos          = self.waveforms[staid].coordinates
            tmppos  = self.waveforms[staid].coordinates
            lat     = tmppos['latitude']
            lon     = tmppos['longitude']
            evz     = tmppos['elevation_in_m']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inv     = self.waveforms[staid].StationXML
            # # # if inv[0].code == 'XL':
            # # #     sxllons         = np.append(sxllons, lon)
            # # #     sxllats         = np.append(sxllats, lat)
            # # #     continue
            # # # print (inv[0][0].code[:2])
            if inv[0].code == 'XL' and inv[0][0].code[:2] =='AT':
                satlons         = np.append(satlons, lon)
                satlats         = np.append(satlats, lat)
                continue
            if inv[0].code == 'XL' and inv[0][0].code[:2] =='HD':
                shdlons         = np.append(shdlons, lon)
                shdlats         = np.append(shdlats, lat)
                continue
            if inv[0].code == 'XL' and inv[0][0].code[:2] =='HV':
                shvlons         = np.append(shvlons, lon)
                shvlats         = np.append(shvlats, lat)
                continue
            if inv[0][0].code =='TLY':
                ssplons         = np.append(ssplons, lon)
                ssplats         = np.append(ssplats, lat)
                continue
            if inv[0].code =='SY':
                nsyn+= 1
                continue
            stalons         = np.append(stalons, lon)
            stalats         = np.append(stalats, lat)
        m                   = self._get_basemap(projection=projection, blon=blon, blat=blat)
        print ('SY: %g' %nsyn)
        if plotetopo:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            import pycpt
            etopodata   = Dataset('/home/lili/gebco_mongo.nc')
            etopo       = (etopodata.variables['elevation'][:]).data
            lons        = (etopodata.variables['lon'][:]).data
            lons[lons>180.] = lons[lons>180.] - 360.
            lats        = (etopodata.variables['lat'][:]).data

            # ind_lon     = (lons <= 106.)*(lons>=-170.)
            # ind_lat     = (lats <= 63.)*(lats>=50.)
            # tetopo      = etopo[ind_lat, :]
            # etopo       = tetopo[:, ind_lon]
            # lons        = lons[ind_lon]
            # lats        = lats[ind_lat]
            
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            # etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/etopo1.cpt_land')
            # mycm1       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/ibcso-bath.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=1000., vmax=3500.))
            # m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-1000., vmax=100.))
            # m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-1000., vmax=1000.))
        
            
        if plotetopo:
            m.fillcontinents(color='none', lake_color='deepskyblue',zorder=0.2, alpha=1.)
            m.drawcountries(linewidth=1.5, color = 'black')
            shapefname  = '/home/lili/code/gem-global-active-faults/shapefile/gem_active_faults'
            m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black', default_encoding='windows-1252')
            m.readshapefile(shapefname, 'faultline', linewidth = 2., color='white', default_encoding='windows-1252')
            
            shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
            shplst      = shapefile.Reader(shapefname)
            for rec in shplst.records():
                lon_vol = rec[4]
                lat_vol = rec[3]
                xvol, yvol            = m(lon_vol, lat_vol)
                m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
                
        if plotgrav:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            import pycpt
            gravitydata = Dataset('/home/lili/Downloads/WGM2012_Bouguer_ponc_2min.grd')
            grav        = (gravitydata.variables['z'][:]).data
            lons        = (gravitydata.variables['x'][:]).data
            lons[lons>180.] = lons[lons>180.] - 360.
            lats        = (gravitydata.variables['y'][:]).data
            
            ind_lon     = (lons <= 106.)*(lons>=90.)
            ind_lat     = (lats <= 54.)*(lats>=40.)
            
            tgrav      = grav[ind_lat, :]
            grav       = tgrav[:, ind_lon]
            lons        = lons[ind_lon]
            lats        = lats[ind_lat]
            
            import surfpy.eikonal._grid_class as _grid_class
            gridder     = _grid_class.SphereGridder(minlon = 90., maxlon = 106., dlon = 0.1, \
                            minlat = 40., maxlat = 54., dlat = 0.1, period = 10., \
                            evlo = 0., evla = 0., fieldtype = 'bou', evid = 'plt')
            lons_2d, lats_2d = np.meshgrid(lons, lats)
            gridder.read_array(inlons = lons_2d.reshape(lons_2d.size), inlats = lats_2d.reshape(lons_2d.size), inzarr = grav)
            outfname    = 'plt_bou.lst'
            prefix      = 'plt_bou_'
            width = 10.
            gridder.gauss_smoothing(workingdir = './temp_plt', outfname = outfname, width = width)

            x, y    = m(gridder.lon2d, gridder.lat2d)
            
            cmap = 'panoply'
            import surfpy.cpt_files as cpt_files
            cpt_path    = cpt_files.__path__._path[0]
            
            import pycpt
            cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')

            im          = m.pcolormesh(x, y, gridder.Zarr, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
            cb.set_label('Bouguer Anomaly (mgal)', fontsize=60, rotation=0)
            cb.ax.tick_params(labelsize=25)
            cb.set_alpha(1)
            cb.draw_all()
            m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
            m.drawcountries(linewidth=1.)
            
            shapefname  = '/home/lili/code/gem-global-active-faults/shapefile/gem_active_faults'
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='grey', default_encoding='windows-1252')
            # m.readshapefile(shapefname, 'faultline', linewidth = 2., color='white', default_encoding='windows-1252')
            
            shapefname  = '/home/lili/data_marin/map_data/volcano_locs/SDE_GLB_VOLC.shp'
            shplst      = shapefile.Reader(shapefname)
            for rec in shplst.records():
                lon_vol = rec[4]
                lat_vol = rec[3]
                xvol, yvol            = m(lon_vol, lat_vol)
                m.plot(xvol, yvol, '^', mfc='white', mec='k', ms=15)
            plt.show()
        # shapefname  = '/home/lili/data_mongo/fault_shp/doc-line'
        # m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black')
        # m.readshapefile(shapefname, 'faultline', linewidth = 2., color='white')

        
        stax, stay          = m(satlons, satlats)
        if plotetopo:
            m.plot(stax, stay, '^', markerfacecolor='blue', mec='k', markersize=12)
        else:
            m.plot(stax, stay, '^', markerfacecolor='blue', mec='k', markersize=8)
            
        stax, stay          = m(shdlons, shdlats)
        if plotetopo:
            m.plot(stax, stay, '^', markerfacecolor='cyan', mec='k', markersize=12)
        else:
            m.plot(stax, stay, '^', markerfacecolor='cyan', mec='k', markersize=8)
            
        stax, stay          = m(shvlons, shvlats)
        if plotetopo:
            m.plot(stax, stay, '^', markerfacecolor='red', mec='k', markersize=12)
        else:
            m.plot(stax, stay, '^', markerfacecolor='red', mec='k', markersize=8)
        
        stax, stay          = m(stalons, stalats)
        m.plot(stax, stay, 'y^', mec='k',markersize=8)
        
        stax, stay          = m(ssplons, ssplats)
        if plotetopo:
            m.plot(stax, stay, '^', markerfacecolor='purple', mec='k', markersize=12)
        else:
            m.plot(stax, stay, '^', markerfacecolor='purple', mec='k', markersize=8)
            
            m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
            m.drawcountries(linewidth=1.)

        # plt.title(str(self.period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        # if showfig:
            # plt.savefig('aacse_sta.png')
        return
    
    def plot_stations_spain(self, projection='merc', showfig=True, blon=.5, blat=0.5,vmin=None, vmax=None, plotetopo=False, plotgrav=False):
        """Plot station map
        ==============================================================================
        Input Parameters:
        projection      - type of geographical projection
        geopolygons     - geological polygons for plotting
        blon, blat      - extending boundaries in longitude/latitude
        showfig         - show figure or not
        ==============================================================================
        """
        # import h5py
        # dset    = h5py.File(infname)
        # sta_grp = dset['sta_pts']
        tmplst = []
        with open('sta.log', 'r') as fid:
            for lines in fid.readlines():
                tmplst.append(lines.split()[0])
            
        
        staLst  = self.waveforms.list()
        stalons = np.array([])
        stalats = np.array([])
        
        outlons = np.array([])
        outlats = np.array([])
        
        # # # minlon=-165.+360.
        # # # maxlon=-147+360.
        # # # minlat=51.
        # # # maxlat=62.
        
        minlon=-10.
        maxlon=5.
        minlat=31.
        maxlat=45.

        nsta = 0
        for staid in staLst:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inv     = self.waveforms[staid].StationXML
            if not inv[0].code in tmplst:
                continue
            nsta += 1
            tmppos  = self.waveforms[staid].coordinates
            # # # tmppos  = self.waveforms[staid].coordinates
            lat     = tmppos['latitude']
            lon     = tmppos['longitude']
            if lon > 180.:
                lon -= 360.
            evz     = tmppos['elevation_in_m']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inv     = self.waveforms[staid].StationXML
            if lon<= maxlon and lon >= minlon and lat <= maxlat and lat >= minlat:
                
                stalons         = np.append(stalons, lon)
                stalats         = np.append(stalats, lat)
            else:
                outlons         = np.append(outlons, lon)
                outlats         = np.append(outlats, lat)
        print (nsta)
        m                   = self._get_basemap(projection=projection, blon=blon, blat=blat)
        # print ('SY: %g' %nsyn)
        if plotetopo:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            import pycpt
            etopodata   = Dataset('/raid/lili/data_spain/GEBCO_2020_30_Mar_2021_30cd972b6f07/gebco_2020_n47.0_s27.0_w-12.0_e8.0.nc')
            etopo       = (etopodata.variables['elevation'][:]).data
            lons        = (etopodata.variables['lon'][:]).data
            lons[lons>180.] = lons[lons>180.] - 360.
            lats        = (etopodata.variables['lat'][:]).data

            # ind_lon     = (lons <= 106.)*(lons>=-170.)
            # ind_lat     = (lats <= 63.)*(lats>=50.)
            # tetopo      = etopo[ind_lat, :]
            # etopo       = tetopo[:, ind_lon]
            # lons        = lons[ind_lon]
            # lats        = lats[ind_lat]
            
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            # etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/etopo1.cpt_land')
            # mycm1       = pycpt.load.gmtColormap('/home/lili/data_marin/map_data/station_map/etopo1.cpt')
            # mycm2       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/ibcso-bath.cpt')
            mycm2       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            # m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=1000., vmax=3500.))
            # m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-1000., vmax=100.))
            # m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-1000., vmax=1000.))

            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0., vmax=5000.))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000., vmax=-0.5))
            
        if projection == 'merc' and os.path.isdir('/raid/lili/geo_map_europe'):
            shapefname  = '/raid/lili/geo_map_europe/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='white')


        

            
        
        if plotetopo:
            stax, stay          = m(stalons, stalats)
            m.plot(stax, stay, 'b^', mec='k',markersize=10)
        else:
            stax, stay          = m(stalons, stalats)
            m.plot(stax, stay, 'b^', mec='k',markersize=8)
            stax, stay          = m(outlons, outlats)
            m.plot(stax, stay, 'r^', mec='k',markersize=8)
            
            # m.plot(stax, stay, '^', markerfacecolor='purple', mec='k', markersize=8)
            
            m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
            m.drawcountries(linewidth=1.)
        

        # plt.title(str(self.period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        # if showfig:
            # plt.savefig('aacse_sta.png')
        return
    
    def plot_stations_spain2(self, projection='merc', showfig=True, blon=.5, blat=0.5,vmin=None, vmax=None, plotetopo=False, plotgrav=False):
        """Plot station map
        ==============================================================================
        Input Parameters:
        projection      - type of geographical projection
        geopolygons     - geological polygons for plotting
        blon, blat      - extending boundaries in longitude/latitude
        showfig         - show figure or not
        ==============================================================================
        """
        # import h5py
        # dset    = h5py.File(infname)
        # sta_grp = dset['sta_pts']
        tmplst = []
        with open('sta.log', 'r') as fid:
            for lines in fid.readlines():
                tmplst.append(lines.split()[0])
            
        
        staLst  = self.waveforms.list()
        stalons = np.array([])
        stalats = np.array([])
        
        iblons = np.array([])
        iblats = np.array([])
        
        x7lons = np.array([])
        x7lats = np.array([])
        
        xblons = np.array([])
        xblats = np.array([])
        

        nsta = 0
        for staid in staLst:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inv     = self.waveforms[staid].StationXML
            if not inv[0].code in tmplst:
                continue
            nsta += 1
            tmppos  = self.waveforms[staid].coordinates
            # # # tmppos  = self.waveforms[staid].coordinates
            lat     = tmppos['latitude']
            lon     = tmppos['longitude']
            if lon > 180.:
                lon -= 360.
            evz     = tmppos['elevation_in_m']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inv     = self.waveforms[staid].StationXML
            if inv[0].code == 'IB':
                iblons         = np.append(iblons, lon)
                iblats         = np.append(iblats, lat)
            elif  inv[0].code == 'X7':
                x7lons         = np.append(x7lons, lon)
                x7lats         = np.append(x7lats, lat)
            elif  inv[0].code == 'XB':
                xblons         = np.append(xblons, lon)
                xblats         = np.append(xblats, lat)
            else:
                stalons         = np.append(stalons, lon)
                stalats         = np.append(stalats, lat)
            
        print (nsta)
        m                   = self._get_basemap(projection=projection, blon=blon, blat=blat)
        # print ('SY: %g' %nsyn)
        if plotetopo:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            import pycpt
            etopodata   = Dataset('/raid/lili/data_spain/GEBCO_2020_30_Mar_2021_30cd972b6f07/gebco_2020_n47.0_s27.0_w-12.0_e8.0.nc')
            etopo       = (etopodata.variables['elevation'][:]).data
            lons        = (etopodata.variables['lon'][:]).data
            lons[lons>180.] = lons[lons>180.] - 360.
            lats        = (etopodata.variables['lat'][:]).data


            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            # etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/etopo1.cpt_land')

            mycm2       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/bathy1.cpt')
            mycm2.set_over('w',0)

            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0., vmax=5000.))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000., vmax=-0.5))
            
        if projection == 'merc' and os.path.isdir('/raid/lili/geo_map_europe'):
            shapefname  = '/raid/lili/geo_map_europe/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='white')

        if plotetopo:
            stax, stay          = m(iblons, iblats)
            
            m.plot(stax, stay, 'b^', mec='k',markersize=10)
            stax, stay          = m(x7lons, x7lats)
            m.plot(stax, stay, 'r^', mec='k',markersize=10)
            stax, stay          = m(xblons, xblats)
            m.plot(stax, stay, 'g^', mec='k',markersize=10)
            stax, stay          = m(stalons, stalats)
            m.plot(stax, stay, 'y^', mec='k',markersize=10)
            
        else:
            stax, stay          = m(iblons, iblats)
            
            m.plot(stax, stay, 'b^', mec='k',markersize=8, label = 'IB')
            stax, stay          = m(x7lons, x7lats)
            m.plot(stax, stay, 'r^', mec='k',markersize=8, label = 'X7')
            stax, stay          = m(xblons, xblats)
            m.plot(stax, stay, 'g^', mec='k',markersize=8, label = 'XB')
            stax, stay          = m(stalons, stalats)
            m.plot(stax, stay, 'y^', mec='k',markersize=8, label = 'others')
            
            # m.plot(stax, stay, '^', markerfacecolor='purple', mec='k', markersize=8)
            
            m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
            m.drawcountries(linewidth=1.)
            plt.legend(fontsize = 20, loc='upper left')
            
        print (stalons.size)
        

        # plt.title(str(self.period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        # if showfig:
            # plt.savefig('aacse_sta.png')
        return
    
    def plot_stations_spain3(self, projection='merc', showfig=True, blon=.5, blat=0.5,vmin=None, vmax=None, plotetopo=False, plotgrav=False):
        """Plot station map
        ==============================================================================
        Input Parameters:
        projection      - type of geographical projection
        geopolygons     - geological polygons for plotting
        blon, blat      - extending boundaries in longitude/latitude
        showfig         - show figure or not
        ==============================================================================
        """

        tmplst = []
        with open('sta.log', 'r') as fid:
            for lines in fid.readlines():
                tmplst.append(lines.split()[0])
            
        
        staLst  = self.waveforms.list()
        stalons = np.array([])
        stalats = np.array([])
        nsta = 0
        for staid in staLst:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inv     = self.waveforms[staid].StationXML
            if not inv[0].code in tmplst:
                continue
            nsta += 1
            tmppos  = self.waveforms[staid].coordinates
            lat     = tmppos['latitude']
            lon     = tmppos['longitude']
            if lon > 180.:
                lon -= 360.
            evz     = tmppos['elevation_in_m']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inv     = self.waveforms[staid].StationXML
            # if inv[0][0].code == 'M215' or inv[0][0].code == 'E026' or inv[0][0].code == 'E028' or inv[0][0].code == 'PE03' or inv[0][0].code == 'JSA':
            if inv[0][0].code == 'M215' or inv[0][0].code == 'E026' or inv[0][0].code == 'PE03' or inv[0][0].code == 'JSA':
                stalons         = np.append(stalons, lon)
                stalats         = np.append(stalats, lat)
            
        print (nsta)
        m                   = self._get_basemap(projection=projection, blon=blon, blat=blat)
        # print ('SY: %g' %nsyn)
        if plotetopo:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
            import pycpt
            etopodata   = Dataset('/raid/lili/data_spain/GEBCO_2020_30_Mar_2021_30cd972b6f07/gebco_2020_n47.0_s27.0_w-12.0_e8.0.nc')
            etopo       = (etopodata.variables['elevation'][:]).data
            lons        = (etopodata.variables['lon'][:]).data
            lons[lons>180.] = lons[lons>180.] - 360.
            lats        = (etopodata.variables['lat'][:]).data


            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            # etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/etopo1.cpt_land')

            mycm2       = pycpt.load.gmtColormap('/raid/lili/data_marin/map_data/station_map/bathy1.cpt')
            mycm2.set_over('w',0)

            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0., vmax=5000.))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000., vmax=-0.5))
            
        if projection == 'merc' and os.path.isdir('/raid/lili/geo_map_europe'):
            shapefname  = '/raid/lili/geo_map_europe/prv4_2l-polygon'
            m.readshapefile(shapefname, 'faultline', linewidth = 4, color='black')
            m.readshapefile(shapefname, 'faultline', linewidth = 2, color='white')

        if plotetopo:
            # # stax, stay          = m(iblons, iblats)
            # # 
            # # m.plot(stax, stay, 'b^', mec='k',markersize=10)
            # # stax, stay          = m(x7lons, x7lats)
            # # m.plot(stax, stay, 'r^', mec='k',markersize=10)
            # # stax, stay          = m(xblons, xblats)
            # # m.plot(stax, stay, 'g^', mec='k',markersize=10)
            stax, stay          = m(stalons, stalats)
            m.plot(stax, stay, 'y^', mec='k',markersize=10)
            
        else:
            # stax, stay          = m(iblons, iblats)
            # 
            # m.plot(stax, stay, 'b^', mec='k',markersize=8, label = 'IB')
            # stax, stay          = m(x7lons, x7lats)
            # m.plot(stax, stay, 'r^', mec='k',markersize=8, label = 'X7')
            # stax, stay          = m(xblons, xblats)
            # m.plot(stax, stay, 'g^', mec='k',markersize=8, label = 'XB')
            stax, stay          = m(stalons, stalats)
            m.plot(stax, stay, 'b^', mec='k',markersize=18, label = 'others')
            
            # m.plot(stax, stay, '^', markerfacecolor='purple', mec='k', markersize=8)
            
            m.fillcontinents(color='grey', lake_color='#99ffff',zorder=0.2, alpha=0.5)
            m.drawcountries(linewidth=1.)
            # plt.legend(fontsize = 20, loc='upper left')
            
        print (stalons.size)
        

        # plt.title(str(self.period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        # if showfig:
            # plt.savefig('aacse_sta.png')
        return