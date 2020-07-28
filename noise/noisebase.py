# -*- coding: utf-8 -*-
"""
Base ASDF for I/O and plotting of noise data
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
import obspy
import warnings
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm

sta_info_default        = {'xcorr': 1, 'isnet': 0}
xcorr_header_default    = {'netcode1': '', 'stacode1': '', 'netcode2': '', 'stacode2': '', 'chan1': '', 'chan2': '',
        'npts': 12345, 'b': 12345, 'e': 12345, 'delta': 12345, 'dist': 12345, 'az': 12345, 'baz': 12345, 'stackday': 0}
xcorr_sacheader_default = {'knetwk': '', 'kstnm': '', 'kcmpnm': '', 'stla': 12345, 'stlo': 12345, 
            'kuser0': '', 'kevnm': '', 'evla': 12345, 'evlo': 12345, 'evdp': 0., 'dist': 0., 'az': 12345, 'baz': 12345, 
                'delta': 12345, 'npts': 12345, 'user0': 0, 'b': 12345, 'e': 12345}
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
            limits_lonlat_param = self.auxiliary_data.NoiseXcorr['limits_lonlat'].parameters
            self.minlat         = limits_lonlat_param['minlat']
            self.maxlat         = limits_lonlat_param['maxlat']
            self.minlon         = limits_lonlat_param['minlon']
            self.maxlon         = limits_lonlat_param['maxlon']
        except:
            pass
        return
    
    def print_info(self):
        """
        print information of the dataset.
        """
        outstr  = '================================================= Ambient Noise Cross-correlation Database =================================================\n'
        outstr  += self.__str__()+'\n'
        outstr  += '--------------------------------------------------------------------------------------------------------------------------------------------\n'
        if 'NoiseXcorr' in self.auxiliary_data.list():
            outstr      += '[NoiseXcorr]              - Cross-correlation seismogram\n'
            if 'data_counts' in self.auxiliary_data.NoiseXcorr.list():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    subdset = self.auxiliary_data.NoiseXcorr['data_counts']
                    data    = subdset.data.value
                stackday    = data[:, 0]
                trcount     = data[:, 1]
                for i in range(stackday.size):
                    outstr  += '--- Stack days >= %5d:                 %8d traces \n' %(stackday[i], trcount[i])
        if 'StaInfo' in self.auxiliary_data.list():
            outstr      += '[StaInfo]                 - Auxiliary station information\n'
        if 'DISPbasic1' in self.auxiliary_data.list():
            outstr      += '[DISPbasic1]              - Basic dispersion curve, no jump correction\n'
        if 'DISPbasic2' in self.auxiliary_data.list():
            outstr      += '[DISPbasic2]              - Basic dispersion curve, with jump correction\n'
        if 'DISPpmf1' in self.auxiliary_data.list():
            outstr      += '[DISPpmf1]                - PMF dispersion curve, no jump correction\n'
        if 'DISPpmf2' in self.auxiliary_data.list():
            outstr      += '[DISPpmf2]                - PMF dispersion curve, with jump correction\n'
        if 'DISPbasic1interp' in self.auxiliary_data.list():
            outstr      += '[DISPbasic1interp]        - Interpolated DISPbasic1\n'
        if 'DISPbasic2interp' in self.auxiliary_data.list():
            outstr      += '[DISPbasic2interp]        - Interpolated DISPbasic2\n'
        if 'DISPpmf1interp' in self.auxiliary_data.list():
            outstr      += '[DISPpmf1interp]          - Interpolated DISPpmf1\n'
        if 'DISPpmf2interp' in self.auxiliary_data.list():
            outstr      += '[DISPpmf2interp]          - Interpolated DISPpmf2\n'
        if 'FieldDISPbasic1interp' in self.auxiliary_data.list():
            outstr      += '[FieldDISPbasic1interp]   - Field data of DISPbasic1\n'
        if 'FieldDISPbasic2interp' in self.auxiliary_data.list():
            outstr      += '[FieldDISPbasic2interp]   - Field data of DISPbasic2\n'
        if 'FieldDISPpmf1interp' in self.auxiliary_data.list():
            outstr      += '[FieldDISPpmf1interp]     - Field data of DISPpmf1\n'
        if 'FieldDISPpmf2interp' in self.auxiliary_data.list():
            outstr      += '[FieldDISPpmf2interp]     - Field data of DISPpmf2\n'
        
        outstr += '============================================================================================================================================\n'
        print(outstr)
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
                        print ('WARNING: repeated station:' +netsta+' in station list!')
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
                        print ('WARNING: repeated station:' +netsta+' in station list!')
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
            print ('Copying/renewing station inventory: '+ staid)
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
        self.add_auxiliary_data(data=np.array([]), data_type='NoiseXcorr', path='limits_lonlat', parameters=paramters)
        self.minlat = minlat
        self.maxlat = maxlat
        self.minlon = minlon
        self.maxlon = maxlon
        return
            
    def _get_basemap(self, projection='lambert', resolution='i', blon=0., blat=0.):
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
        try:
            m.drawcoastlines(linewidth=1.0)
        except:
            pass
        m.drawcountries(linewidth=1.)
        # m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        # m.drawmapboundary(fill_color="white")
        return m
    
    def plot_stations(self, projection='lambert', tomo_vertices=[], showfig=True, blon=.5, blat=0.5):
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
            with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmppos  = self.waveforms[staid].coordinates
            stla            = tmppos['latitude']
            stlo            = tmppos['longitude']
            evz             = tmppos['elevation_in_m']
            stalons         = np.append(stalons, stlo); stalats=np.append(stalats, stla)
        m                   = self._get_basemap(projection=projection, blon=blon, blat=blat)
        m.etopo()
        # m.shadedrelief()
        stax, stay          = m(stalons, stalats)
        m.plot(stax, stay, 'b^', markersize=10)
        if len(tomo_vertices) >= 3:
            lons    = []
            lats    = []
            for vertice in tomo_vertices:
                lons.append(vertice[0])
                lats.append(vertice[1])
            verx, very          = m(lons, lats)
            m.plot(verx, very, 'r-')
            m.plot([verx[-1], verx[0]], [very[-1], very[0]], 'r-')
        
        if showfig:
            plt.show()
        return
    
    def write_sac(self, netcode1, stacode1, netcode2, stacode2, chan1, chan2, outdir='.', pfx='COR'):
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
        try:
            subdset                 = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        except:
            return False
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
        sacTr                       = obspy.io.sac.sactrace.SACTrace(data=np.float64(subdset.data.value), **xcorr_sacheader)
        sacfname                    = outdir+ '/' +pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
        sacTr.write(sacfname)
        return True
    
    def write_sac_allch(self, netcode1, stacode1, netcode2, stacode2, outdir='.', pfx='COR'):
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
                self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                    stacode2=stacode2, chan1=chan1, chan2=chan2, outdir=outdir, pfx=pfx)
        return
    
    def write_sac_all(self, outdir, channels=['LHZ'], pfx='COR'):
        outdir = outdir +'/COR_STACK'
        for staid1 in self.waveforms.list():
            netcode1, stacode1  = staid1.split('.')
            outstadir   = outdir+'/'+staid1
            if not os.path.isdir(outstadir):
                os.makedirs(outstadir)
            for staid2 in self.waveforms.list():
                if staid1 >= staid2:
                    continue
                netcode2, stacode2  = staid2.split('.')
                for chan1 in channels:
                    for chan2 in channels:
                        self.write_sac(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1, chan2=chan2, outdir=outstadir, pfx=pfx)
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
        try:
            subdset             = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        except:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sta1            = self.waveforms[netcode1+'.'+stacode1].StationXML.networks[0].stations[0]
            sta2            = self.waveforms[netcode2+'.'+stacode2].StationXML.networks[0].stations[0]
        tr                  = obspy.core.Trace()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr.data         = subdset.data.value
        tr.stats.sac        = {}
        tr.stats.sac.evla   = sta1.latitude
        tr.stats.sac.evlo   = sta1.longitude
        tr.stats.sac.stla   = sta2.latitude
        tr.stats.sac.stlo   = sta2.longitude
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
                    print ('reading xcorr data: '+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2)
        return
    
    def count_data(self, channel='ZZ', stackday = None, recompute=False):
        """count the number of available xcorr traces
        """
        if stackday is None:
            stackday= np.array([1, 30, 60, 90, 180, 360, 720])
        # check if data counts already exists
        stackday    = np.int32(stackday)
        try:
            subdset = self.auxiliary_data.NoiseXcorr['data_counts']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data= subdset.data.value
            tmpsdays= data[:, 0]
            trcount = data[:, 1]
            if isinstance(stackday, np.ndarray):
                if tmpsdays.size == stackday.size:
                    if np.alltrue(tmpsdays == stackday):
                        for i in range(stackday.size):
                            print ('--- Stack days >= %5d:                 %8d traces ' %(stackday[i], trcount[i]))
                        if not recompute:
                            return
            elif stackday in tmpsdays:
                ind = stackday == tmpsdays
                print ('--- Stack days >= %5d:                 %8d traces ' %(stackday[ind], trcount[ind]))
                if not recompute:
                    return 
        except:
            pass
        print ('*** Recomputing data counts!')
        if not isinstance(stackday, np.ndarray):
            stackday    = np.array([stackday])
        trcount         = np.zeros(stackday.size)
        staLst          = self.waveforms.list()
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1  = staid1.split('.')
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                try:
                    channels1       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2       = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    for chan in channels1:
                        if chan[-1]==channel[0]:
                            chan1   = chan
                    for chan in channels2:
                        if chan[-1]==channel[1]:
                            chan2   = chan
                    subdset         = self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
                except KeyError:
                    continue
                tmpday  = subdset.parameters['stackday']
                trcount += (tmpday >= stackday)
        data        = np.zeros((trcount.size, 2), dtype = np.int32)
        data[:, 0]  = stackday[:]
        data[:, 1]  = trcount[:]
        try:
            del self.auxiliary_data.NoiseXcorr['data_counts']
        except:
            pass
        self.add_auxiliary_data(data=data, data_type='NoiseXcorr', path='data_counts', parameters={})
        for i in range(stackday.size):
            print ('--- Stack days >= %5d:                 %8d traces ' %(stackday[i], trcount[i]))
        return
    
    def plot_waveforms(self, staxml=None, chan1='LHZ', chan2='LHZ'):
        """plot the xcorr waveforms
        """
        if staxml != None:
            inv             = obspy.read_inventory(staxml)
            waveformLst     = []
            for network in inv:
                netcode     = network.code
                for station in network:
                    stacode = station.code
                    waveformLst.append(netcode+'.'+stacode)
            staLst          = waveformLst
            print ('--- Load stations from input StationXML file')
        else:
            print ('--- Load all the stations from database')
            staLst          = self.waveforms.list()
        ax              = plt.subplot()
        for staid1 in staLst:
            netcode1, stacode1  = staid1.split('.')
            for staid2 in staLst:
                netcode2, stacode2  = staid2.split('.')
                if staid1 >= staid2:
                    continue
                tr      = self.get_xcorr_trace(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2, stacode2=stacode2,\
                                    chan1=chan1, chan2=chan2)
                if tr is None:
                    continue
                dist    = tr.stats.sac.dist
                time    = tr.stats.sac.b + np.arange(tr.stats.npts)*tr.stats.delta
                plt.plot(time, tr.data/abs(tr.data.max())*10. + dist, 'k-', lw= 0.1)

        # plt.xlim([-1000., 1000.])
        # plt.ylim([-1., 1000.])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.ylabel('Distance (km)', fontsize=30)
        plt.xlabel('Time (s)', fontsize=30)
        plt.show()