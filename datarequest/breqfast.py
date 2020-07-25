# -*- coding: utf-8 -*-
"""
ASDF for breqfast
    
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
import smtplib
import numpy as np

mondict = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


class breqfastASDF(browsebase.baseASDF):
    
    def request_noise(self, start_date = None, end_date = None, skipinv=True, channels=['LHE', 'LHN', 'LHZ'], label='LF',
            quality = 'B', name = 'LiliFeng', email_address='lfengmac@gmail.com', iris_email='breq_fast@iris.washington.edu'):
        """request continuous data for noise analysis
        """
        if start_date is None:
            start_date  = self.start_date
        else:
            start_date  = obspy.UTCDateTime(start_date)
        if end_date is None:
            end_date    = self.end_date
        else:
            end_date    = obspy.UTCDateTime(end_date)
        header_str1     = '.NAME %s\n' %name + '.INST CU\n'+'.MAIL University of Colorado Boulder\n'
        header_str1     += '.EMAIL %s\n' %email_address+'.PHONE\n'+'.FAX\n'+'.MEDIA: Electronic (FTP)\n'
        header_str1     += '.ALTERNATE MEDIA: Electronic (FTP)\n'
        FROM            = 'no_reply@surfpy.com'
        TO              = iris_email
        title           = 'Subject: Requesting Data\n\n'
        ctime           = start_date
        while(ctime <= end_date):
            year        = ctime.year
            month       = ctime.month
            day         = ctime.day
            ctime       += 86400
            year2       = ctime.year
            month2      = ctime.month
            day2        = ctime.day
            header_str2 = header_str1 +'.LABEL %s_%d.%s.%d\n' %(label, year, mondict[month], day)
            header_str2 += '.QUALITY %s\n' %quality +'.END\n'
            day_str     = '%d %d %d 0 0 0 %d %d %d 0 0 0' %(year, month, day, year2, month2, day2)
            out_str     = ''
            Nsta        = 0
            for network in self.inv:
                for station in network:
                    netcode = network.code
                    stacode = station.code
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # sta_inv     = self.inv.select(network=netcode, station=stacode)[0][0]
                        st_date     = station.start_date
                        ed_date     = station.end_date
                    if skipinv and (ctime < st_date or (ctime - 86400) > ed_date):
                        continue
                    Nsta            += 1
                    for chan in channels:
                        chan_str    = '1 %s' %chan
                        sta_str     = '%s %s %s %s\n' %(stacode, netcode, day_str, chan_str)
                        out_str     += sta_str
            out_str     = header_str2 + out_str
            if Nsta == 0:
                print ('--- [NOISE DATA REQUEST] No data available in inventory, Date: %s' %(ctime - 86400).isoformat().split('T')[0])
                continue
            #========================
            # send email to IRIS
            #========================
            server  = smtplib.SMTP('localhost')
            MSG     = title + out_str
            server.sendmail(FROM, TO, MSG)
            server.quit()
            print ('--- [NOISE DATA REQUEST] email sent to IRIS, Date: %s' %(ctime - 86400).isoformat().split('T')[0])
        return
    
    def request_noise_obs(self, start_date = None, end_date = None, skipinv=True, chanrank=['L', 'B', 'H'],
            label='LF', quality = 'B', name = 'LiliFeng', obs_channels = ['H1', 'H2', 'HZ', 'DH'],
            send_email=False, email_address='lfengmac@gmail.com', iris_email='breq_fast@iris.washington.edu'):
        """request continuous obs data for noise analysis
        """
        #================================
        # Determine the channel type
        #================================
        is_obs_lst      = np.zeros(len(self.waveforms.list()), dtype = bool)
        chan_type_lst   = [] 
        ista            = 0
        for network in self.inv:
            for station in network:
                netcode = network.code
                stacode = station.code
                # use existence of pressure gauge to check if the station is obs or not
                tmpchanDH       = station.select(channel = '?DH')
                tmpEDH          = station.select(channel = 'EDH')
                if len(tmpchanDH) == 0:
                    is_obs      = False
                    tmpchanZ    = station.select(channel = '?HZ')
                elif len(tmpchanDH) == len(tmpEDH):
                    print ('--- [NOISE DATA REQUEST] EDH obs, only request Z component, station: %s.%s' %(netcode, stacode))
                    is_obs      = False
                    tmpchanZ    = station.select(channel = '?HZ')
                else:
                    is_obs          = True
                    is_obs_lst[ista]= True
                ista            += 1
                # determine the channel type
                chantype        = ''
                for tmpchtype in chanrank:
                    if is_obs:
                        tmpchanDH2  = tmpchanDH.select(channel = tmpchtype + 'DH')
                        if len(tmpchanDH2) > 0:
                            chantype= tmpchtype
                            break
                    else:
                        tmpchanZ2   = tmpchanZ.select(channel = tmpchtype + 'HZ')
                        if len(tmpchanZ2) > 0:
                            chantype= tmpchtype
                            break
                if chantype == '':
                    print ('--- [NOISE DATA REQUEST] No data available for station: %s.%s' %(netcode, stacode))
                    chan_type_lst.append('')
                    continue
                if is_obs:
                    # check if all channels exist
                    for tmpch in obs_channels:
                        try:
                            chan    = (station.select(channel = chantype+tmpch)).code
                        except:
                            print ('--- [NOISE DATA REQUEST] Not all obs channels available for station: %s.%s' %(netcode, stacode))
                            break 
                chan_type_lst.append(chantype)
        if not send_email:
            return chan_type_lst, is_obs_lst
        # initialization of date and header strings
        if start_date is None:
            start_date  = self.start_date
        else:
            start_date  = obspy.UTCDateTime(start_date)
        if end_date is None:
            end_date    = self.end_date
        else:
            end_date    = obspy.UTCDateTime(end_date)
        header_str1     = '.NAME %s\n' %name + '.INST CU\n'+'.MAIL University of Colorado Boulder\n'
        header_str1     += '.EMAIL %s\n' %email_address+'.PHONE\n'+'.FAX\n'+'.MEDIA: Electronic (FTP)\n'
        header_str1     += '.ALTERNATE MEDIA: Electronic (FTP)\n'
        FROM            = 'no_reply@surfpy.com'
        TO              = iris_email
        title           = 'Subject: Requesting Data\n\n'
        ctime           = start_date
        while(ctime <= end_date):
            year        = ctime.year
            month       = ctime.month
            day         = ctime.day
            ctime       += 86400
            year2       = ctime.year
            month2      = ctime.month
            day2        = ctime.day
            header_str2 = header_str1 +'.LABEL %s_%d.%s.%d\n' %(label, year, mondict[month], day)
            header_str2 += '.QUALITY %s\n' %quality +'.END\n'
            day_str     = '%d %d %d 0 0 0 %d %d %d 0 0 0' %(year, month, day, year2, month2, day2)
            out_str     = ''
            Nsta        = 0
            ista        = 0
            for network in self.inv:
                for station in network:
                    netcode = network.code
                    stacode = station.code
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        st_date     = station.start_date
                        ed_date     = station.end_date
                    if skipinv and (ctime < st_date or (ctime - 86400) > ed_date):
                        continue
                    chantype        = chan_type_lst[ista]
                    is_obs          = is_obs_lst[ista]
                    ista            += 1
                    if chantype == '':
                        continue 
                    if is_obs:
                        for tmpch in obs_channels:
                            chan        = chantype+tmpch
                            chan_str    = '1 %s' %chan
                            sta_str     = '%s %s %s %s\n' %(stacode, netcode, day_str, chan_str)
                            out_str     += sta_str
                    else:
                        chan        = chantype+'HZ'
                        chan_str    = '1 %s' %chan
                        sta_str     = '%s %s %s %s\n' %(stacode, netcode, day_str, chan_str)
                        out_str     += sta_str
                    Nsta    += 1
            out_str     = header_str2 + out_str
            if Nsta == 0:
                print ('--- [NOISE DATA REQUEST] No data available in inventory, Date: %s' %(ctime - 86400).isoformat().split('T')[0])
                continue
            #========================
            # send email to IRIS
            #========================
            server  = smtplib.SMTP('localhost')
            MSG     = title + out_str
            server.sendmail(FROM, TO, MSG)
            server.quit()
            print ('--- [NOISE DATA REQUEST] email sent to IRIS, Date: %s' %(ctime - 86400).isoformat().split('T')[0])
        return
    
    def request_surf(self):
        """request surface wave data
        """
        return
    
    def request_rf(self):
        """request receiver function data
        """
        return
    