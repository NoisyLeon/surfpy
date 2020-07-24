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
    
    def request_surf(self):
        """request surface wave data
        """
        return
    
    def request_rf(self):
        """request receiver function data
        """
        return
    