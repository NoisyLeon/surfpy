import threestations

dset = threestations.tripleASDF('/home/lili/COR_WUS.h5')
# dset.direct_interfere(datadir = '/home/lili/data_threestation', parallel = False, nprocess = 3, subsize = 200)
dset.dw_aftan(datadir = '/home/lili/data_threestation', prephdir = '/home/lili/data_wus/prep_ph_vel/ph_vel_R')
# tmppos1 = self.waveforms[staid1].coordinates
# lat1    = tmppos1['latitude']
# lon1    = tmppos1['longitude']
# stz1    = tmppos1['elevation_in_m']
# 
# 
# netcode, stacode= staid.split('.')
#                 tmppos          = self.waveforms[staid1].coordinates
#                 lat             = tmppos['latitude']
#                 lon             = tmppos['longitude']


import surfpy.aftan.pyaftan as pyaftan
import glob
import matplotlib.pyplot as plt
import obspy
import numpy as np
f   = '/home/lili/data_threestation/COR/AZ.MONP/COR_AZ.MONP_LHZ_TA.R12A_LHZ.SAC'

st  = obspy.read(f)
tr  = st[0]
tr  = pyaftan.aftantrace(tr.data, tr.stats)

tr.makesym()
tr.aftanf77(piover4=-1., pmf=True, vmin=1.5, vmax=5.0, ffact=1. , tmin=4.0, tmax=70.0,\
            phvelname = '/home/lili/data_wus/prep_ph_vel/ph_vel_R/AZ.MONP.TA.R12A.pre')
tr.get_snr()


pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
c0      = np.interp(pers, tr.ftanparam.arr2_2[1, :tr.ftanparam.nfout2_2], tr.ftanparam.arr2_2[3, :tr.ftanparam.nfout2_2])

fpattern = '/home/lili/data_threestation/SYNC_C3/AZ.MONP/C3_AZ.MONP_Z_TA.R12A_Z_*ELL.npz'
fnamelst = glob.glob(fpattern)
for fname in fnamelst:
    fparam  = pyaftan.ftanParam()
    fparam.load_npy(fname)
    c = np.interp(pers, fparam.arr2_2[1, :fparam.nfout2_2], fparam.arr2_2[3, :fparam.nfout2_2])
    
    # plt.plot(fparam.arr2_2[1, :fparam.nfout2_2], fparam.arr2_2[3, :fparam.nfout2_2], 'grey')
    plt.plot(pers, c, 'grey')

plt.plot(pers, c0, 'red')
# plt.plot(tr2.ftanparam.arr2_2[1, :tr2.ftanparam.nfout2_2], tr2.ftanparam.arr2_2[3, :tr2.ftanparam.nfout2_2], 'blue')
plt.show()
    
