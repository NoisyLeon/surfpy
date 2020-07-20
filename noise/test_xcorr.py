import noisexcorr

# dset = noisexcorr.xcorrASDF('TEST_LH.h5')
dset = noisexcorr.xcorrASDF('EARS_LH.h5')
# # dset.tar_mseed_to_sac(datadir = '/home/lili/data/newbreq_fast/mseed', outdir='/home/lili/data/newbreq_fast/surfpy_out',\
# #                       start_date='20040101', end_date='20081231')
# 
# dset.tar_mseed_to_sac(datadir = '/home/lili/data/newbreq_fast/mseed', outdir='/home/lili/data/newbreq_fast/surfpy_out_resp2',\
#                       start_date='20040101', end_date='20081231')
# 
# # dset.tar_mseed_to_sac(datadir = '/home/lili/data/newbreq_fast/mseed', outdir='/home/lili/data/newbreq_fast/surfpy_out_resp',\
# #                       start_date='20040422', end_date='20040424', verbose2=True)
# 
# dset.tar_mseed_to_sac(datadir = '/home/lili/data/newbreq_fast/mseed', outdir='/home/lili/data/newbreq_fast/surfpy_out_s2c_resp',\
#                       start_date='20040101', end_date='20081231', outtype=1, channels='Z')
# 
# dset.tar_mseed_to_sac(datadir = '/home/lili/data/newbreq_fast/mseed', outdir='/home/lili/data/newbreq_fast/surfpy_out',\
#                       start_date='20040101', end_date='20040228', rmresp=True)

# dset = noisexcorr.xcorrASDF('/home/lili/data_marin/xcorr_Alaska_20190312.h5')

# dset.wsac_xcorr_all(netcode1='AK', stacode1='', netcode2, stacode2='TA')


# a = dset.tar_mseed_to_sac(datadir = '/home/lili/data_ears/tarmseed', outdir='/home/lili/data_ears/debug_sac',\
#                       start_date='20090104', end_date='20090104')

dset.compute_xcorr(datadir = '/home/lili/data_ears/debug_sac', runtype=-1, CorOutflag = 0,  skipinv=True, \
                      start_date='19940501', end_date='19940601', chans=['LHE', 'LHN'],\
                        nprocess=3, verbose=True, verbose2=True, parallel=False)