import noisexcorr

dset = noisexcorr.xcorrASDF('TEST_LH.h5')

# extract raw data from breqfast mseed to SAC, fill gaps
dset.tar_mseed_to_sac(datadir = '/work2/leon/breqfast_EARS/mseed', outdir='/work2/leon/xcorr_ears/SAC',\
                      start_date='19940501', end_date='20200711', rmresp = False, channels='ENZ')



