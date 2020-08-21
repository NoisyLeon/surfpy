from netCDF4 import Dataset
import h5py 
etopodbase  = Dataset('./ETOPO2v2g_f4.nc')
etopo       = etopodbase.variables['z'][:]
lons        = etopodbase.variables['x'][:]
lats        = etopodbase.variables['y'][:]

dset = h5py.File('etopo2.h5')
dset.create_dataset(name = 'etopo', data = etopo, compression="gzip")
dset.create_dataset(name = 'longitudes', data = lons, compression="gzip")
dset.create_dataset(name = 'latitudes', data = lats, compression="gzip")
dset.close()
