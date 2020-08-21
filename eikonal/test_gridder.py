import _grid_class
import numpy as np
import verde as vd
import pyproj
from pyproj import Geod
geodist             = Geod(ellps='WGS84')

# 
# import pycpt
# cmap    = pycpt.load.gmtColormap('../cv.cpt')
#             
tmpfield = _grid_class.SphereGridder(minlat = 52., maxlat = 72., minlon=188., maxlon=238., dlon=0.2, dlat=0.1)

tmpfield.read_ind(fname = './TA.H20K_10.0.txt', zindex=2, dindex=5)
tmpfield.synthetic_field(lon0=110., lat0=22.) # randomize origins

tmpfield.ZarrIn /= 1000.

# tmpfield.synthetic_field(lon0=110., lat0=22.) # randomize origins
tmpfield2 = tmpfield.copy()
# tmpfield3 = tmpfield.copy()
# tmpfield.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
tmpfield.interp_surface(tension = 0.0, do_blockmedian = True)
# tmpfield.interp_verde(mindist = 1e5)
# tmpfield.gradient(method='metpy')
# tmpfield.laplacian(method='metpy')
# 
# tmpfield2.interp_sph(workingdir='./temp_working', outfname='TA.H20K_10sec')

# tmpfield2.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
# tmpfield2.Zarr = grid_full['travel_time'].data
# tmpfield2.interp_verde()
tmpfield2.interp_surface_old(workingdir='./temp_working', outfname='TA.H20K_10sec', tension = 0.0)
# tmpfield2.gradient(method='metpy')
# # tmpfield2.interp_surface(workingdir='./temp_working_2', outfname='TA.H20K_10sec')
# tmpfield.get_apparent_vel()
# tmpfield2.gradient(method='convolve')
# tmpfield2.laplacian(method='green')
# # tmpfield2.grdgradient(workingdir='./temp_working_2', outfname='TA.H20K_10sec')
# # tmpfield3.grdgradient_cartesian2(workingdir='./temp_working_3', outfname='TA.H20K_10sec')
# 
# tmpfield.get_apparent_vel()
# tmpfield2.get_apparent_vel()
# 
# # tmpfield.plot('v', vmin=2.4, vmax=3.4, showfig=False)
# # tmpfield2.plot('v', vmin=2.4, vmax=3.4, showfig=True)
# tmpfield.plot('v', vmin=2.95, vmax=3.05, showfig=False, stations = True)
# tmpfield2.plot('v', vmin=2.95, vmax=3.05, showfig=True, stations = True)
# tmpfield3.plot('v', vmin=2.95, vmax=3.05, showfig=True)
# tmpfield.plot('lplc', vmin=-0.001, vmax=0.001, showfig=False)
# tmpfield2.plot('lplc', vmin=-0.0005, vmax=0.0005, showfig=True)
# 
# lon0=110.
# lat0=22.
# v = 3.0
# az, baz, distevent  = geodist.inv( np.ones(tmpfield.lon2d.size)*lon0, np.ones(tmpfield.lon2d.size)*lat0,\
#                         tmpfield.lon2d.reshape(tmpfield.lon2d.size), tmpfield.lat2d.reshape(tmpfield.lon2d.size))
# Zarr         = distevent/v/1000.
# Zarr = Zarr.reshape(tmpfield.lon2d.shape)
tmpfield2.Zarr = tmpfield2.Zarr - tmpfield.Zarr
tmpfield2.mask = abs(tmpfield2.Zarr) > 2.
tmpfield2.plot('z', vmin = -.0001, vmax = .0001, showfig=True, stations = True)


# 
# # 
# # 
# tmpfield.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
# evid = 'TA.H20K'
# channel = 'ZZ'
# tmpfield.check_curvature(workingdir='./temp_working', outpfx=evid+'_'+channel+'_')
# cdist = 250.
# tmpfield.eikonal(workingdir='./temp_working', inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)



