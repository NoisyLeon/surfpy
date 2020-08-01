import _grid_class
import numpy as np
# 
# import pycpt
# cmap    = pycpt.load.gmtColormap('../cv.cpt')
#             
tmpfield = _grid_class.SphereGridder(minlat = 52., maxlat = 72., minlon=188., maxlon=238., dlon=1., dlat=0.5)
tmpfield.read_ind(fname = '../alaska_eikonal/10.0sec/TA.H20K_10.0.txt', zindex=2, dindex=5)
# # tmpfield.synthetic_field(lon0=110., lat0=22.) # randomize origins
# # tmpfield.synthetic_field(lon0=110., lat0=22.) # randomize origins
tmpfield2 = tmpfield.copy()
# tmpfield3 = tmpfield.copy()
# tmpfield.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
# tmpfield.gradient(method='metpy')
# tmpfield.laplacian(method='metpy')
# 
# tmpfield2.interp_sph(workingdir='./temp_working', outfname='TA.H20K_10sec')
tmpfield2.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
# # tmpfield2.interp_surface(workingdir='./temp_working_2', outfname='TA.H20K_10sec')
# # tmpfield.get_appV()
# tmpfield2.gradient(method='convolve')
# tmpfield2.laplacian(method='green')
# # tmpfield2.grdgradient(workingdir='./temp_working_2', outfname='TA.H20K_10sec')
# # tmpfield3.grdgradient_cartesian2(workingdir='./temp_working_3', outfname='TA.H20K_10sec')
# 
# tmpfield.get_apparent_vel()
# tmpfield2.get_apparent_vel()
# # tmpfield3.get_appV()
# # 
# tmpfield.plot('v', vmin=2.4, vmax=3.4, showfig=True)
# # tmpfield2.plot('v', vmin=2.4, vmax=3.4, showfig=True)
# # tmpfield.plot('v', vmin=2.95, vmax=3.05, showfig=False)
# # tmpfield2.plot('v', vmin=2.95, vmax=3.05, showfig=True)
# # tmpfield3.plot('v', vmin=2.95, vmax=3.05, showfig=True)
# tmpfield.plot('lplc', vmin=-0.001, vmax=0.001, showfig=False)
# tmpfield2.plot('lplc', vmin=-0.0005, vmax=0.0005, showfig=True)









# 
# # 
# # 
# tmpfield.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
# evid = 'TA.H20K'
# channel = 'ZZ'
# tmpfield.check_curvature(workingdir='./temp_working', outpfx=evid+'_'+channel+'_')
# cdist = 250.
# tmpfield.eikonal(workingdir='./temp_working', inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)



