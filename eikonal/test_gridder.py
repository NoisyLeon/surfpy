import _grid_class
import numpy as np
import verde as vd
import pyproj



# 
# import pycpt
# cmap    = pycpt.load.gmtColormap('../cv.cpt')
#             
tmpfield = _grid_class.SphereGridder(minlat = 52., maxlat = 72., minlon=188., maxlon=238., dlon=0.2, dlat=0.1)

tmpfield.read_ind(fname = './TA.H20K_10.0.txt', zindex=2, dindex=5)
# tmpfield.synthetic_field(lon0=110., lat0=22.) # randomize origins
# 
# projection = pyproj.Proj(proj="merc", lat_ts=tmpfield.lats.mean())
# coordinates = (tmpfield.lonsIn, tmpfield.latsIn)
# coordinates = projection(*coordinates)
# 
# tmplats = (np.array([0, 0]), np.array([tmpfield.lats.mean(), tmpfield.lats.mean() + tmpfield.dlat]))
# tmplats = projection(*tmplats)
# dlat_meters    = tmplats[1][1] - tmplats[1][0]
# tmplons = (np.array([0, tmpfield.dlon]), np.array([tmpfield.lats.mean(), tmpfield.lats.mean()]))
# tmplons = projection(*tmplons)
# dlon_meters = tmplons[0][1] - tmplons[0][0]
# spacing = 15 / 60
# # Now we can chain a blocked mean and spline together. The Spline can be regularized
# # by setting the damping coefficient (should be positive). It's also a good idea to set
# # the minimum distance to the average data spacing to avoid singularities in the spline.
# chain = vd.Chain(
#     [
#         # ("mean", vd.BlockReduce(np.mean, spacing=spacing * 111e3)),
#         # ("mean", vd.BlockReduce(np.mean, \
#         #                       spacing=(tmpfield.dlat*111e3, tmpfield.dlon*111e3))),
#         ("mean", vd.BlockMean( \
#                               spacing=(dlat_meters, dlon_meters))),
#         ("spline", vd.Spline(damping=None, mindist=1e-5)),
#         # ("spline", vd.SplineCV()),
#     ]
# )
# print(chain)
# 
# # Fit the model on the training set
# chain.fit(coordinates, data = tmpfield.ZarrIn)
# 
# # And calculate an R^2 score coefficient on the testing set. The best possible score
# # (perfect prediction) is 1. This can tell us how good our spline is at predicting data
# # that was not in the input dataset.
# # score = chain.score(*test)
# # print("\nScore: {:.3f}".format(score))
# 
# # Now we can create a geographic grid of air temperature by providing a projection
# # function to the grid method and mask points that are too far from the observations
# grid_full = chain.grid(
#     region=(tmpfield.minlon, tmpfield.maxlon, tmpfield.minlat, tmpfield.maxlat),
#     spacing=(tmpfield.dlat, tmpfield.dlon),
#     projection=projection,
#     dims=["latitude", "longitude"],
#     data_names=["travel_time"],
# )



# # tmpfield.synthetic_field(lon0=110., lat0=22.) # randomize origins
tmpfield2 = tmpfield.copy()
# tmpfield3 = tmpfield.copy()
# tmpfield.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
tmpfield.interp_verde(mindist = 1e5)
tmpfield.gradient(method='metpy')
# tmpfield.laplacian(method='metpy')
# 
# tmpfield2.interp_sph(workingdir='./temp_working', outfname='TA.H20K_10sec')

# tmpfield2.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
# tmpfield2.Zarr = grid_full['travel_time'].data
tmpfield2.interp_verde()
# tmpfield2.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec', tension = 0.2)
tmpfield2.gradient(method='metpy')
# # tmpfield2.interp_surface(workingdir='./temp_working_2', outfname='TA.H20K_10sec')
# tmpfield.get_apparent_vel()
# tmpfield2.gradient(method='convolve')
# tmpfield2.laplacian(method='green')
# # tmpfield2.grdgradient(workingdir='./temp_working_2', outfname='TA.H20K_10sec')
# # tmpfield3.grdgradient_cartesian2(workingdir='./temp_working_3', outfname='TA.H20K_10sec')
# 
tmpfield.get_apparent_vel()
tmpfield2.get_apparent_vel()

# tmpfield.plot('v', vmin=2.4, vmax=3.4, showfig=False)
# tmpfield2.plot('v', vmin=2.4, vmax=3.4, showfig=True)
tmpfield.plot('v', vmin=2.95, vmax=3.05, showfig=False, stations = True)
tmpfield2.plot('v', vmin=2.95, vmax=3.05, showfig=True, stations = True)
# # tmpfield3.plot('v', vmin=2.95, vmax=3.05, showfig=True)
# tmpfield.plot('lplc', vmin=-0.001, vmax=0.001, showfig=False)
# tmpfield2.plot('lplc', vmin=-0.0005, vmax=0.0005, showfig=True)

tmpfield2.Zarr = tmpfield2.Zarr - tmpfield.Zarr
tmpfield2.mask = abs(tmpfield2.Zarr) > 2.
tmpfield2.plot('z', vmin = -2., vmax = 2., showfig=True, stations = True)


# 
# # 
# # 
# tmpfield.interp_surface(workingdir='./temp_working', outfname='TA.H20K_10sec')
# evid = 'TA.H20K'
# channel = 'ZZ'
# tmpfield.check_curvature(workingdir='./temp_working', outpfx=evid+'_'+channel+'_')
# cdist = 250.
# tmpfield.eikonal(workingdir='./temp_working', inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=cdist)



