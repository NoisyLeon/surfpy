# -*- coding: utf-8 -*-
"""
hdf5 for plotting eikonal/Helmholtz results
    
:Copyright:
    Author: Lili Feng
    email: lfeng1011@gmail.com
"""
import surfpy.eikonal.tomobase as tomobase

try:
    import surfpy.eikonal._eikonal_funcs as _eikonal_funcs
except:
    import _eikonal_funcs
    
try:
    import surfpy.eikonal._grid_class as _grid_class
except:
    import _grid_class
    
import surfpy.cpt_files as cpt_files
cpt_path    = cpt_files.__path__._path[0]

import numpy as np
import numpy.ma as ma
import obspy
import copy
import os
if os.path.isdir('/home/lili/anaconda3/share/proj'):
    os.environ['PROJ_LIB'] = '/home/lili/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt


class ploth5(tomobase.baseh5):
    """h5 database for plotting
    """
    def _get_basemap(self, projection='lambert', resolution='i'):
        """get basemap for plotting results
        """
        plt.figure()
        minlon          = self.minlon
        maxlon          = self.maxlon
        minlat          = self.minlat
        maxlat          = self.maxlat
        
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon,
                      urcrnrlon=maxlon, lat_ts=0, resolution=resolution)
            # m.drawparallels(np.arange(minlat,maxlat,dlat), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(minlon,maxlon,dlon), labels=[1,0,0,1])
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,1,1,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,1,1,0])
            # m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            # m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        elif projection=='regional_ortho':
            m      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            # m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
            #             llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/2., urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth((lat_centre+minlat)/2., minlon, (lat_centre+minlat)/2., maxlon-15) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat-6, minlon) # distance is in m
            m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                        lat_1=minlat, lat_2=maxlat, lon_0=lon_centre-2., lat_0=lat_centre+2.4)
            m.drawparallels(np.arange(-80.0,80.0,5.0), linewidth=1, dashes=[2,2], labels=[0,0,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,0,0], fontsize=15)
        m.drawcountries(linewidth=1.)
        #################
        # coasts = m.drawcoastlines(zorder=100,color = '0.9',linewidth = 0.01)
        # 
        # # Exact the paths from coasts
        # coasts_paths = coasts.get_paths()
        # 
        # # In order to see which paths you want to retain or discard you'll need to plot them one
        # # at a time noting those that you want etc.
        # poly_stop = 10
        # for ipoly in xrange(len(coasts_paths)):
        #     print ipoly
        #     if ipoly > poly_stop:
        #         break
        #     r = coasts_paths[ipoly]
        #     # Convert into lon/lat vertices
        #     polygon_vertices = [(vertex[0],vertex[1]) for (vertex,code) in
        #                         r.iter_segments(simplify=False)]
        #     px = [polygon_vertices[i][0] for i in xrange(len(polygon_vertices))]
        #     py = [polygon_vertices[i][1] for i in xrange(len(polygon_vertices))]
        #     m.plot(px,py,'k-',linewidth=2.)
        ######################
        m.drawstates(linewidth=1.)
        # m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        return m
    
    def plot(self, runid, datatype, period, semfactor=2., Nthresh=None, merged=False, clabel='', cmap='surf',\
             projection='lambert', hillshade = False, vmin = None, vmax = None, showfig = True, v_rel = None):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runtype         - type of run (0 - smooth run, 1 - quality controlled run)
        runid           - id of run
        datatype        - datatype for plotting
        period          - period of data
        sem_factor      - factor multiplied to get the finalized uncertainties
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        geopolygons     - geological polygons for plotting
        vmin, vmax      - min/max value of plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        dataid          = 'tomo_stack_'+str(runid)
        ingroup         = self[dataid]
        pers            = self.attrs['period_array']
        self._get_lon_lat_arr()
        if not period in pers:
            raise KeyError('!!! period = '+str(period)+' not included in the database')
        pergrp          = ingroup['%g_sec'%( period )]
        if datatype == 'vel' or datatype=='velocity' or datatype == 'v':
            datatype    = 'vel_iso'
        elif datatype == 'sem' or datatype == 'un' or datatype == 'uncertainty':
            datatype    = 'vel_sem'
        elif datatype == 'std':
            datatype    = 'slowness_std'
        try:
            data        = pergrp[datatype][()]
        except:
            outstr      = ''
            for key in pergrp.keys():
                outstr  +=key
                outstr  +=', '
            outstr      = outstr[:-1]
            raise KeyError('Unexpected datatype: '+datatype+ ', available datatypes are: '+outstr)
        if datatype=='Nmeasure_aniso':
            factor      = ingroup.attrs['grid_lon'] * ingroup.attrs['grid_lat']
        else:
            factor      = 1
        if datatype=='Nmeasure_aniso' or datatype=='unpsi' or datatype=='unamp' or datatype=='amparr':
            mask        = pergrp['mask_aniso'][()] + pergrp['mask'][()]
        else:
            mask        = pergrp['mask'][()]
        if not (Nthresh is None):
            Narr        = pergrp['NmeasureQC'][()]
            mask        += Narr < Nthresh
        if (datatype=='Nmeasure' or datatype=='NmeasureQC') and merged:
            mask        = pergrp['mask_eikonal'][()]
        if datatype == 'vel_sem':
            data        *= 1000.*semfactor
        mdata           = ma.masked_array(data/factor, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection = projection)
        x, y            = m(self.lonArr, self.latArr)
        try:
            import pycpt
            if os.path.isfile(cmap):
                cmap    = pycpt.load.gmtColormap(cmap)
                # cmap    = cmap.reversed()
            elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
        except:
            pass
        ###################################################################
        # # # if hillshade:
        # # #     from netCDF4 import Dataset
        # # #     from matplotlib.colors import LightSource
        # # #     etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
        # # #     etopo       = etopodata.variables['z'][:]
        # # #     lons        = etopodata.variables['x'][:]
        # # #     lats        = etopodata.variables['y'][:]
        # # #     ls          = LightSource(azdeg=315, altdeg=45)
        # # #     # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
        # # #     etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
        # # #     # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
        # # #     ny, nx      = etopo.shape
        # # #     topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
        # # #     m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
        ###################################################################
        
        if v_rel is not None:
            mdata       = (mdata - v_rel)/v_rel * 100.
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap = cmap, shading = 'gouraud', vmin = vmin, vmax = vmax, alpha=.5)
        else:
            im          = m.pcolormesh(x, y, mdata, cmap = cmap, shading = 'gouraud', vmin = vmin, vmax = vmax)
            # m.contour(x, y, mask_eik, colors='white', lw=1)
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.])
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%', ticks=[20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
        # cb          = m.colorbar(im, "bottom", size="5%", pad='2%', ticks=[4.0, 4.1, 4.2, 4.3, 4.4])
        cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
        cb.set_label(clabel, fontsize=40, rotation=0)
        # cb.outline.set_linewidth(2)
        plt.suptitle(str(period)+' sec', fontsize=20)
        cb.ax.tick_params(labelsize = 20)
        print ('=== plotting data from '+dataid)
        if showfig:
            plt.show()
        return
    
    
    def plot_psi(self, runid, period, factor=5, normv=5., width=0.005, ampref=0.02, datatype='',
            scaled=False, masked=True, clabel='', cmap='surf', projection='lambert', vmin=None, vmax=None, showfig=True):
        """plot maps of fast axis from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        runid           - id of run
        period          - period of data
        factor          - factor of intervals for plotting
        normv           - value for normalization
        width           - width of the bar
        ampref          - reference amplitude (default - 0.05 km/s)
        datatype        - type of data to plot as background color
        masked          - masked or not
        clabel          - label of colorbar
        cmap            - colormap
        projection      - projection type
        hillshade       - produce hill shade or not
        vmin, vmax      - min/max value of plotting
        showfig         - show figure or not
        =================================================================================================================
        """
        dataid  = 'tomo_stack_'+str(runid)
        self._get_lon_lat_arr()
        ingroup = self[dataid]
        # period array
        pers    = self.pers
        if not period in pers:
            raise KeyError('period = '+str(period)+' not included in the database')
        pergrp  = ingroup['%g_sec'%( period )]
        # get the amplitude and fast axis azimuth
        psi     = pergrp['psiarr'][()]
        amp     = pergrp['amparr'][()]
        mask    = pergrp['mask_aniso'][()] + pergrp['mask'][()]
        # get velocity
        try:
            data= pergrp[datatype][()]
            plot_data   = True
        except:
            plot_data   = False
        #-----------
        # plot data
        #-----------
        m       = self._get_basemap(projection = projection)
        if scaled:
            U       = np.sin(psi/180.*np.pi)*amp/ampref/normv
            V       = np.cos(psi/180.*np.pi)*amp/ampref/normv
            Uref    = np.ones(self.lonArr.shape)*1./normv
            Vref    = np.zeros(self.lonArr.shape)
        else:
            U       = np.sin(psi/180.*np.pi)/normv
            V       = np.cos(psi/180.*np.pi)/normv
        # rotate vectors to map projection coordinates
        U, V, x, y  = m.rotate_vector(U, V, self.lonArr, self.latArr, returnxy=True)
        if scaled:
            Uref, Vref, xref, yref  = m.rotate_vector(Uref, Vref, self.lonArr, self.latArr, returnxy=True)
        #--------------------------------------
        # plot background data
        #--------------------------------------
        if plot_data:
            print ('=== plotting data: '+datatype)
            try:
                import pycpt
                if os.path.isfile(cmap):
                    cmap    = pycpt.load.gmtColormap(cmap)
                elif os.path.isfile(cpt_path+'/'+ cmap + '.cpt'):
                    cmap    = pycpt.load.gmtColormap(cpt_path+'/'+ cmap + '.cpt')
            except:
                pass
            if masked:
                data    = ma.masked_array(data, mask = mask )
            im          = m.pcolormesh(x, y, data, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
            cb          = m.colorbar(im, "bottom", size="5%", pad='2%')
            cb.set_label(clabel, fontsize=40, rotation=0)
            # cb.outline.set_linewidth(2)
            plt.suptitle(str(period)+' sec', fontsize=20)
            cb.ax.tick_params(labelsize=40)
            # cb.set_alpha(1)
            cb.draw_all()
            cb.solids.set_edgecolor("face")
        #--------------------------------------
        # plot fast axis
        #--------------------------------------
        x_psi       = x.copy()
        y_psi       = y.copy()
        mask_psi    = mask.copy()
        if factor!=None:
            x_psi   = x_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
            y_psi   = y_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
            U       = U[0:self.Nlat:factor, 0:self.Nlon:factor]
            V       = V[0:self.Nlat:factor, 0:self.Nlon:factor]
            mask_psi= mask_psi[0:self.Nlat:factor, 0:self.Nlon:factor]
        # Q       = m.quiver(x, y, U, V, scale=30, width=0.001, headaxislength=0)
        if masked:
            U   = ma.masked_array(U, mask=mask_psi )
            V   = ma.masked_array(V, mask=mask_psi )
        Q1      = m.quiver(x_psi, y_psi, U, V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        Q2      = m.quiver(x_psi, y_psi, -U, -V, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='k')
        if scaled:
            mask_ref        = np.ones(self.lonArr.shape)
            Uref            = ma.masked_array(Uref, mask=mask_ref )
            Vref            = ma.masked_array(Vref, mask=mask_ref )
            m.quiver(xref, yref, Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
            m.quiver(xref, yref, -Uref, Vref, scale=20, width=width, headaxislength=0, headlength=0, headwidth=0.5, color='g')
        plt.suptitle(str(period)+' sec', fontsize=20)
        if showfig:
            plt.show()
        return
    
    