      SUBROUTINE SET_DEFAULTS
      IMPLICIT NONE
      include "tomo.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*4 d_lat,d_lon
      integer*4 n_pnt1
      logical LOGI
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      write(*,*) 'Setting defaults.....'
      iwght=0
      ipath=NINP
      isele=0
      isymb=1
      ireje=0
      ires=0
      imodel=0
      iresid=0
      icoord=1
      icoout=1
      alph1(1)=1.0 
      alph1(2)=200. 
      alph1(3)=1400.
      alph1(4)=200.
      wexp=1.0
c lat print windows
      dlat0=-20.
      dlat_n=89.
      s_lat=1.
      d_lat=1.0
c lon print windows
      dlon0=0.
      dlon_n=359.
      s_lon=1.
      d_lon=1.0
c
      slat0=dlat0
      slat_n=dlat_n
      slon0=dlon0
      slon_n=dlon_n
c
      eqlat0=30.
      eqlon0=45.
      eqlat_n=30.
      eqlon_n=85.
      irejd=0
      nwavep=3
      lsele= LOGI(isele)
      lwght= LOGI(iwght)
      lres=  LOGI(ires)
      lreje= LOGI(ireje)
      lsymb= LOGI(isymb)
      lrejd= LOGI(irejd)
      lmodel=LOGI(imodel)
      lresid=LOGI(iresid)
      lcoord=LOGI(icoord)
      lcoout=LOGI(icoout)
      cell=2.0
      n_pnt1=SQRT(2.*pi/3.)*180./pi/2.
      n_pnt1=n_pnt1/2
      n_pnt=SQRT(2.*pi/3.)*180./pi/cell
      n_pnt=n_pnt/2
      wexp=FLOAT(2*n_pnt1 +1)/FLOAT(2*n_pnt+1)
      step0=.25d0
      pow=8.0d0
      fname1='contour.ctr'
      fname2='model_map.ctr'
      return 
      end
