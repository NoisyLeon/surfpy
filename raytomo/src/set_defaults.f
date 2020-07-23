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
      ireje=0
      irejd=0
      ireg=1
      ires=0
      isymb=1
      iresid=0
      icoord=1
      icoout=1
      ianiz=0
      imodel=0
      idresol=0
      iresol=0
      itomo=0
      iold=0
      itrace=0
      ipoint=0
c**   alpha setup *******************
      istrip = 0
      alsi(1,1)=1400.
      alsi(1,2)=1.
      alsi(1,3)=200.
      alsi(1,4)=200.
      alsi(1,5)=8.
      alsi(2,1)=800.
      alsi(2,2)=1.
      alsi(2,3)=500.
      alsi(2,4)=500.
      alsi(2,5)=8.
      alsi(3,1)=1200.
      alsi(3,2)=1.
      alsi(3,3)=500.
      alsi(3,4)=500.
      alsi(3,5)=8.
      x_zone=2.0d0
      re_la='R'
      gr_pha='G'
      wexp=1.0
      dlat0=-20.
      dlat_n=89.
c*****************
      s_lat=1.
      s_lon=1.
c*****************
      d_lat=1.0
      dlon0=0.
      dlon_n=359.
      d_lon=1.0
      slat0=dlat0
      slat_n=dlat_n
      slon0=dlon0
      slon_n=dlon_n
      eqlat0=30.
      eqlon0=45.
      eqlat_n=30.
      eqlon_n=85.
      nwavep=5
      pf=90.0d0
      pl=0.0d0
      dump1=-0.147130
      dump2=-0.147130
c***     End Constants ******************************
      lsele=LOGI(isele)
      lwght=LOGI(iwght)
      lres= LOGI(ires)
      lsymb=LOGI(isymb)
      laniz=LOGI(ianiz)
      lreje=LOGI(ireje)
      lrejd=LOGI(irejd)
      lmodel=LOGI(imodel)
      ldresol=LOGI(idresol)
      lresid=LOGI(iresid)
      lcoord=LOGI(icoord)
      lcoout=LOGI(icoout)
      ltomo=LOGI(itomo)
      lold=LOGI(iold)
      ltrace=LOGI(itrace)
      lpoint=LOGI(ipoint)
c    main grid ==================================
      cell=2.0
      n_pnt1=SQRT(2.*pi/3.)*180./pi/2.
      n_pnt1=n_pnt1/2
      n_pnt=SQRT(2.*pi/3.)*180./pi/cell
      n_pnt=n_pnt/2
      wexp=FLOAT(2*n_pnt1 +1)/FLOAT(2*n_pnt+1)
c    anisotropy grid ==================================
      cell_a=7.0
      n_pnt1=SQRT(2.*pi/3.)*180./pi/2.
      na_pnt=n_pnt1/2
      na_pnt=SQRT(2.*pi/3.)*180./pi/cell_a
      na_pnt=na_pnt/2
c====================================================
      step0=.25d0
      fname1='contour.ctr'
      fname2='model_map.ctr'
      fname5='PREM.MODEL'
      return 
      end
