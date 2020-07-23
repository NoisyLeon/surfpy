c****** CELLULAR PAAMETERS, (isotr. part) ************************
      integer*4 NCELL
c     parameter (NCELL=64000)
      parameter (NCELL=4900000)
c---------------COMMON /mat/---------------------------------------
      integer*4    nnd,nd
      real*8 drad,dpi,dt(901),dr(902)
      common /mat/nnd,nd,dpi,drad,dt,dr
c---------------COMMON /win/---------------------------------------
      integer*4    n0,m0,nnn,mmm
      common /win/ n0,m0,nnn,mmm
c---------------COMMON /tras/--------------------------------------
      integer*4     nall,nm,icrpnt,ilocr,ioutr
      real*8        dpx,dlcell
      common /tras/nall,nm,icrpnt(NCELL),ilocr(NCELL),ioutr(NCELL),
     *dlcell, dpx(4,NCELL)
c*****************************************************************
