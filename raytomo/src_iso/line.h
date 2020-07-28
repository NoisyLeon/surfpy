c****** CELLULAR PAAMETERS ****************************************
      integer*4 NCELL
c      parameter (NCELL=64000*10)
      parameter (NCELL=1000000)
c---------------COMMON /mat/---------------------------------------
      integer*4    nnd,nd
      real*8       drad,dpi,dt(601),dr(602)
      common /mat/ nnd,nd,dpi,drad,dt,dr
c---------------COMMON /win/---------------------------------------
      integer*4    n0,m0,nnn,mmm
      common /win/ n0,m0,nnn,mmm
c---------------COMMON /tras/--------------------------------------
      integer*4     nall,nm,icrpnt,ilocr,ioutr
      real*8        dpx
      common /tras/ nall,nm,icrpnt(NCELL),ilocr(NCELL),ioutr(NCELL),
     +dpx(4,NCELL)
c*******************************************************************
