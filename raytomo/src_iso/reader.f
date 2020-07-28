C###############################################################
      SUBROUTINE READER
      IMPLICIT NONE
C    READ INPUT DATA AND TRANSFORM TO GEOCENTRICAL COORDINATES--
      include "tomo.h"
C----------------------------------------------------------------
C------ipath - maximal	permitted number of paths
C------output TE0,TE-latitudes; FI0,FI - longitudes; WEIGHT, 
C------N number of paths for processing--------------------------
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*4    f_lam0,f_lam1
      integer*4 i,icount,np,npath
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      npath=0
      np=0
      icount=0
      DO  I=1,ipath
   10 continue
      IF(LWGHT)
     *  read(7,*,END=20)IIII(I),TE0(I),FI0(I),TE(I),FI(I),T(I),WEIGHT(I)
      IF(.NOT.LWGHT) then
        read(7,*,END=20) IIII(I),TE0(I),FI0(I),TE(I),FI(I),T(I)
        WEIGHT(I)=1.0
      endif
      if(FI0(I).lt.0.0)FI0(I)=FI0(I)+360.0
      if(FI(I).lt.0.0) FI(I) =FI(I) +360.0
      np=np+1
      if(lsele) then
C-----to skip  paths which end point lay outside the region--------
        if (TE0(i).lt.slat0.or.TE0(i).gt.slat_n) go to 10
        if (TE(i).lt.slat0.or.TE(i).gt.slat_n)   go to 10
        f_lam0=FI0(i)
        f_lam1=FI(i)
        if(f_lam0.lt.slon0)  f_lam0=f_lam0+360.0
        if(f_lam0.gt.slon_n) f_lam0=f_lam0-360.0
        if (f_lam0.lt.slon0.or.f_lam0.gt.slon_n) go to 10
        if(f_lam1.lt.slon0)  f_lam1=f_lam1+360.0
        if(f_lam1.gt.slon_n) f_lam1=f_lam1-360.0
        if (f_lam1.lt.slon0.or.f_lam1.gt.slon_n)   go to 10
      endif
      npath=npath+1
      enddo
   20 N=npath 
      if(I.ge.ipath) then
      write(*,*)'WRANING !!!! NINP parameter is to small, used only',
     +ipath,' lines of input data'
      write(8,*)'WRANING !!!! NINP parameter is to small, used only',
     +ipath,' lines of input data'
      endif
      write(*,*) 'There were', np, ' paths; only ',N,' paths left'
      write(8,1000) np,N
 1000 format(65(1H=)/" There were", i7, " paths; only ",i7," paths left")
C------to convert geographical latitudes into geocentrical latitudes
      do 3 I=1,N
        if(lcoord) then
          TE0(I)= ATAN(GEO*TAN(CONST*TE0(I)))/CONST
          TE(I) = ATAN(GEO*TAN(CONST*TE(I)))/CONST 
        endif
    3 continue 
      INDEX=0                                                     
      close (7)
      return
      end                                                       
