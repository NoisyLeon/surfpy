c***************************************************************
c Read input data
c----------------------------------------------------------------
c   NMAX   - maximal permitted number of paths;
c   TE0,TE - latitudes; 
c   FI0,FI - longitudes; 
c   WEIGHT - weight of measurement;
c   IRAY   - array of idicators: =1 - R1/L1, =2 R2/L2
c   N      - number of paths for processing
c***************************************************************
      SUBROUTINE READER
      IMPLICIT NONE
      include "tomo.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*4    f_lam0,f_lam1
      integer*4 i,icount,np,npath
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      npath=0
      np=0
      icount=0
      do I=1,NMAX
   10 continue
      if(iold.eq.0) then
        if(lwght) then
          read(7,*,END=20)IIII(I),TE0(I),FI0(I),TE(I),FI(I),T(I),
     +    WEIGHT(I)
        else
          read(7,*,END=20) IIII(I),TE0(I),FI0(I),TE(I),FI(I),T(I)
          WEIGHT(I)=1.0
        endif
        IRAY2(I)=1
      else
        read(7,*,END=20)IIII(I),TE0(I),FI0(I),TE(I),FI(I),T(I),
     +  WEIGHT(I),IRAY2(I)
        if(.not.lwght) WEIGHT(I)=1.0
      endif
      if(FI0(I).lt.0.0)FI0(I)=FI0(I)+360.0
      if(FI(I).lt.0.0) FI(I) =FI(I) +360.0
      np=np+1
      if(lsele) then
c
c  skip  paths which end point lay outside the region
c
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
      if(I.ge.NMAX) then
      write(*,*)'WRANING !!!! NINP parameter is to small, used only',
     +NMAX,' lines of input data'
      write(8,*)'WRANING !!!! NINP parameter is to small, used only',
     +NMAX,' lines of input data'
      endif
      PRINT*, 'THERE WERE', np, ' PATHS; only ',N,' PATHS LEFT'
      WRITE(8,'( "THERE WERE", i7, " PATHS; only ",i7," PATHS LEFT")') np,N
c
c  convert geographical latitudes into geocentrical latitudes
c
      if(lcoord) then
        do I=1,N
          TE0(I)= ATAN(GEO*TAN(CONST*TE0(I)))/CONST
          TE(I) = ATAN(GEO*TAN(CONST*TE(I)))/CONST 
        enddo 
      endif
      index=0                                                     
      close (7)
      return
      end                                                       
