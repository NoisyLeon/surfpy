c*****************************************************************
c* Subroutine CELLA:   Creates COMMON block /mata/ containing
c* all necessary information for anisotropy cellular grid 
c* map on the sphere.
c* Call from command line: cell n_pntt lat_normal lon_normal azimuth
c* n_pntt             - define the total number of cells by formula:
c*                     n=6*(2*n_pntt +1)*(2*n_pntt +1)
c*****************************************************************
      SUBROUTINE CELLA(na_pntt)
      IMPLICIT NONE
      include "line.h"
      include "linea.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    def
      real*8    dlincell,dsqcell,dd
      integer*4 i,k,na_pntt
c END TYPE definitions ++++++++++++++++++++++++++++++++++
c************* Input parameters**********************************
      k=na_pntt
c************* END input parameters**********************************
c* Spere radius sqrt(3)
      nda=2*k+1
      nnda=nda*nda
      do 10 i=0,k
      def=dpi*DFLOAT((2*i+1)*(2*i+1))/DFLOAT((2*k+1)*(2*k+1))/6.0d0
      dta(k+2+i)=DACOS(DCOS(def)/(1.d0 + DSIN(def)))
      dra(k+2+i)=DTAN(dta(k+2+i))/DSQRT(2.0d0)
   10 dra(k+1-i)=-dra(k+2+i)
      dd=360.d0/2.d0/dpi
      dd=dd*dd
      dsqcell=4.0d0*dpi*dd/FLOAT(nda)/FLOAT(nda)/6.0d0
      dlincell=DSQRT(dsqcell)
      write(8,'("Anisotropy grid cells:")')
      write(*,'("Anisotropy grid cells:")')
      write(*,'(" Average square of cells is:" f17.10," (deg**2)")') dsqcell
      write(8,'(" Average square of cells is:" f17.10," (deg**2)")') dsqcell
      write(*,'(" Average length of cells is:" f17.10," (deg)")') dlincell
      write(8,'(" Average length of cells is:" f17.10," (deg)")') dlincell
      return
      end
c********************************************************
      SUBROUTINE SQUAREA(i,j,k,sq)
      IMPLICIT NONE
      include "line.h"
      include "linea.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8 sq,s,st,a(3),b(3),c(3),d(3),e(3),ee(3),r
      integer*4 i,j,k,ia(3),ic,ii,ix,iy,iz,iver(10,3),jj
      integer*4 kk,ll,lll,nmin,nmax
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      iver(1,1)=i
      iver(1,2)=j
      iver(1,3)=k
      ic=1
      do 10 ii=1,3
      do 10 jj=1,3
      do 10 kk=1,3
      ix=i+ii-2
      iy=j+jj-2
      iz=k+kk-2
      nmin=min0(ix,iy,iz)
      nmax=max0(ix,iy,iz)
      if(nmin.ne.1.and.nmax.ne.(nda+1)) goto 10
      if(nmin.lt.1.or.nmax.gt.(nda+1)) goto 10
      if(IABS(ix-i)+IABS(iy-j)+IABS(iz-k).ne.2) goto 10
      ic=ic+1
      iver(ic,1)=ix
      iver(ic,2)=iy
      iver(ic,3)=iz
   10 continue
cxx   write(*,'(4i3/10(3i3,2x))') ic,i,j,k,((iver(l,ll),ll=1,3),l=1,ic)
      s=0.d0
      do 50 ii=2,ic
      do  40 kk=1,3
      if(iver(1,kk).eq.iver(ii,kk)) goto 40
      do 20 ll=1,3
      if(iver(ii,ll)-iver(1,ll).eq.0) lll=ll
   20 continue
      if(iver(1,lll).ne.1.and.iver(1,lll).ne.(nda+1)) goto 40
      do 30 jj=1,3
   30 ia(jj)=iver(1,jj)
      ia(kk)=iver(ii,kk)
      a(1)=dra(iver(1,1))
      a(2)=dra(iver(1,2))
      a(3)=dra(iver(1,3))
      CALL NORM(a,a)
      b(1)=dra(iver(ii,1))
      b(2)=dra(iver(ii,2))
      b(3)=dra(iver(ii,3))
      CALL NORM(b,b)
      c(1)=dra(ia(1))
      c(2)=dra(ia(2))
      c(3)=dra(ia(3))
      st=0.d0
      CALL NORM(c,c)
      CALL NVECT(a,b,ee,d,r)
      CALL NVECT(c,b,ee,e,r)
      st=st+DACOS(d(1)*e(1)+d(2)*e(2)+d(3)*e(3))
      CALL NVECT(a,c,ee,d,r)
      CALL NVECT(b,c,ee,e,r)
      st=st+DACOS(d(1)*e(1)+d(2)*e(2)+d(3)*e(3))
      CALL NVECT(b,a,ee,d,r)
      CALL NVECT(c,a,ee,e,r)
      st=st+DACOS(d(1)*e(1)+d(2)*e(2)+d(3)*e(3))
      st=st-dpi
      s=s+st
   40 continue
   50 continue
      sq=s/(drad*drad*4.d0)
      return
      end
