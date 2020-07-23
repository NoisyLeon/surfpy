C   CONTOUR SOFTWARE =======================================
      SUBROUTINE TRASS(n_pntt,na_pntt)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      include "linea.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    z0(3),z1(3),vn(3),crdt(3,1000)
      real*8    ff0,fl0,xf0,xl0,dnorm
      integer*4 lcrdt,lnet,net(2,1000),lnblnk
      integer*4 i,j,k,inter,n_pntt,na_pntt,ndk,nmin,nmaxx
      logical   tf
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      COMMON/TRACE/z0,vn,lcrdt,lnet,crdt,net

      CALL CELL4(n_pntt)
      if(ianiz.ne.0) CALL CELLA(na_pntt)
c
c Check if file exists
c
      inquire(file=fname1,exist=tf)
      if(.not.tf) then
        i=lnblnk(fname1)
        write(*,*) '(R0010) File ',fname1(1:i), ' does not exists'
        STOP
      endif
c
c   Read Input Data
c
      open(unit=15,file=fname1,status='OLD')
      read(15,*) fl0,ff0
      z0(1)=DCOS(ff0*drad)*DCOS(fl0*drad)
      z0(2)=DCOS(ff0*drad)*DSIN(fl0*drad)
      z0(3)=DSIN(ff0*drad)
      write(*,"(' Point outside contour: Lat= ',f7.3,' Lon= ',f8.3,i5)")
     +ff0,fl0,nd
      write(8,"(' Point outside contour: Lat= ',f7.3,' Lon= ',f8.3)")
     +ff0,fl0
      read(15,*)lcrdt
      do 10 i=1,lcrdt
      read(15,*) xl0,xf0
      crdt(1,i)=DCOS(xf0*drad)*DCOS(xl0*drad)
      crdt(2,i)=DCOS(xf0*drad)*DSIN(xl0*drad)
   10 crdt(3,i)=DSIN(xf0*drad)
      read(15,*)lnet
      do 20 i=1,lnet
      read(15,*)net(1,i),net(2,i)
   20 continue
c
c Set up local and external tables for isotropic cell
c
      ndk=nd+1
      nall=0
      nm=0
      do 30 i=1,ndk
      do 30 j=1,ndk
      do 30 k=1,ndk
      nmin=min0(i,j,k)
      nmaxx=max0(i,j,k)
      if(nmin.ne.1.and.nmaxx.ne.ndk) goto 30
      z1(1)=dr(i)
      z1(2)=dr(j)
      z1(3)=dr(k)
      CALL NORM(z1,z1)
      CALL SEGM(z1,inter,dpi)
      nall=nall+1
      icrpnt(nall)=i*1000000+j*1000+k
      CALL SQUARE(i,j,k,dpx(4,nall))
      dnorm=dr(i)*dr(i)+dr(j)*dr(j)+dr(k)*dr(k)
      dnorm=DSQRT(dnorm)
      dpx(1,nall)=dr(i)/dnorm
      dpx(2,nall)=dr(j)/dnorm
      dpx(3,nall)=dr(k)/dnorm
      ilocr(nall)=0
      if(inter.eq.1) then
      nm=nm+1
      ioutr(nm)=nall
      ilocr(nall)=nm
      endif
   30 continue
      close(15)
c
c Set up local and external tables for anisotropic cell
c
      nma=0
      if(ianiz.ne.0) then
      ndk=nda+1
      nalla=0
      do 31 i=1,ndk
      do 31 j=1,ndk
      do 31 k=1,ndk
      nmin=min0(i,j,k)
      nmaxx=max0(i,j,k)
      if(nmin.ne.1.and.nmaxx.ne.ndk) goto 31
      z1(1)=dra(i)
      z1(2)=dra(j)
      z1(3)=dra(k)
      CALL NORM(z1,z1)
      CALL SEGM(z1,inter,dpi)
      nalla=nalla+1
      icrpnta(nalla)=i*1000000+j*1000+k
      CALL SQUAREA(i,j,k,dpxa(4,nalla))
      dnorm=dra(i)*dra(i)+dra(j)*dra(j)+dra(k)*dra(k)
      dnorm=DSQRT(dnorm)
      dpxa(1,nalla)=dra(i)/dnorm
      dpxa(2,nalla)=dra(j)/dnorm
      dpxa(3,nalla)=dra(k)/dnorm
      ilocra(nalla)=0
      if(inter.eq.1) then
      nma=nma+1
      ioutra(nma)=nalla
      ilocra(nalla)=nma
      endif
   31 continue
      endif
      return
      end
c************************************************************
      SUBROUTINE SEGM(z1,inter,pi)
      IMPLICIT NONE
c Type definitions ++++++++++++++++++++++++++++++++++++++
      real*8    z0(3),z1(3),vn(3),crdt(3,1000)
      integer*4 lcrdt,lnet,net(2,1000),isg(1000)
      real*8    d,x(3),y(3),r(3),rn(3),sol(3),eps,delta,pi
      integer*4 i,j,inter,lsol,ma,na
c End type definitions ++++++++++++++++++++++++++++++++++
      common/trace/z0,vn,lcrdt,lnet,crdt,net
      data eps/1.d-6/
      CALL NVECT(z0,z1,r,vn,d)
      d=z0(1)*z1(1)+z0(2)*z1(2)+z0(3)*z1(3)
      if(d.ge.1.0d0) d=dsign(1.0d0,d)
      delta=DACOS(d)
      do 10 i=1,lcrdt
      d=crdt(1,i)*vn(1)+crdt(2,i)*vn(2)+crdt(3,i)*vn(3)
      isg(i)=-1
      if(d.ge.0.) isg(i)=1
      if(DABS(d).lt.eps) isg(i)=10
   10 continue
c********** determination intersections ********************
      lsol=0
      do 90 i=1,lnet
      ma=net(1,i)
      na=net(2,i)
      if(isg(ma)*isg(na).gt.0) goto 90
      if(isg(ma).ne.10) goto 30
      do 20 j=1,3
   20 sol(j)=crdt(j,ma)
      goto 80
   30 if(isg(na).ne.10) goto 50
      do 40 j=1,3
   40 sol(j)=crdt(j,na)
      goto 80
   50 do 60 j=1,3
      x(j)=crdt(j,ma)
   60 y(j)=crdt(j,na)
      CALL NVECT(x,y,r,rn,d)
      CALL NVECT(rn,vn,r,sol,d)
      CALL VECT(x,sol,r)
      if(r(1)*rn(1)+r(2)*rn(2)+r(3)*rn(3).gt.0.d0) goto 80
c**      if(x(1)*sol(1)+x(2)*sol(2)+x(3)*sol(3).gt.0.) goto 80
      do 70 j=1,3
   70 sol(j)=-sol(j)
   80 lsol=lsol+1
      d=sol(1)*z0(1)+sol(2)*z0(2)+sol(3)*z0(3)
      if(DABS(d).ge.1.0d0) d=dsign(1.0d0,d)
      d=DACOS(d)
      CALL VECT(z0,sol,r)
      if(vn(1)*r(1)+vn(2)*r(2)+vn(3)*r(3).lt.0.) d=pi*2.0d0-d
      if(DABS(d-delta).lt.eps) goto 90
      if(d.gt.delta) lsol=lsol-1
   90 continue
      inter=lsol-lsol/2*2
      return
      end
c************************************************************
      SUBROUTINE SQUARE(i,j,k,sq)
      IMPLICIT NONE
      include "line.h"
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
      if(nmin.ne.1.and.nmax.ne.(nd+1)) goto 10
      if(nmin.lt.1.or.nmax.gt.(nd+1)) goto 10
      if(IABS(ix-i)+IABS(iy-j)+IABS(iz-k).ne.2) goto 10
      ic=ic+1
      iver(ic,1)=ix
      iver(ic,2)=iy
      iver(ic,3)=iz
   10 continue
      s=0.d0
      do 50 ii=2,ic
      do  40 kk=1,3
      if(iver(1,kk).eq.iver(ii,kk)) goto 40
      do 20 ll=1,3
      if(iver(ii,ll)-iver(1,ll).eq.0) lll=ll
   20 continue
      if(iver(1,lll).ne.1.and.iver(1,lll).ne.(nd+1)) goto 40
      do 30 jj=1,3
   30 ia(jj)=iver(1,jj)
      ia(kk)=iver(ii,kk)
      a(1)=dr(iver(1,1))
      a(2)=dr(iver(1,2))
      a(3)=dr(iver(1,3))
      CALL NORM(a,a)
      b(1)=dr(iver(ii,1))
      b(2)=dr(iver(ii,2))
      b(3)=dr(iver(ii,3))
      CALL NORM(b,b)
      c(1)=dr(ia(1))
      c(2)=dr(ia(2))
      c(3)=dr(ia(3))
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
