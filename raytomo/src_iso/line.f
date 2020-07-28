C*******************************************************************
      SUBROUTINE EGAUSS(lidd,nlidd,iin,nmm,ff,dens,wden,sigmw,tresh)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    swgt,wgt,tresh,s,s1,sigmw,x1(3),wgta
      real*4    wden,ff(NRAZ),dens(NRAZ)
      integer*4 nlidd,lidd(NRAZ)
      integer*4 i,i1,i2,i3,iin,ilo,iou,j,nmm
c END TYPE definitions ++++++++++++++++++++++++++++++++++
C******************************************************
c*** exp(-s**2/2/sigma**2) < 10**(-pow), tresh=COS(s)
C******************************************************
      wden=0.0
      iou=ioutr(iin)
      i1=icrpnt(iou)/1000000
      i3=icrpnt(iou)-i1*1000000
      i2=i3/1000
      i3=i3-i2*1000
      s1=dr(i1)*dr(i1)+dr(i2)*dr(i2)+dr(i3)*dr(i3)
      s=DSQRT(s1)
      x1(1)=dr(i1)/s
      x1(2)=dr(i2)/s
      x1(3)=dr(i3)/s
      swgt=0.0d0
      do 20 i=1,nall
      s=0.0d0
      do 10 j=1,3
   10 s=s+x1(j)*dpx(j,i)
      if(s.gt.1.d0)s=1.d0
      if(s.lt.tresh) goto 20
      s=DACOS(s)
      wgta=s*s/sigmw/sigmw/2.0d0
      wgt=0.0d0
      if(wgta.lt.30.0d0) then
      wgt=DEXP(wgta)
      wgt=dpx(4,i)/wgt
      endif
      swgt=swgt+wgt
      ilo=ilocr(i)
      if(ilo.ne.0) then
      ff(ilo)=wgt
      wden=wden+dens(ilo)*wgt
      endif
   20 continue
      do 30 j=1,nmm
   30 ff(j)=ff(j)/swgt
      ff(iin)=ff(iin)-1.
      wden=wden/SNGL(swgt)
      do 40 i=1,nmm
      if(ff(i).ne.0.0) then
      nlidd=nlidd+1
      lidd(nlidd)=i
      endif
   40 continue
      return
      end
C******************************************************************
      SUBROUTINE ESTRACE(lidd,nlidd,step,e,nmm,denf,ff,xn,gis,ierr)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      integer*4 nmm
      real*4    denf(nmm),ff(nmm),gis(IAZIM,NRAZ)
      real*8    grid(4,3),w(4),e(3),step,xn(3),azms
      integer*4 lidd(NRAZ),nclo(4),ngr_out(4)
      integer*4 i,j,k,ierr,ncl,ncoin,nlidd
      real*4    RBIMOD
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
      CALL E2PNTS(e,ngr_out,grid)
      CALL TRIAW(ngr_out,grid,e,w,ierr)
      do 20 j=1,4
      nclo(j)=ilocr(ngr_out(j))
      if(nclo(j).eq.0) then
      ierr=1
      return
      endif
   20 continue
      do 10 j=1,4
      if(j.eq.1) ncoin=nclo(1)*nmm+nclo(4)
      ncl=nclo(j)
      if(ncl.gt.nm) then
        write(*,*)'ERROR with local number',nm,ncl
        ierr=1
        return
      endif
      do 30 i=1,nlidd
      if(lidd(i).eq.ncl) goto 40
   30 continue
      nlidd=nlidd+1
      if(nlidd.gt.2000) then
      write(*,*) 'TOO SMALL SIZE OF lidd',nlidd-1
      STOP
      endif
      lidd(nlidd)=ncl
   40 ff(ncl)=ff(ncl)+w(j)*step*R/RBIMOD(e)
      if(ncoin0.ne.ncoin) then 
        denf(ncl)=1.0
          CALL AZI(j,grid,e,xn,azms)
          k=azms/const/(180.0/FLOAT(IAZIM))+1.0d0
          if(k.lt.1) k=1
          if(k.gt.IAZIM) k=IAZIM
        gis(k,ncl)=gis(k,ncl)+1.0
      endif
   10 continue
      ncoin0=ncoin
      return
      end
C******************************************************
      SUBROUTINE EINTEGR(step,e,nmm,sum,fsl,ierr)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      integer*4 nmm
      real*4    fsl(nmm)
      real*8    grid(4,3),w(4),e(3),step,scell
      real*4    sum,RBIMOD
      integer*4 ngr_out(4),nclo,ierr,j
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
      CALL E2PNTS(e,ngr_out,grid)
      do 10 j=1,4
      nclo=ilocr(ngr_out(j))
      if(nclo.lt.1) then
      ierr=1
      return
      endif
   10 w(j)=fsl(nclo)
      CALL TRIAI(grid,e,w,scell)
      scell=scell*step*R/RBIMOD(e)
      sum=scell
      return
      end
C******************************************************
      SUBROUTINE EINTPOL (xk,yk,nmm,fsl,value,ierr)
      IMPLICIT NONE
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      integer*4 nmm
      real*4    fsl(nmm),value
      real*8    grid(4,3),w(4),e(3),scell,xk,yk
      integer*4 ngr_out(4)
      integer*4 ierr,j,nclo
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
      e(1)=DSIN(xk)*DCOS(yk)
      e(2)=DSIN(xk)*DSIN(yk)
      e(3)=DCOS(xk)
      CALL E2PNTS(e,ngr_out,grid)
      do 10 j=1,4
      nclo=ilocr(ngr_out(j))
      if(nclo.lt.1) then
      ierr=1
      return
      endif
   10 w(j)=fsl(nclo)
      CALL TRIAI(grid,e,w,scell)
      value=scell
      return
      end
C******************************************************
      SUBROUTINE E2PNTS(e,ngr_out,grid)
      IMPLICIT NONE
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8 grid(4,3),e(3)
      real*8 dmax,de
      integer*4 ngr_out(4),ne(3),nplane,ndp1,ndm1
      integer*4 ic,icol,ix,iy,iz,index,ishift,ival,j,k
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      equivalence (ne(1),ix),(ne(2),iy),(ne(3),iz)
      dmax=DMAX1(DABS(e(1)),DABS(e(2)),DABS(e(3)))
      nplane=0
      do 30 j=1,3
      if(nplane.eq.0.and.DABS(e(j)).eq.dmax) then
      nplane=j
      de=e(j)/dmax
      if(de.lt.-.5) ne(j)=1
      if(de.gt.0.5)ne(j)=nd+1
      goto 30
      endif
      de=e(j)/dmax
      icol=nd
      do 10 k=2,nd
      if(de.lt.dr(k)) then
      icol=k-1
      goto 20
      endif
   10 continue
   20 ne(j)=icol
   30 continue
      if(nplane.eq.0) then
      write(*,*)'NO CELL FOUND'
      stop
      endif
      ic=0 
c****** Get integer coordinates *************************
      ishift=1000000
      do 40 j=1,3
      if(j.ne.nplane) then
      if(ic.lt.1) then
      ic=ic+1
      ngr_out(ic)=1000000*ne(1)+1000*ne(2)+ne(3)
      ic=ic+1
      ngr_out(ic)=1000000*ne(1)+1000*ne(2)+ne(3)+ishift
      else
      ic=ic+1
      ngr_out(ic)=ngr_out(ic-2)+ishift
      ic=ic+1
      ngr_out(ic)=ngr_out(ic-2)+ishift
      endif
      endif
      ishift=ishift/1000
   40 continue
      do 50 j=1,4
      ival=ngr_out(j)
c****** Searching by analitic expression ******************
      ix=ival/1000000
      iz=ival-ix*1000000
      iy=iz/1000
      iz=iz-iy*1000
      ndp1=nd+1
      ndm1=nd-1
      if(ix.eq.1) then
      index=(iy-1)*ndp1+iz
      else if(ix.eq.ndp1) then
      index=ndp1*ndp1+4*ndm1*nd+(iy-1)*ndp1+iz
      else
      index=(ix-2)*4*nd+ndp1*ndp1
      if(iy.eq.1) then
      index=index+iz
      else if(iy.eq.ndp1) then
      index=index+nd+1+2*ndm1+iz
      else
      index=index+2*(iy-2)+nd+2+iz/nd
      endif
      endif
      ngr_out(j)=index
c****** END Searching by analitic expression ******************
      de=DSQRT(dr(ix)*dr(ix)+dr(iy)*dr(iy)+dr(iz)*dr(iz))
      grid(j,1)=dr(ix)/de
      grid(j,2)=dr(iy)/de
   50 grid(j,3)=dr(iz)/de
      return
      end
C******************************************************
      SUBROUTINE NORM(a,b)
      IMPLICIT NONE
      real*8    a(3),b(3),dnorm
      integer*4 i
      dnorm=DSQRT(a(1)*a(1)+a(2)*a(2)+a(3)*a(3))
      do 10 i=1,3
   10 b(i)=a(i)/dnorm
      return
      end
C******************************************************
      SUBROUTINE DNORM(a,b)
      IMPLICIT NONE
      real*8    a(3),b(3),norm
      integer*4 i
      norm=DSQRT(a(1)*a(1)+a(2)*a(2)+a(3)*a(3))
      do 10 i=1,3
   10 b(i)=a(i)/norm
      return
      end
c******************************************************
      SUBROUTINE RTURN(fi,a,x,x1)
      IMPLICIT NONE
      real*8    a(3),x(3),x1(3),r(3),r1(3),co,si,ax,fi
      integer*4 i
      co=DCOS(fi)
      si=DSIN(fi)
      ax=a(1)*x(1)+a(2)*x(2)+a(3)*x(3)
      do 10 i=1,3
   10 r(i)=x(i)-a(i)*ax
      CALL VECT(r,a,r1)
      do 20 i=1,3
   20 x1(i)=r(i)*co-r1(i)*si +a(i)*ax
      return
      end
C******************************************************
      SUBROUTINE TO_CART(t1,fi1,t2,fi2,x1,x2,xn,delta)
      IMPLICIT NONE
      include "line.h"
      real*8 x1(3),x2(3),xn(3),delta
      real*4 t1,fi1,t2,fi2
      x1(1)=DSIN((90.0d0-DBLE(t1))*drad)*DCOS(DBLE(fi1)*drad)
      x1(2)=DSIN((90.0d0-DBLE(t1))*drad)*DSIN(DBLE(fi1)*drad)
      x1(3)=DCOS((90.0d0-DBLE(t1))*drad)
      x2(1)=DSIN((90.0d0-DBLE(t2))*drad)*DCOS(DBLE(fi2)*drad)
      x2(2)=DSIN((90.0d0-DBLE(t2))*drad)*DSIN(DBLE(fi2)*drad)
      x2(3)=DCOS((90.0d0-DBLE(t2))*drad)
      CALL NORM(x1,x1)
      CALL NORM(x2,x2)
      CALL VECT(x1,x2,xn)
      CALL NORM(xn,xn)
      delta=DACOS(x1(1)*x2(1)+x1(2)*x2(2)+x1(3)*x2(3))
      return
      end
c******************************************************
      SUBROUTINE VECT(a,b,c)
      IMPLICIT NONE
      real*8 a(3),b(3),c(3)
      c(1)=a(2)*b(3)-a(3)*b(2)
      c(2)=a(3)*b(1)-a(1)*b(3)
      c(3)=a(1)*b(2)-a(2)*b(1)
      return
      end
c******************************************************
      SUBROUTINE AZI(j,grid,e,xn,az)
      IMPLICIT NONE
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    grid(4,3),e(3),xn(3),az,s
      real*8    d,a(3),b(3),c(3)
      integer*4 i,j
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      CALL NVECT(e,xn,b,a,d)
c* Construct mereian line from grid(1,i) to North pole
      do i=1,3
      c(i)=grid(j,i)
      enddo
      b(1)=-c(1)*c(3)
      b(2)=-c(2)*c(3)
      b(3)=c(1)*c(1)+c(2)*c(2)
      d=DSQRT(b(1)*b(1)+b(2)*b(2)+b(3)*b(3))
      s=0.0d0
      do i=1,3
      b(i)=b(i)/d
      s=s+a(i)*c(i)
      enddo
c* a and b are constructed, s=(a,c), move a to b
c*      s=0.0d0
      d=0.0d0
      do i=1,3
      a(i)=a(i)-s*c(i)
      d=d+a(i)*a(i)
      enddo
      d=DSQRT(d)
      do i=1,3
      a(i)=a(i)/d
      enddo
c*      d=-e(2)*a(1)+e(1)*a(2)
      d=-c(2)*a(1)+c(1)*a(2)
      if(d.lt.0.0d0) then
      do i=1,3
      a(i)=-a(i)
      enddo
      endif
      d=a(1)*b(1)+a(2)*b(2)+a(3)*b(3)
      if(d.ge.1.0d0) d=DSIGN(1.0d0,d)
      az=DACOS(d)
      return
      end
c******************************************************
      SUBROUTINE NVECT(a,b,c,cn,d)
      IMPLICIT NONE
      real*8    d,a(3),b(3),c(3),cn(3)
      integer*4 i
      c(1)=a(2)*b(3)-a(3)*b(2)
      c(2)=a(3)*b(1)-a(1)*b(3)
      c(3)=a(1)*b(2)-a(2)*b(1)
      d=DSQRT(c(1)*c(1)+c(2)*c(2)+c(3)*c(3))
      if(d.eq.0.0d0) then
      if(a(1).ne.0.0d0) then
      cn(1)=-a(2)
      cn(2)=a(1)
      cn(3)=a(3)
      else
      cn(1)=a(1)
      cn(2)=-a(3)
      cn(3)=a(2)
      endif
      d=DSQRT(cn(1)*cn(1)+cn(2)*cn(2)+cn(3)*cn(3))
      do 10 i=1,3
   10 cn(i)=cn(i)/d
      d=0.0d0
      return
      endif
      do 20 i=1,3
   20 cn(i)=c(i)/d
      return
      end
c*****************************************************************
c* Subroutine CELL4:   Creates COMMON block /mat/ containing
c* all necessary information for cellular grid map on the sphere.
c* Call from command line: cell n_pntt lat_normal lon_normal azimuth
c* n_pntt             - define the total number of cells by formula:
c*                     n=6*(2*n_pntt +1)*(2*n_pntt +1)
c*****************************************************************
      SUBROUTINE CELL4 (n_pntt)
      IMPLICIT NONE
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    def,r0
      real*8    dlincell,dsqcell,dd
      integer*4 i,k,n_pntt
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      dpi=4.0d0*DATAN(1.0d0)
      drad=dpi/180.d0
      r0=6371.d0
c************* Input parameters**********************************
      k=n_pntt
c************* END input parameters**********************************
c* Spere radius sqrt(3)
      nd=2*k+1
      nnd=nd*nd
      do 10 i=0,k
      def=dpi*DFLOAT((2*i+1)*(2*i+1))/DFLOAT((2*k+1)*(2*k+1))/6.0d0
      dt(k+2+i)=DACOS(DCOS(def)/(1.d0 + DSIN(def)))
      dr(k+2+i)=DTAN(dt(k+2+i))/DSQRT(2.0d0)
   10 dr(k+1-i)=-dr(k+2+i)
      dd=360.d0/2.d0/dpi
      dd=dd*dd
      dsqcell=4.0d0*dpi*dd/FLOAT(nd)/FLOAT(nd)/6.0d0
      dlincell=DSQRT(dsqcell)
      write(*,'(" Average square of cells is:" f17.10," (deg**2)")') dsqcell
      write(8,'(" Average square of cells is:" f17.10," (deg**2)")') dsqcell
      write(*,'(" Average length of cells is:" f17.10," (deg)")') dlincell
      write(8,'(" Average length of cells is:" f17.10," (deg)")') dlincell
      return
      end
c*************************************************************
c* SOME MISCALANNEOUS SUBS, DON'T USED IN THIS VERSION
c*************************************************************
C******************************************************
      SUBROUTINE ROTATE(norma,i,x1,y1,z1,out)
      IMPLICIT NONE
      real*8    x1,y1,z1,out(3)
      real*8    fi,ar(3),a(3),ar1(3)
      integer*4 i,j,ind1(6),ind2(6),norma
      data ind1/0,0,0,1,1,1/,ind2/0,-1,1,0,-1,1/
      data a/-.57735026918962576450,-.57735026918962576450,
     *.57735026918962576450/,fi/2.09439510239319549229/
      ar(1)=x1
      ar(2)=y1
      ar(3)=z1
      CALL TURN(ind2(i),a,ar,ar1)
      if(ind1(i).eq.1) then
      do 1 j=1,3
    1 ar1(j)=-ar1(j)
      endif
      do 2 j=1,3
    2 out(j)=ar1(j)
      if(norma.eq.1) CALL NORM(out,out)
      return
      end
C******************************************************
      SUBROUTINE TURN(ifi,a,x,x1)
      IMPLICIT NONE
      dimension a(3),x(3),x1(3),r(3),r1(3)
      real*8    sq3d2,co,si,a,x,x1,r,r1,ax
      integer*4 i,ifi
      data sq3d2/.86602540378443864676/
      co=1.d0
      si=0.d0
      if(ifi.ne.0) then
      co=-0.5d0
      si=DBLE(ifi)*sq3d2
      endif
      ax=a(1)*x(1)+a(2)*x(2)+a(3)*x(3)
      do 2 i=1,3
      r(i)=x(i)-a(i)*ax
    2 continue
      CALL VECT(r,a,r1)
      do 3 i=1,3
      x1(i)=r(i)*co-r1(i)*si +a(i)*ax
    3 continue
      return
      end
c*****************************************************************
c* Subroutine XTURN:   Rotate vector around axis a
c* x   - input vector
c* a   - axe of rotation,should be normalized to 1.
c* fi  - rotation angle (rad), pozitve direction clockwise
c*	 seeing from end of vector a side
c* x1  - normalized to 1 output vector
c*****************************************************************
      SUBROUTINE XTURN(fi,a,x,x1)
      IMPLICIT NONE
      dimension a(3),x(3),x1(3),r(3),r1(3)
      real*8    co,si,fi,a,x,x1,r,r1,ax
      integer*4 i
      co=DCOS(fi)
      si=DSIN(fi)
      ax=a(1)*x(1)+a(2)*x(2)+a(3)*x(3)
      do 2 i=1,3
      r(i)=x(i)-a(i)*ax
    2 continue
      CALL VECT(r,a,r1)
      do 3 i=1,3
      x1(i)=r(i)*co-r1(i)*si +a(i)*ax
    3 continue
      CALL NORM(x1,x1)
      return
      end
