c******************************************************
      SUBROUTINE EGAUSS_2(lidd,nlidd,iin,nmm,nmma,ff,dens,wden)
      IMPLICIT NONE
      include "tomo.h"
      include "linea.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8 swgt(3),wgt,tresh(3),s,s1,sigmw(3),x1(3),wgta
      real*4 wden,ff(NRAZ+NRAZA*(NAZIPL-1)),dens(NRAZA)
      integer*4 i,i1,i2,i3,iin,ilo,iou,ish,ish1,j,k,nmm,nmma
      integer*4 ip,nlidd,lidd(NRAZ+NRAZA*(NAZIPL-1))
c END TYPE definitions ++++++++++++++++++++++++++++++++++
C******************************************************
c*** exp(-s**2/2/sigma**2) < 10**(-pow), tresh=COS(s)
C******************************************************
      wden=0.0
      do i=2,ianiz+1
      swgt(i)=0.0d0
cxx   sigmw(i)=((alsi(i,4)-alsi(i,3))*(azmax-dazi(iin))/azmax+alsi(i,3))/R
      sigmw(i)=alsi(i,3)/R
      tresh(i)=DCOS(sigmw(i)*DSQRT(2.d0*DBLE(alsi(i,5))*DLOG(10.0d0)))
      enddo
      iou=ioutra(iin)
      i1=icrpnta(iou)/1000000
      i3=icrpnta(iou)-i1*1000000
      i2=i3/1000
      i3=i3-i2*1000
      s1=dra(i1)*dra(i1)+dra(i2)*dra(i2)+dra(i3)*dra(i3)
      s=DSQRT(s1)
      x1(1)=dra(i1)/s
      x1(2)=dra(i2)/s
      x1(3)=dra(i3)/s
      do 20 i=1,nalla
      s1=0.0d0
      ilo=ilocra(i)
      do 10 j=1,3
   10 s1=s1+x1(j)*dpxa(j,i)
      if(s1.gt.1.d0)s1=1.d0
      do 40 k=1,ianiz
      if(s1.lt.tresh(k+1)) goto 40
      s=DACOS(s1)
      wgta=s*s/sigmw(k+1)/sigmw(k+1)/2.0d0
      wgt=0.0d0
      if(wgta.lt.30.0d0) then
      wgt=DEXP(wgta)
      wgt=dpxa(4,i)/wgt
      endif
      swgt(k+1)=swgt(k+1)+wgt
      if(ilo.ne.0) then
      ff(ilo+(2*k-1)*nmma+nmm)=wgt
      if(k.eq.1) wden=wden+dens(ilo)*wgt
      endif
   40 continue
   20 continue
      do 50 k=1,ianiz
      ish=(2*k-1)*nmma+nmm
      do 30 j=1,nmma
   30 ff(j+ish)=ff(j+ish)*alsi(k+1,1)/swgt(k+1)
   50 ff(iin+ish)=(ff(iin+ish)/alsi(k+1,1)-1.0)*alsi(k+1,1)
      wden=wden/SNGL(swgt(2))
      do 70 k=1,ianiz
      ish=(2*k-1)*nmma+nmm
      ish1=ish-nmma
      do 60 i=1,nmma
   60 ff(i+ish1)=ff(i+ish)
   70 continue
      do j=nmm+1,nmm+2*ianiz*nmma
        if(ff(j).ne.0.0) then
          nlidd=nlidd+1
          ip=(j-nmm-1)/nmma
          lidd(nlidd)=j*10+ip
        endif
      enddo
      return
      end
c******************************************************
      SUBROUTINE E2PNTS_2(e,ngr_out,grid)
      IMPLICIT NONE
      include "linea.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    grid(4,3),e(3)
      real*8    dmax,de
      integer*4 ic,icol,index,ishift,ival,ndp1,ndm1
      integer*4 ix,iy,iz,j,k,nplane,ngr_out(4),ne(3)
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      equivalence (ne(1),ix),(ne(2),iy),(ne(3),iz)
      dmax=DMAX1(DABS(e(1)),DABS(e(2)),DABS(e(3)))
      nplane=0
      do 30 j=1,3
      if(nplane.eq.0.and.DABS(e(j)).eq.dmax) then
      nplane=j
      de=e(j)/dmax
      if(de.lt.-.5) ne(j)=1
      if(de.gt.0.5)ne(j)=nda+1
      goto 30
      endif
      de=e(j)/dmax
      icol=nda
      do 10 k=2,nda
      if(de.lt.dra(k)) then
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
      ndp1=nda+1
      ndm1=nda-1
      if(ix.eq.1) then
      index=(iy-1)*ndp1+iz
      else if(ix.eq.ndp1) then
      index=ndp1*ndp1+4*ndm1*nda+(iy-1)*ndp1+iz
      else
      index=(ix-2)*4*nda+ndp1*ndp1
      if(iy.eq.1) then
      index=index+iz
      else if(iy.eq.ndp1) then
      index=index+nda+1+2*ndm1+iz
      else
      index=index+2*(iy-2)+nda+2+iz/nda
      endif
      endif
      ngr_out(j)=index
c****** END Searching by analitic expression ******************
      de=DSQRT(dra(ix)*dra(ix)+dra(iy)*dra(iy)+dra(iz)*dra(iz))
      grid(j,1)=dra(ix)/de
      grid(j,2)=dra(iy)/de
   50 grid(j,3)=dra(iz)/de
      return
      end
c***************************************************************
      SUBROUTINE TRIAW_2 (ngr_out,grid,e,wei,ierr)
      IMPLICIT NONE
      include "linea.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    grid(4,3),e(3),wei(4)
      real*8    x1(3),x2(3),x3(3),u(3),w(3),uwn(3),x0(3),dnorm
      real*8    dlam(3),du,dw,duw,deu,dew
      integer*4    ngr_out(4),i,j,ierr
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
      do 9 j=0,1
      do 1 i=1,4
    1 wei(i)=0.0d0
      do 2 i=1,3
      x1(i)=grid(1+j,i)
      x2(i)=grid(2+j,i)
      x3(i)=grid(3+j,i)
      u(i)=x2(i)-x1(i)
    2 w(i)=x3(i)-x1(i)
c***** Intersection vector e with plane within x1, x2 and x3 ****
      CALL VECT(u,w,uwn)
      CALL NORM(uwn,uwn)
      dnorm=(uwn(1)*x1(1)+uwn(2)*x1(2)+uwn(3)*x1(3))/
     *(uwn(1)*e(1)+uwn(2)*e(2)+uwn(3)*e(3))
      do 3 i=1,3
    3 x0(i)=e(i)*dnorm-x1(i)
      du=u(1)*u(1)+u(2)*u(2)+u(3)*u(3)
      dw=w(1)*w(1)+w(2)*w(2)+w(3)*w(3)
      duw=u(1)*w(1)+u(2)*w(2)+u(3)*w(3)
      dnorm=du*dw-duw*duw
      deu=x0(1)*u(1)+x0(2)*u(2)+x0(3)*u(3)
      dew=x0(1)*w(1)+x0(2)*w(2)+x0(3)*w(3)
      dlam(2)=(deu*dw-dew*duw)/dnorm
      dlam(3)=(dew*du-deu*duw)/dnorm
      dlam(1)=1.d0-dlam(2)-dlam(3)
      do 5 i=1,3
      if(ilocra(ngr_out(i+j)).eq.0) ierr=1
    5 wei(i+j)=dlam(i)
      return
    9 continue
      end
c***************************************************************
      SUBROUTINE TRIAI_2 (grid,e,fun,funr)
      IMPLICIT NONE
      include "linea.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    grid(4,3),e(3),fun(4),funr
      real*8    x1(3),x2(3),x3(3),u(3),w(3),uwn(3),x0(3),dnorm
      real*8    dlam(3),du,dw,duw,deu,dew,ngr_out(4)
      integer*4 i,j,ierr,isu
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      do 9 j=0,1
      do 2 i=1,3
      x1(i)=grid(1+j,i)
      x2(i)=grid(2+j,i)
      x3(i)=grid(3+j,i)
      u(i)=x2(i)-x1(i)
    2 w(i)=x3(i)-x1(i)
c***** Intersection vector e with plane within x1, x2 and x3 ****
      CALL VECT(u,w,uwn)
      CALL NORM(uwn,uwn)
      dnorm=(uwn(1)*x1(1)+uwn(2)*x1(2)+uwn(3)*x1(3))/
     *(uwn(1)*e(1)+uwn(2)*e(2)+uwn(3)*e(3))
      do 3 i=1,3
    3 x0(i)=e(i)*dnorm-x1(i)
      du=u(1)*u(1)+u(2)*u(2)+u(3)*u(3)
      dw=w(1)*w(1)+w(2)*w(2)+w(3)*w(3)
      duw=u(1)*w(1)+u(2)*w(2)+u(3)*w(3)
      dnorm=du*dw-duw*duw
      deu=x0(1)*u(1)+x0(2)*u(2)+x0(3)*u(3)
      dew=x0(1)*w(1)+x0(2)*w(2)+x0(3)*w(3)
      dlam(2)=(deu*dw-dew*duw)/dnorm
      dlam(3)=(dew*du-deu*duw)/dnorm
      dlam(1)=1.d0-dlam(2)-dlam(3)
      funr=0.0d0
      do 5 i=1,3
      isu=ngr_out(i+j)
      if(ilocra(isu).eq.0) ierr=1
    5 funr=funr+dlam(i)*fun(i+j)
      return
    9 continue
      end
c***************************************************************
      SUBROUTINE EINTPOL_2 (xk,yk,nmm,nmma,fsl,value,nval,ierr)
      IMPLICIT NONE
      include "linea.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      integer*4 nmm,nmma,ierr,j,mo,nclo,nval,ngr_out(4)
      real*4    fsl(nmm),value(5)
      real*8    grid(4,3),w(4),e(3),scell,xk,yk
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
      e(1)=DSIN(xk)*DCOS(yk)
      e(2)=DSIN(xk)*DSIN(yk)
      e(3)=DCOS(xk)
      CALL E2PNTS_2(e,ngr_out,grid)
      do 20 mo=1,nval
      if(nval.gt.1.and.mo.eq.1) goto 20
      do 10 j=1,4
      nclo=ilocra(ngr_out(j))
      if(nclo.lt.1) then
      ierr=1
      return
      endif
      if(nval.eq.1) then
          w(j)=fsl(nclo)
      else
          w(j)=fsl(nmm+nmma*(mo-2)+nclo)
      endif
   10 continue
      CALL TRIAI_2(grid,e,w,scell)
      value(mo)=scell
   20 continue
      return
      end
c******************************************************
