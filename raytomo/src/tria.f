      SUBROUTINE PRPR(x)
      IMPLICIT NONE
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8 x(3)
      real*4 ff,fl
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ff=DACOS(x(3))/drad
      fl=DATAN2(x(2),x(1))/drad
      write(*,*) ff,fl
      return
      end
c***************************************************************
      SUBROUTINE TRIAW (ngr_out,grid,e,wei,ierr)
      IMPLICIT NONE
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    grid(4,3),e(3),wei(4)
      real*8    x1(3),x2(3),x3(3),u(3),w(3),uwn(3),x0(3),dnorm
      real*8    dlam(3),du,dw,duw,deu,dew
      integer*4    ngr_out(4),i,j,ierr,idec
      real*8    d1,d2
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
c Define triangle for point e
      do i=1,3
        u(i)=grid(2,i)
        w(i)=grid(3,i)
      enddo
      CALL VECT(u,w,uwn)
      CALL NORM(uwn,uwn)
      d1=e(1)*uwn(1)+e(2)*uwn(2)+e(3)*uwn(3)
      d2=grid(1,1)*uwn(1)+grid(1,2)*uwn(2)+grid(1,3)*uwn(3)
      idec=1
      if(d1*d2.ge.0.0d0) idec=0
c Perform interpolation
      do 9 j=idec,idec
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
      if(ilocr(ngr_out(i+j)).eq.0) ierr=1
    5 wei(i+j)=dlam(i)
    9 continue
      return
      end
c***************************************************************
      SUBROUTINE TRIAI (grid,e,fun,funr)
      IMPLICIT NONE
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    grid(4,3),e(3),fun(4),funr
      real*8    x1(3),x2(3),x3(3),u(3),w(3),uwn(3),x0(3),dnorm
      real*8    dlam(3),du,dw,duw,deu,dew,ngr_out(4)
      integer*4 i,j,ierr,isu,idec
      real*8    d1,d2
c END TYPE definitions ++++++++++++++++++++++++++++++++++
c Define triangle for point e
      do i=1,3
        u(i)=grid(2,i)
        w(i)=grid(3,i)
      enddo
      CALL VECT(u,w,uwn)
      CALL NORM(uwn,uwn)
      d1=e(1)*uwn(1)+e(2)*uwn(2)+e(3)*uwn(3)
      d2=grid(1,1)*uwn(1)+grid(1,2)*uwn(2)+grid(1,3)*uwn(3)
      idec=1
      if(d1*d2.ge.0.0d0) idec=0
c Perform interpolation
      do 9 j=idec,idec
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
      if(ilocr(isu).eq.0) ierr=1
    5 funr=funr+dlam(i)*fun(i+j)
    9 continue
      return
      end
