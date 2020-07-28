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
      if(dlam(1).lt.0.0d0) goto 9
      if(dlam(2).lt.0.0d0) goto 9
      if(dlam(3).lt.0.0d0) goto 9
      do 5 i=1,3
      if(ilocr(ngr_out(i+j)).eq.0) ierr=1
    5 wei(i+j)=dlam(i)
      return
    9 continue
      write(*,*)'ERROR LAMDA',dlam
      stop
      end
c***************************************************************
      SUBROUTINE TRIAI (grid,e,fun,funr)
      IMPLICIT NONE
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    grid(4,3),e(3),fun(4),funr
      real*8    x1(3),x2(3),x3(3),u(3),w(3),uwn(3),x0(3),dnorm
      real*8    dlam(3),du,dw,duw,deu,dew,ngr_out(4)
      integer*4 i,j,ierr
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
      if(dlam(1).lt.0.0d0) goto 9
      if(dlam(2).lt.0.0d0) goto 9
      if(dlam(3).lt.0.0d0) goto 9
      funr=0.0d0
      do 5 i=1,3
      if(ilocr(ngr_out(i+j)).eq.0) ierr=1
    5 funr=funr+dlam(i)*fun(i+j)
      return
    9 continue
      write(*,*)'ERROR LAMDA',dlam
      stop
      end
