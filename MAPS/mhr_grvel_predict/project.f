C#################################################################
      subroutine  project(npoints,xlat,xlon,vel)
           common/mod/f2(181,360)
C---------to project values from grid on the ray using bilinear interpolation
           parameter (nsize=70000)
	   real*8 xk,yk,drad
           real*8 f2
           real*4 vel(1),xlat(1),xlon(1)
C-----------interpolation starts------------------------S
C          PRint*,'NPOINTS=',npoints
      drad = datan(1.0d0)/45.0d0
 	   do i=1,npoints
           xk=(90.+xlat(i))*drad
           yk=xlon(i)*drad
	   vel(i)=RBIMOD1(xk,yk)
c          PRint*,xlat(i),xlon(i),vel(i)
 	   enddo
C-----------interpolation ends  ------------------------E
	   return
	   end
c------------------------------------------------------------
      REAL FUNCTION RBIMOD1(xk,yk)
      IMPLICIT NONE
        common/mod/f2(181,360)
      real*8      f2
      real*8 dff,dfl,r0,r1,r2,r3,di,dj,xk,yk,drad
      real*4 f11,f12,f21,f22
      integer*4 i,j
      drad = datan(1.0d0)/45.0d0
      dfl=yk/drad
      if(dfl.lt.0.d0)dfl=dfl+360.d0
      dff=xk/drad
      i=dfl+1.d0
      j=dff+1.d0
      if(i.lt.1)i=1
      if(i.gt.359)i=359
      if(j.lt.1)j=1
      if(j.gt.180)j=180
      di=dfl-i+1
      dj=dff-j+1
      r0=f2(j,i)
      r1=f2(j+1,i)-f2(j,i)
      r2=f2(j,i+1)-f2(j,i)
      r3=f2(j,i)+f2(j+1,i+1)-f2(j,i+1)-f2(j+1,i)
      RBIMOD1=r0+r1*dj+r2*di+r3*di*dj
      f11=f2(j,i)
      f12=f2(j+1,i)
      f21=f2(j,i+1)
      f22=f2(j+1,i+1)
C     if(k.eq.1.or.k.eq.52)PRint*,i,j,xk,yk,RBIMOD1,f11,f12,f21,f22
      return
      end
