c Group velocity prediction curve
      subroutine pred_cur(ip,delta,om0,npred,pred,   om1,Gt0)
      implicit none
      integer*4 i,ip,ierr,npred
      real*8  delta,om0,om1,Gt0,pi,s,ss,pred(500,2),x(npred),y(npred)

c transfer prediction curve (T,vel) ==> (omrga,t)
      pi = atan(1.0d0)*4.0d0
      do i =1,npred
          x(i) = 2*pi/pred(npred+1-i,1)
          y(i) = delta/pred(npred+1-i,2)
      enddo
c get velocity for low integral boundary
      call mspline(ip,npred,x,y,0,0.0d0,0,0.0d0)
      call msplder(ip,om0,Gt0,s,ss,ierr)
c construct spline for ph filter
      do i = 1,npred
          y(i) = y(i) - Gt0
      enddo
      call mspline(ip+1,npred,x,y,0,0.0d0,0,0.0d0)
      om1 = om0
      return
      end
