c
c convert phase to phase velocity
c
      subroutine phtovel(delta,ip,n,per,U,pha,npr,prper,prvel,V)
c
c delta  - distance, km
c n      - # of points in per
c per    - apparent periods
c U      - group velocity
c pha    - phase
c npr    - # of points in prper
c prper  - pedicted periods
c prvel  - pedicted phase velocity
c V      - observed phase velocity

      implicit none
      integer*4 ip,n,npr
      real*8    delta,per(100),U(100),pha(100),prper(npr),prvel(npr),V(100)
      real*8    pi,om(100),sU(100),t(100)
      integer*4 i,ierr,k,m
      real*8    Vpred,phpred,s,ss

      pi = datan(1.0d0)*4.0d0
      Vpred = 0.0d0
      do i =1,n
         om(i) = 2.0d0*pi/per(i)
         sU(i) = 1./U(i)
         t(i)  = delta*sU(i)
         V(i)  = 00d0
      enddo
c find velocity for the largest period by spline interpolation
c with not a knot boundary conditions
      call mspline(ip+2,npr,prper,prvel,0,0.0d0,0,0.0d0)
      call msplder(ip+2,per(n),Vpred,s,ss,ierr)
      write(*,*) 'Starting period= ',per(n),', Phase velocity= ',Vpred
      phpred = om(n)*(t(n)-delta/Vpred)
      k = nint((phpred -pha(n))/2.0d0/pi)
      V(n) = delta/(t(n)-(pha(n)+2.0d0*k*pi)/om(n))
      do m = n-1,1,-1
        Vpred =1/(((sU(m)+sU(m+1))*(om(m)-om(m+1))/2.0d0+om(m+1)/V(m+1))/om(m))
        phpred = om(m)*(t(m)-delta/Vpred)
        k = nint((phpred -pha(m))/2.0d0/pi)
        V(m) = delta/(t(m)-(pha(m)+2.0d0*k*pi)/om(m))
      enddo
      return
      end
