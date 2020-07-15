c-------------------------------------------------
c FTAN filter: y =x*exp(-alpha*((om-om0)/om0)**2
c-------------------------------------------------
      subroutine ftfilt(alpha,om0,dom,n,a,fs, b)
      implicit none
      integer*4 k, n
      real*8 alpha,om0,dom,ome,om2,b(32768)
      double complex a(32768), fs(32768)
      do k=1,n
         fs(k) = dcmplx(0.0d0, 0.0d0)
         b(k)=0.0d0
         ome = (k-1)*dom
         om2 = -(ome-om0)*(ome-om0)*alpha/om0/om0
        if( dabs(om2) .le. 40.0d0 ) then
            b(k) = dexp(om2)
            fs(k) = a(k)*b(k)
        endif
      enddo
      return
      end
