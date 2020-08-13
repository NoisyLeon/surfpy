c ==========================================================
c Function filter4. Broadband filreting.
c ==========================================================
c Parameters for filter4 function:
c Input parameters:
c f1,f2   - low corner frequences, f2 > f1, Hz, (double)
c f3,f4   - high corner frequences, f4 > f3, Hz, (double)
c npow    - power of cosine tapering,  (int)
c dt      - sampling rate of seismogram in seconds, (double)
c n       - number of input samples, (int)
c seis_in - input array length of n, (float)
c Output parameters:
c seis_out - output array length of n, (float)
c ==========================================================

      subroutine gaufilt(alpha,c_per,dt,n,seis_in,seis_out)
      implicit none
      include 'myfftw3.h'
      integer*4 n
      real*8    alpha,dt,c_per
      real*4    seis_in(400000),seis_out(400000)
c ---
      integer*4 k,ns,nk
      real*8    plan1,plan2
      real*8    dom,pi,om0
      double complex czero,s(400000),sf(400000)
c ---
      czero = (0.0d0,0.0d0)

c determin the power of FFT
      ns = 2**max0(int(dlog(dble(n))/dlog(2.0d0))+1,13)
      pi = datan(1.0d0)*4.0d0
      om0 = 2.0d0*pi/c_per
      dom = 2*pi/dt/ns
      do k = 1,ns
        s(k) = czero
      enddo
      do k = 1,n
        s(k) = seis_in(k)
      enddo

c make backward FFT for seismogram: s ==> sf
      call dfftw_plan_dft_1d(plan1,ns,s,sf,
     *                         FFTW_BACKWARD, FFTW_ESTIMATE)
      call dfftw_execute(plan1)
      call dfftw_destroy_plan(plan1)
c kill half spectra and correct ends
      nk = ns/2+1
      do k = nk+1,ns
        sf(k) = czero
      enddo
      sf(1) = sf(1)/2.0d0
      sf(nk) = dcmplx(dreal(sf(nk)),0.0d0)
c===============================================================
c   make gaussian tapering
      call gautap(alpha,om0,dom,nk,sf)
c===============================================================
c make forward FFT for seismogram: sf ==> s
      call dfftw_plan_dft_1d(plan2,ns,sf,s,
     *                         FFTW_FORWARD, FFTW_ESTIMATE)
      call dfftw_execute(plan2)
      call dfftw_destroy_plan(plan2)
c forming final result
      do k = 1,n
        seis_out(k) = 2.0*real(dreal(s(k)))/ns
      enddo
      return
      end
c===============================================================
c Gaussian tapering subroutine itself
c===============================================================
      subroutine gautap(alpha,om0,dom,nk,sf)
      implicit none
      integer*4 k, nk
      real*8 alpha,om0,dom,ome,om2,b(32768)
      double complex sf(32768)
      do k=1,nk
         b(k)=0.0d0
         ome = (k-1)*dom
         om2 = -(ome-om0)*(ome-om0)*alpha/om0/om0
c         if( dabs(om2) .le. 40.0d0 ) then
             b(k) = dexp(om2)
c	    if( dabs(b(k)-0.5).le.0.01) write(*,*) ome/2./3.14159264, om0/2./3.14159265
             sf(k) = sf(k)*b(k)
c         endif
      enddo
      return
      end
