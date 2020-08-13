c
c tapering both end of input seismogram
c
      subroutine taper(nb,ne,n,seis,ntapb,ntape,ss,ncorr)
      implicit none
      integer*4 nb,ne,n,ntapb,ntape,ncorr
      real*4    seis(32768)
      real*8    s(32768)
      double complex ss(32768)
      real*8    omb,ome,sums,c,r,pi
      integer*4 k,ns

      pi = datan(1.0d0)*4.0d0
      omb = pi/ntapb
      ome = pi/ntape
      ncorr = ne+ntape
cxx   s = seis(1:ncorr)
c make copy seis to s
      do k = 1,ncorr
        s(k) = seis(k)
      enddo
      print *, "nb: ",nb,"  ntapb: ",ntapb
      if(nb-ntapb-1 .gt. 0) then
          do k=1,nb-ntapb-1
              s(k) = 0.0d0
          enddo
      endif
      sums = 0.0d0
c left end of the signal
      do k = nb,nb-ntapb,-1
          r = (dcos(omb*(nb-k))+1.0d0)/2.0d0
          sums = sums + 2.0d0*r
          s(k) = s(k)*r
      enddo

c set left to zero
c      do k = nb,nb-ntapb,-1
c          r = (dcos(omb*(nb-k))+1.0d0)/2.0d0
c          sums = sums + 2.0d0*r
c          s(k) = s(k)*r
c      enddo
c --
c right end of the signal
      do k = ne,ne+ntape
          s(k) = s(k)*(dcos(ome*(ne-k))+1.0d0)/2.0d0
      enddo
      sums = sums+ne-nb-1
      c = 0.0d0
      do k = 1,ncorr
        c = c + s(k)
      enddo
      c = -c/sums
c left end of the signal
      do k = nb,nb-ntapb,-1
          r = (dcos(omb*(nb-k))+1.0d0)/2.0d0
          s(k) = s(k)+r*c
      enddo
c right end of the signal
      do k = ne,ne+ntape
          r = (dcos(ome*(ne-k))+1.0d0)/2.0d0
          s(k) = s(k)+r*c
      enddo
c middle of the signal
      do k = nb+1,ne-1
          s(k) = s(k)+c
      enddo
c determin the power of FFT
      ns = 2**(min0(max0(int(dlog(dble(ncorr))/dlog(2.0d0))+1,12),15))
      if(ns .gt. ncorr) then
        do k = ncorr+1,ns
          s(k) = 0.0d0
        enddo
      endif
      ncorr = ns
c convert to complex
      do k =1,ns
        ss(k) = complex(s(k),0.0d0)
      enddo
      return
      end
