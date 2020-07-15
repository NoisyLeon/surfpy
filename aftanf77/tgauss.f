c-------------------------------------------------------------
c taper phase matched signal
c-------------------------------------------------------------
      subroutine tgauss(fsnr,gt0,t0,dw,dt,n,fmatch,seis,
     *                  ss)
      implicit none
      integer*4 n, i,ism,nn,nnn
      double complex czero,seis(n),ss(n)
      real*8    smax(32768)
      real*8    pi, dt, gt0, t0, dw, fsnr,sm,smw,fmatch
      integer*4 nc,ii,nnl,nnnl,nnr,nnnr,nleft,nright
      integer*4 left(100),right(100)
      real*8    freq,dzer,dl,dr,vleft(100),vright(100),tre,tresh

      dw = dw
      ism = 1
      dzer = 0.0d0
      czero = (0.0d0,0.0d0)
      pi = datan(1.0d0)*4.0d0
      nc = nint(gt0/dt)+1
c find global max, sm, and index ism
      sm = 0.0d0
      do i = 1,n
         smw = cdabs(seis(i))
         if(smw.ge.sm) then
             sm = smw
             ism = i
         endif
         smax(i) = smw
         ss(i) = seis(i)
      enddo
cc Commented for pyaftan
c      write(*,*) 'Distance between maximas=',gt0-(ism-1)*dt-t0,
c     *           ' in sec,',
c     * ' Spectra point= ',ism
cc End commented for pyaftan
c find some local minima,# < 100 from left and right side of central max ism
c left side 
      nleft = 0
      do i = ism-1,2,-1     
          dl = smax(i)-smax(i-1)
          dr = smax(i+1)-smax(i)
          if((dl.lt.dzer.and.dr.ge.dzer).or.(dl.le.dzer.and.dr.gt.dzer)) then
              nleft = nleft+1
              left(nleft) = i
              vleft(nleft) = smax(i)
          endif
          if(nleft.ge.100) goto 10
      enddo
   10 continue
c right side
      nright = 0
      do i = ism+1,n-1      
          dl = smax(i)-smax(i-1)
          dr = smax(i+1)-smax(i)
          if((dl.lt.dzer.and.dr.ge.dzer).or.(dl.le.dzer.and.dr.gt.dzer)) then
              nright = nright+1
              right(nright) = i
              vright(nright) = smax(i)
          endif
          if(nright.ge.100) goto 20
      enddo
   20 continue
c left side, apply cutting
      ii = 0
      nnl = 0 
      nnnl = 0
      if(nleft.eq.0) goto 21
       do i = 1,nleft
           if(abs(ism-left(i))*dt.gt.5.0d0) then
                if(vleft(i) .lt. fsnr*sm) then
                    nnnl = left(i)
                    ii = i
                    goto 21
                endif
           endif
       enddo
   21 continue
       if(nnnl.ne.0) then
           if(ii.ne.nleft) then
               nnl = left(ii+1)
           else
               nnl = 1
           endif
       endif
c right side, apply cutting
      ii = 0
      nnr = 0 
       nnnr = 0
      if(nright.eq.0) goto 31
       do i = 1,nright
           if(abs(ism-right(i))*dt.gt.5.0d0) then
                if(vright(i) .lt. fsnr*sm) then
                    nnr = right(i)
                    ii = i
                    goto 31
                endif
           endif
       enddo
   31  continue
       if(nnr.ne.0) then
           if(ii.ne.nright) then
               nnnr = right(ii+1)
           else
               nnnr = n
           endif
       endif
c ---
       if(nnnr.ne.0.and.nnnl.ne.0) then
       nn = max0(iabs(ism-nnnl),iabs(ism-nnr))
       nnn = max0(iabs(nnnl-nnl),iabs(nnnr-nnr))
       nnnl = ism -nn
       nnl = nnnl-nnn
       nnr = ism +nn
       nnnr = nnr+nnn
       endif
c setup cutting point for gaussian
       tresh = dlog(sm)-24.0d0
       if(nnl.eq.0.or.nnnl.eq.0) goto 30
c expand left end by factor fmatch
         nnl = nint((nnl-ism)*fmatch)+ism
         nnnl = nint((nnnl-ism)*fmatch)+ism
         nnl = max0(1,nnl)
         nnnl = max0(1,nnnl)
           freq =(nnnl-nnl)+1
       do i = 1,nnnl
           tre = -(i-nnnl)/freq*(i-nnnl)/freq/2.0d0
           if(tre .gt. tresh) then
               ss(i) = ss(i)*dexp(tre)
           else
               ss(i) = czero
           endif
       enddo
   30  continue
       if(nnr.eq.0.or.nnnr.eq.0) goto 40
c expand right end by factor fmatch
         nnr = nint((nnr-ism)*fmatch)+ism
         nnnr = nint((nnnr-ism)*fmatch+ism)
         nnr = min0(n,nnr)
         nnnr = min0(n,nnnr)
           freq =(nnnr-nnr)+1
       do i = nnr,n
           tre = -(i-nnr)/freq*(i-nnr)/freq/2.0d0
           if(tre .gt. tresh) then
               ss(i) = ss(i)*dexp(tre)
           else
               ss(i) = czero
           endif
       enddo
   40 continue
      !do i = 1, n
      !  write(*,*) ss(i)
      !enddo
      !write(*,*) nnl, nnnl, nnr, nnnr
cc Commented for pyaftan       
c     write(*,*) nnl,nnnl,ism,nnr,nnnr
      return
      end
