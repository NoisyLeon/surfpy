c test dispersion curve for jumps
      subroutine trigger(grvel,per,nf,tresh,trig, ftrig, ierr)
      implicit none
      integer*4 nf,ierr,i
      real*8    tresh,grvel(100),per(100),trig(100),ftrig(100)
      real*8    hh1,hh2,hh3,r

      ierr = 0
      ftrig(1) = 0.0d0
      ftrig(nf) = 0.0d0
      trig(1) = 0.0d0
      trig(nf) = 0.0d0

      do i =1,nf-2
        trig(i+1) = 0.0d0
c        hh1 = 1.0d0/per(i)-1.0d0/per(i+1)
c        hh2 = 1.0d0/per(i+1)-1.0d0/per(i+2)
        hh1 = dlog(per(i+1)) - dlog(per(i))
        hh2 = dlog(per(i+2)) - dlog(per(i+1))
        hh3 = hh1+hh2
c        r = (grvel(i)/hh1-(1.0d0/hh1+1.0d0/hh2)*grvel(i+1)+grvel(i+2)/hh2)*hh3/4.0d0*100.0d0
        r = 2.0d0*(grvel(i)*hh2+grvel(i+2)*hh1-grvel(i+1)*hh3)/(hh1*hh2*hh3*3.0d0)
c        write (*,*) "Curvature at ",per(i+1),"sec: ",r," ( ",per(i),per(i+1),per(i+2)," ) "
        if( per(i+1).ge.15.0d0 ) then
           r = r*2.0d0
        endif
        ftrig(i+1) = r
        if(dabs(r).gt.tresh) then
           trig(i+1) = dsign(1.0d0,r)
           ierr = 1
        endif
      enddo
      return
      end
