c------------------------------------------------------------
c test dispersion curve for jumps
c------------------------------------------------------------
      subroutine trigger(grvel,om,nf,tresh,trig, ftrig, ierr)
      implicit none
      integer*4 nf,ierr,i
      real*8    tresh,grvel(100),om(100),trig(100),ftrig(100)
      real*8    hh1,hh2,hh3,r

      ierr = 0
      ftrig(1) = 0.0d0
      ftrig(nf) = 0.0d0
      trig(1) = 0.0d0
      trig(nf) = 0.0d0

      do i =1,nf-2
        trig(i+1) = 0.0d0
        hh1 = om(i+1)-om(i)
        hh2 = om(i+2)-om(i+1)
        hh3 = hh1+hh2
c NEED modifications!!!
        r = (grvel(i)/hh1-(1.0d0/hh1+1.0d0/hh2)*grvel(i+1)+
     *          grvel(i+2)/hh2)*hh3/4.0d0*100.0d0
        ftrig(i+1) = r
        if(dabs(r).gt.tresh) then
           trig(i+1) = dsign(1.0d0,r)
           ierr = 1
        endif
      enddo
      return
      end
