c test function for res_outp
c     include "tomo.h"
c     real*4 f(5,5)
c     real*8 xk(5),yk(5)
c     nmm=5
c     fname7='aaa'
c     root='bbb'
c     call RES_OUTP(nmm,xk,yk,fff)
c     end

c*****************************************************
c Output resolution responses for the set of fixed
c points, (rf(i),rl(i), i=1,...nr), where rf(i),rl(i)
c are geographical latitude and longitude, (deg).
c*****************************************************
      SUBROUTINE RES_OUTP(nmm,xk,yk,fff)
      implicit none
      include "tomo.h"
      include "line.h"
      integer*4 nmm,i,j,k,n_p,lr,lnblnk,icrdt,l
      real*4    fff(nmm,nmm)
      real*8    xk(nmm),yk(nmm)
      real*8    rf(1000),rl(1000),nr(1000),rr(1000)
      real*8    rrr,dd,ff,fl
      character*160 fname7
      character*10 scrdt
      logical   tf
c
c Check that file SURF_POINTS exists
c
      icrdt=0
      fname7='SURF_POINTS'
      inquire(file=fname7,exist=tf)
      if(.not.tf) return
c
c Read input data
c
      open(60,file=fname7,status='old')
      read(60,'(a4)') scrdt
      if(scrdt(1:1).eq."c") icrdt=1
      n_p=0
    1 read(60,*,end=99) ff,fl
      n_p=n_p +1
      rf(n_p)=DATAN(0.993277d0*DTAN(drad*ff))/drad
      rl(n_p)=fl
      goto 1
   99 close(60)
c
c  output coordinates of resolution matrix
c
      if(icrdt.eq.1) then
        lr=lnblnk(root)
        open(59,file=root(1:lr)//'.crdt',form='unformatted',status='unknown')
          write(59) nmm
          write(59) (xk(l),yk(l),l=1,nmm)
        close(59)
      endif
      if(n_p.eq.0) return
c
c Find the closest grid points
c
      do i=1,n_p
         rr(i)=1.0d7
      enddo
      do i=1,nmm
        do j=1,n_p
         rrr=yk(i)-rl(j)
         if(rrr.lt.-180.0d0) rrr=rrr+360.0d0
         if(rrr.gt.180.0d0) rrr=rrr-360.0d0
         dd=(xk(i)-rf(j))*(xk(i)-rf(j))+rrr*rrr
         if(dd.lt.rr(j)) then
           rr(j)=dd
           nr(j)=i
         endif
        enddo
      enddo
c
c Output response maps
c
      lr=lnblnk(root)
      open(59,file=root(1:lr)//'.pnt',status='unknown')
      do j=1,n_p
        k=nr(j)
        ff=DATAN(DTAN(drad*rf(j))/0.993277d0)/drad
        write(59,'(i5,i7,4f10.3)') j,nmm,ff,rl(j),xk(k),yk(k)
        do i=1,nmm
          write(59,'(2f10.4,f10.6)') xk(i),yk(i),fff(i,k)
        enddo
      enddo
      close(59)
      return
      end
