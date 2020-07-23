c******************************************************
c  Fitting output with cilinder. The cilinder radius
c  equals sig/2.
c******************************************************
      SUBROUTINE RES_FORM(lidd,iidd,nlidd,iin,nmm,ff,sig)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      integer*4 nmm
      real*4    ff(nmm),sig
      integer*4 lidd(nmm),iidd(nmm)
      integer*4 i,i1,i2,i3,iin,iou,nlidd,ilo,j
      real*8    s,s1,x1(3),del
c*
      nlidd=0
      do i=1,nmm
      ff(i)=0.0
      iidd(i)=0
      enddo
c* Get the central point coordinates, x1(i)
      iou=ioutr(iin)
      i1=icrpnt(iou)/1000000
      i3=icrpnt(iou)-i1*1000000
      i2=i3/1000
      i3=i3-i2*1000
      s1=dr(i1)*dr(i1)+dr(i2)*dr(i2)+dr(i3)*dr(i3)
      s=DSQRT(s1)
      x1(1)=dr(i1)/s
      x1(2)=dr(i2)/s
      x1(3)=dr(i3)/s
c more wide/close area
      do i=1,nall
        s1=0.0d0
        do j=1,3
          s1=s1+dpx(j,i)*dpx(j,iou)
        enddo
      if(DABS(s1).gt.1.0d0) s1=DSIGN(1.0d0,s1)
      del=DACOS(s1)*DBLE(R)
      if(del.le.sig) then
        ilo=ilocr(i)
        if(ilo.ne.0) then
              nlidd=nlidd+1
              lidd(nlidd)=ilo
              if(del.le.sig/2.0) iidd(nlidd)=1
        ff(ilo)=1.0d0
        endif
      endif
      enddo
      return
      end
