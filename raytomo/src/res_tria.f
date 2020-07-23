c*********************************************************
c This program generate piramidal shape around central
c point iin. After that program construct approcsimation
c cone with radius sig1 and amplitude amp1, only for 
c neigbouring points. The next step is constraction new
c more wider approximation cone for all points iside the
c sircle radiuus of sig1. The new cone radius is sig2 and
c n_f is the numder of points inside the new cone base
c*********************************************************
      SUBROUTINE RES_TRIA(lidd,nlidd,iin,nmm,ff,amp1,sig1,sig2,n_f,n2,
     *tresh,ierr)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      integer*4 nmm
      real*4    ff(nmm),amp1,sig1,sig2,tresh
      integer*4 lidd(nmm),nlidd,n_f,ierr
      real*8    s,s1,x1(3),x2(3),del,factor,s2,s3,atresh,wc,dc
      integer*4 i,i1,i2,i3,iin,i_glob,ilo_glob,ilo,iou,nd1,nve,
     *nsum_c,j,k,kk,mp,n2,nve2
      nlidd=0
      ierr=0
      factor=0.1d0
c
c  Get the central point coordinates, x1
c
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
c
c Define main plane
c
      mp=0
      if(i1.ge.2.and.i1.le.nd.and.i2.ge.2.and.i2.le.nd) mp=3
      if(i1.ge.2.and.i1.le.nd.and.i3.ge.2.and.i3.le.nd) mp=2
      if(i3.ge.2.and.i3.le.nd.and.i2.ge.2.and.i2.le.nd) mp=1
c
c  Get neighbour points coordinate, six points maximum
c
      nd1=nd+1
      s=0.0d0
      nve=0
      do i=i1-1,i1+1
      do j=i2-1,i2+1
      do k=i3-1,i3+1
        if(min0(i,j,k).eq.1.or.max0(i,j,k).eq.nd1) then
        if(min0(i,j,k).ge.1.and.max0(i,j,k).le.nd1) then
        if((mp.ne.1.or.IABS(i-i1).eq.0).and.
     +     (mp.ne.2.or.IABS(j-i2).eq.0).and.
     +     (mp.ne.3.or.IABS(k-i3).eq.0)) then
        if(max0(IABS(i-i1),IABS(j-i2),IABS(k-i3)).lt.2) then
          nsum_c=IABS(i-i1)+IABS(j-i2)+IABS(k-i3)
          if(nsum_c.eq.2.or.nsum_c.eq.1) then
            s1=dr(i)*dr(i)+dr(j)*dr(j)+dr(k)*dr(k)
            s1=DSQRT(s1)
            x2(1)=dr(i)/s1
            x2(2)=dr(j)/s1
            x2(3)=dr(k)/s1
            s1=x1(1)*x2(1)+x1(2)*x2(2)+x1(3)*x2(3)
            if(DABS(s1).gt.1.0d0) s1=DSIGN(1.0d0,s1)
            i_glob=i*1000000+j*1000+k
            ilo_glob=0
            do kk=1,nall
              if(icrpnt(kk).eq.i_glob) then
                 ilo_glob=kk
                 goto 10
              endif
            enddo
            STOP 'Wrong table icrpnt'
   10       ilo=ilocr(ilo_glob)
            if(ilo.ne.0) then
              nlidd=nlidd+1
              lidd(nlidd)=ilo
              del=DACOS(s1)/drad
              if(del.gt.0.1d0) then
                s=s+DBLE(ff(iin)-ff(ilo))/del
                nve=nve+1
              endif
            endif
          endif
        endif
        endif
        endif
        endif
      enddo
      enddo
      enddo
c
c  Evaluate sig1, (degree)
c
      s=s/nve 
      amp1=ff(iin)
      sig2=-1.0
      if(s.le.1.0d-10) then
        sig1=-1.0
      else
        sig1=ff(iin)/s
      endif
c
c more wider area. Incudes all surrounding points
c in the vicinity sig1. The new wider vicinity is
c sig2, (degree)
c
      s=0.0d0
      s2=0.0d0
      nve=0
      nve2=0
      wc=DABS(DBLE(ff(iin)))
      atresh=wc*factor
      wc=wc/2.0d0
      do i=1,nall
        if(i.ne.iou) then
          s1=0.d0
            do j=1,3
              s1=s1+dpx(j,i)*dpx(j,iou)
            enddo
          if(DABS(s1).gt.1.0d0) s1=DSIGN(1.0d0,s1)
          s1=DACOS(s1)*DBLE(R)
          del=s1/DBLE(R)/drad
c
c Cone approximation
c
          if(del.le.DBLE(sig1)) then
            ilo=ilocr(i)
            if(ilo.ne.0) then
              s=s+DBLE(ff(iin)-ff(ilo))/del
              nve=nve+1
            endif
          endif
c
c Gaussian approximation
c
c         if(del.le.DBLE(tresh)) then
          if(del.le.DBLE(sig1*1.2)) then
            ilo=ilocr(i)
            s3=DABS(DBLE(ff(ilo)))
c           if(ilo.ne.0.and.s3.gt.atresh) then
            if(ilo.ne.0) then
              dc=2.0d0*dpi*del/dlcell
              s2=s2+s3*del*del/dc
              wc=wc+ABS(ff(ilo))/dc
              nve2=nve2+1
            endif
          endif
c
        endif
      enddo
      n_f=nve+1
      n2=nve2+1
      if(s.le.1.0d-10) then
        sig1=-1.0
        ierr=1
      else
        s=s/nve
        sig1=ff(iin)/s
      endif
      if(s2.le.1.0d-10) then
        sig2=-1.0
        ierr=2
      else
        sig2=SNGL(DSQRT(s2/wc))
      endif
      return
      end
