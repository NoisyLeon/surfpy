c====================================================================
c  Generate conntrol intervals for Fresnel zone
c====================================================================
      SUBROUTINE GEN_INT(vic,Delta,Left,Right,Num,Ntab,ierr)
      implicit none
      include "line.h"
      real*8    Left(7),Right(7),Point(4),tmp(2)
      real*8    vic,Delta,wor
      integer*4 i,j,Np,ierr,Ntab,Num(7)
c create focus points ---
      ierr=0
      if(Delta.gt.dpi) then
        Np=4
        Point(1)=0.0d0
        Point(2)=Delta-dpi
        Point(3)=dpi
        Point(4)=Delta
      else
        Np=2
        Point(1)=0.0d0
        Point(2)=Delta
      endif
c Create interval table ---
      tmp(1)=0.0d0
      tmp(2)=0.0d0
      j=1
      do i=1,Np
        tmp(1)=tmp(2)
        tmp(2)=Point(i)-vic
        Left(j)=tmp(1)
        Right(j)=tmp(2)
        Num(j)=4
        if(i.gt.1) j=j+1
        tmp(1)=tmp(2)
        tmp(2)=Point(i)+vic
        Left(j)=tmp(1)
        Right(j)=tmp(2)
        Num(j)=4
        j=j+1
      enddo
      Ntab=j-1
c Check sequence or table ---
      wor=Left(1)
      do i=2,Ntab
        if(Left(i).lt.Left(i-1)) then
          ierr=1
          Ntab=0
          return
        endif
      enddo
c Add step for each interval ---
      do i=1,Ntab
        if(MOD(i,2).eq.0) then
          j=(Right(i)-Left(i))/drad+0.5d0
          if(j.gt.Num(i)) Num(i)=j
        else
          j=5.0d0*(Right(i)-Left(i))/drad+0.5d0
          j=j/2
          j=2*j+1
          if(j.gt.Num(i)) Num(i)=j
        endif
      enddo
      return
      end
c====================================================================
c This program compute model parameters, denf, for single fresnel
c zone and travel time, t00, across starting model
c====================================================================
      SUBROUTINE FRESNEL(type,type1,delta,x1,x2,xn,i_ray,nmm,lidd,
     *nlidd,denf,den1f,den2f,ff,tc00)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      character*4 type,type1
      integer*4 nmm
      real*4    denf(nmm),den1f(nmm),den2f(nmm),ff(nmm)
      real*4    dwth,wnum(NRAZ)
      real*8    x1(3),x2(3),xn(3),w(3),ww(3),e(3),period
      real*8    d,dw,slo,delta,wth,wwth,wths,wths1,tc00
      real*8    a(3,3),step_d,sq,RBIMOD_8,rbi_mod,Lambda
      real*8    Left(7),Right(7),vic
      integer*4 n_d,n_w,ke,k,k1,k2,kk,Ntab
      integer*4 Num(7),lidd(NRAZ)
      integer*4 nlidd,i,j,ncl,ierr,ierr1,iyes,i_ray
c     real*8 xx,yy
      integer*4 count
      data count/0/
c get slowness from PREM model ==================
      period=tper
      CALL GET_SLOW(type,period,slo,ierr1)
      if(ierr1.ne.0) stop '(R0015) 1-D model period is out of range'
      Lambda=period/(slo*R)
c ---
      do i=1,NRAZ
        wnum(i)=0.0
      enddo
      wths=0.0d0
      wths1=0.0d0
      tc00=0.0d0
      CALL VECT(xn,x1,w)
      CALL NORM(w,w)
      do i=1,3
      a(i,1)=x1(i)
      a(i,2)=w(i)
      a(i,3)=xn(i)
      enddo
c
c    get intervals
c
      vic=Lambda/x_zone/2.0d0
      CALL GEN_INT(vic,delta,Left,Right,Num,Ntab,ierr)
      if(ierr.eq.1) then
        nlidd=0
        return
      endif
c
c Main loop
c
      do kk=1,Ntab
      n_d=Num(kk)
      step_d=(Right(kk)-Left(kk))/n_d
      do i=1,n_d
        d=(i-1)*step_d+step_d/2.0d0+Left(kk)
        n_w=80.0d0*drad/step_d
        do j=1,n_w
          dw=(j-1)*step_d
          ke=1
          if(j.gt.1) ke=2
          do k=1,ke
          if(j.eq.1) then
c
c Where we are, inside or outside Fresnel zone ?
c
              CALL FUNCT(Lambda,delta,d,step_d,wth,iyes)
              if(iyes.eq.0) goto 2
              wwth=1.0d0/(2.0d0*wth)
              dwth=step_d/wwth
          else
              iyes=1
              if(dw.gt.wwth) iyes=0
          endif
          if(iyes.eq.0) then
          goto 2
          else
cxx        wth=1.0d0
          sq=step_d*step_d*DCOS(dw)
c      write(*,*) 'RRR',delta,d,step_d,wth,dwth
c
c Project zone point on a sphere
c
           ww(1)=dcos(d)
           ww(2)=dsin(d)
           ww(3)=dcos(dpi/2.0d0-dw)
           do k1=1,3
             e(k1)=0.0d0
             do k2=1,3
               e(k1)=e(k1)+a(k1,k2)*ww(k2)
             enddo
           enddo
c ---
c Compute model parameters and travel time fo reference model
c ---
        rbi_mod=1.0d0/RBIMOD_8(e)
        tc00=tc00+wth*sq*R*delta*rbi_mod
        wths1=wths1+wth*sq
        CALL ESTRACE_2(sq,lidd,nlidd,wth*delta,dwth,wnum,e,nmm,
     *                 denf,den1f,den2f,ff,i_ray,rbi_mod,ierr)
        wths=wths+wth*sq
          endif
          dw=-dw
          enddo
        enddo
    2 continue
      enddo
      enddo
c End main LOOP ==============================
cxx   xx=datan2(e(2),e(1))/drad
cxx   yy=90.0d0-dacos(e(3))/drad
      count=count+1
cxx   write(*,*) count,n_d,delta/drad, xx,yy
cxx   if(MOD(count,200).eq.0)write(*,*) count
      if(nlidd.eq.0) return
      do j=1,nlidd
        ncl=lidd(j)
        ff(ncl)=ff(ncl)/wths
c     write(*,*) 'S',ncl,ff(ncl)
c     write(*,*) 'D',ncl,denf(ncl),den1f(ncl),den2f(ncl)
      enddo
      tc00=tc00/wths1
c     write(*,*) delta,wths,wths1
      return
      end
c===============================================================
c Compute residual and travel time for solution, fsl, for single
c fresnel zone.
c===============================================================
      SUBROUTINE FR_RES(type,type1,delta,x1,x2,xn,i_ray,nmm,sum,fsl,n_p,ierr)
      implicit none
      include "tomo.h"
      include "line.h"
      character*4 type,type1
      integer*4 nmm,n_p
      real*4    fsl(nmm*NAZIPL)
      real*8    x1(3),x2(3),xn(3),w(3),ww(3),e(3),period
      real*8    d,dw,slo,delta,wth,wwth,wths,sum,value,rbi_mod
      real*8    step_d,RBIMOD_8,a(3,3),sq,Lambda
      real*8    Left(7),Right(7)
      integer*4 i,j,ierr,ierr1,iyes,i_ray,k1,k2,k,ke,n_w,n_d,Num(7),Ntab,kk
c get slowness from PREM model ==================
      period=tper
      CALL GET_SLOW(type,period,slo,ierr1)
      if(ierr1.ne.0) stop '(R0015) 1-D model period is out of range'
      Lambda=period/(slo*R)
      n_p=0
      sum=0.0d0
      wths=0.0d0
      CALL VECT(xn,x1,w)
      CALL NORM(w,w)
      do i=1,3
      a(i,1)=x1(i)
      a(i,2)=w(i)
      a(i,3)=xn(i)
      enddo
c
c    get intervals
c
      CALL GEN_INT(Lambda/x_zone/2.0d0,delta,Left,Right,Num,Ntab,ierr)
      if(ierr.eq.1) then
        n_p=0
        return
      endif
c
c Main loop
c
      do kk=1,Ntab
      n_d=Num(kk)
      step_d=(Right(kk)-Left(kk))/n_d      
      do i=1,n_d
        d=(i-1)*step_d+step_d/2.0d0+Left(kk)
        n_w=80.0d0*drad/step_d
        do j=1,n_w
          dw=(j-1)*step_d
          ke=1
          if(j.gt.1) ke=2
          do k=1,ke
          if(j.eq.1) then
c
c Where we are, inside or outside Fresnel zone ?
c
              CALL FUNCT(Lambda,delta,d,step_d,wth,iyes)
              if(iyes.eq.0) goto 2
              wwth=1.0d0/(2.0d0*wth)
          else
              iyes=1
              if(dw.gt.wwth) iyes=0
          endif
          if(iyes.eq.0) then
          goto 2
          else
           sq=step_d*step_d*DCOS(dw)
c
c Project zone point on a sphere
c
           ww(1)=dcos(d)
           ww(2)=dsin(d)
           ww(3)=dcos(dpi/2.0d0-dw)
           do k1=1,3
             e(k1)=0.0d0
             do k2=1,3
               e(k1)=e(k1)+a(k1,k2)*ww(k2)
             enddo
           enddo
c ---
c Compute model parameters and travel time fo reference model
c ---
        rbi_mod=1.0d0/RBIMOD_8(e)
        CALL EINTEGR_IZ(delta,e,nmm,value,fsl,ierr)
        if(ierr.ne.0) goto 2
        n_p=n_p+1
        sum=sum+wth*sq*value*rbi_mod
        wths=wths+wth*sq
          endif
          dw=-dw
          enddo
        enddo
    2 continue
      enddo
      enddo
c End main LOOP ==============================
      if(n_p.eq.0) return
      sum=sum/wths
      return
      end
c============================================================
c Estimate slovness for phase veloocity using model file
c============================================================
      subroutine GET_SLOW(type,period,slowness,ierr)
      implicit none
      include "tomo.h"
      character*4 type
      real*8    period,slowness,per(100),r_ph(100),l_ph(100)
      real*8    dum1,dum2
      integer*4 n_t,i,ierr,lnblnk
      logical   tf
      ierr=0
c
c Check that file fname5 exists
c
      inquire(file=fname5,exist=tf)
      if(.not.tf) then
        i=lnblnk(fname5)
        write(*,*) '(R0010) File ',fname5(1:i), ' does not exists'
        STOP
      endif
c
c Read input data
c
      open(26,file=fname5,status='old')
      read(26,*) n_t
      do i=1,n_t
        read(26,*) per(i),dum1,r_ph(i),dum2,l_ph(i)
      enddo
      close(26)
      if(type.eq.'R'.or.type.eq.'r') then
         do i=1,n_t-1
           if(period.ge.per(i).and.period.le.per(i+1)) then
             slowness=1.0/((r_ph(i+1)-r_ph(i))/(per(i+1)-per(i))*
     *       (period-per(i))+r_ph(i))
           goto 2
           endif
         enddo
         ierr=1
    2    continue
      else
         do i=1,n_t-1
           if(period.ge.per(i).and.period.le.per(i+1)) then
             slowness=1.0/((l_ph(i+1)-l_ph(i))/(per(i+1)-per(i))*
     *       (period-per(i))+l_ph(i))
           goto 3
           endif
         enddo
         ierr=1
    3    continue
      endif
      return
      end
c******************************************************************
c Look like ESTRACE but for fresnel zones
c******************************************************************
      SUBROUTINE ESTRACE_2(sq,lidd,nlidd,step,dwth,wnum,e,nmm,denf,
     *den1f,den2f,ff,i_ray,rbi_mod,ierr)
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      integer*4 nmm
      real*4    denf(nmm),den1f(nmm),den2f(nmm),ff(nmm)
      real*4    amed,dwth,wnum(nmm)
      real*8    grid(4,3),w(4),e(3),step,sq,rbi_mod
      integer*4 lidd(NRAZ),nclo(4),ngr_out(4)
      integer*4 i,j,ierr,ncl,nlidd,i_ray
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
      CALL E2PNTS(e,ngr_out,grid)
      CALL TRIAW(ngr_out,grid,e,w,ierr)
      do 20 j=1,4
      nclo(j)=ilocr(ngr_out(j))
      if(nclo(j).eq.0) then
      ierr=1
      return
      endif
   20 continue
      do 10 j=1,4
      ncl=nclo(j)
      if(ncl.gt.nm) then
        write(*,*)'ERROR with local number',nm,ncl
        ierr=1
        return
      endif
      do 30 i=1,nlidd
      if(lidd(i).eq.ncl) goto 40
   30 continue
      nlidd=nlidd+1
      lidd(nlidd)=ncl
   40 ff(ncl)=ff(ncl)+w(j)*step*R*sq*rbi_mod
      amed=denf(ncl)*wnum(ncl)
      amed=amed+dwth
      wnum(ncl)=wnum(ncl)+1.0
      denf(ncl)=amed/wnum(ncl)
      if(i_ray.eq.1) den1f(ncl)=denf(ncl)
      if(i_ray.eq.2) den2f(ncl)=denf(ncl)
   10 continue
      return
      end
