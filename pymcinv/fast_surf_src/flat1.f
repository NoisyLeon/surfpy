C-------------------------------------------------------
      subroutine flat1(h,ro,vp,vs,n,kind)
C-------------------------------------------------------

c     subroutine applies earth-flattening corrections by
c     scaling the model.for reference see Biswas 1972.
c     kind=1 for Love wave calculations
c     kind=2 for Rayleigh wave calculations
c     h -is thickness

c-input h,ro,vp,vs,n,kind; output h1,ro1,vp1,vs1.
C     TRANSFORMATION
C-----R(i)-radius of i-th boundary (free surfase is ignored)
C-----h(i) by R0*(ln(R0/R(i))-ln(R0/R(i-1));R(0)=R0
C-----vs,vp mult by factor "dif"=R0*(1/R(i+1)-1/R(i))/ln(R(i)/R(i+1))
C-----ro  mult by factor (R(i)**pwr-R(i+1)**pwr)/(ln(R(i)/R(i+1))*pwr*R0**pwr)
C-----for Rayleigh pwr=2.275 
C-----for Love     pwr=5.000  

      real*4 h(2),ro(2),vp(2),vs(2),hh(1000)
      data a/6371.0/
c      write(*,*) 'input vs vp:',vs(1),vp(1) LF
      do i=1,n
        hh(i)=h(i)
cc	write(*,*) "hh and h",hh(i),h(i)
      enddo
      pwr=2.2750
      if(kind.eq.1) pwr=5.0
      nm=n-1
      hs=0.0

C-----transfer thickness in radius------
      do 5 i=1,n
        ht=hs
        hs=hs+hh(i)
c	write(*,*) "hhi::::: radias:::",a,hs,hh(i),ht,a-ht
    5   hh(i)=a-ht
c	write(*,*) "hhi::::: radias:::",hh(i)
C-----now hh(i) are radii  (starting from the first boundary)
c -layers' scaling
      do 10 i=1,nm

        ii=i+1
        fltd=alog(hh(i)/hh(ii))
c	write(*,*) "hhi, hhii:",hh(i),hh(ii)
        dif=(1.0/hh(ii)-1.0/hh(i))*a/fltd
        difr=hh(i)**pwr-hh(ii)**pwr
        qqq=difr/(fltd*(a**pwr)*pwr)
        ro(i)=ro(i)*qqq
        vp(i)=vp(i)*dif

c        if (i.eq.1) then
c          write(*,*) "vsi and dif and fltd hhi,hhii:",vs(i),dif,fltd,hh(i),hh(ii)
c	endif

   10 vs(i)=vs(i)*dif
c half space scaling
      fact=a/hh(n)
      vp(n)=vp(n)*fact
c      write(*,*) 'fact,',fact LF
      vs(n)=vs(n)*fact
      ro(n)=ro(n)*(1.0/fact)**pwr
      z0=0.0
C-----new thicknesses----------------------------
      do 15 i=2,n
        z1=a*alog(a/hh(i))
        h(i-1)=z1-z0
   15   z0=z1
      h(n)=0.0
c      write(*,*) "vs in flat!",vs(1),vp(1) LF

      return
      end
