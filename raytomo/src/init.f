c*******************************************************************
c Initialize program constants and parameters
c*******************************************************************
      SUBROUTINE INIT
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      include "version.h"
c TYPE definitions +++++++++++++++++++++++++++++++++++++++++++++++++
      CHARACTER*20 period
      CHARACTER*1  sym,smodel,ssele,swght,sres,ssymb,saniz,tex,sreje,
     +             srejd,sresid,scoord,scoout,sdresol,stomo,sresol,
     +             sold,strace,spoint
      integer*4	   i,narg,iargc,li,lo,lnblnk,lp,l1,l2,lr
      logical      tf
c END TYPE definitions +++++++++++++++++++++++++++++++++++++++++++++
c
c   Constants definition:
c   R     - average radius of the Earts, (km);
c   GEO   - elliptisity coefficient;
c   PI    - pi;
c   CONST - factor to convert degree to radians
c
      DATA R/6371./,GEO/0.993277/,PI/3.14159265/,CONST/0.0174533/
      DATA PMN/2.5/,PMX/4.6/,PMN_RES/200./,PMX_RES/2300./

      dpi=4.0d0*DATAN(1.0d0)
      drad=dpi/180.d0
      narg=iargc()
      if (narg.ne.3) STOP ' Usage: tomo_sp_cu_s INFILE OUTFILE PERIOD'
      call getarg(1,namein)
      call getarg(2,namout)
      call getarg(3,period)
      read(period,*) Tper
      lo=lnblnk(namout)
      lp=lnblnk(period)
      if(period(lp:lp).eq.'.') period(lp:lp)=' '
      lp=lnblnk(period)
      call SET_DEFAULTS
c
c setup root prefix for file path
c
      root=namout(1:lo)//'_'//period(1:lp)
      lr=lnblnk(root)
c Check for existance of namein
      inquire(file=namein,exist=tf)
      if(.not.tf) then
        i=lnblnk(namein)
        write(*,*) '(R0010) File ',namein(1:i), ' does not exists'
        STOP
      endif
      open(unit=7,file=namein,status='OLD')
      open(unit=8,file=root(1:lr)//'.prot',status='UNKNOWN')
c               command routing
   1  write(*,1000)
1000  format('tomo > ',$)
      read(*,1001) com
1001  format(a18)
c
c   main commands
c
      if(com(1:1).eq.'q'.or.com(1:2).eq.'ex') goto 99
      if(com(1:1).eq.'h'.or.com(1:1).eq.'?') CALL HELP
      if(com(1:1).eq.'def') CALL SET_DEFAULTS
      if(com(1:2).eq.'me') CALL MENU('m')
      if(com(1:1).eq.'v') CALL MENU('v')
      if(com(1:1).eq.'q'.or.com(1:2).eq.'ex') goto 99
c
c   start computations
c
      if(com(1:2).eq.'go') goto 100
      go to 1
100   write(*,*) ' '
      do i=1,ireg
      if(i.lt.10)write (sym,'(i1)')i
      if(i.eq.10)write (sym,'(i2)')i
      li=lnblnk(sym)
      outfile(i)=root(1:lr)//'.'//sym(1:li)
      open(unit=9+i,file=outfile(i),status='UNKNOWN')
      enddo
      if(lres) then
      open(unit=9,file=root(1:lr)//'.res',status='UNKNOWN')
      open(unit=11,file=root(1:lr)//'.azi',status='UNKNOWN')
      endif
      if(lresol) then
        open(unit=22,file=root(1:lr)//'.rea',status='UNKNOWN')
      endif
      if(lresid)
     *open(unit=35,file=root(1:lr)//'.resid',status='UNKNOWN')
      if(ldresol.or.lresol) 
     +   fname6=root(1:lr)//'.dr'
      if(ireg.gt.1)open(66,file='TEMP_MAT',status='UNKNOWN')
      if(lmodel) then
        inquire(file=fname2,exist=tf)
        if(.not.tf) then
          i=lnblnk(fname2)
          write(*,*) '(R0010) File ',fname2(1:i), ' does not exists'
          STOP
        endif
        open(unit=16,file=fname2,status='OLD')
      endif
      MMAIN=ireg
      smodel= tex(lmodel)
      sdresol=tex(ldresol)
      sresid= tex(lresid)
      scoord= tex(lcoord)
      scoout= tex(lcoout)
      sresol= tex(lresol)
      ssele=  tex(lsele)
      swght=  tex(lwght)
      sres=   tex(lres)
      ssymb=  tex(lsymb)
      saniz=  tex(laniz)
      sreje=  tex(lreje)
      srejd=  tex(lrejd)
      stomo=  tex(ltomo)
      sold=   tex(lold)
      strace= tex(ltrace)
      spoint= tex(lpoint)
      if(lsymb) then
      do i=1,ireg
      PMIN(i)=PMN
      PMAX(i)=PMX
      end do
      if(lres) then
      PMIN(ireg+1)=PMN_RES
      PMAX(ireg+1)=PMX_RES
      end if
      end if
      NMAX=ipath
      if(itomo.eq.0)write(8,'("Protocol: ",a," (Gaussian rays)"/56(1H=))') VERSION
      if(itomo.eq.1)write(8,'("Protocol: ",a," (Fresnel zones)"/56(1H=))') VERSION
      if(itomo.eq.0)write(*,'("Protocol: ",a," (Gaussian rays)"/56(1H=))') VERSION
      if(itomo.eq.1)write(*,'("Protocol: ",a," (Fresnel zones)"/56(1H=))') VERSION
      write (8,2000), period(1:lp)
2000  FORMAT(' PERIOD= ',a,'(sec)' )
      l1=lnblnk(namein)
      l2=lnblnk(namout)
      WRITE(8,2002),namein(1:l1),namout(1:l2)
2002  FORMAT( ' INFILE=',a,6x,/,' OUTFILE=',a)
      write(8,2003) smodel,swght,sres,ssele,ssymb,saniz,2*ianiz+istrip,sreje,
     +srejd,sresid,sdresol,scoord,scoout,stomo,sresol,sold,strace,spoint
2003  format(' model=   ',a1,' '/' weights=   ',a1,' '/
     +' dens & azimuth maps=',a1/' selection= '
     +,a1 /' percent map=',a1/' anisotropy=',a1,' ',i1,'-psi type'/
     +' rejection by %=',a1/' rejection by dist=',a1/' residuals= ',a1,/
     +' covariance matrix= ',a1,/' geogr-->geoc= ',a1/' geoc-->geogr= ',a1/
     +' Fresnel zone= ',a1,/' resolution analysis = ',a1,/
     +' input data format = ',a1,/' RAY TRACER mode = ',a1,/
     +' resolution response maps = ',a1)
      if(imodel.ne.0) then
         l1=lnblnk(fname2)
         write (8,2014) fname2(1:l1)
      endif
      l1=lnblnk(fname1)
      write (8,2013) fname1(1:l1)
2013  format(' Contour file name is: ',a)
2014  format(' Model file name is: ',a)
      write(8,2004) alsi(1,1),alsi(1,2),alsi(1,3),alsi(1,4)
      if(istrip.eq.2) then
        alsi(2,1) = alsi(3,1)
        alsi(2,3) = alsi(3,3)
        alsi(2,4) = alsi(3,4)
      endif
      if(ianiz.gt.0.and.istrip.eq.0) write(8,2010) alsi(2,1),alsi(2,3),alsi(2,4)
      if(ianiz.gt.0.and.istrip.eq.2) write(8,2009) alsi(2,1),alsi(2,3),alsi(2,4)
      if(ianiz.gt.1) write(8,2009) alsi(3,1),alsi(3,3),alsi(3,4)
      write(8,2012) alsi(1,5),alsi(2,5),alsi(3,5)
2004  format(' 0PSI: alpha1, alpha2, sigma1, sigma2 ',f8.3,f7.3,2f9.3)
2010  format(' 2PSI: alpha1,         sigma1, sigma2 ',f8.3,7x,2f9.3)
2009  format(' 4PSI: alpha1,         sigma1, sigma2 ',f8.3,7x,2f9.3)
2012  format(' 0PSI-prec, 2PSI-prec, 4PSI-prec',3f8.1)
      if(lreje)write (8,*) 'rejection level=',reject,' %'
      nlat=(dlat_n-dlat0)/s_lat+1.5
      dlat_n=s_lat*(nlat-1)+dlat0
      nlon=(dlon_n-dlon0)/s_lon+1.5
      dlon_n=s_lon*(nlon-1)+dlon0
      write (8,2005) dlat0,dlat_n,s_lat ,nlat
2005  format(' Limits of latitude:  ',2(F10.2,1X),' Increment=',
     *F6.3,' npoints=',i5)
      write (8,2006) dlon0,dlon_n,s_lon ,nlon
2006  format(' Limits of longitude: ',2(F10.2,1X),' Increment='
     *,F6.3,' npoints=',i5)
      if(lsele)write(8,2007)slat0,slat_n,slon0,slon_n
2007  format('End points of rays are  between latitudes : ',2(F8.2,1X),/
     *       '                                longitudes: ',2(F8.2,1X))
      write (8,2020) step0
2020  format(' Step of integration:',f7.4)
      write (8,2011) x_zone
2011  format(' X-zone:',f7.4)
      write (8,2016) re_la
2016  format(' Rayleigh/Love: ',a1)
      write (8,2018) gr_pha
2018  format(' Group/Phase velocity: ',a1)
      l1=lnblnk(fname5)
      write (8,2017) fname5(1:l1)
2017  format(' Model for slowness: ',a)
      azpole(1)=DSIN((90.0d0-pf)*drad)*DCOS(pl*drad)
      azpole(2)=DSIN((90.0d0-pf)*drad)*DSIN(pl*drad)
      azpole(3)=DCOS((90.0d0-pf)*drad)
      write (8,2021) pf,pl
2021  format(' Azimus pole: lat =',F8.2,', lon =',F8.2)
C-------------------INITIATION------------------------------------E
      return
99    INDEX=2
      return
      end
C********************************************************
      function tex(inp)
      IMPLICIT NONE
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      logical     inp
      character*1 tex
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      tex='N'
      if(inp) tex='Y'
      return
      end
