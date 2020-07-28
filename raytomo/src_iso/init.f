      SUBROUTINE INIT
      IMPLICIT NONE
      include "tomo.h"
      include "version.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      CHARACTER*20 period
      CHARACTER*1  smodel,ssele,swght,sres,ssymb,sreje,sresid
      CHARACTER*1  srejd,scoord,scoout
      CHARACTER*1 TEX
      integer*4    narg,iargc,lo,lnblnk,lp,l1,l2
c END TYPE definitions ++++++++++++++++++++++++++++++++++
C-----------------------DATA-------------------------------------
      DATA R/6371./,GEO/0.993277/,PI/3.14159265/,CONST/0.0174533/
C---------------------------------------------------------------
       narg=iargc()
       if (narg.ne.3) STOP ' USAGE: tomograph INFILE OUTFILE PERIOD'
       print*,'tomographic processing'
       print*,'optimal rejection of doubtful data'
       call GETARG(1,namein)
       call GETARG(2,namout)
       call GETARG(3,period)
       read(period,*) period_n
       lo=lnblnk(namout)
       lp=lnblnk(period)
       if(period(lp:lp).eq.'.') period(lp:lp)=' '
       lp=lnblnk(period)
       read(period,*) Tper
       call SET_DEFAULTS
1      open(unit=7,file=namein,status='OLD')
       open(unit=8,file=namout(1:lo)//'_'//period(1:lp)//'.prot',
     + status='UNKNOWN')
c               command routing
       print 1000
1000   format('tomo > ',$)
	read(*,1001) com
1001   format(a18)
C------main commands-------------------------------------
C------to exit-------------------------------------------
       if(com(1:1).eq.'q'.or.com(1:2).eq.'ex') goto 99
       if(com(1:1).eq.'h'.or.com(1:1).eq.'?') CALL HELP
       if(com(1:1).eq.'def') CALL SET_DEFAULTS
C---to change defaults----------------------------------
       if(com(1:2).eq.'me') CALL MENU('m')
C---to print menu   ----------------------------------
       if(com(1:1).eq.'v') CALL MENU('v')
       if(com(1:1).eq.'q'.or.com(1:2).eq.'ex') goto 99
C-----to start computations---------------------------
       if(com(1:2).eq.'go') goto 100
       go to 1
100    write(*,*) ' '
       outfile=namout(1:lo)//'_'//period(1:lp)//'.1'
       open(unit=10,file=outfile,status='UNKNOWN')
       if(lres) then
       fname3=namout(1:lo)//'_'//period(1:lp)//'.'//'res'
       fname4=namout(1:lo)//'_'//period(1:lp)//'.'//'eql'
       open(unit=9,file=fname3,status='UNKNOWN')
       open(unit=11,file=namout(1:lo)//'_'//period(1:lp)
     * //'.'//'azi',status='UNKNOWN')
       endif
       if(lresid)
     * open(unit=35,file=namout(1:lo)//'_'//period(1:lp)
     * //'.'//'resid',status='UNKNOWN')
       if(lmodel)
     + open(unit=16,file=fname2,status='OLD')
       smodel=tex(lmodel)
       sresid=tex(lresid)
       scoord=tex(lcoord)
       scoout=tex(lcoout)
       ssele=tex(lsele)
       swght=tex(lwght)
       sres=tex(lres)
       ssymb=tex(lsymb)
       sreje=tex(lreje)
       srejd=tex(lrejd)
      write(8,'("Protocol: ",a/30(1H=))') VERSION
      write (8,2000), period(1:lp)
2000  FORMAT(' PERIOD= ',a,' (sec)' )
      l1=lnblnk(namein)
      l2=lnblnk(namout)
      write(8,2002),namein(1:l1),namout(1:l2)
2002  FORMAT( ' INFILE=',a/' OUTFILE=',a)
      write(8,2003) smodel,swght,sres,ssele,ssymb,sreje,srejd,
     *sresid,scoord,scoout
2003  format(' model= ',a1,' '/' weights= ',a1,' '/
     +' resolution= ',a1/' selection= '
     +,a1 /' percent_map= ',a1/' rejection= ',a1/
     +' reject_delta= ',a1/' residuals= ',a1,/' geogr-->geoc= ',a1/
     +' geoc-->geogr= ',a1)
      if(imodel.ne.0) then
        l1=lnblnk(fname2)
        write (8,2014) fname2(1:l1)
      endif
      l1=lnblnk(fname1)
      write (8,2013) fname1(1:l1),pow
2013  format(' Contour file name is: ',a/
     +' Gaussian cutting acuracy, pow=',f5.1)
2014  format(' Model file name is: ',a)
      write (8,2004) alph1(1)
      write (8,2010) alph1(3)
      write (8,2009) alph1(2)
      write (8,2012) alph1(4)
      if(lreje)write (8,*) 'rejection level=',reject,' %'
2004  format(' Regularization parameter 1 (alpha1) are: ',f8.4)
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
2007  format(' End points of rays are  between latitudes : ',2(F8.2,1X),/
     *       '                                 longitudes: ',2(F8.2,1X))
2009  format(' Radius of correlation (sigma1) are: ',f8.3)
2010  format(' Regularization parameter 2 (alpha2) are: ',f8.3)
      write (8,2011) step0
2011  format(' Step of integration:',f7.4,' (deg)')
2012  format(' MAX Radius of correlation (sigma2) are: ',f8.3)
C-------------------INITIATION------------------------------------E
      return
99    INDEX=2
      return
      end
C########################################################
      FUNCTION TEX(inp)
      IMPLICIT NONE
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      logical inp
      character*1 TEX
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      tex='N'
      if(inp) tex='Y'
      return
      end
