	     parameter ( maxpath = 2000000)

	     character*180 map_name,pathfile_name,perlist,map_file, lfevent
	     character*11 word
	     character*1 wavetype
	     character*3 sym
	     character*8 stnam(maxpath)
             character*16 evnam(maxpath)
c	     character*6 cdummy
             parameter (nmax=2000)
             real*4 per(50),grvel_R(50,maxpath),
     1      grvel_L(50,maxpath),vel(270000)
             real*4  stalat(maxpath),stalon(maxpath),
     1               evlat(maxpath),evlon(maxpath)
             real*4 xalat(nmax),xalon(nmax),xray(nmax)
	     dimension ista(maxpath), iev(maxpath), step_sv(maxpath)
             parameter (nsize=70000)
             logical lf
             real*8 f2
             common/mod/f2(181,360)

	     data marg/4/
c	     data marg/3/,step/0.05/
             data  dlat_min/-90./,dlat_max/90.0/,
     1              dlon_min/-180.0/,dlon_max/180./
             data word/'PREDICTION_'/
             data const/2.0/,iconst/5/
C----------------------------------------------------------------------------------------

	     narg=iargc()
	     if(narg.ne.marg)STOP'USAGE: mhr_grvel_predict pathfile_name  map_name  perlist event_name'
             Print*,'Produces group velocity curves from grv_maps
     1             for a given wavetype and a list of periods'
C--------------------initiation--------------------S
c		note: pathfile_name -- is the name of the station location list (mhr)
             call GETARG(1,pathfile_name)
             call GETARG(2,map_name)
             call GETARG(3,perlist)
             call GETARG(4,lfevent)
             open(3,file=perlist,status='OLD')
	     do i=1,100
	     read(3,*,end=99)per(i)
c	      print *, i, per(i)
             enddo
99           nper=i-1
	     close(3)
c             PRint*,nper,' periods requested'
             lma=lnblnk(map_name)
             open(2,file=word//'R'//'_'//lfevent)
             open(12,file=word//'L'//'_'//lfevent)
C--------------------initiation--------------------E


C--------search for paths-----------------S
	      print*, 'before calll path_searcher'
             call path_searcher(pathfile_name,ista,stalat,
     1          stalon,stnam,iev,evlat,evlon,evnam,npath)
C--------search for paths-----------------E


C-----loop for period-----------------------------------------------------S
C---------------first for R, then for L--------
             DO I=1,nper

              kkk=0
              iper=per(i)
              if(iper.lt.10) write(sym,'(i1)')iper
              if(iper.ge.10.and.iper.lt.100) write(sym,'(i2)')iper
              if(iper.ge.100) write(sym,'(i3)')iper
              lsy=lnblnk(sym)
1000          if(kkk.eq.0)then
               wavetype='R'
               if(i.eq.1)mmm=1
              elseif(kkk.eq.1)then
               wavetype='L'
               mmm=2
              endif
      	     if(mmm.eq.1) print *, ' number of input rays       =',npath
      	     print *,'period=',per(i),' ', wavetype

C------------reading input grid file -------------------------------S
              map_file=map_name(1:lma)//'_'//wavetype//'_'//sym(1:lsy)
              INQUIRE (file=map_file,exist=lf)
              if(lf.eqv. .false.) THEN
      	       PRINT*,'No file ',map_file(1:lnblnk(map_file)),' exist'
	       goto 999
	      endif
              open(1,file=map_file,status='OLD')
              do ilat=1,181
              do ilon=1,360

              read(1,*,end=9898)dum1,dum2,veloc
c		write(*,*) dum1,dum2,veloc
              f2(182-ilat,ilon)=veloc
              enddo
              enddo
9898          close(1)
c              print *,'read ok!'
              if(ilat.ne.182.or.ilon.ne.361)STOP"Wrong Map Size"

C------------reading input grid file -------------------------------E


C------------ loop for rays---------------------------S
      	     do j=1,npath
      	      slat=stalat(j)
      	      slon=stalon(j)
      	      elat=evlat(j)
      	      elon=evlon(j)
      	      if(elon.gt.180.0)elon=elon-360.

      	      if(slon.gt.180.0)slon=slon-360.

c			set step size adaptively (mhr)
c	     print *, 'before call azidl_deg',SLAT,SLON,ELAT,ELON
      	     call AZIDL_deg(SLAT,SLON,ELAT,ELON,DEL,DELS,AZIS,AZIE)
	     step = 0.5
	     step = min(del/20., 0.5)
	     if(step.lt.0.001) step  = 0.001
	     step_sv(j) = step
c             if (dels .eq. 0.) continue
c	     print *, 'Dist, step: ',dels, del, step

      	      call path(step,elat,elon,slat,slon,
     1             xalat,xalon,xray,npoints,i,IND)
c	     print *, 'functions path are called over! '

      	      call project (npoints,xalat,xalon,vel)
c	     print *, 'functions project are called over! ',npoints,t

c             do k=1,npoints
c               PRint*, k, vel(k),xray(k)
c             enddo

      	      call integr(xray,vel,npoints,t)
c	     print *, 'functions are called over! '
      	      if(kkk.eq.0)grvel_R(i,j)=t
      	      if(kkk.eq.1)grvel_L(i,j)=t
      	     enddo
C------------loop for rays---------------------------E

      	     kkk=kkk+1
      	     if(kkk.eq.1)go to 1000
 999	     continue
            ENDDO
C-----loop for period-----------------------------------------------------E

      	    do j=1,npath
       	      write(2,36) iev(j),ista(j), nper, evnam(j),'  ',stnam(j),
     1              evlat(j), evlon(j), stalat(j), stalon(j)
       	      write(12,36) iev(j),ista(j),nper, evnam(j),'  ',stnam(j),
     1                   evlat(j), evlon(j), stalat(j), stalon(j)

cxx    	      write(12,*) ista(j),iev(j),' ',stnam(j),' ',evnam(j), nper
	      if(step_sv(j).gt.0.001) then
       	       do i=1,nper
        	write(2,'(F10.1,F10.4)') per(i),grvel_R(i,j)
        	write(12,'(F10.1,F10.4)') per(i),grvel_L(i,j)
       	       enddo
	      else
       	       do i=1,nper
        	write(2,'(F10.1,F10.4)') per(i),-999.
        	write(12,'(F10.1,F10.4)') per(i),-999.
       	       enddo
	     endif
      	    enddo

            PRINT*,'ALL IS DONE for ',npath,' paths'
36          format(2i5,i3,1x,a15,a2,a15,4f9.3)

           STOP
           end


C#####################################################################
            subroutine path_searcher(pathfile_name,ista,stalat,
     1      stalon,stnam,iev,evlat,evlon,evnam,npath)
            character*180 pathfile_name,curr_name
            character*8 stnam(30000)
            character*16 evnam(300000)
            real*4 stalat(300000),stalon(300000),
     1                         evlat(300000),evlon(300000)
	    dimension ista(300000), iev(300000)
            logical lf
	     curr_name = pathfile_name
            INQUIRE (file=curr_name,exist=lf)
            if(lf) then
             open(7,file=curr_name,status='OLD')
             go to 4
            endif
            print*,'No path found in ',pathfile_name
            STOP
C------------------------read in station coordinates----- from pathfile (mhr)
4           do j=1,2000000
            read(7,2,end=999) iev(j),ista(j),evnam(j),
     1           stnam(j),evlat(j),evlon(j),stalat(j),stalon(j)
c		print *, iev(j),ista(j),' ',evnam(j),stnam(j),
c     1             evlat(j),evlon(j),stalat(j),stalon(j)
            enddo
999         close(7)
 2	  format(2i5,1x,a15,1x,a15,f10.5,1x,f10.5,1x,f10.5,1x,f10.5)

	    npath=j-1
            return
            end

c####################################################################
      SUBROUTINE AZIDL_deg(SLAT1,SLON1,ELAT1,ELON1,DEL,DELS,AZIS,AZIE)
C     AZIS IS AZIMUTH FROM STATION, AZIE - FROM EPICENTER
C     DELS IS THE DISTANCE OVER THE SURFACE OF A SPHERICAL EARTH
C     GEOCENTRIC LATITUDES ARE USED IN ALL COMPUTATIONS
C     ALL IS IN RADIANS. Del is in radians. Dels is in km.
c	Note: i/o modified to degrees. (mhr)
c	AZIS is the backazimuth -- the angle from north where the
c		meridional line goes through the "source": (slat1, slon1).
c	AZIE, is the azimuth from the second station, back to the first.
      DATA R/6371./,PI/3.14159265359/,GEO/0.993277/
C     COLAT(A)=1.57079632679-ATAN(GEO*TAN(A))
      COLAT(A)=1.57079632679-A
C------------------------------------------------------------------
	deg2rad = pi/180.
	rad2deg = 180./pi

	slat = slat1 * deg2rad
	slon = slon1 * deg2rad
	elat = elat1 * deg2rad
	elon = elon1 * deg2rad

      SCOLAT=COLAT(SLAT)
      ECOLAT=COLAT(ELAT)
      C=ELON-SLON
      SC5=COS(C)
      SC6=SIN(C)
      SC1=SIN(SCOLAT)
      SC2=COS(SCOLAT)
      SC3=SIN(SLON)
      SC4=COS(SLON)
      EC3=SIN(ELON)
      EC4=COS(ELON)
      EC1=SIN(ECOLAT)
      EC2=COS(ECOLAT)
      AE=EC1*EC4
      BE=EC1*EC3
C________AZIMUTHS CALCULATION__________________________________
      AZI1=(AE-SC3)**2+(BE+SC4)**2+EC2*EC2-2.
      AZI2=(AE-SC2*SC4)**2+(BE-SC2*SC3)**2+(EC2+SC1)**2-2.
      IF(AZI2.EQ.0.) GO TO 10
      AZIS=ATAN2(AZI1,AZI2)
      GO TO 20
10    AZIS=3.141592653-SIGN(1.570796326,AZI1)
20    CONTINUE
      AS=SC1*SC4
      BS=SC1*SC3
      AZI1=(AS-EC3)**2+(BS+EC4)**2+SC2*SC2-2.
      AZI2=(AS-EC2*EC4)**2+(BS-EC2*EC3)**2+(SC2+EC1)**2-2.
      IF(AZI2.EQ.0.) GO TO 30
      AZIE=ATAN2(AZI1,AZI2)
      GO TO 40
30    AZIE=3.141592653-SIGN(1.570796326,AZI1)
40    CONTINUE
C__________DELTA CALCULATION_____________________________________
      COSD=SC2*EC2+SC1*EC1*SC5
      DEL= ACOS(COSD)
      DELS=R*DEL
	del = del * rad2deg
	azie = azie * rad2deg
	azis = azis * rad2deg
      RETURN
      END
