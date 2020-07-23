c*******************************************************************
c  NINP  -  max number of input rays
c  NRAZ  -  max number of isotropic grids
c  NRAZA -  max number of anisotropic grids
c  NAZIPL - 1, 3 or 5. 1 - 0PSI, 3 - 2PSI, 5 - 4PSI
c*******************************************************************
      integer*4  NRAZ, NRAZA, NINP, NALPH, NAZIPL, IAZIM
cxx   parameter (NRAZ=2700,NINP=15000,NALPH=4,NAZIPL=3,IAZIM=10)
cxx   parameter (NRAZ=8000,NINP=700000,NALPH=4,NAZIPL=3,IAZIM=10)
cxx   parameter (NRAZ=7410,NINP=300000,NALPH=4,NAZIPL=3,IAZIM=10)
cxx   parameter (NRAZ=2650,NRAZA=1020,NINP=70000,NALPH=4,NAZIPL=3,IAZIM=10)
      parameter (NRAZ=8740,NRAZA=3740,NINP=1300000,NALPH=4,NAZIPL=3,IAZIM=10)
c----------------------COMMON /charac/-----------------------------
      character*160 namout,namein,outfile(NALPH)
      character*18 com
      common/charac/namout,namein,outfile,com
c----------------------COMMON /log/--------------------------------
      logical       lwght,lsele,lres,lsymb,laniz,lreje,lrejd,lmodel,
     +              lresid,lcoord,lcoout,ldresol,ltomo,lresol,lold,
     +              ltrace,lpoint
      common/log/   lwght,lsele,lres,lsymb,laniz,lreje,lrejd,lmodel,
     +              lresid,lcoord,lcoout,ldresol,ltomo,lresol,lold,
     +              ltrace,lpoint
c----------------------COMMON /input/------------------------------
      real*4    T(NINP),TE0(NINP),FI0(NINP),TE(NINP),FI(NINP)
      real*4    WEIGHT(NINP),TINMOD(NINP),resid_i(NINP)
      integer*4 IIII(NINP),IRAY2(NINP)
      common/input/ T,TE0,FI0,TE,FI,WEIGHT,IIII,IRAY2,TINMOD,resid_i
c-----------------------COMMON /const/-----------------------------
      real*4        R,geo,pi,const,reject,area
      common/const/ R,geo,pi,const,reject ,area
c-----------------------COMMON /ilog/------------------------------
      integer*4 iwght,isele,ires,isymb,ianiz,ireje,irejd,imodel,NAZIP,
     +          iresid,icoord,icoout,idresol,itomo,iresol,iold,
     +          itrace,ipoint
      common/ilog/iwght,isele,ires,isymb,ianiz,ireje,irejd,imodel,NAZIP,
     +          iresid,icoord,icoout,idresol,itomo,iresol,iold,
     +          itrace,ipoint
c-----------------------COMMON /iperi/------------------------------
      real*4    Tper
      common/iperi/ Tper
c-----------------------COMMON /wor/--------------------------------
      real*8    f2(181,360)
      common/wor/ f2
c-----------------------COMMON /men/--------------------------------
      real*4    alph1(NALPH),alsi(3,5)
      real*8    step0,x_zone
      real*4    cell,cell_a,dlat0,dlat_n,s_lat,dlon0,dlon_n,s_lon,
     +eqlat0,eqlon0,eqlat_n,eqlon_n,slat0,slat_n,slon0,slon_n,wexp,
     +dump1,dump2
      character*4 re_la,gr_pha
      integer*4 istrip,ncoin0,ireg,ipath,n_pnt,na_pnt,nlat,nlon,nwavep
      common/men/ step0,x_zone,ncoin0,cell,cell_a,ireg,ipath,alph1,istrip,
     +alsi,dlat0,dlat_n,s_lat,dlon0,dlon_n,s_lon,eqlat0,eqlon0,eqlat_n,
     +eqlon_n,slat0,slat_n,slon0,slon_n,n_pnt,na_pnt,nlat,nlon,wexp,
     +dump1,dump2,nwavep,re_la,gr_pha
c-----------------------COMMON /symb_map/---------------------------
      real*4 PMIN(20),PMAX(20)
      real*4 PMN,PMX,PMN_RES,PMX_RES
      common/symb_map/PMIN,PMAX,PMN,PMX,PMN_RES,PMX_RES
c-----------------------COMMON /param1/------------------------------
      integer*4 M,MM,MMAIN,MAIN1,N,NMAX,NOUT,NXY,NX1,NY1,INDEX,KOD,KNT
      real*4    c00
      common/param1/ M,MM,MMAIN,MAIN1,N,NMAX,NOUT,NXY,NX1,NY1,INDEX,
     +KOD,KNT,c00
c-----------------------COMMON /pole/--------------------------------
      real*8 pf,pl,azpole(3)
      common/pole/pf,pl,azpole
c-----------------------COMMON /name/--------------------------------
      character*160 fname1,fname2,fname5,fname6,root
      common/name/  fname1,fname2,fname5,fname6,root
c********************************************************************
