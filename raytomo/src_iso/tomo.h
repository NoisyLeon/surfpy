C**********************PARAMETERS**********************************
C-----NINP--number of input rays;
C-----NRAZ--number of rays after rejections;
C-----NALPH-number of regularization constants;
      integer*4  NRAZ, NINP, NALPH, IAZIM
cxx   parameter (NRAZ=7500,NINP=220000,NALPH=4,IAZIM=10)
cxx   parameter (NRAZ=12000,NINP=100000,NALPH=4,IAZIM=10)
      parameter (NRAZ=22000,NINP=800000,NALPH=4,IAZIM=10)
cxx   parameter (NRAZ=20200,NINP=700000,NALPH=4,IAZIM=10)
C----------------------COMMON /charac/-----------------------------
      character*160   namout,namein,outfile
      character*18   com
      common/charac/ namout,namein,outfile,com
C----------------------COMMON /log/--------------------------------
      logical     lwght,lsele,lres,lsymb,lreje,lrejd,lmodel,lresid,
     +            lcoord,lcoout
      common/log/ lwght,lsele,lres,lsymb,lreje,lrejd,lmodel,lresid,
     +            lcoord,lcoout
C----------------------COMMON /input/------------------------------
      real*4        T(NINP),TE0(NINP),FI0(NINP),TE(NINP),FI(NINP)
      real*4        WEIGHT(NINP),TINMOD(NINP),resid_i(NINP)
      integer*4     IIII(NINP)
      common/input/ T,TE0,FI0,TE,FI,WEIGHT,IIII,TINMOD,resid_i
C-----------------------COMMON /const/-----------------------------
      real*4        R,geo,pi,const,reject,area
      common/const/ R,geo,pi,const,reject,area
C-----------------------COMMON /ilog/------------------------------
      integer*4    iwght,isele,ires,isymb,ireje,irejd,imodel,iresid,
     +             icoord,icoout
      common/ilog/ iwght,isele,ires,isymb,ireje,irejd,imodel,iresid,
     +             icoord,icoout
C-----------------------COMMON /iperi/------------------------------
      real*4        period_n
      common/iperi/ period_n
C-----------------------COMMON /wor/--------------------------------
      real*8      f2(181,360)
      common/wor/ f2
C-----------------------COMMON /men/--------------------------------
      real*8      step0,pow
      real*4      cell,alph1(NALPH),dlat0,dlat_n,
     +            s_lat,dlon0,dlon_n,s_lon,eqlat0,eqlon0,eqlat_n,
     +            eqlon_n,slat0,slat_n,slon0,slon_n,wexp,tper
      integer*4   ncoin0,ipath,n_pnt,nlat,nlon,nwavep
      common/men/ step0,pow,ncoin0,cell,ipath,alph1,dlat0,dlat_n,
     +            s_lat,dlon0,dlon_n,s_lon,eqlat0,eqlon0,eqlat_n,
     +            eqlon_n,slat0,slat_n,slon0,slon_n,n_pnt,nlat,
     +            nlon,wexp,tper,nwavep
C-----------------------COMMON /param1/------------------------------
      integer*4      M,MM,N,NOUT,NXY,NX1,NY1,INDEX,KOD,KNT
      real*4         c00
      common/param1/ M,MM,N,NOUT,NXY,NX1,NY1,INDEX,KOD,KNT,c00
C-----------------------COMMON /name/--------------------------------
      character*160 fname1,fname2,fname3,fname4
      common/name/  fname1,fname2,fname3,fname4
C********************************************************************
