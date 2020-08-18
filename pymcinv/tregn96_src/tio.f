c-----
c     ptsmdt - from tdisp96 and is the flattened model of original sperical
c     gtsmdt - from tdisp96 and is the flattened model of original sperical
c-----
c-----
c       common routines for dispersion files for TI
c-----
c       lorr = 1 ISO Love
c       lorr = 2 ISO Rayleigh
c       lorr = 3 ISO Love for SLAT2D
c       lorr = 4 ISO Rayleigh for SLAT2D
c       lorr = 5 TI  Love
c       lorr = 6 TI  Rayleigh
c-----
c       for ISO we have
c       set ipar(1)  = 1 if medium is spherical
c       set ipar(2)  = 1 if source is in fluid
c       set ipar(3)  = 1 if receiver is in fluid
c       set ipar(4)  = 1 if eigenfunctions are output with -DER flag
c       set ipar(5)  = 1 if dc/dh are output with -DER flag
c       set ipar(6)  = 1 if dc/da are output with -DER flag
c       set ipar(7)  = 1 if dc/db are output with -DER flag
c       set ipar(8)  = 1 if dc/dr are output with -DER flag
c-----
c       for TI we have
c-----
c       set ipar(1)   = 1 if medium is spherical
c       set ipar(2)   = 1 if source is in fluid
c       set ipar(3)   = 1 if receiver is in fluid
c       set ipar(4)   = 1 if eigenfunctions are output with -DER flag
c       set ipar(5)   = 1 if dc/dh are output with -DER flag
c       set ipar(6)   = 1 if dc/dav are output with -DER flag
c       set ipar(7)   = 1 if dc/dbv are output with -DER flag
c       set ipar(8)   = 1 if dc/dr are output with -DER flag
c       set ipar(9)   = 1 if dc/dah are output with -DER flag
c       set ipar(10)  = 1 if dc/dn are output with -DER flag
c       set ipar(11)  = 1 if dc/dbh are output with -DER flag
c-----


        subroutine gtshed(lun,ifunc,kmode,t0,ierr)
        implicit none
        integer lun, ifunc, kmode,ierr
        real*8 t0
            ierr = 0
            read(lun,end=200,err=1001) ifunc,kmode,t0
            return
  200       continue
                ierr = 200
                return
 1001       continue
                ierr = 1001
        return
        end

        subroutine ptshed(lun,ifunc,kmode,t0)
        implicit none
        integer lun, ifunc, kmode
        real*8 t0
            write(lun) ifunc,kmode,t0
        return
        end

        subroutine gtsval(lun,cp,kmode,ierr)
        implicit none
        integer lun, kmode,ierr
        integer MAXMOD
        parameter(MAXMOD=2000)
        real*8 cp(MAXMOD)
        integer i
            read(lun,end=2001,err=1001) (cp(i),i=1,kmode)
            return
 2001       continue
                ierr = 200
                return
 1001       continue
                ierr = 1001
            return
        end

        subroutine ptsval(lun,cp,kmode)
        implicit none
        integer lun, kmode
        integer MAXMOD
        parameter(MAXMOD=2000)
        real*8 cp(MAXMOD)
        integer i
            write(lun) (cp(i),i=1,kmode)
            return
        end

c-----
c       common routines for eigenfunction files
c-----


        subroutine gethed(lun,ifunc,kmode,t0,ierr)
            ierr = 0
            read(lun,end=200,err=1001) ifunc,kmode,t0
            return
  200       continue
                ierr = 200
                return
 1001       continue
                ierr = 1001
        return
        end

        subroutine puthed(lun,ifunc,kmode,t0)
            real*4 t0
            write(lun) ifunc,kmode,t0
        return
        end
c-----
c       routines for depth dependent eigenfunctions
c       note these will NEVER be used with SLAT96
c-----

        subroutine getegn(lun,lorr,intext,wvno,u,gamma,
     1      sur,sdur,suz,sduz,sare,wvnsrc,sur0,
     2      rur,rtr,ruz,rtz,rare,wvnrec,rur0,
     3      sumkr,sumgr,sumgv,ierr)
c-----
c       lun I*4 - Logical unit number
c       lorr    I*4 - 1 = Love, 2 = Rayleigh
c       intext  I*4 - 1 write (lr)eigen85 files
c               - 0 write internal files for slat2d96
c       wvno    R*4 - wavenumber for path
c       u   R*4 - group velocity for path
c       gamma   R*4 - anelastic atenuation coefficient for path
c
c       sur R*4 - UR for Rayleigh or UT for Love at source depth
c       sdur    R*4 - d UR/dz for Rayleigh or d UT/dz 
c                   for Love at source depth
c       suz R*4 - UZ for Rayleigh at source depth
c       sdu R*4 - d UZ/dz for Rayleigh at source depth
c       sare    R*4 - energy integral at source
c       wvnsrc  R*4 - wavenumber at source
c       sur0    R*4 - free surface ellipticity ur/uz at source
c
c       wvnrec  R*4 - wavenumber at receiver
c       rur R*4 - UR for Rayleigh or UT for Love at receiver depth
c       rtr R*4 - Tr for Rayleigh or Tt
c                   for Love at receiver depth
c       ruz R*4 - UZ for Rayleigh at receiver depth
c       rtz R*4 - Tz for Rayleigh at receiver depth
c       rare    R*4 - energy integral at receiver
c       rur0    R*4 - ellipticity at receiver
c
c       sumkr   R*4 - sum of kr for slat2d96
c       sumgv   R*4 - sum of r/u for slat2d96
c       sumgr   R*4 - sum of gamma*r for slat2d96
c
c       ierr    I*4 - 1001 EOR or ERR on read
c-----
c       tare    R*4 - place holder
c-----
        integer LER
        parameter (LER=0)
        ierr = 0
        if(intext .ne. 0)then
            sumkr = 0.0
            sumgr = 0.0
            sumgv = 0.0
            if(lorr.eq.1) then
                read(lun,end=1001,err=1001)wvno,u,gamma,
     1              sur,sdur,
     1              rur,rtr,
     1              sare
                wvnsrc = wvno
                wvnrec = wvno
                rare = sare
            else if(lorr.eq.5) then
                read(lun,end=1001,err=1001)wvno,u,gamma,
     1              sur,sdur,
     1              rur,rtr,
     1              sare
                wvnsrc = wvno
                wvnrec = wvno
                rare = sare
            else if(lorr.eq.3) then
                read(lun,end=1001,err=1001)wvno,u,gamma,
     1              sur,sdur,
     1              rur,rtr,
     1              wvnsrc,wvnrec,sare,rare
            else if(lorr.eq.2) then
                read(lun,end=1001,err=1001)wvno,u,gamma,
     1              sur,sdur,suz,sduz,
     1              rur,rtr,ruz,rtz,
     2              sur0, rur0,sare
                wvnsrc = wvno
                wvnrec = wvno
                rare = sare
            else if(lorr.eq.6) then
                read(lun,end=1001,err=1001)wvno,u,gamma,
     1              sur,sdur,suz,sduz,
     1              rur,rtr,ruz,rtz,
     2              sur0, rur0,sare
                wvnsrc = wvno
                wvnrec = wvno
                rare = sare
            else if(lorr.eq.4) then
                read(lun,end=1001,err=1001)wvno,u,gamma,
     1              sur,sdur,suz,sduz,
     1              rur,rtr,ruz,rtz,
     1              wvnsrc,wvnrec,sur0,rur0,sare,rare
            endif
        else
            if(lorr.eq.1) then
                read(lun,end=1001,err=1001)wvno,tare,u,gamma,
     1              sur,sdur,
     1              rur,rtr,
     1              sare,rare,
     1              wvnsrc,wvnrec,
     2              sumkr, sumgr,sumgv
            else if(lorr.eq.5) then
                read(lun,end=1001,err=1001)wvno,tare,u,gamma,
     1              sur,sdur,
     1              rur,rtr,
     1              sare,rare,
     1              wvnsrc,wvnrec,
     2              sumkr, sumgr,sumgv
            else if(lorr.eq.2) then
                read(lun,end=1001,err=1001)wvno,tare,u,gamma,
     1              sur,sdur,suz,sduz,
     1              rur,rtr,ruz,rtz,
     2              sur0, rur0,sare,rare,
     2              wvnsrc,wvnrec,
     2              sumkr, sumgr,sumgv
            else if(lorr.eq.6) then
                read(lun,end=1001,err=1001)wvno,tare,u,gamma,
     1              sur,sdur,suz,sduz,
     1              rur,rtr,ruz,rtz,
     2              sur0, rur0,sare,rare,
     2              wvnsrc,wvnrec,
     2              sumkr, sumgr,sumgv
            endif
        endif
        return
 1001   continue
            ierr = 1001
            return
        end


        subroutine putegn(lun,lorr,intext,wvno,u,gamma,
     1      sur,sdur,suz,sduz,sare,wvnsrc,sur0,
     2      rur,rtr,ruz,rtz,rare,wvnrec,rur0,
     3      sumkr,sumgr,sumgv)
c-----
c       lun I*4 - Logical unit number
c       lorr    I*4 - 1 = Love, 2 = Rayleigh
c       intext  I*4 - 1 write (lr)eigen85 files
c               - 0 write internal files for slat2d96
c       wvno    R*4 - wavenumber for path
c       u   R*4 - group velocity for path
c       gamma   R*4 - anelastic atenuation coefficient for path
c
c       sur R*4 - UR for Rayleigh or UT for Love at source depth
c       sdur    R*4 - d UR/dz for Rayleigh or d UT/dz 
c                   for Love at source depth
c       suz R*4 - UZ for Rayleigh at source depth
c       sduz    R*4 - d UZ/dz for Rayleigh at source depth
c       sare    R*4 - energy integral at source
c       wvnsrc  R*4 - wavenumber at source
c       sur0    R*4 - free surface ellipticity ur/uz at source
c
c       wvnrec  R*4 - wavenumber at receiver
c       rur R*4 - UR for Rayleigh or UT for Love at receiver depth
c       rtr R*4 - Tr for Rayleigh or Tt
c                   for Love at receiver depth
c       ruz R*4 - UZ for Rayleigh at receiver depth
c       rtz R*4 - Tz for Rayleigh at receiver depth
c       rare    R*4 - energy integral at receiver
c       rur0    R*4 - ellipticity at receiver
c
c       sumkr   R*4 - sum of kr for slat2d96
c       sumgv   R*4 - sum of r/u for slat2d96
c       sumgr   R*4 - sum of gamma*r for slat2d96
c-----
c       tare    R*4 - place holder
c-----
        if(intext .ne. 0)then
            if(lorr.eq.1) then
                write(lun)wvno,u,gamma,
     1              sur,sdur,rur,rtr,sare
            else if(lorr.eq.3) then
                write(lun)wvno,u,gamma,
     1              sur,sdur,rur,rtr
     1              ,wvnsrc,wvnrec,sare,rare
            else if(lorr.eq.2) then
                write(lun)wvno,u,gamma,
     1              sur,sdur,suz,sduz,
     2              rur,rtr,ruz,rtz,
     3              sur0, rur0,sare
            else if(lorr.eq.4) then
                write(lun)wvno,u,gamma,
     1              sur,sdur,suz,sduz,rur,rtr,ruz,rtz
     1              ,wvnsrc,wvnrec,sur0,rur0,sare,rare
            else if(lorr.eq.5) then
                write(lun)wvno,u,gamma,
     1              sur,sdur,rur,rtr,sare
            else if(lorr.eq.6) then
                write(lun)wvno,u,gamma,
     1              sur,sdur,suz,sduz,
     2              rur,rtr,ruz,rtz,
     3              sur0, rur0,sare
            endif
        else
            if(lorr.eq.1) then
                write(lun)wvno,tare,u,gamma,
     1              sur,sdur,
     1              rur,rtr,
     1              sare,rare,
     1              wvnsrc,wvnrec,
     2              sumkr, sumgr,sumgv
            else if(lorr.eq.5) then
                write(lun)wvno,tare,u,gamma,
     1              sur,sdur,
     1              rur,rtr,
     1              sare,rare,
     1              wvnsrc,wvnrec,
     2              sumkr, sumgr,sumgv
            else if(lorr.eq.2) then
                write(lun)wvno,tare,u,gamma,
     1              sur,sdur,suz,sduz,
     1              rur,rtr,ruz,rtz,
     2              sur0, rur0,sare,rare,
     2              wvnsrc,wvnrec,
     2              sumkr, sumgr,sumgv
            else if(lorr.eq.6) then
                write(lun)wvno,tare,u,gamma,
     1              sur,sdur,suz,sduz,
     1              rur,rtr,ruz,rtz,
     2              sur0, rur0,sare,rare,
     2              wvnsrc,wvnrec,
     2              sumkr, sumgr,sumgv
            endif
        endif
        return
        end


c-----
c       common routines for eigenfunction files for TI
c-----

        subroutine gtsmdt(lun,mmax,d,ta,tc,tf,
     1      tl,tn,rho,qa,qb,nper,
     1      mname,ipar,fpar)
        implicit none
        integer NL
        parameter (NL=200)
        integer lun, mmax, nper
        real*4 d(NL), ta(NL), tc(NL), tf(NL), tl(NL),
     1      tn(NL),rho(NL), qa(NL), qb(NL)
        character mname*80
        integer ipar(20)
        real*4 fpar(20)
        integer i

            read(lun)mname
            read(lun)ipar
            read(lun)fpar
            read(lun)mmax
            read(lun)(d(i),ta(i),tc(i),tf(i),tl(i),tn(i),rho(i),
     2          i=1,mmax)
            read(lun)nper
        return
        end

        subroutine ptsmdt(lun,mmax,d,ta,tc,tf,
     1      tl,tn,rho,qa,qb,nper,
     1      mname,ipar,fpar)
        implicit none
        integer NL
        parameter (NL=200)
        integer lun, mmax, nper
        real*4 d(NL), ta(NL), tc(NL), tf(NL), tl(NL),
     1      tn(NL),rho(NL), qa(NL), qb(NL)
        character mname*80
        integer ipar(20)
        real*4 fpar(20)
        integer i

            write(lun)mname
            write(lun)ipar
            write(lun)fpar
            write(lun)mmax
            write(lun)(d(i),ta(i),tc(i),tf(i),tl(i),tn(i),rho(i),
     1          i=1,mmax)
            write(lun)nper
        return
        end

        subroutine getmdt(lun,mmax,d,ta,tc,tf,
     1      tl,tn,rho,qa,qb,nper,dphs,dphr,
     1      mname,ipar,fpar)
        parameter (NL=200)
        real *4 d(NL), ta(NL), tc(NL), tf(NL), tl(NL),
     1      tn(NL),rho(NL), qa(NL), qb(NL)
        character mname*80
        integer ipar(20)
        real*4 fpar(20)
            read(lun)mname
            read(lun)ipar
            read(lun)fpar
            read(lun)mmax,(d(i),ta(i),tc(i),tf(i),tl(i),tn(i),rho(i),
     1          qa(i),qb(i),i=1,mmax)
            read(lun)nper,dphs,dphr
        return
        end

        subroutine putmdt(lun,mmax,d,ta,tc,tf,
     1      tl,tn,rho,qa,qb,nper,dphs,dphr,
     1      mname,ipar,fpar)
        parameter (NL=200)
        real *4 d(NL), ta(NL), tc(NL), tf(NL), tl(NL),
     1      tn(NL),rho(NL), qa(NL), qb(NL)
        character mname*80
        integer ipar(20)
        real*4 fpar(20)
            write(lun)mname
            write(lun)ipar
            write(lun)fpar
            write(lun)mmax,(d(i),ta(i),tc(i),tf(i),tl(i),tn(i),rho(i),
     1          qa(i),qb(i),i=1,mmax)
            write(lun)nper,dphs,dphr
        return
        end

c-----
c       routines for depth dependent eigenfunctions
c       note these will NEVER be used with SLAT96
c-----
        subroutine getdrt(lun,lorr,wvno,u,gamma,
     1      sur,sdur,sd2ur,suz,sduz,sd2uz,sare,wvnsrc,sur0,
     2      rur,rtr,ruz,rtz,rare,wvnrec,rur0,
     3      sumkr,sumgr,sumgv,ierr,mmax,dcdh,
     4      dcdav,dcdah, dcdbv, dcdbh, dcdn,dcdr,
     5      ur,tur,uz,tuz,ipar)
c-----
c       lun I*4 - Logical unit number
c       lorr    I*4 - 5 = Love, 6 = Rayleigh
c       wvno    R*4 - wavenumber for path
c       u   R*4 - group velocity for path
c       gamma   R*4 - anelastic atenuation coefficient for path
c
c       sur R*4 - UR for Rayleigh or UT for Love at source depth
c       sdur    R*4 - d UR/dz for Rayleigh or d UT/dz 
c                   for Love at source depth
c       suz R*4 - UZ for Rayleigh at source depth
c       sdu R*4 - d UZ/dz for Rayleigh at source depth
c       sare    R*4 - energy integral at source
c       wvnsrc  R*4 - wavenumber at source
c       sur0    R*4 - free surface ellipticity ur/uz at source
c
c       wvnrec  R*4 - wavenumber at receiver
c       rur R*4 - UR for Rayleigh or UT for Love at receiver depth
c       rtr R*4 - Tr for Rayleigh or Tt
c                   for Love at receiver depth
c       ruz R*4 - UZ for Rayleigh at receiver depth
c       rtz R*4 - Tz for Rayleigh at receiver depth
c       rare    R*4 - energy integral at receiver
c       rur0    R*4 - ellipticity at receiver
c
c       sumkr   R*4 - sum of kr for slat2d96
c       sumgv   R*4 - sum of r/u for slat2d96
c       sumgr   R*4 - sum of gamma*r for slat2d96
c
c       mmax    I*4 - number of layers in model
c
c       dcdh    R*4 - array of layer thickness partials
c       dcdav   R*4 - array of layer pv-velocity partials
c       dcdah   R*4 - array of layer ph-velocity partials
c       dcdbv   R*4 - array of layer sv-velocity partials
c       dcdbh   R*4 - array of layer sh-velocity partials
c       dcdn    R*4 - array of layer eta-velocity partials
c       dcdr    R*4 - array of layer density partials
c
c       ur  R*4 - array of radial eigenfunction
c       tur R*4 - array of radial stress eigenfunction
c       uz  R*4 - array of vertical eigenfunction
c       tuz R*4 - array of vertical stress eigenfunction
c
c       ierr    I*4 - 1001 EOR or ERR on read
c       ipar    I*4 - array of integer controls
c-----
        integer LER
        parameter (LER=0)
        parameter (NL=200)
        real*4 dcdh(NL), dcdav(NL), dcdah(NL)
        real*4 dcdbv(NL), dcdbh(NL), dcdn(NL), dcdr(NL)
        real*4 ur(NL), tur(NL), uz(NL), tuz(NL)
        integer*4 ipar(20)
        ierr = 0
c-----
c       initialize
c-----
            sumkr = 0.0
            sumgr = 0.0
            sumgv = 0.0
            do 185 i=1,mmax
                ur(i) = 0.0
                tur(i) = 0.0
                uz(i) = 0.0
                tuz(i) = 0.0
                dcdh(i) = 0.0
                dcdav(i) = 0.0
                dcdah(i) = 0.0
                dcdbv(i) = 0.0
                dcdbh(i) = 0.0
                dcdn (i) = 0.0
                dcdr(i) = 0.0
  185       continue
c-----
c       read in the data stream
c-----

            if(lorr.eq.5) then
                read(lun,end=1001,err=1001)wvno,u,gamma,
     1              sur,sdur,sd2ur,rur,rtr,sare
                rare = sare
                if(ipar(4).eq.1)then
                read(lun,end=1001,err=1001)(ur(i),i=1,mmax)
                read(lun,end=1001,err=1001)(tur(i),i=1,mmax)
                endif
                if(ipar(5).eq.1)then
                     read(lun,end=1001,err=1001)(dcdh(i),i=1,mmax)
                endif
                if(ipar(7).eq.1)then
                     read(lun,end=1001,err=1001)(dcdbv(i),i=1,mmax)
                endif
                if(ipar(8).eq.1)then
                     read(lun,end=1001,err=1001)(dcdr(i),i=1,mmax)
                endif
                if(ipar(11).eq.1)then
                     read(lun,end=1001,err=1001)(dcdbh(i),i=1,mmax)
                endif
            else if(lorr.eq.6) then
                read(lun,end=1001,err=1001)wvno,u,gamma,
     1              sur,sdur,sd2ur,suz,sduz,sd2uz,
     1              rur,rtr,ruz,rtz,
     2              sur0, rur0,sare
                rare = sare
                if(ipar(4).eq.1)then
                read(lun,end=1001,err=1001)(ur(i),i=1,mmax)
                read(lun,end=1001,err=1001)(tur(i),i=1,mmax)
                read(lun,end=1001,err=1001)(uz(i),i=1,mmax)
                read(lun,end=1001,err=1001)(tuz(i),i=1,mmax)
                endif
                if(ipar(5).eq.1)then
                     read(lun,end=1001,err=1001)(dcdh(i),i=1,mmax)
                endif
                if(ipar(6).eq.1)then
                     read(lun,end=1001,err=1001)(dcdav(i),i=1,mmax)
                endif
                if(ipar(7).eq.1)then
                     read(lun,end=1001,err=1001)(dcdbv(i),i=1,mmax)
                endif
                if(ipar(8).eq.1)then
                     read(lun,end=1001,err=1001)(dcdr(i),i=1,mmax)
                endif
                if(ipar(9).eq.1)then
                     read(lun,end=1001,err=1001)(dcdah(i),i=1,mmax)
                endif
                if(ipar(10).eq.1)then
                     read(lun,end=1001,err=1001)(dcdn (i),i=1,mmax)
                endif
                if(ipar(11).eq.1)then
                     read(lun,end=1001,err=1001)(dcdbh(i),i=1,mmax)
                endif
            endif
        return
 1001   continue
            ierr = 1001
            return
        end


        subroutine putdrt(lun,lorr,wvno,u,gamma,
     1      sur,sdur,sd2ur,suz,sduz,sd2uz,sare,wvnsrc,sur0,
     2      rur,rtr,ruz,rtz,rare,wvnrec,rur0,
     3      sumkr,sumgr,sumgv,mmax,dcdh,
     4      dcdav,dcdah, dcdbv, dcdbh, dcdn,dcdr,
     5      ur,tur,uz,tuz,ipar)
c-----
c       lun I*4 - Logical unit number
c       lorr    I*4 - 5 = Love, 6 = Rayleigh
c       wvno    R*4 - wavenumber for path
c       u   R*4 - group velocity for path
c       gamma   R*4 - anelastic atenuation coefficient for path
c
c       sur R*4 - UR for Rayleigh or UT for Love at source depth
c       sdur    R*4 - d UR/dz for Rayleigh or d UT/dz 
c                   for Love at source depth
c       suz R*4 - UZ for Rayleigh at source depth
c       sduz    R*4 - d UZ/dz for Rayleigh at source depth
c       sare    R*4 - energy integral at source
c       wvnsrc  R*4 - wavenumber at source
c       sur0    R*4 - free surface ellipticity ur/uz at source
c
c       wvnrec  R*4 - wavenumber at receiver
c       rur R*4 - UR for Rayleigh or UT for Love at receiver depth
c       rtr R*4 - Tr for Rayleigh or Tt
c                   for Love at receiver depth
c       ruz R*4 - UZ for Rayleigh at receiver depth
c       rtz R*4 - Tz for Rayleigh at receiver depth
c       rare    R*4 - energy integral at receiver
c       rur0    R*4 - ellipticity at receiver
c
c       sumkr   R*4 - sum of kr for slat2d96
c       sumgv   R*4 - sum of r/u for slat2d96
c       sumgr   R*4 - sum of gamma*r for slat2d96
c
c       mmax    I*4 - number of layers in model
c
c       dcdh    R*4 - array of layer thickness partials
c       dcdav   R*4 - array of layer pv-velocity partials
c       dcdah   R*4 - array of layer ph-velocity partials
c       dcdbv   R*4 - array of layer sv-velocity partials
c       dcdbh   R*4 - array of layer sh-velocity partials
c       dcdn    R*4 - array of layer eta-velocity partials
c       dcdr    R*4 - array of layer density partials
c
c       ur  R*4 - array of radial eigenfunction
c       tur R*4 - array of radial stress eigenfunction
c       uz  R*4 - array of vertical eigenfunction
c       tuz R*4 - array of vertical stress eigenfunction
c       ipar    I*4 - array of integer controls
c
c-----
        parameter (NL=200)
        real*4 dcdh(NL), dcdav(NL), dcdah(NL)
        real*4 dcdbv(NL), dcdbh(NL), dcdn(NL), dcdr(NL)
        real*4 ur(NL), tur(NL), uz(NL), tuz(NL)
        integer*4 ipar(20)
            if(lorr.eq.5) then
                write(lun)wvno,u,gamma,
     1              sur,sdur,sd2ur,rur,rtr,sare
                if(ipar(4).eq.1)then
                     write(lun)(ur(i),i=1,mmax)
                     write(lun)(tur(i),i=1,mmax)
                endif
                if(ipar(5).eq.1)then
                     write(lun)(dcdh(i),i=1,mmax)
                endif
                if(ipar(7).eq.1)then
                     write(lun)(dcdbv(i),i=1,mmax)
                endif
                if(ipar(8).eq.1)then
                     write(lun)(dcdr(i),i=1,mmax)
                endif
                if(ipar(11).eq.1)then
                     write(lun)(dcdbh(i),i=1,mmax)
                endif
            else if(lorr.eq.6) then
                write(lun)wvno,u,gamma,
     1              sur,sdur,sd2ur,suz,sduz,sd2uz,
     2              rur,rtr,ruz,rtz,
     3              sur0, rur0,sare
                if(ipar(4).eq.1)then
                write(lun)(ur(i),i=1,mmax)
                write(lun)(tur(i),i=1,mmax)
                write(lun)(uz(i),i=1,mmax)
                write(lun)(tuz(i),i=1,mmax)
                endif
                if(ipar(5).eq.1)then
                      write(lun)(dcdh(i),i=1,mmax)
                endif
                if(ipar(6).eq.1)then
                      write(lun)(dcdav(i),i=1,mmax)
                endif
                if(ipar(7).eq.1)then
                      write(lun)(dcdbv(i),i=1,mmax)
                endif
                if(ipar(8).eq.1)then
                      write(lun)(dcdr(i),i=1,mmax)
                endif
                if(ipar(9).eq.1)then
                      write(lun)(dcdah(i),i=1,mmax)
                endif
                if(ipar(10).eq.1)then
                      write(lun)(dcdn (i),i=1,mmax)
                endif
                if(ipar(11).eq.1)then
                      write(lun)(dcdbh(i),i=1,mmax)
                endif
            endif
        return
        end

CRBH        subroutine gtsmdl(lun,mmax,d,a,b,rho,qa,qb,nper,
CRBH     1      mname,ipar,fpar)
CRBH        parameter (NL=200)
CRBH        real *4 d(NL), a(NL), b(NL), rho(NL), qa(NL), qb(NL)
CRBH        character mname*80
CRBH        integer ipar(20)
CRBH        real*4 fpar(20)
CRBH            read(lun)mname
CRBH            read(lun)ipar
CRBH            read(lun)fpar
CRBH            read(lun)mmax,(d(i),a(i),b(i),rho(i),
CRBH     2          i=1,mmax)
CRBHCTMP     1          qa(i),qb(i),
CRBH            read(lun)nper
CRBH        return
CRBH        end
CRBH
CRBH        subroutine ptsmdl(lun,mmax,d,a,b,rho,qa,qb,nper,
CRBH     1      mname,ipar,fpar)
CRBH        parameter (NL=200)
CRBH        real *4 d(NL), a(NL), b(NL), rho(NL), qa(NL), qb(NL)
CRBH        character mname*80
CRBH        integer ipar(20)
CRBH        real*4 fpar(20)
CRBH            write(lun)mname
CRBH            write(lun)ipar
CRBH            write(lun)fpar
CRBH            write(lun)mmax,(d(i),a(i),b(i),rho(i),
CRBH     1          i=1,mmax)
CRBHCTMP     1          qa(i),qb(i),
CRBH            write(lun)nper
CRBH        return
CRBH        end
CRBH
CRBH        subroutine getmdl(lun,mmax,d,a,b,rho,qa,qb,nper,dphs,dphr,
CRBH     1      mname,ipar,fpar)
CRBH        parameter (NL=200)
CRBH        real *4 d(NL), a(NL), b(NL), rho(NL), qa(NL), qb(NL)
CRBH        character mname*80
CRBH        integer ipar(20)
CRBH        real*4 fpar(20)
CRBH            read(lun)mname
CRBH            read(lun)ipar
CRBH            read(lun)fpar
CRBH            read(lun)mmax,(d(i),a(i),b(i),rho(i),
CRBH     1          qa(i),qb(i),i=1,mmax)
CRBH            read(lun)nper,dphs,dphr
CRBH        return
CRBH        end
CRBH
CRBH        subroutine putmdl(lun,mmax,d,a,b,rho,qa,qb,nper,dphs,dphr,
CRBH     1      mname,ipar,fpar)
CRBH        parameter (NL=200)
CRBH        real *4 d(NL), a(NL), b(NL), rho(NL), qa(NL), qb(NL)
CRBH        character mname*80
CRBH        integer ipar(20)
CRBH        real*4 fpar(20)
CRBH            write(lun)mname
CRBH            write(lun)ipar
CRBH            write(lun)fpar
CRBH            write(lun)mmax,(d(i),a(i),b(i),rho(i),
CRBH     1          qa(i),qb(i),i=1,mmax)
CRBH            write(lun)nper,dphs,dphr
CRBH        return
CRBH        end
CRBH        subroutine getder(lun,lorr,wvno,u,gamma,
CRBH     1      sur,sdur,sd2ur,suz,sduz,sd2uz,sare,wvnsrc,sur0,
CRBH     2      rur,rtr,ruz,rtz,rare,wvnrec,rur0,
CRBH     3      sumkr,sumgr,sumgv,ierr,mmax,dcdh,dcda,dcdb,dcdr,
CRBH     4      ur,tur,uz,tuz,ipar)
CRBHc-----
CRBHc       lun I*4 - Logical unit number
CRBHc       lorr    I*4 - 5 = Love, 6 = Rayleigh
CRBHc       wvno    R*4 - wavenumber for path
CRBHc       u   R*4 - group velocity for path
CRBHc       gamma   R*4 - anelastic atenuation coefficient for path
CRBHc
CRBHc       sur R*4 - UR for Rayleigh or UT for Love at source depth
CRBHc       sdur    R*4 - d UR/dz for Rayleigh or d UT/dz 
CRBHc                   for Love at source depth
CRBHc       suz R*4 - UZ for Rayleigh at source depth
CRBHc       sdu R*4 - d UZ/dz for Rayleigh at source depth
CRBHc       sare    R*4 - energy integral at source
CRBHc       wvnsrc  R*4 - wavenumber at source
CRBHc       sur0    R*4 - free surface ellipticity ur/uz at source
CRBHc
CRBHc       wvnrec  R*4 - wavenumber at receiver
CRBHc       rur R*4 - UR for Rayleigh or UT for Love at receiver depth
CRBHc       rtr R*4 - Tr for Rayleigh or Tt
CRBHc                   for Love at receiver depth
CRBHc       ruz R*4 - UZ for Rayleigh at receiver depth
CRBHc       rtz R*4 - Tz for Rayleigh at receiver depth
CRBHc       rare    R*4 - energy integral at receiver
CRBHc       rur0    R*4 - ellipticity at receiver
CRBHc
CRBHc       sumkr   R*4 - sum of kr for slat2d96
CRBHc       sumgv   R*4 - sum of r/u for slat2d96
CRBHc       sumgr   R*4 - sum of gamma*r for slat2d96
CRBHc
CRBHc       mmax    I*4 - number of layers in model
CRBHc
CRBHc       dcdh    R*4 - array of layer thickness partials
CRBHc       dcda    R*4 - array of layer p-velocity partials
CRBHc       dcdb    R*4 - array of layer s-velocity partials
CRBHc       dcdr    R*4 - array of layer density partials
CRBHc
CRBHc       ur  R*4 - array of radial eigenfunction
CRBHc       tur R*4 - array of radial stress eigenfunction
CRBHc       uz  R*4 - array of vertical eigenfunction
CRBHc       tuz R*4 - array of vertical stress eigenfunction
CRBHc
CRBHc       ierr    I*4 - 1001 EOR or ERR on read
CRBHc       ipar    I*4 - array of integer controls
CRBHc-----
CRBH        integer LER
CRBH        parameter (LER=0)
CRBH        parameter (NL=200)
CRBH        real*4 dcdh(NL), dcda(NL), dcdb(NL), dcdr(NL)
CRBH        real*4 ur(NL), tur(NL), uz(NL), tuz(NL)
CRBH        integer*4 ipar(20)
CRBH        ierr = 0
CRBHc-----
CRBHc       initialize
CRBHc-----
CRBH            sumkr = 0.0
CRBH            sumgr = 0.0
CRBH            sumgv = 0.0
CRBH            do 185 i=1,mmax
CRBH                ur(i) = 0.0
CRBH                tur(i) = 0.0
CRBH                uz(i) = 0.0
CRBH                tuz(i) = 0.0
CRBH                dcdh(i) = 0.0
CRBH                dcda(i) = 0.0
CRBH                dcdb(i) = 0.0
CRBH                dcdr(i) = 0.0
CRBH  185       continue
CRBHc-----
CRBHc       read in the data stream
CRBHc-----
CRBH
CRBH            if(lorr.eq.5) then
CRBH                read(lun,end=1001,err=1001)wvno,u,gamma,
CRBH     1              sur,sdur,sd2ur,rur,rtr,sare
CRBH            rare = sare
CRBH                if(ipar(4).eq.1)then
CRBH                read(lun,end=1001,err=1001)(ur(i),i=1,mmax)
CRBH                read(lun,end=1001,err=1001)(tur(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(5).eq.1)then
CRBH                read(lun,end=1001,err=1001)(dcdh(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(7).eq.1)then
CRBH                read(lun,end=1001,err=1001)(dcdb(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(8).eq.1)then
CRBH                read(lun,end=1001,err=1001)(dcdr(i),i=1,mmax)
CRBH                endif
CRBH            else if(lorr.eq.6) then
CRBH                read(lun,end=1001,err=1001)wvno,u,gamma,
CRBH     1              sur,sdur,sd2ur,suz,sduz,sd2uz,
CRBH     1              rur,rtr,ruz,rtz,
CRBH     2              sur0, rur0,sare
CRBH            rare = sare
CRBH                if(ipar(4).eq.1)then
CRBH                read(lun,end=1001,err=1001)(ur(i),i=1,mmax)
CRBH                read(lun,end=1001,err=1001)(tur(i),i=1,mmax)
CRBH                read(lun,end=1001,err=1001)(uz(i),i=1,mmax)
CRBH                read(lun,end=1001,err=1001)(tuz(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(5).eq.1)then
CRBH                read(lun,end=1001,err=1001)(dcdh(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(6).eq.1)then
CRBH                read(lun,end=1001,err=1001)(dcda(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(7).eq.1)then
CRBH                read(lun,end=1001,err=1001)(dcdb(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(8).eq.1)then
CRBH                read(lun,end=1001,err=1001)(dcdr(i),i=1,mmax)
CRBH                endif
CRBH            endif
CRBH        return
CRBH 1001   continue
CRBH            ierr = 1001
CRBH            return
CRBH        end
CRBH
CRBH
CRBH        subroutine putder(lun,lorr,wvno,u,gamma,
CRBH     1      sur,sdur,sd2ur,suz,sduz,sd2uz,sare,wvnsrc,sur0,
CRBH     2      rur,rtr,ruz,rtz,rare,wvnrec,rur0,
CRBH     3      sumkr,sumgr,sumgv,mmax,dcdh,dcda,dcdb,dcdr,
CRBH     4      ur,tur,uz,tuz,ipar)
CRBHc-----
CRBHc       lun I*4 - Logical unit number
CRBHc       lorr    I*4 - 5 = Love, 6 = Rayleigh
CRBHc       wvno    R*4 - wavenumber for path
CRBHc       u   R*4 - group velocity for path
CRBHc       gamma   R*4 - anelastic atenuation coefficient for path
CRBHc
CRBHc       sur R*4 - UR for Rayleigh or UT for Love at source depth
CRBHc       sdur    R*4 - d UR/dz for Rayleigh or d UT/dz 
CRBHc                   for Love at source depth
CRBHc       suz R*4 - UZ for Rayleigh at source depth
CRBHc       sduz    R*4 - d UZ/dz for Rayleigh at source depth
CRBHc       sare    R*4 - energy integral at source
CRBHc       wvnsrc  R*4 - wavenumber at source
CRBHc       sur0    R*4 - free surface ellipticity ur/uz at source
CRBHc
CRBHc       wvnrec  R*4 - wavenumber at receiver
CRBHc       rur R*4 - UR for Rayleigh or UT for Love at receiver depth
CRBHc       rtr R*4 - Tr for Rayleigh or Tt
CRBHc                   for Love at receiver depth
CRBHc       ruz R*4 - UZ for Rayleigh at receiver depth
CRBHc       rtz R*4 - Tz for Rayleigh at receiver depth
CRBHc       rare    R*4 - energy integral at receiver
CRBHc       rur0    R*4 - ellipticity at receiver
CRBHc
CRBHc       sumkr   R*4 - sum of kr for slat2d96
CRBHc       sumgv   R*4 - sum of r/u for slat2d96
CRBHc       sumgr   R*4 - sum of gamma*r for slat2d96
CRBHc
CRBHc       mmax    I*4 - number of layers in model
CRBHc
CRBHc       dcdh    R*4 - array of layer thickness partials
CRBHc       dcda    R*4 - array of layer p-velocity partials
CRBHc       dcdb    R*4 - array of layer s-velocity partials
CRBHc       dcdr    R*4 - array of layer density partials
CRBHc
CRBHc       ur  R*4 - array of radial eigenfunction
CRBHc       tur R*4 - array of radial stress eigenfunction
CRBHc       uz  R*4 - array of vertical eigenfunction
CRBHc       tuz R*4 - array of vertical stress eigenfunction
CRBHc       ipar    I*4 - array of integer controls
CRBHc
CRBHc-----
CRBH        parameter (NL=200)
CRBH        real*4 dcdh(NL), dcda(NL), dcdb(NL), dcdr(NL)
CRBH        real*4 ur(NL), tur(NL), uz(NL), tuz(NL)
CRBH        integer*4 ipar(20)
CRBH            if(lorr.eq.5) then
CRBH                write(lun)wvno,u,gamma,
CRBH     1              sur,sdur,sd2ur,rur,rtr,sare
CRBH                if(ipar(4).eq.1)then
CRBH                write(lun)(ur(i),i=1,mmax)
CRBH                write(lun)(tur(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(5).eq.1)then
CRBH                write(lun)(dcdh(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(7).eq.1)then
CRBH                write(lun)(dcdb(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(8).eq.1)then
CRBH                write(lun)(dcdr(i),i=1,mmax)
CRBH                endif
CRBH            else if(lorr.eq.6) then
CRBH                write(lun)wvno,u,gamma,
CRBH     1              sur,sdur,sd2ur,suz,sduz,sd2uz,
CRBH     2              rur,rtr,ruz,rtz,
CRBH     3              sur0, rur0,sare
CRBH                if(ipar(4).eq.1)then
CRBH                write(lun)(ur(i),i=1,mmax)
CRBH                write(lun)(tur(i),i=1,mmax)
CRBH                write(lun)(uz(i),i=1,mmax)
CRBH                write(lun)(tuz(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(5).eq.1)then
CRBH                write(lun)(dcdh(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(6).eq.1)then
CRBH                write(lun)(dcda(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(7).eq.1)then
CRBH                write(lun)(dcdb(i),i=1,mmax)
CRBH                endif
CRBH                if(ipar(8).eq.1)then
CRBH                write(lun)(dcdr(i),i=1,mmax)
CRBH                endif
CRBH            endif
CRBH        return
CRBH        end
