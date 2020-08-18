c----------------------------------------------------------------------c
c                                                                      c
c      COMPUTER PROGRAMS IN SEISMOLOGY                                 c
c      VOLUME III                                                      c
c                                                                      c
c      PROGRAM: TLEGN96                                                c
c                                                                      c
c      COPYRIGHT 1996, 2010                                            c
c      R. B. Herrmann                                                  c
c      Department of Earth and Atmospheric Sciences                    c
c      Saint Louis University                                          c
c      221 North Grand Boulevard                                       c
c      St. Louis, Missouri 63103                                       c
c      U. S. A.                                                        c
c                                                                      c
c----------------------------------------------------------------------c
c
c
c     This program calculates the group velocity and partial
c     derivatives of Love waves for any plane multi-layered
c     model.  The propagator-matrix, instead of numerical-
c     integration method is used, in which the Haskell rather
c     than Harkrider formalisms are concerned.
c
c     Developed by C. Y. Wang and R. B. Herrmann, St. Louis
c     University, Oct. 10, 1981.  Modified for use in surface
c     wave inversion, with addition of spherical earth flattening
c     transformation and numerical calculation of group velocity
c     partial derivatives by David R. Russell, St. Louis
c     University, Jan. 1984.
c
c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c Revision history:
c       07 AUG 2002 - make string lengths 120 characters from 80 
c       13 OCT 2006 - verbose output of energy integrals
c       26 SEP 2008 - fixed undefined LOT in subroutine up
c       14 JUN 2009 - reformulate in general terms as per book
c       27 JUL 2010 - convert to TI
c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        subroutine tlegn96(hs_in,hr_in,
     1      ohr_in,ohs_in, refdep_in, dogam_in, nl_in, iflsph_in,
     1      d_in,TA_in,TC_in,TF_in,TL_in,TN_in,TRho_in,
     1      qai_in,qbi_in,etapi_in,etasi_in,frefpi_in,frefsi_in,
     1      Nt_in, t_in,cp_in,
     1      u_out,ut_out, tut_out, 
     1      dcdh_out,dcdav_out,dcdah_out,
     1      dcdbv_out,dcdbh_out,dcdn_out,dcdr_out)
     
        implicit none
        integer LER, LIN, LOT
        parameter(LER=0,LIN=5,LOT=6)
        integer NL
        parameter(NL=200)
c-----
c       LIN - unit for FORTRAN read from terminal
c       LOT - unit for FORTRAN write to terminal
c       LER - unit for FORTRAN error output to terminal
c       NL  - number of layers in model
c-----
        common/timodel/d(NL),TA(NL),TC(NL),TL(NL),TN(NL),TF(NL),
     1      TRho(NL),
     2      qai(NL),qbi(NL),etapi(NL),etasi(NL),
     3      frefpi(NL), frefsi(NL)
        real d,TA,TC,TN,TL,TF,TRho,qai,qbi,etapi,etasi,frefpi,frefsi
        common/depref/refdep
        real refdep

        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax

        common/eigfun/ uu(NL),tt(NL),dcdh(NL),dcdr(NL),uu0(4),
     1     dcdbv(NL), dcdbh(NL)
        real*8 uu, tt, dcdh, dcdr, uu0, dcdbv, dcdbh
        real*4 sdcdah(NL), sdcdav(NL), sdcdbh(NL), sdcdbv(NL)
        real*4 sdcdn(NL), sdcdh(NL), sdcdr(NL) 
        real*4 spur(NL), sptr(NL), spuz(NL), sptz(NL)
        common/wateri/iwat(NL)
        real iwat
        common/sumi/   sumi0,sumi1,sumi2,flagr,ale,ugr
        real*8 sumi0, sumi1, sumi2, flagr, ale, ugr

        real*4 vtp,dtp,rtp
        common/sphere/vtp(NL),dtp(NL),rtp(NL)
        real*4 s1
        integer MAXMOD
        parameter(MAXMOD=2000)
        real*8 cp(MAXMOD), t
        real*8 c, omega, wvno, gammal, csph, usph
        real*8 Eut, Edut, Eut0, Ett0, Ed2ut
        logical nwlyrs, nwlyrr
        character*12 fname(2)
c-----
c       wired in file names     - output of this program 
c                           is always in the
c               binary file tlegn96.egn or tlegn96.der
c                               - input from tdisp96 is always
c               binary file tdisp96.lov
c-----
        logical dolove, dorayl
        character hsfile*120, hrfile*120, title*120 
        logical dotmp
        logical ext
        logical dogam
        logical dderiv

        character mname*120
        integer ipar(20)
        real*4 fpar(20)
        integer nipar(20)
        real dephr, dephs, deplw, depup, ohr, ohs, dt, faclov, facray
        real depthr, depths
        integer ifunc,mode,ierr
        real hr, hs
        integer iunit, iiso, iflsph, idimen, icnvel
        integer i,j,k,l,ifirst, nper, nmodes,n1,n2
        integer lgstr
        integer lsso,lss,lrro,lrr
        integer npts, nsph, mmaxot
        real rtz, rur0, rut, ruz, sare, sd2ut, sdut, sduz, sum, sumgr
        real sumgv, sumkr, sur0
        real rare, rtt, sd2uz, sut, suz, twopi, wvnrec, wvnsrc
      
        logical verbose
        
c   Input arrays, added by LF
        real*4 hs_in, hr_in, ohr_in, ohs_in, refdep_in
        integer nl_in, iflsph_in
        real*4 d_in(nl_in),TA_in(nl_in),TC_in(nl_in),TF_in(nl_in)
        real*4 TL_in(nl_in),TN_in(nl_in),TRho_in(nl_in)
        real*4 qai_in(nl_in),qbi_in(nl_in),etapi_in(nl_in)
        real*4 etasi_in(nl_in),frefpi_in(nl_in),frefsi_in(nl_in)
        integer*4 mode_in(Nt_in)
        real*4 t_in(Nt_in),cp_in(Nt_in)
        integer Nt_in, idisp
        logical dogam_in
c   Output arrays
        real*8 u_out(Nt_in)
        real*8 ut_out(2049,200), tut_out(2049,200)
        real*8 dcdh_out(2049,200),dcdav_out(2049,200)
        real*8 dcdah_out(2049,200),dcdn_out(2049,200)
        real*8 dcdbv_out(2049,200),dcdbh_out(2049,200)
        real*8 dcdr_out(2049,200)

c-----
c       machine dependent initialization
c-----
        call mchdep()
c-----
c       parse command line information, COMMENTED!
c-----  
cc        call gcmdln(hsfile,hrfile,hs,hr,dotmp,dogam,dderiv,
cc     1      nipar,verbose)
     
        verbose = .false.
        dogam   = dogam_in
        dderiv  = .true.
        nipar(4) = 1
        nipar(5) = 1
        nipar(6) = 1
        nipar(7) = 1
        nipar(8) = 1
        nipar(9) = 1
        nipar(10) = 1
        nipar(11) = 1
c-----
c       dotmp invoked if we use tcomb96.f, DELETED!
c-----

c-----
c       get control parameters from tdisp96.dat, DELETED!
c-----
        ohs = ohs_in
        ohr = ohr_in
        refdep  = refdep_in
c-----
c       get the earth model, DELETED !
c-----
     
        twopi=6.283185307179586d+00
        iflsph = iflsph_in
        mmax = nl_in
        idisp=1
C   assign model arrays, by LF
        do 339 i=1,mmax
            d(i) = d_in(i)
            TA(i) = TA_in(i)
            TC(i) = TC_in(i)
            TF(i) = TF_in(i)
            TN(i) = TN_in(i)
            TL(i) = TL_in(i)
            Trho(i) = TRho_in(i)
            qai(i) = qai_in(i)
            qbi(i) = qbi_in(i)
            etapi(i) = etapi_in(i)
            etasi(i) = etasi_in(i)
            frefpi(i) = frefpi_in(i)
            frefsi(i) = frefsi_in(i)
            if(TN(i).le.1.0e-4*TA(i))then
                iwat(i) = 1
            else
                iwat(i) = 0
            endif
 339   continue
        
c-----
c       get the Q information into the program
c       since this is not carried through in the tdisp96 output
c-----
        do 1234 i=1,mmax
            if(.not.dogam)then
                qai(i) = 0.0
                qbi(i) = 0.0
            endif
            if(qai(i).gt.1.0)qai(i)=1.0/qai(i)
            if(qbi(i).gt.1.0)qbi(i)=1.0/qbi(i)
            zqai(i) = qai(i)
            zqbi(i) = qbi(i)
            if(frefpi(i).le.0.0)frefpi(i) = 1.0
            if(frefsi(i).le.0.0)frefsi(i) = 1.0
            zfrefp(i) = frefpi(i)
            zfrefs(i) = frefsi(i)
            zetap(i) = etapi(i)
            zetas(i) = etasi(i)
 1234   continue
        nsph = iflsph
c-----
c       get source depth, and sphericity control
c-----
        if(hs.lt.-1.0E+20)then
            depths = ohs
        else
            depths = hs
        endif
        if(hr.lt.-1.0E+20)then
            depthr = ohr
        else
            depthr = hr
        endif
        depthr = depthr + refdep
        depths = depths + refdep
c-----
c       if there is a spherical model, map the source 
c            and receiver depth 
c       in the spherical model to the equivalent depth in the flat model
c-----
        if(nsph.gt.0)then
                dephs = 6371.0 *alog( 6371.0/(6371.0 - (depths-refdep)))
                dephr = 6371.0 *alog( 6371.0/(6371.0 - (depthr-refdep)))
        else
                dephs = depths
                dephr = depthr
        endif
c----
c       see if the file tdisp96.lov exists, DELETED !
c-----

c-----
c       open output file of tdisp96 and of tlegn96, DELETED !
c-----

c-----
c       obtain the earth model: note if the original was spherical,
c       this will be the transformed model, DELETED !
c-----

c-----
c       define modulus of rigidity, also get current transformed model
c       parameters
c       set fpar(1) = refdep
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
        deplw = 0.0
        depup = 0.0
        ipar(2) = 0
        ipar(3) = 0
        do 185 i=1,mmax
            zd(i)   = d(i)
            zta(i)  = ta(i)
            ztc(i)  = tc(i)
            ztl(i)  = tl(i)
            ztn(i)  = tn(i)
            ztf(i)  = tf(i)
            zrho(i) = trho(i)
            depup = deplw + d(i)
            if(tn(i) .lt. 0.0001*ta(i))then
                iwat(i) = 1
                if(depths.ge.deplw .and. depths.lt.depup)then
                    ipar(2) = 1
                endif
                if(depthr.ge.deplw .and. depthr.lt.depup)then
                    ipar(3) = 1
                endif
            else
                iwat(i) = 0
            endif
            deplw = depup
  185   continue
        do 186 i=4,11
            ipar(i) = nipar(i)
  186   continue
c    putmdt, DELETED !

c-----
c               If a spherical model is used, reconstruct the thickness
c               of the original model to get the mapping from
c               spherical to flat. We need this for correct
c               spherical partial derivatives.
c
c               THIS MUST BE DONE BEFORE SOURCE LAYER INSERTION
c-----
        if(nsph.gt.0)then
                call bldsph()
        endif
c-----
c       split a layer at the source depth
c       the integer ls gives the interface at which the eigenfunctions
c       are to be defined
c-----
        call insert(dephs,nwlyrs,lsso)
        call insert(dephr,nwlyrr,lrro)
        call srclyr(dephs,lss)
        call srclyr(dephr,lrr)
c-----DEBUG
c       output the new model with source layer
c-----
C       write(6,*)'lss,lrr:',lss,lrr
C        do  i=1,mmax
C                WrItE(6,*)i,iwat(i),zd(i),zta(i),ztc(i),
C     1    ztf(i),ztl(i),ztn(i),zrho(i)
C        enddo
c-----
c       EnD DEBUG
c-----
        twopi=2.*3.141592654
        ifirst = 0
  400   continue
c       gtshed, COMMENTED !
c        call gtshed(1,ifunc,mode,t,ierr)
        mode  = 1
        t     = t_in(idisp)
        if (idisp.gt.Nt_in)then
            ierr=200
        else
            ierr=0
        endif
        idisp = idisp+1
        if(ierr.ne.0)go to 700
        s1=t
c-----
c       DEBUG OUTPUT
c-----
c        write(6,*) ifunc,mode,s1
c-----
c       END DEBUG OUTPUT
c-----
c       puthed, COMMENTED !
c        call puthed(2,ifunc,mode,s1)
        omega=twopi/t
        if(ifunc.lt.0) go to 700
        if(mode.le.0) go to 400

        do 600 k=1,mode
                c=cp_in(idisp-1)
c-----
c       main part.
c-----
                wvno=omega/c
                call shfunc(omega,wvno)
            call energy(omega,wvno,Eut,Edut,Ed2ut,Eut0,Ett0,
     1          lss,lrr)
c-----
c       the gamma routine will use the spherical model, but the
c       frequency dependence and Q of the original model
c-----
C        WRITE(6,*)'c(noq)=',c
            if(dogam)then
                call gammaq(omega,wvno,gammal)
                c = omega/wvno
C        WRITE(6,*)'c(q)=',c
            else
                gammal = 0.0d+00
            endif
c-----
c       output necessary eigenfunction values for
c       source excitation
c-----
            mmaxot = mmax
                if(nsph.gt.0)then
c-----
c               sphericity correction for partial derivatives
c               of the original model
c-----
C        WRITE(6,*)'c(flat)=',c,' u(flat)=',ugr
                 call splove(omega,c,mmaxot,csph,usph,ugr)
                 wvno = omega/csph
                 u_out(idisp-1) = usph 
                else
                    u_out(idisp-1) = ugr 
                endif
c        WRITE(6,*)'c(flat)=',c,' u(flat)=',ugr
c        u_out(idisp-1) = ugr 
c-----
c           check for underflow
c-----
            if(dabs(Eut).lt. 1.0d-36)Eut = 0.0d+00 
            if(dabs(Edut).lt. 1.0d-36)Edut = 0.0d+00 
            if(dabs(Ed2ut).lt. 1.0d-36)Ed2ut = 0.0d+00 
            if(dabs(Eut0).lt. 1.0d-36)Eut0 = 0.0d+00 
            if(dabs(Ett0).lt. 1.0d-36)Ett0 = 0.0d+00 
            if(dabs(ugr).lt.1.0d-36)ugr=0.0d+00
            if(dabs(sumi0).lt.1.0d-36)sumi0=0.0d+00
            if(dabs(sumi1).lt.1.0d-36)sumi1=0.0d+00
            if(dabs(sumi2).lt.1.0d-36)sumi2=0.0d+00
            if(dabs(ale).lt.1.0d-36)ale=0.0d+00
            if(dabs(flagr).lt.1.0d-36)flagr=0.0d+00
            if(dabs(gammal).lt.1.0d-36)gammal=0.0d+00

c-----
c      get the derivatives of the eitenfunctions required
c      for source excitation from the definition of stress. For
c      completeness get the second derivative from the first
c      derivatives and the equation of motion for the medium
c-----
            sut = sngl(Eut)
            sdut = sngl(Edut)
            sd2ut = sngl(Ed2ut)
            suz = 0.0
            sduz = 0.0
            sd2uz = 0.0
            sare = sngl(ale)
            wvnsrc = sngl(wvno)
            sur0 = 0.0

            if( verbose ) then
                 WRITE(LOT,2)t,c,ugr,gammal,
     1                sumi0,sumi1,sumi2,
     2                flagr,ale
    2    format(' T=',e15.7,'  C=',e15.7,'  U=',e15.7,' G=',e15.7/
     1          'I0=',e15.7,' I1=',e15.7,' I2=',e15.7/
     2          ' L=',e15.7,' AL=',e15.7)
C                 WRITE(LOT,'(3e26.17/3e26.17/3e26.17)')t,c,ugr,gammal,
C     1                sumi0,sumi1,sumi2,
C     2                flagr,ale

            endif

            rut = sngl(Eut0)
            rtt = sngl(Ett0)
            ruz = 0.0
            rtz = 0.0
            rare = sngl(ale)
            wvnrec = sngl(wvno)
            rur0 = 0.0

            sumkr = 0.0
            sumgr = 0.0
            sumgv = 0.0
            if(nsph.gt.0)then
                wvno = omega/ csph
                ugr = usph
            endif
        if(dderiv)then
c-----
c               if a layer was inserted, get the partial
c               derivative for the original layer.
c           The sequence is cannot be changed.
c           Originally the model grows because the source layer
c           is added and then the receiver layer
c           This we must first strip the receiver and lastly the
c           source
c----
c       initialize
c-----
            if(nwlyrr)then
                call collap(lrro+1,mmaxot)
            endif
            if(nwlyrs)then
                call collap(lsso+1,mmaxot)
            endif
            call chksiz(uu,spur,mmaxot)
            call chksiz(tt,sptr,mmaxot)
            call chksiz(dcdh,sdcdh,mmaxot)
            call chksiz(dcdbv,sdcdbv,mmaxot)
            call chksiz(dcdbh,sdcdbh,mmaxot)
            call chksiz(dcdr,sdcdr,mmaxot)
c-----
c           up to this point the dcdh are changes to phase velocity if
c           if the layer boundary changes. Here we change this to mean
c           the dc/dh for a change in layer thickness
c
c           A layer becomes thicker if the base increases and the top
c           decreases its position. The dcdh to this point indicates 
c           the effect of moving a boundary down. Now we convert to
c           the effect of changing a layer thickness.
c-----
            do 505 i=1,mmaxot-1
                sum = 0.0
                do 506 j=i+1,mmaxot
                    sum = sum + sdcdh(j)
  506           continue
                sdcdh(i) = sum
  505       continue
            sdcdh(mmaxot) = 0.0
C            if(verbose)then
C              do i=1,mmaxot
C                WRITE(6,'(i5,6f10.6)')
C     1            I,uu(i),tt(i),sdcdh(i),sdcdbv(i),sdcdbh(i),sdcdr(i)
C              enddo
C            endif

c            call putdrt(2,5,sngl(wvno),sngl(ugr), 
c     1          sngl(gammal), 
c     1          sut,sdut,sd2ut,suz,sduz,sd2uz,sare,wvnsrc,sur0,
c     2          rut,rtt,ruz,rtz,rare,wvnrec,rur0,
c     3          sumkr,sumgr,sumgv,mmaxot,
c     4          sdcdh,sdcdav,sdcdah,sdcdbv,sdcdbh,sdcdn,sdcdr,
c     5          spur,sptr,spuz,sptz,ipar)

C       Important NOTE:
c       for Love wave, ut <- spur; tut <-sptr
        do 559 i=1,mmaxot
            ut_out(idisp-1, i)=spur(i)
            tut_out(idisp-1, i)=sptr(i)
            
            dcdh_out(idisp-1, i)=sdcdh(i)
            dcdav_out(idisp-1, i)=sdcdav(i)
            dcdah_out(idisp-1, i)=sdcdah(i)
            dcdbv_out(idisp-1, i)=sdcdbv(i)
            dcdbh_out(idisp-1, i)=sdcdbh(i)
            dcdn_out(idisp-1, i)=sdcdn(i)
            dcdr_out(idisp-1, i)=sdcdr(i)
  559       continue
        else
c            call putegn(2,1,1,sngl(wvno),sngl(ugr),
c     1          sngl(gammal),
c     1          sut,sdut,suz,sduz,sare,wvnsrc,sur0,
c     2          rut,rtt,ruz,rtz,rare,wvnrec,rur0,
c     3          sumkr,sumgr,sumgv)
        endif
c-----
c       DEBUG OUTPUT
c-----
c           write(6,*) wvno,c,ugr,ale
c           write(6,*) uu(ls),dut
cc-----
c       END DEBUG OUTPUT
c-----
  600   continue
        go to 400
  700   continue
c-----
c       close input file from tdisp96 and output file of this program
c-----
        do 900 i=1,2
                close(i,status='keep')
  900   continue

 9999   continue
        end

        subroutine shfunc(omega,wvno)
c-----
c       This routine evaluates the eigenfunctions by calling sub up.
c-----
        implicit double precision (a-h,o-z)
        integer NL
        parameter(NL=200)
        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax
        common/wateri/iwat(NL)
        common/eigfun/ uu(NL),tt(NL),dcdh(NL),dcdr(NL),uu0(4),
     1     dcdbv(NL), dcdbh(NL)
        real*8 uu, tt, dcdh, dcdr, uu0, dcdbv, dcdbh
        common/save/   exl(NL)
c-----
c       get the eigenfunctions in an extended floating point form
c       by propagating from the bottom upward. Note since we move in this
c       direction the propagator matrix A(d) is evaluated as A(-d)
c-----
        call up(omega,wvno,fl)
        uu0(1)=1.0
c-----
c       uu0(2)=stress0 is actually the value of period equation.
c       uu0(3) is used to print out the period equation value before
c       the root is refined.
c-----
        uu0(2)=fl
        uu0(3)=0.0
        uu0(4)=0.0
c-----
c       convert to actual displacement from the extended floating point
c       working top to down, where the amplitude should be small
c       Also normalize so that the V(0) = 1
c-----
        ext=0.0
        umax = uu(1)
        do 100 k=2,mmax
            if(iwat(k).eq.0)then
                ext=ext+exl(k-1)
                fact=0.0
                if(ext.lt.80.0) fact=1./dexp(ext)
                uu(k)=uu(k)*fact
                tt(k)=tt(k)*fact
            else
                uu(k) = 0.0
                tt(k) = 0.0
            endif
            if(abs(uu(k)).gt.umax)then
                umax = uu(k)
            endif
  100   continue
        if(uu(1).ne.0.0)then
                umax = uu(1)
        endif
        if(abs(umax).gt.0.0)then
            do 200 k=1,mmax
                if(iwat(k).eq.0)then
                    uu(k) = uu(k) / umax
                    tt(k) = tt(k) / umax
                endif
  200       continue
        endif
c-----
c       force boundary condition at the free surface
c       free surface is stress free
c-----
        return
        end

        subroutine up(omega,wvno,fl)
c-----
c       This routine calculates the elements of Haskell matrix,
c       and finds the eigenfunctions by analytic solution.
c-----
c       History:
c       
c       16 JUN 2009 - generalized the routine so that  the
c         nature of the propagator matrices is buried in the
c         subroutine hskl
c         This will permit simpler extension to TI
c-----
        implicit double precision (a-h,o-z)
        integer LER, LIN, LOT
        parameter(LER=0,LIN=5,LOT=6)
        integer NL
        parameter(NL=200)
        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax
        common/wateri/iwat(NL)
        common/eigfun/ uu(NL),tt(NL),dcdh(NL),dcdr(NL),uu0(4),
     1     dcdbv(NL), dcdbh(NL)
        real*8 uu, tt, dcdh, dcdr, uu0, dcdbv, dcdbh
        common/save/   exl(NL)

        real*8 rsh, omega, wvno, exl
        double precision cossh,rsinsh,sinshr,exm,exe
        complex*16 esh(2,2), einvsh(2,2)
        double precision hl(2,2)
        logical lshimag
        complex*16 e10, e20
        real*8 ttlast

c
c-----
c       apply the boundary conditions to the halfspace
c-----
c-----
c     kludge for fluid core
c-----
        if(ztn(mmax).gt.0.01)then
               call gtegsh(mmax,wvno, omega, rsh, lshimag)
               if(lshimag)then
                    write(LOT,*) ' imaginary nub derivl'
                 endif
               uu(mmax)=1.0
               tt(mmax)=-ztn(mmax)*rsh
        else
               uu(mmax)=1.0
               tt(mmax)=0.0
        endif
        exl(mmax) = 0.0
        mmx1=mmax-1
        ttlast = 0.0
        do 500 k=mmx1,1,-1
            if(iwat(k).eq.0)then
                call gtegsh(k,wvno, omega, rsh, lshimag)
                call gtesh(esh,einvsh,rsh,wvno,ZTL(k),lshimag,.false.,
     1             iwat(k))
                call varsh(zd(k),rsh,lshimag,cossh,rsinsh,sinshr,exm)
                exe = 0.0
                call hskl(cossh,rsinsh,sinshr,ZTL(k),
     1            iwat(k),hl,exm,exe)
                k1 = k + 1
c-----
c               We actually use A^-1 since we go from the bottom to top
c-----
C            WRITE(6,*)'exm:',exm
C            WRITE(6,*)'hl:',hl

                e10=  hl(1,1) * uu(k1) - hl(1,2)*tt(k1)
                e20= -hl(2,1) * uu(k1) + hl(2,2)*tt(k1)
                xnor=dabs(dreal(e10))
                ynor=dabs(dreal(e20))
                if(ynor.gt.xnor) xnor=ynor
                if(xnor.lt.1.d-40) xnor=1.0d+00
                exl(k) = dlog(xnor) + exe
                uu(k)=dreal(e10)/xnor
                tt(k)=dreal(e20)/xnor
                ttlast = tt(k)
            endif
  500   continue
        fl=ttlast
        return
        end

        subroutine energy(omega,wvno,Eut,Edut,Ed2ut,Eut0,Ett0,
     1      lss,lrr)
c-----
c       This routine calculates the values of integrals I0, I1,
c       and I2 using analytic solutions. It is found
c       that such a formulation is more efficient and practical.
c
c       This is based on a suggestion by David Harkrider in 1980.
c       Given the eigenfunctions, which are properly defined,
c       define the potential coefficients at the bottom and
c       top of each layer. If the V(z) = A exp (nu z) + B exp (-nu z)
c       We want the B at the top of the layer for z=0, and the A at the
c       bottom of the layer, from the relation K=[Kup Kdn]^T
c       = [A B]^T = E^-1 U
c-----
c       History:
c       
c       16 JUN 2009 - generalized the routine so that  the
c         nature of the propagator matrices is buried in the
c         This will permit simpler extension to TI
c-----
        implicit double precision (a-h,o-z)
        complex*16 nub
        integer NL
        parameter(NL=200)
        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax
        common/wateri/iwat(NL)
        common/eigfun/ uu(NL),tt(NL),dcdh(NL),dcdr(NL),uu0(4),
     1     dcdbv(NL), dcdbh(NL)
        real*8 uu, tt, dcdh, dcdr, uu0, dcdbv, dcdbh
        common/sumi/   sumi0,sumi1,sumi2,flagr,ale,ugr
        real*8 sumi0, sumi1, sumi2, flagr, ale, ugr

        real*8 rsh, omega, wvno, dpth
        logical lshimag

        complex*16 kmup, km1dn
        complex*16 esh(2,2), einvsh(2,2)
        complex*16 fb, gb
        complex*16 ffunc, gfunc
        complex*16 intij
        real*8 dtn,  drho
        real*8 tn, tl, rho
        real*8 upup, dupdup

        c=omega/wvno
        omega2=omega*omega
        wvno2=wvno*wvno
        sumi0=0.0d+00
        sumi1=0.0d+00
        sumi2=0.0d+00
c-----
c RBH beware when nub = 0
c-----
        do 300 k=1,mmax
            if(iwat(k).eq.0)then
                RHO = zrho(k)
                TN = ztn(k)
                TL = ztl(k)
                VSHH = dsqrt(ztn(k)/zrho(k))
                VSHV = dsqrt(ztl(k)/zrho(k))
                k1=k+1
                dpth=zd(k)
                call gtegsh(k,wvno, omega, rsh, lshimag)
c-----
                if(rsh .lt. 1.0d-10)rsh = 1.0d-10
c      for safety do not permit rsh = 0
c-----
                if(k.eq.mmax) then
                    upup  =(0.5d+00/rsh)*uu(mmax)*uu(mmax)
                    dupdup=(0.5d+00*rsh)*uu(mmax)*uu(mmax)
                else
c-----
c               use Einv to get the potential coefficients
c               from the U and T
c-----
                if(lshimag)then
                     nub=dcmplx(0.0d+00,rsh)
                else
                     nub=dcmplx(rsh,0.0d+00)
                endif
C                WRITE(6,*)'nub:',nub

                call gtesh(esh,einvsh,rsh,wvno,ZTL(k),lshimag,.true.,
     1             iwat(k))
                km1dn = einvsh(2,1)*uu(k)  + einvsh(2,2)*tt(k)
                kmup  = einvsh(1,1)*uu(k1) + einvsh(1,2)*tt(k1)

                fb = ffunc(nub,dpth)
CRBH                write(6,*)'F:',f
CRBH
CRBH                f3=nub*dpth
CRBH                exqq=0.0d+00
CRBH                if(dreal(f3).lt.40.0) exqq=cdexp(-2.d+00*f3)
CRBH                f=(1.d+00-exqq)/(2.d+00*nub)
CRBH                write(6,*)'f:',f
CRBHc-----
CRBHc               get the g function
CRBHc-----
CRBH                write(6,*)'G:',g
CRBH                exqq=0.0d+00
CRBH                if(dreal(f3).lt.75.0) exqq=cdexp(-f3)
CRBH                g=dpth*exqq
CRBH                write(6,*)'g:',g
                gb  = gfunc(nub,dpth)

                      
                 
                upup  =intij(1,1,fb,gb,km1dn,kmup,esh)
                dupdup=intij(2,2,fb,gb,km1dn,kmup,esh)/(ztl(k)*ztl(k))

CRBH                f1 = fb *(esh(1,1)*esh(1,1)*kmup*kmup 
CRBH     1              + esh(1,2)*esh(1,2)*km1dn*km1dn)
CRBH                f2 =gb *(esh(1,1)*esh(1,2)+esh(1,1)*esh(1,2))*kmup*km1dn
CRBHc-----
CRBHc               cast to a real  upup
CRBHc-----
CRBH                upup = f1 + f2
CRBH
CRBHc-----
CRBHc               cast to a real dupdup
CRBHc-----
CRBH                dupdup = nub*nub*(f1 - f2)
CRBHC              WRITE(6,*)'dupdup:',dupdup
CRBHC                f1 = f*(esh(2,1)*esh(2,1)*kmup*kmup 
CRBHC     1              + esh(2,2)*esh(2,2)*km1dn*km1dn)
CRBHC                f2 = g*(esh(2,1)*esh(2,2)+esh(2,1)*esh(2,2))*kmup*km1dn
CRBHC                dupdup = (f1 + f2)/ztl(k)*ztl(k))
C              WRITE(6,*)'dupdup:',dupdup
c-----
            endif

C            WrItE(*,*)k,upup,dupdup
            sumi0=sumi0+RHO*upup
            sumi1=sumi1+TN*upup
            sumi2=sumi2+TL*dupdup
            DCDBH(k) = c*RHO*VSHH*upup
            DCDBV(k) = c*RHO*VSHV*dupdup/wvno2
            DCDRSH= 0.5*c*(-c*c*upup + VSHH*VSHH*upup +
     1       VSHV*VSHV*dupdup/wvno2)
            dcdr(k)=DCDRSH
            else
                dcdbv(k)=0.0
                dcdbh(k)=0.0
                dcdr(k)=0.0
            endif
  300   continue
        do 400 k=1,mmax
            if(iwat(k).eq.0)then
                dcdbv(k)=dcdbv(k)/sumi1
                dcdbh(k)=dcdbh(k)/sumi1
                dcdr(k) =dcdr(k) /sumi1
            else
                dcdbv(k)=0.0
                dcdbh(k)=0.0
                dcdr(k)=0.0
            endif
  400   continue
        flagr=omega2*sumi0-wvno2*sumi1-sumi2
        ugr=sumi1/(c*sumi0)
        ale=0.5d+00/sumi1
c-----
c       define partial with respect to layer thickness
c-----
c       fac = 0.5d+00*c**3/(omega2*sumi1)
        fac = ale*c/wvno2
        c2 = c*c
        llflag = 0
        do 500 k=1,mmax
        if(iwat(k).eq.0)then
            if(llflag.eq.0)then
                drho = zrho(k)
                dtn  = ztn(k)
                dvdz = 0.0
            else 
                drho = zrho(k) - zrho(k-1)
                dtn  = ZTN(k) - ZTN(k-1)
                dvdz = tt(k)*tt(k)*(1.0/ZTL(k) - 1.0/ZTL(k-1))
            endif
            dfac = fac * ( uu(k)*uu(k)*
     1          (omega2*drho - wvno2*dtn) + dvdz)
            if(dabs(dfac).lt.1.0d-38)then
                dcdh(k) = 0.0
            else
                dcdh(k) = sngl(dfac)
            endif
            llflag = llflag + 1
        else
            dcdh(k) = 0.0
        endif
  500   continue
c-----
c       compute the eigenfuntions and depth derivatives
c       at the source depth
c-----
CCCCC
C       CHANGE THIS FOR THE ODE
CCCCC
        if(iwat(lss).eq.0)then
            Eut = uu(lss)
            Edut = tt(lss)/ztl(lss)
            Ed2ut =Eut*(wvno2*ztn(lss) - zrho(lss)*omega*omega)/ztl(lss)
        else
            Eut = 0.0d+00
            Edut = 0.0d+00
            Ed2ut = 0.0d+00
        endif
        if(iwat(lrr).eq.0)then
            Eut0 = uu(lrr)
            Ett0 = tt(lrr)
        else
            Eut0 = 0.0d+00
            Ett0 = 0.0d+00
        endif
        return
        end

        subroutine insert(dph,newlyr,ls)
        implicit none
        logical newlyr
        real*4 dph
        integer ls
        integer LER, LIN, LOT
        parameter (LER=0, LIN=5, LOT=6)

        integer NL
        parameter(NL=200)
        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax
        common/wateri/iwat(NL)
        integer iwat

        integer m, m1
        real*8 dep, dp, dphh, hsave

c-----
c       Insert a depth point into the model by splitting a layer
c       so that the point appears at the top boundary of the layer
c       dph = depth of interest
c       newlyr  - L .true. layer added
c               .false. no layer added to get source eigenfunction
c-----
c-----
c       determine the layer in which the depth dph lies.
c       if necessary, adjust  layer thickness at the base
c-----
c       Here determine layer corresponding to specific depth dph
c       If the bottom layer is not thick enough, extend it
c
c       dep     - depth to bottom of layer
c       dphh    - height of specific depth above bottom of the layer
c-----
        dep = 0.0
        dp = 0.0
        dphh=-1.0
        ls = 1
        do 100 m =1, mmax
                dp = dp + zd(m)
                dphh = dp - dph
                if(m.eq.mmax)then
                        if(zd(mmax).le.0.0d+00 .or. dphh.lt.0.0)then
                                zd(mmax) = (dph - dp)
                        endif
                endif
                dep = dep + zd(m)
                dphh = dep - dph
                ls = m
                if(dphh.ge.0.0) go to 101
  100   continue
  101   continue
c-----
c       In the current model, the depth point is in the ls layer
c       with a distance dphh to the bottom of the layer
c
c       Do not create unnecessary layers, e.g., 
c            at surface and internally
c       However do put in a zero thickness layer 
c            at the base if necessary
c-----
        if(dph .eq. 0.0)then
            newlyr = .false.
                return
        else if(dphh .eq. 0.0 .and. ls.ne.mmax)then
            ls = ls + 1
            newlyr = .false.
                return
        else
            newlyr = .true.
c-----
c               adjust layering
c-----
                 do 102 m = mmax,ls,-1
                       m1=m+1
                        zd(m1) = zd(m)
                        zta(m1) = zta(m)
                        ztc(m1) = ztc(m)
                        ztf(m1) = ztf(m)
                        ztl(m1) = ztl(m)
                        ztn(m1) = ztn(m)
                        zrho(m1) = zrho(m)
                        zqai(m1) = zqai(m)
                        zqbi(m1) = zqbi(m)
                        zfrefp(m1) = zfrefp(m)
                        zfrefs(m1) = zfrefs(m)
                        zetap(m1) = zetap(m)
                        zetas(m1) = zetas(m)
                        iwat(m1) = iwat(m)
  102           continue
                hsave=zd(ls)
                zd(ls) = hsave - dphh
                zd(ls+1) = dphh
                mmax = mmax + 1
        endif
        return
        end

        subroutine srclyr(depth,lmax)
        implicit none
        real depth
        integer lmax
        integer LER,  LIN, LOT
        parameter (LER=0, LIN=5, LOT=6)
        integer NL
        parameter(NL=200)
        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax
        common/wateri/iwat(NL)
        integer iwat

        real dep
        integer i
c-----
c       Find source/receiver boundary. It is assumed that
c       it will lie upon a boundary
c
c       lmax = source layer 
c       depth = source depth 
c-----
        dep = 0.0
        do 100 i=1,mmax
            if(abs(depth - dep).le.0.001*zd(i))then
                lmax = i
                return
            endif
            dep = dep + zd(i)
  100   continue
        return
        end 

        subroutine gammaq(omega,wvno,gammal)
c-----
c       This routine finds the attenuation gamma value.
c
c-----
        implicit none
        real*8 omega, wvno, gammal
        integer NL
        parameter (NL=200)
        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax
        common/wateri/iwat(NL)
        integer iwat
        common/eigfun/ ut(NL),tt(NL),dcdh(NL),dcdr(NL),uu0(4),
     1     dcdbv(NL), dcdbh(NL)
        real*8 ut, tt, dcdh, dcdr, uu0, dcdbv, dcdbh
        real*8 vshv, vshh, dc, pi, omgref, x, c
        integer i
        gammal=0.0
        dc = 0.0
        pi = 3.141592653589493d+00
        do 100 i=1,mmax
            if(iwat(i).eq.0)then
                vshv = dsqrt(ztl(i)/zrho(i))
                vshh = dsqrt(ztn(i)/zrho(i))
                x=dcdbh(i)*vshh*zqbi(i) + 
     1              dcdbv(i)*vshv*zqbi(i)
                gammal = gammal + x
                omgref=2.0*pi*zfrefs(i)
                dc = dc + dlog(omega/omgref)*x/pi
            endif
  100   continue
        c=omega/wvno
        gammal=0.5*wvno*gammal/c
        c=c+dc
        wvno=omega/c
        return
        end

        subroutine splove(om,c,mmax,csph,usph,ugr)
c-----
c       Transform spherical earth to flat earth
c       and relate the corresponding flat earth dispersion to spherical
c
c       Schwab, F. A., and L. Knopoff (1972). Fast surface wave 
c            and free
c       mode computations, in  
c            Methods in Computational Physics, Volume 11,
c       Seismology: Surface Waves and Earth Oscillations,  
c            B. A. Bolt (ed),
c       Academic Press, New York
c
c       Rayleigh Wave Equations 111, 114 p 144
c
c       Partial with respect to parameter uses the relation
c       For phase velocity, for example,
c
c       dcs/dps = dcs/dpf * dpf/dps, c is phase velocity, p is
c       parameter, and f is flat model
c
c       om      R*8     angular frequency
c       c       R*8     phase velocity
c       mmax    I*4     number of layers 
c-----
        implicit none
        real*8 om
        real*8 c, usph, csph, ugr
        integer mmax
        integer NL
        parameter(NL=200)
        common/eigfun/ ut(NL),tt(NL),dcdh(NL),dcdr(NL),uu0(4),
     1     dcdbv(NL), dcdbh(NL)
        real*8 ut, tt, dcdh, dcdr, uu0, dcdbv, dcdbh
        common/sphere/vtp(NL),dtp(NL),rtp(NL)
        real*4 vtp,dtp,rtp
        real*8 tm
        real a
        integer i

        a = 6371.0d0
        tm=sqrt(1.+(3.0*c/(2.*a*om))**2)
            do 20 i=1,mmax
                dcdbv(i)=dcdbv(i)*  vtp(i)/(tm**3)
                dcdbh(i)=dcdbh(i)*  vtp(i)/(tm**3)
C                dcdh(i) =dcdh(i) *  dtp(i)/(tm**3)
                dcdr(i) =dcdr(i) *  rtp(i)/(tm**3)
   20       continue
        csph = c / tm
        usph = ugr * tm
c       write(6,*)'c flat=',c,' csph=',c/tm
        return
        end

        subroutine bldsph()
c-----
c       Transform spherical earth to flat earth
c
c       Schwab, F. A., and L. Knopoff (1972). 
c            Fast surface wave and free
c       mode computations, 
c            in  Methods in Computational Physics, Volume 11,
c       Seismology: Surface Waves and Earth Oscillations,  
c            B. A. Bolt (ed),
c       Academic Press, New York
c
c       Love Wave Equations  44, 45 , 41 pp 112-113
c       Rayleigh Wave Equations 102, 108, 109 pp 142, 144
c
c     Revised 28 DEC 2007 to use mid-point, assume linear variation in
c     slowness instead of using average velocity for the layer
c     Use the Biswas (1972:PAGEOPH 96, 61-74, 1972) density mapping
c
c       This is for Love Waves
c
c-----
        implicit none
        integer NL
        parameter (NL=200)
        common/sphere/vtp(NL),dtp(NL),rtp(NL)
        real*4 vtp,dtp,rtp
        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax

        real*8 ar, dr, r0, r1, z0, z1
        integer i
        real TMP
c-----
c       vtp is the factor used to convert 
c            spherical velocity to flat velocity
c       rtp is the factor used to convert 
c            spherical density  to flat density 
c       dtp is the factor used to convert 
c            spherical boundary to flat boundary
c-----
c-----
c       duplicate computations of srwvds(IV)
c-----
        ar=6371.0d0
        dr=0.0d0
        r0=ar
        z0 = 0.0d+00
        zd(mmax)=1.0
        do 10 i=1,mmax
            r1 = r0 * dexp(-zd(i)/ar)
            if(i.lt.mmax)then
                z1 = z0 + dble(zd(i))
            else
                z1 = z0 + 1.0d+00
            endif
            TMP=(ar+ar)/(r0+r1)
             vtp(i) = TMP
             rtp(i) = TMP**(-5)
            dtp(i) = sngl(ar/r0)
            r0 = r1
            z0 = z1
   10   continue
c       write(6,*)'vtp:',(vtp(i),i=1,mmax)
c       write(6,*)'rtp:',(rtp(i),i=1,mmax)
c       write(6,*)'dtp:',(dtp(i),i=1,mmax)
c-----
c       at this point the model information is no longer used and
c       will be overwritten
c-----
        return
        end

        subroutine gcmdln(hsfile,hrfile,hs,hr,dotmp,dogam,dderiv,
     1      nipar,verbose)
        implicit none
c-----
c       parse the command line arguments
c-----
c       hsfile  C*120   - name of source depth file
c       hrfile  C*120   - name of receiver depth file
c       hs  R*4 source depth (single one specified
c       hr  R*4 receiver depth (single one specified
c       dotmp   L   - .true. use file tdisp96.lov
c       dogam   L   - .true. incorporate Q
c       dderiv  L   - .true. output depth dependent values
c       nipar   I*4 - array o integer controls
c           set nipar(4)   = 1 if eigenfunctions are output with -DER flag
c           set nipar(5)   = 1 if dc/dh are output with -DER flag
c           set nipar(6)   = 1 if dc/dav are output with -DER flag
c           set nipar(7)   = 1 if dc/dbv are output with -DER flag
c           set nipar(8)   = 1 if dc/dr are output with -DER flag
c           set nipar(9)   = 1 if dc/dah are output with -DER flag
c           set nipar(10)  = 1 if dc/dn are output with -DER flag
c           set nipar(11)  = 1 if dc/dbh are output with -DER flag

c       verbose L   - .true. output information on energy integrals
c-----
        character hrfile*120, hsfile*120
        real hs, hr
        logical dotmp, dogam, dderiv
        integer nipar(20)
        logical verbose


        character name*40
        integer mnmarg
        integer nmarg, i

        hrfile = ' '
        hsfile = ' '
        hs = -1.0E+21
        hr = -1.0E+21
        dotmp = .false.
        dogam = .true.
        dderiv = .false.
        nipar(4) = 0
        nipar(5) = 0
        nipar(6) = 0
        nipar(7) = 0
        nipar(8) = 0
        nipar(9) = 0
        nipar(10) = 0
        nipar(11) = 0
        verbose = .false.
        nmarg = mnmarg()
        i = 0
 1000   continue
            i = i + 1
            if(i.gt.nmarg)go to 2000
                call mgtarg(i,name)
                if(name(1:3).eq.'-HR')then
                    i = i + 1
                    call mgtarg(i,name)
                    read(name,'(bn,f20.0)')hr
                else if(name(1:3).eq.'-HS')then
                    i = i + 1
                    call mgtarg(i,name)
                    read(name,'(bn,f20.0)')hs
C               else if(name(1:4).eq.'-FHR')then
C                   i = i + 1
C                   call mgtarg(i,hrfile)
C               else if(name(1:4).eq.'-FHS')then
C                   i = i + 1
C                   call mgtarg(i,hsfile)
                else if(name(1:2) .eq. '-T')then
                    dotmp = .true.
                else if(name(1:4) .eq. '-NOQ')then
                    dogam = .false.
                else if(name(1:4) .eq. '-DER')then
                    nipar(4)  = 1
                    nipar(5)  = 1
                    nipar(6)  = 0
                    nipar(7)  = 1
                    nipar(8)  = 1
                    nipar(9)  = 0
                    nipar(10) = 0
                    nipar(11) = 1
                    dderiv = .true.
                else if(name(1:3).eq.'-DE')then
                    dderiv = .true.
                    nipar(4) = 1
                else if(name(1:3).eq.'-DH')then
                    dderiv = .true.
                    nipar(5) = 1
                else if(name(1:3).eq.'-DB')then
                    dderiv = .true.
                    nipar(7) = 1
                    nipar(11) = 1
                else if(name(1:3).eq.'-DR')then
                    dderiv = .true.
                    nipar(8) = 1
                else if(name(1:2).eq.'-V')then
                    verbose = .true.
                else if(name(1:2) .eq. '-?')then
                    call usage(' ')
                else if(name(1:2) .eq. '-h')then
                    call usage(' ')
                endif
        go to 1000
 2000   continue
        return
        end

        subroutine usage(ostr)
        implicit none
        character ostr*(*)
        integer LER
        parameter (LER=0)
        integer lostr
        integer lgstr
        if(ostr .ne. ' ')then
            lostr = lgstr(ostr)
            write(LER,*)ostr(1:lostr)
        endif
        write(LER,*)'Usage: tlegn96 ',
C    1      ' -FHR recdep -FHS srcdep -HS hs -HR hr ' ,
     1      ' -HS hs -HR hr ' ,
     2       ' [-NOQ] [-T] [-DER -DE -DH -DB -DR ] [-?] [-h]'
C       write(LER,*)
C     1 '-FHS srcdep (overrides -HS )  Name of source depth  file'
C       write(LER,*)
C     1 '-FHR recdep (overrides -HR )  Name of receiver depth  file'
        write(LER,*)
     1  '-HS hs      (default 0.0 )  Source depth '
        write(LER,*)
     1  '-HR hr      (default 0.0 )  Receiver depth'
        write(LER,*)
     1  '-NOQ        (default Q used) perfectly elastic'
        write(LER,*)
     1  '-T          (default false) use tdisp96.lov not disp96.lov'
        write(LER,*)
     1  '-DER        (default false) output depth dependent values'
        write(LER,*)
     1  '-DE         (default false) output eigenfunctions(depth)'
        write(LER,*)
     1  '-DH         (default false) output DC/DH(depth)'
        write(LER,*)
     1  '-DB         (default false) output DC/DB(depth)'
        write(LER,*)
     1  '-DR         (default false) output DC/DR(depth)'
        write(LER,*)
     1  '-V          (default false) list energy integrals'
        write(LER,*)
     1  '-?          (default none )  this help message '
        write(LER,*)
     1  '-h          (default none )  this help message '
        stop
        end

        subroutine collap(ls,mmaxot)
        implicit none
        integer ls, mmaxot
        integer NL
        parameter(NL=200)
        common/eigfun/ uu(NL),tt(NL),dcdh(NL),dcdr(NL),uu0(4),
     1     dcdbv(NL), dcdbh(NL)
        real*8 uu, tt, dcdh, dcdr, uu0, dcdbv, dcdbh
        integer i
        do 501 i = ls-1,mmaxot
            if(i .eq. ls -1)then
                dcdbv(i) = dcdbv(i) + dcdbv(i+1)
                dcdbh(i) = dcdbh(i) + dcdbh(i+1)
                dcdr(i) = dcdr(i) + dcdr(i+1)
            endif
            if(i.gt.ls)then
                dcdbv(i-1) = dcdbv(i)
                dcdbh(i-1) = dcdbh(i)
                dcdh(i-1) = dcdh(i)
                dcdr(i-1) = dcdr(i)
                uu(i-1) = uu(i)
                tt(i-1) = tt(i)
            endif
  501   continue
        mmaxot = mmaxot - 1
        return
        end

        subroutine chksiz(dp,sp,mmaxot)
c-----
c       correctly convert double precision to single precision
c-----
        implicit none
        integer mmaxot
        real*8 dp(mmaxot)
        real*4 sp(mmaxot)
        integer i
            do 610 i=1,mmaxot
                if(dabs(dp(i)).lt.1.0d-36)then
                    sp(i) = 0.0
                else
                    sp(i) = sngl(dp(i))
                endif
  610       continue
        return
        end

c-----
c       notes:
c       For the surface wave the propagator is always positive
c       However the E and E^-1 matrices can be complex
c-----

        subroutine varsh(h,rsh,lshimag,cossh,rsinsh,sinshr,ex)
        implicit none
        double precision h
        double precision rsh
        logical lshimag
        double precision cossh, rsinsh, sinshr
        double precision ex
        
        double precision q, fac, sinq
        q = rsh*h
        ex =  0.0
        if(lshimag)then
            if(rsh.gt.0.0)then
                 cossh = dcos(q)
                 sinq = sin(q)
                 sinshr = sinq/rsh
                 rsinsh = - rsh*sinq
            else
                 cossh  = 1.0d+00
                 sinshr = dble(h)
                 rsinsh = 0.0
            endif
        else
            ex = q
            if(q.lt.16.0d+00)then
                fac = dexp(-2.0d+00*q)
            else
                fac = 0.0d+00
            endif
            cossh = (1.0d+00 + fac) * 0.5d+00
            sinq   = (1.0d+00 - fac) * 0.5d+00
            sinshr = sinq/rsh
            rsinsh = sinq*rsh
        endif
        return
        end

        subroutine hskl(cossh,rsinsh,sinshr,TL,iwat,hl,exm,exe)
c-----
c       True cosh( rd b )    = exp(exm) cossh
c       True sinh( rd b )/rb = exp(exm) sinshr
c       True rb sinh( rd b ) = exp(exm) rsinsh
c-----
        implicit none
        double precision cossh, rsinsh, sinshr
        double precision hl(2,2),exm,exe
        double precision TL
        integer iwat

        if(iwat.eq.0)then   
            hl(1,1) = cossh
            hl(2,1) = TL*rsinsh
            hl(1,2) = sinshr/(TL)
            hl(2,2) = cossh
            exe = exe + exm
        else
            hl(1,1) = 1.0
            hl(1,2) = 0.0
            hl(2,1) = 0.0
            hl(2,2) = 1.0
        endif
        return
        end 

        subroutine gtesh(esh,einvsh,rsh,wvno,L,lshimag,ltrueinv,iwat)
        complex*16 esh(2,2), einvsh(2,2)
        double precision wvno, rsh
        double precision L
        logical lshimag,ltrueinv
        integer iwat
        double complex rb
c-----
c       get E and E^-1 for Quasi SH
c       if ltrueinv == .true. then return E^-1
c       else return the normalized avoid divide by zero
c       (2k L rsh) E^-1 to avoid dividing by zero when rsh = 0
c-----
        if(iwat.eq.1)then
c-----
c           for a water layer use an identity matrix. The
c           results have validity only for the extreme case of
c           a model that is fluid-solid-fluid, e.g., ocean,mantle,
c           outer core.  The propagator matrix technique cannot 
c           handle anything more complicated
c-----
             esh(1,1) = dcmplx(1.0d+00, 0.0d+00)
             esh(1,2) = dcmplx(0.0d+00, 0.0d+00)
             esh(2,1) = dcmplx(0.0d+00, 0.0d+00)
             esh(2,2) = dcmplx(1.0d+00, 0.0d+00)
             einvsh(1,1) = dcmplx(1.0d+00, 0.0d+00)
             einvsh(1,2) = dcmplx(0.0d+00, 0.0d+00)
             einvsh(2,1) = dcmplx(0.0d+00, 0.0d+00)
             einvsh(2,2) = dcmplx(1.0d+00, 0.0d+00)
        else
             if(lshimag)then
                 rb =  dcmplx(0.0d+00,rsh)
             else
                 rb = dcmplx(rsh,0.0d+00)
             endif
             esh(1,1) =   wvno
             esh(1,2) =   wvno
             esh(2,1) =   wvno * L * rb
             esh(2,2) = - wvno * L * rb
             einvsh(1,1) =   L * rb
             einvsh(2,1) =   L * rb
             einvsh(1,2) =   1.0
             einvsh(2,2) =  -1.0
             if(ltrueinv)then
                 einvsh(1,1) = einvsh(1,1)/(2.*wvno*L*rb)
                 einvsh(1,2) = einvsh(1,2)/(2.*wvno*L*rb)
                 einvsh(2,1) = einvsh(2,1)/(2.*wvno*L*rb)
                 einvsh(2,2) = einvsh(2,2)/(2.*wvno*L*rb)
             endif
        endif
        return
        end

        subroutine gtegsh(m,wvno,omega,rsh,lshimag)
        implicit none

        integer m
        double precision wvno, omega, rsh
        logical lshimag

        integer NL
        parameter (NL=200)
        common/timod/  zd(NL),zta(NL),ztc(NL),ztf(NL),
     1      ztl(NL),ztn(NL),zrho(NL),zqai(NL),zqbi(NL),
     2      zetap(NL),zetas(NL),zfrefp(NL),zfrefs(NL)
        real*8 zd, zta, ztc, ztf, ztl, ztn, zrho, 
     1      zqai, zqbi, zetap, zetas, 
     1      zfrefp, zfrefs
        common/pari/ mmax
        integer mmax
        common/water/iwat(NL)
        integer iwat
c-----
c       internal variables
c-----
        double precision wvno2, omega2
        wvno2 = wvno*wvno
        omega2 = omega*omega
        if(iwat(m).eq.1)then
c-----
c           check for water layer
c-----
            rsh = 0.0
            lshimag = .false.
        else
            rsh = ZTN(m)*wvno2/(ZTL(m)) - ZRho(m)*omega2/(ZTL(m))
            if(rsh .lt. 0.0)then
                rsh = dsqrt(dabs(rsh))
                lshimag = .true.
            else
                rsh = dsqrt(dabs(rsh))
                lshimag = .false.
            endif
        endif
        
        return
        end

        function ffunc(nub,dm)
        implicit none
        complex*16 ffunc
        complex*16 nub
        real*8 dm
        complex*16 exqq
        complex*16 argcd
c-----
c       get the f function
c-----
        if(cdabs(nub).lt. 1.0d-08)then
            ffunc = dm
        else
            argcd = nub*dm
            if(dreal(argcd).lt.40.0)then
                  exqq = cdexp(-2.0d+00*argcd)
            else
                  exqq=0.0d+00
            endif
            ffunc = (1.d+00-exqq)/(2.d+00*nub)
        endif
        return
        end

        function gfunc(nub,dm)
        implicit none
        complex*16 gfunc
        complex*16 nub
        real*8 dm
        complex*16 argcd
        argcd = nub*dm
        if(dreal(argcd).lt.75)then
             gfunc = cdexp(-argcd)*dm
        else
             gfunc = dcmplx(0.0d+00, 0.0d+00)
        endif
        return
        end

        function intij(i,j,fb,gb,km1dn,kmup,esh)
        implicit none
        complex*16 intij
        integer i,j
        complex*16 fb,gb,km1dn,kmup,esh(2,2)
            intij = esh(i,1)*esh(j,1)*kmup*kmup*fb
     1         + esh(i,2)*esh(j,2)*km1dn*km1dn*fb
     2         + (esh(i,1)*esh(j,2)+esh(i,2)*esh(j,1))*
     3              kmup*km1dn*gb
        return
        end
