     
        subroutine flat2sphere(ilvry,dt,nn,iret,
     1          verby,nfval,fval,ccmin,ccmax,
     1      d_in,TA_in,TC_in,TF_in,TL_in,TN_in,TRho_in,
     1      nl_in,refdep_in,mode_in,facl_in,facr_in,
     1      d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out)
c-----
c       Compute and return spherical model transformed from flat model
c================================================================
c        by Lili Feng, for pysurf
c        Sep 20th, 2017
c================================================================
c
c-----
        implicit double precision (a-h,o-z)
        integer NL, NL2
        parameter (LIN=5, LOT=6, NL=200,NL2=NL+NL)
        integer*4 mmaxx,kmaxx
        integer*4 ifuncc(2)
        real*4 vts(NL2,2)
        real*4 dt, cmin, cmax
        real*4 ccmin,ccmax
        logical verby
        parameter(NPERIOD=2049)
        real fval(NPERIOD)

        parameter(MAXMOD=2000)
        common/phas/ cp(MAXMOD)
        common/pari/ mmax,mode
        common/water/iwat(NL)
        common/pard/ twopi,displ,dispr
        common/stor/ dcc(MAXMOD),c(MAXMOD),nlost,index,nroot1
        common/vels/ mvts(2),vts
C       Input model arrays, added by LF
        integer nl_in
        real*4 d_in(nl_in),TA_in(nl_in),TC_in(nl_in),TF_in(nl_in)
        real*4 TL_in(nl_in),TN_in(nl_in),TRho_in(nl_in)
        integer mode_in
        real facl_in, facr_in
c       Output, added by LF
        integer ic
        real*4 d_out(nl_in),TA_out(nl_in),TC_out(nl_in),TF_out(nl_in)
        real*4 TL_out(nl_in),TN_out(nl_in),TRho_out(nl_in)
        
        common/timod/d(NL),TA(NL),TC(NL),TF(NL),TL(NL),TN(NL),
     1      TRho(NL),
     2      qa(NL),qb(NL),etap(NL),etas(NL), 
     3      frefp(NL), frefs(NL)
        real*4 d,TA,TC,TN,TL,TF,TRho,qa,
     1       qb,etap,etas,frefp,frefs
        common/depref/refdep
        real refdep
        common/timodel/od(NL),oTA(NL),oTC(NL),oTL(NL),oTN(NL),oTF(NL),
     1      oTRho(NL),
     2      oqa(NL),oqb(NL),oetap(NL),oetas(NL), 
     3      ofrefp(NL), ofrefs(NL)
        real od,oTA,oTC,oTN,oTL,oTF,oTRho,oqa,oqb,
     1        oetap,oetas,ofrefp,ofrefs

        character mname*80
        integer ipar(20)
        real*4 fpar(20)
        character title*80
        integer iunit, idimen,iiso,iflsph,ierr
c-----
c       initialize
c-----
        data ipar/20*0/
        data fpar/20*0.0/
c
   11 format(1x,' improper initial value  no zero found. cmin=',f6.2)
   10 format(' ' ,5x,7f10.4)
   20 format(' ' ,15x,6f10.4)
   30 format(' ' ,16x,'FLAT CRUSTAL MODEL USED  ' )
   40 format(' ' ,7x,'   THICK     TA        TC        TF',
     1    '        TL        TN      DENSITY')
c-----
c       get the earth model
c       This may seem redundant, but if we apply earth flattening to
c       a spherical model, then Love and Rayleigh flattened models
c       are different
c-----
c-----
c       get the earth model
c-----

c        call getmod(1,mname,mmmax,title,iunit,iiso,iflsph,
c     1      idimen,icnvel,ierr,.false.)
c        mmax = mmmax
c        nsph = iflsph
c        ipar(1) = nsph
c        fpar(1) = refdep
c        fpar(2) = refdep

c   - model information, LF
c   - iflsph_in - 0: flat; 1: spherical
        if (mode_in.ne.1)then
        mode_in = 1
        write(6,*) 'WARNING: Currently only support fundamental mode!'
        endif
        radius = 6371.
        twopi=6.283185307179586d+00
        ic=1 
        mmax = nl_in
        nsph = 1
        ipar(1) = nsph
        fpar(1) = refdep_in
        fpar(2) = refdep_in
        refdep  = refdep_in
        mode    = mode_in
c factor for finer search, can be modifed as input parameters
        displ   = facl_in
        dispr   = facr_in
        
c-----
c       set fpar(1) = refdep in flat earth model
c   ?   set fpar(2) = refdep in original flat or spherical model
c   ?       fpar(1) may be different because of flattening
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
c        do 339 i=1,mmax
c            d(i) = od(i)
c            TA(i) = oTA(i)
c            TC(i) = oTC(i)
c            TF(i) = oTF(i)
c            TN(i) = oTN(i)
c            TL(i) = oTL(i)
c            Trho(i) = oTrho(i)
c            if(TN(i).le.1.0e-4*TA(i))then
c                iwat(i) = 1
c            else
c                iwat(i) = 0
c            endif

C   assign model arrays, LF
        do 339 i=1,mmax
            d(i) = d_in(i)
            TA(i) = TA_in(i)
            TC(i) = TC_in(i)
            TF(i) = TF_in(i)
            TN(i) = TN_in(i)
            TL(i) = TL_in(i)
            Trho(i) = TRho_in(i)
            if(TN(i).le.1.0e-4*TA(i))then
                iwat(i) = 1
            else
                iwat(i) = 0
            endif

c   - End of get model            

c        WRITE(6,*)d(i),TA(i),TC(i),TF(i),TN(i),TL(i),Trho(i),iwat(i)
  339   continue
c-----
c       d = thickness of layer in kilometers
c       a = compressional wave velocity in km/sec
c       b = transverse wave velocity in km/sec
c       rho = density in gm/cc
c-----

c-----
c       if the model is spherical, then perform a wave type dependent
c       earth flattening
c-----

        call sphere(ilvry)

        do 340 i=1,mmax
            d_out(i)=d(i) 
            TA_out(i)=TA(i) 
            TC_out(i)=TC(i)
            TF_out(i)=TF(i)
            TN_out(i)=TN(i)
            TL_out(i)=TL(i)
            TRho_out(i)=Trho(i)
  340       continue
        
        if (verby)then
            write(LOT,30)
            write(LOT,40)
            if(nsph.eq.0)then
                  WRITE(LOT,*)'Velocity model is flat'
            else
                  WRITE(LOT,*)'Velocity model is spherical. ',
     1                'The following is the flattened model'
            endif
            do 341 i=1,mmax-1
                write(LOT,10) d(i),TA(i),TC(i),
     1           TF(i),TL(i),TN(i),Trho(i)
  341       continue
            write(LOT,10) d(mmax),TA(mmax),TC(mmax),TF(mmax),TL(mmax),
     1         TN(mmax),TRho(mmax)
c            write(LOT,20) TA(mmax),TC(mmax),TF(mmax),TL(mmax),    (original)
c     1         TN(mmax),TRho(mmax)
        endif
        
        return
        end