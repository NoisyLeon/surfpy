        subroutine getmod(rlun,mname,mmax,title,iunit,iiso,iflsph,
     1      idimen,icnvel,ierr,listmd)
c-----
c       read isotropic or transverse isotropic model and save
c       as transverse isotropic model
c-----
c       HISTORY
c
c       09 08 2000  gave ierr an initial default value for g77
c       01 13 2001  put in close(lun) if file is not model file
c       04 19 2002  Introduced TI model file
c       03 MAY 2002     Modify to permit read from standard input
c       06 JUL 2005 moved inquire to permit use of STDIN
c
c-----
c       General purpose model input
c       This model specification is designed to be as 
c           general as possible
c
c       Input lines
c       Line 01: MODEL
c       Line 02: Model Name
c       Line 03: ISOTROPIC or ANISOTROPIC or 
c           TRANSVERSELY ANISOTROPIC
c       Line 04: Model Units, First character is 
c           length (k for kilometer
c           second is mass (g for gm/cc), third is time (s for time)
c       Line 05: FLAT EARTH or SPHERICAL EARTH
c       Line 06: 1-D, 2-D or 3-D
c       Line 07: CONSTANT VELOCITY
c       Line 08: open for future use
c       Line 09: open for future use
c       Line 10: open for future use
c       Line 11: open for future use
c       Lines 12-end:   These are specific to the model
c           For ISOTROPIC the entries are
c           Layer Thickness, P-velocity, S-velocity, Density, Qp, Qs,
c           Eta-P, Eta S (Eta is frequency dependence), 
c           FreqRefP, FreqRefP
c-----
cMODEL
cTEST MODEL.01
cISOTROPIC
cKGS
cFLAT EARTH
c1-D
cCONSTANT VELOCITY
cLINE08
cLINE09
cLINE10
cLINE11
c H  VP  VS   RHO   QP  QS   ETAP   ETAS REFP  REFS
c1.0    5.0 3.0 2.5 0.0 0.0 0.0 0.0 1.0 1.0
c2.0    5.1 3.1 2.6 0.0 0.0 0.0 0.0 1.0 1.0
c7.0    6.0 3.5 2.8 0.0 0.0 0.0 0.0 1.0 1.0
c10.0   6.5 3.8 2.9 0.0 0.0 0.0 0.0 1.0 1.0
c20.0   7.0 4.0 3.0 0.0 0.0 0.0 0.0 1.0 1.0
c40.0   8.0 4.7 3.3 0.0 0.0 0.0 0.0 1.0 1.0
c-----
c-----
c       rlun    I*4 - logical unit for reading model file. This
c                 unit is released after the use of this routine
c       mname   C*(*)   - model name
c       mmax    I*4 - number of layers in the model, last layer is
c                    halfspace
c       title   C*(*)   - title of the model file
c       iunit   I*4 - 0 Kilometer, Gram, Sec
c       iiso    I*4 - 0 isotropic 
c                 1 transversely anisotropic 
c                 2 general anisotropic 
c       iflsph  I*4 - 0 flat earth model
c                 1 spherical earth model
c       idimen  I*4 - 1 1-D
c               - 2 2-D
c               - 3 3-D
c       icnvel  I*4 - 0 constant velocity
c                 1 variable velocity
c       ierr    I*4 - 0 model file correctly read in
c               - -1 file does not exist
c               - -2 file is not a model file
c                 -3 error in the model file
c       listmd  L   - .true. list the model
c------

        implicit none
        character mname*(*), title*(*)
        integer rlun
        integer*4 mmax, iunit, iiso, iflsph, idimen, icnvel
        integer*4 ierr
        character string*80
        logical listmd
c-----
c       LIN I*4 - logical unit for standard input
c       LOT I*4 - logical unit for standard output
c-----
        integer LIN, LOT, LER
        parameter (LIN=5,LOT=6,LER=0)

        integer NL
        parameter (NL=200)
        common/timodel/d(NL),TA(NL),TC(NL),TL(NL),TN(NL),TF(NL),
     1      TRho(NL),
     2      qa(NL),qb(NL),etap(NL),etas(NL), 
     3      frefp(NL), frefs(NL)
        real d,TA,TC,TN,TL,TF,TRho,qa,qb,etap,etas,frefp,frefs
        common/depref/refdep
        real refdep

        real avel,bvel
        real vpv, vph, vsv, vsh, vpf

        logical ext

        character ftype*80
        integer lun, j, i, irefdp

c-----
c       test to see if the file exists
c-----
        ierr = 0
        if(MNAME(1:5).eq.'stdin' .or. mname(1:5).eq.'STDIN')then
c-----
c           do not open anything, use standard output
c-----
            lun = LIN
        else
            lun = rlun
            inquire(file=mname,exist=ext)
            if(.not.ext)then
                ierr = -1
                write(LER,*)'Model file does not exist'
                return
            endif
c-----
c           open the file
c-----
            open(lun,file=mname,status='old',form='formatted',
     1          access='sequential')
            rewind lun
        endif
c-----
c       verify the file type
c-----
c-----
c       LINE 01
c-----
        read(lun,'(a)')ftype
        if(ftype(1:5).ne.'model' .and. ftype(1:5).ne.'MODEL')then
            ierr = -2
            write(LER,*)'Model file is not in model format'
            close(lun)
            return
        endif
c-----
c       LINE 02
c-----
        read(lun,'(a)')title
c-----
c       LINE 03
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'ISO' .or. string(1:3).eq.'iso')then
            iiso = 0
        else if(string(1:3).eq.'TRA' .or. string(1:3).eq.'tra')then
            iiso = 1
        else if(string(1:3).eq.'ANI' .or. string(1:3).eq.'ani')then
            iiso = 2
        endif
c-----
c       LINE 04
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'KGS' .or. string(1:3).eq.'kgs')then
            iunit = 0
        endif
c-----
c       LINE 05
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'FLA' .or. string(1:3).eq.'fla')then
            iflsph = 0
        else if(string(1:3).eq.'SPH' .or. string(1:3).eq.'sph')then
            iflsph = 1
        endif
c-----
c       LINE 06
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'1-d' .or. string(1:3).eq.'1-D')then
            idimen = 1
        else if(string(1:3).eq.'2-d' .or. string(1:3).eq.'2-D')then
            idimen = 2
        else if(string(1:3).eq.'3-d' .or. string(1:3).eq.'3-D')then
            idimen = 3
        endif
c-----
c       LINE 07
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'CON' .or. string(1:3).eq.'con')then
            icnvel = 0
        else if(string(1:3).eq.'VAR' .or. string(1:3).eq.'var')then
            icnvel = 1
        endif
c-----
c       get lines 8 through 11
c-----
        do 900 i=8,11
            read(lun,'(a)')string
  900   continue
c-----
c       get model specifically for 1-D flat isotropic
c-----
c-----
c       get comment line
c-----
        if(iiso.eq.0)then
            read(lun,'(a)')string
        else
            read(lun,'(a)')string
            read(lun,'(a)')string
        endif
        mmax = 0
        refdep = 0.0
        irefdp = 0
        if(iiso.eq.0 .or. iiso.eq.1)then
 1000       continue
            j = mmax +1
            if(iiso.eq.0)then
                read(lun,*,err=9000,end=9000)d(j),avel,bvel,
     1              TRho(j),qa(j),qb(j),etap(j),etas(j),
     2              frefp(j),frefs(j)
                TC(j) = TRho(j)*avel*avel
                TA(j) = TRho(j)*avel*avel
                TL(j) = TRho(j)*bvel*bvel
                TN(j) = TRho(j)*bvel*bvel
                TF(j) = TA(j) - 2.0*TN(j)
            else if(iiso.eq.1)then
            read(lun,*,end=9000,err=9000)d(j),vpv,vsv,
     1          TRho(j),qa(j),qb(j),etap(j),etas(j),
     2          frefp(j),frefs(j)
            read(lun,*,end=9000,err=9000)vph,vsh,vpf
                TC(j) = TRho(j)*vpv*vpv
                TA(j) = TRho(j)*vph*vph
                TL(j) = TRho(j)*vsv*vsv
                TN(j) = TRho(j)*vsh*vsh
                TF(j) = TRho(j)*vpf*vpf
            endif
                if(d(j).lt.0.0)then
                    d(j) = -d(j)
                    refdep = refdep + d(j)
                    irefdp = j
                endif
            mmax = j
            go to 1000
 9000       continue
        endif
c-----
c       If listmd.eq. .true. list the model -- 
c           be careful here. If the model is actually
c       isotropic list the isotropic model, else list the TI model
c-----
   11   format(' LAYER              H     P-VEL     S-VEL   DENSITY  ')
   12   format(' ',i5,5x,4f10.3)
   13   format(' ','-SURFACE ','- - - - - ','- - - - - ',
     1      '- - - - - ','- - - - - -')
   21   format(' LAYER              H    PV-VEL    PH-VEL    PF-VEL   ',
     1      ' SV-VEL     SH-VEL   DENSITY  ')
   22   format(' ',i5,5x,7f10.3)
   23   format(' ','-SURFACE ','- - - - - ','- - - - - ','- - - - - -',
     1      '- - - - - ','- - - - - -','- - - - - -','- - - - - -')
        if(mmax.gt.0)then
            ierr = 0
        else
            ierr = -3
            write(LER,*)'Error in model file'
        endif
        if(listmd .and. mmax.gt. 0)then
c-----
c           list isotropic model
c-----
            if(iiso.eq.0)then
                write(LOT,11)
                do 2000 i=1,mmax
                    avel = sqrt(TC(i)/TRho(i))
                    bvel = sqrt(TL(i)/TRho(i))
                    write(LOT,12)
     1              i,d(i),avel,bvel,TRho(i)
                if(i.eq.irefdp)write(LOT,13)
 2000           continue
            else if(iiso.eq.1)then
                write(LOT,21)
                do 3000 i=1,mmax
                    vpv = sqrt(TC(i)/TRho(i))
                    vsv = sqrt(TL(i)/TRho(i))
                    vph = sqrt(TA(i)/TRho(i))
                    vsh = sqrt(TN(i)/TRho(i))
                    vpf = sqrt(TF(i)/TRho(i))
                    avel = sqrt(TC(i)/TRho(i))
                    bvel = sqrt(TL(i)/TRho(i))
                    write(LOT,22)
     1              i,d(i),vpv,vph,vpf,vsv,vsh,TRho(i)
                if(i.eq.irefdp)write(LOT,23)
 3000           continue
            endif
        endif
        if(lun.ne.LIN)close (lun)
        return
        end
