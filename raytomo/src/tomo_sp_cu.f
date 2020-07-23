c********************MAIN TOMOGRAPHY********************************
c* Tomography program: tomo_sp_cu_s, v1.2.                         *
c* Author: M.P. Barmin, Date: 01/23/2003.                          *
c* CIEI, University of Colorado at Boulder                         *
c*                                                                 *
c* This program uses Fresnel zone approach or Gaussian beam.       *
c* There are main and auxilary grids in program. Main is used      *
c* for setting isotopic model paramrters and auxilary for          *
c* anisotropic part.                                               *
c* Fresnel zone integrals don't  depend on main or auxilary grids. *
c* Step of integration zone on a spere is an area ~ 1x1 degree.    *
c* For anisotopy part Gaussian ray approximation is used.          *
c* Additional features: rejection via formulas: delta < n*lambda,  *
c* geographical or geocentrical input data setting, etc.           *
c*******************************************************************
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
      integer i
*       external hand
*       integer mmmm
*       open(0,file='exceptions')
*       mmmm=ieee_handler('set','division',hand)
*       if(mmmm.ne.0) write(*,*) 'ieee_bad'
*       mmmm=0
c
c  Program initiation
c
      CALL INIT
      if(index.ne.0) goto 98   
c
c  Reading Input Data
c
      CALL READER
      if(index.ne.0) goto 98   
c
c  Data rejection by delta (optional)
c
      if(lrejd)then
        CALL REJEDEL
        n=nout
      endif
c
c  Data rejection by wavelength (optional)
c
      if(lreje)then
        CALL REJECTOR
        n=nout
      endif
c
c  Reading Input Model
c
      CALL MODEL
      if(index.ne.0) goto 98
c
c  Inversion Procedure
c
      CALL SWTSPH
      if(kod.gt.0.and.kod.lt.32) goto 20
c
c  Get average velocities and deviations (optional)
c
      if(isymb.ne.0) then
        write(*,*) 
     +        'convert velocity in % of deviation from average value'
        do i=1,ireg
          CALL PERC_MAP(outfile(i),nxy)
        enddo
      endif
      goto 99                            
c
c  Old staff - obsolete
c
   20 write(*,*) 'RUN IS INTERRUPTED; KOD=',kod 
      stop
   99 write(*,*) '(R0099) COMPUTATIONS FINISHED'
      stop
   98 write(*,*) 'RUN IS INTERRUPTED; INDEX=',index
      stop
      end               
