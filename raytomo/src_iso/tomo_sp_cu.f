C--------------------MAIN TOMOGRAPHY-------------------------
C Author: M.P. Barmin  Date: 03/02/01
C Tomography program: tomo_sp_cu_s_2 (v1.4.1)
C ADDED rejection via formulas: delta < n*lambda +
C geographical or geocentrical setting + strict type definition
C--------------------MAIN TOMOGRAPHY-------------------------
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
*      external hand
*      integer iiii
*      open(unit=0,file='exceptions')
*      iiii=ieee_handler('set','division',hand)
*      if(iiii.ne.0) print *,'ieee_bad'
*      iiii=0
C-------------------INITIATION------------------------------------S
      CALL INIT
      if(INDEX.NE.0) goto 98   
C-------------------INITIATION------------------------------------E
C---------------------INPUT DATA READING AND CHECKING-------------S
      CALL READER
      if(INDEX.NE.0) goto 98   
      if(lrejd) then
      CALL REJEDEL
      N=NOUT
      endif
      if(lreje)then
      CALL REJECTOR
      N=NOUT
      endif
C--------------------INPUT MODEL READING AND CHECKING-------------E       
C------------PRELIMINARY PROCEDURES-------------------------------S
      CALL MODEL
C--      CALL EQPL
      if(INDEX.NE.0) goto 98
C------------PRELIMINARY PROCEDURES-------------------------------E
C-----------------INVERSION PROCEDURES-----------------S
      CALL SWTSPH
      if(KOD.GT.0.AND.KOD.LT.32) goto 20
      if(lsymb) then
      print*,'to convert velocity in % of deviation from average value'
      CALL PERC_MAP(outfile,NXY)
      endif
      close(8)
      goto 99                            
C-----------------INVERSION PROCEDURES-----------------E
   20 print *,'RUN IS INTERRUPTED; KOD=',KOD 
      STOP
   99 print *,'COMPUTATIONS FINISHED'
      STOP
   98 print *,'RUN IS INTERRUPTED; INDEX=',INDEX
      STOP
      end               
