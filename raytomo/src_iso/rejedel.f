       SUBROUTINE REJEDEL
       IMPLICIT NONE
C----------------------------------------------------------------
C Rejection Data with path length defining by inequality:
C Delta < nwavep*Lambda, where
C                  Lambda ~ c*T,
C                  c      - phase/group velocities,
C                  T      - period,
C                  Lambda - wave length,
C                  Delta  - path length,
C                  nwavep - number of wave periods
C------N number of paths for processing---------------
      include "tomo.h"
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    x1(3),x2(3),xn(3),delt
      real*4    pe_re(21),c_re(21),am_re(21)
      real*4    delta,dum,dum1,phv_re
      integer*4 i,j
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      data pe_re/5.,6.,7.,8.,9.,10.,20.,40.,60.,80.,100.,125.,150.,175.,200.,225.,
     +250.,275.,300.,350.,400./
      data c_re/2.83,2.9,2.95,2.99,3.02,3.05,3.42,3.875,4.00,4.06,4.115,4.19,4.29,4.40,4.53,
     +4.69,4.89,5.05,5.24,5.585,5.88/
      dpi=4.0d0*DATAN(1.0d0)
      drad=dpi/180.d0
C------- Rejection ----------------------------------------------
      CALL SPLINE(21,pe_re,c_re,am_re,2,0.,2,0.)
      CALL SPLDER(1,21,pe_re,c_re,am_re,Tper,phv_re,dum,dum1,*993)
c**      write (*,*) Tper,phv_re
C------- Rejection ----------------------------------------------
      j=0
      do i=1,n
        reject=phv_re*Tper*FLOAT(nwavep)
        CALL TO_CART(TE0(i),FI0(i),TE(i),FI(i),x1,x2,xn,delt)
c        write(*,*) "check",TE0(i),FI0(i),TE(i),FI(i),x1,x2,xn,delt
c        stop
        delta=delt*DBLE(R)
        if(delta.gt.reject .and. delta.lt.500.) then
          j=j+1
          TE0(j)=TE0(i)
          FI0(j)=FI0(i)
          TE(j)=TE(i)
          FI(j)=FI(i)
          T(j)=T(i)
          WEIGHT(j)=WEIGHT(i)
        else
          write(8,*) ' REJEDEL: Path ',IIII(i), ' is rejected: Delta=',
     +    delta,'; Threshold=  ',reject
c          write(*,*) ' REJEDEL: Path ',IIII(i), ' is rejected: Delta=',
c     +    delta,'; Threshold=  ',reject
        endif
      enddo
C------- End of Rejection ---------------------------------------
      NOUT=j   
      write(8,*) ' REJEDEL: There were', n, ' paths; only ',nout,
     +' paths left after cleaning'
c      write(*,*) ' REJEDEL: There were', n, ' paths; only ',nout,
c     +' paths left after cleaning'
      return
 993  write(*,*)' REJEDEL: WRONG PERIOD=',Tper,'; STOP'
      stop
      end                                                       
