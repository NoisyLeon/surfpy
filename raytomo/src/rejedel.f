c***********************************************************
c Rejection Data with path length defining by inequality:
c Delta < nwavep*Lambda, where
c        Lambda - c*T,
c        c      - phase/group velocities,
c        T      - period,
c        Lambda - wave length,
c        Delta  - path length,
c        nwavep - number of wave periods
c        N      - number of paths for processing
c***********************************************************
      SUBROUTINE REJEDEL
      IMPLICIT NONE
      include "tomo.h"
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*8    x1(3),x2(3),xn(3),delt,period,slo
cxx   real*4    pe_re(16),c_re(16),am_re(20)
cxx   real*4    delta,dum,dum1,phv_re
      real*4    delta,phv_re
      integer*4 i,j,ierr
c END TYPE definitions ++++++++++++++++++++++++++++++++++
cxx   data pe_re/10.,20.,40.,60.,80.,100.,125.,150.,175.,200.,225.,
cxx  +250.,275.,300.,350.,400./
cxx   data c_re/3.05,3.42,3.875,4.00,4.06,4.115,4.19,4.29,4.40,4.53,
cxx  +4.69,4.89,5.05,5.24,5.585,5.88/
cxx   dpi=4.0d0*DATAN(1.0d0)
cxx   drad=dpi/180.d0
c
c---    Rejection
c
cxx   CALL SPLINE(16,pe_re,c_re,am_re,2,0.,2,0.)
cxx   CALL SPLDER(1,16,pe_re,c_re,am_re,Tper,phv_re,dum,dum1,*993)

      period=tper
      CALL GET_SLOW(re_la,period,slo,ierr)
      if(ierr.ne.0) stop '(R0015) 1-D model period is out of range'
      phv_re=1.0d0/slo

      j=0
      do i=1,n
        reject=phv_re*Tper*FLOAT(nwavep)
        CALL TO_CART(TE0(i),FI0(i),TE(i),FI(i),IRAY2(i),x1,x2,xn,delt)
        delta=delt*DBLE(R)
        if(delta.gt.reject) then
          j=j+1
          TE0(j)=TE0(i)
          FI0(j)=FI0(i)
          TE(j)=TE(i)
          FI(j)=FI(i)
          T(j)=T(i)
          WEIGHT(j)=WEIGHT(i)
          IRAY2(j)=IRAY2(i)
        else
          write(8,*) ' REJEDEL: Path ',IIII(i), ' is rejected: Delta=',
     +    delta,'; Threshold=  ',reject
          write(*,*) ' REJEDEL: Path ',IIII(i), ' is rejected: Delta=',
     +    delta,'; Threshold=  ',reject
        endif
      enddo
C------- End of Rejection ---------------------------------------
      nout=j   
      write(8,*) ' REJEDEL: There were', n, ' paths; only ',nout,
     +' paths left after cleaning'
      write(*,*) ' REJEDEL: There were', n, ' paths; only ',nout,
     +' paths left after cleaning'
      return
 993  write(*,*)' REJEDEL: WRONG PERIOD=',Tper,'; STOP'
      stop
      end                                                       
