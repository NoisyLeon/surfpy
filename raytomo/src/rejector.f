c***************************************************************
c Reject doubtful input data by velocity deviation
c N number of paths for processing
c***************************************************************
      SUBROUTINE REJECTOR
      IMPLICIT NONE
      include "tomo.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*4    aver_vel,q,q1,sum_vel
      integer*4 i
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      sum_vel=0.0
c
c   finding average velocity
c
      do  i=1,n
        sum_vel=sum_vel+t(i)
      enddo
      aver_vel=sum_vel/n
      write(*,*) 'REJECTOR: Average velocity for input data=',
     *aver_vel,' (km/s)'
      write(*,*) 'REJECTOR: Permitted deviation=',
     *reject*aver_vel/100.,' (km/s)'
      write(8,*) ' REJECTOR:   Average velocity for input data=',
     *aver_vel,' (km/s)'
      write(8,*) ' REJECTOR: Permitted deviation=',
     *reject*aver_vel/100.,' (km/s)'
      m=0
c
c   cleaning
c
      do i=1,n
        q=(t(i)/aver_vel-1.)*100.
        q1=ABS(q)
        if(q1.le.reject) then
          m=m+1
          te0(m)=te0(i)
          fi0(m)=fi0(i)
          te(m)=te(i)
          fi(m)=fi(i)
          t(m)=t(i)
          weight(m)=weight(i)
          iray2(m)=iray2(i)
        else
          write(8,*) ' REJECTOR: Path ',IIII(i), ' is rejected: deviation=',
     *      q,' %; threshold=  ',reject
          write(*,*) ' REJECTOR: Path ',IIII(i), ' is rejected: deviation=',
     *      q,' %; threshold=  ',reject
        endif
      enddo
      nout=m   
      write(8,*) ' REJECTOR: There were', n, ' paths; only ',nout,
     +' paths left after cleaning'
      write(*,*) ' REJECTOR: There were', n, ' paths; only ',nout,
     +' paths left after cleaning'
      return
      end                                                       
