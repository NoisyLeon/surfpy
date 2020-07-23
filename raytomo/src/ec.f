c******************************************************
      SUBROUTINE ESTRACEN(lidd,nlidd,step,e,nmm,denf,ff,xn,gis,ierr)
      implicit none
      include "tomo.h"
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      real*4    denf(NRAZ)
      real*4    ff(NRAZ+(NRAZA*NAZIPL-1)),gis(IAZIM,NRAZA)
      real*8    grid(4,3),w(4),e(3),step,xn(3),worazi,azms
      real*4    RBIMOD
      integer*4 lidd(NRAZ),nclo(4),ngr_out(4)
      integer*4 i,j,k,ierr,ncl,ncoin,nlidd,nmm,mo,ncl1
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
      CALL E2PNTS(e,ngr_out,grid)
      CALL TRIAW(ngr_out,grid,e,w,ierr)
      do 20 j=1,4
      nclo(j)=ilocr(ngr_out(j))
      if(nclo(j).eq.0) then
      ierr=1
      return
      endif
   20 continue
      do 10 j=1,4
      if(j.eq.1) ncoin=nclo(1)*nmm+nclo(4)
      ncl=nclo(j)
      if(ncl.gt.nm) then
        write(*,*)'ERROR with local number',nm,ncl
        ierr=1
        return
      endif
      CALL SAZI(e,xn,azms)
      worazi=step*R/RBIMOD(e)
      do 30 i=1,nlidd
      if(lidd(i).eq.ncl) goto 40
   30 continue
      nlidd=nlidd+1
      if(nlidd.gt.NRAZ) then
      write(*,*) '(R0014) TOO SMALL SIZE OF lidd',nlidd-1
      STOP
      endif
      lidd(nlidd)=ncl
   40 do mo=1,1
      ncl1=nmm*(mo-1)+ncl
      if(mo.eq.1) ff(ncl1)=ff(ncl1)+w(j)*worazi
cxx   if(mo.eq.2) ff(ncl1)=ff(ncl1)-w(j)*worazi*DCOS(2.0d0*azms)
cxx   if(mo.eq.3) ff(ncl1)=ff(ncl1)-w(j)*worazi*DSIN(2.0d0*azms)
cxx   if(mo.eq.4) ff(ncl1)=ff(ncl1)-w(j)*worazi*DCOS(4.0d0*azms)
cxx   if(mo.eq.5) ff(ncl1)=ff(ncl1)-w(j)*worazi*DSIN(4.0d0*azms)
      enddo
      if(ncoin0.ne.ncoin) then 
        denf(ncl)=1.0
          CALL AZI(j,grid,e,xn,azms)
          k=azms/const/(180.0/FLOAT(IAZIM))+1.0d0
          if(k.lt.1) k=1
          if(k.gt.IAZIM) k=IAZIM
        gis(k,ncl)=gis(k,ncl)+1.0
      endif
   10 continue
      ncoin0=ncoin
      return
      end
C******************************************************
      SUBROUTINE EINTEGRN(step,e,xn,nmm,sum,fsl,ierr)
      implicit none
      include "tomo.h"
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      integer*4 nmm,ierr
      real*4    fsl(nmm),sum
      real*8    grid(4,3),w(4),e(3),step,dsum,smult
      real*8    xn(3),scell
      real*4    RBIMOD
      integer*4 ngr_out(4),nclo,j,mo
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ierr=0
      CALL E2PNTS(e,ngr_out,grid)
      sum=0.0
      dsum=0.0d0
      smult=step*R/RBIMOD(e)
      do 20 mo=1,1
      do 10 j=1,4
      nclo=ilocr(ngr_out(j))
      if(nclo.lt.1) then
      ierr=1
      return
      endif
   10 w(j)=fsl(nmm*(mo-1)+nclo)
      CALL TRIAI(grid,e,w,scell)
      if(mo.eq.1) dsum=dsum+scell*smult
cxx   if(mo.eq.2) dsum=dsum-scell*smult*DCOS(2.0d0*azms)
cxx   if(mo.eq.3) dsum=dsum-scell*smult*DSIN(2.0d0*azms)
cxx   if(mo.eq.4) dsum=dsum-scell*smult*DCOS(4.0d0*azms)
cxx   if(mo.eq.5) dsum=dsum-scell*smult*DSIN(4.0d0*azms)
   20 continue
      sum=dsum
      return
      end
