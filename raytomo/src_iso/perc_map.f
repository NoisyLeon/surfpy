C#############################################################
      SUBROUTINE PERC_MAP(infile,nxy)
      IMPLICIT NONE
      include "line.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      character*160  infile,outfile
      real*8 d_clon,d_clat,vel
      real*8 dcof,dsum,dscof,aver_vel,velper
      integer*4 i,lnd,nq,nxy
      integer*4 LNBLNK
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      lnd=LNBLNK(infile)
      outfile=infile(1:lnd)//'_%_'
      open(unit=61,file=infile,STATUS='OLD')
      open(unit=62,file=outfile,status='UNKNOWN')
      dsum=0.0d0
      dscof=0.0d0
      do i=1,nxy    
        read(61,'(3f12.4)') d_clon,d_clat,vel
        dcof=DABS(DCOS(d_clat*drad))
        dsum=dsum+vel*dcof
        dscof=dscof+dcof
      enddo
      close(61)
      open(unit=61,file=infile,STATUS='OLD')
      nq=i-1
      aver_vel=dsum/dscof
      do i=1,nxy    
        read(61,'(3f12.4)') d_clon,d_clat,vel
        velper=100.*(vel-aver_vel)/aver_vel
        write(62,'(2f12.4,f12.5)') d_clon,d_clat,velper
      enddo
      close(61)
      close(62)
      write (8,1000) outfile(1:lnd)//'_%_',aver_vel,nq
      write (*,1000) outfile(1:lnd)//'_%_',aver_vel,nq
 1000 format(' average velocity for %-map ',a,' =',f12.5,
     *' (km/sec), npoints on a grid=',i7)
      return
      end
