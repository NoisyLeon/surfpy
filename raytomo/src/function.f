c===============================================================
c This is an example of fresnel zone function
c Replace this program with a new one as you like
c For each Distance program determinates width of zode using
c exact definition: abs(d - d') < Lambda/x_zone, so for each
c Distance you need to call program once,
c===============================================================
      SUBROUTINE FUNCT(Lambda,TotalDistance,Distance,Step,Value,Iyes)
      implicit none
      include "tomo.h"
      include "line.h"
      real*8    TotalDistance, Distance, PhaseWidth,Wwexp
      real*8    Lambda,Step,Value,dis,dw,dwc,ar1(2),ar2(2)
      integer*4 Iyes,i,i1,j1,flag
c   The equation used here is courtesy of Jesper Spetzler of 
c   Utrecht University, Netherlands 2001
c
c   The purpose of this subroutine is to calculate the width of 
c   the first Fresnel zone for both phase and group waves. This
c   formula is actually for phase, but the group fresnel zone 
c   width is approximated by making it equal to the phase
c   fresnel zone width. It is not neccessary to specify whether
c   these are Rayleigh or Love waves, only that you input the
c   correct slowness for period and wave type.
c========================================================================
c INPUT PARAMETERS:
c
c   TotalDistance - distance from the source to the receiver, (rad);
c   Distance      - distance from the source to the point, (rad);
c   Step          - size of grid cell for integration (rad);
c
c  OUTPUT PARAMETERS:
c
c   Value         - value of amplitude in fresnel zone;
c   Iyes          - =1, inside zone, =0 - outside;
c========================================================================
c  WORKING VARIABLES:
c   Lambda        - wavelength in (rad);
c   PhaseWidth    - half width of the fresnel zone (phase), (rad);


      
c   R is the Earth's Radius = 6371.0 (km).

      Value=0.0d0
      Wwexp=wexp*drad
      
c   This is the actual formula with group fresnel zone width being 
c   mostly proportional to the phase and so it is treated as such.      

      PhaseWidth=0.0d0
      do i=1,90
          dw=(i-1)*Step
          dwc=dcos(dw)
          ar1(1)=dacos(dwc*dcos(Distance))
          ar1(2)=2.0d0*dpi-ar1(1)
          ar2(1)=dacos(dwc*dcos(TotalDistance-Distance))
          ar2(2)=2.0d0*dpi-ar2(1)
          flag=0
          do i1=1,2
          do j1=1,2
             dis=ar1(i1)+ar2(j1)
             if(DABS(TotalDistance-dis).le.Lambda/x_zone) then
                PhaseWidth=i*Step-Step/2.0d0
                flag=1
             endif
          enddo
          enddo
      if(flag.eq.0) goto 1
      enddo
c   The minimum amount for the fresnel phase width, is
c   Lambda/4, so that we don't end up with infintesimally
c   small fresnel zones which are not physically realized. 
  1   Iyes=0
      if(PhaseWidth.gt.Step/4.0d0) then
cxx      if(PhaseWidth.lt.Lambda/4.0d0) PhaseWidth=Lambda/4.0d0
         if(PhaseWidth.lt.Wwexp) PhaseWidth=Wwexp
         Iyes=1
         Value=1.0d0/(PhaseWidth*2.0d0)
      endif
      return
      end
