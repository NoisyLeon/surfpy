c
c parabolic interpolation of signal amplitude and phase,
c finding phase derivative
c
      subroutine fmax(am1,am2,am3,ph1,ph2,ph3,om,dt,t,dph,tm,ph,piover4)
      implicit none
      real*8    t, dph, tm, pi2,ph1,ph2,ph3,ph,om,dt
      real*8    am1,am2,am3,piover4
      real*8    a1,a2,a3,dd
      integer*4 k
c ---
      pi2 = datan(1.0d0)*8.0d0
      dd=am1+am3-2*am2
      t=0.0d0
      if(dd .ne. 0.0d0) then
          t=(am1-am3)/dd/2.0d0
      endif
c  phase derivative
      a1 = ph1
      a2 = ph2
      a3 = ph3
c  check for 2*pi phase jump
      k = nint((a2-a1-om*dt)/pi2)
      a2 = a2-k*pi2
      k = nint((a3-a2-om*dt)/pi2)
      a3 = a3-k*pi2
c interpolation
      dph=t*(a1+a3-2.0d0*a2)+(a3-a1)/2.0d0
      tm=t*t*(am1+am3-2.0d0*am2)/2.0d0+t*(am3-am1)/2.0d0+am2
      ph=t*t*(a1+a3-2.0d0*a2)/2.0d0+t*(a3-a1)/2.0d0+a2+pi2*piover4/8.0d0
      return
      end
