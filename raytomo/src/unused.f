c******************************************************
      SUBROUTINE TO_ORIG1(x1,y1,z1,xf,ang1,xfl,ang2)
      real*8 xf(3),ang1,xfl(3),ang2,w(3),w1(3)
      w(1)=x1
      w(2)=y1
      w(3)=z1
      CALL XTURN(ang1,xf,w,w1)
      CALL XTURN(-ang2,xfl,w1,w)
      x1=w(1)
      y1=w(2)
      z1=w(3)
      return
      end
C******************************************************
      SUBROUTINE DNORM(a,b)
      real*8 a(3),b(3),norm
      norm=DSQRT(a(1)*a(1)+a(2)*a(2)+a(3)*a(3))
      do 1 i=1,3
    1 b(i)=a(i)/norm
      return
      end
C******************************************************
      SUBROUTINE RROTATE(norma,i,x1,y1,z1,out)
      real*8 x1,y1,z1,out
      dimension ind1(6),ind2(6),out(3)
      real*8 fi,ar(3),a(3),ar1(3)
      data ind1/0,0,0,1,1,1/,ind2/0,1,-1,0,1,-1/
      data a/-.57735026918962576450,-.57735026918962576450,
     *.57735026918962576450/,fi/2.09439510239319549229/
      ar(1)=x1
      ar(2)=y1
      ar(3)=z1
      CALL TURN(ind2(i),a,ar,ar1)
      if(ind1(i).eq.1) then
      do 1 j=1,3
    1 ar1(j)=-ar1(j)
      endif
      do 2 j=1,3
    2 out(j)=ar1(j)
      if(norma.eq.1) CALL NORM(out,out)
      return
      end
c******************************************************
      SUBROUTINE RVECT(a,b,c)
      real*8 a(3),b(3),c(3)
      c(1)=a(2)*b(3)-a(3)*b(2)
      c(2)=a(3)*b(1)-a(1)*b(3)
      c(3)=a(1)*b(2)-a(2)*b(1)
      return
      end
