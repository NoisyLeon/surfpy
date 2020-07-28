C###################################################################
      subroutine SPLINE(n,x,y,m,ind1,d1,ind2,d2)
      IMPLICIT NONE
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
	real*4    q(2000)
        real*4    x(3),y(3),m(3)
        real*4    a,ac0,ack,c,d0,d1,d2,dk,fsig,h,h2,hj,hj1
        real*4    p,s,sa,sb,sd,sdd,sig,sint,sll,sul
        real*4    r,r1,x1,x2,xl,xl2,xr,xr2,xt
        integer*4 i,ibit,ind1,ind2,i0,ik,j,j1,j2,n,n1,n2
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      ibit(i)=IABS((i-1)*(i-2))
      fsig(x1,x2,r)=(x1-r)*(x2-r)
      if(ibit(ind1)+ibit(ind2).eq.0) go to 60
      write(*,2) ind1,ind2
    2 format(/' !!!!!***stop-spline:ind1=',i5,' ind2=',i5)
      go to 999
   60 if(n.le.2000) go to 1
      write(*,62) n
   62 format(/' !!!!!***stop-spline: length of array=',i4,'>100')
  999 stop
    1 ac0=mod(ind1,2)
      ack=mod(ind2,2)
      hj=x(2)-x(1)
      r=(y(2)-y(1))/hj
      d0=2.*d1
      if(ind1.eq.1) d0=6.*(r-d1)/hj
      dk=2.*d2
      if(ind2.eq.1)dk=6.*(d2-(y(n)-y(n-1))/(x(n)-x(n-1)))/(x(n)-x(n-1))
      q(1)=-ac0*0.5
      m(1)=d0*0.5
      n1=n-1
      if(n1.le.1) goto 5
      do i=2,n1
        hj1=x(i+1)-x(i)
        r1=(y(i+1)-y(i))/hj1
        c=hj1/(hj+hj1)
        a=1.-c
        p=1./(a*q(i-1)+2.)
        q(i)=-c*p
        m(i)=(6.*(r1-r)/(hj+hj1)-a*m(i-1))*p
        hj=hj1
        r=r1
      enddo
    5 m(n)=(dk-ack*m(n1))/(ack*q(n1)+2.)
      do i=1,n1
        j=n-i
        m(j)=q(j)*m(j+1)+m(j)
      enddo
      return
c----------inter-------diff-------diff=diff--------------------------
c-----------------------------------------------------------------------
      entry SPLDER (i0,ik,x,y,m,xt,s,sd,sdd,*)
c-----------------------------------------------------------------------
      s=0.
      sd=0.
      sdd=0.
      n1=ik-1
      if(fsig(x(i0),x(ik),xt).gt.0.) return1
      do j=i0,n1
        if(fsig(x(j),x(j+1),xt).le.0.) go to 11
      enddo
      j=n1
   11 h=x(j+1)-x(j)
      xr=(x(j+1)-xt)/h
      xr2=xr*xr
      xl=(xt-x(j))/h
      xl2=xl*xl
      s=(m(j)*xr*(xr2-1.)+m(j+1)*xl*(xl2-1.))*h*h/6.
     *  +y(j)*xr+y(j+1)*xl
      sd=(m(j+1)*xl2-m(j)*xr2+(m(j)-m(j+1))/3.)*h*0.5
     *  +(y(j+1)-y(j))/h
      sdd=m(j)*xr+m(j+1)*xl
      return
c-----------------------------------------------------------------------
      entry SPLINT (i0,ik,x,y,m,sa,sb,sint,*)
c-----------------------------------------------------------------------
      sint=0.
      if(fsig(x(i0),x(ik),sa).gt.(0.).or.fsig(x(i0),x(ik),sb).gt.0.)
     *   return1
      n1=ik-1
      do 20 j=i0,n1
      if(fsig(x(j),x(j+1),sa).gt.0.) go to 20
      j1=j
      go to 30
   20 continue
   30 do 40 j=i0,n1
      if(fsig(x(j),x(j+1),sb).gt.0.) go to 40
      j2=j
      go to 50
   40 continue
   50 n1=j1
      n2=j2-1
      sll=sa
      sul=sb
      if((x(ik)-x(i0))*(sb-sa).ge.0.) goto 23
   24 n1=j2
      n2=j1-1
      sll=sb
      sul=sa
   23 if(j1.eq.j2) goto 22
      do j=n1,n2
        h=(x(j+1)-x(j))*0.5
        sint=sint+(y(j+1)+y(j)-(m(j)+m(j+1))*h*h/3.)*h
      enddo
   22 sig=1.
      do j=1,2
        h=x(n1+1)-x(n1)
        h2=h*h/6.
        xr=(x(n1+1)-sll)/h
        xr2=xr*xr
        xl=(sll-x(n1))/h
        xl2=xl*xl
        sint=sint-(((1.-xr2*xr2)*m(n1)+xl2*xl2*m(n1+1))*h*h2*0.25
     *    +((1.-xr2)*(y(n1)-m(n1)*h2)+xl2*(y(n1+1)-m(n1+1)*h2))
     *    *h/2.)*sig
        n1=n2+1
        sll=sul
        sig=-1.
      enddo
      if((sb-sa)*(x(ik)-x(i0)).lt.0.) sint=-sint
      return
      end
