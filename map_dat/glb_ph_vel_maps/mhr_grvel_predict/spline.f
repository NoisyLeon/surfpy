
C#######################################################################
      subroutine SPLINE (n,x,y,m,ind1,d1,ind2,d2)
c-----------------------------------------------------------------------
	real*4 q(3000)
        real x(n),y(n),m(n)
c-----------------------------------------------------------------------
      ibit(i)=iabs((i-1)*(i-2))
      fsig(x1,x2,r)=(x1-r)*(x2-r)
      if(ibit(ind1)+ibit(ind2).eq.0) go to 60
      print 2 ,ind1,ind2
    2 format(/' !!!!!***stop-spline:ind1=',i5,' ind2=',i5)
      go to 999
   60 if(n.le.3001) go to 1
      print    62,n
   62 format(/' !!!!!***stop-spline: length of array=',i4,'>3000')
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
      do 3 i=2,n1
      hj1=x(i+1)-x(i)
      r1=(y(i+1)-y(i))/hj1
      c=hj1/(hj+hj1)
      a=1.-c
      p=1./(a*q(i-1)+2.)
      q(i)=-c*p
      m(i)=(6.*(r1-r)/(hj+hj1)-a*m(i-1))*p
      hj=hj1
      r=r1
    3 continue
    5 m(n)=(dk-ack*m(n1))/(ack*q(n1)+2.)
      do 4 i=1,n1
      j=n-i
    4 m(j)=q(j)*m(j+1)+m(j)
      return
      end
c----------inter-------diff-------diff=diff--------------------------
c-----------------------------------------------------------------------
c      entry SPLDER (i0,ik,x,y,m,xt,s,sd,sdd,*)
      subroutine SPLDER (i0,ik,x,y,m,xt,s,sd,sdd,*)
        real*4 q(3000)
        real x(ik),y(ik),m(ik)
      ibit(i)=iabs((i-1)*(i-2))
      fsig(x1,x2,r)=(x1-r)*(x2-r)
c-----------------------------------------------------------------------
c      PRint*,"ok here SPLDER"
      s=0.
      sd=0.
      sdd=0.
      n1=ik-1
      
c      PRint*,"ok here SPLDER 1",i0,ik
c      PRint*,"ok here SPLDER 1",x(i0),x(ik)
      if ((x(i0)-xt)*(x(ik)-xt).gt.0.) return1
      do 10 j=i0,n1
      if((x(j)-xt)*(x(j+1)-xt).le.0.) go to 11
   10 continue

c      PRint*,"ok here SPLDER"

      j=n1
   11 h=x(j+1)-x(j)
c      PRint*,"ok here SPLDER",h
      
      xr=(x(j+1)-xt)/h
      xr2=xr*xr
      xl=(xt-x(j))/h
      xl2=xl*xl
      s=(m(j)*xr*(xr2-1.)+m(j+1)*xl*(xl2-1.))*h*h/6.
     *  +y(j)*xr+y(j+1)*xl
      sd=(m(j+1)*xl2-m(j)*xr2+(m(j)-m(j+1))/3.)*h*0.5
     *  +(y(j+1)-y(j))/h
      sdd=m(j)*xr+m(j+1)*xl
c      PRint*,"finish here SPLDER"
      return
      end
c-----------------------------------------------------------------------
c      entry SPLINT (i0,ik,x,y,m,sa,sb,sint,*)
      subroutine SPLINT (i0,ik,x,y,m,sa,sb,sint,*)
c-----------------------------------------------------------------------
c      sint=0.
      ibit(i)=iabs((i-1)*(i-2))
      fsig(x1,x2,r)=(x1-r)*(x2-r)
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
      do 21 j=n1,n2
      h=(x(j+1)-x(j))*0.5
   21 sint=sint+(y(j+1)+y(j)-(m(j)+m(j+1))*h*h/3.)*h
   22 sig=1.
      do 26 j=1,2
      h=x(n1+1)-x(n1)
      h2=h*h/6.
      xr=(x(n1+1)-sll)/h
      xr2=xr*xr
      xl=(sll-x(n1))/h
      xl2=xl*xl
      sint=sint-(((1.-xr2*xr2)*m(n1)+xl2*xl2*m(n1+1))*h*h2*0.25
     *  +((1.-xr2)*(y(n1)-m(n1)*h2)+xl2*(y(n1+1)-m(n1+1)*h2))
     1  *h/2.)*sig
      n1=n2+1
      sll=sul
   26 sig=-1.
      if((sb-sa)*(x(ik)-x(i0)).lt.0.) sint=-sint
      return
      end
