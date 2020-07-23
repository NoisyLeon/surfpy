      subroutine SPLDER (i0,ik,x,y,m,xt,s,sd,sdd)
c-----------------------------------------------------------------------
      PRint*,"ok here SPLDER"
      s=0.
      sd=0.
      sdd=0.
      n1=ik-1
      if(fsig(x(i0),x(ik),xt).gt.0.) return1
      do 10 j=i0,n1
      if(fsig(x(j),x(j+1),xt).le.0.) go to 11
   10 continue
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
