c#######################################################################
c estimate peacewise cubic spline coefficients
c#######################################################################
      subroutine mspline (ip,n,x,y,ind1,d1,ind2,d2)
      implicit none      
      real*8    x(1001),y(1001),c(4,1001),cc(5,10001)
      integer*4 i,j,ip,n,ind1,ind2
      real*8    s,z,d1,d2
c ---
      integer*4 nn(10),ndim(10)
      real*8    xx(1001,10),sc(4,1001,10),scc(5,1001,10)
      common /mspdat/nn,ndim,xx,sc,scc
c ---
c validate input parameters
      if(n.lt.2.or.n.gt.1001) then
        write(*,*) 'mspline: n=',n,
     *             ' is out of bounds. Should be in range 2-1001.'
        stop
      endif
      if(ip.lt.1.or.ip.gt.10) then
        write(*,*) 'mspline: ip=',ip,
     *             ' is out of bounds. Should be in range 1-10.'
        stop
      endif
      if(ind1.lt.0.or.ind1.gt.2) then
        write(*,*) 'mspline: ind1=',ind1,
     *             ' is out of bounds. Should be in range 0-2.'
        stop
      endif
      if(ind2.lt.0.or.ind2.gt.2) then
        write(*,*) 'mspline: ind2=',ind2,
     *             ' is out of bounds. Should be in range 0-2.'
        stop
      endif
      do i =1,n
          c(1,i) = y(i)
      enddo
      c(2,1) = d1
      c(2,n) = d2
      call cubspl(x,c,n,ind1,ind2)
c remove 2 and 6
      do i = 1,n-1
        c(3,i) = c(3,i)/2.0d0
        c(4,i) = c(4,i)/6.0d0
      enddo
c create integral coefficients
      s = 0.0d0
      do i = 1,n-1
        do j =1,4
          cc(j+1,i) = c(j,i)/j
        enddo
        cc(1,i) = s
        z = x(i+1)-x(i)
        s = s+z*(cc(2,i)+z*(cc(3,i)+z*(cc(4,i)+z*cc(5,i))))
      enddo
c save coefficients into common block mspdat
      nn(ip) = n
      do i = 1,n
        xx(i,ip) = x(i)
        do j = 1,4
           sc(j,i,ip) = c(j,i)
           scc(j,i,ip) = cc(j,i)
        enddo
        scc(5,i,ip) = cc(5,i)
      enddo
      return
      end
c#######################################################################
c integral for spline
c#######################################################################
      subroutine msplint (ip,sa,sb,sint,ierr)
      implicit none
      integer*4 ip,ierr,i,ii,j,n
      real*8    sa,sb,sint,x1,x2,r,fsig,s1,s2,z
      real*8    x(1001),cc(5,10001)
c ---
      integer*4 nn(10),ndim(10)
      real*8    xx(1001,10),sc(4,1001,10),scc(5,1001,10)
      common /mspdat/nn,ndim,xx,sc,scc
c ---
      fsig(x1,x2,r)=(x1-r)*(x2-r)

c restore polinomial coefficients
      ierr = 0
      n = nn(ip)
      do i = 1,n
        x(i) = xx(i,ip)
        do j = 1,5
          cc(j,i) = scc(j,i,ip)
        enddo
      enddo
c compute integral for sa
      ii = 0
      do i = 1,n-1
        if(fsig(x(i),x(i+1),sa).le.0.0d0) ii = i
      enddo
      if(ii.eq.0) then
        if(sa.le.x(1)) ii = 1
        if(sa.ge.x(n-1)) ii = n-1
      endif
      z = sa-x(ii)
      s1 = cc(1,ii)+z*(cc(2,ii)+z*(cc(3,ii)+z*(cc(4,ii)+z*cc(5,ii))))
c compute integral for sb
      ii = 0
      do i = 1,n-1
        if(fsig(x(i),x(i+1),sb).le.0.0d0) ii = i
      enddo
      if(ii.eq.0) then
        if(sb.le.x(1)) ii = 1
        if(sb.ge.x(n-1)) ii = n-1
      endif
      z = sb-x(ii)
      s2 = cc(1,ii)+z*(cc(2,ii)+z*(cc(3,ii)+z*(cc(4,ii)+z*cc(5,ii))))
      sint = s2-s1
      return
      end
c#######################################################################
c spline interpolation of funcion and its 1st and 2nd derivatives
c#######################################################################
      subroutine msplder (ip,xt,s,sd,sdd,ierr)
      implicit none
      integer*4 ip,ierr
      real*8    xt,s,sd,sdd
c ---
      integer*4 nn(10),ndim(10)
      real*8    xx(1001,10),sc(4,1001,10),scc(5,1001,10)
      common /mspdat/nn,ndim,xx,sc,scc
c ---
      integer*4 i,j,ii,n
      real*8    x(1001),c(4,1001)
      real*8    x1,x2,r,fsig,z

      fsig(x1,x2,r)=(x1-r)*(x2-r)
c restore polinomial coefficients
      ierr = 0
      n = nn(ip)
      do i = 1,n
        x(i) = xx(i,ip)
        do j = 1,4
          c(j,i) = sc(j,i,ip)
        enddo
      enddo
c find interval for interpolation
      ii = 0
      do i = 1,n-1
        if(fsig(x(i),x(i+1),xt).le.0.0d0) ii = i
      enddo
      if(ii.eq.0) then
        if(xt.le.x(1)) ii = 1
        if(xt.ge.x(n-1)) ii = n-1
      endif
      z = xt-x(ii)
      s = c(1,ii)+z*(c(2,ii)+z*(c(3,ii)+z*c(4,ii)))
      sd = c(2,ii)+z*(2.0d0*c(3,ii)+3.0d0*z*c(4,ii))
      sdd = 2.0d0*c(3,ii)+6.0d0*z*c(4,ii)
      return
      end
c#######################################################################
      subroutine cubspl ( tau, c, n, ibcbeg, ibcend )
c  from  * a practical guide to splines *  by c. de boor    
c     ************************  input  ***************************
c     n = number of data points. assumed to be greater than 1.
c     (tau(i), c(1,i), i=1,...,n) = abscissae and ordinates of the
c        data points. tau is assumed to be strictly increasing.
c     ibcbeg, ibcend = boundary condition indicators, and
c     c(2,1), c(2,n) = boundary condition information. Specifically,
c        ibcbeg = 0  means no boundary condition at tau(1) is given.
c           in this case, the not-a-knot condition is used, i.e. the
c           jump in the third derivative across tau(2) is forced to
c           zero, thus the first and the second cubic polynomial pieces
c           are made to coincide.)
c        ibcbeg = 1  means that the slope at tau(1) is made to equal
c           c(2,1), supplied by input.
c        ibcbeg = 2  means that the second derivative at tau(1) is
c           made to equal c(2,1), supplied by input.
c        ibcend = 0, 1, or 2 has analogous meaning concerning the
c           boundary condition at tau(n), with the additional infor-
c           mation taken from c(2,n).
c     ***********************  output  **************************
c     c(j,i), j=1,...,4; i=1,...,l (= n-1) = the polynomial coefficients
c        of the cubic interpolating spline with interior knots (or
c        joints) tau(2), ..., tau(n-1). precisely, in the interval
c        (tau(i), tau(i+1)), the spline f is given by
c           f(x) = c(1,i)+h*(c(2,i)+h*(c(3,i)+h*c(4,i)/3.)/2.)
c        where h = x - tau(i). the function program *ppvalu* may be
c        used to evaluate f or its derivatives from tau,c, l = n-1,
c        and k=4.
      implicit none
c ---
      integer*4 ibcbeg,ibcend,n,i,j,l,m
      real*8    c(4,n),tau(n),divdf1,divdf3,dtau,g
c ---
c****** a tridiagonal linear system for the unknown slopes s(i) of
c  f  at tau(i), i=1,...,n, is generated and then solved by gauss elim-
c  ination, with s(i) ending up in c(2,i), all i.
c     c(3,.) and c(4,.) are used initially for temporary storage.
      l = n-1
c compute first differences of tau sequence and store in c(3,.). also,
c compute first divided difference of data and store in c(4,.).
      do m=2,n
         c(3,m) = tau(m) - tau(m-1)
         c(4,m) = (c(1,m) - c(1,m-1))/c(3,m)
      enddo
c construct first equation from the boundary condition, of the form
c             c(4,1)*s(1) + c(3,1)*s(2) = c(2,1)
      if (ibcbeg-1) 11,15,16
   11 if (n .gt. 2) goto 12
c     no condition at left end and n = 2.
      c(4,1) = 1.0d0
      c(3,1) = 1.0d0
      c(2,1) = 2.0d0*c(4,2)
      goto 25
c     not-a-knot condition at left end and n .gt. 2.
   12 c(4,1) = c(3,3)
      c(3,1) = c(3,2) + c(3,3)
      c(2,1) =((c(3,2)+2.0d0*c(3,1))*c(4,2)*c(3,3)+c(3,2)**2*c(4,3))/c(3,1)
      goto 19
c     slope prescribed at left end.
   15 c(4,1) = 1.0d0
      c(3,1) = 0.0d0
      goto 18
c     second derivative prescribed at left end.
   16 c(4,1) = 2.0d0
      c(3,1) = 1.0d0
      c(2,1) = 3.0d0*c(4,2) - c(3,2)/2.0d0*c(2,1)
   18 if(n .eq. 2) goto 25
c  if there are interior knots, generate the corresp. equations and car-
c  ry out the forward pass of gauss elimination, after which the m-th
c  equation reads    c(4,m)*s(m) + c(3,m)*s(m+1) = c(2,m).
   19 do m=2,l
         g = -c(3,m+1)/c(4,m-1)
         c(2,m) = g*c(2,m-1) + 3.0d0*(c(3,m)*c(4,m+1)+c(3,m+1)*c(4,m))
         c(4,m) = g*c(3,m-1) + 2.0d0*(c(3,m) + c(3,m+1))
      enddo
c construct last equation from the second boundary condition, of the form
c           (-g*c(4,n-1))*s(n-1) + c(4,n)*s(n) = c(2,n)
c     if slope is prescribed at right end, one can go directly to back-
c     substitution, since c array happens to be set up just right for it
c     at this point.
      if (ibcend-1) 21,30,24
   21 if (n .eq. 3 .and. ibcbeg .eq. 0) goto 22
c     not-a-knot and n .ge. 3, and either n.gt.3 or  also not-a-knot at
c     left end point.
      g = c(3,n-1) + c(3,n)
      c(2,n) = ((c(3,n)+2.0d0*g)*c(4,n)*c(3,n-1)
     *            + c(3,n)**2*(c(1,n-1)-c(1,n-2))/c(3,n-1))/g
      g = -g/c(4,n-1)
      c(4,n) = c(3,n-1)
      goto 29
c     either (n=3 and not-a-knot also at left) or (n=2 and not not-a-
c     knot at left end point).
   22 c(2,n) = 2.0d0*c(4,n)
      c(4,n) = 1.0d0
      goto 28
c     second derivative prescribed at right endpoint.
   24 c(2,n) = 3.0d0*c(4,n) + c(3,n)/2.0d0*c(2,n)
      c(4,n) = 2.0d0
      goto 28
   25 if (ibcend-1) 26,30,24
   26 if (ibcbeg .gt. 0) goto 22
c     not-a-knot at right endpoint and at left endpoint and n = 2.
      c(2,n) = c(4,n)
      goto 30
   28 g = -1.0d0/c(4,n-1)
c complete forward pass of gauss elimination.
   29 c(4,n) = g*c(3,n-1) + c(4,n)
      c(2,n) = (g*c(2,n-1) + c(2,n))/c(4,n)
c ccarry out back substitution
   30 j = l 
   40 c(2,j) = (c(2,j) - c(3,j)*c(2,j+1))/c(4,j)
      j = j - 1
      if (j .gt. 0) goto 40
c****** generate cubic coefficients in each interval, i.e., the deriv.s
c  at its left endpoint, from value and slope at its endpoints.
      do i=2,n
         dtau = c(3,i)
         divdf1 = (c(1,i) - c(1,i-1))/dtau
         divdf3 = c(2,i-1) + c(2,i) - 2.0d0*divdf1
         c(3,i-1) = 2.0d0*(divdf1 - c(2,i-1) - divdf3)/dtau
         c(4,i-1) = (divdf3/dtau)*(6.0d0/dtau)
      enddo
      return
      end
