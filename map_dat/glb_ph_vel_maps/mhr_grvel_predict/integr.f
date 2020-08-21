c ***************************************************************
      subroutine integr(x,y,n,ta)
C------integrating travel time by simpson technique-------------
       parameter (nsize=1000)
       integer errr
       real*4 x(1000),y(1000),ta,am1(1000),u(nsize)
       data eps/1.E-05/
       call spline(n,x,y,am1,2,0.,2,0.)
c       PRint*,"ok here"
       dx=(x(n)-x(1))/float(n-1)
	   do i=1,n
	   xcurr=x(1)+dx*(i-1)
	   if(xcurr.lt.x(1)) xcurr=x(1)+eps
	   if(xcurr.gt.x(n)) xcurr=x(n)-eps

c             do k=1,n
c               PRint*, k, x(k),y(k)
c             enddo


c       PRint*,"call splder",n,x(1),x(22)
	   call splder(1,n,x,y,am1,xcurr,u(i),dum,dum1,*994)
       
c       PRint*,"done splder"
	   u(i)=1./u(i)
	   end do
       ta=0.0
       k=-1
	   do i=1,n
	   k=k+2
           if(k+2.le.n) ta=ta+(u(k)+4.*u(k+1)+u(k+2))*dx/3.
	   if(k+1.eq.n) ta=ta+0.5*(u(n)+u(n-1))*dx
 	   end do
		 ta=x(n)/ta
       		 return
994    STOP' TROUBLES WITH IINTERPOLATION'
	       end
