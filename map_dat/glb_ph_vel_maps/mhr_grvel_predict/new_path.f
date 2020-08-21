C################################################################
       subroutine path(delt,slat1,slon1,slat2,slon2,
     1           slat_curr,slong_curr,xray,np)
C-------input: coordinates of ray starting and ending points (in degrees)
C--------------slat1,slon1; slat2,slon2;
C-------input: increment along the ray (in degrees) - delt; 
C-------output: number of output ray points - np;
C-------output: coordinates of ray points in degrees: slat_curr(i),slong_curr(i)
C-------i=1,np
C---------------------------------------------------------------------
	    parameter (nmax=1000)
	    logical k_mer1,k_mer2
	    real*8 dslong,dangle,dcol1,ddelt,dcolat,dBang
	    real*4 slat_curr(nmax), slong_curr(nmax),xray(nmax)
	    data p/0.0174533/,pi/3.1415927/ ,R0/6371.0/,eps/0.10/,eps1/1.0/
C-------------------------------------------------------------
	    k_mer1=.false.
	    k_mer2=.false.
            del_km=delt*p*R0
	    i_switch=1
	    i_switch2=1
	    if(slon2.lt.slon1)then
	    a=slat1
	    slat1=slat2
	    slat2=a
	    a=slon1
	    slon1=slon2
	    slon2=a
	                     end if
C           Print*,slat1,slon1,slat2,slon2
	    colat1=(90.-slat1)*p
	    colat2=(90.-slat2)*p
	    col1=colat1
	    slat_curr(1)=slat1
	    slong_curr(1)=slon1
	    xray(1)=0.0
	    if(slon1.lt.0.0)slon1=slon1+360.0
	    if(slon2.lt.0.0)slon2=slon2+360.0
	    if(abs(slat1+90.).lt.0.006) slon1=slon2
	    if(abs(slat2+90.).lt.0.006) slon2=slon1
	    call azidl(slat1*p,slon1*p,slat2*p,slon2*p,del,dels,azis,azie)
C	    PRint *,'dist,o=',del/p,';  dist,km=',dels, ';  step,km=',del_km    
C           PRint *,slat1,slon1,slat2,slon2,delt
	    angle=azis
	    sl_abs=abs(slon1-slon2)
	    if(sl_abs.lt.eps)k_mer1=.true.
	    if(abs(sl_abs-180.).lt.eps)k_mer2=.true.
	    if(k_mer1.or.k_mer2)THEN
C	    Print*,kk,'ray goes along meridian'
C-----------ray goes along meridian--------------------------S
	    delta=delt   
	    colsum=(colat1+colat2)/p
	    do i=2,nmax
	    xray(i)=xray(i-1)+del_km
	    if(abs(xray(i)-dels).lt.del_km/10.) go to 77
	    if(xray(i).ge.dels) go to 77
C------------end of the ray found
	    if(k_mer1)                             Then
C-----------the same hemisphere-------------------------------S
C           if(i.eq.2)print*,'The same hemisphere'
            if(colat1.gt.colat2)delta=-delt  
	    slat_curr(i)=slat_curr(i-1) - delta
	    slong_curr(i)=slon1
C-----------the same hemisphere-------------------------------E
	                                            Else 
C-----------different hemispheres-------------------------------S
	    delta=delt   
	    if(360.-colsum.gt.colsum)          theN
C--------across North Pole---------------S
C           if(i.eq.2) print*,'across North Pole'
	    if(slat_curr(i-1)+delta.le.90.0.and.i_switch.eq.1) then
	    slat_curr(i)=slat_curr(i-1)+delta
	    slong_curr(i)=slon1
					  else
            i_switch=2
	    slat_curr(i)=slat_curr(i-1)-delta
	    slong_curr(i)=slon2
                                         endif
C--------across North Pole---------------E
                                                elsE
C--------across South Pole---------------S
C           if(i.eq.2) print*,'across South Pole'
	    if(slat_curr(i-1)-delta.ge.-90.0.and.i_switch.eq.1) then
             slat_curr(i)=slat_curr(i-1)-delta
	    slong_curr(i)=slon1
					 else
            i_switch=2
	    slat_curr(i)=slat_curr(i-1)+delta
	    slong_curr(i)=slon2
					  endif
						endiF
                                                   Endif
C-----------different hemispheres-------------------------------E
            enddo
            go to 77
					    ELSE 
C--------across South Pole---------------E
C-----------ray goes along meridian--------------------------E
C-------------------------normal ray--------------------------------S
C           Print*,'Normal ray'
C-----------------------------------------------------------------
	    do i=1,nmax
C   input: colatitude of the previous point col1;
C          increment of arc delt*p;
C          angle between meridian of the previous point and arc
C   output: colatitude of the next point;
C	    angle between two meridians slong
C	    angle between meridian of the next point and arc;
            dangle=angle
	    dcol1=col1
	    ddelt=delt*p
 	    call sph_tri(dcol1,ddelt,dangle,dcolat,dslong,dBang)
	    Bang=dBang
	    colat=dcolat
	    slong=dslong
            xray(i+1)=xray(i)+del_km     
	    if(xray(i+1).ge.dels) go to 1
	    slat_curr(i+1)=(pi/2.-colat)/p
	    slong_curr(i+1)=slong_curr(i)+slong/p
 	    IF(slong_curr(i+1).gt.360.)slong_curr(i+1)=slong_curr(i+1)-360.
	    IF(slong_curr(i+1).lt.0.)slong_curr(i+1)=slong_curr(i+1)+360.
	    col1=colat
	    angle=pi-Bang 
C-----------near South Pole--------------------------------------S
	    if(i_switch2.eq.1)then
	    IF(col1/p.gt.180.-eps1)then
C-----------to jump across the South Pole----------------S
C           print*,'Too close to the South Pole - jumping!'
	    diff_lat=slat_curr(i+1)-slat_curr(i)
	    diff_lon=slong_curr(i)-slong_curr(i-1)
	    slat_curr(i+1)=slat_curr(i)+diff_lat
	    slong_curr(i+1)=slong_curr(i)+diff_lon +180.
C-----------to find the length of jump--------S
	    call azidl(slat_curr(i+1)*p,slong_curr(i+1)*p,slat_curr(i)*p,slong_curr(i)*p,del,dels_new,azis_new,azie)
            xray(i+1)=xray(i)+delS_new   
C-----------to find the length of jump--------E
C-----------to jump across the South Pole----------------E
	    if(slat_curr(i+1).lt.-90.) slat_curr(i+1)=-(180.+ slat_curr(i+1))
 	    IF(slong_curr(i+1).gt.360.)slong_curr(i+1)=slong_curr(i+1)-360.
	    call azidl(slat_curr(i+1)*p,slong_curr(i+1)*p,slat2*p,slon2*p,del,dels_new,azis_new,azie)
	    angle=azis_new
	    col1=(90.-slat_curr(i+1))*p
	    i_switch2=i_switch2+1
				    endif
				    endif
C-----------near South Pole--------------------------------------E
	    enddo
	    np=i+1
1           xray(i+1)=dels
            slong_curr(i+1)=slon2
	    slat_curr(i+1)=slat2
	    np=i+1
C-------------------------normal ray--------------------------------E
                                     ENDIF
	    GO TO 88
77          slong_curr(i)=slon2
	    slat_curr(i)=slat2
	    xray(i)=dels
	    np=i
88          do i=1,np
            if(slong_curr(i).gt.180.)slong_curr(i)=slong_curr(i)-360.
            enddo
	    return
	    end
C###################################################################
            subroutine  sph_tri(b,c,Aang,a,Cang,Bang)
	    implicit real*8 (a-h,o-z)
	    sin1=dsin((b+c)/2.)
	    sin2=dsin((b-c)/2.)
	    cos1=dcos((b+c)/2.)
	    cos2=dcos((b-c)/2.)
	    ctg=1./dtan(Aang/2.)
	    tgBCs=ctg*cos2/cos1
	    tgBCd=ctg*sin2/sin1
	    BCs=datan(tgBCs)
	    BCd=datan(tgBCd)
	    Bang=(BCs+BCd)
	    Cang=(BCs-BCd)
	    cos_a=dcos(b)*dcos(c)+dsin(b)*dsin(c)*dcos(Aang)
            a=dacos(cos_a)        
	    return
	    end
