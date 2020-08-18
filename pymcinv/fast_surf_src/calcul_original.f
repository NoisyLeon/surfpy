C--------------------------------------------------------------------
        subroutine calcul(ncount,dx,imax,idispr,idispl,number,numbel,
     *kind,t_base,*)
C--------------------------------------------------------------------
        parameter (nsize=1000,nper=2000,ndep=20,nmod=20)
	common/d/  a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
        common/o/  c(nper,nmod),t(nper),ratio(nper,nmod)
        common/c/nmax,mmax,kmax,idrop,iedit,ndiv,mode,fact,ra_1
        common/newly/ t1,dt,c1,dc,lstop,iq,istru,cinit
        common/rar/ depth(nsize),amprfi(nsize),ampz(nsize),strz(nsize),
     *  strrfi(nsize),mmm
        common/rar1/ dcda,dcdb,dcdr,dwx,g1(nsize),g2(nsize)
        common/rco/ ccc,cvar,ugr,wvno,rat,arle,nwmax
        common/rco1/ sumi0,sumi1,sumi2,sumi3,flagr
        common/derivd/ nderiv,ndpth,dpth(ndep),dspz(ndep),dsprfi(ndep),
     *  drvz(5,ndep),drvrfi(5,ndep)
        common/log/ KEY_ATTEN,KEY_DERIV,KEY_EIGEN,KEY_EIG_NORM,KEY_EIGEN
     *_DER1,KEY_EIGEN_DER2
	common/ref/ a_ref(nsize),b_ref(nsize),rho_ref(nsize),
     *d_ref(nsize),qs_ref(nsize)
        COMMON/NEW/derz(nsize),derr(nsize)
        common/dispe/per_R(200),per_L(200),uR(200,2),uL(200,2),cR(200,2)
     *,cL(200,2)
	common/numbers/n_int_RC,n_int_RU,n_int_LC,n_int_LU,n_layer,n_var
        real*8 dcda(nsize),dcdb(nsize),dcdr(nsize),dwx(nsize)
        logical KEY_DERIV,KEY_ATTEN,KEY_EIGEN,KEY_EIG_NORM, KEY_EIGEN_DE
     *R1, KEY_EIGEN_DER2 
	real*4 ugr_plus(nper),ugr_minus(nper),ugr0(nper),dampl(nper)
	real*4 ccc_plus(nper),ccc_minus(nper),ccc0(nper),mult(nper),
     *amp_exp(nper)
        integer imax(nmod)
	data pi/3.1415927/,R0/6371.0/,aM/1.E+15/,dpuass/1.732051/

c  FORMAT STATEMENTS
11      format(a,/2(6x,i3,2x),6x,i2,8x,i1)
44      format(1x,/(4f10.4))
55      format(10x,3f10.4)
12      format(2(7x,i1,2x),5x,f3.0)
1000    format(5x,i3,2x,4(5x,f6.0,2x),6x,i1)
1001    format(1x,/(8f10.0))
1002    format(a)
1004    format(34x,'Love mode ',i2)
1005    format(32x,'Rayleigh mode ',i2)
1006    format(8x,i2,9x,i3)
1007    format(1x,/(7f10.0))
5000    format(1x,/1x,'t=',f6.2,2x,'c=',f6.4,2x,'cvar=',f6.4,2x,'ugr=',
     *  f6.4,2x,'wvnumb=',e11.4,2x,'al=',e11.4)
5002    format(1x,/3x,'I0 =',e11.4,5x,'I1 =',e11.4,5x,'I2 =',e11.4,
     *  5x,'L =',e11.4)
5004    format(1x,/4x,'m',5x,'depth',7x,'disp',7x,'stress',8x,
     *  'dc/db',8x,'dc/dr',10x,'g')
5005    format(2x,i3,f10.2,5e13.4)
5006    format(1x,/1x,'t=',f6.2,2x,'c=',f6.4,2x,'cvar=',f6.4,2x,'ugr=',
     *  f6.4,2x,'wvnumb=',e11.4,2x,'ar=',e11.4,/36x,'ratio =',e10.4)
5007    format(1x,'I0 =',e10.4,2x,'I1 =',e10.4,2x,'I2 =',e10.4,
     *  2x,'I3 =',e10.4,2x,'L =',e10.4)
5008    format(1x,/4x,'m',5x,'depth',7x,'dispr',8x,'dispz',
     *  7x,'stresr',7x,'stresz',8x,'dwx')
5009    format(1x,/4x,'m',5x,'depth',7x,'dcda',9x,'dcdb',9x,
     *  'dcdr',10x,'g1',11x,'g2')
5010    format(1x,/10x,
     *  'Eigenfunction V (n=0) and its derivatives with respect to z',
     *  /25x,'(n - the order of derivative)',//2x,'depth\\n',
     *  6x,'0',11x,'1',11x,'2',11x,'3',11x,'4',11x,'5')
5020    format(1x,/10x,
     *  'Eigenfunction Vr (n=0) and its derivatives with respect to z',
     *  /25x,'(n - the order of derivative)',//2x,'depth\\n',
     *  6x,'0',11x,'1',11x,'2',11x,'3',11x,'4',11x,'5')
5030    format(1x,/10x,
     *  'Eigenfunction Vz (n=0) and its derivatives with respect to z',
     *  /25x,'(n - the order of derivative)',//2x,'depth\\n',
     *  6x,'0',11x,'1',11x,'2',11x,'3',11x,'4',11x,'5')
5011    format(1x,f7.2,6e12.4)
3007    format(1x,/2x,'t / mode',3x,'1',6x,'2',6x,'3',6x,'4',6x,'5',
     *  6x,'6',6x,'7',6x,'8',6x,'9',6x,'10',/)
3008    format(1x,f6.2,3x,10f7.3)
3009    format(1x,/11x,'Mode ',i2,',  period',f6.2,
     *  'sec.  Subroutine Nevill. Too many cycles.')


C--------------INITIATION-------------------------------------------
C       KEY_ATTEN=.false.
c       print *,'calcul A : kind = ',kind

c	 print *, 'd_ref',d_ref(1)
        cc=c1
        ifunc=kind                  
                                        continue
        c1=cc
c
c
        kmode=mode
                                do 2 i=1,nmod
2       imax(i)=0
c        print *,'ncount,dx,imax,idispr,idispl,number,numbel,kind,t_base:',ncount,dx,imax,idispr,idispl,number,numbel,kind,t_base
c        print *,'idispr:', idispr
c        print *,'idispl:', idispl
c        print *,'number:', number
c        print *,'numbel:', numbel
c        print *,'values to calcul.f'

C---------------start of loop for Period k-----------S
c	write(*,*) "kmax",kmax
                                do 9998 k=1,kmax
c        print *,'calcul C : kind = ',kind
        t1=t(k)
c        write(*,*) 'calcul C : kind = ',kind, t1,t(k),k
c        stop
C----------------current model for a given t1---------------------
c	mmax = n_layer
c	write(*,*) "check mmax",mmax
	do i=1,mmax
 	  b(i)=b_ref(i)
c	  write(*,*) "i and bi",i,b(i)
C	  a(i)=b_ref(i)*dpuass
 	  a(i)=a_ref(i)
	  rho(i)=rho_ref(i)
	  d(i)=d_ref(i)
c	  write(*,*) "check",i,d(i),d_ref(i), a_ref(i),a(i),b_ref(i),b(i),t1

 	  if (KEY_ATTEN) then
	    qsq=qs_ref(i)*alog(t_base/t1)/pi
	    qpq=qsq*1.33333333*b_ref(i)**2/a_ref(i)**2
	    
  	    b(i)=b_ref(i)*(1.+qsq)
  	    a(i)=a_ref(i)*(1.+qpq)

c	    write(*,*) "check",i,d(i),a_ref(i),a(i),b_ref(i),b(i),qsq,qpq,t1,pi,t_base;
c	    write(*,*) "new qsq i bi: ",qsq,i,b(i)
 	  endif
	enddo
c	write(*,*) "cvcvcv!!!!d1d1:::" ,d(1),a(1),b(1)
        call flat1(d,rho,a,b,mmax,kind)
c	write(*,*) "cvcvcv!!!!!" ,b(1)
C------------------------loop for modes-----S
c	write (*,*) "kmode k iq",kmode,k,iq
        do 9997 iq=1,kmode
          if(k-1) 605,605,599
599       if(iq-2) 600,601,601



600       c1=0.90*c(k-1,1)
            goto 605
601       if(dc*(c(k-1,iq)-c(k,iq-1))) 602,603,604
602       c1=c(k,iq-1)+0.01*dc
            goto 605
603       c1=c(k,iq-1)+0.01*dc
            goto 605
604       c1=c(k-1,iq)
605       continue
c	  print *, '605 active'
c	  print *, 'c1= ',c1
c	  print *, 'the b1= ',b(1)
            idrop=0
999       del1=dltar(c1,t1,ifunc)
80        c2=c1+dc
          idrop=0
          del2=dltar(c2,t1,ifunc)
          if(sign(1.,del1).ne.sign(1.,del2))  goto 54
81        c1=c2
          del1=del2
          qbq=0.8*b(1)
c	  write(*,*) "c1 and b1:", c1,b(1)
          if(c1-0.8*b(1)) 250,251,251
251       if(c1-(b(mmax)+0.3)) 252,250,250
252         goto 80
54          continue         


c	  write(*,*) "before nevill: c1,c2 ",c1,c2
          call nevill (t1,c1,c2,del1,del2,ifunc,cn)
            if(lstop.eq.0)                  go to 3006
            lstop=0
            if(k.eq.1.and.iq.eq.1)          go to 3005
            if(k.eq.1)                      go to 3004
            do 3003 i=1,k-1
              do 3000 j=1,mode
                if(i.gt.imax(j))                go to 3001
3000            continue
                jmax=mode
                go to 3002
3001    jmax=j-1
3002    continue
3003    continue
3004    if(iq.eq.1)                     go to 3005
          jmax=iq-1
3005    continue 
          go to 9999
3006    c1=cn
           if(c1-b(mmax)) 121,121,250
121     c(k,iq)=c1
           goto (6001,6002),ifunc
6002    continue
           ratio(k,iq)=dltar(c1,t1,3)
           goto 6003
6001    continue
6003    continue
        c1=c1+0.01*dc
        imax(iq)=k
           goto 9997

250     if(k*iq-1) 256,256,255
256        continue 

c        write(*,*) 'cccc', ncount LF

        if(ncount.eq.1)then
          PRINT 258
258     format(1x,/22x,'Improper initial value. No zero found')
        endif
                                                        goto 9999
9997    continue

C------loop for mode---------------------E
C--------------------search of root is finished--------------------
                                                        goto 9998
255     kmode=iq-1
                        if(kmode) 9996,9996,9998
9998                                    continue

C-----------------end of loop for k------------------E
9996    continue
        mmax=nmax
        do 9995 iq=1,mode
          j=imax(iq)
c	  write(*,*) imax(iq),iq
          if(j) 9995,9995,9994
9994    goto (7001,7002),ifunc

C------------------LOVE     WAVE PART------------------------------S
7001    continue
        number=-1
C       if(ncount.eq.1)write(2,1004)iq
C	print *,'MODE=',iq-1,';  NUMBER OF PERIODS IN OUTPUT=',j
c        n_R_out=j
c	write(*,*) "n_R_out",n_R_out
                                do 8889 lip=1,j
	if(KEY_ATTEN)then
	  do i=1,mmax
	    qsq=qs_ref(i)*alog(t_base/t(lip))/pi
	    qpq=qsq*1.33333333*b_ref(i)**2/a_ref(i)**2
	    b(i)=b_ref(i)*(1.+qsq)
	    a(i)=a_ref(i)*(1.+qpq)
	    d(i)=d_ref(i)
	    rho(i)=rho_ref(i)
	    qs(i)=qs_ref(i)
	  enddo
	endif

        call flat1(d,rho,a,b,mmax,kind)
        call leigen(numbel,lip,iq)

        numbel=numbel+1

C------------ATTENUATION OUTPUT---------------S
        qL_app=10000.
	if(KEY_ATTEN)          then
	  skd=0.0
	  do i=1,mmm
	    skd=skd+dcdb(i+1)*b(i)*qs(i)
	  enddo
	  alphL=pi/t(lip)*skd/ccc/ccc
	  qL_app= pi/alphL/ugr/t(lip)
        endif

	if(ncount.eq.1)then
          per_L(lip)=t(lip)
          uL(lip,iq)=ugr
          cL(lip,iq)=ccc
	  ugr0(lip)=ugr
	  ccc0(lip)=ccc
	endif

        if(ncount.eq.2)then
	  ugr_minus(lip)=ugr
	  ccc_minus(lip)=ccc
        endif

	if(ncount.eq.3)then
	  ugr_plus(lip)=ugr
	  ccc_plus(lip)=ccc
        endif

        if(iedit.eq.0)                  go to 8889
        if(iedit.eq.2)                  go to 8890
                                                        go to 8889
8890    if(ncount.eq.1.and.KEY_EIGEN) then
        if(KEY_EIG_NORM) then
	  bmax=0.0
	  do i=1,nmax
            if(abs(amprfi(i)).gt.bmax)bmax=abs(amprfi(i))
	  enddo
	  do i=1,nmax
	    amprfi(i)=amprfi(i)/bmax
	  enddo
        endif

        write (27,*)amprfi(1), depth(1),' F(Hz)=',1000./t(lip), 'mode=',
     *iq      
        do  m=2,nmax 
          if(amprfi(m).eq.0.0)go to 3333
          write (27,*)amprfi(m),  depth(m)
        enddo         

3333    write (27,'(/)'  )

        if(ncount.eq.1.and.KEY_EIGEN_DER1) then
	  write (31,*)'LOVE_H',' F(Hz)=',1000./t(lip)
          do  m=1,ndpth
C-------LOVE WAVE HORIZONTAL EIGENFUNCTION AND ITS DEPTH DERIVATIVES;
            write(31,*  )dpth(m),dsprfi(m),(drvrfi(lm,m),lm=1,nderiv)
          enddo
        endif

        endif
8889    continue

C------------------LOVE     WAVE PART------------------------------E
                                                        go to 3335
C------------------RAYLEIGH WAVE PART------------------------------S
7002               continue                              
c	write(*,*) "j = ",j
        do 8888 lip=1,j
	  if(KEY_ATTEN)then
	    do i=1,mmax
	      qsq=qs_ref(i)*alog(t_base/t(lip))/pi
	      qpq=qsq*1.33333333*b_ref(i)**2/a_ref(i)**2
	      b(i)=b_ref(i)*(1.+qsq)
	      a(i)=a_ref(i)*(1.+qpq)
	      d(i)=d_ref(i)
	      rho(i)=rho_ref(i)
	      qs(i)=qs_ref(i)
	    enddo
	  endif

          call flat1(d,rho,a,b,mmax,kind)
          call reigen(number,lip,iq)
          if(number.eq.-1) number=0
          number=number+1
          qR_app=10000.
  	  if(KEY_ATTEN) then
            skd=0.0
	    do i=1,mmm
	      skd=skd+dwx(i+1)*qs(i)
	    enddo
	    alphR=pi/t(lip)*skd/ccc/ccc
	    qR_app= pi/alphR/ugr/t(lip)
          endif
	  if(ncount.eq.1)then
            per_R(lip)=t(lip)
            uR(lip,iq)=ugr
c            print *,"cv check here!!" ,per_R(lip),uR(lip,iq),ccc
            cR(lip,iq)=ccc
	    ugr0(lip)=ugr
	    ccc0(lip)=ccc
	  end if
	  if(ncount.eq.2)then 
	    ugr_minus(lip)=ugr
	    ccc_minus(lip)=ccc
	  endif
          if(ncount.eq.3)then
	    ugr_plus(lip)=ugr
	    ccc_plus(lip)=ccc
	  endif
          if(iedit.eq.0)                  go to 8888
          if(iedit.eq.2)                  go to 8891
                                          go to 8888
8891    if(ncount.eq.1.and.KEY_EIGEN) then
	mult(lip)=wvno
	dampl(lip)=arle
        if(KEY_EIG_NORM) then
	  bmax=0.0
	  cmax=0.0
	  do i=1,nmax
            if(abs(amprfi(i)).gt.bmax)bmax=abs(amprfi(i))
            if(abs(ampz(i)).gt.cmax)cmax=abs(ampz(i))
	  enddo
	  do i=1,nmax
	    amprfi(i)=amprfi(i)/bmax
	    ampz(i)=ampz(i)/cmax
	  enddo
        endif
        do  m=2,nmax 
	  if(amprfi(m).eq.0.0.or.ampz(m).eq.0.0) go to 3334
        enddo
3334    continue        
        endif
        if(ncount.eq.1.and.KEY_EIGEN_DER1) then
	  write (32,*)'RAYL_V',' F(Hz)=',1000./t(lip)
	  write (31,*)'RAYL_H',' F(Hz)=',1000./t(lip)
          do  m=1,ndpth

C-------RAYLEIGH WAVE VERTICAL EIGENFUNCTION AND ITS DEPTH DERIVATIVES;
C	    write(32,*)dpth(m),dspz(m),(drvz(lm,m),lm=1,nderiv)
C------VERTICAL COMPONENT OF AMPLITUDE SPECTRA FOR EXPLOSION SOURCES:	 
            amp_exp(lip)=dampl(lip)*(-mult(lip)*dsprfi(m)+
     *drvz(1,m))*aM/sqrt(mult(lip))
            if(m.le.nwmax)amp_exp(lip)=dampl(lip)*dspz(m)*
     *aM/sqrt(mult(lip))
C-------RAYLEIGH WAVE HORIZONTAL EIGENFUNCTION AND ITS DEPTH DERIVATIVES;
          enddo
        endif
8888    continue
C------------------RAYLEIGH WAVE PART------------------------------E
3335    continue
9995	continue
9999    continue
        if(ncount.lt.3.and.KEY_DERIV)      return1         
777     if(KEY_DERIV) then
	  call qderiv(ugr_minus,ugr0,ugr_plus,t,kmax,dx,25)
          write(25,'(/)')
          write(26,'(/)')
	  call qderiv(ccc_minus,ccc0,ccc_plus,t,kmax,dx,23)
          write(23,'(/)')
          write(24,'(/)')
	  return 
        endif

        END


C--------------------------------------------------------------------
	subroutine qderiv(u1,u2,u3,t,nper,dx,nfile)
C--------------------------------------------------------------------
	real*4 u1(2),u2(2),u3(2),t(2)
	do i=1,nper
	  du1=(u3(i)-u1(i))/2./dx
	  du2=(u1(i)+u3(i)-2.*u2(i))/dx**2
	  write(nfile,*)1000./t(i),du1
	  write(nfile+1,*)1000./t(i),du2
	enddo

	return
        end
