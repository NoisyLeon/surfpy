c-------------------------------------------------------
        subroutine FAST_SURF(n_layer0,kind0,
     &		a_ref0,b_ref0,rho_ref0,d_ref0,qs_ref0,
     &		cvper, ncvper,
     &		uR0,uL0,cR0,cL0)
C-------------------------------------------------------
c       parameter (nsize=1000,nper=200,ndep=20,nmod=1)
c       parameter (nsize=1000,nper=200,ndep=20,nmod=20)
        parameter (nsize=1000,nper=200,ndep=100,nmod=20)
        parameter (laymaxB=1000)
        parameter (maxlayer=15, maxsublayer=100, max_nd=maxlayer*4, maxm
     &oddim=max_nd, maxdata=1024, maxwave = 10)


        real*4 k_max,per_min,per_max,per_step
        real*8 dcda(nsize),dcdb(nsize),dcdr(nsize),dwx(nsize)
        logical KEY_DERIV,KEY_RHO,KEY_TH,KEY_VP,KEY_VS,KEY_ATTEN,KEY_FLA
     &T
	logical KEY_EIGEN,KEY_EIG_NORM,KEY_EIGEN_DER1,KEY_EIGEN_DER2,KEY_CINIT
        integer imax(nmod)
	integer ncvper
        
        logical         key_RC, key_RU, key_LC, key_LU, key_R, key_L
        logical         key_water
        real*4          h(maxsublayer), beta(maxsublayer), vpvs(maxsubla
     &yer), qa(maxsublayer), qb(maxsublayer)
c       real*4          rho1(maxlayer)
        real*4          rho_layer(100), rho1(maxlayer)
        real*4          observed_data(maxdata,maxwave), predicted_data(m
     &axdata,maxwave),
     &  constant_a(maxwave), constant_c(maxwave), weight(maxwave), q_alp
     &ha(maxlayer), q_beta(maxlayer), rho_val(maxlaye
     &	r)
c       real*4          q_alpha(maxlayer), q_beta(maxlayer)
        real*4          th_cr(maxlayer), vel_cr(maxlayer), vpvs_cr(maxla
     &yer), bspl_val(maxlayer), rho_in(maxlayer)
        real*4          crth, zmin, zmax, zmax_ref
c        real*4 a_ref(nsize),b_ref(nsize),rho_ref(nsize),
c     *d_ref(nsize),qs_ref(nsize)
        real*4          depth_tmp
        integer         ndata(maxwave), attval
        character*40    chars, kname, fname(maxwave)
	real*4 	uL0(nper), uR0(nper), cL0(nper), cR0(nper)
	real*4	a_ref0(n_layer0), b_ref0(n_layer0), rho_ref0(n_layer0), 
     *d_ref0(n_layer0), qs_ref0(n_layer0)
	real*4		cvper(nper)

	common/d/  a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
	common/numbers/n_int_RC,n_int_RU,n_int_LC,n_int_LU,n_layer,n_var
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
     &_DER1,
     *  KEY_EIGEN_DER2,KEY_VP,KEY_VS,KEY_TH,KEY_RHO,KEY_CINIT,KEY_FLAT
        common/ref/a_ref(nsize),b_ref(nsize),rho_ref(nsize),d_ref(nsize)
     &,qs_ref(nsize)
        common/surfr/k_max,per_min,per_max,per_step
        common/dispe/per_R(200),per_L(200),uR(200,2),uL(200,2),cR(200,2)
     &,cL(200,2)
        common /surf_com/observed_data, predicted_data,
     &                  constant_a, constant_c, time_shift, weight,
     &                  time_begin, time_end, q_alpha, q_beta, ndata,
     &                  nwave, fs, lu_mod, fname, rho_val

c       fact - number of wavelenghts below first layer where phase
c       velocity is less than shear velocity that we may consider
c       to be effective halfspace.

	data pi/3.1415927/,t_base/1.0/,ncount/0/,k_min/0/

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c        write(*,*) "begin here!!!!"

	n_layer = n_layer0
	kind = kind0

c        write(*,*) "begin: n_layer!!",n_layer
c        write(*,*) "kind0: ", kind0, b_ref(1), a_ref(1)
        

	do i=1,n_layer
c	   write (*,*) a_ref0(i), b_ref0(i), d_ref0(i),n_layer
	   a_ref(i) = a_ref0(i)
	   b_ref(i) = b_ref0(i)
	   rho_ref(i) = rho_ref0(i)
	   qs_ref(i) = qs_ref0(i)
	   d_ref(i) = d_ref0(i)
c           write (*,*) 'in 1', a_ref(i), b_ref(i), rho_ref(i), qs_ref(i) LF
c     c , d_ref(i) LF 
c           write(*,*) n_layer
	enddo

c	per_max = per_max0
c        per_min = per_min0
c        per_step = per_step0

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c        write(*,*) "begin to call!!"
c	dx = 2.01799774
	call INIT(dx,nlay_deriv,idispr,idispl,cvper,ncvper,k_max,
     *key_R,key_L)
	ndiv_store=ndiv
	t1_store=t1
        mmax=n_layer
c       print *,'mmax/n_layer:', mmax,n_layer
	mmax_store=mmax

c       kmax - number of periods; t1 - initial starting period;
c       dt - period increment; c1 - initial phase velocity guess;
c       dc - phase velocity increment; istru<1 - depth dependent
c       parameters will be calculated in midpoints of sublayers,
c       else program STRUT will be called.
c       nderiv - highest order of derivative of eigenfunction;
c       ndpth - number of depths for which eigenfunction and its
c       derivatives are desired.
c       dpth(j) - array of depths for calculation of eigenfunction
c       and its derivatives.

        ncount=1
	ndiv=ndiv_store
	mmax=mmax_store
	t1=t1_store
        do i=1,nsize
	depth(i)=0.0
	enddo
	do i=1,nper
	  do j=1,nmod 
	    c(i,j)=0.0
	    ratio(i,j)=0.0
	  enddo
	enddo




c       ndiv - number of subdivisions for one layer;
c       mode - number of modes for which dispersion curves are desired;
c !!!       iedit = 0 - only dispersion, iedit = 1 - dispersion, amplitudes
c       of eigenfunctions and derivatives of phase velocity,

        nmax=mmax
        number=0
        numbel=0
	lstop=0
	idrop=0
	iq=0
	imax(1)=0
	ilay=1
	if(b_ref(1).lt.0.1) ilay=2
	b_corr=0.0
c	print *,'ilay=',ilay
c	print *,'qs_ref(ilay),t_base,t1:',qs_ref(ilay),t_base,t1
 	if(KEY_ATTEN) b_corr=qs_ref(ilay)*alog(t_base/t1)/pi
c	if(KEY_ATTEN) b_corr=(1/qs_ref(ilay))*alog(t_base/t1)/pi
c	print *,'b_corr=',b_corr
	qq=b_ref(ilay)
c	print *,'b_ref(ilay)=',b_ref(ilay)
	if(kind.eq.2)qq=0.9*qq
	if(KEY_CINIT)qq=cinit
    	c1=qq*(1.+b_corr)
c        write(*,*) 'b_ref1',b_ref(1) LF
	if (b_ref(1).lt.0.1) c1 = 0.5
c   	print *, 'qref=',qq

c       print *, 'c1= ',c1
c        print * ,'call with model:'
c        write(*,*) "nlayer", n_layer
c        do iv=1,n_layer
c          print *, 'a_ref(iv),b_ref(iv), rho_ref(iv),d_ref(iv),qs_ref(iv):',a_ref(iv),b_ref(iv), rho_ref(iv),d_ref(iv),qs_ref(iv),' iv=',iv
c          print *, a_ref(iv),b_ref(iv), rho_ref(iv),d_ref(iv),qs_ref(iv),' iv=',iv
c        enddo
c	ncount = 0
c	write(*,*) "ncount!!! ",ncount
c	write(*,*) "d_ref in fast::",d_ref(1)

c	do i = 1,nper
c	  do j = 1,nmode
c	     write(*,*) "check_ccc ",c(i,j)
c	  enddo
c	enddo
c	write(*,*) "now call calcul"
        call calcul(ncount,dx,imax,idispr,idispl,numbel,number,kind,t_ba
     &se)
c        write(*,*) "now finish calcul"
        if(kind.eq.2) n_R_out=kmax
        if(kind.eq.1) n_L_out=kmax

	do i=1,kmax
	  if(kind.eq.1) then 
	    cL0(i) = cL(i,1)
            uL0(i) = uL(i,1)
c	    write(*,*) "cL0: ", cL0(i)
          endif
          if (kind.eq.2) then
            cR0(i) = cR(i,1)     
	    uR0(i) = uR(i,1)
c            write(*,*) "cR0: ", cR0(i),uR0(i)
	  endif
	enddo

        RETURN
        END
