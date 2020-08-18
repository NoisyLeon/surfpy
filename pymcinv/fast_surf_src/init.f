C--------------------------------------------------------------------
      subroutine init(dx,nlay_deriv,idispr,idispl,cvper,ncvper,
     * k_max,key_R,key_L) 
C--------------------------------------------------------------------

C---------to initialize the program--------------------------
        parameter (nsize=1000,nper=2000,ndep=100,nmod=20)
	common/d/  a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
        common/c/nmax,mmax,kmax,idrop,iedit,ndiv,mode,fact,ra_1
        common/o/  c(nper,nmod),t(nper),ratio(nper,nmod)
        common/newly/ t1,dt,c1,dc,lstop,iq,istru,cinit
        common/derivd/ nderiv,ndpth,dpth(ndep),dspz(ndep),dsprfi(ndep),
     *  drvz(5,ndep),drvrfi(5,ndep)
        common/log/ KEY_ATTEN,KEY_DERIV,KEY_EIGEN,KEY_EIG_NORM,KEY_EIGEN
     *_DER1,
     *  KEY_EIGEN_DER2,KEY_VP,KEY_VS,KEY_TH,KEY_RHO,KEY_CINIT,KEY_FLAT

	real*4 cvper(nper)
	integer ncvper

        logical KEY_DERIV,KEY_RHO,KEY_TH,KEY_VP,KEY_VS,KEY_ATTEN
	logical KEY_EIGEN,KEY_EIG_NORM,KEY_EIGEN_DER1,KEY_EIGEN_DER2,KEY_CINIT
        logical key_R,key_L, KEY_FLAT

 	data ndiv/5/,fact/4.0/,istru/0/,dc/0.01/,iedit/2/,nderiv/1/

c       fact - number of wavelenghts below first layer where phase
c       velocity is less than shear velocity that we may consider
c       to be effective halfspace.

C--------------INITIATION------------------------------------------S
        KEY_EIGEN=     .FALSE.
        KEY_EIG_NORM=  .FALSE.
        KEY_EIGEN_DER1=.FALSE.
        KEY_EIGEN_DER2=.FALSE.
	KEY_DERIV=     .FALSE.
        KEY_RHO=       .FALSE.
        KEY_VP=        .FALSE.
        KEY_VS=        .FALSE.
        KEY_TH=        .FALSE.
c       KEY_ATTEN set to false until the phase velocity modification is corrected
c       KEY_ATTEN=     .FALSE.
        KEY_ATTEN=     .TRUE.
        KEY_CINIT=     .FALSE.
        KEY_FLAT=      .TRUE.
        idispr=0 
        idispl=0 
        if(key_R)idispr=1
        if(key_L)idispl=1


        ndpth=0
99      mmax=i-1
        close(1)

C-------reading of the input model into "ref_files"-----E
C       mode=k_max-k_min+1
        mode=1

c-------to define an array of periods--------------------------------S
c        kmax=int((per_max-per_min)/per_step)+1
	kmax = ncvper
	if(kmax.gt.nper) then
	  PRINT*,'Too many periods=',kmax,'>',nper
	  kmax=nper
 	endif
        t(1)=per_min
        do i=1,kmax
c	  t(i)=t(1)+per_step*(i-1)
	  t(i)=cvper(i)
c          write(*,*) t1,t(i),i
        enddo
        t1=t(1)

c-------to define an array of periods--------------------------------E
        return
	end
