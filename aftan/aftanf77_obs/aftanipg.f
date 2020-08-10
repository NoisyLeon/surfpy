c
c aftanipg function. Provides ftan analysis with phase match filter,
c jumps correction, phase velocity computation and amplitude map 
c output for input periods.
c
c
c Autor: M. Barmine,CIEI,CU. Date: Jun 15, 2006. Version: 2.00
c
      subroutine aftanipg(piover4,n,sei,t0,dt,delta,vmin,vmax,tmin,tmax,
     *           tresh,ffact,perc,npoints,taperl,nfin,fsnr,fmatch,npred,pred,
     *           cuttype,nphpr,phprper,phprvel,seiout,
     *           nfout1,arr1,nfout2,arr2,tamp,nrow,ncol,amp,ierr);
c======================================================================
c Parameters for aftanipg function:
c Input parameters:
c piover4 - phase shift = pi/4*piover4, for cross-correlation
c           piover4 should be   -1.0 !!!!     (real*8)
c n       - number of input samples, (integer*4)
c sei     - input array length of n, (real*4)
c t0      - time shift of SAC file in seconds, (real*8)
c dt      - sampling rate in seconds, (real*8)
c delta   - distance, km (real*8)
c vmin    - minimal group velocity, km/s (real*8)
c vmax    - maximal value of the group velocity, km/s (real*8)
c tmin    - minimal period, s (real*8)
c tmax    - maximal period, s (real*8)
c tresh   - treshold, usualy = 10, (real*8)
c ffact   - factor to automatic filter parameter, usualy =1, (real*8)
c perc    - minimal length of of output segment vs freq. range, % (real*8)
c npoints - max number points in jump, (integer*4)
c taperl  - factor for the left end seismogram tapering,
c           taper = taperl*tmax,    (real*8)
c nfin    - starting number of frequencies, nfin <= 32,(integer*4)
c fsnr    - phase match filter parameter, spectra ratio to 
c           determine cutting point   (real*8)
c fmatch  - factor to length of phase matching window (real*8)
c npred   - length of the group velocity prediction table
c pred    - group velocity prediction table:    (real*8)
c                             pred[0][] - periods in sec,
c                             pred[1][] - pedicted velocity, km/s
c nphpr   - length of the phase velocity prediction table, =0 - no table
c phprper - predicted phase velocity periods in sec (real*8 phprper(nphpr))
c phprvel - predicted phase velocity, km/s (real*8 phprvel(nphpr))
c ==========================================================
c Output parameters are placed in 2-D arrays arr1 and arr2,
c arr1 contains preliminary results and arr2 - final.
c ==========================================================
c nfout1 - output number of frequencies for arr1, (integer*4)
c arr1   - the first nfout1 raws contain preliminary data,
c          (real*8 arr1 (5 x n), n >= nfin)
c          arr1(1,:) -  central periods, s (real*8)
c          arr1(2,:) -  apparent periods, s (real*8)
c          arr1(3,:) -  group velocities, km/s (real*8)
c          arr1(4,:) -  phase velocities, km/s (real*8)
c          arr1(5,:) -  amplitudes, Db (real*8)
c          arr1(6,:) -  discrimination function, (real*8)
c          arr1(7,:) -  signal/noise ratio, Db (real*8)
c          arr1(8,:) -  maximum half width, s (real*8)
c nfout2 - output number of frequencies for arr2, (integer*4)
c          If nfout2 == 0, no final result.
c arr2   - the first nfout2 raws contains final data,
c          (real*8 arr2 (4 x n), n >= nfin)
c          arr2(1,:) -  central periods, s (real*8)
c          arr2(2,:) -  apparent periods, s (real*8)
c          arr2(3,:) -  group velocities, km/s (real*8)
c          arr1(4,:) -  phase velocities, km/s (real*8)
c          arr2(5,:) -  amplitudes, Db (real*8)
c          arr2(6,:) -  signal/noise ratio, Db (real*8)
c          arr2(7,:) -  maximum half width, s (real*8)
c          tamp      -  time to the beginning of ampo table, s (real*8)
c          nrow      -  number of rows in array ampo, (integer*4)
c          ncol      -  number of columns in array ampo, (integer*4)
c          amp       -  Ftan amplitude array, Db, (real*8)
c ierr   - completion status, =0 - O.K.,           (integer*4)
c                             =1 - some problems occures
c                             =2 - no final results
c======================================================================
      implicit none
      include 'myfftw3.h'
      integer*4 n,npoints,nf,nfin,nfout1,ierr,nrow,ncol,ntmp,cuttype
      real*8    piover4,perc,taperl,tamp,arr1(8,100),arr2(7,100)
      real*8    t0,dt,delta,vmin,vmax,tmin,tmax,tresh,ffact,ftrig(100),tfact
      real*8    fsnr,fmatch
      real*4    sei(32768), seiout(32768)
      double complex dczero,s(32768),sf(32768),fils(32768),tmp(32768)
      real*8    grvel(100),tvis(100),ampgr(100),om(100),per(100),tim(100)
      real*8    grveltmp,tvistmp,ampgrtmp,omtmp,pertmp,timtmp,snrtmp,wdthtmp,phgrtmp
      real*8    pha(32768,32),amp(32768,32),ampo(32768,32)
      real*8    time(32768),v(32768),b(32768)
      real*8    alpha,alphad,pi,omb,ome,omcur,dom,step,amax,t,dph,tm,ph
      integer*4 j,k,k2,m,ntapb,ntape,ne,nb,ntime,ns,ntall,ici,iciflag,ia
      real*8    plan1,plan2,plan3,plan4
      integer*4 ind(2,32768)
      real*8    ipar(6,32768)
      real*8    grvel1(100),tvis1(100),ampgr1(100),ftrig1(100)
      real*8    trig1(100),grvelt(100),tvist(100),ampgrt(100)
      real*8    phgr(100),phgr1(100),phgrt(100),phgrc(100)
c ---
      integer*4 njump,nijmp,i,ii(100),nii,ijmp(100),ki,kk,istrt,ibeg,iend,ima
      real*8    dmaxt,wor,per2(100),om1(100),snr(100),snr1(100),snrt(100)
      real*8    wdth(100),wdth1(100),wdtht(100)
      integer*4 iflag,ierr1,nindx,imax,iimax,ipos,ist,ibe,nfout2,indx(100)
      integer*4 mm,mi,iml,imr,indl,indr,nphpr
      real*8    lm,rm,phprper(nphpr),phprvel(nphpr)
c ---
      integer*4 ip,npred,inds,inde
      real*8    om0,tg0,omstart,dw,pha_corr,ome1,omb1,maxTpr,minTpr
      real*8    pred(500,2),omdom(32768), ampdom(32768)
      double complex dci,dc2,pha_cor(32768),env(32768),spref(32768)

      ierr = 0
      dczero = dcmplx(0.0d0,0.0d0)
      dci    = dcmplx(0.0d0,1.0d0)
      dc2    = dcmplx(2.0d0,0.0d0)
      lm = 0.0d0
      rm = 0.0d0
      iml = 0
      imr = 0
      pi = datan(1.0d0)*4.0d0
c number of FTAN filters
      nf = nfin
c rescale ffact and tresh
c      ffact = ffact*2.0d0
c      tresh = dsqrt(ffact)*tresh
c automatic width of filters * factor ffact
c      alpha = ffact*20.0d0*dsqrt(delta/1000.0d0)
      alpha = ffact*20.0d0
c trigger factor
      tfact = 1.5d0
c  number of samples for tapering, left end
      ntapb = nint(taperl*tmax/dt)
c  number of samples for tapering, right end
      ntape = nint(tmax/dt)
c [omb,ome] - frequency range
      omb = 2.0d0*pi/tmax
      ome = 2.0d0*pi/tmin
c find min/max of prediction period
      maxTpr = pred(1,1)
      minTpr = pred(1,1)
      do i =2,npred
        if(pred(i,1).ge.maxTpr) maxTpr = pred(i,1)
        if(pred(i,1).le.minTpr) minTpr = pred(i,1)
      enddo
c evaluation of spline polinomial forms for phase match filter
      ip = 1
      call pred_cur(ip,delta,dsqrt(omb*ome),npred,pred,om0,tg0)
      write(*,*)'T0= ',2.0d0*pi/om0,', tg0= ',tg0
c seismgram tapering
      nb = max0(2,nint((delta/vmax-t0-50.0d0)/dt))
      tamp = (nb-1)*dt+t0;
      ne = min0(n,nint((delta/vmin-t0+50.0d0)/dt))
      nrow = nfin
      ncol = ne-nb+1
c times for FTAN map
      do k = nb,ne
        time(k) = (k-1)*dt
c velocity for FTAN map
        v(k) = delta/(time(k)+t0)
      enddo
      ntime = ne-nb+1
c tapering both ends of seismogram
      call taper(max0(nb,ntapb+1),min0(ne,n-ntape),n,sei,ntapb,ntape,s,ns);
c prepare FTAN filters
      dom = 2*pi/ns/dt
      step =(dlog(omb)-dlog(ome))/(nf -1)
c log scaling for frequency
      do k = 1,nf
        om(k) = dexp(log(ome)+(k-1)*step)
        per(k) = 2.0d0*pi/om(k)
      enddo
c==================================================================
c Phase match filtering
c make backward FFT for seismogram: s ==> sf
      call dfftw_plan_dft_1d(plan1,ns,s,sf,
     *                         FFTW_FORWARD, FFTW_ESTIMATE)
      call dfftw_execute(plan1)
      call dfftw_destroy_plan(plan1)
c filtering and FTAN amplitude diagram construction
      call dfftw_plan_dft_1d(plan2,ns,fils,tmp,
     *                         FFTW_BACKWARD, FFTW_ESTIMATE)
      sf(1) = sf(1)/2.0d0
      sf(ns/2+1) = dcmplx(dreal(sf(ns/2+1)),0.0d0)
c spectra tapering
      ome1 = min(ome,2.0d0*pi/minTpr)
      omb1 = max(omb,2.0d0*pi/maxTpr)
      call tapers(omb1,ome1,dom,alpha,ns,   omstart,inds,inde,omdom,ampdom)
      omstart = real(nint(omstart/dom))*dom
      inde = min0(inde,ns/2+2)
      do i = 1,ns
          if(i.lt.inds) then
              pha_cor(i) = dczero
          elseif(i.gt.inde) then
              pha_cor(i) = dczero
          else
              call msplint(ip+1,om0,omdom(i),pha_corr,ierr)
              pha_cor(i) = dcmplx(pha_corr,0.0d0)
          endif
          pha_cor(i) = cdexp(dci*pha_cor(i))
          sf(i) = sf(i)*pha_cor(i)*ampdom(i)
      enddo
c  forward FFT to get signal envelope
      call dfftw_plan_dft_1d(plan3,ns,sf,env,
     *                         FFTW_BACKWARD, FFTW_ESTIMATE)
      call dfftw_execute(plan3)
      call dfftw_destroy_plan(plan3)
      do i =1,ns
         env(i) = env(i)*dc2/ns
      enddo
c cutting impulse response in time
      dw = om(1)-om(nf)

      call tgauss(fsnr,tg0,t0,dw,dt,ns,fmatch,env,spref,cuttype)
c back to spectra after filtering
      call dfftw_plan_dft_1d(plan4,ns,spref,sf,
     *                         FFTW_FORWARD, FFTW_ESTIMATE)
      call dfftw_execute(plan4)
      call dfftw_destroy_plan(plan4)
c apply back phase correction for spectra
      do i = 1,ns
         sf(i) = sf(i)/pha_cor(i)
      enddo
c again forward FFT to get the phase-match-filtered signal in time domain
      call dfftw_plan_dft_1d(plan1,ns,sf,s,
     *                         FFTW_BACKWARD, FFTW_ESTIMATE)
      call dfftw_execute(plan1)
      call dfftw_destroy_plan(plan1)
      do i =1,n
         seiout(i) = 2.0*real(dreal(s(i)))/ns
      enddo

c==================================================================
c main loop by frequency
      do k = 1,nf
c filtering
        ntmp = 0
        do m = 2,npred-1
           if(2.0d0*pi/om(k).gt.pred(m,1)) ntmp = m
        enddo
        alphad = alpha * (0.5d0+2.0d0*dabs(pred(ntmp,2)-pred(ntmp+1,2)))
        call ftfilt(alphad,om(k),dom,ns,sf,fils, b)
c fill with zeros half spectra for Hilbert transformation and
c spectra ends ajastment
        do m = ns/2+2,ns
          fils(m) = dczero
        enddo
        fils(1) = dcmplx(dreal(fils(1))/2.0d0,0.0d0)
        fils(ns/2+1) = dcmplx(dreal(fils(ns/2+1)),0.0d0)
c forward FFT: fils ==> tmp
        call dfftw_execute(plan2)
        do m = 1,ns
          tmp(m) = tmp(m)/ns
        enddo
        j = 1
c extraction from FTAN map area of investigation
        do m = nb-1,ne+1
          pha(j,k) = datan2(dimag(tmp(m)),dreal(tmp(m)))
          wor = cdabs(tmp(m))
          ampo(j,k) = wor
          amp(j,k) = 20.0d0*dlog10(wor)
          j = j+1
        enddo
      enddo
      call dfftw_destroy_plan(plan2)
c normalization amp diagram to 100 Db with three decade cutting
      ntall = ntime+2
      amax = -1.0d10
      do j = 1,ntall
        do k = 1,nf
          if(amp(j,k).gt.amax) amax = amp(j,k)
        enddo
      enddo
      do j = 1,ntall
        do k = 1,nf
          amp(j,k) = amp(j,k)+100.0d0-amax
          if(amp(j,k).lt.40.0d0) amp(j,k) = 40.0d0
        enddo
      enddo
c construction reference indices table ind. It points to local maxima.
c table ipar contains three parameter for each local maximum:
c tim - group time; tvis - observed period; ampgr - amplitude values in Db
      ici = 0
      do k = 1,nf
c find local maxima on the FTAN amplitude diagram map
        iciflag = 0
        do j =2,ntall-1
          if(ampo(j,k).gt.ampo(j-1,k).and.ampo(j,k).gt.ampo(j+1,k)) then
            iciflag = iciflag+1
            ici = ici+1
            ind(1,ici) = k
            ind(2,ici) = j
          endif
        enddo
        if(iciflag.eq.0) then
          ici = ici+1
          ind(1,ici) = k
          ind(2,ici) = ntall-1
          if(ampo(2,k).gt.ampo(ntall-1,k)) ind(2,ici) = 2
          iciflag = 1
        endif
c compute parameters for each maximum
        amax = -1.0d10
        ia = 1
        do j = ici-iciflag+1,ici
          m= ind(2,j)
          call fmax(ampo(m-1,k),ampo(m,k),ampo(m+1,k),pha(m-1,k),
     *             pha(m,k),pha(m+1,k),om(k),dt,t,dph,tm,ph,piover4)
          ipar(1,j) = (nb+m-3+t)*dt
          ipar(2,j) = 2*pi*dt/dph
          ipar(3,j) = tm
          ipar(6,j) = ph
          if(tm.gt.amax) then
            amax = tm
            ia = j
          endif
        enddo
c Compute signal to noise ratio (SNR) ------------
        mm = 0
        do j = ici-iciflag+1,ici
          m = ind(2,j)
          mm = mm + 1
c find boundaries around local maximum
          if(mm.eq.1) then
            iml = 1
            imr = ntall
            if(iciflag.gt.1) imr = ind(2,j+1)
          elseif (mm.eq.iciflag) then
            iml = 1
            imr = ntall
            if(iciflag.gt.1) iml = ind(2,j-1)
          else
            iml = ind(2,j-1)
            imr = ind(2,j+1)
          endif
c      compute left minimum -------
            lm = ampo(m,k)
            indl = 1
            do mi = iml,m
              if(ampo(mi,k).le.lm) then
                lm = ampo(mi,k)
                indl = mi
              endif
            enddo
c      compute right minimum -------
            rm = ampo(m,k)
            indr = 1
            do mi = m,imr
              if(ampo(mi,k).le.rm) then
                rm = ampo(mi,k)
                indr = mi
              endif
            enddo
c          ipar(4,j) = 20.0d0*dlog10(ampo(m,k)/dsqrt(lm*rm))
          ipar(4,j) = ampo(m,k)/dsqrt(lm*rm)
          if(indl.eq.1.and.indr.eq.ntall) ipar(4,j) = ipar(4,j)+100.0d0
          ipar(5,j) = dt*(dabs(real(m-indl,8))+dabs(real(m-indr,8)))*0.5;
        enddo
c End of SNR computations
        tim(k)   = ipar(1,ia)
        tvis(k)  = ipar(2,ia)
        ampgr(k) = ipar(3,ia)
        grvel(k) = delta/(tim(k) +t0)
        snr(k)   = ipar(4,ia)
        wdth(k)  = ipar(5,ia)
        phgr(k)  = ipar(6,ia)
      enddo
      nfout1 = nf
c Sort the arrays by tvis
c      do k = 2,nf
c         write(*,*) k
c         omtmp = om(k)
c         timtmp = tim(k)
c         pertmp = per(k)
c         tvistmp = tvis(k)
c         ampgrtmp = ampgr(k)
c         grveltmp = grvel(k)
c         snrtmp = snr(k)
c         wdthtmp = wdth(k)
c         phgrtmp = phgr(k)
c         do k2 = k,2,-1
c            if(tvistmp.ge.tvis(k2-1)) exit
c            om(k2) = om(k2-1)
c            tim(k2) = tim(k2-1)
c            per(k2) = per(k2-1)
c            tvis(k2) = tvis(k2-1)
c            ampgr(k2) = ampgr(k2-1)
c            grvel(k2) = grvel(k2-1)
c            snr(k2) = snr(k2-1)
c            wdth(k2) = wdth(k2-1)
c            phgr(k2) = phgr(k2-1)
c         enddo
c         om(k2) = omtmp
c         tim(k2) = timtmp
c         per(k2) = pertmp
c         tvis(k2) = tvistmp
c         ampgr(k2) = ampgrtmp
c         grvel(k2) = grveltmp
c         snr(k2) = snrtmp
c         wdth(k2) = wdthtmp
c         phgr(k2) = phgrtmp
c      enddo
      k = 2
      do while (k.le.nf)
         if(tvis(k).lt.tvis(k-1)) then
            do k2 = k,nf-1
               om(k2) = om(k2+1)
               tim(k2) = tim(k2+1)
               per(k2) = per(k2+1)
               tvis(k2) = tvis(k2+1)
               ampgr(k2) = ampgr(k2+1)
               grvel(k2) = grvel(k2+1)
               snr(k2) = snr(k2+1)
               wdth(k2) = wdth(k2+1)
               phgr(k2) = phgr(k2+1)
            enddo
            nf = nf-1
         endif
         k = k+1
      enddo
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c       Check dispersion curve for jumps
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      call trigger(grvel,tvis,nf,tresh,trig1, ftrig,ierr)
      if(ierr.ne.0) then
        do k = 1,nf
          grvelt(k) = grvel(k)
          tvist(k)  = tvis(k)
          ampgrt(k) = ampgr(k)
          phgrt(k)  = phgr(k)
          wdtht(k)  = wdth(k)
        enddo
        njump = 0
c find all jumps
        nijmp = 0
        do i = 1,nf-1
          if(dabs(trig1(i+1)-trig1(i)).gt.1.5d0) then
            nijmp = nijmp+1
            ijmp(nijmp) = i
          endif
        enddo
c find correctable jumps
        nii = 0
        do i =1,nijmp-1
          if(ijmp(i+1)-ijmp(i).le.npoints) then
             nii = nii +1
             ii(nii) = i
          endif
        enddo
c main loop by jumps
        if(nii.ne.0) then
          do ki = 1,nii
             kk = ii(ki)
             do i = 1,nf
                grvel1(i) = grvelt(i)
                tvis1(i)  = tvist(i)
                ampgr1(i) = ampgrt(i)
                phgr1(i)  = phgrt(i)
                snr1(i)   = snrt(i)
                wdth1(i)  = wdth(i)
             enddo
             istrt = ijmp(kk)
             ibeg = istrt+1
             iend = ijmp(kk+1)
             ima = 0
             do k = ibeg,iend
               dmaxt = 1.0e10
               do j = 1,ici
                 if(ind(1,j).eq.k) then
                   wor = dabs(delta/(ipar(1,j)+t0)-grvel1(k-1))
                   if(wor.lt.dmaxt) then
                     ima = j
                     dmaxt = wor
                   endif
                 endif
               enddo
               grvel1(k) = delta/(ipar(1,ima)+t0);
               tvis1(k)  = ipar(2,ima);
               ampgr1(k) = ipar(3,ima)
               phgr1(k)  = ipar(6,ima)
               snr1(k)   = ipar(4,ima)
               wdth1(k)  = ipar(5,ima)
             enddo
             call trigger(grvel1,tvis1,nf,tresh,trig1, ftrig1,ierr1)
             iflag = 0
             do k=istrt,iend+1
                if(dabs(ftrig1(k)).ge.(tresh*tfact)) iflag = 1
             enddo
             if(iflag.eq.0) then
               do i =1,nf
                 grvelt(i) = grvel1(i)
                 tvist(i)  = tvis1(i)
                 ampgrt(i) = ampgr1(i)
                 phgrt(i)  = phgr1(i)
                 snrt(i)   = snr1(i)
                 wdtht(i)  = wdth1(i)
                 njump = njump+1
               enddo
             endif
          enddo
        endif
        do i=1,nf
          grvel1(i) = grvelt(i)
          tvis1(i)  = tvist(i)
          ampgr1(i) = ampgrt(i)
          phgr1(i)  = phgrt(i)
          snr1(i)   = snrt(i)
          wdth1(i)  = wdtht(i)
        enddo
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c after removing possible jumps, we cut frequency range to single
c segment with max length
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        call trigger(grvel1,tvis1,nf,tresh,trig1, ftrig1,ierr1)
        if(ierr1.ne.0) then
          nindx = 1
          indx(1) = 1
          do i =1,nf
            if(dabs(ftrig1(i)).ge.(tresh*tfact)) then
              write (*,*) "!!large curvature at",tvis1(i),"sec: ",ftrig1(i)
              nindx = nindx+1
              indx(nindx) = i
            endif
          enddo
          nindx = nindx+1
          indx(nindx) = nf
          imax = 0
          ipos = 0
          do i =1,nindx-1
            iimax = indx(i+1)-indx(i)
            if(iimax.gt.imax) then
               ipos = i
               imax = iimax
            endif
          enddo
          ist = max0(indx(ipos),1);
          ibe = min0(indx(ipos+1),nf);
          nfout2 = ibe -ist+1;
          do i = ist,ibe
            per2(i-ist+1)   = per(i)
            grvel1(i-ist+1) = grvel1(i)
            tvis1(i-ist+1)  = tvis1(i)
            ampgr1(i-ist+1) = ampgr1(i)
            phgr1(i-ist+1)  = phgr1(i)
            snr1(i-ist+1)   = snr1(i)
            wdth1(i-ist+1)  = wdth1(i)
            om1(i-ist+1)    = om(i)
          enddo
          call trigger(grvel1,tvis1,nfout2,tresh,trig1, ftrig1,ierr1)
c          if(nfout2 .lt. nf*perc/100.0d0) then
c          write(*,*) "In: ", tvis(1), "-", tvis(nf), " Out:", tvis1(1), "-", tvis1(nfout2)
          if((dlog(tvis1(nfout2))-dlog(tvis1(1))).lt.(dlog(tvis(nf))-dlog(tvis(1)))*perc/100.0d0) then
            ierr1 = 1
            nfout2 = 0
          endif
        else
            nfout2 = nf
            do i = 1,nf
                per2(i)   = per(i)
            enddo
        endif
      else
        ierr = 0
        nfout2 = nf
        do i = 1,nf
          per2(i)   = per(i)
          tvis1(i)  = tvis(i)
          ampgr1(i) = ampgr(i)
          phgr1(i)  = phgr(i)
          grvel1(i) = grvel(i)
          snr1(i)   = snr(i)
          wdth1(i)  = wdth(i)
        enddo
      endif
cxx   if(nfout2 .ne.0) then
cxx     call lim(nfout2,per2,grvel1,grvel1);
cxx   endif
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c fill out output data arrays
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if(nfout1.ne.0.and.nphpr.ne.0) then
        call phtovel(delta,ip,nfout1,tvis,grvel,phgr,nphpr,phprper,
     *               phprvel,phgrc)
        do j = 1,nfout1
           phgr(j) = phgrc(j)
        enddo
      endif
      do i = 1,nfout1
        arr1(1,i) = per(i)
        arr1(2,i) = tvis(i)
        arr1(3,i) = grvel(i)
        arr1(4,i) = phgr(i)
        arr1(5,i) = ampgr(i)
        arr1(6,i) = ftrig(i)
        arr1(7,i) = snr(i)
        arr1(8,i) = wdth(i)
      enddo
      if(nfout2.ne.0) then
        if(nphpr.ne.0) then
          call phtovel(delta,ip,nfout2,tvis1,grvel1,phgr1,nphpr,phprper,
     *                 phprvel,phgrc)
          do j = 1,nfout2
             phgr1(j) = phgrc(j)
          enddo
        endif
        do i = 1,nfout2
          arr2(1,i) = per2(i)
          arr2(2,i) = tvis1(i)
          arr2(3,i) = grvel1(i)
          arr2(4,i) = phgr1(i)
          arr2(5,i) = ampgr1(i)
          arr2(6,i) = snr1(i)
          arr2(7,i) = wdth1(i)
        enddo
      else
        ierr = 2
      endif
      return
      end
