czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        subroutine NEVILL (t,c1,c2,del1,del2,ifunc,cc)
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        parameter (nsize=1000)
        common/d/   a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
        common/c/nmax,mmax,kmax,idrop,iedit,ndiv,mode,fact,ra_1
        common/newly/t1,dt,cc1,dc,lstop,iqq,istru,cinit
        dimension x(20),y(20)
c--------------------------------------------------------------------
        accur1=0.1e-5
        accur2=0.1e-7
        ic=0
        c3=(c1+c2)/2.
        del3=dltar(c3,t,ifunc)
        nev=1
1310    ic=ic+1
             if(ic.lt.50)           goto 777
c------------------------------------------------------
c         TOO  MANY  CYCLES        ????? STOP ??????
c------------------------------------------------------
c                                                       STOP
        lstop=lstop+10
	PRINT*,'LSTOP=', lstop
	do i=1,7
	PRINT*,'a/b/rho/d/qs:',a(i),b(i),rho(i),d(i),qs(i)
	enddo
	PRINT*,' '
                   goto 2000
c------------------------------------------------------
c
777               continue
             if(c1-c3) 1320,1320,1330
1320         if(c2-c3) 1344,1344,1000
1330         if(c2-c3) 1000,1344,1344
1000    s13=del1-del3
        s32=del3-del2
             if(sign(1.,del3)*sign(1.,del1)) 1441,1441,1443
1441    c2=c3
        del2=del3
                  goto 1444
1443    c1=c3
        del1=del3
1444              continue
             if(abs(c1-c2)-accur1) 20,20,22
22                continue
             if(sign(1.,s13).ne.sign(1.,s32)) nev=0
        ss1=abs(del1)
        s1=0.1*ss1
        ss2=abs(del2)
        s2=0.1*ss2
             if(s1.gt.ss2.or.s2.gt.ss1)           goto 1344
             if(nev.eq.0)           goto 1344
             if(nev.eq.2)           goto 1350
        x(1)=c1
        y(1)=del1
        x(2)=c2
        y(2)=del2
        m=1
                  goto 1355
1344    c3=(c1+c2)/2.
        del3=dltar(c3,t,ifunc)
        nev=1
        m=1
                  goto 1310
1350    x(m+1)=c3
        y(m+1)=del3
1355              do 1360 kk=1,m
        j=m-kk+1
             if(abs(y(m+1)-y(j)).le.accur2)          goto 1344
        x(j)=(-y(j)*x(j+1)+y(m+1)*x(j))/(y(m+1)-y(j))
1360              continue
21                continue
        c3=x(1)
        del3=dltar(c3,t,ifunc)
        nev=2
        m=m+1
             if(m.gt.10) m=10
                  goto 1310
20                continue
        cc=c3
2000              continue
                                                        RETURN
                                                        END
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        function DLTAR (cc,tt,kk)
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
         parameter (nsize=1000,nper=2000,nmod=20)
        common/d/  a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
        common/o/  c(nper,nmod),t(nper),ratio(nper,nmod)
        common/c/nmax,mmax,kmax,idrop,iedit,ndiv,mode,fact,ra_1
c-----------------------------------------------------------------------
             if(idrop) 899,899,905
899               continue
        dmax=fact*cc*tt
        mmax=nmax
        sum=0
                  do 900 ii=1,nmax
             if(cc-b(ii)) 901,900,900
901     sum=sum+d(ii)
             if(sum-dmax) 900,900,902
902     mmax=ii
             goto 904
900               continue
904     idrop=1
             if(mmax.lt.2) mmax=2
905               continue
             goto (1,2,3,4,5),kk
c---------------------------------------------
c       LOVE  WAVE  PERIOD  equation
c------------------
1       dltar=dltar1(cc,tt,1)
                                                        RETURN
c---------------------------------------------
c       RAYLEIGH  WAVE  PERIOD  equation
c------------------
2       dltar=dltar4(cc,tt,1)
                                                        RETURN
c---------------------------------------------
c       RAYLEIGH  WAVE  ellipticity
c------------------
3       dltar=dltar4(cc,tt,2)
                                                        RETURN
c---------------------------------------------
c       RAYLEIGH  WAVE  amplitude  response  component
c------------------
4       dltar=dltar4(cc,tt,3)
                                                        RETURN
c---------------------------------------------
c       LOVE  WAVE  amplitude  response  component
c------------------
5       dltar=dltar1(cc,tt,2)
                                                        RETURN
                                                        END
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        function DLTAR1 (c,t,mup)
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
c-----------------------------------------------------------------------
c       Haskell-Thompson Love wave formulation from halfspace to surface
c-----------------------------------------------------------------------
         parameter (nsize=1000)
        common/d/  a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
        common/c/ nmax,mmax,kmax,idrop,iedit,ndiv,mode,fact,ra_1
        wvno=6.2831853/(c*t)
        covb=c/b(mmax)
        h=rho(mmax)*b(mmax)*b(mmax)
        rb=sqrt(abs(covb**2-1.))
        ut=1
        tt=h*rb
        mmm1=mmax-1
                                do 1340 k=1,mmm1
        m=mmax-k
                        if(b(m).eq.0.0)                 go to 1340
        covb=c/b(m)
        rb=sqrt(abs(covb**2-1.))
        h=rho(m)*b(m)*b(m)
7001    q=-wvno*d(m)*rb
                        if(rb.lt.0.1e-20)               go to 1221
                        if(c-b(m)) 1209,1221,1231
1231    sinq=sin(q)
        y=sinq/rb
        z=rb*sinq
        cosq=cos(q)
                                                        go to 1242
1221    y=-wvno*d(m)
        z=0
        cosq=1
                                                        go to 1242
1209    exqp=exp(q)
        exqm=1./exqp
        y=(exqp-exqm)/(2.*rb)
        z=-rb*rb*y
        cosq=(exqp+exqm)/2.
1242    eut=cosq*ut-y*tt/h
        ett=h*z*ut+cosq*tt
        ut=eut
        tt=ett
1340                                    continue
                                                        go to (1,2),mup
1       dltar1=-ett
                                                                RETURN
2       dltar1=ut
                                                                RETURN
                                                                END
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        function DLTAR4 (c,t,mup)
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
         parameter (nsize=1000)
        common/d/  p(nsize),s(nsize),rho(nsize),d(nsize),qs(nsize)
        common/c/nmax,mmax,kmax,idrop,iedit,ndiv,mode,fact,ra_1
c-------------------------------------------------------------------
        accur=1.e-8
        accurs=1.e-8
99      wvno=6.28318531/(c*t)
        csq=c*c
        jump=1
             if(mup.gt.1) jump=2
1000    b1=0.0
        b2=0.0
        b3=0.0
        b4=0.0
        b5=0.0
             if(jump-2) 1,2,3
1       b1=1.0
             goto 4
2       b2=1.0
             goto 4
3       b3=1.0
             goto 4
4                 continue
                  do 50 m=1,mmax
100     arga=1.-csq/p(m)**2
        ra=sqrt(abs(arga))
        if(m.eq.1)ra_1=ra
             if(arga.gt.0.) ra=-ra
             if(abs(s(m)).gt.accurs)      goto 101
c---------------------------------------------
c       LIQUID  SURFACE  LAYER
c-------------------------------
        pm=wvno*ra*d(m)
             if(mup.gt.1)      goto 50
        rhoc=rho(m)*csq
             if(abs(ra).lt.accur)      goto 312
             if(ra) 313,312,314
312     sinpr=wvno*d(m)
        rsinp=0.0
        cosp=1.0
             goto 315
313     sinpr=(exp(pm)-exp(-pm))/(2.*ra)
        rsinp=-ra*ra*sinpr
        cosp=0.5*(exp(pm)+exp(-pm))
             goto 315
314     sinpr=sin(pm)/ra
        rsinp=ra*sin(pm)
        cosp=cos(pm)
315               continue
        a11=cosp
        a21=rhoc*sinpr
        a31=0.0
        a41=0.0
        a51=0.0
        a12=0.0
        a22=0.0
        a32=0.0
        a42=0.0
        a13=0.0
        a23=0.0
        a33=0.0
        a14=0.0
        a24=0.0
        a15=0.0
             goto 1001
c-------------------------------------------------
101     argb=1.-csq/s(m)**2
        rb=sqrt(abs(argb))
             if(argb.gt.0.) rb=-rb
        g=2.*s(m)**2/csq
        g1=g-1.
             if(mmax-m) 40,52,40
40      rhoc=rho(m)*csq
        pm=wvno*ra*d(m)
        qm=wvno*rb*d(m)
             if(ra) 213,212,214
212     rsinp=0.0
        sinpr=wvno*d(m)
        cosp=1.0
             goto 215
213     rsinp=-ra*0.5*(exp(pm)-exp(-pm))
        sinpr=-rsinp/(ra**2)
        cosp=0.5*(exp(pm)+exp(-pm))
             goto 215
214     rsinp=ra*sin(pm)
        sinpr=rsinp/(ra**2)
        cosp=cos(pm)
215                 continue
             if(abs(rb).lt.accur)      goto 218
             if(rb.gt.0.)      goto 217
        rsinq=-rb*0.5*(exp(qm)-exp(-qm))
        sinqr=-rsinq/(rb**2)
        cosq=0.5*(exp(qm)+exp(-qm))
             goto 219
217     rsinq=rb*sin(qm)
        sinqr=rsinq/(rb**2)
        cosq=cos(qm)
             goto 219
218     rsinq=0.0
        sinqr=wvno*d(m)
        cosq=1.0
             goto 219
219     rr=rsinp*rsinq
        ss=sinpr*sinqr
        cc=cosp*cosq
        rs1=rsinp*cosq
        rs2=sinqr*cosp
        rs3=sinpr*cosq
        rs4=rsinq*cosp
        gm=2.*g-1
        gs=g*g
        g1s=g1*g1
        ccm=1.-cc
        gg1=g*g1
        rhocs=rhoc*rhoc
        suu=gs*rr+g1s*ss
        a11=2.*gs-gm
        a11=a11*cc-suu-2.*gg1
        a12=-(rs1+rs2)/rhoc
        a13=gm*ccm+g1*ss+g*rr
        a13=-2.*a13/rhoc
        a14=(rs3+rs4)/rhoc
        a15=2.*ccm+rr+ss
        a15=a15/rhocs
        a21=rhoc*(g1s*rs3+gs*rs4)
        a22=cc
        a23=2.*(g*rs4+g1*rs3)
        a24=sinpr*rsinq
        a31=rhoc*(gg1*gm*ccm+g1s*g1*ss+gs*g*rr)
        a32=g1*rs2+g*rs1
        a33=1.+2.*(2*gg1*ccm+suu)
        a41=-rhoc*(g1s*rs2+gs*rs1)
        a42=rsinp*sinqr
        a51=rhocs*(2.*gs*g1s*ccm+gs*gs*rr+g1s*g1s*ss)
1001              continue
c--------------------------------------------------
c         MATRIX  MULTIPLICATION
c               effect  of  imaginary  elements  included
c---------------------------
        bb1=a11*b1+a12*b2+a13*b3+a14*b4+a15*b5
        bb2=a21*b1+a22*b2+a23*b3+a24*b4-a14*b5
        bb3=a31*b1+a32*b2+a33*b3-0.5*a23*b4+0.5*a13*b5
        bb4=a41*b1+a42*b2-2.*a32*b3+a22*b4-a12*b5
        bb5=a51*b1-a41*b2+2.*a31*b3-a21*b4+a11*b5
        b1=bb1
        b2=bb2
        b3=bb3
        b4=bb4
        b5=bb5
50                continue
c--------------------------------------------------
c         the  FOLLOWING  EXPRESSION  is  VALID   rb=0
c------------------------------
52                continue
        pp=p(m)
        sss=s(m)**2
        ppp=pp**2
        rhp=rho(m)*pp
        gra=g*ra
        g1s=g1*g1
        rba=rb-1./ra
        a11=-2.*rb*sss/ppp+csq*g1s/ppp/gra
        a12=rhp*pp
        a13=-rb/a12+g1/a12/gra
        a14=rb/a12/gra
        a15=rba/rhp/rhp/csq/g
        a12=-1./g/a12
        bb1=a11*b1+a12*b2+2.*a13*b3+a14*b4+a15*b5
             goto (501,502,503),mup
C-------dispersion
501     dltar4=-bb1
                                                        RETURN
C-------ellipticity
502          if(jump.eq.2) r12=bb1
        jump=jump+1
             if(jump.eq.3)      goto 1000
        dltar4=0.5*bb1/r12
                                                        RETURN
C-------amplitude response
503     dltar4=abs(bb1)
             if(abs(s(1)).gt.accurs)                    RETURN
        ra=c/p(1)
        rad=wvno*d(1)*sqrt(abs(ra*ra-1.))
        dltar4=abs(bb1*cos(rad))
                                                        RETURN
                                                        END
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        subroutine LEIGEN(num,ixa,jxa)
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
c----------------------------------------------------------------------
c       num = 0 - the first call, num > 0 - following calls;
c       ixa - index of period; jxa - index of mode;
c       jmax < 0 or =0 - end of program, jmax > 0 - number of layers;
c       d - layer thickness; a - Pvel; b - Svel; rho - density;
c       depth - depth of the midpoint of layer; xmu,xlamb - Lame const;
c       t - period; c - phase vel; cvar, ugr - phase and group vel 
c       obtained from integrals; wvnum - wave number.
c----------------------------------------------------------------------
        parameter (nsize=1000,nper=2000,ndep=20,nmod=20)
        common/d/  a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
        common/o/ cc(nper,nmod),tt(nper),rat(nper,nmod)
        common/c/ nmax,jmax,kmax,idrop,iedit,ndiv,mode,any,ra_1
        common/rar/ dept1(nsize),amp,ampuz,stresz,stress,mmax
        common/rar1/ dcda,dcdb,dcdr,dwx,g,g2
        common/rco/ c,cvar,ugr,wvno,ratio,ale,nwmax
        common/rco1/ sumi0,sumi1,sumi2,sumi3,flagr
        common/derivd/ nderiv,ndpth,dpth(ndep),dspz(ndep),dsp(ndep),
     *  drvz(5,ndep),drv(5,ndep)
	common/log/ KEY_ATTEN,KEY_DERIV
        real*8 dcda(nsize),dcdb(nsize),dcdr(nsize),dwx(nsize)
        dimension dmm(5),smm(5),amp(nsize),stress(nsize)
        dimension depth(nsize)
        dimension xmu(nsize),xlamb(nsize),g(nsize)
        dimension pr(5*nsize)
        dimension ampuz(nsize),stresz(nsize),g2(nsize)
        logical KEY_ATTEN,KEY_DERIV
	data const_lim/1.E+10/,const_lim1/1.E+5/
        xxmin=1.0e-20
c
c
CCCC                    if(num.gt.0.and..not.KEY_ATTEN)     go to 503
        mmax=jmax
        nmax=mmax
                        if(mmax) 777,777,778
778                                     continue
        base=0.0
        mm1=mmax-1
        ivre=(nsize-1)/mm1
                        if(ndiv.gt.ivre) ndiv=ivre
        div=float(ndiv)
                        if(ndiv.le.1)                   go to 20003
        jj=1
                        if(b(1).le.0.1e-10) jj=2
                                do 10001 j=jj,mm1
        ldiv=(j-jj)*ndiv
                                do 10001 i=1,ndiv
        pr(ldiv+i)=d(j)/div
        pr(ldiv+i+nsize)=b(j)
        pr(ldiv+i+2*nsize)=rho(j)
	if(KEY_ATTEN)pr(ldiv+i+3*nsize)=qs(j)
C       PRint*,' atten=',1./qs(j)
10001                                   continue
        mmax=(mm1-jj+1)*ndiv+jj
        d(mmax)=0.
        a(mmax)=a(nmax)
        b(mmax)=b(nmax)
        rho(mmax)=rho(nmax)
	if(KEY_ATTEN)qs(mmax)=qs(nmax)
C       PRint*,nmax,mmax,1./qs(mmax)
        nmax=mmax
        mm1=mmax-1
                                do 10002 j=jj,mm1
        d(j)=pr(j-jj+1)
c	write(*,*) 'dddd = ',d(j)
        b(j)=pr(j-jj+nsize+1)
        rho(j)=pr(j-jj+2*nsize+1)
        if(KEY_ATTEN)qs(j)=pr(j-jj+3*nsize+1)
C       PRint*,' atten again=',1./qs(j)
10002                                   continue
20003                                   continue
c
c
c
c
c
                                do 22 i=1,mmax
        base=base+d(i)
        depth(i)=base-d(i)*0.5
        xmu(i)=rho(i)*b(i)*b(i)
                        if(i-mmax)22,24,24
22                                      continue
24      depth(mmax)=base-d(mmax)
c
c
503                                     continue
                                do 500 j=1,nsize
        amp(j)=0.0
        stress(j)=0.0
        dcdr(j)=0.0
        dcdb(j)=0.0
500     g(j)=0.0
        t=tt(ixa)
        c=cc(ixa,jxa)
499                                     continue
        mmax=nmax
c-----------------------------------------------------------------------
c       Layer dropping procedure; any is the same as fact.
c-----------------------------------------------------------------------
                        if(any.le.0.) any=7.0
        dmax=any*c*t
899     sum=0.
                                do 900 ii=1,mmax
        max=ii
                        if(c-b(ii)) 901,900,900
901     sum=sum+d(ii)
                        if(ii.eq.mmax)                  go to 902
                        if(sum.le.dmax)                 go to 900
                        if(b(ii+1)-b(ii)) 902,900,90009
900                                     continue
90009   max=max+1
902     mmax=max
        wvno=6.2831853/(c*t)
        wvnum=wvno
        wvnosq=wvno*wvno
        omegsq=(6.2831853/t)**2
C-------the halfspace is handled-------------------------
        ut0=1.
7777    ut=ut0
        covb=c/b(mmax)
        h=rho(mmax)*b(mmax)*b(mmax)
        rb=wvno*sqrt(abs(covb**2-1.))
        tq=-h*rb*ut0
        amp(mmax)=ut
        stress(mmax)=tq
        g(mmax)=tq/xmu(mmax)
                        if(rb) 1001,1000,1001
1000    dm=1.0e25
        sm=0.
                                                        go to 1002
1001    dm=0.5/rb
        sm=0.5*rb
1002                                    continue
        dldr=omegsq*dm
        dldm=-(wvnosq*dm+sm)
        dcdb(mmax)=2.*rho(mmax)*b(mmax)*c*dldm/wvno
        dcdr(mmax)=(c/wvno)*(dldr+b(mmax)*b(mmax)*dldm)
        mmm1=mmax-1
        sumi0=rho(mmax)*dm
        sumi1=h*dm
        sumi2=h*sm
C---------------loop for knots-----------S
                                do 1340 k=1,mmm1
 	IF(abs(ut).gt.const_lim) THEN
         ut0=ut0/const_lim1
 	goto 7777 
 				  ENDIF
        m=mmax-k
                        if(b(m).eq.0.0)                 go to 1340
        covb=c/b(m)
        rb=wvno*sqrt(abs(covb**2-1.))
        h=rho(m)*b(m)*b(m)
        dz=d(m)/4.
        dmm(1)=ut*ut
        smm(1)=(tq/h)**2
                                do 1339 kk=2,5
        xkk=kk-1
        q=rb*dz*xkk
                        if(c-b(m)) 1207,1221,1231
1231    sinq=sin(q)
        y=sinq/rb
        z=-rb*sinq
        cosq=cos(q)
                                                        go to 1242
1221    y=dz*xkk
        z=0.0
        cosq=1.0
                                                        go to 1242
1207    exqp=exp(q)
        exqm=1./exqp
        y=(exqp-exqm)/(2.*rb)
        z=rb*rb*y
        cosq=(exqp+exqm)/2.
1242    eut=cosq*ut-y*tq/h
        ett=-h*z*ut+cosq*tq
        dmm(kk)=eut*eut
        smm(kk)=(ett*ett)/(h*h)
                        if(kk-3) 1339,1301,1339
1301    amp(m)=eut
        stress(m)=ett
        g(m)=ett/xmu(m)
1339                                    continue
        ut=eut
        tq=ett
        dm=(dz/22.5)*(7.*(dmm(1)+dmm(5))+32.*(dmm(2)+dmm(4))+12.*dmm(3))
        sm=(dz/22.5)*(7.*(smm(1)+smm(5))+32.*(smm(2)+smm(4))+12.*smm(3))
        dldm=-(wvnosq*dm+sm)
        dldr=omegsq*dm
        dcdb(m)=2.*rho(m)*b(m)*c*dldm/wvno
        dcdr(m)=(c/wvno)*(dldr+b(m)*b(m)*dldm)
        sumi0=sumi0+rho(m)*dm
        sumi1=sumi1+h*dm
        sumi2=sumi2+h*sm
1340                                    continue
C---------------loop for knots-----------E
        amp(nsize)=1.0
        stress(nsize)=tq/ut
        dcdb(nsize)=0.0
        dcdr(nsize)=0.0
        depth(nsize)=0.0
                        if(b(1).eq.0.0) depth(nsize)=d(1)
        dldk=-2.*wvno*sumi1
                                do 1500 l=1,mmax
                        if(b(l).eq.0.0)                 go to 1500
        amp(l)=amp(l)/ut
        stress(l)=stress(l)/ut
        dcdb(l)=dcdb(l)/dldk
        dcdr(l)=dcdr(l)/dldk
        g(l)=stress(l)/xmu(l)
c--------------------------------------------------------
c       Exclusion of low amplitudes from final output
c--------------------------------------------------------
                        if(b(l)-b(mmax)) 1506,1507,1507
1507                                    continue
                        if(abs(amp(l))-xxmin) 1501,1502,1502
1501    amp(l)=0.0
        stress(l)=0.0
        dcdb(l)=0.0
        dcdr(l)=0.0
        g(l)=0.0
                                                        go to 1500
1502                                    continue
1506                                    continue
1500                                    continue
        sumi0=sumi0/(ut*ut)
        sumi1=sumi1/(ut*ut)
        sumi2=sumi2/(ut*ut)
        wvar=(omegsq*sumi0-sumi2)/sumi1
        cvar=sqrt(omegsq/wvar)
        wvar=sqrt(wvar)
        ugr=sumi1/(c*sumi0)
        flagr=omegsq*sumi0-wvnosq*sumi1-sumi2
        ale=1./(2.*c*ugr*sumi0)/sqrt(6.28318)*1.e-15
                        if(b(1).le.0.0)                 go to 5556
C-------attention - water; ignore for Love waves !!!!!
                                do 5555 l=1,mmax
        i=mmax-l+1
        j=i+1
        dept1(j)=depth(i)
        amp(j)=amp(i)
        stress(j)=stress(i)
        dcdb(j)=dcdb(i)
        dcdr(j)=dcdr(i)
5555    g(j)=g(i)
        mmax=mmax+1
                                                        go to 501
5556                            do 5557 l=1,mmax
5557    dept1(l)=depth(l)
501     dept1(1)=0.
C-------attention - water; ignore for Love waves!!!!!
                        if(b(1).le.0.0) dept1(1)=d(1)
        amp(1)=1.
        stress(1)=0.
        dcdb(1)=0.
        dcdr(1)=0.
        g(1)=0.
c----------------------------------------------------------------------
c       Calculation of eigenfunction and its derivatives
c----------------------------------------------------------------------
                        if(iedit.ne.2)                  go to 800
                                do 799 i=1,5
                                do 799 j=1,ndep 
        dsp(j)=0.0
        drv(i,j)=0.0
799                                     continue
                                do 801 l=1,ndpth
                        if(dpth(l).lt.dept1(1))         go to 801
c----------------------------------------------------------------------
c       If dpth(l) is above the free surface, the output will be zero
c----------------------------------------------------------------------
                                do 802 ii=2,mmax
        rab1=dept1(ii)-dpth(l)
                        if(abs(rab1).lt.1.0e-5) rab1=0.
                        if(rab1.le.0.)                  go to 802
        i=ii-1
        j=ii
                        if(b(1).le.0.0) i=ii
C-------attention - water; ignore for Love waves!!!!!
        rab2=d(i)/2.
        rab1=rab1-rab2
                        if(abs(rab1).lt.1.0e-5) rab1=0.
                        if(rab1.le.0.)                  go to 803
        j=j-1
        i=i-1
                                                        go to 803
802                                     continue
        j=mmax
        i=mmax-1
                        if(b(1).le.0.0) i=j
C-------attention - water!!!!!
803     dz=dpth(l)-dept1(j)
        ampl=amp(j)
        stres=stress(j)
        covb=c/b(i)
        rb=wvno*sqrt(abs(covb**2-1.))
                        if(c-b(i)) 810,811,812
810     nderv=nderiv+1
                                do 804 ii=1,nderv
        k=ii-1
        rab1=rb**k
        q=rb*dz
        expq=exp(q)
        uttq=(ampl+stres/(xmu(i)*rb))/ampl
        rab2=rab1*expq*uttq*ampl/2.
        uttq=abs(uttq)
                        if(uttq.lt.1.0e-5) rab2=0.
        rab3=(-1)**k*rab1*(ampl/2.-stres/(2.*xmu(i)*rb))/expq
                        if(k.ne.0)                      go to 805
        dsp(l)=rab2+rab3
                                                        go to 804
805     drv(k,l)=rab2+rab3
804                                     continue
                                                        go to 801
811     dsp(l)=ampl+stres*dz/xmu(i)
                        if(nderiv.lt.1)                 go to 801
        drv(1,l)=stres/xmu(i)
                                                        go to 801
812     nderv=nderiv+1
                                do 806 ii=1,nderv
        k=ii-1
        rab1=rb**k
        q=rb*dz
        q=q+k*3.1415926/2.
        cosq=cos(q)
        sinq=sin(q)
        rab2=rab1*ampl*cosq
        rab3=stres*rab1*sinq/(xmu(i)*rb)
                        if(k.ne.0)                      go to 807
        dsp(l)=rab2+rab3
                                                        go to 806
807     drv(k,l)=rab2+rab3
806                                     continue
801                                     continue
800                                                             RETURN
777                                     continue
                                                                STOP
                                                                END
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        subroutine REIGEN (num,ixa,jxa)
czzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
         parameter (nsize=1000,nper=2000,ndep=20,nmod=20)
        double precision yy1,yy2,yy3,yy4,yz1,yz2,yz3,yz4,aur1,auz1,
     1  atz1,atr1,aur2,auz2,atz2,atr2,aa,bb,daur1,dauz1,datz1,datr1,
     2  daur2,dauz2,datz2,datr2,eaur1,eauz1,eatz1,eatr1,eaur2,eauz2,
     3  eatz2,eatr2,saur1,sauz1,satz1,satr1,saur2,sauz2,satz2,satr2,
     4  xnorm,dmmr,dmmz,smmz,smmr,drsz,dzsr,dldr,dldm,dldl,dldk,
     5  dcdr(nsize),dcda(nsize),dcdb(nsize),dwx(nsize)
        complex cra,cr1,cr2,cr3
        common/d/   a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
        common/o/   cc(nper,nmod),tt(nper),rat(nper,nmod)
        common/c/   nmax,jmax,kmax,idrop,iedit,ndiv,mode,fact,ra_1 
        common/newly/ te1,dete,ce1,dece,lstop,iqq,istru,cinit  
        common/rar/ dept1(nsize),ampur,ampuz,stresz,stresr,mmax
        common/rar1/ dcda,dcdb,dcdr,dwx,g1,g2
        common/rco/ c,cvar,ugr,wvno,ratio,are,nwmax
        common/rco1/ sumi0,sumi1,sumi2,sumi3,flagr
        common/derivd/ nderiv,ndpth,dpth(ndep),dspz(ndep),dspr(ndep),
     *  drvz(5,ndep),drvr(5,ndep)
	common/log/ KEY_ATTEN,KEY_DERIV
        logical KEY_ATTEN,KEY_DERIV
        dimension   stresr(nsize),stresz(nsize),wwt(4),wt(4)
        dimension yy1(nsize,5),yy2(nsize,5),yy3(nsize,5),yy4(nsize,5)
        dimension yz1(nsize,5),yz2(nsize,5),yz3(nsize,5),yz4(nsize,5)
        dimension dmrsmz(5),dmzsmr(5),depth(nsize),ampur(nsize),
     *ampuz(nsize)
        dimension dmr(5),dmz(5),smr(5),smz(5),g1(nsize),g2(nsize)
        dimension pr(5*nsize),xlamb(nsize),xmu(nsize)
        COMMON/NEW/derz(nsize),derr(nsize)
        data pi2/6.283185307/
c----------------------------------------------------------------------
c
        ch_s(x)=exp(x)+exp(-x)
        sh_s(x)=exp(x)-exp(-x)
                  do 99999 i=1,nsize
                  do 99999 j=1,5
        yy1(i,j)=1.
        yy2(i,j)=1.
        yy3(i,j)=1.
        yy4(i,j)=1.
        yz1(i,j)=2.
        yz2(i,j)=1.
        yz3(i,j)=1.
        yz4(i,j)=1.
99999             continue
        xxmin=1.0e-15
             if(num.gt.0.and..not.KEY_ATTEN)      goto 503
        mmax=jmax
        nmax=mmax
        wwt(1)=0.
        wwt(2)=0.5
        wwt(3)=0.5
        wwt(4)=1.0
        wt(1)=1./6.
        wt(2)=1./3.
        wt(3)=1./3.
        wt(4)=1./6.
             if(fact.le.0) fact=4.
             if(num.eq.-1)      goto 88888
             if(mmax) 777,777,778
c
  778             continue
             if(istru.lt.1)     goto 20005
        call strut
             goto 20004
20005             continue
        base=0.0
        mm1=mmax-1
        ivre=99/mm1
             if(ndiv.gt.ivre) ndiv=ivre
        div=float(ndiv)
        jj=1
             if(ndiv.le.1)      goto 20003
C-------attention - water!!!!!
             if(b(1).le.0.1e-10) jj=2
C------no splitting of water layer
C-------attention - water!!!!!
C------layer splitting-------------------------------------S
                  do 10001 j=jj,mm1
        ldiv=(j-jj)*ndiv
                  do 10001 i=1,ndiv
        pr(ldiv+i)=d(j)/div
        pr(ldiv+nsize+i)=a(j)
        pr(ldiv+2*nsize+i)=b(j)
        pr(ldiv+3*nsize+i)=rho(j)
	if(KEY_ATTEN) pr(ldiv+4*nsize+i)=qs(j)
C        PRint*,'atten=',1./qs(j)
10001             continue
        mmax=(mm1-jj+1)*ndiv+jj
        d(mmax)=0.
        a(mmax)=a(nmax)
        b(mmax)=b(nmax)
        rho(mmax)=rho(nmax)
	if(KEY_ATTEN)qs(mmax)=qs(nmax)
C       PRint*,nmax,mmax,1./qs(nmax)
        nmax=mmax
        mm1=mmax-1
                  do 10002 j=jj,mm1
        d(j)=pr(j-jj+1)
c	write(*,*) "ddddd ",d(j)
        a(j)=pr(j-jj+nsize+1)
        b(j)=pr(j-jj+2*nsize+1)
        rho(j)=pr(j-jj+3*nsize+1)
	if(KEY_ATTEN) qs(j)=pr(j-jj+4*nsize+1)
C        PRint*,'atten again=',1./qs(j)
10002             continue
C------layer splitting-------------------------------------E
20003             continue
c
c
20004             continue
        mmax=nmax
88888   base=0.0
                  do 22 i=1,mmax
        base=base+d(i)
        depth(i)=base-d(i)*0.5
        xmu(i)=rho(i)*b(i)*b(i)
        xlamb(i)=rho(i)*(a(i)*a(i)-2.*b(i)*b(i))
             if(i-mmax) 22,24,24
22                continue
24      depth(mmax)=base-d(mmax)
c
c
503               continue
                  do 500 j=1,nmax
        dcda(j)=0.0
        dcdb(j)=0.0
        dcdr(j)=0.0
        ampur(j)=0.0
        ampuz(j)=0.0
        stresr(j)=0.0
        stresz(j)=0.0
        g1(j)=0.0
        g2(j)=0.0
500               continue
        t=tt(ixa)
        c=cc(ixa,jxa)
        ratio=rat(ixa,jxa)
        mmax=nmax
        dmax=fact*t*c
899     sum=0.0
                  do 900 ii=1,mmax
        max=ii
             if(c-b(ii)) 901,900,900
901     sum=sum+d(ii)
             if(ii.eq.mmax)      goto 902
             if(sum.le.dmax)      goto 900
             if(a(ii+1)-a(ii)) 902,903,90009
903          if(b(ii+1)-b(ii)) 902,900,90009
900               continue
90009   max=max+1
902     mmax=max
        sumi0=0.0
        sumi1=0.0
        sumi2=0.0
        sumi3=0.0
        wvno=6.2831853072/(c*t)
        wvnosq=wvno*wvno
        wvnum=wvno
        omega=6.2831853072/t
        omegsq=omega*omega
             if(b(1).gt.0.0)      goto 4000
C-------attention - water!!!!!
C-----------------------------------------water layer---S
        ra=c/a(1)
        cr1=ra*ra-1.
        cra=wvno*csqrt(cr1)
                        if(cabs(cra).le.1.0e-35)        go to 4001
                                                        go to 4002
4001    sumi0=rho(1)*d(1)
        sumi1=0.0
        sumi2=0.0
        sumi3=0.0
        tzz=0.0
             goto 4000
4002    cr1=csin(2.*cra*d(1))
        cr2=4.*cra
        cr3=cr1/cr2
        sin2ra=real(cr3)
        cr1=ccos(cra*d(1))
        cosra=real(cr1)
        cos2rm=1./(cosra*cosra)
        fac1=(0.5*d(1)+sin2ra)*cos2rm
        fac3=wvno*(0.5*d(1)-sin2ra)*cos2rm
        cr1=cra*cra
        rab1=real(cr1)
        fac2=wvno*fac3/(rab1)
        fac4=rab1*fac3/wvno
        sumi0=rho(1)*(fac1+fac2)
        sumi1=xlamb(1)*fac2
        sumi2=xlamb(1)*fac3
        sumi3=xlamb(1)*fac4
        cr1=csin(cra*d(1))
        cr2=cr1/cra
        rab1=real(cr2)
        tzz=-rho(1)*omegsq*rab1/cosra
C-----------------------------------------water layer---E
4000              continue
        cova=c/a(mmax)
        covb=c/b(mmax)
        gam=2./covb**2
        gamm1=gam-1.
        ra=wvno*sqrt(abs(cova**2-1.))
        rb=wvno*sqrt(abs(covb**2-1.))
        det=wvnosq-ra*rb
        h=rho(mmax)*omegsq
        brkt=-gamm1*wvno+gam*ra*rb/wvno
        iter=0
        aur1=1.d+00
        auz1=0.d+00
        atz1=-h*brkt/det
        atr1=-h*ra/det
        mmm1=mmax-1
                  do 1346 mm=1,mmm1
        m=mmax-mm
             if(b(m).le.0.0)      goto 1346
        xdiv=1.
        ddz=-d(m)/(4.*xdiv)
        a12=1./(xlamb(m)+2.*xmu(m))
        a13=wvno*xlamb(m)*a12
        a21=-omegsq*rho(m)
        a24=wvno
        a31=-wvno
        a34=1./xmu(m)
        a42=-a13
        a43=a21+4.*wvnosq*xmu(m)*(xlamb(m)+xmu(m))*a12
        yy3(m,5)=aur1
        yy1(m,5)=auz1
        yy2(m,5)=atz1
        yy4(m,5)=atr1
                  do 1338 kk=2,5
        k=6-kk
        eaur1=aur1
        eauz1=auz1
        eatz1=atz1
        eatr1=atr1
        daur1=0.d00
        dauz1=0.d00
        datz1=0.d00
        datr1=0.d00
                  do 1401 ll=1,4
        saur1=aur1+wwt(ll)*ddz*daur1
        sauz1=auz1+wwt(ll)*ddz*dauz1
        satz1=atz1+wwt(ll)*ddz*datz1
        satr1=atr1+wwt(ll)*ddz*datr1
        daur1=a31*sauz1+a34*satr1
        dauz1=a12*satz1+a13*saur1
        datz1=a21*sauz1+a24*satr1
        datr1=a42*satz1+a43*saur1
        eaur1=eaur1+wt(ll)*ddz*daur1
        eauz1=eauz1+wt(ll)*ddz*dauz1
        eatz1=eatz1+wt(ll)*ddz*datz1
        eatr1=eatr1+wt(ll)*ddz*datr1
1401              continue
        aur1=eaur1
        auz1=eauz1
        atz1=eatz1
        atr1=eatr1
1400              continue
        yy1(m,k)=auz1
        yy2(m,k)=atz1
        yy3(m,k)=aur1
        yy4(m,k)=atr1
1338              continue
1346              continue
             if(b(1).gt.0.0)      goto 1347
        yy1(1,1)=yy1(2,1)
        yy2(1,1)=yy2(2,1)
        yy3(1,1)=yy3(2,1)
        yy4(1,1)=yy4(2,1)
1347              continue
4003    aur2=0.d00
        auz2=1.d00
        atz2=-h*rb/det
        atr2=-h*brkt/det
             if(iter.eq.0)      goto 4004
        aur1=1.d 00
        auz1=0.d 00
        atz1=-h*brkt/det
        atr1=-h*ra/det
        aur2=aur2+xnorm*aur1
        auz2=auz2+xnorm*auz1
        atz2=atz2+xnorm*atz1
        atr2=atr2+xnorm*atr1
4004              do 2346 mm=1,mmm1
        m=mmax-mm
             if(b(m).le.0.0)      goto 2346
        ddz=-d(m)/(4.*xdiv)
        a12=1./(xlamb(m)+2.*xmu(m))
        a13=wvno*xlamb(m)*a12
        a21=-omegsq*rho(m)
        a24=wvno
        a31=-wvno
        a34=1./xmu(m)
        a42=-a13
        a43=a21+4.*wvnosq*xmu(m)*(xlamb(m)+xmu(m))*a12
        yz3(m,5)=aur2
        yz1(m,5)=auz2
        yz2(m,5)=atz2
        yz4(m,5)=atr2
                do 2338 kk=2,5
        k=6-kk
        eaur2=aur2
        eauz2=auz2
        eatz2=atz2
        eatr2=atr2
        daur2=0.0
        dauz2=0.0
        datz2=0.0
        datr2=0.0
                  do 2401 ll=1,4
        saur2=aur2+wwt(ll)*ddz*daur2
        sauz2=auz2+wwt(ll)*ddz*dauz2
        satz2=atz2+wwt(ll)*ddz*datz2
        satr2=atr2+wwt(ll)*ddz*datr2
        daur2=a31*sauz2+a34*satr2
        dauz2=a12*satz2+a13*saur2
        datz2=a21*sauz2+a24*satr2
        datr2=a42*satz2+a43*saur2
        eaur2=eaur2+wt(ll)*ddz*daur2
        eauz2=eauz2+wt(ll)*ddz*dauz2
        eatz2=eatz2+wt(ll)*ddz*datz2
        eatr2=eatr2+wt(ll)*ddz*datr2
2401              continue
        aur2=eaur2
        auz2=eauz2
        atz2=eatz2
        atr2=eatr2
2400              continue
        yz1(m,k)=auz2
        yz2(m,k)=atz2
        yz3(m,k)=aur2
        yz4(m,k)=atr2
2338              continue
2346              continue
             if(b(1).gt.0.0)      goto 2347
        yz1(1,1)=yz1(2,1)
        yz2(1,1)=yz2(2,1)
        yz3(1,1)=yz3(2,1)
        yz4(1,1)=yz4(2,1)
2347              continue
        aa=yz3(1,1)-ratio*yz1(1,1)
        bb=ratio*yy1(1,1)-yy3(1,1)
             if(dabs(bb).lt.1.d-10) bb=dsign(1.d-10,bb)
        xnorm=aa/bb
        bb=xnorm*yy1(1,1)+yz1(1,1)
             if(dabs(bb).lt.1.d-10) bb=dsign(1.d-10,bb)
        ampur(nsize)=(xnorm*yy3(1,1)+yz3(1,1))/bb
        ampuz(nsize)=(xnorm*yy1(1,1)+yz1(1,1))/bb
        stresz(nsize)=(xnorm*yy2(1,1)+yz2(1,1))/bb
        stresr(nsize)=(xnorm*yy4(1,1)+yz4(1,1))/bb
        iter=iter+1
             if(iter.gt.1)      goto 2011
        xtest=abs(ampur(nsize)/ratio-1.)
             if(xtest.ge.0.00001)      goto 4003
2011    dcdb(nsize-1)=iter
        dcda(nsize)=0.0
        dcdb(nsize)=0.0
        dcdr(nsize)=0.0
        ampur(nsize-1)=yy3(1,1)
        ampur(nsize-2)=yz3(1,1)
        ampuz(nsize-1)=yy1(1,1)
        ampuz(nsize-2)=yz1(1,1)
        stresz(nsize-1)=yy2(1,1)
        stresz(nsize-2)=yz2(1,1)
        stresr(nsize-1)=yy4(1,1)
        stresr(nsize-2)=yz4(1,1)
        dcdb(nsize-2)=bb
        dcda(nsize-2)=0.0
        dcda(nsize-1)=0.0
        dcdr(nsize-2)=0.0
        dcdr(nsize-1)=0.0
                  do 7000 m=1,mmax
             if(b(m).le.0.0)      goto 7000
             if(m-mmax) 7001,77777,77777
7001    dz=d(m)/4.
                  do 1339 kk=1,5
        aur=(xnorm*yy3(m,kk)+yz3(m,kk))/bb
        auz=(xnorm*yy1(m,kk)+yz1(m,kk))/bb
        atz=(xnorm*yy2(m,kk)+yz2(m,kk))/bb
        atr=(xnorm*yy4(m,kk)+yz4(m,kk))/bb
        durdz=atr/xmu(m)-wvno*auz
        duzdz=(atz+wvno*xlamb(m)*aur)/(xlamb(m)+2.*xmu(m))
        dmr(kk)=aur*aur
        dmz(kk)=auz*auz
        smr(kk)=durdz*durdz
        smz(kk)=duzdz*duzdz
        dmrsmz(kk)=aur*duzdz
        dmzsmr(kk)=auz*durdz
             if(kk-3) 1339,1301,1339
1301    ampur(m)=aur
        ampuz(m)=auz
        stresz(m)=atz
        stresr(m)=atr
        derz(m)=(atz+wvno*xlamb(m)*aur)/(xlamb(m)+2.*xmu(m))
        derr(m)=atr/xmu(m)-wvno*auz
        g1(m)=(atz+aur*xlamb(m)*wvno)/(xlamb(m)+2.*xmu(m))
        g2(m)=atr/xmu(m)
1339              continue
        dmmr=(dz/22.5)*(7.*(dmr(1)+dmr(5))+32.*(dmr(2)+dmr(4))+12.
     *  *dmr(3))
        dmmz=(dz/22.5)*(7.*(dmz(1)+dmz(5))+32.*(dmz(2)+dmz(4))+12.
     *  *dmz(3))
        smmz=(dz/22.5)*(7.*(smz(1)+smz(5))+32.*(smz(2)+smz(4))+12.
     *  *smz(3))
        smmr=(dz/22.5)*(7.*(smr(1)+smr(5))+32.*(smr(2)+smr(4))+12.
     *  *smr(3))
        drsz=(dz/22.5)*(7.*(dmrsmz(1)+dmrsmz(5))+32.*(dmrsmz(2)
     *  +dmrsmz(4))+12.*dmrsmz(3))
        dzsr=(dz/22.5)*(7.*(dmzsmr(1)+dmzsmr(5))+32.*(dmzsmr(2)
     *  +dmzsmr(4))+12.*dmzsmr(3))
        sumi0=sumi0+rho(m)*(dmmr+dmmz)
        sumi1=(xlamb(m)+2.*xmu(m))*dmmr+xmu(m)*dmmz+sumi1
        sumi2=xmu(m)*dzsr-xlamb(m)*drsz+sumi2
        sumi3=(xlamb(m)+2.*xmu(m))*smmz+xmu(m)*smmr+sumi3
        dldl=-wvnosq*dmmr+2.*wvno*drsz-smmz
        dldm=-wvnosq*(2.*dmmr+dmmz)-2.*wvno*dzsr-(2.*smmz+smmr)
        dldr=omegsq*(dmmr+dmmz)
        dcdb(m)=2.*rho(m)*b(m)*c*(dldm-2.*dldl)/wvno
        dcda(m)=2.*rho(m)*a(m)*c*dldl/wvno
        dcdr(m)=(c/wvno)*(dldr+xlamb(m)*dldl/rho(m)+xmu(m)*dldm/rho(m))
C
             if(abs(auz)+abs(aur)-xxmin) 7002,7002,7000
7000              continue
77777             continue
             if((b(1).gt.0.1e-10).or.m.ne.2)      goto 7002
        aur=ratio
        auz=1.
        atr=0.
        atz=tzz
7002    ampur(m)=aur
        ampuz(m)=auz
        stresr(m)=atr
        stresz(m)=atz
        g1(m)=(atz+aur*xlamb(m)*wvno)/(xlamb(m)+2.*xmu(m))
        g2(m)=atr/xmu(m)
        ap=-rho(m)*(wvno*aur+rb*auz)/det
        bp=-rho(m)*(-ra*aur/wvno-auz)/det
        a1=-wvno*ap/rho(m)
        a2=-wvno*rb*bp/rho(m)
        a3=ra*ap/rho(m)
        a4=wvnosq*bp/rho(m)
             if(rb) 7005,7006,7005
7005    dmmr=a1*a1/(2.*ra)+2.*a1*a2/(ra+rb)+a2*a2/(2.*rb)
        dmmz=a3*a3/(2.*ra)+2.*a3*a4/(ra+rb)+a4*a4/(2.*rb)
        smmz=ra*a3*a3/2.+2.*ra*rb*a3*a4/(ra+rb)+rb*a4*a4/2.
        smmr=ra*a1*a1/2.+2.*ra*rb*a1*a2/(ra+rb)+rb*a2*a2/2.
        drsz=-a1*a3/2.-(a1*a4*rb+a2*a3*ra)/(ra+rb)-a2*a4/2.
        dzsr=-a1*a3/2.-(a1*a4*ra+a2*a3*rb)/(ra+rb)-a2*a4/2.
             goto 7010
7006    ugr=b(m)
        flagr=0.0
        sumi0=rho(m)*10.**(25)
        sumi1=xmu(m)*10.**(25)
        sumi2=0.0
        sumi3=0.0
        are=0.0
        dcdb(m)=-2.*wvno*10.**(25)
             goto 531
7010              continue
        sumi0=sumi0+rho(m)*(dmmr+dmmz)
        sumi1=(xlamb(m)+2.*xmu(m))*dmmr+xmu(m)*dmmz+sumi1
        sumi2=xmu(m)*dzsr-xlamb(m)*drsz+sumi2
        sumi3=(xlamb(m)+2.*xmu(m))*smmz+xmu(m)*smmr+sumi3
        dldr=omegsq*(dmmr+dmmz)
        dldm=-wvnosq*(2.*dmmr+dmmz)-2.*wvno*dzsr-(2.*smmz+smmr)
        dldl=-wvnosq*dmmr+2.*wvno*drsz-smmz
        dcda(m)=2.*rho(m)*a(m)*c*dldl/wvno
        dcdb(m)=2.*rho(m)*b(m)*c*(dldm-2.*dldl)/wvno
        dcdr(m)=(c/wvno)*(dldr+xlamb(m)*dldl/rho(m)+xmu(m)*dldm/rho(m))
7011              continue
        ugr=(wvno*sumi1+sumi2)/(omega*sumi0)
        flagr=omegsq*sumi0-wvnosq*sumi1-2.*wvno*sumi2-sumi3
        wvar=(-sumi2+sqrt(abs(sumi2**2-sumi1*(sumi3-
     *  omegsq*sumi0))))/sumi1
        cvar=omega/wvar
        are=1./(2.*c*ugr*sumi0)/sqrt(6.28318)*1.e-15
531               continue
501               continue
c
c
c
c
c
c
        n=1
             if(b(1).le.0.0) n=2
                  do 5010 m=n,mmax
        dldk=-2.*(wvnum*sumi1+sumi2)
        dcdr(m)=dcdr(m)/dldk
        dcda(m)=dcda(m)/dldk
        dcdb(m)=dcdb(m)/dldk
        dwx(m)=(dcda(m)*4./3.*b(m)/a(m)+dcdb(m))*b(m)
5010                continue
                        if(b(1).le.0.0)         go to 5556
        do 5555 m=1,mmax
        i=mmax-m+1
        j=i+1
        dept1(j)=depth(i)
        ampur(j)=ampur(i)
        ampuz(j)=ampuz(i)
        stresz(j)=stresz(i)
        stresr(j)=stresr(i)
        dcda(j)=dcda(i)
        dcdb(j)=dcdb(i)
        dcdr(j)=dcdr(i)
        derr(j)=derr(i)
        derz(j)=derz(i)
        dwx(j)=dwx(i)
        g1(j)=g1(i)
        g2(j)=g2(i)
5555                continue
        mmax=mmax+1
                                                go to 5558
5556                            do 5557 m=1,mmax
5557    dept1(m)=depth(m)
5558    dept1(1)=0.
        if(b(1).le.0.0) dept1(1)=d(1)
        ampur(1)=ratio
        ampuz(1)=1.
        stresz(1)=0.
        atz=0.0
        derr(1)=-wvno
        if(b(1).gt.0.0) then
        derz(1)=wvno*xlamb(1)*ratio/(xlamb(1)+2.*xmu(1))
        derr(1)=-wvno
                        else
         derz(1)= (tzz+wvno*xlamb(2)*ratio)/(xlamb(2)+2.*xmu(2))
                        endif
        if(b(1).le.0.0) stresz(1)=tzz
        stresr(1)=0.
        dcda(1)=0.
        dcdb(1)=0.
        dcdr(1)=0.
        dwx(1)=0.
        g1(1)=0.
        g2(1)=0.
c----------------------------------------------------------------------
c       Calculation of eigen function and its derivatives
c----------------------------------------------------------------------
                        if(iedit.ne.2)                  go to 800
                                do 799 i=1,5
                                do 799 j=1,ndep  
        dspz(j)=0.0
        dspr(j)=0.0
        drvz(i,j)=0.0
        drvr(i,j)=0.0
799                                     continue
                                do 801 l=1,ndpth
                        if(dpth(l).lt.dept1(1))         go to 801
c----------------------------------------------------------------------
c       If dpth(l) is above the free surface, the output will be zero
c----------------------------------------------------------------------
                                do 802 ii=2,mmax
        rab1=dept1(ii)-dpth(l)
                        if(abs(rab1).lt.1.0e-5) rab1=0.
                        if(rab1.le.0.)                  go to 802
        i=ii-1
        j=ii
                        if(b(1).le.0.0) i=ii
        rab2=d(i)/2.
        rab1=rab1-rab2
                        if(abs(rab1).lt.1.0e-5) rab1=0.
                        if(rab1.le.0.)                  go to 803
        j=j-1
        i=i-1
                                                        go to 803
802                                     continue
        j=mmax
        i=mmax-1
                        if(b(1).le.0.0) i=j
803     dz=dpth(l)-dept1(j)
        amplz=ampuz(j)
        amplr=ampur(j)
        strsz=stresz(j)
        strsr=stresr(j)
        cova=c/a(i)
        covb=c/b(i)
        rb=wvno*sqrt(abs(covb**2-1.))
        ra=wvno*sqrt(abs(cova**2-1.))
        del=2.*b(i)*b(i)-c*c
c----------------------------------------------------------------
c       Calculation of terms, connected with a-velocity
c----------------------------------------------------------------
        rab1=(c-a(i))/c
                        if(abs(rab1).lt.1.0e-5) rab1=0.
                        if(rab1) 810,811,812
810     nderv=nderiv+1
        q=ra*dz
        expq=exp(q)
        rab2=2.*xmu(i)*wvno*ra*amplr
        rab3=del*rho(i)*wvno*wvno*amplz
        rab4=2.*rho(i)*c*c*ra*wvno
        c1=(rab2-rab3+wvno*strsr-ra*strsz)/rab4
        uttq=abs(c1/amplr)
                        if(uttq.lt.1.0e-5) c1=0.
        c2=(rab2+rab3-wvno*strsr-ra*strsz)/rab4
                                do 804 ii=1,nderv
        k=ii-1
        rab1=ra**k
        rab2=rab1*c1*expq
        rab3=(-1)**k*rab1*c2/expq
                        if(k.ne.0)                      go to 805
        dspr(l)=rab2+rab3
        dspz(l)=(ra*rab2-ra*rab3)/wvno
                                                        go to 804
805     drvr(k,l)=rab2+rab3
        drvz(k,l)=(ra*rab2-ra*rab3)/wvno
804                                     continue
                                                        go to 700
811     rab2=(2.*xmu(i)*wvno*amplr-strsz)/(rho(i)*c*c*wvno)
        rab3=(del*rho(i)*wvno*amplz-strsr)/(rho(i)*c*c)
        dspr(l)=rab2-rab3*dz
        dspz(l)=-rab3/wvno
                        if(nderiv.lt.1)                 go to 700
        drvr(1,l)=-rab3
        drvz(1,l)=0.
                                                        go to 700
812     nderv=nderiv+1
        c1=(2.*xmu(i)*wvno*amplr-strsz)/(rho(i)*c*c*wvno)
        c2=(del*rho(i)*wvno*amplz-strsr)/(rho(i)*c*c*wvno)
                                do 806 ii=1,nderv
        k=ii-1
        rab1=ra**k
        q=ra*dz+k*3.1415926/2.
        cosq=cos(q)
        sinq=sin(q)
                        if(k.ne.0)                      go to 807
        dspr(l)=c1*cosq-wvno*c2*sinq/ra
        dspz(l)=-ra*c1*sinq/wvno-c2*cosq
                                                        go to 806
807     drvr(k,l)=(c1*cosq-wvno*c2*sinq/ra)*rab1
        drvz(k,l)=-(ra*c1*sinq/wvno+c2*cosq)*rab1
806                                     continue
c----------------------------------------------------------------------
c       Calculation of terms, connected with b-velocity
c----------------------------------------------------------------------
700     rab1=(c-b(i))/c
                        if(abs(rab1).lt.1.0e-5) rab1=0.
                        if(rab1) 710,711,712
710     nderv=nderiv+1
        q=rb*dz
        expq=exp(q)
        rab2=2.*xmu(i)*wvno*rb*amplz
        rab3=del*rho(i)*wvno*wvno*amplr
        rab4=2.*rho(i)*c*c*wvno*wvno
        c3=(rab2-rab3+wvno*strsz-rb*strsr)/rab4
        uttq=abs(c3/amplz)
                        if(uttq.lt.1.0e-5) c3=0.
        c4=(-rab2-rab3+wvno*strsz+rb*strsr)/rab4
                                do 704 ii=1,nderv
        k=ii-1
        rab1=rb**k
        rab2=rab1*c3*expq
        rab3=(-1)**k*rab1*c4/expq
                        if(k.ne.0)                      go to 705
        dspr(l)=dspr(l)+rab2+rab3
        dspz(l)=dspz(l)+(rab2-rab3)*wvno/rb
                                                        go to 704
705     drvr(k,l)=drvr(k,l)+rab2+rab3
        drvz(k,l)=drvz(k,l)+(rab2-rab3)*wvno/rb
704                                     continue
                                                        go to 801
711     rab2=(2.*xmu(i)*wvno*amplz-strsr)/(rho(i)*c*c*wvno)
        rab3=(del*rho(i)*wvno*amplr-strsz)/(rho(i)*c*c)
        dspz(l)=dspz(l)+rab2-rab3*dz
        dspr(l)=dspr(l)-rab3/wvno
                        if(nderiv.lt.1)                 go to 801
        drvz(1,l)=drvz(1,l)-rab3
                                                        go to 801
712     nderv=nderiv+1
        c3=(2.*xmu(i)*wvno*amplz-strsr)/(rho(i)*c*c*wvno)
        c4=(del*rho(i)*wvno*amplr-strsz)/(rho(i)*c*c*wvno)
                                do 706 ii=1,nderv
        k=ii-1
        rab1=rb**k
        q=rb*dz+k*3.1415926/2.
        cosq=cos(q)
        sinq=sin(q)
                        if(k.ne.0)                      go to 707
        dspz(l)=dspz(l)+c3*cosq-wvno*c4*sinq/rb
        dspr(l)=dspr(l)-rb*c3*sinq/wvno-c4*cosq
                                                        go to 706
707     drvz(k,l)=drvz(k,l)+(c3*cosq-wvno*c4*sinq/rb)*rab1
        drvr(k,l)=drvr(k,l)-(rb*c3*sinq/wvno+c4*cosq)*rab1
706                                     continue
801                                     continue
800       nwmax=0 
          if(n.eq.2) then
C----------liquid layer exists-------S
        do ixo=1,ndpth
        if(dspr(ixo).ne.0.0)then
C---------inside liquid-------S
          do jxo=1,ixo-1
         ququ=wvno*ra_1*dpth(jxo)
         xaxa=wvno*ra_1*d(1)
         aaa=sh_s(ququ)
         bbb=ch_s(ququ)
         ccc=ch_s(xaxa)
         am_z=bbb/ccc
         am_u=wvno/ra_1*aaa/ccc
         press=aaa/ccc/ra_1
C        PRint'(6(E10.4,1X))',1000./t,dpth(jxo),am_z,am_u,press,stresz(jxo)
         dspz(jxo)=aaa/ccc/ra_1*(wvno*c/a(1))**2
C        PRint '(7(E10.4,1X))',am_z,am_u,wvno,t/1000.,c,a(1)
          enddo
          nwmax=ixo-1
          go to 9555
                   endif        
C---------inside liquid-------E
          enddo
                   endif        
C----------liquid layer exists-------S
9555        RETURN
777                    continue
                                                                STOP
                                                                END
c///////////////////////////////////////////////////////////////////
c-------------------------------------------------------------------
        subroutine STRUT
c-------------------------------------------------------------------
        parameter (nsize=1000)
        common/st/ nst,accur,ds
        common/d/ a(nsize),b(nsize),rho(nsize),d(nsize),qs(nsize)
        common/c/ nmax,q1,q2,q3,q4,q5,q6,q7,ra_1
        dimension ds(nsize),r(nsize),rr(nsize)
c-------------------------------------------------------------------
c
c    ---- NST GT 0 ---- ---- IF B(1)=0   NMAX GT 0 --------
c
c-------------------------------------------------------------------
        d(nmax)=0.
             if(b(1).ge.0.1e-10)      goto 100
                                      do 1  i=1,nst
             if(ds(i).ge.d(1))      goto 100
        ds(i)=d(1)
1                                     continue
100                         continue
        mmax=nmax-1
             if(mmax.eq.0)      goto 101
        r(1)=d(1)
             if(mmax.eq.1)      goto 101
                                      do 2  i=2,mmax
        j=i-1
        r(i)=r(j)+d(i)
2                                     continue
101                         continue
        rr(1)=0.
        l=1
        m=1
        n=1
                                      do 3  i=1,98
             if(n.gt.nst)      goto 103
             if(m.gt.mmax)      goto 102
        h=ds(n)-r(m)
             if(h.ge.accur)      goto 103
102                         continue
        n=n+1
             if(l.gt.1)      goto 104
             if(abs(ds(n-1)).le.accur)      goto 3
             goto 106
104                         continue
        f=ds(n-1)-abs(rr(l))
             if(abs(f).le.accur)      goto 3
106                         continue
             if(m.gt.mmax)      goto 107
             if(abs(h).le.accur)      goto 103
107                         continue
        l=l+1
        rr(l)=ds(n-1)
             goto 3
103                         continue
             if(m.gt.mmax)      goto 105
        l=l+1
        rr(l)=-r(m)
        m=m+1
3                                     continue
105                         continue
        k=nmax
        rr(l+1)=rr(l)
                                      do 4  i=1,l
        j=l-i+1
        a(j)=a(k)
        b(j)=b(k)
        rho(j)=rho(k)
             if(rr(j).ge.0.0)      goto 108
        rr(j)=-rr(j)
        k=k-1
108                         continue
        d(j)=rr(j+1)-rr(j)
c	write(*,*) "ddddd ", d(j)
        
4                                     continue
        nmax=l
                                                        RETURN
                                                        END



