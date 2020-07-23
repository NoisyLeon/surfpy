c****************************************************************
c* Print and edit program menu                                  *
c****************************************************************
      SUBROUTINE MENU(char)
      IMPLICIT NONE
c----------------------------------------------------------------
c char = 'v' ==> view menu only, else settings menu
c----------------------------------------------------------------
      include "tomo.h"
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      character char,chr
      character*2 c
      character*55 text(11)
      integer*4 lc,n_pnt1,l1
      logical logi
      integer*4 LNBLNK
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      data text/
     *'  5.) Limits of the map (latitudes,step) ............',
     *'  6.) Limits of the map (longitudes,step) ...........',
     *' 10.) Step,X-zone,R/L,G/P,Size of cells:iso,anis.....',
     *' 12.) 0PSI: alpha1, alpha2, sigma1, sigma2...........',
     *' 13.) 2PSI: alpha1,         sigma1, sigma2...........',
     *' 14.) 4PSI: alpha1,         sigma1, sigma2...........',
     *' 15.) 0PSI-prec, 2PSI-prec, 2&4PSI prec..............',
     *' 16.) Contour file name..............................',
     *' 17.) Model file name................................',
     *' 18.) Model name file for slowness determination ....',
     *' 23.) Anisotropy pole: lat, lon......................'/

 1    write(*,*) '		TOMO SETTINGS:'
      write(*,*) 'DATA   CHARACTERISTICS:'
      write(*,*) '  0.) Model are given?..................(toggle)...',imodel 
      write(*,*) '  1.) Weights are given?................(toggle)...',iwght 
      write(*,*) '  2.) Number of paths < ...........................',ipath 
      write(*,*) '  3.) Selection of paths ...............(toggle)...',isele 
      write(*,*) 'PLOT CHARACTERISTICS:'
      write(*,*) ' 4.) Path density & azim. coverage ?...(toggle)...',ires
      write(*,1000) text(1),dlat0,dlat_n,s_lat
      write(*,1000) text(2),dlon0,dlon_n,s_lon
      write(*,*) ' 7.) Map of deviations in %?...........(toggle)...',isymb
      write(*,*) ' 8.) Rejecting too strange data?.......(toggle)...',ireje
      write(*,*) ' 9.) Rejecting data by wavelength?.....(toggle)...',irejd
      write(*,1001) text(3),step0,x_zone,re_la,gr_pha,cell,cell_a
      write(*,*) '11.) anisotropy: 0-no,1-2psi,2-2&4psi,3-4psi......',ianiz+istrip
      write(*,1006) text(4),alsi(1,1),alsi(1,2),alsi(1,3),alsi(1,4)
      write(*,1007) text(5),alsi(2,1),          alsi(2,3),alsi(2,4)
      write(*,1007) text(6),alsi(3,1),          alsi(3,3),alsi(3,4)
      write(*,1008) text(7),alsi(1,5),alsi(2,5),alsi(3,5)
      l1=lnblnk(fname1)
      write(*,1009) text(8),fname1
      l1=lnblnk(fname2)
      write(*,1009) text(9),fname2
      l1=lnblnk(fname5)
      write(*,1009) text(10),fname5
      write(*,*) '19.) Output residuals?.................(toggle)...',iresid
      write(*,*) '20.) Covariance matrix?................(toggle)...',idresol
      write(*,*) '21.) Apply geogr --> geoc for input....(toggle)...',icoord
      write(*,*) '22.) Apply geoc --> geogr for output...(toggle)...',icoout
      write(*,1000) text(11),pf,pl
      write(*,*) '24.) Apply Fresnel zones?..............(toggle)...',itomo
      write(*,*) '25.) Produce resolution analysis map?..(toggle)...',iresol
      write(*,*) '26.) Use new input data format R1/R2?..(toggle)...',iold
      write(*,*) '27.) Use RAY TRACER mode?..............(toggle)...',itrace
      write(*,*) '28.) Make response map for some points?(toggle)...',ipoint
      write(*,*) '29.) Path density wts for iso and aniso parts.....',dump1,dump2
      if(char.eq.'v') return
c
c  start of changing menu------------------------------------------------
c
 2    write(*,3)
 3    format(
     +'Choice number (v to review menu; r,q, or x to return;go to run):',$)
      read(*,4) c
 4    format(a2)
1000  format(A55,3F8.2)
1001  format(A55,2F6.3,1x,2(a1,1x),3f6.3)
1006  format(A55,f8.3,f7.3,2f9.3)
1007  format(A55,f8.3,7x,2f9.3)
1008  format(A55,3f8.1)
1009  format(A55,a)
      lc=lnblnk(c)
      if(c(1:lc).eq.'v') goto 1
      if(c(1:lc).eq.'r'.or.c(1:lc).eq.'q'.or.c(1:lc).eq.'x') return
      if(c(1:lc).eq.'0') then
        if(imodel.eq.1)then
          imodel=0
        else
          imodel=1
        endif
      else if(c(1:lc).eq.'1') then
        if(iwght.eq.1)then
          iwght=0
        else
          iwght=1
        endif
      else if(c(1:lc).eq.'2') then
        write(*,*) 'enter maximum number of paths:'
        read(*,*) ipath     
      else if(c(1:lc).eq.'3') then
        if(isele.eq.0)then
          isele =1      
        else 
          isele =0      
        endif
        if(isele.eq.1)     then
          write(*,*) ' Borders for rays are: '
          write(*,'(a12,2F10.2)') ' Latitudes: ',slat0,slat_n
          write(*,'(a12,2F10.2)') 'Longitudes: ',slon0,slon_n
          write(*,*) 'Any changes?(Y/N)'
          read (*,'(a1)') chr
          if(chr.eq.'Y'.or.chr.eq.'y')then
            write(*,*) 'type slat0,slat_n,slon0,slon_n'
            read (*,*) slat0,slat_n,slon0,slon_n
          endif  
        endif  
      else if(c(1:lc).eq.'4') then
        if(ires.eq.1)then
          ires=0
        else
          ires=1
        endif
      else if(c(1:lc).eq.'5') then
        write(*,*) 'enter limits for latitudes and increment'
        read(*,*) dlat0,dlat_n,s_lat
      else if(c(1:lc).eq.'6') then
        write(*,*) 'enter limits for longitudes and increment'
        read(*,*) dlon0,dlon_n,s_lon
      else if(c(1:lc).eq.'7') then
        if(isymb.eq.1) then
          isymb=0
        else
          isymb=1
        endif
      else if(c(1:lc).eq.'8') then
        if(ireje.eq.1) then
          ireje=0
        else
          ireje=1
          write(*,*) 'enter threshold in % for rejection'     
          read(*,*) reject
        endif
      else if(c(1:lc).eq.'9') then
        if(irejd.eq.1) then
          irejd=0
        else
          irejd=1
          write(*,*) 'enter number of wavelength for rejection'     
          read(*,*) nwavep
        endif
      else if(c(1:lc).eq.'10') then
        write(*,*) 'Step of integration'
        read(*,*) step0 
        write(*,*) 'X-zone'
        read(*,*) x_zone
        write(*,*) 'enter type of wave (Rayleigh, Love): R or L)'
        read(*,'(a1)') re_la
        write(*,*) 'enter type of velocity (Group, Phase): G or P)'
        read(*,'(a1)') gr_pha
        write(*,*) 'enter length of main cell (degree)'
        read(*,*) cell
        n_pnt1=SQRT(2.*pi/3.)*180./pi/2.
        n_pnt1=n_pnt1/2
        n_pnt=SQRT(2.*pi/3.)*180./pi/cell
        n_pnt=n_pnt/2
        wexp=FLOAT(2*n_pnt1 +1)/FLOAT(2*n_pnt+1)
        write(*,*) 'enter length of anisotpopy cell (degree)'
        read(*,*) cell_a
        n_pnt1=SQRT(2.*pi/3.)*180./pi/2.
        n_pnt1=n_pnt1/2
        na_pnt=SQRT(2.*pi/3.)*180./pi/cell_a
        na_pnt=na_pnt/2
      else if(c(1:lc).eq.'11') then
        istrip = 0
        write(*,*) 'enter type of anisotropy: 0 - no, 1 - 2psi, 2 - 2&4psi'
        read(*,*) ianiz
        if(ianiz.ne.0.and.ianiz.ne.1.and.ianiz.ne.2.and.ianiz.ne.3) then
          write(*,*) 'BAD type of anisotropy,asumed no anisotropy'
          ianiz=0
        endif
        if(ianiz.eq.3) then
          ianiz = 1
          istrip = 2
        endif
      else if(c(1:lc).eq.'12') then
        write(*,*) 'enter regularization parameters for 0PSI'
        read(*,*) alsi(1,1),alsi(1,2),alsi(1,3),alsi(1,4)
      else if(c(1:lc).eq.'13') then
        write(*,*) 'enter regularization parameters for 2PSI'
        read(*,*) alsi(2,1),          alsi(2,3),alsi(2,4)
      else if(c(1:lc).eq.'14') then
        write(*,*) 'enter regularization parameters for 4PSI'
        read(*,*) alsi(3,1),          alsi(3,3),alsi(3,4)
      else if(c(1:lc).eq.'15') then
        write(*,*) 'enter acuracy for 0PSI, 2PSI, 4PSI'
        read(*,*) alsi(1,5),alsi(2,5),alsi(3,5)
      else if(c(1:lc).eq.'16') then
        write(*,*) 'enter CONTOUR file name'
        read(*,*) fname1
      else if(c(1:lc).eq.'17') then
        write(*,*) 'enter MODEL file name'
        read(*,*) fname2
      else if(c(1:lc).eq.'18') then
        write(*,*) 'enter MODEL file name for slowness determinatrion'
        read(*,*) fname5
      else if(c(1:lc).eq.'19') then
        if(iresid.eq.1) then
          iresid=0
        else
          iresid=1
       endif
      else if(c(1:lc).eq.'20') then
        if(idresol.eq.1) then
          idresol=0
        else
          idresol=1
        endif
      else if(c(1:lc).eq.'21') then
        if(icoord.eq.1) then
          icoord=0
        else
          icoord=1
        endif
      else if(c(1:lc).eq.'22') then
        if(icoout.eq.1) then
          icoout=0
        else
          icoout=1
        endif
      else if(c(1:lc).eq.'23') then
        write(*,*) 'enter pole lat and lon '
        read(*,*) pf,pl 
      else if(c(1:lc).eq.'24') then
        if(itomo.eq.1) then
          itomo=0
        else
          itomo=1
        endif
      else if(c(1:lc).eq.'25') then
        if(iresol.eq.1)then
          iresol=0
        else
          iresol=1
        endif
      else if(c(1:lc).eq.'26') then
        if(iold.eq.1)then
          iold=0
        else
          iold=1
        endif
      else if(c(1:lc).eq.'27') then
        if(itrace.eq.1)then
          itrace=0
        else
          itrace=1
        endif
      else if(c(1:lc).eq.'28') then
        if(ipoint.eq.1)then
          ipoint=0
        else
          ipoint=1
        endif
      else if(c(1:lc).eq.'29') then
        write(*,*) 'enter coefficients for dumping: iso, ani'
        read(*,*) dump1,dump2
      endif
      lmodel= logi(imodel)
      ldresol=logi(idresol)
      lsele=  logi(isele)
      lsele=  logi(isele)
      lwght=  logi(iwght)
      lres=   logi(ires)
      lsymb=  logi(isymb)
      laniz=  logi(ianiz)
      lreje=  logi(ireje)
      lrejd=  logi(irejd)
      lresid= logi(iresid)
      lcoord= logi(icoord)
      lcoout= logi(icoout)
      lresol= logi(iresol)
      ltomo=  logi(itomo)
      lold=   logi(iold)
      ltrace= logi(itrace)
      lpoint= logi(ipoint)
      goto 2
      end
c**************************************
      logical function logi(inp)
c TYPE definitions ++++++++++++++++++++
      integer*4 inp
c END TYPE definitions ++++++++++++++++
      logi=.false.
      if(inp.ge.1)logi=.true.
      return
      end
