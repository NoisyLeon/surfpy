      SUBROUTINE MENU(char)
      IMPLICIT NONE
C----------------------------------------------------------------
c		char = 'v' ==> view menu only, else settings menu
        include "tomo.h" 
C----------------------------------------------------------------
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
	character char,chr 
	character*2 c
	character*55 text(10)
        integer*4 lc,ireg,n_pnt1,l1,l2
        logical LOGI
        integer*4 LNBLNK
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      data text/
     *'  7.) limits of the map (latitudes,step) ............',
     *'  8.) limits of the map (longitudes,step) ...........',
     *'  #.) coordinates of equator (latitudes)...........',
     *'  #.) coordinates of equator (longitudes)..........',
     *'  5.) regularization parameter 2 (alpha2).............',
     *'  6.) reg. param.1 (alpha1),  sigma1 & sigma2.........',
     *' 12.) Step of integration, Length of CUBE cells:......',
     *' 13.) Contour file name...............................',
     *' 14.) Model file name.................................',
     *' 15.) Gaussian cutting acuracy 10**(-pow), pow:.......'/

 1	print *,'		TOMO SETTINGS:'
	print *,'DATA   CHARACTERISTICS:'
	print *,' 0.) model are given?..................(toggle)...',imodel 
	print *,' 1.) weights are given?................(toggle)...',iwght 
	print *,' 2.) number of paths < ...........................',ipath 
	print *,' 3.) selection of paths ...............(toggle)...',isele 
	print *,'PLOT CHARACTERISTICS:'
	print *,' 4.) dens.& azim. coverage is produced?(toggle)...',ires
	print 1006,text(5),alph1(1)
	print 1007,text(6),alph1(3),alph1(2),alph1(4)
	print 1000,text(1),dlat0,dlat_n,s_lat
	print 1000,text(2),dlon0,dlon_n,s_lon
	print *,' 9.) map of deviations in %?...........(toggle)...',isymb
	print *,'10.) rejecting too strange data?.......(toggle)...',ireje
	print *,'11.) rejecting data by delta?..........(toggle)...',irejd
	print 1001,text(7),step0,cell
        l1=LNBLNK(fname1)
        l2=LNBLNK(fname2)
	print 1008,text(8),fname1(1:l1)
	print 1008,text(9),fname2(1:l2)
	print *,'MISCELLANEOUS CHARACTERISTICS:'
	print 1009,text(10),pow
	print *,'16.) output residuals?.................(toggle)...',iresid
	print *,'17.) apply geogr --> geoc for input....(toggle)...',icoord
	print *,'18.) apply geoc --> geogr for output...(toggle)...',icoout
	if(char.eq.'v') return
C---start of changing menu------------------------------------------------
 2	print 3
 3	format('Choice number (v to review menu; r,q, or x to return;go to run):')
	read(*,4) c
 4	format(a2)
1006    format(A55,f8.3)
1007    format(A55,f8.3,2f6.0)
1008    format(A55,a)
1009    format(A55,f5.1)
1000    format(A55,3F8.2)
1001    format(A55,F6.3,F7.4)
1002    format(A55,3F9.4)
1003    format(A55,2F6.1)
        lc=LNBLNK(c)
	if(c(1:lc).eq.'v') goto 1
	if(c(1:lc).eq.'r'.or.c(1:lc).eq.'q'.or.c(1:lc).eq.'x') return
	if(c(1:lc).eq.'0') then
	if(imodel.eq.1)then
	imodel=0
	else
	imodel=1
	end if
	else if(c(1:lc).eq.'1') then
	if(iwght.eq.1)then
	iwght=0
	else
	iwght=1
	end if
	elseif(c(1:lc).eq.'2') then
	 print *,'enter maximum number of paths:'
	 read(*,*) ipath     
	elseif(c(1:lc).eq.'3') then
	 if(isele.eq.0)then
	 isele =1      
	 else 
	 isele =0      
	   endif
	 if(isele.eq.1)     then
	 print* ,' Borders for rays are: '
	 print '(a12,2F10.2)',' Latitudes: ',slat0,slat_n
	 print '(a12,2F10.2)','Longitudes: ',slon0,slon_n
	 print *,'Any changes?(Y/N)'
	 read (*,'(a1)') chr
	 if(chr.eq.'Y'.or.chr.eq.'y')then
	 print*,'type slat0,slat_n,slon0,slon_n'
	 read (*,*) slat0,slat_n,slon0,slon_n
	                               endif  
	                     endif  
        elseif(c(1:lc).eq.'4') then
	 if(ires.eq.1)then
	 ires=0
	 else
	 ires=1
	 endif
	elseif(c(1:lc).eq.'5') then
	 ireg=1
	 print *,'enter regularization parameters'
	 read(*,*) alph1(1)
	elseif(c(1:lc).eq.'6') then
  	 print *,'enter regularization parameters'
	 read(*,*) alph1(3)
	 print *,'enter radius of correlation'
	 read(*,*) alph1(2)
	 print *,'enter MAX radius of correlation'
	 read(*,*) alph1(4)
	 if(alph1(4).lt.alph1(2)) then
	 alph1(4)=alph1(2)
	 print *,'WRANING !!! MAX sigma is less than initial, MAX=initial'
	 endif
	elseif(c(1:lc).eq.'7') then
	 print *,'enter limits for latitudes and increment'
	 read(*,*) dlat0,dlat_n,s_lat
	elseif(c(1:lc).eq.'8') then
	 print *,'enter limits for longitudes and increment'
	 read(*,*) dlon0,dlon_n,s_lon
	elseif(c(1:lc).eq.'9') then
	 if(isymb.eq.1) then
	 isymb=0
	 else
	 isymb=1
	 endif
	elseif(c(1:lc).eq.'10') then
	 if(ireje.eq.1) then
	 ireje=0
	 else
	 ireje=1
	 print *,'enter threshold in % for rejection'     
	 read(*,*) reject
	 endif
	elseif(c(1:lc).eq.'11') then
	 if(irejd.eq.1) then
	 irejd=0
	 else
	 irejd=1
	 print *,'enter number of wave periods for rejection'     
	 read(*,*) nwavep
	 endif
	elseif(c(1:lc).eq.'12') then
	 print *,'enter step of integration (degree)'
	 read(*,*) step0
	 print *,'enter length of cell (degree)'
	 read(*,*) cell
	 n_pnt1=SQRT(2.*pi/3.)*180./pi/2.
	 n_pnt1=n_pnt1/2
	 n_pnt=SQRT(2.*pi/3.)*180./pi/cell
	 n_pnt=n_pnt/2
	 wexp=FLOAT(2*n_pnt1 +1)/FLOAT(2*n_pnt+1)
	elseif(c(1:lc).eq.'13') then
	 print *,'enter CONTOUR file name'     
	 read(*,*) fname1
	elseif(c(1:lc).eq.'14') then
	 print *,'enter MODEL file name'     
	 read(*,*) fname2
	elseif(c(1:lc).eq.'15') then
	 print *,'enter pow, sign is alwaus +'
	 read(*,*) pow
	elseif(c(1:lc).eq.'16') then
	 if(iresid.eq.1) then
	 iresid=0
	 else
	 iresid=1
	 endif
	elseif(c(1:lc).eq.'17') then
	 if(icoord.eq.1) then
	 icoord=0
	 else
	 icoord=1
	 endif
	elseif(c(1:lc).eq.'18') then
	 if(icoout.eq.1) then
	 icoout=0
	 else
	 icoout=1
	 endif
	 endif
	 lmodel=LOGI(imodel)
	 lresid=LOGI(iresid)
	 lcoord=LOGI(icoord)
	 lcoout=LOGI(icoout)
	 lsele= LOGI(isele)
	 lwght= LOGI(iwght)
	 lres=  LOGI(ires)
	 lsymb= LOGI(isymb)
	 lreje= LOGI(ireje)
	 lrejd= LOGI(irejd)
	 go to 2
	end
C#####################################################
      LOGICAL FUNCTION LOGI(inp)
c TYPE definitions ++++++++++++++++++++++++++++++++++++++
      integer*4 inp
c END TYPE definitions ++++++++++++++++++++++++++++++++++
      logi=.false.
      if(inp.eq.1)LOGI=.true.
      return
      end
