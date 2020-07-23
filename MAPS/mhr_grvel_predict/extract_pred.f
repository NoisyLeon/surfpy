	     character*80 infile,inlist
             character*5 word1
             character*4 word2
             character*5 word4
             character*40 word3
             character*4 stan
             real*4 t(50),ur(50),ul(50)
	     data marg/1/
	     narg=iargc()
	     if(narg.ne.marg)STOP'USAGE: extract_pred inlist '
             call GETARG(1,inlist)
             open(7,file=inlist,status='OLD')
             DO j=1,500
             read(7,'(a)',end=999)infile 
             lin=lnblnk(infile)
	     open(1,file=infile(1:lin)//'/GSN/PREDICTION_R',status='OLD')
	     open(2,file=infile(1:lin)//'/GSN/PREDICTION_L',status='OLD')
             word1=infile(1:5)
             word2=infile(lin-3:lin)
             word4=word2(1:2)//'_'//word2(3:4)
             word3=infile(6:lin-4)
             lw3=lnblnk(word3)
1            read(1,'(2F10.3,2X,a4,7X,i3)', end=99)t(1),ur(1),stan,n
             read(2,'(2F10.3,2X,a4,7X,i3)', end=99)t(1),ul(1),stan,n
             do i=2,n
             read(1,*,end=99)t(i),ur(i)
             read(2,*,end=99)t(i),ul(i)
             enddo
             lsta=lnblnk(stan)
	     open(3,file=stan(1:lsta)//'_'//word1//'.'//word4//word3(1:lw3)//'R_PRED')
	     open(4,file=stan(1:lsta)//'_'//word1//'.'//word4//word3(1:lw3)//'L_PRED')
	     do i=1,n
             write(3,'(2F10.3)')t(i),ur(i)
             write(4,'(2F10.3)')t(i),ul(i)
             enddo
	     read(1,'(1X)',end=99)
	     read(1,'(1X)',end=99)
	     read(2,'(1X)',end=99)
	     read(2,'(1X)',end=99)
             go to 1
99           close(1)
             close(2)
             close(3)
             close(4)
             endDO
999          print *,'n_files=',j-1
             end
