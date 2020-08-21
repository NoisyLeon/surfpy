	     character*80 infile,outfile
	     character*6 name1,name2     
             real*4 d(4)
	     data marg/2/
	     narg=iargc()
	     if(narg.ne.marg)STOP'USAGE: stupid infile outfile'
             call GETARG(1,infile)
             call GETARG(2,outfile)
	     open(1,file=infile,status='OLD')
	     open(2,file=outfile)
	     DO i=1,100000
	     read(1,*,end=99) n1,n2,idum,name1,name2,d 
	     write(2,*) n1,n2,' ',name1,name2,d
             do j=1,21
             read(1,'(1X)',end=99)
	     enddo
             endDO
99           np=i-1
	     print *,'npaths=',np
             end
