c*********************************************************
c*  write vector on hard disk                            *
c*********************************************************
      SUBROUTINE WRITE_BIN(nm,arr,filen)
      implicit none
      integer*4 nm
      real*4    arr(nm*nm)
      character*160 filen
      integer*4 i,ie,l
c
c   main loop
c
      open(25,file=filen,form='unformatted',status='unknown')
      do i=1,nm*nm,nm
         ie=i+nm-1
         write(25) (arr(l),l=i,ie)
      enddo
      close(25)
      return
      end
c*********************************************************
c*   read vector from hard disk                          *
c*********************************************************
      SUBROUTINE READ_BIN(nm,arr,filen)
      implicit none
      integer*4 nm
      real*4    arr(nm*nm)
      character*160 filen
      integer*4 i,ie,l
c
c   main loop
c
      open(25,file=filen,form='unformatted',status='old')
      do i=1,nm*nm,nm
         ie=i+nm-1
         read(25) (arr(l),l=i,ie)
      enddo
      close(25)
      return
      end
c*************************************************************
c* Compute A = A * A , where A (nm x nm) simmetric matrix    *
c* located in memory as one dimensional array raw by raw.    *
c* Result is placed in the same array.                       *
c* Input matrix is not saved.                                *
c*************************************************************
      SUBROUTINE MAT_SQ2(nm,arr)
      implicit none
      integer*4 nm
      real*4 arr(nm*nm)
      integer*4 i,j,k,l1,l2
      real*4 s
c
c      Upper diagonal elements
c
      do i=1,nm-1
      do j=i+1,nm
         s=0.0
         do k=1,nm
           if(k.ge.i) l1=(i-1)*nm+k
           if(k.lt.i) l1=(k-1)*nm+i
           if(k.ge.j) l2=(j-1)*nm+k
           if(k.lt.j) l2=(k-1)*nm+j
           s=s+arr(l1)*arr(l2)
         enddo
         arr((j-1)*nm+i)=s
      enddo
      enddo
c
c       Diagonal elements
c
      do i=1,nm
         s=0.0
         do k=1,nm
           if(k.ge.i) l1=(i-1)*nm+k
           if(k.lt.i) l1=(k-1)*nm+i
           s=s+arr(l1)*arr(l1)
         enddo
         arr((i-1)*nm+i)=s
      enddo
c
c        Replase upper diagonal with results
c
      do i=1,nm-1
      do j=i+1,nm
         arr((i-1)*nm+j)=arr((j-1)*nm+i)
      enddo
      enddo
      return
      end
c**************************************************************
c* Compute A = A * B , where A,B (nm x nm) simmetric matrix   *
c* located in memory as one dimensional array raw by raw.     *
c* Result is placed in same array. Input matrix is not saved. *
c**************************************************************
      SUBROUTINE MAT_MULT(nm,arr,filen,filen1)
      implicit none
      include "tomo.h"
      integer*4 nm
      real*4 arr(nm*nm)
      real*4 vect(NRAZ+NRAZA*(NAZIPL-1)),res(NRAZ+NRAZA*(NAZIPL-1)),rr
      integer*4 nlidd,lidd(NRAZ+NRAZA*(NAZIPL-1))
      character*160 filen,filen1
c     character*80 pstr
      integer*4 i,j,k,kk,l,l2
      real*4 s
c
c      open file with matrix B
c
      open(25,file=filen,form='unformatted',status='old')
      open(27,file=filen1,form='unformatted',status='unknown')
c
c      Upper diagonal elements
c
      do i=1,nm
         read(25) (vect(l),l=1,nm)
         nlidd=0
         do k=1,nm
           if(vect(k).ne.0.0) then
             nlidd=nlidd+1
             lidd(nlidd)=k
           endif
         enddo
      do j=1,nm
         s=0.0
         do kk=1,nlidd
           k=lidd(kk)
           l2=(j-1)*nm+k
           s=s+vect(k)*arr(l2)
         enddo
         res(j)=s
      enddo
      write(27) (res(l),l=1,nm)
      enddo
      close(25)
      close(27)
c     pstr='Part1     '
c     CALL PSTIME(pstr)
c
c     Read matrix again
c
      CALL READ_BIN(nm,arr,filen1)
c
c     Replace upper diagonal with results
c
      do i=1,nm-1
      do j=i+1,nm
         rr=arr((i-1)*nm+j)
         arr((i-1)*nm+j)=arr((j-1)*nm+i)
         arr((j-1)*nm+i)=rr
      enddo
      enddo
c     pstr='Part2     '
c     CALL PSTIME(pstr)
      return
      end
