c spectra tapering procedure
cxx function [omstart,inds,inde, omdom, ampdom] = tapers(omb,ome,dom,alpha,ns)
      subroutine tapers(omb,ome,dom,alpha,ns,
     *                  omstart, inds, inde, omdom, ampdom)
      implicit none
      integer*4 ns, i, om1, om2, om3, om4, inds, inde
      real*8    pi, tresh, omb, ome, dom, alpha, omstart, wd
      real*8    om2d,om3d,ampdom(ns), omdom(ns)

      pi = datan(1.0d0)*4.0d0
c     tresh = 2.0d0
      tresh = 0.5d0
cxx ampdom = zeros(ns,1);
      do i =1,ns
          ampdom(i) = 0.0d0
      enddo
      om2d = omb/dom
      wd = max(16.0d0,om2d*dsqrt(tresh/alpha))
      om1 = nint(max(1.0d0,om2d-wd/2.0d0))
      om2 = nint(min(ns*1.0d0,om1+wd))
cxx ampdom = (1-cos(pi/(om2-om1)*([om1:om2]-om1)))/2;
c     write(*,*) 'Y',omb,ome,dom,alpha,ns
      do i = om1,om2
          ampdom(i) = (1.0d0-dcos(pi/(om2-om1)*(i-om1)))/2.0d0
      enddo
      om3d = ome/dom
      wd = max(16.0d0,om3d*dsqrt(tresh/alpha))
      om4 = nint(min(ns*1.0d0,om3d+wd/2))
      om3  = nint(max(1.0d0,om4-wd))
c     write(*,*) om1, om2, om3, om4
cxx amp2 = (1+cos(pi/(om4-om3)*([om3:om4]-om3)))/2;
      do i = om3,om4
          ampdom(i) = (1.0d0+dcos(pi/(om4-om3)*(i-om3)))/2.0d0
      enddo
cxx ampdom([om2:om3]) = ones(max(size([om2:om3])),1);
      do i = om2,om3
          ampdom(i) = 1.0d0
      enddo
cxx ind = [om1:om4];
cxx omdom = (ind-1)*dom;
      do i = 1,ns
          omdom(i) = (i-1)*dom
      enddo
      omstart = omb;
      inds = om1
      inde = om4
      return
      end
