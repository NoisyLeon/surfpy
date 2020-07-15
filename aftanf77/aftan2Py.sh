#f2py -c aftanipg.f  aftanpg.f  driver.f  fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f  tapers.f  tgauss.f  trigger.f -m Aftan -I/projects/life9360/.local/fftw-3.3.4/include -L/projects/life9360/.local/fftw-3.3.4/lib

#f2py -c aftanipg.f  driver.f  fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f  tapers.f  tgauss.f  trigger.f -m Aftan -lfftw3 --f77flags=-g -O -Wall -ffixed-line-length-none --fcompiler=gfortran
#module unload intel
#module load fftw/fftw-3.3.4_openmpi-1.8.2_gcc-4.9.1_double
#module load gcc/gcc-4.9.1

#f2py -h aftan.pyf -m aftan aftanipg.f aftanpg.f fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f tapers.f  tgauss.f  trigger.f 
f2py -c aftan.pyf aftanipg.f aftanpg.f fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f tapers.f  tgauss.f  trigger.f -lfftw3 --f77flags=-ffixed-line-length-none --fcompiler=gfortran
cp aftan.so ..
#f2py -c pyaftan.pyf -m pyaftan aftanipg.f aftanpg.f fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f tapers.f  tgauss.f  trigger.f 
#f2py -c aftanipg.f aftanpg.f fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f tapers.f  tgauss.f  trigger.f -m pyaftan -lfftw3 --f77flags=-ffixed-line-length-none --fcompiler=gfortran

#module load intel/intel-13.0.0 

#cp pyaftan.so /projects/life9360/.local/lib
