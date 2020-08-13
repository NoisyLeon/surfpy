
#f2py -m aftan aftanipg.f aftanpg.f fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f tapers.f  tgauss.f  trigger.f myfftw3.h -h aftan.pyf
#f2py -h aftan.pyf -m aftan aftanipg.f aftanpg.f fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f tapers.f  tgauss.f  trigger.f myfftw3.h

f2py -c aftan.pyf aftanipg.f aftanpg.f fmax.f  ftfilt.f  mspline.f  phtovel.f  pred_cur.f  taper.f tapers.f  tgauss.f  trigger.f myfftw3.h -lfftw3 --f77flags=-ffixed-line-length-none --fcompiler=gfortran

