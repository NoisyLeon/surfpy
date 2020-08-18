#!/bin/bash
#rm -f fast_surf.pyf
#f2py fast_surf.f flat1.f init.f calcul.f surfa.f mchdepsun.f -m fast_surf -h fast_surf.pyf
#cp temp/fast_surf.pyf ./
#f2py -c fast_surf.pyf fast_surf.f flat1.f init.f calcul.f surfa.f mchdepsun.f -lfftw3 --f77flags=-ffixed-line-length-none --fcompiler=gfortran
f2py -c --f77flags="-ffixed-line-length-none -O3" --f90flags="-O3" --fcompiler=gfortran fast_surf.pyf fast_surf.f flat1.f init.f calcul.f surfa.f mchdepsun.f 
#cp fast_surf.so ..
