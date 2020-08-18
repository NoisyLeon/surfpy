#!/bin/bash
#f2py -h theo.pyf -m theo four1.f theo.f qlayer.f 
f2py --f77flags="-O3 -ffixed-line-length-none" --f90flags="-O3" --fcompiler=gfortran -c theo.pyf four1.f theo.f qlayer.f 
#f2py --f77flags="-ffixed-line-length-none -O3" --f90flags="-O3" --fcompiler=gfortran -c theo.pyf four1.f theo.f qlayer.f 
#cp theo.so ..
