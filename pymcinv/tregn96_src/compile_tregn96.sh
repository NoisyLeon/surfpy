####################
## copy subs code
####################
cps_subs_dir='/home/leon/code/PROGRAMS.330/SUBS'

#cp $cps_subs_dir/tgetmod.f .
#cp $cps_subs_dir/mnmarg.f .
#cp $cps_subs_dir/mgtarg.f .
#cp $cps_subs_dir/mchdep.f .
#cp $cps_subs_dir/lgstr.f .


#f2py -h tregn96.pyf -m tregn96 tregn96_subroutine.f mnmarg.f mgtarg.f lgstr.f tio.f mchdep.f  tgetmod.f
f2py --f77flags="-ffixed-line-length-none -O3" --f90flags="-O3" -c tregn96.pyf tregn96_subroutine.f mnmarg.f mgtarg.f lgstr.f tio.f mchdep.f tgetmod.f

#cp tregn96.so ..
