
# #   0.1000E+01                dt 
# #     7   -1   -2             npts    n1(user spcified or not, when < 0: user specified)  n2
# # out.mod                     mname
# #          T         T        dolove  dorayl
# #   0.0000E+00  0.0000E+00    hs      hr
# #     1                       nmodes
# #   0.5000E+01  0.5000E+01    faclov  facrayl
# #   0.10000000E+00
# #   0.66666670E-01
# #   0.50000001E-01
# #   0.39999999E-01
# #   0.33333335E-01
# #   0.28571429E-01
# #   0.25000000E-01

# input
# ilvry : love(1) or rayleigh(2)
# nn - npts
# mname - model name
# verby - screen output or not
# nfval - npts
# fval  - input frequencies (array)
# ccmin, ccmax - min/max phase vel

# output 
# iret


#    subroutine disprs(ilvry,dt,nn,iret,mname,
# 1          verby,nfval,fval,ccmin,ccmax)

import tdisp96
import vmodel
import numpy as np
model=vmodel.Model1d(modelindex=2)
model.read('test_model.txt')
d_in    = model.HArr
TA_in   = model.rhoArr * model.VphArr**2
TC_in   = model.rhoArr * model.VpvArr**2
TL_in   = model.rhoArr * model.VsvArr**2
TF_in   = 1.0 * (TA_in - np.float32(2.)* TL_in)
TN_in   = model.rhoArr * model.VshArr**2
TRho_in = model.rhoArr


ilvry = 1
dt = 1.
npts = 7
iret =1
verby = True
nfval = 7
fval  = 1./(np.arange(7)*5.+10.)
fval  = np.append(fval, np.zeros(2049-nfval))
ccmin = -1.
ccmax = -1.
nl_in = TRho_in.size
iflsph_in = 1 # 0 flat, 1: love, 2: rayleigh
refdep_in = 0.
nmode = 1
# 
# 
c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out=tdisp96.disprs(ilvry,dt,npts,iret,verby, nfval,fval,ccmin,ccmax,\
               d_in,TA_in,TC_in,TF_in,TL_in,TN_in,TRho_in,\
               nl_in, iflsph_in, refdep_in, nmode, 0.5, 0.5)

d_out1,TA_out1,TC_out1,TF_out1,TL_out1,TN_out1,TRho_out1=tdisp96.flat2sphere(ilvry,dt,npts,iret,verby, nfval,fval,ccmin,ccmax,\
               d_in,TA_in,TC_in,TF_in,TL_in,TN_in,TRho_in,\
               nl_in, refdep_in, nmode, 0.5, 0.5)


# real*8 dimension(nl_in),depend(nl_in) :: d_in
# real*8 dimension(nl_in),depend(nl_in) :: ta_in
# real*8 dimension(nl_in),depend(nl_in) :: tc_in
# real*8 dimension(nl_in),depend(nl_in) :: tf_in
# real*8 dimension(nl_in),depend(nl_in) :: tl_in
# real*8 dimension(nl_in),depend(nl_in) :: tn_in
# real*8 dimension(nl_in),depend(nl_in) :: trho_in
# integer, intent(in) :: nl_in

