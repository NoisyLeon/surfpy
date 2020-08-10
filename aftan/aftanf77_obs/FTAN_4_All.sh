#!/bin/bash

# Run FTAN for all the stacked cross-correlations


RunFTAN() {
    FTANexe=/work3/wang/code_bkup/FTAN_Ye/aftani_c_pgl_amp
    sacf=$1
    param=$2
    DispGrv="I_dont_believe_this_file_exists"
    DispPhv="./OBS_phvel.dat"
    echo $param" "$sacf" 1" > ${sacf}_param_R.dat
    $FTANexe ${sacf}_param_R.dat $DispPhv $DispGrv $3
    rm -f ${sacf}_param_R.dat
}


for file in `cat /work3/wang/JdF/Denoised_Stack/XCORR_SAC.lst`
do
    sacf=`echo ${file} | awk -F'/' '{print $NF}'`
    ln -s ${file} ./${sacf}
    RunFTAN ${sacf} '-1 0.15 4.5 0.8 40. 30. 2.0 3. 0.99 12. 15. 1.5 3.0 0.6 3.' 0.08 1>/dev/null
done
wait
cd /work3/wang/JdF/FTAN_4_All
