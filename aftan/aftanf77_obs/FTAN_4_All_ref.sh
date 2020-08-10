#!/bin/bash

# Run FTAN for all the stacked cross-correlations


FindRef() {
    # find reference Rayleigh wave group and phase velocity model
    sta_dep="/work3/wang/JdF/station_7D_deps.lst"
    Pred_dir="/work3/wang/code_bkup/FTAN_Ye/Pred_DISP"
    dep_threshold=1500.
    if [ -f ${Pred_dir}"/"$1"_"$2".R.phv" ]; then
        DispPhv=${Pred_dir}"/"$1"_"$2".R.phv"
        DispGrv=${Pred_dir}"/"$1"_"$2".R.grv"
    else
        # deep--3, shallow--1, continent--0
        out1=`grep $1 ${sta_dep}` && depth1=`echo ${out1} | awk '{print $NF}'`|| depth1=-100.
        out2=`grep $2 ${sta_dep}` && depth2=`echo ${out2} | awk '{print $NF}'`|| depth2=-100.
        if (( $(echo ${depth1} "< 0." | bc -l) )); then
            type1=0
        else
            (( $(echo ${depth1} ">" ${dep_threshold} | bc -l) )) && type1=3 || type1=1
        fi
        if (( $(echo ${depth2} "< 0." | bc -l) )); then
            type2=0
        else
            (( $(echo ${depth2} ">" ${dep_threshold} | bc -l) )) && type2=3 || type2=1
        fi
        type_cc=`echo ${type1}"+"${type2} | bc -l`
        if [ "$type_cc" -eq 0 ]; then
            DispPhv=${Pred_dir}"/Cont_Cont.R.phv"
            DispGrv=${Pred_dir}"/Cont_Cont.R.grv"
        elif [ "$type_cc" -eq 1 ]; then
            DispPhv=${Pred_dir}"/Cont_Shal.R.phv"
            DispGrv=${Pred_dir}"/Cont_Shal.R.grv"
        elif [ "$type_cc" -eq 3 ]; then
            DispPhv=${Pred_dir}"/Cont_Deep.R.phv"
            DispGrv=${Pred_dir}"/Cont_Deep.R.grv"
        elif [ "$type_cc" -eq 2 ]; then
            #DispPhv="Shallow-Shallow doesnt exist"
            DispPhv=${Pred_dir}"/Shal_Deep.R.phv"
            DispGrv="Shallow-Shallow doesnt exist"
        elif [ "$type_cc" -eq 4 ]; then
            DispPhv=${Pred_dir}"/Shal_Deep.R.phv"
            DispGrv=${Pred_dir}"/Shal_Deep.R.grv"
        elif [ "$type_cc" -eq 6 ]; then
            DispPhv=${Pred_dir}"Deep_Deep.R.phv"
            DispGrv=${Pred_dir}"Deep_Deep.R.grv"
        else
            "echo Station types wrong!"
        fi
    fi
}

RunFTAN() {
    FTANexe=/work3/wang/code_bkup/FTAN_Ye/aftani_c_pgl_amp
    sta1=`echo $1 | awk -F'_' '{print $2}'`
    sta2=`echo $1 | awk -F"[_.]" '{print $3}'`
    FindRef ${sta1} ${sta2}
    param=$2
    echo $param" "$1" 1" > ${sacf}_param_R.dat
    echo $1 $DispPhv $DispGrv
    $FTANexe $1_param_R.dat $DispPhv $DispGrv $3
    rm -f $1_param_R.dat
    #rm -f $1_1_AMP
    rm -f $1_2_AMP
    rm -f $1_cld
    if [ ! -d "./${sta1}" ]; then
        mkdir ${sta1}
    fi
    mv $1* ${sta1}
}


for file in `cat /work3/wang/JdF/Denoised_Stack/XCORR_SAC.lst`
do
    sacf=`echo ${file} | awk -F'/' '{print $NF}'`
    ln -s ${file} ./${sacf}
    RunFTAN ${sacf} '-1 0.15 4.5 0.8 40. 30. 2.0 3. 0.99 12. 15. 1.5 3.0 0.6 3.' 0.08 #1>/dev/null
done
wait
cd /work3/wang/JdF/FTAN_4_All
