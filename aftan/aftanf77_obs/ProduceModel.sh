#!/bin/bash
set -e

### Set input paths for maps and models
Initialize() {
   stalst=stations.lst
   AgeMap=/work2/tianye/ASN_OBS_BHZ/PaperResults/DISP_Pred/Age/age.3.6.xyz.HD
   BathyMap=/work2/tianye/ASN_OBS_BHZ/PaperResults/DISP_Pred/Bathymetry/sea_floor.CICE.HD
   SedThk1Map=/work2/tianye/ASN_OBS_BHZ/PaperResults/DISP_Pred/Sediments/map1.thk1_JDF.HD
   SedThk2Map=/work2/tianye/ASN_OBS_BHZ/PaperResults/DISP_Pred/Sediments/map1.thk2_JDF.HD
   fCrust=Crust_Qmiu
   dirMantle=/work2/tianye/ASN_OBS_BHZ/PaperResults/DISP_Pred/Mantle_Q_extrapolate
   dirQmiu=/work2/tianye/ASN_OBS_BHZ/PaperResults/DISP_Pred/Qmiu_30
   #SedVs1Map=../Sediments/map1.vs1_JDF.HD  The SedVs maps from global model turn out to be not useful
   #SedVs2Map=../Sediments/map1.vs2_JDF.HD  at all. Will use Vs = 1. for all sediments
   ### C code that computes path averages from a map
   PathAvgexe=/home/tianye/code/Programs/MapOperation/PathAvg_Accurate/PathAvg

   perl=0.5 
   perh=45.
   rangeper=44.5
   perl2=2.
   fcleanup=filetobedel.txt
}

### Pick source station 
Pickssta() {
   if [ $ssta != $1 ]; then
      let ista++
      continue
   fi
}
### get info of source station
Getssta() {
   sinfo=`awk -v ista=$ista 'NR==ista{print $1"_"$2"_"$3}' $stalst`
   ssta=`echo $sinfo | cut -d_ -f1`
   slon=`echo $sinfo | cut -d_ -f2`
   slat=`echo $sinfo | cut -d_ -f3`
   #Pickssta 'CM01A'
}

### get info of receiver station
Pickrsta() {
   if [ $rsta != $1 ]; then
      continue
   fi
}
Getrsta() {
   rsta=`echo $rinfo | cut -d_ -f1`
   rlon=`echo $rinfo | cut -d_ -f2`
   rlat=`echo $rinfo | cut -d_ -f3`
   #Pickrsta 'FS02B'
}

### compute age, bathymetry, and the 2-layer sedimentary thk and vs
ComputeAll() {
   elamda=1.
   rinfile='rinfile.tmp'
   echo $rlon $rlat > $rinfile
   age=`$PathAvgexe $slon $slat $rinfile $AgeMap $elamda | awk '{printf "%.1f", $3}'`
   bathy=`$PathAvgexe $slon $slat $rinfile $BathyMap $elamda | awk '{print -$3/1000.}'`
   sedthk1=`$PathAvgexe $slon $slat $rinfile $SedThk1Map $elamda | awk '{printf "%f", $3}'`
   sedthk2=`$PathAvgexe $slon $slat $rinfile $SedThk2Map $elamda | awk '{printf "%f", $3}'`
   rm $rinfile
   echo $age $bathy $sedthk1 $sedthk2
   #sedvs1=`$PathAvgexe $SedVs1Map $slon $slat $rlon $rlat $elamda`
   #sedvs2=`$PathAvgexe $SedVs2Map $slon $slat $rlon $rlat $elamda`
   for pavg in $age $bathy $sedthk1 $sedthk2; do
      if [ `echo $pavg | awk '{if($1==-12345.){print 1}else{print 0}}'` -eq 1 ]; then
      #if (( $(bc <<< "$pavg == -12345.") == 1 )); then
         echo "   Cannot get age/bathy/sedthk measurement. Skipped!"
	 break 2 # break the outer loop
      fi
   done #check pavg
   sedthk=`echo $sedthk1 $sedthk2 | awk '{print $1+$2}'`
   sedvs=1
}

### add mantle structure from fMantle underneath the input model file and merge Qmiu from fQmiu
MergeMantleQ() {
   ## 1:thk0 2:fMantle 3:fQmiu 4:fout
   NR=1
   depdone=0
   if (( $(echo "$1 < 0." | bc -l) )); then
      echo "invalid mantle depth"
      continue
   fi
   while read depm vel; do
      if [ $NR -eq 1 ]; then
	 if [ `echo $depm | awk '{if($1==0.){print 0}else{print 1}}'` != 0  ]; then
	    echo "format error in "$2
	    exit
	 fi
      else
	 thklast=`echo $depmlast $depm $depdone | awk '{print ($1+$2)/2-$3}'`
	 Qmiulast=`awk -v thk0=$1 -v depm=$depmlast 'BEGIN{dep=thk0+depm; deplast=0; Qlast=344898}{if($1>=dep){print Qlast+(dep-deplast)*($2-Qlast)/($1-deplast); exit} deplast=$1;Qlast=$2}' $3`
	 echo $thklast $vellast $Qmiulast >> $4
	 depdone=`echo $depdone $thklast | awk '{print $1+$2}'`
      fi
      depmlast=$depm
      vellast=$vel
      let NR++
   done < $2
   thklast=`echo $depmlast $depdone | awk '{print $1-$2}'`
   Qmiulast=`awk -v thk0=$1 -v depm=$depmlast 'BEGIN{dep=thk0+depm; deplast=0; Qlast=344898}{if($1>=dep){print Qlast+(dep-deplast)*($2-Qlast)/($1-deplast); exit} deplast=$1;Qlast=$2}' $3`
   echo $thklast $vellast $Qmiulast >> $4
}

### check for curve file and merge everything into the new layerized model
MergeAll() {
   ### check dirMantle and dirQmiu for current path-age
   fMantle=${dirMantle}/MC.1.${age}.mod_mantle
   fQmiu=${dirQmiu}/Qmiu.30.${age}
   if [ ! -e $fMantle ] || [ ! -e $fQmiu ]; then
      echo "   path-age out of range (="$age")."
      continue
   fi
   ### write the bathy and sediments
   if [ `echo $bathy | awk '{if($1>0){print 1}else{print 0}}'` == 1 ]; then
      echo $bathy 0.000 0 > $fout
      cflag='-c 0.85'
   else
      rm -f $fout
      cflag=''
   fi
   if [ `echo $sedthk | awk '{if($1>0){print 1}else{print 0}}'` == 1 ]; then
      echo $sedthk $sedvs 80 >> $fout
   fi
   ### add crust
   awk '{ if(NR==1){if($1!=0){print "format error!"}else{depdone=0}}else{thklast=(deplast+$1)/2.-depdone; print thklast,vellast,Qlast; depdone+=thklast;} deplast=$1; vellast=$2; Qlast=$3 }END{print deplast-depdone,vellast,Qlast;}' $fCrust >> $fout
   cruthk=`tail -n1 $fCrust | awk '{print $1}'`
   ### merge mantle vel and Qmiu from fMantle and fQmiu
   thksum=`echo $bathy $sedthk $cruthk | awk '{print $1+$2+$3}'`
   MergeMantleQ $thksum $fMantle $fQmiu $fout
   ### convert thk to dep just for plotting purpose
   awk 'BEGIN{dep=0}{print dep,$2,$3; dep+=$1; print dep,$2,$3}' $fout > $fout'_curve'
}

###
PredDisp() {
   ## 1:input_model
   awk '{if($2==0){print $1,1.45,0,0.77,$3}else{if($3==80){ratio=2.}else{ratio=1.73} print $1,$2*ratio,$2,$2*ratio*0.32 + 0.77,$3}}' $1 > model.tmp
   echo $1'.R' $1'.R.att model.tmp' >> $fcleanup
   echo "   Making predictions.."
   SURF_DISP model.tmp $1 R 0 1 $perl $perh 0.3 -a -f $cflag >& /dev/null
   if [ `more $1'.R.grv' | wc -l` == 0 ]; then
      perc=0
   else
      perc=`minmax -C $1'.R.grv' | awk -v range=$rangeper '{print ($2-$1)/range}'`
   fi
   if [ `echo $perc | awk '{if($1<0.8){print 1}else{print 0}}'` == 1 ]; then
      echo "   Bad results from SURF_DISP. Remaking predictions.."
      SURF_DISP model.tmp $1 R 0 1 $perl2 $perh 0.3 -a -f $cflag >& /dev/null
   fi
   echo $1 $1'_curve' >> $fcleanup
}

CheckResult() {
   if [ -e $1 ] && [ `awk '{if($1>0.){print 1}else{print 0}; exit}' $1` == 1 ]; then
      echo "   Old Results are OK. Skipped!"
      continue
   fi
}


### main starts here ###
Initialize
ista=1
nsta=`more $stalst | wc -l`
while [ $ista -le $nsta ]; do
   Getssta # extract source sta info
   for rinfo in `awk -v ista=$ista 'NR>ista{print $1"_"$2"_"$3}' $stalst`; do
   #for rinfo in `awk '$1=="J46A"{print $1"_"$2"_"$3}' $stalst`; do #pick rsta
      Getrsta # extract receiver stat info
      fout=$ssta'_'$rsta
      echo "Working on path "$fout"..."
      CheckResult $fout
      ComputeAll # compute & check, break the while if computation fails
      MergeAll # merge bathy, sediments, crust and mantle structure into path models
      ### Produce Dispersion predictions (.R.grv and .R.phv) from fout
      PredDisp $fout # run SURF_DISP to predict phase and group dispersions
      more $fcleanup | xargs rm -f
      rm -f $fcleanup
   done #rinfo (individual path)
   let ista++
done #ista
