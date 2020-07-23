#!/bin/tcsh
foreach f (*.f *.h)
echo "=========== $f +++++++++++++++++"
set pref=$f:e
if ($pref == h) then
diff $f ../sun/$f
else
diff $f ../sun/$f:r.F
endif
end
exit
