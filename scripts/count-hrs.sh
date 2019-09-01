#!/bin/sh

STAYS=${STAYS:-'data/monitor-dataset-Ichilov_MICU_?????.pkl'}
HR_COUNTS=${HR_COUNTS:-'data/hr_counts'}

for x in $STAYS; do
    stay=`echo $x|sed s'/.*\(Ichilov_MICU_[0-9][0-9]*\).*/\1/'`
    echo $stay `python -c 'import sys, pickle, numpy
done | tee $HR_COUNTS