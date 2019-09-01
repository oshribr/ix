#!/bin/sh

STAYS=${STAYS:-'data/monitor-dataset-Ichilov_MICU_?????.pkl'}
HR_COUNTS=${HR_COUNTS:-'data/hr_counts'}

for x in $STAYS; do
    stay=`echo $x|sed s'/.*\(Ichilov_MICU_[0-9][0-9]*\).*/\1/'`
    echo $stay `python -c 'import sys, pickle, numpywith open(sys.argv[1], "rb") as f: df = pickle.load(f); print(df[numpy.logical_not(numpy.isnan(df["HR"]))].shape[0])' $x`
done | tee $HR_COUNTS
