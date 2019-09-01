#!/bin/bash

ARCH=${ARCH:-rnn}
MODEL=${MODEL:-ALL}
DEPTH=${DEPTH:-5}
DELAY=${DELAY:-1}
DATASET=${DATASET:-Ichilov}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [ $DATASET = Ichilov ] ; then
    PATTERN=${PATTERN:-data/*Ichilov_MICU_[0-9][0-9][0-9][0-9][0-9].pkl}
elif [ $DATASET = Mayo ] ; then
    PATTERN=${PATTERN:-data/*Mayo_MICU_???????_????????.pkl}
else
    echo Unknown dataset $DATASET, expected one of Ichilov, Mayo
    exit 1
fi

for stay in $PATTERN; do 
    for D in $DEPTH ; do
        INFIX=predict-$MODEL-$D:$DELAY-$ARCH
        ONAME=`echo $stay|sed "s/\.pkl/-$INFIX&/"`
        if [ \! -e $ONAME ]; then
            echo -n "$D	"
            OMP_NUM_THREADS=$OMP_NUM_THREADS \
            predict -q -d $DELAY -x predict-$MODEL-$D:$DELAY-$ARCH models/$MODEL-$D-$ARCH.model data/$MODEL $stay
        fi
    done
done | tee -a predict-$DATASET-$MODEL-`echo $DEPTH|sed 's/ /,/g'`:$DELAY-$ARCH.log
