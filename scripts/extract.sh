#!/bin/bash

if [ "$COLUMNS" != "" ] ; then
    DIR=`echo $COLUMNS|sed 's/ /_/g'`
else
    COLUMNS='HR,SpO2,InvBPSys,InvBPDias,RRtotal,Temperature monitor,Central Venous Pressure'
    DIR=ALL
fi
MINROWS=${MINROWS:-60}

extract -c "$COLUMNS" -n $MINROWS -o data/$DIR data/monitor-dataset-Ichilov_MICU_[0-9][0-9][0-9][0-9][0-9].pkl | tee extract-$DIR.log
