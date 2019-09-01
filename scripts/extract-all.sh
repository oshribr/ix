#!/bin/bash

MINROWS=${MINROWS:-60}

# first, extract with all concepts
MINROWS=$MINROWS `dirname $0`/extract.sh

# then, with subsets of concepts
COLUMNS=""
for COLUMN in HR SpO2 InvBPSys,InvBPDias RRtotal; do
    if [ "$COLUMNS" == "" ]; then
        COLUMNS=$COLUMN
    else
        COLUMNS=${COLUMNS},$COLUMN
    fi
    echo $COLUMNS
    COLUMNS=$COLUMNS MINROWS=$MINROWS `dirname $0`/extract.sh
done
