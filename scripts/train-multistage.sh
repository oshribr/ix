#!/bin/bash

# Multistage training of longshot model

# Model
ARCH=${ARCH:-rnn}
HIDDEN_SIZE=${HIDDEN_SIZE:-128}
NLAYERS=${NLAYERS:-3}
P=${P:-0.5}
COLLECTION=${COLLECTION:-HR,SpO2,InvBPSys,InvBPDias,RRtotal}

# Data folder
DATA=${DATA:-/var/data/monitor/data}

# Loss horizon
DEPTH=${DEPTH:-30}
DEPTHS=${DEPTHS:-"5 15 $DEPTH"}

# SGD parameters 
BATCH_SIZE=${BATCH_SIZE:-256}
NEPOCHS=${NEPOCHS:-100}
RATE=${RATE:-0.001}
PATIENCE=${PATIENCE:-10}
BURNIN=${BURNIN:-5}
TEST_FRACTION=${TEST_FRACTION:-0.005}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.1}
EPSILON=${EPSILON:-1E-8}

mlast=
for d in $DEPTHS; do 
    model=$COLLECTION-$d-$ARCH-$HIDDEN_SIZE-$NLAYERS-$P.model
    if [ -e "$mlast" ] ; then
        cp $mlast $model
    fi
    if [ -e $mlast,optim ] ; then
        cp $mlast,optim $model,optim
    fi
    train -a $ARCH -d $d \
        -n $NEPOCHS -b $BATCH_SIZE \
        -t $TEST_FRACTION -T $TRAIN_FRACTION \
        -p $PATIENCE -u $BURNIN \
        -r $RATE -e $EPSILON \
        $model $DATA/$COLLECTION/array \
        $ARCH/hidden_size=$HIDDEN_SIZE $ARCH/nlayers=$NLAYERS $ARCH/p=$P \
        | tee -a $COLLECTION-$DEPTH-$ARCH-$HIDDEN_SIZE-$NLAYERS-$P.log
    mlast=$model
done
