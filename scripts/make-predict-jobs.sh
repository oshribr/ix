#!/bin/bash

ARCH=${ARCH:-rnn}
MODEL=${MODEL:-ALL}
DEPTH=${DEPTH:-5}
DELAY=${DELAY:-1}
DATASET=${DATASET:-Ichilov}

if [ $DATASET = Ichilov ] ; then
    PATTERN=${PATTERN:-data/*Ichilov_MICU_[0-9][0-9][0-9][0-9][0-9].pkl}
elif [ $DATASET = Mayo ] ; then
    PATTERN=${PATTERN:-data/*Mayo_MICU_???????_????????.pkl}
else
    echo Unknown dataset $DATASET, expected one of Ichilov, Mayo
    exit 1
fi

for stay in $PATTERN; do 
    JOBID=`echo $stay|sed 's/.*\///'|sed 's/monitor-dataset-//'|sed 's/\.pkl//'`-$MODEL-$DEPTH:$DELAY-$ARCH
    echo $JOBID
    cat > jobs/$JOBID.job <<__END__
#!/bin/bash

#PBS -N $JOBID
#PBS -q batch
#PBS -d /home/david/work/intensone/temporal/monitor

export PATH=/opt/anaconda/anaconda3/bin:/opt/anaconda/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin

cd /home/david/work/intensone/temporal/monitor
. startenv

OMP_NUM_THREADS=1 \
PATTERN=$stay ARCH=$ARCH MODEL=$MODEL \
DEPTH=$DEPTH DELAY=$DELAY DATASET=$DATASET \
scripts/predict.sh

__END__

    chmod +x jobs/$JOBID.job
done
