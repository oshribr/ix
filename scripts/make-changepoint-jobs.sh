#!/bin/bash

MODEL=${MODEL:-HR,SpO2,InvBPSys,InvBPDias,RRtotal}
DEPTH=${DEPTH:-15}
ARCH=${ARCH:-rnn-128-3-0.5}

for stay in $@; do 
    JOBID=$stay
    echo $JOBID
    cat > jobs/$JOBID.job <<__END__
#!/bin/bash

#PBS -N $JOBID
#PBS -q batch
#PBS -d /home/david/work/intensone/temporal/monitor

export PATH=/opt/anaconda/anaconda3/bin:/opt/anaconda/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin

cd /home/david/work/intensone/temporal/monitor
. startenv

if [ \! -e data/monitor-dataset-${stay}-changepoints.npy ] ; then
    OMP_NUM_THREADS=1 \
    changepoints models/$MODEL-$DEPTH-$ARCH.model \
                 data/$MODEL/ \
                 data/monitor-dataset-$stay.pkl
fi
__END__

    chmod +x jobs/$JOBID.job
done
