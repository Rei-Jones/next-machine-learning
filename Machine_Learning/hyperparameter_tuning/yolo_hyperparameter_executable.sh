#!/bin/bash


JOBID=$1

echo "starting hyperparameter tuning for ${JOBID}"

mkdir ./data
mkdir ./weights
mkdir ./runs

yolo settings datasets_dir="./data"
yolo settings weights_dir="./weights"
yolo settings runs_dir="./runs"

start=$(date +%s)

python3 yolo_hyperparameter_tune.py ${JOBID}

tar -cvf results.tar "runs/tune_results/${JOBID}"

echo "FINISHED....EXITING"

end=$(date +%s)
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds