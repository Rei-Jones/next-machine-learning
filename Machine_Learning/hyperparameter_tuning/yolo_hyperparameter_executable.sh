#!/bin/bash


JOBID=$1

echo "starting hyperparameter tuning for ${JOBID}"

mkdir ./data
mkdir ./weights
mkdir ./runs
ls
mkdir dataset
cd dataset
mkdir images labels
cd labels
mkdir train val
cd ../images
mkdir train val
cd ..
cd ..

echo "extracting data..."
tar -xzf labeltrain.tar.gz -C dataset/labels/train
tar -xzf labelval.tar.gz -C dataset/labels/val
tar -xzf imgtrain.tar.gz -C dataset/images/train
tar -xzf imgval.tar.gz -C dataset/images/val

yolo settings datasets_dir="./data"
yolo settings weights_dir="./weights"
yolo settings runs_dir="./runs"

start=$(date +%s)

python3 yolo_tune_full.py ${JOBID}

tar -cvf results.tar -C /srv/scratch/runs/detect/runs/tune_results results_${JOBID}/

echo "FINISHED....EXITING"

end=$(date +%s)
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds
