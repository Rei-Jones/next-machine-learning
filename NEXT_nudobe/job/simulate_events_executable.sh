#!/bin/bash
echo "Starting Job"

JOBID=$1

JOBNAME=$2

EVENT_TYPE=$3

PRESSURE=$4

echo "The JOBID number is: ${JOBID}"

echo "The JOBNAME number is: ${JOBNAME}"

echo "The event type is: ${EVENT_TYPE}"

echo "The Pressure is: ${PRESSURE}"

start=$(date +%s)

# setup nexus
echo "Setting up nexus"
source /software/nexus/setup_nexus.sh

# config
CONFIG=custom.config.mac
INIT=custom.init.mac

SEED=$((${JOBID} + 1))
echo "The seed is: ${SEED}"

N_EVENTS=120
echo "N_EVENTS: ${N_EVENTS}"
EID=$((${N_EVENTS}*${JOBID} + ${N_EVENTS}))
echo "The eid number is ${EID}"
START_ID=$EID


#edit config

sed -i "s#.*random_seed.*#/nexus/random_seed ${SEED}#" ${CONFIG}
sed -i "s#.*start_id.*#/nexus/persistency/start_id ${EID}#" ${CONFIG}
sed -i "s#.*gas_pressure.*#/Geometry/ATPC/gas_pressure ${PRESSURE}#" ${CONFIG}

cat ${INIT}
cat ${CONFIG}

# run nexus

echo "running nexus"
nexus -n ${N_EVENTS} ${INIT}

# compress file

python3 CompressEvents.py ${EVENT_TYPE}

# <Scale Factor> <CO2Percentage> <binsize> <pressure> <JOBID>
python3 SmearEvents.py ${EVENT_TYPE} 1  0.1 10 1.0 ${JOBID} # 0.1 % CO2
python3 SmearEvents.py ${EVENT_TYPE} 1    5 10 1.0 ${JOBID} # 5.0 % CO2

mv ${EVENT_TYPE}.h5 ${EVENT_TYPE}_nexus_${JOBID}.h5

python3 get_angle_data.py ${EVENT_TYPE}_nexus_${JOBID}.h5 $PRESSURE "nodiff"
python3 get_angle_data.py ${EVENT_TYPE}_0.1percent_smear_${JOBID}.h5 $PRESSURE "0.1percent"
python3 get_angle_data.py ${EVENT_TYPE}_5.0percent_smear_${JOBID}.h5 $PRESSURE "5.0percent"

python3 project_plots.py --input_file ${EVENT_TYPE}_nexus_${JOBID}.h5 --pressure "$PRESSURE" --diffusion "nodiff"  --type "${EVENT_TYPE}"
python3 project_plots.py --input_file ${EVENT_TYPE}_0.1percent_smear_${JOBID}.h5 --pressure "$PRESSURE" --diffusion "0.1percent"  --type "${EVENT_TYPE}"
python3 project_plots.py --input_file ${EVENT_TYPE}_5.0percent_smear_${JOBID}.h5 --pressure "$PRESSURE" --diffusion "5.0percent"  --type "${EVENT_TYPE}"

ls -ltrh

echo "Taring the h5 files"
tar -czf nudobe.tar *.h5 *.txt *.png *.jsonl

#cleanup
rm *.h5
rm *.mac
rm *.txt
rm *.py
rm *.jsonl
rm *.png

ls -ltrh

echo "FINISHED....EXITING"

end=$(date +%s)
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds