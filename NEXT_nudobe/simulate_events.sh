#!/bin/bash

#source simulate_events.sh -p pressure -d diffusion_folder


CONFIG_FILE="ATPC_Leptoquark.config.mac"
source setup_cluster.sh
cat $CONFIG_FILE
# Extract current values from the config file
seed=$(grep "^[[:space:]]*/nexus/random_seed" "$CONFIG_FILE" | awk '{print $2}')
start_id=$(grep "^[[:space:]]*/nexus/persistency/start_id" "$CONFIG_FILE" | awk '{print $2}')

# Read arguments

allowed_p=(1 5 10 15 25) # allowed prerssure values
allowed_d=(0.1 0.05 0.25 5) # allowed diffusion values

valid_p=false
valid_d=false

while getopts ":p:d:t:h" opt; do
    case $opt in
        p) 
            for val in "${allowed_p[@]}"; do
                if [[ "$OPTARG" == "$val" ]]; then
                    valid_p=true
                    param_p=$OPTARG
                    break
                fi
            done
            if ! $valid_p; then
                echo "Invalid pressure value. Allowed values: ${allowed_p[*]}"
                exit 1
            fi
            ;;
        d) 
            for val in "${allowed_d[@]}"; do
                if [[ "$OPTARG" == "$val" ]]; then
                    valid_d=true
                    param_d=$OPTARG
                    break
                fi
            done
            if ! $valid_d; then
                echo "Invalid diffusion value. Allowed values: ${allowed_d[*]}"
                exit 1
            fi
            ;;
        h)
            echo "Usage: $0 -p [${allowed_p[*]}] -d [${allowed_d[*]}]"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires a value" >&2
            exit 1
            ;;
    esac
done

# Destination Directory
DEST_DIR="/home/rei/NEXT/vertex_ML/simulation_data/leptoquark/${param_p}bar/${param_d}"
if [ ! -d "$DEST_DIR" ]; then
    mkdir -p "$DEST_DIR"
fi

for i in {1..200}; do
  new_seed=$((seed + 1))
  new_start_id=$((start_id + (i * 120)))

  # Update config file with new values (so it's preserved)
  sed -i "s|^/nexus/random_seed .*|/nexus/random_seed $new_seed|" "$CONFIG_FILE"
  sed -i "s|^/nexus/persistency/start_id .*|/nexus/persistency/start_id $new_start_id|" "$CONFIG_FILE"

  echo "Updated $CONFIG_FILE"
  echo "seed: $new_seed"
  echo "start id: $new_start_id"

  nexus ATPC_Leptoquark.init.mac -n 120

  echo "Finished creating file $i"

  cp Leptoquark.h5 /home/rei/NEXT/vertex_ML/simulation_data/leptoquark/${param_p}bar/${param_d}/leptoquark_${param_p}bar_$i.h5
  echo "Successfully copied file $i"

  rm Leptoquark.h5
  echo "Event file $i removed from directory"
  
  # Update base seed and start_id for next iteration
  seed=$new_seed
  start_id=$new_start_id
done

# Update final values so next run continues
sed -i "s|^/nexus/random_seed .*|/nexus/random_seed $seed|" "$CONFIG_FILE"
sed -i "s|^/nexus/persistency/start_id .*|/nexus/persistency/start_id $start_id|" "$CONFIG_FILE"
