#!/bin/bash

START_TIME=$(date +%s)

CONFIG_FILE="ATPC_Leptoquark.config.mac"
source setup_cluster.sh
cat $CONFIG_FILE
# Extract current values from the config file
seed=$(grep "^[[:space:]]*/nexus/random_seed" "$CONFIG_FILE" | awk '{print $2}')
start_id=$(grep "^[[:space:]]*/nexus/persistency/start_id" "$CONFIG_FILE" | awk '{print $2}')

for i in {1..500}; do
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

  cp Leptoquark.h5 /home/rei/NEXT/vertex_ML/simulation_data/leptoquark/10bar/0.05/leptoquark_10bar_$i.h5
  echo "Successfully copied file $i"

  rm Leptoquark.h5
  echo "Event file $i removed from directory"
  
  # Update base seed and start_id for next iteration
  seed=$new_seed
  start_id=$new_start_id
done

