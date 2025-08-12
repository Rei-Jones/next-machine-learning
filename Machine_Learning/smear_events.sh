#!/bin/bash

# bash script for running SmearEvents.py
# run as source smear_events.sh -p pressure -d diffusion -t type
#===============================================================

START_TIME=$(date +%s)
echo "Current directory: $(pwd)"

source /home/rei/NEXT/next_env/bin/activate

# Read arguments

allowed_p=(1 5 10 15 25) # allowed prerssure values
allowed_d=(0.1 0.05 0.25 5) # allowed diffusion values
allowed_t=("leptoquark" "0nubb") # allowed event types

valid_p=false
valid_d=false
valid_t=false

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
        t)
            for val in "${allowed_t[@]}"; do
                if [[ "$OPTARG" == "$val" ]]; then
                    valid_t=true
                    param_t=$OPTARG
                    break
                fi
            done
            if ! $valid_t; then
                echo "Invalid event type. Allowed types: ${allowed_t[*]}"
                exit 1
            fi
            ;;
        h)
            echo "Usage: $0 -p [${allowed_p[*]}] -d [${allowed_d[*]}] -t [${allowed_d[*]}]"
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

# create destination directory
DEST_DIR="/home/rei/NEXT/vertex_ML/diffused_data/${param_t}/${param_p}bar/${param_d}"

echo "Destination directory: $DEST_DIR"

if [ ! -d "$DEST_DIR" ]; then
    mkdir -p "$DEST_DIR"
fi

#original dir

ORIGINAL_DIR="/home/rei/NEXT/vertex_ML/simulation_data/${param_t}/${param_p}bar/${param_d}"

echo "Original directory: $ORIGINAL_DIR"

# Loop over files
for file in $ORIGINAL_DIR/*.h5; do
    base=$(basename "$file" .h5)

    # Skip if already processed
    if ls "$DEST_DIR"/${param_t}_${param_p}bar_${param_d}percent_smear_"$base".h5 1>/dev/null 2>&1; then
        echo "Skipping $base, already processed."
        continue
    fi

    echo "Processing: $base"

    # Run diffusion script
    python3 /home/rei/NEXT/vertex_ML/scripts/SmearEvents.py "$file" 1 "$param_d" "$param_p" 1.0 1

    # Find the latest .h5 file (assumed to be the new output)
    newfile=$(ls -t *.h5 | head -n 1)

    # Extract event number from original filename (e.g., leptoquark_1bar_102)
    event_num=$(echo "$base" | grep -o '[0-9]\+' | tr -d '\n')

    # Create desired output filename
    renamed="${param_t}_${param_p}bar_${param_d}percent_smear_${event_num}.h5"

    # Move and rename
    mv "$newfile" "$DEST_DIR/$renamed"
    echo "Moved $newfile to $DEST_DIR/$renamed"

done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Diffusion complete. Time taken: $DURATION seconds."

