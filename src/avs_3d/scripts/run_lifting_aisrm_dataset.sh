#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: You must provide DATASET_ROOT as the first argument."
    echo "Usage: $0 /path/to/dataset"
    exit 1
fi

DATASET_ROOT="$1"

echo "Using DATASET_ROOT=$DATASET_ROOT"

process_task() {
    local part=$1
    local flat=$2
    local room=$3
    local indx=$4
    local gpu=$5
    local use_aisrm=$6
    local threshold=$7

    ROOT="${DATASET_ROOT}/${part}/${flat}/${room}/${indx}"

    if [ -d "${ROOT}/3dgs_scene" ]; then

        if [ "$use_aisrm" = "true" ]; then
            echo "Processing $ROOT on GPU $gpu with AISRM enabled: True and threshold $threshold"
            CUDA_VISIBLE_DEVICES=$gpu python ./src/avs_3d/run_lifting_aisrm.py --path "${ROOT}/3dgs_scene" --use_aisrm
        else
            echo "Processing $ROOT on GPU $gpu with AISRM enabled: False and threshold $threshold"
            CUDA_VISIBLE_DEVICES=$gpu python ./src/avs_3d/run_lifting_aisrm.py --path "${ROOT}/3dgs_scene"
        fi

        # fi
    else
        echo "Directory ${ROOT}/3dgs_scene does not exist, skipping..."
    fi
}

# Prepare the tasks list
task_list=()
for part in "part_1" "part_2"; do
    for flat in "5ZKStnWn8Zo" "q9vSo1VnCiC" "sT4fr6TAbpF" "wc2JMjhGNzB"; do
        for room in "bathroom" "kitchen" "living_room"; do
            for indx in "1" "2" "3"; do
                for gpu in 0; do
                    for use_aisrm in true false; do
                        for threshold in 0.3; do
                            task_list+=("$part $flat $room $indx $gpu $use_aisrm $threshold")
                        done
                    done
                done
            done
        done
    done
done

# Function to run tasks in parallel
run_in_parallel() {
    local max_parallel=1  # Max number of parallel tasks
    local count=0

    for task in "${task_list[@]}"; do
        set -- $task
        part=$1
        flat=$2
        room=$3
        indx=$4
        gpu=$5
        use_aisrm=$6
        threshold=$7

        process_task "$part" "$flat" "$room" "$indx" "$gpu" "$use_aisrm" "$threshold" &

        ((count++))
        if ((count == max_parallel)); then
            wait
            count=0
        fi
    done

    wait
}

# Execute the tasks
run_in_parallel