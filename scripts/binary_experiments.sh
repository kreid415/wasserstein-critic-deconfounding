#!/bin/bash

# 1. Create base directories
mkdir -p ./scripts/binary_experiments/unbalanced
mkdir -p ./scripts/binary_experiments/balanced
mkdir -p ./scripts/binary_experiments/logs

echo "--- Starting Sequential Binary Experiments ---"
echo "Strategy: Strictly sequential (Immune -> Pancreas -> Lung)."
echo "Output logs saved to: ./scripts/binary_experiments/logs"

# Function to execute command and log output to a file
run_task() {
    local cmd="$1"
    local logname="$2"
    local logfile="./scripts/binary_experiments/logs/${logname}.log"
    
    echo "----------------------------------------------------------------"
    echo "Starting task: $logname"
    echo "Command: $cmd"
    echo "Logging to: $logfile"
    
    # Execute the command, redirecting both stdout (1) and stderr (2) to the logfile
    $cmd > "$logfile" 2>&1
    
    echo "Finished task: $logname"
}

# 2. Iterate through datasets sequentially
for dataset in immune pancreas lung; do
    echo "Processing dataset: $dataset..."

    # --- Unbalanced ---
    cmd="python scripts/hyperparameter_search.py --dataset $dataset --batch_count 2 --output_dir ./scripts/binary_experiments/unbalanced"
    logname="${dataset}_unbalanced"
    run_task "$cmd" "$logname"

    # --- Balanced ---
    cmd="python scripts/hyperparameter_search.py --dataset $dataset --batch_count 2 --output_dir ./scripts/binary_experiments/balanced --balance"
    logname="${dataset}_balanced"
    run_task "$cmd" "$logname"
done

echo "All binary experiments completed."
