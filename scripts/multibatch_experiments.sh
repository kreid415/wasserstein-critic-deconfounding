#!/bin/bash

# 1. Create base directories for data AND logs
mkdir -p ./scripts/multibatch_experiments/unbalanced/panc
mkdir -p ./scripts/multibatch_experiments/unbalanced/immune
mkdir -p ./scripts/multibatch_experiments/balanced/panc
mkdir -p ./scripts/multibatch_experiments/balanced/immune
# Create a central logs directory
mkdir -p ./scripts/multibatch_experiments/logs

echo "Starting sequential execution..."
echo "Logs will be saved to: ./scripts/multibatch_experiments/logs/"

# Function to execute command and log output to a file
run_task() {
    local cmd="$1"
    local logname="$2"
    local logfile="./scripts/multibatch_experiments/logs/${logname}.log"
    
    echo "----------------------------------------------------------------"
    echo "Starting task: $logname"
    echo "Command: $cmd"
    echo "Logging to: $logfile"
    
    # Execute the command, redirecting both stdout (1) and stderr (2) to the logfile
    $cmd > "$logfile" 2>&1
    
    echo "Finished task: $logname"
}

# --- PANCREAS SEQUENCE (Batch counts 2 to 8) ---
for i in {2..8}; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/unbalanced/panc/panc_$i --batch_count $i --epochs 500 --reference_batch 0"
    logname="panc_batch${i}_unbalanced"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/balanced/panc/panc_$i --batch_count $i --epochs 500 --reference_batch 0 --balance"
    logname="panc_batch${i}_balanced"
    run_task "$cmd" "$logname"
done

# --- IMMUNE SEQUENCE (Batch counts 2 to 4) ---
for i in {2..4}; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/multibatch_experiments/unbalanced/immune/immune_$i --batch_count $i --epochs 500 --reference_batch 0"
    logname="immune_batch${i}_unbalanced"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/multibatch_experiments/balanced/immune/immune_$i --batch_count $i --epochs 500 --reference_batch 0 --balance"
    logname="immune_batch${i}_balanced"
    run_task "$cmd" "$logname"
done

# --- THE OUTLIER (Panc 9 dir name, Batch Count 2, Ref Batch 9) ---
# Unbalanced
cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/unbalanced/panc/panc_9 --batch_count 2 --epochs 500 --reference_batch 9"
logname="panc_outlier_ref9_unbalanced"
run_task "$cmd" "$logname"

# Balanced
cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/balanced/panc/panc_9 --batch_count 2 --epochs 500 --reference_batch 9 --balance"
logname="panc_outlier_ref9_balanced"
run_task "$cmd" "$logname"

echo "All tasks completed."
