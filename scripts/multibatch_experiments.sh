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

# --- PANCREAS SEQUENCE (Batch counts 2 to 9) ---
for i in {2..9}; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/unbalanced/panc/panc_$i --batch_count $i --reference_batch -1"
    logname="panc_batch${i}_unbalanced"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/balanced/panc/panc_$i --batch_count $i --reference_batch -1 --balance"
    logname="panc_batch${i}_balanced"
    run_task "$cmd" "$logname"
done

# --- IMMUNE SEQUENCE (Batch counts 2 to 4) ---
for i in {2..4}; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/multibatch_experiments/unbalanced/immune/immune_$i --batch_count $i --reference_batch -1"
    logname="immune_batch${i}_unbalanced"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/multibatch_experiments/balanced/immune/immune_$i --batch_count $i --reference_batch -1 --balance"
    logname="immune_batch${i}_balanced"
    run_task "$cmd" "$logname"
done

echo "All tasks completed."
