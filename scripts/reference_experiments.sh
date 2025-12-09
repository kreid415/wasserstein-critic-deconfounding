#!/bin/bash

# 1. Create base directories to prevent race conditions
mkdir -p ./scripts/reference_experiments/unbalanced/panc
mkdir -p ./scripts/reference_experiments/unbalanced/immune
mkdir -p ./scripts/reference_experiments/balanced/panc
mkdir -p ./scripts/reference_experiments/balanced/immune

# Create the central logs directory
mkdir -p ./scripts/reference_experiments/logs

echo "Generating commands and starting parallel execution..."
echo "Logs will be saved to: ./scripts/reference_experiments/logs/"

# 2. Generate the job list + log names and pipe to GNU Parallel
# We use 'echo -e' and '\t' to create two columns: Command [tab] LogName
(
    # --- PANCREAS (0 to 9) ---
    for i in {0..9}; do
        # Unbalanced
        cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/reference_experiments/unbalanced/panc/panc_$i --batch_count 100 --epochs 500 --reference_batch $i"
        logname="panc_ref${i}_unbalanced"
        echo -e "${cmd}\t${logname}"
        
        # Balanced (adds --balance flag)
        cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/reference_experiments/balanced/panc/panc_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
        logname="panc_ref${i}_balanced"
        echo -e "${cmd}\t${logname}"
    done

    # --- IMMUNE (0 to 5) ---
    for i in {0..5}; do
        # Unbalanced
        cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/reference_experiments/unbalanced/immune/immune_$i --batch_count 100 --epochs 500 --reference_batch $i"
        logname="immune_ref${i}_unbalanced"
        echo -e "${cmd}\t${logname}"
        
        # Balanced (adds --balance flag)
        cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/reference_experiments/balanced/immune/immune_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
        logname="immune_ref${i}_balanced"
        echo -e "${cmd}\t${logname}"
    done

) | parallel --colsep '\t' -j 6 --verbose \
    --joblog ./scripts/reference_experiments/logs/master_joblog.txt \
    --results ./scripts/reference_experiments/logs/{2} \
    {1}

# Explanation of the parallel command:
# --colsep '\t': Split input lines into columns using tabs.
# --joblog ...: Save a master summary of all jobs to this text file.
# --results .../{2}: Create a log directory for each job named after column 2 (the logname).
# {1}: Execute the command in column 1.
