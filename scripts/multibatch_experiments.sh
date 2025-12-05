#!/bin/bash

# 1. Create base directories for data AND logs
mkdir -p ./scripts/multibatch_experiments/unbalanced/panc
mkdir -p ./scripts/multibatch_experiments/unbalanced/immune
mkdir -p ./scripts/multibatch_experiments/balanced/panc
mkdir -p ./scripts/multibatch_experiments/balanced/immune
# Create a central logs directory
mkdir -p ./scripts/multibatch_experiments/logs

echo "Generating commands and starting parallel execution..."
echo "Logs will be saved to: ./scripts/multibatch_experiments/logs/"

# 2. Generate commands + log names, piped to Parallel
# We use 'echo -e' so that it recognizes '\t' as a tab separator.
(
    # --- PANCREAS SEQUENCE (Batch counts 2 to 8) ---
    for i in {2..8}; do
        # Unbalanced
        cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/unbalanced/panc/panc_$i --batch_count $i --epochs 500 --reference_batch 0"
        logname="panc_batch${i}_unbalanced"
        echo -e "${cmd}\t${logname}"
        
        # Balanced
        cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/balanced/panc/panc_$i --batch_count $i --epochs 500 --reference_batch 0 --balance"
        logname="panc_batch${i}_balanced"
        echo -e "${cmd}\t${logname}"
    done

    # --- IMMUNE SEQUENCE (Batch counts 2 to 4) ---
    for i in {2..4}; do
        # Unbalanced
        cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/multibatch_experiments/unbalanced/immune/immune_$i --batch_count $i --epochs 500 --reference_batch 0"
        logname="immune_batch${i}_unbalanced"
        echo -e "${cmd}\t${logname}"
        
        # Balanced
        cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/multibatch_experiments/balanced/immune/immune_$i --batch_count $i --epochs 500 --reference_batch 0 --balance"
        logname="immune_batch${i}_balanced"
        echo -e "${cmd}\t${logname}"
    done

    # --- THE OUTLIER (Panc 9 dir name, Batch Count 2, Ref Batch 9) ---
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/unbalanced/panc/panc_9 --batch_count 2 --epochs 500 --reference_batch 9"
    # Give it a distinct log name highlighting the reference batch difference
    logname="panc_outlier_ref9_unbalanced"
    echo -e "${cmd}\t${logname}"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/balanced/panc/panc_9 --batch_count 2 --epochs 500 --reference_batch 9 --balance"
    logname="panc_outlier_ref9_balanced"
    echo -e "${cmd}\t${logname}"

) | parallel --colsep '\t' -j 6 --verbose \
    --joblog ./scripts/multibatch_experiments/logs/master_joblog.txt \
    --results ./scripts/multibatch_experiments/logs/{2} \
    {1}

# Explaining the parallel command at the end:
# --colsep '\t': Split the input line by tabs. {1} is command, {2} is logname.
# --joblog ...: Creates a single summary file of all jobs (exit codes, runtimes).
# --results .../{2}: Creates a directory named after the logname column, containing stdout and stderr files.
# {1}: The actual command to execute.
