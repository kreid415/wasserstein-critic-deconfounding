#!/bin/bash

# 1. Create base directories
mkdir -p ./scripts/binary_experiments/unbalanced
mkdir -p ./scripts/binary_experiments/balanced
# Note: Parallel will automatically create the specific log subdirectories,
# but the parent 'logs' directory needs to exist first.
mkdir -p ./scripts/binary_experiments/logs

# 2. Generate commands + log names, piped to Parallel
# We use 'echo -e' so that it recognizes '\t' as a tab character.
(
    for dataset in immune pancreas lung; do
        # --- Unbalanced ---
        # Define the command string
        cmd="python scripts/hyperparameter_search.py --dataset $dataset --batch_count 2 --epochs 500 --output_dir ./scripts/binary_experiments/unbalanced"
        # Define the desired log name
        logname="${dataset}_unbalanced"
        # Echo them separated by a tab
        echo -e "${cmd}\t${logname}"

        # --- Balanced ---
        cmd="python scripts/hyperparameter_search.py --dataset $dataset --batch_count 2 --epochs 500 --output_dir ./scripts/binary_experiments/balanced --balance"
        logname="${dataset}_balanced"
        echo -e "${cmd}\t${logname}"
    done

) | parallel --colsep '\t' -j 6 --verbose \
    --joblog ./scripts/binary_experiments/logs/master_joblog.txt \
    --results ./scripts/binary_experiments/logs/{2} \
    {1}
