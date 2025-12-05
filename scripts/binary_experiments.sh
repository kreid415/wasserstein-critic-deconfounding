#!/bin/bash

# 1. Create base directories beforehand so parallel jobs don't race to create them
mkdir -p ./scripts/binary_experiments/unbalanced
mkdir -p ./scripts/binary_experiments/balanced

# 2. Generate commands in a subshell and pipe them to GNU Parallel
# The subshell (...) groups the output of the for loop into a single stream.
# Parallel reads from stdin, executing each line as a command.

(
    # Loop through the three target datasets
    for dataset in immune pancreas lung; do
        
        # --- Define the Unbalanced Command ---
        # Note: No --balance flag
        echo "python scripts/hyperparameter_search.py --dataset $dataset --batch_count 2 --epochs 500 --output_dir ./scripts/binary_experiments/unbalanced"
        
        # --- Define the Balanced Command ---
        # Note: Includes the --balance flag
        echo "python scripts/hyperparameter_search.py --dataset $dataset --batch_count 2 --epochs 500 --output_dir ./scripts/binary_experiments/balanced --balance"
        
    done

) | parallel -j 6 --verbose
# -j 6: Run 6 jobs simultaneously
# --verbose: Print the exact command being run before executing it
