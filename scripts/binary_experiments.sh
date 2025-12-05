#!/bin/bash

# Ensure output directories exist before running
mkdir -p ./scripts/binary_results/unbalanced
mkdir -p ./scripts/binary_results/balanced

# Run GNU Parallel
# -j 6: Run 6 jobs simultaneously (adjust based on your CPU cores)
# --verbose: Print the commands as they run so you can see progress
parallel -j 6 --verbose \
    python scripts/hyperparameter_search.py \
    --dataset {1} \
    --batch_count 2 \
    --epochs 500 \
    {2} \
    ::: immune pancreas lung \
    ::: "--output_dir ./scripts/binary_experiments/unbalanced" \
        "--output_dir ./scripts/binary_experiments/balanced --balance"
