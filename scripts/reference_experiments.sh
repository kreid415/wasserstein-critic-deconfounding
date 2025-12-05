#!/bin/bash

# 1. Create base directories to prevent race conditions
mkdir -p ./scripts/reference_experiments/unbalanced/panc
mkdir -p ./scripts/reference_experiments/unbalanced/immune
mkdir -p ./scripts/reference_experiments/balanced/panc
mkdir -p ./scripts/reference_experiments/balanced/immune

# 2. Generate the job list and pipe to GNU Parallel
# We use a subshell ( ... ) to group the output of the loops
(
    # --- PANCREAS (0 to 9) ---
    for i in {0..9}; do
        # Unbalanced
        echo "python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/reference_experiments/unbalanced/panc/panc_$i --batch_count 100 --epochs 500 --reference_batch $i"
        
        # Balanced (adds --balance flag)
        echo "python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/reference_experiments/balanced/panc/panc_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
    done

    # --- IMMUNE (0 to 5) ---
    for i in {0..5}; do
        # Unbalanced
        echo "python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/reference_experiments/unbalanced/immune/immune_$i --batch_count 100 --epochs 500 --reference_batch $i"
        
        # Balanced (adds --balance flag)
        echo "python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/reference_experiments/balanced/immune/immune_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
    done

) | parallel -j 6 --verbose
