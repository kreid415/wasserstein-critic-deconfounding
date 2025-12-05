#!/bin/bash

# 1. Create directories
mkdir -p ./scripts/multibatch_experiments/unbalanced/panc
mkdir -p ./scripts/multibatch_experiments/unbalanced/immune
mkdir -p ./scripts/multibatch_experiments/balanced/panc
mkdir -p ./scripts/multibatch_experiments/balanced/immune

# 2. Generate commands and pipe to Parallel
(
    # --- PANCREAS SEQUENCE (Batch counts 2 to 8) ---
    for i in {2..8}; do
        # Unbalanced
        echo "python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/unbalanced/panc/panc_$i --batch_count $i --epochs 500 --reference_batch 0"
        
        # Balanced
        echo "python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/balanced/panc/panc_$i --batch_count $i --epochs 500 --reference_batch 0 --balance"
    done

    # --- IMMUNE SEQUENCE (Batch counts 2 to 4) ---
    for i in {2..4}; do
        # Unbalanced
        echo "python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/multibatch_experiments/unbalanced/immune/immune_$i --batch_count $i --epochs 500 --reference_batch 0"
        
        # Balanced
        echo "python scripts/hyperparameter_search.py --dataset immune --output_dir ./scripts/multibatch_experiments/balanced/immune/immune_$i --batch_count $i --epochs 500 --reference_batch 0 --balance"
    done

    # --- THE OUTLIER (Panc 9, Batch Count 2, Ref Batch 9) ---
    # Unbalanced
    echo "python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/unbalanced/panc/panc_9 --batch_count 2 --epochs 500 --reference_batch 9"
    # Balanced
    echo "python scripts/hyperparameter_search.py --dataset pancreas --output_dir ./scripts/multibatch_experiments/balanced/panc/panc_9 --batch_count 2 --epochs 500 --reference_batch 9 --balance"

) | parallel -j 6 --verbose
