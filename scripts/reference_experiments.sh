#!/bin/bash

# --- BASE DIRECTORY SETUP ---
BaseDir="./scripts/reference_experiments"
LogDir="$BaseDir/logs"

# 1. Create all necessary directories beforehand
mkdir -p "$BaseDir/unbalanced/panc" "$BaseDir/balanced/panc"
mkdir -p "$BaseDir/unbalanced/immune" "$BaseDir/balanced/immune"
mkdir -p "$BaseDir/unbalanced/lung" "$BaseDir/balanced/lung"
mkdir -p "$LogDir"

echo "--- Starting Sequential Reference Experiments ---"
echo "Strategy: Strictly sequential (Pancreas -> Immune -> Lung)."
echo "Output logs saved to: $LogDir"

# Function to execute command and log output to a file
run_task() {
    local cmd="$1"
    local logname="$2"
    local logfile="$LogDir/${logname}.log"
    
    echo "----------------------------------------------------------------"
    echo "Starting task: $logname"
    echo "Command: $cmd"
    echo "Logging to: $logfile"
    
    # Execute the command, redirecting both stdout (1) and stderr (2) to the logfile
    $cmd > "$logfile" 2>&1
    
    echo "Finished task: $logname"
}

# --- PANCREAS EXPERIMENTS (0 to 9) ---
echo "Starting Pancreas Queue..."
for i in {0..9}; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir $BaseDir/unbalanced/panc/panc_$i --batch_count 100 --epochs 500 --reference_batch $i"
    logname="panc_ref${i}_unbalanced"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir $BaseDir/balanced/panc/panc_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
    logname="panc_ref${i}_balanced"
    # run_task "$cmd" "$logname"
done

# --- IMMUNE EXPERIMENTS (0 to 5) ---
echo "Starting Immune Queue..."
for i in {0..5}; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir $BaseDir/unbalanced/immune/immune_$i --batch_count 100 --epochs 500 --reference_batch $i"
    logname="immune_ref${i}_unbalanced"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir $BaseDir/balanced/immune/immune_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
    logname="immune_ref${i}_balanced"
    # run_task "$cmd" "$logname"
done

# # --- LUNG EXPERIMENTS (0 to 1) ---
# echo "Starting Lung Queue..."
# for i in {0..1}; do
#     # Unbalanced
#     cmd="python scripts/hyperparameter_search.py --dataset lung --output_dir $BaseDir/unbalanced/lung/lung_$i --batch_count 100 --epochs 500 --reference_batch $i"
#     logname="lung_ref${i}_unbalanced"
#     run_task "$cmd" "$logname"
    
#     # Balanced
#     cmd="python scripts/hyperparameter_search.py --dataset lung --output_dir $BaseDir/balanced/lung/lung_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
#     logname="lung_ref${i}_balanced"
#     run_task "$cmd" "$logname"
# done

echo "All reference experiments completed."
