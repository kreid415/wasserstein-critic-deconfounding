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
echo "Strategy: Strictly sequential (PANCREAS -> IMMUNE -> LUNG)."
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

# --- PANCREAS EXPERIMENTS  ---
echo "Starting Pancreas Queue..."
for batch_size in 128 256 512 1024; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir $BaseDir/unbalanced/panc/panc_bs${batch_size} --batch_count 2 --reference_batch 0 --batch_size ${batch_size}"
    logname="panc_ref0_unbalanced_bs${batch_size}"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir $BaseDir/balanced/panc/panc_bs${batch_size} --batch_count 2 --reference_batch 0 --balance --batch_size ${batch_size}"
    logname="panc_ref0_balanced_bs${batch_size}"
    run_task "$cmd" "$logname"
done

# --- IMMUNE EXPERIMENTS ---
echo "Starting Immune Queue..."
for batch_size in 128 256 512 1024; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir $BaseDir/unbalanced/immune/immune_bs${batch_size} --batch_count 2 --reference_batch 0 --batch_size ${batch_size}"
    logname="immune_ref0_unbalanced_bs${batch_size}"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir $BaseDir/balanced/immune/immune_bs${batch_size} --batch_count 2 --reference_batch 0 --balance --batch_size ${batch_size}"
    logname="immune_ref0_balanced_bs${batch_size}"
    run_task "$cmd" "$logname"
done

# LUNG
echo "Starting LUNG Queue..."
for batch_size in 128 256 512 1024; do
    # Unbalanced
    cmd="python scripts/hyperparameter_search.py --dataset lung --output_dir $BaseDir/unbalanced/lung/lung_bs${batch_size} --batch_count 2 --reference_batch 0 --batch_size ${batch_size}"
    logname="lung_ref0_unbalanced_bs${batch_size}"
    run_task "$cmd" "$logname"
    
    # Balanced
    cmd="python scripts/hyperparameter_search.py --dataset lung --output_dir $BaseDir/balanced/lung/lung_bs${batch_size} --batch_count 2 --reference_batch 0 --balance --batch_size ${batch_size}"
    logname="lung_ref0_balanced_bs${batch_size}"
    run_task "$cmd" "$logname"
done

echo "All reference experiments completed."
