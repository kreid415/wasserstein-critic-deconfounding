#!/bin/bash

# Exit immediately if any command fails.
# IMPORTANT: If your python script fails immediately on the first try,
# the background processes will exit instantly, making it look like nothing ran.
# Check the logs if this happens.
set -e

# --- BASE DIRECTORY SETUP ---
# Use absolute paths if possible to avoid ambiguity, but relative is okay for now.
BaseDir="./scripts/reference_experiments"
LogDir="$BaseDir/logs"

# 1. Create all necessary directories beforehand
mkdir -p "$BaseDir/unbalanced/panc" "$BaseDir/balanced/panc"
mkdir -p "$BaseDir/unbalanced/immune" "$BaseDir/balanced/immune"
# Assuming lung exists based on previous requests, remove if not needed
mkdir -p "$BaseDir/unbalanced/lung" "$BaseDir/balanced/lung"
mkdir -p "$LogDir"

echo "--- Starting Dataset-Parallel Execution ---"
echo "Strategy: 3 simultaneous processes (1 Pancreas queue, 1 Immune queue, 1 Lung queue)."
echo "Output logs saved to: $LogDir"
echo "You will see three progress bars below, one for each dataset queue."

# --- JOB GENERATION FUNCTIONS ---
# (These remain the same as before)

generate_panc_jobs() {
    # --- PANCREAS (0 to 9) ---
    for i in {0..9}; do
        # Unbalanced
        cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir $BaseDir/unbalanced/panc/panc_$i --batch_count 100 --epochs 500 --reference_batch $i"
        logname="panc_ref${i}_unbalanced"
        echo -e "${cmd}\t${logname}"
        # Balanced
        cmd="python scripts/hyperparameter_search.py --dataset pancreas --output_dir $BaseDir/balanced/panc/panc_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
        logname="panc_ref${i}_balanced"
        echo -e "${cmd}\t${logname}"
    done
}

generate_immune_jobs() {
    # --- IMMUNE (0 to 5) ---
    for i in {0..5}; do
        # Unbalanced
        cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir $BaseDir/unbalanced/immune/immune_$i --batch_count 100 --epochs 500 --reference_batch $i"
        logname="immune_ref${i}_unbalanced"
        echo -e "${cmd}\t${logname}"
        # Balanced
        cmd="python scripts/hyperparameter_search.py --dataset immune --output_dir $BaseDir/balanced/immune/immune_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
        logname="immune_ref${i}_balanced"
        echo -e "${cmd}\t${logname}"
    done
}

generate_lung_jobs() {
    # --- LUNG (Placeholder - adjust range as needed) ---
    for i in {0..1}; do
        # Unbalanced
        cmd="python scripts/hyperparameter_search.py --dataset lung --output_dir $BaseDir/unbalanced/lung/lung_$i --batch_count 100 --epochs 500 --reference_batch $i"
        logname="lung_ref${i}_unbalanced"
        echo -e "${cmd}\t${logname}"
        # Balanced
        cmd="python scripts/hyperparameter_search.py --dataset lung --output_dir $BaseDir/balanced/lung/lung_$i --batch_count 100 --epochs 500 --reference_batch $i --balance"
        logname="lung_ref${i}_balanced"
        echo -e "${cmd}\t${logname}"
    done
}


# --- PARALLEL EXECUTION ---

# Key changes here:
# 1. Removed 'eval'.
# 2. Changed '--verbose' to '--progress' for a cleaner status bar view.
# 3. Explicitly wrote out the parallel command for each queue to ensure stability.

echo "Launching queues..."

# 1. Pancreas Queue
# We pipe the generator function directly into the parallel command.
generate_panc_jobs | parallel --colsep '\t' -j 1 --progress \
    --joblog "$LogDir/panc_joblog.txt" \
    --results "$LogDir/{2}" \
    {1} &
PANC_PID=$!
echo "Pancreas queue launched (PID $PANC_PID)"

# 2. Immune Queue
generate_immune_jobs | parallel --colsep '\t' -j 1 --progress \
    --joblog "$LogDir/immune_joblog.txt" \
    --results "$LogDir/{2}" \
    {1} &
IMMUNE_PID=$!
echo "Immune queue launched (PID $IMMUNE_PID)"

# 3. Lung Queue (Remove if not needed)
generate_lung_jobs | parallel --colsep '\t' -j 1 --progress \
    --joblog "$LogDir/lung_joblog.txt" \
    --results "$LogDir/{2}" \
    {1} &
LUNG_PID=$!
echo "Lung queue launched (PID $LUNG_PID)"


echo "Waiting for all jobs to complete..."
# Wait for those specific background process IDs to finish
wait $PANC_PID $IMMUNE_PID $LUNG_PID

echo "All experiments completed."

# Combine joblogs for easier checking later
cat "$LogDir"/panc_joblog.txt "$LogDir"/immune_joblog.txt "$LogDir"/lung_joblog.txt > "$LogDir"/combined_master_joblog.txt
echo "Final combined job status log: $LogDir/combined_master_joblog.txt"
