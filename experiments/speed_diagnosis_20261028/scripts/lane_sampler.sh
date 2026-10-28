#!/bin/bash
# Once per $1 seconds (default 60) append one JSON line per fit/score child plus one per-job line (GPU use, cgroup memory, load).
# Read-only (/proc, nvidia-smi, cgroup files); started from the job script as: bash lane_sampler.sh 60 "$OUT/sampler_$SLURM_JOB_ID.jsonl" &
# Children are the python processes whose command line names fit_paper_config.py or score_scib_native.py.
iv=${1:-60}; out=${2:-sampler.jsonl}
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)
while sleep "$iv"; do
  ts=$(date +%s)
  for p in $(pgrep -f '^[^ ]*python[^ ]* [^ ]*(fit_paper_config|score_scib_native)\.py'); do
    [ -r /proc/$p/stat ] || continue
    read -r _ _ st _ < <(cut -d' ' -f1-3 /proc/$p/stat)
    ut=$(awk '{print $14+$15}' /proc/$p/stat); rss=$(awk '/VmRSS/{print $2}' /proc/$p/status)
    echo "{\"ts\":$ts,\"pid\":$p,\"state\":\"$st\",\"wchan\":\"$(cat /proc/$p/wchan 2>/dev/null)\",\"cpu_ticks\":$ut,\"rss_kb\":${rss:-0},\"tag\":\"$(tr '\0' ' ' < /proc/$p/environ 2>/dev/null | grep -o 'TAG=[^ ]*' | head -1)\"}" >> "$out"
  done
  g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null) && g=$(echo "$g" | head -1 | tr -d ' ') || g=na
  echo "{\"ts\":$ts,\"job\":1,\"gpu_util_mem_mb\":\"${g:-na}\",\"cg_mem_cur\":$(cat $cg/memory.current 2>/dev/null || echo 0),\"cg_mem_peak\":$(cat $cg/memory.peak 2>/dev/null || echo 0),\"load\":\"$(cut -d' ' -f1 /proc/loadavg)\"}" >> "$out"
done
