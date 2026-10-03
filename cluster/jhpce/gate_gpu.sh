#!/bin/bash
# Cross-host gate, GPU part (2026-10-02). Runs on ONE pinned JHPCE GPU model, inside a 'gpu' batch job:
#   1. GPU witness: nvidia-smi + env_versions.py --kind fit (torch sees the GPU; device = EXPECT_GPU);
#   2. repo tests in wcd-fit on the GPU node: WCD_SRC=src python -m pytest -q tests/scvi (must not fail);
#   3. the local calibration, replayed: the two serial S_ fits (barycenter_iter5, 3 and 1 epochs; local warm-up
#      and single-lane timing), then the 64-fit mixed-arm queue Q_* in queue_order.txt order through
#      xargs -P 8 with 1 OMP/NUMBA thread per lane, scripts/fit_paper_config.py, the SAME manifest
#      (md5 9cca9c8c38e767ebf13a98ad44bb9e95) and the SAME prepped input bytes as the local run
#      (immune__scib.h5ad md5 3888d2563a7243f9854e41de973f3607, staged copy of the local file).
#      queue_times.tsv = tag, start, end, rc (the local format; scripts/calibrate_concurrency.py).
#   4. completion gate: all 64 latents present, z (33506, 10) finite, provenance = this repo HEAD (clean) + this GPU.
# Usage (repo root): bash cluster/jhpce/gate_gpu.sh BUNDLE_DIR GATE_DIR REPORT_DIR
#   BUNDLE_DIR: bench_queue_manifest.tsv, queue_order.txt (local calibration bundle); GATE_DIR: new, on fastscratch.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
source cluster/jhpce/env.sh
BUNDLE=$(readlink -f "${1:?bundle dir}")
GATE=${2:?gate dir}
REPORT=$(mkdir -p "${3:?report dir}" && cd "$3" && pwd)
EXPECT_GPU=${EXPECT_GPU:?set EXPECT_GPU, e.g. A100}
REF_PREPPED_DIR=${REF_PREPPED_DIR:?dir holding the staged local immune__scib.h5ad}
MANIFEST_MD5=9cca9c8c38e767ebf13a98ad44bb9e95
IMMUNE_MD5=3888d2563a7243f9854e41de973f3607

# ---- pre-submission checks, re-asserted on the node (fail before any GPU work) ----
[ -f "$FIT_ENV/.verified" ] || { echo "FATAL: $FIT_ENV not verified"; exit 1; }
case "$GATE" in /fastscratch/*) ;; *) echo "FATAL: GATE_DIR must be on /fastscratch ($GATE)"; exit 1;; esac
[ ! -e "$GATE" ] || { echo "FATAL: $GATE exists; the gate needs a fresh directory (fit_paper_config.py skips existing latents)"; exit 1; }
echo "$MANIFEST_MD5  $BUNDLE/bench_queue_manifest.tsv" | md5sum -c - | tee "$REPORT/precheck.txt"
echo "$IMMUNE_MD5  $REF_PREPPED_DIR/immune__scib.h5ad" | md5sum -c - | tee -a "$REPORT/precheck.txt"
[ "$(grep -c . "$BUNDLE/queue_order.txt")" -eq 64 ] || { echo "FATAL: queue_order.txt must hold 64 tags"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "FATAL: repo checkout is dirty"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader | tee "$REPORT/nvidia_smi.txt"
[ "$(grep -c . "$REPORT/nvidia_smi.txt")" -eq 1 ] || { echo "FATAL: expected exactly 1 visible GPU"; exit 1; }
grep -q "$EXPECT_GPU" "$REPORT/nvidia_smi.txt" || { echo "FATAL: GPU is not $EXPECT_GPU"; exit 1; }
echo "git HEAD $(git rev-parse HEAD); host $(hostname); cpus $(nproc); SLURM_JOB_ID ${SLURM_JOB_ID:-none}" | tee -a "$REPORT/precheck.txt"
lscpu | grep -E '^Model name|^CPU\(s\)|^Thread|^Socket' >> "$REPORT/precheck.txt"

export KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL PYTHONWARNINGS=ignore

# ---- 1. GPU witness ----
(cd "$TMPDIR" && "$FIT_PY" "$REPO/cluster/jhpce/env_versions.py" --kind fit --out "$REPORT/versions_fit_gpu.json")
"$FIT_PY" -c "import json,sys; d=json.load(open('$REPORT/versions_fit_gpu.json')); sys.exit(0 if d['cuda_available'] and '$EXPECT_GPU' in d['gpu'] else 1)"

# ---- 2. repo tests on the GPU node ----
WCD_SRC="$REPO/src" "$FIT_PY" -m pytest -q -p no:cacheprovider tests/scvi 2>&1 | tee "$REPORT/pytest_tests_scvi.txt"

# ---- 3. replay the local calibration ----
mkdir -p "$GATE/logs"
cp "$BUNDLE/bench_queue_manifest.tsv" "$BUNDLE/queue_order.txt" "$GATE/"
export MANIFEST="$GATE/bench_queue_manifest.tsv" PREPPED_DIR="$REF_PREPPED_DIR" OUT_DIR="$GATE/bench_out" WCD_SRC="$REPO/src"
for t in S_immune_barycenter_iter5 S_immune_barycenter_iter5_e1; do      # serial, as locally
  TAG=$t timeout 3000 "$FIT_PY" scripts/fit_paper_config.py > "$GATE/logs/$t.log" 2>&1
  grep '^\[fit\]' "$GATE/logs/$t.log" | tee -a "$REPORT/single_lane_fits.txt"
done
: > "$GATE/queue_times.tsv"
run_one() {
  local s e rc=0
  s=$(date +%s.%N)
  TAG=$1 timeout 3000 "$FIT_PY" "$REPO/scripts/fit_paper_config.py" > "$GATE/logs/$1.log" 2>&1 || rc=$?
  e=$(date +%s.%N)
  printf '%s\t%s\t%s\t%s\n' "$1" "$s" "$e" "$rc" >> "$GATE/queue_times.tsv"
}
export -f run_one; export FIT_PY REPO GATE
T0=$(date +%s.%N)
xargs -a "$GATE/queue_order.txt" -P 8 -I{} bash -c 'run_one {}'
T1=$(date +%s.%N)
echo "QUEUE wall_s=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.1f", b - a}') fits=$(wc -l < "$GATE/queue_times.tsv") failed=$(awk -F'\t' '$4 != 0' "$GATE/queue_times.tsv" | wc -l)" \
  | tee "$REPORT/queue_summary.txt"
cp "$GATE/queue_times.tsv" "$REPORT/"

# ---- 4. completion gate (exit 1 on any missing/invalid latent) ----
"$FIT_PY" - "$GATE" "$REPORT" "$EXPECT_GPU" "$(git rev-parse HEAD)" <<'PYEOF'
import json, sys, os
import numpy as np
gate, report, gpu, head = sys.argv[1:]
order = [l.strip() for l in open(os.path.join(gate, "queue_order.txt")) if l.strip()]
rows, bad = [], []
times = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(os.path.join(gate, "queue_times.tsv"))}
for t in order:
    f = os.path.join(gate, "bench_out", "latents", f"{t}.npz")
    if t not in times or times[t][3] != "0":
        bad.append(f"{t}: rc={times.get(t, ['', '', '', 'missing'])[3]}"); continue
    if not os.path.exists(f):
        bad.append(f"{t}: no latent"); continue
    d = np.load(f)
    z, cfg = d["z"], json.loads(str(d["config"]))
    ok = z.shape == (33506, 10) and bool(np.isfinite(z).all()) and cfg["git_sha"] == head and not cfg["git_dirty"] \
        and gpu in cfg["gpu"] and cfg["row"]["tag"] == t
    rows.append(dict(tag=t, shape=list(z.shape), finite=bool(np.isfinite(z).all()), git_sha=cfg["git_sha"],
                     git_dirty=cfg["git_dirty"], gpu=cfg["gpu"], torch=cfg["torch"], fit_seconds=cfg["fit_seconds"]))
    if not ok:
        bad.append(f"{t}: invalid latent/provenance {rows[-1]}")
json.dump(dict(expected=len(order), valid=len(order) - len(bad), problems=bad, latents=rows),
          open(os.path.join(report, "gate_completion.json"), "w"), indent=1)
print(f"completion gate: {len(order) - len(bad)}/{len(order)} valid")
if bad:
    print("\n".join(bad)); sys.exit(1)
PYEOF
# 8-lane factor on this GPU with the LOCAL single-lane cost model (throughput ratio vs the local 3.77x)
"$FIT_PY" scripts/calibrate_concurrency.py --manifest "$GATE/bench_queue_manifest.tsv" --times "$GATE/queue_times.tsv" \
    --throughput docs/throughput_rtx3080_stock_backbone.csv --out "$REPORT/concurrency_calibration_jhpce.json"
echo "GATE GPU PART OK"
