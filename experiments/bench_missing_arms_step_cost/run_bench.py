#!/usr/bin/env python
"""Single-lane step-cost benchmark of the new arms on immune (docs/SPECS_missing_arms.md), same method as
docs/throughput_rtx3080_stock_backbone.csv: a 3-epoch and a 1-epoch fit of each configuration through
scripts/fit_paper_config.py, ms per optimiser step = (t3 - t1) / (2 * steps_per_epoch) (scripts/cost_model.py).
Additions to that method: 3 interleaved repeats per arm, 4 existing arms re-measured in the same session as
controls, and a GPU snapshot (utilisation, other compute processes) before and after every fit.

Usage (env scvi-api):
  python experiments/bench_missing_arms_step_cost/run_bench.py --manifest experiments/bench_missing_arms_step_cost/bench_manifest.tsv \
      --out-dir <durable dir> [--only TAG ...]
Fits run one at a time (single lane, OMP_NUM_THREADS=1). Exit 1 if any fit fails or an output is missing.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def gpu_snapshot():
    util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True).stdout.strip()
    return util, (apps.replace("\n", ";") if apps else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prepped-dir", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="run only these tags (smoke run)")
    a = ap.parse_args()
    m = pd.read_csv(a.manifest, sep="\t", dtype=str, keep_default_na=False)
    tags = list(m.tag) if not a.only else a.only
    unknown = sorted(set(tags) - set(m.tag))
    if unknown:
        raise KeyError(f"tags not in the manifest: {unknown}")
    os.makedirs(a.out_dir, exist_ok=True)
    log_path = os.path.join(a.out_dir, "bench_log.csv")
    new_log = not os.path.exists(log_path)
    env = dict(os.environ, MANIFEST=os.path.abspath(a.manifest), PREPPED_DIR=a.prepped_dir, OUT_DIR=a.out_dir,
               WCD_SRC=os.path.join(ROOT, "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KMP_AFFINITY="disabled")
    with open(log_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["tag", "start_unix", "end_unix", "wall_s", "gpu_before", "apps_before", "gpu_after", "apps_after"])
        for tag in tags:
            npz = os.path.join(a.out_dir, "latents", f"{tag}.npz")
            if os.path.exists(npz):
                raise FileExistsError(f"{npz} exists: a timing must come from a fresh fit (delete it or use a new out-dir)")
            g0, p0 = gpu_snapshot()
            t0 = time.time()
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fit_paper_config.py")], check=True,
                           env=dict(env, TAG=tag))
            t1 = time.time()
            g1, p1 = gpu_snapshot()
            w.writerow([tag, round(t0, 1), round(t1, 1), round(t1 - t0, 1), g0, p0, g1, p1])
            fh.flush()
            print(f"[bench] {tag} {t1 - t0:.1f}s gpu_before={g0} apps_before={p0 or '-'}", flush=True)
    missing = [t for t in tags if not os.path.exists(os.path.join(a.out_dir, "latents", f"{t}.npz"))]
    if missing:
        print(f"[bench] missing outputs: {missing}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
