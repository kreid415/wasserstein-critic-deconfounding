#!/usr/bin/env python
"""Feasibility timing for the X13 Scanorama knn grid (SI-35, SI-37) on immune_hum_mou (97,861 cells), the largest task.

Runs the manifest rows knn 20 (tool default, reference), 80 and 160 one at a time through scripts/run_cpu_baselines.py,
each in its own process at nice 19 with 2 threads, a resident-memory cap on the whole process tree (polled every 2 s;
a virtual-memory rlimit would also count address space that libraries reserve but never touch) and a wall-clock cap,
while the A1 stage occupies the GPU and 11 of 12 cores. Records wall seconds, the peak tree RSS, the peak RSS reported by
/usr/bin/time -v for the main process, and the exit status; a row that hits a cap is recorded as such, never retried.
Outputs are timing evidence only (not benchmark latents).
Usage (repo root, wcd-kbet env): python experiments/x13_scanorama_timing/time_scanorama.py --out-dir DIR
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time

TAGS = {20: "X13_immune_hum_mou_scanorama_l20_c0_s0_06b0ab06", 80: "X13_immune_hum_mou_scanorama_l80_c0_s0_890fbda4",
        160: "X13_immune_hum_mou_scanorama_l160_c0_s0_649aa7d2"}


def _tree_rss_kb(root):
    """Sum of VmRSS over root and all its descendants (this PID namespace's /proc)."""
    kids, rss = {}, {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            with open(f"/proc/{d}/status") as f:
                v = [l for l in f if l.startswith("VmRSS:")]
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(d))
        rss[int(d)] = int(v[0].split()[1]) if v else 0
    tot, todo = 0, [root]
    while todo:
        p = todo.pop()
        tot += rss.get(p, 0)
        todo.extend(kids.get(p, []))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default="manifests/paper_manifest_stock_pilot_u5_b10.tsv")
    ap.add_argument("--prepped-dir", default="/home/kendall/experiment_data/wasserstein-critic-deconfounding/prepped_scib")
    ap.add_argument("--rss-cap-gb", type=float, default=24.0)
    ap.add_argument("--timeout-h", type=float, default=6.0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res_csv = os.path.join(a.out_dir, "scanorama_timing.csv")
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads),
               NUMBA_NUM_THREADS=str(a.threads), KMP_AFFINITY="disabled")
    for knn in (20, 80, 160):
        log = os.path.join(a.out_dir, f"knn{knn}.log")
        cmd = ["/usr/bin/time", "-v", "nice", "-n", "19", sys.executable, "scripts/run_cpu_baselines.py", "--manifest", a.manifest,
               "--prepped-dir", a.prepped_dir, "--out-dir", os.path.join(a.out_dir, f"knn{knn}"), "--tag", TAGS[knn]]
        t0 = time.time()
        cap_kb, peak_tree_kb, status = int(a.rss_cap_gb * 1024 ** 2), 0, None
        with open(log, "w") as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            while p.poll() is None:
                peak_tree_kb = max(peak_tree_kb, _tree_rss_kb(p.pid))
                if peak_tree_kb > cap_kb:
                    status = f"rss cap {a.rss_cap_gb} GB exceeded"
                elif time.time() - t0 > a.timeout_h * 3600:
                    status = f"timeout after {a.timeout_h} h"
                if status:
                    os.killpg(p.pid, signal.SIGKILL); p.wait(); break
                time.sleep(2)
            status = status or ("ok" if p.returncode == 0 else f"exit {p.returncode}")
        secs = time.time() - t0
        peak = [l for l in open(log) if "Maximum resident set size (kbytes)" in l]
        peak_gb = round(int(peak[-1].split(":")[1]) / 1024 ** 2, 2) if peak else None   # absent if killed before exit
        row = dict(task="immune_hum_mou", knn=knn, tag=TAGS[knn], status=status, wall_s=round(secs, 1),
                   peak_tree_rss_gb=round(peak_tree_kb / 1024 ** 2, 2), peak_rss_gb_main=peak_gb, rss_cap_gb=a.rss_cap_gb,
                   threads=a.threads, nice=19, log=log)
        new = not os.path.exists(res_csv)
        with open(res_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader() if new else None; w.writerow(row)
        print(row, flush=True)


if __name__ == "__main__":
    main()
