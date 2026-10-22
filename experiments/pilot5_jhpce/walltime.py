#!/usr/bin/env python
"""Predicted wall time of the five pilot5 JHPCE fit jobs (one per dataset), from the manifest's row counts.

Per-fit means: measured in the main benchmark on immune_hum_mou, one L40S GPU shared by about 8 fits (values given by the
user in the task of 2026-10-21; no ledger timing column was opened). 'none' (strength 0, plain scVI) was not measured: it
is priced as a discriminator fit, an upper bound. Wall time = sum of fit seconds / 8 lanes; 'tail' = one longest-arm fit
that runs alone at the end; 'scaled' multiplies by training work relative to immune_hum_mou, cells x epochs
(epochs = scvi-tools heuristic, build_paper_manifest.scvi_epochs), assuming the same cost per training step.
Usage: python3 experiments/pilot5_jhpce/walltime.py [manifest]
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_paper_manifest as BPM  # noqa: E402

MEAN_S = {"discriminator": 1235, "mmd": 1199, "sinkhorn": 2866, "pooled": 4055, "reference": 5039, "barycenter": 5828}
MEAN_S["none"] = MEAN_S["discriminator"]
LANES, LIMIT_H, REF = 8, 72, "immune_hum_mou"


def main(path):
    with open(path) as f:
        rows = list(csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"))
    work = lambda t: BPM.TASKS[t][0] * BPM.scvi_epochs(BPM.TASKS[t][0])   # noqa: E731
    print(f"per-fit mean s {MEAN_S}; {LANES} fits share a GPU; limit {LIMIT_H} h")
    print("dataset          fits  fit-seconds  wall h  +tail h  margin to limit  work vs ref  scaled wall+tail h")
    for t in sorted({r["task"] for r in rows}):
        arms = [r["arm"] for r in rows if r["task"] == t]
        fit_s = sum(MEAN_S[a] for a in arms)
        wall, tail = fit_s / LANES / 3600, max(MEAN_S[a] for a in set(arms)) / 3600
        ratio = work(t) / work(REF)
        print(f"{t:<16} {len(arms):>4} {fit_s:>12,} {wall:>7.1f} {tail:>8.1f} {LIMIT_H / (wall + tail):>15.2f}x "
              f"{ratio:>12.2f} {ratio * wall + tail:>19.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv"))
