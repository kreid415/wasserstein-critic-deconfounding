#!/usr/bin/env python
"""Summarise the step-cost benchmark of the missing arms (run_bench.py output) with scripts/cost_model.py's own
measurement: ms per optimiser step = (t3 - t1) / (2 * steps_per_epoch) for each repeat.

Writes
  docs/throughput_missing_arms_repeats.csv   every repeat of every arm (new arms and same-session controls)
  docs/throughput_rtx3080_stock_backbone.csv  + one row per NEW arm label: the repeat with the median ms/step
                                              (refuses if rows for these labels exist already)
Prints the controls' re-measured ms/step next to their existing rows in the stock-backbone CSV.
Usage: python experiments/bench_missing_arms_step_cost/summarise.py --bench-dir <run_bench out-dir>
"""
import argparse
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cost_model  # noqa: E402

NEW = ["discriminator_r1", "discriminator_ref", "reference_stratified", "pooled_stratified", "discriminator_iw",
       "reference_iw", "pooled_iw", "mmd_iw"]
CONTROLS = ["discriminator", "reference", "pooled", "mmd"]
N_REPEATS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--stock-csv", default=os.path.join(ROOT, "docs", "throughput_rtx3080_stock_backbone.csv"))
    ap.add_argument("--repeats-csv", default=os.path.join(ROOT, "docs", "throughput_missing_arms_repeats.csv"))
    a = ap.parse_args()
    T, _ = cost_model.measure(os.path.join(a.bench_dir, "latents"), os.path.join(a.bench_dir, "measured.csv"))
    T = T.copy()
    m = T.tag.str.extract(r"^S_immune_(?P<label>.+)_rep(?P<rep>\d+)$")
    if m.isna().any().any():
        raise ValueError(f"unexpected tags: {list(T.tag[m.isna().any(axis=1)])}")
    T["label"], T["rep"] = m["label"], m["rep"].astype(int)
    if not (T.label == T.arm).all():
        bad = T[T.label != T.arm][["tag", "arm"]].values.tolist()
        raise ValueError(f"cost_model arm label differs from the bench label: {bad}")
    counts = T.groupby("label").size()
    want = set(NEW + CONTROLS)
    if set(counts.index) != want or (counts != N_REPEATS).any():
        raise ValueError(f"expected {N_REPEATS} repeats of {sorted(want)}, got {counts.to_dict()}")
    T = T.sort_values(["label", "rep"])
    T.to_csv(a.repeats_csv, index=False)
    S = pd.read_csv(a.stock_csv)
    clash = sorted(set(S.arm) & set(NEW))
    if clash:
        raise ValueError(f"{a.stock_csv} already has rows for {clash}; not appending twice")
    pick = []
    for lab in NEW:
        g = T[T.label == lab].sort_values("ms_per_step")
        pick.append(g.iloc[len(g) // 2])
    add = pd.DataFrame(pick)[list(S.columns)]
    pd.concat([S, add], ignore_index=True).to_csv(a.stock_csv, index=False)
    summ = T.groupby("label").ms_per_step.agg(["median", "min", "max"]).round(2)
    summ["spread"] = (summ["max"] / summ["min"]).round(3)
    ctl_nc = {"discriminator": 1, "reference": 5, "pooled": 5, "mmd": 0}
    old = {}
    for lab, nc in ctl_nc.items():
        hit = S[(S.task == "immune") & (S.batch_size == 128) & (S.arm == lab) & (S.n_critic == nc)]
        if len(hit) != 1:
            raise ValueError(f"expected one existing immune row for control {lab} (n_critic {nc}), got {len(hit)}")
        old[lab] = float(hit.ms_per_step.iloc[0])
    summ["existing_csv"] = [old.get(l, float("nan")) for l in summ.index]
    print(summ.to_string())
    print(f"appended {len(add)} rows to {a.stock_csv}; repeats -> {a.repeats_csv}")


if __name__ == "__main__":
    main()
