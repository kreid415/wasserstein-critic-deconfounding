#!/usr/bin/env python
"""Known-answer checks for adding trajectory conservation to the bio score (CONSTRAINTS.md SI-28; PREFLIGHT.md
in this folder). Run from the repo root in an env with scanpy + scib (wcd-gpu or wcd-kbet):
    KMP_AFFINITY=disabled python experiments/metric_bio_trajectory/check_bio_trajectory.py
Writes experiments/metric_bio_trajectory/check_bio_trajectory.json and exits non-zero on any failure."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prereg_rules as P  # noqa: E402
import score_scib_native as SN  # noqa: E402

out = {}
# 1. prereg bio score C (unscaled mean of the task's bio metrics): known answers per task
vals = {m: 0.1 * (j + 1) for j, m in enumerate(SN.BIO_METRICS)}           # 0.1 ... 0.8, trajectory = 0.8
for task in sorted(P.BPM.TASKS):
    used = P.bio_metrics_for(task)
    expect = float(np.mean([vals[m] for m in used]))
    out[f"C|{task}"] = dict(metrics=used, expected=expect)
for task, k in (("immune", 8), ("immune_hum_mou", 8), ("pancreas", 7), ("lung", 7), ("atac_small", 6),
                ("atac_large", 6), ("sim1", 6), ("sim2", 6)):
    assert len(out[f"C|{task}"]["metrics"]) == k, (task, out[f"C|{task}"]["metrics"])
    assert np.isclose(out[f"C|{task}"]["expected"], 0.05 * (k + 1)), (task, out[f"C|{task}"]["expected"])
# 2. scib_overall: unchanged where trajectory is NaN; averaged in on the immune tasks
rows = []
for d, traj in (("pancreas", [np.nan] * 4), ("immune", [0.9, 0.1, 0.5, 0.3])):
    for i in range(4):
        r = {"dataset": d, **{m: 0.1 * (i + 1) for m in SN.BATCH_METRICS}}
        r.update({m: 0.2 + 0.1 * ((i * (j + 1)) % 4) + 0.01 * j for j, m in enumerate(SN.BIO_METRICS)})
        r["trajectory"] = traj[i]
        rows.append(r)
t = pd.DataFrame(rows)
o, o7 = SN.scib_overall(t), SN.scib_overall(t.drop(columns="trajectory"))
pan, imm = t.dataset == "pancreas", t.dataset == "immune"
out["scib_overall_pancreas_max_abs_change"] = float(np.abs(o.bio_score[pan] - o7.bio_score[pan]).max())
out["scib_overall_immune_max_abs_change"] = float(np.abs(o.bio_score[imm] - o7.bio_score[imm]).max())
assert out["scib_overall_pancreas_max_abs_change"] <= 1e-12        # floating-point summation order only
assert out["scib_overall_immune_max_abs_change"] > 1e-3
out["BIO_METRICS"] = SN.BIO_METRICS
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_bio_trajectory.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print(json.dumps({k: v for k, v in out.items() if k.startswith("scib")}), "->", path)
