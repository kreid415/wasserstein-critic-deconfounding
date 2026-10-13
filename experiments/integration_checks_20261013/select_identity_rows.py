"""Rows for the 2026-10-13 fit-identity check (tag c88cce3 vs branch final-review-fixes; SI-52): one row for every
combination of arm, extra keys, subsample kind, adversary input, standardisation, decoder and decoder conditioning
that occurs in the committed manifests, so that every code path of every arm is fitted once at each commit. Rows whose
lambda, adv_input or zstd is still a placeholder (X1 g1..g6, A2, A3) repeat the code path of a resolved row and are
skipped; other lambda placeholders (matched*) become 10. Retired pre-SI-41 X3 rows (composition subsample without
draw nested_v1, e.g. in paper_manifest_stock_pilot_u5_b10.tsv) are refused by both commits and skipped. Prefers the
smallest task. max_epochs = 1.
Usage: python select_identity_rows.py OUT.tsv
"""
import glob
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIZE = ["sim2", "atac_small", "sim1", "pancreas", "lung", "immune", "atac_large", "immune_hum_mou"]   # small first
CPU_ARMS = {"harmony", "pca", "scanorama"}           # X13 CPU baselines: run_cpu_baselines.py, unchanged by the branch

files = sorted(glob.glob(os.path.join(REPO, "manifests", "*.tsv")))
head = open(files[-1]).readline()
m = pd.concat([pd.read_csv(f, sep="\t", comment="#", dtype=str) for f in files]).drop_duplicates("tag")
m = m[~m.arm.isin(CPU_ARMS) & m.adv_input.isin(["mean", "sample"]) & m.zstd.isin(["0", "1"])]
m = m[~m.lam.str.match(r"^g\d$")].copy()
sub_spec = m.extra.map(lambda e: json.loads(e).get("subsample", {}))
# pre-SI-41 X3 rows (composition subsample without draw nested_v1): fit_paper_config.x3_draw refuses them at both commits
m = m[~sub_spec.map(lambda s: s.get("kind") == "composition" and s.get("draw") != "nested_v1")].copy()
m["keys"] = m.extra.map(lambda e: ",".join(sorted(json.loads(e))))
m["sub"] = m.extra.map(lambda e: json.loads(e).get("subsample", {}).get("kind", ""))
m["size"] = m.task.map(SIZE.index)
combo = ["arm", "keys", "sub", "adv_input", "zstd", "decoder", "cond"]
sel = m.sort_values(["size", "tag"]).groupby(combo, as_index=False).head(1).copy()
sel.loc[~sel.lam.str.match(r"^[0-9.eE+-]+$"), "lam"] = "10"
sel["max_epochs"] = "1"
cols = [c for c in m.columns if c not in ("keys", "sub", "size")]
with open(sys.argv[1], "w") as f:
    f.write(head if head.startswith("#") else "")
    sel.sort_values(combo)[cols].to_csv(f, sep="\t", index=False)
print(len(sel), "rows;", sel.arm.value_counts().to_dict())
print(sel[combo + ["task", "experiment", "lam"]].to_string(index=False))
