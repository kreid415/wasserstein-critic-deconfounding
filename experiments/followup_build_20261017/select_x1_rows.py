"""Rows for the X1 fit-identity check of the follow-up build (SI-93..SI-98): one X1 or X13 neural-baseline row for every
combination of arm, decoder conditioning, adversary input (mean / sample) and standardisation (0 / 1) that X1 can take
once its placeholders (lambda g1..g6, adv_input A2, zstd A3) are resolved, on the smallest task, lambda 10 for adversarial
arms, max_epochs 1 (53 rows, scANVI and sysVI included), plus the four conditioned oracle-weighted X3 rows (sim2, dose 50;
the weight code around this path changed). Reads the v3 manifest (design rows only, no scores). Tags get a suffix naming the combination.
Usage: python select_x1_rows.py OUT.tsv
"""
import itertools
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V3 = os.path.join(REPO, "manifests", "paper_manifest_stock_pilot_u5_b10_v3.tsv")
m = pd.read_csv(V3, sep="\t", comment="#", dtype=str, keep_default_na=False)
head = open(V3).readline()
m = m[(m.task == "sim2") & (m.seed == "0") & (m.experiment.isin(["X1", "X13"]))]
out = []
for arm, cond in sorted(set(zip(m.arm, m.cond))):
    src = m[(m.arm == arm) & (m.cond == cond)]
    src = src[src.lam.isin(["0", "g1", "1"])].iloc[0] if arm in ("none", "scvi_adv", "scanvi") else src.iloc[0]
    adversarial = arm not in ("none", "scvi_adv", "scanvi", "sysvi", "harmony", "pca", "scanorama")
    if arm in ("harmony", "pca", "scanorama"):
        continue                                            # X13 CPU rows: run_cpu_baselines.py, not touched
    for adv, z in (itertools.product(["mean", "sample"], ["0", "1"]) if adversarial else [("mean", "0")]):
        r = src.copy()
        r["lam"] = "10" if adversarial or arm == "sysvi" else r["lam"]
        r["adv_input"], r["zstd"], r["max_epochs"] = adv, z, "1"
        r["tag"] = f"{src['tag']}_id{adv}z{z}"
        out.append(r)
# conditioned oracle-weighted X3 rows (SI-93 changes the weight code around this path; it must stay identical): one per
# weighted arm, smallest X3 task, dose 50, seed 10, lambda 10
w3 = m3 = pd.read_csv(V3, sep="\t", comment="#", dtype=str, keep_default_na=False)
w3 = w3[(w3.experiment == "X3") & (w3.task == "sim2") & (w3.cond == "1") & w3.extra.str.contains('"iw"')
        & w3.extra.str.contains('"deplete_pct": 50') & (w3.seed == "10") & (w3.lam == "matched_lo")]
assert sorted(w3.arm) == ["discriminator", "mmd", "pooled", "reference"], sorted(w3.arm)
for _, r in w3.iterrows():
    r = r.copy()
    r["lam"], r["max_epochs"], r["tag"] = "10", "1", f"{r['tag']}_idweighted"
    out.append(r)
d = pd.DataFrame(out)
assert d.tag.is_unique and len(d) == 57, len(d)
with open(sys.argv[1], "w") as f:
    f.write(head)
    d.to_csv(f, sep="\t", index=False)
print(len(d), "rows;", d.arm.value_counts().to_dict())
