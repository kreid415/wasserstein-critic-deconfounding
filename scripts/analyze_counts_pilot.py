#!/usr/bin/env python
"""Counts pilot analysis (scIB native). Two scaling sets, each min-max scaled on its own
(scIB 'within a task'): NEW = real counts (this pilot), OLD = final-sweep log-data latents at the
same arms/lambdas/seeds. Unintegrated is included in each set's scaling, as in scIB.

Outputs (to OUT dir): pilot_rows_scored.csv (every row + scaled metrics + overall),
pilot_summary.csv (mean +/- sd over seeds per set/cond/arm/lambda), pilot_contrasts.csv
(critic vs discriminator: grid-average and leave-one-seed-out lambda selection).
Usage: python scripts/analyze_counts_pilot.py <rows_dir> <out_dir>
"""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from score_scib_native import scib_overall, BATCH_METRICS, BIO_METRICS

rows_dir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{rows_dir}/*.csv"))], ignore_index=True)
df = scib_overall(df)
df.to_csv(f"{out}/pilot_rows_scored.csv", index=False)

keys = ["input", "cond", "adv", "lam"]
S = (df.groupby(keys)[["overall", "batch_score", "bio_score"]]
       .agg(["mean", "std", "count"]).round(4))
S.columns = ["_".join(c) for c in S.columns]
S.reset_index().to_csv(f"{out}/pilot_summary.csv", index=False)


def loso(g):
    """leave-one-seed-out: choose lambda on the other seeds' mean, score the held-out seed."""
    P = g.pivot_table(index="lam", columns="seed", values="overall")
    v = [P.loc[P.drop(columns=s).mean(axis=1).idxmax(), s] for s in P.columns]
    return float(np.nanmean(v))


C = []
for (inp, cond), g in df[df.adv.isin(["discriminator", "reference", "pooled"])].groupby(["input", "cond"]):
    rec = dict(input=inp, cond=cond)
    for arm, ga in g.groupby("adv"):
        rec[f"{arm}_gridmean"] = ga.overall.mean()
        rec[f"{arm}_loso"] = loso(ga)
    C.append(rec)
C = pd.DataFrame(C).round(4)
C.to_csv(f"{out}/pilot_contrasts.csv", index=False)
pd.set_option("display.width", 220)
print(S.reset_index()[keys + ["overall_mean", "overall_std", "overall_count", "batch_score_mean", "bio_score_mean"]].to_string(index=False))
print("\n", C.to_string(index=False))
