"""Pilot collapse counts behind SI-75 (P2) and the recount of 2026-10-14 (NB-20261014-22, NB-20261015-07; DOC-R04).

A run collapses when its biology score C is at least 0.10 below the C of the same seed's lambda = 0 run (the R1 collapse bound
applied per run; failed fits do not occur in the pilot rows used here). Counted over the six registered X1 penalty strengths of
the reference critic (W1 family) and of the discriminator (JS family) in the A1 pilot and the R1 extension rows, Sinkhorn rows
dropped (the Sinkhorn pilot fits are to be redone, SI-57). Reads the pilot scores of tier12_v2/A1 and tier12_v2/A1_ext_r1_round1,
the manifest of record, the manifest of the R1 extension and docs/prereg/r1_record.json. No X1 or X13 score.

Origin: the documents re-check of 2026-10-14 (its script 04). Usage: python experiments/doc_numbers_20261015/p2_pilot_recount.py
(env scvi-api; about 1 minute).
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, RESULTS, data_path, write_inputs  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import prereg_rules as P  # noqa: E402

rd = lambda f: pd.read_csv(REPO / f, sep="\t", comment="#", dtype=str)  # noqa: E731
M, Mx = rd("manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"), rd("manifests/a1_extension_r1_round1.tsv")
man = pd.concat([M[M.experiment == "A1"].assign(src="A1"), Mx.assign(src="ext")]).drop_duplicates("tag")
man = man[man.arm != "sinkhorn"]
files = sorted(glob.glob(str(data_path("tier12_v2", "A1", "scores", "*.csv")))) + \
    sorted(glob.glob(str(data_path("tier12_v2", "A1_ext_r1_round1", "scores", "*.csv"))))
S = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
D = man.merge(S[["tag"] + P.BATCH_METRICS + P.BIO_METRICS], on="tag", how="left", validate="one_to_one", indicator=True)
print("score files", len(files), "| manifest rows without Sinkhorn", len(man), "of which R1 extension", int((man.src == "ext").sum()),
      "| rows without a score:", int((D._merge != "both").sum()))
D = D[D._merge == "both"].copy()
D["C"] = [float(np.mean([float(r[m]) for m in P.bio_metrics_for(r["task"])])) for r in D.to_dict("records")]
for c in ("cond", "seed"):
    D[c] = D[c].astype(int)
D["lam"] = D.lam.astype(float)
zero = D[D.arm == "none"].set_index(["task", "cond", "seed"]).C
assert zero.index.is_unique and len(zero) == 18
D["collapsed"] = [bool(r.C <= zero[(r.task, r.cond, r.seed)] - 0.10 + 1e-12) for r in D.itertuples()]
rec = json.load(open(REPO / "docs/prereg/r1_record.json"))
out = {}


def count(arm, fam, cond, with_ext):
    grid = rec["families"][f"{fam}|{cond}"]["x1_grid"]
    sub = D[(D.arm == arm) & (D.cond == cond) & D.lam.isin(grid)]
    if not with_ext:
        sub = sub[sub.src == "A1"]
    return int(sub.collapsed.sum()), len(sub), [float(x) for x in sorted(sub.lam.unique())]


for with_ext in (False, True):
    print("\n-- with the R1 extension rows:", with_ext)
    for cond in (0, 1):
        r, d = count("reference", "w1", cond, with_ext), count("discriminator", "js", cond, with_ext)
        diff = 100 * (r[0] / r[1] - d[0] / d[1])
        print(f"decoder {'conditioned' if cond else 'unconditioned'}: reference {r[0]}/{r[1]}, discriminator {d[0]}/{d[1]}, "
              f"critic minus discriminator {diff:+.1f} points; discriminator strengths {d[2]}")
        out[f"ext{int(with_ext)}_cond{cond}"] = dict(reference=list(r[:2]), discriminator=list(d[:2]), diff_points=round(diff, 2))
e = D[(D.src == "ext") & (D.arm == "discriminator")]
ext = e.groupby("lam").collapsed.agg(["sum", "size"]).to_dict("index")
print("\nR1 extension rows of the discriminator (unconditioned decoder), collapsed / runs by strength:", ext)
RESULTS.mkdir(exist_ok=True)
json.dump(out, open(RESULTS / "p2_pilot_recount.json", "w"), indent=1)

# The numbers quoted in the documents (PREREG section 9 table row 8; ANALYSIS_SPEC Appendix A.1 Q9 and A.3)
assert out["ext1_cond0"] == dict(reference=[10, 54], discriminator=[8, 54], diff_points=3.7)
assert out["ext0_cond0"] == dict(reference=[10, 54], discriminator=[8, 45], diff_points=0.74)
assert out["ext1_cond1"]["diff_points"] == out["ext0_cond1"]["diff_points"] == -3.7
assert ext[0.03] == {"sum": 0, "size": 9}
print("\nquoted: discriminator 8 of 54 (not 8 of 45) with the nine lambda 0.03 rows, reference minus discriminator +3.7 (not +0.7) "
      "points for the unconditioned decoder, -3.7 for the conditioned; 0 of the 9 lambda 0.03 runs collapsed")
manifests = [REPO / "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv", REPO / "manifests/a1_extension_r1_round1.tsv",
             REPO / "docs/prereg/r1_record.json"]
print("inputs ledger SHA-256:", write_inputs("p2_pilot_recount", manifests + [Path(f) for f in files]))
