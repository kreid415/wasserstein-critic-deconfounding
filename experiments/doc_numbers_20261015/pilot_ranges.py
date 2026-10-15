"""Pilot ranges behind the corner of the frontier area (ANALYSIS_SPEC section 3 and Appendix A.1 Q5; SI-72; DOC-R04).

From the A1 pilot scores (seeds 100-102; 5 non-Sinkhorn arms x 6 task-by-decoder cells x 10 penalty strengths = 300 points):
the biology score C of plain scVI (lambda = 0) per cell, and the shift of the seed-mean C of every other point from it.
Reads the pilot scores of tier12_v2/A1 and the manifest of record; no X1 or X13 score.

Origin: the documents re-check of 2026-10-14 (its script 02, block N13). Usage: python experiments/doc_numbers_20261015/pilot_ranges.py
(env scvi-api; seconds).
"""
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, data_path, write_inputs  # noqa: E402

import pandas as pd  # noqa: E402

import prereg_rules as P  # noqa: E402

MANIFEST = REPO / "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"
mo = pd.read_csv(MANIFEST, sep="\t", comment="#", dtype=str)
M1 = mo[(mo.experiment == "A1") & (mo.arm != "sinkhorn")]
files = sorted(glob.glob(str(data_path("tier12_v2", "A1", "scores", "*.csv"))))
S = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
D = M1.merge(S[["tag"] + P.BATCH_METRICS + P.BIO_METRICS], on="tag", validate="one_to_one")
D["C"] = [sum(float(r[m]) for m in P.bio_metrics_for(r["task"])) / len(P.bio_metrics_for(r["task"])) for r in D.to_dict("records")]
D["cond"] = D.cond.astype(int)
D["lam"] = D.lam.astype(float)
zero = D[D.arm == "none"].groupby(["task", "cond"]).C.mean()
adv = D[D.arm != "none"].groupby(["arm", "task", "cond", "lam"]).C.mean().reset_index()
adv["dC"] = [r.C - zero[(r.task, r.cond)] for r in adv.itertuples()]
q = adv.dC.quantile([0.25, 0.75]).tolist()
cq = adv.groupby(["task", "cond"]).dC.quantile([0.25, 0.75]).unstack()
print(f"score files {len(files)}, A1 manifest rows without Sinkhorn {len(M1)}")
print(f"plain scVI C per cell (6 cells): {zero.min():.3f} to {zero.max():.3f}")
print(f"points (5 non-Sinkhorn arms x 6 cells x 10 penalty strengths): {len(adv)}")
print(f"shift of C from plain scVI, all points: {adv.dC.min():.3f} to {adv.dC.max():.3f}")
print(f"shift of C, middle half of all points: {q[0]:.4f} to {q[1]:.4f}")
print(f"shift of C, within each cell the middle half lies inside: {cq[0.25].min():.3f} to {cq[0.75].max():.3f}")
# Quoted in ANALYSIS_SPEC Appendix A.1 Q5 (documents round to two decimals) and in section 3 (300 points)
assert len(adv) == 300 and len(zero) == 6
assert (f"{zero.min():.2f}", f"{zero.max():.2f}") == ("0.67", "0.90")
assert (f"{adv.dC.min():.2f}", f"{adv.dC.max():.2f}") == ("-0.59", "0.23")
assert abs(q[0] - -0.09) < 0.006 and abs(q[1] - 0.0) < 0.006
assert abs(cq[0.25].min() - -0.19) < 0.005 and abs(cq[0.75].max() - 0.09) < 0.005
print("inputs ledger SHA-256:", write_inputs("pilot_ranges", [MANIFEST] + [Path(f) for f in files]))
