"""Count of the earlier X1 and X13 baseline fits that depend on no rule (PREREG section 0 and section 9, disclosure entry; DOC-R02).

Counts the rows of the manifest of record for the X1 arms none and scvi_adv and the X13 arms scanvi and sysvi on the five tasks
of the two cluster jobs (job A: atac_large, immune_hum_mou; job B: lung, pancreas, sim2; notebook NB-20261004-14, NB-20261006-01).
Reads the committed manifest only; no score.
Usage: python experiments/doc_numbers_20261015/filler_fit_counts.py (env scvi-api; seconds).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, write_inputs  # noqa: E402

import pandas as pd  # noqa: E402

MANIFEST = REPO / "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"
M = pd.read_csv(MANIFEST, sep="\t", comment="#", dtype=str)
f = M[M.experiment.isin(["X1", "X13"]) & M.arm.isin(["none", "scvi_adv", "scanvi", "sysvi"])]
job = {"atac_large": "A", "immune_hum_mou": "A", "lung": "B", "pancreas": "B", "sim2": "B"}
g = f[f.task.isin(job)].assign(job=lambda d: d.task.map(job))
print("rows of the X1 arms none and scvi_adv and the X13 arms scanvi and sysvi on all 8 tasks:", len(f))
print("on the five cluster-job tasks:", len(g))
print("by job:", g.groupby("job").size().to_dict())
print("by task:", g.groupby("task").size().to_dict())
print("by experiment and arm:", {f"{e} {a}": n for (e, a), n in g.groupby(["experiment", "arm"]).size().items()})
assert (len(g), g.groupby("job").size().to_dict()) == (250, {"A": 100, "B": 150})
assert set(g.groupby("task").size()) == {50}
print("inputs ledger SHA-256:", write_inputs("filler_fit_counts", [MANIFEST]))
