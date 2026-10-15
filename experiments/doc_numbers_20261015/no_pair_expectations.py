"""Expected number of fits that meet at least one minibatch with no pair of batches (PREREG section 9, guards entry; DOC-R04).

Critic-free losses (mmd, mmd_ref, sinkhorn) compare the batches present in a minibatch. A minibatch that holds only one batch
(for mmd_ref: no reference cell, or no other batch) has nothing to compare (finding I033, guard G-01). This script computes,
for every critic-free row of the manifest of record, the probability that an epoch contains such a minibatch, assuming
scvi-tools' loader (training cells shuffled every epoch, batch size 128, the partial last minibatch kept, n_train = ceil(0.9 N);
no_pair_assumptions.py reads these assumptions from the installed scvi-tools source) and exact hypergeometric probabilities.
It is a model, not simulated runs. Reads the manifest of record and the obs tables of the prepared datasets; no score.

Origin: the guard review's scripts/recompute_g01.py (not in the repository), reproduced by the documents re-check of 2026-10-14
(its script 03); here the paths are fixed and the numbers quoted in PREREG are asserted at the end.
Usage: python experiments/doc_numbers_20261015/no_pair_expectations.py   (env scvi-api; about 1 minute)
"""
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, RESULTS, data_path, write_inputs  # noqa: E402

import anndata  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import gammaln  # noqa: E402

import fit_paper_config as F  # noqa: E402

warnings.filterwarnings("ignore")
BS = 128
MANIFEST = REPO / "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"


def lchoose(n, k):
    n = np.asarray(n, float)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def p_subset_in(counts, N, m):
    """P(an m-subset drawn without replacement from N cells lies entirely inside one group), summed over the group sizes."""
    counts = np.asarray([c for c in counts if c >= m], float)
    if len(counts) == 0 or m > N:
        return 0.0
    return float(np.exp(lchoose(counts, m) - lchoose(N, m)).sum())


class Shim:   # enough of AnnData for fit_paper_config.subsample (kind 'batches')
    def __init__(self, obs):
        self.obs = obs

    def __getitem__(self, idx):
        return Shim(self.obs.iloc[idx])

    def copy(self):
        return self


M = pd.read_csv(MANIFEST, sep="\t", comment="#", dtype=str)
cf = M[M.arm.isin(["mmd", "mmd_ref", "sinkhorn"])].copy()
obs, prep_files = {}, []
for t in sorted(cf.task.unique()):
    f = data_path("prepped_scib", f"{t}__scib.h5ad")
    prep_files.append(f)
    a = anndata.read_h5ad(f, backed="r")
    obs[t] = a.obs[["batch", "celltype"] if "celltype" in a.obs else ["batch"]].copy()
    if "celltype" not in obs[t]:
        obs[t]["celltype"] = a.obs[[c for c in a.obs.columns if c in ("cell_type", "final_annotation", "Group", "final_cell_label")][0]].values
    a.file.close()

import json  # noqa: E402
rows = []
for r in cf.to_dict("records"):
    o = obs[r["task"]]
    sub = json.loads(r["extra"]).get("subsample")
    if sub is None:
        used = o
    elif sub["kind"] == "batches":
        used = F.subsample(Shim(o), sub, int(r["seed"])).obs
    else:
        idx, _ = F.x3_draw(o, sub)
        used = o.iloc[idx]
    N = len(used)
    n_train = math.ceil(0.9 * N)
    steps = math.ceil(n_train / BS)
    m_last = n_train - BS * (steps - 1)
    cnt = used["batch"].astype(str).value_counts()
    scale = n_train / N
    counts_tr = cnt.values * scale
    E = int(r["max_epochs"])
    if r["arm"] == "mmd_ref":
        names = sorted(cnt.index)
        ref = names[int(r["reference"])] if str(r["reference"]).isdigit() else str(r["reference"])
        n_ref = cnt[ref] * scale

        def p_np(m):
            return p_subset_in([n_train - n_ref], n_train, m) + p_subset_in([n_ref], n_train, m)
    else:
        def p_np(m):
            return p_subset_in(counts_tr, n_train, m)
    per_epoch = steps * p_np(BS) if m_last == BS else (steps - 1) * p_np(BS) + p_np(m_last)
    rows.append(dict(experiment=r["experiment"], task=r["task"], arm=r["arm"], tag=r["tag"], N=N, n_train=n_train, m_last=m_last,
                     steps=steps, epochs=E, per_epoch=per_epoch, share=per_epoch / steps, p_fit=1 - math.exp(-E * per_epoch),
                     exp_steps=E * per_epoch))
R = pd.DataFrame(rows)
G = R.groupby(["experiment", "task", "arm"]).agg(
    fits=("tag", "size"), n_train=("n_train", "first"), m_last=("m_last", "first"), steps=("steps", "first"),
    epochs=("epochs", "first"), fits_expected_hit=("p_fit", "sum"), exp_no_pair_steps_total=("exp_steps", "sum"),
    share_of_steps=("share", "mean")).reset_index().sort_values("fits_expected_hit", ascending=False)
RESULTS.mkdir(exist_ok=True)
G.to_csv(RESULTS / "no_pair_groups.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(G.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("\nA1 and A3 rows are fits that already ran at c88cce3 without a crash; their expected counts are for reference only.")

# The numbers quoted in PREREG section 9 (guards entry, minibatches with nothing to compare)
def hit(exp, task, arm):
    g = G[(G.experiment == exp) & (G.task == task) & (G.arm == arm)].iloc[0]
    return int(g.fits), float(g.fits_expected_hit)

print("\nQuoted in PREREG section 9:")
n, e = hit("X7", "pancreas", "mmd_ref")
print(f"X7 pancreas reference-anchored MMD: {e:.2f} of {n} fits expected to meet at least one (quoted: 25.9 of 27)")
assert n == 27 and f"{e:.1f}" == "25.9"
n, e = hit("X7", "sim1", "mmd_ref")
print(f"X7 sim1 reference-anchored MMD: {e:.2f} of {n} fits (quoted: 18 of 18)")
assert n == 18 and abs(e - 18) < 1e-6
n, e = hit("X3", "atac_small", "mmd")
print(f"X3 atac_small MMD: {e:.3f} of {n} fits = {100 * e / n:.1f} % (quoted: about 14 % of 48)")
assert n == 48 and round(100 * e / n) == 14
n, e = hit("X8", "sim2", "mmd")
print(f"X8 sim2 MMD: {e:.3f} of {n} fits = {100 * e / n:.1f} % (quoted: 4.5 % of 24)")
assert n == 24 and f"{100 * e / n:.1f}" == "4.5"
for arm in ("mmd", "sinkhorn"):
    n, e = hit("X1", "sim1", arm)
    print(f"X1 sim1 {arm}: {e:.4f} of {n} fits = {100 * e / n:.2f} % (quoted: 0.6 % of 60 for each of MMD and Sinkhorn)")
    assert n == 60 and f"{100 * e / n:.1f}" == "0.6"
print("inputs ledger SHA-256:", write_inputs("no_pair_expectations", [MANIFEST, *prep_files]))
