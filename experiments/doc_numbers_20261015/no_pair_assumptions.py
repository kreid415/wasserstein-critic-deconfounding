"""Assumptions behind no_pair_expectations.py, read from the installed scvi-tools source, the manifest and the project code, plus a
Monte Carlo check of one case (X1 sim1) (PREREG section 9, guards entry; DOC-R04). Reads no score.

Origin: the documents re-check of 2026-10-14 (its script 03b). Usage: python experiments/doc_numbers_20261015/no_pair_assumptions.py
(env scvi-api, scvi-tools 1.4.2; about 1 minute).
"""
import inspect
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, data_path, write_inputs  # noqa: E402

import anndata  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scvi  # noqa: E402
import scvi.dataloaders._data_splitting as ds  # noqa: E402
from scipy.special import gammaln  # noqa: E402

MANIFEST = REPO / "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"
src = inspect.getsource(ds.validate_data_split)
sp = inspect.getsource(ds.DataSplitter)
print("scvi-tools", scvi.__version__)
checks = {
    "n_train = ceil(train_size * n_samples) in validate_data_split": "n_train = ceil(train_size * n_samples)" in src,
    "DataSplitter default drop_last is False": 'kwargs.pop("drop_last", False)' in sp,
    "train_dataloader uses shuffle=True and drop_last=self.drop_last": "shuffle=True" in sp and "drop_last=self.drop_last" in sp,
    "a last minibatch of fewer than 3 cells with an explicit train_size is only warned about": "Last batch will have a small size" in src,
}
M = pd.read_csv(MANIFEST, sep="\t", comment="#", dtype=str)
cf = M[M.arm.isin(["mmd", "mmd_ref", "sinkhorn"])]
plan = (REPO / "scripts/scvi_adversarial_plan.py").read_text()
fpc = (REPO / "scripts/fit_paper_config.py").read_text()
checks["the project passes drop_last nowhere"] = not re.search(r"drop_last\s*=", plan + fpc)
checks["all critic-free manifest rows use batch size 128 and train_size 0.9"] = (
    sorted(cf.batch_size.unique()) == ["128"] and sorted(cf.train_size.unique()) == ["0.9"])
checks["the stratified sampler is used only for the arms reference and pooled"] = (
    re.search(r'"sampler": (\{[^}]*\})', fpc).group(1) == '{"reference", "pooled"}')
for k, v in checks.items():
    print(f"{v}: {k}")
assert all(checks.values()), [k for k, v in checks.items() if not v]
print("critic-free rows:", len(cf))

f = data_path("prepped_scib", "sim1__scib.h5ad")
a = anndata.read_h5ad(f, backed="r")
cnt = a.obs["batch"].astype(str).value_counts()
a.file.close()
N = int(cnt.sum())
n_train = math.ceil(0.9 * N)
steps = math.ceil(n_train / 128)
m_last = n_train - 128 * (steps - 1)
c = np.round(cnt.values * n_train / N).astype(int)
c[0] += n_train - c.sum()
lch = lambda n, k: gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)  # noqa: E731
p = float(np.exp(lch(c.astype(float), m_last) - lch(n_train, m_last)).sum())
rng = np.random.default_rng(1)
hits, size = 0, 20_000_000
for _ in range(size // 2_000_000):
    hits += int((rng.multivariate_hypergeometric(c, m_last, size=2_000_000).max(axis=1) == m_last).sum())
print(f"sim1: N {N}, n_train {n_train}, steps {steps}, last minibatch {m_last} cells; per-epoch probability of a minibatch "
      f"with no pair: exact {p:.4e}, Monte Carlo {hits / size:.4e} ({hits} of {size})")
print(f"expected fits with at least one such minibatch out of 60 at 400 epochs: {60 * (1 - math.exp(-400 * p)):.4f}")
assert (N, n_train, steps, m_last) == (12097, 10888, 86, 8) and abs(hits / size - p) < 5e-7
print("inputs ledger SHA-256:", write_inputs("no_pair_assumptions", [MANIFEST, f, REPO / "scripts/scvi_adversarial_plan.py",
                                                                     REPO / "scripts/fit_paper_config.py"]))
