"""Counts of the minibatches behind the Sinkhorn solver validation (PREREG section 9, Sinkhorn entry; DOC-R06, DOC-R04).

Reads experiments/sinkhorn_fix_20261013/results/validation_12mb.csv (committed): one row per validation minibatch, with the
tagged (pre-fix) and converged gradients compared. Reproduces the counts of the minibatches, where they came from, and the
median gradient-norm ratios and the cosine quoted in the same entry. Reads no score.

Origin: the documents re-check of 2026-10-14 (its script 02, blocks N06 and N07).
Usage: python experiments/doc_numbers_20261015/validation_minibatch_counts.py (env scvi-api; seconds).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, write_inputs  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV = REPO / "experiments/sinkhorn_fix_20261013/results/validation_12mb.csv"
v = pd.read_csv(CSV)
fin, early, train = (v[v.kind == k] for k in ("final", "early", "trainmode"))
fits = set(fin.tag)
per_fit = fin.groupby("tag").size()
print(f"minibatches in total: {len(v):,} = {len(fin):,} final-epoch + {len(early)} early + {len(train)} training-mode")
print(f"final-epoch minibatches: {len(fin):,} from {len(fits)} pre-fix fits, {[int(x) for x in sorted(per_fit.unique())]} from each fit; "
      f"fits by stage {fin.groupby('stage').tag.nunique().to_dict()}")
print(f"training-mode minibatches: {len(train)} from {train.tag.nunique()} fits, {len(set(train.tag) & fits)} of them among the {len(fits)} fits")
print(f"early minibatches: {len(early)} from {early.tag.nunique()} replay tags, {len(set(early.tag) & fits)} of them among the {len(fits)} fits; "
      f"by stage {early.groupby('stage').size().to_dict()}")
assert (len(v), len(fin), len(early), len(train)) == (3076, 2808, 192, 76)
assert (len(fits), list(per_fit.unique())) == (234, [12])
assert (train.tag.nunique(), len(set(train.tag) & fits)) == (19, 19)
assert (early.tag.nunique(), len(set(early.tag) & fits)) == (160, 0)
print("\nmedian tagged / converged gradient-norm ratio of the final-epoch minibatches, by stage:")
for stage, want in (("A1", 2.00), ("A2A3", 2.27), ("A1_ext_r1_round1", 14.4)):
    g = fin[fin.stage == stage].tag_norm_ratio
    m = float(np.median(g[np.isfinite(g)]))
    print(f"  {stage}: {m:.3f} (quoted {want})")
    assert abs(m - want) < 0.05
g = fin[fin.stage == "A1_ext_r1_round1"]
ok = np.isfinite(g.tag_cosine)
cos = float(np.median(g.tag_cosine[ok]))
print(f"R1 extension minibatches with a finite tagged gradient: {int(ok.sum())} of {len(g)}; median cosine {cos:.3f} (quoted 204 of 216, 0.28)")
assert (int(ok.sum()), len(g)) == (204, 216) and abs(cos - 0.28) < 0.005
print("inputs ledger SHA-256:", write_inputs("validation_minibatch_counts", [CSV]))
