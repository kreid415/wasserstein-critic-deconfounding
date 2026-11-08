"""Res-5 local-versus-global numbers (Pass 23; the figure moved to Supplementary Figure S8 by user choice, Res-5 keeps one
sentence): at b*, local minus global batch removal per penalty and ranked real task and decoder pair; how much the six
penalties differ within a pair, against how much the pair medians differ. Reads paper/figure_data/paper_numbers.json
(local_minus_global.<arm>.<task>.cond<c>, pair.<task>.cond<c>.low_signal). Writes paper/figure_data/res5_local_global.json.
Run from the repository root:  python paper/tools/res5_local_global.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figure_data"
REAL = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]
ARMS = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]


def numbers(pn):
    v = lambda k: pn[k]["value"]
    pairs = [(t, c) for c in (0, 1) for t in REAL if not v(f"pair.{t}.cond{c}.low_signal")]
    lg = {p: np.array([v(f"local_minus_global.{a}.{p[0]}.cond{p[1]}") for a in ARMS]) for p in pairs}
    positive = sorted({t for (t, c), x in lg.items() if (x > 0).any()})
    assert all((x > 0).all() or (x < 0).all() for x in lg.values())  # every penalty has the same sign within a pair
    spread = {p: float(x.max() - x.min()) for p, x in lg.items()}
    med = {p: float(np.median(x)) for p, x in lg.items()}
    r = lambda x: round(x, 3)
    return {"n_ranked_pairs": len(pairs), "positive_tasks": positive,
            "n_positive_pairs": int(sum((x > 0).all() for x in lg.values())),
            "within_pair_spread_max": r(max(spread.values())), "within_pair_spread_min": r(min(spread.values())),
            "pair_median_min": r(min(med.values())), "pair_median_max": r(max(med.values()))}


if __name__ == "__main__":
    res = numbers(json.load(open(DATA / "paper_numbers.json")))
    (DATA / "res5_local_global.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
