"""Res-2 seed-noise counts (Pass 23): how many neighbouring biology readings at b* differ by less than the larger of their two
leave-one-seed-out standard errors, and how many first places stay first in all three leave-one-seed-out repeats.
Reads paper/figure_data/ranking_pairs.csv (copy of the sweep analysis table, SHA-256 in paper/figure_data/MANIFEST.sha256);
real tasks only, low-signal pairs excluded (they have no rank). Writes paper/figure_data/res2_seed_noise.json.
Run from the repository root:  python paper/tools/res2_seed_noise.py"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figure_data"
SIMULATIONS = {"sim1", "sim2"}
DECODER = {1: "batch_conditioned", 0: "unconditioned"}


def counts(rp):
    ranked = rp[rp["rank"].notna() & ~rp.task.isin(SIMULATIONS)]
    assert ranked.loso_se.notna().all() and (ranked.n_loso == 3).all()
    out = {}
    for c, name in DECODER.items():
        within, gaps = 0, 0
        for _, g in ranked[ranked.cond == c].groupby("task"):
            g = g.sort_values("bio_at_bstar", ascending=False)
            bio, se = g.bio_at_bstar.to_numpy(), g.loso_se.to_numpy()
            for i in range(len(g) - 1):  # neighbours in the order of the reading
                gaps += 1
                within += bool(bio[i] - bio[i + 1] < max(se[i], se[i + 1]))
        first = ranked[(ranked.cond == c) & (ranked["rank"] == 1)]
        out[name] = {"neighbour_gaps": gaps, "gaps_below_larger_se": within, "first_places": len(first),
                     "first_places_stable": int(((first.rank_lo == 1) & (first.rank_hi == 1)).sum())}
    out["all"] = {k: sum(out[n][k] for n in DECODER.values()) for k in out["batch_conditioned"]}
    return out


if __name__ == "__main__":
    res = counts(pd.read_csv(DATA / "ranking_pairs.csv"))
    (DATA / "res2_seed_noise.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
