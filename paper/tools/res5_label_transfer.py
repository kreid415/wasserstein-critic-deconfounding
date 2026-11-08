"""Res-5 label-transfer numbers (Pass 23, user choice: per-task comparisons with plain scVI instead of cross-task mean F1, and
label transfer merged with the direct biology check). For each ranked real task and decoder pair: each penalty's macro F1 at
its label-transfer strength (nearest b*) minus plain scVI's with the same decoder; the discriminator's place among the six
penalties; Kendall tau_b within the pair between the penalties' F1 and their biology at b*; the batch-score distance of the
label-transfer embedding from b*. Reads paper/figure_data/label_transfer.csv and ranking_pairs.csv. Writes
paper/figure_data/res5_label_transfer.json. Run from the repository root:  python paper/tools/res5_label_transfer.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figure_data"
REAL = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]
ARMS = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]


def numbers(lt, rp):
    rp = rp[rp.task.isin(REAL) & rp["rank"].notna()]
    pairs = sorted(set(zip(rp.task, rp.cond)))
    assert lt.computed.all() and not lt.not_evaluable.any()
    plain = lt[lt.arm == "none"].set_index(["task", "cond"]).f1_mean
    pen = lt[lt.arm.isin(ARMS) & np.array([(t, c) in pairs for t, c in zip(lt.task, lt.cond)])].copy()
    assert len(pen) == 6 * len(pairs)
    pen["diff"] = pen.f1_mean.to_numpy() - plain.loc[list(zip(pen.task, pen.cond))].to_numpy()
    pen["place"] = pen.groupby(["task", "cond"]).f1_mean.rank(ascending=False)
    med = pen.groupby(["cond", "arm"])["diff"].median()
    d = {c: pen[(pen.arm == "discriminator") & (pen.cond == c)] for c in (0, 1)}
    others = med.drop("discriminator", level="arm")
    bio = rp.set_index(["task", "cond", "arm"]).bio_at_bstar
    taus = [kendalltau(g.f1_mean, bio.loc[list(zip(g.task, g.cond, g.arm))].to_numpy()).statistic
            for _, g in pen.groupby(["task", "cond"])]
    r = lambda x: round(float(x), 3)
    return {"n_ranked_pairs": len(pairs), "n_pairs": {str(c): int(len(d[c])) for c in (0, 1)},
            "cond0_disc_below_plain": int((d[0]["diff"] < 0).sum()), "cond0_disc_last": int((d[0].place == 6).sum()),
            "cond0_disc_median_diff": r(med[(0, "discriminator")]), "cond1_disc_median_diff": r(med[(1, "discriminator")]),
            "cond1_disc_place_min": int(d[1].place.min()), "cond1_disc_place_max": int(d[1].place.max()),
            "others_median_diff_min": r(others.min()), "others_median_diff_max": r(others.max()),
            "tau_median": r(np.median(taus)), "tau_min": r(min(taus)), "tau_max": r(max(taus)),
            "distance_max": r(pen.distance.abs().max()), "distance_median": r(pen.distance.abs().median())}


if __name__ == "__main__":
    res = numbers(pd.read_csv(DATA / "label_transfer.csv"), pd.read_csv(DATA / "ranking_pairs.csv"))
    (DATA / "res5_label_transfer.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
