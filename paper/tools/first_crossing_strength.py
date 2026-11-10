"""First-crossing strength of every penalty curve (Pass 23, rule fixed by the author on 2026-11-09). The paper ranks penalties by the biology
score read where each penalty's batch-score curve first reaches the matched batch removal b*; label transfer and the direct biology check
use the strength chosen here, so they are read at the same place.

For each task, decoder and penalty with a b* (the 78 rows of ranking_pairs.csv and the 18 of low_signal_pairs.csv): the curve is plain scVI
(strength 0, batch score b0) followed by the penalty's fitted strengths in increasing order with their seed-mean batch score B (curves.csv).
Find the first point whose B >= b*. Its two bracketing points are the point before it and that point. If the point before is plain scVI,
the upper point is chosen; otherwise the point with the smaller |B - b*| (a tie goes to the smaller strength). A curve that never reaches
b* gets a row with no strength and a message on stderr. Later crossings of b* are ignored.
Writes paper/figure_data/first_crossing_strength.csv (columns task, cond, arm, lam, batch_score, bstar, distance, lower_lam, upper_lam,
low_signal; lower_lam is 0 when the point before is plain scVI). Run from the repository root:  python paper/tools/first_crossing_strength.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figure_data"
TIE_TOL = 1e-12   # |B - b*| values closer than this are a tie


def choose(lams, bs, b0, bstar):
    """lams, bs: a penalty's fitted strengths (all > 0, increasing) and their seed-mean batch scores. Returns (lower_lam, upper_lam, lam)
    with lam the chosen strength, or None when no point reaches bstar."""
    assert b0 < bstar, "plain scVI already reaches b*"
    hit = next((i for i, b in enumerate(bs) if b >= bstar), None)
    if hit is None:
        return None
    if hit == 0:
        return 0.0, lams[0], lams[0]
    lower = bstar - bs[hit - 1] <= bs[hit] - bstar + TIE_TOL
    return lams[hit - 1], lams[hit], lams[hit - 1] if lower else lams[hit]


def first_crossing_table(curves, pairs):
    """curves: curves.csv; pairs: ranking_pairs.csv and low_signal_pairs.csv stacked."""
    assert (curves.n_fail == 0).all(), "a curve point with failed fits"   # the label-transfer rule used failure-free strengths only
    rows = []
    for r in pairs.itertuples():
        c = curves[(curves.task == r.task) & (curves.cond == r.cond) & (curves.arm == r.arm) & (curves.lam > 0)].sort_values("lam")
        got = choose(c.lam.tolist(), c.B.tolist(), r.b0, r.bstar)
        if got is None:
            print(f"no point of {r.task} cond {r.cond} {r.arm} reaches b*; no strength chosen", file=sys.stderr)
            got = (np.nan,) * 3
        lo, up, lam = got
        b = np.nan if np.isnan(lam) else float(c.B[c.lam == lam].iloc[0])
        rows.append(dict(task=r.task, cond=r.cond, arm=r.arm, lam=lam, batch_score=b, bstar=r.bstar, distance=abs(b - r.bstar),
                         lower_lam=lo, upper_lam=up, low_signal=r.low_signal))
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover  # command-line entry; choose and first_crossing_table are tested
    read = lambda name: pd.read_csv(DATA / name, float_precision="round_trip")   # the default parser changes the last digit of some floats
    pairs = pd.concat([read("ranking_pairs.csv"), read("low_signal_pairs.csv")], ignore_index=True)
    out = first_crossing_table(read("curves.csv"), pairs)
    out.to_csv(DATA / "first_crossing_strength.csv", index=False)
    print("first_crossing_strength.csv", len(out), "rows;", int(out.lam.isna().sum()), "without a strength")
