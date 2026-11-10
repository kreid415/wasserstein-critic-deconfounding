"""Label-transfer table at the first-crossing strengths (Pass 23, rule fixed by the author on 2026-11-09; see first_crossing_strength.py).
Rebuilds paper/figure_data/label_transfer.csv with the same columns and row order as the table it replaces, kept unchanged as
label_transfer_nearest_any_crossing.csv (the strength whose batch score is nearest b* at any crossing).
Penalty rows: the label-transfer curve (label_transfer_curve.csv) at the strength in first_crossing_strength.csv. The curve file has no
n_runs and not_evaluable columns; they are read from the same curve point (same task, decoder, penalty and strength) in analysis_sweep.json of the
sweep analysis, primary -> label_transfer -> decoders -> <cond> -> tasks -> <task> -> <arm> -> curve, and that point's macro F1 and batch score
must equal the curve file's (1e-12). Plain scVI (arm none) and scvi_adv rows are copied unchanged from the old table.
Run from the repository root:  python paper/tools/label_transfer_first_crossing.py [--sweep DIR]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper/figure_data"
SWEEP = "/home/kendall/experiment_data/wasserstein-critic-deconfounding/sweep_analysis_final_20261105"
KEY = ["cond", "task", "arm"]


def read(name):
    return pd.read_csv(DATA / name, float_precision="round_trip")   # the default parser changes the last digit of some floats, which would alter the rows kept unchanged


def sweep_curve_points(rec):
    return pd.DataFrame([dict(cond=int(c), task=t, arm=a, lam=y["lam"], n_runs=y["n_runs"], not_evaluable=y["not_evaluable"],
                              json_f1=y["mean"], json_batch=y["batch_score"])
                         for c, d in rec["primary"]["label_transfer"]["decoders"].items()
                         for t, arms in d["tasks"].items() for a, p in arms.items() for y in p.get("curve", [])])


def rebuild(old, fc, curve, points):
    """old: the nearest-any-crossing table; fc: first_crossing_strength.csv; curve: label_transfer_curve.csv; points: sweep_curve_points()."""
    assert fc.lam.notna().all(), "a curve with no first-crossing strength"
    pen = fc[KEY + ["lam"]].merge(curve, on=KEY + ["lam"], how="left", validate="1:1", indicator=True)
    assert (pen._merge == "both").all(), "a first-crossing strength has no label-transfer curve point"
    pen = pen.merge(points, on=KEY + ["lam"], how="left", validate="1:1")
    assert pen.n_runs.notna().all(), "a first-crossing strength has no point in analysis_sweep.json"
    assert np.allclose(pen.f1_mean, pen.json_f1, rtol=0, atol=1e-12) and np.allclose(pen.batch_score, pen.json_batch, rtol=0, atol=1e-12)
    pen = pen.set_index(KEY)[[c for c in old.columns if c not in KEY]]
    out = old.set_index(KEY)
    assert set(pen.index) == set(old[~old.arm.isin(["none", "scvi_adv"])].set_index(KEY).index), "penalty rows differ from the old table's"
    out.loc[pen.index, pen.columns] = pen
    return out.reset_index()[list(old.columns)]


if __name__ == "__main__":  # pragma: no cover  # command-line entry; rebuild and sweep_curve_points are tested
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=SWEEP)
    a = ap.parse_args()
    rec = json.load(open(Path(a.sweep) / "analysis_sweep.json"))
    new = rebuild(read("label_transfer_nearest_any_crossing.csv"), read("first_crossing_strength.csv"), read("label_transfer_curve.csv"),
                  sweep_curve_points(rec))
    new.to_csv(DATA / "label_transfer.csv", index=False)
    print("label_transfer.csv", len(new), "rows")
