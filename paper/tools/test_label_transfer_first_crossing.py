import pandas as pd

from label_transfer_first_crossing import DATA, KEY, read, rebuild, sweep_curve_points


def test_rebuilt_table_is_the_curve_at_the_first_crossing():  # test-reason: label_transfer.csv feeds Res-5, Supplementary Table S11 and Figure S3; n_runs is checked against the score rows, not against the file it came from
    old, fc, curve, new = (read(f) for f in ("label_transfer_nearest_any_crossing.csv", "first_crossing_strength.csv",
                                             "label_transfer_curve.csv", "label_transfer.csv"))
    n = pd.read_csv(DATA / "metric_scores.csv").groupby(["task", "cond", "arm", "lam"]).size().rename("n_runs").reset_index()
    points = curve.merge(n, on=["task", "cond", "arm", "lam"]).assign(not_evaluable=False, json_f1=lambda d: d.f1_mean, json_batch=lambda d: d.batch_score)
    assert len(points) == len(curve)
    points = points[KEY + ["lam", "n_runs", "not_evaluable", "json_f1", "json_batch"]]   # the columns sweep_curve_points gives
    pd.testing.assert_frame_equal(rebuild(old, fc, curve, points), new)
    assert new[KEY].equals(old[KEY])                                      # same rows in the same order
    plain = old.arm.isin(["none", "scvi_adv"])
    text = {f: pd.read_csv(DATA / f, dtype=str) for f in ("label_transfer_nearest_any_crossing.csv", "label_transfer.csv")}
    assert text["label_transfer_nearest_any_crossing.csv"][plain].equals(text["label_transfer.csv"][plain])   # unchanged to the last digit
    got = new[~plain].merge(fc, on=KEY, suffixes=("", "_fc"))
    assert len(got) == 96 and (got.lam == got.lam_fc).all() and (got.distance - got.distance_fc).abs().max() < 1e-12


def test_sweep_curve_points_reads_the_curve_of_each_penalty():
    point = dict(lam=0.3, n_runs=3, not_evaluable=False, mean=0.8, batch_score=0.5)
    rec = {"primary": {"label_transfer": {"decoders": {"1": {"tasks": {"lung": {"mmd": {"curve": [point]}, "none": {}}}}}}}}
    got = sweep_curve_points(rec).iloc[0]
    assert (got.cond, got.task, got.arm, got.lam, got.n_runs, got.json_f1) == (1, "lung", "mmd", 0.3, 3, 0.8)
