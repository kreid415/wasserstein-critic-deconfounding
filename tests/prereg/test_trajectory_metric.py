"""SI-28: trajectory conservation is one of scIB's bio metrics (scib-reproducibility plotSingleTaskRNA.R
group_bio). It enters the bio score only where scIB computes it (obs dpt_pseudotime: immune, immune_hum_mou);
everywhere else it is NaN and skipped, so those tasks' bio scores do not change."""
import os

os.environ.setdefault("KMP_AFFINITY", "disabled")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("scanpy")
pytest.importorskip("scib")
import score_scib_native as SN  # noqa: E402


def _table(dataset, trajectory):
    rows = []
    for i in range(4):
        r = {"dataset": dataset}
        r.update({m: 0.1 * (i + 1) + 0.01 * k for k, m in enumerate(SN.BATCH_METRICS)})
        r.update({m: 0.2 + 0.1 * ((i * (j + 1)) % 4) + 0.01 * j for j, m in enumerate(SN.BIO_METRICS)})
        rows.append(r)
    t = pd.DataFrame(rows)
    t["trajectory"] = trajectory
    return t


def test_trajectory_is_a_bio_metric():
    assert SN.BIO_METRICS[-1] == "trajectory" and "trajectory" not in SN.BATCH_METRICS


def test_scib_overall_unchanged_where_trajectory_is_not_computed():
    t = _table("pancreas", np.nan)
    with_col, without = SN.scib_overall(t), SN.scib_overall(t.drop(columns="trajectory"))
    pd.testing.assert_series_equal(with_col.bio_score, without.bio_score)
    pd.testing.assert_series_equal(with_col.overall, without.overall)


def test_scib_overall_averages_trajectory_on_immune():
    t = _table("immune", [0.9, 0.1, 0.5, 0.3])
    o = SN.scib_overall(t)
    scaled = pd.concat([(t[m] - t[m].min()) / (t[m].max() - t[m].min()) for m in SN.BIO_METRICS], axis=1)
    np.testing.assert_allclose(o.bio_score.values, scaled.mean(axis=1).values)
    assert not np.allclose(o.bio_score.values, SN.scib_overall(t.drop(columns="trajectory")).bio_score.values)
