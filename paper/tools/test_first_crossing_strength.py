import numpy as np
import pandas as pd
import pytest

from first_crossing_strength import DATA, choose, first_crossing_table


@pytest.mark.parametrize("lams, bs, b0, bstar, want", [
    ([1, 3, 10, 30], [0.40, 0.62, 0.45, 0.70], 0.3, 0.5, (1, 3, 1)),    # crosses twice: the later point nearer b* is ignored; lower side is nearer
    ([1, 3], [0.40, 0.52], 0.3, 0.5, (1, 3, 3)),                          # upper side is nearer
    ([1, 3], [0.25, 0.75], 0.0, 0.5, (1, 3, 1)),                          # exact tie: the smaller strength
    ([1, 3], [0.55, 0.90], 0.49, 0.5, (0.0, 1, 1)),                       # first point already reaches b*: plain scVI is nearer but not chosen
    ([1, 3], [0.20, 0.40], 0.1, 0.5, None),                               # never reaches b*
])
def test_choose(lams, bs, b0, bstar, want):  # test-reason: real curves may not contain a repeat crossing, a tie, an early first point or no crossing; these cases pin the rule's branches
    assert choose(lams, bs, b0, bstar) == want


def test_committed_table_is_the_rule_on_the_curves():  # test-reason: the first-crossing strengths feed label transfer and the biology check; this also runs the table builder and its no-crossing branch
    pairs = pd.concat([pd.read_csv(DATA / "ranking_pairs.csv"), pd.read_csv(DATA / "low_signal_pairs.csv")], ignore_index=True)
    curves = pd.read_csv(DATA / "curves.csv")
    got = pd.read_csv(DATA / "first_crossing_strength.csv")
    pd.testing.assert_frame_equal(got, first_crossing_table(curves, pairs))
    assert len(got) == 96 and got.lam.notna().all() and got.low_signal.sum() == 18
    assert (got.batch_score.sub(got.bstar).abs() - got.distance).abs().max() < 1e-12
    assert all(((curves.task == r.task) & (curves.cond == r.cond) & (curves.arm == r.arm) & (curves.lam == r.lam)).sum() == 1 for r in got.itertuples())
    never = first_crossing_table(curves.assign(B=0.0), pairs.head(1))      # no point can reach b*
    assert never.lam.isna().all()
