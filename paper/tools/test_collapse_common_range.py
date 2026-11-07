import json

import pandas as pd

from collapse_common_range import DATA, fits, numbers, table


def test_committed_table_and_numbers_match_the_inputs():  # test-reason: Figure 5a, Table S8 and the Res-4 text read these files; the rebuild must reproduce secondary.csv
    f = fits(pd.read_csv(DATA / "metric_scores.csv"), pd.read_csv(DATA / "curves.csv"), json.load(open(DATA / "paper_numbers.json")))
    t = table(f, pd.read_csv(DATA / "secondary.csv"))
    pd.testing.assert_frame_equal(t, pd.read_csv(DATA / "collapse_common_range.csv"), atol=1e-6, check_exact=False)
    assert numbers(t, pd.read_csv(DATA / "ranking_pairs.csv")) == json.loads((DATA / "collapse_common_range.json").read_text())
    assert (t.k_common <= t.k_all).all() and (t.n_common <= t.n_all).all() and (t.k_common <= t.n_common).all()
