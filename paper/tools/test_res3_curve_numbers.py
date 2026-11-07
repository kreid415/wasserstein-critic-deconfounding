import json

import pandas as pd

from res3_curve_numbers import DATA, numbers


def test_committed_numbers_match_the_tables():
    got = numbers(pd.read_csv(DATA / "curves.csv"), pd.read_csv(DATA / "ranking_pairs.csv"))
    assert got == json.loads((DATA / "res3_curve_numbers.json").read_text())
    assert got["n_pairs_barycenter_lowest"] <= got["n_real_pairs"] and got["n_ranked_pairs"] <= got["n_real_pairs"]
