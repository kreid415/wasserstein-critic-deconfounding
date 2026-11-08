import json

import pandas as pd

from res5_label_transfer import DATA, numbers


def test_committed_numbers_match_the_tables():  # test-reason: the Res-5 label-transfer sentences and their ledger rows read this file
    got = numbers(pd.read_csv(DATA / "label_transfer.csv"), pd.read_csv(DATA / "ranking_pairs.csv"))
    assert got == json.loads((DATA / "res5_label_transfer.json").read_text())
    assert got["cond0_disc_last"] <= got["n_pairs"]["0"] and -1 <= got["tau_min"] <= got["tau_median"] <= got["tau_max"] <= 1
