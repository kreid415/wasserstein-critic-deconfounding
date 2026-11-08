import json

from res5_local_global import DATA, numbers


def test_committed_numbers_match_paper_numbers():  # test-reason: the Res-5 sentence and its ledger rows read this file
    got = numbers(json.load(open(DATA / "paper_numbers.json")))
    assert got == json.loads((DATA / "res5_local_global.json").read_text())
    assert got["pair_median_min"] < 0 < got["pair_median_max"] and got["within_pair_spread_max"] < got["pair_median_max"] - got["pair_median_min"]
