import json

import pandas as pd

from res2_seed_noise import DATA, counts


def test_counts_on_a_small_table():
    rows = [  # one ranked pair (task t, conditioned decoder) with three penalties, one low-signal pair, one simulation
        ("t", 1, 0.70, 0.010, 1, 1, 1), ("t", 1, 0.695, 0.002, 2, 1, 2), ("t", 1, 0.60, 0.010, 3, 3, 3),
        ("u", 1, 0.50, None, None, None, None), ("sim1", 0, 0.9, 0.1, 1, 1, 1)]
    rp = pd.DataFrame(rows, columns=["task", "cond", "bio_at_bstar", "loso_se", "rank", "rank_lo", "rank_hi"])
    rp["n_loso"] = [3, 3, 3, None, 3]
    got = counts(rp)
    assert got["batch_conditioned"] == {"neighbour_gaps": 2, "gaps_below_larger_se": 1, "first_places": 1,
                                        "first_places_stable": 1}
    assert got["unconditioned"]["neighbour_gaps"] == 0 and got["all"]["neighbour_gaps"] == 2


def test_committed_counts_match_the_table():
    assert counts(pd.read_csv(DATA / "ranking_pairs.csv")) == json.loads((DATA / "res2_seed_noise.json").read_text())
