import json

import numpy as np
import pandas as pd
import pytest

from pareto_frontier import DATA, hypervolume, numbers, tables


def test_committed_tables_and_numbers_match_the_inputs():  # test-reason: the new figure, Table S8 column and Res-3b numbers all read these files
    per, med = tables(pd.read_csv(DATA / "curves.csv"), json.load(open(DATA / "paper_numbers.json")), pd.read_csv(DATA / "secondary.csv"))
    pd.testing.assert_frame_equal(per, pd.read_csv(DATA / "pareto_frontier.csv"), atol=1e-6, check_exact=False)
    assert numbers(per, med) == json.loads((DATA / "pareto_frontier.json").read_text())
    assert (per.hypervolume_to_common <= per.hypervolume_all + 1e-12).all()


@pytest.mark.parametrize("x, y, cap, want", [
    ([0.5, 1.0], [0.0, -0.05], np.inf, 0.5 * 0.10 + 1.0 * 0.05 - 0.5 * 0.05),  # union of two rectangles
    ([2.0], [0.0], 1.0, 0.10),        # the cap: a point beyond b_common counts at it
    ([2.0], [0.0], np.inf, 0.20),
    ([0.5], [-0.2], np.inf, 0.0),     # below the reference point adds nothing
])
def test_hypervolume_known_answers(x, y, cap, want):  # test-reason: known answers for the capped and uncapped area
    assert hypervolume(x, y, cap) == pytest.approx(want)
