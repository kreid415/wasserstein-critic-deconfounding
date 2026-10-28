"""End to end on the committed synthetic X1/X13 table (scripts/dryrun/make_synthetic_x1.py): generator, analysis driver, tables and figure.

Every expected value follows from the planted case (SCENARIO in the generator), not from the code under test: a tie is two penalties
given identical scores; a penalty that cannot reach b* is given a step to its first extension strength; and so on. Synthetic inputs
only (SI-78: no main-benchmark score is read).
"""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.environ.get("ACCEPT_WORKDIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "dryrun")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import build_pilot_inputs  # noqa: E402,F401  # imported so its module-level code is run; main() needs the pilot data
import figures_x1  # noqa: E402
import make_synthetic_x1 as G  # noqa: E402
import run_analysis  # noqa: E402
import run_mock_figures  # noqa: E402,F401  # as above: main() needs the paper branch


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    out = tmp_path_factory.mktemp("dryrun")
    inputs, record = str(out / "inputs"), str(out / "analysis.json")
    assert G.main(["--out", inputs]) == 0
    assert run_analysis.main(["--inputs", inputs, "--out-json", record]) == 0
    assert figures_x1.main(["--record", record, "--out", str(out / "tables")]) == 0
    return json.load(open(record))["primary"], out


def test_planted_cases(dry):  # test-reason: the planted ties, bottom places, low-signal pair, failures and double crossing reach the record and the tables
    prim, out = dry
    cells = prim["cells"]
    pen = lambda key, arm: cells[key]["penalties"][arm]  # noqa: E731
    assert pen("pancreas|1", "pooled")["rank"] == pen("pancreas|1", "reference")["rank"] == 2.5       # identical scores: tied 2nd and 3rd
    assert pen("sim2|0", "barycenter")["rank"] == pen("sim2|0", "pooled")["rank"]
    for key, arms, rank in (("atac_large|1", ("mmd", "sinkhorn"), 5.5), ("lung|0", ("sinkhorn",), 6.0)):    # steps that start beyond the grid
        for arm in arms:
            assert (pen(key, arm)["status"], pen(key, arm)["bio"], pen(key, arm)["rank"]) == ("unreached_low", None, rank)
    assert cells["immune|1"]["low_signal"] and not any("rank" in v for v in cells["immune|1"]["penalties"].values())
    assert prim["ranking"]["1"]["low_signal"] == ["immune"] and prim["ranking"]["0"]["low_signal"] == []
    assert [d["n_tasks"] for d in prim["ranking"]["1"]["mean_rank"].values()] == [5] * 6      # six real tasks less immune
    for key, c in cells.items():
        assert c["low_signal"] or sum(v["rank"] for v in c["penalties"].values()) == 21, key
    assert pen("lung|1", "pooled")["crossings"] == 2                                          # rises through b*, falls back through it
    assert cells["lung|0"]["collapse"]["pooled"]["failed"] == dict(diverged=5, nonfinite_latent=0)
    assert cells["immune_hum_mou|0"]["collapse"]["mmd"]["failed"] == dict(diverged=0, nonfinite_latent=5)
    pairs = pd.read_csv(out / "tables" / "ranking_pairs.csv")
    assert len(pairs) == 96 and pairs["rank"].isna().sum() == 6 and (out / "tables" / "fig_ranking.png").stat().st_size > 0
