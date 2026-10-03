"""Unit tests for the stage model in scripts/gate_task_split.py (fluid local scoring queue, queue waits, tails)."""
import math
import os
import sys

import pytest

pytest.importorskip("pandas")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import gate_task_split as T  # noqa: E402


def test_fluid_matches_closed_form_for_local_only_stage():
    # local latents arrive over [0, d]; 4 scoring CPUs while the GPU fits, 12 afterwards
    for W, d in [(10.0, 10.0), (100.0, 10.0), (400.0, 50.0)]:
        fin, _ = T.fluid([(0.0, d, W)], d, 4, 12)
        assert math.isclose(fin, d + max(0.0, W - 4 * d) / 12, rel_tol=1e-12)


def test_fluid_late_slow_remote_stream_is_arrival_limited():
    fin, _ = T.fluid([(0.0, 10.0, 20.0), (15.0, 40.0, 50.0)], 10.0, 4, 12)
    assert math.isclose(fin, 40.0)


def test_fluid_capacity_limited_overlap_and_snapshot():
    fin, (arrived, backlog) = T.fluid([(0.0, 10.0, 60.0), (0.0, 10.0, 60.0)], 10.0, 4, 12)
    assert math.isclose(arrived, 120.0) and math.isclose(backlog, 80.0)
    assert math.isclose(fin, 10.0 + 80.0 / 12)


def test_fluid_rejects_work_with_empty_window():
    with pytest.raises(ValueError):
        T.fluid([(5.0, 5.0, 1.0)], 0.0, 4, 12)


def test_stage_queue_wait_and_scoring_tail():
    P = dict(f_l=4.0, f_j=4.0, c_fit=4, c_idle=12)
    s = T.stage(40.0, 1.0, 0.5, 40.0, 1.0, 0.5, P, 1, 24.0)
    assert math.isclose(s["dL"], 10.0) and math.isclose(s["dJ"], 10.0)
    assert math.isclose(s["DJ"], 34.0)          # one 3-day job, one 24 h wait before it
    assert math.isclose(s["end"], 34.5)         # last JHPCE latent arrives at 34 h and takes 0.5 h to score
    assert s["jobs"] == 1


def test_stage_long_jhpce_work_pays_one_wait_per_job():
    P = dict(f_l=4.0, f_j=1.0, c_fit=4, c_idle=12)
    s = T.stage(0.0, 0.0, 0.0, 150.0, 1.0, 0.1, P, 1, 10.0)
    assert s["jobs"] == 3 and math.isclose(s["DJ"], 150.0 + 30.0)
