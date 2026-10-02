"""Unit tests of scripts/prereg_rules.py on synthetic score tables (docs/PREREG.md). Run in any env with
pandas + scipy:  python -m pytest -q tests/prereg"""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import checks
import synth
from synth import P, BPM


# ---- definitions -----------------------------------------------------------------------------
def test_metric_lists_come_from_the_scorer():
    assert P.BATCH_METRICS == ["PCR_batch", "ASW_label/batch", "iLISI", "graph_conn", "kBET"]
    assert "cell_cycle_conservation" in P.BIO_METRICS and len(P.BIO_METRICS) == 7
    assert "cell_cycle_conservation" not in P.bio_metrics_for("atac_small")
    assert "cell_cycle_conservation" not in P.bio_metrics_for("sim1")
    assert "cell_cycle_conservation" in P.bio_metrics_for("immune")
    with pytest.raises(P.PreregError):
        P.bio_metrics_for("not_a_task")


def test_lambda_sequence_and_canonical_strings():
    assert [P.fmt_lam(v) for v in BPM.LAMBDA_GRID] == [str(v) for v in BPM.LAMBDA_GRID]
    assert P.lam_beyond(0.1, -1, 4) == [0.03, 0.01, 0.003, 0.001]
    assert P.lam_beyond(3000, +1, 2) == [10000.0, 30000.0]
    assert [P.lam_step(v, +1) for v in BPM.LAMBDA_GRID[:-1]] == [float(v) for v in BPM.LAMBDA_GRID[1:]]
    with pytest.raises(P.PreregError):
        P.lam_step(2.0, +1)


def test_families_cover_the_x1_arms():
    assert sorted(P.ADV_ARMS) == sorted(BPM.ARMS)
    assert P.FAMILIES["w1"] == ("reference", "pooled", "barycenter")


# ---- R1 ----------------------------------------------------------------------------------------
def test_r1_block_dose_response():
    jit = (0.001, 0.0, -0.001)
    db = [0, 0, 0.004, 0.05, 0.15, 0.25, 0.295, 0.30, 0.30, 0.30]
    dc = [0, 0, 0, 0, -0.01, -0.03, -0.08, -0.15, -0.25, -0.30]
    b = P.r1_block(*checks._block_points(db, dc, jit))
    assert (b["floor"], b["peak"], b["upper"]) == (2, 6, 6)
    assert not b["low_edge"] and not b["high_edge"] and not b["nonresponsive"]


def test_r1_block_low_edge():
    b = P.r1_block(*checks._block_points([0.05] + [0.1] * 9, [0] * 5 + [-0.2] * 5, (0.001, 0, -0.001)))
    assert b["low_edge"] and b["floor"] is None


def test_r1_block_high_edge_when_batch_still_rising():
    db = [0, 0, 0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
    b = P.r1_block(*checks._block_points(db, [0] * 10, (0.001, 0, -0.001)))
    assert b["upper"] == 9 and b["high_edge"]


def test_r1_block_collapse_before_batch_peak_sets_the_upper_end():
    db = [0, 0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.30]
    dc = [0, 0, 0, 0, -0.05, -0.12, -0.15, -0.2, -0.25, -0.3]
    b = P.r1_block(*checks._block_points(db, dc, (0.001, 0, -0.001)))
    assert b["upper"] == 5 and not b["high_edge"]


def test_r1_block_isolated_failure_is_not_a_persistent_regime():
    db = [0, 0, 0.004, 0.05, 0.15, 0.25, 0.295, 0.30, 0.30, 0.30]
    dc = [0, 0, 0, 0, -0.01, -0.03, -0.08, -0.15, -0.25, -0.30]
    b = P.r1_block(*checks._block_points(db, dc, (0.001, 0, -0.001), fail=(4,)))
    assert b["points"][4]["collapsed"] and not b["points"][4]["no_effect"]
    assert (b["floor"], b["upper"]) == (2, 6)
    b = P.r1_block(*checks._block_points(db, dc, (0.001, 0, -0.001), fail=(5, 6, 7, 8, 9)))
    assert b["upper"] == 5


def test_r1_block_nonresponsive():
    b = P.r1_block(*checks._block_points([0.002] * 10, [0.0] * 10, (0.001, 0, -0.001)))
    assert b["nonresponsive"] and b["high_edge"]


def test_r1_collapse_boundary():
    checks.check_collapse_boundary(P)


def test_r1_noise_threshold():
    checks.check_noise_threshold(P)


def test_r1_needs_replicated_points_for_noise():
    pts, zero = checks._block_points([0] * 10, [0] * 10)
    for p in pts:
        p["B"] = {100: p["B"][100]}
        p["C"] = {100: p["C"][100]}
    with pytest.raises(P.PreregError, match="noise estimate"):
        P.r1_block(pts, zero)


@pytest.mark.parametrize("lo,hi,n,want", [
    (0, 9, 10, [0, 2, 4, 5, 7, 9]),            # spread over 4.5 decades
    (1, 6, 10, [1, 2, 3, 4, 5, 6]),            # exactly 6
    (2, 4, 10, [1, 2, 3, 4, 5, 6]),            # pad above, below, above
    (7, 9, 10, [4, 5, 6, 7, 8, 9]),            # top edge: pad below only
    (0, 1, 10, [0, 1, 2, 3, 4, 5]),            # bottom edge: pad above only
    (0, 13, 14, [0, 3, 5, 8, 10, 13]),         # extended grid
])
def test_grid_indices(lo, hi, n, want):
    got = P.grid_indices(lo, hi, n)
    assert got == want and len(set(got)) == 6


def _a1_out(world=synth.world):
    M = synth.pilot_rows()
    A1 = M[M.experiment == "A1"]
    S, F = synth.scores(A1, world)
    return M, P.outcomes(M, S, F, ["A1"])


def test_r1_family_freezes_inside_the_grid():
    _, out = _a1_out()
    r1 = P.freeze_a1_grid(out)
    assert r1["status"] == "frozen"
    for k, fam in r1["families"].items():
        g = fam["x1_grid"]
        assert len(g) == 6 and g == sorted(g) and set(g) <= set(fam["a1_grid"]), k
        assert not fam["flags"], fam["flags"]


def test_r1_family_extends_at_the_low_edge_and_counts_rounds():
    _, out = _a1_out(lambda r: synth.world(r, center={"sinkhorn": -0.8}))
    r1 = P.freeze_a1_grid(out)
    assert r1["status"] == "extend"
    ext = {k: f["extend"] for k, f in r1["families"].items() if f["extend"]}
    assert ext == {"sinkhorn|0": {"low": [0.03, 0.01]}, "sinkhorn|1": {"low": [0.03, 0.01]}}


def test_r1_cap_accepts_the_edge_with_a_flag():
    grid = [0.001, 0.003, 0.01, 0.03] + checks.GRID          # two low rounds already fitted
    rec = []
    for t in ("atac_small", "immune", "sim1"):
        for s in checks.SEEDS:
            rec.append(dict(experiment="A1", task=t, cond=1, arm="none", lam=0, seed=s, B=0.4, C=0.7))
            for arm in ("discriminator",):
                for i, lam in enumerate(grid):
                    j = (0.002, 0.0, -0.002)[checks.SEEDS.index(s)]
                    rec.append(dict(experiment="A1", task=t, cond=1, arm=arm, lam=lam, seed=s,
                                    B=0.4 + min(0.3, 0.05 + 0.03 * i) + j, C=0.7 - (0.2 if i > 9 else 0) + j))
    fam = P.r1_family(synth.out_frame(rec), "js", 1)
    assert fam["status"] == "frozen" and fam["rounds"] == {"low": 2, "high": 0}
    assert any("low_edge_unresolved" in f for f in fam["flags"]) and fam["x1_grid"][0] == 0.001


def test_r1_refuses_an_incomplete_a1_grid():
    M, out = _a1_out()
    drop = out[(out.arm == "pooled") & (out.task == "immune") & (out.lamf == 3000)].index
    with pytest.raises(P.PreregError, match="different lambda grid"):
        P.freeze_a1_grid(out.drop(drop))


# ---- R4 ----------------------------------------------------------------------------------------
def test_r4_tie_goes_to_the_smaller_lambda():
    checks.check_r4_tie(P)


def test_r4_failed_point_is_not_eligible():
    curves = checks.flat_curves(P)
    curves["pooled"] = [0.40, 0.40, 0.45, 0.50, 0.55, 0.60, 0.60, 0.60, 0.60, 0.60]
    r4 = P.r4_stage(checks.r4_out(curves, fail=[("pooled", 3)]), "a1", checks.r1_record(P))
    c = r4["cells"]["immune|1|pooled"]
    assert c["matched"] in (1.0, 10.0) and c["matched"] != 3.0


def test_r4_bstar_is_the_midpoint_of_the_common_range():
    curves = checks.flat_curves(P)
    curves["mmd"] = [0.40, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56]   # weakest arm: max 0.56
    r4 = P.r4_stage(checks.r4_out(curves), "a1", checks.r1_record(P))
    t = r4["targets"]["immune|1"]
    assert math.isclose(t["b_common"], 0.56) and math.isclose(t["bstar"], 0.40 + 0.5 * 0.16)
    assert not t["bstar_low_signal"]


def test_r4_bstar_ignores_runs_outside_the_stage():
    curves = checks.flat_curves(P)
    out = checks.r4_out(curves)
    extra = checks.r4_out({"discriminator": [0.99] * 10}, experiment="X13")
    both = pd.concat([out, extra], ignore_index=True)
    a = P.r4_stage(out, "a1", checks.r1_record(P))["targets"]
    b = P.r4_stage(both, "a1", checks.r1_record(P))["targets"]
    assert a == b


def test_r4_edge_asks_for_one_point_beyond_and_only_for_needed_cells():
    curves = checks.flat_curves(P)
    curves["discriminator"] = [0.49] + [0.60] * 9                  # closest to b* at the lowest lambda
    out = checks.r4_out(curves)
    r4 = P.r4_stage(out, "a1", checks.r1_record(P))
    assert r4["status"] == "extend"
    assert r4["extensions"] == [dict(experiment="A1", task="immune", cond=1, arm="discriminator", lam=0.03)]
    r4n = P.r4_stage(out, "a1", checks.r1_record(P), need={"immune|1|pooled"})
    assert r4n["status"] == "frozen"
    c = r4n["cells"]["immune|1|discriminator"]
    assert c["lo"] == 0.03 and any("edge_not_extended" in f for f in c["flags"])


def test_r4_edge_cap_sets_an_unfitted_neighbour():
    grid = [0.01, 0.03] + checks.GRID                               # two R4 rounds already fitted (A1 grid base)
    curves = {a: [0.40] * 2 + checks.flat_curves(P)[a] for a in P.ADV_ARMS}
    curves["discriminator"] = [0.49] + [0.60] * 11
    r4 = P.r4_stage(checks.r4_out(curves, grid=grid), "a1", checks.r1_record(P))
    c = r4["cells"]["immune|1|discriminator"]
    assert r4["status"] == "frozen" and c["matched"] == 0.01 and c["lo"] == 0.003
    assert any("edge_unresolved_low" in f for f in c["flags"])


# ---- R2 / R3 -----------------------------------------------------------------------------------
def test_bio_at_bstar():
    curve = [(1, 0.40, 0.70, 0), (3, 0.50, 0.66, 0), (10, 0.60, 0.60, 0)]
    assert P.bio_at_bstar(curve, 0.45) == dict(status="bracketed", bio=pytest.approx(0.68))
    assert P.bio_at_bstar(curve, 0.50)["bio"] == 0.66
    assert P.bio_at_bstar(curve, 0.30)["status"] == "unreached_high"
    assert P.bio_at_bstar(curve, 0.70)["status"] == "unreached_low"
    assert P.bio_at_bstar([], 0.5)["status"] == "no_points"


def test_decision_rope_is_strict():
    checks.check_rope_strict(P)


def test_decision_masking_forces_mean():
    checks.check_masking(P)


def test_decision_needs_both_tasks_positive():
    cells = checks._cells(0.05)
    for c in cells:
        if c["task"] == "t1":
            c["sample"]["bio"] = 0.70 - 0.005
    d = P.decide_binary(cells, "mean", "sample")
    assert d["beats_rope"] and not d["per_task_positive"] and d["choice"] == "mean"


def test_decision_needs_coverage():
    cells = checks._cells(0.05)
    for c in cells[:3]:
        c["sample"] = dict(status="unreached_high", bio=None, n_fail=0)
    d = P.decide_binary(cells, "mean", "sample")
    assert d["n_evaluable"] == 3 and not d["coverage"] and d["choice"] == "mean"


def test_decision_refuses_more_failures():
    cells = checks._cells(0.05)
    cells[0]["sample"]["n_fail"] = 1
    d = P.decide_binary(cells, "mean", "sample")
    assert d["coverage"] and d["beats_rope"] and d["choice"] == "mean"
    cells[1]["mean"]["n_fail"] = 1
    assert P.decide_binary(cells, "mean", "sample")["choice"] == "sample"


def test_decision_a3_has_no_masking_clause():
    cells = [dict(task=c["task"], arm=c["arm"], off=c["mean"], on=c["sample"]) for c in checks._cells(0.05)]
    cells[0]["on"] = dict(status="unreached_low", bio=None, n_fail=0)
    d = P.decide_binary(cells, "off", "on")
    assert d["choice"] == "on" and d["masked"] == []


# ---- inputs fail loudly ------------------------------------------------------------------------
def _small():
    M = synth.pilot_rows()
    A1 = M[(M.experiment == "A1")]
    S, F = synth.scores(A1)
    return M, S, F


def test_missing_row_raises():
    M, S, F = _small()
    with pytest.raises(P.PreregError, match="neither scored nor failed"):
        P.outcomes(M, S.iloc[1:], F, ["A1"])


def test_nan_required_metric_raises():
    M, S, F = _small()
    S.loc[S.index[0], "kBET"] = np.nan
    with pytest.raises(P.PreregError, match="non-finite"):
        P.outcomes(M, S, F, ["A1"])


def test_nan_in_a_metric_the_task_does_not_use_is_fine():
    M, S, F = _small()
    out = P.outcomes(M, S, F, ["A1"])
    atac = S.tag.isin(out.tag[out.task == "atac_small"])
    assert S.loc[atac, "cell_cycle_conservation"].isna().all() and out.B.notna().all()


def test_duplicate_score_and_double_status_raise():
    M, S, F = _small()
    with pytest.raises(P.PreregError, match="more than once"):
        P.outcomes(M, pd.concat([S, S.iloc[:1]]), F, ["A1"])
    F2 = pd.DataFrame([dict(tag=S.tag.iloc[0], status="diverged", detail="x")])
    with pytest.raises(P.PreregError, match="both scored and failed"):
        P.outcomes(M, S, F2, ["A1"])


def test_failure_rows_become_failed_outcomes(tmp_path):
    M, S, F = _small()
    F2 = pd.DataFrame([dict(tag=S.tag.iloc[0], status="nonfinite_latent", detail="z had NaN")])
    out = P.outcomes(M, S.iloc[1:], F2, ["A1"])
    assert (out.status == "failed").sum() == 1
    bad = tmp_path / "f.csv"
    pd.DataFrame([dict(tag="x", status="oom", detail="killed")]).to_csv(bad, index=False)
    with pytest.raises(P.PreregError, match="not in"):
        P.read_failures(str(bad))


def test_out_of_range_metric_raises():
    M, S, F = _small()
    S.loc[S.index[0], "iLISI"] = 3.2          # unscaled LISI would be > 1
    with pytest.raises(P.PreregError, match="scaled range"):
        P.outcomes(M, S, F, ["A1"])


def test_seed_separation_is_enforced():
    M = synth.pilot_rows()
    P.check_seed_separation(M)
    M.loc[M.index[M.experiment == "A1"][0], "seed"] = "0"
    with pytest.raises(P.PreregError, match="also X1 seeds"):
        P.check_seed_separation(M)


# ---- power -------------------------------------------------------------------------------------
def test_wilcoxon_floor_matches_scipy():
    p = stats.wilcoxon([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], alternative="two-sided", method="exact").pvalue
    assert math.isclose(p, 2 / 2 ** 6) and p > 0.05 / P.N_ENDPOINTS


def test_power_report_on_synthetic_a1():
    _, out = _a1_out()
    r1 = P.freeze_a1_grid(out)
    pw = P.power_report(out, P.r4_stage(out, "a1", r1, need=set()))
    d = pw["designs"]["real_tasks"]
    assert d["D"] == 6 and math.isclose(d["se"], math.sqrt((pw["tau"] ** 2 + pw["sigma"] ** 2 / 5) / 6))
    assert math.isclose(P._power(d["mde80"], d["se"], d["df"]), 0.8, abs_tol=1e-6)
    assert pw["frequentist_can_reject"] is False and math.isclose(pw["wilcoxon_min_p_two_sided"], 0.03125)
    assert pw["designs"]["families"]["D"] == 4
