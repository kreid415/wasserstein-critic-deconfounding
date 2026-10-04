"""Decision-boundary checks shared by the rule tests and the mutation tests. Each takes the rule module
(the real scripts/prereg_rules.py or a mutated copy) and raises AssertionError if a rule is violated."""
from synth import out_frame

GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
SEEDS = [100, 101, 102]


def r1_record(P, grid=GRID):
    return {"rule": "R1", "status": "frozen",
            "families": {f"{fam}|{c}": {"a1_grid": [float(v) for v in grid], "x1_grid": [float(v) for v in grid[:6]]}
                         for fam in P.FAMILIES for c in (0, 1)}}


def r4_out(curves, b0=0.40, task="immune", cond=1, experiment="A1", grid=GRID, seeds=SEEDS, fail=()):
    """curves: arm -> list of seed-mean B over grid (every seed gets the same B). fail: (arm, lam) failed points."""
    rec = [dict(experiment=experiment, task=task, cond=cond, arm="none", lam=0, seed=s, B=b0, C=0.7) for s in seeds]
    for arm, bs in curves.items():
        for lam, b in zip(grid, bs):
            for s in seeds:
                st = "failed" if (arm, lam) in fail else "ok"
                rec.append(dict(experiment=experiment, task=task, cond=cond, arm=arm, lam=lam, seed=s,
                                B=(b if st == "ok" else float("nan")), C=0.7, status=st))
    return out_frame(rec)


def flat_curves(P, peak=0.60):
    return {a: [0.40, 0.40, 0.45, 0.50, 0.55, peak, peak, peak, peak, peak] for a in P.ADV_ARMS}


def check_r4_tie(P):
    """Two failure-free points equidistant from b* -> the smaller lambda wins."""
    curves = flat_curves(P)
    curves["reference"] = [0.40, 0.40, 0.45, 0.55, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]
    r4 = P.r4_stage(r4_out(curves), "a1", r1_record(P))
    c = r4["cells"]["immune|1|reference"]
    assert abs(r4["targets"]["immune|1"]["bstar"] - 0.5) < 1e-12, r4["targets"]
    assert (c["matched"], c["lo"], c["hi"]) == (1.0, 0.3, 3.0), c


def _cells(D, n_tasks=2, arms=("discriminator", "reference", "pooled"), conds=(1, 0)):
    """Pilot cells task x decoder x arm (A2 shape by default: 2 x 2 x 3 = 12), every cell bracketed."""
    cells = []
    for i in range(n_tasks):
        for c in conds:
            for a in arms:
                cells.append(dict(task=f"t{i}", cond=c, arm=a, group=f"t{i}|{c}",
                                  mean=dict(status="bracketed", bio=0.70, n_fail=0),
                                  sample=dict(status="bracketed", bio=0.70 + D, n_fail=0)))
    return cells


A3_ARMS = ("discriminator", "reference", "pooled", "mmd", "sinkhorn")


def _unevaluable(cells, k):
    """Make k cells unevaluable, spread over the task x decoder groups (each group keeps >= 1 evaluable)."""
    order = sorted(range(len(cells)), key=lambda i: (i % 5, i))
    for i in order[:k]:
        cells[i]["sample"] = dict(status="unreached_high", bio=None, n_fail=0)
    return cells


def check_rope_strict(P):
    """mean D exactly ROPE does not switch; just above does."""
    at = P.decide_binary(_cells(0.01), "mean", "sample", masking_setting="sample", safe_setting="mean")
    assert at["choice"] == "mean" and not at["switch"], at
    above = P.decide_binary(_cells(0.0101), "mean", "sample", masking_setting="sample", safe_setting="mean")
    assert above["choice"] == "sample" and above["switch"], above


def check_masking(P):
    """A cell where samples miss b* while means reach it forces the posterior mean."""
    cells = _cells(0.05)
    cells[0]["sample"] = dict(status="unreached_low", bio=None, n_fail=0)
    d = P.decide_binary(cells, "mean", "sample", masking_setting="sample", safe_setting="mean")
    assert d["n_evaluable"] == 11 and d["coverage"] and d["beats_rope"], d
    assert d["choice"] == "mean" and d["masked"] == ["t0|1|discriminator"], d


def check_coverage_threshold(P):
    """>= ceil(2n/3) evaluable cells: A2 (12 cells) needs 8, A3 (20 cells) needs 14."""
    for arms, n, need in ((("discriminator", "reference", "pooled"), 12, 8), (A3_ARMS, 20, 14)):
        ok = P.decide_binary(_unevaluable(_cells(0.05, arms=arms), n - need), "mean", "sample")
        assert ok["n_cells"] == n and ok["n_evaluable"] == need and ok["coverage"] and ok["switch"], ok
        short = P.decide_binary(_unevaluable(_cells(0.05, arms=arms), n - need + 1), "mean", "sample")
        assert short["n_evaluable"] == need - 1 and not short["coverage"] and short["choice"] == "mean", short


def check_group_sign(P):
    """The sign criterion is checked within each task x decoder, not only within each task."""
    cells = _cells(0.05)
    for c in cells:
        if c["group"] == "t0|0":
            c["sample"]["bio"] = 0.70 - 0.005
    d = P.decide_binary(cells, "mean", "sample")
    assert d["beats_rope"] and d["coverage"] and not d["per_group_positive"] and d["choice"] == "mean", d


def _block_points(db_means, dc_means, noise=(0.0, 0.0, 0.0), fail=()):
    zero = {s: (0.40, 0.70) for s in SEEDS}
    pts = []
    for i, (lam, mb, mc) in enumerate(zip(GRID, db_means, dc_means)):
        nf = 1 if i in fail else 0
        ok = SEEDS[nf:]
        pts.append(dict(lam=float(lam), n_fail=nf,
                        B={s: 0.40 + mb + noise[SEEDS.index(s)] for s in ok},
                        C={s: 0.70 + mc + noise[SEEDS.index(s)] for s in ok}))
    return pts, zero


def check_collapse_boundary(P):
    """A bio loss of exactly C = 0.10 is collapse; 0.0999 is not."""
    jitter = (0.001, 0.0, -0.001)
    pts, zero = _block_points([0, 0, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                              [0, 0, -0.02, -0.0999, -0.10, -0.2, -0.2, -0.2, -0.2, -0.2], jitter)
    b = P.r1_block(pts, zero)
    assert [q["collapsed"] for q in b["points"]][3:5] == [False, True], b["points"][3:5]


def check_noise_threshold(P):
    """With seed noise SD 0.03 a mean shift of 0.02 at the lowest lambda is not an effect (3 SE rule)."""
    noise = (0.03, 0.0, -0.03)
    pts, zero = _block_points([0.02, 0.02, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                              [0, 0, 0, 0, -0.05, -0.2, -0.2, -0.2, -0.2, -0.2], noise)
    b = P.r1_block(pts, zero)
    assert not b["low_edge"] and b["floor"] == 1, {k: b[k] for k in ("floor", "low_edge", "sigma_b")}


# ---- code check CR-05, rule side: scorer provenance of one rule input (prereg_rules.check_provenance) -----------------
def _prov_input():
    """A1 rows of one task x decoder (manifest frame) with uniform synthetic scores (synth.PROVENANCE)."""
    import synth
    M = synth.pilot_rows()
    M = M[(M.experiment == "A1") & (M.task == "immune") & (M.cond == "1")].reset_index(drop=True)
    S, F = synth.scores(M)
    return M, S, F


def _refused(P, M, S, F, pattern):
    import re
    try:
        P.outcomes(M, S, F, ["A1"])
    except P.PreregError as e:
        assert re.search(pattern, str(e)), f"refused for another reason: {e}"
        return
    except Exception as e:      # a crash is not the pre-registered refusal: report it as a failed check
        raise AssertionError(f"{type(e).__name__} instead of PreregError ({pattern}): {e}") from e
    raise AssertionError(f"accepted a rule input that must be refused ({pattern})")


def check_provenance_mix(P):
    """Uniform provenance passes; one row with another scorer commit, CPU type, NUMBA_CPU_NAME, dirty flag or
    package versions is refused for mixing."""
    M, S, F = _prov_input()
    assert len(P.outcomes(M, S, F, ["A1"])) == len(M)
    for col, other in (("scorer_git_sha", "f" * 40), ("cpu_simd", "avx512f"), ("numba_cpu_name", "skylake-avx512"),
                       ("scorer_dirty", 1), ("scorer_versions", '{"scib": "1.1.6"}')):
        S2 = S.copy()
        S2[col] = S2[col].astype(object)
        S2.loc[3, col] = other
        _refused(P, M, S2, F, f"mix scorer provenance: {col}")


def check_provenance_missing(P):
    """Rows without the columns (a score file written before CR-05, concatenated with current ones) and rows with an
    empty value (other than numba_cpu_name) are refused; a score table without a column is refused."""
    import pandas as pd
    M, S, F = _prov_input()
    old = S.iloc[:5].drop(columns=["scorer_git_sha", "scorer_dirty", "cpu_simd", "numba_cpu_name", "scorer_versions"])
    _refused(P, M, pd.concat([old, S.iloc[5:]], ignore_index=True), F, "lack scorer provenance values")
    S2 = S.copy()
    S2.loc[2, "scorer_git_sha"] = ""
    _refused(P, M, S2, F, "lack scorer provenance values")
    _refused(P, M, S.drop(columns=["cpu_simd"]), F, r"lack the scorer provenance columns \['cpu_simd'\]")


# ---- SI-45 (code check CR-11): an R4-a1 window neighbour without an A1 fit -> default-half A1 rows --------------------
SI45_CELL = ("immune", 1, "pooled")
SI45_LOW_EXT = [0.03, 0.01]     # the two R4-a1 extension rounds below the A1 grid (R4_MAX_ROUNDS x R4_EXT_POINTS)


def si45_b(arm, lam):
    """Seed-mean batch score of the edge world (b0 = 0.40): the discriminator tops out at 0.47 (sets b_common, so
    b* = 0.435), the other arms reach 0.60 inside the grid; pooled is already at b* at lambda 0.01 and rises."""
    import math
    x = math.log10(float(lam))
    if arm == "none":
        return 0.40
    if arm == SI45_CELL[2]:
        return min(0.435 + 0.05 * (x + 2), 0.75)
    top = 0.47 if arm == "discriminator" else 0.60
    return 0.40 + (top - 0.40) / (1 + math.exp(-(x - 0.5) / 0.3))


def si45_out(P, include_new_point):
    """outcomes()-shaped frame of the edge cell: A1 'none', every adversarial arm on the A1 grid, pooled also at the two
    R4 extension points (and, if include_new_point, the SI-45 point 0.003); A2 rows at the window."""
    t, c, arm = SI45_CELL
    rec = [dict(experiment="A1", task=t, cond=c, arm="none", lam=0, seed=s, B=0.40, C=0.70, adv_input="mean")
           for s in SEEDS]
    for a in P.ADV_ARMS:
        lams = GRID + (SI45_LOW_EXT if a == arm else []) + ([0.003] if (a == arm and include_new_point) else [])
        rec += [dict(experiment="A1", task=t, cond=c, arm=a, lam=lam, seed=s, B=si45_b(a, lam), C=0.70,
                     adv_input="mean") for lam in lams for s in SEEDS]
    rec += [dict(experiment="A2", task=t, cond=c, arm=arm, lam=lam, seed=s, B=si45_b(arm, lam), C=0.71,
                 adv_input="sample") for lam in (0.003, 0.01, 0.03) for s in SEEDS]
    return out_frame(rec)


def check_si45_flagged(P):
    """The edge cell gets edge_unresolved ['lo'] (lo = 0.003 without an A1 fit) and exactly one default-half spec."""
    t, c, arm = SI45_CELL
    r4 = P.r4_stage(si45_out(P, False), "a1", r1_record(P), need={f"{t}|{c}|{arm}"})
    cell = r4["cells"][f"{t}|{c}|{arm}"]
    assert (cell["matched"], cell["lo"], cell["hi"]) == (0.01, 0.003, 0.03), cell
    assert cell.get("edge_unresolved") == ["lo"], cell
    assert P.default_half_extensions(r4) == [dict(experiment="A1", task=t, cond=c, arm=arm, lam=0.003)]
