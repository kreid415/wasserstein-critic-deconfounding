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


def _cells(D, n_tasks=2, arms=("discriminator", "reference", "pooled")):
    cells = []
    for i in range(n_tasks):
        for a in arms:
            cells.append(dict(task=f"t{i}", arm=a, mean=dict(status="bracketed", bio=0.70, n_fail=0),
                              sample=dict(status="bracketed", bio=0.70 + D, n_fail=0)))
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
    assert d["n_evaluable"] == 5 and d["coverage"] and d["beats_rope"], d
    assert d["choice"] == "mean" and d["masked"] == ["t0|discriminator"], d


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
