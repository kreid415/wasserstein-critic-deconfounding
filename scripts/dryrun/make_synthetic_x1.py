#!/usr/bin/env python
"""Synthetic score table shaped like the X1 + X13 main benchmark. NOT real data and not a prediction of any result.

Purpose: run scripts/analysis_x1.py and the figure and table scripts end to end before any main-benchmark score exists
(docs/ANALYSIS_DRYRUN.md). Fixed seed; every number derives from SEED and the row's tag, so the output is reproducible and
independent of row order.

Shape: 8 tasks (6 real, 2 simulations) x 2 decoders x plain scVI + 6 penalties at the 6-strength grids of tier12_v3/rules/
r1_pass2/r1_record.json (x1_grid) x seeds 0-4; scvi_adv (conditioned decoder); X13 baselines (scANVI, sysVI, Harmony, Scanorama, PCA).
Planted cases, each named in SCENARIO below: failed fits (some seeds of a strength; all seeds of a strength), curves that do not
reach b*, a curve that crosses b* twice, tied penalties, one low-signal
task and decoder pair, and bio collapse at high strength. Like the real flow
(freeze_matched_lambda.py), grid-edge extension rows are added in rounds until R4 is frozen, each round in its own manifest, score
directory (scores: one CSV per fit).

Usage: python scripts/dryrun/make_synthetic_x1.py --out <dir>   (writes <dir>/{manifests,scores,latents,failures.csv,
r1_record.json,r4_x1_record.json,SCENARIO.json})
"""
import argparse
import hashlib
import json
import math
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import prereg_rules as P  # noqa: E402

BPM = P.BPM
SEED = 20261028
# tier12_v3/rules/r1_pass2/r1_record.json families.<family>|<cond>.x1_grid (cond 1 = batch-conditioned decoder)
GRIDS = {"js|1": [0.1, 0.3, 1, 3, 10, 30], "js|0": [0.03, 0.1, 0.3, 1, 3, 10],
         "w1|1": [0.1, 0.3, 3, 10, 100, 300], "w1|0": [0.3, 1, 3, 10, 30, 100],
         "mmd|1": [3, 10, 30, 300, 1000, 3000], "mmd|0": [3, 10, 30, 100, 300, 1000],
         "sinkhorn|1": [0.3, 3, 30, 100, 1000, 10000], "sinkhorn|0": [0.3, 1, 3, 30, 100, 300]}
TASK_ORDER = ["pancreas", "lung", "immune", "immune_hum_mou", "atac_small", "atac_large", "sim1", "sim2"]
B0 = dict(zip(TASK_ORDER, [0.45, 0.38, 0.50, 0.35, 0.40, 0.36, 0.55, 0.42]))        # plain-scVI batch score, unconditioned decoder
C0 = dict(zip(TASK_ORDER, [0.62, 0.55, 0.68, 0.58, 0.45, 0.48, 0.72, 0.60]))        # plain-scVI biology score
TASK_GAIN = dict(zip(TASK_ORDER, [1.0, 1.1, 0.9, 1.0, 1.2, 1.0, 0.8, 1.0]))
COND_GAIN, COND_B0 = {0: 1.0, 1: 0.6}, {0: 0.0, 1: 0.07}
# arm: (batch gain at the last grid strength, growth exponent, bio loss there, loss exponent)
ARM = {"discriminator": (0.30, 0.6, 0.10, 2.0), "reference": (0.35, 0.6, 0.12, 2.0), "pooled": (0.35, 0.7, 0.12, 2.2),
       "barycenter": (0.22, 0.8, 0.04, 1.5), "mmd": (0.20, 0.9, 0.03, 1.5), "sinkhorn": (0.12, 1.0, 0.02, 1.5)}
SCENARIO = {
    # (task, cond, arm) -> arm whose metric values it copies exactly (a tie in biology at b*)
    "tie": {("sim2", 0, "barycenter"): "pooled", ("pancreas", 1, "pooled"): "reference"},
    # (task, cond, arm) -> gain at the last grid strength: batch score 0.03 below plain scVI below it, a large jump on the extension rows beyond it, so the
    # curve on the six grid strengths does not reach b* (b* counts the extension rows)
    "late_curve": {("lung", 0, "sinkhorn"): 0.05, ("atac_large", 1, "mmd"): 0.04, ("atac_large", 1, "sinkhorn"): 0.04},
    # (task, cond, arm) -> u of the batch peak; beyond it batch removal falls back below plain scVI (curve crosses b* twice)
    "fall_back": {("lung", 1, "pooled"): 0.6},
    # (task, cond, arm) -> absolute batch gain: the weakest penalty of this pair sets b_common - b0 below 0.02 (low signal)
    "low_signal": {("immune", 1, "barycenter"): 0.010},
    # (task, cond, arm) -> fraction of the grid (u) from which biology drops by a further 0.15 (collapse)
    "collapse": {("lung", 1, "discriminator"): 0.8, ("pancreas", 0, "reference"): 0.9, ("immune_hum_mou", 0, "pooled"): 0.75},
    # (task, cond, arm, grid position, seeds, status): fits with no score row
    "failed": [("lung", 0, "pooled", 5, [0, 1, 2, 3, 4], "diverged"), ("immune_hum_mou", 0, "mmd", 2, [0, 1, 2, 3, 4], "nonfinite_latent"), ("pancreas", 1, "reference", 4, [2], "nonfinite_latent"),
               ("pancreas", 1, "reference", 5, [0, 2, 4], "diverged"), ("immune_hum_mou", 1, "discriminator", 5, [1, 3], "diverged")],
}
METRIC_OFFSET_B = [0.05, -0.05, 0.03, -0.03, 0.0]
METRIC_OFFSET_C = [0.04, -0.04, 0.02, -0.02, 0.0, 0.01, -0.01, 0.0]
PROVENANCE = dict(scorer_git_sha="0" * 40, scorer_dirty="0", cpu_simd="avx2", numba_cpu_name="",
                  scorer_versions='{"synthetic": "scripts/dryrun/make_synthetic_x1.py"}')


def rng_for(*key):
    return np.random.default_rng([SEED, zlib.crc32("|".join(map(str, key)).encode())])


def ladder(lam):
    return int(round(2 * math.log10(float(lam))))     # half-decade steps


def u_of(arm, cond, lam):
    g = GRIDS[f"{P.FAMILY_OF[arm]}|{cond}"]
    lo, hi = ladder(g[0]), ladder(g[-1])
    return (ladder(lam) - lo) / (hi - lo)


def curve(task, cond, arm, lam):
    """Target (batch, biology) of one penalty at one strength; lam 0 = plain scVI."""
    b0, c0 = B0[task] + COND_B0[cond], C0[task]
    if arm in ("none", ""):
        return b0, c0
    alias = SCENARIO["tie"].get((task, cond, arm), arm)
    u = u_of(alias, cond, lam)
    g, p, d, q = ARM[alias]
    gain = g * TASK_GAIN[task] * COND_GAIN[cond] * max(u, 0.0) ** p if u >= 0 else 0.05 * g * u
    if (task, cond, alias) in SCENARIO["late_curve"]:
        g1 = SCENARIO["late_curve"][(task, cond, alias)]
        gain = -0.03 if u < 1 - 1e-9 else g1 + 14 * g1 * (u - 1)
    peak = SCENARIO["fall_back"].get((task, cond, alias))
    if peak is not None and u > peak:
        gain *= 1 - 1.6 * (u - peak) / (1 - peak)
    if (task, cond, alias) in SCENARIO["low_signal"]:
        gain = SCENARIO["low_signal"][(task, cond, alias)] * max(u, 0.0) ** p
    loss = d * TASK_GAIN[task] * max(u, 0.0) ** q
    cu = SCENARIO["collapse"].get((task, cond, alias))
    if cu is not None and u >= cu - 1e-9:
        loss += 0.15
    return min(b0 + gain, 0.92), c0 - loss


def baseline_curve(task, arm, lam, n_latent):
    b0, c0 = B0[task], C0[task]
    lam = float(lam)
    return {"scanvi": (b0 + 0.04 + COND_B0[1], c0 + 0.03), "pca": (b0 - 0.05, c0 - 0.10),
            "sysvi": (b0 + 0.07 + 0.08 * (1 - math.exp(-lam / 10)), c0 - 0.01 - 0.0006 * lam),
            "harmony": (b0 + 0.12 * (1 - math.exp(-lam / 2)) - (0.02 if n_latent != "10" else 0), c0 - 0.03 - 0.004 * lam),
            "scanorama": (b0 + 0.10 * (1 - math.exp(-lam / 30)) , c0 - 0.04 + 0.0005 * lam)}[arm]


def target(r):
    if r["experiment"] == "X13":
        return baseline_curve(r["task"], r["arm"], r["lam"], str(r["n_latent"]))
    cond = int(r["cond"])
    if r["arm"] == "scvi_adv":
        return B0[r["task"]] + COND_B0[1] + 0.03, C0[r["task"]] - 0.01
    return curve(r["task"], cond, r["arm"], 0 if r["arm"] == "none" else float(r["lam"]))


def noise_key(r):
    """Same key = same noise: a tied penalty reuses its source arm's noise at the same grid position."""
    if r["experiment"] == "X13":
        return (r["task"], r["arm"], r["lam"], r["n_latent"], r["seed"])
    if r["arm"] in ("none", "scvi_adv"):
        return (r["task"], r["cond"], r["arm"], r["seed"])
    arm = SCENARIO["tie"].get((r["task"], int(r["cond"]), r["arm"]), r["arm"])
    return (r["task"], r["cond"], arm, round(u_of(arm, int(r["cond"]), float(r["lam"])), 6), r["seed"])


def build_manifest():
    bb = BPM.BACKBONES["stock"]
    rows = []
    for t in TASK_ORDER:
        for cond in (1, 0):
            for s in range(5):
                rows.append(BPM.row("X1", t, "none", 0, s, bool(cond), bb))
                if cond:
                    rows.append(BPM.row("X1", t, "scvi_adv", 0, s, True, bb))
                for arm in BPM.ARMS:
                    for lam in GRIDS[f"{P.FAMILY_OF[arm]}|{cond}"]:
                        rows.append(BPM.row("X1", t, arm, P.fmt_lam(lam), s, bool(cond), bb))
    for t, s in [(t, s) for t in TASK_ORDER for s in range(5)]:
        rows.append(BPM.row("X13", t, "scanvi", 0, s, True, bb))
        rows += [BPM.row("X13", t, "sysvi", w, s, True, bb) for w in [1, 2, 5, 10, 20, 50]]
    for t in TASK_ORDER:
        for arm, (knob, values, default, tool_dims) in BPM.CPU_BASELINES.items():
            rows += [BPM.cpu_row(t, arm, knob, v, bb["n_latent"]) for v in values] + [BPM.cpu_row(t, arm, knob, default, tool_dims)]
    rows = BPM.finalize(rows)
    return pd.DataFrame([{c: str(r[c]) for c in BPM.COLS} for r in rows], columns=BPM.COLS)


def labels(task):
    rng = rng_for("labels", task)
    batch, ct = rng.integers(0, 3, 180), rng.integers(0, 3, 180)
    ct[:12], batch[:12] = 3, 0             # cell type 3 only in batch 0: dropped by the label-transfer rule (spec 9)
    return np.array([f"batch{b}" for b in batch]), np.array([f"type{c}" for c in ct])


def write_latent(path, r, b):
    batch, ct = labels(r["task"])
    rng = rng_for("latent", r["tag"])
    removal = float(np.clip((b - B0[r["task"]]) / 0.4, 0, 1))
    shift = np.array([[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0]])
    centre = np.array([[3.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 3.0, 0], [0, 0, 0, 3.0]])
    z = centre[[int(c[-1]) for c in ct]] + (1 - removal) * shift[[int(x[-1]) for x in batch]] + rng.normal(0, 0.8, (180, 4))
    np.savez(path, z=z.astype("float32"), celltype=ct, batch=batch)
    return P.sha256(path)


def score_rows(M, latent_dir):
    """Scores and failures of the manifest rows M (X1 and X13), and the latent files of the X1 rows that succeeded."""
    failed = {}
    for task, cond, arm, i, seeds, status in SCENARIO["failed"]:
        lam = P.fmt_lam(GRIDS[f"{P.FAMILY_OF[arm]}|{cond}"][i])
        for tag in M.tag[(M.task == task) & (M.cond.astype(int) == cond) & (M.arm == arm) & (M.lam == lam) & M.seed.astype(int).isin(seeds)]:
            failed[tag] = status
    scores, fails = [], []
    for r in M.to_dict("records"):
        if r["tag"] in failed:
            fails.append(dict(tag=r["tag"], status=failed[r["tag"]], detail="synthetic planted failure", experiment=r["experiment"],
                              task=r["task"], arm=r["arm"], lam=r["lam"], cond=r["cond"], seed=r["seed"], epoch=0, step=0,
                              host="synthetic", recorded_at="2026-10-28T00:00:00"))
            continue
        b, c = target(r)
        rng = rng_for(*noise_key(r))
        common = rng_for("seed", r["task"], r["cond"], r["seed"]).normal(0, 0.01, 2)
        b, c = b + common[0] + rng.normal(0, 0.008), c + common[1] + rng.normal(0, 0.008)
        row = dict(tag=r["tag"], experiment=r["experiment"], task=r["task"], arm=r["arm"], lam=r["lam"], cond=r["cond"], seed=r["seed"])
        for j, m in enumerate(P.BATCH_METRICS):
            row[m] = b + METRIC_OFFSET_B[j] + rng.normal(0, 0.004)
        need_c = P.bio_metrics_for(r["task"])
        for j, m in enumerate(P.BIO_METRICS):
            row[m] = c + METRIC_OFFSET_C[j] + rng.normal(0, 0.004) if m in need_c else np.nan
        assert all(0 <= row[m] <= 1 for m in P.BATCH_METRICS) and all(-1 <= row[m] <= 1 for m in need_c), r["tag"]
        row["latent_sha256"] = write_latent(os.path.join(latent_dir, r["tag"] + ".npz"), r, b) if r["experiment"] == "X1" else ""
        scores.append(dict(row, kbet_seed=0, **PROVENANCE))
    fails = pd.DataFrame(fails, columns=["tag", "status", "detail", "experiment", "task", "arm", "lam", "cond", "seed", "epoch",
                                         "step", "host", "recorded_at"])
    return pd.DataFrame(scores), fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "manifests"))
    os.makedirs(os.path.join(a.out, "latents"))
    r1 = dict(rule="R1", status="frozen", families={k: dict(x1_grid=v) for k, v in GRIDS.items()})
    r1_path = os.path.join(a.out, "r1_record.json")
    P.write_record(r1_path, r1, {})
    r1 = P.read_record(r1_path, "R1")
    rounds, all_scores, all_fails = [], [], []
    M = build_manifest()
    for rnd in range(4):
        name = "x1" if rnd == 0 else f"ext_round{rnd}"
        s, f = score_rows(M, os.path.join(a.out, "latents"))
        os.makedirs(os.path.join(a.out, "scores", name))
        for tag, row in s.groupby("tag"):      # one file per fit, as score_scib_native.py writes them
            row.to_csv(os.path.join(a.out, "scores", name, tag + ".csv"), index=False)
        P.write_manifest(M, ["# synthetic manifest (scripts/dryrun/make_synthetic_x1.py)"], os.path.join(a.out, "manifests", name + ".tsv"),
                         f"round {rnd}")
        rounds.append(M); all_scores.append(s); all_fails.append(f)
        Mall, Sall, Fall = pd.concat(rounds, ignore_index=True), pd.concat(all_scores, ignore_index=True), pd.concat(all_fails, ignore_index=True)
        out = P.outcomes(Mall, Sall, Fall, ["X1", "X13"])
        r4 = P.r4_stage(out, "x1", r1)
        if r4["status"] == "frozen":
            break
        M = P.extension_rows(Mall, r4["extensions"])
    else:
        raise SystemExit("R4 did not freeze in 4 rounds")  # pragma: no cover  # a defect of the planted cases, not of the analysis
    Fall.to_csv(os.path.join(a.out, "failures.csv"), index=False)
    P.write_record(os.path.join(a.out, "r4_x1_record.json"), r4, {"manifest": [os.path.join(a.out, "manifests", n) for n in sorted(os.listdir(os.path.join(a.out, "manifests")))]})
    json.dump({k: [list(map(str, x)) if isinstance(x, tuple) else x for x in v] if isinstance(v, list) else {"|".join(map(str, kk)): vv for kk, vv in v.items()}
               for k, v in SCENARIO.items()}, open(os.path.join(a.out, "SCENARIO.json"), "w"), indent=1)
    print(f"synthetic X1/X13: {len(Mall)} rows in {len(rounds)} rounds, {len(Fall)} failed fits, R4 {r4['status']} -> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
