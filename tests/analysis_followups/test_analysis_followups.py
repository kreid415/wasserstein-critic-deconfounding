"""Known-answer, planted-effect and null tests for scripts/analysis_followups.py (docs/ANALYSIS_SPEC_FOLLOWUPS.md).

Synthetic inputs only (SI-78): no X1, X13 or follow-up score is read. The synthetic design follows the 2026-10-17
decisions (SI-93..SI-98) and the names assumed in the specification's section 1.4. Every planted value is stated
in `world`; the expected estimates follow from it by hand.
"""
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests", "prereg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analysis_followups as F  # noqa: E402
import prereg_rules as P  # noqa: E402
import synth  # noqa: E402

LAM = {"matched_lo": "1", "matched": "3", "matched_hi": "10", "0": "0"}
BSTAR = 0.5
LOW = {"immune|1", "sim1|1"}             # flagged low-signal in the synthetic R4 record
N_TYPES = 10
X7_REFS = {"pancreas": [10, 9, 8, 7, 6, 5, 4, 3, 2], "sim1": [10, 8, 6, 5, 4, 3], "atac_small": [10, 6, 3]}  # covered types
X7_DEPLETED = {"two_thirds": 7, "one_third": 3}      # covered types of the depleted reference (index 0)
X8_TASKS = ("immune_hum_mou", "lung", "sim2")


def row(exp, task, arm, level, seed, cond, n_critic=1, reference="auto", extra=None, **kw):
    r = dict(experiment=exp, task=task, counts="raw", arm=arm, lam=LAM[level], n_critic=n_critic, adv_input="mean", zstd=0,
             cond=cond, decoder="SCVI", n_latent=10, n_layers=1, n_hidden=128, likelihood="zinb", batch_size=128,
             max_epochs=400, train_size=0.9, seed=seed, reference=reference, extra=json.dumps(extra or {}))
    r.update(kw)
    key = "|".join(f"{c}={r[c]}" for c in sorted(r))
    r["tag"] = f"{exp}_{task}_{arm}_l{level}_c{cond}_s{seed}_" + hashlib.sha1(key.encode()).hexdigest()[:8]
    return {c: str(v) for c, v in r.items()}


def design():
    R, S = [], F.FU_SEEDS
    for (t, c), s in itertools.product(sorted(F.EXPECTED["X3"]), S):
        for d in F.X3_DOSES:
            sub = dict(subsample=dict(kind="composition", deplete_pct=d))
            R.append(row("X3", t, "none", "0", s, c, extra=sub))
            for a, lv in itertools.product(F.X3_ARMS, F.FIXED):
                R.append(row("X3", t, a, lv, s, c, extra=sub))
                if d in F.X3_IW_DOSES:
                    R.append(row("X3", t, a, lv, s, c, extra=dict(sub, iw="depletion_oracle")))
    for (t, c), s, (a, k), lv in itertools.product(sorted(F.EXPECTED["X6"]), S, F.X6_CONFIGS.values(), F.LEVELS):
        R.append(row("X6", t, a, lv, s, c, n_critic=k))
    for (t, c), s, f, lv in itertools.product(sorted(F.EXPECTED["X7"]), S, F.X7_FORMS, F.LEVELS):
        R += [row("X7", t, f, lv, s, c, reference=str(i)) for i in range(len(X7_REFS[t]))]
        if t != "atac_small":
            R += [row("X7", t, f, lv, s, c, reference="0", extra=dict(subsample=dict(kind="reference_depletion", level=d)))
                  for d in F.X7_DEPLETION]
    for (t, c), s, V, ch in itertools.product(sorted(F.EXPECTED["X8"]), S, F.X8_V, (0, 1)):
        sub = dict(subsample=dict(kind="batches", n_batches=V, subset=ch))
        R += [row("X8", t, a, "0" if a == "none" else "matched", s, c, extra=sub) for a in F.X8_ARMS]
        R += [row("X8", t, a, "matched", s, c, extra=dict(sub, sampler="stratified")) for a in ("reference", "pooled")]
    for (t, c), s, run, a, lv in itertools.product(sorted(F.EXPECTED["X12"]), S, range(16), ("discriminator", "reference"), F.FIXED):
        A, B, C, D = run & 1, (run >> 1) & 1, (run >> 2) & 1, (run >> 3) & 1
        R.append(row("X12", t, a, lv, s, c, extra=dict(factorial_run=run, adv_width=[32, 128][B], adv_lr=[1e-4, 1e-3][C]),
                     n_latent=[10, 30][A], batch_size=[128, 512][D], decoder=["SCVI", "LinearSCVI"][A ^ B ^ C ^ D]))
    for (t, c), s, kl in itertools.product(sorted(F.EXPECTED["X15"]), S, ("stock", "complete")):
        R.append(row("X15", t, "none", "0", s, c, extra=dict(kl_warmup=kl)))
        R += [row("X15", t, a, lv, s, c, extra=dict(kl_warmup=kl)) for a, lv in itertools.product(F.X15_ARMS, F.FIXED)]
    return pd.DataFrame(R)


def coverage_of(task, ref, level="natural"):
    return (X7_DEPLETED[level] if level != "natural" else X7_REFS[task][ref]) / N_TYPES


def counts_table():
    C = []
    for t, ks in X7_REFS.items():
        for i, k in enumerate(ks):
            C += [dict(task=t, draw="full", batch=f"b{i}", cell_type=f"ct{j}", n=(20 + 5 * j + i) if j < k else 3) for j in range(N_TYPES)]
        if t != "atac_small":
            for lv, k in X7_DEPLETED.items():
                C += [dict(task=t, draw=f"x7_{lv}", batch="b0", cell_type=f"ct{j}", n=(20 + 5 * j) if j < k else 0) for j in range(N_TYPES)]
    for t, ch, V in itertools.product(X8_TASKS, (0, 1), F.X8_V):     # nested chains, 100 cells per batch (SI-94)
        C += [dict(task=t, draw=f"x8_V{V}_s{ch}", batch=f"b{ch}_{b}", cell_type=f"ct{j}",     # chain 1: composition moves with V
                   n=25 if ch == 0 else (40 if j == (b // 2) % 4 else 20)) for b in range(V) for j in range(4)]
    return pd.DataFrame(C, columns=["task", "draw", "batch", "cell_type", "n"])


def world(effects):
    """(status, B, C) per row. Planted (when `effects`): X3 the reference critic loses 0.10 more biology per 100%
    depletion than the other arms, which the oracle weights undo; X6 W1+SN at 1 step keeps 0.05 more biology at every
    lambda; X7 the reference critic's biology rises 0.2 per unit coverage, the comparators' not at all (natural and
    depleted references); X8 the reference critic's batch score falls 0.02 per doubling of V (default sampler only);
    X12 reference minus discriminator = 0.03 + 0.03 x code(A), so factor A's effect is 0.06; X15 none. Seed effects are
    shared by every arm of a task x decoder (pairing). One failed run each in X6 and X3; X12 atac_small|1 run 5
    reference at matched_hi fails on every seed; three X7 curves and one X6 curve never reach b*."""
    def fn(r):
        exp, t, c, s, arm = r["experiment"], r["task"], int(r["cond"]), int(r["seed"]), r["arm"]
        ex, lam = json.loads(r["extra"]), float(r["lam"])
        sb = synth._noise(f"b{t}|{c}|{s}", 0.01)
        sc = synth._noise(f"c{t}|{c}|{s}", 0.01)
        e = synth._noise("e" + r["tag"], 0.001)
        if r["tag"] in FAILED:
            return "diverged", None, None
        if exp == "X3":
            dose = ex["subsample"]["deplete_pct"] / 100
            C = 0.70 - 0.05 * dose + sc
            if arm != "none":
                C -= 0.02 + (0.10 * dose if effects and arm == "reference" and "iw" not in ex else 0.0)
            return "ok", 0.6 + sb + e, C + e
        if exp in ("X6", "X7"):
            B = 0.4 + 0.2 * (math.log10(lam) + 0.5) / 1.5 + sb + e
            lv = ex.get("subsample", {}).get("level", "natural")
            if (exp == "X6" and t == "pancreas" and c == 0 and arm == "discriminator_r1") or \
                    (exp == "X7" and t == "atac_small" and c == 0 and arm == "reference" and r["reference"] == "2") or \
                    (exp == "X7" and t == "sim1" and c == 0 and arm == "discriminator_ref" and lv == "one_third"):
                B = 0.3 + sb                                    # never reaches b* = 0.5
            C = 0.70 - 0.05 * math.log10(lam) + sc + e
            if exp == "X6" and effects and arm == "pooled_sn" and int(r["n_critic"]) == 1:
                C += 0.05
            if exp == "X7" and effects and arm == "reference":
                C += 0.2 * coverage_of(t, int(r["reference"]), lv)
            return "ok", B, C
        if exp == "X8":
            V = ex["subsample"]["n_batches"]
            B = 0.5 + sb + e - (0.02 * math.log2(V) if effects and arm == "reference" and "sampler" not in ex else 0.0)
            return "ok", B, 0.7 + sc + e
        if exp == "X12":
            a = 1 if int(r["n_latent"]) == 30 else -1
            return "ok", 0.55 + sb + e, 0.7 + sc + e + ((0.03 + 0.03 * a) if effects and arm == "reference" else 0.0)
        return "ok", 0.55 + sb + e, 0.7 + sc + e - (0.02 if arm != "none" else 0.0)
    return fn


M = design()
_DOSE100 = json.dumps(dict(subsample=dict(kind="composition", deplete_pct=100)))
FAILED = {M.tag[(M.experiment == "X6") & (M.task == "atac_small") & (M.cond == "1") & (M.arm == "discriminator")
                & (M.lam == "3") & (M.seed == "11")].item(),
          M.tag[(M.experiment == "X3") & (M.task == "pancreas") & (M.cond == "1") & (M.arm == "mmd") & (M.lam == "10")
                & (M.seed == "12") & (M.extra == _DOSE100)].item()} | \
    set(M.tag[(M.experiment == "X12") & (M.task == "atac_small") & (M.cond == "1") & (M.arm == "reference") & (M.lam == "10")
              & M.extra.str.contains('"factorial_run": 5,')])


def r4_record():
    tasks = {t for e in F.EXPECTED.values() for t, _ in e}
    return dict(rule="R4", stage="x1", status="frozen",
                targets={f"{t}|{c}": dict(b0=0.4, bstar=BSTAR, b_common=0.6, bstar_low_signal=f"{t}|{c}" in LOW)
                         for t in tasks for c in (0, 1)})


def write_inputs(tmp, effects):
    man = synth.write_tsv(M, os.path.join(tmp, "manifest.tsv"))
    sp, fp = synth.write_scores(*synth.scores(M, world(effects)), os.path.join(tmp, "scores"), "fu")
    r4 = os.path.join(tmp, "r4.json")
    json.dump(r4_record(), open(r4, "w"))
    cp = os.path.join(tmp, "counts.csv")
    counts_table().to_csv(cp, index=False)
    return ["--manifest", man, "--scores", sp, "--failures", fp, "--r4-record", r4, "--celltype-counts", cp,
            "--out-json", os.path.join(tmp, "followups.json")]


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    out = {}
    for name, eff in (("effect", True), ("null", False)):
        tmp = str(tmp_path_factory.mktemp(name))
        args = write_inputs(tmp, eff)
        assert F.main(args) == 0
        out[name] = json.load(open(args[-1]))
        out[name + "_args"] = args
    return out


def test_known_answers():  # test-reason: jackknife, slope, iLISI ceiling, coverage and bio window against hand values
    vals = {10: 1.0, 11: 2.0, 12: 4.0}
    r = F.loso(lambda keep: F.mean(vals, keep))
    assert math.isclose(r["d"], 7 / 3) and math.isclose(r["s"], np.std([1, 2, 4], ddof=1) / math.sqrt(3))   # jackknife of a mean
    assert F.loso(lambda keep: None if 10 not in keep else 1.0)["reason"] == "readings"
    assert F.slope([0, 1, 2], [1, 3, 5]) == 2.0 and F.slope([0, 1], [1, None]) is None
    assert math.isclose(F.ilisi_ceiling([100, 300]), 0.6) and math.isclose(F.ilisi_ceiling([50] * 8), 1.0)  # 1/(1/16+9/16) = 1.6
    cov = F.coverage(counts_table(), "pancreas", "full", "b1")
    assert cov["coverage"] == 0.9 and cov["size"] == sum(20 + 5 * j + 1 for j in range(9)) + 3
    pts = [(1.0, {}, {10: 0.70, 11: 0.72, 12: 0.74}, 0), (10.0, {}, {10: 0.60, 11: 0.61, 12: 0.62}, 0)]
    assert F.bio_window(pts) == dict(width=0.0, n_points=1, bound=pytest.approx(0.72 - 4.302653 * 0.02 / math.sqrt(3), abs=1e-6))
    beta, se = F.ols([[1, 0.5], [1, 0.5], [1, 0.5]], [1, 2, 3])
    assert se is None                                         # a predictor that does not vary has no slope
    assert F.bio_window(pts[:1] + [(10.0, {}, {10: 0.80}, 2)])["reason"] == "best point has fewer than 2 runs"


def test_planted_effects(records):  # test-reason: each follow-up's endpoint recovers its planted value and declaration
    rec = records["effect"]
    x3 = rec["X3"]
    for c, lv in itertools.product((0, 1), F.FIXED):
        p = x3["pooled"][f"slope_contrast|{c}|{lv}"]
        assert all(x3["contrast"][f"{t}|{c}|{lv}"]["d"] == pytest.approx(-0.10, abs=0.005) for t in p["units"])
        # the pooled mean is shrunk towards 0 by the prior N(0, 0.1^2) (about -0.094 on 3 datasets)
        assert p["n"] == 3 and p["mean"] == pytest.approx(-0.10, abs=0.01) and p["declaration"] == "below"
        assert x3["contrast"][f"sim2|{c}|{lv}"]["d"] == pytest.approx(-0.10, abs=0.005)    # simulation: per task, never pooled
        assert "sim2" not in p["units"]
    cell = x3["cells"]["pancreas|1"]
    assert cell["reference|matched_lo"]["iw_recovery"]["d"] == pytest.approx(0.10 * 0.75, abs=0.003)  # mean dose of 50/80/95
    assert cell["reference|matched_lo"]["collapse"]["100"]["x"] == 3                    # -0.12 against the 0.10 bound
    assert cell["mmd|matched_hi"]["collapse"]["100"]["x"] == 1 and cell["mmd|matched_hi"]["collapse"]["100"]["n_failed"] == 1
    x6 = rec["X6"]
    assert x6["pooled"]["divergence|0"]["mean"] == pytest.approx(0.05, abs=0.01)
    assert x6["contrast"]["pancreas|0|divergence"]["d"] == pytest.approx(0.05, abs=0.003)
    assert x6["pooled"]["divergence|0"]["declaration"] == "above" and x6["pooled"]["js_optimiser|0"]["declaration"] != "above"
    assert x6["pooled"]["divergence|1"]["not_evaluable"] == {"immune": "low_signal"} and x6["pooled"]["divergence|1"]["n"] == 2
    assert x6["contrast"]["pancreas|0|js_r1"]["reason"] == "unreached" and x6["cells"]["pancreas|0"]["js_r1"]["status"] == "unreached_low"
    assert x6["contrast"]["atac_small|1|js_sn"]["m"] == 3                                 # the failed run leaves every reading
    x7 = rec["X7"]
    for c in (0, 1):
        assert x7["association"][f"pancreas|{c}|discriminator_ref"]["d"] == pytest.approx(0.2, abs=0.01)
        assert x7["depleted"][f"pancreas|{c}|discriminator_ref"]["d"] == pytest.approx(0.2, abs=0.01)
        assert x7["pooled"][f"association_discriminator_ref|{c}"]["n"] == 2 - (c == 0)
    assert x7["association"]["atac_small|1|mmd_ref"]["d"] == pytest.approx(0.2, abs=0.01)
    assert x7["association"]["atac_small|0|mmd_ref"]["reason"] == "fewer_than_3_references"   # reference 2 never reaches b*
    assert x7["association"]["sim1|1|discriminator_ref"]["reason"] == "low_signal"
    assert x7["depleted"]["sim1|1|discriminator_ref"]["reason"] == "low_signal"
    assert x7["depleted"]["sim1|0|discriminator_ref"]["reason"] == "unreached"
    assert x7["depleted"]["sim1|0|mmd_ref"]["d"] == pytest.approx(0.2, abs=0.01)
    assert all(x7["pooled"][f"depleted_discriminator_ref|{c}"]["declaration"] == "inconclusive" for c in (0, 1))  # 1 real dataset (AN-R12)
    assert x7["depleted"]["sim1|0|mmd_ref"]["coverage"] == {"natural": 1.0, "two_thirds": 0.7, "one_third": 0.3}
    x8 = rec["X8"]
    for t, c in itertools.product(X8_TASKS, (0, 1)):
        assert x8["slopes"][f"{t}|{c}|ref_dis|iLISI_cn"]["d"] == pytest.approx(-0.02, abs=0.002)
        assert x8["slopes"][f"{t}|{c}|pooled_dis|iLISI_cn"]["d"] == pytest.approx(0.0, abs=0.002)
        assert x8["slopes"][f"{t}|{c}|ref_strat_default|B"]["d"] == pytest.approx(0.02, abs=0.002)
    assert x8["cells"]["lung|1"]["covariates"]["0|16"] == dict(total_cells=1600, ceiling=1.0, cells_per_batch_per_minibatch=8.0,
                                                               tv_from_V2=0.0)
    # V=2: 2 batches of type ct0-heavy, so ct0 = 0.4 and the rest 0.2; V=16: each type heavy in 4 of 16 batches, all 0.25
    assert x8["cells"]["lung|1"]["covariates"]["1|16"]["tv_from_V2"] == pytest.approx(0.15)
    adj = x8["cells"]["lung|1"]["exploratory_adjusted"]   # no planted bio gap; the batch gap falls 0.02 per doubling, so
    assert adj["log2V"] - 0.02 * adj["batch_gap"] == pytest.approx(0.0, abs=0.003)   # only this combination is pinned
    x12 = rec["X12"]
    assert x12["effects"]["atac_small|1|matched_hi|A"]["reason"] == "run_without_paired_seeds"
    assert x12["pooled"]["A|1|matched_hi"]["not_evaluable"] == {"atac_small": "run_without_paired_seeds"}
    for t, c, lv in itertools.product(("atac_small", "immune", "pancreas"), (0, 1), F.FIXED):
        if (t, c, lv) == ("atac_small", 1, "matched_hi"):
            continue
        e = {k: x12["effects"][f"{t}|{c}|{lv}|{k}"]["d"] for k in ("A", "B", "C", "D", "E", "AB", "DE")}
        assert e["A"] == pytest.approx(0.06, abs=0.003) and all(abs(v) < 0.003 for k, v in e.items() if k != "A")
        assert x12["cells"][f"{t}|{c}|{lv}"]["sign"]["n_above"] == 8 and x12["cells"][f"{t}|{c}|{lv}"]["sign"]["sign_stable"]
    assert x12["pooled"]["A|1|matched_lo"]["declaration"] == "above"
    assert all(abs(v["d"]) < 0.005 for v in rec["X15"]["cells"].values())
    assert len(rec["X15"]["B_minus_bstar"]) == 2 * 2 * 3 * 2 and all(abs(v - 0.05) < 0.03 for v in rec["X15"]["B_minus_bstar"].values())
    assert set(rec["confirmatory"]) == set(F.CONFIRMATORY) and rec["n_confirmatory"] == 10
    assert rec["confirmatory"]["X8|ref_dis|iLISI_cn|1"]["declaration"] == "inconclusive"   # two real datasets cannot declare


def test_null_world(records):  # test-reason: with no planted effect no confirmatory result declares a direction
    rec = records["null"]
    assert all(v["declaration"] not in ("above", "below") for v in rec["confirmatory"].values())
    assert abs(rec["X3"]["pooled"]["slope_contrast|1|matched_lo"]["mean"]) < 0.005
    assert abs(rec["X6"]["pooled"]["divergence|0"]["mean"]) < 0.005
    assert all(abs(rec["X12"]["effects"][f"pancreas|0|matched_hi|{k}"]["d"]) < 0.003 for k in F.X12_FACTORS)


def _outcomes():
    S, Fl = synth.scores(M, world(False))
    return P.outcomes(M, S, Fl, list(P.FOLLOWUPS))


@pytest.mark.parametrize("exp, change, match", [
    ("X6", lambda o: o[~((o.experiment == "X6") & (o.cond_i == 0))], "task x decoder cells"),
    ("X6", lambda o: o.assign(seed_i=o.seed_i.where(o.experiment != "X6", 0)), "seeds"),
    ("X6", lambda o: o.assign(extra=o.extra.where(o.experiment != "X6", '{"optimizer": "critic"}')), "unknown extra keys"),
    ("X6", lambda o: o.assign(tag=o.tag.where(o.experiment != "X6", "X6_bad")), "without a lambda label"),
    ("X12", lambda o: o.assign(decoder=o.decoder.where(o.experiment != "X12", "SCVI")), "E = ABCD"),
    ("X6", lambda o: o[~((o.experiment == "X6") & (o.arm == F.X6_COPT))], "arms x steps"),
])
def test_refusals(exp, change, match):  # test-reason: the design checks refuse inputs the 2026-10-17 design does not have
    o = change(_outcomes())
    with pytest.raises(Exception, match=match):
        {"X6": lambda: F.x6(o, r4_record()["targets"]), "X12": lambda: F.x12(o, r4_record()["targets"])}[exp]()


def test_main_refuses_overwrite(records):  # test-reason: the record is written once
    with pytest.raises(Exception, match="exists"):
        F.main(records["effect_args"])
