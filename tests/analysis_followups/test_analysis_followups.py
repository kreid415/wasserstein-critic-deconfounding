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

import anndata as ad
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests", "prereg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analysis_followups as F  # noqa: E402
import prereg_rules as P  # noqa: E402
import build_paper_manifest as BPM  # noqa: E402
import synth  # noqa: E402

LAM = {"matched_lo": "1", "matched": "3", "matched_hi": "10", "0": "0"}
BSTAR = 0.5
LOW = {"immune|1", "sim1|1"}             # flagged low-signal in the synthetic R4 record
N_TYPES = 10
X7_REFS = {"pancreas": [10, 9, 8, 7, 6, 5, 4, 3, 2], "sim1": [10, 8, 6, 5, 4, 3], "atac_small": [10, 6, 3]}  # covered types
X8_TASKS = ("immune_hum_mou", "lung", "sim2")


V4 = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v4.tsv")
SHARE = {"none": 1.0, "two_thirds": 2 / 3, "one_third": 1 / 3}   # nominal kept share of the depleted reference's types


def design():
    """The follow-up rows of manifest v4 (followup-build), with each lambda label replaced by a synthetic value."""
    M, _ = P.read_manifest(V4)
    M = M[M.experiment.isin(P.FOLLOWUPS)].reset_index(drop=True)
    return M.assign(lam=M.lam.map(LAM))


def coverage_of(task, ref):
    return X7_REFS[task][ref] / N_TYPES


def prepped_obs():
    """Synthetic prepped obs: X7 tasks with the coverage pattern of X7_REFS (a covered type has >= 20 cells, an uncovered one
    3); X8 tasks with 17 batches of cells_per_batch + 10 cells whose composition depends on the batch."""
    obs = {}
    for t, ks in X7_REFS.items():
        rows = [(f"b{i}", f"ct{j}") for i, k in enumerate(ks) for j in range(N_TYPES) for _ in range((20 + 5 * j + i) if j < k else 3)]
        obs[t] = pd.DataFrame(rows, columns=["batch", "celltype"])
    for t in X8_TASKS:
        per = BPM.X8_TOTAL[t] // 16
        obs[t] = pd.DataFrame([(f"b{b:02d}", f"ct{i % (b % 3 + 2)}") for b in range(17) for i in range(per + 10)],
                              columns=["batch", "celltype"])
    return obs


def world(effects):
    """(status, B, C) per row. Planted (when `effects`): X3 the reference critic loses 0.10 more biology per 100%
    depletion than the other arms, which the oracle weights undo; X6 W1+SN at 1 step keeps 0.05 more biology at every
    lambda; X7 the reference critic's biology rises 0.2 per unit coverage share (natural references) and by 0.2 x the
    kept share of the depleted reference's types (1, 2/3, 1/3 for levels none, two_thirds, one_third), the comparators' not at all; X8 the reference critic's batch score falls 0.02 per doubling of V (default sampler only);
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
            lv = F.x7_level(t, r["reference"], ex.get("subsample", {})) if exp == "X7" else ""
            if (exp == "X6" and t == "pancreas" and c == 0 and arm == "discriminator_r1") or \
                    (exp == "X7" and t == "atac_small" and c == 0 and arm == "reference" and r["reference"] == "2") or \
                    (exp == "X7" and t == "sim1" and c == 0 and arm == "discriminator_ref" and lv == "one_third"):
                B = 0.3 + sb                                    # never reaches b* = 0.5
            C = 0.70 - 0.05 * math.log10(lam) + sc + e
            if exp == "X6" and effects and arm == "pooled_sn" and int(r["n_critic"]) == 1:
                C += 0.05
            if exp == "X7" and effects and arm == "reference":
                level = F.x7_level(t, r["reference"], ex.get("subsample", {}))
                C += 0.2 * (coverage_of(t, int(r["reference"])) if level == "natural" else SHARE[level])
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
FAILED = {M.tag[(M.experiment == "X6") & (M.task == "atac_small") & (M.cond == "1") & (M.arm == "discriminator")
                & (M.lam == "3") & (M.seed == "11")].item(),
          M.tag[(M.experiment == "X3") & (M.task == "pancreas") & (M.cond == "1") & (M.arm == "mmd") & (M.lam == "10")
                & (M.seed == "12") & M.extra.str.contains('"deplete_pct": 100,') & ~M.extra.str.contains('"iw"')].item()} | \
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
    obs = prepped_obs()
    if effects:      # the effect run builds the count table from prepped files; the null run is given the table
        src = ["--prepped-dir", tmp]
        for t, o in obs.items():
            ad.AnnData(obs=o.astype("category").set_axis([str(i) for i in range(len(o))])).write_h5ad(os.path.join(tmp, f"{t}__scib.h5ad"))
    else:
        cp = os.path.join(tmp, "counts.csv")
        F.celltype_counts(obs, M).to_csv(cp, index=False)
        src = ["--celltype-counts", cp]
    return ["--manifest", man, "--scores", sp, "--failures", fp, "--r4-record", r4, *src, "--out-json", os.path.join(tmp, "followups.json")]


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
    counts = F.celltype_counts(prepped_obs(), M)
    cov = F.coverage(counts, "pancreas", "full", "b1")
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
    assert x6["pooled"]["divergence|1"]["declaration"] == "not_decidable_1_2_real_datasets"     # 2 real datasets (user, round 2)
    assert x6["contrast"]["pancreas|0|js_r1"]["reason"] == "unreached" and x6["cells"]["pancreas|0"]["js_r1"]["status"] == "unreached_low"
    assert x6["contrast"]["atac_small|1|js_sn"]["m"] == 3                                 # the failed run leaves every reading
    x7 = rec["X7"]
    for c in (0, 1):
        assert x7["association"][f"pancreas|{c}|discriminator_ref"]["d"] == pytest.approx(0.2, abs=0.01)
        # each depleted level against the same-size control: 0.2 x (2/3 - 1) and 0.2 x (1/3 - 1)
        assert x7["depleted"][f"pancreas|{c}|two_thirds_discriminator_ref"]["d"] == pytest.approx(-0.2 / 3, abs=0.01)
        assert x7["depleted"][f"pancreas|{c}|one_third_discriminator_ref"]["d"] == pytest.approx(-0.4 / 3, abs=0.01)
        assert x7["pooled"][f"association_discriminator_ref|{c}"]["n"] == 2 - (c == 0)
    assert x7["association"]["atac_small|1|mmd_ref"]["d"] == pytest.approx(0.2, abs=0.01)
    assert x7["association"]["atac_small|0|mmd_ref"]["reason"] == "fewer_than_3_references"   # reference 2 never reaches b*
    assert x7["association"]["sim1|1|discriminator_ref"]["reason"] == "low_signal"
    assert x7["depleted"]["sim1|1|one_third_discriminator_ref"]["reason"] == "low_signal"
    assert x7["depleted"]["sim1|0|one_third_discriminator_ref"]["reason"] == "unreached"
    assert x7["depleted"]["sim1|0|two_thirds_mmd_ref"]["d"] == pytest.approx(-0.2 / 3, abs=0.01)
    assert {k: x7["depleted"]["pancreas|0|one_third_mmd_ref"][k] for k in ("reference", "n_reference", "types_removed")} == \
        dict(reference="inDrop1", n_reference=1572, types_removed=9)
    p = x7["pooled"]["depleted_one_third_discriminator_ref|0"]
    assert p["declaration"] == "not_decidable_1_2_real_datasets" and p["units"] == ["pancreas"] and "sim1" in p["sign_table"]
    x8 = rec["X8"]
    for t, c in itertools.product(X8_TASKS, (0, 1)):
        assert x8["slopes"][f"{t}|{c}|ref_dis|iLISI_cn"]["d"] == pytest.approx(-0.02, abs=0.002)
        assert x8["slopes"][f"{t}|{c}|pooled_dis|iLISI_cn"]["d"] == pytest.approx(0.0, abs=0.002)
        assert x8["slopes"][f"{t}|{c}|ref_strat_default|B"]["d"] == pytest.approx(0.02, abs=0.002)
    cv = x8["cells"]["lung|1"]["covariates"]
    assert {k: cv["0|16"][k] for k in ("total_cells", "ceiling", "cells_per_batch_per_minibatch")} == \
        dict(total_cells=16 * 375, ceiling=1.0, cells_per_batch_per_minibatch=8.0) and cv["0|2"]["tv_from_V2"] == 0.0
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
    assert all(v["x"] == 0 and v["n"] == 3 for v in rec["X15"]["collapse"].values()) and len(rec["X15"]["collapse"]) == 24
    assert x12["cells"]["atac_small|1|matched_hi"]["runs"]["5"]["n_failed"] == {"reference": 3, "discriminator": 0}
    assert x7["cells"]["atac_small|0"]["points"]["reference|2"]["n_failed"] == 0
    assert len(rec["X15"]["B_minus_bstar"]) == 2 * 2 * 3 * 2 and all(abs(v - 0.05) < 0.03 for v in rec["X15"]["B_minus_bstar"].values())
    assert set(rec["confirmatory"]) == set(F.CONFIRMATORY) and rec["n_confirmatory"] == 12
    assert rec["confirmatory"]["X8|ref_dis|iLISI_cn|1"]["declaration"] == "not_decidable_1_2_real_datasets"
    assert os.path.exists(records["effect_args"][-1] + ".celltype_counts.csv")       # built from the prepped files


def test_null_world(records):  # test-reason: with no planted effect no confirmatory result declares a direction
    rec = records["null"]
    assert all(v["declaration"] not in ("above", "below") for v in rec["confirmatory"].values())
    assert abs(rec["X3"]["pooled"]["slope_contrast|1|matched_lo"]["mean"]) < 0.005
    assert abs(rec["X6"]["pooled"]["divergence|0"]["mean"]) < 0.005
    assert all(abs(rec["X12"]["effects"][f"pancreas|0|matched_hi|{k}"]["d"]) < 0.003 for k in F.X12_FACTORS)


def test_v4_rows_enumerated():  # test-reason: the analysis reads every v4 follow-up row under the level it expects
    o = _outcomes()
    got = {e: F.prepare(o, e).groupby(["cond_i", "level"]).size().to_dict() for e in P.FOLLOWUPS}
    for c in (0, 1):    # hand counts from the design (tasks x points x arms x seeds)
        assert got["X3"][(c, "0")] == 4 * 5 * 3 and got["X3"][(c, "matched_lo")] == got["X3"][(c, "matched_hi")] == 4 * (5 + 3) * 4 * 3
        assert all(got["X6"][(c, lv)] == 3 * 9 * 3 for lv in F.LEVELS)
        assert all(got["X7"][(c, lv)] == (9 + 6 + 3) * 4 * 3 + 2 * 3 * 4 * 3 for lv in F.LEVELS)
        assert got["X8"][(c, "0")] == 3 * 4 * 2 * 3 and got["X8"][(c, "matched")] == 3 * 4 * 2 * 3 * 6
        assert got["X12"][(c, "matched_lo")] == got["X12"][(c, "matched_hi")] == 3 * 16 * 2 * 3
    assert got["X15"] == {(1, "0"): 12, (1, "matched_lo"): 36, (1, "matched_hi"): 36}
    x7 = F.prepare(o, "X7")
    assert x7.groupby("ref_level").size().to_dict() == dict(natural=2 * 18 * 4 * 3 * 3, **{lv: 2 * 2 * 4 * 3 * 3 for lv in ("none", "two_thirds", "one_third")})
    assert F.prepare(o, "X3").groupby(["cond_i", "iw"]).size().to_dict()[(0, True)] == 4 * 3 * 4 * 2 * 3   # weighted rows, cond 0


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
    ("X7", lambda o: o.assign(extra=o.extra.str.replace('"types": []', '"types": ["alpha"]', regex=False)), "match no level"),
    ("X7", lambda o: o.assign(extra=o.extra.str.replace("reference_composition", "reference_depletion")), "unknown subsample kinds"),
])
def test_refusals(exp, change, match):  # test-reason: the design checks refuse inputs the 2026-10-17 design does not have
    o = change(_outcomes())
    with pytest.raises(Exception, match=match):
        {"X6": lambda: F.x6(o, r4_record()["targets"]), "X12": lambda: F.x12(o, r4_record()["targets"]),
         "X7": lambda: F.prepare(o, "X7")}[exp]()


def test_main_refuses_overwrite(records):  # test-reason: the record is written once
    with pytest.raises(Exception, match="exists"):
        F.main(records["effect_args"])
