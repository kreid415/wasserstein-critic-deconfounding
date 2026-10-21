"""Known-answer, planted-effect and null tests for scripts/analysis_followups.py (docs/ANALYSIS_SPEC_FOLLOWUPS.md).

Synthetic inputs only (SI-78): no X1, X13 or follow-up score is read. The synthetic design follows the 2026-10-17
decisions (SI-93..SI-98) and the names in the specification's section 1.9. Every planted value is stated
in `world`; the expected estimates follow from it by hand.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
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
LOW = {"immune|1"}                       # flagged low-signal in the synthetic R4 record


V5 = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v5.tsv")


def design():
    """The X6 rows of manifest v5, with each lambda label replaced by a synthetic value."""
    M, _ = P.read_manifest(V5)
    M = M[M.experiment == "X6"].reset_index(drop=True)
    return M.assign(lam=M.lam.map(LAM))


def world(effects):
    """(status, B, C) per row. Planted (when `effects`): W1+SN at 1 step keeps 0.05 more biology and JS+SN with
    the critic's optimiser 0.02 less at every lambda (so the divergence contrast is +0.07, the own-optimiser one +0.05).
    Seed effects are shared by every arm of a task x decoder (pairing). Failures: one run, one whole point, one whole
    arm (FAILED). Curves that miss b*: pancreas|0 R1 stays below it, immune|0 pooled at 10 steps stays above it."""
    def fn(r):
        t, c, s, arm = r["task"], int(r["cond"]), int(r["seed"]), r["arm"]
        lam = float(r["lam"])
        sb = synth._noise(f"b{t}|{c}|{s}", 0.01)
        sc = synth._noise(f"c{t}|{c}|{s}", 0.01)
        e = synth._noise("e" + r["tag"], 0.001)
        if r["tag"] in FAILED:
            return "diverged", None, None
        B = 0.4 + 0.2 * (math.log10(lam) + 0.5) / 1.5 + sb + e
        if t == "pancreas" and c == 0 and arm == "discriminator_r1":
            B = 0.3 + sb                                        # never reaches b* = 0.5 from below
        if t == "immune" and c == 0 and arm == "pooled" and int(r["n_critic"]) == 10:
            B = 0.8 + sb                                        # ... and this one from above
        C =0.70 - 0.05 * math.log10(lam) + sc + e
        if effects and arm == "pooled_sn" and int(r["n_critic"]) == 1:
            C += 0.05
        if effects and arm == F.X6_COPT:
            C -= 0.02
        return "ok", B, C
    return fn


M = design()
FAILED = (
    {M.tag[(M.task == "atac_small") & (M.cond == "1") & (M.arm == "discriminator") & (M.lam == "3") & (M.seed == "11")].item()}   # 1 run
    | set(M.tag[(M.task == "pancreas") & (M.cond == "1") & (M.arm == "pooled") & (M.n_critic == "1") & (M.lam == "3")])         # 1 point (3 runs)
    | set(M.tag[(M.task == "atac_small") & (M.cond == "0") & (M.arm == "pooled_sn") & (M.n_critic == "5")]))                    # 1 arm (9 runs)


def r4_record(low=LOW):
    tasks = {t for e in F.EXPECTED.values() for t, _ in e}
    return dict(rule="R4", stage="x1", status="frozen",
                targets={f"{t}|{c}": dict(b0=0.4, bstar=BSTAR, b_common=0.6, bstar_low_signal=f"{t}|{c}" in low)
                         for t in tasks for c in (0, 1)})


def write_inputs(tmp, effects, low=LOW):
    man = synth.write_tsv(M, os.path.join(tmp, "manifest.tsv"))
    sp, fp = synth.write_scores(*synth.scores(M, world(effects)), os.path.join(tmp, "scores"), "fu")
    r4 = os.path.join(tmp, "r4.json")
    json.dump(r4_record(low), open(r4, "w"))
    return ["--manifest", man, "--scores", sp, "--failures", fp, "--r4-record", r4, "--out-json", os.path.join(tmp, "followups.json")]


ALL_COND0_LOW = LOW | {f"{t}|0" for t in ("atac_small", "immune", "pancreas")}


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    out = {}
    for name, eff, low in (("effect", True, LOW), ("null", False, LOW), ("lowsig", True, ALL_COND0_LOW)):
        tmp = str(tmp_path_factory.mktemp(name))
        args = write_inputs(tmp, eff, low)
        assert F.main(args) == 0
        out[name] = json.load(open(args[-1]))
        out[name + "_args"] = args
    return out


def test_known_answers():  # test-reason: jackknife and Holm against hand values
    vals = {10: 1.0, 11: 2.0, 12: 4.0}
    r = F.loso(lambda keep: F.mean(vals, keep))
    assert math.isclose(r["d"], 7 / 3) and math.isclose(r["s"], np.std([1, 2, 4], ddof=1) / math.sqrt(3))   # jackknife of a mean
    assert F.loso(lambda keep: None if 10 not in keep else 1.0)["reason"] == "readings"
    assert F.holm_all([0.01, None]) == [pytest.approx(0.02), None]    # Holm over 2 even when 1 is missing


def test_planted_effects(records):  # test-reason: X6's endpoint recovers its planted value and declaration
    rec = records["effect"]
    x6 = rec["X6"]
    assert x6["pooled"]["divergence|0"]["mean"] == pytest.approx(0.07, abs=0.01)
    # the confirmatory contrast is pooled_sn minus discriminator_sn_critic_opt (not minus discriminator_sn)
    assert x6["contrast"]["pancreas|0|divergence"]["d"] == pytest.approx(0.07, abs=0.003)
    assert x6["contrast"]["pancreas|0|divergence_own_optimiser"]["d"] == pytest.approx(0.05, abs=0.003)
    assert x6["pooled"]["divergence|0"]["declaration"] == "above" and x6["pooled"]["js_optimiser|0"]["declaration"] != "above"
    assert x6["pooled"]["divergence|1"]["not_evaluable"] == {"immune": "low_signal"} and x6["pooled"]["divergence|1"]["n"] == 2
    assert x6["pooled"]["divergence|1"]["declaration"] == "not_decidable_1_2_real_datasets"     # 2 real datasets (user, round 2)
    # a contrast that cannot be read at b* carries the failing arm's own status as its reason (never a generic one)
    assert x6["contrast"]["pancreas|0|js_r1"]["reason"] == "unreached_low" and x6["cells"]["pancreas|0"]["js_r1"]["status"] == "unreached_low"
    assert x6["contrast"]["immune|0|w1_steps_10"]["reason"] == "unreached_high" and x6["cells"]["immune|0"]["w1_gp_10"]["status"] == "unreached_high"
    assert x6["contrast"]["atac_small|1|js_sn"]["m"] == 3                                 # the failed run leaves every reading
    # a point whose 3 runs all failed is skipped: b* is read across it from the two other points (pooled_sn minus pooled, +0.05)
    assert x6["cells"]["pancreas|1"]["w1_gp_1"]["matched"]["n_fail"] == 3 and x6["contrast"]["pancreas|1|w1_sn_1"]["m"] == 3
    assert x6["contrast"]["pancreas|1|w1_sn_1"]["d"] == pytest.approx(0.05, abs=0.005)
    # an arm with no successful run in a cell does not stop the analysis: status no_points, frozen window rule, reason no_points
    dead = x6["cells"]["atac_small|0"]["w1_sn_5"]
    assert dead["status"] == "no_points" and dead["window"]["status"] == "no_points"
    assert x6["contrast"]["atac_small|0|w1_sn_5"]["reason"] == "no_points" and x6["contrast"]["atac_small|0|w1_sn_5"]["fixed_strength"]["dC"] is None
    assert x6["cells"]["pancreas|0"]["js"]["window"]["status"] == "ok" and x6["cells"]["pancreas|0"]["js"]["window"]["n_points"] >= 1
    # the fixed-strength report: labelled, each arm at its own `matched` strength (here the same synthetic one for every arm)
    fs = x6["contrast"]["pancreas|0|divergence"]["fixed_strength"]
    assert fs["label"] == F.FIXED_LABEL == "fixed-strength, not matched batch removal"
    assert fs["dC"] == pytest.approx(0.07, abs=0.005) and abs(fs["dB"]) < 0.005
    assert x6["contrast"]["pancreas|0|js_r1"]["fixed_strength"]["dB"] < -0.1             # R1 batch score sits far below the plain JS one
    assert set(rec["confirmatory"]) == set(F.CONFIRMATORY) == {"X6|divergence|0", "X6|divergence|1"}
    assert rec["n_confirmatory"] == rec["holm_m"] == 2
    assert rec["confirmatory"]["X6|divergence|1"]["declaration"] == "not_decidable_1_2_real_datasets"
    # Wilcoxon p: 3 datasets all positive 2/2^3, 2 datasets 2/2^2; Holm over the 2: 2 x 0.25 = 0.5 and 0.5
    assert [rec["confirmatory"][k]["wilcoxon_p"] for k in F.CONFIRMATORY] == pytest.approx([0.25, 0.5])
    assert [rec["confirmatory"][k]["holm_p"] for k in F.CONFIRMATORY] == pytest.approx([0.5, 0.5])


def test_pool_without_an_evaluable_dataset(records):  # test-reason: 0 evaluable real datasets in a confirmatory pool, run end to end
    rec = records["lowsig"]
    c0, c1 = rec["confirmatory"]["X6|divergence|0"], rec["confirmatory"]["X6|divergence|1"]
    pool = rec["X6"]["pooled"]["divergence|0"]
    assert pool["n"] == 0 and pool["declaration"] == "not_evaluable" and set(pool["not_evaluable"].values()) == {"low_signal"}
    assert len(pool["not_evaluable"]) == 3 and c0["declaration"] == "not_evaluable" and c0["wilcoxon_p"] is None and c0["holm_p"] is None
    assert rec["holm_m"] == 2 and c1["wilcoxon_p"] == pytest.approx(0.5) and c1["holm_p"] == pytest.approx(1.0)   # the missing p enters as 1
    # a low-signal cell is still reported at fixed strength
    assert rec["X6"]["contrast"]["pancreas|0|divergence"]["reason"] == "low_signal"
    assert rec["X6"]["contrast"]["pancreas|0|divergence"]["fixed_strength"]["dC"] == pytest.approx(0.07, abs=0.005)


def test_null_world(records):  # test-reason: with no planted effect no confirmatory result declares a direction
    rec = records["null"]
    assert all(v["declaration"] not in ("above", "below") for v in rec["confirmatory"].values())
    assert abs(rec["X6"]["pooled"]["divergence|0"]["mean"]) < 0.005


def test_v5_rows_enumerated():  # test-reason: the analysis reads every manifest-v5 X6 row under the level it expects
    got = F.prepare(_outcomes(), "X6").groupby(["cond_i", "level"]).size().to_dict()
    for c in (0, 1):    # hand count from the design (tasks x arms x seeds per lambda label)
        assert all(got[(c, lv)] == 3 * 9 * 3 for lv in F.LEVELS)


def _outcomes():
    S, Fl = synth.scores(M, world(False))
    return P.outcomes(M, S, Fl, ["X6"])


@pytest.mark.parametrize("change, match", [
    (lambda o: o[o.cond_i != 0], "task x decoder cells"),
    (lambda o: o.assign(seed_i=0), "seeds"),
    (lambda o: o.assign(extra='{"optimizer": "critic"}'), "`extra` must be"),
    (lambda o: o.assign(extra=o.extra.where(o.arm != "discriminator_r1", '{"r1_gamma": 3}')), "`extra` must be"),   # wrong gamma
    (lambda o: o.assign(extra='{"r1_gamma": 10}'), "`extra` must be"),                                           # gamma on every arm
    (lambda o: o.assign(tag=o.tag.str.replace("_lmatched_c", "_l0_c")), "without a lambda label"),               # a lambda = 0 tag
    (lambda o: o.assign(tag="X6_bad"), "without a lambda label"),
    (lambda o: o[o.arm != F.X6_COPT], "arms x steps"),
    (lambda o: o[~((o.arm == F.X6_COPT) & (o.task == "atac_small") & (o.cond_i == 0) & o.tag.str.contains("_lmatched_hi_c"))],
     "missing or holding several values"),                                                                         # a missing point
])
def test_refusals(change, match):  # test-reason: the design checks refuse inputs the 2026-10-17 design does not have
    with pytest.raises(Exception, match=match):
        F.x6(change(_outcomes()), r4_record()["targets"])


def test_main_refuses_overwrite(records):  # test-reason: the record is written once
    with pytest.raises(Exception, match="exists"):
        F.main(records["effect_args"])


def test_only_x6_rows_are_accepted():  # test-reason: blinding, a score or failure row of another experiment is refused before any use
    S, Fl = synth.scores(M, world(False))
    F.only_x6(M, S, Fl)                                                         # X6 rows only: accepted
    with pytest.raises(Exception, match="1 score or failure rows are not X6"):
        F.only_x6(M, S.assign(tag=S.tag.where(S.index != 0, "X1_some_row")), Fl)
    with pytest.raises(Exception, match="1 score or failure rows are not X6"):
        F.only_x6(M, S, Fl.assign(tag=Fl.tag.where(Fl.index != 0, "X1_some_row")))
