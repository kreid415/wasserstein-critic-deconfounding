"""Known-answer, planted-effect and null tests for scripts/analysis_followups.py (docs/ANALYSIS_SPEC_FOLLOWUPS.md).

Synthetic inputs only (SI-78): no X1, X13 or follow-up score is read. The synthetic design follows the 2026-10-17
decisions (SI-93..SI-98) and the names assumed in the specification's section 1.4. Every planted value is stated
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


V4 = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v4.tsv")


def design():
    """The X6 rows of manifest v4 (followup-build; identical to manifest v5's), with each lambda label replaced by a synthetic value."""
    M, _ = P.read_manifest(V4)
    M = M[M.experiment == "X6"].reset_index(drop=True)
    return M.assign(lam=M.lam.map(LAM))


def world(effects):
    """(status, B, C) per row. Planted (when `effects`): W1+SN at 1 step keeps 0.05 more biology and JS+SN with
    the critic's optimiser 0.02 less at every lambda (so the divergence contrast is +0.07, the own-optimiser one +0.05).
    Seed effects are shared by every arm of a task x decoder (pairing). One failed run; one curve never reaches b*."""
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
            B = 0.3 + sb                                        # never reaches b* = 0.5
        C = 0.70 - 0.05 * math.log10(lam) + sc + e
        if effects and arm == "pooled_sn" and int(r["n_critic"]) == 1:
            C += 0.05
        if effects and arm == F.X6_COPT:
            C -= 0.02
        return "ok", B, C
    return fn


M = design()
FAILED = {M.tag[(M.task == "atac_small") & (M.cond == "1") & (M.arm == "discriminator") & (M.lam == "3") & (M.seed == "11")].item()}


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
    return ["--manifest", man, "--scores", sp, "--failures", fp, "--r4-record", r4, "--out-json", os.path.join(tmp, "followups.json")]


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


def test_known_answers():  # test-reason: jackknife, Holm and bio window against hand values
    vals = {10: 1.0, 11: 2.0, 12: 4.0}
    r = F.loso(lambda keep: F.mean(vals, keep))
    assert math.isclose(r["d"], 7 / 3) and math.isclose(r["s"], np.std([1, 2, 4], ddof=1) / math.sqrt(3))   # jackknife of a mean
    assert F.loso(lambda keep: None if 10 not in keep else 1.0)["reason"] == "readings"
    pts = [(1.0, {}, {10: 0.70, 11: 0.72, 12: 0.74}, 0), (10.0, {}, {10: 0.60, 11: 0.61, 12: 0.62}, 0)]
    assert F.bio_window(pts) == dict(width=0.0, n_points=1, bound=pytest.approx(0.72 - 4.302653 * 0.02 / math.sqrt(3), abs=1e-6))
    assert F.bio_window(pts[:1] + [(10.0, {}, {10: 0.80}, 2)])["reason"] == "best point has fewer than 2 runs"
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
    assert x6["contrast"]["pancreas|0|js_r1"]["reason"] == "unreached" and x6["cells"]["pancreas|0"]["js_r1"]["status"] == "unreached_low"
    assert x6["contrast"]["atac_small|1|js_sn"]["m"] == 3                                 # the failed run leaves every reading
    assert set(rec["confirmatory"]) == set(F.CONFIRMATORY) == {"X6|divergence|0", "X6|divergence|1"}
    assert rec["n_confirmatory"] == rec["holm_m"] == 2
    assert rec["confirmatory"]["X6|divergence|1"]["declaration"] == "not_decidable_1_2_real_datasets"


def test_null_world(records):  # test-reason: with no planted effect no confirmatory result declares a direction
    rec = records["null"]
    assert all(v["declaration"] not in ("above", "below") for v in rec["confirmatory"].values())
    assert abs(rec["X6"]["pooled"]["divergence|0"]["mean"]) < 0.005


def test_v4_rows_enumerated():  # test-reason: the analysis reads every v4 X6 row under the level it expects
    got = F.prepare(_outcomes(), "X6").groupby(["cond_i", "level"]).size().to_dict()
    for c in (0, 1):    # hand count from the design (tasks x arms x seeds per lambda label)
        assert all(got[(c, lv)] == 3 * 9 * 3 for lv in F.LEVELS)


def _outcomes():
    S, Fl = synth.scores(M, world(False))
    return P.outcomes(M, S, Fl, ["X6"])


@pytest.mark.parametrize("change, match", [
    (lambda o: o[o.cond_i != 0], "task x decoder cells"),
    (lambda o: o.assign(seed_i=0), "seeds"),
    (lambda o: o.assign(extra='{"optimizer": "critic"}'), "unknown extra keys"),
    (lambda o: o.assign(tag="X6_bad"), "without a lambda label"),
    (lambda o: o[o.arm != F.X6_COPT], "arms x steps"),
])
def test_refusals(change, match):  # test-reason: the design checks refuse inputs the 2026-10-17 design does not have
    with pytest.raises(Exception, match=match):
        F.x6(change(_outcomes()), r4_record()["targets"])


def test_main_refuses_overwrite(records):  # test-reason: the record is written once
    with pytest.raises(Exception, match="exists"):
        F.main(records["effect_args"])
