"""Hard requirements of the pilot design in scripts/build_paper_manifest.py (docs/PREREG.md sections 0, 2, 3)."""
import pytest

import synth
from synth import P, BPM


@pytest.fixture(scope="module")
def M():
    return synth.pilot_rows("pilot")


def test_a1_seeds_are_disjoint_from_x1(M):
    a1 = set(M.seed[M.experiment.isin(["A1", "A2", "A3"])].astype(int))
    x1 = set(M.seed[M.experiment == "X1"].astype(int))
    assert a1 == {100, 101, 102} and x1 == {0, 1, 2, 3, 4} and not a1 & x1
    P.check_seed_separation(M)


def test_no_a1_fit_is_an_x1_fit(M):
    a1 = {P.settings_key(r) for r in M[M.experiment == "A1"].to_dict("records")}
    x1 = {P.settings_key(r) for r in M[M.experiment == "X1"].to_dict("records")}
    assert a1 and x1 and not a1 & x1


def test_a1_design_is_complete(M):
    a1 = M[M.experiment == "A1"]
    assert sorted(set(a1.task)) == sorted(BPM.A1_TASKS) and set(a1.cond) == {"0", "1"}
    none = a1[a1.arm == "none"]
    assert len(none) == 3 * 2 * 3 and set(none.lam) == {"0"}
    adv = a1[a1.arm != "none"]
    assert len(adv) == 3 * 2 * 3 * len(BPM.ARMS) * len(BPM.LAMBDA_GRID)
    assert set(adv.adv_input) == {"mean"} and set(adv.zstd) == {"0"}


def test_a2_a3_emit_only_the_new_halves_and_reuse_a1(M):
    a2, a3 = M[M.experiment == "A2"], M[M.experiment == "A3"]
    assert len(a2) == 2 * 3 * 3 * 3 and len(a3) == 2 * 3 * 5 * 3
    assert set(a2.adv_input) == {"sample"} and set(a2.zstd) == {"0"}
    assert set(a3.adv_input) == {"mean"} and set(a3.zstd) == {"1"}
    assert set(a2.lam) | set(a3.lam) == set(BPM.A1_MATCHED) and set(a2.cond) | set(a3.cond) == {"1"}
    a1 = {P.settings_key(r) for r in M[M.experiment == "A1"].to_dict("records")}
    # whatever A1 grid value R4-a1 picks, the default half of every A2/A3 row is an A1 fit
    for r in a2.to_dict("records") + a3.to_dict("records"):
        for lam in BPM.LAMBDA_GRID:
            d = dict(r, lam=P.fmt_lam(lam), adv_input="mean", zstd="0")
            assert P.settings_key(d) in a1, r["tag"]


def test_pilots_are_scheduled_before_x1(M):
    exps = M.experiment.tolist()
    last_pilot = max(i for i, e in enumerate(exps) if e in ("A1", "A2", "A3"))
    assert last_pilot < exps.index("X1")


def test_x1_rows_hold_placeholders_until_the_rules_run(M):
    x1 = M[M.experiment == "X1"]
    adv = x1[x1.arm.isin(BPM.ARMS)]
    assert set(adv.lam) == set(BPM.FAMILY_GRID)
    assert set(adv.adv_input) == {BPM.ADV_INPUT_PENDING} and set(adv.zstd) == {BPM.ZSTD_PENDING}
    inert = x1[x1.arm.isin(["none", "scvi_adv"])]
    assert set(inert.adv_input) == {"mean"} and set(inert.zstd) == {"0"}
    for v in ("g1", "a1_matched"):                       # fit_paper_config.py: float(lam) must fail
        with pytest.raises(ValueError):
            float(v)
    with pytest.raises(ValueError):                      # fit_paper_config.py: int(zstd) must fail
        int(BPM.ZSTD_PENDING)


def test_shared_design_keeps_x1_matched_pilots():
    S = synth.pilot_rows("shared")
    assert "A1" not in set(S.experiment)
    a2 = S[S.experiment == "A2"]
    assert set(a2.adv_input) == {"sample"} and set(a2.lam) == {"matched_lo", "matched", "matched_hi"}
    x1 = S[(S.experiment == "X1") & S.arm.isin(BPM.ARMS)]
    assert set(x1.adv_input) == {"mean"} and set(x1.lam) == {str(v) for v in BPM.LAMBDA_GRID}
