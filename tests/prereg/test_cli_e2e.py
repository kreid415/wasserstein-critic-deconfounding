"""End to end through the three rule scripts on synthetic scores: A1 -> R1 -> R4-a1 -> A2/A3 -> R2/R3 ->
X1 -> R4-x1, including an R1 extension round. No fits, no scoring."""
import json
import os

import pandas as pd
import pytest

import synth
from synth import P, BPM
import decide_a2_a3
import freeze_a1_grid
import freeze_matched_lambda


def _run_a1(tmp, world=synth.world):
    M = synth.pilot_rows()
    man = synth.write_tsv(M, os.path.join(tmp, "manifest.tsv"))
    S, F = synth.scores(M[M.experiment == "A1"], world)
    sp, fp = synth.write_scores(S, F, os.path.join(tmp, "scores"), "a1")
    return M, man, sp, fp


def _r1(tmp, mans, scores, fp):
    args = sum((["--manifest", m] for m in mans), []) + ["--scores", *scores, "--failures", fp,
            "--out-json", os.path.join(tmp, "r1.json"), "--out-manifest", os.path.join(tmp, "m.r1.tsv"),
            "--extension-manifest", os.path.join(tmp, "a1_ext.tsv")]
    return freeze_a1_grid.main(args)


def _r4(tmp, stage, man, scores, fp):
    return freeze_matched_lambda.main(["--stage", stage, "--manifest", man, "--scores", *scores, "--failures", fp,
                                       "--r1-record", os.path.join(tmp, "r1.json"),
                                       "--out-json", os.path.join(tmp, f"r4{stage}.json"),
                                       "--out-manifest", os.path.join(tmp, f"m.r4{stage}.tsv"),
                                       "--extension-manifest", os.path.join(tmp, f"{stage}_ext.tsv")])


def _decide(tmp, scores, fp):
    return decide_a2_a3.main(["--manifest", os.path.join(tmp, "m.r4a1.tsv"), "--scores", *scores, "--failures", fp,
                              "--r4a1-record", os.path.join(tmp, "r4a1.json"), "--out-json", os.path.join(tmp, "r23.json"),
                              "--out-manifest", os.path.join(tmp, "m.x1.tsv")])


def _pilots(tmp, a1_scores, fp, alt_world=synth.world):
    M1, _ = P.read_manifest(os.path.join(tmp, "m.r4a1.tsv"))
    S, F = synth.scores(M1[M1.experiment.isin(["A2", "A3"])], alt_world)
    sp, _ = synth.write_scores(S, F, os.path.join(tmp, "scores"), "a2a3")
    return [a1_scores, sp]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("prereg"))
    M, man, sp, fp = _run_a1(tmp)
    assert _r1(tmp, [man], [sp], fp) == P.EXIT_FROZEN
    assert _r4(tmp, "a1", os.path.join(tmp, "m.r1.tsv"), [sp], fp) == P.EXIT_FROZEN
    scores = _pilots(tmp, sp, fp)
    assert _decide(tmp, scores, fp) == P.EXIT_FROZEN
    Mx, _ = P.read_manifest(os.path.join(tmp, "m.x1.tsv"))
    S, F = synth.scores(Mx[Mx.experiment == "X1"])
    xs, _ = synth.write_scores(S, F, os.path.join(tmp, "scores"), "x1")
    code = _r4(tmp, "x1", os.path.join(tmp, "m.x1.tsv"), [xs], fp)
    return dict(tmp=tmp, fp=fp, a1=sp, scores=scores, x1=xs, code=code, M=M)


def test_r1_resolves_every_x1_grid_label(pipeline):
    tmp = pipeline["tmp"]
    rec = json.load(open(os.path.join(tmp, "r1.json")))
    assert rec["status"] == "frozen" and len(rec["families"]) == 8 and rec["git_sha"]
    M1, _ = P.read_manifest(os.path.join(tmp, "m.r1.tsv"))
    x1 = M1[(M1.experiment == "X1") & M1.arm.isin(BPM.ARMS)]
    assert not x1.lam.isin(BPM.FAMILY_GRID).any()
    for (arm, cond), g in x1.groupby(["arm", "cond"]):
        want = rec["families"][f"{P.FAMILY_OF[arm]}|{cond}"]["x1_grid"]
        assert sorted(set(g.lam.astype(float))) == want
    assert set(M1.tag) == set(pipeline["M"].tag)                      # tags never change
    assert rec["power"]["frequentist_can_reject"] is False


def test_r4_a1_resolves_the_pilot_windows(pipeline):
    tmp = pipeline["tmp"]
    rec = json.load(open(os.path.join(tmp, "r4a1.json")))
    M2, _ = P.read_manifest(os.path.join(tmp, "m.r4a1.tsv"))
    for r in M2[M2.experiment.isin(["A2", "A3"])].to_dict("records"):
        c = rec["cells"][f"{r['task']}|{r['cond']}|{r['arm']}"]
        assert float(r["lam"]) in (c["lo"], c["matched"], c["hi"])
    assert not M2.lam.isin(BPM.A1_MATCHED).any()
    # SI-45 does not occur here: no window neighbour lacks an A1 fit, so no default-half rows and no extension manifest
    assert rec["n_default_half_extension_rows"] == 0 and "default_half_extension_manifest" not in rec
    assert not any("edge_unresolved" in c for c in rec["cells"].values())
    assert not os.path.exists(os.path.join(tmp, "a1_ext.tsv"))


def test_r2_r3_keep_the_defaults_when_the_pilots_show_no_difference(pipeline):
    tmp = pipeline["tmp"]
    rec = json.load(open(os.path.join(tmp, "r23.json")))
    assert rec["adv_input"] == "mean" and rec["zstd"] == "0"
    assert rec["decisions"]["R2"]["n_cells"] == 12 and rec["decisions"]["R3"]["n_cells"] == 20
    assert rec["decisions"]["R2"]["need_evaluable"] == 8 and rec["decisions"]["R3"]["need_evaluable"] == 14
    assert len(rec["decisions"]["R2"]["groups"]) == 4                            # task x decoder
    M3, _ = P.read_manifest(os.path.join(tmp, "m.x1.tsv"))
    adv = M3[M3.experiment.isin(["X1"] + list(P.FOLLOWUPS)) & ~M3.arm.isin(P.NO_ADVERSARY)]
    assert set(adv.adv_input) == {"mean"} and set(adv.zstd) == {"0"}


def test_r4_x1_resolves_followups_and_no_followup_equals_an_x1_row(pipeline):
    tmp = pipeline["tmp"]
    assert pipeline["code"] == P.EXIT_FROZEN
    rec = json.load(open(os.path.join(tmp, "r4x1.json")))
    M3, _ = P.read_manifest(os.path.join(tmp, "m.x1.tsv"))
    M4, _ = P.read_manifest(os.path.join(tmp, "m.r4x1.tsv"))
    assert not M4.lam.isin(["matched_lo", "matched", "matched_hi"]).any()
    assert set(M4.tag) == set(M3.tag)                                       # nothing dropped, nothing reused
    n_fu = int(M4.experiment.isin(P.FOLLOWUPS).sum())
    assert n_fu > 0 and rec["followup_rows_checked_against_x1"] == n_fu and "reuses_x1" not in rec
    x6 = M4[(M4.experiment == "X6") & (M4.arm == "pooled_sn")]
    c = rec["cells"]
    for r in x6.to_dict("records"):
        cell = c[f"{r['task']}|{r['cond']}|pooled"]
        assert float(r["lam"]) in (cell["lo"], cell["matched"], cell["hi"])
    x15 = M4[M4.experiment == "X15"]                       # SI-44: resolved like the other follow-ups
    assert len(x15) == 84
    for r in x15[x15.arm != "none"].to_dict("records"):
        cell = c[f"{r['task']}|{r['cond']}|{r['arm']}"]
        assert float(r["lam"]) == {"matched_lo": cell["lo"], "matched_hi": cell["hi"]}[
            M3.set_index("tag").at[r["tag"], "lam"]]


V5 = os.path.join(synth.ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v5.tsv")
COPT = "discriminator_sn_critic_opt"


@pytest.mark.parametrize("decision, want", [("keep", (0.3, 1.0, 3.0)), ("shift_up", (1.0, 3.0, 10.0)),
                                            ("shift_down", (0.1, 0.3, 1.0)), (None, "pre-check decision None"),
                                            ("maybe", "pre-check decision 'maybe'")])
def test_x6_precheck_moves_only_the_optimiser_matched_arm(decision, want):  # test-reason: X6 strength pre-check (user decision 2026-10-20) on the real X6 rows
    M, _ = P.read_manifest(V5)
    x6 = M[M.experiment == "X6"]
    cells = {f"{t}|{c}|discriminator": dict(lo=0.3, matched=1.0, hi=3.0) for t in set(x6.task) for c in (0, 1)}
    cells.update({f"{t}|{c}|pooled": dict(lo=3.0, matched=10.0, hi=30.0) for t in set(x6.task) for c in (0, 1)})
    if isinstance(want, str):
        with pytest.raises(P.PreregError, match=want):
            freeze_matched_lambda.resolve(x6, dict(cells=cells), "x1", decision)
        return
    out, n = freeze_matched_lambda.resolve(x6, dict(cells=cells), "x1", decision)
    assert n == len(x6) == 486
    lo, mid, hi = (P.fmt_lam(v) for v in want)
    got = {(a, lab): set(out.lam[(x6.arm == a) & (x6.lam == lab)]) for a in set(x6.arm) for lab in ("matched_lo", "matched", "matched_hi")}
    assert {k: v for k, v in got.items() if k[0] == COPT} == {(COPT, "matched_lo"): {lo}, (COPT, "matched"): {mid}, (COPT, "matched_hi"): {hi}}
    for arm in set(x6.arm) - {COPT}:     # every other arm keeps its base arm's strengths
        base = [P.fmt_lam(v) for v in ((0.3, 1.0, 3.0) if P.BASE_ARM[arm] == "discriminator" else (3.0, 10.0, 30.0))]
        assert [got[(arm, lab)] for lab in ("matched_lo", "matched", "matched_hi")] == [{b} for b in base]


def test_x6_precheck_decision_enters_the_record(pipeline, tmp_path):  # test-reason: the CLI reads the decision file and records its value and SHA-256
    tmp = pipeline["tmp"]
    M, _ = P.read_manifest(os.path.join(tmp, "m.x1.tsv"))
    d = M[(M.experiment == "X6") & (M.arm == "discriminator")]
    copt = d.assign(arm=COPT, tag=d.tag.str.replace("_discriminator_", f"_{COPT}_"))
    man = synth.write_tsv(pd.concat([M, copt], ignore_index=True), str(tmp_path / "m.tsv"))
    dec = tmp_path / "decision.json"
    dec.write_text(json.dumps(dict(decision="shift_up")))
    args = ["--stage", "x1", "--manifest", man, "--scores", pipeline["x1"], "--failures", pipeline["fp"],
            "--r1-record", os.path.join(tmp, "r1.json"), "--out-json", str(tmp_path / "r4.json"),
            "--out-manifest", str(tmp_path / "m.r4.tsv"), "--extension-manifest", str(tmp_path / "ext.tsv")]
    with pytest.raises(P.PreregError, match="pre-check decision None"):      # no decision file: refused
        freeze_matched_lambda.main(args)
    assert freeze_matched_lambda.main(args + ["--precheck-decision", str(dec)]) == P.EXIT_FROZEN
    rec = json.load(open(tmp_path / "r4.json"))
    assert rec["precheck_decision"] == "shift_up" and rec["precheck_arm_rows"] == len(copt)
    assert rec["inputs"]["precheck_decision"][0]["sha256"] == P.sha256(str(dec))
    out, _ = P.read_manifest(str(tmp_path / "m.r4.tsv"))
    cell = rec["cells"]
    for r in out[out.arm == COPT].to_dict("records"):
        c = cell[f"{r['task']}|{r['cond']}|discriminator"]
        lab = M.set_index("tag").at[r["tag"].replace(f"_{COPT}_", "_discriminator_"), "lam"]
        assert float(r["lam"]) == P.lam_step(c[{"matched_lo": "lo", "matched": "matched", "matched_hi": "hi"}[lab]], 1)


def test_r4_x1_stops_if_a_followup_row_equals_an_x1_row(pipeline):
    M4, _ = P.read_manifest(os.path.join(pipeline["tmp"], "m.r4x1.tsv"))
    dup = M4[(M4.experiment == "X1") & (M4.arm == "pooled")].iloc[[0]].copy()
    dup["experiment"], dup["tag"] = "X6", "X6_duplicate_of_an_x1_fit"
    with pytest.raises(P.PreregError, match="equal X1 rows"):
        freeze_matched_lambda.check_no_x1_reuse(pd.concat([M4, dup], ignore_index=True))


@pytest.mark.parametrize("low_signal", [False, True])
def test_r2_switches_to_sample_when_it_wins_and_moves_the_followups(tmp_path, low_signal):
    tmp = str(tmp_path)
    M, man, sp, fp = _run_a1(tmp)
    assert _r1(tmp, [man], [sp], fp) == 0 and _r4(tmp, "a1", os.path.join(tmp, "m.r1.tsv"), [sp], fp) == 0
    if low_signal:          # SI-56 / SI-66: a flagged task x decoder stays in n and is not evaluable, so R2 keeps the mean
        p = os.path.join(tmp, "r4a1.json")
        r4 = json.load(open(p))
        r4["targets"]["immune|1"]["bstar_low_signal"] = True
        json.dump(r4, open(p, "w"))

    def better(r):
        st, B, C = synth.world(r)
        return st, B, C + (0.03 if r["experiment"] == "A2" else 0.0)
    assert _decide(tmp, _pilots(tmp, sp, fp, better), fp) == 0
    rec = json.load(open(os.path.join(tmp, "r23.json")))
    want = "mean" if low_signal else "sample"
    r2 = rec["decisions"]["R2"]
    assert rec["adv_input"] == want and r2["switch"] is not low_signal and r2["coverage"] is not low_signal
    flagged = [c for c in r2["cells"] if c["group"] == "immune|1"]
    assert r2["n_cells"] == 12 and len(flagged) == 3 and rec["zstd"] == "0"
    assert all((c["sample"]["status"] == "low_signal" and c["sample"]["reading_status"] == "bracketed") is low_signal
               for c in flagged)
    M3, _ = P.read_manifest(os.path.join(tmp, "m.x1.tsv"))
    adv = M3[M3.experiment.isin(["X1"] + list(P.FOLLOWUPS)) & ~M3.arm.isin(P.NO_ADVERSARY)]
    assert set(adv.adv_input) == {want}


def test_r2_masking_keeps_the_mean(tmp_path):
    tmp = str(tmp_path)
    M, man, sp, fp = _run_a1(tmp)
    assert _r1(tmp, [man], [sp], fp) == 0 and _r4(tmp, "a1", os.path.join(tmp, "m.r1.tsv"), [sp], fp) == 0

    def masked(r):
        st, B, C = synth.world(r)
        if r["experiment"] == "A2" and r["arm"] == "pooled" and r["task"] == "immune":
            return st, B - 0.2, C + 0.05                     # samples keep batch in the means
        return st, B, C + (0.03 if r["experiment"] == "A2" else 0.0)
    assert _decide(tmp, _pilots(tmp, sp, fp, masked), fp) == 0
    rec = json.load(open(os.path.join(tmp, "r23.json")))
    assert rec["adv_input"] == "mean" and rec["decisions"]["R2"]["masked"] == ["immune|0|pooled", "immune|1|pooled"]


def test_r1_extension_round_then_freeze(tmp_path):
    tmp = str(tmp_path)

    def early(r):
        return synth.world(r, center={"sinkhorn": -0.8})
    M, man, sp, fp = _run_a1(tmp, early)
    assert _r1(tmp, [man], [sp], fp) == P.EXIT_EXTEND
    E, _ = P.read_manifest(os.path.join(tmp, "a1_ext.tsv"))
    assert len(E) == 3 * 2 * 3 * 2 and set(E.arm) == {"sinkhorn"} and set(E.lam) == {"0.03", "0.01"}
    assert not os.path.exists(os.path.join(tmp, "m.r1.tsv"))
    S, F = synth.scores(E, early)
    ep, _ = synth.write_scores(S, F, os.path.join(tmp, "scores"), "a1_ext")
    assert _r1(tmp, [man, os.path.join(tmp, "a1_ext.tsv")], [sp, ep], fp) == P.EXIT_FROZEN
    rec = json.load(open(os.path.join(tmp, "r1.json")))
    fam = rec["families"]["sinkhorn|1"]
    assert fam["rounds"]["low"] == 1 and fam["x1_grid"][0] == 0.01


def test_missing_score_stops_the_freeze(tmp_path):
    tmp = str(tmp_path)
    M, man, sp, fp = _run_a1(tmp)
    S = pd.read_csv(sp).iloc[5:]
    S.to_csv(sp, index=False)
    with pytest.raises(P.PreregError, match="neither scored nor failed"):
        _r1(tmp, [man], [sp], fp)
