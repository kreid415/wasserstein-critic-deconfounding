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

COPT, SN = "discriminator_sn_critic_opt", "pooled_sn"       # the two X6 arms moved by their own strength pre-check decision


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


def _r4(tmp, stage, man, scores, fp, extra=()):
    return freeze_matched_lambda.main(["--stage", stage, "--manifest", man, "--scores", *scores, "--failures", fp,
                                       "--r1-record", os.path.join(tmp, "r1.json"),
                                       "--out-json", os.path.join(tmp, f"r4{stage}.json"),
                                       "--out-manifest", os.path.join(tmp, f"m.r4{stage}.tsv"),
                                       "--extension-manifest", os.path.join(tmp, f"{stage}_ext.tsv"), *extra])


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
    keep = os.path.join(tmp, "keep.json")        # the manifest's X6 pooled_sn rows at n_critic 1 need a pre-check decision (SI-107)
    json.dump(dict(decisions={COPT: "keep", SN: "keep"}), open(keep, "w"))
    code = _r4(tmp, "x1", os.path.join(tmp, "m.x1.tsv"), [xs], fp, ["--precheck-decision", keep])
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
LABELS = ("matched_lo", "matched", "matched_hi")
KEEP = {COPT: "keep", SN: "keep"}


@pytest.mark.parametrize("decisions, want", [
    (KEEP, {COPT: (0.3, 1.0, 3.0), SN: (3.0, 10.0, 30.0)}),
    ({COPT: "shift_up", SN: "shift_down"}, {COPT: (1.0, 3.0, 10.0), SN: (1.0, 3.0, 10.0)}),    # each arm takes its own decision
    ({COPT: "shift_down", SN: "shift_up"}, {COPT: (0.1, 0.3, 1.0), SN: (10.0, 30.0, 100.0)}),
    (None, "pre-check decision None"), ("keep", "pre-check decision 'keep'"),                   # not a {arm: decision} dict
    ({COPT: "maybe", SN: "keep"}, f"{COPT}: X6 strength pre-check decision 'maybe'"),
    ({COPT: "keep"}, f"{SN}: X6 strength pre-check decision None")])
def test_x6_precheck_moves_each_checked_arm_by_its_own_decision(decisions, want):  # test-reason: X6 strength pre-check (SI-98, SI-107) on the real X6 rows
    M, _ = P.read_manifest(V5)
    x6 = M[M.experiment == "X6"]
    cells = {f"{t}|{c}|discriminator": dict(lo=0.3, matched=1.0, hi=3.0) for t in set(x6.task) for c in (0, 1)}
    cells.update({f"{t}|{c}|pooled": dict(lo=3.0, matched=10.0, hi=30.0) for t in set(x6.task) for c in (0, 1)})
    if isinstance(want, str):
        with pytest.raises(P.PreregError, match=want):
            freeze_matched_lambda.resolve(x6, dict(cells=cells), "x1", decisions)
        return
    out, n = freeze_matched_lambda.resolve(x6, dict(cells=cells), "x1", decisions)
    assert n == len(x6) == 486
    for arm, k in set(zip(x6.arm, x6.n_critic)):
        moved = arm == COPT or (arm == SN and k == "1")           # pooled_sn at n_critic 5 and every other arm keep their base arm's strengths
        strengths = want[arm] if moved else (0.3, 1.0, 3.0) if P.BASE_ARM[arm] == "discriminator" else (3.0, 10.0, 30.0)
        got = [set(out.lam[(x6.arm == arm) & (x6.n_critic == k) & (x6.lam == lab)]) for lab in LABELS]
        assert got == [{P.fmt_lam(v)} for v in strengths], (arm, k)
    free = x6[~x6.arm.isin([COPT, SN]) | ((x6.arm == SN) & (x6.n_critic == "5"))]     # no row of a checked arm: no decision needed
    assert freeze_matched_lambda.resolve(free, dict(cells=cells), "x1", None)[1] == len(free)


def test_x6_precheck_decisions_enter_the_record(pipeline, tmp_path):  # test-reason: the CLI reads the decisions file and records the decisions, the rows moved and its SHA-256
    tmp = pipeline["tmp"]
    M, _ = P.read_manifest(os.path.join(tmp, "m.x1.tsv"))
    d = M[(M.experiment == "X6") & (M.arm == "discriminator")]
    copt = d.assign(arm=COPT, tag=d.tag.str.replace("_discriminator_", f"_{COPT}_"))
    man = synth.write_tsv(pd.concat([M, copt], ignore_index=True), str(tmp_path / "m.tsv"))
    decisions = {COPT: "shift_up", SN: "shift_down"}
    dec = tmp_path / "decision.json"
    dec.write_text(json.dumps(dict(decisions=decisions)))
    args = ["--stage", "x1", "--manifest", man, "--scores", pipeline["x1"], "--failures", pipeline["fp"],
            "--r1-record", os.path.join(tmp, "r1.json"), "--out-json", str(tmp_path / "r4.json"),
            "--out-manifest", str(tmp_path / "m.r4.tsv"), "--extension-manifest", str(tmp_path / "ext.tsv")]
    with pytest.raises(P.PreregError, match="pre-check decision None"):      # no decision file: refused
        freeze_matched_lambda.main(args)
    for bad, arm in ((dict(decision="shift_up"), COPT), (dict(decisions={COPT: "keep"}), SN)):    # old format; an arm missing
        dec_bad = tmp_path / "bad.json"
        dec_bad.write_text(json.dumps(bad))
        with pytest.raises(P.PreregError, match=f"{arm}: X6 strength pre-check decision None"):
            freeze_matched_lambda.main(args + ["--precheck-decision", str(dec_bad)])
    assert freeze_matched_lambda.main(args + ["--precheck-decision", str(dec)]) == P.EXIT_FROZEN
    rec = json.load(open(tmp_path / "r4.json"))
    sn1 = (M.arm == SN) & (M.n_critic == "1")
    assert rec["precheck_decisions"] == decisions and rec["precheck_arm_rows"] == {COPT: len(copt), SN: int(sn1.sum())}
    assert "precheck_decision" not in rec and 0 < sn1.sum() < (M.arm == SN).sum()
    assert rec["inputs"]["precheck_decision"][0]["sha256"] == P.sha256(str(dec))
    out, _ = P.read_manifest(str(tmp_path / "m.r4.tsv"))
    labels, label_of = M.set_index("tag").lam, {"matched_lo": "lo", "matched": "matched", "matched_hi": "hi"}
    for r in out[out.arm.isin([COPT, SN])].to_dict("records"):
        base, step = ("discriminator", 1) if r["arm"] == COPT else ("pooled", -1 if r["n_critic"] == "1" else 0)
        v = rec["cells"][f"{r['task']}|{r['cond']}|{base}"][label_of[labels[r["tag"].replace(f"_{COPT}_", "_discriminator_")]]]
        assert float(r["lam"]) == (P.lam_step(v, step) if step else v)


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
