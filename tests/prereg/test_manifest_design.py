"""Hard requirements of the pilot design in scripts/build_paper_manifest.py (docs/PREREG.md sections 0, 2, 3)."""
import json
import os
import subprocess
import sys

import pytest

import compare_a1_pilot5 as C
import synth
from synth import P, BPM, ROOT


@pytest.fixture(scope="module")
def M():
    return synth.pilot_rows("pilot")


def test_a1_seeds_are_disjoint_from_x1(M):
    a1 = set(M.seed[M.experiment.isin(["A1", "A2", "A3"])].astype(int))
    x1 = set(M.seed[M.experiment == "X1"].astype(int))
    assert a1 == {100, 101, 102} and x1 == {0, 1, 2, 3, 4} and not a1 & x1
    P.check_seed_separation(M)


def test_followups_use_fresh_seeds_and_never_equal_an_x1_row(M):
    for e in P.FOLLOWUPS:
        assert set(M.seed[M.experiment == e].astype(int)) == set(BPM.FOLLOWUP_SEEDS) == {10, 11, 12}, e
    assert set(M.seed[M.experiment == "X13"].astype(int)) == {0, 1, 2, 3, 4}      # not lambda-matched
    wild = ("tag", "experiment", "lam", "adv_input", "zstd")                      # placeholders resolved later
    key = lambda r: "|".join(f"{c}={r[c]}" for c in BPM.COLS if c not in wild)   # noqa: E731
    x1 = {key(r) for r in M[M.experiment == "X1"].to_dict("records")}
    fu = M[M.experiment.isin(P.FOLLOWUPS)].to_dict("records")
    assert fu and not any(key(r) in x1 for r in fu)
    R = BPM.finalize(BPM.build(BPM.BACKBONES["stock"], "pilot", 3, 5, 8))
    assert not [r["tag"] for r in R if r["experiment"] in P.FOLLOWUPS and r["reuses_x1"]]


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
    assert len(a2) == 2 * 2 * 3 * 3 * 3 and len(a3) == 2 * 2 * 3 * 5 * 3      # tasks x decoders x seeds x arms x window
    assert set(a2.adv_input) == {"sample"} and set(a2.zstd) == {"0"}
    assert set(a3.adv_input) == {"mean"} and set(a3.zstd) == {"1"}
    assert set(a2.lam) | set(a3.lam) == set(BPM.A1_MATCHED)
    assert set(a2.cond) == set(a3.cond) == {"0", "1"}                          # both decoders (SI-27)
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


FIVE = ["immune_hum_mou", "atac_large", "lung", "sim2", "pancreas"]


def _a1_manifest(tmp_path, tasks):
    out = tmp_path / f"a1_{len(tasks)}_{tasks[0]}.tsv"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_paper_manifest.py"), "--backbone", "stock",
                    "--design", "pilot", "--uncond-seeds", "5", "--a1-tasks", *tasks,      # as M: no --bary-iter
                    "--out", str(out)], check=True, capture_output=True)
    return out


def _r1_record(old_paths):
    """A synthetic R1 record whose a1_grid per family x decoder is the base grid plus A1_EXT_R1."""
    fams = {f"{fam}|{c}": dict(arms=list(arms), cond=c, rounds={"low": 0, "high": 0},
                               a1_grid=sorted({float(v) for v in BPM.LAMBDA_GRID} |
                                              {float(lam) for a, k, ls in BPM.A1_EXT_R1 if a in arms and k == c for lam in ls}))
            for fam, arms in P.FAMILIES.items() for c in (0, 1)}
    return dict(families=fams, inputs=dict(manifest=[dict(path=str(p), sha256=C.sha256(p)) for p in old_paths]))


@pytest.mark.parametrize("tasks", [list(BPM.A1_TASKS), FIVE])
def test_a1_tasks_flag_extends_a1_with_the_r1_grid_edge_strengths(M, tmp_path, tasks):
    new_path = _a1_manifest(tmp_path, tasks)
    new = C.read_a1(new_path)
    assert {r["experiment"] for r in new} == {"A1"} and sorted({r["task"] for r in new}) == sorted(tasks)
    assert len(new) == 378 * len(tasks)          # (1 none + 6 arms x 10) x 6 + 12 grid-edge rows, per task
    ext = [r for r in new if r["arm"] != "none" and float(r["lam"]) not in set(map(float, BPM.LAMBDA_GRID))]
    assert len(ext) == 12 * len(tasks)
    assert {(r["arm"], int(r["cond"]), float(r["lam"])) for r in ext} == \
        {(a, c, float(lam)) for a, c, ls in BPM.A1_EXT_R1 for lam in ls}
    if tasks == BPM.A1_TASKS:       # the flag leaves the full design's A1 rows as they are, tag included
        assert set(M.tag[M.experiment == "A1"]) <= {r["tag"] for r in new}
        return
    old_path = _a1_manifest(tmp_path, list(BPM.A1_TASKS))
    old = C.read_a1(old_path)
    assert C.compare(old, new) == []
    (tmp_path / "rec.json").write_text(json.dumps(_r1_record([old_path])))
    assert C.main(["--new", str(new_path), "--old", str(old_path), "--r1-record", str(tmp_path / "rec.json")]) == 0
    bad_n_latent = [dict(r, n_latent="30") if i == 5 else r for i, r in enumerate(new)]
    bad_epochs = [dict(r, max_epochs="399") if i == 7 else r for i, r in enumerate(new)]
    assert len(C.compare(old, bad_n_latent)) == 1 and "n_latent" in C.compare(old, bad_n_latent)[0]
    assert len(C.compare(old, bad_epochs)) == 1 and "max_epochs" in C.compare(old, bad_epochs)[0]
    assert "differ in (arm, cond, lam, seed)" in C.compare(old, new[1:])[0]
    assert "both old and new" in C.compare(old, old)[0]
    tampered = tmp_path / "tampered.tsv"                         # one new row with another batch size: main fails, says which
    tampered.write_text(new_path.read_text().replace("\tzinb\t128\t", "\tzinb\t256\t", 1))
    assert C.main(["--new", str(tampered), "--old", str(old_path), "--r1-record", str(tmp_path / "rec.json")]) == 1
    rec = _r1_record([old_path])
    rec["families"]["js|0"]["a1_grid"].remove(0.01)             # the record no longer lists a strength the rows hold
    (tmp_path / "rec.json").write_text(json.dumps(rec))
    assert C.main(["--new", str(new_path), "--old", str(old_path), "--r1-record", str(tmp_path / "rec.json")]) == 1
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_paper_manifest.py"), "--backbone", "stock",
                        "--a1-tasks", "lung", "--out", str(tmp_path / "x.tsv")], capture_output=True)
    assert r.returncode == 2 and b"--a1-tasks needs --design pilot" in r.stderr


def test_shared_design_keeps_x1_matched_pilots():
    S = synth.pilot_rows("shared")
    assert "A1" not in set(S.experiment)
    a2 = S[S.experiment == "A2"]
    assert set(a2.adv_input) == {"sample"} and set(a2.lam) == {"matched_lo", "matched", "matched_hi"}
    x1 = S[(S.experiment == "X1") & S.arm.isin(BPM.ARMS)]
    assert set(x1.adv_input) == {"mean"} and set(x1.lam) == {str(v) for v in BPM.LAMBDA_GRID}
