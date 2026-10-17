"""Sinkhorn redo stage 2 (experiments/sinkhorn_redo_stage2/stage2_rules.py) on synthetic scores laid out as the stage runner
writes them (scores/<tag>.csv, failures.csv): v2 folders hold pre-fix Sinkhorn outputs under the same tags as the v3 refits.
Walks R1 (extension round, then frozen) -> R4-a1 -> R2/R3 with the real rule scripts. No fits, no real data."""
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

import synth
from synth import P

sys.path.insert(0, os.path.join(synth.ROOT, "experiments", "sinkhorn_redo_stage2"))
import stage2_rules as S2  # noqa: E402
import freeze_a1_grid  # noqa: E402


NO_FAILURES = pd.DataFrame(columns=["tag", "status", "detail"])


def write_runner_dir(d, S, F, rows):
    """Score files as run_stage.py writes them: the scorer's META columns (experiment, task, arm, lam, cond, seed) come
    from the manifest rows the file was fitted for."""
    os.makedirs(os.path.join(d, "scores"))
    S = S.join(rows.set_index("tag")[["experiment", "task", "arm", "lam", "cond", "seed"]], on="tag")
    for i in range(len(S)):
        S.iloc[[i]].to_csv(os.path.join(d, "scores", f"{S.tag.iloc[i]}.csv"), index=False)
    F.to_csv(os.path.join(d, "failures.csv"), index=False)


def early(r):                                   # corrected Sinkhorn reaches batch removal a decade earlier: R1 must extend
    return synth.world(r, center={"sinkhorn": -0.8})


@pytest.fixture(scope="module")
def stage2(tmp_path_factory):
    t = str(tmp_path_factory.mktemp("stage2"))
    co = os.path.join(t, "checkout")
    subprocess.run(["git", "clone", "-q", synth.ROOT, co], check=True)
    head = subprocess.run(["git", "-C", co, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    M = synth.pilot_rows()
    man = synth.write_tsv(M, os.path.join(t, "base.tsv"))
    a1 = M[M.experiment == "A1"]
    S, F = synth.scores(a1)                                       # v2: every arm, pre-fix Sinkhorn included
    sk_tag = a1[a1.arm == "sinkhorn"].tag.iloc[0]                 # one pre-fix Sinkhorn fit failed at v2: its failure row
    S = S[S.tag != sk_tag]                                        # must stay out of the merged failures (SI-54)
    F = pd.concat([F, pd.DataFrame([dict(tag=sk_tag, status="diverged", detail="synthetic pre-fix")])], ignore_index=True)
    write_runner_dir(os.path.join(t, "v2", "A1"), S, F, a1)
    Sn, Fn = synth.scores(a1[a1.arm == "sinkhorn"], early)        # v3: the Sinkhorn refits
    write_runner_dir(os.path.join(t, "v3", "A1_sinkhorn"), Sn, Fn, a1)
    common = dict(checkout=co, expect=head)
    return dict(t=t, M=M, man=man, n_sk=int((a1.arm == "sinkhorn").sum()) - 1, **common)   # n_sk: old Sinkhorn score files


def args(s, step, out, mans, olds, news, *extra):
    t = s["t"]
    return [step, "--checkout", s["checkout"], "--expect-commit", s["expect"], "--out-dir", os.path.join(t, out),
            *sum((["--manifest", m] for m in mans), []), *sum((["--old-dir", os.path.join(t, d)] for d in olds), []),
            *sum((["--new-dir", os.path.join(t, d)] for d in news), []), *extra]


def test_the_two_folders_together_are_refused_but_the_wrapper_selects_one_file_per_tag(stage2):
    s, t = stage2, stage2["t"]
    both = [os.path.join(t, d, "scores", f) for d in ("v2/A1", "v3/A1_sinkhorn") for f in os.listdir(os.path.join(t, d, "scores"))]
    with pytest.raises(P.PreregError, match="scored more than once"):         # trap (a) of the launch review
        freeze_a1_grid.main(["--manifest", s["man"], "--scores", *both, "--failures", os.path.join(t, "v2/A1/failures.csv"),
                             "--out-json", os.path.join(t, "naive.json"), "--out-manifest", os.path.join(t, "naive.tsv"),
                             "--extension-manifest", os.path.join(t, "naive_ext.tsv")])
    assert S2.main(args(s, "r1", "r1_pass1", [s["man"]], ["v2/A1"], ["v3/A1_sinkhorn"])) == P.EXIT_EXTEND
    rec = json.load(open(os.path.join(t, "r1_pass1", "step.json")))
    assert rec["n_old_sinkhorn_skipped"] == s["n_sk"] and rec["n_score_files"] == rec["n_rows"]
    assert rec["n_failure_rows"] == 0                                           # the v2 Sinkhorn failure row is left out
    assert rec["outputs"]["a1_extension.tsv"] == S2.sha256(os.path.join(t, "r1_pass1", "a1_extension.tsv"))
    E, _ = P.read_manifest(os.path.join(t, "r1_pass1", "a1_extension.tsv"))
    assert set(E.arm) == {"sinkhorn"} and not os.path.exists(os.path.join(t, "r1_pass1", "m.r1.tsv"))
    with pytest.raises(FileExistsError):                                       # an existing out folder is never reused
        S2.main(args(s, "r1", "r1_pass1", [s["man"]], ["v2/A1"], ["v3/A1_sinkhorn"]))


def test_ambiguous_or_unrefused_old_sinkhorn_files_stop_the_step(stage2, monkeypatch):
    s = stage2
    monkeypatch.setattr(S2, "EXCLUDE_OLD_ARMS", ())                            # mutation: old Sinkhorn files not excluded
    with pytest.raises(SystemExit, match="more than one folder"):
        S2.main(args(s, "r1", "r1_mutant", [s["man"]], ["v2/A1"], ["v3/A1_sinkhorn"]))


def test_rule_order_r1_extension_then_r4a1_then_r2r3(stage2):
    s, t = stage2, stage2["t"]
    E, _ = P.read_manifest(os.path.join(t, "r1_pass1", "a1_extension.tsv"))
    ext = synth.write_tsv(E, os.path.join(t, "ext1.tsv"))
    Se, Fe = synth.scores(E, early)
    write_runner_dir(os.path.join(t, "v3", "A1_ext_r1"), Se, Fe, E)
    # plan-fits: the Sinkhorn extension rows need fits even though a v2 folder could hold files of that name
    tags = os.path.join(t, "ext1.tags")
    assert S2.main(["plan-fits", "--manifest", ext, "--experiments", "A1", "--old-dir", os.path.join(t, "v2/A1"),
                    "--out-tags", tags]) == 0
    assert sorted(open(tags).read().split("\n")[1:-1]) == sorted(E.tag)
    mans, olds, news = [s["man"], ext], ["v2/A1"], ["v3/A1_sinkhorn", "v3/A1_ext_r1"]
    assert S2.main(args(s, "r1", "r1_pass2", mans, olds, news)) == P.EXIT_FROZEN
    r1 = os.path.join(t, "r1_pass2", "r1_record.json")
    assert json.load(open(r1))["families"]["sinkhorn|1"]["x1_grid"][0] == 0.01
    m1 = os.path.join(t, "r1_pass2", "m.r1.tsv")
    assert S2.main(args(s, "r4a1", "r4", [m1], olds, news, "--r1-record", r1)) == P.EXIT_FROZEN
    m4 = os.path.join(t, "r4", "m.r4a1.tsv")
    M4, _ = P.read_manifest(m4)
    pil = M4[M4.experiment.isin(["A2", "A3"])]
    old_pil = pil[pil.arm != "sinkhorn"]                                        # unchanged windows reuse v2 files
    S_all, _ = synth.scores(pil)
    write_runner_dir(os.path.join(t, "v2", "A2A3"), S_all, NO_FAILURES, pil)    # pre-fix: every arm, same tags
    sk = pil[pil.arm == "sinkhorn"]
    Sk, Fk = synth.scores(sk, early)
    write_runner_dir(os.path.join(t, "v3", "A2A3"), Sk, Fk, sk)
    plan = os.path.join(t, "a2a3.tags")
    assert S2.main(["plan-fits", "--manifest", m4, "--experiments", "A2", "A3", "--old-dir", os.path.join(t, "v2/A2A3"),
                    "--out-tags", plan]) == 0
    assert sorted(open(plan).read().split("\n")[1:-1]) == sorted(sk.tag) and len(old_pil) > 0   # only the Sinkhorn rows are refitted
    olds2, news2 = ["v2/A1", "v2/A2A3"], ["v3/A1_sinkhorn", "v3/A1_ext_r1", "v3/A2A3"]
    assert S2.main(args(s, "r2r3", "r23", [m4], olds2, news2, "--r4a1-record", os.path.join(t, "r4", "r4_a1_record.json"))) == 0
    rec = json.load(open(os.path.join(t, "r23", "step.json")))
    assert rec["exit_code"] == 0 and rec["n_old_sinkhorn_skipped"] == s["n_sk"] + len(sk)     # A1 and A2/A3 pre-fix files


def test_a_moved_non_sinkhorn_a3_window_never_reuses_the_score_fitted_at_the_old_lambda(stage2):
    """F1 (SI-57): A2/A3 tags keep their placeholder name when R4-a1 moves a window, so only the score's recorded lambda tells."""
    s, t = stage2, stage2["t"]
    M4, _ = P.read_manifest(os.path.join(t, "r4", "m.r4a1.tsv"))
    cell = M4[(M4.experiment == "A3") & (M4.arm == "mmd") & (M4.task == "immune") & (M4.cond == "1") & M4.tag.str.contains("_lo_")]
    assert len(cell) == 3                                                       # one window point x 3 seeds
    key = "immune|1|mmd"
    m4 = os.path.join(t, "r4", "m.r4a1.tsv")
    at_old = cell.assign(lam=P.fmt_lam(float(cell.lam.iloc[0]) / 3))            # the cell's v2 fits were made one step lower
    S_old, _ = synth.scores(cell)
    shutil.copytree(os.path.join(t, "v2", "A2A3"), os.path.join(t, "v2", "A2A3_old"))
    for tag in cell.tag:                                                        # same tag, same scores, recorded old lambda
        p = os.path.join(t, "v2", "A2A3_old", "scores", f"{tag}.csv")
        pd.read_csv(p, dtype=str).assign(lam=at_old.set_index("tag").lam[tag]).to_csv(p, index=False)
    plan, v2 = os.path.join(t, "moved.tags"), os.path.join(t, "v2/A2A3_old")
    plan_args = ["plan-fits", "--manifest", m4, "--experiments", "A2", "A3", "--old-dir", v2, "--out-tags", plan]
    assert S2.main(plan_args) == 0
    sk = M4[(M4.experiment.isin(["A2", "A3"])) & (M4.arm == "sinkhorn")]
    todo = open(plan).read().split("\n")
    assert sorted(todo[1:-1]) == sorted(set(sk.tag) | set(cell.tag)) and "3 moved" in todo[0]
    ch = os.path.join(t, "changes.json")                                        # the decision-changes record explains the move
    json.dump(dict(changed=[dict(key=f"R4a1 {key} lo", old=1.0, new=3.0)]), open(ch, "w"))
    assert S2.main(plan_args + ["--changes", ch]) == 0
    json.dump(dict(changed=[dict(key="R4a1 immune|0|mmd lo", old=1.0, new=3.0),            # another cell moved, this one did not
                            dict(key="R4a1 immune|1|sinkhorn lo", old=1.0, new=3.0)]), open(ch, "w"))
    with pytest.raises(SystemExit, match="not explained by the decision changes"):
        S2.main(plan_args + ["--changes", ch])
    # r2r3 with the stale v2 file and no refit of the moved rows is refused; with the refit the v3 file is used
    olds, news = ["v2/A1", "v2/A2A3_old"], ["v3/A1_sinkhorn", "v3/A1_ext_r1", "v3/A2A3"]
    rec = os.path.join(t, "r4", "r4_a1_record.json")
    with pytest.raises(SystemExit, match="no score file.*stale"):
        S2.main(args(s, "r2r3", "r23_stale", [m4], olds, news, "--r4a1-record", rec))
    write_runner_dir(os.path.join(t, "v3", "A2A3_moved"), S_old, NO_FAILURES, cell)       # refit at the manifest's lambda
    assert S2.main(args(s, "r2r3", "r23_moved", [m4], olds, news + ["v3/A2A3_moved"], "--r4a1-record", rec)) == 0
    step = json.load(open(os.path.join(t, "r23_moved", "step.json")))
    assert step["n_old_stale_skipped"] == 3 and step["n_score_files"] == step["n_rows"]
    write_runner_dir(os.path.join(t, "v3", "A2A3_bad"), S_old, NO_FAILURES, at_old)       # a new-dir file at the old lambda
    with pytest.raises(SystemExit, match="recorded settings differ"):
        S2.main(args(s, "r2r3", "r23_badnew", [m4], olds, news + ["v3/A2A3_bad"], "--r4a1-record", rec))


@pytest.mark.parametrize("case, match", [("duplicate_manifest", "duplicate manifest tags"), ("wrong_commit", "expected"),
                                         ("dirty_checkout", "uncommitted"), ("no_failures_csv", "does not exist"),
                                         ("failure_in_two_folders", "more than one folder"), ("no_rows", "no manifest rows"),
                                         ("missing_score", "no score file and no failure row")])
def test_inputs_that_would_make_a_record_wrong_are_refused(stage2, case, match):
    s, t = stage2, stage2["t"]
    one = ["v2/A1"], ["v3/A1_sinkhorn"]
    if case == "duplicate_manifest":
        argv = args(s, "r1", f"x_{case}", [s["man"], s["man"]], *one)
    elif case == "wrong_commit":
        argv = args(s, "r1", f"x_{case}", [s["man"]], *one)
        argv[argv.index("--expect-commit") + 1] = "0" * 40
    elif case == "no_failures_csv":
        os.makedirs(os.path.join(t, "v3", "empty"))
        argv = args(s, "r1", f"x_{case}", [s["man"]], *one[:1], ["v3/empty"])
    elif case == "missing_score":
        shutil.copytree(os.path.join(t, "v2", "A1"), os.path.join(t, "v2", "A1_cut"))
        tag = s["M"][(s["M"].experiment == "A1") & (s["M"].arm == "discriminator")].tag.iloc[0]
        os.remove(os.path.join(t, "v2", "A1_cut", "scores", f"{tag}.csv"))
        argv = args(s, "r1", f"x_{case}", [s["man"]], ["v2/A1_cut"], ["v3/A1_sinkhorn"])
    elif case == "failure_in_two_folders":
        tag = s["M"][(s["M"].experiment == "A1") & (s["M"].arm == "discriminator")].tag.iloc[0]
        for d in ("f1", "f2"):
            os.makedirs(os.path.join(t, "v3", d))
            pd.DataFrame([dict(tag=tag, status="diverged", detail="synthetic")]).to_csv(os.path.join(t, "v3", d, "failures.csv"), index=False)
        argv = args(s, "r1", f"x_{case}", [s["man"]], [], ["v3/f1", "v3/f2"])
    elif case == "no_rows":
        with pytest.raises(SystemExit, match=match):
            S2.main(["plan-fits", "--manifest", s["man"], "--experiments", "A1", "--tasks", "no_such_task", "--out-tags", os.path.join(t, "none.tags")])
        assert S2.main(["plan-fits", "--manifest", s["man"], "--experiments", "A1", "--tasks", "immune", "--out-tags", os.path.join(t, "immune.tags")]) == 0
        return
    else:
        readme = os.path.join(s["checkout"], "README.md")
        old = open(readme).read()
        open(readme, "a").write("\nuncommitted\n")
        try:
            with pytest.raises(SystemExit, match=match):
                S2.main(args(s, "r1", f"x_{case}", [s["man"]], *one))
        finally:
            open(readme, "w").write(old)
        return
    with pytest.raises(SystemExit, match=match):
        S2.main(argv)


def test_diff_records_lists_changed_decisions_and_separates_non_sinkhorn(tmp_path):
    rec = os.path.join(synth.ROOT, "docs", "prereg", "r4_a1_record.json")
    out = str(tmp_path / "same.json")
    assert S2.main(["diff-records", "--old", rec, "--new", rec, "--out", out]) == 0
    assert json.load(open(out))["n_changed"] == 0
    r = json.load(open(rec))
    k_mmd = next(k for k in r["cells"] if k.endswith("|mmd"))
    k_sk = next(k for k in r["cells"] if k.endswith("|sinkhorn"))
    r["cells"][k_mmd]["matched"] = 123.0
    r["cells"][k_sk]["hi"] = 456.0
    new = str(tmp_path / "new.json")
    json.dump(r, open(new, "w"))
    assert S2.main(["diff-records", "--old", rec, "--new", new, "--out", out]) == 0
    got = json.load(open(out))
    assert got["n_changed"] == 2 and got["n_changed_non_sinkhorn"] == 1 and got["changed"][0]["new"] in (123.0, 456.0)
    r1, r23 = (os.path.join(synth.ROOT, "docs", "prereg", n) for n in ("r1_record.json", "r2r3_record.json"))
    for p in (r1, r23):                                   # the R1 and R2/R3 record layouts are readable too
        assert S2.main(["diff-records", "--old", p, "--new", p, "--out", out]) == 0 and json.load(open(out))["n_compared"] > 0
