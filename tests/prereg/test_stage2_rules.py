"""Sinkhorn redo stage 2 (experiments/sinkhorn_redo_stage2/stage2_rules.py) on synthetic scores laid out as the stage runner
writes them (scores/<tag>.csv, failures.csv): v2 folders hold pre-fix Sinkhorn outputs under the same tags as the v3 refits.
Walks R1 (extension round, then frozen) -> R4-a1 -> R2/R3 with the real rule scripts. No fits, no real data."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

import synth
from synth import P

sys.path.insert(0, os.path.join(synth.ROOT, "experiments", "sinkhorn_redo_stage2"))
import stage2_rules as S2  # noqa: E402
import freeze_a1_grid  # noqa: E402


def write_runner_dir(d, S, F=None):
    os.makedirs(os.path.join(d, "scores"))
    for i in range(len(S)):
        S.iloc[[i]].to_csv(os.path.join(d, "scores", f"{S.tag.iloc[i]}.csv"), index=False)
    (F if F is not None else pd.DataFrame(columns=["tag", "status", "detail"])).to_csv(os.path.join(d, "failures.csv"), index=False)


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
    write_runner_dir(os.path.join(t, "v2", "A1"), S, F)
    Sn, Fn = synth.scores(a1[a1.arm == "sinkhorn"], early)        # v3: the Sinkhorn refits
    write_runner_dir(os.path.join(t, "v3", "A1_sinkhorn"), Sn, Fn)
    common = dict(checkout=co, expect=head)
    return dict(t=t, M=M, man=man, n_sk=int((a1.arm == "sinkhorn").sum()), **common)


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
    write_runner_dir(os.path.join(t, "v3", "A1_ext_r1"), Se, Fe)
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
    write_runner_dir(os.path.join(t, "v2", "A2A3"), S_all)                      # pre-fix: every arm, same tags
    sk = pil[pil.arm == "sinkhorn"]
    Sk, Fk = synth.scores(sk, early)
    write_runner_dir(os.path.join(t, "v3", "A2A3"), Sk, Fk)
    plan = os.path.join(t, "a2a3.tags")
    assert S2.main(["plan-fits", "--manifest", m4, "--experiments", "A2", "A3", "--old-dir", os.path.join(t, "v2/A2A3"),
                    "--out-tags", plan]) == 0
    assert sorted(open(plan).read().split("\n")[1:-1]) == sorted(sk.tag) and len(old_pil) > 0   # only the Sinkhorn rows are refitted
    olds2, news2 = ["v2/A1", "v2/A2A3"], ["v3/A1_sinkhorn", "v3/A1_ext_r1", "v3/A2A3"]
    assert S2.main(args(s, "r2r3", "r23", [m4], olds2, news2, "--r4a1-record", os.path.join(t, "r4", "r4_a1_record.json"))) == 0
    rec = json.load(open(os.path.join(t, "r23", "step.json")))
    assert rec["exit_code"] == 0 and rec["n_old_sinkhorn_skipped"] == s["n_sk"] + len(sk)     # A1 and A2/A3 pre-fix files
