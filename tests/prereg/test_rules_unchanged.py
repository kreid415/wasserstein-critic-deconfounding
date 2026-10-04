"""The amendment batch of 2026-10-04 changes nothing when its cases do not occur: the rule scripts of the previous
commit (eb444fe) and of this tree, run on the same synthetic inputs (A1 -> R1 -> R4-a1 -> A2/A3 -> R2/R3 -> X1 ->
R4-x1; the builder's rows without X15, which the previous rules do not know; uniform scorer provenance), write
byte-identical resolved manifests and the same decision records. The only record difference allowed is the SI-45
field n_default_half_extension_rows = 0 of the R4-a1 record (plus git SHA and input paths). No fits, no scoring.
"""
import json
import os
import subprocess
import sys

import pytest

import synth
from synth import ROOT

PREVIOUS = "eb444fe"
VOLATILE = {"git_sha", "git_dirty", "inputs"}


def _old_scripts(tmp):
    """The previous commit's scripts/ as a git repository (the rule scripts record the git SHA of their directory)."""
    d = os.path.join(tmp, "old")
    os.makedirs(d)
    tar = subprocess.run(["git", "-C", ROOT, "archive", PREVIOUS, "scripts"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", d], input=tar, check=True)
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", PREVIOUS]):
        subprocess.run(["git", "-C", d, *cmd], check=True, capture_output=True)
    return os.path.join(d, "scripts")


def _pipeline(scripts, tmp, man, fp, s_a1):
    """Run the four rule CLIs of `scripts` in `tmp`; scores of later stages are synthesized from each stage's output."""
    os.makedirs(tmp)
    run = lambda name, *args: subprocess.run([sys.executable, os.path.join(scripts, name), *args],  # noqa: E731
                                             check=True, capture_output=True, text=True, cwd=tmp)
    j = lambda *p: os.path.join(tmp, *p)  # noqa: E731
    # outputs by run-relative names (cwd = tmp), so the records name them identically in both runs
    run("freeze_a1_grid.py", "--manifest", man, "--scores", s_a1, "--failures", fp, "--out-json", "r1.json",
        "--out-manifest", "m.r1.tsv", "--extension-manifest", "r1_ext.tsv")
    run("freeze_matched_lambda.py", "--stage", "a1", "--manifest", "m.r1.tsv", "--scores", s_a1, "--failures", fp,
        "--r1-record", "r1.json", "--out-json", "r4a1.json", "--out-manifest", "m.r4a1.tsv",
        "--extension-manifest", "a1_ext.tsv")
    M1, _ = synth.P.read_manifest(j("m.r4a1.tsv"))
    S, F = synth.scores(M1[M1.experiment.isin(["A2", "A3"])])
    synth.write_scores(S, F, j("scores"), "a2a3")
    run("decide_a2_a3.py", "--manifest", "m.r4a1.tsv", "--scores", s_a1, "scores/a2a3.csv", "--failures", fp,
        "--r4a1-record", "r4a1.json", "--out-json", "r23.json", "--out-manifest", "m.x1.tsv")
    Mx, _ = synth.P.read_manifest(j("m.x1.tsv"))
    S, F = synth.scores(Mx[Mx.experiment == "X1"])
    synth.write_scores(S, F, j("scores"), "x1")
    run("freeze_matched_lambda.py", "--stage", "x1", "--manifest", "m.x1.tsv", "--scores", "scores/x1.csv",
        "--failures", fp, "--r1-record", "r1.json", "--out-json", "r4x1.json", "--out-manifest", "m.r4x1.tsv",
        "--extension-manifest", "x1_ext.tsv")
    return tmp


def _record(path):
    rec = json.load(open(path))
    # rule records given as inputs embed their writer's git SHA, so they are compared as records, not by SHA-256
    inputs = {k: [x["sha256"] for x in v] for k, v in rec["inputs"].items() if not k.endswith("_record")}
    return {k: v for k, v in rec.items() if k not in VOLATILE}, inputs


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("unchanged"))
    M = synth.pilot_rows()
    M = M[M.experiment != "X15"]
    man = synth.write_tsv(M, os.path.join(tmp, "manifest.tsv"))
    S, F = synth.scores(M[M.experiment == "A1"])
    s_a1, fp = synth.write_scores(S, F, os.path.join(tmp, "scores"), "a1")
    old = _pipeline(_old_scripts(tmp), os.path.join(tmp, "run_old"), man, fp, s_a1)
    new = _pipeline(os.path.join(ROOT, "scripts"), os.path.join(tmp, "run_new"), man, fp, s_a1)
    return old, new


@pytest.mark.parametrize("name", ["m.r1.tsv", "m.r4a1.tsv", "m.x1.tsv", "m.r4x1.tsv"])
def test_resolved_manifests_are_byte_identical(both, name):
    old, new = both
    assert open(os.path.join(new, name), "rb").read() == open(os.path.join(old, name), "rb").read()


@pytest.mark.parametrize("name", ["r1.json", "r4a1.json", "r23.json", "r4x1.json"])
def test_decision_records_are_the_same(both, name):
    old, new = both
    (ro, io), (rn, in_) = _record(os.path.join(old, name)), _record(os.path.join(new, name))
    if name == "r4a1.json":
        assert rn.pop("n_default_half_extension_rows") == 0
    assert rn == ro
    assert in_ == io                                     # same manifests, scores, failures (SHA-256)


def test_no_extension_manifest_is_written(both):
    old, new = both
    for d in (old, new):
        assert not [f for f in os.listdir(d) if f.endswith("_ext.tsv")]
