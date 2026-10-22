"""X1 cluster builder (experiments/X1_jhpce/x1_jhpce_command.py), final review I008: the harness clock counts from
submission, so a job K places down its dependency chain gets K + 1 queue allowances and walltimes; a chain position
is given exactly with the id of the job it waits for."""
import importlib.util
import json
import os
import subprocess

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name):
    path = os.path.join(ROOT, "experiments", "X1_jhpce", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _builder():
    return _load("x1_jhpce_command")


def test_run_lists_follow_the_manifests(tmp_path, monkeypatch):
    """make_x1_tags.py: the committed pilot5 run lists are what it derives from the pilot5 manifest; a manifest short of a
    row is refused; the X1 job still derives its 1,800 rows (filler tags given) and still needs them."""
    m = _load("make_x1_tags")
    monkeypatch.chdir(ROOT)       # the run lists name the manifest by its repo-relative path
    man = "manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv"
    m.main(["--job", "pilot5", "--manifest", man, "--check", "--out-dir", "experiments/pilot5_jhpce/tags"])
    assert len(m.derive(man, None, "pilot5")) == 1890
    short = tmp_path / "short.tsv"
    short.write_text("\n".join(open(man).read().splitlines()[:-1]) + "\n")
    with pytest.raises(SystemExit, match="expected 378 per task"):
        m.derive(str(short), None, "pilot5")
    x1 = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v3.x1.tsv")
    M = pd.read_csv(x1, sep="\t", comment="#", dtype=str, keep_default_na=False)
    filler = tmp_path / "filler.tags"
    filler.write_text("\n".join(M.tag[(M.experiment == "X1") & M.arm.isin(m.RULE_FREE)]) + "\n")
    assert len(m.derive(x1, str(filler))) == 1800
    with pytest.raises(SystemExit):
        m.main(["--manifest", x1, "--out-dir", str(tmp_path)])


@pytest.mark.parametrize("job, n_tags, experiments, slots", [("x1", 360, "X1", None), ("pilot5", 378, "A1", "5")])
def test_chained_job_clock_covers_every_job_up_its_chain(tmp_path, capsys, job, n_tags, experiments, slots):
    b = _builder()
    sha = b.JOBS[job]["sha"]
    with pytest.raises(SystemExit, match="must be"):      # each job kind pins its own tag commit
        b.main(["--job", job, "--task", "pancreas", "--bundle", "none", "--expected-sha", b.JOBS["pilot5" if job == "x1" else "x1"]["sha"]])
    for extra in (["--chain-position", "2"], ["--after-slurm-job", "444"], ["--chain-position", "-1"]):
        with pytest.raises(SystemExit, match="exactly when"):
            b.main(["--job", job, "--task", "pancreas", "--bundle", "none", *extra])
    clone, bundle = tmp_path / "clone", tmp_path / "wcd.bundle"      # a bundle of the tag, as rehearse_x1_job.sh makes
    subprocess.run(["git", "clone", "-q", "--no-checkout", ROOT, str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "update-ref", "refs/heads/x", sha], check=True)
    subprocess.run(["git", "-C", str(clone), "bundle", "create", "-q", str(bundle), "refs/heads/x"], check=True)
    with pytest.raises(SystemExit, match="has no head"):          # the other job kind's commit is not in this bundle
        b.main(["--job", "pilot5" if job == "x1" else "x1", "--task", "pancreas", "--bundle", str(bundle)])
    for pos, extra in ((0, []), (2, ["--after-slurm-job", "444"])):
        b.main(["--job", job, "--task", "pancreas", "--bundle", str(bundle), "--chain-position", str(pos), *extra])
        plan = json.loads(capsys.readouterr().out)
        assert plan["chain_position"] == pos and plan["submit"]["run_timeout_s"] == (pos + 1) * (72 + 120) * 3600
    args = plan["job_args"]
    assert plan["job"] == f"{job}_pancreas" and plan["expected_sha"] == sha and plan["n_tags"] == n_tags
    assert args[args.index("--experiments") + 1] == experiments and args[args.index("--stage") + 1] == b.JOBS[job]["stage"]
    assert (args[args.index("--slots") + 1] if "--slots" in args else None) == slots
