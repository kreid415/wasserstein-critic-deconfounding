"""X1 cluster builder (experiments/X1_jhpce/x1_jhpce_command.py), final review I008: the harness clock counts from
submission, so a job K places down its dependency chain gets K + 1 queue allowances and walltimes; a chain position
is given exactly with the id of the job it waits for."""
import hashlib
import importlib.util
import json
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _builder():
    path = os.path.join(ROOT, "experiments", "X1_jhpce", "x1_jhpce_command.py")
    spec = importlib.util.spec_from_file_location("x1_jhpce_command", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chained_job_clock_covers_every_job_up_its_chain(tmp_path, capsys, monkeypatch):
    b = _builder()
    assert b.DESIGN_SHA == ""                                                  # pre-redo design file: refused until stage 2 sets it
    with pytest.raises(SystemExit, match="DESIGN_SHA is not set"):
        b.main(["--task", "pancreas", "--bundle", "none"])
    good = hashlib.sha256(open(os.path.join(ROOT, b.MANIFEST), "rb").read()).hexdigest()
    monkeypatch.setattr(b, "DESIGN_SHA", good)
    for extra in (["--chain-position", "2"], ["--after-slurm-job", "444"], ["--chain-position", "-1"]):
        with pytest.raises(SystemExit, match="exactly when"):
            b.main(["--task", "pancreas", "--bundle", "none", *extra])
    with pytest.raises(SystemExit, match="prereg-tier12-v3"):                  # the v2 commit is no longer accepted (SI-54)
        b.main(["--task", "pancreas", "--bundle", "none", "--expected-sha", "c88cce3c569e9de47f2d65c6794ca3890637786a"])
    assert b.TAG_SHA == "10a1ca5874fbcdc033b148525b5e71d866306bdb"
    clone, bundle = tmp_path / "clone", tmp_path / "wcd.bundle"      # a bundle of the tag, as rehearse_x1_job.sh makes
    subprocess.run(["git", "clone", "-q", "--no-checkout", ROOT, str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "update-ref", "refs/heads/x", b.TAG_SHA], check=True)
    subprocess.run(["git", "-C", str(clone), "bundle", "create", "-q", str(bundle), "refs/heads/x"], check=True)
    monkeypatch.setattr(b, "DESIGN_SHA", "0" * 64)                              # a design file other than the pinned one
    with pytest.raises(SystemExit, match="not the pinned DESIGN_SHA"):
        b.main(["--task", "pancreas", "--bundle", str(bundle)])
    monkeypatch.setattr(b, "DESIGN_SHA", good)
    for pos, extra in ((0, []), (2, ["--after-slurm-job", "444"])):
        b.main(["--task", "pancreas", "--bundle", str(bundle), "--chain-position", str(pos), *extra])
        plan = json.loads(capsys.readouterr().out)
        assert plan["chain_position"] == pos and plan["submit"]["run_timeout_s"] == (pos + 1) * (72 + 120) * 3600
