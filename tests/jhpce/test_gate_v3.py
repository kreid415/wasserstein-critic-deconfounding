"""CR2-01 launcher (cluster/jhpce/gate_command.py) and local checker (cluster/jhpce/gate_v3_check.py)."""
import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHA = "sha-under-test"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "cluster", "jhpce", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundle(tmp_path, rev):
    clone, bundle = tmp_path / f"clone_{len(rev)}", tmp_path / f"b_{len(rev)}.bundle"
    subprocess.run(["git", "clone", "-q", "--no-checkout", ROOT, str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "update-ref", "refs/heads/x", rev], check=True)
    subprocess.run(["git", "-C", str(clone), "bundle", "create", "-q", str(bundle), "refs/heads/x"], check=True)
    return str(bundle)


def test_command_pins_the_tag_commit_and_refuses_everything_else(tmp_path, capsys, monkeypatch):
    g = _load("gate_command")
    good = _bundle(tmp_path, g.GATE_SHA)
    g.main(["--bundle", good, "--run-timeout-h", "576"])
    plan = json.loads(capsys.readouterr().out)
    cmd = plan["submit"]["command"]
    assert cmd.splitlines()[:3] == ["#SBATCH --partition=gpu", "#SBATCH --account=jhpce", "#SBATCH --gres=gpu:l40s:1"]
    assert cmd.count(g.GATE_SHA) == 2 and "status --porcelain" in cmd and "gate_job.sh" in cmd
    assert plan["submit"]["run_timeout_s"] == 576 * 3600
    other = _bundle(tmp_path, "HEAD~1")                        # a bundle that lacks the tag commit
    for argv, msg in (([ "--bundle", good, "--expected-sha", "0" * 40], "--expected-sha must be"),
                      (["--bundle", other], "no head at"), (["--bundle", good, "--walltime", "2h"], "D-HH:MM:SS"),
                      (["--bundle", str(tmp_path / "none")], "not found")):
        with pytest.raises(SystemExit, match=msg):
            g.main(argv)
    monkeypatch.setattr(g, "TAG", "no-such-tag")                # tag absent in this clone
    with pytest.raises(SystemExit, match="missing here"):
        g.main(["--bundle", good])


def _latents(d, tags, sha=SHA, gpu="NVIDIA L40S", z=None):
    os.makedirs(d, exist_ok=True)
    for i, t in enumerate(tags):
        zz = np.full((4, 10), float(i), dtype=np.float32) if z is None else z(i)
        np.savez(os.path.join(d, f"{t}.npz"), z=zz, config=json.dumps(dict(git_sha=sha, git_dirty=False, gpu=gpu, row=dict(tag=t))))


@pytest.mark.parametrize("new_kw, old_z, code", [
    ({}, None, 0),                                                    # all equal; the Sinkhorn row differs on purpose
    ({"sha": "other"}, None, 1),                                      # written by another commit
    ({"gpu": "NVIDIA A100"}, None, 1),                                # wrong GPU model
    ({"z": lambda i: np.full((4, 10), np.nan, dtype=np.float32)}, None, 1),
    ({}, lambda i: np.full((4, 10), i + 1.0, dtype=np.float32), 2),   # a non-Sinkhorn latent changed
])
def test_check_exit_codes(tmp_path, new_kw, old_z, code):
    c = _load("gate_v3_check")
    tags = ["a", "b", "s"]
    (tmp_path / "m.tsv").write_text("tag\tarm\na\tmmd\nb\tnone\ns\tsinkhorn\n")
    _latents(tmp_path / "new", tags, **new_kw)
    old = tmp_path / "old"
    _latents(old, tags, z=old_z)
    np.savez(old / "s.npz", z=np.ones((4, 10), np.float32) * 9, config="{}")       # Sinkhorn old latent is never read
    argv = ["--new", str(tmp_path / "new"), "--old", str(old), "--manifest", str(tmp_path / "m.tsv"),
            "--expected-sha", SHA, "--out", str(tmp_path / "o.csv")]
    with pytest.raises(SystemExit) as e:
        c.main(argv)
    assert e.value.code == code
