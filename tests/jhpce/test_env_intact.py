"""cluster/jhpce/env_intact.py: detects files deleted from a conda env (fastscratch purge, 2026-10-03)."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, "cluster", "jhpce", "env_intact.py")


def make_env(tmp, stray_record_entry=True):
    env = os.path.join(tmp, "env")
    sp = os.path.join(env, "lib", "python3.11", "site-packages")
    files = ["lib/python3.11/os.py", "lib/python3.11/encodings/__init__.py", "bin/python3.11"]
    for f in files:
        os.makedirs(os.path.dirname(os.path.join(env, f)), exist_ok=True)
        open(os.path.join(env, f), "w").write("x")
    os.makedirs(os.path.join(env, "conda-meta"))
    json.dump(dict(name="python", files=files), open(os.path.join(env, "conda-meta", "python-3.11.json"), "w"))
    os.makedirs(os.path.join(sp, "pkg"))
    os.makedirs(os.path.join(sp, "pkg-1.0.dist-info"))
    open(os.path.join(sp, "pkg", "__init__.py"), "w").write("x")
    rec = ["pkg/__init__.py,sha256=x,1", "pkg/__pycache__/__init__.cpython-311.pyc,,", "pkg-1.0.dist-info/RECORD,,"]
    if stray_record_entry:
        rec.append("../../../bin/pkg3.12,sha256=x,1")          # never written, like pip's bin/pip3.12 locally
    open(os.path.join(sp, "pkg-1.0.dist-info", "RECORD"), "w").write("\n".join(rec) + "\n")
    return env


def run(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


def test_complete_env_passes(tmp_path):
    env = make_env(str(tmp_path), stray_record_entry=False)
    r = run(env)
    assert r.returncode == 0, r.stdout


def test_purged_stdlib_is_detected(tmp_path):
    env = make_env(str(tmp_path), stray_record_entry=False)
    os.remove(os.path.join(env, "lib", "python3.11", "os.py"))
    r = run(env)
    assert r.returncode == 1 and "lib/python3.11/os.py" in r.stdout


def test_baseline_accepts_benign_gaps_but_not_new_ones(tmp_path):
    env = make_env(str(tmp_path))
    assert run(env).returncode == 1                              # stray entry counts without a baseline
    assert run(env, "--write-baseline").returncode == 0
    assert run(env).returncode == 0                              # benign gap recorded at build time
    os.remove(os.path.join(env, "lib", "python3.11", "encodings", "__init__.py"))
    r = run(env)
    assert r.returncode == 1 and "encodings/__init__.py" in r.stdout and "bin/pkg3.12" not in r.stdout.split("e.g.")[-1]


def test_not_a_conda_env(tmp_path):
    assert run(str(tmp_path)).returncode == 2
