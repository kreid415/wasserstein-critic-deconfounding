"""Fixed-seed JHPCE array wrapper (cluster/jhpce/score_array_fixed_seed.sh, SI-115). The real preload library is built on JHPCE and
pinned by SHA-256, so the success path runs on a copy of the script pinned to an ordinary shared library (the one the test's own
interpreter has loaded); the repository's score_array.sh is a stub that writes a one-row score file.
    python -m pytest -q tests/jhpce/test_score_array_fixed_seed.py
"""
import csv
import hashlib
import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WRAPPER = os.path.join(ROOT, "cluster", "jhpce", "score_array_fixed_seed.sh")
STUB = ('#!/bin/bash\nmkdir -p "$3/scores" "$3/logs"\n'
        # STUB_FAIL=race: the first run dies on the numba cache index race, as score_array.sh does (exit 10); STUB_FAIL=other: always exit 10
        'if [ "${STUB_FAIL:-}" = other ] || { [ "${STUB_FAIL:-}" = race ] && [ ! -e "$3/raced" ]; }; then touch "$3/raced"; '
        'echo "${STUB_LOG:-FileExistsError: [Errno 17] File exists: x.nbi}" > "$3/logs/t1.log"; exit 10; fi\n'
        '[ -s "$3/scores/t1.csv" ] || printf "tag,iLISI\\nt1,0.5\\n" > "$3/scores/t1.csv"\n')


def run(script, tmp_path, lib, **env_extra):
    repo = tmp_path / "repo" / "cluster" / "jhpce"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "env.sh").write_text("SCORE_PY=python3\n")
    (repo / "score_array.sh").write_text(STUB)
    (tmp_path / "list").write_text("t1\tnpz\tpre\t{}\n")
    env = dict(os.environ, SCORER_REPO=str(tmp_path / "repo"), FIXED_SEED_LIB=str(lib), SLURM_ARRAY_TASK_ID="1", NUMBA_RACE_PAUSE_S="1", **env_extra)
    return subprocess.run(["bash", str(script), "sha", str(tmp_path / "list"), str(tmp_path / "out")], env=env, capture_output=True, text=True)


def test_wrapper_refuses_a_missing_or_other_preload_library(tmp_path):
    other = tmp_path / "fixed_random_device.so"
    other.write_bytes(b"not the pinned library")
    for lib in (tmp_path / "absent.so", other):
        r = run(WRAPPER, tmp_path, lib)
        assert r.returncode == 9 and "not the pinned preload library" in r.stderr


def test_wrapper_scores_adds_the_provenance_columns_once_and_checks_a_kept_file(tmp_path):
    lib = next((m.group(1) for m in re.finditer(r"\s(/\S+/(?:libm|libc)\.so[.\d]*)$", open("/proc/self/maps").read(), re.M)), None)
    if lib is None:
        pytest.skip("no shared library to preload in this interpreter")      # pragma: no cover  # depends on the interpreter's libraries
    text = open(WRAPPER).read()
    text = re.sub(r"^LIB_SHA256=\S+", f"LIB_SHA256={hashlib.sha256(open(lib, 'rb').read()).hexdigest()}", text, flags=re.M)
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(text.replace("fixed_random_device.so", os.path.basename(lib)))
    assert run(wrapper, tmp_path, lib).returncode == 0
    path = tmp_path / "out" / "scores" / "t1.csv"
    row = list(csv.DictReader(open(path)))[0]
    assert (row["tag"], row["iLISI"], row["score_site"], row["lisi_random_device"]) == ("t1", "0.5", "jhpce", "0xFFFFFFFF")
    assert len(row["lisi_preload_sha256"]) == 64
    assert run(wrapper, tmp_path, lib).returncode == 0 and list(csv.DictReader(open(path)))[0] == row          # a kept file: same columns
    path.write_text(path.read_text().replace("0xFFFFFFFF", "0x12345678"))
    r = run(wrapper, tmp_path, lib)
    assert r.returncode != 0 and "provenance columns differ" in r.stderr
    path.unlink()                                              # a cold-cache collision is run again once; any other failure is not
    assert run(wrapper, tmp_path, lib, STUB_FAIL="race").returncode == 0 and (tmp_path / "out" / "logs" / "t1.log.race1").exists()
    r = run(wrapper, tmp_path, lib, STUB_FAIL="other", STUB_LOG="some other error")
    assert r.returncode == 10 and "numba cache index race" not in r.stderr
    r = run(wrapper, tmp_path, lib, STUB_FAIL="other")         # the race text every time: two more tries, then the task fails with the scorer's code
    assert r.returncode == 10 and r.stderr.count("numba cache index race") == 2
