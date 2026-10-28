"""Workstation backend of the chain (cluster/jhpce/chain_ws.py): what is copied, what is refused, when scoring waits, the end.
The remote folder is a local directory (--host ''), the scorer is a stub that writes one score file per tag of TAGS_FILE.
    python -m pytest -q tests/jhpce/test_chain_ws.py
"""
import json
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cluster", "jhpce"))
import chain_ws as cw  # noqa: E402

TAGS = [f"A1_atac_large_none_l0_c0_s{100 + i}_{i:08x}" for i in range(4)]
STUB = "#!/bin/bash\nmkdir -p $OUT/scores; for t in $(cat $TAGS_FILE); do echo ok > $OUT/scores/$t.csv; done\n"


def make_remote(root, tags, age_min=60, failed=()):
    for t in tags:
        fs = cw.files_for(t, failed=t in failed)
        for f in fs:
            p = os.path.join(root, f)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(f"{f}\n")
            os.utime(p, (time.time() - age_min * 60,) * 2)


@pytest.fixture
def env(tmp_path):
    remote, mirror, state = (tmp_path / n for n in ("remote", "mirror", "state"))
    mirror.mkdir()
    (tmp_path / "tags").write_text("# run list\n" + "\n".join(TAGS) + "\n")
    stub = tmp_path / "score.sh"
    stub.write_text(STUB.replace("$OUT", str(mirror)))
    argv = ["--tags-file", str(tmp_path / "tags"), "--mirror", str(mirror), "--state-dir", str(state), "--host", "",
            "--remote-root", str(remote), "--hash-prefix", "", "--score-script", str(stub), "--interval", "0", "--min-age-min", "20"]
    return remote, mirror, state, argv


def test_verify_marks_missing_and_changed_files_bad():
    ok, bad = cw.verify({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "X"})
    assert ok == ["a"] and bad == ["b", "c"]


def test_copy_verify_score_then_done(env):
    remote, mirror, state, argv = env
    make_remote(remote, TAGS[:3])
    make_remote(remote, TAGS[3:], age_min=1)               # still being written: never copied
    assert cw.main(argv + ["--max-ticks", "1"]) == cw.EXIT_RUNNING
    st = json.load(open(state / "state.json"))
    assert (st["installed"], st["scored"], st["final"], st["expected"]) == (3, 3, 3, 4)
    assert not os.path.exists(mirror / "latents" / f"{TAGS[3]}.npz")
    assert open(mirror / "models" / TAGS[0] / "model.pt").read() == f"models/{TAGS[0]}/model.pt\n"
    make_remote(remote, TAGS[3:])                           # the last fit is now old enough
    assert cw.main(argv) == cw.EXIT_DONE
    assert json.load(open(state / "state.json"))["scored"] == 4


def test_failure_record_counts_as_final_without_scoring(env):
    remote, mirror, state, argv = env
    make_remote(remote, TAGS[:3])
    make_remote(remote, TAGS[3:], failed=TAGS[3:])
    assert cw.main(argv) == cw.EXIT_DONE
    st = json.load(open(state / "state.json"))
    assert (st["failed_fits"], st["scored"]) == (1, 3)


def test_scoring_waits_for_other_run_and_tick_survives_nothing_new(env, tmp_path):
    remote, mirror, state, argv = env
    make_remote(remote, TAGS)
    jw = tmp_path / "jw"
    jw.mkdir()
    (jw / "state.json").write_text(json.dumps({"status": "running"}))
    assert cw.main(argv + ["--after-jw", str(jw), "--max-ticks", "1"]) == cw.EXIT_RUNNING
    assert json.load(open(state / "state.json"))["scored"] == 0 and "waits" in open(state / "chain_ws.log").read()
    (jw / "state.json").write_text(json.dumps({"status": "done"}))
    assert cw.main(argv + ["--after-jw", str(jw)]) == cw.EXIT_DONE


def test_busy_lock_defers_scoring_and_bad_scorer_raises(env, tmp_path):
    remote, mirror, state, argv = env
    make_remote(remote, TAGS)
    lock = mirror / ".ws_scoring.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "30"], start_new_session=True)
    time.sleep(0.5)
    try:
        assert cw.main(argv + ["--max-ticks", "1"]) == cw.EXIT_RUNNING
        assert json.load(open(state / "state.json"))["score_rc"] == cw.BUSY
    finally:
        os.killpg(holder.pid, 9)
    bad = tmp_path / "bad.sh"
    bad.write_text("#!/bin/bash\nexit 3\n")
    with pytest.raises(RuntimeError, match="exited 3"):
        cw.main(argv + ["--score-script", str(bad)])


def test_corrupt_copy_is_not_installed_and_setup_errors(env, tmp_path, monkeypatch):
    remote, mirror, state, argv = env
    make_remote(remote, TAGS[:1])
    real = cw.Remote.sums
    monkeypatch.setattr(cw.Remote, "sums", lambda self, files: {f: "0" * 64 for f in files})   # JHPCE-side hash differs
    assert cw.main(argv + ["--max-ticks", "1"]) == cw.EXIT_RUNNING
    st = json.load(open(state / "state.json"))
    assert (st["installed"], st["bad"], st["mirrored"]) == (0, 1, 0)
    monkeypatch.setattr(cw.Remote, "sums", real)
    assert cw.main(argv + ["--max-ticks", "1"]) == cw.EXIT_RUNNING and json.load(open(state / "state.json"))["installed"] == 1
    assert cw.main(argv[:argv.index("--mirror") + 1] + [str(tmp_path / "nope")] + argv[argv.index("--mirror") + 2:]) == cw.EXIT_USAGE
    dup = tmp_path / "dup"
    dup.write_text(TAGS[0] + "\n" + TAGS[0] + "\n")
    assert cw.main(argv[:argv.index("--tags-file") + 1] + [str(dup)] + argv[argv.index("--tags-file") + 2:]) == cw.EXIT_USAGE
