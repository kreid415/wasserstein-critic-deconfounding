"""scripts/check_tests.py on a throw-away git repo: pass, uncovered change, redundant new test, --strict, failing
tests and an unknown base ref (NB-20261016-23, SI-91). Each case builds the repo, edits it after the base commit and
runs check_tests.main() in process (its own pytest run is a real subprocess on the throw-away repo)."""
import runpy
import subprocess
import sys

import check_tests
import pytest

MOD = "def f(x):\n    if x:\n        return 1\n    return 0\n\n\ndef g():\n    return {g}\n"
TEST = (
    "from pkg.mod import f, g\n\n\nclass TestBase:\n    def test_f(self):\n"
    "        assert f(1) == 1 and f(0) == 0\n\n\ndef test_g():\n    assert g() == {g}\n"
)
BASE = {"conftest.py": "", "pkg/mod.py": MOD.format(g=5), "pkg/old.py": "OLD = 1\n", "pkg/test_inner.py": "def test_inner():\n    assert True\n", "tests/test_mod.py": TEST.format(g=5)}
NEW_TESTS = (  # a class-level test that adds no coverage, one with a reason, one test that is not changed
    "from pkg.mod import f\n\n\nclass TestDup:\n    def test_dup(self):\n        assert f(1) == 1\n\n"
    "    # test-reason: documents the contract\n    def test_reasoned(self):\n        assert f(0) == 0\n\n\n"
    "def helper():\n    return 1\n"
)
CASES = {  # name: (files written after the base commit, deleted files, extra args, exit code, text that must be printed)
    "ok": ({"pkg/mod.py": MOD.format(g=6), "tests/test_mod.py": TEST.format(g=6), "tools/x.py": "X = 1\n"}, [], [], 0, ["check_tests: OK"]),
    "uncovered": (
        {"pkg/mod.py": MOD.format(g=6) + "\n\ndef h(x):\n    if x:\n        return 2\n    return 3\n",
         "tests/test_mod.py": TEST.format(g=6) + "\n\ndef test_h():\n    from pkg.mod import h\n    assert h(1) == 2\n",
         "pkg/sub/new.py": "NEW = 1\n"},
        ["pkg/old.py"], [], 1,
        ["FAIL pkg/mod.py: changed lines never run: [14]", "changed branches never taken", "FAIL pkg/sub/new.py: changed, but no test imports this file"],
    ),
    "warn": ({"tests/test_new.py": NEW_TESTS}, [], [], 0, ["WARN tests/test_new.py::TestDup::test_dup: covers no line"]),
    "warn_strict": ({"tests/test_new.py": NEW_TESTS}, [], ["--strict"], 1, ["WARN tests/test_new.py::TestDup::test_dup"]),
    "tests_fail": ({"tests/test_bad.py": "def test_bad():\n    assert False\n"}, [], [], 2, ["pytest exited 1"]),
}


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True)


def write(repo, files):
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)


@pytest.mark.parametrize("case", sorted(CASES))
def test_check_tests_verdicts(case, tmp_path, monkeypatch, capfd):
    files, deleted, extra, code, printed = CASES[case]
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, BASE)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    write(repo, files)
    for name in deleted:
        (repo / name).unlink()
    git(repo, "add", "-N", *[n for n in files if n != "pkg/sub/new.py"])  # pkg/sub/new.py stays untracked on purpose
    for key in [k for k in __import__("os").environ if k.startswith(("COV_CORE", "COVERAGE"))]:
        monkeypatch.delenv(key)  # the throw-away pytest must not write into this run's coverage data
    monkeypatch.chdir(repo)
    monkeypatch.setattr(sys, "argv", ["check_tests.py", "--base", "HEAD", *extra, "--src", "pkg", "--", "tests", "pkg"])
    assert check_tests.main() == code
    out = capfd.readouterr().out
    for text in printed:
        assert text in out, out[-1500:]


def test_check_tests_unknown_base_is_refused(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_tests.py", "--base", "no-such-ref"])
    with pytest.raises(SystemExit, match="base ref 'no-such-ref' not found"):
        runpy.run_path(check_tests.__file__, run_name="__main__")  # also runs the `__main__` guard
