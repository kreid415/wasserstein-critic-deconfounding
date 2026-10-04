"""The rejected pre-revision pipeline is retired (code check CR-10): its files live in legacy/, each stops with a
RETIRED message and a non-zero exit before doing anything, and none is left in scripts/.
Run: python -m pytest -q tests/test_legacy_retired.py (any env with python and bash)."""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETIRED = ["score_final_config.py", "scvi_final_manifest.tsv", "run_jhpce_pilot.sh", "score_parallel.sh",
           "score_baselines_parallel.sh", "run_final_baselines.py", "scvi_adv_fit.py"]


def retirement_violation(path):
    """None if running `path` stops at once with exit != 0 and 'RETIRED' on stderr; else the reason."""
    cmd = [sys.executable, path] if path.endswith(".py") else ["bash", path]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=os.path.dirname(path),
                       env=dict(os.environ, PATH=os.environ.get("PATH", "")))
    if p.returncode == 0:
        return f"exit 0: {p.stdout[-200:]}"
    if "RETIRED (code check CR-10" not in p.stderr:
        return f"exit {p.returncode} without the RETIRED message: {p.stderr[-300:]}"
    return None


def test_retired_files_moved_out_of_scripts():
    for f in RETIRED:
        assert not os.path.exists(os.path.join(ROOT, "scripts", f)), f
        assert os.path.exists(os.path.join(ROOT, "legacy", f)), f
    assert os.path.exists(os.path.join(ROOT, "legacy", "README.md"))


@pytest.mark.parametrize("f", [f for f in RETIRED if not f.endswith(".tsv")])
def test_every_retired_program_refuses_to_run(f):
    assert retirement_violation(os.path.join(ROOT, "legacy", f)) is None


def test_checker_detects_a_program_without_the_guard(tmp_path):
    """Mutation check (fail-loud R11): the same file without its guard line is reported."""
    for f in ("scvi_adv_fit.py", "score_parallel.sh"):
        src = open(os.path.join(ROOT, "legacy", f)).read().split("\n")
        kept = [x for x in src if "RETIRED (code check CR-10" not in x]
        assert len(kept) == len(src) - 1
        p = tmp_path / f
        p.write_text("\n".join(kept))
        assert retirement_violation(str(p)) is not None, f


def test_current_pipeline_does_not_use_retired_files():
    current = ["run_stage.py", "fit_paper_config.py", "score_scib_native.py", "build_paper_manifest.py", "prereg_rules.py",
               "run_cpu_baselines.py", "fit_outcome.py", "scvi_adversarial_plan.py"]
    for c in current:
        src = open(os.path.join(ROOT, "scripts", c)).read()
        for f in RETIRED:
            stem = f.rsplit(".", 1)[0]
            assert f"import {stem}" not in src and f"from {stem}" not in src and f"scripts/{f}" not in src, (c, f)
