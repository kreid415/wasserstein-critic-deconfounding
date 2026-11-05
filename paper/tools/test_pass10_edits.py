"""Tests for paper/tools/pass10_edits.py: the markup flattener and brace matcher, and a full in-process run on the pre-edit text."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
spec = importlib.util.spec_from_file_location("pass10_edits", TOOLS / "pass10_edits.py")
P10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P10)


def test_flatten_keeps_new_text_and_drops_old_and_deleted():
    assert P10.flatten(r"a \replaced[id=P4]{new \added{X}}{old \deleted{Y}} b \deleted{gone}c") == "a new X b c"
    assert P10.match_brace("{a{b}c}", 0) == 6
    with pytest.raises(ValueError):
        P10.match_brace("{a{b}", 0)


def test_rerun_on_pre_edit_text_gives_the_outline_and_stops_when_run_twice(tmp_path):
    pre = subprocess.run(["git", "-C", str(REPO), "show", "befebcd:paper/outline.tex"], capture_output=True, text=True)
    if pre.returncode:  # test-reason: needs the merge commit; skipped in a shallow copy
        pytest.skip("commit befebcd not available")
    for rel in ("paper/figure_captions_pass9.md", "paper/analysis_pass10/coverage_sim_sweep.json", "paper/analysis_pass10/reference_coverage.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    (tmp_path / "paper/outline.tex").write_text(pre.stdout)
    P10.main(tmp_path)
    assert (tmp_path / "paper/outline.tex").read_text() == (REPO / "paper/outline.tex").read_text()
    assert (tmp_path / "paper/pass10_numbers.csv").read_text() == (REPO / "paper/pass10_numbers.csv").read_text()
    with pytest.raises(AssertionError):      # old texts are gone: the script must stop, not pass silently
        P10.main(tmp_path)
