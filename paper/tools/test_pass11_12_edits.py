"""Pass 11 to Pass 17 edit scripts re-run in a temporary copy on the text they started from; each must reproduce the committed
result and must stop when run a second time (its old texts are gone)."""
import os, runpy, shutil, subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
DATA_ROOT = Path("/home/kendall/experiment_data/wasserstein-critic-deconfounding")


def git_show(rev, rel):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{rel}"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"{rev} not available")  # pragma: no cover  # shallow copies lack the history
    return r.stdout


@pytest.mark.parametrize("script, env, start, result, files, extra", [
    ("pass11_edits.py", "PASS11_ROOT", "717cc4c", "0053042", ["paper/outline.tex"], ["paper/figure_data/paper_numbers.json"]),
    ("pass12_edits.py", "PASS12_ROOT", "0053042", "d207051",
     ["paper/outline.tex", "revision/reviewer_response.md", "paper/lit_search_20261105.md"], []),
    ("pass13_edits.py", "PASS13_ROOT", "fe4b19e", "9faac6e",
     ["paper/outline.tex", "paper/supp/tableS6.tex", "paper/supp/tableS7.tex", "paper/tools/make_supp_tables.py", "revision/reviewer_response.md"], []),
    ("pass14_edits.py", "PASS14_ROOT", "5856fd5", "5dc2c8d",
     ["paper/outline.tex", "paper/supp_items.tex", "revision/reviewer_response.md"], ["paper/replication_data/replication_numbers.json"]),
    ("pass15_edits.py", "PASS15_ROOT", "bfdd82d", "79aa263", ["paper/outline.tex", "revision/reviewer_response.md"], []),
    ("pass16_edits.py", "PASS16_ROOT", "0e79c46", "a11ac79", ["paper/outline.tex", "revision/reviewer_response.md"], []),
    ("pass17_edits.py", "PASS17_ROOT", "dc21cfc", "38279bb",
     ["paper/outline.tex", "paper/supp/note3.tex", "paper/supp/note5.tex", "paper/supp/note6.tex", "paper/supp/note10.tex",
      "paper/supp-analysis_changelog.md", "paper/pass16_changelog.md", "revision/reviewer_response.md"], []),
])
def test_edit_script_reproduces_its_pass(tmp_path, monkeypatch, script, env, start, result, files, extra):
    if script == "pass12_edits.py" and not DATA_ROOT.is_dir():
        pytest.skip("pass 12 numbers are re-read from the data root (workstation only)")  # pragma: no cover
    for rel in files:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(git_show(start, rel))
    for rel in extra:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    monkeypatch.setenv(env, str(tmp_path))
    runpy.run_path(str(TOOLS / script), run_name="__main__")
    for rel in files:
        assert (tmp_path / rel).read_text() == git_show(result, rel), rel
    with pytest.raises(AssertionError):
        runpy.run_path(str(TOOLS / script), run_name="__main__")
