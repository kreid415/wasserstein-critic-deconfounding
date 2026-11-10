"""Pass 23 edit script re-run in a temporary copy on the text it started from (Pass 22, commit 045e62c); every file it writes must
equal the file at the Pass 23 result commit (24f2665) (paper/outline.tex and the supplement files are never edited by hand)."""
import runpy
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
START = "045e62c"
RESULT = "24f2665"  # the Pass 23 result; later passes edit the same files, so the script is compared with this commit
EDITED = ["paper/outline.tex", "paper/supp_items.tex", "paper/supp/figures.tex", "paper/supp/tableS6.tex", "paper/supp/tableS7.tex",
          "paper/supp/note3.tex", "paper/supp/note5.tex", "paper/supp/note6.tex", "paper/supp/note7.tex", "paper/supp/note8.tex",
          "paper/supp/note10.tex", "paper/tools/make_supp_tables.py", "paper/tools/test_supp.py"]
NEW = ["paper/supp/note13.tex", "paper/supp/tableS12.tex", "paper/supp/tableS13.tex", "paper/supp/tableS14.tex"]
DATA = ["paper/figure_data/res5_label_transfer.json", "paper/figure_data/pareto_frontier.json", "paper/figure_data/collapse_common_range.json",
        "paper/figure_data/replication_mean_ranks.json", "paper/replication_data/replication_numbers.json"]


def test_pass23_script_reproduces_the_committed_files(tmp_path, monkeypatch):  # test-reason: runs pass23_edits.py, which no other test runs; catches a hand edit of outline.tex or of a supplement file the script writes
    for rel in EDITED:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{START}:{rel}"], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"{START} not available")  # pragma: no cover  # shallow copies lack the history
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(r.stdout)
    for rel in DATA:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text((REPO / rel).read_text())
    monkeypatch.setenv("PASS23_ROOT", str(tmp_path))
    runpy.run_path(str(TOOLS / "pass23_edits.py"), run_name="__main__")
    for rel in EDITED + NEW:
        committed = subprocess.run(["git", "-C", str(REPO), "show", f"{RESULT}:{rel}"], capture_output=True, text=True).stdout
        assert (tmp_path / rel).read_text() == committed, rel
