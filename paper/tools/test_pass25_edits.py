"""Pass 25 edit script re-run in a temporary copy on the text it started from (the Pass 24 result, commit 1cea7ef); every file it
writes must equal the committed file (paper/outline.tex and the supplement files are never edited by hand)."""
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
from accept_changes import accept  # noqa: E402

START = "1cea7ef"
EDITED = ["paper/outline.tex", "paper/supp/figures.tex", "paper/supp/note9.tex"]


def test_pass25_script_reproduces_the_committed_files(tmp_path, monkeypatch):  # test-reason: runs pass25_edits.py, which no other test runs; catches a hand edit of the files it writes, an edit of the main text, and a caption that loses a decoder name or a file pointer
    start = {}
    for rel in EDITED:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{START}:{rel}"], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"{START} not available")  # pragma: no cover  # shallow copies lack the history
        start[rel] = r.stdout
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(r.stdout)
    monkeypatch.setenv("PASS25_ROOT", str(tmp_path))
    runpy.run_path(str(TOOLS / "pass25_edits.py"), run_name="__main__")
    for rel in EDITED:
        assert (tmp_path / rel).read_text() == (REPO / rel).read_text(), rel
    after = (tmp_path / "paper/outline.tex").read_text()
    main = lambda t: t[t.index(r"\begin{document}"):t.index(r"\hd{Supplementary material}")]
    assert main(after) == main(start["paper/outline.tex"])        # no change before the supplementary material heading
    assert "id=P25" in after and r"\definechangesauthor[name={Pass 25}" in after
    assert "label transfer: F1 minus plain scVI's" in " ".join(accept(after).split())
    caption = (tmp_path / "paper/supp/figures.tex").read_text()
    for text in ("batch-conditioned decoder", "unconditioned decoder", "figS3\\_values.csv", "label\\_transfer\\_types.csv", "15-nearest-neighbour"):
        assert text in caption, text
