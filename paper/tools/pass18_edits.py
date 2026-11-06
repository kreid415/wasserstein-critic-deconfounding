"""Pass 18 edits (2026-11-06): the label-transfer table of Supplementary Note 9 becomes Supplementary Table S11
(numbered caption written by paper/tools/make_supp_supp-analysis.py), and the text that names the supplementary tables
says so. The two outline sentences sit inside Pass 17's \\replaced[id=P17]{...}{...} and `changes` cannot nest, so they are
amended inside the P17 text (no P18 mark); paper/pass18_changelog.md records it. Every old text must occur exactly once or the
script stops; no numbers are written.
Run from the repository root:  python paper/tools/pass18_edits.py   (PASS18_ROOT overrides the root of the edited files)
Then: python paper/tools/make_supp_supp-analysis.py ; python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PASS18_ROOT") or Path(__file__).resolve().parents[2])


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


edit("paper/outline.tex", [
    ("Supplementary Notes 1--12, Tables S1, S2 and S4--S10, Figures S1--S6}{",
     "Supplementary Notes 1--12, Tables S1, S2 and S4--S11, Figures S1--S6}{"),
    ("Tables S1, S2, S4, S5 and S8 are printed below in the place of their entries;",
     "Tables S1, S2, S4, S5 and S8 are printed below in the place of their entries (Table S11 within Note 9);"),
])
edit("paper/supp/note9.tex", [
    ("the score of an embedding is the mean over its held-out batches, and the table gives the mean and standard deviation over seeds.",
     "the score of an embedding is the mean over its held-out batches, and Supplementary Table S11 gives the mean and standard "
     "deviation over seeds."),
])
edit("revision/reviewer_response.md", [
    ("version contains Supplementary Notes 1-12, Tables S1, S2 and S4-S10 and Figures S1-S6",
     "version contains Supplementary Notes 1-12, Tables S1, S2 and S4-S11 and Figures S1-S6"),
])
