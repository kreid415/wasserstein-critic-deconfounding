"""Pass 20 edits (2026-11-06, tracked id P20): Declarations. Ethics approval and consent to participate: drafted wording for a
re-analysis of public data replaces the Pass 3 sentence and its \\tbd. Competing interests: the user states there are none; the
BMC standard sentence replaces the \\tbd. No numbers are written. Every old text must occur exactly once or the script stops.
Run from the repository root:  python paper/tools/pass20_edits.py   (PASS20_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PASS20_ROOT") or Path(__file__).resolve().parents[2])


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


edit("paper/outline.tex", [
    (r"{\added[id=P3]{Only publicly available, previously published data are analysed (the tasks of the single-cell integration "
     r"benchmark). \tbd{confirm wording}}}",
     r"{\added[id=P20]{Not applicable. This study analyses only publicly available, previously published data: the integration "
     r"tasks of the single-cell integration benchmark (figshare article 12420968, version 8), which hold human and mouse single-cell "
     r"data from earlier studies, and two simulated datasets. No new data were collected from human participants or animals.}}"),
    (r"{\added[id=P3]{\tbd{competing interests statement; the earlier outline said none declared, confirm with all authors}}}",
     r"{\added[id=P20]{The authors declare that they have no competing interests.}}"),
    ("\\definechangesauthor[name={Pass 13}, color=orange]{P13}",
     "\\definechangesauthor[name={Pass 20}, color=blue]{P20}% Declarations: ethics wording, competing interests; change log paper/pass20_changelog.md\n"
     "\\definechangesauthor[name={Pass 13}, color=orange]{P13}"),
])
