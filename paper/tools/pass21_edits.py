"""Pass 21 edits (2026-11-06): editing layout. The six main figures move from the 'Figures and tables' block at the end of
paper/outline.tex into the main text, each right after the paragraph that first cites it, so the text can be edited beside
its figure (user request; journal layout, with figures as separate files, comes later). Each figure environment moves
unchanged except its placement option, [t] -> [!htbp]. Layout only: no tracked-change markup, no words or numbers change. The
tables stay at the end. Every anchor must occur exactly once, at the start of a line, or the script stops.
Run from the repository root:  python paper/tools/pass21_edits.py   (PASS21_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("PASS21_ROOT") or Path(__file__).resolve().parents[2])

# figure number -> start of the line after which it is placed (first citation in the live text)
ANCHORS = {
    1: r"\qquad To establish a fair comparison between methods, we use a fixed backbone",   # Methods paragraph 1
    2: r"\replaced[id=P7]{With the batch-conditioned decoder, the ranking rested on five real tasks.",  # Res-2
    3: r"\replaced[id=P7]{The real-task curves overlapped near plain scVI",               # Res-3
    4: r"\replaced[id=P7]{The secondary summaries varied across real tasks (Figure 4;",   # Res-4
    5: r"\replaced[id=P7]{Across the real tasks, including the low-signal immune pair, good-strength widths",  # Res-6
    6: r"\item \replaced[id=P9]{At the strength chosen on real tasks of other families,",  # Res-7 (list closes on the next line)
}
NOTE = "% Pass 21: main figures placed in the text for editing (paper/pass21_changelog.md); journal layout later.\n"


def main():
    p = ROOT / "paper/outline.tex"
    s = p.read_text()
    blocks = {}
    for n in range(1, 7):
        pat = re.compile(r"% -+ Figure " + str(n) + r"\n\\begin\{figure\}\[t\]\n.*?\\end\{figure\}\n\n", re.S)
        found = pat.findall(s)
        assert len(found) == 1, (n, len(found))
        s = s.replace(found[0], "", 1)
        blocks[n] = found[0].replace(r"\begin{figure}[t]", r"\begin{figure}[!htbp]", 1)
    for n, anchor in ANCHORS.items():
        starts = [m.start() for m in re.finditer(r"(?m)^" + re.escape(anchor), s)]
        assert len(starts) == 1, (n, len(starts), anchor[:60])
        line_end = s.index("\n", starts[0]) + 1
        if n == 6:                                   # place after the \end{itemize} that closes Res-7's list
            nxt = s.index("\n", line_end) + 1
            assert s[line_end:nxt] == "\\end{itemize}\n", s[line_end:nxt]
            line_end = nxt
        s = s[:line_end] + "\n" + (NOTE if n == 1 else "") + blocks[n] + s[line_end:]
    assert s.count(r"\begin{figure}[!htbp]") == 6 and s.count(NOTE) == 1
    p.write_text(s)


main()
