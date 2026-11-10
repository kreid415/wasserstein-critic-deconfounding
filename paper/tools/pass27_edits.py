"""Pass 27 edits (2026-11-10, tracked id P27): the collapse-rate figure (the former main Figure 4, fig4.py) is now Table 1
(paper/table1.tex, written by paper/tools/make_table1.py from paper/figure_data/collapse_common_range.csv), by the first author's
decision (a deliberate exception to the earlier rule of no main-text tables); the placement figure (the former Figure 5, fig5.py)
is Figure 4 (fig4.py, figures/fig4.*). This script changes the text only:
- paper/outline.tex, as tracked changes (id P27): the collapse-rate float is struck and commented out as Pass 24 did with the dot
  summary, and \\input{table1} (no markup) takes its place; the Res-4 and Res-6 references, the two figure-list items and the
  history sentence of the writer's-outline box (which named the Pass 23 Figures 4 and 5) follow the new numbering; the placement
  float's marker line, file name and label become Figure 4 (untracked, as in Pass 24);
- paper/supp/note7.tex, tableS7.tex, tableS8.tex and the two generators of the tables (paper/tools/make_supp_tables.py,
  make_supp_supp-analysis.py): "Figure 4" (collapse rate) becomes "Table 1" and "Figure 5" becomes "Figure 4", without tracked
  markup (both PDF builds input supplement files as they are).
No number and no other text changes. Change log paper/pass27_changelog.md. It starts from the text committed at 9db2382 (the Pass 26
result); every old string must occur exactly once, and the main text and the main captions (the Table 1 caption included) must stay
within the limits of Pass 24 (5,500 and 450 words), or the script stops before it writes anything.
Run from the repository root:  python paper/tools/pass27_edits.py   (PASS27_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept_changes import accept  # noqa: E402
from make_table1 import CAPTION  # noqa: E402
from pass24_edits import CAPTION_LIMIT, MAIN_LIMIT, MARK, budget, words  # noqa: E402

ROOT = Path(os.environ.get("PASS27_ROOT") or Path(__file__).resolve().parents[2])

AUTHOR = (r"\definechangesauthor[name={Pass 27}, color=brown]{P27}% collapse-rate figure replaced by Table 1, placement figure "
          "renumbered as Figure 4; change log paper/pass27_changelog.md\n")
TABLE_ITEM = (r"penalty, real task and decoder (percent, with the collapsed and the counted fits), with the median over the six "
              r"real tasks.")
PAIRS = [  # (old, new) in paper/outline.tex, each old string once; a nested \replaced goes inside the P24 new text, as in Pass 26
    (r"(Figure \replaced[id=P24]{4;}{5;} the share of non-dominated",                           # Res-4
     r"(\replaced[id=P27]{Table}{Figure} \replaced[id=P24]{\replaced[id=P27]{1;}{4;}}{5;} the share of non-dominated"),
    (r"(Figure \replaced[id=P24]{5;}{6;} Supplementary Table \ref{tab:4})",                     # Res-6
     r"(Figure \replaced[id=P24]{\replaced[id=P27]{4;}{5;}}{6;} Supplementary Table \ref{tab:4})"),
    (r"\item Figure 4: \replaced[id=P24]{collapse rate}",                                       # figure list
     r"\item \replaced[id=P27]{Table 1}{Figure 4}: \replaced[id=P24]{collapse rate}"),
    (r"\replaced[id=P24]{penalty, one panel per decoder, one point per real task with its 95\% interval and the median.}{task.}",
     r"\replaced[id=P24]{\replaced[id=P27]{" + TABLE_ITEM + r"}{penalty, one panel per decoder, one point per real task with its 95\% "
     r"interval and the median.}}{task.}"),
    (r"\item Figure 5: \deleted[id=P24]{collapse rate up to the highest common}",
     r"\item Figure \replaced[id=P27]{4}{5}: \deleted[id=P24]{collapse rate up to the highest common}"),
    (r"\added[id=P23]{ In Pass 23 Figure 4, the Pareto frontier, was added, and the local-versus-global analysis (panel c of the former "
     r"Figure 4) and the good-strength width (the former Figure 5) were removed, so Figures 1--6 are drawn from the sweep analysis.}",
     r"\added[id=P23]{ In Pass 23 }\replaced[id=P27]{the Pareto frontier was added as a main figure}{Figure 4, the Pareto frontier, was "
     r"added}\added[id=P23]{, and the local-versus-global analysis (}\replaced[id=P27]{a panel of the collapse-rate figure}{panel c of "
     r"the former Figure 4}\added[id=P23]{) and the good-strength width (}\replaced[id=P27]{a main figure}{the former Figure 5}"
     r"\added[id=P23]{) were removed, so Figures 1--6 are drawn from the sweep analysis.}"),
]
SUPP = {  # untracked, in place: path -> [(old, new)]
    "paper/supp/note7.tex": [(r"Figures 3 and 4 and Supplementary Figure S8; the placement is in Figure 5 and",
                              r"Figure 3, Table 1 and Supplementary Figure S8; the placement is in Figure 4 and"),
                             (r"The main text and Figure 4 count only", r"The main text and Table 1 count only")],
    "paper/supp/tableS7.tex": [(r"and of Figure~5)", r"and of Figure~4)")],
    "paper/supp/tableS8.tex": [(r"collapse rate, Figure 4)", r"collapse rate, Table 1)")],
    "paper/tools/make_supp_tables.py": [(r"and of Figure~5)", r"and of Figure~4)")],
    "paper/tools/make_supp_supp-analysis.py": [(r"rate, Figure 4). \textbf{Part B}", r"rate, Table 1). \textbf{Part B}")],
}


def replace_once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new)


def float_span(s, mark):
    """Start of the marker line and the span of the figure environment that follows it."""
    assert s.count(mark) == 1, mark
    i = s.index(mark)
    b = s.index(r"\begin{figure}", i)
    return i, b, s.index(r"\end{figure}", b) + len(r"\end{figure}")


def outline(s):
    s = replace_once(s, r"\definechangesauthor[name={Pass 26}, color=teal]{P26}", AUTHOR + r"\definechangesauthor[name={Pass 26}, color=teal]{P26}")
    # the collapse-rate float is struck and commented out; the generated table is input in its place
    mark = MARK + "Figure 4 (collapse rate; Figure 5 in Pass 23)\n"
    i, b, e = float_span(s, mark)
    assert "figures/fig4.png" in s[b:e] and r"\caption{Collapse rate per" in accept(s[b:e])
    commented = "".join("%" + ln + "\n" for ln in s[b:e].split("\n"))
    note = r"\noindent\deleted[id=P27]{Figure 4 (collapse rate) replaced by Table 1.}" + "\n"
    s = (s[:i] + MARK + "Table 1 (collapse rate; Figure 4 in Pass 24, Figure 5 in Pass 23)\n" + r"\input{table1}" + "\n" + note + commented
         + s[e:].lstrip("\n"))
    # the placement float becomes Figure 4
    mark = MARK + "Figure 5 (placement; Figure 6 in Pass 23)\n"
    i, b, e = float_span(s, mark)
    blk = s[i:e]
    assert blk.count("figures/fig5.png") == blk.count(r"\label{fig:5}") == 1
    blk = blk.replace(mark, MARK + "Figure 4 (placement; Figure 5 in Pass 24, Figure 6 in Pass 23)\n")
    blk = blk.replace("figures/fig5.png", "figures/fig4.png").replace(r"\label{fig:5}", r"\label{fig:4}")
    s = s[:i] + blk + s[e:]
    for a, c in PAIRS:
        s = replace_once(s, a, c)
    return s


def main():
    text = outline((ROOT / "paper/outline.tex").read_text())
    n_main, n_cap = budget(text)
    n_cap += len(words(CAPTION))                      # the Table 1 caption is a main caption now
    assert n_main <= MAIN_LIMIT and n_cap <= CAPTION_LIMIT, (n_main, n_cap)
    edited = {"paper/outline.tex": text}
    for rel, pairs in SUPP.items():
        s = (ROOT / rel).read_text()
        for a, c in pairs:
            s = replace_once(s, a, c)
        edited[rel] = s
    for rel, s in edited.items():
        (ROOT / rel).write_text(s)
    print(f"main text {n_main} words (limit {MAIN_LIMIT}); main captions, Table 1 included, {n_cap} words (limit {CAPTION_LIMIT})")


if __name__ == "__main__":
    main()
