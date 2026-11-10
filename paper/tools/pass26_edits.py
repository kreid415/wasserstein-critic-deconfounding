"""Pass 26 edits (2026-11-10, tracked id P26): Figure 1 keeps the model schematic only (paper/figure_scripts/fig1.py); the reading panel
that was panel b is now Supplementary Figure S9 (paper/figure_scripts/figS9.py; plotted rows in paper/figure_data/figS9_points.csv,
readings in figS9_reading.csv). This script changes the text only, in three files:
- paper/outline.tex, as tracked changes (id P26): the Figure 1 caption describes the model only (the "(a)" and the panel (b) sentences are
  struck), the Methods references to Figure 1a and Figure 1b, the figure range of the Supplementary material heading and its intro
  sentence, and a new Supplementary Figures list item for Figure S9;
- paper/supp/figures.tex: Supplementary Figure S9 appended after S8, without tracked markup (both PDF builds input supplement files as they are);
- paper/supp_items.tex: the figure range "Figures S1--S8" becomes "Figures S1--S9" (no tracked markup).
No other text, no number and no other figure changes. Change log paper/pass26_changelog.md. It starts from the text committed at a33ccc6
(the Pass 25 result); every old string must occur exactly once, and the main text and the main-figure captions must stay within the
limits of Pass 24 (5,500 and 450 words), or the script stops before it writes anything.
Run from the repository root:  python paper/tools/pass26_edits.py   (PASS26_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept_changes import accept  # noqa: E402
from pass24_edits import CAPTION_LIMIT, MAIN_LIMIT, budget  # noqa: E402

ROOT = Path(os.environ.get("PASS26_ROOT") or Path(__file__).resolve().parents[2])

AUTHOR = (r"\definechangesauthor[name={Pass 26}, color=teal]{P26}% Figure 1 panel b moved to Supplementary Figure S9; "
          "change log paper/pass26_changelog.md\n")
CAPTION_START = r"\caption{Model and read-out. (a) "
S9 = r"""
\begin{figure}[p]
\centering\renewcommand{\figurename}{Supplementary Figure}\renewcommand{\thefigure}{S9}%
\includegraphics[width=0.6\textwidth]{figures/figS9.png}
\caption{How biology is read at the matched batch removal $b^*$ (Methods; Supplementary Note 5). ATAC small with the unconditioned decoder, three of the six penalties (JS discriminator, reference critic and MMD), means over three seeds (100--102). Batch score (top) and biology score (bottom) against penalty strength on a log scale; plain scVI (star) is drawn at strength 0, on the left, beyond an axis break. Biology is read where the batch score first reaches $b^*$ (dashed line), interpolating between neighbouring strengths (open circles, joined by dotted lines); later crossings (grey x) are not used. The legend gives each penalty's biology score at $b^*$. The plotted values are in \texttt{figS9\_points.csv} and the readings in \texttt{figS9\_reading.csv}, in \texttt{paper/figure\_data}.}\label{fig:s9}
\end{figure}
"""
S9_ITEM = (r"\item \added[id=P26]{Figure S9, how biology is read at matched batch removal $b^*$: batch score and biology score against "
           r"penalty strength for three penalties, ATAC small with the unconditioned decoder, with the first crossing of $b^*$ marked.}")
S8_ITEM = (r"\item \added[id=P24]{Figure S8, hypervolume up to the highest common batch score per penalty, task and decoder, with the "
           r"median over the ranked real tasks.}")
PAIRS = [  # (old, new) in paper/outline.tex, each old string once; nested \replaced goes inside the P24 new text, as in Pass 9 and 11
    (r"(Figure 1a)", r"(Figure 1\deleted[id=P26]{a})"),
    (r"(Figure \replaced[id=P24]{1b; Supplementary Note 5).}{1b). This matched level works only if the penalties remove clearly more "
     r"batch effect than plain scVI.}",
     r"(\replaced[id=P26]{Supplementary Figure S9}{Figure 1b}\replaced[id=P24]{; Supplementary Note 5).}{). This matched level works "
     r"only if the penalties remove clearly more batch effect than plain scVI.}"),
    (r"(Figure 1b). A leave-one-seed-out standard error", r"(\replaced[id=P26]{Supplementary Figure S9}{Figure 1b}). A leave-one-seed-out standard error"),
    (r"Figures S1--\replaced[id=P24]{S8}{S6}}", r"Figures S1--\replaced[id=P24]{\replaced[id=P26]{S9}{S8}}{S6}}"),
    (r"Figures \replaced[id=P24]{S1--S8,}{S1--S6,}", r"Figures \replaced[id=P24]{\replaced[id=P26]{S1--S9,}{S1--S8,}}{S1--S6,}"),
    (S8_ITEM, S8_ITEM + "\n" + S9_ITEM),
]


def replace_once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new)


def outline(s):
    s = replace_once(s, r"\definechangesauthor[name={Pass 25}, color=purple]{P25}", AUTHOR + r"\definechangesauthor[name={Pass 25}, color=purple]{P25}")
    lines = s.split("\n")
    hit = [i for i, ln in enumerate(lines) if ln.startswith(CAPTION_START)]
    assert len(hit) == 1 and lines[hit[0]].endswith("}") and lines[hit[0]].count(" (b) ") == 1, hit
    head, tail = lines[hit[0]].split(" (b) ")
    old = accept(tail[:-1]).strip()  # the caption's closing brace is the last character
    assert old.startswith("ATAC small") and old.endswith("later crossings are not used."), old
    head = replace_once(head, CAPTION_START, r"\caption{\replaced[id=P26]{Model.}{Model and read-out. (a)} ")
    lines[hit[0]] = head + r"\deleted[id=P26]{ (b) " + old + "}}"  # the panel (b) sentences stay as struck text
    s = "\n".join(lines)
    for a, b in PAIRS:
        s = replace_once(s, a, b)
    return s


def main():
    p, q, r = ROOT / "paper/outline.tex", ROOT / "paper/supp/figures.tex", ROOT / "paper/supp_items.tex"
    text = outline(p.read_text())
    n_main, n_cap = budget(text)
    assert n_main <= MAIN_LIMIT and n_cap <= CAPTION_LIMIT, (n_main, n_cap)
    figures = replace_once(q.read_text(), "% generated; Supplementary Figures S1-S8 (scripts", "% generated; Supplementary Figures S1-S9 (scripts") + S9
    items = replace_once(r.read_text(), "Figures S1-S8 (branch supp-items", "Figures S1-S9 (branch supp-items")
    items = replace_once(items, "Figures S1--S8)}", "Figures S1--S9)}")
    p.write_text(text)
    q.write_text(figures)
    r.write_text(items)
    print(f"main text {n_main} words (limit {MAIN_LIMIT}); main-figure captions {n_cap} words (limit {CAPTION_LIMIT})")


if __name__ == "__main__":
    main()
