"""Pass 17 edits (tracked id P17): the finished supplement of branches supp-methods (17aadb9), supp-analysis (ef9062c) and
supp-records (6bbbab5), merged at fe5156c, is placed in the supplementary list of paper/outline.tex, and the main text, the
supplement and the response are corrected where the three change logs (paper/supp-*_changelog.md) or the lead's check showed
them wrong. Every old text must occur exactly once or the script stops. Numbers written are listed in paper/pass17_numbers.csv.
- Supplementary list: the one-line descriptions of Supplementary Notes 2-12 and Tables S1, S2, S4, S5 and S8 are struck and the
  finished file is input in their place, outside any list (the tables are longtables); the table counter is restored after the
  list so that the main tables still print 1-3.
- Methods: replication scoring sites (lung, simulation 2 and pancreas: sweep fits scored on the workstation, replication fits
  on the cluster); 448 planned baseline fits, 447 fitted and scored; ranking-rule dates and the baseline score files that
  predate the ranking rules.
- Supplementary Note 6: the large chromatin-accessibility task holds all cells of the small one (it does not merely share
  3,618). Supplementary Note 10: the family-level sign-flip p filled in; the early baseline score files stated.
- Response: the supplement is supplied; 448 planned and 447 scored baseline fits.
Run from the repository root:  python paper/tools/pass17_edits.py   (PASS17_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/abstract_words.py ; python paper/tools/check_ledger.py"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept_changes import accept  # noqa: E402  (flattens the struck descriptions: their own markup accepted first)

ROOT = Path(os.environ.get("PASS17_ROOT") or Path(__file__).resolve().parents[2])


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


def add(text):
    return r"\added[id=P17]{" + text + "}"


FILES = {**{f"Supplementary Note {k},": f"supp/note{k}" for k in range(2, 13)},
         **{f"Table S{k},": f"supp/tableS{k}" for k in (1, 2, 4, 5, 8)}}
COMMENT = re.compile(r"(?<!\\)%.*$")


def wire_in(block):
    """The supplementary list with each finished item input in place of its struck description; the other items stay in
    lists, consecutive ones in one list."""
    out, keep = [], []
    lines = block.split("\n")
    for line in lines:
        if line in (r"\begin{itemize}", r"\end{itemize}"):
            continue
        if not line.startswith(r"\item "):
            if keep:
                out += [r"\begin{itemize}", *keep, r"\end{itemize}"]
                keep = []
            out.append(line)
            continue
        body = line[len(r"\item "):]
        m = COMMENT.search(body)
        text, comment = (body[:m.start()], body[m.start():]) if m else (body, "")
        target = [f for k, f in FILES.items() if accept(text).strip().startswith(k)]
        if not target:
            keep.append(line)
            continue
        assert not comment.strip(), line[:80]
        if keep:
            out += [r"\begin{itemize}", *keep, r"\end{itemize}"]
            keep = []
        out += [r"\begin{itemize}\item \deleted[id=P17]{" + accept(text).strip() + r"}\end{itemize}",
                r"\input{" + target[0] + "}"]
    if keep:
        out += [r"\begin{itemize}", *keep, r"\end{itemize}"]
    used = [l for l in out if l.startswith(r"\input{")]
    assert len(used) == len(FILES), used
    return "\n".join(out)


p = ROOT / "paper/outline.tex"
s = p.read_text()
START, END = "\\textbf{Supplementary Notes}\n", "\\textbf{Supplementary Figures}\n"
assert s.count(START) == 1 and s.count(END) == 1
i, j = s.index(START), s.index(END)
s = (s[:i] + "% Pass 17: the supplementary tables input in this list step the table counter; it is restored after the list "
     "(main tables 1-3)\n\\newcounter{maintables}\\setcounter{maintables}{\\value{table}}\\let\\mainTheHtable\\theHtable"
     "\\renewcommand{\\theHtable}{supp.\\arabic{table}}%\n" + START + wire_in(s[i + len(START):j]) + "\n" + s[j:])
p.write_text(s)

OLD_HEAD = (r"\deleted[id=P3]{no citations; }Supplementary Notes 1--\replaced{11}{12}, Tables S1--S\replaced{8}{6}"
            r"\added[id=P5]{ and S9}\added[id=P9]{, Supplementary Note 12 and Table S10}, Figures S1--S\replaced{6}{6}")
OLD_P15 = (r"This document contains Supplementary Note 1, Tables S6, S7, S9 and S10 and Figures S1--S6; the other notes and tables "
           r"listed below are described here and are not yet written.")
edit("paper/outline.tex", [
    (r"\definechangesauthor[name={Pass 16}, color=green]{P16}% family-level sign-flip sensitivity analysis; change log paper/pass16_changelog.md, numbers in paper/pass16_numbers.csv" "\n",
     r"\definechangesauthor[name={Pass 16}, color=green]{P16}% family-level sign-flip sensitivity analysis; change log paper/pass16_changelog.md, numbers in paper/pass16_numbers.csv" "\n"
     r"\definechangesauthor[name={Pass 17}, color=olive]{P17}% finished supplement placed in the list, main-text corrections; change log paper/pass17_changelog.md, numbers in paper/pass17_numbers.csv" "\n"),
    (r"Pass 16 adds the family-level sign-flip sensitivity analysis to the replication result (change log \texttt{paper/pass16\_changelog.md}).}",
     r"Pass 16 adds the family-level sign-flip sensitivity analysis to the replication result (change log \texttt{paper/pass16\_changelog.md}).}"
     + add(r" Pass 17 places the finished supplement (Supplementary Notes 2--12, Tables S1, S2, S4, S5 and S8) in the supplementary "
           r"list and corrects the main text where the supplement's sources showed it wrong (change log \texttt{paper/pass17\_changelog.md}).")),
    # Methods, replication paragraph: scoring sites
    (r"Each task's replication fits are made and scored on the same kind of machine, with the same scorer, as its sweep fits (Supplementary Note 11).",
     r"Each task's replication fits are made \replaced[id=P17]{on the same kind of machine as its sweep fits and scored with the same "
     r"scoring code; for three tasks the scoring site differs (the sweep fits of lung, simulation 2 and pancreas were scored on the "
     r"workstation and their replication fits on the cluster, under the sweep's rule for combining the two sites; Supplementary Note 11).}"
     r"{and scored on the same kind of machine, with the same scorer, as its sweep fits (Supplementary Note 11).}"),
    # Methods, baselines paragraph
    (r"all on every task: 448 fits, including three sensitivity settings",
     r"all on every task: 448 \replaced[id=P17]{planned fits (447 fitted and scored; Results)}{fits}, including three sensitivity settings"),
    # Methods, ranking paragraph: rule dates; baseline score files older than the ranking rules
    (r"whereas no score of the other five tasks existed when the ranking rules were written (22--23 October 2026).}",
     r"whereas no \added[id=P17]{sweep }score of the other five tasks existed when the ranking rules were written "
     r"\replaced[id=P17]{(dated 22 and 23 October 2026; both commits were made on 22 October), although score files of some "
     r"baseline fits on four of them are older, from 6 to 12 October (Supplementary Notes 10 and 11).}{(22--23 October 2026).}}"),
    # Supplementary material heading and the Pass 15 note
    (OLD_HEAD, r"\replaced[id=P17]{Supplementary Notes 1--12, Tables S1, S2 and S4--S10, Figures S1--S6}{" + accept(OLD_HEAD) + "}"),
    (r"\added[id=P15]{\noindent " + OLD_P15 + "}",
     r"\noindent\replaced[id=P17]{Every supplementary note and table listed here is in this document: Supplementary Notes 2--12 and "
     r"Tables S1, S2, S4, S5 and S8 are printed below in the place of their entries; Supplementary Note 1, Tables S6, S7 and S9 "
     r"and Figures S1--S6 follow the main figures and tables, and Table S10 follows Table~\ref{tab:4}. Table S3 (strength ranges "
     r"of the planned fixed-strength benchmark) was removed with that benchmark, and its number is not used.}{" + OLD_P15 + "}"),
    # restore the table counter after the supplementary list
    ("\n" + r"\textbf{Supplementary Figures}" + "\n",
     "\n" + r"\setcounter{table}{\value{maintables}}\let\theHtable\mainTheHtable% Pass 17: main tables continue at their own numbers"
     "\n" + r"\textbf{Supplementary Figures}" + "\n"),
])

edit("paper/supp/note6.tex", [
    ("the human--mouse immune task contains all 33,506 cells of the immune task, and the small and large chromatin-accessibility "
     "tasks share 3,618 cells.",
     "the human--mouse immune task contains all 33,506 cells of the immune task, and the large chromatin-accessibility task "
     "contains all 11,270 cells of the small one (matched by batch and barcode, with the same cell-type labels; 3,618 of them, "
     "the 10x Genomics batch, under identical cell names; Supplementary Table S1)."),
])
edit("paper/supp/note10.tex", [
    (r"(one-sided $p$ \res{family sign-flip p})", r"(one-sided $p=0.0156$, the smallest value six families allow: 1 of $2^6=64$ patterns)"),
    ("the sweep fits of the other five tasks were being made (from 22 October) and their score files are dated 28 to 30 October.",
     "the sweep fits of the other five tasks were being made (from 22 October) and their score files are dated 28 to 30 October. "
     "Score files of some baseline fits on four of these tasks are older: those of the scvi-tools adversary, scANVI and sysVI on "
     "lung, pancreas and simulation 2 are dated 6 and 7 October, and at least one of the scvi-tools adversary on the "
     "human--mouse immune task 12 October (Supplementary Note 11, Table A)."),
])
edit("revision/reviewer_response.md", [
    ("re-review. The supplementary material of this version contains Supplementary Note 1, Tables S6, S7, S9 and S10 and Figures S1-S6;\n"
     "the other supplementary notes and tables named below are described in the outline and are not yet written. Table numbers are",
     "re-review. Pass 17 (2026-11-06; `paper/pass17_changelog.md`) adds the finished supplement: the supplementary material of this\n"
     "version contains Supplementary Notes 1-12, Tables S1, S2 and S4-S10 and Figures S1-S6 (Table S3, the strength ranges of the\n"
     "planned fixed-strength benchmark, was removed with that benchmark). Table numbers are"),
    ("uncorrected principal components (448 baseline fits)",
     "uncorrected principal components (448 planned baseline fits, 447 fitted and scored)"),
])
