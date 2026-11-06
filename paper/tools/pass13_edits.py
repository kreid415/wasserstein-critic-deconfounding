"""Pass 13 edits (tracked id P13): abstract cut to the BMC Bioinformatics limit, consecutive table numbers, the reference-batch
justification (the first author's margin todo). Edits paper/outline.tex, paper/supp/tableS6.tex, paper/supp/tableS7.tex,
paper/tools/make_supp_tables.py (the generator of the two supplement tables) and revision/reviewer_response.md. Every old text must
occur exactly once or the script stops. No number is written that is not already in the text it replaces (numbers kept in the
abstract are re-read from their sources by paper/tools/check_ledger.py through paper/pass13_numbers.csv).
Run from the repository root:  python paper/tools/pass13_edits.py   (PASS13_ROOT overrides the root of the edited files)
Then: python paper/tools/check_ledger.py ; python paper/tools/abstract_words.py"""
import os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract_words as aw  # noqa: E402

ROOT = Path(os.environ.get("PASS13_ROOT") or Path(__file__).resolve().parents[2])


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


# ---- abstract: item k (0-based, in order of \item) is replaced by \replaced[id=P13]{new}{old active text}; item 2 (Question), the
# placeholder item and the keywords are kept as they are.
NEW = {
    1: r"Question: which penalty keeps the most biology at equal batch removal in one scVI (single-cell variational inference) model, and does any beat plain scVI or standard tools?",
    0: r"\textbf{Background:} Batch correction for single-cell data often adds an alignment penalty (a discriminator, a Wasserstein-1 critic or a distance between batches) to a generative model, but penalties have rarely been compared on one backbone.",
    2: r"Exploratory comparison on eight tasks of the single-cell integration benchmark: scVI plus the Jensen--Shannon discriminator, three Wasserstein-1 critics (reference, pooled, barycenter), maximum mean discrepancy (MMD) or the Sinkhorn divergence, each fitted at ten strengths from 0.1 to 3,000, with the decoder given the batch label or not and three seeds (100--102), beside plain scVI and standard tools. A planned fixed-strength run was not made; the two sweep strengths on either side of the matched batch-removal level were refitted with five new seeds (0--4).",
    4: r"With the batch-conditioned decoder, the largest collapse rate on a real task ranged from 27\% (12\%--46\%) for MMD to 73\% (54\%--88\%) for the discriminator (nominal 95\% intervals).",
    5: r"With the batch-conditioned decoder, a penalty at one strength drawn at random exceeded plain scVI's selection score on at most 3 of six real tasks, and the best of six strengths on 4 to 6; at six tries every penalty exceeded Harmony and sysVI on 5 of six real tasks.",
    7: r"\textbf{Conclusions:} Within these tasks, strength mattered more than the choice of penalty: penalties were near-tied at matched batch removal (160 of 195 within-pair differences under 0.02); a penalty improved on batch-conditioned plain scVI mainly when several strengths were tried and scored on the same data, an optimistic gain; and scANVI, trained with the cell-type labels that several biology metrics also use, scored higher than every penalty on most tasks. Code and checksums: github.com/kreid415/wasserstein-critic-deconfounding.",
}


def abstract_edit(tex):
    start = tex.index(r"\hd{Abstract}")
    end = tex.index(r"\end{enumerate}", start)
    block = tex[start:end]
    assert "id=P13" not in block, "the abstract already carries Pass 13 edits"
    parts = re.split(r"(?m)^(?=\\item )", block)       # parts[0] is the heading; parts[1:] the items
    assert len(parts) == 9, len(parts)
    for k, new in NEW.items():
        item = parts[k + 1]
        body = item[len(r"\item "):].rstrip("\n")
        old_flat = aw.active_text(body, []).strip()
        parts[k + 1] = r"\item \replaced[id=P13]{" + new + "}{" + old_flat + "}\n"
    return tex[:start] + "".join(parts) + tex[end:]


p = ROOT / "paper/outline.tex"
p.write_text(abstract_edit(p.read_text()))

outline = [
    # author definition
    (r"\definechangesauthor[name={Pass 12}, color=red]{P12}",
     r"\definechangesauthor[name={Pass 13}, color=orange]{P13}% abstract cut, consecutive table numbers, reference-batch justification; change log paper/pass13_changelog.md, numbers in paper/pass13_numbers.csv" + "\n"
     r"\definechangesauthor[name={Pass 12}, color=red]{P12}"),
    # tables print 1, 2, 3: no counter reset; every citation by label
    ("\\setcounter{table}{2}% Pass 12: number 2 stays free (Table 2 removed in Pass 4), so this float prints as Table 3 and the next as Table 4\n",
     "% Pass 13: no counter reset; the main-text tables print as Tables 1, 2, 3 (labels tab:1, tab:3, tab:4 keep their Pass 9 names)\n"),
    (r"\replaced{Table 1 holds real design facts; Table 2 is removed in Pass 4 (it listed the strengths chosen for the planned fixed-strength benchmark, which is not run)}",
     r"\replaced{Table \ref{tab:1} holds real design facts; the strength-range table of the planned fixed-strength benchmark, which is not run, was removed in Pass 4}"),
    (r" The number 2 is left free, so the tables print as Tables 1, 3 and 4 (renumbering is the first author's call), and the text cites them by label. Pass 12 (",
     r" Pass 13 numbers the main-text tables 1 to 3 (Table \ref{tab:1} the methods, Table \ref{tab:3} the ranking and Table \ref{tab:4} the placement table) and every reference to them is made by label. Pass 12 ("),
    (r"to the normal scVI loss (Table 1).", r"to the normal scVI loss (Table~\ref{tab:1})."),
    (r"the counterpart of Table 4): selection score per method", r"the counterpart of Table~\ref{tab:4}): selection score per method"),
    (r"(typeset after Table 4); per-task", r"(typeset after Table~\ref{tab:4}); per-task"),
    (r"\item \deleted[id=P4]{Table 2: pilot strength ranges", r"\item \deleted[id=P4]{Former Table 2: pilot strength ranges"),
    (r"\noindent\deleted[id=P4]{Table 2 (removed). Provisional,", r"\noindent\deleted[id=P4]{Former Table 2 (removed). Provisional,"),
    # item 4: the first author's margin todo on the reference batch
    (r"which we assign based on the highest cell-type entropy \todo{we need justification here}.\added{ The entropy rule",
     r"which we assign based on the highest cell-type entropy\added[id=P13]{ (the Shannon entropy of a batch's cell-type proportions, computed from the cell-type labels; ties go to the larger batch)}\deleted[id=P13]{ \textit{todo: we need justification here}}.\added{ The entropy rule"),
    (r"so that the target does not favour a few cell types. The chosen reference",
     r"so that the target does not favour a few cell types}\added[id=P13]{, which a merely large batch, the other obvious choice, may do because a few cell types can dominate it}\added{. The chosen reference"),
    # figure 4 caption: the intervals are nominal, not exact
    (r"percentage points with exact \added[id=P12]{nominal }95\% intervals (lower is better)",
     r"percentage points with \deleted[id=P13]{exact }\added[id=P12]{nominal }95\% intervals (lower is better)"),
]
edit("paper/outline.tex", outline)

edit("paper/supp/tableS6.tex", [(r"Table~3 gives the mean ranks", r"Table~\ref{tab:3} gives the mean ranks")])
edit("paper/supp/tableS7.tex", [(r"counterpart of Table~4 and of Figure~6", r"counterpart of Table~\ref{tab:4} and of Figure~6"),
                                (r"equal those of Table~4.", r"equal those of Table~\ref{tab:4}.")])
# the generator writes those two files: it must write the same text
edit("paper/tools/make_supp_tables.py", [
    (r'rf"(Supplementary Notes 5 and 6; Table~3 gives the mean ranks).")', r'rf"(Supplementary Notes 5 and 6; Table~\ref{{tab:3}} gives the mean ranks).")'),
    (r'(counterpart of Table~4 and of Figure~6)', r'(counterpart of Table~\ref{tab:4} and of Figure~6)'),
    (r'equal those of Table~4.', r'equal those of Table~\ref{tab:4}.'),
    ("must reproduce the cells of Table 4 in paper/outline.tex", "must reproduce the cells of the placement table (label tab:4) in paper/outline.tex"),
    ('The cond=1 table must equal Table 4 of the outline', 'The cond=1 table must equal the placement table (label tab:4) of the outline'),
    ('print(f"Table 4 reproduced:', 'print(f"placement table (tab:4) reproduced:'),
])

# the response quotes the numbers the PDF prints: Table 1 methods, Table 2 ranking, Table 3 placement
edit("revision/reviewer_response.md", [
    ("Table numbers are those the PDF prints: the\nnumber 2 is left free (the strength-range table was removed), so the ranking table is Table 3 and the placement table Table 4.",
     "Table numbers are those the PDF prints (Pass 13):\nthe main-text tables are numbered 1 to 3, Table 1 (methods), Table 2 (ranking) and Table 3 (placement); the strength-range table was removed."),
    ("Tables 4 and S7; Figure 6", "Tables 3 and S7; Figure 6"),
    ("(Res-2; Table 3)", "(Res-2; Table 2)"),
])
print("pass 13 edits written under", ROOT)
