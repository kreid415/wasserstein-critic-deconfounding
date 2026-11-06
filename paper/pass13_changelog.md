# Pass 13 change log (author id P13, 2026-11-06)

Branch `paper`, starting point `origin/paper` at fe4b19e (Pass 12 plus the supplement items merge). Tracked changes carry `[id=P13]`
(orange). Not touched, by instruction: the replication placeholders (abstract item 7, Res-2b with the per-task sign-flip sensitivity
`\res`, the Disc-1 replication sentences, the response letter's replication section).

Every text edit of `outline.tex`, `supp/tableS6.tex`, `supp/tableS7.tex`, `tools/make_supp_tables.py` and
`revision/reviewer_response.md` is written by `paper/tools/pass13_edits.py` (each old text must occur exactly once, otherwise it
stops). Figure 4 is redrawn by `paper/figure_scripts/fig4.py`.

Re-run, from the repository root (Python of the figure environment: Python 3.11.15, matplotlib 3.11.0, pandas 2.3.3, numpy 2.4.6):

```
git checkout fe4b19e -- paper/outline.tex paper/supp revision/reviewer_response.md paper/tools/make_supp_tables.py   # Pass 12 text
python paper/tools/pass13_edits.py          # writes those files
python paper/figure_scripts/fig4.py         # Figure 4 (PNG 300 dpi + PDF); the PNG hash is listed in paper/figures/MANIFEST.sha256
python paper/tools/make_supp_tables.py --check-table4   # regenerates Tables S6 and S7; must equal the committed files
python paper/tools/abstract_words.py        # abstract word count; exit 1 above 350
python paper/tools/check_ledger.py          # exit 0 = every ledger value re-read from its source
```

## 1. Abstract (BMC Bioinformatics limit: 350 words)

Counting method (`paper/tools/abstract_words.py`): the abstract as printed with the tracked changes accepted (new text of
`\replaced`, text of `\added`, `\deleted` text and TeX comments dropped), the `\res`/`\tbd` placeholders dropped, `\textbf`,
`\emph` and the keywords line removed, then a split on white space (the count of `wc -w`). `--` is joined to one word
(for example `100--102` is one word); a stricter count that also splits at hyphens and dashes is printed beside it.

| | white-space words | hyphen-split words |
|---|---|---|
| Pass 12 (fe4b19e), placeholder text excluded | 454 | 479 |
| Pass 12, placeholder text included (the Pass 12 note says "about 483") | 482 | |
| Pass 13, placeholder text excluded | 325 | 346 |

Room for the replication sentence: 350 minus 325 is 25 words by the white-space count (limit given: at most 20 words), so the
abstract stays at or below 345 words with that sentence added. The hyphen-split count is 346 now and would exceed 350 with a
20-word sentence; BMC's submission system counts words by white space, which is the primary method here.

What was cut (all still in the main text, Methods or supplement): the half-decade-step wording and the list of baseline names
(now "plain scVI and standard tools"); the rationale of the replication ("to check the sweep's readings on new fits") and the
sentences "Every comparison over the strength range uses the sweep fits" and "the replication's strengths were chosen from the
sweep's batch scores, never from its biology scores" (Methods, replication paragraph, keeps both); the label-transfer clause of
the collapse-rate sentence; the clause that scANVI "was exceeded only on pancreas, by at most 0.010" (and with it the numbers
`0.010` and the word pancreas); "six strengths per penalty" of the planned run; "estimating the earth mover's distance", "deep" and
"direct" in the Background; "which makes" and "for scVI users" in the Conclusions. No number was changed or added. Item 2 (Question)
loses "alignment" and "at equal tuning effort".

Caveats kept (checked in the printed text, page 1 and 2 of the tracked PDF): exploratory ("Exploratory comparison on eight tasks");
nominal 95% intervals ("(nominal 95% intervals)"); settings chosen and scored on the same data ("a penalty improved on
batch-conditioned plain scVI mainly when several strengths were tried and scored on the same data, an optimistic gain"); scANVI
uses labels ("scANVI, trained with the cell-type labels that several biology metrics also use"); observations within these tasks
("Within these tasks"). Note: the same-data caveat sits in the Conclusions sentence, not next to the "best of six strengths" counts in
the Results sentence of item 6; adding it there costs 7 words (332 in all).

Ledger (`paper/pass13_numbers.csv`, 40 rows, column `supersedes`): the abstract rows A01 to A43 of `pass12_numbers.csv` (except A36,
A39, A40, A44, whose locations are outside the printed abstract) are copied with their selectors. Rows whose number is still in
the abstract point to its new text (`abstract:paper/outline.tex::...`); three rows (A11 "six" of the planned run, A37 "pancreas",
A38 "0.010") have no location because the number left the abstract, and their value is still re-read. `check_ledger.py` now looks up an
`abstract:` anchor in the printed (final) abstract only, because the raw source also holds the struck old text, and skips the old
location of a superseded Pass 12 row (the value of that row is still re-read).

## 2. Figure 4 panel b title

`paper/figure_scripts/fig4.py`: "collapse rate, exact 95% interval" is now "collapse rate, nominal 95% interval" (and the same
change in the script's docstring). The longer title ran into the panel letter c (the script's text-overlap check reported it), so
the three panel titles are set at 7 pt (the other figures use 8 pt) and the spacing is unchanged. Before the edit the unedited script
reproduced the committed PNG byte for byte (hash in the manifest), so the figure differs from the Pass 12 one only in the title
text and size. Caption: "exact" is struck, "nominal" (Pass 12) stays: "with nominal 95% intervals". `paper/figures/MANIFEST.sha256`: the
two fig4 lines are updated (the PDF hash changes at every run because the file carries a creation time stamp). Viewed with
`host.view_image`: titles of a, b, c, no overlap.

## 3. Consecutive table numbers

- `\setcounter{table}{2}` removed: the main-text tables print as Table 1 (methods), Table 2 (ranking, label `tab:3`) and Table 3
  (placement, label `tab:4`). The labels keep their Pass 9 names so that no other file changes.
- Every active reference now uses `\ref`: outline Methods "(Table~\ref{tab:1})", the supplement-item lines for Tables S7 and S10,
  the writer's-outline note; `supp/tableS6.tex` and `supp/tableS7.tex` (3 references) and their generator
  `tools/make_supp_tables.py` (the generator writes the same text; regenerating gives files identical to the committed ones, and the
  positive control "placement table reproduced: 13 rows, 78 cells identical" passes). `supp_items.tex` and the other supplement files
  have no table number.
- Struck (deleted) text keeps the numbers of its time (Tables 3 and 4 in old mock text, Table 2 for the removed strength-range
  table); the two struck lines that name the removed table now read "Former Table 2" so that it is not mistaken for the new Table 2.
- `revision/reviewer_response.md` quotes the printed numbers: line 14 (Table 1 methods, Table 2 ranking, Table 3 placement), line 31
  "Tables 3 and S7", line 133 "Table 2"; its other table mentions are Table 1 and the Supplementary tables.
- Check with `pdftotext` on a clean-text build of the same sources (the change macros redefined to show the new text only; the
  package option `[final]` fails with an xkeyval error on this file at fe4b19e too): printed captions "Table 1:", "Table 2:", "Table 3:", then
  Supplementary Tables S10, S6, S7. Each of the 21 main-text citations in the clean text names the printed table of that subject: Table 1 five times (writer's-outline
  note, Pass 13 note, Methods, table list, caption), Table 2 five times (Pass 13 note, Res-2, table list, caption, supplement S6
  caption), Table 3 ten times (Pass 13 note, Res-9 three times including "prints as a tie", the S7 and S10 items, table list, caption,
  S7 caption twice) and "Tables 2 and 3" once (writer's-outline note). Evidence: `paper/pass13_table_check.txt`.

## 4. The `\todo{we need justification here}` in Methods

The sentence: "The reference method aims to align all batches with a single reference batch, which we assign based on the highest
cell-type entropy". Justification taken from the code docstring of `wcd/data.py::select_reference_batch` (maximum cell-type Shannon
entropy, ties broken by size; the reference defines the target for every other batch; the largest batch was rejected as the
alternative because a large batch can be dominated by a few cell types; the earlier default, the alphabetically first batch, picked
small batches). Written into the sentence: "(the Shannon entropy of a batch's cell-type proportions, computed from the cell-type
labels; ties go to the larger batch)" and, after the Pass 10 sentence that the rule prefers evenly represented cell types, "which a
merely large batch, the other obvious choice, may do because a few cell types can dominate it". No number added. The `\todo` is removed;
as the file's convention requires, its text is shown as the struck `\deleted[id=P13]{\textit{todo: ...}}`.

## 5. The margin note near "line 116"

There is exactly one `\todo` in the sources (the one above; `\usepackage{todonotes}` stays loaded and nothing else calls it), so the
margin note at the reference-batch wording is the same note and is removed with item 4. It is settled by the coverage counts of the
text (every cell type in five of the eight tasks; 10 of 17, 14 of 19, 3 of 4 in the others; ledger rows R01 to R06 re-read them from
`paper/analysis_pass10/reference_coverage.json`) and by the entropy-rule rationale. No margin note is left in the PDF.

## Checks run

- Build: the saved LaTeX zip extracted into an empty directory and built with tectonic through doc-build-gate (`dg_check_sources`
  with 13 images, `dg_build_tex`): 43 pages, 13 of 13 figures, 1 run, no failing warning class (2 underfull boxes below the
  threshold, the same classes as before). The page-ink check flags only the page-number line at 17 to 18 pt from the bottom edge
  (32 pages, identical in the Pass 12 build with 33 pages); it passes at a 15 pt margin; the Pass 12 build also had ink at the edge
  of a page from the margin note, which is gone. Pages viewed: all pages whose text changed against the Pass 12 build
  (1 to 20, 24, 28, 34, 36); pages 5, 6, 8, 9, 11, 12, 13, 15 and 17 changed only in where a few words break and were viewed on
  contact sheets (three pages per image), the others one by one.
- `python paper/tools/check_ledger.py`, `python -m pytest paper/tools` and `scripts/check_tests.py`: see the commit message.
