# Pass 21 change log (2026-11-06; layout only, no tracked-change id)

Branch `paper`, starting point Pass 20 (95f44c4). The user asked for the main figures to be in the main text for now, to edit
the text with the figures at hand; the journal layout comes later.

## Change (`paper/tools/pass21_edits.py`, commits 4b68a7d and the next one, which only moves its call to top level)

Figures 1-6 move from the "Figures and tables" block at the end of `paper/outline.tex` into the text, each right after the
paragraph that first cites it. Each figure environment is moved unchanged except its placement option (`[t]` -> `[!htbp]`).
No word or number changes (checked: the set of non-blank source lines is identical apart from the placement option and one
comment line). Tables 1-3 stay at the end.

| figure | placed after | clean PDF page (before: 33-37) | tracked PDF page |
|---|---|---|---|
| 1 model and read-out | Methods, paragraph 1 (backbone) | 4 | 5 |
| 2 ranking at b* | Res-2 | 8 | 10 |
| 3 batch-biology curves | Res-3 | 10 | 12 |
| 4 secondary summaries | Res-4 | 11 | 14 |
| 5 good-strength widths | Res-6 | 13 | 16 |
| 6 placement | Res-7 (after its list) | 14 | 17 |

For BMC Bioinformatics the figures go back out of the text (BMC asks for figures as separate files, captions at the end of the
manuscript); reverse this pass then.

## Checks

- `check_tests.py --base 95f44c4 --src paper/tools -- paper/tools`: OK, no WARN; re-running `pass21_edits.py` on the 95f44c4
  outline reproduces the committed one byte for byte.
- `check_ledger.py` on the workstation with the Pass 21 outline: ledger check OK, 0 failures. Abstract 328 words.
- Build (doc-build-gate, tectonic 0.16.9) from the saved zip extracted into an empty folder: tracked 64 pages, clean 54 pages (was
  55), 13 of 13 figures each, no failing log class (2 underfull boxes, as before); no heading at a page bottom.
- Pages viewed: clean pages 1-18 (all main text) and tracked pages 1-18.

## Re-run, from the repository root

```
git checkout 95f44c4 -- paper/outline.tex
python paper/tools/pass21_edits.py
python paper/tools/accept_changes.py
python scripts/check_tests.py --base 95f44c4 --src paper/tools -- paper/tools
```
