# Pass 16 change log (author id P16, 2026-11-06)

Branch `paper`, starting point `origin/paper` at 0e79c46 (Pass 15). Tracked changes carry `[id=P16]` (green). User decision of
2026-11-06: add the family-level sign-flip sensitivity analysis to the replication result. The committed agreement rule and its
verdict are unchanged; the analysis is labelled as defined after the rule, like the per-task version.

## Analysis (`paper/tools/make_replication.py`)

Statistic: the mean over the 13 ranked pairs of Kendall's tau-b (the rule's statistic), one-sided, with the signs of whole families
flipped together; exact, all 2^F patterns, no random numbers. A pattern counts when its mean reaches the observed mean.

Families (`FAMILY` in `make_replication.py`; 6 among the 13 pairs):

| family | tasks | pairs | source |
|---|---|---|---|
| pancreas | pancreas | 2 | `scripts/prereg_rules.py` `TASK_FAMILY` (4 real-task families) |
| lung | lung | 2 | same |
| immune | immune, immune_hum_mou | 3 | same; all 33,506 immune cells are in immune_hum_mou (`docs/ANALYSIS_SPEC.md` section 1) |
| atac | atac_small, atac_large | 4 | same; the large task holds all 11,270 cells of the small one, 3,618 of them under identical names (corrected in Pass 17; `docs/ANALYSIS_SPEC.md` section 1 counts only the name matches; `docs/PREREG.md` glossary 'dataset family', SI-73) |
| sim1 | sim1 | 1 | not in `TASK_FAMILY`; own family (see below) |
| sim2 | sim2 | 1 | not in `TASK_FAMILY`; own family (see below) |

Simulations: the repository defines a family as a group of related real tasks (`docs/PREREG.md`), `TASK_FAMILY` has no simulation
entry, and `docs/ANALYSIS_SPEC.md` reports sim1 and sim2 per task, never pooled; the two share no data (12,097 and 19,318 cells,
`docs/PAPER_PLAN.md` task table). Each is therefore its own family. `docs/PAPER_PLAN.md` calls the simulations "a separate family"
as a category against the real tasks; if that is read as one family of two, there are 5 families, 2^5 = 32 patterns and p = 1/32 =
0.03125 (computed once for this note, not in the paper, not in the ledger).

Result (read back from `paper/replication_data/replication_numbers.json`): 6 families, 64 patterns, 1 reaches the observed mean
(the summed tau-b is positive in all 6 families), p = 0.015625, the smallest value 6 families allow. The previous agent's 1/64 is
reproduced. Independent check (separate code, same committed readings at a475ea3): tau-b recomputed with `scipy.stats.kendalltau`
from the ranks equals the stored tau for all 13 pairs; brute-force count of the family flips gives 1 of 64.

Tests (`paper/tools/test_make_replication.py`): a hand-checkable case (atac_small +1, atac_large -1, pancreas +1: family flip counts
2 of 4, per-task flip 4 of 8), the fail-loud check for a task without a family, and the real value in the end-to-end test (the
printed line gives per-task 1 of 256 and per-family 1 of 64). `paper/tools/test_pass11_12_edits.py` gains the Pass 16 row.

## Text (`paper/tools/pass16_edits.py`, exact-substring edits; each old text must occur once)

| where | edit |
|---|---|
| Methods, replication paragraph (after the per-task sensitivity analysis) | one sentence clause: the family-level analysis, 2^6 patterns over six families, defined after the rule, approximate |
| Res-2b, sensitivity bullet | two sentences: p = 0.0156 beside the per-task p = 0.0039, 1 of 64, smallest p six families allow, defined after the rule |
| Disc-1, replication bullet | `and p = 0.0156 per task family`; `both values` becomes `all three values` (approximate) |
| header comment, preamble | author id P16 and a pointer to this log |
| `revision/reviewer_response.md` | "What changed overall" item 4 and the replication section (R3.2): the family-level p, approximate |

Not changed: the abstract (it states neither sensitivity analysis; 328 words, hyphen-split count 348, limit 350, unchanged), the
supplement description items in `outline.tex` and `paper/supp/`, the committed rule and its verdict.

## Ledger and checks

- `paper/pass16_numbers.csv`: 10 values (p, hits, patterns, families, families with positive sum), each re-read from
  `replication_numbers.json`; `paper/tools/check_ledger.py` (now includes Pass 16): 0 failures; a copy with 0.0157 written fails (positive control).
- `scripts/check_tests.py --base 0e79c46 --src paper/tools -- paper/tools`): OK, 63 passed, no WARN.
- Clean copy: `python paper/tools/accept_changes.py` (committed with the text edits).
- Build (doc-build-gate `dg_check_sources` + `dg_build_tex`, engine tectonic, from the extracted zip of `outline.tex`, `outline_clean.tex`,
  `refs.bib`, `supp_items.tex`, `supp/`, `figures/`; zip made from commit a11ac79): tracked 46 pages, clean 40 pages, 13 of 13 figures
  each, failing log classes none (2 underfull boxes each, as in Pass 15). Page-ink check: 34 of the 46 pages of the tracked PDF report ink 17 pt
  from the bottom edge (18 pt is the limit; the nearest ink is the page number), the same 34 of 46 as the Pass 15 tracked PDF; the clean PDF
  was flagged the same way and not counted; no blank page was reported. Changed
  text viewed as page crops (tracked pages 5, 9, 12; clean pages 4, 7, 9); not every page was viewed.
  The doc-build-gate helpers ran on the workstation (the zip stayed there; copying it to the Claude Science workspace stalled).
- Response PDF: `pandoc ... --pdf-engine=tectonic`, 8 pages, TeX underfull warnings only.
- `paper/build/*.pdf` and `paper/MANIFEST.sha256` updated (`sha256sum -c`: all OK).

## Re-run, from the repository root (Python: /home/kendall/.claude-science/conda/envs/python/bin/python)

```
git checkout 0e79c46 -- paper/outline.tex revision/reviewer_response.md
python paper/tools/pass16_edits.py
python paper/tools/accept_changes.py            # writes paper/outline_clean.tex
python paper/tools/abstract_words.py            # 328 words; hyphen-split 348
python paper/tools/make_replication.py --check  # replication_numbers.json incl. p_family
python paper/tools/check_ledger.py
python -m pytest paper/tools
```
