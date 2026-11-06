# Pass 14 change log (author id P14, 2026-11-06)

Branch `paper`, starting point `origin/paper` at 5856fd5 (Pass 13). Tracked changes carry `[id=P14]` (violet). Pass 14 fills the
replication across training seeds from its result; nothing else in the text is changed.

## Source of every replication number

The final agreement result on branch `sweep-replication`, commit a475ea3:
`experiments/sweep_replication/agreement_result_with_extension/` (895 fits: the 855 planned fits plus the 40 fits of the rule's
extension step, run on the first author's decision of 2026-11-05). The folder `experiments/sweep_replication/agreement_result/`
at the same commit is the result before the extension step; it is used only for the "before the extension step" values. Both are
read with `git show` and checked against their `MANIFEST.sha256` by `paper/tools/make_replication.py`. The agreement rule and the
prediction were fixed before any replication score (tag `replication-prediction-v1`, commit 3a90d0f).

`paper/tools/make_replication.py` writes:
- `paper/supp/tableS9.tex`: Supplementary Table S9 (78 readings in 13 blocks, plus the four large gaps), generated from
  `replication_readings.csv`, `replication_task_readings.csv`, `replication_large_gaps.csv` and, for the sweep's plain scVI batch
  score, `paper/figure_data/curves.csv` (strength 0).
- `paper/replication_data/replication_numbers.json`: every replication value the text quotes, each with a description. The script
  asserts that what it recomputes equals the result files (mean tau-b, slope-adjusted mean tau-b, the rule's sign-flip p over
  pairs: 4 of 8192 patterns = 0.00048828125, equal to the summary file).

Per-task sign-flip sensitivity (defined after the rule; Methods says so): signs flipped per task, both decoders of a task together,
all 2^8 patterns over the eight tasks, statistic the mean tau-b over the 13 pairs. Computed by `make_replication.py`
(`sign_flip_p`): 1 of 256 patterns reaches the observed mean, p = 0.00390625 (printed 0.0039). It is the smallest value eight tasks
allow, because the summed tau-b is positive in all eight tasks.

## Text filled (outline.tex; all edits written by `paper/tools/pass14_edits.py`, numbers taken from the JSON above)

1. Abstract item 7 (19 words): the four large within-pair differences kept their sign; mean rank correlation 0.456, p = 0.00049.
   Abstract count by `paper/tools/abstract_words.py`: 344 words (limit 350); the stricter hyphen-split count is 366 (it was 346
   before this pass).
2. Res-2b, all five result bullets (fits and the extension step; readings beside the sweep; the rule's two criteria and the
   result before the extension step; the per-task sensitivity placeholder of Pass 12; near-ties and the slope-adjusted reading).
   The near-tie bullet says that the replication resolves within-pair differences of about 0.04, so it does not test the order of
   near-tied penalties.
3. Disc-1: the replication bullet and the closing sentence ("remains to be assessed" replaced).
4. Supplementary Notes 4 and 11: the 40 extension fits (15 on the workstation, 25 on the cluster; source: the README of the final
   result folder), 895 in all, 0 failed.
5. Writer's-outline header: one sentence naming Pass 14, its change log and ledger.
6. `paper/supp_items.tex`: `\input{supp/tableS9}` after Table S7; header comment and section title name Table S9.
7. `revision/reviewer_response.md`: the replication section (was "Pending") written from the same numbers; item 4 of "What changed
   overall" and R2.4 point to it; the header names Pass 14.

Wording: "replication across training seeds" throughout; the text says the rule was fixed before any replication score and that
the replication uses new seeds, not new data. Both results (before and after the extension step) are reported: before it, one of
the four large gaps could not be read (the barycenter critic on the small chromatin-accessibility task with the unconditioned
decoder had not reached b*), the other three kept their sign, mean tau-b 0.426, p 0.00049.

## Ledger

`paper/pass14_numbers.csv` (84 rows, the Pass 12 format): every number of the new text in `outline.tex` and in the response's
replication section, with a source pointer. Primary values point to the committed result files at a475ea3
(`git:a475ea3:experiments/sweep_replication/...`); derived values (plain scVI difference, median and maximum reading difference,
rank counts, the per-task p) point to `paper/replication_data/replication_numbers.json`. `paper/tools/check_ledger.py` now reads
pass14 by default.

## Re-run, from the repository root (Python: /home/kendall/.claude-science/conda/envs/scvi-api/bin/python, Python 3.11.15, pandas 2.3.3)

```
git checkout 5856fd5 -- paper/outline.tex paper/supp_items.tex revision/reviewer_response.md   # Pass 13 text
python paper/tools/make_replication.py        # Table S9 and replication_numbers.json (needs commit a475ea3 in the repository)
python paper/tools/pass14_edits.py            # writes the three files
python paper/tools/make_replication.py --check
python paper/tools/abstract_words.py          # 344; exit 1 above 350
python paper/tools/check_ledger.py            # exit 0
python -m pytest paper/tools                  # includes test_make_replication.py and the Pass 14 reproduction case
```

No random numbers are used (exact enumeration of all sign patterns).

## Placeholders left

`\res`: none. `\tbd`: 7 active, none about the replication (large language model tools and versions; ethics wording; release tag
and commit of the submitted version; competing interests; funding; authors' contributions; acknowledgements), plus 2 inside TeX
comments.
