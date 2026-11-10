# Pass 24 change log (2026-11-09, tracked id P24, pink)

Branch `paper-pass23-wip`, starting point the end of Pass 23 (24f2665). Pre-submission edits from two referee reviews (astra,
Codex gpt-6-astra; fable, Claude Code claude-fable-5-1), merged in `merged_review.md` (sections A-D). The first author approved
every recommendation on 2026-11-09 except D1 (label transfer and the direct biology check read some penalties at a strength
past the first crossing of b*), which is being recomputed. Res-5, the Conclusions item on the direct biology check (iii),
Supplementary Note 12, Supplementary Table S10, Supplementary Figure S3 and the label-transfer paragraph of Methods are
therefore not edited in Part a (checked byte for byte against 24f2665). Part b below completes D1 with the first-crossing data.

All text edits are made by `paper/tools/pass24_edits.py` on the 24f2665 text (word-level tracked changes in `paper/outline.tex`,
in-place edits of the supplement files it lists); `paper/outline_clean.tex` is written by `paper/tools/accept_changes.py`.
Generated tables are changed through their generators. Numbers that are new to a location, or moved, are in the ledger
`paper/pass24_numbers.csv` (checked by `paper/tools/check_ledger.py`, which now reads it by default). No number is new to the
paper: each is in the paper or in a data file.

Word counts (printed text, `pass24_edits.budget`: Background to the end of Conclusions, comments removed, figure environments
counted apart): main text 7,533 -> 5,630 words (limit 5,700; Res-5, 513 words, waits for D1); main-figure captions 803 -> 434
(limit 450). Abstract (`paper/tools/abstract_words.py`): 333 -> 324 words.

## Decisions (merged review section A)

- D2 (both reviewers). "Strength mattered more than penalty choice" is gone from the Abstract, Disc-3 and the Conclusions. The
  narrower statement replaces it: the penalties kept similar biology at matched batch removal and differed mainly at strong
  settings. 0.025 and 0.213 are now called the range between the highest and the lowest penalty (Abstract, Res-3); Disc-3 adds
  that the strongest settings reached different batch scores and depend on the top of the strength grid.
- D3 (both). Res-6 is one prose paragraph with a lead that states the finding. The counts are given once with "scored above"
  and the near-tie caveat; the own-best count (chosen and scored on the same data) and the Rule bullet (a repeat of Methods) are
  gone; the unconditioned counts are left to Supplementary Table S7. New sentences: the scvi-tools adversary scored above plain
  scVI on pancreas, immune and ATAC large; scANVI, trained with the labels, scored above every transferred-strength penalty on
  every real task except pancreas (Supplementary Table S14). The Abstract and Conclusions (iv) now say "every real task except
  pancreas" and rest on Res-6.
- D4 (both). Figure 3 (dot summary) became Supplementary Figure S7 (it repeated Figure 2a and Supplementary Figure S6). Figure 4
  was split: the frontier panels stay as main Figure 3; the hypervolume panels became Supplementary Figure S8, whose axis now
  runs to 0.22 so that every point shows (the old limit, 0.13, hid 6 of the 66 real-task points; maximum 0.202). Renumbering:

  | Before (24f2665) | After (Pass 24) | Script before -> after |
  |---|---|---|
  | Figure 3, dot summary | Supplementary Figure S7 | fig3.py -> figS7.py |
  | Figure 4a-b, Pareto frontier | Figure 3 | fig4.py -> fig3.py |
  | Figure 4c-d, hypervolume | Supplementary Figure S8 | fig4.py -> figS8.py |
  | Figure 5, collapse rate | Figure 4 | fig5.py -> fig4.py |
  | Figure 6, placement | Figure 5 | fig6.py -> fig5.py |

  Every reference, caption, the figure lists, `make_figures.sh` (Figures 1-5), `make_supp_figures.sh` (S1-S8) and
  `paper/figures/MANIFEST.sha256` follow. The collapse-rate axis label reads "collapse rate (%)" instead of "percentage points".
- D5 (both). The abstract quotes the Res-4 medians (34% discriminator, 3% MMD) with no interval, instead of each penalty's
  single-task maximum with its nominal interval.
- D6 (astra). Disc-4: the ranking depends on how scores are aggregated; min-max scaling and z-scores put the Sinkhorn
  divergence first with the batch-conditioned decoder, and z-scores put the discriminator first with the unconditioned decoder
  (read from `paper/figure_data/scaling_check_ranks.csv`: mean ranks 2.00 against the pooled critic's 2.20, and 2.83 against
  MMD's 3.00 under z-scores). "Used for no conclusion" is removed from the Figure S5 caption and Supplementary Note 6.
- D7 (fable). One clause in the Methods (views paragraph) and two Disc-4 sentences: in all 11 ranked real pairs the barycenter
  critic's highest batch score is the highest common batch score, so it fixes b*, the hypervolume cap and the collapse range; the
  cap and the range were chosen after the scores had been read (Supplementary Note 10).
- D8 (astra). Res-6, Disc-3 and Disc-4 qualify the tool results by the settings tested (Harmony and Scanorama at 10
  dimensions). New Supplementary Table S15 (written by `make_supp_tables.py`, `make_s15`) prints the existing default-size
  sensitivity settings from `single_methods.csv`: against batch-conditioned plain scVI, Harmony with 50 principal components
  scored above it on 3 of six real tasks, Scanorama with 100 dimensions on 1 and PCA with 50 components on 1 (Harmony and
  Scanorama at their transferred 10-dimension settings: 1 and 0). Cited in Methods, Res-6, Disc-4 and Supplementary Note 8.

## Factual and consistency fixes (section B)

- Res-4: "whose strongest setting sets that score" -> "whose highest batch score sets that limit".
- Methods and Supplementary Note 1: the fool loss is no longer called bounded; it is a cross-entropy fool loss, a JS-type
  objective that is not bounded above and is not the JS divergence itself.
- Methods paragraph 1: plain scVI and the six penalties share the backbone; the standard tools use their own models. Figure 1
  caption likewise; the Disc-4 backbone bullet says "the penalty fits".
- Supplementary Note 8: scANVI is "the only method here trained with the labels" (reference-batch choice and several biology
  metrics also use them).
- Methods, Res-4, Supplementary Note 7 and the Table S8 caption: "the range of Figure 4" is gone; the collapse subset is the fits
  at strengths whose mean batch score is at most the highest common batch score, strong settings whose score fell back below it
  included.
- Cross-task mean macro F1: deleted from Supplementary Note 9, Table S11 (generator `make_supp_supp-analysis.py`, summary block
  and caption sentence) and Note 5; recorded in Note 10 "Analyses removed".
- Table S7 caption (generator): scANVI, sysVI, Harmony, Scanorama and PCA "were not run in the two decoder settings", not "have
  no decoder"; Figure~6 -> Figure~5. Table S2 (generator `make_supp_supp-methods.py`): the posterior-mean adversary input is
  the six penalties'; the scvi-tools adversary reads a posterior sample. Table S13 caption: the conditioned-immune strength 0.03
  of the discriminator, reference and pooled critics.
- Figure 1 caption: "dotted arrows" -> "dashed arrows".
- Supplementary Note 7: "Table~\ref{tab:4}" -> "Supplementary Tables~\ref{tab:4} and S7"; Figures 4--6 -> Figures 3 and 4,
  Supplementary Figure S8 and, for placement, Figure 5. Supplement list: "Figure S4" -> "Figures S4a and S4b".
- Res-2b: 895 = the 855 planned fits plus 40 of the extension step.
- Editing-history notes removed from lists and captions: "the former main-text Table 1/2/3" (Tables S12-S14), "the former
  main-text Figure 3" (Figure S6 list item) and "(the main-text Figure 3 until Pass 23)" (Figure S6 caption), and the Figure 1
  item's "First to move to the supplement if pages run short".
- "3 to 5 of six ... +0.006" now appears in the Abstract, Res-6 and the first Conclusions item only (Disc-3 paragraph and bullet
  and Conclusions (i) no longer repeat it); "1, 0 and 1" in the Abstract, Res-6 and the first Conclusions item.
- kBET cited (buttner2019test) at its first main-text mention; iDLC marked as a preprint; a List of abbreviations before the
  Declarations.
- Stale scope statements: "We report no significance tests" -> "The ranking has no significance test" (Methods); Note 5 lists
  what leaves out the low-signal pairs (ranking, hypervolume, label-transfer summaries) and says the direct biology check keeps
  them; Note 9 says the standard tools are not part of label transfer but the direct biology check reads their recall and
  confusion (Note 12); Methods "(gaps in Results)" -> "(gaps in Supplementary Note 5)"; the Table S8 caption's Part B wording.
- Not done here (Res-5 items wait for D1): "below" -> "at or below" and plain scVI's own unmatched embedding.

## Length (section C)

Moved, not deleted (the target note already held most of it; what it lacked was added):
- Replication rule detail (reading, one-sided test over 2^13 patterns, its assumptions, the two later sensitivity analyses, no
  precision calculation, seeds reported apart) -> Supplementary Note 13 "Rule"; its three "(Methods)" pointers now point there.
- Sweep history and fit-count arithmetic (378 per task, 3,024 + 9 = 3,033, the pilot and its rules) -> already in Supplementary
  Note 4; dated rule history (22-23 October, baseline score files 6-12 October) -> already in Supplementary Note 10; machines and
  scoring sites -> already in Note 11; baseline fit counts (448, 447) -> Note 8 and Res-1.
- Reference-batch justification and coverage (five of eight tasks; 10 of 17, 14 of 19, 3 of 4) -> new Supplementary Note 1
  paragraph "Reference batch", with its source `paper/analysis_pass10/reference_coverage.json`.
- Metric inventory (batch and biology metric lists) -> already in Supplementary Note 3; the Methods keep kBET (cited) and the
  unscaled selection score.
- Res-2 rank enumeration (per-penalty ranges and the tasks each penalty led or trailed) -> Figure 2 and Supplementary Table S12.
- Res-3 example of the barycenter critic's short range on ATAC large (0.674 against 0.683 to 0.834) -> Supplementary Note 7.
- Disc-1 replication bullet (p values per task and family, tau_b per pair, MMD on pancreas) -> Supplementary Note 13 and Table S9.
- Disc-4 fit counts of the collapse intervals over all strengths (30, 33, 36) -> Supplementary Note 7.
Shortened without moving: Background last paragraph (replication design, repeated in Methods), the Methods backbone, data,
sweep, replication, matched-level, ranking, baselines, views and AI-use paragraphs, Res-2b to one sentence, Res-3b, Res-4,
Res-6, Disc-1, Disc-3, the Disc-4 exploratory and backbone bullets, the scalarization bullet, Conclusions (ii), and all five
main captions. The abstract is three paragraphs (Background, Results, Conclusions) with no numbered list, no "Question:" label,
no seed numbers and no URL; `abstract_words.py` now reads the paragraphs after the struck list. The Res-6, Disc-4 and Disc-5
bullets are full sentences. The \pp{label}{lead} and itemize structure is kept.

## Removed claims

Numbers no longer printed anywhere in the paper; each ledger row that pointed at one is retired by a Pass 24 row pointing here.

- Abstract: each penalty's single-task maximum collapse rate with its nominal interval, 10\% (1\%--30\%) for MMD (D5; the
  discriminator's 56\% (31\%--78\%) stays in Res-4).
- Res-4: the intermediate median collapse rates (Figure 4 shows every median as a black tick; Table S8 has every rate):
  24\% for the Sinkhorn divergence, 23\% for the barycenter critic, 21\% for the pooled critic and 11\% for the reference critic with the batch-conditioned decoder;
  11\% for the discriminator and 10\% for the reference critic with the unconditioned decoder.
- Disc-1 bullet: Kendall's tau_b was slightly negative in 2 pairs (pancreas and the human--mouse immune task, both with the
  batch-conditioned decoder). Supplementary Note 13 gives the range of tau_b and the 11 positive pairs, Table S9 each pair's value.
- Supplementary Note 9: the cross-task mean macro F1 over the real tasks that are not low-signal (five with the batch-conditioned decoder, six with the unconditioned decoder), with its block in Table S11.
- Res-6: the own-best count ("at each task's own best strength, the optimistic bound, they did so on 5 to 6 of six") and the
  unconditioned range ("2 to 6 of six"); the per-penalty values stay in Supplementary Tables S14 and S7.
- Conclusions (ii): the strengths chosen on other task families "lay between 0.1 and 100 across penalties" (a pooled range is
  not a recommendation, because strength scales differ); the per-penalty transferred strengths stay in Res-6.

## Ledger

`paper/pass24_numbers.csv`, 128 rows, read by `check_ledger.py` by default. 90 rows supersede earlier rows whose anchor this
pass broke or struck: 78 move with their number to its new place (abstract, Methods, Res-2b to Res-6, Figure 3 caption,
Supplementary Notes 1, 7, 10 and 13), each with the source and selector of the row it replaces; 12 are retired to the Removed
claims above. 38 rows are new: the abstract medians and the 0.04 threshold, the Res-2 mean ranks, Res-2b's 855 and 40, the
Res-6 transferred strengths and default-size counts, Note 8's sensitivity counts, the D7 count of 11 pairs, the 10 dimensions of
Harmony and Scanorama, the Methods 855 and 3,033, the Res-4 "three penalties" and the Table S13 strength 0.03. Rows already
pointing at struck text before this pass (29) are left as they were. The six heavily rewritten passages (Methods sweep
paragraph, Res-2b, Res-3, Res-3b and Res-4 bodies, Figure 3 caption) are tracked as one replacement each, so their new text is
contiguous in the source and its anchors resolve.

## Tests and checks

New test files: `paper/tools/test_pass24_edits.py` (the script reproduces every file it writes from 24f2665; Table S15; Table
S11 without the cross-task block; the paragraph abstract) and `paper/tools/test_pass24_figures.py` (the five new or renamed
figure scripts; written by a claude-sonnet-5-5 helper agent, checked by the lead). Three existing tests fail by construction of
the approved scope and were not edited (rule: no edits to existing tests):
- `test_pass23_edits.py` compares Pass 23's output with the working tree, which Pass 24 changes (fix: compare with 24f2665).
- `test_fig6.py` runs `figure_scripts/fig6.py`, the placement figure, now `fig5.py` (fix: point it at fig5.py). It still passes
  only because fig6.py could not be deleted; it fails once fig6.py is deleted.
- `test_supp_analysis.py` expects 17 "Plain scVI &" rows in Table S11 (16 blocks plus the summary row removed by item 8; fix: 16).
`fig6.py`, `fig6.png` and `fig6.pdf` are still on disk: file deletion was refused in this session; delete them. New files under
the root .gitignore patterns need `git add -f`: `paper/supp/tableS15.tex`, `paper/pass24_numbers.csv`, `paper/figures/figS7.png`,
`paper/figures/figS8.png`.

## Part b: label transfer and biology at the first crossing (9 November 2026, P24)

The first author changed the strength rule after reading the first results. Both label transfer and the direct biology check
now use, of the two fitted strengths bracketing the first crossing of b*, the one whose seed-mean batch score is nearer b*
(ties to the smaller strength), or the upper one when the lower is plain scVI. The earlier rule chose the strength nearest b*
on any crossing, sometimes far past the first crossing. Supplementary Note 10 records this deviation and where both readings
are kept. Methods and Notes 9 and 12 give the rule explicitly. The main ranking is unchanged.

The new check's tables, summary, record and original manifest are copied unchanged from
`biology_check_20261109_first_crossing/run`, code commit `8073db6`, into `paper/figure_data/biology_check_first_crossing/`.
The earlier `summary.csv` and `summary_by_method.csv` are in `paper/figure_data/biology_check_nearest_any_crossing/tables/`.
`paper/tools/biology_check_table.py` checks the strengths-file hash and each penalty's selected strength, then regenerates
`res5_biology.json`. Its `table_lines` function supplies Table S10 to `pass24_edits.py`; the latter still reproduces the
committed text from 24f2665 with tracked author P24. Table S11 and Figure S3 were regenerated from the rebuilt label-transfer
tables. The Figure S3 caption names the first-crossing strength. README and both figure-data and figure manifests follow.

Headline changes, read from the archived tables and the two label-transfer summaries:

| Readout against plain scVI | Earlier, nearest on any crossing | First crossing |
|---|---:|---:|
| Discriminator median label-transfer macro F1, batch-conditioned | -0.006 | 0.000 |
| Discriminator median label-transfer macro F1, unconditioned | -0.060 | -0.002 |
| Unconditioned discriminator below plain scVI / lowest among penalties, real tasks | 6 / 6 | 5 / 3 |
| Discriminator mean purity, batch-conditioned | -0.052 | -0.013 |
| Discriminator mean purity, unconditioned | -0.077 | -0.001 |
| Sinkhorn mean purity, batch-conditioned | -0.022 | -0.004 |
| Sinkhorn mean purity, unconditioned | -0.033 | 0.000 |
| Discriminator mean rare-type purity, batch-conditioned | -0.107 | 0.000 |
| Rare-type purity losses, task by penalty by decoder comparisons | 23 / 48 | 21 / 48 |
| Other penalties' median label-transfer differences, range over decoders | -0.016 to +0.001 | -0.006 to +0.001 |
| Label-transfer versus biology order, median Kendall tau_b | 0.067 | -0.067 |
| Label-transfer batch-score distance from b*, maximum / median | 0.026 / 0.004 | 0.073 / 0.005 |

The former finding that the discriminator consistently lost more biology does not hold across decoders at the first crossing.
Res-5 says this plainly, keeps the scANVI and Harmony comparisons, and leaves per-cell-type detail to Note 12. Disc-2 and
Conclusions now support only the small average changes observed here. Note 12 compares both purity readings for every penalty
and decoder and both discriminator label-transfer medians. It also reports the first-crossing result for lung ionocytes.

Ledger: current numbers, including every cell of Tables S10 and S11, have Part b rows in `pass24_numbers.csv` with paper-branch
sources. The old `P23T` rows now point to the preserved any-crossing summary and are retired by later rows. The old Pass 9
biology rows have stable `P9B` ids and retirement notes; their sources are the earlier paper-branch summary tables, or the
original committed check for removed per-type details. Part b retirement rows identify these historical records explicitly.
The moved Note 9 and Note 12 seed claims are superseded. The ledger's general CSV selector now supports means, so Table S10's
seed spreads and baseline averages are checked directly from the copied tables.

Printed length: Res-5, including its lead, 275 words; main text 5,439 (limit 5,500); main captions 434 (limit 450).
No existing test or test configuration was edited. Added tests check summary reproduction, the selected-strength checks,
Table S10, the mean selector and Figure S3. No fits, new analyses, external writes or changes under `revision/` or `paper/build/`.


Part b validation: `python paper/tools/accept_changes.py`, `python paper/tools/accept_changes.py --check`,
`python paper/tools/check_ledger.py`, `python paper/tools/abstract_words.py`, `python -m pytest -q paper/tools`, and
`python scripts/check_tests.py --base b622d1d --src paper/tools --src paper/figure_scripts -- paper/tools` all exit 0.
The suite has 123 passing tests; the coverage check reports no WARN. Abstract: 324 words. Data, original-run and figure
checksums match; `git diff --check` exits 0.

### Part b file inventory

Changed existing files:

- `paper/figure_data/MANIFEST.sha256`
- `paper/figure_data/README.md`
- `paper/figure_scripts/figS3.py`
- `paper/figures/MANIFEST.sha256`
- `paper/figures/figS3.pdf`
- `paper/figures/figS3.png`
- `paper/outline.tex`
- `paper/outline_clean.tex`
- `paper/pass23_numbers.csv`
- `paper/pass24_changelog.md`
- `paper/pass24_numbers.csv`
- `paper/pass9_numbers.csv`
- `paper/supp/figures.tex`
- `paper/supp/note10.tex`
- `paper/supp/note12.tex`
- `paper/supp/note5.tex`
- `paper/supp/note9.tex`
- `paper/supp/note9_table.tex`
- `paper/tools/check_ledger.py`
- `paper/tools/make_supp_supp-analysis.py`
- `paper/tools/pass24_edits.py`

New files (including ignored CSV files, which the lead must add):

- `paper/figure_data/biology_check_first_crossing/MANIFEST.sha256`
- `paper/figure_data/biology_check_first_crossing/biology_check.json`
- `paper/figure_data/biology_check_first_crossing/summary.md`
- `paper/figure_data/biology_check_first_crossing/tables/atac_large_confusion.csv`
- `paper/figure_data/biology_check_first_crossing/tables/atac_large_fit_types.csv`
- `paper/figure_data/biology_check_first_crossing/tables/atac_large_fits.csv`
- `paper/figure_data/biology_check_first_crossing/tables/atac_small_confusion.csv`
- `paper/figure_data/biology_check_first_crossing/tables/atac_small_fit_types.csv`
- `paper/figure_data/biology_check_first_crossing/tables/atac_small_fits.csv`
- `paper/figure_data/biology_check_first_crossing/tables/immune_confusion.csv`
- `paper/figure_data/biology_check_first_crossing/tables/immune_fit_types.csv`
- `paper/figure_data/biology_check_first_crossing/tables/immune_fits.csv`
- `paper/figure_data/biology_check_first_crossing/tables/immune_hum_mou_confusion.csv`
- `paper/figure_data/biology_check_first_crossing/tables/immune_hum_mou_fit_types.csv`
- `paper/figure_data/biology_check_first_crossing/tables/immune_hum_mou_fits.csv`
- `paper/figure_data/biology_check_first_crossing/tables/lung_confusion.csv`
- `paper/figure_data/biology_check_first_crossing/tables/lung_fit_types.csv`
- `paper/figure_data/biology_check_first_crossing/tables/lung_fits.csv`
- `paper/figure_data/biology_check_first_crossing/tables/pancreas_confusion.csv`
- `paper/figure_data/biology_check_first_crossing/tables/pancreas_fit_types.csv`
- `paper/figure_data/biology_check_first_crossing/tables/pancreas_fits.csv`
- `paper/figure_data/biology_check_first_crossing/tables/sim1_confusion.csv`
- `paper/figure_data/biology_check_first_crossing/tables/sim1_fit_types.csv`
- `paper/figure_data/biology_check_first_crossing/tables/sim1_fits.csv`
- `paper/figure_data/biology_check_first_crossing/tables/sim2_confusion.csv`
- `paper/figure_data/biology_check_first_crossing/tables/sim2_fit_types.csv`
- `paper/figure_data/biology_check_first_crossing/tables/sim2_fits.csv`
- `paper/figure_data/biology_check_first_crossing/tables/summary.csv`
- `paper/figure_data/biology_check_first_crossing/tables/summary_by_method.csv`
- `paper/figure_data/biology_check_nearest_any_crossing/tables/summary.csv`
- `paper/figure_data/biology_check_nearest_any_crossing/tables/summary_by_method.csv`
- `paper/figure_data/res5_biology.json`
- `paper/tools/biology_check_table.py`
- `paper/tools/test_pass24_biology.py`
