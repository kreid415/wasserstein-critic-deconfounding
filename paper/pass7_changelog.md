# Change log, Pass 7: sweep Results and Discussion (`paper/outline.tex`, 2026-11-02)

Base: branch `paper` at 5983a6ea0d18081acf4a7af2704cf28f1c47138b (Pass 6). File edited: `paper/outline.tex`. New files: this log and `paper/pass7_numbers.csv`. `paper/refs.bib` is unchanged; no citations were added.

Reason for the pass: replace the available sweep-result prompts with descriptive prose, draft Disc-1 and Disc-2, fill Abstract Results items 4 and 5, and record the first author's minimum-reading decision. Replication results and placement against incomplete baselines remain pending.

Marking conventions: `P7` identifies new `\replaced[id=P7]` passages; the author colour is violet. Earlier changes within replaced passages were accepted before making the P7 replacement ("flattened"). Earlier deleted material disappears from these passages, and their accepted text becomes the old argument. Old bullet contents are retained as text rather than an `itemize` environment; old `\res` and `\tbd` commands become literal bracketed prompts. Citations in new old arguments are wrapped in `\mbox`. Earlier history remains in the base commit. No P7 command is nested inside another changes command.

The Methods edit splits its existing `\added` span around the one changed prompt. Supplementary Note 6's enclosing historical replacement was accepted before marking only the minimum-reading phrase. The accepted wording elsewhere in that note is identical to the base. This keeps the scientific changes limited to the two requested phrases without nesting changes commands.

Line numbers below refer to the saved file after this pass.

## Sources read (read only)

Analysis directory: `/home/kendall/experiment_data/wasserstein-critic-deconfounding/sweep_analysis/draft_8595a16/`.

| Source | What was checked |
|---|---|
| `paper_numbers.json` | Reported values and metadata: fits and failures, low-signal targets, ranks, biology readings, leave-one-seed-out standard errors, hypervolume, non-dominated shares, collapse rates and intervals, local-minus-global readings, label transfer, widths, best strengths, task and seed identifiers. |
| `tables/ranking_pairs.csv`, `low_signal_pairs.csv`, `mean_rank.csv` | Task-specific rank order, target coverage, task counts, maximum batch scores, and neighbouring readings. |
| `tables/secondary.csv` | Collapse reflects biology loss rather than numerical failure; defined windows, single-strength windows, and gaps between qualifying strengths. |
| `tables/curves.csv` | Curve endpoints and their batch and biology scores. |
| `tables/label_transfer.csv`, `label_transfer_curve.csv` | Matched-strength F1 and batch-score distances; endpoint label-transfer example. |
| `tables/counts.csv`, `baseline_coverage.csv` | Sweep totals, reading statuses, and incomplete baseline coverage. |
| `analysis_sweep.json` | Analysis commit, minimum readings, full-task and label-transfer scope, input counts. Its SHA-256 matches `_meta.record_sha256` in `paper_numbers.json`. |
| `figures/fig_tradeoff_cond1.png`, `fig_tradeoff_cond0.png`, `fig_ranking.png`, `fig_collapse.png` | Visual checks of overlap, different batch-removal ranges, return toward lower scores, biology loss, ranking orders, and collapse patterns. Figures were viewed, not changed. |
| `../wcds/docs/ANALYSIS_SPEC.md`, sections 2–9 and 12 | Reading and ranking rules, separate simulations, uncertainty, secondary definitions, and sensitivity scope. |
| `../wcds/docs/SWEEP_ANALYSIS.md` | Application to every sweep strength, immune extension fits, three seeds, minimum readings, full label-transfer scope, and incomplete baselines. |
| `CONSTRAINTS.md`; `paper/pass4_changelog.md`, `pass5_changelog.md`, `pass6_changelog.md` | Standing wording and content restrictions, especially SI-78 and SI-114; flattening convention, replication scope, and hypervolume terminology. |
| Task brief | The first author's decision to require at least two of the three leave-one-seed-out readings, and the protected editing scope. The saved analysis already has `min_readings = 2`. |

The analysis metadata identifies commit `8595a169ed853db61eaf54501fb42f6735dea911`, a clean analysis checkout, all tasks, and no label-transfer debug subset. No analysis, scoring, figure generation, or external literature search was run for this pass. No new literature claim or citation was introduced.

## Changes

| # | Section | Line(s) | Old text (short) | New text (short) | Marking | Source of added facts |
|---|---|---|---|---|---|---|
| 1 | Preamble | 15 | No P7 author | Violet Pass 7 author and comment naming this log | Preamble | Task brief |
| 2 | Abstract, item 4 | 43 | Mean-rank prompt | Pooled critic highest with batch-conditioned decoder; MMD highest with unconditioned decoder; their mean ranks and real-task counts | P7 replaced, flattened | `mean_rank.*`, `mean_rank_n_tasks.*`; same values in Res-2 |
| 3 | Abstract, item 5 | 44 | Secondary-results prompt | Largest real-task MMD collapse rates with intervals, per decoder; label-transfer order differs from matched-biology order | P7 replaced, flattened | `collapse_rate.*`, `collapse_ci.*`; same values and observation in Res-4 and Res-5 |
| 4 | Methods, ranking uncertainty | 136 | Minimum-reading `\tbd` | At least two of the three leave-one-seed-out readings must exist | P7 replaced between split earlier `\added` spans | Task brief; `analysis_sweep.json:min_readings`; `_meta.seeds` |
| 5 | Res-1 | 155 | Rule and result bullets | Completed fits, no failures, immune extensions, low-signal scores and gaps, every penalty reaching the target; baseline-count placeholder retained | P7 replaced, flattened | `n_fits.*`, `n_failed.*`, `pair.*`, `n_pairs_no_reading.*`; counts and low-signal tables; sweep documentation |
| 6 | Res-2 | 157–159 | Ranking-rule and result bullets | Two prose paragraphs: all mean ranks and task ranges, contributing task counts, first/last task patterns, neighbouring-reading examples and standard errors, a larger gap, separate simulation orders | P7 replaced, flattened | `mean_rank.*`, `mean_rank_n_tasks.*`, `rank.*`, `bio_at_bstar.*`, `loso_se.*`; ranking table |
| 7 | Res-3 | 169 | Curve-rule and result bullets | Overlap, shorter batch-removal range, endpoint loss of biology and batch removal, task dependence; no early termination from failed fits | P7 replaced, flattened | Trade-off figures; ranking and curve tables; failure counts |
| 8 | Res-4 | 178 | Secondary-rule and result bullets | Per-penalty hypervolume and non-dominated-share ranges across real tasks, and largest real-task collapse rate with its interval and task for each decoder; biology loss rather than numerical failure | P7 replaced, flattened | `hypervolume.*`, `nondominated_share.*`, `collapse_rate.*`, `collapse_ci.*`; secondary table |
| 9 | Res-5 | 180 | Local/global and label-transfer bullets | Local/global direction across real tasks, values for each penalty on large chromatin-accessibility, lung examples, mean F1 per penalty and decoder, matched-strength F1 and distance examples, strong-setting F1 example | P7 replaced, flattened | `local_minus_global.*`, `label_transfer_mean.*`, `label_transfer_f1.*`, `label_transfer_distance.*`; label-transfer curve table |
| 10 | Res-6 | 182 | Width-rule and result bullets | Per-penalty width ranges by decoder, zero-width and defined-window observations, grid-edge best strengths, and gaps within some windows | P7 replaced, flattened | `width_decades.*`, `best_strength.*`; secondary and curve tables |
| 11 | Disc-1 | 203 | Answer and secondary-summary prompts | Decoder-dependent answer, close neighbouring readings, varying task orders, secondary measures with different orders, exploratory limits | P7 replaced, flattened | Res-2, Res-4, Res-5; `_meta.tasks`, `_meta.seeds`. The separate replication prompt at line 205 is unchanged |
| 12 | Disc-2 | 207 | Mechanism prompt and editorial bullets | One hedged paragraph relating biology loss, strength widths, secondary measures, label transfer and local/global readings; no causal attribution to divergence alone | P7 replaced, flattened | Res-3–Res-6; unchanged Methods and Disc-4 statements about regularisation and update steps. The image-GAN and composition-shift prompts remain in the old argument, without a new unsupported literature or composition-shift claim |
| 13 | Supplementary Note 6 | 254 | Minimum-reading `\tbd` | At least two of the three leave-one-seed-out readings must exist | P7 replaced after accepting enclosing earlier markup | Task brief; saved `min_readings`; `_meta.seeds` |
| 14 | Numeric provenance | `pass7_numbers.csv`: 1–249 | No pass-specific numeric ledger | Header and 248 occurrence-level provenance rows | New file | Sources named in each row |
| 15 | Change log | This file | No Pass 7 log | Base, scope, reasons, conventions, every change, checks and deviations | New file | Saved diff and source checks |

## Numeric conventions and checks

- Every numeric literal in a P7 new argument has a ledger row on its final TeX line. Spelled-out numerical design/task counts are also included. The ledger covers the newly written text, not inherited numbers in accepted historical old arguments or unchanged surrounding text. It also records inherited figure, table, note, task and metric identifiers, sourced to `paper/outline.tex` at the base commit; these are not measurements.
- Each empirical value was rendered from a JSON key or a named CSV row, then read back from the saved manuscript and ledger. No new averages, tests, intervals, or analysis outputs were computed. A reported range selects existing task values as endpoints; a reported largest collapse rate selects an existing task's rate and its existing interval. Simulations are excluded from real-task summaries.
- Scores, differences, hypervolume and shares use three decimals; mean ranks two; collapse percentages and their interval endpoints zero; widths one. The grid strengths preserve their numeric values. The seed range is written beside the seed count.
- The only new arithmetic results are the low-signal gaps and the MMD-minus-reference biology difference. Each difference row names both source keys and retains both source values and their unrounded difference.
- Hypervolume ranges omit the low-signal immune pair; share, collapse and width summaries include it. Collapse rates are not pooled across tasks. Where maxima are tied, the text names a task attaining the maximum, without claiming it is unique.
- Matched-label-transfer means were copied from the saved means, not recalculated. The illustrative F1 values include their distances from the target. No incomplete-baseline comparison is made.
- All new numerical literals, source values and rendered precision passed a read-back check. The analysis-record hash matched its numeric metadata. Qualitative checks covered full-seed target crossings, local/global signs, biology-only collapse, and defined width windows.

## Not changed

- Background and all commented-out lines, byte for byte; the removed image-generation paragraph was not restored.
- Methods wording except its specified minimum-reading prompt.
- Res-2b, Res-7 and Res-8, including their rules and `\res` placeholders.
- Disc-1's replication-result placeholder; Disc-3, Disc-4, Disc-5 and Conclusions.
- Abstract items 6 and 7 and the abstract conclusion placeholder.
- Baseline completed/failed counts remain a `\res` in Res-1.
- Figures, figure captions, mock images, tables, Declarations, and `refs.bib`.
- Supplementary material except the minimum-reading phrase and enclosing markup acceptance in Note 6. In particular, Note 5's duplicated minimum-reading `\tbd` and its target-coverage question remain untouched because the brief permits only Note 6 to change.
- Existing fit totals elsewhere in protected Methods and supplementary text remain untouched; see the discrepancy below.

## Structural checks (no LaTeX run)

The saved TeX has balanced unescaped braces after comments are stripped, paired environments, no missing bibliography keys, and no reference without a label. Every P7 replacement has both closed arguments, no nested changes command, and protected citations in its old argument. The modified text passed a scan for the prohibited wording, and `git diff --check` passed.

The existing static checker reports two earlier nested changes, also present at the base commit: the Res-5 heading (base line 198, now 179) and the Figure 1 caption (base line 329, now 296). These were not introduced or edited by Pass 7. The caption is expressly protected. This is a structural source check, not a successful document build; no LaTeX engine was run, as instructed.

The accepted-text comparison verified that the protected regions are unchanged, and that the only accepted-text differences in Methods and the supplement are the requested minimum-reading phrases. The tracked diff can be inspected with `git diff 5983a6e -- paper/outline.tex`; the P7 new and old arguments also show the prose changes in the source.

## Deviations / open questions

1. **Sweep fit total:** the brief names 3,024 fits, but `paper_numbers.json:n_fits.total` is 3,033. `n_fits.immune` is 387; each other task has 378. `SWEEP_ANALYSIS.md`, section 3 item 1, identifies the extra immune batch-conditioned strengths for the discriminator, pooled critic and reference critic. Res-1 reports the artifact total instead of repeating 3,024. Protected earlier design counts still need reconciliation in a later pass.
2. **Baseline snapshot:** the brief says 302 baseline fits scored; the supplied record has `n_baseline_rows.in_analysis = 303`, `n_baseline_rows.missing = 145`, and no baseline failures. Baseline counts and placement remain placeholders, as requested; neither snapshot count is inserted into the manuscript.
3. **Unverified claims:** none were added. The sweep does not establish the proposed image-GAN analogy or a causal composition-shift explanation, so Disc-2 makes neither claim; their original prompts and citations are retained only in the flattened old argument. The requested replication and baseline-dependent conclusions remain unfilled.

## Lead edits after review (Claude Science, 2026-11-02)

The draft above was written by Codex: profile MANUSCRIPT_WRITER, model gpt-6-astra, effort high, commit 1dc2a03. The lead read the full diff, checked every number, and made four edits.

| # | Place | Old | New | Why |
|---|---|---|---|---|
| A | Disc-2, last sentence | "because regularisation and update steps were not varied independently, they do not identify the divergence itself as the cause of the differences" | "These patterns describe each penalty as it is usually trained, with each adversary's update steps and optimiser at its standard setting (Methods)." | Five critic steps and one discriminator step are each method's standard practice, not a confound (first author, 2026-11-02). |
| B | Abstract, Results item 5 (line 44) | largest MMD collapse rate per decoder | with the batch-conditioned decoder, the largest real-task collapse rate ranged from 27% (MMD) to 73% (discriminator), with intervals; the label-transfer order differed from the ranking at matched batch removal | Gives the range across penalties instead of one penalty's value. All numbers appear in Res-4. |
| C | Methods, sweep paragraph (line 120), in place in the Pass 4 text | "3,024 fits in all" | adds the nine fits at strength 0.03 for the discriminator, reference critic and pooled critic on immune with the conditioned decoder, from an earlier strength-extension round (tier12_v2/A1_ext_r4_round1); the analysis reads 3,033 fits; that pair is low-signal and not ranked | Resolves deviation 1: Res-1 reports 3,033 and Methods said 3,024. |
| D | Supplementary Note 4 (line 252), in place in the Pass 4 new argument | "3,024 in all" | the same nine fits, so the analysis reads 3,033 | Same as C. |

Ledger: the 13%, 4% and 31% rows of line 44 were removed. Rows were added for 73%, 54% and 88% (line 44) and for nine, 0.03 and 3,033 (lines 120 and 252).

Checks by the lead (script draft7/check_pass7_v2.py, run on the workstation against the analysis record):
- 233 ledger rows were recomputed from paper_numbers.json, analysis_sweep.json and the tables at the written rounding: 0 mismatches.
- 21 rows are inherited identifiers (figure, note and task numbers) and were not recomputed.
- A scan of every P7 new argument found 0 numbers without a ledger row.
- These qualitative claims were checked against the tables:
  - every ranked curve reaches b* (all statuses "bracketed"; n_pairs_no_reading all 0);
  - MMD has a lower collapse rate than the discriminator on every real task with the conditioned decoder;
  - local-minus-global is positive only on the large chromatin-accessibility task, for both decoders;
  - the lung conditioned barycenter end point (0.466, 0.397) lies below plain scVI (0.633, 0.722);
  - the Sinkhorn divergence's smallest real-task width is 1.0 decade;
  - MMD widths with the conditioned decoder are 4.477 decades on both chromatin-accessibility tasks and 0.0 on human-mouse immune.
| E | Supplementary Note 11, run counts (in place in the Pass 4 text) | "3,024 including 96 edge-strength fits (12 per task)" | adds ", plus nine strength-extension fits on the immune task (3,033 analysed)" | Same as C. |
| F | Res-2 | one `\replaced[id=P7]{}{}` holding two paragraphs | the first paragraph and the old text in `\replaced[id=P7]`; the second paragraph in its own `\added[id=P7]` after a blank line | The `changes` package cannot take a paragraph break inside its argument; the build failed with "Paragraph ended before \addcontentsline was complete". |
| G | Disc-4, update-step bullet (line 216) | "Regularisation ... and the number of adversary update steps were not varied: ... so a difference between penalties includes these settings and is not due to the divergence alone." (earlier pass; flattened, its older Pass 2 text is in git) | "Each adversary is trained with its standard settings: the critics take five update steps per encoder step with a gradient penalty, as is usual for Wasserstein critics, and the discriminator takes one, as is usual for discriminators (Methods; Supplementary Note 2). The penalties are therefore compared as they are normally run." | The update-step difference is each method's best practice, not a limitation (first author, 2026-11-02, who had said so before). |

Check: the Res-2 split and edits A to G leave every ledgered number on its line. The lead's number check was rerun on the final text (see the delivery note).
