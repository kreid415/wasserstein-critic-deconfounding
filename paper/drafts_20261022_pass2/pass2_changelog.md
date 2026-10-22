# Change log, second editing pass (`outline.tex`, 2026-10-22)

Base text = the first author's latest round (the file as received). Every change to existing text is tracked with `\added`, `\deleted` or `\replaced`; old citations inside deleted text are wrapped in `\mbox{}`; no `\todo` sits inside deleted text. Sources read: `outline.tex`, `refs.bib` (keys only), `methods_changelog.md`, `intro_changelog.md`, `ANALYSIS_SPEC.md`. Nothing else under `/home/kendall/experiment_data` was opened.

**Not compiled.** Shell access was denied in this session, so the file was never built. Checks done instead: text search for forbidden words, for `\citep` inside deleted text (all wrapped; 3 `\mbox{\citep}` lines) and for `\todo` (one match, line 103, the kept one in Methods; three `MOCK NEEDS REDRAW` comments, lines 312, 329, 338). After writing, I read back by eye, for balanced braces and matching cross-references, the title and abstract, Introduction paragraphs 6 to 9, Methods paragraphs 1 to 5 and the new paragraphs, Results, Discussion, the supplement outline, the figure list, Figures 2 to 6 and Tables 1 to 3 and the first lines of Table 4. This is a visual check, not a parse; the Figure 1 environment and the Table 4 body were unchanged and not re-read. Please build once (`tectonic outline.tex`) before anyone reads it.

## (a) Changes

| # | Location | What changed | Why |
|---|---|---|---|
| 1 | Title | Replaced "Wasserstein critics versus discriminators..." by "Benchmarking alignment penalties for single-cell batch correction in scVI". | Study is a benchmark, not critic versus discriminator (proposal; see (e) 2). |
| 2 | Abstract, sentences 1-6 | Rewritten to benchmark scope: penalties compared on one backbone; ranking at matched batch removal; secondary analyses per penalty (adds label transfer); placement at equal tries. "Pre-registered" removed; follow-up mention removed. Sentence 7 unchanged. | Decisions 1, 2, 3, 5. |
| 3 | Introduction, "supplementary note XXX" | Now "Supplementary Note 1" (loss equations). | First author's note. |
| 4 | Introduction, "Despite their promise..." | "adversarial methods" changed to "alignment penalties". The sysVI sentence stays specific to adversarial methods. | Cited sentences already in the text: Zhao 2019 (error bound grows as batches are aligned more tightly) and Wu 2019 (any divergence zero only for identical distributions) are not adversary-specific; the sysVI result is. Nothing new claimed. |
| 5 | Introduction, commented image-GAN paragraph | New one-sentence tracked paragraph with `\citep{lucic2018gans,kurach2019large}`: most GAN losses reach similar scores with equal tuning, so methods are compared at equal numbers of tries. Commented paragraph left as is. | First author's note ("short sentence of two"); decision 2 (no follow-up mention). |
| 6 | Introduction, "Controlled comparisons..." and "Batch-correction methods have been compared..." | Merged into one benchmark-framed paragraph: the benchmark sentences now open the first paragraph (new text), the critic-centred opener is replaced, the scIB-E, Li et al. and iDLC sentences and the search sentence are kept, and the second paragraph is deleted in one block (its scIB-E repeat is gone). | Duplicate scIB-E description; critic-centred framing. |
| 7 | Introduction, last paragraph | "training settings ... fixed ... differences come from the alignment penalty alone" changed to say that adversary update steps and optimiser follow each method's usual setting. | Table 1 and Note 2 show those differ by design, and the follow-up that could separate them is dropped (decision 2); "alone" would overclaim. |
| 8 | Introduction, two commented sentences marked "this should be in methods" | Content moved to Methods as new tracked text (rows 13, 15, 16); comments left untouched. | First author's note. |
| 9 | Methods paragraph 1 | Added clause: adversary update steps and optimiser follow each method's usual setting (Supplementary Note 2); added "(Figure 1a)". | "Only intended differences" is not true (methods_changelog C4); Figure 1 was never cited. |
| 10 | Methods, four notes "should be in supplement" | `% DONE:` line added under each: Note 2 (update steps, optimiser, hidden units, strength zero, adversary input, scoring), Note 1 (gradient penalty weight 10), Note 3 (posterior-mean embedding, all cells, unscaled), Note 11 and Table S5 (480 fits each, 3,000 in all, failure definition, 5 seeds). The content was added to those supplement bullets (rows 26-28, 32). | First author's notes. |
| 11 | Methods paragraph 2 (reference batch) | Candidate justification added after the kept `\todo`, marked `\tbd{confirm this justification}`. | First author's note. |
| 12 | Methods paragraph 3 | Missing full stop added. | Typo. |
| 13 | Methods paragraph 4 (pilot) | Ten-strength list replaced by "half-decade steps from 0.1 to 3,000" (list now in Note 4). Extension outcome (two extensions, 36 fits, 0.03/0.01, 10,000/30,000) replaced by `\tbd`. Added `\tbd` for final ranges and the six strengths per family. | Decision 4; settings grid to supplement. |
| 14 | Methods paragraph 5 | Li et al. trade-off sentence and "read biology where all penalties reach the same batch removal" added (moved note). Low-signal rationale no longer says "smallest difference we count as real" (now "cannot be placed reliably"); "never pooled" deleted; "comparisons made at b*" changed to "ranking". The pilot's two low-signal pairs replaced by `\tbd`. | Moved note; decisions 1 and 4. |
| 15 | Methods, new benchmark-analysis paragraph | Added (below the user's commented statistics paragraph, which is left as is): reading at b*, leave-one-seed-out intervals, evaluability, ranking per task and decoder, mean rank over six real tasks, simulations and low-signal pairs separate, no significance tests, selection score compared with plain scVI, rules fixed before scoring. | Decision 1; note on the statistics paragraph. |
| 16 | Methods, baselines paragraph | Grids, seed list, 408 count and sensitivity settings moved to Note 8 and Table S4 (kept as deleted text). Added the user's equal-tries sentence, a plain definition of a try, and the single-strength rule. | Excess detail; moved note; decision 3. |
| 17 | Methods, follow-up paragraph | Stays commented; `% DONE: dropped (first author's decision)` added. Two new tracked paragraphs state, in plain words, the rules for collapse rate, good-strength width, trade-off area, local versus global removal and label transfer (adds citation key `clopper1934use`, which exists in refs.bib). | Decisions 2, 3; moved note on collapse and width. |
| 18 | Results heading and labels | "10 paragraphs" to "8". Labels renumbered with tracked text: Res-5 to 4, Res-6 to 5, Res-7 to 6, Res-9 to 7, Res-10 to 8. | Two paragraphs dropped. |
| 19 | Res-1 | Counts of main fits (3,000, 80/40/480, 408) removed (now Table S5); "never pooled" replaced by "left out of the ranking"; Table S6 to S5. | Excess detail; decisions 1; renumbering. |
| 20 | Res-2 | Premise-check paragraph (4 bullets) deleted; replaced by the ranking paragraph (Figure 2, Table 3): rule, mean rank per penalty and decoder, per-task values with intervals, descriptive. Old Res-4 (reference minus discriminator, pooled model, Wilcoxon) deleted as a block. | Decisions 1, 3. |
| 21 | Res-3 | Only the cross-reference: per-metric curves now Figure S4. | Renumbering. |
| 22 | Res-4 (old Res-5) | Per penalty: trade-off area and collapse rate; pooled difference, margin and prior-sensitivity bullet deleted; rules shortened to pointers to Methods. | Decisions 1, 3. |
| 23 | Res-5 (old Res-6) | Local versus global per penalty without sign test or pooled model; label-transfer rule shortened; Note 10/Figure S4 to Note 9/Figure S3. | Decisions 1, 3. |
| 24 | Res-6 (old Res-7) | Only "Figure 5a" to "Figure 5". | Follow-up panel dropped. |
| 25 | Old Res-8 (follow-up) | Deleted in full (heading and four bullets). | Decision 2. |
| 26 | Res-7 (old Res-9) | Tool settings and seed list removed from the rule; result bullet now includes the difference from plain scVI. | Excess detail; decision 1 (selection score compared with plain scVI). |
| 27 | Res-8 (old Res-10) | "adversary" to "penalty". | Generalised to all penalties. |
| 28 | Discussion Disc-1 to Disc-4 | Disc-1 asks for the ranking answer and neighbouring-rank closeness; Disc-2 about the secondary analyses and the image-GAN comparison, premise-check bullet deleted; Disc-3 "alignment penalty"; Disc-4: power and Wilcoxon sentence deleted, nested-pair sentence kept, strength-grid sentence replaced by `\tbd`, and a new sentence that regularisation and adversary update steps were not varied. Disc-5 unchanged. | Decisions 1, 2, 4. |
| 29 | Supplement heading | Counts now Notes 1-11, Tables S1-S8, Figures S1-S5. | Renumbering. |
| 30 | Notes 1, 2, 3 | Added moved content (critic head size, barycenter and Sinkhorn solver settings, scvi-tools classifier size; by-design differences, training split, generator optimiser, epoch rule; posterior-mean embedding). Note 2: "pre-registered input pilots" replaced by "input pilots, whose switching criteria were fixed before they were run". | Moved notes; decision 5; solver settings. Values from `methods_changelog.md` section 5. |
| 31 | Note 4 | Full ten-strength list; all eight tasks; seeds and fit count, extension outcome, Sinkhorn refit and low-signal pairs replaced by `\tbd`; noise threshold spelled out. | Decision 4. |
| 32 | Note 5 | Pilot low-signal values replaced by `\tbd`; "below the margin" replaced; local versus global added to the low-signal exclusions (spec: its evaluable cells equal those of the main reading); follow-up use of matched strength deleted. | Decisions 1, 2, 4. |
| 33 | Note 6 | Pooled model, priors, probabilities, margins, Holm, power and "run once" replaced by the ranking rules and sensitivity analyses (reading rules from the spec; no tests). | Decision 1. |
| 34 | Notes 7, 8 | Note 7: pooling, margin and sign test removed; simulation strength rule and best-of-k reference added. Note 8: grids and sensitivity settings added. | Decisions 1, 3; excess detail. |
| 35 | Notes 9-11 | Follow-up note (old 9) deleted. Old 10 becomes Note 9 (label transfer only). Old 11 becomes Note 10, "rules fixed before scoring and deviations", dates replaced by `\tbd`. Old 12 becomes Note 11 with the main-benchmark count breakdown and the failure rule; pilot counts `\tbd`; follow-up and pre-check counts deleted. | Decisions 2, 4, 5. |
| 36 | Supplementary tables and figures | Table S5 (follow-up) deleted; old S6 becomes S5; new S6 (ranking detail), S7 (unconditioned single-strength table), S8 (secondary results per penalty). Figure S3 (premise check) deleted; S4, S5, S6 become S3, S4, S5; S1 and S2 re-described. | Decisions 1-3. |
| 37 | "Figures and tables" list | Rewritten for the new plan (Figures 2, 4, 5 and Tables 1-3 re-described); the "first and second to cut" notes reassigned (Figure 5, then Figure 1). | New figure and table plan. |
| 38 | Figure environments | Figure 2 slot reused for the ranking figure; Figures 4 and 5 re-described; Figures 1, 3, 6 untouched. Redraw comments added (list in (c)). | Figure plan. |
| 39 | Table 1 | Last column "Follow-up variants" changed to "Strength grid family (Table 2)"; Sinkhorn smoothing value moved to Table S2. | Decision 2; settings detail. |
| 40 | Table 2 | Caption starts "Provisional, from the earlier three-task pilot; to be replaced."; the old three-task description deleted; cell values kept. | Decision 4. |
| 41 | Table 3 | Primary-results table replaced by a ranking table (same nine columns; header and first column replaced, six rows added, `xx.xx` placeholders). | Decision 1. |
| 42 | Table 4 | Caption points to Supplementary Table S7 instead of a `\tbd`. | Numbering. |

## (b) Numbering map (old to new)

| Item | Old | New |
|---|---|---|
| Figure 1 | schematic | Figure 1 (unchanged) |
| Figure 2 | premise check | dropped; slot reused for the new ranking figure |
| Figure 3 | trade-off curves | Figure 3 (unchanged) |
| Figure 4 | forest plots, reference minus discriminator | Figure 4, per-penalty secondary analyses (content changed) |
| Figure 5 | (a) width, (b) follow-up | Figure 5, width only (panel b dropped) |
| Figure 6 | placement | Figure 6 (unchanged) |
| Tables 1, 2, 4 | | same numbers (Table 1 last column changed; Table 2 provisional; Table 4 caption pointer) |
| Table 3 | primary results | Table 3, benchmark ranking |
| Supplementary Notes 1 to 8 | | same numbers; Note 6 content replaced (pooled model to ranking rules) |
| Note 9 | follow-up | dropped |
| Note 10 | exploratory checks | Note 9, label transfer only |
| Note 11 | registration and deviations | Note 10, rules fixed before scoring and deviations |
| Note 12 | software, counts, checksums | Note 11 |
| Tables S1 to S4 | | same numbers |
| Table S5 | follow-up variants | dropped |
| Table S6 | run counts | Table S5 |
| (new) | | Table S6 ranking detail, S7 unconditioned single-strength placement, S8 secondary results per penalty |
| Figures S1, S2 | | same numbers (S2 re-described: ranking under alternative readings) |
| Figure S3 | premise-check detail | dropped |
| Figures S4, S5, S6 | label transfer, per-metric curves, scaling checks | Figures S3, S4, S5 |
| Results paragraphs | Res-1..10 | Res-1, Res-2 (ranking, from old Res-2 slot), Res-3, Res-4 (old 5), Res-5 (old 6), Res-6 (old 7), Res-7 (old 9), Res-8 (old 10); old Res-4 and Res-8 deleted |

## (c) Figures marked MOCK NEEDS REDRAW

| Figure | Panels to draw |
|---|---|
| 2 (new) | (a) heatmap of biology at b* (colour) and rank (number), six penalties by six real tasks, both decoders, not-ranked cells hatched; (b) mean rank over the six real tasks per penalty and decoder with range across tasks. The existing `fig2.png` is the old premise-check mock and must be overwritten. |
| 4 | Per-penalty dot plots, both decoders, one point per real task plus mean: (a) trade-off area; (b) collapse rate with exact 95% intervals; (c) local minus global at b*. |
| 5 | Single panel: width of the good-strength range per penalty, decoder and task; drop the old follow-up panel (b). |

Figures 1, 3 and 6 keep their content and paths.

## (d) Every `\tbd` added

| Where | Text |
|---|---|
| Methods paragraph 2 | confirm this justification |
| Methods paragraph 4 | number of extended ranges and their values, from the all-task pilot |
| Methods paragraph 4 | final ranges and the six strengths per family and decoder, from the all-task pilot; Table 2 shows provisional values |
| Methods paragraph 5 | low-signal task--decoder pairs, from the all-task pilot |
| Table 3 (12 cells) | tasks (real tasks not ranked) |
| Disc-4 | strength grids set on the eight tasks of the all-task pilot; say what the pilot showed |
| Note 4 | seeds and number of fits, from the all-task pilot |
| Note 4 | ranges extended, extension values and number of fits, from the all-task pilot |
| Note 4 | Sinkhorn pilot refit after the solver correction: confirm for the all-task pilot |
| Note 4 | the low-signal pairs of the all-task pilot |
| Note 5 | pilot low-signal pairs and their values, from the all-task pilot |
| Note 6 | rule for tied values |
| Note 10 | dates for the all-task pilot and the revised ranking analysis; confirm that no main-benchmark score was read before the ranking rules were fixed |
| Note 11 | fits and edge fits of the all-task pilot |

Removed `\tbd`s: Table 4's "S-number" (now S7); the follow-up pre-check outcome. Kept: abstract repository, literature search, Table 1 steps, two in Availability.

## (e) Unsure how to read, or needs the first author

1. **"This section below should be a short sentence of two before the result."** I read it as "a short sentence or two" and kept it where the commented paragraph sits, before the gap and study paragraphs. Other readings: place it at the end of the Introduction, or just before the Results. Moving it is a one-line change.
2. **Title.** I changed it because it named critic versus discriminator. Revert row 1 if you prefer to keep it for now.
3. **Header line (journal target)** left untouched as instructed. Its last sentence ("Tables 1--2 hold real design facts") is now out of date because Table 2 is provisional.
4. **"Each penalty's selection score is compared with plain scVI."** I tied it to the two placement analyses that already exist (one untuned strength per task; expected best of k tries, plain scVI as the one-try reference). No new analysis was added.
5. **Ranking rules not in the specification:** tie handling (marked `\tbd`); only the six penalties are ranked (plain scVI and tools are not read at b*); the evaluability rule now says "every penalty's curve reaches b*", where the specification names only the two arms of the old contrast; the mean rank uses the evaluable real tasks, not necessarily all six.
6. **Sensitivity analyses** named in Note 6 (per-run reading, biology without trajectory score, points with failed fits removed) come from the specification's list; the specification also calls the four scaling checks exploratory, which I cite as rank stability.
7. **Clopper--Pearson intervals** are kept (descriptive). The specification warns they treat the 30 fits of a cell as independent; drop them if you prefer.
8. **"Rules fixed before the main benchmark was scored"** appears in the Methods and Note 10 using your phrase. The old dates (4 and 16 October 2026) were removed because they belonged to the old rules; confirm the statement holds for the revised ranking analysis.
9. **Pilot design.** Methods still says "ten strengths ... across three seeds"; only outcomes were replaced. Confirm the all-task pilot uses the same grid and seeds.
10. **Entropy justification.** The highest-entropy batch does not strictly contain every cell type; the candidate sentence assumes it does, hence the `\tbd`.
11. **Input pilots (Note 2).** "Pre-registered" became "switching criteria were fixed before they were run", inferred from the old wording.
12. **scIB-E** is not spelled out; its expansion is not in the sources I was given.
13. **Words still in the file.** "Pre-registered", "registered" and "follow-up" occur only in your untouched `%` comment lines and inside `\deleted` or the second argument of `\replaced`; nowhere in live text.
14. **Length.** Hand count of the abstract (not by script): 191 words, under the 200 limit. The new Methods paragraphs were not counted; rough hand estimates are 215 words for the benchmark-analysis paragraph, about 230 for the baselines paragraph, about 170 for the first secondary-analysis paragraph and about 90 for the second, so the first three exceed the 100--150 target.
