# Change log: `docs/ANALYSIS_SPEC.md`, contrast design replaced by the ranking design (2026-10-23)

Only `docs/ANALYSIS_SPEC.md` and this file were written. Nothing is committed. No code or test was edited. Labels used here:
SI-nn = entry nn of the decision ledger (`CONSTRAINTS.md`); RD-A to RD-M = the first author's decisions A to M of 2026-10-22 and
2026-10-23, renamed because the letters A to C already name options in Appendix A; LOSO = leave-one-seed-out (repeat a reading five
times, each time without one of the five seeds); "pair" = one task with one decoder; b\* = the matched batch-removal target.

## 1. What was done to each section

Old numbers are those of the file at tag `prereg-tier12-v3`; the full map is in Appendix A.4 of the specification.

| Old section | Action | New section |
|---|---|---|
| Status paragraph | Rewritten (ranking design replaces the contrast design; file, script and tests are updated together before any main-benchmark score is read). The old two status paragraphs are moved, not dropped | top; A.5 |
| "Examined", R1-R4 note, "Words used" | "Examined" kept word for word; words list updated (penalty, pair, reading, reaches b\*, rank, LOSO, S) | top |
| 0 Scope | Table rewritten (ranking, per-penalty secondary analyses, label transfer, X6 and premise check dropped) | 0 |
| 1 Inputs | Kept. Contrast-specific lines replaced (six penalties instead of the reference-versus-discriminator pair; "never reaches b\*" now means bottom places); latent files added as input for label transfer; grid text marked as older than the amendment of 2026-10-22 | 1 |
| 2.1, 2.2 | Kept (reading unchanged); power sentence removed | 2.1, 2.2 |
| 2.3 Not evaluable | Replaced by low-signal and ranked pairs (RD-B, RD-C) | 2.3 |
| 2.4 Contrast and uncertainty | Split: failed runs kept; ranks (RD-A, C, D) and uncertainty (RD-E) are new | 2.4, 3, 4 |
| 2.5 Decoders | Kept | 2.5 |
| 3 Frontier area | Per penalty, descriptive; leave-one-seed-out standard error of the area removed (it fed only the pooled model) | 5 |
| 4 Combining datasets | Removed (pooled model, priors, grid integration, 1e-6 floor, accuracy ranges) | none; git at the tag |
| 5 Declarations | Removed (margin 0.01, 0.95 rule, Holm over six results) | none; git at the tag |
| 6 P2 collapse | Per penalty, task and decoder with Clopper-Pearson intervals; pooled difference (Agresti-Caffo, 5-point margin) and pooled rate removed | 6 |
| 7 Local versus global | Per penalty at the crossing; sign test and pooled model removed | 7 |
| 8 Single-lambda tables | Kept; difference from plain scVI added; Harmony, Scanorama, sysVI added (RD-I) | 10 |
| 9 Best-of-k | Kept; comparison with plain scVI and real-task mean added (RD-A, RD-H) | 11 |
| 10 Width | Kept, per penalty | 8 |
| (new) | Label transfer (RD-G) | 9 |
| 11 Sensitivity | Rewritten as a table for the ranking and per-penalty readings; prior-scale and dataset-structure analyses removed | 12 |
| 12 Not in this file | X6 and premise check dropped (RD-K); scaling checks kept as exploratory | 13 |
| 13 Replaced statements | Split into still-replaced and no-longer-applicable | 14 |
| 14 Implementation | Quantity list replaced by the new quantities (table); test list removed, no new test cases written | 15 |
| 15 Details | Pooling, declaration and accuracy items removed; ranking and label-transfer items added | 16 |
| 16 Limitations | Pooled-model items removed; items on ranks, bottom places, related tasks, step count added; Clopper-Pearson independence stated with coverage (RD-J) | 17 |
| Appendix A | Earlier entries kept as history; A.4 added (RD-A to RD-M, what happens to each earlier decision, section map); A.5 holds the old status paragraphs | A |

## 2. Where the paper outline and the decisions differ (the decisions were followed)

| Outline (`outline_31cadaf.tex`) | Specification |
|---|---|
| A pair is ranked only if no penalty is unreached and at least 4 of 5 LOSO readings exist (Methods; Note 6) | Pair is ranked whenever it is not low-signal; unreached penalties take the bottom places (RD-C); the 4-of-5 rule applies only to the standard error (RD-E) |
| Tie rule marked "to be decided" (Note 6) | Average ranks (RD-D) |
| Table 3 and Figure 2b show the range of per-task ranks and mean biology at b\* | Not defined by the decisions; open point 2 |
| Methods say the rules were fixed before any main score was read | Not stated in the specification: the ranking version is not frozen yet (open point 1) |

## 3. Open points (the decisions leave these undefined; nothing was invented)

Where the specification text needs a placeholder it says "open point n".

| n | Question | Options | Where in the specification |
|---|---|---|---|
| 1 | How is the ranking version frozen? The contrast version was frozen by tag `prereg-tier12-v3` (2026-10-16). Ledger entries for RD-A to RD-M are also missing | (a) new tag on the final ranking commit, after the first author reviews the diff and before any main score is read; (b) dated amendment entry in `PREREG.md` only | status paragraph, 15 |
| 2 | Mean rank when it cannot be formed or is thin: a decoder with no ranked real task; any minimum number of ranked tasks; outline columns "range over tasks" and "mean biology at b\*" | no mean ("not available"), no minimum, count shown (what the text says for the first two); alternatively a minimum such as 3 tasks. Range = smallest and largest per-task rank (no new rule needed). Mean biology = over tasks where the penalty reaches b\*, or omit | 3.2 |
| 3 | Standard error of a penalty that does not reach b\* on the seed-mean curve but does in at least 4 repeats (dropping a seed can raise a curve) | (a) not defined, since there is no reading (what the text says); (b) compute it from the repeats | 4 |
| 4 | A repeat where the four remaining plain-scVI runs put the anchor at or above b\* (needs one left-out run at least 0.04 below the mean when the batch gap is 0.02). The existing reading then returns "unreached_high", or a downward crossing | (a) keep the existing function: no reading in that repeat, bottom rank; (b) read at the anchor; (c) drop such repeats from the standard error and rank range | 4, 16.2 |
| 5 | Per-run sensitivity: how per-run readings give one reading and one rank per penalty when only some seeds reach b\* | (a) mean over seeds that reach it, any number from 1; (b) at least 4 of 5 seeds, otherwise bottom places (the contrast rule); (c) rank within each seed, average the five ranks | 12 |
| 6 | Label transfer details: (a) matched strength: R4's rule (no failed seed, may be a grid-extension point, ties to the smaller strength; the text follows it) or the six grid points only; (b) how held-out batches and five seeds combine: macro F1 per held-out batch then mean, or pool all predictions; which cell types are averaged when a type is absent from the held-out batch; mean and SD over seeds; (c) distance and vote weights of the 15-neighbour classifier (scikit-learn default is Euclidean, equal weights); (d) the cross-task summary that "not summarised" implies for low-signal pairs; (e) a task with no cell type left after dropping single-batch types | as listed | 9, 16.11 |
| 7 | Real-task mean of best-of-k at k above a task's number of configurations (a failed configuration shrinks N) | (a) average only tasks with N at least k, show the count; (b) stop the curve at the smallest N; (c) carry the task's E_N forward | 11, 16.8 |
| 8 | Status of follow-ups X3, X7, X8, X12, X15: all compare the reference critic with the discriminator and are not covered by RD-A to RD-M | keep as locked-later designs; re-specify for the six penalties; drop | 0, 13 |
| 9 | Collapse rate pooled over the six real tasks (it existed in the contrast version, Clopper-Pearson over 180 runs) was removed with the pooled model; RD-F lists per penalty, task and decoder only | leave out (text); add pooled counts as description | 6 |

## 4. Choices made while writing (wording, record, restatement; none changes what is run)

| Choice | Reason |
|---|---|
| Sections reordered and renumbered; RD labels | Brief allowed renumbering; letters A to M clash with Appendix A |
| Old status paragraphs moved to A.5, with a note that tag `prereg-tier12-v3` exists (made 2026-10-16) | Moved, not dropped; the old sentence "which has not been made" is wrong |
| Section 1 grid text left as written, with a dated note that the amendment of 2026-10-22 (SI-109 to SI-113) supersedes it | Brief says R1-R4 and the pilot are unchanged; the note keeps the text honest |
| Differences from plain scVI written as penalty minus plain scVI (S at the transfer strength; E_k against S of plain scVI for each k) | RD-A says "compared descriptively"; outline Res-7 shows differences |
| Distance from b\* shown for unmatched methods (plain scVI, `scvi_adv`) using their own batch score | RD-G asks for the distance beside each value |
| Local minus global = 2.5 (local - b\*) at the crossing, stated in section 7 | Follows from B = (3 local + 2 global)/5; derived by hand |
| Low-signal pairs: collapse, width, best-of-k and placement include them; area, ranking, local minus global and label-transfer summaries leave them out | RD-B names only ranking and a separate table; the rest keeps the contrast rules (those analyses either do not use b\* or already excluded them) |
| Trajectory-free biology applies to every analysis that uses C; failed-run removal to readings, ranks, standard errors, area and local minus global | Same scope as the contrast code |
| No standard error for area or local minus global | RD-E names readings only |
| Tool tie rule: smaller setting value wins | RD-I says "same rule as the penalties" |
| Section 15 quantity table also lists crossings and the five LOSO readings | Both were in the contrast record; they make the standard error reproducible |
| Dropped follow-up identified as X6 | Matches the matrix row X6 and the outline paragraph (spectral normalisation, R1 gradient penalty, 1/5/10 critic steps, three strengths) |
| Limitation 7 (step count and optimiser) rewritten for all six penalties | The experiment that controlled it is dropped |

## 5. Checks made

| Check | Result |
|---|---|
| Words "pre-registered" or "registered" in new text | none; "registered" remains only in history (A.1 Q2 and Q4 reasons, A.5) |
| Cross-references ("section n") in the new text | all point to existing sections (read by search, since a script check was refused) |
| Coverage figures 0.9995 and 0.914 | read from `experiments/analysis_numerics_20261014/first_review/p2_clustering_sim.json` |
| Tag `prereg-tier12-v3` | commit 10a1ca5, 2026-10-16 (`git log`, `git for-each-ref`); `ANALYSIS_SPEC.md`, `analysis_x1.py` and `tests/analysis` at HEAD were identical to the tag before this edit (`git diff` empty) |
| Rank example (1, 2.5, 2.5, 4, 5.5, 5.5) and bottom-places formula | sum 21; places 7 - f to 6 give 6, 5.5, 3.5 for f = 1, 2, 6 |
| Not done | no code run, no tests, no build; the specification is a text file |

## 6. Other files that now disagree with the specification (not edited)

- `docs/PREREG.md` lines 3 and 175 say tag `prereg-tier12-v3` "has not been made"; it was. Its section 5 (power, "underpowered") and
  the contrast amendments describe the superseded design; its entry of 2026-10-20 freezes the X6 rule (tag `prereg-x6-v1`, SI-104,
  SI-106, SI-107), which is dropped.
- `docs/PAPER_PLAN.md` lines 178-206 cite old section numbers (sections 4, 5, 8, 9) of the specification.
- `scripts/analysis_x1.py` and `tests/analysis/test_analysis_x1.py` still implement the contrast design and cite old section numbers.
- `CONSTRAINTS.md` has no entries for RD-A to RD-M.
- The paper outline's Methods paragraph on statistics and Notes 5 to 6 still carry the outline's own evaluability rule (section 2 above).
