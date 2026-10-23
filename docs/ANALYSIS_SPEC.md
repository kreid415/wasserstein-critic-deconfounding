# Analysis specification for the X1 endpoints: descriptive ranking design (rules; not yet frozen)

*Status, 2026-10-23. The ranking design of this file replaces the contrast design: one primary contrast (the reference critic's
biology at b\* minus the discriminator's), pooled over the real tasks in a hierarchical model with a 0.01 margin, a 0.95 posterior
rule and Holm's correction over six results, plus a co-primary frontier area and a collapse-rate comparison. The first author
decided the replacement on 2026-10-22 and 2026-10-23 (decisions RD-A to RD-M, Appendix A.4; their ledger entries are still to be
added to CONSTRAINTS.md). The contrast version of this file, of `scripts/analysis_x1.py` and of `tests/analysis/` is the one at
the tag `prereg-tier12-v3` (commit 10a1ca5, 2026-10-16); `git show prereg-tier12-v3:docs/ANALYSIS_SPEC.md` prints it.
**This file, the script and its tests must be updated together before any main-benchmark score is read.** Until then the script
and its tests still implement the contrast design and must not be run on any X1 or X13 score (SI-78). No main-benchmark score has
been read (the first author's statement; SI-108 records that none has been examined and that the repository's blinding records end
on 2026-10-16). Which tag or other step freezes this version is open (change log, open point 1). Unchanged by the replacement:
the score definitions, the decision rules R1-R4 and the pilot (PREREG sections 1-4 and 9), b\*, the low-signal rule and the curve
construction, the transfer-strength rule, the best-of-k estimator, the definition of "examined" and the provenance requirements.
The earlier status paragraphs, the options not chosen and the reasons of the contrast design are in Appendix A.*

**Examined (user decision SI-78, 2026-10-15, amended 2026-10-16).** A score is examined when a person reads its value or an analysis script uses it; checks that print only counts or flags do not count. This applies to every X1 and X13 score, of any arm.
Of readers, only a person counts: a value shown to an AI agent is disclosed, not examined; use by an analysis script still counts as examined (user, 2026-10-16, NB-20261016-14, NB-20261016-24); the listing of the displays found by the audits as "shown to the agent only" (SI-88; PREREG section 9) follows from this definition.
The user states that nobody read an X1 or X13 score value (NB-20261015-15). The agent's own audit of 2026-10-14 (NB-20261014-21,
made by the agent that ran the checks) is narrower: it searched the agent's session logs for reads of `tier12_v2/X1_local/scores`,
the printed output of the scripts that read them and two log statements, and it cannot show that nobody opened a file. PREREG
section 9 lists the X1 and X13 scores that already exist, the displays and automated reads found, and how they are classified (the
displays are listed as shown to the agent only, not as examined; SI-88).

*The decision rules R1-R4 (PREREG sections 2-4) do not change. Where PAPER_PLAN section 5 and this file differ, this file applies;
PAPER_PLAN has a dated pointer (section 14).*

Words used below. *Penalty* = one of the six alignment penalties: `discriminator` (Jensen-Shannon discriminator), `reference`,
`pooled` and `barycenter` (the three Wasserstein-1 critics), `mmd` (maximum mean discrepancy, a kernel method) and `sinkhorn`
(debiased Sinkhorn divergence, an optimal-transport distance); the code calls them arms. *Strength* (lambda) = a penalty's weight.
*Run* (fit) = one model trained with one seed. *Point* = one strength of one penalty on one task and decoder, i.e. 5 runs (seeds
0-4). *Pair* = task x decoder (the earlier version and the script's record call it a cell). *B* = batch score, *C* = biology score
(PREREG section 1, unscaled scIB means). *S* = selection score 0.4 B + 0.6 C (section 10). *b0, c0* = mean B and C of plain scVI
(lambda = 0) in the pair. *b\** = the matched batch-removal target of rule R4: half-way from b0 to b_common, the highest mean B
that all six penalties reach at failure-free strengths (extension strengths counted). *Reading* = a penalty's biology at b\*
(section 2). *Reaches b\** = the reading exists. *Rank* = place among the six penalties in a pair (section 3). *LOSO* =
leave-one-seed-out (section 4). Elsewhere the words differ: PREREG rules R2 and R3 use *cell* for a task x decoder x arm combination,
and PREREG section 9 calls a task x decoder pair a task-by-decoder block; a cell of the tissue is called a single cell. R0 to R5 in
sections 0, 8, 11 and 13 are PAPER_PLAN's research questions; R1 to R4 elsewhere are the decision rules (PREREG sections 2-4). Labels:
SI-nn is entry nn of the decision ledger (CONSTRAINTS.md) and NB-yyyymmdd-nn an entry of the lab notebook; RD-A to RD-M are the
ranking decisions of 2026-10-22 and 2026-10-23 (Appendix A.4); I0nn and D-nn are findings of the final review of 2026-10-13 and of
its recheck of the design documents; AN-nn and DOC-nn are findings of the rigour review of 2026-10-14 (analysis and documents
parts); AN-Rnn and DOC-Rnn are findings of the independent re-check of the fixes (2026-10-14), fixed on 2026-10-15; Qn is a
question put to the user on 2026-10-14 (Appendix A). PREREG section 9 has a glossary of the other labels and terms.

## 0. Scope and timing (user decision SI-68, question Q1; changed by RD-A, RD-G, RD-K)

| Analysis | Data | Data exist |
|---|---|---|
| Ranking of the six penalties by biology at matched batch removal, with LOSO standard errors and rank ranges (sections 2-4) | X1 | X1 |
| Per penalty: trade-off area, collapse rate, local minus global batch removal (sections 5-7) | X1 | X1 |
| Per penalty: good-strength width (R3, X1 part; section 8) | X1 | X1 |
| Per penalty: label transfer (section 9; part of X10, moved from exploratory to reported) | X1 latents, task labels | X1 |
| R5 placement against plain scVI and standard tools: single-strength tables, best-of-k (sections 10-11) | X13, X1 | X1 |
| X10 downstream biology other than label transfer | X1 latents and models | X1 |
| P2 composition dose-response | X3 | after R4-x1 |
| P3 batch-count scaling | X8 (NO-GO as designed: I015, I077, I078, D-07) | after R4-x1 |
| P4 reference choice | X7 (NO-GO as designed: D-06, I089) | after R4-x1 |
| R3 factor effects | X12 (A3 done except its 36 Sinkhorn rows, which are to be redone, SI-57) | after R4-x1 |
| X15 KL warm-up sensitivity | X15 | after R4-x1 |
| Dropped (RD-K): follow-up X6 (regularisation, optimiser, update steps); premise check R0 | - | - |

Every analysis of X1 or X13 data in sections 2-11 is fixed by this file before any main-benchmark score is read (status
paragraph), except the X10 readouts other than label transfer, which are exploratory (no confirm/refute claim). The follow-up
analyses (X3, X7, X8, X12, X15) are fixed later, together with the X7/X8 design fixes, while X1 runs, before R4-x1 runs and before
anyone examines an X1 result (SI-68's own words; "examined" as defined in the status paragraph, SI-78); their status under the
ranking design is open (change log, open point 8). (Q1's options and reasons: Appendix A.1.)

## 1. Inputs and shared definitions (fixed earlier; no choice)

- Score rows, failures and the X1 manifest are read through `prereg_rules.outcomes` (PREREG section 1: a row neither
  scored nor failed stops the analysis; one scorer commit and CPU type per input, with the SI-54 equivalence). The
  analysis runs once, after all 3,000 X1 rows and the 408 X13 baseline rows are scored or failed (PREREG section 9,
  entry of 2026-10-14).
- B, C as PREREG section 1: C includes trajectory conservation on immune and immune_hum_mou (SI-28); the
  trajectory-free C is a sensitivity analysis (SI-38, section 12).
- b0, b_common, b\* and `bstar_low_signal` per pair are read from the frozen R4 stage-x1 record (PREREG section 4), and the
  analysis refuses unless `prereg_rules.r4_stage` gives the same values from the same X1 rows (section 16.1; re-check findings
  AN-R02, DOC-R03). It therefore needs every row that R4 used, extension rows included.
- Points: lambda = 0 (arm `none`) and the six grid strengths of each penalty's family and decoder (`x1_grid` of the R1
  record given to `scripts/analysis_x1.py` with `--r1-record`). R4-x1 extension rows are not used as points (PREREG section 4;
  SI-55). For every penalty except Sinkhorn the six lambda are those of `docs/prereg/r1_record.json` as it stands at tag
  prereg-tier12-v3; the Sinkhorn grid will be re-derived by R1 on the refitted Sinkhorn fits (SI-57).
  The Sinkhorn redo overwrites docs/prereg/r1_record.json, r1_record_round1.json, r4_a1_record.json, r4_a1_record_round1.json and r2r3_record.json in place; the versions before the redo stay at tag prereg-tier12-v3, which is made before the redo.
  The analysis will therefore be given `docs/prereg/r1_record.json` (user decision SI-87, 2026-10-15). The six non-Sinkhorn grids
  of the redone record are to be checked by hand against the version at tag prereg-tier12-v3
  (`git show prereg-tier12-v3:docs/prereg/r1_record.json`), because the script does not compare grids: it reads `x1_grid` from
  the record it is given (re-check finding DOC-R10). The script stops when a grid strength of a penalty is missing from that
  penalty's X1 rows, so a record that still holds the superseded Sinkhorn grid is refused.
  [Note 2026-10-23: the amendment of 2026-10-22 (PREREG section 9; SI-109 to SI-113) extends the pilot to all 8 tasks and re-runs
  R1 on them with one shared grid per family and decoder, so the grid statements of this bullet predate it. They are not rewritten
  here; the R1 record that exists when the main benchmark is analysed gives the six strengths.]
- b\* is computed after any R4-x1 extension round: R4 takes each penalty's best batch score from all of its X1 points,
  extension points included (PREREG section 4), while the readings and the area use only the six grid points. A penalty whose
  six points never reach b\* does not reach it (section 2.2) and takes the bottom places of its pair (section 3.1); any of the six
  penalties can do this. The number of such pairs for each penalty, next to that penalty's best batch score M in
  the R4 record, and the no-pair training-step counts (CF-06) are tabulated by a reporting script written after the runs, not by
  `scripts/analysis_x1.py` (user decision SI-81 (c), 2026-10-15; re-check finding AN-R15); the tag freezes this promise, not the script (user, 2026-10-16). The analysis record keeps each
  penalty's reading status in each pair (`bracketed`, `unreached_low`, `unreached_high` or `no_points`) and the R4 record keeps M
  (`targets.<pair>.M`, all six penalties), so the script only counts and pairs them and changes no result.
- Real tasks: pancreas, lung, immune, immune_hum_mou, atac_small, atac_large (`prereg_rules.REAL_TASKS`), in 4 families
  (`TASK_FAMILY`: all 33,506 immune cells are in immune_hum_mou; atac_small and atac_large share 3,618 cells, final
  review I073). sim1 and sim2 are ranked and reported per task and never averaged with real tasks (PAPER_PLAN section 5).
- The six penalties (all with the same scVI backbone; one grid of six strengths per family and decoder): discriminator (1 adversary
  step per model update, SI-01); reference, pooled and barycenter, the Wasserstein-1 critics (5 critic steps per model update; one
  shared grid, because lambda multiplies the same quantity in all three); mmd (a kernel method, not optimal transport); sinkhorn
  (an entropy-regularised optimal-transport distance, not an estimate of W1). Collectively they are called alignment penalties.
- Label transfer (section 9) also reads the saved latent (embedding) file of every fit it uses and the task's batch and cell-type
  labels; the SHA-256 sums of these files go into the record with the other inputs (the provenance requirement is unchanged).

## 2. Biology at matched batch removal: the reading

**2.1 Curve.** For a penalty in a pair: the lambda = 0 point (b0, c0) followed by its six grid points in increasing
lambda, each point = mean B and mean C over its successful runs (user decisions SI-69 and SI-71, Q2 and Q4). Anchoring at lambda = 0 is the
user's decision SI-56.

**2.2 Reading.** A penalty's reading is C on its curve where B = b\*: walking up in lambda from the anchor, the first point equal to
b\* (within 1e-9) or the first adjacent pair whose B values bracket b\* (linear interpolation in B), i.e.
`prereg_rules.bio_at_bstar`, the R2/R3 reading (PREREG section 3). In a ranked pair the anchor lies below b\* (b\* - b0 =
0.5 (b_common - b0) >= 0.01), so this is the weakest strength at which the penalty first reaches the target. A penalty reaches b\*
when the reading exists (status `bracketed`); if its curve stays below b\* the status is `unreached_low` and it does not reach b\*.
Further crossings (a curve that falls back past its batch peak, I038, I062) are counted and reported, not used. (RD-A; the
reading itself is unchanged.)

**2.3 Low-signal and ranked pairs (RD-B, RD-C).** A pair is low-signal when b_common - b0 < 0.02, twice PREREG's smallest-effect
constant 0.01 (`bstar_low_signal`, SI-56). Low-signal pairs are not ranked; each penalty's readings there (biology and local
minus global) are shown in a separate table and never ranked or averaged (SI-56). Every other pair is ranked, whether or not a penalty
reaches b\* in it (section 3.1). The earlier conditions that every penalty's curve reaches b\* and that at least 4 of the 5
leave-one-seed-out readings exist are removed.

**2.4 Failed runs.** Failed runs (status `diverged` or `nonfinite_latent`; user decision SI-71, Q4) are left out of the readings and
the frontier area: a point is the mean over its successful runs (PREREG section 1 "seed mean"); a point whose runs all failed is
skipped; every failure counts in the collapse rate (section 6). Sensitivity analysis: points with any failed run removed
(section 12). (Q4's options and reasons: Appendix A.1.)

**2.5 Decoders (user decision SI-70, Q3).** One ranking per decoder; decoders are never combined. (Q3's options and reasons:
Appendix A.1.)

## 3. Ranking (RD-A, RD-C, RD-D)

**3.1 Ranks within a pair.** In each ranked pair the six penalties are ranked by their readings, rank 1 = highest biology at b\*.
- Penalties that reach b\* are ordered by reading, highest first.
- Penalties that do not reach b\* share the last places (RD-C). With f such penalties, each gets the mean of the places 7 - f to 6
  (one failure: 6; two: 5.5; all six: 3.5). scIB gave methods that could not run on a task the rank of unintegrated data; this
  is the same idea. The pair stays ranked.
- Ties, including penalties that share the bottom places, get average (fractional) ranks (RD-D): two tied for 2nd both get 2.5.
  Readings within 1e-9 are equal (PREREG section 1).
- Example: readings 0.70, 0.65, 0.65, 0.60 and two penalties that do not reach b\* give ranks 1, 2.5, 2.5, 4, 5.5, 5.5. The six
  ranks of a pair always sum to 21.

**3.2 Mean rank.** Per decoder, each penalty's mean rank is the average of its per-task ranks over the real tasks that are ranked
(not low-signal) in that decoder (scIB style: rank within each task, then average the ranks). The number of real tasks the mean is
over is reported with it; it is the same for all six penalties, because every penalty is ranked in every ranked pair. If no real task is
ranked in a decoder, no mean rank exists for it (open point 2). The ranks of each pair are reported with the mean, so a rank given by
the bottom places is visible beside the reading status.

**3.3 Simulations.** sim1 and sim2 are ranked per task and decoder with the same rules, listed by task, and never averaged with each
other or with the real tasks (PAPER_PLAN section 5). A low-signal simulation pair is not ranked (RD-B).

## 4. Uncertainty of the reading and of the rank (RD-E)

**(i) Standard error of a reading.** Repeat the whole reading five times, each time without one seed: the same seed is left out
of the penalty's runs and of the plain-scVI runs (so b0 and c0 are the means of the 4 remaining plain-scVI runs), and b\* stays
fixed. A repeat gives a reading when the penalty reaches b\* in it (a repeat in which the remaining plain-scVI runs leave the anchor at or
above b\* is not covered by the existing reading; open point 4). From the m readings x_(i) that exist,
s^2 = ((m - 1)/m) x sum_i (x_(i) - mean x_(.))^2. It is reported only when m >= 4; otherwise "not available". (How the five runs
are combined: average first, then read on the curve of point means; user decision SI-69, Q2. Q2's options and reasons: Appendix A.1.)

**(ii) Rank range.** In each of the five repeats all six penalties are read and ranked again within the pair, with the rules of
section 3.1 (bottom places and ties included). The range is the smallest and the largest rank number the penalty gets over the five
repeats. The rank of the full-seed reading is not part of the range.

A penalty that reaches b\* on the seed-mean curve is ranked even if m < 4. A penalty that does not reach b\* on the seed-mean curve
has no reading, so its standard error is not defined (open point 3). No interval, test or probability is built from the standard
error or the range; they describe the data (RD-A).

## 5. Trade-off (frontier) area

Per penalty, task and decoder (description only; no difference between penalties is tested). The area is the part of the
batch-biology plane reached or dominated by the penalty's six points (point means as in section 2): the union of rectangles
[b_ref, B_i] x [c_ref, C_i] over points with B_i > b_ref and C_i > c_ref. It is divided by the pair's common batch range
(b_common - b0), which expresses it in biology-score units and puts tasks on one scale. Low-signal pairs have no area (the divisor is below 0.02).

The corner (b_ref, c_ref) is (b0, c0 - 0.10): scVI's batch score and 0.10 below scVI's biology score, the collapse
bound of rule R1 (PREREG section 1). Only settings that remove more batch than scVI and keep biology above the collapse
line add area. Pilot-informed: the corner was chosen after the ranges of B and C in the A1 pilot had been seen (A1's 5
non-Sinkhorn arms x 6 cells x 10 lambda = 300 points; rigour review DOC-17; the figures are in Appendix A.1, Q5, and are regenerated by code in
experiments/doc_numbers_20261015/, whose README lists each number with its script, DOC-R04).
Sensitivity analysis: the corner (b0, 0) (section 12). (Decision SI-72, Q5, made the area co-primary; RD-F makes it a per-penalty
description. Q5's options and reasons: Appendix A.1.)

## 6. Collapse rate (user decision SI-75, Q9; per penalty by RD-F, interval by RD-J)

A run collapses when it failed, or when its C is at least 0.10 below the C of the same seed's lambda = 0 run (the R1
collapse bound applied per run). Per penalty, task and decoder: rate = collapsed runs / runs, over the six grid points and 5
seeds (30 runs), with an exact Clopper-Pearson 95% interval (PAPER_PLAN section 5); failure types reported separately. All eight
tasks are used, low-signal pairs included: the low-signal flag concerns b\*, which the collapse rate does not use. Each family's six points span the same stretch of its response by R1's rule (from
no effect to the onset of over-correction or collapse on the pilot tasks), which is what makes rates comparable across
families; the three W1 critics share one grid. No difference between penalties is computed, and no rate is pooled over tasks
(open point 9). The interval treats the 30 runs as independent; section 17, limitation 3, says how far that holds. (Q9's options:
Appendix A.1.)

## 7. Local versus global batch removal (user decision SI-76, Q10; per penalty by RD-F)

Split as in the earlier reviewer response (E6): local = mean(iLISI, kBET, graph_conn), the kNN-neighbourhood metrics;
global = mean(PCR_batch, ASW_label/batch), whole-embedding geometry. Then B = (3 local + 2 global)/5 exactly. For each penalty,
task and decoder, the local score and the global score are read at the same crossing as the biology reading (same interpolation
weights, on the curves of point means), and the value reported is local minus global at that crossing. At the crossing B = b\*, so
3 local + 2 global = 5 b\* and local minus global = 2.5 (local - b\*): more local removal at b\* means less global removal.
A penalty that does not reach b\* has no value. Low-signal pairs go to the separate table of section 2.3. Descriptive: no test and
no contrast between penalties. (Q10's options: Appendix A.1.)

## 8. Good-strength width (R3, X1 part; decided 2026-10-15, SI-80)

Per penalty, task and decoder: the best point maximises mean S; the window is every point whose mean S is at least the
lower 95% bound of the best point's mean (t with n - 1 degrees of freedom); width = log10(largest lambda in the window)
- log10(smallest), in decades (0 for a single point), plus the number of points (PAPER_PLAN section 3, R3 row:
"decades within the seed CI of the maximum"). All pairs are used, low-signal ones included (S does not use b\*). Descriptive.

## 9. Label transfer (RD-G; moved from exploratory to a reported descriptive analysis)

Per penalty, task and decoder, on the five-seed embeddings at the penalty's matched strength: the grid point whose seed-mean batch
score is nearest b\*, chosen as rule R4 chooses its matched strength (PREREG section 4: a point with no failed seed, ties to the
smaller strength; open point 6a). The strength is never chosen by label-transfer accuracy.
- Each batch is held out in turn; its cells' cell types are predicted from the cells of the other batches with a 15-nearest-neighbour
  classifier on the embedding (the posterior mean, as for every score).
- Cell types present in only one batch are dropped; this is decided from the labels alone.
- Score = F1 per cell type, averaged over cell types (macro F1). How the held-out batches and the five seeds are combined is open
  (open point 6b).
- Plain scVI (lambda = 0) and the scvi-tools adversarial classifier (`scvi_adv`, conditioned decoder only) are shown unmatched:
  they have no strength to choose.
- Beside each value the distance of the chosen embedding's seed-mean batch score from b\* is shown, so the reader sees how far
  from matched it is (for the unmatched methods, the distance of their own batch score).
- Low-signal pairs are tabulated and not summarised. No test, no contrast between penalties.

## 10. Single-strength placement: transfer strength and standard tools (selection score: user decision SI-77, Q11; transfer rule: decided 2026-10-15, SI-80; tools: RD-I)

Transfer rule (decided 2026-10-15, SI-80). For each held-out real task, penalty and decoder, the
transfer strength (lambda) is the grid point that maximises the
selection score S averaged over the training tasks: the real tasks of the other families (leave-one-family-out,
so immune and immune_hum_mou, which share cells, never choose each other's strength; likewise atac_small and atac_large).
Only points without a failed run in any training task are eligible (as R4); ties go to the smaller strength.
sim1 and sim2 take the strength chosen on all 6 real tasks. The tables report B, C and S at the transfer strength (mean
and SD over seeds) next to the task's own best strength, labelled an optimistic bound (PAPER_PLAN section 5), and the difference of S from plain
scVI's S (the mean over its five runs, same task and decoder; RD-A).

Standard tools (RD-I). Harmony (6 settings: theta at 10 principal components), Scanorama (6 settings: knn at 10 dimensions) and
sysVI (6 cycle weights) get their setting by the same rule: the setting with the best mean S on the real tasks of other families
(same eligibility, same tie rule with the smaller setting value winning, simulations take the setting chosen on all 6 real tasks),
with the task's own best setting as the optimistic bound. scANVI and PCA have one setting each. Plain scVI and `scvi_adv` are
shown as they are.

The selection score (user decision SI-77, Q11) is S = 0.4 B + 0.6 C, scIB's weights without min-max scaling, so S
does not depend on which methods are compared; the same S is used in sections 8, 10 and 11. (Q11's options: Appendix
A.1.)

## 11. Placement against standard integrators: best-of-k (R5; decided 2026-10-15, SI-80; RD-A, RD-F, RD-H)

Giving every method the same number of tries is the user's decision (SI-68, SI-77). The best-of-k estimator below and the
composition of the method families were the agent's proposals, approved as rules on 2026-10-15 (SI-80).

Per task and decoder, each method family's configurations: each alignment penalty (6 strengths), Harmony (6 theta at 10
PCs), Scanorama (6 knn at 10 dimensions), sysVI (6 cycle weights), and k = 1 methods (scANVI, scVI lambda = 0, scvi_adv); PCA is the uncorrected anchor, not a family (SI-36). Configuration score = mean S over its runs
(deterministic baselines: one run). Expected best of k configurations drawn at random: E_k = sum_i S_(i) C(i-1, k-1) /
C(N, k) with S sorted ascending, k = 1..N; families compared at equal k (final review I084, D-19); labelled optimistic
(chosen and reported on the same data). scANVI is shown in its own label-aware stratum. The 50-PC and 100-dimension
tool defaults are sensitivity points, not family members. Frontier positions: every baseline configuration's (B, C)
against each penalty's six-point front, descriptive.

Comparison with plain scVI (RD-A): each penalty's selection score is compared with plain scVI's descriptively, at one strength not
tuned on the task (section 10) and as the expected best of k tries. Plain scVI has one configuration, so its S is the reference
for every k; the difference E_k(penalty) - S(plain scVI) is shown for k = 1..6 beside E_k.

Main figure (RD-H): for each method and decoder, the mean over the six real tasks of its per-task E_k curve (k = 1 to N; N = 6 for
each penalty, Harmony, Scanorama and sysVI; N = 1 for plain scVI, scANVI and scvi_adv). All six real tasks enter, low-signal pairs
included (S does not use b\*). Per-task curves go to the supplement; simulations are reported per task, never averaged with the
real tasks.

## 12. Sensitivity analyses (all reported; none replaces the main ranking)

Each changes one setting and reruns the analyses it applies to (RD-L). Failure handling of the main analysis is kept: a point is the
mean over its successful runs, a point whose runs all failed is skipped, every failure counts in the collapse rate.

| Sensitivity | Applies to | Change |
|---|---|---|
| Biology without trajectory (SI-38) | every analysis that uses C: readings, ranks, standard errors, area, collapse, width, S-based placement (local minus global and label transfer do not use C) | C on immune and immune_hum_mou without trajectory conservation; b\* unchanged (batch only) |
| Per-run reading (option of Q2 not chosen, SI-69) | readings and ranks; local minus global | each run read on its own curve (its own lambda = 0 run as the anchor); how per-run readings give one reading and one rank per penalty is open (open point 5) |
| Points with any failed run removed (SI-71) | readings, ranks, standard errors, area, local minus global | a point with a failed run is dropped from the curve; a penalty that then does not reach b\* takes the bottom places |
| Area corner (b0, 0) (decided 2026-10-15, SI-80) | area | c_ref = 0 instead of c0 - 0.10 |

R1 without trajectory (SI-53) is an A1 re-analysis, already stated in PREREG section 9. The prior-scale and dataset-structure
sensitivity analyses of the contrast design (pairs first, six independent datasets, prior scales 0.05 and 0.2) are removed with
the pooled model (RD-A).

## 13. Not in this file

Per Q1 (SI-68): P3 (X8), P4 (X7), the X3 dose-response, the R3 factor effects of X12 and X15 are locked later, with the X7/X8 fixes;
their status under the ranking design is open (change log, open point 8). X10 other than label transfer is exploratory.

Dropped by the first author (RD-K): the follow-up experiment X6 (regularisation, optimiser and update steps; do not run it) and the
premise check R0 (batch separability, adversary gradient norms, discriminator accuracy). Their rules are not in this file; the
earlier text is in git at tag prereg-tier12-v3 (the X6 analysis rule also at tag prereg-x6-v1).

PAPER_PLAN section 5's four scaling checks (leave-one-method-out rescaling, z-score aggregation, a composite without the silhouette
metrics, rank stability by Kendall's tau) are reported, labelled exploratory, and used for no conclusion (user
decision SI-82, 2026-10-15). `scripts/analysis_x1.py` has no counterpart for them.

## 14. PAPER_PLAN section 5 statements this file replaces (dated pointer added there at sign-off)

Still replaced: failure imputation (Q4); "same init and minibatch order" (PREREG section 9 CR-07; I067, I081); "every method family gets
the same number of configurations" (best-of-k at equal k, D-19); the undefined transfer lambda (section 10).
No longer applicable, because the analysis they belong to is gone (RD-A): ROPE (Q7), the model formula without an arm-by-dataset term
(I056), "Holm over P1-P4". The contrast design as a whole is replaced by the ranking design (Appendix A.4); PAPER_PLAN's pointer
still names the contrast sections.

## 15. Implementation, tests and freeze

- `scripts/analysis_x1.py` (to be rewritten to this file; it still implements the contrast design): reads the X1 manifest, scores
  and failures through `prereg_rules`, the frozen R1 and R4-x1 records, the X13 rows, and the saved latents and labels that label
  transfer needs; writes one JSON record (inputs' SHA-256, git SHA, constants, the quantities below) with
  `prereg_rules.write_record`; refuses incomplete inputs; no plots. When R4 added extension rows to an X1
  grid, the extension manifests and their score directories are passed with the same options, together with one failures
  table that covers every row (the usage line of the script says so); the script recomputes the R4 targets from those rows
  and refuses a record whose targets differ, printing the recorded and the recomputed values (AN-R03, AN-R07). It reads the R1
  record under the name `docs/prereg/r1_record.json`, which is also its name after the Sinkhorn redo (SI-87, section 1).
- Quantities the script writes (RD-A to RD-I):

| Level | Quantities |
|---|---|
| Per pair (task, decoder) | b0, b_common, b\*, low-signal flag |
| Per pair and penalty | reading (or none), reached flag (reading status), crossings, rank (fractional), LOSO standard error and m (or "not available"), the five LOSO readings, lowest and highest rank over the five repeats |
| Per decoder and penalty | mean rank over the ranked real tasks and the number of real tasks it is over; simulation ranks listed by task |
| Per penalty, task and decoder | area; collapse count, rate, Clopper-Pearson interval and failure types; width (best strength, window, number of points); local minus global at the crossing; label-transfer macro F1 and the distance of the chosen strength's batch score from b\* |
| Best-of-k | E_k per task, decoder and method family; per decoder and method the mean over the six real tasks of E_k |
| Transfer-strength tables | per penalty, Harmony, Scanorama and sysVI: chosen strength or setting, B, C, S (mean and SD over seeds), the task's own best setting and its values, difference of S from plain scVI; scANVI, PCA, plain scVI and `scvi_adv` as they are |
| Sensitivity analyses | the same quantities under each change of section 12 |

- Tests: the known-answer tests in `tests/analysis/` belong to the contrast design and are rewritten with the script; their cases are not
  specified here. The planted-error run of the contrast script (`experiments/analysis_numerics_20261014/mutation/`, 2026-10-15)
  does not carry over to the new script.
- Run commands (AN-18, AN-R04), from the repo root of a checkout that contains the tag prereg-tier12-v3 (inputs that mix the
  two equivalent commits of SI-54 resolve only there): tests `python scripts/check_tests.py --base 973a97d --src scripts --
  tests/analysis tests/prereg tests/scvi tests/stage tests/jhpce tests/test_legacy_retired.py` (env scvi-api plus pytest-cov;
  about 14 minutes). It requires every line and branch changed since 973a97d in `scripts/` to be run by a test; on 2026-10-15 it
  printed OK for the changes of that day on top of the branch at 5d06a27 (429 passed, 18 skipped, 838 s) for the contrast version, and it is run again
  on the final tree of the ranking version before it is frozen. `--src` takes a directory: coverage.py reads a file path such as
  `scripts/analysis_x1.py` as a module name and collects no data, and `check_tests.py` then either stops or prints OK from a
  leftover `.check_tests/cov.json`, so no file path is used. A shorter list of test folders fails on the lines of the scripts it
  leaves out: `tests/analysis tests/prereg` alone fails on `scripts/run_stage.py` lines 662 and 663, which only `tests/stage`
  runs (AN-R04). The analysis runs as in the usage line of `scripts/analysis_x1.py`. "Runs once" is a procedure: the script
  itself only refuses to overwrite an existing record.
- `scripts/check_tests.py`, fail-loud lint, notebook RESULT: frozen with the tag prereg-tier12-v3 for the contrast version. How the
  ranking version is frozen is open (change log, open point 1).

## 16. Implementation details (decided 2026-10-15, SI-80; written 2026-10-14; revised 2026-10-23)

Writing `scripts/analysis_x1.py` needed the details below, which sections 1-12 leave open. Items marked (carried over) were approved
with the contrast version (SI-80, SI-83, SI-84) and apply unchanged. Items marked (ranking) restate the ranking decisions and add no
rule. What the ranking decisions leave undefined is not decided here; it is an open point of the change log.

1. Inputs (carried over). X1 and X13 rows are read as one input through `prereg_rules.outcomes`, so they must share one scorer
   commit (after the SI-54 equivalence) and one CPU type: best-of-k compares them directly, and the same latent scores
   differently on the local and the cluster CPUs (cross-host check G4). The script refuses an input that lacks any of
   the 8 tasks, any grid strength of a penalty, or the X13 configurations (per task 6 Harmony, 6 Scanorama, 6 sysVI,
   1 scANVI), plain-scVI seeds other than 0-4 (rigour review AN-10), and an R4 stage-x1 record whose b0, b_common, b\*
   or low-signal flag differs from what `prereg_rules.r4_stage` computes from the same X1 rows, R4 extension rows
   included (a record made from other or partial scores; amended 2026-10-14, AN-03: only b0 was compared before). It
   never overwrites an existing record. Input rows are sorted by tag before any sum (AN-11).
2. Reading (carried over, ranking). Exact hits of b\* count as crossings. In a LOSO repeat the anchor is the mean of the four remaining
   plain-scVI runs, a point whose remaining runs all failed is skipped, and b\* is the fixed value of the full analysis (an anchor at or above b\* in a repeat: open point 4). Under the
   per-run reading (a sensitivity analysis, section 12) a penalty is read once per seed, on that seed's own curve.
3. Ranks (ranking). Readings are compared with the tolerance 1e-9 (PREREG section 1). Ranks in a LOSO repeat come from the same code as
   the main ranks.
4. Area (carried over). A pair that is not low-signal always gives a value (0 when no point lies beyond the corner); low-signal
   pairs get no area.
5. Collapse (carried over). All 8 tasks enter: the low-signal flag concerns b\*, which the collapse rate does not use. Counts are over
   the penalty's six grid points; extension rows are not counted.
6. Local versus global (carried over). A penalty's value exists exactly when its biology reading exists: the local and global scores are
   read at the same crossing, so they have the same reasons for not existing.
7. Transfer strength (carried over). The per-task best strength (the optimistic bound) uses the same failure-free rule
   within that task. When no strength is failure-free in every training task, the transfer strength is reported as
   missing.
8. Best-of-k (carried over). The decoder-free baselines (Harmony, Scanorama, PCA) and the conditioned-only ones (sysVI,
   scANVI, scvi_adv) appear in both decoders' tables; scVI at lambda = 0 is the table's own decoder. A configuration
   whose runs all failed is left out and its family's N shrinks; frontier positions count front points that dominate
   a configuration and that it dominates (ties within 1e-9, PREREG section 1). E_k does not penalise failures (a
   configuration's score is the mean of its successful runs), so each family's record also gives its number of
   configurations and of failed runs, and the tables state this (rigour review AN-13). When N shrinks, a real-task mean at k above
   a task's N has no defined value (open point 7).
9. Window width (carried over). The lower bound is the lower end of the two-sided 95% t interval (t at 0.975, n - 1
   degrees of freedom) of the best point's mean; with fewer than 2 successful runs at the best point no width is
   computed. The record flags a gap when a point between the window's smallest and largest lambda lies below the
   bound, so that the width spans it (AN-14). A configuration whose runs all failed has no mean and is not a point of the window;
   the record counts those that lie strictly between the window's smallest and largest lambda in `n_all_failed_between`,
   because the width spans them although `gap` does not flag them (AN-R13).
10. Sensitivity analyses (ranking). Each changes one setting and reruns everything it applies to (section 12): trajectory-free C;
    per-run reading (readings, ranks and local minus global; the area is defined on point means only, section 5, so it has no
    per-run version: corrected 2026-10-14, AN-05); dropping points with a failed run (readings, ranks, area, local minus global);
    corner (b0, 0) (area).
11. Label transfer (ranking). Only what section 9 states is fixed. Open: the strength-choice details, how held-out batches and seeds
    are combined, the classifier's distance and vote rule, the cross-task summary (open point 6).

## 17. Limitations stated with the results (rigour review 2026-10-14: AN-07, AN-08, AN-12, AN-15, AN-17, DOC-16; user decision 2026-10-16: NB-20261016-15; revised 2026-10-23)

1. A penalty's standard error comes from 4 or 5 leave-one-seed-out readings of a statistic that is not smooth (a curve crossing).
   No coverage check exists for the reading of a single penalty. In an earlier simulation of six synthetic scenarios, for the
   difference of two readings (`experiments/analysis_numerics_20261014/first_review/`; 20 worlds of 6 datasets and 200 replicates each),
   the intervals d ± 1.96 s of single datasets covered the reading of the noise-free curves in 0.85 to 0.88 of the replicates, so
   single-dataset statements were less certain than their intervals suggested. No interval is built from the new standard errors.
2. A reading is a piecewise-linear curve through six points read at b\*, so it depends on the grid spacing (the W1 critics' grids have
   one-decade gaps); in the pilot, reading one of six pairs on ten points instead of six moved the difference of two readings by 0.011
   (atac_small with the conditioned decoder: reference minus discriminator 0.017 on all ten pilot lambda, 0.028 on those of the
   grid; the other five pairs did not move; `experiments/analysis_numerics_20261014/first_review/pilot_grid_reading.csv`). Penalties
   whose readings differ by less than that can change places on another spacing. No sensitivity analysis is planned.
3. Collapse rates are taken over each family's own six points, placed by R1 from the pilot's onset of collapse, so rates are
   comparable across families only approximately; no sensitivity analysis is planned. The Clopper-Pearson intervals treat the 30
   fits of a penalty in a task (6 strengths x 5 seeds) as independent, although each fit's collapse is judged against the same
   seed's plain-scVI biology score, so the 6 fits of a seed share their threshold (RD-J). In the review's simulation
   (`first_review/p2_clustering_sim.json`, 20,000 replicates per line) the interval covered 0.9995 when the runs were independent
   and 0.914 when each seed had one collapse threshold shared by its six strengths, so actual coverage lies between 0.91 and 1.00.
4. Ranks and mean ranks are descriptive and carry no test (RD-A). Readings of neighbouring penalties can differ by less than their
   standard errors; the rank range over the five LOSO repeats rests on five repeats of the same data.
5. A penalty that never reaches b\* shares the last places of its pair whatever its biology (RD-C): the rank does not tell a penalty
   that is too weak from one that collapsed. The reading status and the collapse rate report that. b\* is set by the penalty with the
   lowest best batch score (b_common), so one weak penalty lowers the target for all six.
6. Related tasks count as separate tasks in the mean rank: all immune cells are in immune_hum_mou, and atac_small and atac_large share
   3,618 cells, so the 6 real tasks are 4 families and two of them weigh twice.
7. The three W1 critics take 5 critic steps per model update and the discriminator 1 step (X1 manifest); mmd and sinkhorn have no
   adversary network. The experiment that varied regularisation, optimiser and update steps (X6) is dropped (RD-K), so a difference
   between penalties can come from these settings and not only from the alignment term.
8. Iteration-cap hits of the Sinkhorn solve (500,000 iterations, PREREG SI-61) and zero-NMI cases (scib finds zero normalised
   mutual information at every clustering resolution, SI-65) are decided by the user case by case, and no default outcome is
   fixed for either (SI-81; user, 2026-10-16, NB-20261016-15). A referee may see a decision taken after the data as
   freedom after the data; each decision is listed with the results.

Sources checked for the contrast version of this specification: PREREG.md (sections 1-5, 9), PAPER_PLAN.md (sections 3, 5),
CONSTRAINTS.md (SI-01..SI-89), `scripts/prereg_rules.py` (`outcomes`, `r4_stage`, `bio_at_bstar`, `power_report`),
`docs/prereg/r1_record.json` and
`r4_a1_record.json`, the X1 and full manifests (v3), review/rereview/design_fable/REVIEW_design_recheck.md (D-03, D-04,
D-06..D-10, D-18..D-20) and review/lead/verified_issues.csv (I001, I003, I038, I040, I056, I062, I067, I072, I073,
I077, I078, I081, I084, I086, I089), revision/reviewer_response.md (E6 local/global split).

Sources read for the ranking design (2026-10-23): `scripts/analysis_x1.py` (contrast version), `scripts/prereg_rules.py` (`r4_stage`,
`window_curve`, `bio_at_bstar`), PREREG.md sections 1-4 and the entry of 2026-10-22 in section 9, CONSTRAINTS.md (SI-92, SI-108 to
SI-113), `docs/paper_experiment_matrix.csv` (X6, X10), `experiments/analysis_numerics_20261014/first_review/p2_clustering_sim.json`,
and the paper outline `outline_31cadaf.tex` (Methods and Supplementary Notes 5-9; where it and RD-A to RD-M differ, RD-A to RD-M apply).

## Appendix A. Decision log (rigour review DOC-13, 2026-10-14)

The user answered Q1 to Q5 (round 1) and Q6, Q7, Q9, Q10 and Q11 (round 2) on 2026-10-14, each time taking the option the
agent recommended. Each question's options and reasons are quoted below from this file as it stood before the sections above
were reduced to rules (commit fd9c8e3), dated corrections included. The questions as put to the user and the answers
are in the notebook DECISION entries named and in the ledger. A.1 to A.3 are history of the contrast design; their section numbers
are those of the contrast version (map in A.4), and which decisions the ranking design keeps is in A.4.

### A.1 Questions decided on 2026-10-14

**Q1, scope and timing (SI-68, NB-20261014-01).**

Options:
- **A (recommended; chosen 2026-10-14, SI-68).** Lock now every analysis of X1 or X13 data, except R0 and X10, which are labelled exploratory
  (no confirm/refute claim; the "JS gradients vanish" premise is then not claimed as shown, PAPER_PLAN section 2 "drop
  or soften"). Lock the follow-up analyses (X3, X6, X7, X8, X12, X15) later, together with the X7/X8 design fixes,
  while X1 runs, before R4-x1 runs and before anyone examines an X1 result.
  Reason: each analysis is fixed before its own data and before any X1 result exist; X7/X8 must be redesigned anyway,
  so P3/P4 code written now would analyse a design that will change; follow-up fits cannot start before R4-x1, which
  needs every X1 score, so the later lock costs no time.
- **B.** Lock only P1 and the frontier area now; everything else is exploratory unless locked before its data are examined.
- **C.** Lock everything now, including P3/P4 as currently designed, with dated amendments after the X7/X8 fixes.

[Note 2026-10-14, documents review DOC-01: 19 X1 rows had been scored on 2026-10-12, so the reason's "before any
X1 result exist" did not hold when the user chose. The lock is stated as "before any X1 adversarial-arm score is
examined" for this file (status line) and as "before anyone examines an X1 result" for the follow-up analyses
(section 0; SI-68's words: "before anyone looks at an X1 result"); PREREG section 9, entry of 2026-10-14. On 2026-10-15
SI-78 widened the status line's wording to every X1 and X13 score, of any arm.]

**Q2, how the five runs are combined (SI-69, NB-20261014-02).**

DECIDED Q2 (SI-69), how the five runs are combined:
- **A (recommended; chosen 2026-10-14, SI-69).** Average first: read bio@b\* on the curve of point means (the R2/R3 reading). Uncertainty by
  leave-one-seed-out: repeat the whole reading 5 times, each time without one seed (the same seed removed from both
  arms and from lambda = 0; b\* stays fixed); s^2 = ((m - 1)/m) x sum_i (d_(i) - mean d_(.))^2 over the m readings that
  exist. Reason: one registered reading for bio@b\* in the whole project; less sensitive to a single noisy run; the
  leave-one-out repeat still carries the noise in where the curve crosses b\*.
- **B.** Read each run on its own curve (its own lambda = 0 run as anchor), d_s per seed, d = mean over seeds where both
  arms are read, s = SD/sqrt(n). Single-run curves are noisier: more crossings, and seeds that never reach b\* drop out.

The option not chosen is reported as a sensitivity analysis.

**Q3, decoders (SI-70, NB-20261014-03).**

- **A (recommended; chosen 2026-10-14, SI-70).** One P1 result per decoder, both primary; never pooled across decoders. Reason: the conditioned
  decoder (scVI as people use it, SI-09) and the unconditioned decoder (the adversary is the only batch correction, the
  original paper's setting) are different uses, and the effect may differ; in the pilot 2 of 3 conditioned cells
  (immune, sim1) were low-signal, so a conditioned-only primary could rest on few datasets.
- **B.** Conditioned decoder primary, unconditioned secondary.
- **C.** One result pooled over both decoders, with decoder as an adjustment term.

[Note 2026-10-14, documents review DOC-04: b_common is the lowest of the six arms' best batch scores, and Sinkhorn
sets it in no block of the R4-a1 record, so replacing the Sinkhorn arm can only lower it: the Sinkhorn redo can flag
further blocks and cannot clear these two.]

**Q4, failed runs (SI-71, NB-20261014-04).**

DECIDED Q4 (SI-71), failed runs (status `diverged` or `nonfinite_latent`):
- **A (recommended; chosen 2026-10-14, SI-71).** Left out of P1 and the frontier area: a point is the mean over its successful runs (PREREG
  section 1 "seed mean"); a point whose runs all failed is skipped; every failure counts in P2 (section 6).
  Sensitivity: points with any failed run removed. Reason: the registered rule; failures are still reported, as P2.
- **B.** Count a failed run as plain scVI (its lambda = 0 scores), PAPER_PLAN section 5 "impute the unintegrated
  score". Problem: a failed run at a strong penalty then looks as if it removed no batch effect, which can create a
  false crossing of b\* (final review D-03).

**Q5, corner of the frontier area (SI-72, NB-20261014-05).**

Options for the corner (b_ref, c_ref):
- **A (recommended; chosen 2026-10-14, SI-72).** (b0, c0 - 0.10): scVI's batch score and 0.10 below scVI's biology score, the collapse bound of
  rule R1 (PREREG section 1). Only settings that remove more batch than scVI and keep biology above the collapse line
  add area. Reason: in the pilot (A1, seeds 100-102) plain scVI's C was 0.67-0.90 per cell, and the methods' points
  moved C by -0.59 to +0.23 from it (the middle half of all 300 points by -0.09 to 0.00; within each cell the middle half lies inside -0.19 to +0.09;
  corrected 2026-10-14, NB-20261014-06; pilot-informed: the 300 points are A1's 5 non-Sinkhorn arms x 6 cells x 10
  lambda, rigour review DOC-17; each figure is regenerated by code in experiments/doc_numbers_20261015/, whose README lists
  the script and output, DOC-R04). With a corner at biology 0 every
  rectangle's height carries the whole 0.67-0.90 baseline, so the area would mostly measure how far a method pushes
  batch removal, not what it costs in biology.
- **B.** (b0, 0), the final review's example (D-03).
- **C.** The area becomes a secondary, descriptive measure; P1's reading at b\* is the single primary result per decoder.

**Q6, combining datasets (SI-73, NB-20261014-07).**

Options: **A (recommended; chosen 2026-10-14, SI-73)** the family term above; **B** average the two related pairs first (4 units, one tau);
**C** six independent datasets (one tau).

**Q7, the margin r (SI-74, NB-20261014-08).**

DECIDED Q7 (SI-74), the margin r:
- **A (recommended; chosen 2026-10-14, SI-74).** r = 0.01, PREREG section 1's constant (delta_min, also the R2/R3 margin), for P1, the area (in
  biology units) and the local-global split. The measurement SD may still be reported, descriptively.
- **B.** r = max(0.01, 2 x measurement SD), PAPER_PLAN section 5. Needs a re-scoring study specified and run before
  any declaration (which 5% of latents, which seeds and machines, which estimator: final review I003, D-18).

**Q9, P2 comparison (SI-75, NB-20261014-09).**

Options: **A (recommended; chosen 2026-10-14, SI-75)** also a pooled reference-minus-discriminator difference: per dataset the difference in
rates with the Agresti-Caffo standard error (one collapse and one non-collapse added per arm), the section 4 model
with priors scaled for proportions (Delta ~ N(0, 0.25^2), tau ~ half-normal(0.25)), margin 0.05 (5 percentage points;
pilot-informed: the margin and the prior scale were chosen after the pilot collapse rates were shown; the question
gave the unconditioned discriminator as 8 of 45 collapsed runs, which leaves out its 9 runs at lambda 0.03 from the
R1 extension; with them it is 8 of 54, so the unconditioned reference-minus-discriminator difference is +3.7 points,
not +0.7 (NB-20261014-22; the recount is regenerated by code in experiments/doc_numbers_20261015/, DOC-R04); whether SI-75 stands
was put to the user at the diff review and it stands, NB-20261015-07);
**B** rates and intervals only, no declaration.

**Q10, local versus global batch removal (SI-76, NB-20261014-10).**

Options: **A (recommended; chosen 2026-10-14, SI-76)** one signed question: at matched batch removal, does the critic get there through more
local mixing (local difference > 0, equivalently global difference < 0)? The local score is read at the same crossing
as P1 (same interpolation weights); per decoder: exact two-sided sign test over the real datasets and the section 4
model with r = 0.01. **B** compare the orderings at each arm's matched lambda (R4-x1), where B is only approximately
matched, so the comparison mixes in the leftover batch difference. **C** descriptive plots only.

**Q11, selection score S (SI-77, NB-20261014-11).**

Options for S (the same S is used in sections 9 and 10): **A (recommended; chosen 2026-10-14, SI-77)** S = 0.4 B + 0.6 C, scIB's weights without
min-max scaling, so S does not depend on which methods are compared; **B** the best C among points with B >= b\* of
the training dataset (final review D-08's example); **C** S = C.

### A.2 Why there is no Q8

No question was numbered Q8. In the first draft of this file (commit 6d0ca1f, 2026-10-13) the multiplicity rule,
which sits between Q7 and Q9, was already written as a proposal for the diff review rather than as a question, and
the later questions kept their numbers. The multiplicity rule of section 5 was therefore the agent's proposal; the user
approved it as a rule on 2026-10-15 (SI-80).

### A.3 The questions of the diff review (answered 2026-10-15)

No open items remain in this file. On 2026-10-15 the user answered all 19 questions of the diff review (ledger SI-78 to SI-87;
notebook NB-20261015-02 to -20, with the correction -21), and later decided how the audit's findings are listed (SI-88,
NB-20261015-25). PREREG section 9 has the complete list of the questions and answers, with the file and line where each is
explained. The answers that decide something in this file:

1. SI-78: what "examined" means (status paragraph). NB-20261015-15: the user states that nobody read an X1 or X13 score value.
   SI-88: how PREREG lists the displays the audit found.
2. SI-80: the agent's proposals are rules (sections 2.3, 5, 8, 9, 10, the unnamed sensitivity analyses of section 11, and 15).
3. SI-81 (c): a reporting script written after the runs tabulates the per-arm counts of cells that never reach b\* (section 1).
4. SI-82: PAPER_PLAN's four scaling checks are exploratory (section 12).
5. SI-83: a standard error below 1e-6 is raised to 1e-6 (section 4). SI-84: five design choices of the code are approved and the
   accuracy ranges are stated (sections 4, 14 and 15).
6. NB-20261015-07: SI-75 stands (section 6). SI-87: the Sinkhorn redo overwrites five record files in place (section 1).

### A.4 The ranking design (decided 2026-10-22 and 2026-10-23, first author)

On 2026-10-22 and 2026-10-23 the first author replaced the contrast design (A.1 to A.3) by a descriptive benchmark and made
decisions RD-A to RD-M below. The contrast design is superseded: its primary contrast, pooled model, margins, declarations, Holm
correction, power statements and two-arm comparisons no longer apply to the main-benchmark analysis. A.1 to A.3 stay as history. No
main-benchmark score had been read when the decisions were made (status paragraph). Ledger entries for RD-A to RD-M are still to be
added to CONSTRAINTS.md.

| Decision | Content | Section |
|---|---|---|
| RD-A | Main analysis = descriptive ranking: per task and decoder the six penalties ranked by biology at b\*; mean rank per decoder over the real tasks that are not low-signal; selection score compared descriptively with plain scVI's (transfer strength; best-of-k); no tests, posterior probabilities, margins, equivalence rules, multiplicity corrections or power statements | 2, 3, 10, 11 |
| RD-B | b\*, low-signal rule and curve construction unchanged; low-signal pairs not ranked, shown in a separate table | 2.3 |
| RD-C | A penalty whose curve does not reach b\* in a ranked pair gets the bottom rank (several share the last places); the pair stays ranked; "every curve reaches b\*" removed | 3.1 |
| RD-D | Ties, including shared bottom places: average (fractional) ranks | 3.1 |
| RD-E | Per penalty, task and decoder: LOSO standard error of the reading (jackknife, m >= 4, else "not available") and lowest and highest rank over the five repeats | 4 |
| RD-F | Secondary analyses per penalty, descriptive: area, collapse rate with Clopper-Pearson intervals, good-strength width, local minus global at the crossing, placement at equal tries | 5-8, 11 |
| RD-G | Label transfer moved from exploratory to reported, descriptive | 9 |
| RD-H | Best-of-k: main figure = mean over the six real tasks of each method's per-task expected-best curve; per-task curves in the supplement | 11 |
| RD-I | Single-strength placement extended to Harmony, Scanorama and sysVI (six settings each, same rule), own best as optimistic bound; scANVI and PCA one setting | 10 |
| RD-J | Collapse intervals stay Clopper-Pearson; limitation: the 30 fits of a penalty in a task are treated as independent (coverage 0.91 to 1.00) | 6, 17 |
| RD-K | Dropped: follow-up X6 (regularisation, optimiser, update steps; do not run) and the premise check (batch separability, adversary gradient norms, discriminator accuracy); their rules removed from the main text | 0, 13 |
| RD-L | Sensitivity analyses kept, applied to the ranking and the per-penalty readings; failure handling unchanged | 12 |
| RD-M | Unchanged: score definitions, rules R1-R4 and the pilot, transfer-strength rule, best-of-k estimator, definition of "examined", provenance requirements | 1, 10, 11 |

What the ranking design does with the decisions of the contrast design:

| Decision | Under the ranking design |
|---|---|
| SI-68 (scope and timing) | kept; R0 and X6 dropped (RD-K), label transfer reported (RD-G) |
| SI-69 (average first; LOSO) | kept for the reading and the standard error; per-run reading is a sensitivity analysis |
| SI-70 (decoders) | kept: one ranking per decoder, never combined |
| SI-71 (failed runs) | kept |
| SI-72 (area corner) | kept; the area is a per-penalty description, not co-primary |
| SI-73 (pooled model), SI-74 (margin), the multiplicity rule, SI-83 (floor of 1e-6), SI-84 (accuracy of the pooled model) | superseded; their text is in git at tag prereg-tier12-v3 |
| SI-75 (collapse) | the rate and Clopper-Pearson interval kept per penalty; the pooled difference with a 5-point margin superseded |
| SI-76 (local versus global) | the reading at the crossing kept per penalty; the sign test and pooled model superseded |
| SI-77 (selection score S) | kept |
| SI-78 (examined), SI-87, SI-88 | kept |
| SI-80 (agent's proposals) | section 2.3 replaced (RD-B, RD-C); the multiplicity rule superseded; transfer rule, best-of-k, width kept; sensitivity analyses changed (section 12) |
| SI-81 (c) | kept; the per-penalty counts of pairs that never reach b\* are tabulated by the reporting script |
| SI-82 (scaling checks) | kept, exploratory |

Section numbers: the earlier version's sections map to this version as follows (A.1 to A.3 and the status paragraphs of A.5 use the
earlier numbers).

| Earlier | Now | Earlier | Now |
|---|---|---|---|
| 0 | 0 | 9 best-of-k | 11 |
| 1 | 1 | 10 width | 8 |
| 2.1, 2.2 | 2.1, 2.2 | 11 sensitivity | 12 |
| 2.3 not evaluable | 2.3 (low-signal pairs) | 12 not in this file | 13 |
| 2.4 contrast and uncertainty | 3, 4; failed runs 2.4 | 13 replaced statements | 14 |
| 2.5 decoders | 2.5 | 14 implementation | 15 |
| 3 area | 5 | 15 details | 16 |
| 4 combining datasets, 5 declarations | removed (git, tag prereg-tier12-v3) | 16 limitations | 17 |
| 6 collapse | 6 | (new) label transfer | 9 |
| 7 local versus global | 7 | 8 single-lambda tables | 10 |

### A.5 Status paragraphs of the contrast version (2026-10-16, moved here on 2026-10-23)

*Status, 2026-10-16 (ledger SI-78 to SI-89): not yet frozen. The user is shown the final diff of branch `final-review-fixes` once more and says yes
before the tag prereg-tier12-v3 is made (SI-52, SI-60). On 2026-10-14 the user decided questions Q1 to Q7 and Q9 to Q11, each time
taking the option the agent recommended (ledger SI-68 to SI-77; notebook NB-20261014-01 to -05 and -07 to -11); no question was
numbered Q8 (Appendix A.2). On 2026-10-15 the user answered the 19 questions of the diff review (ledger SI-78 to SI-87; the
list is in PREREG section 9) and decided how the audit's findings are listed (SI-88). On 2026-10-16 the user answered the findings of the independent re-check (SI-78, SI-81 and SI-83 amended; SI-85 widened; SI-89 added). Those that concern this file: SI-78 (what "examined" means); SI-80 (the agent's proposals of section
2.3, section 5, section 8, sections 9 and 10, the section 11 sensitivity analyses that no decision names, and section 15 are
approved as rules; equal numbers of tries in section 9 is the user's own decision, SI-68 and SI-77, and only the best-of-k estimator
and the composition of the method families were proposals); SI-81 (c) (a reporting script written after the runs); SI-82
(PAPER_PLAN's four scaling checks are exploratory); SI-83 (a standard error below 1e-6 is raised to 1e-6); SI-84 (five design
choices of the analysis code are approved and the accuracy ranges are stated); SI-87 (the Sinkhorn redo overwrites five record
files); SI-88 (how the audit's findings are listed in PREREG). Sections 0 to 16 state the rules only. The options the user did not
choose, the reasons given with each question and the answers of 2026-10-15 are in the decision log, Appendix A.*

*The specification and `scripts/analysis_x1.py` were first committed on 2026-10-14 (56c0762, 990d37d, ff88666; three test gaps closed
in 858101b). After the independent rigour review of 2026-10-14 they were revised in 8b90d79 (code and tests), 6aaa3ac (numerical
checks in `experiments/analysis_numerics_20261014/`), a1d497b, 56febe4 and 4af5069 (this file), and after the independent
re-check and the user's answers of 2026-10-15 again: the zero-spread floor (SI-83), the accuracy ranges and designs (SI-84), the record
names (SI-87), the labels of the agent's proposals (SI-80) and the re-check fixes AN-R01 to AN-R15. The user has not approved the
final text. The tag prereg-tier12-v3, not these commit ids, fixes the registered version: this file, the script and its known-answer
tests (`tests/analysis/`) will be frozen by that tag, which has not been made, before any X1 or X13 score is examined (final review
I001, D-03, D-08, D-09; SI-52).* [Note 2026-10-23: the tag prereg-tier12-v3 was made on 2026-10-16 (commit 10a1ca5; `git tag`).]
