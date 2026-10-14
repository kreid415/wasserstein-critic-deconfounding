# Analysis specification for the X1 endpoints (draft for the user's choices)

*Status: DRAFT, 2026-10-13, branch `final-review-fixes`. Rounds 1 and 2 were decided by the user on
2026-10-14 (ask_user), all as recommended: Q1-Q5 (ledger SI-68 to SI-72, notebook NB-20261014-01 to -05) and Q6, Q7, Q9,
Q10, Q11 (SI-73 to SI-77, NB-20261014-07 to -11), marked **DECIDED Qn**; each lists its options with the recommended
option first and its reason. Details not asked as questions (among them the
not-evaluable rule in section 2.3, the multiplicity rule in section 5, the transfer-lambda rule in section 8, sections
9-11 and the implementation details in section 15) are the agent's proposals and are shown to the user at the diff
review. After the user's answers this file, `scripts/analysis_x1.py` and its known-answer tests
(`tests/analysis/`) are committed and frozen with the tag prereg-tier12-v3, before any X1 adversarial-arm score is
examined (final review I001, D-03, D-08, D-09; SI-52). The decision rules R1-R4 (PREREG sections 2-4) do not change.
Where PAPER_PLAN section 5 and this file differ, this file applies; PAPER_PLAN gets a dated pointer (section 13).*

Words used below. *Run* (fit) = one model trained with one seed. *Point* = one penalty strength (lambda) of one method
on one task and decoder, i.e. 5 runs (seeds 0-4). *Cell* = task x decoder. *B* = batch score, *C* = biology score
(PREREG section 1, unscaled scIB means). *b0, c0* = mean B and C of plain scVI (lambda = 0) in the cell.
*b\** = the matched batch-removal target of rule R4: half-way from b0 to the batch removal that every method reaches.

## 0. Scope and timing (DECIDED Q1: option A, SI-68)

| Analysis | Data | Data exist |
|---|---|---|
| P1 biology at matched batch removal, with the frontier area | X1 | X1 |
| P2 collapse rate; local versus global batch removal | X1 | X1 |
| P2 composition dose-response | X3 | after R4-x1 |
| P3 batch-count scaling | X8 (NO-GO as designed: I015, I077, I078, D-07) | after R4-x1 |
| P4 reference choice | X7 (NO-GO as designed: D-06, I089) | after R4-x1 |
| R3 lambda-window width | X1 | X1 |
| R3 factor effects | X6, X12 (A3 done) | after R4-x1 |
| R5 placement against standard integrators (best-of-k); single-lambda tables | X13, X1 | X1 |
| R0 premise (batch separability, gradient traces) | X1 histories, lambda = 0 latents | X1 |
| X10 downstream biology | X1 latents and models | X1 |
| X15 KL warm-up sensitivity | X15 | after R4-x1 |

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

## 1. Inputs and shared definitions (registered; no choice)

- Score rows, failures and the X1 manifest are read through `prereg_rules.outcomes` (PREREG section 1: a row neither
  scored nor failed stops the analysis; one scorer commit and CPU type per input, with the SI-54 equivalence). The
  analysis runs once, after all 3,000 X1 rows are scored or failed.
- B, C as PREREG section 1: C includes trajectory conservation on immune and immune_hum_mou (SI-28); the
  trajectory-free C is a sensitivity analysis (SI-38, section 11).
- b0, b_common, b\* and `bstar_low_signal` per cell are read from the frozen R4 stage-x1 record (PREREG section 4:
  "The same b\* is proposed for P1"); the analysis never recomputes them.
- Points: lambda = 0 (arm `none`) and the six registered lambda of each arm's family and decoder (`docs/prereg/r1_record.json`
  `x1_grid`). R4-x1 extension rows are not used (PREREG section 4; SI-55).
- Real tasks: pancreas, lung, immune, immune_hum_mou, atac_small, atac_large (`prereg_rules.REAL_TASKS`), in 4 families
  (`TASK_FAMILY`: all 33,506 immune cells are in immune_hum_mou; atac_small and atac_large share 3,618 cells, final
  review I073). sim1 and sim2 are reported per task and never pooled with real tasks (PAPER_PLAN section 5).
- P1 contrast: the reference W1 critic minus the discriminator at 1 adversary step (SI-01; PREREG section 9,
  2026-10-13 (b)).

## 2. P1: biology at matched batch removal

**2.1 Curve.** For an arm in a cell: the lambda = 0 point (b0, c0) followed by its six registered points in increasing
lambda, each point = mean B and mean C over its successful runs (DECIDED Q2 and Q4: SI-69, SI-71). Anchoring at lambda = 0 is the
user's decision SI-56.

**2.2 Reading.** bio@b\* = C on the curve where B = b\*: walking up in lambda from the anchor, the first point equal to
b\* (within 1e-9) or the first adjacent pair whose B values bracket b\* (linear interpolation in B), i.e.
`prereg_rules.bio_at_bstar`, the R2/R3 reading (PREREG section 3). The anchor lies below b\* (b\* - b0 = 0.5 (b_common -
b0) > 0), so this is the weakest penalty at which the method first reaches the target. Further crossings (a curve that
falls back past its batch peak, I038, I062) are counted and reported, not used.

**2.3 Not evaluable.** A cell is not evaluable when (i) it is flagged `bstar_low_signal` (SI-56: b_common - b0 < 2 x
0.01), (ii) either arm's curve never reaches b\*, or (iii) fewer than 4 of the 5 leave-one-seed-out readings exist
(section 2.4, Q2 option A, SI-69). Not-evaluable cells are listed with their reason. Readings of low-signal
cells are shown in a separate table and never pooled (SI-56). For the pooled results a cell is also not evaluable when
its standard error is below 1e-6 (reason `zero_spread`, section 4).

**2.4 Contrast and its uncertainty.** d = bio@b\*(reference) - bio@b\*(discriminator).

DECIDED Q2 (SI-69), how the five runs are combined:
- **A (recommended; chosen 2026-10-14, SI-69).** Average first: read bio@b\* on the curve of point means (the R2/R3 reading). Uncertainty by
  leave-one-seed-out: repeat the whole reading 5 times, each time without one seed (the same seed removed from both
  arms and from lambda = 0; b\* stays fixed); s^2 = ((m - 1)/m) x sum_i (d_(i) - mean d_(.))^2 over the m readings that
  exist. Reason: one registered reading for bio@b\* in the whole project; less sensitive to a single noisy run; the
  leave-one-out repeat still carries the noise in where the curve crosses b\*.
- **B.** Read each run on its own curve (its own lambda = 0 run as anchor), d_s per seed, d = mean over seeds where both
  arms are read, s = SD/sqrt(n). Single-run curves are noisier: more crossings, and seeds that never reach b\* drop out.

The option not chosen is reported as a sensitivity analysis.

DECIDED Q4 (SI-71), failed runs (status `diverged` or `nonfinite_latent`):
- **A (recommended; chosen 2026-10-14, SI-71).** Left out of P1 and the frontier area: a point is the mean over its successful runs (PREREG
  section 1 "seed mean"); a point whose runs all failed is skipped; every failure counts in P2 (section 6).
  Sensitivity: points with any failed run removed. Reason: the registered rule; failures are still reported, as P2.
- **B.** Count a failed run as plain scVI (its lambda = 0 scores), PAPER_PLAN section 5 "impute the unintegrated
  score". Problem: a failed run at a strong penalty then looks as if it removed no batch effect, which can create a
  false crossing of b\* (final review D-03).

**2.5 Decoders (DECIDED Q3: option A, SI-70).**
- **A (recommended; chosen 2026-10-14, SI-70).** One P1 result per decoder, both primary; never pooled across decoders. Reason: the conditioned
  decoder (scVI as people use it, SI-09) and the unconditioned decoder (the adversary is the only batch correction, the
  original paper's setting) are different uses, and the effect may differ; in the pilot 2 of 3 conditioned cells
  (immune, sim1) were low-signal, so a conditioned-only primary could rest on few datasets.
- **B.** Conditioned decoder primary, unconditioned secondary.
- **C.** One result pooled over both decoders, with decoder as an adjustment term.

## 3. Frontier area, co-primary with P1 (DECIDED Q5: option A, SI-72)

For an arm in a cell, the area is the part of the batch-biology plane reached or dominated by its six points (point
means as in section 2): the union of rectangles [b_ref, B_i] x [c_ref, C_i] over points with B_i > b_ref and C_i > c_ref.
For pooling across tasks it is divided by the cell's common batch range (b_common - b0), which expresses it in
biology-score units. Low-signal cells are not evaluable (the divisor is below 0.02). The contrast, uncertainty and
pooling follow sections 2.4 and 4.

Options for the corner (b_ref, c_ref):
- **A (recommended; chosen 2026-10-14, SI-72).** (b0, c0 - 0.10): scVI's batch score and 0.10 below scVI's biology score, the collapse bound of
  rule R1 (PREREG section 1). Only settings that remove more batch than scVI and keep biology above the collapse line
  add area. Reason: in the pilot (A1, seeds 100-102) plain scVI's C was 0.67-0.90 per cell, and the methods' points
  moved C by -0.59 to +0.23 from it (the middle half of all 300 points by -0.09 to 0.00; within each cell the middle half lies inside -0.19 to +0.09;
  corrected 2026-10-14, NB-20261014-06). With a corner at biology 0 every
  rectangle's height carries the whole 0.67-0.90 baseline, so the area would mostly measure how far a method pushes
  batch removal, not what it costs in biology.
- **B.** (b0, 0), the final review's example (D-03).
- **C.** The area becomes a secondary, descriptive measure; P1's reading at b\* is the single primary result per decoder.

## 4. Combining datasets (DECIDED Q6: option A, SI-73)

Per endpoint and decoder, over the evaluable real datasets t: d_t ~ N(theta_t, s_t^2) with s_t from section 2.4
(treated as known), theta_t = Delta + u_family(t) + v_t, u ~ N(0, tau_f^2), v ~ N(0, tau_d^2): the arm-by-dataset term
the final review required (I056) and PAPER_PLAN section 5's "(1 | family/dataset)". Priors: Delta ~ N(0, 0.1^2);
tau_f, tau_d ~ half-normal(0.1). Computed exactly: Delta and the random effects are integrated in closed form given
(tau_f, tau_d), which are integrated on a 201 x 201 grid over [1e-15, 0.5]^2 with points evenly spaced in log(tau)
(amended 2026-10-14, rigour review AN-01: the evenly spaced grid from 0 first written here was too coarse when a
dataset's standard error s is below its step, and could change a declaration; section 15.4); no random sampling, so
results reproduce digit for digit on the same input files (rows are sorted by tag before any sum; on a different CPU
type the last digits can differ). A dataset enters the model only if its s is at least 1e-6; below that it is listed
as not evaluable for that endpoint with reason `zero_spread` (amended 2026-10-14, AN-02; for example, neither arm adds
any frontier area in any leave-one-seed-out repeat, so d = 0 and s = 0 exactly; the model treats s as known and cannot
use s = 0, and 1e-6 is the smallest s at which its double-precision computation was verified).
Reported: posterior mean and 95% interval of Delta; P(Delta < -r), P(|Delta| <= r), P(Delta > r);
the heterogeneity sqrt(tau_f^2 + tau_d^2); per-dataset posteriors; a per-dataset sign table with leave-one-out
intervals, without per-dataset significance claims (PREREG section 9, 2026-10-13 (c)). Frequentist companion: exact
two-sided Wilcoxon signed-rank test on the d_t, descriptive (it cannot reject with 6 datasets, PREREG section 5).

Options: **A (recommended; chosen 2026-10-14, SI-73)** the family term above; **B** average the two related pairs first (4 units, one tau);
**C** six independent datasets (one tau).

## 5. Declarations

Rule (PAPER_PLAN section 5, Corani 2017): a direction if P(Delta > r) >= 0.95 or P(Delta < -r) >= 0.95; practically
equivalent if P(|Delta| <= r) >= 0.95; otherwise inconclusive. At the frozen power estimate equivalence is unreachable
(PREREG sections 5 and 9).

DECIDED Q7 (SI-74), the margin r:
- **A (recommended; chosen 2026-10-14, SI-74).** r = 0.01, PREREG section 1's constant (delta_min, also the R2/R3 margin), for P1, the area (in
  biology units) and the local-global split. The measurement SD may still be reported, descriptively.
- **B.** r = max(0.01, 2 x measurement SD), PAPER_PLAN section 5. Needs a re-scoring study specified and run before
  any declaration (which 5% of latents, which seeds and machines, which estimator: final review I003, D-18).

Multiplicity (proposed, for the diff review): each declaration is made on its own posterior (marginal); the report
states how many primary results were judged (P1 and the area for each decoder, plus P2 for each decoder); the Wilcoxon
companion applies Holm over all of them (descriptive; final review I040).

## 6. P2: collapse rate (DECIDED Q9: option A, SI-75)

A run collapses when it failed, or when its C is at least 0.10 below the C of the same seed's lambda = 0 run (the R1
collapse bound applied per run). Rate per arm and decoder = collapsed runs / runs, over the six registered points and 5
seeds, per dataset and pooled over the 6 real datasets, with Clopper-Pearson 95% intervals (PAPER_PLAN section 5);
failure types reported separately. Each family's six points span the same stretch of its response by R1's rule (from
no effect to the onset of over-correction or collapse on the pilot tasks), which is what makes rates comparable across
families; the three W1 critics share one grid.

Options: **A (recommended; chosen 2026-10-14, SI-75)** also a pooled reference-minus-discriminator difference: per dataset the difference in
rates with the Agresti-Caffo standard error (one collapse and one non-collapse added per arm), the section 4 model
with priors scaled for proportions (Delta ~ N(0, 0.25^2), tau ~ half-normal(0.25)), margin 0.05 (5 percentage points);
**B** rates and intervals only, no declaration.

## 7. Local versus global batch removal (DECIDED Q10: option A, SI-76)

Split as in the earlier reviewer response (E6): local = mean(iLISI, kBET, graph_conn), the kNN-neighbourhood metrics;
global = mean(PCR_batch, ASW_label/batch), whole-embedding geometry. Then B = (3 local + 2 global)/5 exactly. At
matched batch removal both arms have B = b\*, so 3 x (local difference) + 2 x (global difference) = 0: "their ordering
differs between local and global metrics" (PAPER_PLAN section 3, R1 row) is then true by construction whenever the
local difference is not 0.

Options: **A (recommended; chosen 2026-10-14, SI-76)** one signed question: at matched batch removal, does the critic get there through more
local mixing (local difference > 0, equivalently global difference < 0)? The local score is read at the same crossing
as P1 (same interpolation weights); per decoder: exact two-sided sign test over the real datasets and the section 4
model with r = 0.01. **B** compare the orderings at each arm's matched lambda (R4-x1), where B is only approximately
matched, so the comparison mixes in the leftover batch difference. **C** descriptive plots only.

## 8. Single-lambda tables: transfer lambda (DECIDED Q11: option A, SI-77)

For each held-out real dataset, arm and decoder, the transfer lambda is the registered point that maximises the
selection score S averaged over the training datasets: the real datasets of the other families (leave-one-family-out,
so immune and immune_hum_mou, which share cells, never choose each other's lambda; likewise atac_small and atac_large).
Only points without a failed run in any training dataset are eligible (as R4); ties go to the smaller lambda.
sim1 and sim2 take the lambda chosen on all 6 real datasets. The tables report B, C and S at the transfer lambda (mean
and SD over seeds) next to the per-dataset best lambda, labelled an optimistic bound (PAPER_PLAN section 5).

Options for S (the same S is used in sections 9 and 10): **A (recommended; chosen 2026-10-14, SI-77)** S = 0.4 B + 0.6 C, scIB's weights without
min-max scaling, so S does not depend on which methods are compared; **B** the best C among points with B >= b\* of
the training dataset (final review D-08's example); **C** S = C.

## 9. Placement against standard integrators: best-of-k (R5)

Per dataset and decoder, each method family's configurations: each adversarial arm (6 lambda), Harmony (6 theta at 10
PCs), Scanorama (6 knn at 10 dimensions), sysVI (6 cycle weights), and k = 1 methods (scANVI, scVI lambda = 0,
scvi_adv); PCA is the uncorrected anchor, not a family (SI-36). Configuration score = mean S over its runs
(deterministic baselines: one run). Expected best of k configurations drawn at random: E_k = sum_i S_(i) C(i-1, k-1) /
C(N, k) with S sorted ascending, k = 1..N; families compared at equal k (final review I084, D-19); labelled optimistic
(chosen and reported on the same data). scANVI is shown in its own label-aware stratum. The 50-PC and 100-dimension
tool defaults are sensitivity points, not family members. Frontier positions: every baseline configuration's (B, C)
against each arm's six-point front, descriptive.

## 10. Lambda-window width (R3, X1 part)

Per arm, dataset and decoder: the best point maximises mean S; the window is every point whose mean S is at least the
lower 95% bound of the best point's mean (t with n - 1 degrees of freedom); width = log10(largest lambda in the window)
- log10(smallest), in decades (0 for a single point), plus the number of points (PAPER_PLAN section 3, R3 row:
"decades within the seed CI of the maximum"). Descriptive.

## 11. Sensitivity analyses (all reported; none changes a declaration)

- C without trajectory on immune and immune_hum_mou (SI-38) for every comparison above; b\* is unchanged (batch only).
- The other reading rule (Q2: each run read on its own curve, SI-69), failure handling (Q4: points with any failed run
  removed, SI-71), the other corner (Q5: (b0, 0), proposed) and dataset structure (Q6).
- Priors: the prior scales of the overall difference Delta and of both spreads set together to 0.05, and to 0.2, as
  question Q6 proposed (SI-73). (Corrected 2026-10-14: this line said "half-normal scales 0.05 and 0.2", which named
  only the spreads.)
- R1 without trajectory (SI-53) is an A1 re-analysis, already stated in PREREG section 9.

## 12. Not in this file

Per Q1 (SI-68): P3 (X8), P4 (X7), the X3 dose-response, R3 factor effects (X6, X12) and X15, locked later with the
X7/X8 fixes; R0 and X10 exploratory; no measurement-SD re-score study (Q7 option A, SI-74).

## 13. PAPER_PLAN section 5 statements this file replaces (dated pointer added there at sign-off)

ROPE (Q7); failure imputation (Q4); the model formula without an arm-by-dataset term (I056); "same init and minibatch
order" (PREREG section 9 CR-07; I067, I081); "every method family gets the same number of configurations" (best-of-k at
equal k, D-19); the undefined transfer lambda (section 8); "Holm over P1-P4" (section 5).

## 14. Implementation, tests and freeze

- `scripts/analysis_x1.py`: reads the X1 manifest, scores and failures through `prereg_rules`, the frozen R1 and R4-x1
  records and the X13 rows; writes one JSON record (inputs' SHA-256, git SHA, constants, every per-cell quantity and
  declaration) with `prereg_rules.write_record`; refuses incomplete inputs; no plots.
- Known-answer tests in `tests/analysis/`: curves with one and several crossings, unreached, low-signal and failed
  points; the area on hand-computed point sets; the model against closed forms (tau = 0 gives the inverse-variance
  weighted mean) and brute-force integration; exact Wilcoxon p-values; Clopper-Pearson against `scipy.stats.binomtest`;
  transfer lambda with a planted best point; best-of-k against enumeration; the local-global identity at the crossing.
  A planted-error run (`experiments/analysis_numerics_20261014/mutation/`, 2026-10-14) catches 54 of 57 one-line
  errors; of the 3 it misses, two change no reported value by more than 2e-5 (measured) and the third only reorders
  floating-point sums (rigour review AN-04).
- `scripts/check_tests.py`, fail-loud lint, notebook RESULT; frozen with the tag prereg-tier12-v3.

## 15. Implementation details (the agent's proposals, for the diff review; 2026-10-14)

Writing `scripts/analysis_x1.py` needed the details below, which sections 1-11 leave open. Each is the agent's
proposal and goes to the user at the diff review (SI-52); none changes a decision SI-68 to SI-77.

1. Inputs. X1 and X13 rows are read as one input through `prereg_rules.outcomes`, so they must share one scorer
   commit (after the SI-54 equivalence) and one CPU type: best-of-k compares them directly, and the same latent scores
   differently on the local and the cluster CPUs (cross-host check G4). The script refuses an input that lacks any of
   the 8 tasks, any registered lambda of an arm, or the X13 configurations (per task 6 Harmony, 6 Scanorama, 6 sysVI,
   1 scANVI), plain-scVI seeds other than 0-4 (rigour review AN-10), and an R4 stage-x1 record whose b0, b_common, b\*
   or low-signal flag differs from what `prereg_rules.r4_stage` computes from the same X1 rows, R4 extension rows
   included (a record made from other or partial scores; amended 2026-10-14, AN-03: only b0 was compared before). It
   never overwrites an existing record. Input rows are sorted by tag before any sum (AN-11).
2. P1 (section 2). Exact hits of b\* count as crossings. Under the per-run reading (option B, a sensitivity analysis)
   a cell is not evaluable when fewer than 4 of the 5 per-run readings exist, mirroring rule (iii).
3. Area (section 3). In each leave-one-seed-out repeat the corner moves with the dropped seed (b0 and c0 from the other
   4 plain-scVI runs) while the divisor b_common - b0 stays fixed, as b\* does. Every repeat gives a value (0 when no
   point lies beyond the corner), so the area has no readings rule. Low-signal cells get no area.
4. Combining datasets (section 4). Each spread's grid has 201 points evenly spaced in log(tau) from 1e-15 to 5 times
   its prior scale (trapezoid rule in log tau, so each point's weight carries a factor tau): up to 0.5 at scale 0.1 as
   section 4 states, and scaled with the prior at 0.05, 0.2 and P2's 0.25 so the prior is never cut short (the
   half-normal mass beyond 5 scales, 6e-7, is left out). Amended 2026-10-14 (rigour review AN-01): the first version
   spaced the points evenly from 0 (step 0.0025 at scale 0.1), which is too coarse when a dataset's s is below the
   step; on the review's test case (s from 0.000175, prior scale 0.2) it gave P(Delta > 0.01) = 0.949, inconclusive,
   where the converged value is 0.976, a declared direction. The posterior of Delta given the spreads is computed from
   solves with the covariance A of d without Delta's prior variance (precision 1 / 0.1^2 + 1'A^-1 1), which keeps full
   double precision for small s (found while checking AN-01: the subtraction it replaces lost up to 0.2% of the
   variance at s = 1e-4 and was unusable near s = 1e-6). The sign table's "leave-one-out intervals" are each dataset's
   d ± t(0.975, m - 1) s, with s its leave-one-seed-out standard error from m readings (AN-06: with m = 4 or 5,
   d ± 1.96 s covers less than 95%). The heterogeneity is reported as the posterior mean of sqrt(tau_f^2 + tau_d^2).
   The model is fitted however few datasets are evaluable (1 or more); the number is reported with it. Validation
   (`experiments/analysis_numerics_20261014/`, synthetic inputs): on 240 random sets of 1-6 datasets with s from 1e-6
   to 0.05, every probability, posterior mean and heterogeneity is within 2e-6 of a finer and wider reference rule,
   and no declaration differs.
5. Declarations (section 5). Holm runs over the six frequentist p values of the primary results (P1, area and P2,
   each per decoder).
6. P2 (section 6). All 6 real datasets enter the comparison: the low-signal flag concerns b\*, which P2 does not use.
   The per-dataset difference and its standard error are both Agresti-Caffo's (rates after adding one collapse and
   one non-collapse per arm). P2's prior scale 0.25 has no prior-sensitivity variant.
7. Local versus global (section 7). A dataset is evaluable exactly when it is for P1 (same crossing).
8. Transfer lambda (section 8). The per-dataset best lambda (the optimistic bound) uses the same failure-free rule
   within that dataset. When no lambda is failure-free in every training dataset, the transfer lambda is reported as
   missing.
9. Best-of-k (section 9). The decoder-free baselines (Harmony, Scanorama, PCA) and the conditioned-only ones (sysVI,
   scANVI, scvi_adv) appear in both decoders' tables; scVI at lambda = 0 is the table's own decoder. A configuration
   whose runs all failed is left out and its family's N shrinks; frontier positions count front points that dominate
   a configuration and that it dominates (ties within 1e-9, PREREG section 1). E_k does not penalise failures (a
   configuration's score is the mean of its successful runs), so each family's record also gives its number of
   configurations and of failed runs, and the tables state this (rigour review AN-13).
10. Window width (section 10). The lower bound is the lower end of the two-sided 95% t interval (t at 0.975, n - 1
    degrees of freedom) of the best point's mean; with fewer than 2 successful runs at the best point no width is
    computed. The record flags a gap when a point between the window's smallest and largest lambda lies below the
    bound, so that the width spans it (AN-14).
11. Sensitivity analyses (section 11). Each changes one setting and reruns everything: trajectory-free C; per-run
    reading (P1 and local-global; the area is defined on point means only, section 3, so it has no per-run version:
    corrected 2026-10-14, AN-05) and dropping points with a failed run (P1, area, local-global); corner (b0, 0)
    (area); pairs-first and six independent datasets (every combined result, P2 included); prior scales 0.05 and 0.2
    (P1, area, local-global).

Sources checked for this draft: PREREG.md (sections 1-5, 9), PAPER_PLAN.md (sections 3, 5), CONSTRAINTS.md (SI-01..SI-67),
`scripts/prereg_rules.py` (`outcomes`, `r4_stage`, `bio_at_bstar`, `power_report`), `docs/prereg/r1_record.json` and
`r4_a1_record.json`, the X1 and full manifests (v3), review/rereview/design_fable/REVIEW_design_recheck.md (D-03, D-04,
D-06..D-10, D-18..D-20) and review/lead/verified_issues.csv (I001, I003, I038, I040, I056, I062, I067, I072, I073,
I077, I078, I081, I084, I086, I089), revision/reviewer_response.md (E6 local/global split).
