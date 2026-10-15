# Analysis specification for the X1 endpoints (draft for the diff review)

*Status, 2026-10-14: draft until the user has reviewed the full diff of branch `final-review-fixes` (SI-52). The user
decided questions Q1 to Q7 and Q9 to Q11 on 2026-10-14, all as recommended (ledger SI-68 to SI-77; notebook
NB-20261014-01 to -05 and -07 to -11); no question was numbered Q8 (Appendix A.2). Sections 0 to 16 state the rules
only. The options the user did not choose, the reasons given with each question and the open items are in the
decision log, Appendix A (rigour review DOC-13). Details that no question covered are the agent's proposals: section
2.3 (the not-evaluable rule), section 5 (multiplicity), section 8 (the transfer-lambda rule; SI-77 decides only the
selection score), sections 9 and 10, the sensitivity analyses of section 11 that no decision names, and section 15. A
proposal becomes a rule only if the user approves it at the diff review, and each approval will be recorded as a new
ledger entry before the tag is made. After the diff review this file, `scripts/analysis_x1.py` (committed 2026-10-14,
ff88666; three test gaps closed in 858101b) and its known-answer tests (`tests/analysis/`) will be frozen by the tag
prereg-tier12-v3, which has not been made, before any X1 adversarial-arm score is examined (final review I001, D-03,
D-08, D-09; SI-52). "Examined" is defined, as a proposal the user has not yet approved, in PREREG section 9, entry of
2026-10-14. The decision rules R1-R4 (PREREG sections 2-4) do not change. Where PAPER_PLAN section 5 and this file
differ, this file applies; PAPER_PLAN has a dated pointer (section 13).*

Words used below. *Run* (fit) = one model trained with one seed. *Point* = one penalty strength (lambda) of one method
on one task and decoder, i.e. 5 runs (seeds 0-4). *Cell* = task x decoder. *B* = batch score, *C* = biology score
(PREREG section 1, unscaled scIB means). *b0, c0* = mean B and C of plain scVI (lambda = 0) in the cell.
*b\** = the matched batch-removal target of rule R4: half-way from b0 to the batch removal that every method reaches.
Elsewhere the words differ: PREREG rules R2 and R3 use *cell* for a task x decoder x arm combination, and PREREG
section 9 calls a task x decoder pair a task-by-decoder block; a cell of the tissue is called a single cell. Labels:
SI-nn is entry nn of the decision ledger (CONSTRAINTS.md) and NB-yyyymmdd-nn an entry of the lab notebook; I0nn and
D-nn are findings of the final review of 2026-10-13 and of its recheck of the design documents; AN-nn and DOC-nn are
findings of the rigour review of 2026-10-14 (analysis and documents parts); Qn is a question put to the user on
2026-10-14 (Appendix A). PREREG section 9 has a glossary of the other labels and terms.

## 0. Scope and timing (user decision SI-68, question Q1)

| Analysis | Data | Data exist |
|---|---|---|
| P1 biology at matched batch removal, with the frontier area | X1 | X1 |
| P2 collapse rate; local versus global batch removal | X1 | X1 |
| P2 composition dose-response | X3 | after R4-x1 |
| P3 batch-count scaling | X8 (NO-GO as designed: I015, I077, I078, D-07) | after R4-x1 |
| P4 reference choice | X7 (NO-GO as designed: D-06, I089) | after R4-x1 |
| R3 lambda-window width | X1 | X1 |
| R3 factor effects | X6, X12 (A3 done except its 36 Sinkhorn rows, which are to be redone, SI-57) | after R4-x1 |
| R5 placement against standard integrators (best-of-k); single-lambda tables | X13, X1 | X1 |
| R0 premise (batch separability, gradient traces) | X1 histories, lambda = 0 latents | X1 |
| X10 downstream biology | X1 latents and models | X1 |
| X15 KL warm-up sensitivity | X15 | after R4-x1 |

Every analysis of X1 or X13 data is locked now, except R0 and X10, which are labelled exploratory (no confirm/refute
claim; the "JS gradients vanish" premise is then not claimed as shown, PAPER_PLAN section 2 "drop or soften"). The
follow-up analyses (X3, X6, X7, X8, X12, X15) are locked later, together with the X7/X8 design fixes, while X1 runs,
before R4-x1 runs and before anyone examines an X1 result. (Q1's options and reasons: Appendix A.1.)

## 1. Inputs and shared definitions (registered; no choice)

- Score rows, failures and the X1 manifest are read through `prereg_rules.outcomes` (PREREG section 1: a row neither
  scored nor failed stops the analysis; one scorer commit and CPU type per input, with the SI-54 equivalence). The
  analysis runs once, after all 3,000 X1 rows and the 408 X13 baseline rows are scored or failed (PREREG section 9,
  entry of 2026-10-14).
- B, C as PREREG section 1: C includes trajectory conservation on immune and immune_hum_mou (SI-28); the
  trajectory-free C is a sensitivity analysis (SI-38, section 11).
- b0, b_common, b\* and `bstar_low_signal` per cell are read from the frozen R4 stage-x1 record (PREREG section 4:
  "The same b\* is proposed for P1"), and the analysis refuses unless `prereg_rules.r4_stage` gives the same values from the same
  X1 rows (section 15.1; re-check findings AN-R02, DOC-R03). It therefore needs every row that R4 used, extension rows included.
- Points: lambda = 0 (arm `none`) and the six registered lambda of each arm's family and decoder (`x1_grid` of the R1
  record given to `scripts/analysis_x1.py` with `--r1-record`). R4-x1 extension rows are not used as points (PREREG section 4;
  SI-55). For every arm except Sinkhorn the six lambda are those of `docs/prereg/r1_record.json` as it stands at tag
  prereg-tier12-v3; the Sinkhorn grid will be re-derived by R1 on the refitted Sinkhorn fits (SI-57).
  The Sinkhorn redo overwrites docs/prereg/r1_record.json, r1_record_round1.json, r4_a1_record.json, r4_a1_record_round1.json and r2r3_record.json in place; the versions before the redo stay at tag prereg-tier12-v3, which is made before the redo.
  The analysis will therefore be given `docs/prereg/r1_record.json` (user decision SI-87, 2026-10-15). The six non-Sinkhorn grids
  of the redone record are to be checked by hand against the version at tag prereg-tier12-v3
  (`git show prereg-tier12-v3:docs/prereg/r1_record.json`), because the script does not compare grids: it reads `x1_grid` from
  the record it is given (re-check finding DOC-R10). The script stops when a registered lambda of an arm is missing from that
  arm's X1 rows ("registered lambda grid is incomplete"), so a record that still holds the superseded Sinkhorn grid is refused.
- b\* is computed after any R4-x1 extension round: R4 takes each arm's best batch score from all of its X1 points,
  extension points included (PREREG section 4), while P1 and the area read only the six registered points. An arm
  whose six points never reach b\* makes its cell not evaluable (section 2.3 (ii)); only the two arms of the P1 contrast, the
  reference and the discriminator, can do this. The number of such cells for each arm, next to that arm's best batch score M in
  the R4 record, and the no-pair training-step counts (CF-06) are tabulated by a reporting script written after the runs, not by
  `scripts/analysis_x1.py` (user decision SI-81 (c), 2026-10-15; re-check finding AN-R15). The analysis record keeps each cell's
  reading status for the two arms (`p1.status`: `bracketed`, `unreached_low`, `unreached_high` or `no_points`) and the R4 record keeps M
  (`targets.<cell>.M`, all six arms), so the script only counts and pairs them and changes no result.
- Real tasks: pancreas, lung, immune, immune_hum_mou, atac_small, atac_large (`prereg_rules.REAL_TASKS`), in 4 families
  (`TASK_FAMILY`: all 33,506 immune cells are in immune_hum_mou; atac_small and atac_large share 3,618 cells, final
  review I073). sim1 and sim2 are reported per task and never pooled with real tasks (PAPER_PLAN section 5).
- P1 contrast: the reference W1 critic minus the discriminator at 1 adversary step (SI-01; PREREG section 9,
  2026-10-13 (b)).

## 2. P1: biology at matched batch removal

**2.1 Curve.** For an arm in a cell: the lambda = 0 point (b0, c0) followed by its six registered points in increasing
lambda, each point = mean B and mean C over its successful runs (user decisions SI-69 and SI-71, Q2 and Q4). Anchoring at lambda = 0 is the
user's decision SI-56.

**2.2 Reading.** bio@b\* = C on the curve where B = b\*: walking up in lambda from the anchor, the first point equal to
b\* (within 1e-9) or the first adjacent pair whose B values bracket b\* (linear interpolation in B), i.e.
`prereg_rules.bio_at_bstar`, the R2/R3 reading (PREREG section 3). The anchor lies below b\* (b\* - b0 = 0.5 (b_common -
b0) > 0), so this is the weakest penalty at which the method first reaches the target. Further crossings (a curve that
falls back past its batch peak, I038, I062) are counted and reported, not used. The power figures of PREREG section 5
are for the matched-lambda contrast on all six A1 pilot blocks, including the simulation sim1, not for this reading;
they will be recomputed with this reading and only the evaluable blocks when R4-a1 is re-run after the Sinkhorn redo
(rigour review DOC-07; PREREG section 9, entry of 2026-10-13, part (c)). The recomputation changes no rule.

**2.3 Not evaluable (decided 2026-10-15, SI-80).** A cell is not evaluable when (i) it is flagged `bstar_low_signal` (SI-56: b_common - b0 < 2 x
0.01), (ii) either arm's curve never reaches b\*, or (iii) fewer than 4 of the 5 leave-one-seed-out readings exist
(section 2.4; SI-69). Not-evaluable cells are listed with their reason. Readings of low-signal
cells are shown in a separate table and never pooled (SI-56). A very small standard error is not a reason: the pooled
model keeps the cell and raises its standard error to 1e-6 (section 4; SI-83).

**2.4 Contrast and its uncertainty.** d = bio@b\*(reference) - bio@b\*(discriminator).

How the five runs are combined (user decision SI-69, Q2): average first. bio@b\* is read on the curve of point means
(the R2/R3 reading). Uncertainty by leave-one-seed-out: repeat the whole reading 5 times, each time without one seed
(the same seed removed from both arms and from lambda = 0; b\* stays fixed); s^2 = ((m - 1)/m) x sum_i (d_(i) - mean
d_(.))^2 over the m readings that exist. Sensitivity analysis, the per-run reading: read each run on its own curve
(its own lambda = 0 run as anchor), d_s per seed, d = mean over seeds where both arms are read, s = SD/sqrt(n).

Failed runs (status `diverged` or `nonfinite_latent`; user decision SI-71, Q4) are left out of P1 and the frontier
area: a point is the mean over its successful runs (PREREG section 1 "seed mean"); a point whose runs all failed is
skipped; every failure counts in P2 (section 6). Sensitivity analysis: points with any failed run removed.

(Q2's and Q4's options and reasons: Appendix A.1.)

**2.5 Decoders (user decision SI-70, Q3).** One P1 result per decoder, both primary; never pooled across decoders.
(Q3's options and reasons: Appendix A.1.)

## 3. Frontier area, co-primary with P1 (user decision SI-72, Q5)

For an arm in a cell, the area is the part of the batch-biology plane reached or dominated by its six points (point
means as in section 2): the union of rectangles [b_ref, B_i] x [c_ref, C_i] over points with B_i > b_ref and C_i > c_ref.
For pooling across tasks it is divided by the cell's common batch range (b_common - b0), which expresses it in
biology-score units. Low-signal cells are not evaluable (the divisor is below 0.02). The contrast, uncertainty and
pooling follow sections 2.4 and 4.

The corner (b_ref, c_ref) is (b0, c0 - 0.10): scVI's batch score and 0.10 below scVI's biology score, the collapse
bound of rule R1 (PREREG section 1). Only settings that remove more batch than scVI and keep biology above the collapse
line add area. Pilot-informed: the corner was chosen after the ranges of B and C in the A1 pilot had been seen (A1's 5
non-Sinkhorn arms x 6 cells x 10 lambda = 300 points; rigour review DOC-17; the figures are in Appendix A.1, Q5).
Sensitivity analysis: the corner (b0, 0) (section 11). (Q5's options and reasons: Appendix A.1.)

## 4. Combining datasets (user decision SI-73, Q6)

Per endpoint and decoder, over the evaluable real datasets t: d_t ~ N(theta_t, s_t^2) with s_t from section 2.4
(treated as known), theta_t = Delta + u_family(t) + v_t, u ~ N(0, tau_f^2), v ~ N(0, tau_d^2): the arm-by-dataset term
the final review required (I056) and PAPER_PLAN section 5's "(1 | family/dataset)". Priors: Delta ~ N(0, 0.1^2);
tau_f, tau_d ~ half-normal(0.1) (pilot-informed: the A1 pilot's between-task SD of the matched-lambda contrast is
0.109, with 4 degrees of freedom, `docs/prereg/r1_record.json` power.tau; rigour review DOC-17). Computed exactly: Delta and the random effects are integrated in closed form given
(tau_f, tau_d), which are integrated on a 201 x 201 grid over [1e-15, 0.5]^2 with points evenly spaced in log(tau)
(amended 2026-10-14, rigour review AN-01: the evenly spaced grid from 0 first written here was too coarse when a
dataset's standard error s is below its step, and could change a declaration; section 15.4); no random sampling, so
results reproduce digit for digit on the same input files (rows are sorted by tag before any sum; on a different CPU
type the last digits can differ).

A dataset whose leave-one-seed-out standard error is below 0.000001 (1e-6) is kept, with its standard error raised to 1e-6; the same floor applies in the paired-datasets sensitivity analysis, and every raised dataset is listed in the analysis record.
This is user decision SI-83 (2026-10-15). It replaces an exclusion of such datasets that was drafted on 2026-10-14 (rigour review
AN-02); the user rejected the exclusion because dropping a dataset that is certain would bias the result towards the uncertain
datasets. For example, when neither arm adds any frontier area in any leave-one-seed-out repeat, d = 0 and s = 0 exactly, and the
dataset enters the model as d = 0 with s = 1e-6. The model treats s as known and cannot use s = 0; 1e-6 is the smallest s at which
its double-precision computation was verified (section 15.4). In the pairs sensitivity analysis each dataset's s is raised first;
the averaged standard error of a family, sqrt(s1^2 + s2^2) / 2, can fall below 1e-6 even then (two datasets at 1.2e-6 average to
8.5e-7) and is raised again (re-check finding AN-R01). A standard error that is NaN, infinite or negative stops the run. The record
lists every raised dataset or pair in the field `raised` of the pooled result (one list for each endpoint and decoder, in the
primary analysis and in every sensitivity analysis); an entry gives `unit` (the dataset, or the family in the pairs analysis),
`level` (`dataset` or `pair`) and `s` (its value before raising). The per-dataset sign table shows each dataset's own s, not the
raised value.

Reported: posterior mean and 95% interval of Delta; P(Delta < -r), P(|Delta| <= r), P(Delta > r);
the heterogeneity sqrt(tau_f^2 + tau_d^2); per-dataset posteriors; a per-dataset sign table with leave-one-out
intervals, without per-dataset significance claims (PREREG section 9, 2026-10-13 (c)). Frequentist companion: exact
two-sided Wilcoxon signed-rank test on the d_t, descriptive (it cannot reject with 6 datasets, PREREG section 5).

Sensitivity analyses (section 11): the two related pairs averaged first (4 units, one tau), and six independent
datasets (one tau). (Q6's options: Appendix A.1.)

## 5. Declarations

Rule (PAPER_PLAN section 5, Corani 2017): a direction if P(Delta > r) >= 0.95 or P(Delta < -r) >= 0.95; practically
equivalent if P(|Delta| <= r) >= 0.95; otherwise inconclusive. At the frozen power estimate equivalence is unreachable
(PREREG sections 5 and 9).

The margin (user decision SI-74, Q7) is r = 0.01, PREREG section 1's constant (delta_min, also the R2/R3 margin), for
P1, the area (in biology units) and the local-global split. The measurement SD may still be reported, descriptively.
(Q7's options and reasons: Appendix A.1.)

Multiplicity (the agent's proposal, for the diff review; no question covered it, Appendix A.2): each declaration is
made on its own posterior (marginal); the report states how many primary results were judged (P1 and the area for
each decoder, plus P2 for each decoder: six); the Wilcoxon companion applies Holm over all six (descriptive; final
review I040). Holm's first threshold is then 0.05/6 = 0.0083. With six real datasets the smallest exact two-sided
Wilcoxon p is 2/2^6 = 0.03125, so the companion cannot reject at the Holm-adjusted level (rigour review DOC-05;
PREREG section 5, bracketed note of 2026-10-14, which also says that `docs/prereg/r1_record.json` stores the
four-test value 0.0125).

## 6. P2: collapse rate (user decision SI-75, Q9)

A run collapses when it failed, or when its C is at least 0.10 below the C of the same seed's lambda = 0 run (the R1
collapse bound applied per run). Rate per arm and decoder = collapsed runs / runs, over the six registered points and 5
seeds, per dataset and pooled over the 6 real datasets, with Clopper-Pearson 95% intervals (PAPER_PLAN section 5);
failure types reported separately. Each family's six points span the same stretch of its response by R1's rule (from
no effect to the onset of over-correction or collapse on the pilot tasks), which is what makes rates comparable across
families; the three W1 critics share one grid.

Also a pooled reference-minus-discriminator difference: per dataset the difference in rates with the Agresti-Caffo
standard error (one collapse and one non-collapse added per arm), the section 4 model with priors scaled for
proportions (Delta ~ N(0, 0.25^2), tau ~ half-normal(0.25)), margin 0.05 (5 percentage points). Pilot-informed: the
margin and the prior scale were chosen after the pilot collapse rates were shown, and one rate quoted with the question
was miscounted (Appendix A.1, Q9); whether SI-75 stands is an open item for the diff review (Appendix A.3). (Q9's
options: Appendix A.1.)

## 7. Local versus global batch removal (user decision SI-76, Q10)

Split as in the earlier reviewer response (E6): local = mean(iLISI, kBET, graph_conn), the kNN-neighbourhood metrics;
global = mean(PCR_batch, ASW_label/batch), whole-embedding geometry. Then B = (3 local + 2 global)/5 exactly. At
matched batch removal both arms have B = b\*, so 3 x (local difference) + 2 x (global difference) = 0: "their ordering
differs between local and global metrics" (PAPER_PLAN section 3, R1 row) is then true by construction whenever the
local difference is not 0.

The analysis is one signed question: at matched batch removal, does the critic get there through more local mixing
(local difference > 0, equivalently global difference < 0)? The local score is read at the same crossing as P1 (same
interpolation weights); per decoder: exact two-sided sign test over the real datasets and the section 4 model with
r = 0.01. (Q10's options: Appendix A.1.)

## 8. Single-lambda tables: transfer lambda (selection score: user decision SI-77, Q11; transfer rule: the agent's proposal)

Transfer rule (the agent's proposal, for the diff review). For each held-out real dataset, arm and decoder, the
transfer lambda is the registered point that maximises the
selection score S averaged over the training datasets: the real datasets of the other families (leave-one-family-out,
so immune and immune_hum_mou, which share cells, never choose each other's lambda; likewise atac_small and atac_large).
Only points without a failed run in any training dataset are eligible (as R4); ties go to the smaller lambda.
sim1 and sim2 take the lambda chosen on all 6 real datasets. The tables report B, C and S at the transfer lambda (mean
and SD over seeds) next to the per-dataset best lambda, labelled an optimistic bound (PAPER_PLAN section 5).

The selection score (user decision SI-77, Q11) is S = 0.4 B + 0.6 C, scIB's weights without min-max scaling, so S
does not depend on which methods are compared; the same S is used in sections 9 and 10. (Q11's options: Appendix
A.1.)

## 9. Placement against standard integrators: best-of-k (R5; the agent's proposal, for the diff review)

Per dataset and decoder, each method family's configurations: each adversarial arm (6 lambda), Harmony (6 theta at 10
PCs), Scanorama (6 knn at 10 dimensions), sysVI (6 cycle weights), and k = 1 methods (scANVI, scVI lambda = 0,
scvi_adv); PCA is the uncorrected anchor, not a family (SI-36). Configuration score = mean S over its runs
(deterministic baselines: one run). Expected best of k configurations drawn at random: E_k = sum_i S_(i) C(i-1, k-1) /
C(N, k) with S sorted ascending, k = 1..N; families compared at equal k (final review I084, D-19); labelled optimistic
(chosen and reported on the same data). scANVI is shown in its own label-aware stratum. The 50-PC and 100-dimension
tool defaults are sensitivity points, not family members. Frontier positions: every baseline configuration's (B, C)
against each arm's six-point front, descriptive.

## 10. Lambda-window width (R3, X1 part; the agent's proposal, for the diff review)

Per arm, dataset and decoder: the best point maximises mean S; the window is every point whose mean S is at least the
lower 95% bound of the best point's mean (t with n - 1 degrees of freedom); width = log10(largest lambda in the window)
- log10(smallest), in decades (0 for a single point), plus the number of points (PAPER_PLAN section 3, R3 row:
"decades within the seed CI of the maximum"). Descriptive.

## 11. Sensitivity analyses (all reported; none changes a declaration)

- C without trajectory on immune and immune_hum_mou (SI-38) for every comparison above; b\* is unchanged (batch only).
- The per-run reading (section 2.4; the option of Q2 not chosen, SI-69), failure handling (points with any failed run
  removed, section 2.4; SI-71), the corner (b0, 0) (section 3; the agent's proposal) and the dataset structure (pairs
  first, and six independent datasets; section 4; the agent's proposal).
- Priors: the prior scales of the overall difference Delta and of both spreads set together to 0.05, and to 0.2, as
  question Q6 proposed (SI-73). (Corrected 2026-10-14: this line said "half-normal scales 0.05 and 0.2", which named
  only the spreads.)
- R1 without trajectory (SI-53) is an A1 re-analysis, already stated in PREREG section 9.

## 12. Not in this file

Per Q1 (SI-68): P3 (X8), P4 (X7), the X3 dose-response, R3 factor effects (X6, X12) and X15, locked later with the
X7/X8 fixes; R0 and X10 exploratory; no measurement-SD re-score study (SI-74).

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
- Run commands (AN-18), from the repo root of a checkout that contains the tag prereg-tier12-v3 (inputs that mix the
  two equivalent commits of SI-54 resolve only there): tests `python scripts/check_tests.py --base <base> --src scripts
  -- tests/analysis tests/prereg` (env scvi-api plus pytest-cov); the analysis as in the usage line of
  `scripts/analysis_x1.py`. "Runs once" is a procedure: the script itself only refuses to overwrite an existing record.
- `scripts/check_tests.py`, fail-loud lint, notebook RESULT; to be frozen with the tag prereg-tier12-v3, which has not
  been made.

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
2. P1 (section 2). Exact hits of b\* count as crossings. Under the per-run reading (a sensitivity analysis, section 2.4)
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
7. Local versus global (section 7). A dataset is evaluable exactly when it is for P1: the local reading is read at the same
   crossing, so it has the same not-evaluable reasons, and a small standard error is raised, not a reason (SI-83). The end-to-end
   test checks that the two lists of evaluable datasets are equal (re-check finding AN-R05).
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

## 16. Limitations stated with the results (rigour review 2026-10-14: AN-07, AN-08, AN-12, AN-15, AN-17, DOC-16)

1. Each dataset's standard error s comes from 4 or 5 leave-one-seed-out readings of a statistic that is not smooth
   (a curve crossing, a union of rectangles), and the pooled model treats it as known. In the review's simulation this
   made the pooled interval conservative and the per-dataset intervals cover less than 95%, so single-dataset statements
   are less certain than their intervals suggest; at the pilot's noise "practically equivalent" cannot be declared.
2. P1 is the difference of two six-point piecewise-linear curves read at b\*, so it depends on the registered lambda
   spacing (the W1 critics' grids have one-decade gaps); in the pilot, reading one of six cells on ten points instead
   of six moved its difference by 0.011. No sensitivity analysis is planned.
3. P2's collapse rates are taken over each family's own six points, placed by R1 from the pilot's onset of collapse,
   so rates are comparable across families only approximately; no sensitivity analysis is planned. The Clopper-Pearson
   and Agresti-Caffo intervals treat the 30 runs of a cell (6 strengths x 5 seeds) as independent, although each run's
   collapse is judged against the same seed's plain-scVI biology score, so the 6 runs of a seed share their threshold
   (in the review's simulation the intervals' coverage was 0.91 to 1.00).
4. Each declaration is made on its own posterior, without a correction for the number of results judged (section 5,
   a proposal for the diff review); only the Wilcoxon companions get Holm's correction. The local-versus-global
   question (section 7) is one signed question per decoder and lies outside the six primary results and their Holm
   family.
5. The family spread tau_f rests on four dataset families, two of them with two datasets, so its prior can dominate;
   the prior-scale sensitivity analyses (0.05 and 0.2, section 11) show how much.
6. The pairs-first sensitivity analysis combines the two datasets of a family as if their errors were independent,
   although they share cells (immune within immune_hum_mou; atac_small and atac_large); with equal s and fully
   correlated errors the pair's standard error would be 41% larger (1 instead of 0.71 s).
7. P1 compares the reference critic at 5 critic steps per model update with the discriminator at 1 step (X1 manifest);
   the update-budget control (X6) exists only for the pooled critic (PREREG section 9), so P1 does not separate the
   divergence from the extra critic steps.

Sources checked for this draft: PREREG.md (sections 1-5, 9), PAPER_PLAN.md (sections 3, 5), CONSTRAINTS.md (SI-01..SI-67),
`scripts/prereg_rules.py` (`outcomes`, `r4_stage`, `bio_at_bstar`, `power_report`), `docs/prereg/r1_record.json` and
`r4_a1_record.json`, the X1 and full manifests (v3), review/rereview/design_fable/REVIEW_design_recheck.md (D-03, D-04,
D-06..D-10, D-18..D-20) and review/lead/verified_issues.csv (I001, I003, I038, I040, I056, I062, I067, I072, I073,
I077, I078, I081, I084, I086, I089), revision/reviewer_response.md (E6 local/global split).

## Appendix A. Decision log (rigour review DOC-13, 2026-10-14)

The user answered Q1 to Q5 (round 1) and Q6, Q7, Q9, Q10 and Q11 (round 2) on 2026-10-14 with the recommended option
each time. Each question's options and reasons are quoted below from this file as it stood before the sections above
were reduced to rules (commit fd9c8e3), dated corrections included. The questions as put to the user and the answers
are in the notebook DECISION entries named and in the ledger.

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
(section 0; SI-68's words: "before anyone looks at an X1 result"); PREREG section 9, entry of 2026-10-14.]

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
  lambda, rigour review DOC-17). With a corner at biology 0 every
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
not +0.7 (NB-20261014-22); whether SI-75 stands is put to the user at the diff review);
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
the later questions kept their numbers. The multiplicity rule of section 5 is therefore the agent's proposal.

### A.3 Open items for the diff review

1. Whether SI-75 stands. The question quoted 8 of 45 collapsed runs for the discriminator with the unconditioned
   decoder; the recount is 8 of 54 (NB-20261014-22), so the quoted critic-minus-discriminator difference is +3.7
   percentage points, not +0.7. Both are inside the 5-point margin.
2. The definition of "examined" (PREREG section 9, entry of 2026-10-14), and whether anyone read an X1
   adversarial-arm score value.
3. Approval of the agent's proposals: section 2.3, section 5 (multiplicity), section 8 (the transfer rule), sections
   9 and 10, the sensitivity analyses of section 11 that no decision names, and section 15. Each approval becomes a
   new ledger entry before the tag is made.
4. The name of the redone R1 record that section 1 gives to the analysis (`docs/prereg/r1_record_sinkhorn_redo.json`).
5. Whether the scaling sensitivities of PAPER_PLAN section 5 (leave-one-method-out rescaling, z-score aggregation, a
   composite without the silhouette metrics, rank stability) are locked here; this file has no counterpart today.
