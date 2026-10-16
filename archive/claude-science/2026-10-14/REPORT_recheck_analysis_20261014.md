# Re-check, track 1 of 3: the frozen X1 analysis code and its specification

**Reviewer:** Rigor Reviewer (independent; read-only on the project). **Date:** 2026-10-14.  
**Under review:** branch `final-review-fixes` at `08f1e3c39ea449677cea7e34f6201925d18f4b84` (bundle SHA-256 `d399e914...b09205e7` verified): commits 8b90d79 (analysis code, tests and spec fixes AN-01 to AN-06, AN-10, AN-11, AN-13, AN-14, CF-09), 6aaa3ac (numerical checks and planted-error run), 56febe4 and 4af5069 (spec: limitations, labels, run commands; rules moved to sections 0-16, decision log in Appendix A).  
**Code locations** are `file:line` at 08f1e3c. **Blinding:** I did not open, list or compute from any real X1 or X13 score, the X1 runner logs or the identity-review score files. Every run used synthetic inputs on the real design or the A1 pilot data (Sinkhorn dropped). The R4 stage-x1 records used here were written by the real `prereg_rules.r4_stage` from synthetic scores.  
**Numbers:** almost every number below is read from the saved result files by `scripts/make_report.py` (section 8); the few typed ones (run times, line numbers, commit counts) come from the logs and from the code at 08f1e3c.

## 1. Verdict: PASS WITH FIXES

**In plain words.** The fixes do what the first review asked. On a full-size synthetic input the analysis runs to the end in 30 s (24 s before) and writes its record. The three zero-spread inputs that made the old code abort without a record (every area reading collapsed, identical P1 readings, identical local readings) now finish with the reason recorded; a tampered record of matched batch-removal targets is refused; and shuffling the order of the input rows no longer changes a single digit. My own implementation of the pooled model (different algebra, different integration rule, 50-digit arithmetic for the hardest cases) agrees with the code in every declaration probability to 1.13e-9 or better on six-dataset sets with standard errors from 1e-6 to 1.4e-4, and to 1.03e-6 or better on the 385 of 400 random synthetic sets whose largest |d| is at most three prior scales. No declaration changed in the 400 random sets, the 38 hand-built sets or the 20 sets placed within 1e-6 of the 0.95 threshold. I found no wrong result and no broken decision (no blocker). Three items are documents or procedures that must change before the tag, because as written they contradict the code or send the user into a refusal:

| ID | In one sentence | Why it matters |
|---|---|---|
| AN-R02 | Spec section 1 says the analysis "never recomputes" b0, b_common, b* and the low-signal flag; the code recomputes all four and refuses on a difference (section 15.1 says so). | The document that gets frozen contradicts itself and the code. |
| AN-R03 | The analysis command (usage line, spec section 14) names one `--manifest`; whenever R4 added extension rows, the extension manifest must be passed as well, otherwise a valid record is refused. | Demonstrated: 240 synthetic extension rows moved b* in 2 of 16 cells; with the base manifest only, the run is refused for immune_hum_mou\|1; with both manifests it runs. The usage line also names the superseded R1 record. |
| AN-R04 | The test command of spec section 14 prints `FAIL scripts/run_stage.py: changed lines never run: [662, 663]` at 08f1e3c. | Lines 662-663 are the CF-02 change, covered by `tests/stage` (code track). The same command was OK at 8b90d79; narrowing `--src` to `scripts/analysis_x1.py` gives OK, adding `tests/stage` gives OK. |

Twelve further findings are notes (optional): the pairs sensitivity can abort when both datasets of a family have a standard error just above the 1e-6 limit (below about 1.7e-6) (AN-R01); several planted errors survive the tests, including a regression that would abort a run on identical readings (AN-R06 to AN-R09, AN-R14); the numerical accuracy claim holds only up to a data spread of about three prior scales (AN-R10); the lead's reference script shares the code's algebra (AN-R11); with fewer than three evaluable datasets the pooled model cannot declare a direction (AN-R12); a by-arm count that the spec promises and nothing produces (AN-R15); and two small spec and window-flag edge cases (AN-R05, AN-R13).

**Totals:** 15 new findings = 0 blocker + 3 fix + 12 note. Of the 18 original findings, 16 are closed and 2 are partly closed (AN-04 and AN-18); see section 3.

**What the fixes do well (do not change):** the log-spaced grid with weight x tau; the posterior of Delta from solves with A; zero_spread decided by a threshold, not by `== 0` (identical readings give s of about 1e-16, not 0, see section 4.3); recomputing the R4 targets with the function that wrote the record; the tests extended in place rather than added (`check_tests` reports no redundant-test warning); the planted-error run committed with its list.

## 2. Terms used in this report

- **X1**: the main experiment: six adversary types (reference, pooled and barycenter critics, discriminator, MMD, Sinkhorn) x 8 datasets (6 real, 2 simulated) x 2 decoder settings x 6 penalty strengths x 5 random seeds = 2,880 fits, plus 80 plain-scVI and 40 scVI-adversarial fits = 3,000 rows; **X13**: 408 baseline-integrator rows (Harmony, Scanorama, PCA, sysVI, scANVI). **scVI** = single-cell variational autoencoder; **scIB** = single-cell integration benchmark (the score collection). **B** = batch-mixing score, **C** = biology-conservation score (means of scIB metrics, 0 to 1).
- **Reference critic** (W1, Wasserstein-1 critic) and **discriminator**: the two adversaries compared. **P1**: biology kept at a common batch-removal level b*; **d** = reference minus discriminator in one dataset; **area**: frontier area; **local**: the P1 reading repeated on the local batch metrics (iLISI, kBET, graph connectivity); **P2**: collapse rate.
- **b\***: the matched batch-removal target of rule **R4** (half-way from plain scVI to the level every method reaches); **R4 record**: the file that stores b0 (plain scVI), b_common, b* and the low-signal flag per task and decoder; **extension rows**: extra X1 fits R4 adds at lambda values beyond the six registered ones.
- **s**: the leave-one-seed-out standard error of one dataset's d (re-read d five times, each time without one seed). **S_MIN = 1e-6**: below this s a dataset is not pooled (reason `zero_spread`). **TAU_FLOOR = 1e-15**: lower end of the integration over the spreads.
- **Pooled model**: d_t ~ N(theta_t, s_t^2), theta_t = Delta + family effect + dataset effect; Delta ~ N(0, 0.1^2); the two spreads (tau_f between families, tau_d between datasets) ~ half-normal(0.1). It is integrated over the spreads on a 201 x 201 grid. A **prior scale** is the 0.1 of these priors (0.05 and 0.2 in the sensitivity analyses, 0.25 for P2). **Declaration**: "above" if P(Delta > 0.01) >= 0.95, "below" if P(Delta < -0.01) >= 0.95, "equivalent" if P(|Delta| <= 0.01) >= 0.95, else "inconclusive".
- **Planted error (mutant)**: a deliberate one-line change to the code, to see whether the tests notice. **SI-nn**: decision-ledger entry; **NB-yyyymmdd-nn**: lab-notebook entry; **AN-nn**: first-review finding; **AN-Rnn**: finding of this re-check. **Conflict (in prior scales)**: the largest |d| of a data set divided by its prior scale.

## 3. Status of the 18 original findings (check 1)

Status words: **closed** = the fix does what the finding asked and I could not make it fail; **partly closed** = the fix works but something the finding asked for is still open. Locations are at 08f1e3c.

| ID (first review) | What it was, in plain words | Status | Evidence from this re-check | Still open |
|---|---|---|---|---|
| AN-01 | Grid over the two spread parameters was too coarse when a dataset's standard error s is small; a declaration could flip | **closed** | 201 points per spread, evenly spaced in log(tau) from 1e-15 to 5 prior scales, weight tau (`analysis_x1.py:28-31, 163-167`). First review's case at prior scale 0.2: P(Delta > 0.01) was 0.9489 (inconclusive), is 0.9757 (above); my reference 0.9757, error 1.9e-12. No declaration changed in 400 random + 38 hand-built + 20 threshold sets | accuracy limit for strongly spread data (AN-R10) |
| AN-02 | A standard error of exactly 0 aborted the whole analysis with no record | **closed** | `zero_spread` rule (`:454-455`; guard `:153-157`). Three synthetic inputs that the old code aborted (collapsed area, identical P1, identical local readings) now finish (exit 0, 26 to 28 s) and list the reason; pilot A1: smallest s 3.19e-4 = 319 x S_MIN, none below 1e-4 | the pairs sensitivity can still abort (AN-R01); boundary and constant unpinned by tests (AN-R06); floor-versus-exclusion is the user's call |
| AN-03 | Only b0 of the R4 record was compared with the data; b_common, b* and the low-signal flag were trusted | **closed** | all four recomputed with `prereg_rules.r4_stage` and compared at 1e-9 (`:369-378`). Full-size synthetic record: 7 tampered records refused; 2 accepted by design (change of 5e-10 is inside the tolerance; a field the analysis does not read). A valid record built with extension rows is accepted only when the extension manifest is passed | usage and spec do not say so (AN-R03); tolerance, b_common and extension rows untested (AN-R07) |
| AN-04 | 10 of 40 planted errors were not caught by the tests | **partly closed** | lead's run re-run: 54 of 57 caught; of the original 10, 9 are now caught (M13 only by a crash at the grid-length check `:66`); M24 (grid of 101 points) still survives, with M28 and N11. My own 24 planted errors: 11 caught, 13 survive (section 4.6) | AN-R06 to AN-R09, AN-R14 |
| AN-05 | Spec said the per-run reading applies to the area; the code did not | **closed** | spec 15.11 now reads "per-run reading (P1 and local-global)"; sensitivity diff on the full-size input: per-run changes P1 and local only, drop-failed reaches the area (section 4.5) | none |
| AN-06 | Sign-table limits used 1.96, but s rests on 4 or 5 readings | **closed** | t quantile with m - 1 degrees of freedom (`:460`); independent recomputation of all 30 sign rows, 0 mismatches (all m = 5, t = 2.776) | the t-based limit is approximate (jackknife s of a non-smooth statistic); the sign-table constant for m = 4 is unpinned (AN-R09) |
| AN-07 | s is estimated from 4 or 5 readings but treated as known | **closed** | stated in spec section 16 item 1. Its simulation figures restate the first review's own simulation, which I did not repeat | referee point (section 4.10) |
| AN-08 | P1 depends on the six-point grid of penalty strengths | **closed** | spec section 16 item 2 says so; its figure 0.011 matches the first review's pilot file (atac_small, decoder 1: 0.01104) | referee point |
| AN-09 | Question: may R4 extension points move b*? | **closed** | spec 15.1 decides it: the R4 record is compared with `r4_stage` run on the same rows, extension rows included; the six-point curves still exclude them. Demonstrated: 240 synthetic extension rows moved b* in 2 of 16 cells | the sentence in spec section 1 still says the opposite (AN-R02); the by-arm count it promises is not produced (AN-R15) |
| AN-10 | Plain-scVI seeds were not asserted to be 0 to 4 | **closed** | `:374`; a seed set other than 0 to 4 is refused (test `test_refusals`, planted error O14 caught) | none |
| AN-11 | "Digit for digit" held only for one row order | **closed** | rows sorted by tag (`:521`). Full-size input with manifest, scores and failures shuffled: primary and sensitivity JSON identical (0 differences); the old code differed by up to 1.6e-12 | no test of row order (AN-R14) |
| AN-12 | P2 treats the 30 runs of a cell as independent | **closed** | spec section 16 item 3; its coverage figures restate the first review's simulation, not repeated here | referee point |
| AN-13 | Failed runs not shown beside best-of-k scores | **closed** | `configs` and `failed_runs` added (`:415-419`, `:431-434`); independent recomputation from the raw rows: 192 (arm, cell) rows, 0 mismatches | the sum over configurations is untested (AN-R09) |
| AN-14 | Lambda window can span a gap | **closed** | `gap` flag (`:338-341`); independent recomputation of 96 windows, 0 mismatches, 4 flagged | a window whose interior point has no successful run is not flagged (AN-R13) |
| AN-15 | The pairs sensitivity assumes independent errors for datasets that share cells | **closed** | spec section 16 item 6; I checked its figure (the pair standard error is 41% larger than the single-dataset one) | AN-R01 (a different problem in the same path) |
| AN-16 | Three document slips (marker count, Holm threshold, status line) | **closed** | PAPER_PLAN:180 now says seven statements, eight markers (8 `[replaced:` markers counted); PREREG:106, 342-343 and spec section 5 give 0.0125 for four tests and 0.0083 for six; the spec status paragraph says draft until the diff review | the status paragraph still names ff88666 and 858101b (AN-R05) |
| AN-17 | Local-versus-global test lies outside the six primary results | **closed** | spec section 16 item 4 (last sentence) | none |
| AN-18 | Run procedure: exact commands and location | **partly closed** | spec section 14 gives commands and the checkout rule (the SI-54 tag requirement is confirmed: without the tag a mixed-commit input is refused, with it the run completes, 27.6 s). But the test command prints FAIL at 08f1e3c and the analysis command is refused when R4 extension rows exist | AN-R03, AN-R04 |

## 4. The ten checks

### 4.1 Check 1: each fix against the first review's suggestion

Section 3 gives the status of all 18. The triage (planned action) and the fix record (notebook NB-20261014-26, -27) differ in these places. **AN-01:** the triage planned a graded Gauss-Legendre rule; the fix keeps 201 points per spread but spaces them in log(tau) (trapezoid rule), because the lead judged Gauss-Legendre "more code for the same accuracy". My measurements support that on ordinary data (error about 1e-9, and 1e-6 up to a data spread of three prior scales; section 4.2) and not beyond (AN-R10); the triage's own criterion (probabilities within 1e-6 of a converged reference, including s of about 1e-4) is met for ordinary data. **AN-02:** the exclusion option (reason `zero_spread`) was taken, not a floor on s, because "any floor value would be arbitrary" (section 4.9, design 3). **AN-03:** the recompute option was taken and the stored hash rejected because existing records carry none (design 4); the triage also asked for a refusal test with a changed b*, which exists (`test_analysis_x1.py:608`). **AN-04:** the triage asked that the 10 surviving errors fail by extending existing tests; 9 do, M24 does not. **AN-09:** the triage asked for a spec sentence on extension rows and a count of cells not evaluable because b* lies beyond the six points, by cause; the first half exists (spec 15.1), the count is promised but not produced (AN-R15). The other fixes are the one-line suggestions (AN-05, AN-06, AN-10, AN-11, AN-13, AN-14) or spec sentences (AN-07, -08, -12, -15, -17, -18: each checked in section 3 against the code or the evidence).

### 4.2 Check 2: do the pooled-model numbers hold? (AN-01)

**What I did.** (a) Re-ran the lead's `check_pool_numerics.py` in a clean clone (4 min 16 s). (b) Wrote my own reference for the pooled model that shares no code with the analysis: closed-form algebra for the covariance (a diagonal plus one rank-one term per family, and the Sherman-Morrison identity for Delta), composite Gauss-Legendre integration over log tau instead of the trapezoid rule, long-double arithmetic, and a 50-digit mpmath version for the hardest sets. I validated the algebra against a direct 50-digit solve of the full covariance matrix (relative differences 1e-32 to 1e-42 in 50-digit arithmetic; at most 1.6e-12 on the log-likelihood and 6e-19 on the mean and variance in long double), and refined the integration until it stopped changing (finest setting: panels 0.25 wide in log tau, 16 nodes; refinements agree to 1e-15 on ordinary sets). (c) Compared the code's `combine` with it on three groups of synthetic sets.

**(a) The lead's script reproduces.** Every output file is byte-identical to the committed one (`git status` clean after the run). Values: 240 random worlds, largest difference from its reference 1.99e-6 (8.46e-7 when the reference is cut at 5 prior scales like the code), 0 declaration mismatches, smallest s 1.02e-6. The first review's failing case (six datasets, s from 1.8e-4 to 1.4e-3) at prior scale 0.2: old code 0.9489 (inconclusive), amended code 0.9757 (above). My independent reference gives 0.9757; the amended code differs from it by 1.9e-12 (prior 0.1: 3.8e-12; prior 0.05: 1.6e-11), the old code by 2.7e-2, 4.4e-4 and 6.3e-5.

**The lead's reference is not independent of the code.** Its `reference()` uses the same solve with A for Delta and the same V-based weights over the spreads as the code, on a finer log-trapezoid grid over a wider range. A shared algebra error would pass unnoticed. The 40-digit table in `precision_mpmath.csv` is independent but covers only the posterior mean and variance of Delta for given spreads (not the weights over the spreads). My reference is independent of both, and it agrees (below). Finding AN-R11 (note).

**(c) Code against my reference** (error = largest absolute difference in a declaration probability P(Delta > r), P(Delta < -r), P(|Delta| <= r); "all" adds the posterior mean, 95% bounds and heterogeneity):

| Group of synthetic sets | Sets | Largest error, probabilities | Largest error, all quantities |
|---|---|---|---|
| Six datasets, s 9e-5 to 1.4e-4, d about 0.012, prior 0.05 / 0.1 / 0.2, three structures, plus sets at the 0.95 threshold +/- 1e-6, 1e-7 | 13 | 3.83e-13 | 3.83e-13 |
| Six datasets, s 1.0e-6 to 3e-6 (just above S_MIN), d = 0.01 | 7 | 1.13e-9 | 1.13e-9 |
| One dataset (s 1e-6, 1e-4, 5e-3), including thresholds set at P = 0.6 +/- 1e-6, 1e-7 | 12 | 1.24e-7 | 6.29e-7 |
| One tiny s (1e-6) among five of 0.01 to 0.012 | 1 | 2.38e-11 | 2.50e-11 |
| Standard errors at the scale of P2 (0.09 to 0.12), prior 0.25 | 1 | 4.40e-10 | 1.42e-9 |
| Strongly spread datasets (d from -0.3 to +0.4, s = 0.01): prior 0.1 | 1 | 6.62e-7 | 1.20e-6 |
| The same, prior 0.05 (family / independent structure) | 3 | 2.86e-4 | 2.86e-4 |
| 20 sets within 1e-6 / 1e-7 of the declaration threshold (included above) | 20 | 1.24e-7 | declarations flipped: 0 |

**50-digit check.** Run in 50-digit arithmetic with a coarser integration rule (panels 1.5 wide, so these numbers are upper bounds on the arithmetic error), the code differs by at most 1.13e-9 on six-dataset sets with s from 1e-6 to 1.4e-4, by 1.70e-10 on the set with one tiny s, and by 2.63e-7 on a single dataset with s = 1e-6 (the same size as against the finest reference). The strongly spread set was not converged at that setting (3.89e-5) and is not used.

**Random battery** (400 sets from my own generator, seeds 11 and 12: 1 to 6 datasets, three structures, prior scales 0.05 to 0.25, s from 1e-6 to 0.05; smallest s 1.01e-6): 0 declarations changed, 0 refusals; declarations were 363 inconclusive, 24 above, 7 equivalent, 6 below. Error against the finest reference, by how far the data spread beyond the prior (largest |d| in prior scales):

| Largest \|d\| / prior scale | Sets | Largest error, probabilities | Largest error, all quantities |
|---|---|---|---|
| 0 to 1 | 332 | 2.37e-7 | 7.88e-7 |
| 1 to 2 | 39 | 4.92e-7 | 1.18e-6 |
| 2 to 3 | 14 | 1.03e-6 | 2.27e-6 |
| 3 to 4 | 5 | 2.41e-6 | 2.42e-6 |
| 4 to 5 | 5 | 5.51e-6 | 5.51e-6 |
| 5 to 8 | 5 | 5.16e-5 | 5.16e-5 |

Sets with two or more datasets and spread of at most three prior scales (324 sets): 1.37e-7 in probabilities, 3.57e-7 in all quantities. All 329 sets with at least two datasets: at most 1.71e-6. The 18 sets above 1e-6 were re-evaluated with the finest reference (the coarse and fine references agree to better than 1e-8 there); the other sets used a coarser reference whose own error is at most about 2e-7 for spreads up to three prior scales.

**Where the 1e-6 claim stops holding (AN-R10).** The grid ends at 5 prior scales. If the data require a spread tau beyond that (the datasets differ by more than about three or four prior scales), the code leaves out real posterior mass, and the trapezoid end correction adds a second error. Scan (six datasets, s = 0.01, the d values spread with a given standard deviation, in prior scales):

| SD of d / prior scale | prior 0.1: error, probabilities | prior 0.1: error, all | prior 0.05: error, probabilities | prior 0.05: error, all |
|---|---|---|---|---|
| 1 | 6.88e-9 | 1.05e-8 | 6.04e-9 | 6.04e-9 |
| 2 | 5.75e-8 | 1.57e-7 | 9.99e-8 | 9.99e-8 |
| 3 | 3.45e-7 | 1.59e-6 | 6.34e-7 | 7.73e-7 |
| 4 | 1.27e-6 | 1.11e-5 | 2.41e-6 | 5.47e-6 |
| 5 | 1.05e-5 | 7.77e-5 | 2.03e-5 | 3.75e-5 |
| 6 | 2.93e-5 | 3.05e-4 | 5.81e-5 | 1.49e-4 |
| 8 | 2.14e-4 | 2.81e-3 | 4.44e-4 | 1.37e-3 |

No declaration changed in any of these sets (all are inconclusive, as they should be at these spreads). For reference, the pilot's between-dataset SD of the matched contrast is 0.109 (spec section 4), about one prior scale at 0.1 and two at 0.05, where the scan's errors are about 1e-8 and 1e-7. The half-normal prior leaves 5.7e-7 of its mass beyond 5 scales per spread (the spec's "6e-7"); the floor 1e-15 leaves a relative mass of about 1e-9 at s = 1e-6.

**The 1e-6 lower limit on s.** With the guard lowered and compared with 50-digit arithmetic: s_min 1e-6: 2.78e-10 (family) / 1.13e-9 (independent); 1e-7: 9.11e-10 / 2.04e-7; 1e-8: 2.12e-6 / 1.20e-7. So 1e-6 is a conservative numerical bound; accuracy degrades from about 1e-7 down.

**Judgement.** The log-spaced grid and the A-solve are correct and accurate to about 1e-9 on ordinary inputs and to about 1e-6 up to a data spread of three prior scales. Beyond that the stated accuracy ("within 2e-6") describes the spreads of the lead's 240 synthetic worlds, not the method. No declaration is at risk at any spread I tried. If a spread beyond three prior scales matters, a graded Gauss-Legendre rule with the same node count (the first review's suggestion) is the alternative; I did not test it.

### 4.3 Check 3: zero or tiny s (AN-02)

**What I did.** Rebuilt the first review's synthetic full-size input (real design: 3,000 X1 rows and 408 X13 rows from `manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv`, R4 record from the real `r4_stage`) and planted three zero-spread situations and one low-signal situation; ran the new code (08f1e3c) on all four and the old code (1e615eb) on the three zero-spread ones. A fifth path, four of five leave-one-seed-out readings, I checked by calling the reading function directly.

| Planted situation | Old code | New code |
|---|---|---|
| Every point of both arms collapses in lung, decoder 0: area d = 0 and s = 0.0 exactly | aborted: PreregError "pool needs finite equal-length d and positive s", no record (2.4 s) | ran to the end (28.2 s), `zero_spread` listed |
| Reference and discriminator give identical biology readings in pancreas, decoder 0: P1 d = 0, s = 0.0 | aborted: PreregError "pool needs finite equal-length d and positive s", no record (2.1 s) | ran to the end (28.1 s), `zero_spread` listed |
| Identical local-metric readings in every cell: s about 5e-17 to 1.8e-16 in 15 of 16 cells, 0.0 in atac_small decoder 0 | aborted: PreregError "pool needs finite equal-length d and positive s", no record (2.5 s) | ran to the end (25.8 s), `zero_spread` listed; pooled local has n = 0 (not evaluable); P1 stays evaluable |
| All six real datasets low-signal in decoder 1 | not run | ran to the end (24.5 s); decoder-1 pooled results n = 0, `not_evaluable` with reason `low_signal`; decoder 0 unaffected |
| Four of five leave-one-seed-out readings exist and the four are identical (the unit case at `test_analysis_x1.py:90-92`) | s = 0.0 exactly with m = 4; the old guard `s > 0` refuses it | `read_p1` gives m = 4, s = 0.0; the rule at `:454` applies for any m, so `zero_spread` (called directly: `logs/four_of_five_demo.txt`) |

One of my planted errors matters here: with the rule written as `s == 0` instead of `s < S_MIN` (O10), the identical-local-readings input aborts again in 3 s with no record, and **no test notices** (section 4.6). Rounding-level s is the realistic case, not exact zero.

**Pilot comparison.** On the A1 pilot data (3 seeds, 10-point grid, Sinkhorn dropped, 6 cells) the 18 standard errors (3 endpoints x 6 cells) run from 3.19e-4 (local, immune, decoder 1) upward, i.e. 319 times S_MIN; the smallest P1 value is 3.18e-3 and the smallest area value 4.11e-3; 2 of 18 lie below 1e-3 and none below 1e-4; no pilot area is zero. S_MIN therefore excludes no pilot cell. (The pilot has 3 seeds, so its s values are not those X1's five seeds will give; they show the size to expect, not a bound. The pilot data are the only real data I used.)

**A path the fix missed (AN-R01).** In the pairs sensitivity (the two datasets of a family are averaged first) the pair's standard error is sqrt(s1^2 + s2^2) / 2, which can fall below S_MIN although each s passes the rule (`analysis_x1.py:232` against `:454`). `pool` then refuses (`:156`) and the whole run aborts. Demonstrated: two datasets of one family with s = 1.2e-6 each pass the rule, give a pair s of 8.5e-7, and abort; from s = 1.42e-6 each, the pair is accepted (log `logs/small_demos.txt`). The trigger needs both datasets of a family to have s between 1e-6 and about 1.7e-6, which is improbable but costs the run.

### 4.4 Check 4: recomputed R4 targets (AN-03, AN-10, SI-54)

**What I did.** Wrote synthetic R4 stage-x1 records with the real `prereg_rules.r4_stage` on the full-size synthetic input, then changed one thing at a time and ran the analysis (each refusal takes 1.4 s, each accepted run 28 s). Refusal message in every case: "R4 targets (b0, b_common, b* or the low-signal flag) differ from these scores for <task|decoder>" (`analysis_x1.py:375-378`).

| # | Change to the record | Should the analysis accept? | Result |
|---|---|---|---|
| T1 | b* + 0.03 at atac_large, decoder 0 | no | refused (1.4 s) |
| T2 | b_common set to 0.99 at pancreas, decoder 1 | no | refused (1.4 s) |
| T3 | low-signal flag set to true at lung, decoder 0 | no | refused (1.4 s) |
| T4a | b* + 5e-10 (inside the 1e-9 tolerance) | yes | accepted (28.3 s) |
| T4b | b* + 2e-9 (outside) | no | refused (1.4 s) |
| T5 | b0 + 2e-9 at lung, decoder 0 | no | refused (1.4 s) |
| T6 | a field the analysis does not compare (`M`) changed | yes | accepted (28.1 s) |
| T7 | flag set to false on the low-signal cell immune, decoder 1 | no | refused (1.4 s) |
| T8 | b_common alone + 0.01 at lung, decoder 0 | no | refused (1.4 s) |

All nine behave as intended. The tolerance is absolute 1e-9 on b0, b_common and b*, exact on the flag (`:375-377`); the old code compared b0 only (first review).

**Extension rows.** I generated 240 synthetic R4 extension rows (one step below and one above each arm's grid) and built the record the way `r4_stage` does when they exist. They moved b* in 2 of 16 cells (immune_hum_mou, decoder 1: 0.65438 to 0.66363; lung, decoder 1: 0.72087 to 0.72165; flags unchanged). Results:

| Input | Result |
|---|---|
| base manifest + extension manifest, base + extension scores, record built with extension rows | accepted (28 s) |
| base manifest only (as in the usage line), same scores and record | **refused** for immune_hum_mou, decoder 1 (1.4 s) |
| both manifests, record built without the extension rows | refused for immune_hum_mou, decoder 1 |

So the recompute is sound and the refusal is correct, but the documented command (`analysis_x1.py:6-9`; spec section 14) lists one `--manifest`, and the message does not say that extension rows may be missing. Finding AN-R03. The synthetic fixture of the test (one extension row, `test_analysis_x1.py:447-448`) does not move b*: it raises the mmd arm's best batch score (0.7021 to 0.7063) while b_common, the lowest of the arms' best scores, is the barycenter's 0.6996; so no test fails if the recompute ignores extension rows (planted error O24, AN-R07).

**Other checks.** Plain-scVI seeds other than 0 to 4 are refused (`:374`; test `test_refusals`; planted error O14, caught). SI-54: an input whose non-Sinkhorn rows were scored at c88cce3 and Sinkhorn rows at 08f1e3c is accepted in a checkout that has the local tag `prereg-tier12-v3` (27.6 s) and refused without it ("scores of ['X1', 'X13'] mix scorer provenance ...", 0.9 s), as the spec says (section 14).

### 4.5 Check 5: AN-06, AN-11, AN-13, AN-14 and the sensitivity wiring (AN-05)

**What I did.** On the full-size synthetic record I recomputed these quantities from the raw synthetic rows with plain pandas, without calling the analysis functions: the 30 sign-table rows (limits d +/- t(0.975, m-1) s), best-of-k `configs` and `failed_runs` for 192 (arm, cell) rows, and `lams`, `best` and `gap` for 96 lambda windows.

| Item | Rows checked | Mismatches |
|---|---|---|
| Sign-table limits (AN-06; all m = 5, t = 2.776; the old limits used 1.96) | 30 | 0 |
| Best-of-k configurations and failed runs (AN-13) | 192 | 0 |
| Lambda window, best point and gap flag (AN-14; 4 windows flagged) | 96 | 0 |
| Row order (AN-11): manifest, score and failure rows shuffled, primary and sensitivity JSON | all numbers | 0 (label differences 0) |
| Same shuffle with the sort removed (planted error N11) | all numbers | 7864 numbers differ, largest 3.2e-15; no label differs |

The sort is therefore effective but not tested, and without it the differences are at the level of the last digit (AN-R14). "Digit for digit" in the spec is true for the same input files in any row order with the sort, and for the same CPU type (the spec says so).

**Sensitivity wiring** (count of numbers in each sensitivity that differ from the primary, by part of the record; full-size synthetic input):

| Sensitivity | Per-dataset P1 | Per-dataset area | Per-dataset local | Pooled P1 | Pooled area | Pooled local | Pooled P2 |
|---|---|---|---|---|---|---|---|
| per_run | 181 | 0 | 181 | 88 | 0 | 86 | 0 |
| drop_failed | 22 | 31 | 18 | 1 | 38 | 0 | 0 |
| corner_zero | 0 | 117 | 0 | 0 | 86 | 0 | 0 |
| prior_0.05 | 0 | 0 | 0 | 48 | 48 | 48 | 0 |
| prior_0.2 | 0 | 0 | 0 | 48 | 48 | 48 | 0 |
| pairs | 0 | 0 | 0 | 33 | 34 | 34 | 22 |
| independent | 0 | 0 | 0 | 46 | 48 | 47 | 52 |

This matches the amended spec section 15.11: the per-run reading changes P1 and local only (the area has no per-run version: AN-05 closed), dropping failed points reaches the area, the zero corner changes the area only, the prior sensitivities touch the three pooled endpoints and not P2, and the pairs and independent versions touch every combined result.

### 4.6 Check 6: tests, coverage and planted errors (AN-04)

**Tests.** `pytest tests/analysis tests/prereg` at 08f1e3c: 193 passed, 1 skipped (89 s); `tests/analysis` alone collects 62 cases (17 functions, as before; 53 cases at 1e615eb). `check_tests.py --base 973a97d --src scripts -- tests/analysis tests/prereg` (the command of spec section 14, with the branch's base 973a97d as `<base>`):

| Commit and command | Output |
|---|---|
| 8b90d79 (the analysis commit), as in the spec | `193 passed, 1 skipped`, `check_tests: OK (every changed --src line and branch is tested; no redundant new tests)` |
| 08f1e3c, as in the spec | `193 passed, 1 skipped`, **`FAIL scripts/run_stage.py: changed lines never run: [662, 663]`**; no WARN, nothing about `analysis_x1.py` |
| 08f1e3c, `--src scripts/analysis_x1.py` | OK |
| 08f1e3c, with `tests/stage` added to the test list | `259 passed, 2 skipped` (6 min), OK |

Lines 662 and 663 of `run_stage.py` are the CF-02 change (code track, commit 8351999), which `tests/stage` covers. The FAIL is not an analysis defect, but it is what the user sees when following section 14 (AN-R04). (A first run polluted by an untracked virtual environment inside my clone printed 852 spurious warnings; I excluded the environment and re-ran. The numbers above are from the clean runs.)

**The lead's planted-error run.** I re-ran `mutation/run_mut.py` (57 one-line errors) with 4 parallel slots instead of 6: 54 of 57 caught, survivors M24, M28, N11, the same as committed. Of the 10 errors the first review found uncaught (M03, M12, M13, M14, M15, M16, M18, M22, M24, M33), 9 are now caught; M13 is caught only because the end-to-end fixture then fails to build ("atac_small|0|mmd: registered lambda grid is incomplete", the check at `:66`), not by an assertion on the extension row; M24 (grid of 101 points) survives, as do M28 (trapezoid end weights dropped) and N11 (no sort). M32 and N05 are also caught only by a crash. Effects of the survivors (measured on 218 synthetic sets: my headline sets and 200 of the battery):

| Survivor | Largest change, sets with spread up to 3 prior scales | Beyond 3 scales |
|---|---|---|
| M24 (GRID_N 101) | 1.12e-5 | 8.47e-3 |
| M28 (end weights dropped) | 1.01e-6 (all quantities 1.26e-6) | 2.80e-4 |
| N11 (no sort) | at most 3.2e-15 on the full-size record | n/a |

Spec section 14 says of the three misses that "two change no reported value by more than 2e-5 (measured)". That holds on the lead's 240 worlds and on my sets up to three prior scales; for strongly spread data M24 changes a probability by up to 8.5e-3 (still no declaration flip in the sets I tried). The claim needs the domain stated (AN-R10).

**My own 24 planted errors** (chosen to probe what the fixes added; committed list in `scripts/own_mut.py`, `own_mut2.py`; results `results/own_mut_results*.json`): 11 caught, 13 survive. What each survivor would do to a real run:

| ID | Planted change | Caught? | If it survives: what it would do |
|---|---|---|---|
| O01 | grid end point: GRID_SPAN 5 -> 4 (upper end of both spreads) | yes (test_pool_brute_force) |  |
| O02 | grid floor: TAU_FLOOR 1e-15 -> 1e-9 | **no** | probabilities change by up to 6.77e-5 for sets with s of 1e-6 to 3e-6 (H2_s1e-6_family_0.2); no declaration flips in the sets tried |
| O03 | grid spacing: spread grid linear in tau again from the floor (geomspace -> linspace), log-weight kept | yes (test_pool_brute_force) |  |
| O04 | A solves: Delta variance without the 1 in 1 + delta2 * 1'A^-1 1 | yes (test_pool_closed_form) |  |
| O05 | A solves: Delta mean uses the solve of the ones vector instead of d | yes (test_pool_closed_form) |  |
| O06 | A matrix: dataset spread added to the diagonal only for the first level (units term dropped from A) | yes (test_pool_brute_force) |  |
| O07 | 1e-6 threshold: pool accepts s equal to S_MIN no more (>= -> >) | **no** | s exactly 1e-6 is refused by `pool`; the spec says "at least 1e-6" |
| O08 | 1e-6 threshold: analyse excludes s equal to S_MIN (< -> <=) | **no** | s exactly 1e-6 is listed as `zero_spread` |
| O09 | 1e-6 threshold value: S_MIN 1e-6 -> 1.5e-6 | **no** | datasets with s between 1e-6 and 1.5e-6 are excluded, contrary to the spec |
| O10 | 1e-6 threshold: zero_spread tested on d == 0 instead of s < S_MIN | **no** | identical readings (s about 1e-16, not 0.0) abort the run again: the full-size identical-local-readings input exits after 3 s with no record |
| O11 | R4 recompute tolerance 1e-9 -> 1e-3 | **no** | a record whose b0, b_common or b* is off by up to 1e-3 is accepted; the tests tamper b0 by 0.01 and b* by 0.03 only (`test_analysis_x1.py:607-610`) |
| O12 | R4 recompute: b_common not compared | **no** | a record with a wrong b_common alone is accepted, so the area divisor is wrong (tamper T8 shows the real code refuses it) |
| O13 | R4 recompute: low-signal flag compared with 'or' instead of 'and' (flag mismatch alone not refused) | yes (test_refusals) |  |
| O14 | R4 recompute: seeds checked as a subset of 0-4 | yes (test_refusals) |  |
| O15 | t degrees of freedom of the sign table: m - 1 -> m - 2 | yes (test_end_to_end) |  |
| O16 | sort key: rows sorted by lambda instead of tag | **no** | equivalent change: any fixed row order gives the same numbers; the tests cannot tell which order is used |
| O17 | failed-run count counts ok runs instead of failed ones | yes (test_end_to_end) |  |
| O18 | failed-run total: max instead of sum | **no** | a family with several failed configurations reports the largest single count, not the total |
| O19 | gap flag: tests points in the window instead of points outside it | yes (test_window) |  |
| O20 | gap flag: looks only at the best point's neighbours (first interior point) | **no** | a below-bound point deeper inside the window does not set the gap flag |
| O21 | grid Jacobian: tau**2 weight instead of tau (log-grid change of variable) | yes (test_pool_brute_force) |  |
| O22 | grid point count 201 -> 151 | **no** | 3.25e-7 on sets with spread up to 3 prior scales, 9.75e-5 beyond; no flips |
| O23 | sign table: t degrees of freedom a constant 4 (right only when m = 5) | **no** | the sign table would use t with 4 degrees of freedom for every cell: wrong for a cell with 4 readings, where t is 3.182, not 2.776 |
| O24 | R4 recompute ignores the extension rows (computed on the six registered points only) | **no** | a valid record made with R4 extension rows is refused (2 of 16 cells in my synthetic run) |

These point at five gaps: the S_MIN boundary and constant (O07 to O10), the R4 tolerance, b_common and extension rows (O11, O12, O24), the grid floor and size (O02, O22), the failed-run total and gap position (O18, O20, O23), and the row order (O16). Suggested extensions of existing tests, not new tests, are in AN-R06 to AN-R09 and AN-R14. Every test that exists was run against the real code and passes; the survivors show what is not pinned, not that anything is wrong.

### 4.7 Check 7: specification against code

I read spec sections 0 to 16 against the code and tests at 08f1e3c, rule by rule (constants, the P1 reading and crossing rules, area corner, pooled model and priors, declaration thresholds, P2 intervals, local-versus-global, transfer lambda, best-of-k, window, sensitivities). Everything is implemented as written except these items:

| Spec location | What the spec says | What the code does | Finding |
|---|---|---|---|
| Section 1, lines 58-59 | b0, b_common, b* and the flag are read from the R4 record; "the analysis never recomputes them" | recomputes all four with `r4_stage` and refuses on a difference (`analysis_x1.py:369-378`; section 15.1 says so) | AN-R02 |
| Section 14 and usage (`analysis_x1.py:6-9`) | one `--manifest`; `docs/prereg/r1_record.json` | extension manifests and scores must be passed too; section 1 says the redone Sinkhorn R1 record is the input | AN-R03 |
| Section 14, test command | passes | prints FAIL at 08f1e3c (run_stage.py 662-663) | AN-R04 |
| Section 15.7 and the status paragraph (lines 11-13) | local evaluable "exactly when it is for P1"; code frozen at ff88666 / 858101b | zero_spread is decided per endpoint (`:454`), so local can drop out while P1 stays; the fixes are at 8b90d79 | AN-R05 |
| Section 4, lines 141-143 | a dataset enters the model only if s >= 1e-6 | applied to the dataset's own s; in the pairs sensitivity the pair's averaged s (`:232`) is not checked first | AN-R01 |
| Section 1, lines 68-71 | "the report gives the number of such cells by arm, next to that arm's best batch score in the R4 record" (cells whose six points never reach b*) | the record keeps each cell's per-arm status (`p1.status`) and the R4 record keeps the arms' best batch scores (`M`); nothing counts or pairs them | AN-R15 |

Statements in the spec that I checked against evidence and found right: the 240-world numerical validation and the figure 6e-7 for the prior mass beyond 5 scales (measured 5.7e-7 per spread); "54 of 57" (re-run); the pair's standard error 41% larger (arithmetic); the critic steps of the two arms (reference, pooled, barycenter 5; discriminator 1; the other penalty types 0; plain scVI seeds 0 to 4, 80 rows) in the X1 manifest; the amended Holm statements; the SI-54 tag requirement.

**Behaviours of the code that the spec does not state** (none changes a result): (1) the manifest's seed separation is enforced, so the pilot, X1 and follow-up seeds must be pairwise disjoint (`analysis_x1.py:516`, `prereg_rules.check_seed_separation`, SI-13 and SI-26); (2) a failed or duplicated plain-scVI run is refused (`prereg_rules._zero_ref`, through `zero_runs`, `:59-61`); (3) a batch score that is not (3 local + 2 global)/5 within 1e-9 is refused (`:54`); (4) the collapse rows and the transfer-lambda inputs are guarded (statuses, paired seeds, finite scores, equal lambda grids: `:277-283`, `:291-299`); (5) the R4 recompute runs inside every one of the nine analyses, not once. Items 1 to 3 are refusals a user could meet; one sentence in section 14 would prepare them.

### 4.8 Check 8: full-size dry run

**What I did.** The first review's synthetic input: the real X1 design (3,000 rows) and X13 rows (408), 3,303 scored runs and 105 failed ones, an R4 record written by the real `r4_stage`; all eight tasks and both decoders. No real score was used.

| Run | Code | Exit | Time | Peak memory | Output |
|---|---|---|---|---|---|
| Main input | 08f1e3c | 0 | 30.4 s | 205 MB | 4.16 MB |
| Main input | 1e615eb | 0 | 24.5 s | 202 MB | 4.06 MB |
| Main input, rows shuffled | 08f1e3c | 0 | 28.2 s | 205 MB | 4.16 MB |
| Every area collapsed (lung, decoder 0) | 08f1e3c | 0 | 28.2 s | 205 MB | 4.16 MB |
| same | 1e615eb | 1 (refused/aborted) | 2.4 s | 190 MB | none |
| Identical P1 readings (pancreas, decoder 0) | 08f1e3c | 0 | 28.1 s | 205 MB | 4.16 MB |
| same | 1e615eb | 1 (refused/aborted) | 2.1 s | 146 MB | none |
| Identical local readings | 08f1e3c | 0 | 25.8 s | 205 MB | 4.12 MB |
| same | 1e615eb | 1 (refused/aborted) | 2.5 s | 191 MB | none |
| Low-signal cells, decoder 1 | 08f1e3c | 0 | 24.5 s | 205 MB | 4.11 MB |
| Mixed scorer commits, tag present | 08f1e3c | 0 | 27.6 s | 204 MB | 4.16 MB |
| same, tag absent | 08f1e3c | 1 (refused/aborted) | 0.9 s | 141 MB | none |

**Old record against new record** on the same main input: 4,320 new keys (864 `gap`, 1,728 `configs`, 1,728 `failed_runs`); **exactly one label differs**: sensitivity prior 0.2, pooled, decoder 0, local endpoint, `inconclusive` -> `above` (the first review's AN-01 case, now correct); 1,168 numbers differ by more than 1e-9. In the primary record the largest changes are: pooled P1 5.70e-7, pooled area 1.71e-6, P2 1.38e-6, pooled local 4.44e-4 (the first review's AN-01 case, s from 1.8e-4 to 1.4e-3), sign-table limits 2.01e-2 (t instead of 1.96). In the prior-0.2 sensitivity: pooled local 2.68e-2, P1 3.60e-5. Primary declarations are identical.

**Cost.** The new code takes 6 s longer (30.4 s against 24.5 s, +24%), memory is the same, the record is 2.4% larger. A profile of the new run shows `r4_stage` called 9 times (the primary analysis and the 8 sensitivities), 13% of the profiled time; the recomputed targets are the same each time and could be computed once in `main`. Optional.

**Refusals the real input could trigger** (all produce a message and no record): R4 extension rows present but their manifest or scores not passed (AN-R03); an R1 record that still holds the superseded Sinkhorn grid ("registered lambda grid is incomplete", as spec section 1 says); scores of two scorer commits without the local tag `prereg-tier12-v3` (section 4.4); plain-scVI seeds other than 0 to 4; an existing output file; a failed plain-scVI run; and, in the pairs sensitivity only, two datasets of a family with s just above 1e-6 (AN-R01). No crash without a message appeared in any run.

### 4.9 Check 9: the five design choices

**1. Log-spaced grid for the two spreads (AN-01).** *Sound on ordinary data.* A log grid gives the small end of each spread, where s can fall between points, the same resolution as the large end; the weight tau is the change of variable (a planted error that squares it is caught, O21). The accuracy is about 1e-9 for ordinary inputs (section 4.2). It is limited at the top: the grid stops at 5 prior scales, and when the data call for a larger spread (datasets differing by more than about three prior scales) the omitted mass and the trapezoid end correction give errors of 1e-5 to 3e-4. *Alternative:* a graded Gauss-Legendre rule with the same node count, as the first review suggested, resolves both ends; I did not test an extended span or this rule, so I cannot say what error either leaves. The cheaper course is to state the domain (AN-R10). No declaration was affected in any set I tried.

**2. Posterior of Delta from solves with A (the "precision" fix).** *Sound.* It replaces a subtraction of nearly equal numbers. Against 40-digit values the variance of Delta given the spreads is correct to 4.22e-16 relative at every case tested, where the old formula had relative errors of 5.0e-3 (one small s of 1e-5), 0.22 (one small s of 1e-6) and 2.9e+5 (all s equal to 1e-6; `precision_mpmath.csv`, reproduced). The posterior mean given the spreads is correct to 2.87e-12 when one s is small and to 9e-11, 9e-9 and 8e-7 when all s equal 1e-4, 1e-5 and 1e-6 (old: 7e-10, 3e-7, 6e-6); those are errors at one grid point of the spreads, and the end-to-end errors are in section 4.2. What remains is the weight of each (tau_f, tau_d) point: it still uses the inverse and log-determinant of V = A + prior variance x (ones matrix), which my 50-digit check bounds at about 1e-9 at s = 1e-6 and 2e-7 at s = 1e-7. *Alternative:* the matrix-determinant lemma and the Sherman-Morrison identity for these weights remove the remaining dependence on small s; the gain (1e-9) is below any decision threshold, so it is optional.

**3. Exclude a dataset below s = 1e-6 (`zero_spread`) instead of flooring s (AN-02).** *Defensible, with one referee-visible cost.* Exclusion is right for the cases that motivated it: a structural zero (neither arm adds area in any leave-one-seed-out repeat) and a rounding-level s of 1e-16 carry no information about noise. The constant 1e-6 is a numerical bound (section 4.2; the code is verified to it) and sits 319 times below the smallest pilot s, so it excludes no pilot cell. A floor large enough to matter statistically would have to be of the order of 1e-3 and would alter real pilot cells (2 of 18 pilot values are below 1e-3). The cost: the rule is a cliff (a dataset with s = 1.1e-6 is treated as measured exactly and so pins its own effect and informs the spreads strongly; one with s = 0.9e-6 is dropped), and the exclusion depends on the data. *Alternative a referee may ask for:* a floor tied to the other datasets' s, or a stabilised jackknife variance, so the decision does not depend on a threshold. Whether exclusion or floor is preferred is the user's decision and the spec records it as such; the code tests only the exact-zero input (AN-R06).

**4. Recompute the R4 targets instead of trusting the record (AN-03).** *Sound.* It uses the same function that wrote the record, so there is one definition of b0, b_common, b* and the flag, and it refused all seven tamperings of section 4.4. Costs: it needs every input of the R4 stage (extension manifests and scores: AN-R03), the refusal message does not say which quantity differs, and the recompute runs 9 times. *Alternative:* store the SHA-256 of the scores that made the record and compare it (cheaper; the lead rejected it because the existing records carry no hash; it would show that the record was written from these files but not that its numbers are right, so a hand-edited b* beside an intact hash would pass); or build the R4 record from the six registered points only, which removes the extension-row problem and makes the record agree with what the analysis reads (AN-09). The chosen design is the stronger of the three.

**5. Sign-table limits d +/- t(0.975, m-1) s (AN-06).** *Sound, and approximate.* It corrects the 1.96 for 4 or 5 readings (2.776 for m = 5). The limits stay approximate because a leave-one-seed-out s is a jackknife standard error of a non-smooth statistic, so the ratio d/s is not exactly t distributed. The sign table is explicitly descriptive (spec section 4: "without per-dataset significance claims"). *Alternative:* label the limits approximate, or use a rank-based or bootstrap interval over the five seeds. I would add one sentence stating they are approximate.

### 4.10 Check 10: referee pushback and what the fixes may have broken

Points an external reviewer is likely to raise, each tied to evidence:

1. **Weakly identified between-family spread.** tau_f rests on four families, two with two datasets each (spec section 16 item 5). With the default family structure the model cannot declare a direction from fewer than three evaluable datasets: when all datasets agree exactly (d up to 0.5, s from 1e-6 to 0.01, prior 0.1), P(Delta > 0.01) tops out at 0.82 for one dataset and 0.94 for two; three datasets need d of about 0.03 (s = 1e-6) to 0.08 (s = 0.005 to 0.01). Conditioned-decoder cells that are low-signal drop out, so a decoder with one or two evaluable datasets can never read 'above' or 'below'. The spec says nothing about this. Finding AN-R12 (note).
2. **s treated as known**, from 4 or 5 jackknife readings (section 16 item 1). Stated; the sign limits are now t-based.
3. **No multiplicity control across six marginal declarations** (item 4); only the descriptive Wilcoxon companions get Holm.
4. **The cliff at S_MIN and data-dependent exclusion** (design 3), and the unpinned boundary (AN-R06).
5. **Window and best-of-k selection**: failures are counted but not penalised (AN-13 closed); the window gap flag misses an interior point with no successful run (AN-R13).
6. **Numerical-accuracy claims** quoted as "within 2e-6" (spec section 15.4) and "2e-5" (section 14): true on the lead's synthetic worlds, domain-limited in general (AN-R10); the reference is not independent of the code (AN-R11).

What the fixes may have broken, and what I found: (a) a valid record is refused when R4 extension rows exist and the command omits them (AN-R03); (b) the pairs sensitivity can abort (AN-R01); (c) the run takes 24% longer; (d) sign-table limits are 41% wider than before (2.776 against 1.96), by design; (e) values differ from the old code in the last digits through the sort, and in the 5th to 6th digit through the grid (section 4.8), with one corrected declaration in a sensitivity analysis; (f) no primary declaration changes in the full-size dry run. Nothing else broke: the 193 existing tests pass, and the full-size run, the zero-spread variants and the tamper runs behave as specified.

## 5. New findings (AN-R01 to AN-R15)

Severity: **fix** = change before the tag (a document or procedure that contradicts the code, or that sends the user into a refusal); **note** = optional, or a question. No blocker. Locations are at 08f1e3c.

**AN-R01** (note). *Location:* `analysis_x1.py:229-232`, `:156`, `:454-455`; spec section 4 lines 141-143.  
*Problem:* In the pairs sensitivity the pair's standard error sqrt(s1^2 + s2^2)/2 can fall below S_MIN although each dataset's s passed the rule. `pool` then refuses and the whole run aborts with no record.  
*Evidence:* `A.combine(['atac_small','atac_large'], [0.02,0.02], [1.2e-6]*2, 'pairs', 0.1, 0.01)` is refused with PreregError; each s of 1.42e-6 or more is accepted; family and independent structures accept 1e-6 (`logs/small_demos.txt`). Trigger: both datasets of a family have s between 1e-6 and about 1.7e-6.  
*Suggested fix:* Apply the S_MIN rule to the averaged s too (list the family as `zero_spread` in that sensitivity), or check pair s before calling `pool`; add one parameter row to the existing pairs test.

**AN-R02** (fix). *Location:* spec section 1 lines 58-59; `analysis_x1.py:369-378`; spec section 15.1.  
*Problem:* The spec says the analysis "never recomputes" b0, b_common, b* and the low-signal flag; the code recomputes all four with `r4_stage` and refuses on a difference, as section 15.1 says.  
*Evidence:* Spec text at 08f1e3c; tamper runs T1 to T8 (section 4.4).  
*Suggested fix:* Replace the clause with: the analysis reads them from the record and refuses unless `prereg_rules.r4_stage` gives the same values from the same rows (section 15.1).

**AN-R03** (fix). *Location:* `analysis_x1.py:6-9`; spec section 14 (run commands) and section 1 lines 62-65.  
*Problem:* The usage line and section 14 give one `--manifest` and `docs/prereg/r1_record.json`. When R4 added extension rows the extension manifest and scores must be passed as well, otherwise a valid record is refused with a message that does not name the cause. Section 1 says the redone Sinkhorn R1 record is the one to pass, and that its non-Sinkhorn grids are checked against `r1_record.json` first: that check is a procedure, not code.  
*Evidence:* Synthetic run with 240 extension rows that move b* in 2 of 16 cells: both manifests and all scores accepted (28 s); base manifest only refused for immune_hum_mou, decoder 1 (1.4 s); record built without extension rows refused (`dry_runs/ext_runs`). `docs/prereg/r1_record_sinkhorn_redo.json` does not exist at 08f1e3c.  
*Suggested fix:* State in the usage line and section 14: pass every R4 extension manifest as a further `--manifest` and its scores with `--scores`, and the failures file that covers all rows; name the redone R1 record; say who checks its non-Sinkhorn grids. Make the refusal print the recomputed and recorded values of the four quantities for the cell.

**AN-R04** (fix). *Location:* spec section 14 (test command); `scripts/check_tests.py`; `scripts/run_stage.py:662-663`.  
*Problem:* The test command of section 14 prints `FAIL scripts/run_stage.py: changed lines never run: [662, 663]` at 08f1e3c, so the user cannot reach the OK the section implies. The lines are the CF-02 change (code track), covered by `tests/stage`.  
*Evidence:* Same command at 8b90d79: OK. At 08f1e3c: `--src scripts/analysis_x1.py` gives OK (193 passed, 1 skipped); adding `tests/stage` gives OK (259 passed, 2 skipped, 6 min). Logs `logs/log_check_tests_*.txt`.  
*Suggested fix:* Give the base and the source in section 14: `--base 973a97d --src scripts/analysis_x1.py -- tests/analysis tests/prereg`, or add `tests/stage` to the list and say it takes 6 minutes.

**AN-R05** (note). *Location:* spec section 15.7 (line 312) and the status paragraph (lines 11-13).  
*Problem:* Section 15.7 says local is evaluable "exactly when it is for P1"; since AN-02 zero_spread is decided per endpoint, so local can drop out while P1 stays. The status paragraph names ff88666 and 858101b as the code commits; the fixes are in 8b90d79.  
*Evidence:* Identical-local-readings input: all local cells `zero_spread`, pooled local n = 0, P1 evaluable (section 4.3).  
*Suggested fix:* Reword 15.7 ("evaluable for P1 and with s >= 1e-6 on the local reading") and update the commit names.

**AN-R06** (note). *Location:* `analysis_x1.py:454`, `:31`, `:156`; `tests/analysis`.  
*Problem:* The zero_spread rule is tested only with exact zeros. Planted changes survive: `>=` to `>` in `pool` (O07), `<` to `<=` (O08), S_MIN 1.5e-6 (O09), and `s == 0` instead of `s < S_MIN` (O10), which would abort the identical-readings input again. Open question for the user at the diff review: exclusion below s = 1e-6, as written, or a floor tied to the other datasets' s (section 4.9, design 3).  
*Evidence:* `results/own_mut_results.json`; O10 on the full-size identical-local input exits after 3 s with no record.  
*Suggested fix:* Extend the existing zero-spread case with s = 1e-6 (accepted), 0.9e-6 (excluded) and 1e-16 (excluded).

**AN-R07** (note). *Location:* `analysis_x1.py:369-378`; `tests/analysis/test_analysis_x1.py:447-453, 606-610`.  
*Problem:* Tests tamper b0 by 0.01, b* by 0.03 and the flag only. Surviving planted errors: tolerance 1e-3 (O11), b_common not compared (O12), recompute ignoring extension rows (O24). The fixture's extension row does not move b*.  
*Evidence:* Section 4.4: the real code refuses b_common alone (T8) and accepts 5e-10 (T4a), 2e-9 refused (T4b).  
*Suggested fix:* In the existing refusal test add a b_common-only change and a 2e-9 change on b*; move the fixture's extension point to the arm that sets b_common (barycenter, atac_small, decoder 0) and give it a batch score high enough to raise b_common, so that b* shifts.

**AN-R08** (note). *Location:* `analysis_x1.py:28-30`; `test_pool_closed_form`, `test_pool_brute_force` in `tests/analysis/test_analysis_x1.py`.  
*Problem:* The grid floor and size are unpinned: TAU_FLOOR 1e-9 (O02) changes probabilities by up to 6.77e-5; GRID_N 151 (O22) and 101 (M24) survive.  
*Evidence:* Section 4.6 (effects of survivors).  
*Suggested fix:* Add one known-answer case with s of 1e-6 to 3e-6 (the committed 40-digit values are available in `precision_mpmath.csv`) and compare at 1e-8.

**AN-R09** (note). *Location:* `analysis_x1.py:415-419, :431-434` (failed_runs), `:338-341` (gap), `:460` (t quantile); tests.  
*Problem:* Three added outputs are only partly pinned: the failed-run total (O18 uses the largest count), the gap flag beyond the first interior point (O20), and the t quantile for a cell with 4 readings (O23).  
*Evidence:* Section 4.6; my independent recomputation of the full-size record found no mismatch (section 4.5).  
*Suggested fix:* One fixture cell with two failed configurations; one window with a low point deep inside; one sign-table cell with 4 readings (add to the existing end-to-end or window test).

**AN-R10** (note). *Location:* spec sections 14 and 15.4; `experiments/analysis_numerics_20261014/README.md`; `analysis_x1.py:29`.  
*Problem:* The accuracy statements ("within 2e-6", "no reported value by more than 2e-5") hold on the 240 synthetic worlds and for data spread up to about three prior scales. Beyond that the code's error is 1e-5 to 3e-4 (prior 0.05 or 0.1; section 4.2), and the surviving planted errors M24 and M28 change a probability by up to 8.5e-3 and 2.8e-4.  
*Evidence:* Truncation scan and random battery (section 4.2); no declaration changed anywhere.  
*Suggested fix:* Add the domain to the two sentences ("for data spread up to three prior scales"), or switch to a graded Gauss-Legendre grid and re-run `check_pool_numerics.py`.

**AN-R11** (note). *Location:* `experiments/analysis_numerics_20261014/check_pool_numerics.py`.  
*Problem:* The reference in the numerical check uses the same A-solve and V-based weights as the code, on a finer grid; the independent 40-digit table covers only the posterior mean and variance for given spreads. A shared algebra error would not show. My independent implementation agrees (section 4.2).  
*Evidence:* Code reading; `scripts/ref_pool.py`, `ref_mp.py`, `validate_algebra.py`.  
*Suggested fix:* State in the README what the reference shares with the code; optionally add an independent weight check (the block algebra in `ref_pool.py` is 60 lines).

**AN-R12** (note). *Location:* spec sections 4 and 5; `analysis_x1.py:215-234`.  
*Problem:* With the family structure, one evaluable dataset cannot reach a declaration (ceiling about 0.82) and two cannot either (about 0.94); three need d of about 0.03 to 0.08, depending on s. Conditioned-decoder cells that are low-signal drop out, so this can happen in the real run.  
*Evidence:* `scripts/ceilings2.py`, `results/ceilings2.json`: all datasets equal, d up to 0.5, s from 1e-6 to 0.01, prior 0.1.  
*Suggested fix:* Say in the spec that with fewer than three evaluable datasets a decoder's pooled result is inconclusive by construction, and keep reporting n beside each declaration (it already is).

**AN-R13** (note). *Location:* `analysis_x1.py:338-341`.  
*Problem:* The gap flag looks only at configurations that have at least one successful run: a lambda between the window's ends whose runs all failed is invisible, so `gap` is false and the width spans it.  
*Evidence:* Window demo: points at lambda 1 and 10 with an empty point between: lams [1, 10], gap False; with a low (not failed) interior point gap True (`logs/small_demos.txt`).  
*Suggested fix:* Count a configuration with no successful run as an interior point that is not in the window, or add `n_all_failed_between` to the record.

**AN-R14** (note). *Location:* `analysis_x1.py:521`; spec section 4 line 140; tests.  
*Problem:* No test shuffles the input rows; removing the sort (N11) and sorting by lambda instead of tag (O16) both survive. The effect of the sort is at the 3e-15 level on the full-size record, so it matters only for the "digit for digit" wording.  
*Evidence:* Shuffle runs, section 4.5: 7,864 numbers differ without the sort, largest 3.2e-15.  
*Suggested fix:* In the existing end-to-end test compare the record built from a shuffled copy of the rows (about 1 extra second), or soften the sentence to "identical declarations and numbers identical to 1e-12".

**AN-R15** (note). *Location:* spec section 1 lines 68-71; `analysis_x1.py:108`.  
*Problem:* The spec says that when an arm's six registered points never reach b*, the cell is not evaluable and "the report gives the number of such cells by arm, next to that arm's best batch score in the R4 record". The analysis record keeps the per-cell, per-arm status (`p1.status`) and the R4 record keeps the arms' best batch scores (`targets.*.M`), but no field or script produces the count or the pairing.  
*Evidence:* Full-size synthetic record: `cells.*.p1` has bio, crossings, d, m, mode, readings, reason, s, status and no count by arm appears in `primary` or `sensitivity`; `unreached` occurs only at `analysis_x1.py:108`; the R4 record's `targets['lung|0']` carries `M` for the six arms.  
*Suggested fix:* Add a field (arm to number of unreached cells, with M) to the record, or say in the spec that a reporting script written after the run tabulates it, as PREREG:431 does for the no-pair counts. Question for the lead: is such a script part of what the tag freezes?

## 6. Change-size review

Is the change as small as the finding requires? Changed lines between 1e615eb and 08f1e3c (`git diff --numstat`, the four commits):

| File | Added | Deleted | Reading |
|---|---|---|---|
| `scripts/analysis_x1.py` | 40 | 20 | every changed line maps to one finding (constants and guard AN-02; log grid AN-01; solve with A, found while checking AN-01; `gap` AN-14; recompute and seed check AN-03, AN-10; `configs`, `failed_runs` AN-13; zero_spread AN-02; t quantile AN-06; sort AN-11). No unrelated edit |
| `tests/analysis/test_analysis_x1.py` | 110 | 32 | 17 test functions before and after; the brute-force test became parametrized (53 to 62 collected cases); no redundant test (check_tests says so) |
| `docs/ANALYSIS_SPEC.md` | 317 | 114 | includes the restructuring into rules plus decision log (DOC-13, documents track); the AN-related edits are 8b90d79 (57 lines changed) and the limitations section 16 (56febe4: 44 added, 3 deleted) |
| `experiments/analysis_numerics_20261014/` (new) | 1,315 | 0 | 10 files: 336 lines of code (numerical check 214, planted-error runner 78, mutant list 44), the rest result tables |

Questions of the change-size review, each with the concrete case:

| Question | Answer |
|---|---|
| A bug (wrong result, crash, missed edge case, broken caller, fix in one caller only) | AN-R01: the pairs sensitivity aborts when the averaged s falls below S_MIN. AN-R13: the gap flag cannot see an interior point with no successful run. Edge cases I ran and found right: no evaluable dataset (`combine` returns `not_evaluable`, `:216-217`), s of exactly 0 and of 1e-16 (zero_spread), the last grid point (end weights; planted M28 and O01 are caught or measured), a missing plain-scVI seed (refused), one dataset (fits, inconclusive). `combine` has two call sites, both in `analyse` (`:462` and `:474`, the second for P2), and all three structures ran in the dry runs. S_MIN is enforced in `pool` (shared) and in `analyse` (caller) and the pairs path shows they can disagree (AN-R01). No 1.96 constant is left (`grep`) |
| A risk (errors swallowed, data loss, writes in the wrong order) | None found. No `except` in the file; the record is written once, after all nine analyses succeed, so a refusal leaves no partial file (the 1.4-s refusals wrote none); the existing-output refusal comes before any work (`:514`) |
| A missing test | Yes: AN-R06 to AN-R09 and AN-R14, which together cover the 13 surviving planted errors, each with the input that would show it |
| An extra test | None: `check_tests` reports no redundant new test; 17 test functions before and after |
| Code that should not exist or be smaller | No dead code (`vd`, `inverse` and `V` are still used in `pool`). One repeated computation: the R4 recompute runs 9 times (optional) |

**Assessment.** The code change is as small as the ten findings allow (60 changed lines; the largest, the solve with A, is also the fix for a cancellation error the first review had not named). The tests grew by 78 lines in the existing functions. The evidence folder is larger than the fix (1,315 lines), which is reasonable for a numerical claim, but it should say what its reference shares with the code (AN-R11). The cost is 6 s more runtime (+24%) and a record 2.4% larger. One removable cost: the R4 recompute runs 9 times on identical input (13% of profiled time); computing it once in `main` and passing the result to `analyse` would cut about 4 s (optional; a change to frozen-script structure should wait for the diff review).

## 7. What this check did not cover (limits and deviations)

1. **Data.** Only synthetic inputs on the real design, plus the A1 pilot (3 seeds, 10-point grid, Sinkhorn dropped, 6 cells). No real X1 or X13 score was opened or used; the blinded locations were not touched. Statements about real data are therefore statements about how the code behaves, not about results. One synthetic world per dry-run variant (the first review's seeds), so counts such as "one label differs" belong to that world.
2. **Numerics.** 400 random sets, 38 hand-built sets and 20 threshold sets: not exhaustive. The random battery used a coarse reference (panels 1.0 wide, 8 nodes) except for the 18 sets above 1e-6, which were re-evaluated with the finest; errors of about 2e-7 or less in that group may be the reference's own error. The 50-digit runs covered 7 headline sets with a coarser integration (panel 1.5) and are not converged for strongly spread data. Per-dataset posteriors (the reported theta_t) were not compared with an independent computation; they use the same formulas as before, and the fixes changed only the grid under them. The random generator is my own, not the lead's.
3. **Planted errors.** The lead's run was repeated with 4 parallel slots instead of 6 (same result). My 24 errors are a chosen set aimed at what the fixes added, not an exhaustive one. Effect sizes of survivors come from 218 synthetic sets.
4. **Procedure.** The tag `prereg-tier12-v3` was created only in a private clone. `tests/stage` was run for coverage only; `tests/scvi`, `tests/jhpce` and legacy tiers were not run. I used clones instead of worktrees. A first `check_tests` run was spoiled by an untracked virtual environment in my clone and repeated clean. The process cap (4) was exceeded once, for about a minute (5 processes; I interrupted one job and restarted it). The ponytail-review skill is not available to me; I applied its change-size questions by hand (section 6).
5. **Not repeated.** The simulation figures quoted in spec section 16 items 1 and 3 (first review's simulations). No literature search was run, and I make no novelty claim: the fixes use established methods (log-scale quadrature, t intervals, recomputation of a derived record).
6. **Scope.** This is track 1 of 3 (analysis code, its tests and spec, and the numerical folder). The code track and documents track were not reviewed here.

## 8. Files and how to reproduce

Environments: `scvi-api` for the repository's scripts and tests (CPU only: `CUDA_VISIBLE_DEVICES=` empty; `KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL`; `WCD_SRC=src`); `python` (numpy 2.4.6, scipy 1.17.1, mpmath 1.4.1) for my reference. A clone of the git bundle at 08f1e3c was used for every run (`rv`); old-code runs used a clone at 1e615eb. Key commands (from the clone root):

```
python -m pytest tests/analysis tests/prereg -q
python scripts/check_tests.py --base 973a97d --src scripts -- tests/analysis tests/prereg           # FAIL at 08f1e3c (AN-R04)
python scripts/check_tests.py --base 973a97d --src scripts/analysis_x1.py -- tests/analysis tests/prereg   # OK
python experiments/analysis_numerics_20261014/check_pool_numerics.py                                  # 4 min; output identical to committed
python experiments/analysis_numerics_20261014/mutation/run_mut.py                                      # 57 planted errors, about 8 min on 4 slots
```

In the evidence archive (`recheck_analysis_evidence_20261014.tar.gz`): `scripts/` (my code: `ref_pool.py` independent reference, `compare_code.py`, `random_battery.py`, `headline_*.py`, `run_mp.py`, `trunc_scan.py`, `smin_scan.py`, `own_mut*.py`, `dryrun_build2.py` + `build_cli.py` + `run_dry.py` + `diff_records.py`, `check_records.py`, `small_demos.py`, `four_of_five_demo.py`, `ceilings.py`, `ceilings2.py`, `review_case_independent.py`, `dry_diff_facts.py`, `build_facts.py`, `make_report.py`), `results/` (the numbers in this report: `facts.json` first), `logs/`, `dry_runs/` (run records and analysis records of the synthetic dry runs; no real data) and `README.txt`. Paths such as `logs/...`, `results/...` and `scripts/...` in this report are paths inside that archive. Scripts were run from a folder `rc/` beside the clones; set `CODE_SCRIPTS` to the repository's `scripts` folder to point them elsewhere.

*Rigor Reviewer, independent re-check, track 1 of 3. Review only: I changed no code, test or document of the project; the findings above are suggestions for the lead.*

