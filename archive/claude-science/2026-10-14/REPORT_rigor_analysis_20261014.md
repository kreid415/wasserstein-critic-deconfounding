# Independent rigour review, track 1 of 3: the frozen X1 analysis code

**Reviewer:** Rigor Reviewer (independent; read-only on the project). **Date:** 2026-10-14.
**Under review:** `scripts/analysis_x1.py` (515 lines) and `tests/analysis/test_analysis_x1.py` (559 lines), branch `final-review-fixes`, commit `1e615eb321097729c0fb4f117bc4e9fd1e0ac9a8` (commits ff88666 and 858101b).
**Specification:** `docs/ANALYSIS_SPEC.md` sections 1-11 and 15; the user's decisions SI-68 to SI-77.
**Blinding:** I did not open, list or compute from any real X1 or X13 score. Every run below used synthetic inputs or the pilot (A1) data.
**Code locations** are written `file:line` for the commit above. Numbers in this report were filled in from saved result files (`rv_out/facts.json`, built by `rv_out/build_facts.py`), not typed.

## 1. Verdict: PASS WITH FIXES

**In plain words.** The frozen analysis code computes what the specification says it should. On a full-size synthetic input, my independent re-implementation reproduced 41,752 numbers from the code's record (the primary analysis and the eight sensitivity analyses) except for 27 differences beyond my tolerance (0.0002 for probabilities, means and interval ends; 0.0005 for heterogeneity and per-data-set intervals; labels, counts and declarations must match exactly), all of which trace to one numerical weakness (AN-01). Every expected value in the test file (60 values in 15 tests) is correct. 7 of 7 broken inputs I fed to the real command line were refused with a clear message, and the unbroken control ran. I found no blocker: no wrong result on ordinary inputs. Five items should change before the tag `prereg-tier12-v3`, because each can change a reported result, abort the run, or let a later edit slip through unnoticed:

| ID | In one sentence | What can go wrong on the real data |
|---|---|---|
| AN-01 | The numerical grid of the pooled (Bayesian) model is too coarse when a data set's standard error is small. | Probabilities off by up to 0.32 and a wrong declaration ("above" or "equivalent" instead of "inconclusive"); in the pilot, the local-mixing endpoint had standard errors that small in 2 of 6 cells. |
| AN-02 | A data set whose standard error is exactly zero aborts the whole analysis; no record is written. | Plausible when both arms collapse at all six penalty strengths in one cell. |
| AN-03 | Only one of the four numbers of the R4 record (`b0`) is checked against the data. | A record made from partial scores passes and sets the wrong matched batch-removal target `b*`. |
| AN-04 | The tests miss 10 of the 40 errors I planted, including lowering the 0.95 declaration threshold to 0.90. | Any later edit, including the fixes above, is guarded by tests that cannot see these lines. |
| AN-05 | The specification says the per-run-reading sensitivity also applies to the area; the code leaves the area unchanged. | The sensitivity table would not match the specification. |

**Why AN-01 is a fix and not a blocker.** The six primary results were reproduced to within 1e-5 on the dry run and the weakness needs standard errors below about one grid step, which the pilot shows for the local endpoint only; the repair is mechanical. If the local-versus-global declaration or a prior-sensitivity result is to carry a headline claim, treat AN-01 as a blocker for that claim until it is repaired.

Thirteen further findings are notes (optional). Totals: 18 findings = 5 fix + 13 note + 0 blocker.

**What the code already does well** (these close objections a referee would otherwise raise; do not change them): it uses no random numbers; it refuses ambiguous input instead of defaulting (see claim 4); the exact Wilcoxon, Clopper-Pearson, Holm and best-of-k routines agree with exact-arithmetic re-derivations; the leave-one-seed-out standard error is unbiased on the variance scale in my simulations (AN-07); the pooled model is calibrated against its own prior (simulation-based calibration, AN-07); a rerun, or a run with 4 numerical-library threads, is byte-identical (AN-11).

## 2. Terms used in this report

- **X1 analysis**: the pre-registered script `scripts/analysis_x1.py`. It turns the scores of the main experiment X1 (8 data sets x 2 decoder settings x 6 penalty types x 6 penalty strengths x 5 random seeds) and of the baseline methods X13 (Harmony, Scanorama, PCA, sysVI, scANVI) into result tables and decisions. "Frozen" means committed and tagged before any real X1 score is looked at.
- **Arm**: one penalty type added to scVI (a variational autoencoder used to integrate single-cell data). `reference`, `pooled`, `barycenter` are Wasserstein critics (W1); `discriminator` is the usual adversarial classifier; `mmd` and `sinkhorn` are non-adversarial distances; `none` is plain scVI (penalty strength lambda = 0). **lambda** is the penalty strength; each arm has six registered values per decoder setting.
- **B and C**: the batch-mixing score and the biology-conservation score of a run (means of scIB metrics, between 0 and 1). **Local** batch metrics: iLISI, kBET, graph connectivity; **global**: PCR_batch, ASW label/batch. B = (3 local + 2 global) / 5.
- **b\***: the matched batch-removal target: halfway between plain scVI's batch score and the batch score every arm can reach (decided by rule R4).
- **P1**: how much biology the reference critic keeps minus how much the discriminator keeps, each read at batch score b\* on its own penalty-strength curve (linear interpolation between neighbouring points). **d** is this difference in one data set; positive = critic keeps more.
- **Area** (frontier area): the area between an arm's batch-versus-biology curve and a corner point (plain scVI's batch score, its biology score minus 0.10). **P2**: the collapse rate, the share of runs that failed or lost at least 0.10 of biology compared with the same seed's plain scVI run. **Local endpoint**: the P1 reading repeated with only the local batch metrics.
- **s** (leave-one-seed-out standard error): re-read d with each of the 5 seeds removed in turn and take sqrt((m-1)/m x sum of squared deviations of the m readings): a jackknife standard error from 4 or 5 readings.
- **Pooled model / declaration**: a Bayesian model that combines the six data sets' d into one overall difference Delta (with family and data-set random effects, half-normal prior spreads, integrated on a grid). The result is "above" if P(Delta > 0.01) >= 0.95, "below" if P(Delta < -0.01) >= 0.95, "equivalent" if P(|Delta| <= 0.01) >= 0.95, else "inconclusive".
- **Dry run**: running the real script on a synthetic full-size input (real design, invented scores). **Planted error (mutant)**: a deliberate one-token change to the code, used to test whether the tests notice. **Converged value**: my independent high-accuracy calculation (Gauss-Legendre quadrature).
- **Grid step** h: the spacing of the code's grid, 5 x prior scale / 200 (0.0025 at the registered scale 0.1; 0.005 at 0.2; 0.00625 for P2).

## 3. The eight claims

| # | Claim (short) | Status | Evidence |
|---|---|---|---|
| 1 | Every statement of the specification (sections 1-11, 15) is implemented as written. | partly | I read each statement against the code (anchors in section 4). One item is implemented differently: section 15.11 applies the per-run reading to the area, the code does not (`analysis_x1.py:392-393`; AN-05). Behaviours the specification does not state: seeds are inferred from the plain-scVI rows (AN-10); a zero standard error aborts the run (AN-02); only `b0` of the R4 record is verified, the other three numbers are trusted (AN-03; section 15.1 states the b0 check, not the trust); failed runs are invisible to best-of-k (AN-13); the window can span gaps (AN-14, follows the wording of section 10). |
| 2 | Every expected value in `tests/analysis` is correct. | confirmed | 60 of 60 expected values in 15 tests re-derived with my own code (`rederive_tests.py`, `my_ref.py`: exact fractions for P1, area, Wilcoxon, sign test, Holm, Clopper-Pearson by bisection, collapse boundary, Agresti-Caffo, transfer lambda, best-of-k, window, dominance; independent quadrature for the pooled model). None was derived by calling `analysis_x1.py`. |
| 3 | The pooled model is right: covariance, posteriors, grid weights, prior, interval inversion, declarations; grid error at scales 0.05, 0.1, 0.2, 0.25; the leave-one-seed-out standard error judged. | partly | The algebra matches an independent precision-form implementation with Gauss-Legendre quadrature on 6 data sets (largest difference 3.0e-07 on 5 of them; 4.3e-04 on the data set with s = 0.002). Grid error on 480 random data sets (s from 0.0005): largest probability error 1.6e-04, 0 declaration flips (table in Appendix C). That guarantee fails for small s: AN-01 (34 wrong declarations in 512 stress worlds). Judgement of the standard error from 4-5 readings: AN-06, AN-07. |
| 4 | Wiring and refusals: SI-54 equivalence; refusal of a missing task, lambda, X13 configuration, wrong R4 `b0`, existing output; only the six registered lambda; low-signal cells, decoders and sim1/sim2 never pooled; each sensitivity changes only what section 15.11 names. | partly | On the full-size input 7 of 7 broken inputs were refused with the expected message (missing task, lambda, X13 configuration, scvi_adv rows, plain-scVI seed, wrong `b0`, existing output); the control ran. Extension lambda rows are ignored (0 differences from the baseline record). Low-signal cells (immune and atac_large with decoder setting 1) and sim1/sim2 never entered a pooled result; decoders are pooled separately. SI-54: a mixed-commit input is accepted only when run from a checkout containing the tag (otherwise refused: scorer commits mixed); Sinkhorn rows at the old commit are refused. Exceptions: the R4 record is verified only through `b0` (AN-03, demonstrated by a tamper test); per-run reading does not reach the area (AN-05). |
| 5 | Full-size dry run on synthetic data. | confirmed | Synthetic input of 3,408 rows (3,303 scored, 105 failed) on the real design with realistic noise, three low-signal cells (immune and atac_large with the decoder-1 setting, sim2; atac_large and sim2 have a negative batch range), points whose 5 runs all failed, a discriminator curve crossing b\* three times (pancreas, decoder 1), a failed baseline configuration. The script wrote its record: exit 0, 27.3 s, 201 MB peak memory, 4.06 MB of JSON, 16 cells, 6 primary results, 8 sensitivity analyses. It crashes when a data set has zero standard error (AN-02); it refuses mixed scorer commits unless run from a checkout containing the tag (AN-18). |
| 6 | Tests: suites pass; `check_tests.py` OK (184 passed, 1 skipped, no WARN); the 11 surviving mutants are harmless; at least 8 own planted errors. | partly | `tests/analysis`: 53 passed (36 s); `tests/prereg`: 131 passed, 1 skipped. `check_tests.py` is OK with 184 passed, 1 skipped, no WARN, but only when the test paths are added to the command (as quoted it collects `tests/wcd_vae` and stops with 9 collection errors; AN-18). All 11 of the lead's surviving mutants survive for the reason given: 9 change no number at all in the full-size record, two change numbers by at most 1.5e-09 (Appendix B). Of my 40 planted errors, 30 are caught and 10 are not (AN-04, Appendix A). |
| 7 | The PREREG section 9 X1 entry, the PAPER_PLAN section 5 pointer and its [replaced] markers, and ANALYSIS_SPEC section 13 describe what the code does. | partly | The PREREG section 9 entry (`PREREG.md:310-319`) and the PAPER_PLAN pointer (`PAPER_PLAN.md:164-172`) match the code, apart from three slips (AN-16): 8 markers against "seven statements", a stale Holm threshold of 0.0125 at three places, and a "DRAFT" label. The existing local X1 scores are not disclosed in PREREG (the docs track raised this as DOC-01). |
| 8 | Open-ended: referee objections; ways the code could fail or mislead on the real X1 data; order dependence and digit-for-digit reproducibility. | partly | Referee points: AN-06 to AN-15. Failure modes on real data: AN-01, AN-02, AN-03. Reproducibility: byte-identical on a rerun and with 4 library threads; with shuffled input rows 8,227 of 38,001 numbers differ by at most 1.6e-12; other BLAS CPU kernels differ by at most 3.3e-11; every declaration, count and label is identical in all variants (AN-11). So "digit for digit" holds only for the same row order and CPU kernel. |


## 4. Findings, most important first

Severity words follow the brief: **blocker** = a wrong result or a broken decision; **fix** = must change before the new tag; **note** = worth knowing, optional.

### AN-01 (fix): the pooled model's grid is too coarse when a data set's standard error is small

**Where.** `analysis_x1.py:28-29` (`GRID_N = 201`, `GRID_SPAN = 5.0`), `:160-162` (the grid), `:150` (`pool`); `tests/analysis/test_analysis_x1.py::test_pool_brute_force` (moderate standard errors only).

**What is wrong, in plain words.** The pooled model has two unknown "spreads" (how much families differ, how much data sets differ). The code integrates over them on 201 equally spaced points from 0 to 5 prior scales; at the registered scale 0.1 the spacing is 0.0025. When the data sets' standard errors are smaller than about one such step, the data say "the spreads are tiny", but the grid has almost no points there. The posterior is then distorted and the declaration probabilities are wrong, sometimes enough to change the declaration. With larger standard errors the grid is accurate.

**Evidence.**
- Ordinary standard errors are fine: on 480 random data sets at the four prior scales (0.05, 0.1, 0.2, 0.25) the largest error in any probability is 1.6e-04 and there are 0 wrong declarations (`grid_error_summary.csv`).
- The dry-run local endpoint (decoder 0, six data sets, s = 0.0002 to 0.0014): P(Delta > 0.01) is 0.9756 in the code against 0.9760 converged at the registered scale 0.1 (declaration unchanged). In the `prior_0.2` sensitivity analysis the code gives 0.9489 ("inconclusive") against 0.9757 converged ("above"); the heterogeneity is 0.0083 against 0.0051. The pooled local results of that sensitivity analysis (both decoders) produced 13 of the 27 differences in my end-to-end recomputation (largest 0.027); the other 14 differences are at most 4.0e-04, all in the pooled local endpoint.
- The code's own grid converges as `GRID_N` grows: 0.9489 (201 points), 0.9693 (301), 0.9752 (401), 0.9767 (601), against 0.9757 converged. 601 points cost 2.6 s per call instead of 0.3 s.
- Stress test, 512 simulated worlds (four prior scales x eight standard-error levels, no real spread or a spread of three standard errors): in the 272 worlds with s below one grid step, 34 declarations were wrong (31 where the code said "above" or "equivalent" and the converged answer is "inconclusive"; 3 where it missed a real "above"), with errors up to 0.32. For s between one and three steps the largest error is 0.0084 and there are no wrong declarations; for s of three steps or more it is 9.6e-06. All 34 wrong declarations occurred when the true spread between data sets was essentially zero.
- Exposure on real data (pilot A1, 3 seeds, 6 cells, descriptive): the local endpoint's s ranged from 0.0003 to 0.0063 (median 0.0045), and 2 of 6 cells were below the registered-scale grid step of 0.0025. P1 (s from 0.0032, median 0.0089) and the area (s from 0.0041, median 0.0111) were less exposed. Resampling pilot-like local standard errors at the registered scale: 82 of 210 worlds had an error above 0.001, the largest was 0.131, and 2 declarations flipped.
- Why the calibration check did not show it: when the truth is drawn from the model's own prior the spreads are usually larger than one grid step. Even with standard errors of 0.0002-0.002 the calibration passed (coverage of Delta 0.955 at scale 0.1 and 0.963 at scale 0.2). The failure needs tiny standard errors and a true spread near zero, which is exactly the situation of an "equivalent" or "above" result for two arms that really do not differ.

**Suggested fix.** Replace the uniform trapezoid grid by Gauss-Legendre nodes (160 per spread reproduces the failing case to within 2e-05, with fewer nodes than now), or refine until the probabilities change by less than 1e-4, and add one known-answer test with s of about 0.001. Section 15.4 gives the grid as the agent's implementation proposal (decision SI-73 is "exact grid calculation", which a finer rule keeps), so only the spec sentence changes. My independent implementation (`rv_out/indep_pool.py`) can serve as the reference in that test.

### AN-02 (fix): a data set with zero standard error aborts the whole analysis

**Where.** `analysis_x1.py:152-153` (`pool` requires s > 0); s can be exactly zero from `:146` (area) and `:111` (P1 and local endpoint).

**What is wrong.** Section 15.3 of the spec says the area gives a value in every leave-one-seed-out repeat, "0 when no point lies beyond the corner". If neither arm has a point beyond the corner in any repeat, d = 0 and s = 0. The pooled model then refuses (`PreregError`), after the rest of the analysis has been computed, and no record is written. The same happens for the local endpoint if both arms give identical readings.

**Evidence.** Dry run with a cell in which both arms never pass the corner (lung, decoder 0): exit code 1 after 2.4 s, message "pool needs finite equal-length d and positive s", no output record. Two of my planted errors (M25, M32) were "caught" only because they produced this crash in the test fixture, not through an assertion.

**Suggested fix.** Decide in the specification now: (a) a data set with s = 0 is "not evaluable: zero spread" in the pooled results, with the reason recorded; or (b) floor s at a stated minimum (note the link to AN-01). Add one test for the chosen rule. Collapse is an outcome the pre-registration defines (rule R1: a failed fit or a biology loss of 0.10 or more), so a cell in which every point collapses should be reportable, not fatal.

### AN-03 (fix): only one of the four numbers in the R4 record is checked against the data

**Where.** `analysis_x1.py:362-366` (checks that four keys exist and that `b0` equals the mean batch score of the plain-scVI runs); `:388-393` (uses `b_common`, `bstar` and `bstar_low_signal` as given). Spec section 15.1 describes the check as detecting "a record made from other scores".

**What is wrong.** `b0` depends only on the plain-scVI runs. A record computed before every arm's X1 rows were scored has the same `b0` but a different `b_common`, so a different matched target `b*`, a different area divisor, and possibly a different low-signal flag. The analysis would not notice.

**Evidence (tamper test on the full-size dry run).** I changed `bstar` of atac_large|0 by +0.03 (from 0.4694 to 0.4994), `b_common` of pancreas|1 to 0.99 (from 0.8025), and the low-signal flag of lung|0 to true. The script ran to completion (exit 0, 21.9 s) and used the altered values: the lung data set dropped out of the pooled results, the pooled P1 mean for decoder 0 moved from 0.00721 to 0.00560, and the pancreas|1 area difference moved from 0.0313 to 0.0112. The record already holds the per-arm maxima (`M`) from which `b_common` is derived.

**Suggested fix.** Recompute all four numbers from the same scores with the function that wrote the record (`prereg_rules.r4_stage`) and refuse on any difference above 1e-9, or store the SHA-256 of the score files in the R4 record and compare it. One test: a record with a changed `bstar` must be refused.

### AN-04 (fix): the tests do not notice 10 of 40 planted errors

**Where.** `tests/analysis/test_analysis_x1.py`; the lines concerned are in Appendix A.

**What is wrong.** I planted 40 one-token errors where they would change a declaration or a table; the tests caught 30. Not caught: lowering the declaration threshold from 0.95 to 0.90 (M03; raising it to 0.99 is caught), keeping the dropped seed in the plain-scVI anchor of a leave-one-seed-out repeat (M12), not excluding extension-lambda rows (M13), swapping B and C in the frontier counts (M14), computing every arm's window from the reference arm (M15), changing the sign-table multiplier 1.96 to 1.0 (M16), the drop-failed sensitivity not reaching the area (M18), the pooled local endpoint reading P1 values (M22), halving the grid to 101 points (M24; 51 points is caught), and replacing the per-dataset standard error by the P1 one in the pooled area and local results (M33). Most survivors sit in the assembly of results in `analyse` (M13, M14, M15, M18, M22, M33); the rest are constants (M03, M16, M24) and the P1 anchor (M12). The lead's own random mutation runs caught 35 and 34 of 40.

**Suggested fix (extend existing tests, per the fewest-tests rule).** (i) A probability between 0.90 and 0.95 in `test_pool_closed_form`; (ii) plain-scVI numbers that differ by seed in one `test_p1_reading` case; (iii) one extension-lambda row in the end-to-end fixture; (iv) unequal B and C in the frontier check; (v) two arms with different windows; (vi) assert the sign-table limits; (vii) compare the area under `drop_failed`; (viii) recompute the pooled local and area results independently in the end-to-end test; (ix) add the grid test of AN-01.

### AN-05 (fix): the per-run reading sensitivity does not reach the area, although section 15.11 says it does

**Where.** `analysis_x1.py:392-393` (the area receives only `drop_failed=(mode == "drop_failed")`); `ANALYSIS_SPEC.md` section 15.11.

**What is wrong.** Section 15.11 lists "per-run reading and dropping points with a failed run (P1, area, local-global)". Under `per_run` the area record is identical to the primary one; only P1 and the local endpoint change. The area has no per-run definition in section 3 (it is read from 5-run point means), so the sentence is probably what should change.

**Evidence.** `sens_diff.py` on the dry-run record: per_run changes the P1 and local cell readings, the pooled P1 and local results and the Holm table, and no area value; drop_failed changes P1, local and area; corner_zero changes only the area; no_trajectory changes everything except the local endpoint and `b*`; pairs and independent change only the pooled results (P1, area, local, P2); the prior sensitivities change pooled P1, area and local and leave P2 at its own scale 0.25 (stated in section 15.6).

**Suggested fix.** Amend section 15.11 to "per-run reading (P1, local-global)" or define and implement a per-run area.

### AN-06 (note): per-data-set intervals of d plus or minus 1.96 s under-cover

**Where.** `analysis_x1.py:440-441` (the sign table; section 15.4 asks for exactly this).

**What is wrong.** s comes from 4 or 5 readings, yet the interval uses the normal multiplier 1.96. In my simulations (truth known) these intervals covered the true per-dataset effect 0.878 to 0.883 of the time over five scenarios (0.853 with a non-monotone curve), which is what a t distribution with 4 degrees of freedom gives (0.878; 0.855 with 3), not 0.95. The multiplier would be 2.776 for 5 readings and 3.182 for 4. Planted error M16 (1.96 to 1.0) was not caught.

**Suggested fix.** Label the interval "plus or minus 1.96 standard errors (normal approximation)" or use t with m-1 degrees of freedom; pin the constant in a test.

### AN-07 (note): standard errors from 4-5 readings are treated as known; equivalence is out of reach at pilot noise

**What a statistical referee would say.** The pooled model takes each s as known although it is a jackknife estimate from 4-5 readings of a non-smooth statistic (first crossing, linear interpolation). Random-effects meta-analysis guidance says treating within-study variances as known is not tenable when studies are small (Partlett & Riley 2017, abstract read), and delete-one jackknife variance estimators can be inconsistent for non-smooth statistics (Shao & Wu 1989, abstract read). Half-normal priors for spreads with few groups are in the family Gelman (2006, abstract read) recommends, and the 0.05 and 0.2 sensitivity analyses are the right response to weak identification of the spreads.

**Evidence (simulations with the frozen code's own reading and pooling functions; truth known).** The jackknife s was unbiased on the variance scale for smooth curves (median over data-set cells of rms(s) divided by the true standard deviation: 1.003 to 1.026 across five scenarios, 10-90% range 0.93 to 1.16) and conservative for a non-monotone curve (1.115). The pooled 95% interval for Delta covered at least 0.963 in every null and effect scenario (conservative). Under the model's own prior (simulation-based calibration) coverage of Delta was 0.940 at scale 0.1 (N = 700, uniformity test p = 0.77), 0.950 at 0.25 (p = 0.89) and 0.947 at 0.2 (p = 0.42). At scale 0.05 one of three runs gave p = 0.011; two larger repeats gave 0.73 and 0.33, so I read it as chance. Per-dataset effect intervals covered only 0.88 to 0.93. In none of 5,040 simulated pooled worlds (leave-one-seed-out noise 0.01 to 0.02) did the model declare "equivalent".

**Suggested handling.** No code change. State in the Methods that s is a jackknife estimate treated as known, that statements about single data sets are less certain than the pooled one, and that equivalence cannot be declared at this noise level (section 5 of the spec already says so for the planned power). It becomes reachable only for s of about 0.003 or less, where AN-01 applies.

### AN-08 (note): P1 depends on the registered lambda spacing

**Evidence.** Pilot A1 (3 seeds, 6 cells, Sinkhorn dropped): in 1 of 6 cells (atac_small|1) the reference reading changed from 0.6025 (all 10 pilot strengths) to 0.6136 (only the registered six-point W1 grid), moving d by +0.0110, more than the 0.01 margin; the other 5 cells were identical. In the curved-response simulation the bias of d against the smooth truth was +0.0022 on average and at most 0.0083. The W1 grids have one-decade gaps; the JS grid has half-decade steps.
**Suggested handling.** Say in the Methods that P1 is the difference of two six-point piecewise-linear curves read at b\* and depends on the registered grid. No code change.

### AN-09 (note, question): can b\* depend on extension points?

**Evidence.** The analysis keeps only the six registered lambda per family (`analysis_x1.py:60-66`), as PREREG.md:99 says ("R4 extension points are used for matching only"). PREREG.md:92 says the R4 stage-x1 fits are "the family grid plus any R4 extension", so an extension point can change `b_common` and therefore `b*`. In the dry run, recomputing the R4 record with 240 synthetic extension rows moved `b*` in 2 of 16 cells (immune_hum_mou, decoder 1: 0.65438 to 0.66363, d by +0.0023; lung, decoder 1: d by +0.0002); with extension rows in the scores but the record built from the six points, the record is unchanged (0 differences). The line that excludes extension rows is untested (M13).
**Question for the lead.** Is it intended that matching points shift the target at which the six-point curves are read? If yes, say so in section 15.1; if no, build the R4 record from the six registered points.

### AN-10 (note): the seeds are inferred, not asserted

**Evidence.** `analysis_x1.py:374-384` takes the seeds from the plain-scVI rows (`sorted(zero)`) and applies them to every arm; nothing asserts that they are 0-4. A missing plain-scVI seed is caught only indirectly through the `b0` check (refusal "R4 b0 differs from lambda=0", case `missing_zero_seed4_lung_c0`).
**Suggested fix.** Assert the seed set equals {0, 1, 2, 3, 4} (one line).

### AN-11 (note): "digit for digit" holds only for the same row order and CPU kernel

**Evidence (dry-run record, 38,001 numbers).** Rerun and 4 library threads: byte-identical. Shuffled rows of manifest, scores and failures: 8,227 numbers differ by at most 1.6e-12. Other OpenBLAS kernels: NEHALEM 1,059 numbers, at most 6.5e-12; SANDYBRIDGE 1,162 numbers, at most 3.3e-11; HASWELL (this machine's default) byte-identical. Every declaration, count and label is identical in all variants. (The input scores themselves depend on CPU instruction width, a separate known issue.)
**Suggested fix.** Sort the rows by tag in `main` (one line) so that row order cannot matter, and write "identical to within 3e-11 across CPU types; identical declarations" instead of "digit for digit".

### AN-12 (note): P2 intervals treat 30 runs as independent

**Evidence.** The 30 runs of a P2 cell are 6 penalty strengths x 5 seeds; runs with the same seed share initialisation (PREREG CR-07). Synthetic check with the frozen interval functions (20,000 replicates): if collapse rises with lambda but runs are independent, the 95% Clopper-Pearson interval for the grid-average collapse rate covered 1.000 (conservative); if collapse is driven by a per-seed threshold it covered 0.914. The Agresti-Caffo difference covered 0.949 when both arms share the seed effect and 0.919 when they do not.
**Suggested handling.** State the independence assumption; the departure is mild.

### AN-13 (note): best-of-k ignores failed runs and compares seed-averaged arms with single-run baselines

**Evidence.** `analysis_x1.py:405-415`: a configuration's score is the mean over its successful runs, so one surviving run weighs as much as five; a configuration with no successful run is dropped and its family's N shrinks (stated in section 15.9, the partial-failure case is not). Failures are therefore counted only in P2. Scores of scVI-based arms are 5-seed means chosen as the best of 6 noisy values, while Harmony, Scanorama, sysVI, scANVI and PCA configurations are single deterministic runs; selection on noisy means inflates an arm's best-of-k by at most about 1.27 standard errors of a 5-seed mean (arithmetic for six equal configurations). Section 9 labels best-of-k optimistic. The expected-best-of-k statistic follows Dodge et al. 2019 (arXiv abstract read); the negative bias reported by Tang et al. 2020 concerns an estimator of an i.i.d. expectation, whereas here E_k is the exact mean over k-subsets of the six observed configurations.
**Suggested handling.** Show the number of failed runs next to each configuration in the best-of-k table, and say that failures are not penalised in this table.

### AN-14 (note): the lambda window can span gaps

**Evidence.** `analysis_x1.py:329-332`: the window is every point whose mean is at least the lower bound, and the width is log10(largest) - log10(smallest), so a point in between that falls below the bound still counts as inside. This follows spec section 10. Pilot A1 (10-point grid, 3 seeds): 5 of 30 windows were non-contiguous; for example immune, decoder 1, mmd: points [0.1, 3.0, 10.0, 30.0], width 2.48 decades although the points at 0.3 and 1 lie outside.
**Suggested handling.** Also report the contiguous run that contains the best point, or flag windows with gaps.

### AN-15 (note): the "pairs" sensitivity assumes independent errors between data sets that share cells

**Evidence.** `analysis_x1.py:222-226` averages the data sets of a family and takes SE = sqrt(sum of s squared)/k. If the errors of two data sets with equal s are perfectly correlated this understates the standard error by 29%. The project memory records that immune is contained in immune_hum_mou and that atac_small shares 3,618 barcodes with atac_large (not re-verified by me).
**Suggested handling.** State the independence assumption, or add a correlation term and show the effect.

### AN-16 (note): three document slips (analysis-related)

(a) `PAPER_PLAN.md:167` says seven statements are replaced, but there are 8 `[replaced:` markers (`:178` twice, `:181`, `:182`, `:184`, `:187`, `:190`, `:191`); ROPE and measurement SD are two lines against one entry in `ANALYSIS_SPEC.md` section 13. (b) Holm's first threshold "0.0125 = 0.05/4" is stale at `PREREG.md:106`, `PREREG.md:251` and `PAPER_PLAN.md:184`; the code applies Holm over six primary results (first threshold 0.05/6 = 0.0083, `analysis_x1.py:478-479`). The conclusion "cannot reject" is unchanged, because the smallest two-sided exact p with 6 data sets is 2/2^6 = 0.031 (0.0625 with 5). (c) `ANALYSIS_SPEC.md:1` and `:3` still say "draft" and "Status: DRAFT".

### AN-17 (note): the local-versus-global declarations sit outside the six primary results

**Evidence.** `analysis_x1.py:444-446` adds a sign test and a pooled declaration for the local endpoint (one pair per decoder); they are not in the Holm family (`:478`) and `n_primary` stays 6. Section 5 of the spec says the report states how many primary results were judged; it does not say how many local-global ones. Because B = (3 local + 2 global)/5, at matched B the local difference is minus two thirds of the global one (project memory; also a known-answer test), so this is one signed question, not two independent ones.
**Suggested handling.** One sentence in the Methods.

### AN-18 (note): operating procedure

(a) `scripts/check_tests.py --base bba0fd0 --src scripts` as quoted collects `tests/wcd_vae` and stops with 9 collection errors; with `-- tests/analysis tests/prereg` appended it is OK (184 passed, 1 skipped, 125 s). (b) SI-54: an input mixing the old commit (non-Sinkhorn rows) and the tag is accepted only when the analysis is run from a checkout that contains the tag `prereg-tier12-v3` (otherwise refused, "mix scorer provenance", exit 1); Sinkhorn rows at the old commit are refused (exit 1). Say so in the run instructions. (c) PREREG.md:316 says the script "runs once"; the code only refuses to overwrite an existing `--out-json` (`analysis_x1.py:494`).


## 5. Prior-work check (scoped)

The X1 analysis claims no new method; it implements pre-registered rules, so no novelty finding is raised. The components were still searched, as the review protocol asks.

- **Scope.** Quick tier, 2026-10-14: 5 keyword queries (hierarchical Bayesian comparison of algorithms across data sets with a region of practical equivalence; single-cell batch-correction benchmarks with biology conservation at matched batch mixing; expected best of k configurations; jackknife standard errors from seeds treated as known variances; Wasserstein critic versus discriminator in single-cell integration) over OpenAlex, arXiv and Europe PMC (PubMed, bioRxiv, medRxiv). 174 records were retrieved (61 OpenAlex, 60 Europe PMC, 53 arXiv); the top 30 were screened by an automated overlap triage, 3 were rated "partial" and none "substantial", and I read the abstracts of those three. Full texts were not fetched (no escalation). The record is `rv_out/prior_work_search_AN.json`.
- **Verdict: partial overlap by component; no match in scope for the specific combination** (reading biology at a matched batch-removal target, leave-one-seed-out standard errors, a family and data-set hierarchical model, for single-cell integration). Components that exist: ROPE-based Bayesian comparison across data sets (Corani et al. 2017, Machine Learning 106:1817-1837, doi 10.1007/s10994-017-5641-9, abstract read, already cited by the spec; and the Bayesian Bradley-Terry model, arXiv 2208.04935, 2022, abstract read, an additional citation candidate); the expected best of k (Dodge et al. 2019, arXiv 1909.03004, abstract read); the add-one-success-and-one-failure interval for a difference of proportions (Agresti & Caffo 2000, American Statistician 54:280-288, doi 10.2307/2685779, metadata only). The third "partial" record is the project's own earlier preprint (doi 10.1101/2025.11.12.688061), not independent prior work. This is a title-and-abstract screen of 30 of 174 records and does not prove absence.
- **Sources behind the recommendations** (all retrieved in this session; none read in full text): Partlett & Riley 2017 (Statistics in Medicine, doi 10.1002/sim.7140, abstract) on known within-study variances in small studies; Shao & Wu 1989 (Annals of Statistics 17:1176-1197, doi 10.1214/aos/1176347263, abstract) on the jackknife for non-smooth statistics; Gelman 2006 (Bayesian Analysis 1:515-534, doi 10.1214/06-BA117A, abstract) on priors for hierarchical standard deviations; Tang et al. 2020 (arXiv 2004.13705, abstract) on bias of best-of-k estimators.

## 6. Where the lead's own observations stand

The lead recorded five observations before the review. I read them after finishing my findings.

| Lead's observation | My result |
|---|---|
| (1) PREREG section 9 entry says the smallest Wilcoxon p, 0.031, exceeds 0.0125 (Holm over 4); the code uses Holm over 6. | Confirmed and located: `PREREG.md:106`, `PREREG.md:251`, `PAPER_PLAN.md:184` (AN-16 b). Conclusion unchanged. |
| (2) 19 local X1 score files exist from 2026-10-12; one was rescored in the identity check; does PREREG disclose it? | Not disclosed: `PREREG.md:315-316` says "before any X1 adversarial-arm score is examined"; a search of PREREG, PAPER_PLAN, CONSTRAINTS and ANALYSIS_SPEC finds no mention, only the notebook (`notebook/2026-10-12.md:46`). The docs track raised it as DOC-01; I did not open those files (blinding). |
| (3) Best-of-k scores use successful runs only. | Confirmed, with the arm-versus-baseline noise asymmetry (AN-13). |
| (4) The window width spans gaps. | Confirmed: 5 of 30 pilot windows are non-contiguous (AN-14). |
| (5) The "pairs" standard error assumes independent errors for data sets that share cells. | Confirmed (AN-15). |

I found five items the lead did not list: AN-01, AN-02, AN-03, AN-04, AN-05.

## 7. Limits of this review

- I worked in my own clone of the repository at the reviewed commit, because the git metadata of the provided worktree is masked by the sandbox; the clone's `analysis_x1.py` and test file are byte-identical to the worktree's (SHA-256 prefixes c39541cb and b554b2cb). Git failures in the worktree are not findings.
- No real X1 or X13 score was used. All numerical checks used synthetic inputs (one full-size dry-run world and its variants) or the pilot A1 data (seeds 100-102, Sinkhorn rows dropped, dense 10-point grid and 3 seeds instead of the registered 6 points and 5 seeds), so exposure estimates from the pilot are indicative only.
- The simulations are synthetic and provisional: 5,040 pooled worlds in total, 120 data-set cells per scenario, 480 data sets for the grid survey, 400 to 800 data sets per calibration run, 512 stress worlds. My independent reference (Gauss-Legendre quadrature) agrees with the code to 3e-07 on 5 of 6 test data sets, which supports it but is not proof. A bug in my own reference quantile routine produced one spurious large disagreement during the review; I fixed it (root-finding instead of interpolation) and re-ran every comparison, so all numbers here are post-fix.
- Mutation testing used 40 chosen errors, not all possible ones. The spot-check of the lead's 11 survivors combined the test suite with the full-size dry-run record.
- The dry run is one synthetic world per variant; the curve with three crossings of b\* replaces the "twice" in the brief.
- The user's `ponytail-review` skill is not in this profile's catalogue and was not run; `fl_lint` does not exist in the repository and was not run. `ruff` (rules F and E9) was clean on the two files.
- Literature was checked at abstract or metadata level only. The prior-work check stopped at the quick tier.
- The shared-barcode facts in AN-15 are taken from the project memory and were not re-verified. Whether the R4 stage-x1 record is built with extension points (AN-09) is a question, not a determination.
- No project-wide rigor ledger file exists yet in this project. Following the docs track, I saved my entries as `rigor_ledger_entries_AN.json` instead of creating the ledger, to avoid concurrent writes by the three reviewers.
- The process cap of 4 parallel processes was respected.

## 8. How to reproduce

All scripts are in the tar.gz (`rigor_analysis_review_bundle.tar.gz`, folder `rv_out/`); run from the workspace root with the `scvi-api` conda environment (Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, pytest 9.1.1), `KMP_AFFINITY=disabled OMP_NUM_THREADS=1 WCD_SRC=src`, on a clone made with `git clone <repo> rv; cd rv; git checkout --detach 1e615eb321097729c0fb4f117bc4e9fd1e0ac9a8`.

| Check | Command (from the workspace root) | Output |
|---|---|---|
| tests | `pytest -q tests/analysis tests/prereg` inside the clone | logs/baseline_tests_*.log |
| coverage gate | `python scripts/check_tests.py --base bba0fd0 --src scripts -- tests/analysis tests/prereg` inside the clone | logs/check_tests_analysis_prereg.log |
| known answers | `python rv_out/rederive_tests.py` | rederive_tests.json |
| independent end-to-end record | `python rv_out/reference_pipeline.py rv_out/dry_main rv_out/dry_main/analysis.json primary no_trajectory pairs independent prior_0.05 prior_0.2` (and `per_run drop_failed corner_zero`) | reference_pipeline_*.json |
| pooled model | `python rv_out/check_pool_basic.py`; `python rv_out/grid_error.py --scale <s> --n 120 --out rv_out/grid_error_<s>.json` (then `summarise_grid_error.py`); `SBC_SET=main python rv_out/sbc_pool.py` (sets: main, rep005, rep005_bigs, rep02, tiny) | check_pool_basic.json, grid_error_*.json, sbc_pool_*.json |
| grid weakness (AN-01) | `python rv_out/check_prior02_local.py`; `python rv_out/check_code_gridn.py`; `python rv_out/grid_scan_tiny.py`; `python rv_out/grid_pilot_like.py`; `python rv_out/pilot_s_values.py` | the matching .json and .csv files |
| calibration simulations | `python rv_out/sim_pipeline.py --scenario <name> ...` (scenarios S0-S4, W1, P0-P3; the arguments of each run are stored in the `args` field of its output) | sim_*.json, simP_*.json, logs/ |
| dry run and variants | `python rv_out/dryrun_build.py --out rv_out/dry_<variant> --variant <variant>`; `python rv_out/dryrun_run.py --dir rv_out/dry_<variant> --tag <variant>` | dry_*/ |
| sensitivity wiring | `python rv_out/sens_diff.py rv_out/dry_main/analysis.json rv_out/sens_diff_main.json` | sens_diff_main.json |
| refusals; tamper; reproducibility | `python rv_out/refusal_battery.py`; `python rv_out/repro_compare.py` | refusal_battery.json, repro_compare.json |
| planted errors; lead's survivors | `python rv_out/run_mutants.py rv_mut A <ids>`; `python rv_out/spotcheck_lead.py rv_mut <ids>` | mutation_results_all.csv, spotcheck_lead_*.json |
| every number in this report | `python rv_out/build_facts.py` | facts.json |


## Appendix A. The 40 planted errors

Each row is one change I made to a private copy of `analysis_x1.py`, then ran `tests/analysis` (first failure = caught). Script: `rv_out/mutants.py`, `rv_out/run_mutants.py`; results: `rv_out/mutation_results_all.csv`.

| ID | What I changed (one token or one line) | Result |
|---|---|---|
| M01 | sign of d reversed (discriminator minus reference) | caught by `test_p1_reading` |
| M02 | margin 0.01 -> 0.02 (P1, area, local) | caught by `test_end_to_end` |
| M02b | P2 margin 0.05 -> 0.5 | caught by `test_end_to_end` |
| M03 | declaration threshold 0.95 -> 0.90 | not caught |
| M03b | declaration threshold 0.95 -> 0.99 | caught by `test_end_to_end` |
| M04 | P1 prior scale 0.1 -> 0.2 | caught by `test_end_to_end` |
| M04b | P2 prior scale 0.25 -> 0.5 | caught by `test_end_to_end` |
| M05 | family map: every dataset its own family (family term dropped) | caught by `test_combine_structures` |
| M06 | transfer lambda trained leave-one-DATASET-out instead of leave-one-family-out | caught by `test_transfer_lambda` |
| M07 | window: t degrees of freedom n-1 -> n | caught by `test_window` |
| M07b | window: normal quantile instead of t | caught by `test_window` |
| M08 | Holm order: largest p first | caught by `test_sign_test_and_holm` |
| M09 | area corner offset 0.10 -> 0.20 below scVI biology | caught by `test_read_area` |
| M10 | collapse boundary 0.10 -> 0.15 | caught by `test_collapse_counts` |
| M11 | jackknife factor (m-1)/m -> 1/m (read_p1) | caught by `test_p1_reading` |
| M12 | leave-one-seed-out anchor does not drop the seed from plain scVI (P1) | not caught |
| M13 | R4 extension rows not excluded from the arm's points | not caught |
| M14 | frontier positions: (B, C) swapped | not caught |
| M15 | window width of every arm computed from the reference's scores | not caught |
| M16 | sign-table z value 1.96 -> 1.0 | not caught |
| M17 | Holm family contains local instead of area | caught by `test_end_to_end` |
| M18 | drop_failed sensitivity does not reach the area | not caught |
| M19 | pooled P2 rate sums all eight tasks (sim1, sim2 pooled with real tasks) | caught by `test_end_to_end` |
| M20 | Agresti-Caffo adds 0.5/1 instead of 1/2 | caught by `test_collapse_counts` |
| M21 | area divisor: b* - b0 instead of b_common - b0 | caught by `test_end_to_end` |
| M22 | local endpoint pooled from the P1 reading | not caught |
| M23 | GRID_SPAN 5 -> 3 (spread grid cut at 3 prior scales) | caught by `test_pool_brute_force` |
| M24 | GRID_N 201 -> 101 | not caught |
| M24b | GRID_N 201 -> 51 | caught by `test_pool_brute_force` |
| M25 | P1 contrast arm: pooled instead of reference | caught only by the fixture crash (AN-02) |
| M26 | prior scale of the spreads fixed at 0.1 in the sensitivity analyses | caught by `test_combine_structures` |
| M27 | family spread prior half-normal -> weight exp(-tau/scale) (wrong prior density) | caught by `test_pool_brute_force` |
| M28 | tau grid starts at 0 but trapezoid end weights dropped (rectangle rule) | caught by `test_pool_brute_force` |
| M29 | marginal covariance: dataset spread omitted (family term only) | caught by `test_combine_structures` |
| M30 | posterior probability of 'equivalent' uses 2*margin | caught by `test_pool_closed_form` |
| M31 | R4 target: b_common used as b* in P1 | caught by `test_end_to_end` |
| M32 | low-signal flag ignored (cells pooled) | caught only by the fixture crash (AN-02) |
| M33 | per-dataset s of the area replaced by the P1 s | not caught |
| M34 | failed runs counted as plain scVI-like non-collapse (failures not counted in P2) | caught by `test_collapse_counts` |
| M35 | best-of-k indexing wrong (comb(i, k-1)) | caught by `test_best_of_k` |

## Appendix B. The lead's 11 surviving mutants, spot-checked

Source of the list: `mutation_runs_analysis_x1.json` (runs fix1_seed0 and fix2_seed1_tests_858101b). For each I made the same change in a private copy, ran `tests/analysis` (all still pass, so they survive) and re-ran the full-size dry run to compare every number and label of the record with the unmutated record (the only label that differs is the `git_dirty` flag of the modified copy).

| Location | Change | Tests | Full-size dry-run record | Why it is harmless |
|---|---|---|---|---|
| analysis_x1.py:45 | skipna=False to True in the global-score mean | SURVIVED | no number changes | Equivalent on valid input: `prereg_rules.outcomes` already refuses non-finite metrics in ok rows (`prereg_rules.py:330-331`). |
| analysis_x1.py:48 | same, for the trajectory-free biology score | SURVIVED | no number changes | Same reason. |
| analysis_x1.py:82 | fourth field of the plain-scVI anchor point 0 to 1 | SURVIVED | no number changes | Equivalent: the field is never read (`analysis_x1.py:97`, `prereg_rules.py:615-622`). |
| analysis_x1.py:98 | `< 0` to `<= 0` in the crossing count | SURVIVED | no number changes | Equivalent: the product of two distances that are both larger than 1e-9 cannot be 0. |
| analysis_x1.py:117 | `>` to `>=` in the area filter | SURVIVED | no number changes | Equivalent: a rectangle of zero width or zero height adds no area. |
| analysis_x1.py:162 | upper-end half weight of the grid (`GRID_N - 1` to `GRID_N + 1`) | SURVIVED | 1578 numbers change by at most 1.5e-09 | Negligible: it reweights nodes at five prior scales. |
| analysis_x1.py:185 | root-finding bracket of 8 to 9 standard deviations | SURVIVED | 850 numbers change by at most 3.1e-13 | Negligible: the root is inside either bracket. |
| analysis_x1.py:270 | `len(rows) > 0` to `> 1` in `collapse_counts` | SURVIVED | no number changes | Guard only: a one-run input cannot occur (30 runs per cell). |
| analysis_x1.py:284 | `and` to `or` in the input check of `choose_lambda` | SURVIVED | no number changes | Guard only: a weaker refusal; invalid input would fail later with another error. |
| analysis_x1.py:322 | `>=` to `>` in the window tie rule | SURVIVED | no number changes | Equivalent except when a mean equals the bound to within 1e-9 exactly. |
| analysis_x1.py:491 | `--r4-record` made optional | SURVIVED | no number changes | Guard only: a command-line argument check. |

## Appendix C. Grid error

**C1. Random data sets** (`grid_error.py`; standard errors from 0.0005 to 0.06, and from 0.005 to 0.15 at scale 0.25; the code against a converged Gauss-Legendre calculation):

| Prior scale | Random data sets | Wrong declarations | Largest error: mean | interval end | any probability | heterogeneity |
|---|---|---|---|---|---|---|
| 0.05 | 120 | 0 | 5.8e-06 | 1.8e-05 | 2.1e-05 | 1.1e-05 |
| 0.1 | 120 | 0 | 1.2e-05 | 4.5e-05 | 2.0e-05 | 2.9e-05 |
| 0.2 | 120 | 0 | 8.5e-05 | 9.8e-05 | 1.6e-04 | 9.1e-05 |
| 0.25 | 120 | 0 | 5.1e-06 | 1.4e-05 | 7.1e-06 | 1.9e-05 |

**C2. Stress test with small standard errors** (`grid_scan_tiny.py`; six data sets in the registered family structure, standard errors of the stated size, true spread between data sets zero or three standard errors, true effect placed so that the probabilities span 0.5 to 0.999; 16 worlds per row):

| Prior scale | Standard error s of every data set (about) | Grid step h | s / h | Worlds | Largest probability error | Worlds with error above 0.001 | Wrong declarations |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.0002 | 0.00125 | 0.16 | 16 | 2.4e-01 | 16 | 8 |
| 0.05 | 0.0005 | 0.00125 | 0.40 | 16 | 1.1e-01 | 12 | 1 |
| 0.05 | 0.001 | 0.00125 | 0.80 | 16 | 5.9e-02 | 7 | 1 |
| 0.05 | 0.002 | 0.00125 | 1.60 | 16 | 4.7e-04 | 0 | 0 |
| 0.05 | 0.004 | 0.00125 | 3.20 | 16 | 4.2e-06 | 0 | 0 |
| 0.05 | 0.008 | 0.00125 | 6.40 | 16 | 2.3e-09 | 0 | 0 |
| 0.05 | 0.016 | 0.00125 | 12.80 | 16 | 9.8e-09 | 0 | 0 |
| 0.05 | 0.032 | 0.00125 | 25.60 | 16 | 9.6e-06 | 0 | 0 |
| 0.1 | 0.0002 | 0.00250 | 0.08 | 16 | 3.2e-01 | 16 | 2 |
| 0.1 | 0.0005 | 0.00250 | 0.20 | 16 | 2.3e-01 | 15 | 0 |
| 0.1 | 0.001 | 0.00250 | 0.40 | 16 | 7.9e-02 | 16 | 0 |
| 0.1 | 0.002 | 0.00250 | 0.80 | 16 | 2.4e-02 | 9 | 0 |
| 0.1 | 0.004 | 0.00250 | 1.60 | 16 | 1.6e-03 | 1 | 0 |
| 0.1 | 0.008 | 0.00250 | 3.20 | 16 | 3.5e-07 | 0 | 0 |
| 0.1 | 0.016 | 0.00250 | 6.40 | 16 | 2.5e-09 | 0 | 0 |
| 0.1 | 0.032 | 0.00250 | 12.80 | 16 | 1.3e-08 | 0 | 0 |
| 0.2 | 0.0002 | 0.00500 | 0.04 | 16 | 3.0e-01 | 16 | 3 |
| 0.2 | 0.0005 | 0.00500 | 0.10 | 16 | 2.7e-01 | 16 | 3 |
| 0.2 | 0.001 | 0.00500 | 0.20 | 16 | 1.8e-01 | 14 | 3 |
| 0.2 | 0.002 | 0.00500 | 0.40 | 16 | 1.5e-01 | 15 | 2 |
| 0.2 | 0.004 | 0.00500 | 0.80 | 16 | 5.7e-02 | 6 | 0 |
| 0.2 | 0.008 | 0.00500 | 1.60 | 16 | 1.4e-03 | 2 | 0 |
| 0.2 | 0.016 | 0.00500 | 3.20 | 16 | 1.5e-06 | 0 | 0 |
| 0.2 | 0.032 | 0.00500 | 6.40 | 16 | 1.0e-09 | 0 | 0 |
| 0.25 | 0.0002 | 0.00625 | 0.03 | 16 | 3.2e-01 | 16 | 3 |
| 0.25 | 0.0005 | 0.00625 | 0.08 | 16 | 2.6e-01 | 16 | 4 |
| 0.25 | 0.001 | 0.00625 | 0.16 | 16 | 2.5e-01 | 16 | 4 |
| 0.25 | 0.002 | 0.00625 | 0.32 | 16 | 1.9e-01 | 14 | 0 |
| 0.25 | 0.004 | 0.00625 | 0.64 | 16 | 5.1e-02 | 10 | 0 |
| 0.25 | 0.008 | 0.00625 | 1.28 | 16 | 8.4e-03 | 2 | 0 |
| 0.25 | 0.016 | 0.00625 | 2.56 | 16 | 3.7e-05 | 0 | 0 |
| 0.25 | 0.032 | 0.00625 | 5.12 | 16 | 5.6e-09 | 0 | 0 |
