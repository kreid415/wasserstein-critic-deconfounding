# Numerical checks, known answers and planted errors for scripts/analysis_x1.py

Written 2026-10-14 for finding AN-01 of the independent rigour review (the spread grid of the pooled model was too coarse) and
extended 2026-10-15 for the findings AN-R08, AN-R10, AN-R11 and AN-R12 of the independent re-check of the fixes (AN-Rnn is analysis
finding nn of that re-check; DOC-R04 for `first_review/`). The user's decisions SI-83 (a standard error below 1e-6 is raised to 1e-6)
and SI-84 (the accuracy ranges are stated) are in ANALYSIS_SPEC.md sections 4 and 15.4.

Why. The spread grid of the pooled model (201 points evenly spaced from 0 to 5 prior scales) was too coarse when a dataset's standard
error s is below the grid step, and could change a declaration. The amended code spaces the 201 points evenly in log(tau) from 1e-15
to 5 prior scales and takes the posterior of the overall difference Delta from solves with A, the covariance of the dataset
differences without Delta's prior variance (the subtraction it replaced lost precision when s is small). A dataset whose standard
error is below 1e-6 is kept, with its standard error raised to 1e-6 and listed in the record (SI-83); the earlier rule that left such
a dataset out was removed.

Inputs are synthetic, except `first_review/pilot_grid_reading.py`, which reads the pilot A1 files. No X1 or X13 score is read.

## Files and how to run them

Environments: conda env `python` (Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, mpmath 1.4.1) for the numerical scripts;
conda env `scvi-api` (Python 3.11, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, pytest 9.1.1) for the tests, the planted-error run and
`first_review/`. One thread each (`OMP_NUM_THREADS=1`); run times are for one core. Commands are from the repo root.

| File | What it does | Command | Run time | Output |
|---|---|---|---|---|
| `check_pool_numerics.py` | the review's failing case; 240 random sets against a finer, wider reference rule; relative errors of the posterior mean and variance at grid nodes in 40-digit arithmetic; the effect of M24 and M28 | `python experiments/analysis_numerics_20261014/check_pool_numerics.py` | 4 min | `review_case.csv`, `stress_240.csv`, `precision_mpmath.csv`, `summary.json` |
| `ref_pool.py`, `ref_mp.py` | the independent reference (modules; result 4 below) | imported by the scripts below | | |
| `validate_algebra.py` | the reference's block algebra against direct n x n 50-digit algebra at 60 random nodes | `python experiments/analysis_numerics_20261014/validate_algebra.py > experiments/analysis_numerics_20261014/validate_algebra.json` | 1 s | `validate_algebra.json` |
| `compare_with_reference.py` | the code against the independent reference: 18 hand-made sets and a spread scan | `python experiments/analysis_numerics_20261014/compare_with_reference.py` | 13 s | `compare_with_reference.json` |
| `known_answers_small_s.py` | five synthetic sets with standard errors of 1e-6 to 3e-6 and their 50-digit values (read by `tests/analysis`) | `python experiments/analysis_numerics_20261014/known_answers_small_s.py` | 3 min | `known_answers_small_s.csv` |
| `known_answers_effects.py` | the distance from those known answers of the code and of three planted grid errors | `python experiments/analysis_numerics_20261014/known_answers_effects.py > experiments/analysis_numerics_20261014/known_answers_effects.txt` | 3 s | `known_answers_effects.txt` |
| `ceilings.py` | the largest probability a declaration can reach with one to six evaluable datasets | `python experiments/analysis_numerics_20261014/ceilings.py` | 1.5 min | `ceilings.json` |
| `mutation/` | planted errors (section "Planted errors" below) | see that section | 20 min with 4 clones | `mutation/mutation_results_20261015.json` |
| `first_review/` | the first independent review's simulations, committed for the numbers of ANALYSIS_SPEC.md section 16 (last section) | see that section | | |

## Results

1. The review's failing case (dry run on synthetic scores, pooled local endpoint, decoder 0, six datasets, s from 0.000175 to
   0.00142). Reference: trapezoid rule in log(tau), step 0.01, from 1e-18 to 8 prior scales.

   | prior scale | P(Delta > 0.01), code at 973a97d | amended code | reference | largest error, 973a97d / amended |
   |---|---|---|---|---|
   | 0.05 | 0.97679 (above) | 0.97673 (above) | 0.97673 (above) | 6.3e-05 / 1.6e-11 |
   | 0.1 | 0.97560 (above) | 0.97604 (above) | 0.97604 (above) | 4.4e-04 / 3.8e-12 |
   | 0.2 | 0.94891 (inconclusive) | 0.97570 (above) | 0.97570 (above) | 2.7e-02 / 1.9e-12 |

2. 240 random sets of 1 to 6 datasets (s from 1.02e-06 to 0.05; family, independent and pairs structures; prior scales 0.05, 0.1,
   0.2, 0.25): the largest difference from the reference over P(Delta below / within / above the margin), the posterior mean and the
   heterogeneity is 2.0e-06 (1.99e-06); declarations that differ: 0. Most of it is the part of the half-normal prior beyond 5
   scales, 6e-7 of its mass, which the code leaves out: against the same reference cut at 5 scales the largest difference is
   8.5e-07.

3. Largest relative error over 121 grid nodes of the posterior mean and variance of Delta given the spreads, against 40-digit
   arithmetic (mpmath); old = the formulas at 973a97d, new = the amended solves:

   | datasets | s | old mean | old variance | new mean | new variance |
   |---|---|---|---|---|---|
   | one s small | 0.001 | 7.7e-12 | 2.5e-07 | 9.0e-13 | 3.1e-16 |
   | one s small | 0.0001 | 4.9e-11 | 5.7e-05 | 2.5e-12 | 3.8e-16 |
   | one s small | 1e-05 | 3.6e-11 | 5.0e-03 | 2.1e-12 | 3.9e-16 |
   | one s small | 1e-06 | 4.7e-11 | 2.2e-01 | 2.9e-12 | 4.2e-16 |
   | all s equal | 0.001 | 1.3e-11 | 3.3e-07 | 5.6e-13 | 3.8e-16 |
   | all s equal | 0.0001 | 7.4e-10 | 2.3e-03 | 9.4e-11 | 3.5e-16 |
   | all s equal | 1e-05 | 2.5e-07 | 2.2e+01 | 9.4e-09 | 2.8e-16 |
   | all s equal | 1e-06 | 5.8e-06 | 2.9e+05 | 7.9e-07 | 3.6e-16 |

   The new mean's 8e-7 at s = 1e-6 (all equal) occurs at grid nodes whose weight is zero (spreads far below the differences
   between datasets); the variance is exact to double precision in every case. These are relative errors of two quantities at
   fixed spreads, not 40-digit values of the declaration probabilities, which need the weights of the grid nodes as well; results 4
   and 6 give those.

4. What each reference shares with the code (AN-R11).
   - The reference of `check_pool_numerics.py` (its function `reference`; results 1 and 2) computes Delta's posterior from solves
     with A and the weights of the grid nodes from V = A + (prior variance of Delta) x 11' (log determinant and inverse), as the
     code does; only the integration rule, range and arithmetic differ. A mistake in that algebra would be common to both, so
     results 1 and 2 test the grid and the arithmetic, not the algebra.
   - `ref_pool.py` and `ref_mp.py` were written for the independent re-check of 2026-10-14 and never import the code. The
     model is d_t ~ N(theta_t, s_t^2), theta_t = Delta + u_family(t) + v_t. Given the spreads, the covariance of d is a
     diagonal matrix D = s^2 + tau_d^2 plus one rank-one term per family; the reference inverts it in closed form (no n x n
     solve), adds Delta's rank-one term by the Sherman-Morrison formula, and integrates the spreads by composite
     Gauss-Legendre quadrature in log(tau) over [1e-18, 12 prior scales], in long double arithmetic (`ref_pool.py`) or 50-digit
     arithmetic (`ref_mp.py`). Provenance: evidence archive of the re-check, artifact version
     4fc04d2a-1c2e-481c-bb7e-5157820f755f (scripts `ref_pool.py`, `ref_mp.py`, `validate_algebra.py`), copied without change to
     the algebra.
   - `validate_algebra.json`: against direct n x n algebra in 50 digits at 60 random nodes (1 to 6 datasets, family and
     independent structures, s from 1e-6 to 0.03), the largest relative differences of the log-likelihood, posterior mean and
     posterior variance are 1.6e-12, 5.7e-19 and 1.7e-19 for the long double code and 8.2e-42, 5.3e-42 and 7.9e-32 for the
     50-digit code.

5. The code against the independent reference, and the domain in which it is accurate (`compare_with_reference.json`; AN-R10,
   SI-84). Differences are the largest absolute difference over P(Delta > r), P(Delta < -r) and P(|Delta| <= r) ("probabilities"), and
   over those plus the posterior mean, the 95% interval ends and the heterogeneity ("any value"). The 18 sets are the independent
   re-check's uncalibrated hand-made sets (it also made 20 calibrated to lie 1e-6 and 1e-7 from a declaration threshold; those
   are in its evidence archive only).

   | set | datasets | structure, prior scale | s from to | probabilities | any value |
   |---|---|---|---|---|---|
   | H1_s1e-4_family_0.1 | 6 | family, 0.1 | 9.0e-05 to 1.4e-04 | 1.4e-12 | 1.4e-12 |
   | H1_s1e-4_family_0.2 | 6 | family, 0.2 | 9.0e-05 to 1.4e-04 | 1.9e-12 | 1.9e-12 |
   | H1_s1e-4_family_0.05 | 6 | family, 0.05 | 9.0e-05 to 1.4e-04 | 2.3e-12 | 2.3e-12 |
   | H1_s1e-4_independent_0.1 | 6 | independent, 0.1 | 9.0e-05 to 1.4e-04 | 3.7e-13 | 3.7e-13 |
   | H1_s1e-4_pairs_0.1 | 6 | pairs, 0.1 | 9.0e-05 to 1.4e-04 | 3.5e-12 | 3.5e-12 |
   | H2_s1e-6_family_0.1 | 6 | family, 0.1 | 1.0e-06 to 3.0e-06 | 2.8e-10 | 2.8e-10 |
   | H2_s1e-6_family_0.2 | 6 | family, 0.2 | 1.0e-06 to 3.0e-06 | 1.9e-10 | 1.9e-10 |
   | H2_s1e-6_independent_0.1 | 6 | independent, 0.1 | 1.0e-06 to 3.0e-06 | 1.1e-09 | 1.1e-09 |
   | H3_single_s1e-6 | 1 | family, 0.1 | 1.0e-06 | 5.3e-08 | 3.0e-07 |
   | H3_single_s1e-6_independent | 1 | independent, 0.1 | 1.0e-06 | 5.1e-08 | 1.4e-07 |
   | H3_single_s0.005 | 1 | family, 0.1 | 5.0e-03 | 1.0e-07 | 3.1e-07 |
   | H3_single_s1e-4 | 1 | family, 0.2 | 1.0e-04 | 3.0e-08 | 6.1e-07 |
   | H4_strong_het_0.1 | 6 | family, 0.1 | 1.0e-02 | 8.3e-07 | 1.3e-06 |
   | H4_strong_het_0.05 | 6 | family, 0.05 | 1.0e-02 | 1.5e-04 | 1.5e-04 |
   | H4b_strong_het_0.05 | 6 | family, 0.05 | 1.0e-02 | 1.9e-04 | 1.9e-04 |
   | H4_strong_het_independent_0.05 | 6 | independent, 0.05 | 1.0e-02 | 2.4e-04 | 2.4e-04 |
   | H5_one_tiny_s | 6 | family, 0.1 | 1.0e-06 to 1.2e-02 | 2.4e-11 | 2.5e-11 |
   | H6_P2_scale | 6 | family, 0.25 | 9.0e-02 to 1.2e-01 | 4.4e-10 | 1.4e-09 |

   The H4 sets have differences with a standard deviation of 0.28 and 0.29 (2.8 to 2.9 prior scales at the prior scale 0.1, 5.7 to
   5.8 at 0.05). The spread scan below makes six datasets with standard error 0.01 and differences whose standard deviation is 1 to 8
   prior scales:

   | spread of the differences (standard deviation, in prior scales) | prior scale | probabilities | any value | posterior mass of the spreads beyond 5 prior scales (left out by the grid) |
   |---|---|---|---|---|
   | 1 | 0.05 | 6.0e-09 | 6.0e-09 | 1.4e-09 |
   | 2 | 0.05 | 1.0e-07 | 1.0e-07 | 2.1e-08 |
   | 3 | 0.05 | 6.3e-07 | 7.7e-07 | 2.9e-07 |
   | 4 | 0.05 | 2.4e-06 | 5.5e-06 | 3.2e-06 |
   | 5 | 0.05 | 2.0e-05 | 3.8e-05 | 2.6e-05 |
   | 6 | 0.05 | 5.8e-05 | 1.5e-04 | 1.7e-04 |
   | 8 | 0.05 | 4.4e-04 | 1.4e-03 | 3.8e-03 |
   | 1 | 0.1 | 6.9e-09 | 1.1e-08 | 1.1e-09 |
   | 2 | 0.1 | 5.8e-08 | 1.6e-07 | 1.7e-08 |
   | 3 | 0.1 | 3.5e-07 | 1.6e-06 | 2.4e-07 |
   | 4 | 0.1 | 1.3e-06 | 1.1e-05 | 2.6e-06 |
   | 5 | 0.1 | 1.1e-05 | 7.8e-05 | 2.2e-05 |
   | 6 | 0.1 | 2.9e-05 | 3.0e-04 | 1.5e-04 |
   | 8 | 0.1 | 2.1e-04 | 2.8e-03 | 3.3e-03 |

   Domain of accuracy (AN-R10, SI-84). On ordinary sets the code agrees with the reference to 1.5e-9 (sets H1, H2, H5 and H6), for a
   single dataset to 7e-7, and while the datasets' differences spread by at most three prior scales every reported value is
   within 2e-6 (1.6e-6 in the scan, 2.0e-6 in the 240 random sets). Beyond that the error grows, to 7.8e-5 (any value) and 1.1e-5
   (probabilities) at 5 prior scales and to 3.0e-4 and 2.9e-5 at 6, for the prior scale 0.1; part of it is the posterior mass
   beyond 5 prior scales that the grid leaves out, the rest the integration rule. No declaration differed from the reference in
   any of the 18 sets or 14 scan sets, nor in the 240 random sets. The code's behaviour was not changed for this (only a comment): the ranges are stated in
   ANALYSIS_SPEC.md section 15.4 and in the comment at `GRID_SPAN` in the script.

6. Known answers for standard errors of 1e-6 to 3e-6 (AN-R08; `known_answers_small_s.csv`, read by `tests/analysis`). Five
   synthetic sets of the six real datasets: three whose differences straddle the margin 0.01 almost exactly (family structure at
   prior scales 0.1 and 0.2, and independent) and two whose differences spread by half a prior scale of 0.2 (family and
   independent). The values are from `ref_mp.py` in 50 digits; each was checked against the long double reference on its finest
   rule, and the script stops if the two differ by more than 1e-9 (the largest difference was 6.7e-10). The test compares every
   probability, mean, interval end and heterogeneity at 1e-8 and also asserts the registered grid constants (201 points, 5 prior
   scales, floor 1e-15). Note on the original finding: `precision_mpmath.csv` holds the relative errors of two quantities at grid
   nodes (result 3), not 40-digit values of probabilities, so the new values were generated here.

7. The largest probability a declaration can reach with few evaluable datasets (AN-R12; `ceilings.json`). Every evaluable
   dataset has the same difference d (d from 0.015 to 2, and 0 to 0.015 for the equivalence side) and the same standard error s
   (1e-6, 1e-4, 0.005, 0.01); family structure, prior scale 0.1, margin 0.01:

   | evaluable datasets | largest P(Delta > 0.01) over all s and d tried | smallest d tried that reaches 0.95 (range over s) | largest P(abs(Delta) <= 0.01) |
   |---|---|---|---|
   | one | 0.8232 | none | 0.1449 (s = 1e-06) |
   | two different families | 0.9410 | none | 0.3218 (s = 1e-06) |
   | two same family | 0.8783 | none | 0.2573 (s = 1e-06) |
   | three | 0.9975 | 0.03 to 0.08 | 0.8932 (s = 1e-06) |
   | four | 1.0000 | 0.015 to 0.04 | 1.0000 (s = 1e-06) |
   | five | 1.0000 | 0.015 to 0.03 | 1.0000 (s = 1e-06) |
   | six | 1.0000 | 0.015 to 0.03 | 1.0000 (s = 1e-06) |

   With fewer than three evaluable datasets no declaration can be reached on this grid: the pooled result is inconclusive by
   construction (ANALYSIS_SPEC.md section 15.4).

## Planted errors (mutation test of `tests/analysis`, findings AN-04 of 2026-10-14 and AN-R06 to AN-R09, AN-R13, AN-R14 of 2026-10-15)

`mutation/run_mut.py` plants one error at a time in `scripts/analysis_x1.py` (exact text replacements) in a clone of the
repository and runs `pytest -x tests/analysis` (env scvi-api, one thread); an error is killed when pytest fails. Three sets:

- The rigour reviewer's 40 errors (`mutation/mutants_review.py`, copied unchanged from the review bundle, artifact 3de51e25;
  groups "declaration" and "table"). M16, M26 and M27 are re-targeted in `mutation/mutants_lead.py` because the code they changed
  was rewritten.
- The lead's N-series of 2026-10-14 (`mutation/mutants_lead.py`), 16 errors. N05, which disabled the rule that left a dataset with a
  very small standard error out, is retired: SI-83 removed that rule.
- 17 errors of 2026-10-15 (`mutation/mutants_r3.py`): O02, O11, O12, O16, O18, O20, O22, O23 and O24 are errors that the
  independent re-check planted and the tests of 2026-10-14 did not catch (its own M24 and N11 are in the sets above); F07 to F10
  are the analogues of its O07 to O10 for the floor of SI-83 (the threshold itself, a threshold of 1.5e-6, a floor applied only
  to an exact zero, a floor not applied to the pairs-first average); G01 to G04 are errors in the code added on 2026-10-15 (the
  window's count of all-failed configurations, the refusal of non-finite and negative standard errors, the printed values of the R4
  refusal, the standard error used for a family average).

M24 and N11 occur twice and run once; 73 errors in all. Re-run after committing, from this folder (clones of the committed HEAD
beside the script, at most 4; about 20 minutes with 4):

    cd experiments/analysis_numerics_20261014/mutation
    for i in 0 1 2 3; do git clone -q ../../.. c$i; done && python run_mut.py

Result (`mutation/mutation_results_20261015.json`; the unmutated tests: 70 passed in 85 s): **72 of 73 killed.** All 17 errors of
2026-10-15 are killed, and so are M24 (201 grid points reduced to 101) and N11 (input rows not sorted by tag), which survived on
2026-10-14. The survivor is M28 (no half weights at the two end points of the spread grid): `check_pool_numerics.py`
(`uncaught_planted_errors` in `summary.json`) measures its largest effect on any reported value as 2.0e-6 on 60 stress worlds,
because both ends carry almost no weight (tau = 1e-15 and 5 prior scales); a known answer cannot separate that from the code's own
error of up to 1.6e-6 where the datasets' differences spread by three prior scales (result 5). Three kills (M13, M32, O24) come
from an error raised while the end-to-end record is built (the mutated analysis stops or refuses its input), the others from a
failed assertion.

How the grid errors are killed (AN-R08). `test_pool_known_answers_small_s` asserts the registered grid constants first, so O02 (floor
1e-15 to 1e-9), O22 (151 points) and M24 (101 points) fail at that line. The numbers alone kill them too: with the assertion
removed in the scratch clones, the known answers still kill O02 (`H2_family_0.1`), O22 (`H2_spread_independent_0.2`, where 151
points put the result 4.9e-8 from the 50-digit value, against 1.9e-10 for the code and a tolerance of 1e-8) and M24
(`H2_spread_family_0.2`, where 101 points put it 2.7e-6 away); `known_answers_effects.txt` gives these distances for every case.
The first three cases of the file, whose differences are almost equal, do not separate 151 or 101 points from 201 (below 1e-8
there) but do kill the floor error (6.8e-5).

| id | set | what is planted | result | first failing test |
|---|---|---|---|---|
| M01 | declaration | sign of d reversed (discriminator minus reference) | killed | test_p1_reading[ref0-dis0-mean-want0] |
| M02 | declaration | margin 0.01 -> 0.02 (P1, area, local) | killed | test_end_to_end |
| M02b | declaration | P2 margin 0.05 -> 0.5 | killed | test_end_to_end |
| M03 | declaration | declaration threshold 0.95 -> 0.90 | killed | test_pool_closed_form[d4-s4-inconclusive-levels0] |
| M03b | declaration | declaration threshold 0.95 -> 0.99 | killed | test_pool_brute_force[d1-s1-fam1-0.2-above-grid1] |
| M04 | declaration | P1 prior scale 0.1 -> 0.2 | killed | test_end_to_end |
| M04b | declaration | P2 prior scale 0.25 -> 0.5 | killed | test_end_to_end |
| M05 | declaration | family map: every dataset its own family (family term dropped) | killed | test_pool_known_answers_small_s[H2_family_0.1] |
| M06 | declaration | transfer lambda trained leave-one-DATASET-out instead of leave-one-family-out | killed | test_transfer_lambda |
| M07 | table | window: t degrees of freedom n-1 -> n | killed | test_window |
| M07b | table | window: normal quantile instead of t | killed | test_window |
| M08 | declaration | Holm order: largest p first | killed | test_sign_test_and_holm |
| M09 | declaration | area corner offset 0.10 -> 0.20 below scVI biology | killed | test_read_area[collapse-False-fail0-want0] |
| M10 | declaration | collapse boundary 0.10 -> 0.15 | killed | test_collapse_counts |
| M11 | declaration | jackknife factor (m-1)/m -> 1/m (read_p1) | killed | test_p1_reading[ref0-dis0-mean-want0] |
| M12 | declaration | leave-one-seed-out anchor does not drop the seed from plain scVI (P1) | killed | test_p1_reading[ref10-dis10-mean-want10] |
| M13 | declaration | R4 extension rows not excluded from the arm's points | killed | test_end_to_end (error before the assertions) |
| M14 | table | frontier positions: (B, C) swapped | killed | test_end_to_end |
| M15 | table | window width of every arm computed from the reference's scores | killed | test_end_to_end |
| M16 | table | sign-table z value 1.96 -> 1.0 | killed | test_end_to_end |
| M17 | declaration | Holm family contains local instead of area | killed | test_end_to_end |
| M18 | table | drop_failed sensitivity does not reach the area | killed | test_end_to_end |
| M19 | declaration | pooled P2 rate sums all eight tasks (sim1, sim2 pooled with real tasks) | killed | test_end_to_end |
| M20 | declaration | Agresti-Caffo adds 0.5/1 instead of 1/2 | killed | test_collapse_counts |
| M21 | declaration | area divisor: b* - b0 instead of b_common - b0 | killed | test_end_to_end |
| M22 | declaration | local endpoint pooled from the P1 reading | killed | test_end_to_end |
| M23 | declaration | GRID_SPAN 5 -> 3 (spread grid cut at 3 prior scales) | killed | test_pool_brute_force[d0-s0-fam0-0.1-inconclusive-grid0] |
| M24 | declaration | GRID_N 201 -> 101 | killed | test_pool_known_answers_small_s[H2_family_0.1] |
| M24b | declaration | GRID_N 201 -> 51 | killed | test_pool_brute_force[d0-s0-fam0-0.1-inconclusive-grid0] |
| M25 | declaration | P1 contrast arm: pooled instead of reference | killed | test_end_to_end |
| M26 | declaration | prior scale of the spreads fixed at 0.1 in the sensitivity analyses | killed | test_pool_known_answers_small_s[H2_spread_family_0.2] |
| M27 | declaration | family spread prior half-normal -> weight exp(-tau/scale) (wrong prior density) | killed | test_pool_brute_force[d0-s0-fam0-0.1-inconclusive-grid0] |
| M28 | declaration | tau grid starts at 0 but trapezoid end weights dropped (rectangle rule) | SURVIVED | - |
| M29 | declaration | marginal covariance: dataset spread omitted (family term only) | killed | test_pool_known_answers_small_s[H2_family_0.1] |
| M30 | declaration | posterior probability of 'equivalent' uses 2*margin | killed | test_pool_closed_form[d0-s0-inconclusive-levels0] |
| M31 | declaration | R4 target: b_common used as b* in P1 | killed | test_end_to_end |
| M32 | declaration | low-signal flag ignored (cells pooled) | killed | test_end_to_end (error before the assertions) |
| M33 | declaration | per-dataset s of the area replaced by the P1 s | killed | test_end_to_end |
| M34 | declaration | failed runs counted as plain scVI-like non-collapse (failures not counted in P2) | killed | test_collapse_counts |
| M35 | table | best-of-k indexing wrong (comb(i, k-1)) | killed | test_best_of_k[S1] |
| N01 | lead | revert AN-01: uniform spread grid from 0, no Jacobian | killed | test_pool_brute_force[d1-s1-fam1-0.2-above-grid1] |
| N02 | lead | log-grid Jacobian (weight x tau) dropped | killed | test_pool_brute_force[d0-s0-fam0-0.1-inconclusive-grid0] |
| N03 | lead | TAU_FLOOR 1e-15 -> 1e-5 | killed | test_pool_brute_force[d0-s0-fam0-0.1-inconclusive-grid0] |
| N04 | lead | Delta posterior from V^-1 again (cancellation-prone) | killed | test_pool_closed_form[d5-s5-above-levels0] |
| N06a | lead | R4 check compares b0 only | killed | test_refusals[<lambda>-differ1] |
| N06b | lead | R4 check ignores the low-signal flag | killed | test_refusals[<lambda>-differ2] |
| N07 | lead | X1 seed check removed | killed | test_refusals[<lambda>-X1 seeds] |
| N08 | lead | sign table back to the normal quantile | killed | test_end_to_end |
| N09 | lead | window gap always False | killed | test_window |
| N10 | lead | best-of-k failed runs not counted | killed | test_end_to_end |
| N11 | lead | input rows not sorted by tag (expected equivalent at test tolerance) | killed | test_end_to_end |
| N12 | lead | pool accepts any positive s | killed | test_guards |
| N13 | lead | S_MIN 1e-6 -> 1e-9 | killed | test_combine_structures |
| N14 | lead | R4 targets taken from the record instead of recomputed | killed | test_refusals[<lambda>-differ0] |
| N15 | lead | sign-table t with m instead of m - 1 degrees of freedom | killed | test_end_to_end |
| N17 | lead | best-of-k configs = scored configurations | killed | test_end_to_end |
| O02 | round 3 | grid floor: TAU_FLOOR 1e-15 -> 1e-9 | killed | test_pool_known_answers_small_s[H2_family_0.1] |
| O11 | round 3 | R4 recompute tolerance 1e-9 -> 1e-3 | killed | test_refusals[<lambda>-differ4] |
| O12 | round 3 | R4 recompute: b_common not compared | killed | test_refusals[<lambda>-differ3] |
| O16 | round 3 | sort key: rows sorted by lambda instead of tag | killed | test_end_to_end |
| O18 | round 3 | failed-run total: max instead of sum | killed | test_end_to_end |
| O20 | round 3 | gap flag: looks only at the first interior point | killed | test_window |
| O22 | round 3 | grid point count 201 -> 151 | killed | test_pool_known_answers_small_s[H2_family_0.1] |
| O23 | round 3 | sign table: t degrees of freedom a constant 4 (right only when m = 5) | killed | test_end_to_end |
| O24 | round 3 | R4 recompute ignores the extension rows (computed on the six registered points only) | killed | test_end_to_end (error before the assertions) |
| F07 | round 3 | floor: '<' changed to '<=' at the threshold (s = 1e-6 listed as raised) | killed | test_pool_known_answers_small_s[H2_family_0.1] |
| F08 | round 3 | floor: threshold 1.5e-6 | killed | test_pool_known_answers_small_s[H2_family_0.1] |
| F09 | round 3 | floor applied only when s == 0 | killed | test_combine_structures |
| F10 | round 3 | floor not applied to the pair's averaged standard error | killed | test_combine_structures |
| G01 | round 3 | n_all_failed_between counts every failed configuration, not only those between the window's ends | killed | test_window |
| G02 | round 3 | floor: NaN, infinite and negative standard errors not refused | killed | test_guards |
| G03 | round 3 | refusal message: recorded and recomputed values not printed | killed | test_refusals (the case whose id starts `<lambda>-for atac_small`; its id is a long regular expression, in full in the JSON) |
| G04 | round 3 | pairs: the family's standard error averages the datasets' own s, not the raised ones | killed | test_combine_structures |

## `first_review/`: the first independent review's simulations (DOC-R04)

The first independent review (2026-10-14, review bundle artifact 3de51e25-da4a-49d3-b248-7a8143d3197d) made three simulations whose
results ANALYSIS_SPEC.md section 16 quotes; their scripts were not in the repository. Run from the repo root in conda env `scvi-api`.

- `first_review/sim_pipeline.py`: the frozen P1 rule (`read_p1` with its leave-one-seed-out standard error, `combine`, the
  declarations) on synthetic data with a known truth; limitation 1. `first_review/run_simulations.sh` runs its six scenarios
  (S0 null, S1 heterogeneous, S2 effect, S3 noisy null, S4 curved, W1 several crossings) in 7 minutes on 4 cores; the summaries are
  `first_review/sim_pipeline_<scenario>.txt`.
- `first_review/p2_clustering_sim.py`: coverage of the Clopper-Pearson and Agresti-Caffo intervals of P2 when the 6 strengths of a seed
  share a collapse threshold; limitation 3; 10 s; output `first_review/p2_clustering_sim.json`.
- `first_review/pilot_grid_reading.py`: the pilot's P1 reading on all ten lambda versus the registered grid; limitation 2;
  `python experiments/analysis_numerics_20261014/first_review/pilot_grid_reading.py --pilot-dir <path>/tier12_v2`; 5 s; output
  `first_review/pilot_grid_reading.csv`. The first review made this table from an outcome table built interactively; this script
  builds it with `prereg_rules.outcomes` from the pilot files (918 rows; equal to the review's table in all 40 columns) and reads it
  with `analysis_x1.read_p1`; the six readings agree with the review's to 10 digits.

Results of the re-run of 2026-10-15 (committed outputs; the code is the one of this commit):

- `sim_pipeline_*.txt`: the intervals d +/- 1.96 s of single datasets covered the reading of the noise-free curves in 0.878, 0.878,
  0.879, 0.883, 0.878 and 0.853 of the replicates (S0 to S4, W1; ANALYSIS_SPEC.md limitation 1: "0.85 to 0.88"); the pooled
  95% interval for Delta covered the truth in 1.000, 0.998, 1.000, 1.000, 1.000 and 1.000 of the pooled worlds; "practically
  equivalent" was declared in none of the 2,640 pooled worlds (5 x 520 and 40).
- `p2_clustering_sim.json`: Clopper-Pearson coverage 0.9995 (independent runs, collapse probability rising with lambda) and
  0.914 (a collapse threshold shared by the 6 strengths of a seed); Agresti-Caffo coverage of the difference 0.949 (paired seeds) and
  0.919 (independent seeds); limitation 3 quotes these four (it replaced the first review's "0.91 to 1.00").
- `pilot_grid_reading.csv`: the six cells' differences on all ten lambda and on the registered grid; five do not move, atac_small
  with the conditioned decoder moves from 0.0172 to 0.0282, by 0.0110 (limitation 2's "0.011").

Reproducibility. The first review's logs of S1 and S2 differ from these in their last three lines (pooled results; every
per-dataset line, and all of S0, S3, S4 and W1, are identical to the review's logs). The cause is that the review's script drew the
family effects by iterating a set of strings, whose order depends on the process's hash seed, so two scenarios with a family
spread (tau_f above 0) gave different pooled results from process to process: a re-run of S1 on the code of 973a97d gave 0.994 for
the coverage of the interval for Delta, the review's log 0.963 and an earlier re-run of this script 0.981 and 0.969. The script now
draws them in sorted order; repeats of S1 and S2 under two other hash seeds gave summaries identical to the committed ones, and
the committed summaries are the only ones the specification quotes.
