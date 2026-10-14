# Numerical checks of the pooled model in scripts/analysis_x1.py (2026-10-14)

Why: the independent rigour review of branch final-review-fixes at 1e615eb (finding AN-01) showed that the spread grid of the
pooled model (201 points evenly spaced from 0 to 5 prior scales) is too coarse when a dataset's standard error s is below
the grid step, and can change a declaration. Checking the fix found a second limit: the posterior variance of the overall
difference Delta was computed as a difference of two nearly equal numbers, which loses precision when s is small.
The amended code (spec ANALYSIS_SPEC.md sections 4 and 15.4) spaces the 201 points evenly in log(tau) from 1e-15, takes
Delta's posterior from solves with A (the covariance of the dataset differences without Delta's prior variance), and
marks a dataset with s below 1e-6 as not evaluable (`zero_spread`, finding AN-02).

Inputs are synthetic only; no X1 score is read. Re-run from the repo root (conda env `python`: Python 3.11.15, numpy 2.4.6,
scipy 1.17.1, pandas 2.3.3, mpmath 1.4.1; about 3 minutes):

    python experiments/analysis_numerics_20261014/check_pool_numerics.py

Outputs: `review_case.csv`, `stress_240.csv`, `precision_mpmath.csv`, `summary.json`.

## Results (summary.json)

1. The review's failing case (dry run on synthetic scores, pooled local endpoint, decoder 0, six datasets, s from
   0.000175 to 0.00142). Reference: trapezoid rule in log(tau), step 0.01, from 1e-18 to 8 prior scales.

   | prior scale | P(Delta > 0.01), code at 973a97d | amended code | reference | largest error, 973a97d / amended |
   |---|---|---|---|---|
   | 0.05 | 0.97679 (above) | 0.97673 (above) | 0.97673 (above) | 6.3e-05 / 1.6e-11 |
   | 0.1 | 0.97560 (above) | 0.97604 (above) | 0.97604 (above) | 4.4e-04 / 3.8e-12 |
   | 0.2 | 0.94891 (inconclusive) | 0.97570 (above) | 0.97570 (above) | 2.7e-02 / 1.9e-12 |

2. 240 random sets of 1 to 6 datasets (s from 1.02e-06 to 0.05; family, independent and pairs
   structures; prior scales 0.05, 0.1, 0.2, 0.25): the largest difference from the reference over P(Delta below / within /
   above the margin), the posterior mean and the heterogeneity is 2.0e-06; declarations that differ:
   0. (Most of it is the part of the half-normal prior beyond 5 scales, 6e-7 of its mass, which the code leaves
   out: against the same reference cut at 5 scales the largest difference is 8.5e-7.)

3. Largest relative error over 121 grid nodes of the posterior mean and variance of Delta given the spreads, against
   40-digit arithmetic (mpmath); old = the formulas at 973a97d, new = the amended solves:

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

   The new mean's 8e-7 at s = 1e-6 (all equal) occurs at grid nodes whose weight is zero (spreads far below the
   differences between datasets); the variance is exact to double precision in every case.

## Planted errors (mutation test of tests/analysis, finding AN-04)

`mutation/run_mut.py` plants one error at a time in scripts/analysis_x1.py (exact text replacements) and runs
`pytest -x tests/analysis` (env `scvi-api`, one thread). The list is the rigour reviewer's 40 planted errors
(`mutation/mutants_review.py`, copied unchanged from the review bundle, artifact 3de51e25; M16 and M27 re-targeted
because the code they changed was rewritten) plus 17 errors aimed at the 2026-10-14 fixes (N01-N17). Re-run, from
`experiments/analysis_numerics_20261014/mutation` after committing (about 10 minutes):

    for i in 0 1 2 3 4 5; do git clone -q ../../.. c$i; done && python run_mut.py

Result (`mutation/mutation_results_20261014.json`): 54 of 57 caught. Of the reviewer's 10 errors that survived at 1e615eb, 9 are now caught; the three that survive change no result beyond the tests' tolerance:

- M24 (201 grid points per spread reduced to 101): on the log grid this moves any reported probability, mean or
  interval end by at most 1.5e-5 (60 of the stress-test worlds), below the tests' 5e-5; 51 points (M24b) is caught.
- M28 (no half weights at the grid ends): at most 2.0e-6, because both end points of the log grid carry almost no
  weight (tau = 1e-15, and 5 prior scales).
- N11 (input rows not sorted by tag): changes only the order of floating-point sums.

The errors that matter for AN-01 are caught: a uniform grid from 0 (N01), a dropped log-grid factor tau (N02), a
floor of 1e-5 (N03), and the old variance formula (N04).
