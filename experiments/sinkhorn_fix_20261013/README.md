# Sinkhorn fix (final review I031/I032; user decision SI-51), 2026-10-13

## What changed (`src/wcd_vae/wcd/alignment.py`, Sinkhorn path only)
1. **Gradient counted once.** Both final transforms (f from g, g from f) back-propagated through both arguments
   of one shared cost matrix, so autograd returned twice the envelope-theorem gradient (two Diracs: +-2 instead of
   +-1). The cost now carries gradient through its row point only (`_cost`: `z[:, None] - z.detach()[None]`, the
   geomloss `cost(x, y.detach())` convention), and the g transform uses that same matrix.
2. **Final epsilon converged.** After the unchanged eps-scaling warm start, alternating Sinkhorn updates (and damped
   symmetric updates for the two debiasing terms) continue at the final eps until every plan's L1 marginal error is
   below `SINKHORN_TOL = 1e-4`; after `SINKHORN_MAX_ITER = 500,000` iterations (50,000 until SI-59) the loss raises
   `RuntimeError`. A NaN marginal error, which is what a NaN or infinite latent gives, stops each solve at once, so
   the loss is NaN and the training guard records the fit as 'diverged', as at the tag (review F1;
   `f1_nonfinite_probe.py`).
3. **Stable arithmetic.** The solve runs in float64 (`SINKHORN_DTYPE`) on distances from direct coordinate differences. The tagged
   float32 expansion |x|^2 + |y|^2 - 2<x,y> cancelled for latents that were narrow and far from the origin (R1
   extension, immune lambda >= 1000), which made the gradient numerical noise.

Not changed: `_sqdist` (the MMD losses use it; SI-52 requires MMD fits to stay bit-identical), the objective
(debiased S_eps, p = 1 Euclidean cost, blur 0.05 sqrt(2d), the 1e-12 stabiliser), the eps schedule.

## Evidence (re-run commands; env `scvi-api`, CPU unless stated)
| script | output | what it shows |
|---|---|---|
| `validate_sinkhorn.py results/validation_12mb.csv 12 8` | `results/validation_12mb.csv` | branch and tagged code vs the converged float64 reference on 3,076 minibatches |
| `tol_sweep.py results/validation_12mb_tol1e-6_cap1e4.csv results/tol_sweep.csv` | `results/tol_sweep.csv` | accuracy and iterations at tol 1e-3..1e-6 on the 598 hardest minibatches |
| `bench_gpu.py results/bench_gpu.csv` (GPU) | `results/bench_gpu.csv` | ms per loss forward + backward, tagged vs branch |
| `summarize.py` | `results/summary.md` | the tables below |
| `pilot_fit.py` (GPU, see below) | `results/pilot_steps_<tag>.csv` (copied from the run folder) | per-step iterations and loss time in one real fit |
| `r4a1_sinkhorn_sensitivity.py results/r4a1_sinkhorn_sensitivity.csv --search <dirs>` (env `wcd-gpu`) | `results/r4a1_sinkhorn_sensitivity.csv` | how far the refitted Sinkhorn arm's best batch score may fall before R4-a1 moves a non-Sinkhorn cell (NB-20261013-09) |
| `bench_gpu.py results/bench_gpu_si58.csv` (GPU) | `results/bench_gpu_si58.csv` | SI-58: ms per call for the tagged code and the branch with the solve in float64 and in float32, and the two dtypes' GPU agreement |
| `SINKHORN_DTYPE=float64` (and `=float32`) `validate_sinkhorn.py results/validation_12mb_si58_<dtype>.csv 12 6` | `results/validation_12mb_si58_<dtype>.csv` | SI-58: both dtypes at the committed code against the reference; the float64 run is also the regression check of 772d59b |
| `si58_summary.py` | `results/si58_summary.md` | SI-58 decision table (below) |
| `f1_nonfinite_probe.py > results/f1_nonfinite_probe.txt` | `results/f1_nonfinite_probe.txt` | NaN, +inf and -inf latents give a NaN marginal error at the first iteration (float64 and float32); latents up to s = 30,000 never give an infinite one (review F1) |

Inputs: `sinkhorn_reference.py` is the final review's reference, copied unchanged from
`review/esc2/sinkhorn_astra/` (sha256 0e6663fb...a709a8, as in that folder's SHA256SUMS). The minibatches are the
review's: `review/esc2/sinkhorn_astra/inventory.json` (sha256 05dbcaba...b960d; 234 Sinkhorn fits, 12 shuffled
128-cell minibatches each from the saved latents and training indices), its 160 early-stage captures and 76
training-mode captures. A3 inputs are standardised as in training and gradients are taken w.r.t. the raw float32
posterior means. The harness reproduces the review's reference values (relative: A1 5e-12, A3 5e-8, R1 extension
2e-4 where the values are near 0)
and its tagged-code R1-extension figures (median cosine 0.281, norm ratio 14.44).

Results at the committed constants (`results/summary.md`): worst cosine 0.999999721, gradient norm ratio
0.999937-1.000067, worst relative value error 6.7e-5 (reference |value| > 1e-6), cap never reached; at most 5,273
iterations at the final eps (A3), 110 (A1). The tagged code: median norm ratio 2.00 (A1), 2.27 (A3), A3 median
cosine 0.869.

## Choices and the measurements behind them
- **Tolerance 1e-4.** Over the 598 hardest minibatches the relative gradient error stays below about 7x the
  tolerance (max 7.5e-4 at 1e-4). Tighter tolerances lengthen the iteration tail sharply (max 29,190 at 1e-5,
  171,388 at 1e-6) for no material gain.
- **Cap 500,000, then raise (user decision SI-59).** The first cap, 50,000, was about 10x the worst case in the
  validation set, but the independent review (finding F2) found a heavier tail on minibatches drawn from the pre-fix
  A3 latents (max 9,789 iterations, atac_small); 500,000 is about 50x that. A fit whose solve cannot converge stops
  loudly instead of training on an approximate gradient; the pilot checks how far real training stays from the cap.
- **Convergence tested every iteration.** Testing every 10 iterations was 20-27% faster on A3 minibatches and 10%
  slower on A1 ones; not adopted (more code for a modest gain).
- **No over-relaxation.** Over-relaxed updates (omega 1.5-1.8) cut the iterations on the four slowest minibatches
  2-8x, but the convergence test would need both marginals and the result would depend on one more parameter.

## Cost
GPU ms per call (RTX 3080, one process): A1 4.0-11.7 -> 6.7-19.5; A3 standardised 12.6-12.8 -> 117-139. The A3
fits carry most of the extra refit time. Pilot (`A3_immune_sinkhorn_la1_matched_hi_c1_s100_ca522bee`, standardised,
lambda 3, alone on the RTX 3080): 56,404 steps in 7,778 s against 2,793.7 s for the pre-fix fit of the same row in
production; solver iterations per step (all solves) median 198, p99.9 1,678, max 3,786, falling from a median of
278 in the first quarter of training to 135 in the last; the loss forward took 6,643 s (85% of the fit).

## Solve precision (user decision SI-58)
The solve dtype is `alignment.SINKHORN_DTYPE`; `validate_sinkhorn.py` takes it from the environment variable
`SINKHORN_DTYPE`, `bench_gpu.py` times both. Numbers from `results/si58_summary.md`:
- **Regression check.** The float64 re-run at 772d59b (cap 500,000, F1 stop) equals the committed
  `validation_12mb.csv` in all 23 shared columns of the 3,076 rows.
- **598 hardest minibatches: float32 meets the SI-58 criterion.** 1 - worst cosine 2.788e-07 (float64 2.789e-07),
  worst relative gradient error 7.489e-4 (float64 7.49e-4), worst relative value error 2.974e-06 (float64
  3.079e-06), no cap hit, the same iterations (median 154, max 5,273).
- **All 3,076 minibatches (declared superset check): float32 does not.** Worst relative value error 3.17e-3 on
  the collapsed R1-extension minibatches (float64 6.67e-05) and 2.69e-4 on A1 (float64 1.55e-05); gradients as in
  float64.
- **No speed-up on the RTX 3080.** Median float32/float64 time per call 0.989-1.014 across the five regimes (A3:
  117.1 vs 117.8 ms and 140.3 vs 142.7 ms). The call is bound by per-iteration kernel launches and the convergence
  test's host sync, not by float64 arithmetic.
- **Decision: keep float64** (NB-20261013-20). `test_sinkhorn_matches_converged_reference` fails if the solve runs in
  float32.
