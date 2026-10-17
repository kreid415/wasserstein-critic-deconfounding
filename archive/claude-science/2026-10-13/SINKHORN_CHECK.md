# Sinkhorn arm: is the implemented gradient a constant multiple of the converged debiased Sinkhorn gradient?

Escalated double-check (one question), repository `review/real_main` at commit 99e942e (read-only), loss
`src/wcd_vae/wcd/alignment.py::sinkhorn_batch_others` as used by `scripts/scvi_adversarial_plan.py`
(critic-free branch of `training_step`: `adv_term = self._align_fn(z_adv, batch_index)`,
`gen_loss = loss_vae + d_coef * adv_term`, `d_coef = lambda`). All outputs of this check are under
`review/esc2/sinkhorn_fable/`. Nothing outside that folder was written; `tier12_v2/X1_local`,
`jhpce_tier12/fillers_x1_x13`, `review/out`, `review/verify` and `review/rereview` were not opened.

## Verdict

**Split verdict, decided per regime (counts are of the 234 completed Sinkhorn fits):**

| Regime | Fits | Finding | Decision |
|---|---|---|---|
| Non-standardised input (`zstd=0`), latent not shrunk below the blur: A1 atac_small / sim1 all lambdas, immune lambda <= 300 (and 2 of the 6 immune lambda=1000 fits), R1 atac_small lambda 10000/30000 | **176** | Implemented gradient = **c = 2.00** x converged gradient: per-fit median c in [1.937, 2.052], per-minibatch cosine median 0.998, 5th percentile 0.956, minimum 0.901 over 4293 minibatches; value error within -1.5%. Holds at the untrained model (cos 0.96-0.99) and from epoch 10 on (cos >= 0.98); weakest at epochs 1-3 (epoch 1: cos 0.87-0.91, c 1.85-1.96). Refit probe (atac_small, lambda=1, 400 epochs, same seed): the as-is fit is closest to the corrected fit at 2*lambda (Procrustes 0.053 vs 0.172 for corrected lambda, 0.158 for as-is 2*lambda, 0.342 seed-to-seed floor); scIB batch/bio raw means of all variants lie inside the seed spread of the official fits. | **KEEP-AND-DOCUMENT** as a correct-objective fit at effective weight **2*lambda** (plus the plan's `d_coef`). |
| Standardised input (`zstd=1`): all A3 Sinkhorn rows (atac_small, immune; lambda 0.3-30) | **36** | Not a constant multiple. Per-fit median cosine 0.808-0.963 (lambda=0.3-1: ~0.95; lambda=3: 0.91; lambda=10: 0.85; lambda=30: 0.84), minibatch minimum 0.669; least-squares c spread 1.72-2.19 (5-95%) with per-fit medians 1.68-2.16; implemented value below the converged value by 13.2% (lambda=10) to 20.9% (lambda=30), worst minibatch -40.6%. Cause: standardisation keeps the minibatch diameter at ~9 blur units while the batches overlap, so convergence needs ~1,000 Sinkhorn iterations at the final epsilon and the single damped step leaves the potentials far from converged (fact b). Refit probe (atac_small, zstd=1, lambda=10, 100 epochs, same seed): as-is vs corrected-2*lambda Procrustes 0.104 = 2x the seed floor (0.052); scIB bio raw mean 0.552 (as-is) / 0.560 (as-is, seed 101) vs 0.493 (corrected 2*lambda) and 0.554 (corrected lambda); batch raw mean 0.648 / 0.656 vs 0.656 / 0.636. The as-is fit behaves like a corrected fit at an effective weight between lambda and 2*lambda, with a direction that is ~30 degrees off the converged gradient on a typical minibatch. | **REFIT** (or re-label as "one damped Sinkhorn step at the final epsilon", not "debiased Sinkhorn divergence") before any A3 claim that depends on the Sinkhorn rows. |
| Shrunk latent (`zstd=0`, minibatch diameter < 0.05 << blur 0.224): immune lambda >= 1000 (A1 and R1), sim1 lambda >= 10000 (R1) | **22** (19 fully, 2 partly, 1 exactly collapsed) | The float32 expansion `|x|^2+|y|^2-2<x,y>` in `_sqdist` cancels: 0.5-12% of pairwise squared distances clamp to 0, `C = 1e-6` there, and the `1/C` factor in dC/dz makes the implemented gradient numerical noise: cosine ~0.05, norm ratio 10-400 against the converged gradient. The same code in float64 on the same minibatches gives cosine 1.000 and ratio 2.000, so this is precision, not the Sinkhorn approximation. Training histories show the regime was entered during training: for lambda >= 3000 the logged adv_grad_norm rose 2-5x over the run and the posterior-mean cloud drifted to |z| ~ 3.4-7.5 from the origin while its spread stayed < 0.05 (immune lambda 3000 / 10000); at immune lambda = 1000 the drift is present (|z| ~ 1.3) but the logged gradient norm did not rise. | Not correct-objective fits at any weight. **Exclude or flag** (they are the collapse tail of the lambda grid: immune lambda >= 1000, sim1 lambda >= 10000); do not use for claims about the Sinkhorn arm. |

**c estimate:** c = 2 (exact at convergence, by the two-sided final extrapolation, fact a); measured per-fit median c = 1.937-2.052 on the 176 clean fits (5-95% of minibatches 1.93-2.04); c is not a constant on the 36 zstd fits (1.72-2.19) and undefined on the 22 shrunk fits.

**Confidence:** high for the clean-regime verdict (4293 minibatches over all 176 fits, full training trajectories at two lambdas and three seeds, a bit-reproducible same-seed refit probe with wrong-c and seed-floor controls, and scIB scores); high that the zstd fits deviate (36/36 fits, monotone in lambda, mechanism reproduced from the untrained model onward); moderate on how much the zstd deviation changes A3's conclusions (one 100-epoch probe configuration, one seed pair; effect on bio score at lambda=10 is 0.059 raw-mean units between as-is and corrected-2*lambda, comparable to the lambda=10 -> 30 step in the official A3 table T5).

## Question and agreed facts

Agreed: (a) on two Dirac masses the autograd gradient is exactly 2x the true derivative (both final
cross-potential transforms carry the cost gradient); (b) the final epsilon gets one damped update, so the
potentials are not converged. Question: across the geometries the 234 fits actually saw, is the gradient used in
training a constant multiple c of the converged debiased-Sinkhorn gradient at the same epsilon, with direction
agreement high enough that each fit is a correct-objective fit at weight c*lambda?

## What was run

**Implemented loss.** `alignment.py` loaded by file path from `review/real_main` (unchanged), evaluated in
float32 as in training, gradient by autograd w.r.t. the loss input `z_adv` (what the generator step
back-propagates; `d_coef = lambda` multiplies it afterwards, so it is excluded from c).

**Reference.** `sk_ref.converged_divergence`: same cost `sqrt(d2 + 1e-12)` (p = 1), same final epsilon
(= blur = 0.05*sqrt(2d) = 0.2236 for d = 10), same measures (a_k uniform on batch k, b_k uniform on the other
cells, mean over batches present), potentials iterated in float64 (undamped log-domain Sinkhorn; damped
fixed point for the symmetric terms) until the row-marginal violation is < 1e-10 (cap 50,000), value
<a,f>+<b,g> - 0.5 OT(a,a) - 0.5 OT(b,b), gradient by the envelope theorem (dS/dC = P_ab - 0.5 P_aa - 0.5 P_bb,
plans detached, back-propagated through C). Verified: central finite differences on 6 coordinates agree to
<= 2e-7; OT_eps(a,b) and OT_eps(a,a) agree with POT 0.9.7 `ot.sinkhorn(method='sinkhorn_log')` to 1e-13; the
Dirac case reproduces fact (a) exactly (implemented grad +-2.0, reference +-1.0, values equal). 12 of 5,616
minibatches hit the iteration cap (all in the shrunk regime, where the value is ~0 and both gradients are
degenerate anyway).

**Metrics per minibatch.** value error (V_impl - V_ref)/|V_ref|; norm ratio ||g_impl||/||g_ref||; cosine of the
flattened 128x10 gradients; least-squares scalar c = <g_impl,g_ref>/||g_ref||^2; per-cell cosine median and
the fraction of cells with per-cell cosine > 0.9.

**Minibatches as training draws them.** scvi's shuffled loader: 128 cells uniformly without replacement, batch
composition following the data (not stratified: `extra = {}` for every Sinkhorn row). (i) Final latents: 24
minibatches per fit for all 234 fits (A1 180, R1 18, A3 36), from the saved posterior-mean latents; for A3 the
plan's `_adv_view` standardisation (minibatch mean/SD of mu, clamp 1e-4) was applied first, since that is the
loss input. (ii) Untrained model: `probe_train.py` refits with the fit's seed and records train-mode (BatchNorm
batch statistics, dropout on) posterior means on 24 training minibatches at epoch 0 (atac_small cond 0 and 1,
immune, sim1; atac_small zstd=1). (iii) Intermediate stages: the same at epochs 1, 3, 10, 30, 100, 200, 400 of
the probe fits.

**Refit probe (CPU, 2 threads, nice 19, same seed; bit-identical on rerun).** `probe_train.py` reproduces
`fit_paper_config.py`'s data path and `fit_adversarial_scvi` call (SCVI, n_latent 10, batch 128, train_size
0.9, zinb, Adam 1e-3, 400-epoch KL warm-up) and substitutes only `_align_fn` in the corrected variant:
`sk_ref.corrected_sinkhorn_batch_others` = same cost / epsilon / schedule, then undamped Sinkhorn at the final
epsilon to row-marginal error < 1e-6 (cap 2,000), then ONE-SIDED extrapolation (<a, softmin(C, lb, g*)> +
<b, g*>, g* detached; symmetric terms likewise), whose autograd gradient is the envelope gradient once.
Group V (atac_small, cond 0, zstd 0, 400 epochs): V1 as-is lambda=1; V2 corrected 2*lambda; V3 corrected
lambda (wrong-c control); V4 as-is lambda=1 seed 101 (seed floor); V5 as-is 2*lambda (lambda-sensitivity
control). Group Z (atac_small, cond 0, zstd 1, 100 epochs): Z1 as-is lambda=10; Z2 corrected 20; Z3 corrected
10; Z4 as-is 10 seed 101. Final latents compared by orthogonal Procrustes disparity, kNN overlap (k=15, mean
Jaccard), and scored with the repository scorer `scripts/score_scib_native.py` (wcd-kbet env, KBET_SEED=0),
alongside the official GPU fits of the same cells.

## Results

### 1. Final latents of the 234 fits (T1, T6; `part1_final_minibatches.csv`, `part1_fit_classification.csv`)

* Clean regime (176 fits, 4293 minibatches incl. the two partly-shrunk fits): cosine median
  0.998, 1st/5th/10th percentile 0.940/0.956/0.961, minimum 0.901; norm ratio
  median 2.000 (5-95%: 1.978-2.106); c median 1.999 (5-95%: 1.930-2.043);
  per-fit median c 1.937-2.052; value error median -0.0%, worst -1.5%; per-cell cosine median
  1.000. Direction agreement is lowest where the latent is widest relative to the blur (lambda <= 1:
  cos 0.955-0.98, c 1.96-2.02) and becomes exact as lambda shrinks the latent (lambda >= 30: cos >= 0.998).
* zstd regime (36 fits): see verdict table; per-lambda medians in `summary.json` (`zstd_unconverged.per_lam`).
  Iterations to converge at the final epsilon: ~90-140 at lambda <= 1, ~540-950 at lambda 3, ~1,000-1,450 at
  lambda 10-30 (T1 `iters`), against the single damped step the code takes.
* Shrunk regime (22 fits): see verdict table and `fp32_tags` in `summary.json`.

### 2. Training trajectories (T2; `part1_snapshots.csv`)

Untrained module (epoch 0, six configurations): cosine median 0.96-0.99, ratio 1.96-2.07,
c 1.95-2.03, value error within -0.1%. Epoch 1 (latent expanded to RMS 1.3-1.5, the widest geometry of the whole
run): cosine 0.87-0.91, c 1.85-1.96, value error -1.2% to -2.0%; epoch 3: cosine 0.94-0.95; from epoch
10: cosine 0.98-0.99, c 1.94-1.96 (V1), the same for V2-V5. Under zstd (Z1, lambda=10) the trajectory
never recovers: cosine 0.88/0.87/0.87 at epochs 10/30/100, c 1.72/1.72/1.77, value error
-17.0%/-18.9%/-17.6%.

### 3. Refit probe (T3, T3b, T4, T4b, T5; `part2_*.csv`, `scores/`)

Group V (lambda=1, 400 epochs). Procrustes disparity / kNN overlap: V1-V2 (as-is vs corrected 2*lambda)
0.053 / 0.379; V1-V3 (corrected lambda) 0.172 / 0.227; V1-V5 (as-is 2*lambda)
0.158 / 0.212; V1-V4 (seed floor) 0.342 / 0.131; official GPU seeds among themselves
0.375-0.395 / 0.119-0.130. Latent RMS: V1 0.710, V2 0.705, V3 0.830, V5 0.529 (T4), i.e. the
corrected fit at 2*lambda reproduces the as-is contraction, corrected lambda under-contracts and as-is 2*lambda
over-contracts. scIB raw means (batch / bio): V1 0.429/0.688, V2 0.431/0.683, V3 0.414/0.685,
V4 0.441/0.690, V5 0.433/0.684; official lambda=1 seeds 0.423-0.455 / 0.684-0.690. At lambda=1 the scores
do not separate any variant from seed noise (lambda=3 does: batch 0.458-0.503), so the score comparison is
consistent with but not the decisive evidence; the latent-geometry comparison is.

Group Z (zstd=1, lambda=10, 100 epochs). Z1-Z4 (seed floor) 0.052; Z1-Z2 (corrected 2*lambda) 0.104;
Z1-Z3 (corrected lambda) 0.079. kNN batch entropy: Z1 0.290, Z4 0.301, Z2 0.309, Z3 0.267.
scIB raw means (batch / bio): Z1 0.648/0.552, Z4 0.656/0.560, Z2 0.656/0.493, Z3 0.636/0.554.
The corrected fit at 2*lambda mixes more and keeps less biology than the as-is fit; the corrected fit at
lambda matches the as-is bio score but mixes less. Provisional (100 of 400 epochs, KL weight 0.25 at the end),
but the comparison between variants is like-for-like.

## Recommendations (review only; no repository file was changed)

1. Paper/methods text for A1 and R1 Sinkhorn results: state that the implemented loss back-propagates twice
   the envelope gradient, so the effective adversarial weight is 2*lambda (the lambda axis of the Sinkhorn arm
   is shifted by a factor 2 relative to its label; nothing else changes).
2. A3 Sinkhorn rows: refit with a converged implementation (undamped iterations at the final epsilon to a
   marginal tolerance, one-sided extrapolation; `sk_ref.corrected_sinkhorn_batch_others` is a working
   reference, ~2x the as-is step cost on CPU), or re-describe the arm. 36 fits x ~20 GPU-minutes.
3. `_sqdist`: centre z on the minibatch mean before the expansion, or use a cancellation-safe pairwise
   distance; this is what turns the gradient into noise once a latent shrinks below the blur. Flag the 22
   shrunk fits in any lambda-curve figure.
4. Going forward, divide the final-step value by 2 (or use the one-sided extrapolation) so that lambda means
   what the text says.

## Deviations and limits

* Refit probe: two configurations instead of one (lambda=1 zstd=0 at full 400 epochs; lambda=10 zstd=1 at 100
  epochs because the converged variant costs ~18 s/epoch on 2 CPU threads). Both atac_small, cond 0.
* Snapshots use train-mode forward passes with fresh dropout masks (statistically, not bit-wise, the
  minibatches of the original GPU runs); minibatch draws are from the full cell set for (i) and from the
  90% training split for (ii)/(iii) (same composition distribution).
* CPU refits are not bit-identical to the GPU fits of the same seed (Procrustes 0.337, the same as between
  seeds), as expected; all probe comparisons are within the CPU runs.
* kNN overlap at k=15 is low for every pair in group Z (0.05-0.07) and does not discriminate there; Procrustes,
  mixing entropy and scIB scores do.

## Files

`sk_ref.py` (reference, corrected loss, metrics), `run_part1_final.py`, `run_part1_snaps.py`, `probe_train.py`,
`compare_probe.py`, `make_tables.py`, `write_report.py`; data: `part1_final_minibatches.csv` (5,616 rows),
`part1_final_fits.csv`, `part1_fit_classification.csv`, `part1_snapshots.csv`, `part2_pairs*.csv`,
`part2_summary*.csv`, `part2_scores.csv`, `scores/*.csv`, `probe_*.npz` (final latents, history, config),
`probe_*_minibatches.pt`, `summary.json`, `tables.md`, logs `part1_final.log`, `probe_*.log`.
Environment: `/home/kendall/.claude-science/conda/envs/scvi-api` (scvi-tools 1.4.2, torch 2.13.0, POT 0.9.7)
for everything except scoring (`wcd-kbet`); `CUDA_VISIBLE_DEVICES=`, `OMP_NUM_THREADS=2`, `nice -n 19`.

## Tables

### T1. Final latents: implemented vs converged, per design cell (24 minibatches of 128 cells per fit)

| stage | task | zstd | lam | fits | mb | cos_med | cos_q05 | cos_min | ratio_med | ratio_q05 | ratio_q95 | c_q05 | c_med | c_q95 | verr_med | verr_maxabs | cellcos_med | z_rms | diam | iters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | atac_small | 0 | 0.100 | 6 | 144 | 0.967 | 0.957 | 0.950 | 2.041 | 1.970 | 2.110 | 1.900 | 1.972 | 2.041 | -0.001 | 0.003 | 0.991 | 0.947 | 7.699 | 100.000 |
| A1 | atac_small | 0 | 0.300 | 6 | 144 | 0.970 | 0.959 | 0.953 | 2.041 | 1.976 | 2.096 | 1.914 | 1.981 | 2.031 | -0.001 | 0.004 | 0.992 | 0.891 | 7.282 | 90.000 |
| A1 | atac_small | 0 | 1.000 | 6 | 144 | 0.977 | 0.969 | 0.961 | 2.034 | 1.990 | 2.099 | 1.941 | 1.988 | 2.051 | -0.001 | 0.002 | 0.994 | 0.720 | 6.004 | 70.000 |
| A1 | atac_small | 0 | 3.000 | 6 | 144 | 0.989 | 0.985 | 0.982 | 2.024 | 1.990 | 2.078 | 1.971 | 2.001 | 2.051 | -0.000 | 0.001 | 0.998 | 0.425 | 3.515 | 40.000 |
| A1 | atac_small | 0 | 10.000 | 6 | 144 | 0.995 | 0.993 | 0.992 | 2.002 | 1.977 | 2.045 | 1.968 | 1.994 | 2.032 | -0.001 | 0.003 | 1.000 | 0.221 | 1.839 | 20.000 |
| A1 | atac_small | 0 | 30.000 | 6 | 144 | 0.998 | 0.996 | 0.994 | 1.999 | 1.985 | 2.030 | 1.980 | 1.995 | 2.021 | -0.001 | 0.004 | 1.000 | 0.145 | 1.202 | 10.000 |
| A1 | atac_small | 0 | 100.000 | 6 | 144 | 1.000 | 0.999 | 0.997 | 1.997 | 1.986 | 2.001 | 1.984 | 1.996 | 2.001 | -0.000 | 0.003 | 1.000 | 0.050 | 0.406 | 10.000 |
| A1 | atac_small | 0 | 300.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 1.999 | 1.997 | 2.001 | 1.997 | 1.999 | 2.001 | 0.001 | 0.002 | 1.000 | 0.005 | 0.050 | 10.000 |
| A1 | atac_small | 0 | 1000.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 2.000 | 1.998 | 2.001 | 1.998 | 2.000 | 2.001 | 0.000 | 0.001 | 1.000 | 0.003 | 0.027 | 10.000 |
| A1 | atac_small | 0 | 3000.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 2.000 | 1.998 | 2.000 | 1.998 | 2.000 | 2.000 | 0.000 | 0.001 | 1.000 | 0.002 | 0.015 | 10.000 |
| A1 | immune | 0 | 0.100 | 6 | 144 | 0.955 | 0.924 | 0.901 | 2.047 | 1.943 | 2.170 | 1.805 | 1.962 | 2.083 | -0.001 | 0.004 | 0.989 | 0.905 | 8.684 | 145.000 |
| A1 | immune | 0 | 0.300 | 6 | 144 | 0.962 | 0.939 | 0.923 | 2.050 | 1.961 | 2.150 | 1.851 | 1.976 | 2.069 | -0.001 | 0.003 | 0.990 | 0.809 | 7.823 | 130.000 |
| A1 | immune | 0 | 1.000 | 6 | 144 | 0.980 | 0.969 | 0.962 | 2.043 | 1.976 | 2.132 | 1.925 | 2.005 | 2.088 | -0.001 | 0.002 | 0.995 | 0.569 | 5.289 | 80.000 |
| A1 | immune | 0 | 3.000 | 6 | 144 | 0.994 | 0.991 | 0.990 | 2.004 | 1.954 | 2.051 | 1.940 | 1.993 | 2.037 | -0.000 | 0.001 | 0.999 | 0.298 | 2.643 | 30.000 |
| A1 | immune | 0 | 10.000 | 6 | 144 | 0.998 | 0.997 | 0.996 | 1.995 | 1.980 | 2.032 | 1.974 | 1.992 | 2.026 | -0.000 | 0.001 | 1.000 | 0.155 | 1.374 | 20.000 |
| A1 | immune | 0 | 30.000 | 6 | 144 | 0.999 | 0.999 | 0.998 | 1.992 | 1.973 | 2.006 | 1.972 | 1.991 | 2.005 | -0.000 | 0.001 | 1.000 | 0.093 | 0.814 | 10.000 |
| A1 | immune | 0 | 100.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 1.999 | 1.987 | 2.001 | 1.987 | 1.999 | 2.001 | 0.000 | 0.001 | 1.000 | 0.023 | 0.237 | 10.000 |
| A1 | immune | 0 | 300.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 2.000 | 1.999 | 2.001 | 1.999 | 2.000 | 2.001 | 0.000 | 0.001 | 1.000 | 0.003 | 0.026 | 10.000 |
| A1 | immune | 0 | 1000.000 | 6 | 144 | 0.998 | 0.077 | 0.031 | 2.030 | 1.997 | 96.172 | 1.984 | 2.002 | 10.143 | -0.010 | 0.085 | 1.000 | 0.218 | 0.016 | 10.000 |
| A1 | immune | 0 | 3000.000 | 6 | 144 | 0.080 | 0.018 | -0.046 | 159.226 | 20.769 | 571.049 | 1.909 | 10.218 | 65.734 | -0.047 | 0.148 | 0.993 | 1.253 | 0.022 | 10.000 |
| A1 | sim1 | 0 | 0.100 | 6 | 144 | 0.958 | 0.941 | 0.931 | 2.070 | 1.975 | 2.162 | 1.888 | 1.990 | 2.072 | -0.003 | 0.008 | 0.986 | 0.927 | 7.123 | 170.000 |
| A1 | sim1 | 0 | 0.300 | 6 | 144 | 0.960 | 0.948 | 0.929 | 2.076 | 1.987 | 2.176 | 1.907 | 1.991 | 2.079 | -0.003 | 0.007 | 0.987 | 0.875 | 6.571 | 160.000 |
| A1 | sim1 | 0 | 1.000 | 6 | 144 | 0.969 | 0.958 | 0.951 | 2.077 | 2.023 | 2.157 | 1.960 | 2.016 | 2.079 | -0.004 | 0.012 | 0.989 | 0.705 | 5.279 | 140.000 |
| A1 | sim1 | 0 | 3.000 | 6 | 144 | 0.986 | 0.978 | 0.975 | 2.042 | 2.003 | 2.095 | 1.977 | 2.010 | 2.056 | -0.002 | 0.009 | 0.995 | 0.423 | 3.224 | 70.000 |
| A1 | sim1 | 0 | 10.000 | 6 | 144 | 0.997 | 0.995 | 0.994 | 1.994 | 1.968 | 2.036 | 1.964 | 1.988 | 2.028 | -0.001 | 0.002 | 1.000 | 0.209 | 1.641 | 20.000 |
| A1 | sim1 | 0 | 30.000 | 6 | 144 | 0.999 | 0.997 | 0.996 | 1.985 | 1.971 | 1.998 | 1.967 | 1.983 | 1.995 | -0.000 | 0.001 | 1.000 | 0.098 | 0.890 | 20.000 |
| A1 | sim1 | 0 | 100.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 1.999 | 1.989 | 2.000 | 1.989 | 1.999 | 2.000 | 0.000 | 0.001 | 1.000 | 0.007 | 0.058 | 10.000 |
| A1 | sim1 | 0 | 300.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 2.000 | 1.999 | 2.001 | 1.999 | 2.000 | 2.001 | 0.000 | 0.001 | 1.000 | 0.003 | 0.020 | 10.000 |
| A1 | sim1 | 0 | 1000.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 2.000 | 1.999 | 2.000 | 1.999 | 2.000 | 2.000 | 0.000 | 0.000 | 1.000 | 0.001 | 0.011 | 10.000 |
| A1 | sim1 | 0 | 3000.000 | 6 | 144 | 1.000 | 1.000 | 1.000 | 2.000 | 1.999 | 2.000 | 1.999 | 2.000 | 2.000 | 0.000 | 0.001 | 1.000 | 0.001 | 0.004 | 10.000 |
| A2A3 | atac_small | 1 | 1.000 | 3 | 72 | 0.962 | 0.952 | 0.943 | 2.080 | 2.040 | 2.123 | 1.967 | 2.003 | 2.049 | -0.002 | 0.005 | 0.987 | 1.000 | 8.412 | 140.000 |
| A2A3 | atac_small | 1 | 3.000 | 6 | 144 | 0.940 | 0.919 | 0.903 | 2.163 | 2.093 | 2.233 | 1.981 | 2.035 | 2.091 | -0.016 | 0.040 | 0.980 | 1.000 | 9.610 | 540.000 |
| A2A3 | atac_small | 1 | 10.000 | 6 | 144 | 0.844 | 0.772 | 0.739 | 2.298 | 2.175 | 2.546 | 1.801 | 1.962 | 2.123 | -0.155 | 0.246 | 0.959 | 1.000 | 9.279 | 1270.000 |
| A2A3 | atac_small | 1 | 30.000 | 3 | 72 | 0.829 | 0.763 | 0.669 | 2.052 | 1.906 | 2.236 | 1.495 | 1.713 | 1.982 | -0.265 | 0.356 | 0.962 | 1.000 | 8.759 | 1450.000 |
| A2A3 | immune | 1 | 0.300 | 3 | 72 | 0.947 | 0.906 | 0.887 | 2.074 | 1.983 | 2.223 | 1.810 | 1.967 | 2.116 | -0.003 | 0.005 | 0.986 | 1.000 | 9.673 | 200.000 |
| A2A3 | immune | 1 | 1.000 | 3 | 72 | 0.939 | 0.906 | 0.867 | 2.143 | 2.051 | 2.245 | 1.889 | 2.003 | 2.113 | -0.007 | 0.014 | 0.981 | 1.000 | 8.660 | 345.000 |
| A2A3 | immune | 1 | 3.000 | 6 | 144 | 0.890 | 0.851 | 0.821 | 2.394 | 2.208 | 2.601 | 2.009 | 2.133 | 2.268 | -0.042 | 0.104 | 0.972 | 1.000 | 9.143 | 950.000 |
| A2A3 | immune | 1 | 10.000 | 3 | 72 | 0.852 | 0.793 | 0.768 | 2.423 | 2.231 | 2.807 | 1.920 | 2.064 | 2.237 | -0.128 | 0.252 | 0.969 | 1.000 | 8.723 | 1445.000 |
| A2A3 | immune | 1 | 30.000 | 3 | 72 | 0.855 | 0.755 | 0.725 | 2.182 | 1.987 | 2.470 | 1.686 | 1.862 | 2.040 | -0.165 | 0.406 | 0.972 | 1.000 | 8.590 | 820.000 |
| R1 | atac_small | 0 | 1e+04 | 3 | 72 | 1.000 | 1.000 | 1.000 | 2.000 | 1.999 | 2.001 | 1.999 | 2.000 | 2.001 | 0.000 | 0.001 | 1.000 | 0.001 | 0.011 | 10.000 |
| R1 | atac_small | 0 | 3e+04 | 3 | 72 | 1.000 | 1.000 | 1.000 | 2.000 | 2.000 | 2.000 | 2.000 | 2.000 | 2.000 | 0.000 | 0.000 | 1.000 | 0.001 | 0.006 | 10.000 |
| R1 | immune | 0 | 1e+04 | 3 | 72 | 0.047 | -0.007 | -0.053 | 193.262 | 33.087 | 694.959 | -0.627 | 8.708 | 60.899 | -0.054 | 0.162 | 0.991 | 2.373 | 0.034 | 10.000 |
| R1 | immune | 0 | 3e+04 | 3 | 72 | 0.058 | 0.018 | 0.009 | 202.632 | 49.178 | 639.041 | 1.695 | 10.887 | 69.908 | -0.038 | 0.107 | 0.980 | 2.136 | 0.036 | 10.000 |
| R1 | sim1 | 0 | 1e+04 | 3 | 72 | 0.281 | 0.100 | 0.052 | 10.011 | 2.035 | 37.958 | 1.655 | 2.310 | 5.683 | -0.003 | 0.030 | 0.818 | 0.056 | 0.002 | 10.000 |
| R1 | sim1 | 0 | 3e+04 | 3 | 72 | 0.232 | 0.094 | 0.004 | 17.505 | 0.000 | 60.202 | 0.000 | 3.723 | 15.937 | -0.036 | 2.43e+10 | 0.442 | 0.120 | 0.003 | 10.000 |

### T2. Training snapshots (train-mode minibatch latents; epoch 0 = untrained module with the fit's seed)

| probe | epoch | mb | cos_med | cos_q10 | cos_min | ratio_med | ratio_q10 | ratio_q90 | c_q10 | c_med | c_q90 | verr_med | verr_min | cellcos_med | frac09 | z_rms | iters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_atac_small_c1_s100 | 0 | 24 | 0.982 | 0.980 | 0.978 | 2.066 | 2.030 | 2.095 | 2.001 | 2.032 | 2.057 | -0.000 | -0.001 | 0.998 | 0.977 | 0.406 | 30.000 |
| U_atac_small_c1_s100 | 1 | 24 | 0.910 | 0.886 | 0.878 | 2.067 | 1.996 | 2.122 | 1.817 | 1.873 | 1.934 | -0.020 | -0.031 | 0.974 | 0.781 | 1.347 | 295.000 |
| U_immune_c0_s100 | 0 | 24 | 0.981 | 0.954 | 0.885 | 1.961 | 1.878 | 2.062 | 1.792 | 1.946 | 2.025 | -0.000 | -0.001 | 1.000 | 0.996 | 0.384 | 30.000 |
| U_immune_c0_s100 | 1 | 24 | 0.866 | 0.818 | 0.793 | 2.249 | 2.078 | 2.451 | 1.775 | 1.961 | 2.116 | -0.019 | -0.034 | 0.970 | 0.730 | 1.288 | 940.000 |
| U_sim1_c0_s100 | 0 | 24 | 0.988 | 0.983 | 0.982 | 2.020 | 1.999 | 2.063 | 1.972 | 1.995 | 2.035 | -0.000 | -0.000 | 0.998 | 0.992 | 0.420 | 30.000 |
| U_sim1_c0_s100 | 1 | 24 | 0.908 | 0.891 | 0.874 | 2.079 | 2.002 | 2.157 | 1.816 | 1.899 | 1.951 | -0.014 | -0.020 | 0.970 | 0.801 | 1.246 | 400.000 |
| V1_asis_l1_s100 | 0 | 24 | 0.982 | 0.980 | 0.978 | 2.066 | 2.030 | 2.095 | 2.001 | 2.032 | 2.057 | -0.000 | -0.001 | 0.998 | 0.977 | 0.406 | 30.000 |
| V1_asis_l1_s100 | 1 | 24 | 0.893 | 0.881 | 0.878 | 2.095 | 2.018 | 2.182 | 1.797 | 1.889 | 1.964 | -0.015 | -0.023 | 0.969 | 0.734 | 1.492 | 305.000 |
| V1_asis_l1_s100 | 3 | 24 | 0.944 | 0.930 | 0.929 | 2.004 | 1.946 | 2.037 | 1.833 | 1.891 | 1.925 | -0.005 | -0.010 | 0.984 | 0.867 | 1.145 | 210.000 |
| V1_asis_l1_s100 | 10 | 24 | 0.990 | 0.988 | 0.981 | 1.967 | 1.944 | 1.988 | 1.923 | 1.947 | 1.969 | -0.000 | -0.003 | 0.998 | 1.000 | 0.417 | 60.000 |
| V1_asis_l1_s100 | 30 | 24 | 0.994 | 0.994 | 0.993 | 1.957 | 1.942 | 1.969 | 1.929 | 1.946 | 1.961 | -0.000 | -0.000 | 1.000 | 1.000 | 0.283 | 30.000 |
| V1_asis_l1_s100 | 100 | 24 | 0.992 | 0.990 | 0.985 | 1.968 | 1.951 | 1.988 | 1.931 | 1.956 | 1.971 | -0.000 | -0.000 | 0.999 | 1.000 | 0.369 | 35.000 |
| V1_asis_l1_s100 | 200 | 24 | 0.985 | 0.980 | 0.974 | 1.989 | 1.966 | 2.021 | 1.934 | 1.960 | 1.991 | -0.000 | -0.000 | 0.996 | 1.000 | 0.553 | 50.000 |
| V1_asis_l1_s100 | 400 | 24 | 0.979 | 0.973 | 0.965 | 1.980 | 1.954 | 2.022 | 1.899 | 1.942 | 1.979 | -0.000 | -0.001 | 0.995 | 0.992 | 0.736 | 60.000 |
| V2_corr_l2_s100 | 0 | 24 | 0.982 | 0.980 | 0.978 | 2.066 | 2.030 | 2.095 | 2.001 | 2.032 | 2.057 | -0.000 | -0.001 | 0.998 | 0.977 | 0.406 | 30.000 |
| V2_corr_l2_s100 | 1 | 24 | 0.908 | 0.885 | 0.872 | 2.060 | 1.990 | 2.154 | 1.795 | 1.875 | 1.950 | -0.013 | -0.021 | 0.976 | 0.766 | 1.441 | 290.000 |
| V2_corr_l2_s100 | 3 | 24 | 0.949 | 0.940 | 0.935 | 1.997 | 1.971 | 2.036 | 1.860 | 1.905 | 1.936 | -0.005 | -0.008 | 0.990 | 0.883 | 1.086 | 200.000 |
| V2_corr_l2_s100 | 10 | 24 | 0.990 | 0.988 | 0.984 | 1.970 | 1.951 | 1.986 | 1.931 | 1.949 | 1.970 | -0.000 | -0.002 | 0.998 | 1.000 | 0.408 | 60.000 |
| V2_corr_l2_s100 | 30 | 24 | 0.994 | 0.994 | 0.993 | 1.960 | 1.944 | 1.976 | 1.931 | 1.949 | 1.965 | -0.000 | -0.000 | 0.999 | 1.000 | 0.281 | 30.000 |
| V2_corr_l2_s100 | 100 | 24 | 0.992 | 0.988 | 0.987 | 1.963 | 1.945 | 1.980 | 1.928 | 1.947 | 1.963 | -0.000 | -0.000 | 0.998 | 1.000 | 0.367 | 40.000 |
| V2_corr_l2_s100 | 200 | 24 | 0.986 | 0.984 | 0.980 | 1.992 | 1.972 | 2.016 | 1.942 | 1.963 | 1.987 | -0.000 | -0.000 | 0.997 | 1.000 | 0.553 | 50.000 |
| V2_corr_l2_s100 | 400 | 24 | 0.979 | 0.972 | 0.962 | 1.994 | 1.945 | 2.014 | 1.897 | 1.954 | 1.973 | -0.000 | -0.001 | 0.994 | 0.984 | 0.734 | 60.000 |
| V3_corr_l1_s100 | 0 | 24 | 0.982 | 0.980 | 0.978 | 2.066 | 2.030 | 2.095 | 2.001 | 2.032 | 2.057 | -0.000 | -0.001 | 0.998 | 0.977 | 0.406 | 30.000 |
| V3_corr_l1_s100 | 1 | 24 | 0.890 | 0.874 | 0.854 | 2.086 | 1.969 | 2.156 | 1.780 | 1.856 | 1.938 | -0.013 | -0.022 | 0.971 | 0.707 | 1.604 | 340.000 |
| V3_corr_l1_s100 | 3 | 24 | 0.915 | 0.907 | 0.876 | 2.026 | 1.942 | 2.085 | 1.782 | 1.852 | 1.914 | -0.005 | -0.009 | 0.983 | 0.762 | 1.565 | 275.000 |
| V3_corr_l1_s100 | 10 | 24 | 0.979 | 0.973 | 0.968 | 1.976 | 1.935 | 2.002 | 1.897 | 1.933 | 1.962 | -0.001 | -0.004 | 0.996 | 0.988 | 0.656 | 110.000 |
| V3_corr_l1_s100 | 30 | 24 | 0.991 | 0.989 | 0.988 | 1.954 | 1.939 | 1.968 | 1.918 | 1.934 | 1.951 | -0.000 | -0.001 | 0.999 | 1.000 | 0.374 | 40.000 |
| V3_corr_l1_s100 | 100 | 24 | 0.987 | 0.983 | 0.981 | 1.983 | 1.957 | 2.000 | 1.929 | 1.957 | 1.974 | -0.000 | -0.001 | 0.997 | 1.000 | 0.539 | 60.000 |
| V3_corr_l1_s100 | 200 | 24 | 0.979 | 0.973 | 0.968 | 1.997 | 1.961 | 2.022 | 1.916 | 1.953 | 1.977 | -0.001 | -0.001 | 0.995 | 0.992 | 0.747 | 80.000 |
| V3_corr_l1_s100 | 400 | 24 | 0.974 | 0.966 | 0.963 | 1.983 | 1.961 | 2.021 | 1.901 | 1.931 | 1.971 | -0.001 | -0.002 | 0.993 | 0.977 | 0.858 | 70.000 |
| V4_asis_l1_s101 | 0 | 24 | 0.986 | 0.981 | 0.980 | 2.010 | 1.992 | 2.044 | 1.971 | 1.981 | 2.010 | -0.000 | -0.001 | 0.999 | 0.992 | 0.393 | 30.000 |
| V4_asis_l1_s101 | 1 | 24 | 0.902 | 0.882 | 0.872 | 2.030 | 1.984 | 2.109 | 1.780 | 1.846 | 1.912 | -0.012 | -0.022 | 0.974 | 0.746 | 1.403 | 360.000 |
| V4_asis_l1_s101 | 3 | 24 | 0.946 | 0.925 | 0.916 | 2.007 | 1.950 | 2.051 | 1.839 | 1.897 | 1.939 | -0.004 | -0.010 | 0.985 | 0.883 | 1.147 | 220.000 |
| V4_asis_l1_s101 | 10 | 24 | 0.989 | 0.988 | 0.987 | 1.964 | 1.940 | 1.984 | 1.919 | 1.943 | 1.963 | -0.000 | -0.002 | 0.998 | 1.000 | 0.420 | 60.000 |
| V4_asis_l1_s101 | 30 | 24 | 0.995 | 0.993 | 0.992 | 1.956 | 1.945 | 1.963 | 1.932 | 1.945 | 1.954 | -0.000 | -0.001 | 1.000 | 1.000 | 0.282 | 30.000 |
| V4_asis_l1_s101 | 100 | 24 | 0.992 | 0.990 | 0.985 | 1.958 | 1.939 | 1.972 | 1.924 | 1.942 | 1.956 | -0.000 | -0.000 | 0.999 | 1.000 | 0.367 | 40.000 |
| V4_asis_l1_s101 | 200 | 24 | 0.987 | 0.982 | 0.981 | 1.971 | 1.952 | 2.006 | 1.921 | 1.946 | 1.979 | -0.000 | -0.001 | 0.998 | 1.000 | 0.546 | 50.000 |
| V4_asis_l1_s101 | 400 | 24 | 0.980 | 0.972 | 0.962 | 1.987 | 1.970 | 2.028 | 1.922 | 1.949 | 1.990 | -0.000 | -0.001 | 0.995 | 0.992 | 0.726 | 70.000 |
| V5_asis_l2_s100 | 0 | 24 | 0.982 | 0.980 | 0.978 | 2.066 | 2.030 | 2.095 | 2.001 | 2.032 | 2.057 | -0.000 | -0.001 | 0.998 | 0.977 | 0.406 | 30.000 |
| V5_asis_l2_s100 | 1 | 24 | 0.919 | 0.898 | 0.879 | 2.066 | 2.020 | 2.150 | 1.842 | 1.893 | 1.955 | -0.018 | -0.026 | 0.976 | 0.781 | 1.265 | 265.000 |
| V5_asis_l2_s100 | 3 | 24 | 0.968 | 0.962 | 0.951 | 1.982 | 1.968 | 1.999 | 1.894 | 1.919 | 1.940 | -0.004 | -0.007 | 0.991 | 0.945 | 0.750 | 140.000 |
| V5_asis_l2_s100 | 10 | 24 | 0.995 | 0.994 | 0.993 | 1.965 | 1.950 | 1.980 | 1.940 | 1.954 | 1.969 | -0.000 | -0.001 | 0.999 | 1.000 | 0.280 | 40.000 |
| V5_asis_l2_s100 | 30 | 24 | 0.996 | 0.995 | 0.994 | 1.964 | 1.953 | 1.975 | 1.945 | 1.957 | 1.969 | -0.000 | -0.001 | 1.000 | 1.000 | 0.224 | 20.000 |
| V5_asis_l2_s100 | 100 | 24 | 0.994 | 0.994 | 0.993 | 1.963 | 1.943 | 1.976 | 1.932 | 1.952 | 1.966 | -0.001 | -0.001 | 1.000 | 1.000 | 0.263 | 30.000 |
| V5_asis_l2_s100 | 200 | 24 | 0.992 | 0.991 | 0.990 | 1.978 | 1.961 | 1.990 | 1.944 | 1.962 | 1.975 | -0.000 | -0.000 | 0.999 | 1.000 | 0.365 | 30.000 |
| V5_asis_l2_s100 | 400 | 24 | 0.985 | 0.981 | 0.980 | 1.985 | 1.959 | 2.012 | 1.922 | 1.960 | 1.981 | -0.000 | -0.001 | 0.997 | 1.000 | 0.547 | 45.000 |
| Z1_asis_l10_zstd_s100 | 0 | 24 | 0.963 | 0.956 | 0.950 | 2.031 | 2.006 | 2.071 | 1.922 | 1.957 | 1.995 | -0.001 | -0.002 | 0.989 | 0.938 | 1.000 | 90.000 |
| Z1_asis_l10_zstd_s100 | 1 | 24 | 0.886 | 0.857 | 0.839 | 1.936 | 1.867 | 2.059 | 1.609 | 1.721 | 1.831 | -0.100 | -0.168 | 0.952 | 0.676 | 1.000 | 605.000 |
| Z1_asis_l10_zstd_s100 | 3 | 24 | 0.925 | 0.905 | 0.895 | 2.041 | 2.011 | 2.090 | 1.836 | 1.887 | 1.952 | -0.093 | -0.132 | 0.974 | 0.801 | 1.000 | 885.000 |
| Z1_asis_l10_zstd_s100 | 10 | 24 | 0.877 | 0.830 | 0.816 | 1.968 | 1.898 | 2.017 | 1.591 | 1.717 | 1.870 | -0.170 | -0.264 | 0.955 | 0.695 | 1.000 | 1195.000 |
| Z1_asis_l10_zstd_s100 | 30 | 24 | 0.871 | 0.834 | 0.789 | 1.964 | 1.893 | 2.027 | 1.579 | 1.718 | 1.856 | -0.189 | -0.277 | 0.954 | 0.719 | 1.000 | 1060.000 |
| Z1_asis_l10_zstd_s100 | 100 | 24 | 0.873 | 0.809 | 0.799 | 2.032 | 1.930 | 2.077 | 1.592 | 1.775 | 1.944 | -0.176 | -0.230 | 0.965 | 0.750 | 1.000 | 1050.000 |
| Z2_corr_l20_zstd_s100 | 0 | 24 | 0.963 | 0.956 | 0.950 | 2.031 | 2.006 | 2.071 | 1.922 | 1.957 | 1.995 | -0.001 | -0.002 | 0.989 | 0.938 | 1.000 | 90.000 |
| Z2_corr_l20_zstd_s100 | 1 | 24 | 0.897 | 0.871 | 0.846 | 1.966 | 1.914 | 2.067 | 1.692 | 1.760 | 1.887 | -0.101 | -0.179 | 0.958 | 0.699 | 1.000 | 730.000 |
| Z2_corr_l20_zstd_s100 | 3 | 24 | 0.927 | 0.899 | 0.882 | 2.066 | 2.034 | 2.105 | 1.844 | 1.908 | 1.963 | -0.086 | -0.152 | 0.973 | 0.797 | 1.000 | 860.000 |
| Z2_corr_l20_zstd_s100 | 10 | 24 | 0.883 | 0.851 | 0.832 | 1.995 | 1.920 | 2.062 | 1.630 | 1.770 | 1.913 | -0.150 | -0.217 | 0.952 | 0.703 | 1.000 | 1000.000 |
| Z2_corr_l20_zstd_s100 | 30 | 24 | 0.873 | 0.833 | 0.820 | 1.965 | 1.903 | 2.033 | 1.582 | 1.710 | 1.886 | -0.160 | -0.240 | 0.940 | 0.660 | 1.000 | 1075.000 |
| Z2_corr_l20_zstd_s100 | 100 | 24 | 0.897 | 0.866 | 0.833 | 2.012 | 1.968 | 2.075 | 1.690 | 1.825 | 1.923 | -0.110 | -0.170 | 0.952 | 0.719 | 1.000 | 945.000 |
| Z3_corr_l10_zstd_s100 | 0 | 24 | 0.963 | 0.956 | 0.950 | 2.031 | 2.006 | 2.071 | 1.922 | 1.957 | 1.995 | -0.001 | -0.002 | 0.989 | 0.938 | 1.000 | 90.000 |
| Z3_corr_l10_zstd_s100 | 1 | 24 | 0.928 | 0.901 | 0.890 | 2.073 | 2.018 | 2.146 | 1.840 | 1.921 | 1.998 | -0.048 | -0.080 | 0.972 | 0.797 | 1.000 | 550.000 |
| Z3_corr_l10_zstd_s100 | 3 | 24 | 0.938 | 0.920 | 0.899 | 2.095 | 2.054 | 2.127 | 1.910 | 1.963 | 2.014 | -0.043 | -0.082 | 0.975 | 0.840 | 1.000 | 705.000 |
| Z3_corr_l10_zstd_s100 | 10 | 24 | 0.905 | 0.880 | 0.850 | 2.058 | 1.999 | 2.098 | 1.748 | 1.873 | 1.930 | -0.094 | -0.184 | 0.969 | 0.770 | 1.000 | 1025.000 |
| Z3_corr_l10_zstd_s100 | 30 | 24 | 0.893 | 0.861 | 0.841 | 2.018 | 1.990 | 2.069 | 1.727 | 1.830 | 1.910 | -0.119 | -0.198 | 0.960 | 0.715 | 1.000 | 990.000 |
| Z3_corr_l10_zstd_s100 | 100 | 24 | 0.903 | 0.876 | 0.853 | 2.060 | 2.029 | 2.137 | 1.782 | 1.880 | 1.973 | -0.093 | -0.138 | 0.962 | 0.766 | 1.000 | 910.000 |
| Z4_asis_l10_zstd_s101 | 0 | 24 | 0.958 | 0.943 | 0.935 | 2.043 | 1.980 | 2.085 | 1.885 | 1.965 | 1.997 | -0.001 | -0.003 | 0.988 | 0.934 | 1.000 | 110.000 |
| Z4_asis_l10_zstd_s101 | 1 | 24 | 0.901 | 0.856 | 0.825 | 1.954 | 1.886 | 2.050 | 1.639 | 1.764 | 1.868 | -0.087 | -0.130 | 0.960 | 0.727 | 1.000 | 630.000 |
| Z4_asis_l10_zstd_s101 | 3 | 24 | 0.930 | 0.900 | 0.852 | 2.056 | 2.024 | 2.100 | 1.826 | 1.911 | 1.976 | -0.079 | -0.131 | 0.973 | 0.812 | 1.000 | 800.000 |
| Z4_asis_l10_zstd_s101 | 10 | 24 | 0.884 | 0.855 | 0.825 | 1.979 | 1.883 | 2.055 | 1.595 | 1.771 | 1.941 | -0.152 | -0.236 | 0.966 | 0.754 | 1.000 | 1060.000 |
| Z4_asis_l10_zstd_s101 | 30 | 24 | 0.873 | 0.817 | 0.780 | 1.951 | 1.890 | 2.007 | 1.563 | 1.710 | 1.857 | -0.188 | -0.287 | 0.961 | 0.715 | 1.000 | 1160.000 |
| Z4_asis_l10_zstd_s101 | 100 | 24 | 0.866 | 0.845 | 0.827 | 2.035 | 1.952 | 2.060 | 1.671 | 1.766 | 1.834 | -0.200 | -0.287 | 0.961 | 0.738 | 1.000 | 995.000 |

### T3. Refit probe, final latents: pairwise comparison (Procrustes disparity, kNN overlap k=15, median per-cell displacement after Procrustes / RMS radius)

| a | b | procrustes_disparity | knn_overlap | median_cell_disp_over_rms |
|---|---|---|---|---|
| V1_asis_l1_s100 | V2_corr_l2_s100 | 0.0532 | 0.3789 | 0.1950 |
| V1_asis_l1_s100 | V3_corr_l1_s100 | 0.1715 | 0.2268 | 0.3549 |
| V1_asis_l1_s100 | V4_asis_l1_s101 | 0.3417 | 0.1311 | 0.5269 |
| V1_asis_l1_s100 | V5_asis_l2_s100 | 0.1575 | 0.2124 | 0.3579 |
| V1_asis_l1_s100 | GPU_s100 | 0.3366 | 0.1524 | 0.5109 |
| V1_asis_l1_s100 | GPU_s101 | 0.3683 | 0.1307 | 0.5409 |
| V1_asis_l1_s100 | GPU_s102 | 0.3916 | 0.1267 | 0.5624 |
| V2_corr_l2_s100 | V3_corr_l1_s100 | 0.1621 | 0.2358 | 0.3367 |
| V2_corr_l2_s100 | V4_asis_l1_s101 | 0.3498 | 0.1293 | 0.5329 |
| V2_corr_l2_s100 | V5_asis_l2_s100 | 0.1749 | 0.2039 | 0.3766 |
| V2_corr_l2_s100 | GPU_s100 | 0.3497 | 0.1503 | 0.5212 |
| V2_corr_l2_s100 | GPU_s101 | 0.3713 | 0.1298 | 0.5461 |
| V2_corr_l2_s100 | GPU_s102 | 0.4113 | 0.1235 | 0.5767 |
| V3_corr_l1_s100 | V4_asis_l1_s101 | 0.3610 | 0.1275 | 0.5319 |
| V3_corr_l1_s100 | V5_asis_l2_s100 | 0.2287 | 0.1828 | 0.4245 |
| V3_corr_l1_s100 | GPU_s100 | 0.3453 | 0.1469 | 0.5270 |
| V3_corr_l1_s100 | GPU_s101 | 0.3595 | 0.1314 | 0.5281 |
| V3_corr_l1_s100 | GPU_s102 | 0.4113 | 0.1242 | 0.5811 |
| V4_asis_l1_s101 | V5_asis_l2_s100 | 0.3221 | 0.1401 | 0.5083 |
| V4_asis_l1_s101 | GPU_s100 | 0.3838 | 0.1296 | 0.5456 |
| V4_asis_l1_s101 | GPU_s101 | 0.3407 | 0.1432 | 0.5116 |
| V4_asis_l1_s101 | GPU_s102 | 0.4282 | 0.1215 | 0.5926 |
| V5_asis_l2_s100 | GPU_s100 | 0.3332 | 0.1563 | 0.5000 |
| V5_asis_l2_s100 | GPU_s101 | 0.3287 | 0.1337 | 0.5141 |
| V5_asis_l2_s100 | GPU_s102 | 0.3452 | 0.1339 | 0.5280 |
| GPU_s100 | GPU_s101 | 0.3753 | 0.1305 | 0.5461 |
| GPU_s100 | GPU_s102 | 0.3846 | 0.1275 | 0.5641 |
| GPU_s101 | GPU_s102 | 0.3949 | 0.1189 | 0.5678 |

### T3b. Refit probe, zstd=1, lambda=10, 100 epochs: pairwise comparison of final latents

| a | b | procrustes_disparity | knn_overlap | median_cell_disp_over_rms |
|---|---|---|---|---|
| Z1_asis_l10_zstd_s100 | Z2_corr_l20_zstd_s100 | 0.1038 | 0.0626 | 0.2517 |
| Z1_asis_l10_zstd_s100 | Z3_corr_l10_zstd_s100 | 0.0787 | 0.0726 | 0.2032 |
| Z1_asis_l10_zstd_s100 | Z4_asis_l10_zstd_s101 | 0.0521 | 0.0557 | 0.1610 |
| Z2_corr_l20_zstd_s100 | Z3_corr_l10_zstd_s100 | 0.0961 | 0.0624 | 0.2527 |
| Z2_corr_l20_zstd_s100 | Z4_asis_l10_zstd_s101 | 0.1331 | 0.0511 | 0.2949 |
| Z3_corr_l10_zstd_s100 | Z4_asis_l10_zstd_s101 | 0.0960 | 0.0639 | 0.2337 |

### T4b. Refit probe, zstd=1: scale and kNN mixing summaries

| name | z_rms | knn_batch_entropy | knn_celltype_purity |
|---|---|---|---|
| Z1_asis_l10_zstd_s100 | 1.1285 | 0.2903 | 0.8406 |
| Z2_corr_l20_zstd_s100 | 1.1416 | 0.3093 | 0.8359 |
| Z3_corr_l10_zstd_s100 | 1.0699 | 0.2668 | 0.8523 |
| Z4_asis_l10_zstd_s101 | 1.1300 | 0.3008 | 0.8402 |

### T6. Per-fit classification counts

| stage | task | zstd | class | fits |
|---|---|---|---|---|
| A1 | atac_small | 0 | clean_c2 | 60 |
| A1 | immune | 0 | clean_c2 | 50 |
| A1 | immune | 0 | shrunk_fp32_noise | 9 |
| A1 | immune | 0 | shrunk_fp32_partial | 1 |
| A1 | sim1 | 0 | clean_c2 | 60 |
| A2A3 | atac_small | 1 | zstd_unconverged | 18 |
| A2A3 | immune | 1 | zstd_unconverged | 18 |
| R1 | atac_small | 0 | clean_c2 | 6 |
| R1 | immune | 0 | shrunk_fp32_noise | 6 |
| R1 | sim1 | 0 | collapsed_zero_gradient | 1 |
| R1 | sim1 | 0 | shrunk_fp32_noise | 4 |
| R1 | sim1 | 0 | shrunk_fp32_partial | 1 |

### T4. Refit probe, final latents: scale and kNN mixing summaries

| name | z_rms | knn_batch_entropy | knn_celltype_purity |
|---|---|---|---|
| V1_asis_l1_s100 | 0.7095 | 0.2218 | 0.8649 |
| V2_corr_l2_s100 | 0.7049 | 0.2205 | 0.8627 |
| V3_corr_l1_s100 | 0.8297 | 0.2097 | 0.8670 |
| V4_asis_l1_s101 | 0.7120 | 0.2229 | 0.8641 |
| V5_asis_l2_s100 | 0.5294 | 0.2215 | 0.8696 |
| GPU_s100 | 0.7120 | 0.2262 | 0.8647 |
| GPU_s101 | 0.7107 | 0.2189 | 0.8659 |
| GPU_s102 | 0.7133 | 0.2337 | 0.8673 |

### T5. scIB raw metrics (scripts/score_scib_native.py, wcd-kbet env, KBET_SEED=0) for the probe latents and the official A1 atac_small sinkhorn fits at lambda 1 and 3

| tag | lam | PCR_batch | ASW_label/batch | iLISI | graph_conn | kBET | NMI_cluster/label | ARI_cluster/label | ASW_label | isolated_label_F1 | isolated_label_silhouette | cLISI | B_raw_mean | C_raw_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1_asis_l1_s100 | 1 | 0.061 | 0.890 | 0.118 | 0.988 | 0.089 | 0.618 | 0.447 | 0.558 | 0.943 | 0.589 | 0.974 | 0.429 | 0.688 |
| V2_corr_l2_s100 | 2 | 0.058 | 0.890 | 0.121 | 0.987 | 0.097 | 0.605 | 0.433 | 0.557 | 0.937 | 0.591 | 0.973 | 0.431 | 0.683 |
| V3_corr_l1_s100 | 1 | 0.000 | 0.888 | 0.095 | 0.987 | 0.099 | 0.609 | 0.432 | 0.558 | 0.942 | 0.597 | 0.971 | 0.414 | 0.685 |
| V4_asis_l1_s101 | 1 | 0.120 | 0.885 | 0.133 | 0.987 | 0.080 | 0.641 | 0.444 | 0.559 | 0.940 | 0.591 | 0.967 | 0.441 | 0.690 |
| V5_asis_l2_s100 | 2 | 0.076 | 0.890 | 0.128 | 0.987 | 0.085 | 0.623 | 0.429 | 0.561 | 0.938 | 0.588 | 0.969 | 0.433 | 0.685 |
| Z1_asis_l10_zstd_s100 | 10 | 0.983 | 0.838 | 0.217 | 0.960 | 0.243 | 0.426 | 0.217 | 0.494 | 0.759 | 0.454 | 0.960 | 0.648 | 0.552 |
| Z2_corr_l20_zstd_s100 | 20 | 0.976 | 0.832 | 0.267 | 0.965 | 0.238 | 0.382 | 0.102 | 0.479 | 0.647 | 0.404 | 0.946 | 0.656 | 0.493 |
| Z3_corr_l10_zstd_s100 | 10 | 0.962 | 0.832 | 0.201 | 0.983 | 0.202 | 0.440 | 0.174 | 0.502 | 0.777 | 0.472 | 0.963 | 0.636 | 0.555 |
| Z4_asis_l10_zstd_s101 | 10 | 0.978 | 0.843 | 0.224 | 0.975 | 0.258 | 0.442 | 0.198 | 0.496 | 0.814 | 0.455 | 0.957 | 0.656 | 0.560 |
| GPU_A1_l1_c0_s100_a9d0e923 | 1 | 0.059 | 0.889 | 0.120 | 0.986 | 0.096 | 0.628 | 0.472 | 0.559 | 0.928 | 0.576 | 0.976 | 0.430 | 0.690 |
| GPU_A1_l1_c0_s101_ce400114 | 1 | 0.041 | 0.892 | 0.113 | 0.985 | 0.083 | 0.629 | 0.431 | 0.558 | 0.936 | 0.581 | 0.972 | 0.423 | 0.684 |
| GPU_A1_l1_c0_s102_401bafda | 1 | 0.170 | 0.898 | 0.130 | 0.985 | 0.093 | 0.621 | 0.459 | 0.565 | 0.931 | 0.587 | 0.973 | 0.455 | 0.689 |
| GPU_A1_l3_c0_s100_9ba7945a | 3 | 0.155 | 0.894 | 0.143 | 0.988 | 0.113 | 0.629 | 0.477 | 0.559 | 0.940 | 0.582 | 0.971 | 0.458 | 0.693 |
| GPU_A1_l3_c0_s101_118cd1a7 | 3 | 0.275 | 0.901 | 0.170 | 0.988 | 0.107 | 0.631 | 0.481 | 0.552 | 0.924 | 0.562 | 0.968 | 0.488 | 0.686 |
| GPU_A1_l3_c0_s102_216bda4a | 3 | 0.323 | 0.893 | 0.189 | 0.987 | 0.124 | 0.638 | 0.441 | 0.561 | 0.940 | 0.574 | 0.971 | 0.503 | 0.688 |
| GPU_A3_matched_c0_s100_2c5f724b | 10 | 0.950 | 0.858 | 0.268 | 0.981 | 0.214 | 0.537 | 0.385 | 0.491 | 0.899 | 0.491 | 0.967 | 0.654 | 0.628 |
| GPU_A3_matched_c0_s101_0271327c | 10 | 0.955 | 0.844 | 0.271 | 0.978 | 0.168 | 0.536 | 0.516 | 0.490 | 0.879 | 0.491 | 0.966 | 0.643 | 0.646 |
| GPU_A3_matched_c0_s102_87de76e4 | 10 | 0.955 | 0.854 | 0.251 | 0.978 | 0.193 | 0.539 | 0.395 | 0.489 | 0.899 | 0.506 | 0.965 | 0.646 | 0.632 |
| GPU_A3_matched_hi_c0_s100_add694fe | 30 | 0.987 | 0.781 | 0.350 | 0.953 | 0.233 | 0.433 | 0.239 | 0.466 | 0.835 | 0.386 | 0.940 | 0.661 | 0.550 |
| GPU_A3_matched_hi_c0_s101_6988d91d | 30 | 0.988 | 0.754 | 0.388 | 0.940 | 0.231 | 0.480 | 0.343 | 0.470 | 0.822 | 0.367 | 0.947 | 0.660 | 0.572 |
| GPU_A3_matched_hi_c0_s102_ee55e3dc | 30 | 0.988 | 0.754 | 0.364 | 0.937 | 0.259 | 0.407 | 0.399 | 0.469 | 0.752 | 0.343 | 0.940 | 0.660 | 0.552 |
| GPU_A3_matched_lo_c0_s100_1dd55537 | 3 | 0.697 | 0.883 | 0.197 | 0.987 | 0.133 | 0.614 | 0.468 | 0.551 | 0.897 | 0.559 | 0.978 | 0.579 | 0.678 |
| GPU_A3_matched_lo_c0_s101_79f4182e | 3 | 0.710 | 0.882 | 0.204 | 0.985 | 0.141 | 0.596 | 0.447 | 0.537 | 0.928 | 0.564 | 0.973 | 0.584 | 0.674 |
| GPU_A3_matched_lo_c0_s102_e90a82d8 | 3 | 0.723 | 0.886 | 0.218 | 0.989 | 0.138 | 0.643 | 0.516 | 0.551 | 0.927 | 0.552 | 0.976 | 0.591 | 0.694 |

