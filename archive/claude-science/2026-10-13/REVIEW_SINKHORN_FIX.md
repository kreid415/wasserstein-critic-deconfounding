# Review of the Sinkhorn loss fix (commit 0380465)

Independent per-fix review requested before the user sees the diff (decision SI-52: every fix is reviewed independently before a new tag). Reviewer: Rigor Reviewer, 2026-10-13. No file under review was modified. Every run used the CPU only (`CUDA_VISIBLE_DEVICES` empty); the lead's GPU pilot was not touched. Environments: `scvi-api` (torch 2.13.0; numerical experiments and the lead's validation harness) and `wcd-gpu` (torch 2.4.1; the repository tests).

**What was reviewed.** `git diff 5cb75c0 0380465` for `src/wcd_vae/wcd/alignment.py`, `tests/wcd_vae/test_divergences_v2.py` and `experiments/sinkhorn_fix_20261013/`; pre-fix code = tag prereg-tier12-v2 (c88cce3). The lead's result files (`results/*.csv`, `summary.md`, the running pilot's step file) are in the lead's working copy `code/wcd_final-review-fixes/experiments/sinkhorn_fix_20261013/results/`, because they are not committed. Checksums: `alignment.py` at 0380465 sha256 `7a8c1c2ef37694fc`, at c88cce3 `2163b6d75b898b69`; `sinkhorn_reference.py` `0e6663fbc72b85b9` (identical to the final review's copy).

## 1. Verdict: PASS_WITH_FIXES

1. **Is the fixed loss a correct implementation of the stated method? Yes.** The stated method is: for each batch k present in a minibatch, the debiased Sinkhorn divergence S_eps (a smoothed transport distance between two clouds of cells in latent space) between batch k and all other cells, uniform weights, Euclidean cost, epsilon = 0.05 sqrt(2d), averaged with equal weight over batches. The fixed code returns that value and the envelope-theorem gradient counted once (derivation in section 4). On 6 cases (2 to 6 batches, unequal batch sizes including 1-cell and 2-cell batches, duplicated cells, cost |x-y| and |x-y|^2/2) its gradient agrees with my own independent solver and hand-derived gradient to 9.3e-11 relative at a tight tolerance and to 7.1e-05 at the committed tolerance; my solver's gradient agrees with finite differences of the converged value to 1.4e-07. The pre-fix gradient was 1.96 to 2.48 times too large with cosine 0.921 to 0.995 to the true direction, as the final review found.
2. **Non-Sinkhorn arms are untouched (decision SI-52).** The MMD functions, `_sqdist`, `_batch_masks*` and the registry are source-identical to c88cce3 and give bit-identical values and gradients (float32 and float64, 4 shapes, with and without weights). `alignment.py` is the only file under `src/` or `scripts/` (besides the new `scripts/check_tests.py`) that differs from the tag.
3. **Tests: adequate for the defects fixed, with three small gaps (F4).** The new tests fail on the pre-fix code and on 10 of 11 plausible partial fixes or mutants of the fix; one variant survives (float32 solve, F4/F5). The full `tests/wcd_vae` directory passes in `wcd-gpu` (165 passed, 1 skipped; the one skip needs git history that my copy lacks). Every changed Sinkhorn line and branch is executed except the unchanged `K == 0` early return (F3).
4. **Validation evidence: sound for what it covers.** I re-ran the lead's harness on 20 fits x 2 minibatches (40 minibatches: 12 standardised A3 fits, 6 R1-extension fits, 2 A1 fits): every one of the 11 compared columns is identical to the lead's CSV in 40 of 40 rows. The headline numbers of the README reproduce from the CSV (section 3.7). The limits are the geometry of the minibatches (pre-fix fits, 3 tasks) and the iteration tail (F2, F6).
5. **Two items to settle before the 234 refit fits are launched (each is a few lines of code; F2 also needs a policy decision):** F1 (a non-finite latent, or one that has grown to a scale of about 30,000, now makes the solver spin up to the 50,000-iteration cap and fail as a crash instead of being recorded as 'diverged') and F2 (the 50,000-iteration cap is not shown to be safe: the iteration tail on standardised inputs is heavier than the validation sample suggested, and a cap hit stops the stage deterministically). No blocker was found.

## 2. Findings

| ID | severity | where | problem | suggested fix |
|---|---|---|---|---|
| F1 | fix | `alignment.py` `_ot_eps` / `_ot_eps_sym` loops; critic-free branch of `training_step` | NaN or infinite latent values: 91 to 103 s of useless iterations, then `RuntimeError` (pre-fix: NaN loss in 0.00 s, which the training guard records as 'diverged'). The runner treats the exception as an infrastructure error. | finite check before the solver (as the barycenter branch already does) and `isfinite(err)` exit in both loops |
| F2 | fix | `SINKHORN_MAX_ITER = 50000` (line 50) and its comment (lines 47-48) | Cap headroom is 5.1x on real atac_small A3 minibatches (maximum 9,789), not the 9.5x of the validation set; extrapolation gives 0.001 to 4.6 expected cap events in the A3 atac_small refits; a hit stops the stage with nothing recorded. | decide the policy before launch: higher cap, or recorded-and-counted non-convergence, or an accelerated solver |
| F3 | note | `sinkhorn_batch_others`, line 235 (`K == 0` return) | Still returns a graph-free zero when only one batch is in the minibatch; backward raises (final-review issue I033, fixed elsewhere per the lead). | make sure the I033 fix covers this function and add the case to a test |
| F4 | note | `tests/wcd_vae/test_divergences_v2.py` | The tolerance constant is pinned by one threshold only (tolerance 1e-3 survives the standardised case); the float64 solve is not pinned; `two_diracs_exact` cannot tell `C` from `C.T`. | optional; see F4 for the cheapest additions |
| F5 | note | `sinkhorn_batch_others`, `zd = z.double()` | the fixed loss is about 10x slower per call than the tag on the RTX 3080 (lead's measurement; float64 and the convergence loop together); a float32 solve on the same distances passes all tests and matches the reference on every minibatch I tried. | optional speed-up; measure on the 598 hardest minibatches first |
| F6 | note | `experiments/sinkhorn_fix_20261013` validation design | All 3,076 validation minibatches come from the pre-fix fits (3 tasks: 3, 6 and 10 batches); the refit and the later X1 stage include tasks with up to 22 batches and different latents. | state the scope; re-run the harness on the refit latents before any claim of convergence there |
| F7 | note | module docstring lines 9-11 | 'approximates W1 for small blur': at the committed blur the divergence is 0.53 to 0.89 of the exact W1 on real minibatches. | describe the loss as a debiased Sinkhorn divergence and report epsilon in Methods |
| F8 | note | `p == 2` path; `sinkhorn_divergence_weighted` signature | p = 2 with the default blur has epsilon of order 0.01 and does not converge in a million iterations on unit-scale data; `sinkhorn_divergence_weighted` silently double-counts the gradient if given a cost that is differentiable in both arguments. | document, or make the function private; p = 2 is unused by the arm |

Severity: **blocker** = the arm would not implement the stated method or the plan cannot run; **fix** = should be settled before the refit launches; **note** = no action needed for correctness.

### F1 (fix). Non-finite or very large latent values are not handled the way the pre-registered 'diverged' outcome expects

**Problem.** (1) The loop stops when `err < tol`; for NaN that comparison is false, so a non-finite input runs all 50,000 iterations and then raises `RuntimeError`. The pre-fix code returned a NaN loss at once; the training plan's guard turns a non-finite loss into `NonFiniteLossError`, which the fitter records as the pre-registered outcome 'diverged' (a failure that the stage rules count; `fit_outcome.py`, `fit_paper_config.py`). Any other exception ends the fitter with exit status 1, which `run_stage.py` classes as 'infrastructure': retried up to the attempt limit, then the whole stage stops, and nothing is recorded as a failure. Fixed-seed fits repeat bit for bit, so a retry meets the same step. (2) Whether a non-finite value can reach the solver: torch's `Normal` validation refuses NaN parameters but accepts +inf (executed with torch 2.13: loc = NaN refused, scale = NaN refused, loc = +inf accepted, scale = +inf accepted with an infinite sample). In the critic-free branch the plan checks the losses only after the Sinkhorn term has been computed, whereas the barycenter branch calls `self._nf_check(batch_idx, loss_vae=loss_vae)` first, with the comment 'before POT receives the latent'. I did not run a whole fit into an infinite latent, so I cannot say whether the decoder always raises first; the earlier stages recorded 0 failures in 234 Sinkhorn fits (empty `failures.csv` in A1, A1 extension, A2/A3 and local X1), so the event is rare. (3) Very large but finite latents need ever more iterations: for N(0, s^2 I) latents with 128 cells and 2 batches (s = 100: median 6,926, maximum 8,365; s = 300: median 15,694, maximum 18,191; s = 1000: median 12,836, maximum 13,485; s = 3000: median 16,100, maximum 28,524; s = 10000: median 38,700, maximum 49,018; s = 30000: median 64,855, maximum 76,693). From scale 30000 the count needed exceeds the 50,000 cap (this sweep ran with a 400,000 cap), so a latent that grows through such scales before it becomes NaN would end in `RuntimeError` instead of taking the pre-fix path to 'diverged'. The count is not monotone in scale: at 1e30 the solve converged in 3 iterations, because the plan becomes a hard assignment. For reference, the mean per-dimension SD of the final latents of the 234 pilot fits lies between 3.3e-06 and 1.02 (`logs/latent_spreads.csv`), so scales of 1e4 or more have not been seen: this is a hypothetical divergence path, not an observed one.

**Evidence** (`logs/edge_cases.log`; command `python scratch/edge_cases.py`, env scvi-api; `logs/scale_sweep_part1.log`, `logs/scale_sweep_part2.csv`; `scratch/scale_sweep.py`): one NaN coordinate, fixed code 90.5 s then `RuntimeError: Sinkhorn: L1 marginal error nan > tol 0.0001 after 50000 iterations at the final eps`; pre-fix 0.00 s, value NaN. One +inf coordinate: 102.9 s then the same error. N(0, 1000^2 I): 8,259 iterations, 80 s on one CPU core (pre-fix 0.64 s). Times are CPU, 64 to 128 cells, 2 batches; GPU time not measured.

**Suggested fix.** (a) in the critic-free branch of `training_step`, call `self._nf_check(batch_idx, loss_vae=loss_vae)` before `self._align_fn(...)`; and/or (b) in both loops use `if err < tol or not torch.isfinite(err): break`, so a NaN produces a NaN loss that the existing guard records as 'diverged'. One parametrised row (NaN, inf) in `test_sinkhorn_refuses_bad_p_and_unconverged_solve` would cover both. The cap question for finite blow-up is F2.

### F2 (fix). The iteration cap is not shown to be safe on standardised inputs

**Problem.** The lead's notebook entry NB-20261013-04 (decision on solver settings: tolerance 1e-4, cap 50,000 then raise) and the README of the fix (line 45: 'About 10x the worst case seen') justify the cap, the worst case being 5,273 iterations among the 3,076 validation minibatches. That margin does not hold on a larger sample. On standardised inputs (the 36 A3 fits, and every arm if rule R3 switches standardisation on) the iteration count has a heavy tail: in 10,800 real minibatches of the 18 atac_small A3 fits the median is 202, the 99.9th percentile 5,923 and the maximum 9,789 (5.1x below the cap); in 2,160 minibatches of the 18 immune A3 fits the median is 240 and the maximum 8,051 (6.2x below the cap). The tail comes from the strongest-alignment fits: atac_small fits at lambda >= 10 reach 9,789 iterations and those at lambda <= 3 never more than 882; unaligned (lambda = 0) standardised latents never more than 298. A generalised Pareto fit to the upper tail of the atac_small sample gives a probability of exceeding 50,000 iterations per minibatch between 1.3e-09 and 7.9e-06, depending on where the tail is cut (98%, 99%, 99.5% points), i.e. between 0.001 and 4.6 expected cap events over the 576,000 training steps of the 18 atac_small A3 refits (the upper end of the 90% bootstrap interval reaches 23); the data cannot exclude an event. Without standardisation the cap is far away: the largest count on unstandardised minibatches is 110 among the 2,160 A1 minibatches of the validation set (290 in the early captures).

The lead's pilot is a real refit of one A3 configuration (immune, conditioned, lambda 3, seed 100; the first 42,400 of about 56,400 steps, from the snapshot `logs/pilot_steps_snapshot.csv` taken 2026-10-13 15:48 UTC): per-step iterations (sum of the three solves) have median 222, 99th percentile 718 and maximum 3,786 (13x below the cap), and they fall during training (99th percentile 1,058 in the first third, 421 in the last). The pre-fix final latents of the same configuration are harder (120 minibatches: median 342, 99th percentile 2,860, maximum 3,413), consistent with the pre-fix fits having been pulled twice as hard. The A3 fits are run at 'matched' lambdas (chosen to equalise batch removal across arms, rule R4); if the refit's matched lambdas are chosen the same way, its geometry should be closer to my samples than to the pilot (an expectation, not measured). A cap hit raises `RuntimeError`, which the runner treats as an infrastructure error (F1): retry, same failure, stage stops, nothing recorded; a lost immune A3 fit is about 56,000 steps. The project notes list tasks with many more batches (up to 22 for immune_hum_mou; not re-checked here), and none of them has real-latent data in the validation set.

![Iterations at the final epsilon]({{artifact:art_2eb4a108-d8ec-4a47-be3b-02f7c7acd3ee}})

*Figure 1. Fraction of minibatches (or training steps) needing at least a given number of Sinkhorn iterations at the final epsilon to reach tolerance 1e-4. Solid lines are measured; dotted and dashed lines extrapolate the atac_small tail past the cap (vertical line). Source: `logs/tail_real_A3_part*.csv`, `logs/tail_real_A3imm_part*.csv`, `logs/tail_none1.csv`, the lead's `validation_12mb.csv` and pilot step file.*

| sample | n | median | 99th percentile | 99.9th percentile | maximum | cap / maximum | n >= 10,000 | n >= 50,000 |
|---|---|---|---|---|---|---|---|---|
| final review's A3 minibatches (this fix's validation set) | 480 | 200 | 1482 | 4253 | 5273 | 9.5 | 0 | 0 |
| this review: atac_small A3 fits (K=3 batches) | 10800 | 202 | 1584 | 5923 | 9789 | 5.1 | 0 | 0 |
| this review: immune A3 fits (K=10 batches) | 2160 | 240 | 3021 | 4415 | 8051 | 6.2 | 0 | 0 |
| lead's running pilot (immune, real training steps; sum of 3 solves) | 42400 | 222 | 718 | 1833 | 3786 | 13.2 | 0 | 0 |
| this review: lambda=0 control latents, standardised (no alignment) | 4500 | 33 | 82 | 157 | 298 | 167.8 | 0 | 0 |

| task | threshold (quantile) | exceedances used | tail shape | P(>50,000) per minibatch | 90% bootstrap interval | refit steps (A3 fits of this task) | expected cap events |
|---|---|---|---|---|---|---|---|
| atac_small | 0.980 (1110 iterations) | 214 | 0.45 | 7.9e-06 | 3.3e-07 to 4.0e-05 | 576000 | 4.56 (0.19 to 23) |
| atac_small | 0.990 (1584 iterations) | 108 | 0.08 | 1.3e-09 | 0.0e+00 to 8.1e-07 | 576000 | 0.000751 (0 to 0.47) |
| atac_small | 0.995 (2627 iterations) | 54 | 0.16 | 4.3e-08 | 0.0e+00 to 2.3e-06 | 576000 | 0.025 (0 to 1.4) |

**Suggested fix.** Choose the policy before the launch, with one of: (a) raise `SINKHORN_MAX_ITER` tenfold or more (the loop still exits at the tolerance, so the typical cost is unchanged; only a call that really needs that many iterations gets slower); (b) at the cap return the current iterate, count the event, and let the fit fail only above a pre-set count (needs a PREREG decision on what is recorded); (c) use over-relaxed or accelerated updates (the lead measured 2 to 8 times fewer iterations on the four slowest minibatches and declined for simplicity). In every case update the comment in `alignment.py` lines 47-48: 'the hardest needed 5,273 iterations' holds for the validation minibatches only.

### F3 (note). The `K == 0` return is unchanged and not tested

`sinkhorn_batch_others` still returns `z.new_zeros(())` when fewer than two batches are present (line 235). Executed: value 0.0, `requires_grad` False, `grad_fn` None, so `torch.autograd.grad(adv_term, z_adv)` in `training_step` raises 'element 0 of tensors does not require grad' (`logs/edge_cases.log`, `logs/types_check.log`). This is final-review issue I033; the lead says it is fixed in another worktree. Coverage of `alignment.py` by `tests/wcd_vae` leaves lines 78, 80, 101, 119, 235 unexecuted; line 235 is the Sinkhorn one.

### F4 (note). What the new tests pin and what they do not

Mutation check (`scratch/make_variants.py`, `scratch/run_variants.sh`; the two Sinkhorn test files, 28 tests, `wcd-gpu`, CPU): 11 of 12 variants fail at least one new test. Survivor: V4 (float32 solve on direct-difference distances).

| variant | what was changed | failed test cases | failing tests |
|---|---|---|---|
| V0 | pre-fix code (tag prereg-tier12-v2) | 7 | test_sinkhorn_batched_equals_one_problem_at_a_time, test_sinkhorn_matches_converged_reference, test_sinkhorn_refuses_bad_p_and_unconverged_solve, test_sinkhorn_two_diracs_exact |
| V1 | gradient fixed, no extra iterations at the final epsilon | 4 | test_sinkhorn_matches_converged_reference, test_sinkhorn_refuses_bad_p_and_unconverged_solve |
| V2 | convergence loop kept, gradient double-counted again | 4 | test_sinkhorn_matches_converged_reference, test_sinkhorn_two_diracs_exact |
| V3 | gradient + convergence kept, distances from the float32 expansion, no float64 | 2 | test_sinkhorn_matches_converged_reference |
| V4 | everything kept, solve in float32 (float64 cast removed) | 0 | none (all 28 pass) |
| V5 | tolerance 1e-2 | 2 | test_sinkhorn_matches_converged_reference |
| V6 | tolerance 1e-3 | 1 | test_sinkhorn_matches_converged_reference |
| V7 | marginal error averaged over cells instead of summed | 2 | test_sinkhorn_matches_converged_reference |
| V8 | symmetric update without damping | 2 | test_sinkhorn_batched_equals_one_problem_at_a_time, test_sinkhorn_matches_converged_reference |
| V9 | reaching the cap does not raise | 2 | test_sinkhorn_refuses_bad_p_and_unconverged_solve |
| V10 | second potential's transform uses C.T instead of C | 2 | test_sinkhorn_matches_converged_reference |
| V11 | self-transport terms counted twice (4 instead of 2) | 4 | test_sinkhorn_matches_converged_reference, test_sinkhorn_nonnegative_and_zero_for_identical_batches, test_sinkhorn_two_diracs_exact |

1. **Thresholds are sensible.** Fixed code: standardised case value error 3.90e-08, gradient error 9.43e-05 (thresholds 1e-5 and 1e-3); narrow-far case 1.07e-06 and 3.27e-05. That is at most 0.11 of every threshold, so a different torch or CPU will not trip them. The broken variants (pre-fix, gradient fix without the loop, float32 expansion) exceed the gradient threshold by 4.2x to 1450x.
2. **The tolerance constant is pinned by one threshold.** With tolerance 1e-3 the standardised case passes (gradient error 8.30e-04, 0.83x its threshold); only the value check of the narrow-far case fails (2.22x). The README justifies 1e-4 (maximum gradient error 7.5e-04 over the hardest minibatches), so a one-line assertion on the constant, or a stricter standardised threshold, would pin it.
3. **`test_sinkhorn_two_diracs_exact` cannot tell the second potential's `C` from `C.T`** (variant V10): with two batches the two problems mirror each other. The converged-reference tests do catch V10, so there is no gap in practice.
4. The reference tests load `experiments/sinkhorn_fix_20261013/sinkhorn_reference.py` by path; moving or archiving that folder breaks them.
5. The docstring figure for the gradient fix without the convergence loop (4.4e-2) was not reproduced: my construction (convergence loop deleted, float64 and direct differences kept) gives 1.0e-1 on the standardised case. Both exceed the threshold by more than 40x.

### F5 (note). The float64 solve may not be needed

The lead measured A3 calls at 12.6 ms (tag) versus 117 to 139 ms (fix) on the RTX 3080, whose float64 rate is 1/64 of float32. Variant V4 (same direct-difference distances, solve in float32) passes all 28 tests. On the 40-minibatch validation sample it agrees with the converged reference with cosine at least 0.999999968 and norm ratio 0.999977 to 1.000015, and on the 24 hardest real A3 minibatches the gradient error is at most 6.1e-04; only the logged value is less precise (float32 rounding: up to 2.5e-04 relative on collapsed R1-extension minibatches whose value is about 1e-4, 7.4e-06 elsewhere, against 1.1e-06 in float64). If this survives the 598 hardest minibatches of the tolerance sweep, the float64 penalty on the RTX 3080 could be avoided (the speed-up was not measured here). Not needed for correctness, and keeping float64 is consistent with the stated priority of correctness over speed. Distances from direct coordinate differences are what remove the cancellation: variant V3 (float32 expansion) exceeds the narrow-far gradient threshold by a factor of 328.

### F6 (note). Scope of the validation evidence

The harness is sound: same inputs to reference and fix (float32 values converted exactly to float64), the same standardisation function as the training plan for adversary input 'mean' (all 234 pilot fits use it), gradients taken with respect to the raw posterior means through the standardisation, the converged reference's marginal residual (largest absolute deviation of its plan's row and column sums) at most 3.8e-09 over all 3,076 rows, and the reference's values reproduce the final review's to 5.4e-12 (A1), 1.8e-11 (R1 extension), 4.7e-08 (A3). Limits: (1) the 3,076 minibatches come from the final latents of the pre-fix fits (plus 192 early-training captures and 76 training-mode captures), i.e. from fits trained with a gradient twice as strong as the stated method; the refit at the same nominal lambda pulls half as hard, and the matched lambdas will shift. The iteration tail concentrates in the strongest-alignment fits, so the pre-fix A3 latents are a reasonable but not exact stand-in. (2) Tasks: atac_small (3 batches), immune (10), sim1 (6); the project notes list later-stage tasks with up to 22 batches (not re-checked). (3) 12 minibatches per fit. Collapsed latents are covered: scaling real minibatches towards their mean from per-dimension spread 5.5e-01 down to 2.1e-06 keeps the committed-tolerance gradient within 3.4e-05 of the tight-tolerance gradient (cosine at least 0.999999999), because the p = 1 gradient does not shrink with the spread.

### F7 (note). 'Approximates W1' needs a qualifier

The module docstring says the divergence 'approximates W1' for small blur. At the committed blur (epsilon 0.224 for 10 latent dimensions) the debiased divergence is 0.86 to 0.89 of the exact W1 (network simplex, same minibatches and cost) on A1-type latents and 0.53 to 0.77 on standardised A3 latents (`logs/w1_gap.log`, 8 minibatches per row). Methods should call it a debiased Sinkhorn divergence at a stated epsilon, not a W1 estimate.

### F8 (note). Two API hazards, neither used by the arm

1. `p = 2` with the default blur gives epsilon = (0.05 sqrt(2d))^2, about 0.015 for d = 3 and 0.05 for d = 10. In my first gradient check (24 cells, d = 3, unit-scale data) the solver did not reach tolerance 1e-12 in 2,000,000 iterations (L1 error 1.95e-7; `logs/gradient_checks.log`), so with the committed cap it would raise. The arm uses p = 1; the `p = 2` tests only use two Diracs.
2. `sinkhorn_divergence_weighted` is correct only for a cost whose gradient flows through the row point alone (`_cost`). Given a cost differentiable in both points (the old tests build one from `_sqdist`) it double-counts the gradient without any error. The batched-versus-single test does exactly that but compares values only.

## 3. The eight requested checks

### 3.1 Gradient (envelope theorem, exactly once)

Commands (env `scvi-api`, CPU): `python scratch/check_gradient.py 0,1,3,4` and `python scratch/check_gradient.py 2,5` (library `scratch/myref.py`; output `logs/gradient_checks_all.csv`). `myref.py` is written for this review and imports no project code: log-domain Sinkhorn warm start, then damped Newton on the semi-dual until both marginals agree to better than 1e-13, plans of the three problems per batch, and the gradient assembled by hand from the plans (section 4). Columns: finite differences (step 1e-5, central) of the converged value at 6 random coordinates; the fixed code's autograd gradient at tolerance 1e-10 and at the committed 1e-4; the pre-fix code; the final review's reference (`sinkhorn_reference.py`).

| case | cells | batches K | finite-difference vs my plan gradient (worst of 6 coordinates, relative) | fixed code, tight tolerance 1e-10, vs my gradient | fixed code, committed tolerance 1e-4, vs my gradient | pre-fix gradient norm / true norm | pre-fix gradient cosine to true | review's reference vs my gradient |
|---|---|---|---|---|---|---|---|---|
| K2 unequal 2+4 d2 p1 | 6 | 2 | 3.7e-09 | 9.3e-11 | 7.1e-05 | 2.02 | 0.985 | 5e-14 |
| K3 unequal 4/8/12 d3 p1 | 24 | 3 | 1.3e-08 | 4.0e-11 | 4.2e-05 | 1.96 | 0.979 | 9e-12 |
| K3 unequal d3 p2 (spread 0.25) | 24 | 3 | 9.7e-09 | 3.7e-11 | 3.7e-05 | 2.08 | 0.995 | n/a |
| K4 with duplicated points d4 p1 | 24 | 4 | 1.4e-07 | 3.5e-11 | 3.3e-05 | 2.03 | 0.971 | 2e-12 |
| K5 incl. 1-cell and 2-cell batches d10 p1 | 46 | 5 | 1.2e-08 | 3.1e-11 | 3.2e-05 | 2.48 | 0.921 | 3e-13 |
| K6 d10 p1 128-ish cells unequal | 108 | 6 | 6.5e-08 | 1.5e-11 | 2.2e-05 | 2.01 | 0.943 | 2e-13 |

Result: finite differences confirm the hand-derived gradient to 1.4e-07; the fixed code equals it to 9.3e-11 (values to 1.1e-15); at the committed tolerance the gradient error is 2.2e-05 to 7.1e-05; the final review's reference agrees with mine to 9.0e-12, so the lead's reference is itself correct. The pre-fix gradient is 1.96 to 2.48 times the true norm. The p = 2 case is not available for the review's reference (it fixes p = 1). The convention matches the installed geomloss 0.3.1 source (`_legacy/sinkhorn_samples.py`: `C_xy = cost(x, y.detach())`, `C_yx = cost(y, x.detach())`, `C_xx = cost(x, x.detach())`).

### 3.2 Convergence criterion

`_marginal_err(log_w, h_old, h_new, eps)` returns max over the K problems of the L1 distance between `w * exp((h_old - h_new)/eps)` and `w`. Reading the algebra: after the first potential is updated from the second, the plan's row sums equal the weights exactly, and its column sums are `w * exp((h_old - h_new)/eps)`, so the quantity is the L1 error of the other marginal. In `_ot_eps` it is evaluated on the b side after the f update. On 300 real A3 problems (251,894 iterations in all; `scratch/err_monotone.py`, `logs/err_monotone.log`) the tested error never increased from one iteration to the next (increases: 0), and the error of the iteration after the stopping point was below the tested one in all of the problems (largest ratio 0.9998, median 0.9391), so the stopping quantity behaves as a decreasing sequence on these problems; the largest ratios belong to problems that shrink their error by only about 0.02% per iteration, which is why they need thousands of iterations. In `_ot_eps_sym` the plan of the pre-damping potential is measured, so a period-2 oscillation cannot pass: an oscillating iterate is not a fixed point and has nonzero error. Zero-weight cells have `log_w = -inf`, hence `exp(-inf + finite) = 0` and no NaN; this is exercised by the 1-cell and 2-cell batches of the K = 5 case above and the single-cell case in 3.3.

Can the loop stop while a plan is still far from converged? Four experiments.

1. Lead's tolerance sweep, recomputed from `results/tol_sweep.csv` (480 standardised minibatches, plus 118 unstandardised ones). At the committed tolerance 1e-4 the maximum gradient error is 7.5e-04 (7.5x the tolerance) and the maximum iteration count 5,273; at 1e-5 6.0e-05 and 29,190; at 1e-6 171,388 iterations. The README concludes that tighter tolerances lengthen the iteration tail sharply for no material gain. The conclusion stands, but the growth of the maxima is not typical: from 1e-4 to 1e-6 the median iteration count goes from 200 to 334 and the 99th percentile from 1,482 to 3,636. The maxima at 1e-5 and 1e-6 belong to one minibatch (A3_atac_small_sinkhorn_la1_matched_hi_c1..., minibatch 3) that needs 103, 131, 29,190 and 171,388 iterations at tolerances 1e-3, 1e-4, 1e-5 and 1e-6: its marginal error falls below 1e-4 early and then creeps (a plateau). Its gradient error at the committed tolerance is 2.1e-04, so the tolerance did not stop it far from the converged gradient. For the other 479 minibatches the count at 1e-6 is a median 1.7x and at most 5.7x the count at 1e-4. Such plateaus are one reason why the iteration count at a fixed tolerance has a heavy tail (F2).
2. 24 hardest real A3 minibatches (needing 3,678 to 9,789 iterations at the committed tolerance): gradient error against the converged reference median 3.8e-04, maximum 6.1e-04; at tolerance 1e-5 the maximum is 6.1e-05; cosine to the reference at least 0.999999815; norm ratio 0.999928 to 1.000102.
3. Designed near-disconnected geometry (two tight clusters of SD 0.3 a distance 3 to 20 apart, batch 0 half and batch 1 nine tenths in cluster A, standardised): iterations 11 to 20, gradient error against the converged reference at most 1.3e-04.
4. Collapse regime (the other way a fixed absolute tolerance could fail): real latents of the strongest-lambda fits have per-dimension SD down to 3.3e-06 (R1 extension; 2.1e-04 among the A1 fits with lambda >= 1000). Scaling real minibatches towards their mean over spreads 5.5e-01 to 2.1e-06: gradient error at the committed tolerance at most 3.4e-05 against the 1e-11 solve (`logs/narrow_check.csv`).

Conclusion: no instance in these experiments stopped far from the converged gradient. The only open question is the cap (F2), not the tolerance.

### 3.3 Numerics and types

Command: `python scratch/edge_cases.py` and `python scratch/types_check.py` (env `scvi-api`, CPU, float32 inputs unless stated). Iterations are at the final epsilon for [cross, self a, self b].

| case | fixed code | pre-fix code |
|---|---|---|
| single-cell batch (sizes 1/40/87), N(0,I) d10 | 0.23 s: value=3.02376 \|g\|=0.261 iters=[18, 1, 1] | 0.11 s: value=3.02214 \|g\|=0.603 |
| two batches, sizes 1/127 | 0.09 s: value=3.55451 \|g\|=0.678 iters=[1, 1, 1] | 0.17 s: value=3.55387 \|g\|=1.36 |
| only ONE batch present (K=0 branch) | 0.00 s: value=0 backward raises 'does not require grad' iters=[] | 0.00 s: value=0 backward raises 'does not require grad' |
| all cells identical, 2 batches | 0.01 s: value=0 \|g\|=0 iters=[1, 1, 1] | 0.00 s: value=0 \|g\|=0 |
| some duplicated cells, 2 batches | 0.02 s: value=2.47549 \|g\|=0.176 iters=[30, 1, 1] | 0.01 s: value=2.47092 \|g\|=0.373 |
| one NaN coordinate | 90.51 s: RuntimeError after 50,000 iterations (error is NaN) iters=[50000] | 0.00 s: value=nan grad finite=False \|g\|=nan |
| one +inf coordinate | 102.94 s: RuntimeError after 50,000 iterations (error is NaN) iters=[50000] | 0.00 s: value=nan grad finite=False \|g\|=nan |
| scale 1e3 (N(0,1e6 I)), 2 batches | 80.24 s: value=2809.1 \|g\|=0.177 iters=[8259, 1, 1] | 0.64 s: value=2797.01 \|g\|=0.241 |
| scale 1e30, 2 batches | 10.52 s: value=2.78317e+30 \|g\|=0.186 iters=[3, 1, 1] | 0.58 s: value=nan grad finite=False \|g\|=nan |

Types (`logs/types_check.log`): float32 input gives float32 value and gradient; float64 input gives float64; `grad(retain_graph=True)` followed by `backward()` (the training plan's pattern) is consistent; `batch_index` of int16, int32 and int64 all work. NaN and infinity: F1. Cap: F2. Duplicates and identical cells give zero distance, handled by the 1e-12 stabiliser (zero gradient). Float64 solve with float32 input and output: confirmed. p = 2 : F8.

### 3.4 Cap and tolerance justification

See F2 and Figure 1. The tolerance is justified by the evidence (3.2); the cap is not (F2). Cost context: the 234 refit fits comprise 9,696,336 Sinkhorn evaluations, of which 1,591,272 are standardised A3 steps. If rule R3 switches standardisation on for all arms, the X1 Sinkhorn grid also runs in this regime.

### 3.5 MMD untouched

Command: `python scratch/check_mmd_identity.py` (`logs/mmd_identity.log`). Result lines:

```
top-level names only in old: []  only in new: ['SINKHORN_MAX_ITER', 'SINKHORN_TOL', '_cost', '_marginal_err', '_not_converged']
_ot_eps                      source DIFFERENT  sha256(old)=be0eb37f3ab6 sha256(new)=6b53874fda6d
_ot_eps_sym                  source DIFFERENT  sha256(old)=b4db8c83e8ea sha256(new)=35dc1ed8d406
_sqdist                      source IDENTICAL  sha256(old)=129eb4fbb80c sha256(new)=129eb4fbb80c
mmd_batch_others             source IDENTICAL  sha256(old)=22a2782d0719 sha256(new)=22a2782d0719
mmd_batch_pool               source IDENTICAL  sha256(old)=361c27653379 sha256(new)=361c27653379
mmd_reference                source IDENTICAL  sha256(old)=67b3e0c25ab6 sha256(new)=67b3e0c25ab6
sinkhorn_batch_others        source DIFFERENT  sha256(old)=d321006fbae6 sha256(new)=8ffd8953570d
sinkhorn_divergence_weighted source DIFFERENT  sha256(old)=78e0467d2234 sha256(new)=6da87435f515
bitwise identical on all MMD/_sqdist value+gradient checks (float32+float64, 4 shapes, weights on/off)
registry old: {'mmd': 'mmd_batch_others', 'sinkhorn': 'sinkhorn_batch_others', 'mmd_ref': 'mmd_reference'}
registry new: {'mmd': 'mmd_batch_others', 'sinkhorn': 'sinkhorn_batch_others', 'mmd_ref': 'mmd_reference'}
```

Only `_ot_eps`, `_ot_eps_sym`, `sinkhorn_batch_others` and `sinkhorn_divergence_weighted` differ in source; the five new names (`_cost`, `_marginal_err`, `_not_converged`, `SINKHORN_TOL`, `SINKHORN_MAX_ITER`) are additions. `git diff --name-only c88cce3 0380465 -- src scripts` lists `alignment.py` and `scripts/check_tests.py` only.

### 3.6 Tests

Command (env `wcd-gpu`, CPU, from a copy of the tree without `.git`): `PYTHONPATH=src python -m coverage run --branch --include='*/wcd/alignment.py' -m pytest tests/wcd_vae -q -p no:cacheprovider`. Result: 165 passed, 1 skipped. Coverage of `alignment.py`: 137 statements, 30 branches; unexecuted lines 78, 80, 101, 119, 235 (lines 78 and 80 are weight-validation errors of the MMD weights, 101 and 119 are MMD early returns, 235 is the Sinkhorn `K == 0` return). Every new or changed Sinkhorn line and branch, including the p = 2 branch, the `ValueError`, `_not_converged` and both loop `else` clauses, is executed.

Test margins (`scratch/test_margins.py`, same inputs as the two reference tests): 

| code | test case | value error | x threshold (1e-5) | gradient error | x threshold (1e-3) |
|---|---|---|---|---|---|
| fixed | standardised | 3.90e-08 | 0.00 | 9.43e-05 | 0.09 |
| fixed | narrow_far | 1.07e-06 | 0.11 | 3.27e-05 | 0.03 |
| V6_tol_1e-3 | standardised | 1.36e-07 | 0.01 | 8.30e-04 | 0.83 |
| V6_tol_1e-3 | narrow_far | 2.22e-05 | 2.22 | 2.57e-04 | 0.26 |
| V5_tol_1e-2 | standardised | 8.44e-06 | 0.84 | 7.09e-03 | 7.09 |
| V5_tol_1e-2 | narrow_far | 1.42e-03 | 142.25 | 2.12e-03 | 2.12 |
| V4_float32_solve_direct_differences | standardised | 4.84e-08 | 0.00 | 9.43e-05 | 0.09 |
| V4_float32_solve_direct_differences | narrow_far | 6.31e-07 | 0.06 | 3.27e-05 | 0.03 |
| V1_gradient_fixed_no_convergence_loop | standardised | 1.23e-04 | 12.34 | 1.01e-01 | 100.70 |
| V1_gradient_fixed_no_convergence_loop | narrow_far | 9.51e-04 | 95.08 | 4.16e-03 | 4.16 |
| V3_float32_expansion | standardised | 1.33e-04 | 13.34 | 5.68e-01 | 567.63 |
| V3_float32_expansion | narrow_far | 3.19e-02 | 3188.74 | 3.28e-01 | 327.79 |
| V0_prefix | standardised | 1.01e-05 | 1.01 | 1.45e+00 | 1450.05 |
| V0_prefix | narrow_far | 3.10e-02 | 3096.44 | 1.13e+00 | 1130.32 |

Mutation results: see the table under F4.

### 3.7 Validation re-run

Command (env `scvi-api`, CPU): `python scratch/validate_sample.py 0 1 logs/validate_sample.csv`, which imports the lead's `validate_sinkhorn.py` unmodified from the worktree and calls its own `run()` on 20 fits x 2 minibatches (12 A3 fits one per task-conditioning-lambda group, 6 R1-extension fits spread over tasks and lambda, 2 A1 fits). Result: 40 of 40 rows match the lead's CSV on (kind, tag, minibatch), and each of the columns `value_ref`, `new_value_rel_err`, `new_cosine`, `new_norm_ratio`, the three iteration counts, `tag_cosine`, `tag_norm_ratio`, `tag_value_rel_err` and `ref_grad_norm` is exactly equal in 40 of 40 rows (CPU runs are deterministic).

| stage | zstd | minibatches | fits | fix cosine, min | fix norm ratio, min | fix norm ratio, max | max iterations | pre-fix cosine, median | pre-fix norm ratio, median | float32 variant cosine, min | float32 variant ratio, min | float32 variant ratio, max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 0 | 4 | 2 | 1.0 | 0.999993 | 1.000002 | 23 | 0.527016 | 69.649915 | 1.0 | 0.999993 | 1.000002 |
| A1_ext_r1_round1 | 0 | 12 | 6 | 1.0 | 0.999982 | 1.000012 | 6 | 0.198112 | 20.380805 | 1.0 | 0.999982 | 1.000012 |
| A2A3 | 1 | 24 | 12 | 1.0 | 0.999977 | 1.000015 | 752 | 0.818783 | 2.256 | 1.0 | 0.999977 | 1.000015 |

README audit against the lead's full CSVs (recomputed): worst cosine 0.999999721 (README 0.999999721); norm ratio 0.999937 to 1.000067 (README 0.999937 to 1.000067); worst relative value error 6.7e-05 for reference values above 1e-6 (README 6.7e-5); cap reached in 0 of 3,076 rows (README: never); at most 5,273 iterations on A3 and 110 on A1 (README 5,273 and 110); tolerance sweep maximum 7.5e-04 at 1e-4, 29,190 and 171,388 iterations at 1e-5 and 1e-6 (README 7.5e-4, 29,190, 171,388). All agree. Not verified: the GPU timings (A1 4.0 to 11.7 ms becoming 6.7 to 19.5 ms; A3 12.6 to 12.8 ms becoming 117 to 139 ms), the 10 to 27% figures for testing the convergence every 10 iterations, and the over-relaxation figures: the GPU was reserved for the lead's pilot.

Like with like: both sides receive identical float32 inputs; both use the 1e-12 stabilised Euclidean cost and epsilon 0.05 sqrt(2d); the reference weights are uniform on batch k and on the remainder, as `_batch_masks`; the reference's gradient is the plan-weighted cost derivative with the cost differentiable in both points, which is the envelope gradient once (3.1 confirms it independently). The 12 near-collapsed R1-extension minibatches with reference value about 1e-17 have a zero reference gradient, so their cosine is undefined and their relative value error is meaningless (absolute error below 1e-9); the README's filter on reference values above 1e-6 is stated.

### 3.8 Anything else

1. **Does the arm now implement the stated method?** Yes for the loss function (3.1, 3.2, 3.7). What remains uncertain is operational (F1, F2), documentary (F7) and scope (F6).

2. **Pilot.** The lead's pilot (immune, lambda-matched 'hi', conditioned, seed 100) was still running when I read its step file; I report its steps so far. It times the loss at a median of about 130 ms per call on the RTX 3080, so the Sinkhorn evaluation alone is about two hours for one immune A3 fit.

3. **Rigor ledger.** No `rigor_ledger.json` exists in this project's artifact store; I did not start one without the lead's go.

## 4. Derivation of the gradient

Notation. n cells with latent positions z_i. For a batch k, a is uniform on the cells of batch k and b uniform on the other cells. c(x, y) = |x - y| (the code adds 1e-12 inside the root); epsilon = 0.05 sqrt(2d).

OT_eps(m, v) = max over potentials (f, g) of <m, f> + <v, g> - eps <m x v, exp((f + g - c)/eps) - 1>, and S_eps(a, b) = OT_eps(a, b) - OT_eps(a, a)/2 - OT_eps(b, b)/2. The loss is the mean over k of S_eps.

Envelope theorem. At the maximiser the derivative with respect to the potentials is zero, so the derivative with respect to anything else is the explicit derivative of the objective: d OT_eps / d c_ij = P_ij, where P_ij = m_i v_j exp((f_i + g_j - c_ij)/eps) is the optimal plan. For cell positions: a cell in the first measure receives sum_j P_ij d_1 c(z_i, z_j), a cell in the second measure sum_i P_ij d_2 c(z_i, z_j). In the self term OT_eps(a, a) the same cells are in both measures; with c and P symmetric the two contributions are equal, so the gradient is 2 sum_j P_ij d_1 c(z_i, z_j), and the factor 1/2 in S_eps leaves exactly sum_j P_ij d_1 c.

Code. After the potentials (f, g) are found under `no_grad`, the value is <a, f_fin> + <b, g_fin> with f_fin = softmin_C(b, g) and g_fin = softmin_C(a, f). The derivative of <a, f_fin> with respect to the matrix entry C_ij is the plan entry P_ij of the plan whose rows sum to a, and likewise for <b, g_fin> with a plan whose columns sum to b; the two plans coincide at convergence. In the fixed code C carries gradient through its row point only and g_fin uses `C` (not `C.T`), so <a, f_fin> sends P_ij d_1 c to z_i and <b, g_fin> sends P_ij d_1 c(z_j, z_i) = P_ij d_2 c(z_i, z_j) to z_j: the envelope gradient once. The pre-fix code sent both halves of every entry through both transforms (C differentiable in both points, `C.T`), which is the factor 2 (and, away from convergence, a direction error) the final review found. The tests and numbers in 3.1 confirm each step.

## 5. Limits and declared deviations

1. GPU benchmarks, the GPU pilot and the GPU gradient were not run (the GPU was reserved): all timings are the lead's, or my CPU timings labelled as such.
2. The validation re-run is the requested sample (40 minibatches), not the full 3,076. My other samples are: 10,800 atac_small A3 minibatches, 2,160 immune A3 minibatches (final saved latents of the pre-fix fits, random 128-cell draws from the saved training indices), 4,500 minibatches of lambda = 0 control latents, and the hardest real minibatches for the accuracy check. The cap extrapolation is a model-based estimate from the final latents, not a measurement of training trajectories; treat the expected-event numbers as provisional.
3. Iteration counts in my samples are per problem k solved separately and maximised over k; a batched solve stops when its slowest problem converges, which is the same number as long as each problem's error sequence is non-increasing (it was in all 300 problems checked, 3.2).
4. Only `tests/wcd_vae` and the two Sinkhorn test files (for mutants) were run; the other test directories (`tests/scvi`, `tests/stage`, ...) were not. My test copy has no git history, so one git-dependent test is skipped.
5. Runs stopped, replaced or reduced for time (all on CPU): a cold-start reference solver that was too slow (replaced by a warm-started one); tight solves at tolerance 1e-9 and 1e-6 on the hardest instances, which cannot finish in reasonable time (the final review's converged reference, residual about 1e-9, was used as truth instead); the immune tail sample was reduced from 600 to 120 minibatches per fit (2,160 in all); the scale sweep used 2 batches and 2 repetitions per scale and was stopped after scale 30,000 (scales 100,000 and 1,000,000 not completed). Results quoted come from the completed runs.

## 6. Files (all under `review/sinkhorn_fix_review_20261013/`)

- `REVIEW_SINKHORN_FIX.md` (this report; identical copy in `report/`), `report/fig_iteration_tail.png` (the figure; also saved as an artifact).
- `scratch/`: `myref.py` (independent reference), `check_gradient.py`, `edge_cases.py`, `types_check.py`, `check_mmd_identity.py`, `make_variants.py`, `run_variants.sh`, `test_margins.py`, `validate_sample.py`, `tail_real.py`, `tail_analysis.py`, `hard_accuracy.py`, `narrow_check.py`, `stress_iters.py`, `scale_sweep.py`, `latent_spreads.py`, `err_monotone.py`, `solver_common.py`, `w1_gap.py`, `collect.py`, `make_report.py`, `fill_report.py`, `make_figure.py`.
- `logs/`: every output cited above; `src_reviewed/` holds the two `alignment.py` versions as reviewed; `variants/` the mutants.

## 7. Glossary

- **Sinkhorn divergence S_eps**: a smoothed, bias-corrected transport distance between two point clouds; epsilon controls the smoothing. **W1**: the exact (unsmoothed) Earth Mover's distance with Euclidean cost.
- **Plan / marginals**: the table of how much mass moves between each pair of cells; its row and column sums must equal the cell weights, and the 'marginal error' is how far they are.
- **Potentials (f, g)**: the two vectors the Sinkhorn iterations update; the loss gradient is read off at the converged potentials (envelope theorem).
- **Final epsilon**: epsilon is lowered in steps (warm start); 'iterations at the final epsilon' are the extra updates after the last step.
- **Cap**: the maximum number of such iterations (50,000) before the code raises an error. **Tolerance**: the marginal error at which the iterations stop (1e-4).
- **K**: number of batches present in a minibatch. **A1, A3, R1 extension**: the pilot fits that are being redone: A1 sweeps the penalty weight lambda; A3 repeats chosen lambdas with each latent dimension standardised (`zstd = 1`); the R1 extension adds lambda = 10,000 and 30,000. **R1, R3, R4**: pre-registered rules. R1 freezes the 6-point lambda grid per arm family from A1; R3 decides from A3 whether latent standardisation is switched on for all adversarial arms in the main grid X1 (default: off); R4 picks each arm's 'matched' lambda, the grid point whose batch removal is closest to a common target, together with its lower and upper neighbours ('lo', 'hi').
- **Pre-fix / tag**: the code at tag prereg-tier12-v2 (c88cce3). **I031, I032, I033**: final-review issues (gradient double-counted; final epsilon not converged; inactive-batch comparisons return a graph-free zero). **SI-51, SI-52**: the user's decisions to fix the Sinkhorn arm and to require an independent review of each fix.
