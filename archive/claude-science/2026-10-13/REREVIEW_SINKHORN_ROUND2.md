# Independent re-review of the second Sinkhorn code round (commit 6b4aefc)

Reviewer: Rigor Reviewer, independent of the lead agent. Date: 2026-10-13. I changed nothing in the reviewed trees (no edit, commit or push); every run used a copy of commit 6b4aefc in my own folder. Evidence scripts and outputs are in the attached bundle `rereview_sinkhorn_round2_evidence.tar.gz` (section 8).

## 1. Verdict: PASS_WITH_FIXES

The loss code, the float64 decision, the new tests and the numbers in the notebook, README and code comments are sound: everything I could recompute matches, and I found no wrong result and no broken decision. Two items must change before the new tag. Both concern *one rule*, "stop the Sinkhorn solve on a NaN error but not on an infinite one", and not the shipped behaviour:

1. **R2-01 (fix).** The justification for that rule is incomplete and its strong form is false. The claim that an infinite error with finite values "cannot occur in the cross solve" (the wording of the review request; the notebook argues it conditionally and shows a probe that stops at scale 30,000) is wrong: it occurs for latent coordinates of about 1e18 or more (5 of 5 seeds at each of 1e18, 1e19, 1e20 and 1e30; none at 1e17 or less). The rule itself is right, and in fact required, but the user is being asked to accept it without the real reason.
2. **R2-02 (fix).** No test pins the rule: a variant that also stops on an infinite error passes all 24 tests, although it returns an unconverged answer for such latents (worst-case cosine to the converged gradient 0.93 with 10 batches, 0.99 with 2 batches). One extra test row catches it (section 3, R2-02).

Everything else is a note (7 notes); there is no blocker. Severity labels: blocker = a wrong result or a broken decision; fix = must change before the new tag; note = no action needed for correctness. Open items for this round: 2 fix, 7 note. Still open from the first review and not part of this round: F6 (re-run the validation harness on the refit latents) and the finite-blow-up half of F1 (R2-03).

**Plain-language background (terms used below).**
- *Sinkhorn arm.* The adversarial arm that penalises differences between batches in the latent space (the low-dimensional cell embedding). For each batch it computes a *Sinkhorn divergence*: a smoothed transport distance between the cells of that batch and all other cells; *epsilon* is the smoothing strength (0.22 for 10 latent dimensions). Each divergence needs three solves: the *cross* solve (batch against the rest) and two *debiasing* solves (each side against itself).
- *The solve.* An iterative loop that improves two vectors (the *potentials*) until the implied transport plan has the right row and column totals. The *marginal error* is how far those totals are from their targets. The loop stops when the error is below the *tolerance* (1e-4) or, as an error, at the *cap* on iterations (now 500,000).
- *NaN, inf, float32/float64.* NaN is not-a-number (what arithmetic on a NaN or infinite input yields); inf is infinity; float32 and float64 are 32- and 64-bit decimal arithmetic. GPU is the graphics processor (an RTX 3080). *Scale s* below means the standard deviation of each latent coordinate in a random test latent.
- *Fit groups and labels.* A1 = pilot fits on raw latents over a grid of penalty weights lambda; A3 = pilot fits with each latent dimension standardised; R1 extension = fits with a very strong penalty (lambda 10,000 and 30,000), whose latents collapse to a tiny spread. "Pre-fix" is the code at tag prereg-tier12-v2 (commit c88cce3). "The first review" is the earlier review of commit 0380465 (`REVIEW_SINKHORN_FIX.md`); its findings are called F1 to F8.
- *Other terms.* MMD (maximum mean discrepancy) is the other critic-free alignment loss in the same file; W1 is the exact, unsmoothed transport distance (earth mover's distance), computed here with the POT (Python Optimal Transport) library; SD is standard deviation; a *mutation check* means that I break the code on purpose in a small way and see whether a test notices.
- *Decisions and notes.* SI-58 and SI-59 are the user's decisions of 2026-10-13 on the solve precision and the iteration cap (full text in section 4); SI-52 is the user's rule that fits of the non-Sinkhorn arms stay bit-identical to the tag. "NB-20261013-17" and similar are entries of the project notebook.

## 2. Answers to the seven claims

| # | Claim | Status | One-line evidence |
|---|---|---|---|
| 1a | NaN, +inf and -inf latents give a NaN marginal error at the first iteration of each solve | confirmed | 11 non-finite cases on CPU and GPU: 3 error evaluations, all NaN, value NaN in 5 to 15 ms; existing probe output reproduced exactly |
| 1b | An infinite error with finite potentials cannot occur in the cross solve (wording of the review request; NB-17 argues it conditionally) | **wrong** | occurs for finite latents of scale 1e18 or more: 5 of 5 seeds at each of 1e18, 1e19, 1e20, 1e30 (2 and 10 batches); none at 1e17 or less (R2-01) |
| 1c | If it occurred in a debiasing solve it must keep iterating rather than return a finite unconverged value | confirmed as a design rule, and in fact required | the stop-on-any-non-finite variant returns unconverged finite gradients (worst-case cosine 0.93 with 10 batches, 0.99 with 2) |
| 1d | The new test fails on the old code and on either one-loop mutant | confirmed | all three mutants fail both test rows (NaN and inf) |
| 1e | Meets SI-59's intent; no path returns a finite unconverged value or an unrecorded NaN | confirmed, with the wording caveat in R2-01 | the only loop exits are: error below tolerance, NaN error (then the loss is non-finite), or the cap error; end to end the training plan raises `NonFiniteLossError` ('diverged') in 0.3 s |
| 2 | Cap 500,000 and the comment's numbers (7.5e-4; max 5,273; 9,789; about 50x) | confirmed (wording note R2-08) | 7.49e-4 over all 3,076 minibatches; 5,273; 9,789 (first review's 10,800-minibatch sample); 500,000 / 9,789 = 51.1 |
| 3a | `SINKHORN_DTYPE` is the only new switch | confirmed | `git diff 82263e5 6b4aefc -- src` touches one file; diff read in full |
| 3b | Float64 results bit-identical to committed `validation_12mb.csv` | confirmed | my full re-run at 6b4aefc: 23 shared columns, 3,076 rows, no difference (also identical to the committed float64 and float32 SI-58 files, 24 columns) |
| 3c | Float32 meets the SI-58 criterion on the 598 hardest minibatches | confirmed | 1 - worst cosine 2.788e-07; worst relative value error 2.974e-06; no cap hit |
| 3d | ... but not on all 3,076 | confirmed | worst relative value error 3.168e-3 (collapsed R1-extension latents), 2.69e-4 (A1) |
| 3e | No GPU speed-up | confirmed for 128-cell minibatches only (R2-05) | my re-run: median float32/float64 time ratio 0.970 to 1.016 over the five regimes (lead: 0.989 to 1.014) |
| 3f | The 598 selection equals the one in `tol_sweep.py`; value-error filter matches the float64 README | confirmed | same rule on the same file gives 598; identical to the 598 minibatches in `tol_sweep.csv`; the filter excludes 12 rows, none among the 598 |
| 3g | `si58_summary.py` computes the criterion correctly; bench free of timing confounds | confirmed; the bench is valid for what it measures | my independent recomputation matches; the script regenerates the committed `si58_summary.md` exactly |
| 3h | Keeping float64 follows SI-58 | confirmed, with a wording note (R2-06) | SI-58 says "adopt only if"; float32 had no benefit, and fails the full-set check |
| 4 | Test threshold 3e-4 fails tolerance 1e-3 on both cases; dtype spy catches float32 and the bypass | confirmed (narrow-far fails through the value threshold) | mutation table in R2-02; tolerance 2e-4 still passes |
| 5 | F7/F8 docstrings, README, notebook NB-17 to NB-21 accurate and consistent | confirmed; NB-17 and the README probe row are accurate as scoped but incomplete (R2-01) | F7: divergence is 0.20 to 0.89 of exact W1 on 48 real minibatches; F8: gradient 1 vs 2 reproduced |
| 6 | Nothing else changed; tests and `check_tests.py` pass | confirmed | one source file and one test file changed; MMD region source-identical; `tests/wcd_vae`: 170 passed, 1 skipped; `check_tests.py`: OK |
| 7 | Anything missed or broken | 2 fix and 7 notes | section 3 |

## 3. Findings (severity ordered)

### R2-01 (fix). An infinite marginal error does occur with finite values; the evidence for the NaN-only rule is incomplete and the claim "cannot occur" is wrong

**Where.** Notebook entry NB-20261013-17 (the decision on first-review finding F1: its second rationale bullet argues that an infinite error with finite potentials "would mean an unconverged but finite solve", its first cites the probe result "0 infinite errors for latents up to s = 30,000"); the README row for `f1_nonfinite_probe.py` (same scope); and the review request, which states that such an error "cannot occur in the cross solve". The module docstring sentence on the NaN stop (`alignment.py`, lines 39 to 40) is accurate but silent on the infinite-error case.

**Problem.** The loops now end on `not err >= tol`, that is when the error is below the tolerance or NaN. They do not end on +infinity. NB-17 justifies this conditionally: stopping on an infinite error would return a finite unconverged solve. The conditional is correct; the situation it guards against is not hypothetical, and the stronger statement "cannot occur" is false. For finite latents with very large coordinates the cross solve's first error is +infinity (an overflow in `exp`), while every potential is finite. The solver correctly keeps iterating and converges after 2 to 8 more iterations. If the loop had also stopped on an infinite error (the wording of the first review and the literal wording of SI-59, "a non-finite marginal error ends the loop at once"), it would return a state that its own convergence test calls unconverged. The probe behind the premise cannot see this: it caps each solve at 3,000 iterations, uses one seed and 2 batches, and stops at scale 30,000.

**Evidence.**
- `rr/probes/inf_regime.py` (output `rr/out/inf_regime.txt`): 128 cells x 10 dimensions, coordinates N(0, s^2) rounded to float32, 5 seeds per row. First cross-solve error is +infinity in 5 of 5 seeds at s = 1e18, 1e19, 1e20, 1e30, for 2 and for 10 batches; in 0 of 5 at s = 1e10 to 1e17. The branch converges in all of them. Replacing the stop rule by "stop on any non-finite error" leaves the value unchanged but changes the gradient: cosine to the branch's converged gradient 0.932 to 0.948 (10 batches) and 0.989 to 1.000 (2 batches) at s = 1e18 to 1e30.
- `rr/probes/f1_probes.py C` (output `rr/out/f1_probe_C_cpu.txt`): at s = 1e20, 1e30, 3e37 the first error of the cross solve is infinite in 12 of 12 runs, with no non-finite potential (`nonfin_softmin=0`) and 2 to 8 iterations to converge.
- `rr/probes/f1_probes.py A` (outputs `rr/out/f1_probe_A_cpu.txt` and `rr/out/f1_probe_A_cuda.txt`): one coordinate equal to the largest float32 value (3.4e38) in an otherwise standard-normal minibatch: the first cross-solve error is infinite on CPU and on the GPU, while the loss (5.3e36) and the gradients are finite.
- Why. In exact arithmetic the cross-solve error compares two probability vectors and is at most 2; on all 3,076 real minibatches it was at most 0.562 (`rr/probes/err_bound_real.py`, `rr/out/err_bound_real.csv`: no non-finite error and no increase between iterations in any of the three solves). In float64 the potentials of a latent of scale 1e18 are of order 1e18 to 1e19, where the spacing between representable numbers (about 2e-16 times the value) already exceeds 709 x epsilon (about 160 for this blur); a rounding difference of that size overflows `exp`. The onset between 1e17 and 1e18 fits this explanation; I did not test the mechanism separately.
- The committed probe reproduces exactly (`rr/out/f1_nonfinite_probe_rerun.txt` equals `results/f1_nonfinite_probe.txt`); its limits are those named above.

**Consequence.** No wrong result today. The risks are a later "simplification" to the finite-check wording (it passes every test, R2-02) and a user decision on the NaN-only reading taken without the real reason. Latents of this size are far from anything seen: the 234 pilot fits' final latents have a mean per-dimension standard deviation between 3.3e-06 and 1.02 (first review, `latent_spreads.csv`, re-read here).

**Suggested fix.** (a) Add a correction entry to NB-17 (and amend the README row and the comment at the two loops): "an infinite error occurs for finite latents of about 1e18 or more and must not stop the loop; only NaN does". Do not describe the case as impossible in the diff summary to the user, and tell the user that the literal SI-59 wording would be wrong for +infinity. (b) Extend `f1_nonfinite_probe.py` to s = 1e17, 1e18, 1e20, 1e30, 3e37 (it already counts `inf_with_finite_potentials`) and regenerate `results/f1_nonfinite_probe.txt`.

### R2-02 (fix). Nothing pins the NaN-only rule, and the cap constant is unpinned

**Where.** `tests/wcd_vae/test_divergences_v2.py`.

**Problem.** I ran the 24 tests of that file against the unmodified file and 12 variants of `alignment.py` that I wrote (text substitutions, my copy only). The new F1 test and the F4 changes do what the round says. Two variants that matter survive.

| variant | tests failing (of 24) | reading |
|---|---|---|
| `BASE_unmodified` | 0 (none) | committed code: all pass |
| `F1_old_both` | 2 (nonfinite_latent_stops_at_once_with_nan[inf], nonfinite_latent_stops_at_once_with_nan[nan]) | pre-round loops (`err < tol`) in both solves: the new F1 test catches it |
| `F1_old_cross_only` | 2 (nonfinite_latent_stops_at_once_with_nan[inf], nonfinite_latent_stops_at_once_with_nan[nan]) | only the cross solve reverted: caught |
| `F1_old_sym_only` | 2 (nonfinite_latent_stops_at_once_with_nan[inf], nonfinite_latent_stops_at_once_with_nan[nan]) | only the two debiasing solves reverted: caught |
| `F1_stop_on_nonfinite` | 0 (none) | **stop on any non-finite error (the first review's wording): NOT caught (R2-02)** |
| `F4_tol_1e-3` | 2 (matches_converged_reference[_narrow_far], matches_converged_reference[_standardised]) | tolerance 1e-3: caught in both cases |
| `F4_tol_5e-4` | 1 (matches_converged_reference[_standardised]) | tolerance 5e-4: caught (standardised case only) |
| `F4_tol_3e-4` | 1 (matches_converged_reference[_standardised]) | tolerance 3e-4: caught (standardised case, 1.02x its threshold) |
| `F4_tol_2e-4` | 0 (none) | tolerance 2e-4: passes (a twofold loosening is not caught) |
| `F4_solve_float32` | 2 (matches_converged_reference[_narrow_far], matches_converged_reference[_standardised]) | float32 solve: caught by the dtype check |
| `F4_bypass_dtype_switch` | 2 (matches_converged_reference[_narrow_far], matches_converged_reference[_standardised]) | `zd = z` (bypass of `SINKHORN_DTYPE`): caught |
| `F4_cost_float32_then_cast` | 0 (none) | cost computed in float32, solve in float64: passes (numerically equivalent at the thresholds; not a defect) |
| `SI59_cap_back_to_50000` | 0 (none) | **cap reverted to 50,000: NOT caught (R2-04)** |

**Evidence.** `rr/probes/mutate.py`, output `rr/out/mutation.log` and `rr/out/mutation_results.json`. Margins of the reference test at the committed tolerance (`rr/out/test_margins.txt`): standardised case gradient error 9.43e-05 against the threshold 3e-4 (0.31x), value error 3.9e-08 against 1e-5; narrow-far case gradient 3.27e-05 (0.11x), value 1.07e-06 (0.11x). Tolerance 3e-4 fails the standardised case by a hair (1.02x); tolerance 1e-3 fails both cases, the narrow-far one through the value threshold (2.22x), not the gradient threshold (0.86x). So the claim holds, with that precision.

**Suggested fix.** Add one row to `test_sinkhorn_refuses_bad_p_and_unconverged_solve`: a finite latent `1e30 * torch.randn(8, 2, generator=Generator().manual_seed(0), dtype=torch.float64)` with two batches, called as `sinkhorn_divergence_weighted(_cost(z, 1), A, B, 0.1, max_iter=1)`, expecting `RuntimeError` "after 1 iterations". I ran it against three variants (`rr/probes/propose_test.py`): the branch raises (correct: an infinite first error does not stop the loop), the old code raises, the stop-on-any-non-finite variant returns 1.482e30 silently. It catches what no existing test catches, so it respects the fewest-tests rule. Optional: one assertion for `SINKHORN_MAX_ITER == 500000` (SI-59) or accept that the ledger entry is the only record. Optional: an end-to-end test that a poisoned latent reaches `NonFiniteLossError` (`rr/probes/e2e_guard.py` is a template).

### R2-03 (note). A finite blow-up of the latent still ends in a cap error, after about three minutes

**Where.** `SINKHORN_MAX_ITER = 500000` (SI-59) and the first review's F1 item 3 (finite blow-up).

**Problem.** The round fixes the NaN and infinite-latent path. The path where the latent grows through large finite values before turning NaN is unchanged in kind: in my GPU runs a latent with coordinates of scale 1e5 up to 1e14 reaches the cap of 500,000 iterations after 173 to 181 s (6 of the 6 scales tried in that range) and raises `RuntimeError`, while scales of 1e4 or less converge (1e4: 82,360 error evaluations, 30 s). `scripts/run_stage.py` (lines 12 to 17) treats anything other than the 'diverged' exit code as an infrastructure error: retry up to the attempt limit, then the stage stops. SI-59 ("keep raising an error at the cap; the stage stops loudly") accepts this; the observed latents (SD at most 1.02) are 5 orders of magnitude below the scale where it happens, so I rate it a note. It does mean F1 is closed for NaN and infinity, not for finite blow-up.

**Evidence.** `rr/probes/scale_iters_gpu.py` (GPU, committed code and cap, 128 cells x 10 dimensions, 2 batches, seed 0; outputs `rr/out/scale_iters_gpu.txt`, `rr/out/scale_iters_gpu2.txt`):

```
s=   1e+02:    4694 marginal-error evaluations (all three solves) in     2.0 s; value 276.6
s=   1e+03:   25036 marginal-error evaluations (all three solves) in     9.1 s; value 2766
s=   1e+04:   82360 marginal-error evaluations (all three solves) in    30.3 s; value 2.766e+04
s=   1e+05:  500000 marginal-error evaluations (all three solves) in   177.0 s; CAP HIT: Sinkhorn: L1 marginal error 0.0313 > tol 0.0001 after 500000
s=   1e+06:  500000 marginal-error evaluations (all three solves) in   173.3 s; CAP HIT: Sinkhorn: L1 marginal error 0.0625 > tol 0.0001 after 500000
s=   1e+08:  500000 marginal-error evaluations (all three solves) in   180.9 s; CAP HIT: Sinkhorn: L1 marginal error 0.0625 > tol 0.0001 after 500000
s=   1e+10:  500000 marginal-error evaluations (all three solves) in   179.1 s; CAP HIT: Sinkhorn: L1 marginal error 0.0625 > tol 0.0001 after 500000
s=   1e+12:  500000 marginal-error evaluations (all three solves) in   177.1 s; CAP HIT: Sinkhorn: L1 marginal error 0.0625 > tol 0.0001 after 500000
s=   1e+14:  500000 marginal-error evaluations (all three solves) in   176.2 s; CAP HIT: Sinkhorn: L1 marginal error 0.0633 > tol 0.0001 after 500000
```
A cap hit with an A3-like latent (standard normal, 3 batches, tolerance 0 so that it can never converge) gives the last line of `scale_iters_gpu2.txt`:

```
cap hit (tol=0, K=3, n=128, d=10, GPU): 500000 iterations in 178.8 s = 358 us/iter; error: Sinkhorn: L1 marginal error 0 > tol 0 after 500000 iterations at the final eps
```
A CPU scan with a 3,000-iteration cap (`rr/out/f1_probe_C_cpu.txt`) shows the solve converging again from s = 1e16.

**Suggested fix.** None needed to meet SI-59. If the user wants finite blow-up to count as 'diverged', it needs a decision on the pre-registered definition of the outcome (for example a bound on the latent scale); I did not test any such guard.

### R2-04 (note). The cap constant is not pinned

Reverting `SINKHORN_MAX_ITER` to 50,000 passes all 24 tests (variant `SI59_cap_back_to_50000`). The decision lives only in the project's ledger of user decisions (`CONSTRAINTS.md`), the README and the notebook. See R2-02 for the optional assertion.

### R2-05 (note). "No GPU speed-up" is true for 128-cell minibatches only; the comment and README do not say so

**Evidence.** `rr/probes/iter_microbench.py` (output `rr/out/iter_microbench.txt`), time per solver iteration on the RTX 3080 (two potential updates, the error, and the wait for the GPU result that the stopping test needs):

```
n   K   float32_us  float64_us  f32/f64
 128   3     357.3      356.3    1.003
 128  10     358.5      355.2    1.009
 512  10     451.8     1170.9    0.386
1024  10    1576.6     4259.4    0.370
2048  10    5914.7    16441.7    0.360
```
At the production size (128 cells) both dtypes cost the same, which matches the lead's explanation (the fixed cost of launching many small GPU operations per iteration, and the wait for the stopping test, dominate). From 512 cells float32 is 2.6 to 2.8 times faster per iteration, so the benchmark can detect a speed-up; it simply finds none at 128. All Sinkhorn rows of the manifests have `batch_size` 128 (714 rows in `paper_manifest_stock_pilot_u5_b10_v3.x1.tsv` and the other manifests; `rr/out/manifest_batch_sizes.txt`), so the decision holds for every planned fit.

**Suggested fix.** Add "(128-cell minibatches; from 512 cells float32 is about 2.7x faster per iteration)" to the `SINKHORN_DTYPE` comment and to the README line.

### R2-06 (note). The recorded reason for keeping float64 paraphrases SI-58 too loosely

NB-20261013-20 says "SI-58 says keep float64 unless float32 matches". Float32 did match on the user's own test set (the 598 hardest). SI-58 says adopt "only if" it matches; it does not say adopt when it matches. Keeping float64 is therefore consistent with SI-58, but it rests on the agent's reasons, which the README states correctly: no speed-up at 128 cells, the failure on the full set, and the need for a guard on an infinite diameter (R2-07). One more fact belongs with them: the 598 contain no R1-extension minibatch (their regimes: 432 standardised A3, 58 A1, 87 early captures, 21 training-mode), so the user's own criterion could not have seen the collapsed-latent value error (3.17e-3). The round's superset check is what exposed it. The `SINKHORN_DTYPE` switch is still used by two experiment scripts (SI-58 evidence); if the user prefers the smallest code, remove it together with the `solve_dtype` column in the commit that archives them.

### R2-07 (note). An infinite cost diameter makes the epsilon schedule loop forever (unreachable in production)

**Evidence.** `rr/probes/f1_probes.py D` (output `rr/out/f1_probe_D.txt`). Float32 latents (largest value 3.4e38, 256 dimensions) in the float64 solve give a finite diameter 8.4e39 and 413 epsilon steps: the case the round calls unreachable is unreachable. But a float64 latent of 1e200, or a float32 solve (NB-20261013-20) of a float32 latent of 1e20, never returns: both were killed by an 8-second alarm. `_eps_schedule` multiplies an infinite value by 0.8 forever and appends to a list.
**Suggested fix.** None now. If the dtype switch is ever set to float32, add `if not math.isfinite(diameter)` before `_eps_schedule` (return NaN).

### R2-08 (note). One comment is ambiguous about two sample sizes

The `alignment.py` comment says "max 5,273 there, 9,789 on minibatches of the pre-fix A3 latents". Both maxima come from minibatches of the same pre-fix A3 latents: 5,273 is the maximum over the 480 standardised validation minibatches (36 A3 fits x 12, plus 48 early and training-mode captures), 9,789 over 10,800 atac_small minibatches drawn by the first review (immune: 2,160 minibatches, maximum 8,051; `logs/tail_real_A3_part*.csv`). Say so, or give the sample sizes.

### R2-09 (note). The branch moved after the reviewed commit

`git log 6b4aefc..HEAD` in the main checkout shows 7 more commits (guard follow-ups, up to 43a0f91) that change `alignment.py` (a `stats` argument and a helper `_no_pair` for the no-pair return of both MMD functions and the Sinkhorn function), `scvi_adversarial_plan.py`, `test_divergences_v2.py` and others. I did not review them. I did read the `alignment.py` diff: the two convergence loops, the `SINKHORN_*` constants and `_ot_eps*` are untouched, so R2-01 to R2-07 apply to the newest head as well. The MMD function signatures changed, so the SI-52 identity check (non-Sinkhorn arms bit-identical to the tag) should be repeated at the final commit.

## 4. Evidence per claim

### Claim 1: the F1 stop rule (sections R2-01, R2-02 for the details)

- **End to end through the real training plan** (`rr/probes/e2e_guard.py`, scvi-tools 1.4.2, CPU, toy data of 600 cells, adversary `sinkhorn`, one epoch; output `rr/out/e2e_guard.txt`): one entry of the adversary input set to NaN, +inf or -inf at step 0 gives `NonFiniteLossError` (the 'diverged' outcome) at epoch 0, step 0 in 0.3 s, with `loss_vae` finite (120.69), `adv_term` NaN, `gen_loss` NaN. With the old loop (`err < tol`, cap lowered to 300 only to save time) the same poison ends as `RuntimeError: ... after 300 iterations` after 0.5 to 0.6 s; at the committed cap that is 500,000 iterations per solve, an 'infrastructure' error. The existing guard tests (`tests/scvi/test_nonfinite_guard.py`, 13 tests, which include the runner writing status 'diverged') pass at 6b4aefc.
- **Direct calls** (`f1_probes.py A`, CPU and GPU): NaN, +inf, -inf in one coordinate; in a one-cell batch; in two batches; a whole row NaN; all NaN; all +inf; NaN and +inf after the per-dimension standardisation of the A3 arms. Every case: 3 error evaluations (one per solve), all NaN, NaN value in 11 to 15 ms (CPU) or 5 to 7 ms (GPU), non-finite gradient, finiteness test of the training guard false (flagged). Finite control: 27 evaluations, 172 ms (CPU).
- **Exits of the loop.** (i) error below tolerance; (ii) NaN error: arises only when a potential or the cost is non-finite (`inf - inf`), and then the loss is non-finite and the guard flags it; (iii) the cap error. There is no exit that returns a finite value without the convergence test being met. At scales of 1e18 and more the convergence test is met by potentials that have lost most of their precision; I could not check those against an independent reference (the reference solver itself fails there with a NaN residual, `rr/probes/inf_vs_reference.py`).
- **Mutation** (R2-02): both loops reverted, only the cross loop reverted, only the debiasing loop reverted: each fails both rows of the new test (NaN and inf). The new test uses `max_iter=1`, so it also pins "at once".

### Claim 2: the cap and the comment's numbers

`alignment.py` line 56 reads 500000. Comment numbers: gradient within 7.5e-4: worst relative gradient error 7.49e-4 over all 3,076 minibatches and over the 598 hardest (my re-run; `tol_sweep.csv` at tolerance 1e-4: 7.4898e-4). 5,273: maximum iterations over standardised validation minibatches (my re-run; 290 on the unstandardised ones). 9,789: maximum over the first review's 10,800 atac_small A3 minibatches (re-read from its CSV files). 500,000 / 9,789 = 51.1; 50,000 / 9,789 = 5.1, which matches the correction NB-20261013-21 ("about 5x"). Cost of a cap hit on the GPU: 173 to 181 s per 500,000 iterations (0.35 ms per iteration), consistent with NB-20261013-15 ("about 3.5 min").

### Claim 3: SI-58

SI-58 as written: "validate a float32 solve on the 598 hardest minibatches and the GPU timing; adopt it only if the worst gradient cosine and value error stay within the float64 results (cosine >= 0.99999, value error <= 1e-4), ... otherwise keep float64".

My full re-runs (CPU, 6 workers per dtype, `SINKHORN_DTYPE=float32|float64 python validate_sinkhorn.py OUT.csv 12 6`, about 3 min): both outputs equal the committed SI-58 files in all 24 shared columns of 3,076 rows, and the float64 output equals the committed `validation_12mb.csv` in the 23 columns that file has. Numbers recomputed by my own code (`rr/out/si58_recomputed_from_my_rerun.csv`):

| subset | solve | minibatches | 1 - worst cosine | worst relative gradient error | worst relative value error (reference above 1e-6) | cap hits | iterations median / max |
|---|---|---|---|---|---|---|---|
| 598 hardest | float64 | 598 | 2.789e-07 | 7.490e-04 | 3.079e-06 | 0 | 154 / 5273 |
| 598 hardest | float32 | 598 | 2.788e-07 | 7.489e-04 | 2.974e-06 | 0 | 154 / 5273 |
| all 3,076 | float64 | 3076 | 2.789e-07 | 7.490e-04 | 6.667e-05 | 0 | 7 / 5273 |
| all 3,076 | float32 | 3076 | 2.788e-07 | 7.489e-04 | 3.168e-03 | 0 | 7 / 5273 |

Float32 meets both thresholds on the 598 (cosine 0.9999997 against 0.99999; value error 2.97e-06 against 1e-4) and fails the value threshold on the full 3,076 set (3.17e-3; the worst rows are the R1-extension fits at lambda 10,000 to 30,000, reference values of 1e-5 to 7e-5; the 12 rows with reference value at most 1e-6 are excluded from the relative measure, none is among the 598, their absolute float32 error is at most 8.9e-9). There is no NaN cosine or value among the 598, so the minimum and maximum are not hiding missing values. The selection: the rule of `tol_sweep.py` applied to `validation_12mb_tol1e-6_cap1e4.csv` gives 598 rows (480 standardised, others with more than 100 iterations or a cap error) and the same 598 minibatches appear in `tol_sweep.csv` (2,392 rows = 598 x 4 tolerances). The criterion and the pre-launch checklist record (`PREFLIGHT.md`) were written before the results (commit 01b354f at 18:21:16 before the results commit 81f8ce4 at 18:31:22), and the code is the same as at 6b4aefc apart from one comment (`git diff 772d59b 6b4aefc -- src`).

GPU timing (`bench_gpu.py`, alone on the GPU: 124 MiB used and 0% before launch, 22:36:08 to 22:41:42 UTC; `rr/out/bench_gpu_rereview.csv`). Median over 12 minibatches of the per-minibatch ratio float32 time / float64 time, with a bootstrap 95% interval:

| regime | tagged code ms | branch float64 ms | branch float32 ms | ratio, median | 95% interval | lead's ratio |
|---|---|---|---|---|---|---|
| A1_wide_l0.1 | 12.7 | 22.1 | 21.7 | 0.997 | 0.972 to 1.019 | 0.989 |
| A1_l10 | 8.4 | 12.9 | 12.9 | 1.016 | 0.998 to 1.030 | 1.003 |
| A1_shrunk_l3000 | 3.8 | 7.1 | 6.9 | 0.970 | 0.949 to 1.020 | 0.992 |
| A3_zstd_l30 | 14.9 | 141.3 | 137.3 | 0.998 | 0.966 to 1.005 | 1.014 |
| A3_zstd_l3_immune | 13.8 | 159.1 | 167.9 | 0.973 | 0.930 to 1.014 | 1.006 |

The machine was busier than during the lead's run (load average about 5 at launch), which widens the spread (ratios from 0.78 to 1.54 for single minibatches); absolute milliseconds are therefore 6 to 20% higher than the lead's and should not be compared, the ratios agree. Design of the bench: 3 warm-up calls then 20 timed calls with `cuda.synchronize`, median; the order of the two dtypes alternates per minibatch (order effect on the ratio at most 4% per regime); the tagged code is always timed first, which affects only the tag-versus-branch comparison, not SI-58. Together with the micro-benchmark (R2-05) the finding "no speed-up at 128 cells" is robust. Decision: see R2-06.

### Claim 4: F4

Mutation table in R2-02. A dtype spy (`dtypes == [torch.float64]` on the cost matrix handed to the cross solve) fails a float32 solve and the bypass `zd = z` (2 failures each). Variants that still pass all 24 tests: tolerance 2e-4; stop on any non-finite error; cap back to 50,000; cost computed in float32 and then cast to float64 (numerically equivalent at the test thresholds, not a defect). The commit message "tol 1e-3 fails both cases" is correct; for the narrow-far case it is the value threshold that fails.

### Claim 5: docstrings, README, notebook

- **F7** (module docstring: the divergence tends to W1 only as the blur goes to 0 and is well below W1 at this blur): consistent with measurements. Exact W1 by network simplex (POT) against the branch's divergence at the committed blur, mean over batches of each minibatch (`rr/probes/w1_ratio.py`, 24 raw-latent and 24 standardised real minibatches from 16 randomly chosen fits): ratio 0.20 to 0.89 (median 0.31) for raw latents, 0.62 to 0.87 (median 0.77) for standardised ones. The first review's range of 0.53 to 0.89 lies inside my pooled range of 0.20 to 0.89.
- **F8** (docstring: a cost differentiable in both points double-counts the gradient; p = 2 not used and tested on two Diracs only): confirmed. Two Diracs at 0 and 2: gradient (-1, +1) with the committed cost, (-2, +2) with a cost differentiable in both points (`rr/probes/f8_check.py`); `p = 2` appears only in `test_sinkhorn_two_diracs_exact`.
- **README**: all SI-58 numbers reproduce (section above); the cap paragraph and the F1 paragraph are accurate. The README row for the F1 probe says latents up to 30,000 never give an infinite error: true as scoped, but silent on larger finite latents (R2-01).
- **Notebook**: NB-17: decision text accurate; its evidence is incomplete (R2-01). NB-18 (plan), NB-19 (result: all numbers reproduce), NB-20 (decision; wording R2-06), NB-21 (correction of NB-04: 50,000 / 9,789 = 5.1 correct). The index lists all five entries. `f1_nonfinite_probe.py` re-run: output identical to the committed file.

### Claim 6: nothing else changed (SI-52)

- `git diff 82263e5 6b4aefc --stat`: in `src/` and `scripts/` only `alignment.py`; in `tests/` only `test_divergences_v2.py`.
- The region from `_sqdist` to `_log_w` (both MMD functions, mask helpers) and the registry are textually identical between 82263e5 and 6b4aefc. Against the tag c88cce3 the MMD functions differ only in the no-pair return `(z * 0).sum()` (final-review issue I033: a minibatch with a single batch returned a zero without a gradient graph; fixed in an earlier round). On 48 random inputs (float32 and float64, 4 shapes, with and without weights, 3 seeds) value and gradient of `mmd_batch_others`, `mmd_reference` and `_sqdist` are bit-identical to the tag (`rr/out/si52_identity_cpu.txt`).
- `tests/wcd_vae` (env wcd-gpu, CPU): 170 passed, 1 skipped (a critic test that needs an old git revision), 116.8 s. `scripts/check_tests.py --base 82263e5 --src src -- tests/wcd_vae/test_divergences_v2.py` with `PYTHONPATH=src` (venv with pytest-cov 7.1.0): OK, 24 passed ("every changed --src line and branch is tested; no redundant new tests").

## 5. Status of the first review's findings

| First review | Status after this round |
|---|---|
| F1 (non-finite latent spins to the cap) | NaN and infinite latents: resolved and verified end to end. Finite blow-up: unchanged by design (R2-03). The evidence in the notes is incomplete (R2-01); a test row is missing (R2-02) |
| F2 (cap not shown safe) | resolved: 500,000 (SI-59), correction NB-21; cap hit costs about 3 min |
| F3 (no-pair return had no graph) | resolved earlier (final-review issue I033): `test_no_pair_of_batches_gives_differentiable_zero` passes for the Sinkhorn function |
| F4 (tests) | resolved with the residuals in R2-02 |
| F5 (float32 solve) | resolved: measured, not adopted (R2-05, R2-06) |
| F6 (validation scope: pre-fix latents, 3 tasks) | still open, not claimed: re-run the harness on the refit latents |
| F7, F8 (docstrings) | resolved |

## 6. What the round does well (do not change)

Pre-declared selection and thresholds before any float32 result; a declared superset check that exposed the collapsed-latent value error; a bit-identical float64 regression check; a tolerance threshold that is now tight (2e-4 passes, 3e-4 fails); explicit corrections (NB-21); the cap decision stated with its cost.

## 7. Limits of my checks (declared)

See the `deviations` list of the structured output; in short: this review stops at commit 6b4aefc; scale scans use one seed and 2 batches on the GPU; at latent scales of 1e18 and more no independent reference exists; mutation tests used only the file `test_divergences_v2.py`; W1 ratios and the end-to-end test are small samples; no rigor ledger exists in the project and none was created without the lead's go; the `ponytail-review` skill is not in this profile's catalog. Prior-work search: not run, because the round introduces no new method and makes no novelty claim; none of my recommendations depends on literature.

## 8. Files and how to re-run

The bundle `rereview_sinkhorn_round2_evidence.tar.gz` holds `probes/` (scripts) and `out/` (outputs). Paths in the text are relative to its root (`rr/probes`, `rr/out`). Re-run from a copy of commit 6b4aefc made with `git archive 6b4aefc`; commands: `python rr/probes/mutate.py` (env wcd-gpu with pytest-cov), `python rr/probes/inf_regime.py`, `python rr/probes/f1_probes.py <alignment.py> {A|C|D} [cpu|cuda]`, `python rr/probes/err_bound_real.py OUT.csv 12`, `python rr/probes/e2e_guard.py <WCD_SRC> {none|nan|inf|-inf}` (env scvi-api, CPU), `python rr/probes/scale_iters_gpu.py 1e5,1e6` and `... cap` (GPU), `python rr/probes/iter_microbench.py <alignment.py>` (GPU). Environments: `scvi-api` (torch 2.13.0+cu130, scvi-tools 1.4.2, POT 0.9.7.post1) and `wcd-gpu` (torch 2.4.1+cu121). Prefix `KMP_AFFINITY=disabled OMP_NUM_THREADS=1`.
