# Independent review of six guard fixes (final-review issues I033, I034, I042, I050+I051, I094, I008)

Reviewer: Rigor Reviewer (independent sub-agent; read-only on the repository). Date: 2026-10-13.

**Reviewed:** the six commits `c4be14b`, `a4f4be0`, `5f30f17`, `f5ea0c3`, `46d79f2`, `2b52542` on branch `final-review-fixes` (head `2b52542`, the read-only worktree `code/wt_review_guards`). **Pre-fix reference:** tag `prereg-tier12-v2` = `c88cce3`. Every command below ran in clones of the repository under `review/guard_fixes_review_20261013/trees/` (the worktree itself was not modified); CPU only, 1 thread, `nice -n 19`, at most 3 processes at a time.

**Requirement tested (user decision SI-52):** fits and scores that do not trigger a guard must stay bit-identical to c88cce3, and a guard must fail loudly and never silently change a result.

**Bottom line.** None of the six fixes has a blocker. Each does what its review row asked, refuses nothing that is stored or planned (0 of 40,815 manifest rows, 0 of 1,701 stored latents), and leaves every fit and score that does not trigger it unchanged in the tests I could run (CPU toy fits; 2 real scores; the Sinkhorn arm is excluded by design). Three small additions are recommended before the new tag: a trace of skipped steps (G-01), a stronger test of the inner-step check (G-02) and a test of the production call that applies the I042 refusals (G-03).

## 1. Verdicts

| Fix | Commit | What it answers (plain language) | Verdict | Why |
|---|---|---|---|---|
| I033 | `c4be14b` | A training step whose minibatch holds no pair of batches (for example only one batch) crashed the MMD, reference-MMD and Sinkhorn arms. | **PASS WITH FIXES** | Fixes the crash on all three arms and is bit-neutral elsewhere; but the skipped steps leave no trace in any output (G-01). |
| I034 | `a4f4be0` | The non-finite-loss check looked only at the last of the several inner critic steps of a training step. | **PASS WITH FIXES** | Catches a non-finite value at every inner step (12 of 12 injections) and is bit-neutral; its test would pass two plausible partial fixes (G-02). |
| I042 | `5f30f17` | Manifest settings that an arm's fit never reads were accepted silently. | **PASS WITH FIXES** | Refuses them end to end through the real fitter, refuses 0 of 40,815 manifest rows, and the deliberate difference for the `none` arm is right; but no test notices if the production call is removed (G-03). |
| I050 + I051 | `f5ea0c3` | The scorer checked cell identity only by batch label, and treated a legitimate kBET result obtained without any R call as a failure. | **PASS** | Both evidenced gaps are closed; 0 of 1,701 stored latents refused; metric columns bit-identical on 2 real latents; notes only (G-08 to G-11). |
| I094 | `46d79f2` | Completion, resume and harvest did not require a GPU fit's saved model weights or its loss history. | **PASS** | The stage gate refuses a GPU fit without its model and any latent without its history, and the harvest refuses a GPU latent without its model; 0 of 1,701 stored latents refused; every mutant killed; notes only (G-12 to G-14). |
| I008 | `2b52542` | Every chained X1 cluster job got the same 8-day harness clock, so job 5 could be killed mid-run. | **PASS** | The builder now gives 192/192/384/384/576 h as the plan states; every mutant killed; notes only (G-15 to G-17). |

Findings: 0 blockers, 3 fixes (small changes recommended before the new tag: G-01, G-02, G-03), 15 notes. Mutation checks: 25 of 34 mutants (one-line changes that make the fixed code wrong or partial) are killed by the repository's tests; of the 9 survivors, 2 are equivalent rewrites (correct to survive), 6 show test gaps (G-02, G-03, G-09) and 1 is an implausible variant listed for completeness (section 5).

## 2. Terms used

- **Guard**: code that refuses a run or a row that should not be accepted (or, for I033, a fallback value that lets a rare case continue).
- **I033, I034, I042, I050, I051, I094, I008**: issue numbers of the final pre-publication review (`REVIEW_FINAL.md`, `verified_issues.csv`); each is described in the first table. **SI-nn** are numbered user decisions of this project: SI-52 says the fixes are drafted on a branch, shown as a diff, and independently reviewed before a new tag; SI-51 is the Sinkhorn solver fix (commit 3cbaca4), which changes Sinkhorn results by design and is reviewed separately; SI-24 limits the stratified sampler to the reference and pooled arms. **G-nn** numbers the findings of this report.
- **Bit-identical**: every number of a result (latent coordinates, loss curves, model weights, score columns) equal in every bit, not only to rounding.
- **Minibatch / step / epoch**: training processes the cells in groups (a minibatch; 128 cells in the rows discussed here, 512 in some others); one optimiser update per minibatch is a step; one pass over all training cells is an epoch. The last minibatch of an epoch is smaller (5 to 107 cells across the critic-free rows).
- **Latent**: the low-dimensional coordinates (10 to 100 dimensions in the manifests) the model assigns each cell. **Critic arms** use a trained network to measure batch differences (Wasserstein critic); **critic-free arms** (MMD = maximum mean discrepancy, a kernel distance between two groups of cells; Sinkhorn divergence, a regularised optimal-transport distance) compute the difference directly. Several **inner critic steps** (here 5) train the critic before each model update.
- **kBET**: k-nearest-neighbour batch-effect test, one of the scIB batch-mixing scores; scIB's version calls R. **NMI**: normalised mutual information.
- **Mutation check**: I change one line of the fixed code so that it becomes a plausible partial or wrong fix and run the fix's own test; "killed" means the test fails (good), "survived" means it still passes. **check_tests**: the repository's own script that demands every changed line and branch be run by a test.
- **GPU / CPU**: graphics processor (the production fits) / ordinary processor (all checks here; the GPU was busy with a timing pilot).

## 3. Evidence at a glance (the five checks asked for, per fix)

| Check | I033 | I034 | I042 | I050 + I051 | I094 | I008 |
|---|---|---|---|---|---|---|
| 1 Fully fixes the evidenced problem | Yes: rare-batch fits of MMD, reference MMD and Sinkhorn crash at the tag and finish at the head | Yes: 12 of 12 value-only injections caught; 9 of 12 missed before | Yes: real fitter refuses 5 of 5 bad rows, fits the 2 legitimate ones | Yes: 11 of 11 wrong-identity or legacy latents refused (all 14 probe cases reached scib before), 3 of 3 correct ones pass; all-fragmented kBET recorded as 0 | Yes: gate refuses 4 of 4 degraded copies (all passed before); harvest refuses a GPU latent without model | Yes: builder gives 192/192/384/384/576 h |
| 2 Refuses anything legitimate | No refusal path | No: 1e20-scale control passes; stored losses ≤ 1470 | No: 0 of 40,815 manifest rows refused | No: 0 of 1,701 stored latents refused | No: 0 of 1,701 stored latents refused by the gate rule or by harvest | No: the five planned commands accepted; the old bare `--after-slurm-job` is refused on purpose |
| 3 Result-neutral when not triggered | 42 of 42 configurations bit-identical to the commit before; non-Sinkhorn arms bit-identical to the tag | same runs | same real fits at tag and head give the same latents (4.3, check 3) | 2 of 2 real latents: all 15 compared score columns identical | pure checks; no writes | no effect on fits or scores |
| 4 Test fails on the pre-fix code and on a plausible partial fix | new test fails before (3 failed); 4 of 4 wrong or partial mutants killed; the equivalent rewrite survives, as it should | new test fails before (1 failed); 4 of 6 wrong or partial mutants killed, **2 survive**; the equivalent rewrite survives, as it should (G-02) | new test(s) fail before (1 failed); production call removed or row not passed: **survives** (G-03) | new test(s) fail before (2 failed); 7 of 9 wrong or partial mutants killed, **2 survive** (G-09) | new test(s) fail before (2 failed); 5 of 5 wrong or partial mutants killed | new test(s) fail before (1 failed); 5 of 5 wrong or partial mutants killed |
| 5 Edge cases, operator traps | G-01 | G-02 | G-03 to G-07 | G-08 to G-11 | G-12 to G-14 | G-15 to G-17 |

(Check 4 detail: section 5. "Fails before" = the new test was run on the tree just before the fix and failed; evidence in sections 4.1 to 4.6.)

## 4. Findings and evidence by fix

### 4.1 I033 (c4be14b): inactive-batch comparisons return a zero that carries the graph

**What it changes.** `mmd_batch_others`, `mmd_reference` and `sinkhorn_batch_others` returned `z.new_zeros(())` (a zero without autograd history) when a minibatch had no pair of batches, and the training step's `torch.autograd.grad(adv_term, z_adv)` then raised. They now return `(z * 0).sum()`.

**Checks** (numbered as in the brief).

1. Fully fixes the evidenced problem: yes. Rare-batch fits of MMD, reference MMD (rare and common reference) and Sinkhorn crash at the tag and finish at the head (G-01 evidence). A latent that contains inf gives NaN (0 x inf), not 0, which the existing non-finite check on the alignment term flags (it receives `adv_term`): `one batch, z contains inf: mmd = nan | sinkhorn = nan`.
2. Refuses anything legitimate: no. The fix adds no refusal path; it removes a crash.
3. Result-neutral when not triggered: against the commit before the guards, 42 of 42 configurations bit-identical (0 differ, 0 failed); against the tag, 39 of 42 configurations bit-identical (3 differ, 0 failed), and the 3 that differ are all Sinkhorn arms (their numerics change by SI-51, reviewed separately). The 42 configurations (CPU, toy 600-cell data, 2 epochs) cover every arm of the plan, conditioned and unconditioned, both decoders, `sample`/zstd inputs, stratified sampler, importance weights and KL warm-up; latent, loss history and model-weight hash were compared bit by bit.
4. Test fails on the old code and on a partial fix: the new test on the tree just before the fix: `3 failed, 19 deselected in 9.48s`; on the head: `3 passed, 19 deselected in 7.94s`. Mutants: 4 of 4 wrong versions killed; the equivalent rewrite `z.sum()*0.0` survives, as it should. Repository changed-line gate: `169 passed; check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base 1c49b9a)`.
5. Edge cases and operator traps: G-01 (skipped steps leave no trace).

#### G-01 - fix - I033

- **Location:** src/wcd_vae/wcd/alignment.py: the three `return (z * 0).sum()` lines (mmd_batch_others, mmd_reference, sinkhorn_batch_others); scripts/scvi_adversarial_plan.py critic-free branch of training_step
- **Problem:** A minibatch in which no pair of batches can be compared now adds a zero alignment loss and a zero gradient, so the fit finishes. Nothing in the saved latent, its `config` or its loss history records how many steps were skipped (the history keeps per-epoch means of `adv_div`, which average the zeros in). Before the fix such a fit crashed, which was loud. SI-52 (user decision 52: the new guards must fail loudly and never silently change a result) is therefore only partly met: fits that never hit the case are unchanged, but a fit that does hit it used to stop and now finishes without a trace. The final review asked to 'decide and document' what an inactive minibatch contributes; the decision (zero) is written in the alignment.py module docstring and code comments, not in the methods or preregistration text. In the verifier's own crash-rate table the exposed rows are X7 reference-aligned MMD on pancreas (27 fits) and sim1 (18 fits), where nearly every fit hits it (about 0.17% and 0.30% of minibatch steps), and, rarely, X3 atac_small MMD, X8 sim2 MMD and X1 sim1 (table below). The effect on a result is probably tiny, but a referee cannot see it.
- **Suggested fix:** Keep the zero (it is the natural value and bit-neutral for every minibatch that has a pair). Add a counter of no-pair steps that is written into the saved config only when it is greater than 0 (or one log line per fit), so outputs of fits that never hit the case stay byte-identical, and add one sentence to PREREG/methods naming the row groups that can hit it.

**Evidence**

Command: `python probes/i033_rare_batch_fit.py trees/pre` and `... trees/fix` (CPU, 1 thread, 302 cells, batch b1 has 2 cells, 3 epochs, batch size 128; a spy counts steps whose minibatch has no pair of batches). Output (logs/i033_probe_pre.jsonl, logs/i033_probe_fix.jsonl):

```
configuration              tag c88cce3                                                head 2b52542                                              
mmd_rare_batch             RuntimeError: element 0 of tensors does not require…       ok; 4 of 9 steps without a pair; adv_div per epoch [0.312157, 0.151128, 0.028652]
sinkhorn_rare_batch        RuntimeError: element 0 of tensors does not require…       ok; 4 of 9 steps without a pair; adv_div per epoch [0.744272, 0.350772, 0.16218]
mmd_ref_rare_reference     RuntimeError: element 0 of tensors does not require…       ok; 4 of 9 steps without a pair; adv_div per epoch [0.312157, 0.151128, 0.028652]
mmd_ref_common_reference   RuntimeError: element 0 of tensors does not require…       ok; 4 of 9 steps without a pair; adv_div per epoch [0.312157, 0.151128, 0.028652]
control_mmd_balanced       ok; 0 of 9 steps without a pair                            ok; 0 of 9 steps without a pair; adv_div per epoch [0.014997, 0.010271, 0.015002]
control_sinkhorn_balanced  ok; 0 of 9 steps without a pair                            ok; 0 of 9 steps without a pair; adv_div per epoch [0.25199, 0.206525, 0.197064]
control_pooled_rare_batch  ok; 0 of 0 steps without a pair                            ok (arm without this branch)
```

How often this will happen on the real rows (provisional: derived by me from the verifier's per-row crash probabilities in `verify/model/p_inactive_rate.csv`, not re-simulated):

| experiment | task | arm | fits | n_train | cells_in_last_minibatch | steps_per_epoch | epochs | fits_expected_to_hit_it | share_of_steps_without_pair |
|---|---|---|---|---|---|---|---|---|---|
| X1 | sim1 | mmd | 60 | 10888 | 8 | 86 | 400 | 0.3612 | 1.755e-07 |
| X1 | sim1 | sinkhorn | 60 | 10888 | 8 | 86 | 400 | 0.3612 | 1.755e-07 |
| X3 | atac_small | mmd | 48 | 7562 | 10 | 60 | 400 | 6.673 | 6.394e-06 |
| X7 | pancreas | mmd_ref | 27 | 14744 | 24 | 116 | 400 | 25.92 | 0.001672 |
| X7 | sim1 | mmd_ref | 18 | 10888 | 8 | 86 | 400 | 18 | 0.002993 |
| X8 | sim2 | mmd | 24 | 2700 | 12 | 22 | 400 | 1.076 | 5.579e-06 |

`fits` = fits in the group; `fits_expected_to_hit_it` = expected number of those fits with at least one step without a pair (the sum of the verifier's per-fit probabilities; at the tag each such fit stops with the crash above); `share_of_steps_without_pair` = expected share of minibatch steps without a pair, averaged over the group. The three MMD rows of the probe agree to the last digit because the toy data has two batches, for which all three losses equal the MMD between them.

### 4.2 I034 (a4f4be0): the non-finite check sees every inner critic step

**What it changes.** Inside the loop over the inner critic steps, `d_loss = loss_d + gp` is computed, back-propagated as before, and its detached value added to `inner`; `inner` is passed to the existing `_nf_check` as `adv_inner`. The sum is non-finite if any term is, so one value covers all steps.

**Checks** (numbered as in the brief).

1. Fully fixes the evidenced problem: yes. All 12 value-only injections (loss inf or NaN, gradient-penalty inf, a sum that overflows, each at inner steps 0, 2 and 4) are caught at the head; the code before the fix misses 9 of 12 (G-02 evidence).
2. Refuses anything legitimate: no. A 1e20-scale control passes; the largest per-epoch mean losses in the 1,701 stored histories are 1470 (adv_loss) and 296 (adv_gp), about 35 orders of magnitude below the float32 limit (3.4e38) at which a sum of finite losses would count as non-finite.
3. Result-neutral when not triggered: the same 42-configuration bit-identity as in 4.1 (the plan file changed in this commit and every arm runs through it). The check still needs one host synchronisation per training step (by reading the code; not timed).
4. Test fails on the old code and on a partial fix: the new test fails before the fix (G-02 evidence); 4 of 6 wrong or partial mutants are killed, 2 survive (G-02). Repository changed-line gate: `96 passed; check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base c4be14b)`.
5. Edge cases and operator traps: G-02.

#### G-02 - fix - I034

- **Location:** tests/scvi/test_nonfinite_guard.py::test_nonfinite_inner_critic_step_raises
- **Problem:** The test injects +inf into the critic loss at the first inner critic step only. Two plausible partial fixes pass it: a sum that leaves out the gradient-penalty term (`gp`), and a sum that counts only the first inner step. The pre-fix code and four other wrong variants are caught (last inner step only; sum without the loss term; last two inner steps only; sum never passed to the check). The fix itself is correct: all 12 injections below are caught.
- **Suggested fix:** Parametrize the existing test (same lines, no new test function): inject at the first, a middle and the last inner step, and into the loss and into `gp` (for example `@pytest.mark.parametrize('step,term', [(0,'loss'), (2,'gp'), (4,'loss')])`).

**Evidence**

Command: `python probes/i034_inner_step_injection.py trees/p034` (before the fix, commit c4be14b) and `... trees/fix`. A pooled-critic fit with 5 inner critic steps per training step; at training step 3 one inner step's loss is made non-finite by adding inf, NaN or a gradient-penalty inf, or both terms are set to 3e38 so that their sum overflows; the gradients stay finite. Output (logs/i034_injection_*.jsonl):

```
injection (value only, gradients finite)     before the fix (c4be14b)         after the fix (2b52542)         
loss_inf@inner0                              NOT caught (fit completes)       caught: adv_inner
loss_inf@inner2                              NOT caught (fit completes)       caught: adv_inner
loss_inf@inner4                              caught: adv_loss                 caught: adv_inner,adv_loss
loss_nan@inner0                              NOT caught (fit completes)       caught: adv_inner
loss_nan@inner2                              NOT caught (fit completes)       caught: adv_inner
loss_nan@inner4                              caught: adv_loss                 caught: adv_inner,adv_loss
gp_inf@inner0                                NOT caught (fit completes)       caught: adv_inner
gp_inf@inner2                                NOT caught (fit completes)       caught: adv_inner
gp_inf@inner4                                caught: adv_gp                   caught: adv_gp,adv_inner
sum_overflow@inner0                          NOT caught (fit completes)       caught: adv_inner
sum_overflow@inner2                          NOT caught (fit completes)       caught: adv_inner
sum_overflow@inner4                          NOT caught (fit completes)       caught: adv_inner
large_finite@inner0                          passes (correct: no false alarm) passes (correct: no false alarm)
large_finite@inner2                          passes (correct: no false alarm) passes (correct: no false alarm)
large_finite@inner4                          passes (correct: no false alarm) passes (correct: no false alarm)
```

`large_finite` (1e20 in both terms) is the no-false-alarm control. Largest per-epoch mean |value| in all 1,701 stored loss histories: adv_loss 1470, adv_gp 296 (float32 maximum 3.4e38), so the new sum cannot flag a legitimate fit.

Test on the code before the fix (new test copied onto c4be14b):

```
=========================== short test summary info ============================
FAILED tests/scvi/test_nonfinite_guard.py::test_nonfinite_inner_critic_step_raises
1 failed, 12 deselected, 3 warnings in 7.09s
--- on 2b52542 ---
1 passed, 12 deselected, 3 warnings in 6.23s
```

Mutation check (`python probes/mutate.py ID`; one-line change to the fixed code, then the new test):

```
M34_overwrite                    KILLED    last inner step only (inner = d_loss.detach())
M34_omit_loss                    KILLED    sum omits loss_d
M34_omit_gp                      SURVIVED  sum omits gp
M34_first_only                   SURVIVED  only the first inner step enters the sum
M34_last_two_only                KILLED    only the last two inner steps enter the sum
M34_not_passed                   KILLED    inner computed but not passed to _nf_check
M34_equiv_nodetach               SURVIVED  equivalent: accumulate without detach (should survive)
```

### 4.3 I042 (5f30f17): refuse settings an arm does not read

**What it changes.** `check_extra` now takes the manifest row: extra keys that do not apply to the arm (`adv_width`/`adv_lr` only for arms with a network, `bary_iter`/`bary_warm_iter` only for the barycenter arms, `sampler` only for reference and pooled) and row settings that the arm's fit does not read (see the table below) are refused before any data is read. It deliberately differs from the review's suggestion for the `none` arm and adds two groups the review did not name (`n_critic` on MMD, reference MMD and Sinkhorn; the conditioning flag on scANVI and sysVI). **Judgement: the differences are right.** The unread-setting fits show that `none` reads likelihood, decoder and conditioning (changing any of them changes the latent), so refusing them as the review suggested would block legitimate baselines; MMD, reference MMD and Sinkhorn ignore `n_critic`; scANVI and sysVI ignore decoder, conditioning, `n_critic`, `adv_input` and `zstd`, and sysVI also ignores the likelihood while scANVI reads it (so the likelihood is correctly left alone for scANVI).

**Unread-setting fits** (commands `python probes/i042_unread_fits.py trees/fix`, `i042_baseline_fits.py`, `i042_lam_scvi_adv.py`; tiny CPU fits, default against changed, compared bitwise). 'identical' = the arm ignores the setting; 'DIFFERS' = it reads it:

```
arm       setting changed                      fit result (latent + loss history compared bitwise)
none      n_critic=5                           identical
none      adv_input=sample                     identical
none      zstd=1                               identical
none      d_coef(lam)=5                        identical
none      adv_hidden(adv_width)=64             identical
none      disc_lr(adv_lr)=1e-2                 identical
none      critic_lr(adv_lr)=1e-2               identical
none      bary_iter=3                          identical
none      bary_warm_iter=2                     identical
none      sampler=stratified                   DIFFERS
none      gene_likelihood=nb                   DIFFERS
none      model_name=LinearSCVI                DIFFERS
none      conditioned=False                    DIFFERS
scvi_adv  n_critic=5                           identical
scvi_adv  adv_input=sample                     identical
scvi_adv  zstd=1                               identical
scvi_adv  adv_hidden(adv_width)=64             identical
scvi_adv  disc_lr(adv_lr)=1e-2                 identical
scvi_adv  bary_iter=3                          identical
mmd       n_critic=5                           identical
mmd       adv_hidden(adv_width)=64             identical
mmd       critic_lr(adv_lr)=1e-2               identical
mmd       bary_iter=3                          identical
mmd       bary_warm_iter=2                     identical
mmd       sampler=stratified                   DIFFERS
mmd       adv_input=sample (read by mmd)       DIFFERS
mmd       zstd=1 (read by mmd)                 DIFFERS
mmd_ref   n_critic=5                           identical
mmd_ref   adv_hidden(adv_width)=64             identical
mmd_ref   sampler=stratified                   DIFFERS
sinkhorn  n_critic=5                           identical
sinkhorn  adv_hidden(adv_width)=64             identical
sinkhorn  bary_warm_iter=2                     identical
scanvi    cond=0                               identical   [fit_baseline]
scanvi    decoder=LinearSCVI                   identical   [fit_baseline]
scanvi    n_critic=5                           identical   [fit_baseline]
scanvi    adv_input=sample                     identical   [fit_baseline]
scanvi    zstd=1                               identical   [fit_baseline]
scanvi    likelihood=nb                        DIFFERS   [fit_baseline]
sysvi     cond=0                               identical   [fit_baseline]
sysvi     decoder=LinearSCVI                   identical   [fit_baseline]
sysvi     n_critic=5                           identical   [fit_baseline]
sysvi     adv_input=sample                     identical   [fit_baseline]
sysvi     zstd=1                               identical   [fit_baseline]
sysvi     likelihood=nb                        identical   [fit_baseline]
scvi_adv  lam 0 vs 5                           identical
none      lam 0 vs 5                           identical
```

`DIFFERS` for `sampler=stratified` is not an unread setting: stratified sampling changes which cells form each minibatch, SI-24 allows it only for the reference and pooled arms, and the commit refuses it elsewhere through its allow-list, not through the unread-setting table.

**Checks** (numbered as in the brief).

1. Fully fixes the evidenced problem: yes, end to end. The real `fit_paper_config.py` refuses 5 of 5 bad rows and fits the 2 legitimate rows (G-03 evidence); the settings refused match what the fits above show. What stays open: G-05 (`lam`), G-07 (sysVI labels).
2. Refuses anything legitimate: no. 41,583 manifest rows in 8 files, 40,815 evaluated (the rest are CPU-baseline rows that `main()` routes elsewhere), 0 refused (the author reported 40,815 rows and 0 refused: reproduced). Had the review's suggestion for `none` been applied, 0 current rows would also have been refused, but it would block future `none` baselines with another likelihood, decoder or conditioning.
3. Result-neutral when not triggered: the refusal happens before any state is read or written; the two legitimate rows of the real-`main()` runs give identical latents at the tag and the head (none_default: bit-identical; none_linear_nb_uncond: bit-identical).
4. Test fails on the old code and on a partial fix: the extended unit test fails on the tree just before the fix (1 failed, 0 passed) and passes on the head (1 passed, 0 failed); the production call is untested (G-03) and part of the tables is untested (G-04). Repository changed-line gate: `96 passed; check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base a4f4be0)`.
5. Edge cases and operator traps: G-03 to G-07.

#### G-03 - fix - I042

- **Location:** scripts/fit_paper_config.py line 297 `check_extra(r["tag"], arm, extra, r)  # pragma: no cover`; tests/scvi/*
- **Problem:** The production call that makes the new refusals happen is excluded from coverage (`# pragma: no cover`, reason given: 'run by the subprocess tests ... which coverage does not see'). Running a line is not testing its effect: deleting the call, or calling it without the row (which switches off the unread-setting check), leaves the whole of tests/scvi green (96 passed, 16 skipped), and also the data-dependent tests run with PREPPED_DIR set (33 passed). The unit tests call `check_extra` directly, so nothing would notice a refactor that drops the call. The wiring is correct today: the real `fit_paper_config.py` refuses the five bad rows below at the head and fits the two legitimate rows with latents identical to the tag's; at the tag all seven rows fit.
- **Suggested fix:** Add one subprocess case to the existing section-6 test of tests/scvi/test_nonfinite_guard.py (it already runs fit_paper_config.py end to end): a manifest row `none` with zstd=1 must exit 1, print 'does not read', and write no latent. Then remove the pragma.

**Evidence**

Mutation check on the production wiring. The first two mutants ran the full `tests/scvi` (96 passed, 16 skipped); the third ran the three test files that use real prepped data (`test_runner_and_manifest_missing_arms.py`, `test_x3_si41.py`, `test_x3_x13_design.py`) with PREPPED_DIR set and WCD_REQUIRE_DATA=1 (33 passed):

```
M42_row_not_passed               SURVIVED  main() calls check_extra without the row (unread-setting refusal never runs i…
M42_check_extra_call_removed     SURVIVED  main() never calls check_extra at all
M42_call_removed_with_data       SURVIVED  main() never calls check_extra; data-dependent tests run with PREPPED_DIR
```

The wiring is correct today. Command: `python probes/i042_main_e2e.py WORKDIR trees/fix trees/pre` runs the REAL `scripts/fit_paper_config.py` (CPU, 1 epoch, 1,200-cell subsample of atac_small) on seven manifest rows; output (logs/final_i042_main_e2e.jsonl):

```
row (arm: changed setting) tag c88cce3    head 2b52542
none_default               fits (exit 0)  fits (exit 0)
none_zstd1                 fits (exit 0)  REFUSED (exit 1): E2E_none_zstd1: arm 'none' does not read ['zstd']; they must stay {'zstd': '0'}, got {'zstd': '1'}
none_adv_width             fits (exit 0)  REFUSED (exit 1): E2E_none_adv_width: extra keys ['adv_width'] do not apply to arm 'none'
none_ncritic5              fits (exit 0)  REFUSED (exit 1): E2E_none_ncritic5: arm 'none' does not read ['n_critic']; they must stay {'n_critic': '0'}, got {'n…
mmd_ncritic5               fits (exit 0)  REFUSED (exit 1): E2E_mmd_ncritic5: arm 'mmd' does not read ['n_critic']; they must stay {'n_critic': '0'}, got {'n_c…
sysvi_likelihood_nb        fits (exit 0)  REFUSED (exit 1): E2E_sysvi_likelihood_nb: arm 'sysvi' does not read ['likelihood']; they must stay {'likelihood': 'z…
none_linear_nb_uncond      fits (exit 0)  fits (exit 0)
```

#### G-04 - note - I042

- **Location:** tests/scvi/test_runner_and_manifest_missing_arms.py::test_extra_keys_are_consumed_or_refused; scripts/fit_paper_config.py EXTRA_ARMS, UNREAD_SETTINGS
- **Problem:** The unit test exercises 3 of the 5 EXTRA_ARMS keys (adv_lr and bary_warm_iter are never refused in a test) and 6 of the 20 UNREAD_SETTINGS entries, counting the one dictionary shared by `none` and `scvi_adv` twice; deleting any of the other entries, or `reference_fixed_sn` (a valid arm of the plan that is in neither the runner's ADVERSARIAL set nor the new network-arm set), goes unnoticed. The tables are correct today (see the unread-setting fits), so this is a regression-protection gap only.
- **Suggested fix:** In the same test, loop over EXTRA_ARMS and UNREAD_SETTINGS themselves: for every entry assert that the changed value is refused and the default accepted.

**Evidence**

Command: `python probes/i042_test_table_coverage.py trees/fix` deletes one table entry at a time and re-runs the assertions of the extended unit test in-process. Output:

```
{"EXTRA_ARMS_entries": 5, "EXTRA_ARMS_untested": ["adv_lr", "bary_warm_iter"], "UNREAD_entries": 20, "UNREAD_untested_count": 14, "UNREAD_untested": ["none.n_critic", "none.adv_input", "scvi_adv.n_critic", "scvi_adv.adv_input", "scanvi.n_critic", "scanvi.adv_input", "scanvi.zstd", "scanvi.cond", "sysvi.n_critic", "sysvi.adv_input", "sysvi.zstd", "sysvi.decoder", "mmd.n_critic", "mmd_ref.n_critic"]}
```

#### G-05 - note - I042

- **Location:** scripts/fit_paper_config.py UNREAD_SETTINGS (none, scvi_adv, scanvi); scripts/scvi_adversarial_plan.py:198-199
- **Problem:** The same defect class remains for the strength setting `lam`: the fit of `none` and `scvi_adv` gives an identical latent at lam 0 and 5, and scANVI never reads it, yet a non-zero lam on these arms is accepted. No row is affected: all 1,908 manifest rows of these three arms carry lam = 0. The plan-level silent returns named in the review (lines 198-199) are unchanged; only fit_paper_config calls fit_adversarial_scvi in production, so the runner-level check is where it matters.
- **Suggested fix:** Add `"lam": "0"` to the unread settings of none, scvi_adv and scanvi (sysVI reads lam as z_distance_cycle_weight, so not there).

**Evidence**

Command: `python probes/i042_lam_scvi_adv.py trees/fix` and the manifest audit. Output:

```
scvi_adv  lam 0 vs 5                           identical
none      lam 0 vs 5                           identical
```

lam values of these arms in all eight manifest files (logs/i042_manifest_audit_fix.json): none: 1,428 rows, lam values ['0']; scvi_adv: 240 rows, lam values ['0']; scanvi: 240 rows, lam values ['0']. scANVI: `fit_baseline` receives lam but only the sysVI branch passes it on (scripts/fit_paper_config.py, `plan_kwargs=dict(z_distance_cycle_weight=lam)`).

#### G-06 - note - I042

- **Location:** scripts/run_stage.py unfittable() (preflight); scripts/fit_paper_config.py main()
- **Problem:** The refusal runs when a fit starts, not when the stage plans. run_stage classes any uncaught fitter exception as an infrastructure error, retries it (--max-attempts 2 in the X1 job) and then stops the stage (run_stage.py header). A bad row in a future manifest (X8, X12 follow-ups) would therefore cost two attempts and stop a long job instead of being refused at its start. All 40,815 current rows pass, so there is no current impact.
- **Suggested fix:** Call `check_extra(tag, arm, extra, row)` from the stage preflight (next to `unfittable`) for every selected row.

**Evidence**

`scripts/run_stage.py` header: 'infrastructure: any other fitter / scorer failure ... retried up to --max-attempts times in this run, then the stage stops'. X1 job arguments: `--max-attempts 2` (experiments/X1_jhpce/SUBMISSION_PLAN.md, job 1 command). `check_extra` has no caller other than `main()` of fit_paper_config.py (`grep -rn check_extra scripts`). Audit of every manifest row (logs/i042_manifest_audit_fix.json): 41583 rows, 40815 evaluated (the other 768 are X13 CPU-baseline rows, which main() routes elsewhere), 0 refused.

#### G-07 - note - I042

- **Location:** docs / methods text (not code)
- **Problem:** Left unaddressed from the review row: the 150 sysVI filler latents record row likelihood 'zinb' and decoder 'SCVI', labels sysVI does not use (it is Gaussian, SysVAE). The commit stops new rows from carrying other values; it cannot correct the existing provenance text. No commit in the range touches PAPER_PLAN or the methods text.
- **Suggested fix:** State 'sysVI: Gaussian likelihood, SysVAE decoder; the likelihood/decoder columns are placeholders' in the methods and table notes.

**Evidence**

`git show --stat` of the six commits lists only scripts/, src/, tests/, cluster/, experiments/X1_jhpce/ and docs/jhpce/ files. Recorded labels of the 150 sysVI filler latents are those named in the review row (I042, evidence).

### 4.4 I050 + I051 (f5ea0c3): scorer guards

**What it changes.** I050: before scoring, the latent's `obs_names` and `celltype` must equal the prepped file's, in the same order (after the subset re-indexing, so both the full-length and the X3/X8 subset paths are covered); the legacy writers' latents (no `obs_names`) are refused. I051: a finite kBET with zero seeded R calls is recorded when it is exactly 0 (scib set every tested label to rejection 1), and refused otherwise.

**Judgement on I051.** The criterion differs from the review's suggestion (`forced count == tested count`) but is as strong: `kbet_label_flags` already refuses any call-count mismatch, and with 0 seeded calls it forces `forced == tested`; probe (c) below, a bypassed seed wrapper with a real R call that returns exactly 0, is still refused by that accounting error.

**kBET probes** (`REPO=trees/<tree> V=logs/i051_tmp python probes/i051_kbet_probes.py`; real `main()`, only kBET computed by the real scib inside the scorer's wrappers): (a) 8 far-apart islands, every label fragmented; (b) mixed batches with the seed wrapper bypassed; (c) batches separated inside every label (R returns kBET exactly 0) with the wrapper bypassed; (d) the same latent with the wrapper active:

```
case                                 before (5f30f17)                                                                 after (2b52542)
a_all_forced_islands                 REFUSED: kBET is finite but no seeded R kBET call ran: the seed wrapper was b…   RECORDED kBET=0.0, R calls=0, forced labels=6
b_mixed_bypassed_nonzero             REFUSED: kBET is finite but no seeded R kBET call ran: the seed wrapper was b…   REFUSED: kBET is finite and non-zero but no seeded R kBET call ran: the seed wrapper was bypassed
c_separated_bypassed_exact_zero      REFUSED: kBET is finite but no seeded R kBET call ran: the seed wrapper was b…   REFUSED: kBET call accounting does not match scib 1.1.7 (tested labels 3, diffusion_nn 3 (0 Neighb…
d_separated_seeded                   RECORDED kBET=0.0, R calls=3, forced labels=0                                    RECORDED kBET=0.0, R calls=3, forced labels=0
```

**Cell-identity probes** (`REPO=trees/<tree> V=logs/i050_tmp python probes/i050_identity_probe.py`; real `score_scib_native.main()` on a 40-cell prepped file; `scib.metrics.metrics` replaced by a spy that records whether scoring was reached, 'REACHED' = the scorer would have scored with this latent). Cases: the correct latent; cells permuted within each batch; unknown cell names; duplicated cell names (each with true and with swapped cell types); the same for subset latents of the X3/X8 path; latents of the legacy writers that lack `obs_names` or `celltype`:

```
=== before the fix (tree 5f30f17)
correct                  true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell0', 'cell1', 'cell2'] paired with prepped ['cell0', 'cell1', 'cell2']
correct                  wrong_celltype  -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell0', 'cell1', 'cell2'] paired with prepped ['cell0', 'cell1', 'cell2']
within_batch_permuted    true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell1', 'cell10', 'cell18'] paired with prepped ['cell0', 'cell1', 'cell2']
within_batch_permuted    wrong_celltype  -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell1', 'cell10', 'cell18'] paired with prepped ['cell0', 'cell1', 'cell2']
unknown_names            true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['x0', 'x1', 'x2'] paired with prepped ['cell0', 'cell1', 'cell2']
unknown_names            wrong_celltype  -> REACHED scib.metrics.metrics; latent obs_names[:3]=['x0', 'x1', 'x2'] paired with prepped ['cell0', 'cell1', 'cell2']
duplicated_names         true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell0', 'cell1', 'cell2'] paired with prepped ['cell0', 'cell1', 'cell2']
duplicated_names         wrong_celltype  -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell0', 'cell1', 'cell2'] paired with prepped ['cell0', 'cell1', 'cell2']
subset_correct_order     true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell0', 'cell2', 'cell4'] paired with prepped ['cell0', 'cell2', 'cell4']
subset_correct_order     wrong_celltype  -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell0', 'cell2', 'cell4'] paired with prepped ['cell0', 'cell2', 'cell4']
subset_permuted          true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell38', 'cell36', 'cell34'] paired with prepped ['cell38', 'cell36', 'cell34'
subset_permuted          wrong_celltype  -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell38', 'cell36', 'cell34'] paired with prepped ['cell38', 'cell36', 'cell34'
legacy_no_obs_names      -               -> REACHED scib.metrics.metrics
legacy_no_celltype       -               -> REACHED scib.metrics.metrics
=== head (2b52542)
correct                  true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell0', 'cell1', 'cell2'] paired with prepped ['cell0', 'cell1', 'cell2']
correct                  wrong_celltype  -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
within_batch_permuted    true_celltype   -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
within_batch_permuted    wrong_celltype  -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
unknown_names            true_celltype   -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
unknown_names            wrong_celltype  -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
duplicated_names         true_celltype   -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
duplicated_names         wrong_celltype  -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
subset_correct_order     true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell0', 'cell2', 'cell4'] paired with prepped ['cell0', 'cell2', 'cell4']
subset_correct_order     wrong_celltype  -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
subset_permuted          true_celltype   -> REACHED scib.metrics.metrics; latent obs_names[:3]=['cell38', 'cell36', 'cell34'] paired with prepped ['cell38', 'cell36', 'cell34'
subset_permuted          wrong_celltype  -> REFUSED ValueError: the latent's cells or cell types are not those of the prepped file
legacy_no_obs_names      -               -> REFUSED KeyError: 'obs_names is not a file in the archive'
legacy_no_celltype       -               -> REFUSED KeyError: 'celltype is not a file in the archive'
```

**Checks** (numbered as in the brief).

1. Fully fixes the evidenced problem: yes, both. Before the fix all 14 probe cases reached scoring; after it the 3 correct ones pass and 11 of 11 wrong-identity or legacy latents are refused. An all-fragmented kBET is recorded as 0 (probe a), while a non-zero kBET without a seeded call and an accounting mismatch are still refused (b, c).
2. Refuses anything legitimate: no. 1,701 stored latents, all full-length: the new identity check passes for 1,701; longest cell-type label 32 characters (stored width 64) and longest cell name 65 (width 128), so the fixed-width storage cannot truncate a name into a false refusal. 0 of 1,608 stored score rows has an R-call count of 0 for kBET (smallest count 3; logs/audit_stored_scores.json), so no stored row relies on the I051 path.
3. Result-neutral when not triggered: on 2 real atac_small latents all 15 compared score columns are identical between the tagged and the fixed scorer; only `scorer_git_sha` and the run time differ (G-10).
4. Test fails on the old code and on a partial fix: the two new tests fail on the tree just before the fix (2 failed, 0 passed) and pass on the head (2 passed, 0 failed). Mutants: names-only, cell-types-only, set-equality, check removed, check relocated in front of the re-indexing, I051 guard removed and I051 old guard restored are all killed; the full-length-only variant and a guard that only refuses kBET above 0.5 survive (G-09; the latter is not a plausible fix and is listed for completeness). Repository changed-line gate: `14 passed; check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base 5f30f17)`. End to end through the real fitter, `run_stage` and scorer (opt-in `tests/stage/test_run_stage_e2e.py`): 1 passed in 76.33 s.
5. Edge cases and operator traps: G-08 to G-11.

#### G-08 - note - I050+I051

- **Location:** scripts/score_scib_native.py lines 200-203
- **Problem:** A latent without `obs_names` or `celltype` (the legacy writers fit_stock_scvi.py and scvi_fit_pancreas.py) is now refused, as the commit message says, but with a bare KeyError ('obs_names is not a file in the archive'). Before, a latent without `celltype` was accepted, because the scorer never read that key. All 1,701 stored latents have both keys.
- **Suggested fix:** Test `"obs_names" in d and "celltype" in d` first and raise ValueError naming the writer and the missing key.

**Evidence**

The identity probe in section 4.4 contains the two legacy cases (logs/final_i050_identity.txt). After the fix:

```
legacy_no_obs_names      -               -> REFUSED KeyError: 'obs_names is not a file in the archive'
legacy_no_celltype       -               -> REFUSED KeyError: 'celltype is not a file in the archive'
```

Before the fix (tree 5f30f17) both reached scoring.

#### G-09 - note - I050+I051

- **Location:** tests/scoring/test_scorer_provenance_and_subset.py::test_full_length_latent_must_list_the_prepped_cells_in_order
- **Problem:** The check also covers the subsample path (X3/X8) because it sits after the re-indexing; the review asked for the full-length path only. That extra strength is untested: a variant that checks the full-length path only passes all tests. Moving the check in front of the re-indexing (which would refuse every legitimate subset latent) is caught by the existing subsample test.
- **Suggested fix:** Optional: add one subset latent with a swapped cell type to the existing parametrization.

**Evidence**

Mutation check:

```
M50_names_only                   KILLED    check cell names only
M50_celltype_only                KILLED    check cell types only
M50_set_equality                 KILLED    names compared as sets (order ignored)
M50_before_subset_reindex        KILLED    I050 check removed entirely
M50_check_before_reindex3        KILLED    check relocated before the subset re-indexing (refuses legit subset latents)
M50_full_length_path_only        SURVIVED  check runs on the full-length path only (the review's wording); subset latent…
```

#### G-10 - note - I050+I051

- **Location:** scripts/score_scib_native.py (scorer_git_sha column); scripts/prereg_rules.py check_provenance
- **Problem:** The fixed scorer writes score rows whose metric columns, kbet_seed and kbet_r_calls are bit-identical to the tagged scorer's (2 real atac_small latents, same machine), but whose `scorer_git_sha` is the new commit. prereg_rules.check_provenance refuses a rule input that mixes scorer_git_sha values, so rows scored at c88cce3 (A1, A2/A3, fillers) and rows scored at the new tag cannot be combined in one rule input without either re-scoring the old rows (metrics will not change) or an amendment. This is a consequence of making any scorer change, not a defect of the fix.
- **Suggested fix:** Decide before the tag: re-score the 1,608 stored score rows with the new scorer (metric values unchanged, provenance columns updated) or document an amendment of the mixing rule.

**Evidence**

Command: score two stored latents (atac_small) with the tagged scorer (trees/pre) and the fixed scorer (trees/fix) on this machine, then `python scripts/compare_score_rows.py REF CAND`. Output:

```
A1_atac_small_barycenter_l3000_c0_s102_2863247f: columns that differ between the tagged and the fixed scorer: ['scorer_git_sha', 'score_seconds']; kBET=0.0486700982255099, forced labels=2, R calls=4
A1_atac_small_none_l0_c1_s100_8d38e2a3: columns that differ between the tagged and the fixed scorer: ['scorer_git_sha', 'score_seconds']; kBET=0.3128825001300462, forced labels=0, R calls=6
```

`compare_score_rows.py` reported 15 columns compared and 0 differing for both latents (the 13 metric columns, kbet_seed, kbet_r_calls). Against the stored score row of the original run one value differs (PCR_batch 0.435987260568244 against 0.4359877709055503 on the barycenter latent): that is a difference between two runs on two machines, not a difference between the scorers. All 1,608 stored score rows carry scorer_git_sha c88cce3c (A1 1,098; R1 extension 36; R4 extension 9; A2/A3 288; X1 local 19; fillers 158; logs/audit_stored_scores.json). `scripts/prereg_rules.py` lines 219-235 (`check_provenance`) refuse a rule input that mixes values.

#### G-11 - note - I050+I051

- **Location:** scripts/score_scib_native.py (commit message note); scib 1.1.7 cluster_optimal_resolution
- **Problem:** Documented by the author and not fixed: when NMI is exactly 0 at every resolution scib keeps no clustering and the NMI step fails on None labels (a loud error). I did not reproduce it. A fully fragmented latent that I051 now records as kBET 0 could, in principle, also hit it. The accepted I051 case (every tested label fragmented) is recorded correctly (probe a).
- **Suggested fix:** Keep as a known residual; if it ever occurs, record the row as a scoring failure with its reason rather than retrying it as an infrastructure error.

**Evidence**

Author's note in the commit message of f5ea0c3 (quoted in the finding); not reproduced. The I051 case itself (every tested label fragmented) is recorded correctly: probe (a) in section 4.4 (kBET 0.0, 0 R calls, 6 forced labels).

### 4.5 I094 (46d79f2): completion, resume and harvest require model weights and history

**What it changes.** `run_stage.Outputs.latent` (the source of every state: fitted, scored, resume) now refuses a GPU row without `models/<tag>/model.pt` and any latent without a `history` key; `prod_helpers.harvest_files` refuses to pack a GPU latent (config without device `cpu`) whose model is missing.

**Checks** (numbered as in the brief).

1. Fully fixes the evidenced problem: yes for the stage rule and for the model in the harvest; the harvest does not check `history` (G-14). Gate probe (G-13 evidence): before the fix all variants pass the gate; after it the baseline passes and all four degraded copies (models removed, one model removed, history removed, both removed) are refused with exit 4 and a message naming the file.
2. Refuses anything legitimate: no. The stage rule and the harvest refuse none of the 1,701 stored latents (G-12 and G-13 evidence); CPU-baseline rows are exempt from the model rule in both places and write an empty history.
3. Result-neutral when not triggered: pure read-only checks; nothing is written.
4. Test fails on the old code and on a partial fix: the two new tests fail on the tree just before the fix (2 failed, 0 passed) and pass on the head (2 passed, 0 failed). Mutants: model rule removed, history rule removed, model demanded of CPU rows too (killed by `test_cpu_rows_run_on_cpu_lanes`), harvest rule removed, harvest CPU exemption removed: 5 of 5 killed. Suites on the head: `tests/stage` 62 passed, 1 skipped in 542.08 s (the skipped one is the opt-in end-to-end test, run separately in section 7); `tests/jhpce` 42 passed in 217.96 s. The repository gate `check_tests` stopped here with an internal error of my coverage plug-in (section 8), so I measured the changed lines with plain coverage: `cluster/jhpce/prod_helpers.py`: all 7 executable changed lines were run (lines 393, 394, 395, 396, 407, 408, 410) and no branch of them was left unexecuted (pytest: 41 passed in 210.06s (0:03:30)); `scripts/run_stage.py` (it runs in subprocesses; measured with a start-up hook, as the author did): all 3 executable changed lines were run (lines 390, 400, 401) and no branch of them was left unexecuted (pytest: 1 passed in 11.63s).
5. Edge cases and operator traps: G-12 to G-14.

#### G-12 - note - I094

- **Location:** cluster/jhpce/prod_helpers.py harvest_files (SystemExit); scripts/run_stage.py Outputs.latent message
- **Problem:** The refusal is all-or-nothing: one GPU latent without its model makes `pack` exit non-zero, prod_fit_job.sh then exits 10, and none of the job's other latents is packed; they stay on fastscratch, which is purged after 30 days. The messages say 'refit it' but not how (delete latents/<tag>.npz, its score and status files, then resume). This can only happen after a purge or a manual deletion, because the fitter saves the model before the latent becomes visible.
- **Suggested fix:** Add the remedy to both messages; optionally let pack list every offending tag in one error instead of the first.

**Evidence**

`cluster/jhpce/prod_fit_job.sh` lines 221-224: `if ! "$FIT_PY" "$H" pack ... ; then echo "FATAL: harvest pack failed ..." >&2; exit 10`. `harvest_files` is called once for all tags of the job (`cmd_pack`). The fitter writes `model.save(...)` before `os.replace(tmp, npz)` (scripts/fit_paper_config.py end of `main`), so within a job a latent never exists without its model. The 30-day purge is stated in `docs/jhpce/GATE_JHPCE.md` (line 404) and `cluster/jhpce/env.sh`. Audit of the fixed `harvest_files` on every stored stage directory (read only; logs/final_i094_harvest_audit.jsonl):

```
{"dir": "tier12_v2/A1", "tags": 1098, "files": 5509, "model_pt": 1098, "refused": 0}
{"dir": "tier12_v2/A1_ext_r1_round1", "tags": 36, "files": 180, "model_pt": 36, "refused": 0}
{"dir": "tier12_v2/A1_ext_r4_round1", "tags": 9, "files": 45, "model_pt": 9, "refused": 0}
{"dir": "tier12_v2/A2A3", "tags": 288, "files": 1440, "model_pt": 288, "refused": 0}
{"dir": "tier12_v2/X1_local", "tags": 20, "files": 100, "model_pt": 20, "refused": 0}
{"dir": "jhpce_tier12/fillers_x1_x13", "tags": 250, "files": 1162, "model_pt": 250, "refused": 0}
```

#### G-13 - note - I094

- **Location:** scripts/run_stage.py Outputs.latent; docs
- **Problem:** From now on `models/<tag>/model.pt` must stay next to each GPU latent for as long as the stage may be re-gated (`--report-only`, resume, local scoring): deleting models after harvest to save disk (about 6.8 MB per fit, ~12 GB for X1's 1,800 fits) makes the stage refuse (exit 4). There is no opt-out flag. `history` is required as a key but an empty history passes (CPU baselines write `{}` by design).
- **Suggested fix:** Say so in docs/jhpce/PRODUCTION_JOB.md and the SUBMISSION_PLAN 'After each job' section.

**Evidence**

Stage gate on copies of 3 real A1 fits (command: `python probes/gate/i094_gate_variants.py REPO PREPPED FIT_PY LABEL`; `run_stage.py --report-only`; logs/final_i094_gate.jsonl):

```
variant                        before (f5ea0c3)   after (2b52542)
baseline_real_A1_copies        gate PASS (exit 0) gate PASS (exit 0)
model_dirs_removed             gate PASS (exit 0) refused (exit 4)
one_model_removed              gate PASS (exit 0) refused (exit 4)
history_removed                gate PASS (exit 0) refused (exit 4)
models_and_history_removed     gate PASS (exit 0) refused (exit 4)
```

Audit of all 1701 stored latents: with model.pt 1701, with the history key 1701, with a non-empty history 1701 (logs/audit_stored_latents.csv). Size of a model: 6.8 MB per fit (SUBMISSION_PLAN, measured on the filler jobs).

#### G-14 - note - I094

- **Location:** cluster/jhpce/prod_helpers.py harvest_files; commit message of 46d79f2
- **Problem:** The commit title says the harvest requires 'every latent's history', and the review row asks for the model and the `history` key in `Outputs.state` and in `harvest_files`. The code checks `history` only in the stage rule (`Outputs.latent`); `harvest_files` checks the model only. A GPU latent that has its model but no `history` key is therefore packed (probe below), while the stage gate refuses it (exit 4), so the loud refusal comes later, at the next local gate. The current fitter always writes `history` and all 1,701 stored latents have it, so no existing or planned latent is affected.
- **Suggested fix:** Either add the one-line check for the `history` key to `harvest_files` (numpy is already imported there for the device test), or reword the commit message and PRODUCTION_JOB.md to say that the harvest checks the model only.

**Evidence**

Command: `python probes/i094_harvest_nohistory.py trees/fix` (a copy of a real A1 latent with the `history` key removed, plus a model file; `harvest_files` of the head). Output (logs/final_i094_harvest_nohistory.json):

```
{"tag": "A1_atac_small_barycenter_l0.1_c0_s100_ee1decf8", "device_in_config": null, "latent_has_history": false, "model_present": true, "result": "PACKED (no refusal)", "files": ["latents/A1_atac_small_barycenter_l0.1_c0_s100_ee1decf8.npz", "models/A1_atac_small_barycenter_l0.1_c0_s100_ee1decf8/model.pt"]}
```

The stage gate refuses the same defect (section 4.5 gate table, `history_removed`: exit 4). In `harvest_files` only `models/<tag>/model.pt` is tested. `cluster/jhpce/prod_fit_job.sh` runs `pack` after the runner whatever its exit code (lines 221-224), so a refused stage (exit 4) and a packed latent can coexist in one job.

### 4.6 I008 (2b52542): X1 cluster builder scales the harness clock with the chain position

**What it changes.** New `--chain-position K` (given exactly when `--after-slurm-job` is given); `run_timeout_s = (K + 1) x (walltime + queue allowance)` = 192, 384, 576 h for K = 0, 1, 2; the plan and `PRODUCTION_JOB.md` state it.

**Checks** (numbered as in the brief).

1. Fully fixes the evidenced problem: yes. The builder output below (a bundle of the tag; nothing is submitted) gives 192, 192, 384, 384 and 576 h, as SUBMISSION_PLAN lines 21-22 state, and the clock is shown at the end of each approval-card intent. Whether the harness accepts 576 h is not checkable without a submission (G-15).
2. Refuses anything legitimate: no. The five planned commands are accepted; the old bare `--after-slurm-job` is refused on purpose, with an explicit message.
3. Result-neutral when not triggered: the builder only prints a submit plan; no fit or score depends on it.
4. Test fails on the old code and on a partial fix: the new test fails on the tree just before the fix (1 failed, 0 passed) and passes on the head (1 passed, 0 failed). 5 of 5 mutants killed (off by one, walltime-only scaling, position ignored, consistency check removed, only negative positions refused); the failure of the off-by-one mutant is the intended assertion (0 against 691,200 s). Changed lines of the builder run by the new test (plain coverage; `check_tests` stopped with the same plug-in error, section 8): all 7 executable changed lines were run (lines 37, 41, 68, 71, 72, 119, 120) and no branch of them was left unexecuted (pytest: 1 passed in 1.60s).
5. Edge cases and operator traps: G-15 to G-17.

```
immune_hum_mou  chain position 0 after job None  run_timeout_s=691200 (192 h)   card intent ends: ...stscratch; harness clock 192 h
atac_large      chain position 0 after job None  run_timeout_s=691200 (192 h)   card intent ends: ...stscratch; harness clock 192 h
lung            chain position 1 after job 111   run_timeout_s=1382400 (384 h)   card intent ends: ... 111 ends; harness clock 384 h
sim2            chain position 1 after job 222   run_timeout_s=1382400 (384 h)   card intent ends: ... 222 ends; harness clock 384 h
pancreas        chain position 2 after job 444   run_timeout_s=2073600 (576 h)   card intent ends: ... 444 ends; harness clock 576 h
refused: builder --chain-position 2             -> --after-slurm-job is given exactly when --chain-position > 0 (the jobs before this one)
refused: builder --after-slurm-job 444          -> --after-slurm-job is given exactly when --chain-position > 0 (the jobs before this one)
refused: builder --chain-position -1            -> --after-slurm-job is given exactly when --chain-position > 0 (the jobs before this one)
```

#### G-15 - note - I008

- **Location:** experiments/X1_jhpce/x1_jhpce_command.py run_timeout_s; experiments/X1_jhpce/SUBMISSION_PLAN.md lines 21-22
- **Problem:** Jobs 3 to 5 now ask the harness for 384 h and 576 h; the largest value used so far is 192 h. The harness raises `TimeoutCeiling` when run_timeout_s exceeds a per-target ceiling (documented in host.compute, value not published; the review also marked this not checkable without a submission). If the ceiling is below 576 h the refusal is loud and happens at submission, but the plan gives no fallback.
- **Suggested fix:** Add to SUBMISSION_PLAN: if the harness refuses more than 192 h, submit jobs 3-5 one at a time after their predecessor ends (clock 192 h each, as the review's alternative suggested).

**Evidence**

The builder output for the five jobs is the block in section 4.6 (logs/final_i008_builder.jsonl). `help(host.compute)` lists the exception `TimeoutCeiling`: 'run_timeout_s above the target's ceiling (see .limit)'. The ceiling is not given there or in the JHPCE host notes.

#### G-16 - note - I008

- **Location:** experiments/X1_jhpce/x1_jhpce_command.py --chain-position
- **Problem:** The chain position is typed by the operator and cannot be checked against the job's real place in the chain (the builder rejects only inconsistent combinations). A mislabelled job 5 (`--chain-position 1`) silently gets 384 h, which is the original under-provisioning problem again for the worst case. The builder does print the clock in the approval-card intent ('harness clock 576 h'), and the plan states the three values.
- **Suggested fix:** Compare the printed clock with the SUBMISSION_PLAN values (192/192/384/384/576 h) before approving each card; optionally key the position to the task name inside the builder.

**Evidence**

Same output as section 4.6: the clock appears in the plan (`run_timeout_s`) and at the end of the approval-card intent; the builder accepts any positive `--chain-position` together with an id.

#### G-17 - note - I008

- **Location:** experiments/X1_jhpce/x1_jhpce_command.py TAG_SHA and --expected-sha; SUBMISSION_PLAN 'Code and inputs'
- **Problem:** Outside the six fixes but an operator will meet it: the builder refuses any commit other than c88cce3 (`TAG_SHA`), and the plan says the code is 'the tag prereg-tier12-v2 (c88cce3), unchanged', because the 75 X1 baseline fits already made on JHPCE are at c88cce3 and fits from a new commit would trip the runner's mixed-commit check. The six fixes (I033 in particular, which the review required before the X1 sim1 rows run) need a new tag, so after it exists that statement, the builder constant and tests/jhpce/test_x1_command.py (which bundles TAG_SHA) are out of date, and the 75 baseline fits at c88cce3 and the new X1 fits would not share one commit.
- **Suggested fix:** Re-pin TAG_SHA, the plan text and the test when the new tag is cut; decide beforehand how the 75 existing X1 baseline fits are treated (refit at the new tag or amend the one-commit rule).

**Evidence**

`grep -n "TAG_SHA\|expected-sha" experiments/X1_jhpce/x1_jhpce_command.py`: line `TAG_SHA = "c88cce3c..."`, and the refusal `--expected-sha must be prereg-tier12-v2`. SUBMISSION_PLAN 'Code and inputs': 'Code: the tag prereg-tier12-v2 (c88cce3), unchanged'.

### 4.7 Tooling note outside the six commits

**What it concerns.** The repository's own before/after bit-identity gate, which is the tool that would show SI-52's 'results that do not trigger a guard stay bit-identical'. It is not part of the six commits and does not change a verdict.

#### G-18 - note - all (tooling)

- **Location:** scripts/check_arm_bitidentity.py export_base
- **Problem:** The repository's own before/after bit-identity gate cannot run against the tag or any later base: it exports only `scripts/scvi_adversarial_plan.py` and `src/wcd_vae/wcd/` of the base commit, but since commit b0564fb the plan imports `scripts/fit_outcome.py`, so the base fits stop with `ModuleNotFoundError: No module named 'fit_outcome'`. The script's exit code for this crash (1) is the same as for 'a difference was found'. SI-52 asks for bit-identity on non-Sinkhorn arms; with this gate it cannot be shown. I used my own driver (`probes/bitid_driver.py`, same idea, exporting the whole `scripts/` and `src/`) for section 4.1. No defect of the six fixes.
- **Suggested fix:** Add `scripts/fit_outcome.py` (or the whole `scripts/` directory) to the files `export_base` writes, and give the crash a different exit code from a bit difference.

**Evidence**

Command: `python scripts/check_arm_bitidentity.py --base c88cce3 --out X` from the head (the same with base 1c49b9a). Output (logs/bitid_arms_vs_tag_c88cce3.log, first lines; the script then exits with code 1, the code it also uses for 'a difference was found'):

```
Traceback (most recent call last):
  File "<string>", line 9, in <module>
  File "<tmp>/base/scripts/scvi_adversarial_plan.py", line 54, in <module>
    from fit_outcome import NonFiniteLossError   # scripts/fit_outcome.py; scripts/ is on sys.path for every importer
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'fit_outcome'
```

`export_base` writes `scripts/scvi_adversarial_plan.py` and `src/wcd_vae/wcd/*` of the base commit into a temporary tree; the plan has imported `scripts/fit_outcome.py` since commit b0564fb (an ancestor of the tag). Used instead: `probes/bitid_driver.py` and `probes/bitid_compare.py` (same idea; whole `scripts/` and `src/` exported), results in section 4.1.

## 5. Mutation matrix (all mutants; also `mutation_matrix.csv`)

```
M33_mmd_unfixed                  I033   KILLED    mmd_batch_others branch reverted
M33_mmdref_unfixed               I033   KILLED    mmd_reference branch reverted
M33_sinkhorn_unfixed             I033   KILLED    sinkhorn branch reverted
M33_disconnected_graph           I033   KILLED    zero with a graph NOT connected to z (requires_grad leaf)
M33_equiv_sum                    I033   SURVIVED  equivalent form z.sum()*0.0 (should survive)
M34_overwrite                    I034   KILLED    last inner step only (inner = d_loss.detach())
M34_omit_loss                    I034   KILLED    sum omits loss_d
M34_omit_gp                      I034   SURVIVED  sum omits gp
M34_first_only                   I034   SURVIVED  only the first inner step enters the sum
M34_last_two_only                I034   KILLED    only the last two inner steps enter the sum
M34_not_passed                   I034   KILLED    inner computed but not passed to _nf_check
M34_equiv_nodetach               I034   SURVIVED  equivalent: accumulate without detach (should survive)
M42_row_not_passed               I042   SURVIVED  main() calls check_extra without the row (unread-setting refusal never runs in production)
M42_check_extra_call_removed     I042   SURVIVED  main() never calls check_extra at all
M42_call_removed_with_data       I042   SURVIVED  main() never calls check_extra; data-dependent tests run with PREPPED_DIR
M50_names_only                   I050   KILLED    check cell names only
M50_celltype_only                I050   KILLED    check cell types only
M50_set_equality                 I050   KILLED    names compared as sets (order ignored)
M50_before_subset_reindex        I050   KILLED    I050 check removed entirely
M50_check_before_reindex3        I050   KILLED    check relocated before the subset re-indexing (refuses legit subset latents)
M50_full_length_path_only        I050   SURVIVED  check runs on the full-length path only (the review's wording); subset latents unchecked
M51_guard_dropped                I051   KILLED    kBET zero-call guard removed entirely
M51_old_guard                    I051   KILLED    pre-fix guard restored (any finite kBET with zero calls refused)
M51_weak_guard                   I051   SURVIVED  guard raises only above kBET 0.5
M94_stage_no_model_check         I094   KILLED    run_stage: model.pt check removed
M94_stage_no_history             I094   KILLED    run_stage: 'history' no longer required
M94_stage_model_for_cpu_rows     I094   KILLED    run_stage: model.pt demanded for CPU baseline rows too
M94_harvest_no_check             I094   KILLED    harvest_files: model check removed
M94_harvest_no_cpu_exemption     I094   KILLED    harvest_files: CPU baselines also need a model
M8_off_by_one                    I008   KILLED    clock = chain_position x (...) (no +1)
M8_wall_only                     I008   KILLED    chain scales the walltime only, not the queue allowance
M8_ignore_chain                  I008   KILLED    chain position accepted but ignored (previous clock)
M8_no_validation                 I008   KILLED    chain-position/after-slurm-job consistency check removed
M8_weak_validation               I008   KILLED    only negative positions refused
```

Survivors and their meaning: `M33_equiv_sum` and `M34_equiv_nodetach` are equivalent rewrites (correct to survive). `M34_omit_gp`, `M34_first_only` (G-02), `M42_*` (G-03), `M50_full_length_path_only` (G-09) are the 6 test gaps. `M51_weak_guard` (refuse only above kBET 0.5) shows that the bypass test pins one kBET value; it is not a plausible fix and needs no action. The harness itself had one defect that I corrected before reporting (a relocation mutant that replaced the check at both places); only the corrected result is in the matrix.

## 6. Items of the final review, row by row, that these commits leave open

- **I033:** the review asked to "decide and document what an inactive minibatch contributes": decided (0) and documented in the module docstring and code comments of `alignment.py`, not in the methods or preregistration text; no per-fit record (G-01). The review's crash-rate numbers (52.4 expected crashes among 1,302 not-yet-run critic-free rows) are not re-simulated here.
- **I034:** nothing open. The saved `adv_loss` history still shows only the last inner step, which no longer matters because the guard now sees all of them.
- **I042:** the provenance text for the 150 sysVI filler latents (G-07); the plan-level silent returns (`scvi_adversarial_plan.py` 198-199, 554-557) were not changed, which is acceptable because only `fit_paper_config.py` calls the plan in production (G-05); the review's 'one parametrized test' became extra cases in an existing unit test, not exhaustive (G-04).
- **I050 / I051:** nothing open. Residual documented by the author: NMI failing on None labels for a clustering with NMI exactly 0 (G-11).
- **I094:** the review row asks for the model and the `history` key in both the stage rule and `harvest_files`; the harvest checks the model only (G-14); the stage gate and the first local gate still refuse a latent without history.
- **I008:** the harness ceiling for `run_timeout_s` remains not checkable without a submission (G-15).

## 7. Test suites and gates run

| Suite | Tree | Result |
|---|---|---|
| tests/wcd_vae, `check_tests` (wcd-gpu) | c4be14b (I033) | 169 passed; check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base 1c49b9a) |
| tests/scvi, `check_tests` (scvi-api) | a4f4be0 (I034) | 96 passed; check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base c4be14b) |
| tests/scvi, `check_tests` (scvi-api) | 5f30f17 (I042) | 96 passed; check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base a4f4be0) |
| tests/scoring, `check_tests` (wcd-kbet with R) | f5ea0c3 (I050, I051) | 14 passed; check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base 5f30f17) |
| tests/scvi/test_nonfinite_guard.py | head | 13 passed |
| tests/stage | head | 62 passed, 1 skipped in 542.08 s |
| tests/stage/test_run_stage_e2e.py (opt-in: real fitter, `run_stage`, scorer, 1,200-cell atac_small) | head | 1 passed in 76.33 s |
| tests/jhpce | head | 42 passed in 217.96 s |
| tests/prereg | head | 124 passed, 1 skipped in 59.39 s |
| tests/test_gate_task_split.py, tests/test_legacy_retired.py | head | 26 passed in 3.14 s |
| tests/x13 (wcd-kbet) | head | 17 passed |
| `check_tests` for I094 (`cluster/`) and I008 (`experiments/X1_jhpce`) | 46d79f2, 2b52542 | stopped by an internal error of the coverage plug-in (see section 8); replaced by the changed-line coverage lines in 4.5 and 4.6 |

The result column is read from `logs/`. The suites that exercise the six commits all pass on the head; every new test also fails on the tree just before its fix (section 3, row 4).

## 8. Limits and deviations from the brief

- All fits ran on CPU with 1 thread on toy or 1,200-cell data (GPU busy). Bit-identity is therefore shown for the code paths on CPU, not for GPU kernels; the planned re-run of the cross-host GPU gate at the new tag (final review CR2-01) remains the GPU evidence.
- Scorer neutrality was measured on 2 of the 1,701 stored latents (atac_small; one with 2 forced kBET labels, one ordinary); the identity audit covers all 1,701 latents and the manifest audit all 41,583 rows.
- Sinkhorn numerics are out of scope (separate review); the bit-identity comparison to the tag therefore excludes the Sinkhorn arm, and the comparison to the commit before the guards includes it.
- The repository's changed-line gate (`scripts/check_tests.py`) ran to completion and passed for I033, I034, I042 and I050+I051. For I094 and I008 it stopped with `pytest exited 3` after an internal error of the coverage plug-in (`no such table: context`, raised when tests that start subprocesses leave coverage data files; my own plug-in install, not the repository's). I measured the changed lines with plain coverage instead (4.5 and 4.6), including `scripts/run_stage.py` in subprocesses through a start-up hook, and used mutants as the stronger check.
- The crash-exposure numbers in G-01 are provisional (derived from the verifier's per-row probabilities).
- No cluster command was issued; the harness's `TimeoutCeiling` value is unknown.
- This project has no rigor ledger (`rigor_ledger.json`); I did not start one, because the brief asked for this report only. Findings are numbered G-01 to G-18 here.

## 9. Files (all under `review/guard_fixes_review_20261013/`)

`probes/` (probe, mutation and report scripts; `probes/mutants.json` defines every mutant), `logs/` (every output quoted above, including `prefix_tests_*` for the before-fix runs of the new tests, `cov_*` for changed-line coverage, `head_suite_*` for the suites run at the head and `audit_stored_*` for the stored-data audits), `trees/` (clones used; `fix` = 2b52542, `pre` = c88cce3, `p033` ... `p094` = trees just before each fix), `mutation_matrix.csv`, `findings.json`. To rebuild this report: `python probes/build_report.py` (reads only `logs/`).
