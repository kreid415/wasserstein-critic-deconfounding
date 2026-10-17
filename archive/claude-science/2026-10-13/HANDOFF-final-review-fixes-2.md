# Handoff: final-review fixes, part 2

**Session:** 2026-10-13 · 641 messages · 1 fold · frame baa34fcd-1c91-49e8-9dbc-4886fdc87614
**Status:** Sinkhorn fix and six guard fixes committed and independently reviewed (PASS WITH FIXES); non-Sinkhorn fits and scores proven bit-identical to the tag; seven user decisions recorded (SI-53..SI-59). Review fixes, float32 validation, the provenance-equivalence code, the remaining amendments and the P1-P4 analysis code are not started. No tag yet; nothing is running.

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-2.md (artifact id fdef5767-01eb-4634-8f37-07a5ba1e85fc) in this project; read it before doing anything else.
Objective: finish the final-review fixes on branch final-review-fixes, redo Sinkhorn in rule order (SI-57), show the user the diff, then tag and launch X1.
Next action: float32 validation (SI-58), cap 500,000 (SI-59) and review fixes F1, F4, F7 in alignment.py, then a re-review.
Open first: CONSTRAINTS.md; review_findings_sinkhorn_and_guards_20261013.json
Lost with the old kernel: nothing to reload
Pending: no runs or sub-agents; review findings F1-F8 and G-01 to G-18 open
Do not redo: identity proof (done); decisions SI-53 to SI-59
Before reusing any identifier, number or quote from the previous chat (frame baa34fcd-1c91-49e8-9dbc-4886fdc87614), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md, tag prereg-tier12-v2 = c88cce3). Apply the final pre-publication review's fixes on branch `final-review-fixes`, redo the Sinkhorn arm (SI-51, in rule order SI-57), prove that everything else is unchanged, show the user the full diff, then make a new tag and launch X1 (3,000 fits: 1,125 local, 1,800 JHPCE, 75 JHPCE baseline fits already made at c88cce3).

## State
Repository: clone `D/code/wcd_final-review-fixes` (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD 82263e5, pushed to github kreid415/wasserstein-critic-deconfounding; 28+ commits since 5cb75c0, working tree clean.
- Done and verified: Sinkhorn fix 3cbaca4 (gradient through the row point only; final-epsilon solve to L1 marginal error 1e-4; cap 50,000 then raise; float64), validated on 3,076 minibatches against the review's float64 reference (worst gradient cosine 0.999999721); guards c4be14b (I033), a4f4be0 (I034), 5f30f17 (I042), f5ea0c3 (I050+I051), 46d79f2 (I094), 2b52542 (I008).
- Done and verified: identity proof, experiments/integration_checks_20261013 (da1ded4): fit identity 59 of 59 non-Sinkhorn code paths give equal latents, 4 of 4 Sinkhorn rows differ; all suites pass at 2b52542 (569 passed, 1 skipped); score identity 7 of 7 latents old = new = the stored production rows, with the runner's thread settings (NB-20261013-12, -16).
- Done and verified: pilot fit in the hardest regime (A3 immune, standardised, lambda 3): 56,404 steps, max 3,786 solver iterations per step, 7,778 s vs 2,793.7 s pre-fix, loss 85% of the time (NB-20261013-11).
- Done and verified: R4-a1 sensitivity (NB-20261013-09): the refitted Sinkhorn arm's best batch score M may fall by only 0.011 (atac_small, conditioned decoder) or 0.010 (immune, conditioned) before the MMD arm's matched lambda, used by A3 rows, moves; 0.069 and 0.132 in the unconditioned blocks (pooled).
- Drafted, not yet reviewed (018c8f1): PREREG section 9 entries for I091 (R1 trajectory correction, numbers reproduced by experiments/final_review_amendments_20261013/r1_without_trajectory.py) and for the PAPER_PLAN corrections (I006 grid edge, D-01 P1 comparator, D-02 power and frequentist companion); PAPER_PLAN lines 70, 91, 154, 169, 173, 177, 184 edited. The power numbers are from the frozen R4-a1 and must be recomputed after the redo.
- Reviews done (CPU only, subsampled; declared limits in each report): Sinkhorn fix PASS WITH FIXES; guards I033/I034/I042 PASS WITH FIXES, I050+I051/I094/I008 PASS. No finding addressed yet.
- Paused, untouched: local X1 (20 latents, 19 scored), filler part A; JHPCE X1 not submitted.
- No live runs, no sub-agents.

## Artifacts
- [CONSTRAINTS.md]({{artifact:61319a0e-6c23-4cd2-9134-1b391fbef8d7}}) — standing-instructions ledger v23, 59 rules; SI-51..SI-59 are this review round (also in the repo root).
- [review_findings_sinkhorn_and_guards_20261013.json]({{artifact:5ee08839-e3b8-42b6-904e-3346fabbc320}}) — both reviewers' structured outputs (verdicts, every finding with evidence and suggested fix, deviations).
- [REVIEW_GUARD_FIXES.md]({{artifact:9c6d8e2d-67b2-4b63-bf39-9f21059f1680}}) — independent review of the six guard commits (verdict per fix, findings G-01..G-18)  <!-- text/markdown, 72 KB -->
- [findings.json]({{artifact:9c5c744b-def3-4920-bc6f-535ed617af8e}}) — guard review findings, machine-readable  <!-- application/json, 34 KB -->
- [mutation_matrix.csv]({{artifact:0d008ef5-b032-41d7-9e60-d31937031e9a}}) — guard review: 34 one-line mutants and which tests catch them  <!-- text/csv, 5 KB -->
- [REVIEW_SINKHORN_FIX.md]({{artifact:283ca295-7d17-4913-afe5-eec7e3c65e92}}) — independent review of the Sinkhorn fix at 0380465 (findings F1..F8)  <!-- text/markdown, 44 KB -->
- [fig_iteration_tail.png]({{artifact:09037af9-ed54-4e71-831e-250a98018f67}}) — Sinkhorn review: iteration tail on pre-fix A3 latents (F2)  <!-- image/png, 196 KB -->
- [REVIEW_FINAL.md]({{artifact:35db7a05-58a5-4042-bb76-2a14f64dae12}}) and [verified_issues.csv]({{artifact:457d1e04-1317-4d65-b662-ce7cce9f713c}}) — the final pre-publication review and its 99 issues (I001.., D-01..D-09 are in D/review/rereview/design_fable/REVIEW_design_recheck.md).
- [HANDOFF-final-review-fixes.md]({{artifact:aa637ca2-14b7-454a-9d51-7653ca16e912}}) — previous handoff (part 1).

## Kernel state (will be lost)
| in memory | rebuild cost | how to restore |
|---|---|---|
| skill helpers si_*, nb_*, handoff_* | trivial | load skills standing-instructions, lab-notebook, session-handoff |
| `res` (review outputs) | trivial | json.load the review_findings artifact above |
| coverage venvs dev/covenv, dev/covenv_scvi, dev/covenv_kbet (workspace) | cheap | `python -m venv --system-site-packages` on wcd-gpu / scvi-api / wcd-kbet, then `pip install pytest-cov==7.1.0` |

Git: .git/config is not writable (EBUSY); export GIT_AUTHOR_/GIT_COMMITTER_ NAME=wcd EMAIL=noreply@localhost and push with the token URL on the command line. `**/results/` is gitignored; result files are force-added (`git add -f`), as 31 earlier files were. Envs: scvi-api (fits), wcd-gpu (tests/wcd_vae, tests/stage, rule scripts), wcd-kbet (scoring; R_HOME=$C/wcd-kbet/lib/R R_LIBS=D/Rlib_kbet, C = /home/kendall/.claude-science/conda/envs). Prefix KMP_AFFINITY=disabled OMP_NUM_THREADS=1.

## Decisions
- SI-53 (I091): keep the registered R1 and the frozen X1 grids; disclose the trajectory-free R1 as a sensitivity analysis.
- SI-54: declare c88cce3 and the new tag result-identical for non-Sinkhorn arms in an explicit committed list; rule scripts and runner accept either commit for those arms only; Sinkhorn rows must carry the new tag; no rescoring.
- SI-55 (I006): PAPER_PLAN corrected to PREREG section 4 (done, 018c8f1).
- SI-56 (I072, D-04): P1 curves anchored at lambda = 0 (the unintegrated model's scores that define b0); bstar_low_signal cells are not evaluable for P1 and R2/R3, reported separately. With the frozen values this changes neither recorded decision (R2 mean D -0.0042 over 9 cells, R3 -0.0014 over 13, both below the ROPE 0.01; if the cells are kept but not evaluable, coverage fails: no switch either).
- SI-57: Sinkhorn redo in rule order: 180 A1 refits, R1 for the Sinkhorn family plus the extension rows it asks for, R4-a1 over all six arms, the A3 rows it resolves (including any non-Sinkhorn cell whose window moved), then R2 and R3 in full. Not a fixed list of 234 rows.
- SI-58: validate a float32 solve on the 598 hardest minibatches and the GPU timing; adopt only if worst gradient cosine >= 0.99999 and value error <= 1e-4; then tests and an independent review.
- SI-59: SINKHORN_MAX_ITER = 500,000, a hit still raises (about 20 s per 50,000 iterations on the RTX 3080 in float64); with F1 a non-finite marginal error ends the loop at once.
- Agent: identity rows exclude retired pre-SI-41 X3 rows; score identity must use run_stage.thread_env(1) settings and META.

## Dead ends
- Scoring without OPENBLAS_NUM_THREADS=1 and MKL_THREADING_LAYER=SEQUENTIAL differs from production rows by 1e-9 to 2e-6 (PCA metrics). Use experiments/integration_checks_20261013/score_identity.sh.
- `pkill -f PATTERN` in a bash cell kills the cell itself when its own command line contains PATTERN (exit 143).
- scripts/check_arm_bitidentity.py cannot run against c88cce3 (review G-18); use fit_identity.sh.
- From part 1: keeping the old Sinkhorn fits as "effective weight 2 lambda" was rejected.

## Next steps
1. Sinkhorn code round in src/wcd_vae/wcd/alignment.py: float32 validation per SI-58 (dtype switch in tol_sweep.py / validate_sinkhorn.py / bench_gpu.py, GPU free now); cap 500,000 and its comment (SI-59); F1 (`if not torch.isfinite(err) or err < tol: break` in both loops, plus a NaN/inf row in test_sinkhorn_refuses_bad_p_and_unconverged_solve); F4 test pins (tolerance, dtype); F7 docstring (debiased Sinkhorn divergence at epsilon, not W1); F8 note. Update experiments/sinkhorn_fix_20261013 README and NB-20261013-04. Then a RIGOR_REVIEWER re-review.
2. Guard follow-ups: G-01 (count no-pair steps; write to the saved config only when > 0, so other outputs stay byte-identical; one PREREG sentence naming X7 reference-MMD on pancreas and sim1), G-02 (parametrize test_nonfinite_inner_critic_step_raises over step and term), G-03 (subprocess case in tests/scvi/test_nonfinite_guard.py section 6, then remove the pragma at fit_paper_config.py:297); consider G-05, G-08, G-14. check_tests.py per fix; rerun fit_identity.sh at the final commit.
3. SI-54 code: equivalence list read by prereg_rules.check_provenance (prereg_rules.py:219) and run_stage.py (:674, :1085, :1153), non-Sinkhorn arms only; tests; PREREG section 9 entry.
4. P1-P4 specification and frozen code with known-answer tests (I001, D-03, D-04/SI-56, D-08, D-09); define whether not-evaluable cells leave the R2/R3 cell set.
5. Remaining PREREG section 9 entries (SI-51/SI-57, guards, SI-58/59, SI-54, SI-56) and Methods notes (F6 validation scope, G-07 sysVI likelihood labels). RIGOR_REVIEWER review of all amendments.
6. Show the user the full diff 5cb75c0..HEAD (SI-52). After sign-off: new tag; x1_jhpce_command.py TAG_SHA (G-17); TimeoutCeiling for 384/576 h (G-15); experiment-preflight + singlecell pack; Sinkhorn redo per SI-57 as a cs-local host job; CR2-01 L40S gate (needs the user's go); resume local X1 and filler part A (--clear-foreign-claims); ask for the JHPCE go.

## Open questions for the user
- None blocking now. At the diff review: amendment wording (D-01, D-02 reporting rule, I091 note), the I042 allow-list difference from the review's suggestion, the refusal of legacy latent writers (I050), and the scib 1.1.7 NMI-0 edge case (loud error, not fixed).
