# Handoff: final-review fixes, part 4

**Session:** 2026-10-13 evening · 269 messages · 0 folds · frame ebf81ab2-76c3-4a75-b159-6c152bb1c2ec
**Status:** Steps before the X1 launch are being worked through in order. Done this session: re-review notes R2-04/05/06/08, the SI-54 commit-equivalence code, six user decisions, and retiring two old pilot scripts (step 1). Step 2 (the P1-P4 analysis specification and code) is NOT started beyond reading the review findings. No tag. Nothing is running.

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-4.md (artifact id a79f73e9-7a6a-4f4b-a0ef-218d6c2dd5f3) in this project; read it before doing anything else.
Objective: finish the final-review fixes on branch final-review-fixes, show the user the diff, then tag prereg-tier12-v3 and launch X1.
Next action: step 2: draft the P1-P4 analysis specification with each open choice and a recommendation, ask the user in plain words, then write and lock the code.
Open first: CONSTRAINTS.md; REVIEW_design_recheck.md (D-03, D-08, D-09)
Lost with the old kernel: nothing to reload
Pending: no runs or sub-agents; step 2 not started
Do not redo: SI-54 code (done, tested); decisions SI-60 to SI-66 (recorded)
Before reusing any identifier, number or quote from the previous chat (frame ebf81ab2-76c3-4a75-b159-6c152bb1c2ec), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md). Finish the final-review fixes on branch `final-review-fixes`, show the user the full diff (SI-52), create tag `prereg-tier12-v3` (SI-60), redo the Sinkhorn pilots (SI-57), then launch X1. The user wants plain language (SI-67, profile NO-JARGON rule): explain what each step entails and why, spell out every label.

## State
Repository: clone D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD c3fd92d, pushed to github kreid415/wasserstein-critic-deconfounding. Tree clean except `rf_extract.txt` (untracked scratch file; the sandbox refused to delete it; the user was asked to delete it; never `git add -A`).
- Done and verified this session:
  - R2-04/05/06/08 (5d0fe9a; CORRECTION NB-20261013-27): cap pinned by a test assertion; float32 "no speed-up" scoped to 128-cell minibatches; cap comment sample sizes recounted from the data (480 standardised minibatches max 5,273; 10,800 atac_small max 9,789).
  - SI-54 (8ce5a54, 7c32579; PREREG section 9 entry b2c68c8; RESULT NB-20261013-29): docs/EQUIVALENT_COMMITS.json says c88cce3 (tag prereg-tier12-v2) is result-identical to prereg-tier12-v3 for every arm except sinkhorn. prereg_rules.refuse_uncovered and equivalent_commits are used by check_provenance (via outcomes) and by run_stage's preflight (refuses existing Sinkhorn scores or latents at c88cce3, report-only too), gate scorer provenance and latent git_sha gate. If the tag is absent from a checkout the mix stays and is refused. 9 of 9 planted bugs caught; tests/prereg + tests/stage + tests/jhpce 237 passed, 2 skipped; check_tests.py --base 51ea7a7 --src scripts OK (env scvi-api, coverage venv dev/covenv_scvi); fl_lint clean.
  - Step 1, SI-64 (c3fd92d): fit_stock_scvi.py and scvi_fit_pancreas.py moved to legacy/ with the RETIRED guard line; tests/test_legacy_retired.py list extended (22 passed).
- NOT independently reviewed yet: guard follow-ups G-01..G-14 (96f4258..43a0f91), SI-54 code, step 1. Goes to the RIGOR_REVIEWER review (step 4).
- Identity proof (fit_identity.sh, score_identity.sh in experiments/integration_checks_20261013) last ran at 2b52542, before G-01 changed the MMD signatures; must re-run at the final candidate commit (about 35 min fits + 1-2 h scores, local).
- Paused, untouched: local X1 (20 latents, 19 scored), filler part A; JHPCE X1 not submitted.

## Artifacts
- [CONSTRAINTS.md]({{artifact:10282a58-039e-4b91-8ec5-aaf6d5d364c6}}) — rule ledger, SI-1..SI-67 (SI-59 retired, replaced by SI-61); also at the repo root.
- [notebook_2026-10-13.md]({{artifact:86cbbf7f-257f-4921-be6e-8dd21b71a36e}}) and [notebook_INDEX.md]({{artifact:ad7255f7-003e-4b4f-bb1f-480f21bcd786}}) — lab notebook to NB-20261013-36.
- [EQUIVALENT_COMMITS.json]({{artifact:a0342f9f-2fae-42aa-899a-0a5da026aea6}}) — the SI-54 list.
- [REREVIEW_SINKHORN_ROUND2.md]({{artifact:8617428c-646a-4060-ac83-db82581de280}}) — re-review of the Sinkhorn round (R2-01..R2-09; all addressed or no action).
- [HANDOFF-final-review-fixes-3.md]({{artifact:0c2fd11f-8831-41b1-9a24-3b5cde2cf5dd}}) — previous handoff (part 3).
- Review sources for step 2 (on disk, not artifacts): D/docs/review_final/REVIEW_FINAL.md in the repo; D/review/lead/verified_issues.csv (I001, I056, I062); D/review/rereview/design_fable/REVIEW_design_recheck.md (D-03, D-04, D-08, D-09, D-10).

## Kernel state (will be lost)
| in memory | rebuild cost | how to restore |
|---|---|---|
| skill helpers (si_*, nb_*, fl_*, handoff_*) | trivial | load skills standing-instructions, lab-notebook, fail-loud-pipelines, session-handoff, ponytail |
| coverage venv dev/covenv_scvi (scvi-api + pytest-cov 7.1.0) in the workspace | cheap | `python -m venv --system-site-packages dev/covenv_scvi` in env scvi-api, then `dev/covenv_scvi/bin/pip install pytest-cov==7.1.0` |
| worktree metadata wt_si54 | none | `git worktree prune` once the workspace is swept (removal failed: Device or resource busy) |

Git: export GIT_AUTHOR_/GIT_COMMITTER_ NAME=wcd EMAIL=noreply@localhost; push with the token URL. Envs: scvi-api (tests/prereg, tests/stage, tests/jhpce, fits), wcd-gpu (tests/wcd_vae with PYTHONPATH=src), wcd-kbet (scoring). Prefix KMP_AFFINITY=disabled OMP_NUM_THREADS=1 (wcd-kbet aborts without KMP_AFFINITY=disabled). Edit host files with edit_file; never rm there.

## Decisions
User, this session (ledger and notebook ids):
- SI-60 / NB-28: new tag name prereg-tier12-v3.
- SI-61 / NB-30 (SI-59 retired): Sinkhorn solve stops on a NaN error only; +infinity keeps iterating (finite latents of about 1e18+ give +inf then converge); cap 500,000 still raises.
- SI-62 / NB-31: keep float64 (float32 met the SI-58 criterion on the 598 but no speed-up at 128 cells and fails on all 3,076).
- SI-63 / NB-32: I042 refusals as implemented (none reads decoder, likelihood, cond; scANVI reads likelihood).
- SI-64 / NB-33: old pilot writers retired to legacy/ (done, c3fd92d).
- SI-65 / NB-34: scib 1.1.7 NMI-0 edge case stays a loud error (cluster_optimal_resolution keeps no clustering when every score is 0; confirmed in source).
- SI-66 / NB-35: in R2/R3, low-signal cells (bstar_low_signal; now immune with the conditioned decoder, b_common - b0 = 0.0188 < 0.02) stay in n and count as not evaluable. Consequence: while that block is flagged, R2 and R3 cannot switch from the defaults (posterior mean; standardisation off). Must be disclosed in the PREREG amendment and Methods. The user chose this over the recommended option.
- SI-67 / NB-36: communicate more clearly (user correction).
- The 0.02 threshold = 2 x delta_min (0.01, signed off 2026-10-02); equivalently b* - b0 < 0.01. The factor 2 was in the agent's draft rules, not a separate user decision (told to the user).

## Dead ends
- delete_host_files cannot delete files in the repo folder (masked by the sandbox); `git worktree remove` fails with Device or resource busy.
- run_stage runs as a subprocess in tests/stage, so monkeypatching prereg_rules does not reach it; call `rs.main(st.args())` in-process with `monkeypatch.setenv` for each `st.env()` item (see test_si54_old_commit_counts_as_the_tag_in_the_gate).
- Writing scratch files from the python kernel after `os.chdir(R)` lands them in the user's repo. Keep scratch output in the workspace.
- The user did not follow the first explanation of the R2/R3 question (symbols, dense table). Explain with words and a concrete example first.

## Next steps
1. **Step 2: write the P1-P4 analysis specification, get the user's choices, then write and lock the code.** The review (I001, D-03, D-08, D-09, I056, I062) lists choices that must be fixed before any X1 score is read:
   - P1 (bio kept at matched batch removal, reference critic minus 1-step discriminator, SI-01/D-01): interpolation rule (seed-mean or per seed; first bracketing pair), arms that never reach b* (not evaluable, reported), several crossings on the 6-point grid (I062: isotonic fit, all crossings or upper envelope), one P1 per decoder or pooled, failure handling (PAPER_PLAN imputes the unintegrated score; PREREG uses successful seeds; imputation breaks the bracket search), SI-56 anchor at lambda = 0 and low-signal blocks reported separately.
   - Co-primary frontier hypervolume: axes (B x C) and reference point (e.g. (b0, 0)).
   - Statistical model: per-dataset contrast d_t ~ N(Delta, tau^2 + sigma^2/n_t) or score ~ arm*decoder + (1 + arm | family/dataset) + (1 | dataset:seed) (I056); Bayesian (Corani) with the ROPE (PAPER_PLAN max(0.01, 2 x measurement SD) vs PREREG 0.01: D-10 conflict); multiplicity count.
   - Transfer lambda (D-08): selection score, per family x decoder, real tasks only, tie rule.
   - Secondary endpoints (D-09): collapse at X1 (P2), local-global metric sets and sign test, X3 dose-response response variable, P3 ceiling-normalised iLISI, P4 reference coverage, best-of-k selection score; Clopper-Pearson for failure rates; Kendall tau for rank stability.
   - Scope question: P3 (X8/X9) and P4 (X7) depend on follow-ups that the final review rated NO-GO as designed (I015, I077, I078, D-06). Ask whether to specify them now or only P1/P2 before X1.
   Proposed route: draft docs/ANALYSIS_SPEC.md with every registered rule plus each open choice, a recommended option and its reason; ask the user a few choices at a time in plain words (ask_user, one per question); then write the code (scripts/, prereg_rules style, fail-loud), known-answer tests, check_tests.py, fl_lint, notebook RESULT, commit, push.
2. Step 3: remaining PREREG section 9 entries: SI-51/SI-57 (Sinkhorn fix and redo), guard follow-ups (G-01 sentence: no-pair steps can occur in X7 reference-anchored MMD on pancreas and sim1, rarely X3 atac_small MMD, X8 sim2 MMD, X1 sim1 MMD and Sinkhorn), SI-58/SI-61/SI-62, SI-56, SI-66 (with its consequence), Methods notes F6, F7, G-07.
3. Step 4: RIGOR_REVIEWER review of all amendments plus commits after 6b4aefc (profile ROUTING rule: profile=RIGOR_REVIEWER).
4. Step 5: identity proof at the candidate commit (commands in experiments/integration_checks_20261013/README.md).
5. Step 6: show the user the diff 5cb75c0..HEAD with the open questions below; after sign-off tag prereg-tier12-v3.
6. Steps 7-11 after the tag: x1_jhpce_command.py TAG_SHA (G-17); 384/576 h time-limit fallback (G-15); experiment-preflight with the single-cell pack; Sinkhorn redo per SI-57 as a local job, then the Sinkhorn parts of R1, R4-a1, R3; CR2-01 L40S check (needs the user's go); resume local X1 and filler part A (--clear-foreign-claims); ask for the JHPCE go.

## Open questions for the user
- Step 2 choices (listed under Next steps 1), including the P3/P4 scope question.
- At the diff review: wording of the I091, D-01 and D-02 amendments (PREREG section 9, commit 018c8f1); how SI-66's consequence is worded.
- The user's go for each cluster job (CR2-01 L40S check; JHPCE X1).
