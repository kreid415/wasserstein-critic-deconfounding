# Handoff: final-review fixes and Sinkhorn refit

**Session:** 2026-10-13 · 17,704 messages · 84 folds · frame a0b87862-8454-468c-a2f9-6326cd1433fc
**Status:** The final blinded design-and-code review is finished and reported. The user decided to fix the Sinkhorn arm and redo its 234 pilot fits (SI-51), and to draft every other fix and amendment for their review before any new tag (SI-52). Nothing has been fixed yet, and nothing is running.

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes.md (artifact id 041f43b0-04f6-49bf-9ca8-229e5e7cba24) in this project; read it before doing anything else.
Objective: Apply the final pre-publication review's fixes to the Wasserstein-critic batch-correction benchmark, redo the 234 Sinkhorn pilot fits, then launch X1 (the main benchmark).
Next action: Create branch final-review-fixes from main 5cb75c0 and fix the Sinkhorn loss (alignment.py), testing it against review/esc2/sinkhorn_astra/sinkhorn_reference.py.
Open first: REVIEW_FINAL.md (35db7a05-58a5-4042-bb76-2a14f64dae12); verified_issues.csv (457d1e04-1317-4d65-b662-ce7cce9f713c); CONSTRAINTS.md (SI-51, SI-52)
Lost with the old kernel: nothing expensive; reload the lab-notebook and standing-instructions skills
Pending: local X1 and filler scoring part A paused; JHPCE X1 not submitted; no live jobs
Do not redo: keeping Sinkhorn fits at effective weight 2x lambda (rejected); the blinded review itself (finished: 14 of 15 planted errors caught); no new tag before the user sees the diff (SI-52)
Before reusing any identifier, number or quote from the previous chat (frame a0b87862-8454-468c-a2f9-6326cd1433fc), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics), under a pre-registration (docs/PREREG.md; tag prereg-tier12-v2 = c88cce3). The user requires that the final check before the publication experiments be "full proof". The next phase applies the review's fixes, redoes the Sinkhorn pilots, and then launches X1, the main benchmark: 3,000 fits, 1,125 run locally and 1,800 on JHPCE.

## State
- **Done and verified:** final review report {{artifact:35db7a05-58a5-4042-bb76-2a14f64dae12}} (repo docs/review_final/, commit 4988360).
  - 99 unique issues: 17 fix before launch, 33 fix before analysis, 49 notes. They are in verified_issues.csv; each has a status, an executed check, a fix and its impact on completed work.
  - Planted-error test: 14 of 15 caught.
  - Sinkhorn: both escalated checks agree the 36 A3 fits are invalid.
- **Decided, not started:** SI-51 and SI-52 (CONSTRAINTS.md {{artifact:88425625-8f85-4ab7-86d4-630383588472}}; notebook NB-20261013-02; main 5cb75c0, pushed).
- **Paused:**
  - Local X1 (cs-local job 3264ed84 was cancelled on 2026-10-12; 19 scored, 1 fitted but unscored, 1,097 need a fit; output dir tier12_v2/X1_local). It resumes only at the new tag, after the X1 manifest's Sinkhorn grid is re-frozen.
  - Filler scoring part A (scripts run_scoring.sh A; 1 stale claim, so it needs --clear-foreign-claims).
- **Not submitted:** the 5 JHPCE X1 jobs (experiments/X1_jhpce/, SUBMISSION_PLAN.md).
- **Live runs:** none. No compute jobs, no sub-agents.

## Artifacts
- [REVIEW_FINAL.md]({{artifact:35db7a05-58a5-4042-bb76-2a14f64dae12}}): the final review report and verdict (canonical)
- [verified_issues.csv]({{artifact:457d1e04-1317-4d65-b662-ce7cce9f713c}}): 99 issues with status, evidence, fix and impact; the work list for the fixes
- [catch_matrix.csv]({{artifact:17b0c492-4c61-4080-802f-ff5ecdc0b4e7}}): planted-error catches per reviewer
- [CONSTRAINTS.md]({{artifact:88425625-8f85-4ab7-86d4-630383588472}}): standing-rules ledger through SI-52
- [X1_launch_review_prompt.md]({{artifact:ca5b6786-67b6-4e40-9da1-8dfeda831c31}}): the review prompt (scope used)

## Where things are (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding)
- Repo main clone: workspace scratch/wt_int (may be swept; re-clone github.com/kreid415/wasserstein-critic-deconfounding). Tag checkout: D/code/wcd_prereg-tier12-v2 (c88cce3, clean). Clean main clone used by the review: D/review/real_main (99e942e).
- Review outputs: D/review/out (16 reports), D/review/verify (5 Claude verifiers + codex_hi), D/review/rereview (design_fable, code_opus), D/review/esc2 (sinkhorn_fable, sinkhorn_astra), D/review/lead (catch_matrix.csv, issues.json, verified_issues.csv, all_findings_extracted.csv). Sealed key: D/review_sealed/planted_key.json (sha256 502bc9f3…).
- Sinkhorn evidence and reproduction scripts: D/review/esc2/sinkhorn_fable/SINKHORN_CHECK.md and D/review/esc2/sinkhorn_astra/ (SINKHORN_CHECK.md, sinkhorn_reference.py: a converged float64 reference).

## Kernel state (will be lost)
Nothing expensive is held only in memory: every table above is on disk. The lab-notebook, standing-instructions, preflight and session-handoff helpers come from skills; reload them with the skill tool.

## Decisions
- SI-51: fix the Sinkhorn gradient (no double-counted final transform) and the final-ε convergence; redo 234 fits (A1 180, R1 extension 18, A3 36); re-run the Sinkhorn parts of R1, R4-a1 and R3. Measured cost of the old fits: 113 lane-hours, about 14 h at 8 lanes, plus 32 scorer-hours. The corrected solver will be slower.
- SI-52: draft everything on a branch with tests. Prove the non-Sinkhorn fits and scores bit-identical to c88cce3, as at the 2026-10-04 merge. Give each fix an independent review. Show the user the diff before tagging.
- Earlier decisions still in force: all scoring is local; local X1 runs as a cs-local host job; the JHPCE go is asked separately, after its preflight.

## Dead ends
- Keeping the Sinkhorn fits at "effective weight 2λ": rejected. A3 directions are wrong (cosine 0.81–0.96), 22 fits are float32-degenerate, and early-training gradients aren't a constant multiple either.
- One model call that labelled findings during grouping was refused by the safety filter on ordinary review text. It was not resent; keyword rules were used instead, which affected grouping only.

## Next steps
1. Load the ponytail skill and follow the fewest-tests rule (profile). Create branch `final-review-fixes` from main 5cb75c0.
2. Sinkhorn (src/wcd_vae/wcd/alignment.py:145-158): detach one cost argument per potential, as geomloss does; converge the final ε, with a tolerance and an iteration cap; use stable distances in place of the float32 squared-distance expansion in `_sqdist`. Test against esc2/sinkhorn_astra/sinkhorn_reference.py: value, gradient, the Dirac case, and an A3 standardised case.
3. Guards, each with one test (see verified_issues.csv rows for exact lines and fixes): I033 (graph-carrying zero in the inactive-batch branches), I034, I042, I050, I051, I094. Launch files: I008 (per-chain-position clock) and CR2-01 (L40S gate re-run at the new tag).
4. Bit-identity: rerun the 2026-10-04 identity procedure (experiments/integration_checks_20261004) on every non-Sinkhorn arm and on scoring, old tag vs branch.
5. Draft the document amendments, before any X1 score is read: I072 (b* anchoring), I006 (grid-edge rule), I091 (R1 trajectory note), D-01 (P1 comparator), D-02 (power claim), and the P1–P4 analysis code (I001). Give each an independent RIGOR_REVIEWER review, then show the user the diff and wait.
6. After sign-off and the new tag: preflight, then refit the 234 Sinkhorn fits locally as a cs-local host job; re-run freeze_a1_grid.py (R1), freeze_matched_lambda.py (R4-a1) and decide_a2_a3.py (R3) for the Sinkhorn parts; rebuild the X1 manifest; resume local X1 and part A; ask the JHPCE go.

## Open questions for the user
- R1 trajectory note (I091): keep the registered grids and disclose the sensitivity, or amend R1 to compute σ_c without trajectory (about 90 extra A1 fits, estimated)?
- P1 comparator (D-01): which discriminator setting replaces the forbidden 10-step comparator?
- Follow-ups X7, X8 and X12 need redesign before they launch (do not block X1).
