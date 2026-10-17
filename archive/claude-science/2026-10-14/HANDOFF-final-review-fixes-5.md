# Handoff: final-review fixes, part 5

**Session:** 2026-10-13 evening (EDT) · about 130 messages · 0 folds · frame 9a6f1964-cabe-4419-b5d9-f386b46d37aa
**Status:** Step 2 (analysis specification for the planned comparisons P1-P4) is DRAFTED; the five round-1 questions are written but NOT yet asked. The identity proof (step 5, run early) PASSED at c3fd92d. No tag. Nothing is running. Handoff triggered by size (334k tokens of context, trigger 280k).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-5.md (artifact id 1901bbd5-aaf7-4515-8940-ac0790de276f) in this project; read it before doing anything else.
Objective: finish the final-review fixes, show the user the diff, tag prereg-tier12-v3, launch X1.
Next action: ask the five round-1 questions in step2_round1_questions.md with ask_user.
Open first: step2_round1_questions.md; ANALYSIS_SPEC.md
Lost with the old kernel: nothing
Pending: round-1 questions not asked
Do not redo: identity proof at c3fd92d (passed); SI-60 to SI-67
Before reusing any identifier, number or quote from the previous chat (frame 9a6f1964-cabe-4419-b5d9-f386b46d37aa), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md). Finish the final-review fixes on branch `final-review-fixes`, show the user the full diff (SI-52), create tag `prereg-tier12-v3` (SI-60), redo the Sinkhorn pilots (SI-57), then launch X1. The user wants plain language (SI-67, profile NO-JARGON rule): say what each step is for, spell out every label, give a concrete example before asking.

## State
- Repository: D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD b2e3eb2, pushed (ls-remote checked). Untracked strays in the clone: rf_extract.txt, HANDOFF-final-review-fixes-4.md (never `git add -A`; only the user can delete them).
- Done this session:
  - docs/ANALYSIS_SPEC.md DRAFT (commit 6d0ca1f): the registered rules P1 inherits, plus open choices Q1-Q11 with a recommended option and reason each (sections 0-14). Not signed off.
  - Round-1 question texts (Q1 scope, Q2 reading rule, Q3 decoders, Q4 failed runs, Q5 trade-off-area corner), plain words, ready for ask_user: step2_round1_questions.md. NOT ASKED.
  - Identity proof re-run at c3fd92d (old = D/code/wcd_prereg-tier12-v2 at c88cce3; new = D/code/wt_identity, moved from 2b52542 to c3fd92d): fits 59 of 59 non-Sinkhorn identical, 4 of 4 Sinkhorn differ, all exit 0 (NB-20261013-39; D/review/identity_c3fd92d_20261014/fits/fit_identity.csv). Valid for the tag only if the diff from c3fd92d to the candidate commit changes no fit or score code.
  - Identity proof score half: 7 of 7 latents old = new in all 32 compared columns; new = production rows apart from score_host on the local X1 latent, as before (NB-20261013-40). Results committed (ddb113a): experiments/integration_checks_20261013/results/{fit,score}_identity_c3fd92d.csv, run_c3fd92d.sh; README table updated.
  - Notebook NB-20261013-37 (EXPERIMENT), -38 (CORRECTION of a date), -39 (RESULT, fits), -40 (RESULT, scores).
- No live runs.

## Artifacts
- [ANALYSIS_SPEC.md]({{artifact:2d287ad5-95bc-4019-905a-7ef3246966d9}}) — the draft specification (= repo docs/ANALYSIS_SPEC.md at 6d0ca1f); update it after each answer.
- [step2_round1_questions.md]({{artifact:fd03248f-2326-4a06-9dd1-d948056e5db3}}) — the five round-1 questions, options and pros/cons, ready to paste into ask_user.
- [notebook_2026-10-13.md]({{artifact:ade7b754-d130-4e29-9aac-7b07756b2b4e}}) and [notebook_INDEX.md]({{artifact:c58d5654-56c4-4222-8f57-5e1f95a66fc4}}) — lab notebook to NB-20261013-40.
- [CONSTRAINTS.md]({{artifact:10282a58-039e-4b91-8ec5-aaf6d5d364c6}}) — rule ledger SI-1..SI-67, unchanged this session.
- [HANDOFF-final-review-fixes-4.md]({{artifact:d340c246-d968-4cb3-ba57-1dc53077c8b5}}) — previous handoff; its steps 3-11 still apply.

## Kernel state (will be lost)
Nothing expensive. Pilot (A1) frames rebuild in seconds: `M = prereg_rules.read_manifest([R + '/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv'])` (includes the A1 extension rows), `S = read_scores(D + '/tier12_v2/A1/scores')`, `F = read_failures(D + '/tier12_v2/A1/failures.csv')` (0 A1 failures). Skill helpers reload with the skills (standing-instructions, lab-notebook, session-handoff, ponytail, experiment-preflight).

## Decisions
- No new user decisions this session.
- Agent: ran the identity proof early (before the independent review) because it needs no step-2 answer and the machine was idle; re-run only if fit or score code changes after c3fd92d (src/, scripts/fit_paper_config.py, scripts/scvi_adversarial_plan.py, scripts/score_scib_native.py and what they import).
- Spec recommendations awaiting the user: Q1 lock every X1/X13 analysis now, follow-ups (X3, X6, X7, X8, X12, X15) later with the X7/X8 redesign and before R4-x1, premise test R0 and downstream panel X10 exploratory; Q2 average the 5 runs first, leave-one-seed-out uncertainty; Q3 one P1 result per decoder, both primary; Q4 failed runs left out (PREREG section 1), counted in P2; Q5 area corner (b0, c0 - 0.10), area divided by (b_common - b0). Round 2: Q6 family term, priors N(0, 0.1^2) and half-normal(0.1), exact grid integration; Q7 margin r = 0.01; Q9 P2 rate plus pooled difference with a 5-point margin; Q10 local-global reformulated (at matched batch removal 3 x local difference + 2 x global difference = 0 by construction); Q11 selection score S = 0.4 B + 0.6 C unscaled.
- b* for P1 is read from the frozen R4-x1 record, never recomputed; R4-x1 extension rows are not used in P1 or the area (SI-55).

## Dead ends
- `prereg_rules.outcomes` on A1 now raises: SI-54 refuses the c88cce3 Sinkhorn rows. By design; descriptive A1 summaries must drop Sinkhorn or bypass `outcomes`.
- docs/prereg/r4_a1_record.json input paths point to a deleted workspace; use the committed manifests (combining v3.r1.tsv with a1_extension_*.tsv duplicates tags; v3.tsv alone is complete).
- The system date is UTC; the workstation and notebook use America/New_York (an entry first said 2026-10-14; NB-20261013-38 corrects it).

## Next steps
1. Ask round 1 with ask_user exactly as in step2_round1_questions.md (5 tabs, recommended first). After the answers: `si_add` each choice (option text as the source), `nb_entry` DECISION each, save CONSTRAINTS.md as a new artifact version, update docs/ANALYSIS_SPEC.md.
2. Ask round 2: spec sections 4 (dataset structure), 5 (margin r), 6 (P2 rule), 7 (local-global), 8 (selection score S). Same recording. Proposed defaults the user only sees at the diff review: marginal declarations with Holm in the Wilcoxon companion, transfer lambda leave-one-family-out with failure-free eligibility, best-of-k at equal k, window width within the best point's 95% lower bound, prior sensitivity 0.05 and 0.2.
3. Write scripts/analysis_x1.py and tests/analysis/ (known-answer tests listed in spec section 14), reusing prereg_rules (`outcomes`, `bio_at_bstar`, `write_record`). Load ponytail and fail-loud-pipelines first; consider codex-offload with the known-answer tests as acceptance (AUTO-OFFLOAD rule); then check_tests.py (env scvi-api with a coverage venv) and fl_lint, ponytail-review, notebook RESULT, commit, push.
4. Continue with steps 2-6 of HANDOFF-final-review-fixes-4.md (PREREG section 9 entries, now also a dated pointer in PAPER_PLAN section 5 to ANALYSIS_SPEC.md; RIGOR_REVIEWER review; user diff review 5cb75c0..HEAD; tag; post-tag launch steps).

## Open questions for the user
- Round 1 (Q1-Q5) and round 2 (Q6, Q7, Q9, Q10, Q11) of docs/ANALYSIS_SPEC.md.
- Carried over: wording of the I091, D-01 and D-02 amendments; wording of SI-66's consequence; the user's go for each cluster job (CR2-01 L40S check; JHPCE X1).
