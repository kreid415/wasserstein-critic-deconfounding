# Handoff: final-review fixes, part 6

**Session:** 2026-10-14 morning (EDT) · about 110 messages · 0 folds · frame 619cf68a-d6a1-4a4e-b383-f5cc411d72a0
**Status:** Step 2 (analysis specification for the X1 endpoints) questions DONE: rounds 1 and 2 answered, all recommended options (SI-68 to SI-77). Next: write the frozen analysis code and its known-answer tests. No tag. Nothing running. Handoff at the boundary between the questions and the coding phase (267k tokens of context; trigger 280k).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-6.md (artifact id 62f55ac9-97f8-4abc-af53-44016b3cdd6e) in this project; read it before doing anything else.
Objective: finish the final-review fixes, show the user the diff, tag prereg-tier12-v3, launch X1.
Next action: write scripts/analysis_x1.py and tests/analysis/ to docs/ANALYSIS_SPEC.md (all questions decided, SI-68 to SI-77).
Open first: ANALYSIS_SPEC.md; CONSTRAINTS.md; HANDOFF-final-review-fixes-4.md (steps 2-6)
Lost with the old kernel: nothing
Pending: no jobs; unasked spec details go to the diff review
Do not redo: identity proof at c3fd92d (passed); step-2 questions (SI-68 to SI-77); SI-60 to SI-67
Before reusing any identifier, number or quote from the previous chat (frame 619cf68a-d6a1-4a4e-b383-f5cc411d72a0), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md). Finish the final-review fixes on branch `final-review-fixes`, show the user the full diff (SI-52), create tag `prereg-tier12-v3` (SI-60), redo the Sinkhorn pilots (SI-57), then launch X1. Plain language throughout (SI-67, profile NO-JARGON rule): say what each step is for, spell out every label, and give a concrete example and the source of every number before asking.

## State
- Repository: D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD 990d37d, pushed (local = remote, checked with ls-remote). Untracked strays rf_extract.txt and HANDOFF-final-review-fixes-4.md: only the user can delete them; never `git add -A`; `si_scope_check` always lists them, so check `git diff --name-only b2e3eb2..HEAD` (this session's base) instead.
- Done this session:
  - Round 1 (Q1 scope, Q2 reading rule, Q3 decoders, Q4 failed runs, Q5 area corner) and round 2 (Q6 dataset structure, Q7 margin, Q9 P2 comparison, Q10 local-global, Q11 selection score) asked with ask_user; all answered with the recommended option. Ledger SI-68 to SI-77; notebook DECISIONs NB-20261014-01 to -05 and -07 to -11 (author user), linked both ways.
  - docs/ANALYSIS_SPEC.md: every question marked DECIDED with its ledger id (commits 56c0762, 990d37d). The status line names the details that were never asked (section 2.3 not-evaluable rule, section 5 multiplicity, section 8 transfer-lambda rule, sections 9-11): they go to the user at the diff review.
  - Correction: spec section 3 said the middle half of pilot points moved C by -0.19 to +0.09. Recomputed: -0.094 to +0.005; the old figure was the per-cell quartile extremes. Fixed in the spec, NOTE NB-20261014-06, memory mem_47265123b7e0.
  - Checked against primary files before asking (all reproduce): plain-scVI C 0.666-0.897 and point shifts -0.590 to +0.231 (918 non-Sinkhorn A1 rows); low-signal blocks immune|1 (b0 0.6382, b_common 0.6570) and sim1|1 (0.8120, 0.8189), where b_common is set by barycenter and mmd, so the Sinkhorn redo can lower b_common but cannot clear these flags; immune's 33,506 cells all lie inside immune_hum_mou (97,861); atac_small (11,270) shares 3,618 cells with atac_large (84,813); pilot collapse rates at the X1 strengths: reference 10/54 and 15/54, discriminator 8/45 and 17/54 (unconditioned/conditioned).
- Git: commits on this branch use the author wcd with a placeholder no-reply address. The python kernel needs GIT_AUTHOR_NAME/EMAIL and GIT_COMMITTER_NAME/EMAIL set in os.environ, or `nb_entry` silently does not commit. Push from bash with credentials=['GitHub'] and a credential helper that reads $GITHUB_TOKEN (memory mem_3d770f2ce7c5).
- No live runs and no sub-agents.

## Artifacts
- [ANALYSIS_SPEC.md]({{artifact:7dbf519e-d1f0-4f47-ba0f-e4a598ab61fa}}): the specification with every question decided (= repo docs/ANALYSIS_SPEC.md at 990d37d); the code implements it.
- [CONSTRAINTS.md]({{artifact:6da7d2d3-47fa-4625-8a3a-f10c7b9af6e9}}): rule ledger SI-1 to SI-77.
- [notebook_2026-10-14.md]({{artifact:7fb079e6-d5dd-4e76-9701-ced251d43cea}}) and [notebook_INDEX.md]({{artifact:ecf9af90-623f-4b0b-8dd9-55f09e46ff7e}}): lab notebook to NB-20261014-11.
- [HANDOFF-final-review-fixes-5.md]({{artifact:06e26374-71d1-4f26-8407-7a8d3616d275}}): previous handoff. [HANDOFF-final-review-fixes-4.md]({{artifact:d340c246-d968-4cb3-ba57-1dc53077c8b5}}): its steps 2-6 still apply.

## Kernel state (will be lost)
Nothing expensive. The A1 pilot outcomes rebuild in about 3 s: `sys.path.insert(0, R + '/scripts'); import prereg_rules as PR; M, hdr = PR.read_manifest([R + '/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv']); S, _ = PR.read_scores(D + '/tier12_v2/A1/scores'); F = PR.read_failures(D + '/tier12_v2/A1/failures.csv'); O = PR.outcomes(M[~((M.experiment == 'A1') & (M.arm == 'sinkhorn'))], S, F, ['A1'])` (918 rows, all status ok). Skill helpers reload with the skills (standing-instructions, lab-notebook, session-handoff).

## Decisions
User, 2026-10-14, ask_user, all recommended options:
- SI-68: lock every X1/X13 analysis now; lock the follow-up analyses (X3, X6, X7, X8, X12, X15) later, with the X7/X8 redesign, while X1 runs and before any X1 result is examined; R0 (premise) and X10 (downstream biology) exploratory.
- SI-69: read P1 on 5-run point means; leave-one-seed-out uncertainty; per-run reading as sensitivity.
- SI-70: one P1 result per decoder, both primary, never pooled.
- SI-71: failed runs left out of P1 and the area, counted in P2; sensitivity drops points with any failed run.
- SI-72: area corner (b0, c0 - 0.10); area divided by (b_common - b0).
- SI-73: family plus dataset terms (families pancreas, lung, immune, ATAC); priors at scale 0.1, sensitivity 0.05 and 0.2; exact grid calculation.
- SI-74: margin 0.01; no measurement-SD study.
- SI-75: P2 rates with Clopper-Pearson intervals plus a pooled critic-minus-discriminator comparison (P1 model, prior scales 0.25, margin 5 percentage points).
- SI-76: local-global as one signed question at the P1 crossing (sign test plus the P1 model, margin 0.01); PAPER_PLAN gets a dated note.
- SI-77: selection score 0.4 B + 0.6 C, unscaled.
Agent: handed off at the boundary between the questions and the coding phase.

## Dead ends
- `nb_entry` from the python kernel returns committed False without git identity variables (fix in State).
- `si_propagate` over D/review takes about 2 min and 6 GB of memory; search the repo only.
- A CORRECTION entry must correct an earlier NB id. A wrong statement that exists only in a document gets a NOTE instead (as NB-20261014-06).
- Carried over: `prereg_rules.outcomes` refuses A1 input with the c88cce3 Sinkhorn rows (SI-54, by design), so drop Sinkhorn for descriptive A1 summaries. docs/prereg/r4_a1_record.json input paths point to a deleted workspace; use the committed manifests.
- The unconditioned discriminator's X1 grid includes lambda 0.03, which has no A1 row in manifest v3 (an R1 extension point), so pilot summaries at the X1 strengths have 5 of 6 points there.

## Next steps
1. Write scripts/analysis_x1.py and tests/analysis/ exactly to docs/ANALYSIS_SPEC.md (sections 1-11; the known-answer tests listed in section 14), reusing prereg_rules (`outcomes`, `bio_at_bstar`, `write_record`). Load ponytail and fail-loud-pipelines first. AUTO-OFFLOAD rule: run `codex_canary.py --auto`, then consider offloading the implementation, with known-answer tests written first by Claude as the acceptance command. Then check_tests.py (scripts tier: env scvi-api with a coverage venv, memory mem_b73a0dfc1fe5), fl_lint, ponytail-review, notebook RESULT, commit, push. The code must not be run on any X1 adversarial-arm score before the tag.
2. Steps 2-6 of HANDOFF-final-review-fixes-4.md: PREREG section 9 entries (SI-51/57, guards, SI-58/61/62, SI-56, SI-66 with its consequence) and Methods notes F6, F7, G-07; dated pointers in PAPER_PLAN section 5 (to ANALYSIS_SPEC.md, spec section 13) and section 3 (the SI-76 rewording); RIGOR_REVIEWER review of all amendments plus every commit after 6b4aefc; re-run the identity proof only if fit or score code changed after c3fd92d; user diff review of 5cb75c0..HEAD; tag; post-tag launch steps.

## Open questions for the user
- At the diff review, the spec details never asked as questions: 2.3 (a cell is not evaluable when fewer than 4 of the 5 leave-one-seed-out readings exist); 5 (marginal declarations, Holm in the Wilcoxon companion); 8 (transfer lambda leave-one-family-out, failure-free eligibility, ties to the smaller lambda, sim1/sim2 use the lambda chosen on all 6 real datasets); 9 (best-of-k at equal k); 10 (window = points whose mean S is at least the best point's 95% lower bound); 11 (sensitivity list, including the (b0, 0) corner).
- Carried over: wording of the I091, D-01 and D-02 amendments; wording of SI-66's consequence; the user's go for each cluster job (CR2-01 L40S check; JHPCE X1).
