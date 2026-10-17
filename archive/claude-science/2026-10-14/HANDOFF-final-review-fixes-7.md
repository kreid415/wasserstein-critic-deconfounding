# Handoff: final-review fixes, part 7

**Session:** 2026-10-14 (EDT) · 395 messages · 1 fold · frame 4137e77c-9ffd-4510-944a-f9dc73e580fd
**Status:** The frozen X1 analysis code is done: scripts/analysis_x1.py (ff88666) with 53 known-answer test cases (858101b), mutation-tested and pushed. The PREREG section 9 entries and PAPER_PLAN notes are drafted (4d86a55, de741a0). Next: the independent rigour review, then the user's diff review, then the tag. Nothing running. The handoff fired on size (395 messages, 1 fold).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-7.md (artifact id b51694a9-82b3-4114-8c15-bedf25759925) in this project; read it before doing anything else.
Objective: finish the final-review fixes, show the user the diff, tag prereg-tier12-v3, launch X1.
Next action: RIGOR_REVIEWER review of all amendments and commits 6b4aefc..1e615eb; fix or rebut its findings.
Open first: ANALYSIS_SPEC.md; CONSTRAINTS.md; HANDOFF-final-review-fixes-4.md (post-tag steps)
Lost with the old kernel: nothing expensive
Pending: no jobs; diff-review items listed in the handoff
Do not redo: identity proof at c3fd92d; analysis_x1.py and its tests (ff88666, 858101b); questions SI-68 to SI-77
Before reusing any identifier, number or quote from the previous chat (frame 4137e77c-9ffd-4510-944a-f9dc73e580fd), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md). Finish the final-review fixes on branch `final-review-fixes`, show the user the full diff (SI-52), create the tag `prereg-tier12-v3` (SI-60), redo the Sinkhorn pilots (SI-57), then launch X1 (the main experiment: 8 tasks, both decoders, seeds 0-4). Use plain language throughout (SI-67 and the profile NO-JARGON rule): say what each step is for, spell out every label, and give a concrete example and the source of every number before asking.

## State
- Repository D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD 1e615eb, pushed (local = remote, checked with ls-remote). There are 98 commits since 5cb75c0 (the diff-review base) and 59 since 6b4aefc (the rigour-review base). Untracked strays rf_extract.txt and HANDOFF-final-review-fixes-4.md: only the user can delete them; never `git add -A`.
- Done and verified this session (commits after 990d37d):
  - a1d497b: ANALYSIS_SPEC section 11 now says the prior sensitivity (scales 0.05 and 0.2) applies to the effect and to both spreads, as question Q6 was asked (SI-73). New section 15 lists the implementation details the agent chose, for the diff review.
  - f3c3415 (NB-20261014-14 ISSUE, -15 RESULT): SI-56/SI-66 were recorded on 2026-10-13 but never implemented. They say that low-signal cells (batch range too small to read b*) are not evaluable for the R2/R3 switch rules and stay in the count. decide_a2_a3.py now implements this, and it is inert when no cell is flagged (test_rules_unchanged passes). Consequence: immune with the conditioned decoder stays flagged, so R2 and R3 keep their defaults (posterior mean; standardisation off). No existing R2/R3 result changes.
  - 4d86a55 and de741a0 (NB-16, NB-19): PREREG section 9 entries:
    - the Sinkhorn fix and redo in rule order (SI-51, -57, -58, -61, -62);
    - the guards I033, I034, I042, I050, I051, I094 and I008, with the G-01 no-pair counts;
    - P1 starting at lambda = 0, and low-signal cells with the SI-66 consequence;
    - the X1 analyses frozen (SI-68 to SI-77).
    Also PAPER_PLAN Methods notes F6, F7 and G-07, the R1-row rewording for SI-76, and the section 5 dated pointer with [replaced] markers. de741a0 rewords the tag sentences as a commitment ("The tag prereg-tier12-v3 freezes all of them before any X1 adversarial-arm score is examined"), answering two background Reviewer warnings (both resolved).
  - ff88666 (NB-17): scripts/analysis_x1.py (515 lines) was written by Codex (job d534623f, attempt 1, gpt-6-astra xhigh) to the brief and reviewed line by line, with three edits by Claude. It comes with the tests/analysis known-answer cases. It has been run only on synthetic inputs, never on X1 adversarial-arm scores.
  - 858101b (NB-18): mutation testing (40 of 526 one-operator mutants per sample) found three untested places: the selection-score weights 0.4/0.6, the 4-of-5 readings boundary and the P2 prior scale 0.25. All three are now pinned by extending existing tests (53 cases, no new test functions). After the fixes, seed 0 kills 35 of 40 and seed 1 kills 34 of 40. The 11 survivors are equivalent, numerically negligible or guard-only; NB-18 gives the reason for each. check_tests.py --base bba0fd0: OK (184 passed, 1 skipped); ruff F/E9 and fail-loud lint clean.
  - The identity proof (c3fd92d, passed 2026-10-13) still holds. `git diff --name-only c3fd92d..HEAD` touches only docs, notebook, CONSTRAINTS.md, the identity-proof result files, scripts/analysis_x1.py, scripts/decide_a2_a3.py and tests. No fit or score code changed.
- Not done: the RIGOR_REVIEWER review (step 3), the user's diff review (step 5), the tag, and every post-tag step.
- Codex, read at 08:53 EDT: Codex remaining: 5h 58% (resets in 2h 59m), weekly 52% (resets in 5d 10h), credits 1701.2503675222397. The Codex usage ledger (~/codex-offload/ledger.jsonl on cs-local) has no entry for d534623f: the loop was cancelled during attempt 2, before it wrote one, and the compute ledger shows the job as failed.
- No live runs, no sub-agents, no background cells. Git recipe: the python kernel needs GIT_AUTHOR_NAME/EMAIL and GIT_COMMITTER_NAME/EMAIL in os.environ for nb_entry. Push from bash with credentials=['GitHub'] and a credential helper that reads $GITHUB_TOKEN (memory mem_3d770f2ce7c5).

## Artifacts
- [ANALYSIS_SPEC.md]({{artifact:6ffecaea-63c5-4356-9593-0a249a8ef2f2}}): the analysis specification, identical to repo docs/ANALYSIS_SPEC.md at 1e615eb. Section 15 holds the agent-chosen implementation details for the diff review.
- [CONSTRAINTS.md]({{artifact:6da7d2d3-47fa-4625-8a3a-f10c7b9af6e9}}): rule ledger SI-1 to SI-77, identical to the repo file.
- [notebook_2026-10-14.md]({{artifact:c1294de0-994b-4a4a-b4fd-66604547ab6a}}) and [notebook_INDEX.md]({{artifact:ad012679-bc91-45ce-a263-71e9569adc83}}): the lab notebook up to NB-20261014-19.
- [codex_brief_analysis_x1.md]({{artifact:4a4ff7d8-b69a-4a8e-a343-f61eb034f8cc}}): the brief Codex implemented (provenance for ff88666).
- [mutation_runs_analysis_x1.json]({{artifact:ed06d6cc-0b9f-404a-9b20-973624b8e20b}}): the four mutation runs, with their survivors.
- [HANDOFF-final-review-fixes-6.md]({{artifact:ce5c655a-3953-43ad-851e-e14efd418619}}): the previous handoff. [HANDOFF-final-review-fixes-4.md]({{artifact:d340c246-d968-4cb3-ba57-1dc53077c8b5}}): its Next-steps item 6 lists post-tag steps 7-11.

## Kernel state (will be lost)
Nothing expensive. The kernel holds skill constants, notebook-entry return values and `R` (the repo path); they reload with the skills lab-notebook, standing-instructions, fail-loud-pipelines and session-handoff. The idle sweep removes these workspace files:
- dev/covenv_scvi, the coverage venv for check_tests on the scripts tier: a venv with --system-site-packages on env scvi-api plus pytest-cov 7.1.0 and ruff 0.14.0 (memory mem_b73a0dfc1fe5);
- offload/wcd, the Codex clone, with its run records under .codex-offload/loop_20261014T074027;
- offload/mut_tree*;
- acc/, the hidden acceptance tests, superseded by tests/analysis.

The brief and the mutation results are saved as artifacts (above).

## Decisions
- Agent: the implementation was offloaded to Codex under the AUTO-OFFLOAD rule, with hidden known-answer acceptance tests. Attempt 1 was accepted once the agent's acceptance check was fixed: it had counted the loop's own .codex-offload/ records as forbidden changes. The three agent edits are listed in the ff88666 message.
- Agent: the implementation choices that were not asked as questions are listed in ANALYSIS_SPEC section 15. Present them together at the diff review; do not ask them one by one beforehand.
- Agent: the 11 surviving mutants are accepted (NB-18 gives the reason for each).
- Agent: PREREG section 9 gains an X1-analysis entry. The user did not ask for it and reviews it at the diff review. The tag sentences are worded as a commitment (de741a0).
- The user decisions SI-68 to SI-77 are unchanged; see HANDOFF-6.

## Dead ends
- A codex-offload acceptance check that runs git status in the attempt tree must ignore .codex-offload/ (one timestamped folder per run; memory mem_54975f570538). Adding this pitfall to the user's codex-offload skill was proposed but not done.
- Mutation testing on a scratch copy without git fails: every mutant errors at write_record's git rev-parse and is counted as killed (memory mem_51609cf9d7d0). The --test-cmd must git init and commit first, and the unmutated copy must pass.
- nb_entry with type NOTE takes the field `text`, not `note`.
- Carried over from HANDOFF-6:
  - nb_entry without a git identity returns committed False;
  - si_propagate over D/review costs about 2 min and 6 GB, so search the repo only;
  - a CORRECTION must correct an NB id, while an error that exists only in a document gets a NOTE;
  - prereg_rules.outcomes refuses A1 input with the c88cce3 Sinkhorn rows, so drop Sinkhorn for descriptive summaries;
  - the input paths in docs/prereg/r4_a1_record.json point to a deleted workspace.

## Next steps
1. Step 3, the rigour review. Delegate to profile RIGOR_REVIEWER (routing rule) a review of all amendments and every commit in 6b4aefc..1e615eb (59 commits).
   - Focus: scripts/analysis_x1.py against ANALYSIS_SPEC sections 1-11 and 15; the known-answer values in tests/analysis; the SI-66 change in decide_a2_a3.py; the PREREG section 9 entries; the PAPER_PLAN notes.
   - Hand it the spec (6ffecaea) and CONSTRAINTS (6da7d2d3) artifacts and the repo path, and ask for findings with file:line and a severity.
   - Fix or rebut each finding, with notebook entries; commit and push. Load ponytail before any code change. Re-run check_tests if code changes (rebuild the coverage venv if it was swept).
2. If the review changes any fit or score code, re-run the identity proof (experiments/integration_checks_20261013/README.md). Otherwise it stands.
3. Step 5, the user's diff review of 5cb75c0..HEAD (SI-52). Group the commits by topic, in plain words, and put all the open questions below in one pass.
4. After sign-off, tag prereg-tier12-v3 (SI-60) at the reviewed HEAD and push the tag. Then HANDOFF-4 steps 7-11:
   - x1_jhpce_command.py TAG_SHA (G-17);
   - the 384/576 h time-limit fallback (G-15);
   - experiment-preflight with the single-cell pack;
   - the Sinkhorn redo per SI-57 as a local job, with the solver validation repeated on the refitted latents (F6), then the Sinkhorn parts of R1, R4-a1 and R3;
   - the CR2-01 L40S check (needs the user's go);
   - resume local X1 and filler part A (--clear-foreign-claims);
   - ask for the JHPCE go.

## Open questions for the user (all at the diff review)
- The details never asked as questions:
  - spec 2.3: a cell is not evaluable when fewer than 4 of its 5 leave-one-seed-out readings exist;
  - 5: multiplicity;
  - 8: the transfer-lambda rule;
  - 9-11;
  - the section 15 implementation details.
- Wording: the I091, D-01 and D-02 amendments; the SI-66 consequence; the new PREREG section 9 entries, including the X1-analysis entry the agent added and the tag sentences of de741a0.
- How to disclose that SI-66 was recorded on 2026-10-13 but implemented only on 2026-10-14 (f3c3415). No R2/R3 result changes.
- The user's go for each cluster job (the CR2-01 L40S check; JHPCE X1).
- Whether to add the .codex-offload exclusion to the codex-offload skill.
- Whether the user wants to delete the repo strays rf_extract.txt and HANDOFF-final-review-fixes-4.md (only they can).
