# Handoff: final-review fixes, part 14

**Session:** 2026-10-16 (EDT) · 342 messages, 334k tokens of latest context · 0 folds · frame 95c88c4c-ebed-4754-99e3-6251a2898c3f
**Status:** All round-3 patches, the SI-89 code, the re-check fixes and the redone audit are applied and pushed; branch final-review-fixes at fb7edac (remote equals local). Nothing is running. Fired on size.

## Start message for the next chat
```
Chat name: Launch Light-Dataset Experiment Wave
Continue from HANDOFF-final-review-fixes-14.md (artifact id 2c58a2b5-b2c1-46ba-98f3-b769a4c39dd9) in this project; read it before doing anything else.
Objective: confirming re-review of 3a5b728..fb7edac, 2 user questions, final diff, pre-tag audit repeat, tag prereg-tier12-v3.
Next action: dispatch RIGOR_REVIEWER (Opus) on bundle fb7edac to confirm the fixes of re-check findings RC-B1 to RC-m10.
Open first: HANDOFF-final-review-fixes-14.md; recheck_final_review_fixes_20261016.md; CONSTRAINTS.md
Lost with the old kernel: nothing expensive
Pending: confirming re-review not started; pre-tag audit must search session 95c88c4c
Do not redo: round-3 patches, SI-89, docs fixes, audit redo (applied, verified); 8 user questions of 2026-10-16 (answered)
Before reusing any identifier, number or quote from the previous chat (frame 95c88c4c-ebed-4754-99e3-6251a2898c3f), search its archive.
```

## Objective
Tag prereg-tier12-v3 for the Wasserstein-critic vs discriminator batch-correction benchmark (docs/PREREG.md), after the user's yes on the final diff (SI-52, SI-60), then record the tag's GitHub date (SI-79); then the Sinkhorn redo (SI-57) and X1. Plain language with every number's source (SI-67, NO-JARGON).

## State
- Repo D/code/wcd_final-review-fixes, D = /home/kendall/experiment_data/wasserstein-critic-deconfounding. HEAD fb7edac, pushed. Ledger: SI-01..SI-89. Notebook: 236 entries, hash chain OK (NB-20261016-01..21 today).
- Done today (notebook ids in brackets):
  - 44 round-3 patches applied, trees checked against the fixers (-01..-04); lead follow-ups (-08); findings 659801d7 and 8e0ad5ed settled (-06, -07); CORRECTION 54/57 (-05).
  - User answers, all recommended: SI-85 every arm (-09), new SI-89 latent without git_sha refused (-10), SI-81 (c) promise only (-11), SI-81 (b) listing clause kept (-12).
  - SI-89 code 4b02068 plus gate record experiments/lead_gates_20261016 (-18). Tiers: scripts 439 passed/18 skipped OK; source 172 OK; scoring 14 passed/0 skipped (R_LIBS fix). Merged coverage from 5cb75c0: only scripts/check_tests.py, 101 lines, run by no tier. Closures of fitter, plan, scorer unchanged since c3fd92d.
  - Independent re-check (RIGOR_REVIEWER, Opus): NOT READY, 2 blockers, 4 major, 10 minor (-13). User answers: SI-78 people only (-14); cap and NMI-0 stay case by case, stated as a limitation (-15); count-only X1_local tally and status records hashed (-16); raised-SE count reported (-17).
  - Docs-fix series of 12 commits (-21): tests/analysis+prereg 202 passed; check_tests OK; mutation 72/73 (M28).
  - Audit redo fd5fdef (-19): hash list 851 files, 827 old rows unchanged; X1_local ledger 19 scored + 1,097 needs_fit + 9 pending = 1,125; THIRD display found (2026-10-12 23:54, 'Verify cluster' frame 4295f20e msg 117, head of X13_lung_scanvi_l0_c1_s0_8e6f5009) and disclosed under amended SI-78; CORRECTION pass A 8 files, not 1 (-20).
  - Lead PREREG disclosure rewrite fb7edac: RC-B1, RC-B2, RC-M1, RC-M3, RC-m4 (identity-check runs 11:56-12:08, 15:05-15:13, 21:38-21:46 EDT). doc_numbers README PREREG refs verified by content.
- NOT YET DONE: confirming re-review of 3a5b728..fb7edac (the fixes of all 16 findings are believed done, not independently confirmed). The audit LEFT OUT this lead session 95c88c4c (cut-off 2026-10-16 14:37:57 EDT); PREREG says the pre-tag repeat searches it and every later session.
- Q18 (user, outside the app): delete 8 old worktrees and 2 strays (rf_extract.txt, HANDOFF-final-review-fixes-4.md) under D/code, then git worktree prune; wt_suite kept.

## Artifacts
- [wcd_final-review-fixes_fb7edac.bundle]({{artifact:ddd2826b-e797-41d1-9f83-17faa606a7d2}}): bundle of HEAD (SHA-256 43f57b12...)
- [recheck_final_review_fixes_20261016.md]({{artifact:8f59a6b6-4bb1-4515-80e4-acaf15cfba7c}}) and [findings.json]({{artifact:6d739d29-347b-4bb5-b0cb-e20dd856e79c}}): the 16 re-check findings (RC-B1..RC-m10)
- [recheck_decisions_details.md]({{artifact:74cd7463-567a-43b2-9b4b-c2de4256a4f1}}) and [round3_open_questions_details.md]({{artifact:a4ccb66e-3778-4427-a9ce-4e8a59ca552e}}): evidence for today's 8 user questions
- Audit redo: [patches]({{artifact:0af22dab-3d3f-436f-b75d-c5d77c9a8050}}), [report]({{artifact:fdd48d53-6946-468f-85a1-d7733040583b}}). Docs fixes: [patches]({{artifact:f073ee76-1022-473a-a49a-f8a4b1485571}}), [report]({{artifact:d61da931-7016-4cef-9903-db5385fd5e7a}}). SI-89: [patches]({{artifact:8eb4e2cb-edf9-40a3-933a-9ca50150d38c}}), [gates README]({{artifact:69fab0f4-b5b6-47f0-b632-e427441c8670}})
- [CONSTRAINTS.md]({{artifact:6138fbf0-d40e-4877-955c-e52005887d0a}}) (identical to repo), [notebook_2026-10-16.md]({{artifact:663eb859-1d25-4542-9a5e-4d2d3847a0a0}}), [notebook_INDEX.md]({{artifact:eeaa3e23-127a-4d3a-a57b-40c607d40686}})
- [DIFF_REVIEW_final-review-fixes.md]({{artifact:917a8c6d-e0bf-4620-8cf9-1dc9a1c43734}}): guide for the user's final diff; section 5 (full diff) and the answers still to add
- Previous handoff [part 13]({{artifact:3c38f797-ef5f-4398-bef1-2fca28c5ca6f}})

## Kernel state (will be lost)
Nothing expensive. Scratch clones under the workspace (scratch/t_*, c_*) are disposable; re-clone from the bundle. pytest-cov 7.1.0 and coverage 7.16.2 are now installed in scvi-api, wcd-gpu and wcd-kbet (persistent envs).

## Decisions
- Agent: the scoring tier is judged by the merged union record; on its own at base 5cb75c0 it exits 1 because the files it reports are tested by the other tiers.
- Agent: the third display needs no new user decision: SI-78 as amended (NB-20261016-14) says an agent display is disclosed, not examined.
- Agent: the analysis output reports two counts (n_raised_datasets, n_raised_pairs), because in the pairs analysis `raised` mixes datasets and families.
- Agent: in PREREG and ANALYSIS_SPEC, line counts were kept or line references re-checked by content, because experiments/doc_numbers_20261015/README.md cites lines.

## Dead ends
- A literal artifact marker inside a python cell is rewritten to a path before the cell runs; build markers with host.artifact_marker or regex on the id.
- The 2026-10-15 audit searched only 46 names (groups 3-5). Never present an audit result without stating its identifier set and cut-off.
- The scoring tier skipped all tests in a plain shell until R_HOME and R_LIBS were set for wcd-kbet (see experiments/lead_gates_20261016/README.txt).
- The sandbox's git budget: run git through a python subprocess; nb_entry commits need GIT_AUTHOR_*/GIT_COMMITTER_* set in os.environ.

## Next steps
1. Confirming re-review (RIGOR_REVIEWER, Opus) of 3a5b728..fb7edac on the bundle: was each of RC-B1..RC-m10 fixed as described, are the new PREREG disclosure numbers right against the committed outputs, is anything stale. Fix what it finds.
2. Ask the user (one card each): (a) hash attempts/*.jsonl files (X1_local 28, fillers 250) as well? (the audit sub-agent's open question; not status records in the runner's sense); (b) the scripts/check_tests.py lines no test runs (101): report only, as now?
3. Final diff: fill DIFF_REVIEW guide section 5 with `git diff 5d06a27..HEAD --stat` plus per-file summaries and the answers (SI-78..SI-89, NB-20261016-*); the user's yes (SI-52).
4. Just before the tag, re-run audit_displayed_values.py for session 95c88c4c and every session after 2026-10-16 14:37:57 EDT (in the repl kernel; see its docstring). If it finds a display, stop and tell the user. Record the result in the tag's notebook entry.
5. Tag prereg-tier12-v3 on the approved commit, push the tag, record the date GitHub shows for it (SI-79) in a notebook entry; save notebook and ledger artifacts.
6. Then G-17, G-15, preflight, the Sinkhorn redo (SI-57; records overwritten per SI-87), CR2-01 and the JHPCE X1 part, each cluster job only on the user's go.

## Open questions for the user
1. Should attempts/*.jsonl runner logs be hashed with the score files?
2. Accept that scripts/check_tests.py keeps 101 lines no test runs (reported only)?
3. The yes on the final diff, and the go for each cluster job.
