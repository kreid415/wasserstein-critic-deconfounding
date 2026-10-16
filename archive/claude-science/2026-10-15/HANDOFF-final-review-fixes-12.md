# Handoff: final-review fixes, part 12

**Session:** 2026-10-15 (EDT) · about 150 messages · 0 folds · frame 17c506be-e21c-4e42-bae8-d638e83a0c14
**Status:** All 19 diff-review questions are answered and recorded (ledger SI-78 to SI-87; notebook NB-20261015-02 to -20); the re-check RESULT is NB-20261015-01. Branch final-review-fixes is at 5d06a27, pushed (24 record-only commits after 08f1e3c: notebook/ and CONSTRAINTS.md). Nothing is running. The handoff fired on size (314k tokens of latest context).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-12.md (artifact id 228692e5-f555-4c0b-b245-c8dd35eac0a8) in this project; read it before doing anything else.
Objective: make the round-3 fixes the user's 19 answers require, have them re-checked, show the final diff, tag prereg-tier12-v3, then the Sinkhorn redo and X1.
Next action: make a git bundle of final-review-fixes at 5d06a27, then brief the three round-3 fixers (analysis, code, documents) per the handoff's Next steps.
Open first: HANDOFF-final-review-fixes-12.md; CONSTRAINTS.md (SI-78 to SI-87); DIFF_REVIEW_QUESTIONS_draft.md (section C); recheck_20261014_findings.json; DIFF_REVIEW_final-review-fixes.md
Lost with the old kernel: nothing expensive
Pending: round-3 fixes; focused re-check; final diff, the user's yes, tag; worktree deletion (approved)
Do not redo: the 19-question round (all answered 2026-10-15); the three re-checks at 08f1e3c; ledger and notebook records of the answers
Before reusing any identifier, number or quote from the previous chat (frame 17c506be-e21c-4e42-bae8-d638e83a0c14), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction under docs/PREREG.md. Now: the round-3 fixes that the 19 answers require, a focused independent re-check, the final diff for the user's yes (SI-52), tag prereg-tier12-v3 (SI-60), then the Sinkhorn redo (SI-57) and X1. Plain language with every number's source (SI-67, profile NO-JARGON rule).

## State
- Repository D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD 5d06a27, pushed (ls-remote checked). Untracked strays rf_extract.txt and HANDOFF-final-review-fixes-4.md (deletion approved, Q18; only via delete_host_files). Ledger 87 entries; notebook 206 entries, hash chain OK.
- Answers (question number of DIFF_REVIEW_QUESTIONS_draft.md -> record): Q1 zero-uncertainty dataset KEPT with standard error raised to 0.000001, not left out (SI-83; the user changed the drafted AN-02 exclusion because dropping exact ties biases the result); Q2 five analysis designs approved (SI-84); Q3 runner refuses divergence records without a commit or from uncommitted code (SI-85); Q4 power_report fallback before the tag (SI-86); Q5 'examined' = every X1 and X13 score, any arm (SI-78); Q6 user: nobody read an X1 or X13 value (NOTE NB-20261015-15); Q7 hash list of existing X1/X13 score files plus the tag's GitHub date (SI-79); Q8 spec proposals approved, except the parts Q1/Q2 decide (SI-80); Q9 process rules (SI-81); Q10 SI-75 stands (NB-20261015-07); Q11 keep A3 refits and R2/R3 re-run (NB-20261015-08); Q12 redo OVERWRITES the record files in place, old versions at tags v2/v3 (SI-87); Q13 scaling checks exploratory (SI-82); Q14 SI-68 wording confirmed (NB-20261015-09); Q15 coverage base 973a97d plus merged three-tier record from 5cb75c0 (NB-20261015-10); Q16 fl_lint in a notebook NOTE (NB-20261015-11); Q17 leave the 28 ruff findings (NB-20261015-12); Q18 delete 11 worktrees and 2 strays after a check (NB-20261015-13); Q19 wording approved (NB-20261015-14).
- Guide updated (status line, section 4 record line, section 6 short question list); section 5 (full diff) still empty.

## Artifacts
- [DIFF_REVIEW_final-review-fixes.md]({{artifact:917a8c6d-e0bf-4620-8cf9-1dc9a1c43734}}) — plain-language guide to the branch (sections 1-4, 6); add section 5 and the answers before the final yes
- [DIFF_REVIEW_QUESTIONS_draft.md]({{artifact:1a9f2007-ce6d-4bfd-b26d-4842cd2d5f20}}) — the 19 questions; section C = file ownership of the three round-3 fixers (update with the answers above)
- [recheck_20261014_findings.json]({{artifact:3d9eb808-d2ae-49bf-aa63-5a7a6378a48e}}) — three re-checks' structured results (keys analysis, code, docs): give each fixer its key
- [REPORT_recheck_analysis_20261014.md]({{artifact:fc10b264-370f-4274-acf1-b863bac6ffbc}}) — analysis re-check (4.9 designs, 4.10 referee points, 5 AN-R01..R15)
- [REVIEW_CODE_RECHECK_20261014.md]({{artifact:51d41a48-1090-4891-acfa-b692c19c302b}}) — code re-check (6 coverage base, 9 CF-R01..R05; 9.1 rule A for SI-85)
- [RECHECK_DOCS_REPORT_20261014.md]({{artifact:04bf4aa8-1784-4f17-923d-e40c7d71a377}}) — documents re-check (5 DOC-R01..R14)
- [recheck_analysis_evidence_20261014.tar.gz]({{artifact:4fc04d2a-1c2e-481c-bb7e-5157820f755f}}) — analysis re-check scripts (planted errors O01-O24, ref_pool.py)
- [recheck_code_20261014_scripts_outputs.tar.gz]({{artifact:8f06d9c7-787a-4727-b853-7a4175b548f7}}) — code re-check scripts (ct_union_eval.py for the Q15 merged record; r03 probe)
- [recheck_docs_20261014_evidence.tar.gz]({{artifact:900b84e4-ef46-4ea3-bd26-e101b86cbe2f}}) — documents re-check scripts 01-09 (02-04 for DOC-R04; 08_power_report_fallback_fix.diff for SI-86)
- [rigor_ledger_entries_recheck_code_20261014.json]({{artifact:bd1b137f-7305-4e01-8d18-1585abb96708}}) — code re-check rigor-ledger entries to merge
- [rigor_ledger_entries_recheck_docs_20261014.json]({{artifact:e1b6ffc9-4548-4a2b-8a3a-75455a70962f}}) — documents re-check rigor-ledger entries to merge
- [wcd_final-review-fixes_08f1e3c.bundle]({{artifact:2d4e2da6-2170-4989-a222-dd50764cbc45}}) — bundle at 08f1e3c (re-checks); make a new one at 5d06a27 for the fixers
- [CF_FIXES_REPORT.md]({{artifact:fa1b46f9-cff0-48d8-a6d5-df83ce0e1e14}}) — fl_lint version 1.0, 26 rules, kernel.py SHA-256 start 12df2b75b331f456 (for the Q16 NOTE)
- [lead_gates_20261014.tar.gz]({{artifact:856fc931-0942-4a80-810f-a589ac12f6b5}}) — exact test-tier commands (README.txt) and the import-closure output
- [notebook_2026-10-15.md]({{artifact:76f2fc79-ab14-46fd-8d2e-51c0a4ebcb7a}}) — today's notebook page (NB-20261015-01..20)
- [notebook_INDEX.md]({{artifact:f259bafd-ee30-4fb8-80f2-cd66c197c0ae}}) — notebook index
- [CONSTRAINTS.md]({{artifact:d95848c5-8ed7-4b94-a4d9-cb2426e5986b}}) — decision ledger incl. SI-78..SI-87; identical to the repo file
- [HANDOFF-final-review-fixes-11.md]({{artifact:76b4e5ea-ef0d-42c5-99ba-b44867667850}}) — previous handoff (part 11)

## Kernel state (will be lost)
Nothing expensive: the findings JSON and report texts were read into variables; reload from the artifacts above.

## Decisions
- Agent: every number in the 19 questions was checked against its source before asking; four questions changed: Q2 adds AN-05, Q8 says Holm applies only to the descriptive Wilcoxon companion (it can never pass 0.0083 with six datasets), Q13 got a recommendation (exploratory), Q17 corrected (the 23 findings outside the identity-proof code are mixed, not 23 unused imports).
- Agent: SI-80 excludes the zero-spread (sections 2.3, 4) and Q2 parts (section 15) of the spec; those follow SI-83 and SI-84.
- Agent: in the second Q1 wording I said a zero standard error gives one dataset unlimited weight; wrong for this model (one or two pinned datasets cannot give a conclusion, analysis re-check 4.10); corrected in the third wording before the user answered.
- Agent: Q6 is recorded as a user NOTE, Q10, Q11 and Q14-Q19 as one-off notebook DECISIONs without ledger entries.

## Dead ends
- bash cells exhaust the sandbox's 64-file git snapshot budget (the 11 worktrees); git via subprocess in the python kernel and git push from bash both worked for the main repository.
- host.query: the memories text column is `body`; frame_messages has frame_id, idx, msg_json (no seq).
- A spec-heading regex capped at 120 characters missed section 8; it exists ("Single-lambda tables: transfer lambda").

## Next steps
1. New git bundle of final-review-fixes at 5d06a27 with tags v1 and v2; verify by cloning (git bundle verify needs a repository).
2. Round-3 fixes, three fixers in parallel (ROUTING RULE profiles; private clones; format-patch output; same blinding rules as the re-check briefs; paste the ponytail change-size checklist and the FEWEST-TESTS rule into code briefs). Analysis fixer (RUNNER_ENGINEER; scripts/analysis_x1.py, tests/analysis/, docs/ANALYSIS_SPEC.md, experiments/analysis_numerics_20261014/README.md): SI-83 floor in code and sections 2.3/4 incl. the pairs sensitivity (AN-R01) and a list of raised datasets; SI-84 text (AN-R10 accuracy range, approximate limits); AN-R02, AN-R03, AN-R04; AN-R15 sentence; SI-78, SI-80, SI-82 and SI-87 (no *_sinkhorn_redo names) in the spec; spec parts of DOC-R05, -R07, -R08, -R09 (items 1, 6), -R10, -R13. Code fixer (RUNNER_ENGINEER; scripts/run_stage.py, tests/stage, scripts/prereg_rules.py, tests/prereg, tests/wcd_vae/test_divergences_v2.py, the power_without_sim1.py docstring): SI-85 (rule A, one test for records and latents), SI-86 (the re-checker's diff plus one test), CF-R02, docstring facts of DOC-R12. Documents fixer (MANUSCRIPT_WRITER; docs/PREREG.md, docs/PAPER_PLAN.md, docs/EQUIVALENT_COMMITS.json, a new experiments folder for DOC-R04): DOC-R01 (with NB-20261015-15), DOC-R02, DOC-R04, DOC-R05, SI-78, SI-79 sentence, SI-80/SI-81 labels, SI-82, SI-87, DOC-R13 (the 19 questions with answers in PREREG section 9), CF-R04, notes DOC-R06..R11 as adopted.
3. Lead: NOTE for CF-R01/Q16 (fl_lint source, version, rules, checksum; ruff 28 F/E9 findings at 08f1e3c = 23 unused imports, 4 unused variables, 1 f-string, 5 inside the identity-proof code; two environments because scvi-api lacks scib and wcd-gpu cannot import scvi); DOC-R11 NOTE; a committed script that hashes the X1/X13 score files without printing values (SI-79); the Q15 merged coverage record; read every diff, apply, both tiers, import-closure probe, push.
4. Focused independent re-check (RIGOR_REVIEWER) of the round-3 commits; fix what it finds.
5. Final diff files; guide section 5 and the answers in section 6; the user's yes; tag prereg-tier12-v3 and push; record the tag's GitHub date (SI-79).
6. Q18 at any time: run git status --porcelain inside each of the 11 worktree folders, report any with work, delete_host_files for the folders and the two strays, then git worktree prune.
7. Then G-17, G-15, preflight, Sinkhorn redo (SI-57; records overwritten per SI-87), CR2-01 and JHPCE X1, each cluster job only on the user's go.

## Open questions for the user
None of the 19 remain. Still open: the user's yes on the final diff before the tag, and the user's go for each cluster job.
