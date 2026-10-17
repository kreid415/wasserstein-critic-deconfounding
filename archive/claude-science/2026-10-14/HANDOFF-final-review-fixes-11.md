# Handoff: final-review fixes, part 11

**Session:** 2026-10-14 (EDT) · about 175 messages · 0 folds · frame 9a904e38-e2f6-41d7-b05f-b77140a5b2d3
**Status:** The three independent re-checks of branch final-review-fixes at 08f1e3c are done: all "pass with fixes", no blocker. Nothing was committed this session; the branch is unchanged at 08f1e3c (pushed). Nothing is running. The handoff fired on size (357k tokens of latest context).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-11.md (artifact id d850f8f2-44a6-45ba-8487-a29c6a67fb1d) in this project; read it before doing anything else.
Objective: finish the final-review fixes, have them re-checked, show the user the diff, tag prereg-tier12-v3, launch X1.
Next action: write the notebook RESULT entry for the three re-checks, then send the user the diff guide and ask the 19 questions of DIFF_REVIEW_QUESTIONS_draft.md in one pass.
Open first: HANDOFF-final-review-fixes-11.md; DIFF_REVIEW_QUESTIONS_draft.md; DIFF_REVIEW_final-review-fixes.md; recheck_20261014_findings.json; RECHECK_DOCS_REPORT_20261014.md
Lost with the old kernel: nothing expensive
Pending: the user's 19 answers; round-3 fixes; a focused re-check; final diff and tag
Do not redo: the three re-checks at 08f1e3c (done, all pass with fixes); both test tiers at 08f1e3c (re-run by the code re-check, pass); the bundle at 08f1e3c (artifact 2d4e2da6)
Before reusing any identifier, number or quote from the previous chat (frame 9a904e38-e2f6-41d7-b05f-b77140a5b2d3), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration docs/PREREG.md. Finish the final-review fixes on branch final-review-fixes, have them re-checked, show the user the full diff (SI-52), tag prereg-tier12-v3 (SI-60), redo the Sinkhorn pilots (SI-57), then launch X1. Plain language throughout (SI-67, profile NO-JARGON rule; the user answers unclear questions in free text, so give examples and the source of every number).

## State
- Repository D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD 08f1e3c, tree bfc3238d, pushed (ls-remote checked this session: branch and both tag objects match). Untracked strays rf_extract.txt and HANDOFF-final-review-fixes-4.md: only the user can delete them; never `git add -A`. 12 worktree entries (main + 11 old review worktrees).
- Re-checks (RIGOR_REVIEWER, each on private clones of the bundle; no X1/X13 score opened):
  - Analysis (frame c7059f6d): 16 of 18 closed, AN-04 and AN-18 partly; new 3 fix (AN-R02 spec 'never recomputes' contradicts code; AN-R03 run commands omit R4 extension manifests and scores; AN-R04 spec test command prints FAIL for run_stage.py:662-663) and 12 notes (AN-R01 pairs sensitivity can abort when both datasets of a family have s in [1e-6, 1.7e-6); AN-R06..R09, R14 unpinned behaviour: 13 of its 24 planted errors survive; AN-R10 grid accuracy 1e-5..3e-4 beyond about 3 prior scales; AN-R13 gap flag ignores all-failed configurations; AN-R15 per-arm unreached count not produced).
  - Code (frame 7cca0c41): 9 of 10 closed, CF-10 partly; both tiers re-run at 08f1e3c (421 passed 18 skipped; 172 passed; check_tests OK); new CF-R01 fix (record fl_lint version, ruff state 28 F/E9 findings, two-environment reason, base in a notebook NOTE) and notes CF-R02 (cap at call sites unpinned), CF-R03 (user question: refuse divergence record without git_sha or from a dirty checkout; rule A in report 9.1), CF-R04 (PREREG clause: rule scripts see scores only), CF-R05 (user question: coverage base).
  - Documents (frame e7f6ae28): 16 of 19 closed, DOC-01, DOC-11, DOC-18 partly; 72 of 73 numbers agree; new 5 fix (DOC-R01..R05) and 9 notes (DOC-R06..R14; R12 = power_report NaN, fallback diff ready).
- Verified by the lead against files this session: DOC-R03/AN-R02 contradiction (ANALYSIS_SPEC.md:58-59 vs 281-285 and analysis_x1.py:369); CF-R02 (alignment.py:225 default, :232-233 pass-through); DOC-R05 (spec status line, PREREG.md:509); NB-20261007-01 (filler pass B 150/150 scored, values not inspected), NB-20261012-02 (pass A resumed with 99 latents to score), PREREG.md:358-359 (identity check rescored one X1 and one X13 sysVI row).
- Not done: notebook RESULT entry for the re-checks; the question round; round-3 fixes; final diff files.

## Artifacts
- [DIFF_REVIEW_QUESTIONS_draft.md]({{artifact:1a9f2007-ce6d-4bfd-b26d-4842cd2d5f20}}) — the 19 questions for the user's single question round, each with evidence, options (recommended first) and what the answer changes; section C = planned file ownership of the three round-3 fixers  <!-- text/markdown, 11 KB -->
- [DIFF_REVIEW_final-review-fixes.md]({{artifact:38cfa70f-3bab-43fe-9f04-720469793f24}}) — draft plain-language guide to all 143 commits 5cb75c0..08f1e3c by topic (sections 1-4 done; section 5 needs the diff files at the final commit, section 6 the questions)  <!-- text/markdown, 26 KB -->
- [recheck_20261014_findings.json]({{artifact:3d9eb808-d2ae-49bf-aa63-5a7a6378a48e}}) — the three re-checks' structured results (keys analysis, code, docs: verdict, finding_checks, claims, new_findings, deviations, report and evidence ids): give each round-3 fixer its key  <!-- application/json, 123 KB -->
- [recheck_analysis_evidence_20261014.tar.gz]({{artifact:4fc04d2a-1c2e-481c-bb7e-5157820f755f}}) — analysis re-check scripts and logs (independent pooled-model reference ref_pool.py, planted errors O01-O24, dry runs)  <!-- application/gzip, 4044 KB -->
- [REPORT_recheck_analysis_20261014.md]({{artifact:fc10b264-370f-4274-acf1-b863bac6ffbc}}) — analysis re-check report (section 4.9: the five design choices; section 5: AN-R01..R15)  <!-- text/markdown, 68 KB -->
- [recheck_code_20261014_scripts_outputs.tar.gz]({{artifact:8f06d9c7-787a-4727-b853-7a4175b548f7}}) — code re-check scripts (r03_rule_validation_probe.py, ct_union_eval.py for a merged three-tier coverage record)  <!-- application/gzip, 572 KB -->
- [REVIEW_CODE_RECHECK_20261014.md]({{artifact:51d41a48-1090-4891-acfa-b692c19c302b}}) — code re-check report (section 6: base question; section 9: CF-R01..R05 with the recommended CF-R03 rule A)  <!-- text/markdown, 48 KB -->
- [rigor_ledger_entries_recheck_code_20261014.json]({{artifact:bd1b137f-7305-4e01-8d18-1585abb96708}}) — code re-check ledger entries to merge  <!-- application/json, 21 KB -->
- [recheck_docs_20261014_evidence.tar.gz]({{artifact:900b84e4-ef46-4ea3-bd26-e101b86cbe2f}}) — documents re-check scripts 01-09 (02-04 reproduce the no-pair, P2-recount and pilot-range numbers for DOC-R04) and 08_power_report_fallback_fix.diff  <!-- application/gzip, 70 KB -->
- [RECHECK_DOCS_REPORT_20261014.md]({{artifact:04bf4aa8-1784-4f17-923d-e40c7d71a377}}) — documents re-check report (section 1.1: the five fixes; section 5: DOC-R01..R14)  <!-- text/markdown, 56 KB -->
- [rigor_ledger_entries_recheck_docs_20261014.json]({{artifact:e1b6ffc9-4548-4a2b-8a3a-75455a70962f}}) — documents re-check ledger entries to merge  <!-- application/json, 59 KB -->
- [wcd_final-review-fixes_08f1e3c.bundle]({{artifact:2d4e2da6-2170-4989-a222-dd50764cbc45}}) — git bundle of final-review-fixes at 08f1e3c with tags v1 and v2 (sha256 d399e914...), used by the re-checkers  <!-- application/octet-stream, 26605 KB -->
- [rigor_review_20261014_findings.json]({{artifact:40b79c70-4782-43eb-ab8b-24cb05be2520}}) — first review's findings at 1e615eb (keys analysis, code, docs)
- [RIGOR_REVIEW_TRIAGE_20261014.md]({{artifact:739db527-636e-45f3-afef-1d5274db518c}}) — planned action per first-review finding
- [CF_FIXES_REPORT.md]({{artifact:fa1b46f9-cff0-48d8-a6d5-df83ce0e1e14}}) — code fixer's report (fl_lint version 1.0, 26 rules, checksum start: needed for CF-R01)
- [DOC_FIXES_REPORT.md]({{artifact:30dd365f-0a7b-4399-9754-7d9629e237df}}) — documents fixer's report (section 4: its 8 user questions)
- [lead_gates_20261014.tar.gz]({{artifact:856fc931-0942-4a80-810f-a589ac12f6b5}}) — exact test-tier commands (README.txt) and the import-closure output
- [CONSTRAINTS.md]({{artifact:6da7d2d3-47fa-4625-8a3a-f10c7b9af6e9}}) — decision ledger; identical to the repo file, unchanged
- [HANDOFF-final-review-fixes-10.md]({{artifact:a0e9a5f0-de1f-4a3a-b97a-352645c4980c}}) — previous handoff (part 10)

## Kernel state (will be lost)
Nothing expensive. Workspace-only: diffreview/ (commits.txt, files.txt, tagged.json = the 143 commits with topic tags; si51_77.json = ledger rule texts), handoff/ (prev_delegate_code.py = the 1e615eb review briefs; recheck_*.json). The re-check briefs (repl variables A, B, C, schema) can be recovered from this chat's archive (the host.delegate cell).

## Decisions
- Agent: re-checks got a git bundle (branch + tags v1, v2) instead of worktrees: worktrees under D/code can show as 'not a git repository' in the sandbox (64-file git snapshot limit).
- Agent: ask the user's questions BEFORE the round-3 fixes, because most fixes depend on the answers; after the fixes show the final diff again for a yes before the tag.
- Agent: round-3 fixers get disjoint files (DIFF_REVIEW_QUESTIONS_draft.md section C) so parallel patches cannot collide; ANALYSIS_SPEC.md belongs to the analysis fixer only.
- Agent: the change-size checklist of ponytail-review was pasted into code briefs because the RIGOR_REVIEWER and RUNNER_ENGINEER profiles do not carry the ponytail skills.

## Dead ends
- host.query returns at most about 2,000 characters per string value: page long msg_json or memory bodies with substr(x, offset, 1500).
- save_artifacts in the same parallel batch as an edit_file of that file saved the pre-edit version; save after the edit returns.
- `git bundle verify` outside a repository fails ('need a repository'); run it as git -C on the repository, or test by cloning.

## Next steps
1. Notebook RESULT entry for the three re-checks (lab-notebook skill; GIT_AUTHOR/COMMITTER NAME wcd, EMAIL noreply@localhost; push from bash with credentials=['GitHub'] and a credential helper reading $GITHUB_TOKEN). Then the session-handoff size check.
2. Question round: send the user the guide DIFF_REVIEW_final-review-fixes.md, then ask the 19 questions of DIFF_REVIEW_QUESTIONS_draft.md in one pass (ask_user, one decision per tab, recommended first; question 6 is factual, no recommendation). Before deleting worktrees (question 18), check each for uncommitted work; deletion only through delete_host_files.
3. Round-3 fixes per the answers: three fixers in parallel (analysis RUNNER_ENGINEER, code RUNNER_ENGINEER, documents MANUSCRIPT_WRITER), each in a private clone from the bundle at 08f1e3c, returning format-patch files and a report; same blinding and read-only rules as the re-check briefs; acceptance = the re-checkers' planted errors killed, tests/analysis + tests/prereg, both tiers, check_tests OK. Lead: read every diff, apply, notebook DECISION/RESULT entries (incl. CF-R01 NOTE and DOC-R11 NOTE), ledger entries for approvals (SI-78 onward), both tiers, import-closure probe, push.
4. A focused independent re-check (RIGOR_REVIEWER) of the round-3 commits only; fix what it finds.
5. Final diff files (code, documents, experiment scripts without results/, commit list) at the final commit; complete section 5 of the guide; the user's yes; tag prereg-tier12-v3 and push; then G-17 (pin the X1 cluster builder to the tag), G-15 (time-limit fallback), preflight, Sinkhorn redo (SI-57), CR2-01 and JHPCE X1 only on the user's go.

## Open questions for the user
All 19 are in DIFF_REVIEW_QUESTIONS_draft.md: zero_spread exclusion vs floor; the four other analysis designs; CF-R03 refusal; power_report fallback; scope of 'examined' (all X1 and X13 scores); did anyone read a value; hashes of existing score files; the spec proposals; the process proposals; SI-75; Sinkhorn A3 refits and R2/R3 re-run; redo record names; scaling sensitivities; SI-68 scope; coverage base; fl_lint record; unused imports; deleting old worktrees and strays; remaining wording items. Also still open: the user's go for each cluster job.
