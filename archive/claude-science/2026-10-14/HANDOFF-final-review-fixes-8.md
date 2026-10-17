# Handoff: final-review fixes, part 8

**Session:** 2026-10-14 (EDT) · about 215 messages · 0 folds · frame 399019b5-a00d-46ab-97b8-8602b0511f73
**Status:** Step 3, the independent rigour review, is done: three RIGOR_REVIEWER tracks reviewed 1e615eb in parallel, all PASS WITH FIXES, 47 findings, 20 of them to fix before the tag. The lead re-checked 5 findings against the primary files (DOC-01, DOC-04, DOC-05, DOC-15, DOC-17). No fix has been made yet. Nothing is running. The handoff fired on size (351k tokens of latest context).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-8.md (artifact id 9f3a09f8-4638-44b3-9073-f827909ecd11) in this project; read it before doing anything else.
Objective: finish the final-review fixes, show the user the diff, tag prereg-tier12-v3, launch X1.
Next action: verify the remaining rigour-review findings against the primary files, then fix or rebut the 20 fix items (triage table) with notebook entries.
Open first: RIGOR_REVIEW_TRIAGE_20261014.md; rigor_review_20261014_findings.json; ANALYSIS_SPEC.md; CONSTRAINTS.md
Lost with the old kernel: nothing expensive
Pending: no jobs or sub-agents; 20 fix items and 27 notes from the review
Do not redo: the three reviews (done at 1e615eb); the checks of DOC-01, -04, -05, -15, -17; identity proof at c3fd92d
Before reusing any identifier, number or quote from the previous chat (frame 399019b5-a00d-46ab-97b8-8602b0511f73), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md). Finish the final-review fixes on branch `final-review-fixes`, show the user the full diff (SI-52), tag `prereg-tier12-v3` (SI-60), redo the Sinkhorn pilots (SI-57), then launch X1 (the main experiment: 8 tasks, both decoders, seeds 0-4). Plain language throughout (SI-67, profile NO-JARGON rule).

## State
- Repository D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD 973a97d, pushed (local = remote, checked with ls-remote). Since 1e615eb only notebook commits: NB-20261014-20 (review RESULT), -21 (X1 early-rows audit), -22 (NOTE: Q9 premise wrong). Untracked strays rf_extract.txt and HANDOFF-final-review-fixes-4.md: only the user can delete them; never `git add -A`.
- The review: tracks X1 analysis code (frame 808e34dd-b051-468b-bde4-30046910db74; ids AN-nn), other code fixes (dc1fe687-2186-46b2-b0bc-ceb238d9aecc; CF-nn), amendment documents (78da076e-a094-4d0a-9d8d-c19576934b0a; DOC-nn). Fix items: AN-01..AN-05, CF-01, CF-02, DOC-01..09, 11, 12, 13, 15. The triage table lists every finding with a planned action. Declared limits: synthetic or pilot inputs only, chosen (not exhaustive) mutation sets, no GPU re-measurement, fl_lint not in the repository, literature at abstract level.
- Lead-verified on 2026-10-14:
  - DOC-17: the question behind SI-75 (chat 619cf68a, message 91) said the unconditioned discriminator collapsed in 8 of 45 pilot runs ("only 5 of its 6 strengths were in the pilot"), difference +0.7 points. Recomputed from A1 plus A1_ext_r1_round1 scores: 8 of 54 (lambda 0.03 are R1 extension rows, 0 of 9 collapsed), difference +3.7; conditioned 15/54 vs 17/54, -3.7, as stated.
  - DOC-04: docs/prereg/r4_a1_record.json flags immune|1 (b0 0.6382, b_common 0.6570) and sim1|1 (b0 0.8120, b_common 0.8189); PREREG names only immune|1. docs/prereg/r2r3_record.json: R2 12/12 evaluable, mean -0.00145; R3 17/20, +0.00101; both 'criteria not met', defaults kept.
  - DOC-01 and NB-20261014-21: 19 X1 score files from 2026-10-12, 17 adversarial-arm (immune, conditioned decoder, seed 0: discriminator lg1-lg6, pooled lg1-lg5, reference lg1-lg6) plus none and scvi_adv; no chat record displays or analyses an X1 score value.
  - DOC-05: Holm over the six primary results gives 0.05/6 = 0.0083; PREREG:106 and :251 and PAPER_PLAN:184 still give 0.0125 (4 tests). DOC-15: docs/EQUIVALENT_COMMITS.json evidence lists the 2b52542 identity files, not *_c3fd92d.csv.
- Not yet verified by the lead: every other finding, among them AN-01's grid errors, AN-03's tamper test and CF-02's divergence-record gap.
- To reconcile before quoting: distinct lambda = 0 rows of none/scanvi/scvi_adv in the manifest of record are 242+40+40 = 322 per DOC-19 and 302+40+40 = 382 per CF-03; 1,908 counts occurrences over 8 manifest files.
- The identity proof at c3fd92d still holds at 973a97d (the code review checked the import closure: no fit, scorer, cluster-helper or src file changed). CF-02's fix would change scripts/run_stage.py, which is inside that closure.
- Sandbox: worktrees under D/code can be masked (git says 'not a git repository'); use a private clone in the workspace (memory mem_76332f00c93b). Three unused worktrees were created this session: D/code/wt_rigor_{analysis,code,docs}_1e615eb.
- Codex was not used this session.

## Artifacts
- [RIGOR_REVIEW_TRIAGE_20261014.md]({{artifact:739db527-636e-45f3-afef-1d5274db518c}}): the working list: all 47 findings, severity, first sentence and planned action; 'lead-verified' marks the five checked items.
- [rigor_review_20261014_findings.json]({{artifact:40b79c70-4782-43eb-ab8b-24cb05be2520}}): every finding and claim verbatim (problem, evidence, suggested fix) plus each track's deviations.
- Reports: [REPORT_rigor_analysis_20261014.md]({{artifact:2b69abea-5c1d-44cb-bd24-9c7a81c60f3a}}) (its Appendix A lists the 40 planted errors), [REVIEW_CODE_RIGOR_20261014.md]({{artifact:a7e42e48-8a71-4d81-aa7b-618caab4e7ca}}), [REPORT_rigor_docs_20261014.md]({{artifact:322d20a6-d871-4bb8-9e41-c16faaf8a1aa}}) (its section 6 drafts the amendment table of DOC-03).
- Reviewer evidence: [rigor_analysis_review_bundle.tar.gz]({{artifact:3de51e25-da4a-49d3-b248-7a8143d3197d}}) (scripts incl. the Gauss-Legendre reference; mutation list also as [mutation_results_all.csv]({{artifact:0742937f-4537-455b-9fc6-4a28991e41b2}})), [rigor_code_20261014_scripts_outputs.tar.gz]({{artifact:8d340d79-13d7-4c47-9f88-6db2ab9e9fcc}}) (probes, incl. import_closure.py and the R01a/b/c planted bugs), [rigor_docs_review_bundle.tar.gz]({{artifact:13872d76-d52b-4fe9-aae5-fea2dec27723}}), [numbers_recomputed_docs.csv]({{artifact:f7f048b5-87cc-4830-917f-cf0d26987838}}).
- [notebook_2026-10-14.md]({{artifact:03715b82-f8d3-44e3-8bd7-d6277c8247d5}}) and [notebook_INDEX.md]({{artifact:f741f5c5-1143-45e8-a4f5-a11e15ac836e}}): the notebook up to NB-20261014-22.
- Unchanged and still canonical: [ANALYSIS_SPEC.md]({{artifact:6ffecaea-63c5-4356-9593-0a249a8ef2f2}}), [CONSTRAINTS.md]({{artifact:6da7d2d3-47fa-4625-8a3a-f10c7b9af6e9}}), [HANDOFF-final-review-fixes-7.md]({{artifact:97fba492-41e9-49f4-8a3a-b7f45b2b7df0}}) (its open questions still apply), [HANDOFF-final-review-fixes-4.md]({{artifact:d340c246-d968-4cb3-ba57-1dc53077c8b5}}) (post-tag steps 7-11).

## Kernel state (will be lost)
Nothing expensive. The python kernel holds the ledger L, the notebook-entry results e1-e3, the triage dict act, the review results revs and merged (copies of the findings artifact) and the paths R (repository) and W (workspace); all are cheap and reload from the artifacts and the repository. The repl kernel holds res, res2 (review results) and the helper get_msg (pages frame_messages with substr).
Rebuild for notebook entries: set os.environ GIT_AUTHOR_NAME/GIT_COMMITTER_NAME = wcd and GIT_AUTHOR_EMAIL/GIT_COMMITTER_EMAIL = noreply@localhost, then nb_entry(R, ...). Workspace copies under handoff/ and out/ are swept when idle; the artifacts above hold them. No coverage venv was built this session.

## Decisions
- Agent: step 3 ran as three parallel RIGOR_REVIEWER tracks, not one, because one reviewer would have had to hold 59 commits, three long documents and 1,074 lines of analysis code.
- Agent: reviewers were barred from tier12_v2/X1_local, jhpce_tier12, review_sealed, the X1 runner logs and the identity-review score outputs (the tag promises the analysis is frozen before any X1 adversarial-arm score is examined).
- Agent, proposed and not yet built (diff-review items): AN-01 a graded Gauss-Legendre rule over each spread; AN-02 a dataset with s = 0 is not evaluable ('zero_spread'); AN-03 recompute the R4 targets with prereg_rules.r4_stage and refuse on mismatch; CF-02 also check the commit of divergence records; CF-10 leave fit_paper_config.py's unused imports alone so the identity proof stays valid.
- Agent: AN-01 to AN-03 need numerical validation and design calls, so the lead should build them; the test extensions (AN-04, CF-01) could go to Codex under the AUTO-OFFLOAD rule (run codex_canary.py --auto first).

## Dead ends
- Worktrees under granted host folders for sub-agents: the git pointer can be masked; give sub-agents a clone command instead.
- host.query truncates long cell values (about 2,000 characters): page frame_messages with substr(). host.archive.search covers only the current chat; query frame_messages by frame_id for older chats.
- A backgrounded repl collect cell blocks the repl; use fresh: true cells for side queries meanwhile.
- Carried over: nb_entry without a git identity returns committed False; si_propagate over D/review costs about 2 min and 6 GB; a CORRECTION must correct an NB id (an error only in a document or a chat message gets a NOTE); prereg_rules.outcomes refuses A1 input with the c88cce3 Sinkhorn rows; the input paths in docs/prereg/r4_a1_record.json point to a deleted workspace; mutation copies need git init and a passing unmutated copy.

## Next steps
1. Verify each remaining finding against the primary files before acting, starting with the 20 fix items; then fix or rebut each one with a notebook entry, commit and push. Load ponytail before code changes; FEWEST-TESTS rule; check_tests.py needs a venv with --system-site-packages plus pytest-cov 7.1.0 and ruff 0.14.0.
   - analysis_x1.py and tests/analysis: AN-01 (validate against a converged reference down to s about 1e-4; amend spec sections 4 and 15.4), AN-02, AN-03, AN-05 (spec 15.11), AN-06, AN-10, AN-11, AN-13, AN-14, CF-09; AN-04 test extensions from the 40-mutant list. Gate: `scripts/check_tests.py --base 973a97d --src scripts -- tests/analysis tests/prereg` in env scvi-api.
   - CF-01, CF-02, CF-03 (pin value sets), CF-08; gates per tier (src with wcd-gpu, PYTHONPATH=src; scripts and cluster with scvi-api, WCD_SRC=src). If run_stage.py changes, show with the code review's probes/import_closure.py that only the start-up check changed, or re-run the identity proof (experiments/integration_checks_20261013/README.md).
   - Documents: PREREG section 9 (DOC-01, -02, -04 to -08, -11, -12, -14, -15; CF-04, CF-06, CF-07), the status line and amendment table (DOC-03), ANALYSIS_SPEC (DOC-13, AN-05, AN-09 with DOC-10, a limitations list from DOC-16 and AN-07/08/12/15/17, 'pilot-informed' labels from DOC-17), PAPER_PLAN (DOC-05, -08, -09), CONSTRAINTS annotations that leave the user's words unchanged (DOC-18), README counts (CF-03, DOC-19).
   - Send each track's fixes back to its reviewer (host.send_message to the frame ids above) for a focused re-check.
2. Step 5, the user's diff review of 5cb75c0..HEAD: commits grouped by topic in plain words, all open questions in one pass.
3. After sign-off: tag prereg-tier12-v3 (SI-60) and push it; then HANDOFF-4 steps 7-11 (TAG_SHA G-17, G-15 fallback, preflight, Sinkhorn redo per SI-57 with solver validation on refitted latents, CR2-01 and JHPCE X1 only on the user's go).

## Open questions for the user (at the diff review unless noted)
- Does SI-75 (P2 margin 5 points, prior scale 0.25) stand, now that the pilot differences are +3.7 and -3.7 points? (Told to the user in the handoff reply.)
- The fix designs: AN-01 quadrature, AN-02 s = 0 rule, AN-03 R4 check, the spec 15.11 wording (AN-05), and the leave-one-family-out transfer rule, which no user decision covers yet (DOC-08).
- DOC-01 disclosure wording and the definition of 'examined'; DOC-03 amendment-table format.
- Deleting the old review worktrees (wt_identity, wt_rereview_sinkhorn, wt_review_guards, wt_review_sinkhorn, wt_sinkhorn_pilot, wt_suite and the three wt_rigor_*_1e615eb) and pruning the two stale worktree entries: needs the user's approval.
- Still open from HANDOFF-7: spec 2.3, 5, 8, 9-11 and 15; wording of the I091, D-01 and D-02 amendments; how to disclose SI-66's late implementation; the user's go for each cluster job; the codex-offload pitfall; the two repository strays.
