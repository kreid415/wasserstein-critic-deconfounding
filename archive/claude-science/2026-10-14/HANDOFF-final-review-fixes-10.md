# Handoff: final-review fixes, part 10

**Session:** 2026-10-14 (EDT) · 300 messages · 0 folds · frame 794143d3-d15a-4852-b863-d4f7cc968d23
**Status:** Both sub-agent patch sets are applied and pass both test tiers; the lead's spec work for DOC-13 and the spec parts of the documents findings is committed; notebook NB-20261014-29 to -35; branch pushed at 08f1e3c. Nothing is running. The handoff fired on size (476k tokens of latest context).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-10.md (artifact id b32b0e7d-1e47-4eaa-946f-b57a85c50ab1) in this project; read it before doing anything else.
Objective: finish the final-review fixes, have them re-checked, show the user the diff, tag prereg-tier12-v3, launch X1.
Next action: bundle final-review-fixes at 08f1e3c and run three RIGOR_REVIEWER re-checks (analysis, code, documents tracks) in parallel.
Open first: HANDOFF-final-review-fixes-10.md; rigor_review_20261014_findings.json; CF_FIXES_REPORT.md; DOC_FIXES_REPORT.md; lead_gates_20261014.tar.gz
Lost with the old kernel: nothing expensive
Pending: three re-checks; the user's diff review; open user questions listed in the handoff
Do not redo: applying the 24 patches and the CF-02 closure check (done, 08f1e3c); the DOC-13 spec restructure (4af5069); both test tiers on tree 46a2df12 (pass)
Before reusing any identifier, number or quote from the previous chat (frame 794143d3-d15a-4852-b863-d4f7cc968d23), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md). Finish the final-review fixes on branch `final-review-fixes`, have them re-checked independently, show the user the full diff (SI-52), tag `prereg-tier12-v3` (SI-60), redo the Sinkhorn pilots (SI-57), then launch X1. Plain language throughout (SI-67, profile NO-JARGON rule).

## State
- Repository D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD 08f1e3c, pushed (remote = local, checked with ls-remote). Untracked strays rf_extract.txt and HANDOFF-final-review-fixes-4.md: only the user can delete them; never `git add -A`.
- Done and verified this session (every diff read before applying):
  - Code patches applied: CF-01 b82bf49, CF-02 8351999, CF-03 38f5149, CF-05 0110b63, CF-08 ca6dee0, CF-10 2405cd1.
  - CF-02 checked by the lead: import closure c3fd92d..06345ce lists only scripts/run_stage.py in the fit closure (docstring, start-up check reading info['status']['git_sha'], ledger git_sha fallback), nothing in the scorer or cluster-helper closures; `git diff c3fd92d HEAD` over src/ and the other closure files is empty, so the identity proof at c3fd92d carries over (not re-run). Source: record_divergence writes provenance(); record_nonfinite copies the latent's git_sha.
  - Documents patches applied: 18 commits 38b65b0..fd9c8e3. Spot checks pass: power_without_sim1.py regenerates its JSON byte for byte; lam_audit.py output identical; sim1 share 0.666; the no-pair figures equal the documents review's model (25.92 of 27, 18 of 18, 6.673 of 48, 1.076 of 24, 0.3615 of 60) and its assumption holds in scvi-tools 1.4.2 (n_train = ceil(0.9 N), partial last minibatch kept).
  - Lead commits: 4af5069 ANALYSIS_SPEC sections 0-16 rules only, Appendix A decision log (options and reasons moved verbatim, checked line by line; A.2 why there is no Q8; A.3 open items), status line, A3 Sinkhorn line, 408 X13 rows in the run condition, R1 record source after the Sinkhorn redo, Holm figures (DOC-05), power sentence (DOC-07), proposal labels (DOC-08), terms and labels (DOC-11), tense (DOC-12). 06345ce PREREG SI-54 sentence names divergence records (CF-02); SI-68's scope 'an X1 result' restored in PREREG and PAPER_PLAN; glossary: matched lambda among points with no failed seed.
  - Test gates on tree 46a2df12 (= fd9c8e3; the two lead commits change only docs that no test reads): scripts tier exit 0, 421 passed, 18 skipped, check_tests OK; src tier exit 0, 172 passed, OK (commands in lead_gates README).
  - Notebook: -29 code RESULT, -30 code DECISION, -31 documents RESULT, -32 spec RESULT, -33 SI-68 scope DECISION, -34 DOC-18 ledger annotations NOTE, -35 CORRECTION of NB-20261013-03 (268 captures, not 236). nb_check ok.
- Not done: the independent re-checks; the user's diff review. Not recomputed by anyone this round: the F7 Sinkhorn-to-W1 ratios (marked provisional in the text); the 8 of 54 recount rests on NB-20261014-22.
- Found, not yet raised with the user: prereg_rules.power_report starts its root search at 1.0, and scipy's non-central t returns NaN above a non-centrality of about 38, so it fails when the standard error is below about 0.026 (the without-sim1 value is 0.0236); the power recompute after the Sinkhorn redo may hit it (documents report section 6, item 1).

## Artifacts
- [lead_gates_20261014.tar.gz]({{artifact:856fc931-0942-4a80-810f-a589ac12f6b5}}): this session's evidence: both test-tier logs on tree 46a2df12, the import-closure output c3fd92d..06345ce, README with the exact commands.
- [notebook_2026-10-14.md]({{artifact:48954b1b-f83f-4122-83c5-758609540628}}): notebook day page through NB-20261014-35.
- [notebook_INDEX.md]({{artifact:b85f1069-598d-49e3-9e73-5908ffffa4ae}}): notebook index.
- [rigor_review_20261014_findings.json]({{artifact:40b79c70-4782-43eb-ab8b-24cb05be2520}}): findings of the three reviews at 1e615eb (keys analysis, code, docs): give each re-checker its key.
- [RIGOR_REVIEW_TRIAGE_20261014.md]({{artifact:739db527-636e-45f3-afef-1d5274db518c}}): which findings were assigned to which track.
- [CF_FIXES_REPORT.md]({{artifact:fa1b46f9-cff0-48d8-a6d5-df83ce0e1e14}}): code-fix sub-agent report (section 6: its questions).
- [DOC_FIXES_REPORT.md]({{artifact:30dd365f-0a7b-4399-9754-7d9629e237df}}): documents sub-agent report (section 4: 8 user questions; section 6: other findings).
- [subagent_results.json]({{artifact:65d1f7d9-678f-4f87-b493-e4ab917cb64b}}): both sub-agents' structured results and declared limits.
- [rigor_code_20261014_scripts_outputs.tar.gz]({{artifact:8d340d79-13d7-4c47-9f88-6db2ab9e9fcc}}): code-review probes (probes/import_closure.py, my_mutate.py).
- [rigor_docs_review_bundle.tar.gz]({{artifact:13872d76-d52b-4fe9-aae5-fea2dec27723}}): documents-review scripts (scripts/recompute_g01.py: the no-pair model).
- [CONSTRAINTS.md]({{artifact:6da7d2d3-47fa-4625-8a3a-f10c7b9af6e9}}): decision ledger; unchanged this session.
- [HANDOFF-final-review-fixes-9.md]({{artifact:70129017-cc01-4f0e-94dd-33a25f430258}}): previous handoff (part 9).

## Kernel state (will be lost)
Nothing expensive. Workspace-only, swept: scratch clones gate/, gate_src/, spot/; venvs covvenv (env scvi-api; rebuild `python -m venv --system-site-packages covvenv && covvenv/bin/pip install pytest-cov==7.1.0 ruff==0.14.0`) and covvenv_gpu (env wcd-gpu; same without ruff); extracted tars cf/, doc/, probes/, cf_evidence/, doc_tools/, docs_review/ (re-extract from their artifacts). Notebook commits need GIT_AUTHOR_NAME/GIT_COMMITTER_NAME=wcd and GIT_AUTHOR_EMAIL/GIT_COMMITTER_EMAIL=noreply@localhost in os.environ; push from bash with credentials=['GitHub'] and a credential helper reading $GITHUB_TOKEN.

## Decisions
- Agent: CF patches first, then documents; the 24-patch tree was tested in private clones and is identical to the main repo's (46a2df12).
- Agent (NB-20261014-33): where PREREG and PAPER_PLAN cite SI-68 for the later lock of the follow-up analyses, keep the user's scope 'an X1 result'; 'adversarial-arm score' only for the freeze of the X1 analyses.
- Agent: DOC-18 annotations go in the notebook (NB-20261014-34), not CONSTRAINTS.md: the ledger tools rewrite entries from known fields only and would drop a note line.
- Agent: in the DOC-13 restructure the sensitivity variants (per-run reading, corner (b0, 0), pairs first, six independent datasets) stay defined in the body; only unchosen options, reasons and open items moved.
- Agent: after the Sinkhorn redo the analysis gets the redone R1 record through --r1-record; no code change (arm_points stops on a grid the X1 rows do not carry).

## Dead ends
- Both check_tests tiers at once in one clone: they share .check_tests/ and corrupt each other's coverage files (264 setup errors). One clone per tier.
- wcd-gpu tier without KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL: torch aborts at import (OMP Error #179).
- host.archive.search searches only the current chat; for another chat query frame_messages (frame_id, idx, msg_json) with host.query.

## Next steps
1. Focused re-checks: three RIGOR_REVIEWER sub-agents in parallel, one per track. Give each a git bundle of final-review-fixes at 08f1e3c with tags prereg-tier12-v1 and v2 (`git bundle create` with refs/heads/final-review-fixes refs/tags/prereg-tier12-v1 refs/tags/prereg-tier12-v2; save as an artifact), its key of the findings JSON, its fix report and commit range: analysis 8b90d79, 6aaa3ac, 56febe4 and the spec text 4af5069 (fix record NB-20261014-23 to -28, experiments/analysis_numerics_20261014/README.md); code b82bf49..2405cd1 (CF_FIXES_REPORT); documents 38b65b0..fd9c8e3, 4af5069, 06345ce (DOC_FIXES_REPORT). Rules: no X1 score values (X1_local, jhpce_tier12, review_sealed, X1 logs, score_identity files), no writes to the repo, no pushes, no notebook entries, no cluster jobs; the test recipe above; an output_schema with a verdict per finding (closed, partly, not closed), new findings and an evidence artifact.
2. Fix what the re-checks find, with notebook entries; re-run the affected tier.
3. The user's diff review of 5cb75c0..HEAD: commits grouped by topic in plain words, then all open questions below in one pass (ask_user, one question per tab).
4. After sign-off: ledger entries for the approved proposals (DOC-08), tag prereg-tier12-v3 and push; then HANDOFF-4 steps 7-11 (TAG_SHA G-17, G-15 fallback, preflight, Sinkhorn redo per SI-57, CR2-01 and JHPCE X1 only on the user's go).

## Open questions for the user (at the diff review)
- Analysis designs: AN-01 log-spaced grid; the Delta precision fix; zero_spread below s = 1e-6 versus a floor on s (AN-02); recomputed R4 targets (AN-03); t-based sign-table limits (AN-06).
- SI-75: does it stand now that the recount gives 8 of 54 (+3.7 points, not +0.7)?
- The definition of 'examined'; did anyone read an X1 adversarial-arm score value?
- Approve the agent's proposals as ledger entries: spec 2.3, 5 (Holm over six), 8 (transfer rule), 9, 10, the unnamed parts of 11, and 15; the documents' proposals (each NMI-0 case decided by the user and listed; a cap hit stops the stage until the user decides; no-pair reporting script after the runs).
- Keep the Sinkhorn A3 refits and the R2/R3 re-run that cannot change X1 while immune|1 is flagged?
- Names of the redone Sinkhorn records (docs/prereg/*_sinkhorn_redo.json).
- Lock PAPER_PLAN's scaling sensitivities, or leave them unspecified?
- Confirm the restored SI-68 scope (NB-20261014-33).
- Code: refuse a divergence record without git_sha? The other 22 unused-import findings in a separate commit? Commit fl_lint or its rule list? Which check_tests base for the final record (973a97d used)?
- prereg_rules.power_report fails for standard errors below about 0.026: fix its search start before the tag, or leave it?
- Still open from HANDOFF-8 and -7: DOC-01 disclosure wording, DOC-03 amendment-table format, deleting the old review worktrees and the two strays, I091/D-01/D-02 wording, disclosing SI-66's late implementation, the user's go for each cluster job.
