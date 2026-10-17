# Handoff: final-review fixes, part 9

**Session:** 2026-10-14 (EDT) · about 300 messages · 0 folds · frame facce980-f4e2-409a-854a-f37907ceea28
**Status:** The 20 fix items of the rigour review at 1e615eb are all addressed in draft. The lead fixed and validated the X1-analysis track (and found and fixed one precision bug the review missed). Two sub-agents delivered the code-fix and document tracks as patch sets, which are NOT yet applied to the repository; both apply cleanly. Nothing is running. The handoff fired on size (467k tokens of latest context).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-9.md (artifact id f705b906-44df-43ae-8d27-3a02b129a6bd) in this project; read it before doing anything else.
Objective: finish the final-review fixes, show the user the diff, tag prereg-tier12-v3, launch X1.
Next action: read each diff, then git am the two sub-agent patch sets; verify CF-02 with import_closure.py; run the gates.
Open first: CF_FIXES_REPORT.md; DOC_FIXES_REPORT.md; subagent_results.json
Lost with the old kernel: nothing expensive
Pending: 24 unapplied patches; DOC-13; re-checks; user questions
Do not redo: analysis fixes AN-01 to AN-14 (8b90d79, 6aaa3ac, 56febe4); the three reviews at 1e615eb
Before reusing any identifier, number or quote from the previous chat (frame facce980-f4e2-409a-854a-f37907ceea28), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction (scVI backbone, scIB metrics) under the pre-registration (docs/PREREG.md). Finish the final-review fixes on branch `final-review-fixes`, show the user the full diff (SI-52), tag `prereg-tier12-v3` (SI-60), redo the Sinkhorn pilots (SI-57), then launch X1. Plain language throughout (SI-67, profile NO-JARGON rule).

## State
- Repository D/code/wcd_final-review-fixes (D = /home/kendall/experiment_data/wasserstein-critic-deconfounding), branch final-review-fixes, HEAD a90adcd, pushed (local = remote, checked with ls-remote). Works directly from the granted folder this session (no masking seen); use a private clone if git says 'not a git repository'.
- Lead commits this session: 8b90d79 (scripts/analysis_x1.py, tests/analysis, ANALYSIS_SPEC sections 2.3, 4, 14, 15.1, 15.4, 15.9-15.11), 6aaa3ac (experiments/analysis_numerics_20261014: script, outputs, planted-error runner and results), 56febe4 (ANALYSIS_SPEC section 16 limitations, pilot-informed labels, b* after R4 extensions, run commands), notebook NB-20261014-23 to -28 (last commit a90adcd).
- Analysis fixes (all agent proposals for the diff review): AN-01 spreads integrated on 201 points evenly spaced in log(tau) from 1e-15 to 5 prior scales; NEW precision fix: Delta's posterior from solves with A (old formula lost up to 0.2% of the variance at s = 1e-4, unusable near 1e-6; ISSUE NB-20261014-24, CORRECTION -25); AN-02 a dataset with s below 1e-6 is not evaluable ('zero_spread'); AN-03 all four R4 targets recomputed with prereg_rules.r4_stage (extension rows included), refuse on difference; AN-05 spec 15.11 wording; AN-06 sign-table limits use t(0.975, m-1); AN-10 seeds 0-4; AN-11 rows sorted by tag; AN-13 best-of-k records configs and failed runs; AN-14 window gap flag; CF-09 scipy trapezoid.
- Evidence (committed, reproducible: experiments/analysis_numerics_20261014/README.md): review case at prior 0.2 P(Delta > 0.01) old 0.94891 (inconclusive) vs amended 0.97570 (above, error 1.9e-12); 240 random worlds (s from 1.02e-6) largest error 1.99e-6, 0 declaration mismatches; planted errors 54 of 57 caught (M24 101 points: changes values by at most 1.54e-5; M28: 1.96e-6; N11 row order only); gate `scripts/check_tests.py --base 973a97d --src scripts -- tests/analysis tests/prereg` OK, 193 passed, 1 skipped; tests/analysis 62 passed (about 55 s).
- Sub-agent patch sets (NOT applied; tested: `git am --3way` of all 6 + 18 patches onto a90adcd in a scratch clone gives 24 commits, 16 files, no conflicts):
  - Code fixes (RUNNER_ENGINEER, frame 35f15d5f-52a7-4675-a7bd-23adccc80955): CF-01 fixed, CF-02 fixed (scripts/run_stage.py, 9 lines added, 3 removed), CF-08 fixed, CF-03 test part, CF-05 comments, CF-10 one import. Their gates: scripts tier check_tests exit 0 (359 passed, 18 skipped), src tier 172 passed. Declared limits: identity proof not re-run (argued from the import closure), opt-in end-to-end stage test not run, data tests only for tests/scvi.
  - Documents (MANUSCRIPT_WRITER, frame e9b32baf-712b-458c-83b5-9f4f7cc35bff): DOC-01..09, 11, 12, 14, 15, 19, CF-03 README, CF-04, CF-06, CF-07 in PREREG/PAPER_PLAN/EQUIVALENT_COMMITS/README plus experiments/guard_followups_20261013/lam_audit.py and a new experiments/final_review_amendments_20261013/power_without_sim1.py. Their gates: tests/prereg 131 passed, tests/stage 65 passed. Declared limits: some numbers quoted from the review, not recomputed (no-pair 25.9 of 27, 14%, 4.5%, 0.6%; F7 ratios; 8 of 54); spec-side parts left to the lead (their report section 5 has the wording).
- Not yet done by anyone: DOC-13 (spec status line, unchosen options to an appendix decision log, stale A3 line at spec:29, the Q8 gap, X13 rows in the run condition); spec parts of DOC-01, -05, -06, -07, -08, -12; DOC-18 ledger annotations; CF-02 import-closure check by the lead.
- Untracked strays rf_extract.txt and HANDOFF-final-review-fixes-4.md: only the user can delete them; never `git add -A`.

## Artifacts
- [subagent_results.json]({{artifact:65d1f7d9-678f-4f87-b493-e4ab917cb64b}}): both sub-agents' full results (items, gates, deviations, open questions).
- [cf_patches.tar.gz]({{artifact:7710df50-9b51-47f3-9733-1b3fba00d649}}) and [CF_FIXES_REPORT.md]({{artifact:fa1b46f9-cff0-48d8-a6d5-df83ce0e1e14}}) (section 6: decisions to log), evidence [cf_evidence.tar.gz]({{artifact:5a10b655-d89e-4f42-b4da-5c706e6c7753}}).
- [doc_patches.tar.gz]({{artifact:061b61b4-63de-4e63-9990-51241af83104}}) and [DOC_FIXES_REPORT.md]({{artifact:30dd365f-0a7b-4399-9754-7d9629e237df}}) (section 5: proposed ANALYSIS_SPEC wording; 8 user questions), [DOC_FIXES.receipt.txt]({{artifact:a63c1567-5da6-4e58-9c98-32c25f204320}}), tools [doc_fix_tools.tar.gz]({{artifact:e93ef2bd-1974-4f59-a55c-919ffd94115c}}).
- Still canonical from part 8: [RIGOR_REVIEW_TRIAGE_20261014.md]({{artifact:739db527-636e-45f3-afef-1d5274db518c}}), [rigor_review_20261014_findings.json]({{artifact:40b79c70-4782-43eb-ab8b-24cb05be2520}}), [HANDOFF-final-review-fixes-8.md]({{artifact:aa01e99a-8f71-4689-a275-b95c7ccc502b}}), code-review probes incl. probes/import_closure.py [rigor_code_20261014_scripts_outputs.tar.gz]({{artifact:8d340d79-13d7-4c47-9f88-6db2ab9e9fcc}}), [CONSTRAINTS.md]({{artifact:6da7d2d3-47fa-4625-8a3a-f10c7b9af6e9}}).

## Kernel state (will be lost)
Nothing expensive: review findings, ledger, stress-test worlds and results all reload from the artifacts above or from experiments/analysis_numerics_20261014. Workspace-only (swept): covvenv (rebuild: `python -m venv --system-site-packages covvenv && covvenv/bin/pip install pytest-cov==7.1.0 ruff==0.14.0` in env scvi-api), mut/ (6 mutation clones), patchtest/ (scratch clone). mpmath was installed into env python. Notebook commits need GIT_AUTHOR_NAME/GIT_COMMITTER_NAME=wcd and GIT_AUTHOR_EMAIL/GIT_COMMITTER_EMAIL=noreply@localhost in os.environ; push via bash with credentials=['GitHub'] and a credential helper reading $GITHUB_TOKEN.

## Decisions
- Agent: three tracks in parallel; the lead kept scripts/analysis_x1.py, tests/analysis and ANALYSIS_SPEC.md; sub-agents worked in private clones, delivered `git format-patch` sets and wrote no notebook entries (the hash chain would conflict).
- Agent (diff-review items): AN-01 log-spaced trapezoid instead of handoff 8's graded Gauss-Legendre (same accuracy, a 2-line change); AN-02 a 1e-6 floor marks zero_spread rather than flooring s (any floor value would be arbitrary; 1e-6 is about 300 times below the smallest pilot s and is where double precision was verified); AN-03 recompute with r4_stage including extension rows (PREREG section 4 defines b_common over all stage points). Recorded as DECISION NB-20261014-26.
- Agent: test extensions done in Claude, not Codex: codex_loop.py cannot edit existing test files and rejects tests of already-correct code (memory mem_c9265b3381be, mem_1dfaf998379e). Codex was not used.

## Dead ends
- Nested scipy `quad` over both spreads as a reference: more than 10 minutes for one case; a vectorised trapezoid in log(tau) converges to 1e-11 in seconds.
- The lead's first reference used the same cancellation-prone variance formula as the code (CORRECTION NB-20261014-25); references must use the precision form.
- frame_messages has columns frame_id, idx, msg_json, msg_uuid (no role, no rowid); memories has body, not content.
- A collect loop in a backgrounded repl cell blocks the repl; peek at children with `fresh: true` cells.
- `*.csv` is git-ignored; result tables in experiments/ need `git add -f` (26 already tracked that way). `.check_tests/` is ignored.

## Next steps
1. Apply the patch sets in the main repo, CF first then documents: extract the two tars and `git -c user.name=wcd -c user.email=noreply@localhost am --3way $(ls $WS/cf/patches/*.patch | sort)`, then the same with $WS/doc/doc_patches/*.patch (the code tar extracts to patches/, the documents tar to doc_patches/; $WS = the workspace folder they were extracted into); read every diff before applying (the reports list commit by commit). Then verify CF-02 yourself: `python $WS/probes/import_closure.py . c3fd92d HEAD` from the repo root, after extracting the code-review tar (8d340d79, listed under Artifacts) into $WS (probes tar above) must show that only scripts/run_stage.py changed in the closure and only in the SI-54 start-up and final checks; otherwise re-run the identity proof (experiments/integration_checks_20261013/README.md). Spot-check the document numbers the agent did not recompute.
2. Gates on the applied tree: scripts tier `covvenv/bin/python scripts/check_tests.py --base 973a97d --src scripts -- tests/analysis tests/prereg tests/scvi tests/stage tests/jhpce tests/test_legacy_retired.py` (env scvi-api, WCD_SRC=src); src tier tests/wcd_vae (env wcd-gpu, PYTHONPATH=src). Then one notebook RESULT per applied track (decisions from CF report section 6), commit, push.
3. Spec (lead): DOC-13 restructure and the spec parts of DOC-01, -05, -06, -07, -08, -12 using DOC report section 5; add divergence records to the PREREG SI-54 sentence if the documents patch does not (CF open question 2); DOC-18 ledger annotations without changing the user's words.
4. Focused re-checks: new RIGOR_REVIEWER sub-agents, one per track, given the findings JSON and the new commits (the part-8 reviewers belong to another chat and cannot be messaged).
5. The user's diff review of 5cb75c0..HEAD: commits grouped by topic in plain words, all open questions below in one pass.
6. After sign-off: tag prereg-tier12-v3 (SI-60) and push; then HANDOFF-4 steps 7-11 (TAG_SHA G-17, G-15 fallback, preflight, Sinkhorn redo per SI-57, CR2-01 and JHPCE X1 only on the user's go).

## Open questions for the user (at the diff review unless noted)
- Analysis designs: the log-spaced grid (AN-01), the precision fix, zero_spread below 1e-6 versus a floor on s (AN-02), recomputing the R4 targets (AN-03), t-based sign-table limits (AN-06).
- Does SI-75 stand now that the unconditioned pilot difference is +3.7 points (8 of 54), not +0.7?
- Documents agent (8): could anyone have read an X1 adversarial-arm score value (the audit cannot rule out that a file was opened); the definition of 'examined'; SI-75; keep the Sinkhorn A3 refits and R2/R3 re-run that cannot change X1; who decides NMI-0 cases and how a cap hit is classified; names of the redone Sinkhorn records (docs/prereg/*_sinkhorn_redo.json); lock PAPER_PLAN's scaling sensitivities; approve the agent's proposals (spec 2.3, 5, 8, 9-11, 15) as ledger entries before the tag.
- Code agent (5): refuse a divergence record without a git_sha; the other 22 unused-import findings in a separate commit; commit fl_lint or its rule list; which check_tests base for the final record.
- Still open from HANDOFF-8 and -7: the leave-one-family-out transfer rule (DOC-08), DOC-01 disclosure wording, DOC-03 amendment-table format, deleting the old review worktrees and the two strays, I091/D-01/D-02 wording, how to disclose SI-66's late implementation, the user's go for each cluster job.
