# Handoff: final-review fixes, part 13

**Session:** 2026-10-15 (EDT) · 407 messages, 515k tokens of latest context · 0 folds · frame d684be61-24c4-4b98-9c5d-458469d84733
**Status:** The three round-3 fixers have returned. Their patch series are ready but NOT applied: the analysis series has 26 patches, the code series 4, the documents series 14, all made against 5d06a27, with files that do not overlap. Branch final-review-fixes is at 2610107, pushed. Nothing is running. The handoff fired on size (314 messages and 445k tokens when first checked).

## Start message for the next chat
```
Continue from HANDOFF-final-review-fixes-13.md (artifact id 83562e03-35cd-453b-a798-3e0f55ca4f8e) in this project; read it before doing anything else.
Objective: apply the round-3 patches, test, ask 4 user questions, re-check, final diff, tag prereg-tier12-v3.
Next action: git am the three patch series onto 2610107 per Next steps.
Open first: HANDOFF-final-review-fixes-13.md; r3_fixer_results_20261015.json; CONSTRAINTS.md
Lost with the old kernel: nothing expensive
Pending: patches unapplied; 4 user questions; findings 659801d7, 8e0ad5ed
Do not redo: 19-question round, SI-88; hash list and audit; fixer runs
Before reusing any identifier, number or quote from the previous chat (frame d684be61-24c4-4b98-9c5d-458469d84733), search its archive.
```

## Objective
Benchmark Wasserstein critics against discriminator adversaries for single-cell batch correction under docs/PREREG.md. Now: apply the round-3 patches and the lead's small follow-ups, run both test tiers, ask the user the 4 open questions below, run a focused independent re-check, show the final diff for the user's yes (SI-52), tag prereg-tier12-v3 (SI-60) and record the tag's GitHub date (SI-79). Then the Sinkhorn redo (SI-57) and X1. Write in plain language with every number's source (SI-67, NO-JARGON rule).

## State
- Repository D/code/wcd_final-review-fixes, where D = /home/kendall/experiment_data/wasserstein-critic-deconfounding. HEAD 2610107, pushed (ls-remote checked). Ledger: 88 entries, 86 active. Notebook: 215 entries, hash chain OK.
- Commits this session after 5d06a27, all records or new lead files: SI-87 correction (cd15f3a, 77c16a4); SI-79 hash list a863197; audit eaf6cdf and 2610107; notebook NB-20261015-21 to -29; SI-88 (cd3f83e).
- Results of the three fixers (gates, deviations, open questions per track) are in the combined JSON below.
  - Analysis: tests/analysis and tests/prereg 201 passed, 1 skipped; check_tests OK; 72 of 73 planted errors caught, the survivor being M28.
  - Code: tests/stage and tests/prereg 206 passed, 2 skipped; tests/wcd_vae 172 passed; check_tests OK on both tiers; planted errors 10 of 10, 5 of 5 and 6 of 6.
  - Documents: run_all.sh exit 0; tests/prereg 131 passed, 1 skipped.
- Q18: the check found nothing beyond GitHub commits except Python cache files in 8 worktrees (wt_identity, wt_rereview_sinkhorn, wt_review_guards, wt_review_sinkhorn, wt_rigor_analysis_1e615eb, wt_rigor_code_1e615eb, wt_rigor_docs_1e615eb, wt_sinkhorn_pilot) and the 2 strays (rf_extract.txt, HANDOFF-final-review-fixes-4.md). delete_host_files refused 3 times ('masked inside a running sandbox'), so the user deletes them outside the app, then git worktree prune. wt_suite is KEPT: 347 tracked files are missing from its folder and its index records deletions, though all of its content is in a4f4be0.

## Artifacts
- [r3_fixer_results_20261015.json]({{artifact:6e0cf63b-f797-4d38-9a37-24acffd74225}}) — all three fixers: structured output, deviations, open questions, gates (read first)
- [analysis_r3_patches.tar.gz]({{artifact:bf5f5a87-0888-4cf5-9a9d-b7ff1076c878}}) — analysis series, 26 patches · [report]({{artifact:74bd2afb-559f-433d-b982-6551164364f6}}) · [evidence]({{artifact:3a279d86-838d-475f-bdd9-89f856aee152}})
- [code_r3_patches.tar.gz]({{artifact:37424731-7c19-40a0-9c6d-f721386c9d99}}) — code series, 4 patches · [report]({{artifact:39659d51-7dd2-4197-a8e5-e292d6fcd21f}}) (section 10 = proposed PREREG sentence and questions) · [evidence]({{artifact:768850be-15d2-439a-a462-2b030185cfaf}})
- [docs_r3_patches.tar.gz]({{artifact:17a2c4ce-496a-4d82-a5cd-0304862db8c4}}) — documents series, 14 patches · [report]({{artifact:99a615ca-1344-48cc-8e85-6432bac26153}}) · [evidence]({{artifact:48f2b26e-6eb3-48b9-ab4e-a0d8a0827873}})
- [wcd_final-review-fixes_5d06a27.bundle]({{artifact:8b17de10-a79c-49c8-b19c-37bceb145f21}}) — base of all three series (SHA-256 0c7de7d5...)
- [CONSTRAINTS.md]({{artifact:186cedf9-53e2-4d16-a838-8096cce90e2e}}) — ledger including the SI-87 correction and SI-88; identical to the repo file
- [notebook_2026-10-15.md]({{artifact:1f4de9fe-0eef-4993-9680-36de04f25f95}}) — today's notebook page (NB-20261015-01 to -29) · [index]({{artifact:28e5c49b-8b69-41ac-aeac-fb42f90b3efd}})
- [DIFF_REVIEW_final-review-fixes.md]({{artifact:917a8c6d-e0bf-4620-8cf9-1dc9a1c43734}}) — guide for the user; section 5 (full diff) and the answers still to add
- [DIFF_REVIEW_QUESTIONS_draft.md]({{artifact:1a9f2007-ce6d-4bfd-b26d-4842cd2d5f20}}) — the 19 questions (all answered)
- [lead_gates_20261014.tar.gz]({{artifact:856fc931-0942-4a80-810f-a589ac12f6b5}}) — exact test-tier commands (README.txt)
- [recheck_code_20261014_scripts_outputs.tar.gz]({{artifact:8f06d9c7-787a-4727-b853-7a4175b548f7}}) — ct_union_eval.py for the merged coverage record (Q15)
- Briefs: [analysis]({{artifact:a743a0ac-face-4748-9391-20ce5cf3fe76}}), [code]({{artifact:21448739-ffea-4cf7-b905-2018a7402427}}), [documents]({{artifact:4f6099c1-2629-41a0-bdd0-df5d8906445e}}); previous handoff [part 12]({{artifact:4dbff170-2e54-462b-8808-6400eb7d5b3d}}).

## Kernel state (will be lost)
Nothing expensive. A clone of the bundle sat at scratch/bundle_check; re-clone it from the bundle when needed.

## Decisions
- Agent: question 12 had told the user that tag v2 keeps the old record files. It does not: docs/prereg/ is empty at v2. SI-87 now says the versions before the redo stay at tag v3 and in commits 03882dc, 2f17f77, 55e66d4, dae54b0, cea50ca (NB-20261015-21). The user's choice is unchanged.
- Agent: the SI-79 list covers 827 files: score files, scoring logs, comparison tables, X1/X13 runner ledgers, failure tables, runner logs and review_sealed. Code checkouts, test scratch, latents and models are excluded.
- Agent: the audit (counts and digit-masked lines only) found two displays. (1) On 2026-10-02, a sub-agent's output showed all metrics of six test fits on atac_small from one of the two X13 runner tests. (2) On 2026-10-13, the lead session's identity check printed PCR_batch and cell-cycle conservation for X1 immune discriminator, grid point 1, seed 0, and for X13 pancreas sysVI, before the specification was drafted (21:22). The 49 final-review probe files are all registered configurations; none of their values was found displayed. NB-20261014-21 was corrected by NB-20261015-23.
- User SI-88 (NB-20261015-25): both displays are listed as 'shown to the agent only', not as examined.
- Agent: Codex was not used (pre-registration text needs judgement; the code edits are small).

## Dead ends
- host.query cuts every returned value at about 2,000 characters; read long transcript messages with substr(msg_json, start, 1500).
- *.csv and **/results/ are git-ignored: use git add -f. nb_entry commits only if GIT_AUTHOR_NAME/EMAIL and GIT_COMMITTER_NAME/EMAIL are set in os.environ (wcd / noreply@localhost).
- I typed a count into NB-20261015-22 (44 instead of 43; corrected in -24): compute notebook numbers in code. I first called the probe outputs 'unregistered' without checking the manifests (corrected).

## Next steps
1. Read the combined fixer JSON. Apply the series with git am through a python subprocess (bash runs out of the sandbox's 64-file git budget) on HEAD 2610107, in the order documents, code, analysis. Check that each series' file list matches its result.
2. Lead follow-ups, each its own commit:
   - PREREG 'Display 1' wording becomes 'six test fits from one of the two X13 runner-test runs'.
   - Replace the 10 app-only artifact links in PAPER_PLAN lines 7 and 215 with file names.
   - After the analysis patch, refresh the ANALYSIS_SPEC line references in experiments/doc_numbers_20261015/README.md and the PREREG question-9 pointer.
   - Add tests/stage/fake_env.py to the files line of ledger entry SI-85.
   - Notebook CORRECTION: '54 of 57' (2026-10-14.md:319, 2026-10-15.md:10) is now '72 of 73', and mutation_results_20261014.json is now mutation_results_20261015.json.
   - Settle the reviewer findings 659801d7 (compare_with_reference.py docstring says 18 hand-made cases, the code builds 17: was a case dropped?) and 8e0ad5ed (sim_pipeline header says S1 and S2 changed; only S1 was seen to change). Mark each addressed only once fixed.
   - Check that the groups in the hash README match PREREG's disclosure groups.
   - One notebook RESULT per fixer track, with numbers taken from the JSON.
3. Run both test tiers (gates pack README), the import-closure probe (the code fixer added closures_by_entry.py), and the merged three-tier coverage record from 5cb75c0 (Q15, NB-20261015-10; ct_union_eval.py). Push.
4. Ask the user the 4 open questions below. Apply the answers and record them in the ledger and notebook.
5. Make a new bundle, then a focused independent re-check (RIGOR_REVIEWER) of every commit after 5d06a27, including the audit README and the PREREG disclosure. Fix what it finds.
6. Final diff; guide section 5 plus the answers and SI-88 in section 6; the user's yes; tag prereg-tier12-v3; push; a notebook entry with the tag's GitHub date (SI-79).
7. Then G-17, G-15, preflight, the Sinkhorn redo (SI-57; records overwritten per SI-87), CR2-01 and JHPCE X1, each cluster job only on the user's go.

## Open questions for the user
1. SI-85 says 'Sinkhorn divergence record', but the code fixer applied the refusal to the divergence records of EVERY arm, the same way the existing rule treats latents. Restricting it to Sinkhorn is two conditions in scripts/run_stage.py.
2. A latent without a git commit is still accepted, because SI-85 covers records only. Should it be refused too (one line and one test case)?
3. SI-81 (c): does the tag freeze the reporting script that is written after the runs, or only the promise to write one?
4. SI-81 (b): PREREG keeps the clause that each iteration-cap decision 'is listed with the results'; the ledger text omits it. Check the wording the user approved.
For information: the analysis fixer changed specification section 14's command to `--src scripts`, because the file-path form does not measure coverage. Still pending: the user's yes on the final diff, and the go for each cluster job.
