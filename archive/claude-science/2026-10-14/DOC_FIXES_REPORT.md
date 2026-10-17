# Document fixes for the rigour review of branch `final-review-fixes` (documents track)

Prepared 2026-10-14. Base commit 973a97d (the review covered 1e615eb). The tag `prereg-tier12-v3` was not made and nothing was pushed.

**Terms.** SI-nn is entry nn of the decision ledger (CONSTRAINTS.md). NB-yyyymmdd-nn is an entry of the lab notebook. DOC-nn and CF-nn are findings of the independent documents review and code review of 2026-10-14. R1 to R4 are the four decision rules of the pre-registration (PREREG.md). P1 and P2 are the two primary endpoints: biology kept at equal batch removal, and how often runs collapse. A penalty strength is the weight lambda of the batch-alignment term. The glossary that patch 0018 adds to PREREG section 9 defines the other labels.

## 1. What is delivered

1. `doc_patches.tar.gz`: 18 patch files made by `git format-patch 973a97d..HEAD`, one commit per finding. Apply them with `git checkout 973a97d` and `git am doc_patches/*.patch`. I tested this in a fresh clone: the patches apply cleanly and the resulting tree is identical to my private clone. The commit hashes below are those of my private clone; `git am` creates new hashes, and the first line of each patch file carries the original.
2. This report.
3. `doc_fix_tools.tar.gz`: the scripts that rebuild the 18 commits from the base commit, regenerate the numbers quoted in them, and run the reference and number checks of section 7 (the two test suites are run with pytest).
4. `DOC_FIXES.receipt.txt`: the claim receipt (44 readbacks behind the numbers).

Size of the change: 8 files changed, 491 insertions(+), 131 deletions(-). Files changed: docs/PREREG.md, docs/PAPER_PLAN.md, docs/EQUIVALENT_COMMITS.json and experiments/guard_followups_20261013/README.md; outside these four, the existing experiments/guard_followups_20261013/lam_audit.py and results/lam_audit.txt (edited, CF-03), and the new experiments/final_review_amendments_20261013/power_without_sim1.py with results/power_without_sim1.json (DOC-07). Not touched: notebook/, CONSTRAINTS.md, docs/ANALYSIS_SPEC.md, scripts/analysis_x1.py, tests/analysis/, scripts/fit_paper_config.py.

## 2. Result at a glance

| Finding | What it is about | Action | Patch (commit) | Still open |
|---|---|---|---|---|
| DOC-01 | 19 early X1 scores and the meaning of 'examined' | partly fixed | 0012 (0ccbb57) | User: did anyone read a value; approve 'examined'. Lead: ANALYSIS_SPEC lines 10-11 and 40. |
| DOC-02 | Sinkhorn redo described as a fixed list of 234 refits | fixed | 0006 (03a01b2) | none |
| DOC-03 | 'nothing chosen after seeing data' no longer holds for version 3; file list out of date | fixed | 0016 (02a45fc) | none |
| DOC-04 | low-signal rule for R2 and R3: dates, flagged blocks, old record | partly fixed | 0004 (3e2e92f) | User: keep the Sinkhorn A3 refits and the R2/R3 re-run. Lead: spec section 2.5. |
| DOC-05 | Holm correction: four or six tests | fixed | 0001 (78bde1a) | none |
| DOC-06 | 'the frozen X1 grids stand' is not true for Sinkhorn | partly fixed | 0003 (6e051bb) | User: names of the redone records. Lead: spec lines 55-56. |
| DOC-07 | power figures: what they measure, and the effect of the simulation sim1 | partly fixed | 0002 (fa2d4fb) | Lead: spec section 2.2. The power estimate is to be recomputed after the Sinkhorn redo. |
| DOC-08 | user decisions versus the agent's proposals | partly fixed | 0013 (96eab10) | User: approvals become ledger entries after the diff review. Lead: spec lines 6-9 and 187. |
| DOC-09 | PAPER_PLAN counts, datasets and marker count out of date | fixed | 0014 (14be8e7) | User: lock the scaling sensitivities or not. |
| DOC-11 | too many unexplained labels and terms in PREREG section 9 | partly fixed | 0018 (b3a5e20) | Lead: ANALYSIS_SPEC (51 SI, 30 I and 17 D references). |
| DOC-12 | steps not yet done written as done | partly fixed | 0017 (e84fad0) | Lead: spec lines 10 and 248. |
| DOC-14 | NMI-0 case and iteration-cap hit: who decides, how classified | fixed | 0011 (139369e) | User: approve the two proposals. |
| DOC-15 | evidence for the 'result-identical' list | fixed | 0009 (8284fda) | none |
| DOC-19 | number provenance (322 vs 382, gradient ratio, 3,076 minibatches, no-pair shares, F7) | partly fixed | 0007 (412413d) | Lead: notebook CORRECTION for NB-20261013-03 (236 captures; correct is 268). |
| CF-03 | README says 1,908 manifest rows (occurrences, not distinct rows) | partly fixed | 0015 (2871c5f) | Code track: pin the EXTRA_ARMS value sets in tests/scvi. |
| CF-04 | two limits of the mixed-commit rule not stated | fixed | 0010 (12d6f20) | none |
| CF-06 | no-pair counts promised in the results, but no code reads them | fixed | 0008 (7d22d68) | The reporting script is written after the runs. |
| CF-07 | pilot R2/R3 counts quoted from the old record | fixed | 0005 (5485b85) | none |

No finding was rebutted. Every finding was checked against its primary source before I changed anything (section 3). 'Partly fixed' means that the part in PREREG.md, PAPER_PLAN.md, EQUIVALENT_COMMITS.json or the README is done and a part in a file I may not touch (ANALYSIS_SPEC.md, CONSTRAINTS.md, the notebook, the tests), or a step that needs the user, is left.

## 3. What I checked and changed, finding by finding

**DOC-01, 19 early X1 scores and the meaning of 'examined'.** Action: partly fixed. Patch 0012, commit 0ccbb57.

- Checked: NB-20261012-04/-05 (X1 started 12:40 and stopped 16:36: 19 scored, 1 fitted but not scored, 1,097 still to fit); NB-20261014-21 (audit of chat records: by file name 17 adversarial-arm rows on immune, conditioned decoder, seed 0 [discriminator 6, pooled 5, reference 6] and 2 plain rows); NB-20261013-12/-16/-40 (one X1 latent rescored by script, equality only). I opened no X1 file.
- Changed, and what is left: PREREG section 9: new entry of 2026-10-14 (what ran, what was scored, what the audit covers and cannot show); entry heading now says the analyses 'will be frozen before any X1 adversarial-arm score is examined'; 'examined' defined as a proposal; PAPER_PLAN section 5 points to the entry. Open: the user confirms whether anyone read a value; approves the definition; ANALYSIS_SPEC lines 10-11 and 40 (lead).

**DOC-02, Sinkhorn redo described as a fixed list of 234 refits.** Action: fixed. Patch 0006, commit 03a01b2.

- Checked: SI-57 text; counts from the manifests: 180 A1 + 18 R1-extension + 36 A3 = 234 Sinkhorn fits before the fix.
- Changed, and what is left: PREREG Sinkhorn entry: the redo in rule order (SI-57) with 'will be'; the 18, 36 and 234 are the pre-fix counts and may change.

**DOC-03, 'nothing chosen after seeing data' no longer holds for version 3; file list out of date.** Action: fixed. Patch 0016, commit 02a45fc.

- Checked: tag prereg-tier12-v2 resolves to c88cce3; decision times of SI-51 to SI-77 from notebook/INDEX.md; commit times from git log; R4-a1 record committed 2026-10-10 (dae54b0), R2/R3 record 2026-10-12 (cea50ca); A2/A3 runner ended 2026-10-11.
- Changed, and what is left: PREREG line 3 (status of version 2 and version 3), line 5 (file list adds ANALYSIS_SPEC.md, analysis_x1.py, tests/analysis, EQUIVALENT_COMMITS.json), section 9 lead-in, and an 8-row table: decided, committed, data that existed, result changed.

**DOC-04, low-signal rule for R2 and R3: dates, flagged blocks, old record.** Action: partly fixed. Patch 0004, commit 3e2e92f.

- Checked: SI-56, SI-66, NB-20261014-14/-15, f3c3415; R4-a1 record flags immune|1 (b0 0.6382, b_common 0.6570, gap 0.0188) and sim1|1 (0.8120, 0.8189, 0.0069); re-ran prereg_rules.decide_binary on the stored R2/R3 cells with the SI-66 marking (the unmarked run reproduces the record): R2 12/12 (-0.0014) -> 9/12 (-0.0042); R3 17/20 (+0.0010) -> 13/20 (-0.0014, 14 needed); both keep their defaults.
- Changed, and what is left: PREREG entry on low-signal blocks (decided 2026-10-13, implemented 2026-10-14; both flagged blocks; the record predates the rule; 'exist' instead of 'still fitted'). Open: keep the Sinkhorn A3 refits and the R2/R3 re-run (user); ANALYSIS_SPEC section 2.5 (lead).

**DOC-05, Holm correction: four or six tests.** Action: fixed. Patch 0001, commit 78bde1a.

- Checked: scripts/analysis_x1.py:479-482 (six primary results); prereg_rules N_ENDPOINTS = 4 (record stores 0.0125); smallest exact two-sided Wilcoxon p for 6 tasks = 0.03125.
- Changed, and what is left: PREREG section 5 (bracketed amendment), entry (d) and PAPER_PLAN section 5: first threshold 0.05/6 = 0.0083 in the frozen analysis and 0.0125 in the stored record; the companion cannot reject at the Holm-adjusted level in either version.

**DOC-06, 'the frozen X1 grids stand' is not true for Sinkhorn.** Action: partly fixed. Patch 0003, commit 6e051bb.

- Checked: docs/prereg/r1_record.json and the trajectory figures (sigma_c with and without trajectory); which figures come from pre-fix Sinkhorn fits.
- Changed, and what is left: PREREG I091 entry: 'the frozen X1 grids stand for every arm except Sinkhorn', Sinkhorn figures labelled pre-fix, proposed names docs/prereg/{r1_record,r4_a1_record,r2r3_record}_sinkhorn_redo.json. Open: names (user); ANALYSIS_SPEC lines 55-56 (lead).

**DOC-07, power figures: what they measure, and the effect of the simulation sim1.** Action: partly fixed. Patch 0002, commit fa2d4fb.

- Checked: reproduced r1_record.json power (sigma 0.0298, tau 0.1094) to 1e-8 with a new script, then dropped the sim1 blocks.
- Changed, and what is left: PREREG entry (c) and PAPER_PLAN Power bullet. Six tasks: SE 0.045, minimal detectable effect 0.131, power 0.104; without sim1: SE 0.0236, minimal detectable effect 0.068, power 0.182; equivalence unreachable in all four cases. New files: experiments/final_review_amendments_20261013/power_without_sim1.py and results/power_without_sim1.json. Open: ANALYSIS_SPEC section 2.2 (lead).

**DOC-08, user decisions versus the agent's proposals.** Action: partly fixed. Patch 0013, commit 96eab10.

- Checked: CONSTRAINTS.md: SI-55 covers the grid edge only; SI-68 to SI-77 are the user's decisions for the X1 analyses; ANALYSIS_SPEC sections 2.3, 5, 8, 9 to 11 and 15 are proposals.
- Changed, and what is left: PREREG and PAPER_PLAN: SI-55 narrowed to part (a); only SI-68 to SI-77 called user decisions; Holm over six, the transfer rule and equal k labelled proposals pending the diff review. Open: record the user's approvals as new ledger entries after the diff review; ANALYSIS_SPEC lines 6-9 and 187 (lead).

**DOC-09, PAPER_PLAN counts, datasets and marker count out of date.** Action: fixed. Patch 0014, commit 14be8e7.

- Checked: manifest of record (6,930 rows; per-experiment counts; 480 X1 barycenter rows); TASKS in build_paper_manifest.py against docs/prepped_fingerprints_scib.json; CONSTRAINTS SI-02, SI-03, SI-14, SI-18.
- Changed, and what is left: PAPER_PLAN: dated 'superseded' notes at sections 4 and 7; [dropped: SI-02] on five dataset rows; decisions 4 and 5 closed; the pointer's marker count ('seven statements, eight markers') verified by grep; the scaling sensitivities marked as not covered by ANALYSIS_SPEC.

**DOC-11, too many unexplained labels and terms in PREREG section 9.** Action: partly fixed. Patch 0018, commit b3a5e20.

- Checked: definitions checked against CONSTRAINTS.md, docs/review_final/REVIEW_FINAL.md, notebook entries, PREREG sections 1 to 5, PAPER_PLAN 3.1 and 3.2, and alignment.py.
- Changed, and what is left: PREREG section 9: glossary (9 label rows, 26 term rows), each entry heading says what its findings and decisions are, guards entry rewritten as a list, 'penalty-strength family' versus 'dataset family', block versus cell versus single cell; PAPER_PLAN note glossed. ANALYSIS_SPEC not touched (lead).

**DOC-12, steps not yet done written as done.** Action: partly fixed. Patch 0017, commit e84fad0.

- Checked: no tag prereg-tier12-v3 (NB-20261014-19); no tracked file or notebook entry for the Sinkhorn redo; the diff review (SI-52) has not happened.
- Changed, and what is left: PREREG: a status line (done with date and commit, or planned) at the start of each 2026-10-13 and 2026-10-14 entry; 'will be' for unfinished steps in PREREG and PAPER_PLAN. Older entries are frozen in version 2 and carry none. ANALYSIS_SPEC lines 10 and 248 (lead).

**DOC-14, NMI-0 case and iteration-cap hit: who decides, how classified.** Action: fixed. Patch 0011, commit 139369e.

- Checked: SI-61 and SI-65 text in CONSTRAINTS.md.
- Changed, and what is left: PREREG: the user decides each NMI-0 case and it is listed with the results; a cap hit stops the stage until the user decides whether it is recorded as a failed fit (counted in P2). Both are labelled proposals.

**DOC-15, evidence for the 'result-identical' list.** Action: fixed. Patch 0009, commit 8284fda.

- Checked: fit_identity_c3fd92d.csv flags only (63 rows, 59 of 59 non-Sinkhorn identical, 4 Sinkhorn differ, 0 non-zero exits); NB-20261013-39 (fits), NB-20261013-40 (scores); git diff c3fd92d..973a97d over src/ and the fit and score import closure is empty.
- Changed, and what is left: docs/EQUIVALENT_COMMITS.json: evidence list of objects naming the c3fd92d files and notebook entries, plus key valid_for_a_candidate_commit_if; PREREG SI-54 entry. tests/prereg 131 passed, 1 skipped; tests/stage 65 passed, 1 skipped.

**DOC-19, number provenance (322 vs 382, gradient ratio, 3,076 minibatches, no-pair shares, F7).** Action: partly fixed. Patch 0007, commit 412413d.

- Checked: 3,076 validation minibatches = 2,808 final + 192 early + 76 training-mode; gradient-norm ratios A1 2.00, A3 2.27, R1 extension 14.4 (cosine 0.28; 204 of 216 finite); denominators of the no-pair shares from the manifest.
- Changed, and what is left: PREREG and PAPER_PLAN wording. Not recomputed by me (taken from the documents review): the no-pair expectations (25.9 of 27, 14%, 4.5%, 0.6%) and the F7 ratios. NB-20261013-03 says 236 captures; the correct figure is 268 (192 + 76): needs a notebook CORRECTION entry (lead).

**CF-03, README says 1,908 manifest rows (occurrences, not distinct rows).** Action: partly fixed. Patch 0015, commit 2871c5f.

- Checked: ran lam_audit.py: 1,908 occurrences (1428/240/240) over 8 files; 382 distinct tags (302/40/40); manifest of record 322 (242/40/40); the 60 extra tags are X3 `none` rows of the first manifest, re-tagged under SI-41.
- Changed, and what is left: README row G-05 and the evidence row say what each number counts; lam_audit.py prints the three counts; results/lam_audit.txt regenerated. Not done: pinning the EXTRA_ARMS value sets in tests/scvi (code track).

**CF-04, two limits of the mixed-commit rule not stated.** Action: fixed. Patch 0010, commit 12d6f20.

- Checked: ran prereg_rules.equivalent_commits on synthetic commit lists in the patched tree: a whole input at one third commit, Sinkhorn included, is accepted; a Sinkhorn row at c88cce3 raises PreregError; with the tag absent from the checkout an old-commit/other-commit mix stays mixed.
- Changed, and what is left: PREREG SI-54 entry and docs/EQUIVALENT_COMMITS.json `meaning` state both limits; the evidence list names the c3fd92d files.

**CF-06, no-pair counts promised in the results, but no code reads them.** Action: fixed. Patch 0008, commit 7d22d68.

- Checked: grep over src, scripts, experiments and cluster outside tests: no_pair appears only in src/wcd_vae/wcd/alignment.py (the counter, stats['no_pair']) and scripts/fit_paper_config.py:504-506 (writes no_pair_steps into the plan record); nothing reads no_pair_steps.
- Changed, and what is left: PREREG: the counts will be tabulated by arm, task and experiment by a short reporting script written after the runs (proposal). The script itself is not written.

**CF-07, pilot R2/R3 counts quoted from the old record.** Action: fixed. Patch 0005, commit 5485b85.

- Checked: prereg_rules.decide_binary on the stored cells with and without the SI-66 marking (see DOC-04).
- Changed, and what is left: PREREG: the pilot R2/R3 counts come from the frozen c88cce3 record; the current code reports 9 of 12 and 13 of 20 for the same inputs.

## 4. Questions for the user

1. Did anyone read an X1 adversarial-arm score value? Nineteen X1 rows were scored on 2026-10-12, and one of them was rescored in the identity checks. The audit of every chat record found no value displayed or analysed, but it cannot show that nobody opened a score file. PREREG now lists this as an open item; I will not write that nobody read a value unless you confirm it.
2. Do you approve this definition of "examined"? An X1 adversarial-arm score is examined when a value of it is read by a person or entered into an analysis script; integrity checks that print only counts or flags do not count. Until you answer, PREREG labels it a proposal.
3. Does SI-75, which chose collapse rates plus a formal critic-versus-discriminator comparison with a 5-point margin, stand? The question behind it quoted 8 of 45 pilot runs for the discriminator with the unconditioned decoder; the recount is 8 of 54 (NB-20261014-22), so one quoted difference (critic minus discriminator) changes from +0.7 to +3.7 percentage points. Both differences are inside the 5-point margin, so I expect the decision to stand, but its stated premise was off.
4. Keep the Sinkhorn A3 refits and the R2/R3 re-run that SI-57 requires? The immune block with the conditioned decoder is flagged low-signal, and the Sinkhorn redo cannot clear the flag (Sinkhorn sets the common batch score in no block of the record), so R2 and R3 keep their defaults whatever A2 and A3 show. PREREG says so and asks.
5. Who decides a case where scib finds normalised mutual information (NMI) of 0 at every clustering resolution (SI-65), and how is an iteration-cap hit (SI-61) classified? PREREG proposes: you decide each NMI-0 case and it is listed with the results; a cap hit stops the stage until you decide whether it is recorded as a failed fit, which P2 counts.
6. Do you approve the names of the redone Sinkhorn records, docs/prereg/r1_record_sinkhorn_redo.json, r4_a1_record_sinkhorn_redo.json and r2r3_record_sinkhorn_redo.json? The frozen records stay unchanged.
7. Should the scaling sensitivities of PAPER_PLAN section 5 (leave-one-method-out rescaling, z-score aggregation, a composite without the silhouette metrics, rank stability) be locked in ANALYSIS_SPEC? Today they have no counterpart there, because the locked analyses use the unscaled selection score of SI-77. PAPER_PLAN marks them as a question for the diff review.
8. Do you approve the agent's proposals as new ledger entries: ANALYSIS_SPEC section 2.3 (not-evaluable rule), section 5 (Holm over the six primary results, first threshold 0.05/6 = 0.0083), section 8 (leave-one-dataset-family-out transfer rule), sections 9 to 11 and section 15? Each approval would be recorded as a new SI entry before the tag is made.

## 5. Proposed wording for docs/ANALYSIS_SPEC.md

I did not edit the specification. Line numbers refer to the file at 973a97d.

| Lines | Finding | Now | Proposed |
|---|---|---|---|
| 3-12 | DOC-08, DOC-13 | Status: DRAFT, 2026-10-13 ... decided by the user on 2026-10-14, all as recommended: Q1-Q5 ... and Q6, Q7, Q9, Q10, Q11 ... | Status, 2026-10-14: draft until the diff review. The user decided Q1 to Q7 and Q9 to Q11 (SI-68 to SI-77), all as recommended; Q8 was not asked. Details no question covered are the agent's proposals: section 2.3, section 5 (multiplicity, where Q8 would be), section 8 (the transfer rule; SI-77 decides only the selection score), sections 9 to 11 and section 15. A proposal becomes a rule only if the user approves it at the diff review, and each approval is recorded as a new ledger entry. |
| 9-11, 248 | DOC-01, DOC-12 | "committed and frozen with the tag prereg-tier12-v3, before any X1 adversarial-arm score is examined"; "frozen with the tag" | After the diff review, this file, scripts/analysis_x1.py (committed 2026-10-14, ff88666; three test gaps closed in 858101b) and tests/analysis/ will be frozen by the tag prereg-tier12-v3, which has not been made, before any X1 adversarial-arm score is examined. "Examined" is defined, as a proposal the user has not yet approved, in PREREG section 9, entry of 2026-10-14. |
| 40 | DOC-01 | "each analysis is fixed before its own data and before any X1 result exist" | "each analysis is fixed before any X1 adversarial-arm score is examined (19 early X1 rows were scored on 2026-10-12; whether anyone read their values is an open item, PREREG section 9, entry of 2026-10-14)" |
| 15 | DOC-11 | "*Cell* = task x decoder." | Add: "PREREG rules R2 and R3 use 'cell' for a task x decoder x arm combination; PREREG section 9 calls a task x decoder pair a task-by-decoder block; a cell of the tissue is a single cell." (Renaming every later 'cell' would touch field names in scripts/analysis_x1.py; the added sentence avoids that.) |
| 29 | DOC-13 | "R3 factor effects \| X6, X12 (A3 done)" | "R3 factor effects \| X6, X12 (A3 done except its 36 Sinkhorn rows, which are to be redone, SI-57)" |
| 29-31 | DOC-13 | "The analysis runs once, after all 3,000 X1 rows are scored or failed." | "The analysis runs once, after all 3,000 X1 rows and the 408 X13 baseline rows are scored or failed (PREREG section 9, entry of 2026-10-14)." |
| 34-37 | DOC-06, DOC-10 | grid and b* read from the frozen records | Add: "For every arm except Sinkhorn the six lambda come from docs/prereg/r1_record.json; the Sinkhorn grid comes from the redone R1 record (docs/prereg/r1_record_sinkhorn_redo.json, name proposed). b0, b_common and b* are computed after the R4 extension rounds; if an extension raises an arm's best batch score, b* can lie beyond the arm's six points and the arm is not evaluable under 2.3 (ii). The report lists the number of arm-by-task blocks not evaluable for that reason, by cause." |
| 69-74 | DOC-07 | section 2.2 Reading (no power statement) | Add: "The power figures of PREREG section 5 are for the matched-lambda contrast on all six pilot blocks, including the simulation sim1, not for this reading. They will be recomputed with this reading and only the evaluable blocks when R4-a1 is re-run after the Sinkhorn redo." |
| 101-105 | DOC-04 | "in the pilot 2 of 3 conditioned cells (immune, sim1) were low-signal" | Add: "(the common batch score is the minimum over the six arms and Sinkhorn sets it in no block of the record, so replacing the Sinkhorn arm can only lower it: the redo can flag more blocks and cannot clear these two)" |
| 118-124, 132-133, 161-170 | DOC-17 | corner, prior scale 0.1, P2 margin and prior scale 0.25 without a label | Label each "pilot-informed". Corner: chosen after A1 had produced its ranges; the 300 points behind them exclude Sinkhorn. Prior 0.1: close to the pilot between-task standard deviation, 0.109. P2: margin and scale were chosen after the user saw pilot collapse rates; the rates quoted then were reference 10 of 54 and 15 of 54, discriminator 8 of 45 and 17 of 54 (critic-minus-discriminator differences +0.7 and -3.7 points); counting all six strengths (NB-20261014-22) the discriminator figure is 8 of 54 and the differences are +3.7 and -3.7 points. |
| 155-157, 271-272 | DOC-05, DOC-16 | multiplicity: Holm over the primary results | Add: "Six results: first threshold 0.05/6 = 0.0083; with six tasks the smallest exact two-sided p is 0.03125, so the companion cannot reject at the Holm-adjusted level." State the limits of DOC-16 (from the review, not rechecked by me): P2 rates over each family's own six points are only roughly comparable; pooled Clopper-Pearson intervals treat the runs of one dataset and strength as independent; declarations use each posterior on its own; tau_f rests on four dataset families, two with two members, and its half-normal(0.1) prior may dominate; per-dataset standard errors come from five runs and are treated as known; P1 compares the reference critic at 5 critic steps with the discriminator at 1 step, and the update-budget control (X6) exists only for the pooled critic. Name a sensitivity analysis for the first and the last, or say none is planned. |
| 187 (heading of section 8) | DOC-08 | "(DECIDED Q11: option A, SI-77)" | "(Q11 option A decided the selection score, SI-77; the transfer-lambda rule is the agent's proposal)" |
| 239-249 (section 14) | AN-18 | tests and freeze | Add the run location and the exact commands: "Run in a checkout that contains the tag prereg-tier12-v3 (mixed-commit inputs under SI-54 are accepted only there). Commands: scripts/check_tests.py with its arguments followed by '-- tests/analysis tests/prereg', then one run of scripts/analysis_x1.py. The code only refuses to overwrite an existing output record, so 'runs once' is a procedure." The lead fills in the arguments from each script's --help; I did not run them. |

For the ledger (DOC-18, not mine; the user's words stay unedited, so these are agent notes beside the entries): SI-60 was answered at 20:03 New York time on 2026-10-13, which is 2026-10-14 in UTC (source date and added date differ for that reason); in SI-63 'none' is the arm called none; in SI-69 the clause '(the reading R2 and R3 already use)' belongs to the average-first reading, not to the leave-one-seed-out clause; SI-51 is narrowed by SI-57 (rule order) and SI-52 by SI-68 (P3 and P4 are locked later).

## 6. Other things I found

1. `prereg_rules.power_report` starts its search for the effect size at 1.0. With scipy 1.17.1, `scipy.stats.nct.cdf` returns NaN (not a number) when the non-centrality exceeds about 38, so the function stops with an error when the standard error is below about 0.026. The pilot value (0.045) is above that, but the value without sim1 (0.0236) is below it, and the power is to be recomputed after the Sinkhorn redo. My script starts the search at 0.01. prereg_rules.py is frozen rule code, so the change is the lead's decision.
2. NB-20261013-03 says 236 captures; the validation set holds 268 minibatches captured earlier in training or in training mode (192 + 76). The notebook is append-only, so it needs a CORRECTION entry.
3. CF-02 (a Sinkhorn row whose only output is a divergence record carries a commit that nothing reads) may be fixed by editing scripts/run_stage.py. That file is inside the import closure of the identity proof, so the identity check must be re-run after such an edit; the SI-54 entry already says this for any change.
4. Reviewer-derived numbers that I did not recompute: the no-pair expectations (25.9 of 27 fits, 18 of 18, about 14%, 4.5% and 0.6%) and the F7 ratios. I verified only their denominators (27, 18, 48, 24 and 60) against the manifest, and the entries say 'model-based estimate'.
5. The contents of D-09 and I056 are not recorded in this repository (the design recheck report and verified_issues.csv are outside it); their glosses in the patches say only what the entries themselves show.
6. The profile asks me to load the `ponytail` skill before changing code; it is not in this profile's skill catalogue, so I did not. The new script power_without_sim1.py stops by itself if it does not reproduce the committed record, and the lam_audit.py change only adds two printed counts (distinct row tags, manifest of record) and relabels the existing one, which the claim check reads back; I added no unit tests.

## 7. Checks run on the patches

| Check | Result |
|---|---|
| `git am` of the 18 patches in a fresh clone at 973a97d | applies cleanly; tree identical to the private clone |
| `python -m pytest -q tests/prereg` on the patched tree | 131 passed, 1 skipped (34.00 s) |
| `python -m pytest -q tests/stage` (reads the equivalence list) | 65 passed, 1 skipped (123.76 s) |
| Numbers in the added text against the committed records | 39 text checks and 44 readbacks pass; doc_numbers.json regenerated from the patched tree is identical to the one used for the edits |
| References in the added lines | 25 commits, 14 notebook entries, 34 ledger entries and 30 paths exist; 4 paths are absent by design (three proposed record names and the notebook file pattern) |
| Prose scan of the added lines (deslop) | 8,359 words, score 48/50; 6 hard hits, all en dashes already in the repository's label for P1 to P4 or in empty table cells |
| Rendering | markdown-it (CommonMark) and pandoc 3.12 convert PREREG.md, PAPER_PLAN.md and the README without warnings; PREREG tables 4 to 7, all rows with the header's column count |
| Scope | all 8 changed files are in the allowed set; none of the files I was told not to touch changed |

## 8. Limits and deviations

1. Files outside the four named ones: the existing lam_audit.py and its output are edited (CF-03); power_without_sim1.py and its output are new (DOC-07).
2. I opened no file under X1_local, jhpce_tier12 or review_sealed, no X1 log, and no score_identity file; of fit_identity_c3fd92d.csv I read only the flag columns.
3. Older entries of PREREG section 9 (2026-10-03 and 2026-10-04) carry no status line, because version 2 froze them; the lead-in says so.
4. Findings with a part in ANALYSIS_SPEC.md, CONSTRAINTS.md, the notebook or the tests are marked partly fixed; section 5 and section 6 give the wording or the action.
5. CF-06 is fixed in the text only: the reporting script is to be written after the runs. CF-03's test change is left to the code track.
