# What changed on branch final-review-fixes: your review before the tag prereg-tier12-v3

**Status, 2026-10-15 (draft).** This is the "show me" step of your decision SI-52 ("Draft all, show me, then tag", 2026-10-13). Nothing has been tagged. The core experiment X1 is paused. No X1 score value has been displayed in any chat or entered into an analysis; automated checks that print only counts have read the 19 early X1 rows (audit NB-20261014-21, which searched the chat records; question 6 asks you to confirm that nobody read a value). The independent re-checks of the latest fixes finished on 2026-10-14 (section 4). Next: your answers to the 19 questions in section 6, then the fixes that depend on them, a focused re-check of those fixes, and the full diff in section 5, before you are asked for a yes to the tag.

## 0. How to read this

- **What the branch is.** On 2026-10-13 a final pre-publication review judged version 2 of the pre-registration (tag prereg-tier12-v2, commit c88cce3) not ready: the Sinkhorn arm trained with a wrong gradient, several safety checks in the fitting and scoring code were missing, and the X1 analyses were not written down. You decided to fix the Sinkhorn arm and redo its pilot fits (SI-51), and to draft every other fix on this branch with tests, prove that nothing else changes, have it reviewed independently, show you the diff, and only then make the new tag (SI-52).
- **Range.** From commit 5cb75c0 (2026-10-13 07:45 New York time, the commit that records SI-51 and SI-52) to 08f1e3c (2026-10-14 15:50): 143 commits changing 107 files. 68 commits are lab-notebook entries and 7 are decision-ledger entries; the code and test changes are in 31 files and the document changes in 6.
- **Labels.** "SI-nn" is entry nn of the decision ledger (CONSTRAINTS.md); "your decision SI-nn" means the entry records your own answer. A "proposal" is text the agent wrote; it becomes a rule only if you approve it in this review. "NB-yyyymmdd-nn" is a lab-notebook entry. "I0nn" and "D-nn" are findings of the 2026-10-13 final review and of its check of the design documents; "AN-", "CF-" and "DOC-" are findings of the 2026-10-14 rigour review (X1 analysis code, other code, documents). docs/PREREG.md section 9 starts with a glossary of every label and term used below.
- **What the tag will freeze.** docs/PREREG.md version 3 (the rules), docs/ANALYSIS_SPEC.md (the X1 analyses), scripts/analysis_x1.py and tests/analysis/ (their code and tests), docs/EQUIVALENT_COMMITS.json (section 2.3), and all code at the tagged commit.

## 1. At a glance

| # | Topic | What changed, in one sentence | Does any existing result change? | Your decisions behind it | Still needs your answer |
|---|---|---|---|---|---|
| 2.1 | Sinkhorn arm | The optimal-transport penalty of the Sinkhorn arm now trains with the correct gradient and a fully converged solve. | **Yes.** Every Sinkhorn pilot result will be redone in rule order, and the X1 Sinkhorn grid will change. Other arms: no. | SI-51, SI-57, SI-58, SI-61, SI-62 (SI-59 retired) | Names of the redone records; whether to keep two redo steps that cannot change X1 |
| 2.2 | Safety checks | Seven checks asked for by the final review, and six follow-ups from their own review, now refuse or safely handle inputs and states that could silently corrupt a fit, a score or a cluster job. | No. 59 of 59 non-Sinkhorn test fits and 7 of 7 rescored latents are identical to version 2. | SI-52, SI-63, SI-64, SI-65 | Refuse a Sinkhorn record that carries no commit? (CF-02 design) |
| 2.3 | Old and new code as one | Results made with version 2 and with the new tag can be combined, for every arm except Sinkhorn, because the code they ran gives identical results. | No (this is what allows the existing non-Sinkhorn pilot results to stand). | SI-54, SI-60 | None |
| 2.4 | Low-signal blocks in R2 and R3 | Rules R2 and R3 now count a task-by-decoder block whose target batch score cannot be placed reliably as "could not be judged". | No decision changes: both rules keep their defaults with or without it. | SI-56, SI-66 | None |
| 2.5 | X1 analyses | The X1 analyses are written down (ANALYSIS_SPEC) and coded with known-answer tests (analysis_x1.py), and were corrected after the rigour review. | New rules only; not yet run on any X1 data. | SI-68 to SI-77 | Approve the agent's proposals; does SI-75 stand; lock the scaling sensitivities? |
| 2.6 | Pre-registration text | PREREG section 9 has dated entries for every change above, a version-3 status paragraph, an amendment table and a glossary; PAPER_PLAN carries dated "superseded" notes. | Text only. | SI-53, SI-55, SI-67 | Definition of "examined"; did anyone read an X1 score value; wording items |
| 2.7 | Records | Ledger entries SI-53 to SI-77 and notebook entries NB-20261013-03 to -40 and NB-20261014-01 to -35. | Record only. | (the decisions themselves) | None |

## 2. The topics

### 2.1 Sinkhorn arm corrected (14 commits)

**Why.** The Sinkhorn arm penalises the difference between batches with the Sinkhorn divergence, an optimal-transport distance computed by repeated rescaling ("Sinkhorn iterations") at a smoothing strength epsilon that is lowered step by step. The final review found, with two independent checks, that the training gradient was twice the correct one (both final half-steps carried the cost gradient; finding I031) and that the final, least-smoothed step got only one damped update instead of being solved (I032). The 36 A3 Sinkhorn fits did not represent the stated method, so you decided to fix the arm and redo all Sinkhorn pilot work (SI-51).

**What changed** (src/wcd_vae/wcd/alignment.py). Each smoothing step is now solved until the row and column sums of the transport plan miss the cells' weights by less than 1e-4 in total; the gradient is then taken once from the converged plan (by the envelope theorem the cost does not change to first order with the plan, so the plan can be held fixed); distances are computed in double precision (float64). The solve may take at most 500,000 iterations, and reaching that cap stops the fit with an error (SI-61). A solve stops at once if its error becomes not-a-number (NaN); an infinite error does not stop it, because it occurs legitimately for very large latent values (about 1e18 or more) and the solve still converges (SI-61). Single precision (float32) was tested and not adopted (SI-58, SI-62). The other critic-free losses (MMD, maximum mean discrepancy) are unchanged.

**How it was checked.**
- Against a converged double-precision reference on 3,076 minibatches taken from the 234 Sinkhorn pilot fits: worst cosine between the new and reference gradients 0.999999721, gradient length ratio 0.999937 to 1.000067, cap never reached (3cbaca4).
- One full-length pilot fit in the hardest regime: 56,404 training steps, at most 3,786 iterations in one step, 7,778 s, 2.8 times the fit time before the fix (d852a33, NB-20261013-11).
- float32 (SI-58): met your criterion on the 598 hardest minibatches but not on all 3,076 (value error 3.2e-3 on collapsed R1-extension minibatches) and gave no GPU speed-up (time ratio 0.99 to 1.01), so float64 stays (81f8ce4, SI-62).
- Two independent reviews (NB-20261013-13, -25: both "pass with fixes"; the fixes are 772d59b to 5d0fe9a) and the 2026-10-14 rigour review (CF-08, fixed in ca6dee0).

**Effect on results.** Yes. In rule order (SI-57): the 180 A1 Sinkhorn fits are refitted; R1 sets the Sinkhorn grid for X1 and any R1-extension fits it needs; R4-a1 (the matched penalty strengths for A2 and A3) is re-run over all six arms; the A3 rows it then requires are fitted, including any non-Sinkhorn cell whose window moved; R2 and R3 are re-run. 234 = 180 + 18 + 36 was the count before the fix; the extension and A3 counts after the redo depend on what R1 and R4-a1 return. The X1 Sinkhorn grid will change. Non-Sinkhorn fits do not change (section 2.3).

**Commits.**

| Commit | Date (New York) | In plain words |
|---|---|---|
| 3cbaca4 | 10-13 09:56 | The fix: converged final step, gradient once at the converged plan, float64 distances (cap then 50,000, raised later). |
| b205459 | 10-13 10:00 | The pilot script writes its step log every 200 steps, so a stopped run keeps its log. |
| 0871294 | 10-13 11:57 | Script: how much R4-a1's choices for non-Sinkhorn cells depend on the refitted Sinkhorn arm's best batch score. |
| 5e77d18 | 10-13 11:58 | Result tables of the fix (validation, tolerance sweep, GPU timing, R4-a1 sensitivity). |
| d852a33 | 10-13 14:49 | Result of the full-length pilot fit in the hardest regime. |
| 772d59b | 10-13 18:19 | Cap 500,000; stop at a NaN error (review note F1, with a test); a switch to test float32 (float64 unchanged). |
| 01b354f | 10-13 18:21 | Pre-launch checklist for the float32 test: go (24 items: 17 pass, 7 not applicable). |
| daf1c78 | 10-13 18:29 | Tests pin the tolerance (a looser 1e-3 fails them) and that the solve runs in float64. |
| 81f8ce4 | 10-13 18:31 | float32 results; float64 kept; the float64 re-run reproduced the validation table bit for bit. |
| 6b4aefc | 10-13 18:31 | Code comment states the float32 outcome. |
| 43a0f91 | 10-13 19:15 | Reasons recorded for two exceptions to the project's fail-loud lint (a checker for code that could fail silently). |
| 89d5e2c | 10-13 19:46 | Re-review: an infinite error does occur for finite latents, so the solve keeps iterating then; new test row. |
| 5d0fe9a | 10-13 20:00 | Re-review notes: cap value pinned by a test; comments scoped to 128-cell minibatches. |
| ca6dee0 | 10-14 13:48 | Rigour review CF-08: the test pins the cap actually in force (the function default) and the infinite-error row. |

### 2.2 Safety checks in fitting, scoring and cluster jobs (15 commits)

**Why.** The final review listed checks that were missing before X1 could be launched (I033, I034, I042, I050, I051, I094, I008). An independent review of those commits (NB-20261013-13) and the 2026-10-14 rigour review added follow-ups (G-01 to G-05, G-14; CF-01, CF-03, CF-05, CF-10).

**What changed.**

| Check | In plain words | Commit |
|---|---|---|
| I033 | When a minibatch holds no pair of batches to compare, the MMD, reference-MMD and Sinkhorn losses return a zero the optimiser can work with; before, the zero had no gradient record and stopped the fit. | c4be14b |
| I034 | The check for infinite or NaN losses now sees every inner training step of a critic, not only the last one. | a4f4be0 |
| I042 | A design-file setting that an arm does not use is refused instead of silently ignored (0 of 40,815 fittable design rows refused; you accepted the per-arm lists as implemented, SI-63). | 5f30f17 |
| I050 | Before scoring, a latent must list the dataset's cells and cell types in the same order as the data. The two old pilot scripts save no cell names, so you retired them to legacy/ (SI-64). | f5ea0c3, c3fd92d |
| I051 | A kBET result (k-nearest-neighbour batch-effect test) that legitimately needed no call into R is recorded when its value is 0 and refused otherwise. | f5ea0c3 |
| I094 | A GPU fit counts as done, can resume, and can be collected from the cluster only with its saved model weights and every latent's training history. | 46d79f2 |
| I008 | Each chained X1 cluster job gets a clock that allows for the jobs before it (8, 16 and 24 days for positions 0, 1 and 2), so a later job is not killed mid-run. | 2b52542 |
| G-01 | The losses count training steps whose minibatch had no pair of batches; the count is written to the fit's record only when above zero, so other records stay byte-identical. | 96f4258 |
| G-02, G-03 | The non-finite test covers three (step, term) cases; the fitter is tested end to end refusing a setting its arm does not read. | 96f4258 |
| G-04, G-05 | Arms none, scvi_adv and scANVI refuse a non-zero penalty strength (they never read it); the tests pin the full lists of unread and extra settings. | 4f1a570, 39aa51e |
| G-14 | Collecting results refuses a latent without its training history, as the stage rule does. | 4f1a570 |
| CF-01, CF-03 | Tests run the no-pair counter through real training for every loss and pin which arms each extra setting applies to. | b82bf49, 38f5149 |
| CF-05, CF-10 | Two comments corrected (which references a git bundle carries); one unused import removed from a test. | 0110b63, 2405cd1 |
| Evidence | README, penalty-strength audit, and a planted-bug check: 11 of 11 planted bugs made their test fail. | ca18e0d |

**How it was checked.** Each fix came with a test that fails on the code before it. The coverage rule (your FEWEST-TESTS rule) was checked with scripts/check_tests.py. The 2026-10-14 tests on tree 46a2df12 (equal to fd9c8e3; later commits change only documents): scripts tier 421 passed and 18 skipped; source tier 172 passed; both "OK".

**Effect on results.** None. The checks refuse; they do not change what a fit or a score computes. Section 2.3 gives the proof.

### 2.3 Version 2 and the new tag treated as one, and the proof (5 commits)

**Why.** The non-Sinkhorn pilot results (A1, A2, A3) and 19 early X1 rows were made with version 2 (commit c88cce3). Without a rule, the rule scripts would refuse to combine them with anything made at the new tag. You decided to declare the two commits result-identical for every arm except Sinkhorn, in a committed list backed by identity checks (SI-54), with the tag named prereg-tier12-v3 (SI-60).

**What changed.** docs/EQUIVALENT_COMMITS.json says that c88cce3 counts as prereg-tier12-v3 for every arm but Sinkhorn. The rule code (scripts/prereg_rules.py) and the runner (scripts/run_stage.py) use it; Sinkhorn rows (scores, latents and, since CF-02, the runner's Sinkhorn divergence records) must carry the new tag. If the tag does not exist, nothing is combined and a mix is refused.

**The proof.** Rerun of one design row per code path (63 rows) for one epoch on CPU with version 2 and with the branch: 59 of 59 non-Sinkhorn fits gave exactly equal latents and 4 of 4 Sinkhorn fits differed, as they should. Seven stored latents rescored with the old and the new scorer agreed in all 32 compared columns. First at commit 2b52542 (da1ded4), then again at c3fd92d after the safety-check follow-ups (ddb113a). After c3fd92d, the only change to any file the fitting code imports is in scripts/run_stage.py (checks and provenance, CF-02); the independent code re-check is confirming that this cannot change a fit.

**Commits.** da1ded4 (first identity check), 8ce5a54 (the list and the rule code), 7c32579 (test: the same latents are refused without the list), ddb113a (proof at c3fd92d), 8351999 (CF-02: the runner also reads the commit stored in a Sinkhorn divergence record).

### 2.4 Low-signal blocks in R2 and R3 (1 commit)

**Why.** You decided that a task-by-decoder block whose batch-score range is too small to place the target batch score b\* reliably (flag bstar_low_signal: the highest batch score every arm reaches minus plain scVI's is below 0.02) is not evaluable (SI-56), and that R2 and R3 count it as "could not be judged" (SI-66). The rule was recorded on 2026-10-13 but implemented only on 2026-10-14 (f3c3415); PREREG discloses this.

**Effect.** No decision changes. On the frozen record, R2 had 12 of 12 cells evaluable and R3 17 of 20; with the rule, R2 has 9 of 12 and R3 13 of 20, and both keep their defaults (posterior mean as adversary input; no standardisation). R2 and R3 will be re-run after the Sinkhorn redo (SI-57). The immune block with the conditioned decoder stays flagged, so R2 and R3 keep their defaults whatever the redo shows; whether to keep that re-run is one of your questions.

### 2.5 The X1 analyses: specification and code (10 commits)

**Why.** The final review found that the analysis code for the X1 results was not written (I001) and must be frozen before any X1 score is read. You chose what to lock and how (SI-68 to SI-77, two question rounds on 2026-10-14).

**What changed.**
- docs/ANALYSIS_SPEC.md states the rules for P1 (biology score kept at equal batch removal, one result per decoder), the trade-off area, how the six real datasets are combined (a model with a dataset-family term), the margin 0.01, P2 (how often runs collapse), the local-versus-global question, the single-strength tables, the comparison with standard tools at equal numbers of tries, the width of the range of good strengths, the sensitivity analyses and the stated limitations. After the documents review it holds only rules; the options you did not choose, the reasons given with each question and the open items are in its Appendix A (4af5069).
- scripts/analysis_x1.py implements it. It was written by Codex (an external coding model) to a written brief with hidden known-answer tests, read line by line and edited in three places by the agent (ff88666); tests/analysis/ holds its known-answer tests (hand-derived values, brute-force integration, an end-to-end run on the committed X1 design with synthetic scores and planted effects). Planted-change testing found three test gaps, closed in 858101b.
- After the rigour review (8b90d79, 6aaa3ac, 56febe4): a finer, log-spaced integration grid for the model that combines datasets (AN-01), a more precise formula for the overall difference when a dataset's standard error is very small (found while checking AN-01), a dataset with a standard error below 1e-6 marked "not evaluable" instead of stopping the whole analysis (AN-02), the matched targets recomputed from the same rows and checked (AN-03), t-based limits in the sign table (AN-06), and smaller items (AN-10, AN-11, AN-13, AN-14). Numerical checks: on 240 random synthetic cases the largest difference from an independent reference was 2.0e-6 and no declaration differed; 54 of 57 planted errors were caught.

**Effect.** New rules only. The code has run only on synthetic data.

**Which parts are proposals.** Your decisions SI-68 to SI-77 fix the main choices. The agent's proposals, marked as such in the file and listed in section 6: section 2.3 (the not-evaluable rule), section 5 (Holm's correction over the six primary results, first threshold 0.05/6 = 0.0083), section 8 (the transfer rule), sections 9, 10 and the unnamed parts of 11, section 15 (implementation details) and the designs of the rigour-review fixes AN-01, AN-02, AN-03, AN-05 and AN-06.

**Commits.** 6d0ca1f (draft with open questions Q1 to Q11), 56c0762 (round 1: SI-68 to SI-72), 990d37d (round 2: SI-73 to SI-77), a1d497b (prior-sensitivity wording as asked; implementation details listed), ff88666 (code and tests), 858101b (three test gaps), 8b90d79 (rigour-review fixes), 6aaa3ac (numerical checks and planted errors), 56febe4 (limitations, "pilot-informed" labels, run commands), 4af5069 (rules only; decision log in Appendix A).

### 2.6 Pre-registration and paper-plan text (23 commits)

**What changed.** PREREG section 9 has a dated entry for each topic above; each starts with a status line saying what is done (date, commit) and what "will be" done. A version-3 status paragraph at the top of PREREG now says that the sentence "nothing in this file is chosen after seeing A1, A2, A3 or X1 data" holds for version 2 only, because the version-3 amendments were decided after A1, A2 and A3 had been fitted and scored and after the local X1 run had started; an amendment table in section 9 lists, for each change, when you decided it, when it was committed, what data existed then and whether a result changed (DOC-03). PREREG discloses that 19 X1 rows were scored on 2026-10-12 before the analyses were frozen and that one X1 latent was rescored by the identity check, and proposes a definition of "examined" (DOC-01). PAPER_PLAN has dated "superseded" notes where PREREG or ANALYSIS_SPEC now govern (DOC-09).

| Commit | In plain words |
|---|---|
| 018c8f1 | R1 does depend on the trajectory score through its noise estimate (I091, SI-53, disclosed); PAPER_PLAN corrected to PREREG on the grid edge (SI-55), the P1 comparator (D-01) and power (D-02). |
| b2c68c8 | Entry for SI-54 and SI-60. |
| 4d86a55 | Entries for the Sinkhorn fix and redo, the safety checks, P1 at lambda = 0 and low-signal blocks, and the frozen X1 analyses; PAPER_PLAN notes F6, F7, G-07. |
| de741a0 | The tag sentences state a commitment, not a fact. |
| 38b65b0 | DOC-05: one Holm family size everywhere (six results). |
| f9ffec3 | DOC-07: the power figures are for the matched-lambda contrast on six pilot blocks with sim1; the set without sim1 added. |
| 631cdfd | DOC-06: "the frozen X1 grids stand" holds for every arm except Sinkhorn. |
| 5227d0b | DOC-04: both dates of SI-66, both flagged blocks; the R2/R3 record predates the rule. |
| d50348b | CF-07: pilot R2/R3 counts come from the frozen record; the current code reports the new counts. |
| 3cfc3a7 | DOC-02: the Sinkhorn redo stated in rule order, not as a fixed list of 234 refits. |
| bc9d592 | DOC-19: where each number comes from. |
| 8703cee | CF-06: the no-pair counts will be tabulated by a reporting script written after the runs. |
| 9b30d58 | DOC-15: the evidence for "result-identical" names the c3fd92d files; a pre-tag check is added. |
| f4b3b8f | CF-04: the two limits of the SI-54 design are stated. |
| 6746f9f | DOC-14: what happens at an NMI-0 case (SI-65) and at an iteration-cap hit (SI-61) (proposals). |
| 43fdc04 | DOC-01: disclosure of the early X1 scores; "examined" defined (proposal). |
| 43f7386 | DOC-08: only SI-68 to SI-77 are called your decisions; proposals labelled. |
| 3ced541 | DOC-09: dated "superseded" notes in PAPER_PLAN. |
| 8e01f15 | CF-03: what the counts 322, 382 and 1,908 rows mean. |
| 81e8017 | DOC-03: version-3 status paragraph and amendment table. |
| a828f41 | DOC-12: status lines; "will be" for steps not yet done. |
| fd9c8e3 | DOC-11: glossary; every label carries its meaning. |
| 06345ce | SI-54 sentence names divergence records; SI-68's scope "an X1 result" restored; "matched lambda" glossed. |

### 2.7 Records (75 commits)

- Ledger (CONSTRAINTS.md): SI-53 to SI-67 in 40c0468, 2288119, e88a336, 51ea7a7, 4aaaeb9, 9df08c6 and a69ce13; SI-68 to SI-77 in 56c0762 and 990d37d. SI-59 was retired and replaced by SI-61.
- Lab notebook: 68 commits holding NB-20261013-03 to -40 and NB-20261014-01 to -35 (notebook/2026-10-13.md, notebook/2026-10-14.md; index notebook/INDEX.md). Mistakes are corrected by CORRECTION entries, never by editing (for example NB-20261014-35 corrects a count in NB-20261013-03 from 236 to 268).

## 3. Independent checks of this branch

| Date | What was reviewed | Verdict | Where the fixes are |
|---|---|---|---|
| 2026-10-13 | Sinkhorn fix (two reviews) | pass with fixes, twice | 772d59b to 5d0fe9a |
| 2026-10-13 | Safety-check commits | pass with fixes | G-01 to G-05, G-14 (96f4258 to ca18e0d) |
| 2026-10-14 | Everything after 6b4aefc, three tracks at 1e615eb (X1 analysis, other code, documents): 47 findings, 20 to fix, 27 notes, no blocker | pass with fixes on all three tracks | section 2.5, 2.2 and 2.6 commits after 1e615eb |
| 2026-10-14 | Re-check of those fixes at 08f1e3c, three tracks | pass with fixes on all three tracks, no blocker | section 4 |

## 4. Results of the re-checks and changes made after them

Three independent re-checks (Rigor Reviewer profile, each on its own copy of the branch at 08f1e3c, synthetic and pilot data only, no X1 or X13 score opened) finished on 2026-10-14 at about 17:36 New York time. They are recorded in notebook entry NB-20261015-01 (commit beac8d3, so far the only commit after 08f1e3c).

| Track | Findings of the first review | New items | What the re-checker ran |
|---|---|---|---|
| X1 analysis code and specification | 16 of 18 closed, 2 partly (AN-04: planted errors that still pass the tests; AN-18: run commands) | 3 to fix before the tag, 12 notes | tests, the numerical checks and the planted-error run again; its own independent reference for the dataset-combining model; full-size synthetic dry runs; 24 planted errors of its own (11 caught) |
| Other code | 9 of 10 closed, 1 partly (CF-10: the lint tool's version is not recorded in the repository) | 1 to fix, 4 notes | both test tiers at 08f1e3c (421 passed and 18 skipped; 172 passed; both coverage checks OK); 50 runs of the stage runner for the commit-mixing rule; 33 planted bugs |
| Documents | 16 of 19 closed, 3 partly (DOC-01, DOC-11, DOC-18); the 4 document parts of CF items closed | 5 to fix, 9 notes | 73 numbers recomputed from committed files, 72 agree (1 imprecise wording); the power-function failure reproduced and a fix tested |

The items to fix before the tag, in plain words:

- The specification says the analysis never recomputes the matched targets, while the code (correctly) recomputes them and stops on a mismatch (AN-R02, DOC-R03).
- The run commands in the specification are incomplete: the R4 extension files must also be passed, and the test command prints a failure for a line that belongs to the other code track (AN-R03, AN-R04).
- The lint tool's version, the ruff result and the reason for two separate coverage runs are recorded only in a sub-agent's report, not in the repository (CF-R01).
- The audit sentence in PREREG is broader than the search it rests on (DOC-R01), and the disclosure leaves out the 250 earlier X1 and X13 baseline fits (150 of them scored by 2026-10-07, values not inspected) and the X13 row rescored by the identity check (DOC-R02).
- Several numbers in the documents come from scripts that are only in review folders, not in the repository (DOC-R04).
- Status lines and the amendment table do not name the later analysis commits 8b90d79, 6aaa3ac, 56febe4 and 4af5069 (DOC-R05).

These fixes, and the notes adopted, will be made after your answers to section 6, because several depend on them. The commits they add will be listed here before the tag.

## 5. The full diff

(Links to be added: code and tests; documents; experiment scripts and READMEs; notebook pages. Result tables under experiments/*/results/ are listed by name, not printed.)

To regenerate from a copy of the repository: `git diff 5cb75c0 <final commit> -- scripts src tests cluster legacy pyproject.toml` (code), `git diff 5cb75c0 <final commit> -- docs CONSTRAINTS.md` (documents), `git log --reverse --format='%h %ad %s' --date=iso 5cb75c0..<final commit>` (commit list).

## 6. Questions for you

Asked in the app on 2026-10-15, one question per tab; short form here. Each tab gives the evidence, where every number comes from, and what your answer changes. X13 is the comparison with standard tools (Harmony, Scanorama, sysVI, scANVI).

| # | Question | Options (recommended first) |
|---|---|---|
| 1 | A dataset whose run-to-run uncertainty comes out as zero: leave it out of that one result, or raise its uncertainty to a minimum? | Leave it out, as drafted / Minimum tied to the other datasets |
| 2 | Five design choices in the X1 analysis code (grid for combining datasets, a more precise formula, recomputed targets, t-based limits, per-run sensitivity): approve? | Approve all, and state the grid's accuracy range / Approve, but change the grid rule first / Discuss individually |
| 3 | Refuse a Sinkhorn divergence record that carries no code version, or that comes from code with uncommitted changes? | Refuse, as for results / Leave as is |
| 4 | Fix the power function's failure for some small standard errors before the tag? | Fix now / Leave it |
| 5 | What counts as "examined": every X1 and X13 score, or X1 adversarial-arm scores only? | Every X1 and X13 score / Adversarial-arm scores only |
| 6 | Has anyone read an X1 or X13 score value? (factual; no recommendation) | No / Possibly |
| 7 | Record fingerprints (hashes) of the existing score files and the tag's GitHub date? | Record / Not needed |
| 8 | Approve the agent's proposals in the analysis specification? | Approve all / Approve with exceptions |
| 9 | Approve three process rules (cases where scib, the scoring package, finds zero mutual information; iteration-limit hits; counts tabulated after the runs)? | Approve all / Change |
| 10 | Does your decision SI-75 on P2 (how often runs collapse) stand after the recount? | It stands / Revisit |
| 11 | Keep the two Sinkhorn redo steps that cannot change X1? | Keep, in rule order / Skip and disclose |
| 12 | Approve the names of the three redone Sinkhorn records? | Approve / Other names |
| 13 | PAPER_PLAN's four scaling checks: report as exploratory, lock now, or drop? | Exploratory / Lock now / Drop |
| 14 | Where the documents cite your decision SI-68 (which analyses are locked when), keep its own words "an X1 result"? | Confirm / Change |
| 15 | Starting commit for the record that every changed line is run by a test? | 973a97d, plus a merged record for the whole range / 973a97d only |
| 16 | How to record the fail-loud lint tool? | Notebook note / Commit a copy |
| 17 | The 23 small tidiness warnings outside the identity-proof code? | Leave them, count recorded / Remove in a separate commit |
| 18 | Delete the 11 old review copies of the code and two stray files, after checking them? | Delete after the check / Keep |
| 19 | Remaining wording changes from earlier rounds? | Approve as shown / Change |

## 7. What happens after your sign-off

1. Ledger entries for each proposal you approve, then the tag prereg-tier12-v3, pushed to GitHub.
2. Two small fixes to the X1 cluster launch, from the guard review: the command builder accepts only commit c88cce3 today and must be pinned to the new tag, with its plan text and test (G-17); and the submission plan needs a fallback if the app refuses the 384-hour and 576-hour clocks that chained jobs 3 to 5 ask for (submit them one at a time after the previous job ends, 192 hours each; G-15).
3. The pre-launch checklist with its single-cell pack, then the Sinkhorn redo as a local job in rule order (SI-57).
4. Only with your go for each: the check that the cluster's L40S GPUs reproduce local fits, re-run at the new tag (CR2-01, because it last ran on older code), then X1 on the JHPCE cluster.
