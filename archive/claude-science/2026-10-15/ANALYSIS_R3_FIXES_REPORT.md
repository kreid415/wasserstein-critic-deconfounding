# Analysis track, round-3 fixes (2026-10-15): report

Runner Engineer, track 'analysis'. Base: branch final-review-fixes at 5d06a27e39bd0337ac36b69f58703e26ab1e04a1 (bundle sha256 0c7de7d507cba8c2e7b60760d81f30cc07b92eb2009bf91fb55b7c21dbe0915a). Private clone, branch r3-analysis, 26 commits, last commit 0dab01126d, tree 02e816bbc7da. Only these paths changed: scripts/analysis_x1.py, tests/analysis/test_analysis_x1.py, docs/ANALYSIS_SPEC.md and experiments/analysis_numerics_20261014/ (33 files).

Words used. AN-Rnn and DOC-Rnn are findings of the independent re-check of the first round of fixes (analysis part and documents part); SI-nn is an entry of the decision ledger (a decision of the user); NB-... is a notebook entry. 's' is a dataset's leave-one-seed-out standard error. The 'floor' is the user's decision SI-83: a standard error below 1e-6 is raised to 1e-6. 'Pooled model' is the Bayesian model that combines the six real datasets (ANALYSIS_SPEC.md section 4). A 'planted error' is a one-line change of the analysis code made to see whether the tests notice it ('killed' = a test fails). A 'known answer' is a test whose expected value comes from an independent source. X1 is the pre-registered main experiment (3,000 model fits) and X13 its comparison with standard integration tools. P1 is the primary comparison of biology conservation between two methods at matched batch removal. R4 is the pre-registered rule that fixes the matched batch-removal target b* (half-way from plain scVI's batch score b0 to the batch removal every method reaches, b_common) and a low-signal flag; 'R4 extension rows' are the rows of the X1 score table for extra penalty strengths that R4 can add. 'Decoder 0' and 'decoder 1' are the two decoder settings of the experiment. S1, S2 and S3 are the three sentences the lead fixed for the specification (what 'examined' means; the Sinkhorn redo overwrites five record files; the floor). Planted-error ids come from three lists: M.. the reviewer's, N.. the lead's, O.., F.. and G.. of this round. The letters A to E before an item are the lead's item letters in the task.

## Result in brief

- Items: all addressed; DOC-R04, DOC-R05 and DOC-R13 are partly fixed (their other parts are in other tracks' files); AN-R08's premise is partly wrong.
- Tests: `python -m pytest -q tests/analysis tests/prereg` on a fresh clone after `git am`: 201 passed, 1 skipped in 115.17s (0:01:55) (baseline at 5d06a27: 193 passed, 1 skipped). The check_tests runs of acceptance check 2, including the lead's script-tier command (429 passed, 18 skipped, 243 warnings in 838.39s (0:13:58)), were made on earlier trees of the series. The tree of the lead's command (084ba0155e0d) differs from the final tree in 7 files (docs/ANALYSIS_SPEC.md, experiments/analysis_numerics_20261014/README.md, experiments/analysis_numerics_20261014/ceilings.py, experiments/analysis_numerics_20261014/compare_with_reference.py, experiments/analysis_numerics_20261014/known_answers_small_s.py, experiments/analysis_numerics_20261014/mutation/mutation_results_20261015.json, experiments/analysis_numerics_20261014/mutation/run_mut.py); none is under scripts/, src/ or tests/, and the tests read only known_answers_small_s.csv of the numerics folder, which is unchanged, so the coverage and test results carry over.
- Planted errors: 72 of 73 killed; the survivor is M28 (effect at most 2.0e-6). All 15 errors named in the acceptance list are killed.
- Patches: 26 commits apply cleanly with `git am` to a fresh clone of the bundle; the resulting tree is identical to the clone's tree (02e816bbc7da).
- One finding outside the items: the coverage command suggested in AN-R04 (`--src scripts/analysis_x1.py`) does not measure coverage (section 4).

## 1. Items

### A: SI-83, AN-R01, AN-R06 (floor on a small standard error)

- Verified: Read the code at 5d06a27: a dataset whose leave-one-seed-out standard error s was below 1e-6 was left out of the pooled result (reason zero_spread), and `pool` refused any s below 1e-6, including the averaged s of a pair (two datasets at 1.2e-6 average to 8.5e-7). The re-check's planted errors O07 to O10 on that rule survived the tests.
- Done: Rewrote the rule as the floor of SI-83: `raise_s` raises each dataset's s to 1e-6 and lists it (`raised`: unit, level, own s); the pairs-first sensitivity raises the averaged s again (level pair); NaN, infinite and negative s stop the run. Removed the exclusion from code, tests and specification (sections 2.3, 4, 15.7); the S3 sentence stands on one line.
- Action: fixed. Commits: 1897cd3, 684ab3a, 65c5153.
- Evidence: Tests: floor cases in test_combine_structures (s = 1e-6, 0.9e-6, 1e-16, 0.0 against `pool` with floored s; pair cases [1.2e-6, 1.2e-6] and [0.0, 1e-16] with their raised lists), test_guards (nan, inf, negative), the end-to-end test (raised == [lung, dataset, s 0.0], the only entry). Planted errors F07 (killed by test_pool_known_answers_small_s[H2_family_0.1]), F08 (test_pool_known_answers_small_s[H2_family_0.1]), F09 (test_combine_structures), F10 (test_combine_structures), G02 (test_guards), G04 (test_combine_structures) are killed. Full-size synthetic runs: ident_local (identical local readings) exit 0 in 31.0 s, 18 pooled results list raised datasets (6 in the decoder-0 local result, 4 in decoder 1); ident_local_atac exit 0 in 36.7 s, its pairs-first local result lists 1 pair-level entry (family atac, own s 7.07e-07); main lists none.

### B: SI-84, AN-R10 (accuracy ranges and the sign table's approximate limits)

- Verified: The statements 'within 2e-6' and 'no reported value by more than 2e-5' held on the 240 random sets and while the datasets' differences spread by at most three prior scales; beyond that the error is larger (re-check section 4.2).
- Done: Measured the code against the independent reference on the re-check's 18 uncalibrated hand-made sets and a spread scan of 1 to 8 prior scales (two prior scales). Stated the domain in specification section 15.4, in the comment at GRID_SPAN and in the numerics README; stated that the sign table's limits d +/- t s are approximate (s is a jackknife standard error of a statistic that is not smooth). The code's behaviour is unchanged.
- Action: fixed. Commits: bd5f1ad, cf327d0, 48c5aee, e401635.
- Evidence: compare_with_reference.json: largest difference over the 18 sets 2.4e-04, over the scan at 3 prior scales 1.6e-06, at 5 scales 1.1e-05 (probabilities) and 7.8e-05 (any value), at 6 scales 2.9e-05 and 3.0e-04; declarations that differ: 0 (of 32 sets). check_pool_numerics.py re-run: outputs byte-identical (240 sets: 1.99e-06).

### C: AN-R02 = DOC-R03 (specification section 1 against section 15.1)

- Verified: Section 1 said the analysis never recomputes b0, b_common, b* and the low-signal flag; the code recomputes them with `prereg_rules.r4_stage` and refuses a record that differs (section 15.1).
- Done: Rewrote the sentence in section 1: the four quantities are read from the frozen R4 record, recomputed from the X1 rows, and the run stops if they differ.
- Action: fixed. Commits: 10fd446, fb5cd5b.
- Evidence: The refusal tests (test_refusals) alter each recorded quantity; planted errors N14 (test_refusals[<lambda>-differ0]), O11 (test_refusals[<lambda>-differ4]), O12 (test_refusals[<lambda>-differ3]), N06a (test_refusals[<lambda>-differ1]), N06b (test_refusals[<lambda>-differ2]) are killed.

### D: AN-R03, SI-87, DOC-R10 (R4 extension rows, record names, tense)

- Verified: A record made with R4 extension rows is refused when the extension manifests and scores are not passed, and the message did not say why; the usage line did not name them; the specification described the grid check in the present tense and did not name the record after the Sinkhorn redo.
- Done: Usage line names the extension manifests, their score directories and one failures table covering every row; the refusal prints the recorded and the recomputed b0, b_common, b* and flag and says to pass the extension rows; specification section 1 uses the future tense for the hand check of the six non-Sinkhorn grids and names docs/prereg/r1_record.json; S2 stands on one line (line 85).
- Action: fixed. Commits: 45b53c2, 65c5153, 10fd446, fb5cd5b.
- Evidence: test_refusals has a case that drops the extension row (the regex pins the printed values); the end-to-end fixture has an extension row that moves b* by 1.67e-4. Planted errors O24 (killed; test_end_to_end errors while the fixture is built) and G03 (test_refusals (the case whose id starts '<lambda>-for atac_small')) killed.

### E: AN-R04 (the test command of section 14)

- Verified: Reproduced: `check_tests.py --base 973a97d --src scripts -- tests/analysis tests/prereg` fails with 'scripts/run_stage.py: changed lines never run: [662, 663]' (201 passed, 1 skipped in 181.74s (0:03:01)). The suggested `--src scripts/analysis_x1.py` collects no coverage (coverage.py reads a file path as a module name); on a clean checkout it stops with FileNotFoundError for .check_tests/cov.json, and with a leftover cov.json it prints OK without testing anything (reproduced with a function no test calls).
- Done: Section 14 now gives the lead's script-tier command `--base 973a97d --src scripts -- tests/analysis tests/prereg tests/scvi tests/stage tests/jhpce tests/test_legacy_retired.py` (429 passed, 18 skipped, 243 warnings in 838.39s (0:13:58), wall 14:00.72, exit 0) and explains why no file path is used. Also run: `--base 973a97d --src scripts -- tests/analysis tests/prereg tests/stage` (267 passed, 2 skipped in 387.33s (0:06:27), wall 6:28.55) and the track's own check `--base 5d06a27 --src scripts -- tests/analysis tests/prereg` (201 passed, 1 skipped in 183.03s (0:03:03), wall 3:04.24).
- Action: fixed. Commits: fb5cd5b.
- Evidence: logs/accept/check_tests_filepath_form_vacuous_ok.txt, check_tests_directory_form_fails.txt, check_tests_base_*.log in the evidence archive. The lead was told in a mid-task message. The lead's command file (gp/lead_gates_20261014/README.txt, line 4) already uses `--src scripts`, and no tracked file of the clone uses a file path with --src (git grep).

### AN-R05 (local endpoint evaluable exactly when P1 is)

- Verified: Section 15.7 said the local reading is evaluable exactly when P1 is; with the old exclusion a local reading could drop out alone (the re-check's identical-local-readings input: pooled local n = 0 while P1 was evaluable). Second part of the finding: the status paragraph named ff88666 and 858101b as the code commits although the fixes are in 8b90d79.
- Done: With the floor no dataset drops out for a small s, so the sentence is true again; reworded section 15.7; the end-to-end test asserts that the evaluable lists of P1 and of the local reading are equal. The status paragraph now names 8b90d79 (code and tests of the first review's fixes) with the other commits of 2026-10-14.
- Action: fixed. Commits: 1897cd3, 684ab3a, 2e92c17.
- Evidence: ident_local dry run (identical local readings): the pooled local result has n = 6 datasets in decoder 0, where the re-check found n = 0 with the old exclusion; test_end_to_end asserts that the evaluable lists of P1 and of the local reading are equal.

### AN-R07 (R4 recompute compared on b0, b_common, b*, flag)

- Verified: Tests altered b0 and the flag but not b_common alone nor b* by a small amount.
- Done: Added refusal cases: pancreas|1 b_common +0.01; lung|0 b* +2e-9 (just outside the 1e-9 tolerance).
- Action: fixed. Commits: 45b53c2.
- Evidence: Planted errors O12 (b_common not compared; test_refusals[<lambda>-differ3]) and O11 (tolerance 1e-3; test_refusals[<lambda>-differ4]) killed.

### AN-R08 (known answers for s of 1e-6 to 3e-6)

- Verified: The premise that `precision_mpmath.csv` holds 40-digit values is partly wrong: it holds relative errors of the posterior mean and variance at grid nodes, not probabilities. A first set of three known answers killed O02 but not O22 (151 points) or M24 (101 points): the effect of the grid size is below 1e-8 when the datasets' differences are almost equal.
- Done: Generated new 50-digit values with an independent reference (ref_mp.py, checked against a long double reference on its finest rule, largest difference 6.7e-10) for five synthetic sets, two of them with spread differences chosen so that 151 and 101 points move the result by more than the tolerance of 1e-8; asserted the registered grid constants in the same test.
- Action: fixed. Commits: 7186f69, 65c5153, a6b3be5, c72abc6.
- Evidence: Planted errors O02 (test_pool_known_answers_small_s[H2_family_0.1]), O22 (test_pool_known_answers_small_s[H2_family_0.1]), M24 (test_pool_known_answers_small_s[H2_family_0.1]) killed at the constants assertion; with that assertion removed in scratch clones the numbers alone still kill them (O02: H2_family_0.1; O22: H2_spread_independent_0.2, 4.9e-8 from the 50-digit value against 1.9e-10 for the code; M24: H2_spread_family_0.2, 2.7e-6) (logs/nopin_check.txt; known_answers_effects.txt).

### AN-R09 (failed-run total, a cell with 4 readings, a deep gap)

- Verified: Planted errors O18 (failed-run total as a maximum) and O20 (gap flag looks at the first interior point only) survived.
- Done: Added to the existing tests: the end-to-end test asserts the barycenter family's (configurations, failed runs) in pancreas|0 and lung|0, (6, 4) and (6, 2), and a cell with 4 readings (immune_hum_mou|1, m = 4); test_window has a deep-gap case.
- Action: fixed. Commits: 7b0e45e, 0ad06fd.
- Evidence: O18 killed (test_end_to_end), O20 killed (test_window), O23 killed (test_end_to_end).

### AN-R10 (see B)

- Verified: See B.
- Done: See B.
- Action: fixed. Commits: bd5f1ad, cf327d0, 48c5aee, e401635.
- Evidence: See B.

### AN-R11 (what the numerical reference shares with the code)

- Verified: The reference in check_pool_numerics.py uses the same solve with A and the same weights from V = A + (prior variance) 11' as the code, so an error in that algebra would not show; the 40-digit table covers only the posterior mean and variance.
- Done: The README now says what each reference shares; the re-check's independent reference (closed-form block algebra, Gauss-Legendre quadrature, long double and 50-digit arithmetic) and its algebra validation are committed with outputs; the code is compared with it. check_pool_numerics.py now stops when mpmath is missing (the lint rule on swallowed failures).
- Action: fixed. Commits: cf327d0, eaef4ac, 48c5aee.
- Evidence: validate_algebra.json: largest relative differences 1.6e-12 (log-likelihood, long double), 8.2e-42 (50 digits); compare_with_reference.json (see B); check_pool_numerics.py re-run gives byte-identical outputs (4 min).

### AN-R12 (few evaluable datasets cannot reach a declaration)

- Verified: Re-ran the ceilings: with all datasets equal, family structure, prior scale 0.1, margin 0.01.
- Done: ceilings.py extends the re-check's script to the equivalence side and to two datasets of one family; specification section 15.4 says the pooled result is inconclusive by construction with fewer than three evaluable datasets and gives the ceilings.
- Action: fixed. Commits: d0c20e3, e401635.
- Evidence: ceilings.json: largest P(Delta > 0.01) 0.8232 (one dataset), 0.9410 (two, different families), 0.8783 (two, one family); no declaration reachable on the grid; three datasets need d from 0.03 to 0.08.

### AN-R13 (all-failed configurations between the window's ends)

- Verified: A lambda between the window's ends whose runs all failed was invisible: `gap` stayed false while the width spanned it.
- Done: The window record has `n_all_failed_between` (None when there is no window; otherwise the count strictly between its ends); gap rule unchanged; specification section 15.10 says so.
- Action: fixed. Commits: 7b0e45e, 646ee39.
- Evidence: test_window cases (including a failed interior configuration and a deep gap); planted error G01 killed (test_window), N09 killed (test_window).

### AN-R14 (input row order)

- Verified: No test shuffled the input rows; O16 (sort by lambda) and N11 (no sort) survived.
- Done: The end-to-end test builds the record a second time from the same rows in shuffled order and requires equal primary and sensitivity results.
- Action: fixed. Commits: 0ad06fd.
- Evidence: O16 killed (test_end_to_end), N11 killed (test_end_to_end). Also: the full-size record is byte-identical across processes with PYTHONHASHSEED=1, 2 and the default (sha256 prefix b58e533be8acbf96).

### AN-R15 and SI-81 (c) (unreached cells by arm)

- Verified: The record keeps `p1.status` per cell and arm and the R4 record keeps the arms' best batch scores, but no field or script counts the unreached cells by arm.
- Done: Specification section 1 says that a reporting script written after the runs tabulates the count by arm next to the R4 maximum (user decision SI-81 (c)); only P1's reference and discriminator arms can be unreached; lists the status values.
- Action: fixed. Commits: 10fd446.
- Evidence: Specification section 1; no code change was asked for. Open question: whether the tag freezes that script (see open questions).

### DOC-R01 and DOC-R02 (what 'examined' covers; what the audit searched)

- Verified: The audit sentence was broader than the search it rests on; the definition covered X1 adversarial arms only.
- Done: The specification states S1 (every X1 and X13 score, of any arm), the user's statement NB-20261015-15, the narrowed audit sentence (NB-20261014-21: session logs searched for reads of tier12_v2/X1_local/scores, printed output, two log statements; made by the agent that ran the checks; cannot show nobody opened a file) and points to PREREG section 9 for the disclosure; the displays found by the audit are listed there as shown to the agent only (SI-88). No counts and no claim that no value was displayed, as the lead instructed.
- Action: fixed. Commits: 2e92c17.
- Evidence: Specification lines 25-30 (S1 on line 25).

### DOC-R03

- Verified: See C.
- Done: See C.
- Action: fixed. Commits: 10fd446.
- Evidence: See C.

### DOC-R04 (numbers from scripts outside the repository)

- Verified: The specification's '0.011', 'coverage 0.91 to 1.00' and the 'conservative' statement came from the first review's scripts, which were not in the repository; the pilot ranges of the area corner and the P2 recount are in the documents track's folder.
- Done: Committed the first review's pipeline simulation (6 scenarios) and P2 clustering simulation (paths changed; family effects drawn in sorted order because a set's order made two scenarios differ from process to process), and a rebuilt pilot grid-reading table (equal to the review's outcome table in all 40 columns, 918 rows); re-ran them; limitations 1 to 3 now quote the committed outputs; the pilot area-corner ranges and the P2 recount point to experiments/doc_numbers_20261015/.
- Action: partly_fixed. Commits: f48ba02, 7253c9f.
- Evidence: Limitation 1: per-dataset coverage 0.85 to 0.88, pooled 0.998 to 1.000, no 'equivalent' in 2640 pooled worlds; limitation 2: shift 0.0110 in atac_small|1, the other five cells 0; limitation 3: 0.914 to 0.9995 (Clopper-Pearson), 0.9186 and 0.9488 (Agresti-Caffo). Not done: the 'F7 ratios' of the finding are in docs/PAPER_PLAN.md and the area-corner and recount figures are produced in the documents track's folder (not mine).

### DOC-R05 (status lines name later changes)

- Verified: The status paragraph named only the first commits and no later revision of the code and specification; it still listed SI-01..SI-67.
- Done: Status paragraphs list what changed since 2026-10-14 (earlier commit ids, round-3 changes by date and ledger id) and say that the tag, not commit ids, fixes the version; sources checked read SI-01..SI-88. PREREG's status line and table rows 5 and 8 belong to another track.
- Action: partly_fixed. Commits: 2e92c17, ebd88a9.
- Evidence: Specification lines 3-23 and the sources line (line 455).

### DOC-R07 (attribution of sections 9 to 11, equal tries)

- Verified: Section 9 called equal numbers of tries the agent's proposal; SI-68 and SI-77 already decide it.
- Done: Section 9 says equal tries is the user's decision (SI-68, SI-77) and only the estimator and the family composition were proposals, approved in SI-80; sections 5, 8, 9, 10, 11 and 15 carry 'decided 2026-10-15, SI-80'.
- Action: fixed. Commits: ebd88a9.
- Evidence: Specification section 9 and the section headings.

### DOC-R08 (implementation choices under the heading of SI-73)

- Verified: Section 4 is headed as the user's decision SI-73 but the grid, its cap and floor, the exclusion and the CPU remark are the agent's choices.
- Done: Added a note under section 4: the details marked 'amended 2026-10-14' were the agent's choices, approved 2026-10-15 (SI-84; the floor is SI-83); digit for digit holds on the same input files and the same CPU type.
- Action: fixed. Commits: e401635.
- Evidence: Specification section 4.

### DOC-R09 items 1 and 6 (glossary; 'all as recommended')

- Verified: R0 to R5 against R1 to R4 were not explained in the specification's word list; 'all as recommended' is unclear.
- Done: Added the sentence on R0-R5 against R1-R4 and the AN-Rnn and DOC-Rnn labels; 'each time taking the option the agent recommended'.
- Action: fixed. Commits: 6e148d9, 2e92c17.
- Evidence: Specification glossary and Appendix A.1.

### DOC-R10

- Verified: See D.
- Done: See D.
- Action: fixed. Commits: 10fd446.
- Evidence: See D.

### DOC-R13 (a complete list of the diff-review questions)

- Verified: Appendix A.3 listed five open items, not the questions the user answered.
- Done: A.3 now lists where each of the diff review's answers (SI-78 to SI-87, and SI-88) is used in this file and points to PREREG section 9 for the full list.
- Action: partly_fixed. Commits: 0dab011.
- Evidence: Specification Appendix A.3.

### SI-78, SI-80, SI-81, SI-82, SI-87, SI-88 (labels and pointers)

- Verified: The specification still labelled decided items as proposals and had no pointer to the answers of 2026-10-15.
- Done: SI-78: S1 and the section 0 note; SI-80: decided labels; SI-81 (c): section 1; SI-82: exploratory scaling checks in section 12; SI-87: S2; SI-88: one sentence pointing to PREREG's disclosure (no counts).
- Action: fixed. Commits: 2e92c17, ebd88a9, 5e4436d, 5133d0a, 0dab011.
- Evidence: grep of the committed files: no hits for zero_spread and _sinkhorn_redo; the word proposal occurs only in sentences about what the agent proposed before the user decided.

## 2. Acceptance checks (numbered as in the task)

1. python -m pytest -q tests/analysis tests/prereg, fresh clone of the bundle after `git am` of the patches: exit 0, 201 passed, 1 skipped in 115.17s (0:01:55) (baseline at 5d06a27: 193 passed, 1 skipped in 86.7 s).

2. check_tests.py --base 5d06a27 --src scripts -- tests/analysis tests/prereg: exit 0, 201 passed, 1 skipped in 183.03s (0:03:03), wall 3:04.24, 'OK ... no redundant new tests', no WARN. check_tests.py --base 973a97d --src scripts -- tests/analysis tests/prereg tests/stage: exit 0, 267 passed, 2 skipped in 387.33s (0:06:27), wall 6:28.55, OK. The lead's script-tier command (--base 973a97d --src scripts -- tests/analysis tests/prereg tests/scvi tests/stage tests/jhpce tests/test_legacy_retired.py): exit 0, 429 passed, 18 skipped, 243 warnings in 838.39s (0:13:58), wall 14:00.72, OK. With only tests/analysis tests/prereg: exit 1, FAIL scripts/run_stage.py lines 662, 663 (201 passed, 1 skipped in 181.74s (0:03:01)). The file-path form `--src scripts/analysis_x1.py` is not usable (see deviation 1).

3. planted errors: 72 of 73 killed (python run_mut.py, 4 clones, 19 min 35 s; unmutated tests: 70 passed in 84.70s (0:01:24)); all 15 acceptance errors killed; survivor M28; O02, O22, M24 also killed without the constants assertion (logs/nopin_check.txt).

4. numerics re-runs at the final scripts: check_pool_numerics.py (4 min), ceilings.py, compare_with_reference.py (run twice), validate_algebra.py, known_answers_small_s.py (3 min), known_answers_effects.py, pilot_grid_reading.py, p2_clustering_sim.py: every committed output byte-identical (git status clean); sim_pipeline S1 and S2 repeated under other hash seeds: identical.

5. full-size synthetic dry runs (3,408 rows each): main exit 0 29.9 s; ident_local exit 0 31.0 s (18 pooled results list raised datasets); ident_local_atac exit 0 36.7 s (pair-level raised entry in the pairs-first result); one_eval exit 0 35.4 s; main under PYTHONHASHSEED 1 and 2: byte-identical record.

6. greps over scripts/analysis_x1.py, tests/analysis, docs/ANALYSIS_SPEC.md, experiments/analysis_numerics_20261014: zero_spread 0 hits, _sinkhorn_redo 0 hits, no claim that tag v2 holds the Sinkhorn record files, no proposal label on SI-80 or SI-81 items; no commit id of this series is cited inside a tracked file of the track.

7. fl_lint(fail_on='medium') on the 17 changed .py/.sh files of the series (final tree 02e816bbc7da): findings at medium or above = 0; total findings listed = 0; suppressed = 0; si_scope_check (base 5d06a27): 33 files changed, 33 inside the track's paths, 0 outside; si_check_output on this report: 0 automatable rules apply (65 rules matched by keyword are manual; those that bear on this track, SI-67 plain language and SI-80, SI-82, SI-83, SI-84, were applied by hand).

git am: 26 patches on 5d06a27 apply cleanly to a fresh clone of the bundle (sha256 0c7de7d5...0915a); the tree equals the clone's tree 02e816bbc7da; every per-commit tree equal; files changed: only scripts/analysis_x1.py, tests/analysis/test_analysis_x1.py, docs/ANALYSIS_SPEC.md, experiments/analysis_numerics_20261014/.

## 3. Planted errors named in the acceptance list

| id | result | first failing test | what is planted |
|---|---|---|---|
| O02 | killed | test_pool_known_answers_small_s[H2_family_0.1] | grid floor: TAU_FLOOR 1e-15 -> 1e-9 |
| O11 | killed | test_refusals[<lambda>-differ4] | R4 recompute tolerance 1e-9 -> 1e-3 |
| O12 | killed | test_refusals[<lambda>-differ3] | R4 recompute: b_common not compared |
| O16 | killed | test_end_to_end | sort key: rows sorted by lambda instead of tag |
| O18 | killed | test_end_to_end | failed-run total: max instead of sum |
| O20 | killed | test_window | gap flag: looks only at the first interior point |
| O22 | killed | test_pool_known_answers_small_s[H2_family_0.1] | grid point count 201 -> 151 |
| O23 | killed | test_end_to_end | sign table: t degrees of freedom a constant 4 (right only when m = 5) |
| O24 | killed | test_end_to_end | R4 recompute ignores the extension rows (computed on the six registered points only) |
| M24 | killed | test_pool_known_answers_small_s[H2_family_0.1] | GRID_N 201 -> 101 |
| N11 | killed | test_end_to_end | input rows not sorted by tag (expected equivalent at test tolerance) |
| F07 | killed | test_pool_known_answers_small_s[H2_family_0.1] | floor: '<' changed to '<=' at the threshold (s = 1e-6 listed as raised) |
| F08 | killed | test_pool_known_answers_small_s[H2_family_0.1] | floor: threshold 1.5e-6 |
| F09 | killed | test_combine_structures | floor applied only when s == 0 |
| F10 | killed | test_combine_structures | floor not applied to the pair's averaged standard error |

All 73 errors (the reviewer's 40, the lead's 16 and 17 of this round) are listed with their results in `experiments/analysis_numerics_20261014/README.md` and `experiments/analysis_numerics_20261014/mutation/mutation_results_20261015.json`. Survivor: M28 (the two end points of the spread grid get weight 1 instead of 1/2, the rectangle rule in place of the trapezoid rule in log(tau); the reviewer's list words it as 'tau grid starts at 0 but trapezoid end weights dropped (rectangle rule)', although the grid starts at TAU_FLOOR = 1e-15, not at 0). O02, O22 and M24 fail first at the assertion of the registered grid constants; with that assertion removed in scratch clones they are still killed by the numbers (O02 by H2_family_0.1, O22 by H2_spread_independent_0.2, M24 by H2_spread_family_0.2; logs/nopin_check.txt). Distances of the code and of the three grid errors from the 50-digit values (known_answers_effects.txt, tolerance 1e-8):

```
H2_family_0.1                difference from the 50-digit value: code 2.8e-10; GRID_N 151 4.3e-10; GRID_N 101 4.4e-10; TAU_FLOOR 1e-9 6.8e-05
H2_family_0.2                difference from the 50-digit value: code 1.9e-10; GRID_N 151 2.2e-09; GRID_N 101 1.3e-10; TAU_FLOOR 1e-9 6.8e-05
H2_independent_0.1           difference from the 50-digit value: code 1.1e-09; GRID_N 151 9.4e-10; GRID_N 101 7.8e-10; TAU_FLOOR 1e-9 3.5e-05
H2_spread_family_0.2         difference from the 50-digit value: code 4.3e-09; GRID_N 151 6.9e-09; GRID_N 101 2.7e-06; TAU_FLOOR 1e-9 2.7e-09
H2_spread_independent_0.2    difference from the 50-digit value: code 1.9e-10; GRID_N 151 4.9e-08; GRID_N 101 2.1e-05; TAU_FLOOR 1e-9 1.3e-10
```

## 4. Found beyond the items

1. The coverage command suggested in AN-R04 does not measure coverage. coverage.py reads the value of `--cov` as a package or directory; a file path gives 'Module ... was never imported' and 'No data was collected'. check_tests.py then stops with FileNotFoundError for .check_tests/cov.json on a clean checkout, or reads a leftover cov.json from an earlier run and prints OK. The re-check's own log (log_check_tests_narrow.txt) shows the warnings followed by OK. Reproduced here: with a function no test calls appended to analysis_x1.py and a single test selected, the file-path form prints OK (exit 0) and the directory form fails with 'changed lines never run' (logs/accept/check_tests_filepath_form_vacuous_ok.txt, check_tests_directory_form_fails.txt).
2. The first review's pipeline simulation drew the family effects by iterating a set of strings, whose order depends on the process's hash seed; scenarios with a family spread (S1, S2) gave different pooled results in different processes (coverage of the pooled interval for Delta in S1: 0.963 in the review's log, 0.981 and 0.969 in two runs of the committed script before the fix). Drawn in sorted order, repeats under other hash seeds are identical.
3. `git am` strips carriage returns, so two CSV outputs written by Python's csv module (CRLF) differed after applying the patches; the writers now use LF and the history was rewritten so that every commit has LF in those files.
4. compare_with_reference.json carried the run time and so could not be reproduced byte for byte; the field was removed (the time is printed). An automated review noted that its docstring said 18 sets while the first draft built 17: the dropped set H4b_strong_het_0.05 was restored, and the docstring, the output (18 sets) and the README all say 18.
5. check_pool_numerics.py skipped the 40-digit table when mpmath was missing (a swallowed ImportError, rule FL171 of the fail-loud lint); it now stops. fl_lint reports no finding at medium or above for every file changed in this series.

## 5. Deviations from the task

1. AN-R04: the command in section 14 is not the one the finding and the task suggest (`--src scripts/analysis_x1.py`), because that form does not measure coverage (it stops on a clean checkout and prints a vacuous OK next to a leftover coverage file). Section 14 and acceptance check 2 use `--src scripts`; the form with base 973a97d needs the other script-tier test folders as well (the lead's list; about 14 minutes).
2. The ponytail skill named in the standing rules is not in this profile's skill catalog (search found no match), so its check could not be run; I applied its stated aim by hand (smallest change; no new test functions except one parametrized known-answer test with a test-reason; check_tests reports no redundant new test).
3. The check_tests runs of acceptance check 2 were made on clones taken before two clean-ups of the series (the line endings of two CSV outputs and the run-time field of compare_with_reference.json); scripts/analysis_x1.py and tests/analysis/test_analysis_x1.py are byte-identical in those clones and in the final tree (blob ids compared), and pytest was re-run on the final tree after `git am`.
4. AN-R08: the finding's premise (that precision_mpmath.csv holds 40-digit values of probabilities) is partly wrong; new 50-digit values were generated, and two cases with spread differences were added because the first three cannot separate 151 or 101 grid points from 201 at the tolerance of 1e-8.
5. The first review's pipeline simulation was not committed unchanged: paths, progress lines on stderr, and the family effects drawn in sorted order (the set order made two scenarios differ between processes). Its S1 and S2 pooled numbers therefore differ from the review's logs; limitation 1 quotes the committed re-run.
6. DOC-R04 is partly fixed: the pilot area-corner ranges and the P2 recount are cited from experiments/doc_numbers_20261015/ (documents track; not created or checked here), and the 'F7 ratios' are in docs/PAPER_PLAN.md (not my file).
7. DOC-R05, DOC-R09 and DOC-R13 are fixed in the specification only; PREREG.md, the notebook and PAPER_PLAN are other tracks' files.
8. The notebook was not written (the task excludes notebook/); the entries I would add are listed in section 7 of the report.
9. mutation/mutation_results_20261014.json was removed (it holds the retired exclusion rule's error by name and is replaced by mutation_results_20261015.json); it stays in the history of the clone.
10. The dry-run generator and runner (with the new variant ident_local_atac) are in the evidence archive, not in the repository; their outputs (full records) are not committed; two records are in the archive.
11. History of the private clone was rewritten several times before delivery (line endings of two CSVs via filter-branch; then fixups of the comparison output, the status paragraph, the planted-error results, the README and three docstrings); commit ids quoted in messages of this track before delivery are void; the report and the patches use the final ids.
12. The lead's script-tier command (429 passed, 18 skipped, 243 warnings; 14 minutes) and the planted-error run were not repeated on the final tree: their trees (084ba0155e0d, 6f4849601175) differ from it in 7 and 5 files, none under scripts/, src/ or tests/ (specification, numerics README, three script docstrings, planted-error runner and results); code and tests are byte-identical.

## 6. Open questions for the lead

1. Acceptance check 2 of the task and the re-check's suggested fix for AN-R04 use `--src scripts/analysis_x1.py`; that form does not measure coverage (deviation 1). The lead's own command file (gp/lead_gates_20261014/README.txt line 4) already uses `--src scripts` and is unaffected (my mid-task message guessed otherwise); I found no other use of the file-path form in the tracked files of the clone.
2. AN-R15 / SI-81 (c): does the tag freeze the reporting script that is written after the runs (docs/ANALYSIS_SPEC.md section 1, the paragraph on unreached cells)? The specification says only that such a script is written after the runs.
3. docs/ANALYSIS_SPEC.md cites experiments/doc_numbers_20261015/ (and its README, 'whose README lists each number with its script') at three places (the corner paragraph in section 3, the Q5 text and the Q9 text in Appendix A.1); if the folder or the README is named differently, those three sentences need the new name.
4. notebook/2026-10-14.md:319 and notebook/2026-10-15.md:10 quote '54 of 57' planted errors and notebook/2026-10-14.md:321 links mutation_results_20261014.json, which this series removes; the current result is 72 of 73 (README of the numerics folder). A correction entry may be wanted.
5. ATAC overlap wording (specification section 1 and section 16 item 6: name-based 3,618 shared cells against all 11,270 atac_small cells found in atac_large) was flagged earlier and is not one of my items; unchanged.
6. NB-20261015-20 in the clone still says tag v2 holds the old record files; the lead's correction NB-20261015-21 is not in the clone. My specification text uses the S2 sentence only.
7. M28 (no half weights at the two ends of the spread grid) still survives the tests; its effect is at most 2.0e-6 and a known answer cannot separate it from the code's own error. Accepted; killing it would need a graded grid or a different tolerance policy (a decision, not made here).
8. check_tests is to be run again before the tag (specification section 14 says so); decide whether the tag-time run includes tests/stage (6.5 minutes).

## 7. Entries proposed for the lab notebook (not written; the notebook is outside this track's files)

1. NOTE check_tests.py: a file path as --src collects no coverage; on a clean checkout it stops (FileNotFoundError .check_tests/cov.json), next to a leftover cov.json it prints OK. The re-check's 'OK' for `--src scripts/analysis_x1.py` was of the second kind. Use a directory.
2. BUG first review's pipeline simulation: family effects were drawn by iterating a set of strings, so S1 and S2 differed from process to process; fixed by drawing in sorted order; summaries now identical across hash seeds.
3. DECISION planted errors: 73 errors (reviewer's 40, lead's 16, 17 of this round); 72 killed; M28 survives (effect <= 2.0e-6).
4. NOTE AN-R08 premise: precision_mpmath.csv holds relative errors of the mean and variance at grid nodes, not probabilities; new 50-digit known answers.
5. NOTE git am strips carriage returns: two CSV outputs written by the csv module (CRLF) changed on application; the writers now use LF.
6. NOTE outputs that must reproduce byte for byte carry no run time (compare_with_reference.json had a seconds field).
7. NOTE check_pool_numerics.py skipped the 40-digit table without mpmath (swallowed ImportError); it now stops.

## 8. Reproducing

- Apply: clone the bundle, check out final-review-fixes (5d06a27), `git am patches/*.patch` (26 patches, in order).
- Planted errors: `cd experiments/analysis_numerics_20261014/mutation; for i in 0 1 2 3; do git clone -q ../../.. c$i; done && python run_mut.py` (19 min 35 s with 4 clones).
- Numerics, known answers, ceilings, simulations: the commands are in the table of `experiments/analysis_numerics_20261014/README.md`; outputs are byte-identical on re-run (except the timings in the planted-error results).
- Dry runs: `dry/run_variant.sh <variant>` of the evidence archive (variants main, ident_local, ident_local_atac, one_eval), with DRY_ROOT set to a clone.
- Commit ids in this report are those of the private clone (branch r3-analysis); `git am` writes new ids, so the patch number identifies the commit after applying. Patch number, id in the private clone, subject:

| patch | id | subject (first 120 characters) |
|---|---|---|
| 0001 | 1897cd3 | SI-83, AN-R01, AN-R05, AN-R06: raise a standard error below 1e-6 to 1e-6 and list it, instead of leaving the dataset out |
| 0002 | 45b53c2 | AN-R03, AN-R07, SI-78: usage line names the R4 extension manifests; the refusal prints the recorded and recomputed targe |
| 0003 | 7b0e45e | AN-R13, AN-R09: window record counts configurations with no successful run between its ends; tests for the failed-run to |
| 0004 | 0ad06fd | AN-R14: the end-to-end test compares the record built from shuffled input rows |
| 0005 | 7186f69 | AN-R08: known answers for standard errors of 1e-6 to 3e-6 from an independent 50-digit reference |
| 0006 | 684ab3a | SI-83, AN-R01, AN-R05: specification: the floor replaces the exclusion of a dataset with a very small standard error (se |
| 0007 | 10fd446 | AN-R02, DOC-R03, DOC-R10, SI-87, AN-R15, SI-81: specification section 1: the analysis recomputes the R4 targets and refu |
| 0008 | 65c5153 | AN-R08: known answers add two cases with spread differences (a grid of 101 or 151 points moves them by more than the tol |
| 0009 | bd5f1ad | SI-84, AN-R10: the grid comment in analysis_x1.py states the accuracy domain (every value within 2e-6 up to 3 prior scal |
| 0010 | cf327d0 | AN-R11, AN-R10: the code against the independent reference (18 sets and a spread scan of 1 to 8 prior scales; largest di |
| 0011 | d0c20e3 | AN-R12: the largest probability a declaration can reach with one, two and three evaluable datasets (ceilings.py and its  |
| 0012 | eaef4ac | AN-R11: check_pool_numerics.py stops when mpmath is missing instead of skipping the 40-digit table (outputs unchanged wh |
| 0013 | f48ba02 | DOC-R04: the first review's simulations (pipeline coverage, P2 clustering) and the pilot grid reading rebuilt from the p |
| 0014 | c72abc6 | AN-R08, AN-R06, AN-R07, AN-R09, AN-R13, AN-R14: planted-error run over the reviewer's 40, the lead's 16 and 17 errors of |
| 0015 | a6b3be5 | AN-R08: distance of the code and of three planted grid errors from the 50-digit known answers (script and output) |
| 0016 | 48c5aee | AN-R10, AN-R11, AN-R12, AN-R08, DOC-R04: README of the numerics folder rewritten: what each reference shares with the co |
| 0017 | 2e92c17 | SI-78, SI-88, DOC-R01, DOC-R02, DOC-R05, DOC-R07, DOC-R13: specification: status paragraphs (what changed since 2026-10- |
| 0018 | 6e148d9 | DOC-R09: specification: glossary (R0 to R5 against R1 to R4; AN-Rnn and DOC-Rnn labels); Appendix A.1 says the user took |
| 0019 | ebd88a9 | SI-80: specification: the agent's proposals decided on 2026-10-15 are labelled as decided rules (sections 5, 8, 9, 10, 1 |
| 0020 | 5e4436d | SI-82: specification: the four scaling checks of PAPER_PLAN section 5 are exploratory |
| 0021 | 5133d0a | SI-75: specification section 6: the user confirmed that SI-75 stands (NB-20261015-07) |
| 0022 | e401635 | SI-84, DOC-R08, AN-R10, AN-R12: specification: SI-73 note on the agent's implementation choices, the same CPU type, the  |
| 0023 | 646ee39 | AN-R13: specification section 15.10: the window record counts configurations with no successful run between its ends |
| 0024 | fb5cd5b | AN-R03, AN-R04, AN-R07, AN-R08, AN-R13, AN-R14, SI-87: specification section 14: usage with the R4 extension manifests,  |
| 0025 | 7253c9f | DOC-R04, SI-75: specification: the pilot figures and limitations 1 to 3 point to committed scripts and outputs; the Q9 r |
| 0026 | 0dab011 | SI-78, SI-80, SI-81, SI-82, SI-83, SI-84, SI-87, SI-88: specification Appendix A.2 and A.3: the questions of the diff re |


## 9. Numbers read back from files (value and the file it was read from)

| claim | read from | value |
|---|---|---|
| planted errors run / killed | experiments/analysis_numerics_20261014/mutation/mutation_results_20261015.json | 72 of 73 |
| dry run ident_local: pooled results with a raised list | dry/out/analysis_ident_local.json | 18 |
| dry run main: pooled results with a raised list | dry/out/analysis_main.json | 0 |
| records of main, main_h1 (PYTHONHASHSEED=1) and main_h2 (PYTHONHASHSEED=2) identical | dry/out/run_*.json + record sha256 | True |
| pooled worlds in the six scenarios | experiments/analysis_numerics_20261014/first_review/sim_pipeline_*.txt | 2640 |
| equivalent declarations in them | experiments/analysis_numerics_20261014/first_review/sim_pipeline_*.txt | 0.0 |
| pilot reading shift atac_small|1 | experiments/analysis_numerics_20261014/first_review/pilot_grid_reading.csv | 0.011 |
| pilot reading shifts of the other five cells (all zero) | experiments/analysis_numerics_20261014/first_review/pilot_grid_reading.csv | True |
| largest difference code vs reference, 18 sets | experiments/analysis_numerics_20261014/compare_with_reference.json | 0.00024232688972170013 |
| declarations that differ, 18 sets + 14 scan sets | experiments/analysis_numerics_20261014/compare_with_reference.json | 0 |
| 240-set largest difference | experiments/analysis_numerics_20261014/summary.json | 1.9931646030452166e-06 |
| known-answer distances (code) | experiments/analysis_numerics_20261014/known_answers_effects.txt | H2_family_0.1                difference from the 50-digit value: code 2.8e-10; GRID_N 151 4.3e-10; GRID_N 101 4.4e-10; TAU_FLOOR 1e-9 6.8e-05; H2_family_0.2                difference from the 50-digit |
| ceilings: one dataset, largest P(above) | experiments/analysis_numerics_20261014/ceilings.json | 0.8231663434923141 |
| ceilings: two datasets of different families | experiments/analysis_numerics_20261014/ceilings.json | 0.9410006374419926 |
| check_tests base5d06 | logs/accept/check_tests_*.log | 201 passed, 1 skipped in 183.03s (0:03:03); wall 3:04.24 |
| check_tests base973_stage | logs/accept/check_tests_*.log | 267 passed, 2 skipped in 387.33s (0:06:27); wall 6:28.55 |
| check_tests base973_nostage | logs/accept/check_tests_*.log | 201 passed, 1 skipped in 181.74s (0:03:01); wall 3:02.92 |
| check_tests base973_lead | logs/accept/check_tests_*.log | 429 passed, 18 skipped, 243 warnings in 838.39s (0:13:58); wall 14:00.72 |
| pytest tests/analysis tests/prereg on the fresh clone after git am | logs/accept/pytest_final_fresh_am.log | 201 passed, 1 skipped in 115.17s (0:01:55) |
| planted-error run: time, unmutated baseline, clone tree, files that differ from the final tree | logs/mutation_final_20261015.log | 19 min 35 s; 70 passed in 84.70s (0:01:24); 6f4849601175; experiments/analysis_numerics_20261014/README.md, experiments/analysis_numerics_20261014/ceilings.py, experiments/analysis_numerics_20261014/c |
| si_scope_check: files changed / outside the track's paths | logs/accept/si_checks.json | 33; 0 |
| files that differ between the tree of the lead's check_tests run (clone ck_full) and the final tree | git diff --name-only | docs/ANALYSIS_SPEC.md; experiments/analysis_numerics_20261014/README.md; experiments/analysis_numerics_20261014/ceilings.py; experiments/analysis_numerics_20261014/compare_with_reference.py; experimen |
