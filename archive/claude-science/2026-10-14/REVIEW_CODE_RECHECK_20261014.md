# Re-check of the code fixes CF-01 to CF-10 (track 2 of 3)

| | |
|---|---|
| Reviewer, date | RIGOR_REVIEWER (independent re-check; I did not change any code, test or document), 2026-10-14 |
| Code under review | branch `final-review-fixes` at `08f1e3c39ea449677cea7e34f6201925d18f4b84` (git tree `bfc3238d89`). The git bundle `wcd_final-review-fixes_08f1e3c.bundle` has SHA-256 `d399e914787eedcaaf3e6e411b7918a57c1fc9217b7cad576b335d78b09205e7`, which matches the value in the task. All work ran in private clones of the bundle; the main repository was not touched. |
| What was fixed | six code commits `b82bf49` (CF-01), `8351999` (CF-02), `38f5149` (CF-03), `0110b63` (CF-05), `ca6dee0` (CF-08), `2405cd1` (CF-10); the CF-09 change inside `8b90d79`; and the statements about code in the document commits `8e01f15`, `f4b3b8f`, `8703cee`, `d50348b`, `06345ce` |
| Inputs I used | fix report [CF_FIXES_REPORT.md]({{artifact:fa1b46f9-cff0-48d8-a6d5-df83ce0e1e14}}), first review [REVIEW_CODE_RIGOR_20261014.md]({{artifact:a7e42e48-8a71-4d81-aa7b-618caab4e7ca}}) with its probes ({{artifact:8d340d79-13d7-4c47-9f88-6db2ab9e9fcc}}), the lead's gate pack ({{artifact:856fc931-0942-4a80-810f-a589ac12f6b5}}), CONSTRAINTS.md ({{artifact:6da7d2d3-47fa-4625-8a3a-f10c7b9af6e9}}) |
| **Verdict** | **PASS_WITH_FIXES.** No blocker (nothing gives a wrong result or breaks a decision). One item to complete before the diff review (CF-R01: the fail-loud lint version and the open ruff and test-base questions are recorded only in the fix report, not in the repository) and four notes. |

## 1. Result at a glance

"Closed" means the fix does what the finding needed and I verified it by running code. The first review's ten code findings (CF-nn, "code finding nn") are re-checked against the lead's triage (the planned action for each).

| id | topic | status | what I checked |
|---|---|---|---|
| CF-01 | no-pair counter test through the training loop | **closed** | planted bugs killed: first review's 3 of 3, lead's 11 of 11, mine 8 of 8 |
| CF-02 | SI-54 ignored the commit in a divergence record | **closed (question CF-R03)** | stage scenarios that differ from the expected outcome: 0 of 25; my planted bugs 4 of 5 killed (one contrived survivor) |
| CF-03 | table test did not pin the arms of each extra setting | **closed** | in-place table changes caught: deletions 1 of 28 before the fix, 28 after; additions 8 of 77 before, 77 after |
| CF-04 | two SI-54 limits not stated in the documents | **closed (wording note CF-R04)** | both limits in `PREREG.md` and `EQUIVALENT_COMMITS.json` match scenario rows P2-06 and P2-10 |
| CF-05 | two comments on what a git bundle carries | **closed (G-17 and the raw SHA comparison deferred, both written down)** | toy bundles with git 2.43.0 confirm the comments; both scripts are comment-only changes |
| CF-06 | "reported with the results" but nothing reads the counts | **closed** | outside tests and docs the key is mentioned only by the record writer (`scripts/fit_paper_config.py:506`) |
| CF-07 | pilot R2/R3 counts quoted from the frozen record | **closed** | `decide_a2_a3.py` re-run: R2 9 of 12 and R3 13 of 20 evaluable now, against 12 of 12 and 17 of 20 in the frozen record; choices unchanged |
| CF-08 | Sinkhorn cap test read the constant, not the default in force | **closed (follow-on CF-R02)** | my planted bugs 4 of 6 killed; the 2 survivors are the call sites (CF-R02) |
| CF-09 | `np.trapezoid` needs NumPy 2 | **closed** | `test_pool_brute_force` passes under NumPy 1.26.4 and 2.4.6 |
| CF-10 | tooling: ruff, fl_lint, check_tests warnings | **partly** | the `shutil` import is gone, but ruff (F, E9) still reports 28 findings; `fl_lint` version is not in the repository and I cannot run it |

New findings of this re-check (all detailed in section 9): CF-R01 (fix), CF-R02 to CF-R05 (notes).

What I ran, in one paragraph: the lead's two test tiers at `08f1e3c` with the lead's exact commands (scripts tier 421 passed and 18 skipped, src tier 172 passed, both "check_tests: OK"); the first review's three planted bugs and the lead's eleven, plus 19 planted bugs and one equivalent control of my own, and 105 in-place changes of the extra-settings table; 25 stage scenarios (50 runs) of the real stage runner for the commit-mixing rule SI-54; the import-closure probe; a recomputation of the document numbers; a cost probe for the recommendation in CF-R03; and three extra test runs that the lead did not do (tests that need the prepared data folder: 113 passed; the end-to-end stage test with the real fitter and scorer: 1 passed; the suite's "other" group under NumPy 1.26.4: 396 passed).

## 2. Words used

| term | meaning |
|---|---|
| commit, tag, SHA | A commit is a saved version of the code, named by a 40-character fingerprint (SHA, secure hash algorithm), usually shortened to 7 characters. A tag is a readable name for one commit. `c88cce3` is the commit tagged `prereg-tier12-v2` (the code frozen so far); `prereg-tier12-v3` is the new tag, not yet made. |
| SI-nn | Standing instruction nn: a decision of the user recorded in `CONSTRAINTS.md`. SI-52: draft all fixes on a branch with tests, prove that non-Sinkhorn fits and scores are bit-identical to `c88cce3`, have each fix independently reviewed, and show the user the diff before any new tag. SI-54: `c88cce3` and the new tag are declared result-identical for every arm except Sinkhorn in a committed list, so rows of both commits may be mixed for those arms; Sinkhorn rows must carry the new tag; nothing is rescored. SI-56 and SI-66: a cell flagged `bstar_low_signal` (too little signal at the comparison point b*) is not evaluable and counts as "could not be judged" in the rules R2 and R3; while a task-by-decoder block is flagged, those rules keep their defaults (posterior mean; standardisation off). SI-60: the new tag is called `prereg-tier12-v3`. SI-61: the Sinkhorn iteration cap is 500,000 and reaching it stops the stage with an error; a NaN (not a number) error ends the solve at once, an infinite error does not. SI-63: each arm refuses extra settings it does not read; the per-arm lists are accepted as implemented. SI-64: latents without cell names stay refused; two old fitting scripts are moved to `legacy/` as retired. |
| arm | One method variant of the benchmark (for example plain scVI, a discriminator, a Wasserstein critic, MMD, Sinkhorn). MMD (maximum mean discrepancy) and Sinkhorn (an entropy-smoothed optimal-transport distance) are two penalties that compare the latent distributions of batches without a trained critic network; `mmd_ref` is MMD against one reference batch. |
| minibatch, no-pair step | A minibatch is the group of cells used for one training step. A no-pair step is one whose minibatch gives the penalty nothing to compare (only one batch present; for `mmd_ref`, no reference cell or no other batch). The penalty is then 0 and the step is counted. |
| fit, latent, row, stage | A fit is one training run, described by one manifest row (arm, task, settings, seed). Its latent is the saved low-dimensional embedding of every cell. A stage is the list of rows that the runner `scripts/run_stage.py` works through. |
| divergence record | The small file `status/<tag>.json` that the fitter writes, instead of a latent, when the training loss becomes non-finite. The row then counts as the final outcome "diverged", which the analysis counts as a failed fit. |
| start-up check, final check, ledger | The runner's start-up check (called "preflight" in the code) runs before any work and reads every existing output. Its final check (the "completion gate") runs at the end. The ledger is the per-row table and summary that the runner writes. |
| planted bug, killed, survived | A deliberate one-line break of the code, made to test the tests. Killed: at least one test fails. Survived: every test still passes. An "equivalent" planted bug changes nothing and must survive (control). |
| tier, `check_tests.py`, base | A tier is a group of test folders run together in one conda environment. `check_tests.py` is the repository's test-economy check: it fails if a changed line or branch is run by no test and warns about a new test that covers nothing the others miss. Its base is the commit from which changes are counted. |
| identity proof, import closure | The identity proof (at `c3fd92d`) refitted 63 rows (59 plus 4) and rescored 7 latents with the old and the new code and found them equal for every arm except Sinkhorn. The import closure of a program is every code file it loads; if none changed, the program cannot behave differently. |
| R2, R3, evaluable | Pre-registered rules that decide two analysis settings from the pilot (R2: posterior mean or sample as adversary input; R3: standardisation off or on). A cell is evaluable if its two readings could be compared. |
| X1, A1 to A3, G-nn, NB-nn, CF-nn | Project labels. X1 is the main benchmark run (3,000 model fits, as the documents state; its scores are deliberately not examined yet, so this review never opened them). A1 to A3 are the pilot experiments. G-nn are findings of the earlier review of the guard commits (G-17: list the tag in the X1 bundle command, a step for after the tag). NB-yyyymmdd-nn is an entry of the lab notebook. CF-nn are the first review's code findings and CF-Rnn the new ones of this re-check. |

## 3. The ten findings one by one

### 3.1 CF-01: the no-pair counter was tested through a training run for one call path only: **closed**

*Needed.* A test through the real training loop for every way the counter is reached (adversaries `mmd`, `mmd_ref`, `sinkhorn`; with and without importance weights) and for a count of exactly one.

*What I ran.* (a) The first review's planted bugs R01a, R01b, R01c (its script `probes/my_mutate.py`, unchanged) in a clone at `08f1e3c`. (b) The lead's `experiments/guard_followups_20261013/mutate.sh` (11 bugs). (c) 8 bugs of my own: the counter stops at 1, counts every step twice, `mmd_ref` / `sinkhorn` / `mmd` stop counting, the weighted mask keeps a batch that has nothing to compare to, the plain call drops the counter, the record shows at most one step.

*Result.* R01a/b/c: 3 of 3 killed, every time by `test_resolved_plan_known_answers`. Lead's list: 11 of 11 killed, each by its named test. My bugs: 8 of 8 killed by the same test. The critic-free adversaries in `scripts/scvi_adversarial_plan.py:57` are exactly `mmd`, `sinkhorn`, `mmd_ref`, and weights exist only for `mmd` (`_IW_ARMS`, line 60), so the four calls now in the test (mmd, mmd_ref, sinkhorn, weighted mmd) cover every reachable path. The "exactly one step" case sets the counter to 1 and reads the record, which tests the threshold `if n:` (`scripts/fit_paper_config.py:505`); it does not run a training step that produces exactly one no-pair step, which is enough for the threshold.

*Left over.* Nothing.

### 3.2 CF-02: SI-54 ignored the commit stored in a divergence record: **closed** (one open question for the user, CF-R03)

*Needed.* A Sinkhorn divergence record made at `c88cce3` refused at the start-up check and at the final check; say what happens to a record with no `git_sha`; the real fitter writes the commit into the record.

*What I ran.* (a) 25 scenarios (50 runs: each with and without `--report-only`) of the real `run_stage.py` on a stage of five rows (two non-Sinkhorn scored, one Sinkhorn scored, one non-Sinkhorn diverged, one Sinkhorn diverged), with the commit that each output names rewritten per scenario, before the tag exists, with the tag, and at a commit after the tag; the expected outcome comes from an independent restatement of SI-54 in the script, not from the code under test (table in section 5). (b) 5 planted bugs of my own in the new lines of `scripts/run_stage.py` plus one equivalent control. (c) The real fitter, with a forced divergence, to see what a real record contains. (d) The opt-in end-to-end stage test (real fitter, real scorer, R, CPU).

*Result.* (a) Scenarios that differ from the expected outcome: 0 of 25. A Sinkhorn record at `c88cce3` is refused at start-up in both modes, with or without the tag (rows P1-04, P2-02); a non-Sinkhorn record at a third commit is refused at the final check (P2-04); the target mix of the X1 stage (non-Sinkhorn rows at `c88cce3`, Sinkhorn rows at the tag) is accepted (P2-01). (b) 4 of 5 killed: start-up check ignores records (`test_sinkhorn_output_at_the_pre_fix_commit_is_refused[status]`), reads them under another arm's name, final check ignores records (`test_si54_old_commit_counts_as_the_tag_in_the_gate`), refusal skipped in report-only runs. The survivor `M02f` (final check drops only the commit of Sinkhorn records, while the start-up check still refuses `c88cce3`) is contrived: `4 passed, 62 deselected in 11.95s`; the final-check test uses a non-Sinkhorn record. The equivalent control survived, as it must. (c) The real record has the keys `git_sha` and `git_dirty` (its `git_sha` was the clone's commit, `08f1e3c`); `record_divergence` and `provenance()` are byte-identical at `c88cce3` and at `08f1e3c`, so a Sinkhorn record made at `c88cce3` does carry `git_sha`. `run_stage.record_nonfinite` copies the latent's commit (`scripts/run_stage.py:962-967`) and the CPU runner copies its own (`scripts/run_cpu_baselines.py:162-166`). (d) 1 passed in 56 s.

*A record with no `git_sha`* is skipped by both checks, as a latent without one is (rows P2-03, P2-13, P2-15 are accepted). The real fitter cannot write a record without it (`provenance()` runs `git rev-parse HEAD` with `check=True`, `scripts/fit_paper_config.py:267-268`), so only a hand-made or edited file lacks the key. The lead already recorded the question as open (NB-20261014-30). See CF-R03 for my recommendation.

### 3.3 CF-03: the table test did not pin which arms each extra setting applies to: **closed**

*What I ran.* Every deletion and every addition of one arm in every value set of `EXTRA_ARMS` (28 deletions and 77 additions = 105 changes; `adv_width` and `adv_lr` share one set object, as in the code), judged by the test the fix changed, once with the test as it was at `973a97d` and once as it is now. I also recomputed the numbers of commit `8e01f15` (section 7).

*Result.* Before the fix 1 of 28 deletions and 8 of 77 additions were caught; now 28 of 28 and 77 of 77. The first review's R05c (remove `reference` from `_NETWORK_ARMS`) is among the deletions. The lead's `mutate.sh` entries for the two tables are still killed. The README row counts are confirmed (section 7).

### 3.4 CF-04: SI-54 as built has two limits that the documents did not state: **closed** (wording note CF-R04)

*Result.* Both limits now written in `docs/PREREG.md:367-372` and in `docs/EQUIVALENT_COMMITS.json` ("meaning") are what the code does. Limit 1 (score rows carry the scorer's commit, not the fit's): row P2-06 (a Sinkhorn latent fitted at `c88cce3` and rescored at the tag) is refused by the runner's start-up check, and the rule scripts cannot see it because they read scores only. Limit 2 (one commit required, not "the tag"): row P2-10 (a whole stage at one third commit, Sinkhorn included) is accepted. The 24 rule-level cases of the first review give the same result as before (cases not as expected: 0; `scripts/prereg_rules.py` is unchanged). The machine-read fields of `EQUIVALENT_COMMITS.json` (`tag`, `commit`, `commit_ref`, `not_for_arms`) are identical at `c3fd92d` and `08f1e3c`; its `evidence` list was reformatted from file names to records, and no code reads `evidence`, `meaning` or `valid_for_a_candidate_commit_if` (grep over `scripts/`, `cluster/`, `tests/`).

### 3.5 CF-05: two comments about which refs a git bundle carries: **closed** (two follow-ups are deferred, both written down)

*Result.* I rebuilt the first review's toy bundles with git 2.43.0: a clone brings the branches and the tags that the bundle lists and no other ref; a tag resolves in the clone only if the bundle lists it.

| toy bundle | the bundle lists | clone gets (branches; tags) | lightweight tag resolves | annotated tag resolves |
|---|---|---|---|---|
| A_branch_only | `refs/heads/tmpbranch` | origin/tmpbranch; none | NO | NO |
| B_branch_plus_light_tag | `refs/heads/tmpbranch refs/tags/light-v1` | origin/tmpbranch; light-v1 | YES | NO |
| C_tag_only | `refs/tags/light-v1` | none; light-v1 | YES | NO |
| D_non_branch_non_tag | `refs/rehearsal/x` | none; none | NO | NO |
| E_branch_plus_other_ref | `refs/heads/tmpbranch refs/rehearsal/x` | origin/tmpbranch; none | NO | NO |
| F_branch_plus_ann_tag | `refs/heads/tmpbranch refs/tags/ann-v1` | origin/tmpbranch; ann-v1 | NO | YES |
| G_notes_ref | `refs/heads/tmpbranch refs/notes/n` | origin/tmpbranch; none | NO | NO |

In every case no other ref (`refs/rehearsal/*`, `refs/notes/*`) reaches the clone.

Both new comments say this correctly. The two shell scripts are identical to their `973a97d` versions once comments are removed, and both pass `bash -n`. The statements in the comments about SI-54 match the scenario runs: with the tag absent a mix of `c88cce3` and new rows is refused (P1-01, P1-05); a Sinkhorn output at `c88cce3` is refused whatever the bundle lists (P1-02, P1-04). The repository's two bundle commands (`cluster/jhpce/local_rehearsal.sh:46`, `experiments/X1_jhpce/rehearse_x1_job.sh:45`) list only a temporary branch, so the tag does not resolve in their clones.

*Why closed and not partly.* The triage plan for CF-05 had two parts: correct the two comments (done and verified above) and list the tag in the bundle command at the post-tag step G-17, a deferral that the repository documents (`notebook/2026-10-13.md:160`, `experiments/guard_followups_20261013/README.md:16`). The third suggestion of the first review, (ii) below, was not in the plan and is recorded as non-gating (`notebook/2026-10-13.md:361`). Without those notes the status would be partly.

*Left over (both were planned as deferred and both are written down).* (i) Listing the tag in the X1 bundle command is the post-tag step G-17: named in `experiments/guard_followups_20261013/README.md:16` and `notebook/2026-10-13.md:160`, and in the PREREG sentence "a JHPCE bundle that combines rows from both commits must carry the tag" (`docs/PREREG.md:375`). `experiments/X1_jhpce/x1_jhpce_command.py:23` still pins `TAG_SHA` to `c88cce3`; the 2 bundle commands in the repository name no tag. (ii) The third suggestion of the first review (`latent_git_shas_match_head`, `cluster/jhpce/prod_helpers.py:365`, is a raw comparison that reads False for a valid mixed stage) was not in the triage plan and is unchanged; `notebook/2026-10-13.md:361` records it as a non-gating field that compares raw SHAs. After CF-02 the list it compares also holds the commits of divergence records.

### 3.6 CF-06: PREREG says the no-pair counts are "reported with the results" but no code reads them: **closed**

The sentence now says that a reporting script will tabulate them after the runs and that no code reads `no_pair_steps` yet. Outside tests, docs and notebook, the key is mentioned in `scripts/fit_paper_config.py:506` (the record writer); nothing reads it.

### 3.7 CF-07: the pilot R2/R3 counts quoted in documents come from the frozen record: **closed**

I ran `scripts/decide_a2_a3.py` of `08f1e3c` on the recorded pilot inputs (1434 of 1434 recorded inputs located by SHA-256; the pilot score files under `tier12_v2/` only). Because SI-54 refuses the `c88cce3` Sinkhorn rows by design, the run used a temporary equivalence list that excludes no arm, as the first review did.

| | R2 evaluable | R2 mean change | R3 evaluable | R3 mean change | choices |
|---|---|---|---|---|---|
| frozen record (code of `c88cce3`) | 12 of 12 | -0.0014 | 17 of 20 | +0.0010 | mean / off |
| `08f1e3c`, real low-signal flags | 9 of 12 | -0.0042 | 13 of 20 | -0.0014 | mean / off |
| `08f1e3c`, flags cleared (control) | 12 of 12 | -0.0014 | 17 of 20 | +0.0010 | mean / off |

The control reproduces the frozen record, so the SI-66 change is inert when no cell is flagged. With the flags, 3 more of the 12 R2 cells and 4 more of the 20 R3 cells become "could not be judged" (evaluable cells: R2 12 to 9, R3 17 to 13; R3 had 3 cells that could not be judged before and has 7 now). Both rules then fail their coverage condition (`coverage` is false in both). R3 has 13 evaluable cells and needs 14. R2 still has 9 and needs 8, but group `immune|1` (written task|decoder) has no evaluable cell left, so the condition "at least one evaluable cell per task and decoder" fails (`scripts/prereg_rules.py:646`, SI-66). The choices are unchanged (mean for R2, off for R3). The X1 manifest written by the new code is byte-identical to the one the old code writes (SHA-256 starts `7c2fe8679631c882`).

### 3.8 CF-08: the Sinkhorn cap test read the constant, not the default in force: **closed** (follow-on note CF-R02)

*Result.* The test now asserts that the default `max_iter` of `sinkhorn_divergence_weighted` equals `SINKHORN_MAX_ITER` and that the huge-latent row matches `marginal error inf .* after 1 iterations`. Planted bugs: default changed to 50,000 (killed by `test_sinkhorn_matches_converged_reference`), constant changed to 50,000 (killed), the huge-latent row fed an ordinary latent (killed by the new pattern: `test_sinkhorn_refuses_bad_p_and_unconverged_solve`), the cross solve made to stop on an infinite error (killed). Two further bugs survive: the production caller `sinkhorn_batch_others` passes its own `max_iter=50000` (`172 passed in 95.51s (0:01:35)`), and one of the two symmetric solves receives a cap of 1,000 (`172 passed in 86.17s (0:01:26)`). The fix closes the path the finding named; the call sites are CF-R02. The comment on the constant's assertion (`tests/wcd_vae/test_divergences_v2.py:140`) still cites SI-59, which is retired; the live decision SI-61 fixes the same cap (500,000).

### 3.9 CF-09: `np.trapezoid` needs NumPy 2 but the full-suite script runs the test under NumPy 1.26: **closed**

`test_pool_brute_force` passes in env `wcd-gpu` (NumPy 1.26.4; 2 cases) and in env `scvi-api` (NumPy 2.4.6; 2 cases). No call of `np.trapezoid` remains (grep over `scripts`, `tests`, `src`, `cluster`, `experiments`: 1 match, the explanatory comment at `tests/analysis/test_analysis_x1.py:17`; 0 calls). The whole "other" group of `experiments/integration_checks_20261004/run_all_suites.sh` (all tests except `scvi`, `x13`, `stage`, `jhpce`, `scoring`) run in `wcd-gpu` gives 396 passed (the first review saw `test_pool_brute_force` fail there).

### 3.10 CF-10: tooling (ruff, fl_lint, check_tests warnings): **partly**

*Done and verified.* The unused `import shutil` is gone from `tests/test_legacy_retired.py` and no `shutil` reference remains; that file is outside the fit and score import closure. `scripts/fit_paper_config.py` is unchanged (so are the other files of the closure: section 4). Both `check_tests.py` tiers print "OK" with no warning at base `973a97d` (section 6).

*Remaining.* (i) ruff (rules F and E9, version 0.14.0 as in the fix report) still reports **28** findings tree-wide at `08f1e3c` (23 unused imports, 4 unused variables, 1 f-string without placeholder) and 28 at `973a97d`: the removed `shutil` is offset by one new finding, `experiments/analysis_numerics_20261014/check_pool_numerics.py:153` (unused variable `units`), which came with commit `6aaa3ac`, a descendant of the fixer's base `973a97d`. The fix report's "28 before, 27 after" therefore holds for the fixer's tree only. Under ruff 0.11.13 (the version pinned in `conda-lock.yml`) the count is 28; under the repository's own ruff configuration `ruff check` reports 4633 findings, so "ruff clean" in the notes can only mean rules F and E9. Five findings are inside the code of the identity proof (`scripts/fit_paper_config.py` lines 24, 286, 287; `src/wcd_vae/wcd/adversarial.py:15`; `src/wcd_vae/wcd/scvi_backbone.py:31`) and are deliberately unchanged; the other 23 are not in that code. (ii) `scripts/fl_lint.py` is not in the repository. The fix report records `fl_lint` 1.0, 26 rules and the start of the SHA-256 of the helper's `kernel.py` (the helper of the `fail-loud-pipelines` skill), but the repository notes do not: `fl_lint` appears in 14 lines of 5 files of `notebook/`, `docs/` and `CONSTRAINTS.md`; 0 of them give a version, rule count or checksum; one (notebook/2026-10-14.md:232) says the tool is not in the repository. An outside reader can reproduce the ruff count and the `check_tests.py` runs; they cannot check an "fl_lint: 0 findings" line. I cannot run it either (not in my skill catalogue), so I did not check the fix report's "0 findings, 2 suppressions". (iii) "Run `check_tests.py` once with both test sets" is not possible: `scib` imports in `wcd-gpu` but not in `scvi-api`, and `scvi` imports in `scvi-api` but not in `wcd-gpu` (import check: scvi-api: scvi yes, scib no; wcd-gpu: scib yes, scvi import fails (torchmetrics)), so the lead's two runs are the achievable record.

## 4. Does the identity proof still hold? (check 2)

**Yes.** The proof at `c3fd92d` (as quoted in the task: 59 of 59 non-Sinkhorn fits array-equal to `c88cce3`, 4 of 4 Sinkhorn fits different, 7 of 7 latents scored identically) carries over to `08f1e3c` for these reasons, each run or read by me:

1. The import-closure probe of the first review (`probes/import_closure.py`), re-run for `c3fd92d` to `08f1e3c`: of the fit entry points (8 files) only `scripts/run_stage.py` changed; scorer: none; cluster job helpers: none. The output is identical to the lead's for `06345ce` once the name of the last commit is unified (0 differing lines; `outputs/evidence/import_closure_diff_vs_lead.txt`).
2. A direct `git diff c3fd92d 08f1e3c` over `src/`, `fit_paper_config.py`, `scvi_adversarial_plan.py`, `score_scib_native.py`, `fit_outcome.py`, `host_info.py`, `prereg_rules.py`, `run_cpu_baselines.py`, `build_paper_manifest.py`, `prod_helpers.py`, `pyproject.toml`, `environment.yaml` and `conda-lock.yml` is empty. The probe is static, so I checked `src/` directly: the fitter loads `src/wcd_vae/wcd/*.py` by file path.
3. `scripts/run_stage.py` is a program that is run, not imported: the only importers are two test files, and the identity proof calls `fit_paper_config.py` and `score_scib_native.py` directly (`experiments/integration_checks_20261013/fit_identity.sh`, `score_identity.sh`). The three entry points of the proof import only `fit_outcome`, `scvi_adversarial_plan` and `host_info`.
4. Every changed line of `run_stage.py` since `c3fd92d` (3 places, 9 lines added, 3 removed; the file was unchanged between `c3fd92d` and `973a97d`): (a) three docstring lines (the first line, which the argument parser uses as its description, is not touched); (b) at start-up, `made_at.append((status git_sha, arm))` feeds only `refuse_uncovered`, which can only refuse the run; (c) the ledger's `git_sha` column falls back to the record's commit; it is read by the final one-commit check and by the summary list. None builds a fit or score command line, an environment variable or a thread setting, so none can change a fit or a score; they can only refuse a run or write provenance.

I did not refit rows: the question is not ambiguous. The identity CSV files were not opened (blinding rule); I rely on the counts quoted in the task.

## 5. SI-54 end to end (check 3)

Setting: a stage of five rows run with the test fakes (non-Sinkhorn `n1` and `n2` scored; Sinkhorn `s1` scored; non-Sinkhorn `v1` diverged; Sinkhorn `sv1` diverged). `O` = `c88cce3`; `T` = the commit named by the local tag `prereg-tier12-v3`; `L` = a commit made after the tag; `X` = an unrelated third commit; `HEAD` = the clone's HEAD while the tag does not exist. Phase P1: no tag; P2: the tag exists; P3: a commit after the tag exists. "S" in a scenario name means Sinkhorn rows, "non-S" all other arms. Expected outcomes come from the script's own restatement of SI-54 (start-up refusal of Sinkhorn output at `O`; start-up refusal of mixed scorer commits in normal runs; final refusal of mixed fit commits or scorer commits; `O` read as the tag only if the tag resolves and the rows mix). Scenarios that differ from the expected outcome: 0 of 25.

| scenario | normal run | `--report-only` run | note |
|---|---|---|---|
| P1-01 non-S at O, S at HEAD (tag absent) | refused at start-up | refused at final check (fit and scorer commits) | the mix stays and is refused |
| P1-02 non-S at O and S at O (tag absent) | refused at start-up | refused at start-up | Sinkhorn at O is refused whatever the tag |
| P1-03 everything at HEAD (tag absent) | accepted | accepted |  |
| P1-04 only the S divergence record at O (tag absent) | refused at start-up | refused at start-up | CF-02 case without the tag |
| P1-05 all non-S at O, S rows at HEAD, S record without commit | refused at start-up | refused at final check (fit and scorer commits) | mix of O and HEAD while the tag is absent |
| P2-01 X1 target: non-S at O, S at T | accepted | accepted | accepted together |
| P2-02 as 01 but S divergence record at O | refused at start-up | refused at start-up | CF-02 case |
| P2-03 as 01 but S divergence record names no commit | accepted | accepted | limit: a record without a commit is skipped |
| P2-04 as 01 but non-S record at a third commit | refused at final check (fit commits) | refused at final check (fit commits) |  |
| P2-05 non-S fits mixed O and T | accepted | accepted |  |
| P2-06 S latent fitted at O, rescored at T | refused at start-up | refused at start-up | CF-04 limit 1: only the runner sees it |
| P2-07 S latent at T, S score at O | refused at start-up | refused at start-up |  |
| P2-08 non-S fits at T, scores at O; S at T | accepted | accepted | fit and scorer commits are checked separately |
| P2-09 one non-S scorer at a third commit | refused at start-up | refused at final check (scorer commits) |  |
| P2-10 whole stage at a third commit incl. Sinkhorn | accepted | accepted | CF-04 limit 2: accepted |
| P2-11 everything at T | accepted | accepted |  |
| P2-12 non-S at O, S at a third commit | refused at start-up | refused at final check (fit and scorer commits) |  |
| P2-13 S latent names no commit, S score at T | accepted | accepted | limit: a latent without a commit is skipped |
| P2-14 S score at O only (S latent at T) | refused at start-up | refused at start-up |  |
| P2-15 non-S rows at O, one non-S divergence record without commit | accepted | accepted | a record without a commit is skipped |
| P3-01 non-S at O, S at the commit after the tag | refused at start-up | refused at final check (fit and scorer commits) | the tag commit is T, the rows are at L |
| P3-02 whole stage at the commit after the tag | accepted | accepted |  |
| P3-03 non-S at T and L | refused at start-up | refused at final check (fit and scorer commits) |  |
| P3-04 non-S at O, S at T (the tag still resolves to T) | accepted | accepted |  |
| P3-05 non-S at O and T, S at L | refused at start-up | refused at final check (fit and scorer commits) |  |

Reading the table: (1) rows made at `c88cce3` and at the new tag are accepted together for non-Sinkhorn arms only (P2-01, P2-05, P2-08, P3-04); (2) a Sinkhorn score, latent or divergence record at `c88cce3` is refused in every phase (P1-02, P1-04, P2-02, P2-06, P2-07, P2-14); (3) if the tag does not resolve, a mix is refused (P1-01, P1-05); (4) a stage made entirely at one other commit is accepted (P2-10, P3-02), which is limit 2 as documented; (5) a commit after the tag mixed with rows of another commit is refused (P3-01, P3-03, P3-05); (6) records and latents without `git_sha` are skipped (P2-03, P2-13, P2-15), see CF-R03. I found no path that accepts a wrong mix except the two documented limits and (6), and no valid mix that is refused. The rule-level matrix (24 cases of `probes/si54_scenarios.py`) is unchanged.

## 6. Test tiers and the base question (check 4)

Commands, per tier in its own clone at `08f1e3c`, exactly as in the lead's `README.txt` (one thread, `CUDA_VISIBLE_DEVICES=`, `--base 973a97d`):

| tier | result | lead's value (quoted in the task) |
|---|---|---|
| scripts tier (env scvi-api; `tests/analysis tests/prereg tests/scvi tests/stage tests/jhpce tests/test_legacy_retired.py`) | 421 passed, 18 skipped, `check_tests: OK`, 840 s | 421 passed, 18 skipped, about 14 min |
| src tier (env wcd-gpu; `tests/wcd_vae`) | 172 passed, `check_tests: OK`, 104 s | 172 passed |

The tested tree of the lead (`fd9c8e3`, `46a2df12`) differs from `08f1e3c` in 5 files, all `docs/*.md` or `notebook/`; 0 code or data files differ. My runs are at `08f1e3c` itself, so they also cover those commits.

Extra runs beyond the lead's: tests that need the prepared data folder (`tests/scvi` with `PREPPED_DIR`): 113 passed, 1 skipped; end-to-end stage test with the real fitter, scorer and R: 1 passed; "other" group under NumPy 1.26.4: 396 passed; `tests/scoring` (env wcd-kbet with R): 14 passed.

**Which base for the final record?** `check_tests.py` computes coverage once and compares it with the lines changed since the base, so I re-evaluated the saved coverage for other bases without new test runs, and also merged three tiers (scripts, scoring, src):

| base | scripts tier alone | src tier alone | non-test files measured by the 3 tiers merged, and not run |
|---|---|---|---|
| `973a97d` | OK | OK | 2 measured, 0 not run |
| `1e615eb` | OK | OK | not computed (`1e615eb` and `973a97d` differ only in notebook pages) |
| `6b4aefc` | OK | 10 warnings | 7 measured, 0 not run |
| `5cb75c0` | 7 warnings; FAIL: `scripts/check_tests.py`, `scripts/score_scib_native.py` | 14 warnings | 9 measured, 1 not run |
| `c88cce3` | 11 warnings; FAIL: `scripts/check_tests.py`, `scripts/score_scib_native.py` | 18 warnings | 9 measured, 1 not run |

Answer: `1e615eb` gives exactly the result of `973a97d` (the two differ only in notebook pages). `6b4aefc` gives the first review's picture: the scripts tier is clean, the src tier alone prints 10 warnings for tests that belong to the other tier (a limit of the tool). `5cb75c0` (the base of the user's diff review) does **not** pass with the lead's two tiers: `scripts/score_scib_native.py` lines 200, 202, 228, 229 are run only by `tests/scoring`, which needs env wcd-kbet and R and is not in the lead's selection (with `tests/scoring` added, those lines are run); `scripts/check_tests.py` itself (added in `3cbaca4`, after `5cb75c0`) has no tests; and `--src scripts` leaves out `cluster/jhpce/prod_helpers.py`, which changed since `6b4aefc`. So `973a97d` is the correct base for the claim "the code fixes are tested", and it needs only two tiers because only 2 non-test files changed since then (`scripts/analysis_x1.py`, tested by `tests/analysis`, and `scripts/run_stage.py`). A statement "every changed line of the diff-review range is tested" needs the three tiers merged, `--src scripts cluster` and a note that `check_tests.py` is untested tooling. That is CF-R05, an open question for the user.

## 7. Statements in the documents about code (check 5)

| statement | commit | verdict | evidence |
|---|---|---|---|
| "322 / 382 / 1,908 rows" (G-05 row of the guard README and `lam_audit.py`) | `8e01f15` | **confirmed** | `lam_audit.py` re-run: summed over files 1908, distinct tags 382, manifest of record 322; output identical to the committed `results/lam_audit.txt`; the 60 extra distinct tags are all `none` X3 rows of the first manifest, and none of its 828 X3 tags is in the manifest of record (0 shared of 828) |
| the two limits of SI-54 (PREREG, `EQUIVALENT_COMMITS.json`) | `f4b3b8f` | **confirmed** | section 5 rows P2-06 and P2-10; wording note CF-R04 |
| "no code reads `no_pair_steps`" | `8703cee` | **confirmed** | section 3.6 |
| pilot R2/R3 counts come from the frozen record; the current code reports R2 9 of 12, R3 13 of 20 | `d50348b` | **confirmed** | section 3.7 (full `decide_a2_a3.py` run on the recorded inputs, not only `decide_binary`) |
| PREREG SI-54 sentence names divergence records | `06345ce` | **partly** | the runner reads them (confirmed); "prereg_rules refuses the rule input" is true for scores only, because `failures.csv` has no commit column (`FAILURE_COLS`, `scripts/run_stage.py:99`); CF-R04 |
| `np.trapezoid` replaced by scipy's `trapezoid` | `8b90d79` | **confirmed** | section 3.9 |
| fix report: ruff 28 before, 27 after | fix report 3.6 | **partly** | 28 at the fixer's base `973a97d` and 28 at `08f1e3c` (section 3.10) |
| fix report: `fl_lint` 1.0, 26 rules, 0 findings | fix report 3.6 | **not checkable** | the tool is neither in the repository nor in my skill catalogue |

## 8. Change-size review (check 6)

Line counts are `git diff --numstat 973a97d 08f1e3c` (the fixer's base to the final commit). For every changed code or test line I asked the five questions of the brief (a bug, a risk, a missing test, an extra test, code that should not exist). A finding needs a concrete case; where I found none the cell says so.

| change | bug | risk | missing test | extra test | smaller |
|---|---|---|---|---|---|
| `tests/scvi/test_baseline_guard.py` (CF-01, +15/-6) | none: the chained comparison `a == n + 1 > 0` means `a == (n + 1) and n + 1 > 0`; with one batch every step is a no-pair step | none (test only) | none: 3 + 11 + 8 planted bugs killed | none: no new function, `check_tests` prints no warning | no |
| `scripts/run_stage.py` start-up check (CF-02, +2/-0) | none: `(info.get("status") or {}).get("git_sha")` skips a missing or null commit; for a non-finite latent the record copies the latent's commit, so a duplicate entry is harmless | refusal only; no write | one contrived gap (`M02f`) | none | the "commit of a row's outputs" is derived in two places (start-up and ledger); a shared helper would be smaller, no wrong result |
| `scripts/run_stage.py` ledger column (CF-02, +4/-1) | none found | side effects named in the fix report (section 3.2) and in NB-20261014-29: the ledger `git_sha` of a diverged row is no longer empty; `summary['latent_git_shas']` (copied by `prod_helpers.py`) also lists record commits although its name says latent; informational only | covered by `test_si54_old_commit_counts_as_the_tag_in_the_gate` | none | no |
| `tests/stage/test_run_stage.py` (CF-02, +28/-8) | none | the fake `diverge` behaviour still writes a record without `git_sha`, so the tests that need one set it by hand; a fake that follows the real contract does not break them (section 9.1: 0 stage tests fail with such a fake) | none | none: a third parameter and extra lines in an existing test | no |
| `tests/scvi/test_runner_and_manifest_missing_arms.py` (CF-03, +6/-1) | none | none | none: 105 of 105 table changes killed | none | no |
| `cluster/jhpce/local_rehearsal.sh`, `prod_fit_job.sh` (CF-05, +5/-1, +4/-1) | none: comments only, verified | none | none | none | no |
| `tests/wcd_vae/test_divergences_v2.py` (CF-08, +6/-1) | none: the new pattern matches the real infinite-error message and not the finite one | none | the cap at the call sites (CF-R02) | none | no |
| `tests/test_legacy_retired.py` (CF-10, +0/-1) | none: no `shutil` use remains | none | none | none | no |
| `tests/analysis/test_analysis_x1.py` (CF-09) | none: same call order `trapezoid(y, x)`; passes under NumPy 1.26.4 and 2.4.6 | none | none | none | no |

"A fix applied in one caller while the shared function stays broken": the SI-54 check has two callers in `run_stage.py` (start-up and final) and both now read records; `prod_helpers.py` does not call it.

## 9. New findings

| id | severity | where | problem and evidence | recommended change |
|---|---|---|---|---|
| CF-R01 | **fix** | `notebook/2026-10-14.md:333-358` (NB-20261014-29 and -30); `scripts/fit_paper_config.py:24,286,287` | The triage planned "record the fail-loud lint version" (CF-10). The fix report records it (`fl_lint` 1.0, 26 rules, checksum start); the repository does not: `fl_lint` appears in 14 lines of `notebook/`, `docs/` and `CONSTRAINTS.md`, as a step that was run or as a result of "no findings"; 0 give a version, rule count or checksum, and only one (notebook/2026-10-14.md:232) says the tool is not in the repository. The same holds for the ruff result (28 findings for rules F and E9, 5 of them in the code of the identity proof) and for the reason why no single `check_tests.py` run with both test sets exists (two environments). Two of the open items are already in the notebook (`latent_git_shas_match_head`: `notebook/2026-10-13.md:361`; G-17: `:160`). The user's diff review reads the repository. | Add one notebook NOTE with: the `fl_lint` version, rule count and checksum start (from the fix report), stated as the tool in use for the "0 findings" lines; the ruff count with its scope (rules F and E9 only; 4633 under the repository's own configuration); the two-environment reason for two `check_tests.py` runs; the base chosen (CF-R05). Answer to question 3 of the fix report: leave the 23 findings outside the identity proof's code alone for now. None is in the pre-registration, none changes a result, and each removal adds to the diff the user reviews; the NOTE records the count. No code change. |
| CF-R02 | note | `tests/wcd_vae/test_divergences_v2.py:131-142`; `src/wcd_vae/wcd/alignment.py:224-233, 257` | The CF-08 pin covers the default of `sinkhorn_divergence_weighted`, not the cap that the three solves receive in training. Case 1: `sinkhorn_batch_others` passes `max_iter=50000`: `172 passed in 95.51s (0:01:35)`; the training cap would be 50,000, not the 500,000 of SI-61. Case 2: one of the two symmetric solves gets 1,000: `172 passed in 86.17s (0:01:26)`. Both fail loudly at the cap, they do not give a wrong value. | In the existing spy on `alignment._ot_eps` (line 131) also record its last argument, spy on `_ot_eps_sym`, and assert that every recorded cap equals `alignment.SINKHORN_MAX_ITER`; this replaces the `inspect.signature` assertion with a stronger one. |
| CF-R03 | note (the user's open question) | `scripts/run_stage.py:422-423` (latent rule), `:429-440` (`status`), `:662-663`; `tests/stage/fake_env.py:89-92` | A divergence record is not held to the rule that applies to a latent. Probe on a clean clone, run without `--allow-dirty`: a latent whose `git_dirty` is true or missing is refused at start-up, but a divergence record whose `git_dirty` is true or null is accepted (the gate passes). A record with no `git_sha` is also accepted (rows P2-03 and P2-15; P2-13 is the same for a latent). The real fitter cannot write such a record (`provenance()` runs `git rev-parse HEAD` with `check=True`, `scripts/fit_paper_config.py:267-268`, and `record_divergence` writes `**provenance()`, `:519-520`), so only a hand-made or copied file is affected. The runner's own and the CPU runner's `nonfinite_latent` records carry `git_sha` but no `git_dirty` (`scripts/run_stage.py:964-967`, `scripts/run_cpu_baselines.py:162-166`). A divergence record counts as a failed fit in the analysis, so a record from the wrong code matters as much as a latent from it. | Answer to question 1 of the fix report ("refuse a record without `git_sha`?"): yes. In `Outputs.status()` refuse (a) any record without `git_sha` and (b) a record of status `diverged` whose `git_dirty` is not false, unless `--allow-dirty`, as for a latent. Do not demand `git_dirty` of `nonfinite_latent` records: a rule on every record refuses them (scenario S4 below). The test fake's `diverge` behaviour should write `git_sha` and `git_dirty` like the real fitter (one added argument), and one parametrized refusal test is needed. Launcher-only code: re-run the closure probe (section 4); the identity proof is unaffected. Detail in section 9.1. |
| CF-R04 | note | `docs/PREREG.md:372-373`; `docs/EQUIVALENT_COMMITS.json:3` | The sentence "Sinkhorn output made at c88cce3 (a score, a latent or a divergence record) is refused: prereg_rules refuses the rule input, and the run_stage preflight refuses the stage" reads as if the rule scripts also see latents and records. They see scores only (`failures.csv` columns: tag, status, detail, experiment, task, arm, lam, cond, seed, epoch, step, host, recorded_at). Limit 1 names latents only, and the JSON "meaning" says the runner reads "each latent's fit commit" (it also reads divergence records). PREREG is frozen with the tag. | One clause in each: "the rule scripts see scores only; a divergence record's commit is visible to the runner's start-up check only". |
| CF-R05 | note (the user's open question) | `scripts/check_tests.py` (base and `--src` arguments); lead pack `README.txt` | The record at `--base 973a97d` covers 2 non-test files. For `5cb75c0` the lead's scripts tier fails for `score_scib_native.py` (lines 200, 202, 228, 229) and `check_tests.py`; `--src scripts` omits `cluster/jhpce/prod_helpers.py`. Section 6 has the table. | Keep `973a97d` for the fixes; for the diff-review range run three tiers (scripts, scoring with R, src), `--src scripts cluster`, merge the coverage files (`scripts_recheck/ct_union_eval.py` in the evidence archive does this) and state that `check_tests.py` is untested tooling. |

### 9.1 Detail for CF-R03: what the recommended rule does and what it costs

Each rule was applied in a private clone (`scripts_recheck/r03_rule_validation_probe.py`); the stage runs without `--allow-dirty`. Scenarios: S0 a divergence record as the fake fitter writes it (after the one-argument change to the fake for rules A and N); S1 to S3 the same record with `git_dirty` set true, set null, or `git_sha` removed; S4 the `nonfinite_latent` record that the runner writes itself for a non-finite latent. "Now" is `08f1e3c` unchanged; rule A is the recommendation; rule N demands the dirty flag of every record.

| scenario | now (`08f1e3c`) | rule A (recommended) | rule N (dirty flag demanded of every record) |
|---|---|---|---|
| S0 divergence record as the fake fitter writes it | accepted | accepted | accepted |
| S1 divergence record, git_dirty true | accepted | refused at start-up | refused at start-up |
| S2 divergence record, git_dirty null | accepted | refused at start-up | refused at start-up |
| S3 divergence record, no git_sha | accepted | refused at start-up | refused at start-up |
| S4 runner-made nonfinite_latent record (git_sha, no git_dirty) | accepted | accepted | refused at start-up |

Cost on the existing tests (`tests/stage`, `08f1e3c`): the dirty-flag rule for `diverged` records alone makes 0 tests fail, because every stage test passes `--allow-dirty` (`tests/stage/test_run_stage.py:83` and `:388`, `tests/stage/test_run_stage_e2e.py:70`), so the new lines need a refusal test of their own. Adding the refusal of a record without `git_sha` makes 8 stage tests fail, all because the fake's `diverge` record carries no commit; with `git_sha` and `git_dirty` added to that one fake call (`tests/stage/fake_env.py:90-91`), 0 fail (`66 passed, 1 skipped in 142.31s (0:02:22)`). Separately, switching off the runner's existing refusal of a latent from a dirty checkout (`scripts/run_stage.py:422-423`) makes no test in `tests/stage` or `tests/jhpce` fail (`108 passed, 1 skipped in 312.26s (0:05:12)`): that rule has no test either, so the refusal test should cover latents and records together.

## 10. Limits of this re-check

1. `fl_lint`, `ponytail` and `ponytail-review` are not in my skill catalogue: the `fl_lint` claims are not checkable; the change-size questions were applied from the task text.
2. I did not refit rows or rescore latents (not ambiguous, section 4). The identity CSV files and everything under the sealed X1 folders were not opened; the numbers of the identity proof are the ones quoted in the task.
3. The `decide_a2_a3.py` run used a temporary equivalence list that excludes no arm, because the committed list refuses the pilot's `c88cce3` Sinkhorn rows by design.
4. The SI-54 scenarios use the test fakes for the fitter and scorer with commits rewritten in the outputs. The real fitter is represented by one real divergence record and by the end-to-end test (three rows).
5. All planted bugs are my own text replacements (8 + 5 + 1 control + 6) plus the first review's three and the lead's eleven; the 105 table changes were applied in place to the imported table, not to the file. This is not a complete mutation analysis.
6. The other-base evaluation re-uses coverage from the same test selection and a merge of three tiers computed by my own script (`ct_union_eval.py`), not by the repository's tool; the scoring tier ran only `tests/scoring` with `--base 5cb75c0`.
7. CPU only, one thread per library, no GPU, no cluster command; at most four processes at once.

## 11. How to repeat

The evidence archive `recheck_code_20261014_scripts_outputs.tar.gz` (saved with this report) holds the scripts (`scripts_recheck/`), every output (`outputs/`, with the saved code-reading facts in `outputs/evidence/`), the logs, `results.json`, the report template and render script, and the claim receipt `claim_receipt.txt`; `EVIDENCE_README.txt` in it lists the files. Clone: `git clone -q --branch final-review-fixes <bundle> rv && git -C rv remote remove origin`. Tiers: see section 6. Planted bugs: `python probes/my_mutate.py rv out.json R01a_weighted_branch_no_stats R01b_mmdref_lambda_drops_stats R01c_record_needs_two_steps`; `bash experiments/guard_followups_20261013/mutate.sh 08f1e3c <clone>`; `python scripts_recheck/mut_recheck.py <clone> out.json`; `python scripts_recheck/cf03_extra_arms_mutants.py <clone> out.json <test file at 973a97d>`. SI-54: `python scripts_recheck/si54_runner_matrix.py <fresh clone> out.json`. Other bases: `python scripts_recheck/ct_eval.py` and `ct_union_eval.py`. Provenance probe: `python scripts_recheck/status_vs_latent_provenance_probe.py <clean clone> out.json`. Cost of the CF-R03 rule: `python scripts_recheck/r03_cost_probe.py <work dir> out.json`. Saved code-reading facts: `bash scripts_recheck/evidence_greps.sh` and `evidence_greps2.sh` (from the workspace root, clone `rv`). Numbers: `build_results.py` writes `results.json`; this report is rendered from it by `render_report.py`.
