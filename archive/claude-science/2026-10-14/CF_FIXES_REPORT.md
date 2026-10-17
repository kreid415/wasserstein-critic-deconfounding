# Code fixes after the independent rigour review (findings CF-01, CF-02, CF-03, CF-05, CF-08, CF-10)

| | |
|---|---|
| Author, date | Runner Engineer (the implementation agent), track "Code fixes", 2026-10-14 |
| Starting point | commit `973a97d`, the tip of branch `final-review-fixes`. The review was written against `1e615eb`; only notebook pages differ between the two (`logs/final_1e615eb_vs_973a97d.txt`), so every line number in the review still holds. |
| Result | branch `cf-fixes` in a private clone: 6 commits (one per finding), 8 files changed, 73 lines added, 22 removed; tip `324eb83`. |
| Patches | [cf_patches.tar.gz]({{artifact:7710df50-9b51-47f3-9733-1b3fba00d649}}): 6 files for `git am`. Applied to `973a97d` in a fresh clone they give tree `e0cfe884369d`, the same tree as my branch (`e0cfe884369d`). The commit ids in this report belong to my private branch; `git am` creates new commits with other ids and the same content, and each patch subject starts with its finding id. |
| Evidence | [cf_evidence.tar.gz]({{artifact:5a10b655-d89e-4f42-b4da-5c706e6c7753}}): every log, probe output and script behind the numbers below, with a SHA-256 list. |
| Not touched (task rules) | `scripts/fit_paper_config.py`, everything under `src/`, `scripts/analysis_x1.py`, `tests/analysis/`, `docs/`, `notebook/`, `CONSTRAINTS.md`. No X1 output (X1 is the main experiment, sealed until the analysis code is frozen), X1 log or identity-review score file was opened, and `scripts/analysis_x1.py` was not run. No notebook entry was written; the decisions the lead may want to log are in section 6. |

## 1. Result at a glance

The independent reviewer (RIGOR_REVIEWER, a separate reviewing agent) read branch `final-review-fixes` at `1e615eb` and reported 2 findings to fix before the new tag and 8 notes (10 in all, CF-01 to CF-10; CF stands for "code finding"). This report covers the six that were assigned to this track. "As assigned" means the task named the part to do; the rest of that finding belongs to another track or was not requested.

| id | what the reviewer found, in one line | action | commit | proof |
|---|---|---|---|---|
| CF-01 | the counter of training steps with nothing to align was tested through a real training run for one of three call paths | fixed | `d169bf8` | the reviewer's planted bugs R01a, R01b, R01c: 3 of 3 survived before, 3 of 3 killed after |
| CF-02 | SI-54 (the user's rule that `c88cce3` and the new tag count as the same code except for Sinkhorn) did not read the commit stored in a divergence record (the file written when a fit's training loss becomes non-finite) | fixed | `b757932` | gap reproduced at `973a97d` (rerun exit 0, gate PASS) and refused at the new commit (exit 4); 3 of 3 planted bugs killed, plus 1 control that correctly survives; the import-closure check lists only `scripts/run_stage.py` as changed |
| CF-03 | the table test did not pin which arms each extra setting applies to | partly fixed (the test part, as assigned) | `f969a11` | planted bugs on the table: 3 of 3 arm removals survived before (an added arm was caught already), 4 of 4 killed after |
| CF-05 | two comments wrongly describe which refs (branches and tags) a git bundle, the single file that carries a repository to the cluster, contains | partly fixed (the two comments, as assigned) | `5244cf7` | experiment in section 3.4; both scripts are identical once comments are stripped |
| CF-08 | the test of the Sinkhorn cap (the maximum number of iterations) read the constant, not the default in force; the test row with a huge latent matched an error message that an ordinary latent also gives | fixed | `1e2bbe1` | planted bugs: 2 of 2 survived before, 2 of 2 killed after |
| CF-10 | ruff (a Python code checker) is not clean; `fl_lint` (the fail-loud linter) is not in the repository | partly fixed (the tooling part, as assigned) | `324eb83` | the four ruff findings the reviewer named (in two files): 4 before, 3 after; tree-wide 28 before, 27 after |

What each commit changes (lines added / removed):

| commit | finding | files (lines added / removed) |
|---|---|---|
| `d169bf8` | CF-01 | `tests/scvi/test_baseline_guard.py` (15 / 6) |
| `b757932` | CF-02 | `scripts/run_stage.py` (9 / 3); `tests/stage/test_run_stage.py` (28 / 8) |
| `f969a11` | CF-03 | `tests/scvi/test_runner_and_manifest_missing_arms.py` (6 / 1) |
| `5244cf7` | CF-05 | `cluster/jhpce/local_rehearsal.sh` (5 / 1); `cluster/jhpce/prod_fit_job.sh` (4 / 1) |
| `1e2bbe1` | CF-08 | `tests/wcd_vae/test_divergences_v2.py` (6 / 1) |
| `324eb83` | CF-10 | `tests/test_legacy_retired.py` (0 / 1) |

Gates on the final commit (details in section 4): `check_tests.py` exits 0 with 0 warnings for the scripts tier and exits 0 with 0 warnings for the `src` tier; `tests/stage` 66 passed, 1 skipped; `tests/scvi` 98 passed, 16 skipped; the lead's own 11 planted bugs: 11 of 11 still killed. With the reviewer's older base `6b4aefc` the scripts tier is clean as well, and the `src` tier alone prints the review's 10 warnings, a limit of the tool (section 3.6, item 4).

## 2. Words used in this report

| term | meaning |
|---|---|
| commit, SHA, tag | A commit is a saved version of the code, named by a 40-character fingerprint (SHA, secure hash algorithm) usually shortened to 7 characters. A tag is a readable name for one commit. `c88cce3` is the commit tagged `prereg-tier12-v2`, the code frozen so far. `prereg-tier12-v3` is the new tag, not yet made. `973a97d` is where this work starts, `1e615eb` is what the reviewer read, and `c3fd92d` is the commit at which the identity proof was last run. |
| SI-nn | A numbered standing instruction: a decision of the user recorded in `CONSTRAINTS.md`. SI-52: draft all fixes on a branch, get an independent review, and show the user the diff before any new tag. SI-54: treat `c88cce3` and the new tag as the same code for every arm except Sinkhorn; Sinkhorn rows must carry the new tag. |
| G-nn, R2-nn, CF-nn | Finding numbers of earlier reviews: G of the guard review, R2 of the Sinkhorn re-review, CF of this review. |
| arm, MMD, Sinkhorn | An arm is one method variant of the benchmark. MMD (maximum mean discrepancy) and Sinkhorn (an entropy-smoothed optimal-transport distance) are two penalties that measure how different the latent-space distributions of two experimental batches are without training a critic network. `mmd_ref` is MMD against one reference batch. |
| X1, X3, X7, X12 | Experiment numbers of the pre-registered design (`docs/PAPER_PLAN.md`). X1 is the main experiment; its scores stay sealed during this work. X3: composition-shift dose-response, with an importance-weighted control. X7: reference-batch selection (every batch is the reference in turn). X12: robustness to adversary architecture and hyperparameters. |
| importance weights | Per-cell weights that the X3 experiment gives to the MMD penalty. |
| minibatch, no-pair step | A minibatch is the 128 cells used for one training step. A no-pair step is one in which the minibatch gives the alignment penalty nothing to compare (one batch only; for `mmd_ref`, no reference cell or no other batch). The penalty is then 0, and since guard finding G-01 such steps are counted. |
| latent, fit, manifest row | A fit is one training run (one manifest row: arm, task, settings, seed). Its latent is the saved low-dimensional embedding of every cell. |
| divergence record | The small file `status/<tag>.json` that the fitter writes, instead of a latent, when the training loss becomes non-finite (not a number, or infinite). The row then counts as the final outcome `diverged`. |
| runner, stage, preflight, gate, ledger | The runner (`scripts/run_stage.py`) launches the fits and the scoring of a list of rows (a stage). The preflight is its set of checks before any work. The completion gate is its check at the end that every row has a final outcome and that the outputs share one commit. The ledger is the per-row table and summary it writes (`ledger/<stage>.csv` and `.json`). |
| planted bug ("mutant"), killed, survived | A deliberate one-line breakage of the code made to test the tests. Killed: at least one test fails. Survived: every test still passes. |
| identity proof, import closure | The identity proof refits and rescores sample rows with the old and the new code and compares them exactly. The import closure of a program is every code file it loads, directly or indirectly; if none of them changed, the program cannot behave differently. |
| `check_tests.py`, ruff, `fl_lint` | `check_tests.py` is the repository's test-economy check: it fails if a changed line or branch of the checked code is never run by a test, and warns about a new test that covers nothing the others miss. ruff is a Python code checker (rules F: unused imports and variables; E9: syntax errors). `fl_lint` is the fail-loud linter for silent-failure patterns. |
| passed, skipped | Counts of pytest test cases. A skipped case needs something that is absent, mostly the prepared data folder. |
| tier, env | A tier is a group of test folders that run together in one conda environment (env). The scripts tier tests the scripts and cluster code (env `scvi-api`); the `src` tier tests the model code under `src/` (env `wcd-gpu`). `check_tests.py` runs once per tier. |

## 3. Findings

### 3.1 CF-01: the no-pair step counter was tested through a real training run for one call path only (fixed, `d169bf8`)

**What the reviewer found.** During training, the batch-alignment penalty compares the batches present in each minibatch. When a minibatch gives it nothing to compare, the step is counted and, if the count is above zero, the count is saved in the fit's record as `no_pair_steps`; `docs/PREREG.md` says these counts are reported with the results. The only test that read the count after a real training run used the plain `mmd` call. Three one-line breakages passed every test: R01a (the importance-weighted call stops passing the counter), R01b (the wrapper for `mmd_ref` stops forwarding it; this is the arm expected to have the most such steps in X7) and R01c (the record is written only for a count above 1).

**What I checked.**
1. The code at `973a97d`: the wrapper at `scripts/scvi_adversarial_plan.py:231`, the plain call at `:369`, the importance-weighted call at `:371`, the threshold `if n:` at `scripts/fit_paper_config.py:504-506`, and the single existing check at `tests/scvi/test_baseline_guard.py:248-252`. All are as the finding says.
2. Which arms reach which call: importance weights are accepted for the arms in `_IW_ARMS` (`scvi_adversarial_plan.py:60`: discriminator, reference, pooled, mmd); of these only `mmd` is a critic-free loss, and only `mmd_batch_others` takes a `weights` argument (`src/wcd_vae/wcd/alignment.py:109`). The weighted call is therefore exercised with `mmd` plus importance weights.
3. The finding itself: I ran the reviewer's own script (`probes/my_mutate.py`) on a clone at `973a97d`. R01a: survived (98 passed, 16 skipped); R01b: survived (98 passed, 16 skipped); R01c: survived (98 passed, 16 skipped).

**What I changed** (test file only; `scripts/fit_paper_config.py` is untouched, as the identity proof at `c3fd92d` requires). The one-batch fit that already existed inside `test_resolved_plan_known_answers` now runs four times: `mmd`, `mmd_ref`, `sinkhorn` and `mmd` with importance weights. After each fit it overwrites the counter with 1 and checks that the record is still written (the exactly-one-step case). No new test function was added. The test keeps its name on purpose: the lead's own mutation list (`experiments/guard_followups_20261013/mutants.tsv`, mutant `G01_plan_no_stats`) names it, and a move of the check into a new function would have left that mutant without the test the list names for it.

**Proof.** Same script, final commit; the last rows are two extra planted bugs of mine.

| planted bug | what it breaks | before (tests at `973a97d`) | after (tests at the final commit) | test that failed after |
|---|---|---|---|---|
| `R01a_weighted_branch_no_stats` | the importance-weighted call stops passing the counter | survived | killed | tests/scvi/test_baseline_guard.py::test_resolved_plan_known_answers |
| `R01b_mmdref_lambda_drops_stats` | the wrapper for `mmd_ref` stops forwarding the counter | survived | killed | tests/scvi/test_baseline_guard.py::test_resolved_plan_known_answers |
| `R01c_record_needs_two_steps` | the record is written only for a count above 1 | survived | killed | tests/scvi/test_baseline_guard.py::test_resolved_plan_known_answers |
| `C01d_plain_call_drops_stats` | the plain call stops passing the counter (the lead's mutant `G01_plan_no_stats`) | not applicable | killed | tests/scvi/test_baseline_guard.py::test_resolved_plan_known_answers |
| `C01e_sinkhorn_drops_stats` | `sinkhorn_batch_others` stops counting no-pair steps | not applicable | killed | tests/scvi/test_baseline_guard.py::test_resolved_plan_known_answers |

The lead's own script `experiments/guard_followups_20261013/mutate.sh`, run on the final commit: 11 of 11 planted bugs killed, as before.

### 3.2 CF-02: SI-54 ignored the commit stored in a divergence record (fixed, `b757932`)

**What the reviewer found.** SI-54 says that Sinkhorn rows must carry the new tag. The runner enforced this for latents and scores. A Sinkhorn row whose only output is a divergence record (no latent, no score) was not checked, although the record carries the commit of the code that made it. An old-code (`c88cce3`) Sinkhorn failure would therefore have counted as a final outcome of the new stage.

**What I checked.**
1. `scripts/run_stage.py` at `973a97d`: the start-up check collects the commits of existing latents and scores only (lines 659-663); the final one-commit check uses the ledger's `git_sha` column, which was filled from the latent only (lines 1135 and 1166-1167).
2. The real fitter does write the commit into its divergence record: `record_divergence` (`scripts/fit_paper_config.py:514-521`) passes `**provenance()` (`:264-272`), and the reviewer's `real_status_record_output.txt` shows the key `git_sha`. The test stand-in for the fitter (`tests/stage/fake_env.py`, behaviour `diverge`) writes none; I left it unchanged and set the field in the tests, as the reviewer's probe did.
3. The gap itself, with the reviewer's probe on a clone at `973a97d`: rerun exit 0 with '[gate] PASS: 2/2 final', in a normal and in a `--report-only` run. On the final commit the same probe gives: rerun exit 4 (the runner's code for a refused preflight) with "existing output: ['sinkhorn'] rows at c88cce3 ... refit and rescore them at prereg-tier12-v3", in a normal and in a `--report-only` run.

**What I changed** in `scripts/run_stage.py` (three places, exactly the lines below, relative to both `973a97d` and `c3fd92d`, because the file is unchanged between those two: `logs/final_run_stage_c3fd92d_vs_973a97d.txt`):

```
@@ -51,2 +51,3 @@ any of them, so one stage cannot mix scorer commits, hosts' vector widths, numba
-SI-54 (docs/EQUIVALENT_COMMITS.json): in scorer_git_sha and the latents' git_sha, c88cce3 counts as tag
-prereg-tier12-v3 for every arm except sinkhorn; the preflight refuses existing Sinkhorn output made at c88cce3.
+SI-54 (docs/EQUIVALENT_COMMITS.json): in scorer_git_sha and the git_sha of latents and divergence records, c88cce3
+counts as tag prereg-tier12-v3 for every arm except sinkhorn; the preflight refuses existing Sinkhorn output (score,
+latent or divergence record) made at c88cce3.
@@ -660,0 +662,2 @@ class Stage:
+            if (info.get("status") or {}).get("git_sha"):     # a divergence record names its commit too (SI-54)
+                made_at.append((info["status"]["git_sha"], r["arm"]))
@@ -1135 +1138,4 @@ class Stage:
-                            device=lat.get("device", ""), git_sha=lat.get("git_sha", ""),
+                            device=lat.get("device", ""),
+                            # the commit of the fit: its latent's, else its divergence record's (the final SI-54 check
+                            # and summary['latent_git_shas'] read this column)
+                            git_sha=lat.get("git_sha", "") or (info.get("status") or {}).get("git_sha", ""),
```

In words: the start-up check also passes the record's commit to `prereg_rules.refuse_uncovered`; the ledger's `git_sha` column takes the latent's commit and, where there is none, the divergence record's, which is what the final one-commit check and `summary['latent_git_shas']` read; one docstring sentence now names divergence records. Two side effects, both intended and both limited to rows that have a divergence record with a commit: the ledger's `git_sha` for such a row is no longer empty, and `summary['latent_git_shas']` (which `cluster/jhpce/prod_helpers.py` copies into the job summary) now also lists that commit.

In `tests/stage/test_run_stage.py`: a third case `status` of `test_sinkhorn_output_at_the_pre_fix_commit_is_refused` (a Sinkhorn divergence record at `c88cce3` is refused before any work, also with `--report-only`), and the existing in-process gate test `test_si54_old_commit_counts_as_the_tag_in_the_gate` gained one diverged non-Sinkhorn row whose record, at another commit, fails the final check (and is listed in `mixed_git_sha`), and at the old commit counts as the tag's. The second change is needed because the start-up check alone would hide a missing change in the final check.

**Proof.** Planted bugs against the new code (the `before` column is not applicable: they break the new lines):

| planted bug | what it breaks | before (tests at `973a97d`) | after (tests at the final commit) | test that failed after |
|---|---|---|---|---|
| `C02a_preflight_ignores_status` | the start-up check does not read the divergence record's commit | not applicable | killed | tests/stage/test_run_stage.py::test_sinkhorn_output_at_the_pre_fix_commit_is_refused[status] |
| `C02b_preflight_status_other_arm` | the start-up check reads the record's commit but as a non-Sinkhorn arm | not applicable | killed | tests/stage/test_run_stage.py::test_sinkhorn_output_at_the_pre_fix_commit_is_refused[status] |
| `C02c_ledger_latent_only` | the final check ignores the record's commit | not applicable | killed | tests/stage/test_run_stage.py::test_si54_old_commit_counts_as_the_tag_in_the_gate |
| `C02d_ledger_status_first` | control: an equivalent change (the record's commit preferred over the latent's); it must survive | not applicable | survived | none (expected for a control) |

**Identity proof (`c3fd92d`) still stands.** The output of the reviewer's `probes/import_closure.py` on the final commit (`logs/final_import_closure.txt` in the evidence archive) is:
- fit entry points (`run_stage.py`, `fit_paper_config.py`, `scvi_adversarial_plan.py`, `run_cpu_baselines.py`; 8 files): changed between `c3fd92d` and the final commit and in the closure: `['scripts/run_stage.py']`
- scorer (`score_scib_native.py`; 2 files): none
- cluster job helpers (`prod_helpers.py`; 1 file): none

No file of the fit, scoring or cluster-helper closure other than `scripts/run_stage.py`, and nothing under `src/`, differs between `c3fd92d` and the final commit (`git diff --name-only c3fd92d HEAD` over those paths prints nothing; `logs/final_closure_evidence.txt`). Why `run_stage.py` cannot change a fit or a score: it is a launcher. No Python file under `scripts/`, `src/` or `cluster/` imports it (grep, `logs/final_no_importers.txt`); the identity proof refitted with `scripts/fit_paper_config.py` and rescored with `scripts/score_scib_native.py` directly (`experiments/integration_checks_20261013/fit_identity.sh:18`, `score_identity.sh:17`); and none of the three changed places builds a fit or score command line, an environment variable or a thread setting. The README of the proof states the criterion for carrying it to a later commit: the diff changes no fit or score code (`src/`, `fit_paper_config.py`, `scvi_adversarial_plan.py`, `score_scib_native.py` and their imports). That holds, so the proof need not be repeated for these changes.

**One dependency for the documents track.** `docs/PREREG.md` (the SI-54 paragraph, line 258 at `973a97d`) says that `run_stage.py` reads `scorer_git_sha` and the latents' `git_sha`. It now also reads the `git_sha` of a divergence record. The sentence should say so; I did not edit documents.

### 3.3 CF-03: the table test did not pin which arms each extra setting applies to (test part fixed, `f969a11`)

**What the reviewer found.** In `fit_paper_config.py` the table `EXTRA_ARMS` says to which arms each optional row setting (`adv_width`, `adv_lr`, `bary_iter`, `bary_warm_iter`, `sampler`) applies; any other arm is refused. The test pinned the keys of the table but looped over the arms the table itself lists, so deleting an arm from a list failed nothing. The second half of the finding, the README's "1,908 manifest rows" (occurrences summed over overlapping manifest files; the finding counts 382 distinct rows), belongs to the documents track and I did not touch it.

**What I checked.** `scripts/fit_paper_config.py:55-62` and `tests/scvi/test_runner_and_manifest_missing_arms.py:53-84` at `973a97d`; how the settings are consumed (`fit_paper_config.py:342-346` passes `adv_width`, `adv_lr`, `bary_iter`, `bary_warm_iter`, `sampler` to the fit, and the plan builds an adversary head only for the discriminator, reference, pooled and barycenter arms, `scvi_adversarial_plan.py:56-60`); the reviewer's census (`reference_adv_rows_output.txt`: 144 distinct X12 rows use the settings on arm `reference`). The planted bugs below reproduce the finding at `973a97d`.

**What I changed.** In the existing test, the assertion on the keys of `EXTRA_ARMS` now compares the whole table, arms included, with the 11 head-training arms written out as a literal. No new test; `fit_paper_config.py` untouched.

**Proof.**

| planted bug | what it breaks | before (tests at `973a97d`) | after (tests at the final commit) | test that failed after |
|---|---|---|---|---|
| `R05c_network_arm_removed` | `reference` removed from the arms that accept `adv_width` / `adv_lr` (144 distinct manifest rows use them) | survived | killed | tests/scvi/test_runner_and_manifest_missing_arms.py::test_extra_keys_are_consumed_or_refused |
| `C03b_bary_warm_arm_removed` | `barycenter_sn` removed from the arms that accept `bary_warm_iter` | survived | killed | tests/scvi/test_runner_and_manifest_missing_arms.py::test_extra_keys_are_consumed_or_refused |
| `C03c_sampler_reference_removed` | `reference` removed from the arms that accept the stratified sampler | survived | killed | tests/scvi/test_runner_and_manifest_missing_arms.py::test_extra_keys_are_consumed_or_refused |
| `C03d_network_arm_added` | `mmd` added to the arms that accept `adv_width` / `adv_lr` (an addition: the old loop over critic-free arms caught it) | killed | killed | tests/scvi/test_runner_and_manifest_missing_arms.py::test_extra_keys_are_consumed_or_refused |

### 3.4 CF-05: two comments about which refs a git bundle carries (comments fixed, `5244cf7`)

**What the reviewer found.** The comment at `cluster/jhpce/local_rehearsal.sh:40` says that a clone of a bundle "takes refs/heads/* only"; the comment at `cluster/jhpce/prod_fit_job.sh:26` says that latents fitted at another commit fail the runner's mixed-commit check, which stopped being true for non-Sinkhorn latents at `c88cce3` when SI-54 was introduced.

**What I checked.** The git behaviour, on toy bundles with git 2.43.0 (`cf05_bundle_test.sh`, output in `logs/cf05_bundle_refs.txt`):

| bundle lists | clone has branches | clone has tags | tag `v1` resolves in the clone |
|---|---|---|---|
| `refs/heads/tmpbranch` | origin/tmpbranch | none | NO |
| `refs/heads/tmpbranch refs/tags/v1` | origin/tmpbranch | v1 | YES |
| `refs/tags/v1` | none | v1 | YES |
| `refs/rehearsal/x` | none | none | NO |
| `refs/heads/tmpbranch refs/rehearsal/x` | origin/tmpbranch | none | NO |

A clone takes the branches and tags that the bundle lists, and no other ref; the tag resolves in the clone only if the bundle lists it. The consequence for SI-54, with the repository's real `prereg_rules.equivalent_commits` (`cf05_tag_resolution.py`, output in `logs/cf05_tag_resolution.txt`): with the tag absent the two commits stay two; with the tag present they become one; a Sinkhorn row at `c88cce3` is refused in both cases. The job clones its bundle with `git clone --no-checkout "$PWD/wcd.bundle"` (`cluster/jhpce/prod_command.py:91`).

**What I changed.** Comment text only, in two files. `local_rehearsal.sh`: 4 added comment lines above `TMPREF=` and a corrected trailing comment. `prod_fit_job.sh`: the sentence at line 26 now describes the one-commit check, the `c88cce3` exception, the need for the tag in the bundle and the unconditional refusal of Sinkhorn output at `c88cce3`. Both scripts are identical to their `973a97d` versions once comments are removed (`diff` of the comment-stripped files), and both pass `bash -n`.

**Not done (outside the assignment).** The reviewer's other two suggestions for this finding: list the tag in the bundle command of the X1 job (`experiments/X1_jhpce/x1_jhpce_command.py:83-85`, which also still pins `TAG_SHA` to `c88cce3`, the known item G-17), and make `latent_git_shas_match_head` (`cluster/jhpce/prod_helpers.py:365`, a raw comparison that is informational only) compare after `equivalent_commits`.

### 3.5 CF-08: the Sinkhorn cap test read the constant, and one test row did not need its premise (fixed, `1e2bbe1`)

**What the reviewer found.** The Sinkhorn solve stops with an error after a maximum number of iterations (the cap, 500,000, decision SI-61, which replaced SI-59 with the same cap; the existing comment on the asserted line still cites SI-59 and I left it). The test read the constant `SINKHORN_MAX_ITER`, but the training code calls `sinkhorn_divergence_weighted` without `max_iter`, so the function's default is what is in force. Separately, the row `_huge_latent_solve` checked only the text "after 1 iterations", which an ordinary latent also produces at `max_iter=1` (finite marginal error 0.0637); the row worked only because of its premise (an infinite first error).

**What I checked.** `src/wcd_vae/wcd/alignment.py:224-225` (default `max_iter=SINKHORN_MAX_ITER`) and `:249-257` (`sinkhorn_batch_others` passes no `max_iter`); the two real error messages, produced by running the function in the `wcd-gpu` environment (`logs/cf08_messages.json`): at scale 1e30 "L1 marginal error inf > tol 0.0001 after 1 iterations at the final eps", at scale 1 "L1 marginal error 0.0637 > tol 0.0001 after 1 iterations at the final eps". Both findings reproduce at `973a97d` (table below).

**What I changed** (test file only, no new test). In `test_sinkhorn_matches_converged_reference` an added assertion: `inspect.signature(sinkhorn_divergence_weighted).parameters["max_iter"].default == alignment.SINKHORN_MAX_ITER`. The huge-latent row now matches `marginal error inf .* after 1 iterations`, so it keeps the old check and also requires the infinite error.

**Proof.**

| planted bug | what it breaks | before (tests at `973a97d`) | after (tests at the final commit) | test that failed after |
|---|---|---|---|---|
| `R21c_cap_default_literal` | the default of `max_iter` becomes 50,000 while the constant stays 500,000 | survived | killed | tests/wcd_vae/test_divergences_v2.py::test_sinkhorn_matches_converged_reference[_standardised] |
| `C08b_huge_row_ordinary_scale` | the huge-latent test row is fed an ordinary latent (scale 1) | survived | killed | tests/wcd_vae/test_divergences_v2.py::test_sinkhorn_refuses_bad_p_and_unconverged_solve[_huge_latent_solve-RuntimeError-marginal error inf .* after 1 iterations] |

With the repository's verifier helper `fl_mutation_check`, using the real message at scale 1e30 as the good input and the real message at scale 1 as the bad input: the old pattern `after 1 iterations` accepts the bad input (the check cannot fail: "vacuous verifier"); the new pattern accepts the good input and rejects the bad one because it does not match `marginal error inf`.

### 3.6 CF-10: tooling (partly fixed, `324eb83`)

**What the reviewer found.** ruff (rules F and E9) is not clean tree-wide; `fl_lint` is not in the repository, so "fl_lint: 0 findings" cannot be checked by an outside reader; and `check_tests.py` printed 10 warnings in the `src` tier.

**What I did.**
1. Removed the unused `import shutil` in `tests/test_legacy_retired.py` (the only one of the four findings in changed files that lies outside the fit and score import closure). The other three are in `scripts/fit_paper_config.py` (lines 24, 286, 287), which the identity proof depends on; it is unchanged, as instructed.
2. ruff 0.14.0, rules F and E9, whole repository: 28 findings at `973a97d`, 27 at the final commit. In the two files that carried the reviewer's four findings: 4 before, 3 after (the three left are the ones in `fit_paper_config.py`). The remaining 27 findings are spread over 18 files; 5 of them are in `scripts/fit_paper_config.py` or under `src/`, the code the identity proof covers; the others are in other scripts, tests, `legacy/`, `handoff/` and `experiments/`. I did not touch them: they were not requested and a tree-wide clean-up is a separate, larger patch.
3. `fl_lint`: the helper that ships with the `fail-loud-pipelines` skill (version 1.0, 26 rules, `kernel.py` SHA-256 starting `12df2b75b331f456`), run on the 8 changed files: 0 findings at any severity, 2 suppressions with reasons (the same 2 as at `973a97d`). The linter itself is not in the repository; its version and rule count are recorded here, as the reviewer asked, and the decision whether to commit a copy is left to the lead.
4. `check_tests.py`. With `--base 973a97d` (the range of my six commits) both tiers print "OK" and no warning (section 4). The reviewer's 10 warnings came from `--base 6b4aefc` (the whole review range) in the `src` tier alone. On the final commit with that base: the scripts tier, with the reviewer's own test list, gives 418 passed, 18 skipped, no warning, "check_tests: OK" (`logs/final_ct_scripts_base6b4aefc.log`); the `src` tier gives 172 passed and 10 warnings, 7 for `tests/analysis`, 2 for `tests/prereg` and 1 for `tests/stage` (`logs/final_ct_src_base6b4aefc.log`), the very same 10 test functions as in the review (`logs/final_ct_src_warnings_vs_review.txt`: the two lists are identical). They are the reviewer's tool artefact: the script reads every changed test file but runs only the tests it is given, so tests that the scripts tier runs look uncovered in the `src` tier, where the same tests give no warning. I did not change the tool and did not try the reviewer's remedy of one combined run: `scvi-tools` is installed only in `scvi-api` and `scib` only in `wcd-gpu` (import check), and the repository runs the two test sets in those two environments.

## 4. Gates run on the final commit `324eb83`

All runs are CPU only (`CUDA_VISIBLE_DEVICES=`, one thread). "scvi-api" and "wcd-gpu" are the two conda environments the repository's test tiers use; the coverage runs use copies of them with `pytest-cov` 7.1.0 and ruff 0.14.0 added (`--system-site-packages`).

| gate | how | result |
|---|---|---|
| `check_tests.py`, scripts tier | `--base 973a97d --src scripts -- tests/scvi tests/jhpce tests/prereg tests/stage tests/test_legacy_retired.py`; env scvi-api, `WCD_SRC=src` | exit 0; 359 passed, 18 skipped; 0 WARN, 0 FAIL; "check_tests: OK" |
| `check_tests.py`, `src` tier | `--base 973a97d --src src -- tests/wcd_vae`; env wcd-gpu, `PYTHONPATH=src` (no `src` line changed; the tier is run because `tests/wcd_vae/test_divergences_v2.py` changed) | exit 0; 172 passed; 0 WARN, 0 FAIL; "check_tests: OK" |
| `check_tests.py`, scripts tier, the reviewer's base | `--base 6b4aefc --src scripts cluster -- tests/scvi tests/jhpce tests/prereg tests/stage tests/analysis tests/test_legacy_retired.py tests/test_gate_task_split.py` (the review's list); env scvi-api, `WCD_SRC=src` | exit 0; 418 passed, 18 skipped; 0 WARN, 0 FAIL; "check_tests: OK" |
| `check_tests.py`, `src` tier, the reviewer's base | `--base 6b4aefc --src src -- tests/wcd_vae`; env wcd-gpu, `PYTHONPATH=src` | exit 0; 172 passed; 10 WARN (7 for `tests/analysis`, 2 for `tests/prereg`, 1 for `tests/stage`), 0 FAIL: the review's warnings, see section 3.6 |
| `tests/stage`, whole folder | env scvi-api | at `973a97d`: 65 passed, 1 skipped; final: 66 passed, 1 skipped (the extra case is the new `status` case; the skipped one is the opt-in end-to-end test that needs the real fitter, scorer and R) |
| `tests/scvi`, whole folder | env scvi-api, without the data folder | at `973a97d`: 98 passed, 16 skipped; final: 98 passed, 16 skipped (counts unchanged: no test case was added) |
| `tests/scvi` with the prepared data folder | `PREPPED_DIR=.../prepped_scib`, read only | final: 113 passed, 1 skipped |
| `tests/wcd_vae/test_divergences_v2.py` | env wcd-gpu | at `973a97d`: 25 passed; final: 25 passed |
| `tests/test_legacy_retired.py`, `tests/jhpce`, `tests/prereg` | env scvi-api | final: 195 passed, 1 skipped |
| ruff 0.14.0, rules F and E9 | on every changed `.py` file; and on the whole repository | no finding in the 6 changed Python files; tree-wide 28 before, 27 after |
| `fl_lint` 1.0 | on the 8 changed files | 0 findings at any severity; 2 suppressions (as at `973a97d`) |
| syntax | `bash -n` on the two changed shell scripts; `python -m py_compile` on the six changed Python files | all pass |
| import closure | `probes/import_closure.py REPO c3fd92d HEAD` | fit entry points: `['scripts/run_stage.py']` changed; scorer: none; cluster helpers: none |

Planted bugs, all in clean clones of the final commit: 14 of 15 killed (the one that survives is the control `C02d_ledger_status_first`, an equivalent change by design); the lead's list: 11 of 11 killed.

## 5. What is not covered

1. The data-dependent tests ran only for `tests/scvi` (113 passed, 1 skipped, with the prepared data folder read-only); the opt-in end-to-end stage test (real fitter, scorer and R) was not run.
2. The identity proof was not repeated; the argument that it still holds is in section 3.2 (no fit, scoring, cluster-helper or `src/` file differs from `c3fd92d`; `run_stage.py` is a launcher that nothing imports).
3. No GPU and no cluster command were used.
4. A divergence record without a `git_sha` is still not attributable to a commit and is skipped by both checks, as a latent without one is. The real fitter always writes the field (`provenance()` raises if git fails), only the test stand-in does not. See question 1 below.
5. The planted bugs are my own text replacements plus the reviewer's and the lead's; each ran against a chosen test selection, not against a complete mutation analysis.
6. Statements of the review that I used without repeating them: the reviewer's census of manifest rows that use `adv_width` / `adv_lr` (`probes/reference_adv_rows.py`), the count of distinct and summed manifest rows in CF-03, and the check that the no-pair counter equals an independent count in four fits (`probes/g01_end_to_end.py`). None of my changes depends on them: the new assertions compare code with literals and with the code's own constants.
7. The `ponytail` code-simplicity skill that the user's rules ask to load before code changes is not in this agent's skill catalogue (`search_skills` with that prefix returns nothing). I applied the underlying rule through `check_tests.py` (fewest tests, no new test function, extend in place) and kept every change to the smallest set of lines that closes the finding.

## 6. Decisions and questions for the lead

1. Should a divergence record without a `git_sha` be refused by the start-up check instead of skipped? It would be stricter than the reviewer asked, would need the test stand-in to write a commit, and would also apply to latents without one.
2. `docs/PREREG.md` (SI-54 paragraph) and, if wanted, the description of the ledger's `git_sha` column should mention divergence records (section 3.2).
3. Whether to remove the other unused imports and variables outside the fit closure (22 findings) in a separate commit, and whether to commit a copy or the version of `fl_lint`.
4. The remaining parts of CF-05 (tag in the X1 bundle command, `latent_git_shas_match_head`) and CF-03 (README count) stay with their owners.
5. Which base should the final record of `check_tests.py` use? With `--base 973a97d`, or with each tier run on its own tests, both tiers are clean; with `--base 6b4aefc` the `src` tier alone prints the 10 warnings above (the scripts tier prints none). Changing the tool, or running it once with both test sets, is not part of my assignment.
6. Items the lead may want to log as notebook entries: the choice to extend the existing test in place instead of moving it (it keeps the lead's `mutants.tsv` valid); the choice to carry the divergence record's commit through the ledger column; the two side effects named in section 3.2.

## 7. Repeat the checks

1. `git clone <repository> work && cd work && git checkout -b cf-fixes 973a97d && git remote remove origin && git am <patches>/*.patch` (the clone must be a git repository: tests call `git rev-parse`).
2. Test tiers: scripts and cluster tests in `scvi-api` with `WCD_SRC=src`; `tests/wcd_vae` in `wcd-gpu` with `PYTHONPATH=src`; coverage runs: `python scripts/check_tests.py --base 973a97d --src scripts -- tests/scvi tests/jhpce tests/prereg tests/stage tests/test_legacy_retired.py` and `--src src -- tests/wcd_vae`.
3. Planted bugs: `python reviewer_probes_used/my_mutate.py CLONE out.json R01a_weighted_branch_no_stats R01b_mmdref_lambda_drops_stats R01c_record_needs_two_steps`; `python cf_mutate.py CLONE out.json` for the others (both scripts are in the evidence archive; `my_mutate.py` is a copy of the reviewer's; each needs a clean git clone, because it restores the files with `git checkout -- .`); `bash experiments/guard_followups_20261013/mutate.sh <sha> <clone>` for the lead's list.
4. Closure check: `python reviewer_probes_used/import_closure.py CLONE c3fd92d HEAD`.
5. `exec(open('build_cf_report.py').read())` (the script is in the evidence archive) in a Python session that has loaded the claim-gate skill's `cg_render` helper regenerates the numbers of this report from the saved logs; it expects the folders named in its docstring next to `logs/`.

## Appendix A. Output of `probes/import_closure.py REPO c3fd92d HEAD` at the final commit

The reviewer's script lists, for three entry points, the code files they import (static `ast` walk), which of those changed between `c3fd92d` and the final commit, a scan for code that loads other files by name, and every code file changed in the range. The last line lists changed files in the whole range (including the lead's earlier commits and my six); only `scripts/run_stage.py` of those is in a closure.

```
== fit (run_stage.py, fit_paper_config.py, scvi_adversarial_plan.py, run_cpu_baselines.py): 8 files in the static import closure
    scripts/build_paper_manifest.py, scripts/fit_outcome.py, scripts/fit_paper_config.py, scripts/host_info.py, scripts/prereg_rules.py, scripts/run_cpu_baselines.py, scripts/run_stage.py, scripts/scvi_adversarial_plan.py
    changed c3fd92d..HEAD and in closure: ['scripts/run_stage.py']
== scorer (score_scib_native.py): 2 files in the static import closure
    scripts/host_info.py, scripts/score_scib_native.py
    changed c3fd92d..HEAD and in closure: NONE
== cluster job helpers (prod_helpers.py): 1 files in the static import closure
    cluster/jhpce/prod_helpers.py
    changed c3fd92d..HEAD and in closure: NONE

dynamic-reference scan (lines naming another .py file, subprocess, importlib, exec) in closure files:
   scripts/fit_paper_config.py:288: import importlib.util
   scripts/fit_paper_config.py:289: spec = importlib.util.spec_from_file_location("wcd_data", os.path.join(os.environ["WCD_SRC"], "wcd_vae", "wcd", "data.py"))
   scripts/fit_paper_config.py:290: wcd_data = importlib.util.module_from_spec(spec); spec.loader.exec_module(wcd_data)
   scripts/prereg_rules.py:90: def metric_lists(path=os.path.join(HERE, "score_scib_native.py")):
   scripts/run_cpu_baselines.py:42: import importlib.metadata as md
   scripts/run_stage.py:87: FIT_SCRIPT = os.path.join(HERE, "fit_paper_config.py")
   scripts/run_stage.py:88: SCORE_SCRIPT = os.path.join(HERE, "score_scib_native.py")
   scripts/run_stage.py:89: CPU_SCRIPT = os.path.join(HERE, "run_cpu_baselines.py")
   scripts/run_stage.py:885: fitter = "run_cpu_baselines.py" if cpu else "fitter"
   scripts/score_scib_native.py:48: import importlib, os, json, socket, subprocess, sys, time, numpy as np, pandas as pd, scanpy as sc, scib
   scripts/score_scib_native.py:96: kbet_mod = importlib.import_module("scib.metrics.kbet")
   scripts/score_scib_native.py:115: traj = importlib.import_module("scib.metrics.trajectory")
   scripts/score_scib_native.py:148: import importlib.metadata as md
   scripts/scvi_adversarial_plan.py:44: import importlib.util
   scripts/scvi_adversarial_plan.py:111: spec = importlib.util.spec_from_file_location(modname, path)
   scripts/scvi_adversarial_plan.py:112: mod = importlib.util.module_from_spec(spec)
   scripts/scvi_adversarial_plan.py:129: spec = importlib.util.spec_from_file_location(modname, path)
   scripts/scvi_adversarial_plan.py:130: mod = importlib.util.module_from_spec(spec)

changed code files in the range: ['cluster/jhpce/local_rehearsal.sh', 'cluster/jhpce/prod_fit_job.sh', 'experiments/integration_checks_20261013/results/run_c3fd92d.sh', 'scripts/analysis_x1.py', 'scripts/decide_a2_a3.py', 'scripts/run_stage.py', 'tests/analysis/__init__.py', 'tests/analysis/test_analysis_x1.py', 'tests/prereg/test_cli_e2e.py', 'tests/scvi/test_baseline_guard.py', 'tests/scvi/test_runner_and_manifest_missing_arms.py', 'tests/stage/test_run_stage.py', 'tests/test_legacy_retired.py', 'tests/wcd_vae/test_divergences_v2.py']
```
