# Round 3, track "code": runner refusal rule (SI-85), power-search restart (SI-86), Sinkhorn cap test (CF-R02), docstring (DOC-R12)

| | |
|---|---|
| Fixer, date | RUNNER_ENGINEER, 2026-10-15 |
| Base | branch `final-review-fixes` at `5d06a27e39bd0337ac36b69f58703e26ab1e04a1` (git bundle SHA-256 `0c7de7d507cba8c2e7b60760d81f30cc07b92eb2009bf91fb55b7c21dbe0915a`); work in a private clone, nothing else touched |
| Result | 4 commits (head `130bfc3` in my clone), 7 files, +116 / -16 lines, all on the track's own file list. `git am` of the 4 patches onto a fresh clone of the bundle applies cleanly (exit 0) and gives the same file tree as my clone (`4e289467d750`). |
| Verdict | Items A to D adopted and done; no finding rebutted. Item E (identity proof): no changed line can change a fit or a score value. Section 10 has seven questions and notes for the lead; items 1 and 2 need a decision. |
| Blinding (SI-68, SI-78) | No X1 or X13 score, log or score-derived file was opened, listed, hashed or used. Every test input is synthetic or is code. Of the pilot folders (A1, A1_ext_r1_round1, A1_ext_r4_round1, A2A3) only the file names and the row count of each `failures.csv` (0 rows) were read. |

Commits in my clone (their ids change when the lead applies the patches; the patch series is the deliverable):

| commit | subject |
|---|---|
| `8db4475` | SI-85, CF-R03: the stage runner refuses a divergence record without a commit or from a dirty checkout |
| `c46d68e` | CF-R02: the Sinkhorn cap test checks the cap that each of the three solves receives |
| `cd8905b` | SI-86, DOC-R12: power_report restarts its root search at 0.01 when the search from 1.0 fails |
| `130bfc3` | DOC-R12: correct the docstring of power_without_sim1.py about when the search from 1.0 failed |

| file | added | removed |
|---|---|---|
| `experiments/final_review_amendments_20261013/power_without_sim1.py` | +5 | -2 |
| `scripts/prereg_rules.py` | +17 | -5 |
| `scripts/run_stage.py` | +11 | -0 |
| `tests/prereg/test_rules.py` | +18 | -0 |
| `tests/stage/fake_env.py` | +2 | -1 |
| `tests/stage/test_run_stage.py` | +54 | -2 |
| `tests/wcd_vae/test_divergences_v2.py` | +9 | -6 |

## 1. Result at a glance

| item | source | action | commit | evidence |
|---|---|---|---|---|
| A | SI-85 (user decision of 2026-10-15) = CF-R03 (finding of the code re-check), rule A | fixed | `8db4475` | the start-up check and the final check refuse a divergence record with no commit or from a dirty checkout; the re-checker's probe on my HEAD: S0 accepted, S1 to S3 refused at start-up, S4 accepted; 10 of 10 planted bugs killed |
| B | SI-86 (user decision of 2026-10-15) = DOC-R12, code part | fixed | `cd8905b` | 223 of 223 inputs that work today give bit-identical results; 7 of 7 that failed now return a finite value that agrees with an independent search; 5 of 5 planted bugs killed |
| C | CF-R02 | fixed | `c46d68e` | the two planted bugs that survived at 08f1e3c (the production caller passing a cap of 50,000; one symmetric solve getting a cap of 1,000) are now killed; 6 of 6 planted bugs of the CF-08 set killed; `tests/wcd_vae`: 172 passed, 66 warnings in 92.14s (0:01:32) |
| D | DOC-R12, docstring of `power_without_sim1.py` | fixed | `130bfc3` | numbers re-checked against probe 05; syntax tree of the rest of the file unchanged; the script writes the same results file (SHA-256 `0be883be772eb0aa1cc67f39675c12872d9fdcb1bccb9b2a88c1d11237e38510` before and after) |
| E | identity proof (commit c3fd92d) | checked, no code change | none | of the fit, scorer and cluster-helper closures only `scripts/run_stage.py` and `scripts/prereg_rules.py` contain changed files; neither change can alter a fit or score value (section 7) |

## 2. Words used

| term | meaning |
|---|---|
| latent | the saved embedding of all cells that a fit writes (`latents/<tag>.npz`) |
| divergence record | the file `status/<tag>.json` that a fit writes instead of a latent when its training loss becomes non-finite (status `diverged`); it exists for every arm, not only Sinkhorn. When a saved latent has a non-finite value, the runner writes a `nonfinite_latent` record itself |
| start-up check, final check | the runner's start-up check (function `preflight`: reads every existing output before any work) and its final check (function `gate`: re-reads every output at the end of a run and decides the exit status); `--report-only` runs both and starts no work |
| `--allow-dirty`, `git_dirty` | a switch that accepts an uncommitted checkout (tests only); the flag a fit records about its own checkout |
| SI-nn | standing instruction, a user decision recorded in `CONSTRAINTS.md`. SI-85: the runner refuses a divergence record that carries no commit or comes from an uncommitted checkout, as it refuses such a latent (rule A of the code re-check). SI-86: `power_report` restarts its root search at 0.01 when the search from 1.0 fails. SI-61: the Sinkhorn iteration cap is 500,000 |
| CF-Rnn, DOC-Rnn | findings of the re-check of 2026-10-14 (code track, documents track) |
| planted bug, killed, survived | a deliberate break of the code, made to test the tests: one text replacement in a scratch clone (for `BASE_5d06a27`, the file as it was before the change). Killed: a test fails. Survived: all tests pass. A control changes nothing and must survive |
| import closure, identity proof | the closure of a program is every code file it imports. The identity proof (commit c3fd92d) showed that every fit except Sinkhorn and every score is bit-identical to the pre-registered commit c88cce3; code outside the closures of the fit and score programs cannot change it |
| mde80 | the smallest effect that the design detects with 80 % power (`power_report`) |
| fake fitter | the stand-in program that the stage tests run instead of a real fit; it writes latents and records in the real format |
| probe | a short script that runs the code on a made-up case and prints what happens |
| clone | a full copy of the git repository in its own folder |
| bit-identical | equal in every binary digit of the stored number, not only equal after rounding |
| syntax tree | the program text parsed into a tree; comparing two trees ignores comments and spacing |
| parametrised test | one test function that runs once for each row of a list of inputs (here each row is a case) |
| spy | a stand-in for a function that records the arguments it is called with, then calls the real function |
| NaN | "not a number": the result of a calculation that has no valid value |
| non-centrality | in the power calculation (non-central t distribution), the effect divided by its standard error |
| survival function | the upper-tail probability of a distribution; computed directly, it has no NaN where the cumulative form has |
| S0 to S5 | scenarios of the code re-check's probe: S0 a divergence record as the fake fitter writes it; S1 the same with `git_dirty` true; S2 with `git_dirty` null; S3 without `git_sha`; S4 the runner's own `nonfinite_latent` record; S5 (mine) that record without `git_sha` |

## 3. Item A: SI-85 = CF-R03, the runner refuses a divergence record without a commit or from a dirty checkout

**Checked at 5d06a27.**

1. `scripts/run_stage.py:422-423`: a latent whose `git_dirty` is not false is refused unless `--allow-dirty`. `:429-440` (`Outputs.status`): a record is checked for its schema and its manifest row only; no commit, no clean flag. `Outputs.status` is the one function that reads records: the start-up check (`preflight` calls `state`), the final check (`file_states` calls `state`), the re-check after a claim (`recheck`) and the fit-done check (`fit_done`) all go through it, so one change covers all of them. `write_failures` lists every file in `status/` as a table and is not an acceptance check; it is unchanged.
2. `tests/stage/fake_env.py:89-92`: the fake fitter's divergence record had no `git_sha` and no `git_dirty`. The real fitter's record has both (`scripts/fit_paper_config.py:514-520` writes `**provenance()`, defined at `:264-270`); the live check below uses the real writer.
3. No test covered `:422-423`: the re-checker's switch-off survived 108 stage and cluster tests. Planted bug A1 below (the same switch-off) is now killed.
4. Finding and decision confirmed: adopted. Rule A as defined in section 9.1 of the code re-check: the clean flag is asked of `diverged` records only, a commit of every record. Rule N (clean flag of every record) would refuse the runner's own `nonfinite_latent` records (S4).

**Change** (`scripts/run_stage.py`, lines 447 onward; the same switch and message style as the refusal of a latent):

```python
        if rec["status"] == "diverged" and rec.get("git_dirty") is not False and not self.allow_dirty:
            raise InvalidOutput(f"{p}: fitted from a dirty or unknown checkout (git_dirty={rec.get('git_dirty')})")
        if not rec.get("git_sha"):
            raise InvalidOutput(f"{p}: the record names no git_sha")
```

Also: 4 lines in the module docstring and 3 comment lines saying what the rule is; `tests/stage/fake_env.py`: the divergence record now carries `git_sha="fake", git_dirty=False` as the real fitter's `provenance()` does (one added argument); two comments in `test_run_stage.py` that said the fake's record carries no commit are corrected.

**The test** (`tests/stage/test_run_stage.py`, `test_outputs_from_a_dirty_or_unknown_commit_are_refused`, 8 cases of one parametrised test). A stage of three rows is run once with `--allow-dirty`; one output is then edited and the runner is called as a function inside the test (not as a separate program) with `--report-only` and without `--allow-dirty` (every other stage test passes it). Report-only runs the start-up check and the final check but not the check that the runner's own checkout is clean, so the result does not depend on the state of the repository the test is run in. Records of two arms are used, one that is not Sinkhorn and the Sinkhorn arm.

| case | edit | expected |
|---|---|---|
| `latent-git_dirty-true` | the latent of a finished row: `git_dirty` true | refused at start-up (the refusal at `:422-423`, first test) |
| `S1-record-git_dirty-true` | both divergence records: `git_dirty` true | refused |
| `S2-record-git_dirty-null` | `git_dirty` null | refused |
| `S3-record-without-git_sha` | `git_sha` removed | refused |
| `S0-record-as-the-fake-fitter-writes-it` | none | accepted |
| `S4-nonfinite_latent-record-of-the-runner` | none (a commit, no clean flag) | accepted |
| `S5-nonfinite_latent-record-without-git_sha` (added) | the runner's own record, `git_sha` removed | refused: the commit is asked of every record |
| `S1-allow-dirty-accepted` (added) | `git_dirty` true, run with `--allow-dirty` | accepted: the dirty part follows the same switch |

**Evidence, planted bugs** (each is one text replacement, or for `BASE_5d06a27` the restored old file, in a scratch clone of the final HEAD; selection: this test only). Control without change: `8 passed, 66 deselected`. Before the change (run_stage.py of 5d06a27) the test fails in: stage-test[S1-record-git_dirty-true], stage-test[S2-record-git_dirty-null], stage-test[S3-record-without-git_sha], stage-test[S5-nonfinite_latent-record-without-git_sha].

| id | change planted | verdict | tests | failing test(s) |
|---|---|---|---|---|
| `BASE_5d06a27` | run_stage.py as it was before this change | **KILLED** | 4 failed, 4 passed, 66 deselected | stage-test[S1-record-git_dirty-true], stage-test[S2-record-git_dirty-null], stage-test[S3-record-without-git_sha], stage-test[S5-nonfinite_latent-record-without-git_sha] |
| `A1_latent_refusal_off` | existing refusal of a dirty latent (run_stage.py:422-423 at 5d06a27) switched off | **KILLED** | 1 failed, 7 passed, 66 deselected | stage-test[latent-git_dirty-true] |
| `A2_record_dirty_refusal_off` | new dirty-checkout refusal for records switched off | **KILLED** | 2 failed, 6 passed, 66 deselected | stage-test[S1-record-git_dirty-true], stage-test[S2-record-git_dirty-null] |
| `A3_record_no_commit_refusal_off` | new no-commit refusal for records switched off | **KILLED** | 2 failed, 6 passed, 66 deselected | stage-test[S3-record-without-git_sha], stage-test[S5-nonfinite_latent-record-without-git_sha] |
| `A4_dirty_flag_asked_of_every_record` | clean flag asked of every record (rule N of the re-check) | **KILLED** | 2 failed, 6 passed, 66 deselected | stage-test[S4-nonfinite_latent-record-of-the-runner], stage-test[S5-nonfinite_latent-record-without-git_sha] |
| `A5_null_flag_accepted` | git_dirty null or absent accepted | **KILLED** | 1 failed, 7 passed, 66 deselected | stage-test[S2-record-git_dirty-null] |
| `A6_allow_dirty_ignored_for_records` | --allow-dirty no longer accepts a dirty record | **KILLED** | 1 failed, 7 passed, 66 deselected | stage-test[S1-allow-dirty-accepted] |
| `A7_only_sinkhorn_records` | refusal only for the Sinkhorn arm | **KILLED** | 2 failed, 6 passed, 66 deselected | stage-test[S1-record-git_dirty-true], stage-test[S2-record-git_dirty-null] |
| `A8_only_non_sinkhorn_records` | refusal skips the Sinkhorn arm | **KILLED** | 2 failed, 6 passed, 66 deselected | stage-test[S1-record-git_dirty-true], stage-test[S2-record-git_dirty-null] |
| `A9_commit_asked_of_diverged_only` | no-commit refusal only for divergence records | **KILLED** | 1 failed, 7 passed, 66 deselected | stage-test[S5-nonfinite_latent-record-without-git_sha] |
| `EQ_A_status_read_with_get` | control: status read with .get (no change in meaning) | **SURVIVED** | 8 passed, 66 deselected |  |

Formal check with the fail-loud helper `fl_mutation_check` (known-good input accepted, every bad input rejected for the expected test, none for a reason in the test setup):

| verifier | ok | good input passes | bad inputs caught | missed | wrong reason |
|---|---|---|---|---|---|
| `si85_stage_test` | True | True | 10 | 0 | 0 |
| `si86_power_test` | True | True | 5 | 0 | 0 |
| `cf_r02_cap_test` | True | True | 6 | 0 | 0 |

**Evidence, the re-checker's probe** (`r03_rule_validation_probe.py`, unchanged, in its internal mode on a clean clone of my HEAD: launching mode, no `--allow-dirty`): matches the required outcome: yes.

| scenario | outcome | exit status |
|---|---|---|
| S0 divergence record as the fake fitter writes it | accepted | 0 |
| S1 divergence record, git_dirty true | refused at start-up | 4 |
| S2 divergence record, git_dirty null | refused at start-up | 4 |
| S3 divergence record, no git_sha | refused at start-up | 4 |
| S4 runner-made nonfinite_latent record (git_sha, no git_dirty) | accepted | 0 |

**Evidence, the final check** (the functions of the final check, `file_states` and `gate`, called inside one Python process, no `--allow-dirty`, on a record edited after a normal run):

```
record edited: git_dirty true  state of v1: invalid           final check ok: False not_final: {'v1': 'invalid'}
     <tmp>/out/status/v1.json: fitted from a dirty or unknown checkout (git_dirty=True)
record edited: no git_sha      state of v1: invalid           final check ok: False not_final: {'v1': 'invalid'}
     <tmp>/out/status/v1.json: the record names no git_sha
record edited: unedited        state of v1: diverged          final check ok: True  not_final: {}
```

**Evidence, the real record writer** (`fit_paper_config.record_divergence`, with the real `provenance()`; no fit was run, the loss error that triggers the record is raised by hand):

```
real record written at a clean checkout: keys ['detail', 'epoch', 'error', 'git_dirty', 'git_sha', 'gpu', 'host', 'recorded_at', 'row', 'scvi', 'status', 'step', 'tag', 'terms', 'torch'] | git_sha 130bfc36f3 | git_dirty False
Outputs.status (no --allow-dirty), clean checkout: accepted, status diverged
real record written at a dirty checkout: git_dirty True
Outputs.status (no --allow-dirty), dirty checkout: refused: <tmp>/out/status/r2.json: fitted from a dirty or unknown checkout (git_dirty=True)
with --allow-dirty: accepted, status diverged
```

**Notes.** (a) The real fitter always writes both fields, so only a hand-made or copied file is affected, as the re-check said. (b) The pilot folders hold no divergence record (0 failure rows in each), so the rule could not be tried on a real pilot record; the live check above is the substitute. (c) `fl_lint`: fl_lint 1.0 (26 rules), fail_on=low: base 5d06a27 21 files, 0 findings; final 21 files, 0 findings, 0 suppressed. No `try`/`except` was added, no fallback; each refusal raises `InvalidOutput` with the file name and the reason.

## 4. Item B: SI-86 = DOC-R12 (code part), `power_report` restarts its search at 0.01

**Checked at 5d06a27.** `scripts/prereg_rules.py:704-708`: the root search for mde80 (a numerical search for the effect at which the power is 0.8) starts at 1.0. `power_report` is called only by `scripts/freeze_a1_grid.py:62`. Probe 05 of the documents re-check, re-run on my HEAD, prints the re-checker's own output (identical: yes): with scipy 1.17.1 `nct.cdf` is NaN at isolated large non-centralities, first at 42.15 for 5 degrees of freedom and 57.24 for 3; the start-1.0 search raises for 7 of 230 tested (standard error, degrees of freedom) pairs. Finding confirmed: adopted.

**Change.** The re-checker's `08_power_report_fallback_fix.diff` applied (git applied it cleanly), as the helper `_mde80` (`prereg_rules.py:674`) called from `power_report` (`:720`), with one difference: the helper keeps the scipy error and chains it into the `PreregError` that it raises when both starts fail, instead of `except ValueError: continue`, which the fail-loud lint rule FL102 flags (an except block that only continues). Results and exception types are those of the suggested change; the lint needs no suppression.

**The test** (`tests/prereg/test_rules.py`, `test_mde80_restarts_at_0_01_only_when_the_search_from_1_fails`). The lead asked for one parametrised case in the existing power test. That test cannot take a standard error: `power_report` computes it from the data, and the synthetic A1 world cannot reach one on which the old search fails (the standard error grows in proportion to the noise: 0.000508 at noise 0.002 and 0.001271 at 0.005, so about 0.02 needs a noise of 0.08 to 0.1, and every noise level from 0.01 up stops at "R1 record is not frozen"):

```
noise sd=0.002: standard error 0.000508; old search from 1.0 works
noise sd=0.005: standard error 0.001271; old search from 1.0 works
noise sd=0.01: R1 record is not frozen (power_report is not reached)
noise sd=0.02: R1 record is not frozen (power_report is not reached)
noise sd=0.03: R1 record is not frozen (power_report is not reached)
noise sd=0.05: R1 record is not frozen (power_report is not reached)
noise sd=0.08: R1 record is not frozen (power_report is not reached)
noise sd=0.1: R1 record is not frozen (power_report is not reached)
```

So the new case calls `_mde80` with the real failing input, the standard error of the pilot without sim1 (0.02355132223101683, 5 degrees of freedom): the premise (power at 1.0 is NaN) is asserted, the answer equals the value found earlier by a separate script that starts at 0.01 (`power_without_sim1.json`), the power at the answer is checked with the survival function (no NaN: an independent formula), and a design of one dataset (0 degrees of freedom) raises. It would fail loudly if a later scipy removed the NaN (the premise line).

**Evidence.** Inputs that fail today (scipy 1.17.1), all eight of the re-checker's list; `fixed` is `_mde80`, `survival-function search` an independent search with `nct.sf`:

| standard error | df | today | fixed | survival-function search | difference |
|---|---|---|---|---|---|
| 0.0175 | 5 | ValueError (NaN) | 0.0508301827 | 0.0508301827 | 0.0e+00 |
| 0.0185 | 5 | ValueError (NaN) | 0.0537347645 | 0.0537347645 | 0.0e+00 |
| 0.0215 | 5 | ValueError (NaN) | 0.0624485101 | 0.0624485101 | 6.9e-18 |
| 0.022 | 5 | ValueError (NaN) | 0.0639008011 | 0.0639008011 | 0.0e+00 |
| 0.0225 | 5 | ValueError (NaN) | 0.0653530920 | 0.0653530920 | 0.0e+00 |
| 0.0235 | 5 | ValueError (NaN) | 0.0682576739 | 0.0682576739 | 1.4e-17 |
| 0.02355132223101683 | 5 | ValueError (NaN) | 0.0684067435 | 0.0684067435 | 0.0e+00 |
| 0.0165 | 3 | ValueError (NaN) | 0.0544409110 | 0.0544409110 | 0.0e+00 |

- Grid of standard errors 0.003 to 0.06 (step 0.0005), 3 and 5 degrees of freedom, 230 points: the old search works at 223 and the fixed result is bit-identical in 223 (differs in 0); the old search fails at 7 and the fixed code returns a finite number at 7.
- Value check of all 230 points against the survival-function search: largest difference 5.67e-13; points above 1e-9: 0. So the results that worked before are also right, not only unchanged.
- Power values already written into committed files, recomputed from the inputs those files hold (`se`, `df` of each design, `sigma`, `tau`); the only committed record with a power report is `docs/prereg/r1_record.json`; `power_report` has one code caller:

| record | se recomputed from stored sigma, tau equal | mde80 now equals stored (bit-identical) | power at delta equals stored |
|---|---|---|---|
| `docs/prereg/r1_record.json` power.designs.families | True | True | True |
| `docs/prereg/r1_record.json` power.designs.real_tasks | True | True | True |

  `experiments/final_review_amendments_20261013/results/power_without_sim1.json` holds a separate calculation that starts at 0.01 (not changed by SI-86); `_mde80` on its standard errors:

| design | stored mde80 | `_mde80` now | difference |
|---|---|---|---|
| all_blocks / real_tasks | 0.1306631837466057 | 0.13066318374660563 | 8.3e-17 |
| all_blocks / families | 0.18178443022106544 | 0.1817844302210654 | 2.8e-17 |
| without_sim1 / real_tasks | 0.0684067435003012 | 0.0684067435003012 | 0.0e+00 |
| without_sim1 / families | 0.09517050276846684 | 0.09517050276819047 | 2.8e-13 |

- `tests/prereg` alone and `tests/stage` alone on the final tree: 132 passed, 1 skipped in 37.01s; 74 passed, 1 skipped in 176.41s (0:02:56).

**Evidence, planted bugs** (selection: the power tests and `test_decision_records_are_the_same`; control without change: `6 passed, 52 deselected`). `B3` is the one-line alternative named by the documents re-check; the existing exact-equality test already kills it, as that re-check found.

| id | change planted | verdict | tests | failing test(s) |
|---|---|---|---|---|
| `B1_no_fallback` | no restart at 0.01 | **KILLED** | 1 failed, 5 passed, 52 deselected | mde80-test |
| `B2_restart_at_1` | restart starts at 1.0 again | **KILLED** | 1 failed, 5 passed, 52 deselected | mde80-test |
| `B3_always_start_small` | always start at 0.01 (the one-line alternative) | **KILLED** | 1 failed, 5 passed, 52 deselected | records_same[r1.json] |
| `B4_failure_becomes_nan` | a failed search returns NaN | **KILLED** | 1 failed, 5 passed, 52 deselected | mde80-test |
| `B5_second_failure_swallowed` | failure of both starts returns NaN | **KILLED** | 1 failed, 5 passed, 52 deselected | mde80-test |
| `EQ_B_start_tuple_as_list` | control: tuple written as a list | **SURVIVED** | 6 passed, 52 deselected |  |

## 5. Item C: CF-R02, the Sinkhorn cap test checks the cap that each of the three solves receives

**Checked at 5d06a27.** `tests/wcd_vae/test_divergences_v2.py:131-142`: the test spied on `alignment._ot_eps` for its dtype only and pinned the default of `sinkhorn_divergence_weighted` with `inspect.signature` (`:142`); `src/wcd_vae/wcd/alignment.py:224-233` shows that one `sinkhorn_batch_others` call makes three solves (one `_ot_eps`, two `_ot_eps_sym`), each taking `max_iter` as its last argument. The comment at `:140` cited SI-59 (retired). Finding confirmed: adopted.

**Change.** The spy on `_ot_eps` (a stand-in that records the arguments it is called with, then calls the real function) also records its last argument, a second spy does the same for `_ot_eps_sym`, and the test asserts `caps == [alignment.SINKHORN_MAX_ITER] * 3`; this replaces the `inspect.signature` assertion (and the then unused `import inspect`). The comment now cites SI-61 and says what the assertion catches.

**Evidence.** The re-checker's `mut_recheck.py`, unchanged, on my HEAD, ids M08a to M08f (its selection for M08c and M08d is all of `tests/wcd_vae` with `-x`, hence "1 failed, 77 passed" at the first failure):

| id | change planted | verdict | tests | failing test(s) |
|---|---|---|---|---|
| `M08a_default_literal` | default max_iter of sinkhorn_divergence_weighted set to 50,000 | **KILLED** | 1 failed, 5 passed | test_sinkhorn_matches_converged_reference[_standardised] |
| `M08b_constant` | SINKHORN_MAX_ITER set to 50,000 | **KILLED** | 1 failed, 5 passed | test_sinkhorn_matches_converged_reference[_standardised] |
| `M08c_production_caller_passes_cap` | sinkhorn_batch_others passes max_iter=50000 (survived at 08f1e3c) | **KILLED** | 1 failed, 77 passed | test_sinkhorn_matches_converged_reference[_standardised] |
| `M08d_symmetric_terms_smaller_cap` | one symmetric solve gets 1,000 (survived at 08f1e3c) | **KILLED** | 1 failed, 77 passed | test_sinkhorn_matches_converged_reference[_standardised] |
| `M08e_huge_row_ordinary_latent` | huge-latent test row fed an ordinary latent | **KILLED** | 1 failed, 8 passed | test_sinkhorn_refuses_bad_p_and_unconverged_solve[_huge_latent_solve-RuntimeError-marginal error inf .* after 1 iterations] |
| `M08f_loop_stops_on_inf_cross` | cross solve stops on an infinite error | **KILLED** | 1 failed, 8 passed | test_sinkhorn_refuses_bad_p_and_unconverged_solve[_huge_latent_solve-RuntimeError-marginal error inf .* after 1 iterations] |

Formal check with `fl_mutation_check`: see the table in section 3 (row `cf_r02_cap_test`). `tests/wcd_vae`: 172 passed, 66 warnings in 92.14s (0:01:32).

## 6. Item D: DOC-R12, docstring of `experiments/final_review_amendments_20261013/power_without_sim1.py`

**Checked at 5d06a27.** The docstring said that scipy's non-central t returns NaN "above a non-centrality of about 38", reached by the start of 1.0 "when the standard error is below about 0.026". Probe 05 (section 4) shows both are wrong: NaN appears at isolated values, first at 42.15 for 5 degrees of freedom and at 57.24 for 3, and 7 of 230 tested pairs fail.

**Change.** The two sentences are replaced by what probe 05 shows, with SI-86 named and explained in the same sentence. Docstring only: the syntax tree of everything after the docstring is identical to the previous version. The script's output did not change: stdout identical (clone name normalised): True | results JSON sha256 identical: True 0be883be772eb0aa.

## 7. Item E: identity proof, import closure and changed lines

`probes/import_closure.py` of the first code review, unchanged, `python probes/import_closure.py <clone> c3fd92d <HEAD>` (first lines of its output; full output in `final_run/import_closure_c3fd92d_vs_HEAD.txt`):

```
== fit (run_stage.py, fit_paper_config.py, scvi_adversarial_plan.py, run_cpu_baselines.py): 8 files in the static import closure
    scripts/build_paper_manifest.py, scripts/fit_outcome.py, scripts/fit_paper_config.py, scripts/host_info.py, scripts/prereg_rules.py, scripts/run_cpu_baselines.py, scripts/run_stage.py, scripts/scvi_adversarial_plan.py
    changed c3fd92d..130bfc3 and in closure: ['scripts/prereg_rules.py', 'scripts/run_stage.py']
== scorer (score_scib_native.py): 2 files in the static import closure
    scripts/host_info.py, scripts/score_scib_native.py
    changed c3fd92d..130bfc3 and in closure: NONE
== cluster job helpers (prod_helpers.py): 1 files in the static import closure
    cluster/jhpce/prod_helpers.py
    changed c3fd92d..130bfc3 and in closure: NONE

```

Closure of each program separately (same functions, `closures_by_entry.py`), and the files changed in earlier rounds (c3fd92d to 5d06a27) and in this round:

```
== fit_paper_config.py (the fitter; entry point of the identity proof): 3 files: scripts/fit_outcome.py, scripts/fit_paper_config.py, scripts/scvi_adversarial_plan.py
     changed in earlier rounds c3fd92d..5d06a27 and in this closure: NONE
     changed in this round 5d06a27..130bfc3 and in this closure: NONE
== scvi_adversarial_plan.py (the training plan; entry point of the identity proof): 2 files: scripts/fit_outcome.py, scripts/scvi_adversarial_plan.py
     changed in earlier rounds c3fd92d..5d06a27 and in this closure: NONE
     changed in this round 5d06a27..130bfc3 and in this closure: NONE
== run_cpu_baselines.py (X13 CPU runner): 3 files: scripts/fit_outcome.py, scripts/host_info.py, scripts/run_cpu_baselines.py
     changed in earlier rounds c3fd92d..5d06a27 and in this closure: NONE
     changed in this round 5d06a27..130bfc3 and in this closure: NONE
== run_stage.py (the stage runner): 8 files: scripts/build_paper_manifest.py, scripts/fit_outcome.py, scripts/fit_paper_config.py, scripts/host_info.py, scripts/prereg_rules.py, scripts/run_cpu_baselines.py, scripts/run_stage.py, scripts/scvi_adversarial_plan.py
     changed in earlier rounds c3fd92d..5d06a27 and in this closure: ['scripts/run_stage.py']
     changed in this round 5d06a27..130bfc3 and in this closure: ['scripts/prereg_rules.py', 'scripts/run_stage.py']
== score_scib_native.py (the scorer; entry point of the identity proof): 2 files: scripts/host_info.py, scripts/score_scib_native.py
     changed in earlier rounds c3fd92d..5d06a27 and in this closure: NONE
     changed in this round 5d06a27..130bfc3 and in this closure: NONE
== prod_helpers.py (cluster job helpers): 1 files: cluster/jhpce/prod_helpers.py
     changed in earlier rounds c3fd92d..5d06a27 and in this closure: NONE
     changed in this round 5d06a27..130bfc3 and in this closure: NONE
```

**Is `prereg_rules.py` in a closure? Yes, in the closure of `run_stage.py` (the first review's combined "fit" closure), and no in the closures of `fit_paper_config.py`, `scvi_adversarial_plan.py`, `run_cpu_baselines.py` (the programs that compute a fit) and `score_scib_native.py` (the scorer).** `run_stage.py` is run as a program (only two test files import it); it takes only `equivalent_commits`, `refuse_uncovered`, `read_scores`, `read_failures`, `PreregError`, `bio_metrics_for` and `BATCH_METRICS` from `prereg_rules`, none of them from the power code.

A direct `git diff` from c3fd92d to HEAD over `src/`, `fit_paper_config.py`, `scvi_adversarial_plan.py`, `score_scib_native.py`, `fit_outcome.py`, `host_info.py`, `run_cpu_baselines.py`, `build_paper_manifest.py`, `prod_helpers.py`, `pyproject.toml`, `environment.yaml` and `conda-lock.yml` is empty (`final_run/closure_statement_facts.txt`).

| changed lines of this round in a closure file | can it change a fit or a score value? |
|---|---|
| `run_stage.py`, module docstring, 4 lines (SI-85 sentence) | No: text. The argument parser reads only the first docstring line, which is untouched |
| `run_stage.py`, 3 comment lines in `Outputs.status` | No: comments |
| `run_stage.py`, 4 code lines in `Outputs.status` (two `if`, two `raise`) | No: they can only raise `InvalidOutput`, which refuses a run (start-up) or marks a row invalid (final check). Nothing here builds a fit or score command, an environment variable or a thread setting, and when nothing is raised the function returns the same record as before |
| `prereg_rules.py`, new helper `_mde80` (16 lines) and one changed line in `power_report` (5 lines removed, 1 added) | No: the power calculation, called only from `scripts/freeze_a1_grid.py:62` (the R1 record); no fit, no score and not the runner |

The other changed files are tests and one docstring (not code that the fit, score or cluster programs load).

## 8. Acceptance results (all on the final head `130bfc3`; CPU only, thread caps set, one process per run)

| check | command and environment | result |
|---|---|---|
| 1a | `python -m pytest -q tests/stage tests/prereg`, env scvi-api, `WCD_SRC=src` | 206 passed, 2 skipped in 200.72s (0:03:20); exit 0 |
| 1b | `python -m pytest -q tests/wcd_vae`, env wcd-gpu, `PYTHONPATH=src` | 172 passed, 66 warnings in 92.14s (0:01:32); exit 0 (172 passed in the lead's run of this tier) |
| 2a | `covvenv/bin/python scripts/check_tests.py --base 5d06a27 --src scripts -- tests/stage tests/prereg` (venv on scvi-api: pytest-cov 7.1.0 coverage 7.16.2 pytest 9.1.1, ruff 0.14.0), own clone | 206 passed, 2 skipped in 329.15s (0:05:29); `check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base 5d06a27)`; FAIL or WARN lines: 0; exit 0 |
| 2b | `covvenv/bin/python scripts/check_tests.py --base 5d06a27 --src src -- tests/wcd_vae` (venv on wcd-gpu: pytest-cov 7.1.0 coverage 7.16.2 pytest 9.1.1), own clone | 172 passed, 65 warnings in 111.46s (0:01:51); `check_tests: OK (every changed --src line and branch is tested; no redundant new tests; base 5d06a27)`; FAIL or WARN lines: 0; exit 0 |
| 3 | probes of items A, B, C and the switch-off check | sections 3 to 5 |
| 4 | import closure and changed lines | section 7 |
| patches | `git format-patch 5d06a27..HEAD`; `git am` on a fresh clone of the bundle (`scripts/am_check.sh`, log `final_run/am_check.txt`) | applies cleanly (exit 0); `git diff --name-only 5d06a27 HEAD` lists only the 7 files of the table above; tree of the fresh clone equals my clone's tree (`4e289467d750997f7512945a0918254dc6096619`) |

Reconciliation with the lead's earlier run: the rest of the scripts tier (`tests/analysis tests/scvi tests/jhpce tests/test_legacy_retired.py`) on this head: 224 passed, 16 skipped, 243 warnings in 435.89s (0:07:15); exit 0. Check 1a and this run together give 430 passed and 18 skipped for the whole scripts tier, against 421 passed and 18 skipped in the lead's run (`421 passed, 18 skipped, 243 warnings in 817.31s (0:13:37)`): the difference of 9 is the 8 cases of the new stage test and the 1 new power test.

Earlier head: all the checks of this section were first run on head `ee6a679` (logs kept in `earlier_head_ee6a679/`). Then the wording of comments, docstrings and commit messages was made plainer, because the user's plain-language rule names "gate" as a meaningless word, and everything was run again on the head named in this report. The two heads differ in 5 files, only in comments and docstrings: the syntax tree without docstrings is identical in all 5 Python files compared (`final_run/wording_only_check.txt`).

Lint and style: fl_lint 1.0 (26 rules), fail_on=low: base 5d06a27 21 files, 0 findings; final 21 files, 0 findings, 0 suppressed (the lead's rule: zero findings at medium or above). `ruff` (rules F and E9, ruff 0.14.0) on the owned files, base against final, and over the whole repository:

```
== f_d_base (5d06a27): ruff 0.14.0, --isolated --select F,E9, owned files
tests/prereg/test_manifest_v3.py:19:8: F401 [*] `pytest` imported but unused
tests/prereg/test_manifest_v3.py:21:8: F401 [*] `synth` imported but unused
Found 2 errors.
[*] 2 fixable with the `--fix` option.
ruff exit 1
whole repository: 82 findings
== rv (130bfc3): ruff 0.14.0, --isolated --select F,E9, owned files
tests/prereg/test_manifest_v3.py:19:8: F401 [*] `pytest` imported but unused
tests/prereg/test_manifest_v3.py:21:8: F401 [*] `synth` imported but unused
Found 2 errors.
[*] 2 fixable with the `--fix` option.
ruff exit 1
whole repository: 82 findings
```

The two findings are in `tests/prereg/test_manifest_v3.py`, unchanged by this round and left alone as decided (NB-20261015-12).

## 9. Standing rules

Standing rules: SI-85 checked by test and probe; SI-86 checked by test and probes; SI-61 (cap 500,000) still pinned by the cap test; SI-68 and SI-78 (blinding) checked by listing what was read (above); SI-52 (non-Sinkhorn fits and scores bit-identical, diff shown before the tag) checked in section 7 by closure and diff; the fewest-tests rule checked by `check_tests.py` (no FAIL, no WARN; the two `# test-reason:` lines were added because the src-tier run, which does not run tests of the other tier, printed two WARN lines before them: kept in `earlier_tree_66e5641/logs/check_tests_src_tier_first_run.log`); the smallest-change rule: the `ponytail` skill is not in my catalogue, so the task's text of the rule was applied (no new option or flag, no new dependency; the power change is one helper, the runner change 4 code lines); the plain-language rule applied to every added comment, docstring and message. Not applicable: SI-87 (record file names), SI-83 and SI-84 (analysis track).

## 10. Questions and notes for the lead

1. **SI-85 says "Sinkhorn divergence record"; the rule that was implemented has no arm condition.** In the code a divergence record exists for every arm (`fit_outcome.py` header: the training loss became non-finite), the latent rule that SI-85 compares it with has no arm condition, and rule A of the code re-check, which SI-85 names, asks the same of every arm. I implemented rule A. Planted bugs A7 and A8 show the test pins this (a Sinkhorn-only version fails it). If the user meant the Sinkhorn arm only, the change is `and row["arm"] == "sinkhorn"` in the two conditions at `scripts/run_stage.py:447` and below, and the test's expectations for its non-Sinkhorn row (v1) would turn to accepted; suggest saying so at the diff review.
2. **A latent without `git_sha` is still accepted** (re-check row P2-13; SI-85 covers records only). For a non-finite latent without a commit the runner would write a `nonfinite_latent` record with `git_sha` null, which the new rule refuses on its next read (loud: a refusal at start-up or an infrastructure error naming the file). Real fitters always write the commit, so only a hand-made latent is affected. If wanted: add `if not cfg.get("git_sha"): raise InvalidOutput(...)` after the latent's clean-flag refusal (`scripts/run_stage.py:422-423` at 5d06a27) and one case to the new stage test.
3. **Text for the documents track** (not my file): `docs/PREREG.md:372-374` (5d06a27) says the run_stage preflight refuses Sinkhorn output made at c88cce3. Proposed addition after "in report-only runs too.": "The run_stage preflight also refuses a divergence record that names no commit or comes from a checkout with uncommitted changes, as it refuses such a latent (SI-85, the user's decision of 2026-10-15); a nonfinite_latent record is not asked for the clean flag."
4. **Ledger housekeeping** (`CONSTRAINTS.md` is the lead's): SI-85's `files:` line lists `scripts/run_stage.py, tests/stage/test_run_stage.py`; the change also touches `tests/stage/fake_env.py`.
5. **Additions beyond the lead's list** in the new stage test: case S5 and the `--allow-dirty` case. They pin two statements of the decision (any record needs a commit; the dirty part follows the same switch); planted bugs A9 and A6 survive without them. Delete the two parameter lines to drop them.
6. **Existing X1 outputs.** I did not look at them (blinding). If any existing X1 divergence record, or latent, came from a dirty checkout, the stage's start-up check will refuse it when the stage resumes, naming the file and the reason and printing no score value; real fits made from a clean checkout record `git_dirty` false.
7. **The new power test depends on scipy's NaN behaviour** by design: its first assertion (power at 1.0 is NaN) fails loudly if a later scipy removes the NaN, so that the fallback is never left unexercised without notice.

## 11. Proposed notebook entries (notebook/ is outside my files; text for the lead to enter)

1. DECISION-implementation NOTE, refs SI-85, NB-20261015-18: "Runner refuses divergence records without a commit or from a dirty checkout. `Outputs.status` (scripts/run_stage.py) refuses a record of status diverged whose git_dirty is not false unless --allow-dirty, and any record without git_sha; the nonfinite_latent records of the runner and of run_cpu_baselines.py stay accepted. One parametrised stage test (8 cases) runs without --allow-dirty, inside the test process with --report-only; planted bugs 10 of 10 killed; the re-checker's rule probe gives S0 accepted, S1-S3 refused at start-up, S4 accepted. The rule has no arm condition (see question 1)."
2. NOTE, refs SI-86, NB-20261015-19, DOC-R12: "power_report restarts its root search at 0.01 when the search from 1.0 fails (prereg_rules._mde80). Of 230 tested standard errors and degrees of freedom the search from 1.0 worked at 223 (bit-identical results) and failed at 7 (now finite, equal to an independent search with the survival function to 5.67e-13 at most). The committed power report of docs/prereg/r1_record.json recomputes bit-identically."
3. RESULT, refs CF-R02: "The Sinkhorn cap test now records the cap that all three solves of one call receive; the two planted bugs that survived at 08f1e3c are killed."
4. NOTE, refs DOC-R12: "Docstring of power_without_sim1.py corrected (probe 05 numbers); script output unchanged."
5. NOTE (finding): "Two comments in tests/stage said the fake fitter's divergence record carries no commit; corrected when the fake got git_sha and git_dirty. A latent without git_sha is still accepted (question 2)."

## 12. Limits and what was not done

1. The opt-in end-to-end stage test with the real fitter, scorer and R (`tests/stage/test_run_stage_e2e.py`, skipped in the normal run) was not run: it needs the prepared data folder. The real record writer was exercised directly instead (section 3).
2. The planted bugs are my own (item A: 10, of which one is the restored pre-change file; item B: 5 text replacements) plus the re-checker's 6 for item C; this is not a complete mutation analysis.
3. The stage tests use fake fitter and scorer programs; no GPU, no cluster, no R, no data of any experiment was used.
4. The test-tier numbers are for scvi-api (python 3.11.15 numpy 2.4.6 scipy 1.17.1) and wcd-gpu (python 3.11.15 numpy 1.26.4 scipy 1.14.1); the power test is tied to scipy 1.17.1's NaN behaviour (question 7).

## 13. How to repeat

Workspace root with `rv` (my clone), `inputs/` (the task's artifacts) and `covvenv_api`, `covvenv_gpu` (venvs made with `python -m venv --system-site-packages` plus `pip install pytest-cov==7.1.0 ruff==0.14.0`, without ruff for the gpu one). The whole final run is one script, `evidence/code_r3/scripts/run_all_final.sh`, started from the workspace root. It makes fresh clones of `rv`, then runs the four acceptance checks, the rest of the scripts tier, the separate counts, the planted bugs (`r3_mutations.py` for items A and B, `mut_recheck_reused_unchanged.py` for item C, `formal_mutation_check.py` for the formal check), the probes (`run_final_probes.sh`, which includes the item D comparison), the import closures (`evidence/code_r3/probes/`), ruff, the check that the earlier head differs only in wording (`wording_only_check.py`) and the patch series (`am_check.sh`); every output goes to `evidence/code_r3/final_run/`. The lint of the changed files (`fl_lint` of the fail-loud-pipelines skill, result `fl_lint_base_and_final.json`) is run by hand after the script. `build_results.py` writes `results.json` from the saved outputs, and this report is rendered from `report_template.md` and `results.json` (`render_report_cg.py`). Planted bugs need a git clone (tests that write records call `git rev-parse`).
