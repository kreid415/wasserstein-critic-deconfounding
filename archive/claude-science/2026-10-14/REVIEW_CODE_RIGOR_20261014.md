# Rigour review, track 2 of 3: code fixes after commit 6b4aefc (other than the X1 analysis)

| | |
|---|---|
| Reviewer | RIGOR_REVIEWER, an independent methods reviewer. Review only: nothing in the project repository, the lead's worktree or the pilot data was changed. |
| Date | 2026-10-14 |
| Code reviewed | branch `final-review-fixes`, commit `1e615eb`: 59 commits after `6b4aefc`, 47 files changed, 2,588 lines added and 74 removed (25 of the files are Python). |
| How | CPU (central processing unit) only, no GPU (graphics processing unit), in my own clones of the repository detached at `1e615eb` (the lead's worktree pointer is masked in my sandbox). Every number below comes from a command listed in section 5; the output files are in the accompanying tar.gz. |
| Blinding | I did not open the X1 outputs (`X1_local`, `jhpce_tier12`, `review_sealed`, the X1 logs, the identity-review score folders) and ran no rule or analysis script on X1 or X13 scores. I read pilot data (`tier12_v2/A1*`, `A2A3`) and committed records only. |

## 1. Verdict

**PASS_WITH_FIXES: 0 blockers, 2 fixes, 8 notes.** (The three possible verdicts are PASS, PASS_WITH_FIXES and FAIL. Blocker = a wrong result or a broken decision. Fix = change before the new tag `prereg-tier12-v3`. Note = worth knowing, optional.)

I found no wrong result and no broken decision. I re-ran or recomputed all eight claims of the hand-off: five hold as stated (claims 1, 2, 4, 5, 6) and three hold with the gaps listed below (claims 3, 7, 8). The two fixes are small, and both are best done before the tag because the tag freezes the code and the tests:

1. **CF-01.** The counter for training steps that carry no batch-alignment signal is correct today, but the tests would not notice if it broke in three places (the importance-weighted call, the reference-anchored wrapper, a fit with exactly one such step). Parametrize an existing test.
2. **CF-02.** The rule "Sinkhorn rows must carry the new tag" (the user's decision SI-54) is enforced for fitted latents and scores but not for a Sinkhorn row whose only output is a divergence record. Add the record's commit to two lists in the runner, or write the limit into the pre-registration.

| id | severity | what | where |
|---|---|---|---|
| CF-01 | fix | The no-pair counter is pinned through the training loop for one adversary and one call path; three plausible one-line breakages pass every test | `tests/scvi/test_baseline_guard.py:248-252` |
| CF-02 | fix | SI-54's Sinkhorn rule does not look at the commit stored in divergence records | `scripts/run_stage.py:659-668`, `:1167` |
| CF-03 | note | The table tests of G-04/G-05 cannot see a deleted arm entry; the README's "1,908 rows" are 382 distinct rows | `tests/scvi/test_runner_and_manifest_missing_arms.py:53-84` |
| CF-04 | note | SI-54 as built: the rule scripts see only the scoring commit; a uniform third commit is accepted; the evidence list names older files | `scripts/prereg_rules.py:220-262`, `docs/EQUIVALENT_COMMITS.json:8-11` |
| CF-05 | note | Cluster path: the tag must be inside the git bundle; two comments are wrong or stale; one record field reads False for a valid mixed stage | `cluster/jhpce/local_rehearsal.sh:40`, `prod_fit_job.sh:26`, `prod_helpers.py:365` |
| CF-06 | note | The pre-registration promises that no-pair counts are "reported with the results"; no code reads them | `docs/PREREG.md:285-289` |
| CF-07 | note | SI-66 changes the diagnostic numbers (not the decisions) when the frozen pilot rule run is repeated | `scripts/decide_a2_a3.py:69-71` |
| CF-08 | note | The Sinkhorn pins are narrower than their comments (default cap in force, "inf" in the error message) | `tests/wcd_vae/test_divergences_v2.py:139`, `:151-161` |
| CF-09 | note | A new test fails in the wcd-gpu environment that the repo's full-suite script uses for it | `tests/analysis/test_analysis_x1.py:197` |
| CF-10 | note | Tooling: ruff is not clean tree-wide, the fail-loud lint is not in the repo, the src-tier test-quality warnings are a tool artefact | see finding |

## 2. Words used in this report

| term | meaning |
|---|---|
| commit, SHA, tag | A commit is a saved version of the code, named by a 40-character fingerprint (SHA, secure hash algorithm) usually shortened to 7 characters, for example `1e615eb`. A tag is a readable name for one commit: `prereg-tier12-v2` is `c88cce3`; `prereg-tier12-v3` is the new tag and does not exist yet. |
| SI-nn | A numbered standing instruction: a decision of the user recorded in `CONSTRAINTS.md`. SI-52 = draft all fixes, prove non-Sinkhorn fits unchanged, get independent review, show the diff before tagging. SI-54 = declare `c88cce3` and the new tag result-identical for every arm except Sinkhorn. SI-61 = Sinkhorn iteration cap 500,000 and the stop rule (a NaN, not-a-number, error stops the solve at once; an infinite one does not). SI-64 = retire two pilot scripts. SI-66 = low-signal cells count as "not evaluable" in the R2/R3 rules. |
| G-nn, R2-nn, CF-nn | G-nn are findings of the earlier guard review, R2-nn of the Sinkhorn re-review, CF-nn of this review. |
| arm | One method variant in the benchmark, for example `none` (plain scVI, single-cell variational inference), `scvi_adv`, `scanvi` (scVI with cell-type labels), `sysvi`, `discriminator`, `mmd`, `sinkhorn`. |
| minibatch | The small random group of cells (128 here) used for one training step. |
| no-pair minibatch | A minibatch whose cells give the batch-alignment penalty nothing to compare: only one experimental batch is present, or, for reference-anchored MMD, no reference-batch cell or no other batch. The penalty is then set to zero for that step, and since G-01 the step is counted. |
| MMD, Sinkhorn | Two penalties that measure how different the latent distributions of two batches are without training a critic network. MMD = maximum mean discrepancy (a kernel distance); Sinkhorn = an entropy-smoothed optimal-transport distance computed by repeating an update until it converges. |
| marginal error, cap | The Sinkhorn update repeats until the row sums of the transport plan match their targets within a tolerance (1e-4); the mismatch is the marginal error. The cap (500,000 repeats) stops a runaway solve with an error. |
| planted bug ("mutant"), killed, survived | A deliberate one-line breakage of the code made to test the tests. Killed = at least one test fails. Survived = every test still passes. |
| identity proof | Refitting the same rows with the old code (`c88cce3`) and the new code and comparing the latent arrays exactly, and the score tables column by column. |
| import closure | All code files a program loads, directly or indirectly. If none of them changed between two commits, the program cannot behave differently. |
| R2, R3, A1, A2, A3 | Pre-registered decision rules (R2: give the adversary the posterior mean or a posterior sample of the latent; R3: standardise each latent dimension before the adversary or not) and the pilot stages that feed them (A1: fits over a grid of alignment strengths; A2 and A3: the R2 and R3 comparisons). Defaults: posterior mean, standardisation off. |
| b0, b_common, b*, low-signal flag | b0 = batch-removal score with no adversary; b_common = the highest batch-removal score every adversarial arm reaches; b* = a matched target halfway between them. A task x decoder block is flagged `bstar_low_signal` when b_common - b0 < 0.02: the arms barely change batch removal, so a matched target is unreliable. An evaluable cell is one where both settings reach the target. |
| task, decoder, `immune|1` | A task is a benchmark data set: `atac_small` (a small chromatin-accessibility data set; ATAC = assay for transposase-accessible chromatin), `immune` (human and mouse immune cells), `sim1`, `sim2` (simulated). `|0` or `|1` after a task names the decoder setting: 0 = the decoder does not receive the batch label, 1 = it does (a conditioned decoder). `immune|1` is the immune task with the conditioned decoder. |
| fit, manifest row, latent, plan record | A manifest row is one planned training run (arm, task, settings, seed); a fit is that run; its latent is the low-dimensional embedding of every cell, saved as a file; the plan record is the description of the training setup stored in that file. The X1 manifest is the list of planned X1 fits. |
| runner, stage, divergence record, harvest | The runner (`scripts/run_stage.py`) launches fits and scoring for a manifest (a stage) and checks provenance. A divergence record is the small file the fitter writes instead of a latent when the training loss becomes non-finite (NaN, not a number, or infinite). The harvest is the cluster step that collects a job's finished outputs into one archive. |
| scorer, scIB, scib | scIB = single-cell integration benchmarking scores (batch removal and biology conservation); the scorer is the program that computes them from a latent; scib is the Python package that implements them. |
| X1, X3, X7, X8, X12, X13 | Experiment numbers of the pre-registered design (`docs/PREREG.md`). X1 is the main adversarial-arm experiment (its scores were sealed during this review); X13 holds the CPU baselines. The text says what matters about the others where they appear. |
| git bundle, clone, worktree | A git bundle is one file holding a repository's history, used to ship code to the cluster. A clone is a separate copy of the repository; a worktree is a second checked-out folder of the same repository. |
| pytest, passed / skipped / failed | The Python test runner; the numbers count test functions or parameter sets. Skipped tests need something absent (data or a package). |
| `check_tests.py`, ruff, `fl_lint` | The project's script that fails if a changed line or branch is untested or a new test adds no coverage of its own; a Python code checker (rules F = unused imports and variables, E9 = syntax errors); the project's linter for silent-failure code patterns (not in the repository). |

## 3. The eight claims of the hand-off

Status key: confirmed = reproduced as stated; partly = holds with the gap named; K-numbers refer to the commands in section 5.

### Claim 1. Each guard follow-up fixes its finding (G-01 to G-05, G-14), and the lead's planted-bug script kills 11 of 11. Status: **confirmed**

- **Lead's planted-bug script (K01).** In a clean clone at `1e615eb` the unmutated named tests passed first; then `mutate.sh` made 11 planted bugs and **11 were killed, 0 survived** (`logs/mut_lead_1e615eb.tsv`). The planted bug `G01_no_count` was killed by exactly the 3 parameter sets of `test_no_pair_of_batches_gives_differentiable_zero`.
- **G-01, minibatches with no pair of batches are counted and written to the plan record (K06).** I wrapped the registered loss to count, from the minibatch's own batch labels, the steps with no pair, and compared with the recorded `no_pair_steps`. Eight small fits: for `mmd`, `sinkhorn` and `mmd_ref` (reference batch b0 and b1) with a rare 2-cell batch, the recorded count equals the independent count in 4 of 4 fits (12 of 15 steps); in the 4 balanced fits the key is absent, as designed; mismatches: 0. The importance-weighted MMD path used by X3: recorded 12 = independent 12. Code: `src/wcd_vae/wcd/alignment.py:69-73` (the counter), `:118`, `:136`, `:255` (the three losses), `scripts/fit_paper_config.py:504-506` (the record). The functional fix is right; its tests are narrower than they look (CF-01).
- **G-02, the inner-critic non-finite test catches both partial fixes (K07).** The lead's `G02_sum_omits_gp` and `G02_first_step_only` were killed. My own planted bug `R02a` (only the last inner step is summed) was killed by `test_nonfinite_inner_critic_step_raises[0-loss]`.
- **G-03, the production call is tested (K01).** `G03_call_removed` and `G03_row_not_passed` were killed; coverage now follows subprocesses (`pyproject.toml`: `[tool.coverage.run] patch = ["subprocess"]`).
- **G-04, the extra-keys test pins `UNREAD_SETTINGS` and the `EXTRA_ARMS` keys (K01, K07).** `G04_entry_dropped` and `G04_extra_key_dropped` were killed; so was my `R05a` (arm `pooled` removed from the sampler arms). Limit: the arm sets inside `EXTRA_ARMS` are not pinned (CF-03).
- **G-05, `none`, `scvi_adv` and `scanvi` refuse a non-zero lambda (K02-K04).** `lam_audit.py` re-run prints the same text as the committed `results/lam_audit.txt` (19 manifest files, 8 with arm and lam columns, 41,583 rows; none 1,428, scanvi 240, scvi_adv 240, all lam 0). These are occurrences summed over overlapping manifest files; distinct row tags are 302 / 40 / 40 (CF-03). Running `check_extra` over **all 40,815 fittable manifest rows** (768 CPU-baseline rows are not fitted by this code) refused **0**. Through the real fitter subprocess, lambda = 5 exits with status 1, prints "does not read ['lam']" and writes no latent for `none`, `scvi_adv` and `scanvi`; lambda = 0 fits; the string "0.0" is also refused (strict string match; harmless because every manifest writes "0"). `sysvi` still accepts lambda 1 to 50, as it should (its cycle weight).
- **G-14, the harvest checks every latent's training history (K05).** On synthetic output folders: a GPU latent without history, a CPU latent without history and a GPU latent without its model are refused; a tag with only a divergence record is still packed. The real CPU-baseline writer stores an empty history (`scripts/run_cpu_baselines.py:261`) and the runner's own stage rule already requires that key (`scripts/run_stage.py:396`), so the new check refuses nothing the real writers produce. The lead's `G14_history_unchecked` and my `R14a`, `R14b` were killed.

### Claim 2. The Sinkhorn re-review fixes are correct and the comments are accurate (R2-01, R2-02, R2-04, R2-05, R2-08). Status: **confirmed** (one GPU-only figure not re-measured)

- **R2-01, an infinite marginal error does occur from latents of about 1e18 (K08).** I fed 10 solves per scale (128 cells x 10 dimensions, Gaussian latents of scale s rounded to float32, 2 or 10 batches, 5 seeds). The first marginal error of the cross-batch solve was +infinity in **0 of 10 runs at each scale up to s = 3e17**, **9 of 10 at 1e18** (4 of 5 with 2 batches, 5 of 5 with 10) and 10 of 10 at 1e19 and above (to 1e37). The potentials stayed finite, the error was never NaN (0 NaN errors), and every solve from 1e16 upward converged within 5,000 iterations (70 of 70); at 1e14, 0 of 10 did (slow, with finite errors). At every scale from 1e18 up the solve needed at most **14 iterations** at the final smoothing level (`probes/inf_iterations_probe_output.txt`), so "converges a few iterations later" is accurate. The code (`alignment.py:190-198`) already stopped only on NaN: the commit changed a comment, not behaviour.
- **The committed probe reproduces (K09).** `f1_nonfinite_probe.py` re-run gives output identical to the committed `results/f1_nonfinite_probe.txt` (24 lines).
- **R2-02, the NaN-only stop rule is pinned by a test row (K07, K08).** The row `_huge_latent_solve` (`tests/wcd_vae/test_divergences_v2.py:151-161`) uses a latent of scale 1e30 with `max_iter=1`. Premise checked: the probe prints (inf, True) for the row's first error and the finiteness of its potentials, that is an infinite error with every potential finite. The planted bug `R21a` (the cross-batch loop also stops on +infinity) was killed. The two symmetric (debiasing) solves never see an infinite error at any scale up to 1e37 (0 of 60 solves), so the same change in those loops cannot be observed by any input; it is not a gap (`R21b` survives as an equivalent change).
- **R2-04, the cap value is pinned (K07).** `test_divergences_v2.py:139` asserts `SINKHORN_MAX_ITER == 500000`; the planted bugs `R21d` (cap 50,000) and `R21e` (tolerance 5e-4) were killed. `R21c` (the function default no longer reads the constant) survived (CF-08).
- **R2-05 and R2-08, comment scope and sample sizes (K10).** The comment at `alignment.py:51-56` quotes 5,273 iterations over 480 standardised validation minibatches and 9,789 over 10,800 atac_small minibatches; these match the first Sinkhorn review (artifact 283ca295), and 500,000 / 9,789 is about 51 times. Every Sinkhorn manifest row has `batch_size` 128 (4,248 occurrences, 714 distinct tags), as the README now says. README timing recomputed from `results/bench_gpu_si58.csv`: median float32/float64 time ratios (32-bit against 64-bit number precision) 0.989 to 1.014 over five regimes; A3 117.1 vs 117.8 ms and 140.3 vs 142.7 ms (float32 vs float64). **Not re-measured:** "from 512 cells float32 is about 2.7x faster per iteration on the RTX 3080" (`alignment.py:59`, README); this review had no GPU.

### Claim 3. SI-54 does exactly what the user decided. Status: **partly** (one silent path, CF-02; wording points, CF-04)

What was checked, on the real functions `prereg_rules.equivalent_commits` and `refuse_uncovered` with the committed `docs/EQUIVALENT_COMMITS.json` (K11):

- **Scenario matrix.** 24 scenarios plus a length-mismatch check, before the tag exists (tag unresolved), with the tag at the current commit, and with one commit after the tag. **0 not as expected.** Rows at `c88cce3` together with rows at the tag are accepted for non-Sinkhorn arms; a Sinkhorn row at `c88cce3` is refused in every combination; a Sinkhorn row at the tag next to non-Sinkhorn rows at `c88cce3` is accepted; a third commit mixed with `c88cce3` rows is refused; rows at a commit after the tag mixed with `c88cce3` rows are refused; a stage made entirely at one other commit, Sinkhorn included, is accepted; an unresolved tag leaves any mix refused.
- **Real inputs.** The committed pilot inputs contain `c88cce3` Sinkhorn rows. With the committed list, `decide_a2_a3.py` refuses them with "['sinkhorn'] rows at c88cce3 ... refit and rescore them at prereg-tier12-v3" and writes nothing (`probes/si54_real_inputs_refusal_output.txt`).
- **Where the runner applies it** (`probes/grep_si54_callers.txt`): the scoring-provenance comparison (`scripts/run_stage.py:356-362`), the start-up check of every existing latent and score (`:659-679`, also in report-only runs), the final pass/fail check of the latents' commits (`:1167`), and the rule input (`scripts/prereg_rules.py:262`).
- **`latent_git_shas_match_head` is harmless.** It is written to the cluster job summary only (`cluster/jhpce/prod_helpers.py:365`); the single reader is a test that pins False for the fake fitter's SHA (`tests/jhpce/test_prod_job.py:250`), and the notebook calls it non-gating (`notebook/2026-10-13.md:361`). It reads False for a valid mixed stage (CF-05).
- **Gap found.** A Sinkhorn row whose only output is a divergence record made at `c88cce3` is not refused by the start-up check or the final check (CF-02).

### Claim 4. SI-66: flagged cells stay in n as not evaluable, the change is inert when nothing is flagged, and R2 and R3 keep their defaults. Status: **confirmed**

(K12; inputs: 1,434 recorded R2/R3 files and 1,147 recorded R4-a1 files, located by SHA-256 (a 256-bit fingerprint) because the recorded paths point into a deleted workspace; 1,434 and 1,147 match their recorded fingerprints, 0 do not. Mean difference = the average over evaluable cells of the alternative's minus the default's biology-conservation score at the matched target b*.)

| run on the same pilot inputs | R2 choice | R2 evaluable of cells (need) | R2 mean difference | R3 choice | R3 evaluable of cells (need) | R3 mean difference |
|---|---|---|---|---|---|---|
| old code `c88cce3` (reproduces the frozen `docs/prereg/r2r3_record.json`) | mean | 12 of 12 (8) | -0.0014 | off | 17 of 20 (14) | 0.0010 |
| new code `1e615eb`, real flags | mean | 9 of 12 (8) | -0.0042 | off | 13 of 20 (14) | -0.0014 |
| new code, no cell flagged (control) | mean | 12 of 12 (8) | -0.0014 | off | 17 of 20 (14) | 0.0010 |

- The old code reproduces the frozen decisions exactly. To get past SI-54's refusal of the old Sinkhorn rows, the new-code runs used a temporary equivalence list that excludes no arm, set inside the process and not committed, so the only behaviour change is SI-66.
- **Inert when nothing is flagged:** the control matches the old run in choice, switch, coverage, n evaluable and mean difference. **Defaults kept:** with the real flags both rules keep `mean` / `off` and do not switch; the X1 manifest is byte-identical (SHA-256 prefix 7c2fe8679631c882).
- Only the block `immune`, conditioned decoder is affected: 3 of 12 R2 cells and 5 of 20 R3 cells change status from bracketed (or unreached) to `low_signal`; the number of cells n is unchanged.
- **Positive control:** with a synthetic +0.5 gain for the alternative in every cell that can be judged, the flagged run still keeps the defaults (R2 `mean`, R3 `off`), the unflagged run switches (R2 `sample`, R3 `on`).
- **"Stays flagged after the Sinkhorn redo."** In the frozen R4-a1 record the block `immune|1` has b0 = 0.6382 and b_common = 0.6570, the barycenter arm's value, the smallest of 6 arms. The gap is 0.0188, below the 0.02 threshold (`scripts/prereg_rules.py:43,573`). Redoing Sinkhorn can only leave b_common unchanged or lower it, and b0 and the barycenter come from fits that the identity proof shows unchanged, so the block stays flagged either way.
- **Tests for the change (K07):** my planted bugs `R66a` (flag read from the wrong decoder) and `R66b` (flagged cells dropped from n) were killed by `tests/prereg/test_cli_e2e.py`. The diagnostic fields do change when the frozen pilot run is repeated (CF-07).

### Claim 5. SI-64: the two retired scripts cannot run, and nothing live uses them. Status: **confirmed**

(K13, `probes/legacy_retired_check_output.txt`.) `python legacy/fit_stock_scvi.py` and `python legacy/scvi_fit_pancreas.py` each print "RETIRED (code check CR-10 follow-up; final review I050, user decision SI-64 ...)" (2 of 2 messages seen) and exit with status 1 and 1 respectively. The copies in `scripts/` no longer exist (2 of 2 looked up are absent). A search of the whole repository finds the names only in `legacy/README.md`, `tests/test_legacy_retired.py`, `scripts/README_baselines_geometry.md`, `CONSTRAINTS.md` and the notebook index and entries; the number of mentions in `manifests/`, `cluster/`, `docs/PREREG.md`, `scripts/run_stage.py` and `scripts/*.sh` is 0. The two live latent writers save cell names (`scripts/run_cpu_baselines.py:258`, `scripts/fit_paper_config.py:358`), so the scorer's refusal of latents without them (I050) does not touch live rows.

### Claim 6. Identity: 59 of 59 non-Sinkhorn fits equal `c88cce3`, 4 of 4 Sinkhorn fits differ, 7 of 7 latents score identically, and the proof carries over to `1e615eb`. Status: **confirmed**

- **Committed results (K14).** `fit_identity_c3fd92d.csv`: 63 rows, every return code 0 for both code versions; 59 non-Sinkhorn rows of which **59 identical** (largest absolute difference 0.0); 4 Sinkhorn rows of which 0 identical (largest absolute differences 0.43 to 6.70, as expected after the solver changes). `score_identity_c3fd92d.csv`: 7 latents x 32 compared columns, 0 differing between the old and new scorer; against the production scores only the `score_host` column differs, for one latent.
- **Carry-over (K14).** `git diff --name-only c3fd92d..1e615eb` touches only documents, the notebook, `experiments/integration_checks_20261013/results`, `scripts/analysis_x1.py`, `scripts/decide_a2_a3.py` and tests. I built the import closure (static, with a scan for dynamic loading) of the fit entry points (`run_stage.py`, `fit_paper_config.py`, `scvi_adversarial_plan.py`, `run_cpu_baselines.py`: 8 files), of the scorer the runner calls (`score_scib_native.py`: 2 files) and of the cluster job helpers (1 file). **None of the 3 closures contains a file changed in the range, and no `src/` file changed.** I therefore did not refit.
- **Limits.** The proof itself is CPU, one epoch, one thread; I did not repeat it, and it cannot show GPU-only differences. The fit-code changes of the whole range are the counter plumbing (`alignment.py`, `scvi_adversarial_plan.py`), the refusal tables and one record field (`fit_paper_config.py`); none is device-specific. I ran no GPU fit.

### Claim 7. Tests at `1e615eb`: full suites pass, `check_tests.py` is OK with no warning per tier, the fail-loud lint and ruff F/E9 are clean. Status: **partly**

| run | environment | result |
|---|---|---|
| `check_tests.py --base 6b4aefc` for scripts and cluster (K16) | scvi-api | 417 passed, 18 skipped; "check_tests: OK"; 0 fail, 0 warning |
| `check_tests.py --base 6b4aefc` for `src` (K16) | wcd-gpu | 172 passed; 0 fail; **10 warnings**, all for test functions of the other tier (CF-10) |
| `tests/scvi tests/jhpce tests/stage` with the data folder set (K15) | scvi-api | 220 passed, 2 skipped |
| `tests/stage` with the real fitter and scorer switched on (K15) | wcd-gpu + scvi-api + wcd-kbet | 66 passed |
| `tests/x13 tests/scoring` (K15) | wcd-kbet | 31 passed |
| `tests/prereg tests/analysis` and two top-level files (K15) | scvi-api | 212 passed, 1 skipped (a trajectory test that needs scib; it runs in the next row) |
| everything else, the repo's own `run_all_suites.sh` group "other" (K15) | wcd-gpu | 386 passed, **1 failed** (CF-09) |

- **fl_lint is not checkable.** `scripts/fl_lint.py` is not in the repository, on disk or in my skill catalogue. The range adds two suppression comments for it, `# fl: allow FL122` in `experiments/guard_followups_20261013/mutate.sh` and `# fl: allow FL171` in `experiments/sinkhorn_fix_20261013/validate_sinkhorn.py:74`. I checked their stated reasons: the verdict column of `mutate.sh` records every exit status (K01), and `summarize.py:19` and `si58_summary.py:43` count the recorded cap hits. I cannot check the rule texts themselves.
- **ruff 0.14.0, rules F and E9 (K17).** Whole repository: 28 findings (24 unused imports, 3 unused variables, 1 f-string without placeholders), identical at `6b4aefc` (28); 24 are in `scripts`, `src`, `tests`, `cluster` and `legacy`. The files changed in the range contain 4 findings (`scripts/fit_paper_config.py:24,286,287`, `tests/test_legacy_retired.py:5`); all 4 were already present at `6b4aefc` and **0 are on lines changed in the range**. So "ruff clean" holds for the changed lines, not for the changed files or the tree.
- All 25 changed or added Python files compile and both changed shell scripts pass `bash -n` (K18).

### Claim 8. Open-ended: anything these commits broke or missed, including tests that pass for the wrong reason. Status: **partly** (nothing broken; two fixes and eight notes below)

My own planted bugs (K07): 18 one-line changes in the code under review; **10 killed, 8 survived**. Of the survivors, 3 are behaviour-neutral (`R21b` changes a branch no input can reach, `R54a` changes only the wording of a refusal, `R54b` is my control change that must survive) and **5 are real test gaps**: `R01a`, `R01b`, `R01c` (CF-01), `R05c` (CF-03) and `R21c` (CF-08). The survivors were run against the whole `tests/scvi` folder (98 passed and 16 skipped without the data folder) and, for `R05c` and the three counter bugs, again with the data folder (113 passed, 1 skipped in each of those four runs; all four survived).

## 4. Findings

### CF-01 (fix). The no-pair counter is pinned through the training loop for one adversary and one call path

- **Where.** Code: `scripts/scvi_adversarial_plan.py:231` (the reference-anchored MMD wrapper forwards the counter), `:369` and `:371` (the unweighted and the importance-weighted call that pass it), `scripts/fit_paper_config.py:504-506` (writes `no_pair_steps` only when the count is above 0). Tests: `tests/scvi/test_baseline_guard.py:248-252` (the only fit that checks the count; adversary `mmd`, unweighted) and `:285`.
- **What.** Guard G-01 counts training steps in which a minibatch has no pair of batches to compare, so that the zero penalty of those steps is visible. The counter is correct today (claim 1), but three plausible one-line breakages pass every test:
  - `R01a`: the weighted call at `:371` stops passing the counter (the X3 importance-weight arm).
  - `R01b`: the wrapper at `:231` stops forwarding it. This is `mmd_ref`, the arm for which the pre-registration expects the most no-pair steps (X7, pancreas and sim1).
  - `R01c`: the record is written only when the count is above 1 (`if n > 1` at `fit_paper_config.py:505`), so a fit with exactly one such step loses its record.
- **Evidence.** K07 (`logs/my_mut_results.json`): all three SURVIVED the whole `tests/scvi` folder (98 passed, 16 skipped). With the data folder set they also survived (3 of 3; 113 passed, 1 skipped each). K06 shows the code is right today: recorded = independent count in 4 of 4 fits and for the weighted path, so this is a test gap, not a wrong result.
- **Why it matters.** The lead's mutation claim (11 of 11) covers the lead's list; breaking the same counter in the places listed above is not caught. The counts are the only record of how many steps carried no batch-alignment signal for the arm where that happens most, and the pre-registration says they are reported (CF-06).
- **Suggested fix** (no new test function, as the fewest-tests rule asks). Parametrize the existing one-batch fit at `test_baseline_guard.py:248-252` over the adversary (`mmd`, `mmd_ref`, `sinkhorn`) and add one importance-weighted case; add an `n == 1` case where the plan record is built from a fake plan (`:285`). Then re-run `mutate.sh` and the three planted bugs (`probes/my_mutate.py ... R01a_weighted_branch_no_stats R01b_mmdref_lambda_drops_stats R01c_record_needs_two_steps`); all must be killed.

### CF-02 (fix). SI-54 does not look at the commit stored in divergence records

- **Where.** `scripts/run_stage.py:659-668` (the start-up check collects commits from each row's latent and score only), `:1167` (the final check uses the latents' commits only), `scripts/prereg_rules.py:262` and `:289-292` (rule inputs see the scoring commit only; the failure table the rules read has the columns tag, status and detail, and the pilot's has no commit column).
- **What.** SI-54: rows made at `c88cce3` may be mixed with rows at the new tag for non-Sinkhorn arms only; Sinkhorn rows must carry the new tag. The runner enforces this for fitted latents and for scores. A Sinkhorn row whose only output is a divergence record (status `diverged`: the training loss became non-finite; no latent file, no score) carries the commit that made it, but that commit is read nowhere.
- **Evidence.** K11. The real fitter writes `git_sha` into its divergence record (`probes/real_status_record_output.txt`: `1e615eb...`). The repo's own test fixture writes none, so I added the field it would have, set to `c88cce3`, to a Sinkhorn row's divergence record. Re-running the stage printed "[gate] PASS: 2/2 final" (the runner's final pass/fail check) and exited 0, in normal and report-only mode (`probes/si54_status_gap_output.txt`). A Sinkhorn latent or score at `c88cce3` is refused (`tests/stage/test_run_stage.py:664-686`, parametrized over `score` and `latent`); a divergence record is the third kind of output, and that test has no case for it.
- **Consequence if it happened.** An old-code Sinkhorn failure (old solver: a fixed number of updates with no convergence loop; the float64 solve, the convergence loop and the cap are new) would count as a final outcome of the new stage, although the new code might fit that row. No such record exists today: the pilot failure table has 0 data rows (`probes/pilot_failures_rows_output.txt`), and X1 fits are planned at the new tag. This is a hole in the enforcement, not a wrong result.
- **Suggested fix.** Add the divergence record's `git_sha` (`info["status"]["git_sha"]`) to the commits collected at `:659-668` and to the list at `:1167`, and add a third case, `status`, to `test_sinkhorn_output_at_the_pre_fix_commit_is_refused` (`tests/stage/test_run_stage.py:664-686`; the fixture's `plan={"s1": "diverge"}` produces the record). Or, if you prefer not to touch `run_stage.py`, state the limit in one sentence of the pre-registration. `run_stage.py` is inside the identity proof's import closure (claim 6): after a change there, re-run `probes/import_closure.py <clone> c3fd92d <new sha>` (seconds) and write down that the change only affects start-up bookkeeping that the proof (fitted latents and scores) does not exercise, or repeat the proof.

### CF-03 (note). The table tests of G-04 and G-05 cannot see a deleted arm entry, and the README counts occurrences

- **Where.** `tests/scvi/test_runner_and_manifest_missing_arms.py:53-84` (the loop at `:79-84` iterates over the arms the table lists; the test function is `:53-84`); `scripts/fit_paper_config.py:55-62` (`_NETWORK_ARMS`, `EXTRA_ARMS`); `experiments/guard_followups_20261013/README.md` (G-05 row).
- **What.** The test pins the keys of `EXTRA_ARMS` and the contents of `UNREAD_SETTINGS`, but a deleted arm inside a value set only shortens the loop. The planted bug `R05c` removes `reference` from `_NETWORK_ARMS`, after which 144 distinct X12 rows (864 occurrences over overlapping manifests) that use `adv_width` / `adv_lr` would be refused at fit time (`probes/reference_adv_rows_output.txt`); it survived with and without the data folder (113 passed, 1 skipped, verdict SURVIVED). `R05a` was killed only because `pooled` appears in an explicit example. The failure would be loud at fit time (a refusal, never a silent change), and today no committed row is refused (0 of 40,815).
- **Counts.** The README says "all 1,908 manifest rows of these arms carry lam 0". 1,908 = 1,428 + 240 + 240 occurrences summed over overlapping manifest files; the distinct row tags are 302 + 40 + 40 = 382 (`probes/lam_unique_tags_output.txt`). The conclusion (all lambda 0) does not depend on the unit.
- **Suggested fix.** Pin the value sets as the settings are pinned (`{k: sorted(v) for k, v in fpc.EXTRA_ARMS.items()} == {...}`), or loop `check_extra` over the committed manifest rows (K03 shows 0 refused). Say "occurrences" or give both counts in the README.

### CF-04 (note). SI-54 as built: the rule scripts see only the scoring commit, a uniform third commit passes, and the evidence list names older files

- **Where.** `scripts/prereg_rules.py:220-262`; `docs/EQUIVALENT_COMMITS.json:8-11`; `docs/PREREG.md` (the SI-54 paragraph).
- **What.** (a) Score rows carry the commit of the scorer, not of the fit, so a rule input cannot show that a Sinkhorn latent was fitted at `c88cce3` and rescored at the tag; the runner's start-up check catches it from the latent, and SI-54 says "no rescoring". (b) SI-54 reads "Sinkhorn rows must carry the new tag". The code refuses Sinkhorn rows at `c88cce3` and, when `c88cce3` rows are present, requires one commit after the mapping, so Sinkhorn rows must then equal the tag's commit; a stage made entirely at another single commit, Sinkhorn included, is accepted (scenario "uniform third commit"). The pre-registration wording matches the code. (c) The `evidence` field of `docs/EQUIVALENT_COMMITS.json` names `fit_identity.csv` and `score_identity.csv` of the first identity run (commit 2b52542), not the re-run at the later commit (`fit_identity_c3fd92d.csv`, `score_identity_c3fd92d.csv`).
- **Suggested fix.** One sentence in the pre-registration stating (a) and (b); add the `_c3fd92d` files, and the commit they were made at, to `evidence`.

### CF-05 (note). Cluster path: the tag must be inside the bundle, two comments are wrong or stale, one record field reads False for a valid mixed stage

- **Where.** `cluster/jhpce/local_rehearsal.sh:40`; `cluster/jhpce/prod_fit_job.sh:26`; `cluster/jhpce/prod_helpers.py:365`; `experiments/X1_jhpce/x1_jhpce_command.py:23,83-85`.
- **What and evidence (K11d, `probes/bundle_tag_test_output.txt`).** A clone of a git bundle contains the tag only when the bundle lists it: a bundle of the branch alone gives a clone where `prereg-tier12-v3` does not resolve (so any stage that mixes `c88cce3` rows with new rows is refused, loudly); a bundle that lists the tag, with or without a branch, resolves it. The comment at `local_rehearsal.sh:40` ("git clone of a bundle takes refs/heads/* only") is therefore inaccurate. The comment at `prod_fit_job.sh:26` says latents fitted at another commit fail the runner's mixed-SHA check; since SI-54 that holds only for commits outside the equivalence list. `x1_jhpce_command.py:83-85` checks only that the expected commit is a bundle head and still pins `TAG_SHA` to `c88cce3` (`:23`, known to the lead as G-17). `latent_git_shas_match_head` (`prod_helpers.py:365`) is a raw comparison and reads False for a stage that mixes `c88cce3` and tag rows although the runner accepts it.
- **Suggested fix.** Correct the two comments; when a cluster stage must resume `c88cce3` rows, list the tag in the bundle command; compare after `equivalent_commits`, or note in its comment that the field is raw.

### CF-06 (note). The pre-registration promises that no-pair counts are reported; nothing reads them

- **Where.** `docs/PREREG.md:285-289` ("their counts are reported with the results"); no reader in `scripts/run_stage.py`, `scripts/analysis_x1.py`, `scripts/prereg_rules.py` or `cluster/jhpce/prod_helpers.py` (`probes/grep_no_pair_steps.txt`: the only code mentions are the counter and the record writer).
- **What.** The counts exist in each affected latent's plan record, but no step collects them per arm and rule cell. The pre-registration lists where they are expected: X7 reference-anchored MMD on pancreas and sim1, and rarely X3 atac_small MMD, X8 sim2 MMD and X1 sim1 MMD and Sinkhorn.
- **Suggested fix.** After the runs, a short script over the latents' plan records that tabulates `no_pair_steps` by arm, task and experiment, saved with the results (reporting only; it affects no rule, so it can follow the tag).

### CF-07 (note). SI-66 changes the diagnostic numbers, not the decisions, when the frozen pilot rule run is repeated

- **Where.** `scripts/decide_a2_a3.py:69-71` (flagged cells get status `low_signal`; their reading is kept as `reading_status`).
- **What and evidence (K12).** The hand-off says no existing R2/R3 result changes. That holds for the decisions (R2 `mean`, R3 `off`, no switch; X1 manifest identical). The reported diagnostics change if the pilot is re-run with the new code: R2 evaluable cells 12 to 9 of 12, R3 17 to 13 of 20, coverage True to False in both, mean difference R2 -0.0014 to -0.0042 and R3 0.0010 to -0.0014. That is the intended consequence of SI-66 (a flagged block cannot be judged).
- **Suggested fix.** Wherever the paper or notebook quotes the pilot R2/R3 counts (12 of 12, 17 of 20), say that they come from the frozen record made by `c88cce3` and that the current code reports 9 of 12 and 13 of 20 for the same inputs.

### CF-08 (note). The Sinkhorn pins are narrower than their comments

- **Where.** `tests/wcd_vae/test_divergences_v2.py:139` and `:151-161`; `src/wcd_vae/wcd/alignment.py:225`.
- **What and evidence (K07, K08).** (a) The cap pin reads the constant (`SINKHORN_MAX_ITER == 500000`), but production calls `sinkhorn_divergence_weighted` without `max_iter`, so the value in force is the function default at `alignment.py:225`; the planted bug `R21c` (default literal 50,000, constant unchanged) passes all 25 tests. (b) The `_huge_latent_solve` row asserts only the text "after 1 iterations"; with `max_iter=1` a latent of ordinary scale 1 raises the same message (marginal error 0.0637; `probes/huge_row_ordinary_control_output.txt`), so the row works because of its premise (an infinite first error, checked in `probes/huge_row_premise_output.txt`), not because the test says so.
- **Suggested fix.** In the same test, assert `inspect.signature(sinkhorn_divergence_weighted).parameters["max_iter"].default == alignment.SINKHORN_MAX_ITER`; match "marginal error inf" in the row.

### CF-09 (note). A new test fails in the wcd-gpu environment that the repo's full-suite script uses for it

- **Where.** `tests/analysis/test_analysis_x1.py:197` (`np.trapezoid`, available from NumPy 2.0); `experiments/integration_checks_20261004/run_all_suites.sh:14` (group "other", environment wcd-gpu, NumPy 1.26.4).
- **What and evidence (K15).** `tests/analysis/test_analysis_x1.py::test_pool_brute_force` fails there with "module 'numpy' has no attribute 'trapezoid'" (1 failed, 386 passed). It passes in scvi-api (NumPy 2.4.6), the environment the task names for `tests/analysis`, and `conda-lock.yml` pins NumPy 2.2.6, so the locked environment is unaffected. The file already imports `cumulative_trapezoid` from SciPy. The analysis tests are to be frozen with the tag (`docs/ANALYSIS_SPEC.md:10`).
- **Suggested fix.** Replace `np.trapezoid` by `scipy.integrate.trapezoid` in the test, or route `tests/analysis` to scvi-api in `run_all_suites.sh`.

### CF-10 (note). Tooling: ruff is not clean tree-wide, the fail-loud lint is not in the repo, and the src-tier warnings are a tool artefact

- **Where.** `scripts/fit_paper_config.py:24,286,287`; `tests/test_legacy_retired.py:5`; `scripts/check_tests.py:88-119`.
- **What and evidence (K16, K17).** (a) ruff F/E9: 28 findings tree-wide, 4 in changed files, none on changed lines (claim 7). (b) `fl_lint` cannot be run by an outside reader, so the notebook's "fl_lint 0 findings" is not checkable from the repository. (c) `check_tests.py --base 6b4aefc` for the `src` tier prints 10 warnings; 7 are for `tests/analysis`, the rest for three test functions that belong to the scripts tier. The script reads every changed test file but runs only the tests it is given, so a test that is run in the other tier appears to cover nothing. In the tier that runs them, the same tests give no warning (claim 7).
- **Suggested fix.** Remove the four unused imports or mark them deliberate; commit `scripts/fl_lint.py`, or its version and rule list; run `check_tests.py` once with both test sets for the final record.

## 5. What I ran

Environments: scvi-api (Python 3.11.15, NumPy 2.4.6, pandas 2.3.3, torch 2.13, scvi-tools 1.4.2, no scib), wcd-gpu (NumPy 1.26.4, pandas 1.5.3, torch 2.4.1, scib 1.1.7), wcd-kbet (scorer; R_HOME and R_LIBS set). Test runs used copies of the first two with `pytest-cov` 7.1.0 and ruff 0.14.0 added (`--system-site-packages`). All runs: `KMP_AFFINITY=disabled OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=` (CPU only), at most 4 processes at a time. `<clone>` = a detached clone of the repository at `1e615eb` with its remote removed. Output files are in the tar.gz under `logs/`, `probes/` and `r23/`.

| id | what | command (abbreviated) | output |
|---|---|---|---|
| K01 | lead's planted-bug script, in a clean clone, after the unmutated named tests passed | `bash experiments/guard_followups_20261013/mutate.sh 1e615eb <clean clone> > mut_lead_1e615eb.tsv` | `logs/mut_lead_1e615eb.tsv` |
| K02 | lambda audit re-run and distinct-tag count | `python experiments/guard_followups_20261013/lam_audit.py`; `python probes/lam_unique_tags.py <clone>` | `probes/lam_audit_rerun_output.txt`, `probes/lam_audit_diff_output.txt`, `probes/lam_unique_tags_output.txt` |
| K03 | `check_extra` over every committed manifest row | `python probes/check_extra_all_rows.py <clone>` | `probes/check_extra_all_rows_output.txt` |
| K04 | real fitter subprocess with a non-zero lambda | `python probes/g05_fitter_subprocess.py <clone>` | `probes/g05_fitter_subprocess_output.txt` |
| K05 | harvest of synthetic output folders | `python probes/g14_harvest.py <clone>` | `probes/g14_harvest_output.txt` |
| K06 | no-pair counter against an independent count | `python probes/g01_end_to_end.py <clone>`; `python probes/g01_weighted.py <clone>` | `probes/g01_end_to_end_output.txt`, `probes/g01_weighted_output.txt` |
| K07 | my own planted bugs (one text change each) in a clean clone | `python probes/my_mutate.py <clone> out.json [ids]`; with the data folder: `MUT_PREPPED=1 ...` | `logs/my_mut_results*.json`, `logs/my_mut_progress*.log` |
| K08 | infinite-error probes (wcd-gpu): first-error scan, symmetric solves, test-row premise, iterations to converge | `python probes/inf_error_probe.py`, `sym_inf_scan.py`, `huge_row_premise.py`, `inf_iterations_probe.py` (from a clone root) | `probes/inf_error_probe_output.txt`, `sym_inf_scan_output.txt`, `huge_row_premise_output.txt`, `inf_iterations_probe_output.txt` |
| K09 | committed Sinkhorn probe re-run | `python experiments/sinkhorn_fix_20261013/f1_nonfinite_probe.py > rerun.txt; cmp rerun.txt results/f1_nonfinite_probe.txt` | `probes/f1_nonfinite_probe_rerun.txt` |
| K10 | README and comment numbers | `python probes/sinkhorn_readme_checks.py <clone>` | `probes/sinkhorn_readme_checks_output.txt` |
| K11 | SI-54: scenario matrix on the real functions (tag made locally in a scratch clone), divergence-record probe, real record, bundle test, refusal on real inputs | `python probes/si54_scenarios.py <clone>`, `si54_status_gap.py`, `real_status_record.py`; `git bundle` + `git clone` of three bundles; `python scripts/decide_a2_a3.py --manifest ... --scores ... --failures ... --r4a1-record ...` | `probes/si54_*_output.txt`, `probes/real_status_record_output.txt`, `probes/bundle_tag_test_output.txt`, `probes/grep_si54_callers.txt` |
| K12 | SI-66: old code (`c88cce3` clone) and new code on the recorded pilot inputs; control and synthetic-gain runs; fingerprint check of the inputs | `python scripts/decide_a2_a3.py ...` (old); `python probes/si66_run_new.py <root> ...`; `python probes/si66_compare.py`; `python probes/si66_synthetic_gain.py`; `python probes/r23_input_sha_check.py <clone> <pilot folder>` | `r23/`, `probes/si66_*_output.txt`, `probes/r23_input_sha_check_output.txt`, `probes/immune1_flag_arithmetic_output.txt` |
| K13 | retired scripts | `python legacy/fit_stock_scvi.py`; `python legacy/scvi_fit_pancreas.py`; `grep -rnI` for both names | `probes/legacy_retired_check_output.txt` |
| K14 | identity proof (counts only) and import closure | `python probes/identity_csv_summary.py` (in the clone); `python probes/import_closure.py <clone> c3fd92d 1e615eb` | `probes/identity_csv_summary_output.txt`, `probes/import_closure_output.txt` |
| K15 | test groups | `pytest -q -rs` on `tests/scvi tests/jhpce tests/stage` (data folder set); `tests/stage` with the real fitter and scorer; `tests/x13 tests/scoring`; `tests/prereg tests/analysis tests/test_legacy_retired.py tests/test_gate_task_split.py`; group "other" of `run_all_suites.sh` | `logs/pt_*.log` |
| K16 | `check_tests.py --base 6b4aefc` per tier | `--src scripts cluster -- tests/scvi tests/jhpce tests/prereg tests/stage tests/analysis tests/test_legacy_retired.py tests/test_gate_task_split.py` (scvi-api); `--src src -- tests/wcd_vae` (wcd-gpu) | `logs/ct_scripts_cluster.log`, `logs/ct_src.log` |
| K17 | ruff | `ruff check --select F,E9 --no-cache --output-format concise .` at `1e615eb` and `6b4aefc`, and on the files changed in the range; line check against `git diff -U0` | `probes/ruff_*.txt`, `probes/ruff_changed_lines_check.txt` |
| K18 | syntax | `python -m py_compile` on the changed Python files; `bash -n` on the changed shell scripts | `probes/syntax_checks_output.txt` |
| K19 | controls behind CF-03 and CF-08 | `python probes/reference_adv_rows.py <clone>`; `python probes/huge_row_ordinary_control.py` (from a clone root, wcd-gpu) | `probes/reference_adv_rows_output.txt`, `probes/huge_row_ordinary_control_output.txt` |

## 6. Limits of this review

- **No GPU, no cluster.** The GPU speed figure (float32 about 2.7 times faster from 512 cells) was not re-measured. The cluster path was checked by reading code and by toy git bundles only; no cluster command was run.
- **Identity proof not repeated.** The import-closure argument showed that no fit, scoring or cluster-helper file changed since `c3fd92d`, so I did not refit. The committed proof is CPU, one epoch, one thread.
- **Planted bugs are my own text replacements.** 18 of mine plus the lead's 11, each run against a chosen test set, not a full mutation analysis.
- **SI-66 re-run used a temporary change.** To get past SI-54's refusal of the old Sinkhorn rows, the new-code runs used a temporary equivalence list that excludes no arm, set inside the process; nothing was committed.
- **Synthetic latents.** The infinite-error probes use Gaussian latents of chosen scale with 128 cells and 10 dimensions, not the pilot latents.
- **Lint.** `fl_lint` was not available (not run; informal reading of added lines only). ruff ran with `--select F,E9` only.
- **Test runs.** The data-dependent tests ran with the data folder set; group "other" of the repo's full-suite script ran in wcd-gpu as an extra and has one environment-specific failure (CF-09). The `src`-tier warnings are a tool artefact (CF-10).
- **Blinding respected.** No X1, X13 or identity-review score output was opened; the identity proof was read as committed counts only.
- **No literature search.** The review concerns code behaviour against the user's decisions, so no prior-work check applied; no rigor ledger exists in this project and none was started, because this delegated task forbids asking the user.
- **Skills.** The skills my review profile can load do not include `ponytail-review` (the user's code-simplicity review) or a document-build check; I applied the claim-check protocol to this report (`claim_receipt.txt`).

## 7. Repeat the checks

1. `git clone <repository> clone && cd clone && git checkout --detach 1e615eb && git remote remove origin` (a clone must be a git repository, because tests call `git rev-parse`).
2. Unpack the tar.gz; run each probe as listed in section 5 from the clone root with the environment named in its file header (the probes take the clone path as the first argument where they need it).
3. `python build_results.py .` inside the unpacked folder regenerates `results.json` from the saved outputs; `report_template.md` plus `results.json` regenerate this report's numbers.
