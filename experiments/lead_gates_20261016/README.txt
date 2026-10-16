LEAD GATES 2026-10-16: SI-89 code change, three test tiers, merged coverage, import closure
Tested commit da8542e8941c3005e6ef1c7d7981dbabb8a410be (tree 23ae51b8ecad237ee393df025022552d20fb7c6e), parent 5c743ea636ea (bundle SHA-256 d8d7e8f602c2ddab20c94ab19dcba1fdfc33fb9a0e34cbdbfadcf673065191bd).
This folder is added in a later commit; it changes no file outside experiments/lead_gates_20261016/, so the tested tree is the HEAD tree minus this folder.
Each tier ran in its own fresh clone of da8542e (local clone of the bundle's branch). Thread caps in every run:
KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL CUDA_VISIBLE_DEVICES=
Blinding kept: no X1_local, jhpce_tier12, review_sealed, scores*, logs/, review/identity_* or X13 baseline folder was opened.

1. SI-89 CODE CHANGE (commit da8542e)
scripts/run_stage.py: after the clean-flag refusal, `if not cfg.get("git_sha"): raise InvalidOutput(f"{p}: the latent names no git_sha")`.
Producers of latent configs, all checked by reading the code: scripts/fit_paper_config.py provenance() (line 271: git_sha from `git rev-parse HEAD`, check=True, so never empty);
scripts/run_cpu_baselines.py provenance() (line 146, same); tests/stage/fake_env.py (lines 100 and 150 write git_sha="fake"); scripts/run_stage.py git_state() is the runner's own record, not a latent;
cluster/jhpce/* only reads latents (gate_gpu.sh line 89 reads cfg["git_sha"]; prod_helpers.py reads) and writes none; scripts/score_counts_pilot.sh, legacy/*, src/wcd_vae/wcd/experiment.py write other files that run_stage.py does not read (not producers for the stage runner).
Test: one new parametrize case `latent-without-git_sha` in tests/stage/test_run_stage.py::test_outputs_from_a_dirty_or_unknown_commit_are_refused (no new function; two docstring lines added).
Planted-bug check: in a scratch copy the two new source lines were deleted; the whole tests/stage directory then gave 1 failed, 74 passed, 1 skipped (182.7 s); the one failure was
test_outputs_from_a_dirty_or_unknown_commit_are_refused[latent-without-git_sha]. With the lines present the 9 cases of that test pass (18.4 s). Scratch copy deleted afterwards.

2. ENVIRONMENTS (versions read in each env)
scvi-api: Python 3.11.15, pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2, rpy2 not installed, no R on path.
wcd-gpu: Python 3.11.15, pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2, rpy2 not installed.
wcd-kbet: Python 3.11.15, pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2 (the two coverage packages were missing and were installed with pip at exactly these pins), rpy2 3.6.7, R 4.5.2 (/home/kendall/.claude-science/conda/envs/wcd-kbet/bin/R).

3. TIERS (exact commands, run from the root of each clone; wall time is from the shell)
a) scripts tier, env scvi-api, WCD_SRC=src:
   python scripts/check_tests.py --base 973a97d --src scripts cluster -- tests/analysis tests/prereg tests/scvi tests/stage tests/jhpce tests/test_legacy_retired.py
   RESULT: exit 0, 439 passed, 18 skipped, 950.6 s pytest (shell wall 953 s). "check_tests: OK", no FAIL, no WARN.
   Skips: 16 were reproduced with pytest -rs on the data-dependent files: 15 need PREPPED_DIR (prepped scIB h5ad files) and 1 is the opt-in end-to-end test (needs STAGE_E2E_FIT_PY, STAGE_E2E_SCORE_PY, R_HOME, R_LIBS, PREPPED_DIR); 2 further skips were not traced (not asked, not in the data-dependent files). The command given does not set PREPPED_DIR, as in the earlier tier runs.
b) source tier, env wcd-gpu, PYTHONPATH=src:
   python scripts/check_tests.py --base 973a97d --src src -- tests/wcd_vae
   RESULT: exit 0, 172 passed, 0 skipped, 131.1 s pytest (shell wall 134 s). "check_tests: OK", no FAIL, no WARN.
c) scoring tier, env wcd-kbet, WCD_SRC=src, R_HOME=$CONDA_PREFIX/lib/R, R_LIBS=/home/kendall/experiment_data/wasserstein-critic-deconfounding/Rlib_kbet:
   python scripts/check_tests.py --base 5cb75c0 --src scripts -- tests/scoring
   RESULT: exit 1. The tests RAN: 14 passed, 0 skipped, 129.5 s pytest (shell wall 132 s). The exit is 1 because check_tests (base 5cb75c0, scripts/ changed since that base) reports FAIL for 7 files tested by the other tiers
   (analysis_x1.py, check_tests.py, decide_a2_a3.py, fit_paper_config.py, prereg_rules.py, run_stage.py, scvi_adversarial_plan.py) and 14 WARN for new tests that are in other tiers. See check_tests_tier_c.log (lines 76-97). Not edited, not hidden: this tier alone cannot satisfy check_tests at base 5cb75c0; the earlier rigour re-check evaluated this base by merging the tiers (section 4), and its report says the scoring tier ran only tests/scoring with --base 5cb75c0.
   The task text ended the command with "tests/scoring ." ; I read the "." as the end of the sentence. Run literally with "." in wcd-kbet, pytest collected the whole repo and stopped with 13 collection errors (tests/wcd_vae needs PYTHONPATH=src; 'ot' missing), 3 skipped.

4. RPY2 FIX (tier c)
Diagnosis in wcd-kbet: `import rpy2.robjects` works there (rpy2 3.6.7, R 4.5.2). The module-level skip in both scoring test files comes from the next line: library(kBET) fails ("there is no package called kBET") because R_LIBS is not set; the pytest skip reason in the first run was: R kBET package not loadable (set R_LIBS as in scripts/run_wave.sh). Fix: export R_HOME=$CONDA_PREFIX/lib/R and R_LIBS=/home/kendall/experiment_data/wasserstein-critic-deconfounding/Rlib_kbet (the library that holds kBET) before running; with this the 14 scoring tests run and none is skipped.
   A shell in which the env is not activated (python not from wcd-kbet) fails earlier, on the rpy2 import itself; I did not reproduce that case.

5. MERGED COVERAGE (ct_union_eval.py, byte-identical copy of recheck_code_20261014/scripts_recheck/ct_union_eval.py; the tier-a clone was passed twice, as tier_a:scripts and tier_a:cluster, so that cluster/ files count; files under no listed src dir are not measured)
   python ct_union_eval.py <dev clone> BASE tier_a:scripts tier_a:cluster tier_b:src tier_c:scripts   (output: union_of_tiers.txt)
   base 5cb75c0: 10 changed .py files measured; 1 with lines run by no tier: scripts/check_tests.py, 101 lines, 0 branches (as in the earlier record).
   base 973a97d: 3 changed .py files measured; 0 with lines or branches run by no tier.

6. IMPORT CLOSURE (probes from code_r3_evidence.tar.gz, byte-identical copies in this folder; outputs closures_by_entry.txt, import_closure_reused_unchanged.txt; run on da8542e)
   python closures_by_entry.py <clone> import_closure_reused_unchanged.py c3fd92d 5d06a27 HEAD
   python import_closure_reused_unchanged.py <clone> c3fd92d HEAD
   fit_paper_config.py, scvi_adversarial_plan.py, run_cpu_baselines.py, score_scib_native.py, prod_helpers.py: no changed file in the closure, in either round. run_stage.py closure (8 files): changed earlier: run_stage.py; changed 5d06a27..HEAD: prereg_rules.py and run_stage.py. As expected; no difference.

7. FILES: check_tests_tier_{a,b,c}.log and .exit (none above 1 MB, none trimmed), union_of_tiers.txt, closures_by_entry.txt, import_closure_reused_unchanged.txt, ct_union_eval.py, closures_by_entry.py, import_closure_reused_unchanged.py.
   The .log files were produced in work clones outside the repo; git add -f is used for them.

NOTE (2026-10-16, confirming re-review CR-01): the commit id da8542e8941c3005e6ef1c7d7981dbabb8a410be cited at lines 2, 4, 8 and 45 above is the id the tested clone had. In the branch final-review-fixes the same change is commit 4b02068f041a289c2e7ff4f52830fb3c01a1a176 (same tree 23ae51b8ecad237ee393df025022552d20fb7c6e). Read da8542e as 4b02068.
