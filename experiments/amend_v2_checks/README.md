# amend-v2 checks (amendment batch 2026-10-04: SI-44 X15, SI-45, CR-05 rule side, SI-46, manifest v3)

All results below are for commit 54aeefc (clean checkout, git worktree), run locally at nice 19 with
OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=1, KMP_AFFINITY=disabled, CUDA_VISIBLE_DEVICES="" (`run_suites.sh <checkout> <out> <suite>`;
each `results/<suite>.txt` starts with the HEAD, the dirty count and the exact command).

| suite | env | command (see run_suites.sh) | result | lead's count at eb444fe |
|---|---|---|---|---|
| stage (incl. the real e2e) | wcd-gpu | `PYTHONPATH=src WCD_SRC=src STAGE_E2E_FIT_PY=scvi-api STAGE_E2E_SCORE_PY=wcd-kbet R_HOME R_LIBS PREPPED_DIR pytest tests/stage` | 62 passed | 62 |
| scvi | scvi-api | `WCD_SRC=src PREPPED_DIR WCD_REQUIRE_DATA=1 pytest tests/scvi` | 110 passed, 1 skipped (scvi_adv unconditioned, by design) | 104 + 1 skipped |
| x13 + scoring | wcd-kbet | `R_HOME R_LIBS PREPPED_DIR PYTHONPATH=src pytest tests/x13 tests/scoring` | 29 passed | 29 |
| other | wcd-gpu | `PYTHONPATH=src pytest tests --ignore=tests/{stage,scvi,x13,scoring,jhpce}` | 311 passed | 269 |
| jhpce | wcd-gpu | `PYTHONPATH=src pytest tests/jhpce` | 36 passed | 36 |

New tests (pytest --collect-only): scvi +6 = tests/scvi/test_kl_warmup.py (6); other +42 = tests/prereg/test_manifest_v3.py
(8) + test_provenance_rule.py (7) + test_si45_edge_extension.py (6) + test_rules_unchanged.py (9) + test_mutation.py (28
collected, 16 at eb444fe: 6 new mutants x 2 = 12).

Mutation checks of the new verifiers (`results/mutation_results.txt`; each mutant applied to a copy at 54aeefc, then reverted):

| mutant | target | test file | outcome |
|---|---|---|---|
| kl_stock_becomes_complete | 'stock' gets n_epochs_kl_warmup = max_epochs | tests/scvi/test_kl_warmup.py | killed (5 failed) |
| kl_complete_ignored | 'complete' passes nothing | tests/scvi/test_kl_warmup.py | killed (3 failed) |
| kl_not_forwarded | fit_adversarial_scvi drops n_epochs_kl_warmup | tests/scvi/test_kl_warmup.py | killed (4 errors: check_kl_warmup refuses) |
| rules_note_changed | freeze_matched_lambda.py frozen-manifest note | tests/prereg/test_rules_unchanged.py | killed (5 failed) |
| rules_tie_rule_changed | R4 ties -> larger lambda | tests/prereg/test_rules_unchanged.py | survived: the synthetic scores hold no exact tie; the tie rule is covered by the r4_tie_to_larger_lambda mutant of tests/prereg/test_mutation.py |

Other outputs: `results/manifest_v3_report.json` (scripts/manifest_v3_report.py; reproduces docs/manifest_v3_*.csv and
docs/x15_stock_equal_followups.csv byte for byte), `results/make_tags_check_v3_*.txt` (tags check before and after
regeneration), `results/wall_time_report_*.txt` (stock regenerated; scib fails at eb444fe too).
