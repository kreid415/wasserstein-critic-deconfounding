# Code changes (always loaded)

Translated from the user's PONYTAIL and FEWEST-TESTS rules (full versions: skills `.claude/skills/ponytail`, `ponytail-review`, `ponytail-audit`; core rules follow).

1. Make the smallest change that fully solves the task. No new abstractions, options or files that the task does not need; delete rather than add where possible.
2. Correctness and reproducibility override "smallest change": never drop seeds, provenance, environment capture, independent checks, or tests of risky code to keep a change small.
3. Add the fewest tests that run every changed line and branch of non-trivial code. Extend or parametrize an existing test before adding a new one. A new test must cover or catch something no existing test does; otherwise delete it, fold it in, or mark it `# test-reason: ...`.
4. Never trim regression, result-validity or risky-code tests. Delete an existing test only after mutation testing and the user's OK.
5. Before calling code done: run, from the repo root with the thread guards (`KMP_AFFINITY=disabled OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL`), `python scripts/check_tests.py --base <commit before your change> --src scripts -- tests/prereg tests/stage tests/analysis tests/scvi tests/jhpce tests/test_legacy_retired.py tests/test_gate_task_split.py` (exit 0 and no WARN). On the workstation, with env `wcd-gpu` available, run the full `tests/` suite as well; a cloud VM cannot collect `tests/wcd_vae` (needs scib). Then re-read your own diff once as a reviewer would.
6. Fail loudly: raise on empty selections, non-finite values and missing keys; write failed rows with a `failed` flag (see `docs/claude/notes/verified-code-facts.md`).
7. Audits of the whole repository, or clean-up of deferred-work markers, run only when the user asks, through the `auditor` subagent and the `ponytail-audit` skill.
8. Pass these rules to any subagent that edits code, because subagents do not inherit them.
