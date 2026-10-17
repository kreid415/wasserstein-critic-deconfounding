---
name: fail-loud-pipelines
description: "Fail-loud coding standard and gates for experiment runners, launchers, manifests, Nextflow/SLURM pipelines and metric code. Load BEFORE writing or launching a multi-config / multi-seed / multi-hour run, and whenever you write run_*.py, a launcher, a .done marker, a resume key, a config-to-CLI mapping, an output filename scheme, a log or CSV parser, a verification script or test, or a conda/GPU environment for a job. Also load when something 'finished' but results are NaN, missing, duplicated, overwritten, or produced by the wrong file/column/env, or when a fix 'had no effect'. Provides fl_lint (static scan for silent-failure patterns), fl_args_consumed, fl_unique_outputs, fl_completion_gate, fl_columns, fl_env_guard, fl_mutation_check. Triggers: silent failure, swallowed exception, .done marker, resume skipped, all-NaN embeddings, output overwritten, config key ignored, wrong column, parse_known_args, vacuous test."
---

> **Helper loading (Claude Code).** `kernel.py` in this folder is plain Python with no Claude Science API. Load it from a Python session or script with
> `p = ".claude/skills/fail-loud-pipelines/kernel.py"; exec(compile(open(p).read(), p, "exec"), globals())`; the helpers are then defined in that namespace.

# Fail-loud pipelines
Applies: before writing/launching multi-config/seed/hour runs; when writing runners, launchers, `.done`, resume keys, config->CLI maps, output names, parsers, tests, envs; when a run "finished" but is NaN/missing/duplicated/overwritten/wrong file|column|env, or a fix had no effect.
Standard: every failure raises, OR becomes a failure row AND non-zero exit AND blocked `.done`. Helpers (kernel.py) raise a specific exception with `.result` dict; with `strict=False` they return it.

Helpers: `fl_lint(paths, fail_on="medium")` FLLintError, raises on findings >= fail_on (severities high/medium/low), run after every edit (runner/launcher/.nf/.sh/.R); `fl_args_consumed(config_or_argv, parser, ignore=[...])` FLArgsError (ignore = keys only the launcher uses); `fl_unique_outputs(rows, path_template_or_fn, swept_keys, reserved_paths=)` FLUniqueOutputsError; `fl_env_guard(require_cuda=True, packages={name: version}, imports, env_vars, not_under_home, require_thread_caps)` FLEnvError; `fl_columns(df_or_adata, names, values=)` FLColumnsError, for anything you did not create; `fl_completion_gate(expected, found_dir, results_csv=, expected_rows=, done_marker=, allow_unchecked=)` FLCompletionError, BEFORE `.done`; `fl_mutation_check(check_fn, good_input, bad_inputs)` FLMutationError, for every test, CI gate, regex check, smoke test and "reproduces exactly" comparison; `fl_rules()` lists rules; `fl_exc("FLLintError")` gives class.

Pre-launch order (paste outputs in notebook/run log):
1. `fl_lint("src/ experiments/")`: zero findings at medium or above (blocking), or each remaining one carries a reasoned `fl: allow`.
2. `fl_args_consumed` for EVERY runner each config row will be sent to, using its own `build_parser()` (never re-declare in launcher).
3. `fl_unique_outputs` on full expanded manifest; `reserved_paths` = other experiments' outputs; second call for resume/dedup key fn.
4. `fl_env_guard` inside the job (first batch-script lines), not only login node.
5. Smoke-run ONE config, `fl_completion_gate`; only then fan out.
6. Per shard, last statement: `fl_completion_gate(expected, out_dir, results_csv=..., done_marker=".done")`.

Rules:
R1 Failure is never a fallback value: no bare/`except Exception` unless re-raise/non-zero exit; narrowest type, only with identical-semantics recovery; per-config record-and-continue = failure row (`status`,`error`) + `sys.exit(1 if failures else 0)`.
R2 Exit status is the contract: `set -euo pipefail`; never `set +e`, `|| true`, `exit 0` after failure; `subprocess.run(check=True)`; no `2>/dev/null` on git/rsync/conda activate; verify state after "success" (git status, file exists, row count).
R3 Never pick file by position (`listdir()[0]`, `glob()[0]`, `head -1`, `next(iter())`): exact name, or `assert len(matches)==1` + validate structure (key exists, finite, shape).
R4 Every config key consumed or rejected: `parse_args` not `parse_known_args`; no argparse `type=bool` (`store_true` or strict `str2bool`); Groovy "false" is truthy (`== 'true'`/`.toBoolean()`); `fl_args_consumed` must pass for every stored config after flag renames; diff old vs new manifest flags before submit.
R5 Output, cache and resume keys encode every swept key; zero/None/empty levels still get a suffix; run `fl_unique_outputs`.
R6 `.done` only after gate passes: enumerated expected files (no globs), non-empty, finite, no failed rows; a stale marker from an earlier attempt is a bug (the gate reports it); merges assert counts (rows == configs x seeds, all experiment ids), never silently filter (`dropna`, `notna`, flat `glob`).
R7 Verify env: `fl_env_guard` at job start (CPU torch swap, missing optional deps, unexported env vars); cap BLAS/OpenMP threads in parallel CPU jobs, then confirm with the gate that each job wrote output.
R8 Durable/big outputs to scratch or another durable path, never `$HOME`, `/tmp`, ephemeral workspace; set `APPTAINER_CACHEDIR`, `TMPDIR`, `XDG_CACHE_HOME` in every shell that runs jobs, not only in the pipeline; decide up front which artifacts downstream metrics need (embeddings, latents, per-seed predictions) and persist them BEFORE a long wave starts; commit generating code (`experiments/` configs not gitignored; see `reproducible`).
R9 Parse by name: header names not `iloc`; `fl_columns`; exact-case labels; `pd.to_numeric(errors="raise")` for metric columns; log parsers use named groups + positive control; strip `#`/`//` comments before regex config parsing; one shared best-config groupby function.
R10 Metrics: assert inputs finite/in range and outputs finite; no `fastmath=True`; raise (no constant fallback) when denominator collapses or implementation missing; re-filter and assert no empty cells/rows after dropout/subsample/perturbation.
R11 Verifier must fail: every test, CI gate, regex check, smoke test and "reproduces exactly" comparison gets `fl_mutation_check` with real producer output as good_input + >=1 bad input rejected for the right reason; fixtures from producer convention; never chain conclusion with `&&` into `echo`; run from artifact's directory; print counted `n=...`.
R12 Fixes proven live: before reporting a fix done, show its effect (value read inside child process, task log, `nextflow config`/`-preview` on installed version; keys from installed schema, not web docs). Nextflow: no `executor` in process bodies (use profiles), no `errorStrategy 'ignore'`, no `ifEmpty([])` on channels feeding `collect()` (FL160-163).

Suppress: `# fl: allow FLxxx <reason>` (`//` in .nf), same line or comment-only line above; reason >=3 chars mandatory, bare allow = high-severity FL000 and no suppression; suppressed items in `result["suppressed"]`. Never disable a rule globally.
Gate specs: dicts `{"path","obsm":[..]}`, `{"path","keys":[..]}`, `{"path","min_rows":N,"finite":True}`; .h5ad finiteness without `anndata` = `unverifiable` = failure unless `allow_unchecked=True`.
Gate/guard failure: no blind retry; ISSUE entry via `nb_entry` with exception text, fix cause, delete stale marker, rerun shard.
Not caught: finite-but-wrong semantics (`rigor-review`), oversubscription, GPU exclusivity (two-job collision test), unobserved fixes (R12). Lint is heuristic (regex+AST): FL140/FL171 assume returncode/non-zero exits are visible in the same file; suppress with a reason when wrong for a line.
Interlocks: `reproducible`, `lab-notebook`, `rigor-review`, `cluster-autoscout`. 
Pointers: references/rules.md (lint rule table FL000+, severities, per-rule rationale), references/incident-map.md (incident ids and paraphrases per rule).
Details, rationale, examples, failure modes: references/full.md; read the relevant section when a check fails, a helper raises, or the case is ambiguous.
