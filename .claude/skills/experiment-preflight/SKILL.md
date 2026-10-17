---
name: experiment-preflight
description: "Pre-launch go/no-go gate for any computational experiment (ML, statistics, simulation, bioinformatics, NLP, imaging). Load BEFORE launching a run, sweep, wave, array job or benchmark, before adopting a new metric, and before quoting a cost, runtime or walltime. Triggers: 'ready to launch', 'kick off the sweep', 'submit the wave', 'is this metric valid', 'how long will this take', 'compare method A vs B', 'best lambda then report', 'train/test split', 'add a no-op baseline', 'benchmark the timing'. Checks split unit vs sample unit (leakage), winner's-curse selection, matched tuning across arms, unequal coverage in pooled means, identity baselines, sample-size collapse, metric vs source equation, timing confounds, pilot-based cost, launch flags vs intended settings, provenance filters. Writes experiments/ID/PREFLIGHT.md + preflight.json with a GO/NO-GO verdict; domain packs plug in."
---

> **Helper loading (Claude Code).** `kernel.py` in this folder is plain Python with no Claude Science API. Load it from a Python session or script with
> `p = ".claude/skills/experiment-preflight/kernel.py"; exec(compile(open(p).read(), p, "exec"), globals())`; the helpers are then defined in that namespace.

# experiment-preflight

Go/no-go gate run before compute is spent; writes `experiments/<id>/PREFLIGHT.md` + `preflight.json`. Domain packs expose `<prefix>_items()` (e.g. `sc_`). `pass` needs evidence, `n/a` needs a reason; a check that cannot fail is not a check; helpers raise a specific exception when `strict=True` (default) and return a result dict either way.

**Applies:** run/sweep/wave/array job costing more than a few CPU/GPU-h OR more than ~1 h wall; adopting/rewriting/porting a metric; before any cost/runtime/walltime/speed-up figure is given to the user or written into a request; before a table/claim ranking methods; re-run on any launch after manifest generator, grid, metric, split or filter changed.

**Protocol**
1. Write the plan first (pre-declaration, before any run or result), then create the record. Plan keys: question, primary_outcome (ONE metric), unit_of_analysis, split_unit, selection_rule, n_comparisons, multiple_comparison_correction, data_filters. `rec = pf_new(id, plan, root=".", extra_items=None)` refuses incomplete plan, never overwrites silently (`overwrite=True` archives old).
2. `pf_open(rec)` lists open items; read each item's id, severity, question, how_to_check.
3. Item's `applies_if` true: run its auto helper on REAL inputs (not toy), `pf_record_check(rec, "PF-03", helper(..., strict=False))`; `auto=None`: do check, `pf_answer(rec, id, "pass", evidence)`.
4. Condition false: `pf_answer(rec, id, "n/a", reason)`.
5. Fix every `fail`, re-run, re-record (failed answer stays in history).
6. `v = pf_verdict(rec, extra_items=)`; launch only if `v["go"]`, else clear `v["blocking"]`; state `v["major_fail"]` (non-blocking) in report.
7. `pf_save(rec)`; commit PREFLIGHT.* with run code; log GO/NO-GO as DECISION via `nb_entry` if lab notebook in use.

`v` keys: go, blocking (failed OR unanswered), unanswered (all severities), major_fail, minor_fail, plan_changed, n_items, n_pass, n_na.

**GO** iff every blocking item (generic + pack) is pass or n/a-with-reason AND plan hash equals the one at `pf_new` (plan edited after the fact = NO-GO; revise via a new record: `pf_new(..., overwrite=True)`, which archives the old one). Evidence >= 15 chars, not 'ok'/'done' (best: log/notebook path + number seen). Unregistered pack ids raise `InputError`; pack blocking items gate GO.

**Helpers** (PF item; fails when)
- `pf_check_plan(plan)` 01/02: key missing/placeholder, >1 primary, n_comparisons>1 w/o correction
- `pf_check_grouped_split(train_ids,test_ids,groups)` 03: unit or sample id on both sides; unmapped -> `MissingGroupError`
- `pf_check_id_collisions(df,group_col,source_col)` 04: unit id in >1 source
- `pf_check_nesting(df,cols,allow=None)` 05: a sets b, unacknowledged
- `pf_check_selection(selection_split_ids,report_split_ids)` 06: id in both
- `pf_check_matched_arms(arm_policies,varied=None)` 07: arms differ on a policy field not in `varied` (budget, selection rule, lambda policy, width)
- `pf_check_coverage(df,unit_col,item_col,value_col=None)` 08: item lacks valid result on a unit; returns `common_units`
- `pf_check_identity_baseline(fn,x,atol)` 09: fn(x) differs, reshapes or mutates x
- `pf_check_n_per_condition(df,cond_cols,min_n,n_col,axis_col,max_ratio)` 10: cell < min_n, empty grid cell, max/min n along `axis_col` > `max_ratio`
- `pf_check_metric_known_answer(fn,cases,atol)` 12: wrong/NaN/exception, or < 2 distinct answers
- `pf_check_timing_repeats(timings)` 16: needs >= 3 repeats; first call >> rest or wide spread
- `pf_estimate_cost(pilot_seconds,pilot_units,total_units,safety=2.0,pilot_settings,production_settings,requested_walltime_s,units_per_task)` 17: pilot != production settings (or not given), or walltime < estimate x safety
- `pf_check_manifest(manifest_rows,intended,expected_rows=None)` 19/20: intended flag absent/differs in any row, row count differs; warns on boolean strings
- `pf_check_replicates_differ(outputs)` 21: all identical
- `pf_check_inclusion(df,col,allowed,max_frac_outside=0.0)` 23: outside share > limit (NaN = outside)

Record helpers: `pf_items, pf_new, pf_load, pf_answer, pf_record_check, pf_verdict, pf_open, pf_save, pf_render, pf_validate_items`; exceptions `pf_exc("SplitLeakageError")` etc., all subclass `pf_exc("PreflightError")`. Manual items (`pf_items()`): PF-11 metric vs model structure; 13 estimator noise/null reference; 14 test vs control, circularity; 15 causal attribution; 18 co-scheduling from measured footprint; 22 output persistence/failure propagation; 24 config valid for actual input.

**Pitfalls the helpers cannot see:** helper only checks what it is given: pass production table/split/manifest/matrix. PF-03 `groups` = true unit from verified metadata (PF-04), not a column assumed to be the subject. Pilot (PF-17) must exercise late stages (early stop, final epoch, save/eval). Green PREFLIGHT is not a result check: verify outputs after the run (rigor-review).

**Packs:** own `kernel.py` with `<prefix>_items()` in `pf_items()` schema (id, area, question, severity blocking|major|minor, applies_if, how_to_check, auto); ids `SC-`/`IMG-`/`NLP-`, never `PF-`; auto helpers take real inputs, return `pf_result`-shaped dict (item, check, ok, summary; may call `pf_finish`, `pf_result`), raise `pf_exc("<Name>Error")` when strict and failing; register `extra_items=sc_items()` at creation or later: `pf_new(..., extra_items=)`, then `pf_answer`, `pf_verdict` (`pf_open`, `pf_record_check` also take `extra_items`); a registered pack can never be silently skipped (unregistered-id answers raise `InputError`). Example pack: skill `experiment-preflight-singlecell` (prefix `sc_`).

Interlocks: reproducible (`experiments/NN_name/` holds PREFLIGHT.* beside the run script), lab-notebook, rigor-review (post-hoc), cluster-autoscout (target choice happens after GO).

References (in this skill's references/): `full.md` original SKILL.md verbatim (incidents, rationale, examples); `checklist.md` per-item rationale for PF-xx; `domain-packs.md` pack skeleton. Item text and how-to-check: `pf_items()`. Original names tests `experiment-preflight_tests.py` (not in this skill dir).

Details, rationale, examples, failure modes: references/full.md; read the relevant section when a check fails, a helper raises, or the case is ambiguous.
