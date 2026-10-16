---
name: experiment-preflight-singlecell
description: "Single-cell, spatial and perturbation-omics domain pack for experiment-preflight (AnnData, scanpy, scvi-tools). Load it together with experiment-preflight BEFORE planning or launching an experiment that uses donor/patient-level labels, train/test or held-out splits, gene panels or HVG selection, foundation-model gene vocabularies, perturbation/noise/dropout axes, batch keys, disentanglement or kNN metrics, block-structured latents, or per-model averages pooled over datasets. Adds checklist items SC-01..SC-22 and scpf_ helpers (donor-grouped split and patient-ID collision, species ENSG vs ENSMUSG, zero-intensity no-op, cell-count collapse, obs keys by name, kNN validity, coverage-balanced pooling). Trigger on: preflight, sanity-check before launch, is this split leaky, is the panel human, is the baseline a no-op, before a cluster run on omics data."
---

> **Helper loading (Claude Code).** `kernel.py` in this folder is plain Python with no Claude Science API. Load it from a Python session or script with
> `p = ".claude/skills/experiment-preflight-singlecell/kernel.py"; exec(compile(open(p).read(), p, "exec"), globals())`; the helpers are then defined in that namespace. Load `experiment-preflight/kernel.py` first.

# experiment-preflight-singlecell

Domain pack for `experiment-preflight` (SC-01..SC-22, `scpf_` helpers). Load with it BEFORE planning/launching runs using donor/patient labels, splits, gene panels/HVG, FM vocabularies, perturbation/noise/dropout axes, batch keys, disentanglement/kNN metrics, block latents, pooled per-model means; or on "is this split leaky / panel human / baseline a no-op".

## Protocol
1. Load `experiment-preflight` + this skill (sidecar defines `scpf_*`).
2. `rec = pf_new(experiment_id, plan)`.
3. `for it in scpf_items(): print(it["id"], it["severity"], it["applies_if"])`; mark non-applicable `n/a` with reason, never skip silently.
4. `auto` items: run helper on the REAL objects (realized split, final panel, perturbed matrix), not the code meant to produce them; `pf_answer(rec,"SC-05","pass" if r["ok"] else "fail",scpf_evidence(r))`.
5. `auto=None` items: manual answer with evidence (file:line, table row, source excerpt) from the opened primary source, not recall.
6. `pf_save(rec)`; `v=pf_verdict(rec, extra_items=scpf_items())`; launch ONLY on `v["go"]`.

## Helpers (other optional kwargs: references/helper_api.md)
- SC-01/02 `scpf_check_donor_split(adata,label_cols,donor_col,cohort_col=,sample_col=,split_col=)`: leaky donor-level split; donor ids collide across cohorts; sample->several donors
- SC-03 `scpf_check_split_leakage(adata,group_col,split_col,cohort_col=)`: a group has cells on both sides (cohort_col= composite key guards cross-cohort id collision)
- SC-04 `scpf_check_split_fn(split_fn,group_ids,n_seeds=10,check_optimal=)`: groups straddle; fraction misses target; false optimality
- SC-05 `scpf_check_species(x,expected="human")`: >`max_other_frac` features other species
- SC-06 `scpf_check_vocab_coverage(var_names,vocab,min_frac=)`: low coverage
- SC-07 `scpf_check_noop_perturbation(raw,perturbed)`: intensity-0 != raw
- SC-08 `scpf_check_cell_counts(table,axis_col,group_cols=,max_ratio=2.0,extra_cols=)`: n_cells (or extra stat, e.g. extra_cols=["median_genes"]) varies > ratio; pass group_cols="dataset" or a multi-dataset table is pooled as one ratio
- SC-09 `scpf_check_empty_cells(x,min_genes=1)`: too few expressed genes
- SC-10 `scpf_check_gene_counts(n_genes_by_model,expected_n=,hvg_n=,n_vars_input=,n_hvg_stages=)`: counts drift; n_top_genes>=n_vars; HVG applied twice (each kwarg activates its sub-check, omitted = skipped; without expected_n only cross-model drift is tested)
- SC-11 `scpf_check_obs_keys(adata,required,expected_values=,used=)`: key missing, wrong values, or resolved to a different column (pass expected_values=/used= or only existence is checked)
- SC-13 `scpf_check_nesting(adata,factors)`: one factor determines another
- SC-14 `scpf_check_block_scoring(layouts,bases)`: block model scored per-dimension, or reverse
- SC-15 `scpf_check_block_pairing(blocks,factors,pairing,shared_blocks=)`: pairing not a name map; shared block credited (tested only if shared_blocks= passed)
- SC-16 `scpf_check_knn_validity(n_cells,k,kind=)`: trustworthiness k>=n/2; kNN k>=n; kBET k>n; pass n_cells per condition ({condition:n}) so every condition is checked
- SC-18 `scpf_check_pool_coverage(table,model_col,dataset_col,value_col=)`: models cover different datasets (value_col= adds rank_flip comparison)
- SC-19 `scpf_check_allowed_values(values,allowed,name=)`: a value (not just a missing field) is outside the allowed set
- SC-20 `scpf_scan_fallbacks(text_or_path)`: refit/fallback/swallowed error
- Manual: SC-12 batch conditioning vs disentanglement claim; SC-17 non-computable metric = NaN + status; SC-21 continuous factors not binned; SC-22 field-standard claim needs single-cell source.
- Support: `scpf_items`, `scpf_classify_gene`, `scpf_obs`, `scpf_names`, `scpf_matrix`, `scpf_finish`, `scpf_evidence(result)`.

## Rules
- Returns `{ok,item,check,problems,warnings,details}`; `strict=True` default raises ValueError (KeyError missing column); empty input raises (never passes). Inputs: AnnData-like (duck-typed `.obs`,`.X`,`.var_names`) or pandas/numpy/scipy. `strict=False` collects results for several checks; answer each item from its result. A failing helper raises before `pf_answer`: fix the cause, or run `strict=False` and answer `fail` with evidence so the verdict records it.
- `problems` block the item; `warnings` don't but go in evidence.
- `donor_split` without `split_col` only warns that a grouped split is required: pass realized split labels to get a measurement.
- Species check is heuristic (all-caps mouse symbols e.g. `H2-K1` read human): prefer Ensembl ids/orthology; state which.
- Thresholds are arguments: change them in the plan, not after a failure. Choose `max_ratio` in the plan before looking at results, and record it.
- Interlocks: `reproducible` (experiments/NN_name/ holds PREFLIGHT.md, preflight.json); `lab-notebook` (failed blocking item -> ISSUE); `rigor-review` is post-hoc.

Details, rationale, examples, failure modes: references/full.md; read the relevant section when a check fails, a helper raises, or the case is ambiguous. Also: references/checklist.md (per-item rationale, check recipes, incident mapping), references/helper_api.md (arguments, return fields, limits).
