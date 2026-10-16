# SC checklist: rationale and recipes

Each item is returned by `scpf_items()` with the shared schema
(id, area, question, severity, applies_if, how_to_check, auto). Incident ids
refer to `workflow_incidents.csv`.

| Item | Sev | Auto | Incidents | Why it exists |
|---|---|---|---|---|
| SC-01 donor-level labels need grouped splits | blocking | donor_split | 238 | Response, treatment time and cancer type were constant per patient; a random 80/20 cell split put the same patient on both sides, so the headline probe results were retracted. |
| SC-02 composite, verified grouping key | blocking | donor_split | 239, 306 | A "donor-held-out" control used Batch as patient (36 levels, 14 patients); bare Patient ids collide across cohorts with different Response. A finding recorded earlier in the session was not consulted. |
| SC-03 realized split has no straddling group | blocking | split_leakage | 238, 239 | Check the saved split labels, not the splitting code. |
| SC-04 split function guarantees tested | major | split_fn | 304, 305 | A docstring claimed deterministic side assignment (an 85-cell group was in train 4/10 seeds) and then closest-possible fraction (counterexample sizes [16, 23, 16]). Subset-sum optimum is computed exactly. |
| SC-05 species of every feature | blocking | species | 342, 454, 456 | Panel built from a combined reference was ~29% mouse; a mouse-Ensembl dataset was configured as human; the user had to repeat the exclusion instruction three times. |
| SC-06 gene-vocabulary coverage per model | major | vocab_coverage | 454, 456 | Zero overlap between mouse Ensembl ids and a human-vocabulary model; report which datasets each model is excluded from, and why. |
| SC-07 zero intensity is an identity | blocking | noop_perturbation | 443, 430 | The baseline differed from the raw input in 2,494 entries; a dispersion default that should have been a no-op cut median genes/cell 37 -> 14 and left 15% of cells passing QC. Check every dataset and perturbation type, and check default shape parameters. |
| SC-08 sample size along the axis | blocking | cell_counts | 431, 430 | Cell counts fell from 52,055 to 375 along the noise axis, unreported, while kNN metrics compared fits on different n. Choose max_ratio in the plan. |
| SC-09 no empty cells after perturbation | blocking | empty_cells | 435, 460 | min_genes=0 let dropout empty cells; 7 of 13 runners raised and the retry-then-ignore wrapper left a silently missing column. Scan the full grid, not one setting. |
| SC-10 gene counts and HVG | major | gene_counts | 224, 429, 463 | Panels meant to have 1000 genes arrived with 992/994; n_top_genes=2000 on a 1000-gene input silently fell back; HVG re-applied per stage; hvg == panel size disabled another fix's tripwire. |
| SC-11 keys by name | blocking | obs_keys | 47, 451 | `iloc[:, 2]` returned donor ids instead of condition; a smoke test picked the first column with 1 < nunique <= 60 (Age instead of Batch) and reported success. |
| SC-12 conditioning vs disentanglement | major | none | 299 | scVI/LDVAE demote batch because the decoder is conditioned on the batch key. List each model's conditioning inputs from source before calling a result disentanglement; require an ablation without conditioning. |
| SC-13 nested factors | major | nesting | 266 | Batch determines Response, TreatmentTime and CancerType, so their "collisions" with Batch are arithmetic; the pair count fell from 10 to 7 informative pairs. |
| SC-14 native metric basis | blocking | block_scoring | 440, 352, 12, 382, 349 | Per-dimension DCI penalised block-structured models for their design; block metrics silently resolved to per-dimension for three models; a continuous discriminant-score correlation put a perfectly separating block model below null; Biolord was grouped with block models against its own source. Derive layouts from model code. |
| SC-15 pairing by name | major | block_pairing | 150 | Positional block-factor pairing credited the unowned shared subspace to Batch (score -0.1417 vs +0.1608). |
| SC-16 kNN parameter validity | blocking | knn_validity | 146, 448 | Trustworthiness needs k < n/2 and returned NaN silently; kBET required k = 12,500 with 4,999 cells and was reported as 1.0000. |
| SC-17 non-computable = NaN + status | major | none | 448, 344 | Infeasible cells must carry NaN and a status string; trace NaN columns to their real cause by reading the code path. |
| SC-18 coverage-balanced pooling | major | pool_coverage | 457, 369 | Foundation models run on 3 of 5 datasets while others run on 5; a bare groupby-mean flipped ranks in a synthetic test; a dataset exclusion was undisclosed. Pool over common datasets or show coverage. |
| SC-19 allowed values, not field presence | major | allowed_values | 461, 148 | A retired-type rejection tested for an absent key, but 15 stale files carried the key; 6 of 18 stale rows landed in the noise = 0 baseline slice. |
| SC-20 no silent fallback | blocking | scan_fallbacks | 453, 460 | RcppML projection failed on newer R, was swallowed, and the model refit on all cells (test-cell leakage). Also confirm every expected cell of the result grid exists. |
| SC-21 continuous factors stay ordered | minor | none | 390 | Binning a program score into classes was incoherent with the metric design. |
| SC-22 single-cell source for "standard metric" | minor | none | 383 | MIG/SAP/DCI were called standard in single-cell benchmarking; evidence supported only general ML. |

## Manual item recipes

- SC-12: for each model write the conditioning inputs found in its encoder/decoder source (file:line). If the covariate under claim is among them, the result is an integration effect.
- SC-17: grep result tables for constant 1.0 / 0.0 columns on cells whose parameter regime is infeasible; confirm a status column exists.
- SC-21: record the metric variant per factor type (categorical vs continuous) in the plan.
- SC-22: open the cited benchmark's full text and quote where the metric appears; otherwise phrase as borrowed from general ML.

## Extending the pack

Add an item to `SCPF_ITEMS` in `kernel.py` (next free SC-NN id), add the helper
(`scpf_` prefix, structured result, strict raising), a known-bad and a
known-good test fixture, and a row here.
