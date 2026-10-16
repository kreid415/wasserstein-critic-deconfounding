# Cluster B incident map (139 incidents: silent failures in pipeline code)

Source: workflow_incidents.csv, themes silent_error_suppression, config_parameter_silent_mismatch, misread_or_misparsed_output, faulty_verification_tooling, non_durable_output_storage.
Status: `tested` = reproduced in fail-loud-pipelines_tests.py; `rule only` = covered by a standard rule or lint rule but not separately tested; `out of scope` = not a pipeline-code defect (prose misread, document build, infra outage, design) - handled by other skills.

| id | sev | incident | control | status |
|---|---|---|---|---|
| 6 | medium | Fabricated PAGA-Spearman column in zero-concept list | - | out of scope |
| 9 | medium | Agent understated recompute scope of new metadata-injection process | - | out of scope |
| 10 | low | Unverified checksum-match claim for handoff_paper.tex | - | out of scope |
| 23 | low | Incorrect legend-ordering description for Patches model | - | out of scope |
| 24 | medium | Silent failure in flagship-datasets render loop | FL101 | rule only |
| 25 | medium | Palette tiling causes color collisions across 19 models | - | out of scope |
| 39 | medium | Widespread manual entry of unverified references due to arXiv/proxy outages | - | out of scope |
| 42 | medium | Network access loss forced fallback to placeholder citations | - | out of scope |
| 47 | high | Metadata column off-by-one mislabels cytokine condition as donor ID | R9 (parse by name) | rule only |
| 49 | medium | Mislabeled NMF as a VAE in LPS discovery summary | - | out of scope |
| 51 | medium | Figure caption claims per-cell recovery range from mean-only values | - | out of scope |
| 60 | high | Discarded experiment-generation code during idle cleanup, breaking reproducibility | R8 (persist before launch; commit generators) | rule only |
| 76 | high | run_uce.py silently selected wrong output file, producing all-NaN embeddings | FL110 + fl_completion_gate(obsm) | tested |
| 80 | medium | Config contained parameters silently ignored by runner scripts, with no effect on prior ru | fl_args_consumed | tested |
| 84 | high | $SCRATCH left unexpanded due to shlex.quote, breaking workdir convention | FL240 | tested |
| 86 | high | Decoder grouping dropped from best-adversarial-config computation | R9 (parse by name) | rule only |
| 106 | medium | Weakest-metric-family column wrong for 3 of 6 models | - | out of scope |
| 111 | high | Executor retry-hardening config key wrong, silently inert, twice | R12 (prove fix is live) | rule only |
| 112 | medium | Biolord known-latent attributes silently dropped during export | R10 (assert at both ends) | rule only |
| 113 | medium | Apptainer cache writing to home instead of scratch | fl_env_guard(not_under_home) + FL180 | tested |
| 114 | low | Plot panel silently rendering blank due to column name mismatch | fl_columns (names, values, shadowing) | tested |
| 117 | low | Incorrect verification flag set on MOFAcell | - | out of scope |
| 124 | medium | Silent git push failure due to branch divergence | R2 (verify state, FL140) | rule only |
| 130 | medium | Heat-table census undercounted RQ2 subspace figure | - | out of scope |
| 143 | medium | Empty Nextflow channel silently skips PNG publishing | FL161 | tested |
| 144 | medium | Path-input cache key never stabilized | R12 (prove fix is live) | rule only |
| 145 | low | Incorrect test assertions misdiagnosed as stratification defects | R11 (fl_mutation_check) | rule only |
| 149 | medium | Env-var leak between tests and missing mock attribute caused regressions | R12 (prove fix is live) | rule only |
| 151 | medium | CLI silently dropped layout parameter | R12 (prove fix is live) | rule only |
| 152 | high | Silent merge-conflict resolver committed conflict markers | FL120 | tested |
| 153 | medium | Proposed re-addition would have silently cut task timeout | R12 (prove fix is live) | rule only |
| 155 | low | Faulty debug probe produced false positive for pipeline compile failure | fl_mutation_check | tested |
| 158 | medium | Misread sinfo %F field, overstated GPU idle capacity ~2x | - | out of scope |
| 166 | low | Overstated non-gated remaining work by ~70% in own cost table | - | out of scope |
| 176 | medium | GPU silhouette patch silently no-op due to wrong import target | R12 (prove fix is live) | rule only |
| 178 | medium | Incomplete resume-key would have silently truncated E10 sweep | fl_unique_outputs | tested |
| 182 | medium | Image-embed fix used wrong identifier form, didn't actually fix rendering | - | out of scope |
| 183 | high | String-replace bug corrupted saved paper results section with garbled placeholder values | R9/R11 positive control | rule only |
| 193 | medium | Added audit check has a latent bug causing false failure, never re-run to confirm | R11 (fl_mutation_check) | rule only |
| 197 | low | Agent falsely claimed a document section was missing content that was actually present | R11 (fl_mutation_check) | rule only |
| 203 | medium | 'No build-log files' claim verified against the wrong/missing zip | R11 (fl_mutation_check) | rule only |
| 204 | medium | Table-parser regex failed on nested-brace tabularx column specs | R11 (fl_mutation_check) | rule only |
| 207 | medium | TikZ-figure converter silently dropped diagrams, mischaracterized as intentional | - | out of scope |
| 208 | high | Undefined-reference CI gate was blind to \ref warnings, so 'zero undefined references' was | fl_mutation_check | tested |
| 212 | low | Math-protection regex excluded newlines, corrupting escaped percent signs | R9/R11 positive control | rule only |
| 213 | low | Subagent overflow-check regex silently zeroed every measured width | fl_mutation_check | tested |
| 215 | low | Undefined-citation grep had no detection power | fl_mutation_check | tested |
| 216 | low | Shared kernel namespace caused mirrored toy examples to clobber notebook globals | - | out of scope |
| 217 | low | Conflated 'allowed citation key' list with 'defined in references.bib' | - | out of scope |
| 220 | medium | Missing hashlib import breaks new kNN cache | FL230 | tested |
| 226 | medium | NXF_TASK_CPUS never exported, threading fix silently inert | fl_env_guard(env_vars) | tested |
| 228 | medium | np.savez path bug caused silent write failures | fl_completion_gate (missing .npz) | rule only |
| 230 | medium | scipy sparse used without import, breaking PAGA cache | FL230 | tested |
| 231 | high | Cluster executor hardcoded in process bodies silently overrides local profile | FL160 | tested |
| 232 | low | Agent's own verification check used wrong path, falsely reported failure | fl_mutation_check | tested |
| 234 | medium | Pathway hypothesis doc overstates asymmetry as ~3x instead of ~2.05x | - | out of scope |
| 241 | high | DAG_Alignment_Spearman computed against an all-zero target due to misnamed columns | fl_columns (names, values, shadowing) | tested |
| 242 | low | Config key errors: wrong seeds key and undefined results_dir | fl_args_consumed | tested |
| 251 | medium | Docstring falsely claims clean vs perturbed reference give identical model rankings | - | out of scope |
| 254 | medium | Leak-ratio figures cited from wrong aggregation level (pair-means vs row-level) | - | out of scope |
| 256 | medium | MOFA+ block-identifiability trend mislabeled 'still rising' via flatness-only check | R11 (fl_mutation_check) | rule only |
| 259 | medium | Model-name filter bug silently dropped both VAE models from workflow figure | fl_columns (names, values, shadowing) | tested |
| 260 | medium | RcppML lift ranking inverted in slide annotation and text | - | out of scope |
| 270 | medium | Automated verification of fit_quality.py silently failed due to case-sensitive string mism | fl_columns (names, values, shadowing) | tested |
| 274 | high | git add silently skipped all 17 figure PNGs | R2 (verify state, FL140) | rule only |
| 280 | medium | Unverified claim that PROVENANCE.csv 'has no generator' | - | out of scope |
| 285 | medium | Inverted 'mostly mouse-determined' characterization of Infected factor | - | out of scope |
| 290 | medium | Direction-alignment claim inverted (scVI called 'least aligned' when it is most aligned) | - | out of scope |
| 301 | medium | Wrong-entity coordinate attribution in perturbation analysis | - | out of scope |
| 312 | medium | Backwards epistasis-direction interpretation | - | out of scope |
| 318 | low | Label-mismatch join produced blank PCA ceiling value | fl_columns (names, values, shadowing) | tested |
| 321 | medium | CSV baseline silently NaN'd | FL250 | tested |
| 323 | medium | Grep filter hid Harmony's own values, causing misattribution | R11 (fl_mutation_check) | rule only |
| 327 | medium | DRVI ranked second-best MIG despite lower score than ICA | - | out of scope |
| 329 | medium | iVAE misclassified as at-or-below PCA in committed memory | - | out of scope |
| 331 | high | Fit results lost by writing outside the granted durable path | FL180/FL181 + fl_completion_gate | tested |
| 334 | high | Multi-GPU dispatch doesn't enforce documented exclusivity | - | out of scope |
| 336 | medium | Damage report misidentifies which cancelled job actually ran | - | out of scope |
| 337 | medium | SLURM OR-constraint silently dropped job's CPU/memory request | - | out of scope |
| 338 | high | CPU and GPU model rosters silently thinned by unverified env builds | fl_env_guard(imports=...) | tested |
| 341 | high | Oversubscribed scoring job silently failed for 29/35 models | fl_env_guard(require_thread_caps) + fl_completion_gate | tested |
| 344 | low | Misattributed NaN column cause (iLISI vs MIG) in Leng results explanation | - | out of scope |
| 348 | medium | ContrastiveVI condition key misidentified as crispr_target instead of nperts | - | out of scope |
| 352 | medium | Block-wise DCI silently fell back to per-dimension for DIVA/CausalVAE/scCausalVI without a | R10 (assert at both ends) | rule only |
| 353 | medium | Per-seed CSV merge silently failed due to dataset-name key mismatch | fl_columns(values) | rule only (analog of 259) |
| 359 | medium | Auditor finding: two unrelated project-memory findings falsely marked 'addressed' | - | out of scope |
| 360 | high | Driver script invoked analysis scripts with nonexistent CLI arguments | fl_args_consumed | tested |
| 365 | medium | Clustering NMI claim contradicted agent's own retrieved data | - | out of scope |
| 376 | medium | sign_flips column referenced in outline/panel instructions but never saved to CSV | fl_columns on saved CSV | rule only |
| 388 | high | Agent reversed which cluster had run out of compute credits | - | out of scope |
| 398 | medium | Subagent repeatedly misreported repo as clean despite scratch files inside it | R11 (fl_mutation_check) | rule only |
| 399 | high | scGen latent extraction silently failed with 0% finite values | FL101 + fl_completion_gate | tested |
| 400 | medium | scINSIGHT2 comparator counts were actually comparisons against scINSIGHT | - | out of scope |
| 406 | medium | Misread sim2 winner, overclaimed barycenter uniqueness | - | out of scope |
| 410 | high | Metrics filename collision between SCVI and LinearSCVI | fl_unique_outputs | tested |
| 411 | high | False .done markers hide incomplete/OOM'd seeds | fl_completion_gate (enumerated expected, no globs) | tested |
| 413 | medium | Quoting bug broke recovery shards | R2 (verify state, FL140) | rule only |
| 424 | medium | Miscalculated critic completion rate inflated progress projection | - | out of scope |
| 425 | high | Divergence log parser took wrong token, dropping all divergence data | fl_mutation_check positive control | tested (analogous) |
| 428 | medium | Lambda-parsing regex silently dropped decimal lambda configs | R9 (parse by name) | rule only |
| 434 | high | Earlier config fixes silently lost due to gitignored directory | R8 (persist before launch; commit generators) | rule only |
| 435 | medium | min_genes=0 allowed dropout perturbation to create empty cells | R10 (assert at both ends) | rule only |
| 436 | low | Agent's own smoke test falsely flagged cloned environment as broken | fl_mutation_check | tested |
| 450 | high | Grid-count error inflated every cost/runtime estimate by ~1.78x | R9 (parse by name) | rule only |
| 451 | high | local_smoke.sh derives wrong batch_key (Age instead of Batch) | R9 (parse by name) | rule only |
| 452 | medium | Verification gate reported false PASS due to silently failed conda activate | fl_mutation_check | tested |
| 453 | high | RcppML silently refits on full data instead of held-out projection (test-cell leakage) | FL101 + fl_completion_gate | tested |
| 455 | medium | Agent incorrectly claimed twice that experiments/ was gitignored, blocking commits | R11 (fl_mutation_check) | rule only |
| 458 | high | Test fixture fabricated filenames matching verifier's own convention, hiding a real plotti | fl_mutation_check | tested |
| 459 | medium | CLI flag --allow_empty_metrics false parsed as truthy string in Groovy, silently disabling | FL131 + FL163 | tested |
| 460 | medium | Binomial dropout thinning can empty cells, crashing 7 of 13 runners and silently dropping  | FL162 | tested |
| 461 | high | Stale var_shift files load silently into baseline metrics due to narrow rejection conditio | R10 (assert at both ends) | rule only |
| 462 | medium | UCE taxonomy slug mismatch leaves model miscategorized | R9 (parse by name) | rule only |
| 465 | medium | Earlier audit misdiagnosed root cause of a swallowed scVI exception | - | out of scope |
| 469 | high | LISI metric silently degrades under fastmath numba build | FL150 | tested |
| 472 | medium | Subsampling documented as 'stratified' but implemented as plain random | - | out of scope |
| 486 | medium | Overstated convergence plateau in saved convergence_check.md artifact | - | out of scope |
| 493 | low | Misdiagnosed walled job status for E4/E9 heavy tasks | - | out of scope |
| 497 | medium | Figure caption numbers mismatched actual rendering order | - | out of scope |
| 498 | medium | Images delivered as PDF despite explicit PNG request | - | out of scope |
| 500 | medium | Pandoc→typst backend silently dropped table/column formatting | - | out of scope |
| 501 | low | Sized math delimiters leaked internal typst symbol names | - | out of scope |
| 511 | medium | Stray editor markers leaked into reduced-text document | - | out of scope |
| 512 | medium | Recurring LaTeX/unicode escape artifacts leaking into titles and author names | - | out of scope |
| 519 | high | E8 miscount included a protocol-excluded dataset in the live schedule | - | out of scope |
| 522 | high | Launcher marks shards done despite recorded failures | FL170/FL171 + fl_completion_gate(results_csv) | tested |
| 524 | high | Environment fork silently downgraded to CPU-only torch | fl_env_guard(require_cuda) | tested |
| 525 | high | Embeddings not persisted for a 24-hour wave, forcing costly retraining | R8 (persist before launch; commit generators) | rule only |
| 530 | high | E8 batch-count sweep embeddings overwrote each other | fl_unique_outputs | tested |
| 531 | high | Manifest rewrite silently dropped the 500-epoch flag, reverting to 150 | fl_args_consumed(required args) | partially tested (generic required-arg case) |
| 532 | medium | Comparison join omitted swept keys, producing a cross-join | R5 (join keys) | rule only |
| 533 | low | Column/method name collision silently produced NaN | fl_columns (names, values, shadowing) | tested |
| 534 | high | Wrong row selector deleted healthy latents and left a CSV headerless | fl_completion_gate (enumerated expected, no globs) | tested |
| 535 | high | E4 positional reference resolution silently overwrote E1 latents | fl_unique_outputs | tested |
| 538 | medium | WCD_WORKERS fix not exported, silently inert in production | fl_env_guard(env_vars) | tested |
| 540 | medium | Artifact caption claimed 251 configs, file actually had stale 158 rows | R8 (persist before launch; commit generators) | rule only |
| 541 | high | Three of nine experiments silently dropped from merged results | fl_completion_gate (enumerated expected, no globs) | tested |
| 542 | high | Latent embeddings written to ephemeral scratch and lost | FL180/FL181 + fl_completion_gate | tested |
| 549 | medium | Direction-of-effect error on iLISI under conditioning | - | out of scope |

Counts: {'tested': 48, 'rule': 39, 'oos': 52} of 139