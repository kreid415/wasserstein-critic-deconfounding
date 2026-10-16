# PREFLIGHT: preflight_sinkhorn_redo_20261016

**Verdict: GO**  (checked 2026-10-16T22:03:15Z; plan_sha 8601ded3835f45f4)

## Plan

- **question**: Can stage 1 of the Sinkhorn redo (refit of the 180 A1 Sinkhorn rows with the corrected Sinkhorn code, SI-51/SI-57) be launched from a clean checkout of tag prereg-tier12-v3, and what will it cost?
- **primary_outcome**: Stage 1 launch gate (no scientific outcome is computed here): 180 of 180 rows reach a final state (scored, diverged or nonfinite_latent) with one scorer provenance, run_stage completion gate exit 0. The registered primary outcome of the downstream analysis (R1, noise-calibrated batch and bio scores, PREREG section 2) is unchanged and is not touched.
- **unit_of_analysis**: one A1 Sinkhorn fit = one manifest row (task, decoder condition, penalty strength, seed)
- **split_unit**: not applicable: no held-out split is created or used by stage 1 (all cells are used for fitting; train_size 0.9 is the registered scVI validation split inside the fitter)
- **selection_rule**: none at this stage: the 180 rows are fixed by manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv; the penalty strength is selected later by R1 (rule order, SI-57), not here
- **n_comparisons**: 1
- **multiple_comparison_correction**: not applicable: one gate decision (launch / do not launch); no hypothesis test or ranking is made by stage 1
- **data_filters**: rows with experiment == A1 and arm == sinkhorn (tasks atac_small, immune, sim1; both decoder conditions; seeds 100-102; 10 penalty strengths); no cell filtering

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 8601ded3835f45f4) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; correction: not applicable: one gate decision (launch / do not launch); no hypothesis test or ranking is made by stage 1 |
| PF-03 | blocking | Unit of analysis | n/a | no probe, classifier or held-out split in stage 1: unsupervised scVI fit of all cells per task; the 0.9 train fraction is the registered scVI validation split inside the fitter |
| PF-04 | blocking | ID integrity | pass | 180 tags unique (prereg_rules.py:184 refuses duplicate manifest tags); three prepped files, one source each, fingerprint-verified; tag set equals the original A1 Sinkhorn tag set |
| PF-05 | major | Nesting of factors | n/a | no factors are compared in stage 1; factor nesting belongs to the later analyses (unchanged) |
| PF-06 | blocking | Selection bias | n/a | no best-of choice precedes any reported number in stage 1; penalty-strength selection is R1's registered rule, run later in rule order (SI-57) |
| PF-07 | blocking | Matched comparison | pass | 12 of 12 task x decoder blocks: all six adversarial A1 arms share n_latent 10, n_layers 1, n_hidden 128, zinb, batch_size 128, max_epochs (400; 239 for immune), train_size 0.9, adv_input mean, zstd 0, seeds 100-102 and the same 10-point lambda grid; only n_critic (0 for the critic-free arms mmd a... |
| PF-08 | blocking | Aggregation | pass | all 6 items cover all 180 units / real inputs: 180 Sinkhorn A1 rows of manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv at the tag |
| PF-09 | blocking | No-op baselines | n/a | stage 1 adds no perturbation knob; the lambda = 0 rows are arm none, are not refitted, and their code path is bit-identical to c88cce3 (59 of 59, NB-20261013-39) |
| PF-10 | blocking | Sample size | pass | 60 conditions, min n=3 / real inputs: 180 Sinkhorn A1 rows of manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv at the tag |
| PF-11 | major | Metric vs model structure | n/a | no metric is applied across different latent structures: every arm has one 10-dimensional latent block |
| PF-12 | blocking | Metric implementation | n/a | no metric is written or changed: the scorer is unchanged (7 of 7 stored latents rescored identically, NB-20261013-40); the Sinkhorn loss is a training loss, covered by tests/wcd_vae/test_divergences_v2.py (445 passed, 18 skipped at 8f223c8, NB-20261016-31; not re-run here, see not_checked) |
| PF-13 | major | Estimator noise | n/a | stage 1 produces fits, not an estimate; seed noise is handled by the registered 3 seeds and the R1 noise estimate downstream |
| PF-14 | major | Test vs control | n/a | no new design, probe or filter is proposed; the 180 rows are the registered A1 Sinkhorn rows |
| PF-15 | major | Attribution | n/a | no mechanism is attributed to any method in stage 1 |
| PF-16 | blocking | Timing confounds | pass | timing basis stated with its confounds: pre-fix production fit_seconds (D/tier12_v2/A1/ledger/A1__atac_small+immune+sim1.csv, 180 Sinkhorn rows, sum 307,735 s) were taken with 8 concurrent lanes; the 7,778 s hardest-regime pilot (NB-20261013-11) ran alone on another regime (A3 immune, standardise... |
| PF-17 | blocking | Cost estimate | pass | PROVISIONAL, with a stated gap: no end-to-end timing of a corrected-code A1 Sinkhorn fit at production settings exists (a pilot is forbidden here). Estimate = pre-fix production lane-seconds 307,735 + 7,368,240 training steps x measured added loss time per call (bench_gpu.csv, A1 regimes zstd 0: ... |
| PF-18 | major | Co-scheduling | pass | one RTX 3080 shared by 8 fit lanes + 3 scoring workers during fits (11 of 12 cores); same lane settings as the original A1 (experiments/A1_lambda_pilot/run_A1.sh); VRAM basis: 477 MiB per fit measured on another arm (experiments/bench_missing_arms_step_cost/PREFLIGHT.md:39) x 8 = about 3.8 GiB < ... |
| PF-19 | blocking | Launch manifest | pass | 180 rows carry all 15 intended flags / real inputs: 180 Sinkhorn A1 rows of manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv at the tag; known-bad mutations all caught (batch_size 64 in one row, 179 rows, max_epochs 400 in one arm, an mmd row): True |
| PF-20 | major | Count reconciliation | pass | 180 manifest rows = 180 tags in sinkhorn_a1_tags.txt (sha256 a19e4683640a) = 180 score files named *sinkhorn* in D/tier12_v2/A1/scores (tag sets equal); 6 task x decoder blocks x 10 lambda x 3 seeds = 180; PREREG.md:73 and CONSTRAINTS.md SI-51 name 180 A1 Sinkhorn fits |
| PF-21 | major | Replicates vary | n/a | no outputs exist yet to compare; seeds 100, 101, 102 are distinct in every cell (PF-10 check: 3 per condition); after the run, check that latents of different seeds differ (post-run check, not a launch gate) |
| PF-22 | blocking | Output persistence | pass | run_stage writes per-tag latents, models, scores, attempt logs and ledger; rerun skips valid outputs (resumable); outputs that break the contract are refused or quarantined; exit codes EXIT_PREFLIGHT 4 / EXIT_INFRA 5 / EXIT_GATE 6 (run_stage.py:97, :1263-1309); attempts: max 2 per tag then the st... |
| PF-23 | blocking | Provenance and inclusion | pass | all 180 items inside ['sinkhorn']; prepped inputs atac_small, immune, sim1 match docs/prepped_fingerprints_scib.json (fingerprint_prepped.py: OK x3); SI-54 refusal verified: prereg_rules.refuse_uncovered([c88cce3],['sinkhorn']) raises PreregError, so pre-fix Sinkhorn output cannot enter this stag... |
| PF-24 | major | Config vs input | pass | rows, settings and tags are those of the original A1 (manifest unchanged since c88cce3: git diff c88cce3..prereg-tier12-v3 -- manifests/ adds only extension and r1/r4a1/x1 files); solver constants read back at runtime from the fit env (scvi-api, scvi 1.4.2, torch 2.13, CUDA RTX 3080): SINKHORN_TO... |
| SC-01 | blocking | splits | n/a | no donor/patient labels and no train/test split in stage 1 (all cells fitted; task batch and cell-type labels are used only by the unchanged scorer) |
| SC-02 | blocking | splits | n/a | no donor/patient labels and no train/test split in stage 1 (all cells fitted; task batch and cell-type labels are used only by the unchanged scorer) |
| SC-03 | blocking | splits | n/a | no donor/patient labels and no train/test split in stage 1 (all cells fitted; task batch and cell-type labels are used only by the unchanged scorer) |
| SC-04 | major | splits | n/a | no custom split function |
| SC-05 | blocking | gene panel | n/a | gene panels are fixed inside the prepped files, fingerprint-verified (var_names included) for atac_small, immune, sim1; stage 1 builds no panel |
| SC-06 | major | gene panel | n/a | no foundation model or fixed gene vocabulary |
| SC-07 | blocking | perturbation | n/a | no perturbation, filtering or dropout step in stage 1 |
| SC-08 | blocking | perturbation | n/a | no perturbation, filtering or dropout step in stage 1 |
| SC-09 | blocking | perturbation | n/a | no perturbation, filtering or dropout step in stage 1 |
| SC-10 | major | gene panel | n/a | HVG selection is in the prepped files (fingerprint-verified); nothing is configured in stage 1 |
| SC-11 | blocking | inputs | n/a | obs keys come from the same prepped files and unchanged loader; the fingerprint covers obs batch, celltype and uns batch_key (OK x3) |
| SC-12 | major | claims | n/a | stage 1 makes no disentanglement or invariance claim |
| SC-13 | major | metrics | n/a | no covariates are scored against each other in stage 1 |
| SC-14 | blocking | metrics | n/a | single 10-dimensional latent block, no block models |
| SC-15 | major | metrics | n/a | no block-aware metric |
| SC-16 | blocking | metrics | n/a | no new kNN metric: the scorer is unchanged (NB-20261013-40) |
| SC-17 | major | metrics | pass | a metric that is undefined is not scored: run_stage.py score() refuses non-finite required metrics and out-of-range values (a pipeline fault), a non-finite latent is recorded as nonfinite_latent, a divergence as diverged (run_stage.py:455-493, docstring lines 9-22) |
| SC-18 | major | aggregation | n/a | no pooled per-model mean is made in stage 1; coverage is complete (PF-08: all six arms cover all 180 units) |
| SC-19 | major | inputs | pass | earlier-version outputs exist (D/tier12_v2/A1 holds 180 Sinkhorn latents and scores at c88cce3) so the launcher writes to a NEW directory D/tier12_v3/A1_sinkhorn (does not exist yet); the runner refuses c88cce3 Sinkhorn output anyway (SI-54, verified above) |
| SC-20 | blocking | runners | pass | scpf_scan_fallbacks: run_stage.py 0 hits; fit_paper_config.py 1 hit at line 50, which is a comment describing the I042 guard against silently ignored keys (read: lines 46-58), not a fallback; no substituting of another file, model or split found in the scan |
| SC-21 | minor | metrics | n/a | no disentanglement metric on continuous factors |
| SC-22 | minor | claims | n/a | no field-convention claim is made |


## Verdict and launch (added after the automated record)

**Verdict: GO** (blocking items all pass or n/a; major failures: none). Cost figures are provisional (PF-17).

- Where: local machine (RTX 3080, 12 cores, 94 GiB), host job on ssh:cs-local in tmux; the repository gives no other target (A1 and its extensions ran this way: experiments/A1_lambda_pilot/run_A1.sh, D/code/run_A1_host.sh).
- Rows: 180 (atac_small, immune, sim1 x decoder condition 0/1 x 10 penalty strengths 0.1 to 3000 x seeds 100-102); identical tag set to the original A1 Sinkhorn fits.
- Workers: 8 fit lanes, 3 scoring workers during fits, 8 after (as the original A1); VRAM basis 8 x about 0.5 GiB < 10 GiB.
- Expected wall time: about 13 h (12 to 15 h range; stress bound 30 h); about 103 lane-hours.
- Command: `tmux new -d -s a1sk 'bash /home/kendall/experiment_data/wasserstein-critic-deconfounding/code/run_A1_sinkhorn_redo.sh'`, which runs: `cd D/code/wcd_prereg-tier12-v3 && KMP_AFFINITY=disabled /home/kendall/.claude-science/conda/envs/scvi-api/bin/python scripts/run_stage.py --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv --experiments A1 --tasks atac_small immune sim1 --tags-file D/tier12_v3/manifests/sinkhorn_a1_tags.txt --out-dir D/tier12_v3/A1_sinkhorn --prepped-dir D/prepped_scib --fit-python <scvi-api python> --score-python <wcd-kbet python> --r-home <wcd-kbet>/lib/R --r-libs D/Rlib_kbet --expect-device 'RTX 3080' --fit-lanes 8 --fit-threads 1 --score-workers-during 3 --score-workers-after 8`
- Report-only start-up check at the tag: exit code 6 (EXIT_GATE=6: nothing is fitted yet; EXIT_PREFLIGHT=4 would be a refusal); 180 rows needs_fit, runner git 10a1ca5 clean.

### Launch conditions
1. Create the pinned checkout (not done by this preflight; D is read-only for it): git clone <live repo> D/code/wcd_prereg-tier12-v3 && git -C D/code/wcd_prereg-tier12-v3 checkout prereg-tier12-v3 (HEAD must be 10a1ca5874fbcdc033b148525b5e71d866306bdb, tag object ce056088ee, clean tree). The live repo D/code/wcd_final-review-fixes is now at 857234a, NOT at the tag: never launch from it.
2. Copy experiments/preflight_sinkhorn_redo_20261016/sinkhorn_a1_tags.txt to D/tier12_v3/manifests/ (sha256 a19e4683640aff7922d4124afff8f04d1a492ba83fbf9444d113ddc8f41b6ebf) and install run_A1_sinkhorn_redo.sh as D/code/run_A1_sinkhorn_redo.sh (outside the checkout). The launcher refuses on any hash, commit, dirty-tree, live-runner or GPU-memory mismatch.
3. Output goes to the NEW directory D/tier12_v3/A1_sinkhorn. Do not point stage 1 at D/tier12_v2/A1: its 180 Sinkhorn outputs were made at c88cce3, the runner refuses them (SI-54), and they are the pre-fix record.
4. GPU idle and no other run_stage process (the launcher checks), and the user's go.

### What stage 1 must produce for later stages
1. 180 scores/<tag>.csv and 180 latents/models in D/tier12_v3/A1_sinkhorn, all rows final with one scorer provenance, ledger A1__atac_small+immune+sim1__tags-a19e4683640a.done, failures.csv (divergence records, SI-85); every latent's git_sha = 10a1ca5 and git_dirty false.
2. R1 (freeze_a1_grid.py) reads score files and refuses duplicate tags (prereg_rules.py:306): D/tier12_v2/A1/scores still holds 180 pre-fix Sinkhorn files with the same tags. Stage 2 must therefore pass R1 the 918 non-Sinkhorn score files of tier12_v2/A1 plus the 180 new ones, not both directories; and tier12_v2/A1/failures.csv must be merged or filtered the same way. This was not tested here.
3. R4-a1 and R2/R3 then run in rule order on the redone R1 record; the 5 records in docs/prereg are overwritten in place (SI-87); a Sinkhorn iteration-cap hit stops the stage for the user's decision (SI-81 b).

### Not checked
1. Sinkhorn fit VRAM per lane (corrected code, float64) and the 8-lane speed-up for Sinkhorn fits: not measured (no GPU run allowed).
2. End-to-end time of one corrected-code A1 Sinkhorn fit: not measured; estimate rests on the loss-call micro-benchmark.
3. Sinkhorn unit tests re-run: attempted in env scvi-api (no scib) and wcd-gpu (crashed at import, no output); relied on NB-20261016-31 (445 passed, 18 skipped at 8f223c8).
4. run_stage's dirty-checkout and GPU-device checks do not run in --report-only; the tree was checked clean by git status and the device by a torch query (RTX 3080 visible).
5. fail-loud-pipelines fl_lint / fl_mutation_check not run: no Python code was written; the only new executable is a bash launcher (bash -n OK, never executed).
6. Stage 2 onward (R1, R1 extension, R4-a1, A3 rows, R2/R3): only the hand-off requirement stated; no input tested. X1/X13 score values: not opened.
7. Full-set fingerprint compare covers 3 of 8 prepped files (the 3 this stage uses).
