# PREFLIGHT: bench_missing_arms_step_cost

**Verdict: GO**  (checked 2026-10-03T15:48:26Z; plan_sha 8045c40943bb29d0)

## Plan

- **question**: Single-lane ms per optimiser step of each new arm (discriminator_r1, discriminator_ref, reference/pooled with the stratified sampler, discriminator/reference/pooled/mmd with X3 importance weights) on immune, stock backbone, batch 128, local RTX 3080, re-measuring 4 existing arms as same-session controls; run4 (2026-10-03): re-measurement started by an automatic quiet gate (>= 60 s with < 2 busy CPU cores and no GPU compute process), with a 5-s host-load sampler, after run1 (contention), run2 ...
- **primary_outcome**: ms per optimiser step = (t_3epoch - t_1epoch) / (2 x steps_per_epoch) per repeat (cost_model.measure), median of 3 repeats
- **unit_of_analysis**: arm (one 3-epoch + 1-epoch fit pair per repeat)
- **split_unit**: none: timing benchmark, no evaluation split
- **selection_rule**: none: every arm is reported; the median repeat is appended to docs/throughput_rtx3080_stock_backbone.csv and all repeats are kept; run4 replaces the 8 provisional rows only if PF-16 passes for all 12 arms, otherwise the rows stay provisional; runs 1-4 are all reported
- **n_comparisons**: 1
- **multiple_comparison_correction**: none: one descriptive comparison per arm (new arm vs its same-session control), no hypothesis test
- **data_filters**: immune__scib.h5ad as distributed, var highly_variable; IW arms: X3-style depletion (largest batch, its two most abundant types present in >= 3 batches) at 50 percent

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 8045c40943bb29d0) |
| PF-02 | major | Multiple comparisons | pass | plan declares n_comparisons=1; correction: none: one descriptive comparison per arm (new arm vs its same-session control), no hypothesis test [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-03 | blocking | Unit of analysis | n/a | no classifier, probe or held-out evaluation: the run measures fit wall time only [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-04 | blocking | ID integrity | n/a | no merging of sources: one scIB file (immune__scib.h5ad) as distributed (SI-15) [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-05 | major | Nesting of factors | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-06 | blocking | Selection bias | n/a | no best-of selection: every arm is reported; the appended row is the median of 3 repeats, declared in the plan before the run [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-07 | blocking | Matched comparison | pass | all 12 arms share task, backbone, batch 128, seed, lambda, conditioning and adversary input (pf_check_manifest: 72 rows carry all 9 intended flags); n_critic follows the manifest design (critics 5, JS 1, MMD 0) [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-08 | blocking | Aggregation | n/a | no pooled means across datasets: one task (immune) [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-09 | blocking | No-op baselines | n/a | no perturbation knob with an identity level in this run; the IW identity (weights 1 at dose 0) is tested in tests/scvi/test_runner_and_manifest_missing_arms.py [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-10 | blocking | Sample size | pass | n differs only for IW arms (30344 vs 33506 cells after 50% depletion of 6325 cells); ms/step is per 128-cell step and steps_per_epoch uses each fit's own n_cells (cost_model.measure) [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-11 | major | Metric vs model structure | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-12 | blocking | Metric implementation | n/a | no metric adopted in this run; loss known-answer tests in tests/wcd_vae/test_missing_arms.py (159 passed at c3e3ad4; training-path files unchanged between f97f3f2 and main e3051e4, git diff empty) |
| PF-13 | major | Estimator noise | pass | timing noise is estimated from 3 repeats per arm and reported with the medians; controls show session drift against the existing CSV [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-14 | major | Test vs control | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-15 | major | Attribution | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-16 | blocking | Timing confounds | pass | launch gated: the launcher waits (max 90 min) for 3 consecutive 20-s windows with < 2 busy cores of 12 and no GPU compute process, logs them to run4/quiet_gate.log, then runs the run2 protocol (single lane, OMP 1, 3 interleaved repeats, 4 controls, GPU snapshot per fit) with the 5-s sampler; post... |
| PF-17 | blocking | Cost estimate | pass | per-step costs feed scripts/cost_model.py at production settings (same backbone, batch 128, n_critic as the manifest rows); this run is 72 short fits, single lane, no walltime request [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-18 | major | Co-scheduling | pass | single lane; one fit at a time; VRAM per fit below 1 GB (smoke: 477 MiB in use); other GPU users visible as utilisation in bench_log.csv [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-19 | blocking | Launch manifest | pass | [auto:pf_check_manifest] 72 rows carry all 9 intended flags |
| PF-20 | major | Count reconciliation | pass | 72 rows = 12 arms x 3 repeats x {3, 1} epochs (pf_check_manifest expected_rows=72): 72 rows carry all 9 intended flags |
| PF-21 | major | Replicates vary | n/a | the repeats are timing replicates of identical work (same seed by design), not statistical replicates [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-22 | blocking | Output persistence | pass | every fit writes latents/<tag>.npz (config with fit_seconds, git sha) to /home/kendall/experiment_data/wasserstein-critic-deconfounding/bench_missing_arms_step_cost/run4 plus bench_log.csv; completion gate on all 72 after the run |
| PF-23 | blocking | Provenance and inclusion | n/a | input is the scIB immune file as distributed; no reference-building inclusion filters [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| PF-24 | major | Config vs input | pass | stratified quota ceil(128/10) = 13 cells per batch vs smallest immune batch 1022 cells; IW keep fraction 0.5 lies in (0, 1); HVG set as distributed [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-01 | blocking | splits | n/a | no donor-level labels, probes or cohort pooling in a timing benchmark [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-02 | blocking | splits | n/a | no donor-level labels, probes or cohort pooling in a timing benchmark [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-03 | blocking | splits | n/a | no evaluation split; scvi-tools' internal 90/10 training split is part of the backbone and identical for all arms [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-04 | major | splits | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-05 | blocking | gene panel | n/a | no gene panel is built: the scIB highly_variable set as distributed is used by every arm [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-06 | major | gene panel | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-07 | blocking | perturbation | n/a | no perturbation axis in this run [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-08 | blocking | perturbation | n/a | no severity axis: one depletion level (50%) for the IW arms; n_cells recorded per fit [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-09 | blocking | perturbation | n/a | the depletion removes whole cells; no count-level thinning, so no cell or gene can become empty [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-10 | major | gene panel | n/a | HVG set as distributed, identical for all arms; not a result comparison [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-11 | blocking | inputs | pass | PASS SC-11 scpf_check_obs_keys () (immune__scib.h5ad obs: batch, celltype) |
| SC-12 | major | claims | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-13 | major | metrics | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-14 | blocking | metrics | n/a | no block-structured scoring: nothing is scored [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-15 | major | metrics | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-16 | blocking | metrics | n/a | no kNN-derived metric in this run [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-17 | major | metrics | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-18 | major | aggregation | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-19 | major | inputs | pass | fresh output directory run4 (does not exist before launch); run_bench.py refuses an existing latent (FileExistsError), so no stale timing is read |
| SC-20 | blocking | runners | pass | scpf_scan_fallbacks on scratch/testlogs/bench_smoke.log (2 smoke fits): no refit/fallback/swallowed-error lines [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-21 | minor | metrics | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
| SC-22 | minor | claims | n/a | timing benchmark: no factors, metrics, splits, foundation models, block latents or pooled scores [re-confirmed for run4: same manifest, same frozen worktree at e3051e4] |
