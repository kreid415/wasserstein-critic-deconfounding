# Code check of tag prereg-tier12-v1 (129142b) before the JHPCE launch

Reviewer: Rigor Reviewer (sub-agent), 2026-10-03. Review only; nothing in the repository or in `tier12/` was changed, run or read. Clone at the tag in the reviewer workspace; prepped inputs read read-only; checks run CPU-only at `nice -n 19` with 1 thread, one at a time, plus one GPU micro-test (<50 MB).

## Verdicts

- **JHPCE launch: NO-GO.** Blocking: CR-02 (the runner cannot launch the planned filler rows or X13), CR-03 (no production job script; the tagged fit code has never run on an L40S). Must be fixed before the follow-up stage that runs on JHPCE: CR-01 (X3/X8 PCR reference), CR-04 (X3 dose confounded with N), CR-06 (X13 divergences). No defect was found in the fit or scoring code that would bias X1.
- **A1 fits and scores so far: valid (yes).** No defect invalidates them; see 'A1 validity'. Lead not messaged (no A1-invalidating defect).
- Findings: 0 critical, 3 high, 3 medium, 8 low; 21 prior items re-checked (table below).

## Findings (severity-ordered)

| ID | Severity | A1? | Stages | Title |
|---|---|---|---|---|
| CR-01 | high | no | X3, X8 | PCR_batch reference differs between the full-data and subsampled scoring paths; X3 changes PCR definition between dose 0 and doses 50-100 on pancreas and sim2 |
| CR-02 | high | no | X1 (none/scvi_adv fillers), X13 | The tagged runner cannot launch the planned JHPCE filler work (X1 none/scvi_adv, X13) or score/gate X13 CPU latents |
| CR-03 | high | no | X1, X3, X6, X7, X8, X12, X13 on JHPCE (atac_large, immune_hum_mou, lung, pancreas, sim2) | No production JHPCE job script, and the tagged fit code (missing-arm code) has never run on an L40S node |
| CR-04 | medium | no | X3 | X3: depletion dose is confounded with total cell count and training steps on 3 of 4 tasks |
| CR-05 | medium | yes | A1, A2, A3, X1, all follow-ups (rule inputs) | Score rows carry no scorer provenance and the rule scripts cannot detect scores from another host or scorer commit |
| CR-06 | medium | no | X13 | X13 scANVI/sysVI fits have no non-finite-loss guard: a divergence becomes an infrastructure error that stops the stage |
| CR-07 | low | yes | A1, A2, A3, X1 (R1/R4 paired differences) | PREREG 'Pairing' claim is inaccurate across arms: same seed gives the same init but not the same minibatch order as lambda=0 |
| CR-08 | low | yes | A1 (immune trajectory, kBET), X1, all scored stages | Silent finite fallbacks inside scib 1.1.7 enter B and C unflagged |
| CR-09 | low | yes | all fitted stages | Design settings held in code defaults are not recorded in the latent provenance; docstring says there are none |
| CR-10 | low | no | reproducibility of the repository | Rejected legacy pipeline still in scripts/ and runnable (N9 only partly closed) |
| CR-11 | low | no | A2, A3 (R2/R3), documentation | Pre-registration text inconsistencies and one undefined case |
| CR-12 | low | yes | A1, X1, X13 | Declared but undisclosed differences between arms and baselines (fair-comparison statement) |
| CR-13 | low | no | analysis/reporting | Promised diagnostics and sensitivity code are not implemented |
| CR-14 | low | no | X3, X8 | Scorer subset branch does not check for unknown cell names |

### CR-01 (high): PCR_batch reference differs between the full-data and subsampled scoring paths; X3 changes PCR definition between dose 0 and doses 50-100 on pancreas and sim2

- **Location (at the tag):** scripts/score_scib_native.py:91-93; scripts/prep_scib_task.py:31,174; scib 1.1.7 metrics/pcr.py:161-183 (wcd-kbet); scripts/build_paper_manifest.py:221
- **What is wrong:** Full-data fits: the prepped file has obsm['X_pca'] but no uns['pca'] (all 8 prepped files), so scib.metrics.pcr() takes its 'recompute PCA' branch on adata.X with ALL genes; the HVG X_pca that prep_scib_task.py documents as the 'PCR reference for scoring' is never used. Subsampled fits: the scorer first runs sc.pp.pca(pre, n_comps=50, mask_var='highly_variable'), which writes uns['pca'], so scib uses an HVG-only reference. In X3, n_cells equals the full task size for pancreas and sim2, so the dose-0 latent has as many cells as the prepped file (full-data path, all-gene reference) while the depleted doses 50/80/95/100 have fewer cells and send the scorer into its subset branch (HVG reference): the PCR_batch definition, and hence B, changes inside the dose series for every arm. X8 and X3 immune_hum_mou use the HVG reference throughout, so their PCR_batch is not on the scale of X1. atac_small is unaffected (all features are training features).
- **Evidence / what was checked:** Check review/checks/check_pcr_reference.py on the prepped files (read-only): pcr_before pancreas: all-gene 0.482 vs HVG 0.191 (ratio 2.52); sim1: all-gene 0.521 vs HVG 0.275 (ratio 1.90); immune: all-gene 0.764 vs HVG 0.564 (ratio 1.35); sim2: all-gene 0.832 vs HVG 0.717 (ratio 1.16); atac_small: all-gene 0.124 vs HVG 0.124 (ratio 1.00). An uncorrected 10-PC HVG embedding scores PCR_batch (all-gene ref vs HVG ref) pancreas 0.559 vs 0.000; sim1 0.259 vs 0.000; immune 0.179 vs 0.000. scib-pipeline (theislab, main) scripts/metrics/Snakefile:6 passes the scenario's original file as -u and scripts/metrics/metrics.py:79 reads it without reduce_data, i.e. scIB's own reference is the all-gene uncorrected data, which is what the full-data path does. The e2e and kBET tests build prepped files with sc.pp.pca (uns['pca'] present), so the production path is untested.
- **Stages affected:** X3, X8. **Affects A1:** no.
- **Suggested fix:** Make the subsample branch use the full-data convention: drop the HVG sc.pp.pca call and let scib recompute the PCA on the subset's full X (or precompute that all-gene PCA with its variance explicitly for both paths). Do NOT add an HVG uns['pca'] to the prepped files: that would change A1/X1 scores and depart from scib-pipeline. Correct the prep and scorer docstrings; add a test on a prepped file without uns['pca']. Fix before any X3/X8 scoring.

### CR-02 (high): The tagged runner cannot launch the planned JHPCE filler work (X1 none/scvi_adv, X13) or score/gate X13 CPU latents

- **Location (at the tag):** scripts/run_stage.py:260-270 (select_rows), 273-284 (unfittable), 326 (device check); scripts/fit_paper_config.py:184-186
- **What is wrong:** run_stage.py selects rows by experiment and task only, and its preflight refuses the whole stage if any selected row holds a placeholder. Every X1 task block contains g1..g6 / 'A2' / 'A3' rows, and every X13 task block contains CPU rows whose adv_input 'na' unfittable() flags as a placeholder, so neither block can run before R1-R3, although PREREG sec 0 and GATE_JHPCE.md plan to run these fillers (on JHPCE) while A1-A3 are in progress. X13 CPU rows are also refused by fit_paper_config.py, and their latents (written by run_cpu_baselines.py without a 'gpu' field) fail Outputs.latent's device check, so the runner can neither score nor gate them; no other harness records their outcome.
- **Evidence / what was checked:** Dry runs at the tag (review/outputs/dryrun/dryrun_log.txt, --dry-run): A1 atac_small+immune+sim1 -> 1098 rows needs_fit (exit 0); X1 pancreas -> exit 4, 360 rows refused (lam g1..g6, adv_input 'A2', zstd 'A3'), so its 15 filler rows (none x 2 decoders, scvi_adv) cannot run either; X13 pancreas -> exit 4, the 16 CPU rows refused (adv_input 'na'), blocking its 35 scANVI/sysVI rows; X3 (192) / A2 (54) refused for matched placeholders, as expected. GATE_JHPCE.md (main) l.27 and l.309: fillers (X1 none/scvi_adv, X13) run in JHPCE GPU gaps.
- **Stages affected:** X1 (none/scvi_adv fillers), X13. **Affects A1:** no.
- **Suggested fix:** Add a row filter to run_stage.py (--arms or --tags-file, recorded in the ledger) or commit derived filler manifests (rows copied verbatim, tags unchanged, SHA-256 recorded) before the JHPCE launch; route X13 CPU rows to run_cpu_baselines.py and extend the runner (or a sibling) to score and gate CPU latents (record device 'cpu' in their config, expect-device per row kind).

### CR-03 (high): No production JHPCE job script, and the tagged fit code (missing-arm code) has never run on an L40S node

- **Location (at the tag):** cluster/jhpce/ (no production script); docs/jhpce/GATE_JHPCE.md (main) l.400; cluster/jhpce/gate_gpu.sh
- **What is wrong:** The JHPCE path has env build, staging and gate scripts but no production job. The gate jobs ran commits 6675f8c (A100) and 2f29004 (L40S); GATE_JHPCE.md states that the missing-arm code (adversarial.py, alignment.py, critic.py changes; discriminator_losses.py, sampling.py new) was run by no gate job. tests/scvi at the tag (72 passed locally) and the X3/X6/X7/X8 code paths have therefore not been exercised with torch 2.13.0+cu126 on an L40S. Required contents of the job script are listed in the report (section 'JHPCE production job script').
- **Evidence / what was checked:** ls cluster/jhpce at the tag: build_envs.sh, compare_freeze.py, env.sh, env_versions.py, gate_gpu.sh, gate_job.sh, kbet_fingerprint.R, setup_job.sh, stage_scib.sh; GATE_JHPCE.md (main) l.400 caveat quoted above.
- **Stages affected:** X1, X3, X6, X7, X8, X12, X13 on JHPCE (atac_large, immune_hum_mou, lung, pancreas, sim2). **Affects A1:** no.
- **Suggested fix:** Write the production job (requirements in the report) and run, as its first step on the allocated L40S node, nvidia-smi/env_versions.py and WCD_SRC=src pytest -q tests/scvi at the exact commit to be fitted; launch only after a dry-run preflight of the derived manifest and the user's go (SI-29).

### CR-04 (medium): X3: depletion dose is confounded with total cell count and training steps on 3 of 4 tasks

- **Location (at the tag):** scripts/build_paper_manifest.py:221 (n = min(20000, TASKS[t][0])); scripts/fit_paper_config.py:136-149 (subsample: cap only when len(idx) > n_cells)
- **What is wrong:** n_cells equals the full task size for atac_small, pancreas and sim2, so the post-depletion subsample to n_cells never triggers and every dose removes cells; max_epochs stays 400, so higher doses also get fewer gradient steps. Only immune_hum_mou is held at 20,000 cells. docs/PAPER_PLAN.md:185 promises 'composition-matched subsamples'. Each dose also draws its non-depleted cells with a different RNG seed (40000 + dose), adding subsample noise to the dose contrast.
- **Evidence / what was checked:** review/checks/check_x3_x8_subsample_sizes.py with the tagged subsample() on obs-only AnnData from the prepped files: atac_small 11,270->8,402 cells (-25.4%), steps/epoch 80->60; pancreas 16,382->14,409 cells (-12.0%), steps/epoch 116->102; sim2 19,318->17,872 cells (-7.5%), steps/epoch 136->126; immune_hum_mou 20,000->20,000 cells (0.0%), steps/epoch 141->141. X8 is at fixed N (16,000 / 6,000 / 3,000; sim2 V=16 2,992).
- **Stages affected:** X3. **Affects A1:** no.
- **Suggested fix:** Before the X3 stage (dated PREREG amendment, X3 rows rebuilt and re-tagged): set n_cells per task to a common N not above the dose-100 count (e.g. 8,402 / 14,409 / 17,872 / 20,000) so every dose has identical N and steps, and draw nested subsamples from one dose-independent permutation.

### CR-05 (medium): Score rows carry no scorer provenance and the rule scripts cannot detect scores from another host or scorer commit

- **Location (at the tag):** scripts/score_scib_native.py:109-113; scripts/prereg_rules.py:195-212 (read_scores), 240-245 (outcomes: kBET-seed check only)
- **What is wrong:** Each score CSV holds tag, META, metrics, kbet_seed, kbet_r_calls and score_seconds only: no scorer git SHA, scib/scanpy/numba/pynndescent versions, host or CPU vector width. G4 showed that identical latents score differently on AVX-512 and AVX2 CPUs, so 'all scoring local' is enforced only by procedure; R1-R4 would silently accept a mixed score directory. A1's score rows have the same gap, but A1's provenance is recoverable from attempts/<tag>.jsonl (host, runner git SHA per attempt), so A1 stays valid; affects A1 as a provenance gap only.
- **Evidence / what was checked:** score_scib_native.py l.109 row = dict(tag=..., **META) ... l.111 kbet_seed/kbet_r_calls; prereg_rules.outcomes l.240-245 checks kbet_seed uniformity only; GATE_JHPCE.md G4 FAIL (5/12 deterministic metrics within 1e-6).
- **Stages affected:** A1, A2, A3, X1, all follow-ups (rule inputs). **Affects A1:** yes (documentation / disclosure only; fits and scores stay valid).
- **Suggested fix:** Write scorer git SHA + dirty flag, hostname, CPU model/flags, NUMBA_CPU_NAME and package versions into every score row; make read_scores/outcomes refuse mixed values within one rule input (as for kbet_seed).

### CR-06 (medium): X13 scANVI/sysVI fits have no non-finite-loss guard: a divergence becomes an infrastructure error that stops the stage

- **Location (at the tag):** scripts/fit_paper_config.py:248-273 (fit_baseline uses stock SCVI/SCANVI/SysVI training plans)
- **What is wrong:** The NonFiniteLossError guard lives only in WassersteinAdversarialTrainingPlan. In fit_baseline a NaN loss either produces a NaN latent (recorded as nonfinite_latent by the runner) or a torch.distributions ValueError, which exits 1 and is retried as an infrastructure error until the stage stops (EXIT_INFRA). PREREG sec 1's 'diverged' outcome is therefore unavailable to the neural baselines (e.g. sysVI at z_distance_cycle_weight 50).
- **Evidence / what was checked:** fit_paper_config.py __main__ catches only NonFiniteLossError (l.287-292); fit_baseline calls vae.train / m.train on stock plans (l.258-273); run_stage.fit_done maps any other non-zero exit to 'infrastructure' (l.713-715).
- **Stages affected:** X13. **Affects A1:** no.
- **Suggested fix:** Add a Lightning callback (or a thin plan subclass) for scANVI/sysVI that applies the same per-step finiteness check and distribution-validation mapping, so their divergences are recorded as 'diverged'.

### CR-07 (low): PREREG 'Pairing' claim is inaccurate across arms: same seed gives the same init but not the same minibatch order as lambda=0

- **Location (at the tag):** docs/PREREG.md:31; scripts/scvi_adversarial_plan.py:235 (head created inside model.train); scvi-tools 1.4.2 dataloaders/_ann_dataloader.py:114 (RandomSampler)
- **What is wrong:** The adversary head's weight initialisation draws from torch's CPU generator after the scVI module is built and before the first epoch's RandomSampler permutation, so arms with a head (discriminator, reference, pooled, barycenter) see a different minibatch order from the 'none' fit of the same seed; MMD/Sinkhorn (no head) match. R1/R4 differences against lambda=0 are still valid (no bias; sigma is the empirical SD of the paired differences) but pair less than claimed.
- **Evidence / what was checked:** review/checks/check_seed_pairing_minibatch_order.py (CPU toy, seed 100, 1 epoch): generator init identical for all arms; first-epoch minibatches equal to 'none': 0/5 for barycenter, discriminator, pooled, reference; 5/5 for mmd; identical within each arm across lambda 1 and 10.
- **Stages affected:** A1, A2, A3, X1 (R1/R4 paired differences). **Affects A1:** yes (documentation / disclosure only; fits and scores stay valid).
- **Suggested fix:** Correct the wording in a dated PREREG amendment / Methods (shared: module init and train/validation split; minibatch order shared only within an arm). Do not change RNG handling while A1 runs.

### CR-08 (low): Silent finite fallbacks inside scib 1.1.7 enter B and C unflagged

- **Location (at the tag):** scib 1.1.7 metrics/trajectory.py:64-66 (RootCellError -> 0); metrics/kbet.py:152-154, 185-189 (score = 1), 255-259 (R error -> NaN per label, then nanmean l.200)
- **What is wrong:** Trajectory conservation returns 0 when no root cell is found; kBET forces a label's rejection rate to 1 when fewer than 75% of its cells are in large components or neighbours are insufficient, and drops a label (NaN) on an R error. These are finite and pass PREREG's 'required values' check, so a scorer failure can look like collapse (C) or like batch retention (B). Behaviour is scib's own, so A1 scores remain valid as scIB defines them.
- **Evidence / what was checked:** Installed source read in wcd-kbet (paths as in location); score_scib_native.py wraps kBET_single for seeding but records no fallback counts.
- **Stages affected:** A1 (immune trajectory, kBET), X1, all scored stages. **Affects A1:** yes (documentation / disclosure only; fits and scores stay valid).
- **Suggested fix:** Record per row whether trajectory hit RootCellError and how many kBET labels were skipped or forced to 1 (wrap get_root / kBET as seed_kbet does); report these flags with R1/R4 inputs.

### CR-09 (low): Design settings held in code defaults are not recorded in the latent provenance; docstring says there are none

- **Location (at the tag):** scripts/fit_paper_config.py:4 (docstring), 226-228 (extra.get defaults); scripts/scvi_adversarial_plan.py:162, 500 (critic_lr, critic_betas, adv_hidden, disc_lr defaults)
- **What is wrong:** adv_width (128), adv_lr (critic 1e-4, discriminator 1e-3), bary_iter (10), critic betas (0, 0.9) and lambda_GP 10 are code defaults; the npz config stores only the manifest row and the git SHA, so these values are recoverable only by reading the code at that SHA.
- **Evidence / what was checked:** A1 rows carry extra '{}' (barycenter '{"bary_iter": 10}'); cfg = dict(row=r, reference_name, n_cells, n_batches, fit_seconds, iw_keep_fraction, provenance).
- **Stages affected:** all fitted stages. **Affects A1:** yes (documentation / disclosure only; fits and scores stay valid).
- **Suggested fix:** Write the resolved plan kwargs into cfg and correct the docstring. Documentation-level for A1.

### CR-10 (low): Rejected legacy pipeline still in scripts/ and runnable (N9 only partly closed)

- **Location (at the tag):** scripts/score_final_config.py:21 (batch score adds raw pcr with +1); scripts/scvi_final_manifest.tsv; run_jhpce_pilot.sh; score_parallel.sh; score_baselines_parallel.sh; run_final_baselines.py; scvi_adv_fit.py
- **What is wrong:** The new path (build_paper_manifest -> run_stage -> fit_paper_config -> score_scib_native) is complete, but the rejected scorer, manifest and drivers remain without a retirement guard or README note.
- **Evidence / what was checked:** All seven files present at the tag; none contains a retired/deprecated guard (grep).
- **Stages affected:** reproducibility of the repository. **Affects A1:** no.
- **Suggested fix:** Move to legacy/ with a raise-on-run guard (or delete) and name the new path in README / Code availability.

### CR-11 (low): Pre-registration text inconsistencies and one undefined case

- **Location (at the tag):** docs/PREREG.md:140; CONSTRAINTS.md:118,125 (SI-19, SI-20); scripts/decide_a2_a3.py:41,48
- **What is wrong:** PREREG sec 8 still says no runner writes the failures table (run_stage.py writes failures.csv, l.847-865). SI-19/SI-20 still state the pre-SI-27 thresholds (>= 4 of 6, >= 7 of 10 cells); PREREG sec 3 and decide_binary use ceil(2n/3) of 12/20 (8/14). If an R4-a1 window neighbour is edge_unresolved (no A1 fit at that lambda), decide_a2_a3.py stops ('default half lacks window points'); PREREG does not define this case.
- **Evidence / what was checked:** Quoted lines; prereg_rules.decide_binary l.562 need = math.ceil(2*n/3).
- **Stages affected:** A2, A3 (R2/R3), documentation. **Affects A1:** no.
- **Suggested fix:** Dated amendment: update sec 8, annotate SI-19/SI-20 as superseded by SI-27, and pre-specify the edge_unresolved case for R2/R3 before A2/A3 start.

### CR-12 (low): Declared but undisclosed differences between arms and baselines (fair-comparison statement)

- **Location (at the tag):** scripts/fit_paper_config.py:250-273; scvi-tools 1.4.2 external/sysvi/_model.py:150-159; scripts/scvi_adversarial_plan.py:243-260 (optimizers), 391-408 (adversary loop on one minibatch)
- **What is wrong:** Epochs, batch size, latent size and backbone are equal within each task (checked on all 6,846 rows). Remaining differences: scANVI adds min(10, max(2, round(E/3))) epochs after the scVI pretraining and trains at train_size 0.9 (comment says 'on all cells'; scib uses 1.0); scvi-tools' SysVI.train forces n_epochs_kl_warmup = 0 while scVI arms warm up over 400 epochs; critics take 5 Adam(1e-4, betas 0/0.9) steps on the same minibatch per generator step (WGAN-GP Alg. 1 line 4 samples fresh data each critic iteration), the discriminator 1 step of Adam(1e-3, eps 0.01, wd 1e-6).
- **Evidence / what was checked:** Manifest fairness check (review: backbone fields constant within experiment x task x subsample; X1/X13 epochs = scvi heuristic 400/246/239/82/400/400/400/94); sources quoted in the report.
- **Stages affected:** A1, X1, X13. **Affects A1:** yes (documentation / disclosure only; fits and scores stay valid).
- **Suggested fix:** State these in Methods (they are design choices or tool defaults); fix the scANVI comment.

### CR-13 (low): Promised diagnostics and sensitivity code are not implemented

- **Location (at the tag):** scripts/scvi_adversarial_plan.py training_step (logged keys); manifests (no N8 rows); no SI-38 analysis script
- **What is wrong:** SI-38 sensitivity (C without trajectory) has no code yet (known). N8's warm-up-complete sensitivity has no manifest rows; N1's masking gap / sigma^2/Var(mu) and N6's per-minibatch n_k are not logged (models are saved, so sigma^2 can be computed post hoc); N13's clustering grid (scib 1.1.7: 10 Leiden resolutions 0.2-2.0, clustering.py:110-111) is not stated in the docs.
- **Evidence / what was checked:** Logged keys: z_rms, adv_loss, adv_gp, adv_w1, adv_acc, adv_grad_norm, adv_div, kl_weight, ELBO terms; grep of docs for 'resolution' finds no clustering-grid statement.
- **Stages affected:** analysis/reporting. **Affects A1:** no.
- **Suggested fix:** Add the SI-38 script before X1 analysis; decide whether the N8 sensitivity is still planned (add rows by amendment) or drop it explicitly; compute sigma^2/Var(mu) from saved models.

### CR-14 (low): Scorer subset branch does not check for unknown cell names

- **Location (at the tag):** scripts/score_scib_native.py:92
- **What is wrong:** pre[pd.Index(pre.obs_names).get_indexer(names)] maps an unknown name to -1 (the last cell) silently; only the subsequent batch-order assertion can catch it.
- **Evidence / what was checked:** Code read; the full-data path never enters this branch.
- **Stages affected:** X3, X8. **Affects A1:** no.
- **Suggested fix:** Assert (idx >= 0).all() and that the indexer is unique before subsetting.

## A1 validity

A1's fit path matches its sources: the critic loss is the negated Kantorovich-Rubinstein estimate (critic.py:89, plan l.417), the gradient penalty is WGAN-GP's two-sided (||grad||-1)^2 at uniform interpolates with lambda 10 (critic.py:154-214), critics use Adam(1e-4, betas 0/0.9) with n_critic 5 and the discriminator one step (SI-01); 'none' is stock scVI (bit-identity test passes) and the unconditioned decoder never receives the batch (scvi-tools 1.4.2 FCLayers.forward zips n_cat_list with the covariates, strict=False, nn/_base_components.py:165, so the trailing label argument is ignored and n_batch = 1 adds no input; tests/scvi passes). The manifest is fair (equal backbone, epochs, batch, latent within each task) and A1 has 1,098 rows on seeds 100-102. Scoring calls scib 1.1.7's own metrics with a seeded kBET; PCR_batch for full-data fits uses the all-gene uncorrected PCA, which is what scib-pipeline does, and is identical for every A1 row of a task. prereg_rules implements PREREG sec 1-4 (84 rule tests pass). The A1-touching findings are documentation or disclosure (CR-07 pairing wording, CR-08 scib's own fallbacks, CR-09 provenance of code defaults, CR-12). Condition: the CR-01 fix must leave the full-data scoring path unchanged (do not add an HVG uns['pca'] to the prepped files), otherwise A1 must be rescored.

## Correctness against primary sources (checked, no defect unless a finding is cited)

- **WGAN (arXiv 1701.07875v3).** Eq. 2 (p. 6): W = sup over 1-Lipschitz f of E_Pr[f] - E_Ptheta[f]. critic.py `ReferenceWassersteinLoss` returns -(mean over active heads of E_{P_k}[C_k] - E_{T_k}[C_k]) (l.56-89); the generator adds +lambda x that estimate (plan l.417, 432). Correct sign and form.
- **WGAN-GP (arXiv 1704.00028v3).** Algorithm 1 (p. 4): "default values of λ = 10, ncritic = 5, α = 0.0001, β1 = 0, β2 = 0.9". Implemented: lambda_gp 10 (critic.py:158), n_critic 5 (manifest), Adam(1e-4, (0, 0.9)) (plan l.257). Two-sided penalty (p. 5): implemented as (||g||-1)^2 (critic.py:210) at x_hat = eps z + (1-eps) t, eps ~ U[0,1] (l.200, 204); partners drawn from each head's own target. No batch norm in the critic (p. 5 recommends omitting it): the head is Linear-ReLU-Linear-ReLU-Linear (adversarial.py:50-52). Deviation: Alg. 1 line 4 samples fresh real data for every critic iteration; the plan reuses one detached minibatch for all 5 steps (l.391-408) - known item L6, disclose (CR-12).
- **DANN (arXiv 1505.07818v4).** Domain classifier vs feature extractor via gradient reversal (p. 1-2), lambda ramp lambda_p = 2/(1+exp(-10p)) - 1 for the feature extractor (p. 21), half of each 128-sample batch from each domain (p. 22). The discriminator arm instead uses scvi-tools' fool loss (cross-entropy to uniform over other batches; plan l.291-297 = scvi-tools AdversarialTrainingPlan.loss_adversarial_classifier l.651-667) at a fixed lambda; a declared design choice shared with scvi-tools. X8's stratified sampler cites p. 22 correctly.
- **R1 penalty (Mescheder et al. 2018, arXiv 1801.04406v4).** Eq. 9 (p. 7): R1 = (γ/2) E_pD[||∇D||²], on the true data distribution only. discriminator_losses.r1_penalty applies (γ/2) mean ||∇ D_{b_i}(z_i)||² over all minibatch cells with the one-vs-rest log-odds (signed off as SI-22; there is no 'real' side in batch alignment).
- **scvi-tools 1.4.2 (installed source).** AdversarialTrainingPlan (train/_trainingplans.py:535): classifier n_hidden 32, kappa = 1 - kl_weight, generator step before the classifier step, adversary on z = inference_outputs['z'] (l.689, as PREREG cites). KL warm-up is epoch-based, n_epochs_kl_warmup 400 (l.596; _compute_kl_weight l.32-73), so immune ends at 238/400 = 0.595 (N8). get_max_epochs_heuristic = min(round(20000/n x 400), 400) (model/_utils.py:60), matching every manifest max_epochs. scvi.settings.seed sets cudnn.deterministic and seed_everything (_settings.py:153-167). get_latent_representation returns the posterior mean in eval mode (dropout off). The plan's D-before-G order differs from scvi-tools' G-before-D; both are standard.
- **scIB (Luecken et al. 2022) and scib 1.1.7.** Batch metrics PCR (pcr_comparison, scaled and clipped at 0, pcr.py:83-91), batch ASW, graph iLISI (use_rep default 'X_emb' = the scorer's key), graph connectivity, kBET (1 - mean rejection, kbet.py:200); bio metrics NMI/ARI on Leiden at 10 resolutions 0.2-2.0 (clustering.py:110-111), cell-type ASW, isolated-label F1/ASW, cLISI, cell cycle, trajectory. scib-pipeline computes metrics against the scenario's original file (scripts/metrics/Snakefile:6, metrics.py:79), i.e. an all-gene PCA reference when the file carries no uns['pca'] - what the full-data path does (CR-01 concerns the subsample path). scib's kBET wrapper sets no seed and R's kBET samples test cells; the scorer's set.seed wrapper is called through the module global (seeded calls counted; tests/scoring passes).
- **POT 0.9.7.** ot.lp.free_support_barycenter (lp/_barycenter_solvers.py:160): fixed point X <- sum_i w_i diag(1/b) T_i Y_i with exact EMD plans T_i and squared-Euclidean cost (Cuturi & Doucet 2014, theta = 1); barycenter.py passes uniform support weights, equal batch weights and numItermax = 10 (SI-14).
- **MMD / Sinkhorn.** mmd_batch_others is the biased (V-statistic) MMD² with an average of 5 Gaussian kernels exp(-||x-y||²/(2σ²)), σ = sqrt(d) x {0.25, 0.5, 1, 2, 4}, batch k vs all other cells. sinkhorn_batch_others is the debiased S_ε = OT_ε(a,b) - ½OT_ε(a,a) - ½OT_ε(b,b) via dual potentials, ε-annealing without gradient and one differentiable extrapolation step, cost |x-y| (p = 1), blur 0.05 sqrt(2d): the standard debiased construction (Feydy et al. 2019; paper and geomloss source not retrieved here, so this is a code-level check only). CUDA check: values and gradients bitwise reproducible.
- **Harmony / Scanorama (installed harmony-pytorch 0.1.7, scanorama 1.7.4; papers not fetched).** harmonize(theta = 2.0 default, random_state 0) and scanorama defaults (dimred 100, knn 20, seed 0) match SI-34/SI-35's tool defaults; run_cpu_baselines reproduces scib.integration.harmony / scanorama on HVG-subset normalised X.
- **scANVI / sysVI (installed scvi-tools; papers not fetched).** scANVI: scib.integration.scanvi uses min(10, max(2, round(E/3))) epochs only when max_epochs is None and train_size 1.0 (integration.py:292-307); the fitter uses the same epoch rule with train_size 0.9 (SI-10). sysVI: SysVI.train forces n_epochs_kl_warmup = 0 (sysvi/_model.py:150-159); z_distance_cycle_weight reaches SysVAE via plan_kwargs (module forward reads loss_kwargs, _module.py:360); the cycle batch is a random other batch per cell (_module.py:460-483).

## Fair comparison (user directive 2026-10-02, SI-04/SI-10/SI-11)

Checked on all 6,846 manifest rows and on the executed code: within every experiment x task x subsample, decoder, n_latent, n_layers, n_hidden, likelihood, batch_size, max_epochs, train_size and counts are constant (X12 varies them by design); X13 scANVI/sysVI share X1's epochs per task (400/246/239/82/400/400/400/94). In code, every arm shares scvi-tools' generator optimiser (Adam 1e-3, eps 0.01, wd 1e-6), the KL warm-up, the train/validation split (numpy RandomState(seed)) and the posterior-mean latent. Declared differences: adversary optimiser and steps per source (discriminator 1 step Adam 1e-3; critics 5 steps WGAN-GP Adam), adversary width 128 for both, critic-free arms have no adversary. Remaining disclosure items: CR-12. Arm comparisons within a task never mix machines (SI-17/SI-39).

## Pre-registration fidelity (PREREG sec 0-9, CONSTRAINTS SI-01..SI-40)

- R1 (prereg_rules.r1_block / r1_family / grid_indices): Δb, Δc, σ via median-unbiased seed SDs, δ = max(0.01, 3σ/√n), floor, collapse (n_fail ≥ 1 or Δc ≤ -0.10 with ties), batch peak, upper end, low/high edges, whole-family 2-point extensions for ≤ 2 rounds, 6 evenly spaced indices ⌊i_lo + j(i_hi - i_lo)/5 + ½⌋ or padding above first: all as written. Unregistered stop: fewer than 3 grid points with ≥ 2 paired seeds (fail-loud, low risk).
- R4 (r4_stage): b0, M_a over failure-free points, b_common = min over the 6 arms, b* = b0 + 0.5(b_common - b0), closest failure-free point with ties to the smaller λ, 1-point extensions for ≤ 2 rounds only for needed cells, variant-arm inheritance (BASE_ARM), check_no_x1_reuse: as written.
- R2/R3 (decide_binary, decide_a2_a3): ⌈2n/3⌉ evaluable (8 of 12, 14 of 20) with ≥ 1 per task x decoder, mean D > ROPE, mean D > 0 per task x decoder, no excess failures, masking ⇒ mean; one pooled decision over both decoders (SI-27): as written. Gaps: CR-11.
- Failure statuses (diverged / nonfinite_latent), infrastructure reruns, completion gate (every row final, none both scored and failed, none scored twice, single git SHA): implemented in fit_outcome.py and run_stage.py. B = mean of 5 batch metrics; C = mean of the task's bio metrics (8 immune tasks, 7 pancreas/lung, 6 ATAC/sims); CC_TASKS and TRAJ_TASKS agree with the prepped files' uns/obs.
- Row counts and seeds: A1 1,098 = 3 x 2 x 3 x (1 + 6 x 10); A2 108; A3 180; X1 3,000; seed sets disjoint (check_seed_separation passes on the tagged manifest). SI-01 holds (discriminator always 1 step; X6 n_critic 10 only on pooled). SI-38's analysis code does not exist yet (known; CR-13).

## Fail-loud behaviour

Placeholders refused (fitter and runner preflight), unknown extra keys refused, NaN losses and non-finite parameters raise NonFiniteLossError and are recorded, non-finite latents recorded, contract-breaking outputs quarantined, dirty trees and mixed SHAs refused, manifest snapshot hashed. Gaps: CR-06 (X13 baselines), CR-08 (scib fallbacks), CR-14 (unchecked indexer), CR-09 (code defaults). No fit or score timeout by default (`--fit-timeout-s 0`): set one on JHPCE.

## Reproducibility

Seeds fixed per row; kBET seeded (identical on re-score, tests/scoring); git SHA, dirty flag, scvi/torch versions and GPU name in every latent; latent SHA-256 in every score row; rule records hold input SHA-256s, constants and git SHA; env specs and freezes in cluster/envspec. Gaps: CR-05 (no scorer provenance in score rows), CR-09. Same-GPU determinism: critic loss values vary in the last bits on CUDA (index_add_ atomics) while gradients were bitwise identical in 30 repetitions (tiny test), so refits should match closely but bitwise identity is not guaranteed.

## Prior issues (docs/code_issues_for_review.csv; its status column is stale)

| ID | State at the tag | Evidence |
|---|---|---|
| N9 | partly fixed | New path at the tag: build_paper_manifest.py writes every knob as a column; fit_paper_config.load_row refuses missing columns/empty fields (l.100-112); run_stage scores only via score_scib_native.py; each npz stores row, git SHA, dirty flag, scvi/torch versions, GPU. Not done: score_final_config.py (raw PCR +1, l.21), scvi_final_manifest.tsv, run_jhpce_pilot.sh, score_parallel.sh etc. still present without guard (CR-10). |
| N1 | fixed by design (A2 decides); diagnostic missing | adv_input column ('mean' default, 'sample' in A2), plan l.358 z_used = mu if mean else z; A2/R2 implemented (decide_a2_a3.py, masking rule). The promised per-run masking gap and sigma^2/Var(mu) logging is not implemented (CR-13); models are saved so it can be computed post hoc. |
| C1 | fixed | score_scib_native.py calls scib.metrics.metrics(pcr_=True) -> pcr_comparison (scaled, 1 = best; scib pcr.py:83-91); B uses PCR_batch with + sign. Old scorer still in repo (CR-10). |
| C2 | fixed (superseded by SI-15) | fit_adversarial_scvi sets a.X = a.layers['counts'] (plan l.522); prep 'scib' mode keeps scIB's counts layer as distributed and records per-batch integer status in uns (SI-15); the integer guard exists only in 'valid' mode. |
| N4 | fixed | critic.py:75-78 pooled target = mean over cells of OTHER batches; alignment.py _batch_masks uses other cells; X7 reference analyses on K>=3 tasks (pancreas 9, sim1 6, atac_small 3 batches). |
| N3 | fixed (rebuilt) | barycenter.py: free-support W2 barycenter via POT 0.9.7 ot.lp.free_support_barycenter (exact EMD fixed point, _barycenter_solvers.py:160), cold start from the minibatch, 10 iterations (SI-14), detached target; adversarial.py anchors = None. |
| N2 | fixed | plan _adv_view (l.262-269): per-dimension standardisation with minibatch posterior-mean mean/SD, gradients kept; A3 tests it. |
| H2/N16 | fixed | alignment.py: fixed kernel bandwidths sqrt(d)*{0.25..4} and blur 0.05*sqrt(2d) (no no_grad rescaling); Sinkhorn cost p=1, debiased S_eps with dual potentials and final extrapolation; MMD is a V-statistic RBF mixture. |
| N10 | fixed | prep_scib_task.py keeps cells as distributed (no min_genes filter); all 8 prepped files exist (pancreas, lung, immune_hum_mou included). |
| N11 | fixed | prep TASKS use scIB keys (pancreas tech, lung batch, immune batch, sims Batch/SubBatch, ATAC batchname_all); batch counts 9/16/10/23/6/16/3/11 in build_paper_manifest.TASKS. |
| N12 | fixed, with disclosures | X13 scANVI/sysVI share epochs, batch 128, n_latent 10, 1x128 layers, train_size 0.9 with X1 (checked on the manifest); CPU baselines at 10 dims + tool default; run once (seed 0); scANVI in a label-aware stratum (PAPER_PLAN l.95, 190). Remaining differences to disclose: scANVI extra epochs, sysVI no KL warm-up (CR-12); no divergence guard (CR-06). |
| N13 | mostly fixed | Raw metrics primary (B/C unscaled; scib_overall not used by rules); organism from prepped uns; cell cycle only RNA human tasks; trajectory where dpt_pseudotime exists; one scorer for all. Not done: the scib 1.1.7 clustering grid (10 Leiden resolutions 0.2-2.0) is not stated (CR-13). |
| N5 | fixed (control added) | reference_fixed arm detaches reference cells in the generator step (plan l.411-415); X7 runs reference, reference_fixed, mmd_ref, discriminator_ref for every reference batch. |
| N6 | partly fixed | X8 stratified sampler (sampling.py, SI-24) for reference/pooled; per-minibatch n_k is not logged (CR-13). |
| N7 | fixed / disclose | scvi_adv anchor arm = scvi-tools AdversarialTrainingPlan unchanged (Classifier 32 units, kappa = 1 - kl_weight); X12 varies adversary width and lr; shared head and minibatch reuse still need disclosure (CR-12). |
| N8 | partly fixed | kl_weight logged per step and saved in history; PAPER_PLAN 3.4 plans reporting final kl_weight (immune 0.595 = 238/400). No warm-up-complete sensitivity rows in the manifest (CR-13). |
| N14 | superseded | HVGs follow scIB (scib.pp.hvg_batch, 2,000 genes, cell_ranger on normalised X) instead of seurat_v3 on counts, per SI-02/SI-03; ATAC uses all gene-activity features. |
| N15 | fixed (except masking gap) | plan logs z_rms, adv_loss, adv_gp, adv_w1, adv_acc, adv_grad_norm, adv_div, kl_weight and the ELBO terms; fit_paper_config saves the history in each npz. |
| M6 | fixed | X13 CPU rows have seed 0 only (run_cpu_baselines.select_rows refuses other seeds); PCA/Scanorama deterministic, Harmony in a fresh pinned-thread process (tests/x13: 13 passed). |
| L6 | unchanged (disclose) | Critic inner loop still reuses the same detached minibatch for all n_critic steps (plan l.391-408); WGAN-GP Alg. 1 samples fresh data each critic iteration; disclosure item (CR-12). |
| models | fixed | fit_paper_config.py saves models/<tag>/ (model.save, l.243); run_stage.fit_done requires model.pt before accepting a fit (l.697-698). |

## JHPCE production job script: required contents (it does not exist yet)

1. Slurm header: partition gpu, account jhpce, one GPU pinned to L40S (`--gres=gpu:l40s:1` or the site's constraint), `--cpus-per-task` >= fit lanes + 2 (gate: 10 for 8 lanes), `--mem` from a measured per-lane RSS on the largest JHPCE tasks (immune_hum_mou 97,861 cells; atac_large 84,813 x 3,580 features; the gate measured only immune at 64 GB), `--time` <= the modelled 3-day job, `--signal=B:TERM@300` so the runner's interrupt (30 s grace) can stop lanes and release claims, logs on fastscratch, explicit requeue policy.
2. `source cluster/jhpce/env.sh`; refuse unless `$FIT_ENV/.verified` exists; everything (OUT_DIR, TMPDIR, caches) on fastscratch, nothing in $HOME.
3. Repository: clean checkout of the exact commit the stage is fitted from (the tag, or the commit that holds the committed resolved manifest and decision records, PREREG sec 6.3); compare `git rev-parse HEAD` with an expected SHA passed as an argument; refuse a dirty tree.
4. Inputs: the committed manifest for the stage (derived filler manifest before R1-R3; resolved manifest afterwards) with its SHA-256 checked; prepped files checked against docs/prepped_fingerprints_scib.json (fingerprint_prepped.py --compare exit 0) or the local md5s.
5. Node witness and tests: nvidia-smi shows exactly one GPU whose name contains L40S; env_versions.py --kind fit; `WCD_SRC=src pytest -q tests/scvi` at that commit must pass on the node (CR-03).
6. Runner: `$FIT_PY scripts/run_stage.py --no-score --manifest M --experiments E --tasks <this job's JHPCE tasks> --out-dir $SCRATCH/tier12/<stage> --prepped-dir $PREPPED_DIR --fit-python $FIT_PY --wcd-src $REPO/src --expect-device L40S --fit-lanes 8 --fit-threads 1 --fit-timeout-s <about 3x the longest measured fit> --max-attempts 2`, with KMP_AFFINITY=disabled; never --allow-dirty or --allow-mixed-sha; run --dry-run first and log its output.
7. Claims and concurrency: claims of a previous job on another node are 'foreign'; clear them with --clear-foreign-claims only after sacct shows that job (owner.json slurm_job_id) is no longer running; never `scancel -u` (shared account). Concurrent jobs must select disjoint --tasks (claims and ledgers are per tag / stage key).
8. Exit handling: 0 = gate passed (all rows fitted/diverged/nonfinite_latent), 5 = stopped on infrastructure errors, 6 = incomplete, 128+n = signal; write a job summary with counts from ledger/<stage>.json.
9. Harvest: tar latents/, models/, status/, attempts/, ledger/, failures.csv, logs/ and manifests/ with md5 (split parts as gate_job.sh), copied off fastscratch before its 30-day purge; then score locally with run_stage.py (scoring mode) and `--expect-device L40S` on the harvested OUT_DIR (all scoring local, G4).
10. Stage gating: X1 adversarial rows only after the R1, R4-a1 and R2/R3 records and the resolved manifest are committed; follow-ups only after R4-x1; each submission needs the user's explicit go (SI-29).

## Corrections to the brief

- `tests/test_gate_task_split.py` does not exist at the tag; it was added on main after the tag (gate tooling).
- `tests/x13` cannot run in wcd-gpu (module skip: harmony-pytorch absent, pytest exit 5); it runs in wcd-kbet (13 passed). The other suites ran where the brief says.
- 'Every fit and scoring path is unchanged on main since the tag': confirmed (git diff lists no fit, scoring, runner, rule or src/ file).

## Checks run

- `PYTHONPATH=src WCD_SRC=src nice -n 19 wcd-gpu/bin/python -m pytest -q tests/prereg (CUDA hidden, 1 thread; review/checks/run_test_suites.sh)`: rc=0, 31s: 84 passed, 212 warnings in 28.64s
- `PYTHONPATH=src WCD_SRC=src nice -n 19 wcd-gpu/bin/python -m pytest -q tests/stage (CUDA hidden, 1 thread; review/checks/run_test_suites.sh)`: rc=0, 96s: 26 passed, 1 skipped in 95.36s (0:01:35)
- `PYTHONPATH=src WCD_SRC=src nice -n 19 wcd-gpu/bin/python -m pytest -q tests/wcd_vae (CUDA hidden, 1 thread; review/checks/run_test_suites.sh)`: rc=0, 120s: 159 passed, 66 warnings in 117.32s (0:01:57)
- `PYTHONPATH=src WCD_SRC=src nice -n 19 wcd-gpu/bin/python -m pytest -q tests/x13 (CUDA hidden, 1 thread; review/checks/run_test_suites.sh)`: rc=5, 8s: 1 skipped in 6.24s (module skipped by importorskip('harmony'): harmony-pytorch is not installed in wcd-gpu; the suite runs in wcd-kbet)
- `PYTHONPATH=src WCD_SRC=src nice -n 19 wcd-kbet/bin/python -m pytest -q tests/x13 (CUDA hidden, 1 thread; review/checks/run_test_suites.sh)`: rc=0, 489s: 13 passed, 12 warnings in 487.55s (0:08:07)
- `PYTHONPATH=src WCD_SRC=src nice -n 19 wcd-kbet/bin/python -m pytest -q tests/scoring (CUDA hidden, 1 thread; review/checks/run_test_suites.sh)`: rc=0, 59s: 4 passed, 129 warnings in 56.33s
- `PYTHONPATH=src WCD_SRC=src nice -n 19 scvi-api/bin/python -m pytest -q tests/scvi (CUDA hidden, 1 thread; review/checks/run_test_suites.sh)`: rc=0, 101s: 72 passed, 1 skipped, 162 warnings in 98.01s (0:01:38)
- `PYTHONPATH=src WCD_SRC=src nice -n 19 wcd-gpu/bin/python -m pytest -q tests/stage/test_run_stage_e2e.py (CUDA hidden, 1 thread; review/checks/run_test_suites.sh)`: rc=0, 115s: 1 passed in 114.26s (0:01:54)
- `git diff --name-only prereg-tier12-v1 origin/main (c9568fd)`: Only docs, gate tools (gate_compare_scores.py, gate_report.py, gate_task_split.py), experiments/x13_scanorama_timing/time_scanorama.py, a prep_scib_task.py docstring line and tests/test_gate_task_split.py changed; no fit, scoring, runner, rule or src/ file changed.
- `python: manifest audit (prereg_rules.read_manifest + check_seed_separation; backbone fields per experiment x task x subsample)`: 6,846 rows (A1 1098, A2 108, A3 180, X1 3000, X3 828, X6 216, X7 216, X8 504, X12 288, X13 408); seed sets disjoint (A1-A3 100-102, X1/X13 0-4, follow-ups 10-12); no backbone-field differences within any experiment x task x subsample (X12 excluded by design); epochs = scvi heuristic per task; scvi_adv only conditioned, lam 0.
- `review/checks/check_pcr_reference.py (wcd-kbet, read-only prepped files; sim1 atac_small immune sim2 pancreas)`: pcr_before pancreas: all-gene 0.482 vs HVG 0.191 (ratio 2.52); sim1: all-gene 0.521 vs HVG 0.275 (ratio 1.90); immune: all-gene 0.764 vs HVG 0.564 (ratio 1.35); sim2: all-gene 0.832 vs HVG 0.717 (ratio 1.16); atac_small: all-gene 0.124 vs HVG 0.124 (ratio 1.00); uncorrected 10-PC embedding PCR_batch all-gene vs HVG reference: pancreas 0.559 vs 0.000; sim1 0.259 vs 0.000; immune 0.179 vs 0.000 (sim2 and atac_small 0 vs 0).
- `h5py listing of prepped_scib/*__scib.h5ad (read-only)`: No uns['pca'] in any of the 8 files; obsm X_pca present; organism/modality: pancreas, lung, immune, immune_hum_mou human/rna; sim1, sim2 none/rna; atac_small, atac_large mouse/atac; dpt_pseudotime only in immune and immune_hum_mou (matches prereg_rules CC_TASKS / TRAJ_TASKS).
- `review/checks/check_x3_x8_subsample_sizes.py (scvi-api; tagged fit_paper_config.subsample on obs-only AnnData)`: X3: atac_small 11,270->8,402 cells (-25.4%), steps/epoch 80->60; pancreas 16,382->14,409 cells (-12.0%), steps/epoch 116->102; sim2 19,318->17,872 cells (-7.5%), steps/epoch 136->126; immune_hum_mou 20,000->20,000 cells (0.0%), steps/epoch 141->141. X8: fixed totals 16,000 / 6,000 / 3,000 (sim2 V=16: 2,992), equal cells per batch.
- `review/checks/check_seed_pairing_minibatch_order.py (scvi-api, CPU toy fits, seed 100)`: Generator init identical across arms; first-epoch minibatch order equal to 'none' for mmd (5/5) but 0/5 for discriminator, reference, pooled, barycenter; identical within arm across lambda.
- `review/checks/check_cuda_determinism_critic.py (scvi-api, RTX 3080, <50 MB)`: 30 repeated forward/backward passes on identical inputs and RNG state: reference/pooled critic loss values not bitwise identical (index_add_ atomics), gradients bitwise identical (max diff 0.0); MMD and Sinkhorn values and gradients identical. Informational: logged loss values, not training, vary in the last bits.
- `run_stage.py --dry-run on the tagged clone (A1, X1 pancreas, X13 pancreas, X3 pancreas, A2 atac_small; empty out dir in the workspace)`: A1: 1098 rows needs_fit, exit 0; X1 pancreas: exit 4, 360 placeholder rows refused (blocks the 15 filler rows); X13 pancreas: exit 4, 16 CPU rows refused (adv_input 'na'; blocks the 35 scANVI/sysVI rows); X3 pancreas 192 and A2 atac_small 54 placeholder rows refused (expected before the rules). Log: review/outputs/dryrun/dryrun_log.txt.
- `Source reads: scvi-tools 1.4.2 (train/_trainingplans.py AdversarialTrainingPlan l.535-735 (read from l.535 to about l.735), _compute_kl_weight, model/_utils.get_max_epochs_heuristic, module/_vae.py generative, dataloaders, external/sysvi), scib 1.1.7 (metrics.py, pcr.py, kbet.py, lisi.py, clustering.py, cell_cycle.py, trajectory.py, integration.py scvi/scanvi), POT 0.9.7 lp/_barycenter_solvers.py:160, harmony-pytorch 0.1.7 harmonize, scanorama 1.7.4 defaults; scib-pipeline main scripts/metrics/{metrics.py,Snakefile}`: See report section 'Correctness against primary sources'.
- `Primary sources fetched: arXiv 1701.07875v3 (WGAN), 1704.00028v3 (WGAN-GP), 1505.07818v4 (DANN), 1801.04406v4 (R1); Luecken et al. 2022 Nat Methods 10.1038/s41592-021-01336-8`: Quotes and page locations in the report.

## Not reviewed

- Prior-work / novelty search: not part of this code check (no method or claim was assessed for novelty).
- The running A1 outputs in tier12/ (hard rule): A1 validity is judged from the tagged code, the manifest and the checks above, not from A1 latents or scores.
- No cluster job (SI-29): JHPCE env builds, staging and L40S tests were reviewed as code only.
- Harmony (Korsunsky 2019), Scanorama (Hie 2019), scANVI (Xu 2021) and sysVI (Hrovatin et al.) papers were not fetched; their behaviour was checked against the installed package sources and scib's integration wrappers only. Gretton et al. 2012 and Feydy et al. 2019 were not fetched (standard definitions checked in code).
- No full-scale fit was run; behavioural checks used toy data (pairing), read-only prepped inputs (PCR, subsample sizes) and a GPU micro-test.
- Analysis code downstream of the rules (frontier, P1-P4 tests, SI-38) - mostly not written yet.
- scripts/gate_*.py, cost/wall-time models and the task-split tooling (not on the fit or scoring path).
- Legacy scripts beyond confirming they are present and unguarded (CR-10).
