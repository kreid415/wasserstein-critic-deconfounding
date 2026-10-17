# Verification of review findings on the real repository (model track)

Scope: 16 findings (I015-I020, I028-I037) checked on REAL main 99e942e (`review/real_main`); the code and docs involved
are byte-identical in the tagged checkout c88cce3 (cmp of alignment.py, scvi_adversarial_plan.py, fit_paper_config.py,
build_paper_manifest.py, prep_scib_task.py, fingerprint_prepped.py, adversarial.py, sampling.py, primitives.py,
score_scib_native.py, decide_a2_a3.py, PAPER_PLAN.md, PREREG.md, paper_experiment_matrix.csv), so every verdict applies
to the code the cluster jobs check out. `experiments/X1_jhpce/` exists on main only.

Machine limits respected: CPU only (CUDA_VISIBLE_DEVICES=), nice 19, <= 2 threads, small toy fits (<= 3 epochs) and
128-cell minibatch probes. Probes and raw outputs: `probes/*.py`, `*.json`, `*.out`, `*.csv` in this directory.
Environments: scvi-tools 1.4.2 fit env `scvi-api` (the env run_X1_local.sh uses) for training-path probes; `wcd` (torch
2.12.1) for loss probes; geomloss 0.2.6 installed only into `pylib/` here as an external reference. No repository,
data directory, X1_local or fillers directory was written.

Status key: CONFIRMED = defect reproduced; PARTLY = part reproduced (named); NOT REPRODUCED; NOT CHECKABLE.
Completed stages referred to: A1 1,098 fits; R1 extension 36 + R4 extension 9; A2/A3 288; rule records R1, R4-a1, R2/R3;
250 filler fits (158 scored); 19 local X1 scores.

## I031 - CONFIRMED - fix before launch

**Location.** src/wcd_vae/wcd/alignment.py:145-148 (_ot_eps final step), :158-159 (_ot_eps_sym), cost :176-179; docstring claim :25-27

**Evidence (executed).** probes/p_sinkhorn.py -> p_sinkhorn.json: two Diracs at 0 and 2 (and 0, 2.5): autograd [-2.0, 2.0], central finite difference of the code's own value 0.99999999992 (1.00000000014), value 1.999999 (2.499999). probes/p_sinkhorn2.py -> p_sinkhorn2.json on 8 real 128-cell minibatches of A1 latents (atac_small/immune, none l0 and sinkhorn l1, c1 s100): gradient norm code/geomloss 0.2.6 = 2.002-2.018, cosine 0.9975-0.9996; 0.5 x code gradient / fully converged envelope gradient = 0.995-1.014 (cos 0.994-0.998). Cause: both final half-steps f_fin = softmin(C, lb, g) and g_fin = softmin(C^T, la, f) carry gradient through the full pairwise cost of one shared z, so each potential contributes the full plan P to dS/dC; geomloss detaches the other argument of each cost. Same applies to the symmetric terms (2<a,f_fin>).

**Impact on completed work.** No refit if the code is kept and the factor is documented: every Sinkhorn fit is a fit at an effective weight ~2.0x the nominal lambda (uniform factor; direction cos >= 0.997), and Sinkhorn has its own R1 family grid and is compared at matched batch removal, so A1 (180 Sinkhorn of 1,098), R1 extension (18 Sinkhorn of 36), A3 (36 Sinkhorn of 288), R1/R4-a1/R3 records stay internally consistent. If the code is changed instead, the X1 Sinkhorn grid no longer maps to A1: refit 234 Sinkhorn fits (180 A1 + 18 ext + 36 A3), re-run R1 (sinkhorn|0, sinkhorn|1), R4-a1 Sinkhorn cells and R3 (pooled over arms incl. Sinkhorn), plus a new tag (main and tag c88cce3 have identical alignment.py). R4 extension 9, the 250 fillers and the 19 local X1 scores contain no Sinkhorn arm: unaffected.

**Recommended fix.** Decide before X1: keep tag code and state 'gradient = 2x the envelope gradient, i.e. effective weight 2*lambda' in Methods and the alignment.py docstring (recommended, no refits), or detach one cost argument per potential (geomloss convention) and redo the 234 Sinkhorn fits and R1/R4-a1/R3. Do not change the code between A1 and X1 without refitting.

## I015 - PARTLY - fix before launch

**Location.** scripts/fit_paper_config.py:174-183 (subsample, per = n_cells // n_batches at :179); scripts/build_paper_manifest.py:109-111 (X8_TOTAL), :220-230; docs/PAPER_PLAN.md:92,126; docs/paper_experiment_matrix.csv:10

**Evidence (executed).** probes/p_x8.py -> p_x8.json (real subsample() on prepped_scib obs): per-batch cells immune_hum_mou 8000/4000/2000/1000, lung 3000/1500/750/375, sim2 1500/750/375/187 at V=2/4/8/16; stratified-sampler minibatch cells per head 64/32/16/8; sim2 V=16 gives 2,992 cells (both subsets) vs 3,000 requested (V=2/4/8 exact). Cell-type total variation, subset 0: immune_hum_mou V2 vs V4 0.542 (17 vs 11 cell types), V2 vs V16 0.312; lung V2 vs V16 0.482; sim2 V2 vs V16 0.184. Parts: (a) per-head sample size not fixed - CONFIRMED (PAPER_PLAN:92/126 and matrix:10 claim it fixed; impossible with fixed N and batch 128); (b) total N not exact - CONFIRMED but 8 cells (0.27%) at one level; (c) composition varies with V - CONFIRMED numerically, but no design document claims composition is held fixed, so it is an unacknowledged confound rather than a broken claim. Also: matrix:10 still lists 'BMMC (13); 20k cells each' while the code uses lung and 16,000/6,000/3,000.

**Impact on completed work.** None on completed work: X8 has not run (no X8 row among A1, extensions, A2/A3, the 250 fillers or the 19 local X1 scores). 504 X8 rows in the manifest are affected.

**Recommended fix.** Before X8 launch: restate the X8 claim (total cells fixed; per-batch and per-head cells = N/V and 128/V), or redesign (fixed cells per batch with N growing in V, or batch size proportional to V); add composition (cell-type TV vs V) as a covariate or restrict to shared cell types; fix the doc drift in paper_experiment_matrix.csv:10; optionally round per = ceil or top up remainders for sim2 V=16.

## I028 - CONFIRMED - fix before launch

**Location.** docs/paper_experiment_matrix.csv:14 (says resolution IV); scripts/build_paper_manifest.py:259-270 (generators D=AB, E=AC; its own comment says resolution III)

**Evidence (executed).** probes/p_x12.py -> p_x12.out, read from the 288 X12 manifest rows (8 distinct factorial_run values): defining words ABD, ACE, BCDE -> resolution 3; main-effect/2FI aliases A=-BD, A=-CE, B=-AD, C=-AE, D=-AB, E=-AC. A 2^(5-2) design cannot exceed resolution III; the matrix label is the only place that says IV (grep over docs/scripts).

**Impact on completed work.** None on completed work: X12 (tier 2) has not run; 288 manifest rows.

**Recommended fix.** Before X12 launch: either relabel as resolution III and restrict analysis to main effects under the stated aliases (and to the arm x factor sign-stability question), or switch to the 16-run 2^(5-1) E=ABCD resolution-V option already coded (x12_runs=16; doubles X12 to 576 fits).

## I017 - CONFIRMED - note

**Location.** scripts/fingerprint_prepped.py:26-27 (UNS_KEYS), :50-57 (fields), :75 (compared keys)

**Evidence (executed).** probes/p_prep_meta.py -> p_prep_meta.json: fingerprint() of immune__scib with mutations, compare() problems: reverse_dpt [], delete_dpt [], add_uns_pca_variance [], permute_X_pca_rows [] (column abs-mean/std are permutation invariant); positive control (one batch label changed) -> ['batch']. Harmless today: the fingerprint is the cross-machine check of FIT inputs (counts, X, HVG, batch, celltype, task uns - all covered); fitting reads none of the missed fields (grep: no X_pca/dpt use in fit_paper_config.py or the plan; run_cpu_baselines.py recomputes PCA). obs dpt_pseudotime and uns['pca'] matter only to scoring (score_scib_native.py:186-209), and all scoring (A1, A2/A3, the 250 fillers, X1) runs on the workstation's own prepped files (fillers_x1_x13_local_scoring PREFLIGHT; X1_jhpce/SUBMISSION_PLAN.md:7,37).

**Impact on completed work.** None: no completed score used a prepped copy that was only fingerprint-verified; no rescoring.

**Recommended fix.** Add obs dpt_pseudotime (sha), the presence/content of uns['pca'], and a row-order-sensitive X_pca statistic (e.g. per-cell norms sha at rounded precision) before any scoring moves to a machine with separately prepped files.

## I034 - CONFIRMED - fix before launch

**Location.** scripts/scvi_adversarial_plan.py:392-408 (critic loop overwrites loss_d, gp), :437 (_nf_check sees only the last inner values)

**Evidence (executed).** probes/p_nf_guard.py (real fit_adversarial_scvi, scvi-tools 1.4.2, pooled critic n_critic=5, toy data, 3 epochs) -> p_nf_guard_1e38.json: making the FIRST inner critic loss genuinely non-finite (critic input x1e38 for that call) gives loss_d=inf, gp=8.11, the fit completes with no NonFiniteLossError, critic parameters stay finite (max change 2.85e-4) and the latent is finite. Reviewer-style injections (constant NaN added to loss_d; loss_d and gp each 3e38 so the sum overflows) are likewise not caught (p_nf_guard.json). The undetected case is a non-finite loss VALUE with finite gradients (here a reduction overflow); a NaN that reaches the gradients would make the Adam step non-finite and be caught at later inner steps or by on_train_end. So PREREG's outcome definition ('training loss became non-finite' -> diverged) is not enforced for inner critic steps, but the missed events leave the model finite.

**Impact on completed work.** Not checkable for completed fits: only the last inner value per step reaches the guard and the log (adv_loss), so the saved records cannot show whether an inner step was non-finite. Such an event needs per-cell critic outputs ~1e35, far from normal WGAN-GP ranges; no refit or rule re-run recommended for A1 (critic arms reference/pooled/barycenter, 540 fits), extensions, A2/A3, or the 19 local X1 scores (fillers have no critic arm).

**Recommended fix.** Accumulate a non-finite flag over every inner step (e.g. check torch.isfinite(loss_d + gp) each iteration, one host sync per step at the end) and pass it to _nf_check; result-neutral for fits without such events. Needs a new tag (code identical in c88cce3).

## I033 - CONFIRMED - fix before launch

**Location.** src/wcd_vae/wcd/alignment.py:86-87 (mmd), :104-105 (mmd_reference), :174-175 (sinkhorn) return z.new_zeros(()); scripts/scvi_adversarial_plan.py:371 (unconditional torch.autograd.grad)

**Evidence (executed).** probes/p_inactive_fit.py (real fit_adversarial_scvi, scvi-tools 1.4.2): mmd and sinkhorn with a 2-of-302-cell batch and mmd_ref with that batch as reference raise RuntimeError 'element 0 of tensors does not require grad and does not have a grad_fn' at scvi_adversarial_plan.py:371; balanced controls (mmd, sinkhorn) and pooled critic on the rare-batch data complete. probes/p_inactive_rate.py -> p_inactive_rate.csv/.out: expected per-fit crash probability from real batch shares, n_train = ceil(0.9 n), full minibatches plus the last r = n_train mod 128 cells per epoch (scvi-tools drop_last=False): not-yet-run rows 52.4 expected crashes of 1,302 critic-free rows - X7 mmd_ref sim1 18.0 of 18 and pancreas 25.9 of 27 (reference absent from the 8- or 24-cell last minibatch), X3 atac_small mmd 6.7 of 48, X8 sim2 mmd 1.08 of 24, X1 sim1 0.36 per arm (60 mmd + 60 sinkhorn). A crash is an uncaught exception -> 'infrastructure' -> retried, then the stage stops (run_stage.py:16-19); it is not recorded as a failure, so it cannot bias the rules.

**Impact on completed work.** None on completed work: grep of all fit logs finds 0 'does not require grad' errors (A1 1,114 logs, ext_r1 36, A2/A3 288, X1_local 28, fillers 250), consistent with the 0.76 expected over the 450 completed critic-free fits. A fix that returns a zero carrying the graph (e.g. z.sum()*0) changes nothing on minibatches that never hit the branch, so no refit or rescoring.

**Recommended fix.** Return (z * 0).sum() (or skip the grad/log when adv_term has no grad_fn) in the three inactive branches; decide and document what an inactive minibatch contributes (0 is natural). Required before X1 sim1, X3, X7, X8 critic-free rows run; needs a new tag.

## I032 - PARTLY - note

**Location.** src/wcd_vae/wcd/alignment.py:125-130 (_eps_schedule: one damped update per eps, final eps once), :136-143

**Evidence (executed).** Confirmed part: probes/p_sinkhorn2.py -> p_sinkhorn2.json: the loop ends after a single damped update at the final eps; its last update sup-norm is 0.23-0.61 eps, so potentials are not converged to any strict tolerance. On synthetic clustered unequal-mass points (p_sinkhorn.json) the value is 8.7-14.6% below a converged solution and the gradient direction cos 0.90-0.96. Not reproduced as a material error on real inputs: on 8 real 128-cell minibatches of A1 latents the code value is 0.10-0.24% below the converged value, equal to geomloss 0.2.6 with the same eps-scaling (|code/geomloss - 1| <= 1.4e-4), and 0.5 x code gradient matches the converged envelope gradient in norm (0.995-1.014) and direction (cos 0.994-0.998).

**Impact on completed work.** None: the approximation is the standard geomloss eps-scaling scheme; error on the latents actually trained is ~0.2%. No refit of the 234 Sinkhorn fits.

**Recommended fix.** Optional: add 2-5 extra iterations at the final eps or a residual check; state in Methods that S_eps is computed with geomloss-style eps-scaling (scaling 0.8).

## I035 - PARTLY - note

**Location.** scripts/scvi_adversarial_plan.py:262-268 (_adv_view standardisation with gradient through minibatch mean/SD), :413-415 (reference rows detached after it)

**Evidence (executed).** Mechanism CONFIRMED: probes/p_reffixed.py -> p_reffixed.json with the real _adv_view: gradient max |dL/dmu| on reference rows = 2.775 with zstd=1, 0.0 with zstd=0. Not reachable: all 54 reference_fixed rows (X7) have zstd=0 and adv_input=mean in the resolved manifest; decide_a2_a3.py:81-82 refuses any follow-up row with zstd!=0; R3 chose zstd=0 for X1 (r2r3_record.json). A3's zstd=1 rows contain no reference_fixed arm.

**Impact on completed work.** None on completed or planned work under the current R3 outcome.

**Recommended fix.** Add a guard refusing reference_fixed with zstd=1 (or compute the standardisation statistics from detached reference rows) so a later design change cannot reach the leaking path.

## I016 - CONFIRMED - fix before analysis

**Location.** scripts/fit_paper_config.py:174-178 (eligible batches, pick with default_rng(20000+100*subset+V), no distinctness check); scripts/build_paper_manifest.py:222

**Evidence (executed).** probes/p_x8.py -> p_x8.json: subsets 0 and 1 select the same batches at V=2 on all three tasks (immune_hum_mou {10X, MCA_BM_3}; lung and sim2 likewise) and at V=16 on lung and sim2; immune_hum_mou V=16 subsets differ. Cell draws differ slightly (cell-type TV between subsets 0.005-0.008 at V=2, 0.013-0.020 at V=16). probes/p_x8_elig.py -> p_x8_elig.out: only 3 batches are eligible at V=2 for every task (so at most 3 distinct pairs) and lung/sim2 have exactly 16 eligible at V=16.

**Impact on completed work.** None on completed work (X8 not run).

**Recommended fix.** Before analysis (or cheaply before launch): treat 'subset' as a batch-composition replicate only at V=4 and V=8 (and immune V=16), or force distinct pairs at V=2 (3 possible) and model subset as nested in V; do not count V=2/V=16 subsets as independent composition replicates.

## I029 - PARTLY - fix before analysis

**Location.** docs/PAPER_PLAN.md:68,91-95,193; docs/PREREG.md:104; experiments/X1/PREFLIGHT.md:55 (SC-12); manifest cond column

**Evidence (executed).** Confirmed part: resolved manifest cond counts - X3 828, X6 216, X7 216, X8 504, X12 288, X15 84 rows are all cond=1 (0 unconditioned); X1 has 1,480 cond=0 and 1,520 cond=1. Not reproduced: no document pools unconditioned with conditioned results for the primary endpoint - PREREG.md:104 defines P1 per task x decoder and SC-12 states claims are reported per decoder. The real gap is that mechanism attributions (PAPER_PLAN:193 'Mechanistic language used only for factors isolated by X3-X8') carry no decoder restriction, although the unconditioned setting is the original paper's (PAPER_PLAN:68).

**Impact on completed work.** None on completed work; affects how X1 unconditioned results may be explained.

**Recommended fix.** State in the analysis plan and Methods that mechanism follow-ups (X3, X6, X7, X8, X12, X15) explain conditioned-decoder results only; no mechanistic explanation of unconditioned-decoder X1 results unless an unconditioned follow-up is added.

## I018 - CONFIRMED - note

**Location.** scripts/prep_scib_task.py:88 (round(_frac_int, 5)), :153 (valid-mode guard < 1.0)

**Evidence (executed).** probes/p_count_guard.py -> p_count_guard.json: 1 fractional value in 300,000 nonzeros -> reported 1.0, guard passes; 1 in 150,000 or 100,000 -> 0.99999, guard trips. Harmless here: probes/p_valid_exact.py -> p_valid_exact.out: exact count of non-integer values in layers/counts of every __valid file = 0 (immune 47.4M nnz, immune_hum_mou 130.8M, lung 5.2M, pancreas 49.1M); every stage manifest uses counts=scib only.

**Impact on completed work.** None.

**Recommended fix.** Compare the unrounded fraction (or count of non-integers) in the guard; keep rounding only for the reported record.

## I020 - CONFIRMED - note

**Location.** experiments/X1_jhpce/make_x1_tags.py:97 (writes x1_jhpce_counts.csv), :101-103 (--check includes it); file never committed

**Evidence (executed).** Ran from real_main: python experiments/X1_jhpce/make_x1_tags.py --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv --filler-tags <tag checkout>/cluster/jhpce/tags/fillers_x1_x13.tags --out-dir experiments/X1_jhpce/tags --check -> exit 1, stderr 'MISMATCH: [experiments/X1_jhpce/tags/x1_jhpce_counts.csv]'; the 6 printed tags sha256 equal sha256sum of the committed files (e.g. x1_jhpce.tags 9de07275...fa5c). git ls-files shows no counts.csv. The --check is not used by x1_jhpce_command.py, SUBMISSION_PLAN.md or rehearse_x1_job.sh (grep).

**Impact on completed work.** None: the six tags files the X1 jobs would use reproduce byte-for-byte.

**Recommended fix.** Commit x1_jhpce_counts.csv or drop it from the --check set.

## I019 - CONFIRMED - note

**Location.** scripts/prep_scib_task.py:140 (source read with its uns), :175 (X_pca replaced), :178-183 (a.uns.update keeps any source uns['pca'])

**Evidence (executed).** Code path confirmed by reading: prep never deletes a source uns['pca'] while it overwrites obsm['X_pca'] with a 50-PC training-feature PCA; scib pcr() uses obsm X_pca with uns['pca']['variance'] whenever both exist (scib metrics/pcr.py:162-168), which would pair stale variances with the new PCs and bypass the all-feature PCR reference. Not present in any real input: probes/p_prep_meta.json - all 8 data_final source files have empty uns and no obsm; all 12 prepped files have no uns['pca'] (uns keys listed). score_scib_native.py:186-191 refuses subsampled scoring when uns['pca'] exists, but not full-data scoring.

**Impact on completed work.** None (no stored input carries uns['pca']).

**Recommended fix.** Drop uns['pca'] (and any source obsm/varm PCA) in prep, and make the scorer refuse uns['pca'] on full-data scoring too.

## I030 - PARTLY - note

**Location.** docs/prereg/r4_a1_record.json targets immune|1, sim1|1; rule docs/PREREG.md:95; scripts/freeze_matched_lambda.py

**Evidence (executed).** Numbers reproduced independently from the raw A1 + extension score rows (probes/p_bstar.py -> p_bstar.json): immune|1 b0 0.638167, b_common 0.656991 (barycenter), diff 0.018824, b* 0.647579; sim1|1 diff 0.006917. These are BELOW the 2*delta_min = 0.02 threshold, not 'barely above' it, and the record already flags both bstar_low_signal=true, as the rule prescribes. Consequence checked (p_r2r3_sensitivity.out): dropping immune|1 cells moves mean D from -0.00145 to -0.00416 (R2) and 0.00101 to -0.00139 (R3), both still below ROPE 0.01, so the R2/R3 decisions (mean, zstd off) do not depend on these low-signal cells.

**Impact on completed work.** None: no rule re-run; R2/R3 outcome robust. Report the flag when P1 uses b* for immune|1 and sim1|1.

**Recommended fix.** Carry the bstar_low_signal flag into X1 reporting (R4-x1 b* for these task x decoder pairs) and mark P1 results there as low-signal.

## I036 - CONFIRMED - fix before launch

**Location.** src/wcd_vae/wcd/adversarial.py:120-129 (GP computed whenever critic and with_gp); scripts/scvi_adversarial_plan.py:394 (critic loop passes with_gp=True), :287-288 (gp zeroed for spectral_norm)

**Evidence (executed).** probes/p_sn_gp.py -> p_sn_gp.json (real fit path): pooled_sn, 3 training steps x 5 critic updates -> 15 gradient-penalty evaluations (values 9.25-9.28) while logged adv_gp is 0.0; discriminator_sn -> 0 (JS head takes the with_gp=False path). Microbenchmark on the real SN head (128 x 10 input, CPU): critic step 0.0325 s with GP vs 0.0230 s without (x1.42), and after 200 identical steps fc1 weights differ by 0.0059 max-abs, i.e. the discarded GP forward passes also advance the spectral-norm power iteration and change the training trajectory, not only the cost.

**Impact on completed work.** None on completed work: SN arms occur only in X6 (pooled_sn 54 rows, discriminator_sn 27), not yet run; no SN fit in A1, extensions, A2/A3, fillers or local X1.

**Recommended fix.** Pass with_gp=False to the head when spectral_norm is set (one line in _head); needed before X6 for the documented SN-without-GP arm and for any runtime comparison.

## I037 - PARTLY - note

**Location.** src/wcd_vae/wcd/adversarial.py:11,20 (docstring/import); scripts/scvi_adversarial_plan.py:89-107 (stub module); src/wcd_vae/wcd/primitives.py:50-58

**Evidence (executed).** Confirmed: the pipeline replaces wcd_vae.wcd.primitives by a stub, so the real primitives.py is not executed in fits. Not a defect in behaviour: probes/p_primitives.py -> p_primitives.json - the discriminator head's loss class comes from the stub (module scvi_adversarial_plan), i.e. the stub IS on the training path of every discriminator fit, and stub vs real MultiClassCrossEntropy give equal values and gradients (torch.equal) for reduction mean/sum/none; the two forward bodies are the same F.cross_entropy call.

**Impact on completed work.** None.

**Recommended fix.** Reword the adversarial.py docstring (the plan supplies a copy of MultiClassCrossEntropy) or import the real class in the plan.

## Summary table

| issue_id | status | severity | location | impact |
|---|---|---|---|---|
| I031 | CONFIRMED | fix before launch | src/wcd_vae/wcd/alignment.py:145-148 (_ot_eps final step), :158-159 (_ot_eps_sym), cost :176-179 | No refit if code kept and factor (effective 2x lambda) documented; if code changed: refit 234 Sinkhorn fits (A1 180, ext 18, A3 36) + re-run R1 Sinkhorn families, R4-a1, R3 |
| I015 | PARTLY | fix before launch | scripts/fit_paper_config.py:174-183 (subsample, per = n_cells // n_batches at :179) | None (X8 not run; 504 rows affected) |
| I028 | CONFIRMED | fix before launch | docs/paper_experiment_matrix.csv:14 (says resolution IV) | None (X12 not run) |
| I017 | CONFIRMED | note | scripts/fingerprint_prepped.py:26-27 (UNS_KEYS), :50-57 (fields), :75 (compared keys) | None (scoring local; fit inputs covered) |
| I034 | CONFIRMED | fix before launch | scripts/scvi_adversarial_plan.py:392-408 (critic loop overwrites loss_d, gp), :437 (_nf_check sees only the last inner values) | Not checkable from records; no refit recommended |
| I033 | CONFIRMED | fix before launch | src/wcd_vae/wcd/alignment.py:86-87 (mmd), :104-105 (mmd_reference), :174-175 (sinkhorn) return z.new_zeros(()) | None (0 crashes in completed logs); ~52 expected crashes in 1,302 future critic-free rows |
| I032 | PARTLY | note | src/wcd_vae/wcd/alignment.py:125-130 (_eps_schedule: one damped update per eps, final eps once), :136-143 | None (0.1-0.24% value error on real latents, = geomloss) |
| I035 | PARTLY | note | scripts/scvi_adversarial_plan.py:262-268 (_adv_view standardisation with gradient through minibatch mean/SD), :413-415 (reference rows detached after it) | None (path unreachable: zstd=0 for all reference_fixed rows) |
| I016 | CONFIRMED | fix before analysis | scripts/fit_paper_config.py:174-178 (eligible batches, pick with default_rng(20000+100*subset+V), no distinctness check) | None (X8 not run) |
| I029 | PARTLY | fix before analysis | docs/PAPER_PLAN.md:68,91-95,193 | None; scope statement for analysis |
| I018 | CONFIRMED | note | scripts/prep_scib_task.py:88 (round(_frac_int, 5)), :153 (valid-mode guard < 1.0) | None (valid files exactly integer; counts=scib only) |
| I020 | CONFIRMED | note | experiments/X1_jhpce/make_x1_tags.py:97 (writes x1_jhpce_counts.csv), :101-103 (--check includes it) | None (6 tags files byte-identical) |
| I019 | CONFIRMED | note | scripts/prep_scib_task.py:140 (source read with its uns), :175 (X_pca replaced), :178-183 (a.uns.update keeps any source uns['pca']) | None (no input carries uns['pca']) |
| I030 | PARTLY | note | docs/prereg/r4_a1_record.json targets immune|1, sim1|1 | None (already flagged low-signal; R2/R3 robust) |
| I036 | CONFIRMED | fix before launch | src/wcd_vae/wcd/adversarial.py:120-129 (GP computed whenever critic and with_gp) | None (SN only in X6, not run) |
| I037 | PARTLY | note | src/wcd_vae/wcd/adversarial.py:11,20 (docstring/import) | None (stub = real class) |

Counts: status {'CONFIRMED': 10, 'PARTLY': 6}; severity {'fix before launch': 6, 'note': 8, 'fix before analysis': 2}. No blocker remains: I031 is downgraded from blocker because the factor is a near-uniform rescaling of lambda inside the Sinkhorn arm's own grid (direction cos >= 0.997 vs geomloss on real latents), so it needs a pre-launch decision and documentation, not refits, unless the code is changed.
