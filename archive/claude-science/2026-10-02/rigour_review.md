# Rigour and positioning review — adversarial objectives on stock scVI (BIOINF-2026-1434 resubmission)

Scope: keeps the original question and its four results sections (RS1 local integration vs biological conservation; RS2 multi-batch scalability/stability; RS3 hyperparameter sensitivity; RS4 reference batch selection). Every number below was computed in this session from the counts-pilot tables, the source h5ad files, the committed code at 5709a98, or the primary sources listed. Pilot-derived numbers come from a single dataset (atac_small), 3 seeds, on the pre-stock-default schedule; treat them as **provisional**.

## A. Positioning

SCOPE (per user direction): keep the original question (Wasserstein critic vs JS discriminator in adversarial batch integration), its narrative and its four results sections (RS1 local integration vs biological conservation; RS2 multi-batch scalability/stability; RS3 hyperparameter sensitivity; RS4 reference batch selection). Make each original claim rigorous; add experiments only where reviewers or the audit require them.



WHAT A REVIEWER WILL SAY IS ALREADY KNOWN (each item checked against the source named in literature[]):

1. W1 critics for invariant representations are not new. WDGRL (Shen 2018) replaced the DANN classifier with a W1 domain critic. In single-cell, iMAP (Wang 2021) already aligns batches to an 'anchor' batch with WGAN-GP (Methods text), and SAUCIE (Amodio 2019) aligns batches to a reference batch with MMD. A reference-anchored W1 critic is therefore an existing design, not a contribution.

2. JS adversaries on single-cell autoencoders are established: AD-AE, scDREAMER (cross-entropy batch classifier), scCRAFT (the original backbone, which adds a triplet loss explicitly to counter the over-correction of domain adaptation under heterogeneous cell distributions), and scvi-tools' AdversarialTrainingPlan. From the scvi-tools 1.4.2 source: 2-layer/32-unit classifier, fool loss = CE to uniform over the other batches, κ = 1 − kl_weight annealing, on by default in totalVI only when proteins are missing. Divergence alternatives: trVAE (MMD in the decoder), MMD-ResNet, BERMUDA (cluster-level MMD for different compositions), uniPort (minibatch unbalanced OT), scVI+dHSIC (Lopez 2018).

3. The original 'vanishing JS gradient for disjoint batches' premise comes from Arjovsky & Bottou 2017. Their perfect-discriminator theorems require disjoint supports or low-dimensional manifolds, and the same paper shows that adding continuous noise makes the distributions absolutely continuous, which breaks those assumptions (Thm 3.1). scVI's adversary trains on reparameterised posterior samples (Encoder.forward returns rsample(); verified), so the premise must be measured, not assumed (R1.major.2).

4. In GAN practice the loss is second-order to tuning and regularisation. Lucic 2018: most losses reach similar scores with enough tuning and restarts, and none consistently beat the non-saturating GAN. Kurach 2019: the non-saturating loss is stable across datasets and hyperparameters, and regularisation/normalisation (spectral norm most consistently) matters more than the loss. Fedus 2018: GP helps non-Wasserstein GANs. Mescheder 2018: instance noise or zero-centred gradient penalties make GAN training (locally) convergent, while WGAN/WGAN-GP with finite critic steps need not converge. The current critic arm differs from the discriminator in four ways at once: GP, 10 vs 1 inner steps, reference target, and divergence (RI-05).

5. Composition (label) shift provably makes marginal alignment mix cell types. Zhao 2019 Thm 4.3: ε_S+ε_T ≥ ½(d_JS(Y_S,Y_T) − d_JS(Z_S,Z_T))², and Lemma 4.8: the tighter the latent alignment, the larger the forced error. Wu 2019: alignment with any D that is zero only for identical distributions (explicitly the JS classifier, Wasserstein and MMD) breaks under label shift; relaxed alignment helps. Johansson 2019: invariance is often too strict. Tachet des Combes 2020: importance-weighted alignment substantially improves DANN/CDAN/JAN under large label-distribution mismatch.

6. Single-cell evidence of the same mechanism. sysVI (Hrovatin 2025): adversarial alignment (cVAE+ADV and GLUE) gave the strongest batch correction but mixed unrelated cell types with unbalanced proportions, increasingly with correction weight. Maan 2024: imbalance changes clustering, DE, mapping and trajectories across 5 methods. scIB-E (Yi 2025, Genome Biology) is the closest controlled-backbone study, comparing GAN, HSIC, orthogonality and MI-minimisation batch losses in one scVI/scANVI framework with Ray-Tune tuning.



HAS ANYONE COMPARED JS vs W1 vs MMD/SINKHORN UNDER A CONTROLLED SINGLE-CELL BACKBONE? Not found. scIB-E covers GAN/HSIC/Orthog/MIM but has no W1 critic, MMD or Sinkhorn. The original preprint compared JS against a reference W1 critic on scCRAFT. No published budget-matched JS vs W1 vs MMD/Sinkhorn comparison on a stock scVI backbone with real counts turned up (search-based absence, not exhaustive). That gap is the defensible novelty, provided the comparison is factorial (divergence × regularisation × update budget × target × latent scale) and made at matched alignment.



DOES COMPOSITION SHIFT EXPLAIN 'OVER-CORRECTION' BETTER THAN JS-vs-W1? For its existence, the theory says yes. Every proper divergence has the same zero set, so near-full alignment forces cross-type mixing for every formulation (Zhao, Wu). The data are compatible with this. Every task here has label shift: mean JS distance of batch composition to the pool is 0.15–0.44 under the registry keys, and still 0.08–0.34 after the original type-intersection 'balancing'. With scIB keys, 29% of lung and 32.5% of immune_hum_mou non-reference cells belong to types absent from the max-entropy reference. In the pilot, the critic's composite gain at λ=1 (+0.011) decomposes into batch +0.151 and bio −0.082, with iLISI up and cLISI and isolated-label F1 down. That is the signature of stronger alignment.

W1 could still differ in four testable ways: (a) a stronger operating point per nominal λ (X2 matched-alignment frontier); (b) transport geometry, since W1 moves surplus mass to the nearest type, predicting merges of similar types (X10 confusion structure); (c) scale sensitivity, since W1/MMD/Sinkhorn shrink under latent contraction while JS is scale-invariant (Bellemare 2017; X5); (d) minibatch/per-head estimation, which worsens with the number of batches (Fatras 2020; X8). The default expectation from the literature is that over-correction depends on alignment strength × composition shift. The paper is publishable if it shows how much of the original trade-off is left after controlling for that.



RECOMMENDED FRAMING (same title question, same four sections). A controlled decomposition of the critic-vs-discriminator trade-off on stock scVI: how much is operating point, composition-driven over-alignment (divergence-agnostic, predicted by theory), W1-specific geometry or scale, regularisation/update budget, and target design. Guidance for users of conditioned scVI (R3.3) follows directly: in the pilot the best conditioned adversarial arms were within seed noise of λ=0 (+0.006/+0.011) and below stock scVI (0.726 vs 0.635 on the old schedule). Claims to drop or soften: 'Wasserstein gives informative gradients for disjoint technology batches' (untested in VAEs); 'first systematic comparison of adversarial objectives' (scIB-E, WDGRL, iMAP); 'critics over-correct' as a W1 property (theory and sysVI predict it for any strong aligner); 'discriminator weakness is protective' unless it is shown as lower collapse risk at matched alignment (it is an instance of relaxed alignment, Wu 2019). Watch concurrent work framing integration as confounded domain adaptation (McCarter 2026, ConDo).

## B. Rigour issues

| id | severity | title |
|---|---|---|
| RI-01 | critical | Unit of replication is the dataset, and the 8 datasets are only ~5 independent families |
| RI-02 | critical | λ is not commensurable across formulations, so 'λ-curve dominance' is not selection-free |
| RI-03 | critical | Committed final manifest contradicts the decisions in force and the pilot |
| RI-04 | high | The pre-registered λ (20) was chosen from benchmark data and favours the discriminator |
| RI-05 | critical | Divergence is confounded with regularisation, adversary update budget and target design |
| RI-06 | critical | Composition (label) shift is uncontrolled, and the original 'balanced' setting does not remove it |
| RI-07 | high | 'Superior local mixing' rests on iLISI, which is label-agnostic and rewards cross-type mixing |
| RI-08 | high | scIB min-max scores depend on the comparison set and amplify metric noise |
| RI-09 | high | Seed noise is large relative to effects, and per-dataset inference is underpowered |
| RI-10 | high | Winner's curse and edge-of-grid optima in λ selection |
| RI-11 | high | Tuning budgets are unequal across methods |
| RI-12 | high | The disjoint-support / vanishing-gradient premise is untested on a scVI latent |
| RI-13 | high | Latent-scale shortcut: W1, MMD and Sinkhorn can be lowered by shrinking z; JS cannot |
| RI-14 | high | Batch-count scaling is confounded by per-head minibatch sample size |
| RI-15 | medium | The stock schedule ties dataset size to KL weight and step count |
| RI-16 | medium | Batch keys differ from scIB task definitions |
| RI-17 | medium | Raw counts are unavailable for parts of pancreas, immune and lung |
| RI-18 | medium | Failed runs can bias the results (survivorship) |
| RI-19 | medium | The composite hides opposite movements, and three of its metrics are silhouette-based |
| RI-20 | medium | The figure specification presupposes the result |
| RI-21 | medium | Simulations give limited ground truth, and sim2 is scored on the top-level key only |
| RI-22 | low | Mechanistic language and terminology |
| RI-23 | medium | The max-entropy reference rule can select a small batch with poor cell-type coverage |

### RI-01 (critical): Unit of replication is the dataset, and the 8 datasets are only ~5 independent families
- **Location:** scripts/build_final_manifest.py @5709a98 (DATASETS x 5 SEEDS); planned primary figure/ranking
- **Issue:** Claims of the form 'critic vs discriminator' generalise over integration tasks, so the dataset is the unit of inference; seeds only estimate within-dataset noise. Three dataset pairs are not independent, so N_eff is about 5.
- **Evidence (checked):** Read from the source h5ad barcodes: all 33,506 immune cells are the Human half of immune_hum_mou (100% barcode overlap). atac_small shares 3,618 barcodes with atac_large, and both come from the same three studies (Fang, Cusanovich, 10x). sim1/sim2 are both Splatter. Exact two-sided Wilcoxon/sign-test floor: N=8 gives p_min=0.0078, so at most 6 Holm-corrected contrasts can ever be significant. N=5 gives p_min=0.0625, so nothing can be. Nemenyi critical difference (Demsar 2006 formula, q from the studentized range, which matches Demsar's table at k=10: 3.164): k=13, N=8 gives CD=6.45 mean ranks; k=7, N=8 gives 3.18.
- **Reviewer risk:** Reviewer view: 8 datasets x 5 seeds = 40 replicates is pseudo-replication, and the secondary mean-rank figure cannot separate most methods.
- **Recommendation:** Pre-declare dataset (or family) as the unit. Fit a hierarchical model: seeds nested in datasets nested in families. Add at least 4 independent raw-count tasks (see dataset_recommendations) to reach about 10 real families. Make no per-dataset significance claims. Report simulations as a separate family.

### RI-02 (critical): λ is not commensurable across formulations, so 'λ-curve dominance' is not selection-free
- **Location:** build_final_manifest.py docstring (same ADV_LAMBDAS for discriminator/reference/pooled/barycenter; 'W1 critic output O(1) — NOT independently measured'); deliverable scvi_final_lambda_curves.png
- **Issue:** The discriminator's fool loss is a cross-entropy bounded near ln(V−1). The W1 critic term is an unbounded dual value whose scale depends on the gradient penalty, the critic's Lipschitz constant and the latent scale. MMD/Sinkhorn have their own scales. Equal nominal λ therefore means unequal adversary strength. Curves compared on the λ axis depend on that arbitrary reparametrisation, and the λ↔strength map also varies by dataset (ELBO magnitude was measured only on immune, ≈868).
- **Evidence (checked):** atac_small real-counts pilot (3 seeds; conditioned SCVI). Every formulation peaks at the grid floor λ=1 (reference 0.646, discriminator 0.641, pooled 0.561). At λ=5 critic bio collapses to 0.22, while discriminator bio is 0.55 at λ=5 and only collapses at λ=20 (0.30). So the formulations sit at different points of their own strength curves at the same λ.
- **Reviewer risk:** A reviewer will say 'discriminator dominates at every λ' only shows that the discriminator is weaker per unit λ.
- **Recommendation:** Make the primary analysis reparametrisation-invariant. Use the per-seed Pareto frontier in (batch, bio) space traced by λ; summarise it by (i) bio at matched batch removal and (ii) frontier hypervolume against a fixed reference point on unscaled metrics. Show λ-curves as descriptive only. Optionally calibrate λ per formulation and dataset by the ratio ||∇_z λL_adv|| / ||∇_z L_ELBO|| at initialisation, and report that ratio.

### RI-03 (critical): Committed final manifest contradicts the decisions in force and the pilot
- **Location:** scripts/scvi_final_manifest.tsv and build_final_manifest.py @5709a98
- **Issue:** The committed manifest (the 'ONE manifest' deliverable) has 2,000 configs, all with cond=0. Its '2 decoders' means LinearSCVI vs SCVI, not conditioned vs unconditioned. The discriminator runs only with di=1 (no 10-step arm). The adversarial λ grid is {5,20,50,150}, with nothing below 1.
- **Evidence (checked):** Manifest parsed: adv×di×cond counts = none 80, discriminator(di=1) 320, reference/pooled/barycenter(di=10) 320 each, mmd/sinkhorn 320 each, cond=0 throughout. The pilot shows the conditioned optimum at λ≤1 and critic bio collapse at λ≥5. The committed grid therefore has no points in the useful conditioned regime.
- **Reviewer risk:** Running it would reproduce the 'critic collapses' result as a grid artefact.
- **Recommendation:** Regenerate before any run: cond∈{1,0}; discriminator di∈{1,10}; a log-spaced λ grid per family that brackets the optimum (e.g. 0.01–30 for adversarial families, re-derived for MMD/Sinkhorn); stock-default schedule for all arms. Verify an interior optimum per dataset. Freeze with a hash and commit before results exist.

### RI-04 (high): The pre-registered λ (20) was chosen from benchmark data and favours the discriminator
- **Location:** build_final_manifest.py: PREREG adv=20 'where the immune probe put the discriminator peak'
- **Issue:** The secondary-table λ was picked from a discriminator probe on immune, one of the 8 evaluation datasets. That is leakage into the pre-registration, and it puts the critics in their collapsed regime.
- **Evidence (checked):** In the counts pilot at λ=20, conditioned critic bio is 0.19 (reference) and 0.20 (pooled) and the conditioned discriminator's is 0.30. The unconditioned discriminator peaks exactly at λ=20 (0.657).
- **Reviewer risk:** A pre-registered λ that equals one arm's empirical peak is not neutral.
- **Recommendation:** Pre-register a rule, not a number. Either (a) per-formulation λ fixed by the calibration rule in RI-02, computed on a pilot dataset excluded from the benchmark, or (b) nested leave-one-dataset-out 'transfer λ' (chosen on N−1 datasets, evaluated on the held-out one), which also meets the earlier nested-selection requirement. Report the per-dataset oracle argmax only as a labelled optimistic bound.

### RI-05 (critical): Divergence is confounded with regularisation, adversary update budget and target design
- **Location:** scripts/scvi_adversarial_plan.py (critics: multi_class_gradient_penalty, lambda_gp=10, disc_iter=10; discriminator: no penalty); src/wcd_vae/wcd/critic.py; adversarial.py
- **Issue:** 'Critic vs discriminator' currently differs in four things at once: (1) W1 vs JS; (2) gradient penalty vs none; (3) 10 vs 1 adversary steps (the 10-step discriminator exists in code but is not in the committed manifest); (4) reference-anchored multi-head target vs symmetric V-way classifier. Any of these can explain the original results.
- **Evidence (checked):** Read in code. Literature (verified abstracts/full text): gradient penalties also help non-Wasserstein GANs (Fedus 2018). Instance noise or zero-centred GP make GAN training locally convergent, while WGAN/WGAN-GP with finite critic steps need not converge (Mescheder 2018). The non-saturating loss is stable across datasets and hyperparameters, and regularisation/normalisation (spectral norm most consistently) matters more than the loss (Kurach 2019). Most GAN losses reach similar scores given enough tuning (Lucic 2018).
- **Reviewer risk:** R2.3/R2.5: reviewers already suspected the weakness comes from the design, not from Wasserstein.
- **Recommendation:** Factorial ablation (experiment X6): {JS, W1} × {none, gradient penalty (zero-centred R1 for JS, WGAN-GP for W1), spectral norm} × n_adv {1,5,10}, plus {reference, symmetric} × {JS, W1, MMD} (X7). Attribute effects only to factors that survive.

### RI-06 (critical): Composition (label) shift is uncontrolled, and the original 'balanced' setting does not remove it
- **Location:** src/wcd_vae/wcd/data.py prep(balance=True) keeps only cell types shared by all batches; original claims 2/3/5
- **Issue:** Zhao 2019 (Thm 4.3/Lemma 4.8), Wu 2019 and Tachet des Combes 2020 prove or show that any marginal aligner (JS, W1 or MMD) must mix cell types when composition differs, and more so as alignment tightens. sysVI (Hrovatin 2025) shows this for adversarial cVAEs. Removing batch-specific types does not equalise proportions.
- **Evidence (checked):** Computed from the source h5ad labels with the registry batch keys. Mean JS distance (sqrt base-2 JSD) between each batch's cell-type composition and the pool is 0.15–0.44. After type intersection it is still 0.08–0.34. Intersection discards up to 73% of cells (immune 33,506→8,928; 16→4 types; pancreas 14→4 types) and can increase shift to the reference (immune 0.569→0.606, because the max-entropy reference changes to smart-seq2). With scIB batch keys, lung's max-entropy reference lacks 7/17 types, and 29% of non-reference cells belong to them. immune_hum_mou (species key): 34.5% of human cells are in types absent from the Mouse reference.
- **Reviewer risk:** 'Over-correction' is then explained by known theory, independent of JS vs W1 (R2.3, R2.6).
- **Recommendation:** Add the composition dose-response with an oracle importance-weighting control (X3). Report d_JS(Y) per dataset and batch as a covariate in every model. Attribute over-correction to W1 only if the critic is worse at matched alignment AND has a steeper slope against d_JS(Y).

### RI-07 (high): 'Superior local mixing' rests on iLISI, which is label-agnostic and rewards cross-type mixing
- **Location:** Original claim 1; scIB metric definitions
- **Issue:** scIB measures kBET, graph connectivity and batch ASW per cell-identity label, while graph iLISI and PCR ignore labels (verified in Luecken 2022 Results). Under label shift, a gain in iLISI can come entirely from neighbours of different cell types, i.e. over-correction.
- **Evidence (checked):** Pilot, conditioned SCVI, reference critic λ=5 vs λ=0: iLISI 0.523→0.786, while cLISI 0.913→0.840 and isolated-label F1 0.835→0.349. At λ=1, the critic's composite gain of +0.011 is a +0.151 batch gain and a −0.082 bio loss.
- **Reviewer risk:** R1.major.1 (local/global mismatch) and R1.minor.1 (iLISI alone) will be repeated.
- **Recommendation:** Decompose iLISI into within-label and cross-label neighbour mixing, and report label-conditional iLISI plus the cross-label kNN rate. Restate claim 1 as within-type mixing. Always show batch and bio separately next to the composite.

### RI-08 (high): scIB min-max scores depend on the comparison set and amplify metric noise
- **Location:** Scoring path (scIB native, per-dataset min-max over all rows of a run)
- **Issue:** Min-max scaling makes every score relative to whichever configurations are in the table. For near-constant metrics it inflates tiny raw differences.
- **Evidence (checked):** Recomputed the pilot composite (reproduces stored values to 6e-17). Dropping only the unintegrated row moves unconditioned discriminator λ=5 from rank 3 to 7 and conditioned reference λ=1 from 4 to 2. It also reverses unconditioned-disc-λ20 vs conditioned-ref-λ1 (0.6572 > 0.6457 becomes 0.6080 < 0.6287). Graph connectivity spans only 0.9716–0.9910 across the pilot set, so a 0.005 raw jitter (the observed cross-machine level) moves the composite by 0.021, and by 0.030 if all 7 graph metrics drift together. That exceeds the 0.006–0.011 gaps between the best conditioned arms and λ=0. Maintained scib-metrics (v0.5.10 source) defaults Benchmarker.get_results(min_max_scale=False). Benavoli 2016 shows mean-rank post-hoc tests have the same pool dependence.
- **Reviewer risk:** Rankings that change when an unrelated row is added will not survive review.
- **Recommendation:** Primary analysis on unscaled metrics. Compute scIB-native min-max once over a single pre-declared full comparison set, and never on subsets. Add leave-one-method-out rescaling and z-score aggregation sensitivities, and report rank stability. Re-score a stratified 5% of embeddings on a second machine to estimate per-metric measurement SD, and set the ROPE ≥ 2 SD.

### RI-09 (high): Seed noise is large relative to effects, and per-dataset inference is underpowered
- **Location:** Counts pilot (3 seeds); planned 5 seeds
- **Issue:** Adversarial arms are several times noisier than stock scVI, mostly in bio. Seed pairing across arms is weak.
- **Evidence (checked):** Composite seed SD: stock scVI 0.006; λ=0 0.012; discriminator/reference/pooled at λ=1 0.030/0.037/0.074. Bio SD reaches 0.127 (pooled λ=1) vs 0.007 (stock). Paired SD of disc−ref is 0.048. Per-dataset 80%-power MDE: 0.148 (3 seeds), 0.080 (5), 0.048 (10), 0.032 (20). Mean correlation of seed residuals across 14 configurations is 0.14 (median 0.26).
- **Reviewer risk:** Differences of 0.006–0.011 reported as findings will be rejected as noise.
- **Recommendation:** Use the same seed for init and minibatch order in every arm, but do not rely on pairing. Pool via the hierarchical model; ≥5 seeds per cell; treat seed SD as a stability outcome. Re-estimate σ with a 1-dataset re-pilot at stock defaults before fixing N and seeds. (These values come from a pre-stock-default schedule and are provisional.)

### RI-10 (high): Winner's curse and edge-of-grid optima in λ selection
- **Location:** Original λ-sensitivity section; any per-dataset argmax table
- **Issue:** Choosing λ on the data used to report the result inflates effects. Optima at a grid edge mean the true optimum was not sampled.
- **Evidence (checked):** Pilot: all conditioned formulations peak at the grid floor (λ=1). For pooled, leave-one-seed-out selection gives 0.523 vs 0.561 in-sample (optimism 0.038). An earlier, now-invalidated sweep measured 85% inflation from per-dataset critic-favourable λ (project memory; data superseded).
- **Reviewer risk:** R2.2 ('over-correction may be λ tuning') is answered only by selection-free or nested analyses.
- **Recommendation:** Primary analysis is frontier-based (no selection). The single-λ table uses a pre-registered rule or nested leave-one-dataset-out selection. Check that every per-dataset optimum is interior, and extend the grid otherwise.

### RI-11 (high): Tuning budgets are unequal across methods
- **Location:** Manifest: 4 λ per adversarial family (critics also at 10 inner steps) vs baselines at defaults
- **Issue:** The adversarial families get a sweep; Harmony, Scanorama, scVI and others get one setting. Best-of-grid against default-only is biased toward the swept families.
- **Evidence (checked):** Lucic 2018: differences between GAN variants largely reflect computational budget and tuning. Musgrave 2022: with oracle selection, gaps between UDA methods are smaller than reported. Gulrajani & Lopez-Paz 2021: a method without its selection rule is incomplete.
- **Reviewer risk:** R2.1/R2.4/R3.1 ask how the method compares to SOTA. Unequal budgets make any 'beats/loses to' claim contestable.
- **Recommendation:** Give each baseline its native strength knob (e.g. Harmony theta; scVI/sysVI regularisation weight), swept over the same number of configurations as each adversarial family. Report best-of-k budget curves. Put scANVI (uses labels) in its own stratum. Add the stock scvi-tools AdversarialTrainingPlan (2-layer, 32-unit classifier; κ=1−kl_weight) as the 'as people use it' adversarial anchor.

### RI-12 (high): The disjoint-support / vanishing-gradient premise is untested on a scVI latent
- **Location:** Original abstract/introduction motivation; R1.major.2
- **Issue:** The perfect-discriminator theorems need disjoint supports or low-dimensional manifolds, and continuous noise removes those conditions (Arjovsky & Bottou 2017, Thm 3.1). In training, scVI's adversary sees reparameterised posterior samples. The plan also uses the bounded fool loss, a non-saturating form.
- **Evidence (checked):** scvi-tools 1.4.2 source: Encoder.forward returns z_transformation(dist.rsample()), and the plan feeds inference_outputs['z'] to the adversary. So each batch's latent distribution is Gaussian-noised and absolutely continuous.
- **Reviewer risk:** R1.major.2 asked exactly this. Assuming it again will be read as unsupported mechanism (R2.3).
- **Recommendation:** Measure it (X4): held-out batch-classifier AUC on samples and on posterior means at λ=0; fool-loss and critic gradient norms over training; per-batch posterior variance. Keep the vanishing-gradient motivation only where the measurements show it.

### RI-13 (high): Latent-scale shortcut: W1, MMD and Sinkhorn can be lowered by shrinking z; JS cannot
- **Location:** src/wcd_vae/wcd/alignment.py: _median_bandwidth under torch.no_grad (l.32-36); Sinkhorn scale under no_grad (l.110-113); W1 critic
- **Issue:** W1 is homogeneous of degree 1 in scale. MMD with a detached bandwidth and Sinkhorn with a detached normaliser also fall when z contracts. JS is an f-divergence and is invariant to invertible rescaling. 'Collapsed representations' under critics may therefore be a scale shortcut rather than over-alignment.
- **Evidence (checked):** Code read at 5709a98. Bellemare 2017 lists scale sensitivity as a property of the Wasserstein metric. The audit flagged Sinkhorn contraction. The plan has a zstd option (z centred and divided by pooled minibatch SD, with gradients) that has not yet been run.
- **Reviewer risk:** Reviewers will ask whether 'collapse' is geometric or a scaling artefact.
- **Recommendation:** Run zstd {off,on} as a factor (X5) and log latent scale, per-dimension variance and participation ratio for every run. Use zstd-on as primary for the scale-sensitive families if the pilot shows a shortcut.

### RI-14 (high): Batch-count scaling is confounded by per-head minibatch sample size
- **Location:** Original claim 4; stock batch_size=128; critic.multi_class_gradient_penalty skips the head when the reference is absent
- **Issue:** With V batches in a 128-cell minibatch, each reference-critic head sees n_k and n_ref cells. W1/OT estimates on tiny minibatches are a different, biased objective. More batches means fewer cells per head.
- **Evidence (checked):** Computed from batch sizes, batch 128, scIB keys. Lung (16 batches): the max-entropy reference (462 cells) averages 1.8 cells/step and is absent from 16% of minibatches. immune_hum_mou (23 batches): smallest batch 0.3 cells/step (absent 77%), reference 3.4 cells/step. Pancreas fluidigmc1: 5.0 cells/step. Fatras 2020: minibatch OT acts as an implicit regularisation and loses the distance property. Bellemare 2017: Wasserstein sample gradients are biased.
- **Reviewer risk:** 'Critic scales poorly' would not distinguish W1 from sampling starvation.
- **Recommendation:** In X8, add sampler {stock random, stratified with a fixed m cells per batch per step} as a factor. Vary V with random batch subsets at fixed total cells. Report per-head effective sample sizes.

### RI-15 (medium): The stock schedule ties dataset size to KL weight and step count
- **Location:** scvi-tools 1.4.2: get_max_epochs_heuristic; _compute_kl_weight (linear in epoch over n_epochs_kl_warmup=400)
- **Issue:** Under stock defaults, final KL weight ≈ (epochs−1)/400: ≈1.0 for atac_small (400 epochs), ≈0.60 for immune (239), ≈0.03 for HLCA core (14). Total steps are fixed at ~62,500 for n>20k. The adversary/KL balance therefore differs by dataset.
- **Evidence (checked):** Source read: kl_weight = epoch/n_epochs_kl_warmup; max_epochs = min(round(20000/n·400), 400).
- **Reviewer risk:** Dataset-size effects can masquerade as dataset-difficulty or scaling effects (R3.4).
- **Recommendation:** Keep stock defaults (user decision), but record final kl_weight and steps per dataset as covariates. Run one atlas-scale sensitivity with warmup scaled to max_epochs.

### RI-16 (medium): Batch keys differ from scIB task definitions
- **Location:** configs/dataset_registry.json @5709a98
- **Issue:** Registry vs scIB keys: immune 'chemistry' (4) vs 'batch' (10); lung 'protocol' (2) vs 'batch' (16); immune_hum_mou 'species' (2) vs 'batch' (23); sim2 'Batch' (4) vs 'SubBatch' (16); atac_large 'batchname' (3) vs 'batchname_all' (11). Two-batch tasks make the reference critic single-headed and the discriminator binary.
- **Evidence (checked):** Registry and obs columns read from the source files; scIB task table read in the paper.
- **Reviewer risk:** Results are not comparable to scIB, and multi-batch claims rest mostly on 2–4-batch tasks.
- **Recommendation:** Justify each key. Use the scIB keys at least as a sensitivity and for every scaling analysis.

### RI-17 (medium): Raw counts are unavailable for parts of pancreas, immune and lung
- **Location:** configs/dataset_registry.json; data.py _assert_integer_counts
- **Issue:** The prep now refuses non-integer input, but pancreas (celseq, celseq2, fluidigmc1, smarter) and lung (10x v2, 70% of cells) fail it. Immune smart-seq2 is rounded.
- **Evidence (checked):** Integrality audit CSV plus GEO listings (see dataset_recommendations): Xin GSE81608 deposits only RPKM; Lawlor GSE86469 deposits RSEM expected counts; Villani GSE94820 matrix is 99% non-integer, log-scale; scIB obtained the lung 10x data directly from the authors.
- **Reviewer risk:** 'scVI as people use it' requires counts; repaired batches must be disclosed.
- **Recommendation:** Apply the per-dataset decisions in dataset_recommendations, record each one in the registry with md5s, and run with/without sensitivity for rounded batches.

### RI-18 (medium): Failed runs can bias the results (survivorship)
- **Location:** Sweep drivers; analysis scripts
- **Issue:** Earlier sweeps had discriminator divergence (NaN) on atac_large (λ≥20) and immune_hum_mou (λ=150). This was in the old code; the bounded fool loss is the intended fix but is unverified. Dropping failed runs favours unstable formulations.
- **Evidence (checked):** Project memory of the 2026-08-26 sweep (old code; status under the new code unknown).
- **Reviewer risk:** Instability is part of the 'trade-off' claim (RS2), so it must be measured.
- **Recommendation:** Count every failure. Report rates with Clopper–Pearson CIs by formulation × λ × dataset. Impute the unintegrated score in endpoints, and run a sensitivity analysis that excludes failures.

### RI-19 (medium): The composite hides opposite movements, and three of its metrics are silhouette-based
- **Location:** Composite = 0.4·batch + 0.6·bio over 11 metrics
- **Issue:** The composite nets batch gains against bio losses. Silhouette metrics (ASW_label/batch, ASW_label, isolated_label_silhouette) are unreliable for integration (Rautenstrauch & Ohler 2025).
- **Evidence (checked):** Pilot reference λ=1: composite +0.011 = batch +0.151, bio −0.082.
- **Reviewer risk:** Reviewers want individual metrics (R1.minor.1, R2.6).
- **Recommendation:** Report all metrics and the batch/bio aggregates. Show the frontier. Add a sensitivity composite without silhouette metrics, plus the balanced metrics of Maan 2024.

### RI-20 (medium): The figure specification presupposes the result
- **Location:** Deliverable 'scvi_final_lambda_curves.png … showing discriminator-curve dominance'
- **Issue:** Writing the expected conclusion into a figure spec before the data exist invites confirmation bias and HARKing.
- **Evidence (checked):** Deliverable text.
- **Reviewer risk:** A pre-registration must be outcome-neutral.
- **Recommendation:** Restate as a two-sided hypothesis with the decision rules in statistics_plan, and specify figures neutrally.

### RI-21 (medium): Simulations give limited ground truth, and sim2 is scored on the top-level key only
- **Location:** sim1/sim2 (Splatter, from scIB)
- **Issue:** Simulators can make integration look over-optimistic (Crowell 2023). With key 'Batch' (4), sim2's 16 sub-batches are never scored as batch.
- **Evidence (checked):** Crowell 2023 abstract; registry key.
- **Reviewer risk:** Pooling simulations with real data in the main test inflates N.
- **Recommendation:** Keep simulations as a separate mechanism family. Add parametrised composition/DE simulations (X11). Score sim2 with SubBatch as well.

### RI-22 (low): Mechanistic language and terminology
- **Location:** Manuscript text (R2.3, R1.minor.3)
- **Issue:** Words like 'topological collapse' and 'geometric bottleneck' imply mechanisms that were not tested.
- **Evidence (checked):** Reviewer comments.
- **Reviewer risk:** Unsupported mechanism is cited directly as a reason to reject.
- **Recommendation:** Use mechanism words only for factors isolated by X3–X8, and fix notation against the code (λ, n_adv, GP coefficient, target definitions).

### RI-23 (medium): The max-entropy reference rule can select a small batch with poor cell-type coverage
- **Location:** src/wcd_vae/wcd/data.py select_reference_batch (max entropy, then size); plan default reference_batch=0 (scvi_adversarial_plan.py)
- **Issue:** Entropy is computed over the types a batch contains, so a batch that is missing many types can still score highest. A reference-anchored critic maps every other batch onto the reference marginal, so missing types force mismatched alignment (Johansson 2019 support coverage). The plan's own default is index 0, so the sweep driver must pass the entropy rule explicitly.
- **Evidence (checked):** Computed from source labels with scIB keys. Lung: the max-entropy batch '3' is also the smallest (462 cells) and lacks 7/17 types; 29.1% of non-reference cells belong to those types. immune_hum_mou ('batch'): reference Oetjen_A lacks 5/19 types (32.5% of non-reference cells). Registry keys: immune_hum_mou species reference = Mouse, lacking 3/19 types (34.5% of human cells).
- **Reviewer risk:** RS4's 'critic needs a topologically dense reference' may reflect the rule picking poorly covered references.
- **Recommendation:** Pre-register a coverage-aware rule (maximise the number of types present, then entropy, then size) and keep max-entropy as a sensitivity. In X7, model performance against coverage and size rather than against reference identity. Confirm which rule the driver passes.

## C. Required experiments

| experiment | priority |
|---|---|
| X1 [prerequisite; all four results sections] Clean re-run on integer counts with the corrected scorer | critical (blocking) |
| X2 [RS1: local integration vs biological conservation; claims 1, 2, 5] Matched-alignment frontier comparison (selection-free) | critical |
| X3 [RS1 mechanism; claims 2/5] Composition-shift dose–response with oracle-reweighting control | critical |
| X4 [RS1 motivation; R1.major.2] Disjoint-support and gradient diagnostics | high |
| X5 [RS1/RS3; audit Sinkhorn no_grad] Latent-scale shortcut control | high |
| X6 [RS1/RS3; claims 1/2/5] Divergence × regularisation × update-budget factorial | critical |
| X7 [RS4: reference batch selection; claim 3] Target-design factorial and reference coverage | high |
| X8 [RS2: multi-batch scalability/stability; claim 4] Batch-count scaling with controlled per-batch sampling | high |
| X9 [RS2 stability] Failure and variance accounting across all runs | high |
| X10 [RS1 downstream biology] Over-correction validated beyond scIB | high |
| X11 [RS1 ground truth] Parametrised simulations | medium |
| X12 [RS3: hyperparameter sensitivity] Architecture/hyperparameter robustness | medium |
| X13 [RS1 positioning] Baselines under a matched tuning budget | high |
| X14 [R3.4] Compute and atlas-scale behaviour | medium |

### X1 [prerequisite; all four results sections] Clean re-run on integer counts with the corrected scorer (critical (blocking))
- **Question:** Do the original four results survive on stock scvi-tools 1.4.2 trained on integer counts, scored with scIB-native pcr_comparison?
- **Design:** Regenerate and freeze the manifest (RI-03). Backbone: SCVI (ZINB, stock defaults n_latent=10, batch 128, epoch heuristic, KL warmup 400, no early stopping), conditioned (primary, as used) and unconditioned (mechanism control). LinearSCVI only in X12. Arms: λ=0 (bit-identity gate vs stock), discriminator di=1 and di=10, reference/pooled/barycenter critics (di per X6), MMD, Sinkhorn, and the stock scvi-tools AdversarialTrainingPlan(adversarial_classifier=True). λ: ≥6 log-spaced points per family that bracket the optimum (adversarial ~0.01–30; MMD/Sinkhorn re-derived), interior peak verified. Seeds: 5, paired. Datasets: all usable after dataset_recommendations (target ≥10 real families). Provisional cost from two measured anchors (single RTX 3080 lane, stock defaults): reference ≈0.079 s/step, pooled ≈0.10, discriminator ≈0.011 (1 step) / 0.023 (10 steps). Fits run 35k–62.5k steps, so reference ≈47–82 min, pooled ≈60–105, discriminator ≈7–23 min per fit. A full (dataset, backbone, seed, λ) cell with all arms is ≈6 GPU-h (barycenter/MMD/Sinkhorn costs assumed, not measured). Stage it: run X6 first to fix n_adv/GP; run conditioned on all datasets and unconditioned on ~4; use cluster GPUs.
- **Why reviewers need it:** Code audit C1–C3 invalidated every prior number. R2.2, R3.1, R3.2 all assume a correct backbone.

### X2 [RS1: local integration vs biological conservation; claims 1, 2, 5] Matched-alignment frontier comparison (selection-free) (critical)
- **Question:** At the same degree of batch removal, does the W1 critic conserve less biology than the JS discriminator? Or do both lie on one frontier and differ only in operating point per λ?
- **Design:** Uses X1 runs. For each dataset × backbone × seed × formulation, build the (batch, bio) λ-frontier on unscaled scIB aggregates (pcr_comparison; label-conditional and label-free metrics listed separately). Endpoints: (i) bio at matched batch b* (pre-fixed rule, e.g. midpoint of the overlap of the two frontiers' batch ranges); (ii) frontier hypervolume against a fixed reference point; (iii) maximum batch score reachable without collapse (bio ≥ bio_λ0 − δ, δ pre-fixed); (iv) within-label vs cross-label iLISI decomposition. Confirms claim 2 if the critic's bio@b* < the discriminator's beyond the ROPE in most datasets (hierarchical posterior ≥0.95). Refutes it if the frontiers coincide (equivalence within ROPE): 'over-correction' is then an operating-point effect. Claim 1 holds only as within-label mixing. Claim 5 ('protective') holds if the discriminator's collapse probability across λ is lower and its frontier is not dominated.
- **Why reviewers need it:** R1.major.1, R2.2 (Pareto front), R2.6, R1.minor.1. This is the selection-free answer to 'it is just λ tuning'.

### X3 [RS1 mechanism; claims 2/5] Composition-shift dose–response with oracle-reweighting control (critical)
- **Question:** Is over-correction driven by cell-type composition differences between batches, for every divergence, as theory predicts? Is W1 any worse at matched alignment?
- **Design:** Base tasks: composition-matched subsamples (equal per-type counts in every batch; computed feasibility: atac_small 6,075 cells/6 types, immune_hum_mou species 17,262/11, sim2 10,572/4, sim1 2,544/3) plus one replacement real dataset (HLCA or BMMC subset). Perturbation: deplete 1–2 pre-chosen types in one batch at {0, 50, 80, 95, 100}%; report the resulting d_JS(Y). Arms: λ=0; discriminator(10); reference critic; pooled critic; MMD. Each at 2 λ chosen on the composition-matched base to give matched batch removal. Positive control: an importance-weighted oracle adversary that weights each cell by p_ref(c)/p_b(c) from true labels (the Tachet des Combes 2020 GLS reweighting idea). Negative control: λ=0. 5 seeds. Endpoints: cross-label kNN rate and isolated-label F1 of the depleted type, label-transfer recall for that type, balanced ARI (Maan 2024). Composition is the driver if slopes against d_JS(Y) are >0 for every formulation and IW removes them. W1-specific harm requires a steeper critic slope at matched alignment.
- **Why reviewers need it:** R2.3 (mechanism), R2.5, R2.6 (direct validation). Zhao 2019, Wu 2019 and sysVI 2025 predict the effect for any marginal aligner, so a reviewer will demand this control.

### X4 [RS1 motivation; R1.major.2] Disjoint-support and gradient diagnostics (high)
- **Question:** Are batch latent distributions in scVI actually near-disjoint, and does the discriminator's gradient vanish?
- **Design:** Logging only, on every X1 run at λ=0 and the frontier points: held-out batch-classifier AUC/accuracy (fresh 2-layer MLP, 5-fold) on posterior samples and on posterior means; per-step norms of ∇_z(fool loss) and ∇_z(critic term); critic W1 estimate over training; per-batch mean posterior variance. All datasets, 3 seeds. The original premise is confirmed only if AUC≈1 and discriminator gradients vanish early in training.
- **Why reviewers need it:** R1.major.2 asked directly. scVI's Gaussian posterior sampling (verified) removes the disjoint-support condition of Arjovsky & Bottou 2017.

### X5 [RS1/RS3; audit Sinkhorn no_grad] Latent-scale shortcut control (high)
- **Question:** Is critic 'collapse' a scale shortcut (W1/MMD/Sinkhorn shrink when z contracts) rather than over-alignment?
- **Design:** zstd {off, on} × {reference, pooled, MMD, Sinkhorn, discriminator(10)} × 3 λ (low/mid/high of the calibrated grid) × 3 datasets (atac_small, pancreas-repaired, one replacement) × 5 seeds. Outcomes: mean pairwise latent distance, per-dimension variance, participation ratio, bio/batch, frontier endpoints. Collapse counts as W1-specific over-correction only if it persists with zstd on.
- **Why reviewers need it:** Code audit HIGH (Sinkhorn scale under no_grad); R2.3; R3.2.

### X6 [RS1/RS3; claims 1/2/5] Divergence × regularisation × update-budget factorial (critical)
- **Question:** Which part of the critic–discriminator difference is due to the divergence (W1 vs JS) rather than the gradient penalty, spectral norm or n_adv?
- **Design:** {JS (bounded fool loss), W1 (critic)} × {none, gradient penalty (zero-centred R1 for JS; one-centred WGAN-GP for W1), spectral norm} × n_adv {1, 5, 10}. Target symmetric for both (pooled critic vs V-way classifier) so target design is held fixed. 3 datasets × 3 calibrated λ × 3–5 seeds, conditioned SCVI. Mixed model on frontier endpoints with main effects and interactions. Run this before X1 to fix the main-grid n_adv and penalty.
- **Why reviewers need it:** R2.3/R2.5 and the GAN literature (Fedus 2018; Mescheder 2018; Kurach 2019; Lucic 2018) make this the first question a methods reviewer asks.

### X7 [RS4: reference batch selection; claim 3] Target-design factorial and reference coverage (high)
- **Question:** Does reference dependence come from W1 or from reference-anchored alignment in general?
- **Design:** {reference-anchored, symmetric (pooled/barycenter)} × {W1 critic, JS (reference discriminator: one binary ref-vs-k head per batch), MMD (SAUCIE-style reference vs pooled)}. References: every batch as reference on pancreas (8 after repair), sim1 (6) and immune (scIB key, 10), or max-entropy + largest + 4 random. Covariates: reference size, number of types absent or <1% in the reference, fraction of non-reference cells in such types, d_JS(Y_k, Y_ref). 3 seeds, calibrated λ. Claim 3 holds for W1 only if the formulation × coverage interaction is larger for the reference critic than for the reference discriminator.
- **Why reviewers need it:** R2.5 says the weakness may come from the reference design, not Wasserstein. iMAP (WGAN-GP to an anchor) and SAUCIE (MMD to a reference) are reference-anchored prior designs.

### X8 [RS2: multi-batch scalability/stability; claim 4] Batch-count scaling with controlled per-batch sampling (high)
- **Question:** Does the critic degrade with the number of batches once per-head sample size and total cells are held fixed?
- **Design:** Datasets with ≥13 batches and integer counts: immune_hum_mou (scIB key 'batch', 23), sim2 ('SubBatch', 16), HLCA core subset (donor/dataset batches), NeurIPS BMMC (site×donor). V ∈ {2, 4, 8, 16}; 3 random batch subsets per V; total cells fixed by subsampling. Sampler {stock random batch 128, stratified with m=16 cells per batch per step}. Arms: discriminator(10), reference critic, pooled/barycenter critic, MMD at the pre-registered λ, plus λ=0. 3 seeds. Outcomes: full metric panel (not iLISI alone), label transfer, divergence rate, wall time, per-head effective n. Claim 4 holds if the critic − discriminator gap grows with log2 V under stratified sampling. If it disappears there, it was sampling starvation.
- **Why reviewers need it:** R1.minor.1 (scaling claim rests on iLISI alone), R3.4 (atlas scale). The confound is quantified in RI-14.

### X9 [RS2 stability] Failure and variance accounting across all runs (high)
- **Question:** Are adversaries less stable than stock scVI, and does stability differ between formulations?
- **Design:** Derived from X1–X8, no new fits: NaN/divergence rate with Clopper–Pearson CIs; seed-SD ratio vs λ=0 (bootstrap CI); adversary-loss oscillation; time to divergence. Failures enter endpoints as worst-case (unintegrated) scores, with a sensitivity analysis excluding them.
- **Why reviewers need it:** RS2 asserts a stability trade-off. The pilot already shows bio seed SD up to 18× stock scVI's.

### X10 [RS1 downstream biology] Over-correction validated beyond scIB (high)
- **Question:** Do composite differences translate into loss of biology that users care about?
- **Design:** On matched-alignment configs from X2 and the pre-registered λ, with 5 seeds: (a) label transfer: kNN on the latent from source batches to a held-out batch; balanced accuracy, macro-F1, per-type confusion showing which types merge. (b) Marker preservation: per-type top-50 markers from scVI normalised expression vs within-batch markers from unintegrated counts (Jaccard). (c) Rare-type recovery (types <1%: recall/F1). (d) Balanced ARI/AMI (Maan 2024). (e) Within-type structure (e.g. Moran's I as in sysVI). (f) Condition preservation on Kang 2018 IFN-β: stim/ctrl separation within each cell type after donor integration. (g) Genotype- or design-defined identity on CellBench mixtures.
- **Why reviewers need it:** R1.minor.2, R2.6, R3.minor.2 (DE, label transfer, trajectory).

### X11 [RS1 ground truth] Parametrised simulations (medium)
- **Question:** With known truth, when do the formulations merge distinct types?
- **Design:** Splatter (as in scIB) with a factorial over composition shift (d_JS(Y) 0–0.6), a batch-specific type (present/absent), batch-effect strength (2 levels) and known DE genes. 5 seeds. Report exact merge rates and DE-gene recovery. Analyse as a separate family; following Crowell 2023, do not use simulations as generalisation evidence.
- **Why reviewers need it:** R2.6 asks for direct validation. Simulations give the only exact over-correction ground truth.

### X12 [RS3: hyperparameter sensitivity] Architecture/hyperparameter robustness (medium)
- **Question:** Does the sign of the critic–discriminator difference depend on the backbone or adversary settings?
- **Design:** Fractional factorial (resolution IV): n_latent {10, 30}; adversary width {32 (stock), 128}; adversary lr {1e-4, 1e-3}; batch size {128, 512}; decoder {SCVI-ZINB, LinearSCVI-NB}. 3 datasets × 2 calibrated λ × 3 seeds; discriminator(10) vs reference critic. Report sign stability of Δbio@b*.
- **Why reviewers need it:** R3.2 (robustness to architecture and hyperparameters). The pilot shows schedule alone moved λ=0 from 0.635 to 0.726.

### X13 [RS1 positioning] Baselines under a matched tuning budget (high)
- **Question:** Where do adversarial formulations sit relative to standard integrators, under the same configuration budget?
- **Design:** scVI (λ=0, stock), scvi-tools stock adversarial scVI, Harmony, Scanorama, sysVI (in scvi-tools), and others in the 6-baseline set. Each gets its native strength knob swept over as many configurations as an adversarial family; 5 seeds; all datasets. scANVI goes in a separate label-using stratum. Report best-of-k budget curves and frontier position, not only a rank table.
- **Why reviewers need it:** R2.1, R2.4, R3.1, R3.3 (practical guidance). It answers the scVI user's question: does adding any adversary to a conditioned decoder help? In the pilot the best conditioned arms were within noise of λ=0 (Δ +0.006/+0.011).

### X14 [R3.4] Compute and atlas-scale behaviour (medium)
- **Question:** What does each formulation cost, and do conclusions hold at atlas scale?
- **Design:** Wall-clock, GPU memory and steps for every run (logged). On the HLCA core (584,944 cells, CELLxGENE), run the best frontier config of each family plus baselines at stock defaults (14 epochs; KL weight reaches ≈0.03) and with warmup scaled to max_epochs as a sensitivity. 3 seeds.
- **Why reviewers need it:** R3.4. The measured cost gap (reference critic ≈7× the 1-step discriminator per epoch) is itself practical guidance.

## D. Statistics plan

PRE-REGISTRATION (freeze the manifest hash, analysis code and this plan before any run; all hypotheses two-sided; the figure spec must be outcome-neutral, RI-20).

UNIT AND STRUCTURE. Inference unit = dataset (task), grouped into families. immune + immune_hum_mou share all 33,506 human cells; atac_small and atac_large share 3,618 barcodes and the same three studies; sim1/sim2 share a simulator. Seeds (5, paired: same init seed and minibatch order across arms) are nested replicates. Model: y = arm + (1|family/dataset) + (1|dataset:seed). Target ≥10 independent real families. Floors: with N=8 datasets the exact Wilcoxon/sign-test minimum two-sided p is 0.0078 (≤6 Holm contrasts can ever be significant); with 5 families it is 0.0625 (nothing can be). Nemenyi CD at N=8 is 3.18 ranks for k=7 and 6.45 for k=13, so no mean-rank CD diagrams as inference (Benavoli 2016).

METRICS. Primary scale is unscaled scIB metrics (pcr_comparison; scib-metrics default min_max_scale=False). Batch and bio aggregates are reported separately. scIB-native min-max is secondary and computed once over the pre-declared full comparison set (all arms × λ, baselines, unintegrated), never on subsets. Sensitivities: leave-one-method-out rescaling, z-score aggregation, a composite without the three silhouette metrics; report rank stability as Kendall τ. Measurement noise: re-score a stratified 5% of embeddings on a second machine and with different kNN/Leiden seeds; per-metric SD sets ROPE = max(0.01, 2·SD) on the aggregate scale.

PRIMARY ENDPOINTS (family of 4, Holm for the frequentist companions):

P1 (RS1; claims 1–2): Δbio@b* = bio_critic(b*) − bio_disc(b*). bio(b) is the per-seed λ-frontier (monotone piecewise-linear interpolation of the (batch, bio) points over the λ grid). b* is fixed by rule: midpoint of the overlap of the two frontiers' batch ranges, or the conditioned λ=0 batch score + 0.05 if there is no overlap. Contrast: reference critic vs discriminator(10 steps), conditioned SCVI. Co-primary: frontier hypervolume difference against the fixed reference point (0, 0) on unscaled aggregates.

P2 (claim 5): collapse probability. Collapse = bio < bio_λ0 − 0.10 (unscaled; pre-fixed) at any grid λ; logistic mixed model, formulation fixed effect.

P3 (RS2; claim 4): slope of the P1 endpoint (or of bio at the pre-registered λ) on log2(V) under the stratified sampler; mixed model with random dataset slopes and random batch-subset draws.

P4 (RS4; claim 3): formulation × reference-coverage interaction. Coverage = fraction of non-reference cells in types absent or <1% in the reference; d_JS(Y_k, Y_ref) as a second covariate. Contrast: reference critic vs reference discriminator (same target design).

DECISION RULES. Bayesian hierarchical model (Corani 2017; seeds within datasets) gives P(Δ < −ROPE), P(|Δ| ≤ ROPE), P(Δ > ROPE). Declare a direction if its probability ≥ 0.95, equivalence if P(rope) ≥ 0.95, otherwise inconclusive. Frequentist companion: Wilcoxon signed-rank on dataset means with Holm over P1–P4. Bayesian signed-rank (Benavoli 2017) as a check.

SECONDARY (estimation only, BH-FDR if any p-values are shown). All 11 metrics; label-conditional iLISI and cross-label kNN rate; downstream panel (label-transfer balanced accuracy/macro-F1, marker Jaccard, rare-type F1, bARI/bAMI, condition preservation); seed SD ratio vs λ=0; divergence rate; wall time and memory; composition dose-response slopes (X3) with the IW positive control; factor main effects from X5/X6/X7 (mixed ANOVA, effect sizes with CIs).

λ HANDLING. The primary analysis needs no λ selection (frontier). The single-λ table uses (i) a per-formulation λ fixed by a calibration rule (gradient-norm ratio at init) on a pilot dataset outside the benchmark, and (ii) nested leave-one-dataset-out transfer λ (argmax of mean endpoint on N−1 datasets, evaluated on the held-out one). This meets the earlier nested-selection requirement. The per-dataset oracle argmax is shown only as a labelled optimistic bound. Every per-dataset optimum must be interior to the grid; otherwise extend the grid and re-run before analysis.

BUDGET PARITY. Each method family gets the same number of configurations. Report best-of-k curves (k = 1…K configurations; Lucic 2018). scANVI forms a separate label-using stratum.

EFFECT SIZES. Report Δ in metric units with 95% intervals (hierarchical posterior; dataset-then-seed BCa bootstrap as a check). Never standardise by seed SD alone.

FAILURES. NaN or diverged runs are outcomes: rates with Clopper–Pearson CIs. Endpoints impute the unintegrated score; a sensitivity analysis excludes failures. No silent drops.

SIMULATIONS form a separate family and are excluded from P1–P4 primary pooling.

POWER (provisional: atac_small pilot, 3 seeds, pre-stock-default schedule). Paired seed SD of the composite is 0.048. Per-dataset 80%-power MDE is 0.148 (3 seeds), 0.080 (5), 0.048 (10). The pooled 95% CI half-width for a cross-dataset mean is 0.018–0.025 at N=8 and 0.014–0.019 at N=12 (5 seeds, between-dataset SD τ = 0–0.02). Observed best-arm vs λ=0 gaps were 0.006–0.011, so the design can certify differences of about ≥0.02 or equivalence within about ±0.02. Re-estimate σ (and τ, from ≥3 datasets) in a stock-default re-pilot before freezing N and seeds.

## E. Dataset recommendations

Checks per dataset: integrality audit table (source_counts_integrality.csv), the source h5ad obs columns, configs/dataset_registry.json @5709a98, scIB data-availability text, GEO FTP listings, CELLxGENE API and schema 5.3.0.

PANCREAS (scIB 9-tech task). Keep, with repairs.

- celseq (Grün, GSE81076) and celseq2 (Muraro, GSE85241): GEO deposits only UMI-collision-corrected tables. Invert the correction to integer UMIs; the audit found this exactly invertible.

- fluidigmc1 (Lawlor, GSE86469): GEO deposits RSEM 'raw.expected.counts'. Round them (registry 'round_count_batches', as for immune) and run a with/without sensitivity.

- smarter (Xin, GSE81608): GEO deposits only an RPKM table. Drop these 1,492 cells (9.1%); re-quantifying from SRA reads is the only alternative.

- smartseq2 (E-MTAB-5061) and inDrop1–4 (Baron, GSE84133): integer in the source.

Report the result as an 8-batch task that deviates from scIB.

IMMUNE (human). Keep. The smart-seq2 batch (Villani GSE94820, 1,022 cells): the GEO matrix is not counts (streamed first ~200 genes: 99.1% of non-zero values non-integer, range ~0.9–7). The h5ad counts layer holds 2-decimal expected counts (registry note: median library ~860k). Rounding is defensible but these are full-length read counts, not UMIs, so add a sensitivity without this batch. Use the scIB key 'batch' (10 donors) at least as a sensitivity and for scaling; the registry uses 'chemistry' (4).

IMMUNE_HUM_MOU. Its 33,506 human cells are identical to IMMUNE (100% barcode overlap), so analyse it as the same family. The species key (2 batches) gives a binary discriminator and a single-head reference critic: useful for the cross-species question, not for scaling. For RS2 use scIB key 'batch' (23). Apply the same smart-seq2 rule.

LUNG (scIB lung atlas). The 10x v2 batch (22,771 of 32,472 cells, 70%) holds log-normalised values. scIB states the 10X lung data were obtained directly from the authors, so no public raw source is documented. The Drop-seq part (GSE130148_raw_counts.csv.gz) is raw but has no protocol partner. Replace with the HLCA core (CELLxGENE: 584,944 cells, 10x 3' v1/v2/v3 and 5'; schema 5.3.0 requires de-duplicated integer UMI counts in raw.X). Use a ~30–50k-cell subsample of several studies/assays for the main grid, the full core for atlas scale (R3.4), and donor/study batches for RS2 scaling.

SIM1/SIM2 (Splatter). Integer. Keep as a separate mechanism family (Crowell 2023: simulators flatter integration). Score sim2 with 'SubBatch' (16) as well as 'Batch' (4).

ATAC_SMALL/ATAC_LARGE (gene activity). Integer. One family (same three studies; 3,618 shared barcodes). atac_large has 17.4× batch imbalance; the scIB key 'batchname_all' (11) is available for scaling. Report it as the modality-generalisation task (R3.minor.1).

ADD (raw counts confirmed on GEO/CELLxGENE; each serves a named experiment):

(1) HLCA core: lung replacement, atlas scale, many batches.

(2) NeurIPS 2021 BMMC (GSE194122; 12 donors, 4 sites, Multiome + CITE GEX): nested site/donor batches and many batches for X8. Processed h5ad deposited; confirm the counts layer on download (PROVISIONAL).

(3) CellBench (GSE118767; per-sample count CSVs; cell-line mixtures across droplet and plate protocols): identity known by design, so it is ground truth for over-correction (X10g).

(4) Ding 2020 (GSE132044; PBMC/cortex count matrices across 7 methods at one centre): technology batches with similar composition (inferred from the design), a natural low-composition-shift control for X3.

(5) Kang 2018 (GSE96583; 10x matrices; IFN-β vs control, 8 donors): condition preservation after donor integration (X10f).

Feasibility of composition-matched subsamples (equal per-type counts in every batch, registry keys): sim2 54.7% of cells retained, atac_small 53.9%, lung (protocol) 43.6%, pancreas 21.7%, sim1 21.0%, immune_hum_mou 17.6%, atac_large 13.0%, immune (chemistry) 5.2%. So X3 should use atac_small, immune_hum_mou, sim1/sim2 and one replacement real dataset.

Record every repair (inversion, rounding, dropped batch, subsample seed) with md5s in the registry. The prep's integer assertion must pass for every dataset.

## Supporting tables (this session)

- composition_shift_by_dataset.csv: per-dataset label-shift statistics (registry keys, type-intersection, scIB keys)
- composition_matched_feasibility.csv: cells retained under equal-count composition matching
- minibatch_cells_per_batch.csv: expected cells per batch per minibatch and P(absent) at batch 128/512
- pilot_comparison_set_dependence.csv: min-max rescaling with different comparison sets (rank reversals)
- pilot_minmax_noise_amplification.csv: composite shift caused by a 0.005 raw jitter per metric
- pilot_lambda_selection_optimism.csv: in-sample vs leave-one-seed-out λ selection
- pilot_raw_metrics_cond.csv: raw metrics for the conditioned arms (iLISI vs cLISI/isolated F1)
- nemenyi_cd_table.csv, ci_planning_table.csv: power and critical-difference planning

## F. Literature (verified)

| citation | DOI/URL | how verified |
|---|---|---|
| Arjovsky M et al. Wasserstein GAN. ICML 2017 (arXiv:1701.07875). | 10.48550/arXiv.1701.07875 | OpenAlex DOI record (title/authors/year) and abstract; arXiv PDF downloaded. |
| Arjovsky M et al. Towards principled methods for training generative adversarial networks. arXiv:1701.04862 (2017; ICLR 2017 submission version). | 10.48550/arXiv.1701.04862 | arXiv PDF text: located the perfect-discrimination section and the noise remedy (Thm 3.1, Gaussian ε). |
| Gulrajani I, Ahmed F, Arjovsky M, Dumoulin V, Courville A. Improved training of Wasserstein GANs. NeurIPS 2017, pp. 5767–5777 (arXiv:1704.00028). | 10.48550/arXiv.1704.00028 | OpenAlex abstract; project critic.py read (WGAN-GP penalty). |
| Ganin Y et al. Domain-adversarial training of neural networks. JMLR 2016 (arXiv:1505.07818). | 10.48550/arXiv.1505.07818 | OpenAlex record and abstract of the arXiv version. |
| Shen J et al. Wasserstein distance guided representation learning for domain adaptation. AAAI 32(1), 2018. | 10.1609/aaai.v32i1.11784 | OpenAlex abstract. |
| Zhao H et al. On learning invariant representations for domain adaptation. ICML 2019 (arXiv:1901.09453). | 10.48550/arXiv.1901.09453 | arXiv PDF text: Theorem 4.3, Lemma 4.8 and the JS-distance definition (bounded in [0,1]) located. |
| Johansson FD et al. Support and invertibility in domain-invariant representations. PMLR 2019 (arXiv:1903.03448). | 10.48550/arXiv.1903.03448 | OpenAlex abstract; arXiv PDF downloaded. |
| Tachet des Combes R, Zhao H, Wang Y-X, Gordon GJ. Domain adaptation with conditional distribution matching and generalized label shift. NeurIPS 33:19276–19289, 2020 (arXiv:2003.04475). | 10.48550/arXiv.2003.04475 | OpenAlex abstract; arXiv PDF passage (sharp drop of base DANN/CDAN as label-JSD grows; IW versions hold). |
| Wu Y et al. Domain adaptation with asymmetrically-relaxed distribution alignment. ICML 2019 (arXiv:1903.01689). | 10.48550/arXiv.1903.01689 | arXiv PDF text: passage listing JS, Wasserstein (Shen 2018) and MMD; relaxed f-divergence and Wasserstein sections. |
| Lucic M et al. Are GANs created equal? A large-scale study. arXiv:1711.10337 (2017). | 10.48550/arXiv.1711.10337 | OpenAlex abstract. |
| Kurach K et al. A large-scale study on regularization and normalization in GANs. ICML 2019, PMLR 97. | 10.48550/arXiv.1807.04720 | arXiv PDF text: introduction conclusions and Fig. 1/loss-comparison captions. |
| Fedus W et al. Many paths to equilibrium: GANs do not need to decrease a divergence at every step. ICLR 2018 (arXiv:1710.08446). | 10.48550/arXiv.1710.08446 | OpenAlex abstract. |
| Mescheder L et al. Which training methods for GANs do actually converge? arXiv:1801.04406 (2018). | 10.48550/arXiv.1801.04406 | OpenAlex abstract. |
| Bellemare MG et al. The Cramer distance as a solution to biased Wasserstein gradients. arXiv:1705.10743 (2017). | https://arxiv.org/abs/1705.10743 | arXiv API abstract. |
| Fatras K et al. Learning with minibatch Wasserstein: asymptotic and gradient properties. arXiv:1910.04091 (2019). | https://arxiv.org/abs/1910.04091 | arXiv API abstract. |
| Feydy J et al. Interpolating between optimal transport and MMD using Sinkhorn divergences. arXiv:1810.08278 (2018). | 10.48550/arXiv.1810.08278 | OpenAlex abstract; project alignment.py read. |
| Gulrajani I et al. In search of lost domain generalization. ICLR (arXiv:2007.01434, 2020). | 10.48550/arXiv.2007.01434 | OpenAlex abstract. |
| Musgrave K et al. Unsupervised domain adaptation: a reality check. arXiv:2111.15672 (2021). | 10.48550/arXiv.2111.15672 | OpenAlex abstract. |
| Demsar J. Statistical comparisons of classifiers over multiple data sets. JMLR 7:1–30, 2006. | https://jmlr.org/papers/v7/demsar06a.html | JMLR PDF text: recommendation sentence, CD formula and q_alpha table (q0.05 = 3.164 for k=10, reproduced numerically). |
| Benavoli A et al. Should we really use post-hoc tests based on mean-ranks? arXiv:1505.02288 (2015). | 10.48550/arXiv.1505.02288 | Abstract and PDF text. |
| Benavoli A et al. Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. JMLR (arXiv:1606.04316). | 10.48550/arXiv.1606.04316 | arXiv PDF text: Bayesian signed-rank and rope sections. |
| Corani G et al. Statistical comparison of classifiers through Bayesian hierarchical modelling. Machine Learning 106(11):1817–1837, 2017. | 10.1007/s10994-017-5641-9 | OpenAlex DOI record; arXiv:1609.08905 PDF text (model description, rope default). |
| Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat 6:65–70, 1979. | 10.2307/4615733 | OpenAlex record and abstract. |
| Lopez R et al. Deep generative modeling for single-cell transcriptomics (scVI). Nat Methods 15(12):1053–1058, 2018. | 10.1038/s41592-018-0229-2 | OpenAlex record; scvi-tools 1.4.2 source read (scvi-ref env). |
| Gayoso A et al. A Python library for probabilistic analysis of single-cell omics data (scvi-tools). Nat Biotechnol 40(2):163–166, 2022. | 10.1038/s41587-021-01206-w | OpenAlex record; scvi/train/_trainingplans.py in 1.4.2 read (loss_adversarial_classifier, training_step, configure_optimizers). |
| Gayoso A et al. Joint probabilistic modeling of single-cell multi-omic data with totalVI. Nat Methods 18(3):272–282, 2021. | 10.1038/s41592-020-01050-x | PubMed abstract; scvi/model/_totalvi.py (1.4.2) read: _use_adversarial_classifier set True only with protein batch mask. |
| Lopez R et al. Information constraints on auto-encoding variational Bayes. arXiv:1805.08672 (2018). | 10.48550/arXiv.1805.08672 | OpenAlex abstract. |
| Dincer AB et al. Adversarial deconfounding autoencoder for learning robust gene expression embeddings (AD-AE). Bioinformatics 36(Suppl 2):i573–i582, 2020. | 10.1093/bioinformatics/btaa796 | OpenAlex abstract. |
| Wang D et al. iMAP: integration of multiple single-cell datasets by adversarial paired transfer networks. Genome Biol 22:63, 2021. | 10.1186/s13059-021-02280-8 | Genome Biology PDF text: Methods (WGAN-GP; anchor batch). |
| Zhao J et al. Adversarial domain translation networks for integrating large-scale atlas-level single-cell datasets (Portal). Nat Comput Sci 2(5):317–330, 2022. | 10.1038/s43588-022-00251-y | Abstract only (full text not retrievable). |
| Shree A, Pavan MK, Zafar H. scDREAMER for atlas-level integration of single-cell datasets using deep generative model paired with adversarial classifier. Nat Commun 14:7781, 2023. | 10.1038/s41467-023-43590-8 | OpenAlex abstract; publisher text (search result) describing the CE batch classifier and Bhattacharyya loss. |
| Amodio M et al. Exploring single-cell data with deep multitasking neural networks (SAUCIE). Nat Methods 16(11):1139–1145, 2019. | 10.1038/s41592-019-0576-7 | bioRxiv full text (via Semantic Scholar): MMD batch penalty and 'aligning them to a reference batch'. |
| Lotfollahi M et al. Conditional out-of-distribution generation for unpaired data using transfer VAE (trVAE). Bioinformatics 36(Suppl 2):i610–i617, 2020. | 10.1093/bioinformatics/btaa800 | OpenAlex abstract. |
| Shaham U et al. Removal of batch effects using distribution-matching residual networks. Bioinformatics 33(16):2539–2546, 2017. | 10.1093/bioinformatics/btx196 | OpenAlex abstract. |
| Wang T et al. BERMUDA: a novel deep transfer learning method for single-cell RNA sequencing batch correction. Genome Biol 20:165, 2019. | 10.1186/s13059-019-1764-6 | Genome Biology PDF text. |
| Cao K et al. A unified computational framework for single-cell data integration with optimal transport (uniPort). Nat Commun 13:7419, 2022. | 10.1038/s41467-022-35094-8 | OpenAlex abstract. |
| He C et al. Partially characterized topology guides reliable anchor-free scRNA-integration (scCRAFT). Commun Biol 8:561, 2025. | 10.1038/s42003-025-07988-y | OpenAlex abstract. |
| Luecken MD et al. Benchmarking atlas-level data integration in single-cell genomics (scIB). Nat Methods 19(1):41–50, 2022 (online 2021). | 10.1038/s41592-021-01336-8 | Nature Methods PDF text: Results, Methods (min-max), Data availability. |
| scib-metrics (YosefLab), v0.5.10. | https://github.com/YosefLab/scib-metrics | Installed package source read (dbench-models env): benchmark/_core.py. |
| Hrovatin K et al. Integrating single-cell RNA-seq datasets with substantial batch effects (sysVI). BMC Genomics 26:974, 2025. | 10.1186/s12864-025-12126-3 | BMC Genomics PDF text: introduction (proportion argument) and results passages. |
| Maan H et al. Characterizing the impacts of dataset imbalance on single-cell data integration. Nat Biotechnol 42(12):1899–1908, 2024. | 10.1038/s41587-023-02097-9 | PubMed abstract; Nature page (search result) on balanced metrics. |
| Yi C et al. Benchmarking deep learning methods for biologically conserved single-cell integration (scIB-E). Genome Biol 26:398, 2025. | 10.1186/s13059-025-03869-z | Genome Biology PDF text: 'Level-1: batch effect removal' section and methods (Ray Tune). |
| Rautenstrauch P, Ohler U. Shortcomings of silhouette in single-cell integration benchmarking. Nat Biotechnol 44(6):954–958, 2026 (online 2025). | 10.1038/s41587-025-02743-4 | OpenAlex abstract; PubMed record (search result). |
| Hu X et al. Reference-informed evaluation of batch correction for single-cell omics data with overcorrection awareness (RBET). Commun Biol 8:521, 2025. | 10.1038/s42003-025-07947-7 | OpenAlex abstract. |
| Crowell HL et al. The shaky foundations of simulating single-cell RNA sequencing data. Genome Biol 24:62, 2023. | 10.1186/s13059-023-02904-1 | OpenAlex abstract. |
| Zappia L, Phipson B, Oshlack A. Splatter: simulation of single-cell RNA sequencing data. Genome Biol 18:174, 2017. | 10.1186/s13059-017-1305-0 | OpenAlex record; scIB Methods text. |
| Reid KA, Stein-O'Brien G, Guven E. Topology Matters: the trade-off between Wasserstein critics and discriminators in single-cell data integration. bioRxiv 2025. | 10.1101/2025.11.12.688061 | OpenAlex abstract (full text not read; claims taken from the abstract and the task summary). |
| McCarter C. Confounder-aware feature correction for single-cell batch integration (ConDo). arXiv:2608.28849, 2026. | 10.48550/arXiv.2608.28849 | arXiv abstract page (search result) and OpenAlex record. |
| Sikkema L et al. An integrated cell atlas of the lung in health and disease (HLCA). Nat Med 29(6):1563–1577, 2023. | 10.1038/s41591-023-02327-2 | CELLxGENE curation API (collection 6f6d381a…: core and full datasets, assays, cell counts); CELLxGENE schema 5.3.0 requires integer UMI counts in raw.X. |
| Luecken MD et al. NeurIPS 2021 multimodal BMMC dataset ('A sandbox for prediction and integration of DNA, RNA, and proteins in single cells'). | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122 | GEO series text and supplementary listing (two processed BMMC h5ad files); counts layer not yet checked (provisional). |
| Tian L et al. Benchmarking single cell RNA-sequencing analysis pipelines using mixture control experiments (CellBench). Nat Methods 16(6):479–487, 2019. | 10.1038/s41592-019-0425-8 | PubMed abstract; GEO GSE118767 filelist (per-sample *_counts.csv.gz). |
| Ding J et al. Systematic comparison of single-cell and single-nucleus RNA-sequencing methods. Nat Biotechnol 38(6):737–746, 2020. | 10.1038/s41587-020-0465-8 | PubMed abstract; GEO GSE132044 listing (pbmc/cortex count_matrix.mtx). |
| Kang HM et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation. Nat Biotechnol 36(1):89–94, 2018. | 10.1038/nbt.4042 | PubMed abstract; GEO GSE96583 filelist (10x .mtx matrices). |
| Source accessions behind scIB pancreas/immune/lung (GSE81608 Xin; GSE86469 Lawlor; GSE85241 Muraro; GSE81076 Grün; GSE84133 Baron; GSE94820 Villani; GSE130148 lung Drop-seq). | https://ftp.ncbi.nlm.nih.gov/geo/series/ | GEO FTP supplementary listings fetched for each accession. GSE94820 first ~200 genes streamed: 99.1% of non-zero values non-integer, range ~0.9–7. |