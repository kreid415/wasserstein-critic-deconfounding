# Audit of the original submission (BIOINF-2026-1434, 'Topology Matters') against current knowledge

Scope: main text read in full (oup-authoring-template.tex, 436 lines), supplement.tex, reference.bib, result figures; submission-era code at commit cd64f05 and current code at 5709a98 (github.com/kreid415/wasserstein-critic-deconfounding, read-only clone); scvi-tools 1.4.2 and scib 1.1.7 source; source h5ad files; counts-pilot tables; the verbatim decision letter (recovered); WGAN, WGAN-GP and Arjovsky & Bottou PDFs (arXiv); IMAAE full text; ResPAN, AD-AE and Agueh-Carlier abstracts.

Claim status counts: invalidated_by_bugs 11, untested 6, partially_supported 5, contradicted 3 (n=25).

Framing rule applied (user direction): keep the original question, narrative and four Results sections; each item says how to make the ORIGINAL claim rigorous and what would confirm or refute it; new experiments only where a reviewer or the code audit requires them.

## 1. Claims

### C01 — partially_supported

**Claim.** Premise: the JS divergence 'suffers from vanishing gradients when distributions do not overlap, a common phenomenon when datasets come from different sequencing technologies, leading to failed training'; a discriminator 'can perfectly separate the batches, leading to gradient saturation and training instability'.

**Where.** Abstract/Motivation (l.104); Introduction para 2 (l.130)

**Original evidence.** Literature only (cites WGAN, key 10.5555/3305381.3305404). No experiment measured batch support overlap or showed a failing discriminator.

**Basis / rigorous test.** Checked WGAN (arXiv 1701.07875) Thm 1 and Example 1 (JS = log 2 for disjoint parallel lines) and Arjovsky & Bottou 2017 (arXiv 1701.04862) Thm 2.4 (vanishing generator gradients for the minimax cost as D approaches D*); the same paper states the -log D step 'doesn't necessarily suffer from vanishing gradients' but is unstable. So the theory covers the saturating / near-optimal-discriminator case, not automatically the bounded fool loss the plan now uses (verified in scvi-tools 1.4.2 AdversarialTrainingPlan.loss_adversarial_classifier(predict_true_class=False) and 5709a98 _fool_loss). 'Leading to failed training' is contradicted by the paper's own data: the 1-step discriminator integrated (higher ASW-batch in 6/6 Table 2 cells, higher ARI in 5/6). 'Common phenomenon' was never measured; the revision's E6 overlap numbers were computed on the double-normalised prep (audit C2) and must be recomputed. RIGOROUS TEST: (a) model-free support overlap per dataset and batch pair on the corrected prep; (b) log discriminator accuracy and generator-side ||grad_z L_adv|| per batch during training for disc-1, disc-10 and each critic. CONFIRM if on low-overlap datasets discriminator accuracy -> 1 while its generator gradient norm collapses and batch metrics stall, whereas critic gradients persist; REFUTE if discriminator gradients stay comparable to the critics' and it reaches comparable batch scores. Arjovsky & Bottou 2017 is not in reference.bib and should be cited.

### C02 — untested

**Claim.** Novelty: 'no study has rigorously isolated the adversarial objective' / an isolation 'has not been performed'.

**Where.** Abstract/Motivation (l.104); Introduction (l.132); Related Work (l.139)

**Original evidence.** Assertion; no systematic search reported.

**Basis / rigorous test.** One web search (this audit) returned only this work's own public versions: bioRxiv 10.1101/2025.11.12.688061 (v1 titled 'Wasserstein Critics Outperform Discriminators in Adversarial Deconfounding of Gene Expression Data'; v3 = the submitted abstract) and the project's GitHub README. No systematic literature search was done here, so absence of prior work is not established. RIGOROUS: structured search over adversarial / OT / MMD batch-correction methods (AD-AE, ResPAN, IMAAE verified below; MMD-ResNet is already in reference.bib but uncited; scDREAMER, an adversarial-classifier VAE, appeared in a search-result reference list); soften to 'to our knowledge' and cite the earlier preprint versions.

### C03 — partially_supported

**Claim.** Method contribution: 'We introduce a multi-class reference-based Wasserstein critic' (multi-headed, per-head GP on source-reference interpolates, stratified minibatches guaranteeing reference samples).

**Where.** Abstract/Results (l.105); Introduction (l.134); Methods 'Multi-Headed Wasserstein Critic' (l.203-224)

**Original evidence.** Implementation (cd64f05 src/wcd_vae/scCRAFT/networks.py ReferenceWassersteinLoss + multi_class_gradient_penalty).

**Basis / rigorous test.** Read cd64f05: the head and W1-dual loss exist as described, but (i) the binary and reference-sensitivity runs used the GP fallback D4 (interpolation toward random cells of the whole minibatch, every head incl. the reference head penalised), so the described GP was used only in the multibatch runs; (ii) the generator back-propagates the full W estimate including the reference term (z not detached), contradicting l.217-221. 5709a98 keeps the reference critic with the reference index passed explicitly (correct GP) and adds pooled and barycenter targets (src/wcd_vae/wcd/critic.py). RIGOROUS: state the implemented generator objective exactly and present the reference critic as one of three critic formulations.

### C04 — invalidated_by_bugs

**Claim.** 'The Wasserstein critic yields superior mixing' / 'is a stronger integrator' / 'achieved higher iLISI scores across all scenarios'.

**Where.** Abstract (l.105); Introduction (l.134); Results 1 'Integration Performance' (l.319); Results 3 (l.401); Discussion (l.422, l.424)

**Original evidence.** Table 2 iLISI critic vs discriminator, 6/6 cells: Pancreas 0.736/0.620 (p=0.0064) and 0.740/0.648 (p=0.0987); Immune 0.713/0.603 (4.8e-4) and 0.689/0.562 (0.0040); Lung 0.622/0.498 (0.0175) and 0.455/0.370 (0.0080); Fig 2; Fig 4 iLISI above the discriminator at every nominal lambda in {0.01..0.5}.

**Basis / rigorous test.** Produced by the submission pipeline with D1 (NB fit to log-normalised values), D2 (train/test PCA misalignment inside the LISI kNN graph), D3 (sampled z), D4 (GP fallback in the binary runs), D5 (critic 10 updates vs discriminator 1), D6 (unbounded discriminator loss), on the scCRAFT backbone, iLISI-centric. Re-testing Table 2's 36 reported p-values with Holm leaves 6/36 significant and only Immune-balanced among the iLISI rows. Only valid evidence now: atac_small counts pilot (PROVISIONAL: 1 dataset, 3 seeds, 239 ep / batch 512 / n_latent 30, i.e. before the stock-default switch; discriminator still 1-step with the unbounded max-CE loss; reference = alphabetical index 0; scIB overall min-max scaled within the pilot's own run set). There, at lambda=1 (conditioned) the reference critic has iLISI 0.681 vs discriminator 0.556 (batch score 0.808 vs 0.669), but the discriminator reaches iLISI 0.782 at lambda=20 and overall scores tie (0.646 vs 0.640). RIGOROUS: per dataset x decoder, batch-score vs bio-score Pareto fronts from the fixed lambda grid (5 seeds) for reference/pooled/barycenter critics vs disc-1 and disc-10 (bounded loss). CONFIRM if the critic front gives a higher batch score at matched bio score (seed CIs separated; paired test across the 8 datasets); REFUTE if fronts overlap or the discriminator front dominates. Matched-nominal-lambda comparisons are not admissible (CE and W-estimate scales differ).

### C05 — invalidated_by_bugs

**Claim.** Local/global discrepancy: ASW-batch 'did not show the same effect'; 'the discriminator effectively aligns global distributions, the critic better resolves fine-grained batch structures within local cell neighborhoods'.

**Where.** Results 1 'Integration Performance' (l.319)

**Original evidence.** Table 2 ASW-batch higher for the discriminator in 6/6 cells (Pancreas 0.894/0.930, 0.895/0.932; Immune 0.853/0.872, 0.873/0.878; Lung 0.870/0.910, 0.861/0.898) while iLISI favoured the critic.

**Basis / rigorous test.** Same defects; D2 specifically corrupts the kNN-based (local) metrics, whereas ASW was computed on test cells alone. R1.major.1 asks for this to be a central result. Provisional counterexample from the atac_small counts pilot (PROVISIONAL: 1 dataset, 3 seeds, 239 ep / batch 512 / n_latent 30, i.e. before the stock-default switch; discriminator still 1-step with the unbounded max-CE loss; reference = alphabetical index 0; scIB overall min-max scaled within the pilot's own run set): at lambda=1 the reference critic leads on BOTH iLISI (0.681 vs 0.556) and ASW-batch (0.972 vs 0.957). RIGOROUS: per-metric lambda curves for iLISI, kBET, graph connectivity (local) vs ASW-batch, PCR comparison (global); per-dataset Spearman between local and global batch metrics across configs; test whether the critic-minus-discriminator sign differs between scales at matched lambda and at matched batch score. CONFIRM if the sign flips between local and global metrics on most datasets; REFUTE if the orderings agree.

### C06 — invalidated_by_bugs

**Claim.** The critic over-corrects: 'reduction in cell-type purity', 'prone to over-correction resulting in collapsed cellular representations'; consistent integration-conservation trade-off.

**Where.** Abstract (l.105); Introduction (l.134); Results 1 'Biological Conservation' and 'Summary' (l.321, l.325); Discussion (l.424, l.432)

**Original evidence.** Table 2: cLISI worse for the critic 6/6, ASW-cell lower 6/6, ARI lower on Pancreas and Immune (0.670/0.942, 0.692/0.938, 0.447/0.556, 0.439/0.554); Fig 4 critic cLISI rising with lambda.

**Basis / rigorous test.** Pipeline defects D1-D6, AND the backbone contained a triplet loss added explicitly 'To mitigate latent over-correction' (l.172) plus a cosine term at weight 20, so the conservation axis was largely set by auxiliary losses that the stock-scVI plan removes. The abstract attributes over-correction to the 'reference sensitivity analysis', which (Fig 5) only measured iLISI/cLISI spread over references. A later native-backbone wave (also on the C2-bugged prep) reversed the direction; neither is valid now. atac_small counts pilot (PROVISIONAL: 1 dataset, 3 seeds, 239 ep / batch 512 / n_latent 30, i.e. before the stock-default switch; discriminator still 1-step with the unbounded max-CE loss; reference = alphabetical index 0; scIB overall min-max scaled within the pilot's own run set): critics' bio score collapses at lambda>=5 (reference 0.220, pooled 0.227) while the discriminator keeps 0.551 at lambda=5 but collapses at lambda=20 (0.296): every arm over-corrects, critics at lower nominal lambda. RIGOROUS: compare bio metrics (cLISI, ARI/NMI, isolated-label F1/ASW) and direct readouts (per-type purity, rare-type recall, label-transfer F1, sim1/sim2 ground truth) at MATCHED batch score, relative to the lambda=0 control. CONFIRM if at equal batch score the critic's bio is lower on most datasets (CI-separated); REFUTE if equal or higher.

### C07 — invalidated_by_bugs

**Claim.** PAGA: 'a trend of cell type collapse'; Discussion: the manifold 'was more warped in the case of the Wasserstein critic, forcing unrelated cell types together'.

**Where.** Results 1 (l.321); Summary (l.325); Discussion (l.424)

**Original evidence.** PAGA Spearman lower for the critic in 6/6 cells but p<0.05 only in Lung-balanced (1.2e-4; survives Holm); the other five p=0.057-0.44.

**Basis / rigorous test.** Custom metric (cd64f05 hyperparameter.py compute_mean_paga_spearman: PAGA on the ~20% held-out test cells vs each technology's raw-PCA PAGA) computed on D1-D3 embeddings; the Discussion upgrades a 1/6-significant trend to a mechanism. Not part of scIB-native scoring (trajectory conservation is disabled: no pseudotime annotations). RIGOROUS: replace with cell-type confusion matrices / per-type neighbourhood purity on all cells with seeds (R2.6). CONFIRM 'forcing unrelated types together' only if confusion between distinct types rises for critics at matched batch score.

### C08 — contradicted

**Claim.** Sensitivity to confounding structure: 'no systematic pattern across datasets' (discriminator iLISI dropped on Immune but 'actually improved in the Pancreas dataset'; critic dropped on Immune and Lung).

**Where.** Results 1 'Sensitivity to Confounding Structure' (l.323)

**Original evidence.** Table 2 balanced vs unbalanced columns.

**Basis / rigorous test.** The scenario manipulation did not do what it claims. Computed from Supplementary Table S2: the 'balanced' binary sets keep strong batch-cell-type dependence (Cramer's V 0.33 pancreas, 0.42 lung, 0.33 immune vs 0.33/0.54/0.45 unbalanced), and pancreas balanced differs from unbalanced by just the 2 inDrop3 t_cells (3,605 -> 3,603 cells), so the Pancreas 'improvement' (0.620 -> 0.648) compares near-identical data and is run-to-run variation. RIGOROUS: define 'balanced' as composition-matched (subsample every batch to identical cell-type proportions) vs natural composition, 5 seeds, test the scenario x method interaction. CONFIRM a confounding effect only if unbalanced degrades batch or bio metrics beyond seed noise.

### C09 — invalidated_by_bugs

**Claim.** 'The discriminator acts as a conservative regularizer'; the '"weak" integration characteristic of discriminators acts as a protective measure against over-correction'.

**Where.** Abstract (l.105); Results 1 'Summary' (l.325)

**Original evidence.** Aggregate of Table 2 patterns.

**Basis / rigorous test.** Confounded by D5/D6: the critic took 10 adversary updates per encoder step and the discriminator 1 (cd64f05 hyperparameter.py:195), with an unbounded generator loss; nominal lambda is not comparable across objectives. Weaker integration conserving more is the trade-off curve itself, not protection. RIGOROUS: include the budget-matched disc-10 arm and compare at matched batch score. CONFIRM 'protective' only if, at equal batch score, the discriminator conserves more biology; REFUTE if the advantage disappears when integration is matched.

### C10 — invalidated_by_bugs

**Claim.** 'Integration difficulty increases with the number of batches'; 'a decrease in performance across all scenarios for both methods'.

**Where.** Results 2 'Performance Decay' (l.383)

**Original evidence.** Fig 3 (iLISI and cLISI vs number of batches; pancreas 2->9, immune 2->4; balanced and unbalanced).

**Basis / rigorous test.** Multibatch runs used the correct reference GP but D1-D3, D5, D6 apply. Part of the decay is a metric ceiling: scaled iLISI cannot exceed about (1/sum p_b^2 - 1)/(V-1) under perfect mixing; from Table S2 sizes this drops 0.965 -> 0.703 when immune's 1,022-cell smart-seq2 batch is added (3 -> 4 batches) and 0.933 -> 0.807 for pancreas (3 -> 9), comparable to the discriminator's immune decline in Fig 3. The balanced sets also change composition with V: balanced immune goes from 10 shared types (29,282 cells) at V=3 to 4 myeloid/DC types (8,928 cells) at V=4, and balanced pancreas drops to alpha/beta/delta/gamma once SMARTer enters (V>=6). RIGOROUS: ceiling-normalised iLISI (or batch-size-balanced subsamples), full scIB suite, composition held fixed across V.

### C11 — invalidated_by_bugs

**Claim.** The critic 'scales poorly with the number of batches'; 'The Discriminator Out-scales the Wasserstein Critic'; the critic 'degrades significantly faster'.

**Where.** Abstract (l.105); Introduction (l.134); Results 2 (l.385); Discussion (l.426)

**Original evidence.** Fig 3 iLISI crossovers (critic higher at V=2, lower from V~3-4); iLISI only; no statistical test.

**Basis / rigorous test.** Reviewer R1.minor.1 objected to iLISI-only evidence. Defects as above; 'significantly faster' was never tested; ceiling and composition confounds (previous item). The revision's E8 numbers came from an invalidated pipeline. RIGOROUS: batch-count series (batches added in descending size, as originally) on >=3 datasets with >=4 count batches, 5 seeds, full scIB suite, all formulations incl. pooled/barycenter and disc-10, at the pre-registered lambda (or per-V Pareto fronts). CONFIRM if the critic-minus-discriminator difference in ceiling-normalised batch score and in bio score declines with V (interaction slope CI excludes 0) on most datasets; REFUTE if flat.

### C12 — untested

**Claim.** Mechanism: a single-reference 'topological bottleneck' / 'uncoordinated radial projection topology creates a geometric bottleneck', whereas the discriminator's joint V-way objective 'implicitly coordinates the latent geometry'.

**Where.** Introduction (l.134); Results 2 (l.385); Discussion (l.426)

**Original evidence.** None beyond the iLISI trend; no reference-free critic was run.

**Basis / rigorous test.** Flagged by R1.minor.1, R2.3 and R2.5. Now directly testable with the pooled and barycenter critics (no privileged batch; 5709a98 critic.py). CONFIRM a reference-design effect if the reference critic's scaling deficit vanishes for pooled/barycenter and tracks reference coverage; attribute it to the W objective only if every critic formulation degrades while the discriminator does not. Until tested, write 'single-reference design' without geometric language.

### C13 — partially_supported

**Claim.** Because the scaling drop 'occurs equally in both biologically balanced and unbalanced scenarios, it cannot be attributed to the reference batch lacking specific cell types'; hence 'a fundamental architectural limitation'.

**Where.** Discussion 'Structural Bottlenecks' (l.426)

**Original evidence.** Fig 3 balanced vs unbalanced panels.

**Basis / rigorous test.** The exclusion logic holds for the balanced sets (all batches share the same cell types by construction: cd64f05 data.py balance = intersection), but the evidence comes from the defective pipeline, the balanced composition itself changes with V (see 'Performance Decay'), and 'architectural limitation' requires the pooled/barycenter control.

### C14 — contradicted

**Claim.** Critic performance 'is dependent on a topologically dense reference batch'; iLISI outliers 'are strictly driven by the cell-type representation present in the target reference batch'.

**Where.** Abstract (l.105); Results 4 'Sensitivity to Batch Completeness' (l.411); Discussion (l.428)

**Original evidence.** Fig 5 box plots over every reference choice (iLISI roughly 0.36-0.44 pancreas balanced, 0.32-0.40 unbalanced, 0.33-0.40 immune); no per-reference coverage analysis.

**Basis / rigorous test.** Contradicted by the paper's own design: in the balanced scenario every candidate reference contains the same cell-type set, yet the pancreas balanced spread is as large as the unbalanced one (read from Fig 5), so cell-type representation cannot be the 'strict' driver; no coverage covariate was analysed. These runs also used the GP fallback D4 and alphabetical reference indices. RIGOROUS (new experiment, original section 4): reference sweep, every batch as reference, >=3 seeds each, at the pre-registered lambda on pancreas (up to 8 batches once CEL-seq inversion / Fluidigm rounding are added and SMARTer is excluded), immune (4), sim1 (6), atac_large (11 via batchname_all); regress batch/bio scores on reference coverage (fraction of dataset cell types present), Shannon entropy and size; seed variance as the noise floor; composition-matched negative control. CONFIRM if coverage explains variance beyond size and seed noise; REFUTE otherwise. The plan's entropy rule fixes a single reference and does not test this.

### C15 — untested

**Claim.** 'Because the Wasserstein objective enforces a strict optimal transport plan', query-specific populations are projected onto unrelated reference populations, 'effectively erasing true biological variance'.

**Where.** Results 4 (l.411)

**Original evidence.** None (mechanistic narrative).

**Basis / rigorous test.** W1 between normalised batch distributions does require matching all mass, but the critic estimates W1 through its dual and no transport plan is computed, and no per-cell-type evidence was shown (R2.3: 'which cell types collapse'). RIGOROUS: with a reference lacking type T (natural, or T held out of the reference), measure where query cells of type T land (label-transfer predictions / confusion into reference types) for the reference critic vs pooled/barycenter and the discriminator. CONFIRM if type-T cells are absorbed into reference types specifically for the reference critic.

### C16 — invalidated_by_bugs

**Claim.** 'Both the Wasserstein critic and the discriminator exhibit broad optimal operating windows on the iLISI metric, with the critic performing consistently higher'.

**Where.** Results 3 'Robustness to Regularization' (l.401); Fig 4

**Original evidence.** Fig 4 = inner-CV validation records (40-epoch models), lambda in {0.01, 0.05, 0.1, 0.2, 0.5}.

**Basis / rigorous test.** D1-D3, D5-D7; the grid spans 1.7 decades and the discriminator's lung iLISI is still rising at lambda=0.5 (optimum not bracketed); nominal lambda is not comparable across objectives. Because the submission's NB term was averaged over 2,000 genes, its lambda grid corresponds to about 20-1000 in summed-NB (scVI) units, so it cannot be ported. Pilot critics peaked at the lambda=1 grid floor (bracketing failed again). RIGOROUS: fixed log grid bracketing each arm's optimum and collapse (planned 0.01-30; check the upper end reaches discriminator collapse on every dataset), lambda curves with 5-seed bands, plus curves against achieved batch score. CONFIRM 'broad window' if the score stays within the seed CI of its maximum over >=1 decade of lambda.

### C17 — invalidated_by_bugs

**Claim.** 'For the cLISI metric, the discriminator maintains a broad, flat curve that is highly insensitive to the choice of adversarial coefficient'; the critic's cLISI rises with lambda.

**Where.** Results 3 (l.401); Fig 4

**Original evidence.** Fig 4 cLISI panels.

**Basis / rigorous test.** Same defects and grid limits. Provisional counterexample from the atac_small counts pilot (PROVISIONAL: 1 dataset, 3 seeds, 239 ep / batch 512 / n_latent 30, i.e. before the stock-default switch; discriminator still 1-step with the unbounded max-CE loss; reference = alphabetical index 0; scIB overall min-max scaled within the pilot's own run set): the conditioned discriminator's bio score falls with lambda (0.621, 0.551, 0.296 at lambda 1, 5, 20; isolated-label F1 0.826 -> 0.490). RIGOROUS: per-metric bio lambda curves with seed bands; claim insensitivity only if bio stays within the seed CI across the bracketed grid.

### C18 — contradicted

**Claim.** Wasserstein models need 'approximately ~24% more training time per epoch'; the discriminator 'achieves faster convergence'.

**Where.** Methods (l.240); Supplementary Note 1 (Fig S1 caption, Table S1)

**Original evidence.** Table S1: critic minus discriminator wall-clock +19.6% to +27.5%.

**Basis / rigorous test.** Holds only for the large scCRAFT backbone, where 10 critic steps on detached 256-d z were cheap relative to the 1024/512-unit VAE. On the plan's stock scVI (128 hidden units) the measured cost (project context: atac_small, batch 128, RTX 3080, single lane) is ~7 s/epoch reference critic and ~9 s pooled vs ~1 s discriminator (~2 s at 10 steps), i.e. 3.5-7x. Convergence was never measured (fixed epoch budgets). RIGOROUS: report per-epoch and per-fit wall-clock for every arm on one GPU, including disc-10.

### C19 — invalidated_by_bugs

**Claim.** 'In scenarios with strong confounding or disjoint distributions, the discriminator frequently failed to mix local neighborhoods effectively. In contrast, the Wasserstein critic forced mixing even when batches were initially far apart.'

**Where.** Discussion (l.422)

**Original evidence.** Table 2 iLISI (discriminator 0.370-0.648 scaled).

**Basis / rigorous test.** Pipeline defects; initial separation was never quantified; scaled iLISI of 0.37-0.65 is not a failure to mix, and lung's perfect-mixing ceiling is about 0.53 (Table 1 sizes 22,771 vs 6,485), so the discriminator's 0.370 is ~70% of attainable. RIGOROUS: relate per-dataset gains to measured pre-integration support overlap (R1.major.2). CONFIRM if the critic's batch-score advantage (at matched bio) grows as overlap decreases.

### C20 — untested

**Claim.** Practical recommendation: the Wasserstein objective 'is ideal for scenarios with severe, disjoint batch effects and overlapping cell types, whereas the discriminator scales more effectively in fragmented atlases with disjoint cell types or sensitive cellular boundaries'.

**Where.** Introduction (l.134); implied in Abstract (l.105)

**Original evidence.** None: no experiment varied batch-effect severity against cell-type overlap.

**Basis / rigorous test.** RIGOROUS: stratify datasets by measured support overlap and cell-type overlap (immune_hum_mou cross-species as the disjoint control; sim1/sim2 controllable) and test the objective x regime interaction on the lambda curves. Keep the recommendation only if that interaction is significant.

### C21 — untested

**Claim.** A Wasserstein barycenter (learned 'virtual' center, 'the geometric median of all batch distributions') would combine the critic's gradients with the discriminator's global robustness.

**Where.** Discussion 'From Reference Matching to Wasserstein Barycenters' (l.428)

**Original evidence.** None (proposal).

**Basis / rigorous test.** Now in the plan (5709a98 barycenter = W1 critic against 64 learnable anchors moved by the generator optimiser). Cautions: audit M2 found barycenter configs shrink the latent (max|z|<1 in 78-100% of configs at lambda>=50), so the zstd pilot must precede conclusions; and 'geometric median' mis-describes the cited Agueh-Carlier object (abstract: a barycenter generalising McCann's interpolation, i.e. a mean). CONFIRM if the barycenter matches the reference critic's batch score with discriminator-level bio and no reference or scaling sensitivity.

### C22 — invalidated_by_bugs

**Claim.** 'The primary limitation of the Wasserstein objective in its isolated form is its propensity for biological over-correction'; future work should use adaptive / annealed adversarial weights.

**Where.** Discussion (l.424, l.432)

**Original evidence.** Summary of the claims above.

**Basis / rigorous test.** Depends on the over-correction and protection claims. An annealing schedule already ships in stock scvi-tools: AdversarialTrainingPlan uses kappa = 1 - kl_weight when scale_adversarial_loss='auto' (verified, 1.4.2 source); testing it as one arm answers R2.2's 'scheduling strategy' with the tool people actually use.

### C23 — partially_supported

**Claim.** The Wasserstein distance 'remains a continuous and differentiable metric even when distributions are non-overlapping, providing robust gradients that drive alignment regardless of the initial separation'.

**Where.** Introduction (l.132); Abstract (l.104)

**Original evidence.** Literature (WGAN).

**Basis / rigorous test.** WGAN Thm 1 (verified): W is continuous everywhere and differentiable almost everywhere if the generator is locally Lipschitz and a regularity assumption holds; false for JS/KL. A gradient-penalty critic only approximates the 1-Lipschitz potential (soft constraint, WGAN-GP), so 'robust gradients regardless of separation' describes the ideal distance, not the estimator. Reword accordingly.

### C24 — partially_supported

**Claim.** Prior Wasserstein methods (ResPAN, IMAAE) add auxiliary steps that obscure the adversary's contribution; IMAAE attributes reference-alignment failure to a 'lack of constraints' with sparse anchors, 'motivating their decision to enforce a global Gaussian prior'.

**Where.** Introduction (l.132); Related Work (l.139)

**Original evidence.** Citations.

**Basis / rigorous test.** Verified: ResPAN abstract (WGAN + random-walk mutual-nearest-neighbour pairing + fully skip-connected autoencoders); AD-AE abstract (autoencoder + adversary predicting the confounder). IMAAE full text (PMC10056671): 'lack of constraints' occurs for an autoencoder experiment in which batches with fewer cell types are anchors (their Fig 11a); IMAAE keeps an anchor batch as decoder target and adversarially matches q(z) to N(0,I) with a WGAN-GP-form loss (the word 'Wasserstein' does not appear). It also reports a GAN that works on two batches but under-corrects with three or more, which is prior evidence relevant to the scaling question. The 'motivating their decision' link is the authors' interpretation; rephrase.

### C25 — untested

**Claim.** The unbalanced scenario is 'known to drive severe over-correction and the ablation of biological signal in standard adversarial deconfounding'.

**Where.** Experiments 'Confounding Scenarios' (l.256)

**Original evidence.** None (uncited).

**Basis / rigorous test.** Needs a citation or a test via the composition-matched vs natural design (see 'Sensitivity to confounding structure').

## 2. Reviewer comments (17)

### R1.major.1

**Must show.** Original claim to make rigorous: l.319 'the critic better resolves fine-grained batch structures within local cell neighborhoods' while the discriminator aligns global distributions. Show, per dataset x decoder, whether the critic-vs-discriminator ordering differs between local batch metrics (iLISI, kBET, graph connectivity) and global ones (ASW-batch, PCR comparison), across the lambda grid and at matched batch/bio level; quantify the decoupling (per-dataset rank correlation between iLISI and ASW-batch/PCR across configs) and discuss it in Results section 1 as a primary result. Confirm if the sign flips between scales on most datasets; refute if orderings agree (provisional pilot counterexample: atac_small lambda=1, the reference critic leads on both iLISI 0.681 vs 0.556 and ASW-batch 0.972 vs 0.957).

**Type.** Analysis of the planned runs (no new training) + Results/Discussion text.

**Covered by current plan.** PARTIAL. scIB-native scoring computes all five batch metrics, but the planned figures show only the aggregated overall score; add per-metric lambda curves (scripts/build_metric_curves.py exists) and the decoupling statistic. The earlier E6 numbers (iLISI 0.05-0.45 swing, r=0.38) come from the invalidated pipeline and must not be reused.

### R1.major.2

**Must show.** The reviewer raised disjoint support as a FAIRNESS concern for the JS discriminator ('questions may arise regarding the fairness of the comparison'), especially in the unbalanced scenario. Show: (a) pre-integration, model-free support overlap per dataset and batch pair on the corrected counts-based prep (chance-normalised batch-classifier accuracy, kNN cross-batch fraction, per-cell-type overlap); (b) the premise in action: discriminator accuracy and generator-side ||grad_z L_adv|| over training for disc-1, disc-10 and each critic on high- vs low-overlap datasets; (c) fairness by construction: non-saturating bounded fool loss and an equal 10-step update budget for the discriminator; (d) text on the expected behaviour of each objective (WGAN Thm 1/Example 1; Arjovsky & Bottou 2017 Thm 2.4-2.5).

**Type.** Analysis (support overlap; scripts/support_overlap.py exists) + light training instrumentation (log gradient norms/accuracy in the planned runs) + text.

**Covered by current plan.** PARTIAL. (c) is covered (5709a98: bounded fool loss, disc_iter honoured, disc at 1 and 10 steps). (a) must be re-run because E6 used the double-normalised prep (audit C2). (b) is not in the plan. The earlier response argued 'premise holds, therefore fair'; reverse that framing.

### R1.minor.1

**Must show.** Original claims to make rigorous: 'scales poorly with the number of batches' (l.105/l.385) and the 'topological bottleneck' intuition (l.426). Show a batch-count series (2 -> V, batches added in descending size as in the original) with the full scIB suite, iLISI reported relative to its perfect-mixing ceiling (immune ceiling 0.965 -> 0.703 when the 1,022-cell smart-seq2 batch is added) or with batch-size-balanced subsamples, composition held fixed across V, 5 seeds, all formulations. Test the intuition directly: reference vs pooled vs barycenter critics vs the joint V-way discriminator. Confirm 'critic scales poorly' if the critic-minus-discriminator gap in ceiling-normalised batch and bio scores widens with V (negative interaction slope) on >=3 datasets; attribute it to the single reference only if pooled/barycenter do not show it.

**Type.** NEW EXPERIMENT required by this comment and by original Results section 2: batch-count series on datasets with >=4 count batches (pancreas up to 8 once CEL-seq/CEL-seq2 inversion and Fluidigm C1 rounding are added and SMARTer RPKM is excluded; immune 4; sim1 6; atac_large 11 via batchname_all; immune_hum_mou chemistry 5 or batch 23; lung donors 16 only if counts are recovered).

**Covered by current plan.** NO for the series (the plan runs each dataset at its full batch set only). YES for the formulations and discriminator budgets needed to interpret it.

### R1.minor.2

**Must show.** Downstream biology inside the quantitative framework: automatic annotation via cross-batch label transfer (classifier trained on one batch, predicted on the others; macro-F1, per-type F1, rare types) and marker-gene preservation (marker sets from unintegrated data; for scVI-based arms, batch-transformed decoder expression and DE recovery, or graph-based marker localisation on the integrated kNN graph), compared at matched batch score.

**Type.** Post-hoc analysis on saved latents/models (run_biology.py exists from E5; its numbers are invalid).

**Covered by current plan.** NO. Decide before the run whether trained models (not only latents) are kept, because marker/DE analyses need the decoder.

### R1.minor.3

**Must show.** Implementation-faithful equations for the NEW methods: ZINB/NB likelihood with library size and gene-specific inverse dispersion theta_g, summed over genes (scvi-tools 1.4.2 VAE.loss); KL with its warm-up weight; generator loss L_scVI + lambda*L_adv; discriminator h_psi: R^d -> R^|V| logits trained with cross-entropy, generator using the bounded fool loss; critic heads C_rho: R^d -> R^|V| with W1 dual + gradient penalty; formulation-specific targets (reference / pooled / barycenter anchors); distinct symbols (the original used upsilon next to v, and r for both dispersion and the reference index).

**Type.** Text only (Methods rewrite).

**Covered by current plan.** NO (text). The E0 memo's corrections are not reusable: they describe the scCRAFT code and misstate it (NB 'summed over genes' whereas cd64f05 averages over cells x genes; theta_g gene-specific whereas px_r is cell x gene; 'the prior form omitted the normalisation' whereas the submitted Eq. 2 already has L2 norms). The reviewer quoted mu_g, r_g, <x_g, ...> and h_v, which do not appear in the zip version; confirm which version was reviewed.

### R2.1

**Must show.** Whether the original trade-off holds across adversarial variants and backbones: >=2 discriminator variants (1 vs 10 steps; bounded loss), 3 critic formulations (reference / pooled / barycenter), 2 decoders (SCVI-ZINB nonlinear, LinearSCVI-NB), conditioned and unconditioned; claims scoped to 'adversarial regularisers added to scVI-family VAEs'.

**Type.** Experiment (in plan) + claim scoping in text.

**Covered by current plan.** YES. Recommended addition: stock scvi-tools' own adversarial classifier (AdversarialTrainingPlan, Classifier n_hidden=32, n_layers=2, kappa = 1 - kl_weight) as the canonical 'discriminator as people use it', because the plan's discriminator uses the authored 128-128 head with a fixed lambda.

### R2.2

**Must show.** Original claim to make rigorous: the critic 'is prone to over-correction'. Show full lambda-response curves and batch-vs-bio Pareto fronts per dataset x decoder x formulation with 5-seed CIs on a grid that brackets every arm's optimum and its collapse (pilot critics peaked at the lambda=1 floor; unconditioned discriminator at lambda=20); compare at matched batch score (and matched bio), never at matched nominal lambda; add one scheduling strategy (scvi-tools' kappa = 1 - kl_weight annealing, verified in AdversarialTrainingPlan). Confirm 'intrinsically worse trade-off' only if the critic front lies below the discriminator front at equal batch score.

**Type.** Experiment (grid and seeds in plan) + analysis (Pareto, matched comparisons) + one small schedule arm.

**Covered by current plan.** PARTIAL. Fixed grid (0.01-30), 5 seeds and a primary lambda-curve figure are planned; Pareto / matched-batch analysis and the schedule arm are not; verify the grid's upper end reaches collapse for the discriminator on every dataset. The original lambda grid (0.01-0.5) maps to about 20-1000 in summed-NB units, so it cannot be reused.

### R2.3

**Must show.** Evidence behind the mechanistic language: which cell types collapse (per-type purity, label-transfer F1 and confusion matrices vs the lambda=0 control and vs the discriminator at matched batch score), which references fail (per-reference results), and how reference coverage explains them (regress scores on coverage / entropy / size, with a composition-matched negative control). Use 'topology' or 'bottleneck' only where these tests support it.

**Type.** NEW EXPERIMENT: reference sweep (also original Results section 4) + post-hoc per-type analysis + text.

**Covered by current plan.** NO. The plan fixes one entropy-chosen reference per dataset and has no per-cell-type analysis.

### R2.4

**Must show.** Stronger external baselines and more datasets on identical scoring, or explicit narrow-ablation framing.

**Type.** Experiment (in plan) + framing text.

**Covered by current plan.** YES in design (8 datasets; 6 baselines including stock scVI re-run with identical epochs/batch/latent; scIB-native). BLOCKER: as committed (5709a98) only atac_small and immune have been prepped on counts; pancreas (CEL-seq/CEL-seq2 UMI-collision-corrected, Fluidigm C1 expected counts, SMARTer RPKM), lung (10x v2 batch = 22,771 of 32,472 cells non-integer) and immune_hum_mou (integrality-table non-integer fraction 0.072 for the Human batch) will be refused by the integer check without registry decisions, so two of the three original datasets are currently unusable. Also fix deterministic-baseline pseudo-replication (audit M6).

### R2.5

**Must show.** Separate the objective from the reference design: reference-free critics (pooled, barycenter) vs the reference critic vs the discriminator on the scaling and reference-sensitivity endpoints; attribute failures to the reference design only if pooled/barycenter avoid them; remove the latent-scale shortcut (zstd) before attributing.

**Type.** Experiment (formulations in plan) + the scaling series and reference sweep (new) + text.

**Covered by current plan.** PARTIAL. Formulations and the zstd pilot are planned; the endpoints that showed the original weakness (batch-count scaling, reference sweep) are not. Audit M2 (barycenter latent shrinkage) must be resolved first.

### R2.6

**Must show.** Direct biological validation of over-correction: marker preservation, cell-type confusion, rare-cell retention and per-cell-type neighbourhood purity, plus ground truth on sim1/sim2, at matched batch score.

**Type.** Post-hoc analysis on saved latents/models.

**Covered by current plan.** NO (E5 code exists; earlier E5 numbers are invalid).

### R3.1

**Must show.** Relation to scVI and current methods: adversaries as add-ons to stock scVI (does any adversary improve on stock scVI trained with identical epochs/batch/latent?), placement against Harmony/Scanorama/scANVI etc., and whether the trade-off is backbone-specific (SCVI vs LinearSCVI; conditioned vs unconditioned).

**Type.** Experiment (in plan) + positioning text.

**Covered by current plan.** YES. Pilot hint (provisional): no conditioned adversary beat stock scVI (0.726 vs <=0.646), but stock used a different schedule; the equal-schedule rule now controls this.

### R3.2

**Must show.** Robustness of the head comparison to loss weights, depth, latent dimensionality and training strategy.

**Type.** Small sensitivity experiment on 1-2 datasets at the pre-registered lambda (n_latent 10 vs 30; n_layers 1 vs 2; max_kl_weight 1 vs lower; batch size/epochs) + existing decoder/conditioning/lambda/step axes.

**Covered by current plan.** PARTIAL. Decoder type, conditioning, lambda and discriminator steps are covered; latent dim, depth, KL weight and batch size are not. The pilot shows schedule alone moved the lambda=0 control from 0.635 to 0.726 (stock defaults). The earlier response's statement that scIB scoring 'requires >=50 embedding dimensions' is wrong: scib 1.1.7 pc_regression caps n_comps at min(matrix.shape).

### R3.3

**Must show.** When to prefer each objective, derived from selection-free lambda curves, Pareto fronts, scaling and reference results; a label-free lambda-selection recipe; mitigation strategies actually tested (kappa schedule, zstd, barycenter).

**Type.** Text + analysis.

**Covered by current plan.** PARTIAL (zstd pilot and barycenter planned; schedule arm and guidance analysis not).

### R3.4

**Must show.** Either one atlas-scale dataset (>=5e5 cells) run at the pre-registered lambda with a reduced arm set, or explicit scoping of claims to <=1e5 cells. With the stock epoch heuristic min(400, 20000/n*400) the total cell-updates stay at about 8e6 for n>=20k, so a large fit costs about the same as a 20k-cell fit; scIB ASW/kBET need subsampling at that scale.

**Type.** Optional experiment or scoping text.

**Covered by current plan.** NO (largest planned: immune_hum_mou ~98k cells, atac_large ~85k).

### R3.minor.1

**Must show.** Whether the trade-off generalises beyond scRNA-seq (snRNA-seq or other modalities).

**Type.** Experiment (in plan) + scoped text.

**Covered by current plan.** PARTIAL. Two scATAC gene-activity datasets (integer counts verified in the integrality audit) are planned; snRNA-seq is not. Describe them as gene-activity matrices, not peak-level scATAC.

### R3.minor.2

**Must show.** Downstream consequences: label transfer, differential expression and trajectory inference.

**Type.** Post-hoc analysis: label transfer on latents; DE via the scVI decoder on 1-2 datasets; trajectory only on a dataset with a defined trajectory and pseudotime reference (otherwise state why it is omitted).

**Covered by current plan.** NO (scIB trajectory conservation is disabled in score_scib_native.py for lack of pseudotime annotations).

## 3. Methods statements that conflict with the new implementation or are non-standard

1. DEFECT LEGEND (submission-era code read at commit cd64f05, the last commit before the July revision; codes used in the claims table). D1 counts bug: src/wcd_vae/data.py:49 sets layers['counts'] = X.copy() before normalisation; the scIB source files' X is log-normalised (checked: Immune_ALL_human X max 10.18 and all stored values non-integer; Lung_atlas_public X max 5.81, non-integer; pancreas X max 11.4 in the first 200 rows) while their layers['counts'] are integer, so the NB was fit to log-normalised values and the encoder saw log1p(normalise(log-normalised)). D2: model.py obtain_embeddings fits a new PCA(50) on every call and hyperparameter.py:334-338 calls it separately on train and test embeddings before concatenating them into one kNN graph, so test-node LISI mixes two non-aligned coordinate frames (same for inner-CV lambda selection). D3: the evaluated embedding is a reparameterised sample z (Encoder.forward always returns rsample), not the posterior mean. D4: hyperparameter_search.py:74 passes a reference name only for --reference_batch -1, so the binary (no flag) and reference-sweep (index flag) runs hit the GP fallback in networks.py (ref_samples = whole minibatch; every head penalised); only multibatch runs used the described source-reference GP. D5: hyperparameter.py:195 iters = disc_iter if critic else 1 (critic 10 updates, discriminator 1), both reusing one detached minibatch z. D6: discriminator generator loss -lambda*CE (unbounded). D7: lambda chosen by iLISI - 10*cLISI from 40-epoch inner models (warmup 10) while reported outer models trained 80 epochs with warmup 5 (warmup_epoch not passed at hyperparameter.py:308-319; default 5 at model.py:353). D8: custom LISI with @njit(fastmath=True) and a fallback to k (lockfile pins numba 0.61.2; the miscompile was documented on 0.66; Table 2 values are not the constant-fallback pattern, so wholesale corruption is unlikely, partial fallback unverifiable). D9: single seed (set_seed(42)); 5 overlapping outer folds used as replicates; 95% CIs in Figs 2-5 not corrected for fold overlap; 36 Table 2 tests without multiplicity control.
2. Materials and Methods intro (l.142) 'fixed Variational Autoencoder (VAE) backbone from scCRAFT' with four loss components (reconstruction, prior, triplet, adversarial) -> new: stock scvi-tools 1.4.2 SCVI (ZINB, nonlinear decoder) or LinearSCVI (NB) with reconstruction + KL + adversarial term only; no triplet, no cosine term.
3. Input / target (l.152, l.262) 'takes log1p-transformed gene expression ... aims to reconstruct the original count values'; 'raw count matrices ... retained as the reconstruction target' -> false for the submitted runs (D1). New: scVI encodes log1p(raw counts) (log_variational=True, no library-size normalisation) and reconstructs integer counts; prep now takes the source counts layer and refuses non-integer input (5709a98 data.py).
4. Encoder (l.152) |x|->1024->512->(mu, sigma^2) of size 256 with BN+ReLU -> scVI default: 1 hidden layer of 128 units (n_hidden=128, n_layers=1), dropout 0.1, batch norm, n_latent=10 (verified in scvi 1.4.2 model/_scvi.py, module/_vae.py).
5. Decoder (l.154-158) G1(z,v)+G2(v), exponential mean with no library size, cell x gene dispersion r -> scVI DecoderSCVI with covariates injected in every layer when conditioned (deeply_inject_covariates=True), softmax scale x library size, gene-specific dispersion (dispersion='gene'), ZINB adds dropout logits; the unconditioned arm omits batch_key at setup and passes batch to the adversary through the labels slot. The original decoder was batch-conditioned, so the conditioned arm is the closest analogue.
6. Eq. 1 (l.161-162) L_NB = -(1/G) sum_g log P_NB(x|mu,r): the summand is not indexed by g (notation error); code averages over cells AND genes (reconst_loss.mean()); scVI sums over genes and averages over cells. With beta_KL=0.005 on a latent-summed KL and NB averaged over 2,000 HVGs, the submitted objective's effective KL weight in summed-NB units was about 10, lambda_cos=20 was about 4e4, and the lambda_adv grid 0.01-0.5 was about 20-1000, so no lambda value transfers between the original and new backbones.
7. Eq. 2 cosine loss and lambda_cos=20 (l.166-170, l.190) -> removed. As typeset in the zip, Eq. 2 already includes the L2 norms; the reviewer's quoted form <x_g, log(1+mu_g)> does not match this file.
8. Eq. 3 triplet loss on Leiden pseudo-labels (l.172-178; resolutions 1.0/7.0, margin 5) -> removed. Undisclosed details: create_triplets draws anchor, positive and negative within the same batch; Leiden was run on all cells in prep before the CV split, contradicting 'to prevent data leakage'; the cap is 15 pairs per Leiden cluster per batch, not 'per cell-type'.
9. KL weight (l.190) beta_KL=0.005 'identical to the original implementation' -> new: kl_weight annealed linearly from 0 to max_kl_weight=1.0 over n_epochs_kl_warmup=400 epochs (scvi 1.4.2 TrainingPlan defaults); because the stock epoch heuristic gives <400 epochs for every dataset above 20k cells, kl_weight never reaches 1 (immune, 239 epochs, ends near 0.6).
10. Discriminator (l.193-200) 'h_upsilon: Z -> V' (codomain should be R^|V| logits; upsilon read as v by Reviewer 1); 256->128->128->|V| MLP; CE 'effectively approximating the Jensen-Shannon divergence' -> the V-way CE estimates a multi-distribution JS weighted by the sampling proportions: equal weights under the original class-balanced sampler, batch-size weights under scVI's stock sampler. New head: same 128-128 MLP on n_latent=10 inputs (not scvi-tools' Classifier with n_hidden=32, n_layers=2); state this explicitly.
11. Critic (l.203-215): architecture never stated (code: the same MLP as the discriminator, no normalisation layers); sign convention W_i = E_ref[C_i] - E_src[C_i] in text vs E_src - E_ref in code (equivalent up to the sign of C; harmonise); 'evaluated on samples interpolated strictly between the source batch and the reference batch' is false for the binary and reference-sweep runs (D4); 'stratified sampling ... forces every mini-batch to contain samples from the reference' does not hold under scVI's stock sampler (with a small reference batch at batch size 128 the W estimate rests on few reference cells; the loss returns 0 when none are present). In 5709a98 the reference GP is correct and the old fallback survives as the pooled formulation's GP.
12. Generator objective (l.217-221) 'Because the generator has no influence over the reference batch distribution P_r ... simplifies to L_adv = -E_{P_i}[C(z)_i]' -> false for a shared encoder (P_r is the encoder's output on reference cells), and both cd64f05 (all_loss = -d_coef*loss_da with z not detached) and 5709a98 (gen_loss = loss_vae - d_coef*loss_da_g) back-propagate the full W estimate including the reference term, so reference cells move too. Rewrite Eq. (adv).
13. Update ratio (l.224) 'We use a 10:1 (critic:encoder) update ratio' -> WGAN and WGAN-GP default n_critic=5 (verified, arXiv 1701.07875 Alg. 1; 1704.00028 Alg. 1); the discriminator's 1 step (D5) was not reported; both old and new code reuse one minibatch's detached z for all inner steps (audit L6), unlike WGAN-GP Algorithm 1, which samples fresh real/fake batches each critic iteration. New plan: discriminator at 1 and 10 steps, critics 10; disclose the minibatch reuse.
14. Discriminator generator loss (l.229-232) L_VAE - lambda_adv*L_CE, 'minimizing the negative cross-entropy to maximize classification error' (D6) -> new: scvi-tools' bounded fool loss, cross-entropy to the uniform distribution over the OTHER batches (scvi 1.4.2 loss_adversarial_classifier(predict_true_class=False); 5709a98 _fool_loss).
15. Adversarial weight schedule (l.234-242) -> new plan uses a FIXED d_coef from step 0 (no adversary warm-up, no kappa annealing), whereas stock scvi-tools scales the adversary by kappa = 1 - kl_weight; with the 400-epoch KL warm-up the adversary-to-KL balance therefore drifts during training. State it and justify the fixed lambda.
16. Datasets and batch keys (l.240; Table 1) Pancreas tech (9), Lung protocol (2), Immune chemistry (4) -> the scIB files also hold finer keys (checked: immune 'batch' 10 categories, 'study' 5; lung 'batch'/'donor' 16, 'dataset' 3); 'chemistry' merges studies (by cell counts, v2_10X = Freytag + Oetjen A/P/U and '10X' = Sun samples 1-4) and treats within-chemistry donor/study effects as biology. The new registry keeps chemistry/protocol; justify (technology-level confounder) and consider the scIB-standard keys for comparability. Under the real-counts rule the original lung and pancreas cannot be used as committed (see R2.4 blocker), and lung (only 2 protocols) could enter the scaling series via its 16 donors only if counts were recovered.
17. Hardware / optimiser / batch size (l.240) RTX 4090, Adam lr 1e-3 betas (0.5, 0.9) for VAE and adversary, batch 1024 -> new: scVI generator optimiser (lr 1e-3, eps 0.01, weight decay 1e-6, betas 0.9/0.999); adversary Adam lr 1e-3 betas (0.5, 0.9) (differs from scvi's own classifier optimiser Adam lr 1e-3, eps 0.01, wd 1e-6, and from WGAN-GP's recommended lr 1e-4, beta1 0, beta2 0.9); batch 128; local RTX 3080 / cluster GPUs.
18. Runtime (l.240, Supp. Note 1) '~24% more training time per epoch' and 'faster convergence' -> backbone-specific; current measurement 3.5-7x per epoch (critic ~7-9 s vs discriminator ~1-2 s, atac_small, batch 128); convergence never measured.
19. Warm-up (l.242) 'first 10 epochs where lambda_adv=0 and the adversary's weights were frozen' -> true for inner-CV models only; reported outer models used the default 5 (D7). New: no adversary warm-up.
20. Sampling (l.242) 'class-balanced sampling with replacement across all experiments' -> new: scVI's stock shuffled sampler (proportional to batch size). This changes which divergence each adversary estimates and the number of reference cells per minibatch; report it, and decide whether balanced sampling is part of 'scVI as people use it' (it is not by default).
21. Model selection (l.242) nested CV (5 outer x 3 inner; inner 40 epochs, outer 80), selection by '(iLISI - 10*cLISI)' -> new: no CV; 5 seeds on all cells; selection-free lambda curves plus a PRE-REGISTERED lambda; scIB-native 0.4/0.6 overall. Note the scIB overall is min-max scaled within each dataset across the runs included, so the scaling set must be fixed in advance and raw metrics reported (pilot overall values are relative to the pilot's own run set).
22. Experiments (l.248-250): binary = two largest batches (reference = alphabetical index 0, which coincides with the largest batch for all three datasets); multibatch adds batches in descending size with the largest batch as reference (true in code); reference sweep over alphabetical indices with the D4 fallback -> new: entropy-rule reference (select_reference_batch), and the plan currently contains no binary, batch-count or reference-sweep experiment (original sections 1, 2 and 4).
23. Balanced scenario (l.255) 'restricted the datasets to the intersection of shared cell types ... batch and cell-type distributions are independent' -> false: Cramer's V between batch and cell type in the balanced binary sets is 0.33 (pancreas), 0.42 (lung), 0.33 (immune) (computed from Supplementary Table S2); pancreas balanced differs from unbalanced by 2 cells; in the multibatch series the intersection shrinks to 4 types (immune V=4: monocytes/DCs only; pancreas V>=6: alpha/beta/delta/gamma). Use composition-matched subsampling if independence is the intended manipulation.
24. Statistics (l.259) corrected paired t-test (Nadeau & Bengio; notebook uses correction 1/n + 0.25 with 5 folds) -> 26/36 Table 2 p-values <0.05 uncorrected, 6/36 after Holm, 25/36 after BH (recomputed from the reported p-values). New design has independent seeds and 8 datasets: pre-specify paired within-dataset seed comparisons and an across-dataset test with multiplicity control.
25. Preprocessing / Table 1 (l.262-288): QC 300 genes / 5 cells, 2,000 Seurat HVGs (code: sc.pp.highly_variable_genes with batch_key, on the double-normalised data under D1); Table 1 lung Drop-Seq 6,485 vs 9,701 cells in the source (post-QC) -> regenerate Table 1/S2 from the counts-based prep for all 8 datasets; composition changes if non-count batches are excluded.
26. Evaluation space (l.292) '256-dimensional latent embeddings were reduced to 50 principal components' -> D2/D3; new: n_latent=10 posterior means (get_latent_representation give_mean=True), no PCA.
27. LISI (l.294-301) custom Euclidean-kNN LISI, mean over cells, cLISI scaled (x-1)/(C-1) with 'lower values indicate superior conservation', 'LISI ... primary evaluation tool' -> scib 1.1.7 graph LISI uses the median and scales cLISI as (C - cLISI)/(C - 1), higher = better (verified lisi.py); the plan uses scib.metrics.metrics with an overall 0.4 batch / 0.6 bio score, so drop the 'LISI primary' framing and flip the cLISI direction in all text/figures.
28. Global metrics (l.303-308): ASW-batch is computed within cell types (scib silhouette_batch with group_key; unstated); 'PAGA Topology Preservation' is a custom metric outside scIB (justified loosely by nonlinear compression); ARI uses NMI-optimised clustering (consistent with scIB) -> new scIB-native set: PCR comparison, ASW-batch, graph iLISI, graph connectivity, kBET; NMI, ARI, ASW-label, isolated-label F1/ASW, graph cLISI, cell-cycle conservation (where genes allow); trajectory disabled.
29. Generalisation (l.311) 'compute all reported metrics on held-out test data ... unseen query cells' -> test cells came from the same batches (StratifiedKFold on cell labels) and D2 applies; new scoring is transductive on all cells. Drop this framing or redesign as a genuine query-to-reference mapping experiment (not requested by reviewers).
30. Figure 1 (l.145-150) caption 'we employ a fixed VAE backbone' -> redraw for scVI encoder/decoder with the optional batch-conditioning input to the decoder and the swappable adversary head (discriminator / reference / pooled / barycenter critic).
31. Availability (l.106): the data link covers only the scIB immune and pancreas figshare item; lung, simulations, ATAC and cross-species files need sources, and the code link should pin the commit that regenerates the results.
32. NEW-IMPLEMENTATION DISCLOSURES (5709a98) that the rewritten Methods must state because they are non-standard relative to stock scvi-tools: authored 128-128 adversary head instead of scvi's Classifier; fixed lambda instead of kappa = 1 - kl_weight; adversary optimiser betas (0.5, 0.9); critic inner loop reuses one minibatch; optional z-standardisation before the adversary; the unconditioned arm passes batch through the labels slot. The fit driver's defaults (scvi_adv_fit.py: NLAT=30, BATCH=512, MAXEP=239, SCVI_MODEL=LinearSCVI; fit_adversarial_linearscvi defaults n_latent=30, max_epochs=239, batch_size=512) differ from the stock-default decision (10 / 128 / heuristic), so every manifest row must set and record them explicitly.
33. The E0 notation memo (v12c27680) and the July reviewer response cannot be used as the Methods source: A.1 says the NB is 'summed over genes' and uses gene-specific theta_g, but cd64f05 averages over cells x genes and px_r is cell x gene; A.2 says the prior cosine form 'omitted the normalisation', but submitted Eq. 2 has it; both describe the scCRAFT implementation that the new paper no longer uses.

## 4. Reusable material

1. Research question and narrative backbone: Wasserstein critic vs JS discriminator in adversarial batch integration, with the four Results sections (1 local integration vs biological conservation, 2 batch-count scalability, 3 lambda sensitivity, 4 reference-batch selection). Reuse them as pre-registered hypotheses with the confirm/refute criteria in the claims table, so the narrative survives either outcome.
2. Introduction paras 1-3 (l.128-132): batch-effect motivation and the JS-vs-W argument, after edits: cite Arjovsky & Bottou 2017 (arXiv 1701.04862) alongside WGAN; 'continuous everywhere and differentiable almost everywhere' (WGAN Thm 1); acknowledge non-saturating / bounded generator losses; drop 'leading to failed training' unless shown.
3. Related Work (l.137-139): the statistical / geometric / deep-generative taxonomy and its citations (limma, ComBat, MNN, Harmony, Seurat, scVI, AD-AE and ResPAN verified against abstracts, IMAAE verified against full text with the rephrasing noted in the claims table). Replace 'controlled, state-of-the-art autoencoder framework' (scCRAFT) with stock scVI; add scvi-tools, LDVAE/LinearSCVI, scANVI, Scanorama, MMD-ResNet (10.1093/bioinformatics/btx196, already in reference.bib but uncited), scDREAMER, and MMD/Sinkhorn references if those arms are kept.
4. Methods text for the multi-headed reference critic (l.203-215) and discriminator baseline (l.192-200), with the corrections in method_text_conflicts (generator objective incl. reference term; GP description; architecture; bounded fool loss), extended to the pooled and barycenter targets.
5. Experimental design paragraphs (l.244-257): binary -> multibatch (descending size) -> reference sweep, crossed with balanced/unbalanced; keep the structure, redefine 'balanced' as composition-matched, and run it on the stock-scVI arms.
6. Figure designs (all numbers must be regenerated): Fig 2 grid (datasets x metrics x scenario) with scIB-native metrics; Fig 3 (metrics vs number of batches) with ceiling-normalised iLISI; Fig 4 (metric vs lambda) becomes the primary lambda-curve figure with 5-seed bands; Fig 5 (per-reference box plots) annotated with reference coverage; Supp. Fig S1 timing bars; Supp. UMAP layouts; arch.png concept (encoder -> z -> adversary -> confounder) redrawn for scVI with the conditioning arrow.
7. Table 1 / Supplementary Table S2 structure (cells and cell types per batch), regenerated from the counts-based prep for all 8 datasets.
8. Discussion structure (four headed conclusions + limitations) and the barycenter paragraph (l.428), which becomes a tested hypothesis now that the barycenter arm exists (fix 'geometric median'); the limitation about a single categorical confounder still applies.
9. Bibliography: all 23 cited keys exist in reference.bib (51 entries). Keep them, remove duplicate keys (Tran2020-ar / Tran2020-op; Rousseeuw1987-mw / ROUSSEEUW198753), add Arjovsky & Bottou 2017 and the scvi-tools / LinearSCVI / baseline citations; cite He2025-ip (scCRAFT) only where the backbone change from the earlier preprint is explained.
10. Code carried forward: authored adversary heads (src/wcd_vae/wcd/adversarial.py, critic.py), scripts/support_overlap.py (R1.major.2), scripts/run_biology.py (R1.minor.2/R2.6), scripts/build_metric_curves.py (per-metric curves), the dataset registry and the scIB-native scorer (score_scib_native.py).
11. Recovered primary source: the verbatim decision letter with all 17 comments (19-Jul-2026), extracted from the earlier session because the pasted attachment is no longer in the artifact store; saved alongside this audit.
12. NOT reusable: every number in Table 2, Figs 2-5 and Table S1; the E0 memo's equations; all E1-E8 numbers and conclusions in the July reviewer response (scCRAFT backbone and/or the C2 counts bug).

## 5. Framing risks

1. 'Topology' vocabulary (title 'Topology Matters', 'topological bottleneck', 'geometric bottleneck', 'uncoordinated radial projection topology', 'topologically dense reference', 'topological collapse'): nothing topological was measured (the PAGA Spearman was a custom metric, significant in 1/6 cells). R2.3 called the mechanistic language unsupported and R1.minor.1 found the bottleneck intuition unclear. Use 'single-reference design', 'reference coverage', 'batch-count scaling'; consider a title that states the trade-off question rather than 'Topology Matters'.
2. Class-level generalisation ('Wasserstein critics vs discriminators' as method classes, 'fundamental trade-off'): R2.1/R3.1. Scope every claim to adversarial regularisers added to scVI-family VAEs, the formulations tested, and the 8 datasets.
3. Equating 'mixing' or 'removing batch effects' with iLISI (Discussion l.424 'outperformed the discriminator in removing batch effects (higher iLISI)'): ASW-batch pointed the other way (R1.major.1). Report batch metrics separately and the scIB aggregate.
4. 'Over-correction' / 'collapsed cellular representations' without direct biological readouts (R2.6) and without matched-integration comparisons (R2.2). In the pilot every arm collapses biology at high lambda, so over-correction must be shown at matched batch score to be objective-specific.
5. Attributing reference sensitivity and scaling failures to the Wasserstein objective (R2.5; l.411 'Because the Wasserstein objective enforces a strict optimal transport plan'). Only the pooled/barycenter comparison can separate objective from reference design, and the original balanced-scenario evidence already contradicts the coverage explanation.
6. 'Weak integration ... acts as a protective measure': weaker integration conserving more biology is the trade-off curve, not protection; confounded by 10 vs 1 adversary updates and an unbounded discriminator loss.
7. Stating the vanishing-gradient premise as established for practical discriminators ('leading to failed training'): it was never shown, the original discriminator trained fine, and R1.major.2 raised disjointness as a FAIRNESS problem for the discriminator. The earlier response's 'premise holds, so the comparison is fair' inverts the reviewer's point.
8. Matched-nominal-lambda comparisons ('the critic performing consistently higher' at every lambda): CE and W-estimate scales differ and the original lambda grid corresponds to about 20-1000 in summed-NB units. Compare at matched achieved batch score or via selection-free curve dominance with a pre-registered lambda.
9. Statistical wording ('significantly', 'consistent'): Table 2 has 36 uncorrected tests (6/36 survive Holm), five overlapping CV folds and a single seed. Pre-register tests, use seeds as replicates, control multiplicity, and avoid 'significant' for untested claims such as 'degrades significantly faster'.
10. Public record: the submitted abstract is public as bioRxiv 10.1101/2025.11.12.688061 (v3), v1 was titled 'Wasserstein Critics Outperform Discriminators in Adversarial Deconfounding of Gene Expression Data', and the GitHub README still says the work 'uses scCRAFT ... as the backbone' and that discriminators avoid 'the geometric bottleneck of a single fixed reference'. The revision must explicitly supersede those claims, post a new preprint version, update the README, and disclose that the submitted pipeline fit the NB to log-normalised values (D1).
11. Recycling the July reviewer response: its E1-E8 numbers come from invalidated pipelines, and several statements are wrong on inspection (R3.2 'requires >=50 embedding dimensions' is contradicted by scib's n_comps cap; R1.minor.3 corrections misdescribe the code; R1.major.2 framing). Write the new response from new results only.
12. Design mislabels: 'Balanced' does not mean independence of batch and cell type, and 'Unbalanced ... known to drive severe over-correction' is uncited. Rename or redesign (composition-matched vs natural composition).
13. Batch-count 'decay' presented as task difficulty when part of it is the scaled-iLISI ceiling (immune 0.965 -> 0.703 at V=4) and composition change (balanced immune V=4 = 4 myeloid/DC types). Normalise, or say what the metric can and cannot show.
14. Held-out evaluation sold as 'generalization to unseen data': the test cells came from seen batches and their PCA frame was misaligned (D2). Do not reuse this claim.
15. Runtime and convergence claims (~24% overhead, 'faster convergence') do not transfer to stock scVI (now 3.5-7x per epoch) and convergence was never measured.
16. Barycenter oversell ('the theoretical solution', 'geometric median'): the implemented arm is a W1 critic against 64 learned anchors, not the Agueh-Carlier barycenter (a mean), and it shrinks the latent at high lambda (audit M2) unless z is standardised.
17. Pilot numbers quoted as findings: the atac_small pilot is one dataset, 3 seeds, pre-stock schedule, old discriminator loss, alphabetical reference, and its scIB overall is relative to its own run set. Use it only to set the lambda grid and pre-registration.
18. Guidance that outruns results (R3.3): if no adversary beats stock scVI under equal training budgets (the pilot hint), the guidance must say so rather than recommend either adversary; avoid promising mitigation strategies that were not tested.
19. Batch-key comparability: chemistry/protocol keys differ from scIB's standard keys (immune 10 batches, lung 16), so scores are not comparable to published scIB tables unless that is stated or the standard keys are added.
