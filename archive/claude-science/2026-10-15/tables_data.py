# Tables for the Introduction review. Each CLAIMS row: (id, line, claim, source, what was read, verdict).
# Verdict 'OK' = statement supported as worded; 'OK + INT-nn' = supported, wording or tag fix in finding INT-nn; 'FINDING INT-nn' = not supported as worded.

CLAIMS = [
# --- literature, Batch effects and Adversarial integration
("CL-01", ":118", "Batch effects are technical variation (laboratory conditions, reagent lots, personnel, instruments, time, pipeline); they mislead when correlated with the outcome", "Leek 2010; Yu 2024", "Leek 2010 (Nat Rev Genet 11:733, PMC text): abstract (laboratory conditions, reagent lots, personnel; correlated with an outcome); Yu 2024 (Genome Biol 25:254): abstract", "OK"),
("CL-02", ":119", "Separately processed replicates differ substantially in scRNA-seq", "Tung 2017", "Tung et al. 2017 (Sci Rep 7:39921) full text: results on separately processed replicates (Fluidigm C1 plates)", "OK"),
("CL-03", ":119", "Batch effects remain a single-cell problem", "Goh 2022", "Abstract only (full text closed)", "OK + INT-19"),
("CL-04", ":120", "Residual effects mislead; over-correction removes biology", "Yu 2024", "Abstract: 'misleading outcomes if uncorrected, or hinder biomedical discovery if over-corrected'", "OK + INT-18"),
("CL-05", ":121", "Integration is scored on two axes, batch removal and conservation of biological variation", "Luecken 2022 (scIB)", "Nat Methods 19:41 full text: abstract and metric overview", "OK"),
("CL-06", ":126", "An encoder is trained against an adversary that predicts the batch from the latent code", "Ganin 2016", "JMLR 17(59) full text: domain classifier trained against the feature extractor; author list on page 1", "OK; metadata INT-16"),
("CL-07", ":126", "... or another confounder such as age", "Dincer 2020 (AD-AE)", "bioRxiv full text and published abstract: adversarial deconfounding of the latent code; age named as a confounder", "OK"),
("CL-08", ":127", "Optimal discriminator gives -log 4 + 2 JSD", "Goodfellow 2014 [Thm. 1]", "NeurIPS 27 full text Sec. 4.1: Prop. 1, Thm. 1, Eq. 5-6", "OK + INT-08"),
("CL-09", ":128", "A W1 critic estimates the earth mover's distance through its 1-Lipschitz dual", "Arjovsky 2017 (WGAN)", "PMLR v70 full text Sec. 3 (Kantorovich-Rubinstein duality)", "OK"),
("CL-10", ":128", "WGAN-GP enforces the Lipschitz constraint softly", "Gulrajani 2017 [Sec. 4]", "NeurIPS 30 full text Sec. 4 (gradient penalty on interpolates)", "OK"),
# --- JS versus W1
("CL-11", ":133", "W1 is continuous / differentiable a.e. under the stated conditions; JS and KL are not", "WGAN [Thm. 1]", "PMLR v70 Thm. 1 (statements 1-3), Example 1", "OK + INT-18"),
("CL-12", ":134", "Disjoint parallel lines: JS = log 2 for every nonzero offset, W1 = the offset", "WGAN [Example 1]", "PMLR v70 Example 1 (W = |theta|, JS = log 2 for theta not 0)", "OK"),
("CL-13", ":135", "Disjoint supports or non-aligned manifolds: a perfect discriminator with zero gradient exists", "Arjovsky and Bottou 2017 [Thms. 2.1-2.2]", "arXiv 1701.04862v1 Sec. 2.1 (Thms. 2.1, 2.2)", "OK + INT-18"),
("CL-14", ":136", "Saturating loss: vanishing generator gradients near the optimal discriminator", "Arjovsky and Bottou 2017 [Thm. 2.4]", "arXiv v1 Thm. 2.4, Cor. 2.1", "OK + INT-06"),
("CL-15", ":137", "Non-saturating loss need not vanish; with a noisy discriminator its gradient is Cauchy-distributed", "Arjovsky and Bottou 2017 [Thms. 2.5-2.6]", "arXiv v1 Sec. 2.2.2, Thms. 2.5, 2.6 and the remark after the proof", "FINDING INT-07"),
("CL-16", ":138", "Continuous noise on the discriminator input removes the disjoint-support assumption", "Arjovsky and Bottou 2017 [Thm. 3.1]", "arXiv v1 Sec. 3, Thm. 3.1, Cor. 3.1", "OK + INT-18"),
# --- scvi-tools, code
("CL-17", ":139", "In X1 the stock scvi-tools adversary sees posterior samples", "scvi-tools 1.4.2; plan.py:348-351; manifest", "scvi/train/_trainingplans.py:689 (z = inference_outputs['z']); scvi/nn/_base_components.py:304-305 (rsample); scvi/module/_vae.py:390, :413; plan.py:348-351 (arms none and scvi_adv take the stock step); manifest scvi_adv rows ({{scvi_adv_rows}}), cond = 1", "OK + INT-17"),
("CL-18", ":140", "Authored arms see the posterior mean by default", "SI-19; plan.py:359", "plan.py:359 (z_used = mu if adv_input == 'mean' else sample); SI-19 text; manifest authored X1 rows hold the placeholders A2 (adv_input) and A3 (standardisation)", "OK"),
("CL-19", ":140", "The frozen rule-R4 record flags immune (conditioned), so R2 and R3 cannot switch; refitting Sinkhorn cannot remove the flag", "SI-66; r4_a1_record.json", "docs/prereg/r4_a1_record.json (frozen, git {{r4_git}}, {{r4_extensions}} extensions): immune|1 b_common - b0 = {{immune1_diff:.4f}}, low_signal true; setters barycenter in 5 blocks, mmd in sim1|1, sinkhorn in none; SI-66 text", "OK + INT-15 (two blocks flagged)"),
("CL-20", ":141", "The discriminator arm minimises scvi-tools' fool loss (cross-entropy to uniform over the other batches, over all cells) at one adversary step", "SI-01; plan.py:8-9, 292-298", "plan.py:8-9 ('ONE adversary step per generator step'), :292-298 (_fool_loss: other = one-hot complement / (k - 1); mean over cells); scvi-tools _trainingplans.py:651-667 (fool loss: cross-entropy to the uniform distribution over the other classes); SI-01", "OK"),
("CL-21", ":141", "For two batches this is the label-flipped non-saturating loss", "(none)", "Algebra from plan.py:297-298 with k = 2", "OK + INT-12 (add tag)"),
("CL-22", ":142", "Thms. 2.4-2.6 assume a fixed data distribution; every batch's latent moves here", "(none)", "arXiv v1 Thms. 2.4-2.6 (P_r the real data distribution); WGAN Thm. 1 (P_r fixed)", "OK + INT-06"),
("CL-23", ":143", "The vanishing-gradient premise is examined, not assumed: R0 is exploratory", "SI-68; PAPER_PLAN section 3; decision letter, Reviewer 1, major comment 2", "SI-68 (exploratory: no confirmed or refuted claim); decision letter Reviewer 1 'Major comments' second bullet (disjoint-support / fairness)", "OK + INT-15 (reviewer remark in plain text)"),
# --- composition shift
("CL-24", ":148", "Aligning latent distributions forces error: eps_S + eps_T >= 1/2 (d_JS(Y) - d_JS(Z))^2", "Zhao 2019 [Thm. 4.3; binary labels, JS distance]", "PMLR v97 Sec. 4.3: Lemma 4.8, Thm. 4.3, Remark", "OK + INT-05"),
("CL-25", ":149", "Exact alignment with zero source error leaves target error of at least the label-proportion gap", "Wu 2019 [Prop. 3.1; binary labels]", "PMLR v97 Sec. 2-3, Prop. 3.1 and the text after it", "OK"),
("CL-26", ":150", "Over-alignment under composition shift is predicted for every divergence arm", "(none)", "Zhao Thm. 4.3 (JS only), Wu Prop. 3.1 (exact alignment only)", "FINDING INT-04"),
("CL-27", ":151", "An adversarial cVAE and GLUE mixed unrelated cell types with unbalanced proportions, more so at higher correction strength", "Hrovatin 2025 (sysVI)", "BMC Genomics 26:974 full text: Introduction and Results on adversarial cVAE and GLUE under unbalanced proportions", "OK + INT-15 (spell out GLUE)"),
("CL-28", ":152", "scCRAFT added a triplet loss against the over-correction of its adversarial loss under heterogeneous compositions", "He 2025 (scCRAFT)", "Commun Biol 8:561 full text: abstract, Introduction, Methods (dual-resolution triplet loss)", "OK + INT-14 (pending marker)"),
("CL-29", ":153", "Dataset imbalance changed clustering, annotation, DE, mapping, trajectories for five methods", "Maan 2024", "Abstract via Europe PMC (published); 2022 preprint text as substitute", "OK + INT-19"),
# --- GAN
("CL-30", ":158", "Image GANs: most losses reach similar scores with enough tuning and restarts; none consistently beats the non-saturating GAN", "Lucic 2018", "NeurIPS 31 full text: abstract, Sec. 1 and 3", "OK"),
("CL-31", ":159", "Non-saturating loss stable; spectral normalisation consistently helped; gradient penalty only sometimes", "Kurach 2019", "PMLR v97 full text: abstract, Sec. 3-4, Fig. 2", "OK"),
("CL-32", ":160", "A critic-discriminator difference may reflect regularisation, update budget or tuning (image results)", "(none)", "Inference from CL-30, CL-31", "OK + INT-12 (add marker)"),
# --- Gap
("CL-33", ":165", "scIB-E: six batch losses in one scVI framework; no W1, MMD or OT term", "Yi 2025 (scIB-E)", "Genome Biol 26:398 full text: Methods 'Level-1' list; 0 hits for Wasserstein, MMD, maximum mean, optimal transport, Sinkhorn, earth mover", "OK + INT-20"),
("CL-34", ":166", "No budget-matched comparison of JS, W1 and MMD/OT objectives on a stock scVI backbone was found", "PAPER_PLAN section 2; evidence.md", "Own gap search (section 7); evidence.md:62-67, :75", "FINDING INT-01, INT-02, INT-03"),
("CL-35", ":167", "Reference anchoring exists for W1 critics (iMAP) and MMD (SAUCIE, MMD-ResNet)", "PAPER_PLAN section 2", "iMAP full text (Genome Biol 22:63, Methods: WGAN-GP; anchor dataset); SAUCIE bioRxiv Sec. 4.1.3 (MMD to a reference batch); MMD-ResNet arXiv 1610.04181 Introduction", "OK + INT-02 (status, citations)"),
# --- Contribution
("CL-36", ":172", "One stock scVI backbone at scvi-tools defaults; equal epochs, minibatch and latent size per experiment and task for every arm, plain scVI and sysVI, with declared exceptions", "SI-10; SI-04; SI-02", "Manifest of record: X1 epochs, minibatch ({{x1_batch}}) and latent size ({{x1_latent}}) equal across arms per task; X13 sysVI and scANVI rows match X1 per task ({{x13_match}}); scANVI's extra training epochs after pre-training (fit_paper_config.py:401-402) are declared in Methods", "OK + INT-13 (decoders), INT-24"),
("CL-37", ":173", "The decomposition as the paper's stated contribution is not locked", "pending", "CONSTRAINTS.md: no entry adopts the decomposition; PAPER_PLAN section 2", "OK (correctly pending)"),
("CL-38", ":175", "P1: biology kept at equal batch removal", "SI-68", "SI-68 text; ANALYSIS_SPEC section 0", "OK"),
("CL-39", ":176", "X3 dose-response", "pending (analysis)", "SI-25, SI-31..33, SI-41; manifest", "OK + INT-21 (design untagged)"),
("CL-40", ":177", "W1 critics against MMD and Sinkhorn arms", "SI-18", "SI-18 (arms and grids); SI-68 (single-strength tables and good-strength ranges cover every arm)", "FINDING INT-10"),
("CL-41", ":177", "Standardisation compared in pilot A3 under rule R3; off in X1", "SI-20; SI-27; SI-66; r4_a1_record.json", "SI-20, SI-27 text; record as CL-19", "OK"),
("CL-42", ":178", "X6: discriminator plain, spectrally normalised or R1; pooled critic at 1, 5, 10 steps; spectrally normalised at 1 and 5", "SI-22; manifest", "Manifest X6 rows: discriminator[1], discriminator_sn[1], discriminator_r1[1], pooled[1], pooled[5], pooled[10], pooled_sn[1], pooled_sn[5]: {{x6_rows_each}} rows each; SI-22 text", "OK + INT-21 (citations)"),
("CL-43", ":179", "X7 redesign pending; NO-GO as designed (D-06, D-07)", "pending", "paper.tex:63; ANALYSIS_SPEC section 0; REVIEW_design_recheck.md:39-40", "FINDING INT-09"),
("CL-44", ":181", "Guidance for scVI users follows from the results; comparison with standard tools at equal numbers of tries", "SI-68, SI-77; pending outcome", "SI-68, SI-77 text", "FINDING INT-11"),
]

CONFLICTS = [
# (n, short conflict, outline resolution, my check, verdict)
(1, "Zhao Thm 4.3 / Wu: 'any proper divergence forces errors, tighter forces more'", "JS-distance form (Zhao) plus exact-alignment form (Wu), binary labels", "Zhao PMLR v97 Thm. 4.3 and Wu Prop. 3.1 read as text", "Resolution right; :150 re-inflates it (INT-04)"),
(2, "scIB-E compares GAN, HSIC, orthogonality, MI losses", "Lists all six level-1 losses; no W1, MMD, OT", "scIB-E full text: six level-1 losses; 0 hits for Wasserstein, MMD, OT, Sinkhorn", "Right; HSIC wording INT-20"),
(3, "'Loss is second-order to tuning and regularisation' (Lucic, Kurach)", "States what each paper found, image GANs only", "Lucic abstract; Kurach abstract and Sec. 3", "Right; the inference needs a marker (INT-12)"),
(4, "'Premise becomes a tested hypothesis' (PAPER_PLAN section 3)", "Premise 'examined, not assumed'; R0 exploratory", "SI-68 text", "Right"),
(5, "Gap 'trained on real counts' (PAPER_PLAN section 2)", "'Real counts' dropped (SI-15); search-based absence kept", "SI-15; evidence.md:75; own gap search", "Right; scope and status of the gap line INT-02"),
(6, "scVI adversary trains on posterior samples so noise removes disjoint support", "True for scvi_adv only; authored arms use the posterior mean by default", "scvi-tools 1.4.2 lines (CL-17); plan.py:359; manifest", "Right; tags INT-17"),
(7, "'Theory covers the saturating loss; fool loss not covered automatically'", "Fool loss = label-flipped non-saturating loss for two batches; theorems assume a fixed distribution", "plan.py:292-298; arXiv v1 Thms 2.4-2.6", "Right; caveat incomplete (INT-06)"),
(8, "IMAAE 'Wasserstein-based' (July :139)", "Dropped; loss has WGAN-GP form", "IMAAE PMC text: 0 hits for Wasserstein and WGAN; gradient penalty on interpolates, generator loss -E[D]", "Right (drop list :201-202)"),
(9, "Batch effects 'arise from the sequencing technology' (Tung 2017)", "Cited only for replicate-level batch effects", "Tung 2017 full text: separately processed replicates, one platform", "Right (:119)"),
(10, "Three-way taxonomy cited to Luecken 2022 (July :137)", "Not cited to scIB", "scIB groups outputs as graph, embedding, corrected matrix", "Right (Related work track)"),
(11, "'Sample age' cited to Stegle 2015 (July :128)", "Dropped; AD-AE cited for age", "Stegle 2015 not retrieved (also by me); AD-AE abstract names age", "Right"),
(12, "scCRAFT 'only as the earlier backbone' versus 'no mention anywhere'", "One backbone line pending; scCRAFT also cited as related work", "PAPER_PLAN line 65; evidence.md:82; paper.tex:152, :305, :306", "Resolution incomplete (INT-14)"),
(13, "Barycenter 'first' claim (August revision)", "Not carried; listed in the drop list", "manuscript_tm/oup-authoring-template.tex:140 at 08f1e3c; drop list :214-215", "Right"),
(14, "Seed-list key corani2017time", "Entry kept under the seed key", "evidence.md:84 (Methods and Related-work tracks)", "Outside the Introduction; not checked here"),
]

DROPLIST = [
# (lines, item, audit status read from CSV, my check, verdict)
(":187-188", "JS fails on disjoint technology batches (C01)", "partially_supported", "SI-68: R0 exploratory", "OK"),
(":189-190", "First systematic comparison (C02)", "untested", "scIB-E six losses in scVI; WDGRL, iMAP, SAUCIE", "OK; wording INT-23(a); prior preprint INT-03"),
(":191-192", "Critics over-correct as a W1 property (C06)", "invalidated_by_bugs", "Zhao, Wu, sysVI", "Reason overreaches (INT-04)"),
(":193", "Weak integration is protective (C09)", "invalidated_by_bugs", "audit basis", "OK"),
(":194-195", "Topology language (C12, C14)", "untested; contradicted", "decision letter, Reviewer 2 main concern 3; PAPER_PLAN title note", "OK; attribution INT-23(b)"),
(":196-197", "W1 gives robust gradients regardless of initial separation (C23)", "partially_supported", "WGAN Thm. 1; Gulrajani Sec. 4", "OK"),
(":198-200", "Superior local mixing, scales poorly, broad windows, 24% more time, ideal for disjoint, barycenter 'theoretical solution', primary limitation (C04, C05, C10, C11, C16-C18, C20-C22)", "invalidated_by_bugs / contradicted / untested as listed", "claims audit CSV statuses; July :428", "OK"),
(":201-202", "IMAAE 'Wasserstein-based' and 'motivating' a global Gaussian prior (C24)", "partially_supported", "IMAAE PMC text", "OK"),
(":203-204", "'Unbalanced scenario known to drive severe over-correction' (C25); 'balanced' not composition-independent (C08)", "untested; contradicted", "audit basis; sysVI, scCRAFT, Maan", "OK"),
(":205-208", "July background attributions (Tung, taxonomy, Stegle)", "n/a", "Tung 2017, scIB, AD-AE texts; July :128, :137", "OK"),
(":209-210", "'Trained on real counts' in the gap line", "n/a", "SI-15", "OK"),
(":211", "'The premise becomes a tested hypothesis'", "n/a", "SI-68", "OK"),
(":212-213", "Pilot hint 'stock scVI 0.726 vs best conditioned arm 0.646' provisional", "n/a", "PAPER_PLAN:46 reads 0.726 and 0.646", "OK"),
(":214-215", "August-revision barycenter 'first' sentence", "n/a", "manuscript_tm tex:140 at 08f1e3c", "OK"),
]

EDITS = [
# (n, where, edit, my check, verdict)
(1, ":140 ('authored arms')", "R2 outcome no longer pending; X1 keeps the posterior mean; only the formal R2/R3 re-run pending", "Same wording as Methods :369; record values in section 5; SI-66; monotonic argument: a refit of Sinkhorn cannot raise b_common above the arm that caps it (barycenter {{r4_bary_blocks}} blocks, mmd {{r4_mmd_blocks}}, sinkhorn {{r4_sink_blocks}})", "Correct. Residual: sim1|1 also flagged (INT-15); plain wording (INT-15)"),
(2, ":177 (item 3)", "R3 outcome no longer pending; standardisation off in X1; formal re-run pending", "Same record; SI-20, SI-27, SI-66, SI-57", "Correct; INT-10 concerns the first clause of the same item"),
(3, ":172 (Scope)", "'for all arms and baselines' narrowed to every arm, plain scVI and sysVI with declared exceptions", "Manifest: epochs, minibatch, latent equal across X1 arms and sysVI per task; scANVI extra epochs and X12 levels declared in Methods; Harmony, Scanorama, PCA have no epochs", "Correct; decoder dimension missing (INT-13); Methods :334 still unnarrowed (INT-24)"),
(4, ":178 (item 4)", "'factorial X6' replaced by the glossary's X6 design", "Manifest X6: eight arm-step combinations, {{x6_rows_each}} rows each; JS x steps cells absent by SI-01; glossary :62; Methods :448", "Correct; citations INT-21"),
(5, "Introduction and Related work", "Three file paths 'scripts/x.py' written 'x.py'", "Reading guide (paper.tex How to read this outline); scvi_adversarial_plan.py exists as scripts/scvi_adversarial_plan.py at 08f1e3c", "Correct"),
]


STATUS = {
# line: (tags found, classification check, finding)
139: ("1 source tag", "True at 08f1e3c (scvi-tools 1.4.2 lines; plan.py:348-351; manifest scvi_adv rows)", "INT-17 (tag completeness)"),
140: ("2 source tags, 1 pending", "Both source tags true (SI-19 and plan.py:359; SI-66 and the frozen record); pending correct (SI-57)", "INT-15 (second flagged block; plain wording)"),
141: ("1 source tag", "True (SI-01; plan.py:8-9, 292-298); the untagged clause after it is an inference", "INT-12"),
143: ("1 source tag", "True (SI-68; PAPER_PLAN section 3; decision letter, Reviewer 1, major comment 2)", "INT-15 (reviewer remark in plain text)"),
166: ("1 source tag", "Source is a design document and a search record that holds for 2024 on only; the statement is a search outcome", "INT-02, INT-01, INT-03 (should be pending)"),
167: ("1 source tag", "Source is PAPER_PLAN section 2; 'no first claim' is a writing choice", "INT-02 (should be pending)"),
172: ("2 source tags", "True (SI-10, SI-04, SI-02; manifest checked); decoder dimension missing", "INT-13, INT-24"),
173: ("1 pending", "Correctly pending: no ledger entry adopts the decomposition", "none"),
175: ("1 source tag", "True (SI-68)", "none"),
176: ("1 pending", "Analysis correctly pending; the design is locked but untagged", "INT-21"),
177: ("3 source tags, 1 pending", "SI-18 fixes arms and grids only; SI-68 puts W1, MMD and Sinkhorn side by side in all-arm tables, but no locked analysis attributes a difference to W1 geometry; the other two tags and the pending marker are right", "INT-10"),
178: ("1 source tag, 1 pending", "True (SI-22; manifest X6 combinations); pending correct", "INT-21 (citations)"),
179: ("1 pending", "Classification right; finding labels wrong (D-07 belongs to X8)", "INT-09"),
181: ("1 source tag, 1 pending", "Tags right for the comparison; the guidance clause has no tag of its own", "INT-11"),
}
