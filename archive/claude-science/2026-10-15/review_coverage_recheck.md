# Rigor re-check: coverage additions and removals, paper.tex d556bb1 -> 7ad3eb4

Reviewer: Rigor Reviewer. Date: 2026-10-15. Scope: the decision diff only. User decisions 1-5 were taken as given.

## Verdict
No blockers. 7 fixes, 6 notes.

Reviewed the diff d556bb1 -> 7ad3eb4 against the five user decisions. Applied correctly: scCRAFT, he2025partially and reid2025topology are gone from printed text and bib (59 cited keys = 59 bib entries, none orphaned; tectonic build clean, no undefined citations); no explicit 'first' statement remains; the S-X Disclosures list (15 items left) now concerns only the preregistered study with no dangling sentence; all eight coverage methods plus GLUE are added with the right keys, and the nine BibTeX entries match Crossref (Portal byline checked: positions 0-5 and 163-165 of 166). No blockers. Seven fixes: iDLC full text IS retrievable and shows a WGAN-GP discriminator plus Sinkhorn term towards a reference batch, so its line, placement and the Gap bullets need changing (CV-01); Li et al. is misplaced and missing qualifiers (CV-02); GLUE gloss omits adversarial alignment (CV-03); two printed leftovers about the earlier submission remain at :35 and :147 (CV-04, CV-05); scStarCorrect/Portal sit under a JS-type heading with unknown loss and StarGAN used WGAN-GP (CV-06); evidence.md contradicts itself in five places (CV-07). Six notes. Prior-work scope: no new database search run; the 'no comparison on one stock scVI backbone' claim was re-read against iDLC and Li et al. only, and remains accurate if the 'stock scVI' qualifier is kept. Not read: full texts of iDLC Results tables, scStarCorrect, Portal; the sysVI text beyond GLUE passages.

## What was checked
- Diff (310 lines) and full paper.tex at 7ad3eb4; printed text (comments excluded) scanned by regex for scCRAFT, July, preprint, earlier, original, submission, BIOINF, Topology, audit, D1-D9, defect, reviewer-letter and novelty wording; S-X Disclosures read item by item.
- Full texts read for Beaconet (PMC), AIF (PLOS CB), ABC (PMC), IMGG (PMC), Li et al. (BMC Bioinf), GLUE (Nat Biotechnol), and the sysVI passages on GLUE; iDLC full text fetched and read in the architecture and objective sections; abstracts re-read for scStarCorrect and Portal (journal and bioRxiv versions). Portal and scStarCorrect full texts not obtainable.
- Nine BibTeX entries compared with Crossref (title, journal, volume, issue, pages, year, author list; Portal byline); refs.bib rebuilt from the diff (59 entries); paper compiled with tectonic: no undefined citations, new entries render.
- Cited-keys vs bib-keys: 59 vs 59, none orphaned; he2025partially and reid2025topology absent from both.
- Not checked: iDLC Results tables for a JS or MMD arm; scStarCorrect and Portal loss forms; the SI-13 wording; CONSTRAINTS.md (not an input).

## Findings
### CV-01 (fix) - paper.tex:341 (Kernel and optimal-transport alignment, iDLC) and :170; evidence.md iDLC entry and line 66
**Issue.** The iDLC line is built from the abstract only and its pending marker says bioRxiv returned HTTP 429. The full text is retrievable (Unpaywall lists a green-OA bioRxiv PDF; the fetch tool returned it today). The full text shows that iDLC is the closest published design to the W1-critic and Sinkhorn arms studied here: a WGAN-GP discriminator towards a reference batch, plus an entropy-regularised Sinkhorn term in the generator loss. The line understates this and files it under Kernel and OT, where a reader would not expect a W1 critic. The Gap bullet at :170 ("W1 critics already exist") does not cite it. The claim at :169 (no JS/W1/MMD/OT comparison on one stock scVI backbone) is still accurate, because iDLC uses a residual autoencoder rather than scVI, but a referee will look for iDLC there.

**Evidence.** Full text 10.64898_2026.02.17.706490.pdf: Results "Overview of the iDLC algorithm" (stage 1: residual autoencoder, hard-partitioned latent space; stage 2: "optimal transport-regularized generative adversarial network" trained on MNN pairs); Methods "Optimal Transport-guided Adversarial Correction": generator maps source-batch expression profiles to the reference batch (Eq. 7, G(x)=ReLU(F(x)+x)); discriminator separates reference-batch cells from corrected cells; objective "based on the WGAN-GP framework" with a gradient-penalty term (Eq. 9); generator loss = -E[D(G(x))] + lambda * OT(A, G(x_b)), OT computed with the entropy-regularised Sinkhorn algorithm (Eq. 10 and following). evidence.md:66 (earlier round) already recorded "WGAN-GP discriminator with a Sinkhorn regulariser ... not cited", which the new evidence entry ("Not stated in the abstract: the adversary's divergence, any reference batch") does not reconcile. I searched the extracted text for discriminator/adversarial/Wasserstein/Sinkhorn/reference only; I did not read the Results tables for a JS or MMD arm (provisional).

**Proposed text.**
```
Move the line to "Wasserstein critics" (after IMGG), drop the pending marker, and replace with:
\item iDLC: a residual autoencoder whose latent space is split into biological and technical parts, then a generator that maps the expression profiles of each source batch to a reference batch, trained against a WGAN-GP discriminator with a Sinkhorn optimal-transport term added to the generator loss, using mutual-nearest-neighbour cell pairs as supervision \citep{jiang2026geometric}.
:170 -> \item Reference anchoring and W1 critics already exist: a W1 critic for domain-invariant features \citep{shen2018wasserstein}, reference-anchored WGAN-GP integration \citep{wang2021imap}, a WGAN-GP critic with a Sinkhorn term towards a reference batch \citep{jiang2026geometric} and MMD alignment to a reference or target batch \citep{amodio2019exploring,shaham2017removal}.
evidence.md: replace "abstract only (bioRxiv full text not obtained ...)" with "Introduction; Results (Overview of the iDLC algorithm); Methods (Optimal Transport-guided Adversarial Correction, Eqs. 7-10); text: bioRxiv PDF via Unpaywall (green OA)"; delete "Not stated in the abstract ..."; change line 66 "not cited" to "cited in Related work".
```

### CV-02 (fix) - paper.tex:299 (Benchmarks, Li et al.) and :169 (Gap)
**Issue.** Li et al. is a method paper (scVI plus a batch penalty), not a benchmark; the notes assigned it to Kernel and OT alignment and the diff put it under Benchmarks. The line also drops two qualifiers that matter for this paper: the backbone is a preliminary scVI (version 0.3.0), and the MINE penalty is trained adversarially. It is the only retrieved work that puts an adversarially trained penalty and MMD side by side on scVI, so the Gap bullet (:169) should name it and say why it is not the JS/W1/MMD comparison.

**Evidence.** Full text: Abstract (learns "the entire trade-off curve"; Pareto MTL; MINE vs MMD, "benefits over" the standard MMD measure); footnote 3 (page 3): preliminary scVI 0.3.0 (6 Mar 2019 commit); "Pareto MTL with MINE": MINE is maximised over its own network parameters before each minimisation step (adversarial training); Discussion: MINE needs adversarial training, "can be unstable". MINE estimates mutual information, so it is neither a JS discriminator nor a W1 critic.

**Proposed text.**
```
Move to "Kernel and optimal-transport alignment" and replace with:
\item Li et al.: on a preliminary scVI (version 0.3.0), adds a latent-space batch measure to the scVI loss, either maximum mean discrepancy or an adversarially trained neural estimate of mutual information (MINE, Mutual Information Neural Estimation), and learns the whole trade-off between the two objectives by Pareto multi-task learning \citep{li2022trade}.
:169 -> append before "a search-based absence": "; the nearest scVI designs are Li et al., which compares MMD with an adversarial mutual-information penalty on a preliminary scVI \citep{li2022trade}, and iDLC, a W1 critic with a Sinkhorn term on a residual autoencoder \citep{jiang2026geometric}, neither with a JS discriminator and W1 critic side by side [check iDLC Results for a JS or MMD arm before locking]"
```

### CV-03 (fix) - paper.tex:155 (Composition shift, GLUE gloss)
**Issue.** The citation is correct (cao2022multi is the GLUE paper, and sysVI cites GLUE as [34]) and the gloss matches the paper title. But the sentence is about adversarial methods mixing cell types, and the gloss omits the one property that makes GLUE relevant: alignment of omics layers by an adversarial discriminator. sysVI calls GLUE an adversarial model; a reader of the outline sees only "graph-linked embedding". evidence.md already flags this omission, but the line was not changed. The finding itself (GLUE mixed unrelated cell types with unbalanced proportions) is hrovatin2025integrating's, which I re-read: Fig. 2b and the Results text report mixing of delta, acinar and immune cells in the GLUE embedding of the pancreatic mouse-human data.

**Evidence.** glue.txt (publisher PDF): abstract "graph-linked unified embedding"; Methods "Weighted adversarial alignment": canonical adversarial alignment "amounts to minimizing a generalized form of Jensen-Shannon divergence" among omics-layer embedding distributions; GLUE adds balancing weights for unbalanced populations. sysVI (10.1186/s12864-025-12126-3, PDF read today): Fig. 2 legend and Results call GLUE a "well-established adversarial model"; the text notes the GLUE authors down-weight cells from unbalanced populations using a preliminary integration round.

**Proposed text.**
```
\item An adversarial conditional variational autoencoder (VAE) and GLUE (graph-linked unified embedding, a multi-omics integration method that aligns the embeddings of omics layers with an adversarial discriminator; \citealp{cao2022multi}) mixed unrelated cell types with unbalanced proportions, more so at higher correction strength \citep{hrovatin2025integrating}.
```

### CV-04 (fix) - paper.tex:35 (title block)
**Issue.** Printed text still names the earlier submission: "Outline of the revision of Bioinformatics submission BIOINF-2026-1434". User decision 3 says the paper does not mention the earlier submission. The same string is in the line-1 comment, which is excluded.

**Evidence.** Non-comment scan of paper.tex at 7ad3eb4 (regex over scCRAFT, July, preprint, earlier, original, submission, BIOINF, Topology, audit, D1-D9, defect): this is the only printed hit that names the earlier submission; the scan found no printed scCRAFT, July, "Topology" or D1-D9 text.

**Proposed text.**
```
{\small Outline of the paper; repository state: commit 08f1e3c, 2026-10-14.}
```

### CV-05 (fix) - paper.tex:147 (JS versus W1, last bullet; inside \src{})
**Issue.** The source tag cites "decision letter, Reviewer 1, major comment 2 (fairness of the comparison)", a peer-review letter on the earlier submission. It is printed grey text, so it mentions the earlier submission and its review. The claim it supports (R0 is exploratory) is supported by SI-68 and PAPER_PLAN section 3 already named in the same tag.

**Evidence.** paper.tex:147 as printed; PAPER_PLAN section 3 and SI-68 are already in the tag.

**Proposed text.**
```
\src{SI-68; PAPER\_PLAN \S3}
```

### CV-06 (fix) - paper.tex:318 (scStarCorrect), :319 (Portal) under "Adversarial (JS-type) integration"; pending markers at the ends of both lines
**Issue.** Both lines sit under a heading that states the loss class (JS-type), although the abstract of neither paper gives the adversarial loss form, and the evidence entries say so. For scStarCorrect the placement is likely to be wrong: the model is StarGAN-based, and StarGAN replaced its original GAN objective with a WGAN-GP objective for training stability. If scStarCorrect keeps that objective, it is a W1-critic method with a fixed reference batch and belongs under "Wasserstein critics", which is the most relevant class for this paper. The pending marker names "paragraph placement" but not the reason. Line text itself is accurate to the abstract (abstract re-read today: shared VAE conditioned on batch labels; domain-aware discriminator enforcing alignment to a fixed reference batch; one-way correction; four loss terms). Portal line is accurate to its abstract (re-read today, journal and bioRxiv versions).

**Evidence.** scStarCorrect abstract via fetch tool (IEEE closed; no OA location). StarGAN paper arXiv 1711.09020, Section on improved GAN training: the original objective is replaced by a Wasserstein GAN objective with gradient penalty (read on arXiv today; the scStarCorrect text is not read, so the inheritance is a hypothesis, not a finding). Portal: full text not obtainable (Unpaywall no OA location; bioRxiv DOI returned HTTP 429), so its loss class is unknown.

**Proposed text.**
```
:318 pending -> \pending{full-text check, including the adversarial loss form (the StarGAN model it builds on used a WGAN-GP objective; if kept, move this line to Wasserstein critics) and paragraph placement (abstract read only; IEEE closed access)}{full text obtained}
:319 pending -> \pending{full-text check, including the adversarial loss form and paragraph placement (abstract read only; subscription journal; bioRxiv version not retrievable)}{full text obtained}
Alternative that avoids a class label on unread papers: move both lines to a neutral final paragraph "Other adversarial and reference-guided methods (loss form not yet read)" until the full texts are in.
```

### CV-07 (fix) - paper/evidence.md: iDLC entry (new); line 66; line 85; line 92 (Version substitutions and limits); line 65
**Issue.** The additions do not reconcile with older lines in the same file. (a) iDLC entry says "full text not obtained ... no mirror tried"; the full text was obtainable (CV-01). (b) Line 66 says iDLC was "not cited"; it is now cited. (c) Line 85 ("Cite scCRAFT only as the earlier backbone ... two instructions conflict ... related.tex keeps one scCRAFT-backbone line marked pending") describes a conflict that user decision 1 closed, and refers to a line that no longer exists. (d) The abstract-only summary at line 92 lists maan2024, lotfollahi2020, goh2022 and was edited only to drop reid2025topology; it omits the three new abstract-only keys (jiang2026geometric until CV-01 is applied, gu2025scstarcorrect, zhao2022adversarial). (e) The GLUE entry states that the paper.tex gloss "omits the adversarial alignment", which is a known defect left in place (CV-03). (f) The Gap search record (lines 62-70) does not say that iDLC, Li et al., Beaconet, IMGG and the other six were assessed against the "no comparison" claim and why none is one.

**Evidence.** diff hunks of evidence.md at d556bb1->7ad3eb4; evidence.md at d556bb1 (artifact e493a4e8), lines 65, 66, 85, 92.

**Proposed text.**
```
line 85 -> mark "Resolved 2026-10-15: user decision, main text does not name scCRAFT; entry kept as history" or delete.
line 92 -> "Abstract only (closed access): maan2024characterizing, lotfollahi2020conditional, goh2022batch, gu2025scstarcorrect, zhao2022adversarial; ..."
Gap search section, add: "Coverage works assessed against the gap claim 2026-10-15: iDLC (W1 critic + Sinkhorn term, reference batch, residual autoencoder), Li et al. (MMD vs adversarial MINE, preliminary scVI 0.3.0), Beaconet, IMGG, AIF, ABC, scStarCorrect, Portal; none compares JS, W1 and MMD/OT on one stock scVI backbone [iDLC Results not yet read for such an arm]."
```

### CV-08 (note) - paper.tex:329 (Beaconet)
**Issue.** The line is accurate. It omits what distinguishes Beaconet's per-batch networks from this paper's arms: each network compares one batch with the mixture of the other batches, which is the same pooling idea as the pooled critic here. Also, the source calls the networks "auxiliary encoder-L" and never uses the word "critic"; "critic" is the outline's gloss, consistent with how the outline glosses IMAAE.

**Evidence.** beac.txt (PMC11234410) Methods: Eq. 5 defines the Wasserstein distance between P_j and "the mixture of batch {i | i != j}"; Eq. 8 adds the soft 1-Lipschitz penalty; lambda default 10 after Gulrajani et al.; Corrector input is log-scaled expression and batch index; abstract and Introduction: reference-free, no cell-type labels or cell pairs. The text contains no "gradient penalty" string; it says "soft penalty term for 1-Lipschitz condition" (the form is the WGAN-GP one).

**Proposed text.**
```
\item Beaconet (Batch Effect Adversarial Correction Network): a corrector network on log-scaled gene expression, trained against one Wasserstein-distance network per batch that compares the batch with the mixture of the other batches, with a soft 1-Lipschitz penalty (weight 10, after WGAN-GP); no reference batch is chosen \citep{xu2024beaconet}.
```

### CV-09 (note) - paper.tex:330 (IMGG)
**Issue.** The line is accurate but omits the L2 reconstruction loss and the authors' remark that the WGAN-GP loss alone made integration of several batches difficult. The outline gives the parallel IMAAE remark its own bullet (:327), so the omission makes IMGG look like a pure WGAN-GP method. The source calls D a "discriminator" with WGAN-GP loss and never writes "Wasserstein" (the word appears only in a reference title); the outline's "Wasserstein GAN with gradient penalty" is the standard expansion, but IMAAE is flagged as "never named Wasserstein" while IMGG is not.

**Evidence.** imgg.txt (PMC) Sec. 4.4: adversarial losses "optimized via the WGAN-GP" (Eqs. 1-2), then: only the WGAN-GP loss made integrating multiple batches difficult, so an L2 reconstruction loss (Eq. 3) was added; Sec. 4: intermediate target built per similar-cell cohort (Mean, Max, Min patterns); MNN graphs built in PCA space of HVGs; residual fully connected generator. Count of "Wasserstein": 1 (reference list).

**Proposed text.**
```
\item IMGG (integrating multiple single-cell datasets through connected graphs and generative adversarial networks): a generator on highly variable genes trained with a WGAN-GP loss (never named Wasserstein) plus an L2 reconstruction loss, which the authors added because the adversarial loss alone made integrating several batches difficult, towards a synthetic intermediate target built from similar-cell cohorts matched across batches by mutual nearest neighbours \citep{wang2022imgg}.
```

### CV-10 (note) - paper.tex:316 (AIF), :317 (ABC)
**Issue.** Both lines are accurate to the full texts. AIF: the source calls the correction target "the reference batch" (the study used the most represented batch), so "one chosen batch label" hides a reference design that is directly relevant here; the line also omits AIF's separate real/fake GAN discriminator. ABC: verified, including "no reference batch" (the string occurs only in the References heading) and the semi-supervised use of cell-type labels.

**Evidence.** aif.txt p5 "Batch effect correction": cells projected "onto the reference batch" by decoding under one batch label; "This study used the label corresponding to the most represented batch"; Fig 2 caption: CVAE, GAN (discriminator C_chi, real vs reconstructed, cross-entropy) and auxiliary network A_psi (batch from latent, cross-entropy, adversarial). abc.txt: abstract "semi-supervised ... supervised cell type classifier branches", Sec. 2.1 Eqs. 1-3.

**Proposed text.**
```
\item AIF (Adversarial Information Factorization): a conditional variational autoencoder on gene expression with a real-versus-reconstructed GAN discriminator and an auxiliary cross-entropy classifier of batch from the latent code that the encoder is trained against; cells are then decoded under one batch label, which acts as a reference (the study used the most represented batch) \citep{monnier2024novel}.
```

### CV-11 (note) - paper.tex:426 (Methods, "none selected from earlier sweeps")
**Issue.** The phrase "re-derived from new pilot fits, none selected from earlier sweeps" reads as a contrast with earlier sweeps, which in the decision-3 reading are the earlier submission's sweeps. It does not name them, so a reader cannot tell what "earlier" means, and "new" has no referent once the earlier work is not mentioned. I could not read SI-13 (CONSTRAINTS.md is not in the inputs), so the intended meaning is a question.

**Evidence.** paper.tex:426 as printed; S-X Disclosures item 2 already states the disjointness (grids from seeds 100-102, disjoint from X1).

**Proposed text.**
```
all strength grids derived from the pilot fits A1--A3 on seeds disjoint from X1 \src{SI-13}  [confirm against SI-13; if "earlier sweeps" means the prereg-tier12-v1 fits, name them and point to S-X item 1]
```

### CV-12 (note) - paper/evidence.md lines 51-60 history, 65, 66, 81-85, 94, and the new "Removed keys" section; paper.tex comment lines 209, 212, 223, 277; printed Gap bullet :169
**Issue.** Decision 3 covers the paper text, and comments are out of scope for this check. Two consequences to be aware of: (a) the printed Gap bullet (:169) tells readers to find "scope and limits" in paper/evidence.md, and that file still names scCRAFT, the July submission, the earlier preprint, and (in the new "Removed keys" section) the removal itself; (b) the comment block in paper.tex carries the same names and claim IDs (C01-C25). If paper/ is public on the GitHub branch, these are visible. No change is required by the decision as worded; this is recorded so the choice is explicit. Also: with the preprint no longer cited, the Gap bullet must keep the qualifier "on one stock scVI backbone"; without it the claim would be contradicted by the authors' own earlier work and by iDLC.

**Evidence.** Non-comment scan found none of the earlier-work names in printed text other than CV-04 and CV-05; evidence.md at d556bb1 (artifact e493a4e8): grep for scCRAFT, July, Topology, original/earlier submission finds lines 51, 60, 65, 66, 81-85, 94, 112, 115; the diff removes 51, 60, 112 and 115 and adds the "Removed keys" section, so 65, 66, 81-85 and 94 remain.

**Proposed text.**
```
No text change required. Option: before submission, strip the comment block and the earlier-work lines of evidence.md, or keep them in a non-public copy.
```

### CV-13 (note) - paper.tex:310-320 (Adversarial (JS-type) integration)
**Issue.** GLUE is cited only in the Introduction. Its adversarial alignment (multi-class discriminator, cross-entropy, generalised JS form) fits this paragraph and the Related work claims to cover the adversarial methods that the Introduction names. Optional; the user decision only asked for the reference at the Introduction.

**Evidence.** glue.txt Methods "Weighted adversarial alignment" (generalised Jensen-Shannon form); evidence.md GLUE entry.

**Proposed text.**
```
\item GLUE (graph-linked unified embedding): omics-specific variational autoencoders aligned by an adversarial multi-class discriminator, which in its canonical form minimises a generalised Jensen--Shannon divergence; within-layer batch effects enter the decoder as a covariate \citep{cao2022multi}.
```

## Status of earlier findings touched by this diff
- RC-01 / INT-14 / REL-04: closed by user decision 1 and 3. Diff removes paper.tex:154 (scCRAFT line), :169 (Earlier comparison) and :315-316 (scCRAFT lines) and their pending markers; scan of printed text finds no scCRAFT.
- INT-03: closed by user decision 3. The Gap bullet naming the earlier JS-vs-W1 comparison is removed on purpose. Residual: keep the "stock scVI" qualifier on :169 (CV-12).
- REL-03 / REL-06 / RC-08(a): closed by removal. reid2025topology deleted from refs.bib and paper.tex; no citation left; 59 cited keys = 59 bib entries.
- REL-07: addressed, 2 of 8 still pending. All eight coverage methods are in; iDLC full text is available (CV-01); scStarCorrect and Portal remain abstract-only with pending markers (CV-06).
- RC-07 (evidence.md bookkeeping): still open, extended. CV-07 adds five stale or missing entries introduced or left by this diff.

## Prior-work scope
No new database search. The gap claim ("no comparison of JS, W1 and MMD or OT objectives on one stock scVI backbone") was re-read against the two closest new works, iDLC and Li et al.; it stays accurate with the qualifier. Verdict is provisional for iDLC until its Results are read. Earlier searches (OpenAlex, arXiv, Europe PMC; 2026-10-15) are reused.