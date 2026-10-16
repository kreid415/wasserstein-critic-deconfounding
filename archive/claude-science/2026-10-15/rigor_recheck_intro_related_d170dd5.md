# Focused re-check: Introduction, Related work and references of paper/paper.tex at d170dd5

Track recheck_intro_related. Reviewer: RIGOR_REVIEWER. Date: 2026-10-15. Review commit d170dd5 (branch paper; parent add2f73, the commit the two rigour reviews read). Inputs: re-check bundle 4788d866, Introduction report ff66652f and ledger 24547ea4, Related-work report aed01ae8 and ledger fae7af2b.

## 1. Verdict

**PASS_WITH_FIXES.** New findings: 0 blockers, 2 fixes, 6 notes (8 in total). Of the 42 earlier findings: 34 resolved, 5 resolved differently (acceptable), 3 partly resolved, 0 not resolved. No locked statement was found false and no theorem, section or page pointer was wrong. The two fixes are a missing pending marker on a new line (RC-01) and a parenthetical about scIB-E that its cited source does not state (RC-02).

| id | severity | location (at d170dd5) | issue |
|---|---|---|---|
| RC-01 | fix | paper.tex:169 (Gap, 'Earlier comparison') | New Gap bullet names scCRAFT without the pending marker for the open naming decision (the other three mentions carry it). |
| RC-02 | fix | paper.tex:298 (Related work, Benchmarks, scIB-E) | scIB-E 'two adversarial; four not' is not stated by the source; the authors' code labels reverse cross-entropy adversarial. |
| RC-03 | note | paper.tex:138 (JS versus W1, non-saturating loss) | Thm. 2.5 is stated without its hypothesis (distributions with densities). |
| RC-04 | note | paper.tex:83, :139-140, :170 | GAN, WDGRL, iMAP, SAUCIE, MMD-ResNet and scVI are expanded after their first use. |
| RC-05 | note | paper.tex:142 (adversary input and the R4 flags) | One bullet holds six claims; rule R4, pilot A1, A2/A3 and the Sinkhorn refit are not glossed. |
| RC-06 | note | paper.tex:133, :144, :175, :185, :603 | Five source tags are weaker than the claim (SI-02, SI-01, SI-05, barycenter.py:8-10, scvi_adversarial_plan.py:346-360). |
| RC-07 | note | paper/evidence.md lines 4, 39, 42, 45, 68-69, 87, 91 and 112 | evidence.md: stale or missing entries (genevay, corani, shaham, lotfollahi) and no query strings for the two review searches. |
| RC-08 | note | paper/refs.bib lines 4, 83, 583 | refs.bib: verified stamps and header comment no longer match the corrections; year choice unrecorded. |

## 2. What was checked

- Repository: private clone of kreid415/wasserstein-critic-deconfounding at d170dd5 (origin removed); only paper/ differs from 08f1e3c, so every file:line tag points to the code the paper cites. Bundle paper.tex, refs.bib and evidence.md are byte-identical to the commit; the bundled patch equals `git diff add2f73 d170dd5` (34 hunks in paper.tex, 3 files).
- Every one of the 42 earlier findings against the text at d170dd5, with the lead's deviations judged on the evidence (section 5).
- Every changed line: 34 hunks of paper.tex (section 6), the refs.bib and evidence.md changes (section 8).
- Source texts read this round (pdftotext or XML, no page images): Goodfellow et al. 2014 (arXiv 1406.2661v1); Arjovsky et al. 2017 (PMLR v70); Arjovsky and Bottou 2017 (arXiv 1701.04862v1); Zhao et al. 2019 (PMLR v97); Genevay et al. 2018 (PMLR v84); Cuturi and Doucet 2014 (PMLR v32); Mescheder et al. 2018 (PMLR v80); Corani et al. (arXiv 1609.08905v3); scIB-E (PMC12636214) and its code repository; sysVI (PMC12577435); iMAP (PMC7891139); IMAAE (PMC10056671); scDREAMER (PMC10682386); scCRAFT (PMC11971424); scIB (PMC8748196); abstracts of Goh 2022, Yu 2024, trVAE, MMD-ResNet, AD-AE, BERMUDA, scANVI and eight coverage candidates; the JMLR page and PDF for Ganin et al.; Crossref and the bioRxiv API for the preprint; POT 0.9.7.post1 and scvi-tools 1.4.2 source.
- Code and records: scvi_adversarial_plan.py, fit_paper_config.py, alignment.py, barycenter.py, build_paper_manifest.py, the manifest of record (6930 rows, 3000 X1 rows), docs/prereg/r4_a1_record.json, docs/prepped_fingerprints_scib.json, CONSTRAINTS.md (SI entries cited), PREREG.md, PAPER_PLAN.md, the July submission, the decision letter. Machine checks are in recheck_checks.py (output recheck_checks_results.json).
- Rendering: paper.tex at d170dd5 was compiled with tectonic (bibtex, plainnat). The extracted text equals the bundled paper.pdf (0 differing lines); no undefined citation or reference; two underfull-box warnings, both at unchanged lines (paper.tex:388 and :839); 52 bibliography entries, 52 cited keys, none missing, none uncited; all three sources are ASCII. Page 7 of the PDF was viewed (inline mathematics, grey source tags and orange pending markers render as intended).

## 3. New findings

### RC-01 (fix) - paper.tex:169 (Gap, 'Earlier comparison') at d170dd5; residual of INT-14 and REL-04 (introduced by the INT-03 fix)

**Problem.** The new Gap bullet names scCRAFT as the backbone of the July comparison but carries no pending marker for the open decision whether the main text names scCRAFT. That decision is marked pending at :154, :315 and :316 (the INT-14 and REL-04 fixes). The two instructions on naming conflict and neither is a ledger entry (evidence.md, Conflicts), so a 'no mention' or 'earlier backbone only' decision would silently leave :169 in the text. The \src tag on :169 supports the fact about the July manuscript, not the choice to name the backbone in the Introduction.

**Evidence.** recheck_checks.py, sccraft_lines: scCRAFT occurs outside comments at paper.tex:154, :169, :315, :316; :154, :315 and :316 carry 'names scCRAFT' markers, :169 does not. docs/PAPER_PLAN.md:65 ('Cite scCRAFT only as the earlier backbone'); evidence.md Conflicts bullet 'Cite scCRAFT only as the earlier backbone': the two instructions conflict, neither is a ledger entry. July tex has no MMD or Sinkhorn text (0 hits in oup-authoring-template.tex and supplement.tex), so 'without MMD or Sinkhorn arms' is supported; 'OT arms' can be read as the W1 critic itself.

**Replacement / fix.** paper.tex:169 - replace the line with: \item Earlier comparison: our July submission compared a JS discriminator with a reference-based W1 critic on a fixed scCRAFT backbone \pending{whether the main text names scCRAFT (PAPER\_PLAN \S3: only as the earlier backbone; July convention: no mention)}{your decision}, without MMD or Sinkhorn arms \src{Topology\_Matters.zip: oup-authoring-template.tex:134, :142}; its public preprint \citep{reid2025topology} is to be replaced by the next version \src{SI-07}. \pending{statement that its results are not carried over, and the disclosure of its pipeline defects (PAPER\_PLAN \S1)}{your decision}

### RC-02 (fix) - paper.tex:298 (Related work, Benchmarks, scIB-E) at d170dd5; from the REL-16 fix

**Problem.** The moved line classifies the six scIB-E level-1 losses as 'two adversarial, a GAN and gradient reversal; four not'. The scIB-E paper does not make this classification, and the authors' own code labels the reverse cross-entropy loss adversarial, so the count rests on a definition the source does not give. The target of reverse cross-entropy (uniform over the other classes) has the same form as the scvi-tools fool loss that paper.tex:314 and :143 treat as adversarial, so the outline is inconsistent with itself if reverse cross-entropy is counted as 'not adversarial'.

**Evidence.** scIB-E full text (Genome Biol 26:398, PMC12636214, Methods, Level-1): only the GAN is described as 'an adversarial framework' with 'adversarial training'; gradient reversal is 'a domain adaptation method that leverages a Gradient Reversal Layer'; reverse cross-entropy 'encourages a uniform probability distribution across incorrect classes' (outside the reference list, 'adversarial' occurs only in the Introduction, 'adversarial learning [22, 24]', and in the GAN description). Code (github.com/Chenxin-Yi/scIB-E, shallow clone 2026-10-15), src/scIB-E/scvi/train/batch_correction_trainingplans/level1_batch_removal/_RCE.py: RCE_loss docstring 'Loss for adversarial classifier'; cls_target = (~one_hot)/(n_classes-1); configure_optimizers docstring 'Configure optimizers for adversarial training' with one optimizer over module and classifier parameters (no separate min-max step); _RBP.py: gradient-reversal layer, same docstring.

**Replacement / fix.** paper.tex:298 - replace '(two adversarial, a GAN and gradient reversal; four not)' with the list of the six losses: \item scIB-E (extended scIB): six losses on the batch label in one scVI framework (GAN, HSIC, orthogonal projection, mutual-information minimisation, gradient reversal, reverse cross-entropy), then label-aware levels in scANVI; loss weights tuned with Ray Tune (automated hyperparameter search) on the immune task \citep{yi2025benchmarking}.

### RC-03 (note) - paper.tex:138 (JS versus W1, non-saturating loss) at d170dd5; from the INT-07 fix

**Problem.** The line states what Thm. 2.5 gives 'at the optimal discriminator' without its hypothesis: two distributions with densities. The paragraph is about the disjoint-support regime of Thms. 2.1-2.2, where densities do not exist and KL(P_g||P_r) is not finite, so a reader can take Thm. 2.5 as covering that regime.

**Evidence.** Arjovsky and Bottou 2017, arXiv 1701.04862v1, Sec. 2.2.2, Thm. 2.5: 'Let P_r and P_g_theta be two continuous distributions, with densities ... D* = P_r/(P_g_theta0 + P_r)'; Thm. 2.6 (the noisy case) is the one stated for the conditions of Thms. 2.1 or 2.2. Text read by pdftotext; the ICLR camera-ready was not read.

**Replacement / fix.** paper.tex:138 - replace 'need not have vanishing gradients: at the optimal discriminator its expected gradient' with 'need not have vanishing gradients: for distributions with densities, at the optimal discriminator its expected gradient'.

### RC-04 (note) - paper.tex:83, :139-140, :170 at d170dd5; from the REL-08, REL-17, INT-02 and INT-15 fixes

**Problem.** Three expansions sit later than the first use. (a) 'GAN' is first used at :83 (glossary, arm row) and expanded only at :93 (inside WGAN-GP) and :127; ':83' also uses 'JS-type' before '(JS)'. (b) The Introduction (:170) now names WDGRL, iMAP, SAUCIE and MMD-ResNet; their expansions are in Related work (:321, :322, :332), which comes later; the REL-17 expansions were placed by Related-work order. Same pattern for scCRAFT at :154 and scIB-E at :167 (both pre-existing). (c) scVI is first used at :139 ('scVI's posterior noise') and expanded at :140.

**Evidence.** Rendered text and source order at d170dd5: first 'GAN' at paper.tex:83 (PDF text line 132 of the extraction); expansion 'generative adversarial network (GAN)' at :127; :170 names the four methods without expansion; :321, :322, :332 expand them; scVI expansion at :140 follows the use at :139. Readability standard of the task: acronyms spelled out at first use.

**Replacement / fix.** (a) paper.tex:83 - 'discriminator (JS-type, after the Jensen--Shannon (JS) form of the objective of the original generative adversarial network (GAN))'. (b) paper.tex:170 - drop the acronym tags and keep the citations: a W1 critic for domain-invariant features \citep{shen2018wasserstein}, reference-anchored WGAN-GP integration \citep{wang2021imap} and MMD alignment to a reference or target batch \citep{amodio2019exploring,shaham2017removal}. (c) paper.tex:139-140 - end :139 after '(Cor.~3.1) \citep{arjovsky2017towards}.' and, in :140, insert before the \src tag: '; the noise scale is cell-specific, a case Thm.~3.1 does not cover directly'.

### RC-05 (note) - paper.tex:142 (adversary input and the R4 flags) at d170dd5; residual of INT-15

**Problem.** The bullet holds six claims and about 170 words (default input; X1 keeps it; the pilot rules; the two flagged blocks; which block bears on rules R2/R3; why refitting Sinkhorn cannot clear the flags). 'rule-R4 record', 'pilot A1', 'A2 or A3 task' and 'refitting Sinkhorn' carry no plain meaning in the sentence, against the standard that project labels carry their meaning where they are used and that an outline line holds one claim.

**Evidence.** paper.tex:142 read at d170dd5. Facts in the bullet verified: flagged blocks immune|1 (0.0188) and sim1|1 (0.0069); setters barycenter 5, mmd 1, sinkhorn 0; A2 and A3 cover atac_small and immune only (recheck_checks_results.json: r4_blocks, r4_setter_counts, a2_tasks, a3_tasks); the Sinkhorn arm was corrected after the pilot and its pilot fits are redone (docs/PREREG.md:209 and :377).

**Replacement / fix.** paper.tex:142 - replace by three bullets: (1) \item The six arms implemented in this project (discriminator; reference, pooled and barycenter W1 critics; MMD; Sinkhorn) are given the posterior mean by default \src{SI-19; \file{scvi_adversarial_plan.py}:359}. X1 keeps this default and keeps standardisation off \src{SI-56; SI-66}: the pilot rules that could switch the adversary input to posterior samples (rule R2) or standardise it per latent dimension (rule R3) keep their defaults while a block they use is flagged low-signal, that is, while the batch removal that all six arms reach exceeds plain scVI's by less than 0.02. \pending{formal re-run of rules R2 and R3}{after the Sinkhorn redo (SI-57)} (2) \item The frozen record of the strength-matching rule (rule R4), run on the penalty-strength pilot (A1), flags two blocks, both with the conditioned decoder: immune (difference 0.0188) and sim1 (0.0069). Rules R2 and R3 are run on pilots A2 and A3, which cover atac\_small and immune only, so only immune bears on them \src{\file{docs/prereg/r4_a1_record.json}; manifest}. (3) \item Refitting Sinkhorn (the Sinkhorn arm was corrected after the pilot, so its pilot fits are redone) cannot clear either flag: the common level is set by the arm that reaches the least batch removal (barycenter in five of the six blocks, MMD in sim1 with the conditioned decoder, Sinkhorn in none), so refitting Sinkhorn can only lower it \src{\file{docs/prereg/r4_a1_record.json}; SI-57}.

### RC-06 (note) - paper.tex:133, :144, :175, :185, :603 at d170dd5; new or changed \src tags

**Problem.** Five new or changed source tags are weaker than the claim they follow. (a) :175 'eight tasks of the scIB benchmark \src{SI-02}': SI-02 says 'standard scIB datasets; no new datasets'; the number is in the manifest tasks and docs/prepped_fingerprints_scib.json. (b) :144 'the discriminator takes one step per encoder step \src{SI-01}': SI-01 forbids a 10-step discriminator; one step is in the manifest (n_critic 1) and scvi_adversarial_plan.py:8-9. (c) :185 'architecture, hyperparameters ... (X12, X15) \src{SI-05; SI-44}': SI-05 reads 'tier 1 + 2'; X12 is in Tier 2 only through docs/PAPER_PLAN.md:155 ('Tier 1 + 2 (adds X3, X6, X12)') and the glossary. (d) :603 'Algorithm 2 with theta = 1': read in src/wcd_vae/wcd/barycenter.py:8-10, which is not in the tag (:38, :61, :71-73). (e) :133 'the encoder plays the generator': the tag covers :346-360 where the latent enters the adversary; the encoder's update against the adversary term is at :433-439 (:366-376 for critic-free arms); the mapping is an interpretive reading of those lines.

**Evidence.** CONSTRAINTS.md SI-01, SI-02, SI-05 read; manifest read (X1 discriminator n_critic 1); docs/PAPER_PLAN.md:155; src/wcd_vae/wcd/barycenter.py:8-10 read ('Alg. 2 with theta = 1'); scripts/scvi_adversarial_plan.py:346-360, :366-376, :433-439 read.

**Replacement / fix.** (a) :175 \src{SI-02; \file{docs/prepped_fingerprints_scib.json}}. (b) :144 \src{SI-01; \file{scvi_adversarial_plan.py}:8--9; manifest}. (c) :185 \src{SI-05; SI-44; PAPER\_PLAN \S4}. (d) :603 add ':8--10' to the wcd/barycenter.py lines in the tag. (e) :133 \src{\file{scvi_adversarial_plan.py}:346--360, :433--439}.

### RC-07 (note) - paper/evidence.md lines 4, 39, 42, 45, 68-69, 87, 91 and 112 at d170dd5

**Problem.** The record is accurate for the two review searches (counts match the reports) but has stale or missing entries after this round. (a) Line 4 lists genevay2018learning among 'the 3 keys added at integration'; it was added after the review (line 110). (b) Line 39 (corani2017statistical) keeps '[methods track; not cited here]', 'key name does not follow the convention (see notes)' and 'arXiv 1609.08905 (accepted manuscript)'; the key now follows the convention, is cited at paper.tex:443, and arXiv v3 (2016-11-22) predates the journal version (online 2017-05-18, Crossref), so it is the pre-acceptance preprint; lines 87 and 91 still use corani2017time. (c) Line 45 (shaham2017removal) still records 'several sources calibrated to a single target' (arXiv v6) as the checked claim, while paper.tex:331 now follows the published text (one source, one target; REL-02) and the last section has no entry for the change. (d) Lines 68-69 record the two review searches by counts and artifact ids (which resolve only inside the project; the repository is public); the 6 and 14 query strings are not listed, unlike the first search. (e) Line 42 (lotfollahi2020conditional) still says abstract only; the last section says abstract plus arXiv v2 full text.

**Evidence.** evidence.md read in full at d170dd5; Crossref 10.1007/s10994-017-5641-9 (issued 2017-05-18); arXiv abs 1609.08905 (v3 2016-11-22); report ff66652f section 7 (6 queries, 194 hits, 175 unique, 59 rated: 45 none, 14 partial, 0 substantial, 54 forward-citation works) and report aed01ae8 section 8 (14 queries, 330 rows, 315 unique, 18 candidates, recall 9 of 16) agree with lines 68-69.

**Replacement / fix.** Edit line 4 ('the 2 keys added at integration (mescheder2018training, miyato2018spectral)'; genevay2018learning is in the last section); replace line 39's tail with 'cited once in Methods (SI-74 rule); text from arXiv 1609.08905v3 (2016-11-22, before the journal version)' and rename corani2017time in lines 87 and 91; add under 'Changes after the rigour review': '- shaham2017removal -- paper.tex:331 follows the published text (PMC5870543: one source replicate, one target replicate); the arXiv v6 sentence on several sources is not used.'; append the query strings to lines 68-69 (Introduction track: 'Sinkhorn divergence single-cell integration batch effect autoencoder'; 'WGAN-GP batch effect removal scRNA-seq'; 'Wasserstein critic versus discriminator single-cell batch correction'; 'adversarial MMD optimal transport batch correction variational autoencoder single-cell comparison'; 'controlled benchmark batch correction loss functions single-cell integration same backbone'; 'latent alignment loss comparison scVI batch effect Jensen-Shannon Wasserstein maximum mean discrepancy'; Related-work track: set 1 of 8 and set 2 of 6 in report aed01ae8 section 8); update line 42 to match the last section.

### RC-08 (note) - paper/refs.bib lines 4, 83, 583 at d170dd5

**Problem.** Bookkeeping in refs.bib no longer matches the corrections. (a) reid2025topology carries 'verified = 2026-10-14 via doi', but its title now follows the bioRxiv record of versions 2 and 3 ('... in Single-Cell Data Integration'), not the Crossref title ('... for ...'); ganin2016domain carries '2026-10-14 via url', but the JMLR landing page has the misspelling and the correction comes from the JMLR PDF. (b) The header comment (line 4) says corani2017statistical was renamed 'at integration on 2026-10-15'; the rename is in d170dd5, after the review. (c) reid2025topology prints 'bioRxiv, 2025. doi: ... Version 3, posted 2026-05-05.': the year follows the DOI posting (Crossref issued 2025-11-13), not the cited version; evidence.md does not record this choice.

**Evidence.** bioRxiv API (retrieved 2026-10-15): v1 2025-11-13, v2 2026-01-21, v3 2026-05-05; v2 and v3 titles end 'in Single-Cell Data Integration'. Crossref 10.1101/2025.11.12.688061: title 'for', posted 2025-11-13, deposited 2026-05-08. JMLR page citation_author 'Mario March'; JMLR PDF page 1 'Mario Marchand'. Compiled bbl entry for reid2025topology (tectonic build of d170dd5).

**Replacement / fix.** refs.bib: set verified = {2026-10-15 via bioRxiv API} for reid2025topology and {2026-10-15 via JMLR PDF} for ganin2016domain; change line 4 to 'renamed after the rigour review on 2026-10-15'; add to the evidence.md reid2025topology bullet: 'year 2025 = DOI posting (Crossref issued 2025-11-13); the cited version is 3 (2026-05-05)'.

## 4. Status of the 42 earlier findings

| id | severity | status at d170dd5 | where | note |
|---|---|---|---|---|
| INT-01 | fix | resolved | paper.tex:168, :176 | 'budget-matched' removed; :176 states 5 versus 1 critic steps and that X6 varies steps for the pooled critic only. Manifest read: X1 n_critic discriminator 1; reference, pooled, barycenter 5; X6 pooled 1/5/10, pooled_sn 1/5, discriminator variants 1. Omitting the ANALYSIS_SPEC tag is correct (agent draft). |
| INT-02 | fix | resolved differently, acceptable | paper.tex:168, :170; evidence.md:68-70 | Gap line locked to evidence.md 'Gap search' with scope and a pending repeat; 'no first claim' pending; WDGRL, iMAP, SAUCIE, MMD-ResNet cited (iMAP full text and MMD-ResNet abstract re-read). Better than the proposal on locking; acronym tags at :170 unexpanded (RC-04). |
| INT-03 | fix | resolved differently, acceptable | paper.tex:169; refs.bib:583 | Comparison attributed to the July submission (tex :134, :142 read; no MMD or Sinkhorn text), preprint cited only as its public version (title and authors match; bioRxiv API v1-v3); SI-07 and pending disclosure present. More careful than the proposal (preprint text unread). scCRAFT naming marker missing here (RC-01). |
| INT-04 | fix | resolved | paper.tex:152, :196-198 | Restated as hypothesis tested by X3; drop-list text as proposed. 'expected to' dropped; acceptable because :151 states the zero-source-error and binary-label conditions of Wu et al. Prop. 3.1. |
| INT-05 | fix | resolved | paper.tex:150 | Precondition and symbols added; Zhao et al. Thm. 4.3 text read (Markov chain condition, d_JS(Y) >= d_JS(Z), JS distance). |
| INT-06 | fix | resolved differently, acceptable | paper.tex:144 | Covers WGAN Thm. 1, Example 1 and Thms. 2.4-2.6 and the near-optimal discriminator of Thm. 2.4 (assumptions read in both papers). Last clause says R0 'describes' whether the conditions arise instead of 'discriminator close to optimal'; this fits R0 (separability and gradient norms, :233). Tag \src{SI-01} for 'one step' is weak (RC-06). |
| INT-07 | fix | resolved | paper.tex:138 | Noise model, Thm. 2.5 content and 'too strong' stated; Thms. 2.5 and 2.6 and the remark read. Thm. 2.5 precondition omitted (RC-03). |
| INT-08 | fix | resolved | paper.tex:127 | Sec. 4.1, Eq. 6, in the proof of Thm. 1 read in Goodfellow et al. (C(G) = -log 4 + 2 JSD). |
| INT-09 | fix | resolved | paper.tex:183 | D-06 and I089, as in the glossary :63. |
| INT-10 | fix | resolved | paper.tex:181 | As proposed; SI-18 and SI-68 read; attribution to W1 geometry pending. |
| INT-11 | fix | resolved | paper.tex:186 | Comparison locked (SI-68, SI-77); guidance pending. |
| INT-12 | fix | resolved | paper.tex:143, :162 | Fool-loss clause tagged (scvi_adversarial_plan.py:297-298 read: for k = 2 the target is the other batch alone); image-GAN inference marked pending. |
| INT-13 | fix | resolved | paper.tex:175 | Both decoders named; SI-09, SI-11, SI-16, SI-70 read; glossary :82 and tests/scvi/test_adversarial_plan_v2.py:118-121 agree. |
| INT-14 | fix | partly resolved | paper.tex:154, :315 (and :316) | Pending markers present on the three mentions the finding named; the new mention at :169 (INT-03 fix) has none (RC-01). |
| INT-15 | fix | partly resolved | paper.tex:133, :140-142, :145, :153 | Done: mapping bullet; posterior mean and sample defined; GLUE glossed; reviewer remark moved into the tag; both flagged blocks, setters and values verified against the frozen record. Not done: 'rule R4', 'pilot A1', 'A2/A3' and the reason for refitting Sinkhorn are not glossed in :142, and scVI is used at :139 before its expansion at :140 (RC-04, RC-05). |
| INT-16 | fix | resolved | refs.bib:83 | 'Mario Marchand'; JMLR PDF page 1 read; prints correctly. |
| INT-17 | note | resolved | paper.tex:141 | scvi-tools 1.4.2 _vae.py:390, :413 and _trainingplans.py:689 read in the wheel; manifest 'mean' for the 40 X1 scvi_adv rows verified; two flagged blocks stated. |
| INT-18 | note | resolved | paper.tex:120, :134, :136, :139 | As proposed; Yu et al. abstract, WGAN Thm. 1(3), Thms. 2.1-2.2, Thm. 3.1 and Cor. 3.1 read. |
| INT-19 | note | partly resolved | paper.tex:119, :155; evidence.md:116 | Goh line softened (abstract re-read; 'for modelling them' glosses the next abstract sentence). Maan line unchanged and not relocated; it stays within the abstract and evidence.md lists both as abstract-only. Risk low. |
| INT-20 | note | resolved | paper.tex:167 | As proposed; scIB-E full text: 0 hits for Wasserstein, MMD, optimal transport, Sinkhorn; HSIC definition read. |
| INT-21 | note | resolved | paper.tex:180, :182 | X3 design tagged (SI-41; manifest: one n_cells per task, 8402 / 20000 / 14409 / 17872); X6 citations added. |
| INT-22 | note | resolved | paper.tex:185 | X8, local mixing, X12, X15 added with status; SI-05 is indirect (RC-06). |
| INT-23 | note | resolved | paper.tex:195, :201-202 | As proposed; decision letter Reviewer 2 main concern 3 (letter lines 65-67) read. |
| INT-24 | note | resolved | paper.tex:345 | Methods narrowed; fit_paper_config.py:393-396 and :404-413 and docs/PAPER_PLAN.md:83 read; manifest sysvi and scanvi rows carry n_latent 10, 1 layer, 128 hidden, batch 128, train_size 0.9. |
| INT-25 | note | resolved | paper.tex:223-225 | C03 line added as proposed. |
| REL-01 | fix | resolved | refs.bib:83 | As INT-16; printed 'Mario Marchand'. |
| REL-02 | fix | resolved | paper.tex:331 | 'one source batch and one target batch'; published abstract re-read ('two replicates, measured in different batches'). |
| REL-03 | fix | resolved | paper.tex:316 | July :142 locked; preprint backbone pending; 'the revision uses stock scVI' with SI-09 and SI-10. README at a755735 (2025-11-08) names scCraft as base model; bioRxiv v1 is 2025-11-13. |
| REL-04 | fix | resolved | paper.tex:315, :154 | Pending markers added on both lines (new mention at :169 is under INT-14 and RC-01). |
| REL-05 | fix | resolved | paper.tex:296, :397, :175 | 'eight of its 13 tasks'; gene-activity ATAC files (fingerprints JSON: large_atac_gene_activity.h5ad, small_atac_gene_activity.h5ad); scIB text re-read (13 tasks). |
| REL-06 | fix | resolved differently, acceptable | refs.bib:583-591 | @article with journal bioRxiv (DOI prints), note 'Version 3, posted 2026-05-05', year 2025. Title 'in' follows the bioRxiv record of v2 and v3 (API), against the Crossref title and the July manuscript ('for'); the lead checked the point the finding left open. Verified stamp stale (RC-08). |
| REL-07 | fix | resolved differently, acceptable | paper.tex:336; evidence.md:69 | Applied as one pending item; both searches recorded. Candidate descriptions re-read at abstract level (iDLC, Beaconet, Portal, AIF, ABC, IMGG, Li 2022); scStarCorrect checked by title only. |
| REL-08 | note | resolved | paper.tex:83, :363 | 'JS-type' in glossary and Methods; 'GAN' unexpanded at :83 (RC-04). |
| REL-09 | note | resolved | paper.tex:459, :628 | Eq. 9 pointer; gamma = 10 as this study's setting. Mescheder et al. Sec. 4.1 Eq. 9 read; no gamma value in the text. |
| REL-10 | note | resolved | paper.tex:603 | Cuturi and Doucet Alg. 2 (theta in [0,1] with line search or a preset value) and POT 0.9.7.post1 docstring (theta = 1) read; barycenter.py:8-10 not in the tag (RC-06). |
| REL-11 | note | resolved | paper.tex:335, :368; refs.bib:642 | genevay2018learning entry matches the PMLR page (pages 1608-1617, AISTATS 2018); Theorem 1 Eq. 4 and the abstract read; alignment.py:12-13 read. |
| REL-12 | note | resolved | paper.tex:443; refs.bib:343 | Cited once at the SI-74 rule; rule read in arXiv v3 Secs. 1, 3.1 and 3.2 (decision on P(rope), P(left), P(right) > 1 - alpha; 95% threshold named in Sec. 1). evidence.md entry stale (RC-07). |
| REL-13 | note | resolved | refs.bib:604, :593 | Consortium author added (Crossref: 55 authors, 26th = 'Open Problems Jamboree Members'; list compared); Leon spelled as in arjovsky2017wasserstein. Optional items (3)-(6) unchanged; 'ICLR 2017' venue not re-checked (OpenReview and DBLP challenge, OpenAlex lists arXiv only, Semantic Scholar 429). |
| REL-14 | note | resolved | paper.tex:325 | As proposed; PMC10056671 Sec. 3.6 read (Pancreas-subset, UMAP Fig. 11). |
| REL-15 | note | resolved | paper.tex:306 | As proposed; published abstract read. |
| REL-16 | note | resolved | paper.tex:298 | scIB-E moved to Benchmarks and removed from the adversarial list. The added classification 'two adversarial; four not' is not in the source (RC-02). |
| REL-17 | note | resolved | paper.tex:304-333 | Expansions as proposed (sysVI VampPrior expansion read in the sysVI text); ResPAN, uniPort not expanded (no expansion in the sources). First-use placement: RC-04. |

## 5. Deviations from the proposed text, judged on the evidence

- INT-01: the ANALYSIS_SPEC tag was not added. Correct: that section is an agent draft; the sentence is a deduction from locked facts (n_critic 5 versus 1; X6 steps vary for the pooled critic only) and agrees with the Limitations line (:562).
- INT-02: the gap line is locked to the evidence.md record and also pending a repeat, where the proposal made it pending only. Acceptable: the record exists and its counts match both reports (section 8).
- INT-03: the earlier comparison is attributed to the July submission (text read) and the preprint is cited only as its public version. Better than the proposal, which would have attributed the comparison to preprint text that nobody read. It needs the scCRAFT marker (RC-01).
- INT-06: the last clause says R0 'describes' whether the conditions arise instead of 'whether the discriminator gets close to optimal'. Correct: R0 measures separability and gradient norms (:233), not optimality.
- INT-15: applied with a different bullet structure; every fact verified, wording partly open (RC-04, RC-05).
- INT-19: the Maan line was left in place. Acceptable at abstract level; listed as abstract-only in evidence.md.
- REL-03: matches the proposal; the pending 'locks when' text adds evidence that was verified (README at a755735, 2025-11-08; bioRxiv v1 2025-11-13).
- REL-06: title 'in' follows the bioRxiv record of versions 2 and 3, against the Crossref title and the July manuscript ('for'). This settles the point the finding left open and is recorded in evidence.md. Year 2025 follows the DOI posting; stamps are stale (RC-08).
- REL-07: applied as one pending item. Acceptable; the eight descriptions agree with the abstracts (scStarCorrect by title only).
- REL-12: 'rule of' where the proposal had 'style of'. Acceptable: Corani et al. Sec. 3.2 step 4 states the three-way decision at 1 - alpha; the ROPE idea is credited there to Kruschke (not retrieved).
- REL-13: only the two required edits were made; the optional items remain open.

## 6. Changed-line audit (34 hunks of paper.tex)

| lines at d170dd5 | change | findings | result |
|---|---|---|---|
| :83 | glossary 'arm': JS-type wording | REL-08 | OK; GAN unexpanded (RC-04) |
| :119-120 | Goh softened; Yu wording | INT-18, INT-19 | OK |
| :127 | Goodfellow pointer | INT-08 | OK |
| :133-134 | mapping bullet; W1 wording | INT-15, INT-18 | OK; tag (RC-06) |
| :136 | Thms. 2.1-2.2 precision | INT-18 | OK |
| :138-145 | non-saturating loss, noise, scVI definition, scvi_adv, six arms and R4 flags, fool loss, assumptions, R0 | INT-06, 07, 12, 15, 17, 18 | OK; RC-03, RC-04, RC-05, RC-06 |
| :150 | Thm. 4.3 precondition and symbols | INT-05 | OK |
| :152-154 | hypothesis tied to X3; GLUE gloss; scCRAFT pending | INT-04, 14, 15 | OK |
| :162 | image-GAN inference pending | INT-12 | OK |
| :167-170 | scIB-E wording; gap scope; earlier comparison; prior W1 and reference anchoring | INT-02, 03, 20 | RC-01 (:169), RC-04 (:170) |
| :175-176 | Scope with both decoders; limit of P1 | INT-01, 13 | OK; tag (RC-06) |
| :180-183 | X3 design tag; W1 geometry pending; X6 citations; X7 references | INT-09, 10, 21 | OK |
| :185-186 | also-asked bullet; comparison and guidance split | INT-11, 22 | OK; tag (RC-06) |
| :195-198 | drop list: WDGRL/iMAP; over-alignment wording | INT-04, 23 | OK |
| :201-202 | drop list: Reviewer 2 concern 3 | INT-23 | OK |
| :223-225 | drop list: C03 | INT-25 | OK |
| :296 | eight of 13 scIB tasks | REL-05 | OK |
| :298 | scIB-E moved to Benchmarks | REL-16, 17 | RC-02 |
| :304-306 | scANVI, sysVI, trVAE | REL-15, 17 | OK |
| :311-313 | DANN, AD-AE, scDREAMER (scIB-E line removed) | REL-16, 17 | OK |
| :315-316 | scCRAFT pending; backbone of the original submission | REL-03, 04, 17 | OK |
| :321-322 | WDGRL, iMAP expansions | REL-17 | OK |
| :324-325 | IMAAE expansion; design-stage statement | REL-14, 17 | OK |
| :331-333 | MMD-ResNet; SAUCIE; BERMUDA | REL-02, 17 | OK |
| :335-336 | Genevay added; coverage candidates pending | REL-07, 11 | OK |
| :345 | Backbone narrowed | INT-24 | OK |
| :363 | discriminator: JS-type | REL-08 | OK |
| :368 | Sinkhorn: Genevay; eps = blur | REL-11 | OK |
| :380 | Methods: sim1 flag (lead addition) | lead | OK; verified against the record |
| :397 | gene-activity ATAC | REL-05 | OK |
| :443 | Corani rule | REL-12 | OK |
| :459 | R1 Eq. 9; gamma | REL-09 | OK |
| :603 | barycenter: Cuturi and Doucet Alg. 2 | REL-10 | OK; tag (RC-06) |
| :628 | R1 one-vs-rest form | REL-09 | OK |

## 7. Claims checked against the source text (36 rows)

| line | claim | source and location read | result |
|---|---|---|---|
| :119 | scRNA-seq: batch effects expected to remain relevant, new challenges for modelling them | Goh et al. 2022 abstract (Europe PMC, PMID 35282901): 'presents new challenges'; 'will not only remain relevant ... but will become even more important'; next sentence on BE modeling | supported at abstract level ('for modelling' glosses the next sentence) |
| :120 | uncorrected batch effects mislead; over-correction can hinder biomedical discovery | Yu et al. 2024 abstract (PMID 39363244) | supported verbatim in substance |
| :127 | optimal-discriminator objective = -log 4 + 2 JS; Sec. 4.1, Eq. 6, in the proof of Thm. 1 | Goodfellow et al. 2014 (arXiv 1406.2661v1) Sec. 4.1: Prop. 1, Thm. 1, Eqs. 5-6 | supported; pointer exact |
| :134 | W1 continuous / a.e. differentiable; false for JS and KL | Arjovsky et al. 2017 (PMLR v70) Thm. 1 parts 1-3 | supported |
| :136 | Thms. 2.1-2.2: compact disjoint supports / closed non-aligned manifolds; gradient zero everywhere / a.e. | Arjovsky and Bottou 2017 (arXiv v1) Thms. 2.1, 2.2 | supported |
| :138 | Thm. 2.5 gradient KL - 2 JS; Thm. 2.6 white-noise Cauchy gradient under Thms. 2.1 or 2.2; 'too strong'; Sec. 2.2.2 | same, Sec. 2.2.2, Thms. 2.5-2.6 and the remark after Thm. 2.6 | supported; Thm. 2.5 precondition (densities) omitted: RC-03 |
| :139 | Thm. 3.1 density; Cor. 3.1 Gaussian noise gives positive density; scVI noise scale cell-specific | same, Sec. 3; scvi-tools 1.4.2 nn/_base_components.py:301-305 (q_v from the encoder, rsample) | supported |
| :140-141 | scVI Gaussian posterior; scvi_adv given posterior samples; manifest adv_input 'mean' for 40 X1 rows | scvi-tools 1.4.2 wheel: module/_vae.py:390, :413; nn/_base_components.py:304-305; train/_trainingplans.py:689; scvi_adversarial_plan.py:348-351, :359; manifest (recheck_checks_results.json) | supported |
| :142 | two flagged blocks (0.0188, 0.0069), conditioned decoder; setters barycenter 5, mmd 1, sinkhorn 0; A2/A3 = atac_small, immune | docs/prereg/r4_a1_record.json (git_sha c88cce3, status frozen), computed in recheck_checks.py; manifest A2/A3 rows; SI-56, SI-66; PREREG.md:462-472 | supported; readability: RC-05 |
| :143-144 | fool loss for k = 2 is the label-flipped non-saturating loss; assumptions of WGAN Thm. 1, Thms. 2.4-2.6 | scvi_adversarial_plan.py:292-298; WGAN Thm. 1 ('fixed distribution'); Arjovsky and Bottou Thms. 2.4-2.6 ('P_r the real data distribution'; \|\|D - D*\|\| < eps) | supported |
| :150 | Thm. 4.3 bound with precondition and symbols | Zhao et al. 2019 (PMLR v97) Lemma 4.8, Thm. 4.3 and Remark | supported |
| :153 | adversarial cVAE and GLUE mixed unrelated cell types; GLUE gloss | Hrovatin et al. 2025 (PMC12577435): GLUE 'a well-established adversarial model'; mixing at higher strength | supported |
| :167 | scIB-E: six level-1 losses; no W1/OT/explicit MMD; HSIC measures dependence | Yi et al. 2025 (PMC12636214) Level-1 list; HSIC paragraph; 0 hits for Wasserstein, MMD, optimal transport, Sinkhorn | supported |
| :168 | gap: no comparison found; scope (2026-10-15, OpenAlex, arXiv, Europe PMC from 2017, citation tracking, web searches) | evidence.md:62-70; reports ff66652f section 7 and aed01ae8 section 8 | supported by the record; queries not listed (RC-07) |
| :169 | July submission: JS discriminator versus reference-based W1 critic on a fixed scCRAFT backbone, no MMD/OT arms | Topology_Matters.zip oup-authoring-template.tex:132, :134, :142; 0 hits for MMD or Sinkhorn in tex and supplement | supported; naming marker missing: RC-01 |
| :170 | WDGRL W1 critic; iMAP reference-anchored WGAN-GP; SAUCIE and MMD-ResNet MMD to a reference or target | iMAP full text (PMC7891139): 'one dataset is specified as the anchor', 'WGAN-GP'; MMD-ResNet abstract (PMID 28419223); WDGRL and SAUCIE: first-round record (evidence.md), not re-read | supported (WDGRL, SAUCIE not re-read) |
| :175-176 | two decoders; P1 at 5 versus 1 critic steps; X6 varies steps for pooled only | manifest: X1 n_critic by arm; X6 n_critic by arm; SI-01, SI-09, SI-11, SI-16, SI-70; glossary :82 | supported; tags: RC-06 |
| :180 | X3: same number of cells at every dose | manifest X3 n_cells per task; SI-41 | supported |
| :185 | X8 pending; local mixing SI-76; X12, X15 SI-05, SI-44 | CONSTRAINTS.md SI-05, SI-44, SI-76; PAPER_PLAN.md:155 | supported; SI-05 indirect (RC-06) |
| :195-202 | drop-list wording (WDGRL/iMAP; Reviewer 2 concern 3; sysVI) | decision letter lines 65-67; sysVI text; iMAP text | supported |
| :296, :397 | eight of 13 scIB tasks; gene-activity ATAC | scIB (PMC8748196): 13 tasks; docs/prepped_fingerprints_scib.json: 8 entries, atac source files gene activity | supported |
| :298 | scIB-E: six losses (two adversarial, four not), scANVI levels, Ray Tune on immune | Yi et al. 2025 full text and code repository (_RCE.py, _RBP.py) | six losses, scANVI levels, Ray Tune on the immune dataset supported; adversarial count not stated by the source: RC-02 |
| :305 | VampPrior = multimodal variational mixture of posteriors | sysVI text (PMC12577435) | supported |
| :306 | trVAE: conditional OOD generation, perturbation response, MMD in the decoder layer after the bottleneck | trVAE published abstract (PMID 33381839) | supported |
| :313, :321-333 | method-name expansions (AD-AE, scDREAMER, scANVI, WDGRL, iMAP, IMAAE, SAUCIE, BERMUDA, scCRAFT) | re-read this round: AD-AE, BERMUDA and scANVI abstracts, iMAP title, IMAAE abstract, scDREAMER (PMC10682386) and scCRAFT (PMC11971424) full texts; WDGRL title as in refs.bib; SAUCIE expansion not re-read (PMC10164410 not retrievable from Europe PMC) | supported except SAUCIE (not re-read); first-use placement: RC-04 |
| :316 | scCRAFT backbone of the original submission; preprint backbone pending; README names scCraft as base model on 2025-11-08 | July tex:142; git a755735 (2025-11-08) README line 1; bioRxiv API (v1 2025-11-13) | supported |
| :325 | IMAAE design-stage GAN-only model under-corrects with three or more batches (one dataset, UMAP) | IMAAE (PMC10056671) Sec. 3.6 and Fig. 11 legend | supported |
| :331 | MMD-ResNet: one source batch and one target batch | MMD-ResNet published abstract (two replicates measured in different batches) | supported |
| :335, :368 | Sinkhorn divergences interpolate OT and MMD; debiased divergence; eps = blur = 0.05 sqrt(2d) | Genevay et al. 2018 (PMLR v84) abstract and Thm. 1 Eq. 4; alignment.py:12-13, :50, :249 | supported |
| :336 | coverage candidates (iDLC, scStarCorrect, Beaconet, Portal, AIF, ABC, IMGG, Li et al. 2022) | abstracts via Europe PMC and Crossref | supported at abstract level; scStarCorrect by title only |
| :345 | SCVI defaults for arms, plain scVI and scANVI pretraining; sysVI same sizes, own Gaussian likelihood | fit_paper_config.py:393-396, :404-413; PAPER_PLAN.md:83; manifest scanvi and sysvi rows; SI-10, SI-63 | supported |
| :363 | discriminator JS-type; minimax objective of Goodfellow et al. stated for two distributions | Goodfellow et al. Sec. 2 objective for p_data and p_g | supported |
| :443 | region-of-practical-equivalence rule of Corani et al.; margin 0.01; 0.95 | Corani et al. (arXiv 1609.08905v3) Secs. 1, 3.1, 3.2; SI-74 | supported (published ML version not read) |
| :459, :628 | R1 penalty Eq. 9; gamma = 10 is this study's setting | Mescheder et al. 2018 (PMLR v80) Sec. 4.1, Eq. 9; SI-22; scvi_adversarial_plan.py:21-22 | supported |
| :603 | barycenter update = Cuturi and Doucet Alg. 2 with theta = 1, as in POT free_support_barycenter | Cuturi and Doucet 2014 (PMLR v32) Alg. 2; POT 0.9.7.post1 ot/lp/_barycenter_solvers.py:185-190; barycenter.py:8-10, :71-73 | supported; tag: RC-06 |
| refs.bib | genevay2018learning; ganin2016domain; luecken2025defining; arjovsky2017towards; reid2025topology; corani2017statistical | PMLR page meta tags; JMLR page and PDF; Crossref author list (55); bioRxiv API; Crossref | entries correct; stamps stale (RC-08) |

## 8. evidence.md, refs.bib, rendering, readability

- evidence.md search records: Introduction track (6 queries, up to 20 results per source and query, 194 hits, 175 unique, 59 rated: 45 none, 14 partial, 0 substantial, 54 forward-citation works, two web searches) and Related-work track (14 queries, 330 rows, 315 unique, 18 candidates read, known-item recall 9 of 16) match report ff66652f section 7 and report aed01ae8 section 8 number for number. Gaps: query strings missing; stale entries (RC-07). The 35 DOIs of the retraction check still equal the DOI count of refs.bib (35).
- refs.bib: genevay2018learning matches the PMLR page (title, authors, pages 1608-1617, AISTATS 2018); ganin2016domain 'Mario Marchand' matches the JMLR PDF; luecken2025defining has 55 authors in Crossref order (26th = 'Open Problems Jamboree Members'); arjovsky2017towards 'L{\'e}on'; corani2017statistical renamed, cited once; reid2025topology entry type, title (bioRxiv v2 and v3), note and year as recorded. Printed entries checked in the compiled bibliography. Stamps and comments: RC-08.
- Rendering: no stray markup in the changed lines; citations resolve; the reference list prints the DOI for the preprint.
- Readability for the stated reader: the new theory lines (:127, :136, :138, :150) define their symbols; open items are RC-04 (expansions) and RC-05 (:142).

## 9. Prior-work probe (screen only)

Scope: OpenAlex, arXiv and Europe PMC, publications since 2023, 4 new queries ('comparison of alignment losses adversarial MMD Sinkhorn single-cell batch integration'; 'Wasserstein GAN versus MMD versus discriminator latent alignment scRNA-seq variational autoencoder'; 'which divergence for batch correction single-cell deep generative model ablation of alignment objective'; 'optimal transport penalty versus adversarial classifier scVI batch effect removal benchmark'), 53 unique records (OpenAlex 33, Europe PMC 11, arXiv 9); the top 30 were triaged by a language model against the method card and I read the notes of the six rated partial or substantial. Result: only works already recorded (the authors' own preprint; scDREAMER; iDLC; the sysVI preprint) and two multi-omics items outside the question. No new overlap with the gap sentence; escalation not run; the two review searches (175 and 315 unique records) were not repeated. Verdict: no match in scope beyond the recorded works; not exhaustive.

## 10. Not re-checked, versions read, deviations

- Not re-checked: ICLR 2017 venue of arjovsky2017towards (OpenReview and DBLP challenge, OpenAlex lists only the arXiv record, Semantic Scholar rate-limited); theorem numbers are those of arXiv v1. Published Machine Learning version of Corani et al. (arXiv v3 read). bioRxiv full text of the preprint (not attempted; metadata and abstract level only). Maan et al. 2024, Goh et al. 2022 beyond the abstract (closed access); Maan abstract not re-read (unchanged line). WDGRL and SAUCIE claims and the SAUCIE expansion (unchanged first-round record). Kruschke (2013). Unchanged lines (Wu et al. Prop. 3.1, Lucic, Kurach, Hrovatin beyond the cited passage). shen2018wasserstein page range. scStarCorrect abstract (title only).
- No project rigor_ledger.json exists; entries are delivered as flat lists (rigor_ledger_entries_recheck_intro_related.json, recheck_status_42_findings.json), as in the first round. The ponytail skill is not in this profile's catalogue; the one script written (recheck_checks.py) is read-only.
- The PDF of the bundle was not read page by page; text of my own compile was compared with it (0 differing lines) and page 7 was viewed.

## 11. Files

- rigor_ledger_entries_recheck_intro_related.json: the 8 new findings.
- recheck_status_42_findings.json: status of INT-01 to INT-25 and REL-01 to REL-17 at d170dd5.
- recheck_checks.py, recheck_checks_results.json: machine checks (manifest, frozen record, citations, scCRAFT mentions, July text).

## 12. Claim receipt

```
CLAIM RECEIPT
claim: Re-check of paper/paper.tex at d170dd5: 42 earlier findings re-checked (34 resolved, 5 resolved differently, 3 partly resolved, 0 not resolved); 8 new findings (0 blocker, 2 fix, 6 note); counts read back from the saved ledger and status files; machine checks rerun from recheck_checks.py.
- readback: rigor_ledger_entries_recheck_intro_related.json {"expect_rows": 8} = [{"id": "RC-01", "severity": "fix", "status": "open", "track": "recheck_intro... | /home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/5f557877-da1f-44ed-8e1f-243b2fc35208/work/out/rigor_ledger_entries_recheck_intro_related.json sha256:9493eb28b3ae @2026-10-15T15:14:10Z
- readback: recheck_status_42_findings.json {"expect_rows": 42} = [{"id": "INT-01", "severity": "fix", "status": "resolved", "where": "paper.te... | /home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/5f557877-da1f-44ed-8e1f-243b2fc35208/work/out/recheck_status_42_findings.json sha256:9ef03f384f29 @2026-10-15T15:14:10Z
- readback: recheck_checks_results.json {"key": "bib_entries"} = 52 | /home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/5f557877-da1f-44ed-8e1f-243b2fc35208/work/out/recheck_checks_results.json sha256:1a3f0895685f @2026-10-15T15:10:14Z
- readback: recheck_checks_results.json {"key": "cited_keys"} = 52 | /home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/5f557877-da1f-44ed-8e1f-243b2fc35208/work/out/recheck_checks_results.json sha256:1a3f0895685f @2026-10-15T15:10:14Z
- readback: recheck_checks_results.json {"key": "x1_scvi_adv_rows"} = 40 | /home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/5f557877-da1f-44ed-8e1f-243b2fc35208/work/out/recheck_checks_results.json sha256:1a3f0895685f @2026-10-15T15:10:14Z
- readback: recheck_checks_results.json {"key": "july_mmd_sinkhorn_hits"} = 0 | /home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/5f557877-da1f-44ed-8e1f-243b2fc35208/work/out/recheck_checks_results.json sha256:1a3f0895685f @2026-10-15T15:10:14Z
- reconcile: counts agree: {'total': 8, 'sum(parts)': 8} | in-memory
- reconcile: counts agree: {'total': 42, 'sum(parts)': 42} | in-memory
not checked: sources and claims listed in section 10 of the report (ICLR venue, preprint full text, published Corani version, Maan and Goh beyond the abstract, WDGRL and SAUCIE claims, SAUCIE expansion, scStarCorrect abstract)
```
