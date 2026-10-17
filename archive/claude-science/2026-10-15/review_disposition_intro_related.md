# Disposition of the rigour-review findings of 2026-10-15 (commit add2f73 -> d170dd5)

Reports: Introduction track ff66652f-c92d-4b19-8db7-ec52871d1937; Related-work track aed01ae8-eb07-447b-a137-aaa3acee8e81.

| finding | severity | disposition | what was done |
|---|---|---|---|
| INT-01 | fix | applied with changes | 'budget-matched' dropped; gap line states the search scope; new 'Limit of the primary contrast' bullet tagged SI-01 (no 10-step discriminator) and manifest (ANALYSIS_SPEC section 16 not cited: agent draft) |
| INT-02 | fix | applied with changes | gap line locked to paper/evidence.md 'Gap search' (both review searches recorded there) with \pending{search repeated}{before submission}; 'no first claim' made a pending user decision; WDGRL, iMAP, SAUCIE, MMD-ResNet cited |
| INT-03 | fix | applied with changes | earlier comparison attributed to the July submission (oup-authoring-template.tex:134, :142, read); the preprint is cited only as the public preprint of that work (its text not readable), replaced per SI-07; disclosure wording pending |
| INT-04 | fix | applied | line restated as a hypothesis tested by X3; drop-list wording corrected |
| INT-05 | fix | applied | precondition and symbols added; wording compressed |
| INT-06 | fix | applied with changes | last clause reads 'whether their conditions arise here is what the exploratory premise check R0 describes' (R0 measures separability and gradient norms, not optimality) |
| INT-07 | fix | applied | compressed; Thm 2.5 content and the Thm 2.6 noise model stated |
| INT-08 | fix | applied | pointer Sec. 4.1, Eq. 6, in the proof of Thm 1 |
| INT-09 | fix | applied | X7: D-06 and I089 |
| INT-10 | fix | applied | attribution to W1 geometry versus latent scale pending |
| INT-11 | fix | applied | guidance clause split off as pending |
| INT-12 | fix | applied | fool-loss clause sourced to scvi_adversarial_plan.py:297-298; image-GAN inference marked pending |
| INT-13 | fix | applied | both decoders named in Scope (SI-09, SI-11, SI-16, SI-70) |
| INT-14 | fix | applied | Introduction scCRAFT line pending the naming decision (same as REL-04) |
| INT-15 | fix | applied with changes | mapping bullet tagged scvi_adversarial_plan.py:346-360; posterior mean and sample defined; R-11 line rewritten in plain words with both flagged blocks (verified in docs/prereg/r4_a1_record.json: immune|1 0.0188, sim1|1 0.0069; setters barycenter x5, mmd x1) and tagged SI-56, SI-66; GLUE glossed as 'a graph-linked embedding method'; reviewer remark moved into the source tag; scVI expanded at first use |
| INT-16 | fix | applied | Mario Marchand |
| INT-17 | note | applied | scvi-tools _vae.py:390, :413 added (verified in the 1.4.2 wheel); manifest adv_input 'mean' for the 40 scvi_adv X1 rows stated (verified) |
| INT-18 | note | applied | four lines; the cell-specific posterior scale is tagged to _base_components.py:304-305 |
| INT-19 | note | applied in part | Goh line softened; Maan line unchanged (within its abstract), listed as abstract-only in evidence.md |
| INT-20 | note | applied | HSIC named; 'no W1, OT or explicit MMD term between batch distributions' |
| INT-21 | note | applied | X3 design tagged; X6 citations added as bracketed citations |
| INT-22 | note | applied | 'Also asked' bullet: X8 (pending), local mixing (SI-76), X12/X15 (SI-05, SI-44; analyses pending) |
| INT-23 | note | applied | drop-list reasons reworded |
| INT-24 | note | applied | Methods Backbone narrowed: SCVI defaults for every arm, plain scVI and scANVI's scVI pretraining (fit_paper_config.py:393-394); sysVI same sizes with its own Gaussian likelihood (fit_paper_config.py:404-413; SI-63) |
| INT-25 | note | applied | C03 line added to the drop list |
| REL-01 | fix | applied | as INT-16 |
| REL-02 | fix | applied | one source batch and one target batch |
| REL-03 | fix | applied with changes | backbone of the original submission locked (July text :142); preprint backbone pending (bioRxiv refused access again on 2026-10-15, HTTP 429); 'the revision uses stock scVI' tagged SI-09, SI-10 |
| REL-04 | fix | applied | Related work scCRAFT line pending the naming decision |
| REL-05 | fix | applied | 'eight of its 13 tasks'; Methods Data: gene-activity versions of the ATAC tasks; Introduction Scope says eight scIB tasks |
| REL-06 | fix | applied with changes | @article, journal bioRxiv (DOI prints); title 'in' as on bioRxiv versions 2-3 (Crossref deposit says 'for'; noted in evidence.md); note 'Version 3, posted 2026-05-05'; year 2025 (DOI posting) |
| REL-07 | fix | applied as pending | no method lines added; the candidates are one \pending item at the end of Related work; both searches recorded in evidence.md |
| REL-08 | note | applied with changes | glossary 'JS-type, after the Jensen--Shannon (JS) form of the original GAN objective'; Methods 'JS-type; the minimax objective of Goodfellow et al., stated there for two distributions' |
| REL-09 | note | applied | Eq. 9 pointer; gamma = 10 as this study's setting; one-vs-rest form in the supplement |
| REL-10 | note | applied | barycenter update: Cuturi and Doucet Algorithm 2 with theta = 1, as implemented in POT (verified in src/wcd_vae/wcd/barycenter.py:8-10) |
| REL-11 | note | applied | genevay2018learning added (lead verified PMLR page and PDF p3, Theorem 1, Eq. 4); also cited in the Related work Sinkhorn line; 'entropic regularisation epsilon = blur' (alignment.py:12-13) |
| REL-12 | note | applied | cited at the SI-74 rule; key renamed corani2017statistical |
| REL-13 | note | applied in part | (1) consortium author and (2) L{\'e}on added; optional items (3)-(6) not changed |
| REL-14 | note | applied | design-stage GAN-only model, one dataset, judged from UMAP plots |
| REL-15 | note | applied | trVAE purpose stated |
| REL-16 | note | applied | scIB-E moved to Benchmarks with two adversarial and four non-adversarial losses |
| REL-17 | note | applied | acronyms spelled out (ResPAN, uniPort not expanded: no expansion found in the sources) |

Also changed (lead, not from a finding): Methods 'Adversary input and standardisation' names the sim1 flag (0.0069; not an A2/A3 task), for consistency with the Introduction.

## Focused re-check of d170dd5 (report 52953dc3-fdfb-472a-a13f-155e7d0ff897): 0 blockers, 2 fixes, 6 notes; applied in d556bb1

| finding | severity | disposition | what was done |
|---|---|---|---|
| RC-01 | fix | applied | scCRAFT naming pending added to the 'Earlier comparison' line; 'without MMD or Sinkhorn arms' |
| RC-02 | fix | applied | scIB-E line lists the six losses without classifying them |
| RC-03 | note | applied | 'for distributions with densities' added to Thm 2.5 |
| RC-04 | note | applied | GAN expanded in the glossary; scIB expanded at first use in the Introduction; method acronym tags dropped from the anchoring line (citations kept); scVI used only after its expansion |
| RC-05 | note | applied | adversary-input bullet split into three, labels given their plain meaning |
| RC-06 | note | applied | five source tags narrowed or extended as proposed (scvi_adversarial_plan.py:433-439 checked: encoder update with the adversary term) |
| RC-07 | note | applied | evidence.md: line 4, corani entries, query strings of both review searches, shaham and trVAE notes |
| RC-08 | note | applied | refs.bib: verified stamps for reid2025topology and ganin2016domain, header comment; evidence.md: year note for reid2025topology |

Status of the 42 first-round findings at d170dd5 (re-check): 34 resolved, 5 resolved differently but acceptably (INT-02, INT-03, INT-06, REL-06, REL-07), 3 partly resolved (INT-14 and INT-15, completed by RC-01 and RC-05; INT-19, Maan line kept within its abstract).
