# Related work cut and reference reduction, 2026-10-19

Basis: `paper/paper.tex` at commit 82d7090 (old) against the working copy (new). User decision 2026-10-19: keep a short Related work section in the main text that names the eight methods asked for on 2026-10-15; cut other references so that the main text cites at most 50 keys. Edited: `paper/paper.tex` only (`paper/refs.bib` unchanged: no entry deleted). Roles read from `paper/BUDGET.md` section 3 and `paper/evidence.md`. Not built, not committed.

## 1. Counts (python3 script, regex over `\cite*` commands; main text = everything before `\section*{Supplementary material}`)

| Quantity | Old (82d7090) | New |
|---|---|---|
| Distinct keys cited in the main text | 59 | **49** |
| Distinct keys cited in the whole file | 59 | 59 |
| Entries in refs.bib | 59 | 59 (all still cited somewhere) |
| Keys cited in refs but missing from refs.bib | 0 | 0 |
| Bullet items in the main Related work | 36 (`\item` count, old section) | 12 |

Spare slots under the limit of 50: 1.

## 2. Keys removed from the main text (10)

All ten are supplement-only now (cited only in S-R). Whether the journal counts supplement references toward 50 is not known (BUDGET.md section 3.1 flag).

| Key | Old line | Claim | Moved to |
|---|---|---|---|
| ritchie2015limma | 267 | limma: gene-wise linear models with batch terms | S-R, Statistical and geometric integration, unchanged |
| johnson2007adjusting | 268 | ComBat: empirical Bayes batch adjustment | S-R, same |
| haghverdi2018batch | 269 | MNN: mutual nearest neighbours between batches | S-R, same |
| stuart2019comprehensive | 270 | Seurat: anchors between datasets | S-R, same |
| tran2020benchmark | 277 | 14 methods, five scenarios | S-R, Benchmarks, unchanged |
| luecken2025defining | 279 | Open Problems: benchmarks disagree | S-R, same |
| lotfollahi2020conditional | 288 | trVAE | S-R, Deep generative models, unchanged |
| shree2023scdreamer | 295 | scDREAMER | S-R, Adversarial (JS-type) integration, unchanged |
| wang2019bermuda | 319 | BERMUDA | S-R, Kernel and optimal-transport alignment, unchanged |
| cao2022unified | 320 | uniPort | S-R, same |

Related-work items removed whose keys stay cited in the main text elsewhere (so not supplement-only): DANN (ganin2016domain, Introduction l.122) and AD-AE (dincer2020adversarial, l.122), both to S-R; WDGRL (shen2018wasserstein), iMAP (wang2021imap), MMD-ResNet (shaham2017removal), SAUCIE (amodio2019exploring), all cited in the Gap paragraph (l.166), to S-R. Sixteen items in total moved to S-R, wording unchanged.

### Keys on the BUDGET.md 3.1 list that I did NOT cut

| Key | Why kept |
|---|---|
| wang2022respan, wang2023integrating | Put back (BUDGET.md section 3.1 "optional put-back"): one bullet naming other WGAN-GP-type integrators, in the Wasserstein critics group; brings the count to 49, inside the 48-50 target |
| cao2022multi | Kept in the Introduction (l.151, GLUE parenthetical). Task condition not met: `evidence.md` line 24 records the sysVI paper's result for "adversarial alignment (cVAE+ADV, GLUE)" together, so it does not show that the sentence about the adversarial conditional VAE alone, with "more so at higher correction strength", holds for that model alone. I could not open the paper (network call refused). Also `evidence.md` line 121 and `review_disposition_coverage.md` (CV-13) record a user decision to cite GLUE where the Introduction names it |
| leek2010tackling, goh2022batch | Not cut: 10 cuts already meet the limit; BUDGET.md lists them as first to go only if a slot is needed |

## 3. Kept Related work items (12 bullets, 5 paragraphs)

The eight requested methods are in bold. "Trimmed" lists words dropped from the old line; no claim was reworded. The full old lines are in commit 82d7090, lines 263-323.

| # | Paragraph | Item (key) | Trimmed from the old line |
|---|---|---|---|
| 1 | Baselines of this study | Harmony (korsunsky2019fast); Scanorama (hie2019efficient) | none; two bullets merged |
| 2 | Baselines | scVI (lopez2018deep, gayoso2022python); scANVI (xu2021probabilistic); sysVI (hrovatin2025integrating) | scANVI expansion "(single-cell annotation using variational inference)"; "multimodal" in the VampPrior gloss; three bullets merged |
| 3 | Classifier and GAN discriminators | scvi-tools' adversarial classifier (arm scvi_adv; \src only) | none |
| 4 | same | **AIF** (monnier2024novel); **ABC** (danino2024batch) | AIF: "on gene expression", "cross-entropy", "then", "(the study used the most represented batch)"; ABC: "with categorical cross-entropy", "to retain biological signal" |
| 5 | same | **scStarCorrect** (gu2025scstarcorrect); **Portal** (zhao2022adversarial) | none; both \pending full-text markers kept word for word |
| 6 | Wasserstein critics | **Beaconet** (xu2024beaconet) | "on log-scaled gene expression", "weight 10"; relative clause turned into a parenthesis |
| 7 | same | **IMGG** (wang2022imgg) | "on highly variable genes"; "which the authors added because the adversarial loss alone made integrating several batches difficult" |
| 8 | same | **iDLC** (jiang2026geometric) | "whose latent space is split" to "with a latent space split"; "the expression profiles of each source batch" to "each source batch" |
| 9 | same | ResPAN (wang2022respan); IMAAE (wang2023integrating) | IMAAE expansion; two bullets merged |
| 10 | Kernel and optimal-transport alignment | MMD (gretton2012kernel); entropy-regularised OT, Sinkhorn divergences, barycenters, POT (cuturi2013sinkhorn, genevay2018learning, feydy2019interpolating, cuturi2014fast, flamary2021pot) | none; two bullets merged |
| 11 | Controlled comparisons | scIB (luecken2022benchmarking, with its \src); scIB-E (yi2025benchmarking) | scIB: "68 method and preprocessing combinations"; scIB-E: the list of six losses (pointer to the Introduction, Gap paragraph, which lists them), "(automated hyperparameter search)" |
| 12 | same | **Li et al. 2022, Pareto front on scVI** (li2022trade) | "maximum mean discrepancy" written as "MMD" (spelled out in the Introduction) |

Placement of scStarCorrect and Portal under the discriminator group is unchanged from the old outline; their pending notes say to move scStarCorrect to Wasserstein critics if the full text shows a WGAN-GP loss.

## 4. Other edits

- New `\subsection*{S-R Further related work}` placed before S-M (after the supplement's opening note). It names the ten supplement-only keys in its opening sentence.
- Nothing else in paper.tex was changed (diff hunks: Related work, S-R only). Introduction, Methods and the other supplement sections are untouched. Every kept main-text claim still carries its citation (script check: none of the ten removed keys appears before the supplement).

## 5. Unresolved

1. Length: the main Related work is 512 words by a regex count (headings, citation commands, \src and \pending text excluded; the task asked for about 250-300). I did not count the eight methods or the baselines separately. Further cuts would trim the eight descriptions or move the baseline lines to Methods, where Harmony, Scanorama, scVI, scANVI and sysVI are already cited.
2. Reference margin: 49 leaves 1 slot. BUDGET.md section 5 expects new citations for the pending items (X10 classifier, R0 statistic, Holm and Wilcoxon tests, scanpy; number unknown). First candidates to free slots, with their claims: wang2022respan and wang2023integrating (item 9, to S-R: 47), cao2022multi (GLUE parenthetical, if the sysVI check below is done: 46), leek2010tackling and goh2022batch (Introduction l.114-115), corani2017statistical and clopper1934use (Methods l.429-430; clopper also in S-M16).
3. cao2022multi: the Introduction sentence at l.151 needs a check against the sysVI full text (does the higher-correction-strength result apply to the adversarial conditional VAE separately from GLUE?) before the GLUE parenthetical can go. I did not do this check.
4. The supplement rule (own reference list allowed, counts toward 50 or not) is still unverified; ten keys are supplement-only.
5. `paper/BUDGET.md` sections 1 and 3 (Related work: 0 words; 19 keys leave) no longer match the decision; not edited (out of scope of this task).
6. Not done (outside the allowed files): CHANGELOG entry and notebook (NB) entry for this change; nb_check and outline_check were not run.
