# Main-text / supplement budget for an 8-page Bioinformatics Advances article

Status: plan only. `paper/paper.tex` is not edited. Every line number below is a line of `paper/paper.tex` at commit bf6cbdd (868 lines), read on 2026-10-19. No result is used or assumed anywhere in this file.

## 0. Plain meanings of the labels used here

| Label | Plain meaning (source line in paper.tex) |
|---|---|
| R0 | Premise check, exploratory: are batches nearly separable before integration, and does the discriminator's gradient vanish early while the critics' persist? No confirmed or refuted claim (226-231, 453). |
| R1 | Biology kept at the same amount of batch removal, and whether batch removal is more local (233-241). |
| R3 | Range of penalty strengths that work well, and whether a critic-discriminator difference is due to the divergence or to regularisation and adversary steps (243-249). |
| R5 | Where the adversarial arms sit relative to plain scVI, Harmony, Scanorama, sysVI and scANVI at equal numbers of tries (251-257). |
| P1 | Primary endpoint 1: biology score C at the matched batch score b*, reference critic minus discriminator, one result per decoder (67, 423-425). |
| Trade-off area | Co-primary endpoint: area in the batch-score / biology-score plane between plain scVI and the arm's curve (427, S-M14, 637-642). |
| P2 | Primary endpoint 2: how often runs collapse (failed, or C at least 0.10 below the same seed's plain scVI) (74, 430). |
| b* | Matched batch-removal target: half-way from plain scVI to the batch score every arm reaches (72). |
| X1 | Main benchmark: 8 tasks, both decoders, seeds 0-4, 3,000 fits (61). |
| X6 | Follow-up on regularisation and adversary steps (62, 445). |
| X13 | Standard integrators: Harmony, Scanorama, PCA, scANVI, sysVI (64). |
| X10 | Exploratory label-transfer check at b*, no new fits (63, 456). |
| S = 0.4B + 0.6C | Selection score used for strength ranges and placement (432). |
| Decoder, conditioned / unconditioned | The decoder does / does not receive the batch label (78). |
| JS, W1, MMD | Jensen-Shannon discriminator; Wasserstein-1 critic; maximum mean discrepancy. |

## 1. Page budget

Assumption (mine, not verified): 900 words of running text per page of the OUP two-column layout. It is the middle of the task's rule of thumb (5,000-6,000 words over about 6 text pages). The real figure needs a build in the OUP template; the repository has a July template at `manuscript_tm/oup-authoring-template/` that I did not build or check against the journal's current one. The task's 5,000-6,000 words appear to leave out tables and references; I counted both, so the text target is lower (about 4,100 words plus captions).

| Section | Target words | Pages (est.) | Stays in the main text (outline items, paper.tex lines) | Moves to the supplement |
|---|---|---|---|---|
| Abstract | 200 (limit) | 0.2 | Skeleton in section 4 (outline 98-108) | Provenance tags |
| Introduction | 700 | 0.8 | Batch effects, 2-3 sentences (112-118). Adversarial integration (120-125). JS vs W1: reading of the theory (129), W1 continuity and the parallel-lines example (130-131), disjoint supports and the saturating loss, one sentence each (132-134), noise (135), assumptions do not transfer and R0 tests them (142-143). Composition shift (148-152). Loss vs tuning (157-159). Gap (164-166), compressed. Contribution, scope, limit of P1, two parts (171-179). The four research questions R0, R1, R3, R5 as one paragraph of short questions (228, 235, 245, 253); the status lines, R2 and R4 removal notes and SI numbers (222-224) do not appear in the paper | S-M8 and S-M11 take 136-140 (posterior mean vs sample, scvi_adv input, rules R2/R3, rule-R4 flags, Sinkhorn pilot fits). 141 goes to Table 1 and S-M2. Theorem-by-theorem detail of 132-134 goes to a new S-I (about 250 words). The claims-to-drop block (182-220) is a writer's guard, never printed |
| Related work | 0 | 0 | No section. The nearest designs are already in the Gap paragraph (164-166) | Whole section 263-323 becomes S-R (about 700 words, one line per method, as now) |
| Methods | 1,150 | 1.3 | Backbone and decoders (331-332, 334-335, 341-342): 130 words. Arms: Table 1 plus 200 words (349-355, 357). Adversary input: defaults only (363, 366 first clause): 60. Training (372, 374-375): 90. Data and evaluation (383-386, 395-397, 401; tasks table is in S-D1): 180. Pre-registered design: stage order, seeds, rules R1 and R4, low-signal rule, code tag (407-413, 415 first sentence): 150. Statistical analysis (422-432, 433 first clause, 437, 438): 200. X6 and R0/X10 (445-446, 453-456): 70. X13 (462-467, 469): 60. Reproducibility pointer (475, 479): 10 | S-M1-S-M18 already exist (558-668). Add: 356, 373, 376-377 to S-M3, S-M6; 387-389 to S-D1; 398-400 and 414-416 (freeze, hashes) to S-X; 434-436 (approved proposals, "examined", scaling checks) to S-X; 447 to S-X; 463-465 grids to S-E2; 472-483 to a new S-E7 (compute, identity checks) |
| Results | 1,350 | 1.5 | R0: 150 words (489-494). R1: 450 (497-503; X10 as one sentence). R3: 400 (506-511). R5: 350 (514-519). Text is written after the single analysis run; each subsection states endpoint, rule and figure, then the result | Per-task detail, sim1 and sim2, sensitivity analyses (S-M13-S-M17), per-metric curves, X10 table |
| Discussion | 650 | 0.7 | Interpretation, 400 words (524-526). Limitations, 250 words, in one paragraph: 531-533, 536-540, 542 | Full limitation list as S-L: 534-535, 541, 543-544 and the long forms of the rest. Overlaps S-X items 2 and 7 (851, 856) |
| Availability statement | 50 | 0.05 | Repository, tag, commit (107, 415) | Checksums S-D2 (690-707) |
| Figures (6) | captions about 420 | 1.8 | Section 2 | S-F figures |
| Tables (3) | notes about 100 | 0.6 | Section 2 | S-T tables |
| References | <= 50 keys; plan 40 | 0.85 | Section 3 | S-R references (see flag) |
| Total | about 4,100 words body and abstract, plus captions and notes | about 7.8 | | |

Not settled: whether the 8 pages include references, tables and the availability and declaration statements (funding, conflicts); I read nothing from the journal beyond line 35 of the outline. The estimate of 7.8 pages has a margin of about 0.2 page and an uncertainty I put at plus or minus 0.7 page.

Cut ladder, in order, if the template build exceeds 8 pages (rules fixed now, before results):
1. Figure 2 (R0) to the supplement; R0 keeps 2 sentences and a pointer.
2. Figure 1 panel (a) (architecture) to the supplement; keep panel (b).
3. Table 3 (single-strength placement) to the supplement; Figure 6 stays.
4. Limitations to 150 words; the rest to S-L.

## 2. Figure and table plan (outcome-neutral)

Rule for all figures: the tasks, arms, decoders and panels are fixed here, before any X1 or X13 score is read. No example task is picked from results. Every arm appears in every panel where it has data. Low-signal blocks (b_common minus b0 below 0.02; defined at 73) are drawn in a fixed way (hollow marker, labelled) and never pooled. sim1 and sim2 go to S-F (line 499 and 389: simulations are reported per task, not pooled). Layout is tested with placeholder data only.

| Fig | Panels (plot type) | Shows | Data source | Needed? |
|---|---|---|---|---|
| 1 | (a) Schematic: scVI encoder and decoder, decoder with or without batch label, swappable alignment term (discriminator, W1 critic, MMD, Sinkhorn). (b) Cartoon, not data: how b* is read: plain-scVI point, one curve of biology C against batch score B, vertical line at b*, reading of C at the crossing | Design; what "matched batch removal" means (all questions) | None (drawing; use the FIGURE_EDITOR subagent and the bio-diagram skill; icons only from licensed sources) | Yes. b* is hard to follow from text. First cut if space is short: panel (a) |
| 2 | (a) Dot plot of the pre-integration batch-separability statistic, one dot per task (statistic and "early" still pending, 455). (b) Line plot of adv_grad_norm against epoch, one line per arm, median with a band across seeds, one panel per decoder at fixed, pre-declared strengths. (c) Line plot of discriminator accuracy (adv_acc) against epoch | R0 (exploratory, no claim) | X1 latents: per-epoch histories and lambda = 0 latents (453-454, 491); no new fits | Required by the task as the R0 slot. Single column, smallest figure; first to move to the supplement (ladder step 1). Strengths for (b) must be named in the R0 analysis script before results |
| 3 | Small multiples, rows = 2 decoders, columns = the 6 real tasks. Each panel: biology C (y) against batch score B (x), one line per arm through the 6 registered strengths, plain scVI as a point at (b0, c0), vertical line at b*; scvi_adv as a point (conditioned decoder only); Harmony, Scanorama, sysVI, scANVI as reference points | R1 frontier: trade-off area and where P1 is read | X1 (3,000 fits) and X13 (408 rows) | Yes. Legibility of 12 panels is the risk: if layout with placeholder data fails, move the unconditioned row to S-F, a rule fixed now |
| 4 | Four forest-style panels with the same layout (one row per real task, then the pooled estimate with its interval; two colours for the two decoders): (a) P1, reference minus discriminator; (b) trade-off area; (c) P2 collapse rate difference; (d) local-minus-global difference at the P1 crossing | R1 endpoints P1, area, P2, local vs global | X1, one analysis run (`analysis_x1.py`) | Yes. The only figure showing all primary endpoints |
| 5 | (a) Window width (decades of penalty strength within the confidence bound of the best selection score S), one dot per task, grouped by arm, two panels by decoder. (b) Dot-and-interval plot, one row per X6 arm (discriminator plain, spectrally normalised, R1 penalty, spectrally normalised with the critic's optimiser; pooled critic at 1, 5, 10 steps; spectrally normalised pooled critic at 1 and 5), biology at the matched strength minus the plain discriminator, one panel per decoder, the confirmatory contrast marked | R3: range of good strengths (X1) and divergence vs regularisation vs steps (X6) | (a) X1; (b) X6, 486 rows, seeds 10-12 (445). Analysis pending (446) | Yes for (b), the explanation half of the paper. (a) could become a table of widths if (b) needs the room |
| 6 | (a) Line plot of expected best S against k tries (k = 1..6), one line per method family, one panel per decoder; scANVI in its own stratum. (b) Scatter of B against C for each method at its single transfer strength, one point per real task, colour by family, plain scVI and PCA marked | R5 placement | X1 and X13, rules S-M18 (662-667) | Yes |

Tables (up to 6 allowed; I plan 3 in the main text):

| Table | Content | Replaces | Source |
|---|---|---|---|
| 1 | Methods compared: arm, alignment term, network, adversary steps per step, six strengths per decoder (S-E4, 785-803; Sinkhorn grids pending), plus the X6 variant arms | Prose at 349-356 and 445 | Manifest v5, `r1_record.json` |
| 2 | The six primary results (P1, area, P2, each per decoder): pooled estimate, 95% interval, three probabilities, call, number of evaluable tasks | Prose in Results R1 | `analysis_x1.py` |
| 3 | Single-strength placement: S per method and decoder, per task (strength chosen leave-one-family-out) | Prose in Results R5 | X1, X13 |

In the supplement: tasks table (S-D1, 672-688), per-task frontiers for sim1 and sim2, per-metric strength curves with seed bands (PAPER_PLAN section 3 figure 4, line 107), X10 label-transfer table, sensitivity analyses, all-method tables.

Left out of PAPER_PLAN section 3 (lines 103-112): figures 5, 6, 7 (batch-count scaling, reference coverage, composition dose-response; R2, R4, X3 removed), figure 9 (X10 only as supplement), figure 8 (only the X6 part survives, in Figure 5b).

## 3. Reference cut: 59 keys cited, limit 50

I counted 59 distinct keys in cite commands (grep of `\cite` on paper.tex), and `refs.bib` has 59 entries, each cited. All are cited between lines 110 and 483; Results and Discussion cite none. I did not re-check any citation against its source here; `paper/evidence.md` records those checks. "Role" below is how the main text uses the key.

Proposal: 19 keys leave (59 to 40), leaving 10 free slots. Reason for the margin: the pending items will need citations not yet in `refs.bib` (the X10 classifier, the R0 separability statistic, Holm and Wilcoxon tests, scanpy), and I do not know which. Reserve 6, leaving 4 optional put-backs.

### 3.1 Keys that leave the main list (19)

| Key | Cited at | Role | Why safe |
|---|---|---|---|
| ritchie2015limma | 267 | drop from main | Related-work line only; line moves to S-R |
| johnson2007adjusting | 268 | drop from main | same |
| haghverdi2018batch | 269 | drop from main | same |
| stuart2019comprehensive | 270 | drop from main | same |
| tran2020benchmark | 277 | drop from main | same |
| luecken2025defining | 279 | drop from main | same |
| lotfollahi2020conditional | 288 | drop from main | same |
| shree2023scdreamer | 295 | drop from main | same |
| monnier2024novel | 297 | drop from main | same |
| danino2024batch | 298 | drop from main | same |
| gu2025scstarcorrect | 299 | drop from main | same. Also has an open full-text check (299, pending); it no longer blocks the main list |
| zhao2022adversarial | 300 | drop from main | same; open full-text check at 300 |
| wang2022respan | 307 | drop from main | same |
| wang2023integrating | 308 | drop from main | same |
| xu2024beaconet | 309 | drop from main | same |
| wang2022imgg | 310 | drop from main | same |
| wang2019bermuda | 319 | drop from main | same |
| cao2022unified | 320 | drop from main | same |
| cao2022multi | 151 | drop from main | Cited only in the GLUE parenthetical. The parenthetical (151, "GLUE ... \citealp{cao2022multi}") is deleted from the main text and the claim of 151 is cited to hrovatin2025integrating alone. Check that sentence against hrovatin2025integrating before wording it; I did not re-read that paper |

Every one of the 19 sits in a sentence that moves to the supplement with it (lines 267-270, 277, 279, 288, 295-300, 307-310, 319-320) or, for cao2022multi, in a clause deleted from the main text. No kept main-text claim loses its reference. The claims at 164-166 (nearest designs; W1 and MMD alignment already exist) keep all their keys (li2022trade, jiang2026geometric, shen2018wasserstein, wang2021imap, amodio2019exploring, shaham2017removal, yi2025benchmarking).

Optional put-back (my default: none): one sentence in the Gap paragraph, text already in the outline at 306-310, naming other WGAN-GP-type integrators (wang2022respan, wang2023integrating, xu2024beaconet, wang2022imgg). That keeps 4 more keys (44 total).

FLAG, not verified: whether Bioinformatics Advances lets the supplement carry its own reference list, and whether references cited only there count toward the 50. WebFetch of the author guidelines was refused in this run, and line 35 of the outline records only the 50-reference limit. The supplement of this outline already cites keys (580, 605, 606, 654), but that is not evidence of the journal's rule. If supplement references are not allowed or count toward 50: S-R names the 19 works by author and year in plain text without a numbered entry, or, if even that is unwanted, S-R keeps only the six that use W1, MMD or optimal transport on single-cell data (respan, imaae, beaconet, imgg, bermuda, uniport) and drops the rest together with their lines. Decide before submission; it does not change the main list.

### 3.2 Keys that stay (40)

"First to go" means: if a slot is needed, remove this key together with the claim named; I did not choose to cut them now.

| Key | Cited at (lines) | Role |
|---|---|---|
| leek2010tackling | 114 | supporting; first to go with the "mislead when correlated with the outcome" half of 114 |
| yu2024assessing | 114, 116 | essential (two claims) |
| tung2017batch | 115 | supporting |
| goh2022batch | 115 | supporting; first to go with the second clause of 115 |
| luecken2022benchmarking | 117, 278, 383 | essential (metrics, tasks) |
| ganin2016domain | 122, 293 | essential |
| dincer2020adversarial | 122, 294 | essential |
| goodfellow2014generative | 123, 349 | essential |
| arjovsky2017wasserstein | 124, 130, 131, 350 | essential |
| gulrajani2017improved | 124, 356 | essential |
| arjovsky2017towards | 132-135 | essential |
| zhao2019learning | 148 | essential |
| wu2019domain | 149 | essential |
| hrovatin2025integrating | 151, 287, 467 | essential (composition shift; sysVI baseline) |
| maan2024characterizing | 152 | essential |
| lucic2018gans | 157 | essential (motivates X6) |
| kurach2019large | 158 | essential (motivates X6) |
| yi2025benchmarking | 164, 280 | essential (closest study) |
| li2022trade | 165, 322 | essential (nearest design) |
| jiang2026geometric | 165, 166, 311 | essential (nearest design) |
| shen2018wasserstein | 166, 305 | essential |
| wang2021imap | 166, 306 | essential |
| amodio2019exploring | 166, 318 | essential |
| shaham2017removal | 166, 317 | essential |
| miyato2018spectral | 176, 445, 606 | essential (X6) |
| mescheder2018training | 176, 445, 605 | essential (X6) |
| korsunsky2019fast | 271, 463 | essential (X13 Harmony) |
| hie2019efficient | 272, 464 | essential (X13 Scanorama) |
| lopez2018deep | 285, 331 | essential |
| gayoso2022python | 285, 331 | essential |
| xu2021probabilistic | 286, 466 | essential (scANVI) |
| gretton2012kernel | 316, 353 | essential (MMD arm) |
| cuturi2013sinkhorn | 321, 354 | essential (Sinkhorn arm) |
| genevay2018learning | 321, 354 | essential (Sinkhorn arm) |
| feydy2019interpolating | 321, 354 | essential (Sinkhorn arm) |
| cuturi2014fast | 321, 352, 580 | essential (barycenter arm) |
| flamary2021pot | 321, 580 | supporting (software; in S-M4 already) |
| buttner2019test | 396 | supporting (kBET) |
| corani2017statistical | 429 | supporting; could move to S-M15 if the main text states the rule without attribution |
| clopper1934use | 430, 654 | supporting; could move to S-M16 the same way |

Count: 19 leave + 40 stay = 59.

## 4. Abstract skeleton (limit 200 words; target 185)

Journal form assumed from PAPER_PLAN line 57: Motivation, Results, Availability. Square brackets are filled from the single analysis run; nothing here states a result. Numbers in plain text are design facts at the cited lines.

| Part | Sentence | Words |
|---|---|---|
| Motivation | 1. Adversarial batch correction trains an encoder against a discriminator that predicts each cell's batch (122). | 15 |
| | 2. We asked whether a Wasserstein-1 (W1) critic, maximum mean discrepancy (MMD) or Sinkhorn divergence keeps a different amount of biology than the Jensen-Shannon (JS) discriminator at equal batch removal in scVI (101). | 35 |
| Results | 3. In a pre-registered comparison on 8 single-cell integration benchmark (scIB) tasks, with and without the batch label in the decoder, we tested the discriminator, three W1 critics, MMD and Sinkhorn, 5 seeds and 6 penalty strengths each (410, 342). | 38 |
| | 4. At matched batch removal, the reference critic minus the discriminator differed in biology by [P1 estimate, interval, call] (conditioned decoder) and [same] (unconditioned decoder). | 25 |
| | 5. [Trade-off area, collapse rate and local-versus-global result: estimates, calls.] | 12 |
| | 6. [Range of good strengths per arm; X6: whether the difference is due to the divergence or to regularisation and adversary steps: result.] | 20 |
| | 7. [Placement against plain scVI, Harmony, Scanorama, sysVI and scANVI at equal numbers of tries, including whether any adversary improves on plain scVI.] | 20 |
| | 8. [One sentence of guidance for scVI users, worded from the rules.] | 10 |
| Availability | 9. Code and the tag of record (prereg-tier12-v3, commit 10a1ca5) are at github.com/kreid415/wasserstein-critic-deconfounding; task checksums are in the supplement (107). | 22 |
| Total | | about 197 (counts are my estimates; placeholders may grow, so count the filled text) |

If the filled text exceeds 200 words, trim in this fixed order, set before results: drop "with and without the batch label in the decoder" from sentence 3; merge sentences 5 and 6; shorten sentence 8. The R0 premise check is left out of the abstract (exploratory, no claim; 229). No "first" claim (183-220 forbid it). Five keywords are still to be chosen (not stated by the user).

## 5. Unsettled

1. Supplement reference rule and what counts toward 8 pages: not verified (WebFetch refused; section 3.1 flag, section 1).
2. Page estimate rests on 900 words per page, not built in the journal template.
3. Pending items that will add text or citations: R0 statistic and X10 classifier (455-456), X6 analysis (446), Sinkhorn pilot records (354, 366, 798-799), recomputed power (437).
4. Figure 2 strengths and Figure 3 row choice are fixed by rules above, but the R0 script that names them does not exist yet.
5. Title, authors, keywords and R5's hypothesis wording are the user's decisions (34, 255).
