# Final review of the merged paper/outline.tex (commit 16bbebd, branch paper)

Reviewer: Claude Code (claude-fable-5-1), 2026-10-21. Independent of the track writers; every value below was read from the file and line cited in this session. No main-benchmark, follow-up or baseline score, latent or result file was opened (the manifest of record was counted by tag pattern only; the pilot rule records r1_record.json and r4_a1_record.json hold pilot values only).

Source aliases: paper.tex = `paper/paper.tex`; PREREG = `code/meth_rewrite_20261022/prereg-x6-v1/docs/PREREG.md`; SPEC = `.../docs/ANALYSIS_SPEC.md`; SPEC-FU = `.../docs/ANALYSIS_SPEC_FOLLOWUPS.md`; MAN5 = `.../manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv`; r1 = `tier12_v3/rules/r1_pass2/r1_record.json`; r4 = `tier12_v3/rules/r4_pass2/r4_a1_record.json`; evidence = `paper/evidence.md`; registry = `configs/dataset_registry.json`; mockpy = `paper/drafts_20261022/make_mock_figures.py`. Line numbers of outline.tex are those of commit 16bbebd before my edits.

## 1. Issues found

| # | Location | Issue | Severity | Source checked | Fix |
|---|---|---|---|---|---|
| 1 | Methods, baselines paragraph (line 99) | "408 fits including two sensitivity settings": the 408 rows hold three settings beyond the base grids (Harmony 7 rows per task = 6 + 50 PCs; Scanorama 7 = 6 + 100 dimensions; PCA 2 = 10 and 50 components) | error | MAN5 X13 rows by tool: harmony 56, scanorama 56, pca 16, scanvi 40, sysvi 240 = 408; paper.tex:440-442 (sensitivity settings named) | Applied: "three sensitivity settings (Harmony at 50 principal components, Scanorama at 100 dimensions and PCA at 50 components)" |
| 2 | Methods, statistics paragraph (line 96) | Says Supplementary Note 6 gives the evaluability rules; the supplement outline puts them in Note 5 (line 209) and Res-4 cites Note 5 (line 126) | inconsistency | outline.tex:126, 209; SPEC:121-125 | Applied: "Supplementary Note 5" |
| 3 | Table 2 caption (line 328) vs Table S3 bullet (line 222) | Caption said the six strengths are "spread on a log scale over the chosen range", but in three of eight cells the grid is wider than the range (neighbours added: JS conditioned 10, 30; W1 unconditioned 100; MMD unconditioned 1,000). Table S3 bullet duplicated Table 2 and did not hold the range ends the caption promised | inconsistency | r1 `interval` vs `x1_grid` at lines 541-557 (JS uncond.), 955-971 (JS cond.), 1369-1385 (MMD uncond.), 1783-1799 (MMD cond.), 2197-2213 (Sinkhorn uncond.), 2679-2695 (Sinkhorn cond.), 3827-3843 (W1 uncond.), 4975-4991 (W1 cond.); PREREG:65 (w < 6 rule, "starting above") | Applied: caption states the neighbour rule; Table S3 bullet now holds the range ends, the added neighbours and edge extensions |
| 4 | Introduction P4 (line 59) | "GAN" never spelled out ("Wasserstein GAN gradient penalty (WGAN-GP)"); first use in the paper | style (NO-JARGON) | outline.tex: no earlier expansion | Applied: "gradient penalty of the Wasserstein generative adversarial network (WGAN-GP)" |
| 5 | Introduction P10 (line 71) | "KL warm-up" with KL not spelled out in the main text (first expansion was in Supplementary Note 1) | style (NO-JARGON) | outline.tex:71, 192, 205 | Applied: "Kullback--Leibler (KL) warm-up" |
| 6 | Methods, follow-up paragraph (line 102) | "R1 gradient penalty" reads like a project code label (rule R1) | style (NO-JARGON) | evidence:101 (Mescheder 2018, gradient penalty on real data only) | Applied: gloss "a penalty on the discriminator's gradient at real data only" |
| 7 | Table 1 (lines 314-320) | Four `\tbd{layers and width}` cells, `\tbd{blur}` and `\tbd{regularisation}` had values in the sources | style (open placeholder with a source) | paper.tex:332 (discriminator and critics: latent, 128, 128, one output per batch), paper.tex:330 (blur 0.05 sqrt(2d)), paper.tex:331 (scvi_adv: 32 hidden units, weight 1 - KL weight) | Applied: filled; `\tbd{steps}` for the scvi-tools adversary kept (no source read states its step count) |
| 8 | Table 4 caption (line 365) | "Unconditioned decoder in Supplementary Table \tbd{S-number}": the fixed supplement has no table slot for it (S1-S6 are assigned) | inconsistency | task numbering; outline.tex:218-226 | Left for author: either add a Supplementary Figure (figures may be added freely), fold both decoders into Table 4, or ask the coordinator for a seventh supplementary table |
| 9 | Methods, pooled model (line 96); Disc-4 (line 186); Note 6 (line 210) | "the large chromatin-accessibility task contains all cells of the small one": paper.tex:364 (data check) says every one of the 11,270 small-task cells is in the large task; SPEC:101 says the two share 3,618 cells. Sources disagree; the outline follows the later data check | inconsistency between sources, not in the outline | paper.tex:364; SPEC:101 | Left for author: confirm against the data files before submission (not opened here) |
| 10 | Methods, reference critic (line 84) | Rationale sentence "This reduces the number of missing cell-types ..." carries `\tbd{this rationale is in no registered document; confirm or drop}` | open item (already flagged) | PREREG:50; paper.tex:326, 333 give the rule only | Left for author (user's own sentence) |
| 11 | Introduction P9 (line 69) | "searches of OpenAlex, arXiv and Europe PMC run in October 2026" | checked, correct | evidence:66-67 (OpenAlex, arXiv and Europe PMC, publication years 2017 to 2026-10-15, 6 and 14 queries; 175 and 315 unique works) | None; the `\tbd{repeat the literature search before submission}` stays |
| 12 | Introduction header (line 44) | "about 1452 words ... target 850": the Introduction is well above the 850-word target and the paragraphs P3, P5, P7, P10 exceed 150 words | style (budget) | outline.tex:31 (paragraph rule 100-150 words); word count not re-run (no shell) | Left for author: cutting is a content decision |

## 2. Spot checks that passed (value, outline location, source)

| # | Item | Outline | Source |
|---|---|---|---|
| 1 | Main benchmark 3,000 fits = 80 plain scVI + 40 scvi-tools adversary + 6 x 480 | Methods line 93; Res-1 | MAN5 counts by pattern `^X1_` 3000, `^X1_\w+_none_` 80, `scvi_adv` 40; PREREG:197 |
| 2 | Baselines 408 fits | Methods line 99; Res-1 | MAN5 `^X13_` 408 |
| 3 | Follow-up 486 fits = 3 x 2 x 9 x 3 x 3; seeds 10-12; nine settings | Methods line 102; Note 9 | MAN5 `^X6_` 486; PREREG:721-727 |
| 4 | Strength pilot 1,098 fits; tasks small ATAC, immune, sim1; seeds 100-102; grid 0.1 to 3,000 | Methods line 90; Note 4 | MAN5 `^A1_` 1098; PREREG:34, 48 |
| 5 | Edge extensions: two (JS unconditioned low edge, Sinkhorn conditioned high edge), 36 fits, values 0.03/0.01 and 10,000/30,000 | Methods line 90 | r1:545-548 (low 1), 2683-2686 (high 1); PREREG:34, 61 |
| 6 | Table 2, all eight cells of six strengths | Table 2 | r1 `x1_grid` lines listed in issue 3 |
| 7 | Range rule (no effect: no failed fit, both scores within noise; upper end: stopped gaining batch removal or collapsed; collapse 0.10; six log-spaced points; neighbours above first) | Methods line 90; Note 4 | PREREG:52-65, 40 |
| 8 | b* = midpoint between plain scVI and the lowest of the six penalties' best batch scores at strengths with no failed fit; low-signal when gap < 0.02 (2 x 0.01) | Methods line 93; Note 5 | PREREG:93-95, 38, 193 |
| 9 | Pilot low-signal pairs: immune conditioned (gap 0.019), sim1 conditioned (0.007); the other four pairs not flagged; Note 5 values 0.638/0.657 and 0.812/0.819 | Methods line 93; Note 5 | r4:5176-5179, 5204-5207, 5137, 5151, 5165, 5193 |
| 10 | Minimal detectable effect 0.131 (six real tasks), 0.182 (four families), power 0.104 at a 0.02 difference, equivalence not showable, Wilcoxon smallest p 0.031 | Methods line 96; Disc-4; Note 6 | r1:9587 (delta 0.02), 9596-9597, 9601-9608, 9612-9620 |
| 11 | Holm first threshold 0.05/6 = 0.0083 over six primary results (r1 stores the older four-test 0.0125) | Methods line 96; Res-4 | SPEC:211-217; PREREG:106 |
| 12 | Pooled model: six real tasks, four families, nesting, priors scale 0.1, exact integration, 0.95 rule, margin 0.01, simulations never pooled | Methods line 96; Note 6 | SPEC:100-102, 167-176, 203-208 |
| 13 | Pilot rules frozen 4 October 2026; amendments and analysis 16 October 2026; 19 fits scored and unread | Methods line 96; Note 11 | PREREG:232; paper.tex:392, 226 |
| 14 | Follow-up confirmatory contrast (SN pooled critic at 1 step minus SN discriminator with the critics' optimiser), Holm over 2, companion smallest p 0.25 with three tasks | Methods line 102; Res-8; Note 9 | PREREG:771-776; SPEC-FU:153 |
| 15 | Pre-check: 36 fits, 18 per arm, pilot tasks, seed 100, rule "at least 3 of 4 counted cells" one half-decade step, no second pre-check | Methods line 102; Note 9 | PREREG:749-766 |
| 16 | "no batch label reaches the encoder or decoder" (unconditioned) | Methods line 81 | paper.tex:317 |
| 17 | Reference batch = highest cell-type Shannon entropy, uses labels; "the reference batch's own cells also move" | Methods line 84 | paper.tex:326, 333 |
| 18 | Discriminator 1 step with the generator's Adam; critics 5 steps, Adam 1e-4, betas (0, 0.9); gradient penalty 10 | Methods line 81, 84; Table 1; Note 2 | paper.tex:325-326, 332, 348 |
| 19 | scvi-tools defaults: 10 latent dimensions, 128 hidden units, ZINB; epochs, minibatch, latent size equal within a task | Methods line 81 | paper.tex:307-308 |
| 20 | Pooled critic: each batch against all other batches pooled; barycenter per minibatch, recomputed each step; Sinkhorn not W1 at the smoothing used | Methods line 84 | paper.tex:327-328, 330 |
| 21 | Tasks: four RNA (pancreas, lung, immune, human-mouse immune), two ATAC gene-activity (mouse brain), two simulations; 2,000 HVGs for RNA, all ATAC features; counts as distributed | Methods line 87 | paper.tex:359-361; registry modalities (rna x4, rna_sim x2, atac x2) |
| 22 | Batch metrics (five) and biology metrics; cell cycle on the four real RNA tasks; trajectory on the two immune tasks; unscaled; all cells | Methods line 87; Note 3 | paper.tex:371-373 |
| 23 | Selection score 0.4 batch + 0.6 biology | Methods line 87; Res-7, Res-9 | SPEC:258 |
| 24 | Harmony theta 0-8 at 10 PCs, Scanorama knn 2-80 at 10 dimensions, sysVI cycle weight 1-50, six settings each; scANVI and PCA one; neural tools seeds 0-4 | Methods line 99; Res-9 | paper.tex:439-444 |
| 25 | Best-of-k expected best selection score, k = 1-6, optimistic, scANVI own stratum | Methods line 99; Res-10 | SPEC:267-274 |
| 26 | Collapse rate: 30 fits per method, task, decoder; Clopper-Pearson; margin 5 percentage points; low-signal kept | Res-5; Note 7 | SPEC:221-230 |
| 27 | Local = mean(iLISI, kBET, graph connectivity), global = mean(PCR, batch ASW), B = (3 local + 2 global)/5, sign test plus pooled model | Res-6; Note 7 | SPEC:237-246 |
| 28 | Trade-off area corner (b0, c0 - 0.10), divided by the batch gap, low-signal excluded | Res-5; Note 7 | SPEC:146-153 |
| 29 | Good-strength width: lower 95% t bound of the best strength's mean, width in decades | Res-7; Note 7 | SPEC:278-281 |
| 30 | Premise check: 80 plain-scVI embeddings; 2,880 penalty fits; 15-NN classifier, five folds, AUC >= 0.95 in all five seeds; vanish rule 10% of peak within first 10% of epochs | Res-2; Note 10 | paper.tex:430-432 |
| 31 | Label transfer: leave one batch out, 15-NN, types in fewer than two batches dropped, F1 per type, matched strength never chosen by F1 | Res-6; Note 10 | paper.tex:433 |
| 32 | iMAP: critic trained with WGAN-GP, anchor dataset not corrected | Intro P4 | evidence:21 |
| 33 | WDGRL, Beaconet, IMGG, ResPAN, IMAAE, AIF, ABC, iDLC, scStarCorrect (abstract only), Portal (clamped logits) descriptions | Intro P4-P5 | evidence:20, 51-52, 122-128, 136-138 |
| 34 | Theory sentences (W1 continuous, parallel-lines example, perfect discriminator with zero gradient, assumptions) | Intro P6 | evidence:16-17 |
| 35 | Composition-shift sentences (Zhao 2019 Thm 4.3, Wu 2019 Prop 3.1, sysVI, Maan 2024) | Intro P7 | evidence:24-26, 29, 74 |
| 36 | Image-GAN sentences (Lucic 2018, Kurach 2019) | Intro P8 | evidence:27-28, 76 |
| 37 | scIB-E six batch losses, one a GAN discriminator; Li et al. MMD versus adversarial penalty on scVI 0.3.0; iDLC ablation has no JS or MMD arm | Intro P9 | evidence:23, 75, 122, 129 |
| 38 | All 49 citation keys used in outline.tex are present in refs.bib | whole file | `paper/refs.bib` (grep of every key) |
| 39 | Discussion agrees with Methods: five critic steps vs one discriminator step, update-step control for the pooled critic only, power 0.131, four families with two nested pairs, grids set on three of eight tasks, KL warm-up incomplete on the shorter tasks (four tasks have fewer than 400 epochs by the epoch rule: immune, lung, human-mouse immune, large ATAC, from the registry cell counts and paper.tex:308) | Disc-4, Disc-5 | outline.tex:81-102; paper.tex:308, 350 |
| 40 | Rules: no `\res{}` in Methods; the only Methods `\tbd` placeholders are the pre-check outcome, the entropy rationale and the scvi-tools adversary step count; no `\cite` in the supplement section; all six figure captions begin "MOCK -- synthetic data."; Tables 3 and 4 begin "MOCK -- placeholder values"; no X1/X6/X13/A1/SI code labels in prose (only "R1" for the published regulariser, now glossed) | whole file | grep of outline.tex |
| 41 | Every `\citep` inside `\deleted{}` or the second argument of `\replaced{}{}` is wrapped in `\mbox{}` (lines 53, 61, 65, 71, 84, 87) | Intro, Methods | read |

## 3. Mock panel titles that state a finding (make_mock_figures.py; PNGs not regenerated)

| Figure, panel | Title text | Line in mockpy |
|---|---|---|
| Fig 1b | "biology is read where each curve reaches $b^*$" (describes the rule, not a finding; listed for completeness) | 99 |
| Fig 2a | "batches nearly separable" | 117 |
| Fig 2b | "discriminator gradient fades early" | 133 |
| Fig 2c | "accuracy saturates quickly" | 141 |
| Fig 4 (suptitle) | "reference critic minus discriminator, per task and pooled" (neutral) and the footer "favours discriminator / favours reference critic ... lower collapse with critic ... critic removes batch more locally" (directional reading aids, neutral) | 216 |
| Fig 5a | "good-strength width, one point per task"; annotation "wider = less tuning needed" (interpretive) | 235-236 |
| Fig 5b | "follow-up: which ingredient moves biology?" (question, neutral) | 249 |
| Fig 6a | "expected best score against number of tries" (neutral) | 272 |
| Fig 6b | "single strength chosen on other task families" (neutral) | 288 |

Recommendation for the figure track: replace the three Figure 2 titles with neutral ones ("batch separability of plain scVI", "adversary gradient norm per epoch", "discriminator accuracy per epoch") and drop "wider = less tuning needed" before any real data are plotted.

## 4. Not checked

- No main-benchmark, follow-up or baseline score, latent or result file (by instruction).
- Word counts of the Introduction and of each paragraph (no shell or kernel access in this session; the header's "about 1452 words" and the paragraph-length rule were not re-measured).
- LaTeX compilation (no tectonic). Brace balance of the lines I edited was checked by eye; the whole file was not machine-checked.
- Supplementary Table S1 cell counts (the outline does not print them; the registry values quoted in methods_changelog.md section 5 were used only for the four-shorter-tasks epoch derivation in check 39 and were not re-read from the registry).
- The overlap between the small and large chromatin-accessibility tasks (issue 9) against the data files.
- Portal's clamp description beyond evidence.md:136-138 (the preprint itself was not opened).
- The scvi-tools adversary's update-step count (left as `\tbd{steps}` in Table 1).
- The number of cells with pseudotime for the trajectory metric, and the exact scIB biology metric list per simulation task (taken from paper.tex:372 and the registry modality fields; score_scib_native.py was not opened).
- paper.tex lines outside 226, 307-311, 317, 324-333, 339-342, 348-353, 359-364, 371-374, 392, 430-433, 439-446.
