# outline.tex: where each paragraph comes from

Checking aid only; not part of the outline. Line numbers are lines of `paper/paper.tex` as read on 2026-10-19 (844 lines). "BUDGET" = `paper/BUDGET.md`. Numbers in the outline (3,000 fits, 0.131, 0.031, 0.05/6, 0.02, 0.10, 0.01, 0.95) were read from the lines listed.

| Paragraph | paper.tex lines | Notes |
|---|---|---|
| Abstract (sentences 1-7) | 104-110; BUDGET section 4 | Skeleton follows BUDGET sentence order; results left as placeholders |
| Intro-1 | 117-120 | |
| Intro-2 | 125-127 | |
| Intro-3 | 133-138, 145-146 | Theory limited as in 133-138; premise check exploratory, no hypothesis (146, 232) |
| Intro-4 | 151-155 | Cited to sysVI paper only (cao2022multi not cited in the outline) |
| Intro-5 | 160-162 | Image-GAN inference stated as inference (162) |
| Intro-6 | 278-279, 284-287, 298 | scStarCorrect from its abstract only (279); ResPAN, IMAAE included (287); IMAAE not called Wasserstein-based (205-206) |
| Intro-7 | 167-169, 168 | Search-based absence, no "first" claim (168, 188-191); search repeat open (168) |
| Intro-8 | 174-176, 178-181, 231, 238-240, 248, 256, 258 | Contribution sentence as fixed in the task and 176; four questions in plain words from 231, 238-240, 248, 256 |
| Meth-1 | 307-311, 317-318 | |
| Meth-2 | 324-333 | Reference batch by cell-type entropy (333); Sinkhorn not W1 (330) |
| Meth-3 | 339-342, 348, 350-351 | Pilot outcomes for input and standardisation: rules R2/R3 re-run 2026-10-20, both keep the default (342; docs/prereg/r2r3_record.json at final-review-fixes 2168d75; NB-20261020-01) |
| Meth-4 | 359-365 | Immune-task and ATAC-task nesting in 363-364 |
| Meth-5 | 371-373, 408, 518 | Scoring on all cells (518) |
| Meth-6 pilot, strength ranges | paper.tex 61, 688, 695, 828; PREREG 40, 48, 50, 52-67; R1pass2 3-14, 48, 66-67, 545-557, 959-971, 1373-1385, 1787-1799, 2201-2213, 2683-2695, 3831-3843, 4979-4991 | Ranges from the `x1_grid` fields (lines 550, 964, 1378, 1792, 2206, 2688, 3836, 4984; `js\|1` = conditioned, `js\|0` = unconditioned). paper.tex 775-776 still lists the Sinkhorn ranges as pending; R1pass2 followed (the tag's r1_record.json holds the same values at the same lines). Supplement bullet: noise threshold PREREG 54; edge extensions PREREG 61, R1pass2 545-548 and 2683-2686 |
| Meth-7 main benchmark | MAN5 (counts by tag prefix: X1 3,000 = none 80 + scvi_adv 40 + six penalties 2,880); manifest_v5_counts.csv 2; R0X10 34; paper.tex 64, 226, 331, 359, 392, 415, 836; PREREG 197, 698-713; SPEC 73-76; SPEC-FU 3-9 | Freeze date 16 October 2026: paper.tex 392 (tag on GitHub 2026-10-16T21:48:53Z); old row Meth-6 gave 17:48 EDT. PREREG 3 still says the tag "has not been made" (stale status line). Follow-up registration: PREREG 708-713 and SPEC-FU 3-7 say the tag freezes the entry before the matched strengths are set; no source read gives the tag's creation date, so the text states none |
| Meth-8 matched level, low-signal | PREREG 36-44, 50, 91-95, 200; SPEC 121-125; paper.tex 342, 387, 389, 601; R4pass2 5124-5208 (immune\|1 at 5167-5180; sim1\|1 at 5195-5208); outline.tex Intro-4 and Res-3 | Pilot gaps: 0.6569909962526824 - 0.6381673018004433 = 0.0188 and 0.8188775558554181 - 0.8119601403099188 = 0.0069 (paper.tex 342 gives the same). The trade-off premise is taken from the outline's own wording (Intro-4, Res-3), not from a new source |
| Meth-9 primary comparison, trade-off area | SPEC 106-157; paper.tex 400-403, 608-618 | Reading, leave-one-seed-out, not-evaluable rule: SPEC 108-137; area and corner: SPEC 146-157 |
| Meth-10 collapse, local versus global | SPEC 219-246; paper.tex 406-407, 631-637 | Collapse and 30 fits (six points x 5 seeds): SPEC 221-223; local/global formula and single question: SPEC 237-246 |
| Meth-11 pooling, rank test, power | SPEC 100-102, 123-125, 159-217; PREREG 102-106, 333-350; paper.tex 363-364, 404-405, 409, 413-414, 621-627; R1pass2 9553-9620 | Power from the pilot: `mde80` 0.13066 (line 9607, real tasks) and 0.18178 (9596, families), `power_at_delta` 0.10436 (9608), equivalence not feasible (9592-9595, 9603-9606). R1pass2 9613 holds the four-test Holm value 0.0125; SPEC 214 and PREREG 106 give 0.05/6 (followed). Overlap of atac_small in atac_large: paper.tex 364 (SPEC 101 gives 3,618 shared cells; not re-checked against the data) |
| Meth-12 good-strength width | SPEC 37, 258-260, 276-281; paper.tex 408, 641, 644 | Window, width in decades, descriptive: SPEC 278-281 |
| Meth-13 baselines, single-strength tables, best of k | SPEC 248-256, 262-274; PREREG 197-198, 203; paper.tex 87, 256, 258, 415, 439-446, 642-643, 699; MAN5 (X13 counts by tag: harmony, scanorama, pca 128; scanvi 40; sysvi 240) | Old outline said six settings for every baseline; scANVI and PCA have one (paper.tex 439-446, SPEC 267-269) |
| Meth-14 follow-up design | PREREG 721-733, 742-746; SPEC-FU 32-36, 184-186, 226-231, 268-270; paper.tex 422, 582-584, 723; MAN5 (X6 count 486); followup_design 129-143 | Nine settings, 486 = 3 tasks x 2 decoders x 9 settings x 3 strengths x 3 seeds (PREREG 721-728) |
| Meth-15 follow-up contrast and decision rule | SPEC-FU 94-155, 226-239, 258-270; PREREG 771-776 | Explanation rule: SPEC-FU 258-268 (made from the records when results are written; not computed by the script, 266-268) |
| Meth-16 strength pre-check (planned) | PREREG 749-770, 782-790; SPEC-FU 188-203; PRECHK 1-38; R4pass2 5124-5208, 87-98, 159-170, 377-388 | 36 fits = 36 data rows of PRECHK (lines 3-20 discriminator arm, 21-38 pooled arm). Old row said "18-fit strength check"; an 18-row manifest was superseded on 2026-10-21 (file name only; first 4 lines opened); PREREG 754 and SPEC-FU 195 say 36, followed. Not run: no outcome stated |
| Meth-17 premise check | R0X10 16-19, 29-33, 42-59, 61-87, 119-134; paper.tex 430-432; outline.tex Intro-3 | 2,880 runs: R0X10 67; 80 plain-scVI embeddings: R0X10 46 |
| Meth-18 label transfer | R0X10 20-23, 89-117; paper.tex 433 | Matched-strength latents: R0X10 96-99; unmatched plain scVI and scvi-tools adversary: R0X10 23, 98 |
| Res-1 | 64, 389, 611 | Counts left as placeholders |
| Res-2 | 465-470; 429-431 | Exploratory, Figure 2 (468) |
| Res-3 | 477; BUDGET section 2 (Figure 3) | |
| Res-4 | 474-478; BUDGET section 2 (Figure 4, Table 2) | |
| Res-5 | 474-478; 614-617, 630-631 | |
| Res-6 | 474-478; 635-636; 432 | |
| Res-7 | 483-485; 643 | |
| Res-8 | 484-485; 421-422 | |
| Res-9 | 491-493; 641 | Table 3 and Figure 6b from BUDGET section 2 |
| Res-10 | 491-494; 642, 258 | No decision rule for placement (258) |
| Disc-1 | 500; 176 | Placeholder only |
| Disc-2 | 502; 160-162, 146, 151-155 | |
| Disc-3 | 501; 182 | |
| Disc-4 | 507-515, 518, 520; 333; 413 | Underpowered, Wilcoxon descriptive (513); power block of stage-2 R1 re-run (D/tier12_v3/rules/r1_pass2/r1_record.json) identical in all 43 fields to the tag's docs/prereg/r1_record.json (checked 2026-10-20) |
| Disc-5 | 515-517 | |
| Avail-1 | 110; BUDGET section 4 (availability sentence) | Tag and commit left as placeholder because tag names are not allowed in the outline |
| Supplementary material (8 lines) | 534-668, 672-707, 685-840 (S-M, S-D, S-E, S-X headings) | |
| Figures and tables (9 lines) | 462, 468, 477, 485, 493; BUDGET section 2 | |

Abbreviations in the Meth-6 to Meth-18 rows (added 2026-10-21; the Meth paragraphs were rewritten and renumbered, old Meth-6 to Meth-10 became Meth-6 to Meth-18). Line numbers are of the file as read on that date. Full value-by-value check: `paper/methods_6_10_factcheck.md`.

- PREREG, SPEC, SPEC-FU, followup_design: `docs/PREREG.md`, `docs/ANALYSIS_SPEC.md`, `docs/ANALYSIS_SPEC_FOLLOWUPS.md`, `docs/followup_design_decisions_20261017.md` in the export of tag prereg-x6-v1 (`code/meth_rewrite_20261022/prereg-x6-v1/`).
- MAN5: `manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv` in the same export (counted by pattern, never read row by row).
- R0X10: `paper/R0_X10_DEFAULTS.md`.
- R1pass2, R4pass2: `tier12_v3/rules/r1_pass2/r1_record.json`, `tier12_v3/rules/r4_pass2/r4_a1_record.json` (corrected-Sinkhorn pilot; the records in force).
- PRECHK: `tier12_v3/x6_precheck/manifest.tsv`.
- No main-benchmark or follow-up score, latent or result file was opened for these rows.
