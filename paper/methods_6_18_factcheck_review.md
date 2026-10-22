# Fact check: outline.tex Methods heading and paragraphs Meth-6 to Meth-18

Checked 2026-10-21 by the rigor reviewer. Text checked: `paper/outline.tex` line 92 (Methods heading) and lines 123-217 (Meth-6 to the line before Res-1). Every value was read in this session from the source named in the table. The writer's own fact table (`paper/methods_6_10_factcheck.md`) was used only to find where to look.

**Counts.** 170 statements (table rows): 157 correct, 6 wrong, 3 with no supporting source, 3 that agree with the project's evidence record but whose cited paper I could not open (web access was denied), and 1 not checked (word count; shell access was denied).

**Most important, in severity order.**
1. Meth-15: "read at b\* as in the primary comparison" is wrong. The follow-up reading has three points and no plain-scVI starting point, and it needs all three leave-one-seed-out readings.
2. Meth-15: the fixed-strength fallback is misdescribed. A task that does not bracket b\* is *not evaluable* for the confirmatory result. The fixed-strength difference is a separate descriptive report, given for every task.
3. Meth-15 result placeholder: "each setting minus plain discriminator" is not the registered set of secondary contrasts.
4. Meth-7: "decision rules and analysis were frozen on 16 October 2026". The pilot rules were frozen on 4 October 2026. The 16 October tag froze their amendments and the main-benchmark analysis.
5. Meth-6: the range rule as written ("from ... no effect ... to ... over-corrected or collapsed") does not hold for 3 of the 8 grids. Two range endpoints (0.03 and 10,000) fall outside the stated pilot grid (0.1 to 3,000), and the main text does not explain why.

No result of the blinded main benchmark or follow-up appears in the text. Every outcome stated is from the pilot or is a design property, such as the smallest attainable p-value.

Source aliases: **PREREG**, **SPEC**, **SPEC-FU** = `docs/PREREG.md`, `docs/ANALYSIS_SPEC.md`, `docs/ANALYSIS_SPEC_FOLLOWUPS.md` in the export of tag prereg-x6-v1 (`code/meth_rewrite_20261022/prereg-x6-v1`). **FDD** = `docs/followup_design_decisions_20261017.md` in that export. **MAN5** = `manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv` in that export, all 5,280 rows parsed by column. **paper.tex**, **R0X10**, **evidence.md** = files in `code/wcd_paper/paper/`. **r1** = `tier12_v3/rules/r1_pass2/r1_record.json`. **r4** = `tier12_v3/rules/r4_pass2/r4_a1_record.json`. JSON values are cited by key path rather than line number. The r1, r4 and r2r3 records in force are byte-identical (same SHA-256) to the copies under `docs/prereg/` in the tag export: r1 dd73d849..., r4 15b4e8ea..., r2r3 ab836f9f.... **PRECHK** = `tier12_v3/x6_precheck/manifest.tsv` (36 data rows, all read). "cond 1" means the conditioned decoder (SPEC-FU:28-29; paper.tex:769-774).

Verdicts: **C** correct; **W** wrong; **U** not supported by any listed source; **C-sec** agrees with the project's evidence record, but I did not open the cited paper; **NC** not checked.

## 1. Statement table

| paragraph | statement | verdict | source file:line | source value | suggested fix |
|---|---|---|---|---|---|
| Heading | 18 paragraphs | C | outline.tex:93-211 | Meth-1 to Meth-18 present | - |
| Heading | about 2,500 words | NC | - | writer's hand count: 2,096 words (Meth-6 to Meth-18) plus 59 (supplement bullets); Meth-1 to Meth-5 not counted | Recount at build. The shell (needed for a word count) was denied in this session |
| Meth-6 | pilot on three tasks | C | PREREG:48; MAN5 A1 | atac_small, immune, sim1 | - |
| Meth-6 | reason: strength has a different scale for each penalty | C | PREREG:50 | "have different loss scales" | - |
| Meth-6 | "gave each method its own strength range" | W (minor) | PREREG:50; r1 `families` (8 keys) | one grid per penalty family and decoder; the three critics share one grid | "gave each penalty family its own strength range for each decoder" |
| Meth-6 | fitted plain scVI and the six penalties | C | MAN5 A1 arms | none 18; barycenter, discriminator, mmd, pooled, reference, sinkhorn 180 each | Wording: plain scVI is fitted at strength 0, not "at ten strengths" |
| Meth-6 | ten strengths from 0.1 to 3,000 | C | r1 `constants.A1_BASE_GRID`; MAN5 A1 lam set | 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000 | See Completeness gap 1 (extension fits) |
| Meth-6 | small ATAC, immune, simulation 1 | C | PREREG:48 | "Tasks atac_small, immune, sim1" | - |
| Meth-6 | both decoders | C | PREREG:48 | "conditioned and unconditioned decoders" | - |
| Meth-6 | seeds 100-102 | C | PREREG:48; MAN5 A1 seeds | 100, 101, 102 | - |
| Meth-6 | 1,098 fits | C | MAN5 (A1 = 1,098 rows); paper.tex:695 | 3 tasks x 2 decoders x 3 seeds x (1 + 6 x 10) = 1,098 | - |
| Meth-6 | six log-spaced strengths | C | PREREG:64; r1 `N_GRID` 6 | "6 points evenly spaced in log λ", rounded to the half-decade grid | - |
| Meth-6 | range starts at the largest strength where every pilot task still shows no effect | C | PREREG:56, 63 | i_lo = min over blocks of the floor; "at λ_{i_lo} every block shows no effect" | Optional: "every pilot task and method" (the blocks are task x decoder x method) |
| Meth-6 | range ends where every task has over-corrected or collapsed | W (partial) | PREREG:65; r1 `families[*].interval` vs `x1_grid` | js\|1 interval 0.1-3, grid 0.1-30; mmd\|0 interval 3-300, grid 3-1,000; w1\|0 interval 0.3-30, grid 0.3-100 | Add: "where that span holds fewer than six grid values, the next values above (then below) are added" |
| Meth-6 | over-corrected = batch removal stops rising | C | PREREG:58-59 | "batch removal no longer increases, within noise" | - |
| Meth-6 | collapsed = a fit fails, or biology 0.10 or more below plain scVI | C | PREREG:40, 57; paper.tex:77 | n_fail ≥ 1 or Δc ≤ -0.10 (seed-mean change) | - |
| Meth-6 | discriminator, conditioned 0.1-30 | C | r1 `families["js\|1"].x1_grid` | 0.1, 0.3, 1, 3, 10, 30 | - |
| Meth-6 | discriminator, unconditioned 0.03-10 | C | r1 `families["js\|0"].x1_grid` | 0.03, 0.1, 0.3, 1, 3, 10 (0.03 from one low-edge extension round) | Explain 0.03 (Completeness gap 1) |
| Meth-6 | critics, conditioned 0.1-300 | C | r1 `families["w1\|1"].x1_grid` | 0.1, 0.3, 3, 10, 100, 300 | - |
| Meth-6 | critics, unconditioned 0.3-100 | C | r1 `families["w1\|0"].x1_grid` | 0.3, 1, 3, 10, 30, 100 | - |
| Meth-6 | MMD, conditioned 3-3,000 | C | r1 `families["mmd\|1"].x1_grid` | 3, 10, 30, 300, 1000, 3000 | - |
| Meth-6 | MMD, unconditioned 3-1,000 | C | r1 `families["mmd\|0"].x1_grid` | 3, 10, 30, 100, 300, 1000 | - |
| Meth-6 | Sinkhorn, conditioned 0.3-10,000 | C | r1 `families["sinkhorn\|1"].x1_grid`; r1_pass2/changes_vs_tag.json | 0.3, 3, 30, 100, 1000, 10000 (corrected Sinkhorn pilot; 10,000 from one high-edge extension round) | Explain 10,000 (Completeness gap 1) |
| Meth-6 | Sinkhorn, unconditioned 0.3-300 | C | r1 `families["sinkhorn\|0"].x1_grid` | 0.3, 1, 3, 30, 100, 300 | - |
| Meth-6 | Supplement: noise threshold, six strengths, extra edge fits | C | PREREG:44, 54, 61 | δ = max(0.01, 3σ/√n); extension of 2 points per round, at most 2 rounds | Add the Sinkhorn pilot refit to this list (Completeness gap 2) |
| Meth-7 | each method at its six pilot strengths on all eight tasks | C | MAN5 X1 (8 tasks); r1 x1_grid | six registered strengths per family | "each penalty" (plain scVI and the scvi-tools classifier have no strength grid) |
| Meth-7 | question: reference critic vs discriminator at equal batch removal | C | SPEC:103, 127 | P1 = reference minus discriminator at b\* | - |
| Meth-7 | both decoders, seeds 0-4 | C | MAN5 X1 seeds 0-4, 600 each; PREREG:197 | - | - |
| Meth-7 | plain scVI 80 fits | C | MAN5 X1 none | cond 0: 40, cond 1: 40 | - |
| Meth-7 | scvi-tools classifier 40 fits, conditioned only | C | MAN5 X1 scvi_adv | cond 1: 40 | - |
| Meth-7 | six penalties x six strengths, 480 each | C | MAN5 X1 | each arm 240 + 240 | - |
| Meth-7 | 3,000 fits | C | MAN5 X1 = 3,000 | 80 + 40 + 6 x 480 | - |
| Meth-7 | "decision rules and analysis were frozen on 16 October 2026" | W (partial) | PREREG:3; paper.tex:44, 46, 392 | rules v2 frozen at tag prereg-tier12-v2, 2026-10-04, before the restarted pilot; v3 amendments and the analysis spec frozen 2026-10-16 | "The pilot rules were frozen on 4 October 2026, before the first pilot fit; their amendments and the main-benchmark analysis were frozen on 16 October 2026 ..." |
| Meth-7 | before any main-benchmark score was examined | C (with caveat) | paper.tex:226, 836-837; SPEC:25-26 | "examined" = a person read the value (SI-78); 19 main-benchmark fits had already been scored, unread; user statement 2026-10-15 | Say it plainly: "before anyone read a main-benchmark score (19 fits had been scored but not read; Supplement)" |
| Meth-7 | the analysis is run once | C | SPEC:75; paper.tex:415 | "runs once, after all 3,000 X1 rows and the 408 X13 rows are scored or failed" | - |
| Meth-7 | follow-up rule registered before any follow-up fit was scored | C | PREREG:708-713; SPEC-FU:3-7 | pre-check fits "planned and not done"; follow-up fits need the main-benchmark matching rule first | Optional, stronger and supported: "before the matching rule ran on the main benchmark and before any main-benchmark result was examined" |
| Meth-7 | Supplement list | C | paper.tex:390-392, 836 | dates, amendments, 19 early-scored rows, R4 extension | - |
| Meth-8 | equal strength is not equal correction; scales differ | C | PREREG:50 | different loss scales | - |
| Meth-8 | biology trades off against batch removal | U (motivation) | no listed source; Intro-4 cites literature | - | Acceptable as motivation; cite \citep{luecken2022benchmarking} if a source is wanted |
| Meth-8 | b\* halfway between plain scVI's mean and the highest mean all six reach without a failed fit | C | PREREG:94-95 | b0 + 0.5 (b_common - b0); b_common = min over arms of the best failure-free seed-mean B | - |
| Meth-8 | low-signal if gap < 0.02 | C | PREREG:95, 38; paper.tex:389 | b_common - b0 < 2 x 0.01 | - |
| Meth-8 | reason: b\* cannot be placed reliably | C | PREREG:193 | "so that b\* cannot be placed reliably" | - |
| Meth-8 | low-signal pairs reported separately | C | SPEC:123-124 | "shown in a separate table and never pooled" | - |
| Meth-8 | pilot low-signal: immune and sim1, conditioned | C | r4 `targets` | immune\|1 and sim1\|1 true; the other four false | - |
| Meth-8 | gaps 0.019, 0.007 | C | r4 `targets` | 0.65699 - 0.63817 = 0.01882; 0.81888 - 0.81196 = 0.00692 | - |
| Meth-9 | curve joins plain scVI and six strengths by mean scores | C | SPEC:108-110 | (b0, c0) then six points, means over successful runs | - |
| Meth-9 | read where the curve first reaches b\*, interpolating | C | SPEC:112-115 | first point equal, or first bracketing pair, linear in B | - |
| Meth-9 | reference minus discriminator per task and decoder | C | SPEC:127, 141 | one P1 result per decoder | - |
| Meth-9 | leave-one-seed-out uncertainty | C | SPEC:129-133 | 5 readings, jackknife | - |
| Meth-9 | not evaluable: low-signal, never reaches b\*, fewer than 4 of 5 readings | C | SPEC:121-123 | (i)-(iii) | - |
| Meth-9 | area uses all six points | C | SPEC:146 | "its six points" | - |
| Meth-9 | area beyond a corner at b0 and 0.10 below c0 | C | SPEC:146-147, 152 | union of rectangles [b0, B_i] x [c0 - 0.10, C_i] | - |
| Meth-9 | divided by the batch gap that sets b\* | C | SPEC:148 | divided by (b_common - b0) | - |
| Meth-10 | motivation: a penalty that sometimes ruins the embedding is hard to use | U (motivation) | no listed source | - | Acceptable as motivation |
| Meth-10 | collapse = fails or C ≥ 0.10 below the same seed's plain scVI | C | SPEC:221 | as stated | - |
| Meth-10 | 30 fits per method, task and decoder | C | SPEC:222-223 | 6 points x 5 seeds; also pooled over the 6 real tasks | - |
| Meth-10 | Clopper-Pearson intervals | C | SPEC:223 | "Clopper-Pearson 95% intervals" | - |
| Meth-10 | gloss "exact binomial 95% intervals" | C | evidence.md:40 (title of clopper1934use: "confidence or fiducial limits ... binomial") | standard definition; paper not opened | - |
| Meth-10 | difference pooled, margin 5 percentage points | C | SPEC:228-230; paper.tex:632 | margin 0.05, priors 0.25 | - |
| Meth-10 | local = iLISI, kBET, graph connectivity | C | SPEC:237 | as stated | - |
| Meth-10 | global = PCR, batch ASW | C | SPEC:238 | as stated | - |
| Meth-10 | B = (3 local + 2 global)/5 | C | SPEC:238 | "exactly" | - |
| Meth-10 | at b\* more local means less global | C | SPEC:239-241 | 3ΔL + 2ΔG = 0 | - |
| Meth-10 | tested at the primary crossing, exact sign test over real tasks and the pooled model | C | SPEC:243-246 | two-sided sign test plus model, r = 0.01 | "whether the reference critic ..." (name the critic) |
| Meth-11 | six real tasks pooled in one model | C | SPEC:100, 167 | - | - |
| Meth-11 | rank test is description only | C | SPEC:196, 213 | "descriptive" | - |
| Meth-11 | each task weighted by its uncertainty | C | SPEC:167 | d_t ~ N(θ_t, s_t²), s_t known | - |
| Meth-11 | related tasks share a family effect | C | SPEC:168 | u_family(t) | - |
| Meth-11 | all immune cells are in the human-mouse immune task | C | SPEC:101; paper.tex:363 | 33,506 cells | - |
| Meth-11 | small ATAC cells are in the large ATAC task | C (sources differ) | paper.tex:364 vs SPEC:101 | paper.tex: all 11,270 (3,618 under identical names); SPEC: "share 3,618 cells" | Keep, but make SPEC and the supplement agree |
| Meth-11 | simulations and low-signal pairs never pooled | C | SPEC:101-102, 123-124 | - | - |
| Meth-11 | direction if P ≥ 0.95 beyond ±0.01 on one side | C | SPEC:203-207 | - | - |
| Meth-11 | equivalence if P(\|Δ\| ≤ 0.01) ≥ 0.95, else inconclusive | C | SPEC:203-204 | - | - |
| Meth-11 | 0.01 = smallest biology difference counted as real | C | PREREG:38, 194 | δ_min "the registered smallest difference that counts" | - |
| Meth-11 | Corani et al. rule | C | SPEC:203 | "Corani 2017" | - |
| Meth-11 | Wilcoxon signed-rank with Holm over six (comparison, area, collapse rate, each per decoder) | C | SPEC:211-214 | "P1 and the area for each decoder, plus P2 for each decoder: six" | - |
| Meth-11 | cannot reject with six tasks | C | SPEC:214-215 | - | - |
| Meth-11 | smallest p 0.031 | C | SPEC:215; r1 `power.wilcoxon_min_p_two_sided` | 0.03125 | - |
| Meth-11 | first threshold 0.05/6 = 0.0083 | C | SPEC:214 | (r1 stores the four-test value 0.0125; PREREG:106 explains) | - |
| Meth-11 | design underpowered (from the pilot) | C | r1 `power.underpowered` | true | - |
| Meth-11 | minimal detectable effect at 80% power 0.131 (six real tasks) | C | r1 `power.designs.real_tasks.mde80` | 0.13066 | - |
| Meth-11 | 0.182 (four families) | C | r1 `power.designs.families.mde80` | 0.18178 | - |
| Meth-11 | 80% power | C | PREREG:104 | - | - |
| Meth-11 | power for 0.02 is 0.104 | C | r1 `power.designs.real_tasks.power_at_delta` | 0.10436 (four families: 0.0878) | Add "(six real tasks)" |
| Meth-11 | equivalence within ±0.02 cannot be shown | C | r1 `...equivalence_feasible["0.02"]` | false (also false for ±0.01) | - |
| Meth-11 | gloss: posterior probability = the model's probability given the data | C | standard | - | Optional: "given the data and the priors" |
| Meth-12 | S = 0.4B + 0.6C, scIB weights, unscaled | C | SPEC:258 | - | - |
| Meth-12 | S does not depend on which methods are compared | C | SPEC:258-259 | - | - |
| Meth-12 | for each method, task and decoder | C | SPEC:278 | "Per arm, dataset and decoder" | - |
| Meth-12 | best strength = highest mean S over five seeds | C | SPEC:278 | mean over successful runs | - |
| Meth-12 | good range ≥ lower end of the best point's 95% interval | C | SPEC:278-279 | t interval, n - 1 df | - |
| Meth-12 | width in decades, smallest to largest | C | SPEC:280 | log10 difference; 0 for a single point | - |
| Meth-12 | descriptive, no decision rule | C | SPEC:281 | "Descriptive." | - |
| Meth-13 | asks whether any adversary improves on plain scVI | C | paper.tex:256 | - | - |
| Meth-13 | comparators; scANVI uses labels; PCA is the uncorrected reference | C | SPEC:267-269; paper.tex:442-443 | - | - |
| Meth-13 | Harmony, Scanorama, sysVI at six settings each | C | paper.tex:440-441, 444; SPEC:267-268 | θ {0, 0.5, 1, 2, 4, 8}; knn {2, ..., 80}; cycle weight {1, ..., 50} | - |
| Meth-13 | scANVI and PCA at one | C | paper.tex:442-443 | PCA 10 components (50 as sensitivity) | - |
| Meth-13 | seeds 0-4 neural, one run otherwise | C | paper.tex:87; MAN5 X13 seeds | scANVI and sysVI seeds 0-4; Harmony, Scanorama, PCA seed 0 | - |
| Meth-13 | 408 fits with sensitivity settings | C | MAN5 X13 | Harmony 56, Scanorama 56, PCA 16, scANVI 40, sysVI 240 | - |
| Meth-13 | single strength: best mean S on real tasks of other families | C | SPEC:250-254 | leave-one-family-out; only failure-free points eligible | Optional: simulations take the strength chosen on all six real tasks (SPEC:255) |
| Meth-13 | related tasks never choose for each other | C | SPEC:252-253 | - | - |
| Meth-13 | own best strength shown as an optimistic bound | C | SPEC:255-256 | - | - |
| Meth-13 | expected best of k drawn at random, k = 1-6 | C | SPEC:270-271 | E_k formula, k = 1..N | - |
| Meth-13 | so methods with more settings gain no advantage | C | SPEC:271 | "families compared at equal k" | - |
| Meth-13 | optimistic: chosen and scored on the same data | C | SPEC:271-272 | - | - |
| Meth-13 | scANVI has its own stratum | C | SPEC:272 | - | - |
| Meth-13 | both descriptive | C | paper.tex:258 | - | - |
| Meth-14 | topic: divergence vs smoothing, optimiser, update steps | C | SPEC-FU:226-227 | Lipschitz control, optimiser, steps | Replace "smoothing" (Meth-2 uses it for Sinkhorn): "slope control (regularisation)" |
| Meth-14 | small ATAC, immune, pancreas | C | PREREG:721; MAN5 X6 tasks | 162 rows each | - |
| Meth-14 | both decoders | C | MAN5 X6 cond | 243 + 243 | - |
| Meth-14 | seeds 10-12 | C | PREREG:723; MAN5 X6 | 162 each | - |
| Meth-14 | three strengths: the base method's matched strength and its two grid neighbours | C | PREREG:727-728; SPEC-FU:32-36; MAN5 X6 lam labels | matched_lo, matched, matched_hi (resolved on the main benchmark) | Add "on the main benchmark"; two arms then shift by the pre-check (Meth-16) |
| Meth-14 | nine settings | C | MAN5 X6 arm x n_critic | 9 combinations, 54 rows each | - |
| Meth-14 | 486 fits | C | MAN5 X6 = 486 | 3 x 2 x 9 x 3 x 3 | - |
| Meth-14 | discriminator plain, SN, R1, SN + critic's optimiser, each at one step | C | PREREG:723-725; MAN5 | n_critic 1 for all four | - |
| Meth-14 | SN gloss: each layer rescaled so its output cannot change steeply | C-sec | evidence.md:102 (quotes Miyato et al. Sec. 2.1, Eq. 8) | W / σ(W) bounds the Lipschitz norm by 1 | Optional, more exact: "each layer's weights divided by their largest stretch, so the output cannot change faster than the input" |
| Meth-14 | R1 gloss: penalising the output's slope at the cells | C-sec | evidence.md:101 (Mescheder et al. Eq. 9); paper.tex:582 | (γ/2)(1/n) Σ ‖∇_z D(z_i)‖² at each cell's latent code | - |
| Meth-14 | critic's optimiser = the critics' learning rate and momentum settings | C (incomplete) | paper.tex:584; FDD:129 | Adam lr 1e-4, betas (0, 0.9), replacing lr 1e-3, ε 0.01, weight decay 1e-6 | "the critics' Adam settings (learning rate 10^-4, betas 0 and 0.9)" |
| Meth-14 | pooled critic: GP at 1, 5, 10 steps; SN at 1 and 5 | C | PREREG:725-726; MAN5 | pooled 1/5/10; pooled_sn 1/5 | - |
| Meth-14 | follow-up uses the pooled critic, not the reference critic | C | SPEC-FU:270 | - | - |
| Meth-14 | Supplement: R1 weight, no GP on SN critics, incomplete factorial | C | PREREG:724, 742-744; SPEC-FU:268 | γ = 10 | - |
| Meth-15 | one confirmatory contrast (two results, one per decoder) | C | SPEC-FU:45, 148 | - | - |
| Meth-15 | SN pooled critic minus SN discriminator with the critic's optimiser, both at one step | C | SPEC-FU:235-237 | - | - |
| Meth-15 | they differ only in the loss and its target | C | SPEC-FU:236-238 | - | - |
| Meth-15 | "Biology is read at b\* as in the primary comparison" | W | SPEC-FU:99-103, 88, 209, 217 | "This is the R2/R3 reading ..., not P1's": three points, no (b0, c0) anchor; all 3 of 3 leave-one-seed-out readings required | "Biology is read at b\* by interpolation along the three strengths (with no plain-scVI starting point, unlike the primary comparison) where they bracket b\*" |
| Meth-15 | where the three strengths bracket b\* | C | SPEC-FU:96-101 | - | - |
| Meth-15 | "otherwise at the matched strength alone and labelled fixed-strength" | W (meaning) | SPEC-FU:105-114 | otherwise the task is *not evaluable* for the contrast; the fixed-strength difference is reported for every task, at each arm's own inherited strength, and is not a matched comparison | "otherwise the task is not evaluable; for every task the difference at the matched strengths is also reported, labelled fixed-strength" |
| Meth-15 | same 0.01 margin and 0.95 rule | C | SPEC-FU:140, 145 | - | - |
| Meth-15 | Holm over the two companion-test results | C | SPEC-FU:149-153 | m = 2 | Add "(it cannot reject: smallest p 0.25 with three tasks)" |
| Meth-15 | explanation rule: primary declared, same direction, every regularisation and step contrast smaller | C | SPEC-FU:258-266 | "smaller" = absolute pooled posterior mean; a contrast with no estimate counts as not smaller; the reading is descriptive | Optional: "the contrast has a declared direction of the same sign" |
| Meth-15 | \res "each setting minus plain discriminator" | U | SPEC-FU:241-246 (registered secondary contrasts) | SN - none, R1 - none (JS); SN - GP at 1 and 5 (W1); GP 5-1, GP 10-1, SN 5-1; divergence with own optimiser; optimiser within JS+SN; W1 GP 5 steps - JS 1 step | "the registered secondary contrasts (regularisation, optimiser and step count)" |
| Meth-15 | Supplement: no direction with fewer than three evaluable tasks | C | SPEC-FU:124-128 | "a pool reduced to 1 or 2 ... is inconclusive by construction" | - |
| Meth-16 | discriminator arm inherits the plain discriminator's strengths; learning rate ten times lower | C | PREREG:750-751; FDD:129 | 1e-4 vs 1e-3 | - |
| Meth-16 | pooled-critic arm inherits the 5-step critic's strengths, never piloted | C | PREREG:752-754 | - | - |
| Meth-16 | either may never reach b\*, so the contrast could not be read | C | PREREG:751-754 | - | - |
| Meth-16 | pre-check not run | C | PREREG:708; x6_precheck folder (manifests and tags only) | "Planned and not done" | - |
| Meth-16 | 36 models, 18 per arm | C | PRECHK | 18 discriminator_sn_critic_opt, 18 pooled_sn (n_critic 1) | - |
| Meth-16 | small ATAC, immune, sim1; both decoders; seed 100 | C | PRECHK task, cond, seed columns | all 36 rows seed 100 | - |
| Meth-16 | at the three strengths the pilot gave the base method | C | PRECHK lam vs r4 `cells[task\|cond\|base].lo/matched/hi` | 12 of 12 task-decoder-arm cells identical | - |
| Meth-16 | rule fixed before any pre-check fit | C | PREREG:756 | registered text; PLAN.md was not among the sources | - |
| Meth-16 | four pairs not low-signal in the pilot | C | PREREG:757; r4 `targets` | atac_small 0/1, immune 0, sim1 0 | - |
| Meth-16 | ≥3 below at all three → up; ≥3 above → down; else keep | C | PREREG:758-759 | "never reach b\* from below / from above" | State which b\* (see Meaning problems, P10) |
| Meth-16 | "the sequence 0.1, 0.3, 1, 3, 10" | W (minor) | PREREG:34, 765 | "..., 0.1, 0.3, 1, 3, 10, ..." (continues both ways; pre-check strengths include 0.03 and 3,000) | "one step along the half-decade sequence (..., 0.03, 0.1, 0.3, 1, 3, ...)" |
| Meth-16 | one step, no second pre-check | C | PREREG:766; SPEC-FU:202 | - | - |
| Meth-17 | theory: vanishing gradients when batches are perfectly separable | C-sec | outline.tex:57 (Intro-3, cites arjovsky2017towards) | paper not opened in this check | - |
| Meth-17 | question wording | C | paper.tex:231; R0X10:61 | - | - |
| Meth-17 | no new fits | C | R0X10:39; paper.tex:429 | - | - |
| Meth-17 | 80 plain-scVI embeddings (8 x 2 x 5) | C | R0X10:46; MAN5 X1 none = 80 | - | - |
| Meth-17 | per-epoch logs of the 2,880 penalty fits | C | R0X10:65-67 | 8 x 2 x 6 x 6 x 5 | - |
| Meth-17 | no strengths or seeds selected | C | R0X10:19 | - | - |
| Meth-17 | 15-nearest-neighbour, five-fold cross-validation | C | R0X10:47-50 | stratified, fold seed = run seed | - |
| Meth-17 | AUC averaged over batches | C | R0X10:53 | macro one-vs-rest | - |
| Meth-17 | 0.5 chance, 1 perfect | C | R0X10:9 | - | - |
| Meth-17 | nearly separable if ≥ 0.95 in all five seeds | C | R0X10:56-57 | - | - |
| Meth-17 | gradient norm before multiplying by strength, w.r.t. the embedding | C | R0X10:30; paper.tex:431 | per-cell norm w.r.t. the latent the adversary reads, averaged per epoch | - |
| Meth-17 | discriminator accuracy at naming each cell's batch | C | R0X10:32 | in-sample, after its update | Supplement: in-sample accuracy overstates held-out accuracy (R0X10:85) |
| Meth-17 | vanishes early: ≤10% of own peak from within the first 10% of epochs | C | R0X10:71-73; paper.tex:432 | T ≤ 0.10 | - |
| Meth-17 | descriptive, no pooled test, no claim | C | R0X10:77; paper.tex:432 | - | - |
| Meth-18 | downstream use at matched batch removal | C | paper.tex:66, 433 | - | - |
| Meth-18 | no new fits | C | R0X10:3; paper.tex:429 | - | - |
| Meth-18 | matched-strength embeddings, five seeds | C | R0X10:23, 96-98 | matching-rule matched point on the main benchmark | - |
| Meth-18 | chosen by batch score, never by label-transfer accuracy | C | R0X10:23, 132 | - | - |
| Meth-18 | plus plain scVI and the scvi-tools classifier, labelled unmatched | C | R0X10:23, 96-98 | "not matched" | - |
| Meth-18 | each batch held out; 15-NN trained on the others | C | R0X10:100-101 | uniform weights | - |
| Meth-18 | cell types in only one batch dropped, from labels alone | C | R0X10:21, 93-95 | - | - |
| Meth-18 | F1 = harmonic mean of precision and recall | C | R0X10:9 | - | - |
| Meth-18 | per cell type, averaged over types | C | R0X10:102-103 | pooled out-of-fold predictions; macro mean | - |
| Meth-18 | gap to b\* printed | C | R0X10:23, 105 | - | - |
| Meth-18 | no test or decision rule | C | R0X10:107 | "No test, no p-value" | - |
| Meth-18 | low-signal pairs tabulated, left out of summaries | C | R0X10:108-109 | - | - |

## 2. Meaning problems

Each item is a sentence whose numbers are right but whose description of a rule is wrong or would mislead a reader.

- **P1 (Meth-15, wrong).** "as in the primary comparison". The follow-up curve has three points and no plain-scVI starting point. SPEC-FU:102-103 says outright that this is not the primary comparison's reading. It also needs all three leave-one-seed-out readings (SPEC-FU:88), where the main benchmark needs 4 of 5.
- **P2 (Meth-15).** The fixed-strength fallback reads as if a fixed-strength value could stand in for the confirmatory reading. Under SPEC-FU:105-114, such a task is not evaluable and drops out of the pooled confirmatory result. The fixed-strength difference is reported for every task, with the two arms at different inherited strengths, so it is not a matched comparison.
- **P3 (Meth-15 result placeholder).** "each setting minus plain discriminator" promises contrasts that were not registered. Only SN minus none, R1 minus none and the W1 GP 5-step minus JS 1-step contrast use the plain discriminator (SPEC-FU:241-246). Any other setting-minus-discriminator contrast would be exploratory.
- **P4 (Meth-7).** The freeze date. The 16 October tag froze the version-3 amendments and the analysis. Those amendments were decided after the pilots had been fitted and scored, and after 19 main-benchmark fits had been scored but not read (PREREG:3, 149-153; paper.tex:836). "Examined" carries a project-specific definition (a person read the value; values shown only to an AI agent are disclosed, not counted). A reviewer will not know this, so state it.
- **P5 (Meth-6).** Range rule. In 3 of 8 grids the six strengths extend one or two half-decade steps above the over-correction or collapse end, because that span held fewer than six grid values (PREREG:65): discriminator conditioned (span 0.1-3, plus 10 and 30), MMD unconditioned (span 3-300, plus 1,000) and critics unconditioned (span 0.3-30, plus 100).
- **P6 (Meth-6, Meth-7 topics).** "each method its own strength range" and "each method at its six pilot strengths". The ranges are per penalty family and decoder, and the three critics share one. Plain scVI and the scvi-tools classifier have no strength grid.
- **P7 (Meth-14).** "smoothing" is used for slope control (gradient penalty, spectral normalisation, R1), but Meth-2 uses "smoothing" for the Sinkhorn entropic regularisation. Elsewhere the outline says "regularisation" (Abstract, Intro-5).
- **P8 (Meth-14 gloss).** "the critics' learning rate and momentum settings". The swap replaces the discriminator's whole Adam setting (learning rate 1e-3, ε 0.01, weight decay 1e-6) with the critics' setting (learning rate 1e-4, betas 0 and 0.9) (paper.tex:584; FDD:129). Betas are not only momentum: beta1 = 0 means no first-moment momentum. The safe wording is "the critics' Adam settings".
- **P9 (Meth-16).** The sequence is written as a closed list, but it continues in both directions (PREREG:34, 765). The pre-check strengths already include 0.03 and 3,000.
- **P10 (Meth-16, question, not an issue).** The text does not say which b\* the pre-check compares against. The pre-check runs on pilot tasks with pilot seed 100, which implies the pilot record's b\* (PREREG:757 counts pairs "not low-signal in the pilot record"). The registered texts I read do not state it; PLAN.md was not among the sources. State "the pilot's b\*" if that is what PLAN.md says.
- **P11 (Meth-11, minor).** The power figures are for the pilot's matched-strength contrast on all six pilot task-decoder pairs, including the simulation and the two low-signal pairs. They are not for the b\* reading (paper.tex:625; SPEC:116-119). The 0.104 power is for six real tasks; it is 0.088 for four families.
- **P12 (Meth-10, minor).** "whether the critic reaches b\*": name the reference critic, the one in the primary comparison (SPEC:243-245).
- **Gloss checks.** Spectral normalisation and R1 agree with the project evidence record (evidence.md:101-102, which quotes Miyato et al. Eq. 8 and Mescheder et al. Eq. 9). I did not open either paper myself (web tools denied). Clopper-Pearson as "exact binomial 95% intervals" is the standard definition; the project record holds metadata only (evidence.md:40). The glosses for F1, AUC and posterior probability are correct.

## 3. Completeness gaps

| # | Paragraph | Gap | Fix |
|---|---|---|---|
| 1 | Meth-6 | Two range endpoints lie outside the stated pilot grid: 0.03 (discriminator, unconditioned) and 10,000 (Sinkhorn, conditioned). They come from 36 grid-edge extension fits not counted in the 1,098: 18 discriminator fits at 0.01 and 0.03, and 18 Sinkhorn fits at 10,000 and 30,000 (r1 `rounds`, `a1_grid`; r1_pass2/step.json n_rows 1,134 = 1,098 + 36). The supplement bullet does not explain this in the main text | Add one clause: "where a range reached the grid edge, two further strengths were fitted (36 fits), which is why the discriminator (unconditioned) range starts at 0.03 and the Sinkhorn (conditioned) range ends at 10,000" |
| 2 | Meth-6 | The Sinkhorn pilot fits were redone after a solver correction, and the Sinkhorn ranges come from the refits (paper.tex:834-835; r1_pass2/changes_vs_tag.json). This is not mentioned in Meth-6 to Meth-18 | Add to the Meth-6 supplement list: "Sinkhorn pilot refitted after a solver correction" |
| 3 | Meth-14 | The text does not say that the follow-up strengths are matched on the main benchmark, so the follow-up runs only after the main benchmark is scored, or that the follow-up has no strength-0 fits and so reports failure counts, not collapse rates (SPEC-FU:65, 159-163) | One clause each, or supplement |
| 4 | Meth-15 | The companion test cannot reject with three tasks (smallest p 0.25; SPEC-FU:153). The registered secondary contrasts and the biology-based width on three strengths (SPEC-FU:241-252) are not named | Name them in one line or in the supplement |
| 5 | Meth-16 | Which b\* is used (P10) | State it |
| 6 | Meth-13 | Simulations take the strength chosen on all six real tasks; only failure-free strengths are eligible (SPEC:254-255) | Supplement |
| 7 | Meth-18 | The two paired contrasts reported (each method minus plain scVI; reference minus discriminator) are not mentioned (R0X10:106-107) | Optional |

Every experiment or analysis has motivation, design, measurement or decision rule, and an outcome: a pilot outcome (Meth-6, Meth-8, Meth-11), a \res placeholder (Meth-7 to Meth-10, Meth-12, Meth-13, Meth-15 to Meth-18), or "has not been run" (Meth-16). Meth-14 has no outcome line of its own, but its outcome placeholder is in Meth-15. **No reported outcome comes from the blinded main benchmark or follow-up.** 0.031 and 0.0083 are design properties; 0.131, 0.182, 0.104, 0.019 and 0.007 are pilot values.

## 4. Style problems

- **Code labels.** None of R4, X1, X6, A1, SI- or NB- appears in Meth-6 to Meth-18. "R1" appears only as the name of Mescheder's gradient penalty (outline.tex:184, 186), which is allowed. It is not the rule name, but the supplement uses "rule R1", so consider "R1 penalty (Mescheder et al.)" to keep the two apart. The \res placeholders carry outline cross-references (Res-1, Res-2, Res-4 to Res-10); these are internal and must go when results are filled in.
- **No commit ids, file paths or tag names** in these paragraphs.
- **Acronyms not spelled out within Meth-6 to Meth-18:** scVI (first use Meth-6) and scIB (Meth-12). Both are spelled out earlier in the outline (Abstract item 2; Meth-4 topic). ATAC, MMD, PCA, AUC and ROC are spelled out at first use here.
- **Citation keys:** all 9 keys used here are in refs.bib: clopper1934use, corani2017statistical, korsunsky2019fast, hie2019efficient, hrovatin2025integrating, xu2021probabilistic, miyato2018spectral, mescheder2018training, arjovsky2017towards.
- **LaTeX lists:** 13 `\begin{itemize}` and 13 `\end{itemize}` in lines 123-217, and no `\item` outside a list.
- **Paragraph length:** the outline's target is 100-150 words per paragraph (outline.tex:28). By the writer's own hand count, 9 of 13 paragraphs exceed 150 words (Meth-9 175, Meth-10 190, Meth-11 194, Meth-13 197, Meth-14 152, Meth-15 151, Meth-16 174, Meth-17 214, Meth-18 166). I did not recount (shell denied).

## 5. Not read

| Source | What was opened |
|---|---|
| paper.tex | Lines 40-119, 215-454 and 560-845 read. Not read: 1-39 (preamble and title), 120-214 (Introduction and comment block), 455-559 (supplement S-M1 to S-M5) |
| R0_X10_DEFAULTS.md | Read in full |
| PREREG.md (tag export) | Lines 1-230 and 640-790 read. Lines 231-639 (older section 9 entries and disclosure) not read |
| ANALYSIS_SPEC.md (tag export) | Lines 1-300 read. Lines 301-607 (sections 12 to 16, Appendix A) not read |
| ANALYSIS_SPEC_FOLLOWUPS.md | Read in full |
| amendment_20261017_followups.md | Lines 1-20 only (superseded draft, PREREG:703) |
| followup_design_decisions_20261017.md | Searched for keywords only (lines 9-33, 85, 129, 155-184 seen). Line 129 used |
| docs/prereg/*.json (tag export) | r1_record, r4_a1_record and r2r3_record checked byte-identical (SHA-256) to the records in force. r1_record_round1.json and r4_a1_record_round1.json not opened |
| manifests/ (tag export) | v5 parsed in full (all 5,280 rows, counted by experiment, arm, decoder, seed, step count and strength label). v1 to v4 and the extension manifests not opened |
| r1_pass2/r1_record.json | constants, all 8 `families` entries (fields other than `blocks`), `power` (fields other than `blocks`) read. Per-block detail not opened. changes_vs_tag.json and step.json read in full |
| r4_pass2/r4_a1_record.json | Top-level fields, `targets` (all 6), and `cells` for the pre-check base arms and a few discriminator cells read. Other cells not opened. changes_vs_tag.json and step.json read in full |
| r23/r2r3_record.json | Checksum only (it bears on Meth-3, outside the paragraphs checked) |
| x6_precheck/manifest.tsv | All 36 rows read. manifest_tags.txt: first 3 lines. The 18-row superseded manifest not opened |
| refs.bib | Searched for the 9 keys used (entries read to the title line) |
| outline.tex | Lines 1-220 read |
| methods_6_10_factcheck.md | Read in full, used only as a pointer |
| evidence.md (not a listed source) | Lines 99-104 plus a keyword search, used for the three glosses |
| Cited papers (Miyato 2018, Mescheder 2018, Clopper and Pearson 1934, Arjovsky and Bottou 2017) | Not retrieved: WebFetch and WebSearch were denied in this session |
| Main-benchmark and follow-up scores, latents, results, X1_local/ | Not opened, as required |

## 6. Question outside the text check

In the pilot record, the discriminator with the unconditioned decoder on small ATAC is matched at strength 1,000, with batch score 0.496 against b\* 0.522 (r4 `cells["atac_small|0|discriminator"]`). This is outside that family's main-benchmark range (0.03-10). The discriminator pre-check arm therefore runs this pair at 300, 1,000 and 3,000, while the shift it decides is applied to strengths matched on the main benchmark. The text is accurate about this. Whether a pilot window that far from the main grid gives a meaningful up/down signal is a design question for the authors, not a text error.
