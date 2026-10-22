# Fact-check of the rewritten Methods paragraphs (outline.tex Meth-6 to Meth-18)

Written 2026-10-21. The five old paragraphs Meth-6 to Meth-10 became thirteen, Meth-6 to Meth-18 (old Meth-6 -> Meth-6 and Meth-7; old Meth-7 -> Meth-8, Meth-9, Meth-10; old Meth-8 -> Meth-11; old Meth-9 -> Meth-12 and Meth-13; old Meth-10 -> Meth-14 to Meth-18). Nothing else in `outline.tex` changed except the Methods heading line.

Source aliases (line numbers are lines of the file as read on 2026-10-21):

| Alias | File |
|---|---|
| paper.tex | `paper/paper.tex` (845 lines as read) |
| outline.tex | `paper/outline.tex` |
| R0X10 | `paper/R0_X10_DEFAULTS.md` |
| PREREG | `code/meth_rewrite_20261022/prereg-x6-v1/docs/PREREG.md` |
| SPEC | same folder, `docs/ANALYSIS_SPEC.md` |
| SPEC-FU | same folder, `docs/ANALYSIS_SPEC_FOLLOWUPS.md` |
| MAN5 | same folder, `manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv` (counted by pattern with ripgrep, not read row by row) |
| counts.csv | same folder, `docs/manifest_v5_counts.csv` |
| R1pass2 | `tier12_v3/rules/r1_pass2/r1_record.json` |
| R4pass2 | `tier12_v3/rules/r4_pass2/r4_a1_record.json` |
| PRECHK | `tier12_v3/x6_precheck/manifest.tsv` |

Checks that are not a quotation from a file are marked "(arithmetic)" or "(count)". Each number below was read from the line cited in this session; none was retyped from memory or from the old outline.

## 1. Statements and values

### Meth-6 (pilot)

| Statement in the outline | Source file:line | Value exactly as shown |
|---|---|---|
| Pilot used ten strengths from 0.1 to 3,000 | paper.tex:61; R1pass2:3-14 | "ten strengths (0.1 to 3000)"; `A1_BASE_GRID` = 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000 |
| Pilot tasks: small ATAC, immune, simulation 1 | paper.tex:61; PREREG:48 | "atac_small, immune and sim1"; "Tasks atac_small, immune, sim1" |
| Both decoders, seeds 100--102 | paper.tex:61; PREREG:48 | "both decoders, seeds 100--102"; "seeds 100, 101, 102" |
| 1,098 pilot fits | paper.tex:695; counts.csv:3; PREREG:13 | "1,098"; "A1,1098,1098"; (arithmetic) 3 tasks x 2 decoders x 3 seeds x (1 + 6 x 10) = 1,098 |
| Six penalties: discriminator, reference, pooled and barycenter critics, MMD, Sinkhorn divergence | PREREG:48 | "arms discriminator, reference, pooled, barycenter, mmd, sinkhorn" |
| Six log-spaced strengths per range | PREREG:64; R1pass2:67 | "6 points evenly spaced in log λ"; `"N_GRID": 6` |
| Range starts at the largest strength where every pilot task still shows no effect | PREREG:63 | "at λ_{i_lo} every block shows no effect" |
| Range ends where every task has over-corrected (batch removal stops rising) | PREREG:59, 63 | "the onset of over-correction (batch removal no longer increases, within noise)"; "at λ_{i_hi} every block has reached over-correction or collapse" |
| Collapsed = a fit fails or biology 0.10 or more below plain scVI | PREREG:57, 40; R1pass2:48 | "n_fail ≥ 1, or Δc ≤ −C"; "C (collapse: bio loss vs λ=0) \| 0.10"; `"COLLAPSE_BIO": 0.1` (the pilot applies it to the seed-mean change) |
| The three critics share one range | PREREG:50 | "The three W1 critics share a grid" |
| Discriminator, conditioned decoder: 0.1--30 | R1pass2:964-971 (`js\|1`; labelled conditioned at paper.tex:769) | 0.1, 0.3, 1.0, 3.0, 10.0, 30.0 |
| Discriminator, unconditioned: 0.03--10 | R1pass2:550-557 (`js\|0`; paper.tex:770) | 0.03, 0.1, 0.3, 1.0, 3.0, 10.0 |
| Critics, conditioned: 0.1--300 | R1pass2:4984-4991 (`w1\|1`; paper.tex:771) | 0.1, 0.3, 3.0, 10.0, 100.0, 300.0 |
| Critics, unconditioned: 0.3--100 | R1pass2:3836-3843 (`w1\|0`; paper.tex:772) | 0.3, 1.0, 3.0, 10.0, 30.0, 100.0 |
| MMD, conditioned: 3--3,000 | R1pass2:1792-1799 (`mmd\|1`; paper.tex:773) | 3.0, 10.0, 30.0, 300.0, 1000.0, 3000.0 |
| MMD, unconditioned: 3--1,000 | R1pass2:1378-1385 (`mmd\|0`; paper.tex:774) | 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0 |
| Sinkhorn, conditioned: 0.3--10,000 | R1pass2:2688-2695 (`sinkhorn\|1`) | 0.3, 3.0, 30.0, 100.0, 1000.0, 10000.0 |
| Sinkhorn, unconditioned: 0.3--300 | R1pass2:2206-2213 (`sinkhorn\|0`) | 0.3, 1.0, 3.0, 30.0, 100.0, 300.0 |
| Supplement bullet: noise threshold for no effect | PREREG:54, 38-39 | "δ_b(λ) = max(δ_min, k σ_b / √n(λ))"; δ_min 0.01; k 3 |
| Supplement bullet: extra fits at grid edges | PREREG:44, 61; R1pass2:545-548, 2683-2686 | "2 points per round, ≤ 2 rounds per edge (R1)"; `js\|0` rounds `"high": 0, "low": 1`; `sinkhorn\|1` rounds `"high": 1, "low": 0` |

### Meth-7 (main benchmark)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Eight tasks, both decoders, seeds 0--4 | paper.tex:64; PREREG:197 | "8 tasks, both decoders, seeds 0--4"; "8 tasks, both decoders, seeds 0 to 4" |
| Plain scVI 80 fits | R0X10:34; MAN5 (count) | "`none` 80 (8 tasks x 2 decoders x 5 seeds, strength 0)"; 80 tags match `^X1_<task>_none_l0_` |
| scvi-tools adversarial classifier 40 fits, conditioned decoder only | R0X10:34; MAN5 (count); paper.tex:331 | "`scvi_adv` 40 (8 tasks x 5 seeds, all with the conditioned decoder"; 40 tags match `_scvi_adv_`; "conditioned decoder only" |
| Six penalties at six strengths, 480 fits each | R0X10:34; MAN5 (count) | "six arms x 480 rows (6 strengths x 2 decoders x 5 seeds x 8 tasks)"; 2,880 tags match the six arm names |
| 3,000 fits in all | paper.tex:64; PREREG:197; counts.csv:2; MAN5 (count) | "3,000 fits"; "3,000 model fits in the manifest of record"; "X1,3000,3000"; 3,000 tags start `X1_`; (arithmetic) 80 + 40 + 2,880 = 3,000 |
| Rules and analysis frozen on 16 October 2026 | paper.tex:392; outline_sources.md (old Meth-6 row) | "made on 2026-10-16 on the user's decision"; "reached GitHub at 2026-10-16T21:48:53Z"; "2026-10-16 17:48 EDT" |
| Before any main-benchmark score was examined | paper.tex:226; SPEC:25-27 | "The user stated on 2026-10-15 that nobody had read an X1 or X13 score value"; "A score is examined when a person reads its value or an analysis script uses it" |
| The analysis is run once | paper.tex:415; SPEC:73-76 | "One analysis run, after all 3,000 X1 and 408 X13 rows are scored or failed"; "runs once, after all 3,000 X1 rows and the 408 X13 baseline rows are scored or failed" |
| Follow-up rule registered before any follow-up fit was scored | PREREG:708-713; SPEC-FU:3-7 | "Planned and not done: the 36 pre-check fits ... and the new code tag `prereg-x6-v1`, which will freeze this entry ... before R4-x1 runs and before any X1 result is examined"; "No X1, X13 or follow-up score was read to prepare this entry" |

### Meth-8 (matched level)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Strength has a different scale for each method | PREREG:50 | "have different loss scales" |
| b* halfway | PREREG:95, 41 | "b0 + q (b_common − b0), q = 0.5"; "q (b* position) \| 0.5" |
| Highest batch score all six penalties reach at strengths with no failed fit | PREREG:94 | "M_a = the highest seed-mean B of arm a over its grid points with no failed seed; b_common = min over the six adversarial arms of M_a" |
| Means over five seeds | PREREG:93; SPEC:37 | "b0 = seed mean of B for λ=0"; "5 runs (seeds 0-4)" |
| Low-signal if the gap is under 0.02 | PREREG:95, 38; paper.tex:389 | "Flag `bstar_low_signal` when b_common − b0 < 2 δ_min"; δ_min 0.01; "< 0.02" |
| Low-signal pairs reported separately | paper.tex:389; SPEC:123-124 | "reported separately"; "shown in a separate table and never pooled" |
| Pilot: immune, conditioned, flagged; gap 0.019 | R4pass2:5167-5180; paper.tex:342 | `b0` 0.6381673018004433, `b_common` 0.6569909962526824, `bstar_low_signal` true; (arithmetic) difference 0.01882; paper.tex "0.0188" |
| Pilot: simulation 1, conditioned, flagged; gap 0.007 | R4pass2:5195-5208; paper.tex:342 | `b0` 0.8119601403099188, `b_common` 0.8188775558554181, `bstar_low_signal` true; (arithmetic) difference 0.00692; paper.tex "0.0069" |
| The other four pilot pairs are not flagged (implied by "two" being named) | R4pass2:5137, 5151, 5165, 5193 | `"bstar_low_signal": false` |
| Biology trades off against batch removal (premise) | outline.tex Res-3 topic, Intro-4 | Taken from the outline's own wording, no new source |

### Meth-9 (primary comparison, trade-off area)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Curve = plain scVI followed by six strengths, means over successful runs | SPEC:108-110 | "the lambda = 0 point (b0, c0) followed by its six registered points in increasing lambda, each point = mean B and mean C over its successful runs" |
| Read at the first crossing, linear interpolation | SPEC:112-115 | "the first point equal to b* (within 1e-9) or the first adjacent pair whose B values bracket b* (linear interpolation in B)" |
| Reference critic minus discriminator, per task and decoder | SPEC:127, 141 | "d = bio@b*(reference) - bio@b*(discriminator)"; "One P1 result per decoder" |
| Repeat reading five times, one seed left out | SPEC:129-133 | "repeat the whole reading 5 times, each time without one seed" |
| Not evaluable: low-signal, curve never reaches b*, fewer than four of five readings | SPEC:121-123 | "(i) it is flagged `bstar_low_signal` ... (ii) either arm's curve never reaches b*, or (iii) fewer than 4 of the 5 leave-one-seed-out readings exist" |
| Area uses six points, corner at plain scVI's batch score and biology minus 0.10 | SPEC:146-152 | "its six points"; "The corner (b_ref, c_ref) is (b0, c0 - 0.10)" |
| Area divided by the batch gap that sets b* | SPEC:148-149 | "divided by the cell's common batch range (b_common - b0)" |

### Meth-10 (collapse, local versus global)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Collapse = fails or biology 0.10 or more below same seed's plain scVI | SPEC:221-222 | "A run collapses when it failed, or when its C is at least 0.10 below the C of the same seed's lambda = 0 run" |
| Collapse rate over 30 fits per method, task, decoder | SPEC:222-223 | "over the six registered points and 5 seeds" ((arithmetic) 6 x 5 = 30) |
| Clopper--Pearson 95% intervals | SPEC:223 | "Clopper-Pearson 95% intervals" |
| Reference minus discriminator difference, margin 5 percentage points | SPEC:228-231; paper.tex:406 | "margin 0.05 (5 percentage points)"; "margin 5 points" |
| Local = iLISI, kBET, graph connectivity; global = PCR, batch ASW | SPEC:237-238 | "local = mean(iLISI, kBET, graph_conn) ... global = mean(PCR_batch, ASW_label/batch)" |
| Batch score = (3 local + 2 global)/5 | SPEC:238-239 | "B = (3 local + 2 global)/5 exactly" |
| One signed question: more local mixing; same crossing as primary; exact sign test over real tasks and pooled model | SPEC:243-246 | "does the critic get there through more local mixing"; "read at the same crossing as P1"; "exact two-sided sign test over the real datasets and the section 4 model with r = 0.01" |

### Meth-11 (pooling, rank test, power)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Six real tasks | SPEC:100 | "pancreas, lung, immune, immune_hum_mou, atac_small, atac_large" |
| All immune cells are in the human--mouse immune task | paper.tex:363; SPEC:101 | "immune_hum_mou contains all 33,506 immune cells"; "all 33,506 immune cells are in immune_hum_mou" |
| Small ATAC cells are in the large ATAC task | paper.tex:364 | "Every one of the 11,270 atac_small cells is also in atac_large" (see Conflicts, item 9) |
| Simulations and low-signal pairs never pooled | SPEC:101-102, 123-125 | "sim1 and sim2 are reported per task and never pooled with real tasks"; "Readings of low-signal cells are shown in a separate table and never pooled" |
| Model: task difference = overall + family effect + task effect; each weighted by its own uncertainty | SPEC:167-169 | "d_t ~ N(theta_t, s_t^2) with s_t ... (treated as known), theta_t = Delta + u_family(t) + v_t" |
| Direction if probability at least 0.95 beyond margin 0.01 | SPEC:203-207 | "a direction if P(Delta > r) >= 0.95 or P(Delta < -r) >= 0.95"; "r = 0.01" |
| Practical equivalence if probability within the margin at least 0.95 | SPEC:204 | "practically equivalent if P(|Delta| <= r) >= 0.95; otherwise inconclusive" |
| Wilcoxon signed-rank, Holm over six primary results | SPEC:212-215; paper.tex:409 | "the Wilcoxon companion applies Holm over all six"; "Holm's first threshold is then 0.05/6 = 0.0083" |
| Primary results = comparison, trade-off area, collapse rate, each per decoder | SPEC:212-213 | "P1 and the area for each decoder, plus P2 for each decoder: six" |
| Smallest p 0.031 | SPEC:214-215; paper.tex:409 | "2/2^6 = 0.03125"; "2/2^6 = 0.031" |
| Minimal detectable effect 0.131 (six real tasks) | R1pass2:9607 | `"mde80": 0.13066318374660563` under `real_tasks` (`"D": 6`) |
| Minimal detectable effect 0.182 (four families) | R1pass2:9596 | `"mde80": 0.1817844302210654` under `families` (`"D": 4`) |
| 80% power | PREREG:104 | "the minimal detectable effect at 80 %" |
| Power for a difference of 0.02 is 0.104 | R1pass2:9587, 9608 | `"delta": 0.02`; `"power_at_delta": 0.10435882408578023` (real tasks) |
| Equivalence within +-0.02 cannot be shown | R1pass2:9603-9606 | `"equivalence_feasible": {"0.01": false, "0.02": false}` |
| Power is from the pilot | R1pass2:9586; PREREG:104 | "P1: bio(reference) - bio(discriminator) at the R4-a1 matched lambda, unscaled"; "From A1, for the P1 contrast" |
| Supplement bullet: priors, sensitivity analyses | SPEC:169-171, 283-292 | "Priors: Delta ~ N(0, 0.1^2); tau_f, tau_d ~ half-normal(0.1)"; sensitivity list of section 11 |

### Meth-12 (good-strength width)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Selection score = 0.4 batch + 0.6 biology, unscaled, scIB weights | SPEC:258-259 | "S = 0.4 B + 0.6 C, scIB's weights without min-max scaling" |
| Does not depend on which methods are compared | SPEC:258-259 | "so S does not depend on which methods are compared" |
| Best strength has the highest mean S; five seeds | SPEC:278; SPEC:37 | "the best point maximises mean S"; "5 runs (seeds 0-4)" |
| Good range = strengths with mean S at least the lower 95% bound (t) of the best point's mean | SPEC:278-280 | "every point whose mean S is at least the lower 95% bound of the best point's mean (t with n - 1 degrees of freedom)" |
| Width in decades; zero for one strength | SPEC:280-281 | "width = log10(largest lambda in the window) - log10(smallest), in decades (0 for a single point)" |
| Descriptive | SPEC:281 | "Descriptive." |

### Meth-13 (baselines, single-strength tables, best of k)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Question: does any adversary improve on plain scVI | paper.tex:256 | "does any adversary improve on plain scVI" |
| Harmony six settings, theta 0--8 | paper.tex:440 | "θ ∈ {0, 0.5, 1, 2, 4, 8} at 10 PCs, once each" |
| Scanorama six settings, neighbours 2--80 | paper.tex:441 | "knn ∈ {2, 5, 10, 20, 40, 80} at 10 dimensions, once each" |
| sysVI six settings, cycle weight 1--50 | paper.tex:444 | "cycle weight ∈ {1, 2, 5, 10, 20, 50}, seeds 0--4" |
| scANVI one setting, uses labels; PCA one setting | paper.tex:443, 442; SPEC:268-269 | "scANVI ...: scIB's protocol, scVI then scANVI with all labels; seeds 0--4"; "PCA: once at 10 (and 50) components"; "k = 1 methods (scANVI, scVI lambda = 0, scvi_adv); PCA is the uncorrected anchor" |
| Seeds 0--4 neural, one run otherwise | paper.tex:87 | "the neural X13 baselines (scANVI, sysVI): 0--4, Harmony, Scanorama and PCA once at seed 0" |
| 408 fits including sensitivity settings | paper.tex:415, 699; MAN5 (count) | "408 X13 rows"; "X13 \| 408"; counts of tags: harmony, scanorama, pca 128; scanvi 40; sysvi 240; (arithmetic) 128 + 40 + 240 = 408 |
| Single strength = best mean S on real tasks of other families; related tasks never choose for each other | SPEC:250-254 | "the real datasets of the other families (leave-one-family-out, so immune and immune_hum_mou, which share cells, never choose each other's lambda"; "registered point that maximises the selection score S averaged over the training datasets" |
| Own best strength shown as optimistic bound | SPEC:255-256 | "next to the per-dataset best lambda, labelled an optimistic bound" |
| Expected best of k settings drawn at random, k = 1 to 6 | SPEC:267-272 | "each adversarial arm (6 lambda), Harmony (6 theta ...), Scanorama (6 knn ...), sysVI (6 cycle weights)"; "Expected best of k configurations drawn at random: E_k = ..."; "k = 1..N" |
| Equal numbers of tries | SPEC:264; PREREG:203 | "Giving every method the same number of tries is the user's decision (SI-68, SI-77)" |
| Optimistic; scANVI own stratum | SPEC:271-272 | "labelled optimistic (chosen and reported on the same data). scANVI is shown in its own label-aware stratum" |
| Descriptive | paper.tex:258 | "its results are reported descriptively" |

### Meth-14 (follow-up design)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Tasks small ATAC, immune, pancreas | PREREG:721; paper.tex:422 | "3 tasks (atac_small, immune, pancreas)" |
| Both decoders | PREREG:721-722 | "x 2 decoders (SI-93" |
| Seeds 10--12 | PREREG:723; SPEC-FU:179 | "x 3 seeds (10, 11, 12; SI-26)"; "exactly 10, 11, 12 in every point (SI-26)" |
| Nine settings, three strengths each, 486 fits | PREREG:721-723; MAN5 (count); counts.csv:4 | "x 9 settings of arm and adversary steps per update x 3 penalty strengths x 3 seeds"; 486 tags start `X6_`; "X6,486,486"; (arithmetic) 3 x 2 x 9 x 3 x 3 = 486 |
| Strengths: matched strength of the base method and its two grid neighbours | PREREG:727-728; SPEC-FU:32-36 | "`matched_lo`, `matched` and `matched_hi` (the point whose batch score is closest to the target b* and its two grid neighbours)"; "a variant arm takes the strengths of its base arm" |
| Discriminator plain, with spectral normalisation, with R1 penalty, with spectral normalisation and the critic's optimiser, each at 1 step | PREREG:723-725 | "JS discriminator with no penalty, with spectral normalisation (SN), with the R1 penalty (weight 10 ...) and with SN and the critic's optimiser (the new arm), each at 1 step" |
| Pooled critic with gradient penalty at 1, 5, 10 steps; with spectral normalisation at 1 and 5 | PREREG:725-726 | "the pooled Wasserstein-1 (W1) critic with a gradient penalty (GP) at 1, 5 and 10 steps; the pooled W1 critic with SN at 1 and 5 steps" |
| The critic's optimiser = the critics' learning rate and momentum settings | PREREG:731-732; paper.tex:584 | "the critic's Adam settings (learning rate 1e-4, betas (0, 0.9)) instead of the discriminator's (learning rate 1e-3)" |
| Follow-up question: divergence versus smoothing, optimiser, steps | SPEC-FU:226-228 | "due to the divergence (W1 vs JS), or to the Lipschitz control (gradient penalty, spectral normalisation, R1), the optimiser, or the number of adversary steps" |
| Follow-up uses the pooled critic, not the reference critic | SPEC-FU:269-270 | "the X6 critic is the pooled one, while P1 uses the reference critic" |
| R1 penalty acts on the output's slope at the cells (plain gloss) | paper.tex:582 | "(γ/2)(1/n)Σ_i ‖∇_z D_{b_i}(z_i)‖²" |
| Spectral normalisation: "each layer rescaled so its output cannot change steeply" (plain gloss) | paper.tex:583 gives only "every head layer spectrally normalised" | The gloss is the general definition of spectral normalisation (Miyato et al. 2018); the cited paper was not retrieved in this task |
| Supplement bullet: R1 weight 10; no gradient penalty on spectrally normalised critics; incomplete factorial | PREREG:724-725, 742-746; SPEC-FU:268 | "weight 10"; "`with_gp=not spectral_norm`"; "X6 is an incomplete factorial" |

### Meth-15 (confirmatory contrast and decision rule)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Contrast: spectrally normalised pooled critic at 1 step minus spectrally normalised discriminator with the critic's optimiser, one per decoder | SPEC-FU:235-237; PREREG:771-773 | "W1+SN at 1 step (`pooled_sn`, n_critic 1) minus JS+SN with the critic's optimiser (`discriminator_sn_critic_opt`), per decoder: 2 results" |
| They differ only in the loss and its target | SPEC-FU:236-238 | "these two differ only in the loss ... and the target" |
| Biology read at b* where three strengths bracket b*, otherwise at fixed strength | SPEC-FU:96-104, 109-111 | "interpolate biology at b* only where the cell's own fits bracket b*"; "label 'fixed-strength, not matched batch removal'" |
| Margin 0.01; threshold 0.95 | SPEC-FU:140, 145; PREREG:773-774 | "r = 0.01"; "a direction if P(Delta > r) >= 0.95"; "declaration threshold 0.95" |
| Holm over the two results | SPEC-FU:149-153; PREREG:774 | "Holm's adjustment over the 2" |
| Explanation rule: P1 direction declared, contrast same sign, regularisation and step contrasts smaller | SPEC-FU:258-266 | "(a) P1 for that decoder has a declared direction ...; (b) the confirmatory contrast has a declared direction with the same sign as P1's ...; (c) the regularisation contrasts ... and the step contrasts ... are each smaller" |
| Supplement bullet: no direction declared with fewer than three evaluable tasks | SPEC-FU:124-128 | "a pool reduced to 1 or 2 by a low-signal cell is inconclusive by construction" |

### Meth-16 (strength pre-check, planned)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Discriminator arm inherits the plain discriminator's strengths; learning rate ten times lower | PREREG:750-751; SPEC-FU:190-191 | "The new arm inherits the strengths of the plain discriminator, which is trained at a ten times higher learning rate" |
| Pooled-critic arm inherits the five-step critic's strengths and was never piloted | PREREG:752-754; SPEC-FU:192-193 | "inherits the strengths of the main pooled critic (gradient penalty, 5 steps) and was never piloted" |
| Either may never reach b* and the contrast cannot be read | PREREG:751-754 | "its batch score never reaches b* and the confirmatory result cannot be read" |
| 36 models, 18 per arm | PREREG:754-755; SPEC-FU:195; PRECHK:3-38 | "36 fits, 18 per arm"; "(36 fits: 18 per arm"; (count) 36 data rows: lines 3-20 `discriminator_sn_critic_opt`, 21-38 `pooled_sn` |
| Tasks small ATAC, immune, simulation 1; both decoders; seed 100 | PREREG:755; PRECHK | "atac_small, immune and sim1, both decoders, 3 strengths, pilot seed 100"; `task` and `cond` and `seed` columns |
| Strengths = the pilot's three strengths for the base method | PRECHK:6-8 vs R4pass2:87-98; PRECHK:9-11 vs R4pass2:159-170; PRECHK:18-20 vs R4pass2:377-388 | PRECHK atac_small cond 1 `discriminator_sn_critic_opt` 0.3, 1, 3 = R4pass2 `atac_small\|1\|discriminator` lo 0.3, matched 1.0, hi 3.0; immune cond 0: 0.03, 0.1, 0.3 = lo 0.03, matched 0.1, hi 0.3; sim1 cond 1: 100, 300, 1000 = lo 100.0, matched 300.0, hi 1000.0 (other cells not compared) |
| Rule fixed before any pre-check fit | PREREG:756, 782-790 | "applies a rule fixed before any pre-check fit" |
| Four counted pairs = pilot pairs not low-signal | PREREG:757; R4pass2:5124-5208 | "of the 4 counted cells (not low-signal in the pilot record)"; 6 pilot pairs, 2 flagged true |
| Up if at least three never reach b* from below; down if at least three never reach it from above; otherwise keep | PREREG:758-759 | "`shift_up` if at least 3 never reach b* from below, `shift_down` if at least 3 never reach it from above, otherwise `keep`" |
| Strengths move one step in 0.1, 0.3, 1, 3, 10 | PREREG:764-766 | "one step up (`shift_up`) or down (`shift_down`) the sequence ..., 0.1, 0.3, 1, 3, 10, ..." |
| No second pre-check | PREREG:766; SPEC-FU:202 | "One step only, no second pre-check" |
| Not yet run | PREREG:708 | "Planned and not done: the 36 pre-check fits" |

### Meth-17 (premise check)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Theory: perfect discriminator on separable batches gives vanishing gradients | outline.tex:57 (Intro-3) | "Perfect discriminator, disjoint supports: vanishing gradients \citep{arjovsky2017towards}" |
| Questions: nearly separable before integration; discriminator gradient vanishes early, critics' persist | R0X10:42, paper.tex:231 | "are batches nearly separable ... does the discriminator's gradient vanish early in training while the critics' gradients persist" |
| 80 plain-scVI embeddings | R0X10:46 | "the 80 X1 latents of arm `none` (8 tasks x 2 decoders x 5 seeds)" |
| 2,880 fits of the six penalties | R0X10:65-67 | "8 x 2 x 6 x 6 x 5 = 2,880 runs" |
| No strengths or seeds selected | R0X10:19, 76-77 | "All six registered strengths of each arm, every seed, no selection" |
| 15-nearest-neighbour classifier, five-fold cross-validation | R0X10:47-50 | "n_neighbors=15"; "n_splits=5" |
| Area under the ROC curve, averaged over batches | R0X10:52-53 | "`roc_auc_score(y, P, multi_class="ovr", average="macro" ...)`" |
| 0.5 chance, 1 perfect | R0X10:9 | "(0.5 = chance, 1 = perfect)" |
| Nearly separable if at least 0.95 in all five seeds | R0X10:56-57 | "only if the macro AUC is >= 0.95 in all 5 seeds of a cell" |
| Gradient norm before multiplying by strength | R0X10:30; paper.tex:431 | "the norm of the gradient of the unscaled adversary term with respect to the latent the adversary reads; the factor lambda (`d_coef`) is applied afterwards" |
| Discriminator accuracy at naming each cell's batch | R0X10:32 | "share of the minibatch's cells whose batch the head predicts correctly after its update" |
| Vanishes early if at or below 10% of own peak from within the first 10% of epochs | R0X10:71-73; paper.tex:432 | "stays at or below 10% of the run's own peak from some epoch on; T = that epoch / E"; "Early = T <= 0.10" |
| No pooled test, no claim | R0X10:77; paper.tex:432 | "No threshold on k, no pooled test"; "no claim" (SI-68 exploratory, paper.tex:467) |

### Meth-18 (label transfer)

| Statement | Source file:line | Value exactly as shown |
|---|---|---|
| Downstream use at matched batch removal | paper.tex:66, 433 | "Downstream biology at matched batch removal, minimal form"; "(downstream biology at matched batch removal)" |
| No new fits | R0X10:3, 39; paper.tex:429 | "no new fits" |
| Embeddings at the matched strength, 5 seeds; plain scVI and scvi-tools adversary labelled unmatched | R0X10:23, 96-98; paper.tex:433 | "The matched grid point of rule R4 for each task x decoder x arm: all 5 seeds' latents"; "plus plain scVI (strength 0) and `scvi_adv` ..., both labelled 'not matched'" |
| Matched strength chosen from batch scores only, never from F1 | R0X10:23, 132 | "it uses only batch scores and never F1"; "R4 matched point, chosen by batch score only" |
| Leave one batch out; 15-nearest-neighbour | R0X10:100-101; paper.tex:433 | "leave-one-batch-out; `KNeighborsClassifier(n_neighbors=15, weights="uniform")`" |
| Cell types found in only one batch dropped, decided from labels | R0X10:21, 93-95 | "cell types present in >= 2 batches"; "from the labels of the prepped file (not from any latent or score)" |
| F1 per cell type, averaged over types | R0X10:9-10, 102-103 | "F1 = harmonic mean of precision and recall of one cell type"; "macro F1 = unweighted mean over the evaluable types" |
| Gap between matched batch score and b* printed | R0X10:23, 105 | "The gap `|matched_B - b*|` is printed next to every F1" |
| No test or decision rule; low-signal pairs tabulated but not summarised | R0X10:107-109 | "No test, no p-value"; "tabulated but left out of any summary sentence" |

## 2. Conflicts between sources

| # | Disagreement | Followed |
|---|---|---|
| 1 | Size of the follow-up strength pre-check: the old outline note (outline_sources.md, old Meth-10 row) says "the 18-fit strength check"; PREREG:754 and SPEC-FU:195 say 36 fits (18 per arm); PRECHK holds 36 data rows (lines 3-38); an 18-row manifest was superseded on 2026-10-21 (file `manifest_18row_superseded_20261021.tsv`, only its first 4 lines opened) | 36 (the registered text at the follow-up tag and PRECHK) |
| 2 | Freeze status of the main registration: PREREG:3 and SPEC:3-23 say the tag `prereg-tier12-v3` "has not been made" (status lines written before the tag); paper.tex:392 says it was made on 2026-10-16 | paper.tex:392 (later, with the GitHub time); date 16 October 2026 |
| 3 | Holm first threshold: R1pass2:9613 and PREREG:105-106 hold 0.0125 (four tests); SPEC:214 and PREREG:106 (bracketed note) give 0.05/6 = 0.0083 for six primary results | 0.0083 (analysis specification in force) |
| 4 | Sinkhorn strength ranges: paper.tex:775-776 lists them as pending; the earlier tag's r1_record.json had other values (changes_vs_tag.json: sinkhorn\|0 old 0.3, 3, 10, 100, 300, 3000; sinkhorn\|1 old 0.1 to 30) | R1pass2 (new values 0.3, 1, 3, 30, 100, 300 and 0.3, 3, 30, 100, 1000, 10000). The copy of r1_record.json in the follow-up tag export holds the same values (grep of `x1_grid`, lines 2206-2213 and 2688-2695) |
| 5 | Pilot low-signal pairs: paper.tex:342 and R4pass2 agree (immune and simulation 1, conditioned); the older text in the amendment table (PREREG:217) gives rule results from before the Sinkhorn redo (R3: 13 of 20 cells, mean change -0.0014) while the record R23 gives 14 of 20 and -0.0033 (R23:2153-2156). Not used in Meth-6 to Meth-18 (they belong to Meth-3) | R23 for any use |
| 6 | Old outline Meth-9 said "six settings each" for the baselines; Harmony, Scanorama and sysVI have six, scANVI and PCA one (paper.tex:439-446; SPEC:267-269) | Sources |
| 7 | Old outline Meth-10 said the follow-up rule was "fixed and dated (21 October 2026) before any main-benchmark result was examined". No source read gives the creation date of the follow-up tag; PREREG:708-713 and SPEC-FU:3-7 say only that the tag freezes the entry "before R4-x1 runs and before any X1 result is examined" | The rewrite states no date for the follow-up registration and claims only "before any follow-up fit was scored" (supported by PREREG:708) |
| 8 | Date of dropping the other follow-ups: outline_sources.md (old row) "PREREG amendment 2026-10-18"; PREREG:663-665 "decided by the user on 2026-10-17 ... written into the ledger and the notebook on 2026-10-18". The date is not used in the rewritten text | n/a |
| 9 | Overlap of small and large ATAC: SPEC:101 "atac_small and atac_large share 3,618 cells"; paper.tex:364 "Every one of the 11,270 atac_small cells is also in atac_large ... 3,618 of them, the 10x Genomics batch, under identical names" (the 3,618 are the cells with identical names). Not re-checked against the data (not opened) | paper.tex:364 |
| 10 | Outline Res-5 (not edited) lists "sensitivity to prior scale and to dropping low-signal tasks"; the registered sensitivity analyses (SPEC:283-292) are biology without trajectory, per-run reading, failed-run handling, corner, dataset structure, prior scales 0.05 and 0.2; low-signal pairs are excluded by rule, not by a sensitivity analysis | Meth-11 Supplement bullet says "sensitivity analyses" without naming them; Res-5 left for the author |
| 11 | SPEC:3-23 describes the specification as "not yet frozen" (status text of 2026-10-16) while paper.tex:392 says it is frozen by the tag | paper.tex:392 |

## 3. Claims not backed by a retrieved source (flagged, not invented)

- "Spectral normalisation: each layer rescaled so its output cannot change steeply" and "R1 penalty: slope at the cells" are plain-word glosses of the cited methods (Miyato et al. 2018; Mescheder et al. 2018). The repository gives the formula (paper.tex:582-583) but not the gloss; the cited papers were not retrieved in this task (the citation keys are the outline's existing ones).
- "Biology trades off against batch removal" (Meth-8) is the outline's own premise (Intro-4, Res-3), not a number from a source.
- "A penalty that sometimes ruins the embedding is hard to use even when its best fit is good" (Meth-10) and "a user must tune it" (Meth-12 topic) are the reasons for the measures, written as motivation; no source states them in these words.
- The follow-up tag creation date and the claim that the follow-up registration preceded any main-benchmark result are not stated in the text (no source read gives the date).

## 4. Not read

| File or part | What was and was not opened |
|---|---|
| `paper.tex` | Lines 1-120, 200-259 and 300-845 read. Lines 121-199 (Introduction text) and 260-299 not opened. |
| `PREREG.md` | Lines 1-459 and 600-791 read. Lines 460-599 (guard amendments and the disclosure entry of 2026-10-14) not opened. |
| `ANALYSIS_SPEC.md` | Lines 1-349 read. Lines 350 to the end (section 15 details, section 16, Appendix A) not opened. |
| `R1pass2/r1_record.json` | Lines 1-150 and 9540-9625 read; the `x1_grid`, `rounds`, `status` and `git_sha` lines found by search. Per-block detail (lines 150-9540) not opened. |
| `R4pass2/r4_a1_record.json` | Lines 1-80, 5118-5210 and the matched/lo/hi lines of the cells named in the fact-check read. Other cells not opened. |
| `r23/r2r3_record.json` (R23) | Lines 1-130, 838-873 and 2134-2173 read, plus a search of summary fields. The rest (per-cell curves) not opened. |
| Frozen records `SRC/docs/prereg/*.json` | Only `r1_record.json` was searched for `x1_grid` (to compare with R1pass2). `r1_record_round1.json`, `r4_a1_record.json`, `r4_a1_record_round1.json` and `r2r3_record.json` in the export were not opened. |
| `SRC/manifests/` | Only `paper_manifest_stock_pilot_u5_b10_v5.tsv` was searched, by pattern counts (X1 total, none, scvi_adv, six penalties, X13 by method, X6). No row was read. Other manifests (v2 to v4, extension and redo manifests) not opened. |
| `PRECHK/manifest.tsv` | Read in full. `manifest_18row_superseded_20261021.tsv`: first 4 lines only (not a listed source). `manifest_tags.txt` not opened. |
| `changes_vs_tag.json` (r1_pass2, r4_pass2) | Read in full. |
| `docs/prereg/amendment_20261017_followups.md`, `followup_design_decisions_20261017.md`, `ANALYSIS_SPEC_FOLLOWUPS.md`, `R0_X10_DEFAULTS.md`, `outline.tex`, `outline_sources.md`, `docs/manifest_v5_counts.csv` | Read in full. |
| `refs.bib` | Searched for the citation keys used (all present: corani2017statistical, clopper1934use, korsunsky2019fast, hie2019efficient, xu2021probabilistic, hrovatin2025integrating, mescheder2018training, miyato2018spectral, arjovsky2017towards); not read otherwise. |
| Rule folders `r1_pass1`, `r4_pass1`, other `tier12_v3/rules` files | Not opened (not listed as sources). |
| Main-benchmark and follow-up scores, latents, `X1_local/`, any `scores/` folder | Not opened and not listed, as required. |

## 5. Length of the rewritten paragraphs (hand count)

Counted by hand from the final text: topic sentence plus bullets, `\res{}` text included, citation keys and macro names excluded, each number one word. Accuracy about +-3%.

| Paragraph | Words (main bullets) | Supplement bullet words |
|---|---|---|
| Meth-6 | 124 | 18 |
| Meth-7 | 107 | 11 |
| Meth-8 | 126 | 0 |
| Meth-9 | 175 | 0 |
| Meth-10 | 190 | 0 |
| Meth-11 | 194 | 5 |
| Meth-12 | 126 | 0 |
| Meth-13 | 197 | 0 |
| Meth-14 | 152 | 13 |
| Meth-15 | 151 | 12 |
| Meth-16 | 174 | 0 |
| Meth-17 | 214 | 0 |
| Meth-18 | 166 | 0 |
| Total | about 2,096 | about 59 |

The requested ceiling was about 1,100 words. Thirteen paragraphs of at least 120 words each already need about 1,560, so the ceiling and the 120-200 words per paragraph cannot both hold for this many separate experiments. Largest savings if the ceiling has to be met: move Meth-16, Meth-17 and Meth-18 to the supplement (174 + 214 + 166 = 554 words by the table above), shorten Meth-13 and Meth-11 to two bullets each (about 150), merge Meth-14 and Meth-15 (about 40).
