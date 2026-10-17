# Independent rigour check: Introduction of paper/paper.tex (commit add2f73, lines 114-217)

Reviewer: Rigor Reviewer (review only; paper.tex was not edited) | Date: 2026-10-15 | Track: intro_review | Repository: kreid415/wasserstein-critic-deconfounding, branch paper

## 1. Verdict

**PASS_WITH_FIXES.** 25 findings: 0 blockers, 16 fixes, 9 notes.

| severity | count | ids |
|---|---|---|
| blocker | 0 | none |
| fix | 16 | INT-01, INT-02, INT-03, INT-04, INT-05, INT-06, INT-07, INT-08, INT-09, INT-10, INT-11, INT-12, INT-13, INT-14, INT-15, INT-16 |
| note | 9 | INT-17, INT-18, INT-19, INT-20, INT-21, INT-22, INT-23, INT-24, INT-25 |

A blocker would be a false locked statement, a citation whose source does not say what is attributed (including a wrong theorem, section or page pointer), or bibliographic data that misidentifies a work. None was found. All 15 source tags in the Introduction (lines 114-182) point at facts that are true at commit 08f1e3c; three of them do not support the whole statement they follow: the two Gap lines (:166-167 carry design-document tags on a search outcome and a writing choice, INT-02) and the SI-18 tag at :177 (it fixes the arms and strength grids, not a comparison, INT-10). All 23 citation commands (25 key uses, 19 distinct works) support their claims at the level read, with the wording fixes below. The fixes are what a methods reviewer or a first-time reader would object to first:

1. Gap paragraph (INT-01, INT-02, INT-03): 'budget-matched' is undefined (PAPER_PLAN uses 'budget' both for the number of configurations tried and for critic steps) and the primary contrast P1 compares 5 critic steps with 1; the search scope is not stated and the recorded search covers 2024 onward only; the authors' own earlier preprint (discriminator versus W1 critic on a fixed backbone) is not mentioned.
2. Theory paragraphs (INT-04 to INT-08): the over-alignment sentence (:150) claims more than the two cited theorems; the precondition of Zhao's theorem, the noise model of Arjovsky and Bottou's Theorem 2.6, the fixed-distribution caveat and one theorem pointer need correcting.
3. Locked versus pending (INT-09 to INT-12, INT-14, INT-21): the finding labels of X7, the W1-versus-MMD/Sinkhorn comparison tagged with a ledger entry that fixes only the arms and grids, a guidance claim without a tag, four untagged inferences, and three scCRAFT mentions under one pending marker.
4. Scope and readability (INT-13, INT-15): the decoder choice (batch label given to the decoder or withheld) is never named in the Introduction; the paragraph at :139-140 uses terms that no sentence defines.
5. Bibliography (INT-16): one author name in refs.bib.

### Terms and labels used in this report

| term | plain meaning |
|---|---|
| paper.tex:N | line N of paper/paper.tex at commit add2f73 (the outline the paper is written from) |
| source tag, pending marker | \src{...} = locked statement with its source; \pending{what}{when it locks} = not locked; author-year \citep = literature claim checked against the paper |
| SI-nn | entry nn of the decisions ledger CONSTRAINTS.md (a decision by the user) |
| X1, X3, X6, X7, X8, X12, X13, X15 | experiments: X1 = core benchmark (3,000 model fits); X3 = cell types removed from one batch at rising doses; X6 = regularisation and critic-step follow-up; X7 = choice of reference batch; X8 = number of batches; X12, X15 = robustness checks; X13 = standard integration tools |
| A1, A2, A3 | pilot experiments: A1 = penalty strengths; A2 = adversary input (posterior sample instead of mean); A3 = per-dimension standardisation of that input |
| rules R1-R4; results sections R0-R5 | rules R1-R4 = pre-registered decision rules (R1 strengths, R2 adversary input, R3 standardisation, R4 matched strengths); R0-R5 = results sections of the paper (R0 = premise check) |
| P1-P4 | primary endpoints; P1 = biology score at matched batch removal, reference critic minus discriminator |
| b0, b_common, low-signal block | b0 = batch score of plain scVI; b_common = batch score that every adversarial arm reaches (the lowest of the arms' highest); low-signal block = one task with one decoder where b_common - b0 < 0.02 |
| JS, W1, KL, MMD, OT | Jensen-Shannon divergence; Wasserstein-1 distance; Kullback-Leibler divergence; maximum mean discrepancy; optimal transport |
| GAN, WGAN-GP, HSIC, GLUE | generative adversarial network; Wasserstein GAN with gradient penalty; Hilbert-Schmidt independence criterion; GLUE (graph-linked embedding, as named in the sysVI reference list) |
| scVI, scIB, sysVI, scIB-E | single-cell variational inference; single-cell integration benchmarking; a published scVI-based integration method; a published benchmark of batch losses in scVI |
| D-06, D-07, I089, I015, I077, I078 | labels of earlier design-review findings: D-06 = X7 variant arms not at matched batch removal; D-07 = X8 per-batch cell numbers and composition change with batch count; I089 = X7 reference coverage confounded; I015, I077, I078 = X8 design issues |
| C01-C25 | claim numbers of the claims audit of the original submission |
| D1-D9 | pipeline defects of the original submission listed in PAPER_PLAN section 1 (D5 = 10 critic updates versus 1 discriminator update) |
| R-11, G-07 | R-11 = finding of the earlier Methods-track review: the outcome of rules R2 and R3 for X1 is fixed by SI-66 and the frozen rule-R4 record (defaults kept), only the formal re-run is pending; G-07 = guard-review note that sysVI rows carry placeholder columns the fit does not read |
| PREREG, PAPER_PLAN, ANALYSIS_SPEC, plan.py | docs/PREREG.md (pre-registration), docs/PAPER_PLAN.md (paper plan), docs/ANALYSIS_SPEC.md (analysis specification), scripts/scvi_adversarial_plan.py (training code of the arms) |
| INT-nn | findings of this review |

## 2. Scope, inputs and method

- Reviewed: paper/paper.tex at commit add2f73, lines 114-217: the Introduction (114-182) and the commented list 'CLAIMS TO DROP OR SOFTEN' (184-216). File:line tags inside paper.tex (for example scvi_adversarial_plan.py:359) refer to commit 08f1e3c; add2f73 changed only files under paper/ relative to 08f1e3c, so every source file I read is identical at both commits.
- Sources were read as text (PDF to text, no page images). Literature: full texts or preprint versions of the works in section 5.1 except Goh 2022 and Maan 2024 (abstract only; section 9). Project facts: the decisions ledger CONSTRAINTS.md (SI-01 to SI-77), PAPER_PLAN, PREREG sections 0-9, ANALYSIS_SPEC, the scripts, the manifest of record, the frozen record docs/prereg/r4_a1_record.json, the design recheck, the decision letter and the claims audit. Code: scvi-tools 1.4.2 (wheel downloaded; the cited lines read) and the repository scripts at add2f73.
- Task items map to sections: (1) literature truth, 5.1 and INT-04 to INT-08, INT-18 to INT-20; (2) source statements against source, 5.1 rows CL-17 to CL-23 and 5.5; (3) locked versus pending, 5.5; (4) the 14 framing conflicts and the drop list, 5.2 and 5.3; (5) the lead's integration edits, 5.4; (6) the gap search repeated, section 7; (7) coverage and contradictions, section 6; (8) readability, section 8.
- Severity as defined in the delegation: blocker = false or wrongly sourced locked statement, citation whose source does not say what is attributed, or bibliographic metadata that misidentifies a work; fix = lock/pending misclassification, claim beyond what was read, missing locked item, inconsistency with Methods or the glossary; note = wording, readability, style.
- Kept: paper.tex not edited (replacement text is given); nothing pushed; no write to the /home/kendall folders; the shared project rigour ledger was not modified (entries are in the separate flat file rigor_ledger_entries_intro_review.json, ids INT-01 to INT-25).

## 3. Findings (severity order)

No blockers. Fixes first, then notes. 'Replacement text' is exact LaTeX for the outline (ASCII only, same macros as the outline).

### INT-01 (fix) - paper.tex:166 (Gap), :172-181 (Contribution); related :160, :178, :231, :352-353

**Problem.** 'Budget-matched' (:166, from PAPER_PLAN's working title and gap line) is not defined in the Introduction, which uses 'budget' for two different things. In PAPER_PLAN, 'budget parity' means the same number of tried configurations per method (here six penalty strengths per arm) and 'update budget' means critic steps per encoder step. The study matches the first and not the second: P1 compares the reference critic at 5 critic steps with the discriminator at 1 step. Because :160 and :178 use 'update budget' for steps, a reader will take 'budget-matched' at :166 as step-matched, and the first thing a methods reviewer will raise is the 5-versus-1 contrast, which the Introduction never states.

**Evidence.** PAPER_PLAN lines 5 and 37 ('budget-matched'), 203 ('Budget parity. Every method family gets the same number of configurations'), 43 and 70 ('update budget' = critic steps). Manifest of record (SHA-256 prefix 6efbe422c8f09f98, 6930 rows), X1 (3000 rows): n_critic is 1 for discriminator, 5 for reference, pooled and barycenter, 0 for mmd, sinkhorn, none and scvi_adv (intro_checks_results.json, x1_n_critic_by_arm). SI-01 rules out a 10-step discriminator. docs/ANALYSIS_SPEC.md section 16, item 7 (lines 355-357 at 08f1e3c): P1 'does not separate the divergence from the extra critic steps'; the step control (X6) exists for the pooled critic only (manifest X6: pooled at 1, 5 and 10 steps, discriminator variants at 1); PREREG.md line 326 records the same. paper.tex:231 states one step for the discriminator and :353 five critic steps; no sentence links them, and the outline abstract (:108) presents P1 without the step difference. PAPER_PLAN section 1 (line 19) lists the original submission's update imbalance (10 critic updates versus 1 discriminator update) as defect D5; P1 narrows it to 5 versus 1 but keeps it, which a reader of the earlier preprint will recognise.

**Recommended fix.** Say which budget is matched, drop 'budget-matched', and state the limit of P1 in the Contribution.

Replacement text for paper.tex, :166 (replace):

```latex
\item No comparison was found that gives JS, W1 and MMD/OT objectives the same backbone, epochs, minibatch, latent size and number of penalty strengths on one stock scVI backbone: a search-based absence, not an exhaustive one \pending{gap statement and its search scope}{search record added to \file{paper/evidence.md}; search repeated at submission}.
```

Replacement text for paper.tex, after :172 (new bullet):

```latex
\item Limit of the primary contrast: P1 compares the reference critic at 5 critic steps per encoder step with the discriminator at 1 step, not 10 \src{SI-01; manifest}, so P1 alone does not separate the divergence from the extra critic steps; X6 varies the steps for the pooled critic only \src{\file{docs/ANALYSIS_SPEC.md} section 16, item 7; manifest}.
```

### INT-02 (fix) - paper.tex:166-167 (Gap)

**Problem.** (a) :166 is tagged as locked, \src{PAPER_PLAN section 2; search repeated 2026-10-15, evidence.md}. PAPER_PLAN section 2 is a design-time statement that no ledger entry adopts, and the sentence reports the outcome of a search, which holds only for the scope searched. The search record in evidence.md (:64-67, :75) covers OpenAlex from 2024-01-01 only (4 queries, 15 results each, 52 works) plus one web search; the line states no scope, and work from 2017-2023 is outside that record. (b) :167 'No first claim is made' is a writing choice, not a ledger decision, yet it carries a \src tag. (c) :167 states a literature fact (iMAP, SAUCIE, MMD-ResNet) without an author-year citation although the claims were checked (evidence.md); the W1-critic precedent for domain invariance (WDGRL), named in PAPER_PLAN section 2 and in the drop list (:190), is missing.

**Evidence.** evidence.md:62-67 and :75; PAPER_PLAN section 2 ('Already known', line 29-37); legend paper.tex:40. This review repeated the search without the 2024 cut (section 7): OpenAlex, arXiv and Europe PMC, 2017 to 2026-10-15, 6 queries, 175 unique works, plus 54 works citing scIB-E, sysVI, scCRAFT or ResPAN, plus 2 web searches: no comparison of JS, W1 and MMD/OT objectives on one scVI backbone. Re-read for :167: iMAP (Methods: 'adversarial losses are optimized via the WGAN-GP'; one dataset is the anchor), SAUCIE (bioRxiv Sec. 4.1.3: non-reference batches aligned to a reference batch by MMD), MMD-ResNet (arXiv Introduction: a learned map calibrates a source sample to match a target sample).

**Recommended fix.** Split the lines by status, state the search scope, cite the four works, and add this review's search record (section 7) to paper/evidence.md; then the gap line may carry \src{evidence.md, Gap search}.

Replacement text for paper.tex, :166 (replace; supersedes the INT-01 wording only in the search clause):

```latex
\item No comparison of JS, W1 and MMD/OT objectives on one stock scVI backbone was found in OpenAlex, arXiv and Europe PMC (2017 to 2026-10-15; 6 queries; 175 works screened; works citing scIB-E, sysVI, scCRAFT and ResPAN read; two web searches): a search-based absence, not an exhaustive one \pending{gap statement}{search record added to \file{paper/evidence.md}; search repeated at submission}.
```

Replacement text for paper.tex, :167 (replace):

```latex
\item Reference anchoring and W1 critics already exist: W1 critics for domain invariance (WDGRL; \citealp{shen2018wasserstein}), reference-anchored WGAN-GP integration (iMAP; \citealp{wang2021imap}), reference-anchored MMD (SAUCIE, \citealp{amodio2019exploring}; MMD-ResNet, \citealp{shaham2017removal}). \pending{whether the text states that no first claim is made}{your decision}
```

### INT-03 (fix) - paper.tex:163-168 (Gap); related :111, :306

**Problem.** The Gap paragraph names scIB-E as the closest controlled-backbone study but omits the only earlier discriminator-versus-W1-critic comparison on a fixed backbone: the authors' own public preprint (the submitted manuscript). A reader who finds it will see 'no comparison' contradicted. The Introduction does not cite the preprint (it appears once, in Related work :306, for the backbone), and the outline does not say that this study replaces it and does not carry its results forward.

**Evidence.** refs.bib reid2025topology (bioRxiv 10.1101/2025.11.12.688061; Crossref posted 2025-11). The July submission (Topology_Matters.zip, oup-authoring-template.tex:132, :134, :142) compares the standard discriminator with a multi-headed reference-based Wasserstein critic 'while holding the autoencoder backbone constant' on the scCRAFT backbone and says no such isolation had been done (claims audit C02, 'untested'). A web search on 2026-10-15 returned the preprint's v1 and v3 pages (v1 title: Wasserstein Critics Outperform Discriminators in Adversarial Deconfounding of Gene Expression Data); the bioRxiv full text was not readable (HTTP 429, evidence.md:60). PAPER_PLAN section 1 (line 14) lists the pipeline defects D1-D9 of that submission and says they 'must be disclosed, since bioRxiv v1-v3 are public'; line 298 plans a superseding preprint version. SI-07: the preprint is replaced by the next version once results exist (the outline abstract, :111, says so; the Introduction does not).

**Recommended fix.** Add the earlier comparison to the Gap paragraph with its status; leave the disclosure wording pending.

Replacement text for paper.tex, after :166 (new bullet):

```latex
\item The only earlier JS-versus-W1 comparison on a fixed backbone known to us is our own preprint \citep{reid2025topology} (scCRAFT backbone), which the next version replaces \src{SI-07}. \pending{statement that its results are not carried over, and the disclosure of its pipeline defects in the Supplement (PAPER\_PLAN \S1 and Supplement outline)}{your decision}
```

### INT-04 (fix) - paper.tex:150; drop-list wording at :191-192

**Problem.** ':150 'Over-alignment under composition shift is thus predicted for every divergence arm, not only W1' goes beyond the two theorems. Zhao et al. Thm 4.3 is a bound for the JS distance with binary labels. Wu et al. Prop. 3.1 covers exact alignment (any distance that is zero only for identical distributions) with zero source error, binary labels. Neither gives a result for partial alignment under W1, MMD or Sinkhorn, and the penalty arms at finite strength do not align exactly. The sentence is a hypothesis (what X3 tests), not a prediction of the cited theorems. The drop list repeats the overreach (':191-192 'Zhao 2019, Wu 2019 and sysVI predict over-alignment for any strong aligner') and uses 'predict' for sysVI, an empirical study. The overreach comes from PAPER_PLAN section 2 (lines 34 and 41: 'any proper divergence', 'predicted for every divergence'), which evidence.md:71 had already narrowed in :148-149.

**Evidence.** Zhao et al. 2019 (PMLR v97, p7): Thm 4.3 uses d_JS and binary Y (Lemmas 4.6-4.8); the Remark under it says the lower bound is a necessary condition for domain-adaptation methods that learn invariant representations. Wu et al. 2019 (PMLR v97, p3): Prop. 3.1 says that if a distance D between the two latent distributions is zero only when they coincide, then zero source error together with D = 0 implies a target error of at least the gap between the two label proportions (binary labels); the domain-classifier loss (Ganin et al.), Wasserstein distance (Shen et al.) and MMD (Huang et al.) are named in Sec. 2 as distances with that property. evidence.md:71 states the same limit; :148-149 respect it; :150 does not.

**Recommended fix.** Restate :150 as a hypothesis tied to X3 and correct the drop-list wording.

Replacement text for paper.tex, :150 (replace):

```latex
\item Exact alignment under any such distance is therefore expected to cost accuracy under composition shift; at the finite penalty strengths used here, whether every divergence arm over-aligns is a hypothesis, tested by the X3 dose-response \pending{X3 analysis}{specified during X1, before any X1 result is examined (SI-68)}.
```

Replacement text for paper.tex, :191-192 (replace both lines):

```latex
%  - "Critics over-correct" as a W1 property (C06 invalidated), unless shown at matched batch removal; Zhao 2019 (JS distance)
%    and Wu 2019 (exact alignment), both with binary labels, bound the error of aligned representations under composition shift;
%    sysVI reports that strong adversarial alignment mixed unrelated cell types in its methods.
```

### INT-05 (fix) - paper.tex:148

**Problem.** The inequality is stated as if general. Thm 4.3 holds when the label distributions are farther apart than the latent distributions, d_JS(D^Y_S, D^Y_T) >= d_JS(D^Z_S, D^Z_T), and for a classifier h applied to the latent codes (Markov chain X -> Z -> prediction), the same h in both domains. The symbols eps_S, eps_T, d_JS, D^Y and D^Z are undefined in the Introduction, and 'forces error' does not say whose error: that of any cell-type classifier applied to the aligned latent codes (this study trains no classifier).

**Evidence.** Zhao et al. 2019 (PMLR v97, p6-7): Lemma 4.8 assumes the Markov chain X -> Z -> Y-hat; Thm 4.3 is stated under the condition of Lemma 4.8 and requires d_JS(D_S^Y, D_T^Y) >= d_JS(D_S^Z, D_T^Z); d_JS is the square root of the JS divergence.

**Recommended fix.** Add the precondition and define the symbols in the same sentence.

Replacement text for paper.tex, :148 (replace):

```latex
\item When cell-type (label) proportions differ between two batches (domains), any one classifier applied to a shared latent space must make errors: if the label distributions are farther apart than the latent distributions, the sum of its classification errors in the two batches is at least $\tfrac12\bigl(d_{JS}(\mathcal{D}^Y_S,\mathcal{D}^Y_T)-d_{JS}(\mathcal{D}^Z_S,\mathcal{D}^Z_T)\bigr)^2$, which grows as the latent distributions are aligned more tightly \citep[Thm.~4.3; binary labels; $d_{JS}$ is the square root of the JS divergence, $\mathcal{D}^Y$ the label distribution, $\mathcal{D}^Z$ the latent distribution]{zhao2019learning}.
```

### INT-06 (fix) - paper.tex:133-134, :142

**Problem.** The caveat at :142 names Thms 2.4-2.6 only. WGAN Thm 1 and Example 1 rest on the same assumption (a fixed second distribution P_r and a generator g_theta that moves), and Thm 2.4 also assumes a discriminator close to the optimal one. In this study every batch's latent moves and the discriminator takes one step per encoder step, so none of these results transfers automatically; whether the discriminator gets close to optimal is an empirical question (the exploratory premise check R0).

**Evidence.** Arjovsky et al. 2017 (PMLR v70, Sec. 2): Thm 1 begins 'Let P_r be a fixed distribution over X'; Example 1 fixes P_0. Arjovsky and Bottou 2017 (arXiv v1, Sec. 2.2.1): Thm 2.4 requires the discriminator D to be within epsilon of the optimal D* (in norm) and a bound M on the generator's Jacobian, with P_r the real data distribution. SI-01 (one discriminator step); manifest (n_critic 1).

**Recommended fix.** Extend the caveat to every result in the paragraph that has these assumptions.

Replacement text for paper.tex, :142 (replace):

```latex
\item The results above assume a fixed second distribution (the real data) and a moving generator (WGAN Thm.~1 and Example~1; Thms.~2.4--2.6), and Thm.~2.4 a discriminator close to the optimal one. Here every batch's latent moves and the discriminator takes one step per encoder step \src{SI-01}, so they do not transfer automatically; whether the discriminator gets close to optimal is part of the exploratory premise check R0 \src{SI-68}.
```

### INT-07 (fix) - paper.tex:137

**Problem.** ':137 'with a noisy discriminator its gradient is Cauchy-distributed' hides the noise model: Thm 2.6 assumes, under the disjoint-support conditions of Thms 2.1-2.2, that the discriminator differs from the optimal one by independent Gaussian white noise at every point (and independent noise in its gradient), which the authors themselves call too strong. 'The non-saturating loss need not vanish' is prose before Thm 2.5; Thm 2.5 states what the loss optimises at the optimal discriminator (reverse KL minus twice the JS divergence), not that it does not vanish.

**Evidence.** Arjovsky and Bottou 2017 (arXiv v1): Sec. 2.2.2 text before Thm 2.5 ('doesn't necessarily suffer from vanishing gradients'); Thm 2.5: E_z[-grad log D*(g(z))] = grad[KL(P_g || P_r) - 2 JSD(P_g || P_r)] at theta_0, D* fixed; Thm 2.6: D* - D and grad D* - grad D are independent centred Gaussian processes ('white noise'), each coordinate of the expected gradient is a centred Cauchy with infinite expectation and variance; the text after the proof says the decorrelation assumption 'is albeit too strong'.

**Recommended fix.** Name the noise model and the exact content of Thm 2.5.

Replacement text for paper.tex, :137 (replace):

```latex
\item The non-saturating loss $-\log D$ does not necessarily have vanishing gradients: at the optimal discriminator its expected gradient is that of $\mathrm{KL}(P_g\Vert P_r)-2\,\mathrm{JS}(P_g\Vert P_r)$ (Thm.~2.5); under the conditions of Thms.~2.1 or 2.2, if the discriminator equals the optimal one plus independent Gaussian white noise at every point (with independent noise in its gradient), each gradient coordinate is Cauchy-distributed with infinite mean and variance (Thm.~2.6; the authors call this independence assumption too strong) \citep[Sec.~2.2.2, Thms.~2.5--2.6]{arjovsky2017towards}.
```

### INT-08 (fix) - paper.tex:127

**Problem.** The pointer '[Thm. 1]' is imprecise. Theorem 1 of Goodfellow et al. states only that the global minimum of the criterion is reached if and only if p_g = p_data and equals -log 4. The identity C(G) = -log 4 + 2 JSD for the optimal discriminator is Eq. 6, inside the proof of that theorem. A reader who opens the theorem statement does not find the identity. Not rated a blocker because the identity is derived in the proof of the cited theorem; a stricter reading of 'wrong theorem pointer' would make it one. 'JSD' is also not spelled out in the Introduction ('JS' is).

**Evidence.** Goodfellow et al. 2014 (NeurIPS 27), Sec. 4.1: Proposition 1 (optimal discriminator), Theorem 1 (minimum -log 4 reached iff p_g = p_data), then Eq. 5-6 in the proof: C(G) = -log(4) + 2 JSD(p_data || p_g).

**Recommended fix.** Point to Sec. 4.1, Eq. 6 and say 'for the optimal discriminator'.

Replacement text for paper.tex, :127 (replace):

```latex
\item A discriminator targets the Jensen--Shannon (JS) divergence: for the optimal discriminator the generative adversarial network (GAN) objective equals $-\log 4$ plus twice the JS divergence between data and generator \citep[Sec.~4.1, Eq.~6, in the proof of Thm.~1]{goodfellow2014generative}.
```

### INT-09 (fix) - paper.tex:179 (decomposition item 5)

**Problem.** The NO-GO note for X7 cites 'D-06, D-07'. D-07 is the X8 finding. The glossary (:63-64), the Research questions (:255, :240), Methods (:452-453) and the analysis spec give X7: D-06 and I089; X8: D-07, I015, I077, I078. The Introduction omits I089 and attaches X8's finding to X7.

**Evidence.** paper.tex:63 (X7: NO-GO as designed: D-06, I089), :64 (X8: D-07, I015, I077, I078). docs/ANALYSIS_SPEC.md section 0 table lines 36-37: P3/X8 -> I015, I077, I078, D-07; P4/X7 -> D-06, I089. review/rereview/design_fable/REVIEW_design_recheck.md lines 39-40: D-06 = the variant arms of X7, X12 and X6 inherit the matched lambda of their base arm and are not at matched batch removal; D-07 = X8: per-batch cell numbers and cell-type composition change with the number of batches. The recheck's summary row (line 11) names X7 and X8 together ('NO-GO as designed until D-06 / D-07 are resolved'), the likely source of the mix-up.

**Recommended fix.** Use the glossary's references.

Replacement text for paper.tex, :179 (replace):

```latex
\item target design, reference versus symmetric: X7 \pending{X7 redesign (rated NO-GO as designed; final-review findings D-06 and I089)}{redesigned during X1, locked before any X1 result is examined (SI-68)}.
```

### INT-10 (fix) - paper.tex:177 (decomposition item 3)

**Problem.** 'W1 critics against MMD and Sinkhorn arms \src{SI-18}' presents a comparison as locked. SI-18 fixes the arm families and strength grids, not an analysis. SI-68 locks tables of every method at one penalty strength and the width of each method's range of good strengths, which put the W1 critics, MMD and Sinkhorn side by side, but no locked analysis attributes a difference to W1 geometry rather than latent scale; the decomposition itself is pending (:173) and SI-68 defers the other follow-up analyses to X1 time.

**Evidence.** CONSTRAINTS.md SI-18 (rule: lambda-grid families, divergence x decoder, fixed from A1); SI-68 (list of analyses locked now; the rest during X1). Every line that mentions MMD or Sinkhorn was listed and read (the first 150 characters of each line): 19 lines of CONSTRAINTS.md (SI-18 grids, SI-25 X3 weights, SI-44 KL warm-up arms, the Sinkhorn fix, redo, precision and iteration-cap entries), 12 lines of docs/ANALYSIS_SPEC.md (Sinkhorn redo, grids, A3 rows, R3 factor effects; section 10, window width, is per arm) and 64 lines of docs/PREREG.md (grids, A3, X15 arms, Sinkhorn redo, minibatch handling, rule-R4 flags). None defines an analysis that contrasts W1 critics with MMD or Sinkhorn. This is a line-level scan, weak evidence on its own; the conclusion rests mainly on SI-68's closed list of analyses locked now.

**Recommended fix.** Keep the arms and the locked all-arm tables tagged and mark the attribution to W1 geometry pending.

Replacement text for paper.tex, :177 (replace the first clause):

```latex
W1 geometry or latent scale: the W1 critics, MMD and Sinkhorn arms side by side in the single-strength tables and the widths of the ranges of good strengths \src{SI-18; SI-68} \pending{attribution of a difference to W1 geometry rather than latent scale, and its analysis}{follow-up analyses locked during X1 (SI-68)}; per-dimension standardisation of the adversary input, ... (rest of the line unchanged)
```

### INT-11 (fix) - paper.tex:181

**Problem.** The first clause ('Guidance for scVI users follows from the results, possibly that no adversary beats stock scVI') has no tag of its own. The item's \src{SI-68, SI-77} follows the second clause (the comparison with standard tools), and \pending{outcome} marks only the result, so the claim that guidance follows reads as locked. No ledger entry says that guidance follows. The outline abstract marks the same idea pending (:110), so the two sections classify it differently.

**Evidence.** CONSTRAINTS.md SI-68, SI-77 (comparison with standard tools; selection score); PAPER_PLAN section 2 line 46; paper.tex:110 (pending). Decision letter, Reviewer 3, major comment 3 asks for practical guidelines.

**Recommended fix.** Split the item so the locked comparison and the pending guidance are separate.

Replacement text for paper.tex, :181 (replace):

```latex
\item Comparison with stock scVI and standard tools (Harmony, Scanorama, sysVI, scANVI) at equal numbers of tries \src{SI-68; SI-77}. \pending{guidance for scVI users, including the case that no adversary beats stock scVI (PAPER\_PLAN \S2)}{results of X1 and of the baseline experiment X13}
```

### INT-12 (fix) - paper.tex:141 (last clause), :160 (and :142, :150; see INT-06, INT-04)

**Problem.** The legend (:40) has three status forms (source tag, pending marker, author-year citation). Four Introduction lines carry none of them: :141 'for two batches this is the label-flipped non-saturating loss', :142, :150 and :160. They are the drafting agent's inferences, not statements found in a source, and read as settled. Checked here: :141 is correct (algebra in the evidence); :142 is correct but incomplete (INT-06); :150 overreaches (INT-04); :160 is plausible (both image-GAN studies vary regularisation and the discriminator update ratio) and already says '(image results)', but has no marker.

**Evidence.** scvi_adversarial_plan.py:297-298: other = (not one_hot(batch)) / (k - 1); loss = -(log p * other).sum(1).mean(); for k = 2, 'other' is the one-hot vector of the other batch, so the loss is -log p(other batch | z), the label-flipped non-saturating loss. Kurach et al. 2019 (PMLR v97): the number of discriminator updates per generator update (1 or 5) is among the searched settings, with regularisation and normalisation; Lucic et al. 2018: 'imbalanced update frequencies of generator and discriminator' are among the searched settings.

**Recommended fix.** Turn :141's clause into a sourced code fact and mark :160 as an inference.

Replacement text for paper.tex, :141 (replace the end of the line):

```latex
...at one adversary step \src{SI-01; \file{scvi_adversarial_plan.py}:8--9, 292--298}; for two batches the target is the other batch alone, so this is the label-flipped non-saturating loss \src{\file{scvi_adversarial_plan.py}:297--298 with k = 2}.
```

Replacement text for paper.tex, :160 (replace):

```latex
\item A critic--discriminator difference may thus reflect regularisation, update budget or tuning, not the divergence (image results) \pending{inference from image-GAN studies, tested for single-cell data by X6}{X6 analysis specified during X1 (SI-68)}.
```

### INT-13 (fix) - paper.tex:172 (Contribution, Scope)

**Problem.** The Scope bullet omits the decoder. 'One stock scVI backbone at scvi-tools defaults' holds for the conditioned decoder (scVI as people use it). In the unconditioned half of X1 no batch label reaches the model, so the adversary is the only batch correction; that is not how scVI is usually used. Both decoders carry their own primary result (SI-70), and the outline abstract (:106) names both, but the Introduction never does.

**Evidence.** CONSTRAINTS.md SI-09 ('scVI as people use it'), SI-11, SI-16 ('5 seeds in both X1 blocks'), SI-70 (one P1 result per decoder, never averaged); manifest X1: cond = 0 and 1 for the six authored arms, cond = 1 for scvi_adv (40 rows); paper.tex:82 and :344 (decoder definitions).

**Recommended fix.** Name both decoders in the Scope bullet.

Replacement text for paper.tex, :172 (insert after 'at scvi-tools defaults;'):

```latex
two decoders, conditioned (the decoder receives the batch label, as scVI is usually used) and unconditioned (no batch label reaches the model, so the adversary is the only batch correction), each with its own primary result \src{SI-09; SI-11; SI-16; SI-70};
```

### INT-14 (fix) - paper.tex:152 and :305, versus :306

**Problem.** The open decision 'whether the main text names scCRAFT' is marked pending only on :306, which is the one mention PAPER_PLAN section 3 allows ('Cite scCRAFT only as the earlier backbone'). :152 and :305 cite scCRAFT as related work (its triplet loss against over-correction) with no marker, although both PAPER_PLAN section 3 and the July-August convention ('no scCRAFT mention anywhere in the main tex') would exclude them. If the decision goes against naming scCRAFT, :152 and :305 must also change.

**Evidence.** paper.tex:306 (pending: 'PAPER_PLAN section 3: only as the earlier backbone; July convention: no mention'); docs/PAPER_PLAN.md line 65; evidence.md:82 (the two instructions conflict, neither is a ledger entry; 'scCRAFT is also cited as related work').

**Recommended fix.** Mark the other two mentions with the same pending decision.

Replacement text for paper.tex, :152 (append):

```latex
... \citep{he2025partially} \pending{whether the main text names scCRAFT (see Related work)}{your decision}.
```

Replacement text for paper.tex, :305 (append):

```latex
... \citep{he2025partially} \pending{whether the main text names scCRAFT}{your decision}.
```

### INT-15 (fix) - paper.tex:126-143 (JS versus W1), :148, :151, :165

**Problem.** The densest paragraph is not readable for a genetics PhD student who has not read earlier documents (SI-67, active; NO-JARGON rule). (1) The theory speaks of 'generator' and 'discriminator' while :126 speaks of 'encoder'; no sentence says the encoder is the generator and the two batches' latent codes are the two distributions. (2) 'posterior samples' and 'posterior mean' are used at :139-140 without a definition (the glossary has none). (3) 'authored arms', 'the frozen rule-R4 record on A1', 'b_common', 'low-signal' and 'refitting Sinkhorn' carry no plain meaning in the same sentence (:140); the glossary has some of them (:75, :77). (4) scVI is first met inside 'scvi-tools' at :139 and expanded only at :165. (5) GLUE (:151) is not spelled out. (6) :143 carries '(a fairness concern of Reviewer 1)' in plain text; reviewer remarks belong in the source tag. (7) The :140 sentence mentions one flagged block; the record flags two.

**Evidence.** paper.tex:126-151 and :165 read line by line; glossary :56-95 has no entry for 'posterior mean', 'posterior sample' or 'authored arm'. The cited source of GLUE (sysVI reference 34) names it 'graph-linked embedding'. Frozen record (docs/prereg/r4_a1_record.json, git c88cce3): immune|1 b_common - b0 = 0.0188 (setter barycenter) and sim1|1 = 0.0069 (setter mmd), both below 0.02, both flagged low-signal; barycenter sets b_common in five of six blocks, mmd in one, sinkhorn in none (intro_checks_results.json, r4_a1_blocks). A2 and A3 use atac_small and immune only (glossary :58-59), so only immune blocks rules R2 and R3; docs/PREREG.md lines 469-471 state the same (sim1 is not an A2/A3 task; the Sinkhorn redo cannot clear either flag).

**Recommended fix.** Add a mapping bullet, define the two adversary inputs once, and rewrite :139, :140, :143 and :151 in plain words (replacements below); then drop the scVI expansion at :165.

Replacement text for paper.tex, before :133 (new bullet):

```latex
\item Reading the theory for batch correction: the encoder plays the generator, the latent codes of two batches are the two distributions to be aligned, and the discriminator or critic is the adversary.
```

Replacement text for paper.tex, :139 (replace):

```latex
\item scVI (single-cell variational inference) encodes each cell as a Gaussian in latent space, with a mean and a variance. The posterior mean is that mean; a posterior sample adds Gaussian noise with that variance. In the main experiment X1, scvi-tools' own adversary (arm scvi\_adv) is given posterior samples \src{scvi-tools 1.4.2: scvi/nn/\_base\_components.py:304--305, scvi/module/\_vae.py:390, :413, scvi/train/\_trainingplans.py:689; \file{scvi_adversarial_plan.py}:348--351; arm scvi\_adv (40 rows; its adv\_input column is not read) in \file{manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv}}.
```

Replacement text for paper.tex, :140 (replace):

```latex
\item The six arms implemented in this project (discriminator; reference, pooled and barycenter W1 critics; MMD; Sinkhorn) are given the posterior mean by default \src{SI-19; \file{scvi_adversarial_plan.py}:359}. X1 keeps this default: the pre-registered pilot rules that could switch the adversary input to posterior samples (rule R2) or standardise it per dimension (rule R3) stay at the defaults while a block they use is flagged low-signal (the batch removal that all six arms reach exceeds plain scVI's by less than 0.02) \src{SI-66}. The frozen record of the strength-matching rule (rule R4, run on pilot A1) flags two blocks, both with the conditioned decoder: immune (0.0188) and sim1 (0.0069); only immune is used by rules R2 and R3. Refitting Sinkhorn cannot clear the flags: the common level is capped by the arm that reaches the least batch removal (barycenter in five of the six blocks, MMD in sim1 with the conditioned decoder, Sinkhorn in none), and refitting another arm cannot raise that cap \src{\file{docs/prereg/r4_a1_record.json}}. \pending{formal re-run of rules R2 and R3}{after the Sinkhorn redo (SI-57)}
```

Replacement text for paper.tex, :143 (replace):

```latex
\item The vanishing-gradient premise is examined, not assumed: the premise check R0 (batch separability before integration; adversary gradient norms) is exploratory and gives no confirmed or refuted claim \src{SI-68; PAPER\_PLAN \S3; decision letter, Reviewer 1, major comment 2 (fairness of the comparison)}.
```

Replacement text for paper.tex, :151 (replace the first words):

```latex
An adversarial conditional variational autoencoder (VAE) and GLUE (graph-linked embedding) mixed ...
```

### INT-16 (fix) - refs.bib:84 (ganin2016domain), cited at paper.tex:126

**Problem.** The author list has 'Mario March'; the author is Mario Marchand. Not a blocker: title, venue, year, volume, number, pages and arXiv id identify the work.

**Evidence.** JMLR 17(59):1-35, first page (work/lit_pdfs/ganin2016.txt): 'Mario Marchand' (Universite Laval). A substring check of title and author surnames of the 16 retrieved source texts against refs.bib found no other mismatch (the check cannot flag 'March', a substring of 'Marchand'; this error was found by reading the author list). Ten entries with DOIs (Leek, Yu, Tung, Goh, Luecken, Dincer, Hrovatin, He, Maan, Yi) were also compared with their Crossref records earlier in this review (volume, pages, year).

**Recommended fix.** Correct the author name.

Replacement text for paper.tex, refs.bib:84 (replace the substring):

```latex
... and Mario Marchand and Victor Lempitsky}
```

### INT-17 (note) - paper.tex:139-140 (source tags)

**Problem.** (a) The chain from the encoder's random draw to the adversary's input lacks a link in the :139 tags: _trainingplans.py:689 reads inference_outputs['z'] and _base_components.py:304-305 draws z with rsample, but scvi/module/_vae.py:390 and :413, which connect them, are not cited. (b) The tag cites the manifest for scvi_adv, whose adv_input column reads 'mean' for all 40 rows although the arm never reads it (evidence.md says so; paper.tex does not); a reader opening the manifest finds the opposite of the sentence. (c) The accepted R-11 wording mentions one flagged block; the record flags two (INT-15).

**Evidence.** scvi-tools 1.4.2 wheel: _vae.py:390 'qz, z = self.z_encoder(...)', :413 'MODULE_KEYS.Z_KEY: z'; _base_components.py:304-305 (Normal(q_m, q_v.sqrt()); rsample); _trainingplans.py:689 (z = inference_outputs["z"]); scvi_adversarial_plan.py:359 (z_used = mu if adv_input == "mean" else inference_outputs["z"]), :348-351 (arms none and scvi_adv take the stock step). Manifest X1 scvi_adv rows: adv_input values 'mean' (intro_checks_results.json).

**Recommended fix.** Already included in the INT-15 replacement for :139.

### INT-18 (note) - paper.tex:120, :133, :135, :138

**Problem.** Wording a step beyond the source. :120 'over-correction removes biology' (source: over-correction may 'hinder biomedical discovery'). :133 'JS and KL divergences are not' (Thm 1 part 3 says statements 1-2 are false for them in general, as Example 1 shows). :135 omits 'compact' (Thm 2.1) and that Thm 2.2 gives zero gradient at almost every point of closed manifolds that do not align and lack full dimension. :138 Thm 3.1 shows that X + noise has a density; overlap of supports needs the positive Gaussian density of Cor. 3.1; scVI's posterior noise has a cell-specific scale, which the theorem does not cover directly.

**Evidence.** Yu et al. 2024 abstract (over-correction may 'hinder biomedical discovery'); Arjovsky et al. 2017 Thm 1 (3), Example 1; Arjovsky and Bottou 2017 arXiv v1 Thms 2.1, 2.2, 3.1, Cor. 3.1 (statements read as text).

**Recommended fix.** Adopt the replacement wording.

Replacement text for paper.tex, :120 (replace):

```latex
\item Correction can fail both ways: uncorrected batch effects can mislead, and over-correction can hinder biomedical discovery \citep{yu2024assessing}.
```

Replacement text for paper.tex, :133 (replace the end):

```latex
... differentiable almost everywhere for locally Lipschitz ones (with a regularity assumption); for JS and Kullback--Leibler (KL) divergences these statements fail in general \citep[Thm.~1, part 3]{arjovsky2017wasserstein}.
```

Replacement text for paper.tex, :135 (replace):

```latex
\item For disjoint compact supports (Thm.~2.1), or supports on closed manifolds that do not perfectly align and do not have full dimension (Thm.~2.2), a perfect discriminator exists whose gradient is zero on the supports (everywhere in Thm.~2.1, at almost every point in Thm.~2.2) \citep[Thms.~2.1--2.2]{arjovsky2017towards}.
```

Replacement text for paper.tex, :138 (replace):

```latex
\item Continuous noise added to the discriminator input gives each distribution a density (Thm.~3.1); for Gaussian noise the density is positive everywhere, so the supports overlap \citep[Thm.~3.1, Cor.~3.1]{arjovsky2017towards}; scVI's posterior noise has a cell-specific scale, which the theorem does not cover directly.
```

### INT-19 (note) - paper.tex:119 (goh2022batch), :153 (maan2024characterizing)

**Problem.** Both citations rest on abstracts because the publisher texts are closed. The lines stay within the abstracts. Goh et al.: the abstract (PubMed 35282901) forecasts that batch effects 'will not only remain relevant in the age of big data' and says that single-cell RNA sequencing presents new challenges; ':119 'remain a single-cell problem' narrows that. Maan et al.: the published abstract lists clustering, cell-type classification, differential expression, marker annotation, query mapping and trajectory inference across five integration methods (2,600 experiments). The 2022 preprint, read as a substitute for corroboration, is a different version: its abstract has 1,600 scenarios and lists clustering, classification, marker-gene annotation and query mapping (no trajectories; differential expression only in the body), and it names BBKNN, Harmony, Scanorama, scVI and Seurat; the word 'adversarial' does not occur. The line therefore supports 'imbalance matters for integration', not adversarial over-alignment, although it sits in the over-alignment bullets.

**Evidence.** Goh et al. 2022: abstract via Europe PMC (PubMed 35282901; Crossref has no abstract); full text closed. Maan et al. 2024: abstract via Europe PMC (PubMed 38429430, as recorded in evidence.md:89); full text not obtainable (Unpaywall, Semantic Scholar, PMC and Crossref text-mining links all negative); preprint 10.1101/2022.10.06.511156 read as text (work/lit_pdfs/maan_preprint.txt).

**Recommended fix.** Keep both lines on the abstract-only list; soften Goh; move Maan out of the over-alignment bullets or say that its methods are not adversarial once the full text has been read.

Replacement text for paper.tex, :119 (replace):

```latex
\item In single-cell RNA sequencing (scRNA-seq), separately processed replicates differ substantially \citep{tung2017batch}; batch effects are expected to remain relevant, and scRNA-seq brings new challenges for modelling them \citep{goh2022batch}.
```

### INT-20 (note) - paper.tex:165

**Problem.** 'No W1, MMD or optimal-transport term' is literally true of scIB-E but invites an objection: scIB-E's HSIC term is a kernel dependence measure, and HSIC equals the MMD between the joint distribution of latent code and batch label and the product of their marginals. The sentence should say 'no explicit MMD between batch distributions'.

**Evidence.** scIB-E full text (Genome Biol 26:398): 0 occurrences each of Wasserstein, MMD, maximum mean, optimal transport, Sinkhorn and earth mover; level-1 list GAN, HSIC, Orthog, MIM, RBP, RCE; HSIC 'measures the dependence between two random variables using kernel methods'. HSIC = MMD between joint and product of marginals: arXiv 2503.04820 'A Practical Introduction to Kernel Discrepancies: MMD, HSIC & KSD' (seen as a search-result entry; the paper was not opened in full).

**Recommended fix.** Adopt the replacement wording.

Replacement text for paper.tex, :165 (replace the end):

```latex
... in one scVI framework; no W1, optimal-transport (OT) or explicit MMD term between batch distributions (its HSIC term measures dependence between latent code and batch label) \citep{yi2025benchmarking}.
```

### INT-21 (note) - paper.tex:176, :178

**Problem.** Missing tags on locked material. :176 describes the X3 design (cell types depleted in one batch at rising doses) in plain text with no source tag; the design is locked (SI-25, SI-31 to SI-33, SI-41; manifest) and only the analysis is open. :178 names spectral normalisation and the R1 gradient penalty without their citations (given in Methods :448), and 'R1 gradient penalty' collides with 'rule R1' and results section R1 (the glossary notes this at :62).

**Evidence.** CONSTRAINTS.md SI-25, SI-31..33, SI-41 (same cell count at every dose); manifest X3 rows; refs.bib miyato2018spectral, mescheder2018training (added at integration, evidence.md:94-101); glossary :62.

**Recommended fix.** Add the design tag and the two citations.

Replacement text for paper.tex, :176 (replace):

```latex
\item composition-driven over-alignment: dose-response X3 (cell types depleted in one batch at rising doses, the same number of cells at every dose) \src{SI-25; SI-31--33; SI-41; manifest} \pending{X3 analysis}{specified during X1, before any X1 result is examined (SI-68)};
```

Replacement text for paper.tex, :178 (replace the first clause):

```latex
regularisation and update budget: X6 (the discriminator plain, spectrally normalised \citep{miyato2018spectral} or with an R1 gradient penalty \citep{mescheder2018training}, unrelated to rule R1; the pooled critic at 1, 5 and 10 critic steps, and spectrally normalised at 1 and 5) ... (rest unchanged)
```

### INT-22 (note) - paper.tex:170-182 (Contribution)

**Problem.** Locked topics of the Research questions are absent from the Contribution: batch-count scaling (R2, X8; P3 pending redesign), local versus global batch removal (SI-76, the first major comment of Reviewer 1), and the robustness analyses X12 (architecture and hyperparameters; Tier 2, SI-05) and X15 (KL warm-up, SI-44), which answer Reviewer 3, major comment 2. The decomposition follows PAPER_PLAN section 2, which has none of them.

**Evidence.** paper.tex:232 (local mixing), :239-240 (R2), :246-248 (R3); CONSTRAINTS.md SI-05, SI-44, SI-76; decision letter, Reviewer 1 major comment 1, Reviewer 3 major comment 2.

**Recommended fix.** Add one line with the status of each.

Replacement text for paper.tex, after :180 (new bullet):

```latex
\item Also asked: whether the critic--discriminator difference changes with the number of batches (X8) \pending{wording and design}{X8 redesign, during X1 (SI-68)}; whether the critic reaches matched batch removal through more local mixing \src{SI-76}; and how robust the results are to architecture and KL warm-up (X12, X15) \src{SI-05; SI-44} \pending{analyses}{follow-up analyses locked during X1 (SI-68)}.
```

### INT-23 (note) - paper.tex:190, :195 (drop list)

**Problem.** Two reasons in the commented list are worded beyond their sources. (a) :190 'WDGRL and iMAP already use W1 critics': WDGRL estimates the empirical W1; iMAP's full text says its adversarial losses are optimised via WGAN-GP and has no occurrence of 'Wasserstein', the same treatment the list gives IMAAE (:201). (b) :195 'Reviewer 2 comment 3: nothing topological was measured': the letter (Reviewer 2, main concern 3) says the work does not justify mechanistic language about topology and does not show which cell types collapse or which references fail; 'nothing topological was measured' is PAPER_PLAN's own statement. (c) :191-192 (INT-04).

**Evidence.** iMAP (Genome Biol 22:63, Methods: 'adversarial losses are optimized via the WGAN-GP'; 0 occurrences of 'Wasserstein', 1 of 'WGAN'); decision letter, Reviewer 2, main concern 3 (letter lines 65-67); PAPER_PLAN title note. All other drop-list reasons were checked and hold (section 5).

**Recommended fix.** Reword (a) and (b).

Replacement text for paper.tex, :190 (replace):

```latex
%    batch losses inside scVI; WDGRL uses a W1 critic and iMAP WGAN-GP critics; iMAP and SAUCIE already use reference anchoring.
```

Replacement text for paper.tex, :195 (replace):

```latex
%    reference", "topological collapse" (C12 untested, C14 contradicted; Reviewer 2, main concern 3: the work does not justify
%    topological mechanisms; PAPER_PLAN: nothing topological was measured).
```

### INT-24 (note) - paper.tex:334 (Methods, Backbone; other track), versus :172

**Problem.** The Introduction's narrowed Scope is right; Methods :334 still says SCVI 'at its defaults for every arm and baseline' with the ZINB likelihood. PAPER_PLAN section 3.4 (amended 2026-10-14; G-07; SI-63) says sysVI fits a Gaussian likelihood with its own decoder, and the fit script fits sysVI on scIB's normalised X with the backbone's latent size, hidden width and layers; Harmony, Scanorama and PCA have no such settings. SI-10's wording ('every arm and baseline') is the user's, so the Methods line should be narrowed as the Introduction was.

**Evidence.** fit_paper_config.py:404-410 (sysVI setup and construction); PAPER_PLAN section 3.4; CONSTRAINTS.md SI-10, SI-63; manifest X13 sysvi rows carry a likelihood value that the fit does not read.

**Recommended fix.** Narrow Methods :334 in the Methods review.

### INT-25 (note) - paper.tex:184-216 (drop list); July Introduction l.134

**Problem.** The drop list names 20 of the 25 audited claims by C-number; the others are C03, C07, C13, C15, C19. Of these, C03 ('We introduce a multi-class reference-based Wasserstein critic'; status partially supported) is the one that the July Introduction states (l.134: 'we implement a multi-headed critic formulation'). The outline's Contribution makes no method claim, but nothing tells the writer not to port the July sentence, and the audit's own recommendation (present the reference critic as one of three critic formulations and state the implemented generator objective exactly) is not carried. C07, C13, C15 and C19 sit in the July Results and Discussion, outside this section.

**Evidence.** original_manuscript_claims_audit.csv C03 (where: Abstract/Results l.105; Introduction l.134; Methods l.203-224; status partially_supported; basis: the reference-critic head and W1 loss exist, but the binary and reference-sensitivity runs used a fallback gradient penalty and the generator back-propagated the full W estimate); Topology_Matters.zip oup-authoring-template.tex:105, :134; paper.tex:185-216 (C-numbers found by pattern match: C01, C02, C04, C05, C06, C08, C09, C10, C11, C12, C14, C16, C17, C18, C20, C21, C22, C23, C24, C25).

**Recommended fix.** Add one line to the drop list.

Replacement text for paper.tex, after :215 (new comment line):

```latex
%  - July sentence "we implement a multi-headed critic formulation" / "We introduce a multi-class reference-based Wasserstein critic" (C03 partially supported): here the reference critic is one of three critic formulations (reference, pooled, barycenter); state the implemented generator objective exactly (claims audit C03).
```

## 4. What already holds (no change needed)

- Code and data facts behind the source tags: scvi-tools 1.4.2 draws the encoder sample with rsample and passes it to the stock adversary (_trainingplans.py:689, _base_components.py:304-305, _vae.py:390 and :413); scripts/scvi_adversarial_plan.py:359 uses the posterior mean by default for the authored arms; its fool loss (:292-298) is the cross-entropy to the uniform distribution over the other batches, as in scvi-tools (_trainingplans.py:651-667); the manifest of record (6930 rows, SHA-256 prefix 6efbe422c8f09f98) has 3000 X1 rows, equal epochs, minibatch and latent size across arms per task, and the X6 combinations described at :178.
- R-11 wording (the rule-R4 record on A1) in the two Introduction places is correct. Monotonic argument: b_common is the lowest, over the six arms, of each arm's highest batch score. A refit of Sinkhorn changes only Sinkhorn's term, so it cannot raise b_common above the arm that caps it (barycenter in 5 of 6 blocks, MMD in 1, Sinkhorn in 0); in the flagged blocks b_common - b0 is 0.0188 (immune, conditioned) and 0.0069 (sim1, conditioned), both below 0.02, with 0 extension fits.
- Framing conflicts: 12 of 14 resolutions are right as written (their Introduction-side residuals are separate findings), 1 is incomplete (conflict 12, INT-14), 1 lies outside the Introduction (conflict 14).
- Drop list: the 9 audit statuses written out in the list (C01, C02, C06, C08, C09, C12, C14, C18, C20) were matched to the claims audit CSV by script (0 mismatches); the July background lines (Tung, taxonomy, Stegle), the 'real counts' wording, the premise wording, the pilot hint (0.726 versus 0.646, PAPER_PLAN:46) and the August barycenter sentence check out against their sources.
- The pending marker on the decomposition (:173), the X3, X6 analyses and the formal re-run of rules R2 and R3 are correctly classified.

## 5. Verification tables

### 5.1 Claims checked (44 rows; each against the source text named)

| id | paper.tex line | claim | source cited | what was read | verdict |
|---|---|---|---|---|---|
| CL-01 | :118 | Batch effects are technical variation (laboratory conditions, reagent lots, personnel, instruments, time, pipeline); they mislead when correlated with the outcome | Leek 2010; Yu 2024 | Leek 2010 (Nat Rev Genet 11:733, PMC text): abstract (laboratory conditions, reagent lots, personnel; correlated with an outcome); Yu 2024 (Genome Biol 25:254): abstract | OK |
| CL-02 | :119 | Separately processed replicates differ substantially in scRNA-seq | Tung 2017 | Tung et al. 2017 (Sci Rep 7:39921) full text: results on separately processed replicates (Fluidigm C1 plates) | OK |
| CL-03 | :119 | Batch effects remain a single-cell problem | Goh 2022 | Abstract only (full text closed) | OK + INT-19 |
| CL-04 | :120 | Residual effects mislead; over-correction removes biology | Yu 2024 | Abstract: 'misleading outcomes if uncorrected, or hinder biomedical discovery if over-corrected' | OK + INT-18 |
| CL-05 | :121 | Integration is scored on two axes, batch removal and conservation of biological variation | Luecken 2022 (scIB) | Nat Methods 19:41 full text: abstract and metric overview | OK |
| CL-06 | :126 | An encoder is trained against an adversary that predicts the batch from the latent code | Ganin 2016 | JMLR 17(59) full text: domain classifier trained against the feature extractor; author list on page 1 | OK; metadata INT-16 |
| CL-07 | :126 | ... or another confounder such as age | Dincer 2020 (AD-AE) | bioRxiv full text and published abstract: adversarial deconfounding of the latent code; age named as a confounder | OK |
| CL-08 | :127 | Optimal discriminator gives -log 4 + 2 JSD | Goodfellow 2014 [Thm. 1] | NeurIPS 27 full text Sec. 4.1: Prop. 1, Thm. 1, Eq. 5-6 | OK + INT-08 |
| CL-09 | :128 | A W1 critic estimates the earth mover's distance through its 1-Lipschitz dual | Arjovsky 2017 (WGAN) | PMLR v70 full text Sec. 3 (Kantorovich-Rubinstein duality) | OK |
| CL-10 | :128 | WGAN-GP enforces the Lipschitz constraint softly | Gulrajani 2017 [Sec. 4] | NeurIPS 30 full text Sec. 4 (gradient penalty on interpolates) | OK |
| CL-11 | :133 | W1 is continuous / differentiable a.e. under the stated conditions; JS and KL are not | WGAN [Thm. 1] | PMLR v70 Thm. 1 (statements 1-3), Example 1 | OK + INT-18 |
| CL-12 | :134 | Disjoint parallel lines: JS = log 2 for every nonzero offset, W1 = the offset | WGAN [Example 1] | PMLR v70 Example 1 (W = \|theta\|, JS = log 2 for theta not 0) | OK |
| CL-13 | :135 | Disjoint supports or non-aligned manifolds: a perfect discriminator with zero gradient exists | Arjovsky and Bottou 2017 [Thms. 2.1-2.2] | arXiv 1701.04862v1 Sec. 2.1 (Thms. 2.1, 2.2) | OK + INT-18 |
| CL-14 | :136 | Saturating loss: vanishing generator gradients near the optimal discriminator | Arjovsky and Bottou 2017 [Thm. 2.4] | arXiv v1 Thm. 2.4, Cor. 2.1 | OK + INT-06 |
| CL-15 | :137 | Non-saturating loss need not vanish; with a noisy discriminator its gradient is Cauchy-distributed | Arjovsky and Bottou 2017 [Thms. 2.5-2.6] | arXiv v1 Sec. 2.2.2, Thms. 2.5, 2.6 and the remark after the proof | FINDING INT-07 |
| CL-16 | :138 | Continuous noise on the discriminator input removes the disjoint-support assumption | Arjovsky and Bottou 2017 [Thm. 3.1] | arXiv v1 Sec. 3, Thm. 3.1, Cor. 3.1 | OK + INT-18 |
| CL-17 | :139 | In X1 the stock scvi-tools adversary sees posterior samples | scvi-tools 1.4.2; plan.py:348-351; manifest | scvi/train/_trainingplans.py:689 (z = inference_outputs['z']); scvi/nn/_base_components.py:304-305 (rsample); scvi/module/_vae.py:390, :413; plan.py:348-351 (arms none and scvi_adv take the stock step); manifest scvi_adv rows (40), cond = 1 | OK + INT-17 |
| CL-18 | :140 | Authored arms see the posterior mean by default | SI-19; plan.py:359 | plan.py:359 (z_used = mu if adv_input == 'mean' else sample); SI-19 text; manifest authored X1 rows hold the placeholders A2 (adv_input) and A3 (standardisation) | OK |
| CL-19 | :140 | The frozen rule-R4 record flags immune (conditioned), so R2 and R3 cannot switch; refitting Sinkhorn cannot remove the flag | SI-66; r4_a1_record.json | docs/prereg/r4_a1_record.json (frozen, git c88cce3, 0 extensions): immune\|1 b_common - b0 = 0.0188, low_signal true; setters barycenter in 5 blocks, mmd in sim1\|1, sinkhorn in none; SI-66 text | OK + INT-15 (two blocks flagged) |
| CL-20 | :141 | The discriminator arm minimises scvi-tools' fool loss (cross-entropy to uniform over the other batches, over all cells) at one adversary step | SI-01; plan.py:8-9, 292-298 | plan.py:8-9 ('ONE adversary step per generator step'), :292-298 (_fool_loss: other = one-hot complement / (k - 1); mean over cells); scvi-tools _trainingplans.py:651-667 (fool loss: cross-entropy to the uniform distribution over the other classes); SI-01 | OK |
| CL-21 | :141 | For two batches this is the label-flipped non-saturating loss | (none) | Algebra from plan.py:297-298 with k = 2 | OK + INT-12 (add tag) |
| CL-22 | :142 | Thms. 2.4-2.6 assume a fixed data distribution; every batch's latent moves here | (none) | arXiv v1 Thms. 2.4-2.6 (P_r the real data distribution); WGAN Thm. 1 (P_r fixed) | OK + INT-06 |
| CL-23 | :143 | The vanishing-gradient premise is examined, not assumed: R0 is exploratory | SI-68; PAPER_PLAN section 3; decision letter, Reviewer 1, major comment 2 | SI-68 (exploratory: no confirmed or refuted claim); decision letter Reviewer 1 'Major comments' second bullet (disjoint-support / fairness) | OK + INT-15 (reviewer remark in plain text) |
| CL-24 | :148 | Aligning latent distributions forces error: eps_S + eps_T >= 1/2 (d_JS(Y) - d_JS(Z))^2 | Zhao 2019 [Thm. 4.3; binary labels, JS distance] | PMLR v97 Sec. 4.3: Lemma 4.8, Thm. 4.3, Remark | OK + INT-05 |
| CL-25 | :149 | Exact alignment with zero source error leaves target error of at least the label-proportion gap | Wu 2019 [Prop. 3.1; binary labels] | PMLR v97 Sec. 2-3, Prop. 3.1 and the text after it | OK |
| CL-26 | :150 | Over-alignment under composition shift is predicted for every divergence arm | (none) | Zhao Thm. 4.3 (JS only), Wu Prop. 3.1 (exact alignment only) | FINDING INT-04 |
| CL-27 | :151 | An adversarial cVAE and GLUE mixed unrelated cell types with unbalanced proportions, more so at higher correction strength | Hrovatin 2025 (sysVI) | BMC Genomics 26:974 full text: Introduction and Results on adversarial cVAE and GLUE under unbalanced proportions | OK + INT-15 (spell out GLUE) |
| CL-28 | :152 | scCRAFT added a triplet loss against the over-correction of its adversarial loss under heterogeneous compositions | He 2025 (scCRAFT) | Commun Biol 8:561 full text: abstract, Introduction, Methods (dual-resolution triplet loss) | OK + INT-14 (pending marker) |
| CL-29 | :153 | Dataset imbalance changed clustering, annotation, DE, mapping, trajectories for five methods | Maan 2024 | Abstract via Europe PMC (published); 2022 preprint text as substitute | OK + INT-19 |
| CL-30 | :158 | Image GANs: most losses reach similar scores with enough tuning and restarts; none consistently beats the non-saturating GAN | Lucic 2018 | NeurIPS 31 full text: abstract, Sec. 1 and 3 | OK |
| CL-31 | :159 | Non-saturating loss stable; spectral normalisation consistently helped; gradient penalty only sometimes | Kurach 2019 | PMLR v97 full text: abstract, Sec. 3-4, Fig. 2 | OK |
| CL-32 | :160 | A critic-discriminator difference may reflect regularisation, update budget or tuning (image results) | (none) | Inference from CL-30, CL-31 | OK + INT-12 (add marker) |
| CL-33 | :165 | scIB-E: six batch losses in one scVI framework; no W1, MMD or OT term | Yi 2025 (scIB-E) | Genome Biol 26:398 full text: Methods 'Level-1' list; 0 hits for Wasserstein, MMD, maximum mean, optimal transport, Sinkhorn, earth mover | OK + INT-20 |
| CL-34 | :166 | No budget-matched comparison of JS, W1 and MMD/OT objectives on a stock scVI backbone was found | PAPER_PLAN section 2; evidence.md | Own gap search (section 7); evidence.md:62-67, :75 | FINDING INT-01, INT-02, INT-03 |
| CL-35 | :167 | Reference anchoring exists for W1 critics (iMAP) and MMD (SAUCIE, MMD-ResNet) | PAPER_PLAN section 2 | iMAP full text (Genome Biol 22:63, Methods: WGAN-GP; anchor dataset); SAUCIE bioRxiv Sec. 4.1.3 (MMD to a reference batch); MMD-ResNet arXiv 1610.04181 Introduction | OK + INT-02 (status, citations) |
| CL-36 | :172 | One stock scVI backbone at scvi-tools defaults; equal epochs, minibatch and latent size per experiment and task for every arm, plain scVI and sysVI, with declared exceptions | SI-10; SI-04; SI-02 | Manifest of record: X1 epochs, minibatch (128) and latent size (10) equal across arms per task; X13 sysVI and scANVI rows match X1 per task (all 8 tasks, for sysVI and for scANVI); scANVI's extra training epochs after pre-training (fit_paper_config.py:401-402) are declared in Methods | OK + INT-13 (decoders), INT-24 |
| CL-37 | :173 | The decomposition as the paper's stated contribution is not locked | pending | CONSTRAINTS.md: no entry adopts the decomposition; PAPER_PLAN section 2 | OK (correctly pending) |
| CL-38 | :175 | P1: biology kept at equal batch removal | SI-68 | SI-68 text; ANALYSIS_SPEC section 0 | OK |
| CL-39 | :176 | X3 dose-response | pending (analysis) | SI-25, SI-31..33, SI-41; manifest | OK + INT-21 (design untagged) |
| CL-40 | :177 | W1 critics against MMD and Sinkhorn arms | SI-18 | SI-18 (arms and grids); SI-68 (single-strength tables and good-strength ranges cover every arm) | FINDING INT-10 |
| CL-41 | :177 | Standardisation compared in pilot A3 under rule R3; off in X1 | SI-20; SI-27; SI-66; r4_a1_record.json | SI-20, SI-27 text; record as CL-19 | OK |
| CL-42 | :178 | X6: discriminator plain, spectrally normalised or R1; pooled critic at 1, 5, 10 steps; spectrally normalised at 1 and 5 | SI-22; manifest | Manifest X6 rows: discriminator[1], discriminator_sn[1], discriminator_r1[1], pooled[1], pooled[5], pooled[10], pooled_sn[1], pooled_sn[5]: 27 rows each; SI-22 text | OK + INT-21 (citations) |
| CL-43 | :179 | X7 redesign pending; NO-GO as designed (D-06, D-07) | pending | paper.tex:63; ANALYSIS_SPEC section 0; REVIEW_design_recheck.md:39-40 | FINDING INT-09 |
| CL-44 | :181 | Guidance for scVI users follows from the results; comparison with standard tools at equal numbers of tries | SI-68, SI-77; pending outcome | SI-68, SI-77 text | FINDING INT-11 |

### 5.2 The 14 framing conflicts

| # | conflict (PAPER_PLAN, July text or earlier review) | resolution in the outline | what I checked | verdict |
|---|---|---|---|---|
| 1 | Zhao Thm 4.3 / Wu: 'any proper divergence forces errors, tighter forces more' | JS-distance form (Zhao) plus exact-alignment form (Wu), binary labels | Zhao PMLR v97 Thm. 4.3 and Wu Prop. 3.1 read as text | Resolution right; :150 re-inflates it (INT-04) |
| 2 | scIB-E compares GAN, HSIC, orthogonality, MI losses | Lists all six level-1 losses; no W1, MMD, OT | scIB-E full text: six level-1 losses; 0 hits for Wasserstein, MMD, OT, Sinkhorn | Right; HSIC wording INT-20 |
| 3 | 'Loss is second-order to tuning and regularisation' (Lucic, Kurach) | States what each paper found, image GANs only | Lucic abstract; Kurach abstract and Sec. 3 | Right; the inference needs a marker (INT-12) |
| 4 | 'Premise becomes a tested hypothesis' (PAPER_PLAN section 3) | Premise 'examined, not assumed'; R0 exploratory | SI-68 text | Right |
| 5 | Gap 'trained on real counts' (PAPER_PLAN section 2) | 'Real counts' dropped (SI-15); search-based absence kept | SI-15; evidence.md:75; own gap search | Right; scope and status of the gap line INT-02 |
| 6 | scVI adversary trains on posterior samples so noise removes disjoint support | True for scvi_adv only; authored arms use the posterior mean by default | scvi-tools 1.4.2 lines (CL-17); plan.py:359; manifest | Right; tags INT-17 |
| 7 | 'Theory covers the saturating loss; fool loss not covered automatically' | Fool loss = label-flipped non-saturating loss for two batches; theorems assume a fixed distribution | plan.py:292-298; arXiv v1 Thms 2.4-2.6 | Right; caveat incomplete (INT-06) |
| 8 | IMAAE 'Wasserstein-based' (July :139) | Dropped; loss has WGAN-GP form | IMAAE PMC text: 0 hits for Wasserstein and WGAN; gradient penalty on interpolates, generator loss -E[D] | Right (drop list :201-202) |
| 9 | Batch effects 'arise from the sequencing technology' (Tung 2017) | Cited only for replicate-level batch effects | Tung 2017 full text: separately processed replicates, one platform | Right (:119) |
| 10 | Three-way taxonomy cited to Luecken 2022 (July :137) | Not cited to scIB | scIB groups outputs as graph, embedding, corrected matrix | Right (Related work track) |
| 11 | 'Sample age' cited to Stegle 2015 (July :128) | Dropped; AD-AE cited for age | Stegle 2015 not retrieved (also by me); AD-AE abstract names age | Right |
| 12 | scCRAFT 'only as the earlier backbone' versus 'no mention anywhere' | One backbone line pending; scCRAFT also cited as related work | PAPER_PLAN line 65; evidence.md:82; paper.tex:152, :305, :306 | Resolution incomplete (INT-14) |
| 13 | Barycenter 'first' claim (August revision) | Not carried; listed in the drop list | manuscript_tm/oup-authoring-template.tex:140 at 08f1e3c; drop list :214-215 | Right |
| 14 | Seed-list key corani2017time | Entry kept under the seed key | evidence.md:84 (Methods and Related-work tracks) | Outside the Introduction; not checked here |

### 5.3 Drop-list items

| lines | item dropped or softened | audit status (CSV) | what I checked | verdict |
|---|---|---|---|---|
| :187-188 | JS fails on disjoint technology batches (C01) | partially_supported | SI-68: R0 exploratory | OK |
| :189-190 | First systematic comparison (C02) | untested | scIB-E six losses in scVI; WDGRL, iMAP, SAUCIE | OK; wording INT-23(a); prior preprint INT-03 |
| :191-192 | Critics over-correct as a W1 property (C06) | invalidated_by_bugs | Zhao, Wu, sysVI | Reason overreaches (INT-04) |
| :193 | Weak integration is protective (C09) | invalidated_by_bugs | audit basis | OK |
| :194-195 | Topology language (C12, C14) | untested; contradicted | decision letter, Reviewer 2 main concern 3; PAPER_PLAN title note | OK; attribution INT-23(b) |
| :196-197 | W1 gives robust gradients regardless of initial separation (C23) | partially_supported | WGAN Thm. 1; Gulrajani Sec. 4 | OK |
| :198-200 | Superior local mixing, scales poorly, broad windows, 24% more time, ideal for disjoint, barycenter 'theoretical solution', primary limitation (C04, C05, C10, C11, C16-C18, C20-C22) | invalidated_by_bugs / contradicted / untested as listed | claims audit CSV statuses; July :428 | OK |
| :201-202 | IMAAE 'Wasserstein-based' and 'motivating' a global Gaussian prior (C24) | partially_supported | IMAAE PMC text | OK |
| :203-204 | 'Unbalanced scenario known to drive severe over-correction' (C25); 'balanced' not composition-independent (C08) | untested; contradicted | audit basis; sysVI, scCRAFT, Maan | OK |
| :205-208 | July background attributions (Tung, taxonomy, Stegle) | n/a | Tung 2017, scIB, AD-AE texts; July :128, :137 | OK |
| :209-210 | 'Trained on real counts' in the gap line | n/a | SI-15 | OK |
| :211 | 'The premise becomes a tested hypothesis' | n/a | SI-68 | OK |
| :212-213 | Pilot hint 'stock scVI 0.726 vs best conditioned arm 0.646' provisional | n/a | PAPER_PLAN:46 reads 0.726 and 0.646 | OK |
| :214-215 | August-revision barycenter 'first' sentence | n/a | manuscript_tm tex:140 at 08f1e3c | OK |

### 5.4 Integration edits 1 to 5 (edits 6 to 8 concern Methods, the reference list and the legend, outside this section)

| # | where | edit by the lead | what I checked | verdict |
|---|---|---|---|---|
| 1 | :140 ('authored arms') | R2 outcome no longer pending; X1 keeps the posterior mean; only the formal R2/R3 re-run pending | Same wording as Methods :369; record values in section 5; SI-66; monotonic argument: a refit of Sinkhorn cannot raise b_common above the arm that caps it (barycenter 5 blocks, mmd 1, sinkhorn 0) | Correct. Residual: sim1\|1 also flagged (INT-15); plain wording (INT-15) |
| 2 | :177 (item 3) | R3 outcome no longer pending; standardisation off in X1; formal re-run pending | Same record; SI-20, SI-27, SI-66, SI-57 | Correct; INT-10 concerns the first clause of the same item |
| 3 | :172 (Scope) | 'for all arms and baselines' narrowed to every arm, plain scVI and sysVI with declared exceptions | Manifest: epochs, minibatch, latent equal across X1 arms and sysVI per task; scANVI extra epochs and X12 levels declared in Methods; Harmony, Scanorama, PCA have no epochs | Correct; decoder dimension missing (INT-13); Methods :334 still unnarrowed (INT-24) |
| 4 | :178 (item 4) | 'factorial X6' replaced by the glossary's X6 design | Manifest X6: eight arm-step combinations, 27 rows each; JS x steps cells absent by SI-01; glossary :62; Methods :448 | Correct; citations INT-21 |
| 5 | Introduction and Related work | Three file paths 'scripts/x.py' written 'x.py' | Reading guide (paper.tex How to read this outline); scvi_adversarial_plan.py exists as scripts/scvi_adversarial_plan.py at 08f1e3c | Correct |

### 5.5 Locked versus pending: every line of the Introduction that carries a tag (15 source tags and 7 pending markers on 14 lines)

| paper.tex line | tags on the line | classification check | finding |
|---|---|---|---|
| :139 | 1 source tag | True at 08f1e3c (scvi-tools 1.4.2 lines; plan.py:348-351; manifest scvi_adv rows) | INT-17 (tag completeness) |
| :140 | 2 source tags, 1 pending | Both source tags true (SI-19 and plan.py:359; SI-66 and the frozen record); pending correct (SI-57) | INT-15 (second flagged block; plain wording) |
| :141 | 1 source tag | True (SI-01; plan.py:8-9, 292-298); the untagged clause after it is an inference | INT-12 |
| :143 | 1 source tag | True (SI-68; PAPER_PLAN section 3; decision letter, Reviewer 1, major comment 2) | INT-15 (reviewer remark in plain text) |
| :166 | 1 source tag | Source is a design document and a search record that holds for 2024 on only; the statement is a search outcome | INT-02, INT-01, INT-03 (should be pending) |
| :167 | 1 source tag | Source is PAPER_PLAN section 2; 'no first claim' is a writing choice | INT-02 (should be pending) |
| :172 | 2 source tags | True (SI-10, SI-04, SI-02; manifest checked); decoder dimension missing | INT-13, INT-24 |
| :173 | 1 pending | Correctly pending: no ledger entry adopts the decomposition | none |
| :175 | 1 source tag | True (SI-68) | none |
| :176 | 1 pending | Analysis correctly pending; the design is locked but untagged | INT-21 |
| :177 | 3 source tags, 1 pending | SI-18 fixes arms and grids only; SI-68 puts W1, MMD and Sinkhorn side by side in all-arm tables, but no locked analysis attributes a difference to W1 geometry; the other two tags and the pending marker are right | INT-10 |
| :178 | 1 source tag, 1 pending | True (SI-22; manifest X6 combinations); pending correct | INT-21 (citations) |
| :179 | 1 pending | Classification right; finding labels wrong (D-07 belongs to X8) | INT-09 |
| :181 | 1 source tag, 1 pending | Tags right for the comparison; the guidance clause has no tag of its own | INT-11 |

### 5.6 Replacement text compiled

The replacement text of the findings that change lines of the outline (INT-01 to INT-22; INT-02 supplies :166-167) was applied to a copy of paper.tex at add2f73 (27 lines replaced, 4 inserted, and the refs.bib author corrected) and the copy was compiled with tectonic: 0 TeX errors, 0 undefined citations or references, 2 underfull-line warnings against 2 in the unchanged file. The change set is intro_replacements.diff. Not applied in the test: the replacements of commented lines (INT-04 :191-192, INT-23, INT-25), because comments do not compile. paper.tex itself was not edited.

## 6. Coverage and contradictions against Research questions, Methods and the glossary

| topic | Introduction | other section | result |
|---|---|---|---|
| decoder (batch label given or withheld) | not named | abstract :106, glossary :82, Methods :344, results per decoder (SI-70) | missing in the Introduction (INT-13) |
| rule-R4 record, low-signal flag | :140, :177 | Methods :369, glossary :77 | consistent in substance; both Introduction and Methods :369 mention immune only while the record flags immune and sim1 (INT-15; Methods track) |
| X6 design | :178 | glossary :62, Methods :448 | consistent; citations missing in the Introduction (INT-21) |
| X7 finding labels | :179 (D-06, D-07) | glossary :63, Research questions :255, Methods :452 (D-06, I089) | contradiction (INT-09) |
| X8, batch-count scaling, local mixing, X12, X15 | absent from Contribution | Research questions :232, :239-240, :246-248 | coverage gap (INT-22) |
| P1 critic steps | :175 (no mention) | Research questions :231 (one step), Methods :353 (5 steps) | unstated limit of P1 (INT-01) |
| backbone scope | :172 (narrowed) | Methods :334 (every arm and baseline, ZINB) | Methods line needs the same narrowing (INT-24) |
| earlier preprint | not mentioned | abstract :111 (replaced by the next version), Related work :306 | INT-03 |
| scCRAFT | :152 | Related work :305-306 (pending only on :306) | INT-14 |
| guidance for scVI users | :181 (untagged first clause) | abstract :110 (pending) | classified differently (INT-11) |

## 7. Gap search repeated (task item 6)

Scope: OpenAlex (with the project's API key), arXiv and Europe PMC (PubMed, bioRxiv and medRxiv records); publication year 2017 to 2026-10-15; run on 2026-10-15 without the 2024 cut used in evidence.md. 6 queries, up to 20 results per source and query:

| query | OpenAlex hits | arXiv hits | Europe PMC hits |
|---|---|---|---|
| Sinkhorn divergence single-cell integration batch effect autoencoder | 20 | 0 | 6 |
| WGAN-GP batch effect removal scRNA-seq | 13 | 0 | 4 |
| Wasserstein critic versus discriminator single-cell batch correction | 20 | 0 | 6 |
| adversarial MMD optimal transport batch correction variational autoencoder single-cell comparison | 20 | 9 | 14 |
| controlled benchmark batch correction loss functions single-cell integration same backbone | 20 | 20 | 20 |
| latent alignment loss comparison scVI batch effect Jensen-Shannon Wasserstein maximum mean discrepancy | 4 | 16 | 2 |

194 raw hits gave 175 unique works. A language model rated the top 60 on title and abstract against the question 'compares JS, W1 and MMD/OT objectives on one scVI backbone': 59 were rated (45 none, 14 partial, 0 substantial); 116 (one inside the top 60 that had no rating, the rest below the cut) were screened by title by me, and I read the abstracts of all partial matches and of the 2025-2026 candidates. Forward citations (OpenAlex 'cites' filter) of scIB-E (6), sysVI (16), scCRAFT (6) and ResPAN (26) gave 54 works, screened by title (the file records titles, not abstracts). Two further web searches (one for the comparison, one for adversarial versus MMD or optimal-transport losses on one autoencoder) returned the authors' own preprint pages and Con-AAE; iDLC came from the OpenAlex hits and is also recorded in evidence.md:66. Files: gap_search_screen.csv, gap_forward_citations.csv, gap_search_numbers.json.

**Verdict (scope above): no controlled comparison of JS, W1 and MMD/OT objectives on one scVI backbone was found.** Partial overlap only with the authors' own preprint (JS versus reference-based W1 on the scCRAFT backbone; no MMD or OT). This supports a search-based absence, not an exhaustive one, and it is not a claim of novelty; the record in evidence.md covered 2024 onward only (INT-02).

| work | type | what it does | why it is not the comparison | what I read |
|---|---|---|---|---|
| scIB-E (Yi et al., Genome Biol 26:398, 2025) | benchmark | six level-1 batch losses (GAN, HSIC, orthogonal projection, mutual information, reverse backpropagation, reverse cross-entropy) in one scVI framework | no W1, MMD, OT or Sinkhorn arm (0 hits in the full text) | full text |
| Authors' preprint (Topology Matters, bioRxiv 10.1101/2025.11.12.688061) | comparison | discriminator versus a multi-headed reference-based W1 critic with the backbone held fixed (scCRAFT) | two objectives only; no MMD or OT; not scVI | July submission text; web-search entries for v1 and v3; bioRxiv full text unreadable (HTTP 429) |
| sysVI (Hrovatin et al., BMC Genomics 26:974, 2025) | method with ablations | KL weight, adversarial module, VampPrior and cycle-consistency inside one conditional VAE | no W1, MMD or OT arm | full text |
| iDLC (bioRxiv 10.64898/2026.02.17.706490, 2026) | new method | explicit disentanglement plus optimal transport; evidence.md:66 records a WGAN-GP discriminator with a Sinkhorn regulariser | a method, not a controlled comparison | abstract |
| scBIOT (Research Square 10.21203/rs.3.rs-8310482/v1, 2026) | new method (by title) | atlas integration method named scBIOT | method; no abstract in the OpenAlex record | title only |
| ResPAN (Wang et al., Bioinformatics 2022) | new method | WGAN-GP with a residual generator and mutual-nearest-neighbour pairing | one model, no comparison of objectives | bioRxiv text and published abstract |
| iMAP (Wang et al., Genome Biol 22:63, 2021) | new method | autoencoder plus WGAN-GP with paired transfer; one anchor dataset | one model | full text |
| IMAAE (Int J Mol Sci 24:5502, 2023) | new method | adversarial autoencoder with WGAN-GP-form losses; its design exploration reports that a plain GAN under-corrects with three or more batches | design motivation, no JS-versus-W1 comparison | full text (PMC) |
| uniPort, IMGG, transmorph, scPoli, Iterative Alignment Flows | new methods or frameworks | optimal-transport or GAN-based integration frameworks | each proposes one method | title and abstract in the screening table |
| Maan et al., Nat Biotechnol 42:1899, 2024 (and its 2022 preprint) | benchmark of imbalance | five integration methods (BBKNN, Harmony, Scanorama, scVI, Seurat in the preprint) | none adversarial; no loss comparison | published abstract; preprint text |
| Con-AAE (Bioinformatics 39:btad162, 2023) | new method (multi-omics) | contrastive cycle adversarial autoencoder for multi-omics alignment | multi-omics alignment, not scVI batch-loss comparison | seen as a web-search result only; not opened |

## 8. Readability (task item 8)

Test applied: could a genetics PhD student who has not read earlier documents follow each sentence (SI-67)? Passes: :118-121 (background), :126-128 (acronyms are spelled out at first use except JSD), :158-160, :165. Fails or needs a definition in the same sentence: the encoder-as-generator mapping, posterior mean and sample, 'authored arms', the rule-R4 record on A1, b_common, low-signal, refitting Sinkhorn, GLUE, the unexpanded scVI before :165, and a reviewer remark in plain text at :143. All are in INT-15 with replacement text; INT-08 spells out JS divergence at :127.

## 9. Sources not re-checked, and limits

- Maan et al. 2024 (maan2024characterizing) and Goh et al. 2022 (goh2022batch): publisher full texts closed (Unpaywall, Semantic Scholar, PMC, Crossref text-mining links negative); claims checked against the abstracts. For Maan the 2022 preprint (10.1101/2022.10.06.511156) was read as corroboration; it is a different version.
- Arjovsky and Bottou 2017 (arjovsky2017towards): read as arXiv v1 (1701.04862v1); the ICLR camera-ready on OpenReview was not opened (bot challenge), so theorem numbers are those of arXiv v1.
- SAUCIE, AD-AE and the weighted-distance preprint sources: read as bioRxiv or arXiv versions, not the published versions; Stegle 2015 not retrieved (not cited in the Introduction).
- The HSIC-equals-MMD relation (INT-20): seen as a search-result entry for arXiv 2503.04820; the paper itself was not opened.
- Con-AAE, scBIOT: not opened or title only (section 7).
- Related work and Methods citation sites belong to the parallel tracks; only cross-section inconsistencies are noted (INT-24, section 6).

## 10. Deviations from the delegated task

- Goh 2022 and Maan 2024: publisher full texts are closed; the claims were checked against the abstracts (Maan also against its 2022 preprint, a different version), not against full text.
- Arjovsky and Bottou 2017: read as arXiv v1 only; the ICLR camera-ready on OpenReview was not opened, so theorem numbers are those of v1.
- Gap search: of the 175 unique hits, 59 (the top 60 minus one without a rating) were rated by a language model from title and abstract; the other 116 were screened by title by me, with abstracts read for all partial matches and all 2025-2026 candidates; the 54 forward-citation works were screened by title; Con-AAE was seen only as a web-search result and scBIOT by title only.
- SAUCIE and AD-AE were read as bioRxiv versions and MMD-ResNet as arXiv, not the published versions; Stegle 2015 was not retrieved; the HSIC-equals-MMD relation was seen only as a search-result entry.
- The shared project rigour ledger was not modified (the ledger loader found no project ledger file); findings are in a separate flat file, ids INT-01 to INT-25.
- Related work and Methods citation sites were not reviewed (parallel tracks); only cross-section inconsistencies were noted (INT-24, section 6).
- The ponytail skill (smallest-change rule) is not in this profile's catalogue, so it was not loaded; the check script was kept to read-only readbacks.

## 11. Reproducibility

- Environment: conda env `python` (pandas, requests), plus the scvi-tools 1.4.2 wheel read without installation; repository checkout at add2f73 (origin removed, detached).
- intro_checks.py reads the manifest, the frozen record and the cited scvi-tools and repository lines: `python intro_checks.py <repo_dir> <scvi_wheel_dir> intro_checks_results.json`. findings_data.py, tables_data.py and build_intro_review.py rebuild this report, the claims table and the ledger file from those JSON and CSV files; every number in the report is read back from report_values.json, which is derived from intro_checks_results.json, gap_search_screen.csv, gap_forward_citations.csv and gap_search_numbers.json.
- Inputs copied from project artifacts: the claims audit of the original submission (version 6c946526-684e-4e4b-9306-95c648230aae), the outline review record (dd13e9fa-068d-4e83-9c4d-15b6fa0faf71) and the review input bundle (13f3c1f2-fa87-4438-bfd8-f2fa2ddc72b2); the repository at add2f73 (paper/paper.tex, paper/refs.bib, paper/evidence.md, docs/, scripts/, manifests/) was read from a local clone with its remote removed.
- apply_replacements_test.py applies the replacement text to a copy of paper.tex and compiles it with tectonic (compile_check.json, intro_replacements.diff).
- Search record: gap_search_screen.csv (175 rows, with year, DOI, rating, note, URL), gap_forward_citations.csv, gap_search_numbers.json.

## 12. Claim-gate receipt

```
CLAIM RECEIPT
claim: Introduction of paper/paper.tex at add2f73 reviewed: 25 findings (0 blocker, 16 fix, 9 note); report, flat ledger entries and claims table written; numbers in the report read back from the result files; replacement text compiled with 0 TeX errors.
- reconcile: counts agree: {'total': 25, 'findings': 25, 'sum(parts)': 25, 'sum(partition sizes)': 25, 'len(union(partitions))': 25, 'len(items)': 25} | in-memory
- reconcile: counts agree: {'claims_table_rows': 44, 'claims_in_data': 44} | in-memory
- reconcile: counts agree: {'conflicts_rows': 14, 'conflict_sources': 14} | in-memory
- readback: report_values.json {"key": "immune1_diff"} = 0.01882369445223908 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "sim1_1_diff"} = 0.006917415545499295 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "manifest_rows"} = 6930 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "x1_rows"} = 3000 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "manifest_sha"} = "6efbe422c8f09f98" | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "gap_unique"} = 175 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "gap_forward"} = 54 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "gap_raw"} = 194 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "n_status_checked"} = 9 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "n_status_mismatch"} = 0 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "tex_errors"} = 0 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
- readback: report_values.json {"key": "undefined_citations_or_references"} = 0 | ./intro_review/report_values.json sha256:d4aa4822230a @2026-10-15T14:19:11Z
not checked: maan2024characterizing: published full text closed; abstract (Europe PMC) and the 2022 preprint only; goh2022batch: published full text closed; abstract only; arjovsky2017towards: ICLR camera-ready (OpenReview) not opened; arXiv v1 read; saucie (Amodio 2019), dincer2020adversarial, mmd-resnet (Shaham 2017): bioRxiv or arXiv versions read, not published versions; Stegle 2015 (dropped by the literature track): not retrieved; HSIC equals MMD between joint and product of marginals (arXiv 2503.04820): seen as a search-result entry only; Con-AAE (Bioinformatics 39:btad162) and scBIOT (Research Square 10.21203/rs.3.rs-8310482/v1): not opened / title only; bioRxiv full text of the authors' own preprint 10.1101/2025.11.12.688061 (HTTP 429); July submission text read instead; Related work and Methods citation sites (parallel tracks)
```
