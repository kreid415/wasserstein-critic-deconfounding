# Final pre-publication review: design and code (wasserstein-critic-deconfounding)

Date: 2026-10-13. Real repository checked: main `99e942e` (tag `prereg-tier12-v2` = `c88cce3` plus files added on main). Nothing was changed in the repository or the data; no fix has been applied.

## Verdict
**Not ready to launch X1 yet. The Sinkhorn arm must be fixed and its pilots redone, and six small code changes plus five document amendments are needed first.** Every other completed result stands: no fit or score outside the Sinkhorn arm needs redoing, as long as the fixes are shown bit-identical on the other arms, as at the 2026-10-04 merge.

## 1. How much the review can be trusted (planted-error test)
Before the review, 15 realistic errors were planted in a separate copy. The key was sealed (SHA-256 `502bc9f377d7…`, checked before opening). An audit of all 16 reviewers' commands and tool calls found no access to the key, to other copies or to other reviewers' work.
- Caught by at least one of the 16 reviews: **14 of 15**. By model family: Codex gpt-6.1-sol 14, Claude opus-5.5 12, Claude fable-5.1 13, Codex gpt-6-astra 13. The escalated round added no catch the first round lacked.
- Missed by all: P02, a selection rule in a design document that overlapped with evaluation data. Caught once: P03, an endpoint the follow-up's rows could not measure. **The weak area is design and analysis-plan text**, so it was re-reviewed on the real repository (section 3).
- Estimated miss rate for errors of this kind: about 1 in 15 (95% Clopper–Pearson interval 0.2–32%). A clean review is therefore strong evidence, not proof.

Files: `catch_matrix.csv`; notebook NB-20261012-07.

## 2. Findings on the real repository
The 16 reports held 242 findings: 78 described planted errors, and the remaining 164 were grouped into **99 unique issues**. Each issue was re-checked by running code on the real repository (5 Claude opus-5.5 verifiers). The 38 most serious were checked again, independently, by Codex gpt-6.1-sol at max reasoning.

| Verified status | Issues |
|---|---|
| Confirmed | 65 |
| Partly confirmed | 32 |
| Not reproduced | 2 |

| Verified severity (Claude verifiers) | Issues |
|---|---|
| Fix before launch | 17 |
| Fix before analysis (before any X1 score is read) | 33 |
| Note | 49 |

The two verifier teams agreed on status for 58% of the 38 shared issues and on severity for 63%. Where they disagreed, Codex usually rated the issue higher. The largest disagreement, the Sinkhorn gradient, was escalated (section 4).

### Fix before launch (17 issues)
| Issue | Area | What is wrong | Claude verifier | Codex verifier (status / severity) |
|---|---|---|---|---|
| I006 | cluster/harvest | Grid assumed interior but lacks edge-extension rule for JHPCE scale | PARTLY | NOT / note |
| I008 | cluster/harvest | Chained job timeouts don't scale with predecessor runtime | CONFIRMED | PARTLY / fix before launch |
| I015 | data | X8 batch-subsampling fails to hold composition/size/N fixed across V | PARTLY | CONFIRMED / fix before launch |
| I028 | follow-up design | X12 design mislabeled resolution IV but is resolution III | CONFIRMED | CONFIRMED / fix before launch |
| I031 | losses/training | Sinkhorn envelope-theorem gradient double-counted | CONFIRMED | CONFIRMED / blocker |
| I033 | losses/training | Inactive-batch MMD/Sinkhorn comparisons return non-differentiable zero, crashing backward | CONFIRMED | CONFIRMED / fix before launch |
| I034 | losses/training | Non-finite critic loss detection misses intermediate/combined loss terms | CONFIRMED | CONFIRMED / fix before launch |
| I036 | losses/training | Gradient penalty wastefully computed for spectral-norm critics | CONFIRMED | not in Codex set |
| I042 | rules | Inapplicable manifest/config keys silently accepted instead of refused | CONFIRMED | CONFIRMED / fix before launch |
| I072 | runner/provenance | b* not guaranteed to lie within every X1 arm grid | CONFIRMED | PARTLY / fix before launch |
| I077 | runner/provenance | X8 per-head sample size is not actually fixed across V | CONFIRMED | CONFIRMED / fix before launch |
| I078 | runner/provenance | X8 label composition is not held fixed across batch count V | CONFIRMED | CONFIRMED / fix before launch |
| I091 | runner/provenance | R1 is not actually unaffected by trajectory-score instability | CONFIRMED | CONFIRMED / fix before analysis |
| I094 | runner/provenance | Completion/resume checks do not require saved model weights | CONFIRMED | CONFIRMED / fix before launch |
| I048 | scoring | Variant arms inherit mismatched lambda without re-validation | CONFIRMED | PARTLY / fix before launch |
| I050 | scoring | Cell identity checked by batch label/row count, not per-cell order | CONFIRMED | CONFIRMED / fix before launch |
| I051 | scoring | Legitimate zero-R-call kBET result misclassified as failure | CONFIRMED | CONFIRMED / fix before launch |

Grouped by what they block:
- **Block the X1 launch (code; needs a new tag):**
  - I031/I032: the Sinkhorn arm (section 4).
  - I033: minibatches with an inactive batch crash MMD and Sinkhorn backward. 0 crashes so far in 1,716 fit logs, but X1 sim1 and the follow-ups will hit them.
  - I034: non-finite inner critic steps go undetected.
  - I050: the scorer checks cell identity by count, not cell order. All 1,701 stored latents were verified to match their data cell by cell.
  - I051: a legitimate kBET result with zero R calls is treated as a failure.
  - I094: resume and harvest checks don't require the saved model weights.
  - I042: inapplicable design-file settings are accepted silently.
- **Block the cluster submission (launch files only):**
  - I008: the clocks of chained jobs don't allow for their predecessors' run time, so job 5 can be killed mid-run.
  - Code recheck CR2-01: the cross-host GPU gate ran at commits `6675f8c`/`2f29004`, before the tag; the fit path differs from the tag in 7 files. Re-run the L40S check at the new tag.
- **Must be amended in the documents before any X1 score is read (no refits):**
  - I072: b* may fall outside an arm's 6-value grid, so P1 needs an anchoring rule.
  - I006: an X1 grid-edge rule.
  - I091: PREREG §9 wrongly says R1 is unaffected by the unstable trajectory score. Keep the registered grids and disclose this, or amend R1, which costs about 90 extra A1 fits (estimate).
  - Design recheck D-01: P1's comparator, a 10-step discriminator, is forbidden by SI-01 and present in no design file.
  - D-02: the frozen power report says the design is UNDERPOWERED (minimum detectable difference 0.131 at 6 datasets; power 0.10 at Δ = 0.02), but PAPER_PLAN still claims 0.02 differences and equivalence can be certified.
  - I001: the P1–P4 analysis code is not written and must be frozen before X1 scores are read.
- **Block only the follow-up experiments, not X1:**
  - X8 (I015, I077, I078): batch count changes cell-type composition and per-head sample size.
  - X12 (I028): the design is resolution III, not IV.
  - X6 (I036): spectral-norm critics still compute the gradient penalty.
  - X7/X8 (I048): single-λ variants are not matched on batch removal.

Full list with evidence, locations, fixes and impact: `verified_issues.csv`.

## 3. Re-review of the least-covered areas (real repository)
- **Design documents** (claude-fable-5-1): 20 findings, 2 fix-before-launch (D-01, D-02 above), 9 before analysis, 9 notes. The follow-ups X7 and X8 are NO-GO as designed. Declared limit: the rule arithmetic (R1 blocks, R4 matching, R2/R3) was checked for consistency only, not recomputed from scores.
- **The 34 active code files earlier reviewers never or barely cited** (claude-opus-5-5): 13 findings, 1 fix-before-launch (CR2-01), 12 notes. Among the notes: the A1 identity check could not fail through its exit status; the cross-host agreement check G3 has no fail outcome; the cs-local check failed its environment criterion C1, already disclosed. Declared limit: 15 of 34 files were executed; the other 19 were read only.

## 4. Sinkhorn arm (escalated; two independent checks)
Agreed facts: the implemented gradient is twice the converged one (both final half-steps carry the cost gradient), and the final ε gets one damped update.

| | claude-fable-5-1 | Codex gpt-6-astra (ultra) |
|---|---|---|
| Conclusion | SPLIT: keep 176 fits as fits at effective weight 2λ; refit or re-describe the 36 A3 rows; flag 22 fits whose latents shrank (float32 cancellation) | REFIT all 234 |
| A1/R1 fits | c = 2.00 (per-minibatch median 1.999, 5–95% 1.93–2.04; cosine median 0.998) | stable-arithmetic control: cosine median 0.99996, c 1.97–2.03; as-is float32 R1 extension: cosine 0.28 |
| A3 (standardised input) | cosine 0.81–0.96 per fit, c 1.72–2.19; not a constant multiple | 14 of 36 fits below cosine 0.95; c up to 1.28–1.98 within one fit |
| Refit probe | λ=1, 400 epochs: as-is ≈ corrected at 2λ (within the seed floor); A3 λ=10: not reproduced | 3 epochs: λ=10 vs corrected λ=20, 30-NN overlap 40% |

**Both checks agree the 36 A3 fits do not represent the stated method.** Since R3 was decided on arms pooled together, Sinkhorn included, R3 is affected too. The A1/R1 fits approximate the method at 2λ, but not during early training, and 22 of them are numerically degenerate. **Recommendation:** fix the gradient and the final-ε convergence, then redo all 234 Sinkhorn fits and re-run the Sinkhorn parts of R1, R4-a1 and R3. Documenting "effective weight 2λ" cannot cover A3 or the degenerate fits, and a partial keep makes the Methods harder to defend.

**Cost:** the 234 completed fits took 113 lane-hours, about 14 h of wall time at 8 lanes, plus 32 scorer-hours, about 11 h at 3 scorers. A converged solver will be slower, so treat this as a lower bound.

Declared limits: Fable's A3 refit probe ran 100 of 400 epochs on one seed pair, so the size of the A3 effect is moderate-confidence. The probes ran on CPU, not the original GPU. Astra's float32 magnitudes are CPU measurements, not the historical CUDA gradients.

## 5. What happens to completed work
- **Redo:** the 234 Sinkhorn fits (A1 180, R1 extension 18, A3 36), and the Sinkhorn cells of R1, R4-a1 and R3.
- **Stands, subject to one check:** all other A1/extension fits, A2/A3 non-Sinkhorn rows, the R2 record, the 250 filler fits and the 19 local X1 scores. Every other code fix is a guard that should leave results unchanged. That must be shown, before the new tag is used, by bit-identity of fits and scores on the non-Sinkhorn arms, as at the 2026-10-04 merge. If any difference appears, the affected stage is redone.
- The X1 manifest's Sinkhorn λ grid will change after R1 is re-run; the other arms' grids won't.

## 6. Disclosures about this review
- Each review was scored against the key by a reasoning-model call; 8 judgements were checked by hand, and all agreed.
- Grouping findings into issues was model-assisted. One labelling call was refused by the platform's safety filter (ordinary code-review text). I did not resend it; its 44 findings were labelled by keyword rules instead. This affects only how findings were grouped, not their verification.
- Verifier limits, declared by the sub-agents, include:
  - small or synthetic probe inputs;
  - 158 of the 250 filler latents scored locally so far;
  - A1 data used as a stand-in for X1 in I072;
  - no cluster command run (sacct/squeue stubbed).
- Codex usage at the end: 5-hour window used up (resets ~10:45 EDT), weekly 69% left, credits 1,702.50, down from 2,117.76 before the escalated Sinkhorn check.
