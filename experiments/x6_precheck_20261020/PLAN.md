# Pre-check of the optimiser-matched discriminator's strengths (follow-up X6)

Written and committed 2026-10-20, before any fit of this pre-check, and before any main-benchmark (X1) or baseline (X13)
score was examined. User decision 2026-10-20 (option a of the pre-lock review's open question).

## Question
The follow-up's confirmatory comparison is the spectrally normalised pooled critic at 1 step minus the spectrally
normalised discriminator trained with the critic's optimiser (`discriminator_sn_critic_opt`: Adam, learning rate 1e-4,
betas (0, 0.9)). That arm inherits the plain discriminator's three strengths (lo / matched / matched_hi), and the plain
discriminator learns at 1e-3. If the slower discriminator removes less batch effect at those strengths, its batch score
never reaches the matched target b*, and the confirmatory result is undecidable by construction. The arm was never
piloted. This pre-check asks one thing: does its batch-score curve over the inherited window bracket b*?

## Design (18 fits)
- Arm `discriminator_sn_critic_opt`, n_critic 1, on the pilot tasks atac_small, immune and sim1 (pancreas has no pilot
  target), both decoders (cond 0 and 1), pilot seed 100 only.
- Strengths: the plain discriminator's lo / matched / hi of the frozen pilot matched-strength record
  (`tier12_v3/rules/r4_pass2/r4_a1_record.json`, cells `<task>|<cond>|discriminator`).
- Every other setting is copied from the pilot (A1) discriminator row of the same task, decoder, seed and strength
  (adversary input `mean`, no standardisation, as in the pilot that defined b*). Experiment label `PC6`.
- Code: branch `x6-precheck` (= followup-build plus this folder; no code change). Output: `tier12_v3/x6_precheck/`.

## Primary outcome and rule (fixed now)
- Batch score B per fit = mean of the batch metrics (`prereg_rules.outcomes`, the frozen definition).
- Target b* per task and decoder = `targets[<task>|<cond>].bstar` of the same pilot record.
- Per cell (task x decoder), over the fits that did not fail: reached if min B <= b* <= max B; unreached_low if
  max B < b*; unreached_high if min B > b*; not_readable if fewer than 2 fits succeeded.
- Counted cells: the 4 cells not flagged low-signal in the pilot record (atac_small|0, atac_small|1, immune|0, sim1|0).
  The 2 low-signal cells (immune|1, sim1|1) are fitted and reported, not counted.
- Decision: `shift_up` if at least 3 of the 4 counted cells are unreached_low; `shift_down` if at least 3 are
  unreached_high; otherwise `keep`.
- Effect on X6: `shift_up` / `shift_down` moves each of this arm's three X6 strengths one step on the strength grid
  (0.1, 0.3, 1, 3, ...; `prereg_rules.lam_step`) up / down from the plain discriminator's main-benchmark matched window.
  One step only; no second pre-check. `keep` leaves the inherited window. The decision is written into the follow-up
  amendment before the follow-up analysis is locked.
- One comparison, one decision: no multiplicity correction. Single seed: the reading is a design check, not a result.

## Outputs
`decide.py` writes `decision.json` (per-cell B at each strength, b*, status, counts, decision). Reported descriptively:
the plain discriminator's 3-seed mean B at its matched strength (`matched_B` in the record).
