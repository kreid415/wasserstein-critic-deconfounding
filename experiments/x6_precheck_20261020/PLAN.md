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
- Rows are read from the frozen pilot rule-output manifest `tier12_v3/rules/r4_pass2/m.r4a1.tsv` (it holds the pilot
  extension strengths such as 0.03). Every other setting is copied from the pilot (A1) discriminator row of the same task, decoder, seed and strength
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

## Addendum 2026-10-21: second arm (`pooled_sn` at 1 step)
Written 2026-10-21, before any fit of this pre-check (no fit of either arm exists) and without opening any X1 or X13 score.
User decision 2026-10-21 (ledger SI-107). The text above is unchanged; where it says "18 fits", "this arm" or "the decision",
read the numbers and the rule below.

- **Why.** The other arm of the confirmatory contrast, the spectrally normalised pooled critic at 1 step (`pooled_sn`,
  n_critic 1), inherits the strengths of the main pooled critic (gradient penalty, 5 steps) and was never piloted. If its
  batch score never reaches b* over the inherited window, the contrast is undecidable for the same reason as above.
  It gets the same check.
- **Design: 36 fits.** The 18 fits above, unchanged, plus 18 fits of `pooled_sn` at n_critic 1: atac_small, immune and sim1,
  both decoders (cond 0 and 1), pilot seed 100, three strengths. The strengths are the lo / matched / hi of the cells
  `<task>|<cond>|pooled` of the same pilot record. Each row is the pilot (A1) `pooled` row of the same task, decoder, seed and
  strength, with three changes: experiment label `PC6`, arm `pooled_sn`, n_critic `1` (the pilot pooled critic took 5 steps).
  Every other setting is unchanged (adversary input `mean`, no standardisation). Rows come from the same frozen pilot
  rule-output manifest; tags follow the builder's own rule. `build_manifest.py` builds and checks all 36.
- **Rule, per arm.** The primary outcome, the status of each cell and the decision rule above apply unchanged to each arm over
  its own six cells and the same 4 counted cells (atac_small|0, atac_small|1, immune|0, sim1|0). Each arm gets its own decision
  (`keep`, `shift_up` or `shift_down`); the two are not combined and no multiplicity correction is made. The base arm of
  `discriminator_sn_critic_opt` in the pilot record is `discriminator`; that of `pooled_sn` is `pooled`.
- **Output.** `decision.json` holds `decisions` (arm -> decision, the part `freeze_matched_lambda.py --precheck-decision`
  reads) and `per_arm` (counts, counted cells and the six cells of each arm, with b*, the base arm's matched strength and its
  3-seed `matched_B`).
- **Effect on X6.** `pooled_sn` at n_critic 1: each of its three X6 strengths moves one step up or down the strength grid by its
  own decision, from the main pooled critic's matched window. `pooled_sn` at n_critic 5 and every arm other than the two
  checked ones keep their base arm's strengths. One step only, no second pre-check; the decision is applied as described in
  PREREG section 9 (entry of 2026-10-20, timing paragraph).
- **Code for the new fits.** The `pooled_sn` fits must be made with code that holds the finding I036 fix (spectral-norm
  critics compute no gradient penalty). The branch `x6-precheck` and `x6-lock` both do (`with_gp=not self.spectral_norm` in
  `scripts/scvi_adversarial_plan.py`); fits made without it are not comparable. The 36-row manifest and the launch commit are
  pinned at launch.
