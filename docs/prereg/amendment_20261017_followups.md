# PREREG section 9 amendment (DRAFT): follow-up design of 2026-10-17

Status: **draft, not in force.** Written on branch `followup-build` for the user's review. It is not part of `docs/PREREG.md`
(frozen by tag `prereg-tier12-v3`); to take effect it is pasted into section 9 of PREREG.md and committed together with the
code it describes, and a new code tag is made, before the first follow-up fit and before R4-x1 (see "Order").
Glossary: SI-nn = entry nn of the decision ledger (CONSTRAINTS.md); NB-yyyymmdd-nn = lab-notebook entry; Ixxx and D-xx =
findings of the final review (docs/followup_design_decisions_20261017.md names each); λ = adversary penalty strength;
b* = the target batch score; X3, X6, X7, X8, X12, X15 = the follow-up experiments (composition dose-response, divergence
versus regularisation, reference choice, batch count, architecture robustness, KL warm-up); "conditioned decoder" = the
decoder that receives the batch label (scVI's usual setting); "unconditioned decoder" = the adversary alone removes batch
effects (the original paper's setting).

Data that existed when the user decided: X1 had 19 of 1,125 local rows scored and no follow-up had been fitted. No X1 or X13
score was opened to write this amendment or the code it describes (SI-78, SI-79).

## What changes (X3, X6, X7, X8, X12 only; every other row of the manifest is unchanged)

| SI | Experiment | Registered design before (manifest v3) | Design after (manifest v4) |
|---|---|---|---|
| SI-93 | X3, X6, X7, X8, X12 | conditioned decoder only | both decoders (cond 1 and cond 0); X15 stays conditioned. X3's oracle-reweighted rows (72 per task and decoder) run under both: the unconditioned model keeps the batch in its labels slot, so its weights are a per-cell column registered as an extra data field (`_iw_cell_vector`; same numbers as the conditioned model's (batch, cell type) table), not read through the labels slot (user answer 2026-10-17). |
| SI-94 | X8 | total cells fixed per task (immune_hum_mou 16,000, lung 6,000, sim2 3,000); batches drawn afresh at each V | every batch contributes N//16 cells (1,000 / 375 / 187) at every V = 2, 4, 8, 16; the 2-batch set lies inside the 4-batch set inside the 8- and 16-batch sets; the two subsets are two independent chains (each chain has its own random order of the eligible batches); a batch contributes the same cells at every V within a chain. Eligible batches (those with at least N//16 cells, counted on the prepared data): immune_hum_mou 19 of 23; lung 16 of 16; sim2 16 of 16. **For lung and sim2 the two chains therefore use the same 16 batches at V = 16** (they differ there only in which cells are drawn from batches larger than N//16); for V < 16 the chains can select different batches, and immune_hum_mou has 3 spare batches, so its two chains can differ at V = 16 as well. Estimates over V are conditional on the two drawn chains. Total cells = V × N//16 (sim2 at V = 16: 2,992). Draw name `nested_chain_v1`. Total cells, training steps and composition therefore change with V and are reported as covariates (I078). |
| SI-95 | X7 | one λ ("matched", inherited from the discriminator or reference arm) | λ window matched_lo / matched / matched_hi as in X3, X6, X12, X15. X8 keeps one fixed λ. |
| SI-96 | X7 | every batch as reference | in addition, a depleted-reference arm on pancreas and sim1 (spec `reference_composition`, draw `reference_fixed_v1`, `fit_paper_config.x7_draw`): the reference batch (the dose-0 automatic choice: inDrop1, Batch5) loses its RAREST cell types entirely at two levels (user answer 2026-10-17), keeping round(2c/3) and round(c/3) of its c types (count ascending, ties by name); plus a level 'none' with no type removed, a control that differs from the depleted levels in coverage only. **Design decision (user, 2026-10-17, answer "Thin all levels to one size", chosen over refilling the freed slots): the reference is thinned, without replacement, to one size per task at every level, the control included, and is not refilled.** That size is the number of cells the reference keeps at the deepest level (pancreas inDrop1 1,572 of its 1,937 cells, a 19% cut; sim1 Batch5 1,005 of its 1,761 cells, a 43% cut), so the reference does **not** keep its natural size and has the same size at all three levels; the cells are the first ones, among the kept types, of one fixed permutation of the batch (seed 50,000), the same permutation at every level, so no cell is duplicated and the control holds the first n cells of that permutation over all types; every other batch is untouched (total 16,017 and 11,341 cells). The natural X7 rows (every batch as reference) use the full references, so the depleted-arm contrast is read within the equal-size levels. The ledger wording of SI-96 ("refilled to the same size, reusing the X3 depletion-and-refill code") is superseded by this decision (a new ledger entry is recorded by the lead). Share of reference cells removed: pancreas 2.2% (5 of 14 types) and 18.8% (9 types); sim1 11.0% (2 of 7 types) and 42.9% (5 types). Rule, table and sizes in `build_paper_manifest.X7_DEPLETED`. Arms: reference, reference_fixed, mmd_ref, discriminator_ref; 3 seeds; both decoders; λ window. |
| SI-97 | X12 | 8 runs, 2^(5-2), D = AB, E = AC (resolution III) | 16 runs, 2^(5-1), E = ABCD (resolution V; flag `--x12-runs 16`). The matrix label "resolution-IV" (docs/paper_experiment_matrix.csv:14) becomes "resolution V". |
| SI-98 | X6 | JS discriminator (+ spectral norm) trained with the classifier's Adam settings, critics with the critic's | one new arm `discriminator_sn_critic_opt`: spectral-norm JS discriminator trained with the critic's Adam settings (lr 1e-4, betas (0, 0.9)); λ window matched_lo / matched / matched_hi; one adversary step. Inherits the discriminator's window (BASE_ARM). |
| I036 | X6 | spectral-norm critics computed, then discarded, a gradient penalty (which also advanced the spectral-norm power iteration) | no penalty pass for spectral-norm critics (`with_gp=not spectral_norm`). Changes the training trajectory of X6 pooled_sn only; no X1 arm uses spectral norm. The removed penalty pass also drew random numbers (`torch.rand` in `multi_class_gradient_penalty`, `src/wcd_vae/wcd/critic.py:200`), so every later random draw of a `pooled_sn` fit moves too: `pooled_sn` fits made at the new code tag are not comparable with any made earlier (none exist; keep it that way). |

## Effects on registered rules

- **R4-x1 inheritance** (section 4, "Variant arms"): add `discriminator_sn_critic_opt → discriminator`. New cells are resolved per
  task × decoder × base arm; the unconditioned X1 cells exist for every follow-up task (X1 runs both decoders), so
  R4-x1 resolves cond 0 rows with no new fits. Subsampled tasks (X8, X3, X7 depleted) inherit the full task's value.
- **Grid-edge rule**: R4-x1 may extend the X1 grid at an edge for any cell a follow-up row draws on. With SI-93 the cells
  include the unconditioned decoder for every task a follow-up uses (X3: atac_small, immune_hum_mou, sim2, pancreas; X6 and
  X12: atac_small, immune, pancreas; X7: pancreas, sim1, atac_small; X8: immune_hum_mou, lung, sim2). Extension fits, if
  any, are additional to the counts below.
- **Fresh seeds** (SI-26): all new rows use seeds 10-12; `freeze_matched_lambda.check_no_x1_reuse` still applies.
- **λ for X8**: inherited from the full task's X1 cell; results are reported at that fixed λ with the achieved batch
  score minus b* as a covariate (I048 default analysis rule), and are not called "matched batch removal".
- **λ and the reading point for the X7 depleted-reference arm** (implements SI-95: biology is read at the same
  batch-removal target): the λ window (matched_lo / matched / matched_hi) is inherited from the full task's X1 cell, as
  for the natural X7 rows. The P4 depleted contrast (depleted level minus the same-size control level, both read from the
  depleted-task fits) is read at the biology score at b\* of the **full task** (the target is defined once, on the full
  task's fits, not re-estimated on a subset: a depleted subset has fewer cells than the full task: 16,017 against
  16,382 for pancreas, 2.2% fewer, and 11,341 against 12,097 for sim1, 6.2% fewer). The achieved batch score minus b\* is reported for every
  depleted row. For a cell where b\* is not evaluable (the cell's fits do not bracket b\*), the contrast is read at the
  fixed `matched` λ instead and is reported as a fixed-λ reading, not as matched batch removal. The analysis code
  (`analysis_followups.py`, `x7()`) and its specification implement both readings.

## Counts (provisional: GPU hours from `scripts/cost_model.py`, stock-backbone throughput table, 8 lanes not applied)

| | fits v3 | fits v4 | lane-hours v3 | lane-hours v4 |
|---|---|---|---|---|
| X3 | 828 | 1,656 | 259.7 | 519.3 |
| X6 | 216 | 486 | 74.3 | 159.5 |
| X7 | 216 | 1,728 | 68.3 | 541.7 |
| X8 | 504 | 1,008 | 101.3 | 96.9 |
| X12 | 288 | 1,152 | 66.7 | 266.6 |
| X15 | 84 | 84 | 24.1 | 24.1 |
| follow-ups total | 2,136 | 6,114 | 594.4 | 1,608.1 |

(X8 cost falls because total cells per fit fall; the cost model has no decoder term, so cond 0 costs the same as cond 1.
Row-level diff: `experiments/followup_build_20261017/results/manifest_diff_v3_v4.csv`.)

## Order and what is not yet done

1. The user reviews the diff (`docs/DIFF_followup_build.md`); open questions there change what is run.
2. This text is pasted into PREREG.md section 9, the follow-up analysis specification is written (SI-68), and a new code tag is
   made. Before the tag: the X1 fit-identity check on the merged commit against a clean checkout of the v3 tag
   (`experiments/followup_build_20261017/fit_identity.sh`: 57 one-epoch fits at each commit; not run yet, the CPU was
   busy with the Sinkhorn redo) and the tests in the diff guide. A no-fit comparison of the resolved training plan of the
   same 57 rows (`plan_identity.py`, zero optimiser steps) was run on the branch head; run instructions and its result
   are in `experiments/followup_build_20261017/README.md`.
3. The v4 manifest is the follow-up manifest of record only after step 2; manifest v3 is kept unchanged.
4. A preflight GO per follow-up wave (X3 and X8 need the single-cell pack's perturbation items SC-07, SC-08).
