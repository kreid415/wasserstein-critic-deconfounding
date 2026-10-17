# Review guide: follow-up build (branch `followup-build`, base c241af6)

Implements the user's decisions of 2026-10-17: SI-93 (unconditioned decoder for X3/X6/X7/X8/X12), SI-94 (X8 nested batches,
fixed cells per batch), SI-95 (X7 λ window), SI-96 (X7 depleted-reference arm), SI-97 (X12 16 runs), SI-98 (X6 new arm), plus
review fix I036 (gradient penalty for spectral-norm critics). Round 2 (user answers of 2026-10-17): X7 depletion removes the
rarest types first with the reference held at one size; X3's oracle-weighted rows run unconditioned too; the new arms run
under both decoders. Terms: λ = adversary penalty strength; V = number of batches; cond 1 / cond 0 = conditioned /
unconditioned decoder; X1 = the main benchmark; "tag" = git tag `prereg-tier12-v3` (10a1ca5), which freezes `scripts/` and
`src/`. **No new tag was made, no GPU was used, no X1 or X13 score was opened.**

## What changed

| File | Change | SI |
|---|---|---|
| `scripts/build_paper_manifest.py` | flag `--followup-design {v3,v4}` (default v3 = byte-identical to the committed v3 manifest). v4: cond 0 rows for X3/X6/X7/X8/X12 (X3's weighted rows included; X15 conditioned only); X8 spec with `cells_per_batch`, `draw=nested_chain_v1`; X7 λ window; X7 depleted rows from table `X7_DEPLETED` (`draw=reference_fixed_v1`); X6 arm `discriminator_sn_critic_opt` | 93-96, 98 |
| `scripts/fit_paper_config.py` | `x8_draw` (nested chains); `x7_draw` (reference loses its rarest types, thinned to `n_reference`; other batches untouched); `subsample` dispatches both (a batches spec without `draw` is the earlier draw, kept so v3 rows resolve); `resolve_reference` requires reference == depleted batch for `reference_composition`; new arm in `ADVERSARIAL` and `_NETWORK_ARMS` | 94, 96, 98 |
| `scripts/scvi_adversarial_plan.py` | arm `discriminator_sn_critic_opt` (parsed as `discriminator_sn`, critic Adam settings); `with_gp=not spectral_norm` in the critic's inner step; **unconditioned importance weights**: `_iw_cell_vector` builds the per-cell weight column, registered as an extra data field of the unconditioned model (`iw_cell` plan option), read in `_iw_weights`, and removed from the model's registry after training so the saved model reloads without it; the conditioned table path is untouched; the refusal of unconditioned weights is removed | 98, I036, 93 |
| `scripts/prereg_rules.py` | `BASE_ARM[discriminator_sn_critic_opt] = discriminator` | 98 |
| `scripts/cost_model.py` | arm label strips `_critic_opt` (same step cost as `discriminator_sn`) | 98 |
| `manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv` | 10,764 rows (v3: 6,930), built with `--x12-runs 16 --followup-design v4`; v3 untouched | all |
| `docs/prereg/amendment_20261017_followups.md` | draft PREREG section 9 amendment (not pasted into PREREG.md) | all |
| `experiments/followup_build_20261017/` | `diff_report.py`, `static_x1_paths.py`, `select_x1_rows.py`, `fit_identity.sh`, `score_identity.sh`, README, results | proof |
| `tests/scvi/test_followup_v4.py` (new), `test_adversarial_plan_v2.py`, `test_runner_and_manifest_missing_arms.py`, `test_missing_arms_plan.py` | see Tests | all |

## Row diff v3 to v4 (`experiments/followup_build_20261017/results/manifest_diff_v3_v4.csv`)

| experiment | decoder | rows v3 | rows v4 | added | removed |
|---|---|---|---|---|---|
| X3 | cond 1 / cond 0 | 828 / 0 | 828 / 828 | 0 / 828 | 0 / 0 |
| X6 | cond 1 / cond 0 | 216 / 0 | 243 / 243 | 27 / 243 | 0 / 0 |
| X7 | cond 1 / cond 0 | 216 / 0 | 792 / 792 | 576 / 792 | 0 / 0 |
| X8 | cond 1 / cond 0 | 504 / 0 | 504 / 504 | 504 / 504 | 504 / 0 |
| X12 | cond 1 / cond 0 | 288 / 0 | 576 / 576 | 504 / 576 | 216 / 0 |
| X15 | cond 1 | 84 | 84 | 0 | 0 |

X1, X13, A1-A3 and X15: **identical in v3 and v4 (4,878 rows, tags and all columns)**; `diff_report.py` fails otherwise and was
shown to reject a one-cell change. X8 rows are replaced (new draw); X12 conditioned rows keep 72 of 288 (settings that
coincide between the 8- and 16-run designs). Provisional cost (`cost_model.py`, stock-backbone throughput; no decoder term):
follow-ups 5,970 fits and 1,564.1 lane-hours against 2,136 and 594.4 in v3 (X3 519.3, X6 159.5, X7 497.8, X8 96.9, X12 266.6,
X15 24.1), before any R4-x1 grid-edge extension fits. Of these: unconditioned weighted X3 rows 288 fits, 94.6 lane-hours;
depleted-reference arm (both decoders) 288 fits, 87.9; new X6 arm (both decoders) 54 fits, 10.9.

## X7 depleted-reference arm: rule, sizes, share removed (derived from prepped_scib; re-derived by a test when PREPPED_DIR is set)

Rule: reference = automatic choice (inDrop1, Batch5). Level 'two_thirds' / 'one_third' keeps round(2c/3) / round(c/3) of the c
types by removing the RAREST types (count ascending, ties by name). The reference is drawn, without replacement, to one size per
task, the number of cells it keeps at the deeper level (first cells of one fixed permutation of the batch among the kept
types); every other batch is kept whole. No replacement or duplication occurs.

| task | reference cells in the task | c types | level | types removed | reference cells removed | share removed | reference size used (both levels) | total cells |
|---|---|---|---|---|---|---|---|---|
| pancreas (inDrop1) | 1,937 | 14 | two_thirds | 5 | 42 | 2.2% | 1,572 (323 more cells of kept types thinned) | 16,017 |
| pancreas | | | one_third | 9 | 365 | 18.8% | 1,572 (none thinned) | 16,017 |
| sim1 (Batch5) | 1,761 | 7 | two_thirds | 2 | 194 | 11.0% | 1,005 (562 more thinned) | 11,341 |
| sim1 | | | one_third | 5 | 756 | 42.9% | 1,005 (none thinned) | 11,341 |

The reference therefore has the same size at both levels (pancreas 1,572 of 1,937 natural, sim1 1,005 of 1,761); it differs
from the natural X7 reference (full tasks, 1,937 and 1,761 cells) and the total differs from the full task (16,382 and
12,097). There is no level-0 control at the same size in the manifest (not asked for; it would add 72 fits per decoder: 2 tasks x 3 seeds x 4 arms x 3 λ).
Types left in the reference: pancreas 9 and 5; sim1 5 and 2.

## X3 oracle-weighted rows with the unconditioned decoder

The unconditioned model keeps the batch in its labels slot, so cell types cannot ride there. The weights are one value per cell,
`w[batch_i, cell type_i]` (`_iw_cell_vector`), stored as an extra numeric column registered with the model's data manager
(`iw_cell`), read per minibatch in `_iw_weights`, and used exactly as the table weights are (self-normalised averages in the
adversary loss). The conditioned path still builds its (batch, cell type) table and reads it through the labels slot: its code
is not touched. After training the extra field is removed from the registry (the saved model reloads without the column: checked on a
toy fit). Real-data smoke fit (CPU, 1 epoch, nice): X3 sim2 discriminator dose 50 unconditioned weighted: finite, 17,872 cells.

## X1 result-identity (requirement: every X1 fit path unchanged)

1. Static, run: `static_x1_paths.py` rebuilds the v3 manifest byte for byte, lists the 17 functions changed since the tag (git
   diff + ast), and evaluates each declared guard on all 3,280 X1/X13 fit rows: 0 rows reach any changed behaviour (new arm name only;
   `subsample` absent from X1; no `_sn` arm in X1; the weight code needs an `iw` row with cond 0). The 288 conditioned weighted X3 rows
   of v3 run changed lines of `fit_adversarial_scvi`, `__init__`, `_iw_weights`, `training_step`, `subsample` with unchanged behaviour
   by construction; lines that every adversarial row runs with an unchanged value (`w = ... if self._iw_on else None`,
   `with_gp=not self.spectral_norm`) are likewise only covered dynamically. A changed function without a declared guard fails the check
   (shown by running it against tag v2). Reachability argument, not a bit comparison.
2. Dynamic, **not run** (CPU shared with the Sinkhorn redo): `select_x1_rows.py` now selects 57 rows (the 53 X1/X13 code paths, plus the
   4 conditioned oracle-weighted X3 rows) and `fit_identity.sh` (old = tag checkout, new = candidate; 1 epoch, CPU; exit 1 unless every
   row gives equal latents). Run before the code tag.

## Tests (check_tests.py, base c241af6, four touched files: OK, 66 passed, 1 skipped; plus 71 passed in the prereg manifest, X3, baseline-guard and legacy tests)

- `test_followup_v4.py::test_v4_manifest_design`: counts per experiment and decoder, weighted rows under both decoders, X8 draw
  fields, depleted rows (reference == depleted batch, `n_reference`, `n_cells`, types from the table), X6 arm; the v3 manifest fails
  the same checker.
- `::test_x12_sixteen_runs_are_resolution_v`: main effects and two-factor interactions orthogonal; the 8-run design is rejected.
- `::test_x8_draw_is_nested_with_equal_cells_per_batch`: equal cells per batch, nested batches and cells, two chains; the earlier
  draw fails the nesting check.
- `::test_x7_depleted_reference_spec`: types removed, reference exactly `n_reference` at both levels, other batches whole, bad specs
  refused, reference must be the depleted batch.
- `::test_x7_depleted_table_and_x8_eligibility_follow_the_rules` (PREPPED_DIR): re-derives `X7_DEPLETED` (rarest-first rule, sizes) and the ≥16 eligible X8 batches.
- `test_missing_arms_plan.py`: unconditioned weighted fits for the four weighted arms (finite latents); unit weights reproduce the
  unweighted fit for both decoders (pooled and mmd bit for bit); incomplete and negative weights refused;
  `test_unconditioned_weights_equal_the_conditioned_weights_of_the_same_cells` (per-cell vector == the conditioned table lookup and
  the known answer; a different table is rejected); `test_importance_weights_change_the_unconditioned_loss` (minibatches carry the
  weights {1, 2}, the fit differs from the unweighted one, the extra field is gone afterwards).
- `test_adversarial_plan_v2.py`: new arm in the every-arm-trains-finite list; optimiser settings and gradient-penalty calls
  (I036, SI-98). `test_runner_and_manifest_missing_arms.py`: the arm added to the `heads` table.
- `fl_lint` on the touched scripts: nothing at medium or above (two low: `set -e` absent in the copied identity shells).
  `check_extra` accepts all 5,970 follow-up rows; `fl_unique_outputs` ok.
- Real-data CPU smoke fits (1 epoch, nice, 1 thread; outputs in /tmp, discarded): X3 sim2 unconditioned weighted, X7 sim1 depleted
  reference (cond 0 reference arm; cond 1 discriminator_ref), X8 sim2 V=4 unconditioned (748 cells), X6 atac_small new arm unconditioned:
  all finite, cell counts as designed. These are smoke checks, not X1-identity evidence.

## Decisions I made (no user question needed, SI-92)

- v4 is a flag, v3 stays the default: the committed v3 manifest, `test_x3_si41` and the base-subsample test stay valid. Switching the default is the user's later call.
- New arm name `discriminator_sn_critic_opt`, a separate arm so the analysis cannot confuse it with `discriminator_sn`.
- X8 spec stores `n_cells` = V × N//16 (cells drawn) so epochs and cost are right; N//16 = 1,000 / 375 / 187; ≥16 eligible batches for all three tasks (checked on the data).
- X7 depleted: reference thinned to the deeper level's size by a fixed permutation (`X7_PERMUTATION_SEED` 50,000); the depleted rows' λ is inherited from the full task's X1 cell.
- Unconditioned weights ride in an extra numeric data field, removed again after training (the minimal way to get a per-cell value into the minibatch without the model reading it).

## Open questions

None that block the next step. Two things to know: (1) there is no same-size level-0 reference control in X7 (natural X7 rows use full references of 1,937 / 1,761 cells); (2) reference size 1,005 on sim1 keeps only 2 cell types at the deeper level (43% of the reference removed). Neither was asked about; both are one-table changes.

## NOT done

- No new tag; no GPU fit; the X1 fit-identity run (above); PREREG.md, ANALYSIS_SPEC.md, PAPER_PLAN.md untouched (the amendment is a draft file).
- Document-only defaults of the brief section 4 (I028 label, D-07 / I057 PAPER_PLAN and matrix drift, D-15, D-16 X15 in the stage table, I088/I089/I092 wording, I087 contrast list); the follow-up analysis specification (SI-68) and endpoint definitions (D-09).
- Preflight records per follow-up wave (X3/X8 need SC-07/SC-08); GPU smoke fits; launchers under `experiments/` still point at v3.
- Notebook and ledger entries (returned in the structured report; CONSTRAINTS.md and notebook/ untouched).
