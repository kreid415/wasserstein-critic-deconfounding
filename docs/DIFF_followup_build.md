# Review guide: follow-up build (branch `followup-build`, base c241af6)

Implements the user's decisions of 2026-10-17: SI-93 (unconditioned decoder for X3/X6/X7/X8/X12), SI-94 (X8 nested batches,
fixed cells per batch), SI-95 (X7 λ window), SI-96 (X7 depleted-reference arm), SI-97 (X12 16 runs), SI-98 (X6 new arm), plus
review fix I036 (gradient penalty for spectral-norm critics). Terms: λ = adversary penalty strength; V = number of batches;
cond 1 / cond 0 = conditioned / unconditioned decoder; X1 = the main benchmark; "tag" = git tag `prereg-tier12-v3`
(10a1ca5), which freezes `scripts/` and `src/`. **No new tag was made, no fit was run, no X1 or X13 score was opened.**

## What changed

| File | Change | SI |
|---|---|---|
| `scripts/build_paper_manifest.py` | new flag `--followup-design {v3,v4}` (default v3 = byte-identical to the committed v3 manifest). v4: cond 0 rows for X3/X6/X7/X8/X12 (X3's oracle-weighted rows stay cond 1); X8 spec with `cells_per_batch`, `draw=nested_chain_v1`; X7 λ window; X7 depleted rows from table `X7_DEPLETED`; X6 arm `discriminator_sn_critic_opt` | 93-96, 98 |
| `scripts/fit_paper_config.py` | `x8_draw` (nested chains); `subsample` dispatches `draw=nested_chain_v1` to it (spec without `draw` = earlier draw, kept so v3 rows resolve); `reference_composition` kind = `x3_draw` with the reference as the depleted batch, `resolve_reference` requires reference == depleted batch; new arm added to `ADVERSARIAL` and `_NETWORK_ARMS` | 94, 96, 98 |
| `scripts/scvi_adversarial_plan.py` | arm `discriminator_sn_critic_opt` (parsed as `discriminator_sn`, trained with the critic Adam settings); `with_gp=not spectral_norm` in the critic's inner step | 98, I036 |
| `scripts/prereg_rules.py` | `BASE_ARM[discriminator_sn_critic_opt] = discriminator` (R4-x1 resolves its λ from the discriminator) | 98 |
| `scripts/cost_model.py` | arm label strips `_critic_opt` (same step cost as `discriminator_sn`) | 98 |
| `manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv` | 10,476 rows (v3: 6,930), built with `--x12-runs 16 --followup-design v4`; v3 untouched | all |
| `docs/prereg/amendment_20261017_followups.md` | draft PREREG section 9 amendment (not pasted into PREREG.md) | all |
| `experiments/followup_build_20261017/` | `diff_report.py`, `static_x1_paths.py`, `select_x1_rows.py`, `fit_identity.sh`, `score_identity.sh`, README, results | proof |
| `tests/scvi/test_followup_v4.py` (new), `test_adversarial_plan_v2.py`, `test_runner_and_manifest_missing_arms.py` | see Tests | all |

## Row diff v3 to v4 (`experiments/followup_build_20261017/results/manifest_diff_v3_v4.csv`)

| experiment | decoder | rows v3 | rows v4 | added | removed |
|---|---|---|---|---|---|
| X3 | cond 1 / cond 0 | 828 / 0 | 828 / 540 | 0 / 540 | 0 / 0 |
| X6 | cond 1 / cond 0 | 216 / 0 | 243 / 243 | 27 / 243 | 0 / 0 |
| X7 | cond 1 / cond 0 | 216 / 0 | 792 / 792 | 576 / 792 | 0 / 0 |
| X8 | cond 1 / cond 0 | 504 / 0 | 504 / 504 | 504 / 504 | 504 / 0 |
| X12 | cond 1 / cond 0 | 288 / 0 | 576 / 576 | 504 / 576 | 216 / 0 |
| X15 | cond 1 | 84 | 84 | 0 | 0 |

X1, X13, A1-A3 and X15: **identical in v3 and v4 (4,878 rows, tags and all columns)**; `diff_report.py` fails otherwise and was
shown to reject a one-cell change. X8 rows are replaced (new draw); X12 conditioned rows keep 72 of 288 (settings that
coincide between the 8- and 16-run designs). Provisional cost (`cost_model.py`, stock-backbone throughput; no decoder term):
follow-ups 5,682 fits and 1,462.3 lane-hours against 2,136 and 594.4 in v3 (X3 424.8, X6 159.4, X7 490.6, X8 96.8, X12 266.6,
X15 24.1), before any R4-x1 grid-edge extension fits. Of the X7 total, the depleted arm (both decoders) is 288 fits and about 80.7 lane-hours.

## X1 result-identity (SI requirement: every X1 fit path unchanged)

1. Static, run: `static_x1_paths.py` rebuilds the v3 manifest byte for byte, lists the 14 functions changed since the tag
   (found by git diff and ast), and evaluates each function's declared guard on all 3,280 X1/X13 fit rows: 0 rows reach any
   changed behaviour (new arm name only; `subsample` absent from X1; no `_sn` arm in X1). A changed function without a declared
   guard fails the check (shown by running it against tag v2). This is argument by reachability, not a bit-comparison.
2. Dynamic, **not run** (CPU in use by the Sinkhorn redo, as instructed): `select_x1_rows.py` (53 rows, every X1 arm × decoder ×
   adversary input × standardisation, plus scANVI and sysVI) and `fit_identity.sh` (old = tag checkout, new = candidate;
   1 epoch, CPU, exits 1 unless all rows give equal latents). Run before the code tag.

## Tests (check_tests.py, base c241af6, touched files only: OK, 40 passed, 3 skipped without data; with PREPPED_DIR the data test also passes)

- `test_followup_v4.py::test_v4_manifest_design`: counts per experiment × decoder, X3 weighted rows cond 1 only, X8 draw fields,
  depleted rows name the depleted batch, X6 arm (BASE_ARM, ADVERSARIAL, cost label); the v3 manifest fails the same checker.
- `::test_x12_sixteen_runs_are_resolution_v`: 5 main effects and 10 two-factor interactions orthogonal; the 8-run design is
  rejected (SI-97's one test).
- `::test_x8_draw_is_nested_with_equal_cells_per_batch`: equal cells per batch, nested batches and cells, two chains,
  ineligible batches; the earlier per-V draw fails the nesting check.
- `::test_x7_depleted_reference_spec`: types removed, N kept, reference must be the depleted batch.
- `::test_x7_depleted_table_and_x8_eligibility_follow_the_rules` (needs PREPPED_DIR): re-derives `X7_DEPLETED` and the ≥16 eligible
  batches for X8 from the prepped data.
- `test_adversarial_plan_v2.py`: new arm in the every-arm-trains-finite list; `test_discriminator_optimiser_and_gradient_penalty_calls`
  (Adam lr/betas of each discriminator arm; no penalty pass for `pooled_sn`, with `pooled` as positive control).
- `test_runner_and_manifest_missing_arms.py`: the arm added to the `heads` table. Also run: tests/prereg manifest tests, test_x3_si41,
  test_legacy_retired (48 passed, 9 skipped). The full suites were not run (CPU budget).
- `fl_lint` on the touched scripts: no finding at medium or above (two low: `set -e` absent in the two copied identity shells).

## Decisions I made (no user question needed, SI-92)

- v4 is a flag, v3 stays the default: the committed v3 manifest, `test_x3_si41` and the base-subsample test stay valid. Switching the default is the user's later call.
- New arm name `discriminator_sn_critic_opt` (a separate arm, so the analysis cannot confuse it with `discriminator_sn`).
- X8 spec stores `n_cells` = V × N//16 (cells actually drawn) so the row's epochs and the cost model are correct; N//16 = 1,000 (immune_hum_mou), 375 (lung), 187 (sim2). Eligible = batches with ≥ N//16 cells (≥16 for all three tasks, checked on the data).
- X7 depleted arm: reference = the automatic choice (pancreas inDrop1, sim1 Batch5); levels keep round(c × 2/3) and round(c × 1/3) of the c cell types by removing the most abundant types; one N per task = cells left after the deeper level (14,487; 10,530). It also takes the unconditioned decoder (SI-93 read as covering everything inside X7) and the λ window (SI-95).

## Open questions that change what is run (also in the structured report)

1. X3 oracle-reweighted rows (288 fits, cond 1): the weights use the model's labels slot, which an unconditioned model uses for the batch, so the code refuses them. v4 has no unconditioned weighted rows (SI-93 says "also run"; this is the one exception). Options: leave out (as built); or add index-based weights for cond 0 (new code, +288 fits, about +95 lane-hours).
2. X7 depleted arm, size confound: with the brief's rule the reference batch shrinks from 1,707 to 360 and 42 cells (pancreas) and from 1,541 to 719 and 194 (sim1), so coverage and reference size fall together; the 42-cell reference also leaves most minibatches with no reference cell (the critic arms were not smoke-tested on this). Options: as built; or remove the rarest types (table change only; reference size stays near 1,700 / 1,500 and only coverage of rare types falls); or run both rules (+288 fits).
3. Whether the depleted arm and the new X6 arm also run unconditioned (as built, from SI-93) or conditioned only (the brief's counts: +48 and +27 fits).

## NOT done

- No new tag; no fit; the X1 fit-identity run (above); PREREG.md, ANALYSIS_SPEC.md, PAPER_PLAN.md untouched (amendment is a draft file).
- Brief section-4 wording defaults that are documents, not code (I028 label, D-07 / I057 PAPER_PLAN and matrix drift, D-15, D-16 X15 in the stage table, I088/I089/I092 wording, I087 contrast list); the follow-up analysis specification (SI-68) and endpoint definitions (D-09).
- Preflight records for the follow-up waves (X3/X8 need SC-07/SC-08); smoke fits of the new arm and the new draws on the GPU; the launchers under `experiments/` still point at v3.
- Notebook and ledger entries (returned to the lead in the structured report; CONSTRAINTS.md and notebook/ untouched).
