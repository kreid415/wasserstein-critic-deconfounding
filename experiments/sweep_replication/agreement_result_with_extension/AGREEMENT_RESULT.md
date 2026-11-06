# Replication across training seeds: agreement with the sweep prediction (final, with the rule's extension step)

Rule fixed before any replication score: AGREEMENT.md and PREDICTION.json at tag replication-prediction-v1 (commit 3a90d0f). Code: agreement_test.py, extension_rows.py, make_report.py (this folder's parent); tests in tests/analysis. All 895 of 895 fits are scored (scorer 10a1ca5; 285 on the workstation, 610 on JHPCE with the fixed LISI seed): the 855 replication fits plus the 40 extension fits the rule prescribes; no fit failed (0 failure rows).

Pair labels: task and decoder (d0 and d1 are decoder conditions 0 and 1 of the manifest).

## Result (raw new-seed reading, primary)

**The agreement rule is met: criterion 1 keeps 4 of 4 signs, criterion 2 mean tau-b is 0.456 with p = 0.000488.** After the extension step 1 of 78 reading(s) still do not bracket b* (status unreached_low, 'did not replicate crossing'); they take the bottom places of their pair.

| Criterion | Before the extension step | After (final) |
|---|---|---|
| 1. Large sweep gaps (>= 0.04) keep their sign | 3 of 4 | 4 of 4 |
| 2. Mean Kendall tau-b over 13 ranked pairs | 0.426 (11 pairs positive) | 0.456 (11 pairs positive) |
| 2. One-sided sign-flip p (all 2^13 patterns) | 0.000488 | 0.000488 |
| Readings not reaching b* | 8 of 78 | 1 of 78 |
| Rule met | no | yes |

The 'before' column is the result of agreement_result/ (computed before the extension fits existed, with the 8 missing readings at the bottom places). The final verdict is the 'after' column: the rule itself says to add the next grid strength once.

## Extension fits (next sweep-grid strength above the highest replication strength, seeds 0-4)

| Pair | Penalty | Highest strength before | Strength added | Seed-mean batch score at the added strength | b* | Status after | Biology at b* (reading) |
|---|---|---|---|---|---|---|---|
| atac_small d0 | barycenter | 3 | 10 | 0.6187 | 0.5221 | bracketed | 0.6775 |
| atac_small d0 | sinkhorn | 10 | 30 | 0.5956 | 0.5221 | bracketed | 0.6851 |
| immune_hum_mou d1 | barycenter | 1 | 3 | 0.5565 | 0.5536 | bracketed | 0.6152 |
| pancreas d0 | pooled | 1 | 3 | 0.5357 | 0.4924 | bracketed | 0.5845 |
| pancreas d1 | mmd | 1 | 3 | 0.6743 | 0.6841 | unreached_low | none |
| pancreas d1 | sinkhorn | 3 | 10 | 0.6961 | 0.6841 | bracketed | 0.6992 |
| sim1 d0 | pooled | 1 | 3 | 0.4868 | 0.3693 | bracketed | 0.6819 |
| sim2 d0 | sinkhorn | 30 | 100 | 0.4317 | 0.3183 | bracketed | 0.5765 |

## Large gaps

| Pair | Comparison | Sweep difference | Replication difference | Sign kept (raw) | Sign kept (slope-adjusted, sensitivity) |
|---|---|---|---|---|---|
| atac_small d0 | barycenter - reference | +0.041 | +0.013 | yes | yes |
| atac_small d1 | barycenter - discriminator | +0.046 | +0.053 | yes | yes |
| atac_small d1 | discriminator - pooled | -0.046 | -0.034 | yes | yes |
| atac_small d1 | discriminator - sinkhorn | -0.046 | -0.043 | yes | yes |

The atac_small d0 barycenter - reference gap is 0.041 in the sweep and keeps its sign with a replication difference of +0.013, so it is the closest of the four to flipping.

## Per-pair rank agreement

| Pair | Readings that reach b* (final) | tau-b before | tau-b final | tau-b slope-adjusted final | Top penalty, sweep | Top penalty, replication (final) |
|---|---|---|---|---|---|---|
| atac_large d0 | 6/6 | 0.467 | 0.467 | 0.067 | mmd | reference |
| atac_large d1 | 6/6 | 0.600 | 0.600 | 0.733 | reference | reference |
| atac_small d0 | 6/6 | 0.000 | 0.600 | 0.467 | barycenter | mmd |
| atac_small d1 | 6/6 | 0.467 | 0.467 | 0.200 | pooled | barycenter |
| immune_hum_mou d0 | 6/6 | 0.333 | 0.333 | 0.467 | mmd | discriminator |
| immune_hum_mou d1 | 6/6 | 0.067 | -0.067 | 0.333 | sinkhorn | sinkhorn |
| immune d0 | 6/6 | 0.200 | 0.200 | 0.333 | discriminator | discriminator |
| lung d0 | 6/6 | 0.600 | 0.600 | 0.467 | reference | mmd |
| lung d1 | 6/6 | 0.867 | 0.867 | 0.733 | sinkhorn | sinkhorn |
| pancreas d0 | 6/6 | 0.333 | 0.200 | 0.067 | mmd | discriminator |
| pancreas d1 | 5/6 | 0.000 | -0.067 | -0.200 | barycenter | reference |
| sim1 d0 | 6/6 | 0.733 | 0.867 | 0.467 | discriminator | discriminator |
| sim2 d0 | 6/6 | 0.867 | 0.867 | 0.733 | discriminator | discriminator |

## Reported alongside (not part of the rule)

- Sweep near-ties (difference < 0.02): 160 (matches PREDICTION.json); 155 of them keep a replication difference within +/-0.04 (before the extension: 127; a tie with an unreached side counts as outside).
- Achieved batch scores of every replication point: replication_readings.csv columns B_lo, B_hi (bracketing points) and B_max (strongest point).

## Sensitivity (slope-adjusted; sensitivity only; defined after the rule)

The rule names a slope-adjusted reading without a formula. Defined after the rule was fixed: for each penalty, the replication point nearest b* in batch score, moved to b* with one common within-penalty slope dC/dB per pair (pooled over the pair's penalties); it also gives a value for unreached cells. Result: 4 of 4 signs kept, mean tau-b 0.374, permutation p 0.000732 (before the extension: 0.374, p 0.001343). This does not replace the primary reading.

## Limits

- 1 reading(s) still never reach b* after the one allowed extension: see the extension table (status unreached_low). The rule does not allow a second extension.
- The verdict changed from 'not met as computed' to 'met' only because the rule's extension step was run; both results are kept in this folder and its parent. The extension fits were chosen by the rule's own condition (unreached readings), so they add no tuning.
- Scores come from two machines (workstation and JHPCE with the fixed LISI seed); provenance is not required to match (user decision 2026-10-28, SI-115). Pairs were fitted and scored on one machine each.
- 13 pairs give a smallest possible p of 1/8192; the test treats the pairs as exchangeable.
- Wording in the paper: "a replication across training seeds".

Files: replication_readings.csv (78 rows), replication_task_readings.csv (13 rows, per-pair summary for the Res-2b paragraph), replication_large_gaps.csv, replication_agreement_summary.json, score_validity.csv (895 units).
