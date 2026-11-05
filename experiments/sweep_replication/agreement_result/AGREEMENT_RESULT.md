# Replication across training seeds: agreement with the sweep prediction

Rule fixed before any replication score: AGREEMENT.md and PREDICTION.json at tag replication-prediction-v1 (commit 3a90d0f). Code: agreement_test.py (this folder), tests in tests/analysis/test_replication_agreement.py. All 855 of 855 replication fits are scored (scorer 10a1ca5; 270 on the workstation, 585 on JHPCE with the fixed LISI seed); no fit failed (0 failure rows).

Pair labels: task and decoder (d0 and d1 are decoder conditions 0 and 1 of the manifest).

## Result (raw new-seed reading, primary)

**The agreement rule is not met as computed: criterion 1 keeps 3 of 4 signs, criterion 2 is met.** The one sign that is not kept has no replication value: the barycenter critic does not reach b* in atac_small, decoder 0, so its reading is missing. The rule's own next step for such a cell (add the next grid strength outward once, five seeds) has not been run. The verdict is therefore provisional until that step is run or declined.

| Criterion | Result |
|---|---|
| 1. Large sweep gaps (>= 0.04) keep their sign | 3 of 4 |
| 2. Mean Kendall tau-b over 13 ranked task x decoder pairs | 0.426 (11 of 13 pairs positive) |
| 2. One-sided sign-flip permutation p (all 2^13 patterns) | 0.000488 (= 4/8192) |

### Large gaps

| Pair | Comparison | Sweep difference | Replication difference | Sign kept (raw) | Sign kept (slope-adjusted, sensitivity) |
|---|---|---|---|---|---|
| atac_small d0 | barycenter - reference | +0.041 | undefined (no reading) | no | yes |
| atac_small d1 | barycenter - discriminator | +0.046 | +0.053 | yes | yes |
| atac_small d1 | discriminator - pooled | -0.046 | -0.034 | yes | yes |
| atac_small d1 | discriminator - sinkhorn | -0.046 | -0.043 | yes | yes |

### Readings that do not bracket b* in the replication

8 of 78 penalty readings never reach b* in the new seeds (status unreached_low). They take the bottom places of their pair (ANALYSIS_SPEC 3.1). The new-seed mean batch score at the strongest strength fitted is 0.0002 to 0.0108 below b*.

| Pair | Penalty | Strongest strength fitted | Seed-mean batch score there | b* |
|---|---|---|---|---|
| atac_small d0 | barycenter | 3 | 0.5199 | 0.5221 |
| atac_small d0 | sinkhorn | 10 | 0.5207 | 0.5221 |
| immune_hum_mou d1 | barycenter | 1 | 0.5522 | 0.5536 |
| pancreas d0 | pooled | 1 | 0.4922 | 0.4924 |
| pancreas d1 | mmd | 1 | 0.6743 | 0.6841 |
| pancreas d1 | sinkhorn | 3 | 0.6820 | 0.6841 |
| sim1 d0 | pooled | 1 | 0.3602 | 0.3693 |
| sim2 d0 | sinkhorn | 30 | 0.3075 | 0.3183 |

## Per-pair rank agreement

| Pair | Readings that reach b* | tau-b (raw) | tau-b (slope-adjusted) | Top penalty, sweep | Top penalty, replication |
|---|---|---|---|---|---|
| atac_large d0 | 6/6 | 0.467 | 0.067 | mmd | reference |
| atac_large d1 | 6/6 | 0.600 | 0.733 | reference | reference |
| atac_small d0 | 4/6 | 0.000 | 0.467 | barycenter | mmd |
| atac_small d1 | 6/6 | 0.467 | 0.200 | pooled | barycenter |
| immune_hum_mou d0 | 6/6 | 0.333 | 0.467 | mmd | discriminator |
| immune_hum_mou d1 | 5/6 | 0.067 | 0.333 | sinkhorn | sinkhorn |
| immune d0 | 6/6 | 0.200 | 0.333 | discriminator | discriminator |
| lung d0 | 6/6 | 0.600 | 0.467 | reference | mmd |
| lung d1 | 6/6 | 0.867 | 0.733 | sinkhorn | sinkhorn |
| pancreas d0 | 5/6 | 0.333 | 0.067 | mmd | discriminator |
| pancreas d1 | 4/6 | 0.000 | -0.333 | barycenter | reference |
| sim1 d0 | 5/6 | 0.733 | 0.467 | discriminator | discriminator |
| sim2 d0 | 5/6 | 0.867 | 0.867 | discriminator | discriminator |

## Reported alongside (not part of the rule)

- Sweep near-ties (difference < 0.02): 160 (matches PREDICTION.json); 127 of them keep a replication difference within +/-0.04 (a tie with an unreached side counts as outside).
- Achieved batch scores of every replication point: replication_readings.csv columns B_lo, B_hi (bracketing points) and B_max (strongest point).

## Sensitivity (slope-adjusted; sensitivity only)

The rule names a slope-adjusted reading without a formula. Defined here: for each penalty, the replication point nearest b* in batch score, moved to b* with one common within-penalty slope dC/dB per pair (pooled over the pair's penalties); it also gives a value for unreached cells. Result: 4 of 4 signs kept, mean tau-b 0.374, permutation p 0.001343. This does not replace the primary reading.

## Limits

- Eight readings are missing (above); the extension fits the rule prescribes for them (5 seeds x 8 cells = 40 fits) have not been run. Only the atac_small, decoder 0 barycenter cell affects criterion 1; all eight affect the ranks in criterion 2.
- Scores come from two machines (workstation and JHPCE with the fixed LISI seed); provenance is not required to match (user decision 2026-10-28, SI-115). Pairs were scored on one machine each.
- 13 pairs give a smallest possible p of 1/8192; the test treats the pairs as exchangeable.
- Wording in the paper: "a replication across training seeds".

Files: replication_readings.csv (78 rows: sweep and replication reading, ranks, bracketing strengths and batch scores per pair and penalty), replication_task_readings.csv (13 rows: per-pair readings summary for the Res-2b paragraph), replication_large_gaps.csv, replication_agreement_summary.json.
