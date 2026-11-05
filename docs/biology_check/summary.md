# Direct biology check on the saved sweep embeddings (reviewer comment R2.6)

Exploratory and descriptive; rules in `docs/BIOLOGY_CHECK.md` (written after sweep scores were read, before any value of this check was computed). Code commit `c4a793f` (tree clean: True). 783 saved fits on 8 tasks; no new fits. Penalties at the label-transfer strength (batch score nearest b*), seeds 100-102; plain scVI both decoders; every scored baseline setting. 15 nearest neighbours in the latent.

Differences are from plain scVI (penalties: same decoder and seed; baselines: conditioned plain scVI, mean of 3 seeds). Values: mean over tasks of the per-task seed means; 'seed SD' = mean over the 6 real tasks of the SD over the 3 seeds (corrected 2026-11-05; the first version of this page averaged all 8 tasks). Real = 6 real tasks for purity; the rare-type columns average the 4 real tasks that have rare types; markers = the 4 RNA tasks; sim = 2 simulations. Baselines with several settings: mean over settings (purity range in brackets); their seed SD is over seeds 0-4 (n/a for one-run tools: Harmony, Scanorama, PCA).

| method | purity diff (real) | seed SD | purity diff (sim) | rare purity diff | rare recall diff | marker Jaccard | Jaccard diff | n real / n RNA |
|---|---|---|---|---|---|---|---|---|
| discriminator, unconditioned | -0.077 | 0.028 | -0.001 | -0.076 | -0.129 | 0.392 | +0.001 | 6 / 4 |
| discriminator, conditioned | -0.052 | 0.030 | -0.156 | -0.107 | -0.121 | 0.398 | -0.000 | 6 / 4 |
| reference, unconditioned | -0.011 | 0.011 | -0.000 | +0.002 | +0.008 | 0.394 | +0.003 | 6 / 4 |
| reference, conditioned | -0.004 | 0.008 | +0.013 | -0.002 | +0.005 | 0.399 | +0.000 | 6 / 4 |
| pooled, unconditioned | +0.001 | 0.005 | -0.000 | +0.010 | +0.011 | 0.390 | -0.001 | 6 / 4 |
| pooled, conditioned | -0.019 | 0.008 | -0.013 | -0.024 | -0.036 | 0.398 | -0.001 | 6 / 4 |
| barycenter, unconditioned | +0.001 | 0.007 | -0.050 | +0.010 | +0.005 | 0.391 | +0.000 | 6 / 4 |
| barycenter, conditioned | -0.007 | 0.007 | -0.002 | -0.004 | +0.001 | 0.395 | -0.003 | 6 / 4 |
| mmd, unconditioned | -0.004 | 0.005 | -0.001 | +0.001 | +0.003 | 0.394 | +0.003 | 6 / 4 |
| mmd, conditioned | -0.001 | 0.006 | +0.001 | +0.003 | +0.007 | 0.397 | -0.001 | 6 / 4 |
| sinkhorn, unconditioned | -0.033 | 0.008 | -0.003 | -0.045 | -0.079 | 0.397 | +0.006 | 6 / 4 |
| sinkhorn, conditioned | -0.022 | 0.008 | -0.003 | +0.006 | +0.002 | 0.394 | -0.005 | 6 / 4 |
| scvi, unconditioned | +0.000 | 0.000 | +0.000 | +0.000 | +0.000 | 0.391 | +0.000 | 6 / 4 |
| scvi, conditioned | +0.000 | 0.000 | +0.000 | +0.000 | +0.000 | 0.399 | +0.000 | 6 / 4 |
| scvi_adv, 1 setting (mean) | -0.007 | 0.009 | -0.015 | -0.001 | +0.013 | 0.398 | -0.001 | 6 / 4 |
| scanvi, 1 setting (mean) | +0.115 | 0.003 | +0.163 | +0.069 | +0.061 | 0.390 | -0.008 | 6 / 4 |
| sysvi, 6 settings (mean, range) [-0.066, -0.031] | -0.040 | 0.023 | +0.050 | -0.045 | -0.078 | 0.392 | -0.007 | 6 / 4 |
| harmony, 7 settings (mean, range) [-0.186, -0.079] | -0.141 | n/a | +0.049 | -0.234 | -0.330 | 0.367 | -0.031 | 6 / 4 |
| scanorama, 7 settings (mean, range) [-0.058, -0.009] | -0.049 | n/a | +0.158 | -0.076 | -0.087 | 0.382 | -0.017 | 6 / 4 |
| pca, 2 settings (mean, range) [-0.045, +0.003] | -0.021 | n/a | +0.163 | -0.062 | -0.088 | 0.371 | -0.028 | 6 / 4 |

**Rare types** (< 1% of cells): per task atac_large 0, atac_small 0, immune 3, immune_hum_mou 4, lung 1, pancreas 5, sim1 0, sim2 0. Over the 48 task x penalty x decoder cells with rare types (immune, immune_hum_mou, lung, pancreas), rare-type purity is below plain scVI's in 23; the largest drop is -0.406 (discriminator, conditioned, lung).

scANVI is trained with the cell-type labels; its readouts are not label-free. Low-signal pairs are included (flagged in `biology_check.json`, `matched`: 18 of 96 task x decoder x penalty cells). No hypothesis test; numbers describe these fits only.

Tables: `tables/<task>_fit_types.csv` (per fit x type), `<task>_fits.csv`, `<task>_confusion.csv` (seed-mean label-transfer confusion), `summary.csv`, `summary_by_method.csv`.
