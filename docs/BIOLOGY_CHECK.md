# Direct biology check on the saved sweep embeddings (reviewer comment R2.6)

Written 2026-11-05 on branch `biology-check` (from `origin/sweep-paper` at 8595a16), before any value of this check was computed.
**Exploratory and descriptive.** These rules were written AFTER sweep scores had been read: the batch and biology summary scores
of every sweep fit, b*, the matched strengths and the label-transfer macro F1 at the matched strength had all been computed and read in
the sweep-analysis development runs (`sweep_analysis_dev/`, `sweep_analysis/draft_8595a16`). No per-cell-type purity, per-type
label-transfer recall, confusion table or marker overlap of a sweep or baseline fit had been computed when this file was committed.
(The older script `scripts/run_biology.py` belongs to an earlier design that trained its own fits; nothing from it is used here.)
No new fits are made, no hypothesis test is run and no claim of significance is made: every number is reported per method with
its spread over seeds.

## 1. Which fits

Inputs: the sweep-analysis input listing written by `scripts/sweep_inputs.py` (`inputs/manifest.tsv`, `scores_list.txt`,
`failures.csv`, `latents.tsv`, `inputs.json`), read with the same code as `scripts/analysis_sweep.py`, so the rows, the latent files and
their SHA-256 checks are the sweep analysis's own. Expression and labels: `D/prepped_scib/<task>__scib.h5ad`.

- **Penalties** (discriminator, reference, pooled, barycenter critics, MMD, Sinkhorn) x decoder (unconditioned c0, conditioned c1)
  x task: the single sweep strength used for label transfer in `docs/SWEEP_ANALYSIS.md` section 3 item 7 (spec 9; the
  "Supp Note 9" rule): the failure-free strength whose seed-mean batch score is nearest b*, ties to the smaller strength
  (`analysis_sweep.sweep_targets`). The choice uses batch scores only; no readout of this check takes part in it. All three seeds
  (100, 101, 102). Low-signal pairs are included and flagged.
- **Plain scVI**: both decoders, seeds 100-102 (the reference, section 3).
- **Baselines**: every scored row of the committed baseline run list, at each of its settings: scvi_adv, scANVI, sysVI (each setting
  x seeds 0-4), Harmony, Scanorama, PCA (one run per setting). A setting is (method, strength or knob value, latent dimension).
  scANVI is trained with the cell-type labels, so its readouts are not label-free; this is stated wherever it is reported.

Expected count per task: 36 penalty fits + 6 plain-scVI fits + the baseline rows scored for that task in the inputs' coverage table.
The run stops if the count differs.

## 2. Neighbours

Euclidean distance in the saved latent `z`; for each cell its **15 nearest other cells** (the cell itself excluded). The same
neighbour lists serve purity (3a) and smoothing (3c). The latent's `obs_names` must equal the prepped file's `obs_names` in order, and
its `celltype` must equal the prepped file's cell-type column (`uns['celltype_key']`); otherwise the run stops.

## 3. Readouts (per fit, per cell type)

**(a) Neighbourhood purity.** For each cell, the share of its 15 neighbours carrying its own cell-type label; per type, the mean over
the type's cells. Per fit: the unweighted mean over types (every type counts once).

**(b) Rare-type retention.** Rare type: fewer than 1% of the task's cells (strictly). (a) restricted to rare types, plus the per-type
label-transfer recall from (d). Tasks without a rare type report none.

**(c) Marker-gene preservation** (real RNA tasks only: pancreas, lung, immune, immune_hum_mou; the prepped file must say modality
`rna`). Expression: the prepped file's `X` (log-normalised as distributed by scIB) restricted to its 2,000 highly variable genes
(`var['highly_variable']`). Smoothed expression of a cell: the mean of `X` over its 15 neighbours (section 2). Markers: top 20 genes
per type, `scanpy.tl.rank_genes_groups(method='wilcoxon')`, each type against all other cells, ordered by score; computed once on the
unsmoothed data (the same for every fit of a task) and once on each fit's smoothed data. Readout: the Jaccard overlap of the two
top-20 sets, per type. Types with fewer than 10 cells get no marker readout (empty).

**(d) Label-transfer confusion.** The label-transfer classifier of the sweep analysis (`analysis_x1.label_transfer_f1`): cell types
present in only one batch are dropped; each batch is held out in turn and its cells are predicted by a 15-nearest-neighbour classifier
(Euclidean, equal weights) trained on the other batches. Every kept cell is predicted once. Per type: recall (share of the type's cells
predicted as that type; dropped types get none). Per fit: the confusion table (rows = true type, row-normalised); the per-batch
macro F1 computed from these predictions must equal `label_transfer_f1`'s, and the run checks it.

## 4. Differences from plain scVI

For each per-type readout r (purity, recall, marker Jaccard): r minus the plain-scVI value of the same type.
- Penalty fits and plain scVI: plain scVI with the **same decoder and the same seed** (seed-matched; plain scVI's own difference is 0).
- Baselines (seeds 0-4 or one run; no seed matches the sweep seeds): the **mean of plain scVI with the conditioned decoder over seeds
  100-102**. scvi_adv, scANVI and sysVI use the conditioned decoder; Harmony, Scanorama and PCA have no decoder.
A fit-level difference is the mean of its per-type differences over the types where both values exist.

## 5. Reporting

Per task: a per-fit-per-type table, a per-fit table and the seed-mean confusion table. Summary: per task x method x setting x decoder,
the number of fits, mean and SD over seeds (SD empty for one run); and per method x setting x decoder, the mean over tasks of the
per-task means, separately for real and simulated tasks, with the number of tasks. A one-page summary states the per-method mean
purity difference, the rare-type pattern and the marker Jaccard by method, all read from these tables. Descriptive only.

## 6. Checks the run applies (it stops on failure)

Latent SHA-256 equals the score row's (`analysis_x1.read_latent`); cell order and labels match the prepped file; fit counts per
task match section 1; every penalty and plain-scVI fit has its seed-matched reference; purity and Jaccard lie in [0, 1]; no missing
value where a readout is defined (empty only for: recall of dropped types, markers outside the four RNA tasks or for types under
10 cells, rare readouts in tasks with no rare type); the F1 cross-check of 3d.

Workstation run: at most 6 worker processes, one BLAS thread each, peak memory under 47 GB.
