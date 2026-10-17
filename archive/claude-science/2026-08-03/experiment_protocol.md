# Experiment protocol — current standing decisions

Status: **methods settled, verified, and implemented; manuscript updated (`a98c1e7`)**.
Repo: `main` @ a98c1e7. Scope: **all 8 datasets**. Supersedes the protocol used for every result before 2026-08-03.

## 1. What changed and why

| area | before | now | reason |
|---|---|---|---|
| epochs | 150 fixed | **500 ceiling + early stopping** | conservation degrades with training; a fixed count was undisclosed regularisation |
| metric suite | 10 metrics, ~1,570 s | **full scIB 14 + probes + PAGA, 876 s** | duplicate Leiden sweeps eliminated (1.8×) |
| λ selection | none (fixed 0.2) | **nested, inside each CV training fold** | reviewers flagged λ as confounding; post-hoc selection inflates the effect 85% |
| selection criterion | `iLISI − 10·cLISI` | **scIB 0.4·batch + 0.6·bio** | old form was effectively pure-cLISI (17.7× scale imbalance) |
| reference batch | `reference_batch=0` (alphabetical) | **max cell-type Shannon entropy, tie-broken by size** | index 0 was one of the *smallest* batches on 4/6 datasets |
| significance unit | dataset × backbone (n=30) | **dataset (n=6), NB-corrected within dataset** | backbones on shared datasets are pseudo-replication |
| embeddings | discarded | **persisted to scratch** | metrics-only CSVs cannot be re-analysed |
| array concurrency | %20, 64 GB/task | **%70, 32 GB/task** | no cluster limit existed; measured peak RSS is 23 GB |

## 2. Epoch policy

`epochs=500` is a **ceiling**, matching the original nested-CV protocol. Early stopping
decides the realised count per config: a held-out linear cell-type probe is scored every
10 epochs (`es_check_every`), training halts after 5 checks without improvement
(`es_patience`), and the **best-scoring weights are restored**.

Evidence for needing this — independent full runs at 150/300/450 epochs, 3 datasets × both
heads:

| epochs | iLISI | batch-probe lift | ARI | label-probe lift |
|---|---|---|---|---|
| 150 | 0.605 | 0.013 | **0.058** | **0.168** |
| 300 | 0.606 | −0.001 | 0.015 | 0.132 |
| 450 | 0.619 | 0.003 | 0.018 | 0.097 |

Mixing is flat; conservation falls ~70% (ARI) and ~42% (probe lift). The
conservation-optimal epoch count varies by dataset *and* head — 150 wins only 10/18
arm × metric cells — so no fixed number is defensible. Worst case sim2/critic: ARI
0.190 → 0.0001.

Validated on pancreas: early stopping fired at **epoch 180, best weights from epoch 130**,
probe rising 0.686 → 0.871 then drifting. Training cost 283 s rather than a full 500.

**Disclosure obligations.** Early stopping monitors *conservation only* (mixing improves
monotonically in epochs, so it needs no guard) — a deliberate asymmetry that must be
stated. The realised `es_best_epoch` is recorded per config and should be reported.

## 3. Metric suite — measured cost, pancreas (16,382 cells, 16 cores)

| component | time |
|---|---|
| training (500 ceiling, ES stopped 180) | 283 s |
| iLISI / cLISI / graph_conn / PCR / asw_batch | 0.3–1.2 s each |
| asw_celltype, isolated_asw | 3.5 s, 4.2 s |
| probe metrics | 5.2 s |
| PAGA Spearman | 7.1 s |
| cell-cycle conservation | 7.0 s |
| **full optimised suite (leidenalg, scib parity)** | **876 s** |
| *(same suite with igraph flavour — see caveat)* | *79 s* |
| *(before optimisation)* | *~1,570 s* |

**The optimisation.** scib's `cluster_optimal_resolution` sweeps 10 resolutions per
caller, and `isolated_labels_f1` calls it **once per isolated label** (~40 Leiden runs on
pancreas). Those two sweeps cost 767 s + 773 s = 9.4× the training time and re-cluster the
same graph at the same resolutions. They optimise *different* objectives (NMI vs per-label
max-F1), so a single "optimal" clustering cannot be shared — but the underlying
clusterings can. `_resolution_cache()` computes each resolution once; each metric selects
its own best from the cache.

**Speedup attribution — measured, and smaller than first reported.** De-duplicating the
sweeps is worth **1.8×** (1,570 s → 876 s) and preserves scib parity exactly. An initial
78.7 s measurement used `flavor="igraph"`, which is a further 11× but **changes the
numbers**: pancreas ARI 0.003567 (scib) vs 0.024778 (igraph), Δ = 2.1e-02. Flavour is
therefore pinned to scanpy's default (leidenalg).

A synthetic A/B had wrongly cleared igraph as identical (|Δ| = 0 across structured, weak
and degenerate latents) because 3,000 cells × 6 well-separated clusters converge to the
same trivial partition under both implementations in ~1 s. The divergence and the speed gap
appear only on a real 16k-cell weakly-structured latent, where the partition is genuinely
ambiguous. **Never validate a performance optimisation on data easier than production.**

If the 11× is later judged necessary, igraph is defensible *only* if reported as "Leiden
(igraph implementation)" and used for every arm — within-study comparisons stay valid,
but the numbers are no longer comparable to scIB-reported values.

## 4. Selection criterion

`score = 0.4 · batch_correction + 0.6 · bio_conservation`, each group min-max scaled
within dataset — the scIB paper's published weighting. scIB ships **no** scalar aggregate
(verified in scib 1.1.7), so computing it is reproducing their method, not inventing one.

Excluded, with evidence: kBET (all-NaN — needs rpy2 + R kBET, not installed); asw_batch
(within-dataset sd 0.0031), PCR (0.0016), asw_celltype (0.0015), isolated_asw (0.0041) are
degenerate below sd 0.01; NMI redundant with ARI (ρ = 0.95).

Robustness: the selected λ set is unchanged across `w_batch` ∈ {0.3, 0.4, 0.5, 0.6}, so the
weight is not load-bearing. Adding graph connectivity (scIB classes it as batch
correction) leaves all 12 selections unchanged.

**Responsiveness to λ** (signal across λ ÷ seed noise) — only SNR > 1 carries usable
selection signal:

| metric | SNR | | metric | SNR |
|---|---|---|---|---|
| batch-probe lift | 4.64 | | ARI | 1.22 |
| PCR | 4.31 | | label probes | 1.16 |
| kNN batch probe | 2.23 | | cLISI | 1.03 |
| graph connectivity | 1.80 | | **asw_batch** | **0.88** |
| iLISI | 1.57 | | **asw_celltype** | **0.62** |
| NMI | 1.45 | | **isolated_asw** | **0.45** |

## 5. λ selection — nested, in-fold

λ is selected **inside each CV training fold** and evaluated on the held-out fold. Post-hoc
selection on the same data inflates the effect by **85%** (mean Δ batch lift −0.0595
cherry-picked vs −0.0321 at pre-committed λ=0.2), and the per-dataset "best" λ is unstable
across the grid — noise, not a discovered optimum.

`LAMBDA_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]`. Valid results
currently cover only the 4-point subset {0, 0.05, 0.2, 0.5}, where the criterion is still
rising at the subset endpoint in 11/12 arms — so grid coverage must be filled in before any
claim about an interior optimum. `--d-coef-only` silently selects **zero** configs for a
value absent from the grid.

## 6. Reference batch

**Max Shannon entropy of the batch's cell-type distribution, tie-broken by cell count.**
Pre-registered.

The previous `reference_batch=0` was the *alphabetically* first batch
(`sort_values().unique()`), which on 4 of 6 datasets was one of the smallest — pancreas
`celseq`, 1,004 cells, 8th of 9. Entropy selection moves pancreas to `inDrop1` (1,937
cells, 14/14 cell types, H=1.851) and sim1 to `Batch5` (7/7 types); immune and lung
unchanged. `total_var` and `mean_pairdist` are *not* valid diversity proxies — they track
batch size and technology dispersion.

Sensitivity to this choice is tested by E4's `rotating` and `joint` arms.

## 7. Statistics

The comparison unit is the **dataset**. Treating dataset × backbone pairs as independent
moved p from 0.0008 to 0.0705 on the E2 batch-lift effect with an unchanged effect size
(−0.041) — pseudo-replication, ICC = 0.40.

Nadeau-Bengio corrects for overlapping *training sets across CV folds of one dataset*
(`t = mean(d) / sqrt((1/n + n_test/n_train)·var(d))`, ratio = 1/(k−1); the implementation in
`notebooks/results.ipynb` cell 14 matches the published formula). It does **not** apply
across datasets, which share no data. CV therefore buys per-dataset significance in the
paper's Table 1 format; the across-dataset claim stays n=6.

Do not report a cluster bootstrap at this n — a null simulation gave a **13.1%**
false-positive rate at α=0.05.

## 8. Compute

DSAI, partition `cpu`: 112 cores / 503 GB per node, `MaxNodes=10`, `MaxTime=3-00:00:00`.
No MaxJobs/MaxSubmit on the association. Standard submission:
`--cpus-per-task=16 --mem=32G --array=0-N%70` (measured peak RSS across 456 tasks: median
8–10 GB, p90 19 GB, max 23 GB).

Embeddings persist to `/scratch/gsteino1/kreid20/wcd_embeddings` via `--embed-out`, one
`.npz` per config (`z`, `batch`, `celltype`, `X_pca`; labels cast to `U64` or `np.load`
fails without `allow_pickle`).

## 9. Open items

1. ~~Equivalence verification~~ **DONE.** On the real pancreas latent the shared-cache
   path is numerically identical to scib's own sweeps: ARI 0.010173 vs 0.010173
   (|Δ| 5.2e-18, float rounding), NMI 0.071084 vs 0.071084 (|Δ| 0), isolated_f1 0.018519
   vs 0.018519 (|Δ| 0). Our single pass costs 876 s against scib's 811 s + 815 s = 1,626 s
   for the same three metrics — 1.9× with bit-level parity.
2. ~~Reference-batch rule~~ **DONE** (`0ebd02c`). `select_reference_batch()` in `data.py`,
   default `reference_rule="entropy"` in `load_task`, threaded by *name* through E1/E5/E9
   via `reference_batch_name_str`; E4 stays index-based because sweeping references is its
   purpose. Controls pass: prefers an even 300-cell batch over a skewed 1,000-cell one,
   ties break by size.
3. ~~Nested-CV plumbing~~ **DONE** (`0ebd02c`). `backbone` (default `NB_uncond`),
   `registry`, and early stopping threaded into both inner and outer fits; inner selection
   now maximises the scIB overall score.
4. **Cost re-scope** once equivalence lands — the 2,973 CPU-h nested estimate predates the
   20× suite speedup and assumed the suite was 60% of a run (it was ~90%).
5. **The underlying model still lacks cell-type structure** (ARI ≈ 0.02, NMI ≈ 0.04 on
   pancreas). The methods above are correct; this is the open scientific problem and no
   conclusions should rest on these embeddings.


## 10. Cost of the remaining programme (measured, supersedes all earlier estimates)

Per-config basis, measured on pancreas: **300 s training** (500-epoch ceiling, early
stopping fired at 180/200) **+ 876 s optimised suite = 1,176 s** — 4.3× cheaper than the
5,060 s measured under the old protocol. Training scales ~linearly in cells; the suite
scales as n^1.69 (Leiden and kNN dominate).

| per config | discriminator | critic |
|---|---|---|
| atac_small | 0.19 h | 0.27 h |
| sim1 | 0.21 h | 0.30 h |
| pancreas | 0.33 h | 0.45 h |
| sim2 | 0.42 h | 0.57 h |
| lung | 0.80 h | 1.02 h |
| immune | 0.99 h | 1.24 h |

| component | CPU-h | days @70 |
|---|---|---|
| **CV+NB, nested in-fold λ over the full 10-point grid** (5 outer × 3 inner, 1,860 fits) | 1,050 | 0.63 |
| E1 Pareto, full grid × 3 seeds | 203 | 0.12 |
| E2 backbones (8 × 2 heads × 3 seeds) | 163 | 0.10 |
| E8 multibatch | 64 | 0.04 |
| E9 formulations | 43 | 0.03 |
| E5 biology | 27 | 0.02 |
| E3 baselines | 12 | 0.01 |
| **total, 6 light datasets** | **1,562** | **0.93** |

### All 8 datasets (adopted scope)

Heavy datasets are far costlier per config — cross-species 5.49 h / 6.23 h (disc/critic),
large-ATAC 4.35 h / 5.00 h — because the suite scales as n^1.69:

| component | CPU-h | days @70 |
|---|---|---|
| **CV+NB nested λ, 8 datasets, 10-point grid** | 4,316 | 2.57 |
| E1 Pareto full grid × 3 seeds | 835 | 0.50 |
| E2 backbones | 668 | 0.40 |
| E4 reference designs | 259 | 0.15 |
| E9 formulations | 174 | 0.10 |
| E5 biology | 111 | 0.07 |
| E3 baselines | 56 | 0.03 |
| E8 multibatch | 54 | 0.03 |
| **TOTAL, all 8 datasets** | **6,474** | **3.85 days** |

The two heavy datasets carry ~78% of the total; the 6-dataset scope is 1,562 CPU-h
(0.93 d), so full coverage is 4.1× that.

Earlier figures (2,973 / 5,824 / 4,806 CPU-h) are obsolete — they predate early stopping,
the suite optimisation, and the discovery that the %20 array throttle was self-imposed.
