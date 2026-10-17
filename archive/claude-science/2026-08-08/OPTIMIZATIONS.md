# Performance work on the WCD evaluation pipeline

**Repository:** `wasserstein-critic-deconfounding`, `main` @ `eea7a19`
**Hardware:** local workstation, 12 CPU cores, NVIDIA RTX 2080 (8 GB VRAM)
**Status:** 39 tests pass, 1 skipped; manuscript and supplement compile clean

This document records the optimizations applied to the metric suite and training
pipeline, the measurements behind each decision, the changes that were **rejected** and
why, and the bugs found along the way. Every figure quoted here was measured on this
machine; none are estimates unless labelled as such.

**Net effect on the local programme** (including experiment E10, and the measured
concurrency limits in §8):

| scope | before | after |
|---|---|---|
| 6 light datasets | 2.1 days | **1.3 days** |
| all 8 datasets | 7.2 days | **4.1 days** |

Per-configuration cost on the two datasets that dominate the schedule fell by roughly
50%: `atac_large` from 836 s to 420 s, `immune_hum_mou` (extrapolated) from 1024 s to
473 s.

---

## 1. Why the metric suite, not the model

Profiling established early that training was a small fraction of each configuration and
the scIB metric suite was the bulk. The suite is CPU-bound, so during evaluation the GPU
sat near 20% utilization. That gap — an idle accelerator next to an expensive
pairwise-distance workload — is what the first two optimizations exploit.

### Two profiling traps, both of which produced wrong answers first

**Just-in-time compilation charged to the wrong metric.** Timing each metric once in a
fresh process attributes numba's and pynndescent's compilation cost to whichever metric
happens to run first. This made a redundant neighbour-graph rebuild look like a ~22 s
saving when the true figure was 2.26 s. *Convention adopted: every component profile
discards a warm-up call.*

**Profiling the convenient dataset instead of the expensive one.** Metric shares differ
substantially with dataset size, and the small dataset understates the wins that matter:

| | pancreas (16k) | atac_large (85k) |
|---|---|---|
| silhouette share of config | 17% | **40%** |
| LISI k-NN search | 0.70 s | **92.5 s** |

On pancreas the k-NN optimization looks worth ~3% and was explicitly dismissed on that
basis. On `atac_large` it is a third of the suite. *Convention adopted: profile the
dataset that dominates the programme, and re-profile after every change, because the
bottleneck moves.*

---

## 2. Optimizations applied

### 2.1 Leiden resolution sweep → igraph (`3b9bea3`)

The clustering sweep that feeds ARI, NMI and the isolated-label F1 was switched to the
igraph backend with `n_iterations=2`, worth 1.55× on the suite. This is the one change
with a residual numerical difference rather than machine-precision agreement; it is
disclosed in Supplementary Note 4 and the strict-parity option is retained for readers
who prefer it.

### 2.2 GPU silhouette backend (`c29e703`)

Three metrics — `asw_celltype`, `asw_batch`, `isolated_asw` — reduce to the same kernel,
together 35% of the warm pancreas suite and **40% of an entire `atac_large`
configuration** (353 s of ~875 s).

*Implementation.* Chunked `torch.cdist` reduced directly into per-cluster distance sums
via `index_add_`, so the full n×n matrix is never materialised. Peak VRAM 0.69 GB
(pancreas) and 3.25 GB (`atac_large`) — inside the 8 GB card.

*Precision.* Double precision, deliberately. Single precision is ~94× but disagrees with
`scikit-learn` by up to 7.6e-05 per cell; double precision retains most of the speedup
and agrees to machine precision.

*Approach.* Only the sklearn **backend** is swapped, inside a context manager. scib's own
aggregation runs unchanged — the `abs()`, the `1-x` rescaling, the per-group means, the
singleton and single-batch skips — so results stay comparable to the published benchmark.
Reimplementing those wrappers would have risked silent drift in exactly those conventions.

Measured end-to-end through scib's wrappers, on one fixed embedding:

| dataset | CPU | GPU | speedup | max abs diff |
|---|---|---|---|---|
| pancreas (16,382) | 15.14 s | 3.00 s | 5.06× | 2.3e-08 |
| atac_large (84,813) | 362.05 s | 35.04 s | **10.33×** | 3.6e-10 |

### 2.3 Exact GPU k-NN with a cross-metric cache (`081bdf6`)

Re-profiling after the silhouette change showed the bottleneck had moved: LISI became 65%
of the suite. Within it, the split is stark — the neighbour search is **92.53 s against
1.34 s** for the numba Simpson kernel, i.e. 99%/1%. The hand-written numba kernel was
never the problem.

Two changes. First, the search runs on the GPU (chunked `cdist` + `topk`). It remains
**exact**: neighbour indices are 100.000% identical to sklearn, distances agree to
4.8e-07, and the resulting LISI values to 1.9e-14. `scikit-learn`'s `algorithm='auto'`
already selects brute force at 256 dimensions, so only where the arithmetic runs has
changed. Second, iLISI and cLISI differ *only* in the label being scored — same
embedding, same k, same neighbours — so an identical 92 s search was running twice; one
memoised search now serves both.

| | CPU | GPU + cache |
|---|---|---|
| both LISI calls, atac_large | 99.05 s | **19.25 s** (5.15×) |

Saves 79.8 s per configuration; iLISI |Δ| 3.1e-09, cLISI |Δ| 5.2e-09.

### 2.4 PAGA baseline cache (`28d78e6`)

The PAGA trajectory metric compares per-batch reference graphs, built on `X_pca` — the
**unintegrated** representation — against a global graph built on the integrated
embedding. The reference graphs depend only on the dataset, not on the embedding, head,
λ, seed or backbone, yet every configuration in a sweep rebuilt them identically.

| dataset | before | after | saved/config |
|---|---|---|---|
| pancreas | 19.59 s | 3.49 s | 16.10 s |
| atac_large | 20.18 s | 10.87 s | 9.32 s |

Values bit-identical (|Δ| = 0).

---

## 3. Changes evaluated and rejected

Each was rejected on measurement, not preference.

**Larger minibatch (bs 4096).** A real 1.31× on training, but it breaks the critic — see
§5, where it became an experiment rather than an optimization.

**Further numba work.** The remaining hot metrics are already compiled library calls with
no Python loop to lift. The project's own numba Simpson kernel is already in the path and
is the fastest thing in the suite (1.34 s against a 92.53 s search feeding it).

**Training-loop optimization.** Training is now ~48% of a heavy configuration, but GPU
utilization during it is already **71% mean / 87% median** — the loop is not Python-bound,
so there is no dataloader or overhead win available. `torch.distributions` argument
validation *looked* like 4.18 s of a 21.5 s profile; measured directly it is 0.26 s per
10 epochs. That was profiler instrumentation inflating a 1,660-call path — the same trap
as §1.

**Replacing PAGA's `sc.pp.neighbors`.** Worth 5.67 s, but scanpy uses pynndescent, an
*approximate* algorithm. Substituting an exact GPU search would produce a different graph
and therefore a different metric value. That is a correctness change wearing an
optimization costume, and PAGA is a headline conservation metric with reported p-values.

**Subsampling the inner folds.** Changed the selected hyperparameter on the critic arm,
with errors biased toward the region where the critic performs worst.

**A fast-metric subset as the selection criterion.** Disagreed with the full criterion on
a quarter of arms; its economics also inverted once the expensive metrics were sped up.

**`cell_cycle`** (7.53 s, the largest single remaining item) is scanpy gene scoring bound
by pandas indexing — no GPU angle.

---

## 4. Bugs found while verifying

Every optimization was checked for equivalence on a fixed embedding before being kept.
That process found six defects, four of which would have been invisible in normal use.

1. **The silhouette patch initially did nothing.** `from scib.metrics import silhouette`
   binds the *function*, not the module, so two of five patch targets did not exist.
   `setattr` succeeded, the suite ran at CPU speed, and every number was correct — the only
   symptom was the speedup measuring 1.0×. A `hasattr` guard was actively hiding it. The
   context manager now resolves modules via `importlib` and **raises** on a missing target,
   so a future scib reorganisation fails loudly rather than silently reverting to CPU.
2. **`metric=` keyword rejected.** scib passes it explicitly. A non-euclidean metric now
   falls through to sklearn instead of silently returning euclidean numbers under another
   metric's name.
3. **Capital `X` keyword.** scib calls `silhouette_score(X=..., labels=...)` using
   sklearn's own parameter name; both call conventions are now accepted.
4. **`id()`-keyed cache.** CPython reuses object ids after garbage collection, so an
   id-keyed cache can hand one embedding's neighbours to a different array at the same
   address. Replaced with a content hash before it ever ran.
5. **Hashing object arrays.** `obs[k].astype(str).to_numpy().tobytes()` hashes Python
   string *pointers*, not text. The PAGA cache key therefore changed on every call (never
   hit — caught because the dataset "saved" −1.54 s), and pointer reuse could equally have
   produced a false *hit* on different labels. Labels are now hashed via category codes
   plus category names.
6. **Parameter dropped mid-chain.** The call chain is `evaluate_config → train_one →
   train_integration_model → train_model → optim.Adam`. A parameter dropped at any layer
   fails invisibly: the run completes, the result row records the value you *asked for*,
   and the model trains at the default. `lr_g`/`lr_d` were dropped at two layers.

Each is now covered by a regression test, including a signature-chain test asserting that
every layer accepts the training parameters *and* that Adam reads them rather than a
literal.

---

## 5. When an optimization became a finding

The larger minibatch was the last remaining lever. Testing it (pancreas, 150 epochs, 3
seeds, both heads) showed the two adversarial arms are **not symmetric**:

| arm | baseline | bs 4096, lr 1× | lr 2× | lr 4× |
|---|---|---|---|---|
| discriminator iLISI | 0.492 | 0.486 | 0.481 | 0.498 |
| critic iLISI | 0.252 | **0.027** | 0.143 | 0.368 |

The discriminator is insensitive at every learning rate tested (all metrics within 1.4
s.d.). The critic collapses — iLISI 0.252 → 0.027 is essentially unmixed batches, 2.4 s.d.,
with residual batch signal rising 0.274. Raising the learning rate recovers it
monotonically but not completely: at 4× it overshoots in the opposite direction on
residual batch signal (−0.145).

The mechanism is consistent with the paper's thesis. Quadrupling the batch quarters the
gradient updates (2,400 → 600 on pancreas); a Wasserstein critic approximates a supremum
over Lipschitz functions and needs many updates to track a generator moving beneath it,
while the discriminator's cross-entropy objective tolerates a coarser schedule.

**Decision: keep bs=1024, lr=1e-3.** The saving was ~18 hours across the full programme.
A setting that degrades one head while leaving the other untouched would bias the paper's
central comparison in favour of the discriminator — which is precisely the claim under
test.

This was promoted to **experiment E10** (`5fae680`, `eea7a19`): both heads × 3 seeds ×
{bs1024/lr1×, bs4096 at lr 1×/2×/4×}, 24 configurations per dataset, with the production
cell included for both heads so every comparison has a matched within-experiment
reference. Documented in Supplementary Note 5. Cost: +9% of the programme.

---

## 6. Runtime model

Costs are projected from three measured warm anchors (discriminator, 150 epochs, full
suite, GPU, in-process):

| dataset | n | warm per-config |
|---|---|---|
| pancreas | 16,382 | 86.0 s |
| immune | 33,506 | 201.1 s |
| atac_large | 84,813 | 875.2 s |

Fit: `t = 8.444e-05 · n^1.42`. Critic configurations cost ~2.5× discriminator ones.

An earlier two-point fit gave `n^1.30` and **under-predicted `atac_large` by 1.45×**
(605.7 s predicted against 875.2 s measured). Validating the scaling law at the expensive
end, rather than extrapolating to it, is the reason the current estimates are trustworthy.
`immune_hum_mou` (97,861 cells) remains extrapolated, though now only 1.15× beyond the
largest measured point.

Measured parallel efficiency is 1.62× at 3 concurrent workers. Every harness loops
configurations serially by design — that is what makes the per-config split filters
(`--seed-only`, `--d-coef-only`, `--ref-design-only`) the natural unit of parallelism.

---

## 7. Reproducing the verification

```bash
# full test suite, including all equivalence and cache-correctness tests
python -m pytest tests/ -q

# the specific guards added by this work
python -m pytest tests/wcd_vae/test_metrics.py -q -k "silhouette or knn or paga or optimiser or e10"

# manuscript + supplement compile check
./scripts/check_manuscript.sh
```

Supplementary Note 4 ("Metric Implementation and Equivalence") documents the silhouette,
neighbourhood-search and PAGA-baseline changes with their equivalence figures.
Supplementary Note 5 documents the optimiser-sensitivity result.

---

## 8. VRAM limits the wave, not CPU

The GPU work in §2 made the pipeline VRAM-hungry, and this constrains how many
configurations can run concurrently. Measured on the RTX 2080 (7.6 GiB usable):

| dataset class | peak VRAM per worker | safe concurrency |
|---|---|---|
| light (≤ 35k cells) | 0.66 GB | **3 workers** |
| heavy (≥ 85k cells) | 2.98 GB | **2 workers** — 3 OOM |

Three concurrent workers on `atac_large` all die with CUDA out-of-memory. With the
original fixed 2048-row chunk, even two failed: a float64 distance block of that size is
~1.4 GB at n = 84,813.

`_vram_safe_chunk()` now sizes the row-block from free VRAM at call time (a third of it,
floor 256 rows), so memory pressure degrades to a slower run rather than a crash. Two
workers on the heavy datasets now succeed. Three still fail, but in model and training
allocation rather than in the metric chunks, so chunking cannot rescue that case — the
heavy datasets are capped at two concurrent workers, and the schedule above reflects it.

The concurrency ceiling costs **+0.6 days** on the all-8 programme (3.5 days if the heavy
datasets could run three-wide, against 4.1 actual). The headline figures in this document
moved from 1.6/4.5 days to 1.3/4.1 days for a separate reason: the earlier estimate
carried a flat +35% term for nested cross-validation, which the current one replaces with
an explicit per-experiment accounting. Both changes are folded into the table at the top.

Chunk size is a memory knob and must never be a numerical one: otherwise a result would
depend on what else happened to be running on the GPU. Both kernels are verified exactly
block-invariant (silhouette max |Δ| 1.9e-15 between chunk 2048 and 256; k-NN neighbour
indices identical, zero distance difference), and a regression test asserts it.
