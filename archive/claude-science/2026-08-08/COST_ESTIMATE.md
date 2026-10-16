# Computational cost of the six-dataset light wave — measured, 2026-08-08 (rev. 08-10)

> **Revision 08-10.** Both open decisions are settled and implemented: the wave is
> **sharded and load-balanced** across 6 workers, and the epoch budget is **500 as a
> ceiling** with early stopping deciding. Revised headline: **918 configs, ~90
> worker-hours, 15.3 h wall-clock** (20.2 h worst case), against 26.7 h for
> one-dataset-per-worker. Launch with `scripts/run_wave.sh`.
>
> The config count is corrected from **822 to 918**: E8 sweeps `batch_count` from 2 to
> n_batches (2 heads × 3 seeds per level), so it contributes 6×(n_batches−1) per dataset
> rather than a flat 6 — while still excluding the 2-batch datasets, so `lung` has no E8.
> The "Bottom line" table below carries the corrected figures and the per-dataset
> breakdown; the sections after it are the original 08-08 text and are superseded where
> they disagree.


**Scope:** `pancreas, immune, lung, sim1, sim2, atac_small` × E1 + E2 + E8 + E10.
**Hardware:** local workstation, 8 usable cores (`nproc`), RTX 2080 (7.6 GiB usable).
**Nothing has been launched.** All figures below come from probe runs totalling 42
configurations.

---

## Bottom line

| quantity | value |
|---|---|
| configurations in the wave | **918** (see the per-dataset table below; E8 varies with batch count and is excluded for `lung`) |
| total compute | **~90 worker-hours** |
| wall-clock, one dataset per worker (the HANDOFF layout) | **26.7 h** (set by `pancreas`) |
| wall-clock, rebalanced across 6 workers | **15.3 h**, worst case **20.2 h** (see §3) |
| shards in the launch manifest | **56**, balanced to 98% |

### Configurations per dataset

E8 sweeps `batch_count` from 2 to n_batches (2 heads × 3 seeds per level), so its size
depends on the dataset, and it **excludes the two 2-batch datasets by construction** —
which is why `lung` carries no E8 shards. Every dataset also runs E1 (60), E2 (48) and
E10 (24).

| dataset | batches | E8 configs | total configs | worker-hours |
|---|---|---|---|---|
| `pancreas` | 9 | 48 | 180 | 26.7 |
| `sim2` | 4 | 18 | 150 | 21.9 |
| `lung` | 2 | **0** (excluded) | 132 | 17.4 |
| `immune` | 4 | 18 | 150 | 12.6 |
| `sim1` | 6 | 30 | 162 | 7.0 |
| `atac_small` | 3 | 12 | 144 | 4.1 |
| **total** | | **126** | **918** | **89.7** |

The headline number is close to the 1.2 days in `HANDOFF.md`, but **the agreement is a
coincidence and the old model should not be reused** — see below.

---

## The old estimate was structurally wrong

`OPTIMIZATIONS.md`, `HANDOFF.md` and `worker_scaling.json` (`light_serial_h = 50.0`) all
descend from one cost function. Reconstructing it reproduces its 50.0 serial hours
exactly, which pins down what it contained:

1. **It counted only the scIB metric suite, not model training.** The per-config
   constants in that model (pancreas 57.8 s, immune 147.6 s, `atac_large` 419.5 s) are
   suite-only timings taken from the optimization profiling. Training was never added.
2. **It applied a flat 2.5× "critic is more expensive" factor to the suite cost** — but
   the suite does not depend on the adversarial head at all. The factor was standing in
   for the critic's longer *training*, applied to the wrong term.

Measured, the critic:discriminator **training** ratio is not 2.5× and is not a constant:

| dataset | batches | disc | critic | ratio |
|---|---|---|---|---|
| `atac_small` | 3 | 81 s | 235 s | 2.9× |
| `lung` | 2 | 168 s | 601 s | 3.6× |
| `immune` | 4 | 120 s | 719 s | 6.0× |
| `sim2` | 4 | 115 s | 726 s | 6.3× |
| `sim1` | 6 | 65 s | 438 s | 6.7× |
| `pancreas` | 9 | 90 s | 1041 s | **11.6×** |

The ratio tracks **batch count** (r = 0.95), not cell count (r = −0.20), and the
mechanism is in the architecture: the critic takes `disc_iter = 10` steps per generator
step across `|V|` per-batch potentials, so its cost scales with `n_obs × n_batches` while
the discriminator is a single `|V|`-way classifier at `disc_iter = 1`. The old 2.5×
figure is roughly right at 2–3 batches and wrong by 4× on the 9-batch dataset.

**Why the two errors nearly cancel.** Omitting training understates cost by ~5×;
early stopping then recovers most of it — only 10 of 36 benchmark configs reached the
150-epoch budget (mean 106 epochs, 71% of budget). Two large errors in opposite
directions is not a validated estimate, and they will not cancel on a different grid.

---

## What was measured

**Production-shaped benchmark** — 6 concurrent workers, one dataset each, λ = 0.2, both
heads × 3 seeds = 36 configs. This is the layout `HANDOFF.md` prescribes, so contention,
warm-up amortization and VRAM pressure are all as they would be in the real wave.
All 36 succeeded; no failures, and `paga_spearman` is populated on every row.

The per-config seconds below are the calibration constants and still stand. The
`worker-hours` column, however, was computed at the 150-epoch budget and with the E8 count
that this revision corrects — for current per-dataset hours use the table in "Bottom line".

| dataset | cells | batches | disc s/cfg | critic s/cfg | suite+overhead | worker-hours (superseded) |
|---|---|---|---|---|---|---|
| `pancreas` | 16,382 | 9 | 90 | 1041 | 42 | **23.3** |
| `immune` | 33,506 | 4 | 120 | 719 | 61 | 18.4 |
| `sim2` | 19,318 | 4 | 115 | 726 | 44 | 17.8 |
| `lung` | 32,472 | 2 | 168 | 601 | 58 | 16.2 |
| `sim1` | 12,097 | 6 | 65 | 438 | 30 | 10.8 |
| `atac_small` | 11,270 | 3 | 81 | 235 | 18 | 6.7 |

**Sensitivity probes** (6 further configs) for the two axes the benchmark held fixed:

- **λ** (E1 sweeps 0 → 1; benchmark used 0.2): critic training moves ±40–69% of its mean
  with **no monotone trend** — early stopping ends runs at different epochs. λ = 0.2 is a
  central value, not a floor. This is the dominant uncertainty term.
- **backbone** (E2 sweeps 8; benchmark used NB): LDVAE 161 s and Gaussian 162 s against
  NB's 235 s — non-NB backbones are ~0.68× as expensive. Costing everything at NB is
  therefore **conservative**, and is the source of the low end of the range.

Seed-to-seed CV on identical configs is 12–46% (mean 27%), so per-dataset figures are
±1 significant figure, not better.

---

## Two things to decide before launching

**1. `pancreas` is the critical path and the layout wastes a third of the machine.**
One-dataset-per-worker finishes `atac_small` in 6.7 h and then idles that core for 16 h
while `pancreas` runs. Splitting `pancreas` and `immune` by head or by seed across the
freed workers moves the wall-clock from ~23 h toward the ~16 h packing bound — roughly a
third off, for a scheduling change only.

**2. Three datasets are budget-limited, not converged, at 150 epochs.** On the critic
head, `pancreas`, `lung` and `sim2` ran the full 150 epochs in every seed, with
`es_best_epoch` at 130/123/147 — early stopping never triggered, so those runs stopped
because the budget ran out. `run_experiment.py` defaults to `--epochs 150`. If the
intent is "train to convergence, let early stopping decide", the budget needs raising for
those three, which costs proportionally more time (a 500-epoch budget would be ~1.4× on
the affected configs, not 3.3×, since early stopping would fire before the ceiling on
most). If 150 is a deliberate fixed budget — as the convergence check concluded — then
nothing changes, but the manuscript should not describe those runs as early-stopped.

## VRAM

Peak GPU memory at 6 concurrent light-dataset workers was **7670 MiB of 8192**. This is
far tighter than the 0.66 GB/worker figure in `OPTIMIZATIONS.md` §8 implies, and it means
6 workers is at the ceiling, not comfortably inside it. It held for the full benchmark
with no OOM, but a seventh worker would not fit, and this is worth knowing before any
attempt to raise concurrency.

---

# Revision 08-10 — both decisions implemented

## 1. Epoch budget: 500 as a ceiling

At `--epochs 150` the critic head was **budget-limited, not converged** on three datasets.
Re-run at a 500 ceiling (1 seed, λ=0.2), early stopping now fires on all three:

| dataset | epochs at 150 budget | epochs at 500 ceiling | best epoch | critic train s |
|---|---|---|---|---|
| `pancreas` | 150 (never stopped) | **160** | 110 | 891 |
| `lung` | 150 (never stopped) | **190** | 140 | 512 |
| `sim2` | 150 (never stopped) | **200** | 150 | 651 |

The discriminator was already stopping below 150 and barely moves (`atac_small` 120,
`lung` 110, `sim2` 130). Epoch inflation is therefore **1.22× on the affected critic
arms and ~1.03× elsewhere** — not the 3.3× a naive 150→500 reading would suggest, because
early stopping fires long before the ceiling. This matches the project's settled epoch
policy (500 is a ceiling, early stopping decides) and removes a silent truncation that
would otherwise have been reported as early-stopped.

## 2. Rebalanced schedule

`scripts/run_wave.sh` + `scripts/wave_manifest.tsv` (committed `60a7a3c`).

**56 shards** at (dataset, experiment, head), seed-split where a shard exceeds 5 h,
packed longest-first onto 6 lanes at **98% balance**. `large` counts shards on `immune` or
`lung` — the pair the contention penalty in §3 was measured on:

| worker | shards | hours | large shards |
|---|---|---|---|
| 1 | 9 | 15.28 | 3 |
| 2 | 9 | 15.21 | 3 |
| 3 | 9 | 15.03 | 3 |
| 4 | 10 | 15.20 | 3 |
| 5 | 10 | 15.18 | 2 |
| 6 | 9 | 15.04 | 2 |

**~90 nominal worker-hours; makespan 15.3 h** against 26.7 h for one-dataset-per-worker.
That is the best-mix end of the range: with a large-heavy concurrent mix the makespan
becomes 20.2 h (§3), so the saving against the 26.7 h baseline is **6.5–11.4 h
(24–43%)** rather than a single figure.

Shard boundaries use only splits `run_experiment.py` already supports, so each shard is
one process looping its own grid. This matters: process startup is ~81 s, so 56 shards
cost ~1.3 h of startup while per-config sharding (918 processes) would cost ~21 h.

## 3. Contention is a real term, and it bounds the gain

The estimate's one soft spot. Two concurrency regimes were measured:

- **2 large + 4 smaller datasets** (the original benchmark, one dataset per worker) — the
  calibration baseline.
- **6 workers all on large datasets** (3×`immune` + 3×`lung`, critic) — **1.32× slower**
  per config.

The rebalanced schedule's mix moves between these over its run, so the honest figure is a
range: **15.3 h if the mix stays benchmark-like, 20.2 h if it goes large-heavy.**

The manifest mitigates this by spreading the 16 `immune`/`lung` shards across lanes under a
per-lane quota — **3, 3, 3, 3, 2, 2** — rather than letting the packer cluster them. The
quota is `ceil`/`floor` of 16/6, so the spread is even to within one shard; the packer then
picks the least-loaded eligible lane, which is why balance stays at 98%. Without the quota
the same shards land 2, 4, 4, 3, 3, 2 and three lanes run large-heavy simultaneously,
making the 20.2 h end of the range the default rather than the worst case.

## 4. VRAM: the earlier figure was contaminated

The "7670 MiB of 8192" in the 08-08 revision was measured while other work shared the GPU.
Re-measured cleanly on an idle card, the schedule's **worst** co-residency (6 workers,
`immune` + `lung`, critic head) peaks at **6166 MiB with zero OOM** and all 6 configs
valid. There is more headroom than previously believed, but 6 workers remains the
recommended ceiling — the 1.32× contention penalty, not memory, is what limits throughput.

## Launch

```bash
cd wcd_git
WCD_DATA=$(pwd)/../data scripts/run_wave.sh          # 6 workers, 500-epoch ceiling
```

Idempotent: each finished shard drops `results/wave/<tag>.done`, and `--resume` skips
configs already in a shard's CSV, so an interrupted wave restarts without repeating work.
Verified end-to-end — 56/56 shards dispatch under a stubbed interpreter (`WCD_PYTHON`
override), a re-run skips all 56, no `lung` E8 shard is dispatched, and one real shard
(`atac_small`/E8/disc) produced 3 valid rows with early stopping at 100–120 epochs and
`paga_spearman` populated.
