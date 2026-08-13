#!/usr/bin/env python
"""Build the wave manifest for run_wave.sh.

# WHY A SCRIPT, NOT AD-HOC CODE: the manifest determines what actually runs. Every
#   defect this programme has hit came from a manifest that was generated once by hand
#   and then diverged from the harnesses -- E8 shards emitted without --batch-count (so
#   they silently ran the full dataset and produced no scaling sweep), lung given E8
#   shards although E8 excludes 2-batch datasets, and a whole wave launched without
#   --embed-out. Generating it from one checked-in script with assertions makes those
#   failures loud instead of silent.
#
# USAGE
#   python scripts/build_manifest.py --scope light  -o scripts/wave_manifest.tsv
#   python scripts/build_manifest.py --scope all    -o scripts/wave_manifest_all.tsv
#
# COLUMNS: phase, worker, tag, est_hours, cmd
#   phase=parallel -> run across the lanes; phase=serial -> run alone, after them.
"""

import argparse
import json
import os

# ---------------------------------------------------------------------------
# Cost constants, all MEASURED on the local RTX 2080 during the completed light wave.
# Per-config training seconds at 6-wide contention, lambda=0.2, 500-epoch ceiling.
# ---------------------------------------------------------------------------
TRAIN_DISC = {"pancreas": 89.5, "immune": 120.4, "sim2": 114.6, "lung": 168.4,
              "sim1": 65.2, "atac_small": 81.1}
TRAIN_CRIT = {"pancreas": 1041.4, "immune": 719.0, "sim2": 725.9, "lung": 601.4,
              "sim1": 437.7, "atac_small": 235.4}
SUITE = {"atac_small": 35.7, "sim1": 39.2, "pancreas": 57.8, "sim2": 71.1,
         "lung": 141.6, "immune": 147.6, "atac_large": 419.5, "immune_hum_mou": 473.2}
# kBET adds a per-config cost that scales with n; measured directly.
KBET = {"atac_small": 47.0, "sim1": 53.7, "pancreas": 74.9, "sim2": 85.5,
        "lung": 137.8, "immune": 140.2, "atac_large": 355.0, "immune_hum_mou": 410.0}
STARTUP = 81.0          # process startup, paid once per shard
EPOCHS = 500            # CEILING; early stopping decides. Never 150 -- see build()

# VRAM-AWARE PACKING. The 8 GiB card cannot hold six concurrent LARGE-dataset shards.
#   MEASURED, isolated, one config at a time: peak torch allocation is 795 MiB for immune
#   (33.5k cells), 731 for lung (32.5k), 532 for pancreas (16.4k) -- it tracks CELL COUNT,
#   not head (critic and discriminator are within 1 MiB of each other). But
#   max_memory_allocated UNDERSTATES what fills the card: nvidia-smi charges the same
#   immune config 1018 MiB once the CUDA context and the caching allocator's reserved
#   blocks are counted.
#   Even that understates the multi-process reality. At the observed failure five immune
#   lanes plus one lung lane held 7058 MiB of 8192 and allocations of just 16-256 MiB were
#   failing -- roughly 1 GB above what a per-process model predicts, because allocator
#   fragmentation across six long-lived processes is not a static quantity. The metric
#   suite is CPU-only, so it is not the source.
#   Rather than trust a model that mispredicts the one failure on record, cap concurrency
#   on the measured fact: six large lanes OOM, so allow at most LARGE_LANE_CAP of them.
#   4 lanes at the observed per-lane share is ~4.7 GiB, leaving ~1.1 GiB of headroom.
BIG_DATASET_CELLS = 25000   # immune (33.5k) and lung (32.5k) qualify; pancreas (16.4k) does not
# LARGE_LANE_CAP IS DELIBERATELY 6 (= no constraint). The first fix for the OOMs capped
# this at 4 on the theory that co-resident LARGE DATASETS exhaust the card. That was the
# wrong variable and the next wave OOM'd again, including sim1 at 12k cells, which no
# size heuristic protects. The real driver is _vram_safe_chunk in evaluation.py: the
# metric suite's GPU kNN sizes its distance block from FREE VRAM at call time and takes
# 33%, which is safe for one process and oversubscribes for six -- and the suite runs for
# EVERY dataset. Fixed there by dividing the budget by WCD_WORKERS. The knob is kept
# because it is the right lever if a single dataset ever genuinely does not fit alongside
# others, but it must not be mistaken for the OOM fix.
LARGE_LANE_CAP = 6          # max simultaneous large-dataset shards across all lanes
LIGHT = ["pancreas", "immune", "lung", "sim1", "sim2", "atac_small"]
HEAVY = ["atac_large", "immune_hum_mou"]


def _heavy_scale(reg, ds):
    """Extrapolate a heavy dataset's training cost from the light-6 fit.

    Critic cost scales with n_obs * n_batches (10 critic steps over V per-batch heads);
    the exponent 0.79 was fitted on the light six and reproduced them to within 3%.
    """
    ref = "immune"
    w = (reg[ds]["n_obs"] * reg[ds]["n_batches"]) / (reg[ref]["n_obs"] * reg[ref]["n_batches"])
    return w ** 0.79


def costs(reg, ds):
    """(disc_seconds, critic_seconds, suite_seconds) per config for one dataset."""
    if ds in TRAIN_DISC:
        return TRAIN_DISC[ds], TRAIN_CRIT[ds], SUITE[ds] + KBET.get(ds, 0.0)
    f = _heavy_scale(reg, ds)
    return TRAIN_DISC["immune"] * f, TRAIN_CRIT["immune"] * f, SUITE[ds] + KBET.get(ds, 0.0)


def build(reg, datasets, embed_root):
    """Return a list of shard dicts. Each shard is ONE process looping its own grid."""
    shards = []

    def add(tag, cmd, seconds, serial=False, big=False):
        shards.append({"tag": tag, "cmd": cmd, "seconds": seconds + STARTUP,
                       "phase": "serial" if serial else "parallel", "big": big})

    for ds in datasets:
        d_s, c_s, o_s = costs(reg, ds)
        nb = reg[ds]["n_batches"]
        # A shard is "big" when its dataset alone pushes per-process VRAM high enough that
        # six concurrent copies exhaust the card -- see BIG_DATASET_CELLS. E8 shards are
        # exempt: they train a batch_count SUBSET, so their footprint is proportionally
        # smaller than the full dataset's.
        big_ds = reg[ds]["n_obs"] >= BIG_DATASET_CELLS
        # WHY --epochs 500 ON EVERY SHARD: 500 is a CEILING, not a target -- early
        #   stopping decides when to stop. Every harness DEFAULTS to 150, and at 150 the
        #   critic head never early-stopped on pancreas, lung or sim2: it was truncated
        #   mid-improvement. Measured on the run this fixed: 41% of rows hit the 150
        #   ceiling with a median es_best_epoch of 140, i.e. still improving. The default
        #   is silent, so the flag must be explicit here.
        common = (f'--registry configs/dataset_registry.json --dataset {ds} '
                  f'--data-root "$R" --embed-out "$EMB" --epochs {EPOCHS}')

        # ---- E1: lambda sweep, 10 lambdas x 3 seeds x 2 heads ----
        for head, tsec in (("disc", d_s), ("critic", c_s)):
            h = "discriminator" if head == "disc" else "critic"
            for seed in (0, 1, 2):
                add(f"{ds}_E1_{head}_s{seed}",
                    f'$PY scripts/run_experiment.py --experiment E1 {common} '
                    f'--head {h} --seed-only {seed} --resume '
                    f'--out results/wave/{ds}_E1_{head}_s{seed}.csv',
                    10 * (tsec + o_s), big=big_ds)

        # ---- E2: 8 backbones x 3 seeds x 2 heads, at the operating point ----
        for head, tsec in (("disc", d_s), ("critic", c_s)):
            h = "discriminator" if head == "disc" else "critic"
            for seed in (0, 1, 2):
                add(f"{ds}_E2_{head}_s{seed}",
                    f'$PY scripts/run_experiment.py --experiment E2 {common} '
                    f'--head {h} --seed-only {seed} --resume '
                    f'--out results/wave/{ds}_E2_{head}_s{seed}.csv',
                    8 * (tsec + o_s), big=big_ds)

        # ---- E8: multi-batch scaling. EXCLUDED for 2-batch datasets by construction:
        #      the experiment sweeps batch_count from 2..n_batches, which is a single
        #      degenerate point when n_batches == 2. Emitting shards for those datasets
        #      silently produced full-dataset duplicates of E10 in an earlier wave.
        if nb > 2:
            for k in range(2, nb + 1):
                # sub-level cost scales with the cells actually retained (top-k batches)
                frac = k / nb
                for head, tsec in (("disc", d_s), ("critic", c_s)):
                    h = "discriminator" if head == "disc" else "critic"
                    add(f"{ds}_E8_{head}_bc{k}",
                        f'$PY scripts/run_experiment.py --experiment E8 {common} '
                        f'--head {h} --batch-count {k} --resume '
                        f'--out results/wave/{ds}_E8_{head}_bc{k}.csv',
                        3 * (tsec * frac ** 0.79 + o_s * frac))

        # ---- E10: optimiser sensitivity, SPLIT BY BATCH SIZE.
        #      The bs=4096 arm exhausts the 8 GiB card whenever ~6 workers share it --
        #      observed three times in one wave, on BOTH heads (immune/critic lost 9 of
        #      12 configs, sim1/critic 9 of 12, lung/DISC 4 of 12), so scoping the
        #      mitigation to the critic is not enough. Those shards run SERIAL.
        #      bs=1024 is the production setting, has never OOM'd, and stays parallel --
        #      serialising it too would idle five lanes for no reason.
        for head, tsec in (("disc", d_s), ("critic", c_s)):
            h = "discriminator" if head == "disc" else "critic"
            add(f"{ds}_E10_{head}_bs1024",
                f'$PY scripts/run_experiment.py --experiment E10 {common} '
                f'--head {h} --batch-size-only 1024 --resume '
                f'--out results/wave/{ds}_E10_{head}_bs1024.csv',
                3 * (tsec + o_s), big=big_ds)
            add(f"{ds}_E10_{head}_bs4096",
                f'$PY scripts/run_experiment.py --experiment E10 {common} '
                f'--head {h} --batch-size-only 4096 --resume '
                f'--out results/wave/{ds}_E10_{head}_bs4096.csv',
                9 * (tsec + o_s), serial=True, big=big_ds)

        # ---- E4: reference-design sweep (critic only; designs are dataset-specific) ----
        designs = [f"fixed_ref{i}" for i in range(nb)] + ["rotating", "joint", "discriminator"]
        for dsg in designs:
            add(f"{ds}_E4_{dsg}",
                f'$PY scripts/run_reference.py {common} --ref-design-only {dsg} --resume '
                f'--out results/wave/{ds}_E4_{dsg}.csv',
                3 * (c_s + o_s), big=big_ds)

        # ---- E9: critic formulation comparison, 4 arms x 3 seeds ----
        for arm in ("reference", "pooled", "barycenter", "discriminator"):
            tsec = d_s if arm == "discriminator" else c_s
            add(f"{ds}_E9_{arm}",
                f'$PY scripts/run_formulations.py {common} --arms {arm} --resume '
                f'--out results/wave/{ds}_E9_{arm}.csv',
                3 * (tsec + o_s), big=big_ds)

        # ---- E5: biological readouts (both heads inside one process) ----
        add(f"{ds}_E5",
            f'$PY scripts/run_biology.py --registry configs/dataset_registry.json '
            f'--dataset {ds} --data-root "$R" --embed-out "$EMB" --epochs {EPOCHS} '
            f'--outdir results/wave/E5_{ds}',
            (d_s + c_s + 2 * o_s), big=big_ds)

        # ---- E3: external baselines. No adversary, so cheap; still persists latents.
        #      NOTE: run_baselines.py has NO --epochs flag (scVI/scANVI own their own
        #      training schedules and harmony/scanorama/combat do not train at all), so it
        #      cannot reuse `common` -- passing --epochs there is an argparse error that
        #      kills the shard instantly. ----
        # E3 IS a large-VRAM shard despite not training OUR model: its default method list
        # includes scvi and scanvi, which fit their own torch models on the GPU. The first
        # three methods (unintegrated/harmony/scanorama) are CPU-only, which is why a
        # partially-complete E3 shard looks harmless.
        add(f"{ds}_E3",
            f'$PY scripts/run_baselines.py --registry configs/dataset_registry.json '
            f'--dataset {ds} --data-root "$R" --embed-out "$EMB" '
            f'--out results/wave/{ds}_E3.csv',
            5 * o_s, big=big_ds)

    # ---- E6: support overlap. Trains NO model, runs over all datasets at once. ----
    add("E6_support_overlap",
        f'$PY scripts/support_overlap.py --registry configs/dataset_registry.json '
        f'--out results/wave/E6_support_overlap.csv '
        f'--datasets {" ".join(datasets)}',
        300.0)
    return shards


def pack(shards, workers):
    """Longest-processing-time packing of the PARALLEL shards onto lanes.

    # WHY LARGE SHARDS ARE CONFINED TO A LANE SUBSET, NOT MERELY COUNTED:
    #   lanes run independently and each pulls its next shard the moment it is free, so how
    #   many large shards are co-resident at any instant is a property of RUNTIME
    #   INTERLEAVING, which a static assignment cannot bound by counting alone. Confining
    #   every large shard to the first LARGE_LANE_CAP lanes makes the bound STRUCTURAL: at
    #   most that many can ever run at once, whatever the timings turn out to be.
    #   This costs makespan balance -- the large-capable lanes carry the heavy work -- which
    #   main() reports so the trade is visible rather than silent.
    """
    load = [0.0] * workers
    big_lanes = min(LARGE_LANE_CAP, workers)
    par = sorted([x for x in shards if x["phase"] == "parallel"], key=lambda x: -x["seconds"])
    for s in par:
        allowed = range(big_lanes) if s.get("big") else range(workers)
        i = min(allowed, key=lambda j: load[j])
        s["worker"] = i + 1
        load[i] += s["seconds"]
    for s in shards:
        if s["phase"] == "serial":
            s["worker"] = 0          # lane 0 = the serial phase
    return load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["light", "all"], default="light",
                    help="light = the 6 light datasets; all = all 8 including "
                         "atac_large and immune_hum_mou")
    ap.add_argument("--registry", default="configs/dataset_registry.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--embed-root", default="$EMB")
    ap.add_argument("-o", "--out", default="scripts/wave_manifest.tsv")
    args = ap.parse_args()

    with open(args.registry) as fh:
        reg = json.load(fh)
    datasets = LIGHT if args.scope == "light" else LIGHT + HEAVY

    shards = build(reg, datasets, args.embed_root)
    load = pack(shards, args.workers)

    # ---- assertions: the manifest must be wrong LOUDLY, not silently ----
    for s in shards:
        assert "--embed-out" in s["cmd"] or "support_overlap" in s["cmd"], \
            f"{s['tag']} does not persist embeddings"
        assert "\t" not in s["cmd"], f"{s['tag']} command contains a tab"
    tags = [s["tag"] for s in shards]
    assert len(tags) == len(set(tags)), "duplicate shard tags"
    for s in shards:
        if "_E8_" in s["tag"]:
            assert "--batch-count" in s["cmd"], f"{s['tag']} missing --batch-count"
            ds = s["tag"].split("_E8_")[0]
            assert reg[ds]["n_batches"] > 2, f"E8 emitted for 2-batch dataset {ds}"
        if "_E10_" in s["tag"]:
            # Only the VRAM-heavy arm must be serial; bs=1024 is the production setting.
            assert ("--batch-size-only" in s["cmd"]), f"{s['tag']} must split E10 by batch size"
            if "bs4096" in s["tag"]:
                assert s["phase"] == "serial", f"{s['tag']} must be serial (bs=4096 OOMs)"
            else:
                assert s["phase"] == "parallel", f"{s['tag']} need not be serial"
    # every harness that trains must offer --resume so a failure costs only its remainder
    for s in shards:
        if any(k in s["tag"] for k in ("_E1_", "_E2_", "_E8_", "_E10_", "_E4_", "_E9_")):
            assert "--resume" in s["cmd"], f"{s['tag']} is not resumable"
    # Every shard that TRAINS must set the epoch ceiling explicitly. E3 runs external
    # baselines (no adversary, no epoch budget) and E6 trains nothing.
    for s in shards:
        if s["tag"].endswith("_E3") or "support_overlap" in s["cmd"]:
            continue
        assert f"--epochs {EPOCHS}" in s["cmd"], (
            f"{s['tag']} does not set --epochs {EPOCHS}; the harness default is 150, "
            f"which truncates the critic head mid-improvement")

    with open(args.out, "w") as fh:
        fh.write("phase\tworker\ttag\test_hours\tcmd\n")
        for s in sorted(shards, key=lambda x: (x["phase"] != "parallel", x["worker"])):
            fh.write(f"{s['phase']}\t{s['worker']}\t{s['tag']}\t"
                     f"{s['seconds'] / 3600:.3f}\t{s['cmd']}\n")

    par = [s for s in shards if s["phase"] == "parallel"]
    ser = [s for s in shards if s["phase"] == "serial"]
    tot = sum(s["seconds"] for s in shards) / 3600
    print(f"scope={args.scope}  datasets={len(datasets)}  shards={len(shards)} "
          f"({len(par)} parallel, {len(ser)} serial)")
    print(f"total {tot:.1f} worker-hours")
    print(f"parallel makespan {max(load) / 3600:.1f} h across {args.workers} lanes "
          f"(balance {min(load) / max(load) * 100:.0f}%)")
    print(f"serial phase      {sum(s['seconds'] for s in ser) / 3600:.1f} h")
    print(f"estimated wall-clock {(max(load) + sum(s['seconds'] for s in ser)) / 3600:.1f} h")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
