# JHPCE production fit job (code check CR-03)

Branch `jhpce-production` (from main c9568fd). Nothing here has been submitted: every JHPCE submission needs the user's
explicit go (CONSTRAINTS.md SI-29), and the fixes of the code check land first (SI-40).

## Files

| file | role |
|---|---|
| `cluster/jhpce/prod_fit_job.sh` | the job body on the allocated L40S node (steps below, exit codes in its header) |
| `cluster/jhpce/prod_helpers.py` | inputs check, concurrency slots, claims guard, versions check, stop-guard deadline, stage key read from the runner's output, job summary, harvest pack |
| `cluster/jhpce/prod_command.py` | prints the submission plan (harness `command` with the `#SBATCH` header, bundle input, run_timeout_s, card intent); submits nothing |
| `cluster/jhpce/harvest_local.py` | local side of the harvest: verify every part / the tar / every file against SHA-256, then an all-or-nothing merge into the durable OUT_DIR |
| `cluster/jhpce/make_tags.py`, `cluster/jhpce/tags/` | tags files of the first job, derived read-only from the tagged manifest (`--check` re-derives and compares) |
| `cluster/jhpce/expected_fit_versions.json` | the wcd-fit module versions, CUDA 12.6 and cuDNN 9.10.2 measured on the L40S gate node; the job aborts on any difference |
| `cluster/jhpce/local_rehearsal.sh` | local rehearsal of a job's full dry-run path at one commit (bundle clone, every check, the runner's dry run); stubs only the GPU node and Slurm |
| `tests/jhpce/test_prod_job.py` | tests on the real `scripts/run_stage.py` (fits by the runner's own fake fitter `tests/stage/fake_env.py`; fake nvidia-smi / sacct / squeue): refusals, stage key, stop guard + resume, SIGKILL + stale claims, slots, claims, harvest checks |
| `docs/jhpce/prod_data/fit_peak_rss_cpu_1epoch.csv` | measured per-fit peak RSS used for `--mem` |

## Requirements of the code check, item by item

1. **Slurm header** (`prod_command.py`): `--partition=gpu --account=jhpce --gres=gpu:l40s:1 --cpus-per-task=10` (8 lanes + 2),
   `--mem=96G`, `--time=1-00:00:00`, `--signal=B:TERM@300`, `--no-requeue` (a failed or killed job is resubmitted only after
   a new go). Memory: per-fit peak RSS measured with the fitter itself (one epoch, CPU, local scvi-api env):
   immune_hum_mou 4.01 GiB (none), 4.01 (scanvi), 3.82 (sysvi); atac_large 2.75 (none), 2.50 (sysvi); immune 2.05 GiB,
   which matches the gate's 8-lane L40S job (MaxRSS 18.4 GiB / 8 = 2.3 GiB per lane). 8 lanes x ~4.3 GiB + runner
   ~ 36 GiB for the largest tasks; 96 GB leaves about 2.6x headroom (L40S nodes have 1 TB, CPUs are the scarce resource:
   24 per 4 GPUs). The harness parses `#SBATCH` lines only at the top of the command and breaks on `|` constraints;
   the header has neither problem.
2. **Environment**: `source cluster/jhpce/env.sh`; refuses unless `$FIT_ENV/.verified` exists; OUT, logs, TMPDIR, caches,
   the repo clone and the harvest are on `/fastscratch/myscratch/$USER/wcd`; the job refuses if any of them resolves under
   `$HOME`. Only the harness's own workdir (bundle, Slurm log, two KB-sized JSON files) is in HOME.
3. **Repository**: the command clones the staged bundle on fastscratch and checks out `--expected-sha`; the job compares
   `git rev-parse HEAD` with it and refuses a dirty tree.
4. **Inputs**: manifest and tags-file SHA-256 against the values in the command (computed from the files as committed at the
   expected SHA); every tag present once; the tags' experiments and tasks equal the job's; `fingerprint_prepped.py --compare
   docs/prepped_fingerprints_scib.json` must exit 0.
5. **Node witness and tests** (the first step on the node once the commit is confirmed): `nvidia-smi` must list exactly one GPU whose name contains L40S; `env_versions.py --kind fit`
   must equal `expected_fit_versions.json` (torch 2.13.0+cu126, scvi 1.4.2, ..., CUDA available, GPU L40S);
   `WCD_SRC=src pytest -q tests/scvi` must pass at that commit. Any failure aborts the job (exit 2) before any fit.
6. **Runner**: `$FIT_PY scripts/run_stage.py --no-score --manifest M --experiments X1 X13 --tasks <job tasks> --tags-file
   <job tags> --out-dir $SCRATCH/tier12/fillers_x1_x13 --prepped-dir $PREPPED_DIR --fit-python $FIT_PY --wcd-src $REPO/src
   --expect-device L40S --fit-lanes 8 --fit-threads 1 --fit-timeout-s 7200 --max-attempts 2`, KMP_AFFINITY=disabled,
   never --allow-dirty / --allow-mixed-sha; `--dry-run` first, logged. Fit timeout: the longest predicted fit of these rows
   is 0.201 local-lane-hours (immune_hum_mou sysvi), i.e. 21 min on an L40S running 8 lanes (8-lane factor 4.62);
   7200 s is about 5.7x that, because no fit of these tasks has been timed on an L40S yet. `--tags-file` is the
   interface of branch fix-runner (fda7578, unchanged since c8da1f8: one tag per line, '#' comments, every tag a row
   of --experiments x --tasks).
   **Stage key.** A `--tags-file` run has its own key `<experiments>__<tasks>__tags-<sha256(tags file)[:12]>` and its
   own `ledger/<key>.{csv,json,done}` (jobA `X1+X13__atac_large+immune_hum_mou__tags-74dc397a9337`, jobB
   `X1+X13__lung+pancreas+sim2__tags-8d5d4b07c0f3`). The job never rebuilds the key: `prod_helpers.py stagekey` reads it
   from the runner's own `[stage] <key>: N rows ...; out <OUT>` line and checks that it carries this tags file's hash, that
   N equals the number of tags, that `<OUT>` is the job's OUT and that the run's key equals the dry run's (exit 11
   otherwise). The summary reads `ledger/<key>.json` and checks `stage`, `out_dir`, `tags_file` (SHA-256, n_tags),
   `manifest_sha256`, `runner_git` (the expected commit, clean) and `row_kinds` (GPU rows only) against the job; a `.done`
   marker counts only if its runner id is that of the ledger and both gates passed. A run refused in its preflight (exit 4)
   prints no key, and no ledger is attributed to the job. A run stopped before it planned (killed during start-up)
   prints none either; the key then comes from the `--report-only` pass's own line, which must equal the dry run's.
7. **Claims and concurrency**: claims of earlier jobs on the job's tags are cleared (`--clear-foreign-claims
   --clear-stale-claims`) only when sacct shows the owning job ended; a live or unidentifiable owner aborts the job (exit 7).
   At most 2 production jobs run at once: each holds one of two slots on fastscratch (atomic mkdir; a slot of an ended job
   is reclaimed), and the two live jobs must have disjoint tasks. The job never cancels anything (shared account).
8. **Exit handling**: 0 gate passed; runner codes 4 / 5 / 6 / 128+n passed through; job codes 2 node witness or tests,
   3 repository, 7 concurrency or claims, 8 inputs, 9 environment, 10 harvest pack, 11 the runner's output breaks the
   stage-key / ledger contract (with a passing gate). `job_summary.json` holds the stage key, the ledger counts, the gate,
   the `.done` check, the stop reason, any contract problems and the job facts. `--dry-run` stops after the runner's dry
   run and writes the plan (key, rows, fits) instead; nothing is fitted or packed.
9. **Harvest**: `prod_helpers.py pack` collects this job's tags only (latents, models, status, attempts, fit logs,
   quarantine) plus the ledger files of the key read from the runner, the manifest snapshot named in that ledger,
   failures.csv and the job's witness files; it writes a per-file SHA-256 manifest (with the stage key, the ledger files
   and the ledger's runner id), one tar, parts of at most 250 MiB (c.download limit 256 MiB) and `SHA256SUMS`, all on
   fastscratch. The agent downloads the parts, then `harvest_local.py --parts-dir <parts> --dest <durable OUT_DIR>` checks
   every part, the reassembled tar and every file, that `out/ledger/` holds exactly the recorded ledger files, all named
   after the recorded key, that the ledger JSON names that stage, tags-file hash and runner id, and that a `.done` comes
   from the passing run that wrote the ledger; then it merges all-or-nothing (new files copied, identical skipped, grown
   attempt logs replaced, any other difference refused). Views (ledger, failures.csv) and witness files are kept under
   `<dest>/_harvests/<base>/`; local scoring rebuilds the views.
   **Proposed durable path:** `/home/kendall/experiment_data/wasserstein-critic-deconfounding/jhpce_tier12/fillers_x1_x13/`
   (parts staged in `.../jhpce_tier12/_incoming/<base>/`). harvest_local refuses any destination inside a `tier12/` directory.
   Then score locally: `run_stage.py` (scoring mode) on that directory with `--expect-device L40S` (all scoring local, G4).
10. **Stage gating**: the first jobs run only the 250 rows PREREG sec 0 allows before the rules; X1 adversarial rows wait for
   the committed R1, R4-a1 and R2/R3 records, the follow-ups for R4-x1; each submission needs a go.

**Stop guard.** The harness wrapper exits as soon as Slurm signals it, so a Slurm signal alone does not leave time to stop
cleanly. The job therefore reads its own end time (`squeue %L`) and sends SIGTERM to the runner 45 min before it; the runner
stops its lanes (30 s grace), releases its claims and writes its views; if it is still alive 240 s later the job kills it.
After either, the views are not final (no gate) or may be stale, so the job rebuilds them from the per-tag files with the
runner's own `--report-only` (same key, starts nothing, writes no `.done`), then writes the summary and packs the harvest.
Rerunning the same command at the same commit resumes on the same key: fitted rows are kept by the runner's preflight,
interrupted rows rerun, and the claims of a killed runner (it still holds those of its finished rows as well) are
cleared once Slurm shows its job ended (tested). A resume
at another commit would fail the runner's mixed-commit gate (latents of one stage come from one commit).
The harness's own run clock counts from submission (queue time included), so the plan sets `run_timeout_s` = wall + 120 h.

## First job: the 250 pre-freeze rows of the JHPCE tasks

Derived with `make_tags.py` from `manifests/paper_manifest_stock_pilot_u5_b10.tsv` (sha256 c0244ea4fd76ea05628060c69034d8947e5e4b338cb5e9952de166614a416afb):
X1 arms none and scvi_adv, X13 arms scanvi and sysvi, tasks of SI-39. Per task: X1 none 10 (5 seeds x cond 0/1), X1
scvi_adv 5 (cond 1), X13 scanvi 5, X13 sysvi 30 = 50; total 250. No row holds a placeholder.

| job | tasks | rows | tags file (sha256) | predicted lane-h | predicted wall on 1 L40S |
|---|---|---|---|---|---|
| jobA | atac_large, immune_hum_mou | 100 | fillers_x1_x13_jobA.tags (74dc397a...) | 18.0 | 3.9 h |
| jobB | lung, pancreas, sim2 | 150 | fillers_x1_x13_jobB.tags (8d5d4b07...) | 25.0 | 5.4 h |

Lane-hours: scripts/cost_model.py (local per-step throughput); wall = lane-h / 4.62 (L40S 8-lane factor, gate G5, measured
on 3-epoch immune fits only). The 24 h wall request is >= 4x the estimate (first full-length fits of these tasks on an L40S).

## X13 CPU baselines of the JHPCE tasks (proposal)

80 rows (per task: harmony 7, scanorama 7, pca 2). SI-17 puts every fit of a task on one machine and SI-39 puts these
tasks' fits on JHPCE L40S, so they should not run locally (the local CPUs are also the scoring bottleneck).

- **Recommended:** inside the L40S jobs, on the same node type as the task's GPU fits (the L40S nodes; compute-171
  reports AMD EPYC 7443P), no extra queue wait. The runner at fix-runner fda7578 fits X13 CPU rows itself
  (`--cpu-lanes N --cpu-python <wcd-score python>`, run_cpu_baselines.py, device "cpu"; since fda7578 a CPU row's only
  failure outcome is nonfinite_latent, exit 0 with the latent saved), concurrently with the GPU lanes,
  so the CPU rows can join the same tags file and run once the GPU lanes leave CPUs free; each CPU lane needs its own
  CPUs (L40S nodes have 24 CPUs per 4 GPUs). The job as written passes no `--cpu-lanes`, so the runner refuses CPU rows.
- **Alternative:** one `shared`-partition CPU job (starts within minutes) pinned to a single CPU type with a single-feature
  `--constraint`; cfg["cpu_model"] (pinned interface (b)) records the CPU.
- **Prerequisites:** (1) runner support for CPU rows: present at fix-runner fda7578 (CR-02); the job needs `--cpu-lanes`,
  `--cpu-python` and a CPU count sized for them;
  (2) harmony-pytorch 0.1.7 is not in the JHPCE wcd-score env (it was added to local wcd-kbet after the env was built;
  cluster/envspec has harmonypy and scanorama but not harmony-pytorch): an env update job on `shared`, which needs a go;
  (3) sizing from the Scanorama timing: knn 20 on immune_hum_mou took 1,843 s and 10.46 GB peak at 2 threads under load;
  knn 80 / 160 are still running, so the memory cap and wall time of the CPU phase are not fixed yet.
