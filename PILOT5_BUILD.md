# PILOT5_BUILD: penalty-strength pilot on 5 more datasets (run list, job plan, rule check)

Branch `pilot5` of the wcd clone, 2026-10-21. Nothing was submitted, nothing was run on the cluster, nothing was pushed.

Plain meanings: **A1** = the penalty-strength pilot (seeds 100-102; first 3 datasets atac_small, immune, sim1). **X1** = the main benchmark (stopped). **R1** = the pre-registered rule that picks the 6-point penalty-strength grid per penalty family and decoder (docs/PREREG.md section 2). **R4** = the rule that picks the "matched" strength, the strength whose batch-mixing score is closest to a target. **SI-nn** = numbered decision entries in CONSTRAINTS.md. Decoder: `cond` 1 = conditioned (receives the batch label), 0 = unconditioned. JHPCE = the Johns Hopkins cluster. L40S = its GPU model.

## 1. Result at a glance

| Item | Result |
|---|---|
| Manifest | `manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv`, 1,890 rows = 5 datasets x 378, sha256 `5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff` |
| Column-by-column comparison with the existing pilot rows | PASS (18 columns compared, 3 ignored; section 4) |
| Grid-edge strengths verified from the records | discriminator unconditioned 0.01, 0.03; Sinkhorn **conditioned** 10,000, 30,000 (the task text said unconditioned for Sinkhorn; the v3 records say conditioned; section 3) |
| Jobs | 5, one per dataset, 1 L40S each, all start at submission, `--slots 5` (section 6) |
| Predicted wall time per job | 43.2 h of fits + 1.6 h tail = about 45 h (limit 72 h; margin 1.6x) |
| Rules R1 and R4 stage a1 with 8 datasets | work unchanged; no line needs editing (section 7) |
| Tests | `tests/prereg` 140 passed, 1 skipped; coverage check OK (section 5) |

## 2. How the manifest was built

```
python3 scripts/build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5 --bary-iter 10 \
    --a1-tasks immune_hum_mou atac_large lung sim2 pancreas --out manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv
```
Output: `1890 fits (0 design rows reuse X1 fits)`. The flag `--a1-tasks` (builder, `build(..., a1_tasks=)`) writes only the A1 rows of the named datasets: no-penalty rows + 6 arms x 10 base strengths, both decoders, seeds 100-102, posterior-mean adversary input, standardisation off, then the R1 round-1 grid-edge rows from the constant `A1_EXT_R1`. Without the flag the builder output is unchanged (`test_v3_is_the_builder_output` byte-compares it with the committed v3 manifest).

Rows per dataset (identical for all 5; `compare_a1_pilot5.py` prints them per dataset):

| Arm | Unconditioned (cond 0) | Conditioned (cond 1) | Total |
|---|---|---|---|
| none (strength 0) | 3 | 3 | 6 |
| discriminator | 36 (12 strengths x 3 seeds) | 30 | 66 |
| mmd | 30 | 30 | 60 |
| sinkhorn | 30 | 36 (12 strengths x 3 seeds) | 66 |
| reference | 30 | 30 | 60 |
| pooled | 30 | 30 | 60 |
| barycenter | 30 | 30 | 60 |
| **Total** | 189 | 189 | **378** |

378 = 366 base + 12 grid-edge rows, as expected; 5 x 378 = 1,890. Run lists (one per dataset, 378 tags each) in `experiments/pilot5_jhpce/tags/`, made by `python3 experiments/X1_jhpce/make_x1_tags.py --job pilot5 --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv --out-dir experiments/pilot5_jhpce/tags` (the `none` rows are included; the X1 version leaves them out because earlier filler jobs fitted them):

| Dataset | Run list sha256 |
|---|---|
| immune_hum_mou | e76c97f4f450ce20c44b4c2ca2d27a014f8b3a9e5c38ac4c506ad322a0a13a88 |
| atac_large | 32532edd410d4a93a20d80d1792595090b9630f4df3d1b87d98be0a907756f47 |
| lung | 22be99ea6502a5699a8aaa99d87c7bf0142239553cf0a3e9db93d1d9cfdb190c |
| sim2 | 94a5340b140b5e382994ca3678442ab78cd90d5ca6000a7a344b9b414d6d2888 |
| pancreas | a6efc1a44c2c64a7068d5bafe69e08c69ddef3a858540406b8c6d94cbc1a2baf |

## 3. The grid-edge strengths: the records disagree with the task text, and with part of the repo

| Source | Discriminator extension | Sinkhorn extension |
|---|---|---|
| `D/tier12_v3/rules/r1_pass2/r1_record.json` (the frozen R1 record) | `js|0` unconditioned: a1_grid starts at 0.01 (L78-91), rounds low=1 (L545-548) | `sinkhorn|1` **conditioned**: a1_grid ends at 30000 (L2216-2229), rounds high=1 (L2683-2686); `sinkhorn|0` rounds 0/0 (L2201-2204) |
| `D/tier12_v3/rules/r1_pass1/a1_extension.tsv` (R1's input, sha256 6c38c116...ecffe) | 0.03, 0.01 with `_c0_` | 10000, 30000 with `_c1_` |
| `D/tier12_v3/rules/{r1_pass2/m.r1,r4_pass2/m.r4a1,r23/m.x1}.tsv` | `_c0_` | `_c1_` |
| repo `manifests/a1_extension_r1_round1.tsv`, `manifests/*_v3.{r1,r4a1,x1}.tsv`, `experiments/A1_ext_r1_round1/` | `_c0_` | `_c0_`: matches the round that `experiments/A1_ext_r1_round1/PREFLIGHT.md` line 7 describes ("sinkhorn/0 c0", 2026-10-10, before the 2026-10-13 Sinkhorn fix). INFERENCE: how the v3-named copies came to carry it was not checked |

The v3 R1 record is the one the rules consumed (its inputs are listed at L5003-5011), so the new rows follow it: **Sinkhorn on the conditioned decoder**. `compare_a1_pilot5.py` re-reads the record and the extension file and checks `A1_EXT_R1` against them (output in section 4). The stale repo copies under `manifests/` were not changed.

Existing pilot rows used as the comparison reference (v3): builder manifest (sha256 6efbe422...cf24, = the repo's `manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv`) + `r1_pass1/a1_extension.tsv` = 3 datasets x 378 rows. The 9 R4 extension rows (immune, conditioned, strength 0.03: discriminator, reference, pooled) are per-dataset matched-strength extensions, not grid points; they are not copied.

## 4. Comparison script and its output

`scripts/compare_a1_pilot5.py` (standard library only). Ignored columns: `tag` (dataset name and a hash that includes it), `task`, `max_epochs` (a function of the dataset's cell count; checked separately against the scvi-tools formula for every row). Compared columns (18): experiment, counts, arm, lam, n_critic, adv_input, zstd, cond, decoder, n_latent, n_layers, n_hidden, likelihood, batch_size, train_size, seed, reference, extra.

```
python3 scripts/compare_a1_pilot5.py --new manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv \
    --old manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv D/tier12_v3/rules/r1_pass1/a1_extension.tsv \
    --r1-record D/tier12_v3/rules/r1_pass2/r1_record.json
```
(D = /home/kendall/experiment_data/wasserstein-critic-deconfounding.) Output (exit 0; absolute paths shortened to `...` and 7 per-dataset count lines abbreviated, as marked):

```
PASS old input .../manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv sha256 6efbe422c8f09f9884001a84f7d7b363506f435803d238280e00bbb7b7e1cf24 is listed in the R1 record's inputs
PASS old input .../tier12_v3/rules/r1_pass1/a1_extension.tsv sha256 6c38c11671862637d6f5e4ff998549eeeb87304c845231c56270ec3c78eecffe is listed in the R1 record's inputs
new manifest .../manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv sha256 5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff
old tasks ['atac_small', 'immune', 'sim1'], new tasks ['atac_large', 'immune_hum_mou', 'lung', 'pancreas', 'sim2']
ignored columns ['tag', 'task', 'max_epochs']; compared columns (18) ['experiment', 'counts', 'arm', 'lam', 'n_critic', 'adv_input', 'zstd', 'cond', 'decoder', 'n_latent', 'n_layers', 'n_hidden', 'likelihood', 'batch_size', 'train_size', 'seed', 'reference', 'extra']
grid-edge strengths in R1 record a1_grid: [('discriminator', 0, 0.01), ('discriminator', 0, 0.03), ('sinkhorn', 1, 10000.0), ('sinkhorn', 1, 30000.0)]
grid-edge strengths in old rows: [('discriminator', 0, 0.01), ('discriminator', 0, 0.03), ('sinkhorn', 1, 10000.0), ('sinkhorn', 1, 30000.0)]
grid-edge strengths in builder A1_EXT_R1: [('discriminator', 0, 0.01), ('discriminator', 0, 0.03), ('sinkhorn', 1, 10000.0), ('sinkhorn', 1, 30000.0)]
grid-edge strengths in new rows: [('discriminator', 0, 0.01), ('discriminator', 0, 0.03), ('sinkhorn', 1, 10000.0), ('sinkhorn', 1, 30000.0)]
PASS grid-edge strengths (arm, cond, strength) agree in all four
  R1 js|0: rounds {'high': 0, 'low': 1}, a1_grid 0.01 .. 3000.0 (12 points)
  R1 js|1: rounds {'high': 0, 'low': 0}, a1_grid 0.1 .. 3000.0 (10 points)
  R1 mmd|0: rounds {'high': 0, 'low': 0}, a1_grid 0.1 .. 3000.0 (10 points)
  R1 mmd|1: rounds {'high': 0, 'low': 0}, a1_grid 0.1 .. 3000.0 (10 points)
  R1 sinkhorn|0: rounds {'high': 0, 'low': 0}, a1_grid 0.1 .. 3000.0 (10 points)
  R1 sinkhorn|1: rounds {'high': 1, 'low': 0}, a1_grid 0.1 .. 30000.0 (12 points)
  R1 w1|0: rounds {'high': 0, 'low': 0}, a1_grid 0.1 .. 3000.0 (10 points)
  R1 w1|1: rounds {'high': 0, 'low': 0}, a1_grid 0.1 .. 3000.0 (10 points)
PASS 1890 new rows over 5 tasks equal the 378 old rows per task in 18 columns; max_epochs follows the cell count in all 3024 rows
rows per task, arm and decoder (cond 0 = unconditioned, 1 = conditioned):
  atac_large (new, 378 rows): barycenter c0 30, barycenter c1 30, discriminator c0 36, discriminator c1 30, mmd c0 30, mmd c1 30, none c0 3, none c1 3, pooled c0 30, pooled c1 30, reference c0 30, reference c1 30, sinkhorn c0 30, sinkhorn c1 36
  atac_small (old, 378 rows): barycenter c0 30, ... (the same 14 counts)
  immune (old, 378 rows): ... (the same 14 counts)
  immune_hum_mou (new, 378 rows): ... (the same 14 counts)
  lung (new, 378 rows): ... (the same 14 counts)
  pancreas (new, 378 rows): ... (the same 14 counts)
  sim1 (old, 378 rows): ... (the same 14 counts)
  sim2 (new, 378 rows): ... (the same 14 counts)
```
(The 7 abbreviated lines are shortened here only; the script prints all 14 counts for every dataset, the first line shows them, and the table in section 2 gives them.)

## 5. Tests

| Command (repo root, Python 3.11 of the `scvi-api` conda env) | Result |
|---|---|
| `python3 /home/kendall/scratch/run_in_env.py -m pytest tests/prereg -q -p no:cacheprovider` (the requested `python3 -m pytest tests/prereg -q` does not run: system python3 has no pytest or pandas, so a 3-line launcher runs the same command in the repo's own env) | 140 passed, 1 skipped (36 s) |
| `scripts/check_tests.py --base 10a1ca5874fbcdc033b148525b5e71d866306bdb --src scripts experiments/X1_jhpce -- tests/prereg tests/jhpce/test_x1_command.py` | `check_tests: OK (every changed --src line and branch is tested; no redundant new tests)`; 143 passed, 1 skipped |
| `pytest tests/jhpce tests/x13 tests/analysis tests/test_legacy_retired.py tests/scvi/test_runner_and_manifest_missing_arms.py tests/scvi/test_x3_si41.py tests/scvi/test_x3_x13_design.py` (the other tests that import the builder) | 154 passed, 16 skipped: 15 need prepped data files (`PREPPED_DIR`), 1 needs the `scib` package; none failed |

Tests added or extended (fewest that run every changed line and branch): `tests/prereg/test_manifest_design.py::test_a1_tasks_flag_extends_a1_with_the_r1_grid_edge_strengths` (one parametrized test, 2 cases; covers the builder flag, its error branch, and all of `compare_a1_pilot5.py`, including that it fails on a changed column, a changed `max_epochs`, a dropped row, an old/new overlap, a record that disagrees); `tests/jhpce/test_x1_command.py::test_chained_job_clock_covers_every_job_up_its_chain` (existing test, now parametrized over `--job x1` / `--job pilot5`) and `test_run_lists_follow_the_manifests` (new: committed pilot5 run lists = what `make_x1_tags.py` derives; X1 path still derives 1,800 rows).

Regression checks outside the test suite (scratch scripts, not committed): the changed launcher printed byte-identical X1 plans to the original at 10a1ca5 for all 5 datasets, chain positions 0 and 2 (10 of 10 equal); the changed `make_x1_tags.py --check` gives the same result as the original on the X1 files (all 6 `.tags` files match; `x1_jhpce_counts.csv` differs from its committed copy with the original script too, so that mismatch was already there).

## 6. Job plan (not run)

Launcher: `experiments/X1_jhpce/x1_jhpce_command.py --job pilot5` (default `--job x1` is unchanged). What differs from the X1 job:

| What | X1 job | pilot5 job | Why |
|---|---|---|---|
| Pinned commit | c88cce3 (prereg-tier12-v2) | `10a1ca5874fbcdc033b148525b5e71d866306bdb` (prereg-tier12-v3) | SI-54 (docs/EQUIVALENT_COMMITS.json) refuses Sinkhorn rows at c88cce3 |
| Manifest | `...v3.x1.tsv` | `...v3.pilot5.tsv` | |
| Run lists / stage name | `x1_jhpce_<task>.tags` / `x1_jhpce` | `pilot5_jhpce_<task>.tags` / `pilot5_jhpce` (own output folder on scratch) | |
| `--experiments` | X1 | A1 | |
| `--slots` | job script default (2 at once) | 5 | `cluster/jhpce/prod_fit_job.sh` L12, L37, L116: at most N production GPU jobs at once, datasets disjoint; the cap is 5 |
| Chain | optional | none: all 5 start at submission | 5 GPUs allowed at once |

Unchanged: partition `gpu`, account `jhpce`, `gpu:l40s:1`, 10 CPUs, 96 GB, `--time=3-00:00:00`, 8 fits at a time, per-fit timeout 28,800 s, 2 attempts, stop guard 2,700 s before the limit. The job runs `scripts/run_stage.py --no-score` (fit only). Nothing in the launcher, run lists, manifest or plans says where the latents are scored (the comment block at the top of the tagged `prod_fit_job.sh` mentions a scoring host; it is a comment in tagged code and was left untouched). The job's working folder is still called `x1_inputs` (cosmetic; `rehearse_x1_job.sh` looks for that name).

Commands (run once per dataset from the repo root; `wcd_prereg-tier12-v3.bundle` is a git bundle whose only head is the v3 commit, made with `git update-ref refs/heads/pilot5-bundle-tmp 10a1ca5...; git bundle create <file> refs/heads/pilot5-bundle-tmp; git update-ref -d refs/heads/pilot5-bundle-tmp`):
```
python3 experiments/X1_jhpce/x1_jhpce_command.py --job pilot5 --task <dataset> --bundle <path>/wcd_prereg-tier12-v3.bundle
```
The five generated plans (JSON: submit command, input files, harness clock 192 h) are saved as `experiments/pilot5_jhpce/plans/pilot5_<dataset>.json`. They are not submitted; submission needs your go (SI-29). Dataset-specific lines of the five submit commands (everything else is identical):

| Dataset | `--tasks` and run list in `prod_fit_job.sh` call | tags sha256 (run list) |
|---|---|---|
| immune_hum_mou | `--tags-file ../x1_inputs/pilot5_jhpce_immune_hum_mou.tags --tasks immune_hum_mou` | e76c97f4...3a88 |
| atac_large | `...pilot5_jhpce_atac_large.tags --tasks atac_large` | 32532edd...6f47 |
| lung | `...pilot5_jhpce_lung.tags --tasks lung` | 22be99ea...190c |
| sim2 | `...pilot5_jhpce_sim2.tags --tasks sim2` | 94a5340b...2888 |
| pancreas | `...pilot5_jhpce_pancreas.tags --tasks pancreas` | a6efc1a4...2baf |

Submit command of the first job (immune_hum_mou), exactly as generated:
```
#SBATCH --partition=gpu
#SBATCH --account=jhpce
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --no-requeue
set -euo pipefail
SCR="/fastscratch/myscratch/$USER/wcd"
JOBDIR="$SCR/prod/job_${SLURM_JOB_ID}"
mkdir -p "$JOBDIR/x1_inputs"
printf "%s  %s\n" 5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv e76c97f4f450ce20c44b4c2ca2d27a014f8b3a9e5c38ac4c506ad322a0a13a88 pilot5_jhpce_immune_hum_mou.tags | sha256sum -c --quiet
cp paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv pilot5_jhpce_immune_hum_mou.tags "$JOBDIR/x1_inputs/"
git clone -q --no-checkout "$PWD/wcd.bundle" "$JOBDIR/repo"
git -C "$JOBDIR/repo" -c advice.detachedHead=false checkout -q 10a1ca5874fbcdc033b148525b5e71d866306bdb
rc=0
bash "$JOBDIR/repo/cluster/jhpce/prod_fit_job.sh" --expected-sha 10a1ca5874fbcdc033b148525b5e71d866306bdb --stage pilot5_jhpce --tags-file ../x1_inputs/pilot5_jhpce_immune_hum_mou.tags --tags-sha256 e76c97f4f450ce20c44b4c2ca2d27a014f8b3a9e5c38ac4c506ad322a0a13a88 --manifest ../x1_inputs/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv --manifest-sha256 5dd65550fdb18e5c80e73dcfa731d72bdeb6128bdeac7ffb36df6c54450396ff --experiments A1 --tasks immune_hum_mou --fit-lanes 8 --fit-timeout-s 28800 --max-attempts 2 --guard-margin-s 2700 --slots 5 || rc=$?
for f in "$JOBDIR/logs/job_summary.json" "$JOBDIR/harvest.json"; do [ ! -f "$f" ] || cp "$f" .; done   # KB-sized; the harvest stays on fastscratch
ls -l .
exit "$rc"
```

Checks run locally (no cluster): the job script's own input check (`cluster/jhpce/prod_helpers.py inputs --experiments A1 --tasks <dataset>`) returns `ok: true, n_tags: 378, problems: []` for all 5; the runner's dry run (`scripts/run_stage.py --no-score --expect-device L40S --dry-run` with the job's arguments) reports `378 rows {'needs_fit': 378}` for all 5 (stage keys `A1__<dataset>__tags-<first 12 hex of the run list sha>`).

Predicted wall time per job (`python3 experiments/pilot5_jhpce/walltime.py`): per-fit means measured in the main benchmark on immune_hum_mou, L40S, about 8 fits sharing a GPU (given by you in the task; no ledger timing was opened): discriminator 1,235 s, mmd 1,199 s, sinkhorn 2,866 s, pooled 4,055 s, reference 5,039 s, barycenter 5,828 s; the no-penalty fits were not measured and are priced as discriminator fits (upper bound). Wall time = fit seconds / 8; tail = one barycenter fit running alone at the end; "scaled" multiplies the wall time by training work (cells x epochs) relative to immune_hum_mou, assuming equal cost per training step.

```
dataset          fits  fit-seconds  wall h  +tail h  margin to limit  work vs ref  scaled wall+tail h
atac_large        378    1,245,336    43.2      1.6            1.61x         0.99                44.6
immune_hum_mou    378    1,245,336    43.2      1.6            1.61x         1.00                44.9
lung              378    1,245,336    43.2      1.6            1.61x         1.00                44.7
pancreas          378    1,245,336    43.2      1.6            1.61x         0.82                36.9
sim2              378    1,245,336    43.2      1.6            1.61x         0.96                43.3
```
Plan on 45 h per job (the unscaled 43.2 + 1.6 h is the safe figure for every dataset). The margin to the 72 h limit is 1.6x, below the usual 2-3x padding, because 72 h is the cluster's cap; the X1 plan measured per-dataset speed ratios of 0.90-1.21 (`SUBMISSION_PLAN.md` L12-16), and 1.21x would give 52 h, still inside. If a job reaches the stop guard, finished fits are kept and the same command rerun at the same commit fits the rest (`prod_fit_job.sh` L25-29). The per-fit timeout (28,800 s) is 4.9x the longest mean (5,828 s).

## 7. Rule check: R1 grid rule and R4 matched-strength rule at the pilot stage, with 8 datasets

**Verdict: both work unchanged with 8 datasets; no line needs editing for the number of datasets.** Read from code, then run on synthetic scores (`python3 experiments/pilot5_jhpce/rule_check_8datasets.py`, env with pandas and scipy; real rule scripts, made-up scores, 8 datasets x 378 rows = 3,024 A1 rows):

| Case | Result |
|---|---|
| A. all arms respond inside the grid | R1 exit 0, `frozen`; 8 grids; rounds used read back: js|0 low 1, sinkhorn|1 high 1, all others 0; R4 stage a1 exit 0, `frozen`, 96 cells (8 datasets x 2 decoders x 6 arms), 0 extensions requested |
| B1. pancreas, discriminator responds only above the grid's top | R1 exit 3 (extend): js|0 and js|1 high [10000, 30000]; **96 extension rows over all 8 datasets**, arm discriminator, both decoders, 3 seeds |
| B2. pancreas, reference critic likewise | R1 exit 3: w1|0 and w1|1 high [10000, 30000]; **288 rows over all 8 datasets**, arms reference, pooled, barycenter |
| C. lung, discriminator responds already at the bottom | R1 exit 3: js|0 low [0.003, 0.001] (second low round), js|1 low [0.03, 0.01]; 96 rows over all 8 datasets |

(The synthetic effects were applied to both decoders, so cases B and C show two families x decoders: 48 rows per decoder for discriminator, MMD or Sinkhorn, 144 per decoder for the three Wasserstein critics.)

Lines that name specific datasets or assume a pilot size:

| Line | What | Effect with 8 datasets |
|---|---|---|
| `prereg_rules.py` L64-68 | CC_TASKS, TRAJ_TASKS, REAL_TASKS, TASK_FAMILY: the 8 benchmark datasets, not the pilot's | none; includes the 5 new datasets |
| `prereg_rules.py` L717, L723 | power report uses D = len(REAL_TASKS) = 6 real datasets | none: describes the X1 design, not the pilot size |
| `prereg_rules.py` L714, L401 | "need >= 2 A1 tasks per decoder"; ">= 3 grid points" | satisfied; the 3 in L401 counts grid points |
| `freeze_a1_grid.py` L69 | prints the label `mde80(D=6)` | text only |
| `build_paper_manifest.py` L20 (docstring), L66 `A1_TASKS`, L156 `a2_tasks`, L165, L174 | name the 3 pilot datasets / the 2 companion-pilot datasets | A2/A3 are not extended, so L156/L174 stay; `--a1-tasks` overrides L165's list only for the pilot5 manifest |
| `tests/prereg/test_manifest_design.py` L41 | A1 tasks == `A1_TASKS` | still true for the default build |
| docs `PREREG.md` L13, L48; `CONSTRAINTS.md` L99 (SI-16), L105, L199, L260 (SI-17, SI-30) | say "3 pilot datasets" / local-GPU placement | wording only; SI-17 (one machine and one GPU model per dataset) still holds |

The rule code reads the dataset list from the data: `prereg_rules.py` L480 (`tasks = sorted(set(a1.task))`), L494, L561; `freeze_a1_grid.py` L53.

Conditions and hazards:
1. **Same grid on every dataset** (`prereg_rules.py` L486-488, `test_rules.py` L181): R1 raises "different lambda grid" if any dataset's strengths differ within a family and decoder. The 4 grid-edge strengths therefore must be on all 5 new datasets (they are; case A freezes).
2. **Keep the R4 extension rows out of R1**: the 9 immune rows at 0.03 (conditioned) give immune a different grid from the other datasets for those families, so R1 would raise at L487-488. R1 originally ran before them; rerun R1 with the base manifest, the R1 extension and the pilot5 manifest only.
3. **Every A1 row must be scored or recorded as failed** (`outcomes`, `prereg_rules.py` L311-313): R1 cannot run before all 1,890 new fits are scored.
4. **One scorer for all rows** (`check_provenance`, L263-266, `test_provenance_rule.py` L82-83): scorer commit, CPU type, numba setting, package versions must be identical across all 8 datasets, else R1 raises "mix scorer provenance". Score the new fits with the same scorer commit, CPU type and package versions as the first 3 datasets, or rescore all 8.
5. A new R1 result can change the 6-point X1 grid (interval = min/max over all datasets, L522). The frozen 3-dataset R1 record, and records built on its grid (R2/R3, R4 stage x1), no longer describe the benchmark once R1 is rerun with 8 datasets.

**Rounds left** (R1_MAX_ROUNDS = 2, `prereg_rules.py` L50, used at L499 and L501; counted by `_rounds`, L469-473; confirmed by case A, from the record's `rounds`):

| Penalty family, decoder | Low-end rounds left (next points) | High-end rounds left (next points) |
|---|---|---|
| discriminator (JS), unconditioned | 1 (0.003, 0.001; case C) | 2 (10000, 30000, then 100000, 300000) |
| discriminator, conditioned | 2 (0.03, 0.01, then 0.003, 0.001) | 2 |
| MMD, both decoders | 2 | 2 |
| Sinkhorn, unconditioned | 2 | 2 |
| Sinkhorn, conditioned | 2 | 1 (100000, 300000) |
| W1 critics (reference, pooled, barycenter), both decoders | 2 | 2 |

**If a new dataset hits a grid edge** (`prereg_rules.py` L499-502): that family and decoder get 2 more points on that side if rounds are left, otherwise the edge is flagged "unresolved" and nothing is extended (L511-519). The extension is requested for **all 8 datasets**, not only the one that hit the edge (`freeze_a1_grid.py` L53-56 loops over every A1 dataset and every arm of the family; cases B1, B2, C above).

**R4 stage a1** (`prereg_rules.py` L552-598; `freeze_matched_lambda.py`): each cell (dataset x decoder x arm) takes its own target (L563-571), so adding datasets cannot change an existing cell. Only cells used by A2/A3 rows may request extension fits (`needed_cells`, `freeze_matched_lambda.py` L63-70, passed at L105; `prereg_rules.py` L587); A2/A3 use atac_small and immune only, so the 5 new datasets get matched values flagged `edge_not_extended` (L592; `test_si45_edge_extension.py` L56-59) and request no fits (case A: 0 extensions). Each new dataset needs at least one failure-free strength per arm (L568), else the rule stops for a human decision. R4 has its own limit of 2 points beyond each edge of R1's grid per cell (`R4_MAX_ROUNDS` = 2, L51, L588); immune conditioned has used 1 on the low side for 3 arms.

## 8. Not done / before anyone submits

- No submission, no `ssh`, nothing pushed (your rules). Submitting needs your go (SI-29) and the experiment pre-launch check (experiment-preflight skill, standing rule) for this launch; it was not run here.
- Not checked: that the 5 prepped datasets and the fit environment are staged on JHPCE scratch as the X1 jobs expected (the job's own checks will refuse otherwise).
- The old A1 fits of non-Sinkhorn arms were made at c88cce3, the new ones at the v3 commit; SI-54 declares them result-identical for every arm except Sinkhorn (all pilot5 Sinkhorn fits are at v3; at which commit the old pilot's Sinkhorn fits were made was not checked).
- Stale `manifests/*` copies and `experiments/A1_ext_r1_round1/` still show Sinkhorn grid-edge rows on the unconditioned decoder (section 3); left as they are.
- The branch holds two automatic checkpoint commits (`097c1d1`, `50150c2`, only this file's earlier drafts); `HEAD` is not the v3 tag commit, so the launcher pins the tag commit explicitly rather than reading HEAD.
- No lab-notebook entry was written.

## 9. Commits (branch `pilot5`, not pushed)

- `89ee2df` builder flag, manifest, comparison script, run lists, launcher and run-list script `--job` switch, tests
- `0c20fa3` test additions (bundle check, comparison failure report)
- the commit after these (this file, the 5 plans, `walltime.py`, `rule_check_8datasets.py`): see `git log -1` on the branch.
