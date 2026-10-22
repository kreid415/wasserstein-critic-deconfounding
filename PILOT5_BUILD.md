# PILOT5_BUILD: penalty-strength pilot on 5 more datasets (PARTIAL DRAFT, uncommitted)

Written 2026-10-21 on branch `pilot5`. **The shell tool (Bash) was denied in this session, so nothing was run:** no manifest was built, no comparison, no tests, no commit. Everything below was read from files with the Read/Grep/Glob tools. Hand arithmetic is marked "(by hand)".

Plain meanings of the labels used here: **A1** = the penalty-strength pilot (seeds 100-102, 3 datasets); **X1** = the main benchmark (stopped); **R1** = the pre-registered rule that picks the 6-point penalty-strength grid per penalty family and decoder; **R4** = the rule that picks the "matched" strength (the strength that gives a target batch-mixing score); **SI-nn** = numbered decision entries in CONSTRAINTS.md. The conditioned decoder receives the batch label; the unconditioned decoder does not (`cond` column 1 / 0).

## 1. Status

| Deliverable | State |
|---|---|
| Run list (manifest TSV), SHA-256, counts | NOT DONE (needs Python) |
| Column-by-column comparison script and output | NOT DONE |
| Job commands (5 JHPCE jobs) | PLAN ONLY, commands need the manifest SHA-256 |
| Rule check (R1, R4) with file:line | DONE (section 5) |
| Wall-time estimate | DONE (section 4, by hand) |
| Tests, commit hash | NOT DONE |

## 2. Grid-edge strengths: the records disagree with the task text

The task text expects Sinkhorn extension strengths on the **unconditioned** decoder. The v3 records say **conditioned**:

| Source | JS (`discriminator`) extension | Sinkhorn extension |
|---|---|---|
| `D/tier12_v3/rules/r1_pass2/r1_record.json` (the frozen R1 record) | `js|0` (unconditioned): a1_grid starts at 0.01 (L78-91), rounds low=1 (L545-548) | `sinkhorn|1` (conditioned): a1_grid ends at 30000 (L2216-2229), rounds high=1 (L2683-2686); `sinkhorn|0` rounds 0/0 (L2201-2204) |
| `D/tier12_v3/rules/r1_pass1/a1_extension.tsv` (R1's input, sha256 6c38c116...ecffe) | 0.03 and 0.01, `_c0_` | 10000 and 30000, `_c1_` (row 22: `A1_atac_small_sinkhorn_l10000_c1_s100`) |
| `D/tier12_v3/rules/{r1_pass2/m.r1,r4_pass2/m.r4a1,r23/m.x1}.tsv` | `_c0_` | `_c1_` |
| repo `manifests/a1_extension_r1_round1.tsv`, `manifests/*_v3.{r1,r4a1,x1}.tsv`, `experiments/A1_ext_r1_round1/` | `_c0_` | `_c0_` (matches the round-1 extension that `experiments/A1_ext_r1_round1/PREFLIGHT.md` line 7 describes as "sinkhorn/0 c0", dated 2026-10-10, before the 2026-10-13 Sinkhorn fix; INFERENCE: how the v3-named copies came to carry it was not checked) |

Decision used here: follow the v3 records, because R1 at the tag read them (`r1_record.json` inputs, L5003-5011). Grid-edge strengths per new dataset: discriminator, unconditioned, 0.01 and 0.03; Sinkhorn, **conditioned**, 10000 and 30000. Row counts are the same either way. The repo copies under `manifests/` are inconsistent with the records and should not be used as the comparison reference for Sinkhorn.

Existing pilot rows (A1, v3): 1,098 base + 36 R1 extension + 9 R4 extension (immune, conditioned, 0.03: discriminator, reference, pooled; `manifests/a1_extension_r4_round1.tsv`) = 1,143 (grep count on `manifests/paper_manifest_stock_pilot_u5_b10_v3.r4a1.tsv`). The 9 R4 rows are a per-dataset matched-strength extension, not grid points; they are not copied.

## 3. Expected rows per new dataset (from `build_paper_manifest.py` L162-168 and the extension manifest; not yet generated)

| Arm | Unconditioned | Conditioned | Total |
|---|---|---|---|
| none (strength 0) | 3 | 3 | 6 |
| discriminator | 3 x 12 = 36 | 3 x 10 = 30 | 66 |
| mmd | 30 | 30 | 60 |
| sinkhorn | 30 | 3 x 12 = 36 | 66 |
| reference | 30 | 30 | 60 |
| pooled | 30 | 30 | 60 |
| barycenter | 30 | 30 | 60 |
| **Total** | | | **378** (366 base + 12 extension); 5 datasets = **1,890** |

Check against existing: 1,098 / 3 datasets = 366 base rows per dataset; 36 / 3 = 12 extension rows per dataset.

Planned builder change (smallest): factor the A1 loop (L165-168) into a function and add `--a1-tasks T [T ...]` (with `--design pilot`); when given, emit only A1 rows for those tasks plus the R1 round-1 extension (`A1_EXT = [("discriminator", 0, [0.03, 0.01]), ("sinkhorn", 1, [10000, 30000])]`, taken from r1_pass1 `extend`). Default output stays byte-identical (test: existing `tests/prereg/test_manifest_design.py` and `test_manifest_v3.py`, extended with one parametrized case). Planned comparison script: new rows vs the v3 pilot rows (`code/wcd_prereg-tier12-v3/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv`, sha256 6efbe422...cf24, plus `r1_pass1/a1_extension.tsv`), keyed on (arm, cond, strength, seed); ignored columns: `tag` (holds dataset name and a hash of it), `task`, `max_epochs` (derived from cell count by `scvi_epochs`, L114-116; checked separately against the formula for every row). All other 18 columns compared.

## 4. Job plan and wall time

Launcher `experiments/X1_jhpce/x1_jhpce_command.py` cannot be used unchanged:

| Line | Hard-coded | Change for the pilot |
|---|---|---|
| L23, L73-74 | refuses any commit but `prereg-tier12-v2` (c88cce3) | use the v3 tag commit `10a1ca5874fbcdc033b148525b5e71d866306bdb` (packed-refs L7-8). Required: Sinkhorn rows at c88cce3 are refused by `docs/EQUIVALENT_COMMITS.json` (`not_for_arms: sinkhorn`) |
| L24 | X1 manifest | the new pilot manifest |
| L26 | the 5 datasets | unchanged (immune_hum_mou, atac_large, lung, pancreas, sim2) |
| L27, L93 | stage `x1_jhpce`, `--experiments X1` | stage `pilot5_jhpce`, `--experiments A1` |
| L86 | tags file `x1_jhpce_<task>.tags` | new tags files, 378 tags each; selection must keep the `none` rows (`make_x1_tags.py` L23, L56 drops them because X1's were fitted by earlier filler jobs; here there are none) |
| L91-94 | no `--slots` | add `--slots 5`: `cluster/jhpce/prod_fit_job.sh` defaults to 2 simultaneous GPU jobs (L12, L37); the user allows 5. Its guard also requires disjoint datasets, which holds |
| L66-72 | chain option | not used: 5 jobs, chain position 0, all start at submission |

Job: partition `gpu`, account `jhpce`, `gpu:l40s:1`, 10 CPUs, 96 GB, 8 fits at a time (`--fit-lanes 8`), per-fit timeout 28,800 s, `--walltime 3-00:00:00`. Fit only (scoring separate). Same SBATCH lines as `SUBMISSION_PLAN.md` L45-65.

Fits per dataset and measured mean per fit (immune_hum_mou, L40S, about 8 fits sharing a GPU, as given by the user), fit-seconds by hand:

| Arm | Fits | Mean s | Fit-seconds |
|---|---|---|---|
| discriminator | 66 | 1,235 | 81,510 |
| mmd | 60 | 1,199 | 71,940 |
| sinkhorn | 66 | 2,866 | 189,156 |
| pooled | 60 | 4,055 | 243,300 |
| reference | 60 | 5,039 | 302,340 |
| barycenter | 60 | 5,828 | 349,680 |
| none | 6 | not measured; upper bound = discriminator 1,235 | 7,410 |
| **Total** | 378 | | **1,245,336** |

Wall time = fit-seconds / 8 at immune_hum_mou speed = 155,667 s = **43.2 h**; plus a tail of up to about one longest fit (1.6 h mean) = about **45 h**. Margin to the 72 h limit: 1.6x (below the usual 2-3x padding; the cluster limit is the cap). If a job runs out of time, the runner stops 45 min early (`--guard-margin-s 2700`) and keeps finished fits; one resume job fits the rest (`SUBMISSION_PLAN.md` L19).

Scaling by training work (cells x epochs, epochs = min(400, round(8,000,000 / cells))), by hand: immune_hum_mou 97,861 x 82 = 8.02M (1.00); atac_large 84,813 x 94 = 7.97M (0.99); lung 32,472 x 246 = 7.99M (1.00); sim2 19,318 x 400 = 7.73M (0.96); pancreas 16,382 x 400 = 6.55M (0.82). Estimated hours: immune_hum_mou 43, atac_large 43, lung 43, sim2 42, pancreas 35 (plus tail). This assumes cost per training step is the same across datasets; the X1 plan's measured per-dataset ratios ranged 0.90-1.21 (`SUBMISSION_PLAN.md` L12-16), so allow up to 1.21x = 52 h, still inside 72 h.

Commands are NOT generated (they embed the manifest and tags SHA-256, unavailable until the manifest is built).

## 5. Rule check (R1 grid rule, R4 matched-strength rule at the pilot stage)

**Verdict: R1 and R4 stage a1 run unchanged with 8 pilot datasets. No line needs editing for the count of datasets.** Four conditions must hold, and one hazard exists.

Lines that name specific datasets or assume a pilot size:

| Line | What | Effect with 8 datasets |
|---|---|---|
| `prereg_rules.py` L64-68 | CC_TASKS, TRAJ_TASKS, REAL_TASKS, TASK_FAMILY: the 8 benchmark datasets, not the pilot's | none; covers the 5 new datasets |
| `prereg_rules.py` L717, L723 | power report uses D = len(REAL_TASKS) = 6 real datasets | none: describes the X1 design, not the pilot size |
| `prereg_rules.py` L714, L401 | "need >= 2 A1 tasks per decoder"; ">= 3 grid points" | satisfied; the 3 in L401 counts grid points |
| `freeze_a1_grid.py` L69 | prints the label `mde80(D=6)` | text only |
| `build_paper_manifest.py` L20 (docstring), L66 `A1_TASKS`, L156 `a2_tasks`, L165, L174 | name the 3 pilot datasets / the 2 companion-pilot datasets | A2/A3 are not extended, so L156/L174 stay; L66/L165 feed the planned flag |
| `tests/prereg/test_manifest_design.py` L41 | A1 tasks == `A1_TASKS` | still true for the default build |
| docs: `PREREG.md` L13, L48; `CONSTRAINTS.md` L99 (SI-16), L105, L199, L260 (SI-17, SI-30) | say "3 pilot datasets" and local-GPU placement | wording; SI-17 (one machine per dataset) still holds: the 5 new datasets stay on one GPU model |

The rule code reads the dataset list from the data: `prereg_rules.py` L480 (`tasks = sorted(set(a1.task))`), L494, L561; `freeze_a1_grid.py` L53.

Conditions and hazards:
1. **Same grid on every dataset** (`prereg_rules.py` L486-488): R1 raises if any dataset's strength grid differs within a family and decoder. So the 4 extension strengths must be in all 5 new datasets (they are, by design). Leaving them out would stop R1.
2. **Do not feed the R4 extension rows to R1**: the 9 immune rows at 0.03 (conditioned) differ from the other datasets' grids, so R1 would raise at L487-488. R1 originally ran before them; rerun R1 with the base manifest, the R1 extension, and the new pilot manifest only.
3. **Every A1 row must be scored or recorded as failed** (`outcomes`, L311-313): R1 cannot run before all 1,890 new fits are scored.
4. **One scorer for all rows** (`check_provenance`, L263-266): scorer commit, CPU type, numba setting and package versions must be identical across all 8 datasets, else R1 raises "mix scorer provenance". Score the new fits on the same workstation and scorer as the first 3 datasets (or rescore all).
5. Not a code problem: a new R1 result can change the 6-point X1 grid (interval is min/max over all datasets, L522). The frozen 3-dataset R1 record, and records built on its grid (R2/R3, R4 stage x1), no longer describe the benchmark once R1 is rerun.

**Rounds left** (R1_MAX_ROUNDS = 2, `prereg_rules.py` L50, used at L499 and L501; rounds counted by `_rounds`, L469-473, from the record's `rounds` fields):

| Penalty family, decoder | Low-end rounds left (points) | High-end rounds left (points) |
|---|---|---|
| discriminator (JS), unconditioned | 1 (0.003, 0.001) | 2 (10000, 30000, then 100000, 300000) |
| discriminator, conditioned | 2 | 2 |
| MMD, both decoders | 2 | 2 |
| Sinkhorn, unconditioned | 2 | 2 |
| Sinkhorn, conditioned | 2 | 1 (100000, 300000) |
| W1 critics (reference, pooled, barycenter), both decoders | 2 | 2 |

Source: `r1_pass2/r1_record.json` rounds fields (L545-548 js|0; L959-962 js|1; L1373 and L1787 mmd; L2201-2204 sinkhorn|0; L2683-2686 sinkhorn|1; L3831 and L4979 w1); next points from `lam_step`/`lam_beyond` (L130-143) (by hand: 0.01 -> 3e-3 -> 1e-3; 30000 -> 1e5 -> 3e5).

**If a new dataset hits a grid edge** (`prereg_rules.py` L499-502): the family and decoder get 2 more points on that side if rounds are left, otherwise the edge is flagged "unresolved" and the run is not extended (L511-519). The extension is requested for **all 8 datasets**, not only the one that hit the edge: `freeze_a1_grid.py` L53-56 loops over every A1 dataset and every arm of the family (3 seeds each, rows copied by `extension_rows`, `prereg_rules.py` L737-756). Cost per trigger: 48 fits (JS, MMD, Sinkhorn: 1 arm x 8 datasets x 2 points x 3 seeds) or 144 fits (W1: 3 arms).

**R4 stage a1** (`prereg_rules.py` L552-598; `freeze_matched_lambda.py`): each cell (dataset x decoder x arm) takes its own target (L563-571), so adding datasets cannot change an existing cell. Only cells used by A2/A3 rows may request extension fits (`needed_cells`, `freeze_matched_lambda.py` L63-70, passed at L105; `prereg_rules.py` L587); A2/A3 use atac_small and immune only, so the 5 new datasets get matched values flagged `edge_not_extended` (L592) and request no fits. Each new dataset needs at least one failure-free strength per arm (L568), else the rule stops for a human decision. R4 has its own limit of 2 points beyond each edge of R1's grid per cell (`R4_MAX_ROUNDS` = 2, L51, L588); immune conditioned has used 1 on the low side for 3 arms.

## 6. Still to do once a shell is allowed

1. Builder flag + comparison script + extend 1 parametrized test; run `python -m pytest tests/prereg -q` and the scripts-tier check.
2. Build `manifests/pilot5_a1_manifest.tsv`; record SHA-256 and counts per dataset/arm/decoder; run the comparison; paste output here.
3. Adapt the launcher (section 4), generate (not run) the 5 submit plans.
4. Commit on `pilot5` (no push); record the hash.
