# Scoring on JHPCE CPUs: job, match check and time estimate (built 2026-10-21, nothing submitted)

Plain words used below: **JHPCE** = the Johns Hopkins HPC cluster; **A1** = the first-round penalty-strength pilot on atac_small, immune and sim1 (3 seeds, both decoders; its scores may be read); **X1** = the main benchmark (blinded: no X1 score file or X1 ledger timing column was opened); **score file** = the one-row CSV the scorer writes per latent; **latent** = the fitted embedding (`.npz`) of one model; **decoder** = unconditioned (`c0`) or batch-conditioned (`c1`); **SHA-256** = a fingerprint of a file's bytes; **G4** = the 2026-10-03 cross-host scoring test (docs/jhpce/GATE_JHPCE.md); **SI-nn** = a numbered standing rule in CONSTRAINTS.md.

## 1. Answer

| question | answer |
|---|---|
| Which JHPCE environment scores? | `wcd-score` = `$HOME/envs/wcd-score` plus the R kBET library `$HOME/envs/Rlib_kbet` (cluster/jhpce/env.sh; built by build_envs.sh from `cluster/envspec/wcd-kbet.explicit.txt` + `score-pip.txt`, a replica of the workstation env `wcd-kbet`). |
| Do its package versions equal the workstation's? | On 2026-10-03 yes for every scorer package (G1, table in section 3). **Not re-read since**: the fastscratch copy was purge-damaged on 2026-10-04 and envs moved to `$HOME/envs` (SI-49); the `$HOME` copy could not be read from here. The check itself settles it: every score row records its versions, and the comparison fails if they differ. |
| Will "every metric identical" pass? | **Probably not, on past evidence.** G4 ran this comparison on 2 latents and failed (section 4). The check below is still the right next step: it is cheap (the plan's estimate for the 20 latents is 0.15-0.29 h of wall at 20 CPUs) and says which nodes and which metrics differ. |
| The 20 latents | 17 A1 non-Sinkhorn latents (tier12_v2/A1) + 3 A1 Sinkhorn latents (tier12_v3/A1_sinkhorn): atac_small 6, immune 7, sim1 7; 7 arm labels (none 3, discriminator 3, reference 3, pooled 3, mmd 3, barycenter 2, Sinkhorn 3); decoders 10 + 10; strengths 0 to 1000; seeds 100-102. **No immune_hum_mou or atac_large latent exists outside X1** in the folders I could read (section 5). |
| Pass rule | section 2 |
| Time to score 1,890 latents | about **2.0 h** compute at 400 CPUs (planning case), 1.1 h if JHPCE cores are as fast as an idle workstation core; **up to about 11 h** if only 2 TB of RAM can be held at once (section 8). Queue wait is not included. |
| Blockers | (1) G4 history above. (2) JHPCE `shared` limits and the `$HOME` env state could not be read (section 9). (3) Each submission needs the user's go (SI-29). |

## 2. Pass rule (printed by the comparison script)

**PASS only if, for every one of the 20 latents, (a) both score files hold the same latent SHA-256 and the same package-version string (`scorer_versions`), and (b) every metric column is identical: bit for bit as 64-bit numbers, an empty cell counting as "not a number", which equals only another empty cell.** Metric columns are all columns except the identity columns (tag, latent_sha256, experiment, task, arm, lam, cond, seed), the provenance columns (scorer_git_sha, scorer_dirty, score_host, cpu_model, cpu_simd, numba_cpu_name, scorer_versions) and score_seconds: that is the 14 scIB outputs (13 metrics + hvg_overlap, empty by design for embeddings) plus kbet_seed, kbet_r_calls, traj_root_fallback, kbet_labels_forced_one, kbet_labels_skipped = 19 columns. No tolerance. If not identical the script prints, per metric column, the largest absolute difference and how many latents differ, then the differing latents with the cluster node and CPU type they were scored on. Exit code 0 = PASS, 1 = FAIL, 2 = a file, row or column missing.

The same script run with `--ref` on two cluster runs tests whether the cluster repeats itself; if run 2 differs from run 1, no cluster-versus-workstation PASS is possible.

## 3. Scorer environments

Workstation (conda env `wcd-kbet`, R 4.5.2, kBET 0.99.6 = theislab/kBET afc5f431), read from the `scorer_versions` column of the A1 score rows (all 180 Sinkhorn rows' gate record and the 17 chosen non-Sinkhorn rows carry exactly this string; CPU AMD Ryzen 5 3600, `cpu_simd` avx2, `numba_cpu_name` empty):
R_kBET 0.99.6, anndata 0.11.4, numba 0.66.0, numpy 1.26.4, pynndescent 0.5.13, rpy2 3.6.7, scanpy 1.11.5, scib 1.1.7, scikit-learn 1.9.0.

JHPCE `wcd-score`, from the G1 record of 2026-10-03 (docs/jhpce/GATE_JHPCE.md, "G1 versions", score rows): anndata 0.11.4, numba 0.66.0, numpy 1.26.4, pynndescent 0.5.13, scanpy 1.11.5, scib 1.1.7, sklearn 1.9.0, kBET 0.99.6 with the same namespace fingerprint (md5 3b541efa...), R 4.5.2, python 3.11.15; also scipy 1.14.1, pandas 1.5.3, igraph 0.11.9, leidenalg 0.10.2, llvmlite 0.48.0, umap 0.5.12, all equal to local. rpy2: the 2026-10-03 table lists "3.6.5 (dist metadata)" on both hosts (a different package, rpy2-robjects, read by env_versions.py); the score rows read the `rpy2` package (3.6.7); `cluster/envspec/wcd-kbet.explicit.txt` pins rpy2 3.6.7. G1 summary: "wcd-score vs local wcd-kbet: identical=152 differing=1 unexpected=0".

**To check on JHPCE** (cannot be read from here): `ls $HOME/envs/wcd-score/.verified $HOME/envs/Rlib_kbet/.verified` (if missing, `build_envs.sh score` must run, a shared CPU job that needs a go and 7.5 GB of `$HOME` room); `env_versions.py --kind score` output against the table above; and the 8 prepped input files against `docs/prepped_fingerprints_scib.json` (step 3 of the commands).

## 4. What G4 already showed (why the check may fail)

G4 scored 2 latents locally and on JHPCE (docs/jhpce/GATE_JHPCE.md, "G4 scoring equivalence"): only 5 of 12 deterministic metrics agreed within 1e-6.
- The scorer's nearest-neighbour graph (pynndescent, compiled by numba) depends on the CPU's vector width. On compute-174 (AMD EPYC 9555, AVX-512) 251 of 33,506 graph rows differ from the workstation (AVX2); that moves ARI, NMI, isolated-label F1, graph connectivity and trajectory. An AVX-only Opteron (compute-053) built the same graph as the workstation. With `NUMBA_CPU_NAME=generic` the graph was identical on all three CPU types.
- graph iLISI and cLISI differed across hosts and **between two runs on the same JHPCE node** (same graph); they are identical between local runs. Cause unresolved.
- kBET was unseeded then; since the seed fix (`set.seed` before every R kBET call) it is deterministic on one host; its cross-host behaviour has not been measured.

So a PASS needs nodes that build the workstation's graph (not AVX-512, or the `generic` setting) and iLISI/cLISI that repeat exactly on the cluster. The workstation pilot was scored with the default setting (`numba_cpu_name` empty), so the job leaves `NUMBA_CPU_NAME` unset; the `generic` variant is a separate run (section 6) whose PASS would show that cluster-`generic` equals workstation-default on these 20, and would then be the setting to adopt (a user decision; `numba_cpu_name` is recorded in every row and the stage runner refuses stages that mix it).

## 5. The 20 latents

Rule: A1 latents only (A1 = pilot; no X1, X13 or follow-up latent), each with a workstation score file; spread over arms (every arm at least twice), both decoders (10 + 10, each arm with both decoders), the three pilot datasets (atac small: no cell-cycle or trajectory; immune: both; sim1: neither), strengths 0, 1, 3, 10, 30, 100, 300, 1000, and seeds 100, 101, 102. Picked by hand with that rule; list and hashes in `cluster/jhpce/score_check/check_latents.tsv` (columns tag, sha256, task, arm, cond, lam, seed, src_latent, ref_score).

Hash source and checks: for each tag, the SHA-256 is the `latent_sha256` the stage runner recorded in the workstation score file when it scored that latent; I checked that all 20 rows (tag, hash, task, arm, lam, cond, seed) equal the first 8 columns of the score files by searching the files, and that all 20 tags are in the committed manifest (its SHA-256 6efbe422c8f09f98... equals the `manifest_sha256` of the A1 ledger). The 3 Sinkhorn latents I re-hashed with `sha256sum` here: equal. The 17 latents under tier12_v2 I could **not** hash myself (my shell may not read that folder); the staging commands in section 6 verify them with `sha256sum -c`.

| # | tag | dataset | arm | decoder | strength | seed | workstation source | SHA-256 |
|---|---|---|---|---|---|---|---|---|
| 1 | A1_atac_small_none_l0_c1_s100_8d38e2a3 | atac_small | none | c1 | 0 | 100 | tier12_v2/A1 | d0eec57499d8a079f0386b6b2e6b486fdbc9765c8829760e023f9588692b1387 |
| 2 | A1_atac_small_discriminator_l10_c0_s101_b841ed69 | atac_small | discriminator | c0 | 10 | 101 | tier12_v2/A1 | c811de4def2f4c504e7689f41b492ac34c910bc15aa69d524c3041f4086d2b2e |
| 3 | A1_atac_small_reference_l30_c0_s102_17472a8a | atac_small | reference | c0 | 30 | 102 | tier12_v2/A1 | 34e492100ca4903eaea9ccc5b57639cd9f6cc236e3c360b6dff1fcdb9d591c73 |
| 4 | A1_atac_small_pooled_l100_c1_s101_e65e48b7 | atac_small | pooled | c1 | 100 | 101 | tier12_v2/A1 | f08ae0b09d78b90e8b82eea99b0a791f0a0a172c87876b891c83ff6ece58ba3d |
| 5 | A1_atac_small_mmd_l3_c0_s100_0107d6f1 | atac_small | mmd | c0 | 3 | 100 | tier12_v2/A1 | e06aea3590a25c32619b753c88148762a13c019071ff64a930a990e8485f4ba7 |
| 6 | A1_atac_small_sinkhorn_l300_c1_s102_b78dd77a | atac_small | sinkhorn | c1 | 300 | 102 | tier12_v3/A1_sinkhorn | ed161f36c0f3b471dd07ff780cef9bd7e7285f7ac5f6f53b47ba613d0f4eecb9 |
| 7 | A1_immune_none_l0_c0_s101_78c7bb6a | immune | none | c0 | 0 | 101 | tier12_v2/A1 | c986a8c74402fd82ab67835daa3117011c0f550f0fb2c197d9958050f9fe9532 |
| 8 | A1_immune_discriminator_l100_c1_s102_32d8eb57 | immune | discriminator | c1 | 100 | 102 | tier12_v2/A1 | 9870e9dabdd81c05a81ded03f511d8147a13b8a7cfd89f8758aed5d7be8d3ea0 |
| 9 | A1_immune_reference_l30_c1_s100_31992831 | immune | reference | c1 | 30 | 100 | tier12_v2/A1 | bf34238d60dfbfb983125d79cd43bf794f819ba11da240a75e0f78af2e542622 |
| 10 | A1_immune_pooled_l10_c0_s101_75d8d59e | immune | pooled | c0 | 10 | 101 | tier12_v2/A1 | 81e686cbdd084abbdd9af4df6dc153b8afee4345d3be345473a94576ba0b0381 |
| 11 | A1_immune_barycenter_l1_c0_s102_c23fbdae | immune | barycenter | c0 | 1 | 102 | tier12_v2/A1 | 710b2455d1400b05656c3d809cb283baeae652454e1c7916a0f45463b01ee561 |
| 12 | A1_immune_mmd_l1000_c1_s100_d437ee0d | immune | mmd | c1 | 1000 | 100 | tier12_v2/A1 | 458d47ab599089c319986f3cee97649f895a1c1147c3031070f1eb97cb4235ff |
| 13 | A1_immune_sinkhorn_l3_c1_s101_6c0fba12 | immune | sinkhorn | c1 | 3 | 101 | tier12_v3/A1_sinkhorn | e2b7c16ed7a8c45c39a5066ac2c2628203c405fb15610be4dd8926184ecc28a2 |
| 14 | A1_sim1_none_l0_c1_s102_2ec5aa2e | sim1 | none | c1 | 0 | 102 | tier12_v2/A1 | 4a3450cfd97f8275634e7fd39dced51ad4952bba50b34ca1846058c66fc8e214 |
| 15 | A1_sim1_discriminator_l100_c0_s100_36fb5e2c | sim1 | discriminator | c0 | 100 | 100 | tier12_v2/A1 | 247049a15176fe68bd1b4ba5b99ffd368ad34669b7bf43b19ab08969321b9d2b |
| 16 | A1_sim1_reference_l3_c0_s101_43834004 | sim1 | reference | c0 | 3 | 101 | tier12_v2/A1 | 9dab75c22a77f34410165196ef774b5c1830651839e18cc8f495cc64228b0995 |
| 17 | A1_sim1_pooled_l10_c0_s102_855bf21b | sim1 | pooled | c0 | 10 | 102 | tier12_v2/A1 | d9730b81768958408fcb39f23d4a27139c3c89474b24f94e4dc9aa9eb85a9883 |
| 18 | A1_sim1_barycenter_l300_c1_s100_f51f28be | sim1 | barycenter | c1 | 300 | 100 | tier12_v2/A1 | 58e2e3ec0d63e69e4c0fe7499b144995a06cdc0d733369bf44da4c7f395f7b7e |
| 19 | A1_sim1_mmd_l30_c1_s101_39564dc6 | sim1 | mmd | c1 | 30 | 101 | tier12_v2/A1 | 734d386686b4988cbbf93a3bc2ae52f7d845a396351993c582a295a49fa6b5d9 |
| 20 | A1_sim1_sinkhorn_l30_c0_s100_ebcbe68a | sim1 | sinkhorn | c0 | 30 | 100 | tier12_v3/A1_sinkhorn | 7dc9097d7434221f335b90bd9a17369c91d7c05ce0a44db6bef6dad71d638b2e |

Non-Sinkhorn rows come from tier12_v2/A1 (commit c88cce3 = prereg-tier12-v2, scored by its scorer). That scorer differs from the current one by two guard checks only (`git diff c88cce3 HEAD -- scripts/score_scib_native.py`: an input-order check and a relaxed kBET guard; no metric code), and SI-54 / docs/EQUIVALENT_COMMITS.json declare scores of the two commits identical for non-Sinkhorn arms (7 stored latents rescored by both: identical in all 32 columns). All 17 rows carry the same package versions, `cpu_simd` avx2 and empty `numba_cpu_name` as the Sinkhorn rows (searched, 17 of 17).

**Big datasets are not covered.** I looked for scored non-X1 latents of immune_hum_mou and atac_large in tier12_v2, tier12_v3, jhpce_gate_v3 and experiments/: there are none (the gate latents `Q_*`, `S_*` are immune; the fillers and every other fit of those two datasets are X1/X13, blinded). I could not list the rest of the data root. The check therefore does not test the 20-25 GB, 85,000-98,000-cell case, where kNN-graph differences could be larger. Recommended second check, once the pilot's first fits of those two datasets exist: score 2 of each on both hosts (about 45 minutes each locally) and run the same comparison.

## 6. Commands

Scoring commit: **af67f3a7b6e4f787222a61fc3c85bd377370c15b** (branch `score5`; the scorer, host_info.py, run_stage.py and manifest are unchanged since 10a1ca5, the commit that scored the A1 Sinkhorn rows: `git diff --stat 10a1ca5 -- scripts src configs docs manifests` is empty; blob ids score_scib_native.py 800d78ee, host_info.py e47b195c, run_stage.py 5fe6fb35). Every cluster step needs the user's go (SI-29); scratch is purged by age, so fetch results at once.

**1. Workstation: stage the 20 latents and the repository** (repo root, branch score5)
```
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
S=$HOME/scratch/score5_check; L=cluster/jhpce/score_check/check_latents.tsv
mkdir -p "$S/latents"
tail -n +2 "$L" | while IFS=$'\t' read -r tag sha task arm cond lam seed src ref; do
  cp "$src" "$S/latents/$tag.npz"; printf '%s  %s.npz\n' "$sha" "$tag"; done > "$S/latents.sha256"
(cd "$S/latents" && sha256sum -c ../latents.sha256)          # must print 20 lines, all OK
git bundle create "$S/wcd.bundle" score5
```
**2. Copy to JHPCE scratch** (`JH` = the ssh target of JHPCE; not read from any config here)
```
JH=<ssh target of JHPCE>; SC=/fastscratch/myscratch/kreid/wcd/score_check
ssh "$JH" "mkdir -p $SC/latents"
rsync -av "$S/latents/" "$JH:$SC/latents/" && rsync -av "$S/latents.sha256" "$S/wcd.bundle" "$JH:$SC/"
```
**3. On JHPCE: verify, clone at the commit, preflight** (data and clone on scratch, environments stay in `$HOME/envs`)
```
export SC=/fastscratch/myscratch/kreid/wcd/score_check; CODE_SHA=af67f3a7b6e4f787222a61fc3c85bd377370c15b
(cd $SC/latents && sha256sum -c ../latents.sha256)           # 20 x OK again after the copy
git clone -q --no-checkout $SC/wcd.bundle $SC/repo && git -C $SC/repo -c advice.detachedHead=false checkout -q $CODE_SHA
cd $SC/repo && source cluster/jhpce/env.sh && ls $SCORE_ENV/.verified $WCD_R_LIBS/.verified    # both must exist
sbatch -p shared -A jhpce -c 1 --mem=32G -t 01:00:00 -o $SC/preflight.out --wrap \
  'cd $SC/repo && source cluster/jhpce/env.sh && score_r_env && "$SCORE_PY" scripts/fingerprint_prepped.py --dir "$PREPPED_DIR" --compare docs/prepped_fingerprints_scib.json && "$SCORE_PY" cluster/jhpce/env_versions.py --kind score --out $SC/versions_score_jhpce.json'
# preflight.out must show 8 x OK; compare versions_score_jhpce.json with section 3
```
**4. Job 1 = the score-only array, run 1** (the plan writes 3 lists and 3 sbatch files and hashes each latent against the list)
```
cd $SC/repo && "$SCORE_PY" cluster/jhpce/score_plan.py --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv \
  --tags cluster/jhpce/score_check/check_latents.tsv --latents-dir $SC/latents --prepped-dir "$PREPPED_DIR" --out-dir $SC/plan --run run1
for t in atac_small immune sim1; do sbatch $SC/plan/score_$t.sbatch; done           # note the 3 job ids
# repeat for the cluster's own repeatability: same two lines with --run run2 (the plan rewrites the sbatch files)
# optional variant: --run generic, and sbatch --export=ALL,NUMBA_CPU_NAME=generic $SC/plan/score_$t.sbatch
sacct -j <ids> -P --format=JobID,JobName%20,State,Elapsed,MaxRSS,NodeList            # RAM and time actually used
```
Cancel only the job ids you submitted (`scancel <id>`); never `scancel -u`.

**5. Fetch and compare on the workstation** (standard library only)
```
R=$HOME/scratch/score5_check/results; mkdir -p "$R"
for r in run1 run2; do rsync -av "$JH:$SC/plan/$r/scores/" "$R/$r/scores/"; done
python3 cluster/jhpce/compare_scores.py --list "$L" --cand "$R/run1/scores" | tee "$R/compare_run1_vs_workstation.txt"
python3 cluster/jhpce/compare_scores.py --list "$L" --ref "$R/run1/scores" --cand "$R/run2/scores" | tee "$R/compare_run2_vs_run1.txt"
```
Exit 0 = PASS. Both outputs end in `PASS` or `FAIL`; a FAIL lists the metrics, the differing latents and the node and CPU type of each.

## 7. The score-only job (job 1)

| file | what it does |
|---|---|
| `cluster/jhpce/score_array.sh` | one Slurm array task = one latent: refuses unless it is a Slurm array task, the clone's HEAD equals the expected commit and is clean, no path is under `$HOME`, `wcd-score` is verified, and the latent's SHA-256 equals the list's; runs `scripts/score_scib_native.py` with the environment variables `scripts/run_stage.py` gives its scorer (`thread_env(1)`: OMP, MKL, OPENBLAS and NUMBA threads 1, `MKL_THREADING_LAYER=SEQUENTIAL`, `KMP_AFFINITY=disabled`, `PYTHONWARNINGS=ignore`, no GPU, `KBET_SEED=0`, `NUMBA_CPU_NAME` unset); writes `OUT/<run>/scores/<tag>.csv` in the workstation's format (same columns, including latent_sha256, experiment, task, arm, lam, cond, seed from the manifest row); keeps a valid existing file, so a rerun of the same array scores only the missing latents. `scripts/score_list.sh` was not reused: it leaves MKL and OPENBLAS threads unset and writes no per-latent stage files. |
| `cluster/jhpce/score_plan.py` | from a manifest, a tags file, the latents folder and the prepped folder writes `list_<task>.tsv` and `score_<task>.sbatch` per dataset, prints sizes and the time estimate. `--assume task=N ...` prints the estimate only. |
| `cluster/jhpce/compare_scores.py` | the match check (section 2). |

Sizes per array (partition `shared`, account `jhpce`, 1 CPU per task, `--no-requeue`; the plan's table):

| dataset | cells | RAM per task | walltime | basis |
|---|---|---|---|---|
| immune_hum_mou | 97,861 | 32 GB | 3:00:00 | measured peak 19.4 GB (user: 20-25 GB) x 1.65; 3.6x the loaded estimate (3,000 s) |
| atac_large | 84,813 | 32 GB | 3:00:00 | measured peak 24.8 GB x 1.3; 4.2x the loaded estimate |
| lung | 32,472 | 16 GB | 1:00:00 | **not measured**: scaled by cells (about 9.5 GB) x 1.7 |
| immune | 33,506 | 16 GB | 1:00:00 | not measured, as lung |
| sim2 | 19,318 | 8 GB | 0:40:00 | not measured: scaled (about 5.7 GB) x 1.4 |
| pancreas | 16,382 | 8 GB | 0:40:00 | not measured |
| sim1, atac_small | 12,097; 11,270 | 8 GB | 0:30:00 | not measured |

Peak RAM of the two big datasets: experiments/fillers_x1_x13_local_scoring/PREFLIGHT.md PF-18 (a measurement of the scorer at the production setting). Check `sacct` MaxRSS of the first finished tasks of each dataset and resize before the full run.

## 8. Time estimate for the 1,890 pilot latents

Output of `python3 cluster/jhpce/score_plan.py --assume immune_hum_mou=378 atac_large=378 lung=378 sim2=378 pancreas=378` (default 400 CPUs):

```
task            latents   cells RAM GB     wall  array s/latent idle  loaded
immune_hum_mou      378   97861     32 03:00:00    156          1584    3000
atac_large          378   84813     32 03:00:00    135          1373    2600
lung                378   32472     16 01:00:00     51           526     995
sim2                378   19318      8 00:40:00     30           313     592
pancreas            378   16382      8 00:40:00     26           265     502
idle  : 1890 latents, 426 CPU-hours, compute wall at 400 CPUs 1.07 h (queue wait not included)
loaded: 1890 latents, 807 CPU-hours, compute wall at 400 CPUs 2.02 h (queue wait not included)
RAM: 26.5 GB per running task on average (CPU-time weighted); every array at its throttle holds 10576 GB at once. With only F GB free on the nodes the wall is CPU-hours / min(400, F / 26.5)
```

Array sizes 378 each (5 arrays, 1,890 tasks), concurrency caps 156 / 135 / 51 / 30 / 26 = 398 CPUs (shares by CPU-seconds, so the arrays finish together).

| scenario | compute wall |
|---|---|
| idle workstation core speed (docs/scoring_time.csv: 1,584 s for immune_hum_mou, machine otherwise idle) | 1.07 h |
| **planning case**: loaded workstation speed (user: 2,500-3,000 s per big latent with 3-4 at once; 3,000 s used for immune_hum_mou, cells-scaled for the rest) | **2.02 h** |
| JHPCE cores 2x slower than the loaded case (older node types exist) = the same work on 200 CPUs | 4.04 h |
| RAM-limited, loaded speed, F = 10.6 / 5 / 2 / 1 TB held at once | 2.0 / 4.3 / 10.8 / 21.8 h |

Assumptions: (1) 1,890 = 5 datasets x 378, equal split (the pilot manifest is not in the repo; the real counts come from `score_plan.py`); (2) one CPU per latent, time proportional to cells; idle rate from the one measured idle run, loaded rate from the user's figure; (3) all 400 CPUs usable at once, no other job of the account, no queue wait, no retry; (4) the RAM figures of section 7, measured only for the two big datasets; (5) JHPCE cores as fast as the workstation's Ryzen 5 3600 (CPU types seen in the 2026-10-03 gate: EPYC 9555 and an old Opteron on `shared` nodes; EPYC 7443P and Xeon Gold 5317 on GPU nodes); (6) the latents of all datasets are already on scratch (fits write them there). Not included: transfer of scores back (about 2 KB per file), the check and any resize after the first tasks. Memory is the likely limit, not CPUs: the two big arrays alone ask for 9.3 TB at their caps, so how much RAM the `shared` nodes can give one user at once decides between 2 h and 11 h.

## 9. Not checked (could not be read or run from here)

- Anything on JHPCE (no ssh, by instruction): the state of `$HOME/envs/wcd-score`, the `shared` partition's time limit, memory-per-CPU cap (a cap below 32 GB per CPU would make Slurm allot more CPUs per task, cutting the 400-task concurrency), node RAM and CPU types, array-size limit.
- The job was run only with a stub scorer (below); scib, R and kBET were never run in `score_array.sh` here.
- Existing test suites and `scripts/check_tests.py` (section 10).

## 10. Tests

New: `tests/jhpce/test_score_check.py` (6 tests, standard library only, `unittest` classes that pytest also collects):
- `CompareScores` (2): PASS despite different provenance and timing, also through `--ref`; FAIL cases (one metric differing by 5e-10, a NaN against a value, other versions, other latent hash) with the printed per-metric count and largest difference and the CPU-type table; six exit-2 cases (missing file, two rows, tag mismatch, extra column, non-numeric value, empty list).
- `ScorePlan` (2): lists, meta equal to the manifest row, sbatch headers, throttle; known answer for 5 x 378 latents at 400 CPUs (426 and 807 CPU-hours, 26.5 GB, 10,576 GB); five refusals (wrong hash, missing latent, tag not in manifest, duplicate tag, empty tags file) and the missing-arguments exit.
- `ScoreArrayTask` (2): the real `env.sh` and `score_array.sh` in a throwaway git tree with a stub scorer: the stub receives the workstation's environment variables and the list's meta, the score file is written, a rerun skips it; twelve refusals (wrong commit, dirty checkout, not an array task, env not verified, job path under `$HOME`, no such list line, bad tag, latent missing, latent changed, meta without SHA-256, scorer fails, scorer writes another tag), each leaving no score file. Every exit path of the script is covered.

Commands and results (my shell has only the system python 3.12 without numpy, pandas or pytest, and may not read the conda envs):
- `python3 tests/jhpce/test_score_check.py -v` -> `Ran 6 tests in 0.65s ... OK`.
- Mutation check (three edits, each undone afterwards, final run OK again): dropping `MKL_NUM_THREADS=1` from the job, adding a 1e-6 tolerance to the comparison, and deleting the commit check each made the intended test fail.
- Line trace of the two Python scripts (stdlib `trace`): every line ran except the two `if __name__ == "__main__"` entry lines.
- **Not run**: `python -m pytest -q tests/jhpce` (existing tests) and `python scripts/check_tests.py` (needs pytest and pytest-cov). No existing file was changed (only `cluster/README.md` gained two table rows), so the existing suites' inputs are unchanged. Run both in the wcd env before relying on this: `python -m pytest -q tests/jhpce/test_score_check.py` and `python scripts/check_tests.py --base 10a1ca5 --src cluster/jhpce`.
- `compare_scores.py` on real workstation score files (the 3 Sinkhorn check rows, file against itself): `3 latents, 19 metric columns ... PASS`, so the real file format and the 19-column count of section 2 are confirmed.
- Plan dry run on the 3 Sinkhorn check latents (readable here): the plan accepted the SHA-256 from the TSV for all 3, and each list line's meta equals the identity columns of the workstation's own score row (3 of 3).

## 11. Files and commits

Added: `cluster/jhpce/score_array.sh`, `cluster/jhpce/score_plan.py`, `cluster/jhpce/compare_scores.py`, `cluster/jhpce/score_check/check_latents.tsv`, `tests/jhpce/test_score_check.py`, `SCORE5_BUILD.md`. Changed: `cluster/README.md` (two table rows). No scorer, source, config or manifest file was touched. Nothing was submitted, nothing was pushed, no cluster host was contacted.

Commits on branch `score5`: code, check list and tests `af67f3a7b6e4f787222a61fc3c85bd377370c15b`; this document in the commit after it (the document cannot hold its own hash; `git log --oneline -n 2` shows both).
