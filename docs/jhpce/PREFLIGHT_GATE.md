# Pre-submission check: JHPCE cross-host gate queue (2026-10-02)

The experiment-preflight skills are not available to this agent; this is the short check the task asked for instead.
Verdict: **GO** for the 66-fit gate job only (2 serial single-lane fits + the 64-fit queue); no benchmark fits.

| check | evidence | result |
|---|---|---|
| manifest identical to local | `bench_queue_manifest.tsv` from the local calibration bundle (artifact fa640ada), md5 `9cca9c8c38e767ebf13a98ad44bb9e95`; re-asserted with `md5sum -c` on the node before any fit (`cluster/jhpce/gate_gpu.sh`) | pass |
| same queue and order | `queue_order.txt` from the same bundle: 64 unique tags, all in the manifest; `xargs -P 8` in file order, 1 OMP/NUMBA thread per lane, `MKL_THREADING_LAYER=SEQUENTIAL` (the local `run_queue.sh` settings) | pass |
| same input bytes | gate fits read a staged copy of the local `immune__scib.h5ad` (md5 `3888d2563a7243f9854e41de973f3607`, asserted on the node), so G3 compares hosts on identical inputs; the JHPCE-regenerated file is checked separately (G2) | pass |
| same fit code | repo `jhpce-setup` @ 6675f8c; `src/`, `fit_paper_config.py`, `scvi_adversarial_plan.py` identical to 8cda6de; vs the local run's commit fe56044 only a docstring in `barycenter.py` differs. Clean checkout asserted; completion gate requires `git_sha == HEAD`, `git_dirty == False` in every latent | pass |
| pinned GPU | `#SBATCH --gres=gpu:tesa100:1` (A100); on the node: exactly 1 visible GPU and its name contains `A100`, else exit before any fit | pass |
| outputs on fastscratch | latents/models/logs under `/fastscratch/myscratch/kreid/wcd/gate/run_<slurm id>` (asserted `/fastscratch/*` and new); only logs, small reports and the latent tar parts (~105 MB) go to the HOME job workdir for harvest | pass |
| output uniqueness | `fl_unique_outputs` over the 66 manifest rows, template `latents/{tag}.npz`: no collisions | pass |
| discriminator updates | manifest `n_critic` = 1 for every discriminator row (SI-01) | pass |
| fail-loud code | `fl_lint` on `cluster/jhpce/` and `scripts/score_list.sh`: 0 findings | pass |
| completion criterion | 64 queue latents + 2 single-lane latents, `z` shape (33506, 10), finite, provenance (sha, clean, GPU) checked; rc of every queue fit recorded in `queue_times.tsv`; job exits 1 otherwise | defined |
| walltime | estimate ~30 min (tests ~3 min, 2 serial fits ~1 min, queue ~8-15 min, setup wait if any); request 2 h (4x, first run on this host) | set |
