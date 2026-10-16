# Conda environments of this project

Exported 2026-10-16 from `/home/kendall/.claude-science/conda/envs/<name>` on the user's workstation with `micromamba env export --no-builds` (the `prefix:` line was removed). Provenance: repo commit 10a1ca5 (branch final-review-fixes) was the checkout at export time.

| file | what it runs | key versions (live import, checked 2026-10-16) |
|---|---|---|
| `scvi-api.yml` | fits (scVI backbone, scvi-tools = single-cell variational inference toolkit), `tests/prereg`, `tests/stage`, `tests/analysis`, `tests/scvi`, `tests/jhpce`, `scripts/check_tests.py` | python 3.11.15, numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scanpy 1.11.5, anndata 0.12.19, scvi-tools 1.4.2, torch 2.13.0+cu130 |
| `wcd-gpu.yml` | CUDA training of `src/wcd_vae`, `tests/wcd_vae` (run with `PYTHONPATH=src`; the package is not pip-installed) | numpy 1.26.4, pandas 1.5.3, scipy 1.14.1, scanpy 1.11.5, anndata 0.11.4, scib 1.1.7, scvi-tools 0.14.6, torch 2.4.1+cu121 |
| `wcd-kbet.yml` | scIB scoring (scib = single-cell integration benchmark metrics), `tests/x13`, `tests/scoring`; has R 4.5.2 + rpy2 3.6.7 + anndata2ri 1.3.2 for kBET (k-nearest-neighbour batch effect test) | same Python stack as wcd-gpu, plus r-base 4.5.2 |
| `testgate.yml` | only pytest 9.1.1 + pytest-cov 7.1.0 for the fewest-tests gate | python 3.11.17 |
| `cloud-cpu.yml` | the Claude Code cloud VM (CPU only); built by `.claude/cloud-setup.sh` | same pins as scvi-api, torch 2.13.0+cpu, NVIDIA/Triton wheels removed |

## Which specification is canonical

1. **Byte-exact (canonical for rebuilding the fit and scoring environments):** `cluster/envspec/scvi-api.explicit.txt` + `cluster/envspec/fit-pip.txt` (fit) and `cluster/envspec/wcd-kbet.explicit.txt` + `cluster/envspec/score-pip.txt` (scoring). `cluster/jhpce/build_envs.sh` builds `wcd-fit` / `wcd-score` from them on JHPCE (Johns Hopkins cluster) and refuses to continue if the conda layer differs (`same_explicit`). The only declared difference to the local scvi-api env is torch 2.13.0+cu126 instead of +cu130 (docs/jhpce/GATE_JHPCE.md).
2. **Readable exports (this folder):** same versions, easier to read, not hash-checked. Most packages of `scvi-api` and the Python stack of the other two live in the `pip:` section, so the `.yml` files record versions, not conda builds.
3. **Not canonical:** the repo-root `environment.yaml`, `conda-lock.yml` and `Dockerfile` (last changed 2026-03-21, before the revision work). They describe an unpinned generic image (name `myenv`, pyro4, scib/scanorama/bbknn from pip, CUDA 12.8 base) and do not match any environment above. Do not use them to reproduce pre-registered results.
4. `wcd-gpu` has no byte-exact spec in the repo; `wcd-gpu.yml` is its only record.

Quirks to know: (a) in `wcd-gpu`/`wcd-kbet` the conda layer lists scanpy 1.9.8 but a later pip install replaced it (live scanpy 1.11.5); the pip layer wins. (b) `pip check` reports `scib 1.1.7 requires pandas>=2` in wcd-kbet; this is expected (pandas 1.5.3 is pinned on purpose, scib calls `pd.value_counts`). (c) Import of torch/MKL libraries in these environments fails with `OMP: Error #179 pthread_setaffinity_np` inside sandboxes unless `KMP_AFFINITY=disabled` is set.

## kBET R library (`Rlib_kbet`)

- Location on the user's machine: `/home/kendall/experiment_data/wasserstein-critic-deconfounding/Rlib_kbet` (248 KB, package `kBET` version 0.99.6). On JHPCE: `$HOME/envs/Rlib_kbet` (`WCD_R_LIBS` in `cluster/jhpce/env.sh`).
- Source: GitHub `theislab/kBET`, commit `afc5f431bcbefd73267acc066a0f2e4eaa10a355`, tarball `https://codeload.github.com/theislab/kBET/tar.gz/<commit>`, md5 `4feba535835dd48b545263e015f005b5`.
- Rebuild (recipe in `cluster/jhpce/build_envs.sh`, function `build_score`): create the scoring env from `cluster/envspec/wcd-kbet.explicit.txt` (it already holds r-fnn, r-rcolorbrewer, r-ggplot2, r-mass, r-cluster), download the tarball, check the md5, then
  `R_HOME=<env>/lib/R <env>/bin/R CMD INSTALL --library=<Rlib_kbet dir> kBET-<commit>.tar.gz`.
- Check: `R_LIBS=<Rlib_kbet dir> Rscript cluster/jhpce/kbet_fingerprint.R` must print `md5=3b541efa166047024303705943926906`.
- Use: scoring runs pass `--r-home <env>/lib/R --r-libs <Rlib_kbet dir>`; tests need `R_HOME`, `R_LIBS`, `PREPPED_DIR`. kBET needs R, so it cannot run on the cloud VM.
