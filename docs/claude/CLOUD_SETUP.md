# Claude Code cloud environment: how to configure it

The cloud VM holds only a git clone and what the setup step installs. Assume it has no GPU, none of the data in `docs/claude/DATA_LOCATIONS.md`, no JHPCE (Johns Hopkins cluster) access and no Claude Science tools until checked (`nvidia-smi`, `ssh`); none of these was tested from a cloud VM. It can run the CPU test tiers, edit code and documents, and use git.

## 1. Environment settings (claude.ai/code, environment configuration)

1. **Setup (automatic):** `.claude/settings.json` registers a SessionStart hook (a command Claude Code runs when a session starts or resumes) that runs `.claude/cloud-setup.sh` only when `CLAUDE_CODE_REMOTE=true`, which cloud sessions set; local sessions skip it. Nothing needs pasting. Optional: the environment's own setup-script box is cached between sessions but runs before the session, with no documented repository path, so leave it empty unless start-up time matters. The script installs micromamba 2.9.0-0 from GitHub, builds the conda environment `wcd-cpu` from `envs/cloud-cpu.yml` (python 3.11.15, numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scanpy 1.11.5, scvi-tools 1.4.2, torch 2.13.0+cpu, pytest 9.1.1, pytest-cov 7.1.0), appends the env `bin` directory to `~/.bashrc` PATH, and prints the versions. It needs no secrets and is idempotent: a second run only checks a hash of `envs/cloud-cpu.yml` (3 s) and rebuilds only if that file changed. It was tested on a Linux x86-64 machine with an empty prefix: first build 70-100 s (two runs: 70 s and 101 s), 2.2 GB env + 117 MB package cache, second run 3 s. It also sets the repo-local git identity (user.name kreid415, noreply email).
2. **Network access:** choose **Custom** (not None, and not Trusted unless it already lists these) and allow these domains. Every one was reached by the setup test or is named in the repo; domains marked * were found necessary during the test.
   - `github.com`, `codeload.github.com`, `release-assets.githubusercontent.com`* and `objects.githubusercontent.com` (git, micromamba binary, kBET tarball; the micromamba download redirects to a githubusercontent host)
   - `conda.anaconda.org` (conda-forge packages)
   - `pypi.org`, `files.pythonhosted.org` (pip packages)
   - `download.pytorch.org` and `download-r2.pytorch.org`* (CPU torch wheel; the first host redirects to the second, and without the second pip fails with a 403 tunnel error)
   - Only if you want to rebuild datasets on the VM (not recommended, 14 GB): `figshare.com`, `ndownloader.figshare.com`, `s3-eu-west-1.amazonaws.com` (figshare file host; the exact redirect host was not tested)
   - Not needed: OpenAlex (no repo code calls it), CRAN (R packages come from conda-forge), GEO (accessions are provenance only).
3. **Environment variables / secrets:** none required. `.claude/settings.json` sets the variables the code needs (`KMP_AFFINITY=disabled`, `OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `MKL_THREADING_LAYER=SEQUENTIAL`, `PYTHONPATH=src`); do not add tokens to the repo. If the cloud environment pushes through the Claude GitHub app, no `GITHUB_TOKEN` is needed in the VM. Secret names that could be added later, as names only: `GITHUB_TOKEN` (only if `gh` is used outside the app), `OPENALEX_API_KEY` (only for literature scripts, none in the repo today).
4. **Git identity:** `git config user.name kreid415` and `git config user.email kreid415@users.noreply.github.com` (the setup script sets both, repo-local; use this noreply address only).
5. **Branch:** name `claude-code-migration` in the first prompt (the branch assembled on final-review-fixes at 1ad44cb, two commits after tag prereg-tier12-v3 = 10a1ca5). Notebook and ledger appends happen on one branch only, `final-review-fixes`. The Claude Science sessions have finished and pushed (last commits 857234a, notebook entry NB-20261016-34, and 1ad44cb, the Sinkhorn-redo preflight GO); this branch sits on 1ad44cb. First action of the first Claude Code session: `git checkout final-review-fixes && git pull && git merge origin/claude-code-migration`, push, and work on `final-review-fixes` (or short branches merged back into it). Do not push to `main`; do not move or delete the tags prereg-tier12-v1, -v2, -v3.

## 2. First prompt (copy)

See `docs/claude/FIRST_PROMPT.md` (kept in one place so it cannot drift from CLAUDE.md).

## 3. What the VM can and cannot run

Measured with the setup script's environment (CPU torch), repo commit 10a1ca5, `KMP_AFFINITY=disabled OMP_NUM_THREADS=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES=`:

| command | result |
|---|---|
| `python -m pytest -q tests/prereg tests/stage tests/analysis` | 283 passed, 2 skipped, 4 min 40 s |
| `python -m pytest -q tests/scvi tests/jhpce tests/test_legacy_retired.py tests/test_gate_task_split.py` | 168 passed, 16 skipped, 5 min 46 s |

The 18 skips are opt-in tests that need data or other environments (`PREPPED_DIR`, and for `tests/stage/test_run_stage_e2e.py` the fit and scoring interpreters plus R and kBET). `scripts/check_tests.py --base 973a97d --src scripts -- tests/analysis tests/prereg tests/scvi tests/stage tests/jhpce tests/test_legacy_retired.py` (the gate described in docs/ANALYSIS_SPEC.md, about 14 minutes) was not run in this test; it needs a git base ref present in the clone and pytest-cov, which is installed.

Cannot run on the VM (use the user's machine or JHPCE):

1. **GPU fits and any training** (needs the RTX 3080 workstation, envs `scvi-api`/`wcd-gpu`, or the JHPCE L40S nodes via `cluster/jhpce/`; every JHPCE submission needs the user's explicit go, CONSTRAINTS.md SI-29).
2. **scIB scoring with kBET** (needs R 4.5.2 and `Rlib_kbet`; env `wcd-kbet`, see `envs/README.md`).
3. **Anything that reads prepared data, latents, models or score files** (`tests/x13`, `tests/scoring`, `PREPPED_DIR` tests, analysis of real results). The data are listed in `docs/claude/DATA_LOCATIONS.md`.
4. **`tests/wcd_vae`** (needs the pinned `wcd-gpu` stack: pandas 1.5.3, scib 1.1.7, torch 2.4.1); not part of `cloud-cpu.yml`.

For these, run Claude Code locally in a clone on the workstation (so `/home/kendall/experiment_data/...` is reachable) or use Remote Control to drive a session on that machine. The blinding rule for X1/X13 files applies there too. `.claude/settings.json` denies reads of the blinded folders under `/home/kendall/experiment_data/wasserstein-critic-deconfounding` (`tier12*`, `jhpce_tier12`, `x13_cpu_baselines_*`, `review/identity_*`, `review/out`, `review/verify`, `review_sealed`) and of any `failures.csv` or `attempts/`, but only for built-in file tools and simple `cat`/`head`-style commands; it does not stop `grep -r` or Python `open()`, so the rule itself is what protects the blinding.
