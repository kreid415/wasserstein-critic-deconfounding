#!/bin/bash
# Build this project's two JHPCE conda envs on fastscratch as replicas of the local envs (2026-10-02).
#
#   wcd-fit    replica of local 'scvi-api' (fits, tests/scvi). Same conda layer
#              (cluster/envspec/scvi-api.explicit.txt: python 3.11.15, pip, setuptools, wheel builds), same
#              PyPI pins (cluster/envspec/fit-pip.txt, generated from the local env), EXCEPT torch: local
#              torch 2.13.0+cu130 needs a CUDA 13 driver; JHPCE GPU nodes run driver 555 (CUDA 12.5), so the
#              same torch release is installed as its CUDA 12.6 build (torch==2.13.0+cu126, the lowest CUDA 12
#              build of 2.13.0 on download.pytorch.org; it relies on CUDA minor-version compatibility on
#              driver 555, which the GPU job verifies with a real fit, not assumed here).
#   wcd-score  replica of local 'wcd-kbet' (prep_scib_task.py and scIB-native scoring). Same conda layer
#              (cluster/envspec/wcd-kbet.explicit.txt: 434 packages incl. R 4.5.2, MKL BLAS), then the dists
#              that were pip-installed over it locally, same versions, --no-deps --force-reinstall (pip files
#              are the live ones locally, checked file-by-file against RECORD), except igraph: locally the conda
#              python-igraph 0.11.9 files are live over a stale pip igraph 0.11.8 dist-info, so igraph is not
#              pip-installed (runtime igraph 0.11.9 on both hosts; pip list differs only in that stale record).
#              Then R kBET 0.99.6 from theislab/kBET commit afc5f431 into $WCD_R_LIBS; its namespace
#              fingerprint must equal the local Rlib_kbet copy (3b541efa...).
#
# Fail-loud verification: `conda list --explicit --md5` must equal the local spec line for line; `pip list
# --format=freeze` must equal the local freeze except the declared differences; `pip check` must equal the
# local result (wcd-fit clean; wcd-score: only scib's pandas>=2 metadata pin, as locally).
# Idempotent: an env carrying .verified is reused only if cluster/jhpce/env_intact.py finds nothing missing beyond the
# gaps recorded at build time; otherwise it is deleted and rebuilt. Purge safety (2026-10-04): /fastscratch deletes files
# by modification time after 30 days and conda keeps the package's file dates, so on 2026-10-03 the purge removed most
# of both envs' standard library (built 2026-10-02) while .verified survived. Each build therefore uses a fresh conda
# package cache and, after verification, re-dates every file of the env (and the R library), so the purge clock starts
# at the build; .verified records the date until which the env is safe.
# Usage (repo root, inside a 'shared' batch job): bash cluster/jhpce/build_envs.sh OUT_DIR [fit|score|all]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
source cluster/jhpce/env.sh
OUT=$(mkdir -p "${1:?usage: build_envs.sh OUT_DIR [fit|score|all]}" && cd "$1" && pwd)
WHAT=${2:-all}
SPEC=cluster/envspec
module load conda/3-24.3.0
conda --version
export CONDA_PKGS_DIRS="$SCRATCH/conda/pkgs_build_${SLURM_JOB_ID:-$$}"   # fresh cache: purged unpacked packages are never linked
mkdir -p "$CONDA_PKGS_DIRS"
intact() { python3 "$REPO/cluster/jhpce/env_intact.py" "$1"; }        # $1 env prefix
redate() { local d; for d in "$@"; do [ ! -e "$d" ] || find "$d" -xdev -exec touch -h -c {} +; done; }
mark_verified() {  # $1 env prefix: record benign layering gaps, then the build date and purge horizon
  python3 "$REPO/cluster/jhpce/env_intact.py" "$1" --write-baseline
  echo "built $(date -Is); all files re-dated; safe from the 30-day purge until about $(date -I -d '+29 days')" > "$1/.verified"
}

TORCH_FIT="torch==2.13.0+cu126";  TORCH_FIT_INDEX=https://download.pytorch.org/whl/cu126
TORCH_SCORE="torch==2.4.1+cu121"; TORCH_SCORE_INDEX=https://download.pytorch.org/whl/cu121
KBET_SHA=afc5f431bcbefd73267acc066a0f2e4eaa10a355
KBET_TGZ_MD5=4feba535835dd48b545263e015f005b5        # codeload tarball, checked locally 2026-10-02
KBET_FP=3b541efa166047024303705943926906             # kbet_fingerprint.R on the local Rlib_kbet copy

same_explicit() {  # $1 env prefix, $2 reference explicit spec, $3 out file; compares package file name + md5
  conda list -p "$1" --explicit --md5 > "$3"
  if ! diff <(grep '^https' "$2" | sed 's#.*/##' | sort) <(grep '^https' "$3" | sed 's#.*/##' | sort) > "$3.diff"; then
    echo "FATAL: conda layer of $1 differs from $2:"; cat "$3.diff"; exit 1
  fi
  echo "conda layer identical to $2 ($(grep -c '^https' "$3") packages)"
}

build_fit() {
  if [ -f "$FIT_ENV/.verified" ] && intact "$FIT_ENV"; then echo "[fit] reusing verified, complete env $FIT_ENV"; return 0; fi
  if [ -e "$FIT_ENV" ]; then echo "[fit] $FIT_ENV is unverified or incomplete: deleting and rebuilding"; fi
  rm -rf "$FIT_ENV"
  conda create -y -q -p "$FIT_ENV" --file "$SPEC/scvi-api.explicit.txt"
  # phase 1: torch and its own pinned CUDA 12.6 runtime wheels (exact versions come from torch's metadata)
  "$FIT_PY" -m pip install -q --index-url "$TORCH_FIT_INDEX" --extra-index-url https://pypi.org/simple "$TORCH_FIT"
  # phase 2: every other dist exactly as in the local env (no resolver: the local freeze is the lock)
  "$FIT_PY" -m pip install -q --no-deps -r "$SPEC/fit-pip.txt"
  "$FIT_PY" -m pip check | tee "$OUT/wcd-fit_pip_check.txt"
  "$FIT_PY" -m pip list --format=freeze > "$OUT/wcd-fit_pip_freeze.txt"
  "$FIT_PY" cluster/jhpce/compare_freeze.py --local "$SPEC/local_scvi-api_pip_freeze.txt" \
      --remote "$OUT/wcd-fit_pip_freeze.txt" \
      --allow '^(torch|triton|nvidia-.*|cuda-bindings|cuda-pathfinder|cuda-toolkit)$' | tee "$OUT/wcd-fit_freeze_diff.txt"
  same_explicit "$FIT_ENV" "$SPEC/scvi-api.explicit.txt" "$OUT/wcd-fit.explicit.txt"
  (cd "$TMPDIR" && OMP_NUM_THREADS=1 "$FIT_PY" "$REPO/cluster/jhpce/env_versions.py" --kind fit --out "$OUT/versions_fit_buildnode.json")
  redate "$FIT_ENV"
  mark_verified "$FIT_ENV"
}

build_score() {
  if [ -f "$SCORE_ENV/.verified" ] && intact "$SCORE_ENV" && [ -f "$WCD_R_LIBS/.verified" ]; then
    echo "[score] reusing verified, complete env $SCORE_ENV"; return 0
  fi
  if [ -e "$SCORE_ENV" ]; then echo "[score] $SCORE_ENV or its R library is unverified or incomplete: deleting and rebuilding"; fi
  rm -rf "$SCORE_ENV" "$WCD_R_LIBS"
  conda create -y -q -p "$SCORE_ENV" --file "$SPEC/wcd-kbet.explicit.txt"
  "$SCORE_PY" -m pip install -q --no-deps --force-reinstall -r "$SPEC/score-pip.txt"
  "$SCORE_PY" -m pip install -q --no-deps --force-reinstall --index-url "$TORCH_SCORE_INDEX" "$TORCH_SCORE"
  # R kBET, pinned commit, checksum-verified source, into the R_LIBS library (scripts/run_wave.sh layout)
  mkdir -p "$WCD_R_LIBS"
  curl -sL --fail -o "$TMPDIR/kBET-$KBET_SHA.tar.gz" "https://codeload.github.com/theislab/kBET/tar.gz/$KBET_SHA"
  echo "$KBET_TGZ_MD5  $TMPDIR/kBET-$KBET_SHA.tar.gz" | md5sum -c -
  R_HOME="$SCORE_ENV/lib/R" "$SCORE_ENV/bin/R" CMD INSTALL --library="$WCD_R_LIBS" "$TMPDIR/kBET-$KBET_SHA.tar.gz"
  score_r_env
  fp=$(PATH="$SCORE_ENV/bin:$PATH" Rscript cluster/jhpce/kbet_fingerprint.R)
  echo "$fp" | tee "$OUT/kbet_fingerprint.txt"
  case "$fp" in *"md5=$KBET_FP "*) ;; *) echo "FATAL: kBET fingerprint differs from local ($KBET_FP)"; exit 1;; esac
  # scripts/run_wave.sh preflight: the kBET stack must import (else kBET would be NaN for every config)
  (cd "$TMPDIR" && "$SCORE_PY" -c "import anndata2ri, rpy2; from rpy2.robjects.packages import importr; importr('kBET'); print('kBET stack OK')")
  local prc=0
  "$SCORE_PY" -m pip check > "$OUT/wcd-score_pip_check.txt" || prc=$?   # exit 1 expected (as locally); content checked next
  if [ "$prc" -ne 1 ] || [ "$(cat "$OUT/wcd-score_pip_check.txt")" != "scib 1.1.7 has requirement pandas>=2, but you have pandas 1.5.3." ]; then
    echo "FATAL: pip check differs from local wcd-kbet:"; cat "$OUT/wcd-score_pip_check.txt"; exit 1
  fi
  "$SCORE_PY" -m pip list --format=freeze > "$OUT/wcd-score_pip_freeze.txt"
  "$SCORE_PY" cluster/jhpce/compare_freeze.py --local "$SPEC/local_wcd-kbet_pip_freeze.txt" \
      --remote "$OUT/wcd-score_pip_freeze.txt" \
      --allow-name "igraph=local pip list reads a stale pip 0.11.8 dist-info; the live code is conda python-igraph 0.11.9 on both hosts" \
      | tee "$OUT/wcd-score_freeze_diff.txt"
  same_explicit "$SCORE_ENV" "$SPEC/wcd-kbet.explicit.txt" "$OUT/wcd-score.explicit.txt"
  (cd "$TMPDIR" && PATH="$SCORE_ENV/bin:$PATH" OMP_NUM_THREADS=1 "$SCORE_PY" "$REPO/cluster/jhpce/env_versions.py" --kind score --out "$OUT/versions_score_buildnode.json")
  redate "$SCORE_ENV" "$WCD_R_LIBS"
  date -Is > "$WCD_R_LIBS/.verified"
  mark_verified "$SCORE_ENV"
}

case "$WHAT" in
  fit) build_fit ;;
  score) build_score ;;
  all) build_fit; build_score ;;
  *) echo "unknown target $WHAT"; exit 2 ;;
esac
for d in "$FIT_ENV" "$SCORE_ENV" "$WCD_R_LIBS"; do if [ -e "$d" ]; then du -sh "$d"; fi; done
echo "BUILD OK ($WHAT)"
