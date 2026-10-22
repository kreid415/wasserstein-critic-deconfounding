#!/bin/bash
# JHPCE score-only array task (CPU, `shared` partition): scores ONE latent with scripts/score_scib_native.py, the way
# scripts/run_stage.py scores a latent on the workstation: the same script and the same environment variables as
# run_stage.launch() (thread_env(1), KBET_SEED 0 = score_scib_native.KBET_SEED_DEFAULT, no GPU; NUMBA_CPU_NAME is
# not set, as on the workstation). scripts/score_list.sh is not reused: it leaves MKL_NUM_THREADS and
# OPENBLAS_NUM_THREADS unset and does not write the stage's per-latent score files.
# Usage (one Slurm array task; cluster/jhpce/score_plan.py writes the sbatch files):
#   bash cluster/jhpce/score_array.sh EXPECTED_SHA LIST OUT_DIR
#   LIST     line $SLURM_ARRAY_TASK_ID = tag <TAB> latent npz <TAB> prepped h5ad <TAB> meta json (score_plan.py)
#   OUT_DIR  scores/<tag>.csv (the workstation's one-row score file) and logs/<tag>.log
# Checks (any failure ends the task non-zero and leaves no score file): Slurm array task; repo HEAD == EXPECTED_SHA and
# clean (the score row records scorer_git_sha and scorer_dirty); no path under $HOME; wcd-score env verified; the
# latent's SHA-256 equals meta.latent_sha256; the score file holds one row of this tag, latent and commit. A valid
# score file is kept, so rerunning the same array scores only the missing latents (another scoring setting needs
# another OUT_DIR).
# Exit: 0 scored or already valid | 3 repository | 8 inputs | 9 environment | 10 scorer failed or wrote no valid file
set -euo pipefail

EXPECTED_SHA=${1:?expected commit sha}
LIST=${2:?list tsv}
OUT=$(mkdir -p "${3:?out dir}" && cd "$3" && pwd -P)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
log() { echo "[$(date -Is)] $*"; }

# ---- environment: scratch-first, scoring env verified ---------------------------------------------------------------
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: not a Slurm array task (SLURM_ARRAY_TASK_ID unset)" >&2; exit 9; }
# shellcheck source=/dev/null
source "$REPO/cluster/jhpce/env.sh"                  # SCRATCH, TMPDIR, caches on fastscratch; SCORE_ENV, SCORE_PY in $HOME/envs
score_r_env                                          # R_HOME, R_LIBS for rpy2 + kBET
[ -f "$SCORE_ENV/.verified" ] || { echo "FATAL: $SCORE_ENV/.verified missing (cluster/jhpce/build_envs.sh)" >&2; exit 9; }
HOME_REAL=$(cd "$HOME" && pwd -P)       # fl: allow FL180 read only: the scratch-first guard below compares paths with $HOME
for p in "$WCD_SCRATCH" "$TMPDIR" "$REPO" "$OUT"; do
  # fl: allow FL180 this guard refuses any job path under $HOME (scratch-first rule)
  case "$(cd "$p" && pwd -P)/" in "$HOME_REAL"/*) echo "FATAL: $p is under \$HOME ($HOME_REAL): scratch-first rule" >&2; exit 9;; esac
done

# ---- repository: the commit the plan was made for, clean --------------------------------------------------------------
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA" ] || { echo "FATAL: repo HEAD != expected $EXPECTED_SHA" >&2; exit 3; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "FATAL: repo $REPO has uncommitted changes" >&2; exit 3; }

# ---- this task's latent -----------------------------------------------------------------------------------------------
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$LINE" ] || { echo "FATAL: $LIST has no line $SLURM_ARRAY_TASK_ID" >&2; exit 8; }
IFS=$'\t' read -r TAG NPZ PRE META <<< "$LINE"
[[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "FATAL: bad tag '$TAG'" >&2; exit 8; }
[ -f "$NPZ" ] && [ -f "$PRE" ] || { echo "FATAL: $TAG: latent $NPZ or prepped file $PRE missing" >&2; exit 8; }
WANT=$("$SCORE_PY" -c 'import json, sys; print(json.loads(sys.argv[1])["latent_sha256"])' "$META") \
  || { echo "FATAL: $TAG: meta has no latent_sha256" >&2; exit 8; }
GOT=$(sha256sum "$NPZ" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo "FATAL: $TAG: latent SHA-256 $GOT != meta $WANT" >&2; exit 8; }

CSV="$OUT/scores/$TAG.csv"
valid() {   # 0 iff CSV holds exactly one row of this tag, this latent, scored at EXPECTED_SHA from a clean checkout
  [ -s "$CSV" ] && "$SCORE_PY" - "$CSV" "$TAG" "$GOT" "$EXPECTED_SHA" <<'PYEOF'
import csv, sys
path, tag, sha, git = sys.argv[1:]
rows = list(csv.DictReader(open(path)))
ok = (len(rows) == 1 and rows[0]["tag"] == tag and rows[0]["latent_sha256"] == sha
      and rows[0]["scorer_git_sha"] == git and rows[0]["scorer_dirty"] == "0")
sys.exit(0 if ok else 1)
PYEOF
}
mkdir -p "$OUT/scores" "$OUT/logs"
if valid; then log "$TAG: valid score file exists, skipped"; exit 0; fi
rm -f "$CSV"

# ---- score ------------------------------------------------------------------------------------------------------------
rc=0
( cd "$OUT" && env PREPPED="$PRE" NPZ="$NPZ" TAG="$TAG" META="$META" OUT_CSV="$CSV" KBET_SEED=0 CUDA_VISIBLE_DEVICES= \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL \
    KMP_AFFINITY=disabled PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    "$SCORE_PY" "$REPO/scripts/score_scib_native.py" ) > "$OUT/logs/$TAG.log" 2>&1 || rc=$?
if [ "$rc" -ne 0 ] || ! valid; then
  rm -f "$CSV"
  echo "FATAL: $TAG: scorer exit $rc or no valid score file (see $OUT/logs/$TAG.log)" >&2
  exit 10
fi
log "$TAG scored on $(hostname)"
