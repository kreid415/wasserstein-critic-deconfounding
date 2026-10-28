#!/bin/bash
# JHPCE score-only array task with the LISI subsample seed fixed (CONSTRAINTS.md SI-115; docs/SCORER_EQUIVALENCE.json).
# scib 1.1.7 scores graph iLISI/cLISI on a random half of the cells, seeded from the CPU's RDRAND. JHPCE's RDRAND is random; the
# workstation's always returns 0xffffffff, so every workstation LISI is the score of one fixed half. This wrapper loads
# fixed_random_device.so (experiments/lisi_jhpce_investigation_20261028/, branch lisi-jhpce-investigation@ed280a2) so that JHPCE
# draws the same half. It then runs the repository's own score_array.sh unchanged and appends the three provenance columns the
# rule check reads (score_site, lisi_random_device, lisi_preload_sha256) to the score file.
# Usage (one Slurm array task): SCORER_REPO=<clean checkout of the scorer commit that holds cluster/jhpce/score_array.sh> \
#   FIXED_SEED_LIB=<path of fixed_random_device.so> bash cluster/jhpce/score_array_fixed_seed.sh EXPECTED_SHA LIST OUT_DIR
# Concurrent array tasks that start on a cold numba cache (NUMBA_CACHE_DIR, shared on scratch) can collide writing a cache index
# (FileExistsError ... .nbi, seen 2026-10-28); the cache changes compile time only, never a value, so such a task is run again, at most
# twice more, after a random pause (its log is kept as <tag>.log.race<N>). Any other failure ends the task.
# Exit codes are those of score_array.sh (0 scored or already valid | 3 repository | 8 inputs | 9 environment | 10 scorer failed);
# 9 also when the preload library is missing, has another SHA-256 or is not loaded into a process started from this shell.
set -euo pipefail
EXPECTED_SHA=${1:?expected commit sha}; LIST=${2:?list tsv}; OUT=${3:?out dir}
SCORER_REPO=${SCORER_REPO:?checkout of the scorer commit}; LIB=${FIXED_SEED_LIB:?path of fixed_random_device.so}
LIB_SHA256=fda4d45779c0971626018a0e68abb405b1a1b8f5810841a6bef4da6f5d9feea3   # lisi_jhpce_investigation_20261028.md section 2
VALUE=0xFFFFFFFF                                                               # what RDRAND/RDSEED return on the workstation
# shellcheck source=/dev/null
source "$SCORER_REPO/cluster/jhpce/env.sh"          # SCORE_PY
[ "$(sha256sum "$LIB" | cut -d' ' -f1)" = "$LIB_SHA256" ] || { echo "FATAL: $LIB is not the pinned preload library ($LIB_SHA256)" >&2; exit 9; }
export LD_PRELOAD="$LIB" LISI_RANDOM_DEVICE_VALUE="$VALUE"
"$SCORE_PY" -c 'import sys; sys.exit(0 if "fixed_random_device.so" in open("/proc/self/maps").read() else 1)' \
  || { echo "FATAL: the preload library is not loaded into a child process" >&2; exit 9; }
TAG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST" | cut -f1)
for attempt in 1 2 3; do
  rc=0; bash "$SCORER_REPO/cluster/jhpce/score_array.sh" "$EXPECTED_SHA" "$LIST" "$OUT" || rc=$?
  [ "$rc" -eq 10 ] && [ "$attempt" -lt 3 ] && grep -q 'FileExistsError.*\.nbi' "$OUT/logs/$TAG.log" || break
  echo "numba cache index race in $TAG (attempt $attempt): running it again" >&2
  mv "$OUT/logs/$TAG.log" "$OUT/logs/$TAG.log.race$attempt"; sleep $((RANDOM % ${NUMBA_RACE_PAUSE_S:-25} + 1))
done
[ "$rc" -eq 0 ] || exit "$rc"
"$SCORE_PY" - "$OUT/scores/$TAG.csv" "$LIB_SHA256" "$VALUE" <<'PYEOF'
import csv, os, sys
path, sha, value = sys.argv[1:]
with open(path, newline="") as f:
    rd = csv.DictReader(f)
    names, rows = list(rd.fieldnames), list(rd)
assert len(rows) == 1, f"{path}: {len(rows)} rows"
add = {"score_site": "jhpce", "lisi_random_device": value, "lisi_preload_sha256": sha}
if "score_site" in names:       # a valid file kept by score_array.sh from an earlier run of this wrapper
    assert all(rows[0].get(k) == v for k, v in add.items()), f"{path}: provenance columns differ from this run"
else:
    names += list(add)
    rows[0].update(add)
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)
PYEOF
