#!/usr/bin/env bash
# Score identity, tag vs branch (SI-52): scripts/score_scib_native.py of OLD and NEW worktrees on the same latents
# (LIST: lines "<task> <latent npz>"), env wcd-kbet with R as in run_wave.sh, KBET_SEED 0, the runner's thread settings
# and META, LANES at a time. OUT/score_identity.csv: per latent, the columns other than scorer_git_sha, scorer_dirty and
# score_seconds that differ between OLD and NEW, and between NEW and the stage's own score row (<stage>/scores/<tag>.csv).
# Usage: score_identity.sh OLD NEW OUT LIST [LANES]   (env: SCORE_PY, R_HOME, R_LIBS, PREPPED_DIR)
set -uo pipefail
OLD=$1; NEW=$2; OUT=$3; LIST=$4; LANES=${5:-3}
mkdir -p "$OUT/old" "$OUT/new"; cp "$LIST" "$OUT/list.txt"
sc() { local v=$1 task=$2 npz=$3 R t meta; R=$([ "$v" = old ] && echo "$OLD" || echo "$NEW"); t=$(basename "$npz" .npz)
  # META and thread settings as run_stage.py gives the scorer (thread_env(1)); META copied from the stage's own score row
  meta=$("$SCORE_PY" -c "import json, sys, pandas as pd; r = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False).iloc[0]
print(json.dumps({k: r[k] for k in ('latent_sha256', 'experiment', 'task', 'arm', 'lam', 'cond', 'seed')}))" "$(dirname "$(dirname "$npz")")/scores/$t.csv")
  ( cd "$R" && PREPPED="$PREPPED_DIR/${task}__scib.h5ad" NPZ="$npz" TAG="$t" META="$meta" OUT_CSV="$OUT/$v/$t.csv" KBET_SEED=0 \
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
      MKL_THREADING_LAYER=SEQUENTIAL KMP_AFFINITY=disabled PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 \
      nice -n 19 "$SCORE_PY" scripts/score_scib_native.py > "$OUT/$v/$t.log" 2>&1 ); echo "$v $t $?" >> "$OUT/exit_codes.txt"; }
export -f sc; export OLD NEW OUT SCORE_PY R_HOME R_LIBS PREPPED_DIR; : > "$OUT/exit_codes.txt"
while read -r task npz; do echo "old $task $npz"; echo "new $task $npz"; done < "$LIST" | xargs -P "$LANES" -L 1 bash -c 'sc "$0" "$1" "$2"'
"$SCORE_PY" - "$OUT" <<'PY'
import csv, os, sys
import pandas as pd
out = sys.argv[1]; skip = {"scorer_git_sha", "scorer_dirty", "score_seconds"}; rows = []
for line in open(f"{out}/list.txt"):
    t = os.path.basename(line.split()[1])[:-4]
    p = {v: f"{out}/{v}/{t}.csv" for v in ("old", "new")}
    p["prod"] = os.path.join(os.path.dirname(os.path.dirname(line.split()[1])), "scores", f"{t}.csv")
    if not all(os.path.exists(x) for x in p.values()):
        rows.append(dict(tag=t, n_columns="", differing="missing", differing_vs_production="missing")); continue
    a, b, c = (pd.read_csv(p[v], keep_default_na=False, dtype=str).iloc[0] for v in ("old", "new", "prod"))
    cols = sorted((set(a.index) | set(b.index) | set(c.index)) - skip)
    rows.append(dict(tag=t, n_columns=len(cols), differing=";".join(k for k in cols if a.get(k) != b.get(k)),
                     differing_vs_production=";".join(k for k in cols if b.get(k) != c.get(k))))
with open(f"{out}/score_identity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("SCORE IDENTITY", sum(r["differing"] == "" for r in rows), "of", len(rows), "latents old = new in every column;",
      sum(r["differing_vs_production"] == "" for r in rows), "new = stage score row")
PY
