#!/usr/bin/env bash
# Fit identity, tag prereg-tier12-v3 vs the follow-up build branch (SI-93..SI-98): scripts/fit_paper_config.py of OLD
# (tag checkout) and NEW (worktree of the candidate commit) on the same X1 and X13 rows (ROWS = select_x1_rows.py output,
# max_epochs 1), CPU only, 1 thread per fit, LANES fits at a time. Writes OUT/fit_identity.csv: per row both exit codes and
# whether the latents z are equal (numpy.array_equal). Copy of experiments/integration_checks_20261013/fit_identity.sh with
# the summary changed: the tag already holds the Sinkhorn fix, so EVERY row must be identical (no Sinkhorn exception).
# NOT RUN by the follow-up build (the CPU was in use by the Sinkhorn redo); run it before the new code tag.
# Usage: fit_identity.sh OLD_WORKTREE NEW_WORKTREE OUT_DIR ROWS_TSV [LANES]   (env: PREPPED_DIR, FIT_PY)
set -uo pipefail
OLD=$1; NEW=$2; OUT=$3; ROWS=$4; LANES=${5:-4}
P=${PREPPED_DIR:?}; E=${FIT_PY:?}
mkdir -p "$OUT/out_old" "$OUT/out_new"
cp "$ROWS" "$OUT/manifest.tsv"
"$E" -c "import pandas as pd, sys; m = pd.read_csv(sys.argv[1], sep='\t', comment='#', dtype=str); print('\n'.join(m.tag))" \
    "$OUT/manifest.tsv" > "$OUT/tags.txt"
fit() {   # fit <old|new> <tag>
  local v=$1 t=$2 R; R=$([ "$v" = old ] && echo "$OLD" || echo "$NEW")
  ( cd "$R" && MANIFEST="$OUT/manifest.tsv" PREPPED_DIR="$P" OUT_DIR="$OUT/out_$v" WCD_SRC="$R/src" TAG="$t" \
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled \
      PYTHONWARNINGS=ignore nice -n 19 "$E" scripts/fit_paper_config.py > "$OUT/out_$v/$t.log" 2>&1 )
  echo "$v $t $?" >> "$OUT/exit_codes.txt"
}
export -f fit; export OLD NEW OUT P E
: > "$OUT/exit_codes.txt"
for t in $(cat "$OUT/tags.txt"); do echo "old $t"; echo "new $t"; done | xargs -P "$LANES" -L 1 bash -c 'fit "$0" "$1"'
"$E" - "$OUT" <<'PY'
import csv, os, sys
import numpy as np
out = sys.argv[1]
rc = {(v, t): int(c) for v, t, c in (l.split() for l in open(f"{out}/exit_codes.txt"))}
rows = []
for t in open(f"{out}/tags.txt").read().split():
    p = {v: f"{out}/out_{v}/latents/{t}.npz" for v in ("old", "new")}
    r = dict(tag=t, rc_old=rc.get(("old", t)), rc_new=rc.get(("new", t)), identical="", max_abs_diff="")
    if all(os.path.exists(x) for x in p.values()):
        za, zb = (np.load(p[v], allow_pickle=False)["z"] for v in ("old", "new"))
        r["identical"] = str(za.shape == zb.shape and np.array_equal(za, zb))
        r["max_abs_diff"] = "" if za.shape != zb.shape else repr(float(np.max(np.abs(za - zb))))
    else:
        r["identical"] = "missing"
    rows.append(r)
with open(f"{out}/fit_identity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
bad = [r for r in rows if r["identical"] != "True" or r["rc_old"] != 0 or r["rc_new"] != 0]
print("IDENTITY", len(rows) - len(bad), "of", len(rows), "rows identical with exit code 0 at both commits;",
      "not identical:", [r["tag"] for r in bad])
sys.exit(1 if bad else 0)
PY
