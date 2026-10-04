#!/usr/bin/env bash
# A1 latents at two commits must be bit-identical: scripts/fit_paper_config.py of OLD vs NEW worktree on the same
# 14 manifest rows (A1, atac_small, seed 100, lambda=0 + every arm at lambda 10, both decoders; rows taken from the OLD
# worktree's tagged manifest), max_epochs = 1, CPU only, 1 thread. Writes OUT/a1_fit_identity.csv (z arrays compared
# with numpy.array_equal; config metadata is allowed to differ).
# Usage: a1_fit_identity.sh OLD_WORKTREE NEW_WORKTREE OUT_DIR   (env: PREPPED_DIR, FIT_PY)
set -uo pipefail
OLD=$1; NEW=$2; OUT=$3; mkdir -p $OUT
P=${PREPPED_DIR:?}; E=${FIT_PY:?}
python3 - "$OLD/manifests/paper_manifest_stock_pilot_u5_b10.tsv" "$OUT" <<'PY'
import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
head = open(src).readline()
m = pd.read_csv(src, sep="\t", comment="#", dtype=str)
a = m[(m.experiment == "A1") & (m.task == "atac_small") & (m.seed == "100")]
sel = a[(a.arm == "none") | (a.lam == "10")].copy()
assert sorted(sel.arm.unique()) == ["barycenter", "discriminator", "mmd", "none", "pooled", "reference", "sinkhorn"] and len(sel) == 14
sel["max_epochs"] = "1"
with open(f"{out}/manifest.tsv", "w") as f:
    f.write(head); sel.to_csv(f, sep="\t", index=False)
open(f"{out}/tags.txt", "w").write("\n".join(sel.tag) + "\n")
PY
for v in old new; do
  R=$([ $v = old ] && echo $OLD || echo $NEW); mkdir -p $OUT/out_$v
  while read -r t; do
    ( cd $R && MANIFEST=$OUT/manifest.tsv PREPPED_DIR=$P OUT_DIR=$OUT/out_$v WCD_SRC=$R/src TAG=$t CUDA_VISIBLE_DEVICES= \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled PYTHONWARNINGS=ignore \
        nice -n 19 $E scripts/fit_paper_config.py > $OUT/out_$v/$t.log 2>&1 ); echo "$v $t exit $?"
  done < $OUT/tags.txt
done
$E - "$OUT" <<'PY'
import sys, os, csv, numpy as np
out = sys.argv[1]; rows = []
for t in open(f"{out}/tags.txt").read().split():
    p = [f"{out}/out_{v}/latents/{t}.npz" for v in ("old", "new")]
    if not all(os.path.exists(x) for x in p):
        rows.append(dict(tag=t, identical="missing", max_abs_diff="")); continue
    za, zb = (np.load(x, allow_pickle=False)["z"] for x in p)
    same = za.shape == zb.shape and np.array_equal(za, zb)
    rows.append(dict(tag=t, identical=str(same), max_abs_diff="" if za.shape != zb.shape else repr(float(np.max(np.abs(za - zb))))))
with open(f"{out}/a1_fit_identity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("IDENTITY", sum(r["identical"] == "True" for r in rows), "of", len(rows), "identical")
PY
