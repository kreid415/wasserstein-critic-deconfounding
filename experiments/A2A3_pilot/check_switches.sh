#!/usr/bin/env bash
# A2/A3 switch check (preflight evidence, 2026-10-10). No test exercises adv_input="sample" (A2) or zstd=1 (A3),
# so before the 288 A2/A3 fits this runs scripts/fit_paper_config.py from the tag checkout for 1 epoch on CPU
# (1 thread, deterministic as in experiments/integration_checks_20261004) on atac_small, conditioned decoder, at the
# R4-a1 matched lambda (reference 100, mmd 30):
#   D_ref   A1 default half (mean, zstd 0), reference, seed 100     -- fitted twice (determinism)
#   S_ref   A2 new half (sample), reference, seed 100;  S_ref101 the same with seed 101
#   Z_ref   A3 new half (zstd 1), reference, seed 100
#   D_mmd   A1 default half, mmd, seed 100;  Z_mmd A3 new half, mmd, seed 100 (critic-free code path)
# PASS = D_ref bit-identical across the two runs, every new half differs from its default half, the two seeds
# differ, all latents finite, every fit exit 0. Verdict and switch_check.csv by switch_compare.py; the script's exit
# code is the verdict (0 PASS, 1 FAIL, 2 refused before any fit).
# Usage: check_switches.sh OUT_DIR
set -euo pipefail
OUT=${1:?usage: check_switches.sh OUT_DIR}
[ ! -e "$OUT" ] || { echo "OUT_DIR $OUT exists; the check needs a fresh directory" >&2; exit 2; }
mkdir -p "$OUT"
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
REPO=$D/code/wcd_prereg-tier12-v2; E=/home/kendall/.claude-science/conda/envs/scvi-api/bin/python
MAN=$D/tier12_v2/manifests/paper_manifest_stock_pilot_u5_b10_v3.r4a1.tsv
MAN_SHA=ca28d991767d85f8fb7f25b8335267676846d70e8c7887ea9cb3e87900c33459
[ "$(git -C "$REPO" rev-parse HEAD)" = c88cce3c569e9de47f2d65c6794ca3890637786a ] && [ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "checkout not clean at c88cce3" >&2; exit 2; }
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || { echo "resolved manifest differs from the committed one" >&2; exit 2; }
"$E" - "$MAN" "$OUT" <<'PY' || { echo "row selection failed" >&2; exit 2; }
import sys, json, pandas as pd
src, out = sys.argv[1], sys.argv[2]
head = open(src).readline()
m = pd.read_csv(src, sep="\t", comment="#", dtype=str)
def one(**kw):
    s = m
    for k, v in kw.items():
        s = s[s[k] == v]
    assert len(s) == 1, (kw, len(s))
    return s.iloc[0]
base = dict(task="atac_small", cond="1")
rows = {"D_ref": one(experiment="A1", arm="reference", lam="100", seed="100", **base),
        "S_ref": one(experiment="A2", arm="reference", lam="100", seed="100", **base),
        "S_ref101": one(experiment="A2", arm="reference", lam="100", seed="101", **base),
        "Z_ref": one(experiment="A3", arm="reference", lam="100", seed="100", **base),
        "D_mmd": one(experiment="A1", arm="mmd", lam="30", seed="100", **base),
        "Z_mmd": one(experiment="A3", arm="mmd", lam="30", seed="100", **base)}
for k, (a, b) in {"S_ref": ("D_ref", "adv_input"), "Z_ref": ("D_ref", "zstd"), "Z_mmd": ("D_mmd", "zstd")}.items():
    diff = [c for c in m.columns if c not in ("tag", "experiment") and rows[k][c] != rows[a][c]]
    assert diff == [b], (k, diff)          # the new half differs from its default half in that one setting only
sel = pd.DataFrame(list(rows.values())); sel["max_epochs"] = "1"
with open(f"{out}/manifest.tsv", "w") as f:
    f.write(head); sel.to_csv(f, sep="\t", index=False)
json.dump({k: r["tag"] for k, r in rows.items()}, open(f"{out}/roles.json", "w"), indent=1)
PY
[ -s "$OUT/roles.json" ] || { echo "row selection failed" >&2; exit 2; }
: > "$OUT/fit_rc.tsv"
run() {  # run <out subdir> <tag>; appends "<subdir>\t<tag>\t<exit code>" to fit_rc.tsv (the verdict requires 7 zeros)
  mkdir -p "$OUT/$1"
  ( cd "$REPO" && MANIFEST=$OUT/manifest.tsv PREPPED_DIR=$D/prepped_scib OUT_DIR=$OUT/$1 WCD_SRC=$REPO/src TAG=$2 CUDA_VISIBLE_DEVICES= \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled PYTHONWARNINGS=ignore \
      nice -n 19 "$E" scripts/fit_paper_config.py > "$OUT/$1/$2.log" 2>&1 ) && rc=0 || rc=$?   # keep going: the verdict judges rc
  printf '%s\t%s\t%s\n' "$1" "$2" "$rc" >> "$OUT/fit_rc.tsv"; echo "$1 $2 exit $rc"
}
tag() { "$E" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$OUT/roles.json" "$1"; }
for r in D_ref S_ref S_ref101 Z_ref D_mmd Z_mmd; do run run1 "$(tag $r)" & done
run run2 "$(tag D_ref)" &
wait
# verdict: exit 0 = PASS, 1 = FAIL (any non-zero fit, missing or non-finite latent, or a failed comparison)
"$E" "$(dirname "${BASH_SOURCE[0]}")/switch_compare.py" "$OUT"
