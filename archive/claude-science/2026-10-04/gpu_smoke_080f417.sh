#!/usr/bin/env bash
# Re-review GPU smoke test (local RTX 3080): scripts/fit_paper_config.py of the review commit on CUDA, 1-2 epochs, for
# the code paths that changed since the tag and that the CPU-only suites do not run on a GPU: resolved_plan /
# check_kl_warmup after every fit (A1 arms, both decoders), the X13 guard (scANVI, sysVI), scvi_adv, X15 'complete',
# X3 nested_v1 + realised-composition weights. Placeholder lambdas (matched_*) are replaced by 10 in a scratch copy of
# the rows (smoke only; no result is used). Outputs to OUT; the prepped files are read only.
# Usage: gpu_smoke_080f417.sh REPO OUT
set -uo pipefail
REPO=$(realpath $1); OUT=$(realpath -m $2); mkdir -p $OUT
E=/home/kendall/.claude-science/conda/envs; P=/home/kendall/experiment_data/wasserstein-critic-deconfounding/prepped_scib
$E/scvi-api/bin/python - "$REPO/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv" "$OUT" <<'PY'
import sys, json, pandas as pd
src, out = sys.argv[1], sys.argv[2]
head = open(src).readline()
m = pd.read_csv(src, sep="\t", comment="#", dtype=str, keep_default_na=False)
pick = []
a1 = m[(m.experiment == "A1") & (m.task == "atac_small") & (m.seed == "100") & ((m.arm == "none") | (m.lam == "10"))]
pick.append(a1)
j = m[(m.task == "pancreas") & (m.seed == "0") & (((m.experiment == "X1") & m.arm.isin(["none", "scvi_adv"])) |
      ((m.experiment == "X13") & m.arm.isin(["scanvi", "sysvi"])))]
pick.append(j.drop_duplicates(subset=["experiment", "arm", "cond"]))
x15 = m[(m.experiment == "X15") & (m.task == "immune") & (m.seed == "10") & m.arm.isin(["none", "discriminator"]) &
        m.extra.str.contains("complete") & m.lam.isin(["0", "matched_lo"])]
pick.append(x15)
x3 = m[(m.experiment == "X3") & (m.task == "pancreas") & (m.seed == "10") & (m.arm == "discriminator") &
       (m.lam == "matched_lo") & m.extra.str.contains('"deplete_pct": 50') & m.extra.str.contains("depletion_oracle")]
pick.append(x3)
s = pd.concat(pick)
s.loc[~s.lam.str.match(r"^[0-9.]+$"), "lam"] = "10"
s["max_epochs"] = "1"
s.loc[s.experiment == "X15", "max_epochs"] = "2"
with open(f"{out}/manifest.tsv", "w") as f:
    f.write(head); s.to_csv(f, sep="\t", index=False)
open(f"{out}/tags.txt", "w").write("\n".join(s.tag) + "\n")
print(len(s), s.groupby(["experiment", "arm"]).size().to_dict())
PY
while read -r t; do
  ( cd $REPO && MANIFEST=$OUT/manifest.tsv PREPPED_DIR=$P OUT_DIR=$OUT/out WCD_SRC=$REPO/src TAG=$t \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 KMP_AFFINITY=disabled PYTHONWARNINGS=ignore \
      nice -n 19 $E/scvi-api/bin/python scripts/fit_paper_config.py > $OUT/$t.log 2>&1 ); echo "$t exit $?"
done < $OUT/tags.txt
$E/scvi-api/bin/python - "$OUT" <<'PY'
import sys, os, json, numpy as np
out = sys.argv[1]
for t in open(f"{out}/tags.txt").read().split():
    p = f"{out}/out/latents/{t}.npz"
    if not os.path.exists(p):
        print(t, "NO LATENT"); continue
    d = np.load(p, allow_pickle=False); cfg = json.loads(str(d["config"]))
    fit = cfg["plan"].get("fit") or cfg["plan"].get("scanvi")
    print(t, cfg.get("gpu"), "finite", bool(np.isfinite(d["z"]).all()), "n_cells", cfg["n_cells"], "kl", fit.get("n_epochs_kl_warmup"),
          "gp", fit.get("gradient_penalty"), fit.get("lambda_gp"), "iw", None if cfg.get("iw_weights") is None else len(cfg["iw_weights"]),
          "draw", (cfg.get("subsample_info") or {}).get("draw"))
PY
