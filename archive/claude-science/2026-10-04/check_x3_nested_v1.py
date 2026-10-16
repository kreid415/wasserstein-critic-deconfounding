"""Re-review check (X3, SI-41/SI-42): run the fitter's own x3_draw / depletion_oracle_weights (scripts/fit_paper_config.py
at the review commit) on the obs of the prepped scIB files for every X3 design cell of the manifest, and check
(1) equal N per dose = PREREG sec 9 sizes, (2) nesting of kept declared cells and refill across doses, (3) overlap
between doses = N - |difference of declared-cell counts|, (4) per-type kept = h0 - round(d*h0), minimum kept at dose 95,
(5) importance weights: weighted (batch, type) counts of K_d equal K0's, number/share of cells with weight 0.
Read-only on PREPPED. Usage: python check_x3_nested_v1.py REPO MANIFEST PREPPED OUT_DIR"""
import json, os, sys
import numpy as np, pandas as pd, anndata as ad
repo, man, prepped, out = sys.argv[1:5]
sys.path.insert(0, os.path.join(repo, "scripts"))
os.environ.setdefault("WCD_SRC", os.path.join(repo, "src"))
import fit_paper_config as F
M = pd.read_csv(man, sep="\t", comment="#", dtype=str, keep_default_na=False)
x3 = M[M.experiment == "X3"]
rows, cells = [], {}
for e in x3.extra.unique():
    sp = json.loads(e)["subsample"]
    key = (x3[x3.extra == e].task.iloc[0], sp["batch"], tuple(sp["types"]), sp["n_cells"], sp.get("draw"))
    cells.setdefault(key, set()).add(sp["deplete_pct"])
for (task, batch, types, n, draw), doses in sorted(cells.items()):
    a = ad.read_h5ad(os.path.join(prepped, f"{task}__scib.h5ad"), backed="r")
    obs = a.obs[["batch", "celltype"]].copy(); a.file.close()
    res = {}
    for d in sorted(doses):
        spec = dict(kind="composition", batch=batch, types=list(types), deplete_pct=d, n_cells=n, draw=draw)
        idx, info = F.x3_draw(obs, spec)
        res[d] = (idx, info)
    bat, ct = obs.batch.astype(str).values, obs.celltype.astype(str).values
    hit = (bat == batch) & np.isin(ct, list(types))
    k0 = set(res[0][0])
    for d in sorted(doses):
        idx, info = res[d]
        s = set(idx); decl = {i for i in idx if hit[i]}; fill = s - decl
        sub = ad.AnnData(obs=obs.iloc[idx].copy())
        w_ok, n_w0, share_w0, maxw = "", 0, 0.0, ""
        if 0 < d < 100:
            w, keep = F.depletion_oracle_weights(sub, spec | dict(deplete_pct=d), info)
            kd = F.group_counts(sub.obs)
            ok = all(abs(w[b].get(c, 0) * kd.get(b, {}).get(c, 0) - n0) < 1e-6 for b, r in info["k0_counts"].items() for c, n0 in r.items())
            cw = np.array([w[b][c] for b, c in zip(sub.obs.batch.astype(str), sub.obs.celltype.astype(str))])
            w_ok, n_w0, share_w0, maxw = ok, int((cw == 0).sum()), float((cw == 0).mean()), float(cw.max())
        prev = res[max([x for x in doses if x < d], default=d)]
        pdecl = {i for i in prev[0] if hit[i]}; pfill = set(prev[0]) - pdecl
        rows.append(dict(task=task, batch=batch, types="|".join(types), dose=d, draw=draw, n_cells=len(idx), n_spec=n,
                         n_task=len(obs), n_hit_task=int(hit.sum()), n_nonhit_task=int((~hit).sum()),
                         declared_kept=len(decl), refill_outside_K0=len(s - k0), overlap_with_dose0=len(s & k0),
                         expected_overlap=n - abs(len({i for i in res[0][0] if hit[i]}) - len(decl)),
                         nested_declared_vs_prev_dose=decl <= pdecl, nested_fill_vs_prev_dose=pfill <= fill,
                         per_type=json.dumps(info["per_type"]),
                         min_kept_type=min(v["kept"] for v in info["per_type"].values()),
                         kept_formula_ok=all(v["kept"] == v["h0"] - int(round(v["h0"] * d / 100)) for v in info["per_type"].values()),
                         iw_restores_k0=w_ok, cells_weight0=n_w0, share_weight0=round(share_w0, 4), max_weight=maxw))
df = pd.DataFrame(rows)
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, "x3_nested_v1_check.csv"), index=False)
print(df[["task", "dose", "n_cells", "declared_kept", "overlap_with_dose0", "expected_overlap", "nested_declared_vs_prev_dose",
          "nested_fill_vs_prev_dose", "min_kept_type", "kept_formula_ok", "iw_restores_k0", "cells_weight0", "share_weight0", "max_weight"]].to_string())
