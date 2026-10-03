#!/usr/bin/env python
"""X3 design profiles (docs/SPECS_missing_arms.md section 4) from the obs of prepped_scib/<task>__scib.h5ad.

For every X3 target of scripts/build_paper_manifest.py (X3_TARGETS: depleted batch, depleted types, reference) it
checks, and refuses to write anything otherwise, that
  * the fixed reference equals the repo rule (select_reference_batch, max cell-type entropy) on the dose-0
    subsample, is not the depleted batch, and is present at every dose (SI-31);
  * the depleted batch is present at every dose (sim2: only Group1 is depleted, SI-32);
and writes, with the runner's own functions (scripts/fit_paper_config.py subsample, depletion_oracle_weights):
  docs/x3_iw_weight_profile.csv      oracle importance weights per task x dose: keep fraction, weight of the
                                     depleted-type cells, effective-sample-size fraction of the depleted batch
                                     ESS = (sum_b w)^2 / (n_b sum_b w^2), depleted-type cells per 128-cell minibatch,
                                     and the reference the automatic rule would pick at that dose (for disclosure)
  docs/x3_shared_support_profile.csv a common all-batch target restricted to the types present in every batch
                                     (>= 1 or >= 10 cells): pooled composition over S / batch composition, max weight,
                                     min ESS fraction over batches (cells outside S weigh 0). Not the design: it shows
                                     why a common target is not usable (n_shared = 0 on immune_hum_mou and sim2).
Usage (env scvi-api): python scripts/x3_design_profile.py --prepped-dir <prepped_scib> [--out-dir docs]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_paper_manifest as bpm  # noqa: E402
import fit_paper_config as fpc  # noqa: E402

DOSES = [0, 50, 80, 95, 100]
MINIBATCH = 128


def _repo_data_module():
    spec = importlib.util.spec_from_file_location("wcd_data", os.path.join(ROOT, "src", "wcd_vae", "wcd", "data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_obs(prepped_dir, task):
    import anndata as ad
    import h5py
    with h5py.File(os.path.join(prepped_dir, f"{task}__scib.h5ad"), "r") as f:
        obs = ad.io.read_elem(f["obs"])
    for col in ("batch", "celltype"):
        if col not in obs:
            raise KeyError(f"{task}: obs lacks {col!r}")
    return ad.AnnData(obs=obs[["batch", "celltype"]].copy())


def _cells(task, a0, b, types, dose):
    spec = dict(kind="composition", batch=b, types=types, deplete_pct=dose, n_cells=min(20000, bpm.TASKS[task][0]))
    a, sel = fpc.subsample(a0, spec, 0, return_info=True)
    a.obs["batch"] = a.obs["batch"].astype(str).astype("category")
    return a, sel, spec


def weight_profile(task, a0, b, types, ref, select):
    rows = []
    for dose in DOSES:
        a, sel, spec = _cells(task, a0, b, types, dose)
        bb, cc = a.obs["batch"].astype(str).values, a.obs["celltype"].astype(str).values
        if dose in (50, 80, 95):
            table, kappa = fpc.depletion_oracle_weights(a, spec, sel)
            w = np.array([table[x][y] for x, y in zip(bb, cc)], dtype=float)
        else:   # every weight is 1 at doses 0 and 100 (no IW rows there)
            w, kappa = np.ones(a.n_obs), None
        inb = bb == b
        hit = inb & np.isin(cc, types)
        auto = str(select(a, "batch", "celltype"))
        rows.append(dict(task=task, depleted_batch=b, depleted_types=",".join(types), reference=ref, dose=dose,
                         n_cells=int(a.n_obs), K=int(a.obs["batch"].nunique()), n_hit_full=int(sel["n_hit"]),
                         drop=int(sel["n_drop"]), keep_fraction=(round(float(kappa), 4) if kappa is not None else ""),
                         w_depleted=(round(float(w[hit].max()), 3) if hit.any() else ""),
                         n_hit_kept=int(hit.sum()), n_batch_b=int(inb.sum()), sum_w_hit=round(float(w[hit].sum()), 1),
                         ess_frac_batch_b=round(float(w[inb].sum() ** 2 / (inb.sum() * (w[inb] ** 2).sum())), 3),
                         exp_hit_cells_per_minibatch=round(MINIBATCH * hit.sum() / a.n_obs, 2),
                         reference_present=bool((bb == ref).any()), depleted_batch_present=bool(inb.any()),
                         auto_reference_at_dose=auto))
    return rows


def shared_support_profile(task, a0, b, types):
    rows = []
    for dose in DOSES:
        a, _, _ = _cells(task, a0, b, types, dose)
        ct = pd.crosstab(a.obs["batch"].astype(str), a.obs["celltype"].astype(str))
        for nmin in (1, 10):
            S = [y for y in ct.columns if (ct[y] >= nmin).all()]
            nS = int(ct[S].values.sum()) if S else 0
            maxw = miness = np.nan
            if S:
                pi = ct[S].sum(0) / nS                       # pooled composition over the shared types
                pk = ct.div(ct.sum(1), axis=0)               # each batch's composition over all its types
                w = pi / pk[S]
                ess = [(lambda wk: (wk.sum() ** 2 / (wk ** 2).sum()) / ct.loc[k].sum())(
                    np.repeat(w.loc[k].values, ct.loc[k, S].values)) for k in ct.index]
                maxw, miness = float(w.values.max()), float(np.min(ess))
            rows.append(dict(task=task, dose=dose, n=int(a.n_obs), K=int(ct.shape[0]), n_types=int(ct.shape[1]),
                             min_cells_present=nmin, n_shared=len(S), frac_cells_in_S=round(nS / a.n_obs, 3),
                             max_w=round(maxw, 2), min_ess_frac=round(miness, 3),
                             depleted_in_S=",".join(t for t in types if t in S)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepped-dir", required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "docs"))
    a = ap.parse_args()
    select = _repo_data_module().select_reference_batch
    wrows, srows = [], []
    for task, (b, types, ref) in bpm.X3_TARGETS.items():
        a0 = read_obs(a.prepped_dir, task)
        a_d0, _, _ = _cells(task, a0, b, types, 0)
        auto0 = str(select(a_d0, "batch", "celltype"))
        if auto0 != ref:
            raise ValueError(f"{task}: X3_TARGETS reference {ref!r} but the rule picks {auto0!r} on the dose-0 subsample")
        if ref == b:
            raise ValueError(f"{task}: the reference {ref!r} is the depleted batch")
        w = weight_profile(task, a0, b, types, ref, select)
        bad = [r["dose"] for r in w if not (r["reference_present"] and r["depleted_batch_present"])]
        if bad:
            raise ValueError(f"{task}: reference or depleted batch absent at doses {bad}")
        wrows += w
        srows += shared_support_profile(task, a0, b, types)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(wrows).to_csv(os.path.join(a.out_dir, "x3_iw_weight_profile.csv"), index=False)
    pd.DataFrame(srows).to_csv(os.path.join(a.out_dir, "x3_shared_support_profile.csv"), index=False)
    print(f"[x3] {len(wrows)} weight rows, {len(srows)} shared-support rows -> {a.out_dir}")


if __name__ == "__main__":
    main()
