"""Realised cell counts of the X3 composition and X8 batch-number subsamples, computed with the tagged
fit_paper_config.subsample() on obs-only AnnData objects built from the prepped files (read-only)."""
import json, os, sys
import anndata as ad, h5py, numpy as np, pandas as pd
REPO = sys.argv[1]; PREPPED = sys.argv[2]; OUT = sys.argv[3]
sys.path.insert(0, os.path.join(REPO, "scripts"))
import fit_paper_config as F
man = pd.read_csv(os.path.join(REPO, "manifests/paper_manifest_stock_pilot_u5_b10.tsv"), sep="\t", comment="#",
                  dtype=str, keep_default_na=False)
def obs_of(task):
    a = ad.read_h5ad(os.path.join(PREPPED, f"{task}__scib.h5ad"), backed="r")
    o = a.obs[["batch", "celltype"]].copy(); a.file.close()
    return ad.AnnData(obs=o)
rows = []
for exp in ("X3", "X8"):
    sub = man[man.experiment == exp]
    specs = {}
    for r in sub.itertuples():
        ex = json.loads(r.extra)
        key = (r.task, json.dumps(ex["subsample"], sort_keys=True))
        specs.setdefault(key, set()).add(r.max_epochs)
    cache = {}
    for (task, spec_s), epochs in sorted(specs.items()):
        if task not in cache:
            cache[task] = obs_of(task)
        a = cache[task]
        spec = json.loads(spec_s)
        if spec["kind"] == "composition":
            s, info = F.subsample(a, spec, 0, return_info=True)
        else:
            s, info = F.subsample(a, spec, 0), {}
        bc = s.obs["batch"].value_counts()
        tr = s.obs.groupby("batch", observed=True)["celltype"].value_counts()
        rows.append(dict(experiment=exp, task=task, spec=spec_s, max_epochs=sorted(epochs), n_cells=int(s.n_obs),
                         n_batches=int(bc.size), min_batch=int(bc.min()), steps_per_epoch=int(np.ceil(0.9 * s.n_obs / 128)),
                         **{k: v for k, v in info.items()}))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
x3 = df[df.experiment == "X3"].copy()
x3["dose"] = [json.loads(s)["deplete_pct"] for s in x3.spec]
print(x3[["task", "dose", "n_cells", "n_hit", "n_drop", "steps_per_epoch", "max_epochs"]].to_string(index=False))
x8 = df[df.experiment == "X8"].copy()
x8["V"] = [json.loads(s)["n_batches"] for s in x8.spec]; x8["subset"] = [json.loads(s)["subset"] for s in x8.spec]
print(x8[["task", "V", "subset", "n_cells", "n_batches", "min_batch", "max_epochs"]].to_string(index=False))
