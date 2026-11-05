"""Cell-type coverage of the reference batch chosen by the entropy rule (wcd.data.select_reference_batch, maximum cell-type
Shannon entropy, ties by size) on each of the eight task files. Reads only the scIB task files (backed mode); no random numbers.
Run (python with anndata, pandas, numpy; the scvi-api environment on the workstation):
  WCD_DATA_ROOT=/home/kendall/experiment_data/wasserstein-critic-deconfounding python paper/analysis_pass10/reference_coverage.py paper/analysis_pass10/reference_coverage.json"""
import json, os, sys
import anndata as ad, numpy as np
root = os.environ["WCD_DATA_ROOT"] + "/prepped_scib"
TASKS = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou", "sim1", "sim2"]
out = {"anndata": ad.__version__}
for t in TASKS:
    o = ad.read_h5ad(f"{root}/{t}__scib.h5ad", backed="r").obs
    bk = next(c for c in ("batch", "Batch") if c in o.columns)
    ck = next(c for c in ("cell_type", "celltype", "Group", "cell_type_label", "label") if c in o.columns)
    rows = []
    for name, sub in o.groupby(bk, observed=True):
        cnt = sub[ck].value_counts(); cnt = cnt[cnt > 0]; p = cnt.values / cnt.values.sum()
        rows.append((float(-(p * np.log(p)).sum()), len(sub), str(name), int(len(cnt))))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    out[t] = dict(batch_key=bk, celltype_key=ck, reference=rows[0][2], entropy=rows[0][0], cells=rows[0][1],
                  types_in_reference=rows[0][3], types_in_task=int(o[ck].nunique()), batches=int(o[bk].nunique()))
    print(t, out[t])
json.dump(out, open(sys.argv[1], "w"), indent=1)
