"""Exact comparison of score rows: cmp_scores.py <ref_dir_or_label> <new_dir>; matches files by tag (name after pN_).
Metric columns = those of scorer BATCH/BIO lists + kbet counts; prints per tag the number of exactly equal columns and max abs diff."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
M = ["NMI_cluster/label","ARI_cluster/label","ASW_label","ASW_label/batch","PCR_batch","cell_cycle_conservation","isolated_label_F1",
     "isolated_label_silhouette","graph_conn","kBET","iLISI","cLISI","trajectory","kbet_r_calls","kbet_labels_forced_one","kbet_labels_skipped"]
def tag(f): return re.sub(r"^p\d+_", "", os.path.basename(f))[:-4]
def load(d): return {tag(f): pd.read_csv(f).iloc[0] for f in glob.glob(os.path.join(d, "*.csv"))}
ref, new = load(sys.argv[1]), load(sys.argv[2])
out = {}
for t in sorted(set(ref) & set(new)):
    a, b = ref[t], new[t]
    eq = [m for m in M if m in a and m in b and ((a[m] == b[m]) or (pd.isna(a[m]) and pd.isna(b[m])))]
    bad = {m: float(abs(a[m] - b[m])) for m in M if m in a and m in b and m not in eq}
    out[t] = dict(n_equal=len(eq), n_cols=sum(m in a and m in b for m in M), diffs=bad)
print(json.dumps(out, indent=1))
