"""Builds results/lisi_values_by_latent.csv (one row per latent: iLISI/cLISI of every condition, hex-exact via float.hex
strings in the JSONs, shown here as repr floats) and results/helper_rows_kept.csv (rows the knn_graph helper kept per call).
Usage: summarize_probe.py RESULTS_DIR   (RESULTS_DIR holds jh/, ws/, ws_stored_reference_lisi.json one level up)"""
import csv, glob, json, os, sys
res = sys.argv[1]; ref = json.load(open(os.path.join(res, "..", "ws_stored_reference_lisi.json")))
conds = {"jh_nat": ("jh", "natural_run1"), "jh_shim_final_1": ("jh", "final_shim_run1"), "jh_shim_final_2": ("jh", "final_shim_run2"),
         "ws_nat_1": ("ws", "natural_run1"), "ws_nat_2": ("ws", "natural_run2"), "ws_shim_final": ("ws", "final_shim_run1")}
vals = {k: {} for k in conds}
for k, (d, c) in conds.items():
    for f in glob.glob(f"{res}/{d}/{c}_*.json"):
        r = json.load(open(f))["rows"][0]; vals[k][r["latent"][:-4]] = r
tags = sorted(vals["ws_nat_1"]); assert len(tags) == 20 and all(set(v) == set(tags) for v in vals.values())
with open(f"{res}/lisi_values_by_latent.csv", "w") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["latent", "n_cells", "stored_ws_iLISI", "stored_ws_cLISI"] + [f"{k}_{m}" for k in conds for m in ("iLISI", "cLISI")])
    for t in tags:
        w.writerow([t, vals["ws_nat_1"][t]["n_cells"], ref[t]["iLISI"], ref[t]["cLISI"]] + [x for k in conds for x in (repr(vals[k][t]["ilisi"]), repr(vals[k][t]["clisi"]))])
with open(f"{res}/helper_rows_kept.csv", "w") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["latent", "n_cells"] + [f"{k}_{m}" for k in conds for m in ("iLISI_call", "cLISI_call")])
    for t in tags:
        w.writerow([t, vals["ws_nat_1"][t]["n_cells"]] + [vals[k][t][m + "_helper"]["rows_kept"] for k in conds for m in ("ilisi", "clisi")])
print("wrote", len(tags), "latents")
