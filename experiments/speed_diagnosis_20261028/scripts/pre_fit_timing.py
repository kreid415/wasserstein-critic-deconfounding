"""Time the steps that run before the fit timer in scripts/fit_paper_config.py (imports, data.py load,
read_h5ad, highly-variable-gene subset) for one task. Writes one JSON line. No model, no GPU, no outputs besides the line."""
import json, os, resource, sys, time, importlib.util
t = {"t_start": time.time()}
def mark(k): t[k] = round(time.time() - t["t_start"], 2)
import anndata as ad; mark("import_anndata")
import torch; mark("import_torch")
import scvi; mark("import_scvi")
spec = importlib.util.spec_from_file_location("wcd_data", os.path.join(os.environ["WCD_SRC"], "wcd_vae", "wcd", "data.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mark("load_data_py_scanpy")
a = ad.read_h5ad(os.path.join(os.environ["PREPPED_DIR"], sys.argv[1] + "__scib.h5ad")); mark("read_h5ad")
a = a[:, a.var["highly_variable"].values].copy(); mark("hvg_copy")
t.update(pid=os.getpid(), host=os.uname().nodename, n_obs=a.n_obs, n_vars=a.n_vars,
         maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2), t_start=round(t["t_start"], 1))
print(json.dumps(t), flush=True)
