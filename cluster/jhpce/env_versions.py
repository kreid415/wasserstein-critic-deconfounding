#!/usr/bin/env python
"""Print a JSON record of the versions that matter for fits (--kind fit) or scoring (--kind score), as seen by
the RUNNING interpreter (import-time __version__, not pip metadata: local wcd-kbet carries a stale igraph 0.11.8
dist-info next to the live conda igraph 0.11.9, so pip metadata alone would misreport it).

Usage: python cluster/jhpce/env_versions.py --kind fit|score [--out FILE]
Fit kind adds the CUDA stack and, if a GPU is visible, the device name and driver. Score kind adds R, rpy2 and
the kBET fingerprint (requires R_HOME / R_LIBS as in scripts/run_wave.sh).
"""
import argparse
import importlib
import json
import platform
import subprocess
import sys

FIT = ["torch", "scvi", "lightning", "ot", "anndata", "scanpy", "numpy", "scipy", "pandas", "sklearn", "numba",
       "pynndescent", "h5py", "pyro"]
SCORE = ["scib", "scanpy", "anndata", "numpy", "scipy", "pandas", "sklearn", "igraph", "leidenalg", "numba",
         "llvmlite", "pynndescent", "umap", "h5py", "rpy2", "anndata2ri", "torch"]   # scvi-tools 0.14.6 in wcd-kbet does not import (torchmetrics API); scoring never imports it


DIST = {"rpy2": "rpy2-robjects", "ot": "POT", "sklearn": "scikit-learn", "umap": "umap-learn", "scvi": "scvi-tools",
        "pyro": "pyro-ppl"}


def mod_version(name):
    m = importlib.import_module(name)          # an ImportError here is a real failure of the env: let it raise
    v = getattr(m, "__version__", None)
    if v is None:                              # e.g. rpy2 3.6 has no top-level __version__: use the dist metadata
        from importlib.metadata import version
        v = version(DIST.get(name, name)) + " (dist metadata)"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["fit", "score"])
    ap.add_argument("--out")
    a = ap.parse_args()
    rec = dict(kind=a.kind, python=platform.python_version(), executable=sys.executable, host=platform.node(),
               machine=platform.machine())
    rec["modules"] = {n: mod_version(n) for n in (FIT if a.kind == "fit" else SCORE)}
    if a.kind == "fit":
        import torch
        rec["torch_cuda"] = torch.version.cuda
        rec["cudnn"] = torch.backends.cudnn.version()
        rec["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
            rec["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, check=True)
            rec["nvidia_smi"] = smi.stdout.strip().splitlines()
    else:
        import numpy  # noqa: F401  (load the BLAS/OpenMP libraries numpy, scipy and sklearn use at runtime)
        import scipy.linalg  # noqa: F401
        import sklearn.utils  # noqa: F401
        from threadpoolctl import threadpool_info
        rec["runtime_threadpools"] = sorted(
            f"{d.get('internal_api')} {d.get('version')} {d.get('filepath', '').rsplit('/', 1)[-1]} "
            f"{d.get('architecture') or d.get('threading_layer') or ''}".strip() for d in threadpool_info())
        r = subprocess.run(["Rscript", "-e", "cat(R.version.string)"], capture_output=True, text=True, check=True)
        rec["R"] = r.stdout.strip()
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        k = subprocess.run(["Rscript", os.path.join(here, "kbet_fingerprint.R")], capture_output=True, text=True, check=True)
        rec["kbet"] = k.stdout.strip()
    out = json.dumps(rec, indent=1, sort_keys=True)
    print(out)
    if a.out:
        open(a.out, "w").write(out + "\n")


if __name__ == "__main__":
    main()
