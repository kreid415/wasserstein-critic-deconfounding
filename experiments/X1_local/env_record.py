#!/usr/bin/env python
"""Environment record for the cs-local identity check: what could make a fit or a score differ between the chat sandbox
and a cs-local host job. Writes one JSON: per interpreter (fit, score) the executable, version, package versions, torch
CUDA/cuDNN and GPU, BLAS/thread pools; process-wide env vars that steer threads, BLAS, numba, CUDA, R and paths; CPU
model and flag set; GPU driver; and where the process runs (PID namespace, cgroup, session) -- the last group is
expected to differ and is reported, not compared.
Each interpreter is probed with the environment the tagged runner gives that kind of child (child_env).
Usage: env_record.py --fit-python PY --score-python PY --r-home DIR --r-libs DIR --out env.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

PROBE = r"""
import json, sys, importlib
out = dict(executable=sys.executable, version=sys.version, prefix=sys.prefix)
mods = {}
for m in ("torch", "lightning", "scvi", "numpy", "scipy", "pandas", "anndata", "scanpy", "sklearn", "numba", "llvmlite",
          "ot", "scib", "rpy2", "anndata2ri", "igraph", "leidenalg", "pynndescent", "umap", "h5py", "threadpoolctl"):
    try:
        mods[m] = str(getattr(importlib.import_module(m), "__version__", "?"))
    except Exception as e:
        mods[m] = f"absent ({type(e).__name__})"
out["modules"] = mods
try:
    import torch
    out["torch_cuda"] = torch.version.cuda
    out["cudnn"] = torch.backends.cudnn.version()
    out["cuda_available"] = torch.cuda.is_available()
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    out["torch_threads"] = torch.get_num_threads()
except Exception as e:
    out["torch_error"] = repr(e)
try:
    import threadpoolctl
    out["threadpools"] = sorted((p.get("internal_api"), p.get("version"), p.get("num_threads"), p.get("architecture"))
                                for p in threadpoolctl.threadpool_info())
except Exception as e:
    out["threadpools_error"] = repr(e)
try:
    import numba
    out["numba_cpu_name"] = numba.config.CPU_NAME
    out["numba_cpu_features"] = numba.config.CPU_FEATURES
except Exception as e:
    out["numba_error"] = repr(e)
print(json.dumps(out, default=str))
"""
ENV_PREFIXES = ("OMP_", "MKL_", "OPENBLAS_", "KMP_", "NUMBA_", "CUDA_", "CUBLAS_", "NVIDIA_", "PYTHON", "R_", "CONDA",
                "LD_", "LANG", "LC_", "TZ", "TMPDIR", "XDG_", "HOME", "USER", "PATH")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"error {r.returncode}: {r.stderr.strip()[-200:]}"


def child_env(kind, r_home, r_libs):
    """The environment scripts/run_stage.py (tag) gives its children: thread_env(1) (OMP/MKL/OPENBLAS/NUMBA = 1,
    KMP_AFFINITY disabled, PYTHONWARNINGS ignore, PYTHONUNBUFFERED 1); scorers also get R_HOME, R_LIBS and no GPU."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               KMP_AFFINITY="disabled", PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    if kind == "score":
        env.update(R_HOME=r_home, R_LIBS=r_libs, CUDA_VISIBLE_DEVICES="")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-python", required=True)
    ap.add_argument("--score-python", required=True)
    ap.add_argument("--r-home", required=True)
    ap.add_argument("--r-libs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = dict(interpreters={})
    for k, py in (("fit", a.fit_python), ("score", a.score_python)):
        r = subprocess.run([py, "-c", PROBE], capture_output=True, text=True, env=child_env(k, a.r_home, a.r_libs))
        if r.returncode != 0:
            raise SystemExit(f"{k} interpreter probe failed: {r.stderr[-400:]}")
        rec["interpreters"][k] = json.loads(r.stdout.strip().splitlines()[-1])
    rec["env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}
    cpu = open("/proc/cpuinfo").read().split("\n\n")[0]
    fields = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in cpu.splitlines() if ":" in l)}
    flags = sorted(fields.get("flags", "").split())
    if not flags:
        raise SystemExit("no CPU flags parsed from /proc/cpuinfo")
    rec["cpu"] = dict(model=fields.get("model name", ""), n_flags=len(flags),
                      flags_sha256=hashlib.sha256(" ".join(flags).encode()).hexdigest(),
                      cpu_count=os.cpu_count(), affinity=len(os.sched_getaffinity(0)))
    rec["gpu_driver"] = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    rec["where"] = dict(pid_ns=os.readlink("/proc/self/ns/pid"), mnt_ns=os.readlink("/proc/self/ns/mnt"),
                        cgroup=open("/proc/self/cgroup").read().strip().splitlines()[:3], sid=os.getsid(0), pid=os.getpid(),
                        uid=os.getuid(), hostname=os.uname().nodename)
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"env record written: {a.out}")


if __name__ == "__main__":
    sys.exit(main())
