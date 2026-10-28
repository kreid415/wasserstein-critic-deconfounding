"""Driver (runs on cs-local): launch N concurrent copies of profile_score.py under one thread setting, wait, report wall.
usage: run_cfg.py <label> <n_proc> <thread_setting> <latent.npz,...> [nice]; copy i scores latent i % len(latents)"""
import json, os, subprocess, sys, time
lab, n, setting, lats = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")
SD = os.environ["SD"]; OUT = f"{SD}/out/{lab}"; os.makedirs(OUT, exist_ok=True)
KP = "/home/kendall/.claude-science/conda/envs/wcd-kbet"
base = {k: str(1) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
base.update(KMP_AFFINITY="disabled", PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1", MKL_THREADING_LAYER="SEQUENTIAL")
S = {"t1": {}, "numba2": {"NUMBA_NUM_THREADS": "2"}, "numba4": {"NUMBA_NUM_THREADS": "4"},
     "blas2": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_THREADING_LAYER": "INTEL"},
     "all2": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMBA_NUM_THREADS": "2", "MKL_THREADING_LAYER": "INTEL"}}[setting]
env0 = {**os.environ, **base, **S, "R_HOME": f"{KP}/lib/R", "R_LIBS": f"{os.environ['D']}/Rlib_kbet", "CUDA_VISIBLE_DEVICES": "",
        "KBET_SEED": "0", "REPO": f"{SD}/repo", "PYTHONSAFEPATH": "1"}
procs, t0 = [], time.time()
for i in range(n):
    lat = lats[i % len(lats)]; tag = os.path.basename(lat)[:-4]; task = next(t for t in ("atac_small", "atac_large", "immune_hum_mou", "immune", "lung", "pancreas", "sim1", "sim2") if tag.startswith("A1_" + t + "_"))
    env = dict(env0, PREPPED=f"{os.environ['D']}/prepped_scib/{task}__scib.h5ad", NPZ=lat, TAG=tag, META="{}",
               OUT_CSV=f"{OUT}/p{i}_{tag}.csv", PROFILE_JSON=f"{OUT}/p{i}_{tag}.json")
    cmd = (["nice", "-n", sys.argv[5]] if len(sys.argv) > 5 else []) + [f"{KP}/bin/python", f"{SD}/profile_score.py"]
    procs.append(subprocess.Popen(cmd, env=env, stdout=open(f"{OUT}/p{i}.log", "w"), stderr=subprocess.STDOUT))
rc = [p.wait() for p in procs]
print(json.dumps(dict(label=lab, n=n, setting=setting, wall=round(time.time() - t0, 1), rc=rc, load_start=os.getloadavg())))
