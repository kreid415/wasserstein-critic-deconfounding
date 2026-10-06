"""Peak memory and wall time of the scorer (scripts/score_scib_native.py at c88cce3) on one latent each of
immune_hum_mou and atac_large, both at once, nice 19, beside the running A1 (production setting of the filler scoring).
Scores go to this scratch dir, not the filler OUT_DIR. RSS = sum over the process tree, sampled every 2 s."""
import glob, hashlib, json, os, subprocess, time
import pandas as pd
R, S, D = "/home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/a0b87862-8454-468c-a2f9-6326cd1433fc/scratch/wt_v2", "/home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/a0b87862-8454-468c-a2f9-6326cd1433fc/scratch/score_probe", "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
ENVS = "/home/kendall/.claude-science/conda/envs"
man = pd.read_csv(f"{R}/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv", sep="\t", comment="#", dtype=str).set_index("tag")
def tree(pid):
    out, todo = [], [pid]
    while todo:
        p = todo.pop(); out.append(p)
        for t in glob.glob(f"/proc/{p}/task/*/children"):
            try: todo += [int(x) for x in open(t).read().split()]
            except OSError: pass
    return out
def rss(pids):
    tot = 0
    for p in pids:
        try: tot += int(open(f"/proc/{p}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError: pass
    return tot
jobs = {}
for t in ("immune_hum_mou", "atac_large"):
    npz = sorted(glob.glob(f"{D}/jhpce_tier12/fillers_x1_x13/latents/X1_{t}_none_l0_c1_s0_*.npz"))[0]
    tag = os.path.basename(npz)[:-4]; r = man.loc[tag]
    sha = hashlib.sha256(open(npz, "rb").read()).hexdigest()
    meta = dict(latent_sha256=sha, experiment=r["experiment"], task=r["task"], arm=r["arm"], lam=r["lam"], cond=r["cond"], seed=r["seed"])
    env = dict(os.environ, PREPPED=f"{D}/prepped_scib/{t}__{r['counts']}.h5ad", NPZ=npz, TAG=tag, META=json.dumps(meta), OUT_CSV=f"{S}/{tag}.csv",
               KBET_SEED="0", R_HOME=f"{ENVS}/wcd-kbet/lib/R", R_LIBS=f"{D}/Rlib_kbet", CUDA_VISIBLE_DEVICES="",
               KMP_AFFINITY="disabled", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    log = open(f"{S}/{tag}.log", "w")
    p = subprocess.Popen(["nice", "-n", "19", f"{ENVS}/wcd-kbet/bin/python", f"{R}/scripts/score_scib_native.py"], env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    jobs[t] = dict(tag=tag, p=p, t0=time.time(), peak=0)
while any(j["p"].poll() is None for j in jobs.values()):
    for j in jobs.values():
        if j["p"].poll() is None: j["peak"] = max(j["peak"], rss(tree(j["p"].pid)))
    time.sleep(2)
res = {t: dict(tag=j["tag"], returncode=j["p"].returncode, wall_s=round(time.time() - j["t0"], 1), peak_rss_gb=round(j["peak"] / 1e9, 2)) for t, j in jobs.items()}
for t, j in jobs.items():
    res[t]["wall_s"] = round(os.path.getmtime(f"{S}/{j['tag']}.log") - j["t0"], 1)
json.dump(res, open(f"{S}/probe.json", "w"), indent=1); print(json.dumps(res))
