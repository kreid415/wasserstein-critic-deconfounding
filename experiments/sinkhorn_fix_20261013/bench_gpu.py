"""GPU cost of one Sinkhorn loss forward + backward (tagged vs branch) on 128-cell minibatches of the review's
inventory: per regime, the fits named below, 12 minibatches each (drawn as in validate_sinkhorn.py), 20 timed calls
per minibatch after 3 warm-up calls; reports the median ms per call. The branch is timed with the solve in float64
and in float32 (SI-58; order alternates per minibatch), with the two results' gradient cosine and value difference.
Usage: python bench_gpu.py OUT.csv
"""
import csv, json, os, sys, time
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # PYTHONSAFEPATH=1 in this sandbox
import validate_sinkhorn as V  # noqa: E402
V.NEW._marginal_err = V._err          # time the unpatched solver (no iteration counter)

REGIMES = {   # regime -> (experiment, task, lam, cond, zstd) of the fit used, seed 100
    "A1_wide_l0.1": ("A1", "atac_small", "0.1", "0", "0"),
    "A1_l10": ("A1", "immune", "10", "1", "0"),
    "A1_shrunk_l3000": ("A1", "immune", "3000", "0", "0"),
    "A3_zstd_l30": ("A3", "atac_small", "30", "0", "1"),
    "A3_zstd_l3_immune": ("A3", "immune", "3", "1", "1"),
}


def minibatches(r, n_mb=12):
    q = np.load(r["latent_path"], allow_pickle=False)
    z, b = q["z"], np.unique(q["batch"], return_inverse=True)[1]
    tr = np.asarray(torch.load(r["model_path"], map_location="cpu", weights_only=False)["attr_dict"]["train_indices_"])
    rng = torch.Generator().manual_seed(80301 + int(r["seed"]))
    for idx in tr[torch.randperm(len(tr), generator=rng).numpy()[:128 * n_mb]].reshape(-1, 128):
        yield z[idx], b[idx]


def time_call(mod, z, b, zstd, reps=20):
    dev = torch.device("cuda")
    bt = torch.as_tensor(b, device=dev)
    ts = []
    for i in range(reps + 3):
        mu = torch.tensor(z, dtype=torch.float32, device=dev, requires_grad=True)
        torch.cuda.synchronize(); t = time.perf_counter()
        v = mod.sinkhorn_batch_others(V._adv(mu, zstd), bt)
        (g,) = torch.autograd.grad(v, mu)
        torch.cuda.synchronize()
        if i >= 3:
            ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e3, float(v), g.double().cpu().numpy().ravel()


if __name__ == "__main__":
    inv = json.load(open(f"{V.REVIEW}/inventory.json"))
    rows = []
    for name, key in REGIMES.items():
        zstd = key[-1]
        r = next(x for x in inv if (x["experiment"], x["task"], x["lam"], x["cond"], x["zstd"], x["seed"]) == key + ("100",))
        for j, (z, b) in enumerate(minibatches(r)):
            res = {"tag": time_call(V.OLD, z, b, zstd == "1")}
            for dt in ((torch.float64, torch.float32) if j % 2 == 0 else (torch.float32, torch.float64)):  # order
                V.NEW.SINKHORN_DTYPE = dt
                res[dt] = time_call(V.NEW, z, b, zstd == "1")
            (m64, v64, g64), (m32, v32, g32) = res[torch.float64], res[torch.float32]
            # new_ms: float64 solve (as before SI-58); new32_ms: float32 solve; agreement of the two on the GPU
            rows.append(dict(regime=name, tag=r["tag"], minibatch=j, tag_ms=res["tag"][0], new_ms=m64, new32_ms=m32,
                             gpu_cos_32_64=float(g32 @ g64 / (np.linalg.norm(g32) * np.linalg.norm(g64))),
                             gpu_value_rel_32_64=(v32 - v64) / abs(v64)))
        print(name, r["tag"], "median ms: tag %.1f, branch float64 %.1f, float32 %.1f" % tuple(
            np.median([x[k] for x in rows if x["regime"] == name]) for k in ("tag_ms", "new_ms", "new32_ms")), flush=True)
    with open(sys.argv[1], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
