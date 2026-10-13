"""Sinkhorn fix (SI-51): value and gradient of the branch's sinkhorn_batch_others, and of the tagged one, against the
converged float64 reference of the final review (sinkhorn_reference.py, copied unchanged from
review/esc2/sinkhorn_astra, sha256 0e6663fb...), on the minibatches that review used:
  final: N_MB shuffled 128-cell minibatches of every saved latent of the 234 Sinkhorn fits (review inventory.json,
         sha256 05dbcaba..., train indices from each model.pt), A3 standardisation applied as in training;
  early: the 160 untrained / epoch-2-3 replay captures (untrained atac_small and immune also standardised);
  trainmode: the 76 training-mode minibatches of 19 loaded final models.
Gradients are w.r.t. the raw float32 posterior means (through the standardisation for zstd rows), as in training.
Also records the branch solver's iterations at the final eps (cross, self-a, self-b).
Usage: python validate_sinkhorn.py OUT.csv [N_MB=4] [WORKERS=6]   (CPU only)
"""
import csv, glob, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # CPU unless the caller sets a GPU
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
D = "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
REVIEW = f"{D}/review/esc2/sinkhorn_astra"
TAG_SRC = f"{D}/code/wcd_prereg-tier12-v2/src/wcd_vae/wcd/alignment.py"      # tag prereg-tier12-v2 = c88cce3
sys.path.insert(0, HERE)
from sinkhorn_reference import cost_matrix, reference_cost  # noqa: E402


def _load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


NEW = _load(f"{REPO}/src/wcd_vae/wcd/alignment.py", "align_new")
OLD = _load(TAG_SRC, "align_tag")
ITERS = []
_err = NEW._marginal_err


def _counted(*a):
    ITERS[-1] += 1
    return _err(*a)


NEW._marginal_err = _counted


def _adv(mu, zstd):
    return (mu - mu.mean(0, keepdim=True)) / mu.std(0, unbiased=False, keepdim=True).clamp_min(1e-4) if zstd else mu


def compare(z, b, zstd):
    bt = torch.as_tensor(b, dtype=torch.long)
    out = {}
    m64 = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    a64 = _adv(m64, zstd)
    vr, stats = reference_cost(cost_matrix(a64), bt, 0.05 * math.sqrt(2 * a64.shape[1]), tol=1e-10)
    gr = torch.autograd.grad(vr, m64)[0].numpy().ravel()
    out["value_ref"], out["ref_max_residual"] = float(vr.detach()), max(s["residual"] for s in stats)
    for name, mod in (("new", NEW), ("tag", OLD)):
        mu = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        if name == "new":
            ITERS.clear()
            orig = NEW._ot_eps, NEW._ot_eps_sym
            NEW._ot_eps = lambda *a, _f=orig[0]: (ITERS.append(0), _f(*a))[1]
            NEW._ot_eps_sym = lambda *a, _f=orig[1]: (ITERS.append(0), _f(*a))[1]
        try:
            v = mod.sinkhorn_batch_others(_adv(mu, zstd), bt)
        except RuntimeError as e:          # cap reached: record it and the iterations so far
            out.update(new_error=str(e), new_iters=json.dumps(ITERS))
            continue
        finally:
            if name == "new":
                NEW._ot_eps, NEW._ot_eps_sym = orig
        g = torch.autograd.grad(v, mu)[0].double().numpy().ravel()
        ng, nr = np.linalg.norm(g), np.linalg.norm(gr)
        dot = float(g @ gr)
        c = dot / nr ** 2 if nr > 0 else float("nan")
        out.update({f"{name}_value_rel_err": (float(v.detach()) - out["value_ref"]) / abs(out["value_ref"]) if out["value_ref"] else float("nan"),
                    f"{name}_cosine": dot / (ng * nr) if ng * nr > 0 else float("nan"),
                    f"{name}_norm_ratio": ng / nr if nr > 0 else float("nan"),
                    f"{name}_c": c,
                    f"{name}_resid_after_c": float(np.linalg.norm(g - c * gr) / ng) if ng > 0 else float("nan")})
        if name == "new":
            out["new_iters_cross"], out["new_iters_self_a"], out["new_iters_self_b"] = ITERS
            out.update(new_error="", new_iters=json.dumps(ITERS))
    out["ref_grad_norm"], out["n_batches"] = float(np.linalg.norm(gr)), int(len(np.unique(b)))
    return out


def jobs(n_mb):
    for r in json.load(open(f"{REVIEW}/inventory.json")):
        yield ("final", r, n_mb)
    for p in sorted(glob.glob(f"{REVIEW}/early_stage_*.npz")):
        yield ("early", p, None)
    for p in sorted(glob.glob(f"{REVIEW}/final_trainmode/*.npz")):
        yield ("trainmode", p, None)


def run(job):
    kind, x, n_mb = job
    rows = []
    if kind == "final":
        q = np.load(x["latent_path"], allow_pickle=False)
        z, b = q["z"], np.unique(q["batch"], return_inverse=True)[1]
        tr = np.asarray(torch.load(x["model_path"], map_location="cpu", weights_only=False)["attr_dict"]["train_indices_"])
        rng = torch.Generator().manual_seed(80301 + int(x["seed"]))
        for j, idx in enumerate(tr[torch.randperm(len(tr), generator=rng).numpy()[:128 * n_mb]].reshape(-1, 128)):
            rows.append(dict(kind=kind, tag=x["tag"], stage=x["stage"], task=x["task"], lam=x["lam"], cond=x["cond"],
                             zstd=int(x["zstd"]), minibatch=j, **compare(z[idx], b[idx], bool(int(x["zstd"])))))
    elif kind == "early":
        q = np.load(x, allow_pickle=False)
        m = json.loads(str(q["metadata"]))
        row = m["config"]["row"]
        zs = [0, 1] if m["stage"] == "untrained" and row["task"] in ("atac_small", "immune") else [int(row["zstd"])]
        for s in zs:
            rows.append(dict(kind=kind, tag=os.path.basename(x)[:-4], stage=m["stage"], task=row["task"], lam=row["lam"],
                             cond=row["cond"], zstd=s, minibatch=m["batch_idx"], **compare(q["mu"], q["batch"], bool(s))))
    else:
        q = np.load(x, allow_pickle=False)
        m = json.loads(str(q["meta"]))
        for j in range(len(q["z"])):
            rows.append(dict(kind=kind, tag=os.path.basename(x)[:-4], stage="trainmode", task=m.get("task", ""),
                             lam=m.get("lam", ""), cond=m.get("cond", ""), zstd=int(m.get("zstd", 0)), minibatch=j,
                             **compare(q["z"][j], q["batch"][j], bool(int(m.get("zstd", 0))))))
    return rows


if __name__ == "__main__":
    out = sys.argv[1]
    n_mb = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    torch.set_num_threads(1)
    allrows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, rows in enumerate(ex.map(run, list(jobs(n_mb)))):
            allrows += rows
            if i % 25 == 0:
                print(i, len(allrows), flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in allrows for k in r)), restval="")
        w.writeheader()
        w.writerows(allrows)
    print("rows", len(allrows))
