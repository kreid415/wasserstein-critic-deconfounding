"""Choice of SINKHORN_TOL: accuracy (vs the converged float64 reference) and iterations at the final eps of the branch
solver at tolerances 1e-3..1e-6 (cap 200,000), on the minibatches of validation_12mb.csv that are standardised (zstd=1)
or needed more than 100 iterations at tol 1e-6 or did not converge there. Usage: python tol_sweep.py VALIDATION.csv OUT.csv
"""
import csv, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # PYTHONSAFEPATH=1 in this sandbox
import numpy as np
import pandas as pd
import torch
import validate_sinkhorn as V  # noqa: E402

TOLS = (1e-3, 1e-4, 1e-5, 1e-6)
CAP = 200000


def minibatch(row):
    if row["kind"] == "final":
        r = next(x for x in json.load(open(f"{V.REVIEW}/inventory.json")) if x["tag"] == row["tag"])
        q = np.load(r["latent_path"], allow_pickle=False)
        z, b = q["z"], np.unique(q["batch"], return_inverse=True)[1]
        tr = np.asarray(torch.load(r["model_path"], map_location="cpu", weights_only=False)["attr_dict"]["train_indices_"])
        rng = torch.Generator().manual_seed(80301 + int(r["seed"]))
        idx = tr[torch.randperm(len(tr), generator=rng).numpy()[:128 * 12]].reshape(-1, 128)[int(row["minibatch"])]
        return z[idx], b[idx]
    if row["kind"] == "early":
        q = np.load(f"{V.REVIEW}/{row['tag']}.npz", allow_pickle=False)
        return q["mu"], q["batch"]
    q = np.load(f"{V.REVIEW}/final_trainmode/{row['tag']}.npz", allow_pickle=False)
    return q["z"][int(row["minibatch"])], q["batch"][int(row["minibatch"])]


def run(row):
    z, b = minibatch(row)
    zstd = bool(int(row["zstd"]))
    bt = torch.as_tensor(b, dtype=torch.long)
    m64 = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    a64 = V._adv(m64, zstd)
    vr, _ = V.reference_cost(V.cost_matrix(a64), bt, 0.05 * math.sqrt(2 * a64.shape[1]), tol=1e-10)
    gr = torch.autograd.grad(vr, m64)[0].numpy().ravel()
    out = []
    for tol in TOLS:
        V.NEW.sinkhorn_divergence_weighted.__defaults__ = (1, V.NEW.SINKHORN_SCALING, tol, CAP)
        V.ITERS.clear()
        orig = V.NEW._ot_eps, V.NEW._ot_eps_sym
        V.NEW._ot_eps = lambda *a, _f=orig[0]: (V.ITERS.append(0), _f(*a))[1]
        V.NEW._ot_eps_sym = lambda *a, _f=orig[1]: (V.ITERS.append(0), _f(*a))[1]
        mu = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        try:
            v = V.NEW.sinkhorn_batch_others(V._adv(mu, zstd), bt)
        finally:
            V.NEW._ot_eps, V.NEW._ot_eps_sym = orig
        g = torch.autograd.grad(v, mu)[0].double().numpy().ravel()
        ng, nr = np.linalg.norm(g), np.linalg.norm(gr)
        out.append(dict(kind=row["kind"], tag=row["tag"], minibatch=row["minibatch"], zstd=int(zstd), tol=tol,
                        iters_cross=V.ITERS[0], iters_self=max(V.ITERS[1:]),
                        cosine=float(g @ gr / (ng * nr)), norm_ratio=float(ng / nr),
                        rel_grad_err=float(np.linalg.norm(g - gr) / nr),
                        value_rel_err=(float(v.detach()) - float(vr.detach())) / abs(float(vr.detach()))))
    return out


if __name__ == "__main__":
    val = pd.read_csv(sys.argv[1])
    it = val[["new_iters_cross", "new_iters_self_a", "new_iters_self_b"]].max(axis=1)
    sel = val[(val.zstd == 1) | (it > 100) | val.new_error.fillna("").ne("")]
    torch.set_num_threads(1)
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(run, sel.to_dict("records"))):
            rows += r
            if i % 50 == 0:
                print(i, len(sel), flush=True)
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("rows", len(rows))
