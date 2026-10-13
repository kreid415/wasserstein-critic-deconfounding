"""Review F1 (stop the Sinkhorn solve at a NaN marginal error): which marginal errors do non-finite and very large
latents give? 128 cells x 10 dims, 2 batches, seed 0; cases N(0, s^2 I) for s in 1..30,000 and N(0, I) with one
coordinate NaN, +inf or -inf; solve dtype float64 and float32; max_iter 3,000 per solve (CPU, one thread).
Per case: outcome (ok / cap = RuntimeError), final-eps iterations, and how many errors were inf or NaN, and how many
were inf while both potentials were finite. Usage: python f1_nonfinite_probe.py > results/f1_nonfinite_probe.txt
"""
import importlib.util, math, os
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("al", f"{REPO}/src/wcd_vae/wcd/alignment.py")
al = importlib.util.module_from_spec(spec)
spec.loader.exec_module(al)
log = []
_err = al._marginal_err


def _rec(log_w, h_old, h_new, eps):
    e = _err(log_w, h_old, h_new, eps)
    log.append((float(e), bool(torch.isfinite(h_new).all() and torch.isfinite(h_old).all())))
    return e


al._marginal_err = _rec
torch.set_num_threads(1)
g = torch.Generator().manual_seed(0)
cases = {f"s={s:g}": s * torch.randn(128, 10, generator=g) for s in (1, 30, 100, 300, 1e3, 1e4, 3e4)}
for bad in (math.nan, math.inf, -math.inf):
    z = torch.randn(128, 10, generator=g)
    z[5, 3] = bad
    cases[f"one {bad}"] = z
b = torch.randint(0, 2, (128,), generator=g)
for name, z in cases.items():
    for dt in (torch.float64, torch.float32):
        log.clear()
        zd = z.to(dt)
        A, B, _ = al._batch_masks(b, zd)
        try:
            v = al.sinkhorn_divergence_weighted(al._cost(zd, 1), A, B, 0.05 * math.sqrt(20), max_iter=3000)
            st = "ok " + ("nan" if torch.isnan(v).any() else "finite")
        except RuntimeError:
            st = "cap"
        n_inf = sum(math.isinf(e) for e, _ in log)
        n_nan = sum(math.isnan(e) for e, _ in log)
        inf_fin = sum(math.isinf(e) and fin for e, fin in log)
        print(f"{name:10s} {str(dt)[6:]:8s} {st:10s} iters={len(log):5d} err_inf={n_inf:4d} err_nan={n_nan:4d} "
              f"inf_with_finite_potentials={inf_fin:4d}")
