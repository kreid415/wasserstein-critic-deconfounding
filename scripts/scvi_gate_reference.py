#!/usr/bin/env python
"""Stage 1 of the LinearSCVI equivalence gate (runs in scvi-env, real scvi-tools 1.4.2).

Builds a real scvi LDVAE module, runs ONE deterministic forward pass on a fixed synthetic
count matrix, and dumps to disk: the module state_dict, the exact per-cell reconstruction
loss, z-KL, library-KL, and the intermediate tensors (z sample, library sample, px_rate).
Stage 2 (scvi_gate_compare.py, wcd-kbet) transplants these weights into our
LinearSCVIBackbone and asserts the forward numbers match to < 1e-5.

Determinism: fixed torch seed, model in eval() so BatchNorm uses running stats (both start
at running_mean=0/running_var=1, identical), dropout off. We drive inference/generative/loss
DIRECTLY (not .train()) so no optimizer, no data split, no annealing — a pure forward compare.
"""
import os, numpy as np, torch

OUT = os.environ.get("GATE_OUT", "results/scvi_gate")
os.makedirs(OUT, exist_ok=True)
G, N, NB, DZ = 40, 64, 3, 10           # genes, cells, batches, latent
torch.manual_seed(0); np.random.seed(0)

# fixed synthetic raw counts + batch labels
counts = torch.poisson(torch.rand(N, G) * 5.0)
batch = torch.randint(0, NB, (N, 1))

from scvi.module import LDVAE
# scVI stores per-batch library params as shape (1, n_batch): _compute_local_library_params
# does one_hot(batch, n_batch) @ library_log_means, and n_batch = library_log_means.shape[1].
lm = np.zeros((1, NB), dtype="float32"); lv = np.ones((1, NB), dtype="float32")
m = LDVAE(n_input=G, n_batch=NB, n_latent=DZ, dispersion="gene", gene_likelihood="nb",
          log_variational=True, use_observed_lib_size=False,
          library_log_means=lm, library_log_vars=lv)
m.eval()

torch.manual_seed(0)   # reset so the rsample draws are reproducible across stages
with torch.no_grad():
    inf = m.inference(counts, batch)                       # z, qz, ql, library
    gen = m.generative(inf["z"], inf["library"], batch)    # px, pl, pz
    # encoder distribution params (to prove the encoder weights transplant too)
    qz = inf["qz"]; ql = inf["ql"]
    qz_m, qz_v = qz.loc.cpu(), qz.scale.cpu() ** 2
    ql_m, ql_v = ql.loc.cpu(), ql.scale.cpu() ** 2
    x_key = counts
    recon = -gen["px"].log_prob(x_key).sum(-1)             # [N]
    from torch.distributions import kl_divergence
    kl_z = kl_divergence(inf["qz"], gen["pz"]).sum(-1)     # [N]
    kl_l = kl_divergence(inf["ql"], gen["pl"]).sum(-1)     # [N]

torch.save({
    "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
    "counts": counts, "batch": batch,
    "z": inf["z"].cpu(), "library": inf["library"].cpu(),
    "px_rate": gen["px"].mu.cpu(), "px_theta": gen["px"].theta.cpu(),
    "recon": recon.cpu(), "kl_z": kl_z.cpu(), "kl_l": kl_l.cpu(),
    "qz_m": qz_m, "qz_v": qz_v, "ql_m": ql_m, "ql_v": ql_v,
    "library_log_means": torch.tensor(lm), "library_log_vars": torch.tensor(lv),  # (1,NB)
    "dims": dict(G=G, N=N, NB=NB, DZ=DZ),
}, f"{OUT}/scvi_ref.pt")
print(f"[gate ref] wrote {OUT}/scvi_ref.pt")
print(f"  recon[:3]={recon[:3].tolist()}")
print(f"  kl_z[:3]={kl_z[:3].tolist()}  kl_l[:3]={kl_l[:3].tolist()}")
print(f"  state_dict keys: {sorted(m.state_dict().keys())}")
