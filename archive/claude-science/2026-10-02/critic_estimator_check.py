"""Does the multi-head critic estimate what the formulation claims? Fixed synthetic latents (no encoder):
two isotropic Gaussian batches in 10-D, shifted by delta along e1 (true W1 = delta), unequal sizes (78%/22%, as lung).
Train the repo's own critic head (Discriminator + ReferenceWassersteinLoss + multi_class_gradient_penalty) with the
plan's critic optimiser (Adam 1e-3, betas 0.5/0.9) and read off per-head gaps.
Expected from the algebra: reference -> W1 = delta; pooled head k -> (1 - pi_k) * delta; pooled loss (mean over heads) -> delta/2."""
import os, sys, json, torch, numpy as np
REPO = os.path.abspath("review/wcd")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import scvi_adversarial_plan as P
Discriminator, critic_mod, _ = P._load_wcd_heads(os.path.join(REPO, "src"))
torch.manual_seed(0)
d, delta, n = 10, 2.0, 4096
pis = (0.78, 0.22)
n0 = int(pis[0] * n); n1 = n - n0
z = torch.cat([torch.randn(n0, d), torch.randn(n1, d) + delta * torch.eye(d)[0]])
b = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)])
out = {}
for form, ref in (("reference", 0), ("reference", 1), ("pooled", None)):
    torch.manual_seed(1)
    head = Discriminator(n_input=d, domain_number=2, critic=True, reference_batch=ref, formulation=form)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, betas=(0.5, 0.9))
    for it in range(3000):
        idx = torch.randint(0, n, (512,))
        loss, gp = head(z[idx], b[idx], reference_batch=ref)
        opt.zero_grad(); (loss + gp).backward(); opt.step()
    with torch.no_grad():
        o = head(z, None)
        gaps = {}
        for k in range(2):
            if form == "reference" and k == ref: continue
            tgt = o[b == ref, k] if form == "reference" else o[:, k]
            gaps[f"head{k}"] = float(o[b == k, k].mean() - tgt.mean())
        full_loss = -float(np.mean(list(gaps.values())))
    key = f"{form}" + (f"_ref{ref}" if ref is not None else "")
    out[key] = dict(per_head_gap=gaps, objective_mean_over_heads=-full_loss)
    print(key, json.dumps(out[key]), flush=True)
out["truth"] = dict(W1=delta, pooled_expected_head0=(1 - pis[0]) * delta, pooled_expected_head1=(1 - pis[1]) * delta,
                    pooled_expected_objective=delta / 2)
json.dump(out, open("checks/critic_estimator_check.json", "w"), indent=1)
print(out["truth"])
