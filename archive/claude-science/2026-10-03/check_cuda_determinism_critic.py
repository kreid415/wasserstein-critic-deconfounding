"""Is the critic loss + gradient penalty backward pass bitwise reproducible on CUDA for identical inputs and RNG
state? (index_add_ on CUDA tensors is documented as nondeterministic unless torch.use_deterministic_algorithms is on.)
Tiny workload: one 128-cell minibatch, 10-d latent, 6 batches; < 50 MB of GPU memory."""
import json, os, sys, types
import torch, torch.nn as nn, torch.nn.functional as F
REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ["WCD_SRC"] = os.path.join(REPO, "src")
import scvi_adversarial_plan as plan
Discriminator, critic_mod, align_mod, bary_mod = plan._load_wcd_heads(os.environ["WCD_SRC"])
dev = torch.device("cuda")
g = torch.Generator().manual_seed(0)
z = torch.randn(128, 10, generator=g).to(dev)
b = torch.randint(0, 6, (128,), generator=g).to(dev)
res = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
for form in ("reference", "pooled"):
    torch.manual_seed(1)
    head = Discriminator(10, 6, critic=True, reference_batch=(0 if form == "reference" else None),
                         formulation=form, n_hidden=128).to(dev)
    sig = []
    for rep in range(30):
        head.zero_grad(set_to_none=True)
        torch.manual_seed(123); torch.cuda.manual_seed_all(123)       # identical GP draws every repetition
        loss, gp = head(z, b, reference_batch=(0 if form == "reference" else None), with_gp=True)
        (loss + gp).backward()
        flat = torch.cat([p.grad.reshape(-1) for p in head.parameters()]).double().cpu()
        sig.append((float(loss), float(gp), flat))
    ref = sig[0]
    res[form] = dict(loss_identical=all(s[0] == ref[0] for s in sig), gp_identical=all(s[1] == ref[1] for s in sig),
                     grad_bitwise_identical=all(torch.equal(s[2], ref[2]) for s in sig),
                     max_abs_grad_diff=float(max((s[2] - ref[2]).abs().max() for s in sig)))
# MMD / Sinkhorn (critic-free) forward+backward
for name, fn in (("mmd", align_mod.mmd_batch_others), ("sinkhorn", align_mod.sinkhorn_batch_others)):
    outs = []
    for rep in range(30):
        zz = z.clone().requires_grad_(True)
        v = fn(zz, b); v.backward()
        outs.append((float(v), zz.grad.double().cpu()))
    res[name] = dict(value_identical=all(o[0] == outs[0][0] for o in outs),
                     grad_bitwise_identical=all(torch.equal(o[1], outs[0][1]) for o in outs))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
