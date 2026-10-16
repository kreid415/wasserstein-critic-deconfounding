"""Does 'same seed = same initialisation and minibatch order' (docs/PREREG.md sec. 1, Pairing) hold ACROSS arms?
Toy CPU fits with the tagged training plan; each training step records a fingerprint of its minibatch (per-cell
count totals) and the plan records the generator's initial parameters. Compares arm none (the lambda=0 reference
of R1/R4) with the adversarial arms at the same seed."""
import hashlib, json, os, sys
import numpy as np, anndata as ad, torch
REPO = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ["WCD_SRC"] = os.path.join(REPO, "src")
import scvi_adversarial_plan as plan
from scvi import REGISTRY_KEYS

def toy(n=600, g=50, k=3, seed=0):
    r = np.random.default_rng(seed)
    b = np.repeat(np.arange(k), n // k)
    mu = r.gamma(2.0, 1.0, size=(k, g))[b] * (1 + 0.5 * (np.arange(n) % 4)[:, None])
    X = r.poisson(mu).astype(np.float32)
    a = ad.AnnData(X=X.copy())
    a.layers["counts"] = X
    a.obs["batch"] = np.array([f"b{i}" for i in b]); a.obs["batch"] = a.obs["batch"].astype("category")
    a.obs["celltype"] = np.array([f"t{i % 4}" for i in range(n)])
    return a

LOG = {}
class Recording(plan.WassersteinAdversarialTrainingPlan):
    def on_train_start(self):
        super().on_train_start()
        h = hashlib.sha1()
        for p in self.module.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        LOG["init_sha"] = h.hexdigest()[:12]
    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            x = batch[REGISTRY_KEYS.X_KEY].sum(1).detach().cpu().numpy()
            LOG.setdefault("steps", []).append(hashlib.sha1(np.round(x, 3).tobytes()).hexdigest()[:10])
        return super().training_step(batch, batch_idx)
plan.WassersteinAdversarialTrainingPlan = Recording

ARMS = {"none": dict(adversary="none", d_coef=0.0, n_critic=None),
        "discriminator": dict(adversary="discriminator", d_coef=1.0, n_critic=1),
        "reference": dict(adversary="reference", d_coef=1.0, n_critic=5),
        "pooled": dict(adversary="pooled", d_coef=1.0, n_critic=5),
        "mmd": dict(adversary="mmd", d_coef=1.0, n_critic=None),
        "barycenter": dict(adversary="barycenter", d_coef=1.0, n_critic=5)}
res = {}
for arm, kw in ARMS.items():
    for lam in ([0.0] if arm == "none" else [1.0, 10.0]):
        LOG.clear()
        kw2 = dict(kw, d_coef=lam)
        plan.fit_adversarial_scvi(toy(), "batch", reference_batch="b0", adv_input="mean", zstd=False, n_latent=4,
                                  max_epochs=1, batch_size=128, seed=100, conditioned=True, **kw2)
        res[f"{arm}@{lam}"] = dict(init=LOG["init_sha"], steps=list(LOG["steps"]))
ref = res["none@0.0"]
summary = {k: dict(same_init_as_none=v["init"] == ref["init"],
                   first_epoch_minibatches_equal_to_none=sum(a == b for a, b in zip(v["steps"], ref["steps"])),
                   n_steps=len(v["steps"])) for k, v in res.items()}
for arm in ARMS:
    if arm != "none":
        summary[f"{arm}: lambda 1 vs 10 same minibatch order"] = res[f"{arm}@1.0"]["steps"] == res[f"{arm}@10.0"]["steps"]
json.dump(dict(summary=summary, raw=res), open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1))
