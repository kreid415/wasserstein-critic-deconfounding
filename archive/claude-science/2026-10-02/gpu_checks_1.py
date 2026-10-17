"""Short GPU checks on atac_small (read-only input; nothing written outside ./checks):
 (1) do SCVI / LinearSCVI use the labels slot with default settings? (stock fit with labels_key=batch vs none)
 (1b) is adversary='none' through the custom plan bit-identical to stock SCVI (conditioned, production config)?
 (2) GPU run-to-run determinism of adversarial arms at a fixed seed
 (3) barycenter anchor dynamics vs lambda (Adam eps=0.01 floor on the generator optimizer)"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, scanpy as sc
REPO = os.path.abspath("review/wcd")
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ["WCD_SRC"] = os.path.join(REPO, "src")
import scvi
import scvi_adversarial_plan as P
from scvi.model import SCVI, LinearSCVI
PREP = "/home/kendall/experiment_data/wasserstein-critic-deconfounding/pilot_counts/prepped/atac_small_prepped.h5ad"
ad = sc.read_h5ad(PREP)
ad = ad.copy()
bk = ad.uns["batch_key"]; REF = str(ad.uns["reference_batch"])
res = {"labels_identity": {}, "none_vs_stock": {}, "determinism": {}, "barycenter": {}}
t0 = time.time()

def stock(cls, setup_kw, seed=0, ep=2):
    scvi.settings.seed = seed
    a = ad.copy(); a.X = a.layers["counts"].copy()
    cls.setup_anndata(a, **setup_kw)
    m = cls(a)
    m.train(max_epochs=ep, batch_size=128, enable_progress_bar=False)
    return m.get_latent_representation()

# (1) labels slot: unconditioned model with labels_key=batch vs no labels at all
for cls in (SCVI, LinearSCVI):
    z_lab = stock(cls, dict(labels_key=bk)); z_nol = stock(cls, {})
    res["labels_identity"][cls.__name__] = float(np.abs(z_lab - z_nol).max())
    print("labels", cls.__name__, res["labels_identity"][cls.__name__], f"{time.time()-t0:.0f}s", flush=True)

# (1b) adversary='none' via the custom plan vs stock SCVI, conditioned (production backbone)
z_plan = P.fit_adversarial_linearscvi(ad, bk, adversary="none", n_latent=10, max_epochs=2, batch_size=128,
                                      seed=0, conditioned=True, model_name="SCVI")
z_stock = stock(SCVI, dict(batch_key=bk))
res["none_vs_stock"]["SCVI_cond"] = float(np.abs(z_plan - z_stock).max())
print("none_vs_stock", res["none_vs_stock"], flush=True)

# (2) determinism at fixed seed
for adv, di in (("reference", 10), ("discriminator", 1)):
    zs = [P.fit_adversarial_linearscvi(ad, bk, adversary=adv, d_coef=1.0, disc_iter=di, reference_batch=REF,
                                       n_latent=10, max_epochs=2, batch_size=128, seed=0, conditioned=True,
                                       model_name="SCVI") for _ in range(2)]
    res["determinism"][adv] = float(np.abs(zs[0] - zs[1]).max())
    print("determinism", adv, res["determinism"][adv], f"{time.time()-t0:.0f}s", flush=True)

# (3) barycenter anchors
PLANS = []
Orig = P.WassersteinAdversarialTrainingPlan
class Probe(Orig):
    def __init__(self, module, **kw):
        super().__init__(module, **kw); self.stats = []; PLANS.append(self)
    def training_step(self, batch, batch_idx):
        h = self._wcd_head
        a0 = h.anchors.detach().clone() if (h is not None and getattr(h, "anchors", None) is not None) else None
        out = super().training_step(batch, batch_idx)
        if a0 is not None and batch_idx % 15 == 0:
            a = h.anchors; g = a.grad
            self.stats.append(dict(epoch=int(self.current_epoch),
                anchor_rms=float(a.detach().pow(2).mean().sqrt()),
                anchor_centroid_norm=float(a.detach().mean(0).norm()),
                anchor_step_abs_mean=float((a.detach() - a0).abs().mean()),
                anchor_grad_abs_mean=(float(g.abs().mean()) if g is not None else None)))
        return out
P.WassersteinAdversarialTrainingPlan = Probe
for lam in (0.01, 1.0, 30.0):
    Z = P.fit_adversarial_linearscvi(ad, bk, adversary="barycenter", d_coef=lam, disc_iter=10, n_latent=10,
                                     max_epochs=6, batch_size=128, seed=0, conditioned=True, model_name="SCVI")
    pl = PLANS[-1]; A = pl._wcd_head.anchors.detach().cpu().numpy()
    Zc = Z - Z.mean(0)
    st = pl.stats
    res["barycenter"][str(lam)] = dict(
        z_rms=float(np.sqrt((Zc ** 2).mean())), z_centroid_norm=float(np.linalg.norm(Z.mean(0))),
        anchor_rms_final=float(np.sqrt((A ** 2).mean())),
        anchor_spread_final=float(np.sqrt(((A - A.mean(0)) ** 2).mean())),
        anchor_to_z_centroid=float(np.linalg.norm(A.mean(0) - Z.mean(0))),
        anchor_grad_abs_mean_median=float(np.median([s["anchor_grad_abs_mean"] for s in st if s["anchor_grad_abs_mean"] is not None])),
        anchor_step_abs_mean_median=float(np.median([s["anchor_step_abs_mean"] for s in st])),
        n_logged=len(st), trace=st[::6])
    print("barycenter", lam, {k: v for k, v in res["barycenter"][str(lam)].items() if k != "trace"}, f"{time.time()-t0:.0f}s", flush=True)
P.WassersteinAdversarialTrainingPlan = Orig
res["seconds"] = round(time.time() - t0, 1)
res["versions"] = dict(scvi=scvi.__version__, torch=torch.__version__, cuda=torch.cuda.is_available(),
                       gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
json.dump(res, open("checks/gpu_checks_1.json", "w"), indent=1)
print("DONE", res["seconds"], flush=True)
