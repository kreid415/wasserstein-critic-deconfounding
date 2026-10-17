"""Short GPU checks part 2 on atac_small (read-only input):
 (4) adversary-on-samples vs scoring-on-means: posterior noise vs signal, batch decodability on mean vs sample
 (5) z-standardisation (pooled RMS) anisotropy: variance concentration of the latent, with vs without zstd
Deviation for speed: 12 epochs with n_epochs_kl_warmup=2 (KL at full weight from epoch 2), not the stock schedule."""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
REPO = os.path.abspath("review/wcd")
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ["WCD_SRC"] = os.path.join(REPO, "src")
import scvi
import scvi_adversarial_plan as P
PREP = "/home/kendall/experiment_data/wasserstein-critic-deconfounding/pilot_counts/prepped/atac_small_prepped.h5ad"
ad = sc.read_h5ad(PREP).copy()
bk = ad.uns["batch_key"]; REF = str(ad.uns["reference_batch"])
y = ad.obs[bk].astype(str).values
PLANS = []
Orig = P.WassersteinAdversarialTrainingPlan
class Probe(Orig):
    def __init__(self, module, **kw):
        super().__init__(module, **kw); PLANS.append(self)
P.WassersteinAdversarialTrainingPlan = Probe
X = torch.tensor(np.asarray(ad.layers["counts"]), dtype=torch.float32)

def posterior(mod):
    mod = mod.to("cuda").eval(); L, S = [], []
    with torch.no_grad():
        for s in range(0, X.shape[0], 2048):
            xb = X[s:s+2048].cuda()
            qz, _ = mod.z_encoder(torch.log1p(xb), torch.zeros(len(xb), 1, dtype=torch.long, device="cuda"))
            L.append(qz.loc.cpu().numpy()); S.append(qz.scale.cpu().numpy())
    return np.concatenate(L), np.concatenate(S)

def batch_acc(Z):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, Z, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="balanced_accuracy").mean())

def knn_same(Z, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z); idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    return float((y[idx] == y[:, None]).mean())

def spectrum(Z):
    ev = np.sort(np.linalg.eigvalsh(np.cov(Z.T)))[::-1]; ev = np.clip(ev, 0, None)
    # batch eta^2 along each principal axis
    Zc = Z - Z.mean(0); U = np.linalg.eigh(np.cov(Z.T))[1][:, ::-1]; Pz = Zc @ U
    eta = []
    for j in range(Pz.shape[1]):
        tot = Pz[:, j].var(); bw = sum((y == b).mean() * (Pz[y == b, j].mean() - Pz[:, j].mean()) ** 2 for b in np.unique(y))
        eta.append(float(bw / tot) if tot > 0 else 0.0)
    return dict(participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()), top1_share=float(ev[0] / ev.sum()),
                top3_share=float(ev[:3].sum() / ev.sum()), batch_eta2_by_pc=[round(e, 3) for e in eta[:5]],
                pooled_rms=float(np.sqrt((Zc ** 2).mean())))

CONFIGS = [("none", 0.0, 1, False), ("discriminator", 20.0, 1, False), ("reference", 5.0, 10, False),
           ("reference", 30.0, 10, False), ("reference", 30.0, 10, True)]
res = {}; t0 = time.time()
for adv, lam, di, zs in CONFIGS:
    key = f"{adv}{'_zs' if zs else ''}_lam{lam:g}"
    Z = P.fit_adversarial_linearscvi(ad, bk, adversary=adv, d_coef=lam, disc_iter=di, reference_batch=REF, zstd=zs,
                                     n_latent=10, max_epochs=12, batch_size=128, seed=0, conditioned=True,
                                     model_name="SCVI", n_epochs_kl_warmup=2)
    loc, sd = posterior(PLANS[-1].module)
    rng = np.random.default_rng(0); samp = loc + sd * rng.standard_normal(loc.shape)
    res[key] = dict(
        loc_matches_get_latent=float(np.abs(loc - Z).max()),
        mean_posterior_sd=float(sd.mean()), sd_of_means_per_dim_mean=float(loc.std(0).mean()),
        noise_to_signal=float((sd ** 2).mean() / loc.var(0).mean()),
        batch_bal_acc_mean=batch_acc(loc), batch_bal_acc_sample=batch_acc(samp),
        knn15_same_batch_mean=knn_same(loc), knn15_same_batch_sample=knn_same(samp),
        spectrum_mean=spectrum(loc))
    print(key, json.dumps(res[key]), f"{time.time()-t0:.0f}s", flush=True)
P.WassersteinAdversarialTrainingPlan = Orig
res["chance_bal_acc"] = 1.0 / len(np.unique(y)); res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open("checks/gpu_checks_2.json", "w"), indent=1)
print("DONE", res["seconds"], flush=True)
