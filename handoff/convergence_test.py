"""Convergence test: train NB backbone (both heads) for 300 epochs on immune-scale
synthetic data, record loss trajectories, decide whether 150 epochs suffices."""
import sys, time
import numpy as np, anndata as ad, scanpy as sc, torch
from wcd_vae.wcd.training import SCIntegrationModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

rng=np.random.default_rng(0)
n,g,V=16000,2000,4
ct=rng.integers(0,8,n); b=rng.integers(0,V,n)
X=(rng.poisson(1.5+ct[:,None]%4,size=(n,g))+b[:,None]*rng.poisson(0.5,size=(n,g))).astype(np.float32)
A=ad.AnnData(X); A.obs["batch"]=[f"b{i}" for i in b]; A.obs["celltype"]=[f"c{i}" for i in ct]
A.obs["batch"]=A.obs["batch"].astype("category"); A.obs["celltype"]=A.obs["celltype"].astype("category")
A.layers["counts"]=A.X.copy(); sc.pp.normalize_per_cell(A,counts_per_cell_after=1e4); sc.pp.log1p(A)

EPOCHS=300
hist={}
for head,crit in [("discriminator",False),("critic",True)]:
    m=SCIntegrationModel(A,"batch",z_dim=256,critic=crit,reference_batch=(0 if crit else None),seed=0,backbone="NB")
    t0=time.time()
    h=m.train_model(A,"batch",epochs=EPOCHS,d_coef=0.2,kl_coef=0.005,warmup_epoch=10,
                    disc_iter=(10 if crit else 1),batch_size=1024,
                    reference_batch_name_str=("b0" if crit else None))
    dt=time.time()-t0
    hist[head]=h
    vae=np.array(h["loss_vae"]); adv=np.array(h["loss_da"])
    print(f"\n=== {head} (NB) | {dt/EPOCHS:.2f}s/epoch on {dev} ===", flush=True)
    for e in [25,50,100,150,200,250,299]:
        print(f"  epoch {e:3d}: L_vae={vae[e]:8.2f}  L_adv={adv[e]:7.3f}")
    d150_200=abs(vae[199]-vae[149])/abs(vae[149])*100
    d250_299=abs(vae[299]-vae[249])/abs(vae[249])*100
    print(f"  L_vae Δ% [150->200]={d150_200:.3f}%  [250->300]={d250_299:.3f}%")

# save trajectories + a plot
np.savez("handoff/convergence.npz",
         disc_vae=hist["discriminator"]["loss_vae"], disc_adv=hist["discriminator"]["loss_da"],
         crit_vae=hist["critic"]["loss_vae"], crit_adv=hist["critic"]["loss_da"])
print("\nSAVED handoff/convergence.npz")
