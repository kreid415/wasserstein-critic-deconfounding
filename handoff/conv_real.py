"""Real-data convergence on local GPU: pancreas, NB backbone, both heads, 300 epochs."""
import time
import numpy as np, anndata as ad, scanpy as sc, torch
from wcd_vae.wcd.data import prep_data
from wcd_vae.wcd.training import SCIntegrationModel

dev="cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev} torch={torch.__version__}", flush=True)

# pancreas registry metadata: batch=tech, celltype=celltype, 9 batches
A, _largest=prep_data("/home/kendall/.claude-science/orgs/7339da5c-ddcf-4ba9-9b06-df362dd1208a/workspaces/418e2055-dad2-49a2-aee6-bcbf79bc1e49/handoff/realdata/pancreas.h5ad", batch_key="tech", celltype_key="celltype",
            batch_count=9)
print("prepped:", A.shape, "n_batches:", A.obs["tech"].nunique(), flush=True)

for head,crit in [("discriminator",False),("critic",True)]:
    m=SCIntegrationModel(A,"tech",z_dim=256,critic=crit,reference_batch=(0 if crit else None),seed=0,backbone="NB")
    t0=time.time()
    h=m.train_model(A,"tech",epochs=300,d_coef=0.2,kl_coef=0.005,warmup_epoch=10,
                    disc_iter=(10 if crit else 1),batch_size=1024,reference_batch_name_str=None)
    dt=time.time()-t0
    vae=np.array(h["loss_vae"]); adv=np.array(h["loss_da"])
    print(f"\n=== {head} (NB) | {dt/300:.2f}s/epoch {dev} ===", flush=True)
    for e in [25,50,75,100,125,150,175,200,250,299]:
        print(f"  epoch {e:3d}: L_vae={vae[e]:9.3f}  L_adv={adv[e]:8.3f}", flush=True)
    dif=np.abs(np.diff(vae)); j=int(np.argmax(dif))
    print(f"  largest |Δ| at {j}->{j+1}: {vae[j]:.3f}->{vae[j+1]:.3f}; n(|Δ|>0.5)={int((dif>0.5).sum())}; "
          f"last-50 Δ%={abs(vae[-1]-vae[-50])/abs(vae[-50])*100:.2f}", flush=True)
np.savez("handoff/conv_real_pancreas.npz",
         disc_vae=[0], note="see stdout")
print("\nDONE", flush=True)
