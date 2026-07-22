"""One-off: real-data convergence check. Train NB backbone (both heads) for N epochs on
a registered dataset and print the L_vae / L_adv trajectory so we can pick a stable epoch
count. Uses the real prep_data pipeline (HVG, counts, log1p)."""
import argparse
import json

import numpy as np

from wcd_vae.wcd.experiment import load_task
from wcd_vae.wcd.training import SCIntegrationModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.registry) as fh:
        registry = json.load(fh)
    adata, batch_key, _celltype_key, _ = load_task(
        args.dataset, data_root=args.data_root, registry=registry,
    )
    print(f"[{args.dataset}] n_obs={adata.n_obs} n_batches={adata.obs[batch_key].nunique()}",
          flush=True)

    out = {}
    for head, crit in [("discriminator", False), ("critic", True)]:
        m = SCIntegrationModel(adata, batch_key, z_dim=256, critic=crit,
                               reference_batch=(0 if crit else None), seed=0, backbone="NB")
        h = m.train_model(adata, batch_key, epochs=args.epochs, d_coef=0.2, kl_coef=0.005,
                          warmup_epoch=10, disc_iter=(10 if crit else 1), batch_size=1024,
                          reference_batch_name_str=None)
        vae = np.array(h["loss_vae"])
        adv = np.array(h["loss_da"])
        out[head] = {"vae": vae.tolist(), "adv": adv.tolist()}
        print(f"\n=== {head} (NB) ===", flush=True)
        for e in [25, 50, 75, 100, 125, 150, 175, 200, 250, args.epochs - 1]:
            if e < len(vae):
                print(f"  epoch {e:3d}: L_vae={vae[e]:9.3f}  L_adv={adv[e]:8.3f}", flush=True)
        # largest single-step jump
        dif = np.abs(np.diff(vae))
        j = int(np.argmax(dif))
        print(f"  largest |Δ| at {j}->{j+1}: {vae[j]:.3f}->{vae[j+1]:.3f}; "
              f"n(|Δ|>0.5)={int((dif>0.5).sum())}; last-50 Δ%="
              f"{abs(vae[-1]-vae[-50])/abs(vae[-50])*100:.2f}", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh)
        print(f"\nSAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
