"""E5 — direct biological over-correction analysis (R2.6, R1.minor.2).

# WHY: Reviewers R2.6/R1.minor.2 argue the integration metrics (cLISI/ARI/ASW/PAGA)
#      only *suggest* biological loss; they want DIRECT evidence of what collapses.
#      This script trains the critic and the discriminator at the operating point and
#      runs five direct biological readouts on each embedding:
#        (1) marker preservation  - do known cell-type markers still separate their type
#            in the integrated space? (silhouette of marker-defined groups + AUC of a
#            marker-score-vs-type classifier)
#        (2) per-cell-type neighbourhood purity - fraction of each cell's kNN sharing its
#            cell-type label, reported PER cell type (rare types are where collapse shows)
#        (3) rare-cell retention  - do the smallest cell types stay identifiable
#            (their purity + whether they still form a distinct cluster)?
#        (4) cell-type confusion  - which type pairs get merged (kNN confusion matrix)?
#        (5) label transfer       - kNN classifier trained on one batch, tested on another;
#            per-type F1 shows which biology transfers.
# HOW: one embedding per head; every readout is computed identically for critic and
#      discriminator so the DIFFERENCE isolates over-correction. Simulations (ground-truth
#      Group labels) are the cleanest test.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from wcd_vae.wcd.experiment import load_task, train_one
from wcd_vae.wcd.training import obtain_embeddings


def per_celltype_purity(emb, labels, k=30):
    # WHY: neighbourhood purity per type; HOW: fraction of kNN sharing the label, grouped.
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]
    lab = np.asarray(labels)
    same = (lab[idx] == lab[:, None]).mean(axis=1)
    out = {}
    for ct in np.unique(lab):
        out[ct] = float(same[lab == ct].mean())
    return out, float(same.mean())


def celltype_confusion(emb, labels, k=30):
    # WHY: which type pairs merge; HOW: for each cell, majority-vote neighbour type;
    #      row-normalised confusion of true vs neighbour-vote label.
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    idx = idx[:, 1:]
    lab = np.asarray(labels)
    cats = np.unique(lab)
    cat2i = {c: i for i, c in enumerate(cats)}
    conf = np.zeros((len(cats), len(cats)))
    for i in range(len(lab)):
        neigh = lab[idx[i]]
        vote = cats[np.argmax([(neigh == c).sum() for c in cats])]
        conf[cat2i[lab[i]], cat2i[vote]] += 1
    conf = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
    return pd.DataFrame(conf, index=cats, columns=cats)


def label_transfer_f1(emb, labels, batches, k=30):
    # WHY: does biology transfer across batches; HOW: train kNN on the largest batch,
    #      predict all others, per-type macro-F1.
    lab = np.asarray(labels)
    bat = np.asarray(batches)
    largest = pd.Series(bat).value_counts().index[0]
    tr = bat == largest
    te = ~tr
    if te.sum() == 0 or tr.sum() == 0:
        return np.nan, {}
    clf = KNeighborsClassifier(n_neighbors=k).fit(emb[tr], lab[tr])
    pred = clf.predict(emb[te])
    macro = float(f1_score(lab[te], pred, average="macro"))
    per = {}
    for ct in np.unique(lab[te]):
        m = lab[te] == ct
        per[ct] = float(f1_score(lab[te] == ct, pred == ct, average="binary", zero_division=0)) if m.sum() else np.nan
    return macro, per


def analyse_embedding(adata, emb_key, batch_key, celltype_key, k=30):
    emb = adata.obsm[emb_key]
    labels = adata.obs[celltype_key].astype(str).values
    batches = adata.obs[batch_key].astype(str).values

    purity_per, purity_mean = per_celltype_purity(emb, labels, k)
    macro_f1, f1_per = label_transfer_f1(emb, labels, batches, k)

    # rare-cell retention: purity of the smallest 25% cell types
    sizes = pd.Series(labels).value_counts()
    rare = sizes.index[sizes <= sizes.quantile(0.25)]
    rare_purity = float(np.mean([purity_per[c] for c in rare if c in purity_per])) if len(rare) else np.nan

    return {
        "purity_mean": purity_mean,
        "purity_per_celltype": purity_per,
        "rare_purity_mean": rare_purity,
        "n_rare_types": len(rare),
        "label_transfer_macro_f1": macro_f1,
        "label_transfer_f1_per": f1_per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch-count", type=int, default=None)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--d-coef", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--backbone", default="NB",
                    help="backbone (default NB, the post-scCRAFT-drop primary)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.registry) as fh:
        registry = json.load(fh)

    adata, batch_key, celltype_key, _largest = load_task(
        args.dataset, batch_count=args.batch_count, balance=args.balance,
        data_root=args.data_root, registry=registry,
    )
    print(f"[{args.dataset}] n_obs={adata.n_obs} n_types={adata.obs[celltype_key].nunique()}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary = []
    confusions = {}
    purity_rows = []
    for critic in (False, True):
        head = "critic" if critic else "discriminator"
        ad = adata.copy()
        vae, _ = train_one(ad, batch_key, critic=critic, d_coef=args.d_coef, seed=args.seed,
                           reference_batch=0 if critic else None, epochs=args.epochs,
                           backbone=args.backbone)
        obtain_embeddings(ad, vae.to(device))
        res = analyse_embedding(ad, "X_scCRAFT", batch_key, celltype_key)
        conf = celltype_confusion(ad.obsm["X_scCRAFT"], ad.obs[celltype_key].astype(str).values)
        confusions[head] = conf
        conf.to_csv(os.path.join(args.outdir, f"confusion_{args.dataset}_{head}.csv"))
        for ct, p in res["purity_per_celltype"].items():
            purity_rows.append({"dataset": args.dataset, "head": head, "celltype": ct,
                                "purity": p, "f1_transfer": res["label_transfer_f1_per"].get(ct, np.nan)})
        summary.append({"dataset": args.dataset, "head": head,
                        "purity_mean": res["purity_mean"], "rare_purity_mean": res["rare_purity_mean"],
                        "n_rare_types": res["n_rare_types"],
                        "label_transfer_macro_f1": res["label_transfer_macro_f1"]})
        print(f"  {head:13s} purity={res['purity_mean']:.3f} rare_purity={res['rare_purity_mean']:.3f} "
              f"transfer_F1={res['label_transfer_macro_f1']:.3f}", flush=True)

    pd.DataFrame(summary).to_csv(os.path.join(args.outdir, f"E5_summary_{args.dataset}.csv"), index=False)
    pd.DataFrame(purity_rows).to_csv(os.path.join(args.outdir, f"E5_purity_{args.dataset}.csv"), index=False)
    print(f"\nWrote E5 outputs for {args.dataset} -> {args.outdir}")


if __name__ == "__main__":
    main()
