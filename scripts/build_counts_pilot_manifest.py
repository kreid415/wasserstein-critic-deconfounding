#!/usr/bin/env python
"""Counts pilot manifest: does the critic-vs-discriminator picture survive real counts + a
batch-CONDITIONED decoder (scVI as people use it)?

WHY: the code audit (2026-10-01) found every final-sweep fit trained on log-normalised values
stored as 'counts' (data.py fix C2) with an UNconditioned decoder. Before any full rerun, this
pilot checks on one dataset whether the conclusions move.

Design (fixed before results; no lambda is selected from this run):
  dataset   atac_small (11,270 cells, 3 batches; same cells/genes/order as the old prep)
  decoder   SCVI (nonlinear; scvi-tools default ZINB likelihood)
  cond=1    none (lambda=0 = stock scVI on our schedule) + discriminator / reference / pooled
            at lambda in {1, 5, 20, 100}  (old atac_small peaks: critics ~1, discriminator ~20)
  cond=0    none + discriminator on the same grid (cheap; separates the counts effect from
            the conditioning effect for the discriminator)
  seeds     0, 1, 2
  schedule  239 epochs, batch 512, no early stopping, disc_iter 1 (disc) / 10 (critics),
            identical to the final sweep.
Tags use the XC prefix so they can never collide with final-sweep (XZ) latents.
"""
import itertools, pandas as pd

LAMS = [1, 5, 20, 100]
SEEDS = [0, 1, 2]
DS, MODEL, DEC = "atac_small", "SCVI", "nl"
DI = {"discriminator": 1, "reference": 10, "pooled": 10, "none": 1}

rows = []
def add(cond, adv, lam, seed):
    c = "cond" if cond else "uncond"
    tag = f"{DS}_XC_{DEC}_{c}_{adv}_lam{lam:g}_s{seed}"
    rows.append(dict(model=MODEL, cond=cond, adv=adv, lam=lam, di=DI[adv], seed=seed,
                     dataset=DS, dec=DEC, tag=tag))

for cond in (1, 0):
    for s in SEEDS:
        add(cond, "none", 0, s)
    arms = ["discriminator", "reference", "pooled"] if cond else ["discriminator"]
    for adv, lam, s in itertools.product(arms, LAMS, SEEDS):
        add(cond, adv, lam, s)

M = pd.DataFrame(rows)
assert M.tag.is_unique
out = "scripts/counts_pilot_manifest.tsv"
M.to_csv(out, sep="\t", index=False)
print(out, len(M), "configs |", M.groupby(["cond", "adv"]).size().to_dict())
