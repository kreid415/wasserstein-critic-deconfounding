import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import scanpy as sc
import anndata as ad

from wcd_vae.data import get_dataloader_from_adata
from wcd_vae.model import VAE, VAEDiscriminator, VAEConfig, Discriminator
from wcd_vae.metrics import compute_metrics


def main():
    # Set seeds for reproducibility
    pl.seed_everything(42)

    # Load the data
    anndata_path = "data/vu_2022_ay_wh.h5ad"
    anndata = ad.read_h5ad(anndata_path)
    anndata.layers["normalized"] = anndata.X

    # Find/subset HVGs & swap to raw counts
    sc.pp.highly_variable_genes(anndata, n_top_genes=3000, batch_key="sample")
    sc.pl.highly_variable_genes(anndata)
    anndata = anndata[:, anndata.var["highly_variable"]].copy()
    # anndata.X = anndata.layers["counts"]

    # Data sanity checks
    X = anndata.X
    if not isinstance(X, np.ndarray):
        X = X.A if hasattr(X, "A") else X.toarray()
    X = X.astype(np.float64)
    print("Checking for NaNs in anndata.X:", np.isnan(X).any())
    print("Checking for infinite values in anndata.X:", np.isinf(X).any())
    print("Max value in anndata.X:", np.max(X))
    print("Min value in anndata.X:", np.min(X))

    if "counts" in anndata.layers:
        counts = anndata.layers["counts"]
        if not isinstance(counts, np.ndarray):
            counts = counts.A if hasattr(counts, "A") else counts.toarray()
        counts = counts.astype(np.float64)
        print("Checking for NaNs in anndata.layers['counts']:", np.isnan(counts).any())
        print("Checking for infinite values in anndata.layers['counts']:", np.isinf(counts).any())
        print("Max value in anndata.layers['counts']:", np.max(counts))
        print("Min value in anndata.layers['counts']:", np.min(counts))

    print(anndata.obs["age"].value_counts(normalize=True))
    print(anndata.obs["age"].value_counts(normalize=False))

    # VAE config (copied from notebook)
    config = VAEConfig(
        input_dim=anndata.shape[1],
        latent_dim=128,
        encoder_hidden_dims=[512, 256],
        decoder_hidden_dims=[256, 512],
        dropout=0.2,
        batchsize=128,
        num_epochs=100_000,
        lr=1e-4,
        weight_decay=1e-5,
        kl_anneal_start=0,
        kl_anneal_end=100,
        kl_anneal_max=1,
        decon_weight=1,
    )
    vae = VAE(config, linear_decoder=True)

    critic = Discriminator(config.latent_dim, critic=False)

    vae_discriminator = VAEDiscriminator(vae, critic, lambda_critic=config.decon_weight)

    # Data loaders
    train_loader, test_loader, domain_encoder, cell_encoder = get_dataloader_from_adata(
        anndata, by="age", batch_size=config.batchsize, num_workers=0
    )

    # this scripts filename
    script_name = os.path.basename(__file__).split(".")[0]

    # add checkpint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_vae_loss",
        mode="min",
        save_top_k=1,
        filename=f"{script_name}" + "-{epoch:02d}-{val_loss:.2f}",
        dirpath="checkpoints/",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(vae_discriminator, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
