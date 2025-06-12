import os

import anndata as ad
import pytorch_lightning as pl
import scanpy as sc

from wcd_vae.data import get_dataloader_from_adata
from wcd_vae.model import VAE, VAEConfig


def main():
    # Set seeds for reproducibility
    pl.seed_everything(42)

    # Load the anndata object (adjust path if needed)
    anndata_path = "./data/immune.h5ad"
    anndata = ad.read_h5ad(anndata_path)
    anndata.layers["normalized"] = anndata.X

    anndata = anndata[anndata.obs["batch"].isin(["MCA_BM_2", "MCA_BM_3"])].copy()

    # Find/subset HVGs & swap to raw counts
    import scanpy as sc

    sc.pp.highly_variable_genes(anndata, n_top_genes=3000, batch_key="batch")
    anndata = anndata[:, anndata.var["highly_variable"]].copy()
    anndata.X = anndata.layers["counts"]

    # VAE config (copied from notebook)
    config = VAEConfig(
        input_dim=anndata.shape[1],
        latent_dim=32,
        encoder_hidden_dims=[128, 128],
        decoder_hidden_dims=[128, 128],
        dropout=0.2,
        batchsize=128,
        num_epochs=1_000,
        lr=1e-3,
        weight_decay=1e-5,
        kl_anneal_start=0,
        kl_anneal_end=100,
        kl_anneal_max=1.0,
        use_batchnorm=True,
        zinb=True,
        recon_weight=1,
        variational=True,
    )
    vae = VAE(config, linear_decoder=True)

    # Data loaders
    train_loader, test_loader, domain_encoder, cell_encoder = get_dataloader_from_adata(
        anndata,
        by="batch",
        batch_size=config.batchsize,
        num_workers=0,
        cell_label="final_annotation",
    )

    # this scripts filename
    script_name = os.path.basename(__file__).split(".")[0]

    # add checkpint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
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

    trainer.fit(vae, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
