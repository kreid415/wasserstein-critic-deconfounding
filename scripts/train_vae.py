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
    anndata_path = "/workspaces/data/human_pancreas_norm_complexBatch.h5ad"
    anndata = ad.read_h5ad(anndata_path)
    anndata.layers["normalized"] = anndata.X

    # combine all tech with "seq" in the name and all tech with "drop" in the name
    anndata.obs["tech"] = anndata.obs["tech"].replace(
        to_replace=["celseq2", "celseq", "smartseq2"], value="seq"
    )
    anndata.obs["tech"] = anndata.obs["tech"].replace(
        to_replace=["inDrop1", "inDrop2", "inDrop3", "inDrop4"], value="drop"
    )

    # remove all non-seq/drop techs
    anndata = anndata[anndata.obs["tech"].isin(["seq", "drop"])].copy()

    # remove all celltypes with less than 500 cells in each tech
    celltype_counts = anndata.obs.groupby(["celltype", "tech"]).size()
    valid_celltypes = celltype_counts[celltype_counts >= 500].index.get_level_values(0).unique()
    anndata = anndata[anndata.obs["celltype"].isin(valid_celltypes)].copy()

    # Find/subset HVGs & swap to raw counts
    import scanpy as sc

    sc.pp.highly_variable_genes(anndata, n_top_genes=3000, batch_key="tech")
    anndata = anndata[:, anndata.var["highly_variable"]].copy()

    # VAE config (copied from notebook)
    config = VAEConfig(
        input_dim=anndata.shape[1],
        latent_dim=32,
        encoder_hidden_dims=[128, 128],
        decoder_hidden_dims=[128, 128],
        dropout=0.2,
        batchsize=128,
        num_epochs=100_000,
        lr=1e-3,
        weight_decay=1e-5,
        kl_anneal_start=0,
        kl_anneal_end=100,
        kl_anneal_max=1.0,
        use_batchnorm=True,
        zinb=True,
        recon_weight=1,
        variational=True,
        linear_decoder=False,
        num_batches=2,
        learn_lib=False,
        num_pseudo_inputs=32,
        vamprior=True,
    )
    vae = VAE(config)

    # Data loaders
    train_loader, test_loader, domain_encoder, cell_encoder = get_dataloader_from_adata(
        anndata,
        by="tech",
        batch_size=config.batchsize,
        num_workers=0,
        cell_label="celltype",
    )

    # this scripts filename
    script_name = os.path.basename(__file__).split(".")[0]

    # add checkpint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_recon_loss",
        mode="min",
        save_top_k=1,
        filename=f"{script_name}" + "-{epoch:02d}-v-{val_recon_loss:.2f}",
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
