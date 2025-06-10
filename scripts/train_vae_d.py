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
from wcd_vae.model import VAEDiscriminatorAdv, VAEConfig
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
    anndata = anndata[:, anndata.var["highly_variable"]]
    anndata.X = anndata.layers["counts"]

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
        num_epochs=10000,
        lr=1e-4,
        weight_decay=1e-5,
        kl_anneal_start=0,
        kl_anneal_end=100,
        kl_anneal_max=1,
    )
    vae = VAEDiscriminatorAdv(config, linear_decoder=True)

    # Data loaders
    train_loader, test_loader, domain_encoder, cell_encoder = get_dataloader_from_adata(
        anndata, by="age", batch_size=config.batchsize, num_workers=0
    )

    # this scripts filename
    script_name = os.path.basename(__file__).split(".")[0]

    # add checkpint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{script_name}-{epoch:02d}-{val_loss:.2f}",
        dirpath="checkpoints/",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(vae, train_dataloaders=train_loader, val_dataloaders=test_loader)

    # Plot the loss
    log_dir = sorted(glob.glob("lightning_logs/version_*"), key=lambda x: int(x.split("_")[-1]))
    if log_dir:
        log_dir = log_dir[-1]
    else:
        log_dir = None

    print("Using log directory:", log_dir)
    metrics_path = os.path.join(log_dir, "metrics.csv")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.dropna(subset=["train_loss_step", "val_loss_step"], how="all")

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["step"], metrics["train_loss_step"], label="Train Loss")
    plt.plot(metrics["step"], metrics["val_loss_step"], label="Val Loss")
    plt.plot(metrics["step"], metrics["kl_weight_step"], label="KL Weight")
    plt.plot(metrics["step"], metrics["kl_div_step"] * metrics["kl_weight_step"], label="KL Div")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.ylim([0, 1000])
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.close()

    # Compute embeddings and metrics
    from tqdm import tqdm

    embeddings = []
    batches = []
    cell_type = []

    vae.eval()
    vae = vae.to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating VAE"):
            x, batch_label, cell_label = batch
            x, batch_label, cell_label = (
                x.to(vae.device),
                batch_label.to(vae.device),
                cell_label.to(vae.device),
            )
            recon_batch, embed, mu, logvar = vae(x)
            embeddings.append(embed.cpu())
            batches.append(batch_label.cpu())
            cell_type.append(cell_label.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    batches = torch.cat(batches, dim=0)
    cell_type = torch.cat(cell_type, dim=0)

    metrics_dict = compute_metrics(
        embeddings=embeddings,
        batch_labels=batches,
        cell_type_labels=cell_type,
    )
    print("Evaluation metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v}")

    # UMAP visualization
    import umap
    import seaborn as sns

    embeddings_np = embeddings.numpy()
    batches_np = batches.argmax(dim=1).numpy()
    cell_type_np = cell_type.argmax(dim=1).numpy()

    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = umap_model.fit_transform(embeddings_np)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=batches_np, palette="tab10", s=10
    )
    plt.title("UMAP colored by Batch")
    plt.legend(title="Batch", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("umap_by_batch.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=cell_type_np, palette="tab20", s=10
    )
    plt.title("UMAP colored by Cell Type")
    plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("umap_by_celltype.png")
    plt.close()


if __name__ == "__main__":
    main()
