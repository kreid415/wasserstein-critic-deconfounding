import scanpy as sc
from scCRAFT.model import obtain_embeddings, train_integration_model
from scCRAFT.utils import multi_resolution_cluster
import torch


def main():
    # set the torch random seed
    torch.manual_seed(42)

    adata = sc.read_h5ad("/workspaces/data/human_pancreas_norm_complexBatch.h5ad")
    # combine all tech with "seq" in the name and all tech with "drop" in the name
    adata.obs["tech"] = adata.obs["tech"].replace(
        to_replace=["celseq2", "celseq", "smartseq2"], value="seq"
    )
    adata.obs["tech"] = adata.obs["tech"].replace(
        to_replace=["inDrop1", "inDrop2", "inDrop3", "inDrop4"], value="drop"
    )
    # remove all non-seq/drop techs
    adata = adata[adata.obs["tech"].isin(["seq", "drop"])].copy()
    # remove all celltypes with less than 500 cells in each tech
    celltype_counts = adata.obs.groupby(["celltype", "tech"]).size()
    valid_celltypes = celltype_counts[celltype_counts >= 500].index.get_level_values(0).unique()
    adata = adata[adata.obs["celltype"].isin(valid_celltypes)].copy()

    adata.raw = adata
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="tech")
    adata = adata[:, adata.var["highly_variable"]]

    multi_resolution_cluster(adata, resolution1=1, method="Leiden")
    vae = train_integration_model(
        adata, batch_key="tech", z_dim=256, d_coef=0.2, epochs=1000, critic=True
    )
    obtain_embeddings(adata, vae.to("cuda:0"))
    sc.pp.neighbors(adata, use_rep="X_scCRAFT")
    sc.tl.umap(adata, min_dist=0.5)
    sc.pl.umap(adata, color=["tech", "celltype"], frameon=False, ncols=1)


if __name__ == "__main__":
    main()
