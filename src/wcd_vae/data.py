import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.utils.data as utils

from wcd_vae.scCRAFT.utils import multi_resolution_cluster


def get_dataloader_from_adata(
    adata_concat,
    by: str,
    cell_label: str,
    test_size=0.2,
    batch_size: int = 128,
):
    # transform expression data into tensor
    data = adata_concat.X.toarray() if hasattr(adata_concat.X, "toarray") else adata_concat.X
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # make domain labels
    domain_encoder = OneHotEncoder(sparse_output=False)
    domain_labels = domain_encoder.fit_transform(adata_concat.obs[by].to_numpy().reshape(-1, 1))
    domain_labels_tensor = torch.tensor(domain_labels, dtype=torch.float32)

    cell_encoder = OneHotEncoder(sparse_output=False)

    obs = (
        adata_concat.obs[cell_label].to_numpy()
        if hasattr(adata_concat.obs[cell_label], "to_numpy")
        else adata_concat.obs[cell_label]
    )
    cell_labels = cell_encoder.fit_transform(obs.reshape(-1, 1))
    cell_labels_tensor = torch.tensor(cell_labels, dtype=torch.float32)

    dataset = utils.TensorDataset(
        data_tensor.float(), domain_labels_tensor.float(), cell_labels_tensor.float()
    )

    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=42)
    train_loader, test_loader = (
        utils.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
        ),
        utils.DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )

    return train_loader, test_loader, domain_encoder, cell_encoder


import scanpy as sc
import numpy as np
# Ensure multi_resolution_cluster is imported


def prep_data(
    anndata_path,
    batch_key,
    celltype_key,
    batch_count=2,
    min_genes=300,
    min_cells=5,
    norm_val=1e4,
    n_top_genes=2000,
    balance=False,
):
    adata = sc.read_h5ad(anndata_path)

    # 1. Initial selection: Keep top 'batch_count' largest batches
    top_batches = adata.obs[batch_key].value_counts().index[:batch_count]
    adata = adata[adata.obs[batch_key].isin(top_batches)].copy()

    # 2. Balancing (using the robust intersection method discussed previously)
    if balance:
        remaining_batches = adata.obs[batch_key].unique()
        if len(remaining_batches) > 0:
            # Start with first batch's types
            first_batch_name = remaining_batches[0]
            common_celltypes_set = set(
                adata[adata.obs[batch_key] == first_batch_name].obs[celltype_key].unique()
            )
            # Intersect with all others
            for batch_name in remaining_batches[1:]:
                current_batch_celltypes = set(
                    adata[adata.obs[batch_key] == batch_name].obs[celltype_key].unique()
                )
                common_celltypes_set.intersection_update(current_batch_celltypes)

            if not common_celltypes_set:
                raise ValueError("Balancing failed: No common cell types found.")

            # Filter adata to keep only common types
            adata = adata[adata.obs[celltype_key].isin(list(common_celltypes_set))].copy()
        else:
            raise ValueError("No batches remained before balancing.")

    # 3. Standard preprocessing (HVGs, scaling, etc.)
    adata.raw = adata
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=norm_val)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key)
    adata = adata[:, adata.var["highly_variable"]]

    multi_resolution_cluster(adata, resolution1=1, method="Leiden")

    # --- FINAL STEP: Determine Largest Batch Name from FINAL data ---
    # This is safe because balancing is finished.
    largest_batch_name = adata.obs[batch_key].value_counts().idxmax()
    print(f"Final preprocessed data has {adata.n_obs} cells.")
    print(f"The largest batch in the final dataset is: '{largest_batch_name}'")

    # Return the data AND the NAME string
    return adata, largest_batch_name
