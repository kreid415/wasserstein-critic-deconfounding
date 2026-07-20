import scanpy as sc

from wcd_vae.scCRAFT.utils import multi_resolution_cluster


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
    largest_batch_name = adata.obs[batch_key].value_counts().idxmax()
    print(f"Final preprocessed data has {adata.n_obs} cells.")
    print(f"The largest batch in the final dataset is: '{largest_batch_name}'")

    return adata, largest_batch_name
