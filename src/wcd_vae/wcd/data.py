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
    modality="rna",
    cluster=True,
):
    """Preprocess an integration task for the adversarial VAE.

    modality:
      "rna"  - standard scRNA-seq path: cell/gene QC filters, per-cell normalisation,
               log1p, and batch-aware HVG selection to ``n_top_genes``.
      "atac" - scATAC gene-activity path. HVG selection and the aggressive gene/cell
               QC filters are SKIPPED: gene-activity matrices already summarise
               accessibility over a modest, curated gene set (a few thousand features),
               so HVG subsetting would discard informative signal and the standard
               min_genes threshold (tuned for RNA depth) would drop most ATAC cells.
               Raw counts, per-cell normalisation, and log1p are retained so the NB
               decoder and cosine term operate on the same footing as RNA.
    """
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

    # 3. Preprocessing (modality-dependent)
    adata.raw = adata
    adata.layers["counts"] = adata.X.copy()
    if modality == "atac":
        # WHY: gene-activity ATAC has few, curated features and lower per-cell depth;
        #      RNA-tuned QC + HVG selection would discard signal and most cells.
        # HOW: keep counts, normalise + log1p only; no gene/cell filtering, no HVG subset.
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=norm_val)
        sc.pp.log1p(adata)
    else:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=norm_val)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key)
        adata = adata[:, adata.var["highly_variable"]]

    # WHY: the dual-resolution Leiden clustering is only needed as the triplet-loss
    #      supervision for TRAINING; analyses that only need the PCA embedding (e.g.
    #      E6 support-overlap) can skip this expensive step.
    if cluster:
        multi_resolution_cluster(adata, resolution1=1, method="Leiden")
    else:
        sc.tl.pca(adata, n_comps=50)

    # --- FINAL STEP: Determine Largest Batch Name from FINAL data ---
    largest_batch_name = adata.obs[batch_key].value_counts().idxmax()
    print(f"Final preprocessed data has {adata.n_obs} cells.")
    print(f"The largest batch in the final dataset is: '{largest_batch_name}'")

    return adata, largest_batch_name
