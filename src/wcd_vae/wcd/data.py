import scanpy as sc


def _assert_integer_counts(X, source, batches=None):
    """Raise unless EVERY stored value of X is a non-negative integer.

    WHY every value: an earlier version checked only the first 200k stored values and passed the
    immune file, whose last batch (smart-seq2, 3% of cells) holds non-integer expected counts.
    When `batches` (per-cell labels) is given, the error names the offending batches."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.tocsr()
        bad_val = (X.data < 0) | (np.abs(X.data - np.round(X.data)) > 1e-6)
        bad_cell = np.add.reduceat(bad_val.astype(np.int64), X.indptr[:-1]) > 0 if X.nnz else np.zeros(X.shape[0], bool)
        bad_cell &= np.diff(X.indptr) > 0
    else:
        A = np.asarray(X)
        bad_cell = ((A < 0) | (np.abs(A - np.round(A)) > 1e-6)).any(axis=1)
    if bad_cell.any():
        where = ""
        if batches is not None:
            import pandas as pd
            where = " Offending batches: " + str(pd.Series(np.asarray(batches)[bad_cell]).value_counts().to_dict())
        raise ValueError(
            f"{source}: {int(bad_cell.sum())} cells hold negative or non-integer values; scVI's NB/ZINB "
            f"likelihood needs raw counts.{where} Fix the source or list the batch under the registry's "
            "'round_count_batches' (only for quantifier expected counts, never for normalised data).")


def select_reference_batch(adata, batch_key, celltype_key):
    """Pick the reference batch by MAXIMUM CELL-TYPE SHANNON ENTROPY, ties broken by size.

    # WHY: the reference batch defines the target distribution every other batch is
    #   aligned onto, so it should be the batch that best REPRESENTS the biology. The
    #   previous default (``reference_batch=0``) was the ALPHABETICALLY first batch --
    #   ``sort_values().unique()`` sorts by name, not size or content -- which on 4 of 6
    #   datasets selected one of the smallest batches (pancreas 'celseq': 1,004 cells,
    #   8th of 9). Choosing the largest batch instead would optimise for cell count, but a
    #   large batch dominated by one or two cell types is a poor alignment target: it
    #   pulls the shared latent toward whatever biology happens to be over-represented.
    #   Shannon entropy of the cell-type distribution directly measures how evenly a batch
    #   covers the biology, which is the property a reference should have.
    # HOW: H = -sum(p log p) over cell-type proportions within the batch; ties (rare) fall
    #   back to cell count. Measured effect: pancreas moves from 'celseq' (n=1,004) to
    #   'inDrop1' (n=1,937, 14/14 cell types, H=1.851 vs the largest batch's 1.757), sim1
    #   from 'Batch1' to 'Batch5' (7/7 types); immune and lung are unchanged because their
    #   largest batch is already the most diverse.
    # NOTE 'total variance' and 'mean pairwise distance' were evaluated as diversity
    #   proxies and REJECTED -- they track batch size and technology dispersion, not
    #   cell-type coverage.
    """
    import numpy as np

    rows = []
    for name, sub in adata.obs.groupby(batch_key, observed=True):
        counts = sub[celltype_key].value_counts()
        counts = counts[counts > 0]
        if len(counts) == 0:
            continue
        p = counts.to_numpy(dtype=float)
        p = p / p.sum()
        rows.append((float(-(p * np.log(p)).sum()), len(sub), str(name)))
    if not rows:
        raise ValueError(f"no batches with cell-type labels under {batch_key!r}")
    # max entropy, then max size
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return rows[0][2]


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
    round_count_batches=None,
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

    # 3. Raw counts.
    # WHY: the benchmark h5ads store LOG-NORMALISED values in X and the integer counts in
    #   layers["counts"]. The previous code copied X into layers["counts"], so every scVI-family
    #   fit used an NB/ZINB likelihood on log data and PCA saw doubly-normalised values
    #   (code audit 2026-10-01, finding C2). scVI's likelihood is defined on counts.
    # HOW: take the source counts layer when present, then REFUSE non-integer input so the
    #   bug cannot recur silently (scVI only warns, and the sweep logs suppressed warnings).
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    if round_count_batches:
        # Explicit, registry-recorded opt-in for batches whose source holds transcript-quantifier
        # EXPECTED counts (fractional only from multi-mapping read assignment). Rounding these to
        # integers is standard practice (e.g. tximport/DESeq2); normalised values must NOT be listed.
        import numpy as np
        import scipy.sparse as sp
        rows = np.where(adata.obs[batch_key].astype(str).isin(round_count_batches).values)[0]
        X = adata.X.tocsr().astype(np.float32) if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
        if sp.issparse(X):
            for i in rows:
                X.data[X.indptr[i]:X.indptr[i + 1]] = np.rint(X.data[X.indptr[i]:X.indptr[i + 1]])
            X.eliminate_zeros()
        else:
            X[rows] = np.rint(X[rows])
        adata.X = X
        print(f"[prep] rounded expected counts in {len(rows)} cells of batches {round_count_batches}")
    _assert_integer_counts(adata.X, anndata_path, batches=adata.obs[batch_key].astype(str).values)

    # 4. Preprocessing (modality-dependent), from the counts above
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

    # PCA embedding is used as the PCR baseline in the metric suite. (The dual-resolution
    #      Leiden clustering that formerly supervised a triplet loss is no longer used: the
    #      native backbones train on reconstruction + KL only, so it has been removed.)
    sc.tl.pca(adata, n_comps=50)

    # --- FINAL STEP: Determine Largest Batch Name from FINAL data ---
    largest_batch_name = adata.obs[batch_key].value_counts().idxmax()
    print(f"Final preprocessed data has {adata.n_obs} cells.")
    print(f"The largest batch in the final dataset is: '{largest_batch_name}'")

    return adata, largest_batch_name
