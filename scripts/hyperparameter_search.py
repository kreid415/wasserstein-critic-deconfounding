import argparse
import warnings

from wcd_vae.data import prep_data
from wcd_vae.hyperparameter import nested_cv_hyperparameter_tuning
from wcd_vae.plot import create_paper_assets
from wcd_vae.scCRAFT.utils import set_seed

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for WCD-VAE")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )

    args = parser.parse_args()

    set_seed(42)

    data_set = args.dataset.lower()
    output_dir = args.output_dir

    if data_set == "pancrease":
        batch_key = "tech"
        celltype_key = "celltype"
        data_path = "/workspaces/data/human_pancreas_norm_complexBatch.h5ad"

    elif data_set == "immune":
        batch_key = "batch"
        celltype_key = "final_annotation"
        data_path = "/workspaces/data/Immune_ALL_human.h5ad"

    elif data_set == "lung":
        batch_key = "batch"
        celltype_key = "cell_type"
        data_path = "/workspaces/data/Lung_atlas_public.h5ad"

    # pancrease
    pancrease_adata = prep_data(
        data_path,
        batch_key=batch_key,
        celltype_key=celltype_key,
        batch_count=2,
        min_genes=300,
        min_cells=5,
        norm_val=1e4,
        n_top_genes=2000,
        balance=False,
    )

    results_df, outer_fold_results = nested_cv_hyperparameter_tuning(
        pancrease_adata,
        batch_key=batch_key,
        celltype_key=celltype_key,
        reference_batch=0,
        epochs=100,
        n_outer_folds=100,
        n_inner_folds=10,
        output_dir=output_dir,
        output_prefix=f"{data_set}_binary",
    )

    create_paper_assets(
        results_df, outer_fold_results, output_dir=output_dir, output_prefix=f"{data_set}_binary"
    )


if __name__ == "__main__":
    main()
