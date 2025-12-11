import argparse
import warnings

from wcd_vae.data import prep_data
from wcd_vae.hyperparameter import run_comprehensive_nested_cv
from wcd_vae.plot import create_paper_assets
from wcd_vae.scCRAFT.utils import set_seed

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for WCD-VAE")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument("--batch_count", type=int, default=2, help="Number of batches to consider")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--reference_batch", type=int, default=0, help="Reference batch index")
    parser.add_argument(
        "--balance", action="store_true", help="Whether to balance batches during training"
    )

    args = parser.parse_args()

    set_seed(42)

    data_set = args.dataset.lower()
    output_dir = args.output_dir
    batch_count = args.batch_count
    epochs = args.epochs
    reference_batch = args.reference_batch
    balance = args.balance

    if data_set == "pancreas":
        batch_key = "tech"
        celltype_key = "celltype"
        data_path = "/workspaces/data/human_pancreas_norm_complexBatch.h5ad"

    elif data_set == "immune":
        batch_key = "chemistry"
        celltype_key = "final_annotation"
        data_path = "/workspaces/data/Immune_ALL_human.h5ad"

    elif data_set == "lung":
        batch_key = "protocol"
        celltype_key = "cell_type"
        data_path = "/workspaces/data/Lung_atlas_public.h5ad"

    adata = prep_data(
        data_path,
        batch_key=batch_key,
        celltype_key=celltype_key,
        batch_count=batch_count,
        min_genes=300,
        min_cells=5,
        norm_val=1e4,
        n_top_genes=2000,
        balance=balance,
    )

    results_df, outer_fold_results, sensitivity_results = run_comprehensive_nested_cv(
        adata,
        batch_key=batch_key,
        celltype_key=celltype_key,
        reference_batch=reference_batch,
        epochs=epochs,
        n_outer_folds=2,
        n_inner_folds=2,
        output_dir=output_dir,
        output_prefix=f"{data_set}",
        random_state=42,
        num_workers=4,
    )

    create_paper_assets(
        results_df, outer_fold_results, output_dir=output_dir, output_prefix=f"{data_set}"
    )


if __name__ == "__main__":
    main()
