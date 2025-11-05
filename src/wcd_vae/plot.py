import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
import seaborn as sns


def plot_umap_by_technology(
    adata, batch_key="tech", color_key="celltype", ncols=3, figsize_per_panel=(5, 5)
):
    """
    Plot UMAP with consistent x and y scales and consistent colors for each technology/batch.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object with UMAP coordinates in obsm['X_umap']
    batch_key : str, default 'tech'
        Key in adata.obs containing batch/technology information
    color_key : str, default 'celltype'
        Key in adata.obs for coloring points
    ncols : int, default 3
        Maximum number of columns in subplot grid
    figsize_per_panel : tuple, default (5, 5)
        Size of each subplot panel

    Returns:
    --------
    None (displays plots)
    """

    sc.tl.umap(adata, min_dist=0.5)

    # Ensure cell types are categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[color_key]):
        adata.obs[color_key] = adata.obs[color_key].astype("category")

    # Get unique technologies/batches and cell types
    technologies = adata.obs[batch_key].unique()
    cell_types = adata.obs[color_key].cat.categories

    # Create a consistent colormap for cell types
    cmap = plt.cm.get_cmap("tab20", len(cell_types))
    colors = [cmap(i) for i in range(len(cell_types))]
    color_dict = dict(zip(cell_types, colors))

    # Get the overall x and y limits from the full UMAP
    x_coords = adata.obsm["X_umap"][:, 0]
    y_coords = adata.obsm["X_umap"][:, 1]
    x_min, x_max = x_coords.min() - 0.5, x_coords.max() + 0.5
    y_min, y_max = y_coords.min() - 0.5, y_coords.max() + 0.5

    # Create subplots - adjust the number of columns based on preference
    n_techs = len(technologies)
    ncols = min(ncols, n_techs)
    nrows = (n_techs + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    )

    # Handle single subplot case
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each technology separately
    for i, tech in enumerate(technologies):
        ax = axes[i]

        # Subset data for this technology
        tech_mask = adata.obs[batch_key] == tech
        tech_coords = adata.obsm["X_umap"][tech_mask]
        tech_celltypes = adata.obs.loc[tech_mask, color_key]

        # Plot each cell type with consistent colors
        for cell_type in cell_types:
            cell_mask = tech_celltypes == cell_type
            if np.sum(cell_mask) > 0:  # Only plot if there are cells of this type
                ax.scatter(
                    tech_coords[cell_mask, 0],
                    tech_coords[cell_mask, 1],
                    color=color_dict[cell_type],
                    s=1,
                    alpha=0.7,
                )

        # Set consistent limits for all subplots
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Set labels and title
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"{batch_key.capitalize()}: {tech}")
        ax.set_aspect("equal")

    # Hide any unused subplots
    for i in range(n_techs, len(axes)):
        axes[i].set_visible(False)

    # Add a legend to the figure (outside the plots)
    handles, labels = [], []
    for cell_type in cell_types:
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color=color_dict[cell_type], markersize=5, linestyle="None"
            )
        )
        labels.append(cell_type)

    # fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    plt.show()

    # Also create a combined plot with all technologies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Plot colored by technology/batch
    sc.pl.umap(adata, color=batch_key, ax=ax1, frameon=False, show=False, legend_loc=None)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title(f"Colored by {batch_key.capitalize()}")

    # Plot colored by celltype with consistent colors
    for cell_type in cell_types:
        cell_mask = adata.obs[color_key] == cell_type
        if np.sum(cell_mask) > 0:
            ax2.scatter(
                adata.obsm["X_umap"][cell_mask, 0],
                adata.obsm["X_umap"][cell_mask, 1],
                color=color_dict[cell_type],
                s=1,
                alpha=0.7,
            )

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title(f"Colored by {color_key.capitalize()}")

    # Add legend to the second plot
    handles, labels = [], []
    for cell_type in cell_types:
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color=color_dict[cell_type], markersize=5, linestyle="None"
            )
        )
        labels.append(cell_type)

    # fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    plt.show()


def create_cv_results_table(results_df, outer_fold_results, confidence_level=0.95):
    """
    Create a comprehensive table with mean and confidence intervals for CV results.

    Parameters:
    -----------
    results_df : DataFrame
        Results from nested CV
    outer_fold_results : dict
        Raw results from outer folds
    confidence_level : float
        Confidence level for intervals (default 0.95)

    Returns:
    --------
    summary_table : DataFrame
        Formatted table with means and confidence intervals
    """

    alpha = 1 - confidence_level

    # Calculate statistics for each method and metric
    summary_data = []

    for method in ["critic", "no_critic"]:
        method_label = "Critic" if method == "critic" else "Discriminator"

        for metric in ["ilisi", "clisi"]:
            values = np.array(outer_fold_results[method][metric])
            n = len(values)

            # Calculate mean and standard error
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample standard deviation
            se_val = std_val / np.sqrt(n)  # Standard error

            # Calculate confidence interval using t-distribution
            t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci_lower = mean_val - t_critical * se_val
            ci_upper = mean_val + t_critical * se_val

            # Format the results
            summary_data.append(
                {
                    "Method": method_label,
                    "Metric": metric.upper(),
                    "Mean": f"{mean_val:.4f}",
                    f"{int(confidence_level * 100)}% CI": f"({ci_lower:.4f}, {ci_upper:.4f})",
                    "Std Dev": f"{std_val:.4f}",
                    "Min": f"{np.min(values):.4f}",
                    "Max": f"{np.max(values):.4f}",
                    "N Folds": n,
                }
            )

    summary_table = pd.DataFrame(summary_data)

    # Create a pivot table for better visualization
    pivot_data = []
    methods = ["Critic", "Discriminator"]
    metrics = ["ILISI", "CLISI"]

    for method in methods:
        row_data = {"Method": method}
        for metric in metrics:
            method_key = "critic" if method == "Critic" else "no_critic"
            metric_key = metric.lower()

            values = np.array(outer_fold_results[method_key][metric_key])
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            se_val = std_val / np.sqrt(len(values))
            t_critical = stats.t.ppf(1 - alpha / 2, df=len(values) - 1)
            ci_lower = mean_val - t_critical * se_val
            ci_upper = mean_val + t_critical * se_val

            row_data[f"{metric} Mean"] = mean_val
            row_data[f"{metric} CI"] = f"({ci_lower:.4f}, {ci_upper:.4f})"
            row_data[f"{metric} Formatted"] = f"{mean_val:.4f} {row_data[f'{metric} CI']}"

        pivot_data.append(row_data)

    pivot_table = pd.DataFrame(pivot_data)

    return summary_table, pivot_table


def create_comparison_table(outer_fold_results, confidence_level=0.95):
    """
    Create a table comparing critic vs discriminator with statistical tests.
    """
    alpha = 1 - confidence_level

    comparison_data = []

    for metric in ["ilisi", "clisi"]:
        critic_values = np.array(outer_fold_results["critic"][metric])
        no_critic_values = np.array(outer_fold_results["no_critic"][metric])

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(critic_values, no_critic_values)

        # Effect size (Cohen's d for paired samples)
        diff = critic_values - no_critic_values

        # Mean difference and CI
        mean_diff = np.mean(diff)
        se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
        t_critical = stats.t.ppf(1 - alpha / 2, df=len(diff) - 1)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        comparison_data.append(
            {
                "Metric": metric.upper(),
                "Mean Diff": f"{mean_diff:.4f}",
                f"{int(confidence_level * 100)}\% CI of Difference": f"({ci_lower:.4f}, {ci_upper:.4f})",  # noqa: E501
                "t-stat": f"{t_stat:.4f}",
                "p-value": f"{p_value:.6f}",
            }
        )

    return pd.DataFrame(comparison_data)


def plot_cv_results(outer_fold_results):
    """
    Create visualization of CV results with error bars.
    """
    # Prepare data for plotting
    plot_data = []
    for method in ["critic", "no_critic"]:
        method_label = "Critic" if method == "critic" else "Discriminator"
        for metric in ["ilisi", "clisi"]:
            values = outer_fold_results[method][metric]
            for fold, value in enumerate(values, 1):
                plot_data.append(
                    {
                        "Method": method_label,
                        "Metric": metric.upper(),
                        "Fold": fold,
                        "Value": value,
                    }
                )

    plot_df = pd.DataFrame(plot_data)

    # Create subplot for each metric
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, metric in enumerate(["ILISI", "CLISI"]):
        ax = axes[i]
        metric_data = plot_df[plot_df["Metric"] == metric]

        # Box plot
        sns.boxplot(data=metric_data, x="Method", y="Value", ax=ax)

        ax.set_title(f"{metric} Scores Across CV Folds")
        ax.set_ylabel(f"{metric} Score")

    plt.tight_layout()

    return fig


def create_paper_assets(results_df, outer_fold_results, output_dir, output_prefix=""):
    # Create the main results table
    summary_table, pivot_table = create_cv_results_table(results_df, outer_fold_results)

    # Create a clean formatted table
    formatted_table = pivot_table[["Method", "ILISI Formatted", "CLISI Formatted"]].copy()
    formatted_table.columns = ["Method", "iLISI (Mean ± 95% CI)", "cLISI (Mean ± 95% CI)"]

    comparison_table = create_comparison_table(outer_fold_results)
    comparison_table.to_latex(
        f"{output_dir}/{output_prefix}_cv_statistical_comparison.tex",
        index=False,
        float_format="%.4f",
    )

    # Create visualization
    fig = plot_cv_results(outer_fold_results)

    pub_summary = []
    for method in ["Critic", "Discriminator"]:
        method_key = "critic" if method == "Critic" else "no_critic"

        ilisi_values = outer_fold_results[method_key]["ilisi"]
        clisi_values = outer_fold_results[method_key]["clisi"]

        ilisi_mean = np.mean(ilisi_values)
        ilisi_ci = stats.t.interval(
            0.95, len(ilisi_values) - 1, loc=ilisi_mean, scale=stats.sem(ilisi_values)
        )

        clisi_mean = np.mean(clisi_values)
        clisi_ci = stats.t.interval(
            0.95, len(clisi_values) - 1, loc=clisi_mean, scale=stats.sem(clisi_values)
        )

        pub_summary.append(
            {
                "Method": method,
                "iLISI": f"{ilisi_mean:.3f} (95% CI: {ilisi_ci[0]:.3f}-{ilisi_ci[1]:.3f})",
                "cLISI": f"{clisi_mean:.3f} (95% CI: {clisi_ci[0]:.3f}-{clisi_ci[1]:.3f})",
            }
        )

    # Add statistical test summary
    ilisi_tstat, ilisi_pval = stats.ttest_rel(
        outer_fold_results["critic"]["ilisi"], outer_fold_results["no_critic"]["ilisi"]
    )
    clisi_tstat, clisi_pval = stats.ttest_rel(
        outer_fold_results["critic"]["clisi"], outer_fold_results["no_critic"]["clisi"]
    )

    # create a summary table for statistical tests
    stat_summary = pd.DataFrame(
        {
            "Metric": ["iLISI", "cLISI"],
            "t-statistic": [ilisi_tstat, clisi_tstat],
            "p-value": [ilisi_pval, clisi_pval],
        }
    )

    # write statistical summary to latex
    stat_summary.to_latex(
        f"{output_dir}/{output_prefix}_cv_statistical_tests.tex", index=False, float_format="%.4f"
    )

    # Save tables to files
    summary_table.to_csv(f"{output_dir}/{output_prefix}_cv_summary_detailed.csv", index=False)
    formatted_table.to_csv(f"{output_dir}/{output_prefix}_cv_summary_formatted.csv", index=False)
    comparison_table.to_csv(
        f"{output_dir}/{output_prefix}_cv_statistical_comparison.csv", index=False
    )

    # save figure
    fig.savefig(f"{output_dir}/{output_prefix}_cv_results_plot.png", dpi=300)
