# Comparing Wasserstein Critics and Discriminators for Adversarial Deconfounding

[![License: LaTeX Project Public License](https://img.shields.io/badge/License-LPPL-blue.svg)](https://www.latex-project.org/lppl.txt)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository contains the implementation and benchmarking framework for **Wasserstein Critic Deconfounding**, as described in the paper: *"Topology Matters: The Trade-off Between Wasserstein Critics and Discriminators for Single-Cell Data Integration"*.

## Overview
Adversarial autoencoders are a popular solution to correct for technical batch effects that confound single-cell analysis.  This project provides a rigorous, controlled comparison between two adversarial objectives:
1.  **Standard Discriminators**: Approximating Jensen-Shannon divergence. 
2.  **Wasserstein Critics**: Approximating Earth Mover’s distance via a multi-headed reference-based formulation.

Our research highlights a fundamental trade-off: while Wasserstein critics provide superior local batch mixing, they are more prone to biological over-correction and topological collapse compared to the more "conservative" standard discriminator.

This work uses scCRAFT (He et al., 2025) as the backbone network and has been adopted for these experiments. The original implementation is available [here](https://github.com/ch2343/scCRAFT).

## Repository Structure
* **`src/wcd_vae/`**: Core package containing the VAE backbone and adversarial modules.
    * **`scCRAFT/model.py`**: Main training class (`SCIntegrationModel`) and integration logic.
    * **`scCRAFT/networks.py`**: Neural network architectures for the VAE and Discriminator/Critic.
* **`scripts/`**: Bash and Python scripts for reproducing core experiments.
    * **`binary_experiments.sh`**: Foundational two-batch integration.
    * **`multibatch_experiments.sh`**: Scalability analysis as batch count increases.
    * **`reference_experiments.sh`**: Sensitivity analysis of the reference-based formulation.
* **`notebooks/`**:
    * **`results.ipynb`**: Visualization tools for generating all figures and tables in the manuscript.

## Installation
The project requires Python 3.11+.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kreid415/wasserstein-critic-deconfounding
    cd wasserstein-critic-deconfounding
    ```
2.  **Install the package:**
    ```bash
    pip install .
    ```

## Data Acquisition
Data is available here: [Figshare Dataset](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968/1).

We utilize the `pancreas_norm_complexBatch`, `Immune_ALL_human`, and `Lung_atlas_public` datasets.

**Command to download the data:**
```bash
curl -L 'https://figshare.com/ndownloader/articles/12420968/versions/8' \
  -o data_download.zip
```

## Usage
### Revision pipeline (current, 2026-10)
The revision experiments (docs/PAPER_PLAN.md, pre-registered rules in docs/PREREG.md, constraints in CONSTRAINTS.md)
run through one path:
1. `scripts/prep_scib_task.py`: prepped scIB task files (`<task>__scib.h5ad`).
2. `scripts/build_paper_manifest.py`: the manifest, one row per fit, every setting a column
   (`manifests/paper_manifest_stock_pilot_u5_b10.tsv`, tag prereg-tier12-v1; X3 rebuilt under SI-41 in
   `manifests/paper_manifest_stock_pilot_u5_b10_v2.tsv`; X15 KL warm-up sensitivity (SI-44) and Scanorama knn 2-80
   (SI-46) in `manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv`, the manifest in use; report:
   `scripts/manifest_v3_report.py`).
3. `scripts/run_stage.py`: fits each row with `scripts/fit_paper_config.py` (scVI arms and neural baselines) and scores
   it with `scripts/score_scib_native.py` (scib.metrics.metrics, seeded kBET). X13 CPU baselines are fitted with
   `scripts/run_cpu_baselines.py`.
4. `scripts/prereg_rules.py`, `scripts/freeze_a1_grid.py`, `scripts/freeze_matched_lambda.py`, `scripts/decide_a2_a3.py`:
   the pre-registered decision rules R1-R4.

The rejected pre-revision pipeline (`scvi_final_manifest.tsv`, `score_final_config.py` and its drivers) is retired in
`legacy/` (see legacy/README.md); its files refuse to run.

### 1. Reproducing Paper Experiments (original submission)
To execute the experiments, run the following scripts in the `scripts` directory:
```bash
./scripts/binary_experiments.sh
./scripts/reference_experiments.sh
./scripts/multibatch_experiments.sh
```

### 2. Generating Results
Afterwards, all figures and tables can be generated from **`results.ipynb`** in the `notebooks` directory.

## Key Findings
* **Mixing vs. Conservation**: The Wasserstein Critic achieves significantly higher **iLISI** (Integration) scores but is more prone to biological over-correction.
* **Reference Sensitivity**: Integration performance is dependent on a topologically dense reference batch.
* **Scalability**: Standard discriminators scale more effectively to large numbers of batches by avoiding the geometric bottleneck of a single fixed reference.
* **Optimisation fragility**: The two adversarial heads have different optimisation requirements. Quadrupling the minibatch size leaves the discriminator unchanged but causes the critic to stop mixing batches almost entirely; raising the learning rate moves the critic back only partially. Run as experiment `E10`:
  ```bash
  python scripts/run_experiment.py --experiment E10 --dataset pancreas \
    --registry configs/dataset_registry.json --out results/E10
  ```

## Citation
If you use this code or our findings in your research, please cite:

```bibtex
@UNPUBLISHED{Reid2025-sf,
  title       = "Wasserstein critics outperform discriminators in adversarial 
                 deconfounding of gene expression data",
  author      = "Reid, Kendall and Stein-O'Brien, Genevieve and Guven, Erhan",
  journal     = "bioRxiv",
  institution = "bioRxiv",
  year        = 2026,
  copyright   = "http://creativecommons.org/licenses/by-nc-nd/4.0/"
}

@ARTICLE{He2025-dy,
  title     = "Partially characterized topology guides reliable anchor-free {scRNA-integration}",
  author    = "He, Chuan and Filippidis, Paraskevas and Kleinstein, Steven H and Guan, Leying",
  journal   = "Commun. Biol.",
  volume    = 8,
  number    = 1,
  pages     = "561",
  year      = 2025
}
```

## Contact
**Kendall Reid** - [kreid20@jh.edu](mailto:kreid20@jh.edu)
