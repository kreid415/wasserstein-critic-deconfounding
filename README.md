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
curl 'https://figshare.com/ndownloader/articles/12420968/versions/8' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-US,en;q=0.9' \
  -b 'fig_tracker_client=df7ffe61-2bbb-4fe0-8bcb-9bc57a17efc2; GLOBAL_FIGSHARE_SESSION_KEY=05a1baad4778e8acefd3ad8dbaa84e1ff45215f3431635e120e44fb69cda775d8d0bf275; FIGINSTWEBIDCD=05a1baad4778e8acefd3ad8dbaa84e1ff45215f3431635e120e44fb69cda775d8d0bf275; figshare-cookies-essential=true; figshare-cookies-performance=true' \
  -H 'priority: u=0, i' \
  -H 'referer: https://figshare.com/' \
  -H 'sec-ch-ua: "Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: document' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: same-origin' \
  -H 'sec-fetch-user: ?1' \
  -H 'sec-gpc: 1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36' \
  -o data_download.zip
```

## Usage
### 1. Reproducing Paper Experiments
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
