Data is available here: "https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-\_integration_task_datasets_Immune_and_pancreas_/12420968"

We use the pancreas_norm_complexBatch, Immune_ALL_human, and Lung_atlas_public datasets.

Command to download the data:
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

Install the package with pip install .

To execute the experiments run binary_experiments.sh, reference_experiments.sh, and multibatch_experiments.sh in the scripts directory.

Afterwards, all figures and tables can be generated from results.ipynb in the notebooks directory.
