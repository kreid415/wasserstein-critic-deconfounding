# Manifest v5 (pruning amendment, user decision 2026-10-17)

v5 = v4 without the follow-ups X3, X7, X8, X12, X15. X1, X13, A1-A3 and X6 rows are byte-identical to v4. Nothing here reads
an X1 or X13 score. Run from the repo root (the builder and the check use the standard library only).

```
python3 scripts/build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5 --bary-iter 10 --x12-runs 16 \
    --followup-design v4 --exclude-experiments X3 X7 X8 X12 X15 --out manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv
python3 experiments/prune_manifest_20261017/check_v5.py     # fails unless v5 = v4 minus those rows; writes docs/manifest_v5_counts.csv and the .sha256
```

| file | sha256 |
|---|---|
| manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv (input, unchanged) | c409b6c35e7afae522d83dcf62bf6c11a272b7eb6a13e20fb3240aaae5538ab6 |
| manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv | ce8dfbf2008b4dbca3b33fbe266039f68a3cf8befb2dc66efcdfa00774a46432 |

The builder without `--exclude-experiments` still reproduces v4 byte for byte (checked 2026-10-18). Counts: docs/manifest_v5_counts.csv.
