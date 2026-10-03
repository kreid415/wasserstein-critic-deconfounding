# Split on the tagged design of record (lead, 2026-10-03)

Manifest `manifests/paper_manifest_stock_pilot_u5_b10.tsv` (tag prereg-tier12-v1, 6,846 rows, sha256 c0244ea4fd76ea05...).
Per model M in {L40S, A100} (queue wait 7.8239 h / 9.4111 h, observed) and score efficiency e in {0.512, 0.40, 0.60}:

    python scripts/gate_task_split.py --manifest manifests/paper_manifest_stock_pilot_u5_b10.tsv \
      --jhpce-cal docs/jhpce/gate_data_v2/gate_report_M/concurrency_calibration_jhpce.json --label M \
      --base-queue-wait-h <wait> --force-local atac_small immune sim1 --expect-fits 6846 --score-efficiency e \
      --out-dir docs/jhpce/gate_data_v2/split/final_6846/eff_e

`free_0.512/`: the same for L40S without `--force-local` (SI-30 not applied).
