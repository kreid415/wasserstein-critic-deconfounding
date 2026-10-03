# Gate v2: commands that produced docs/jhpce/GATE_JHPCE.md (2026-10-03)

Workspace layout used below (not in the repo): `calib/` = local calibration bundle (artifact
local_queue_latents_stock.tar.gz: bench_queue_manifest.tsv md5 9cca9c8c38e767ebf13a98ad44bb9e95, queue_order.txt,
64 local latents), `hpc/<job_id>/` = files of the two JHPCE gate jobs (A100 8749bada = Slurm 36131135,
L40S a2b6ae1f = Slurm 36133219), copied back from the job workdirs, `setup/setup_report/` = JHPCE setup-job report
(artifact jhpce_setup_report.tar.gz). Envs: `scvi-api` (analysis python), `wcd-kbet` (scoring; R_HOME=$E/lib/R,
R_LIBS=<experiment_data>/Rlib_kbet). Evidence files are copied to `docs/jhpce/gate_data_v2/`.

1. Harvest integrity, per job: `cat gate_latents_jhpce.tar.part00 gate_latents_jhpce.tar.part01 > gate_latents_jhpce.tar`,
   then `md5sum` must equal `gate_latents_jhpce.tar.md5`; extract to `jl/<MODEL>/latents/`; local check: every config in
   queue_order.txt present with keys z/batch/obs_names/config, z finite, shape and obs/batch order equal to the local latent.
2. G3 latent agreement, per model and A100 vs L40S:
   `python scripts/gate_compare_latents.py --local calib/bench_out_stock/latents --remote jl/<MODEL>/latents
   --order calib/queue_order.txt --manifest calib/bench_queue_manifest.tsv --k 15 --out-csv latent_agreement_<MODEL>.csv
   --out-json latent_agreement_<MODEL>.json`
3. Scoring, all 192 latents in ONE env (local wcd-kbet) with the scorer on main 878d8ea (trajectory in BIO_METRICS,
   kBET seeded): list `score_list_g3v2.tsv` = per config in queue order, tags `L_<cfg>` (local latent),
   `A100_<cfg>`, `L40S_<cfg>`, prepped `immune__scib.h5ad` (md5 3888d2563a7243f9854e41de973f3607);
   `PY=$E/bin/python R_HOME=$E/lib/R R_LIBS=... bash scripts/score_list.sh score_list_g3v2.tsv g3v2/scores_out 10`
4. G3 scores, per model: `python scripts/gate_compare_scores.py g3 --scores g3v2/scores_out/scores.csv
   --manifest calib/bench_queue_manifest.tsv --local-prefix L_ --remote-prefix <MODEL>_ --out-dir g3_<MODEL>`
5. Queue waits: submit -> start from `sacct -j 36131135,36133219 -o JobID,Submit,Start,...` (sacct_gate_jobs.txt).
6. Design-of-record manifest: `python scripts/build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5
   --bary-iter 10 --out manifest_dor.tsv` (6,718 fits).
7. Split, per model: `python scripts/gate_task_split.py --manifest manifest_dor.tsv
   --jhpce-cal hpc/<job>/gate_report/concurrency_calibration_jhpce.json --label <MODEL> --base-queue-wait-h <observed>
   --force-local atac_small immune sim1 --expect-fits 6718 --score-efficiency <measured> --out-dir split`
   (score efficiency = single-thread reference time for one immune latent from docs/scoring_time.csv's rate divided by
   the median score_seconds of the 192 latents scored 10 at a time); repeated with --score-efficiency 0.40 and 0.60.
8. Report: `python scripts/gate_report.py --local docs/jhpce/gate_data --setup setup/setup_report
   --gate A100=<gate_report> L40S=<gate_report> --latent-dir <latent> --g3-dir <g3> --g4-dir docs/jhpce/gate_data
   --split-dir <split> --eff-sens <eff_0.40> <eff_0.60> --queue-waits queue_waits.json --scoring scoring.json
   --sacct sacct_gate_jobs.txt --meta report_meta_v2.json --out docs/jhpce/GATE_JHPCE.md`
