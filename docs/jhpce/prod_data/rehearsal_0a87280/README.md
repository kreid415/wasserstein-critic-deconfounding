# Local rehearsal of the JHPCE production jobs at 0a87280

`cluster/jhpce/local_rehearsal.sh --job jobA|jobB --sha 0a872805e855a52deffad3297a6fec8c199eeece --work <dir>
--fit-python <local scvi-api python> --prepped-dir <local prepped_scib>` on the local machine, 2026-10-03/04, at
nice 19 with CUDA hidden. Each directory holds the rehearsal record (`rehearsal.json`), the submission plan
(`plan.json`), the job's stdout and its logs (dry run, stage key, summary, inputs, versions, tests/scvi, fingerprints,
slot, claims). Local paths are replaced by `<work>`, `<local scvi-api python>` and `<local prepped_scib>`.
Stubbed: nvidia-smi, the env_versions.py record handed to the job (`versions_fit.json`, = expected_fit_versions.json;
the real local record is `versions_fit.json.local.json`, its differences `versions_fit.json.local_diff.json`), sacct,
squeue, SLURM_JOB_ID, WCD_SCRATCH and HOME. Everything else ran as on the node.
