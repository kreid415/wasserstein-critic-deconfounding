# Compute and clusters (always loaded)

Translated from the user's cluster rules. Terms: SLURM = cluster job scheduler; scratch = purged cluster work disk; `$HOME` = durable home directory.

1. Before any cluster job, environment build, storage or path decision, or scheduler command, re-read `docs/claude/CLOUD_SETUP.md`, `docs/claude/DATA_LOCATIONS.md` and `cluster/README.md`.
2. Storage: conda or venv environments live in `$HOME`. All data (work directories, logs, checkpoints, TMPDIR, model, dataset, package and container caches) lives on cluster scratch. Scratch is purged and is never the durable copy: harvest outputs, check every file against cluster-side checksums, and mirror them to the durable location in `DATA_LOCATIONS.md` before a job's work directory can be purged.
3. Never issue account-wide scheduler commands (`scancel -u`, cancelling by partition or name pattern). Cancel only job ids this session submitted and recorded, because other people and sessions share the account.
4. Every cluster submission needs the user's explicit go (see the production-job notes in `docs/jhpce/`). Show the submission plan (command, resources, walltime, output path, job count) first.
5. Long jobs must run outside the chat session. A command started inside a Claude Code session dies when the session ends or the VM restarts: use `sbatch` for cluster work, and `nohup` or `tmux` plus a log file and a done-marker file for anything local that runs longer than a few minutes. Do not wait on a job with a polling loop; check status on request, or with a supervisor script if one is bundled (see `docs/claude/SKILLS.md`). A supervisor script, job-watch, once validated on JHPCE, touches only recorded job ids and resubmits after a timeout or checkpoint signal.
6. One configuration per task; wall time from a measured run with headroom; per-task output files; throttle arrays that read the same large file (`docs/claude/notes/pitfalls.md`).
7. GPU and memory: the local GPU is a 10 GB RTX 3080; assume a cloud VM has no GPU and no project data until `nvidia-smi` and `docs/claude/DATA_LOCATIONS.md` show otherwise, so do not plan GPU fits in a cloud session. Check VRAM and RAM limits before choosing the number of concurrent workers (too many workers ran out of memory on the heavy datasets).
8. Cluster-side `git push` fails (no credentials); push from a machine that has the token.
