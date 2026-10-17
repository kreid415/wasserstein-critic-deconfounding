# Incident map (ids from workflow_incidents.csv)

| id | what happened | board feature |
|---|---|---|
| 157 | Empty logs for array tasks 32-43 read as timed out; sacct/squeue showed PENDING, never started | range expansion; `pending` state; rule: no "walled" without TIMEOUT state |
| 168 | 280 TIMEOUT tasks blamed on short walls; a shared CPU-minute cap killed them | `timeout_suspect` (Elapsed vs Timelimit), `reason_blocking` for pending |
| 477 | Jobs believed stuck from timestamps; the cluster clock was misread | mandatory `#CAPTURED` stamp, offset-aware times |
| 412 | "Hung for 3 h" was 34 min after a -4 h offset; the driver was killed and a false HANG note stored | `rs_since`, tz required, negative age raises |
| 411 | A shard with an OOM'd seed got `.done` with 4/5 seeds; resume would skip the seed | validity by content, `oom_in_step`, marker never consulted |
| 522 | Launcher exited 0 and wrote `.done` over failed configs | `done_invalid` via check; anomaly on COMPLETED + invalid output |
| 160 | Outputs called complete had 1 of 3 seeds | `found/expected` sub-unit tally |
| 414 | "5 seeds each" claimed, one seed missing | same |
| 484 | Waves repeatedly called complete/closed out against partial data | headline fraction; "done" only when done_valid == n_units |
| 328 | "Job just finished" while last check showed running | states come from the snapshot; timestamp quoted |
| 491, 492 | User asked status five ways and finally paused work | one artifact answering status, with ETA or an honest refusal |
| 109, 110 | Pass counts claimed before the async job returned / not in any output | scheduler state per unit; no result claim without an output the check passed |
| 536 | Six large shards co-resident on one GPU hit CUDA OOM | `oom_*` notes and failure_summary make the cause visible at once |
| 523 | OOM fix scoped to one head; sibling shards still failed | `needs_action` lists all failed units, not those already known |
| 479 | Unstarted analyses recorded as DONE with artifact ids | units without output stay `missing` |
| 167 | 45 tasks died after full training; wave yield invisible until late | failed units listed with cause as soon as sacct shows them |
