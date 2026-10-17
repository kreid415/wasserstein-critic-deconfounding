# Run status board

**4/5 units done_valid (80.0%) as of 2026-10-03T22:04:44+00:00 | running 0, pending 0 (0 blocked by a limit reason), failed 1, done_invalid 0, missing 0**

- generated_at: 2026-10-03T22:04:44+00:00 | run started: 2026-10-02T17:16:57-04:00 | elapsed: 24.8 h
- scheduler snapshot: 2026-10-03T18:04:20-04:00 (age 25s)
- output validity basis: user-supplied validity check
- sub-units (e.g. seeds) found/expected across checked outputs: 128/128

| state | units |
|---|---|
| done_valid | 4 |
| done_invalid | 0 |
| running | 0 |
| pending | 0 |
| failed | 1 |
| missing | 0 |
| **total expected** | **5** |

ETA: nothing outstanding
- method: bootstrap of inter-completion gaps; assumes concurrency stays as observed and unsubmitted units dispatch at the same rate

## failed (1)
- g4_diag_numba: FAILED; partial output invalid: sacct FAILED exit 1 [job 36132749]

## failure causes (scheduler state or unit state)

- FAILED: 1

_Status claims must quote this board. Anything not in done_valid is not done._
