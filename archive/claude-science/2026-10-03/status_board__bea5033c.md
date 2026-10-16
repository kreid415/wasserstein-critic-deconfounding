# Run status board

**5/6 units done_valid (83.3%) as of 2026-10-03T21:05:49+00:00 | running 0, pending 0 (0 blocked by a limit reason), failed 1, done_invalid 0, missing 0**

- generated_at: 2026-10-03T21:05:49+00:00 | run started: 2026-10-02T17:16:00-04:00 | elapsed: 23.8 h
- scheduler snapshot: 2026-10-03T11:35:08-04:00 (age 19842s)
- output validity basis: user-supplied validity check

| state | units |
|---|---|
| done_valid | 5 |
| done_invalid | 0 |
| running | 0 |
| pending | 0 |
| failed | 1 |
| missing | 0 |
| **total expected** | **6** |

ETA: nothing outstanding
- method: bootstrap of inter-completion gaps; assumes concurrency stays as observed and unsubmitted units dispatch at the same rate

## failed (1)
- g4_diag_compute053: FAILED; partial output invalid: job exit 1 on compute-053 (AMD Opteron, no AVX2): the AVX2-pinned run died there; redone on compute-174 as 36132943 [job 36132749]

## failure causes (scheduler state or unit state)

- FAILED: 1

_Status claims must quote this board. Anything not in done_valid is not done._
