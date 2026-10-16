# Run status board

**11/1098 units done_valid (1.0%) as of 2026-10-03T21:58:25+00:00 | running 8, pending 1079 (0 blocked by a limit reason), failed 0, done_invalid 0, missing 0**

- generated_at: 2026-10-03T21:58:25+00:00 | run started: 2026-10-03T17:07:40-04:00 | elapsed: 0.8 h
- scheduler snapshot: 2026-10-03T17:58:17-04:00 (age 8s)
- output validity basis: user-supplied validity check

| state | units |
|---|---|
| done_valid | 11 |
| done_invalid | 0 |
| running | 8 |
| pending | 1079 |
| failed | 0 |
| missing | 0 |
| **total expected** | **1098** |

**ETA (median, 80% band)** 2026-10-05T02:50+00:00 (2026-10-05T01:44+00:00 to 2026-10-05T03:57+00:00), from 11 completions; 1087 units outstanding; 0 need resubmission and are excluded.
- method: bootstrap of inter-completion gaps; assumes concurrency stays as observed and unsubmitted units dispatch at the same rate
- warning: no completion for 1290s vs median gap 146s: throughput may have dropped. Inspect running/pending units on the board; this is not a diagnosis.

## running (8)
- A1_atac_small_reference_l0.1_c1_s100_dc329e29: job A1_atac_small_reference_l0.1_c1_s100_dc329e29 [job A1_atac_small_reference_l0.1_c1_s100_dc329e29]
- A1_atac_small_reference_l0.3_c1_s100_886e9d1e: job A1_atac_small_reference_l0.3_c1_s100_886e9d1e [job A1_atac_small_reference_l0.3_c1_s100_886e9d1e]
- A1_atac_small_reference_l1_c1_s100_07023713: job A1_atac_small_reference_l1_c1_s100_07023713 [job A1_atac_small_reference_l1_c1_s100_07023713]
- A1_atac_small_reference_l3_c1_s100_03467eb2: job A1_atac_small_reference_l3_c1_s100_03467eb2 [job A1_atac_small_reference_l3_c1_s100_03467eb2]
- A1_atac_small_reference_l10_c1_s100_b95922a7: job A1_atac_small_reference_l10_c1_s100_b95922a7 [job A1_atac_small_reference_l10_c1_s100_b95922a7]
- A1_atac_small_reference_l30_c1_s100_647c4c9f: job A1_atac_small_reference_l30_c1_s100_647c4c9f [job A1_atac_small_reference_l30_c1_s100_647c4c9f]
- A1_atac_small_reference_l100_c1_s100_697aa0ae: job A1_atac_small_reference_l100_c1_s100_697aa0ae [job A1_atac_small_reference_l100_c1_s100_697aa0ae]
- A1_atac_small_reference_l300_c1_s100_35bd9f84: job A1_atac_small_reference_l300_c1_s100_35bd9f84 [job A1_atac_small_reference_l300_c1_s100_35bd9f84]

## pending (1079)
- A1_atac_small_reference_l1000_c1_s100_57e5f24e: runner state needs_fit [job A1_atac_small_reference_l1000_c1_s100_57e5f24e]
- A1_atac_small_reference_l3000_c1_s100_82ba8b13: runner state needs_fit [job A1_atac_small_reference_l3000_c1_s100_82ba8b13]
- A1_atac_small_pooled_l0.1_c1_s100_31f7ac37: runner state needs_fit [job A1_atac_small_pooled_l0.1_c1_s100_31f7ac37]
- A1_atac_small_pooled_l0.3_c1_s100_57e3d8ea: runner state needs_fit [job A1_atac_small_pooled_l0.3_c1_s100_57e3d8ea]
- A1_atac_small_pooled_l1_c1_s100_3ffa4b69: runner state needs_fit [job A1_atac_small_pooled_l1_c1_s100_3ffa4b69]
- A1_atac_small_pooled_l3_c1_s100_adb1fe17: runner state needs_fit [job A1_atac_small_pooled_l3_c1_s100_adb1fe17]
- A1_atac_small_pooled_l10_c1_s100_96186f42: runner state needs_fit [job A1_atac_small_pooled_l10_c1_s100_96186f42]
- A1_atac_small_pooled_l30_c1_s100_2260f8f1: runner state needs_fit [job A1_atac_small_pooled_l30_c1_s100_2260f8f1]
- A1_atac_small_pooled_l100_c1_s100_3093ce10: runner state needs_fit [job A1_atac_small_pooled_l100_c1_s100_3093ce10]
- A1_atac_small_pooled_l300_c1_s100_8b30e89e: runner state needs_fit [job A1_atac_small_pooled_l300_c1_s100_8b30e89e]
- A1_atac_small_pooled_l1000_c1_s100_b1c08d6b: runner state needs_fit [job A1_atac_small_pooled_l1000_c1_s100_b1c08d6b]
- A1_atac_small_pooled_l3000_c1_s100_e12915be: runner state needs_fit [job A1_atac_small_pooled_l3000_c1_s100_e12915be]
- A1_atac_small_barycenter_l0.1_c1_s100_b038d321: runner state needs_fit [job A1_atac_small_barycenter_l0.1_c1_s100_b038d321]
- A1_atac_small_barycenter_l0.3_c1_s100_545e394e: runner state needs_fit [job A1_atac_small_barycenter_l0.3_c1_s100_545e394e]
- A1_atac_small_barycenter_l1_c1_s100_9899778b: runner state needs_fit [job A1_atac_small_barycenter_l1_c1_s100_9899778b]
- A1_atac_small_barycenter_l3_c1_s100_cc22ce98: runner state needs_fit [job A1_atac_small_barycenter_l3_c1_s100_cc22ce98]
- A1_atac_small_barycenter_l10_c1_s100_a1e1f0be: runner state needs_fit [job A1_atac_small_barycenter_l10_c1_s100_a1e1f0be]
- A1_atac_small_barycenter_l30_c1_s100_b2c5a5e4: runner state needs_fit [job A1_atac_small_barycenter_l30_c1_s100_b2c5a5e4]
- A1_atac_small_barycenter_l100_c1_s100_87419a42: runner state needs_fit [job A1_atac_small_barycenter_l100_c1_s100_87419a42]
- A1_atac_small_barycenter_l300_c1_s100_55723b12: runner state needs_fit [job A1_atac_small_barycenter_l300_c1_s100_55723b12]
- A1_atac_small_barycenter_l1000_c1_s100_a050ece8: runner state needs_fit [job A1_atac_small_barycenter_l1000_c1_s100_a050ece8]
- A1_atac_small_barycenter_l3000_c1_s100_8a4b497f: runner state needs_fit [job A1_atac_small_barycenter_l3000_c1_s100_8a4b497f]
- A1_atac_small_mmd_l0.1_c1_s100_10f59640: runner state needs_fit [job A1_atac_small_mmd_l0.1_c1_s100_10f59640]
- A1_atac_small_mmd_l0.3_c1_s100_730b9b0d: runner state needs_fit [job A1_atac_small_mmd_l0.3_c1_s100_730b9b0d]
- A1_atac_small_mmd_l1_c1_s100_05befd03: runner state needs_fit [job A1_atac_small_mmd_l1_c1_s100_05befd03]
- ... 1054 more in status_board.json

_Status claims must quote this board. Anything not in done_valid is not done._
