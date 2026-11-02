# Agreement rule for the fresh-seed replication (fixed 2026-11-02, before any replication fit was scored)

Decided by the first author (choice card, 2026-11-02: "Large gaps plus rank correlation"). Prediction: PREDICTION.json in this folder
(sweep readings from the sweep analysis, branch sweep-paper commit a361ba2; source table and its SHA-256 recorded there).

Reading: biology score at the matched batch-removal level b* (b* fixed from the sweep, PREDICTION.json `bstar`), read on the replication's
own fits (seeds 0-4) by linear interpolation in batch score between the two bracketing strengths, walking up from plain scVI at seeds 0-4,
first crossing only (ANALYSIS_SPEC.md section 2.2). Raw new-seed reading is primary; a slope-adjusted reading is a sensitivity analysis only.
If the new-seed means do not bracket b*: add the next grid strength outward once (five seeds); if still unbracketed, status
unreached_low / unreached_high, bottom places as in the spec, flagged "did not replicate crossing".

The replication AGREES with the sweep if both hold:
1. Large gaps: each of the sweep's within-pair penalty differences of 0.04 or more (PREDICTION.json `large_gaps`, 4 comparisons) has
   the same sign in the replication.
2. Order: the mean, over the 13 ranked task x decoder pairs, of Kendall's rank correlation (tau-b) between the sweep ranks and the
   replication ranks of the six penalties is above 0, by a one-sided sign-flip permutation test over pairs (all 2^13 sign patterns),
   p < 0.05.
Reported alongside, not part of the rule: the share of sweep near-ties (difference < 0.02) whose replication difference stays within
+/-0.04; per-pair tau; the achieved batch score of every replication point.
Wording in the paper: "a replication across training seeds", never "confirmatory".
