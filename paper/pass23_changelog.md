# Pass 23 change log (2026-11-06 to 2026-11-08, tracked id P23, blue)

Branch `paper-pass23-wip`, starting point Pass 22 (045e62c). A result-by-result walkthrough of the Results with the first author;
every change below is the user's decision. All edits are made by `paper/tools/pass23_edits.py` from 045e62c; numbers come from
scripts in `paper/tools/` and are in the ledger `paper/pass23_numbers.csv`. This log is completed before the merge into `paper`.

## Removed claims

Claims deleted from the paper (not moved). Each ledger row that pointed at one is retired by a Pass 23 row pointing here.

- Local versus global batch removal (removed 2026-11-08, user choice: the six penalties did not differ on it). Removed: the
  Res-5 sentences, the Methods definition, the Discussion sentences, Supplementary Figure S8, the Supplementary Note 7 paragraph,
  the Local, Global and L-G columns of Supplementary Table S8, the Note 5 mention, and the Supplementary Note 3 definition: local
  removal was the mean of three neighbourhood metrics (iLISI, kBET and graph connectivity), global removal the mean of two
  whole-embedding metrics (PCR and batch ASW), with batch score B = (3 local + 2 global)/5. The deviation is recorded in
  Supplementary Note 10 ("An analysis removed").
- Label transfer as cross-task mean macro F1 per penalty and decoder (removed 2026-11-08, user choice: a raw score averaged
  across tasks), the small chromatin-accessibility example of label-transfer distances (0.840 and 0.851 at 0.024 and 0.010 from
  b*) and the sentence that these orders differed from the biology ranking. Replaced in the merged Res-5 by per-task differences
  from plain scVI, the within-task Kendall tau_b against biology at b*, and the range of distances (paper/tools/res5_label_transfer.py).
  The Kendall tau_b values first proposed in the walkthrough (-0.07 and 0.28) compared cross-task mean F1 with mean rank and are not used.
- Good-strength width (removed 2026-11-08, user choice: it mostly reflected the seed-to-seed spread of the best strength,
  Spearman 0.61 between width and that strength's interval half-width). Removed: Res-6 (Results labels Res-7 and Res-8 became
  Res-6 and Res-7), the former Figure 5 (so the placement figure keeps the number 6), the Methods definition, the Disc-2 clause,
  the Note 5 mention, the Best, Width and Good columns of Supplementary Table S8 and the Supplementary Note 7 paragraph. That
  paragraph defined the window by the lower end of the best strength's 95% t interval (t quantile 0.975, n=3 fits), flagged
  windows with a gap (19 of the 96 windows), and noted that the widest window spans 5.5 decades. Recorded in Supplementary Note 10
  ("Analyses removed"); values remain in the window columns of secondary.csv.
