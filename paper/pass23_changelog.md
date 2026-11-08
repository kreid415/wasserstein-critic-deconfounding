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
