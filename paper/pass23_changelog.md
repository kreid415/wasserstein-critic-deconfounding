# Pass 23 change log (2026-11-06 to 2026-11-09, tracked id P23, blue)

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
- Best-of-k comparison (removed 2026-11-09, user choice: it mostly shows that tuning a strength helps, its one-try value depends
  on the width of the strength grid, 0.1 to 3,000, and Res-6 already compares each method at one strength chosen on other task
  families with plain scVI and the standard tools). The analysis was the expected best selection score among k of a method's
  settings drawn at random, k = 1 to 6, so that methods with different numbers of settings were compared at equal numbers of
  tries. Removed: Figure 6a (the first step, next section); the Results paragraph Res-7 (labelled Res-8 before Pass 23; Results now
  end with Res-6); the Methods definition (expected best selection score among k settings) and the description of placement at equal
  numbers of tries; the Abstract, Discussion (Disc-3) and Conclusions sentences that cited it; Supplementary Figure S6 (best of k
  per task; the trade-off curves, the former Supplementary Figure S7, took the number S6, `paper/figure_scripts/figS6.py`);
  panel b of Supplementary Table S7 (real tasks on which a penalty beats a tool at k = 1 and k = 6); the Supplementary Note 7
  paragraph with its equation (label eq:s7-bestk); the Note 5 and Note 10 lists and Note 10's Scanorama sentence. Recorded in
  Supplementary Note 10 ("Analyses removed"). The data files `best_of_k.csv`, `best_of_k_real_mean.csv` and `placement_summary.csv`
  stay in `paper/figure_data`, read by no figure or table (`paper/figure_data/README.md`).
  - Sentences rewritten, with the results they now rest on (numbers already printed elsewhere in the paper). Abstract Results
    item: "a penalty at one strength drawn at random exceeded plain scVI's selection score
    on at most 3 of six real tasks, and the best of six strengths on 4 to 6; at six tries every penalty exceeded Harmony and sysVI on 5 of six real tasks"
    now reads on Res-6 (penalties above plain scVI on 3 to 5 of six real tasks, median difference +0.006; Harmony, Scanorama and
    sysVI on 1, 0 and 1). Abstract Conclusions, Disc-3 and the Conclusions observation (i): "the strength mattered more than the
    choice of penalty" now rests on Res-3 (biology at b* differed between penalties by a median of 0.025, against 0.213 at each
    penalty's strongest fitted setting) and the 160 of 195 near-ties; "plain scVI remained a strong candidate" and "when to add a
    penalty" now rest on Res-6 (3 to 5 of six real tasks, median +0.006, 11 of 36 differences under 0.005). The Disc-3
    bullet on standard tools and the first Conclusions item now cite Harmony, Scanorama and sysVI above plain scVI on 1, 0 and 1
    of six (Res-6), where they had cited the penalties above these tools at equal tries. The sentence that scANVI was exceeded
    only on pancreas at six tries is dropped; the Conclusions keep that scANVI scored higher than every penalty on most tasks
    (Supplementary Table S14).
  - Res-7 numbers removed with the text (batch-conditioned decoder, six real tasks): plain scVI scored 0.648 (six-task mean); at
    one try the penalties' expected best score exceeded plain scVI's on 0, 2, 2, 0, 3, 1 of six tasks (8 of 36 task-penalty cells)
    and at six tries on 4, 5, 5, 5, 5, 6 (30 of 36); six-task means 0.609 to 0.653 at one try and 0.661 to 0.666 at six; first above
    plain scVI at 4, 3, 2, 2, 1, 2 tries (discriminator, reference critic, pooled critic, barycenter critic, MMD, Sinkhorn
    divergence); at six tries every penalty above Harmony and sysVI on 5 of six tasks and above Scanorama on all 5 tasks with six
    settings; scVI adversary mean 0.651; scANVI mean 0.700, exceeded only on pancreas at six tries, by 0.002 to 0.010;
    unconditioned-decoder plain scVI mean 0.578. Disc-3: going from one to six tries raised each penalty's mean by 0.012 to 0.053.
  - Supplementary Note 7 paragraph (equation S7-bestk): "Each penalty has its 10 to 12 fitted strengths, Harmony, Scanorama and sysVI six settings each (Scanorama five on the large
    chromatin-accessibility task, where its 80-neighbour setting could not be fitted)"; "but comparisons stop at $k=6$"; "The mean over the six real tasks
    at each $k$ is taken over the tasks where the method has at least $k$ settings". Note 10: Scanorama's best-of-$k$ curve on that task stops at $k=5$
    (large chromatin-accessibility task).
  - Supplementary Figure S6 caption (best of k per task), before its deletion: plain scVI scored 0.648; at k = 1 every penalty except MMD
    scored below plain scVI; Scanorama's mean at $k=6$ covers five tasks; each penalty has ten settings (an eleventh on the immune task only for three) or,
    for the Sinkhorn divergence, twelve; the curves stopped at k = 6.
  - Ledger: the rows that read these numbers from the removed text (pass13 A25-A35, supp-analysis SA7_34-SA7_40, supp-records
    N10_113) are retired by P23B01-P23B19 (`paper/pass23_numbers.csv`); P23F01-03, which read the numbers of the Figure S6 caption,
    point here instead of at `paper/supp/figures.tex`. Rows of the older keyed ledgers (pass9) have no text anchor and still re-read their sources.

## Figure 6 panel (a) cut from the main text

- Figure 6, panel (a), the expected best selection score among k settings against the number of tries (mean over the six real
  tasks), is cut from the main text (2026-11-09, first author's decision: a best-of-k curve that rises with k only shows that
  tuning helps). Figure 6 is now the former panel (b) alone, batch against biology score per real task at the single strength
  chosen on other task families (`paper/figure_scripts/fig6.py`, which no longer reads `best_of_k_real_mean.csv`).
- The same day the rest of the best-of-k analysis followed (first author's decision, see Removed claims): the Res-7 numbers that
  were first kept as text, Supplementary Figure S6 (per-task curves, with the six-task details of the former Figure 6a caption) and the
  other places that cited it are removed too, so nothing of the analysis is printed any more.
- References: "Figure 6b" is now "Figure 6" (Res-6, Table S14 note, Figure list). Supplementary Figure S6 is now the trade-off
  curves (the former Figure S7; `paper/figure_scripts/figS6.py` writes `paper/figures/figS6.png`, byte-identical to the former
  `figS7.png`), so every "Supplementary Figure S7" is now "Supplementary Figure S6" and the supplementary figures are S1-S6 again.
