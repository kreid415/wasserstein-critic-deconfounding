# Pass 9 change log (author id P9, 2026-11-05)

Branch `paper`, file `paper/outline.tex` (`refs.bib` unchanged; `paper/figures/` untouched). Starting point: `origin/paper` at
67019a8 (Pass 8). Tracked changes carry `[id=P9]` (colour olive); as in Pass 8, a placeholder that sits inside an earlier `\added`
or `\replaced` first argument is filled in place without new markup, because `changes` cannot nest; those in-place fills are
listed below with "(in place)". Standing rules checked against `CONSTRAINTS.md`: SI-78 (no "pre-registered", "registered",
"confirmatory" in the new text), SI-114 (image-GAN paragraph untouched), SI-67 (plain wording), update-step rule (no sentence frames
the critics' five update steps, the discriminator's one, or minibatch reuse as a limitation).

## How the numbers were put in

All new numbers are written by `paper/tools/pass9_fill.py`, which reads the result files, renders the new text from templates and
applies exact edits to `paper/outline.tex` (each old text must occur exactly once, or it stops). It writes
`paper/pass9_numbers.csv`, one row per number in the new text: line in `outline.tex`, edit id, the number as written, source file,
key or row selector (or the computation over named keys), and the raw source value. Statements that depend on the data (for
example "only on pancreas", "the discriminator lost the most purity", "every penalty exceeded Harmony on 5 of six") are asserted in
the script, so a change in the inputs stops it instead of leaving wrong text. Re-run from the Pass 8 text:

    git show 67019a8:paper/outline.tex > paper/outline.tex
    /home/kendall/.claude-science/conda/envs/scvi-api/bin/python paper/tools/pass9_fill.py [--replication paper/tools/replication_result.json]

`paper/tools/test_pass9_fill.py` tests the number formats and the fail-loud paths, and checks that re-running the script on the
Pass 8 text reproduces the outline and ledger of the commit that last wrote the ledger (skipped where the result files are absent).

Sources (read on the workstation; SHA-256 prefixes checked equal to the lead's artifacts before use):

| source | path | sha256 (16) |
|---|---|---|
| sweep analysis (8595a16), `paper_numbers.json` and `tables/` | `D/sweep_analysis_final_20261105/` | 30caefc14c0614f8 (paper_numbers.json) |
| independent placement check (3280bd4) | `D/code/wt_sweep_paper/experiments/sweep_checks/out/placement_summary.csv`, `placement_equal_tries.csv` | 668d740c7e6535b8 (summary) |
| biology check (c4a793f code, c04ed09/83c452e results) | `D/code/wt_biology_check/docs/biology_check/` (`biology_check.json`, `tables/`) | 95d1e18296745cf2 (summary_by_method.csv), 3aa227489195358d (summary.csv) |

`D` = `/home/kendall/experiment_data/wasserstein-critic-deconfounding`.

## Edits (line numbers in the Pass 9 `outline.tex`)

1. Preamble (line 17): author P9 defined.
2. Writer's-outline header (line 42, `\replaced[id=P9]`): Tables 3 and 4 hold sweep values; figures remain mocks from another branch.
3. Abstract item 6 (line 51, in place): placement at equal tries (one strength drawn at random beats batch-conditioned plain scVI on
   at most 3 of six real tasks, the best of six on 4 to 6; Harmony and sysVI exceeded on 5 of six at six tries; scANVI exceeded only
   on pancreas, by at most 0.010).
4. Abstract Conclusions (line 53, `\replaced[id=P9]`): one sentence of guidance (strength matters more than penalty choice, 160 of
   195 near-ties, a penalty helps mainly when several strengths are tried, scANVI where labels exist).
5. Methods, matched batch removal (line 136, in place): the three low-signal pairs with plain scVI's batch score, the highest
   common batch score and the gap; none with the unconditioned decoder.
6. Methods, new paragraph on the direct biology check (line 157, `\added[id=P9]`), exploratory, defined after the sweep scores had
   been read (Supplementary Note 12). No citation needed.
7. Results header (line 161): "and one for the direct biology check".
8. Res-1 (line 163, in place): baseline fits per method (447 of 448 scored, 0 failed); the Scanorama 80-neighbour setting on the
   large chromatin-accessibility task is an incomplete setting, not a failed fit (lead decision), so Scanorama is placed with five
   of its six settings there.
9. New Res-5b (lines 189-190, `\added[id=P9]`): direct biology check (R2.6): purity per cell type against plain scVI, rare types,
   the ionocyte example from the lung confusion table, label-transfer recall over all types, marker-gene Jaccard; descriptive.
10. Res-7 (line 203, `\replaced[id=P9]`): single strength chosen on other families; counts of real tasks beating plain scVI per
    penalty, the range and median of differences, own-best bound, the three tools, the unconditioned decoder, the transferred
    strengths per penalty.
11. Res-8 rule (line 207, in place): upper k = 6, the number of settings of Harmony, Scanorama and sysVI.
12. Res-8 (line 208, `\replaced[id=P9]`): equal-tries placement against plain scVI, Harmony, Scanorama, sysVI, the scvi-tools
    adversary, scANVI, both decoders; best-of-k real-task means; optimism stated.
13. Disc-3 (line 218, in place) and its bullet (line 220, `\replaced[id=P9]`): guidance (strength over penalty choice; near-ties;
    k from 1 to 6 gains 0.012 to 0.053 against a 0.661 to 0.666 spread at six tries; when a penalty helps; strength range 0.1 to
    100; scANVI).
14. Disc-5 (line 236, `\deleted[id=P9]`): "Adversary steps reuse one minibatch." removed from the list of factors outside the scope,
    so that no sentence frames the update steps as a limitation; the fact stays as a training setting in Supplementary Note 2.
    The Disc-2 closing sentence ("each adversary's update steps and optimiser at its standard setting") describes standard
    settings and was kept. No other sentence in the active text frames update steps as a limitation.
15. Conclusions (lines 241-242, in place): main conclusion and four practical guidelines (R3.3), each tied to a number.
16. Supplementary Note 1 (line 263, `\added[id=P9]`): notation for the reconstruction term and the discriminator loss (R1 minor 3).
17. Supplementary Note 5 (line 267, in place): every curve reaches $b^*$ by construction (0 of 96 combinations lacked a reading;
    the bottom-rank rule was never used); "at least two of the three readings"; low-signal values.
18. Supplementary Note 7 (line 269, in place): upper k = 6, comparisons stop at six.
19. New Supplementary Note 12 (line 275, `\added[id=P9]`): methods of the direct biology check, no citations.
20. Supplementary header (line 260) and Supplementary Tables list (line 289): Note 12 and Table S10 added.
21. Figure S6 item (line 299) and Figure 6 caption (line 371, in place): upper k = 6. **For the lead:** these two caption texts
    may be replaced by the figure branch's captions at merge.
22. Table 3 (lines 417-440, shown as "Table 2" in the build because Table 2 was removed in Pass 4): all cells filled (mean rank,
    lowest-highest rank over ranked real tasks, mean biology at $b^*$, simulation ranks with the unconditioned decoder, real tasks
    not ranked, tasks ranked); "MOCK -- placeholder values" deleted (tracked); caption sentence on the columns added.
23. Table 4 (lines 443-468, shown as "Table 3"): all cells, conditioned decoder, three decimals (selection score at the transferred
    strength, own best in parentheses); "MOCK" deleted; note on the Scanorama setting and the low-signal immune pair.
24. New Supplementary Table S10 (lines 471-504, after Table 4, typeset; numbered "Supplementary Table S10" inside its float only): per method and
    decoder, purity difference, seed SD, rare-type purity and recall differences, marker Jaccard and its difference.

The task brief said "Tables 2 and 3 xx.xx cells"; the cells were in the tables numbered 3 and 4 in the source (Table 2 was removed
in Pass 4; LaTeX prints them as Tables 2 and 3). Both are filled.

## Decisions made here (wording and reporting, not what is run)

- Upper k is 6 everywhere: the equal-tries comparison in the independent check stops at six tries, the number of settings of
  Harmony, Scanorama and sysVI.
- The headline "scANVI is not beaten" is written as "exceeded only on pancreas" because the placement check shows every penalty
  above scANVI on pancreas at six tries with the batch-conditioned decoder (differences 0.002 to 0.010).
- With the unconditioned decoder, plain scVI has no batch information at all; the text says so wherever penalties are compared
  with it, and the guidance is written for the batch-conditioned decoder.
- Table S10 "seed SD" is the mean over the six real tasks of the SD over seeds. **Discrepancy found:** the biology check's
  `summary.md` labels its seed-SD column as real-task values but `make_summary.py` averages it over all eight tasks, including
  the simulations (for example the conditioned discriminator: 0.031 there, 0.030 over the six real tasks). The paper uses the
  real-task value; the summary page should be relabelled or recomputed on branch `biology-check`.
- The label-transfer recall over all cell types (Res-5b) is computed by the script from the per-type tables (mean over types per
  fit, then seed mean, then mean over the six real tasks); it is not in the biology check's summary tables.

## Placeholders still open (active text)

Result placeholders: the replication across training seeds (abstract item 7 line 52; Res-2b lines 171-174; Disc-1 bullet line
215; the last Disc-1 sentence still says agreement "remains to be assessed") wait for the lead's replication result.
The lead's replication message had not arrived when Pass 9 was submitted, so these 6 `\res` placeholders stay open (Pass 9 status: partial). `paper/tools/pass9_fill.py` takes `--replication` for them; the module `paper/tools/pass9_replication.py` that will read the agreement test output (`replication_agreement_summary.json`, `replication_large_gaps.csv`, `replication_task_readings.csv`, `replication_readings.csv` from `experiments/sweep_replication/agreement_test.py`) is not written yet, and the script's replication branch is the one line set that `scripts/check_tests.py` reports as untested.

`\tbd` items that the sweep and the biology check cannot fill (16):

| line | text | what is needed; evidence found |
|---|---|---|
| 53, 250 | repository address and version tag at submission | release tag at submission |
| 83 | repeat the literature search before submission | a new search |
| 116 | confirm the reference-batch justification | first author |
| 142 | confirm the ranking rules in the code; whether sweep scores had been read when they were fixed | ranking design written 2026-10-22/23 (`docs/ANALYSIS_SPEC.md` commits 065a486, 5d01da0 on sweep-paper); by then the three-task sweep (small chromatin-accessibility, immune, simulation 1) had been read by the strength-choice rules; when the other five tasks' sweep scores were first read was not established here |
| 159 | LLM tools, versions and roles | first author |
| 232 | coverage simulation for ten strengths and three seeds | a new simulation; not run |
| 246, 254-257 | ethics wording, competing interests, funding, contributions, acknowledgements | authors |
| 266 | Sinkhorn refits after the solver correction for all eight tasks | `docs/SWEEP_ANALYSIS.md` (sweep-paper) line 43 says the Sinkhorn redo superseded the earlier Sinkhorn files of the three local tasks; the five cluster tasks were fitted from the v3 manifest; not confirmed from fit records here |
| 273 (two), 274 | dates of the sweep and ranking rules; planned-benchmark fits made before it was dropped | `docs/PREREG.md` lines 512-515: on 2026-10-12, 19 planned-benchmark rows scored, 1 fitted but not scored, 9 running jobs stopped, 1,097 never fitted; whether more were made before 2026-11-02 was not established |

## Build

Built with tectonic 0.16.9 (conda env `latex`) through doc-build-gate (`dg_check_sources`, `dg_build_tex` with `engine="tectonic"`, `dg_page_ink_check`, `dg_render_pages`): 28 pages, 6 of 6 figures embedded (all still mocks), no undefined citations or references, no overfull boxes, one underfull box (badness 5533, in the bibliography, the same as in Pass 8). The page-ink check lists ink near the bottom edge (page numbers) and at the right edge of page 4 (the first author's margin `\todo` note in the Methods), both present before this pass. Pages viewed as images: 2, 3, 9, 10, 12, 14, 16, 18, 20, 23, 25, 26 (tables, Results, Discussion, Conclusions, supplement). The saved LaTeX zip was extracted into an empty directory and built there as well.
