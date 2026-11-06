# Supplement items of branch supp-analysis (Supplementary Notes 4, 5, 6, 7 and 9; Supplementary Table S8)

Base: branch `paper` at 0e79c46 (Pass 15). `paper/outline.tex` is not edited; a later pass replaces the one-line `\item` descriptions
(supplement list, lines 275-278, 281 and 296 of `paper/outline.tex` at 0e79c46) with `\input` of the files below.

## Files (item -> file)

| Item | File | Made by |
|---|---|---|
| Supplementary Note 4, penalty-strength sweep: design and coverage (with the replication design) | `paper/supp/note4.tex` | written |
| Supplementary Note 5, matched batch removal and the low-signal rule | `paper/supp/note5.tex` | written |
| Supplementary Note 6, benchmark ranking | `paper/supp/note6.tex` | written |
| Supplementary Note 7, rules for the secondary analyses (incl. the coverage simulation text moved from the outline description) | `paper/supp/note7.tex` | written |
| Supplementary Note 9, label transfer | `paper/supp/note9.tex` (inputs `paper/supp/note9_table.tex`) | written; table generated |
| Supplementary Table S8, secondary results | `paper/supp/tableS8.tex` | generated |

Generator: `paper/tools/make_supp_supp-analysis.py` (inputs in `paper/figure_data`, no typed numbers; it re-reads local and global
batch removal at b* from `metric_scores.csv` and stops if local minus global differs from the analysis record by more than 1e-9).
Tests: `paper/tools/test_supp_analysis.py`. New input: `paper/figure_data/secondary.csv`, an unchanged copy of `tables/secondary.csv` of
D/sweep_analysis_final_20261105 (commit 8595a16; SHA-256 added to `paper/figure_data/MANIFEST.sha256`; `copy_supp_inputs.sh`,
`make_supp_figures.sh`, `figure_data/README.md` and `REPRODUCE.md` updated). Numbers in the prose: `paper/supp-analysis_numbers.csv`
(169 rows, Pass 12 format), re-read by `python paper/tools/check_ledger.py paper/supp-analysis_numbers.csv`.

## Corrections to the outline descriptions (what the repository says instead)

1. Note 4, Sinkhorn refits. The description says "the 180 fits of the three local tasks were refitted on 17-19 October 2026". The fit
   ledgers give 198 refits: the 180 at the ten grid strengths fitted 17-18 October (`D/tier12_v3/A1_sinkhorn/ledger`, first and last
   fit start 2026-10-17 and 2026-10-18) and the 18 at the edge strengths 10,000 and 30,000 on 19 October
   (`D/tier12_v3/A1_ext_r1_sinkhorn_round1/ledger`), all at commit 10a1ca5. Note 4 states both counts.
2. Note 4, machines. "Every fit of a task on one machine" is written as one kind of machine: the three workstation tasks on the
   workstation's RTX 3080 (`experiments/sweep_replication/README.md`), and each other task in one cluster job on L40S cards
   (`docs/PREREG.md`, 1,890 fits, one job per dataset); the cluster jobs are not shown to have used one node.
3. Note 6, related tasks. The description says the large chromatin-accessibility task contains the small task. The repository says the two
   tasks share 3,618 cells (`docs/ANALYSIS_SPEC.md` section 1); the small task has three batches (3,618, 3,750 and 3,902 cells) and the
   large task eleven (`docs/prepped_fingerprints_scib.json`), so containment is not supported. Note 6 says the tasks share 3,618 cells.
4. Note 5, analyses that keep low-signal pairs. The description lists the share of non-dominated strengths and the collapse rate. The
   good-strength width, the single-strength placement and the best of k settings also keep them (they use the selection score, not b*;
   `ANALYSIS_SPEC.md` sections 8 and 11); label transfer tabulates them and leaves them out of the cross-task means. Note 5 lists all.
5. Table S8, "local and global readings at b*". The analysis record holds only local minus global. Table S8 prints local, global and
   their difference; local and global are recomputed from the per-fit metrics on the same curves and checked against the record's
   difference (all 96 within 1e-9).
6. Note 9, "the full label-transfer table". No supplementary table number is assigned to it in the outline; it is typeset inside Note 9
   as an unnumbered table (`note9_table.tex`). Numbering it is left to the integration pass.

Additions that the descriptions did not name (each from a file): the eleven-strength curves of three penalties on the immune task with
the batch-conditioned decoder (Note 4); the number of crossings per curve (Note 6); that the prediction file's source table is
byte-identical to `paper/figure_data/ranking_pairs.csv` (SHA-256 786cb8c8..., Note 4); that the hypervolume with the reference point at
biology 0 is in the analysis record and not used (Note 7); the 3,073 label-transferred embeddings (Note 9). The replication's extension
fits are mentioned only by reference to Supplementary Note 11 and Table S9, as asked.

## Notes for the integration pass

- Each note starts with `\par\medskip` and ends with `\par\endgroup`, so it can follow list text directly. In the build check the
  `\item` lines were replaced by `\end{itemize}` + `\input{supp/...}` + `\begin{itemize}` (consecutive items grouped), so that the
  longtables of Note 9 and Table S8 are not inside a list.
- `tableS8.tex` ends with `\addtocounter{table}{-1}`: its `\caption` steps the main table counter, and without the correction the main
  tables after it would be renumbered when it is input before them.
- Equations: Note 5 `eq:s5-bstar`, Note 6 `eq:s6-loso`, Note 7 `eq:s7-bestk`; they take the document's equation numbers.

## Checks run (2026-11-06)

- `check_ledger.py paper/supp-analysis_numbers.csv`: 169 values re-read, 0 failures.
- Build: a test copy of `outline.tex` (not committed) with the six descriptions replaced by `\input` of these files, tectonic 0.16.9 via
  doc-build-gate (`dg_check_sources`, `dg_build_tex`): 52 pages, 13 of 13 figures, no failing log class (2 underfull boxes, both in
  text that is not from these files: the repository line, outline.tex line 258, and the bibliography). Pages 15-25 (the new files) viewed.
  `dg_page_ink_check` flags ink within 17 pt of the bottom edge on almost every page of the document; this is the page number under the
  outline's 1.7 cm geometry, present on pages without the new files, and was not changed.
- `scripts/check_tests.py --base origin/paper --src paper/tools -- paper/tools`: see the commit message.
