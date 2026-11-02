# Change log, Pass 6: hypervolume and share of non-dominated strengths (`paper/outline.tex`, 2026-11-02)

Base: branch `paper` at fd4b1c3 (Pass 5). Files edited: `paper/outline.tex`, `paper/refs.bib` (one entry added: `zitzler1998multiobjective`, checked against Crossref record 10.1007/BFb0056872). New file: this log.

Reason for the pass. The first author asked to use Li et al.'s front measure instead of a measure of our own: "rather than use our own area measurement, shouldn't we just use Li et al's" (2026-11-02). The trade-off area was already computed as the hypervolume indicator of Zitzler and Thiele (1998), the measure Li et al. (2022, Table 1 and 2) use: a union of rectangles from a reference point. The pass renames it and cites both papers.
Two choice cards, same day:
- "Keep plain-scVI point (Recommended)": the reference point stays at plain scVI's batch score and 0.10 below its biology score, and the division by the batch gap stays.
- "Add non-dominated share only (Recommended)": Li et al.'s share of non-dominated points is added; their number of distinct choices is not, because it needs an arbitrary grid width.

Matching code and specification: branch `hypervolume`, made from `sweep-paper` (ledger SI-118). Area values had been computed by the analysis pipeline in dev runs on 2026-11-01, but none had been quoted in any document before the change.

Marking conventions as in Passes 3 to 5. "P6" = new `\added/\replaced[id=P6]`. "In place" = an edit inside an earlier `\added{}` or inside the first argument of an earlier `\replaced{}{}`, with no new markup, because the `changes` package does not nest its commands. The old wording is in git at fd4b1c3.

| # | Place | Kind | Old | New |
|---|---|---|---|---|
| 1 | Preamble | P6 author | - | `\definechangesauthor[name={Pass 6}, color=brown]{P6}` |
| 2 | Abstract, results item 2 | in place | `trade-off area, collapse rate, ...` | `hypervolume, share of non-dominated strengths, collapse rate, ...` |
| 3 | Methods, secondary-analysis paragraph | in place | `its trade-off area, ...; all five views`; definition sentence of the trade-off area | `its hypervolume, its share of non-dominated strengths, ...; all six views`; hypervolume sentence citing zitzler1998multiobjective and li2022trade (same reference point and scale); one sentence defining the share of non-dominated strengths |
| 4 | Res-4 heading | in place | `Trade-off area and collapse rate give two further views` | `Hypervolume, share of non-dominated strengths and collapse rate give three further views` |
| 5 | Res-4 rule bullet | in place | `Rule, trade-off area: as in Methods; low-signal pairs are left out` | `Rule, hypervolume and share of non-dominated strengths: as in Methods; low-signal pairs are left out of the hypervolume and kept for the share, which does not use b*` |
| 6 | Res-4 result bullet | in place | `trade-off area per penalty, task and decoder` | `hypervolume and share of non-dominated strengths per penalty, task and decoder` |
| 7 | Discussion result bullet | in place | `trade-off area and collapse rate together` | `hypervolume, share of non-dominated strengths and collapse rate together` |
| 8 | Supplementary Note 5 | P6 replaced + added | `trade-off area` in the list of analyses that exclude low-signal pairs | `hypervolume`; `share of non-dominated strengths` added to the list that keeps them |
| 9 | Supplementary Note 7 | P6 replaced + added | `trade-off area (corner` | `hypervolume (reference point`; new entry for the share of non-dominated strengths (tie tolerance, failed points skipped, all pairs, number of distinct choices not used and why) |
| 10 | Table S8 list | in place | `trade-off area,` | `hypervolume, share of non-dominated strengths,` |
| 11 | Figure 4 list item and caption; figure comment | in place | `(a) trade-off area` / `(a) Trade-off area` | `(a) hypervolume` / `(a) Hypervolume` |

Not changed:
- The old arguments of earlier `\replaced{}{}` that still say "trade-off area" (history).
- The commented-out contrast-design paragraph.
- The mock Figure 4 image. It is synthetic and is redrawn from the real record, where the column is now `hypervolume`.
- Citations: two `\citep` added inside an earlier `\added{}` (allowed; only deleted text needs `\mbox`).
