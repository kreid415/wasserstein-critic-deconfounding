# Independent review of `outline.tex` (second editing pass), 2026-10-22

Sources: `outline.tex` (line numbers are after my edits), `outline_author_round.tex`, `pass2_changelog.md`, `refs.bib`, `ANALYSIS_SPEC.md` (prereg-x6-v1/docs). Every edit I made is tracked (`\added`, `\replaced{new}{old}`) or is inside an editor `\added{}` block that has not been accepted yet. After my edits a script check found: braces balanced (depth 0); every `\cite` key is in refs.bib; no `\res`, `\tbd`, `\textit`, `\todo` or bare `\citep`/`\citet` inside `\deleted{}` or inside the second argument of `\replaced`.

| # | Location | Issue | Severity | Source checked | Fix |
|---|---|---|---|---|---|
| 1 | Table 3 body, outline.tex:398-404 | Header row and the first three penalty rows were rewritten with no tracked markup. The old header was Outcome/Estimate/95% int./P(<-m)/P(\|.\|<=m)/P(>m)/Call/Tasks, the old row names were Biology at b\*, Trade-off area and Collapse rate (pp), and the old Call column held `\tbd{call}`. | error | outline_author_round.tex:374-381 | Fixed: each changed cell is now `\replaced{new}{old}`. The old `\tbd{call}` is written as plain "[to be decided: call]". |
| 2 | Intro, deleted benchmark paragraph, outline.tex:73 | `\mbox{\citep{yi2025benchmarking}}` inside `\deleted{}` breaks the markup rule (must be `\mbox{\citealt{}}`). | error | markup rules of the brief | Fixed: `(\mbox{\citealt{yi2025benchmarking}})`. |
| 3 | Disc-2, deleted bullet, outline.tex:219 | Same problem: `\mbox{\citep{arjovsky2017towards}}`. | error | markup rules of the brief | Fixed: `(\mbox{\citealt{arjovsky2017towards}})`. |
| 4 | Methods ranking paragraph, outline.tex:127 | "rules were fixed before the main benchmark was scored" is wrong: 19 main-benchmark fits had already been scored (they were not read). The claim is also unconfirmed for the new ranking rules. | error | outline.tex:257 (Supp. Note 10: "19 main-benchmark fits were scored and not read"); ANALYSIS_SPEC.md:490 (19 X1 rows scored 2026-10-12) | Fixed: "fixed before any main-benchmark score was read \tbd{confirm for the ranking rules}". |
| 5 | Intro, outline.tex:69; Disc-2, outline.tex:218 | `kurach2019large` was cited for "most GAN losses reached similar scores with equal tuning". Its refs.bib title is a study of regularisation and normalisation, and the author's own commented paragraph gives that finding to Lucic et al. only. | inconsistency | refs.bib:188-203; outline.tex:67 (author's commented text) | Fixed: cite `lucic2018gans` only. I did not open the full texts of either paper. |
| 6 | Header note, outline.tex:31 | "Tables 1--2 hold real design facts", but Table 2 now holds provisional ranges. | inconsistency | outline.tex:376 (Table 2 caption) | Fixed: "Table 1 holds real design facts and Table 2 provisional pilot ranges". |
| 7 | Ranking rules: Methods outline.tex:127; Res-2 :155; Supp. Note 6 :252 | Rules the ranking needs that the spec does not have: (a) rank rule and ties; (b) not-evaluable rule for six penalties (the spec's rule covers only the two arms of the old contrast); (c) the leave-one-seed-out interval for one penalty's reading (the spec defines it only for the difference d); (d) how the mean rank is formed when a real task is not ranked. | inconsistency (spec gap) | ANALYSIS_SPEC.md:103-104, 121-123, 127-132 | Left for the author. Inventing these rules is not allowed; a tie `\tbd` already sits in Note 6. |
| 8 | Methods :127 and Res-2 :155 vs Table 3 caption :393 | The text says "mean rank over the six real tasks", but the Table 3 caption says "over the evaluable real tasks". The two disagree whenever a real task is low-signal, which the spec says can happen with the conditioned decoder. | inconsistency | ANALYSIS_SPEC.md:382-383 | Left for the author: this depends on rule 7(d). |
| 9 | Methods :127, Supp. Note 6 :252 | "as in scIB, the mean rank" is not checked against scIB's own ranking procedure. | inconsistency (unverified) | Luecken et al. 2022 not opened | Left for the author: check against the paper or drop "as in scIB". |
| 10 | Label-transfer rule: Methods :140, Res-5 :184, Supp. Note 9 :256 | The spec has no label-transfer rule (it lists only "X10 downstream biology", exploratory). The rule in the outline (15-nearest-neighbour classifier, held-out batch, matched strength) cannot be checked against the spec. | inconsistency (spec gap) | ANALYSIS_SPEC.md:62, 65 | Left for the author: add the rule to the spec or label it exploratory. |
| 11 | Figure 6 caption (a), outline.tex:349 | "Expected best ... pooled over real tasks": the spec defines best-of-k only per task and decoder and gives no way of combining tasks. | inconsistency (spec gap) | ANALYSIS_SPEC.md:267-273 | Left for the author: state the rule (for example, the mean over tasks) or show per task. |
| 12 | Table 4 caption, outline.tex:419 | Standard tools are shown at "their best setting", which is tuned on the task, while each penalty is shown at an untuned strength. The spec defines the single-strength tables for penalties only. | inconsistency | ANALYSIS_SPEC.md:250-256 | Left for the author (design choice); at least label the tools' values as optimistic. |
| 13 | Supp. Note 5, outline.tex:251 | Missing fact: the highest common batch score behind b\* counts edge-extension strengths, but curves use only the six strengths. Without it, the rule "every penalty's curve reaches b\*" looks like it can never fail. | inconsistency | ANALYSIS_SPEC.md:92-94 | Fixed: added parenthesis. |
| 14 | Supp. Note 7, outline.tex:253 | Placement rule was missing two parts: only strengths with no failed fit in the choosing tasks are eligible, and ties go to the smaller strength. | inconsistency | ANALYSIS_SPEC.md:254 | Fixed: added. |
| 15 | Supp. Note 2, outline.tex:248 | The editor's added text repeated update steps, optimiser, latent size, hidden units, adversary input and posterior-mean scoring, which the same note already lists. | style | outline.tex:248 (later items of the note) | Fixed: the added text is cut to what is new (by-design differences, discriminator optimiser, strength zero, likelihood). |
| 16 | Supp. Note 3, outline.tex:249 | "on all cells" appeared twice. | style | outline.tex:249 | Fixed: the duplicate is removed from the added text. |
| 17 | Methods data paragraph, outline.tex:111 | ATAC not spelled out at first use. | style | brief item 6 | Fixed: "assay for transposase-accessible chromatin, ATAC". |
| 18 | Table 1 row, outline.tex:368; Disc-5 :234 | KL not spelled out in the main text (it is spelled out only in Supp. Note 1). | style | outline.tex:247 | Fixed: added "KL, Kullback--Leibler" to the Table 1 caption; Disc-5 now reads "Kullback--Leibler (KL) warm-up". |
| 19 | Intro, outline.tex:71 | scIB-E not spelled out. | style | refs.bib:134-144 (title gives no expansion) | Left for the author: the expansion is not in the sources available to me. |
| 20 | Figures-and-tables list, Table 3, outline.tex:292 | The list left out the Biology column of the Table 3 environment. | inconsistency | outline.tex:398 | Fixed: "mean biology at b\*" added. |
| 21 | Res-5 heading, outline.tex:180 | Cited Figure 4c for label transfer, which is in Supplementary Figure S3. | inconsistency | outline.tex:332 (Fig. 4 panels), :277 (Fig. S3) | Fixed: "; label transfer in Supplementary Figure S3". |
| 22 | Intro comments, outline.tex:78-80 | The author's note "this should be in methods" had been acted on (Methods :121, :130, :138), but had no DONE marker, unlike the other notes. | style | outline_author_round.tex:75-77 | Fixed: DONE comment added. |
| 23 | Order of first citation of supplementary items | Note 10 is cited at :127, before Notes 7-9. Table S2 (:91) comes before S1 (:111). S4 (:130) and S8 (:173) come before S3 (:189). Figure S4 (:164) comes before S3 (:184). Numbering itself has no gaps (Notes 1-11, Tables S1-S8, Figures S1-S5). | style | outline.tex:244 (heading) | Left for the author: renumbering touches about 20 references, and the brief gives no journal ordering rule. |
| 24 | Deleted Results and supplement items: outline.tex:151-154, 166-171, 178, 192-197, 219, 255, 266, 276 | `\item` and `\pp{}` sit outside `\deleted{}`. A build with `[final]{changes}` therefore prints empty bullets, plus an empty bold "." heading for old Res-4 and Res-8. | style | changes-package behaviour | Left: moving `\item` inside `\deleted` breaks the strike-out. Delete these lines when the changes are accepted. |
| 25 | Table 1, outline.tex:362-368 | Keeps "gradient penalty, weight 10" and "32 hidden units". The author moved the weight out of Methods into Supp. Note 1, and the editor moved the Sinkhorn smoothing into Table S2. | style | outline_author_round.tex:97 | Left for the author: Table 1 is a design table, and whether to keep these values there is the author's call. |
| 26 | Methods pilot paragraph, outline.tex:114 | States the all-task pilot design (ten strengths, three seeds) as fact, while Supp. Note 4 :250 leaves seeds and fit counts as `\tbd`. | inconsistency | pass2_changelog.md:121 | Left for the author: confirm the all-task pilot design. |
| 27 | Methods penalties paragraph, outline.tex:104 | The candidate justification says the reference batch contains every cell type. The highest-entropy batch need not contain every type. It is marked `\tbd`. | inconsistency | pass2_changelog.md:122 | Left for the author (already marked `\tbd`). |
| 28 | Methods secondary paragraph, outline.tex:138 | The Clopper--Pearson intervals treat the 30 fits of a cell as independent, but the 6 fits of each seed share one plain-scVI threshold. The spec records under-coverage (0.914). Disc-4 does not mention this. | inconsistency | ANALYSIS_SPEC.md:437-444 | Left for the author: add it to Disc-4 limitations or drop the intervals. |

Checked with no change needed:
- **Item 1.** In live text there is no mention of critic versus discriminator, the follow-up, the premise check, significance testing (other than "no significance tests"), margins, posteriors, Holm, power or minimal detectable effect, "(pre-)registered", 1,098 fits, 36 edge fits, or immune/simulation 1 as low-signal pairs. A search of outline.tex found every hit inside `\deleted`, the second argument of `\replaced`, or `%` comments. The old three-task pilot appears only as the labelled source of provisional values (:250, :264, :291, :376).
- **Rules that match the spec:**

  | Rule | outline.tex | ANALYSIS_SPEC.md |
  |---|---|---|
  | b\* | :121, :251 | :39 |
  | low-signal (gap < 0.02) | :121 | :121 |
  | first-crossing reading | :127 | :112 |
  | trade-off area | :138, :175 | :146-152 |
  | collapse, low-signal kept | :138, :176 | :221-224, :396 |
  | local versus global | :140, :182 | :237-245, :399 |
  | good-strength width | :138, :189 | :278-281, :411 |
  | single-strength placement | :130, :201 | :250-256 |
  | best-of-k | :130, :206, :253 | :267-273 |
  | sensitivity analyses and scaling checks | :252 | :285-288, :299-301 |

- Run-count arithmetic in Supp. Note 11 (:258): 80 + 40 + 6 x 480 = 3,000, which matches the spec (:75).
- **Item 3.** The Results labels Res-1 to Res-8 (heading "8 paragraphs") match the Figure 2-6 and Table 1-4 environments and the Figures-and-tables list (except #20 and #21, now fixed).
- **Item 5.** All 8 first-author `%` notes in outline_author_round.tex were acted on: lines 65, 75-77, 90-93, 97, 101-102, 110-111, 117 and 125.
- No em dashes in outline.tex.

## Not checked

- **Not compiled.** No TeX engine (tectonic, pdflatex) is installed, and `figures/mock/` is missing. The build check of the brief's preamble (doc-build-gate) was not run.
- Full texts of Lucic 2018, Kurach 2019, Luecken 2022 (scIB) and the other Introduction citations were not opened, so the claims in the citing sentences were not checked against them. refs.bib metadata was not re-verified.
- Word counts (abstract at most 200 words; paragraphs of 100-150 words) were not recounted. The changelog's own estimates say three Methods paragraphs are over 150 words.
- The mock PNGs, `methods_changelog.md` and `intro_changelog.md` were not available. The Supp. Note 1 and 2 solver and optimiser values come from methods_changelog via the editor and were not checked.
- Prose quality of the first author's Introduction and Methods was checked only for the items above, not line-edited.
