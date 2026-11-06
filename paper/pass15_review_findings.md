# Independent re-review of the Pass 12-14 text (Pass 15)

Reviewer: Codex CLI 0.159.3, profile RIGOR_REVIEWER, model gpt-6.1-sol, effort high, read-only (cli-subagents agent_task.py, status DECLARED).
Input: git diff 0053042..bfdd82d of paper/outline.tex, paper/supp*, revision/reviewer_response.md; Pass 11 findings; Pass 12-14 ledgers and sources; replication rule and results at sweep-replication a475ea3; supplement scripts; CONSTRAINTS.md.
Every finding was checked against the cited lines by the lead (Claude Science) before it was fixed. Fix commits on branch paper: 0596835, 308acdb, 0a5548f, 79aa263 (tests 1f4d84e, f25f154, 76c4dab, 47414d2).

## Status of the Pass 11 findings

| id | reviewer status | evidence (reviewer) |
|---|---|---|
| F01 | fixed | revision/reviewer_response.md:119–122 now reports reference coverage and distinguishes it from varying the reference or testing its effect. Counts agree with paper/outline.tex:122 and paper/analysis_pass10/reference_coverage.json: 10 of 17, 14 of 19, and 3 of 4. |
| F02 | fixed | paper/outline.tex:239 explicitly says the illustrative simulation does not bound the reported intervals’ coverage. Supplementary Note 7 at :276 gives the estimand, assumptions, parameters and simulation error. Intervals are labelled nominal at :56, :159 and :358. |
| F03 | partly_fixed | paper/outline.tex:135 preserves the committed rule, qualifies its p-value and labels the per-task sensitivity as added after the rule was fixed. However, paper/tools/make_replication.py:68 groups by task, leaving the nested task pairs identified at outline.tex:231 in separate sign-flip groups. See finding F03. |
| F04 | fixed | paper/outline.tex:59 and :249 now scope guidance to these tasks, disclose selection and scoring on the same data, and explain scANVI’s label use. At :210, small differences are explicitly qualified; :449 distinguishes Sinkhorn’s 0.6846 from plain scVI’s 0.6850. |
| F05 | rejection_disputed | The rejection’s claim that the disputed sentence occurs only in removed text is incorrect. The active first argument of replaced at paper/outline.tex:232 still says 'no strength was chosen from the data'. The same active paragraph says replication strengths were set from sweep batch scores. Note 4 at :273 was clarified, but this occurrence remains. Its remaining severity is minor. |
| F06 | fixed | paper/pass12_numbers.csv supplies role-specific design, collapse-rate, interval-endpoint and threshold entries. paper/pass13_numbers.csv updates abstract locations, and pass14_numbers.csv covers replication additions. All 90, 40 and 84 rows, respectively, passed source checks; independent spot-checks also agreed. |
| F07 | fixed | paper/pass11_numbers.csv:2–11 now uses paper/figure_data/paper_numbers.json. pass12_numbers.csv:85–91 traces workstation counts to a dated manifest and historical counts to git:98732ff:docs/PREREG.md. That historical file at :685 records 405 and 'About 572'; outline.tex:281 identifies the cluster count as recorded rather than recounted. |
| F08 | fixed | paper/outline.tex:275 replaces the supplement citation with a main-text reference. No active citation commands were found in the supplement or included supplementary files. Counted 44 unique active main-text citation keys; counting historical text and comments too gives 49. |
| F09 | rejection_agreed | paper/outline.tex:11 explicitly identifies the intended tracked build. Removed historical claims should not be treated as current assertions. No prohibited study-status wording was found in accepted text. A clean submission PDF was not independently built. |
| F10 | partly_fixed | The original examples are supplied: equations in paper/supp/note1.tex, numerical Tables S6 and S7, and Figure S1–S6 inclusions in supp/figures.tex. Tables S6 and S7 reproduced exactly in memory. However, other referenced supplementary tables and the promised metric formulas remain descriptions of intended content. See finding F10. |
| F11 | fixed | paper/outline.tex:379, :421 and :447 define consecutive main tables without a counter adjustment; references use their labels. revision/reviewer_response.md:14–15 correctly calls them Tables 1, 2 and 3. Supplementary tables explicitly set their S numbers. |
| F12 | fixed | paper/outline.tex:239 and :276 explicitly include 33 fits for the conditioned immune discriminator, reference critic and pooled critic, alongside 30 and 36. These agree with tables/secondary.csv:44–46. Note 7 also states that the simulation excludes the 33-fit design. |

## Findings

| id | severity | location | problem | lead check | fix |
|---|---|---|---|---|---|
| F03 | major | paper/outline.tex:135,180,222; paper/tools/make_replication.py:68 | The per-task sensitivity preserves decoder dependence but still separates nested datasets. Its reported p-value therefore does not resolve the dependence concern. | confirmed | Methods, Res-2b, Disc-1 and response say both sign-flip p-values are approximate; the per-task version still flips nested tasks separately. No new analysis in the paper; a per-family version (6 families) would give 1 of 64 patterns, p = 0.015625 (computed for the authors' decision only). |
| F10 | major | paper/outline.tex:109,129,169,192,272,286–294; paper/supp_items.tex:5–13 | The document still directs readers to supplementary material that the supplied document does not contain. | confirmed | Partly fixed: the supplementary list and the response now say which items this version contains (Note 1, Tables S6, S7, S9, S10, Figures S1-S6) and that the others are not yet written. Writing Notes 2-12 and Tables S1, S2, S4, S5, S8 is open. |
| F05 | minor | paper/outline.tex:232 | The unqualified statement that no strength was chosen from the data remains in active text. | confirmed | Disc-4 reworded: grid fixed in advance, all ten strengths reported; later choices from sweep scores named. |
| N01 | minor | revision/reviewer_response.md:89–90; paper/outline.tex:270; paper/supp/note1.tex:5,22 | The response and supplementary outline describe an expected discriminator loss under the posterior, but the supplied equation uses posterior means. | confirmed | Outline Note 1 item and response give the posterior-mean form, matching Supplementary Note 1. |
| N02 | minor | paper/figure_scripts/figS4.py:29–38; paper/supp/figures.tex:23,29 | The trajectory-conservation row has no metric label in either Figure S4 panel. | confirmed | figS4.py places each row label on the row's first drawn panel; Figures S4a/S4b regenerated. |
| N03 | major | outline.tex Background, Methods, Res-2b, Disc-1, Disc-4; reviewer_response.md | 'With five seeds the replication resolves within-pair differences of about 0.04' has no precision calculation behind it (raised by the reviewer as an open question in its declaration; 0.04 is only the rule's large-gap threshold, AGREEMENT.md line 13) | confirmed | each occurrence says the rule tests only the sign of the gaps of 0.04 or more and that no precision calculation was made (six places; the Background one was found when the clean PDF was read) |

Found by the lead while reading the built PDFs (fixed in Pass 15): empty bullets in the clean copy (comment-only items), the response's 'What changed' item 4 still calling the pairs exchangeable, and `$\pm$0.04` not rendering in pandoc.

## Reviewer summary

Five items remain: two major and three minor. Replication numbers match the committed results: the rule is met only after its extension step, the pre-extension result is reported, and one reading remains unreached. Both added sensitivities disclose their timing. All Pass 12–14 ledger values matched their sources; Tables S6, S7 and S9 reproduced. Counted 44 active main-text references; found no active prohibited wording, supplement citations, or treatment of standard adversary steps as a limitation. Prior-work check: controlled loss comparisons and trade-off analysis already exist in [Li et al.](https://link.springer.com/article/10.1186/s12859-022-05003-3) and [Yi et al.](https://link.springer.com/article/10.1186/s13059-025-03869-z). The limited retrieval and targeted search did not establish whether the exact full comparison already exists.

## Reviewer's declared limits

- No LaTeX PDF was built or inspected. Numbering and inclusions were checked from source; all supplementary PNGs and the revised Figure 4 PNG were viewed.
- Supplementary scripts were read, but their file-writing figure workflows were not executed. Tables S6 and S7 were regenerated with writes intercepted in memory; Table S9 and replication_numbers.json were checked using --check.
- Replication interpolation, rankings, agreement statistics and counts were checked from both result folders. Training and scoring were not repeated, and individual raw score files and latent files were not independently audited; supplied score_validity.csv records were checked.
- Loss equations were checked for internal consistency, not exhaustively against the underlying training implementation. refs.bib was checked for citation-key coverage and relevant entries, not every bibliographic detail.
- A limited external methods and prior-work retrieval supplemented the file review. The logged OpenAlex, Europe PMC and arXiv searches were not repeated in full.
- Open question: the supplied replication rule and result files do not establish the stated precision claim that five seeds resolve differences of about 0.04; no supporting power or precision calculation was independently verified.
- The additional Pass 11 minor items F13–F15 were checked: the biology summary now matches the paper, 'alignment penalty' replaces 'adversary', and the arXiv-search limitation is corrected. Separate status entries were limited to the requested F01–F12.
