# Pass 15 change log (author id P15, 2026-11-06)

Branch `paper`, starting point `origin/paper` at bfdd82d (Pass 14). Tracked changes carry `[id=P15]` (cyan). All text edits are
written by `paper/tools/pass15_edits.py` (exact-substring edits; each old text must occur once). No number is added or changed.

## Independent re-review

Codex CLI 0.159.3, specialist profile RIGOR_REVIEWER, model gpt-6.1-sol, effort high, read-only, through cli-subagents
`agent_task.py` (status DECLARED). Input: `git diff 0053042..bfdd82d -- paper/outline.tex paper/supp* revision/reviewer_response.md`,
the Pass 11 findings (Claude Science artifact version d0b592c9), the Pass 12-14 change logs and ledgers with their sources, the
replication rule and results at sweep-replication a475ea3 (before and after the extension step), the supplement scripts and
CONSTRAINTS.md. Reviewer's own limits (its declaration): no PDF built; figure scripts read, not run; raw score and latent files not
re-audited; limited literature retrieval.

Pass 11 findings: F01, F02, F04, F06, F07, F08, F11, F12 fixed; F09 rejection agreed; F03 and F10 partly fixed; F05 rejection
disputed (see below). Replication claims checked against the committed rule and both result folders: they match (the rule is met
only after its own extension step, the result before it is reported, 1 reading is unreached, both sensitivity analyses are labelled
as defined after the rule). All Pass 12-14 ledger values matched their sources; Tables S6, S7 and S9 reproduced. Rules: 44 unique
main-text citation keys (limit 50); none of the study-status words ruled out by CONSTRAINTS.md SI-78 in the printed text; no supplement citations; the update steps are nowhere described as a limitation.

Every finding was checked by the lead against the cited lines before it was fixed:

| id | severity | finding | checked | fix |
|---|---|---|---|---|
| F03 | major | the per-task sign-flip keeps the two decoders of a task together but flips the tasks of the two nested pairs separately, so it does not settle the dependence it was added for | confirmed (`make_replication.py` groups by task; Disc-4 names the nested pairs) | Methods: the rule's test assumption stated (independent pair correlations, symmetric about zero), and the per-task version also called approximate; Res-2b and Disc-1 say the same; response updated. No new analysis in the paper (see "Open for the first author") |
| F10 | major | Supplementary Tables S1, S2, S4, S5, S8 and Notes 2-12 are described in the supplementary list but not supplied | confirmed (`supp_items.tex` holds Note 1, Tables S6, S7, S9 and Figures S1-S6; Table S10 is in the main file) | the supplementary list and the response now say which items this version contains and that the others are not yet written. Writing them is open (below) |
| N03 | major | "with five seeds the replication resolves within-pair differences of about 0.04" (Background, Methods, Res-2b, Disc-1, Disc-4, response) has no precision calculation behind it; 0.04 is only the rule's threshold for a large gap (AGREEMENT.md line 13) | confirmed (no calculation in AGREEMENT.md, PREDICTION.json, README.md or the design folder); a sixth occurrence, in the last Background paragraph, was found when the clean PDF was read | each occurrence now says that the rule tests only the sign of the gaps of 0.04 or more and that no precision calculation was made, so the order of closer penalties is not tested. Raised by the reviewer as an open question in its declaration; numbered here |
| F05 | minor | Disc-4 still says "no strength was chosen from the data" although the replication and single-strength strengths were chosen from sweep scores | confirmed | reworded: the grid was fixed in advance and the sweep reports all ten strengths; the later choices from sweep scores are named |
| N01 | minor | outline (Note 1 item) and response describe the discriminator loss as an expectation under q(z given x); Supplementary Note 1 and the code use the posterior mean | confirmed (`supp/note1.tex`: every penalty is applied to the posterior mean, u_i = mu_i) | outline item and response now give the posterior-mean form |
| N02 | minor | Figure S4: the trajectory-conservation row has no label (its first column, pancreas, is empty) | confirmed on the PNG | `figS4.py` puts each row label on the row's first drawn panel; Figure S4 regenerated |

Findings rejected: none.

Follow-up found when the response PDF was read: item 4 of "What changed overall" still called the 13 pairs exchangeable and did
not say the per-task version is approximate (F03, now reworded like Methods); `$\pm$0.04` did not render in pandoc (now `$\pm 0.04$`).
Follow-up found while building: the supplementary-contents sentence first closed its `\added` group after `\par`, which stops
the tracked build (fixed in 308acdb).

## Abstract

Cut by wording only (all numbers and caveats kept): "Single-cell batch correction"; "Instead of a planned fixed-strength run, the
two sweep strengths bracketing matched batch removal were refitted"; "and MMD with the unconditioned decoder"; "With this decoder";
"penalty choice"; "exceeded" twice; "; scANVI". `paper/tools/abstract_words.py`: 328 words, hyphen-split count 348 (limit 350; it
was 344 and 366).

## Clean copy

`\usepackage[final]{changes}` stops with an xkeyval error on the nested `\replaced` of the writer's-outline header. The clean copy is
therefore written by `paper/tools/accept_changes.py` into `paper/outline_clean.tex` (generated, never edited by hand;
`--check` exits 1 when it is out of date). Method: preamble copied unchanged; after `\begin{document}` the new text of every
`\added`/`\replaced` (nested markup included) is kept, `\deleted` and `\droppeditem` are dropped, comments are copied verbatim,
bullets (also those left with only a TeX comment), lists and `\pp{}{}` headings left empty by deletions are removed, and any markup left over stops the script. Tests in
`paper/tools/test_accept_changes.py`, including a check that the clean copy prints the same abstract the word counter reads.

## Other edits

- `README.md`: the data-download command carried a browser session cookie and browser headers; replaced by a plain
  `curl -L` of the same figshare URL (article 12420968, version 8). The cookie remains in the git history (open, below).
- `paper/REPRODUCE.md` (new): re-run commands per result, environments, seeds, data provenance; `paper/MANIFEST.sha256` (new).

## Open for the first author

1. Supplementary Notes 2-12 and Tables S1, S2, S4, S5 and S8 are not written (F10).
2. Whether to add a sign-flip sensitivity analysis by task family (nested pairs and both decoders together, 6 families, 2^6
   patterns). Computed for this decision only, not in the paper: 1 of 64 patterns reaches the observed mean (p = 0.015625), because
   the summed tau-b is positive in all six families. It would be a further analysis defined after the rule.
3. The seven author-only `\tbd` items (AI-tool statement, ethics wording, release tag, competing interests, funding, contributions,
   acknowledgements) are unchanged.

## Re-run, from the repository root (Python: /home/kendall/.claude-science/conda/envs/python/bin/python, Python 3.11, pandas 2.3.3, matplotlib 3.11.0)

```
git checkout bfdd82d -- paper/outline.tex revision/reviewer_response.md
python paper/tools/pass15_edits.py
python paper/tools/accept_changes.py            # writes paper/outline_clean.tex
python paper/tools/abstract_words.py            # 328 words; hyphen-split 348
python paper/tools/check_ledger.py
python paper/figure_scripts/figS4.py
python -m pytest paper/tools
```
