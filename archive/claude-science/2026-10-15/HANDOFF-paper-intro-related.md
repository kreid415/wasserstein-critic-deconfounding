# Handoff: paper intro and related work

**Session:** 2026-10-15 · 329 messages · 0 folds · frame 544c8a5f-7c01-4a8c-9cd2-00270b8c0eb7 (follows frame 5d857d52-e570-4a88-b55e-1dc0fdb1df37)
**Status:** Introduction, Related work and the reference list are in paper/paper.tex, reviewed in two rounds (0 blockers), fixed, built and pushed (branch `paper`, commit d556bb1). Nothing is running.

## Start message for the next chat
```
Chat name: Write Introduction Methods Supplement
Continue from HANDOFF-paper-intro-related.md (artifact id d9619ada-7e85-4cf9-9462-7b2463eb036b) in this project; read it before doing anything else.
Objective: finish the paper/paper.tex outline of locked information (Introduction, Related work and references are done and reviewed)
Next action: ask the user the open questions in the handoff in one round, then apply the answers to paper/paper.tex and rebuild
Open first: HANDOFF-paper-intro-related.md; paper.tex; review_disposition_intro_related.md
Lost with the old kernel: clone of branch paper at d556bb1 (re-clone)
Pending: none running; user decisions on scCRAFT naming, the 'first' claim, the preprint statement, coverage candidates
Do not redo: the two review rounds (all 50 findings dispositioned); the integration of the literature drafts
Before reusing any identifier, number or quote from the previous chat (frame 544c8a5f-7c01-4a8c-9cd2-00270b8c0eb7), search its archive.
```

## Objective
The user asked (2026-10-14) for paper/paper.tex: an outline holding all locked information, which they will write the revision of BIOINF-2026-1434 from. Locked = a user decision in CONSTRAINTS.md, a rule frozen at tag prereg-tier12-v2, or a fact verified in a file or a source text; everything else is \pending{what}{when}. This chat completed the Introduction, Related work and references; what remains are user decisions and items that lock with the X1 results.

## State
- Done and verified: [paper.tex]({{artifact:85b3cb1a-f56d-460a-ba66-39ea3b3c36db}}) = paper/paper.tex at d556bb1 (sha256 21bb94c2dadef78a); [paper.pdf]({{artifact:4ddf0d47-aa3b-413b-941d-549546345cff}}) (tectonic, 35 pages, no failing TeX warning class, 2 underfull boxes below the threshold, every page viewed, saved files byte-identical to the build); [refs.bib]({{artifact:e1b9885c-8002-492f-a0ac-0515b50247dd}}) (52 entries, all cited, no \unverified key left); [evidence.md]({{artifact:e493a4e8-1592-433f-a098-d0ecfd74c303}}) (claim check per key, three gap searches, limits; committed as paper/evidence.md).
- Reviews (RIGOR_REVIEWER): Introduction track on add2f73 ([report]({{artifact:ff66652f-c92d-4b19-8db7-ec52871d1937}}); 0 blockers, 16 fixes, 9 notes), Related work + Methods citations + refs.bib track ([report]({{artifact:aed01ae8-eb07-447b-a137-aaa3acee8e81}}); 0, 7, 10), focused re-check on d170dd5 ([report]({{artifact:52953dc3-fdfb-472a-a13f-155e7d0ff897}}); 0, 2, 6). Every finding and what was done: [review_disposition_intro_related.md]({{artifact:a1515951-4d46-4edc-998b-d826d50bce3a}}). Ledger entry files: [intro]({{artifact:24547ea4-fd57-4286-b480-049e50b6da40}}), [related]({{artifact:fae7af2b-ceda-4b7c-baeb-58046c048a3f}}), [re-check]({{artifact:4d01effd-8f8d-4659-92ca-072bb6d8067b}}).
- Not verified (declared limits, recorded in evidence.md): abstract only for goh2022batch, maan2024characterizing, johnson2007adjusting, stuart2019comprehensive, buttner2019test; metadata only for clopper1934use, gayoso2022python; preprint full text unreadable (bioRxiv HTTP 429), so 'the preprint used the scCRAFT backbone' is pending; several keys read in preprint/arXiv versions (listed in evidence.md).
- Pending user decisions added in this chat: whether the main text names scCRAFT (marked on 4 lines: Introduction composition-shift and 'Earlier comparison' lines, both Related-work scCRAFT lines); whether to state that no 'first' claim is made; the statement that the preprint's results are not carried over plus disclosure of its pipeline defects; whether to add the coverage candidates ([coverage_candidates_related_review.csv]({{artifact:cf884a05-d8fe-49ac-adc8-9db6a92925e9}})).

## Artifacts
- [paper.tex]({{artifact:85b3cb1a-f56d-460a-ba66-39ea3b3c36db}}): canonical outline source (artifact 3e045613-6807-4579-a8b9-e468796617e7); edit this, or paper/paper.tex on branch paper.
- [paper.pdf]({{artifact:4ddf0d47-aa3b-413b-941d-549546345cff}}): its build, for reading.
- [refs.bib]({{artifact:e1b9885c-8002-492f-a0ac-0515b50247dd}}) and [evidence.md]({{artifact:e493a4e8-1592-433f-a098-d0ecfd74c303}}): bibliography and its claim-check record (both also in the repository's paper/ folder).
- [review_disposition_intro_related.md]({{artifact:a1515951-4d46-4edc-998b-d826d50bce3a}}): the 50 review findings and their dispositions.
- [gap_search_screen.csv]({{artifact:83309f81-8e92-490b-82bd-ee380970beba}}): the Introduction reviewer's screened gap-search hits (175 works).
- [HANDOFF-paper-outline.md]({{artifact:317e8d49-fecc-49c6-a7ce-88602d5a5418}}): previous handoff (methods outline; its open questions are carried below).

## Kernel state (will be lost)
| in memory | rebuild cost | how to restore |
|---|---|---|
| clone `paperrepo` of branch paper at d556bb1 in the workspace | cheap | `git clone --branch paper https://github.com/kreid415/wasserstein-critic-deconfounding.git paperrepo` (public repo; to push, run in a bash call with credentials=['GitHub'] and push to the token URL, never store it) |
| nothing else of value | none | everything else is an artifact |

## Decisions
- Gap line locked to paper/evidence.md 'Gap search' (three searches with scope and limits) plus \pending{search repeated}{before submission}.
- The earlier JS-versus-W1 comparison is attributed to the July submission (oup-authoring-template.tex:134, :142, read); the preprint is cited only as its public version, replaced per SI-07.
- R-11 wording: X1 keeps posterior-mean input and standardisation off (SI-66); the A1 rule-R4 record flags two blocks (immune 0.0188 and sim1 0.0069, conditioned decoder); only the formal R2/R3 re-run is pending.
- Methods Backbone now says sysVI uses its own Gaussian likelihood on normalised values (fit_paper_config.py:404-413; SI-63), although SI-10's wording says ZINB for every arm and baseline.
- corani2017time renamed corani2017statistical, cited only at the SI-74 decision rule (not for the SI-73 pooling model). reid2025topology title follows bioRxiv versions 2-3 ('in'), not Crossref ('for'); @article so plainnat prints the DOI.
- No lab-notebook entries on branch paper: the notebook hash chain lives on final-review-fixes, where another chat commits.
- Not offloaded to Codex: the edits were small and judgement-dependent.

## Dead ends
- OpenReview forum pages and api.openreview.net answer with a bot challenge: do not try to get around it; use arXiv or PMLR.
- bioRxiv full text (.source.xml) answers HTTP 429 (Cloudflare 1015) from the sandbox; the bioRxiv API (api.biorxiv.org) works for metadata.
- doc-build-gate with engine='auto' picks a broken pdflatex in env latex: pass engine='tectonic'.
- claim-gate cg_receipt accepts only cg_* result objects as evidence, not strings.

## Next steps
1. Ask the user the open questions below in one round (ask_user, one tab per question).
2. Apply the answers in paper/paper.tex (the scCRAFT decision touches the 4 marked lines), run doc-build-gate (engine 'tectonic', env latex, view every changed page), commit on branch paper (check `git show --stat HEAD`), push, and save paper.tex and paper.pdf as new versions of artifacts 3e045613-6807-4579-a8b9-e468796617e7 and ce867e68-69c2-4c9e-8c76-22e98260ccdb.
3. When bioRxiv allows, read the preprint text and lock or drop the pending backbone claim; before submission, repeat the gap search and record it in paper/evidence.md.

## Open questions for the user
- Should the main text name scCRAFT (PAPER_PLAN section 3: only as the earlier backbone; July convention: no mention)?
- Should the Introduction state that no 'first' claim is made, and that the preprint's results are not carried over (with its pipeline defects disclosed, PAPER_PLAN section 1)?
- Add any of the eight coverage candidates (iDLC, scStarCorrect, Beaconet, Portal, AIF, ABC, IMGG; Li et al. 2022) to Related work?
- Carried from the previous handoff: an on-disk checkout of branch paper; fixing the atac overlap count (3,618 vs all 11,270 cells) in ANALYSIS_SPEC and the final-review record; title, authors, target journal and figure list.
