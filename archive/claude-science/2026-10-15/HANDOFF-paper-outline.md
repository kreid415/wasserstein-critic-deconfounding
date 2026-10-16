# Handoff: paper outline (paper.tex)

**Session:** 2026-10-14 to 2026-10-15 · about 255 messages · 0 folds · frame 5d857d52-e570-4a88-b55e-1dc0fdb1df37
**Status:** paper/paper.tex with all locked methods and supplement content is committed and pushed (branch `paper`, 622c8c2). The Introduction, Related work and reference list exist as reviewed-by-nobody fragments from the literature specialist and are NOT yet integrated.

## Start message for the next chat
```
Continue from HANDOFF-paper-outline.md (artifact id a6a84db2-553e-471a-a761-75c489e4c0e0) in this project; read it before doing anything else.
Objective: add the Introduction, Related work and references to paper/paper.tex (outline of locked information)
Next action: integrate the literature-track fragments and refs.bib, then a RIGOR_REVIEWER check
Open first: HANDOFF-paper-outline.md; paper.tex; framing_track.tar.gz
Lost with the old kernel: clone of branch paper (re-clone)
Pending: none running; fix R2/R3 wording in the framing text (R-11)
Do not redo: methods review (passed); on-disk worktree unasked
Before reusing any identifier, number or quote from the previous chat (frame 5d857d52-e570-4a88-b55e-1dc0fdb1df37), search its archive.
```

## Objective
The user asked (2026-10-14): in the paper dir, make a LaTeX file called paper.tex, first populated as an outline that they will write from, holding all locked information. Locked = a user decision in CONSTRAINTS.md (SI-42 is a lead-agent decision, not locked), a rule frozen at tag prereg-tier12-v2 (c88cce3), or a fact verified in a file; everything else is written as \pending{what}{when it locks}.

## State
- Done and verified: [paper.tex]({{artifact:1d2eae97-7ffb-4a2f-ba81-5cf735c16387}}) (25 pages; identical to paper/paper.tex at commit 622c8c2 on branch `paper` of github kreid415/wasserstein-critic-deconfounding, branched from final-review-fixes 08f1e3c) and [paper.pdf]({{artifact:ef75b2da-1d76-4500-8a27-d2bc7f09e2ba}}). Holds the legend and glossary, abstract skeleton, research-question block, Methods, Results and Discussion skeletons and the supplement (S-M1 to S-M18, S-D1 to S-D2, S-E1 to S-E6, S-X): 333 locked and 71 pending items ([lock_audit_v2.csv]({{artifact:222250a4-08f1-4ad0-a936-3ce3bac9c71a}})). RIGOR_REVIEWER: first review PASS_WITH_FIXES (6 blockers, 16 fixes, 19 notes) [rigor_review_methods_track_20261014.md]({{artifact:eef1e505-33a8-47df-9633-9df9f21ee501}}); fixes applied ([applied_fixes.csv]({{artifact:7206dcd8-7573-4c85-bed8-1c63aee44719}})); focused re-check PASS [rigor_recheck_methods_track_v2_20261014.md]({{artifact:bcbe7b6b-c015-4007-8ec5-d5b41ffbdd1c}}); its 5 notes applied by the lead. Build check: tectonic, 25 pages, no failing warning classes, every page viewed, saved files byte-identical to the build.
- Delivered, NOT integrated and NOT independently reviewed: literature track [framing_track.tar.gz]({{artifact:9ea19432-50a3-410b-a0cf-3fa761e81bcf}}) (intro.tex 760 words, related.tex 435 words), [refs.bib]({{artifact:a848c9d0-7cbe-4cde-af4e-91bad244f536}}) (49 entries, all marked verified), [evidence.md]({{artifact:d22794a5-3196-47b0-8461-4aaee248a796}}); 14 framing conflicts with PAPER_PLAN, key table and limits in [paper_outline_review_record.json]({{artifact:dd13e9fa-068d-4e83-9c4d-15b6fa0faf71}}). Its declared limits: maan2024characterizing, lotfollahi2020conditional and goh2022batch checked on the abstract only, clopper1934use on metadata only, gayoso2022python on metadata plus its first paragraph; 8 keys read as preprints; gap search = 4 OpenAlex queries + 1 web search.
- Known inconsistency to fix at integration: the framing fragments call the X1 adversary input and standardisation outcome pending. Rigour finding R-11 (lead-verified in docs/prereg/r4_a1_record.json: b_common is set by barycenter in 5 blocks and by mmd in sim1|1, never by sinkhorn) shows X1 keeps the defaults (posterior mean; standardisation off) under SI-66; only the formal R2/R3 re-run after the Sinkhorn redo is pending.
- Nothing is running (no sub-agents, no jobs). paper.tex currently shows Introduction and Related work as pending placeholders, 19 cited keys as unverified leads, and no reference list.

## Artifacts
- [paper.tex]({{artifact:1d2eae97-7ffb-4a2f-ba81-5cf735c16387}}): canonical outline source (edit this file from now on).
- [paper.pdf]({{artifact:ef75b2da-1d76-4500-8a27-d2bc7f09e2ba}}): its build, for reading.
- [framing_track.tar.gz]({{artifact:9ea19432-50a3-410b-a0cf-3fa761e81bcf}}), [refs.bib]({{artifact:a848c9d0-7cbe-4cde-af4e-91bad244f536}}), [evidence.md]({{artifact:d22794a5-3196-47b0-8461-4aaee248a796}}): literature-track output to integrate.
- [paper_outline_review_record.json]({{artifact:dd13e9fa-068d-4e83-9c4d-15b6fa0faf71}}): key verification table, 14 framing conflicts, reviewer report ids.
- [methods_track_v2.tar.gz]({{artifact:e0037fd8-3936-4d4d-9f7c-fee9610c55e7}}): SCRIBE fragments v2 (before the lead's last edits; paper.tex has them).
- [paper_assembly_kit.tar.gz]({{artifact:72107904-4931-438c-920a-2352f9e5b8b6}}): assembler, abstract, preamble and lead-edited fragments; only needed to re-assemble.
- [paper_preamble.tex]({{artifact:db40bf17-cf88-4e52-a0d3-0b81d4819099}}): shared preamble v2 (adds needspace to keep run-in headings with their text).

## Kernel state (will be lost)
| in memory | rebuild cost | how to restore |
|---|---|---|
| private clone `paperrepo` (branch paper at 622c8c2) in the workspace | cheap | `git clone --branch paper https://x-access-token:$GITHUB_TOKEN@github.com/kreid415/wasserstein-critic-deconfounding.git paperrepo` in a bash call with credentials=['GitHub'], then reset the remote URL to the plain https form |
| nothing else of value | none | paper.tex, fragments and kit are artifacts |

Env `latex` now has pypdfium2, pillow and numpy (installed for the build check; persistent).

## Decisions
- Location: new branch `paper` from final-review-fixes 08f1e3c, not final-review-fixes itself (keeps the pre-registration diff under review clean; the other chat 794143d3 commits there), and no on-disk checkout: the sandbox already hides 4 of the 12 checkouts in D/code beyond its git-structure budget, and a new one could hide wcd_final-review-fixes. The user has not been asked yet.
- paper/.gitignore negates the root `*.tex` ignore rule (root .gitignore line 26) with !paper.tex and !refs.bib.
- Commit author wcd / noreply@localhost (repository convention); push with the token in the URL, never stored.
- Citations: \citealp{key} inside existing parentheses for verified keys (natbib, plainnat, author-year); \unverified{key} prints [?key] for leads.
- R-11 accepted (see State). R-01 accepted: all 11,270 atac_small cells are in atac_large (batch + barcode, cell types agree; 3,618 identical names); ANALYSIS_SPEC sections 1 and 16 item 6 and final-review I073 still carry the name-based 3,618 (reported, not fixed there).
- Correction to tell the user: in the readiness answer of 2026-10-14 the lead said the low-signal flag could change after the Sinkhorn redo; it can only persist.

## Dead ends
- doc-build-gate with engine="auto" picks latexmk/pdflatex, broken in env latex (no pdflatex.fmt): pass engine="tectonic".
- The first commit silently dropped paper.tex because of the root `*.tex` ignore: check `git show --stat HEAD` after every commit.
- assemble.py reads paper_preamble.tex from the working directory: run it from the folder that holds the preamble.

## Next steps
1. Integrate the literature track into paper/paper.tex: replace the Introduction and Related work placeholder items with intro.tex and related.tex from framing_track.tar.gz; add paper/refs.bib; turn \unverified{key} into \citealp{key} for every key present in refs.bib (check the 19 methods-track keys, incl. corani2017time, mescheder2018training, miyato2018spectral); replace the References pending block and the leads list with \bibliographystyle{plainnat} and \bibliography{refs} (keep a leads list only for keys missing from refs.bib).
2. Fix the R2/R3 inconsistency in the framing text (State, third item).
3. RIGOR_REVIEWER check of the integrated Introduction and Related work: claims against evidence.md and the sources, the 14 framing conflicts, the abstract-only keys, locked versus pending classification.
4. doc-build-gate on paper/paper.tex (engine tectonic; view every changed page); commit on branch paper; push; save paper.tex and paper.pdf as new versions of artifacts 3e045613-6807-4579-a8b9-e468796617e7 and ce867e68-69c2-4c9e-8c76-22e98260ccdb (version_of).

## Open questions for the user
- Should an on-disk checkout of branch `paper` be created (for example D/code/wcd_paper)? It risks hiding another checkout from the sandbox; removing old review worktrees (needs the user's approval) would free room.
- Fix the atac overlap count (3,618 vs all 11,270 cells) in ANALYSIS_SPEC and the final-review record as part of the pre-registration diff?
- Title, authors, target journal and figure list are pending the user's decision.
