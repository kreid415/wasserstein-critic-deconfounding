# Working style (always loaded)

Translated from the user's Claude Science profile rules. Source row ids are in `README.md` in this folder. Terms: SI-xx = numbered standing rule in `CONSTRAINTS.md`.

## Answers
1. Answer first. Every sentence carries a number, identifier, decision or action. Use tables over prose. No filler, hedging, restating the question, preamble, recap of tool output or closing summary.
2. Never shorten results, provenance, reproducibility, rigor or safety content to save words. Put long detail in a file and link it.
3. Write so a genetics PhD student who has not read earlier messages can follow: spell out each acronym at first use in each message; give every internal label (SI-xx, NB-..., X1, A1, P1) its plain meaning in the same sentence; define project-coined terms in plain words; no tool vocabulary unless asked; no metaphor jargon. Plain swaps: "load-bearing" becomes "what the result depends on"; "gate" becomes "check that must pass before X"; "canary" becomes "quick test run"; "fan out" becomes "run in parallel"; "harness" becomes "the app"; "surface" becomes "show" or "find".
4. Read PDFs as text (pdftotext, pypdf, or a DOI full-text fetch), not as page images. Look at page images only for figures, tables or layout, at most 2 pages at a time; run OCR on scanned PDFs first. Put this in subagent briefs.

## Decisions
5. One decision per message, at most 120 words: the question in one plain sentence, what happens if the user does nothing, your pick with a one-sentence reason. Offer 2 or 3 options; each label at most 8 words, each description at most 20 words; put the recommended option first.
6. Everything else (what was checked, test output, exact replacement text, line numbers) goes into a saved, linked details file (`docs/claude/details/<topic>.md`): moved, never dropped. If the user says a question is unclear, rewrite it shorter with a concrete example, never longer.
7. Only ask about decisions that change what is run, measured or analysed (SI-92: defer other decisions until after the experiments are launched). Decide wording, provenance, documentation and tooling choices yourself using your recommended option and record them in the lab notebook, or queue them until after launch.

## Delegation (replaces Claude Science sub-agent and model-pinning rules)
8. Work that needs more than about 5 tool calls and is self-contained goes to a Task subagent from `.claude/agents/` with a self-contained brief and a stop point (about 60 tool calls). Run independent subagents in parallel. Keep in the main session: planning, talking to the user, decisions, tasks of 5 calls or fewer, live back-and-forth, and checking and combining subagent results.
9. Use `model: sonnet` for mechanical and bounded work. Use opus for final review verdicts, interpreting analyses, scientific judgment calls and anything biosecurity-sensitive (stop and ask the user for the latter). Check every sonnet result independently before reporting it (rerun the acceptance command, check the claim against the file, or have opus re-check a sample). Match the subagent to the specialist role (literature, rigor review, notebook, repo audit, figures, manuscript prose, cluster operations); see `docs/claude/SKILLS.md`.
10. Subagent briefs must include the output format required, the files to read, the blinding rule, and the instruction not to push, tag or ask the user questions.

## Sessions
11. Before a long session ends, or when the conversation has grown very long, update `docs/claude/STATUS.md` (what is done, what is running with job ids and where outputs go, the next step, open decisions) and commit it. Give the user a short paste-ready starter message for the next session. Never rotate silently.
12. Schematic or icon-based biological figures: use only licensed icon sources (Bioicons, SciDraw via Zenodo DOIs), one consistent editable SVG, export SVG, PDF and 300-dpi PNG plus an attribution file (source, author, licence, changes). A schematic never replaces a panel that should plot data.
