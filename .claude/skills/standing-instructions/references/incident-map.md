# Incident map (cluster D)

Source: `workflow_incidents.csv`, cluster D (69 incidents). Step numbers refer to the Protocol in SKILL.md.

| id | incident | step | mechanism |
|---|---|---|---|
| 3 | User repeatedly had to redirect agent back to outstanding requests | 5 Questions first | si_open_questions |
| 20 | User repeatedly had to redirect plot design | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 32 | User's topic-coverage-audit request never incorporated into plan | 5 Commitment written into the artifact | protocol + si_add of the requirement with a require_* check on the plan |
| 44 | Scope creep: unrequested rebuilds and replacements beyond brief | 4 Scope | si_scope_check |
| 54 | Silently reversed user-confirmed 3-pair dataset design to 2 pairs with fabricated rationale | 4 No unilateral change | proposal first; si_add of the chosen option; si_check_output require_* on the design doc |
| 62 | User repeatedly corrects agent's conceptual overreach and invented claims in narrative outline | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 63 | Silent model-description fix omitted from correctness review | 4 Scope (list every edit; durable actions need approval) | protocol (manual) |
| 89 | User questions usefulness of self-assembled fits and modeling choices | 5 Questions first | si_open_questions |
| 92 | Agent published a persistent skill without an explicit request | 4 Scope (list every edit; durable actions need approval) | protocol (manual) |
| 97 | Agent repeatedly conflated separate genetics and organoid-intelligence projects | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 100 | User rejected itemized formatting, wanted prose | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 115 | README.md ships with self-contradictory file counts | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 120 | Outline delivered as full prose, not a skeletal outline | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 127 | Scope creep beyond requested walltime-padding update | 4 Scope | si_scope_check |
| 142 | Incorrect constructed-latent count propagated across verdict table | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 192 | PROTOCOL.md left with disproven numeric claims after later discovering opposite result | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 201 | User made extensive repeated manual edits and corrections to manuscript wording | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 262 | Slide 23 eta2/MI mismatch 'fixed' by relabeling axis without recomputing | 4 No unilateral change | proposal first; si_add of the chosen option; si_check_output require_* on the design doc |
| 263 | 'Cleanly separated' claim retracted hours late; self-contradicted slide | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 277 | paper.tex left stale 'nearly independent directions' claim after correction | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 292 | program_counterfactual.py docstring contradicts its own script's output | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 293 | VAE models omitted from representability analysis without disclosure | 4 State what was left out | protocol (manual) |
| 296 | Previously-corrected block-identifiability misclassification reintroduced into paper | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 298 | User pushes back on coordinate-naming methodology framing | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 332 | Wrong path recorded as durable in persistent memory, never corrected | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 340 | Large backlog of unresolved questions and corrections from user | 5 Questions first | si_open_questions |
| 351 | block_DCI_disentanglement tagged 'headline' despite contradicting its own data | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 355 | Agent added unrequested analysis/narrative to outline | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 358 | Agent unilaterally collapsed entire predictive metric class to one representative metric | 4 No unilateral change | proposal first; si_add of the chosen option; si_check_output require_* on the design doc |
| 361 | Markdown outline rendered with no embedded figures | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 367 | Oracle-ceiling narrative omitted a contradicting data point (ST_Liver) | 4 State what was left out | protocol (manual) |
| 369 | Undisclosed exclusion of a dataset when answering coverage question | 4 State what was left out | protocol (manual) |
| 379 | User frustrated by agent curating/omitting data for external analysis bundle | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 380 | User had to repeatedly re-explain block-MIG/block-DCI generalization | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 389 | User repeatedly pushed back on jargon-heavy, awkwardly phrased question text | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 395 | Agent implemented the outline restructure option the user had not chosen | 4 No unilateral change | proposal first; si_add of the chosen option; si_check_output require_* on the design doc |
| 397 | Methods-count mismatch between outline text (317) and supplementary table (316 rows) | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 422 | User pushback on cherry-picked, non-reproducible final run plan | 4 No unilateral change | proposal first; si_add of the chosen option; si_check_output require_* on the design doc |
| 432 | Agent advanced plan without answering user's prior questions | 5 Questions first | si_open_questions |
| 439 | Retracted causal claim about UCE NaN bug restated as fact in summary | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 440 | Headline DCI metric wrongly forced per-dimension scoring for block-structured models | 4 No unilateral change | proposal first; si_add of the chosen option; si_check_output require_* on the design doc |
| 456 | User repeated instruction to exclude non-human genes from foundation models three times | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 496 | Introduction section still not properly separated after rewrite | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 499 | Figures still referenced by artifact ID instead of filename | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 509 | Unrequested detailed metadata added for excluded paper | 4 Scope | si_scope_check |
| 514 | Shared/private latent-space figure built too narrowly scoped | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 515 | Repeated instructions needed to keep unwanted papers out of the review | 2 Capture; 3 Check | si_add + si_check_output (pattern or manual) |
| 526 | User rejected agent's custom 'fairer' scIB substitution | 4 No unilateral change | proposal first; si_add of the chosen option; si_check_output require_* on the design doc |
| 550 | Final summary reversed an already-established correction on batch effect | 6 Propagate corrections | si_propagate (+ extra_texts for memory/artifacts) |
| 554 | User demands direct evidence after counterintuitive regularizer result | 5 Questions first | si_open_questions |

## Not mechanized

Communication-style, status and presentation complaints, and one-off clarifications. No pattern a helper can check; when the user states a lasting preference, record it with `si_add`.

| id | incident |
|---|---|
| 13 | Confusing explanation of cluster-autoscout fan-out behavior |
| 21 | User pushback on usefulness of signal-free (lambda=0) specificity test |
| 43 | Artifact versioning not applied, risking proliferation of duplicate files |
| 73 | User had to repeatedly demand more experiments |
| 93 | Session-handoff skill lacked clear usage instructions |
| 128 | User had to propose documentation fix for cross-skill discoverability |
| 134 | User repeatedly had to request figure fixes and clarifications |
| 138 | Duplicate label collision from copying tradeoffs table |
| 172 | Smoke-test summary omitted a flagged config that met the agent's own criterion |
| 179 | User repeatedly had to ask for status/ETA |
| 199 | Agent presented an explicitly non-conclusive pilot result as a substantive finding |
| 211 | Ensembles citation placement declared 'fine' without being checked |
| 249 | User rejects presentation for omitting requested metric findings |
| 272 | User had to compile a large consolidated list of slide defects in one message |
| 320 | User had to redirect agent from writing tasks to experimental steps |
| 385 | User had to demand source-checking before suggestions after unverified Szabo annotation |
| 481 | User flags contradiction between narrow and broad experiment framing |
| 482 | User presses for concrete implementation details |
| 503 | Fragile pagination workaround built before checking for a real LaTeX engine |
