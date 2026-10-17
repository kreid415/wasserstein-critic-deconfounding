---
name: claim-gate
description: "Inline gate to run BEFORE sending any message that says done, fixed, verified, complete, passing, saved, updated or 'all/every X', reports a number or count, summarizes results or a table, writes numbers into a paper/README/commit message/memory note, or repeats an earlier claim after the user pushed back. Use it whenever you are about to tell the user what a run, test, edit, figure or table shows. Reads values back from the saved file, scopes the claim to what was actually checked, reconciles counts, checks staleness, renders numbers from a results dict instead of typing them, and appends a claim receipt. Triggers: 'is it done', 'did the edit land', 'summarize the results', 'update the paper numbers', 'write the commit message', 'verify', 'are you sure', 'that number looks wrong'. Counterpart to reproducible (post-hoc audit)."
---

> **Helper loading (Claude Code).** `kernel.py` in this folder is plain Python with no Claude Science API. Load it from a Python session or script with
> `p = ".claude/skills/claim-gate/kernel.py"; exec(compile(open(p).read(), p, "exec"), globals())`; the helpers are then defined in that namespace.

# claim-gate

Run in the same turn BEFORE any message (chat, file, commit, memory note, TODO, reviewer reply) that: says done/fixed/complete/verified/passes/saved/updated/applied/matches; states a number, count, %, range, rank or comparison (better/best/only/tops/monotonic); says all/every/each/no/always about a set; summarizes a table/log/test/job/figure/file you did not just read; repeats an earlier claim; or writes any of this into paper/README/commit/memory/TODO. Pure planning or a question: skip.

`cg_*` helpers load with the skill. Each raises a ClaimGateError subclass (AssertionError) when strict (default) and returns {ok, summary, sources}.

## Protocol (in order, every time)
1. List atomic claims, one line each.
2. Read back from the saved file in this turn (not the variable, plan or memory); a finished render/save/copy/`|| true` job/commit is not evidence: `cg_readback(path, selector, allow_nan=False)`. Selector keys: column, where, row, expect_rows, key, index. Fails on missing/empty file, unknown column/key, 0 rows, wrong row count, NaN.
3. Scope the claim to what was checked: `cg_scope(claimed, verified, universe=None, noun=, strict=True)`; claimed may be 'all' only with universe=. Overreach raises (the scoped phrasing is in the message); with strict=False read s['scoped_phrase'] and use that wording when s['ok'] is False.
4. Reconcile counts (total = sum of parts, disjoint partitions, prose count = len(list)): `cg_reconcile(total=, parts=, partitions=, items=, **named_counts)`; `cg_reconcile_table(df, expected_n, by, expected_keys, value_col, balanced)`.
5. Staleness: outputs newer than code/inputs and last remediation commit: `cg_stale(outputs, inputs_or_code, since_commit, repo)`; dirs count by oldest (outputs)/newest (inputs) file. Before repeating or defending an earlier claim (yours, the user's, or from memory), re-read the file; the user's correction beats your recollection.
6. Numbers enter documents only from a results dict: `cg_render(template, values, allow_unused=False, strict_literals=False)`; placeholders `{{key}}`/`{{key:.3f}}`; readback results as leaves keep provenance; fails on unresolved/unused placeholder, None/NaN/non-scalar, float without format spec. Format at render time from raw values, never rounded ones; never retype a number from tool output; strict_literals=True for results paragraphs.
7. External facts (DOI, venue, citation count, benchmark timing, GPU utilization, 'measured') are verified only if a tool in this session returned them; else 'unverified' or omit. Never invent a measurement or source. Non-computable metric: report undefined with reason.
8. Prose about results: `cg_contradictions(text, table_df, tol, ignore_numbers, fail_on_quantifiers)`; fails when a prose number matches nothing, only elsewhere than the named row/column, only in absolute value, or a range excludes a table value. Recompute every all/every/only/top-k/monotonic sentence with an explicit predicate.
9. Append receipt: `cg_receipt(claim, evidence_list, not_checked=[...])` (refuses missing, failed or hand-typed evidence). If a check could not run, write 'not verified: <why>' instead of the claim; never soften an unchecked claim into wording that still implies checking.

If a helper raises: fix the artifact or narrow/remove the claim. Never catch the error and send anyway; never relax a tolerance or add ignore_numbers to get green without saying so in the receipt.

Other helpers: `cg_sha256(path)` hash BEFORE editing; `cg_changed(path, before_sha256)` after edit/save/copy (fails if byte-identical); support: `cg_error(kind)`, `cg_fail`, `cg_has_nan`, `cg_source`, `cg_native`.

Receipt (every result/completion message; one evidence line per check): `CLAIM RECEIPT` / `claim: <exact wording>` / `readback: <file> {selector} = <value> | <abs path> sha256:<12> @<mtime UTC>` / `changed: <file> differs from pre-edit version | <abs path> sha256:<12> @<mtime UTC>` / `reconcile: counts agree: {...} | in-memory` / `not checked: <what is not covered, or 'nothing outside the claim'>`. Chat 2-4 lines; documents: commit body or `*.receipt.txt`.

Visual claims have no helper: open the saved image (open it with the Read tool) and name what you saw in the receipt; a render call is not a view. After `cg_changed`, LOOK at the file before saying edits landed. Async/remote job: no claim until the result file is read: pass/fail counts via `cg_readback`, exit code not masked by `|| true`, then `cg_stale(results, [code], since_commit=...)`. Commit/memory/TODO: same gate; numbers from `cg_render` output or readbacks; state scope ('on Kang, 3 seeds'); reconcile any list-of-N. Limits (state them, do not paper over them): `cg_contradictions` matches numbers as written (small integers match by coincidence; review matched[*].source) and only returns count_claims/quantifier_claims for you to recompute; `cg_stale` uses mtimes (copies defeat it; prefer since_commit or hashes); `cg_render` flags typed numbers only as warnings unless strict_literals=True (allow figure/table numbers via `allow_literals`); no helper judges method validity (rigor-review), only whether the message matches the output.

Details, rationale, examples, failure modes: references/full.md (original full text); references/claim_types.md (evidence required per claim type, wording bank); references/incident_index.md (all logged incidents with the guard for each); read the relevant section when a check fails, a helper raises, or the case is ambiguous.
