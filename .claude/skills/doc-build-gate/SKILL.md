---
name: doc-build-gate
description: "Gate every LaTeX, markdown/pandoc, HTML or slide deliverable before it is called built, compiled, rendered, verified or done. Load when writing or editing a .tex/.md/.qmd/.html paper, outline, review, report or deck; when embedding figures; when the user says compile, build, render, Overleaf, pandoc, PDF, 'figures missing', blank figure, overlapping legend or p-values, text cut off, duplicate figures, undefined reference/citation, or asks 'did it build / did you check'. Builds the exact saved file and fails on chat artifact markers or sandbox paths in sources, any TeX warning class, dropped images, blank/duplicate/low-res figures, text/legend overlaps, saved-vs-built mismatch, and unresolvable artifact ids; emits a build receipt."
---

> **Helper loading (Claude Code).** `kernel.py` in this folder is plain Python with no Claude Science API. Load it from a Python session or script with
> `p = ".claude/skills/doc-build-gate/kernel.py"; exec(compile(open(p).read(), p, "exec"), globals())`; the helpers are then defined in that namespace.

# doc-build-gate

**Applies:** before any LaTeX/md/qmd/html/slide deliverable is called built, compiled, rendered, verified or done; when writing/editing one or embedding figures; on compile/build/render/Overleaf/pandoc/PDF, blank/missing figures, overlaps, cut-off text, duplicates, undefined ref/cite, "did it build/did you check".
"Built" = exact saved file compiled by a real engine, every warning class classified, every promised figure embedded, receipt says so. `dg_*` helpers load with the skill; each raises `Dg*Error` (`.result`). Never catch-and-continue: fix the cause or pass explicit `allow=` and say so in the report.

**Protocol (in order; stop at first failure)**
1. Sources: relative paths only; figures in `figures/` beside doc, as `figures/name.png`. Never put artifact markers, sandbox paths or artifact-id filenames in .tex/.md/.html/.qmd.
2. `dg_overlap_check(fig)` on the live Figure before `fig.savefig`; `dg_figure_qa(paths)` on saved PNGs. After a fix re-run both; no fix claim without the pass.
3. `dg_check_sources(path, min_images=N)` (N = figures generated for the doc).
4. Build the exact file, never a copy: `dg_build_tex(path, expected_figures=N)` or `dg_build_md(path, to="pdf"|"html", expected_images=N)` (re-runs source gate, classifies warnings, counts images, hashes source before/after; writes `.dg_build/<stem>.build_receipt.json`).
5. `dg_page_ink_check(pdf)` (catches overflow/blank pages TeX is silent about: minted/verbatim/tables); `dg_render_pages(pdf)`; view EVERY page PNG with the Read tool (`pages_to_view`). "Page 1 of 3 fine" is not verification.
6. After copying the deliverable to where it is delivered (repo path, upload, zip): `dg_verify_saved(built_path, delivered_path)` per file. Overleaf/handoff bundle = .tex + figures/ + .bib zipped; extract the delivered zip into an empty dir and run `dg_build_tex` there.
7. Report from receipt verbatim: engine, pages, figures embedded vs expected, failing warning classes (none), source sha256 prefix, pages viewed. Before sending, `dg_verify_links(final_text)` (full UUIDs that resolve; label names its file).

**Other helpers:** `dg_parse_tex_log(log_or_path)` classify TeX warnings; `dg_count_figures(file, expected, exact=True, expected_captions=None)` pdf/html/docx/pptx (per-page/slide counts, HTML data-URI validity, captions; decks: `dg_count_figures("deck.pptx", expected=N)`; overlap/QA helpers still apply to deck figures); `dg_receipt(source, outputs, checks, out_path)` raises on any `ok=False`.
**Defaults (`strict=True` where applicable, i.e. all but `dg_render_pages`; `strict=False` returns dict `ok, problems/errors, details` for receipts):** `dg_check_sources(allowed_image_ext=None, min_images=0, check_labels=True, allow=None, allow_dynamic=False)`; `dg_build_tex(engine="auto", outdir=None, expected_figures=None, allow=None, overfull_pt=1.0, underfull_badness=10000)`; `dg_build_md(to="html", pdf_engine="auto", expected_images=None, allow=None)`; `dg_count_figures(exact=True, expected_captions=None)`; `dg_figure_qa(min_side=300, dup_hamming=6, dup_rel=0.15, allow_duplicates=None)`; `dg_overlap_check(min_frac=0.1, expect_labels=None, ignore=None)`; `dg_page_ink_check(min_margin_pt=18.0, allow_blank_pages=None)`; `dg_render_pages(outdir=None, dpi=100)`.

**Rules**
- `DgToolMissing`: run the printed `manage_packages` command or say the check did not run; check for a real engine before workarounds.
- Never raise a threshold or add `allow` just to go green; a failing underfull/overfull on a deliberately loose paragraph may be allowed with `allow=["underfull"]`; record the allowance in the report.
- Paired near-duplicates: `allow_duplicates=[("a.png","b.png")]`. Legend over axhline/axvline is warning-only; image/heatmap cells aren't data: look at them.
- New checker needs a positive control (known-bad input must fail).
- Manual only: titles vs plotted values, caption numbers vs render order, every page visually.
- State limits: TikZ/pgfplots/vector-only figures uncounted (use `expected_captions` or inspect pages); PDF count ignores images <32px; no overfull check for minted/verbatim (ink check substitutes); `dg_build_md` prefers LaTeX-family PDF engines and warns on typst/weasyprint fallback (drops column widths/longtable pagination).

**Helper fail conditions (code-enforced; each raises):**
- `dg_check_sources`: artifact markers; `/home/`, `/tmp/`, `/workspace`, `file://` paths; absolute or unresolvable image targets (LaTeX-style extension search, `\graphicspath`); 0-byte/corrupt images; markdown image paths with spaces (pandoc silently drops them); `\ref` to undefined/duplicate labels; cite keys absent from the .bib; too few images; disallowed image format.
- `dg_build_tex`: non-zero exit; stale/no PDF; source changed mid-build; any failing log class; figure-count mismatch. Log classes (`dg_parse_tex_log`): `undefined_citation, undefined_reference, multiply_defined, missing_file, missing_character, font_substitution, rerun_needed, overfull, underfull, error`; `other_warning` reported only unless `fail_on_other=True`.
- `dg_build_md` (always also an embedded-resources HTML probe): any pandoc WARNING/ERROR; images not embedded or fewer than referenced; missing/stale output; PDF image count below raster refs.
- `dg_count_figures`: count differs from expected. `dg_figure_qa`: blank/near-uniform (opaque and transparent), low resolution, identical files, near-duplicates (aHash + dHash + relative ink-map difference), and zero input images.
- `dg_overlap_check`: text/text overlap (titles, ticks, annotations, legend entries); legend over data, bars, text or another axes; text or legend outside the canvas; axes with no finite data (silent filter/column mismatch); `expect_labels` missing (guards against panels silently missing models).
- `dg_page_ink_check`: ink near page edge; blank pages. `dg_render_pages`: no pages. `dg_verify_saved`: any byte difference. `dg_verify_links`: hash prefixes, invented ids, mislabeled links.
- Coverage: latexmk/pdflatex/biber paths are implemented but exercised only where those binaries exist; tectonic and pandoc paths are covered by `doc-build-gate_tests.py`.

Details, rationale, examples, failure modes: references/full.md (also references/checklist.md: manual checks, slide defects, positive-control recipe; references/log-classes.md: each warning class, real log line, fix; references/engines.md: engine selection, install commands, bibliography handling, output layout); read the relevant section when a check fails, a helper raises, or the case is ambiguous.
