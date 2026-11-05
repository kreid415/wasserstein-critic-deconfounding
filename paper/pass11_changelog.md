# Pass 11 change log (author id P11, 2026-11-05)

Branch `paper`, file `paper/outline.tex`. Starting point: `origin/paper` at 717cc4c (Pass 10). Tracked changes carry `[id=P11]` (blue).
Every edit is written by `paper/tools/pass11_edits.py` (each old text must occur exactly once, otherwise it stops); the numbers it writes
are in `paper/pass11_numbers.csv`. The replication text (Res-2b, abstract item 7, the Disc-1 bullet and closing sentence) was not touched.

## 1. Figure 1 replaced (mock removed)

- `\includegraphics` of Figure 1 now points to `figures/fig1.png` (was `figures/mock/fig1.png`). Caption replaced (`\replaced[id=P11]`, old caption shown struck through).
- Header note (line 43): "Figure 1 is a placeholder schematic and Figures 2--6 are drawn ..." replaced by "Figures 1--6 are drawn from the sweep analysis ..." (tracked P11).
- New author definition `P11`; one comment line in the figure block records that Figure 1 is now `figures/fig1.png`.

## 2. How Figure 1 is made

- Script `paper/figure_scripts/fig1.py` (shared settings in `figcommon.py`: the `figure-style` ladder 8/7/6 pt, penalty colours of Figures 2-6; `check_and_save` got an optional `exts` argument so Figure 1 also writes SVG; the PNGs of Figures 2-6 rebuild byte-identically).
- Outputs `paper/figures/fig1.png` (300 dpi), `fig1.pdf`, `fig1.svg` (text kept as text; element ids fixed by `svg.hashsalt`, so PNG and SVG rebuild byte-identically; PDF bytes differ between runs). Listed in `paper/figures/MANIFEST.sha256`; `make_figures.sh` rebuilds Figures 1-6.
- Inputs: `paper/figure_data/paper_numbers.json` and `curves.csv` (already in the repository with SHA-256 in `figure_data/MANIFEST.sha256`; the originals in `sweep_analysis_final_20261105` have the same hashes, checked 2026-11-05). No new data files.
- Panel a: drawn from matplotlib shapes, no icons, no attribution file needed. Names all six penalties in two groups (adversarial: JS discriminator, 1 update step; Wasserstein critics reference, pooled, barycenter, 5 update steps with gradient penalty; non-adversarial: MMD with fixed multi-scale Gaussian kernel, Sinkhorn divergence), the decoder with and without the batch label, and loss = scVI loss + strength x penalty. The update steps appear as each method's setting, not as a limitation.
- Panel b: one real pair, ATAC small with the unconditioned decoder; discriminator, reference critic and MMD (one per penalty family). The script recomputes the first crossing of b* from `curves.csv` and asserts it equals `bio_at_bstar.*` in `paper_numbers.json` (tolerance 1e-9), that b* is `pair.atac_small.cond0.bstar`, that all curves start at the plain-scVI point `pair.atac_small.cond0.b0`, that each reading status is "bracketed", that the number of fitted strengths equals `n_strengths`, and that there are no failed fits.

## 3. Choice of example (differs from the suggestion in the brief)

The brief suggested pancreas with the batch-conditioned decoder. In the data its readings for the six penalties lie within 0.011 of each other and at the height of plain scVI (b* is 0.012 above plain scVI's batch score), so the three reading arrows would run through the plain-scVI star and its curves are non-monotone from the first strength (for the reference critic the batch score falls from 0.672 to 0.666 at strength 0.1). ATAC small with the unconditioned decoder (b* is 0.120 above plain scVI) has readings of 0.659, 0.666 and 0.692 for the three penalties, separated enough to read, and curves that rise to b* before they wander. This is the unconditioned decoder; the batch-conditioned pairs are in Figure 3 (row a). Changing the pair is a one-line edit of `TASK, COND` in `fig1.py`, followed by `pass11_edits.py` on the Pass 10 text.
