# Point-by-point response to the reviewers of BIOINF-2026-1434

**Earlier submission:** *Topology Matters: The Trade-off Between Wasserstein Critics and Discriminators for Single-Cell Data
Integration*, *Bioinformatics*, manuscript BIOINF-2026-1434, decision of 19 July 2026 (rejected; the editor would not consider a
revised version).
**This submission:** *Benchmarking alignment penalties for single-cell batch correction*, *BMC Bioinformatics*, Research article.

Rewritten on 2026-11-05 (Pass 9) against `paper/outline.tex` on branch `paper`; it replaces the response of 2026-07-22, which
described experiments of an earlier design that are not in this paper. Section names below are those of the outline (Methods
paragraphs by their topic; Results paragraphs Res-1 to Res-8, Res-2b and Res-5b; Discussion paragraphs Disc-1 to Disc-5).
Every number below is copied from the paper text, whose numbers are listed with their source files in `paper/pass9_numbers.csv`
and, from Pass 12, `paper/pass12_numbers.csv` (every abstract number; `paper/tools/check_ledger.py` re-reads the ledgers from
their sources). Updated in Pass 12 (2026-11-05) after the Pass 11 rigour review. Table numbers are those the PDF prints (Pass 13):
the main-text tables are numbered 1 to 3, Table 1 (methods), Table 2 (ranking) and Table 3 (placement); the strength-range table was removed.

## What changed overall

We thank the three reviewers. Their comments led us to redesign the study rather than extend the earlier comparison:

1. **More penalties, one standard backbone.** Instead of one discriminator against one reference-based critic inside a custom
   autoencoder, the paper compares six alignment penalties added to the same stock scVI model (scvi-tools): the Jensen--Shannon
   discriminator, three Wasserstein-1 critics (reference, pooled and barycenter), maximum mean discrepancy (MMD) and the debiased
   Sinkhorn divergence, each with the decoder given the batch label (as scVI is usually run) or not (Methods, first two
   paragraphs; Table 1).
2. **The whole strength range, not one operating point.** Every penalty is fitted at ten strengths from 0.1 to 3,000 on eight tasks
   of the single-cell integration benchmark (six real, two simulated), with three seeds: 3,033 fits, none failed (Methods,
   "Penalty-strength sweep"; Res-1). Penalties are compared at matched batch removal, and trade-off curves, hypervolume, the share
   of non-dominated strengths, collapse rates and the range of good strengths describe the rest of each curve (Res-2 to Res-6).
3. **Standard tools at equal tuning effort.** Plain scVI, scvi-tools' built-in adversary, Harmony, Scanorama, sysVI, scANVI and
   uncorrected principal components (448 baseline fits) are compared with the penalties at equal numbers of tries (Methods,
   baselines paragraph; Res-7, Res-8; Tables 3 and S7; Figure 6).
4. **A replication across training seeds** refits each penalty at the two sweep strengths around the matched level with five new
   seeds, checked against a prediction and an agreement rule fixed before any of these fits was scored (Methods, "Replication
   across training seeds"; Res-2b). The rule's sign-flip test treats the 13 task and decoder pairs as exchangeable, although the
   two decoders of a task share its data, so its $p$-value is approximate; a sensitivity analysis that flips the signs per task
   (both decoders together), added after the rule was fixed, is reported beside it. Its result had not reached this pass of the response; Res-2b and this section are completed from the agreement test output.
5. **Direct biology readouts** per cell type (Methods, direct biology check paragraph; Res-5b; Supplementary Note 12; Table S10).
6. **Claims scoped to the evidence.** The study is described throughout as an exploratory benchmark on one backbone; the topology
   and reference-bottleneck explanations of the earlier paper are removed, and the limits are listed in Disc-4 and Disc-5.

---

## Reviewer 1

### R1.1 Local versus global batch removal

> "There is a mismatch between integration at local and global scales"

**Changed.** Local versus global removal is now one of the per-penalty analyses. The batch score is split into the three
neighbourhood metrics (iLISI, kBET, graph connectivity) and the two whole-embedding metrics (principal-component regression, batch
ASW), and each penalty's local-minus-global reading is taken at its crossing of the matched batch-removal level $b^*$ (Methods,
secondary-analyses paragraph; Supplementary Note 7). Res-5 and Figure 4c report it: at $b^*$ every penalty removed more batch effect
globally than locally on the ranked real tasks except the large chromatin-accessibility task, and the order among penalties
changed from task to task. Disc-2 discusses it: the local-versus-global differences do not explain the ranking. The earlier claim
that the critic mixes better locally is withdrawn.

### R1.2 Is there disjoint support between batches in the datasets used?

**Answered in part, by the design, not by a new analysis.** We did not add a measurement of support overlap. A planned check of
batch separability before integration was dropped from the redesign, so the paper makes no claim that the tasks have disjoint
support and no longer motivates the critic by vanishing discriminator gradients. On fairness: each penalty is now run with its
standard settings (the discriminator as in scvi-tools; the critics with a gradient penalty), every penalty is compared at a batch
removal that all of them reach, and every penalty's curve reached $b^*$ on every task and decoder pair, so no penalty is scored at
a level it could not attain (Methods, first paragraph and matched-batch-removal paragraph; Res-1). The Background cites the
theory on aligning batches with different cell-type proportions (Zhao et al.; Wu et al.) and the sysVI study, which is the setting
the reviewer describes.

### R1 minor 1 Degradation with the number of batches; topological bottleneck

**Removed.** The claim and the bottleneck explanation are not in the paper. The number of batches is not varied (Disc-5 lists it
as not tested). The redesign adds two critics that do not align to a reference batch, the pooled critic (each batch against all
others) and the barycenter critic (each batch against the minibatch Wasserstein barycenter), so the reference design can be
compared with joint alternatives on the same tasks (Methods, penalties paragraph; Res-2).

### R1 minor 2 Cell-type annotation and marker-gene preservation

**Added.** Automatic annotation is tested by label transfer: each batch is held out in turn and its cell types are predicted from
the other batches with a 15-nearest-neighbour classifier (Methods; Res-5; Supplementary Note 9; Supplementary Figure S3).
Marker-gene preservation is part of the new direct biology check (Res-5b; Supplementary Note 12; Table S10): on the four RNA tasks
the Jaccard overlap of each type's top 20 markers was 0.390 to 0.399 for every penalty and plain scVI, with differences from plain
scVI of at most 0.006, whereas Harmony and PCA were lower.

### R1 minor 3 Notation

**Changed with the model.** The custom autoencoder, and with it the cosine-similarity term, is replaced by stock scVI, whose
reconstruction term is the zero-inflated negative binomial likelihood of scvi-tools. All loss equations (scVI loss, discriminator
fool loss, the three critics, MMD, Sinkhorn divergence) are collected in Supplementary Note 1, where we use the notation the
reviewer suggests: the likelihood written with its mean and dispersion per gene, and the discriminator loss written as a
minimisation over the classifier of the expected cross-entropy under $q(z \mid x)$.

---

## Reviewer 2

### R2.1 The comparison is too narrow

**Changed.** Six penalties instead of two: one discriminator, three Wasserstein-1 critics (reference, pooled, barycenter) and two
network-free distances (MMD, Sinkhorn divergence), all on one scVI backbone with two decoder settings, beside scvi-tools' own
adversary and five standard tools (Methods; Table 1). The conclusions are stated for this backbone only, and Disc-4 says that the
ranking may not hold for other backbones, architectures or latent sizes.

### R2.2 Over-correction may be a tuning issue; show a Pareto front across the adversarial weight

**Changed.** No single operating point is used. Every penalty is fitted at ten strengths spanning four and a half decades, with
two extra edge strengths where needed, and compared where all penalties reach the same batch removal (Methods, sweep and
matched-batch-removal paragraphs). The trade-off curves (Res-3; Figure 3), the hypervolume and share of non-dominated strengths of
Li et al. (Res-4; Figure 4a), the collapse rate (Res-4; Figure 4b) and the width of the range of good strengths (Res-6; Figure 5)
describe each penalty over its whole range. Disc-4 states that this sweep is what Li et al. call scalarization, so it describes the
penalties as users run them, not the best front a multi-objective solver could find.

### R2.3 An ablation, not a mechanism; which cell types collapse, which references fail

**Changed.** The paper is now framed as a benchmark and the mechanistic language about topology and reference bottlenecks is
removed (title, Background last paragraph, Discussion). The direct biology check now shows which cell types lose most (Res-5b): rare
types lost purity in 23 of 48 task, penalty and decoder cells, and the largest drop was for the 46 ionocytes of the lung task with
the batch-conditioned discriminator, whose label-transfer recall fell to 0.326 against 0.797 for plain scVI. On references: the
reference batch is fixed by a rule (highest cell-type entropy, so that the target does not favour a few cell types) and is not
varied (Disc-5). We report how much of each task the chosen reference covers (Methods, penalties paragraph): it holds every cell
type in five of the eight tasks, and 10 of 17 (lung), 14 of 19 (human--mouse immune) and 3 of 4 (simulation 2) in the others, so
in those three tasks some cells of other batches have no cell of their type in the reference. We did not vary the reference or
test whether its coverage explains a penalty's performance.

### R2.4 The benchmark is weak for the strength of the claims

**Changed.** Eight tasks instead of three (pancreas, lung, immune, human--mouse immune, small and large chromatin accessibility,
two simulations), five standard tools plus plain scVI and the scvi-tools adversary, and an explicitly exploratory framing (Abstract;
Background last paragraph; Methods, sweep paragraph; Disc-4). We did not add atlas-scale data (see R3.4).

### R2.5 The weakness may come from the reference-based design

**Agreed and tested.** The pooled and barycenter critics use no reference batch (Methods, penalties paragraph). At matched batch
removal the pooled critic had the best mean rank with the batch-conditioned decoder (2.20) and the reference critic 3.80, whereas
with the unconditioned decoder the two were tied at 3.33 (Res-2; Table 2). The paper attributes no result to Wasserstein critics in
general, and neighbouring readings are often within one standard error (Res-2; Disc-1).

### R2.6 Biological over-correction is not directly validated

**Added: a direct biology check** (Methods; Res-5b; Supplementary Note 12; Table S10), with the four readouts the reviewer lists:
per-cell-type neighbourhood purity against plain scVI, rare-cell retention (purity and label-transfer recall of types under 1% of
cells), marker-gene overlap, and the label-transfer confusion table. It is exploratory: its rules were written after the sweep
scores had been read and before any of its values was computed. Over the six real tasks, purity differed from plain scVI by
$-0.052$ (batch-conditioned decoder) and $-0.077$ (unconditioned) for the discriminator, and by $-0.019$ to $+0.001$ for the
reference, pooled and barycenter critics and MMD. The direct readouts agree with the summary scores in singling out the
discriminator, and with the batch-conditioned decoder its loss fell more on rare types.

---

## Reviewer 3

### R3.1 Significance and generality; relation to scVI and other frameworks

**Changed.** The backbone is now stock scVI, the most widely used deep generative model for integration, and every penalty is
compared with plain scVI and with Harmony, Scanorama, sysVI and scANVI at equal numbers of tries (Res-7, Res-8; Figure 6). With
the batch-conditioned decoder, a penalty at one strength drawn at random beat plain scVI in 8 of 36 task--penalty cells and the
best of six strengths in 30 of 36, a count that is optimistic because the strengths were chosen and scored on the same data;
scANVI, which is trained with the cell-type labels that several biology metrics also use, was exceeded only on pancreas (Res-8). The single backbone
remains a limit, stated in Disc-4.

### R3.2 Robustness to architecture and hyperparameters

**Answered in part; the architecture is not varied.** The relative weight of the alignment term, the hyperparameter that sets the
trade-off, is swept over its whole range for every penalty (R2.2), the decoder is run with and without the batch label, and the
replication across training seeds checks the readings against new seeds (Res-2b). Network depth, latent dimension (10) and hidden
width (128) are fixed, as in scvi-tools' defaults. Disc-4 states this:

> "the ranking may not hold for other backbones, architectures or latent sizes"

We did not run an architecture sensitivity analysis.

### R3.3 Practical guidance

**Added.** Disc-3 and the Conclusions now give observations from the scores on these eight tasks, not tested on independent
data: the strength mattered more than the choice of penalty,
because the penalties were near-tied at matched batch removal (160 of 195 within-pair differences under 0.02) while going from one
to six tries raised each penalty's mean selection score by 0.012 to 0.053; plain scVI with the batch-conditioned decoder remained
a strong candidate, since a penalty beat it mainly when several strengths were tried and scored on the same data (an optimistic
reading); the strengths chosen on other task families lay between 0.1 and 100 across penalties with the batch-conditioned
decoder; the biology was worth checking directly at the chosen strength; and scANVI scored higher than every penalty on most
tasks when cell-type labels existed, although it is trained with labels that several biology metrics also use.

### R3.4 Atlas-scale data

**Not addressed by new data; stated as a limit.** The eight tasks are of the size of the single-cell integration benchmark.
Disc-4 says that the ranking may not hold for integrations of atlas scale. We did not add an atlas-scale task.

### R3 minor 1 Other modalities

**Answered in part.** Two chromatin-accessibility tasks (mouse brain, a small and a large dataset, as gene-activity matrices) are
now among the eight tasks, and their results are reported beside the RNA tasks (Res-2 to Res-6). snRNA-seq is not tested, and
chromatin accessibility is analysed only as gene-activity matrices (Disc-4).

### R3 minor 2 Downstream analyses

**Added.** Label transfer (Res-5; Supplementary Note 9), trajectory conservation inside the biology score of the two immune tasks
(Methods, data-and-metrics paragraph), and the direct biology check with marker-gene overlap (Res-5b). Differential-expression
testing is not added.

---

## Replication across training seeds (relevant to R2.2, R2.4, R3.2)

Pending. When this response was written (2026-11-05, Pass 9), the result of the agreement test (whether the four large sweep gaps kept their sign, the mean Kendall rank correlation over the 13 ranked pairs and its permutation $p$, and the readings per task) had not been received. This section, abstract item 7, Res-2b and the Disc-1 bullet will be written from the agreement test output; until then no replication result is claimed anywhere in the paper or in this response.
