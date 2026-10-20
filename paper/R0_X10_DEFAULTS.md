# Proposed defaults for the premise check (R0) and the label-transfer analysis (X10 minimal)

Status: proposal of 2026-10-19, written before any X1 or X13 score or latent was read. Only committed files were opened
(branches `final-review-fixes` and `paper`). Nothing under `/home/kendall/experiment_data` was opened. No repository file was
edited, nothing was committed.

Words used: **X1** = the main experiment (8 arms, 2 decoders, 6 strengths, 5 seeds, 8 tasks). **Strength** = the penalty weight
lambda of an arm. **b\*** = the matched batch-removal level of a task and decoder (PREREG.md:95). **kNN** = k-nearest-neighbour.
**AUC** = area under the receiver-operating-characteristic curve (0.5 = chance, 1 = perfect). **F1** = harmonic mean of precision
and recall of one cell type. **Run** = one fit with one seed. **Cell of the tables** = one task x decoder block.

## 1. The defaults, one line each

| Item | Default | Reason (one sentence) |
|---|---|---|
| R0(a) separability statistic | Macro-average one-vs-rest AUC of a kNN classifier (k = 15, Euclidean, raw `z`) that predicts the batch label of held-out cells, 5-fold stratified cross-validation (fold seed = the run's own seed), on the `none` (strength 0) latents; also report the smallest per-batch AUC. Wording "nearly separable" only if the macro AUC is >= 0.95 in all 5 seeds of a cell. | AUC has the same chance level (0.5) for 3 to 23 batches and ignores batch sizes; kNN has no capacity setting to tune and uses the neighbourhood size of the scIB graph (`score_scib_native.py:207`). |
| R0(b) "early" | The first ceil(0.10 x E) epochs of a run, E = that run's own number of epochs (82 to 400; 9 to 40 epochs). | The schedule length differs 5-fold across tasks (`manifest`, max_epochs 82 to 400), so only a fraction of training is the same clock for all tasks; epochs and gradient steps are proportional inside a task. |
| R0(b) "vanishes" | Per run, on the epoch trace of `adv_grad_norm`: the gradient has vanished when it stays at or below 10% of the run's own peak from some epoch on; T = that epoch / E (or "never"). Early = T <= 0.10. Also report P = mean of the last 10% of epochs / peak. | A within-run ratio removes the different loss scale and latent units of each arm, which makes absolute gradient norms incomparable across arms (`scvi_adversarial_plan.py:372,433-436`). |
| R0(b) strengths | All six registered strengths of each arm, every seed, no selection; tables per strength index g1 to g6 (n = 5 seeds each) and a tally of k of 30 runs. | Choosing a strength after seeing the traces would choose the contrast. |
| X10 classifier | kNN, k = 15, uniform weights, Euclidean, on the raw latent `z` (no standardisation). | The batch score B that defines b\* is measured on the same raw-latent kNN geometry (`score_scib_native.py:206-207`); no parameter is tuned. |
| X10 batch split | Leave-one-batch-out over all V batches of the task: train on V-1 batches, predict the held-out batch; every cell is predicted exactly once. Cells of cell types found in only one batch are removed before training and testing (decided from labels only). | It works unchanged for V = 3 to 23, has no random choice, and a type that exists in one batch cannot be transferred by any method. |
| X10 score | F1 per cell type from the pooled out-of-fold predictions; headline = unweighted mean over the evaluable cell types (macro F1); the per-type table is also reported. | Every cell type counts equally, as "F1 per cell type" asks. |
| X10 latents standing for b\* | The matched grid point of rule R4 for each task x decoder x arm: all 5 seeds' latents at that strength (`cells["task|cond|arm"]["matched"]` of the R4 x1 record), plus plain scVI (strength 0) and `scvi_adv` at its single fixed setting, both labelled "not matched". The gap `|matched_B - b*|` is printed next to every F1. | It is the registered definition of "the fit matched to b\*", it uses only batch scores and never F1 (no winner's curse), and it needs 75 latents per task instead of 375. |

## 2. Facts relied on (all read from committed files)

| Fact | Where |
|---|---|
| Each X1 latent `.npz` holds `z` (posterior mean), `obs_names`, `batch`, `celltype`, `config`, `history` (one list per logged key, one value per epoch). | `scripts/fit_paper_config.py:16` (docstring), `:351-361` |
| `adv_grad_norm` = mean over a minibatch's cells of the norm of the gradient of the unscaled adversary term with respect to the latent the adversary reads; the factor lambda (`d_coef`) is applied afterwards. Logged as one epoch average (`on_step=False, on_epoch=True`). | `scripts/scvi_adversarial_plan.py:300-301, 372, 377-378, 433-436` |
| The adversary reads the posterior mean or a sample, possibly standardised (`adv_input`, `zstd`; both are still placeholders `A2`, `A3` in the manifest, to be set by rules R2 and R3). | `scvi_adversarial_plan.py:355-360`; manifest X1 rows |
| `adv_acc` (discriminator only) = share of the minibatch's cells whose batch the head predicts correctly after its update, on the same minibatch: an in-sample accuracy. | `scvi_adversarial_plan.py:430-432` |
| Arms `none` and `scvi_adv` take the stock path and log neither trace. | `scvi_adversarial_plan.py:348-351`; `paper.tex` (branch `paper`) line 454 |
| X1 manifest: 3,000 rows = `none` 80 (8 tasks x 2 decoders x 5 seeds, strength 0), `scvi_adv` 40 (8 tasks x 5 seeds, all with the conditioned decoder, strength 0 = not used; checked with `/tmp/r0x10/check_scvi_adv.py`), and six arms x 480 rows (6 strengths x 2 decoders x 5 seeds x 8 tasks). Seeds 0-4. Batch size 128, 10 latent dimensions, train_size 0.9. | `manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv`, summarised by `/tmp/r0x10/inspect_manifest.py` (not committed) |
| Epochs per task, cells and batches (V): pancreas 400, 16,382, 9; lung 246, 32,472, 16; immune 239, 33,506, 10; immune_hum_mou 82, 97,861, 23; atac_small 400, 11,270, 3; atac_large 94, 84,813, 11; sim1 400, 12,097, 6; sim2 400, 19,318, 16. All eight epoch counts equal min(round(8,000,000 / cells), 400). | manifest `max_epochs`; `docs/PAPER_PLAN.md:228-230` (cells, V; "superseded in part" note); arithmetic in `/tmp/r0x10/derive_table.py` |
| KL warm-up length is scvi-tools' default 400 epochs, so the KL weight at the last epoch is (E-1)/400 when E < 400: 0.203 (immune_hum_mou), 0.233 (atac_large), 0.595 (immune), 0.613 (lung), 0.998 (the four tasks with 400 epochs). | `fit_paper_config.py:120-130, 367-376` |
| b\* = b0 + 0.5 (b_common - b0); matched strength = the failure-free grid point whose seed-mean B is closest to b\*, ties to the smaller strength; the R4 record keeps `bstar`, `bstar_low_signal`, `matched`, `matched_B`, `distance`, `flags` per cell or arm. | `docs/PREREG.md:95-96`; `scripts/prereg_rules.py:540-549, 572-573, 581-583` |
| P1 reads biology at b\* by linear interpolation on the seed-mean curve of the six registered points; extension points are not used as points. | `prereg_rules.py:604-624`; `scripts/analysis_x1.py:68-73, 76-120`; `docs/ANALYSIS_SPEC.md:82-84` |
| R0 and X10 are exploratory (no confirm or refute claim); no new fits; the classifier and batch split belong to the analysis script. | `CONSTRAINTS.md` SI-68, SI-103; `docs/PREREG.md:684-691`; `docs/ANALYSIS_SPEC.md:65-66` |
| Cell types in the latent file are checked equal to the prepped file's labels in the same order. | `score_scib_native.py:200-204` |

## 3. R0(a): are batches nearly separable before any adversary acts?

Specification for the script:

- Input: the 80 X1 latents of arm `none` (8 tasks x 2 decoders x 5 seeds). Read `z` and `batch` only. No cell-type label is read.
- Classifier: `KNeighborsClassifier(n_neighbors=15, weights="uniform")` on the raw `z`. The script asserts that
  `inspect.signature(scanpy.pp.neighbors).parameters["n_neighbors"].default == 15` so that "same neighbourhood size as the
  scIB graph" is checked, not assumed (not verified by me, see section 8).
- Folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=<the run's seed, 0-4>)` on the batch label. The script stops if a
  batch has fewer than 5 cells.
- Statistic per run: pooled out-of-fold class probabilities (n_cells x V, rows sum to 1, columns in sorted batch order), then
  `roc_auc_score(y, P, multi_class="ovr", average="macro", labels=sorted_batches)`. Also the smallest of the V one-vs-rest AUCs.
  (The scikit-learn documentation states the row-sum and column-order requirements; read 2026-10-19.)
- Report per task and decoder: 5 seed values, mean, min-max, and the smallest per-batch AUC. No pooling over tasks or decoders.
  The words "nearly separable" are used for a cell only if the macro AUC is >= 0.95 in all 5 seeds; otherwise "not nearly
  separable". 0.95 is a descriptive label fixed now, not a test.
- Why the conditioned and unconditioned decoders are both shown: a conditioned decoder receives the batch label, so it has less
  need to put batch information into `z`; the two blocks can differ.

## 4. R0(b): does the discriminator's gradient vanish early while the critics' persist?

Specification for the script:

- Input: `history` of the X1 latents of the six adversarial arms (discriminator; reference, pooled and barycenter critics; MMD and
  Sinkhorn divergences, which have no learned critic and are shown as a separate group), the six registered strengths of each
  arm's grid (indices g1 to g6; extension rows are not used), both decoders, seeds 0-4: 8 x 2 x 6 x 6 x 5 = 2,880 runs.
  `none` and `scvi_adv` have no trace.
- The script first lists the history keys of one file per arm and asserts that exactly one key matches `adv_grad_norm` (and
  `adv_acc` for the discriminator) and that its length equals the row's `max_epochs`; a non-finite or negative value stops the run.
- Per run with epoch trace g_1..g_E: peak = max g_e; threshold = 0.10 x peak; T = (the first epoch e such that g_e' <= threshold
  for every e' >= e) / E, or "never" if g_E > threshold; P = mean of g over the last ceil(0.10 E) epochs / peak. The run "vanishes
  early" if T <= 0.10 (it is below a tenth of its peak, and stays there, within the first ceil(0.10 E) epochs).
- Discriminator only: mean `adv_acc` over the first and over the last ceil(0.10 E) epochs, next to the largest batch's share of
  the cells (the accuracy a constant guess reaches; from labels only).
- Report per task, decoder, arm and strength index: number of seeds (of 5) with T <= 0.10, median T (">1" for never), median P and
  its range; and per arm the tally k of 30 runs. No threshold on k, no pooled test: the text states "k of 5" or "k of 30".

Known limits (state them next to the table):

1. The log has one value per epoch (0.25% to 1.2% of training), so a collapse inside the first epoch is invisible, and a gradient
   that is small from the start has a peak close to its level and shows as "persisting".
2. Absolute norms of different arms are not comparable; only each run's own ratio is.
3. The KL weight at an equal fraction of training differs by task (section 2), so "early" is not the same KL phase everywhere.
4. `adv_acc` is measured on the minibatch the head was just trained on, so it overstates held-out accuracy.
5. The "stalled batch metrics" half of the old criterion (`PAPER_PLAN.md:97`) cannot be done: no batch score is logged per epoch
   and X1 logging is unchanged.

## 5. X10 minimal: cross-batch label transfer at matched batch removal

Specification for the script:

1. Evaluable cell types: from the labels of the prepped file (not from any latent or score), the cell types present in >= 2
   batches. Cells of other types are removed from the task's data for X10. The list and the number of removed cells per task are
   written to a committed JSON before any latent is opened.
2. Latents: for every task and decoder, the five seeds of (a) plain scVI (strength 0), (b) each of the six matched arms at the
   strength in `cells["<task>|<cond>|<arm>"]["matched"]` of `docs/prereg/r4_x1_record.json` (an R4 extension strength is used
   if R4 chose one), (c) `scvi_adv`, conditioned decoder only. That is 75 latents per task, 600 in all. Strengths are read from
   the record, never recomputed; the script refuses a record whose `bstar` differs from the value `prereg_rules.r4_stage` recomputes from the same scores (as `analysis_x1.py:396-403` does).
3. Classifier and split: for each latent, leave-one-batch-out; `KNeighborsClassifier(n_neighbors=15, weights="uniform")`
   trained on the other V-1 batches (evaluable-type cells only), predicting the held-out batch. No random number is used.
4. Score per latent: F1 per cell type from the pooled predictions of all folds (F1_c = 2TP / (2TP + FP + FN)), and macro F1 =
   unweighted mean over the evaluable types.
5. Report per task, decoder and arm: 5 seed values, mean, SD and min-max of macro F1 (n = 5); the per-type table (5-seed mean);
   the matching gap `distance = |matched_B - b*|` (also the gap difference for the reference - discriminator pair) and the R4 flags
   (`edge_not_extended`, `edge_unresolved`). Two paired contrasts per cell, paired by seed index: arm minus plain scVI, and
   reference minus discriminator (the P1 pair): mean of the 5 paired differences and their range. No test, no p-value.
6. Cells with `bstar_low_signal` = true (b\* not reliably placed; `prereg_rules.py:573`) are tabulated but left out of any
   summary sentence, as P1 does. No other exclusion is allowed; in particular no arm or cell is dropped for its gap or its F1.

Cost (not measured): 7,050 kNN fits over 600 latents (sum of V over tasks = 94; 94 x 75). The largest task (97,861 cells, 23
batches) dominates. A pilot at production size is needed before a time is quoted (section 8). CPU only; no cluster request.

Why the matched point and not interpolation: reading F1 by interpolation along each arm's curve (the P1 rule) needs every grid
latent (375 per task, 3,000 in all), and interpolating a noisy per-type F1 adds a second source of error. The matched point
carries a gap, which is why the gap is printed. The draft X6 specification (`paper.tex` line 446, branch `followup-analysis-spec`,
not read by me) uses the same fallback: matched strength alone, labelled fixed-strength.

## 6. Choices that could bias the result if they were made after seeing X1 (fix them now)

| Choice | How it could bias | Fixed here |
|---|---|---|
| R0(a): statistic, classifier, k, number of folds | Trying kNN, linear and MLP variants and reporting the one nearest 1 favours the premise. | Macro one-vs-rest AUC, kNN k = 15, 5 folds |
| R0(a): the 0.95 label, macro vs minimum | A threshold set after the AUCs are seen can be placed to include or exclude tasks. | 0.95, all 5 seeds; minimum also shown, no threshold |
| R0(a): which tasks, decoders, seeds are shown | Dropping non-separable tasks. | All 8 tasks x 2 decoders x 5 seeds |
| R0(b): early fraction (0.10), vanishing level (0.1 of peak), peak vs first epoch as reference | Each can be tuned to make the discriminator collapse earlier or the critic persist. | 0.10, 0.1, peak, sustained |
| R0(b): which strengths are shown or headlined | The strength with the cleanest contrast. | All six, none selected |
| R0(b): which arms count as "critics" | Counting MMD or Sinkhorn with the critics or the discriminator. | Critics = reference, pooled, barycenter; MMD and Sinkhorn separate |
| X10: classifier, k, standardisation of `z` | A classifier that favours the critic. | kNN k = 15, raw `z` |
| X10: split (leave-one-batch-out vs random halves, which batches train) | A split where one arm transfers better. | Leave-one-batch-out over all batches |
| X10: evaluable-type rule, micro vs macro vs weighted F1, per-fold vs pooled F1 | Rare types or fold choice can reorder arms. | >= 2 batches, macro over types, pooled predictions |
| X10: which latent stands for b\* (matched point, interpolation, or the strength with the best F1) | Choosing the strength by F1 is a winner's-curse selection on the reported outcome. | R4 matched point, chosen by batch score only |
| X10: handling of large gaps, `bstar_low_signal` cells, `scvi_adv` | Dropping cells where the reference looks worse. | Flag and print, exclude only `bstar_low_signal` from summary sentences |
| Run order | Looking at latents before the locked analysis could steer later choices; SI-78 treats a script that uses scores as examining them. | Commit and test the scripts now (known-answer tests on synthetic data, not X1); run them only after `analysis_x1.py` has produced and saved its record |

## 7. Limits of what the numbers can say

- R0(a) uses no cell-type label (rule: labels only where the analysis is about labels), so batches that differ in cell-type
  composition look separable even with no technical batch effect. The AUC measures separability of batches in `z`, not
  technology alone.
- X10 is at matched batch removal only up to the printed gap; a difference in F1 between arms with different gaps is partly a
  difference in batch removal.
- All of it is exploratory: 5 seeds per cell, 8 tasks, no pooled test, no confirm or refute wording (SI-68, SI-103). Simulations
  (sim1, sim2) are listed separately from the six real tasks, as in `ANALYSIS_SPEC.md:100-102`.

## 8. Deviations and unchecked points

- No preflight record (`experiment-preflight`) was written: nothing is launched here. It must be completed, with the singlecell
  pack, before either script is run. Pre-answers: split unit = batch for X10 (no leakage of a batch's cells across train and
  test); the baseline is plain scVI at strength 0; R0(a) cross-validates over cells because the target is the batch label itself.
- The names of the history keys (`adv_grad_norm`, `adv_acc`, or a suffixed form) are not verified: no committed code reads them
  and I did not open any latent. The script's first step checks them on one non-X1 latent (an A1 pilot latent) or a fresh
  test fit.
- scanpy's default `n_neighbors` = 15 is from memory: the fetch of the scanpy documentation was denied. The script asserts it
  (section 3). If it is not 15, k stays 15 and the stated reason changes.
- Cost of X10 and R0 is an operation count, not a measurement.
- I did not read the R4 x1 record (it does not exist yet); the field names come from `prereg_rules.py:572-583`.
