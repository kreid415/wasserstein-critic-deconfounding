# experiment-preflight: checklist rationale

Source of truth for item text is `pf_items()`. This file gives the reason each item exists, the failure it prevents (incident ids in `workflow_incidents.csv`), and tips beyond the helper. Severity: blocking = no launch until pass/n-a with reason; major = list in report if failed; minor = note.

Contents: A. Design and splits | B. Comparison and aggregation | C. Baselines and sample size | D. Metrics | E. Timing and cost | F. Launch and persistence | G. Provenance and config

## A. Design and splits

- **PF-01 Pre-declaration (blocking).** One primary outcome fixed before results; selection rule and data filters stated. Prevents outcome switching and code that reproduces a number hinted in the brief (140), and designs that lose the real test arm (391). The plan hash is stored; any later edit makes the verdict NO-GO, so a changed outcome must be a new, visible record.
- **PF-02 Multiple comparisons (major).** Declare the family size and correction, or label results exploratory. Cheap to state; expensive to retrofit after ranking many arms.
- **PF-03 Split unit (blocking).** When labels are constant within a unit (donor, patient, document, site), a sample-level random split lets the classifier memorise the unit (238). `pf_check_grouped_split` needs a sample-id -> unit map; build it from verified metadata.
- **PF-04 ID integrity (blocking).** A remediation control is only as good as the unit map: 36 batch levels were 14 patients, with IDs reused across sources (239). Count distinct subjects per assumed level; namespace IDs by source.
- **PF-05 Nesting (major).** If A deterministically sets B, 'overlap' between A and B is arithmetic (266) and holding out A holds out B. Use `allow=` to record intended nesting.
- **PF-06 Selection (blocking).** Choosing the best hyperparameter on a run's own curve and then reporting seed-aggregated CIs there is winner's curse (421). Select on selection seeds/units; report on fresh ones.

## B. Comparison and aggregation

- **PF-07 Matched comparison (blocking).** Arms must get equal tuning budget, selection rule, lambda policy (417). Also check that arms differ only in the factor under test: harness/scaffolding vs model capability (75), modality coverage (74), forced common width vs published defaults (402). If forcing equality changes comparability to published results, record that trade-off.
- **PF-08 Coverage-balanced aggregation (blocking).** A method covering 3 of 5 datasets is not comparable to ones covering 5 in a pooled mean (457). Pool over `common_units` or show per-unit. A non-computable value stays missing (448), never 1.0 or 0.

## C. Baselines and sample size

- **PF-09 No-op baselines (blocking).** A knob's neutral setting must be an identity on the production data (443: differed in 2,494 entries up to 0.403; 430: alpha=1 dispersion left 15% of cells passing QC). Test on real data at full size.
- **PF-10 Sample size (blocking).** Perturbation or filtering can collapse n silently (431: 52,055 -> 375 cells, a 140x drop) so metrics compare latents fitted on different n. Report n beside each metric; bound max/min n along the axis.

## D. Metrics

- **PF-11 Metric vs model structure (major).** Dimension-wise probes inflate for block-structured latents (12); continuous-score metrics misread concept-structured models (382); a mean-shift target is met by linear decoders by construction (381); positional pairing misassigned shared subspace (150). A proxy substituted for an unavailable quantity (505: journal-level for paper-level citations) needs an in-table caveat.
- **PF-12 Metric implementation (blocking).** Known-answer tests cover: perfect/worst cases, the small-n regime (146: NaN below k < n/2 bound), sign/orientation against the source library (527), array orientation per library (476). Always >= 2 distinct known answers so a constant or flipped function cannot pass.
- **PF-13 Estimator noise (major).** A subsampled numerator can be mostly noise (246); report repeat-to-repeat spread and a shuffled-label null (a score 'below null' signals a metric/model mismatch, 382).
- **PF-14 Test vs control (major).** Two of three planned parts were controls and the real test was missing (391); co-expression of top loadings is circular (392); a number matching the hint is not confirmation (140).
- **PF-15 Attribution (major).** Batch-conditioning doing its job was labelled disentanglement (299); findability in a big embedding was labelled compactness (319). Name trivial mechanisms and include a control for each.

## E. Timing and cost

- **PF-16 Timing confounds (blocking).** Idle machine, load recorded, first call separated (JIT of 34 s inside a 35.5 s timing, 174), startup cost separated from steady-state (17: 1.62x vs 2.03x), no foreign processes (16: 204-217 s vs 88-90 s) and no self-interrupts (517). A contention multiplier must not be applied to the reference condition itself (518).
- **PF-17 Cost estimate (blocking).** A pilot at a diagnostic setting under-estimated production by 25x (163; 272 vs 6,896 CPU-h); hand profiling put a 20% cost at 1% (225); a measured factor was copied to another head untested (521); a quoted ~2 h contradicted the manifest's own 0.30 h makespan (548). Use profiler measurements at production settings, a safety factor, and walltime per task (161).
- **PF-18 Co-scheduling (major).** Memory sized for one process, six lanes launched (539); lanes balanced by runtime only (536); a subset exempted by name though it had 32,484 cells (537). Measure footprint and use true counts.

## F. Launch and persistence

- **PF-19 Launch manifest (blocking).** A manifest rewrite dropped `--epochs 500`; 90 shards ran the 150-epoch default (531). CLI booleans can arrive as strings that are truthy (459). Diff every row against the intended settings dict.
- **PF-20 Count reconciliation (major).** Count by running the generator, not regex over source (450: commented-out block inflated costs 1.78x); apply protocol exclusions (519: 924 vs 822).
- **PF-21 Replicates vary (major).** Seed loops over scripts that have no seed parameter yield bit-identical replicates (409).
- **PF-22 Output persistence (blocking).** A 24 h wave never saved embeddings (525); early-stopping code killed 45 tasks after full training (167); failures recorded as rows with exit 0 got a done marker so `--resume` would skip them (522). Smoke-test the late stages and a forced failure.

## G. Provenance and config

- **PF-23 Provenance and inclusion (blocking).** A panel built from combined reference without a species filter was 29% mouse genes in a human dataset (342); an assumed deposit lacked the modality (468). Compute a category per kept item and assert it.
- **PF-24 Config vs input (major).** Top-N larger than the filtered feature count silently falls back to a different procedure (429); guards added separately can disable each other (463); categorical values with dots break framework identifiers (339); one fixed orientation assumed across libraries (476).

## Suggested applicability shortcuts

- Pure descriptive analysis without models: PF-01, 02, 04, 08, 10, 12, 23 usually apply.
- Hyperparameter sweeps: add PF-06, 07, 17, 19, 20, 21, 22.
- Timing/scaling studies: PF-16, 17, 18 and 10.
- New metric: PF-11, 12, 13, 14.
