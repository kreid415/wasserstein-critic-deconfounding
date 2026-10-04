# PREFLIGHT: jhpce_envbuild_wcd_fit_home

**Verdict: GO**  (checked 2026-10-04T22:56:26Z; plan_sha 5d96eb021c2505da)

## Plan

- **question**: Infrastructure job, not an experiment: rebuild the JHPCE fit env wcd-fit in $HOME/envs (user go SI-48, placement SI-49) so the tagged filler jobs (prereg-tier12-v2) can start; it fits nothing and writes no data or metric
- **primary_outcome**: build_envs.sh fit exits 0 (conda md5 and pip versions equal cluster/envspec), env_intact.py finds 0 missing files, imports succeed through the tag's path $SCRATCH/conda/envs/wcd-fit
- **unit_of_analysis**: one software environment build; nothing is analysed
- **split_unit**: no data are split; the job reads no data
- **selection_rule**: nothing is selected; the env spec is the committed cluster/envspec/scvi-api.explicit.txt
- **n_comparisons**: 1
- **multiple_comparison_correction**: none needed: a single pass/fail check of the built env against its committed spec
- **data_filters**: the job reads no data, so no filter applies

## Items

| id | severity | area | status | evidence |
|---|---|---|---|---|
| PF-01 | blocking | Pre-declaration | pass | [auto:pf_check_plan] plan complete (sha 5d96eb021c2505da) |
| PF-02 | major | Multiple comparisons | n/a | one pass/fail check of the built env; no model, metric or dataset contrast is tested |
| PF-03 | blocking | Unit of analysis | n/a | no train/test split: the job builds a software environment and reads no data |
| PF-04 | blocking | ID integrity | n/a | no unit identifiers are used: no data are read |
| PF-05 | major | Nesting of factors | n/a | no experimental factors: a single env build |
| PF-06 | blocking | Selection bias | n/a | nothing is selected from or reported on data |
| PF-07 | blocking | Matched comparison | n/a | no arms: a single env build |
| PF-08 | blocking | Aggregation | n/a | no results are aggregated over units |
| PF-09 | blocking | No-op baselines | n/a | no data transformation, so no identity baseline |
| PF-10 | blocking | Sample size | n/a | no conditions or per-condition counts |
| PF-11 | major | Metric vs model structure | n/a | no metric is computed |
| PF-12 | blocking | Metric implementation | n/a | no metric is computed |
| PF-13 | major | Estimator noise | n/a | no estimator is computed |
| PF-14 | major | Test vs control | n/a | no test/control design |
| PF-15 | major | Attribution | n/a | the job makes no causal claim; the purge diagnosis it responds to rests on file-date evidence (notebook NB-20261004-15) |
| PF-16 | blocking | Timing confounds | pass | runtime used only to size the walltime: two independent builds agree within 2x after size scaling (Slurm 35203494, Aug 2026, scvi-api 3,810 MiB into HOME, 4 CPUs, shared: 4:55; Slurm 36131132, Oct 2, wcd-fit 5.5 GB on fastscratch: env files written 17:20:07-17:20:22 after a 17:16:57 start, ~3.5 m... |
| PF-17 | blocking | Cost estimate | pass | [auto:pf_estimate_cost] estimate 0.13 h (x2 safety = 0.26 h); cpu-hours 0.1; recommended walltime 960s |
| PF-18 | major | Co-scheduling | n/a | one process on one node, nothing co-scheduled; 32 GB requested vs 16 GB that sufficed for the August 2026 HOME build of scvi-api |
| PF-19 | blocking | Launch manifest | n/a | no manifest: the command is a fixed script pinned to one commit |
| PF-20 | major | Count reconciliation | n/a | no grid or row count |
| PF-21 | major | Replicates vary | n/a | no replicate outputs: a single environment build |
| PF-22 | blocking | Output persistence | pass | build_envs.sh runs under set -euo pipefail: conda md5 and pip versions are compared line for line with cluster/envspec; .verified and .intact_baseline.json are written only after both pass; home_room exits 3 before any write when HOME lacks room; compat_link exits 3 if its target is missing; the ... |
| PF-23 | blocking | Provenance and inclusion | n/a | no items are filtered by inclusion criteria |
| PF-24 | major | Config vs input | pass | checked against the actual target: HOME 42,408 MiB after SI-50 (du, 18:50 EDT) + 6,000 MiB build + 5,000 MiB headroom = 53,408 <= cap 95,367 MiB; shared is an allowed partition (JHPCE notes), limit 90 d; dispatch tested for fit/score/all/unknown with stubs at cfa7d5b; the tag job's HOME guard che... |
