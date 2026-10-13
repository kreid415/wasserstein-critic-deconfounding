# Guard review follow-ups, 2026-10-13
Findings of the independent review of the six guard commits (report REVIEW_GUARD_FIXES.md, artifact 9c6d8e2d;
findings G-01..G-18) addressed on branch final-review-fixes. Plain-language summary per finding:

| finding | what was wrong | change | commit |
|---|---|---|---|
| G-01 | a minibatch with no pair of batches to compare (one batch present; for reference-anchored MMD no reference cell or no other batch) now adds a zero alignment loss, but no output recorded how often | the critic-free losses count such minibatches (`stats['no_pair']`, `alignment._no_pair`); the fitter writes `no_pair_steps` into the latent's plan record only when it is above 0, so every other config stays byte-identical | 96f4258 |
| G-02 | the inner-critic non-finite test injected at the first step's loss only; a sum without the gradient penalty, or of the first step only, passed | the test runs (step, term) = (0, loss), (2, gp), (4, loss) | 96f4258 |
| G-03 | nothing tested that the fitter's `main()` calls `check_extra` with the row (the call was excluded from coverage) | an end-to-end row `none` with zstd 1 must exit 1 with "does not read ['zstd']" and write nothing; the coverage exclusion is removed and coverage now measures subprocesses (`pyproject.toml`: `[tool.coverage.run] patch = ["subprocess"]`) | 96f4258 |
| G-05 | `none`, `scvi_adv` and scANVI accepted a non-zero `lam`, which their fits never read | `lam` must stay 0 for these arms; `lam_audit.py`: all 1,908 manifest rows of these arms carry lam 0 | 4f1a570 |
| G-04 | the extra-keys test checked 3 of 5 EXTRA_ARMS keys and 6 of the unread settings | it checks every entry of both tables, and pins their contents so a deleted entry fails | 4f1a570, 39aa51e |
| G-14 | the harvest packed a latent without its training history, which the stage rule later refuses | the harvest refuses it too | 4f1a570 |

Not changed here: G-08 (only the error type for latents of two legacy writers; none of the 1,701 stored latents
lacks the keys, and leaving the scorer untouched keeps its identity proof); G-06, G-09, G-11..G-13 (notes); G-07
(Methods text), G-15..G-17 (at the new tag), G-18 (identity tooling) belong to later steps of the plan.

## Evidence (re-run commands)
| command | output | what it shows |
|---|---|---|
| `python lam_audit.py > results/lam_audit.txt` | `results/lam_audit.txt` | lam values per arm in every manifest file of the repo |
| `bash mutate.sh 39aa51e <scratch worktree> > results/mutation_results.tsv` (envs scvi-api, wcd-gpu) | `results/mutation_results.tsv` | each of the 11 planted one-line bugs in `mutants.tsv` makes its named test fail |
| `scripts/check_tests.py --base 82263e5 --src scripts cluster -- tests/scvi/... tests/jhpce/test_prod_job.py` (pytest-cov 7.1.0) | console | every changed line and branch is run by a test; no redundant new test |
