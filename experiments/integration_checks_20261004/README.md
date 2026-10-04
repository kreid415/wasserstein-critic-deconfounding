# Integration checks, 2026-10-04 (branch integrate-fixes)

- `a1_fit_identity.sh`: A1 latents of tag prereg-tier12-v1 (129142b) vs integrate-fixes 4fde2b6, 14 rows (atac_small,
  seed 100, lambda=0 and every arm at lambda 10, both decoders, 1 epoch, CPU, 1 thread). Result
  `results/a1_fit_identity_129142b_vs_4fde2b6.csv`: 14/14 identical, max |dz| = 0. The same check at e7db443 (before
  amend-v2) also gave 14/14.
- `run_all_suites.sh`: the five test suites, one at a time; `results/summary.txt` at 4fde2b6: stage incl. real e2e 62
  passed; scvi (WCD_REQUIRE_DATA=1) 110 passed, 1 skipped; x13 + scoring 29 passed; other 311 passed; jhpce 36 passed.
  Commits after 4fde2b6 change only docs/PREREG.md and this directory.
