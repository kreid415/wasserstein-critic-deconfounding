# supp-records: Supplementary Notes 10, 11 and 12 (change log)

Branch `supp-records`, from `origin/paper` at 0e79c46 (Pass 15). `paper/outline.tex` is not edited: a later pass replaces the
one-line `\item` descriptions of Supplementary Notes 10, 11 and 12 (supplement list) with `\input{supp/note10}`,
`\input{supp/note11}` and `\input{supp/note12}`. The notes contain no citations.

## Files

| item | file |
|---|---|
| Supplementary Note 10, rules fixed before scoring and deviations | `paper/supp/note10.tex` (timeline table generated) |
| Supplementary Note 11, software, hardware, run counts and checksums | `paper/supp/note11.tex` (Tables A and B generated) |
| Supplementary Note 12, direct biology check | `paper/supp/note12.tex` |
| facts behind the generated tables | `paper/supp/supp-records_facts.json` (written by `collect`) |
| generator | `paper/tools/make_supp_supp-records.py` |
| test | `paper/tools/test_supp_records.py` |
| number ledger (prose numbers) | `paper/supp-records_numbers.csv` (pass12 format, 226 rows) |

## Re-run (from the repository root, on the workstation; Python of the paper-tools environment, `paper/ENVIRONMENT_paper_tools.explicit.txt`)

```
python paper/tools/make_supp_supp-records.py collect      # reads the data root and git; writes paper/supp/supp-records_facts.json
python paper/tools/make_supp_supp-records.py              # rewrites the generated blocks of note10.tex and note11.tex
python paper/tools/make_supp_supp-records.py --check      # exit 1 if a generated block is out of date
python paper/tools/check_ledger.py paper/supp-records_numbers.csv
python scripts/check_tests.py --base origin/paper --src paper/tools -- paper/tools
```

`collect` reads, on the data root: `sweep_analysis_final_20261105/inputs/` (manifest, score list, latent list, inputs.json) and
`ENVIRONMENT_scvi-api.yml`; `tier12_v3/rules/r23/scores_list.txt` (input pilots); the replication score folders
`tier12_v3/sweep_replication/scores` and `sweep_replication/jhpce_scores/scores` with the run list
`a475ea3:manifests/sweep_replication_x1_ext1.tsv`; the `config` record of every latent file and the provenance columns of every score
file. From git: `10a1ca5:cluster/envspec/*` (environment exports), `697cc6c:docs/biology_check/biology_check.json`, and the commit
or tag date of every revision in the timeline. No random numbers are used. `check_ledger.py`'s default ledger list does not include
`paper/supp-records_numbers.csv`; the merge pass should add it.

## Corrections to the outline's one-line descriptions (what the repository shows)

1. Note 10, sweep rules: the description says "amendments of 4 and 16 October 2026, the last at commit 10a1ca5". The rules were first
   fixed on 3 October (tagged commit 129142b, before the first sweep fit), amended on 4 October (tagged c88cce3; the sweep restarted
   from zero, 202 fits unused), and the amendments decided from 13 to 16 October were fixed on 16 October (tagged 10a1ca5).
2. Note 10, pilot extension "decided on 21 October": decided on 21 and 22 October and written on 22 October (98732ff, PREREG entry of
   2026-10-22, status line).
3. Note 10, ranking rules "written on 22 and 23 October 2026, commits 065a486 and 5d01da0": the decisions are dated 22 and 23 October,
   but both commits were made on 22 October, New York time (22:25 and 22:34). The rules were never fixed by a tag (the specification
   planned one before any benchmark score was read; the benchmark was not run).
4. Note 10, "the ranking analysis run once, after all sweep and baseline fits are scored or failed": not so. Development runs on three
   tasks and on eight tasks (`all8_debug`, 1 November) came before the final run (5 November, 8595a16), and the final run was made
   without `--require-baselines` because one baseline row (Scanorama, 80 neighbours, large ATAC task) could be neither scored nor
   recorded as failed (`RUN_RECORD.md`). Written as change (x) of the ranking specification.
5. Note 10, rescope: `docs/RESCOPE_20261102.md` dates the decision 2 November; its commit 2bc5b00 is dated 1 November 20:16 New York
   time. Both stated.
6. Note 10, replication prediction: the sweep readings written as the prediction come from the sweep analysis at a361ba2 (1 November),
   not from the final run at 8595a16 (`PREDICTION.json` source block). Stated.
7. Note 11 (and the Methods replication paragraph): "each task fitted and scored on the same kind of machine as its sweep fits" holds
   for fitting only. Score files show that the sweep fits of lung, simulation 2 and pancreas were scored on the workstation (AMD Ryzen
   5 3600) and their replication fits on JHPCE (AMD EPYC 7702, fixed LISI seed); one human--mouse immune sweep fit was scored on the
   workstation, the other 377 on JHPCE. The mix is the one the scoring-site rule admits after its 5-latent identity check
   (`docs/SCORER_EQUIVALENCE.json`). Written into Note 11 with Table A. The lead may want to correct the Methods sentence
   ("made and scored on the same kind of machine, with the same scorer").
8. Note 11, "baselines 448": 448 rows in the run list, 447 scored, 0 failed, 1 setting not fitted. The scvi-tools adversary, scANVI and
   sysVI fits of lung, pancreas, simulation 2, ATAC large and human--mouse immune were fitted at c88cce3 (declared result-identical)
   on the L40S; Table A gives every commit.
9. Note 11, "input and standardisation pilots": 108 and 180 fits, 0 failed (`tier12_v3/rules/r23`).
10. Note 11, "Scaling checks (Supplementary Figure S5)": kept as one run-count line (no further fits); the checks themselves are in
    Note 6 and Figure S5, and Note 10 lists them as added after the rules.
11. Note 12: the rules were written at 639c302 (5 November 09:57), before the code (c4a793f, 10:30) and the run; 697cc6c changed only
    the summary page and its script. The outline names only c4a793f.
12. New fact, not in the descriptions: the workstation's graphics driver was updated on 2 November 2026, between the sweep and the
    replication fits (`experiments/sweep_replication/README.md`, a475ea3). Stated in Note 11.

## Decisions taken here (wording and presentation; noted per the standing rules)

- Tag names that contain the old "prereg" wording are not printed; the timeline shows the commit with "(git tag)". The tag
  `replication-prediction-v1` is printed. The path `docs/PREREG.md` is printed once, as the location of the rule document.
- The date of the family-level sign-flip analysis is not stated (no committed file records it yet); its p-value is the placeholder
  `\res{family sign-flip p}` for the merge pass.
- Limitation 7 of `docs/ANALYSIS_SPEC.md` section 17 (update steps) is not carried over: the critics' five steps with gradient penalty
  and the discriminator's one step are each method's standard setting.
- Dates are New York time from git (commit date, or tagger date of a tag); score-file dates are file modification dates.

## Checks

- `make_supp_supp-records.py --check`: exit 0.
- `check_ledger.py paper/supp-records_numbers.csv`: 226 values re-read, 0 failures.
- `scripts/check_tests.py --base origin/paper --src paper/tools -- paper/tools`: OK, 63 passed, no WARN.
- Build: a test copy of `paper/outline.tex` (not committed) with the three descriptions replaced by `\input{supp/note10}`,
  `\input{supp/note11}`, `\input{supp/note12}` (inside the closed itemize) built with tectonic through doc-build-gate
  (`dg_check_sources`, `dg_build_tex`): 50 pages, 13 of 13 figures, no failing log class; the warning list equals that of the
  unmodified `outline.tex` (2 underfull boxes, not failing). The pages that carry Notes 10 to 12 (18 to 23) were viewed.
