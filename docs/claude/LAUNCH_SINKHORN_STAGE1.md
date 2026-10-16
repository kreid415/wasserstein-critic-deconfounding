# Launch: Sinkhorn redo stage 1 (180 A1 pilot Sinkhorn-arm refits at tag prereg-tier12-v3)

Independent launch-readiness review, 2026-10-16 (Opus rigor reviewer run from Claude Science). Preflight record:
`experiments/preflight_sinkhorn_redo_20261016/` (commit 1ad44cb, verdict GO). Terms: A1 = the penalty-strength pilot; SI-xx = numbered
standing rule in `CONSTRAINTS.md`; NB-... = lab-notebook entry; D = `/home/kendall/experiment_data/wasserstein-critic-deconfounding`.
Blinding: this document and its commands print counts, flags, names and hashes only; no score value.

## Verdict

GO-WITH-CONDITIONS for stage 1 of the Sinkhorn redo (180 A1 pilot Sinkhorn-arm refits at tag prereg-tier12-v3 = 10a1ca5). The code and the preflight record hold up: no blocking design or code defect was found, and every blocking preflight item is pass or n/a with evidence I re-checked. It cannot start yet, because two launch prerequisites are missing on disk and the launcher refuses (exit 2) until they exist: the pinned checkout D/code/wcd_prereg-tier12-v3 and the directory D/tier12_v3 holding the tags file. Conditions: (C1) create the pinned clean checkout at 10a1ca5; (C2) install the tags file (sha256 a19e4683…) and the launcher outside the checkout; (C3) start it in tmux from an ordinary terminal on the workstation, not inside a Claude Science cell (the runner died with the session twice: NB-20261007-04, NB-20261008-01), with the exit code captured; (C4) take a listing snapshot of tier12_v2/A1 before launch; (C5) the user's go. Later steps (STATUS steps 5, 7, 8, 9) are NOT ready. Step 6 (identity proof) needs no re-run.

## Findings

1. **[BLOCKER (unmet launch condition; fixed by commands C1-C2, not a design flaw)]** `/home/kendall/experiment_data/wasserstein-critic-deconfounding/code/wcd_prereg-tier12-v3; /home/kendall/experiment_data/wasserstein-critic-deconfounding/tier12_v3/manifests/sinkhorn_a1_tags.txt`
   - Issue: The pinned checkout at the tag does not exist, and neither does D/tier12_v3. The launcher checks HEAD, tag and tags-file sha256 first, so it would exit 2 at once.
   - Evidence: ls D/code shows wcd_final-review-fixes, wcd_prereg-tier12-v2 and others, but no wcd_prereg-tier12-v3; `ls D/tier12_v3` gives 'No such file or directory'; run_A1_sinkhorn_redo.sh lines 12-23. PREFLIGHT.md 'Launch conditions' 1-2 already list both as to-do.
   - Action: Run steps 1-2 of the launch sequence (clone, detach at prereg-tier12-v3, check HEAD 10a1ca5874fb… and tag object ce056088ee3b…, check the tree is clean; mkdir D/tier12_v3/manifests, copy the tags file, check sha256 a19e4683640aff79…).
2. **[MAJOR (contingent; does not block stage 1)]** `scripts/fit_outcome.py:19,70; scripts/prereg_rules.py:63; scripts/run_stage.py:98 (all at 10a1ca5)`
   - Issue: Rule SI-81 says each iteration-cap hit (b) and each zero-NMI case, NMI being the normalised mutual information batch/biology metric (a, SI-65), stops the stage until the user decides, and a fit judged failed counts as a collapse in P2. No status code exists to record that decision. Only 'diverged' and 'nonfinite_latent' are allowed (write_status refuses anything else), and the completion gate counts only scored, diverged or nonfinite_latent as final. If the user rules a cap hit 'failed', it therefore cannot be recorded, and R1 cannot run on a complete stage, unless code frozen by the tag is changed.
   - Evidence: FAILURE_STATUSES = ('diverged', 'nonfinite_latent') at fit_outcome.py:19 and prereg_rules.py:63; fit_outcome.py:70 raises for any other status. A cap hit raises RuntimeError (alignment.py:171). The guard turns only ValueError into NonFiniteLossError (scvi_adversarial_plan.py:320-327), so a cap hit leaves exit 1, which run_stage treats as an infrastructure error, retries once (--max-attempts 2, run_stage.py:1223), then stops ('No new work is started; running work finishes', run_stage.py:1001-1003). A cap hit is unlikely: the hardest-regime pilot reached at most 3,786 iterations, with the cap now at 500,000 (NB-20261013-11; SI-59).
   - Action: Starting stage 1 is safe, because a stop keeps all outputs. If a hit occurs, the user decides (SI-81). A 'failed' ruling then needs a registered status, e.g. 'not_converged' added to both FAILURE_STATUSES tuples with one test, recorded as a PREREG section 9 amendment. Re-check docs/EQUIVALENT_COMMITS.json 'valid_for_a_candidate_commit_if' at the new commit. Per SI-92, prepare this only if a hit or a zero-NMI case actually happens.
3. **[MAJOR (blocks STATUS step 5, not stage 1)]** `STATUS step 5; scripts/prereg_rules.py:306-307; scripts/freeze_matched_lambda.py:123; scripts/freeze_a1_grid.py; D/tier12_v2/{A1,A1_ext_r1_round1,A2A3}`
   - Issue: Re-running the rules (R1, R4-a1, the A3 Sinkhorn fits, R2/R3) has no plan or preflight yet, and four traps are visible now. (a) R1 refuses a tag that is scored twice, and tier12_v2/A1/scores still holds the 180 pre-fix Sinkhorn score files with the same tags. (b) Pre-fix Sinkhorn outputs made at c88cce3 also sit in tier12_v2/A1_ext_r1_round1 (18 latents, 18 scores) and tier12_v2/A2A3 (36 latents, 36 scores). The runner refuses them (SI-54), so the 18 extension fits and the 36 A3 fits need new output directories. (c) The 18 Sinkhorn extension rows in manifests/a1_extension_r1_round1.tsv were chosen by the pre-fix R1, so R1 round 1 must be re-derived. freeze_a1_grid.py has no existence check, so it would silently overwrite that tag-frozen manifest, and SI-87 covers only the 5 record files, not manifests. (d) freeze_matched_lambda.py refuses an existing --extension-manifest (manifests/a1_extension_r4_round1.tsv exists at the tag), so the R4-a1 redo needs a new file name.
   - Evidence: prereg_rules.py:306-307 ('tags scored more than once'); name counts: A1_ext_r1_round1 latents_sinkhorn=18 scores_sinkhorn=18, A2A3 latents_sinkhorn=36 scores_sinkhorn=36, so 180+18+36 = 234 = SI-51. The failures.csv of tier12_v2/A1, A1_ext_r1_round1 and A2A3 each have 0 data rows. git grep 'exists(' finds only analysis_x1.py:539 and freeze_matched_lambda.py:123. PREFLIGHT.md 'What stage 1 must produce', item 2, already flags (a) as untested.
   - Action: Before step 5: write a stage-2 plan and run a preflight. Contents: an explicit --scores file list (918 non-Sinkhorn A1 files from tier12_v2 plus the 180 from tier12_v3/A1_sinkhorn) and a merged failures file; new output directories under tier12_v3 for the extension and A3 Sinkhorn fits; redo-specific names for the extension manifests, recorded in the notebook. That is a naming and procedure choice, so per SI-92 decide and record it without asking. Then confirm by --report-only that no c88cce3 Sinkhorn output is selected.
4. **[MINOR]** `experiments/preflight_sinkhorn_redo_20261016/run_A1_sinkhorn_redo.sh:8,32-38`
   - Issue: The exit status and any refusal are lost. The launcher ends with `exec … > $LOG 2>&1`, and a detached tmux session closes when its command exits. The launcher's own refusals (exit 2/3, written to stderr before the log exists) vanish with the pane, and exit codes 4, 5 and 6 can only be inferred from log lines.
   - Evidence: Launcher usage line 8: `tmux new -d -s a1sk 'bash …'`; the checks in lines 20-27 echo to stderr; the log is created at line 29.
   - Action: Launch with an exit-code trailer and keep the pane open (step 5 of the command sequence): `…; echo "exit $? $(date -Is)" >> D/tier12_v3/logs/A1_sinkhorn_exit.txt; exec bash`.
5. **[MINOR]** `run_A1_sinkhorn_redo.sh:38 ("$@"); run_stage.py:426,449,1186`
   - Issue: Extra arguments are passed through unfiltered, so --allow-dirty or --allow-mixed-sha would switch off the provenance refusals (SI-85, SI-89) without anyone noticing.
   - Evidence: run_stage.py:426 (`if cfg.get('git_dirty') is not False and not self.allow_dirty`) and :1186 (`if not one_sha and not self.a.allow_mixed_sha`).
   - Action: Pass only --dry-run, --clear-stale-claims or --clear-foreign-claims. Optionally add a launcher guard refusing --allow-*. This is a tooling choice, so decide without asking.
6. **[MINOR]** `run_A1_sinkhorn_redo.sh:25`
   - Issue: `pgrep -f scripts/run_stage.py` matches any process whose command line contains that string, such as an editor or pager open on scripts/run_stage.py, or a shell one-liner. That gives a false 'another run_stage.py is alive' (exit 3). This fails safe.
   - Evidence: My own probe shell matched (pgrep printed my `bash -c …` line, rc 0). No runner was actually alive.
   - Action: If exit 3 appears, run `pgrep -af run_stage.py` and close the matching viewer or editor. No code change needed.
7. **[MINOR]** `PREFLIGHT.md PF-17 (cost), PF-18 (VRAM)`
   - Issue: The cost estimate is provisional: about 13 h (12-15 h; stress bound about 30 h). There is no corrected-code end-to-end timing and no measured VRAM per Sinkhorn lane, and the PF-17 tripwire is manual. I accept this: the run is local, resumable and has no walltime limit, so a misestimate costs only time. VRAM risk is low because the annealed Sinkhorn iterations run without autograd and only one final step carries the gradient (alignment.py:29-31), so memory does not grow with the iteration count. The pre-fix A1 also ran 8 lanes that included Sinkhorn rows.
   - Evidence: preflight.json answers PF-17 and PF-18; alignment.py:29-31, 56-58.
   - Action: After 24 or more scored fits, run the tripwire check in the command sequence (it prints flags only). If any task's median fit_seconds is over 2x its prediction (atac_small 1,480 s, immune 3,060 s, sim1 1,620 s), stop and re-estimate. An out-of-memory error becomes an infrastructure error, then a retry, then a stop, which is safe.
8. **[MINOR (stop cause not named)]** `run_stage.py:991-1003`
   - Issue: A cap hit or a zero-NMI scorer error appears as a generic '[STOP] fit|score of <tag> failed 2 time(s) with infrastructure errors'. The cause is only in the per-attempt log. A deterministic cap hit is also refitted once before the stop, at a cost of about 0.5-1 lane-hour.
   - Evidence: run_stage.py:1001-1003; log path D/tier12_v3/A1_sinkhorn/logs/<kind>/<tag>.a<n>.log (run_stage.py:811).
   - Action: On any [STOP], grep the attempt logs for 'Sinkhorn: L1 marginal error' and for the scib NMI error, as in the command sequence, then bring the case to the user under SI-81. Do not clear it and rerun on your own.
9. **[MINOR (resolved)]** `PREFLIGHT.md 'Not checked' 3`
   - Issue: The preflight did not re-run the Sinkhorn unit tests at the tag.
   - Evidence: `git diff --stat 8f223c8 10a1ca5 -- src tests scripts` is empty, so NB-20261016-31 (445 passed, 18 skipped at 8f223c8 on a fresh clone) applies byte-for-byte to the tag's code.
   - Action: None needed. Cite the empty diff in the launch notebook entry.
10. **[MINOR (deferred under SI-92)]** `docs/ANALYSIS_SPEC.md:1,3,22 at 10a1ca5`
   - Issue: The frozen specification calls itself 'rules; not yet frozen' and says the tag 'has not been made'. A referee reading the tagged file would see wording that contradicts it. Its content is fixed by the tag, so this is wording only.
   - Evidence: ANALYSIS_SPEC.md line 1 title; lines 3 and 22. STATUS.md section 3 already lists this as a deferred wording fix.
   - Action: Record the correction in docs/NEXT_TAG_ERRATA.md or PREREG section 9 after the launches. Do not edit the frozen file.

## Readiness of the later steps (STATUS.md steps 4-9)

| Step (STATUS.md, branch claude-code-migration) | Ready? | Still missing before launch (evidence) |
|---|---|---|
| 4. Sinkhorn redo stage 1: 180 A1 pilot Sinkhorn refits at tag v3 | GO-WITH-CONDITIONS | C1 pinned checkout D/code/wcd_prereg-tier12-v3 at 10a1ca5 (absent); C2 D/tier12_v3/manifests/sinkhorn_a1_tags.txt (absent; source file sha256 a19e4683... matches launcher); C3 tmux from a host terminal with exit-code trailer; C4 tier12_v2/A1 listing snapshot; C5 user's go. Present: prepped atac_small/immune/sim1__scib.h5ad (31.8 MB/270 MB/503 MB), envs scvi-api (torch 2.13.0+cu130, scvi 1.4.2, RTX 3080 visible) and wcd-kbet (scib 1.1.7 + rpy2 + kBET 0.99.6, R 4.5.2; the runner's own probe passes with KMP_AFFINITY=disabled), D/Rlib_kbet, 2.4 TB free vs ~1.4 GB needed (pre-fix 180 Sinkhorn: 163 MB latents + 1.2 GB models), GPU idle (124 MiB), no runner alive. |
| 5. Re-run rules R1 -> R1 extension (18 Sinkhorn) -> R4-a1 (all 6 arms; refit moved non-Sinkhorn A3 cells) -> 36 Sinkhorn A3 fits -> R2/R3; overwrite 5 records in place (SI-87) | NO-GO (not yet planned) | Stage 1 complete (gate PASS, .done). Stage-2 plan + preflight covering finding 3: explicit score-file list (no duplicate tags, prereg_rules.py:306), merged failures file, new tier12_v3 out dirs for the 18 + 36 Sinkhorn fits (pre-fix c88cce3 outputs in tier12_v2/A1_ext_r1_round1 and A2A3 are refused), redo names for extension manifests (freeze_matched_lambda.py:123 refuses existing; freeze_a1_grid.py would overwrite a frozen manifest), notebook entry listing changed non-Sinkhorn decisions (SI-87). Contingent: a recording path for cap-hit / NMI-0 'failed' rulings (finding 2). No cluster go needed (local GPU; rules on CPU). |
| 6. Identity proof (non-Sinkhorn fits/scores bit-identical to c88cce3) | READY: no re-run needed | Verified: `git diff --stat c3fd92d 10a1ca5` over src/, fit_paper_config.py, scvi_adversarial_plan.py, score_scib_native.py and their local imports (fit_outcome.py, host_info.py) is empty; run_stage.py changed (+22/-3) but builds no fit/score command (notebook/2026-10-14.md:340 per STATUS). Re-run run_c3fd92d.sh only if finding 2 forces a change to frozen code. |
| 7. Rule-free runs: X1 arms none and scvi_adv, X13 baselines, filler scoring pass A | NO-GO (launchers not retargeted) | run_X1_local.sh still pins checkout wcd_prereg-tier12-v2 and c88cce3 (lines 10, 14) and the v2-era manifest MAN_SHA 7c2fe867... (line 12); retarget to the v3 checkout (SI-54 lets the v3 runner accept the 19 existing c88cce3 non-Sinkhorn X1 scores; 0 Sinkhorn files among X1_local latents/scores by name) and a rule-free tags file; X13 needs --cpu-lanes and --cpu-python; resume with --clear-foreign-claims; own preflight. Cannot share the GPU with stage 1: the stage-1 launcher refuses if GPU memory >= 500 MiB or another run_stage is alive, so run after stage 1 (or between stages). |
| 8. X1 local part (1,125 rows; 19 scored, 1,097 need fits, 9 pending per NB-20261016-19) | NO-GO | Steps 5 complete (lambda, adversary input and standardisation from redone R1/R2/R3); rebuilt resolved X1 manifest and new MAN_SHA; run_X1_local.sh retargeted to v3 (line 14); preflight; analyses of P3/P4 and follow-ups X3/X6/X7/X8/X12/X15 plus the X7/X8 redesign must be locked while X1 runs and before R4-x1 or any X1 score is examined (SI-68, ANALYSIS_SPEC.md:65-68). |
| 9. X1 on JHPCE (5 jobs, 1,800 fits, ~8.5 GPU-days predicted) | NO-GO | x1_jhpce_command.py:23 TAG_SHA still c88cce3 (also --expected-sha check line 73); SUBMISSION_PLAN.md describes v2; harness time limits 384 h / 576 h unverified (G-15); L40S cross-host GPU gate not re-run at v3 (CR2-01, cluster/jhpce/gate_gpu.sh); step 5 complete for the resolved manifest; cluster-rules skill + preflight; the user's explicit go per job (project GO rule). |


## Command sequence (the user runs this on the workstation; a Claude Code session only verifies)

```bash
# Run these on the workstation in an ORDINARY terminal (not in a Claude Science bash cell, whose processes die with the session: NB-20261007-04, NB-20261008-01).
D=/home/kendall/experiment_data/wasserstein-critic-deconfounding
# 1. Pinned, clean checkout at the tag (public repo; a local clone of $D/code/wcd_final-review-fixes works too)
git clone https://github.com/kreid415/wasserstein-critic-deconfounding.git $D/code/wcd_prereg-tier12-v3
git -C $D/code/wcd_prereg-tier12-v3 checkout --detach prereg-tier12-v3
test "$(git -C $D/code/wcd_prereg-tier12-v3 rev-parse HEAD)" = 10a1ca5874fbcdc033b148525b5e71d866306bdb && echo HEAD_OK
test "$(git -C $D/code/wcd_prereg-tier12-v3 rev-parse prereg-tier12-v3)" = ce056088ee3b6f2ef2b825f51c90fe79641a3025 && echo TAG_OBJECT_OK
test -z "$(git -C $D/code/wcd_prereg-tier12-v3 status --porcelain)" && echo CLEAN_OK
# 2. Tags file and launcher, from the live checkout at 1ad44cb (outside the pinned checkout)
P=$D/code/wcd_final-review-fixes/experiments/preflight_sinkhorn_redo_20261016
mkdir -p $D/tier12_v3/manifests $D/tier12_v3/logs
cp $P/sinkhorn_a1_tags.txt $D/tier12_v3/manifests/
cp $P/run_A1_sinkhorn_redo.sh $D/code/run_A1_sinkhorn_redo.sh
sha256sum $D/tier12_v3/manifests/sinkhorn_a1_tags.txt      # must be a19e4683640aff7922d4124afff8f04d1a492ba83fbf9444d113ddc8f41b6ebf
test ! -e $D/tier12_v3/A1_sinkhorn && echo OUT_IS_NEW
# 3. Listing snapshot of the pre-fix record (names, sizes, mtimes only; no content is read)
find $D/tier12_v2/A1 -type f -printf '%P\t%s\t%T@\n' | sort > $D/tier12_v3/logs/tier12_v2_A1_listing_before.tsv
# 4. Dry run: preflight and plan only; starts nothing (run_stage.py:1231, :1286-1287). Expect exit 0 and a log line "180 rows {'needs_fit': 180}; to run: 180 fits"
bash $D/code/run_A1_sinkhorn_redo.sh --dry-run; echo "dry-run exit $?"
grep -E "\[stage\]|REFUSED" $(ls -t $D/tier12_v3/logs/A1_sinkhorn_runner_*.log | head -1)
# 5. Launch (after the user's go). Exit code captured; pane stays open to show refusals
tmux new -d -s a1sk "bash $D/code/run_A1_sinkhorn_redo.sh; echo \"exit \$? \$(date -Is)\" >> $D/tier12_v3/logs/A1_sinkhorn_exit.txt; exec bash"
# Pass ONLY --dry-run / --clear-stale-claims / --clear-foreign-claims to the launcher; never --allow-dirty or --allow-mixed-sha.

# --- Progress (counts and flags only) ---
L=$(ls -t $D/tier12_v3/logs/A1_sinkhorn_runner_*.log | head -1)
tmux ls; tail -n 3 "$L"
grep -c '\[done\]' "$L"                                  # rows finished so far (of 180)
grep -E '\[(STOP|INFRA)\]|REFUSED|\[gate\]' "$L" | tail -n 5
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
grep -rl --include='*.log' 'Sinkhorn: L1 marginal error' $D/tier12_v3/A1_sinkhorn/logs 2>/dev/null   # any hit = iteration cap -> STOP, user decides (SI-81 b)
grep -rlE --include='*.log' 'NMI|normalized_mutual|cluster_optimal_resolution' $D/tier12_v3/A1_sinkhorn/logs/score 2>/dev/null | head   # inspect only if a score attempt failed (SI-65, SI-81 a)
# Cost tripwire (PF-17), after >= 24 scored fits; prints flags only
$HOME/.claude-science/conda/envs/scvi-api/bin/python - <<'EOF'
import pandas as pd, glob
L=pd.read_csv(glob.glob('/home/kendall/experiment_data/wasserstein-critic-deconfounding/tier12_v3/A1_sinkhorn/ledger/A1__*__tags-a19e4683640a.csv')[0])
L=L[L.fit_seconds.notna()].sort_values('fit_ended').head(24)
pred=dict(atac_small=1480, immune=3060, sim1=1620)
print(len(L), {t: bool(g.fit_seconds.median() > 2*pred[t]) for t,g in L.groupby('task')})   # True = stop and re-estimate
EOF
# Resume after a crash or reboot: confirm no runner is alive (pgrep -af run_stage.py), then
#   tmux new -d -s a1sk "bash $D/code/run_A1_sinkhorn_redo.sh --clear-stale-claims; echo \"exit \$?\" >> $D/tier12_v3/logs/A1_sinkhorn_exit.txt; exec bash"
#   (use --clear-foreign-claims only if the runner reports foreign claims). Never clear claims while a runner is alive.

# --- Completion check for a Claude Code session (reads ledger metadata, counts, hashes and listings; never opens score values, latents' contents or failures details) ---
$HOME/.claude-science/conda/envs/scvi-api/bin/python - <<'EOF'
import json, glob, os, hashlib, collections, csv
D='/home/kendall/experiment_data/wasserstein-critic-deconfounding'; O=D+'/tier12_v3/A1_sinkhorn'; key='A1__atac_small+immune+sim1__tags-a19e4683640a'
J=json.load(open(f'{O}/ledger/{key}.json')); g=J['gate']
chk=dict(done_marker=os.path.exists(f'{O}/ledger/{key}.done'), gate_ok=g['ok'], n_final=g['n_final'], n_rows=g['n_rows'],
  counts=J['counts'], stop_reason_none=J['stop_reason'] is None, n_infra=len(J['infrastructure_errors']),
  latent_git_shas=J['latent_git_shas'], runner_git=J['runner_git'], devices=J['devices'],
  tags_sha_ok=J['tags_file']['sha256'].startswith('a19e4683640a'), manifest_sha_ok=J['manifest_sha256'].startswith('6efbe422c8f0'))
n_scores=len(glob.glob(O+'/scores/*.csv')); n_fail=sum(1 for _ in csv.reader(open(O+'/failures.csv')))-1
chk.update(n_score_files=n_scores, n_failure_rows=n_fail, scores_plus_failures_eq_180=(n_scores+n_fail==180))
# PF-21: replicate seeds must give different latents (hash bytes only)
h=collections.defaultdict(set)
for p in glob.glob(O+'/latents/*.npz'):
    t=os.path.basename(p)[:-4]; base=t.replace('_s100','_sX').replace('_s101','_sX').replace('_s102','_sX')
    h[base.rsplit('_',1)[0]].add(hashlib.sha256(open(p,'rb').read()).hexdigest())
chk['cells_with_identical_seed_latents']=sum(1 for v in h.values() if len(v)<3)
# tier12_v2/A1 untouched
import subprocess
now=subprocess.run(['bash','-c',f"find {D}/tier12_v2/A1 -type f -printf '%P\\t%s\\t%T@\\n' | sort"],capture_output=True,text=True).stdout
chk['tier12_v2_A1_unchanged']= now==open(D+'/tier12_v3/logs/tier12_v2_A1_listing_before.tsv').read()
print(json.dumps(chk, indent=1))
# PASS requires: done_marker, gate_ok, n_final==n_rows==180, stop_reason_none, latent_git_shas==['10a1ca5874fbcdc033b148525b5e71d866306bdb'],
# runner_git.git_dirty False, devices all 'NVIDIA GeForce RTX 3080', scores_plus_failures_eq_180, cells_with_identical_seed_latents==0
# (tag pattern verified on the tags file: 60 cells x 3 seeds, e.g. A1_atac_small_sinkhorn_l0.1_c1_s100_802e0dfc), tier12_v2_A1_unchanged True.
EOF
cat $D/tier12_v3/logs/A1_sinkhorn_exit.txt     # expect 'exit 0'; 4 = refused before work, 5 = stopped on infrastructure errors (check cap/NMI greps), 6 = incomplete
# Then record a notebook RESULT entry (lab-notebook nb_entry) with these counts/flags and the log path; no score values.
```

## Limits of this review

- Launcher --dry-run not executed: its prerequisites (pinned checkout D/code/wcd_prereg-tier12-v3 and D/tier12_v3 tags file) do not exist, so it would exit 2; the dry-run step is included in the user's command sequence instead.
- Sinkhorn unit tests not re-run; instead verified git diff 8f223c8..10a1ca5 over src/tests/scripts is empty, so the recorded NB-20261016-31 result applies.
- No prior-work/novelty search: not applicable to a launch-readiness review of an internal run.
- Stage-2+ readiness (STATUS steps 5-9) assessed from code, manifests, file names/counts and STATUS.md only; no stage-2 input was executed or tested.
- Rigor ledger updated as a separate entries artifact, not merged into a project ledger file (no commits allowed).
