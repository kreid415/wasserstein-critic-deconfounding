#!/usr/bin/env python
"""Ranking analysis of the penalty-strength sweep (experiment A1, eight tasks, every fitted strength, seeds 100-102) with the
baselines (docs/ANALYSIS_SPEC.md applied as docs/SWEEP_ANALYSIS.md says; the shared code is scripts/analysis_x1.py).

Differences from the X1 path, all stated in docs/SWEEP_ANALYSIS.md: the points of a penalty are every strength fitted in the sweep
(no six-strength grid, no R1 or R4 record); b0, b_common and b* are computed from the sweep rows themselves (prereg_rules.pair_targets,
the formula of R4); the matched strength of label transfer is the failure-free point closest to b* (the R4 choice); the standard error
of a reading needs all but one leave-one-seed-out repeat (2 of 3 seeds); label transfer runs on every penalty embedding and on plain
scVI and scvi_adv; baselines enter as scored (coverage in the record).

Usage: python scripts/analysis_sweep.py --inputs <dir written by sweep_inputs.py> --out-json <record.json> [--jobs N]
           [--debug-label-transfer-subset N]
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402

EXP = "A1"


def sweep_targets(out, tasks, seeds):
    """b0, M, b_common, b* and the low-signal flag of every pair from the sweep rows (every point counted), and each penalty's
    matched strength: the failure-free point whose seed-mean batch score is closest to b* (ties to the smaller strength)."""
    targets, matched = {}, {}
    for task, cond in itertools.product(tasks, (0, 1)):
        zero = A.zero_runs(out, task, cond, "C", EXP)
        P._require(sorted(zero) == list(seeds), f"{task}|{cond}: plain-scVI seeds {sorted(zero)} are not the sweep seeds {list(seeds)}")
        pts = {arm: A.arm_points(out, task, cond, arm, None, seeds, "C", EXP) for arm in P.ADV_ARMS}
        targets[f"{task}|{cond}"] = P.pair_targets(zero, pts, f"{EXP} {task} cond={cond}")
        for arm in P.ADV_ARMS:
            i, _ = P._closest(pts[arm], targets[f"{task}|{cond}"]["bstar"])
            matched[(task, cond, arm)] = pts[arm][i]["lam"]
    return targets, matched


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True, help="directory written by scripts/sweep_inputs.py")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="processes for label transfer (the result does not depend on it)")
    ap.add_argument("--debug-label-transfer-subset", type=int, default=None,
                    help="DEBUG ONLY: label transfer on the first N embeddings per task; the record is marked and is not a result")
    a = ap.parse_args(argv)
    P._require(not os.path.exists(a.out_json), f"analysis record {a.out_json} exists")
    i = a.inputs
    info = json.load(open(os.path.join(i, "inputs.json")))
    M, _ = P.read_manifest(os.path.join(i, "manifest.tsv"))
    P.check_seed_separation(M)
    score_files = [ln.strip() for ln in open(os.path.join(i, "scores_list.txt")) if ln.strip()]
    S, score_files = P.read_scores(score_files)
    F = P.read_failures(os.path.join(i, "failures.csv"))
    out = A.add_scores(P.outcomes(M, S, F, [EXP, "X1", "X13"]).sort_values("tag", ignore_index=True))   # row order (AN-11)
    sweep = out[out.experiment == EXP]
    tasks = sorted(set(sweep.task))
    P._require(tasks == sorted(info["tasks"]), f"manifest tasks {tasks} differ from inputs.json {sorted(info['tasks'])}")
    seeds = sorted(set(sweep.seed_i))
    P._require(seeds == [100, 101, 102], f"sweep seeds {seeds} are not 100-102")
    P._require(set(out.experiment[out.arm == "scvi_adv"]) <= {"X1"} and set(out.experiment) <= {EXP, "X1", "X13"},
               "unexpected experiment labels (sweep rows are A1, baselines X13 or scvi_adv rows labelled X1)")
    fail_status = dict(zip(F.tag, F.status))
    P._require("latent_sha256" in S.columns, "scores lack latent_sha256: label transfer cannot tie the embeddings to the scores")
    paths = dict(ln.rstrip("\n").split("\t") for ln in open(os.path.join(i, "latents.tsv")) if ln.strip())
    latents = dict(paths=paths, sha=dict(zip(S.tag, S.latent_sha256)))
    targets, matched = sweep_targets(out, tasks, seeds)
    common = dict(exp=EXP, seeds=seeds, targets=targets, grids=None, matched=matched, fail_status=fail_status, same_grid=False,
                  min_readings=len(seeds) - 1)
    primary = A.analyse_pairs(out, latents=latents, lt_scope="all", jobs=a.jobs, debug_subset=a.debug_label_transfer_subset, **common)
    sensitivity = {name: A.analyse_pairs(out, **common, **kw) for name, kw in A.SENSITIVITY.items()}   # label transfer uses none
    n_fits = {t: int((sweep.task == t).sum()) for t in tasks}
    rec = dict(rule="analysis_sweep", status="done", spec="docs/ANALYSIS_SPEC.md", deviations="docs/SWEEP_ANALYSIS.md",
               experiment=EXP, tasks=tasks, real_tasks=[t for t in P.REAL_TASKS if t in tasks], seeds=seeds,
               partial_tasks=len(tasks) < len(P.BPM.TASKS), min_readings=common["min_readings"],
               label_transfer_debug_subset=a.debug_label_transfer_subset,
               counts=dict(sweep_fits=n_fits, sweep_failed={t: int(((sweep.task == t) & (sweep.status == "failed")).sum()) for t in tasks},
                           baseline_rows=int((out.experiment != EXP).sum()),
                           baseline_failed=int(((out.experiment != EXP) & (out.status == "failed")).sum())),
               baselines=dict(coverage=info["baseline_coverage"], missing=info["baseline_rows_missing"],
                              complete=not info["baseline_rows_missing"], dirs=info["baseline_dirs"]),
               primary=primary, sensitivity=sensitivity)
    P.write_record(a.out_json, rec, dict(manifest=[os.path.join(i, "manifest.tsv")], scores=score_files,
                                         failures=[os.path.join(i, "failures.csv")], inputs=[os.path.join(i, "inputs.json")],
                                         sources=[s["path"] for v in info["sources"].values() for s in v],
                                         latents=primary["label_transfer"]["files"]))
    print(f"sweep analysis: {len(primary['cells'])} pairs on {len(tasks)} tasks, {len(sensitivity)} sensitivities, label transfer on "
          f"{primary['label_transfer']['n_embeddings']} embeddings"
          + (f" (DEBUG subset {a.debug_label_transfer_subset} per task)" if a.debug_label_transfer_subset else "")
          + f", {len(info['baseline_rows_missing'])} baseline rows missing -> {a.out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
