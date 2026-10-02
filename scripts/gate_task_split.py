#!/usr/bin/env python
"""Assign whole scIB tasks to the local RTX 3080 (1 GPU, 8 lanes) or JHPCE (G GPUs of ONE pinned model) so that the
STAGE-GATED Tier 1+2 makespan is minimal (CONSTRAINTS.md SI-17: every fit of a task stays on that task's machine).

Stages (docs/PREREG.md sec 0 on prereg-rules): S1 = A1 (fitted and scored) -> R1, R4-a1 -> S2 = A2 + A3 (fitted and
scored) -> R2, R3 -> S3 = X1 adversarial rows (fitted and scored) -> R4-x1 -> S4 = X3, X6, X7, X8, X12. Every gate is
global (a rule freezes once over all tasks), so makespan = sum over stages of the slower host's stage time. Fillers
(X1 arms none / scvi_adv, X13) depend on no rule: they run in a host's idle time while it waits at a gate; whatever
does not fit extends that host's S3. Rule-freeze time and JHPCE queue waits are not modelled (job starts reported).

Stage time on a host = fit time + scoring that the host's CPUs cannot overlap with fitting, at least the scoring time of
one latent of the stage's largest task (the last fit must still be scored):
  fit time = lane-hours / R;  R_local = local 8-lane factor (mixed-arm queue); R_jhpce = G x JHPCE 8-lane factor,
  both in local-lane-hours per hour (scripts/calibrate_concurrency.py with the local per-step cost model).
  scoring: CPU-h = cells x docs/scoring_time.csv rate (single thread). Local: 12 CPUs, 8 busy with fit lanes, so 4
  score while fitting and 12 when the GPU is idle; JHPCE: --jhpce-score-cpus on 'shared' nodes.
Fit cost per row: scripts/cost_model.py on the design-of-record manifest (--manifest, built on prereg-rules with
scripts/build_paper_manifest.py --backbone stock --design pilot --uncond-seeds 5).

Usage: python scripts/gate_task_split.py --manifest M.tsv --jhpce-cal CAL.json --gpus 1 2 3 4 --out-dir DIR
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM  # noqa: E402

STAGES = ["S1_A1", "S2_A23", "S3_X1adv", "S4_followup"]
FILL = "F_filler"
LOCAL_CPUS, LOCAL_LANES = 12, 8


def stage_of(r):
    if r.experiment == "A1":
        return "S1_A1"
    if r.experiment in ("A2", "A3"):
        return "S2_A23"
    if r.experiment == "X13" or (r.experiment == "X1" and r.arm in ("none", "scvi_adv")):
        return FILL
    if r.experiment == "X1":
        return "S3_X1adv"
    if r.experiment in ("X3", "X6", "X7", "X8", "X12"):
        return "S4_followup"
    raise ValueError(f"unmapped experiment {r.experiment}")


def costs(manifest, scen):
    T = pd.read_csv("docs/throughput_rtx3080_stock_backbone.csv")
    S = pd.read_csv("docs/scoring_time.csv")
    rate = float(S.seconds.sum() / S.n_cells.sum())                    # s per cell per config, one thread
    M = CM.cost(manifest, T, float("nan"), rate)
    M["stage"] = [stage_of(r) for r in M.itertuples()]
    M["cells"] = M.score_cpu_hours * 3600 / rate
    M["w"] = 1.0
    if scen.get("x6_extra"):                                             # extra X6 fits, proportional per task
        n6 = int((M.experiment == "X6").sum())
        M.loc[M.experiment == "X6", "w"] = (n6 + scen["x6_extra"]) / n6
    if scen.get("a23_uncond"):                                           # unconditioned halves of A2/A3 (same cost per row)
        M.loc[M.experiment.isin(["A2", "A3"]), "w"] = 2.0
    M["lane_w"] = M.lane_hours * M.w
    M["score_w"] = M.score_cpu_hours * M.w
    g = M.groupby(["task", "stage"]).agg(lane=("lane_w", "sum"), score=("score_w", "sum"),
                                         tail_h=("cells", lambda c: c.max() * rate / 3600), fits=("w", "sum"))
    return g, M, rate


def stage_time(g, tasks, st, R, c_fit, c_idle):
    rows = [g.loc[(t, st)] for t in tasks if (t, st) in g.index]
    F = sum(r.lane for r in rows); W = sum(r.score for r in rows)
    if F == 0 and W == 0:
        return 0.0
    tail = max(r.tail_h for r in rows)
    d_fit = F / R
    backlog = max(0.0, W - c_fit * d_fit) / c_idle
    return d_fit + max(backlog, tail)


def makespan(g, on_l, on_j, f_l, f_j, n_gpu, c_j):
    R = {"L": f_l, "J": n_gpu * f_j}
    cpu = {"L": (LOCAL_CPUS - LOCAL_LANES, LOCAL_CPUS), "J": (c_j, c_j)}
    tasks = {"L": on_l, "J": on_j}
    D = {h: {st: stage_time(g, tasks[h], st, R[h], *cpu[h]) for st in STAGES} for h in "LJ"}
    fill_need = {h: stage_time(g, tasks[h], FILL, R[h], *cpu[h]) for h in "LJ"}   # hours if run alone
    fill_done = {h: 0.0 for h in "LJ"}
    for st in STAGES:                                                     # fillers into gate waits, in stage order
        top = max(D["L"][st], D["J"][st])
        for h in "LJ":
            use = min(top - D[h][st], fill_need[h] - fill_done[h])
            fill_done[h] += max(0.0, use)
    for h in "LJ":                                                        # remainder extends the host's X1 stage
        D[h]["S3_X1adv"] += fill_need[h] - fill_done[h]
    wall_h = sum(max(D["L"][st], D["J"][st]) for st in STAGES)
    jobs = sum(math.ceil(D["J"][st] / 72.0) for st in STAGES if D["J"][st] > 0) * n_gpu
    return dict(wall_days=wall_h / 24, stage_days={st: [round(D["L"][st] / 24, 2), round(D["J"][st] / 24, 2)] for st in STAGES},
                local_busy_days=(sum(D["L"].values()) + fill_done["L"]) / 24,
                jhpce_busy_days=(sum(D["J"].values()) + fill_done["J"]) / 24,
                jhpce_gpu_days=n_gpu * (sum(D["J"].values()) + fill_done["J"]) / 24,
                jhpce_job_starts=jobs, filler_hours_in_gaps={h: round(min(fill_done[h], fill_need[h]), 1) for h in "LJ"})


def optimise(g, f_l, f_j, n_gpu, c_j):
    tasks = sorted({t for t, _ in g.index})
    best = None
    for mask in itertools.product([0, 1], repeat=len(tasks)):
        on_j = [t for t, m in zip(tasks, mask) if m]
        on_l = [t for t, m in zip(tasks, mask) if not m]
        r = makespan(g, on_l, on_j, f_l, f_j, n_gpu, c_j)
        key = (round(r["wall_days"], 4), round(r["jhpce_gpu_days"], 4))
        if best is None or key < best[0]:
            best = (key, on_l, on_j, r)
    _, on_l, on_j, r = best
    return dict(n_gpu=n_gpu, local_tasks=on_l, jhpce_tasks=on_j, **r)


def rnd(d):
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--jhpce-cal", required=True)
    ap.add_argument("--local-cal", default="docs/concurrency_calibration_stock.json")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--jhpce-score-cpus", type=int, default=48)
    ap.add_argument("--expect-fits", type=int, default=6061)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    loc, jh = json.load(open(a.local_cal)), json.load(open(a.jhpce_cal))
    f_l, f_j = float(loc["effective_factor_window"]), float(jh["effective_factor_window"])
    scenarios = {"base": {}, "x6_plus54": {"x6_extra": 54}, "a23_uncond_plus144": {"a23_uncond": True},
                 "both": {"x6_extra": 54, "a23_uncond": True}}
    out = dict(manifest=a.manifest, manifest_md5=hashlib.md5(open(a.manifest, "rb").read()).hexdigest(),
               local_factor_window=f_l, jhpce_factor_window=f_j, per_gpu_speed_ratio=round(f_j / f_l, 3),
               local_makespan_s=loc["makespan_s"], jhpce_makespan_s=jh["makespan_s"],
               makespan_ratio_local_over_jhpce=round(loc["makespan_s"] / jh["makespan_s"], 3),
               jhpce_score_cpus=a.jhpce_score_cpus, scenarios={})
    rows = []
    for name, scen in scenarios.items():
        g, M, rate = costs(a.manifest, scen)
        if name == "base":
            if len(M) != a.expect_fits:
                raise ValueError(f"manifest has {len(M)} fits, expected {a.expect_fits}")
            g.round(3).to_csv(os.path.join(a.out_dir, "task_stage_costs.csv"))
            out["fits"] = len(M); out["lane_hours_total"] = round(float(M.lane_hours.sum()), 1)
            out["score_cpu_hours_total"] = round(float(M.score_cpu_hours.sum()), 1)
            out["scoring_ms_per_cell"] = round(rate * 1e3, 2)
        all_local = makespan(g, sorted({t for t, _ in g.index}), [], f_l, f_j, 1, a.jhpce_score_cpus)
        res = {"all_local": rnd(all_local)}
        for n in a.gpus:
            best = optimise(g, f_l, f_j, n, a.jhpce_score_cpus)
            res[f"G{n}"] = rnd(best)
            rows.append(dict(scenario=name, gpus=n, wall_days=round(best["wall_days"], 2), local=" ".join(best["local_tasks"]),
                             jhpce=" ".join(best["jhpce_tasks"]), local_busy_days=round(best["local_busy_days"], 2),
                             jhpce_busy_days=round(best["jhpce_busy_days"], 2), jhpce_job_starts=best["jhpce_job_starts"]))
        out["scenarios"][name] = res
    # robustness: the base-optimal split of each G evaluated under every scenario
    rob = []
    for n in a.gpus:
        b = out["scenarios"]["base"][f"G{n}"]
        for name, scen in scenarios.items():
            g, _, _ = costs(a.manifest, scen)
            r = makespan(g, b["local_tasks"], b["jhpce_tasks"], f_l, f_j, n, a.jhpce_score_cpus)
            rob.append(dict(gpus=n, scenario=name, base_split_wall_days=round(r["wall_days"], 2),
                            scenario_optimum_wall_days=out["scenarios"][name][f"G{n}"]["wall_days"]))
    out["robustness"] = rob
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "task_split_table.csv"), index=False)
    json.dump(out, open(os.path.join(a.out_dir, "task_split.json"), "w"), indent=1)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(rob).to_string(index=False))


if __name__ == "__main__":
    main()
