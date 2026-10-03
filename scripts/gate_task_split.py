#!/usr/bin/env python
"""Assign whole scIB tasks to the local RTX 3080 (1 GPU, 8 lanes) or JHPCE (G GPUs of ONE pinned model) so that the
STAGE-GATED benchmark makespan is minimal. CONSTRAINTS.md SI-17: every fit of a task stays on that task's machine;
SI-30: the pilot tasks (atac_small, immune, sim1) run locally (--force-local).

ALL SCORING RUNS ON THE LOCAL CPUs (cross-host gate G4 failed: scIB-native scores of identical latents differ between
the local CPU and JHPCE CPUs), so JHPCE latents are copied back and join the local scoring backlog of their stage.

Stages (docs/PREREG.md sec 0): S1 = A1 (fitted and scored) -> R1, R4-a1 -> S2 = A2 + A3 (fitted and scored) -> R2, R3
-> S3 = X1 adversarial rows (fitted and scored) -> R4-x1 -> S4 = X3, X6, X7, X8, X12. Every gate is global (a rule
freezes once over all tasks), so makespan = sum over stages of the stage end (all fits of both hosts done AND all
their latents scored). Rule-freeze time is not modelled.

Within a stage (fluid queue, hours from stage start):
  local fits run [0, dL], dL = local lane-hours / f_local; their latents arrive uniformly over [0, dL].
  JHPCE fits run after a queue wait q per 3-day GPU job (k = ceil(dJ / 72) jobs back to back per GPU):
  DJ = dJ + q k, dJ = JHPCE lane-hours / (G f_jhpce); their latents arrive uniformly over [q, DJ].
  Local scoring capacity: LOCAL_CPUS - LOCAL_LANES single-thread scoring processes while the local GPU fits (12 - 8 =
  4), LOCAL_CPUS (12) when it does not; each process scores at the single-thread rate (docs/scoring_time.csv) times
  --score-efficiency (throughput per process measured with many concurrent processes on this 6-core / 12-thread CPU).
  Stage end = max(dL, DJ, fluid scoring finish, last arrival + scoring time of one latent of that host's largest task).
Fillers (X1 arms none / scvi_adv, X13; no rule depends on them) are fitted in GPU idle time at gates: on the local GPU
only as far as the spare local CPU capacity in the gap allows (each filler-fitting hour takes LOCAL_LANES CPUs from
scoring); on JHPCE in the gap after the stage's own JHPCE fits (a stage without JHPCE work needs a filler job, which
waits q first). Filler latents are scored with the spare local CPU capacity of the stage they are fitted in; filler
fits that fit in no gap extend that host's S3, and filler scoring left at the end extends the makespan.
Fit cost per row: scripts/cost_model.py (per-step throughput, +5 s setup) on the design-of-record manifest.
f_local, f_jhpce: effective 8-lane factors over the steady window, in local-single-lane-hours per hour
(scripts/calibrate_concurrency.py with the local per-step cost model; gate job concurrency_calibration_jhpce.json).

Usage: python scripts/gate_task_split.py --manifest M.tsv --jhpce-cal CAL.json --label L40S --base-queue-wait-h 7.8
           --force-local atac_small immune sim1 --score-efficiency E --gpus 1 2 3 4 --out-dir DIR
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
JOB_H = 72.0                     # one JHPCE GPU job = 3 days of fitting


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


_COST_CACHE = {}


def costs(manifest, scen, eff):
    key = (manifest, json.dumps(scen, sort_keys=True), eff)
    if key not in _COST_CACHE:
        _COST_CACHE[key] = _costs(manifest, scen, eff)
    return _COST_CACHE[key]


def _costs(manifest, scen, eff):
    T = pd.read_csv("docs/throughput_rtx3080_stock_backbone.csv")
    S = pd.read_csv("docs/scoring_time.csv")
    rate = float(S.seconds.sum() / S.n_cells.sum()) / eff             # s per cell per latent, one scoring process
    M = CM.cost(manifest, T, float("nan"), rate)
    M["stage"] = [stage_of(r) for r in M.itertuples()]
    M["cells"] = M.score_cpu_hours * 3600 / rate
    M["w"] = 1.0
    for ex, f in scen.get("scale", {}).items():                        # count scenarios (e.g. X3/X13 still changing)
        if not (M.experiment == ex).any():
            raise ValueError(f"scenario scales {ex}, which the manifest does not have")
        M.loc[M.experiment == ex, "w"] = float(f)
    M["lane_w"] = M.lane_hours * M.w
    M["score_w"] = M.score_cpu_hours * M.w
    g = M.groupby(["task", "stage"]).agg(lane=("lane_w", "sum"), score=("score_w", "sum"),
                                         tail_h=("cells", lambda c: c.max() * rate / 3600), fits=("w", "sum"))
    return g, M, rate


def fluid(streams, t_fit_end, c_fit, c_idle):
    """Single fluid scoring queue. streams: (t0, t1, work CPU-h) arriving uniformly on [t0, t1]. Capacity c_fit before
    t_fit_end (local GPU fitting), c_idle after. Returns (finish time of all work, (arrived, backlog) at t_fit_end)."""
    for t0, t1, w in streams:
        if w > 0 and not t1 > t0:
            raise ValueError(f"work {w} with an empty arrival window [{t0}, {t1}]")
    pts = sorted({0.0, t_fit_end} | {t for s in streams for t in (s[0], s[1])})
    A = B = 0.0
    snap = (0.0, 0.0)
    for a_t, b_t in zip(pts[:-1], pts[1:]):
        if a_t == t_fit_end:
            snap = (A, B)
        rate = sum(w / (t1 - t0) for t0, t1, w in streams if w > 0 and t0 <= a_t < t1)
        c = c_fit if a_t < t_fit_end else c_idle
        dt = b_t - a_t
        A += rate * dt
        B = B + (rate - c) * dt if rate >= c else max(0.0, B - (c - rate) * dt)
    if pts[-1] == t_fit_end:
        snap = (A, B)
    return pts[-1] + B / c_idle, snap


def host_work(g, tasks, st):
    rows = [g.loc[(t, st)] for t in tasks if (t, st) in g.index]
    return (sum(r.lane for r in rows), sum(r.score for r in rows), max([r.tail_h for r in rows], default=0.0))


def stage(FL, WL, tL, FJ, WJ, tJ, P, n_gpu, q):
    """One stage: fit lane-hours F, scoring CPU-h W, one-latent scoring tail t per host. Returns a dict of times (h)."""
    c_fit, c_idle = P["c_fit"], P["c_idle"]
    dL = FL / P["f_l"]
    dJ = FJ / (n_gpu * P["f_j"]) if FJ > 0 else 0.0
    k = math.ceil(dJ / JOB_H) if dJ > 0 else 0
    DJ = dJ + q * k
    streams = [s for s in ((0.0, dL, WL), (q if dJ > 0 else 0.0, DJ, WJ)) if s[2] > 0]
    if not streams and dL == 0 and DJ == 0:
        return dict(end=0.0, dL=0.0, dJ=0.0, DJ=0.0, jobs=0, spare_total=0.0, spare_gap=0.0, W=0.0)
    fin, (a_fe, b_fe) = fluid(streams, dL, c_fit, c_idle)
    tail = (tJ if WJ > 0 else 0.0) if DJ >= dL else (tL if WL > 0 else 0.0)
    end = max(dL, DJ, fin, max(dL, DJ) + tail)
    W = WL + WJ
    spare_total = c_fit * dL + c_idle * (end - dL) - W
    spare_gap = c_idle * (end - dL) - (W - (a_fe - b_fe))
    return dict(end=end, dL=dL, dJ=dJ, DJ=DJ, jobs=k * n_gpu, spare_total=max(0.0, spare_total),
                spare_gap=max(0.0, spare_gap), W=W)


def makespan(g, on_l, on_j, P, n_gpu, q=0.0):
    c_fit, c_idle = P["c_fit"], P["c_idle"]
    S = {}
    for st in STAGES:
        S[st] = stage(*host_work(g, on_l, st), *host_work(g, on_j, st), P, n_gpu, q)
    fL, wfL, _ = host_work(g, on_l, FILL)
    fJ, wfJ, _ = host_work(g, on_j, FILL)
    need = {"L": fL / P["f_l"], "J": fJ / (n_gpu * P["f_j"]) if (on_j and fJ > 0) else 0.0}
    wfill = {"L": wfL, "J": wfJ}
    done = {"L": 0.0, "J": 0.0}
    backlog, extra_jobs = 0.0, 0
    for st in STAGES:
        s = S[st]
        if s["end"] == 0:
            continue
        xL = min(s["end"] - s["dL"], s["spare_gap"] / (c_idle - c_fit), need["L"] - done["L"])
        xJ = 0.0
        if on_j and need["J"] > done["J"]:
            gapJ = s["end"] - s["DJ"] - (q if s["DJ"] == 0 else 0.0)
            xJ = min(max(0.0, gapJ), need["J"] - done["J"])
            if s["DJ"] == 0 and xJ > 0:
                extra_jobs += n_gpu * math.ceil(xJ / JOB_H)
        xL = max(0.0, xL)
        for h, x in (("L", xL), ("J", xJ)):
            if x > 0:
                backlog += wfill[h] * x / need[h]
                done[h] += x
        spare = s["spare_total"] - (c_idle - c_fit) * xL
        backlog -= min(backlog, max(0.0, spare))
        s["fill_L"], s["fill_J"] = xL, xJ
    rem = {h: need[h] - done[h] for h in "LJ"}
    if rem["L"] > 1e-9 or rem["J"] > 1e-9:                             # leftover filler fits extend S3
        FL, WL, tL = host_work(g, on_l, "S3_X1adv")
        FJ, WJ, tJ = host_work(g, on_j, "S3_X1adv")
        aL = rem["L"] / need["L"] if need["L"] else 0.0
        aJ = rem["J"] / need["J"] if need["J"] else 0.0
        s3 = stage(FL + aL * fL, WL + aL * wfL, tL, FJ + aJ * fJ, WJ + aJ * wfJ, tJ, P, n_gpu, q)
        s3["fill_L"], s3["fill_J"] = S["S3_X1adv"].get("fill_L", 0.0), S["S3_X1adv"].get("fill_J", 0.0)
        S["S3_X1adv"] = s3
    wall_h = sum(S[st]["end"] for st in STAGES) + backlog / c_idle
    jobs = sum(S[st]["jobs"] for st in STAGES) + extra_jobs
    loc_gpu_h = sum(S[st]["dL"] for st in STAGES) + done["L"] + (rem["L"] if rem["L"] > 1e-9 else 0.0)
    jh_gpu_h = sum(S[st]["dJ"] for st in STAGES) + done["J"] + (rem["J"] if rem["J"] > 1e-9 else 0.0)
    jh_span_h = sum(S[st]["DJ"] for st in STAGES)
    return dict(wall_days=wall_h / 24,
                stage_days={st: dict(end=round(S[st]["end"] / 24, 2), local_fit=round(S[st]["dL"] / 24, 2),
                                     jhpce_fit=round(S[st]["dJ"] / 24, 2), jhpce_incl_wait=round(S[st]["DJ"] / 24, 2))
                            for st in STAGES},
                local_gpu_busy_days=loc_gpu_h / 24, jhpce_gpu_busy_days_per_gpu=jh_gpu_h / 24,
                jhpce_gpu_days=n_gpu * jh_gpu_h / 24, jhpce_span_days_incl_waits=jh_span_h / 24,
                local_scoring_cpu_days=(sum(S[st]["W"] for st in STAGES) + wfL + wfJ) / 24,
                jhpce_job_starts=jobs, filler_scoring_left_h=round(backlog / c_idle, 2),
                filler_fit_hours_in_gaps={h: round(done[h], 1) for h in "LJ"})


def optimise(g, P, n_gpu, q=0.0, force_local=()):
    tasks = sorted({t for t, _ in g.index})
    missing = set(force_local) - set(tasks)
    if missing:
        raise ValueError(f"--force-local tasks not in the manifest: {sorted(missing)}")
    free = [t for t in tasks if t not in force_local]
    best = None
    for mask in itertools.product([0, 1], repeat=len(free)):
        on_j = [t for t, m in zip(free, mask) if m]
        on_l = sorted(set(tasks) - set(on_j))
        r = makespan(g, on_l, on_j, P, n_gpu, q)
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
    ap.add_argument("--jhpce-cal", required=True, help="gate job concurrency_calibration_jhpce.json (one GPU model)")
    ap.add_argument("--label", required=True, help="GPU model label, e.g. A100 or L40S")
    ap.add_argument("--local-cal", default="docs/concurrency_calibration_stock.json")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--max-gpus", type=int, default=5, help="shared JHPCE QOS cap (gres/gpu)")
    ap.add_argument("--expect-fits", type=int, required=True)
    ap.add_argument("--force-local", nargs="*", default=[], help="tasks that must run locally (SI-30)")
    ap.add_argument("--base-queue-wait-h", type=float, required=True, help="observed submit->start wait (base case)")
    ap.add_argument("--queue-wait-days", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0],
                    help="sensitivity grid: JHPCE queue wait per 3-day GPU job")
    ap.add_argument("--score-efficiency", type=float, default=1.0,
                    help="scoring throughput per process under full local load / single-thread reference (<= 1)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if max(a.gpus) > a.max_gpus:
        raise ValueError(f"--gpus {a.gpus} exceeds the {a.max_gpus}-GPU cap")
    if not 0 < a.score_efficiency <= 1.5:
        raise ValueError(f"implausible --score-efficiency {a.score_efficiency}")
    os.makedirs(a.out_dir, exist_ok=True)
    loc, jh = json.load(open(a.local_cal)), json.load(open(a.jhpce_cal))
    P = dict(f_l=float(loc["effective_factor_window"]), f_j=float(jh["effective_factor_window"]),
             c_fit=LOCAL_CPUS - LOCAL_LANES, c_idle=LOCAL_CPUS)
    fl = tuple(a.force_local)
    base_q = a.base_queue_wait_h
    g, M, rate = costs(a.manifest, {}, a.score_efficiency)
    if len(M) != a.expect_fits:
        raise ValueError(f"manifest has {len(M)} fits, expected {a.expect_fits}")
    g.round(3).to_csv(os.path.join(a.out_dir, f"task_stage_costs_{a.label}.csv"))
    out = dict(label=a.label, manifest=os.path.basename(a.manifest),
               manifest_md5=hashlib.md5(open(a.manifest, "rb").read()).hexdigest(),
               fits=len(M), lane_hours_total=round(float(M.lane_hours.sum()), 1),
               score_cpu_hours_total=round(float(M.score_cpu_hours.sum()), 1),
               scoring_s_per_cell_per_process=round(rate, 5), score_efficiency=a.score_efficiency,
               local_factor_window=P["f_l"], jhpce_factor_window=P["f_j"], per_gpu_speed_ratio=round(P["f_j"] / P["f_l"], 3),
               local_makespan_s=loc["makespan_s"], jhpce_makespan_s=jh["makespan_s"],
               local_cpus=LOCAL_CPUS, local_lanes=LOCAL_LANES, force_local=list(fl), base_queue_wait_h=base_q,
               fits_per_experiment=M.groupby("experiment").size().to_dict(), fits_per_task=M.groupby("task").size().to_dict())
    out["all_local"] = rnd(makespan(g, sorted({t for t, _ in g.index}), [], P, 1, 0.0))
    rows, base = [], {}
    for n in a.gpus:                                                    # base case: SI-30 + observed queue wait
        best = optimise(g, P, n, base_q, fl)
        base[f"G{n}"] = rnd(best)
    out["base_case"] = base
    for q_d in sorted(set([base_q / 24] + list(a.queue_wait_days))):   # sensitivity grid (+ unconstrained comparison)
        for n in a.gpus:
            for label, force in (("SI-30", fl), ("free", ())):
                r = optimise(g, P, n, 24 * q_d, force)
                rows.append(dict(label=a.label, queue_wait_days=round(q_d, 3), is_base_wait=abs(24 * q_d - base_q) < 1e-9,
                                 gpus=n, constraint=label, wall_days=round(r["wall_days"], 2),
                                 jhpce=" ".join(r["jhpce_tasks"]), local=" ".join(r["local_tasks"]),
                                 local_gpu_busy_days=round(r["local_gpu_busy_days"], 2),
                                 jhpce_gpu_busy_days_per_gpu=round(r["jhpce_gpu_busy_days_per_gpu"], 2),
                                 jhpce_gpu_days=round(r["jhpce_gpu_days"], 2), jhpce_job_starts=r["jhpce_job_starts"]))
    out["queue_wait_grid"] = rows
    scen = {"x3_x13_half": {"scale": {"X3": 0.5, "X13": 0.5}}, "x3_x13_plus50pct": {"scale": {"X3": 1.5, "X13": 1.5}}}
    rob = []
    for n in a.gpus:                                                    # base split under X3/X13 count changes
        b = base[f"G{n}"]
        for name, sc in scen.items():
            gs, Ms, _ = costs(a.manifest, sc, a.score_efficiency)
            r = makespan(gs, b["local_tasks"], b["jhpce_tasks"], P, n, base_q)
            o = optimise(gs, P, n, base_q, fl)
            rob.append(dict(gpus=n, scenario=name, fits=round(float(Ms.w.sum())), base_split_wall_days=round(r["wall_days"], 2),
                            scenario_optimum_wall_days=round(o["wall_days"], 2), scenario_optimum_jhpce=" ".join(o["jhpce_tasks"])))
    out["x3_x13_robustness"] = rob
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, f"task_split_grid_{a.label}.csv"), index=False)
    json.dump(out, open(os.path.join(a.out_dir, f"task_split_{a.label}.json"), "w"), indent=1)
    show = pd.DataFrame([dict(gpus=n, wall_days=v["wall_days"], jhpce=" ".join(v["jhpce_tasks"]),
                              local_gpu_busy=v["local_gpu_busy_days"], jhpce_busy_per_gpu=v["jhpce_gpu_busy_days_per_gpu"],
                              jobs=v["jhpce_job_starts"]) for n, v in ((int(k[1:]), v) for k, v in base.items())])
    print(f"{a.label}: ratio {out['per_gpu_speed_ratio']}, base wait {base_q:.2f} h, all-local {out['all_local']['wall_days']} d")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
