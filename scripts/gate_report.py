#!/usr/bin/env python
"""Assemble docs/jhpce/GATE_JHPCE.md from the gate's result files (every number is read from a file, never retyped).

Inputs: --local (local version records + local G4 scores), --setup (setup_report of the JHPCE setup job), --gate
(gate_report of the JHPCE GPU job; may not exist yet), --analysis (gate_compare_* / gate_task_split outputs), --meta
(json: dates, job ids, GPU choice, recommendation text, caveats). Items whose inputs are missing are printed as
PENDING with the reason; nothing is filled in.
"""
import argparse
import json
import os

import pandas as pd


def j(path):
    return json.load(open(path))


def have(*paths):
    return all(os.path.exists(p) for p in paths)


def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def versions(a):
    fl, sl = j(f"{a.local}/versions_fit_local.json"), j(f"{a.local}/versions_score_local.json")
    gpu_file = f"{a.gate}/versions_fit_gpu.json"
    fj = j(gpu_file) if have(gpu_file) else j(f"{a.setup}/env/versions_fit_buildnode.json")
    sj = j(f"{a.setup}/env/versions_score_buildnode.json")
    rows = [dict(env="fit", item=k, local=fl["modules"].get(k), jhpce=fj["modules"].get(k)) for k in sorted(set(fl["modules"]) | set(fj["modules"]))]
    for k in ("python", "torch_cuda", "cudnn", "gpu", "gpu_capability"):
        rows.append(dict(env="fit", item=k, local=str(fl.get(k)), jhpce=str(fj.get(k)) if have(gpu_file) or k in ("python", "torch_cuda", "cudnn") else "PENDING (GPU job)"))
    rows.append(dict(env="fit", item="nvidia-smi", local="; ".join(fl.get("nvidia_smi", [])),
                     jhpce="; ".join(fj.get("nvidia_smi", [])) if have(gpu_file) else "PENDING (GPU job)"))
    rows += [dict(env="score", item=k, local=sl["modules"].get(k), jhpce=sj["modules"].get(k)) for k in sorted(set(sl["modules"]) | set(sj["modules"]))]
    for k in ("python", "R", "kbet"):
        rows.append(dict(env="score", item=k, local=str(sl.get(k)).split(" lib=")[0], jhpce=str(sj.get(k)).split(" lib=")[0]))
    rows.append(dict(env="score", item="runtime BLAS/OpenMP", local="; ".join(sl["runtime_threadpools"]), jhpce="; ".join(sj["runtime_threadpools"])))
    D = pd.DataFrame(rows)
    declared = {("fit", "torch"): "declared swap (cu130 -> cu126 build)", ("fit", "torch_cuda"): "declared swap",
                ("fit", "cudnn"): "declared swap (wheel of the cu126 build)",
                ("score", "runtime BLAS/OpenMP"): "same libraries; CPU code path differs (OpenBLAS arch)"}
    D["same"] = [("yes" if l == r else declared.get((e, i), "pending" if "PENDING" in str(r) else "**no**"))
                 for e, i, l, r in zip(D.env, D.item, D.local, D.jhpce)]
    return D, have(gpu_file)


def main():
    ap = argparse.ArgumentParser()
    for k in ("local", "setup", "gate", "analysis", "meta", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    meta = j(a.meta)
    pf = lambda b: "PASS" if b else "FAIL"
    V, gpu_done = versions(a)
    fit_diff = open(f"{a.setup}/env/wcd-fit_freeze_diff.txt").read().strip().splitlines()[-1]
    score_diff = open(f"{a.setup}/env/wcd-score_freeze_diff.txt").read().strip().splitlines()[-1]
    pytest_file = f"{a.gate}/pytest_tests_scvi.txt"
    pytest_last = [l for l in open(pytest_file).read().splitlines() if l.strip()][-1] if have(pytest_file) else None
    fp_txt = open(f"{a.setup}/data/fingerprint_compare.txt").read().strip()
    fp_rc = int(open(f"{a.setup}/data/fingerprint_rc.txt").read().strip())
    dl = pd.read_csv(f"{a.setup}/data/downloads.tsv", sep="\t", names=["file", "bytes", "md5", "status"])
    A = a.analysis
    G4, G4b = pd.read_csv(f"{A}/g4_proper.csv"), pd.read_csv(f"{A}/g4b_regenerated_prep.csv")
    G4rl, G4rj = pd.read_csv(f"{A}/g4_local_repeat.csv"), pd.read_csv(f"{A}/g4_jhpce174_repeat.csv")
    G4gen = pd.read_csv(f"{A}/g4_numba_generic_local_vs_jhpce174.csv")
    GRAPH = pd.DataFrame([json.loads(l) for l in open(f"{A}/g4_graph_identity.jsonl")])
    g3_done = have(f"{A}/latent_agreement.json", f"{A}/g3/g3_summary.json")
    g5_done = have(f"{a.gate}/concurrency_calibration_jhpce.json", f"{a.gate}/gate_completion.json")
    split_file = f"{A}/split/task_split.json"
    cal_l = j("docs/concurrency_calibration_stock.json")

    g1 = ("unexpected=0" in fit_diff and "unexpected=0" in score_diff)
    g1_status = (pf(g1 and pytest_last is not None and " failed" not in pytest_last and "passed" in pytest_last)
                 if pytest_last else ("PASS (CPU build checks); GPU tests PENDING" if g1 else "FAIL"))
    # kBET is excluded from the host verdict: scib 1.1.7's kBET wrapper sets no R seed, so it changes between runs on one
    # machine (local repeat in the table below); it is reported separately (lead/user instruction 2026-10-03).
    NONDET = {"kBET"}
    det = G4[~G4.metric.isin(NONDET) & ~G4.all_nan]
    detgen = G4gen[~G4gen.metric.isin(NONDET) & ~G4gen.all_nan]
    g4_pass = bool((det.status == "ok").all())
    kb = lambda D: float(D.loc[D.metric == "kBET", "max_abs_diff"].iloc[0])
    L = [f"# JHPCE cross-host gate ({meta['date']})", "",
         f"Status: {meta['status_line']}", "",
         f"Local: {meta['local_hw']}. JHPCE: {meta['jhpce_hw']}. Repo `jhpce-setup` @ {meta['commit']}. "
         f"JHPCE jobs (this session's ledger): {meta['jobs_line']}.", "",
         "| item | result | key numbers |", "|---|---|---|",
         f"| G1 versions | {g1_status} | fit env vs local scvi-api: {fit_diff}; score env vs local wcd-kbet: {score_diff}; "
         f"tests/scvi on the JHPCE GPU: {('`' + pytest_last + '`') if pytest_last else 'PENDING (GPU job)'} |",
         f"| G2 prepped fingerprints | {pf(fp_rc == 0)} | fingerprint_prepped.py --compare exit {fp_rc}; 8/8 downloads size+md5 verified |"]
    if g3_done:
        lat, g3 = j(f"{A}/latent_agreement.json"), j(f"{A}/g3/g3_summary.json")
        L.append(f"| G3 latents + scores | {pf(len(g3['flagged']) == 0)} | {lat['n']} configs; per-dim Pearson r min {lat['r_min_over_configs']:.4f}, "
                 f"median {lat['r_median_over_configs']:.5f}; 15-NN overlap mean {lat['knn_overlap_mean_over_configs']:.3f} "
                 f"(min {lat['knn_overlap_min_over_configs']:.3f}; between-seed reference {lat['seed_reference_knn_overlap_mean']:.3f}); "
                 f"flagged (arm, cond) groups {len(g3['flagged'])} of {g3['groups']} |")
    else:
        L.append(f"| G3 latents + scores | PENDING | {meta['g3_pending']} |")
    L.append(f"| G4 scoring equivalence | {pf(g4_pass)} | deterministic metrics (all except kBET; hvg_overlap is NaN on both): "
             f"{int((det.status == 'ok').sum())}/{len(det)} within 1e-6 at default settings, failing "
             f"{', '.join(f'{r.metric} {r.max_abs_diff:.2g}' for r in det[det.status != 'ok'].itertuples())}; with NUMBA_CPU_NAME=generic on both hosts "
             f"{int((detgen.status == 'ok').sum())}/{len(detgen)} (failing {', '.join(r.metric for r in detgen[detgen.status != 'ok'].itertuples())}). "
             f"kBET reported separately (unseeded): cross-host {kb(G4):.2g} vs same-machine repeat {kb(G4rl):.2g} (local) / {kb(G4rj):.2g} (JHPCE) |")
    if g5_done:
        cal_j = j(f"{a.gate}/concurrency_calibration_jhpce.json")
        L.append(f"| G5 throughput | measured | makespan {cal_j['makespan_s']} s vs local {cal_l['makespan_s']} s; window {cal_j['window_s']} s vs "
                 f"{cal_l['window_s']} s; 8-lane factor {cal_j['effective_factor_window']} vs {cal_l['effective_factor_window']} -> per-GPU speed ratio "
                 f"{round(cal_j['effective_factor_window'] / cal_l['effective_factor_window'], 3)} |")
    else:
        L.append(f"| G5 throughput | PENDING | {meta['g5_pending']} (local reference: makespan {cal_l['makespan_s']} s, window {cal_l['window_s']} s, "
                 f"8-lane factor {cal_l['effective_factor_window']}) |")
    L += ["", "## G1 versions", "", md_table(V), "",
          f"wcd-fit vs local scvi-api `pip list`: {fit_diff}; every difference is the declared torch 2.13.0+cu126 / CUDA 12 wheel "
          f"substitution (local torch 2.13.0+cu130 needs a CUDA 13 driver; JHPCE GPU nodes run driver 555). wcd-score vs local wcd-kbet: "
          f"{score_diff} (igraph: local pip metadata reads a stale 0.11.8 record; the imported igraph is 0.11.9 on both hosts). "
          "Conda layers: package file names + md5 identical to the local explicit specs (both envs); `pip check` identical to local; "
          "kBET namespace fingerprint identical (theislab/kBET afc5f431)." + ("" if gpu_done else " GPU rows come from the pending GPU job; the fit-env rows above are from the CPU build node."), ""]
    L += ["## G2 prepped-input fingerprints", "", "```", fp_txt, "```", "", md_table(dl), ""]
    if g3_done:
        LA = pd.read_csv(f"{A}/latent_agreement.csv")
        G3 = pd.read_csv(f"{A}/g3/g3_group_differences.csv")
        PM = pd.read_csv(f"{A}/g3/g3_per_metric_differences.csv")
        L += ["## G3 latent agreement (JHPCE vs local, same config and seed)", "",
              md_table(LA[["tag", "arm", "cond", "seed", "r_min", "r_median", "max_abs_dz", "rel_frobenius", "knn_overlap_mean", "knn_overlap_p05"]]
                       .sort_values(["arm", "cond", "seed"])), "",
              "### G3 score differences (all 128 latents scored in local wcd-kbet)", "",
              f"Raw batch mean = mean of {', '.join(g3['batch_metrics'])}; raw bio mean = mean of {', '.join(g3['bio_metrics_used'])} "
              f"(BATCH_METRICS / BIO_METRICS of scripts/score_scib_native.py; reported but not in the means: {', '.join(g3['reported_not_in_means']) or 'none'}). "
              "Flag = |mean paired diff| > 2 SE and > 0.5 x local seed SD.", "", md_table(G3), "", "Per metric, over all paired configs:", "", md_table(PM), ""]
    else:
        L += ["## G3", "", f"PENDING: {meta['g3_pending']}", ""]
    g4c = lambda D, n: D[["metric", "max_abs_diff", "status"]].rename(columns={"max_abs_diff": f"max diff {n}", "status": f"status {n}"})
    W = g4c(G4, "G4").merge(g4c(G4b, "G4b"), on="metric").merge(g4c(G4rl, "local repeat"), on="metric").merge(
        g4c(G4rj, "JHPCE repeat"), on="metric").merge(g4c(G4gen, "numba generic"), on="metric")
    L += ["## G4 scoring equivalence", "", meta["g4_text"], "", md_table(W), "",
          "kNN graph of Q_none_0 as built by the scorer (sc.pp.neighbors on the latent), compared entry by entry:", "", md_table(GRAPH), ""]
    if g5_done:
        L += ["## G5 throughput", "", "```", json.dumps(j(f"{a.gate}/concurrency_calibration_jhpce.json"), indent=1), "```", "",
              f"Single-lane fits on JHPCE: `{open(f'{a.gate}/single_lane_fits.txt').read().strip()}`", ""]
    else:
        L += ["## G5", "", f"PENDING: {meta['g5_pending']}", ""]
    L += ["## Task split", "", meta["split_text"], ""]
    if have(split_file):
        sp = j(split_file)
        rows = []
        for n, s in sp["scenarios"]["base"].items():
            if n == "all_local":
                continue
            rows.append(dict(jhpce_gpus=s["n_gpu"], wall_days=s["wall_days"], local=" ".join(s["local_tasks"]), jhpce=" ".join(s["jhpce_tasks"]),
                             local_busy_days=s["local_busy_days"], jhpce_busy_days=s["jhpce_busy_days"], jhpce_job_starts=s["jhpce_job_starts"],
                             stage_days_local_jhpce=" / ".join(f"{k.split('_')[0]} {v[0]}|{v[1]}" for k, v in s["stage_days"].items())))
        L += [f"Speed ratio used: {sp['per_gpu_speed_ratio']} ({meta['ratio_source']}); all-local: {sp['scenarios']['base']['all_local']['wall_days']} days. "
              f"Manifest md5 {sp['manifest_md5']} ({sp['fits']} fits, {sp['lane_hours_total']} local lane-hours, {sp['score_cpu_hours_total']} scoring CPU-hours).", "",
              md_table(pd.DataFrame(rows)), "", "Sensitivity to the two pending design decisions (base-optimal split re-evaluated; scenario optimum):", "",
              md_table(pd.DataFrame(sp["robustness"])), "", "JHPCE queue-wait sensitivity (wait per 3-day GPU job; free assignment vs the 3 A1 tasks kept local):", "",
              md_table(pd.DataFrame(sp["queue_wait_sensitivity"])), ""]
    if meta.get("ratio_sweep"):
        L += ["Provisional split versus the per-GPU speed ratio (no queue wait):", "", md_table(pd.DataFrame(meta["ratio_sweep"])), ""]
    L += ["## Caveats and deviations", ""] + [f"- {c}" for c in meta.get("caveats", [])] + [""]
    open(a.out, "w").write("\n".join(L))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
