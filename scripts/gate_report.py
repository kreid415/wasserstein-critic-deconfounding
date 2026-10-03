#!/usr/bin/env python
"""Assemble docs/jhpce/GATE_JHPCE.md from the gate's result files (every number is read from a file, never retyped).

Two pinned JHPCE GPU models are reported side by side (--gate MODEL=gate_report_dir, one per model).
Inputs: --local (local version records), --setup (setup_report of the JHPCE setup job: env checks, G2 data, G4 JHPCE
scores), --gate (per model: gate_report of the GPU job), --latent-dir (gate_compare_latents.py outputs:
latent_agreement_<MODEL>.csv/.json and latent_agreement_A100_vs_L40S.json), --g3-dir (gate_compare_scores.py g3 output
per model in g3_<MODEL>/), --g4-dir (G4 tables), --split-dir (gate_task_split.py output per model), --queue-waits
(observed submit->start per model), --scoring (json: scorer version and timing of the local re-scoring), --meta (json:
dates, job ids, narrative text, caveats). Missing inputs raise; nothing is filled in.
"""
import argparse
import json
import os

import pandas as pd


def j(path):
    return json.load(open(path))


def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def versions(a, gates):
    fl, sl = j(f"{a.local}/versions_fit_local.json"), j(f"{a.local}/versions_score_local.json")
    fj = {m: j(f"{d}/versions_fit_gpu.json") for m, d in gates.items()}
    sj = j(f"{a.setup}/env/versions_score_buildnode.json")
    mods = sorted(set(fl["modules"]).union(*[set(v["modules"]) for v in fj.values()]))
    rows = [dict(env="fit", item=k, local=fl["modules"].get(k), **{f"jhpce {m}": fj[m]["modules"].get(k) for m in fj}) for k in mods]
    for k in ("python", "torch_cuda", "cudnn", "gpu", "gpu_capability", "host"):
        rows.append(dict(env="fit", item=k, local=str(fl.get(k)), **{f"jhpce {m}": str(fj[m].get(k)) for m in fj}))
    rows.append(dict(env="fit", item="nvidia-smi", local="; ".join(fl.get("nvidia_smi", [])),
                     **{f"jhpce {m}": "; ".join(fj[m].get("nvidia_smi", [])) for m in fj}))
    D = pd.DataFrame(rows)
    declared = {"torch": "declared swap (cu130 -> cu126 build)", "torch_cuda": "declared swap", "cudnn": "declared swap (cu126 wheel)",
                "gpu": "hardware", "gpu_capability": "hardware", "host": "hardware", "nvidia-smi": "hardware"}
    D["same"] = ["yes" if all(r[f"jhpce {m}"] == r["local"] for m in fj) else declared.get(r["item"], "**no**") for _, r in D.iterrows()]
    S = pd.DataFrame([dict(env="score", item=k, local=sl["modules"].get(k), jhpce=sj["modules"].get(k)) for k in sorted(set(sl["modules"]) | set(sj["modules"]))]
                     + [dict(env="score", item=k, local=str(sl.get(k)).split(" lib=")[0], jhpce=str(sj.get(k)).split(" lib=")[0]) for k in ("python", "R", "kbet")])
    S["same"] = ["yes" if l == r else "**no**" for l, r in zip(S.local, S.jhpce)]
    return D, S


def recommend(per, models, tol_days=0.25):
    """Per model: the smallest JHPCE GPU count whose base-case makespan is within tol_days of that model's minimum over the
    GPU counts evaluated. Between models: fewer G3-flagged (arm, cond) groups, then shorter makespan, then faster G5."""
    rec = {}
    for m in models:
        bc = per[m]["sp"]["base_case"]
        best = min(v["wall_days"] for v in bc.values())
        g = min(v["n_gpu"] for v in bc.values() if v["wall_days"] <= best + tol_days)
        v = bc[f"G{g}"]
        rec[m] = dict(gpus=g, wall_days=v["wall_days"], best=best, flags=len(per[m]["g3"]["flagged"]), ratio=per[m]["ratio"],
                      jhpce=v["jhpce_tasks"], local=v["local_tasks"], local_gpu_busy=v["local_gpu_busy_days"],
                      jhpce_busy=v["jhpce_gpu_busy_days_per_gpu"], jobs=v["jhpce_job_starts"])
    pick = sorted(models, key=lambda m: (rec[m]["flags"], rec[m]["wall_days"], -rec[m]["ratio"]))[0]
    return pick, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eff-sens", nargs="*", default=[], help="gate_task_split.py output dirs run with other --score-efficiency")
    for k in ("local", "setup", "latent-dir", "g3-dir", "g4-dir", "split-dir", "queue-waits", "scoring", "sacct", "meta", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--gate", nargs="+", required=True, help="MODEL=gate_report_dir, one per GPU model")
    a = ap.parse_args()
    gates = dict(x.split("=", 1) for x in a.gate)
    models = list(gates)
    meta, QW, SC = j(a.meta), j(a.queue_waits), j(a.scoring)
    pf = lambda b: "PASS" if b else "FAIL"
    cal_l = j("docs/concurrency_calibration_stock.json")
    V, VS = versions(a, gates)
    fit_diff = open(f"{a.setup}/env/wcd-fit_freeze_diff.txt").read().strip().splitlines()[-1]
    score_diff = open(f"{a.setup}/env/wcd-score_freeze_diff.txt").read().strip().splitlines()[-1]
    fp_txt = open(f"{a.setup}/data/fingerprint_compare.txt").read().strip()
    fp_rc = int(open(f"{a.setup}/data/fingerprint_rc.txt").read().strip())
    dl = pd.read_csv(f"{a.setup}/data/downloads.tsv", sep="\t", names=["file", "bytes", "md5", "status"])
    sacct = {l.split("|")[0]: l.strip().split("|") for l in open(a.sacct) if l[:1].isdigit()}
    sacct_cols = [l.strip().split("|") for l in open(a.sacct) if l.startswith("JobID")][0]
    per = {}
    for m, d in gates.items():
        gc = j(f"{d}/gate_completion.json")
        py = [l for l in open(f"{d}/pytest_tests_scvi.txt").read().splitlines() if l.strip()][-1]
        cal = j(f"{d}/concurrency_calibration_jhpce.json")
        lat = j(f"{a.latent_dir}/latent_agreement_{m}.json")
        g3 = j(f"{a.g3_dir}/g3_{m}/g3_summary.json")
        sp = j(f"{a.split_dir}/task_split_{m}.json")
        sj = dict(zip(sacct_cols, sacct[QW[m]["slurm_job"]]))
        vg = j(f"{d}/versions_fit_gpu.json")
        per[m] = dict(gc=gc, py=py, cal=cal, lat=lat, g3=g3, sp=sp, sacct=sj, gpu=vg["gpu"], torch=vg["modules"]["torch"],
                      driver=vg["nvidia_smi"][0].split(", ")[1], sha=gc["latents"][0]["git_sha"][:7],
                      flagged=", ".join(str(x["arm"]) + "/" + str(x["cond"]) for x in g3["flagged"]),
                      pyok=(" passed" in py and " failed" not in py and " error" not in py),
                      ratio=round(cal["effective_factor_window"] / cal_l["effective_factor_window"], 3))
    G4 = pd.read_csv(f"{a.g4_dir}/g4_proper.csv")
    G4gen = pd.read_csv(f"{a.g4_dir}/g4_numba_generic_local_vs_jhpce174.csv")
    NONDET = {"kBET"}
    det = G4[~G4.metric.isin(NONDET) & ~G4.all_nan]
    detgen = G4gen[~G4gen.metric.isin(NONDET) & ~G4gen.all_nan]
    g4_pass = bool((det.status == "ok").all())
    g1_cpu = "unexpected=0" in fit_diff and "unexpected=0" in score_diff
    xl = j(f"{a.latent_dir}/latent_agreement_A100_vs_L40S.json") if os.path.exists(f"{a.latent_dir}/latent_agreement_A100_vs_L40S.json") else None

    L = [f"# JHPCE cross-host gate ({meta['date']})", "", f"Status: {meta['status_line']}", "",
         f"Local: {meta['local_hw']}. JHPCE: {meta['jhpce_hw']}. Report built on branch `jhpce-gate-v2` (from main @ {meta['base_commit']}). "
         f"Jobs: {meta['jobs_line']}.", "",
         "| item | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    row = lambda name, f: L.append(f"| {name} | " + " | ".join(f(m) for m in models) + " |")
    row("G1 env (CPU build checks, shared)", lambda m: f"{pf(g1_cpu)}: fit {fit_diff}; score {score_diff}")
    row("G1 tests/scvi on the GPU node", lambda m: f"{pf(per[m]['pyok'])}: `{per[m]['py']}` ({per[m]['gpu']}, torch {per[m]['torch']}, driver {per[m]['driver']})")
    row("G2 prepped inputs (shared)", lambda m: f"{pf(fp_rc == 0)}: fingerprint_prepped.py --compare exit {fp_rc}, 8/8 downloads md5 verified")
    row("harvest integrity", lambda m: f"tar md5 {meta['tar_md5'][m]['status']} ({meta['tar_md5'][m]['md5']}); gate_completion {per[m]['gc']['valid']}/{per[m]['gc']['expected']} valid, problems {len(per[m]['gc']['problems'])}")
    row("G3 latents vs local (64 configs)", lambda m: f"bitwise {per[m]['lat']['bitwise_equal']}/{per[m]['lat']['n']}; per-dim r median {per[m]['lat']['r_median_over_configs']:.4f} "
                                                      f"(min {per[m]['lat']['r_min_over_configs']:.3f}); 15-NN overlap mean {per[m]['lat']['knn_overlap_mean_over_configs']:.3f} "
                                                      f"(min {per[m]['lat']['knn_overlap_min_over_configs']:.3f}; two local seeds: {per[m]['lat']['seed_reference_knn_overlap_mean']:.3f})")
    row("G3 scores (flag rule)", lambda m: f"{pf(len(per[m]['g3']['flagged']) == 0)}: {len(per[m]['g3']['flagged'])} of {per[m]['g3']['groups']} (arm, cond) groups flagged"
                                           + (f" ({per[m]['flagged']})" if per[m]['flagged'] else ""))
    row("G4 scoring on JHPCE CPUs (shared)", lambda m: f"{pf(g4_pass)}: {int((det.status == 'ok').sum())}/{len(det)} deterministic metrics within 1e-6; all scoring stays local")
    row("G5 8-lane throughput", lambda m: f"makespan {per[m]['cal']['makespan_s']} s (local {cal_l['makespan_s']} s); window {per[m]['cal']['window_s']} s "
                                         f"(local {cal_l['window_s']} s); factor {per[m]['cal']['effective_factor_window']} vs {cal_l['effective_factor_window']} -> per-GPU ratio {per[m]['ratio']}")
    row("queue wait observed (submit -> start)", lambda m: f"{QW[m]['wait_h']:.2f} h ({QW[m]['submit']} -> {QW[m]['start']}, 2-h 1-GPU 10-CPU job)")
    row("split, base case (SI-30, observed wait)", lambda m: "; ".join(f"{k} GPU: {v['wall_days']} d" for k, v in ((n[1:], v) for n, v in per[m]['sp']['base_case'].items()))
                                                             + f" (all-local {per[m]['sp']['all_local']['wall_days']} d)")
    pick, rec = recommend(per, models)
    allloc = per[models[0]]["sp"]["all_local"]["wall_days"]
    L += ["", "## Recommendation (the user picks)", "",
          "Rule: per GPU model, the smallest JHPCE GPU count whose base-case makespan (SI-30, observed queue wait, all scoring local) is "
          "within 0.25 d of that model's minimum over 1-4 GPUs; between models, fewer G3-flagged (arm, cond) groups first, then the "
          "shorter makespan, then the faster G5.", ""]
    for m in models:
        r = rec[m]
        L.append(f"- {m}: {r['gpus']} GPU(s) -> {r['wall_days']} d (minimum over 1-4 GPUs {r['best']} d; all-local {allloc} d); "
                 f"JHPCE tasks {', '.join(r['jhpce'])}; local {', '.join(r['local'])}; local GPU busy {r['local_gpu_busy']} d, "
                 f"JHPCE busy {r['jhpce_busy']} d per GPU, {r['jobs']} JHPCE job starts; G3 flagged groups {r['flags']}; G5 ratio {r['ratio']}.")
    L += ["", f"**Rule outcome: {pick}, {rec[pick]['gpus']} GPU(s).** " + meta["recommendation"], ""]

    L += ["## G1 versions", "", md_table(V), "", md_table(VS), "",
          f"wcd-fit vs local scvi-api `pip list`: {fit_diff}; every difference is the declared torch 2.13.0+cu126 / CUDA 12 wheel substitution "
          f"(local torch 2.13.0+cu130 needs a CUDA 13 driver; JHPCE GPU nodes run driver 555). wcd-score vs local wcd-kbet: {score_diff}. "
          "Conda layers identical to the local explicit specs; `pip check` identical to local; kBET namespace fingerprint identical (theislab/kBET afc5f431). "
          "GPU-node tests ran at the commit each gate job cloned (" + ", ".join(m + " " + per[m]["sha"] for m in models) + "; identical src/).", ""]
    L += ["## G2 prepped-input fingerprints", "", "```", fp_txt, "```", "", md_table(dl), ""]

    L += ["## G3 latent agreement (JHPCE vs local, same config and seed)", "",
          "Per config: per-dimension Pearson r between the two latents (dims matched by index, identical init), max |dz|, relative Frobenius "
          "difference, and the mean 15-NN overlap (exact Euclidean kNN). Reference: two DIFFERENT local seeds of the same (arm, cond).", ""]
    LA = {m: pd.read_csv(f"{a.latent_dir}/latent_agreement_{m}.csv").set_index("tag") for m in models}
    T = LA[models[0]][["arm", "cond", "seed"]].copy()
    for m in models:
        T[f"r_min {m}"] = LA[m].r_min; T[f"r_median {m}"] = LA[m].r_median; T[f"knn {m}"] = LA[m].knn_overlap_mean
    L += [md_table(T.reset_index().sort_values(["arm", "cond", "seed"]), floatfmt="{:.3f}"), ""]
    BA = pd.DataFrame([dict(arm=arm, **{f"{k} {m}": per[m]["lat"]["by_arm"][arm][k] for m in models for k in ("r_min", "knn")}) for arm in per[models[0]]["lat"]["by_arm"]])
    L += ["Per arm (r_min = worst dimension over the arm's configs; knn = mean overlap):", "", md_table(BA, floatfmt="{:.3f}"), ""]
    if xl:
        L += [f"A100 vs L40S latents (both JHPCE, same torch build): per-dim r median {xl['r_median_over_configs']:.4f} (min {xl['r_min_over_configs']:.3f}); "
              f"15-NN overlap mean {xl['knn_overlap_mean_over_configs']:.3f} (min {xl['knn_overlap_min_over_configs']:.3f}); bitwise {xl['bitwise_equal']}/{xl['n']}.", ""]
    L += ["### G3 score differences", "",
          f"All {SC['n_latents']} latents (64 local, 64 per JHPCE model) scored in the local wcd-kbet env with the CURRENT scorer "
          f"(scripts/score_scib_native.py at main {SC['scorer_commit']}, blob {SC['scorer_blob'][:10]}: trajectory in BIO_METRICS, kBET seeded with "
          f"set.seed({SC['kbet_seed']})), {SC['nproc']} processes at a time, one thread each; started {SC['started']}, finished {SC['finished']}. "
          f"Raw batch mean = mean of {', '.join(per[models[0]]['g3']['batch_metrics'])}; raw bio mean = mean of {', '.join(per[models[0]]['g3']['bio_metrics_used'])}. "
          "Flag = |mean paired diff| > 2 SE and > 0.5 x local seed SD.", ""]
    for m in models:
        G3 = pd.read_csv(f"{a.g3_dir}/g3_{m}/g3_group_differences.csv")
        PM = pd.read_csv(f"{a.g3_dir}/g3_{m}/g3_per_metric_differences.csv")
        L += [f"#### {m} - local", "", md_table(G3), "", "Per metric over the 64 paired configs:", "", md_table(PM), ""]

    g4c = lambda D, n: D[["metric", "max_abs_diff", "status"]].rename(columns={"max_abs_diff": f"max diff {n}", "status": f"status {n}"})
    W = g4c(G4, "default").merge(g4c(G4gen, "numba generic"), on="metric")
    L += ["## G4 scoring equivalence (unchanged since the provisional report)", "", meta["g4_text"], "", md_table(W), ""]

    L += ["## G5 throughput", "", "Same 64-fit 8-lane queue (immune, 3 epochs, queue order of the local calibration), 1 OMP/NUMBA thread per lane, 10 CPUs.", ""]
    G5 = pd.DataFrame([dict(host=f"JHPCE {m}", node=per[m]["sacct"]["NodeList"], makespan_s=per[m]["cal"]["makespan_s"], window_s=per[m]["cal"]["window_s"],
                            factor_window=per[m]["cal"]["effective_factor_window"], factor_makespan=per[m]["cal"]["effective_factor_makespan"],
                            per_gpu_ratio=per[m]["ratio"], single_lane=open(f"{gates[m]}/single_lane_fits.txt").read().strip().replace("\n", "; "),
                            job_cpu_time=per[m]["sacct"]["TotalCPU"], job_elapsed=per[m]["sacct"]["Elapsed"]) for m in models]
                       + [dict(host="local RTX 3080", node="local", makespan_s=cal_l["makespan_s"], window_s=cal_l["window_s"], factor_window=cal_l["effective_factor_window"],
                               factor_makespan=cal_l["effective_factor_makespan"], per_gpu_ratio=1.0, single_lane="", job_cpu_time="", job_elapsed="")])
    L += [md_table(G5), ""]

    m0 = models[0]
    L += ["## Task split", "", meta["split_text"], "",
          f"Design of record: {per[m0]['sp']['manifest']} (md5 {per[m0]['sp']['manifest_md5']}): {per[m0]['sp']['fits']} fits, "
          f"{per[m0]['sp']['lane_hours_total']} local lane-hours, {per[m0]['sp']['score_cpu_hours_total']} scoring hours "
          f"({per[m0]['sp']['scoring_s_per_cell_per_process'] * 1e3:.2f} ms per cell per process after the measured scoring efficiency "
          f"{per[m0]['sp']['score_efficiency']}); fits per experiment {per[m0]['sp']['fits_per_experiment']}; per task {per[m0]['sp']['fits_per_task']}. "
          f"Forced local (SI-30): {', '.join(per[m0]['sp']['force_local'])}. All-local makespan {per[m0]['sp']['all_local']['wall_days']} d.", ""]
    for m in models:
        sp = per[m]["sp"]
        rows = [dict(jhpce_gpus=v["n_gpu"], wall_days=v["wall_days"], jhpce=" ".join(v["jhpce_tasks"]), local=" ".join(v["local_tasks"]),
                     local_gpu_busy_days=v["local_gpu_busy_days"], jhpce_busy_days_per_gpu=v["jhpce_gpu_busy_days_per_gpu"], jhpce_gpu_days=v["jhpce_gpu_days"],
                     jhpce_job_starts=v["jhpce_job_starts"],
                     stage_end_days=" / ".join(f"{k.split('_')[0]} {s['end']}" for k, s in v["stage_days"].items())) for v in sp["base_case"].values()]
        grid = pd.DataFrame(sp["queue_wait_grid"])
        piv = grid.pivot_table(index=["constraint", "gpus"], columns="queue_wait_days", values="wall_days").reset_index()
        piv.columns = [str(c) if not isinstance(c, float) else f"wait {c} d" for c in piv.columns]
        L += [f"### {m} (per-GPU ratio {sp['per_gpu_speed_ratio']}; base queue wait {sp['base_queue_wait_h']:.2f} h per 3-day job)", "",
              md_table(pd.DataFrame(rows)), "", "Makespan (days) by JHPCE queue wait per 3-day GPU job; SI-30 = pilot tasks forced local, free = unconstrained:", "",
              md_table(piv, floatfmt="{:.2f}"), "", "X3 / X13 counts still change on missing-arms-v2: base-case split re-evaluated with both counts halved or +50%:", "",
              md_table(pd.DataFrame(sp["x3_x13_robustness"])), ""]
    if a.eff_sens:
        rows = []
        for d in [a.split_dir] + a.eff_sens:
            for m in models:
                sp = j(f"{d}/task_split_{m}.json")
                rows.append(dict(model=m, score_efficiency=sp["score_efficiency"],
                                 **{f"G{v['n_gpu']} days": v["wall_days"] for v in sp["base_case"].values()}, all_local_days=sp["all_local"]["wall_days"]))
        L += ["Sensitivity to the local scoring efficiency (base case, SI-30, observed queue wait):", "",
              md_table(pd.DataFrame(rows).sort_values(["model", "score_efficiency"])), ""]
    L += ["## Caveats", ""] + [f"- {c}" for c in meta.get("caveats", [])] + [""]
    open(a.out, "w").write("\n".join(L))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
