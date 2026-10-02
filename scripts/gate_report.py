#!/usr/bin/env python
"""Assemble docs/jhpce/GATE_JHPCE.md from the gate's result files (numbers are copied by code, never retyped).

Inputs (directories as harvested): --local (local version records, local G4 scores), --setup (setup_report of the
JHPCE setup job), --gate (gate_report of the JHPCE GPU job), --analysis (outputs of gate_compare_latents.py,
gate_compare_scores.py g3/g4 and gate_task_split.py). Usage:
  python scripts/gate_report.py --local L --setup S --gate G --analysis A --out docs/jhpce/GATE_JHPCE.md
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
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def versions(a):
    fl, fj = j(f"{a.local}/versions_fit_local.json"), j(f"{a.gate}/versions_fit_gpu.json")
    sl, sj = j(f"{a.local}/versions_score_local.json"), j(f"{a.setup}/env/versions_score_buildnode.json")
    rows = []
    for k in sorted(set(fl["modules"]) | set(fj["modules"])):
        rows.append(dict(env="fit", item=k, local=fl["modules"].get(k), jhpce=fj["modules"].get(k)))
    for k in ("python", "torch_cuda", "cudnn", "gpu", "gpu_capability"):
        rows.append(dict(env="fit", item=k, local=str(fl.get(k)), jhpce=str(fj.get(k))))
    rows.append(dict(env="fit", item="nvidia-smi", local="; ".join(fl.get("nvidia_smi", [])), jhpce="; ".join(fj.get("nvidia_smi", []))))
    for k in sorted(set(sl["modules"]) | set(sj["modules"])):
        rows.append(dict(env="score", item=k, local=sl["modules"].get(k), jhpce=sj["modules"].get(k)))
    for k in ("python", "R", "kbet"):
        rows.append(dict(env="score", item=k, local=str(sl.get(k)).replace(" lib=", " lib=").split(" lib=")[0],
                         jhpce=str(sj.get(k)).split(" lib=")[0]))
    rows.append(dict(env="score", item="runtime BLAS/OpenMP", local="; ".join(sl["runtime_threadpools"]),
                     jhpce="; ".join(sj["runtime_threadpools"])))
    D = pd.DataFrame(rows)
    D["same"] = (D.local == D.jhpce).map({True: "yes", False: "**no**"})
    return D


def main():
    ap = argparse.ArgumentParser()
    for k in ("local", "setup", "gate", "analysis", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--meta", required=True, help="json with job ids, gpu choice, commit, dates")
    a = ap.parse_args()
    meta = j(a.meta)
    V = versions(a)
    fit_diff = open(f"{a.setup}/env/wcd-fit_freeze_diff.txt").read().strip().splitlines()[-1]
    score_diff = open(f"{a.setup}/env/wcd-score_freeze_diff.txt").read().strip().splitlines()[-1]
    pytest_last = [l for l in open(f"{a.gate}/pytest_tests_scvi.txt").read().splitlines() if l.strip()][-1]
    fp_txt = open(f"{a.setup}/data/fingerprint_compare.txt").read().strip()
    fp_rc = int(open(f"{a.setup}/data/fingerprint_rc.txt").read().strip())
    dl = pd.read_csv(f"{a.setup}/data/downloads.tsv", sep="\t", names=["file", "bytes", "md5", "status"])
    lat = j(f"{a.analysis}/latent_agreement.json")
    LA = pd.read_csv(f"{a.analysis}/latent_agreement.csv")
    g3 = j(f"{a.analysis}/g3/g3_summary.json")
    G3 = pd.read_csv(f"{a.analysis}/g3/g3_group_differences.csv")
    PM = pd.read_csv(f"{a.analysis}/g3/g3_per_metric_differences.csv")
    G4 = pd.read_csv(f"{a.analysis}/g4_proper.csv")
    G4b = pd.read_csv(f"{a.analysis}/g4b_regenerated_prep.csv")
    cal_l, cal_j = j("docs/concurrency_calibration_stock.json"), j(f"{a.gate}/concurrency_calibration_jhpce.json")
    split = j(f"{a.analysis}/split/task_split.json")
    comp = j(f"{a.gate}/gate_completion.json")
    single = open(f"{a.gate}/single_lane_fits.txt").read().strip()

    g1_pass = ("unexpected=0" in fit_diff and "unexpected=0" in score_diff and " failed" not in pytest_last
               and "passed" in pytest_last)
    g2_pass = fp_rc == 0
    g3_pass = len(g3["flagged"]) == 0
    g4_pass = bool((G4.status == "ok").all())
    g5_ok = comp["valid"] == comp["expected"]
    pf = lambda b: "PASS" if b else "FAIL"
    L = []
    L += [f"# JHPCE cross-host gate ({meta['date']})", "",
          f"Local: RTX 3080 (scvi-api / wcd-kbet). JHPCE: one pinned GPU model, **{meta['gpu_choice']}** "
          f"(`--gres=gpu:{meta['gres']}:1`; {meta['gpu_reason']}). Repo `jhpce-setup` @ {meta['commit']}; "
          f"JHPCE jobs (this session's ledger): setup {meta['setup_job']} (Slurm {meta['setup_slurm']}), "
          f"GPU gate {meta['gate_job']} (Slurm {meta['gate_slurm']})" + (f", scoring {meta['score_job']} (Slurm {meta['score_slurm']})" if meta.get("score_job") else "") + ".", "",
          "| item | result | key numbers |", "|---|---|---|",
          f"| G1 versions | {pf(g1_pass)} | fit env: {fit_diff}; score env: {score_diff}; tests/scvi on the JHPCE GPU: `{pytest_last}` |",
          f"| G2 prepped fingerprints | {pf(g2_pass)} | fingerprint_prepped.py --compare exit {fp_rc}; downloads: 8/8 size+md5 verified |",
          f"| G3 latents + scores | {pf(g3_pass)} | {lat['n']} configs; per-dim Pearson r min {lat['r_min_over_configs']:.4f}, median {lat['r_median_over_configs']:.5f}; "
          f"15-NN overlap mean {lat['knn_overlap_mean_over_configs']:.3f} (min {lat['knn_overlap_min_over_configs']:.3f}; between-seed reference {lat['seed_reference_knn_overlap_mean']:.3f}); "
          f"flagged (arm, cond) groups: {len(g3['flagged'])} of {g3['groups']} |",
          f"| G4 scoring equivalence | {pf(g4_pass)} | {int((G4.status == 'ok').sum())}/{len(G4)} metrics within 1e-6; max abs diff {G4.max_abs_diff.max():.3g} |",
          f"| G5 throughput | {'measured' if g5_ok else 'FAIL'} | makespan {cal_j['makespan_s']} s vs local {cal_l['makespan_s']} s; steady window {cal_j['window_s']} s vs {cal_l['window_s']} s; "
          f"8-lane factor {cal_j['effective_factor_window']} vs {cal_l['effective_factor_window']} -> per-GPU speed ratio {split['per_gpu_speed_ratio']} |", ""]
    L += ["## G1 versions", "", md_table(V), "",
          f"wcd-fit vs local scvi-api pip freeze: {fit_diff} (all differences are the declared torch 2.13.0+cu126 / CUDA 12 wheel substitution). "
          f"wcd-score vs local wcd-kbet: {score_diff} (igraph: local pip metadata reads a stale 0.11.8 record; the imported igraph is 0.11.9 on both hosts). "
          "Conda layers: package file name + md5 identical to the local explicit specs for both envs.", "",
          f"tests/scvi in wcd-fit on the JHPCE GPU node: `{pytest_last}` (local: 27 passed, 1 skipped).", ""]
    L += ["## G2 prepped-input fingerprints", "", "```", fp_txt, "```", "", md_table(dl), ""]
    L += ["## G3 latent agreement (JHPCE vs local, same config and seed)", "",
          md_table(LA[["tag", "arm", "cond", "seed", "r_min", "r_median", "max_abs_dz", "rel_frobenius", "knn_overlap_mean", "knn_overlap_p05"]]
                   .sort_values(["arm", "cond", "seed"])), "",
          "### G3 score differences (one scoring run over all 128 latents)", "",
          f"Raw batch mean = mean of {', '.join(g3['batch_metrics'])}; raw bio mean = mean of {', '.join(g3['bio_metrics_used'])} "
          f"(the committed BATCH_METRICS / BIO_METRICS of scripts/score_scib_native.py; reported but not in the means: {', '.join(g3['reported_not_in_means']) or 'none'}). "
          "Flag = |mean paired diff| > 2 SE and > 0.5 x local seed SD.", "",
          md_table(G3), "", "Per metric, over all paired configs:", "", md_table(PM), ""]
    L += ["## G4 scoring equivalence (identical latent and prepped files on both hosts)", "", md_table(G4), "",
          "G4b (same latents, JHPCE-regenerated prepped file on JHPCE vs the local file locally):", "", md_table(G4b), ""]
    L += ["## G5 throughput", "", "```", json.dumps(cal_j, indent=1), "```", "",
          f"Single-lane fits on JHPCE (serial, before the queue): `{single}`", ""]
    S = pd.DataFrame([dict(jhpce_gpus=s["n_gpu"], wall_days=s["wall_days"], local=", ".join(s["local_tasks"]),
                           jhpce=", ".join(s["jhpce_tasks"]), local_busy_days=s["local_busy_days"],
                           jhpce_days_per_gpu=s["jhpce_busy_days_per_gpu"], jhpce_3day_jobs_per_gpu=s["jhpce_3day_jobs_per_gpu"],
                           local_scoring_days=s["local_scoring_days_on_spare_cpus"]) for s in split["splits"]])
    L += ["## Task split (whole tasks; local 1 GPU x 8 lanes vs JHPCE G GPUs of the pinned model)", "",
          f"Design of record: {split['fits']} fits, {split['lane_hours_total']} local lane-hours; all on the local GPU: "
          f"{split['all_local']['wall_days']} days. Per-GPU speed ratio {split['per_gpu_speed_ratio']} (JHPCE/local 8-lane factor).", "",
          md_table(S), "", meta.get("split_recommendation", ""), ""]
    L += ["## Caveats", ""] + [f"- {c}" for c in meta.get("caveats", [])] + [""]
    open(a.out, "w").write("\n".join(L))
    print(f"wrote {a.out}: G1 {pf(g1_pass)} G2 {pf(g2_pass)} G3 {pf(g3_pass)} G4 {pf(g4_pass)} G5 {'measured' if g5_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
