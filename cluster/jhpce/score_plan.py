#!/usr/bin/env python
"""Plan of a JHPCE score-only job: one Slurm array per task (dataset) on the `shared` partition, one latent per array
task (cluster/jhpce/score_array.sh). Writes OUT_DIR/list_<task>.tsv and OUT_DIR/score_<task>.sbatch, prints the sizes and
the time estimate. Nothing is submitted: each submission needs the user's explicit go (CONSTRAINTS.md SI-29).

Run it on JHPCE, in the clone of the commit the tasks will score at (it hashes every latent; the sbatch files run the
scripts of that clone, so keep it on fastscratch):
  python cluster/jhpce/score_plan.py --manifest M.tsv --tags TAGS --latents-dir D --prepped-dir P --out-dir O [--run run1]
  python cluster/jhpce/score_plan.py --assume immune_hum_mou=378 atac_large=378 ...        # time estimate only
TAGS: one tag per line (first column; '#' lines and a 'tag' header are skipped); an optional second column holding a
SHA-256 is checked against D/<tag>.npz. D/<tag>.npz = the latent, M = the manifest the latents were fitted from (the
score row's experiment, task, arm, lam, cond, seed come from it, as in scripts/run_stage.py), P/<task>__<counts>.h5ad =
the prepped files. Scores land in O/<run>/scores/<tag>.csv.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# task -> (RAM per task in GB, walltime). RAM: measured peak RSS of the scorer at the production setting, atac_large
# 24.79 GB and immune_hum_mou 19.36 GB (experiments/fillers_x1_x13_local_scoring/PREFLIGHT.md PF-18), x1.3 margin; the
# other tasks scale with cells and are NOT measured (x1.7 margin on the scaled value): check sacct MaxRSS of the first
# tasks. Walltime >= 3x the loaded estimate of estimate() (the wall is hard).
SIZES = {"immune_hum_mou": (32, "03:00:00"), "atac_large": (32, "03:00:00"), "lung": (16, "01:00:00"),
         "immune": (16, "01:00:00"), "sim2": (8, "00:40:00"), "pancreas": (8, "00:40:00"), "sim1": (8, "00:30:00"),
         "atac_small": (8, "00:30:00")}
LOADED_S_BIG = 3000.0   # user, 2026-10-21: 2,500-3,000 s per immune_hum_mou / atac_large latent with 3-4 scorers at once locally


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def estimate(counts, cpus):
    """Rows per task: latents, cells, RAM, walltime, array throttle (CPUs shared by CPU-seconds, so the arrays end
    together) and seconds per latent at the idle rate (docs/scoring_time.csv: immune_hum_mou, machine otherwise idle)
    and at the loaded rate (LOADED_S_BIG for immune_hum_mou's cells); both scale with cells."""
    with open(os.path.join(REPO, "docs", "scoring_time.csv")) as f:
        r = next(x for x in csv.DictReader(f) if x["tag"] == "BENCH_immune_hum_mou_pooled")
    idle, loaded = float(r["seconds"]) / int(r["n_cells"]), LOADED_S_BIG / int(r["n_cells"])
    with open(os.path.join(REPO, "configs", "dataset_registry.json")) as f:
        reg = json.load(f)
    rows = [dict(task=t, latents=n, cells=reg[t]["n_obs"], mem_gb=SIZES[t][0], walltime=SIZES[t][1],
                 idle_s=reg[t]["n_obs"] * idle, loaded_s=reg[t]["n_obs"] * loaded) for t, n in counts.items()]
    work = sum(x["latents"] * x["cells"] for x in rows)
    for x in rows:
        x["throttle"] = max(1, min(x["latents"], cpus * x["latents"] * x["cells"] // work))
    return rows


def report(rows, cpus):
    print(f"{'task':15s}{'latents':>8s}{'cells':>8s}{'RAM GB':>7s}{'wall':>9s}{'array':>7s}{'s/latent idle':>14s}{'loaded':>8s}")
    for x in rows:
        print(f"{x['task']:15s}{x['latents']:8d}{x['cells']:8d}{x['mem_gb']:7d}{x['walltime']:>9s}{x['throttle']:7d}"
              f"{x['idle_s']:14.0f}{x['loaded_s']:8.0f}")
    for k in ("idle", "loaded"):
        cpu_s = sum(x["latents"] * x[k + "_s"] for x in rows)
        wall = max(cpu_s / cpus, max(x[k + "_s"] for x in rows))
        print(f"{k:6s}: {sum(x['latents'] for x in rows)} latents, {cpu_s / 3600:.0f} CPU-hours, compute wall at "
              f"{cpus} CPUs {wall / 3600:.2f} h (queue wait not included)")
    avg = sum(x["latents"] * x["loaded_s"] * x["mem_gb"] for x in rows) / sum(x["latents"] * x["loaded_s"] for x in rows)
    print(f"RAM: {avg:.1f} GB per running task on average (CPU-time weighted); every array at its throttle holds "
          f"{sum(x['throttle'] * x['mem_gb'] for x in rows)} GB at once. With only F GB free on the nodes the wall is "
          f"CPU-hours / min({cpus}, F / {avg:.1f})")


def sbatch(x, n, out, run, sha, lst):
    return "\n".join([
        "#!/bin/bash", f"#SBATCH --job-name=wcd-score-{x['task']}", "#SBATCH --partition=shared", "#SBATCH --account=jhpce",
        f"#SBATCH --array=1-{n}%{x['throttle']}", "#SBATCH --cpus-per-task=1", f"#SBATCH --mem={x['mem_gb']}G",
        f"#SBATCH --time={x['walltime']}", "#SBATCH --no-requeue", f"#SBATCH --output={out}/logs/slurm-%x_%A_%a.out",
        "set -euo pipefail",
        f'exec bash {REPO}/cluster/jhpce/score_array.sh {sha} {lst} {out}/{run}', ""])


def read_tags(path):
    out = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if not f[0] or f[0].startswith("#") or f[0] == "tag":
            continue
        if f[0] in out:
            raise SystemExit(f"{path}: tag {f[0]} twice")
        out[f[0]] = f[1] if len(f) > 1 and re.fullmatch(r"[0-9a-f]{64}", f[1]) else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest")
    ap.add_argument("--tags")
    ap.add_argument("--latents-dir")
    ap.add_argument("--prepped-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--run", default="run1", help="scores go to OUT_DIR/RUN (one directory per scoring setting or repeat)")
    ap.add_argument("--cpus", type=int, default=400, help="CPUs the arrays may use together (the user's JHPCE limit)")
    ap.add_argument("--assume", nargs="+", metavar="TASK=N", help="estimate only, for N latents per task")
    a = ap.parse_args(argv)
    if a.assume:
        report(estimate({k: int(v) for k, v in (s.split("=") for s in a.assume)}, a.cpus), a.cpus)
        return
    need = ("manifest", "tags", "latents_dir", "prepped_dir", "out_dir")
    if [k for k in need if not getattr(a, k)]:
        ap.error("give --manifest --tags --latents-dir --prepped-dir --out-dir, or --assume")
    with open(a.manifest) as f:
        man = {r["tag"]: r for r in csv.DictReader((ln for ln in f if not ln.startswith("#")), delimiter="\t")}
    lat, pre_dir, out = (os.path.abspath(p) for p in (a.latents_dir, a.prepped_dir, a.out_dir))
    errs, lines = [], {}
    for tag, want in read_tags(a.tags).items():
        r, npz = man.get(tag), os.path.join(lat, tag + ".npz")
        pre = os.path.join(pre_dir, f"{r['task']}__{r['counts']}.h5ad") if r else None
        if r is None or r["task"] not in SIZES or not os.path.isfile(npz) or not os.path.isfile(pre):
            errs.append(f"{tag}: not in the manifest, task without a size, or latent {npz} / prepped file {pre} missing")
            continue
        sha = sha256(npz)
        if want and sha != want:
            errs.append(f"{tag}: latent SHA-256 {sha} != expected {want}")
            continue
        meta = dict(latent_sha256=sha, experiment=r["experiment"], task=r["task"], arm=r["arm"], lam=r["lam"],
                    cond=r["cond"], seed=r["seed"])
        lines.setdefault(r["task"], []).append("\t".join([tag, npz, pre, json.dumps(meta)]))
    if errs or not lines:
        raise SystemExit("\n".join(errs[:20] + [f"{len(errs)} problems in {a.tags}" if errs else "no tags"]))
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    rows = estimate({t: len(v) for t, v in lines.items()}, a.cpus)
    for x in rows:
        lst, sb = (os.path.join(out, f"{p}_{x['task']}.{e}") for p, e in (("list", "tsv"), ("score", "sbatch")))
        with open(lst, "w") as f:
            f.write("\n".join(lines[x["task"]]) + "\n")
        with open(sb, "w") as f:
            f.write(sbatch(x, x["latents"], out, a.run, head, lst))
        print(f"sbatch {sb}")
    print(f"scorer commit {head}; each submission needs the user's go (SI-29)")
    report(rows, a.cpus)


if __name__ == "__main__":
    main()
