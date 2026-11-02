#!/usr/bin/env python
"""Map the eight-task penalty-strength sweep (experiment A1) and the baselines into one analysis input (docs/SWEEP_ANALYSIS.md).

Sources (data root D, read only):
  three original tasks (atac_small, immune, sim1): run list D/tier12_v3/rules/r4_pass2/m.r4a1.tsv (A1 rows), score files listed in
      scores_list.txt there (the Sinkhorn redo superseded tier12_v2 Sinkhorn fits under the same tags: the listed file and the latent
      next to it are used), failures_merged.csv;
  five new tasks (immune_hum_mou, atac_large, lung, sim2, pancreas): run list manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv,
      D/jhpce_tier12/pilot5_jhpce/{scores,latents,failures.csv};
  baselines: the 448 rows of manifests/paper_manifest_stock_pilot_u5_b10_v3.x1_sinkhorn_redo.tsv with experiment X13 or arm scvi_adv,
      scored in the --baseline-dirs (runner layout scores/<tag>.csv, latents/<tag>.npz, failures.csv). A tag found in two directories
      is refused.
Writes to --out: manifest.tsv (every row the analysis reads), scores_list.txt (one score file per scored row), failures.csv (merged),
latents.tsv (tag -> latent file), baseline_coverage.csv (task x method: expected, scored, failed, missing) and inputs.json (sources
with SHA-256, counts, missing baseline rows). Without --require-baselines, missing baseline rows are left out of manifest.tsv and
listed; with it, any missing row stops the run.

Usage: python scripts/sweep_inputs.py --out <dir> [--data-root D] [--baseline-dirs DIR ...] [--require-baselines] [--tasks T ...]
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_rules as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.environ.get("WCD_DATA_ROOT", "/home/kendall/experiment_data/wasserstein-critic-deconfounding")
PILOT3_TASKS = ("atac_small", "immune", "sim1")
PILOT5_TASKS = ("immune_hum_mou", "atac_large", "lung", "sim2", "pancreas")
PILOT5_MANIFEST = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv")
RUN_LIST = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v3.x1_sinkhorn_redo.tsv")
SWEEP_SEEDS = ("100", "101", "102")


def runner_layout(d):
    """(scores, latents, failures) of a stage directory in the runner's layout: {tag: file}, {tag: file}, DataFrame."""
    for sub in ("scores", "latents"):
        P._require(os.path.isdir(os.path.join(d, sub)), f"{d}: no {sub}/ directory (runner layout expected)")
    fp = os.path.join(d, "failures.csv")
    P._require(os.path.isfile(fp), f"{d}: no failures.csv (runner layout expected; an empty table has a header only)")
    files = {sub: {f[:-len(ext)]: os.path.join(d, sub, f) for f in os.listdir(os.path.join(d, sub)) if f.endswith(ext)}
             for sub, ext in (("scores", ".csv"), ("latents", ".npz"))}
    return files["scores"], files["latents"], P.read_failures(fp)


def select(rows, scores, latents, failures, source):
    """Which rows of `rows` are scored (file + latent), failed, or neither; a scored row without its latent is refused."""
    failed = set(failures.tag) & set(rows.tag)
    scored = [t for t in rows.tag if t in scores]
    both = sorted(failed & set(scored))
    P._require(not both, f"{source}: {len(both)} rows both scored and failed, e.g. {both[:3]}")
    no_latent = sorted(t for t in scored if t not in latents)
    P._require(not no_latent, f"{source}: {len(no_latent)} scored rows without a latent file, e.g. {no_latent[:3]}")
    missing = sorted(set(rows.tag) - set(scored) - failed)
    return scored, sorted(failed), missing


def sweep_rows(root, tasks):
    """The sweep's rows, score files, latent files and failure tables, from the two pilot sources."""
    parts = []
    r4 = os.path.join(root, "tier12_v3", "rules", "r4_pass2")
    m3, _ = P.read_manifest(os.path.join(r4, "m.r4a1.tsv"))
    m3 = m3[(m3.experiment == "A1") & m3.task.isin(tasks)]
    listed = [ln.strip() for ln in open(os.path.join(r4, "scores_list.txt")) if ln.strip()]
    s3 = {os.path.basename(f)[:-4]: f for f in listed}
    P._require(len(s3) == len(listed), "scores_list.txt lists a tag twice")
    l3 = {t: os.path.join(os.path.dirname(os.path.dirname(f)), "latents", t + ".npz") for t, f in s3.items()}
    l3 = {t: f for t, f in l3.items() if os.path.isfile(f)}
    parts.append(("tier12_v3/rules/r4_pass2", m3, s3, l3, P.read_failures(os.path.join(r4, "failures_merged.csv")),
                  [os.path.join(r4, "m.r4a1.tsv"), os.path.join(r4, "scores_list.txt"), os.path.join(r4, "failures_merged.csv")]))
    m5, _ = P.read_manifest(PILOT5_MANIFEST)
    m5 = m5[(m5.experiment == "A1") & m5.task.isin(tasks)]
    d5 = os.path.join(root, "jhpce_tier12", "pilot5_jhpce")
    s5, l5, f5 = runner_layout(d5)
    parts.append(("jhpce_tier12/pilot5_jhpce", m5, s5, l5, f5, [PILOT5_MANIFEST, os.path.join(d5, "failures.csv")]))
    return parts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--baseline-dirs", nargs="*", default=[])
    ap.add_argument("--require-baselines", action="store_true", help="stop when any of the 448 baseline rows is neither scored nor failed")
    ap.add_argument("--tasks", nargs="+", default=sorted(PILOT3_TASKS + PILOT5_TASKS))
    ap.add_argument("--run-list", default=RUN_LIST, help="the committed run list whose X13 and scvi_adv rows are the baselines")
    a = ap.parse_args(argv)
    bad = sorted(set(a.tasks) - set(PILOT3_TASKS + PILOT5_TASKS))
    P._require(not bad, f"unknown tasks {bad}")
    os.makedirs(a.out, exist_ok=False)
    manifests, score_files, latent_files, failures, sources, counts = [], {}, {}, [], {}, {}
    for name, rows, scores, latents, fails, src in sweep_rows(a.data_root, a.tasks):
        scored, failed, missing = select(rows, scores, latents, fails, name)
        P._require(not missing, f"{name}: {len(missing)} sweep rows are neither scored nor failed, e.g. {missing[:3]}")
        P._require(set(rows.seed) <= set(SWEEP_SEEDS), f"{name}: sweep seeds {sorted(set(rows.seed))} are not {SWEEP_SEEDS}")
        manifests.append(rows)
        score_files.update({t: scores[t] for t in scored})
        latent_files.update({t: latents[t] for t in scored})
        failures.append(fails[fails.tag.isin(set(failed))])
        sources[name] = [dict(path=p, sha256=P.sha256(p)) for p in src]
        counts[name] = dict(rows=len(rows), scored=len(scored), failed=len(failed))
    # baselines: the committed run list, scored in the given directories
    run, _ = P.read_manifest(a.run_list)
    base = run[((run.experiment == "X13") | (run.arm == "scvi_adv")) & run.task.isin(a.tasks)]
    sources["run_list"] = [dict(path=a.run_list, sha256=P.sha256(a.run_list))]
    scores, latents, fails_b, where = {}, {}, [], {}
    for d in a.baseline_dirs:
        s, l, f = runner_layout(d)
        s = {t: p for t, p in s.items() if t in set(base.tag)}
        twice = sorted(set(s) & set(scores))
        P._require(not twice, f"{len(twice)} baseline tags scored in two directories, e.g. {twice[:3]} ({d} and {[where[t] for t in twice[:3]]})")
        scores.update(s)
        where.update({t: d for t in s})
        latents.update({t: p for t, p in l.items() if t in set(base.tag)})
        fails_b.append(f[f.tag.isin(set(base.tag))])
        sources[d] = [dict(path=os.path.join(d, "failures.csv"), sha256=P.sha256(os.path.join(d, "failures.csv")))]
    fails_b = pd.concat(fails_b, ignore_index=True) if fails_b else pd.DataFrame(columns=["tag", "status", "detail"])
    P._require(not fails_b.tag.duplicated().any(), "a baseline tag is listed as failed in two directories")
    scored, failed, missing = select(base, scores, latents, fails_b, "baselines")
    status = pd.Series("missing", index=base.tag)
    status[scored], status[failed] = "scored", "failed"
    cov = base.assign(status=status.values).groupby(["task", "arm"]).status.value_counts().unstack(fill_value=0)
    cov = cov.reindex(columns=["scored", "failed", "missing"], fill_value=0)
    cov.insert(0, "expected", cov.sum(axis=1))
    cov = cov.reset_index()
    if a.require_baselines:
        P._require(not missing, f"--require-baselines: {len(missing)} of {len(base)} baseline rows are neither scored nor failed: "
                                + ", ".join(missing))
    manifests.append(base[~base.tag.isin(set(missing))])
    score_files.update({t: scores[t] for t in scored})
    latent_files.update({t: latents[t] for t in scored})
    failures.append(fails_b)
    M = pd.concat(manifests, ignore_index=True)
    P._require(not M.tag.duplicated().any(), "a tag appears twice across the sources")
    F = pd.concat(failures, ignore_index=True)
    P._require(not F.tag.duplicated().any(), "a tag is listed as failed twice")
    P.write_manifest(M, ["# sweep analysis input (scripts/sweep_inputs.py): A1 sweep rows of the eight tasks + the scored or failed "
                         "baseline rows of the committed run list"], os.path.join(a.out, "manifest.tsv"),
                     f"tasks {' '.join(a.tasks)}; {len(missing)} baseline rows missing (listed in inputs.json)")
    with open(os.path.join(a.out, "scores_list.txt"), "w") as f:
        f.write("".join(score_files[t] + "\n" for t in sorted(score_files)))
    F.to_csv(os.path.join(a.out, "failures.csv"), index=False)
    with open(os.path.join(a.out, "latents.tsv"), "w") as f:
        f.write("".join(f"{t}\t{latent_files[t]}\n" for t in sorted(latent_files)))
    cov.to_csv(os.path.join(a.out, "baseline_coverage.csv"), index=False)
    info = dict(tasks=a.tasks, data_root=a.data_root, baseline_dirs=a.baseline_dirs, require_baselines=a.require_baselines,
                sources=sources, counts=dict(counts, baselines=dict(rows=len(base), scored=len(scored), failed=len(failed),
                                                                     missing=len(missing))),
                baseline_rows_missing=missing, baseline_coverage=cov.to_dict("records"),
                n_manifest_rows=len(M), n_score_files=len(score_files), n_failed=len(F))
    with open(os.path.join(a.out, "inputs.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"sweep inputs: {len(M)} rows ({len(score_files)} scored, {len(F)} failed) on {len(a.tasks)} tasks; baselines "
          f"{len(scored)} scored, {len(failed)} failed, {len(missing)} missing of {len(base)} -> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
