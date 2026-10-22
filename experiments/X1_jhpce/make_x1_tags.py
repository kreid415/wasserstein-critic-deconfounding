#!/usr/bin/env python
"""Tags files of the five X1 JHPCE fit jobs (one per task), derived from the R2/R3-resolved X1 manifest (read-only).

Selection: experiment X1, the five JHPCE tasks of CONSTRAINTS.md SI-39, every arm except none / scvi_adv (those
rule-free X1 rows were fitted by the filler jobs at prereg-tier12-v2 and are refused here if they reappear). Expected:
360 rows per task, 1,800 in all. Rows are copied by tag only (manifest order); a row that still holds a placeholder
(non-numeric lam, adv_input not mean/sample, non-integer zstd) is refused, as in cluster/jhpce/make_tags.py.
Writes x1_jhpce_<task>.tags (one per job), x1_jhpce.tags (union) and x1_jhpce_counts.csv into --out-dir;
--check compares instead of writing (exit 1 on any difference).
Usage (main repo root):
  python experiments/X1_jhpce/make_x1_tags.py --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.x1.tsv \
      --filler-tags <prereg-tier12-v2 checkout>/cluster/jhpce/tags/fillers_x1_x13.tags --out-dir experiments/X1_jhpce/tags
--job pilot5: the same for the A1 pilot on the five tasks (every A1 row, 378 per task, no filler tags; files
pilot5_jhpce_<task>.tags):
  python experiments/X1_jhpce/make_x1_tags.py --job pilot5 \
      --manifest manifests/paper_manifest_stock_pilot_u5_b10_v3.pilot5.tsv --out-dir experiments/pilot5_jhpce/tags
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

JHPCE_TASKS = ("atac_large", "immune_hum_mou", "lung", "pancreas", "sim2")     # CONSTRAINTS.md SI-39
RULE_FREE = ("none", "scvi_adv")                                                 # fitted by the filler jobs
RULE = ("docs/PREREG.md sec 0: X1 after R2/R3 (docs/prereg/r2r3_record.json); X1 adversarial arms on the JHPCE tasks of "
        "CONSTRAINTS.md SI-39; rule-free X1 rows (none, scvi_adv) were fitted by the filler jobs")
# job kind: file prefix, experiment, arms left out, rows per task (x 5 tasks = total), rule line. 'pilot5' = the A1
# penalty-strength pilot on the five JHPCE tasks (manifest build_paper_manifest.py --a1-tasks): every A1 row, the
# strength-0 'none' rows included, since no filler job fits them.
JOBS = {"x1": dict(stage="x1_jhpce", experiment="X1", skip=RULE_FREE, per_task=360, rule=RULE),
        "pilot5": dict(stage="pilot5_jhpce", experiment="A1", skip=(), per_task=378,
                       rule="A1 penalty-strength pilot on the five JHPCE tasks of CONSTRAINTS.md SI-39: every A1 row of the "
                            "manifest (build_paper_manifest.py --a1-tasks), strength-0 rows included")}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_tags(path):
    return [l.split("#", 1)[0].strip() for l in open(path) if l.split("#", 1)[0].strip()]


def placeholders(r):
    out = []
    try:
        float(r["lam"])
    except ValueError:
        out.append(f"lam={r['lam']!r}")
    if r["arm"] not in RULE_FREE and r["adv_input"] not in ("mean", "sample"):
        out.append(f"adv_input={r['adv_input']!r}")
    try:
        int(r["zstd"])
    except ValueError:
        out.append(f"zstd={r['zstd']!r}")
    return out


def derive(manifest, filler_tags, job="x1"):
    j = JOBS[job]
    M = pd.read_csv(manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    if M.tag.duplicated().any():
        raise SystemExit(f"{manifest}: duplicate tags")
    sel = M[(M.experiment == j["experiment"]) & M.task.isin(JHPCE_TASKS) & ~M.arm.isin(j["skip"])].copy()
    bad = {r["tag"]: placeholders(r) for r in sel.to_dict("records") if placeholders(r)}
    if bad:
        raise SystemExit(f"selected rows hold placeholders: {list(bad.items())[:5]}")
    per = sel.groupby("task").size().to_dict()
    if len(sel) != j["per_task"] * len(JHPCE_TASKS) or any(per.get(t, 0) != j["per_task"] for t in JHPCE_TASKS):
        raise SystemExit(f"selection has {len(sel)} rows {per}, expected {j['per_task']} per task "
                         f"({j['per_task'] * len(JHPCE_TASKS)})")
    if job != "x1":      # no filler jobs fitted rows of this job
        return sel
    fill = set(read_tags(filler_tags))
    overlap = sorted(fill & set(sel.tag))
    if overlap:
        raise SystemExit(f"{len(overlap)} selected tags were filler rows: {overlap[:5]}")
    fill_x1 = M[M.tag.isin(fill) & (M.experiment == "X1")]
    if not fill_x1.arm.isin(RULE_FREE).all():
        raise SystemExit("filler tags include adversarial X1 rows")
    return sel


def render(manifest, msha, label, rows, tasks, job="x1"):
    j = JOBS[job]
    counts = rows.groupby(["task", "arm"]).size()
    skip = f" except {', '.join(j['skip'])}" if j["skip"] else ""
    lines = ["# tags file for scripts/run_stage.py --tags-file (one manifest tag per line; '#' starts a comment)",
             f"# selection: {j['stage']} {label}: {j['experiment']} arms{skip}; tasks {' '.join(tasks)}",
             f"# rule: {j['rule']}",
             f"# manifest: {manifest} sha256 {msha}",
             f"# rows: {len(rows)}; per task and arm: "
             + "; ".join(f"{t} " + ", ".join(f"{a} {int(n)}" for (tt, a), n in counts.items() if tt == t) for t in tasks),
             "# generated by: python experiments/X1_jhpce/make_x1_tags.py (manifest order)"]
    return "\n".join(lines + list(rows.tag)) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--job", choices=sorted(JOBS), default="x1")
    ap.add_argument("--filler-tags", help="required for --job x1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.job == "x1" and not a.filler_tags:
        ap.error("--job x1 needs --filler-tags")
    stage = JOBS[a.job]["stage"]
    msha = sha256_bytes(open(a.manifest, "rb").read())
    sel = derive(a.manifest, a.filler_tags, a.job)
    files = {f"{stage}.tags": render(a.manifest, msha, "all", sel, list(JHPCE_TASKS), a.job)}
    for t in JHPCE_TASKS:
        files[f"{stage}_{t}.tags"] = render(a.manifest, msha, t, sel[sel.task == t], [t], a.job)
    files[f"{stage}_counts.csv"] = sel.groupby(["task", "arm", "cond"]).size().rename("rows").reset_index().to_csv(index=False)
    bad = []
    for fn, text in files.items():
        p = os.path.join(a.out_dir, fn)
        if a.check:
            if not os.path.isfile(p) or open(p).read() != text:
                bad.append(p)
        else:
            os.makedirs(a.out_dir, exist_ok=True)
            with open(p, "w") as f:
                f.write(text)
    print(json.dumps(dict(manifest=a.manifest, manifest_sha256=msha, rows=len(sel),
                          per_task=sel.groupby("task").size().to_dict(),
                          tags_sha256={fn: sha256_bytes(t.encode()) for fn, t in files.items() if fn.endswith(".tags")}), indent=1))
    if bad:
        print(f"MISMATCH: {bad}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
