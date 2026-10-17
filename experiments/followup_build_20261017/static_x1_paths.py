#!/usr/bin/env python
"""Static check (no fit, no score read): which X1 and X13 rows and code paths does the follow-up build touch?

1. The builder at this commit with the v3 settings rebuilds manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv byte for byte.
2. v4 and v3 agree on every row outside X3 / X6 / X7 / X8 / X12 (tags and all columns): see diff_report.py.
3. Every function changed since the tag (scripts/ and src/, found with git diff and python's ast) must have a declared
   GUARD: the condition on a manifest row under which the changed code behaves differently. Each guard is evaluated on
   every X1 and X13 row of the v3 manifest; the check fails if any row satisfies one, or if a changed function has no
   guard (new changed code must be declared here).
Writes results/x1_static_paths.csv (changed function, guard text, X1/X13 rows reaching it) and prints n=. Run from the
repo root: python experiments/followup_build_20261017/static_x1_paths.py [BASE]   (BASE default prereg-tier12-v3)
"""
import ast
import csv
import json
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
V3 = "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"
OUT = os.path.join(ROOT, "experiments", "followup_build_20261017", "results", "x1_static_paths.csv")
NEW_ARM = "discriminator_sn_critic_opt"


def sub(r):
    return json.loads(r["extra"]).get("subsample", {})


# (file, enclosing function) -> (guard text, predicate on a manifest row: True = the row reaches changed behaviour)
GUARDS = {
    ("scripts/build_paper_manifest.py", "<module>"): ("constants used by design v4 only", lambda r: False),
    ("scripts/build_paper_manifest.py", "build"): ("followup_design == 'v4' (default v3; byte-identical, check 1)", lambda r: False),
    ("scripts/build_paper_manifest.py", "main"): ("--followup-design flag and header text; v3 header unchanged (check 1)", lambda r: False),
    ("scripts/fit_paper_config.py", "<module>"): (f"arm == {NEW_ARM} (set members) or X8/X7 spec constants", lambda r: r["arm"] == NEW_ARM),
    ("scripts/fit_paper_config.py", "resolve_reference"): ("subsample kind == reference_composition", lambda r: sub(r).get("kind") == "reference_composition"),
    ("scripts/fit_paper_config.py", "subsample"): ("'subsample' in extra (X1 and X13 rows have none)", lambda r: bool(sub(r))),
    ("scripts/fit_paper_config.py", "x8_draw"): ("new function: batches spec with draw == nested_chain_v1", lambda r: sub(r).get("draw") == "nested_chain_v1"),
    ("scripts/fit_paper_config.py", "main"): ("'subsample' in extra of kind composition / reference_composition", lambda r: sub(r).get("kind") in ("composition", "reference_composition")),
    ("scripts/scvi_adversarial_plan.py", "<module>"): (f"arm == {NEW_ARM} (_CRITIC_OPT_ARMS)", lambda r: r["arm"] == NEW_ARM),
    ("scripts/scvi_adversarial_plan.py", "WassersteinAdversarialTrainingPlan.__init__"): (f"arm == {NEW_ARM}; any other arm: dict.get returns the arm", lambda r: r["arm"] == NEW_ARM),
    ("scripts/scvi_adversarial_plan.py", "WassersteinAdversarialTrainingPlan.configure_optimizers"): (f"arm == {NEW_ARM}", lambda r: r["arm"] == NEW_ARM),
    ("scripts/scvi_adversarial_plan.py", "WassersteinAdversarialTrainingPlan.training_step"): ("critic arm with spectral norm (arm ends _sn); I036", lambda r: r["arm"].endswith("_sn")),
    ("scripts/prereg_rules.py", "<module>"): (f"BASE_ARM gains the key {NEW_ARM}", lambda r: r["arm"] == NEW_ARM),
    ("scripts/cost_model.py", "arm_label"): ("arm name contains _critic_opt (cost model only)", lambda r: "_critic_opt" in r["arm"]),
}


def changed_functions(base):
    """{(file, enclosing function): [new line numbers]} for scripts/ and src/ python files, base..working tree."""
    files = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", base, "--", "scripts", "src"], capture_output=True, text=True,
                           check=True, cwd=ROOT).stdout.split()
    out = {}
    for f in [x for x in files if x.endswith(".py")]:
        diff = subprocess.run(["git", "diff", "-U0", "--no-color", base, "--", f], capture_output=True, text=True,
                              check=True, cwd=ROOT).stdout
        lines = set()
        for h in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
            s, n = int(h[1]), int(h[2]) if h[2] is not None else 1
            lines |= set(range(s, s + max(n, 1)))
        tree = ast.parse(open(os.path.join(ROOT, f)).read())
        spans = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append((node.lineno, node.end_lineno, node.name))
            elif isinstance(node, ast.ClassDef):
                spans += [(m.lineno, m.end_lineno, f"{node.name}.{m.name}") for m in node.body
                          if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for ln in sorted(lines):
            name = next((n for a, b, n in spans if a <= ln <= b), "<module>")
            out.setdefault((f, name), []).append(ln)
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "prereg-tier12-v3"
    cmd = [sys.executable, "scripts/build_paper_manifest.py", "--backbone", "stock", "--design", "pilot", "--uncond-seeds", "5",
           "--bary-iter", "10", "--out", "/tmp/static_v3_rebuilt.tsv"]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    if open("/tmp/static_v3_rebuilt.tsv", "rb").read() != open(os.path.join(ROOT, V3), "rb").read():
        raise SystemExit("check 1 failed: the builder no longer rebuilds the v3 manifest byte for byte")
    m = pd.read_csv(os.path.join(ROOT, V3), sep="\t", comment="#", dtype=str, keep_default_na=False)
    rows = m[m.experiment.isin(["X1", "X13"])].to_dict("records")
    fit_rows = [r for r in rows if r["arm"] not in ("harmony", "pca", "scanorama")]
    res, bad = [], []
    for (f, fn), lines in sorted(changed_functions(base).items()):
        if (f, fn) not in GUARDS:
            bad.append(f"changed function without a declared guard: {f}::{fn} lines {lines[:5]}")
            continue
        text, pred = GUARDS[(f, fn)]
        # rows that run this file's code: the builder's own rows are the manifest (check 1); fits and the plan: fit rows
        n = sum(bool(pred(r)) for r in fit_rows)
        res.append(dict(file=f, function=fn, changed_lines=len(lines), guard=text, x1_x13_rows_reaching=n, rows_checked=len(fit_rows)))
        if n:
            bad.append(f"{f}::{fn}: {n} X1/X13 rows satisfy the guard '{text}'")
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    for r in res:
        print(r["file"], r["function"], "lines", r["changed_lines"], "| rows reaching:", r["x1_x13_rows_reaching"])
    print(f"n={len(fit_rows)} X1/X13 fit rows checked against {len(res)} changed functions; v3 manifest rebuilt byte-identical")
    if bad:
        raise SystemExit("FAIL: " + "; ".join(bad))


if __name__ == "__main__":
    main()
