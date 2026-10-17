#!/usr/bin/env python
"""Compare the outputs of the same rows fitted at two commits (OLD = tag prereg-tier12-v3, NEW = the commit to be tagged).

Used by fit_identity.sh (real 1-epoch fits) and plan_identity.sh (no fit: the training plan is built and zero optimiser
steps are taken). Reads OUT/tags.txt, OUT/exit_codes.txt (lines "<label> <tag> <exit code>") and OUT/out_<label>/ of the
two labels, compares per tag

  * both exit codes are 0;
  * every array saved in latents/<tag>.npz is equal (np.array_equal: z, obs_names, batch, celltype);
  * the saved config (JSON) is equal apart from git_sha, git_dirty and fit_seconds: the row, the reference batch, cell
    counts, subsample info, importance weights, library versions and the resolved training plan (every optimiser param
    group, gradient_penalty, lambda_gp, spectral_norm, adversary settings); the loss history is equal too;
  * the saved model weights models/<tag>/model.pt are equal (state_dict tensors; skipped with --no-models);
  * with --steps-zero (plan_identity.sh): the trainer took no optimiser step at either commit (OUT/out_<label>/steps.txt).

Writes OUT/<csv name>, prints one summary line, exits 1 unless every tag passes every check. A missing file is a failure.
Usage: identity_compare.py OUT CSV_NAME [--labels old new] [--tags FILE] [--no-models] [--steps-zero]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

PROVENANCE = ("git_sha", "git_dirty", "fit_seconds")   # the only config keys allowed to differ between commits


def _diff(a, b, path=""):
    """Paths of the leaves at which two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in _diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [path or "/"]


def compare_latents(pa, pb):
    """(arrays identical, max |z_a - z_b| or '', config paths that differ, history identical) of two latent files."""
    with np.load(pa, allow_pickle=False) as a, np.load(pb, allow_pickle=False) as b:
        keys = sorted(set(a.files) | set(b.files))
        arrays = [k for k in keys if k not in ("config", "history")]
        same = all(k in a.files and k in b.files and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
                   for k in arrays)
        zdiff = "" if a["z"].shape != b["z"].shape else repr(float(np.max(np.abs(a["z"] - b["z"]))))
        ca, cb = (json.loads(str(x["config"])) for x in (a, b))
        for c in (ca, cb):
            for k in PROVENANCE:
                c.pop(k, None)
        hist = str(a["history"]) == str(b["history"])
    return same, zdiff, _diff(ca, cb), hist


def compare_models(pa, pb):
    import torch
    sa, sb = (torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"] for p in (pa, pb))

    def eq(x, y):       # the state dict may nest dicts (scvi-tools wraps module and plan states)
        if isinstance(x, dict) or isinstance(y, dict):
            return isinstance(x, dict) and isinstance(y, dict) and sorted(x) == sorted(y) and all(eq(x[k], y[k]) for k in x)
        return torch.equal(x, y)
    return eq(sa, sb)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("csv_name")
    ap.add_argument("--labels", nargs=2, default=["old", "new"])
    ap.add_argument("--tags", help="tag list file (default OUT/tags.txt)")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--steps-zero", action="store_true")
    args = ap.parse_args(argv)
    out, (la, lb) = args.out, args.labels
    tags = open(args.tags or f"{out}/tags.txt").read().split()
    rc = {(v, t): int(c) for v, t, c in (l.split() for l in open(f"{out}/exit_codes.txt"))}
    steps = {}
    if args.steps_zero:
        for v in (la, lb):
            steps[v] = {t: int(s) for t, s in (l.split() for l in open(f"{out}/out_{v}/steps.txt"))}
    rows = []
    for t in tags:
        r = dict(tag=t, **{f"rc_{v}": rc.get((v, t)) for v in (la, lb)})
        p = {v: f"{out}/out_{v}/latents/{t}.npz" for v in (la, lb)}
        if all(os.path.exists(x) for x in p.values()):
            same, zdiff, cfg, hist = compare_latents(p[la], p[lb])
            r.update(arrays_identical=str(same), max_abs_diff=zdiff, config_identical=str(not cfg),
                     config_differs_at=";".join(cfg), history_identical=str(hist))
        else:
            r.update(arrays_identical="missing", max_abs_diff="", config_identical="missing", config_differs_at="",
                     history_identical="missing")
        if not args.no_models:
            m = {v: f"{out}/out_{v}/models/{t}/model.pt" for v in (la, lb)}
            r["model_identical"] = str(compare_models(m[la], m[lb])) if all(os.path.exists(x) for x in m.values()) else "missing"
        if args.steps_zero:
            r["steps_" + la], r["steps_" + lb] = steps[la].get(t, "missing"), steps[lb].get(t, "missing")
        rows.append(r)
    with open(f"{out}/{args.csv_name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def ok(r):
        return (r[f"rc_{la}"] == 0 and r[f"rc_{lb}"] == 0 and r["arrays_identical"] == "True"
                and r["config_identical"] == "True" and r["history_identical"] == "True"
                and r.get("model_identical", "True") == "True"
                and (not args.steps_zero or (r["steps_" + la] == 0 and r["steps_" + lb] == 0)))
    bad = [r["tag"] for r in rows if not ok(r)]
    print("IDENTITY", len(rows) - len(bad), "of", len(rows), f"rows pass every check ({la} vs {lb}; exit codes 0, arrays, "
          "config apart from provenance, history" + ("" if args.no_models else ", model weights")
          + (", zero optimiser steps" if args.steps_zero else "") + "); failing:", bad)
    return 1 if bad else 0


if __name__ == "__main__":   # pragma: no cover - entry point
    sys.exit(main())
