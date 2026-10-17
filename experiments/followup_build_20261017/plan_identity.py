#!/usr/bin/env python
"""Resolved-plan identity WITHOUT fitting: tag prereg-tier12-v3 (OLD) vs the commit to be tagged (NEW) (review F5, item 3).

Runs the 57 X1/X13/X3-weighted identity rows through each checkout's own scripts/fit_paper_config.py with zero optimiser
steps (nofit_worker.py): the model, training plan and optimisers are built as in a real fit and the saved config holds
the resolved plan (every optimiser param group, gradient_penalty, lambda_gp, spectral_norm, adversary settings), the
reference batch, cell counts and library versions. identity_compare.py then requires, per row: exit code 0 at both
commits, equal arrays (the seeded initial latent), equal config apart from git_sha / git_dirty / fit_seconds, equal model
weights, zero optimiser steps. Control (determinism): OLD is run a second time on the first --control rows (default 5)
and compared with the first OLD run by the same checks; if the control fails, a difference elsewhere is not interpretable.
Takes minutes on one thread (run under `nice`). It makes a setting change visible that a 1-epoch latent could hide; it does
not replace fit_identity.sh (real 1-epoch fits).
Input: --prepped DIR (the real prepped_scib folder) or --synthetic (stand-in sim2 input made by nofit_worker.py; the
plan does not depend on the data values, but batch names and sizes differ from the real sim2, so a real-data run is
the record to cite). Exactly one is required.
Usage: python plan_identity.py OLD_CHECKOUT NEW_CHECKOUT OUT_DIR ROWS_TSV (--prepped DIR | --synthetic) [--control 5]
Exit code 0 only if every row and the control pass.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*cmd, env=None):
    return subprocess.run([sys.executable, *cmd], env=env, check=False).returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("out")
    ap.add_argument("rows")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepped")
    g.add_argument("--synthetic", action="store_true")
    ap.add_argument("--control", type=int, default=5)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    if os.path.exists(f"{out}/exit_codes.txt"):
        raise SystemExit(f"{out} already holds a run: use a new folder (the worker skips existing latents)")
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    if a.synthetic:
        prepped = f"{out}/synthetic_prepped"
        if run(f"{HERE}/nofit_worker.py", "--make-synthetic", f"{prepped}/sim2__scib.h5ad"):
            raise SystemExit("could not make the synthetic input")
    else:
        prepped = os.path.abspath(a.prepped)
    env["PREPPED_DIR"] = prepped
    lines = open(a.rows).read().splitlines()           # line 1: '#' comment, line 2: column names, then the rows
    tags = [l.split("\t")[0] for l in lines[2:]]
    if len(tags) != len(set(tags)) or not tags or a.control > len(tags):
        raise SystemExit(f"{a.rows}: need distinct tags and at least {a.control} rows, got {len(tags)}")
    open(f"{out}/manifest.tsv", "w").write("\n".join(lines) + "\n")
    open(f"{out}/control_manifest.tsv", "w").write("\n".join(lines[:2 + a.control]) + "\n")
    open(f"{out}/tags.txt", "w").write("\n".join(tags) + "\n")
    open(f"{out}/control_tags.txt", "w").write("\n".join(tags[:a.control]) + "\n")
    open(f"{out}/exit_codes.txt", "w").close()
    for label, checkout, manifest in (("old", a.old, "manifest.tsv"), ("new", a.new, "manifest.tsv"),
                                      ("old2", a.old, "control_manifest.tsv")):
        if run(f"{HERE}/nofit_worker.py", checkout, label, out, f"{out}/{manifest}", f"{out}/exit_codes.txt", env=env):
            print(f"WARNING: worker {label} reported failed rows; the comparison below fails them", file=sys.stderr)
    rc_control = run(f"{HERE}/identity_compare.py", out, "plan_identity_control.csv", "--labels", "old", "old2",
                     "--tags", f"{out}/control_tags.txt", "--steps-zero")
    rc = run(f"{HERE}/identity_compare.py", out, "plan_identity.csv", "--steps-zero")
    return 1 if (rc or rc_control) else 0


if __name__ == "__main__":
    sys.exit(main())
