#!/usr/bin/env python
"""Link the reusable non-Sinkhorn X1 outputs of the v2 local folder into the new v3 local folder, by FILE NAME only.

Why (review F3): the 19 X1 rows scored before the Sinkhorn redo sit in D/tier12_v2/X1_local (hashed under SI-79; that folder
is never written to). The post-redo X1 run uses a new folder, D/tier12_v3/X1_local. SI-54 lets non-Sinkhorn outputs made at
c88cce3 count as v3, so for every X1 manifest row whose arm is not in EXCLUDE_OLD_ARMS (sinkhorn: refused) this script
symlinks latents/<tag>.npz, models/<tag> and scores/<tag>.csv when they exist. It opens no file and prints counts only
(SI-78/79): a symlink copies no bytes. The stage runner then checks every linked file against the manifest row (its settings
lam, adv_input, zstd, cond, seed, commit); a row whose settings changed in the redo is reported as `invalid output` with its tag,
and its links are removed with --unlink TAG ... before the run (PLAN.md step X2). Nothing here accepts an output by itself.

Usage: python link_v2_outputs.py --manifest X1.tsv --tasks atac_small immune sim1 --old-dir D/tier12_v2/X1_local \
           --new-dir D/tier12_v3/X1_local            (creates --new-dir; refuses an existing one)
       python link_v2_outputs.py --new-dir D/tier12_v3/X1_local --unlink TAG [TAG ...]   (removes those tags' links only)
"""
import argparse
import os

import pandas as pd

EXCLUDE_OLD_ARMS = ("sinkhorn",)        # same rule as experiments/sinkhorn_redo_stage2/stage2_rules.py (SI-54)
KINDS = {"latents": "{tag}.npz", "models": "{tag}", "scores": "{tag}.csv"}


def paths(root, tag):
    return {k: os.path.join(root, k, v.format(tag=tag)) for k, v in KINDS.items()}


def link(a):
    M = pd.read_csv(a.manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    rows = M[(M.experiment == "X1") & M.task.isin(a.tasks)]
    if not len(rows):
        raise SystemExit(f"no X1 rows of {a.tasks} in {a.manifest}")
    if os.path.exists(a.new_dir):
        raise SystemExit(f"{a.new_dir} exists: the link step creates the folder (refused to reuse it)")
    n = dict.fromkeys(KINDS, 0)
    n_tags = 0
    for tag, arm in zip(rows.tag, rows.arm):
        old = paths(a.old_dir, tag)
        have = [k for k in KINDS if os.path.lexists(old[k])]
        if arm in EXCLUDE_OLD_ARMS or not have:
            continue
        n_tags += 1
        for k in have:
            dst = paths(a.new_dir, tag)[k]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(os.path.abspath(old[k]), dst)
            n[k] += 1
    os.makedirs(a.new_dir, exist_ok=True)                  # also when nothing was reusable
    print(f"[link] {len(rows)} X1 rows; {n_tags} with reusable v2 outputs; linked {n}")
    return 0


def unlink(a):
    n = 0
    for tag in a.unlink:
        for dst in paths(a.new_dir, tag).values():
            if os.path.islink(dst):                        # only links: a real file or folder is never removed
                os.remove(dst)
                n += 1
            elif os.path.lexists(dst):
                raise SystemExit(f"{dst} is not a link made by this script; not removed")
    print(f"[unlink] {n} links removed for {len(a.unlink)} tags")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--unlink", nargs="+", default=None)
    ap.add_argument("--manifest")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--old-dir")
    a = ap.parse_args(argv)
    if a.unlink:
        return unlink(a)
    if not (a.manifest and a.tasks and a.old_dir):
        raise SystemExit("--manifest, --tasks and --old-dir are required to link")
    return link(a)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
