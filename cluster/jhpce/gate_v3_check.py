#!/usr/bin/env python
"""Local check of the harvested v3 L40S gate latents (CR2-01), run after the job, no cluster needed.
For every manifest row: the new latent exists, is finite, shape (n_cells, 10), and was written by this commit
(git_sha == --expected-sha, clean) on an L40S. Non-Sinkhorn rows must equal the 2026-10-03 L40S gate latent
(--old, same GPU model, code before the final-review fixes) bit for bit; Sinkhorn rows are excluded from that
comparison because the solver was corrected on purpose (SI-51).
Exit 0 all pass; 1 a latent is missing, non-finite, mislabelled or from the wrong commit; 2 provenance fine but some
non-Sinkhorn row differs from --old (then judge with scripts/gate_compare_latents.py against the local latents).
Usage: python cluster/jhpce/gate_v3_check.py --new DIR --old DIR --manifest M.tsv --expected-sha SHA [--out CSV]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def load(d, tag):
    z = np.load(os.path.join(d, f"{tag}.npz"))
    return np.asarray(z["z"]), json.loads(str(z["config"]))


def check(new, old, manifest, sha, gpu="L40S"):
    rows = []
    for _, r in pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False).iterrows():
        tag, sink = r["tag"], r["arm"] == "sinkhorn"
        z, cfg = load(new, tag)
        prov = (bool(np.isfinite(z).all()) and z.ndim == 2 and z.shape[1] == 10 and cfg["git_sha"] == sha
                and not cfg["git_dirty"] and gpu in cfg["gpu"] and cfg["row"]["tag"] == tag)
        if sink:
            same = None
        else:
            zo, _ = load(old, tag)
            same = z.shape == zo.shape and bool(np.array_equal(z, zo))
        rows.append(dict(tag=tag, arm=r["arm"], valid=prov, equal_to_old=same, max_abs_diff=(
            None if same is None or z.shape != zo.shape else float(np.abs(z.astype(np.float64) - zo).max()))))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--out", default="gate_v3_check.csv")
    a = ap.parse_args(argv)
    df = check(a.new, a.old, a.manifest, a.expected_sha)
    df.to_csv(a.out, index=False)
    n_bad, n_diff = int((~df.valid).sum()), int((df.equal_to_old == False).sum())  # noqa: E712 (None means not compared)
    print(f"rows={len(df)} invalid={n_bad} non-sinkhorn compared={int(df.equal_to_old.notna().sum())} differ={n_diff}")
    sys.exit(1 if n_bad else 2 if n_diff else 0)


if __name__ == "__main__":  # pragma: no cover (entry point)
    main()
