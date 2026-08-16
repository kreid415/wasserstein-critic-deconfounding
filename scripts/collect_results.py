#!/usr/bin/env python
"""Collect every wave result into one table, and PROVE nothing was dropped.

WHY THIS SCRIPT EXISTS. The wave was merged with ad-hoc kernel code that globbed
results/wave/*.csv and then filtered on epochs_run.notna(). That silently lost THREE of
the nine experiments:

  E5 (run_biology.py)      writes a DIRECTORY -- results/wave/E5_<ds>/E5_summary_<ds>.csv --
                           so a flat results/wave/*.csv glob never matched it at all.
  E3 (run_baselines.py)    external baselines (harmony/scanorama/scvi/unintegrated) have no
                           epochs_run, because they are not our training loop, so the
                           notna() filter dropped every row.
  E6 (support_overlap.py)  a data-property analysis that trains no model: likewise no
                           epochs_run, likewise dropped.

Nothing failed and nothing warned. The merged table looked healthy at 1128 rows and was
reported as the complete wave; the gap only surfaced when a workspace wipe forced a
re-audit against the manifest.

So this script does the merge AND asserts coverage against the manifest. A missing
experiment is an error, not a silently shorter table.
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd


def read_all(results_dir):
    """Every result CSV, including the nested per-dataset directories E5 writes."""
    frames = []
    flat = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    # E5 writes E5_summary_<ds>.csv (one row per config) alongside purity/confusion
    # sidecars that have a different shape; only the summary belongs in the merged table.
    nested = sorted(glob.glob(os.path.join(results_dir, "E5_*", "E5_summary_*.csv")))
    for path in flat + nested:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            # A headerless/1-byte CSV means the shard failed and wrote nothing usable --
            # exactly how E6 failed with FileNotFoundError. Surface it, never skip silently.
            print(f"  WARNING empty/headerless: {path}", file=sys.stderr)
            continue
        if not len(df):
            continue
        df["_source"] = os.path.relpath(path, results_dir)
        if "experiment" not in df.columns or df["experiment"].isna().all():
            # E5/E6 summaries may not carry the column; derive it from the path.
            m = re.search(r"(E\d+)", df["_source"].iloc[0])
            if m:
                df["experiment"] = m.group(1)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no result CSVs under {results_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def manifest_experiments(manifest):
    M = pd.read_csv(manifest, sep="\t")
    return set(M["tag"].str.extract(r"(E\d+)", expand=False).dropna())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/wave")
    ap.add_argument("--manifest", default="scripts/wave_manifest.tsv")
    ap.add_argument("--out", default="wave_results_full.csv")
    ap.add_argument("--allow-missing", nargs="*", default=[],
                    help="experiments legitimately absent (e.g. still running)")
    args = ap.parse_args()

    D = read_all(args.results)
    want = manifest_experiments(args.manifest)
    got = set(D["experiment"].dropna().unique())
    missing = sorted(want - got - set(args.allow_missing))

    print(f"rows {len(D)}  experiments {sorted(got)}")
    for e in sorted(got):
        sub = D[D["experiment"] == e]
        has_ep = "epochs_run" in sub.columns and sub["epochs_run"].notna().any()
        print(f"  {e:4s} rows {len(sub):5d}  training_rows={has_ep}")

    # kBET is only defined where a latent exists, i.e. on training rows.
    if "epochs_run" in D.columns:
        tr = D[D["epochs_run"].notna()]
        if "kbet" in D.columns and len(tr):
            print(f"kbet on training rows: {int(tr['kbet'].notna().sum())}/{len(tr)}")

    D.to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    # THE ASSERTION THAT WOULD HAVE CAUGHT THE BUG.
    assert not missing, (
        f"experiments in the manifest but ABSENT from the merged table: {missing}. "
        "Do not report this as the complete wave. E5 writes a nested directory and "
        "E3/E6 have no epochs_run column -- check the reader, not just the shard."
    )


if __name__ == "__main__":
    main()
