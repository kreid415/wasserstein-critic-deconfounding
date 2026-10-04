#!/usr/bin/env python3
"""Is a conda env complete? Every file listed by conda (conda-meta/*.json "files") and by pip (site-packages
*.dist-info/RECORD, compiled .pyc excluded) must exist. Standard library only, so any python can run it.

Why: /fastscratch is purged by file modification time (files older than 30 days are deleted), and conda unpacks
package files with the dates stored in the package, often years old. On 2026-10-03 the purge deleted most of the
standard library of wcd-fit and wcd-score (built 2026-10-02) while their .verified markers, written at build time,
survived; the next job died at its first python call. cluster/jhpce/build_envs.sh therefore reuses an env only if
this check passes, and re-dates every file after a build.

Layering leaves benign gaps that are not damage: a pip dist installed over a conda package leaves conda-meta entries
for files pip removed, and some RECORDs list files outside site-packages that were never written (seen locally:
scvi-api 2 such entries, wcd-kbet 1,569). So the build records the gaps of the freshly built and fully verified env
(--write-baseline -> ENV/.intact_baseline.json), and an env is complete when nothing beyond its baseline is missing.
Without a baseline every gap counts, which can only cause an unnecessary rebuild (the safe direction).

Usage: env_intact.py ENV_PREFIX [--write-baseline] [--max-show N]
       exit 0 = complete, 1 = files missing beyond the baseline, 2 = not a conda env
"""
import argparse
import glob
import json
import os
import sys


def missing_files(prefix):
    meta = glob.glob(os.path.join(prefix, "conda-meta", "*.json"))
    if not meta:
        return None, 0
    miss, n = [], 0
    for f in meta:
        with open(f) as fh:
            rec = json.load(fh)
        for rel in rec.get("files", []):
            n += 1
            if not os.path.lexists(os.path.join(prefix, rel)):
                miss.append(rel)
    for record in glob.glob(os.path.join(prefix, "lib", "python3*", "site-packages", "*.dist-info", "RECORD")):
        site = os.path.dirname(os.path.dirname(record))
        with open(record) as fh:
            for line in fh:
                rel = line.rstrip("\n").rsplit(",", 2)[0]
                if not rel or rel.endswith(".pyc"):
                    continue
                n += 1
                path = os.path.normpath(os.path.join(site, rel))
                if not os.path.lexists(path):
                    miss.append(os.path.relpath(path, prefix))
    return sorted(set(miss)), n


BASELINE = ".intact_baseline.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--write-baseline", action="store_true",
                    help="record the current gaps as benign (run once, right after a verified build)")
    ap.add_argument("--max-show", type=int, default=5)
    a = ap.parse_args(argv)
    miss, n = missing_files(a.prefix)
    if miss is None:
        print(f"{a.prefix}: no conda-meta/*.json, not a conda env")
        return 2
    bpath = os.path.join(a.prefix, BASELINE)
    if a.write_baseline:
        with open(bpath, "w") as fh:
            json.dump(dict(recorded_files=n, missing=miss), fh, indent=0)
        print(f"{a.prefix}: baseline written, {n} recorded files, {len(miss)} benign gaps")
        return 0
    base = set()
    if os.path.exists(bpath):
        with open(bpath) as fh:
            base = set(json.load(fh)["missing"])
    new = [m for m in miss if m not in base]
    print(f"{a.prefix}: {n} recorded files, {len(new)} missing beyond the baseline ({len(base)} benign gaps"
          + ("" if os.path.exists(bpath) else ", no baseline file") + ")"
          + (f", e.g. {new[:a.max_show]}" if new else ""))
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
