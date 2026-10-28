#!/usr/bin/env python
"""Run scripts/analysis_x1.py on an input directory made by make_synthetic_x1.py or build_pilot_inputs.py.

Builds the command line from the directory layout (manifests/*.tsv, scores/*, failures.csv, r1_record.json, r4_x1_record.json,
latents/ or the directories listed in pilot_meta.json). A pilot directory (pilot_meta.json present) has three tasks and seeds 0-2,
while analysis_x1.py and prereg_rules.py fix eight tasks and seeds 0-4: unless --strict is given, this driver sets those constants
in memory (no file is changed) so the run can get past that first refusal. --strict runs the script as it is.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    i = a.inputs
    meta_path = os.path.join(i, "pilot_meta.json")
    latents = [os.path.join(i, "latents")]
    if os.path.exists(meta_path):  # pragma: no cover  # pilot inputs exist only on the workstation
        meta = json.load(open(meta_path))
        latents = meta["latent_dirs"]
        if not a.strict:
            tasks = tuple(meta["pilot_tasks"])
            P.BPM.TASKS = {t: P.BPM.TASKS[t] for t in tasks}
            P.REAL_TASKS = tuple(t for t in P.REAL_TASKS if t in tasks)
            A.X1_SEEDS = [0, 1, 2]
    args = [x for m in sorted(glob.glob(os.path.join(i, "manifests", "*.tsv"))) for x in ("--manifest", m)]
    args += ["--scores", *sorted(glob.glob(os.path.join(i, "scores", "*")))]
    args += ["--failures", os.path.join(i, "failures.csv"), "--r1-record", os.path.join(i, "r1_record.json"),
             "--r4-record", os.path.join(i, "r4_x1_record.json"), "--latents", *latents, "--out-json", a.out_json]
    return A.main(args)


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
