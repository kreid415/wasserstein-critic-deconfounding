#!/usr/bin/env python
"""Run the real rule scripts (R1 freeze_a1_grid.py, R4 stage a1 freeze_matched_lambda.py) on SYNTHETIC scores of an
8-dataset A1 pilot: the 3 old datasets plus the 5 new ones, every dataset with the R1 round-1 grid-edge rows.
No fits, no real scores. Checks what docs: PILOT5_BUILD.md section 'Rule check' states:
  A. 8 datasets, every arm responding inside the grid: R1 freezes, rounds used per family x decoder are read back,
     R4 stage a1 freezes and yields one matched cell per dataset x decoder x arm;
  B. one new dataset (pancreas) whose discriminator / reference arm responds only above the grid's top: R1 asks for
     extension fits on ALL 8 datasets (every arm of the family, the decoder of the edge, 3 seeds, 2 strengths);
  C. one new dataset (lung) whose discriminator responds already at the grid's bottom, unconditioned decoder (one
     low-edge round is already used): R1 asks for the second round, 0.003 and 0.001.
Usage (env with pandas + scipy): python experiments/pilot5_jhpce/rule_check_8datasets.py
"""
import json
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "prereg"))
import synth  # noqa: E402  (puts scripts/ on sys.path)
from synth import BPM, P  # noqa: E402
import freeze_a1_grid  # noqa: E402
import freeze_matched_lambda  # noqa: E402

OLD, NEW = list(BPM.A1_TASKS), ["immune_hum_mou", "atac_large", "lung", "sim2", "pancreas"]


def manifest():
    """The full pilot design (A1 base on the 3 old datasets, A2, A3, X1, ...) plus the grid-edge rows of the old datasets
    and all A1 rows of the new ones: what the rule scripts would read for 8 datasets."""
    full = synth.pilot_rows("pilot")
    a1_8 = BPM.finalize(BPM.build(BPM.BACKBONES["stock"], "pilot", 3, 5, 8, OLD + NEW))
    add = pd.DataFrame([{c: str(r[c]) for c in BPM.COLS} for r in a1_8 if r["tag"] not in set(full.tag)], columns=BPM.COLS)
    return pd.concat([full, add], ignore_index=True)


def r1(tmp, M, world):
    man = synth.write_tsv(M, os.path.join(tmp, "manifest.tsv"))
    S, F = synth.scores(M[M.experiment == "A1"], world)
    sp, fp = synth.write_scores(S, F, os.path.join(tmp, "scores"), "a1")
    code = freeze_a1_grid.main(["--manifest", man, "--scores", sp, "--failures", fp, "--out-json", os.path.join(tmp, "r1.json"),
                                "--out-manifest", os.path.join(tmp, "m.r1.tsv"),
                                "--extension-manifest", os.path.join(tmp, "a1_ext.tsv")])
    return code, sp, fp


def extension(tmp):
    E, _ = P.read_manifest(os.path.join(tmp, "a1_ext.tsv"))
    return E


def main():
    M = manifest()
    a1 = M[M.experiment == "A1"]
    print(f"A1 rows {len(a1)} over {a1.task.nunique()} datasets, {a1.groupby('task').size().unique().tolist()} per dataset")
    with tempfile.TemporaryDirectory() as tmp:
        print("== A. 8 datasets, all arms respond inside the grid")
        code, sp, fp = r1(tmp, M, synth.world)
        rec = json.load(open(os.path.join(tmp, "r1.json")))
        print(f"R1 exit {code} status {rec['status']}; families {len(rec['families'])}")
        for k, f in sorted(rec["families"].items()):
            print(f"  {k}: rounds used {f['rounds']}, grid {f['a1_grid'][0]} .. {f['a1_grid'][-1]} ({len(f['a1_grid'])} points), "
                  f"blocks {len(f['blocks'])}, flags {len(f['flags'])}")
        code = freeze_matched_lambda.main(["--stage", "a1", "--manifest", os.path.join(tmp, "m.r1.tsv"), "--scores", sp,
                                           "--failures", fp, "--r1-record", os.path.join(tmp, "r1.json"),
                                           "--out-json", os.path.join(tmp, "r4a1.json"), "--out-manifest", os.path.join(tmp, "m.r4a1.tsv"),
                                           "--extension-manifest", os.path.join(tmp, "a1_ext_r4.tsv")])
        r4 = json.load(open(os.path.join(tmp, "r4a1.json")))
        tasks = sorted({c["task"] for c in r4["cells"].values()})
        print(f"R4 stage a1 exit {code} status {r4['status']}; {len(r4['cells'])} cells over {len(tasks)} datasets; "
              f"extensions requested {len(r4['extensions'])}")
    for title, world in (
        ("B1. pancreas, discriminator responds only above the grid's top (both decoders)",
         lambda r: synth.world(r, shift=lambda q: 3.0 if q["task"] == "pancreas" and q["arm"] == "discriminator" else 0.0)),
        ("B2. pancreas, reference critic responds only above the grid's top (both decoders)",
         lambda r: synth.world(r, shift=lambda q: 3.0 if q["task"] == "pancreas" and q["arm"] == "reference" else 0.0)),
        ("C. lung, discriminator responds already at the grid's bottom (both decoders)",
         lambda r: synth.world(r, shift=lambda q: -3.5 if q["task"] == "lung" and q["arm"] == "discriminator" else 0.0)),
    ):
        print("==", title)
        with tempfile.TemporaryDirectory() as tmp:
            code, _, _ = r1(tmp, M, world)
            rec = json.load(open(os.path.join(tmp, "r1.json")))
            print(f"R1 exit {code} status {rec['status']}; extend: " +
                  str({k: f["extend"] for k, f in sorted(rec["families"].items()) if f["extend"]}))
            if code == P.EXIT_EXTEND:
                E = extension(tmp)
                print(f"  extension rows {len(E)}: datasets {E.task.nunique()} ({sorted(E.task.unique())}), arms {sorted(E.arm.unique())}, "
                      f"cond {sorted(E.cond.unique())}, strengths {sorted(E.lam.unique(), key=float)}, seeds {sorted(E.seed.unique())}")


if __name__ == "__main__":
    main()
