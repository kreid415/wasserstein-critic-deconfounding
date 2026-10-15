"""How much does the P1 reading depend on the lambda grid? (limitation 2 of docs/ANALYSIS_SPEC.md section 16; finding DOC-R04)

Pilot A1 (3 tasks x 2 decoder conditions = the six cells; seeds 100-102; the five non-Sinkhorn adversarial arms; 10 lambda) read with
scripts/analysis_x1.py `read_p1` (seed-mean curves, b* of docs/prereg/r4_a1_record.json) twice: on every pilot lambda, and on only those
pilot lambda that belong to the registered X1 grid of the arm's family (docs/prereg/r1_record.json). The reading is the reference
critic's biology at b* minus the discriminator's; `d_shift` is the registered-grid difference minus the all-lambda difference.
Descriptive only: 6 cells, 3 seeds, no standard error.

The first independent review made this table with its own reading code (interp_grid_pilot.py, my_ref.py) from a table of outcomes built
interactively (no builder was kept); this script builds the outcomes from the pilot files with `prereg_rules.outcomes`. The Sinkhorn
rows are left out of the manifest because `outcomes` refuses the Sinkhorn files made before the Sinkhorn fix (commit c88cce3).

Run (conda env scvi-api; a few seconds):
    python experiments/analysis_numerics_20261014/first_review/pilot_grid_reading.py --pilot-dir <path>/tier12_v2
(<pilot-dir>/A1 holds scores/ and failures.csv.) Output: pilot_grid_reading.csv.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402
import run_stage as RS  # noqa: E402

ARMS = (("reference", "w1"), ("discriminator", "js"))


def pilot_outcomes(pilot_dir):
    M = RS.load_manifest(os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v3.tsv"))
    M = M[(M.experiment == "A1") & (M.arm != "sinkhorn")]
    S, _ = P.read_scores(os.path.join(pilot_dir, "A1", "scores"))
    F = P.read_failures(os.path.join(pilot_dir, "A1", "failures.csv"))
    return P.outcomes(M, S, F, ["A1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "pilot_grid_reading.csv"))
    a = ap.parse_args()
    out = pilot_outcomes(a.pilot_dir)
    r1 = json.load(open(os.path.join(ROOT, "docs", "prereg", "r1_record.json")))
    r4 = json.load(open(os.path.join(ROOT, "docs", "prereg", "r4_a1_record.json")))
    rows = []
    for (task, cond), cell in out[out.arm != "none"].groupby(["task", "cond_i"]):
        zero = P._zero_ref(out, "A1", task, cond)
        seeds = sorted(zero)
        bstar = r4["targets"][f"{task}|{cond}"]["bstar"]
        row = dict(task=task, cond=int(cond), bstar=bstar)
        read = {}
        for arm, family in ARMS:
            points = P._cell_points(out, "A1", task, cond, arm, seeds)
            grid = r1["families"][f"{family}|{cond}"]["x1_grid"]
            read[arm] = dict(dense=points, registered=[p for p in points if any(abs(p["lam"] - g) < 1e-9 for g in grid)])
            row[f"{arm}_points_all"], row[f"{arm}_points_registered"] = len(points), len(read[arm]["registered"])
        for which in ("dense", "registered"):
            r = A.read_p1(zero, read["reference"][which], read["discriminator"][which], bstar)
            row[f"d_{which}"] = r["d"]
            for arm, key in (("reference", "ref"), ("discriminator", "dis")):
                row[f"{arm}_bio_{which}"] = r["bio"][key]
        row["d_shift"] = None if None in (row["d_dense"], row["d_registered"]) else row["d_registered"] - row["d_dense"]
        rows.append(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        shift = "" if r["d_shift"] is None else f"{r['d_shift']:+.4f}"
        print(f"{r['task']}|{r['cond']}: points all/registered reference {r['reference_points_all']}/{r['reference_points_registered']}, "
              f"discriminator {r['discriminator_points_all']}/{r['discriminator_points_registered']}; d all-lambda {r['d_dense']} "
              f"registered {r['d_registered']} shift {shift}")


if __name__ == "__main__":
    main()
