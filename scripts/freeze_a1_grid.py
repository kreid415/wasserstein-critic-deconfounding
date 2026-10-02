#!/usr/bin/env python
"""R1 (docs/PREREG.md section 2): fix the 6-point X1 lambda grid per arm family x decoder from the A1
pilot, and report the power of the fixed design (section 5). The design is never changed here.

Usage (repo root, any env with pandas + scipy):
  python scripts/freeze_a1_grid.py --manifest scripts/paper_manifest.tsv [--manifest <A1 extension>.tsv ...] \
      --scores <score dir or csv> [...] --failures <failures.csv> \
      --out-json docs/prereg/r1_record.json --out-manifest scripts/paper_manifest.r1.tsv \
      --extension-manifest scripts/a1_extension_r1_<round>.tsv
Exit status: 0 = frozen (record + manifest with X1 lam g1..g6 resolved); 3 = A1 extension needed (extension
manifest + an 'extend' record written, nothing frozen: fit and score it, then rerun with it as an extra
--manifest); anything else = error.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_rules as P  # noqa: E402
from prereg_rules import BPM  # noqa: E402


def resolve_x1_grid(M, r1):
    """X1 rows with lam g1..g6 take their family x decoder grid; no other row may hold a g-label."""
    M = M.copy()
    hit = M.lam.isin(BPM.FAMILY_GRID)
    bad = M[hit & ~((M.experiment == "X1") & M.arm.isin(P.ADV_ARMS))]
    P._require(bad.empty, f"rows outside X1's adversarial arms hold a grid label: {bad.tag.tolist()[:5]}")
    for i in M.index[hit]:
        fam = r1["families"][f"{P.FAMILY_OF[M.at[i, 'arm']]}|{int(M.at[i, 'cond'])}"]
        M.at[i, "lam"] = P.fmt_lam(fam["x1_grid"][BPM.FAMILY_GRID.index(M.at[i, "lam"])])
    P._require(not M.lam.isin(BPM.FAMILY_GRID).any(), "unresolved grid labels remain")
    return M, int(hit.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", action="append", required=True, help="builder manifest, then any A1 extension manifests")
    ap.add_argument("--scores", nargs="+", required=True, help="score_scib_native.py CSVs or directories of them")
    ap.add_argument("--failures", required=True, help="tag,status,detail (header-only file if there are none)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--extension-manifest", required=True)
    a = ap.parse_args(argv)
    M, header = P.read_manifest(a.manifest)
    P.check_seed_separation(M)
    S, score_files = P.read_scores(a.scores)
    F = P.read_failures(a.failures)
    out = P.outcomes(M, S, F, ["A1"])
    r1 = P.freeze_a1_grid(out)
    inputs = dict(manifest=a.manifest, scores=score_files, failures=[a.failures])
    if r1["status"] == "extend":
        tasks = sorted(set(out.task[out.experiment == "A1"]))
        specs = [dict(experiment="A1", task=t, cond=fam["cond"], arm=arm, lam=lam)
                 for fam in r1["families"].values() for lams in fam["extend"].values()
                 for t in tasks for arm in fam["arms"] for lam in lams]
        E = P.extension_rows(M, specs)
        P.write_manifest(E, header[:1], a.extension_manifest, f"R1 A1 extension, {len(E)} rows (freeze_a1_grid.py)")
        P.write_record(a.out_json, dict(r1, extension_manifest=a.extension_manifest, n_extension_rows=len(E)), inputs)
        print(f"R1 extend: {len(E)} A1 rows -> {a.extension_manifest}")
        return P.EXIT_EXTEND
    power = P.power_report(out, P.r4_stage(out, "a1", r1, need=set()))   # estimate only: no extensions
    M2, n = resolve_x1_grid(M, r1)
    P.write_manifest(M2, header, a.out_manifest, f"R1 frozen: {n} X1 rows took their family x decoder grid "
                                                  f"(freeze_a1_grid.py, record {os.path.basename(a.out_json)})")
    rec = P.write_record(a.out_json, dict(r1, power=power, resolved_rows=n, out_manifest=a.out_manifest), inputs)
    for k, fam in sorted(rec["families"].items()):
        print(f"R1 {k}: x1_grid {[P.fmt_lam(v) for v in fam['x1_grid']]} flags {len(fam['flags'])}")
    print(f"power: sigma {power['sigma']:.4f} tau {power['tau']:.4f} mde80(D=6) "
          f"{power['designs']['real_tasks']['mde80']:.4f} underpowered={power['underpowered']}")
    return P.EXIT_FROZEN


if __name__ == "__main__":
    sys.exit(main())
