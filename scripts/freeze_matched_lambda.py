#!/usr/bin/env python
"""R4 (docs/PREREG.md section 4): matched lambda (lo / matched / hi) per task x arm x decoder.

  --stage a1  on A1 scores: resolves the A2/A3 window (a1_matched_lo / a1_matched / a1_matched_hi).
  --stage x1  on X1 scores: resolves the follow-ups' matched_lo / matched / matched_hi (variant arms take
              their base arm's value, prereg_rules.BASE_ARM) and drops follow-up rows that are then
              identical to an X1 row (the analysis reads the X1 fit; recorded in 'reuses_x1').

b* = b0 + 0.5 (b_common - b0) on the unscaled batch score (mean of the 5 raw scIB batch metrics); the
matched point is the failure-free grid point closest to b* (ties -> smaller lambda). A matched point on
a grid edge needs one more grid point beyond it (exit 3, extension manifest), up to 2 per edge.

Usage (repo root):
  python scripts/freeze_matched_lambda.py --stage a1 --manifest scripts/paper_manifest.r1.tsv \
      --scores <dir> --failures <csv> --r1-record docs/prereg/r1_record.json \
      --out-json docs/prereg/r4_a1_record.json --out-manifest scripts/paper_manifest.r4a1.tsv \
      --extension-manifest scripts/a1_extension_r4_<round>.tsv
Exit status: 0 = frozen; 3 = extension needed; anything else = error.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_rules as P  # noqa: E402
from prereg_rules import BPM  # noqa: E402

X1_MATCHED = ["matched_lo", "matched", "matched_hi"]


def _value(cell, label):
    return cell[{"lo": "lo", "": "matched", "hi": "hi"}[label.rsplit("matched", 1)[1].lstrip("_")]]


def resolve(M, r4, stage):
    """Replace the stage's placeholders; any placeholder left for this stage is an error."""
    M = M.copy()
    labels, exps = (BPM.A1_MATCHED, ["A2", "A3"]) if stage == "a1" else (X1_MATCHED, list(P.FOLLOWUPS))
    hit = M.lam.isin(labels)
    stray = M[hit & ~M.experiment.isin(exps)]
    P._require(stray.empty, f"{stage}: placeholder in unexpected experiments: {stray.tag.tolist()[:5]}")
    n = 0
    for i in M.index[hit]:
        arm = M.at[i, "arm"]
        P._require(arm in P.BASE_ARM, f"{M.at[i, 'tag']}: arm {arm!r} has no base arm in prereg_rules.BASE_ARM")
        base = arm if stage == "a1" else P.BASE_ARM[arm]
        if stage == "a1":
            P._require(arm in P.ADV_ARMS, f"{M.at[i, 'tag']}: A2/A3 arm {arm!r} is not an A1 arm")
        key = f"{M.at[i, 'task']}|{int(M.at[i, 'cond'])}|{base}"
        P._require(key in r4["cells"], f"{M.at[i, 'tag']}: no matched cell {key}")
        v = _value(r4["cells"][key], M.at[i, "lam"])
        P._require(v is not None, f"{M.at[i, 'tag']}: {key} has no {M.at[i, 'lam']}")
        M.at[i, "lam"] = P.fmt_lam(v)
        n += 1
    P._require(not M.lam.isin(labels).any(), f"{stage}: unresolved placeholders remain")
    return M, n


def needed_cells(M, stage):
    """'task|cond|base arm' of every row that will take a value at this stage."""
    labels = BPM.A1_MATCHED if stage == "a1" else X1_MATCHED
    rows = M[M.lam.isin(labels)]
    P._require(len(rows), f"{stage}: no rows hold {labels}")
    bad = sorted(set(rows.arm) - set(P.BASE_ARM))
    P._require(not bad, f"arms {bad} have no base arm in prereg_rules.BASE_ARM")
    return {f"{t}|{int(c)}|{P.BASE_ARM[a]}" for t, c, a in zip(rows.task, rows.cond, rows.arm)}


def dedupe_against_x1(M):
    """Follow-up rows whose settings equal an X1 row's (seed included) are not refitted."""
    x1 = {P.settings_key(r): r["tag"] for r in M[M.experiment == "X1"].to_dict("records")}
    reuse, keep = {}, []
    for r in M.to_dict("records"):
        k = P.settings_key(r)
        if r["experiment"] in P.FOLLOWUPS and k in x1:
            reuse[r["tag"]] = x1[k]
        else:
            keep.append(r["tag"])
    return M[M.tag.isin(set(keep))].copy(), reuse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["a1", "x1"])
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--r1-record", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--extension-manifest", required=True)
    a = ap.parse_args(argv)
    M, header = P.read_manifest(a.manifest)
    P.check_seed_separation(M)
    r1 = P.read_record(a.r1_record, "R1")
    S, score_files = P.read_scores(a.scores)
    F = P.read_failures(a.failures)
    exp = "A1" if a.stage == "a1" else "X1"
    if a.stage == "x1":
        x1 = M[(M.experiment == "X1") & M.arm.isin(P.ADV_ARMS)]
        P._require(x1.adv_input.isin(["mean", "sample"]).all() and x1.zstd.isin(["0", "1"]).all(),
                   "X1 adv_input / zstd placeholders are unresolved: run decide_a2_a3.py first")
    out = P.outcomes(M, S, F, [exp])
    r4 = P.r4_stage(out, a.stage, r1, need=needed_cells(M, a.stage))
    inputs = dict(manifest=a.manifest, scores=score_files, failures=[a.failures], r1_record=[a.r1_record])
    if r4["status"] == "extend":
        E = P.extension_rows(M, r4["extensions"])
        P.write_manifest(E, header[:1], a.extension_manifest, f"R4 {a.stage} extension, {len(E)} {exp} rows "
                                                              f"(freeze_matched_lambda.py)")
        P.write_record(a.out_json, dict(r4, extension_manifest=a.extension_manifest, n_extension_rows=len(E)), inputs)
        print(f"R4 {a.stage} extend: {len(E)} {exp} rows -> {a.extension_manifest}")
        return P.EXIT_EXTEND
    M2, n = resolve(M, r4, a.stage)
    reuse = {}
    if a.stage == "x1":
        M2, reuse = dedupe_against_x1(M2)
    P.write_manifest(M2, header, a.out_manifest, f"R4 {a.stage} frozen: {n} rows resolved, {len(reuse)} follow-up rows "
                                                  f"reuse X1 fits (freeze_matched_lambda.py, record {os.path.basename(a.out_json)})")
    P.write_record(a.out_json, dict(r4, resolved_rows=n, reuses_x1=reuse, out_manifest=a.out_manifest), inputs)
    flagged = sum(1 for c in r4["cells"].values() if c["flags"])
    print(f"R4 {a.stage} frozen: {len(r4['cells'])} cells ({flagged} flagged), {n} rows resolved, {len(reuse)} reuse X1")
    return P.EXIT_FROZEN


if __name__ == "__main__":
    sys.exit(main())
