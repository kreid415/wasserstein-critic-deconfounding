#!/usr/bin/env python
"""R2 and R3 (docs/PREREG.md section 3): decide X1's adversary input (A2: posterior mean vs sample) and
per-dimension standardisation (A3: off vs on) from the A2/A3 pilots, and resolve the 'A2' / 'A3'
placeholders of X1. Follow-up adversarial rows (built with mean / 0) are set to the same decision.

Defaults (user sign-off 2026-10-02, SI-19 / SI-20): posterior mean, standardisation off. The alternative
replaces the default only if >= 2/3 of the cells are evaluable (>= 1 per task), the mean Dbio@b* > ROPE,
the mean is > 0 within each task, and the alternative has no more failed fits. A2 only: if posterior
samples miss b* in any cell where posterior means reach it, X1 uses means (masking).

Usage (repo root):
  python scripts/decide_a2_a3.py --manifest scripts/paper_manifest.r4a1.tsv --scores <dir> --failures <csv> \
      --r4a1-record docs/prereg/r4_a1_record.json --out-json docs/prereg/r2r3_record.json \
      --out-manifest scripts/paper_manifest.x1.tsv
Exit status: 0 = decided; anything else = error.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_rules as P  # noqa: E402
from prereg_rules import BPM  # noqa: E402

SETTINGS = {"A2": dict(column="adv_input", default=("mean", "mean"), alt=("sample", "sample"), masking="sample"),
            "A3": dict(column="zstd", default=("off", "0"), alt=("on", "1"), masking=None)}


def cells_for(out, exp, r4a1):
    """One cell per (task, arm) of the pilot: default half from A1 fits, alternative half from exp fits."""
    st = SETTINGS[exp]
    pil = out[out.experiment == exp]
    cells = []
    for (t, c, arm) in sorted(set(zip(pil.task, pil.cond_i, pil.arm))):
        cell = r4a1["cells"][f"{t}|{c}|{arm}"]
        window = [cell["lo"], cell["matched"], cell["hi"]]
        P._require(None not in window, f"{exp} {t} {arm}: window {window} is incomplete")
        got = sorted(set(pil.lamf[(pil.task == t) & (pil.cond_i == c) & (pil.arm == arm)]))
        P._require(len(got) == 3 and all(math.isclose(g, w) for g, w in zip(got, window)),
                   f"{exp} {t} {arm}: lambdas {got} differ from the R4-a1 window {window}")
        seeds = sorted(P._zero_ref(out, "A1", t, c))
        d_pts = [p for p in P._cell_points(out, "A1", t, c, arm, seeds, adv_input="mean", zstd="0")
                 if any(math.isclose(p["lam"], w) for w in window)]
        P._require(len(d_pts) == 3, f"A1 {t} {arm}: default half lacks window points {window}")
        a_pts = P._cell_points(out, exp, t, c, arm, seeds, **{st["column"]: st["alt"][1]})
        bstar = r4a1["targets"][f"{t}|{c}"]["bstar"]
        rec = dict(task=t, cond=c, arm=arm, window=window, bstar=bstar)
        for name, pts in ((st["default"][0], d_pts), (st["alt"][0], a_pts)):
            rec[name] = dict(P.bio_at_bstar(P.window_curve(pts), bstar), n_fail=sum(p["n_fail"] for p in pts),
                             curve=P.window_curve(pts))
        cells.append(rec)
    P._require(cells, f"no {exp} cells")
    return cells


def apply_decisions(M, choice):
    """X1 placeholders -> decision; follow-up adversarial rows (mean / 0 by construction) -> decision."""
    M = M.copy()
    adv, z = choice["A2"], choice["A3"]
    x1 = (M.experiment == "X1") & M.arm.isin(P.ADV_ARMS)
    P._require((M.adv_input[x1] == BPM.ADV_INPUT_PENDING).all() and (M.zstd[x1] == BPM.ZSTD_PENDING).all(),
               "X1 adversarial rows do not all hold the A2 / A3 placeholders")
    stray = M[~x1 & (M.adv_input.eq(BPM.ADV_INPUT_PENDING) | M.zstd.eq(BPM.ZSTD_PENDING))]
    P._require(stray.empty, f"placeholders outside X1's adversarial rows: {stray.tag.tolist()[:5]}")
    fu = M.experiment.isin(P.FOLLOWUPS) & ~M.arm.isin(P.NO_ADVERSARY)
    odd = M[fu & ~((M.adv_input == "mean") & (M.zstd == "0"))]
    P._require(odd.empty, f"follow-up rows with a deliberate adv_input / zstd: {odd.tag.tolist()[:5]}")
    M.loc[x1 | fu, "adv_input"] = adv
    M.loc[x1 | fu, "zstd"] = z
    return M, int(x1.sum()), int(fu.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--r4a1-record", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-manifest", required=True)
    a = ap.parse_args(argv)
    M, header = P.read_manifest(a.manifest)
    P.check_seed_separation(M)
    r4a1 = P.read_record(a.r4a1_record, "R4", stage="a1")
    S, score_files = P.read_scores(a.scores)
    F = P.read_failures(a.failures)
    out = P.outcomes(M, S, F, ["A1", "A2", "A3"])
    dec, choice = {}, {}
    for exp, rule in (("A2", "R2"), ("A3", "R3")):
        st = SETTINGS[exp]
        cells = cells_for(out, exp, r4a1)
        d = P.decide_binary(cells, st["default"][0], st["alt"][0], masking_setting=st["masking"],
                            safe_setting=(st["default"][0] if st["masking"] else None))
        dec[rule] = dict(d, cells=cells)
        choice[exp] = dict([st["default"], st["alt"]])[d["choice"]]
    M2, n_x1, n_fu = apply_decisions(M, choice)
    P.write_manifest(M2, header, a.out_manifest, f"R2/R3 decided: adv_input={choice['A2']} zstd={choice['A3']} on "
                                                  f"{n_x1} X1 + {n_fu} follow-up rows (decide_a2_a3.py)")
    P.write_record(a.out_json, dict(rule="R2R3", status="frozen", decisions=dec, adv_input=choice["A2"],
                                    zstd=choice["A3"], x1_rows=n_x1, followup_rows=n_fu, out_manifest=a.out_manifest),
                   dict(manifest=a.manifest, scores=score_files, failures=[a.failures], r4a1_record=[a.r4a1_record]))
    for rule in ("R2", "R3"):
        d = dec[rule]
        print(f"{rule}: {d['choice']} ({d['reason']}; evaluable {d['n_evaluable']}/{d['n_cells']}, mean D {d['mean_D']:.4f})")
    return P.EXIT_FROZEN


if __name__ == "__main__":
    sys.exit(main())
