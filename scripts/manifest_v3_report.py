#!/usr/bin/env python
"""Manifest v3 report (amendment batch 2026-10-04: SI-44 X15 KL warm-up sensitivity, SI-46 Scanorama knn 2-80).

Compares manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv with v2 (and v2 with the tagged manifest) line by line and
writes, under --out-dir:
  manifest_v3_diff.csv          every row only in v2 (removed) or only in v3 (added): change, tag, experiment, task, arm,
                                lam, seed, extra
  manifest_v3_counts.csv        rows per experiment in the tagged manifest, v2 and v3
  manifest_v3_cost.csv          scripts/cost_model.py per experiment, v2 vs v3: fits, GPU lane-hours (stock-backbone
                                throughput, docs/throughput_rtx3080_stock_backbone.csv) and scoring CPU-hours
                                (docs/scoring_time.csv)
  x15_stock_equal_followups.csv X15 rows with kl_warmup 'stock' whose settings equal another follow-up row (stock = the
                                scvi-tools default, so {"kl_warmup": "stock"} is compared as {}); reported, not removed
and prints a JSON summary (SHA-256 of the three manifests, line checks, counts, totals). Exit 1 if a line check fails.

Usage (repo root): python scripts/manifest_v3_report.py [--out-dir docs]
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_paper_manifest as BPM  # noqa: E402
import cost_model as CM  # noqa: E402

MAN = os.path.join(ROOT, "manifests")
TAGGED = os.path.join(MAN, "paper_manifest_stock_pilot_u5_b10.tsv")
V2 = os.path.join(MAN, "paper_manifest_stock_pilot_u5_b10_v2.tsv")
V3 = os.path.join(MAN, "paper_manifest_stock_pilot_u5_b10_v3.tsv")
FOLLOWUPS = ("X3", "X6", "X7", "X8", "X12", "X15")      # = prereg_rules.FOLLOWUPS (checked in tests/prereg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lines(path):
    with open(path) as f:
        return f.read().splitlines()


def fields(line):
    return dict(zip(BPM.COLS, line.split("\t")))


def expected_change(f):
    """The rows the amendment adds or removes: X13 Scanorama knn 160 (removed) / knn 2 (added) at the backbone's
    10 dimensions (SI-46), and X15 (SI-44)."""
    if f["experiment"] == "X15":
        return "added"
    if f["experiment"] == "X13" and f["arm"] == "scanorama" and f["n_latent"] == "10":
        return {"160": "removed", "2": "added"}.get(f["lam"])
    return None


def v3_vs_v2_violations(old, new):
    """Differences of v3 from v2 other than the signed-off ones: header line (only the row count may change), column
    line, rows removed / added other than expected_change, and the relative order of the common rows."""
    bad = []
    hdr = lambda s: s.rsplit("(", 1)[0]  # noqa: E731  '... omitted (0 of N).' -> the text before the count
    if hdr(old[0]) != hdr(new[0]):
        bad.append("header line differs beyond the row count")
    if old[1] != new[1]:
        bad.append("column line differs")
    so, sn = set(old[2:]), set(new[2:])
    removed = [x for x in old[2:] if x not in sn]
    added = [x for x in new[2:] if x not in so]
    bad += [f"unexpected removed row {fields(x)['tag']}" for x in removed if expected_change(fields(x)) != "removed"][:5]
    bad += [f"unexpected added row {fields(x)['tag']}" for x in added if expected_change(fields(x)) != "added"][:5]
    if [x for x in old[2:] if x in sn] != [x for x in new[2:] if x in so]:
        bad.append("common rows are not in the same relative order")
    return bad, removed, added


def non_x3_violations(tagged, v2):
    """Every non-X3 row of v2 byte-identical, in order, to the tagged manifest (plus header and column line)."""
    bad = [] if tagged[:2] == v2[:2] else ["header or column line differs"]
    a = [x for x in tagged[2:] if fields(x)["experiment"] != "X3"]
    b = [x for x in v2[2:] if fields(x)["experiment"] != "X3"]
    if a != b:
        bad.append(f"non-X3 rows differ ({len(a)} vs {len(b)} rows)")
    return bad


def _settings(r, drop_stock):
    ex = json.loads(r["extra"])
    if drop_stock and ex.get("kl_warmup") == "stock":
        ex.pop("kl_warmup")
    return tuple((c, r[c]) for c in BPM.COLS if c not in ("tag", "experiment", "extra")) + \
        (("extra", json.dumps(ex, sort_keys=True)),)


def x15_stock_equal_followups(M):
    """X15 'stock' rows whose settings (every column but tag and experiment; extra parsed, kl_warmup 'stock' dropped)
    equal a row of another follow-up experiment. A 'matched*' lambda resolves from the same R4-x1 cell for equal
    task / cond / base arm, so equal placeholders mean equal lambdas after resolution."""
    other = {}
    for r in M[M.experiment.isin([e for e in FOLLOWUPS if e != "X15"])].to_dict("records"):
        other.setdefault(_settings(r, False), []).append((r["experiment"], r["tag"]))
    out = []
    for r in M[M.experiment == "X15"].to_dict("records"):
        if json.loads(r["extra"]).get("kl_warmup") != "stock":
            continue
        for exp, tag in other.get(_settings(r, True), []):
            out.append(dict(x15_tag=r["tag"], task=r["task"], arm=r["arm"], lam=r["lam"], n_critic=r["n_critic"],
                            seed=r["seed"], equal_experiment=exp, equal_tag=tag))
    return pd.DataFrame(out, columns=["x15_tag", "task", "arm", "lam", "n_critic", "seed", "equal_experiment", "equal_tag"])


def cost_by_experiment(manifest):
    T = pd.read_csv(os.path.join(ROOT, "docs", "throughput_rtx3080_stock_backbone.csv"))
    S = pd.read_csv(os.path.join(ROOT, "docs", "scoring_time.csv"))
    rate = float(S.seconds.sum() / S.n_cells.sum())
    M = CM.cost(manifest, T, None, rate)
    return M.groupby("experiment").agg(fits=("tag", "size"), gpu_lane_hours=("lane_hours", "sum"),
                                       score_cpu_hours=("score_cpu_hours", "sum"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "docs"))
    a = ap.parse_args()
    t, o, n = lines(TAGGED), lines(V2), lines(V3)
    bad, removed, added = v3_vs_v2_violations(o, n)
    bad += [f"v2 vs tagged: {x}" for x in non_x3_violations(t, o)]
    rd = lambda p: pd.read_csv(p, sep="\t", comment="#", dtype=str, keep_default_na=False)  # noqa: E731
    Mt, M2, M3 = rd(TAGGED), rd(V2), rd(V3)
    diff = pd.DataFrame([dict(change=ch, **{k: fields(x)[k] for k in ("tag", "experiment", "task", "arm", "lam", "seed",
                                                                         "n_latent", "extra")})
                         for ch, rows in (("removed", removed), ("added", added)) for x in rows])
    counts = pd.DataFrame({"tagged": Mt.experiment.value_counts(), "v2": M2.experiment.value_counts(),
                           "v3": M3.experiment.value_counts()}).fillna(0).astype(int)
    counts = counts.loc[[e for e in pd.unique(pd.concat([M3.experiment, M2.experiment, Mt.experiment]))]]
    counts.loc["TOTAL"] = counts.sum()
    c2, c3 = cost_by_experiment(V2), cost_by_experiment(V3)
    cost = c2.add_suffix("_v2").join(c3.add_suffix("_v3"), how="outer").fillna(0)
    cost.loc["TOTAL"] = cost.sum()
    for k in ("gpu_lane_hours", "score_cpu_hours"):
        cost[f"{k}_change"] = cost[f"{k}_v3"] - cost[f"{k}_v2"]
    eq = x15_stock_equal_followups(M3)
    os.makedirs(a.out_dir, exist_ok=True)
    diff.to_csv(os.path.join(a.out_dir, "manifest_v3_diff.csv"), index=False)
    counts.rename_axis("experiment").to_csv(os.path.join(a.out_dir, "manifest_v3_counts.csv"))
    cost.rename_axis("experiment").round(2).to_csv(os.path.join(a.out_dir, "manifest_v3_cost.csv"))
    eq.to_csv(os.path.join(a.out_dir, "x15_stock_equal_followups.csv"), index=False)
    summary = dict(sha256={"tagged": sha256(TAGGED), "v2": sha256(V2), "v3": sha256(V3)},
                   rows={"tagged": len(Mt), "v2": len(M2), "v3": len(M3)},
                   removed=diff[diff.change == "removed"].groupby(["experiment", "arm", "lam"]).size().to_dict()
                   if len(diff) else {},
                   added=diff[diff.change == "added"].groupby(["experiment", "arm", "lam"]).size().to_dict()
                   if len(diff) else {},
                   common_rows=len(set(o[2:]) & set(n[2:])), x15_stock_equal_rows=len(eq),
                   total_gpu_lane_hours={"v2": round(float(cost.loc["TOTAL", "gpu_lane_hours_v2"]), 1),
                                         "v3": round(float(cost.loc["TOTAL", "gpu_lane_hours_v3"]), 1)},
                   total_score_cpu_hours={"v2": round(float(cost.loc["TOTAL", "score_cpu_hours_v2"]), 1),
                                          "v3": round(float(cost.loc["TOTAL", "score_cpu_hours_v3"]), 1)},
                   violations=bad)
    print(json.dumps({k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                      for k, v in summary.items()}, indent=1))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
