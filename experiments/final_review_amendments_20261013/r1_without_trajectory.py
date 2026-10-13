"""I091 check: rule R1 (prereg_rules.freeze_a1_grid) with and without trajectory in the bio score C.

Loads the inputs recorded in docs/prereg/r1_record.json (sha256-checked; a file no longer at its recorded path is found by
file name and sha256 under --search). Round 1 = all recorded inputs (A1 + R1 extension fits); round 0 = the A1 manifest
and the A1 stage's scores only. Stops unless the as-registered round 1 reproduces the record's statuses and X1 grids, then
re-runs both rounds with trajectory removed (prereg_rules.TRAJ_TASKS emptied). Round 1 without trajectory is not a true
counterfactual: its extension fits were the ones the with-trajectory round 0 asked for.
Writes per round and family: status, X1 grid, extension request, and sigma_c (R1's bio noise SD) of the family's immune
blocks, recovered as del_c * sqrt(n) / K_NOISE where del_c is above its floor DELTA_MIN.
Usage: python r1_without_trajectory.py OUT.csv --search DIR [DIR ...]
"""
import argparse
import csv
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import prereg_rules as P  # noqa: E402


def resolve(entry, search):
    path = entry["path"] if os.path.isabs(entry["path"]) else os.path.join(REPO, entry["path"])
    if os.path.exists(path) and P.sha256(path) == entry["sha256"]:
        return path
    name = os.path.basename(path)
    for d in search:
        for root, _, files in os.walk(d):
            if name in files and P.sha256(os.path.join(root, name)) == entry["sha256"]:
                return os.path.join(root, name)
    sys.exit(f"no file named {name} with sha256 {entry['sha256']} under {search}")


def sigma_c(block):
    p = block["points"][0]
    return round(p["del_c"] * math.sqrt(p["n"]) / P.K_NOISE, 4) if p["del_c"] > P.DELTA_MIN + 1e-12 else f"<={P.DELTA_MIN}"


def immune(fam_result):
    return ";".join(f"{k.split('|')[1]}:{sigma_c(b)}" for k, b in sorted(fam_result["blocks"].items()) if k.startswith("immune|"))


def run(manifests, scores, failures):
    M, _ = P.read_manifest(manifests)
    S, _ = P.read_scores(scores)
    return P.freeze_a1_grid(P.outcomes(M, S, P.read_failures(failures), ["A1"]))["families"]


ap = argparse.ArgumentParser()
ap.add_argument("out_csv")
ap.add_argument("--search", nargs="+", default=[])
a = ap.parse_args()
rec = json.load(open(os.path.join(REPO, "docs", "prereg", "r1_record.json")))
inp = {k: [resolve(e, a.search) for e in v] for k, v in rec["inputs"].items()}
a1_scores = [s for s, e in zip(inp["scores"], rec["inputs"]["scores"]) if "/A1/scores/" in e["path"]]
rounds = {"round1_A1+R1ext": (inp["manifest"], inp["scores"]), "round0_A1only": (inp["manifest"][:1], a1_scores)}
res = {}
for traj in (True, False):
    if not traj:
        P.TRAJ_TASKS = ()          # bio_metrics_for reads the module global at call time
    for rnd, (man, sc) in rounds.items():
        res[traj, rnd] = run(man, sc, inp["failures"][0])
for fam, f in rec["families"].items():
    got = res[True, "round1_A1+R1ext"][fam]
    P._require(got["status"] == f["status"] and got.get("x1_grid") == f.get("x1_grid"), f"{fam}: re-run differs from the record")
print(f"as-registered round 1 reproduces the record ({len(rec['families'])} families); round 0 used {len(a1_scores)} A1 scores")
rows = []
for rnd in rounds:
    for fam in sorted(rec["families"]):
        w, n = res[True, rnd][fam], res[False, rnd][fam]
        key = lambda r: (r["status"], r.get("x1_grid"), r.get("extend"))     # no x1_grid while a family extends
        rows.append(dict(round=rnd, family=fam, status_with=w["status"], x1_grid_with=w.get("x1_grid"), extend_with=w.get("extend"),
                         status_without=n["status"], x1_grid_without=n.get("x1_grid"), extend_without=n.get("extend"),
                         changed=key(w) != key(n),
                         sigma_c_immune_with=immune(w), sigma_c_immune_without=immune(n)))
with open(a.out_csv, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
for r in rows:
    if r["changed"] or r["round"].startswith("round0"):
        print({k: r[k] for k in ("round", "family", "status_with", "extend_with", "status_without", "extend_without",
                                  "sigma_c_immune_with", "sigma_c_immune_without")})
