"""Sensitivity of rule R4 stage a1 to the Sinkhorn refit (SI-51).

R4-a1 sets b_common = min over the six adversarial arms of M_a (an arm's highest seed-mean batch score B over its
failure-free A1 lambda points) and b* = b0 + 0.5 (b_common - b0); each arm's matched lambda is its point closest to b*,
with lo / hi its grid neighbours. A refitted Sinkhorn arm whose M falls below b_common therefore moves b*, and possibly
the matched lambda (hence the A2/A3 window) of non-Sinkhorn arms.

Re-runs prereg_rules.r4_stage on the inputs recorded in docs/prereg/r4_a1_record.json (sha256-checked; a file no longer
at its recorded path is found by file name and sha256 under --search), stops unless that reproduces the record, then for
each task|decoder block lowers M_sinkhorn in steps of 1e-5 down to b0 and reports the highest M_sinkhorn at which some
non-Sinkhorn arm's matched lambda changes ("needed" = an A2/A3 row takes its value from that cell).
Usage: python r4a1_sinkhorn_sensitivity.py OUT.csv --search DIR [DIR ...]
"""
import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import freeze_matched_lambda as FM  # noqa: E402
import prereg_rules as P  # noqa: E402

STEP = 1e-5


def resolve(entry, search):
    if os.path.exists(entry["path"]) and P.sha256(entry["path"]) == entry["sha256"]:
        return entry["path"]
    name = os.path.basename(entry["path"])
    for d in search:
        for root, _, files in os.walk(d):
            if name in files and P.sha256(os.path.join(root, name)) == entry["sha256"]:
                return os.path.join(root, name)
    sys.exit(f"no file named {name} with sha256 {entry['sha256']} under {search}")


ap = argparse.ArgumentParser()
ap.add_argument("out_csv")
ap.add_argument("--search", nargs="+", default=[])
a = ap.parse_args()
rec = json.load(open(os.path.join(REPO, "docs", "prereg", "r4_a1_record.json")))
inp = {k: [resolve(e, a.search) for e in v] for k, v in rec["inputs"].items()}
M, _ = P.read_manifest(inp["manifest"])
S, _ = P.read_scores(inp["scores"])
out = P.outcomes(M, S, P.read_failures(inp["failures"][0]), ["A1"])
need = FM.needed_cells(M, "a1")
r4 = P.r4_stage(out, "a1", P.read_record(inp["r1_record"][0], "R1"), need=need)
P._require(r4["status"] == rec["status"] and set(r4["cells"]) == set(rec["cells"]), "re-run status/cells differ from record")
for k, c in rec["cells"].items():
    P._require(all(r4["cells"][k][f] == c[f] for f in ("matched", "lo", "hi")), f"{k}: re-run differs from the record")
print(f"re-run reproduces the record: {len(rec['cells'])} cells, status {rec['status']}")

rows = []
for blk, t in sorted(rec["targets"].items()):
    task, c = blk.split("|")
    zero = P._zero_ref(out, "A1", task, int(c))
    pts = {arm: P._cell_points(out, "A1", task, int(c), arm, sorted(zero)) for arm in P.ADV_ARMS if arm != "sinkhorn"}
    base = {arm: P._closest(p, t["bstar"])[0] for arm, p in pts.items()}
    m_others = min(v for arm, v in t["M"].items() if arm != "sinkhorn")
    m_snk, crit = t["M"]["sinkhorn"], {}
    for i in range(int((m_snk - t["b0"]) / STEP) + 1):
        m = m_snk - i * STEP
        bstar = t["b0"] + P.Q_BSTAR * (min(m_others, m) - t["b0"])
        moved = [arm for arm, p in pts.items() if P._closest(p, bstar)[0] != base[arm]]
        for kind, arms in (("any", moved), ("needed", [x for x in moved if f"{blk}|{x}" in need])):
            if arms and kind not in crit:
                crit[kind] = (m, arms)
        if "needed" in crit:
            break
    row = dict(block=blk, b0=round(t["b0"], 5), b_common=round(t["b_common"], 5), argmin=min(t["M"], key=t["M"].get),
               M_sinkhorn=round(m_snk, 5), bstar=round(t["bstar"], 5))
    for kind in ("any", "needed"):   # first change of any non-Sinkhorn cell / of a cell an A2/A3 row uses
        m, arms = crit.get(kind, (None, []))
        row[f"allowed_drop_{kind}"] = round(m_snk - m, 5) if m is not None else "none down to b0"
        row[f"arms_moved_{kind}"] = ";".join(arms)
    rows.append(row)
with open(a.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
