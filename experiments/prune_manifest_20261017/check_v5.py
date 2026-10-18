"""Check manifest v5 against v4 and write docs/manifest_v5_counts.csv and manifests/*_v5.tsv.sha256 (stdlib only; no score is read).

Fails unless v5 = v4 minus the rows of X3, X7, X8, X12, X15 (same order, byte-identical rows, header equal apart from the
excluded_experiments field and the row count) and the per-experiment counts are the expected ones (user decision 2026-10-17).
Run from the repo root: python3 experiments/prune_manifest_20261017/check_v5.py
"""
import collections
import hashlib

V4 = "manifests/paper_manifest_stock_pilot_u5_b10_v4.tsv"
V5 = "manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv"
DROP = {"X3", "X7", "X8", "X12", "X15"}
WANT = {"X1": 3000, "A1": 1098, "X6": 486, "X13": 408, "A3": 180, "A2": 108}


def read(path):
    lines = open(path).read().split("\n")
    assert lines[-1] == "", f"{path}: no final newline"
    return lines[0], lines[1], lines[2:-1]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    h4, cols4, r4 = read(V4)
    h5, cols5, r5 = read(V5)
    assert cols4 == cols5, "column header differs"
    assert h5 == h4.replace("followup_design=v4, ", "followup_design=v4, excluded_experiments=X3,X7,X8,X12,X15, ").replace(
        "(0 of 10908)", "(0 of 5280)"), "comment header differs beyond excluded_experiments and the row count"
    exp = lambda r: r.split("\t")[1]
    assert [r for r in r4 if exp(r) not in DROP] == r5, "v5 rows are not v4 minus the excluded experiments"
    c4, c5 = collections.Counter(map(exp, r4)), collections.Counter(map(exp, r5))
    assert dict(c5) == WANT, f"v5 counts {dict(c5)} differ from the expected {WANT}"
    assert sum(c5.values()) == 5280 and not (set(c5) & DROP) and sum(c4.values()) == 10908, (c4, c5)
    with open("docs/manifest_v5_counts.csv", "w") as f:
        f.write("experiment,v4,v5\n")
        for e in sorted(c4, key=lambda e: (-c5.get(e, 0), e)):
            f.write(f"{e},{c4[e]},{c5.get(e, 0)}\n")
        f.write(f"TOTAL,{sum(c4.values())},{sum(c5.values())}\n")
    with open(V5 + ".sha256", "w") as f:
        f.write(f"{sha256(V5)}  {V5}\n")
    print(f"OK v4 {sha256(V4)} ({len(r4)} rows) -> v5 {sha256(V5)} ({len(r5)} rows)")


if __name__ == "__main__":
    main()
