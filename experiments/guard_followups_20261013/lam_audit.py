"""Guard review G-05: lam values of the arms that never read it (none, scvi_adv, scanvi) in every manifest file of the
repo, so that refusing a non-zero lam refuses no planned row. Usage: python lam_audit.py > results/lam_audit.txt"""
import collections, csv, glob, os
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
files = sorted(glob.glob(f"{R}/manifests/*.tsv")) + sorted(glob.glob(f"{R}/scripts/*manifest*.tsv"))
n, used, count, vals = 0, 0, collections.Counter(), collections.defaultdict(set)
for f in files:
    with open(f) as fh:
        rows = list(csv.DictReader((line for line in fh if not line.startswith("#")), delimiter="\t"))
    if not rows or not {"arm", "lam"} <= set(rows[0]):
        continue
    used, n = used + 1, n + len(rows)
    for r in rows:
        if r["arm"] in ("none", "scvi_adv", "scanvi", "sysvi"):
            count[r["arm"]] += 1
            vals[r["arm"]].add(r["lam"])
print(f"{len(files)} manifest files, {used} with arm and lam columns, {n} rows")
for a in sorted(count):
    print(f"{a}: {count[a]} rows, lam values {sorted(vals[a])}")
