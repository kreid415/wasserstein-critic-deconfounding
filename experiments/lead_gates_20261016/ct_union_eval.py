"""Which lines changed since --base are run by NO test in ANY of the tiers?  (union of the per-tier coverage that real
check_tests runs collected; check_tests.py itself only looks at one tier at a time.)

usage: python ct_union_eval.py REPO BASE TIER_DIR:SRC [TIER_DIR:SRC ...]
  REPO     a clone at the commit under review (git diff BASE is taken in it)
  TIER_DIR a clone that holds .check_tests/cov.json from a real run;  SRC = the --src directory of that run (scripts | src)
A changed non-test line or branch counts as run if at least one tier that measures its file ran it."""
import json, os, re, subprocess, sys
from pathlib import Path

repo, base = os.path.abspath(sys.argv[1]), sys.argv[2]
tiers = [a.split(":") for a in sys.argv[3:]]
sh = lambda *c: subprocess.run(c, check=True, capture_output=True, text=True, cwd=repo).stdout


def changed_lines(base):
    out, cur = {}, None
    for ln in sh("git", "diff", "-U0", "--no-color", "--no-ext-diff", base, "--", "*.py").splitlines():
        if ln.startswith("+++ "):
            cur = None if ln.endswith("/dev/null") else ln[6:]
        elif ln.startswith("@@") and cur:
            m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
            s, n = int(m[1]), int(m[2] if m[2] is not None else 1)
            out.setdefault(cur, set()).update(range(s, s + n))
    return out


def is_test(p):
    n = Path(p).name
    return n.startswith("test_") or n.endswith("_test.py") or n == "conftest.py"


data = []                                          # (src root, {relative path: file record})
for d, src in tiers:
    rep = json.load(open(os.path.join(os.path.abspath(d), ".check_tests", "cov.json")))["files"]
    data.append((src, {os.path.normpath(os.path.relpath(os.path.abspath(os.path.join(os.path.abspath(d), k)), os.path.abspath(d))): v for k, v in rep.items()}))
chg = changed_lines(base)
bad, ok_files = [], 0
for p, lines in sorted(chg.items()):
    if is_test(p) or not Path(os.path.join(repo, p)).exists():
        continue
    recs = [rep[os.path.normpath(p)] for src, rep in data if p.startswith(src + "/") and os.path.normpath(p) in rep]
    if not recs:
        continue                                   # no tier measures this file (e.g. experiments/, cluster/)
    miss = set.intersection(*[set(r["missing_lines"]) for r in recs]) & lines
    mbr = set.intersection(*[set(map(tuple, r.get("missing_branches", []))) for r in recs])
    mbr = sorted((s, t) for s, t in mbr if s in lines and s not in miss)
    if miss or mbr:
        bad.append((p, sorted(miss), mbr))
    else:
        ok_files += 1
print(f"base {base}: changed .py files measured by at least one tier: {ok_files + len(bad)}; with lines/branches run by no tier: {len(bad)}")
for p, miss, mbr in bad:
    print(f"  {p}: lines {miss[:12]}{' ...' if len(miss) > 12 else ''} ({len(miss)}), branches {mbr[:6]} ({len(mbr)})")
