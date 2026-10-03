#!/usr/bin/env python
"""Compare two `pip list --format=freeze` files (local reference vs a rebuilt env). Stdlib only.

Every difference (missing, extra, other version) is printed. Differences whose normalised name matches --allow
(a regex, declared substitutions such as the torch/CUDA wheels) or is listed in --allow-name NAME=REASON are
reported as ALLOWED; anything else makes the exit status 1.

Usage: python compare_freeze.py --local A.txt --remote B.txt [--allow REGEX] [--allow-name igraph="reason"]
"""
import argparse
import re
import sys


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"{path}: not a pinned freeze line: {line!r}")
        name, ver = line.split("==", 1)
        if norm(name) in out:
            raise ValueError(f"{path}: duplicate entry for {name}")
        out[norm(name)] = ver
    if not out:
        raise ValueError(f"{path}: empty freeze")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--allow", default=None, help="regex on normalised names whose differences are declared")
    ap.add_argument("--allow-name", action="append", default=[], help="NAME=REASON, one declared difference")
    a = ap.parse_args()
    loc, rem = read(a.local), read(a.remote)
    named = dict(x.split("=", 1) for x in a.allow_name)
    named = {norm(k): v for k, v in named.items()}
    rx = re.compile(a.allow) if a.allow else None
    bad = 0
    same = 0
    for k in sorted(set(loc) | set(rem)):
        lv, rv = loc.get(k), rem.get(k)
        if lv == rv:
            same += 1
            continue
        why = named.get(k) or ("declared substitution" if rx and rx.match(k) else None)
        status = f"ALLOWED ({why})" if why else "UNEXPECTED"
        bad += why is None
        print(f"{status}: {k}: local={lv} remote={rv}")
    print(f"identical={same} differing={len(set(loc) | set(rem)) - same} unexpected={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
