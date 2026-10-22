#!/usr/bin/env python
"""Prove that an A1 manifest for further tasks (build_paper_manifest.py --a1-tasks) matches the existing A1 pilot rows.

Three checks, each printed as PASS / FAIL (exit 1 if any fails); standard library plus build_paper_manifest only:
  1. grid-edge strengths: the strengths outside the base grid in the R1 record's a1_grid per family x decoder, in the
     old rows (R1's extension manifest) and in A1_EXT_R1 (what the builder adds) are the same (arm, decoder, strength) set;
  2. every new row equals the old row of the same arm, decoder (cond), strength and seed in every column except
     IGNORED; old rows of different tasks agree with each other, new and old tasks share one key set, and no task is
     both old and new;
  3. max_epochs of every row (old and new) equals the scvi-tools heuristic of its task's cell count.
Also checks that each old input file has the SHA-256 the R1 record lists for it, and prints counts per task, arm, decoder.

Usage: python3 scripts/compare_a1_pilot5.py --new <pilot5 manifest> --old <builder manifest> <R1 extension manifest> \\
           --r1-record <r1_record.json>
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_paper_manifest as BPM  # noqa: E402

# tag = <experiment>_<task>_<arm>_l<lam>_c<cond>_s<seed>_<hash that includes the task>; max_epochs = f(cells of the task)
IGNORED = ("tag", "task", "max_epochs")
COMPARED = [c for c in BPM.COLS if c not in IGNORED]
KEY = ("arm", "cond", "lam", "seed")


def read_a1(path):
    with open(path) as f:
        rows = list(csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"))
    assert rows and list(rows[0]) == BPM.COLS, f"{path}: header differs from the builder's columns"
    return [r for r in rows if r["experiment"] == "A1"]


def by_task(rows):
    out = {}
    for r in rows:
        k = tuple(r[c] for c in KEY)
        assert k not in out.setdefault(r["task"], {}), f"duplicate row {r['tag']}"
        out[r["task"]][k] = r
    return out


def compare(old_rows, new_rows):
    """Problems (strings) found comparing new A1 rows with old ones; [] means they match."""
    old, new = by_task(old_rows), by_task(new_rows)
    bad = [f"task {t} is both old and new" for t in sorted(set(old) & set(new))]
    ref = old[sorted(old)[0]]
    for t, rows in {**old, **new}.items():
        if set(rows) != set(ref):
            bad.append(f"{t}: {len(set(rows) ^ set(ref))} rows differ in (arm, cond, lam, seed) from {sorted(old)[0]}")
            continue
        bad += [f"{t} {k}: {c} {r[c]!r} != {ref[k][c]!r}" for k, r in rows.items() for c in COMPARED if r[c] != ref[k][c]]
    bad += [f"{r['tag']}: max_epochs {r['max_epochs']} != {BPM.scvi_epochs(BPM.TASKS[r['task']][0])}"
            for r in old_rows + new_rows if int(r["max_epochs"]) != BPM.scvi_epochs(BPM.TASKS[r["task"]][0])]
    return bad


def edges(triples):
    """(arm, cond, strength) triples outside the base grid; strengths as floats."""
    base = [float(x) for x in BPM.LAMBDA_GRID]
    return {(a, int(c), float(l)) for a, c, l in triples if not any(math.isclose(float(l), b) for b in base)}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", nargs="+", required=True, help="the builder manifest R1 read, then R1's extension manifest")
    ap.add_argument("--r1-record", required=True)
    a = ap.parse_args(argv)
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + msg)

    rec = json.load(open(a.r1_record))
    listed = {x["sha256"] for x in rec["inputs"]["manifest"]}
    for p in a.old:
        check(sha256(p) in listed, f"old input {p} sha256 {sha256(p)} is listed in the R1 record's inputs")
    print(f"new manifest {a.new} sha256 {sha256(a.new)}")
    old_rows = [r for p in a.old for r in read_a1(p)]
    new_rows = read_a1(a.new)
    print(f"old tasks {sorted({r['task'] for r in old_rows})}, new tasks {sorted({r['task'] for r in new_rows})}")
    print(f"ignored columns {list(IGNORED)}; compared columns ({len(COMPARED)}) {COMPARED}")

    from_rec = edges((arm, f["cond"], lam) for f in rec["families"].values() for arm in f["arms"] for lam in f["a1_grid"])
    from_old = edges((r["arm"], r["cond"], r["lam"]) for r in old_rows if r["arm"] != "none")
    from_new = edges((r["arm"], r["cond"], r["lam"]) for r in new_rows if r["arm"] != "none")
    from_const = edges((arm, cond, lam) for arm, cond, lams in BPM.A1_EXT_R1 for lam in lams)
    for name, s in (("R1 record a1_grid", from_rec), ("old rows", from_old), ("builder A1_EXT_R1", from_const), ("new rows", from_new)):
        print(f"grid-edge strengths in {name}: {sorted(s)}")
    check(from_rec == from_old == from_const == from_new, "grid-edge strengths (arm, cond, strength) agree in all four")
    for k, f in sorted(rec["families"].items()):
        print(f"  R1 {k}: rounds {f['rounds']}, a1_grid {f['a1_grid'][0]} .. {f['a1_grid'][-1]} ({len(f['a1_grid'])} points)")

    problems = compare(old_rows, new_rows)
    old, new = by_task(old_rows), by_task(new_rows)
    check(not problems, f"{sum(map(len, new.values()))} new rows over {len(new)} tasks equal the {len(next(iter(old.values())))} "
                        f"old rows per task in {len(COMPARED)} columns; max_epochs follows the cell count in all "
                        f"{len(old_rows) + len(new_rows)} rows")
    for p in problems[:20]:
        print("  " + p)
    print("rows per task, arm and decoder (cond 0 = unconditioned, 1 = conditioned):")
    for t, rows in sorted({**old, **new}.items()):
        cnt = {}
        for r in rows.values():
            cnt[(r["arm"], r["cond"])] = cnt.get((r["arm"], r["cond"]), 0) + 1
        print(f"  {t} ({'old' if t in old else 'new'}, {len(rows)} rows): " +
              ", ".join(f"{arm} c{c} {n}" for (arm, c), n in sorted(cnt.items())))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
