#!/usr/bin/env python
"""Verdict of the A2/A3 switch check (check_switches.sh): compares the 1-epoch latents written in OUT_DIR.

PASS needs: every fit exited 0 (OUT_DIR/fit_rc.tsv), the default half fitted twice is bit-identical, each new half
differs from its default half (A2 sample vs mean; A3 zstd 1 vs 0 on the critic and on the critic-free path), the two
seeds differ, and every latent exists and is finite. Writes OUT_DIR/switch_check.csv; exit 0 = PASS, 1 = FAIL.
compare(out_dir) returns the rows; check(out_dir) raises SwitchCheckFailed (used by the mutation check).
"""
import csv
import json
import os
import sys

import numpy as np

ROLES = ("D_ref", "S_ref", "S_ref101", "Z_ref", "D_mmd", "Z_mmd")
CHECKS = (  # (name, a, b, want_equal)
    ("determinism (default half fitted twice)", "D_ref", "D_ref_repeat", True),
    ("A2 switch live: sample vs mean (reference)", "S_ref", "D_ref", False),
    ("A3 switch live: zstd 1 vs 0 (reference)", "Z_ref", "D_ref", False),
    ("A3 switch live: zstd 1 vs 0 (mmd, critic-free)", "Z_mmd", "D_mmd", False),
    ("seeds differ (A2 reference, 100 vs 101)", "S_ref", "S_ref101", False),
)


class SwitchCheckFailed(RuntimeError):
    pass


def _latent(out, run, tag):
    p = os.path.join(out, run, "latents", f"{tag}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as f:
        return np.array(f["z"])


def compare(out):
    roles = json.load(open(os.path.join(out, "roles.json")))
    if sorted(roles) != sorted(ROLES):
        raise SwitchCheckFailed(f"roles.json has {sorted(roles)}, expected {sorted(ROLES)}")
    rows = []
    rc_path = os.path.join(out, "fit_rc.tsv")
    rcs = [l.rstrip("\n").split("\t") for l in open(rc_path) if l.strip()] if os.path.exists(rc_path) else []
    bad_rc = [r for r in rcs if len(r) != 3 or r[2] != "0"]
    rows.append(dict(check="every fit exited 0", a=f"n={len(rcs)}", b="expected n=7", want="True",
                     equal=(len(rcs) == 7 and not bad_rc), max_abs_diff="", ok=(len(rcs) == 7 and not bad_rc)))
    Z = {r: _latent(out, "run1", roles[r]) for r in ROLES}
    Z["D_ref_repeat"] = _latent(out, "run2", roles["D_ref"])
    for name, a, b, want_equal in CHECKS:
        za, zb = Z[a], Z[b]
        if za is None or zb is None:
            rows.append(dict(check=name, a=a, b=b, want="equal" if want_equal else "differ", equal="missing", max_abs_diff="", ok=False))
            continue
        same_shape = za.shape == zb.shape
        eq = same_shape and np.array_equal(za, zb)
        mad = float(np.nanmax(np.abs(za - zb))) if same_shape else float("nan")
        rows.append(dict(check=name, a=a, b=b, want="equal" if want_equal else "differ", equal=eq, max_abs_diff=mad, ok=(eq == want_equal)))
    fin = all(v is not None and v.size > 0 and bool(np.isfinite(v).all()) for v in Z.values())
    rows.append(dict(check="all latents present and finite", a=f"n={sum(v is not None for v in Z.values())}", b="expected n=7",
                     want="True", equal=fin, max_abs_diff="", ok=fin))
    return rows


def check(out):
    rows = compare(out)
    failed = [r["check"] for r in rows if not r["ok"]]
    if failed:
        raise SwitchCheckFailed("; ".join(failed))
    return rows


def main():
    out = sys.argv[1]
    rows = compare(out)
    with open(os.path.join(out, "switch_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{'PASS' if r['ok'] else 'FAIL'} | {r['check']} | equal={r['equal']} max|dz|={r['max_abs_diff']}")
    ok = all(r["ok"] for r in rows)
    print(f"SWITCH CHECK {'PASS' if ok else 'FAIL'} (n={len(rows)} checks)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
