"""Planted-error (mutation) run for scripts/analysis_x1.py against tests/analysis (rigour review AN-04 of 2026-10-14, extended
2026-10-15). Every mutant is an exact text replacement in the script; it is applied alone to one clone's copy, `pytest -x tests/analysis`
runs, and the mutant is killed when pytest exits non-zero. The result records whether the first failure is an assertion (FAILED)
or an error before the assertions (ERROR, e.g. the fixture could not be built).

The mutants: the rigour reviewer's 40 (mutants_review.py), the lead's N-series of 2026-10-14 (mutants_lead.py; the reviewer's M16, M26
and M27 and N05 changed with the code), and the errors of the 2026-10-15 round (mutants_r3.py). M24 and N11 occur in two lists and run once.

Re-run after committing, from this folder (clones of the committed HEAD beside this script; one per parallel process, at most 4; about
20 minutes with 4; conda env scvi-api, CPU only):
    for i in 0 1 2 3; do git clone -q ../../.. c$i; done && python run_mut.py
usage: run_mut.py [--ids O02 F07 ...] [--out results.json]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mutants_lead import NEW, RETARGET  # noqa: E402
from mutants_r3 import ACCEPTANCE, EXTRA  # noqa: E402
from mutants_review import MUTANTS as REVIEW  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ids", nargs="*", default=None)
ap.add_argument("--out", default=os.path.join(HERE, "mutation_results_20261015.json"))
a = ap.parse_args()

muts = []
for mid, grp, desc, old, new, cnt in REVIEW:
    muts.append(dict(id=mid, group=grp, description=desc, reps=RETARGET.get(mid, [(old, new)]), cnt=None if mid in RETARGET else cnt))
for mid, desc, reps in NEW:
    muts.append(dict(id=mid, group="lead", description=desc, reps=reps, cnt=None))
seen = {m["id"] for m in muts}
for mid, desc, reps in ACCEPTANCE + EXTRA:
    if mid not in seen:
        muts.append(dict(id=mid, group="round 3", description=desc, reps=reps, cnt=1))
if a.ids:
    unknown = sorted(set(a.ids) - {m["id"] for m in muts})
    assert not unknown, f"unknown mutant ids {unknown}"
    muts = [m for m in muts if m["id"] in a.ids]
slots = sorted(d for d in os.listdir(HERE) if re.fullmatch(r"c\d+", d))
assert slots, "no clones c0, c1, ... beside this script (see the usage in the header)"
orig = open(os.path.join(HERE, slots[0], "scripts", "analysis_x1.py")).read()
for m in muts:
    for old, _ in m["reps"]:
        k = orig.count(old)
        assert k >= 1 and (m["cnt"] is None or k == m["cnt"]), (m["id"], old[:70], k)
env = dict(os.environ, WCD_SRC="src", CUDA_VISIBLE_DEVICES="", KMP_AFFINITY="disabled", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MKL_THREADING_LAYER="SEQUENTIAL")


def pytest(clone, *flags, timeout=None):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *flags, "tests/analysis"], cwd=os.path.join(HERE, clone),
                          env=env, capture_output=True, text=True, timeout=timeout)


def run(slot, m):
    text = orig
    for old, new in m["reps"]:
        text = text.replace(old, new)
    t0 = time.time()
    path = os.path.join(HERE, slot, "scripts", "analysis_x1.py")
    open(path, "w").write(text)
    try:
        r = pytest(slot, "-x", "--tb=line", "-rfE", timeout=900)
        out, code = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        out, code = "TIMEOUT", -9
    finally:
        open(path, "w").write(orig)
    f = re.search(r"^(FAILED|ERROR) (.+?)(?: - |$)", out, flags=re.M)  # a test id can contain spaces; pytest ends it with " - reason"
    return dict(id=m["id"], group=m["group"], description=m["description"], status="killed" if code != 0 else "SURVIVED", exit=code,
                kind=f[1] if f else "", first_failure=f[2] if f else "", secs=round(time.time() - t0, 1))


def worker(k):
    return [run(slots[k], m) for i, m in enumerate(muts) if i % len(slots) == k]


# the unmutated clone must pass first (a failing baseline would make every mutant a kill)
base = pytest(slots[0])
assert base.returncode == 0, f"unmutated tests fail in {slots[0]}: {base.stdout.strip().splitlines()[-1]}"
print("unmutated:", base.stdout.strip().splitlines()[-1], flush=True)
with cf.ThreadPoolExecutor(len(slots)) as ex:
    results = sorted([r for rs in ex.map(worker, range(len(slots))) for r in rs], key=lambda r: [m["id"] for m in muts].index(r["id"]))
json.dump(results, open(a.out, "w"), indent=1)
for r in results:
    print(r["id"], r["status"], r["kind"], r["first_failure"][:100], r["secs"])
print("killed", sum(r["status"] == "killed" for r in results), "of", len(results))
