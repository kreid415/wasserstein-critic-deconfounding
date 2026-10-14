"""Planted-error (mutation) run for scripts/analysis_x1.py against tests/analysis (rigour review AN-04 re-check).
The reviewer's 40 mutants (rv_an/rv_out/mutants.py; M16 and M27 re-targeted where the code text changed) plus the
lead's N-series for the 2026-10-14 fixes. Each mutant: exact text replacement(s), pytest -x tests/analysis."""
import concurrent.futures as cf, json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutants_review import MUTANTS as REVIEW
HERE = os.path.dirname(os.path.abspath(__file__))
orig = open(os.path.join(HERE, 'c0', 'scripts', 'analysis_x1.py')).read()
RETARGET = {
    "M16": [('q = stats.t.ppf(0.975, reading["m"] - 1)', 'q = 1.0')],
    "M27": [('- (tau ** 2).sum(axis=1) / (2 * tau_scale ** 2)', '- tau.sum(axis=1) / tau_scale')],
}
muts = []
for mid, grp, desc, old, new, cnt in REVIEW:
    reps = RETARGET.get(mid, [(old, new)])
    muts.append(dict(id=mid, group=grp, description=desc, reps=reps, cnt=None if mid in RETARGET else cnt))
NEW = [
    ("N01", "revert AN-01: uniform spread grid from 0, no Jacobian",
     [("tau = np.geomspace(TAU_FLOOR, GRID_SPAN * tau_scale, GRID_N)[indices]", "tau = np.linspace(0, GRID_SPAN * tau_scale, GRID_N)[indices]"),
      ("log_weight += np.log(tau).sum(axis=1) - (tau", "log_weight += - (tau")]),
    ("N02", "log-grid Jacobian (weight x tau) dropped", [("log_weight += np.log(tau).sum(axis=1) - (tau", "log_weight += - (tau")]),
    ("N03", "TAU_FLOOR 1e-15 -> 1e-5", [("TAU_FLOOR = 1e-15", "TAU_FLOOR = 1e-5")]),
    ("N04", "Delta posterior from V^-1 again (cancellation-prone)",
     [("    delta_var = delta2 / (1 + delta2 * x[:, :, 0].sum(axis=1))\n    delta_mean, delta_sd = delta_var * x[:, :, 1].sum(axis=1), np.sqrt(delta_var)",
       "    delta_mean = delta2 * vd.sum(axis=1)\n    delta_sd = np.sqrt(delta2 - delta2 ** 2 * inverse.sum(axis=(1, 2)))")]),
    ("N05", "zero_spread rule disabled", [('elif reading["s"] < S_MIN:', 'elif False:')]),
    ("N06a", "R4 check compares b0 only", [('for k in ("b0", "b_common", "bstar")', 'for k in ("b0",)')]),
    ("N06b", "R4 check ignores the low-signal flag",
     [('P._require(got["bstar_low_signal"] == want["bstar_low_signal"]\n                   and all(', 'P._require(all(')]),
    ("N07", "X1 seed check removed", [("P._require(sorted(zero_runs(out, task, cond)) == X1_SEEDS,", "P._require(True,")]),
    ("N08", "sign table back to the normal quantile", [('q = stats.t.ppf(0.975, reading["m"] - 1)', 'q = 1.959963984540054')]),
    ("N09", "window gap always False", [("gap=any(min(lams) < lam < max(lams) and lam not in lams for lam in means))", "gap=False)")]),
    ("N10", "best-of-k failed runs not counted",
     [('failed.setdefault(arm, []).append(int((config.status == "failed").sum()))', 'failed.setdefault(arm, []).append(0)')]),
    ("N11", "input rows not sorted by tag (expected equivalent at test tolerance)", [('.sort_values("tag", ignore_index=True)', '')]),
    ("N12", "pool accepts any positive s", [("(s >= S_MIN).all()", "(s > 0).all()")]),
    ("N13", "S_MIN 1e-6 -> 1e-9", [("S_MIN = 1e-6 ", "S_MIN = 1e-9 ")]),
    ("N14", "R4 targets taken from the record instead of recomputed", [('targets = P.r4_stage(out, "x1", r1)["targets"]', 'targets = r4["targets"]')]),
    ("N15", "sign-table t with m instead of m - 1 degrees of freedom", [('stats.t.ppf(0.975, reading["m"] - 1)', 'stats.t.ppf(0.975, reading["m"])')]),
    ("N17", "best-of-k configs = scored configurations", [("configs=len(failed.get(arm, [])),", "configs=len(v),")]),
]
for mid, desc, reps in NEW:
    muts.append(dict(id=mid, group="lead", description=desc, reps=reps, cnt=None))
for m in muts:
    for old, _ in m["reps"]:
        k = orig.count(old)
        assert k >= 1 and (m["cnt"] is None or k == m["cnt"]), (m["id"], old[:60], k)
env = dict(os.environ, WCD_SRC='src', KMP_AFFINITY='disabled', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
slots = [f"c{i}" for i in range(6)]
def run(job):
    i, m = job
    clone = os.path.join(HERE, slots[i % len(slots)])
    text = orig
    for old, new in m["reps"]:
        text = text.replace(old, new)
    t0 = time.time()
    open(os.path.join(clone, 'scripts', 'analysis_x1.py'), 'w').write(text)
    try:
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '--tb=line', '-rf', 'tests/analysis'],
                           cwd=clone, env=env, capture_output=True, text=True, timeout=900)
        out, code = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        out, code = 'TIMEOUT', -9
    finally:
        open(os.path.join(clone, 'scripts', 'analysis_x1.py'), 'w').write(orig)
    first = (re.findall(r'^FAILED (\S+)', out, flags=re.M) or re.findall(r'^ERROR (\S+)', out, flags=re.M) or [''])[0]
    return dict(id=m["id"], group=m["group"], description=m["description"], status='killed' if code != 0 else 'SURVIVED',
                exit=code, first_failure=first, secs=round(time.time() - t0, 1))
# one worker per clone: job i always runs in clone i % 6, and a clone runs one job at a time
by_slot = {s: [(i, m) for i, m in enumerate(muts) if i % len(slots) == k] for k, s in enumerate(slots)}
def worker(slot):
    return [run(j) for j in by_slot[slot]]
with cf.ThreadPoolExecutor(len(slots)) as ex:
    results = sorted([r for rs in ex.map(worker, slots) for r in rs], key=lambda r: [m["id"] for m in muts].index(r["id"]))
json.dump(results, open(os.path.join(HERE, 'mutation_results_20261014_lead.json'), 'w'), indent=1)
for r in results:
    print(r["id"], r["status"], r["first_failure"][:90], r["secs"])
print('killed', sum(r['status'] == 'killed' for r in results), 'of', len(results))
