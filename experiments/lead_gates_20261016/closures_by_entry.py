"""Per-entry-point import closures with the functions of the first code review's probe (import_closure.py, reused through
exec of its text up to the dynamic-reference scan), and who imports the two files changed in round 3.
usage: python closures_by_entry.py REPO PROBE_PY FROM_SHA MID_SHA TO_SHA
FROM_SHA..MID_SHA = earlier rounds (c3fd92d..5d06a27), MID_SHA..TO_SHA = this round (5d06a27..HEAD)."""
import os, re, subprocess, sys
repo, probe, a, mid, b = sys.argv[1:6]
src = open(probe).read().split("DYN = ")[0]
sys.argv = ["import_closure.py", repo, a, b]
ns = {}
exec(compile(src, probe, "exec"), ns)
closure = ns["closure"]
git = lambda *x: subprocess.run(["git", "-C", repo, *x], capture_output=True, text=True, check=True).stdout
entries = {
    "fit_paper_config.py (the fitter; entry point of the identity proof)": ["scripts/fit_paper_config.py"],
    "scvi_adversarial_plan.py (the training plan; entry point of the identity proof)": ["scripts/scvi_adversarial_plan.py"],
    "run_cpu_baselines.py (X13 CPU runner)": ["scripts/run_cpu_baselines.py"],
    "run_stage.py (the stage runner)": ["scripts/run_stage.py"],
    "score_scib_native.py (the scorer; entry point of the identity proof)": ["scripts/score_scib_native.py"],
    "prod_helpers.py (cluster job helpers)": ["cluster/jhpce/prod_helpers.py"],
}
rounds = {f"earlier rounds {a[:7]}..{mid[:7]}": set(git("diff", "--name-only", f"{a}..{mid}").split()),
          f"this round {mid[:7]}..{b[:7]}": set(git("diff", "--name-only", f"{mid}..{b}").split())}
for k, e in entries.items():
    cl = closure(e)
    print(f"== {k}: {len(cl)} files: {', '.join(cl)}")
    for r, ch in rounds.items():
        print(f"     changed in {r} and in this closure: {sorted(set(cl) & ch) or 'NONE'}")
print()
print("files changed in this round (all):", sorted(rounds[f"this round {mid[:7]}..{b[:7]}"]))
print()
for mod in ("run_stage", "prereg_rules"):
    pat = re.compile(rf"^\s*(import {mod}\b|from {mod} import|import {mod} as)", re.M)
    hits = []
    for d in ("scripts", "cluster", "experiments", "src", "tests"):
        for root, _, fs in os.walk(os.path.join(repo, d)):
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(root, f)
                    if pat.search(open(p, errors="replace").read()):
                        hits.append(os.path.relpath(p, repo))
    print(f"python files that import {mod}: {len(hits)}")
    for h in sorted(hits):
        print("    ", h)
