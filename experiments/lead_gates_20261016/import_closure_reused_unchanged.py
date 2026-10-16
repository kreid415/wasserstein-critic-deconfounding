"""Static import closure (ast) of the fit and score entry points, plus the dynamic-reference scan, intersected with the files changed
between two commits. usage: python import_closure.py REPO FROM_SHA TO_SHA"""
import ast, os, re, subprocess, sys
repo, a, b = sys.argv[1:4]
changed = set(subprocess.run(["git", "-C", repo, "diff", "--name-only", f"{a}..{b}"], capture_output=True, text=True, check=True).stdout.split())
# module name -> file, for the repo's own code
index = {}
for d in ("scripts", "cluster/jhpce"):
    for f in os.listdir(os.path.join(repo, d)):
        if f.endswith(".py"):
            index.setdefault(f[:-3], os.path.join(d, f))
pkg = {}
for root, _, files in os.walk(os.path.join(repo, "src")):
    for f in files:
        if f.endswith(".py"):
            rel = os.path.relpath(os.path.join(root, f), os.path.join(repo, "src"))
            mod = rel[:-3].replace(os.sep, ".")
            if mod.endswith(".__init__"): mod = mod[: -len(".__init__")]
            pkg[mod] = os.path.join("src", rel)

def imports(path):
    tree = ast.parse(open(os.path.join(repo, path)).read())
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [x.name for x in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                parent = os.path.dirname(path)
                if path.startswith("src/"):
                    modpath = os.path.relpath(parent, os.path.join(repo, "src")) if False else parent[len("src/"):].replace("/", ".")
                    base = ".".join(modpath.split(".")[: len(modpath.split(".")) - (n.level - 1)] + ([base] if base else []))
                else:
                    base = base
            names.append(base)
            names += [f"{base}.{x.name}" for x in n.names]
    return names

def resolve(name, frm):
    out = []
    top = name.split(".")[0]
    if name in pkg: out.append(pkg[name])
    # parent packages' __init__ run too
    parts = name.split(".")
    for i in range(1, len(parts)):
        p = ".".join(parts[:i])
        if p in pkg and parts[0] == "wcd_vae": out.append(pkg[p])
    if top in index and "." not in name: out.append(index[top])
    return out

def closure(entries):
    seen, todo = {}, list(entries)
    while todo:
        f = todo.pop()
        if f in seen or not os.path.isfile(os.path.join(repo, f)): continue
        seen[f] = True
        for n in imports(f):
            for r in resolve(n, f):
                if r not in seen: todo.append(r)
    return sorted(seen)

DYN = re.compile(r"importlib|__import__|runpy|exec\(|spec_from_file_location|subprocess|os\.system|Popen|\.py[\"']")
entries = {
  "fit (run_stage.py, fit_paper_config.py, scvi_adversarial_plan.py, run_cpu_baselines.py)": ["scripts/run_stage.py", "scripts/fit_paper_config.py", "scripts/scvi_adversarial_plan.py", "scripts/run_cpu_baselines.py"],
  "scorer (score_scib_native.py)": ["scripts/score_scib_native.py"],
  "cluster job helpers (prod_helpers.py)": ["cluster/jhpce/prod_helpers.py"],
}
allfiles = set()
for k, ents in entries.items():
    cl = closure(ents)
    allfiles |= set(cl)
    print(f"== {k}: {len(cl)} files in the static import closure")
    print("   ", ", ".join(cl))
    print("    changed", f"{a[:7]}..{b[:7]}", "and in closure:", sorted(set(cl) & changed) or "NONE")
print()
print("dynamic-reference scan (lines naming another .py file, subprocess, importlib, exec) in closure files:")
for f in sorted(allfiles):
    for i, line in enumerate(open(os.path.join(repo, f)), 1):
        if DYN.search(line) and not line.lstrip().startswith("#"):
            m = re.findall(r"[\w./-]+\.py", line)
            hit = [x for x in m if os.path.basename(x)[:-3] in index or os.path.basename(x) in {os.path.basename(c) for c in changed}]
            if hit or re.search(r"importlib|__import__|runpy|exec\(|spec_from_file_location", line):
                print(f"   {f}:{i}: {line.strip()[:170]}")
print()
print("changed code files in the range:", sorted(c for c in changed if c.endswith(('.py', '.sh', '.toml', '.json', '.tsv')) and not c.startswith(('notebook/', 'docs/'))))
