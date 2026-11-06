"""Generated parts of Supplementary Notes 10 and 11 (branch supp-records); no number in them is typed by hand.

  collect  Read the data root and git; write paper/supp/supp-records_facts.json: per experiment and task, the fits, the
           scored fits, the fitting device, commit and package versions recorded in each latent, and the scoring CPU,
           site, commit, package versions and file dates recorded in each score file; package versions of the committed
           environment exports; dates of the commits and tags named in Note 10.  Run on the workstation (needs the data
           root):  python paper/tools/make_supp_supp-records.py collect [--data-root D]
  render   (default) Write the generated blocks of paper/supp/note10.tex and note11.tex from the facts file;
           --check exits 1 when a block differs from what the facts give.
"""
import argparse, csv, json, re, subprocess, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
FACTS = "paper/supp/supp-records_facts.json"
SWEEP_DIR = "sweep_analysis_final_20261105"
REPL_REV, REPL_MANIFEST = "a475ea3", "manifests/sweep_replication_x1_ext1.tsv"
REPL_SCORES = ["tier12_v3/sweep_replication/scores", "sweep_replication/jhpce_scores/scores"]
REPL_LATENTS = "tier12_v3/sweep_replication/latents"
PILOT_SCORES = "tier12_v3/rules/r23/scores_list.txt"
BIO_REV = "697cc6c"
PKGS = ["scvi-tools", "torch", "lightning", "pyro-ppl", "anndata", "scanpy", "numpy", "scipy", "pandas", "scikit-learn",
        "numba", "pynndescent", "scib", "R_kBET", "rpy2", "harmony-pytorch", "scanorama"]
# label -> (git revision, what happened); the order is the order of Note 10's table
TIMELINE = [
    ("prereg-tier12-v1", "Decision rules of the sweep (then a pilot) and of the planned fixed-strength benchmark first fixed, before the first sweep fit"),
    ("prereg-tier12-v2", "Amendments of 4 October; the sweep restarted from zero at this tag (fits made at the first tag unused)"),
    ("prereg-tier12-v3", "Amendments of 13 to 16 October (Sinkhorn correction and redo, checks before the tag, low-signal rule, first analysis specification); code of every fit and score"),
    ("57241a2", "Five follow-up experiments dropped before any was run (decision of 17 October)"),
    ("prereg-x6-v1", "Rule of the follow-up on regularisation, optimiser and update steps fixed"),
    ("98732ff", "Amendment of 22 October: planned fixed-strength benchmark stopped (21 October); sweep extended to the five further tasks with unchanged code and rules"),
    ("065a486", "Ranking-analysis rules: descriptive ranking design replaces the contrast design; the follow-up on regularisation, optimiser and update steps and the premise check dropped"),
    ("5d01da0", "Ranking-analysis rules: answers to the nine open points written in"),
    ("2bc5b00", "Rescope: the planned fixed-strength benchmark is not run; the sweep is the result; all baselines of the run list added"),
    ("8e32bb2", "Sweep analysis code for all eight tasks (first run on all eight tasks the same day, for debugging)"),
    ("2113ec7", "Replication design: 855-row run list from the sweep's batch scores"),
    ("replication-prediction-v1", "Replication prediction and agreement rule fixed, before any replication fit was scored"),
    ("8595a16", "Hypervolume and share of non-dominated strengths added; final sweep analysis run (5 November) at this commit"),
    ("639c302", "Direct biology check: rules written, before any of its values was computed"),
    ("803ed80", "Replication result before the extension step (first replication scores read)"),
    ("648f7e8", "Replication extension run list: 40 fits required by the agreement rule"),
    ("f618484", "Scaling checks of Supplementary Figure S5 computed (paper branch, supplement figures)"),
    ("9faac6e", "Per-task sign-flip sensitivity analysis of the replication defined (paper branch, text)"),
    ("a475ea3", "Replication result with the extension step"),
]
GROUPS = {"sweep": "Sweep", "input_pilot": "Input pilot", "standardisation_pilot": "Standardisation pilot",
          "scvi_adv": "scvi-tools adversary", "scanvi": "scANVI", "sysvi": "sysVI", "harmony": "Harmony",
          "scanorama": "Scanorama", "pca": "PCA", "replication": "Replication"}
TASKS = {"pancreas": "pancreas", "lung": "lung", "immune": "immune", "immune_hum_mou": "human--mouse immune",
         "atac_small": "ATAC small", "atac_large": "ATAC large", "sim1": "simulation 1", "sim2": "simulation 2"}


def git(*args, repo=ROOT):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def latent_info(path):
    """(device, fit commit, versions) recorded in a latent file; device 'not on the workstation' if the file is absent."""
    if not path.is_file():
        return "latent not on the workstation", "", {}
    c = json.loads(str(np.load(path)["config"]))
    if "gpu" in c:
        return c["gpu"], c["git_sha"][:7], {"scvi-tools": c["scvi"], "torch": c["torch"]}
    return "CPU " + c["cpu_model"], c["git_sha"][:7], c["versions"]


def score_info(path):
    with open(path, newline="") as f:
        r = next(csv.DictReader(f))
    return {"score_where": f'{r["cpu_model"]} ({r.get("score_site") or "workstation"})', "scorer_commit": r["scorer_git_sha"][:7],
            "scorer_versions": r["scorer_versions"], "score_date": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")}


def tsv(lines):
    return csv.DictReader((l for l in lines if not l.startswith("#")), delimiter="\t")


def fit_rows(d):
    """(group, task, tag, score file or None, latent file) for every fit of the sweep, baselines, input pilots and replication."""
    s = d / SWEEP_DIR / "inputs"
    scores = {Path(p).stem: Path(p) for p in (s / "scores_list.txt").read_text().split()}
    latents = dict(line.split("\t") for line in (s / "latents.tsv").read_text().splitlines() if line)
    for r in tsv(open(s / "manifest.tsv")):
        group = "sweep" if r["experiment"] == "A1" else r["arm"]
        yield group, r["task"], r["tag"], scores.get(r["tag"]), Path(latents.get(r["tag"], "/nonexistent"))
    for p in map(Path, (d / PILOT_SCORES).read_text().split()):
        exp, task = p.stem.split("_")[0], re.match(r"A\d_(.+?)_(none|discriminator|reference|pooled|barycenter|mmd|sinkhorn)_", p.stem).group(1)
        if exp in ("A2", "A3"):
            yield ("input_pilot" if exp == "A2" else "standardisation_pilot"), task, p.stem, p, p.parent.parent / "latents" / (p.stem + ".npz")
    found = {p.stem: p for sd in REPL_SCORES if (d / sd).is_dir() for p in (d / sd).glob("*.csv")}
    for r in tsv(git("show", f"{REPL_REV}:{REPL_MANIFEST}").splitlines()):
        yield "replication", r["task"], r["tag"], found.get(r["tag"]), d / REPL_LATENTS / (r["tag"] + ".npz")


def collect_provenance(d):
    agg = defaultdict(lambda: defaultdict(set))
    n = defaultdict(lambda: [0, 0])
    where = defaultdict(lambda: defaultdict(int))  # scoring CPU and site -> number of score files
    for group, task, tag, score, latent in fit_rows(d):
        k = (group, task)
        n[k][0] += 1
        dev, commit, vers = latent_info(latent)
        agg[k]["fit_device"].add(dev)
        agg[k]["fit_commit"].add(commit)
        agg[k]["fit_versions"].add(json.dumps(vers, sort_keys=True))
        if score is not None and score.is_file():
            n[k][1] += 1
            info = score_info(score)
            where[k][info.pop("score_where")] += 1
            for key, v in info.items():
                agg[k][key].add(v)
    return [{"group": g, "task": t, "n_fits": n[(g, t)][0], "n_scored": n[(g, t)][1], "score_where": dict(where[(g, t)]),
             **{key: sorted(v - {""}) for key, v in agg[(g, t)].items()}} for g, t in sorted(agg)]


def pins(text):
    """Package versions from a pip freeze (name==version) or a conda explicit export (URL .../name-version-build.conda)."""
    out = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)", text, re.M))
    out.update(re.findall(r"/(python|r-base)-(\d[\w.]*)-[^/]*\.(?:conda|tar\.bz2)", text))
    return out


def collect_versions(d, prov):
    show = lambda path, rev="10a1ca5": git("show", f"{rev}:{path}")
    fit_ws = {**pins(show("cluster/envspec/local_scvi-api_pip_freeze.txt")), **pins(show("cluster/envspec/scvi-api.explicit.txt"))}
    score = {**pins(show("cluster/envspec/wcd-kbet.explicit.txt")), **pins(show("cluster/envspec/local_wcd-kbet_pip_freeze.txt"))}
    one = lambda key, groups: {v for r in prov if r["group"] in groups for v in r.get(key, [])}
    scorer = one("scorer_versions", set(GROUPS))
    cpu = one("fit_versions", {"harmony", "scanorama", "pca"})
    assert len(scorer) == 1 and len(cpu) == 1, (scorer, cpu)  # one scoring and one CPU-tool environment, as the rule scripts require
    score.update(json.loads(scorer.pop()))
    SCORE_PKGS = set(json.loads(next(iter(one("scorer_versions", set(GROUPS)))))) | {"python", "r-base", "pandas", "scipy"}
    sweep = dict(re.findall(r"^\s*- (\S+?)[=]+(\S+)$", (d / SWEEP_DIR / "ENVIRONMENT_scvi-api.yml").read_text(), re.M))
    torch = lambda gpu: {json.loads(v)["torch"] for r in prov if r["group"] == "sweep" and gpu in r["fit_device"] for v in r["fit_versions"]}
    (t_ws,), (t_cl,) = torch("NVIDIA GeForce RTX 3080"), torch("NVIDIA L40S")  # one torch build per fitting machine
    bio = json.loads(show("docs/biology_check/biology_check.json", BIO_REV))["versions"]
    bio["scikit-learn"] = bio.pop("sklearn")
    return {"fitting, workstation": {**{k: v for k, v in fit_ws.items() if k in PKGS + ["python"]}, "torch": t_ws},
            "fitting, JHPCE": {**{k: v for k, v in pins(show("cluster/envspec/fit-pip.txt")).items() if k in PKGS}, "torch": t_cl},
            "Harmony, Scanorama, PCA": json.loads(cpu.pop()),
            "scoring": {k: v for k, v in score.items() if k in SCORE_PKGS},
            "sweep analysis": {k: v for k, v in sweep.items() if k in PKGS + ["python"]}, "biology check": bio}


def collect_git():
    out = {}
    for label, _ in TIMELINE:
        tag = label in git("tag", "--list", label).split()
        date = git("for-each-ref", "--format=%(taggerdate:iso-strict)", f"refs/tags/{label}").strip() if tag else \
            git("log", "-1", "--format=%cI", label).strip()
        out[label] = {"commit": git("rev-parse", "--short=7", f"{label}^{{commit}}").strip(), "date": date, "tag": tag}
    return out


def collect(data_root):
    d = Path(data_root)
    prov = collect_provenance(d)
    inputs = json.loads((d / SWEEP_DIR / "inputs" / "inputs.json").read_text())
    return {"provenance": prov, "versions": collect_versions(d, prov), "git": collect_git(),
            "baseline_rows_missing": inputs["baseline_rows_missing"]}


P = r">{{\raggedright\arraybackslash}}p{{{}\linewidth}}"  # ragged-right column of a given share of the line width


def tex(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt_date(iso):
    return datetime.fromisoformat(iso).strftime("%-d %b %Y, %H:%M")


def timeline_block(facts):
    rows = [r"\begin{tabular}{@{}" + "".join(P.format(w) for w in (0.14, 0.22, 0.5)) + "@{}}", r"\toprule",
            r"Date (New York) & Commit (tag) & Rule record or change \\", r"\midrule"]
    for label, what in TIMELINE:
        g = facts["git"][label]
        # tags whose names carry the old 'prereg' wording are shown as tagged commits (paper wording rule); the replication tag by name
        ref = g["commit"] + ((" (git tag)" if label.startswith("prereg") else rf" (tag \texttt{{{tex(label)}}})") if g["tag"] else "")
        rows.append(rf"{fmt_date(g['date'])} & {ref} & {tex(what)} \\")
    return rows + [r"\bottomrule", r"\end{tabular}"]


def short_device(s):
    return s.replace("(jhpce)", "(JHPCE)").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").replace(" 6-Core Processor", "").replace(" 64-Core Processor", "")


def provenance_block(facts):
    """One line per experiment and machine combination: tasks, fits scored of fits, fitting device and commit, scoring CPU and site, score file dates."""
    lines = defaultdict(lambda: {"tasks": [], "n": 0, "s": 0, "commits": set(), "dates": set()})
    for r in facts["provenance"]:
        key = (r["group"], "; ".join(map(short_device, r["fit_device"])), "; ".join(short_device(w) + (f": {k:,}" if len(r["score_where"]) > 1 else "")
                                                                           for w, k in sorted(r["score_where"].items())))
        x = lines[key]
        x["tasks"].append(TASKS[r["task"]])
        x["n"] += r["n_fits"]
        x["s"] += r["n_scored"]
        x["commits"] |= set(r["fit_commit"]) | set(r.get("scorer_commit", []))  # absent when no fit of the task is scored
        x["dates"] |= set(r.get("score_date", []))
    rows = [r"\begin{tabular}{@{}" + "".join(P.format(w) for w in (0.11, 0.18)) + "r" + "".join(P.format(w) for w in (0.11, 0.18, 0.08, 0.11)) + "@{}}",
            r"\toprule", r"Experiment & Tasks & Scored/fits & Fitted on & Scored on & Commits & Score files dated \\", r"\midrule"]
    order = list(GROUPS)
    for (g, dev, sc), x in sorted(lines.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1], kv[0][2])):
        dates = sorted(x["dates"])
        span = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else (dates[0] if dates else "--")
        tasks = "all eight" if len(x["tasks"]) == len(TASKS) else ", ".join(x["tasks"])
        rows.append(rf"{GROUPS[g]} & {tasks} & {x['s']:,}/{x['n']:,} & {tex(dev)} & {tex(sc) or '--'} & "
                    rf"{', '.join(sorted(x['commits'])) or '--'} & {span} \\")
    return rows + [r"\bottomrule", r"\end{tabular}"]


HEAD = {"fitting, workstation": "Fits, workstation", "fitting, JHPCE": "Fits, JHPCE", "Harmony, Scanorama, PCA": "CPU tools",
        "scoring": "Scoring", "sweep analysis": "Sweep analysis", "biology check": "Biology check"}


def versions_block(facts):
    envs = ["fitting, workstation", "fitting, JHPCE", "Harmony, Scanorama, PCA", "scoring", "sweep analysis", "biology check"]
    assert sorted(envs) == sorted(facts["versions"])
    names = ["python", "r-base"] + PKGS
    rows = [r"\begin{tabular}{@{}l" + "l" * len(envs) + "@{}}", r"\toprule",
            "Package & " + " & ".join(HEAD[e] for e in envs) + r" \\", r"\midrule"]
    for p in names:
        vals = [facts["versions"][e].get(p, "--") for e in envs]
        if any(v != "--" for v in vals):
            rows.append(tex({"r-base": "R", "R_kBET": "kBET (R)", "python": "Python"}.get(p, p)) + " & " +
                        " & ".join(tex(v) for v in vals) + r" \\")
    return rows + [r"\bottomrule", r"\end{tabular}"]


BLOCKS = {"paper/supp/note10.tex": {"timeline": timeline_block},
          "paper/supp/note11.tex": {"provenance": provenance_block, "versions": versions_block}}


def render(facts, root=ROOT, check=False):
    """Replace each generated block; with check=True only report blocks that differ. Returns the list of stale blocks."""
    stale = []
    for rel, blocks in BLOCKS.items():
        p = Path(root) / rel
        text = p.read_text()
        for name, fn in blocks.items():
            pat = re.compile(rf"(% BEGIN generated: {name}\n).*?(% END generated: {name}\n)", re.S)
            if not pat.search(text):
                raise SystemExit(f"{rel}: no generated block {name!r}")
            new = pat.sub(lambda m: m.group(1) + "\n".join(fn(facts)) + "\n" + m.group(2), text)
            if new != text:
                stale.append(f"{rel}:{name}")
            text = new
        if not check:
            p.write_text(text)
    return stale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", choices=["render", "collect"], default="render")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    facts_path = Path(a.root) / FACTS
    if a.mode == "collect":
        facts_path.write_text(json.dumps(collect(a.data_root), indent=1, sort_keys=True) + "\n")
        print("wrote", facts_path)
        return 0
    stale = render(json.loads(facts_path.read_text()), a.root, check=a.check)
    print(("out of date: " if a.check else "rewrote: ") + (", ".join(stale) or "nothing"))
    return 1 if (a.check and stale) else 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry; main() is tested
    sys.exit(main())
