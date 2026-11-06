#!/usr/bin/env python
"""Supplementary Tables S1, S2, S4 and S5 (tasks, training settings, baseline grids, run counts) as LaTeX; no typed numbers.

Step 1, --collect (on the workstation; Python with h5py, numpy and pandas, e.g. the paper-tools environment) reads
  D/prepped_scib/<task>__scib.h5ad                      the eight prepared task files (cells, batches, cell types, nesting)
  D/sweep_analysis_final_20261105/inputs/              run list (manifest.tsv), latents.tsv and scores_list.txt of the sweep
                                                        analysis (sweep-paper 8595a16): what was fitted and scored
  D/sweep_analysis_final_20261105/tables/baseline_coverage.csv   baseline fits planned and scored
  committed files at fixed commits (git show): fingerprints of the prepared files, the scorer and the penalty code at
  10a1ca5, the figshare download record at 3202725, the rule constants of the sweep analysis at 8595a16, the
  replication run list at a475ea3 and the input-pilot plan at 10a1ca5
and writes paper/figure_data/supp_methods/ (tasks.csv, settings.csv, manifest.tsv, baseline_coverage.csv,
sources.json, MANIFEST.sha256, and scib_<version>/: the scib metric files the scorer ran, with their MIT licence). Every training setting in settings.csv is the value recorded in the fits' own latent
files (the resolved training plan of each fit), counted over all fits.
Step 2 (default) reads paper/figure_data/supp_methods and writes paper/supp/tableS1.tex, tableS2.tex, tableS4.tex, tableS5.tex;
--check exits 1 when a committed table differs from the regenerated one (and writes nothing).
Usage, from the repository root:
  python paper/tools/make_supp_supp-methods.py --collect [--data-root D]
  python paper/tools/make_supp_supp-methods.py [--check]
"""
import argparse
import ast
import collections
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                     # .../paper
REPO = ROOT.parent
DATA = ROOT / "figure_data" / "supp_methods"
OUT = ROOT / "supp"
DATA_ROOT = "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
SWEEP = "sweep_analysis_final_20261105"
CODE, ANALYSIS = "10a1ca5", "8595a16"                          # fitting and scoring code; sweep analysis code
GIT = dict(fingerprints=(CODE, "docs/prepped_fingerprints_scib.json"),
           raw_sources=("3202725", "archive/claude-science/2026-10-02/scib_raw_sources.tsv"),
           scorer=(CODE, "scripts/score_scib_native.py"), alignment=(CODE, "src/wcd_vae/wcd/alignment.py"),
           head=(CODE, "src/wcd_vae/wcd/adversarial.py"),
           rules=(ANALYSIS, "scripts/prereg_rules.py"),
           replication=("a475ea3", "manifests/sweep_replication_x1_ext1.tsv"),
           pilot=(CODE, "experiments/A2A3_pilot/PREFLIGHT.md"))
SCIB = "/home/kendall/.claude-science/conda/envs/wcd-kbet/lib/python3.11/site-packages"   # scorer environment
SCIB_FILES = ["metrics.py", "pcr.py", "silhouette.py", "lisi.py", "graph_connectivity.py", "kbet.py", "nmi.py", "ari.py",
              "clustering.py", "isolated_labels.py", "cell_cycle.py", "trajectory.py"]
SHARED_COLS = ["n_latent", "n_layers", "n_hidden", "likelihood", "batch_size", "max_epochs", "train_size", "adv_input",
               "zstd", "n_critic"]
TABLES = ("tableS1.tex", "tableS2.tex", "tableS4.tex", "tableS5.tex")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_supp_tables import DEC_LABEL, PEN, PEN_LABEL, REAL, SIMS, TASK_LABEL  # noqa: E402  (labels shared with Tables S6-S7)

TASKS = REAL + SIMS


# ------------------------------------------------------------------ step 1: collect
def git_show(rev, path, repo=REPO):
    return subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def constants(src, names):
    """Literal module-level assignments `names` of a Python source, read without importing it."""
    out = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
           if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
           and n.targets[0].id in names}
    missing = sorted(set(names) - set(out))
    if missing:
        raise ValueError(f"constants not found: {missing}")
    return out


def _js(v):
    return json.loads(v) if isinstance(v, str) else v


def h5_obs(path, keys):
    """obs columns of an h5ad file as string arrays ('_index' = cell names), read with h5py only."""
    import h5py
    with h5py.File(path, "r") as h:
        o = h["obs"]

        def col(k):
            g = o[k]
            if isinstance(g, h5py.Group):
                codes = g["codes"][:]
                if (codes < 0).any():
                    raise ValueError(f"{path}: obs {k} has missing values")
                return g["categories"][:].astype(str)[codes]
            return g[:].astype(str)
        return {"_index": col(o.attrs["_index"]), **{k: col(k) for k in keys if k in o}}


def metric_presence(scores, metrics):
    """{metric: 'all' | 'none'} over the score rows of one task; a metric present in only some rows is an error."""
    out = {}
    for m in metrics:
        k = int(scores[m].notna().sum()) if m in scores else 0
        if 0 < k < len(scores):
            raise ValueError(f"{m}: present in {k} of {len(scores)} score rows")
        out[m] = "all" if k else "none"
    return out


def epochs_rule(n_cells, cap=400, decay_at=20000):
    """scvi-tools 1.4.2 get_max_epochs_heuristic: min(round(decay_at / n * cap), cap), at least 1."""
    return max(min(round(decay_at / n_cells * cap), cap), 1)


def task_rows(prepped, fingerprints, raw, rules, metrics, manifest, scores, tasks=TASKS):
    """One record per task; every derived count is cross-checked against a second record."""
    obs = {t: h5_obs(Path(prepped) / f"{t}__scib.h5ad", ["batch", "celltype", "Batch"]) for t in tasks}
    # a cell is matched across tasks by its batch and barcode (the cell name up to its first hyphen): the files add
    # different numeric suffixes to the same barcode (for example ATAC large '-0-0-0-1' where ATAC small has '-1')
    names = {t: {f"{b}|{c.split('-')[0]}" for b, c in zip(o["batch"], o["_index"])} for t, o in obs.items()}
    rows = []
    for t in tasks:
        fp, o = fingerprints[f"{t}__scib.h5ad"], obs[t]
        uns = fp["uns"]
        status, hvg = _js(uns["count_status_as_distributed"]), _js(uns["hvg"])
        if len(o["_index"]) != fp["shape"][0] or len(set(o["batch"])) != len(status):
            raise ValueError(f"{t}: cells or batches differ from the fingerprint record")
        src = raw[raw.file == uns["source_file"]]
        if len(src) != 1 or src.md5.iloc[0] != uns["source_md5"]:
            raise ValueError(f"{t}: download record and prepared file disagree on the source md5")
        a1 = manifest[(manifest.experiment == "A1") & (manifest.task == t)]
        epochs = sorted(a1.max_epochs.unique())
        if len(epochs) != 1 or epochs[0] != epochs_rule(len(o["_index"])):
            raise ValueError(f"{t}: epochs {epochs} are not the scvi-tools rule for {len(o['_index'])} cells")
        pres = metric_presence(scores[scores.task == t], metrics["batch"] + metrics["bio"] + ["hvg_overlap"])
        bio = [m for m in metrics["bio"] if pres[m] == "all"]
        used = metrics["batch"] + bio
        want = [m for m in metrics["bio"] if m not in ("cell_cycle_conservation", "trajectory")
                or (m == "cell_cycle_conservation" and t in rules["CC_TASKS"])
                or (m == "trajectory" and t in rules["TRAJ_TASKS"])]
        if bio != want or any(pres[m] != "all" for m in metrics["batch"]):
            raise ValueError(f"{t}: scored metrics {bio} differ from the rule's list {want}")
        parent = ""
        if uns["scib_batch_key"] != "Batch" and "Batch" in o:   # sub-batches nested in batches (simulation 2)
            if len(set(zip(o["batch"], o["Batch"]))) != len(set(o["batch"])):
                raise ValueError(f"{t}: a batch lies in more than one parent batch")
            parent = len(set(o["Batch"]))
        others = [u for u in tasks if u != t]
        nonint = [b for b, s in status.items() if s["frac_integer_values"] < 1]
        rows.append(dict(
            task=t, kind="simulated" if t in SIMS else "real", modality=uns["modality"], organism=uns["organism"],
            cells=len(o["_index"]), batches=len(status), batch_key=uns["scib_batch_key"],
            cell_types=len(set(o["celltype"])), label_key=uns["scib_label_key"], genes_in_file=fp["shape"][1],
            features=fp["n_hvg"], feature_rule=hvg["method"], nonint_batches=len(nonint),
            nonint_names="|".join(nonint), parent_batches=parent, family=rules["TASK_FAMILY"].get(t, ""),
            shares_cells_with="|".join(f"{u}:{len(names[t] & names[u])}" for u in others if names[t] & names[u]),
            shared_batches_with="|".join(f"{u}:{len(set(o['batch']) & set(obs[u]['batch']))}" for u in others
                                         if set(o["batch"]) & set(obs[u]["batch"])),
            epochs=int(epochs[0]), bio_metrics="|".join(bio), n_bio=len(bio),
            blank_metrics="|".join(m for m, p in pres.items() if p == "none"),
            metric_min=float(scores.loc[scores.task == t, used].min().min()),
            metric_max=float(scores.loc[scores.task == t, used].max().max()),
            fits_below_zero=int((scores.loc[scores.task == t, used] < 0).any(axis=1).sum()),
            metrics_below_zero="|".join(m for m in used if (scores.loc[scores.task == t, m] < 0).any()),
            source_file=uns["source_file"], figshare_file_id=int(src.figshare_file_id.iloc[0]),
            source_bytes=int(src.bytes.iloc[0]), source_md5=uns["source_md5"], counts_sha1=fp["counts"],
            prepared_sha256=sha256(Path(prepped) / f"{t}__scib.h5ad")))
    return pd.DataFrame(rows)


def config_fields(cfg):
    """Resolved settings of one fit's latent config: training-plan values per phase (strength excluded, it varies by
    design) or a CPU tool's arguments, plus library versions."""
    if "tool_kwargs" in cfg:
        out = {f"tool.{k}": v for k, v in cfg["tool_kwargs"].items()}
        out.update({f"version.{k}": v for k, v in cfg["versions"].items()})
        return {k: json.dumps(v, sort_keys=True) for k, v in out.items()}
    out = {"version.scvi-tools": cfg["scvi"]}
    for phase, plan in cfg["plan"].items():
        for k, v in plan.items():
            if k == "optimizers":
                out.update({f"{phase}.optimizer{i}": o for i, o in enumerate(v)})
            elif k != "d_coef":
                out[f"{phase}.{k}"] = v
    return {k: json.dumps(v, sort_keys=True) for k, v in out.items()}


def settings_rows(configs):
    """[(experiment, arm, config)] -> long table experiment, arm, field, value, n_fits."""
    c = collections.Counter((e, a, f, v) for e, a, cfg in configs for f, v in config_fields(cfg).items())
    return pd.DataFrame([dict(experiment=e, arm=a, field=f, value=v, n_fits=k) for (e, a, f, v), k in sorted(c.items())])


def replication_rows(rep, manifest):
    """Seeds of the replication run list, after checking that its other settings are the sweep's per task and method."""
    sweep = manifest[manifest.experiment == "A1"]
    for (t, a), g in rep.groupby(["task", "arm"]):
        s = sweep[(sweep.task == t) & (sweep.arm == a)]
        for c in SHARED_COLS:
            if sorted(g[c].astype(str).unique()) != sorted(s[c].astype(str).unique()):
                raise ValueError(f"replication {t} {a}: {c} differs from the sweep")
    return [dict(experiment="replication", arm="all", field="manifest.seed",
                 value=json.dumps(sorted(int(x) for x in rep.seed.unique())), n_fits=len(rep))]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv(text):
    return pd.read_csv(io.StringIO(text), sep="\t", comment="#")


def collect(data_root=DATA_ROOT, out=DATA, tasks=TASKS, site=SCIB):
    """Step 1 (module docstring). Returns the output folder."""
    import numpy as np
    d = Path(data_root)
    inp = d / SWEEP / "inputs"
    manifest = pd.read_csv(inp / "manifest.tsv", sep="\t", comment="#")
    latents = pd.read_csv(inp / "latents.tsv", sep="\t", header=None, names=["tag", "path"]).merge(
        manifest[["tag", "experiment", "arm"]], on="tag", validate="one_to_one")
    if len(latents) != len(manifest):
        raise ValueError(f"{len(manifest) - len(latents)} run-list rows have no latent")
    sc = constants(git_show(*GIT["scorer"]), ["BATCH_METRICS", "BIO_METRICS"])
    metrics = dict(batch=sc["BATCH_METRICS"], bio=sc["BIO_METRICS"])
    paths = [p.strip() for p in open(inp / "scores_list.txt") if p.strip()]
    scores = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).merge(
        manifest[["tag", "task"]], on="tag", validate="one_to_one", suffixes=("_score", ""))
    if len(scores) != len(manifest):
        raise ValueError(f"{len(scores)} score rows for {len(manifest)} run-list rows")
    rules = constants(git_show(*GIT["rules"]), ["TASK_FAMILY", "CC_TASKS", "TRAJ_TASKS"])
    task_tab = task_rows(d / "prepped_scib", json.loads(git_show(*GIT["fingerprints"])),
                         read_tsv(git_show(*GIT["raw_sources"])), rules, metrics, manifest, scores, tasks)
    cfgs = [(r.experiment, r.arm, json.loads(str(np.load(r.path, allow_pickle=True)["config"])))
            for r in latents.itertuples()]
    al = constants(git_show(*GIT["alignment"]), ["BW_MULTIPLIERS", "SINKHORN_BLUR_FRAC"])
    pilot = re.search(r"seed: (\d+)-(\d+)", git_show(*GIT["pilot"]))
    layers = len(re.findall(r"self\.fc\d+ = _sn\(nn\.Linear\(", git_show(*GIT["head"])))   # hidden layers + output
    extra = replication_rows(read_tsv(git_show(*GIT["replication"])), manifest) + [
        dict(experiment="code", arm="mmd", field="BW_MULTIPLIERS", value=json.dumps(al["BW_MULTIPLIERS"]), n_fits=0),
        dict(experiment="code", arm="sinkhorn", field="SINKHORN_BLUR_FRAC", value=json.dumps(al["SINKHORN_BLUR_FRAC"]),
             n_fits=0),
        dict(experiment="code", arm="adversary head", field="hidden_layers", value=json.dumps(layers - 1), n_fits=0),
        dict(experiment="input pilots", arm="all", field="manifest.seed",
             value=json.dumps(list(range(int(pilot.group(1)), int(pilot.group(2)) + 1))), n_fits=0)]
    settings = pd.concat([settings_rows(cfgs), pd.DataFrame(extra)], ignore_index=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    task_tab.to_csv(out / "tasks.csv", index=False)
    settings.to_csv(out / "settings.csv", index=False)
    manifest.to_csv(out / "manifest.tsv", sep="\t", index=False)
    cov = d / SWEEP / "tables" / "baseline_coverage.csv"
    pd.read_csv(cov).to_csv(out / "baseline_coverage.csv", index=False)
    src = {f"{SWEEP}/inputs/{f}": sha256(inp / f) for f in ("manifest.tsv", "latents.tsv", "scores_list.txt")}
    src[f"{SWEEP}/tables/baseline_coverage.csv"] = sha256(cov)
    src.update({f"git:{rev}:{p}": hashlib.sha256(git_show(rev, p).encode()).hexdigest() for rev, p in GIT.values()})
    (out / "sources.json").write_text(json.dumps(dict(data_root=str(d), sha256=src), indent=1) + "\n")
    files = ["tasks.csv", "settings.csv", "manifest.tsv", "baseline_coverage.csv", "sources.json"]
    # the scib metric code the scorer ran (version from every score row), copied with its MIT licence for Note 3
    version = scores.scorer_versions.map(lambda v: json.loads(v)["scib"]).unique()
    info = Path(site) / f"scib-{version[0]}.dist-info"
    if len(version) != 1 or not info.is_dir():
        raise ValueError(f"scorer scib versions {version}; {info} missing")
    lib = out / f"scib_{version[0]}"
    lib.mkdir(exist_ok=True)
    for f in SCIB_FILES:
        (lib / f).write_bytes((Path(site) / "scib" / "metrics" / f).read_bytes())
    (lib / "LICENSE.txt").write_bytes((info / "licenses" / "LICENSE.txt").read_bytes())
    files += [f"{lib.name}/{f}" for f in SCIB_FILES + ["LICENSE.txt"]]
    (out / "MANIFEST.sha256").write_text("".join(f"{sha256(out / f)}  {f}\n" for f in files))
    return out


# ------------------------------------------------------------------ step 2: tables
def n(x):
    return f"{int(x):,}"


def tex(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def setting(settings, arm, field, experiment=None):
    """The one value of a resolved setting over all fits of a method (an error if the fits disagree)."""
    s = settings[(settings.arm == arm) & (settings.field == field)]
    if experiment:
        s = s[s.experiment == experiment]
    if s.value.nunique() != 1:
        raise ValueError(f"{arm} {field}: {s.value.nunique()} distinct values")
    return json.loads(s.value.iloc[0])


def sci(x):
    """Powers of ten from 1e-3 down as $10^{k}$; other numbers as %g."""
    e = math.log10(x) if x > 0 else 0.0
    return f"$10^{{{int(e)}}}$" if e == int(e) and e <= -3 else f"{x:g}"


def adam(o):
    """One Adam optimiser record (cls, param_groups) as text."""
    if o["cls"] != "Adam" or len(o["param_groups"]) != 1:
        raise ValueError(f"unexpected optimiser {o}")
    g = o["param_groups"][0]
    return (f"Adam, learning rate {sci(g['lr'])}, betas ({g['betas'][0]:g}, {g['betas'][1]:g}), epsilon {sci(g['eps'])}, "
            + (f"weight decay {sci(g['weight_decay'])}" if g["weight_decay"] else "no weight decay"))


def seeds(values):
    v = sorted(int(x) for x in values)
    return f"{v[0]}--{v[-1]}" if len(v) > 2 and v == list(range(v[0], v[-1] + 1)) else ", ".join(map(str, v))


def kl_final(epochs, warmup):
    """KL weight in the last epoch (0-based epoch E-1) of scvi-tools' linear warm-up over `warmup` epochs."""
    return min(1.0, (epochs - 1) / warmup)


def P(w):
    """A ragged-right paragraph column of width w (fraction of the text width)."""
    return r">{\raggedright\arraybackslash}p{" + f"{w}" + r"\textwidth}"


HEAD = "% generated by paper/tools/make_supp_supp-methods.py from paper/figure_data/supp_methods; do not edit"
OPEN = r"{\scriptsize\renewcommand{\tablename}{Supplementary Table}\renewcommand{\thetable}{%s}%%"


def relation(r, u, k, cells):
    """How task r relates to task u with which it shares k batch names; cell names are used only to state that one
    task holds all cells of the other (a partial overlap is not reported)."""
    shared = dict(p.split(":") for p in r.shares_cells_with.split("|") if p).get(u)
    if shared and int(shared) == r.cells:
        return f"all cells and batches also in {TASK_LABEL[u]}"
    if shared and int(shared) == cells[u]:
        return f"holds all cells and batches of {TASK_LABEL[u]}"
    return f"{k} batches also in {TASK_LABEL[u]}"


def table_s1(tasks):
    lab = dict(TASK_LABEL)
    cells = dict(zip(tasks.task, tasks.cells))
    body, ck = [], []
    for r in tasks.itertuples():
        kind = r.kind + (", gene activity" if r.modality == "atac" else ", RNA")
        feats = f"all {n(r.features)}" if r.features == r.genes_in_file else f"{n(r.features)} of {n(r.genes_in_file)}"
        batches = f"{r.batches} ({tex(r.batch_key)})" + (f" in {r.parent_batches}" if r.parent_batches else "")
        rel = [r.family or "--"] + [relation(r, u, int(k), cells) for u, k in
                                    (p.split(":") for p in r.shared_batches_with.split("|") if p)]
        body.append(f"{lab[r.task]} & {kind} & {n(r.cells)} & {batches} & {r.cell_types} & {feats} & "
                    f"{r.nonint_batches} of {r.batches} & {'; '.join(rel)} \\\\")
        ck.append(f"{lab[r.task]} & {{\\tiny\\ttfamily {tex(r.source_file)}}} & {r.figshare_file_id} & "
                  f"{{\\tiny\\ttfamily {r.source_md5}}} & {{\\tiny\\ttfamily {r.prepared_sha256}}} \\\\")
    nonint = "; ".join(f"{lab[r.task]}: {', '.join(tex(b) for b in sorted(r.nonint_names.split('|')))}"
                       for r in tasks.itertuples() if r.nonint_batches)
    return "\n".join([
        HEAD, OPEN % "S1", r"\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{@{}" + P(0.11) + P(0.08) + "r" + P(0.11) + "r" + P(0.08) + "l" + P(0.27) + "@{}}",
        r"\caption{Tasks: the eight integration tasks of the single-cell integration benchmark scIB (main-text "
        r"reference; figshare article 12420968, version 8), as prepared for every fit and score. ``Batches'': number "
        r"of batches of the batch variable used (scIB's name in parentheses); for simulation 2 these are sub-batches, "
        r"nested in the batches whose number follows ``in''. ``Features'': training features, scIB's highly variable "
        r"genes for RNA and all gene-activity features for the chromatin-accessibility tasks, of the genes in the file. "
        r"``Non-integer'': batches whose counts, used as distributed, hold non-integer values (" + nonint + r"). "
        r"``Family'': tasks of one family never choose a strength or setting for each other in the single-strength "
        r"placement; the simulations belong to no family and are ranked separately. Shared batches are counted by "
        r"batch name, shared cells by batch and barcode (the cell name up to its first hyphen). Part B: the downloaded file, its figshare file id and MD5 checksum (checked at download), and the "
        r"SHA-256 checksum of the prepared file; the prepared files also match the content fingerprints committed in "
        r"\texttt{docs/prepped\_fingerprints\_scib.json} at commit " + CODE + r".}\label{tab:s1}\\",
        r"\toprule Task & Kind & Cells & Batches & Cell types & Features & Non-integer & Family; shared batches \\",
        r"\midrule", r"\multicolumn{8}{@{}l}{\textbf{Part A: task contents}} \\", r"\midrule", *body, r"\bottomrule",
        r"\end{longtable}", r"\begin{longtable}{@{}lllll@{}}",
        r"\multicolumn{5}{@{}l}{\textbf{Part B: files and checksums}} \\", r"\toprule",
        r"Task & File & figshare id & MD5 of download & SHA-256 of prepared file \\", r"\midrule", *ck,
        r"\bottomrule", r"\end{longtable}", "}", ""])


def table_s2(tasks, manifest, settings):
    sw = manifest[manifest.experiment == "A1"]
    one = {}
    for c in ["n_latent", "n_layers", "n_hidden", "likelihood", "batch_size", "train_size", "adv_input", "zstd"]:
        v = sw[c].unique()
        if len(v) != 1:
            raise ValueError(f"sweep fits differ in {c}: {v}")
        one[c] = v[0]
    gen = setting(settings, "none", "fit.optimizer0", "A1")
    warm = setting(settings, "none", "fit.n_epochs_kl_warmup", "A1")
    for a in PEN + ["scvi_adv"]:
        if setting(settings, a, "fit.optimizer0") != gen or setting(settings, a, "fit.n_epochs_kl_warmup") != warm:
            raise ValueError(f"{a}: encoder optimiser or KL warm-up differs from plain scVI")
    shared = [
        ("Latent dimensions", str(one["n_latent"])),
        ("Encoder and decoder", f"{one['n_layers']} hidden layer of {one['n_hidden']} units each"),
        ("Likelihood", "zero-inflated negative binomial" if one["likelihood"] == "zinb" else tex(one["likelihood"])),
        ("Training split", f"{one['train_size']:.0%}".replace("%", r"\%") + " of cells for training, the rest for "
         "validation; no early stopping: every fit trains for its task's epochs (Part C)"),
        ("Minibatch", f"{one['batch_size']} cells"),
        ("Encoder and decoder optimiser", adam(gen)),
        ("KL warm-up", f"linear from {setting(settings, 'none', 'fit.min_kl_weight', 'A1'):g} to "
                       f"{setting(settings, 'none', 'fit.max_kl_weight', 'A1'):g} over {warm} epochs (scvi-tools "
                       "default); the penalty is not scaled by it"),
        ("Adversary input", ("posterior mean" if one["adv_input"] == "mean" else "posterior sample")
         + (", no per-dimension standardisation" if one["zstd"] == 0 else ", standardised per dimension")),
        ("Scored embedding", "posterior mean of every cell (training and validation cells)")]
    d, rows = int(one["n_latent"]), [("Plain scVI", "--", "--", "--", r"strength $\lambda=0$")]
    for a in PEN:
        steps, gp = setting(settings, a, "fit.adv_steps", "A1"), setting(settings, a, "fit.lambda_gp", "A1")
        opt = [json.loads(v) for v in settings[(settings.arm == a) & (settings.field == "fit.optimizer1")].value]
        if len(opt) > 1:
            raise ValueError(f"{a}: adversary optimisers differ between fits")
        hidden = (f"head: {setting(settings, 'adversary head', 'hidden_layers')} hidden layers of "
                  f"{setting(settings, a, 'fit.adv_hidden', 'A1')} units") if steps else ""
        other = {"discriminator": hidden, "reference": hidden + "; target: the reference batch",
                 "pooled": hidden + "; target: the other batches",
                 "barycenter": hidden + f"; target: barycenter, at most {setting(settings, a, 'fit.bary_iter', 'A1')} "
                                        "iterations per step",
                 "mmd": r"Gaussian kernels, $\sigma=\sqrt{" + str(d) + r"}\times\{"
                        + ", ".join(f"{m:g}" for m in setting(settings, "mmd", "BW_MULTIPLIERS")) + r"\}$",
                 "sinkhorn": r"$\epsilon=" + f"{setting(settings, 'sinkhorn', 'SINKHORN_BLUR_FRAC'):g}" + r"\sqrt{2\times"
                             + str(d) + r"}$, Euclidean cost"}[a]
        rows.append((PEN_LABEL[a], str(steps) if steps else "--", adam(opt[0]) if opt else "--",
                     f"{gp:g}" if gp is not None else "--", other))
    adv = setting(settings, "scvi_adv", "fit.optimizer1", "X1")
    rows.append(("scvi-tools adversary", "as scvi-tools", adam(adv), "--",
                 r"scvi-tools' adversarial classifier (Supplementary Note 1); batch-conditioned decoder only; "
                 r"weight $1-\beta$"))
    per_task = [f"{TASK_LABEL[r.task]} & {n(r.cells)} & {r.epochs} & {kl_final(r.epochs, warm):.4f} \\\\"
                for r in tasks.itertuples()]
    rep = settings[(settings.experiment == "replication") & (settings.field == "manifest.seed")]
    sd = [("Penalty-strength sweep (plain scVI and the six penalties)", seeds(sw.seed.unique()), n(len(sw))),
          ("Replication across training seeds", seeds(json.loads(rep.value.iloc[0])), n(rep.n_fits.iloc[0])),
          ("Input pilots (two tasks, both decoders)", seeds(setting(settings, "all", "manifest.seed", "input pilots")),
           "--")]
    for tools, lab in ((["scvi_adv", "scanvi", "sysvi"], "Baselines: scvi-tools adversary, scANVI, sysVI"),
                       (["harmony", "scanorama", "pca"], "Baselines: Harmony, Scanorama, PCA (one run each)")):
        b = manifest[manifest.arm.isin(tools)]
        sd.append((lab, seeds(b.seed.unique()), n(len(b))))
    return "\n".join([
        HEAD, OPEN % "S2", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}" + P(0.2) + P(0.76) + "@{}}",
        r"\caption{Hyperparameters and training settings (Supplementary Note 2). Every value is the one recorded in the "
        r"fits' own latent files (the resolved training plan of each fit), and every fit of a method has the same value. "
        r"Part A: settings shared by every scVI-backbone fit (plain scVI, the six penalties and the scvi-tools "
        r"adversary); Part B: settings that differ by method, each method's standard setting; Part C: settings by task; "
        r"Part D: seeds. The loss of each penalty is in Supplementary Note 1, the baselines' settings in Supplementary "
        r"Table S4.}\label{tab:s2}\\",
        r"\toprule", r"\multicolumn{2}{@{}l}{\textbf{Part A: shared settings}} \\", r"\midrule",
        *[f"{k} & {v} \\\\" for k, v in shared], r"\bottomrule", r"\end{longtable}",
        r"\begin{longtable}{@{}" + P(0.13) + P(0.09) + P(0.35) + P(0.09) + P(0.28) + "@{}}",
        r"\multicolumn{5}{@{}l}{\textbf{Part B: settings by method}} \\", r"\toprule",
        r"Method & Adversary steps per encoder step & Adversary optimiser & Gradient penalty weight & Other \\",
        r"\midrule", *[" & ".join(r) + r" \\" for r in rows], r"\bottomrule", r"\end{longtable}",
        r"\begin{longtable}{@{}lrrr@{}}", r"\multicolumn{4}{@{}l}{\textbf{Part C: settings by task}} \\", r"\toprule",
        r"Task & Cells & Epochs & KL weight in the last epoch \\", r"\midrule", *per_task, r"\bottomrule",
        r"\multicolumn{4}{@{}p{0.6\textwidth}@{}}{Epochs: scvi-tools' rule $\min(400,\mathrm{round}(20000\times400/n))$ "
        r"for $n$ cells, the same for every method of a task. KL weight in the last of $E$ epochs: $(E-1)/" + str(warm)
        + r"$, as the warm-up counts epochs from 0.} \\", r"\end{longtable}",
        r"\begin{longtable}{@{}llr@{}}", r"\multicolumn{3}{@{}l}{\textbf{Part D: seeds}} \\", r"\toprule",
        r"Experiment & Seeds & Fits \\", r"\midrule", *[" & ".join(r) + r" \\" for r in sd], r"\bottomrule",
        r"\end{longtable}", "}", ""])


TOOLS = [("scvi_adv", "scvi-tools adversary", "none (weight $1-\\beta$)"),
         ("scanvi", "scANVI (uses cell-type labels)", "none"),
         ("sysvi", "sysVI", "cycle-consistency weight"),
         ("harmony", "Harmony", r"diversity penalty $\theta$"),
         ("scanorama", "Scanorama", "neighbours (knn)"),
         ("pca", "PCA (uncorrected)", "none")]
INPUT = {"scvi_adv": "counts; batch-conditioned decoder", "scanvi": "counts and cell-type labels; scVI pre-training",
         "sysvi": "scIB's normalised values; Gaussian likelihood", "harmony": "PCA of scIB's normalised values",
         "scanorama": "scIB's normalised values", "pca": "scIB's normalised values"}
VERSION = {"scvi_adv": "scvi-tools", "scanvi": "scvi-tools", "sysvi": "scvi-tools", "harmony": "harmony-pytorch",
           "scanorama": "scanorama", "pca": "scanpy"}


def table_s4(manifest, settings, coverage):
    body, planned, scored = [], 0, 0
    for arm, label, knob in TOOLS:
        b = manifest[manifest.arm == arm]
        main, sens = b[b.n_latent == b.n_latent.min()], b[b.n_latent != b.n_latent.min()]
        cov = coverage[coverage.arm == arm]
        planned, scored = planned + int(cov.expected.sum()), scored + int(cov.scored.sum())
        varied = not knob.startswith("none")
        grid = ", ".join(f"{v:g}" for v in sorted(main.lam.unique())) if varied else "--"
        sensitivity = "; ".join((f"{v:g} at " if varied else "") + f"{k} dimensions"
                                for k, v in sorted(set(zip(sens.n_latent, sens.lam)))) or "--"
        body.append(f"{label} & {knob} & {grid} & {main.n_latent.iloc[0]} & {sensitivity} & {seeds(b.seed.unique())} & "
                    f"{int(cov.expected.sum())} & {int(cov.scored.sum())} & {INPUT[arm]} & "
                    f"{VERSION[arm]} {setting(settings, arm, 'version.' + VERSION[arm])} \\\\")
    lab = {a: x for a, x, _ in TOOLS}
    missing = "; ".join(f"{r.missing} of {lab[r.arm]} on {TASK_LABEL[r.task]}" for r in coverage.itertuples() if r.missing)
    return "\n".join([
        HEAD, OPEN % "S4", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}" + P(0.11) + P(0.1) + P(0.1) + "r" + P(0.09) + "lrr" + P(0.15) + P(0.08) + "@{}}",
        r"\caption{Baseline grids (Supplementary Note 8), on each of the eight tasks. ``Grid'': the settings of the "
        r"varied knob at the main embedding size (``Dims'', latent or principal-component dimensions), each run on "
        r"every task; ``Sensitivity'': a further setting at the tool's default embedding size, not used to choose a "
        r"setting. ``Planned'' and ``Scored'' count fits over the eight tasks and all seeds; not fitted: "
        + (missing or "none") + r". The neural baselines use the scVI-backbone settings of Supplementary Table S2 "
        r"(epochs, minibatch, training split, hidden units and latent size of each task); the other tools read the same "
        r"training features.}\label{tab:s4}\\",
        r"\toprule Tool & Knob & Grid & Dims & Sensitivity & Seeds & Planned & Scored & Input & Version \\", r"\midrule",
        *body, r"\midrule", f"All & & & & & & {planned} & {scored} & & \\\\", r"\bottomrule", r"\end{longtable}", "}",
        ""])


def table_s5(manifest, coverage):
    """Run counts per method, decoder and task: sweep fits of the analysed run list (collect checks one score row per
    run-list row) and baseline fits scored of planned (baseline coverage table)."""
    if int(coverage.failed.sum()):
        raise ValueError("a baseline fit failed; Table S5 has no column for failed fits")
    sweep, base = manifest[manifest.experiment == "A1"], manifest[manifest.experiment != "A1"]
    if set(sweep.arm) != {"none", *PEN}:
        raise ValueError(f"sweep arms {sorted(set(sweep.arm))}")
    cnt = sweep.groupby(["arm", "cond", "task"]).size()
    lab = dict(PEN_LABEL, none="Plain scVI")
    body = []
    for arm in ["none", *PEN]:
        for cond in (1, 0):
            row = [int(cnt.get((arm, cond, t), 0)) for t in TASKS]
            body.append(f"{lab[arm]} & {DEC_LABEL[cond]} & " + " & ".join(map(str, row)) + f" & {n(sum(row))} \\\\")
    tot = [int((sweep.task == t).sum()) for t in TASKS]
    body += [r"\midrule", "Sweep, all & & " + " & ".join(map(n, tot)) + f" & {n(sum(tot))} \\\\", r"\midrule"]
    cov = coverage.groupby(["arm", "task"])[["expected", "scored"]].sum()
    for arm, label, _ in TOOLS:
        dec = base[base.arm == arm][["cond", "decoder"]].drop_duplicates()
        if len(dec) != 1:
            raise ValueError(f"{arm}: fits with {len(dec)} decoder settings")
        d = "--" if dec.decoder.iloc[0] == "na" else DEC_LABEL[int(dec.cond.iloc[0])]
        cells = [cov.loc[(arm, t)] for t in TASKS]
        cells.append(cov.loc[arm].sum())
        body.append(f"{label} & {d} & " + " & ".join(
            n(c.scored) if c.scored == c.expected else f"{n(c.scored)} of {n(c.expected)}" for c in cells) + r" \\")
    allb = [cov.xs(t, level="task").sum() for t in TASKS]
    allb.append(cov.sum())
    body += [r"\midrule", "Baselines, all & & " + " & ".join(
        n(c.scored) if c.scored == c.expected else f"{n(c.scored)} of {n(c.expected)}" for c in allb) + r" \\"]
    missing = "; ".join(f"{r.missing} of {dict((a, x) for a, x, _ in TOOLS)[r.arm]} on {TASK_LABEL[r.task]}"
                        for r in coverage.itertuples() if r.missing)
    num = r">{\raggedleft\arraybackslash}p{0.074\textwidth}"
    return "\n".join([
        HEAD, OPEN % "S5", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}" + P(0.12) + P(0.1) + num * len(TASKS) + "r@{}}",
        r"\caption{Run counts (Supplementary Note 11): fits per method, decoder and task. Sweep: the fits read by the sweep "
        r"analysis (the ten strengths, the edge strengths and the strength-extension fits on the immune task, Supplementary "
        r"Note 4); each was fitted and scored, and none failed (Supplementary Note 11). Baselines: fits scored, with the "
        r"planned number where it differs (``55 of 56''); -- for tools without a decoder; not fitted: " + (missing or "none")
        + r" (Supplementary Note 8; Supplementary Table S4 gives the settings). The input and standardisation pilots, the "
        r"replication across training seeds and the stopped fixed-strength benchmark are counted in Supplementary Note 11, "
        r"the replication per penalty in Supplementary Table S9.}\label{tab:s5}\\",
        r"\toprule Method & Decoder & " + " & ".join(TASK_LABEL[t] for t in TASKS) + r" & All \\", r"\midrule",
        *body, r"\bottomrule", r"\end{longtable}", "}", ""])


def build(data=DATA):
    """Step 2: {file name: LaTeX text} of the four tables from the collected inputs."""
    data = Path(data)
    tasks = pd.read_csv(data / "tasks.csv", keep_default_na=False).set_index("task").loc[TASKS].reset_index()
    manifest = pd.read_csv(data / "manifest.tsv", sep="\t")
    settings = pd.read_csv(data / "settings.csv", keep_default_na=False)
    coverage = pd.read_csv(data / "baseline_coverage.csv")
    return {"tableS1.tex": table_s1(tasks), "tableS2.tex": table_s2(tasks, manifest, settings),
            "tableS4.tex": table_s4(manifest, settings, coverage), "tableS5.tex": table_s5(manifest, coverage)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--collect", action="store_true", help="step 1: read the data root and write the inputs")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--data", default=str(DATA), help="collected inputs (default paper/figure_data/supp_methods)")
    ap.add_argument("--out", default=str(OUT), help="folder of the tables (default paper/supp)")
    ap.add_argument("--check", action="store_true", help="exit 1 if a committed table differs from the regenerated one")
    a = ap.parse_args(argv)
    if a.collect:
        print("wrote", collect(a.data_root, a.data))
        return 0
    texts, out = build(a.data), Path(a.out)
    if a.check:
        bad = [f for f, t in texts.items() if not (out / f).is_file() or (out / f).read_text() != t]
        print("tables:", ("differ: " + ", ".join(bad)) if bad else "identical")
        return 1 if bad else 0
    for f, t in texts.items():
        (out / f).write_text(t)
    print("wrote", ", ".join(texts))
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry; main() is tested
    sys.exit(main())
