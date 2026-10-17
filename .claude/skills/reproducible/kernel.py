"""Helpers for the `reproducible` skill: git-based provenance records, seed scans, a traceability matrix and a
SHA-256 manifest. Standard library only. See SKILL.md ## Audit workflow.
"""
import os
import re
import csv
import json

SEED_PATTERNS = (
    r"random_state\s*=\s*\d+",
    r"np\.random\.seed\s*\(\s*\d+\s*\)",
    r"random\.seed\s*\(\s*\d+\s*\)",
    r"torch\.manual_seed\s*\(\s*\d+\s*\)",
    r"tf\.random\.set_seed\s*\(\s*\d+\s*\)",
    r"set_seed\s*\(\s*\d+\s*\)",
    r"\bseed\s*=\s*\d+",
    r"default_rng\s*\(\s*\d+\s*\)",
)

TRACE_COLUMNS = ("claim", "value", "artifact_path", "source", "verify")


def scan_seeds(code):
    """Return RNG-seed / random_state lines found in `code`.

    A stochastic result that pins no seed is an audit red flag: it cannot be
    re-derived exactly. Use this to decide whether a provenance record should
    warn the reviewer.

    Parameters
    ----------
    code : str
        Source code (e.g. an artifact's reproduction code from lineage).

    Returns
    -------
    list of str
        Distinct stripped source lines that set an RNG seed. Empty if none.

    Example
    -------
    >>> scan_seeds("model = RF(random_state=0)\\nx = 1")
    ['model = RF(random_state=0)']
    >>> scan_seeds("x = compute(data)")
    []
    """
    if not code:
        return []
    combined = re.compile("|".join(SEED_PATTERNS))
    hits = []
    for line in code.splitlines():
        if combined.search(line):
            s = line.strip()
            if s not in hits:
                hits.append(s)
    return hits


def _git(args, root="."):
    import subprocess
    r = subprocess.run(["git"] + args, cwd=root, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def provenance_record(path, script=None, root="."):
    """Build a human-readable provenance dict for one result file from git and the file itself.

    Parameters
    ----------
    path : str
        Result file (repo-relative or absolute) whose value a claim rests on.
    script : str, optional
        The script that wrote it (repo-relative). Its text is scanned for RNG seeds.
    root : str
        Repository root (default current directory).

    Returns
    -------
    dict
        Keys: ``path``, ``sha256``, ``commit`` (last commit touching the file, '' if untracked), ``dirty``
        (uncommitted changes to the file), ``script``, ``script_commit``, ``environment`` (env spec files found),
        ``seeds`` (list of str), ``code`` (script text), ``seed_warning`` (stochastic code with no seed).

    Example
    -------
    >>> import tempfile, os
    >>> d = tempfile.mkdtemp(); open(os.path.join(d, "r.csv"), "w").write("a\\n1\\n") and None
    >>> rec = provenance_record("r.csv", root=d)
    >>> len(rec["sha256"]), rec["commit"]
    (64, '')
    """
    import glob, hashlib
    fp = path if os.path.isabs(path) else os.path.join(root, path)
    with open(fp, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    rel = os.path.relpath(fp, root)
    commit = _git(["log", "-1", "--format=%H", "--", rel], root)
    dirty = bool(_git(["status", "--porcelain", "--", rel], root)) if commit else False
    code, scommit = "", ""
    if script:
        sp = script if os.path.isabs(script) else os.path.join(root, script)
        with open(sp) as fh:
            code = fh.read()
        scommit = _git(["log", "-1", "--format=%H", "--", os.path.relpath(sp, root)], root)
    envs = [os.path.relpath(f, root) for pat in ("conda-lock.yml", "environment.y*ml", "envs/*.yml", "pyproject.toml")
            for f in sorted(glob.glob(os.path.join(root, pat)))]
    seeds = scan_seeds(code)
    stochastic = bool(re.search(r"random|sample|shuffle|rng|stochastic|bootstrap|permut", code, re.I))
    return {"path": rel, "sha256": sha, "commit": commit, "dirty": dirty, "script": script or "",
            "script_commit": scommit, "environment": envs, "seeds": seeds, "code": code,
            "seed_warning": stochastic and not seeds}


def render_provenance_md(records):
    """Render one provenance record (or a list) as reviewer-facing markdown.

    Example
    -------
    >>> md = render_provenance_md({"path": "results/corr.csv", "sha256": "ab", "commit": "c1", "dirty": False,
    ...     "script": "src/run.py", "script_commit": "c0", "environment": [], "seeds": [], "code": "x=1",
    ...     "seed_warning": False})
    >>> "## results/corr.csv" in md
    True
    """
    if isinstance(records, dict):
        records = [records]
    out = ["# Provenance records\n",
           "_Each result below lists the file hash, the commit, the script, the environment files and the seeds "
           "behind it. Generated from git and the files themselves._\n"]
    for r in records:
        out.append(f"## {r.get('path')}\n")
        out.append(f"- **sha256:** `{r.get('sha256')}`")
        out.append(f"- **last commit touching it:** `{r.get('commit') or 'untracked'}`"
                   + ("  (WARNING: uncommitted changes)" if r.get("dirty") else ""))
        out.append(f"- **script:** `{r.get('script') or 'not given'}` (commit `{r.get('script_commit') or 'n/a'}`)")
        envs = r.get("environment") or []
        out.append("- **environment files:** " + (", ".join(f"`{e}`" for e in envs) if envs else "none found"))
        if r.get("seeds"):
            out.append("- **seeds:** " + ", ".join(f"`{s}`" for s in r["seeds"]))
        elif r.get("seed_warning"):
            out.append("- **seeds:** WARNING none pinned, but the code looks stochastic; this result may not re-derive exactly.")
        else:
            out.append("- **seeds:** none (result appears deterministic)")
        code = r.get("code") or ""
        if code:
            out.append("\n<details><summary>script text</summary>\n")
            out.append("```python\n" + code.rstrip() + "\n```\n</details>\n")
    return "\n".join(out) + "\n"


def write_traceability(rows, path="audit/traceability.csv"):
    """Write a claim->artifact traceability matrix to CSV.

    Canonical columns: claim, value, artifact_path, source, verify.
    Extra keys in a row are appended as additional columns. Creates parent
    directories as needed.

    Parameters
    ----------
    rows : list of dict
        One dict per load-bearing claim. ``value`` should be read FROM the
        result file, never retyped from prose.
    path : str
        Output CSV path. Defaults to ``audit/traceability.csv``.

    Returns
    -------
    str
        The path written.

    Example
    -------
    >>> import tempfile, os, csv
    >>> p = os.path.join(tempfile.mkdtemp(), "t.csv")
    >>> write_traceability([{"claim": "r", "value": "0.83",
    ...     "artifact_path": "results/t.csv", "source": "src/run.py @ abc1234",
    ...     "verify": "python audit/verify.py --check corr"}], p)  # doctest: +ELLIPSIS
    '...t.csv'
    >>> next(csv.reader(open(p)))
    ['claim', 'value', 'artifact_path', 'source', 'verify']
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    extra = []
    for r in rows:
        for k in r:
            if k not in TRACE_COLUMNS and k not in extra:
                extra.append(k)
    header = list(TRACE_COLUMNS) + extra
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    return path

def write_manifest(root=".", path=None):
    """Write a ``MANIFEST.sha256`` of every file under ``root``.

    Lets a reviewer confirm nothing drifted since the audit: each line is
    ``<sha256>  <relpath>``, sorted by path, standard ``sha256sum -c`` format.
    Skips the manifest itself, VCS internals, and Python/build junk.

    Parameters
    ----------
    root : str
        Directory to walk. Defaults to the current directory.
    path : str, optional
        Output path. Defaults to ``<root>/MANIFEST.sha256``.

    Returns
    -------
    str
        The path written.

    Example
    -------
    >>> import tempfile, os
    >>> d = tempfile.mkdtemp()
    >>> _ = open(os.path.join(d, "a.txt"), "w").write("hello")
    >>> p = write_manifest(d)
    >>> line = open(p).read().strip()
    >>> line.split("  ")[1]
    'a.txt'
    >>> len(line.split("  ")[0])
    64
    """
    import hashlib
    if path is None:
        path = os.path.join(root, "MANIFEST.sha256")
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
                 ".mypy_cache", ".ruff_cache"}
    manifest_abs = os.path.abspath(path)
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs
                       and not d.endswith(".egg-info")]
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if os.path.abspath(fp) == manifest_abs:
                continue
            if fn.endswith((".pyc", ".pyo")):
                continue
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            rel = os.path.relpath(fp, root)
            lines.append(f"{h.hexdigest()}  {rel}")
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return path

