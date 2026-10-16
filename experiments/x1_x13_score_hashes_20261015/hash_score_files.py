#!/usr/bin/env python3
"""SHA-256 list of every existing X1 and X13 score file (user decision SI-79, 2026-10-15; status records added 2026-10-16).

Reads each selected file in binary mode only to hash it; never decodes, parses or prints file
contents, so no score value is shown (SI-78: checks that print only counts or flags do not count
as examining a score). Prints counts per group and kind only.

Usage (standard library only; any Python >= 3.8):
    python experiments/x1_x13_score_hashes_20261015/hash_score_files.py \
        --data-root /home/kendall/experiment_data/wasserstein-critic-deconfounding
Writes score_file_hashes.csv and summary.json next to this script (or into --out-dir).
"""
import argparse, csv, datetime, hashlib, json, os, re, sys
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
TASKS = sorted(["atac_large", "atac_small", "immune_hum_mou", "immune", "lung", "pancreas", "sim1", "sim2"], key=len, reverse=True)
PASS_OF = {"atac_large": "A", "immune_hum_mou": "A", "lung": "B", "pancreas": "B", "sim2": "B"}   # filler scoring passes
TASK_RE = re.compile(r"^X1\d*_(" + "|".join(TASKS) + r")_")


def task_of(rel):
    """Task named in a file name that starts X1_ or X13_ (else 'n/a')."""
    m = TASK_RE.match(rel.split("/")[-1])
    return m.group(1) if m else "n/a"


def ny_time(mtime_utc):
    return datetime.datetime.strptime(mtime_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).astimezone(NY)

PRUNE_DIRS = {".git", "models", "latents", "__pycache__"}
PRUNE_TOP = {"code", "Rlib_kbet", "prepped_final", "prepped_scib", "data_final"}


def kind_of(rel):
    """Return the kind of a score-bearing file, or None if it is not selected."""
    parts = rel.split("/")
    name = parts[-1]
    if parts[0] in PRUNE_TOP or any("pytest" in p or p == "codex_work" for p in parts):
        return None
    if parts[0].startswith("embeddings"):
        return None
    if parts[0] == "review_sealed":
        return "sealed"
    if "status" in parts[:-1] and name.endswith(".json"):      # OUT/status/<tag>.json: the divergence and
        return "status_record"                                 # nonfinite_latent records the stage runner reads (SI-79, 2026-10-16)
    if rel.startswith("tier12_v2/logs/") and name.startswith("X1"):
        return "runner_log"
    if parts[0] == "jhpce_tier12" and len(parts) == 3 and parts[1] == "logs":
        return "runner_log"
    xname = name.startswith(("X1_", "X13_"))
    if any(p.startswith("scores") for p in parts[:-1]):
        if xname:
            return "score" if name.endswith((".csv", ".json", ".tsv", ".parquet")) else "scoring_log"
        if rel.startswith("review/identity_") and name.startswith("score_identity"):
            return "comparison"
        return None
    if "ledger" in parts[:-1] and name.startswith("X1"):
        return "ledger"
    if len(parts) >= 3 and parts[-3] == "logs" and parts[-2] == "score" and xname:
        return "scoring_log"
    if name == "failures.csv" and rel.startswith(("tier12_v2/X1_local/", "jhpce_tier12/fillers_x1_x13/")):
        return "runner_record"
    return None


def group_of(rel):
    for prefix, grp in (("tier12_v2/X1_local/", "X1_local"), ("tier12_v2/logs/", "X1_local"),
                        ("jhpce_tier12/", "fillers"), ("review/identity_", "identity_check"),
                        ("review/out/", "review_probes"), ("review/verify/", "review_probes"),
                        ("x13_cpu_baselines_atac_small_test/", "x13_cpu_test"), ("review_sealed/", "sealed")):
        if rel.startswith(prefix):
            return grp
    return "other"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="/home/kendall/experiment_data/wasserstein-critic-deconfounding")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    root = os.path.abspath(a.data_root)
    if not os.path.isdir(root):
        sys.exit("data root not found: " + root)
    rows = []
    for dp, dn, fn in os.walk(root):
        rel_dir = os.path.relpath(dp, root)
        top = rel_dir.split("/")[0]
        if rel_dir != "." and (top in PRUNE_TOP or top.startswith("embeddings")):
            dn[:] = []
            continue
        dn[:] = sorted(d for d in dn if d not in PRUNE_DIRS and "pytest" not in d and d != "codex_work")
        for f in sorted(fn):
            rel = f if rel_dir == "." else rel_dir + "/" + f
            kind = kind_of(rel)
            if kind is None:
                continue
            p = os.path.join(dp, f)
            st = os.stat(p)
            rows.append({"group": group_of(rel), "kind": kind, "path": rel, "bytes": st.st_size,
                         "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "sha256": sha256(p)})
    rows.sort(key=lambda r: (r["group"], r["path"]))
    if not rows:
        sys.exit("no file selected; wrong --data-root?")
    os.makedirs(a.out_dir, exist_ok=True)
    out_csv = os.path.join(a.out_dir, "score_file_hashes.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "kind", "path", "bytes", "mtime_utc", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    counts = Counter((r["group"], r["kind"]) for r in rows)
    by = Counter((r["group"], r["kind"], task_of(r["path"]), ny_time(r["mtime_utc"]).strftime("%Y-%m-%d")) for r in rows)
    fill = defaultdict(Counter)
    for r in rows:
        if r["group"] == "fillers" and r["kind"] == "score":
            fill["pass " + PASS_OF.get(task_of(r["path"]), "?")][ny_time(r["mtime_utc"]).strftime("%Y-%m-%d")] += 1
    ident = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["group"] == "identity_check":
            ident[r["kind"]][ny_time(r["mtime_utc"]).strftime("%Y-%m-%d")].append(ny_time(r["mtime_utc"]))
    ident = {k: {day: {"files": len(v), "first_mtime_ny": min(v).strftime("%Y-%m-%d %H:%M:%S %Z"),
                       "last_mtime_ny": max(v).strftime("%Y-%m-%d %H:%M:%S %Z")} for day, v in sorted(dd.items())}
             for k, dd in sorted(ident.items())}
    summary = {"data_root": root, "files": len(rows),
               "hashed_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "counts": {g + "/" + k: n for (g, k), n in sorted(counts.items())},
               "counts_by_group_kind_task_date_ny": [{"group": g, "kind": k, "task": t, "date_ny": dt, "files": n} for (g, k, t, dt), n in sorted(by.items())],
               "filler_score_files_per_pass_and_date_ny": {k: dict(sorted(v.items())) for k, v in sorted(fill.items())},
               "identity_check_files_per_kind_and_date_ny": ident,
               "csv_sha256": sha256(out_csv), "script_sha256": sha256(os.path.abspath(__file__))}
    with open(os.path.join(a.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for key, n in summary["counts"].items():
        print(key, n)
    print("total", len(rows), "csv sha256", summary["csv_sha256"][:16])


if __name__ == "__main__":
    main()
