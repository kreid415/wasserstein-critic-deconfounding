#!/usr/bin/env python
"""Sinkhorn redo, stage 2 (SI-57): run the pre-registered rule scripts on explicit, de-duplicated file lists, and plan
which rows still need a fit. Nothing under scripts/ or src/ is changed (frozen by tag prereg-tier12-v3).

Why a wrapper: tier12_v2 stage folders still hold the pre-fix Sinkhorn outputs (made at c88cce3, SI-51/SI-54) next to the
non-Sinkhorn outputs that stay valid. The rule scripts refuse a tag scored twice (prereg_rules.outcomes), so each rule step
is given one score file per manifest tag, chosen by name: from --new-dir folders (v3 outputs) always, from --old-dir
folders (v2 outputs) only for arms other than sinkhorn. A tag found in two folders is an error, never a pick by position.
Every rule step writes into a NEW --out-dir (refused if it exists) so no record or manifest is overwritten, and writes
step.json with the exit code, the commit, the input file counts and the SHA-256 of every input list and output.

Usage (env with pandas, numpy, scipy; scvi-api has them):
  python stage2_rules.py r1     --checkout CO --expect-commit SHA --out-dir OUT --manifest M [--manifest EXT ...] \
        --old-dir tier12_v2/A1 [...] --new-dir tier12_v3/A1_sinkhorn [...]
  python stage2_rules.py r4a1   ... (same) --r1-record OUT_R1/r1_record.json
  python stage2_rules.py r2r3   ... (same) --r4a1-record OUT_R4/r4_a1_record.json
  python stage2_rules.py plan-fits --manifest M --experiments A2 A3 [--tasks T ...] [--old-dir D] --out-tags FILE \
        [--changes CHANGES.json]   (moved-window rows are listed and checked against the diff-records output)
  python stage2_rules.py diff-records --old OLD.json --new NEW.json --out CHANGES.json   (decisions only, SI-87 list)
Exit status of r1 / r4a1: 0 = frozen, 3 = extension needed (a normal outcome: see the extension manifest in OUT),
anything else = error. plan-fits: 0 = tags file written.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd

EXCLUDE_OLD_ARMS = ("sinkhorn",)       # EQUIVALENT_COMMITS.json not_for_arms: their v2 outputs are refused (SI-54)
META_KEYS = ("experiment", "task", "arm", "cond", "seed")   # score columns that must equal the manifest row's (with lam)
SCRIPT = {"r1": "freeze_a1_grid.py", "r4a1": "freeze_matched_lambda.py", "r2r3": "decide_a2_a3.py"}
EXPERIMENTS = {"r1": ["A1"], "r4a1": ["A1"], "r2r3": ["A1", "A2", "A3"]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_manifests(paths):
    M = pd.concat([pd.read_csv(p, sep="\t", comment="#", dtype=str, keep_default_na=False) for p in paths],
                  ignore_index=True)
    dup = M.tag[M.tag.duplicated()].tolist()
    if dup:
        raise SystemExit(f"duplicate manifest tags across --manifest files: {dup[:5]}")
    return M


def check_checkout(checkout, expect):
    git = ["git", "-C", checkout]
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != expect:
        raise SystemExit(f"checkout {checkout} is at {head}, expected {expect}")
    dirty = subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"checkout {checkout} has uncommitted changes (records would say git_dirty true)")
    return head


def settings_differ(path, row):
    """Names of the settings a score file records (the scorer's META columns, run_stage.py launch()) that differ from the
    manifest row it would be used for. lam is compared as a number. A2/A3 tags keep their placeholder name when R4-a1
    moves a window, so the tag alone cannot show that a score was fitted at the row's penalty strength (SI-57).
    adv_input and zstd are not recorded in a score file; they are fixed per A2/A3 cell and R4-a1 does not touch them."""
    S = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("tag", "lam") + META_KEYS if c not in S.columns]
    if missing or len(S) != 1:
        raise SystemExit(f"{path}: expected one score row with the recorded settings {missing or ''} (rows: {len(S)}); "
                         f"cannot show it was fitted at the manifest row's settings")
    r = S.iloc[0]
    diff = [c for c in ("tag",) + META_KEYS if r[c] != row[c]]
    try:
        same_lam = float(r["lam"]) == float(row["lam"])
    except ValueError:                                        # a placeholder or non-numeric lambda is never a match
        same_lam = False
    return diff + ([] if same_lam else ["lam"])


def select_scores(rows, old_dirs, new_dirs):
    """One score file per manifest row: new dirs for every arm, old dirs for arms not excluded, and only an old file
    fitted at the row's own settings. Returns (files, skipped, stale): skipped = old Sinkhorn files left out; stale = tags
    whose old file records other settings (a moved R4-a1 window) and so needs a new file. A new-dir file whose recorded
    settings differ from the row is an error."""
    files, skipped, stale = [], 0, []
    for row in rows.to_dict("records"):
        tag = row["tag"]
        new = [p for p in (os.path.join(d, "scores", f"{tag}.csv") for d in new_dirs) if os.path.isfile(p)]
        old = [p for p in (os.path.join(d, "scores", f"{tag}.csv") for d in old_dirs) if os.path.isfile(p)]
        if row["arm"] in EXCLUDE_OLD_ARMS:
            skipped += len(old)
            old = []
        for p in new:
            diff = settings_differ(p, row)
            if diff:
                raise SystemExit(f"{p}: recorded settings differ from the manifest row of {tag} in {diff}")
        fitted_here = [p for p in old if not settings_differ(p, row)]
        if old and not fitted_here:
            stale.append(tag)
        old = fitted_here
        found = new + old
        if len(found) > 1:
            raise SystemExit(f"tag {tag} has score files in more than one folder: {found}")
        files += found
    return files, skipped, stale


def merge_failures(rows, old_dirs, new_dirs, path):
    """failures.csv of every folder, kept for the selected rows only (old folders: not for excluded arms)."""
    arm = dict(zip(rows.tag, rows.arm))
    parts = []
    for d, is_old in [(d, False) for d in new_dirs] + [(d, True) for d in old_dirs]:
        f = os.path.join(d, "failures.csv")
        if not os.path.isfile(f):
            raise SystemExit(f"{f} does not exist (every stage folder writes one, header only when there are no failures)")
        F = pd.read_csv(f, dtype=str, keep_default_na=False)
        keep = F.tag.isin(list(arm))
        if is_old:
            keep &= ~F.tag.map(arm).isin(EXCLUDE_OLD_ARMS)
        parts.append(F[keep])
    F = pd.concat(parts, ignore_index=True).fillna("")
    if F.tag.duplicated().any():
        raise SystemExit(f"a tag is in the failures of more than one folder: {F.tag[F.tag.duplicated()].tolist()[:5]}")
    F.to_csv(path, index=False)
    return F


def run_rule(a):
    head = check_checkout(a.checkout, a.expect_commit)
    os.makedirs(a.out_dir)                                   # refuses an existing folder: no overwrite
    M = read_manifests(a.manifest)
    rows = M[M.experiment.isin(EXPERIMENTS[a.cmd])]
    files, skipped, stale = select_scores(rows, a.old_dir, a.new_dir)
    lst = os.path.join(a.out_dir, "scores_list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(files) + "\n")
    fails = os.path.join(a.out_dir, "failures_merged.csv")
    F = merge_failures(rows, a.old_dir, a.new_dir, fails)
    n_fail = len(F)
    have = {os.path.basename(p)[:-len(".csv")] for p in files} | set(F.tag)
    missing = [t for t in rows.tag if t not in have]
    if missing:                                              # stop here, with the tags, before any rule script runs
        raise SystemExit(f"{len(missing)} of {len(rows)} rows have no score file and no failure row, e.g. {missing[:5]}; "
                         f"{len(set(missing) & set(stale))} of them have only a stale old file fitted at other settings "
                         f"(a moved R4-a1 window): they need a refit in a --new-dir folder")

    def o(name):
        return os.path.join(a.out_dir, name)
    cmd = [sys.executable, os.path.join(a.checkout, "scripts", SCRIPT[a.cmd])]
    for m in a.manifest:
        cmd += ["--manifest", m]
    cmd += ["--scores", *files, "--failures", fails]
    if a.cmd == "r1":
        cmd += ["--out-json", o("r1_record.json"), "--out-manifest", o("m.r1.tsv"),
                "--extension-manifest", o("a1_extension.tsv")]
    elif a.cmd == "r4a1":
        cmd += ["--stage", "a1", "--r1-record", a.r1_record, "--out-json", o("r4_a1_record.json"),
                "--out-manifest", o("m.r4a1.tsv"), "--extension-manifest", o("a1_extension.tsv")]
    else:
        cmd += ["--r4a1-record", a.r4a1_record, "--out-json", o("r2r3_record.json"), "--out-manifest", o("m.x1.tsv")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    outs = {n: sha256(o(n)) for n in sorted(os.listdir(a.out_dir))}
    rec = dict(step=a.cmd, exit_code=r.returncode, commit=head, n_rows=len(rows), n_score_files=len(files),
               n_old_sinkhorn_skipped=skipped, n_old_stale_skipped=len(stale), stale_tags=stale, n_failure_rows=n_fail, manifests={m: sha256(m) for m in a.manifest},
               outputs=outs)
    with open(o("step.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print(f"[stage2] {a.cmd} exit {r.returncode}: {len(files)} score files, {skipped} old Sinkhorn files skipped, "
          f"{len(stale)} stale old files skipped, {n_fail} failure rows; outputs {sorted(outs)}")
    return r.returncode


def plan_fits(a):
    """Tags file of the rows that still need a fit: every selected row except non-Sinkhorn rows with a v2 score file fitted
    at the row's own settings (task, arm, cond, seed, lam). An old file at other settings = a moved window: the row is refitted.
    A missing old failures.csv, or a v2 failure row of a selected non-Sinkhorn row, stops the plan before the tags file is written."""
    M = read_manifests(a.manifest)
    rows = M[M.experiment.isin(a.experiments)]
    if a.tasks:
        rows = rows[rows.task.isin(a.tasks)]
    if not len(rows):
        raise SystemExit(f"no manifest rows for {a.experiments} {a.tasks}")
    arm = dict(zip(rows.tag, rows.arm))
    for d in a.old_dir:                                      # a v2 failure of a non-Sinkhorn row needs a person's decision
        f = os.path.join(d, "failures.csv")
        if not os.path.isfile(f):
            raise SystemExit(f"{f} does not exist (every stage folder writes one, header only when there are no failures)")
        F = pd.read_csv(f, dtype=str, keep_default_na=False)
        bad = F.tag[F.tag.isin(list(arm)) & ~F.tag.map(arm).isin(EXCLUDE_OLD_ARMS)].tolist()
        if bad:
            raise SystemExit(f"{f} has failure rows for selected non-Sinkhorn tags {bad[:5]} ({len(bad)} in all): a person must "
                             f"decide whether that v2 failure stands or the row is refitted; no tags file was written")
    reuse, moved = [], []
    for row in rows.to_dict("records"):
        if row["arm"] in EXCLUDE_OLD_ARMS:
            continue
        old = [p for p in (os.path.join(d, "scores", f"{row['tag']}.csv") for d in a.old_dir) if os.path.isfile(p)]
        ok = [p for p in old if not settings_differ(p, row)]
        if len(ok) > 1:
            raise SystemExit(f"tag {row['tag']} has score files in more than one old folder: {ok}")
        if ok:
            reuse.append(row["tag"])
        elif old:                                            # an old file at other settings: the window moved (SI-57)
            moved.append(row)
    moved_tags = {r["tag"] for r in moved}
    cells = sorted({f"{r['task']}|{r['cond']}|{r['arm']}" for r in moved})
    if a.changes:
        changed = {c["key"].split(" ")[1] for c in json.load(open(a.changes))["changed"]
                   if len(c["key"].split(" ")) == 3 and c["key"].startswith("R4a1 ") and c["key"].endswith((" lo", " matched", " hi"))}
        unexplained = [c for c in cells if c not in changed]
        if unexplained:
            raise SystemExit(f"moved cells {unexplained} are not explained by the decision changes in {a.changes}: "
                             f"their old scores differ from the manifest for another reason; stop and investigate")
    todo = rows[~rows.tag.isin(reuse)]
    with open(a.out_tags, "w") as f:
        f.write(f"# rows of {a.experiments} that need a fit: {len(todo)} of {len(rows)} "
                f"({len(reuse)} reuse a v2 non-Sinkhorn score, {len(moved_tags)} moved: old score fitted at other settings, "
                f"cells {cells})\n")
        f.write("\n".join(todo.tag) + ("\n" if len(todo) else ""))
    print(f"[plan-fits] {len(todo)} of {len(rows)} rows need a fit, {len(reuse)} reused, {len(moved_tags)} moved "
          f"(cells {cells}); to fit by arm {todo.groupby(['experiment', 'arm']).size().to_dict()}; tags sha256 {sha256(a.out_tags)}")
    return 0


def decisions(rec):
    """Flat {key: value} of the decisions in an R1, R4-a1 or R2/R3 record: grids and flags, windows, the two switches."""
    if rec["rule"] == "R1":
        return {f"R1 {k} {f}": v[f] for k, v in rec["families"].items() for f in ("status", "x1_grid", "flags") if f in v}
    if rec["rule"] == "R4":
        return {f"R4a1 {k} {f}": c[f] for k, c in rec["cells"].items() for f in ("lo", "matched", "hi", "flags")}
    return {"R2R3 adv_input": rec["adv_input"], "R2R3 zstd": rec["zstd"],
            **{f"R2R3 {r} choice": rec["decisions"][r]["choice"] for r in ("R2", "R3")}}


def diff_records(a):
    old, new = (decisions(json.load(open(p))) for p in (a.old, a.new))
    keys = sorted(set(old) | set(new))
    ch = [dict(key=k, old=old.get(k), new=new.get(k)) for k in keys if old.get(k) != new.get(k)]
    sink = [c for c in ch if "sinkhorn" in c["key"]]
    with open(a.out, "w") as f:
        json.dump(dict(n_compared=len(keys), changed=ch, n_changed=len(ch), n_changed_non_sinkhorn=len(ch) - len(sink)), f, indent=1)
    print(f"[diff-records] {len(keys)} decisions compared, {len(ch)} changed, {len(ch) - len(sink)} of them not Sinkhorn")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("r1", "r4a1", "r2r3"):
        p = sub.add_parser(name)
        p.add_argument("--checkout", required=True)
        p.add_argument("--expect-commit", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--manifest", action="append", required=True)
        p.add_argument("--old-dir", action="append", default=[])
        p.add_argument("--new-dir", action="append", default=[])
        if name == "r4a1":
            p.add_argument("--r1-record", required=True)
        if name == "r2r3":
            p.add_argument("--r4a1-record", required=True)
    p = sub.add_parser("plan-fits")
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--old-dir", action="append", default=[])
    p.add_argument("--out-tags", required=True)
    p.add_argument("--changes", default=None, help="diff-records output: every moved cell must be a cell whose R4-a1 window changed")
    p = sub.add_parser("diff-records")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"plan-fits": plan_fits, "diff-records": diff_records}.get(a.cmd, run_rule)(a)


if __name__ == "__main__":  # pragma: no cover  (entry point; main() is exercised by the tests)
    sys.exit(main())
