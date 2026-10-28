#!/usr/bin/env python
"""One status board for the wcd runs: pilot fits, pilot scoring, main-benchmark (X1) fits and main-benchmark scoring.

Per group the board lists every EXPECTED unit (manifest rows, never files found), judges each output with a validity check
and reads Slurm accounting for explicit job ids (cluster/vendor/run_status_board.py, the run-status-board skill).
  python cluster/jhpce/status_board.py --config cluster/jhpce/board_config.json --out-dir DIR --sites jhpce [--import local=inv.json]
Validity: groups with check "full" (the pilot; not blinded) use scripts/run_stage.py's own output contract, from the checkout
named by the group's stage_repo (cluster/jhpce/chain_lib.py). Groups with check "presence" (the main benchmark, blinded) only
stat files (exist, non-empty): no main score file and no main ledger is ever opened.
A site is one place that holds output folders (jhpce, local). --sites says which sites this host can read; the other sites must
come in with --import SITE=file, a file written on that host with --export-inventory FILE. The first full pass hashes every
pilot latent (about 1,900 x 4 MB): run it as a 1-CPU job (cluster/jhpce/status_board_job.sh); later passes read only changed files.
"""
import argparse
import collections
import datetime
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vendor"))
import chain_lib  # noqa: E402
import run_status_board as RSB  # noqa: E402

SACCT_FMT = "JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,ReqMem,MaxRSS,Timelimit,AllocCPUS"


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    if not cfg.get("groups"):
        raise SystemExit(f"{path}: no groups")
    return cfg


def read_submissions(path):
    """tag -> 'jobid_index' from the chain supervisor's submissions.jsonl (empty before the first submission)."""
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            rec = json.loads(line)
            if rec.get("event") == "submitted":
                for i, tag in enumerate(rec["tags"], 1):
                    out[tag] = f"{rec['job_id']}_{i}"
    return out


def expected_rows(rs, g):
    rows = chain_lib.read_expected(rs, abspath(g["manifest"]), g["manifest_sha256"], g["experiments"],
                                   [abspath(t) for t in g.get("tags_files", [])])
    for r in rows:
        r["job"] = str(g.get("jobs", {}).get(r["task"], ""))
    return rows


def collect_site(g, rows, site, rs, state_dir):
    """{tag: [valid, reason, detail] or None} for one group at one site."""
    kind, out = g["kind"], g["sites"][site]
    res = {}
    if g["check"] == "full":
        ck = chain_lib.Checker(rs, out, g.get("score_sites", g["sites"])[site], g["expect_device"],
                               os.path.join(state_dir, f"validity_cache_{site}.json"))
        for r in rows:
            res[r["tag"]] = getattr(ck, kind)(r)
        ck.save()
        return res
    sub, ext = ("latents", "npz") if kind == "fit" else ("scores", "csv")
    for r in rows:
        p = os.path.join(out, sub, f"{r['tag']}.{ext}")
        if os.path.isfile(p) and os.path.getsize(p) > 0:      # stat only: a blinded main file is never opened
            res[r["tag"]] = [True, "presence only (not opened)", {"finished_at": chain_lib.iso_mtime(p)}]
    return res


def merge(site_results, tag):
    """First valid result over the sites, else the first invalid one, else None; a second valid copy is noted."""
    found = [(s, r[tag]) for s, r in site_results.items() if r.get(tag)]
    valid = [(s, r) for s, r in found if r[0]]
    if valid:
        s, r = valid[0]
        extra = [x for x, _ in valid[1:]]
        return (True, r[1] + (f" [site {s}; also at {extra}]" if extra else f" [site {s}]"), r[2])
    return tuple(found[0][1]) if found else None     # tuples: run-status-board rejects the lists a JSON cache gives


def capture_sched(ids, a):
    """(sacct text, squeue text) for explicit job ids, each with a #CAPTURED stamp (read-only)."""
    if a.sacct_file:
        return open(a.sacct_file).read(), (open(a.squeue_file).read() if a.squeue_file else None)
    if not ids or not shutil.which("sacct"):
        return None, None
    stamp = f"#CAPTURED {datetime.datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
    idl = ",".join(sorted(ids))
    sa = subprocess.run(["sacct", "-P", "-j", idl, "--format", SACCT_FMT], check=True, capture_output=True, text=True).stdout
    sq = subprocess.run(["squeue", "-h", "-j", idl, "-o", "%i|%j|%T|%M|%l|%V|%S|%R"], capture_output=True, text=True)
    sqt = stamp + "JOBID|NAME|STATE|TIME|TIME_LIMIT|SUBMIT_TIME|START_TIME|NODELIST(REASON)\n" + (sq.stdout if sq.returncode == 0 else "")
    return stamp + sa, sqt


def unit_sched(parsed, rows):
    """One Slurm job covers many units (a fit job: 378 tags), but run-status-board maps a scheduler row to one unit: give every
    unit a copy of the rows of its job (unit['job'] = 'jobid' or 'jobid_arrayindex')."""
    by_job = {}
    for p in parsed:
        for r in p["rows"]:
            by_job.setdefault(r["job_id"], []).append(r)
    return [{"rows": [dict(sr, unit=r["tag"]) for r in rows if r["job"] for sr in by_job.get(r["job"], [])],
             "captured_at": p["captured_at"], "warnings": [], "kind": p.get("kind", "scheduler")} for p in parsed if p]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.path.join(HERE, "board_config.json"))
    ap.add_argument("--out-dir", required=True, help="board, history and validity caches are written here")
    ap.add_argument("--sites", nargs="+", required=True, help="sites this host can read")
    ap.add_argument("--import", dest="imports", nargs="*", default=[], metavar="SITE=FILE")
    ap.add_argument("--export-inventory", help="write this host's per-site results to FILE (no board)")
    ap.add_argument("--sacct-file", help="saved sacct text with a #CAPTURED line (offline use, tests)")
    ap.add_argument("--squeue-file")
    ap.add_argument("--groups", nargs="*", help="only these groups")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    groups = {k: v for k, v in cfg["groups"].items() if not a.groups or k in a.groups}
    imported = {}
    for s in a.imports:
        site, path = s.split("=", 1)
        with open(path) as f:
            imported[site] = json.load(f)
    os.makedirs(a.out_dir, exist_ok=True)
    stage_mods, inv, rows_by = {}, {}, {}
    for name, g in groups.items():
        sites = set(g["sites"])
        blind = sorted(sites - set(a.sites) - set(imported))
        if blind and not a.export_inventory:
            raise SystemExit(f"group {name}: sites {blind} are neither readable here (--sites) nor imported (--import)")
        repo = abspath(g.get("stage_repo", "."))
        if repo not in stage_mods:
            stage_mods[repo] = chain_lib.load_run_stage(repo)
        rs = stage_mods[repo]
        rows_by[name] = expected_rows(rs, g)
        inv[name] = {}
        for site in sorted(sites):
            if site in a.sites:
                inv[name][site] = collect_site(g, rows_by[name], site, rs, a.out_dir)
            elif site in imported:
                inv[name][site] = imported[site][name][site]
    if a.export_inventory:
        with open(a.export_inventory, "w") as f:
            json.dump({n: {s: r for s, r in v.items() if s in a.sites} for n, v in inv.items()}, f)
        print(f"inventory of sites {a.sites} -> {a.export_inventory}")
        return 0
    ids, subs = set(), {}
    for name, g in groups.items():
        ids |= {str(v) for v in g.get("jobs", {}).values()}
        if g.get("submissions"):
            subs[name] = read_submissions(g["submissions"])
            ids |= {j.split("_")[0] for j in subs[name].values()}
    sa_text, sq_text = capture_sched(ids, a)
    sa = RSB.rs_parse_sacct(sa_text, strict=False) if sa_text else None
    sq = RSB.rs_parse_squeue(sq_text, strict=False) if sq_text else None
    for fn, txt in (("sacct.txt", sa_text), ("squeue.txt", sq_text)):
        if txt:
            with open(os.path.join(a.out_dir, fn), "w") as f:
                f.write(txt)
    out, md = {}, ["# wcd run status board", ""]
    for name, g in groups.items():
        rows = rows_by[name]
        if name in subs:
            for r in rows:
                r["job"] = subs[name].get(r["tag"], "")
        gids = {str(v) for v in g.get("jobs", {}).values()} | {j.split("_")[0] for j in subs.get(name, {}).values()}
        per_site = inv[name]
        # a unit with a valid output keeps no job row: a job the stop guard ended (FAILED, exit 15) after 250 good fits is not a conflict
        open_rows = [r for r in rows if not (merge(per_site, r["tag"]) or (False,))[0]]
        sched = unit_sched([p for p in (sa, sq) if p], open_rows) if gids and sa else None
        sched = sched if sched and any(p["rows"] for p in sched) else None
        jobs = collections.Counter((r["job_id"].split("_")[0], r["state"]) for r in (sa or {"rows": []})["rows"]
                                   if r["job_id"].split("_")[0] in gids)
        board = RSB.rs_board(rows, "tag", lambda r, p=per_site: merge(p, r["tag"]), sched, sched_key=lambda sr: sr["unit"], strict=bool(sched))
        board["validity_basis"] = "presence_only" if g["check"] == "presence" else board["validity_basis"]
        live = board["totals"]["running"] + board["totals"]["pending"]
        root = os.path.join(a.out_dir, name)
        eta = RSB.rs_eta(board, history=RSB.rs_history(root), strict=False) if live else None
        RSB.rs_save(board, root, eta)
        out[name] = {"title": g["title"], "board": board, "eta": eta}
        t = board["totals"]
        md += [f"## {g['title']} (`{name}`)", "", f"**{board['headline']}**", "",
               f"- validity: {'PRESENCE ONLY (blinded main files are not opened)' if g['check'] == 'presence' else 'run_stage output contract'}",
               "- ETA: " + (("none: " + eta["reason"]) if eta and eta["refused"] else
                            (f"{eta['eta_median']} ({eta['eta_low']} to {eta['eta_high']}), {eta['n_completed_used']} completions" if eta else
                             "not applicable: no live job (nothing running or pending)")),
               "- jobs (Slurm accounting, " + (sa["captured_at"] if sa else "none") + "): " +
               (", ".join(f"{j} {st}" + (f" x{n}" if n > 1 else "") for (j, st), n in sorted(jobs.items())) or "none"),
               f"- needs action: {t['failed'] + t['done_invalid']} units (failed {t['failed']}, done_invalid {t['done_invalid']}); "
               f"missing {t['missing']}; details in {name}/status_board.md", ""]
    md.append(f"_generated {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}; scheduler snapshot "
              f"{(sa or {}).get('captured_at') or 'none'}_")
    with open(os.path.join(a.out_dir, "status_board.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.out_dir, "status_board.json"), "w") as f:
        json.dump({"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "groups": out}, f, indent=1)
    print("\n".join(md))
    return 0


if __name__ == "__main__":  # pragma: no cover  # entry point
    sys.exit(main())
