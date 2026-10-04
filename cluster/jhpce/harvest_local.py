#!/usr/bin/env python
"""Verify one JHPCE production-job harvest and merge it into the local durable OUT_DIR (all-or-nothing).

Input (--parts-dir): the files written by `prod_helpers.py pack` on fastscratch and downloaded part by part
(c.download caps at 256 MiB per file): <base>.tar.partNNN, <base>.SHA256SUMS, <base>.harvest_manifest.json.
Checks, before anything is written to --dest: every part's SHA-256, the reassembled tar's SHA-256, the manifest's
SHA-256, safe member names, and every extracted file's size and SHA-256 against the manifest (no missing or extra
files). The runner's views are those the pack recorded from the runner's own output: out/ledger/ must hold exactly
the manifest's ledger_files, all named <stage_key>.{csv,json,done} (a --tags-file run's own key, read from the
runner's '[stage]' line, never rebuilt), the ledger JSON must name that stage, the job's tags-file hash and the
recorded runner id, and a .done marker must come from the run that wrote the ledger. Merge of the tar's out/ tree
into --dest (the runner layout: latents/, models/, status/, attempts/, logs/, manifests/, quarantine/): new files are
copied, identical files skipped, an attempts/<tag>.jsonl that grew (the old content is a prefix) is replaced; any
other difference is a conflict and nothing is merged. The views (ledger/, failures.csv) and the job's witness files
(job/) are kept under --dest/_harvests/<base>/, not merged: the local scoring run (scripts/run_stage.py on --dest)
rebuilds its own views from the per-tag files.
Refuses a --dest inside a 'tier12' directory (the local A1 tree is off limits).

Usage: python cluster/jhpce/harvest_local.py --parts-dir DIR --dest OUT_DIR [--verify-only]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fail(msg, code=1):
    print(f"[harvest] REFUSED: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def load_sums(path):
    sums = {}
    for line in open(path):
        if line.strip():
            h, name = line.strip().split(None, 1)
            sums[name.strip()] = h
    return sums


def verify(parts_dir):
    sums_f = [f for f in os.listdir(parts_dir) if f.endswith(".SHA256SUMS")]
    if len(sums_f) != 1:
        fail(f"expected one *.SHA256SUMS in {parts_dir}, found {sums_f}")
    base = sums_f[0][: -len(".SHA256SUMS")]
    sums = load_sums(os.path.join(parts_dir, sums_f[0]))
    parts = sorted(n for n in sums if n.startswith(f"{base}.tar.part"))
    if not parts:
        fail("SHA256SUMS lists no parts")
    on_disk = sorted(f for f in os.listdir(parts_dir) if f.startswith(f"{base}.tar.part"))
    if on_disk != parts:
        fail(f"parts on disk {on_disk} != parts in SHA256SUMS {parts}")
    for n in parts:
        if sha256_file(os.path.join(parts_dir, n)) != sums[n]:
            fail(f"{n}: SHA-256 mismatch (download corrupted or incomplete)")
    tar_path = os.path.join(parts_dir, f"{base}.tar")
    h = hashlib.sha256()
    with open(tar_path + ".tmp", "wb") as out:
        for n in parts:
            with open(os.path.join(parts_dir, n), "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
                    out.write(b)
    if h.hexdigest() != sums.get(f"{base}.tar"):
        os.remove(tar_path + ".tmp")
        fail(f"reassembled {base}.tar SHA-256 {h.hexdigest()} != {sums.get(base + '.tar')}")
    os.replace(tar_path + ".tmp", tar_path)
    man_f = os.path.join(parts_dir, f"{base}.harvest_manifest.json")
    if sha256_file(man_f) != sums.get(f"{base}.harvest_manifest.json"):
        fail("harvest_manifest.json SHA-256 mismatch")
    man = json.load(open(man_f))
    return base, tar_path, man, sums


def extract(tar_path, staging, man):
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    want = {f["path"]: f for f in man["files"]}
    seen = set()
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            name = os.path.normpath(m.name)
            if os.path.isabs(name) or name.startswith(".."):
                fail(f"unsafe member name {m.name!r}")
            if not m.isfile():
                fail(f"member {m.name!r} is not a regular file")
            if name == "harvest_manifest.json":
                continue
            if name not in want:
                fail(f"member {name} is not listed in the harvest manifest")
            tf.extract(m, staging, set_attrs=False)
            p = os.path.join(staging, name)
            if os.path.getsize(p) != want[name]["bytes"] or sha256_file(p) != want[name]["sha256"]:
                fail(f"{name}: size/SHA-256 differs from the harvest manifest")
            seen.add(name)
    missing = sorted(set(want) - seen)
    if missing:
        fail(f"{len(missing)} manifest files missing from the tar: {missing[:5]}")
    return sorted(seen)


def check_views(staging, names, man):
    """The ledger files in the tar are exactly those the pack recorded for the runner's stage key, and agree with it."""
    for k in ("stage_key", "ledger_files", "ledger_runner_id", "ledger_final", "tags_sha256"):
        if k not in man:
            fail(f"harvest manifest lacks {k!r}: packed by a prod_helpers.py older than the runner's stage keys; repack")
    key, in_tar = man["stage_key"], sorted(n for n in names if n.startswith("out/ledger/"))
    if in_tar != sorted(man["ledger_files"]):
        fail(f"ledger files in the tar {in_tar} != those recorded by the pack {sorted(man['ledger_files'])}")
    if key is None:
        if in_tar:
            fail(f"ledger files {in_tar} but no stage key recorded")
        return dict(stage_key=None)
    named = {f"out/ledger/{key}.{ext}" for ext in ("csv", "json", "done")}
    other = sorted(set(in_tar) - named)
    if other:
        fail(f"ledger files {other} are not named after the recorded stage key {key}")
    info = dict(stage_key=key, final=None, gate_ok=None, counts=None)
    jp = f"out/ledger/{key}.json"
    if jp in in_tar:
        with open(os.path.join(staging, jp)) as f:
            L = json.load(f)
        if L.get("stage") != key:
            fail(f"{jp} names stage {L.get('stage')!r}, the pack recorded {key!r}")
        if (L.get("tags_file") or {}).get("sha256") != man["tags_sha256"]:
            fail(f"{jp} was written for tags file {(L.get('tags_file') or {}).get('sha256')}, the job's is {man['tags_sha256']}")
        if L.get("runner_id") != man["ledger_runner_id"]:
            fail(f"{jp} runner id {L.get('runner_id')} != recorded {man['ledger_runner_id']}")
        info.update(final="gate" in L, gate_ok=(L.get("gate") or {}).get("ok"), counts=L.get("counts"))
        dp = f"out/ledger/{key}.done"
        if dp in in_tar:
            with open(os.path.join(staging, dp)) as f:
                D = json.load(f)
            if D.get("runner_id") != L.get("runner_id") or not (D.get("gate") or {}).get("ok"):
                fail(f"{dp} was not written by the passing run that wrote {jp}")
    elif f"out/ledger/{key}.done" in in_tar:
        fail(f"a .done marker without the ledger JSON of {key}")
    return info


def plan_merge(staging, names, dest):
    """(copy, replace, skip, conflicts) for the out/ tree; views and job files go to _harvests/."""
    copy, replace, skip, conflicts = [], [], [], []
    for n in names:
        if not n.startswith("out/"):
            continue
        rel = n[4:]
        if rel.startswith("ledger/") or rel == "failures.csv" or rel.startswith("claims/"):
            continue
        src, dst = os.path.join(staging, n), os.path.join(dest, rel)
        if not os.path.exists(dst):
            copy.append(rel)
        elif sha256_file(dst) == sha256_file(src):
            skip.append(rel)
        elif rel.startswith("attempts/") and open(src, "rb").read().startswith(open(dst, "rb").read()):
            replace.append(rel)
        else:
            conflicts.append(rel)
    return copy, replace, skip, conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    dest = os.path.realpath(a.dest)
    if "tier12" in dest.split(os.sep):
        fail(f"--dest {dest} is inside a 'tier12' directory (the local A1 tree is off limits)")
    base, tar_path, man, sums = verify(a.parts_dir)
    staging = os.path.join(a.parts_dir, f"{base}.extracted")
    names = extract(tar_path, staging, man)
    views = check_views(staging, names, man)
    copy, replace, skip, conflicts = plan_merge(staging, names, dest)
    rec = dict(base=base, slurm_job_id=man.get("slurm_job_id"), stage=man.get("stage"), stage_key=man["stage_key"],
               ledger_files=man["ledger_files"], ledger=views,
               tar_sha256=sums[f"{base}.tar"], n_files=len(names), n_latents=man.get("n_latents"), copy=len(copy),
               replace=len(replace), skip=len(skip), conflicts=conflicts, dest=dest, verified_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if conflicts:
        print(json.dumps(rec, indent=1))
        fail(f"{len(conflicts)} files differ from the copies already in {dest} (nothing merged): {conflicts[:5]}")
    if a.verify_only:
        print(json.dumps(dict(rec, merged=False), indent=1))
        return
    for rel in copy + replace:
        src, dst = os.path.join(staging, "out", rel), os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.harvest_tmp{os.getpid()}"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    keep = os.path.join(dest, "_harvests", base)
    os.makedirs(keep, exist_ok=True)
    for n in names:
        if n.startswith("job/") or n.startswith("out/ledger/") or n == "out/failures.csv":
            dst = os.path.join(keep, n)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(staging, n), dst)
    for f in (f"{base}.SHA256SUMS", f"{base}.harvest_manifest.json"):
        shutil.copyfile(os.path.join(a.parts_dir, f), os.path.join(keep, f))
    rec["merged"] = True
    with open(os.path.join(keep, "receipt.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    shutil.rmtree(staging)
    os.remove(tar_path)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
