"""Audit: were X1 or X13 score values displayed? (counts and digit-masked lines only; no value is shown)

Two parts, both print counts and digit-masked lines only (SI-78: checks that print counts or flags do
not count as examining a score):
- scan_files(data_root, id_re, folders, listed): log, text, markdown and JSON files under review/out and
  review/verify and, since the redo of 2026-10-16, in every folder that holds a listed file (not recursive),
  excluding the listed files themselves (never printed);
- scan_transcripts(host, project_id, id_re, folders, since_ms, counts, skip): the app's stored session
  transcripts (table frame_messages), every message of every session of the project active since `since_ms`,
  up to the message count `counts` recorded at the cut-off. Runs only inside the Claude Science control-plane
  kernel (`repl`), where `host.query` exists.
Identifiers (redo of 2026-10-16) = the file stems of EVERY file in score_file_hashes.csv (all groups and kinds);
folders = every folder that holds a listed file. A flagged line is one that holds an identifier and >= 3
decimal numbers, or a metric name and >= 2 decimal numbers inside a block that mentions an identifier or a
folder. Identifiers match as whole words. classify() sorts each flagged line into a known class or
'unclassified'. The stored output of the run is audit_displayed_values.json (the run of 2026-10-15 is in git
history, commit eaf6cdf).
"""
import csv, datetime, json, os, re
from collections import Counter
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

DEC = re.compile(r'(?<![\w.])-?\d+\.\d{2,}(?![\w.])')
METRIC = re.compile(r'(?i)(ilisi|clisi|kbet|asw|nmi|ari\b|pcr|graph_conn|isolated|silhouette|batch_score|bio_score|overall|scib|cell_cycle)')
TEXT_EXT = ('.log', '.txt', '.md', '.jsonl', '.json', '.out', '.err')


def mask(line):
    return re.sub(r'\d', '#', line)


def norm(text):
    """Turn escaped \\n, \\t, \\r and \\" (JSON strings, stored transcripts) into plain separators so that whole-word matching works."""
    return text.replace("\\n", "\n").replace("\\t", " ").replace("\\r", " ").replace('\\"', '"')


def load_inventory(csv_path):
    """(identifiers, folders, listed paths) from score_file_hashes.csv; paths are relative to the data root."""
    with open(csv_path, newline='') as fh:
        paths = [r['path'] for r in csv.DictReader(fh)]
    stem = lambda p: p.split('/')[-1].rsplit('.', 1)[0]
    return sorted({stem(p) for p in paths}), sorted({os.path.dirname(p) for p in paths}), set(paths)


def make_id_re(idents):
    """Whole-word match. Stems that start X1/X13 may touch '_' on either side; all others must not."""
    alt = lambda xs: '|'.join(re.escape(t) for t in sorted(xs, key=len, reverse=True))
    xs = [t for t in idents if re.match(r'X1\d*[_+]', t)]
    rest = [t for t in idents if t not in set(xs)]
    return re.compile(r'(?<![A-Za-z0-9])(?:' + alt(xs) + r')(?![A-Za-z0-9])|(?<![A-Za-z0-9_])(?:' + alt(rest) + r')(?![A-Za-z0-9_])')


def flagged_lines(text, id_re, folders):
    out = []
    text = norm(text)
    if not (id_re.search(text) or any(f in text for f in folders)):
        return out
    for ln in text.split('\n'):
        m = id_re.search(ln)
        if (m and len(DEC.findall(ln)) >= 3) or (METRIC.search(ln) and len(DEC.findall(ln)) >= 2):
            out.append((m.group(0) if m else None, mask(ln.strip())[:230]))
    return out


def scan_files(data_root, id_re, folders, listed):
    res = {"files_scanned": 0, "files_mentioning": 0, "files_scanned_outside_review": 0, "flagged": []}
    seen = set()

    def one(p, rel):
        if rel in listed or rel in seen or not rel.endswith(TEXT_EXT):
            return
        seen.add(rel)
        res["files_scanned"] += 1
        if not rel.startswith(("review/out/", "review/verify/")):
            res["files_scanned_outside_review"] += 1
        txt = norm(open(p, errors='ignore').read())
        if id_re.search(txt):
            res["files_mentioning"] += 1
        for ident, ln in flagged_lines(txt, id_re, folders):
            res["flagged"].append({"file": rel, "id": ident, "masked": ln})

    for sub in ("review/out", "review/verify"):
        for dp, dn, fn in os.walk(os.path.join(data_root, sub)):
            dn[:] = [d for d in dn if d not in ("latents", "models", ".git", "__pycache__")]
            for f in fn:
                p = os.path.join(dp, f)
                one(p, os.path.relpath(p, data_root))
    for fol in folders:
        base = os.path.join(data_root, fol)
        if os.path.isdir(base):
            for f in sorted(os.listdir(base)):
                p = os.path.join(base, f)
                if os.path.isfile(p):
                    one(p, fol + "/" + f)
    return res


def _full_msg(host, fid, idx, step=1500):
    """Whole msg_json of one message (cells are capped near 2,000 characters, so it is read in windows)."""
    q = ("WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n+? FROM s WHERE n+? <= "
         "(SELECT length(msg_json) FROM frame_messages WHERE frame_id=? AND idx=?)) "
         "SELECT substr(msg_json, n, ?) FROM s, frame_messages WHERE frame_id=? AND idx=? ORDER BY n")
    r = host.query(q, [step, step, fid, idx, step, fid, idx])
    if not r.get("truncated"):
        return "".join(x[0] for x in r["rows"])
    n = host.query("SELECT length(msg_json) FROM frame_messages WHERE frame_id=? AND idx=?", [fid, idx])["rows"][0][0]
    return "".join(host.query("SELECT substr(msg_json, ?, ?) FROM frame_messages WHERE frame_id=? AND idx=?",
                              [s, step, fid, idx])["rows"][0][0] for s in range(1, n + 1, step))


def scan_transcripts(host, project_id, id_re, folders, since_ms, counts, skip=()):
    """counts = {frame_id: number of messages at the cut-off}; idx is contiguous from 0 (the run checks this), so messages with idx >= that number are not read."""
    res = {"frames_scanned": 0, "frames_mentioning": 0, "messages_scanned": 0, "messages_mentioning": 0, "flagged": []}
    frames = host.query("SELECT id, agent_name, delegate_name, created_at FROM frames WHERE project_id = ? "
                        "AND COALESCE(updated_at, created_at) >= ? ORDER BY created_at", [project_id, since_ms])["rows"]
    for fid, agent, dname, created in frames:
        if fid in skip:
            continue
        res["frames_scanned"] += 1
        hit = False
        for idx in range(counts[fid]):                   # idx runs 0..n-1 without gaps (checked at the run; a query would cap its rows)
            raw = _full_msg(host, fid, idx)
            res["messages_scanned"] += 1
            nraw = norm(raw)
            if id_re.search(nraw) or any(f in nraw for f in folders):
                res["messages_mentioning"] += 1
                hit = True
            else:
                continue
            try:
                m = json.loads(raw)
            except Exception:
                continue
            c = m.get("content")
            blocks = c if isinstance(c, list) else [{"type": "text", "text": str(c)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    cc = b.get("content")
                    txt = "\n".join(y.get("text", "") for y in cc if isinstance(y, dict)) if isinstance(cc, list) else str(cc)
                    kind = "tool_result"
                elif b.get("type") == "text":
                    txt, kind = b.get("text", ""), "text:" + str(m.get("role"))
                else:
                    continue
                for ident, ln in flagged_lines(txt, id_re, folders):
                    t = m.get("_ts")
                    res["flagged"].append({"frame": fid, "session": dname or agent, "msg": idx, "block": kind, "id": ident, "masked": ln,
                                           "ts_ny": datetime.datetime.fromtimestamp(t / 1000, NY).strftime("%Y-%m-%d %H:%M") if t else None})
        res["frames_mentioning"] += hit
    return res


def committed_blobs(repo):
    """Masked lines of the committed notebook pages, docs and the ledger, one blob per source."""
    blobs = {}
    for name, roots in (("notebook", ["notebook"]), ("docs", ["docs"]), ("constraints", ["CONSTRAINTS.md"])):
        lines = []
        for root in roots:
            base = os.path.join(repo, root)
            files = [base] if os.path.isfile(base) else [os.path.join(dp, f) for dp, _, fn in os.walk(base) for f in fn if f.endswith(".md")]
            for p in files:
                lines += [mask(l.strip()) for l in open(p, errors="ignore").read().split("\n")]
        blobs[name] = "\n".join(l for l in lines if len(l) >= 30)
    lines = []                                           # every other committed text file of the repository
    for dp, dn, fn in os.walk(repo):
        dn[:] = [x for x in dn if x not in (".git", "notebook", "docs", "manuscript_tm")]
        for f in fn:
            p = os.path.join(dp, f)
            if f.endswith((".md", ".json", ".py", ".txt", ".sh", ".yaml", ".yml", ".tsv")) and os.path.getsize(p) < 5_000_000:
                lines += [mask(l.strip()) for l in open(p, errors="ignore").read().split("\n")]
    blobs["repository file"] = "\n".join(l for l in lines if len(l) >= 30)
    return blobs


GENERIC_WORDS = ("failures", "nan", "div", "h1")
KNOWN = {("9c73f1c4", 327): "known display 1 (2026-10-02, X13 runner tests on atac_small)",
         ("baa34fcd", 554): "known display 2 (2026-10-13, scorer identity check)",
         ("4295f20e", 117): "known display 3 (2026-10-12 23:54, X13_lung_scanvi_l0_c1_s0_8e6f5009, head -c 600)"}


def classify(e, blobs, reviewed=frozenset(), first_listed_ny=None):
    """Class of one flagged entry (frame/msg known displays, coverage report, copy of committed text, line reviewed 2026-10-15)."""
    key = (e.get("frame", "")[:8], e.get("msg"))
    if key in KNOWN:
        return KNOWN[key]
    m = re.sub(r'^\S+?:#*:', '', e["masked"].replace('\\"', '"')).strip()
    if re.search(r'\.py\b', m) and re.search(r'#+%', m):
        return "coverage report"
    if len(m) >= 30:
        for name, blob in blobs.items():
            if m in blob:
                return "verbatim copy of committed " + name
    if e.get("id") in GENERIC_WORDS:
        return "match on an ordinary word that is also a file stem (" + e["id"] + "); no listed file named"
    if first_listed_ny and e.get("ts_ny") and e["ts_ny"] < first_listed_ny:
        return "message older than every listed file (before " + first_listed_ny + " New York time)"
    if (key[0], key[1], e["masked"]) in reviewed:
        return "line already reviewed on 2026-10-15 (power report, test count, project description or fixture value)"
    return "unclassified"


def _first_listed_ny(csv_path):
    with open(csv_path, newline="") as fh:
        return min(datetime.datetime.strptime(r["mtime_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
                   for r in csv.DictReader(fh)).astimezone(NY).strftime("%Y-%m-%d %H:%M")


def _finish(part, blobs, reviewed, first):
    for e in part["flagged"]:
        e["class"] = classify(e, blobs, reviewed, first)
    part["classes"] = dict(Counter(e["class"] for e in part["flagged"]))
    return part


def run_files(data_root, repo, csv_path, reviewed=frozenset()):
    """File part (plain Python; needs the data root). Returns the classified result."""
    idents, folders, listed = load_inventory(csv_path)
    fl = scan_files(data_root, make_id_re(idents), folders, listed)
    return _finish(fl, committed_blobs(repo), reviewed, _first_listed_ny(csv_path))


def run_transcripts(host, repo, csv_path, project_id, since_ms, own_frame, reviewed=frozenset()):
    """Transcript part (`repl` tool: needs host.query). The cut-off is taken first: every message stored
    at that moment is read, later ones are not."""
    cut = datetime.datetime.now(NY)
    rows = host.query("SELECT f.id, count(m.idx), max(m.idx) FROM frames f LEFT JOIN frame_messages m ON m.frame_id=f.id "
                      "WHERE f.project_id=? AND COALESCE(f.updated_at,f.created_at)>=? GROUP BY f.id", [project_id, since_ms])
    assert not rows.get("truncated") and all(r[1] == 0 or r[2] + 1 == r[1] for r in rows["rows"]), "message indices not contiguous"
    counts = {r[0]: r[1] for r in rows["rows"]}
    idents, folders, listed = load_inventory(csv_path)
    tr = scan_transcripts(host, project_id, make_id_re(idents), folders, since_ms, counts, skip=(own_frame,))
    tr.update(cutoff_new_york=cut.strftime("%Y-%m-%d %H:%M:%S %Z"), since="2026-10-02 00:00 New York",
              excluded_session=own_frame, sessions_in_scope=len(counts), messages_expected=sum(v for k, v in counts.items() if k != own_frame))
    return _finish(tr, committed_blobs(repo), reviewed, _first_listed_ny(csv_path))


MEMORY_NOTE = re.compile(r'\[(observed|stated|inferred)\]|^\s*(- )?\[[#\w ]+ ago\]')


def apply_manual_review(part):
    """Second step of the audit of 2026-10-16: every line the automatic classes left open ('unclassified') was read, digit-masked,
    by the agent that ran the audit (distinct masked lines; repeats of the same text in other messages share the class).
    Result: one new display (found by its identifier), the rest pilot (A1/A2/A3), code, power-report or memory-note text.
    These classes come from reading, not from a rule; the lines stay listed so that a reader can check them."""
    for e in part["flagged"]:
        if e["class"] != "unclassified":
            continue
        if e.get("id") and e["id"].startswith("X13_lung_scanvi_l0_c1_s0_8e6f5009"):
            e["manual_class"] = "NEW DISPLAY (not known on 2026-10-15): first row of a filler score file printed by `head -c 600`"
        elif MEMORY_NOTE.search(e["masked"]):
            e["manual_class"] = "stored project-memory note (earlier pilot results and decisions) repeated in a prompt or tool output"
        else:
            e["manual_class"] = "pilot (A1/A2/A3) score table or gate result, code, power-report or review text; no listed identifier on the line"
    part["manual_classes"] = dict(Counter(e["manual_class"] for e in part["flagged"] if "manual_class" in e))
    return part
