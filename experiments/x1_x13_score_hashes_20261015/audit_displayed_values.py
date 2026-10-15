"""Audit: were X1 or X13 score values displayed? (counts and digit-masked lines only; no value is shown)

Two parts, both print counts and digit-masked lines only (SI-78: checks that print counts or flags do
not count as examining a score):
- scan_files(data_root, idents): log, text, markdown and JSON files of the final pre-publication review
  (review/out, review/verify), excluding the score files themselves;
- scan_transcripts(host, idents, folders, since_ms, skip): the app's stored session transcripts
  (table frame_messages). Runs only inside the Claude Science control-plane kernel, where `host.query`
  exists; the stored output of the run of 2026-10-15 is audit_displayed_values.json.
Identifiers = file stems of the X1/X13 score files of groups review_probes, x13_cpu_test and identity_check
in score_file_hashes.csv. A flagged line is one that holds an identifier and >= 3 decimal numbers, or a
metric name and >= 2 decimal numbers inside a block that mentions an identifier or a probe folder.
"""
import json, os, re

DEC = re.compile(r'(?<![\w.])-?\d+\.\d{2,}(?![\w.])')
METRIC = re.compile(r'(?i)(ilisi|clisi|kbet|asw|nmi|ari\b|pcr|graph_conn|isolated|silhouette|batch_score|bio_score|overall|scib|cell_cycle)')


def mask(line):
    return re.sub(r'\d', '#', line)


def flagged_lines(text, id_re, folders):
    out = []
    if not (id_re.search(text) or any(f in text for f in folders)):
        return out
    for ln in text.split('\n'):
        m = id_re.search(ln)
        if (m and len(DEC.findall(ln)) >= 3) or (METRIC.search(ln) and len(DEC.findall(ln)) >= 2):
            out.append((m.group(0) if m else None, mask(ln.strip())[:230]))
    return out


def scan_files(data_root, idents, folders):
    id_re = re.compile('|'.join(re.escape(t) for t in idents))
    res = {"files_scanned": 0, "files_mentioning": 0, "flagged": []}
    for sub in ("review/out", "review/verify"):
        for dp, dn, fn in os.walk(os.path.join(data_root, sub)):
            dn[:] = [d for d in dn if d not in ("latents", "models", ".git", "__pycache__")]
            for f in fn:
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, data_root)
                if rel.split('/')[-2] == 'scores' or not f.endswith(('.log', '.txt', '.md', '.jsonl', '.json', '.out', '.err')):
                    continue
                res["files_scanned"] += 1
                txt = open(p, errors='ignore').read()
                if id_re.search(txt):
                    res["files_mentioning"] += 1
                for ident, ln in flagged_lines(txt, id_re, folders):
                    res["flagged"].append({"file": rel, "id": ident, "masked": ln})
    return res


def _full_msg(host, fid, idx, step=1500):
    n = host.query("SELECT length(msg_json) FROM frame_messages WHERE frame_id=? AND idx=?", [fid, idx])["rows"][0][0]
    return "".join(host.query("SELECT substr(msg_json, ?, ?) FROM frame_messages WHERE frame_id=? AND idx=?",
                              [s, step, fid, idx])["rows"][0][0] for s in range(1, n + 1, step))


def scan_transcripts(host, project_id, idents, folders, since_ms, skip=()):
    id_re = re.compile('|'.join(re.escape(t) for t in idents))
    pats = ['%' + f + '%' for f in folders] + ['%' + t + '%' for t in idents]
    where = " OR ".join(["msg_json LIKE ?"] * len(pats))
    frames = host.query("SELECT id, agent_name, delegate_name, created_at FROM frames WHERE project_id = ? "
                        "AND COALESCE(updated_at, created_at) >= ? ORDER BY created_at", [project_id, since_ms])["rows"]
    res = {"frames_scanned": 0, "frames_mentioning": 0, "flagged": []}
    for fid, agent, dname, created in frames:
        if fid in skip:
            continue
        res["frames_scanned"] += 1
        idxs = [r[0] for r in host.query(f"SELECT idx FROM frame_messages WHERE frame_id = ? AND ({where}) ORDER BY idx", [fid] + pats)["rows"]]
        if idxs:
            res["frames_mentioning"] += 1
        for idx in idxs:
            try:
                m = json.loads(_full_msg(host, fid, idx))
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
                for ident, ln in flagged_lines(txt.replace("\\n", "\n"), id_re, folders):
                    res["flagged"].append({"frame": fid, "session": dname or agent, "msg": idx, "block": kind, "id": ident, "masked": ln})
    return res
