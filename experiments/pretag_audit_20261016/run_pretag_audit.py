"""Runner of the pre-tag display audit for tag prereg-tier12-v3 (2026-10-16).
Uses audit_displayed_values.py in this folder (copy of the 2026-10-16 script; only change: KNOWN gets Display 3).
Steps (see README.md): files_part() in python; transcripts_part() in the repl tool (needs host.query); finish() in python.
Only counts and digit-masked lines are stored or printed."""
import datetime, json, os, re, sys
from collections import Counter
from zoneinfo import ZoneInfo
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_displayed_values as A

NY = ZoneInfo("America/New_York")
PROJECT = "proj_f1d989766996"
SINCE_MS = int(datetime.datetime(2026, 10, 2, 0, 0, tzinfo=NY).timestamp() * 1000)
CSV = os.path.join(HERE, "..", "x1_x13_score_hashes_20261015", "score_file_hashes.csv")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_ROOT = "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
OUT = os.path.join(REPO, "..", "partial")  # outside the repository so that partial output is not read as a committed copy
os.makedirs(OUT, exist_ok=True)


def files_part(data_root=DATA_ROOT):
    r = A.run_files(data_root, REPO, CSV)
    json.dump(r, open(os.path.join(OUT, "files.json"), "w"))
    return {k: v for k, v in r.items() if k != "flagged"}


def transcripts_part(host, own_frame):
    r = A.run_transcripts(host, REPO, CSV, PROJECT, SINCE_MS, own_frame)
    json.dump(r, open(os.path.join(OUT, "transcripts.json"), "w"))
    return {k: v for k, v in r.items() if k != "flagged"}


GENERIC = set(A.GENERIC_WORDS)


def is_candidate_id(i, listed_ids):
    return bool(i) and i in listed_ids and i not in GENERIC and re.match(r"X1\d*[_+]", i) is not None


def manual_classes(part, listed_ids):
    """Reading of every 'unclassified' masked line (done by the auditor; rules below reproduce the reading).
    candidate = a listed X1/X13 identifier on a line with >= 3 decimals that is not a known display."""
    for e in part["flagged"]:
        if e["class"] != "unclassified":
            continue
        if is_candidate_id(e.get("id"), listed_ids):
            e["manual_class"] = "CANDIDATE NEW DISPLAY"
        elif A.MEMORY_NOTE.search(e["masked"]):
            e["manual_class"] = "stored project-memory note (earlier pilot results and decisions) repeated in a prompt or tool output"
        else:
            e["manual_class"] = "pilot (A1/A2/A3) score table or gate result, code, power-report or review text; no listed X1/X13 identifier on the line"
    part["manual_classes"] = dict(Counter(e["manual_class"] for e in part["flagged"] if "manual_class" in e))
    return part


def finish(own_frame, head, readings=None):
    readings = readings or {}
    """Step 3 (python): classify, combine files + transcripts, write audit_pretag.json next to this script."""
    idents, folders, listed = A.load_inventory(CSV)
    L = set(idents)
    F = manual_classes(json.load(open(os.path.join(OUT, "files.json"))), L)
    T = manual_classes(json.load(open(os.path.join(OUT, "transcripts.json"))), L)
    cands = [{"frame": e["frame"], "session": e.get("session"), "msg": e["msg"], "ts_ny": e["ts_ny"], "id": e["id"], "block": e["block"],
              "masked": e["masked"], "note": "flagged by rule: listed identifier + >= 3 decimal numbers; not a known display",
              "auditor_reading": readings.get(e["frame"][:8] + ":" + str(e["msg"]))}
             for e in T["flagged"] if e.get("manual_class") == "CANDIDATE NEW DISPLAY"]
    cands += [{"file": e["file"], "id": e["id"], "masked": e["masked"]} for e in F["flagged"] if e.get("manual_class") == "CANDIDATE NEW DISPLAY"]
    allc = Counter()
    for part in (F, T):
        allc.update(part["classes"])
    manual = Counter()
    for part in (F, T):
        manual.update(part["manual_classes"])
    lines = lambda part, kind: [{"part": kind, "frame": e.get("frame"), "file": e.get("file"), "msg": e.get("msg"), "ts_ny": e.get("ts_ny"), "id": e.get("id"),
                                 "manual_class": e["manual_class"], "masked": e["masked"]} for e in part["flagged"] if "manual_class" in e]
    out = {"head": head, "cutoff_new_york": T["cutoff_new_york"], "since": T["since"], "own_session_excluded": own_frame,
           "sessions_in_scope": T["sessions_in_scope"], "sessions_searched": T["frames_scanned"], "sessions_mentioning": T["frames_mentioning"],
           "messages_searched": T["messages_scanned"], "messages_mentioning": T["messages_mentioning"], "messages_expected": T["messages_expected"],
           "identifiers": len(idents), "folders": len(folders), "listed_files": len(listed),
           "files_scanned": F["files_scanned"], "files_mentioning": F["files_mentioning"], "files_scanned_outside_review": F["files_scanned_outside_review"],
           "flagged_files_part": F["classes"], "flagged_transcripts_part": T["classes"], "flagged_by_class": dict(allc),
           "manual_classes_files": F["manual_classes"], "manual_classes_transcripts": T["manual_classes"], "manual_classes": dict(manual),
           "known_displays": {k[0] + ":" + str(k[1]): v for k, v in A.KNOWN.items()},
           "new_display_found": bool(cands), "candidates": cands, "manual_review_lines": lines(F, "files") + lines(T, "transcripts")}
    json.dump(out, open(os.path.join(HERE, "audit_pretag.json"), "w"), indent=1, ensure_ascii=False)
    return {k: v for k, v in out.items() if k not in ("manual_review_lines",)}
