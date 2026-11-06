"""Re-read the paper's number ledgers from their sources; exit 1 on any mismatch or any source pointer that does not resolve.

Usage, from the repository root:  python paper/tools/check_ledger.py [--data-root D] [ledger.csv ...]
(default ledgers: paper/pass7_numbers.csv, pass9 to pass16, the supplement ledgers supp-methods, supp-analysis and
supp-records, then pass17; a later ledger may retire an earlier row with `supersedes`).

Source pointers: a relative path resolves from the repository root; an absolute path must lie under the data root
(DATA_ROOT below, re-rooted at --data-root); git:<rev>:<path> reads a committed file with `git show`; a directory resolves but
carries no value. Ledger formats:
- pass12 (id, location, written, source, selector, fmt): the value is re-computed from the source with the selector, formatted
  with fmt and compared with `written`; `location` (<file>::<anchor>) must occur in that file and contain `written`.
- pass7/pass9 (source_file, source_key_or_row, source_value): the path must resolve; where the key is a JSON key (no spaces) the
  value is re-read and compared with source_value; free-text keys are checked for the path only (counted in the report).
- pass10/pass11 (item, value, source): source is <path> or <path>:<JSON key>; with a key, the value is re-read and compared with
  `value` at the precision written.
Selector clauses, separated by ' ; ': key=<JSON key> | where=<term>&<term> (terms: col==v, col!=v, col in a|b, col<v, |col|<v,
also > <= >=) with col=<column> and agg=value|min|max|count|nunique|sum|argmax:<col>|argmin:<col> | pairdiff=<col> with
group=<cols> and optional lt=<x> (count of within-group pairs, |difference| < x) | re=<regex> (group 1, else the whole match) |
expr=<python expression over the raw values of earlier rows, by id>.  fmt: a format spec (.3f, .0%, d, ',d'), len, range_n
("a-b" -> b-a+1), present (a design statement: the re= pattern must be found; `written` is then checked only at its location)
or empty (precision taken from `written`)."""
import argparse, csv, io, json, math, re, subprocess, sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
LEDGERS = [f"paper/pass{n}_numbers.csv" for n in (7, 9, 10, 11, 12, 13, 14, 15, 16)] + [
    f"paper/supp-{part}_numbers.csv" for part in ("methods", "analysis", "records")] + ["paper/pass17_numbers.csv"]
WORDS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen".split())}
TERM = re.compile(r"^(\|?)([\w.]+)\|?\s*(==|!=|<=|>=|<|>| in )\s*(.*)$")


class LedgerError(Exception):
    pass


def read_source(src, root, data_root):
    """Text of a source pointer; None for a directory."""
    if src.startswith("git:"):
        _, rev, path = src.split(":", 2)
        r = subprocess.run(["git", "-C", str(root), "show", f"{rev}:{path}"], capture_output=True, text=True)
        if r.returncode:
            raise LedgerError(f"unresolvable git pointer {src}: {r.stderr.strip()}")
        return r.stdout
    if src.startswith("/"):
        if not (src + "/").startswith(DATA_ROOT + "/"):
            raise LedgerError(f"absolute path outside the data root: {src}")
        p = Path(data_root + src[len(DATA_ROOT):])
    else:
        p = Path(root) / src
    if p.is_dir():
        return None
    if not p.is_file():
        raise LedgerError(f"unresolvable path {src}")
    return p.read_text()


def json_get(d, key):
    """Value at a key; flat keys that contain dots (paper_numbers.json) and nested keys and list indices are all tried."""
    def walk(v, parts):
        if not parts:
            return True, v
        for i in range(len(parts), 0, -1):
            k = ".".join(parts[:i])
            if isinstance(v, dict) and k in v:
                found, got = walk(v[k], parts[i:])
                if found:
                    return True, got
        if isinstance(v, list) and parts[0].isdigit() and int(parts[0]) < len(v):
            return walk(v[int(parts[0])], parts[1:])
        return False, None
    def unwrap(v):
        return v["value"] if isinstance(v, dict) and "value" in v and "description" in v else v
    found, v = walk(d, key.split("."))
    if not found:  # a list inside a {description, value} record, e.g. _meta.seeds.0
        head, _, idx = key.rpartition(".")
        if head and idx.isdigit():
            base = json_get(d, head)
            if isinstance(base, list) and int(idx) < len(base):
                return base[int(idx)]
        raise LedgerError(f"JSON key {key} not found")
    return unwrap(v)


def _filter(df, where):
    for term in where.split("&"):
        m = TERM.match(term.strip())
        if not m or m.group(2) not in df.columns:
            raise LedgerError(f"bad where term {term!r}")
        absval, col, op, val = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        if op in ("==", "!=", "in"):
            s = df[col].astype(str)
            mask = s.isin(val.split("|")) if op == "in" else (s == val if op == "==" else s != val)
        else:
            s = df[col].abs() if absval else df[col]
            mask = {"<": s < float(val), ">": s > float(val), "<=": s <= float(val), ">=": s >= float(val)}[op]
        df = df[mask]
    return df


def evaluate(text, sel, values):
    """Raw value chosen by a selector from a source text (text is None for expr= rows)."""
    c = dict(part.split("=", 1) for part in sel.split(" ; ") if part)
    if "expr" in c:
        names = {"__builtins__": {}, "math": math, "max": max, "min": min, "abs": abs, "sum": sum, "int": int, "float": float}
        return eval(c["expr"], {**names, **values})  # one namespace, so generator expressions see the row values
    if text is None:
        raise LedgerError("selector needs a file, the source is a directory or empty")
    if "re" in c:
        m = re.search(c["re"], text)
        if not m:
            raise LedgerError(f"pattern {c['re']!r} not found")
        return m.group(1) if m.groups() else m.group(0)
    if "key" in c:
        return json_get(json.loads(text), c["key"])
    df = pd.read_csv(io.StringIO(text), sep="\t" if "\t" in text.split("\n", 1)[0] else ",")
    if "where" in c:
        df = _filter(df, c["where"])
    if "pairdiff" in c:
        diffs = [abs(a - b) for _, g in df.groupby(c["group"].split(",")) for a, b in combinations(g[c["pairdiff"]], 2)]
        return sum(d < float(c["lt"]) for d in diffs) if "lt" in c else len(diffs)
    agg = c.get("agg", "value")
    if agg == "count":
        return len(df)
    if not len(df):
        raise LedgerError("no rows selected")
    if agg.startswith(("argmax:", "argmin:")):
        by = agg.split(":", 1)[1]
        df = df.loc[[df[by].idxmax() if agg.startswith("argmax") else df[by].idxmin()]]
        agg = "value"
    s = df[c["col"]]
    if agg == "value":
        if len(s) != 1:
            raise LedgerError(f"{len(s)} rows selected, expected 1")
        return s.iloc[0].item() if hasattr(s.iloc[0], "item") else s.iloc[0]
    return {"min": s.min, "max": s.max, "nunique": s.nunique, "sum": s.sum}[agg]()


def norm(s):
    s = str(s).strip().replace("$", "").replace("\\%", "%").replace("{,}", ",").replace("--", "-").replace("\u2212", "-")
    return WORDS.get(s.lower(), s)


def fmt_value(v, fmt, written):
    if fmt == "present":  # the selector's pattern states the fact; finding it is the check
        return written
    if fmt == "len":
        return str(len(v))
    if fmt == "range_n":
        a, b = re.split(r"-+", str(v))
        return str(int(b) - int(a) + 1)
    if not fmt:
        w = norm(written)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and re.fullmatch(r"[-+]?\d+(\.\d+)?", w):
            d = len(w.split(".")[1]) if "." in w else 0
            return f"{v:.{d}f}"
        return str(v)
    if fmt in ("d", ",d") and isinstance(v, float) and v.is_integer():
        v = int(v)
    return format(v, fmt)


def final_abstract(tex):
    """The abstract as printed with \\usepackage[final]{changes}: new text of \\replaced, \\added kept, \\deleted and placeholders dropped."""
    import abstract_words as aw
    return aw.active_text(aw.abstract(tex), [])


def superseded_ids(ledgers, root):
    """Ids of earlier rows whose location a later ledger (column `supersedes`) has moved; the value is still re-read, the old
    location (raw source text, which also holds struck old text) is no longer a claim about the printed text."""
    ids = set()
    for led in ledgers:
        rows = list(csv.DictReader(open(Path(root) / led, newline="")))
        ids |= {r["supersedes"] for r in rows if r.get("supersedes")}
    return ids


def check_pass12(rows, root, data_root, superseded=()):
    values, errors = {}, []
    for r in rows:
        try:
            if not r["selector"]:
                raise LedgerError("no selector")
            raw = evaluate(read_source(r["source"], root, data_root) if r["source"] else None, r["selector"], values)
            values[r["id"]] = raw
            got = fmt_value(raw, r["fmt"], r["written"])
            if norm(r["written"]) != norm(got):
                raise LedgerError(f"written {r['written']!r} but source gives {got!r}")
            if r["location"] and r["id"] not in superseded:  # `superseded`: ids a later ledger has moved
                f, anchor = r["location"].split("::", 1)
                text = read_source(f.removeprefix("abstract:"), root, data_root) or ""
                if f.startswith("abstract:"):  # anchor must be in the printed (final) abstract, not in struck old text
                    text = final_abstract(text)
                if anchor not in text:
                    raise LedgerError(f"anchor not found in {f}: {anchor[:60]!r}")
                if r["written"] not in anchor:
                    raise LedgerError(f"written value not inside its anchor: {anchor[:60]!r}")
        except (LedgerError, KeyError, ValueError, TypeError, NameError, SyntaxError) as e:
            errors.append(f"{r['id']}: {e}")
    return len(rows), 0, errors


def check_keyed(rows, root, data_root, path_col, key_col, val_col):
    checked, path_only, errors = 0, 0, []
    for i, r in enumerate(rows, 2):
        src, key = r[path_col], (r[key_col] if key_col else None)
        if key_col is None and not src.startswith("git:") and ":" in src:
            src, key = src.split(":", 1)
        try:
            text = read_source(src, root, data_root)
            if key and re.fullmatch(r"[\w.|\-]+", key) and src.endswith(".json"):
                got, want = json_get(json.loads(text), key), r[val_col]
                if key_col is None:
                    ok = norm(want) == norm(fmt_value(got, "", want))
                else:
                    ok = str(got) == want or (isinstance(got, (list, dict)) and json.dumps(got) == want) or (
                        isinstance(got, (int, float)) and not isinstance(got, bool)
                        and math.isclose(float(got), float(want), rel_tol=1e-12))
                if not ok:
                    raise LedgerError(f"recorded {want!r} but source gives {got!r}")
                checked += 1
            else:
                path_only += 1
        except (LedgerError, ValueError) as e:
            errors.append(f"line {i}: {e}")
    return checked, path_only, errors


def check_ledger(path, root=ROOT, data_root=DATA_ROOT, superseded=()):
    """(values re-read, rows checked for the path only, list of errors) for one ledger file."""
    rows = list(csv.DictReader(open(Path(root) / path, newline="")))
    cols = set(rows[0]) if rows else set()
    if "selector" in cols:
        return check_pass12(rows, Path(root), data_root, superseded)
    if "source_file" in cols:
        return check_keyed(rows, Path(root), data_root, "source_file", "source_key_or_row", "source_value")
    if "source" in cols:
        return check_keyed(rows, Path(root), data_root, "source", None, "value")
    raise LedgerError(f"{path}: unknown ledger format {sorted(cols)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-read the paper's number ledgers from their sources.")
    ap.add_argument("ledgers", nargs="*", default=LEDGERS)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--root", default=str(ROOT), help="repository root (default: this checkout)")
    a = ap.parse_args(argv)
    bad = 0
    for i, led in enumerate(a.ledgers):  # a row is retired only by a later ledger, so a value may move more than once
        n, p, errs = check_ledger(led, a.root, a.data_root, superseded_ids(a.ledgers[i + 1:], a.root))
        print(f"{led}: {n} values re-read, {p} pointers checked for the path only, {len(errs)} failures")
        for e in errs:
            print("  FAIL", e)
        bad += len(errs)
    print("ledger check:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry; main() is tested
    sys.exit(main())
