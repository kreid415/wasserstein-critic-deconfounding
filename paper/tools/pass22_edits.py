"""Pass 22 edits (2026-11-06, tracked id P22): a pass that cuts wordiness from the main text (Background, Methods, Results,
Discussion) and keeps its content: every number, citation, cross-reference, caveat and hedge stays. The new printed text of each
edited paragraph is in paper/tools/pass22_cut.json (old source line -> new text; null deletes a list item). Each edited line is
rewritten as word-level tracked changes against its printed (accepted) old text, so the tracked copy shows each cut; earlier
tracked marks on those lines are accepted first (changes cannot nest). Abstract, figure captions, tables, Declarations and the
supplement are not edited. Every old line must occur exactly once or the script stops.
Run from the repository root:  python paper/tools/pass22_edits.py   (PASS22_ROOT overrides the root of the edited files)
Then: python paper/tools/accept_changes.py ; python paper/tools/check_ledger.py"""
import difflib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accept_changes import accept, group  # noqa: E402

ROOT = Path(os.environ.get("PASS22_ROOT") or Path(__file__).resolve().parents[2])
TAG = "P22"
CITE = re.compile(r"\\cite[pt]?\{[^}]*\}")


def tokens(s):
    """Whitespace-separated tokens that never split a brace group, inline math or a control symbol such as \\ or \\%."""
    s = s.strip()                                     # inputs are never empty, so the last token is never empty
    out, cur, depth, math, i = [], "", 0, False, 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            cur += s[i:i + 2]
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "$":
            math = not math
        if ch.isspace() and depth == 0 and not math:
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
        i += 1
    out.append(cur)
    assert depth == 0 and not math, s[:80]
    return out


def struck(text):
    """Struck text goes through ulem's strikeout, which breaks on a bare citation; earlier passes box it the same way."""
    return CITE.sub(lambda m: r"\mbox{" + m.group(0) + "}", text)


def tracked(old, new):
    a, b = tokens(old), tokens(new)
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        o, n = struck(" ".join(a[i1:i2])), " ".join(b[j1:j2])
        o = " ".join(a[i1:i2]) if op == "equal" else o
        parts.append({"equal": o, "replace": rf"\replaced[id={TAG}]{{{n}}}{{{o}}}",
                      "delete": rf"\deleted[id={TAG}]{{{o}}}", "insert": rf"\added[id={TAG}]{{{n}}}"}[op])
    return " ".join(parts)


def rewrite(old_line, new):
    acc = accept(old_line).strip()
    if new is None:                                   # struck list item; accept_changes.py drops the empty \item
        return r"\item " + rf"\deleted[id={TAG}]{{{struck(acc[6:])}}}"
    if acc.startswith(r"\item "):
        return r"\item " + tracked(acc[6:], new[6:])
    if acc.startswith(r"\pp{"):
        label, j = group(acc, 3)
        body_old, _ = group(acc, j)
        label_new, j2 = group(new, 3)
        assert label_new == label, (label, label_new)
        return r"\pp{" + label + "}{" + tracked(body_old, group(new, j2)[0]) + "}"
    return tracked(acc, new)


def main():
    p = ROOT / "paper/outline.tex"
    s = p.read_text()
    for e in json.loads((Path(__file__).resolve().parent / "pass22_cut.json").read_text()):
        old = e["old_line"]
        assert s.count("\n" + old + "\n") == 1, old[:80]
        line = rewrite(old, e["new"])
        assert " ".join(accept(line).split()) == " ".join((e["new"] or r"\item").split()), old[:80]
        s = s.replace("\n" + old + "\n", "\n" + line + "\n", 1)
    anchor = "\\definechangesauthor[name={Pass 13}, color=orange]{P13}"
    assert s.count(anchor) == 1
    s = s.replace(anchor, "\\definechangesauthor[name={Pass 22}, color=red]{P22}% cut wordiness, content kept; change log "
                  "paper/pass22_changelog.md\n" + anchor)
    p.write_text(s)


if __name__ == "__main__":  # pass 23 imports rewrite() from here
    main()
