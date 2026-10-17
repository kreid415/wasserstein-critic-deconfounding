"""Helpers for the session-handoff skill (standard library only)."""
import os
import re
import subprocess

REQUIRED_SECTIONS = ("Objective", "State", "Decisions", "Next steps", "Open questions")
SECRET_PATTERNS = (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}", r"sk-[A-Za-z0-9]{20,}",
                   r"AKIA[0-9A-Z]{16}", r"x-access-token:[^@\s]+@")


def handoff_start_message(objective, next_action, path="docs/claude/STATUS.md", max_words=150):
    """Return the text to paste to open the next session (names the file and the single next action).

    Example
    -------
    >>> m = handoff_start_message("finish the analysis", "run scripts/x.py")
    >>> m.startswith("Continue from docs/claude/STATUS.md")
    True
    """
    msg = (f"Continue from {path}; read it before doing anything else, then run `git log --oneline -15`.\n"
           f"Objective: {objective}\nNext action: {next_action}")
    n = len(msg.split())
    if n > max_words:
        raise ValueError(f"start message is {n} words; the cap is {max_words}")
    return msg


def handoff_check(path="docs/claude/STATUS.md", root="."):
    """Check a handoff file against the repository. Returns a dict with lists: missing_sections,
    missing_paths, bad_commits, secrets, and ok (True when all are empty).

    Example
    -------
    >>> import tempfile
    >>> d = tempfile.mkdtemp(); p = os.path.join(d, "S.md")
    >>> open(p, "w").write("## Objective\\n## State\\n## Decisions\\n## Next steps\\n## Open questions\\n") and None
    >>> handoff_check(p, root=d)["ok"]
    True
    """
    text = open(path, encoding="utf-8").read()
    missing_sections = [s for s in REQUIRED_SECTIONS if not re.search(r"^#+\s*" + re.escape(s), text, re.M)]
    paths = set()
    for m in re.finditer(r"`([\w./-]+\.[A-Za-z0-9]{1,6})`", text):
        c = m.group(1)
        if "/" in c and not c.startswith(("http", "~")):
            paths.add(c)
    missing_paths = sorted(p for p in paths if not os.path.exists(os.path.join(root, p)))
    bad_commits = []
    for m in re.finditer(r"(?:commit|@)\s*`?([0-9a-f]{7,40})\b", text):
        h = m.group(1)
        r = subprocess.run(["git", "cat-file", "-e", h + "^{commit}"], cwd=root, capture_output=True)
        if r.returncode != 0:
            bad_commits.append(h)
    secrets = [p for p in SECRET_PATTERNS if re.search(p, text)]
    res = {"missing_sections": missing_sections, "missing_paths": missing_paths,
           "bad_commits": sorted(set(bad_commits)), "secrets": secrets}
    res["ok"] = not any(res.values())
    return res
