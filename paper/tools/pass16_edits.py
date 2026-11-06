"""Pass 16 edits (tracked id P16): the family-level sign-flip sensitivity analysis added to the replication result (user decision
2026-11-06). Edits paper/outline.tex (Methods replication paragraph, Res-2b, Disc-1, header) and revision/reviewer_response.md;
every old text must occur exactly once or the script stops. The number written (p = 0.0156, 1 of 64 patterns, six families) is
computed by paper/tools/make_replication.py (replication_numbers.json: p_family, p_family_hits, p_family_patterns, n_families,
n_families_tau_sum_positive) and listed in paper/pass16_numbers.csv. The abstract is not touched: it states neither sensitivity analysis.
Run from the repository root:  python paper/tools/pass16_edits.py   (PASS16_ROOT overrides the root of the edited files)
Then: python paper/tools/abstract_words.py ; python paper/tools/check_ledger.py ; python paper/tools/accept_changes.py"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PASS16_ROOT") or Path(__file__).resolve().parents[2])


def edit(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s)


def add(text):
    return r"\added[id=P16]{" + text + "}"


PATTERNS = (r"all $2^{6}$ sign patterns over six families: the two nested pairs, pancreas, lung, simulation 1 and simulation 2, "
            r"which share no data")
edit("paper/outline.tex", [
    # preamble and writer's-outline header
    (r"\definechangesauthor[name={Pass 15}, color=cyan]{P15}% re-review fixes and abstract cut; change log paper/pass15_changelog.md" "\n",
     r"\definechangesauthor[name={Pass 15}, color=cyan]{P15}% re-review fixes and abstract cut; change log paper/pass15_changelog.md" "\n"
     r"\definechangesauthor[name={Pass 16}, color=green]{P16}% family-level sign-flip sensitivity analysis; change log paper/pass16_changelog.md, numbers in paper/pass16_numbers.csv" "\n"),
    (r"\texttt{paper/outline\_clean.tex} is written by \texttt{paper/tools/accept\_changes.py}.}",
     r"\texttt{paper/outline\_clean.tex} is written by \texttt{paper/tools/accept\_changes.py}.}"
     + add(r" Pass 16 adds the family-level sign-flip sensitivity analysis to the replication result (change log "
           r"\texttt{paper/pass16\_changelog.md}).")),
    # Methods, replication paragraph
    (r"its $p$-value is approximate too.}{it is reported beside the rule and is not part of it.}}",
     r"its $p$-value is approximate too"
     + add(r"; a second sensitivity analysis, also added after the rule was fixed, flips the signs per task family, so that the two tasks "
           r"of each nested pair and both decoders flip together (" + PATTERNS + r"); it handles the nesting but not other dependence "
           r"between tasks, so it is approximate as well")
     + r".}{it is reported beside the rule and is not part of it.}}"),
    # Res-2b
    (r"so it is approximate as well (Methods).}}{[result: sensitivity analysis, not part of the rule:",
     r"so it is approximate as well (Methods).}"
     + add(r" With the signs flipped per task family (the two tasks of each nested pair and both decoders together; all $2^{6}$ patterns "
           r"over six families), $p=0.0156$. Only the observed pattern reaches the observed mean (1 of 64), because the summed $\tau_b$ is "
           r"positive in all 6 families, so this is the smallest $p$ that six families allow; it was defined after the rule and is "
           r"approximate as well (Methods).")
     + r"}{[result: sensitivity analysis, not part of the rule:"),
    # Disc-1
    (r"$p=0.0039$ with signs flipped per task; both values are approximate, because related tasks are not independent)}",
     r"$p=0.0039$ with signs flipped per task" + add(r" and $p=0.0156$ per task family")
     + r"; \replaced[id=P16]{all three values}{both values} are approximate, because related tasks are not independent)}"),
])
edit("revision/reviewer_response.md", [
    ("because it still flips the two tasks of a nested pair separately. The agreement rule is met",
     "because it still flips the two tasks of a nested pair separately; a further analysis added after the rule, which flips the signs\n"
     "   per task family (the two tasks of each nested pair and both decoders together, six families, $2^{6}$ patterns), gives\n"
     "   $p$ = 0.0156 and is approximate too. The agreement rule is met"),
    ("and still approximate, because the two tasks of a nested pair are flipped separately; the slope-adjusted reading",
     "and still approximate, because the two tasks of a nested pair are flipped separately; with signs flipped per task family (the two\n"
     "  tasks of each nested pair and both decoders together, $2^{6}$ patterns over six families) $p$ = 0.0156, the smallest value six\n"
     "  families allow, also approximate; the slope-adjusted reading"),
])
