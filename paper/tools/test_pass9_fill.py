"""Tests of paper/tools/pass9_fill.py: number formats, fail-loud edits and placeholders, and a regeneration check
(re-running the script on the Pass 8 outline reproduces the outline and ledger of the Pass 9 commit; skipped where the data are absent)."""
import os, subprocess, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pass9_fill as pf

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                      cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()


@pytest.mark.parametrize("x,fmt,out", [(0.6481, "f3", "0.648"), (2.1666, "f2", "2.17"), (-0.0523, "sf3", "$-0.052$"),
                                       (0.115, "sf3", "$+0.115$"), (-0.0004, "sf3", "$0.000$"), (2.5, "g", "2.5"),
                                       (3.0, "g", "3"), (3.3333, "g", "3.33"), (0.1, "lam", "0.1"), (1000.0, "lam", "1,000"),
                                       (448, "d", "448"), ("c4a793f", "s", "c4a793f")])
def test_fmtv(x, fmt, out):  # test-reason: checks the exact text of each format; the other tests only reach the code paths
    assert pf.fmtv(x, fmt) == out


def test_fmtv_rejects():
    with pytest.raises(ValueError):
        pf.fmtv(2.5, "d")
    with pytest.raises(ValueError):
        pf.fmtv(1.0, "pct")


def test_vals_and_apply_fail_loud():
    V = pf.Vals()
    V.put("a", 0.5, "f.csv", "row 1")
    with pytest.raises(KeyError):
        V.put("a", 0.6, "f.csv", "row 2")
    with pytest.raises(ValueError):
        V.put("b", float("nan"), "f.csv", "row 3")
    with pytest.raises(KeyError):
        V.render("{{missing:f3}}", "s")
    with pytest.raises(AssertionError):   # a placeholder the pattern cannot parse is left unresolved
        V.render("{{a:f3 }}", "s")
    tex, starts = pf.apply("x OLD y\nz", [("e", "OLD", "N={{a:f3}}")], V)
    assert tex == "x N=0.500 y\nz" and starts == {"e": 1}
    assert V.rows[-1]["number_as_written"] == "0.500" and V.rows[-1]["source_key_or_row"] == "row 1"
    for text in ("no match", "OLD OLD"):
        with pytest.raises(ValueError):
            pf.apply(text, [("e", "OLD", "N")], V)
    rows = pf.ledger_rows("line one\nN=0.500\n", {"e": 1}, V)
    assert rows[-1]["tex_line"] == 2
    with pytest.raises(ValueError):   # a number that is not in the text after its snippet's first line
        pf.ledger_rows("N=0.400\n", {"e": 1}, V)


@pytest.mark.skipif(not os.path.isdir(pf.D), reason="result files not on this machine")
def test_regenerates_committed_outline(tmp_path):
    base = subprocess.run(["git", "show", "67019a8:paper/outline.tex"], capture_output=True, text=True, cwd=REPO, check=True).stdout
    (tmp_path / "outline.tex").write_text(base)
    args = ["--tex", str(tmp_path / "outline.tex"), "--ledger", str(tmp_path / "ledger.csv")]
    rep = os.path.join(REPO, "paper/tools/replication_result.json")
    if os.path.exists(rep):
        args += ["--replication", rep]
    pf.main(args)
    # compare with the Pass 9 commit that wrote the ledger; Pass 12 later corrected two source pointers in it (pass12_changelog.md)
    sha = "fbd41d7"
    show = lambda f: subprocess.run(["git", "show", f"{sha}:{f}"], capture_output=True, text=True, cwd=REPO, check=True).stdout
    assert (tmp_path / "outline.tex").read_text() == show("paper/outline.tex")
    assert (tmp_path / "ledger.csv").read_text() == show("paper/pass9_numbers.csv")
