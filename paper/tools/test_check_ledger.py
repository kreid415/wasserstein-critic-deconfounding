"""Tests of paper/tools/check_ledger.py on small synthetic ledgers (run: python -m pytest paper/tools/test_check_ledger.py)."""
import csv, json, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_ledger as cl  # noqa: E402

P12 = ["id", "location", "written", "source", "selector", "fmt"]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository with JSON, CSV and text sources, plus a data root outside it."""
    r = tmp_path / "repo"
    (r / "paper").mkdir(parents=True)
    (r / "paper/n.json").write_text(json.dumps({"a.b": {"description": "x", "value": 0.2667}, "nest": {"l": [100, 101, 102]},
                                                "c": 3000.0, "s": "0-4", "m.s": {"description": "x", "value": [7, 8, 9]},
                                                "NL": {"p0.3_cov": 0.97}}))
    (r / "paper/t.csv").write_text("task,cond,arm,v,lo\npan,1,mmd,0.10,0.05\npan,1,disc,0.13,0.02\nlung,1,mmd,0.30,0.12\n"
                                   "lung,1,disc,0.25,0.11\nlung,0,mmd,-0.40,0.0\n")
    (r / "paper/f.tsv").write_text("dir\tname\nmodels\ta\nmodels\tb\nscores\tc\n")
    (r / "paper/doc.md").write_text("level 1 has six losses; near-ties (difference < 0.02)\n")
    (r / "paper/outline.tex").write_text("rate 27\\% here; over six losses; mean 0.30 and 0.25.\n")
    (r / "paper/sub").mkdir()
    data = tmp_path / "data"
    (data / "tables").mkdir(parents=True)
    (data / "tables/x.json").write_text(json.dumps({"k": 5}))
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
    monkeypatch.setattr(cl, "DATA_ROOT", "/fixed/data/root")
    return r, str(data)


def write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def row(id, written, source, selector, fmt="", location=""):
    return dict(id=id, location=location, written=written, source=source, selector=selector, fmt=fmt)


GOOD = [
    row("flat", "27\\%", "paper/n.json", "key=a.b", ".0%", "paper/outline.tex::rate 27\\% here"),
    row("nested", "101", "paper/n.json", "key=nest.l.1"),
    row("len", "three", "paper/n.json", "key=nest.l", "len"),
    row("reclist", "9", "paper/n.json", "key=m.s.2"),
    row("dotted", "0.97", "paper/n.json", "key=NL.p0.3_cov"),
    row("comma", "3,000", "paper/n.json", "key=c", ",d"),
    row("rng", "five", "paper/n.json", "key=s", "range_n"),
    row("rngtxt", "0--4", "paper/n.json", "key=s"),
    row("data", "5", "/fixed/data/root/tables/x.json", "key=k"),
    row("git", "six", "git:HEAD:paper/doc.md", "re=level 1 has (six) losses", "", "paper/outline.tex::over six losses"),
    row("whole", "0.02", "paper/doc.md", "re=0\\.02"),
    row("present", "six", "paper/doc.md", "re=level 1 has", "present"),
    row("argmax", "0.30", "paper/t.csv", "where=cond==1&arm in mmd|disc ; agg=argmax:v ; col=v"),
    row("argmin", "0.02", "paper/t.csv", "where=cond==1 ; agg=argmin:lo ; col=lo"),
    row("cnt", "2", "paper/t.csv", "where=|v|>0.2&cond!=0 ; agg=count"),
    row("lt", "0.10", "paper/t.csv", "where=v<0.11&v>=0.1&v<=0.1 ; col=v"),
    row("minmax", "0.13", "paper/t.csv", "where=task==pan ; col=v ; agg=max"),
    row("uniq", "2", "paper/t.csv", "where=cond==1 ; col=task ; agg=nunique"),
    row("sum", "0.78", "paper/t.csv", "where=cond==1 ; col=v ; agg=sum"),
    row("mn", "-0.40", "paper/t.csv", "col=v ; agg=min"),
    row("str", "lung", "paper/t.csv", "where=arm==disc&cond==1 ; col=task ; agg=min"),
    row("pairs", "2", "paper/t.csv", "where=cond==1 ; pairdiff=v ; group=task,cond"),
    row("near", "1", "paper/t.csv", "where=cond==1 ; pairdiff=v ; group=task,cond ; lt=0.04"),
    row("tsv", "2", "paper/f.tsv", "where=dir==models ; agg=count"),
    row("expr", "3,005", "", "expr=int(max(c for c in (comma, data)) + data)", ",d"),
]


def test_pass12_all_selector_kinds_pass(repo):  # test-reason: known answers for every selector kind on controlled data; the real-ledger run cannot tell a wrong value from a right one
    r, data = repo
    write(r / "paper/l12.csv", P12, GOOD)
    assert cl.check_ledger("paper/l12.csv", r, data) == (len(GOOD), 0, [])


@pytest.mark.parametrize("bad, message", [
    (row("m", "28\\%", "paper/n.json", "key=a.b", ".0%"), "but source gives"),
    (row("m", "1", "paper/n.json", ""), "no selector"),
    (row("m", "1", "paper/sub", "key=a"), "directory or empty"),
    (row("m", "1", "paper/missing.json", "key=a"), "unresolvable path"),
    (row("m", "1", "/elsewhere/x.json", "key=a"), "outside the data root"),
    (row("m", "1", "git:nosuchrev:paper/doc.md", "re=x"), "unresolvable git pointer"),
    (row("m", "1", "paper/n.json", "key=nest.l.7"), "not found"),
    (row("m", "1", "paper/n.json", "key=m.s.5"), "not found"),
    (row("m", "1", "paper/n.json", "key=nope"), "not found"),
    (row("m", "1", "paper/doc.md", "re=absent"), "not found"),
    (row("m", "1", "paper/t.csv", "where=nocol==1 ; agg=count"), "bad where term"),
    (row("m", "1", "paper/t.csv", "where=task==none ; col=v"), "no rows selected"),
    (row("m", "1", "paper/t.csv", "where=task==pan ; col=v"), "expected 1"),
    (row("m", "1", "", "expr=undefined_id + 1"), "undefined_id"),
    (row("m", "27\\%", "paper/n.json", "key=a.b", ".0%", "paper/outline.tex::not in the file"), "anchor not found"),
    (row("m", "27\\%", "paper/n.json", "key=a.b", ".0%", "paper/outline.tex::over six losses"), "not inside its anchor"),
])
def test_pass12_failures_are_reported(repo, bad, message):
    r, data = repo
    write(r / "paper/l12.csv", P12, [bad])
    n, p, errs = cl.check_ledger("paper/l12.csv", r, data)
    assert n == 1 and len(errs) == 1 and message in errs[0], errs


def test_pass7_and_pass11_formats(repo):
    r, data = repo
    write(r / "paper/l7.csv", ["tex_line", "number_as_written", "source_file", "source_key_or_row", "source_value"], [
        dict(tex_line=1, number_as_written="27", source_file="paper/n.json", source_key_or_row="a.b", source_value="0.2667"),
        dict(tex_line=2, number_as_written="1", source_file="paper/n.json", source_key_or_row="c", source_value="3000.0000000001"),
        dict(tex_line=3, number_as_written="x", source_file="paper/t.csv", source_key_or_row="max over rows", source_value="1"),
        dict(tex_line=4, number_as_written="x", source_file="paper/n.json", source_key_or_row="c", source_value="12"),
        dict(tex_line=5, number_as_written="x", source_file="paper/gone.csv", source_key_or_row="a", source_value="1"),
        dict(tex_line=6, number_as_written="x", source_file="paper/n.json", source_key_or_row="nest.l", source_value="[100, 101, 102]")])
    n, p, errs = cl.check_ledger("paper/l7.csv", r, data)
    assert (n, p) == (3, 1) and len(errs) == 2 and "source gives 3000.0" in errs[0] and "unresolvable" in errs[1], errs
    write(r / "paper/l11.csv", ["item", "value", "source"], [
        dict(item="a", value="0.267", source="paper/n.json:a.b"),
        dict(item="b", value="0.30", source="paper/n.json:a.b"),
        dict(item="c", value="10 of 17", source="paper/doc.md"),
        dict(item="d", value="six", source="git:HEAD:paper/doc.md")])
    n, p, errs = cl.check_ledger("paper/l11.csv", r, data)
    assert (n, p) == (1, 2) and len(errs) == 1 and "line 3" in errs[0], errs
    write(r / "paper/odd.csv", ["x"], [dict(x=1)])
    with pytest.raises(cl.LedgerError, match="unknown ledger format"):
        cl.check_ledger("paper/odd.csv", r, data)


def test_main_exit_codes(repo, capsys):
    r, data = repo
    write(r / "paper/ok.csv", P12, GOOD[:2])
    write(r / "paper/bad.csv", P12, [row("m", "2", "paper/n.json", "key=nest.l.1")])
    assert cl.main(["paper/ok.csv", "--root", str(r), "--data-root", data]) == 0
    assert cl.main(["paper/ok.csv", "paper/bad.csv", "--root", str(r), "--data-root", data]) == 1
    out = capsys.readouterr().out
    assert "ledger check: OK" in out and "FAIL m:" in out and "ledger check: FAIL" in out
