"""First-crossing biology summaries, provenance checks, Table S10 and Figure S3."""
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import biology_check_table as bio
import check_ledger as cl
from accept_changes import accept


def test_biology_summary_and_printed_s10(tmp_path, monkeypatch):  # test-reason: checks the new summary arithmetic, every Table S10 row and Note 12 reproduction against the archived inputs
    result = bio.summarize()
    assert result == json.loads((bio.DATA / "res5_biology.json").read_text())
    disc = [r for r in result["rows"] if r["method"] == "discriminator"]
    assert [r["purity_diff"] for r in disc] == pytest.approx([-0.013391369902451773, -0.0012097626810744183])
    assert result["rare_negative"] == 21 and result["rare_cells"] == 48
    assert result["ionocytes"]["purity_diff"] > 0
    assert result["ionocytes"]["recall"] > result["ionocytes"]["plain_recall"]
    tex = accept((bio.DATA.parent / "outline.tex").read_text())
    assert len(bio.table_lines(result)) == 20
    assert all(' '.join(row.split()) in ' '.join(tex.split()) for row in bio.table_lines(result))
    assert bio.signed(-0.0004) == bio.signed(0.0004) == "$0.000$"
    import pass24_edits as p24
    monkeypatch.setattr(p24, "ROOT", tmp_path)
    figures = tmp_path / "paper/supp/figures.tex"
    figures.parent.mkdir(parents=True)
    figures.write_text(subprocess.run(["git", "show", "24f2665:paper/supp/figures.tex"],
                                      check=True, text=True, capture_output=True).stdout)
    p24.first_crossing_supplement()
    assert (tmp_path / "paper/supp/note12.tex").read_text() == (bio.DATA.parent / "supp/note12.tex").read_text()


@pytest.mark.parametrize("bad", ["hash", "strength"])
def test_biology_refuses_wrong_strength_provenance(tmp_path, bad):  # test-reason: plants a wrong input hash or selected strength; successful-run tests cannot verify that either mismatch stops the summary
    for rel in ["first_crossing_strength.csv", "biology_check_first_crossing/biology_check.json",
                "biology_check_first_crossing/tables/summary.csv", "biology_check_first_crossing/tables/summary_by_method.csv"]:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bio.DATA / rel, dest)
    if bad == "hash":
        p = tmp_path / "first_crossing_strength.csv"
        p.write_text(p.read_text() + "\n")
    else:
        p = tmp_path / "biology_check_first_crossing/tables/summary.csv"
        df = pd.read_csv(p)
        df.loc[df.method == "discriminator", "lam"] = 999
        df.to_csv(p, index=False)
    with pytest.raises(AssertionError):
        bio.summarize(tmp_path)


def test_ledger_mean_uses_only_selected_rows():  # test-reason: known mean of unequal selected values distinguishes the new mean selector from min, max, sum, median and an unfiltered mean
    text = "method,value\na,1\na,2\na,9\nb,100\n"
    assert cl.evaluate(text, "where=method==a ; col=value ; agg=mean", {}) == 4


def test_figs3_reads_rebuilt_types_and_renders(tmp_path, monkeypatch):  # test-reason: runs Figure S3, which had no test; ensures every first-crossing column still plots and both formats are written
    scripts = bio.DATA.parent / "figure_scripts"
    sys.path.insert(0, str(scripts))
    import figcommon
    monkeypatch.setattr(figcommon, "OUT", tmp_path)
    with mpl.rc_context():
        result = runpy.run_path(str(scripts / "figS3.py"), run_name="__main__")
    fig = result["fig"]
    assert len(fig.axes) == 9
    assert all(ax.images[0].get_array().shape[1] == 15 for ax in fig.axes[:8])
    for ext in ("png", "pdf"):
        assert (tmp_path / f"figS3.{ext}").stat().st_size > 0
