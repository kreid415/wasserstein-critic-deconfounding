"""First-crossing biology summaries, provenance checks, Table S10 and Figure S3."""
import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
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


def test_figs3_reads_rebuilt_types_and_renders(tmp_path, monkeypatch):  # test-reason: runs Figure S3, which had no test; ensures every first-crossing column still plots, both formats are written, figS3_values.csv holds the plotted differences and the drawn medians are the medians of its macro rows over the ranked tasks
    scripts = bio.DATA.parent / "figure_scripts"
    sys.path.insert(0, str(scripts))
    import figcommon
    data = tmp_path / "figure_data"                     # the script writes figS3_values.csv next to its inputs: keep the repository copy untouched
    data.mkdir()
    for name in ("label_transfer_types.csv", "label_transfer.csv", "low_signal_pairs.csv"):
        shutil.copyfile(bio.DATA / name, data / name)
    monkeypatch.setattr(figcommon, "OUT", tmp_path)
    monkeypatch.setattr(figcommon, "DATA", data)
    with mpl.rc_context():
        result = runpy.run_path(str(scripts / "figS3.py"), run_name="__main__")
    fig = result["fig"]
    assert len(fig.axes) == 9                           # panels a and b, six heat maps, one colour bar
    for ax in fig.axes[2:8]:                            # 2 plain-F1 columns + 13 method columns per task
        assert sum(np.asarray(m.get_array()).shape[1] for m in ax.collections) == 15
    for ext in ("png", "pdf"):
        assert (tmp_path / f"figS3.{ext}").stat().st_size > 0
    v = pd.read_csv(data / "figS3_values.csv")
    assert list(v.columns) == ["cond", "task", "arm", "celltype", "f1", "f1_plain", "diff"] and len(v) == 1092
    assert (v["diff"] - (v.f1 - v.f1_plain)).abs().max() < 1e-12
    types = v[v.celltype != "macro F1"]
    drawn = np.sort(np.concatenate([np.asarray(m.get_array()).ravel() for ax in fig.axes[2:8] for m in ax.collections if m.get_cmap().name == "RdBu"]))
    assert drawn == pytest.approx(np.sort(types["diff"].to_numpy()), abs=1e-12)          # the heat maps draw exactly the table's per-type differences
    macro = v[v.celltype == "macro F1"]
    med = macro[~((macro.task == "immune") & (macro.cond == 1))].groupby(["cond", "arm"])["diff"].median()
    for ax, c in zip(fig.axes[:2], (1, 0)):
        ticks = [ln for ln in ax.lines if ln.get_linewidth() == 1.6]
        arms = figcommon.ARMS + (["scvi_adv"] if c == 1 else [])
        assert [t.get_xdata()[0] for t in ticks] == pytest.approx([med[(c, a)] for a in arms], abs=1e-12)


def test_table_s10_tracked_cells_do_not_span_columns():  # test-reason: a word diff once wrapped several cells in one tracked-change argument, which breaks the tracked PDF build only
    tex = (bio.DATA.parent / "outline.tex").read_text()
    block = tex[tex.index("Method & Decoder & Purity diff."):]
    block = block[:block.index(r"\bottomrule")]
    args = re.findall(r"\\(?:replaced|added|deleted)\[id=P24\]((?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})+)", block)
    assert args, "no tracked cells found"
    assert not any("&" in a for a in args), args
    import pass24_edits as p24
    assert p24.track_row("A & $-0.1$ & 2 \\\\", "A & $-0.2$ & 2 \\\\") == "A & \\replaced[id=P24]{$-0.2$}{$-0.1$} & 2 \\\\"
    assert accept(p24.track_row("& x & 1 \\\\", "& y & 1 \\\\")) == "& y & 1 \\\\"
