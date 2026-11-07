"""Tests of the supplement helpers (run from the repository root: python -m pytest paper/tools/test_supp.py)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figure_scripts"))
sys.path.insert(0, str(ROOT / "tools"))
import scaling_checks as sc  # noqa: E402
import make_supp_tables as mt  # noqa: E402


def test_ranks_ties_and_unreached():
    r = sc.ranks({"a": 0.9, "b": 0.8, "c": 0.8, "d": None, "e": None})
    assert r == {"a": 1.0, "b": 2.5, "c": 2.5, "d": 4.5, "e": 4.5}      # tied penalties and penalties that do not reach b* share places


def test_reading_interpolates_first_crossing_and_none_when_unreached():
    curve = [(0.0, 0.2, 0.9), (1.0, 0.6, 0.7), (3.0, 0.4, 0.5), (10.0, 0.8, 0.3)]     # crosses 0.5 twice: the first crossing counts
    assert sc.reading(curve, 0.5) == pytest.approx(0.75)
    assert sc.reading(curve, 0.9) is None


def test_kendall_variant_b():
    assert sc.kendall([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert sc.kendall([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_s6_row_formats_na_and_not_ranked():
    import pandas as pd
    base = dict(arm="mmd", bio_at_bstar=0.5, loso_se=0.01, n_loso=3, rank=2.5, rank_lo=2.0, rank_hi=3.0, crossings=1, status="bracketed")
    assert mt.row_s6(pd.Series(base)) == r"MMD & 0.500 & 0.010 & 3 & 2.5 & 2--3 & 1 \\"
    assert "n/a & 1 & 2.5 & n/a" in mt.row_s6(pd.Series(dict(base, n_loso=1)))        # fewer than two readings: SE and range not given
    low = pd.Series(dict(base, n_loso=float("nan"), loso_se=float("nan"), rank=float("nan"), rank_lo=float("nan"), rank_hi=float("nan"), status="unreached_low"))
    assert "not ranked" in mt.row_s6(low) and r"$^\dagger$" in mt.row_s6(low)


def test_supp_tables_cite_main_tables_by_label_and_placement_table_reproduces(tmp_path, monkeypatch, capsys):  # test-reason: runs the two generator functions no other test runs; catches a hard-coded main-text table number coming back
    monkeypatch.setattr(mt, "OUT", tmp_path)
    mt.make_s6()
    mt.make_s7()
    mt.check_table4()
    s6, s7 = (tmp_path / "tableS6.tex").read_text(), (tmp_path / "tableS7.tex").read_text()
    assert r"Supplementary Table~\ref{tab:3} gives the mean ranks" in s6
    assert s7.count(r"Table~\ref{tab:4}") == 2 and "Table~4" not in s7
    assert "reproduced" in capsys.readouterr().out
