"""Tests of paper/tools/make_supp_supp-analysis.py (run from the repository root: python -m pytest paper/tools/test_supp_analysis.py)."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("make_supp_sa", Path(__file__).with_name("make_supp_supp-analysis.py"))
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)


def test_first_crossing_and_number_formats():
    curve = [(0.2, 0.1, 0.4), (0.6, 0.5, 0.8), (0.4, 0.3, 0.6), (0.8, 0.7, 1.0)]       # (B, L, G); crosses 0.5 twice, the first counts
    assert sa.first_crossing(curve, 0.4) == pytest.approx((0.3, 0.6))
    assert sa.first_crossing(curve, 0.6) == (0.5, 0.8)                                 # exact hit of a point
    assert sa.first_crossing(curve, 0.9) is None
    assert [sa.lam(x) for x in (0.03, 0.3, 1.0, 30.0, 3000.0, 30000.0)] == ["0.03", "0.3", "1", "30", "3{,}000", "30{,}000"]
    assert sa.fs(-0.3354) == "$-0.335$"


def test_tables_generate_and_local_global_check_refuses_a_wrong_record(tmp_path, monkeypatch):  # test-reason: runs both generators end to end on the paper's inputs and plants a wrong record value
    monkeypatch.setattr(sa, "OUT", tmp_path)
    assert sa.make_s8()["rows"] == 96 and sa.make_note9()["rows"] == 120
    s8, n9 = (tmp_path / "tableS8.tex").read_text(), (tmp_path / "note9_table.tex").read_text()
    assert s8.count(r"decoder; $b^*=") == 16 and r"\addtocounter{table}{-1}" in s8   # 16 pairs; main table numbers untouched
    assert n9.count("Plain scVI &") == 16 and "(low-signal)" in n9                                      # 16 blocks (the cross-task summary row was removed in Pass 24)
    assert r"\renewcommand{\thetable}{S11}" in n9 and r"\label{tab:s11}" in n9 and r"\addtocounter{table}{-1}" in n9
    rk = sa.read("ranking_pairs.csv").head(1).copy()
    rk.loc[rk.index[0], "local_minus_global"] += 1e-6
    with pytest.raises(AssertionError):
        sa.local_global(rk)
