"""Table 1 (Pass 27): paper/tools/make_table1.py rebuilds the committed table from paper/figure_data/collapse_common_range.csv
(run from the repository root: python -m pytest paper/tools/test_table1.py)."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import make_table1 as mt

REPO = Path(__file__).resolve().parents[2]
DATA = mt.DATA  # the committed inputs and outputs; mt.DATA is pointed at a temporary copy below


def test_table1_files_match_collapse_common_range(tmp_path, monkeypatch):  # test-reason: runs make_table1.py, which no other test runs; catches a hand edit of table1.tex or table1_collapse.csv, a k, n or rate that differs from collapse_common_range.csv, a rate not in percent (the old Figure 4 axis once read "percentage points"), a wrong median, and a row, task or block out of order
    data = tmp_path / "paper/figure_data"
    data.mkdir(parents=True)
    for name in ("collapse_common_range.csv", "paper_numbers.json"):
        shutil.copy(DATA / name, data / name)
    monkeypatch.setattr(mt, "DATA", data)
    monkeypatch.setattr(mt, "ROOT", tmp_path)
    mt.main()
    assert (data / "table1_collapse.csv").read_bytes() == (DATA / "table1_collapse.csv").read_bytes()
    tex = (tmp_path / "paper/table1.tex").read_text()
    assert tex == (REPO / "paper/table1.tex").read_text()
    # an independent read of the input: every cell, every median, every label and the order of the rows
    cc = pd.read_csv(DATA / "collapse_common_range.csv").set_index(["cond", "arm", "task"])
    t = pd.read_csv(DATA / "table1_collapse.csv")
    assert len(t) == 84 and (t.task == "median").sum() == 12 and t.rate_pct.between(0, 100).all()
    rows = [ln for ln in tex.split("\n") if ln.endswith(r" \\") and not ln.startswith((r"\multicolumn", "Penalty"))]
    assert len(rows) == 12 and [ln.split(" &")[0] for ln in rows[:6]] == list(mt.ARM_LABEL.values()) == [ln.split(" &")[0] for ln in rows[6:]]
    assert tex.index("batch-conditioned decoder}") < tex.index(rows[0]) < tex.index("unconditioned decoder}") < tex.index(rows[6])
    for row, (cond, arm) in zip(rows, [(c, a) for c in (1, 0) for a in mt.ARM_LABEL]):
        r = cc.loc[[(cond, arm, task) for task in mt.REAL]]
        cells = [f"{round(100 * x.rate_common):d} ({x.k_common}/{x.n_common})" for x in r.itertuples()]
        assert row.removesuffix(r" \\").split(" & ")[1:] == cells + [f"{np.median(100 * r.rate_common):.1f}"], row
    assert tex.count(r"\label{tab:collapse}") == 1 and r"\small" in tex and not any(m in tex for m in (r"\added", r"\deleted", r"\replaced"))
