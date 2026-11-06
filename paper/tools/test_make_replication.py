"""Tests of paper/tools/make_replication.py (run from the repository root: python -m pytest paper/tools/test_make_replication.py)."""
import subprocess, sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_replication as mr  # noqa: E402


@pytest.mark.parametrize("values, groups, expected", [
    ([1, 1], [0, 1], (1, 4)),        # two positive groups: only the observed pattern reaches the observed mean
    ([1, 1], [0, 0], (1, 2)),        # one group (two decoders of a task): its two members flip together
    ([1, -1], [0, 1], (3, 4)),       # observed mean 0: ties count as reaching it
])
def test_sign_flip_p_counts_patterns_per_group(values, groups, expected):  # test-reason: ties at the observed mean and whole-group flips, which the real result (no zero tau) never exercises
    assert mr.sign_flip_p(values, groups) == expected


def test_family_groups_flip_related_tasks_together_and_unknown_task_stops():  # test-reason: the family map and its fail-loud check, which sign_flip_p's own cases do not reach
    tasks, taus = ["atac_small", "atac_large", "pancreas"], [1, -1, 1]
    # hand count: family sums (atac 0, pancreas 1), observed total 1; of the 4 patterns, (+,+) and (-,+) give 1, the others -1 -> 2 of 4
    assert mr.sign_flip_p(taus, mr.family_groups(tasks)) == (2, 4)
    assert mr.sign_flip_p(taus, tasks) == (4, 8)  # per task: (+,+,+), (+,-,+), (+,-,-), (-,-,+) reach 1
    assert sorted(set(mr.family_groups(["immune", "immune_hum_mou", "sim1", "sim2"]))) == ["immune", "sim1", "sim2"]
    with pytest.raises(AssertionError, match="without a family"):
        mr.family_groups(["pancreas", "new_task"])


def test_strength_and_row_formats():  # test-reason: strengths of 1,000 and more (thousands separator) and a fractional rank, absent from the real table
    assert [mr.lam_str(x) for x in (0, 0.03, 3, 3000)] == ["0", "0.03", "3", "3{,}000"]
    r = pd.Series(dict(arm="mmd", status="bracketed", lam_lo=0.0, lam_hi=0.3, B_lo=0.5, B_hi=0.6, C_lo=0.7, C_hi=0.8,
                       rep_bio=0.75, sweep_bio=0.74, rep_rank=2.5, sweep_rank=1.0, lam_max=0.3, B_max=0.6))
    assert mr.row_s9(r, False) == r"MMD & 0--0.3 & 0.500/0.600 & 0.700/0.800 & 0.750 & 0.740 & 2.5 (1) &  \\"
    low = mr.row_s9(pd.Series(dict(r, status="unreached_low", lam_max=3.0)), True)
    assert "up to 3 & 0.600 (max) & -- & --" in low and "extension; not reached$^\\dagger$" in low


def test_outputs_reproduce_and_check_detects_a_change(tmp_path, monkeypatch, capsys):  # test-reason: runs build (git sources, manifest hashes, the rule's p recomputed) and both main branches
    if subprocess.run(["git", "-C", str(mr.REPO), "cat-file", "-e", mr.REV], capture_output=True).returncode:
        pytest.skip(f"{mr.REV} not available")  # pragma: no cover  # shallow copies lack the history
    committed = mr.OUT_TEX.read_text(), mr.OUT_JSON.read_text()
    monkeypatch.setattr(mr, "OUT_TEX", tmp_path / "tableS9.tex")
    monkeypatch.setattr(mr, "OUT_JSON", tmp_path / "sub" / "replication_numbers.json")
    assert mr.main([]) == 0
    assert (mr.OUT_TEX.read_text(), mr.OUT_JSON.read_text()) == committed
    assert "per-task sign-flip p = 0.00390625 (1 of 256 patterns), per-family p = 0.015625 (1 of 64 patterns)" in capsys.readouterr().out
    assert mr.main(["--check"]) == 0
    mr.OUT_TEX.write_text(committed[0].replace("0.456", "0.457", 1) + "x")
    assert mr.main(["--check"]) == 1
