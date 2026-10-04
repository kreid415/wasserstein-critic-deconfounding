"""Code check CR-05, rule side (lead decision 2026-10-04): prereg_rules refuses a rule input whose score rows mix
scorer provenance (scorer_git_sha, scorer_dirty, cpu_simd, numba_cpu_name, scorer_versions) and refuses rows without
these columns (A1 restarts at the new tag, so every rule input row carries them). Mutants of the check are killed in
tests/prereg/test_mutation.py (provenance_*).
"""
import ast
import os

import pandas as pd
import pytest

import checks
import synth
from synth import P, ROOT


def _scorer_provenance_cols():
    tree = ast.parse(open(os.path.join(ROOT, "scripts", "score_scib_native.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") == "PROVENANCE_COLS":
            return list(ast.literal_eval(node.value))
    raise AssertionError("score_scib_native.py defines no PROVENANCE_COLS")


def test_rule_columns_are_written_by_the_scorer():
    assert P.PROVENANCE_COLS == ("scorer_git_sha", "scorer_dirty", "cpu_simd", "numba_cpu_name", "scorer_versions")
    assert set(P.PROVENANCE_COLS) <= set(_scorer_provenance_cols())


def test_mixed_provenance_is_refused():
    checks.check_provenance_mix(P)


def test_missing_provenance_is_refused():
    checks.check_provenance_missing(P)


def test_csv_round_trip_keeps_an_empty_numba_cpu_name(tmp_path):
    M, S, F = checks._prov_input()
    sp, _ = synth.write_scores(S, F, str(tmp_path), "a1")
    assert ",," in open(sp).read().splitlines()[1]                    # the empty value is written as an empty field
    S2, _ = P.read_scores([sp])
    assert (S2.numba_cpu_name == "").all() and (S2.scorer_dirty == "0").all()
    assert len(P.outcomes(M, S2, F, ["A1"])) == len(M)


def test_a_score_file_without_provenance_is_refused_through_read_scores(tmp_path):
    M, S, F = checks._prov_input()
    d = tmp_path / "scores"
    d.mkdir()
    S.iloc[:4].drop(columns=list(P.PROVENANCE_COLS)).to_csv(d / "a_pre_cr05.csv", index=False)
    S.iloc[4:].to_csv(d / "b_current.csv", index=False)
    S2, files = P.read_scores([str(d)])
    assert len(files) == 2 and S2.scorer_git_sha.isna().sum() == 4
    with pytest.raises(P.PreregError, match="4 score rows of \\['A1'\\] lack scorer provenance values"):
        P.outcomes(M, S2, F, ["A1"])


def test_only_the_rows_of_the_rule_input_are_compared():
    """Score rows of other experiments (tags outside the rule input) may carry other provenance."""
    M, S, F = checks._prov_input()
    other = S.iloc[:3].copy()
    other["tag"] = [f"X1_other_{i}" for i in range(3)]
    other["scorer_git_sha"] = "f" * 40
    assert len(P.outcomes(M, pd.concat([S, other], ignore_index=True), F, ["A1"])) == len(M)


def test_failed_rows_need_no_score_row():
    M, S, F = checks._prov_input()
    gone = S.tag.iloc[:2].tolist()
    F2 = pd.DataFrame(dict(tag=gone, status="diverged", detail="synthetic"))
    out = P.outcomes(M, S[~S.tag.isin(gone)], F2, ["A1"])
    assert (out.status == "failed").sum() == 2
