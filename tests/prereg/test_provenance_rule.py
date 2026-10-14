"""Code check CR-05, rule side (lead decision 2026-10-04): prereg_rules refuses a rule input whose score rows mix
scorer provenance (scorer_git_sha, scorer_dirty, cpu_simd, numba_cpu_name, scorer_versions) and refuses rows without
these columns (A1 restarts at the new tag, so every rule input row carries them). Mutants of the check are killed in
tests/prereg/test_mutation.py (provenance_*).
"""
import ast
import json
import os
import subprocess

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


@pytest.mark.parametrize("sinkhorn, tag, refused", [
    (None, "HEAD", None),                                    # no Sinkhorn row, all at the old commit: one commit
    ("tag", "HEAD", None),                                   # old commit for the other arms, Sinkhorn at the tag
    ("old", "HEAD", "\\['sinkhorn'\\] rows at 0123456"),       # Sinkhorn at the old commit: refused
    ("f" * 40, "HEAD", "mix scorer provenance"),             # an undeclared commit stays a mix
    ("tag", "no-such-tag", "mix scorer provenance"),         # tag absent from the checkout (RR-01): the mix stays
])
def test_si54_equivalent_commits_through_outcomes(tmp_path, monkeypatch, sinkhorn, tag, refused):
    """SI-54: docs/EQUIVALENT_COMMITS.json reads its old commit as the tag's commit for every arm but sinkhorn.
    Here the old commit is synth's scorer_git_sha and the tag is the checkout's HEAD."""
    M, S, F = checks._prov_input()
    old = synth.PROVENANCE["scorer_git_sha"]
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    eq = tmp_path / "eq.json"
    eq.write_text(json.dumps(dict(tag=tag, commit=old, commit_ref="prereg-tier12-v2", not_for_arms=["sinkhorn"])))
    monkeypatch.setattr(P, "EQUIVALENT_COMMITS", str(eq))
    if sinkhorn is None:
        M = M[M.arm != "sinkhorn"]
    else:
        S = S.copy()
        S.loc[S.tag.isin(M.tag[M.arm == "sinkhorn"]), "scorer_git_sha"] = dict(tag=head, old=old).get(sinkhorn, sinkhorn)
    if refused is None:
        assert len(P.outcomes(M, S, F, ["A1"])) == len(M)
    else:
        with pytest.raises(P.PreregError, match=refused):
            P.outcomes(M, S, F, ["A1"])


# test-reason: pins the committed list's content (user decisions SI-54, SI-60); the other tests use a temporary list
def test_the_committed_equivalence_list():
    """SI-54 / SI-60: the committed list names c88cce3 (tag prereg-tier12-v2), the new tag, and excludes sinkhorn."""
    E = json.load(open(P.EQUIVALENT_COMMITS))
    assert (E["commit"], E["commit_ref"], E["tag"], E["not_for_arms"]) == (
        "c88cce3c569e9de47f2d65c6794ca3890637786a", "prereg-tier12-v2", "prereg-tier12-v3", ["sinkhorn"])
