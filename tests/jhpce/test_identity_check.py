"""Identity check of the JHPCE fixed-seed scoring (experiments/pilot5_jhpce_scoring/identity_check.py): equal rows pass; one differing
metric value, a column on one side only and a missing file are not passed over.
    python -m pytest -q tests/jhpce/test_identity_check.py
"""
import csv
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("identity_check", os.path.join(ROOT, "experiments", "pilot5_jhpce_scoring", "identity_check.py"))
IC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IC)
ROW = {"tag": "t1", "latent_sha256": "ab", "experiment": "A1", "iLISI": "0.25", "hvg_overlap": "", "scorer_versions": "{}", "scorer_git_sha": "c1",
       "cpu_model": "x", "score_host": "h", "score_seconds": "3"}


def write(path, **over):
    row = {**ROW, **over}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


def run(tmp_path, ref_over=None, cand_over=None):
    write(tmp_path / "ref.csv", **(ref_over or {}))
    (tmp_path / "cand").mkdir(exist_ok=True)
    write(tmp_path / "cand" / "t1.csv", score_site="jhpce", score_seconds="9", **(cand_over or {}))
    (tmp_path / "list.tsv").write_text(f"tag\tref_score\nt1\t{tmp_path / 'ref.csv'}\n")
    code = IC.main(["--ref-list", str(tmp_path / "list.tsv"), "--cand", str(tmp_path / "cand"), "--out", str(tmp_path / "out.json")])
    return code, json.load(open(tmp_path / "out.json"))


def test_identity_check_passes_equal_rows_and_flags_every_kind_of_difference(tmp_path):   # test-reason: identity_check.py lives under experiments/ (outside --src) and its verdict is the evidence the rule check reads
    code, res = run(tmp_path)
    assert code == 0 and res["all_columns_identical"] and res["n_latents"] == 1 and res["n_metric_columns"] == [2]    # provenance columns do not count
    code, res = run(tmp_path, cand_over={"iLISI": "0.2500001"})
    assert code == 1 and res["latents"][0]["differing_columns"] == ["iLISI"] and not res["all_columns_identical"]
    code, res = run(tmp_path, cand_over={"hvg_overlap": "0.5"})                       # NaN against a number
    assert code == 1 and res["latents"][0]["differing_columns"] == ["hvg_overlap"]
    code, res = run(tmp_path, cand_over={"scorer_versions": "{\"scib\": 2}"})
    assert code == 1 and not res["latents"][0]["same_package_versions"]
    write(tmp_path / "ref.csv", extra="1")                                           # a metric column only the reference has
    (tmp_path / "cand" / "t1.csv").unlink()
    write(tmp_path / "cand" / "t1.csv", score_site="jhpce")
    assert IC.main(["--ref-list", str(tmp_path / "list.tsv"), "--cand", str(tmp_path / "cand"), "--out", str(tmp_path / "o2.json")]) == 1
    assert json.load(open(tmp_path / "o2.json"))["latents"][0]["columns_on_one_side_only"] == ["extra"]
    (tmp_path / "cand" / "t1.csv").unlink()
    with pytest.raises(SystemExit, match="no such file"):
        IC.main(["--ref-list", str(tmp_path / "list.tsv"), "--cand", str(tmp_path / "cand"), "--out", str(tmp_path / "o3.json")])
