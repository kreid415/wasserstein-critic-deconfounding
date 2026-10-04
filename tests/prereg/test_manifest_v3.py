"""Manifest v3 = manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv (amendment batch 2026-10-04): X15 KL warm-up
sensitivity (CONSTRAINTS.md SI-44) and Scanorama knn {2, 5, 10, 20, 40, 80} (SI-46). No fits, no data.

1. v3 is the builder's output; v2 is the builder's output at the previous commit (eb444fe).
2. v3 differs from v2 only by the signed-off rows: the 8 X13 Scanorama knn-160 rows (10 dimensions) are replaced by
   knn-2 rows and the 84 X15 rows are appended; every other row is byte-identical and in the same order, and the
   header line differs only in its row count. Every non-X3 row of v2 equals the tagged manifest. The line checker is
   shown to fail on changed, reordered and unexpected rows (fail-loud R11).
3. X15: immune and atac_large x conditioned x {lambda=0, discriminator / pooled / MMD at matched lo, hi} x seeds 10-12
   x kl_warmup {stock, complete} = 84 rows; X15 is a follow-up of prereg_rules (R4-x1 resolves its placeholders by base
   arm, like X3 / X6); the X15 'stock' rows equal to another follow-up row are reported (12, all X6 on immune).
"""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

import synth
from synth import P, BPM, ROOT
import freeze_matched_lambda
import manifest_v3_report as R

PREVIOUS = "eb444fe"    # integrate-fixes tip: builder of manifest v2
ARGS = ["--backbone", "stock", "--design", "pilot", "--uncond-seeds", "5", "--bary-iter", "10"]


def _rows(path):
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)


def test_v3_is_the_builder_output(tmp_path):
    out = tmp_path / "v3.tsv"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_paper_manifest.py"), *ARGS, "--out", str(out)],
                   check=True, capture_output=True)
    assert out.read_bytes() == open(R.V3, "rb").read()


def test_v2_is_the_builder_output_of_the_previous_commit(tmp_path):
    src = tmp_path / "bpm_previous.py"
    src.write_text(subprocess.run(["git", "-C", ROOT, "show", f"{PREVIOUS}:scripts/build_paper_manifest.py"],
                                  check=True, capture_output=True, text=True).stdout)
    out = tmp_path / "v2.tsv"
    subprocess.run([sys.executable, str(src), *ARGS, "--out", str(out)], check=True, capture_output=True)
    assert out.read_bytes() == open(R.V2, "rb").read()


def test_v3_differs_from_v2_only_by_the_signed_off_rows():
    old, new = R.lines(R.V2), R.lines(R.V3)
    bad, removed, added = R.v3_vs_v2_violations(old, new)
    assert bad == []
    rem, add = pd.DataFrame([R.fields(x) for x in removed]), pd.DataFrame([R.fields(x) for x in added])
    assert len(rem) == 8 and set(rem.arm) == {"scanorama"} and set(rem.lam) == {"160"} and set(rem.n_latent) == {"10"}
    assert sorted(rem.task) == sorted(BPM.TASKS)
    knn2 = add[add.experiment == "X13"]
    assert len(knn2) == 8 and set(knn2.arm) == {"scanorama"} and set(knn2.lam) == {"2"} and sorted(knn2.task) == sorted(BPM.TASKS)
    assert (add.experiment == "X15").sum() == 84 and len(add) == 92
    assert [R.fields(x)["experiment"] for x in new[-84:]] == ["X15"] * 84          # appended last
    assert len(new) - 2 == 6930 and len(set(old[2:]) & set(new[2:])) == 6846 - 8
    assert old[0].rsplit("(", 1)[1] == "0 of 6846)." and new[0].rsplit("(", 1)[1] == "0 of 6930)."


def test_v2_non_x3_rows_equal_the_tagged_manifest():
    assert R.non_x3_violations(R.lines(R.TAGGED), R.lines(R.V2)) == []


def test_line_checker_fails_on_changed_reordered_and_unexpected_rows():
    old, new = R.lines(R.V2), R.lines(R.V3)
    i = next(k for k, x in enumerate(new) if k >= 2 and R.fields(x)["experiment"] == "X1")
    changed = new.copy()
    changed[i] = changed[i].replace("\tauto\t", "\t0\t", 1)
    assert changed[i] != new[i] and R.v3_vs_v2_violations(old, changed)[0]
    swapped = new.copy()
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    assert "common rows are not in the same relative order" in R.v3_vs_v2_violations(old, swapped)[0]
    header = new.copy()
    header[0] = header[0].replace("design=pilot", "design=shared")
    assert "header line differs beyond the row count" in R.v3_vs_v2_violations(old, header)[0]
    knn = new.copy()
    j = next(k for k, x in enumerate(new) if k >= 2 and R.fields(x)["arm"] == "scanorama" and R.fields(x)["lam"] == "40")
    knn.pop(j)
    assert any(v.startswith("unexpected removed row") for v in R.v3_vs_v2_violations(old, knn)[0])


def test_x15_rows():
    M = _rows(R.V3)
    x = M[M.experiment == "X15"]
    assert len(x) == 84 and set(x.task) == {"immune", "atac_large"} and set(x.cond) == {"1"}
    assert set(x.seed) == {str(s) for s in BPM.FOLLOWUP_SEEDS} and not x.tag.duplicated().any()
    assert {json.dumps(json.loads(e), sort_keys=True) for e in x.extra} == \
        {json.dumps({"kl_warmup": k}) for k in ("stock", "complete")}
    got = x.groupby(["task", "arm", "lam", "n_critic"]).size().to_dict()
    want = {(t, "none", "0", "0"): 6 for t in ("immune", "atac_large")}
    for t in ("immune", "atac_large"):
        for a, k in (("discriminator", "1"), ("pooled", "5"), ("mmd", "0")):
            for lam in ("matched_lo", "matched_hi"):
                want[(t, a, lam, k)] = 6
    assert got == want
    assert x.groupby("task").max_epochs.unique().map(list).to_dict() == {"atac_large": ["94"], "immune": ["239"]}
    assert set(x.adv_input) == {"mean"} and set(x.zstd) == {"0"} and set(x.reference) == {"auto"}
    same = ["counts", "decoder", "n_latent", "n_layers", "n_hidden", "likelihood", "batch_size", "train_size"]
    x1 = M[(M.experiment == "X1") & M.task.isin(["immune", "atac_large"]) & (M.arm == "none") & (M.cond == "1")]
    assert set(map(tuple, x[same].to_numpy())) == set(map(tuple, x1[same].to_numpy())) and len(x1)


def test_x15_is_a_followup_and_r4_x1_resolves_it_by_base_arm():
    assert "X15" in P.FOLLOWUPS and tuple(R.FOLLOWUPS) == tuple(P.FOLLOWUPS)
    M = _rows(R.V3)
    x = M[M.experiment == "X15"].reset_index(drop=True)
    cells = {f"{t}|1|{a}": dict(lo=0.3, matched=1.0, hi=3.0) for t in ("immune", "atac_large")
             for a in ("discriminator", "pooled", "mmd")}
    out, n = freeze_matched_lambda.resolve(x, dict(cells=cells), "x1")
    assert n == 72 and set(out.lam) == {"0", "0.3", "3"}
    assert (out.lam[x.lam == "matched_lo"] == "0.3").all() and (out.lam[x.lam == "matched_hi"] == "3").all()
    P.check_seed_separation(M)


def test_x15_stock_rows_equal_to_other_followups_are_reported_not_removed():
    M = _rows(R.V3)
    eq = R.x15_stock_equal_followups(M)
    assert len(eq) == 12 and set(eq.equal_experiment) == {"X6"} and set(eq.task) == {"immune"}
    assert eq.groupby(["arm", "n_critic"]).size().to_dict() == {("discriminator", "1"): 6, ("pooled", "5"): 6}
    stock = set(M.tag[(M.experiment == "X15") & M.extra.str.contains('"stock"')])
    assert set(eq.x15_tag) <= stock and set(eq.equal_tag) <= set(M.tag[M.experiment == "X6"])
    assert (M.experiment == "X15").sum() == 84                          # reported, not deduplicated
    # the comparison is exact: changing one compared field of the matching X6 row removes the match
    j = M.index[M.tag == eq.equal_tag.iloc[0]][0]
    M2 = M.copy()
    M2.at[j, "max_epochs"] = "240"
    assert len(R.x15_stock_equal_followups(M2)) == 11
