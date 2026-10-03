"""X3 reference/target design (SI-31..SI-33) and X13 CPU baseline rows (SI-34..SI-36), 2026-10-03.
Run in the scvi env: WCD_SRC=src PREPPED_DIR=<prepped_scib> python -m pytest -q tests/scvi/test_x3_x13_design.py

1. X3_TARGETS re-derived from the prepped files (needs PREPPED_DIR): reference = select_reference_batch on the
   dose-0 subsample; depleted batch = the largest non-reference batch; types = its two most abundant types present
   in >= 3 batches, except sim2 (Group1 only; Batch3Sub1 holds only Group1 and Group2); the reference and the
   depleted batch are present at every dose; scripts/x3_design_profile.py reproduces the committed profiles.
2. Every X3 row of the manifest names its task's reference and carries its task's depletion spec.
3. resolve_reference: 'auto', an index and a name resolve as documented; on composition rows 'auto', an index,
   a missing name and the depleted batch are refused.
4. X13 CPU rows: exactly the signed-off knob values and dimensions per task, seed 0, knob named in extra; the
   GPU runner refuses them before reading data; cost_model gives them 0 GPU lane-hours.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_paper_manifest as bpm  # noqa: E402
import cost_model  # noqa: E402
import fit_paper_config as fpc  # noqa: E402
import x3_design_profile as x3p  # noqa: E402

NEEDS_DATA = pytest.mark.skipif(not os.environ.get("PREPPED_DIR"), reason="needs PREPPED_DIR (prepped scIB h5ad files)")


def _rows():
    return pd.DataFrame(bpm.finalize(bpm.build(bpm.BACKBONES["stock"], "pilot", 3, 5, 8)))


@NEEDS_DATA
@pytest.mark.parametrize("task", sorted(bpm.X3_TARGETS))
def test_x3_targets_follow_the_signed_off_rules(task):
    b, types, ref = bpm.X3_TARGETS[task]
    select = x3p._repo_data_module().select_reference_batch
    a0 = x3p.read_obs(os.environ["PREPPED_DIR"], task)
    a_d0, _, _ = x3p._cells(task, a0, b, types, 0)
    assert str(select(a_d0, "batch", "celltype")) == ref != b
    obs = a0.obs.astype(str)
    sizes = obs.batch.value_counts()
    assert b == sizes.drop(ref).idxmax()                                   # largest non-reference batch
    n_batches_with = obs.groupby("celltype").batch.nunique()
    in_b = obs[obs.batch == b].celltype.value_counts()
    eligible = in_b[in_b.index.isin(n_batches_with[n_batches_with >= 3].index)]
    if task == "sim2":
        assert set(in_b.index) == {"Group1", "Group2"} and types == ["Group1"]
    else:
        assert types == list(eligible.index[:2])                            # two most abundant eligible types
    for dose in x3p.DOSES:
        a, _, _ = x3p._cells(task, a0, b, types, dose)
        present = set(a.obs["batch"].astype(str))
        assert ref in present and b in present, (task, dose)


@NEEDS_DATA
def test_x3_profile_script_reproduces_the_committed_profiles(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "x3_design_profile.py"), "--prepped-dir",
                    os.environ["PREPPED_DIR"], "--out-dir", str(tmp_path)], check=True,
                   env=dict(os.environ, KMP_AFFINITY="disabled"))
    for f in ("x3_iw_weight_profile.csv", "x3_shared_support_profile.csv"):
        assert (tmp_path / f).read_text() == open(os.path.join(ROOT, "docs", f)).read(), f


def test_every_x3_row_names_its_reference_and_target():
    R = _rows()
    x3 = R[R.experiment == "X3"]
    assert len(x3) > 0
    for r in x3.itertuples():
        b, types, ref = bpm.X3_TARGETS[r.task]
        spec = json.loads(r.extra)["subsample"]
        assert (r.reference, spec["batch"], spec["types"]) == (ref, b, types)
    assert set(R[R.experiment != "X3"].reference) <= {"auto"} | {str(i) for i in range(16)}


def test_resolve_reference():
    batches = ["inDrop1", "inDrop3", "celseq"]
    comp = {"kind": "composition", "batch": "inDrop3", "types": ["alpha"], "deplete_pct": 50, "n_cells": 100}
    assert fpc.resolve_reference("t", "auto", batches, None, select=lambda: "celseq") == "celseq"
    assert fpc.resolve_reference("t", "1", batches) == "inDrop1"            # sorted: celseq, inDrop1, inDrop3
    assert fpc.resolve_reference("t", "inDrop1", batches, comp) == "inDrop1"
    for bad in ("auto", "0", "inDrop2", "inDrop3"):
        with pytest.raises(ValueError):
            fpc.resolve_reference("t", bad, batches, comp, select=lambda: "inDrop1")
    with pytest.raises(IndexError):
        fpc.resolve_reference("t", "3", batches)
    with pytest.raises(ValueError):
        fpc.resolve_reference("t", "missing", batches)


def test_x13_cpu_rows_are_the_signed_off_configurations():
    R = _rows()
    cpu = R[R.arm.isin(list(bpm.CPU_BASELINES))]
    assert set(cpu.experiment) == {"X13"} and set(cpu.seed) == {0} and not cpu.tag.duplicated().any()
    expect = {"harmony": ({(v, 10) for v in [0, 0.5, 1, 2, 4, 8]} | {(2, 50)}, "theta"),
              "scanorama": ({(v, 10) for v in [5, 10, 20, 40, 80, 160]} | {(20, 100)}, "knn"),
              "pca": ({(0, 10), (0, 50)}, None)}
    for t in bpm.TASKS:
        for arm, (cfgs, knob) in expect.items():
            g = cpu[(cpu.task == t) & (cpu.arm == arm)]
            assert {(float(r.lam), int(r.n_latent)) for r in g.itertuples()} == {(float(v), d) for v, d in cfgs}
            assert len(g) == len(cfgs)
            assert {json.loads(e).get("knob") for e in g.extra} == {knob}
    assert len(cpu) == 16 * len(bpm.TASKS)


def test_gpu_runner_refuses_cpu_rows(tmp_path):
    R = _rows()
    row = R[R.arm == "harmony"].iloc[0]
    man = tmp_path / "m.tsv"
    pd.DataFrame([row])[fpc.REQUIRED].to_csv(man, sep="\t", index=False)
    env = dict(os.environ, MANIFEST=str(man), TAG=row.tag, PREPPED_DIR=str(tmp_path / "absent"),
               OUT_DIR=str(tmp_path / "out"), WCD_SRC=os.path.join(ROOT, "src"), KMP_AFFINITY="disabled")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fit_paper_config.py")], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "run_cpu_baselines.py" in r.stderr
    assert not (tmp_path / "out").exists()


def test_cost_model_gives_cpu_rows_no_gpu_time(tmp_path):
    R = _rows()
    sel = pd.concat([R[R.arm.isin(list(bpm.CPU_BASELINES))].head(16), R[(R.experiment == "X13") & (R.arm == "scanvi") & (R.task == "immune")].head(1)])
    man = tmp_path / "m.tsv"
    sel[bpm.COLS].to_csv(man, sep="\t", index=False)
    T = pd.read_csv(os.path.join(ROOT, "docs", "throughput_rtx3080_stock_backbone.csv"))
    M = cost_model.cost(str(man), T, 3.77, score_s_per_cell=0.016)
    cpu = M.arm.isin(list(bpm.CPU_BASELINES))
    assert (M.lane_hours[cpu] == 0).all() and (M.lane_hours[~cpu] > 0).all()
    assert np.isfinite(M.score_cpu_hours).all()
