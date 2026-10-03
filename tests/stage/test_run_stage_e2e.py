"""scripts/run_stage.py end to end with the REAL fitter (scripts/fit_paper_config.py) and scorer
(scripts/score_scib_native.py) on CPU: a 1,200-cell subsample of atac_small (400 cells per batch), 1 epoch, one fit
lane and one scorer (2 threads in total). Rows: lambda = 0, discriminator lambda = 1, discriminator lambda = 1e39
(inf in float32: the generator loss is inf at step 0, a forced divergence). Expected: scored, scored, diverged; gate
passes; failures.csv and the scores are read by the rule code.

Opt-in (skipped unless all are set): STAGE_E2E_FIT_PY (scvi env python), STAGE_E2E_SCORE_PY (scoring env python),
R_HOME, R_LIBS (scoring env R and the kBET library), PREPPED_DIR (prep_scib_task.py outputs; atac_small is read).
    CUDA_VISIBLE_DEVICES= python -m pytest -q tests/stage/test_run_stage_e2e.py
"""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fit_paper_config as fpc  # noqa: E402
import prereg_rules as pr  # noqa: E402

NEED = ("STAGE_E2E_FIT_PY", "STAGE_E2E_SCORE_PY", "R_HOME", "R_LIBS", "PREPPED_DIR")
pytestmark = pytest.mark.skipif(not all(os.environ.get(k) for k in NEED), reason=f"opt-in: needs {NEED}")

SUBSAMPLE = r'''
import sys, numpy as np, scanpy as sc
src, dst, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
a = sc.read_h5ad(src)
rng = np.random.default_rng(0)
idx = np.sort(np.concatenate([rng.choice(np.where(a.obs["batch"].values == b)[0], per, replace=False)
                              for b in sorted(a.obs["batch"].unique())]))
s = a[idx].copy()
sc.pp.pca(s, n_comps=50, mask_var="highly_variable")       # the unintegrated PCA of these cells (PCR reference)
s.write_h5ad(dst)
print(s.n_obs, s.n_vars, s.obs["batch"].nunique())
'''


def _row(tag, **over):
    r = dict(tag=tag, experiment="E2E", task="atac_small", counts="scib", arm="discriminator", lam="1",
             n_critic="1", adv_input="mean", zstd="0", cond="1", decoder="SCVI", n_latent="10", n_layers="1",
             n_hidden="128", likelihood="zinb", batch_size="128", max_epochs="1", train_size="0.9", seed="100",
             reference="auto", extra="{}")
    r.update(over)
    return r


def test_real_fitter_and_scorer(tmp_path):
    prepped = tmp_path / "prepped"
    prepped.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", KMP_AFFINITY="disabled")
    p = subprocess.run([os.environ["STAGE_E2E_SCORE_PY"], "-c", SUBSAMPLE,
                        os.path.join(os.environ["PREPPED_DIR"], "atac_small__scib.h5ad"),
                        str(prepped / "atac_small__scib.h5ad"), "400"], env=env, capture_output=True, text=True,
                       check=True)
    assert p.stdout.split() == ["1200", "3429", "3"], p.stdout
    rows = [_row("none", arm="none", lam="0", n_critic="0"), _row("disc"), _row("div", lam="1e39")]
    man = tmp_path / "manifest.tsv"
    pd.DataFrame(rows)[fpc.REQUIRED].to_csv(man, sep="\t", index=False)
    out = tmp_path / "out"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_stage.py"), "--manifest", str(man), "--experiments",
           "E2E", "--out-dir", str(out), "--prepped-dir", str(prepped), "--fit-python", os.environ["STAGE_E2E_FIT_PY"],
           "--score-python", os.environ["STAGE_E2E_SCORE_PY"], "--expect-device", "cpu", "--fit-lanes", "1",
           "--score-workers-during", "1", "--score-workers-after", "1", "--allow-dirty", "--view-interval-s", "5"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    (tmp_path / "runner.log").write_text(p.stdout + p.stderr)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    led = pd.read_csv(out / "ledger" / "E2E__all.csv", dtype=str, keep_default_na=False).set_index("tag")
    assert led.state.to_dict() == {"none": "scored", "disc": "scored", "div": "diverged"}, led.state.to_dict()
    F = pr.read_failures(str(out / "failures.csv"))
    assert F.tag.tolist() == ["div"] and F.status.tolist() == ["diverged"]
    st = json.loads((out / "status" / "div.json").read_text())
    assert (st["epoch"], st["step"], st["terms"]["gen_loss"]) == (0, 0, "inf"), st
    S, files = pr.read_scores(str(out / "scores"))
    assert sorted(S.tag) == ["disc", "none"] and set(S.kbet_seed) == {0} and (S.kbet_r_calls > 0).all()
    need = pr.BATCH_METRICS + pr.bio_metrics_for("atac_small")
    assert S[need].notna().all().all(), S[need]
    g = json.loads((out / "ledger" / "E2E__all.json").read_text())["gate"]
    assert g["ok"] and g["counts"] == {"diverged": 1, "scored": 2}
