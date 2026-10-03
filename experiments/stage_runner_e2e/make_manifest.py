#!/usr/bin/env python
"""Tiny manifest for the GPU end-to-end test of scripts/run_stage.py (engineering test; no result is used).

Rows are copied from the design of record (scripts/build_paper_manifest.py --backbone stock --design pilot
--uncond-seeds 5 --bary-iter 10), A1, atac_small, seed 100, one per training-step branch of the plan:
    none           lambda 0, conditioned        (scvi-tools' own training step)
    discriminator  lambda 1, conditioned        (JS arm)
    reference      lambda 1, unconditioned      (W1 critic, n_critic 5)
    barycenter     lambda 1, conditioned        (POT barycenter target)
    mmd            lambda 1, conditioned        (critic-free)
    discriminator  lambda 1e39, conditioned     forced divergence: 1e39 is inf in float32, so the generator loss
                                                is inf at step 0 (tests/scvi/test_nonfinite_guard.py)
Changed from the design rows: experiment 'E2E', max_epochs 2 (design 400), tags 'E2E_<arm>_l<lam>_c<cond>_s<seed>'
(never a design tag, so these latents cannot be mistaken for A1 fits).

Usage: python experiments/stage_runner_e2e/make_manifest.py --out experiments/stage_runner_e2e/manifest.tsv
"""
import argparse
import os
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import build_paper_manifest as BPM  # noqa: E402

PICK = [("none", "0", "1"), ("discriminator", "1", "1"), ("reference", "1", "0"), ("barycenter", "1", "1"),
        ("mmd", "1", "1")]
EPOCHS = "2"


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        design = os.path.join(d, "design.tsv")
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "build_paper_manifest.py"), "--backbone",
                        "stock", "--design", "pilot", "--uncond-seeds", "5", "--bary-iter", "10", "--out", design],
                       check=True, capture_output=True, text=True)
        m = pd.read_csv(design, sep="\t", comment="#", dtype=str, keep_default_na=False)
    base = m[(m.experiment == "A1") & (m.task == "atac_small") & (m.seed == "100")]
    rows = []
    for arm, lam, cond in PICK:
        r = base[(base.arm == arm) & (base.lam == lam) & (base.cond == cond)]
        if len(r) != 1:
            raise ValueError(f"design rows for {arm} lam {lam} cond {cond}: {len(r)}, expected 1")
        rows.append(r.iloc[0].to_dict())
    rows.append(dict(rows[1], lam="1e39"))                       # the discriminator row with an overflowing lambda
    out = pd.DataFrame(rows)[BPM.COLS]
    out["experiment"], out["max_epochs"] = "E2E", EPOCHS
    out["tag"] = [f"E2E_{r.arm}_l{r.lam}_c{r.cond}_s{r.seed}" for r in out.itertuples()]
    if out.tag.duplicated().any():
        raise ValueError("duplicate tags")
    with open(a.out, "w") as f:
        f.write("# stage-runner GPU end-to-end test (experiments/stage_runner_e2e/make_manifest.py): design rows A1 "
                "atac_small seed 100, max_epochs 2, plus discriminator lambda 1e39 (forced divergence)\n")
        out.to_csv(f, sep="\t", index=False)
    print(f"{len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
