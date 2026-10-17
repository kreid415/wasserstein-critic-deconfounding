#!/usr/bin/env python
"""Run scripts/fit_paper_config.py of ONE checkout on many rows with ZERO optimiser steps (plan_identity.sh calls this).

Each row goes through the checkout's own script exactly as in a real fit (data, subsample, reference choice, model,
training plan, optimisers, saved config, latent, model file) except that the Lightning trainer is built with
limit_train_batches=0, limit_val_batches=0, num_sanity_val_steps=0: the training plan and its optimisers are created
and configured, but no minibatch is run, so the saved latent is the seeded initial network's and the saved config holds
the resolved plan. The number of optimiser steps each trainer took is written to OUT/steps.txt (must be 0).
One process per checkout (imports once); every row runs in its own try block and its exit code goes to
EXIT_CODES as "<label> <tag> <code>". Nothing here reads a score.
Usage: nofit_worker.py CHECKOUT LABEL OUT_DIR ROWS_TSV EXIT_CODES   (env: PREPPED_DIR)
       nofit_worker.py --make-synthetic PATH   (stand-in sim2 input for a run without the real prepped data)
"""
import os
import runpy
import sys
import traceback

import numpy as np
import pandas as pd


def make_synthetic(path):
    """Stand-in for prepped_scib/sim2__scib.h5ad: 6 batches x 4,000 cells, 5 cell types, 30 genes (counts layer, all
    highly variable), batch and cell-type names as the X3 identity rows refer to them (Batch3Sub1, Batch4Sub2, Group1).
    It only has to let the plan be built; no scientific value is read from it."""
    import anndata as ad
    rng = np.random.default_rng(0)
    batches = ["Batch1Sub1", "Batch2Sub1", "Batch3Sub1", "Batch4Sub2", "Batch5Sub2", "Batch6Sub2"]
    n = 4000 * len(batches)
    obs = pd.DataFrame({"batch": pd.Categorical(np.repeat(batches, 4000)),
                        "celltype": pd.Categorical(rng.choice([f"Group{i}" for i in range(1, 6)], n))},
                       index=[f"cell{i}" for i in range(n)])
    counts = rng.poisson(2.0, (n, 30)).astype(np.float32)
    a = ad.AnnData(X=counts.copy(), obs=obs, var=pd.DataFrame({"highly_variable": True}, index=[f"g{i}" for i in range(30)]))
    a.layers["counts"] = counts
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a.write_h5ad(path)


def main(checkout, label, out, rows, exit_codes):
    checkout = os.path.abspath(checkout)
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                      KMP_AFFINITY="disabled", PYTHONWARNINGS="ignore", WCD_SRC=f"{checkout}/src", MANIFEST=rows,
                      OUT_DIR=f"{out}/out_{label}")
    sys.path.insert(0, f"{checkout}/scripts")
    os.chdir(checkout)
    import torch
    torch.set_num_threads(1)
    from scvi.train import _trainer
    steps = []
    init, fit = _trainer.Trainer.__init__, _trainer.Trainer.fit

    def init0(self, *a, **k):
        k.update(limit_train_batches=0, limit_val_batches=0, num_sanity_val_steps=0)
        init(self, *a, **k)

    def fit0(self, *a, **k):
        res = fit(self, *a, **k)
        steps.append(int(self.global_step))
        return res
    _trainer.Trainer.__init__, _trainer.Trainer.fit = init0, fit0
    tags = pd.read_csv(rows, sep="\t", comment="#", dtype=str).tag.tolist()
    os.makedirs(f"{out}/out_{label}", exist_ok=True)
    failed = 0
    with open(f"{out}/out_{label}/steps.txt", "w") as fs, open(exit_codes, "a") as fc:
        for t in tags:
            os.environ["TAG"] = t
            del steps[:]
            try:
                runpy.run_path(f"{checkout}/scripts/fit_paper_config.py", run_name="__main__")
                code = 0
            except SystemExit as e:
                code = 0 if e.code in (None, 0) else (e.code if isinstance(e.code, int) else 1)
            except BaseException:   # fl: allow FL101 per-row failure is recorded (traceback file, exit code 1, final sys.exit); other rows still run so the comparison lists every failing row
                with open(f"{out}/out_{label}/{t}.log", "w") as lf:
                    traceback.print_exc(file=lf)
                code = 1
            if not steps:
                code = code or 3         # no trainer ran: nothing to compare
            fs.write(f"{t} {sum(steps)}\n"); fs.flush()
            fc.write(f"{label} {t} {code}\n"); fc.flush()
            failed += code != 0
    if failed:
        sys.exit(f"{label}: {failed} of {len(tags)} rows failed (exit codes in {exit_codes}, tracebacks in {out}/out_{label}/)")


if __name__ == "__main__":
    if sys.argv[1] == "--make-synthetic":
        make_synthetic(sys.argv[2])
    else:
        main(*sys.argv[1:6])
