"""Stand-in for the fit and scoring interpreters in tests/stage: the runner calls `<python> -c <probe>`,
`<python> scripts/fit_paper_config.py` and `<python> scripts/score_scib_native.py`; this script answers each with the
contract of the real program (outputs, status records, exit codes), driven per tag by a JSON plan.

Env: FAKE_PLAN (json {tag: behaviour or [behaviour of attempt 1, 2, ...]}), FAKE_SCORE_PLAN (same, scorer),
     FAKE_STATE (dir for attempt counters), FAKE_DEVICE (default 'cpu'), FAKE_SLEEP (s, default 60).
Fit behaviours: ok, nan, diverge, crash, sleep, no_latent, wrong_device, diverge_no_status.
Score behaviours: ok, nan, crash.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fit_outcome  # noqa: E402

BATCH = ["PCR_batch", "ASW_label/batch", "iLISI", "graph_conn", "kBET"]
BIO = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "isolated_label_F1", "isolated_label_silhouette",
       "cLISI", "cell_cycle_conservation", "trajectory"]


def behaviour(plan_var, tag, kind):
    plan = {}
    if os.environ.get(plan_var):
        with open(os.environ[plan_var]) as f:
            plan = json.load(f)
    b = plan.get(tag, "ok")
    counter = os.path.join(os.environ["FAKE_STATE"], f"{kind}.{tag}")
    n = 1
    if os.path.exists(counter):
        with open(counter) as f:
            n = int(f.read()) + 1
    with open(counter, "w") as f:
        f.write(str(n))
    if isinstance(b, list):
        return b[min(n, len(b)) - 1]
    return b


def load_row(manifest, tag):
    import pandas as pd
    m = pd.read_csv(manifest, sep="\t", comment="#", dtype=str, keep_default_na=False)
    return m[m.tag == tag].iloc[0].to_dict()


def fit():
    tag, out = os.environ["TAG"], os.environ["OUT_DIR"]
    row = load_row(os.environ["MANIFEST"], tag)
    b = behaviour("FAKE_PLAN", tag, "fit")
    print(f"fake fit {tag}: {b} threads={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    if b == "sleep":
        time.sleep(float(os.environ.get("FAKE_SLEEP", "60")))
        b = "ok"
    if b == "crash":
        print("Traceback (most recent call last):\nRuntimeError: CUDA out of memory (fake)", file=sys.stderr)
        sys.exit(1)
    if b == "no_latent":
        sys.exit(0)
    if b == "diverge_no_status":
        sys.exit(fit_outcome.EXIT_DIVERGED)
    if b == "diverge":
        fit_outcome.write_status(out, tag, "diverged", detail="non-finite training loss at epoch 3, step 41: "
                                 "gen_loss=inf", row=row, epoch=3, step=41, terms={"gen_loss": float("inf")})
        sys.exit(fit_outcome.EXIT_DIVERGED)
    n, k = 30, int(row["n_latent"])
    z = np.random.default_rng(0).normal(size=(n, k)).astype(np.float32)
    if b == "nan":
        z[3, 1] = np.nan
    device = "NVIDIA A100-SXM4-80GB" if b == "wrong_device" else os.environ.get("FAKE_DEVICE", "cpu")
    cfg = dict(row=row, reference_name="b0", n_cells=n, n_batches=3, fit_seconds=0.1, iw_keep_fraction=None,
               git_sha="fake", git_dirty=False, scvi="fake", torch="fake", gpu=device)
    npz = os.path.join(out, "latents", f"{tag}.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, z=z, obs_names=np.array([f"c{i}" for i in range(n)]),
                        batch=np.array([f"b{i % 3}" for i in range(n)]), celltype=np.array(["t"] * n),
                        config=json.dumps(cfg), history=json.dumps({}))
    os.makedirs(os.path.join(out, "models", tag), exist_ok=True)
    with open(os.path.join(out, "models", tag, "model.pt"), "w") as f:
        f.write("fake")
    os.replace(tmp, npz)


def score():
    tag = os.environ["TAG"]
    b = behaviour("FAKE_SCORE_PLAN", tag, "score")
    if b == "crash":
        sys.exit(1)
    meta = json.loads(os.environ["META"])
    row = dict(tag=tag, **meta)
    row.update({m: 0.5 for m in BATCH + BIO})
    if meta["task"] not in ("pancreas", "lung", "immune", "immune_hum_mou"):
        row["cell_cycle_conservation"] = float("nan")       # not computed for this task: not required
    if meta["task"] not in ("immune", "immune_hum_mou"):
        row["trajectory"] = float("nan")
    if b == "nan":
        row["kBET"] = float("nan")
    row.update(kbet_seed=int(os.environ["KBET_SEED"]), kbet_r_calls=1, score_seconds=0.1)
    import pandas as pd
    tmp = os.environ["OUT_CSV"] + ".tmp"
    pd.DataFrame([row]).to_csv(tmp, index=False)
    os.replace(tmp, os.environ["OUT_CSV"])


if __name__ == "__main__":
    if sys.argv[1] == "-c":
        if "torch" in sys.argv[2]:
            print(json.dumps(dict(torch="fake", scvi="fake", device=os.environ.get("FAKE_DEVICE", "cpu"))))
        else:
            print("kBET stack OK")
    elif sys.argv[1].endswith("fit_paper_config.py"):
        fit()
    elif sys.argv[1].endswith("score_scib_native.py"):
        score()
    else:
        raise SystemExit(f"fake_env: unexpected arguments {sys.argv[1:]}")
