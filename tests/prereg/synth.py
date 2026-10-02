"""Synthetic manifests and score tables for the pre-registered rules (no fits, no scoring).

A 'world' maps a manifest row to (B, C): seed-paired noise around a dose-response in log10(lambda)
with a batch gain that saturates and a bio loss that sets in a decade later. Score rows carry the
builder's tags and every scIB metric column that score_scib_native.py writes; batch metrics = B,
the task's bio metrics = C, metrics scIB does not compute for the task = NaN.
"""
import hashlib
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_paper_manifest as BPM  # noqa: E402
import prereg_rules as P  # noqa: E402

ALL_METRICS = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "ASW_label/batch", "PCR_batch",
               "cell_cycle_conservation", "isolated_label_F1", "isolated_label_silhouette", "graph_conn",
               "kBET", "iLISI", "cLISI", "hvg_overlap", "trajectory"]
CENTER = {"discriminator": 1.0, "reference": 1.5, "pooled": 1.5, "barycenter": 1.5, "mmd": 1.0, "sinkhorn": 0.8}


def pilot_rows(design="pilot"):
    R = BPM.finalize(BPM.build(BPM.BACKBONES["stock"], design, 3, 5, 8))
    return pd.DataFrame([{c: str(r[c]) for c in BPM.COLS} for r in R if not r["reuses_x1"]], columns=BPM.COLS)


def write_tsv(M, path, header="# synthetic test manifest"):
    with open(path, "w") as f:
        f.write(header + "\n" + "\t".join(BPM.COLS) + "\n")
        for r in M[BPM.COLS].itertuples(index=False):
            f.write("\t".join(str(x) for x in r) + "\n")
    return path


def _noise(key, sd):
    h = int(hashlib.sha1(key.encode()).hexdigest()[:12], 16)
    return float(np.random.default_rng(h).normal(0, sd))


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def world(r, center=None, gain=0.30, loss=0.30, sd=0.002, shift=None):
    """Default world: no effect at lambda=0.1, batch saturates ~1 decade above the arm's centre and bio
    collapses ~1 decade later; every arm reaches both inside the A1 grid."""
    center = dict(CENTER, **(center or {}))
    b0 = 0.40 + 0.01 * sorted(BPM.TASKS).index(r["task"]) + 0.02 * int(r["cond"])
    c0 = 0.70
    seedkey = f"{r['task']}|{r['cond']}|{r['seed']}"
    B = b0 + _noise("b" + seedkey, 0.02)          # seed effect shared by every arm (pairing)
    C = c0 + _noise("c" + seedkey, 0.02)
    if r["arm"] in P.BASE_ARM and r["arm"] not in P.NO_ADVERSARY:
        x = math.log10(float(r["lam"]))
        m = center[P.BASE_ARM[r["arm"]]] + (shift(r) if shift else 0.0)
        B += gain * sig((x - m) / 0.25) + _noise("B" + r["tag"], sd)
        C -= loss * sig((x - m - 1.0) / 0.25) + _noise("C" + r["tag"], sd)
    return "ok", min(max(B, 0.0), 1.0), min(max(C, 0.0), 1.0)


def scores(M, fn=world):
    """Score rows (and failure rows) for every manifest row of M."""
    S, F = [], []
    for r in M.to_dict("records"):
        res = fn(r)
        if res[0] != "ok":
            F.append(dict(tag=r["tag"], status=res[0], detail="synthetic"))
            continue
        _, B, C = res
        bio = P.bio_metrics_for(r["task"])
        row = {"tag": r["tag"], "score_seconds": 1.0}
        for m in ALL_METRICS:
            row[m] = B if m in P.BATCH_METRICS else (C if m in bio else float("nan"))
        S.append(row)
    return (pd.DataFrame(S, columns=["tag"] + ALL_METRICS + ["score_seconds"]),
            pd.DataFrame(F, columns=["tag", "status", "detail"]))


def write_scores(S, F, d, name="scores"):
    os.makedirs(d, exist_ok=True)
    sp, fp = os.path.join(d, f"{name}.csv"), os.path.join(d, f"{name}_failures.csv")
    S.to_csv(sp, index=False)
    F.to_csv(fp, index=False)
    return sp, fp


def out_frame(records):
    """A hand-built outcomes() frame: records with experiment, task, cond, arm, lam, seed, B, C[, status]."""
    df = pd.DataFrame(records)
    df["status"] = df["status"].fillna("ok") if "status" in df else "ok"
    df["tag"] = [f"t{i}" for i in range(len(df))]
    df["cond_i"] = df.cond.astype(int)
    df["seed_i"] = df.seed.astype(int)
    df["lamf"] = df.lam.astype(float)
    df["lam"] = [P.fmt_lam(v) for v in df.lamf]
    for c, v in (("adv_input", "mean"), ("zstd", "0")):
        if c not in df:
            df[c] = v
    return df
