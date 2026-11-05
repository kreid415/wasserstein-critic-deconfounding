#!/usr/bin/env python
"""Extension run list for the replication agreement rule (AGREEMENT.md line 9, tag replication-prediction-v1):
for each reading whose new-seed means do not bracket b*, add the next sweep-grid strength outward, seeds 0-4.
Input: the unreached readings (replication_readings.csv, status unreached_low/high), the replication run list, and the committed
X1 manifest (same source as replication_design.py). Output: manifests/sweep_replication_ext1.tsv (+ .sha256).
Strengths with a committed X1 row are copied; others are built with prereg_rules.extension_rows (as replication_design.py did).
Run: <scvi-api python> extension_rows.py   (from the prereg-tier12-v3 checkout's scripts/ directory)
"""
import hashlib, math, sys
import os, sys
import pandas as pd
sys.path.insert(0, os.getcwd())
import prereg_rules as P

D = "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
REPO = D + "/code/wt_sweep_replication"
SRC = D + "/code/wcd_final-review-fixes/manifests/paper_manifest_stock_pilot_u5_b10_v3.x1_sinkhorn_redo.tsv"
READ = REPO + "/experiments/sweep_replication/agreement_result/replication_readings.csv"
REP = REPO + "/manifests/sweep_replication_x1.tsv"
OUT = REPO + "/manifests/sweep_replication_ext1.tsv"
GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]  # sweep grid (A1 grid of all 8 tasks, checked against the sweep manifests)
SEEDS = {"0", "1", "2", "3", "4"}

def main():
    R = pd.read_csv(READ)
    U = R[R.status.str.startswith("unreached")]
    assert len(U) == 8 and set(U.status) == {"unreached_low"}, U.status.tolist()
    S, _ = P.read_manifest(SRC)
    S = S[S.experiment == "X1"]
    n_built = 0
    REPM = pd.read_csv(REP, sep="\t", comment="#", dtype=str)
    rows = []
    for u in U.itertuples():
        cond = str(int(u.cond))
        lm = REPM[(REPM.task == u.task) & (REPM.cond == cond) & (REPM.arm == u.arm)]
        lam_max = max(float(x) for x in lm.lam)
        assert math.isclose(lam_max, float(u.lam_max)), (u, lam_max)
        i = [k for k, g in enumerate(GRID) if math.isclose(g, lam_max)]
        assert len(i) == 1 and i[0] + 1 < len(GRID)
        nxt = GRID[i[0] + 1]
        sel = S[(S.task == u.task) & (S.cond == cond) & (S.arm == u.arm) & (pd.to_numeric(S.lam) == nxt)]
        if not len(sel):
            sel = P.extension_rows(S, [dict(experiment="X1", task=u.task, cond=int(cond), arm=u.arm, lam=float(nxt))])
            n_built += 5
        assert sorted(sel.seed) == sorted(SEEDS), (u.task, cond, u.arm, nxt, sorted(sel.seed))
        # same settings as the replication rows at lam_max except tag, lam, seed
        keep = [c for c in S.columns if c not in ("tag", "lam", "seed")]
        a = sel[keep].drop_duplicates(); b = lm[keep].drop_duplicates()
        assert len(a) == 1 and len(b) == 1 and a.iloc[0].tolist() == b.iloc[0].tolist(), (u.task, u.arm, a, b)
        rows.append(sel)
    X = pd.concat(rows, ignore_index=True)
    assert len(X) == 40 and not X.tag.duplicated().any() and not set(X.tag) & set(REPM.tag)
    hdr = "# replication extension (AGREEMENT.md line 9): next sweep-grid strength above the highest replication strength, seeds 0-4, 8 unreached_low readings; rows copied from the committed X1 manifest " + SRC.split("/")[-3] + "/" + SRC.split("/")[-1] + "\n"
    with open(OUT, "w") as f:
        f.write(hdr); X.to_csv(f, sep="\t", index=False)
    h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    open(OUT + ".sha256", "w").write(f"{h}  {OUT.split('/')[-1]}\n")
    local = X[X.task.isin(["atac_small", "sim1"])]  # workstation tasks, as in the sweep and the replication
    remote = X[~X.tag.isin(local.tag)]
    assert len(local) == 15 and len(remote) == 25 and set(remote.task) == {"immune_hum_mou", "pancreas", "sim2"}
    for name, T, note in (("ext1_local.tags", local, "workstation RTX 3080"), ("replication_sE.tags", remote, "JHPCE L40S shard E")):
        p = REPO + "/experiments/sweep_replication/" + ("" if name.startswith("ext1") else "jhpce_stage/") + name
        open(p, "w").write(f"# replication extension rows ({note}), sha256 of the run list in {OUT.split('/')[-1]}.sha256\n" + "\n".join(T.tag) + "\n")
        print(p.split("/")[-1], len(T), hashlib.sha256(open(p, "rb").read()).hexdigest())
    print(len(X), "built with extension_rows:", n_built, h)
    print(X.groupby(["task", "cond", "arm", "lam"]).size().to_string())

if __name__ == "__main__":
    main()
