#!/usr/bin/env python3
"""Independent check of the sweep analysis' placement numbers (docs/ANALYSIS_SPEC.md sections 5, 5.1, 10, 11).

Recomputes from the raw score CSVs (inputs/scores_list.txt gives the files, inputs/manifest.tsv the run-list rows: task, arm,
lambda, decoder condition, seed, latent size) WITHOUT importing any pipeline module (only pandas, numpy, json, math):
  (a) equal-tries placement: per task and decoder, each penalty's expected best of k strengths E_k against plain scVI, Harmony,
      Scanorama, scANVI, sysVI, scvi_adv (family E_k at the same k; single-setting methods: their S) and PCA's S;
      and the mean over the real tasks per family and k (paper_numbers best_of_k_real_mean.*), single-setting S (single.*);
  (b) hypervolume per penalty, task and decoder (corner (b0, c0 - 0.10), divided by b_common - b0; null if < 0.02);
  (c) share of non-dominated points per penalty, task and decoder.
Every recomputed value is compared with paper_numbers.json (tolerance 1e-9); the per-task E_k are also compared with
tables/best_of_k.csv (paper_numbers.json holds only their real-task means).

usage: check_sweep_placement.py <analysis-out-dir> <check-out-dir>
Writes <check-out-dir>/{check_results.csv, placement_equal_tries.csv, placement_summary.csv, check_summary.json}; exit 1 on any mismatch.
"""
import json, math, os, sys
import numpy as np
import pandas as pd

TOL = 1e-9
BATCH = ["PCR_batch", "ASW_label/batch", "iLISI", "graph_conn", "kBET"]
BIO = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "isolated_label_F1", "isolated_label_silhouette", "cLISI"]
CC_TASKS = ("pancreas", "lung", "immune", "immune_hum_mou")
TRAJ_TASKS = ("immune", "immune_hum_mou")
REAL = ["pancreas", "lung", "immune", "immune_hum_mou", "atac_small", "atac_large"]
PENALTIES = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
TOOLS = ["harmony", "scanorama", "sysvi"]
SEEDS_PER_SETTING = {"harmony": 1, "scanorama": 1, "sysvi": 5}   # run list (docs/SWEEP_ANALYSIS.md item 11); a setting lacking a seed is incomplete
WB, WC = 0.4, 0.6          # S = 0.4 B + 0.6 C (spec 10)


def load(out):
    M = pd.read_csv(os.path.join(out, "inputs/manifest.tsv"), sep="\t", comment="#", dtype=str, keep_default_na=False)
    files = [l.strip() for l in open(os.path.join(out, "inputs/scores_list.txt")) if l.strip()]
    S = pd.concat([pd.read_csv(f, dtype={"tag": str}) for f in files], ignore_index=True)
    assert not S.tag.duplicated().any(), "a tag is scored twice"
    fails = pd.read_csv(os.path.join(out, "inputs/failures.csv"), dtype=str, keep_default_na=False)
    assert len(fails) == 0, "this check assumes no failed runs (counts of failed runs differ)"
    df = M.merge(S, on="tag", how="inner", suffixes=("", "_s"))
    assert len(df) == len(S), "score rows without a manifest row"
    for c in ("task", "arm", "cond", "seed"):          # the score file must agree with the run list
        assert (df[c].astype(str) == df[c + "_s"].astype(str)).all(), f"{c} differs between run list and score file"
    assert (df.lam.astype(float) == df.lam_s.astype(float)).all()
    B, C = [], []
    for r in df.to_dict("records"):
        bio = BIO + (["cell_cycle_conservation"] if r["task"] in CC_TASKS else []) + (["trajectory"] if r["task"] in TRAJ_TASKS else [])
        B.append(np.mean([float(r[m]) for m in BATCH]))
        C.append(np.mean([float(r[m]) for m in bio]))
    df["B"], df["C"] = B, C
    assert np.isfinite(df.B).all() and np.isfinite(df.C).all()
    df["S"] = WB * df.B + WC * df.C
    df["lamf"], df["condi"] = df.lam.astype(float), df.cond.astype(int)
    return df


def best_of_k(scores):
    s = sorted(scores)
    n = len(s)
    return [sum(s[i - 1] * math.comb(i - 1, k - 1) for i in range(k, n + 1)) / math.comb(n, k) for k in range(1, n + 1)]


def hypervolume(points, b_ref, c_ref):
    """Area of the union of rectangles [b_ref, B] x [c_ref, C] (strip decomposition on the batch axis)."""
    pts = [(b, c) for b, c in points if b > b_ref and c > c_ref]
    edges = sorted({b_ref} | {b for b, _ in pts})
    area = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        height = max([c for b, c in pts if b >= hi] + [c_ref]) - c_ref      # a point reaches the whole strip iff its B >= hi
        area += (hi - lo) * height
    return area


def non_dominated(points):
    def dominates(q, p):    # q at least as good on both (within TOL) and better on one by more than TOL (spec 5.1)
        return q[0] >= p[0] - TOL and q[1] >= p[1] - TOL and (q[0] > p[0] + TOL or q[1] > p[1] + TOL)
    return sum(not any(dominates(q, p) for q in points if q is not p) for p in points)


def main(argv):
    out, chk = argv
    os.makedirs(chk, exist_ok=True)
    df = load(out)
    PN = json.load(open(os.path.join(out, "paper_numbers.json")))
    BOK = pd.read_csv(os.path.join(out, "tables/best_of_k.csv"))
    tasks = sorted(df.task.unique())
    A1, base = df[df.experiment == "A1"], df[df.experiment != "A1"]
    results = []

    def compare(key, value, what, source="paper_numbers.json", ref=None):
        if ref is None:
            ref = PN[key]["value"] if key in PN else "MISSING KEY"
        numeric = isinstance(ref, (int, float)) and value is not None
        diff = abs(float(value) - float(ref)) if numeric else None
        ok = (diff <= TOL) if numeric else (ref is None and value is None)     # null must match null
        results.append(dict(key=key, what=what, source=source, recomputed=value, reported=ref, abs_diff=diff, match=bool(ok)))

    fam_E, single_S, rows = {}, {}, []
    for task in tasks:
        for cond in (0, 1):
            pen = A1[(A1.task == task) & (A1.condi == cond)]
            plain = pen[pen.arm == "none"]
            b0, c0 = plain.B.mean(), plain.C.mean()
            plain_S = float(plain.S.mean())
            pts, scores = {}, {}
            for arm in PENALTIES:
                g = pen[pen.arm == arm].groupby("lamf")
                pts[arm] = [(float(g.get_group(l).B.mean()), float(g.get_group(l).C.mean())) for l in sorted(g.groups)]
                scores[arm] = [float(g.get_group(l).S.mean()) for l in sorted(g.groups)]
            E = {arm: best_of_k(scores[arm]) for arm in PENALTIES}
            E["none"] = best_of_k([plain_S])
            # baselines: decoder-free and conditioned-only rows enter both decoders' cells; a setting counts only with all its expected seeds
            bt = base[base.task == task]
            for arm in TOOLS:
                g = bt[(bt.arm == arm) & ((bt.n_latent == "10") if arm != "sysvi" else True)]
                exp = SEEDS_PER_SETTING[arm]
                sets = [float(x.S.mean()) for _, x in g.groupby("lamf") if len(x) == exp]
                E[arm] = best_of_k(sets) if sets else []
            for arm in ("scanvi", "scvi_adv"):
                g = bt[bt.arm == arm]
                E[arm] = best_of_k([float(g.S.mean())])
            pca = bt[(bt.arm == "pca") & (bt.n_latent == "10")]
            pca_S = float(pca.S.mean())
            single_S[(task, cond)] = dict(none=plain_S, scanvi=E["scanvi"][0], scvi_adv=E["scvi_adv"][0], pca10=pca_S)
            fam_E[(task, cond)] = E
            # (b), (c)
            Ms = [max(b for b, _ in pts[arm]) for arm in PENALTIES]
            b_common = min(Ms)
            scale = b_common - b0
            low = scale < 0.02
            compare(f"pair.{task}.cond{cond}.b_common", b_common, "b_common")
            for arm in PENALTIES:
                hv = None if low else hypervolume(pts[arm], b0, c0 - 0.10) / scale
                compare(f"hypervolume.{arm}.{task}.cond{cond}", hv, "hypervolume")
                n = len(pts[arm])
                compare(f"nondominated_share.{arm}.{task}.cond{cond}", non_dominated(pts[arm]) / n, "non-dominated share")
            # (a) per-task E_k against the table
            for fam, e in E.items():
                for k, v in enumerate(e, 1):
                    t = BOK[(BOK.task == task) & (BOK.cond == cond) & (BOK.family == fam) & (BOK.k == k)]
                    compare(f"E_k.{fam}.{task}.cond{cond}.k{k}", v, "per-task expected best of k (table)", "tables/best_of_k.csv",
                            float(t.E.iloc[0]) if len(t) == 1 else "MISSING ROW")
            # placement rows
            for arm in PENALTIES:
                for tool in ["none", *TOOLS, "scanvi", "scvi_adv"]:
                    et = E[tool]
                    for k in range(1, (len(et) if len(et) > 1 else 6) + 1):
                        ref = et[k - 1] if len(et) > 1 else et[0]
                        rows.append(dict(task=task, cond=cond, penalty=arm, tool=tool, k=k, E_penalty=E[arm][k - 1], E_tool=ref,
                                         diff=E[arm][k - 1] - ref, penalty_better=bool(E[arm][k - 1] > ref)))
                for k in range(1, 7):
                    rows.append(dict(task=task, cond=cond, penalty=arm, tool="pca", k=k, E_penalty=E[arm][k - 1], E_tool=pca_S,
                                     diff=E[arm][k - 1] - pca_S, penalty_better=bool(E[arm][k - 1] > pca_S)))
    # single-setting S: look up keys by suffix because the key embeds the pipeline's setting name
    singles = {k: v["value"] for k, v in PN.items() if k.startswith("single.") and k.endswith(".S")}
    for (task, cond), d in single_S.items():
        for name, v in d.items():
            tag = {"none": "none|", "scanvi": "scanvi|", "scvi_adv": "scvi_adv|", "pca10": "pca|"}[name]
            hits = [k for k in singles if k.startswith("single." + tag) and k.endswith(f".{task}.cond{cond}.S") and (name != "pca10" or "|10." in k)]
            if len(hits) == 1:
                compare(hits[0], v, f"single-setting S ({name})")
            else:
                results.append(dict(key=f"single.{name}.{task}.cond{cond}.S", what="single-setting S", source="paper_numbers.json",
                                    recomputed=v, reported=f"{len(hits)} KEYS", abs_diff=None, match=False))
    # real-task means per family and k: mean over the real tasks with N >= k
    for cond in (0, 1):
        fams = sorted({f for (t, c), E in fam_E.items() if c == cond for f in E})
        for fam in fams:
            per = [fam_E[(t, cond)][fam] for t in REAL if (t, cond) in fam_E]
            for k in range(1, max(len(e) for e in per) + 1):
                v = [e[k - 1] for e in per if len(e) >= k]
                compare(f"best_of_k_real_mean.cond{cond}.{fam}.k{k}", float(np.mean(v)), f"mean over {len(v)} real tasks")
    R = pd.DataFrame(results)
    P = pd.DataFrame(rows)
    R.to_csv(os.path.join(chk, "check_results.csv"), index=False)
    P.to_csv(os.path.join(chk, "placement_equal_tries.csv"), index=False)
    # headline: number of real tasks (of 6) on which the penalty's E_k exceeds the tool's at equal k
    realP = P[P.task.isin(REAL)]
    summ = realP[realP.k.isin([1, 6])].groupby(["cond", "penalty", "tool", "k"]).agg(n_real_tasks_better=("penalty_better", "sum"), n_real_tasks=("penalty_better", "size")).reset_index()
    summ.to_csv(os.path.join(chk, "placement_summary.csv"), index=False)
    bad = R[~R.match]
    kinds = R.groupby("what").agg(n=("match", "size"), matched=("match", "sum")).reset_index()
    json.dump(dict(n_checked=len(R), n_matched=int(R.match.sum()), n_mismatched=len(bad), tolerance=TOL,
                   max_abs_diff=float(R.abs_diff.dropna().max()), by_kind=kinds.to_dict("records"),
                   mismatches=bad.head(50).to_dict("records")), open(os.path.join(chk, "check_summary.json"), "w"), indent=1, default=str)
    print(f"checked {len(R)} values, matched {int(R.match.sum())}, mismatched {len(bad)}, max abs diff {R.abs_diff.dropna().max():.3e}")
    print(kinds.to_string())
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
