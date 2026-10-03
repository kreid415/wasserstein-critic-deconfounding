#!/usr/bin/env python
"""Score ONE latent with scIB's own entry point, scib.metrics.metrics(), against the unintegrated
prepped data. Replaces the project's hand-assembled composite (score_final_config.py), which
added raw scib.me.pcr with the wrong sign (code audit 2026-10-01, C1).

What runs (scIB, Luecken et al. 2022, Supplementary Table 2, embedding outputs):
  batch: PCR (pcr_comparison vs unintegrated PCA), batch ASW, graph iLISI, graph connectivity, kBET
  bio:   NMI, ARI (optimal-resolution clustering), cell-type ASW, isolated-label F1 and ASW,
         graph cLISI, cell-cycle conservation (where the organism's cell-cycle genes are present),
         trajectory conservation (where the prepped file has obs dpt_pseudotime: immune, immune_hum_mou;
         scib-pipeline scripts/metrics/metrics.py computes it iff 'dpt_pseudotime' is in obs)
  not applicable to embeddings: HVG conservation.
BIO_METRICS includes trajectory conservation, as scIB's bio score does (scib-reproducibility
visualization/plotSingleTaskRNA.R: group_bio lines 45-46; bio score = rowMeans over the available scaled
bio metrics, lines 142-143). Where it is not computed it is NaN and skipped (CONSTRAINTS.md SI-28).
This script writes RAW metric values only. The scIB overall score (per-metric min-max scaling
within a dataset across all runs, then 0.4*batch + 0.6*bio) is computed at analysis time by
scib_overall() below, because the scaling depends on the full set of runs.

Env: PREPPED (prep_scib_task.py output: X = scIB's normalised values, obsm X_pca, obs batch/celltype,
     uns organism / modality), NPZ (latent: z, batch, celltype, obs_names), TAG, META (json), OUT_CSV,
     KBET_SEED (optional integer, default KBET_SEED_DEFAULT = 0).
kBET is deterministic: R's RNG is set with set.seed(KBET_SEED) immediately before every R kBET call (see
seed_kbet); the seed and the number of seeded R calls are written as kbet_seed / kbet_r_calls. The
measurement-SD re-scoring (PAPER_PLAN section 5) varies KBET_SEED explicitly.
Per-task settings follow scIB (code review N13): cell-cycle conservation only for RNA tasks with a
real organism (scIB did not compute it for ATAC, and simulations have no cell-cycle genes);
trajectory conservation for the immune tasks (obs dpt_pseudotime, scIB's trajectory label); the
organism comes from the prepped file, never from a default. Subsampled fits (X3, X8) are scored
against the same cells of the unintegrated data (matched by obs_names; every latent cell must be in the prepped
file exactly once, code check CR-14).
PCR reference (code check CR-01): the prepped files carry obsm X_pca but no uns['pca'], so scib.metrics.pcr()
recomputes the unintegrated PCA on adata.X with ALL features (scib 1.1.7 pcr.py, pc_regression with
use_highly_variable=False). That is scIB's own convention (scib-pipeline scores against the scenario's original
file) and the full-data path is unchanged. A subsample is scored by the same convention: no PCA is precomputed
for it, so scib recomputes the all-feature PCA of the subsample's cells. A subsampled latent against a prepped
file that does carry uns['pca'] is refused (the reference would silently become rows of the full-data PCA).
Every row also records (code check CR-05, CR-08; columns pinned by the lead 2026-10-03; values only read, no
metric changes): scorer_git_sha, scorer_dirty (0/1), score_host, cpu_model, cpu_simd (highest of avx512f / avx2 /
sse4_2 in /proc/cpuinfo flags, else 'none'), numba_cpu_name (env NUMBA_CPU_NAME or ''), scorer_versions (JSON:
scib, scanpy, anndata, numba, pynndescent, scikit-learn, numpy, rpy2, R_kBET), and scib's silent fallbacks:
traj_root_fallback (1 if trajectory conservation hit RootCellError and scib set it to 0, 0 if a root cell was
found, '' if trajectory is not computed), kbet_labels_forced_one (labels whose kBET rejection rate scib set to
1: fewer than 75% of the label's cells in large components, or not enough neighbours), kbet_labels_skipped
(labels with cells that scib left NaN and dropped from its mean: < 10 cells or a single batch, or an R error
reading the kBET summary).
"""
import importlib, os, json, socket, subprocess, time, numpy as np, pandas as pd, scanpy as sc, scib

BATCH_METRICS = ["PCR_batch", "ASW_label/batch", "iLISI", "graph_conn", "kBET"]
BIO_METRICS = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "isolated_label_F1",
               "isolated_label_silhouette", "cLISI", "cell_cycle_conservation", "trajectory"]
KBET_SEED_DEFAULT = 0   # R seed set before every R kBET call; env KBET_SEED overrides (measurement-SD re-scoring)
VERSION_PACKAGES = ["scib", "scanpy", "anndata", "numba", "pynndescent", "scikit-learn", "numpy", "rpy2"]
PROVENANCE_COLS = ["scorer_git_sha", "scorer_dirty", "score_host", "cpu_model", "cpu_simd", "numba_cpu_name",
                   "scorer_versions"]
DIAGNOSTIC_COLS = ["traj_root_fallback", "kbet_labels_forced_one", "kbet_labels_skipped"]


def kbet_seed():
    """The R seed for kBET: env KBET_SEED (an integer) or KBET_SEED_DEFAULT."""
    v = os.environ.get("KBET_SEED", str(KBET_SEED_DEFAULT))
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"KBET_SEED={v!r} is not an integer") from None


def seed_kbet(seed):
    """Make kBET deterministic. scib 1.1.7's wrapper (scib/metrics/kbet.py) sets no seed and R's kBET samples
    test cells (testSize, n_repeat), so kBET differs on identical latents. scib.metrics.kbet.kBET calls the
    module-level kBET_single once per cell type; it is wrapped so that R's RNG is set with set.seed(seed)
    immediately before every R kBET call. Returns the wrapper (its .calls counts the seeded R calls)."""
    import rpy2.robjects as ro
    import scib.metrics.kbet as kbet_mod
    base = getattr(kbet_mod.kBET_single, "__wrapped__", kbet_mod.kBET_single)

    def seeded(*args, **kwargs):
        ro.r(f"set.seed({int(seed)})")
        seeded.calls += 1
        score = base(*args, **kwargs)
        if score != score:                 # NaN: scib could not read the R kBET summary (kbet.py, kBET_single)
            seeded.nan_returns += 1
        return score

    seeded.__wrapped__, seeded.seed, seeded.calls, seeded.nan_returns = base, int(seed), 0, 0
    kbet_mod.kBET_single = seeded
    return seeded


def count_kbet_neighbours():
    """Wrap scib's diffusion_nn as kBET calls it (module global of scib.metrics.kbet; once per tested label that has
    a single component or >= 75% of its cells in large components). Counts calls and NeighborsError raises (scib
    then sets the label's rejection rate to 1). Values pass through unchanged."""
    kbet_mod = importlib.import_module("scib.metrics.kbet")
    base = getattr(kbet_mod.diffusion_nn, "__wrapped__", kbet_mod.diffusion_nn)

    def counted(*args, **kwargs):
        counted.calls += 1
        try:
            return base(*args, **kwargs)
        except kbet_mod.NeighborsError:
            counted.neighbors_errors += 1
            raise

    counted.__wrapped__, counted.calls, counted.neighbors_errors = base, 0, 0
    kbet_mod.diffusion_nn = counted
    return counted


def watch_trajectory_root():
    """Wrap scib's get_root as trajectory_conservation calls it (module global of scib.metrics.trajectory). Counts
    calls and RootCellError raises (scib then returns trajectory conservation 0). Values pass through unchanged."""
    traj = importlib.import_module("scib.metrics.trajectory")
    base = getattr(traj.get_root, "__wrapped__", traj.get_root)

    def watched(*args, **kwargs):
        watched.calls += 1
        try:
            return base(*args, **kwargs)
        except traj.RootCellError:
            watched.root_errors += 1
            raise

    watched.__wrapped__, watched.calls, watched.root_errors = base, 0, 0
    traj.get_root = watched
    return watched


def kbet_label_flags(obs, label_key, batch_key, seeded, nn):
    """(forced_one, skipped) for one kBET run, from scib 1.1.7's own label rule (kbet.py: labels with >= 10 cells and
    > 1 batch are tested, the others are NaN) and the wrapped calls: a tested label without a diffusion_nn call
    failed the 75% rule, and one whose diffusion_nn raised NeighborsError had too few neighbours; both are set to 1.
    Skipped = labels with cells that are not tested plus tested labels whose R summary read gave NaN."""
    n = obs[label_key].value_counts()
    n = n[n > 0]
    nb = obs.groupby(label_key, observed=True)[batch_key].nunique().reindex(n.index)
    tested = int(((n >= 10) & (nb > 1)).sum())
    if nn.calls > tested or seeded.calls != nn.calls - nn.neighbors_errors:
        raise RuntimeError(f"kBET call accounting does not match scib 1.1.7 (tested labels {tested}, diffusion_nn "
                           f"{nn.calls} ({nn.neighbors_errors} NeighborsError), R kBET calls {seeded.calls})")
    return tested - nn.calls + nn.neighbors_errors, int(len(n) - tested) + seeded.nan_returns


def cpu_info(path="/proc/cpuinfo"):
    """(model name, highest of avx512f / avx2 / sse4_2 in the CPU flags or 'none') of the first CPU."""
    model = flags = None
    with open(path) as f:
        for line in f:
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "model name" and model is None:
                model = value.strip()
            elif key == "flags" and flags is None:
                flags = set(value.split())
            if model is not None and flags is not None:
                break
    if model is None or flags is None:
        raise RuntimeError(f"{path} has no 'model name' or 'flags' line")
    return model, next((x for x in ("avx512f", "avx2", "sse4_2") if x in flags), "none")


def scorer_provenance():
    """The pinned provenance columns of every score row (code check CR-05)."""
    import importlib.metadata as md
    import rpy2.robjects as ro
    here = os.path.dirname(os.path.abspath(__file__))
    sha = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True, check=True).stdout.strip() != ""
    versions = {p: md.version(p) for p in VERSION_PACKAGES}
    versions["R_kBET"] = str(ro.r('as.character(utils::packageVersion("kBET"))')[0])
    model, simd = cpu_info()
    return dict(scorer_git_sha=sha, scorer_dirty=int(dirty), score_host=socket.gethostname(), cpu_model=model,
                cpu_simd=simd, numba_cpu_name=os.environ.get("NUMBA_CPU_NAME", ""),
                scorer_versions=json.dumps(versions, sort_keys=True))


def scib_overall(df):
    """scIB overall score: min-max scale every metric within each dataset across ALL rows in df,
    average within batch / bio (NaN metrics skipped, as in scIB), then 0.4*batch + 0.6*bio."""
    out = df.copy()
    for m in BATCH_METRICS + BIO_METRICS:
        if m not in out:
            continue
        g = out.groupby("dataset")[m]
        lo, hi = g.transform("min"), g.transform("max")
        out[m + "_scaled"] = (out[m] - lo) / (hi - lo).replace(0, np.nan)
    b = [m + "_scaled" for m in BATCH_METRICS if m + "_scaled" in out]
    o = [m + "_scaled" for m in BIO_METRICS if m + "_scaled" in out]
    out["batch_score"] = out[b].mean(axis=1, skipna=True)
    out["bio_score"] = out[o].mean(axis=1, skipna=True)
    out["overall"] = 0.4 * out["batch_score"] + 0.6 * out["bio_score"]
    return out


def main():
    t0 = time.time()
    pre = sc.read_h5ad(os.environ["PREPPED"])
    bk, ck = pre.uns.get("batch_key", "batch"), pre.uns.get("celltype_key", "celltype")
    d = np.load(os.environ["NPZ"], allow_pickle=True)
    if "obs_names" in d and len(d["obs_names"]) != pre.n_obs:
        # subsampled fit (X3, X8): the same cells of the unintegrated data, PCR reference by the full-data
        # convention (no precomputed PCA: scib recomputes the all-feature PCA of these cells; CR-01)
        if "pca" in pre.uns:
            raise ValueError(f"{os.environ['PREPPED']} carries uns['pca']: a subsample's PCR reference would be rows "
                             f"of that PCA, not the all-feature PCA of the scored cells (code check CR-01)")
        names = d["obs_names"].astype(str)
        idx = pd.Index(pre.obs_names).get_indexer(names)    # refuses a prepped file with repeated obs_names
        if (idx < 0).any():
            raise ValueError(f"{int((idx < 0).sum())} latent cells are not in {os.environ['PREPPED']}, "
                             f"e.g. {list(names[idx < 0][:3])} (code check CR-14)")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"the latent lists {len(idx) - len(np.unique(idx))} cells more than once (CR-14)")
        pre = pre[idx].copy()
    assert (d["batch"].astype(str) == pre.obs[bk].astype(str).values).all(), "cell order mismatch"
    integ = pre.copy()
    integ.obsm["X_emb"] = d["z"].astype(np.float32)
    sc.pp.neighbors(integ, use_rep="X_emb")          # scIB pipeline: kNN graph on the embedding
    kb = seed_kbet(kbet_seed())
    nn, root = count_kbet_neighbours(), watch_trajectory_root()
    prov = scorer_provenance()
    org = str(pre.uns["organism"])
    cell_cycle = pre.uns.get("modality") == "rna" and org in ("human", "mouse")
    trajectory = "dpt_pseudotime" in pre.obs and pre.obs["dpt_pseudotime"].notna().any()
    res = scib.metrics.metrics(
        pre, integ, batch_key=bk, label_key=ck, embed="X_emb", type_="embed",
        ari_=True, nmi_=True, silhouette_=True, pcr_=True, isolated_labels_f1_=True,
        isolated_labels_asw_=True, graph_conn_=True, kBET_=True, ilisi_=True, clisi_=True,
        cell_cycle_=cell_cycle, organism=(org if cell_cycle else "human"),
        hvg_score_=False, trajectory_=trajectory,
    )
    row = dict(tag=os.environ["TAG"], **json.loads(os.environ.get("META", "{}")))
    clash = sorted(set(row) & set(PROVENANCE_COLS + DIAGNOSTIC_COLS))
    if clash:
        raise ValueError(f"META sets scorer columns {clash}")
    row.update({k: float(v) for k, v in res.iloc[:, 0].items()})
    if np.isfinite(row["kBET"]) and kb.calls == 0:
        raise RuntimeError("kBET is finite but no seeded R kBET call ran: the seed wrapper was bypassed")
    row["kbet_seed"], row["kbet_r_calls"] = kb.seed, kb.calls
    if trajectory:
        if root.calls != 1:
            raise RuntimeError(f"trajectory conservation called get_root {root.calls} times, expected 1")
        row["traj_root_fallback"] = int(root.root_errors > 0)
    else:
        if root.calls:
            raise RuntimeError("get_root ran although trajectory conservation is off")
        row["traj_root_fallback"] = ""
    row["kbet_labels_forced_one"], row["kbet_labels_skipped"] = kbet_label_flags(integ.obs, ck, bk, kb, nn)
    row.update(prov)
    row["score_seconds"] = round(time.time() - t0, 1)
    tmp = os.environ["OUT_CSV"] + ".tmp"
    pd.DataFrame([row]).to_csv(tmp, index=False)
    os.replace(tmp, os.environ["OUT_CSV"])
    print(f"[scored] {row['tag']} in {row['score_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
