"""Run scripts/score_scib_native.py unchanged, with a timer around every scIB metric call.
Env as the scorer (PREPPED, NPZ, TAG, META, OUT_CSV, KBET_SEED) plus PROFILE_JSON. Values pass through unchanged.
Records wall (perf_counter) and CPU (process_time) seconds per call, so CPU/wall shows how many threads a metric used."""
import json, os, resource, sys, time
sys.path.insert(0, os.path.join(os.environ["REPO"], "scripts"))
import scanpy as sc, scib.metrics  # noqa: F401  (scib.metrics.metrics the attribute is the function, the module is in sys.modules)
sm = sys.modules["scib.metrics.metrics"]
import score_scib_native as S
T = {}
def timed(mod, name, label=None):
    f = getattr(mod, name)
    def w(*a, **k):
        lab = label or name
        if name == "isolated_labels": lab = "isolated_label_F1" if k.get("cluster") else "isolated_label_ASW"
        w0, c0 = time.perf_counter(), time.process_time()
        try: return f(*a, **k)
        finally:
            d = T.setdefault(lab, [0.0, 0.0, 0]); d[0] += time.perf_counter() - w0; d[1] += time.process_time() - c0; d[2] += 1
    setattr(mod, name, w)
for n in ["cluster_optimal_resolution", "nmi", "ari", "silhouette", "silhouette_batch", "pcr_comparison", "cell_cycle",
          "isolated_labels", "graph_connectivity", "kBET", "ilisi_graph", "clisi_graph", "trajectory_conservation"]:
    timed(sm, n)
timed(sc, "read_h5ad", "_read_prepped"); timed(sc.pp, "neighbors", "_neighbors")
t0, c0 = time.perf_counter(), time.process_time()
S.main()
T["_total"] = [time.perf_counter() - t0, time.process_time() - c0, 1]
T["_maxrss_gb"] = [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 0, 1]
with open(os.environ["PROFILE_JSON"], "w") as f:
    json.dump({k: dict(wall=round(v[0], 2), cpu=round(v[1], 2), calls=v[2]) for k, v in T.items()}, f)
