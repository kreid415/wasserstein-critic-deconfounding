"""SI-58 decision table: float32 vs float64 Sinkhorn solve against the converged float64 reference, and GPU cost.
Inputs (results/): validation_12mb_si58_float64.csv, validation_12mb_si58_float32.csv (validate_sinkhorn.py at the
committed code with SINKHORN_DTYPE set), validation_12mb.csv (committed float64 run, for the regression check),
validation_12mb_tol1e-6_cap1e4.csv (selection of the 598 hardest, as in tol_sweep.py), bench_gpu_si58.csv.
Criterion (SI-58): adopt float32 only if worst gradient cosine >= 0.99999 and worst relative value error <= 1e-4
(|reference value| > 1e-6, as in the float64 README) with no cap hit, on the 598; the full 3,076 is a declared superset
check. Usage: python si58_summary.py  (writes results/si58_summary.md)"""
import os
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KEY = ["kind", "tag", "minibatch", "zstd"]


def load(name):
    d = pd.read_csv(f"{R}/{name}")
    d["minibatch"], d["zstd"] = d.minibatch.astype(int), d.zstd.astype(int)
    return d.set_index(KEY).sort_index()


f64, f32, old = load("validation_12mb_si58_float64.csv"), load("validation_12mb_si58_float32.csv"), load("validation_12mb.csv")
assert f64.index.equals(f32.index) and f64.index.equals(old.index) and len(f64) == 3076
sel = load("validation_12mb_tol1e-6_cap1e4.csv")
it = sel[["new_iters_cross", "new_iters_self_a", "new_iters_self_b"]].max(axis=1)
hard = sel.index[(sel.index.get_level_values("zstd") == 1) | (it > 100) | sel.new_error.fillna("").ne("")]
assert len(hard) == 598

# regression check: the float64 re-run at the committed code equals the committed float64 run, column by column
shared = [c for c in old.columns if c in f64.columns]
diff = [c for c in shared if not (old[c].equals(f64[c]) or np.allclose(old[c].astype(float), f64[c].astype(float),
                                                                        rtol=0, atol=0, equal_nan=True))
        ]
lines = [f"## Regression check: float64 re-run vs committed validation_12mb.csv\n\n{len(shared)} shared columns, "
         f"{len(f64)} rows; columns that differ: {diff if diff else 'none (bit-identical)'}\n"]


def table(d, idx):
    d = d.loc[idx]
    big = d.value_ref.abs() > 1e-6
    rge = np.sqrt(np.maximum(d.new_norm_ratio ** 2 - 2 * d.new_norm_ratio * d.new_cosine + 1, 0))
    iters = d[["new_iters_cross", "new_iters_self_a", "new_iters_self_b"]].max(axis=1)
    return {"minibatches": len(d), "cap_hits": int(d.new_error.fillna("").ne("").sum()),
            "worst_cosine": d.new_cosine.min(), "1-worst_cosine": 1 - d.new_cosine.min(), "worst_rel_grad_err": rge.max(),
            "norm_ratio_min": d.new_norm_ratio.min(), "norm_ratio_max": d.new_norm_ratio.max(),
            "rows_ref_gt_1e-6": int(big.sum()), "worst_rel_value_err": d.new_value_rel_err[big].abs().max(),
            "worst_abs_value_err_ref_le_1e-6": (d.new_value_rel_err * d.value_ref)[~big].abs().max() if (~big).any() else np.nan,
            "iters_median": iters.median(), "iters_max": iters.max()}


rows = {(s, dt): table(d, idx) for s, idx in (("598 hardest", hard), ("all 3,076", f64.index))
        for dt, d in (("float64", f64), ("float32", f32))}
t = pd.DataFrame(rows).T
ok = {s: bool(t.loc[(s, "float32"), "worst_cosine"] >= 0.99999 and t.loc[(s, "float32"), "worst_rel_value_err"] <= 1e-4
               and t.loc[(s, "float32"), "cap_hits"] == 0) for s in ("598 hardest", "all 3,076")}
lines.append("## Accuracy against the converged float64 reference (SINKHORN_TOL 1e-4, cap 500,000)\n\n```\n"
             + t.to_string(float_format=lambda x: f"{x:.4g}") + "\n```\n\nSI-58 accuracy criterion met by float32: "
             + ", ".join(f"{s}: {'yes' if v else 'NO'}" for s, v in ok.items()) + "\n")
# where float32 is worst on the value (regimes)
w = f32.assign(regime=f32.stage.where(f32.index.get_level_values("kind") == "final", f32.index.get_level_values("kind")),
               abs_rel=f32.new_value_rel_err.abs(), f64_abs_rel=f64.new_value_rel_err.abs())
w = w[w.value_ref.abs() > 1e-6]
reg = w.groupby(["regime", w.index.get_level_values("zstd")]).agg(n=("abs_rel", "size"), f32_value_err_max=("abs_rel", "max"),
                                                                  f64_value_err_max=("f64_abs_rel", "max"))
lines.append("## Worst relative value error by regime (|reference| > 1e-6)\n\n```\n" + reg.to_string(float_format=lambda x: f"{x:.3g}") + "\n```\n")
b = pd.read_csv(f"{R}/bench_gpu_si58.csv")
b["f32_over_f64"] = b.new32_ms / b.new_ms
bg = b.groupby("regime").agg(tag_ms=("tag_ms", "median"), float64_ms=("new_ms", "median"), float32_ms=("new32_ms", "median"),
                             f32_over_f64_median=("f32_over_f64", "median"), f32_over_f64_min=("f32_over_f64", "min"),
                             gpu_cos_32_64_min=("gpu_cos_32_64", "min"),
                             gpu_value_rel_32_64_absmax=("gpu_value_rel_32_64", lambda s: s.abs().max()))
lines.append("## GPU ms per loss forward + backward, median of 20 calls x 12 minibatches (RTX 3080, alone)\n\n```\n"
             + bg.to_string(float_format=lambda x: f"{x:.4g}") + "\n```\n")
open(f"{R}/si58_summary.md", "w").write("\n".join(lines))
print("\n".join(lines))
