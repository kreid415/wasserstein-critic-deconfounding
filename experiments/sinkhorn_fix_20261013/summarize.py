"""Tables of results/summary.md from validation_12mb.csv, tol_sweep.csv and bench_gpu.csv. Usage: python summarize.py"""
import os

import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def md(df):
    return "```\n" + df.to_string(float_format=lambda x: f"{x:.4g}") + "\n```\n"


v = pd.read_csv(f"{R}/validation_12mb.csv")
v["regime"] = v.kind.where(v.kind != "final", v.stage) + " zstd=" + v.zstd.astype(str)
v["iters"] = v[["new_iters_cross", "new_iters_self_a", "new_iters_self_b"]].max(axis=1)
v["new_rel_err_value_abs"] = v.new_value_rel_err.abs()
fails = int(v.new_error.fillna("").ne("").sum())
val = v.groupby("regime").agg(minibatches=("tag", "size"), branch_cos_min=("new_cosine", "min"),
                              branch_norm_ratio_min=("new_norm_ratio", "min"), branch_norm_ratio_max=("new_norm_ratio", "max"),
                              branch_value_err_median=("new_rel_err_value_abs", "median"),
                              iters_median=("iters", "median"), iters_max=("iters", "max"),
                              tag_cos_median=("tag_cosine", "median"), tag_norm_ratio_median=("tag_norm_ratio", "median"))
t = pd.read_csv(f"{R}/tol_sweep.csv")
t["iters"] = t[["iters_cross", "iters_self"]].max(axis=1)
sweep = t.groupby(["zstd", "tol"]).agg(minibatches=("tag", "size"), rel_grad_err_max=("rel_grad_err", "max"),
                                       iters_median=("iters", "median"), iters_p99=("iters", lambda s: s.quantile(0.99)),
                                       iters_max=("iters", "max"))
b = pd.read_csv(f"{R}/bench_gpu.csv").groupby("regime")[["tag_ms", "new_ms"]].median()
with open(f"{R}/summary.md", "w") as f:
    f.write(f"## Validation at the committed constants ({len(v)} minibatches; cap reached: {fails})\n\n" + md(val))
    f.write("\n## Tolerance sweep (cap 200,000)\n\n" + md(sweep))
    f.write("\n## GPU ms per loss forward + backward, median (RTX 3080, one process)\n\n" + md(b))
print(open(f"{R}/summary.md").read())
