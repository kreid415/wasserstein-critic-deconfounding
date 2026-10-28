"""Rebuild every table and figure of speed_diagnosis_20261028 from the raw inputs listed in README.md.
usage: python analysis.py <input_dir> <output_dir>
input_dir holds: logs/run_stage_<slurm id>.log (5 pilot jobs), sacct_pilot_and_test_jobs.psv, ws_out/x/<run>/*.json|csv (workstation scorer runs),
jhpce_scorer/out/n*_p*.json|csv, check_results/run1|run2/scores/*.csv, prod_scores/*.csv (local A1 production rows of the 20 check latents)."""
import datetime as dt, glob, json, os, re, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, pandas as pd
IN, OUT = sys.argv[1] + "/", sys.argv[2] + "/"; os.makedirs(OUT, exist_ok=True)
JOBS = {"36551148": "immune_hum_mou", "36551149": "atac_large", "36551150": "lung", "36551151": "sim2", "36551152": "pancreas"}
START = {"36551149": dt.datetime(2026, 10, 22, 9, 10, 18)}          # first line of slurm-<id>.out (log lines carry no date)
M = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label", "ASW_label/batch", "PCR_batch", "cell_cycle_conservation", "isolated_label_F1",
     "isolated_label_silhouette", "graph_conn", "kBET", "iLISI", "cLISI", "trajectory", "kbet_r_calls", "kbet_labels_forced_one", "kbet_labels_skipped"]

# ---- 1. per-fit timing from run_stage.log (day rollover handled) ----
rows = []
for j, ds in JOBS.items():
    prev, day, st = None, 0, {}
    for ln in open(f"{IN}logs/run_stage_{j}.log"):
        m = re.match(r"\[(\d\d):(\d\d):(\d\d)\] (.*)", ln)
        if not m: continue
        t = dt.timedelta(hours=int(m[1]), minutes=int(m[2]), seconds=int(m[3]))
        if prev is not None and t < prev - dt.timedelta(hours=1): day += 1
        prev = t; ts = dt.datetime(2026, 10, 22) + dt.timedelta(days=day) + t
        s = re.match(r"\[fit\] start (\S+) \(attempt", m[4])
        if s: st[s[1]] = ts
        d = re.match(r"\[done\] (\S+): fitted fit ([\d.]+) s", m[4])
        if d: rows.append(dict(job=j, ds=ds, tag=d[1], arm=d[1][len("A1_" + ds + "_"):].split("_")[0], start=st.get(d[1]), done=ts, fit_s=float(d[2])))
df = pd.DataFrame(rows); df["wall_s"] = (df.done - df.start).dt.total_seconds()
# the child writes its latent ~4.5 s after the timed fit ends (npz mtime vs done line, one pooled fit and one repeat), so the timed fit began at done - fit_s - 4.5 s
df["train_start"] = df.done - pd.to_timedelta(df.fit_s + 4.5, unit="s"); df["wait_before_fit_s"] = (df.train_start - df.start).dt.total_seconds()
df.to_csv(OUT + "pilot_fit_timing_all_jobs.csv", index=False)
a = df[df.ds == "atac_large"].copy()
summ = df.groupby("ds").agg(n_fits=("tag", "count"), wait_median_s=("wait_before_fit_s", "median"), wait_max_s=("wait_before_fit_s", "max"), wall_over_fit=("wall_s", "sum"))
summ["wall_over_fit"] = df.groupby("ds").wall_s.sum() / df.groupby("ds").fit_s.sum(); summ.round(3).to_csv(OUT + "wait_summary_by_dataset.csv")

# ---- 2. job utilisation from sacct ----
def hms(s):
    d = 0
    if "-" in s: d, s = s.split("-"); d = int(d)
    p = s.split(":"); h, mi, sec = (int(p[0]), int(p[1]), float(p[2])) if len(p) == 3 else (0, int(p[0]), float(p[1]))
    return d * 86400 + h * 3600 + mi * 60 + sec
sa = pd.read_csv(IN + "sacct_pilot_and_test_jobs.psv", sep="|", nrows=15)
ut = []
for j, ds in JOBS.items():
    top, bat = sa[sa.JobID == j].iloc[0], sa[sa.JobID == j + ".batch"].iloc[0]
    el, cpu, rss = hms(top.Elapsed), hms(top.TotalCPU), float(str(bat.MaxRSS).rstrip("K")) / 1048576
    d = df[df.job == j]; span = (d.done.max() - d.start.min()).total_seconds()
    ut.append(dict(job=j, dataset=ds, fits_done=len(d), elapsed_h=round(el / 3600, 1), lane_use_logged_fit=round(d.fit_s.sum() / (8 * span), 3), lane_use_wall=round(d.wall_s.sum() / (8 * span), 3),
                   cpu_busy=round(cpu / (10 * el), 3), cpu_s_per_logged_fit_s=round(cpu / d.fit_s.sum(), 3), sacct_maxrss_gb=round(rss, 1), maxrss_over_96G=round(rss / 96, 2)))
pd.DataFrame(ut).to_csv(OUT + "pilot_job_utilisation.csv", index=False)

# ---- 3. atac_large throughput vs number of lanes in a timed fit ----
ev = sorted([(r.train_start, 1) for r in a.itertuples()] + [(r.done - pd.Timedelta(seconds=4.5), -1) for r in a.itertuples()])
times = [e[0] for e in ev]; lv = np.cumsum([e[1] for e in ev])
import bisect
def avg_conc(t0, t1):
    i = bisect.bisect_right(times, t0) - 1; level = lv[i] if i >= 0 else 0; t = t0; tot = 0.0
    for j in range(i + 1, len(times)):
        if times[j] >= t1: break
        tot += level * (times[j] - t).total_seconds(); t = times[j]; level = lv[j]
    return (tot + level * (t1 - t).total_seconds()) / (t1 - t0).total_seconds()
a["k"] = [avg_conc(r.train_start, r.done - pd.Timedelta(seconds=4.5)) for r in a.itertuples()]
a["kb"] = pd.cut(a.k, [0, 2, 3, 4, 6, 8.1], labels=["<=2", "2-3", "3-4", "4-6", "6-8"])
t = a.groupby(["arm", "kb"], observed=True).agg(n=("fit_s", "count"), k=("k", "median"), fit_s=("fit_s", "median")).reset_index()
t["fits_per_ks"] = t.k / t.fit_s * 1000; t["rel_to_k2_3"] = t.fits_per_ks / t.arm.map(t[t.kb == "2-3"].set_index("arm").fits_per_ks)
t.round(3).to_csv(OUT + "atac_large_throughput_vs_lanes.csv", index=False)

# ---- 4. figure: atac_large wait ----
t0 = START["36551149"]; fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
ax[0].scatter((a.start - t0).dt.total_seconds() / 3600, a.wait_before_fit_s / 3600, s=9, color="#1f4e79"); ax[0].set_ylabel("wait before the timed fit (h)")
ax[0].set_title("Every fit started after hour 8.3 waits 1.7-3.2 h before its timed fit begins", loc="left")
grid = pd.date_range(t0, a.done.max(), freq="10min"); x = (grid - t0).total_seconds() / 3600
ax[1].plot(x, [((a.train_start <= g) & (a.done > g)).sum() for g in grid], color="#1f4e79", lw=1, label="in timed fit")
ax[1].plot(x, [((a.start <= g) & (a.train_start > g)).sum() for g in grid], color="#c0504d", lw=1, label="waiting before timed fit")
ax[1].set_ylabel("lanes (of 8)"); ax[1].set_xlabel("hours since job start (atac_large, Slurm 36551149)"); ax[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(OUT + "fig_atac_large_wait.png", dpi=200)

# ---- 5. scorer profile tables ----
def prof(d): return {re.sub(r"^p\d+_", "", os.path.basename(f))[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{IN}ws_out/x/{d}/*.json"))}
def csvs(d): return {re.sub(r"^p\d+_", "", os.path.basename(f))[:-4]: pd.read_csv(f).iloc[0] for f in sorted(glob.glob(f"{IN}ws_out/x/{d}/*.csv"))}
def same(x, y): return [m for m in M if m in x and m in y and not ((x[m] == y[m]) or (pd.isna(x[m]) and pd.isna(y[m])))]
quiet = {n: list(prof(f"quiet_n{n}").values()) for n in (1, 3, 6, 8)}; wall = {1: 178.1, 3: 249.6, 6: 384.2, 8: 506.2}   # driver wall (s) printed by run_cfg.py
q = pd.DataFrame([dict(n=n, driver_wall_s=wall[n], per_proc_wall_s=np.mean([j["_total"]["wall"] for j in js]), per_proc_cpu_s=np.mean([j["_total"]["cpu"] for j in js]),
                       scores_per_h=n / wall[n] * 3600, peak_rss_gb=np.mean([j["_maxrss_gb"]["wall"] for j in js])) for n, js in quiet.items()])
q["speed_vs_single"] = q.per_proc_wall_s[0] / q.per_proc_wall_s; q["throughput_rel_n1"] = q.scores_per_h / q.scores_per_h[0]; q.round(2).to_csv(OUT + "workstation_concurrency_pancreas_quiet.csv", index=False)
pm = pd.DataFrame({n: {k: np.mean([j[k]["wall"] for j in js]) for k in js[0] if not k.startswith("_max")} for n, js in quiet.items()}); pm["x6_over_x1"] = pm[6] / pm[1]; pm.round(2).to_csv(OUT + "workstation_per_metric_pancreas_quiet.csv")
ref = list(csvs("quiet_n1").values())[0]
ident = all(not same(ref, r) for n in (3, 6, 8) for r in csvs(f"quiet_n{n}").values())
big = list(prof("large_single_t1").values())[0]; n3 = prof("large_n3_t1")
rows = []
for s in ("t1", "numba2", "blas2", "all2"):
    for tag, j in prof("exact_t1_rerun" if s == "t1" else "exact_" + s).items():
        rows.append(dict(setting=s, latent=tag[:40], total_s=j["_total"]["wall"], cpu_s=j["_total"]["cpu"]))
pd.DataFrame(rows).round(1).to_csv(OUT + "workstation_thread_settings.csv", index=False)
exact = {s: csvs("exact_t1_rerun" if s == "t1" else "exact_" + s) for s in ("t1", "numba2", "blas2", "all2")}
base = csvs("exact_t1")
verdict = {s: {t[:40]: same(base[t], r) for t, r in d.items()} for s, d in exact.items()}
json.dump(dict(quiet_copies_identical_to_n1=bool(ident), exact_vs_first_baseline_columns_that_differ=verdict,
               large_single_total_s=big["_total"]["wall"], large_n3_total_s={k[:30]: v["_total"]["wall"] for k, v in n3.items()}, large_n3_rss_gb={k[:30]: v["_maxrss_gb"]["wall"] for k, v in n3.items()}),
          open(OUT + "workstation_verdicts.json", "w"), indent=1)

# ---- 6. JHPCE scorer test and JHPCE-vs-workstation match ----
jr = pd.DataFrame([dict(n=int(re.search(r"n(\d+)_p", f)[1]), total_s=json.load(open(f))["_total"]["wall"]) for f in glob.glob(IN + "jhpce_scorer/out/n*_p*.json")]).groupby("n").mean()
jr.round(1).to_csv(OUT + "jhpce_scorer_concurrency.csv")
def rd(dp): return {os.path.basename(f)[:-4]: pd.read_csv(f).iloc[0] for f in glob.glob(dp + "/*.csv")}
r1, r2, pr = rd(IN + "check_results/run1/scores"), rd(IN + "check_results/run2/scores"), rd(IN + "prod_scores")
rows = []
for tg in sorted(pr):
    row = dict(tag=tg, same_latent_hash=r1[tg]["latent_sha256"] == pr[tg]["latent_sha256"])
    for nm, x, y in (("jhpce_run1_vs_run2", r1[tg], r2[tg]), ("jhpce_run1_vs_workstation", r1[tg], pr[tg])):
        bad = same(x, y); row[nm + "_columns_differing"] = ",".join(bad) or "none"; row[nm + "_max_abs_diff"] = max([abs(x[m] - y[m]) for m in bad], default=0.0)
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT + "jhpce_vs_workstation_score_match_20latents.csv", index=False)
print("done", OUT)
