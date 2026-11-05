#!/usr/bin/env python3
"""Pass 9: fill the result placeholders of paper/outline.tex from the result files and write the number ledger.

usage (from the repo root):
  python paper/tools/pass9_fill.py [--sweep DIR] [--placement CSV] [--bio DIR] [--replication JSON]
                                   [--tex paper/outline.tex] [--ledger paper/pass9_numbers.csv]

Every number written into the .tex is read from a source file (paper_numbers.json keys, sweep tables, the independent
placement check, the biology-check tables, the replication result) and rendered from a template placeholder {{name:fmt}};
the ledger gets one row per rendered number with its source file, key or row, and raw value. Statements in the text that
depend on the data (who is lowest, which task, equal counts) are asserted here, so the script stops if the data change.
Fails loud: an edit whose old text does not occur exactly once, a placeholder without a value, a source selector that
does not match exactly one row. Without --replication the replication edits are skipped and listed.
"""
import argparse, csv, json, os, re, sys
import pandas as pd

D = "/home/kendall/experiment_data/wasserstein-critic-deconfounding"
REAL = ["pancreas", "lung", "immune", "immune_hum_mou", "atac_small", "atac_large"]
PEN = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
SHORT = {"discriminator": "discriminator", "reference": "reference critic", "pooled": "pooled critic",
         "barycenter": "barycenter critic", "mmd": "MMD", "sinkhorn": "Sinkhorn divergence"}
TOOLS = ("none", "harmony", "scanorama", "sysvi", "scanvi", "scvi_adv", "pca")
BASE = ("scvi_adv", "scanvi", "sysvi", "harmony", "scanorama", "pca")


class Vals:
    """Named values, each with its source; render() substitutes {{name:fmt}} and records one ledger row per number."""

    def __init__(self):
        self.v, self.rows = {}, []

    def put(self, name, value, file, key):
        if name in self.v:
            raise KeyError(f"value {name} registered twice")
        if value is None or (isinstance(value, float) and value != value):
            raise ValueError(f"{name}: missing value from {file} {key}")
        self.v[name] = dict(value=value, file=file, key=key)
        return value

    def __getitem__(self, name):
        return self.v[name]["value"]

    def render(self, template, snippet):
        def sub(m):
            name, fmt = m.group(1), m.group(2) or "s"
            if name not in self.v:
                raise KeyError(f"{snippet}: no value named {name}")
            e = self.v[name]
            txt = fmtv(e["value"], fmt)
            self.rows.append(dict(snippet=snippet, number_as_written=txt.replace("$", "").replace("\\%", "%"),
                                  source_file=e["file"], source_key_or_row=e["key"], source_value=repr(e["value"])))
            return txt
        out = re.sub(r"\{\{([A-Za-z0-9_.\-]+)(?::([a-z0-9]+))?\}\}", sub, template)
        assert "{{" not in out, f"{snippet}: unresolved placeholder in {out[out.index('{{'):][:60]}"
        return out


def fmtv(x, fmt):
    if fmt == "s":
        return str(x)
    if fmt == "d":
        if float(x) != int(x):
            raise ValueError(f"not an integer: {x}")
        return f"{int(x):,}"
    if fmt in ("f2", "f3"):
        return f"{x:.{fmt[1]}f}"
    if fmt == "sf3":          # signed, math minus; no sign on a value that rounds to zero
        return "$0.000$" if round(x, 3) == 0 else f"${x:+.3f}$"
    if fmt == "g":            # ranks: 2, 2.5, 3.33
        return f"{x:.0f}" if float(x) == int(x) else f"{x:.2f}".rstrip("0")
    if fmt == "lam":          # penalty strength as in the text: 0.1, 3, 1,000
        return f"{x:,.0f}" if x >= 1 else f"{x:g}"
    raise ValueError(f"unknown format {fmt}")


def load(a, V):
    pnf = os.path.join(a.sweep, "paper_numbers.json")
    PN = json.load(open(pnf))

    def pn(name, key):
        return V.put(name, PN[key]["value"], pnf, key)

    def one(df, file, name, col, **where):
        q = df
        for k, v in where.items():
            q = q[q[k] == v]
        assert len(q) == 1, f"{name}: {len(q)} rows in {file} for {where}"
        x = q[col].iloc[0]
        return V.put(name, x.item() if hasattr(x, "item") else x, file,
                     ",".join(f"{k}={v}" for k, v in where.items()) + f" col={col}")

    def derived(name, value, file, how):
        return V.put(name, value, file, how)

    # --- low-signal pairs (Methods, Supp Note 5)
    for t, tag in (("immune", "imm"), ("sim1", "s1"), ("sim2", "s2")):
        b0 = pn(f"ls_{tag}_b0", f"pair.{t}.cond1.b0")
        bc = pn(f"ls_{tag}_bc", f"pair.{t}.cond1.b_common")
        derived(f"ls_{tag}_gap", bc - b0, pnf, f"pair.{t}.cond1.b_common - pair.{t}.cond1.b0")
        assert PN[f"pair.{t}.cond1.low_signal"]["value"] is True
    assert PN["low_signal_tasks.cond0"]["value"] == [] and PN["low_signal_tasks.cond1"]["value"] == ["immune", "sim1", "sim2"]
    derived("n_noreading", sum(PN[f"n_pairs_no_reading.cond{c}.{p}"]["value"] for c in (0, 1) for p in PEN), pnf,
            "sum of n_pairs_no_reading.cond{0,1}.{six penalties}")
    for p in PEN:
        for c in (0, 1):
            pn(f"mr_{p}_{c}", f"mean_rank.cond{c}.{p}")
            pn(f"lt_{p}_{c}", f"label_transfer_mean.cond{c}.{p}")
    assert min(PEN, key=lambda p: V[f"mr_{p}_1"]) == "pooled" and min(PEN, key=lambda p: V[f"mr_{p}_0"]) == "mmd"
    assert max(PEN, key=lambda p: V[f"mr_{p}_1"]) == "discriminator"
    assert all(min(PEN, key=lambda p: V[f"lt_{p}_{c}"]) == "discriminator" for c in (0, 1))

    # --- near-ties over the 13 ranked pairs (sweep table)
    rpf = os.path.join(a.sweep, "tables/ranking_pairs.csv")
    rp = pd.read_csv(rpf)
    d = []
    for _, g in rp.groupby(["task", "cond"]):
        b = dict(zip(g.arm, g.bio_at_bstar))
        d += [abs(b[x] - b[y]) for i, x in enumerate(PEN) for y in PEN[i + 1:]]
    assert rp.groupby(["task", "cond"]).ngroups == 13 and len(d) == 195
    derived("n_within", len(d), rpf, "pairs of penalties within the 13 ranked task-decoder pairs")
    derived("n_ties", sum(x < 0.02 for x in d), rpf, "within-pair |bio_at_bstar difference| < 0.02")
    derived("n_large", sum(x >= 0.04 for x in d), rpf, "within-pair |bio_at_bstar difference| >= 0.04")

    # --- baselines (Res-1)
    for m in BASE:
        for f in ("expected", "scored", "failed", "missing"):
            derived(f"bl_{m}_{f}", sum(PN[f"baseline_coverage.{t}.{m}.{f}"]["value"] for t in REAL + ["sim1", "sim2"]), pnf,
                    f"sum over 8 tasks of baseline_coverage.<task>.{m}.{f}")
    for f in ("expected", "scored", "failed", "missing"):
        derived(f"bl_{f}", sum(V[f"bl_{m}_{f}"] for m in BASE), pnf, f"sum over methods and tasks of baseline_coverage.*.*.{f}")
    assert V["bl_expected"] == V["bl_scored"] + V["bl_failed"] + V["bl_missing"] and V["bl_missing"] == 1
    assert V["bl_scanorama_missing"] == 1 and PN["baseline_coverage.atac_large.scanorama.missing"]["value"] == 1

    # --- placement at equal tries (independent check, placement_summary.csv) and best-of-k means
    ps = pd.read_csv(a.placement)
    for c in (0, 1):
        for p in PEN:
            for tool in TOOLS:
                for k in (1, 6):
                    one(ps, a.placement, f"pl_{c}_{p}_{tool}_{k}", "n_real_tasks_better", cond=c, penalty=p, tool=tool, k=k)
                    one(ps, a.placement, f"pln_{c}_{p}_{tool}_{k}", "n_real_tasks", cond=c, penalty=p, tool=tool, k=k)
        for tool in TOOLS:
            for k in (1, 6):
                xs = [V[f"pl_{c}_{p}_{tool}_{k}"] for p in PEN]
                derived(f"plmin_{c}_{tool}_{k}", min(xs), a.placement, f"min over six penalties of n_real_tasks_better (cond={c}, tool={tool}, k={k})")
                derived(f"plmax_{c}_{tool}_{k}", max(xs), a.placement, f"max over six penalties of n_real_tasks_better (cond={c}, tool={tool}, k={k})")
                derived(f"plsum_{c}_{tool}_{k}", sum(xs), a.placement, f"sum over six penalties of n_real_tasks_better (cond={c}, tool={tool}, k={k})")
    derived("pl_cells", 36, a.placement, "six penalties x six real tasks (n_real_tasks = 6 for tool=none)")
    assert all(V[f"pln_1_{p}_none_{k}"] == 6 for p in PEN for k in (1, 6))
    # statements written as exact counts for every penalty
    for tool in ("harmony", "sysvi", "scanorama"):
        assert V[f"plmin_1_{tool}_6"] == V[f"plmax_1_{tool}_6"], tool
    assert {V[f"pln_1_{p}_scanorama_6"] for p in PEN} == {5} and V["plmin_1_scanorama_6"] == 5   # 'all 5 tasks'
    derived("n_combos", 6 * sum(1 for k in PN if k.startswith("pair.") and k.endswith(".bstar")), pnf, "six penalties x task-decoder pairs (pair.*.bstar keys)")
    assert V["plmax_0_scanvi_1"] == 0 and V["plmax_1_scanvi_1"] == 0 and V["plmax_0_scanvi_6"] == 0
    pef = os.path.join(os.path.dirname(a.placement), "placement_equal_tries.csv")
    pe = pd.read_csv(pef)
    sc = pe[pe.task.isin(REAL) & (pe.tool == "scanvi") & (pe.k == 6) & pe.penalty_better]
    assert set(sc.task) == {"pancreas"} and set(sc.cond) == {1} and len(sc) == 6, \
        "scANVI placement changed: the text says only pancreas, batch-conditioned decoder, all six penalties"
    derived("scanvi_dmin", sc["diff"].min(), pef, "min diff, tool=scanvi, k=6, penalty_better, real tasks (all pancreas, cond 1)")
    derived("scanvi_dmax", sc["diff"].max(), pef, "max diff, tool=scanvi, k=6, penalty_better, real tasks (all pancreas, cond 1)")
    for c in (0, 1):
        pn(f"bk_none_{c}", f"best_of_k_real_mean.cond{c}.none.k1")
        pn(f"bk_scanvi_{c}", f"best_of_k_real_mean.cond{c}.scanvi.k1")
        pn(f"bk_adv_{c}", f"best_of_k_real_mean.cond{c}.scvi_adv.k1")
        for p in PEN:
            for k in (1, 6):
                pn(f"bk_{p}_{c}_{k}", f"best_of_k_real_mean.cond{c}.{p}.k{k}")
        for k in (1, 6):
            xs = [V[f"bk_{p}_{c}_{k}"] for p in PEN]
            derived(f"bkmin_{c}_{k}", min(xs), pnf, f"min over six penalties of best_of_k_real_mean.cond{c}.<penalty>.k{k}")
            derived(f"bkmax_{c}_{k}", max(xs), pnf, f"max over six penalties of best_of_k_real_mean.cond{c}.<penalty>.k{k}")
        gains = [V[f"bk_{p}_{c}_6"] - V[f"bk_{p}_{c}_1"] for p in PEN]
        derived(f"bkgain_min_{c}", min(gains), pnf, f"min over penalties of best_of_k_real_mean.cond{c}.<p>.k6 - .k1")
        derived(f"bkgain_max_{c}", max(gains), pnf, f"max over penalties of best_of_k_real_mean.cond{c}.<p>.k6 - .k1")
        for p in PEN:  # smallest number of tries at which a penalty's real-task mean exceeds plain scVI's
            ks = sorted(int(k.rsplit(".k", 1)[1]) for k in PN if k.startswith(f"best_of_k_real_mean.cond{c}.{p}.k"))
            first = next(k for k in ks if PN[f"best_of_k_real_mean.cond{c}.{p}.k{k}"]["value"] > V[f"bk_none_{c}"])
            derived(f"kfirst_{p}_{c}", first, pnf, f"smallest k with best_of_k_real_mean.cond{c}.{p}.k > best_of_k_real_mean.cond{c}.none.k1")
    assert min(PEN, key=lambda p: V[f"kfirst_{p}_1"]) == "mmd" and max(PEN, key=lambda p: V[f"kfirst_{p}_1"]) == "discriminator"
    assert V["bkgain_min_1"] > V["bkmax_1_6"] - V["bkmin_1_6"]   # 'strength matters more than penalty choice'

    # --- single strength from other families (Res-7, Table 4)
    tsf = os.path.join(a.sweep, "tables/transfer_strength.csv")
    tr = pd.read_csv(tsf)
    tr = tr[tr.task.isin(REAL)]
    for c in (0, 1):
        for fam in PEN + ["harmony", "scanorama", "sysvi"]:
            g = tr[(tr.cond == c) & (tr.family == fam)]
            assert len(g) == 6, (c, fam)
            derived(f"tpos_{fam}_{c}", int((g.S_minus_plain_scvi > 0).sum()), tsf, f"real tasks with S_minus_plain_scvi > 0 (cond={c}, family={fam})")
            derived(f"bpos_{fam}_{c}", int((g.best_S_minus_plain_scvi > 0).sum()), tsf, f"real tasks with best_S_minus_plain_scvi > 0 (cond={c}, family={fam})")
        for which in ("tpos", "bpos"):
            xs = [V[f"{which}_{p}_{c}"] for p in PEN]
            derived(f"{which}min_{c}", min(xs), tsf, f"min over six penalties of {which} (cond={c})")
            derived(f"{which}max_{c}", max(xs), tsf, f"max over six penalties of {which} (cond={c})")
    g = tr[(tr.cond == 1) & tr.family.isin(PEN)]
    derived("tdiff_min", g.S_minus_plain_scvi.min(), tsf, "min S_minus_plain_scvi, cond=1, six penalties, real tasks")
    derived("tdiff_med", g.S_minus_plain_scvi.median(), tsf, "median S_minus_plain_scvi, cond=1, six penalties, real tasks")
    derived("tdiff_max", g.S_minus_plain_scvi.max(), tsf, "max S_minus_plain_scvi, cond=1, six penalties, real tasks")
    for p in PEN:
        lams = sorted(set(PN[f"transfer_strength.cond1.{p}.{t}"]["value"] for t in REAL))
        derived(f"tlam_lo_{p}", lams[0], pnf, f"min over real tasks of transfer_strength.cond1.{p}.<task>")
        derived(f"tlam_hi_{p}", lams[-1], pnf, f"max over real tasks of transfer_strength.cond1.{p}.<task>")
    allx = [PN[f"transfer_strength.cond1.{p}.{t}"]["value"] for p in PEN for t in REAL]
    derived("tlam_all_lo", min(allx), pnf, "min over penalties and real tasks of transfer_strength.cond1.*")
    derived("tlam_all_hi", max(allx), pnf, "max over penalties and real tasks of transfer_strength.cond1.*")
    for t in REAL:  # Table 4 cells, conditioned decoder
        for nm, key in (("plain", "none|0|10"), ("adv", "scvi_adv|0|10"), ("scanvi", "scanvi|0|10"), ("pca", "pca|0|10")):
            pn(f"t4_{nm}_{t}", f"single.{key}.{t}.cond1.S")
        for fam in PEN + ["harmony", "scanorama", "sysvi"]:
            one(tr, tsf, f"t4_{fam}_{t}", "S", cond=1, family=fam, task=t)
            one(tr, tsf, f"t4b_{fam}_{t}", "best_S", cond=1, family=fam, task=t)

    # --- Table 3
    for c in (0, 1):
        ranked = [t for t in REAL if not PN[f"pair.{t}.cond{c}.low_signal"]["value"]]
        assert ranked == (REAL if c == 0 else [t for t in REAL if t != "immune"])
        derived(f"nreal_{c}", 6, pnf, "six real tasks (_meta.tasks minus sim1, sim2)")
        for p in PEN:
            rk = [PN[f"rank.{p}.{t}.cond{c}"]["value"] for t in ranked]
            bio = [PN[f"bio_at_bstar.{p}.{t}.cond{c}"]["value"] for t in ranked]
            assert len(rk) == PN[f"mean_rank_n_tasks.cond{c}.{p}"]["value"]
            derived(f"t3lo_{p}_{c}", min(rk), pnf, f"min over ranked real tasks of rank.{p}.<task>.cond{c}")
            derived(f"t3hi_{p}_{c}", max(rk), pnf, f"max over ranked real tasks of rank.{p}.<task>.cond{c}")
            derived(f"t3bio_{p}_{c}", sum(bio) / len(bio), pnf, f"mean over ranked real tasks of bio_at_bstar.{p}.<task>.cond{c}")
            pn(f"t3n_{p}_{c}", f"mean_rank_n_tasks.cond{c}.{p}")
            if c == 0:
                pn(f"t3s1_{p}", f"sim_rank.sim1.cond0.{p}")
                pn(f"t3s2_{p}", f"sim_rank.sim2.cond0.{p}")

    # --- biology check (summary_by_method.csv, summary.csv, per-task tables)
    T = os.path.join(a.bio, "tables")
    bmf, bsf = os.path.join(T, "summary_by_method.csv"), os.path.join(T, "summary.csv")
    bym, summ = pd.read_csv(bmf), pd.read_csv(bsf)
    rb = bym[bym.task_group == "real"]
    dec = {0: "unconditioned", 1: "conditioned"}
    COLS = ("purity_diff", "rare_purity_diff", "rare_lt_recall_diff", "marker_jaccard", "marker_jaccard_diff")
    for p in PEN + ["scvi"]:
        for c in (0, 1):
            for col in COLS:
                one(bym, bmf, f"bio_{col}_{p}_{c}", col, task_group="real", method=p, decoder=dec[c])
            s = summ[(summ.method == p) & (summ.decoder == dec[c]) & summ.task.isin(REAL)]
            assert len(s) == 6
            derived(f"bio_sd_{p}_{c}", s.purity_diff_sd.mean(), bsf, f"mean over 6 real tasks of purity_diff_sd (method={p}, decoder={dec[c]})")
    for m in BASE:
        r = rb[rb.method == m]
        for col in COLS:
            derived(f"bio_{col}_{m}", r[col].mean(), bmf, f"mean over the {len(r)} settings of {col} (task_group=real, method={m})")
        derived(f"bio_nset_{m}", len(r), bmf, f"settings of {m} (task_group=real)")
        s = summ[(summ.method == m) & summ.task.isin(REAL)]
        derived(f"bio_sd_{m}", s.purity_diff_sd.mean() if s.purity_diff_sd.notna().any() else "n/a", bsf,
                f"mean over real tasks and settings of purity_diff_sd ({m}; n/a for one-run tools)")
    for c in (0, 1):
        assert min(PEN, key=lambda p: V[f"bio_purity_diff_{p}_{c}"]) == "discriminator"
        assert min(PEN, key=lambda p: V[f"bio_rare_purity_diff_{p}_{c}"]) == "discriminator"
        assert sorted(PEN, key=lambda p: V[f"bio_purity_diff_{p}_{c}"])[1] == "sinkhorn"
    assert V["bio_rare_purity_diff_discriminator_1"] < V["bio_purity_diff_discriminator_1"]
    others = [V[f"bio_purity_diff_{p}_{c}"] for p in ("reference", "pooled", "barycenter", "mmd") for c in (0, 1)]
    derived("bio_oth_lo", min(others), bmf, "min purity_diff (real) over reference, pooled, barycenter, MMD x both decoders")
    derived("bio_oth_hi", max(others), bmf, "max purity_diff (real) over reference, pooled, barycenter, MMD x both decoders")
    mj = [V[f"bio_marker_jaccard_{p}_{c}"] for p in PEN + ["scvi"] for c in (0, 1)]
    derived("bio_mj_lo", min(mj), bmf, "min marker_jaccard (real = 4 RNA tasks) over six penalties and plain scVI, both decoders")
    derived("bio_mj_hi", max(mj), bmf, "max marker_jaccard (real = 4 RNA tasks) over six penalties and plain scVI, both decoders")
    derived("bio_mjd_abs", max(abs(V[f"bio_marker_jaccard_diff_{p}_{c}"]) for p in PEN for c in (0, 1)), bmf,
            "max |marker_jaccard_diff| over six penalties, both decoders (real)")
    assert V["bio_marker_jaccard_diff_harmony"] < -V["bio_mjd_abs"] and V["bio_marker_jaccard_diff_pca"] < -V["bio_mjd_abs"]
    rp_ = summ[summ.method.isin(PEN) & summ.task.isin(REAL) & summ.rare_purity_diff_mean.notna()]
    assert sorted(set(rp_.task)) == sorted(["pancreas", "lung", "immune", "immune_hum_mou"])
    derived("bio_rare_cells", len(rp_), bsf, "real task x penalty x decoder rows with rare_purity_diff_mean")
    derived("bio_rare_neg", int((rp_.rare_purity_diff_mean < 0).sum()), bsf, "of those, rare_purity_diff_mean < 0")
    w = rp_.loc[rp_.rare_purity_diff_mean.idxmin()]
    assert (w.task, w.method, w.decoder) == ("lung", "discriminator", "conditioned"), \
        "largest rare-type drop moved; the text names lung, discriminator, conditioned"
    derived("bio_rare_worst", w.rare_purity_diff_mean, bsf, "min rare_purity_diff_mean (lung, discriminator, conditioned)")
    derived("bio_lung_lam", w.lam, bsf, "lam of that row (label-transfer strength)")
    ftf = os.path.join(T, "lung_fit_types.csv")
    ft = pd.read_csv(ftf)
    rare = ft[ft.rare.astype(bool)]
    assert rare.celltype.unique().tolist() == ["Ionocytes"]
    io = rare[(rare.method == "discriminator") & (rare.decoder == "conditioned")]
    io0 = rare[(rare.method == "scvi") & (rare.decoder == "conditioned")]
    assert len(io) == 3 and len(io0) == 3 and io.n_cells.nunique() == 1
    derived("io_n", int(io.n_cells.iloc[0]), ftf, "n_cells, Ionocytes")
    derived("io_rec", io.lt_recall.mean(), ftf, "mean over 3 seeds of lt_recall (Ionocytes, discriminator, conditioned)")
    derived("io_rec0", io0.lt_recall.mean(), ftf, "mean over 3 seeds of lt_recall (Ionocytes, plain scVI, conditioned)")
    cff = os.path.join(T, "lung_confusion.csv")
    cf = pd.read_csv(cff)
    q = cf[(cf.method == "discriminator") & (cf.decoder == "conditioned") & (cf.true_type == "Ionocytes") & (cf.pred_type != "Ionocytes")]
    top = q.sort_values("share_mean").iloc[-1]
    assert top.pred_type == "Dendritic cell"
    derived("io_dc", top.share_mean, cff, "share_mean, discriminator conditioned, Ionocytes -> Dendritic cell (largest off-diagonal)")
    one(cf, cff, "io_rec_sd", "share_sd", method="discriminator", decoder="conditioned", true_type="Ionocytes", pred_type="Ionocytes")
    # all-type label-transfer recall difference: mean over types per fit, seed mean per task, mean over real tasks
    ftp = os.path.join(T, "<real task>_fit_types.csv")
    fts = pd.concat([pd.read_csv(os.path.join(T, f"{t}_fit_types.csv")) for t in REAL])
    fts = fts[fts.method.isin(PEN)]
    per = fts.groupby(["task", "method", "decoder", "seed"]).lt_recall_diff.mean().groupby(["task", "method", "decoder"]).mean()
    for p in PEN:
        for c in (0, 1):
            x = per.xs((p, dec[c]), level=("method", "decoder"))
            assert len(x) == 6
            derived(f"bio_rec_{p}_{c}", x.mean(), ftp, f"mean over 6 real tasks of seed-mean of per-fit mean over types of lt_recall_diff ({p}, {dec[c]})")
    o = [V[f"bio_rec_{p}_{c}"] for p in ("reference", "pooled", "barycenter", "mmd") for c in (0, 1)]
    derived("bio_rec_oth_lo", min(o), ftp, "min of the above over reference, pooled, barycenter, MMD x decoders")
    derived("bio_rec_oth_hi", max(o), ftp, "max of the above over reference, pooled, barycenter, MMD x decoders")
    bjf = os.path.join(a.bio, "biology_check.json")
    rec = json.load(open(bjf))
    derived("bio_nfits", sum(sum(v.values()) for v in rec["counts"].values()), bjf, "sum of counts")
    derived("bio_commit", rec["git_commit"][:7], bjf, "git_commit")
    return PN


# ---------------------------------------------------------------- templates
def snippets(V):
    def r2(lo, hi, fmt="d"):  # one number if both ends are equal
        return f"{{{{{lo}:{fmt}}}}}" if V[lo] == V[hi] else f"{{{{{lo}:{fmt}}}}} to {{{{{hi}:{fmt}}}}}"
    cnt = lambda c, tool, k: ", ".join(f"{{{{pl_{c}_{p}_{tool}_{k}:d}}}}" for p in PEN)
    S = {}
    S["abs6"] = ("At equal numbers of tries with the batch-conditioned decoder, a penalty at one strength drawn at random exceeded "
                 "plain scVI's selection score on at most {{plmax_1_none_1:d}} of six real tasks, and the best of six strengths on "
                 + r2("plmin_1_none_6", "plmax_1_none_6") + "; at six tries every penalty exceeded Harmony and sysVI on "
                 "{{plmin_1_harmony_6:d}} of six real tasks, whereas scANVI, which uses cell-type labels, was exceeded only on "
                 "pancreas, by at most {{scanvi_dmax:f3}}.")
    S["abs8"] = ("For scVI users, the choice of strength mattered more than the choice of penalty: the penalties were near-tied at "
                 "matched batch removal ({{n_ties:d}} of {{n_within:d}} within-pair differences under 0.02), a penalty improved on "
                 "batch-conditioned plain scVI mainly when several strengths were tried and scored, and scANVI, where cell-type "
                 "labels exist, scored higher than every penalty on most tasks.")
    S["m135"] = ("The sweep identifies which pairs are low-signal: the immune task and both simulations with the batch-conditioned "
                 "decoder, where plain scVI's batch score and the highest batch score every penalty reaches were {{ls_imm_b0:f3}} "
                 "and {{ls_imm_bc:f3}} for immune (gap {{ls_imm_gap:f3}}), {{ls_s1_b0:f3}} and {{ls_s1_bc:f3}} for simulation 1 "
                 "(gap {{ls_s1_gap:f3}}) and {{ls_s2_b0:f3}} and {{ls_s2_bc:f3}} for simulation 2 (gap {{ls_s2_gap:f3}}); no pair "
                 "is low-signal with the unconditioned decoder.")
    S["res1"] = ("Of the {{bl_expected:d}} baseline fits, {{bl_scored:d}} were scored and {{bl_failed:d}} failed: the scvi-tools "
                 "adversary {{bl_scvi_adv_scored:d}} of {{bl_scvi_adv_expected:d}}, scANVI {{bl_scanvi_scored:d}} of "
                 "{{bl_scanvi_expected:d}}, sysVI {{bl_sysvi_scored:d}} of {{bl_sysvi_expected:d}}, Harmony {{bl_harmony_scored:d}} "
                 "of {{bl_harmony_expected:d}}, Scanorama {{bl_scanorama_scored:d}} of {{bl_scanorama_expected:d}} and PCA "
                 "{{bl_pca_scored:d}} of {{bl_pca_expected:d}}. The remaining fit, Scanorama with 80 neighbours on "
                 "the large chromatin-accessibility task, could not be fitted (it ran out of memory, and a retry stopped at a 32-bit "
                 "sparse-matrix index limit); it is reported as an incomplete setting, not as a failed fit, so on that task "
                 "Scanorama is placed with five of its six settings (Supplementary Table S5).")
    S["res7"] = ("At the strength chosen on real tasks of other families, the penalties with the batch-conditioned decoder exceeded "
                 "plain scVI's selection score on " + ", ".join(f"{{{{tpos_{p}_1:d}}}} ({SHORT[p]})" for p in PEN)
                 + " of the six real tasks, with differences from plain scVI from {{tdiff_min:sf3}} to {{tdiff_max:sf3}} (median "
                 "{{tdiff_med:sf3}}; Table 4); at each task's own best strength, the optimistic bound, they did so on "
                 + r2("bposmin_1", "bposmax_1") + " of six. At their transferred settings, Harmony, Scanorama and sysVI exceeded "
                 "batch-conditioned plain scVI on {{tpos_harmony_1:d}}, {{tpos_scanorama_1:d}} and {{tpos_sysvi_1:d}} of six real "
                 "tasks. With the unconditioned decoder, whose plain scVI receives no batch information, the penalties at the "
                 "transferred strength exceeded plain scVI on " + r2("tposmin_0", "tposmax_0") + " of six (Supplementary Table S7). "
                 "The transferred strengths with the batch-conditioned decoder were " +
                 "; ".join(f"{SHORT[p]} " + r2(f"tlam_lo_{p}", f"tlam_hi_{p}", "lam").replace(" to ", "--") for p in PEN) + ".")
    S["res8"] = ("With the batch-conditioned decoder, a penalty's expected best selection score at one try exceeded plain scVI's on "
                 + cnt(1, "none", 1) + " of the six real tasks (in the order discriminator, reference, pooled and barycenter "
                 "critics, MMD, Sinkhorn divergence), {{plsum_1_none_1:d}} of {{pl_cells:d}} task--penalty cells, and at six tries "
                 "on " + cnt(1, "none", 6) + " ({{plsum_1_none_6:d}} of {{pl_cells:d}}). Averaged over the six real tasks, plain "
                 "scVI scored {{bk_none_1:f3}}; the penalties ranged from {{bkmin_1_1:f3}} to {{bkmax_1_1:f3}} at one try and from "
                 "{{bkmin_1_6:f3}} to {{bkmax_1_6:f3}} at six (Figure 6a), and their real-task means first exceeded plain scVI's at "
                 + ", ".join(f"{{{{kfirst_{p}_1:d}}}}" for p in PEN) + " tries in the same order. Against the standard tools at "
                 "six tries, every penalty exceeded Harmony on {{plmin_1_harmony_6:d}} of six real tasks, sysVI on "
                 "{{plmin_1_sysvi_6:d}} of six and Scanorama on all {{pln_1_pooled_scanorama_6:d}} tasks where its six "
                 "settings exist; at one try the counts were Harmony " + r2("plmin_1_harmony_1", "plmax_1_harmony_1")
                 + ", sysVI " + r2("plmin_1_sysvi_1", "plmax_1_sysvi_1") + " and Scanorama " + r2("plmin_1_scanorama_1", "plmax_1_scanorama_1")
                 + " of six. The scvi-tools adversary (mean {{bk_adv_1:f3}}) was exceeded at one try on "
                 + r2("plmin_1_scvi_adv_1", "plmax_1_scvi_adv_1") + " of six tasks and at six tries on "
                 + r2("plmin_1_scvi_adv_6", "plmax_1_scvi_adv_6") + ". scANVI, which uses the cell-type labels (mean "
                 "{{bk_scanvi_1:f3}}), was exceeded by no penalty at one try with either decoder; at six tries with the "
                 "batch-conditioned decoder every penalty exceeded it on pancreas only, by {{scanvi_dmin:f3}} to {{scanvi_dmax:f3}}, "
                 "and with the unconditioned decoder on no task. With the unconditioned decoder the reference is plain scVI without "
                 "batch information (mean {{bk_none_0:f3}}), which the penalties exceeded at one try on "
                 + r2("plmin_0_none_1", "plmax_0_none_1") + " of six real tasks and at six tries on "
                 + r2("plmin_0_none_6", "plmax_0_none_6") + ". Every best-of-$k$ value is optimistic for methods with more than one "
                 "setting, because the best setting is picked and scored on the same data (Supplementary Note 7).")
    S["disc3"] = ("For users of scVI, the sweep points to the penalty strength, more than the choice among penalties, as the "
                  "setting that matters. At matched batch removal the six penalties were near-tied: {{n_ties:d}} of the "
                  "{{n_within:d}} within-pair differences in biology over the 13 ranked pairs were under 0.02, and {{n_large:d}} were "
                  "0.04 or more. With the batch-conditioned decoder, going from one to six tries raised each penalty's mean expected "
                  "best selection score over the real tasks by {{bkgain_min_1:f3}} to {{bkgain_max_1:f3}}, whereas at six tries the "
                  "six penalties' means spanned only {{bkmin_1_6:f3}} to {{bkmax_1_6:f3}}. Added to scVI as it is usually run, with "
                  "the batch-conditioned decoder, a penalty at one strength drawn at random rarely beat plain scVI "
                  "({{plsum_1_none_1:d}} of {{pl_cells:d}} task--penalty cells); the best of six strengths did in "
                  "{{plsum_1_none_6:d}} of {{pl_cells:d}}, by small margins (real-task means {{bkmin_1_6:f3}}--{{bkmax_1_6:f3}} "
                  "against {{bk_none_1:f3}}).")
    S["disc3b"] = ("When to add a penalty: when several strengths can be fitted and scored on the task, because the real-task mean "
                   "of the best of $k$ first exceeded batch-conditioned plain scVI at {{kfirst_mmd_1:d}} (MMD) to "
                   "{{kfirst_discriminator_1:d}} (discriminator) tries. Strength range: the strengths chosen on other task families "
                   "with the batch-conditioned decoder lay between {{tlam_all_lo:lam}} and {{tlam_all_hi:lam}}. Standard tools: at "
                   "equal tries the penalties exceeded Harmony, Scanorama and sysVI on most real tasks, but not scANVI, which uses "
                   "cell-type labels and was exceeded only on pancreas at six tries. These are optimistic best-of-$k$ readings on "
                   "eight scIB tasks.")
    S["concl1"] = ("On eight tasks of the single-cell integration benchmark, six alignment penalties added to one scVI model kept "
                   "similar biology at matched batch removal: the best mean rank was the pooled critic's with the batch-conditioned "
                   "decoder ({{mr_pooled_1:f2}}) and MMD's with the unconditioned decoder ({{mr_mmd_0:f2}}), but {{n_ties:d}} of the "
                   "{{n_within:d}} within-pair differences were under 0.02, so the ranking does not separate most penalties. At equal "
                   "numbers of tries with the batch-conditioned decoder, the penalties exceeded Harmony, Scanorama and sysVI on most "
                   "real tasks and plain scVI mainly when several strengths were tried; scANVI, which uses cell-type labels, was "
                   "exceeded only on pancreas.")
    S["concl2"] = ("Practical guidance for scVI users, within these tasks: (i) tune the strength rather than the penalty, and keep "
                   "plain scVI with the batch-conditioned decoder as a candidate, since a penalty at one strength drawn at random "
                   "beat it in {{plsum_1_none_1:d}} of {{pl_cells:d}} task--penalty cells and the best of six strengths in "
                   "{{plsum_1_none_6:d}}; (ii) start from the strengths chosen on other task families with that decoder, "
                   "{{tlam_all_lo:lam}} to {{tlam_all_hi:lam}} across penalties (Results); (iii) check the biology directly at the "
                   "chosen strength, because the discriminator lost the most neighbourhood purity in the direct biology check "
                   "({{bio_purity_diff_discriminator_1:sf3}} against plain scVI, {{bio_rare_purity_diff_discriminator_1:sf3}} for "
                   "rare types, batch-conditioned decoder); (iv) when cell-type labels are available, use scANVI.")
    S["sn5"] = ("$b^*$ lies below the highest common score, which every curve reaches, so with no failed fit every curve reaches "
                "$b^*$ by construction; in the sweep {{n_noreading:d}} of the {{n_combos:d}} task, decoder and penalty "
                "combinations lacked a reading, and the bottom-rank rule was never used")
    S["sn5b"] = ("the immune task, plain scVI {{ls_imm_b0:f3}}, highest common {{ls_imm_bc:f3}}, gap {{ls_imm_gap:f3}}; "
                 "simulation 1, {{ls_s1_b0:f3}}, {{ls_s1_bc:f3}}, gap {{ls_s1_gap:f3}}; simulation 2, {{ls_s2_b0:f3}}, "
                 "{{ls_s2_bc:f3}}, gap {{ls_s2_gap:f3}}; all with the conditioned decoder, none with the unconditioned decoder")
    S["meth_bio"] = ("\\qquad A direct biology check reads four per-cell-type quantities on the saved embeddings, with no new fits: "
                     "neighbourhood purity, the share of a cell's 15 nearest neighbours in the latent space that carry its own "
                     "cell-type label, averaged within each type; the same purity and the label-transfer recall for rare types, "
                     "those with under 1\\% of a task's cells; marker-gene preservation, the Jaccard overlap between each type's "
                     "top 20 markers from the expression as distributed and from the expression averaged over the 15 neighbours, "
                     "on the four RNA tasks; and the label-transfer confusion table. Penalties are read at the label-transfer "
                     "strength and every baseline at each of its settings, and each value is reported as a difference from plain "
                     "scVI. The check is exploratory: its rules were written after the sweep scores had been read and before any of "
                     "its values was computed (Supplementary Note 12).")
    S["res_bio_head"] = ("A direct biology check reads purity, rare types, marker genes and label-transfer confusion per cell "
                         "type (Supplementary Table S10).")
    oth = "the reference, pooled and barycenter critics and MMD"
    S["res_bio"] = ("This check is exploratory and descriptive: it was defined after the sweep scores had been read, uses "
                    "{{bio_nfits:d}} saved fits (penalties at the label-transfer strength, seeds 100--102; plain scVI; every scored "
                    "baseline setting) and runs no test (Methods; Supplementary Note 12). Over the six real tasks, neighbourhood "
                    "purity per cell type differed from plain scVI with the same decoder and seed by "
                    "{{bio_purity_diff_discriminator_1:sf3}} for the discriminator with the batch-conditioned decoder and "
                    "{{bio_purity_diff_discriminator_0:sf3}} with the unconditioned decoder, and by {{bio_purity_diff_sinkhorn_1:sf3}} "
                    "and {{bio_purity_diff_sinkhorn_0:sf3}} for the Sinkhorn divergence; " + oth + " stayed between "
                    "{{bio_oth_lo:sf3}} and {{bio_oth_hi:sf3}} with either decoder. Harmony, averaged over its {{bio_nset_harmony:d}} "
                    "settings, differed from batch-conditioned plain scVI by {{bio_purity_diff_harmony:sf3}}, and scANVI, which is "
                    "trained with the labels, by {{bio_purity_diff_scanvi:sf3}}. Rare types (present on the pancreas, lung, immune "
                    "and human--mouse immune tasks) lost purity in {{bio_rare_neg:d}} of {{bio_rare_cells:d}} task, penalty and "
                    "decoder cells, most with the discriminator ({{bio_rare_purity_diff_discriminator_1:sf3}} with the "
                    "batch-conditioned decoder). The largest drop, {{bio_rare_worst:sf3}}, was for the {{io_n:d}} ionocytes of the "
                    "lung task with the batch-conditioned discriminator at strength {{bio_lung_lam:lam}}: their label-transfer recall "
                    "was {{io_rec:f3}} (seed SD {{io_rec_sd:f3}}) against {{io_rec0:f3}} for plain scVI, and a share of "
                    "{{io_dc:f3}} of them was predicted as dendritic cells. Over all cell types, label-transfer recall differed from "
                    "plain scVI by {{bio_rec_discriminator_1:sf3}} and {{bio_rec_discriminator_0:sf3}} for the discriminator with the "
                    "two decoders and by {{bio_rec_oth_lo:sf3}} to {{bio_rec_oth_hi:sf3}} for " + oth + ". Marker genes were kept "
                    "alike: the Jaccard overlap of the top 20 markers was {{bio_mj_lo:f3}} to {{bio_mj_hi:f3}} for every penalty and "
                    "plain scVI, with differences from plain scVI of at most {{bio_mjd_abs:f3}}, whereas Harmony "
                    "({{bio_marker_jaccard_diff_harmony:sf3}}) and PCA ({{bio_marker_jaccard_diff_pca:sf3}}) were lower. The direct "
                    "readouts single out the discriminator, which also had the worst mean rank with the batch-conditioned decoder "
                    "and the lowest mean label-transfer F1 with both decoders (Res-5); with the batch-conditioned decoder its loss "
                    "of purity was larger for rare types than over all types.")
    S["sn12"] = ("Supplementary Note 12, direct biology check: exploratory and descriptive; the rules were written after the sweep "
                 "scores (batch and biology scores, $b^*$, matched strengths, label-transfer macro F1) had been read and before any "
                 "value of this check was computed; no new fits, no test. Fits: for each penalty, decoder and task, the label-transfer "
                 "strength (the failure-free strength whose seed-mean batch score is nearest $b^*$, ties to the smaller strength), "
                 "seeds 100--102; plain scVI with both decoders, seeds 100--102; every scored baseline row at each of its settings "
                 "({{bio_nfits:d}} fits on eight tasks; low-signal pairs included and flagged). Neighbours: the 15 nearest other "
                 "cells by Euclidean distance in the saved latent. Readouts per fit and cell type: neighbourhood purity (share of "
                 "the 15 neighbours with the cell's own label, averaged over the type's cells; per fit, the unweighted mean over "
                 "types); rare types, under 1\\% of a task's cells, with their purity and label-transfer recall; marker-gene "
                 "preservation on the four RNA tasks (log-normalised expression as distributed, restricted to the 2,000 highly "
                 "variable genes; top 20 Wilcoxon markers per type against all other cells, computed once on the expression as "
                 "distributed and once on each fit's expression averaged over the 15 neighbours; Jaccard overlap of the two sets; "
                 "types under 10 cells get no marker readout); label-transfer confusion from the classifier of Supplementary Note 9 "
                 "(each batch held out in turn, 15-nearest-neighbour classifier, types present in one batch only dropped), with "
                 "per-type recall and the row-normalised confusion table. Differences: penalties and plain scVI against plain scVI "
                 "with the same decoder and seed; baselines against the mean of plain scVI with the conditioned decoder over seeds "
                 "100--102; a fit's difference is the mean of its per-type differences. Summaries: per task, method, setting and "
                 "decoder, the mean and SD over seeds, then the mean over the real tasks (Table S10). scANVI is trained with the "
                 "cell-type labels, so its readouts are not label-free. Checks that stop the run: latent checksums, cell order and "
                 "labels against the task file, fit counts, seed-matched references, readouts within [0, 1], and the macro F1 "
                 "recomputed from the confusion predictions equal to the label-transfer F1 of Supplementary Note 9. Code commit "
                 "{{bio_commit}}, branch \\texttt{biology-check}.")
    return S


PROWS = [("discriminator", "JS discriminator"), ("reference", "Reference critic"), ("pooled", "Pooled critic"),
         ("barycenter", "Barycenter critic"), ("mmd", "MMD"), ("sinkhorn", "Sinkhorn")]


def table3():
    rows = []
    for p, nm in PROWS:
        for c, dn in ((1, "conditioned"), (0, "unconditioned")):
            s1, s2 = (f"{{{{t3s1_{p}:g}}}}", f"{{{{t3s2_{p}:g}}}}") if c == 0 else ("--", "--")
            nr = "immune" if c == 1 else "none"
            rows.append(f"{nm if c == 1 else ' ' * 17} & {dn} & {{{{mr_{p}_{c}:f2}}}} & {{{{t3lo_{p}_{c}:g}}}}--{{{{t3hi_{p}_{c}:g}}}} & "
                        f"{{{{t3bio_{p}_{c}:f3}}}} & {s1} & {s2} & {nr} & {{{{t3n_{p}_{c}:d}}}}/{{{{nreal_{c}:d}}}} \\\\")
    return "\n".join(rows)


T4COLS = ["pancreas", "lung", "immune", "atac_small", "atac_large", "immune_hum_mou"]


def table4():
    one = lambda nm, key: nm + " & " + " & ".join(f"{{{{t4_{key}_{t}:f3}}}}" for t in T4COLS) + " \\\\"
    two = lambda nm, fam: nm + " & " + " & ".join(f"{{{{t4_{fam}_{t}:f3}}}} ({{{{t4b_{fam}_{t}:f3}}}})" for t in T4COLS) + " \\\\"
    out = [one("Plain scVI", "plain"), one("scvi-tools adversary", "adv")] + [two(nm, p) for p, nm in PROWS] + ["\\midrule"]
    out += [two(nm, f) for f, nm in (("harmony", "Harmony"), ("scanorama", "Scanorama"), ("sysvi", "sysVI"))]
    out += [one("scANVI (labels)", "scanvi"), one("PCA (uncorrected)", "pca")]
    return "\n".join(out)


def table_s10(V):
    out = []
    for p, nm in PROWS + [("scvi", "Plain scVI")]:
        for c, dn in ((1, "conditioned"), (0, "unconditioned")):
            out.append(f"{nm if c == 1 else ''} & {dn} & {{{{bio_purity_diff_{p}_{c}:sf3}}}} & {{{{bio_sd_{p}_{c}:f3}}}} & "
                       f"{{{{bio_rare_purity_diff_{p}_{c}:sf3}}}} & {{{{bio_rare_lt_recall_diff_{p}_{c}:sf3}}}} & "
                       f"{{{{bio_marker_jaccard_{p}_{c}:f3}}}} & {{{{bio_marker_jaccard_diff_{p}_{c}:sf3}}}} \\\\")
    out.append("\\midrule")
    for m, nm, dn in (("scvi_adv", "scvi-tools adversary", "conditioned"), ("scanvi", "scANVI (labels)", "--"),
                      ("sysvi", "sysVI", "--"), ("harmony", "Harmony", "--"), ("scanorama", "Scanorama", "--"), ("pca", "PCA", "--")):
        sd = f"{{{{bio_sd_{m}:f3}}}}" if V[f"bio_sd_{m}"] != "n/a" else "n/a"
        out.append(f"{nm} ({{{{bio_nset_{m}:d}}}}) & {dn} & {{{{bio_purity_diff_{m}:sf3}}}} & {sd} & {{{{bio_rare_purity_diff_{m}:sf3}}}} & "
                   f"{{{{bio_rare_lt_recall_diff_{m}:sf3}}}} & {{{{bio_marker_jaccard_{m}:f3}}}} & {{{{bio_marker_jaccard_diff_{m}:sf3}}}} \\\\")
    return "\n".join(out)


# ---------------------------------------------------------------- edits: (id, old text, new template); old must occur once
def edits(S, V):
    X = "xx.xx & [xx.xx, xx.xx] & x.xx & x.xx & x.xx & xx & x/6 \\\\"
    t3_old = "\n".join((f"{nm} & conditioned & {X}\n" + " " * 18 + f"& unconditioned & {X}") for _, nm in
                       [(0, "JS discriminator"), (0, "Reference critic"), (0, "Pooled critic"), (0, "Barycenter critic"), (0, "MMD"), (0, "Sinkhorn")])
    pen_rows, one_rows = " & ".join(["xx.xx (xx.xx)"] * 6), " & ".join(["xx.xx"] * 6)
    t4_old = "\n".join([f"Plain scVI & {one_rows} \\\\", f"scvi-tools adversary & {one_rows} \\\\"] +
                       [f"{nm} & {pen_rows} \\\\" for _, nm in PROWS] + ["\\midrule"] +
                       [f"{m} & {pen_rows} \\\\" for m in ("Harmony", "Scanorama", "sysVI")] +
                       [f"scANVI (labels) & {one_rows} \\\\", f"PCA (uncorrected) & {one_rows} \\\\"])
    return [
        ("preamble", "\\usepackage{todonotes}\n",
         "\\definechangesauthor[name={Pass 9}, color=olive]{P9}% result placeholders filled from the sweep, the placement check and the biology check; Conclusions; change log paper/pass9_changelog.md, numbers in paper/pass9_numbers.csv\n\\usepackage{todonotes}\n"),
        ("header", "Figures 1--6 and Tables 3--4 are mocks;",
         "\\replaced[id=P9]{Figures 1--6 are mocks (made on another branch); Tables 3 and 4 hold sweep values (Pass 9, change log \\texttt{paper/pass9\\_changelog.md}, number ledger \\texttt{paper/pass9\\_numbers.csv});}{Figures 1--6 and Tables 3--4 are mocks;}"),
        ("abs6", "\\res{placement against plain scVI, Harmony, Scanorama, sysVI, scANVI at equal tries, including whether any penalty beats plain scVI.}", S["abs6"]),
        ("abs8", "\\res{one-sentence guidance for scVI users.}",
         "\\replaced[id=P9]{" + S["abs8"] + "}{[result: one-sentence guidance for scVI users.]}"),
        ("m135", "The sweep identifies which pairs are low-signal: \\res{low-signal task--decoder pairs, with plain scVI's batch score, the highest batch score every penalty reaches and the gap}.", S["m135"]),
        ("meth_bio", "\\added[id=P3]{\\qquad Large language model tools were used",
         "\\added[id=P9]{" + S["meth_bio"] + "}\n\n\\added[id=P3]{\\qquad Large language model tools were used"),
        ("res_hd", "plus one for the replication across training seeds}}",
         "plus one for the replication across training seeds}\\added[id=P9]{ and one for the direct biology check}}"),
        ("res1", "\\res{baseline fits completed and failed per method, task and decoder; scoring remains incomplete.}", S["res1"]),
        ("res_bio", "\\pp{\\replaced{Res-6}{Res-7}}",
         "\\pp{\\added[id=P9]{Res-5b}}{\\added[id=P9]{" + S["res_bio_head"] + "}}\n\\added[id=P9]{" + S["res_bio"] + "}\n\\pp{\\replaced{Res-6}{Res-7}}"),
        ("res7", "\\item \\res{selection score per method, decoder, task at strength from other families\\added{, with its difference from plain scVI}; own-best strength as optimistic bound; batch against biology score per real task.}",
         "\\item \\replaced[id=P9]{" + S["res7"] + "}{[result: selection score per method, decoder, task at strength from other families, with its difference from plain scVI; own-best strength as optimistic bound; batch against biology score per real task.]}"),
        ("res8k", "$k=1$--\\replaced[id=P4]{\\tbd{upper $k$: the tools have six settings, the penalties ten or twelve sweep strengths}}{6}",
         "$k=1$--\\replaced[id=P4]{6, the number of settings of Harmony, Scanorama and sysVI (comparisons at equal tries stop at six, although the penalties have ten or twelve sweep strengths)}{6}"),
        ("res8", "\\item \\res{expected best selection score at $k=1$--\\replaced[id=P4]{\\tbd{upper $k$}}{6} per family and decoder\\added[id=P3]{ (Figure 6a shows the mean over the six real tasks of each method's per-task curve)}; whether any \\replaced{penalty}{adversary} exceeds plain scVI, Harmony, Scanorama or sysVI.}",
         "\\item \\replaced[id=P9]{" + S["res8"] + "}{[result: expected best selection score at $k=1$--upper $k$ per family and decoder (Figure 6a shows the mean over the six real tasks of each method's per-task curve); whether any penalty exceeds plain scVI, Harmony, Scanorama or sysVI.]}"),
        ("disc3", "\\res{guidance for scVI users, including the outcome that no alignment penalty improves on plain scVI.}", S["disc3"]),
        ("disc3b", "\\item \\res{when a penalty is worth adding; strength range; whether a standard tool does better at equal tries.} Word from placement results only.",
         "\\item \\replaced[id=P9]{" + S["disc3b"] + "}{[result: when a penalty is worth adding; strength range; whether a standard tool does better at equal tries.] Word from placement results only.}"),
        ("disc5", "not a dose-response test. Adversary steps reuse one minibatch.", "not a dose-response test.\\deleted[id=P9]{ Adversary steps reuse one minibatch.}"),
        ("concl1", "\\res{main conclusion: which penalties keep the most biology at matched batch removal, and whether any beats plain scVI or a standard tool at equal tries.}", S["concl1"]),
        ("concl2", "\\res{why it matters for scVI users: when a penalty is worth adding and at what strength.}", S["concl2"]),
        ("sn5", "\\tbd{confirm from the analysis code whether a curve can still fail to reach $b^*$; if it cannot, the bottom-rank rule is never used}", S["sn5"]),
        ("sn5min", "reported only where \\tbd{minimum number of the three readings} exist", "reported only where at least two of the three readings exist"),
        ("sn5b", "\\res{low-signal pairs and their values (plain scVI batch score, highest common batch score, gap)}", S["sn5b"]),
        ("sn7", "$k=1$--\\replaced[id=P4]{\\tbd{upper $k$}}{6}, optimistic",
         "$k=1$--\\replaced[id=P4]{6, the number of settings of Harmony, Scanorama and sysVI; the per-task tables continue the penalties' curves to their ten or twelve strengths, but comparisons at equal tries stop at six}{6}, optimistic"),
        ("sn12", "\\end{itemize}\n\\textbf{Supplementary Tables}",
         "\\item \\added[id=P9]{" + S["sn12"] + "}\n\\end{itemize}\n\\textbf{Supplementary Tables}"),
        ("supp_hd", "Tables S1--S\\replaced{8}{6}\\added[id=P5]{ and S9}",
         "Tables S1--S\\replaced{8}{6}\\added[id=P5]{ and S9}\\added[id=P9]{, Supplementary Note 12 and Table S10}"),
        ("ts10", "\\end{itemize}\n\\textbf{Supplementary Figures}",
         "\\item \\added[id=P9]{Table S10, direct biology check: per method and decoder, the mean over the six real tasks of neighbourhood purity minus plain scVI's with its seed spread, rare-type purity and label-transfer recall minus plain scVI's, and marker-gene Jaccard overlap with its difference from plain scVI (typeset after Table 4); per-task, per-cell-type and confusion tables in the repository.}\n\\end{itemize}\n\\textbf{Supplementary Figures}"),
        ("sn1", "\\item Supplementary Note 1, model and loss equations: scVI loss",
         "\\item Supplementary Note 1, model and loss equations:\\added[id=P9]{ notation, with the reconstruction term written per gene $g$ as $p_{\\mathrm{ZINB}}(x_g \\mid \\mu_g, \\theta_g, \\pi_g)$ (mean, dispersion, dropout) and the discriminator loss as $\\min_{h}\\, \\mathbb{E}_{z \\sim q(z \\mid x)}[\\mathcal{L}_{\\mathrm{CE}}(h(z), s)]$ for batch label $s$;} scVI loss"),
        ("figs6", "$k=1$ to \\tbd{upper $k$}, one panel per real task", "$k=1$ to 6, one panel per real task"),
        ("fig6", "$k=1$--\\replaced[id=P4]{\\tbd{upper $k$}}{6}, \\replaced[id=P3]", "$k=1$--\\replaced[id=P4]{6}{6}, \\replaced[id=P3]"),
        ("t3cap", "\\caption{\\replaced{MOCK -- placeholder values. Benchmark ranking", "\\caption{\\replaced{\\deleted[id=P9]{MOCK -- placeholder values. }Benchmark ranking"),
        ("t3cap2", "Per-task values are in Supplementary Table S6.}{MOCK",
         "Per-task values are in Supplementary Table S6.\\added[id=P9]{ Range, lowest and highest rank over the ranked real tasks; biology, mean over those tasks of the biology score at $b^*$; the simulations are low-signal with the conditioned decoder (--); ranked, real tasks ranked of six.}}{MOCK"),
        ("t3", t3_old, table3()),
        ("t4cap", "\\caption{MOCK -- placeholder values. Single-strength placement:", "\\caption{\\deleted[id=P9]{MOCK -- placeholder values. }Single-strength placement:"),
        ("t4cap2", "Unconditioned decoder in Supplementary Table \\replaced{S7}{[to be decided: S-number]}.}",
         "Unconditioned decoder in Supplementary Table \\replaced{S7}{[to be decided: S-number]}.\\added[id=P9]{ Scanorama on the large chromatin-accessibility task has five of its six settings (80 neighbours could not be fitted). The immune task is low-signal with this decoder.}}"),
        ("t4", t4_old, table4()),
        ("ts10tab", "\\end{tabular}\n\\end{table}\n\\clearpage\n\\vspace{4pt}",
         "\\end{tabular}\n\\end{table}\n\n% ---------------------------------------------------------------- Supplementary Table S10 (Pass 9)\n"
         "\\begin{table}[t]\n\\centering\\small\n\\renewcommand{\\tablename}{Supplementary Table}\\renewcommand{\\thetable}{S10}% numbered apart from the main tables, inside this float only\n\\caption{\\added[id=P9]{Direct biology check (exploratory; rules written after the sweep scores had been read; Supplementary Note 12). Means over the six real tasks of per-task seed means. Purity and rare-type columns are differences from plain scVI (penalties and plain scVI: same decoder and seed; baselines: conditioned plain scVI, mean of seeds 100--102). Seed SD, mean over the six real tasks of the SD of the purity difference over seeds (n/a for one-run tools). Rare types on the four real tasks that have them; marker-gene Jaccard on the four RNA tasks. Baselines with several settings: mean over settings (number of settings in parentheses; Scanorama has six on the large chromatin-accessibility task). scANVI uses cell-type labels.}}\n"
         "\\label{tab:s10}\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
         "Method & Decoder & Purity diff. & Seed SD & Rare purity diff. & Rare recall diff. & Marker Jaccard & Jaccard diff. \\\\\n\\midrule\n"
         + table_s10(V) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\\clearpage\n\\vspace{4pt}"),
    ]


def apply(tex, E, V):
    for eid, old, new in E:
        n = tex.count(old)
        if n != 1:
            raise ValueError(f"edit {eid}: old text occurs {n} times")
        tex = tex.replace(old, f"\0{eid}\0")
    starts = {}
    for eid, old, new in E:  # render after all old texts are located, so new text never matches a later old text
        pre = tex.split(f"\0{eid}\0")[0]
        starts[eid] = pre.count("\n") + 1
        tex = tex.replace(f"\0{eid}\0", V.render(new, eid))
    return tex, starts


def ledger_rows(tex, starts, V):
    lines, out, pos = tex.split("\n"), [], {}
    for r in V.rows:   # rows of one snippet come in text order: walk forward from the snippet's first line
        i = pos.get(r["snippet"], starts[r["snippet"]])
        while r["number_as_written"] not in lines[i - 1].replace("$", "").replace("\\%", "%"):
            i += 1
            if i > len(lines):
                raise ValueError(f"ledger: {r['number_as_written']} of {r['snippet']} not found")
        pos[r["snippet"]] = i
        out.append(dict(tex_line=i, **r))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=f"{D}/sweep_analysis_final_20261105")
    ap.add_argument("--placement", default=f"{D}/code/wt_sweep_paper/experiments/sweep_checks/out/placement_summary.csv")
    ap.add_argument("--bio", default=f"{D}/code/wt_biology_check/docs/biology_check")
    ap.add_argument("--replication", default=None, help="replication result JSON (optional; its edits are skipped without it)")
    ap.add_argument("--tex", default="paper/outline.tex")
    ap.add_argument("--ledger", default="paper/pass9_numbers.csv")
    a = ap.parse_args(argv)
    V = Vals()
    load(a, V)
    E = edits(snippets(V), V)
    if a.replication:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pass9_replication
        E += pass9_replication.edits(a.replication, V)
    tex, _ = apply(open(a.tex).read(), E, V)
    open(a.tex, "w").write(tex)
    # line numbers: locate each snippet's first rendered line in the final text
    lines = tex.split("\n")
    starts = {}
    for eid, old, new in E:
        probe = Vals()
        probe.v = V.v
        first = probe.render(new, eid).split("\n")[0][:100]
        starts[eid] = next(i for i, l in enumerate(lines, 1) if first in l)
    rows = ledger_rows(tex, starts, V)
    with open(a.ledger, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tex_line", "snippet", "number_as_written", "source_file", "source_key_or_row", "source_value"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(dict(edits=len(E), ledger_rows=len(rows), values=len(V.v),
                          skipped_replication=not a.replication)))


if __name__ == "__main__":  # pragma: no cover  # script entry point; tests call main() directly
    main()
