"""End to end for the sweep pipeline (scripts/run_sweep_analysis.sh: sweep_inputs.py -> analysis_sweep.py -> sweep_report.py) on a
synthetic data root laid out like the real one (tier12_v3/rules/r4_pass2 score list, jhpce_tier12/pilot5_jhpce runner layout, two
baseline directories), with the committed pilot5 manifest and baseline run list. Every expected value follows from the planted case
(SCENARIO), not from the code under test. Synthetic inputs only."""
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.environ.get("ACCEPT_WORKDIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests", "prereg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analysis_x1 as A  # noqa: E402
import prereg_rules as P  # noqa: E402
import sweep_inputs as I  # noqa: E402
import synth  # noqa: E402

SEEDS = ["100", "101", "102"]
TASKS = sorted(I.PILOT3_TASKS + I.PILOT5_TASKS)
CENTRE = {("js", 0): 0.0, ("js", 1): 0.5, ("w1", 0): 1.0, ("w1", 1): 1.0, ("mmd", 0): 2.0, ("mmd", 1): 2.0,
          ("sinkhorn", 0): 1.5, ("sinkhorn", 1): 0.5}
BONUS = {"reference": 0.20, "pooled": 0.12, "barycenter": 0.06, "sinkhorn": -0.08}
SCENARIO = dict(
    low_signal=("immune", "1"),                 # every penalty gains 0.015: b_common - b0 < 0.02
    anchor_above=("sim2", "1"),                 # sinkhorn's batch score lies below plain scVI's at every strength: b_common < b0, so b* < b0
    tie=("pancreas", "0"),                      # barycenter is given pooled's scores exactly
    one_seed=("immune_hum_mou", "1"),           # reference responds in seed 101 only: without it the curve stays below b*
    failed={("lung", "0", "mmd", "3000", s): "diverged" for s in SEEDS} | {("pancreas", "1", "mmd", "3000", "102"): "nonfinite_latent",
                                                                           ("atac_small", "1", "reference", "10", "101"): "diverged"},
    baseline_failed={"X13_lung_sysvi_l5_c1_s2": "diverged"},
    cpu_tools_on=("pancreas", "lung"),          # harmony, scanorama and pca scored on these tasks only
    not_evaluable="atac_small")                 # every cell type in one batch only: label transfer has nothing to evaluate


def _gain(r):
    key = (r["task"], r["cond"])
    if key == SCENARIO["low_signal"]:
        return 0.015
    if key == SCENARIO["anchor_above"] and r["arm"] == "sinkhorn":
        return 0.0
    if key == SCENARIO["one_seed"] and r["arm"] == "reference":
        return 0.30 if r["seed"] == "101" else 0.004
    return 0.30


def world(r):
    """(status, B, C) of one row; seed effects are shared by every arm of a pair (pairing)."""
    key = f"{r['task']}|{r['cond']}|{r['seed']}"
    B = 0.40 + 0.01 * TASKS.index(r["task"]) + synth._noise("b" + key, 0.01)
    C = 0.75 + synth._noise("c" + key, 0.01)
    if r["arm"] in P.ADV_ARMS:
        arm = "pooled" if (r["task"], r["cond"]) == SCENARIO["tie"] and r["arm"] == "barycenter" else r["arm"]
        x = math.log10(float(r["lam"])) - CENTRE[(P.FAMILY_OF[arm], int(r["cond"]))]
        tag = f"{r['task']}|{r['cond']}|{arm}|{r['lam']}|{r['seed']}"
        B += _gain(dict(r, arm=arm)) * synth.sig(x / 0.3) + synth._noise("B" + tag, 0.003)
        C += -0.25 * synth.sig((x - 1) / 0.3) + BONUS.get(arm, 0.0) * synth.sig(x / 0.3) + synth._noise("C" + tag, 0.003)
        if (r["task"], r["cond"]) == SCENARIO["anchor_above"] and arm == "sinkhorn":
            B -= 0.05 + 0.05 * synth.sig(x / 0.3)
    elif r["arm"] == "none":
        pass                                     # plain scVI: the pair's reference point
    elif r["arm"] == "scvi_adv":
        B, C = B + 0.03, C - 0.01
    elif r["arm"] == "pca":
        B, C = B - 0.05, C - 0.10
    elif r["arm"] == "scanvi":
        B, C = B + 0.05, C + 0.03
    else:                                        # harmony, scanorama, sysvi: better with the setting
        B, C = B + 0.05 + 0.01 * math.log1p(float(r["lam"])), C - 0.02
    status = SCENARIO["failed"].get((r["task"], r["cond"], r["arm"], r["lam"], r["seed"]), SCENARIO["baseline_failed"].get(r["tag"][:-9], "ok"))
    return status, min(max(B, 0.0), 1.0), min(max(C, 0.0), 1.0)


def cells_of(task):
    if task == SCENARIO["not_evaluable"]:
        return [(f"b{b}", f"T{b}{t}") for b in range(3) for t in "ABC" for _ in range(8)]
    return [(f"b{b}", t) for b in range(3) for t in "ABC" for _ in range(8)] + [("b0", "D")] * 4


def write_fit(r, score_dir, latent_dir):
    """A score file and a latent file as the runner writes them; the reference critic's embedding follows the batches."""
    status, B, C = world(r)
    if status != "ok":
        return dict(tag=r["tag"], status=status, detail="synthetic planted failure")
    batches, types = zip(*cells_of(r["task"]))
    rng = np.random.default_rng(int(hashlib.sha1(r["tag"].encode()).hexdigest()[:8], 16))
    centre = dict(A=(0, 0), B=(10, 0), C=(0, 10), D=(10, 10), b0=(0, 0), b1=(10, 0), b2=(0, 10))
    z = np.array([centre.get(b if r["arm"] == "reference" else t, (20, 20)) for b, t in zip(batches, types)]) + 0.3 * rng.normal(size=(len(types), 2))
    lp = os.path.join(latent_dir, r["tag"] + ".npz")
    np.savez_compressed(lp, z=z.astype(np.float32), obs_names=np.array([f"c{i}" for i in range(len(types))]), batch=np.array(batches),
                        celltype=np.array(types))
    row = dict(tag=r["tag"], latent_sha256=P.sha256(lp), kbet_seed=0, score_seconds=1.0, cpu_model="AMD Ryzen 5 3600 6-Core Processor",
               **synth.PROVENANCE)
    bio = P.bio_metrics_for(r["task"])
    for m in synth.ALL_METRICS:
        row[m] = B + (0.01 if m in A.LOCAL else -0.015) if m in P.BATCH_METRICS else (C if m in bio else float("nan"))
    pd.DataFrame([row]).to_csv(os.path.join(score_dir, r["tag"] + ".csv"), index=False)
    return None


def runner_dir(d, rows):
    """Score, latent and failure files of `rows` in the runner's layout; returns the failure rows."""
    for sub in ("scores", "latents"):
        os.makedirs(os.path.join(d, sub))
    fails = [f for f in (write_fit(r, os.path.join(d, "scores"), os.path.join(d, "latents")) for r in rows.to_dict("records")) if f]
    pd.DataFrame(fails, columns=["tag", "status", "detail"]).to_csv(os.path.join(d, "failures.csv"), index=False)
    return fails


def frame(rows):
    return pd.DataFrame([{c: str(r[c]) for c in P.BPM.COLS} for r in rows], columns=P.BPM.COLS)


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("sweep_root"))
    out = str(tmp_path_factory.mktemp("sweep_out"))
    bb = P.BPM.BACKBONES["stock"]
    # three original tasks: A1 rows plus R1 grid-edge rows (BPM.build) plus the R4 extension 0.03 of three arms on immune only
    a1 = frame(P.BPM.finalize(P.BPM.build(bb, "pilot", 3, 5, 8, a1_tasks=list(I.PILOT3_TASKS))))
    ext = P.extension_rows(a1, [dict(experiment="A1", task="immune", cond=1, arm=arm, lam=0.03) for arm in ("discriminator", "pooled", "reference")])
    run, _ = P.read_manifest(I.RUN_LIST)
    base = run[(run.experiment == "X13") | (run.arm == "scvi_adv")]
    m3 = pd.concat([a1, ext, run[run.experiment == "X1"].head(20)], ignore_index=True)     # as m.r4a1.tsv: A1 rows among others
    r4 = os.path.join(root, "tier12_v3", "rules", "r4_pass2")
    os.makedirs(r4)
    P.write_manifest(m3, ["# synthetic m.r4a1.tsv"], os.path.join(r4, "m.r4a1.tsv"), "test")
    a1 = m3[m3.experiment == "A1"]
    v2, v3 = os.path.join(root, "tier12_v2", "A1"), os.path.join(root, "tier12_v3", "A1_sinkhorn")
    f2 = runner_dir(v2, a1[a1.arm != "sinkhorn"])
    f3 = runner_dir(v3, a1[a1.arm == "sinkhorn"])
    runner_dir(os.path.join(root, "tier12_v2", "A1_stale_sinkhorn"), a1[a1.arm == "sinkhorn"].head(5))    # superseded fits, same tags
    assert len(f2) == 1 and f3 == []
    with open(os.path.join(r4, "scores_list.txt"), "w") as f:
        f.write("".join(os.path.join(v3 if r["arm"] == "sinkhorn" else v2, "scores", r["tag"] + ".csv") + "\n"
                        for r in a1.to_dict("records") if r["tag"] != f2[0]["tag"]))
    pd.DataFrame(f2, columns=["tag", "status", "detail"]).to_csv(os.path.join(r4, "failures_merged.csv"), index=False)
    # five new tasks: the committed pilot5 manifest, scored in the runner layout
    m5, _ = P.read_manifest(I.PILOT5_MANIFEST)
    f5 = runner_dir(os.path.join(root, "jhpce_tier12", "pilot5_jhpce"), m5)
    assert len(f5) == 4
    # baselines in two directories: scanvi, sysvi and scvi_adv everywhere; the CPU tools on two tasks only
    cpu = base.arm.isin(["harmony", "scanorama", "pca"])
    d1, d2 = os.path.join(root, "baselines_1"), os.path.join(root, "baselines_2")
    first = base.task.isin(("pancreas", "lung", "immune", "sim1"))
    fb = runner_dir(d1, base[first & (~cpu | base.task.isin(SCENARIO["cpu_tools_on"]))])
    fb += runner_dir(d2, base[~first & ~cpu])
    assert [f["tag"][:-9] for f in fb] == list(SCENARIO["baseline_failed"])
    # A single scored seed with four neither-scored-nor-failed seeds must not count as a usable scANVI setting.
    partial = base[(base.task == "pancreas") & (base.arm == "scanvi")].groupby(["lam", "n_latent"], sort=True).head(1)
    remaining = base[(base.task == "pancreas") & (base.arm == "scanvi") & ~base.tag.isin(partial.tag)]
    for row in remaining.itertuples():
        for sub in ("scores", "latents"):
            os.unlink(os.path.join(d1, sub, row.tag + (".csv" if sub == "scores" else ".npz")))
    env = dict(os.environ, WCD_DATA_ROOT=root, PYTHON=sys.executable)
    sh = os.path.join(ROOT, "scripts", "run_sweep_analysis.sh")
    full = os.path.join(out, "full")
    r = subprocess.run(["bash", sh, full, "--baseline-dirs", d1, d2, "--jobs", "2"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return dict(root=root, out=out, full=full, env=env, sh=sh, dirs=[d1, d2], base=base, a1=a1, m5=m5, baseline_missing_extra=remaining.tag.tolist(),
                rec=json.load(open(os.path.join(full, "analysis_sweep.json"))), numbers=json.load(open(os.path.join(full, "paper_numbers.json"))))


def test_inputs_and_coverage(sweep):  # test-reason: the mapping of the three sources, the latent next to the listed score file, the coverage table and the refusals
    full, base = sweep["full"], sweep["base"]
    info = json.load(open(os.path.join(full, "inputs", "inputs.json")))
    M, _ = P.read_manifest(os.path.join(full, "inputs", "manifest.tsv"))
    assert info["counts"]["tier12_v3/rules/r4_pass2"] == dict(rows=len(sweep["a1"]), scored=len(sweep["a1"]) - 1, failed=1)
    assert info["counts"]["jhpce_tier12/pilot5_jhpce"] == dict(rows=1890, scored=1886, failed=4)
    missing = base[(base.arm.isin(["harmony", "scanorama", "pca"]) & ~base.task.isin(SCENARIO["cpu_tools_on"])) | base.tag.isin(sweep["baseline_missing_extra"])]
    assert sorted(info["baseline_rows_missing"]) == sorted(missing.tag) and len(missing) == 16 * 6 + len(sweep["baseline_missing_extra"])
    assert len(M) == len(sweep["a1"]) + 1890 + 448 - len(missing) and set(M.experiment) == {"A1", "X1", "X13"}
    cov = pd.read_csv(os.path.join(full, "inputs", "baseline_coverage.csv")).set_index(["task", "arm"])
    assert cov.loc[("lung", "sysvi")].to_dict() == dict(expected=30, scored=29, failed=1, missing=0)
    assert cov.loc[("atac_small", "harmony")].to_dict() == dict(expected=7, scored=0, failed=0, missing=7)
    assert cov.loc[("pancreas", "pca")].to_dict() == dict(expected=2, scored=2, failed=0, missing=0)
    setting = next(r for r in info["baseline_setting_coverage"] if r["task"] == "pancreas" and r["arm"] == "scanvi")
    assert setting["expected"] == 5 and setting["scored"] == 1 and setting["failed"] == 0 and setting["missing"] == 4
    assert setting["status"] == "incomplete"
    assert cov.expected.sum() == 448 and list(cov.columns) == ["expected", "scored", "failed", "missing"]
    lat = dict(ln.rstrip("\n").split("\t") for ln in open(os.path.join(full, "inputs", "latents.tsv")))
    sk = sweep["a1"][sweep["a1"].arm == "sinkhorn"].tag.iloc[0]
    assert lat[sk] == os.path.join(sweep["root"], "tier12_v3", "A1_sinkhorn", "latents", sk + ".npz")    # not the stale copy
    assert set(lat) == set(M.tag) - set(pd.read_csv(os.path.join(full, "inputs", "failures.csv")).tag)
    rec = sweep["rec"]
    assert rec["rule"] == "analysis_sweep" and rec["tasks"] == TASKS and rec["seeds"] == [100, 101, 102] and not rec["partial_tasks"]
    assert rec["baselines"]["complete"] is False and rec["min_readings"] == 2 and rec["label_transfer_debug_subset"] is None
    assert rec["counts"]["sweep_failed"] == dict.fromkeys(TASKS, 0) | dict(lung=3, pancreas=1, atac_small=1)
    assert len(rec["inputs"]["scores"]) == len(lat) and all(len(x["sha256"]) == 64 for v in rec["inputs"].values() for x in v)
    # --require-baselines stops on the missing rows and names them
    r = subprocess.run(["bash", sweep["sh"], os.path.join(sweep["out"], "strict"), "--baseline-dirs", *sweep["dirs"], "--require-baselines"],
                       env=sweep["env"], capture_output=True, text=True)
    log = open(os.path.join(sweep["out"], "strict", "logs", "inputs.log")).read()
    assert r.returncode != 0 and f"{len(missing)} of 448 baseline rows" in log and missing.tag.iloc[0] in log
    with pytest.raises(P.PreregError, match="two directories"):
        I.main(["--out", os.path.join(sweep["out"], "twice"), "--data-root", sweep["root"], "--baseline-dirs", sweep["dirs"][0], sweep["dirs"][0]])
    with pytest.raises(P.PreregError, match="unknown tasks"):
        I.main(["--out", os.path.join(sweep["out"], "badtask"), "--data-root", sweep["root"], "--tasks", "kidney"])


def test_sweep_readings_and_ranks(sweep):  # test-reason: every strength is a point; the planted tie, low-signal, anchor and one-seed cases; the 2-of-3 standard-error rule
    rec, cells = sweep["rec"], sweep["rec"]["primary"]["cells"]
    # every fitted strength is a point: the base grid, the R1 grid-edge strengths on all tasks, the R4 extension on immune only
    assert cells["lung|0"]["n_points"] == dict.fromkeys(P.ADV_ARMS, 10) | dict(discriminator=12)
    assert cells["immune|1"]["n_points"] == dict.fromkeys(P.ADV_ARMS, 10) | dict(sinkhorn=12, discriminator=11, pooled=11, reference=11)
    assert cells["atac_small|1"]["n_points"]["pooled"] == 10
    # with every strength a point, b_common is the lowest best batch score, so every curve brackets b* in a ranked pair
    for key, c in cells.items():
        if not c["low_signal"]:
            assert all(r["status"] == "bracketed" for r in c["penalties"].values()), key
            assert sum(r["rank"] for r in c["penalties"].values()) == 21
        assert min(c["best_batch"].values()) == pytest.approx(c["b_common"])
    pen = cells["pancreas|0"]["penalties"]
    assert pen["pooled"]["bio"] == pen["barycenter"]["bio"] and pen["pooled"]["rank"] == pen["barycenter"]["rank"] == 2.5
    low = cells["immune|1"]
    assert low["low_signal"] and low["hypervolume"] is None and set(low["nondominated"]) == set(P.ADV_ARMS) and all("rank" not in r for r in low["penalties"].values())
    anchor = cells["sim2|1"]
    assert anchor["low_signal"] and anchor["bstar"] < anchor["b0"]
    assert all((r["status"], r["bio"], r["local_minus_global"]) == (A.ANCHOR_STATUS, None, None) for r in anchor["penalties"].values())
    # reference in immune_hum_mou|1 reaches b* on the three-seed curve; without seed 101 it does not: 2 of 3 repeats read, and the
    # sweep rule (all but one repeat) gives a standard error where the five-seed rule (m >= 4) would not
    ref = cells["immune_hum_mou|1"]["penalties"]["reference"]
    assert ref["status"] == "bracketed" and [x is None for x in ref["loso"]] == [False, True, False] and ref["m"] == 2 and ref["s"] is not None
    assert A.loso_se(ref["loso"])[0] is None and ref["rank_hi"] == 6
    rk = rec["primary"]["ranking"]
    assert rk["1"]["low_signal"] == ["immune", "sim2"] and rk["0"]["low_signal"] == []
    assert all(v["n_tasks"] == 5 for v in rk["1"]["mean_rank"].values()) and all(v["n_tasks"] == 6 for v in rk["0"]["mean_rank"].values())
    assert set(rk["1"]["sim_ranks"]) == {"sim1"} and set(rk["0"]["sim_ranks"]) == {"sim1", "sim2"}
    # collapse over every strength and three seeds; the planted failures counted by type
    col = cells["lung|0"]["collapse"]
    assert col["mmd"]["n"] == 30 and col["mmd"]["failed"] == dict(diverged=3, nonfinite_latent=0) and col["discriminator"]["n"] == 36
    assert cells["pancreas|1"]["collapse"]["mmd"]["failed"] == dict(diverged=0, nonfinite_latent=1)
    # the curves are in the record for the figures: anchor first, one entry per strength with a successful run
    cv = cells["lung|0"]["curve"]["mmd"]
    assert cv[0][0] == 0.0 and len(cv) == 10 and [x[0] for x in cv[1:]] == sorted(x[0] for x in cv[1:]) and 3000.0 not in [x[0] for x in cv]
    # best-of-k: every strength is a try for a penalty, each tool its six settings where scored; plain scVI one
    bk = cells["lung|0"]["best_of_k"]
    assert (bk["pooled"]["n"], bk["discriminator"]["n"], bk["harmony"]["n"], bk["sysvi"]["n"], bk["none"]["n"], bk["scanvi"]["n"]) == (10, 12, 6, 6, 1, 1)
    assert cells["atac_small|0"]["best_of_k"]["harmony"]["n"] == 0 and cells["atac_small|0"]["best_of_k"]["harmony"]["E"] == []
    assert (bk["sysvi"]["configs"], bk["sysvi"]["failed_runs"]) == (6, 1)
    rm = rec["primary"]["best_of_k_real_mean"]["0"]
    assert [n for _, n in rm["harmony"]] == [2] * 6 and [n for _, n in rm["pooled"]] == [6] * 10 and [n for _, n in rm["discriminator"]] == [6] * 12
    # transfer strength: chosen on the other families' real tasks among strengths fitted in the held-out task and failure-free in
    # every training task; immune's extra strength 0.03 is never chosen for a task that did not fit it; a tool missing in a
    # training task has no eligible setting, its own best is still reported
    tr = rec["primary"]["transfer"]["1"]
    for t, v in tr["pooled"].items():
        assert v["lam"] in {x[0] for x in cells[f"{t}|1"]["curve"]["pooled"]} and v["at"]["n_ok"] == 3
    assert tr["pooled"]["atac_small"]["train"] == ["immune", "immune_hum_mou", "lung", "pancreas"]
    assert all(v["lam"] is None and v["at"] is None for v in tr["harmony"].values())
    assert tr["harmony"]["lung"]["best_lam"] == 8.0 and tr["harmony"]["lung"]["best"]["n_ok"] == 1 and tr["harmony"]["atac_small"]["best_lam"] is None
    # the sensitivity analyses ran on the sweep too
    assert set(rec["sensitivity"]) == set(A.SENSITIVITY) and len(rec["sensitivity"]["per_run"]["cells"]["lung|0"]["penalties"]["mmd"]["readings"]) == 3


def test_sweep_label_transfer(sweep):  # test-reason: label transfer on every penalty embedding, the matched-strength table and curve, the process pool
    rec, cells = sweep["rec"], sweep["rec"]["primary"]["cells"]
    lt = rec["primary"]["label_transfer"]
    n = len(sweep["a1"]) + 1890 - 5 + 40          # every successful sweep row (5 planted failures) and the 40 scvi_adv rows
    assert lt["scope"] == "all" and lt["debug_subset"] is None and lt["n_embeddings"] == n == len(lt["per_tag"]) == len(lt["files"])
    assert [x["path"] for x in rec["inputs"]["latents"]] == lt["files"]
    for t, arms in lt["decoders"]["1"]["tasks"].items():
        assert set(arms) == set(P.ADV_ARMS) | {"none", "scvi_adv"}
        bstar, curves = cells[f"{t}|1"]["bstar"], cells[f"{t}|1"]["curve"]
        for arm in P.ADV_ARMS:
            x, pts = arms[arm], curves[arm][1:]
            assert x["computed"] and len(x["curve"]) == len(pts) and [y["lam"] for y in x["curve"]] == [p[0] for p in pts]
            assert all(y["computed"] for y in x["curve"])
            # the matched strength: the failure-free point whose seed-mean batch score is closest to b*
            assert x["lam"] == min((p for p in pts if p[3] == 0), key=lambda p: (abs(p[1] - bstar), p[0]))[0]
            assert x["n_runs"] == 3 and (x["not_evaluable"] if t == SCENARIO["not_evaluable"] else (x["mean"] < 0.7 if arm == "reference" else x["mean"] == 1.0))
        assert arms["scvi_adv"]["n_runs"] == 5 and arms["none"]["lam"] is None and "curve" not in arms["none"]
    assert "scvi_adv" not in lt["decoders"]["0"]["tasks"]["lung"]
    s = lt["decoders"]["1"]["summary"]
    assert s["pooled"] == dict(mean=1.0, n_tasks=4, not_evaluable=[SCENARIO["not_evaluable"]]) and s["reference"]["mean"] < 0.7
    # the pool gives the same result as one process, and the record holds exactly that
    paths = dict(ln.rstrip("\n").split("\t") for ln in open(os.path.join(sweep["full"], "inputs", "latents.tsv")))
    S, _ = P.read_scores([ln.strip() for ln in open(os.path.join(sweep["full"], "inputs", "scores_list.txt"))][:40])
    tags = sorted(S.tag)[:6]
    latents = dict(paths=paths, sha=dict(zip(S.tag, S.latent_sha256)))
    one, files1 = A.label_transfer_many(tags, latents, jobs=1)
    two, files2 = A.label_transfer_many(tags, latents, jobs=2)
    assert one == two and files1 == files2 and all(json.loads(json.dumps(one[t])) == lt["per_tag"][t] for t in tags)
    with pytest.raises(P.PreregError, match="0 latent files"):
        A.label_transfer_many(["nope"], dict(paths={}, sha={"nope": "0" * 64}))
    with pytest.raises(P.PreregError, match="at least 1"):
        A.transfer_labels(pd.DataFrame(), pd.DataFrame(), {}, latents, {}, "all", 1, 0)


def test_report_and_paper_numbers(sweep):  # test-reason: tables and figures exist, paper_numbers.json carries the record's values under documented keys
    full, rec, N = sweep["full"], sweep["rec"], sweep["numbers"]
    tdir, fdir = os.path.join(full, "tables"), os.path.join(full, "figures")
    names = {"ranking_pairs", "low_signal_pairs", "mean_rank", "secondary", "curves", "baseline_points", "label_transfer", "label_transfer_curve", "best_of_k",
             "best_of_k_real_mean", "transfer_strength", "single_methods", "baseline_coverage", "baseline_setting_coverage", "counts"}
    assert {f[:-4] for f in os.listdir(tdir)} == names
    for fig in ("fig_ranking", "fig_tradeoff_cond1", "fig_tradeoff_cond0", "fig_best_of_k", "fig_collapse"):
        assert all(os.path.getsize(os.path.join(fdir, fig + ext)) > 0 for ext in (".png", ".pdf"))
    pairs = pd.read_csv(os.path.join(tdir, "ranking_pairs.csv"))
    assert len(pairs) == 84 and pairs["low_signal"].sum() == 0 and pairs["rank"].isna().sum() == 0
    low_signal = pd.read_csv(os.path.join(tdir, "low_signal_pairs.csv"))
    assert len(low_signal) == 12 and low_signal.low_signal.all()
    scanvi = pd.read_csv(os.path.join(tdir, "best_of_k.csv"))
    assert set(scanvi[scanvi.family == "scanvi"].stratum) == {"label_aware"}
    scanvi_mean = pd.read_csv(os.path.join(tdir, "best_of_k_real_mean.csv"))
    assert set(scanvi_mean[scanvi_mean.family == "scanvi"].stratum) == {"label_aware"}
    coverage = pd.read_csv(os.path.join(tdir, "baseline_setting_coverage.csv"))
    assert (coverage[(coverage.task == "pancreas") & (coverage.arm == "scanvi")].status == "incomplete").all()
    for cond in (0, 1):
        c = rec["primary"]["cells"][f"pancreas|{cond}"]
        assert c["best_of_k"]["scanvi"]["n"] == 0
        assert not any(v["arm"] == "scanvi" for v in c["baseline_points"].values())
    assert "fig_best_of_k.pdf" in os.listdir(fdir)
    assert set(pd.read_csv(os.path.join(tdir, "curves.csv")).arm) == set(P.ADV_ARMS) and len(pd.read_csv(os.path.join(tdir, "label_transfer_curve.csv"))) > 96
    assert all(isinstance(v, dict) and set(v) == {"value", "description"} for v in N.values())
    c = rec["primary"]["cells"]["lung|0"]
    assert N["rank.pooled.lung.cond0"]["value"] == c["penalties"]["pooled"]["rank"] and N["bio_at_bstar.pooled.lung.cond0"]["value"] == c["penalties"]["pooled"]["bio"]
    assert N["collapse_ci.mmd.lung.cond0"]["value"] == [c["collapse"]["mmd"]["lo"], c["collapse"]["mmd"]["hi"]]
    assert N["hypervolume.pooled.lung.cond0"]["value"] == c["hypervolume"]["pooled"] and not any(k.startswith("area.") for k in N)
    assert N["nondominated_share.pooled.lung.cond0"]["value"] == c["nondominated"]["pooled"]["share"]
    sec = pd.read_csv(os.path.join(tdir, "secondary.csv"))
    assert {"hypervolume", "nondominated_share", "nondominated_n"} <= set(sec.columns) and "area" not in sec.columns
    assert N["pair.sim2.cond1.low_signal"]["value"] is True and "rank.pooled.sim2.cond1" not in N and N["reading_status.pooled.sim2.cond1"]["value"] == A.ANCHOR_STATUS
    assert N["mean_rank.cond0.reference"]["value"] == rec["primary"]["ranking"]["0"]["mean_rank"]["reference"]["mean"]
    assert N["n_fits.total"]["value"] == len(sweep["a1"]) + 1890 and N["n_failed.total"]["value"] == 5
    assert N["baseline_coverage.atac_small.harmony.missing"]["value"] == 7 and N["_meta.baselines_complete"]["value"] is False
    assert N["_meta.partial_tasks"]["value"] is False and N["_meta.label_transfer_debug_subset"]["value"] is None
    assert N["n_pairs_no_reading.cond1.sinkhorn"]["value"] == 1 and N["n_pairs_no_reading.cond0.sinkhorn"]["value"] == 0
    assert N["label_transfer_f1.pooled.lung.cond1"]["value"] == 1.0 and N["_meta.record_sha256"]["value"] == P.sha256(os.path.join(full, "analysis_sweep.json"))
    assert N["best_of_k_real_mean.cond0.pooled.k10"]["value"] == rec["primary"]["best_of_k_real_mean"]["0"]["pooled"][9][0]
    man = open(os.path.join(full, "MANIFEST.sha256")).read()
    assert "analysis_sweep.json" in man and "paper_numbers.json" in man and "figures/fig_ranking.pdf" in man
    assert list(pd.read_csv(os.path.join(full, "timings.tsv"), sep="\t").stage) == ["inputs", "analysis", "report"]


def test_debug_subset_and_partial_tasks(sweep):  # test-reason: the debug option marks the record and the paper numbers and leaves the rest uncomputed; a task subset runs
    out = os.path.join(sweep["out"], "debug")
    r = subprocess.run(["bash", sweep["sh"], out, "--baseline-dirs", sweep["dirs"][0], "--tasks", "atac_small", "atac_large", "sim1", "--jobs", "1",
                        "--debug-label-transfer-subset", "2"], env=sweep["env"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.load(open(os.path.join(out, "analysis_sweep.json")))
    N = json.load(open(os.path.join(out, "paper_numbers.json")))
    assert rec["partial_tasks"] and rec["tasks"] == ["atac_large", "atac_small", "sim1"] and rec["label_transfer_debug_subset"] == 2
    lt = rec["primary"]["label_transfer"]
    assert lt["n_embeddings"] == 6 and lt["debug_subset"] == 2 and len(lt["files"]) == 6
    d = lt["decoders"]["1"]
    computed = [(t, a) for t, arms in d["tasks"].items() for a, x in arms.items() if x["computed"]]
    assert len(computed) <= 6 and all(x["mean"] is None for t, arms in d["tasks"].items() for a, x in arms.items() if (t, a) not in computed)
    assert "not_computed" in d["summary"]["pooled"] and len(d["summary"]["pooled"]["not_computed"]) + d["summary"]["pooled"]["n_tasks"] <= 2
    assert N["_meta.partial_tasks"]["value"] is True and N["_meta.label_transfer_debug_subset"]["value"] == 2
    # atac_small has no real task of another family here: no transfer strength; the simulation trains on both atac tasks
    tr = rec["primary"]["transfer"]["0"]["pooled"]
    assert (tr["atac_small"]["train"], tr["atac_small"]["lam"], tr["atac_small"]["at"]) == ([], None, None) and tr["atac_small"]["best_lam"] is not None
    assert tr["sim1"]["train"] == ["atac_large", "atac_small"]
    assert all(v["n_tasks"] == 2 for v in rec["primary"]["ranking"]["0"]["mean_rank"].values())
    assert os.path.getsize(os.path.join(out, "figures", "fig_tradeoff_cond0.png")) > 0
    # an existing output directory is refused
    r = subprocess.run(["bash", sweep["sh"], out, "--baseline-dirs", sweep["dirs"][0]], env=sweep["env"], capture_output=True, text=True)
    assert r.returncode == 2 and "refusing" in r.stderr


def test_simulation_only_report_skips_real_task_figures(sweep):  # test-reason: --tasks sim1 leaves real-task summaries empty
    out = os.path.join(sweep["out"], "sim1_only")
    r = subprocess.run(["bash", sweep["sh"], out, "--baseline-dirs", *sweep["dirs"], "--tasks", "sim1", "--jobs", "1"],
                       env=sweep["env"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.load(open(os.path.join(out, "analysis_sweep.json")))
    files = set(os.listdir(os.path.join(out, "figures")))
    assert rec["real_tasks"] == []
    assert not any(name.startswith(("fig_ranking", "fig_best_of_k")) for name in files)
    assert {"fig_tradeoff_cond0.pdf", "fig_tradeoff_cond1.pdf", "fig_collapse.pdf"} <= files


def test_anchor_at_or_above_bstar():  # test-reason: spec 4 (i) and 16.2: an anchor at or above b* gives no reading even when the curve falls through b*
    def point(b):
        return [dict(lam=1.0, n_fail=0, B={s: b for s in range(5)}, C={s: 0.60 for s in range(5)})]

    zero = {s: (0.60, 0.70) for s in range(5)}
    got = A.read_arms(zero, dict(a=point(0.50)), 0.55, sorted(zero))["a"]
    assert (got["status"], got["bio"]) == (A.ANCHOR_STATUS, None) and P.bio_at_bstar(got["curve"], 0.55)["status"] == "bracketed"
    assert A.read_arms({s: (0.55, 0.70) for s in range(5)}, dict(a=point(0.50)), 0.55, list(range(5)))["a"]["status"] == A.ANCHOR_STATUS
    below = A.read_arms({s: (0.50, 0.70) for s in range(5)}, dict(a=point(0.60)), 0.55, list(range(5)))["a"]
    assert below["status"] == "bracketed" and below["bio"] == pytest.approx(0.65)
    ranked = A.rank_one_pair(zero, dict(a=point(0.50)), 0.55)
    assert ranked["a"]["rank"] == 1 and ranked["a"]["loso"] == [None] * 5 and ranked["a"]["s"] is None
