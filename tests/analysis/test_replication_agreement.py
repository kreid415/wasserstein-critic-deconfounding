"""Tests of experiments/sweep_replication/agreement_test.py (replication agreement rule, tag replication-prediction-v1)."""
import importlib.util, json, os, sys
import numpy as np, pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prereg_rules as P  # noqa: E402

spec = importlib.util.spec_from_file_location("agreement_test", os.path.join(ROOT, "experiments/sweep_replication/agreement_test.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
TASKS = ["sim1", "sim2", "lung", "pancreas", "immune"]  # real task names: prereg_rules.bio_metrics_for needs them
ARMS = ["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"]
PRED_FILE = os.path.join(ROOT, "experiments/sweep_replication/PREDICTION.json")


@pytest.mark.parametrize("bios,expected", [
    ({"a": .70, "b": .65, "c": .65, "d": .60, "e": None, "f": None}, dict(a=1, b=2.5, c=2.5, d=4, e=5.5, f=5.5)),  # ANALYSIS_SPEC 3.1 example
    ({k: None for k in "abcdef"}, {k: 3.5 for k in "abcdef"}),
])
def test_rank_readings(bios, expected):  # test-reason: tie averaging and shared bottom places of ANALYSIS_SPEC 3.1
    assert A.rank_readings(bios) == expected


def test_sign_flip_p():  # test-reason: exact permutation p on hand-countable inputs and non-finite rejection
    assert A.sign_flip_p([1, 1]) == 0.25 and A.sign_flip_p([1, -1]) == 0.75
    with pytest.raises(ValueError):
        A.sign_flip_p([1, np.nan])
    with pytest.raises(ValueError):
        A.sign_flip_p([])


def test_committed_prediction_is_self_consistent():  # test-reason: ranks and gap list of the committed file follow from its readings
    pred = json.load(open(PRED_FILE))
    assert len(pred["bstar"]) == pred["n_ranked_pairs"] == 13 == len(pred["readings"])
    for key, rd in pred["readings"].items():
        assert A.rank_readings({a: rd[a]["bio_at_bstar"] for a in ARMS}) == {a: rd[a]["rank"] for a in ARMS}
    assert A.sign_flip_p([0.5] * 13) == 1 / 8192


def _synthetic(sweep_bio, reverse=False, unreach=None, drop_seed=False):
    """Prediction dict with 5 pairs and a replication outcomes frame whose reading at b* = sweep bio (or reversed order)."""
    pred = dict(bstar={}, readings={}, large_gaps=[], n_near_ties=0)
    rows = []
    for k, bios in enumerate(sweep_bio):
        key = f"{TASKS[k]}|0"; pred["bstar"][key] = 0.5
        pred["readings"][key] = {a: dict(bio_at_bstar=v, rank=A.rank_readings(dict(zip(ARMS, bios)))[a]) for a, v in zip(ARMS, bios)}
        for s in A.SEEDS:
            rows.append(dict(experiment="X1", task=TASKS[k], cond_i=0, arm="none", seed_i=s, lamf=0.0, status="ok", B=0.2, C=0.9))
            for a, v in zip(ARMS, bios):
                c = (1 - v) if reverse else v
                lo, hi = (0.1, 0.2) if a == unreach and k == 0 else (0.4, 0.6)
                if drop_seed and k == 0 and a == "mmd" and s == 4:
                    continue
                rows += [dict(experiment="X1", task=TASKS[k], cond_i=0, arm=a, seed_i=s, lamf=1.0, status="ok", B=lo, C=c + 0.05),
                         dict(experiment="X1", task=TASKS[k], cond_i=0, arm=a, seed_i=s, lamf=10.0, status="ok", B=hi, C=c - 0.05)]
    pred["n_near_ties"] = sum(abs(x - y) < A.NEAR_TIE for b in sweep_bio for i, x in enumerate(b) for y in b[i + 1:])
    g = sweep_bio[0]
    pred["large_gaps"] = [dict(pair=f"{TASKS[0]}|0", a=ARMS[0], b=ARMS[1], sweep_diff=g[0] - g[1], sign=int(np.sign(g[0] - g[1])))]
    df = pd.DataFrame(rows)
    df["lam"] = df.lamf.astype(str)
    df["tag"] = df.task + "_" + df.arm + "_" + df.lamf.astype(str) + "_s" + df.seed_i.astype(str)
    return pred, df


SWEEP = [[.70, .60, .55, .50, .45, .40], [.66, .61, .50, .52, .47, .41], [.30, .35, .40, .45, .50, .55],
         [.71, .62, .63, .50, .44, .43], [.58, .57, .52, .49, .48, .46]]


@pytest.mark.parametrize("kw,agree,kept,n_unreached", [
    ({}, True, 1, 0),                          # replication reproduces the sweep: tau = 1 in all 5 pairs, p = 1/32
    ({"reverse": True}, False, 0, 0),          # reversed order: sign lost, tau < 0
    ({"unreach": "reference"}, False, 0, 1),   # a gap side that never reaches b*: no replication sign, counted as not kept
])
def test_evaluate(kw, agree, kept, n_unreached):  # test-reason: verdict logic (agree, reversed, unreached side of a gap) on known readings
    pred, out = _synthetic(SWEEP, **kw)
    R, PR, G, S = A.evaluate(pred, A.read_pairs(P, out, pred["bstar"], ARMS))
    assert (S["agree"], S["sign_kept"], S["n_unreached"]) == (agree, kept, n_unreached)
    assert PR.top_rep.notna().all() and R.lam_max.eq(10.0).all()
    if not kw:
        assert (PR.top_rep == PR.top_sweep).all()
        assert S["mean_tau"] == pytest.approx(1.0) and S["p_value"] == 1 / 32 and S["agree_slope_adj"]
    if "reverse" in kw:
        assert S["mean_tau"] < 0 and not S["agree_slope_adj"]


def test_missing_seed_fails_loudly():  # test-reason: an incomplete lambda x seed design must stop the reading
    pred, out = _synthetic(SWEEP, drop_seed=True)
    with pytest.raises(Exception, match="seeds"):
        A.read_pairs(P, out, pred["bstar"], ARMS)


def test_slope_adjusted_needs_variation():  # test-reason: slope guard when all points of a pair share one batch score
    with pytest.raises(ValueError):
        A.slope_adjusted({"a": [(0.4, 0.5), (0.4, 0.6)]}, 0.5)


def _inputs(out):
    """Fake manifest and score tables from a synthetic outcomes frame (every batch metric = B, every biology metric = C)."""
    M = out.assign(cond=out.cond_i, seed=out.seed_i, lam=out.lamf)[["tag", "experiment", "task", "cond", "seed", "arm", "lam"]]
    S = pd.DataFrame([{"tag": r.tag, "kbet_seed": 0, **{m: r.B for m in P.BATCH_METRICS}, **{m: r.C for m in P.bio_metrics_for(r.task)}}
                      for r in out.itertuples()])
    S["scorer_git_sha"] = "10a1ca5874fbcdc033b148525b5e71d866306bdb"
    for m in P.BIO_METRICS:  # bio metrics of other tasks are not read, but read_scores requires the columns
        S[m] = S.get(m, 0.5)
    return M, S


@pytest.mark.parametrize("drop_score", [False, True])
def test_main_end_to_end(tmp_path, monkeypatch, drop_score):  # test-reason: command-line path and fail-loud on a manifest row without a score
    pred, out = _synthetic(SWEEP)
    M, S = _inputs(out)
    if drop_score:
        S = S[S.tag != M.tag.iloc[-1]]
    monkeypatch.setattr(P, "read_manifest", lambda p: (M, None))
    monkeypatch.setattr(P, "read_scores", lambda p: (S, []))
    (tmp_path / "pred.json").write_text(json.dumps(pred))
    argv = ["--rules-scripts", os.path.join(ROOT, "scripts"), "--manifest", "m", "--scores", "s", "--prediction", str(tmp_path / "pred.json"), "--out", str(tmp_path / "o")]
    if drop_score:
        with pytest.raises(P.PreregError, match="neither scored nor failed"):
            A.main(argv)
        return
    A.main(argv)
    summ = json.load(open(tmp_path / "o" / "replication_agreement_summary.json"))
    assert summ["agree"] and summ["n_scored"] == len(M) and summ["sign_kept"] == 1
