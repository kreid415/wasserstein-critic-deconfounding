"""Pre-check of the optimiser-matched discriminator and of pooled_sn at n_critic 1 (experiments/x6_precheck_20261020/PLAN.md)."""
import importlib.util, json, os
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
E = os.path.join(ROOT, "experiments", "x6_precheck_20261020")
V5 = os.path.join(ROOT, "manifests", "paper_manifest_stock_pilot_u5_b10_v5.tsv")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(E, name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("pairs,bstar,want", [
    ([(1, 0.40), (3, 0.45), (10, None)], 0.50, "unreached_low"),
    ([(1, 0.60), (3, 0.70), (10, 0.80)], 0.50, "unreached_high"),
    ([(1, 0.40), (3, 0.50), (10, 0.60)], 0.50, "reached"),        # b* on a point counts as reached
    ([(1, None), (3, 0.70), (10, None)], 0.50, "not_readable"),
])
def test_cell_status(pairs, bstar, want):
    assert load("decide").cell_status(pairs, bstar) == want


@pytest.mark.parametrize("sts,want", [
    (["unreached_low"] * 3 + ["reached"], ("shift_up", 3, 0)),
    (["unreached_high"] * 3 + ["not_readable"], ("shift_down", 0, 3)),
    (["unreached_low"] * 2 + ["unreached_high"] * 2, ("keep", 2, 2)),
])
def test_decide(sts, want):
    assert load("decide").decide(dict(zip("abcd", sts))) == want


def test_build_copies_pilot_rows(tmp_path):
    lams = {"discriminator": {"atac_small|0": (300, 1000, 3000), "atac_small|1": (0.3, 1, 3), "immune|0": (0.1, 0.3, 1),
                              "immune|1": (0.1, 0.3, 1), "sim1|0": (10, 30, 100), "sim1|1": (100, 300, 1000)},
            "pooled": {"atac_small|0": (100, 300, 1000), "atac_small|1": (1, 3, 10), "immune|0": (0.3, 1, 3),
                       "immune|1": (3, 10, 30), "sim1|0": (30, 100, 300), "sim1|1": (0.03, 0.1, 0.3)}}   # 0.03: no pilot fit
    rec = {"cells": {f"{k}|{base}": dict(lo=float(a), matched=float(b), hi=float(c))
                     for base, cells in lams.items() for k, (a, b, c) in cells.items()}}
    r4 = tmp_path / "r4.json"; r4.write_text(json.dumps(rec))
    df = load("build_manifest").build(V5, str(r4))
    v5 = pd.read_csv(V5, sep="\t", comment="#", dtype=str, keep_default_na=False)
    assert len(df) == 36 and df.tag.is_unique and not set(df.tag) & set(v5.tag)
    assert set(df.experiment) == {"PC6"} and set(df.seed) == {"100"} and set(df.n_critic) == {"1"}
    same = [c for c in v5.columns if c not in ("tag", "experiment", "arm", "n_critic")]
    for arm, base in (("discriminator_sn_critic_opt", "discriminator"), ("pooled_sn", "pooled")):
        d = df[df.arm == arm]
        src = v5[(v5.experiment == "A1") & (v5.arm == base)][same]
        assert len(d) == 18 and len(d[same].merge(src, how="inner")) == 18 - (arm == "pooled_sn")   # every other field equals a pilot base-arm row
        keys = set(map(tuple, src.drop(columns="lam").values))          # the 0.03 row: equal apart from the strength
        assert all(tuple(r) in keys for r in d[same].drop(columns="lam").values)
        got = {f"{t}|{c}": tuple(sorted(float(x) for x in g.lam)) for (t, c), g in d.groupby(["task", "cond"])}
        assert got == {k: tuple(float(x) for x in v) for k, v in lams[base].items()}
        assert d.tag.str.match(rf"^PC6_(atac_small|immune|sim1)_{arm}_l[0-9.]+_c[01]_s100_[0-9a-f]{{8}}$").all()


def test_summarise_decides_each_arm_on_its_own_cells():  # test-reason: decide.py is outside --src scripts; catches an arm reading another arm's cells or base arm
    mod, cells = load("decide"), [(t, c) for t in ("atac_small", "immune", "sim1") for c in "01"]
    B = {"discriminator_sn_critic_opt": (0.40, 0.45, 0.48), "pooled_sn": (0.60, 0.70, 0.80)}   # b* is 0.5: below / above
    O = pd.DataFrame([dict(arm=a, task=t, cond=c, lam=str(lam), lamf=float(lam), B=b,
                           status="failed" if (a, t, c, lam) == ("pooled_sn", "atac_small", "0", 10) else "ok")
                      for a, bs in B.items() for t, c in cells for lam, b in zip((1, 3, 10), bs)])
    low = {"immune|1", "sim1|1"}
    rec = {"targets": {f"{t}|{c}": dict(bstar=0.5, bstar_low_signal=f"{t}|{c}" in low) for t, c in cells},
           "cells": {f"{t}|{c}|{b}": dict(matched=m, matched_B=0.5) for t, c in cells
                     for b, m in (("discriminator", 1.0), ("pooled", 10.0))}}
    out = mod.summarise(O, rec)
    json.dumps(out)                                                     # decision.json must be writable
    assert out["decisions"] == {"discriminator_sn_critic_opt": "shift_up", "pooled_sn": "shift_down"}
    a, b = (out["per_arm"][k] for k in B)
    assert (a["n_unreached_low"], a["n_unreached_high"], b["n_unreached_low"], b["n_unreached_high"]) == (4, 0, 0, 4)
    assert a["counted"] == list(mod.COUNTED) and len(a["cells"]) == len(b["cells"]) == 6
    cell = b["cells"]["atac_small|0"]                                   # a failed fit is a None point, still readable
    assert cell["points"] == [(1.0, 0.6), (3.0, 0.7), (10.0, None)] and cell["status"] == "unreached_high"
    assert (a["cells"]["sim1|1"]["low_signal"], a["cells"]["sim1|0"]["low_signal"]) == (True, False)
    assert (a["cells"]["immune|0"]["base_arm_matched"], cell["base_arm_matched"]) == (1.0, 10.0)   # each arm reads its base arm
    with pytest.raises(AssertionError, match="pooled_sn"):             # an arm without fits, or a cell without fits
        mod.summarise(O[O.arm != "pooled_sn"], rec)
    with pytest.raises(AssertionError, match="pooled_sn"):
        mod.summarise(O[~((O.arm == "pooled_sn") & (O.task == "sim1") & (O.cond == "1"))], rec)
