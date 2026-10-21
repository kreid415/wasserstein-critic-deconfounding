"""Pre-check of the optimiser-matched discriminator (experiments/x6_precheck_20261020/PLAN.md)."""
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
    lams = {"atac_small|0": (300, 1000, 3000), "atac_small|1": (0.3, 1, 3), "immune|0": (0.1, 0.3, 1),
            "immune|1": (0.1, 0.3, 1), "sim1|0": (10, 30, 100), "sim1|1": (100, 300, 1000)}
    rec = {"cells": {k + "|discriminator": dict(lo=float(a), matched=float(b), hi=float(c)) for k, (a, b, c) in lams.items()}}
    r4 = tmp_path / "r4.json"; r4.write_text(json.dumps(rec))
    df = load("build_manifest").build(V5, str(r4))
    v5 = pd.read_csv(V5, sep="\t", comment="#", dtype=str, keep_default_na=False)
    assert len(df) == 18 and df.tag.is_unique and not set(df.tag) & set(v5.tag)
    assert set(df.experiment) == {"PC6"} and set(df.arm) == {"discriminator_sn_critic_opt"} and set(df.seed) == {"100"}
    same = [c for c in v5.columns if c not in ("tag", "experiment", "arm")]
    src = v5[(v5.experiment == "A1") & (v5.arm == "discriminator")][same]
    assert len(df[same].merge(src, how="inner")) == 18          # every other field equals a pilot discriminator row
    got = {f"{t}|{c}": tuple(sorted(float(x) for x in g.lam)) for (t, c), g in df.groupby(["task", "cond"])}
    assert got == {k: tuple(float(x) for x in v) for k, v in lams.items()}
    assert df.tag.str.match(r"^PC6_(atac_small|immune|sim1)_discriminator_sn_critic_opt_l[0-9.]+_c[01]_s100_[0-9a-f]{8}$").all()
