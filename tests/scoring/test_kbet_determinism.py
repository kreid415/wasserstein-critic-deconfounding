"""kBET must be identical when the same latent is scored twice (score_scib_native.seed_kbet: R's RNG is set
with set.seed(KBET_SEED) immediately before every R kBET call). Runs in the wcd-kbet env with R_HOME and
R_LIBS as in scripts/run_wave.sh; skipped, with the reason, where rpy2 or the R kBET package is missing."""
import os
import sys
import types

os.environ.setdefault("KMP_AFFINITY", "disabled")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ro = pytest.importorskip("rpy2.robjects", reason="rpy2 is not installed (kBET runs in the wcd-kbet env)")
from rpy2.rinterface_lib.embedded import RRuntimeError  # noqa: E402

try:
    ro.r("suppressMessages(library(kBET))")
except RRuntimeError as exc:
    pytest.skip(f"R kBET package not loadable (set R_LIBS as in scripts/run_wave.sh): {exc}", allow_module_level=True)
ad = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")
import score_scib_native as SN  # noqa: E402


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """540 cells, 3 batches x 3 cell types, a latent with a partial batch effect (kBET strictly between 0 and 1)."""
    d = tmp_path_factory.mktemp("kbet")
    rng = np.random.default_rng(0)
    b, ct = np.repeat(np.arange(3), 180), np.tile(np.repeat(np.arange(3), 60), 3)
    centers, shift = rng.normal(0, 3, (3, 40)), rng.normal(0, 1, (3, 40))
    X = (centers[ct] + 0.5 * shift[b] + rng.normal(0, 1, (540, 40))).astype(np.float32)
    a = ad.AnnData(X)
    a.obs_names = [f"cell{i}" for i in range(540)]
    a.var_names = [f"g{i}" for i in range(40)]
    a.obs["batch"] = pd.Categorical([f"b{i}" for i in b])
    a.obs["celltype"] = pd.Categorical([f"c{i}" for i in ct])
    a.var["highly_variable"] = True
    sc.pp.pca(a, n_comps=20)
    a.uns.update(batch_key="batch", celltype_key="celltype", organism="none", modality="rna")
    pre = str(d / "toy__scib.h5ad")
    a.write_h5ad(pre)
    z = 0.3 * centers[ct][:, :8] + 0.8 * shift[b][:, :8] + rng.normal(0, 0.6, (540, 8))
    npz = str(d / "toy.npz")
    np.savez(npz, z=z.astype(np.float32), batch=a.obs["batch"].astype(str).to_numpy(),
             celltype=a.obs["celltype"].astype(str).to_numpy(), obs_names=a.obs_names.to_numpy())
    return d, pre, npz


def _score(monkeypatch, d, pre, npz, name, seed=None, mod=SN):
    out = str(d / f"{name}.csv")
    for k, v in dict(PREPPED=pre, NPZ=npz, TAG=name, OUT_CSV=out).items():
        monkeypatch.setenv(k, v)
    if seed is None:
        monkeypatch.delenv("KBET_SEED", raising=False)
    else:
        monkeypatch.setenv("KBET_SEED", str(seed))
    mod.main()
    return pd.read_csv(out).iloc[0]


def test_same_latent_twice_gives_identical_kbet(monkeypatch, inputs):
    d, pre, npz = inputs
    r1, r2 = _score(monkeypatch, d, pre, npz, "run1"), _score(monkeypatch, d, pre, npz, "run2")
    assert 0.0 < r1["kBET"] < 1.0, r1["kBET"]
    assert r1["kBET"] == r2["kBET"]                                   # bit-identical, not approximately equal
    assert r1["kbet_seed"] == r2["kbet_seed"] == SN.KBET_SEED_DEFAULT
    assert r1["kbet_r_calls"] == r2["kbet_r_calls"] == 3              # one seeded R call per cell type
    others = [m for m in SN.BATCH_METRICS + SN.BIO_METRICS if m != "kBET" and np.isfinite(r1[m])]
    assert others and all(r1[m] == r2[m] for m in others)


def test_the_seed_is_live(monkeypatch, inputs):
    """Negative control: a different KBET_SEED changes kBET on the same latent, so the identity above comes
    from the seed, not from kBET being deterministic on this input."""
    d, pre, npz = inputs
    a, b = _score(monkeypatch, d, pre, npz, "seed1", seed=1), _score(monkeypatch, d, pre, npz, "seed2", seed=2)
    assert a["kbet_seed"] == 1 and b["kbet_seed"] == 2
    assert a["kBET"] != b["kBET"]


def test_a_non_integer_seed_is_refused(monkeypatch):
    monkeypatch.setenv("KBET_SEED", "zero")
    with pytest.raises(ValueError, match="not an integer"):
        SN.kbet_seed()


def test_mutant_without_set_seed_fails_the_identity_check(monkeypatch, inputs):
    """Mutation test (fail-loud R11): the same scorer without the set.seed line gives different kBET on the
    same latent, so test_same_latent_twice_gives_identical_kbet can detect a missing seed."""
    src = open(SN.__file__).read()
    old = '        ro.r(f"set.seed({int(seed)})")\n'
    assert src.count(old) == 1
    mod = types.ModuleType("score_scib_native_unseeded")
    mod.__file__ = SN.__file__
    exec(compile(src.replace(old, ""), SN.__file__ + "<mutant>", "exec"), mod.__dict__)
    d, pre, npz = inputs
    try:
        r1 = _score(monkeypatch, d, pre, npz, "unseeded1", mod=mod)
        r2 = _score(monkeypatch, d, pre, npz, "unseeded2", mod=mod)
    finally:
        SN.seed_kbet(SN.KBET_SEED_DEFAULT)            # leave the real, seeded wrapper installed
    assert r1["kBET"] != r2["kBET"]
