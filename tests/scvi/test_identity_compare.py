"""experiments/followup_build_20261017/identity_compare.py (the X1 fit-identity and no-fit plan-identity comparison): a pair
of identical latent files passes even when git_sha / fit_seconds differ; every other difference (latent, saved config at
an optimiser setting, loss history, exit code, a missing file, an optimiser step) fails and the config path is named.
    PYTHONPATH=. python -m pytest -q tests/scvi/test_identity_compare.py
"""
import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location(
    "identity_compare", os.path.join(ROOT, "experiments", "followup_build_20261017", "identity_compare.py"))
ic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ic)

CFG = {"row": {"arm": "pooled"}, "git_sha": "a", "git_dirty": False, "fit_seconds": 1.0,
       "plan": {"fit": {"spectral_norm": False, "optimizers": [{"param_groups": [{"lr": 1e-3}, {"lr": 1e-4}]}]}}}


def _write(out, label, tag, z=1.0, cfg=None, hist="[]"):
    d = out / f"out_{label}" / "latents"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{tag}.npz", z=np.full((4, 2), z, dtype=np.float32), batch=np.array(["a", "b"]),
             config=json.dumps(cfg or CFG), history=hist)


def _cfg(**over):
    c = json.loads(json.dumps(CFG))
    c.update(over)
    return c


@pytest.mark.parametrize("name,new,rc_new,steps_new,passes,differs_at", [
    ("identical apart from provenance", dict(cfg=_cfg(git_sha="b", fit_seconds=9.0, git_dirty=True)), 0, 0, True, ""),
    ("latent differs", dict(z=1.0 + 1e-6), 0, 0, False, ""),
    ("an optimiser setting differs", dict(cfg=_cfg(plan={"fit": {"spectral_norm": False, "optimizers": [
        {"param_groups": [{"lr": 1e-3}, {"lr": 1e-3}]}]}})), 0, 0, False, "/plan/fit/optimizers[0]/param_groups[1]/lr"),
    ("spectral norm differs", dict(cfg=_cfg(plan={"fit": {"spectral_norm": True, "optimizers": CFG["plan"]["fit"]["optimizers"]}})),
     0, 0, False, "/plan/fit/spectral_norm"),
    ("loss history differs", dict(hist="[1]"), 0, 0, False, ""),
    ("exit code 1 at NEW", {}, 1, 0, False, ""),
    ("an optimiser step was taken", {}, 0, 3, False, ""),
])
def test_identity_compare_known_answers(tmp_path, name, new, rc_new, steps_new, passes, differs_at):
    for label, kw in (("old", {}), ("new", new)):
        _write(tmp_path, label, "t1", **kw)
    (tmp_path / "tags.txt").write_text("t1\n")
    (tmp_path / "exit_codes.txt").write_text(f"old t1 0\nnew t1 {rc_new}\n")
    for label, steps in (("old", 0), ("new", steps_new)):
        (tmp_path / f"out_{label}" / "steps.txt").write_text(f"t1 {steps}\n")
    code = ic.main([str(tmp_path), "r.csv", "--no-models", "--steps-zero"])
    row = next(__import__("csv").DictReader(open(tmp_path / "r.csv")))
    assert (code == 0) is passes, (name, row)
    assert differs_at in row["config_differs_at"], (name, row)


def test_compare_models_nested_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    state = lambda w: {"model_state_dict": {"module": {"w": torch.tensor([1.0, w])}, "b": torch.zeros(2)}}  # noqa: E731
    for name, w in (("a", 2.0), ("b", 2.0), ("c", 2.5)):
        torch.save(state(w), tmp_path / f"{name}.pt")
    assert ic.compare_models(tmp_path / "a.pt", tmp_path / "b.pt")
    assert not ic.compare_models(tmp_path / "a.pt", tmp_path / "c.pt")


def test_identity_compare_missing_file_fails(tmp_path):
    _write(tmp_path, "old", "t1")
    (tmp_path / "tags.txt").write_text("t1\n")
    (tmp_path / "exit_codes.txt").write_text("old t1 0\nnew t1 0\n")
    assert ic.main([str(tmp_path), "r.csv", "--no-models"]) == 1
