"""experiments/X1_local/link_v2_outputs.py (review F3): reusable v2 non-Sinkhorn X1 outputs reach the new folder as links,
chosen by file name; Sinkhorn outputs and other tasks' outputs never do; the old folder is not written to."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("link_v2_outputs", os.path.join(ROOT, "experiments", "X1_local", "link_v2_outputs.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def test_links_by_name_skip_sinkhorn_and_refuse_to_touch_real_files(tmp_path, capsys):
    man = tmp_path / "x1.tsv"
    rows = [("X1", "immune", "t_disc", "discriminator"), ("X1", "immune", "t_sink", "sinkhorn"), ("X1", "immune", "t_new", "mmd"),
            ("X1", "lung", "t_lung", "mmd"), ("A1", "immune", "t_a1", "mmd"), ("X1", "immune", "t_killed", "mmd")]
    man.write_text("# test\nexperiment\ttask\ttag\tarm\n" + "".join("\t".join(r) + "\n" for r in rows))
    old, new = tmp_path / "old", tmp_path / "new"
    for tag in ("t_disc", "t_sink", "t_lung", "t_a1"):                       # t_new has no v2 output; t_lung is not in --tasks
        for d, name in (("latents", f"{tag}.npz"), ("scores", f"{tag}.csv")):
            (old / d).mkdir(parents=True, exist_ok=True)
            (old / d / name).write_text("x")
    (old / "models" / "t_disc").mkdir(parents=True)
    (old / "models" / "t_killed").mkdir()                                    # killed fit: model and score saved, latent never moved in
    (old / "scores" / "t_killed.csv").write_text("x")
    args = ["--manifest", str(man), "--tasks", "immune", "--old-dir", str(old), "--new-dir", str(new)]
    assert L.main(args) == 0
    assert "1 with reusable v2 outputs" in (out := capsys.readouterr().out) and "1 left unlinked (no latent)" in out
    assert not any(os.path.lexists(new / k / "t_killed") for k in ("latents", "models")) and not os.path.lexists(new / "scores" / "t_killed.csv")
    assert sorted(os.listdir(new / "scores")) == ["t_disc.csv"] and os.path.islink(new / "models" / "t_disc")
    assert os.path.realpath(new / "latents" / "t_disc.npz") == str((old / "latents" / "t_disc.npz").resolve())
    with pytest.raises(SystemExit, match="exists"):
        L.main(args)
    with pytest.raises(SystemExit, match="no X1 rows"):
        L.main(["--manifest", str(man), "--tasks", "nope", "--old-dir", str(old), "--new-dir", str(tmp_path / "n2")])
    with pytest.raises(SystemExit, match="are required"):
        L.main(["--new-dir", str(new)])
    os.remove(new / "scores" / "t_disc.csv")
    (new / "scores" / "t_disc.csv").write_text("real")                       # a real file is never removed
    with pytest.raises(SystemExit, match="not a link"):
        L.main(["--new-dir", str(new), "--unlink", "t_disc"])
    os.remove(new / "scores" / "t_disc.csv")
    assert L.main(["--new-dir", str(new), "--unlink", "t_disc"]) == 0 and not os.path.lexists(new / "models" / "t_disc")
    assert (old / "scores" / "t_disc.csv").read_text() == "x"                  # the old folder is untouched
