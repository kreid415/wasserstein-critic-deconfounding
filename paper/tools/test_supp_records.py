"""Tests of paper/tools/make_supp_supp-records.py: the committed tables match the committed facts, and collect + render run end to
end on a synthetic data root (with this repository's git history and committed environment exports)."""
import importlib.util, json, shutil
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("supp_records", HERE / "make_supp_supp-records.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
NOTES = ["paper/supp/note10.tex", "paper/supp/note11.tex"]


def test_committed_tables_match_facts():  # test-reason: golden check that the committed note tables equal what the committed facts give
    assert m.main(["--check"]) == 0


def _fit(d, sub, tag, config, scored=True, site="", cpu="AMD Ryzen 5 3600 6-Core Processor"):
    """A latent with a config record and, if scored, a one-row score file, in runner layout under d/sub."""
    (d / sub / "latents").mkdir(parents=True, exist_ok=True)
    (d / sub / "scores").mkdir(parents=True, exist_ok=True)
    np.savez(d / sub / "latents" / f"{tag}.npz", config=np.array(json.dumps(config)))
    score = d / sub / "scores" / f"{tag}.csv"
    if scored:
        score.write_text("tag,cpu_model,score_site,scorer_git_sha,scorer_versions\n"
                         f'{tag},{cpu},{site},10a1ca5874fb,"{{""scib"": ""1.1.7""}}"\n')
    return score


def test_collect_and_render_synthetic(tmp_path):
    d, root = tmp_path / "data", tmp_path / "repo"
    gpu = lambda g, t: {"git_sha": "10a1ca5874fb", "scvi": "1.4.2", "torch": t, "gpu": g}
    cpu = {"git_sha": "10a1ca5874fb", "cpu_model": "AMD Ryzen 5 3600 6-Core Processor", "versions": {"scanorama": "1.7.4"}}
    rows = [("A1", "sim1", "none", "A1_sim1_none_l0_c1_s100_x", gpu("NVIDIA GeForce RTX 3080", "2.13.0+cu130"), ""),
            ("A1", "lung", "mmd", "A1_lung_mmd_l1_c1_s100_y", gpu("NVIDIA L40S", "2.13.0+cu126"), "jhpce"),
            ("X13", "lung", "harmony", "X13_lung_harmony_l1_c0_s0_z", cpu, "")]
    inp = d / m.SWEEP_DIR / "inputs"
    inp.mkdir(parents=True)
    scores = [_fit(d, "sweep", tag, cfg, site=site) for _, _, _, tag, cfg, site in rows]
    (inp / "manifest.tsv").write_text("# comment line\nexperiment\ttask\tarm\ttag\n" + "".join(f"{e}\t{t}\t{a}\t{g}\n" for e, t, a, g, _, _ in rows))
    (inp / "scores_list.txt").write_text("\n".join(map(str, scores)) + "\n")
    (inp / "latents.tsv").write_text("".join(f"{g}\t{d / 'sweep' / 'latents' / (g + '.npz')}\n" for _, _, _, g, _, _ in rows))
    (inp / "inputs.json").write_text(json.dumps({"baseline_rows_missing": ["X13_missing"]}))
    (d / m.SWEEP_DIR / "ENVIRONMENT_scvi-api.yml").write_text("  - python=3.11.15\n    - pandas==2.3.3\n")
    pilots = [_fit(d, "pilot", tag, gpu("NVIDIA GeForce RTX 3080", "2.13.0+cu130"))
              for tag in ("A1_immune_reference_l1_c0_s100_a", "A2_immune_reference_l1_c0_s100_b", "A3_atac_small_mmd_l1_c1_s101_c")]
    (d / m.PILOT_SCORES).parent.mkdir(parents=True)
    (d / m.PILOT_SCORES).write_text("\n".join(map(str, pilots)) + "\n")
    (d / m.REPL_SCORES[0]).mkdir(parents=True)  # replication: the 895 committed rows, none scored here
    for rel in NOTES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(m.ROOT / rel, root / rel)

    assert m.main(["collect", "--data-root", str(d), "--root", str(root)]) == 0
    facts = json.loads((root / m.FACTS).read_text())
    prov = {(r["group"], r["task"]): r for r in facts["provenance"]}
    assert prov[("sweep", "lung")]["score_where"] == {"AMD Ryzen 5 3600 6-Core Processor (jhpce)": 1}
    assert prov[("harmony", "lung")]["fit_device"] == ["CPU AMD Ryzen 5 3600 6-Core Processor"]
    assert (prov[("input_pilot", "immune")]["n_fits"], prov[("standardisation_pilot", "atac_small")]["n_fits"]) == (1, 1)
    assert ("sweep", "immune") not in prov  # the pilots' A1 rows are the sweep's and are not counted twice
    assert sum(r["n_fits"] for r in facts["provenance"] if r["group"] == "replication") == 895
    assert all(r["n_scored"] == 0 and r["fit_device"] == ["latent not on the workstation"]
               for r in facts["provenance"] if r["group"] == "replication")
    v = facts["versions"]
    assert (v["fitting, workstation"]["torch"], v["fitting, JHPCE"]["torch"], v["scoring"]["scib"]) == ("2.13.0+cu130", "2.13.0+cu126", "1.1.7")
    assert v["fitting, workstation"]["python"] == "3.11.15" and v["sweep analysis"] == {"python": "3.11.15", "pandas": "2.3.3"}
    assert facts["git"]["prereg-tier12-v3"] == {"commit": "10a1ca5", "date": "2026-10-16T17:48:50-04:00", "tag": True}
    assert facts["baseline_rows_missing"] == ["X13_missing"]

    assert m.main(["--root", str(root), "--check"]) == 1  # the copied notes hold the real tables, not the synthetic ones
    assert m.main(["--root", str(root)]) == 0 and m.main(["--root", str(root), "--check"]) == 0
    note11 = (root / NOTES[1]).read_text()
    assert r"Sweep & lung & 1/1 & L40S & AMD Ryzen 5 3600 (JHPCE) & 10a1ca5 &" in note11
    assert "Replication & all eight & 0/895 & latent not on the workstation & -- & -- & --" in note11
    (root / NOTES[0]).write_text("no generated block")
    with pytest.raises(SystemExit, match="no generated block"):
        m.main(["--root", str(root)])
