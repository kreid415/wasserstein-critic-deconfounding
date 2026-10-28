"""compare_probe.py must pass the committed probe outputs and reject a known-bad one (fail-loud: a verifier needs a negative case)."""
import json, shutil, subprocess, sys
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "experiments" / "lisi_jhpce_investigation_20261028"
SPECS = [("jh_nat_1", "jh", "natural_run1"), ("jh_final_1", "jh", "final_shim_run1"), ("jh_final_2", "jh", "final_shim_run2"),
         ("ws_nat_1", "ws", "natural_run1"), ("ws_nat_2", "ws", "natural_run2"), ("ws_final_1", "ws", "final_shim_run1")]

def run(res, out):
    args = [f"{l}={res}/{d}:{c}" for l, d, c in SPECS]
    return subprocess.run([sys.executable, str(E / "compare_probe.py"), str(out), str(E / "ws_stored_reference_lisi.json"),
                           str(E / "latents.sha256"), *args], capture_output=True, text=True)

def test_compare_probe_accepts_real_outputs_and_rejects_one_changed_bit(tmp_path):
    good = run(E / "results", tmp_path / "ok.csv")
    assert good.returncode == 0, good.stdout + good.stderr
    res = tmp_path / "res"; shutil.copytree(E / "results", res)
    f = sorted((res / "jh").glob("final_shim_run1_*.json"))[0]
    d = json.loads(f.read_text()); d["rows"][0]["ilisi_hex"] = "0x1.0000000000001p-1"; f.write_text(json.dumps(d))
    bad = run(res, tmp_path / "bad.csv")
    assert bad.returncode == 1 and "FAILED" in bad.stdout, bad.stdout + bad.stderr


def test_compare_full_scores_passes_real_rows_and_fails_when_a_shim_row_changes(tmp_path):
    def go(base):
        return subprocess.run([sys.executable, str(E / "compare_full_scores.py"), str(base / "full_scorer"), str(E / "latents.sha256")],
                              capture_output=True, text=True)
    good = go(E / "results")
    assert good.returncode == 0 and "PASS" in good.stdout, good.stdout + good.stderr
    base = tmp_path / "res"; shutil.copytree(E / "results" / "full_scorer", base / "full_scorer")
    f = sorted((base / "full_scorer" / "jh_shim_runA").glob("*.csv"))[0]
    f.write_text(f.read_text().replace(",0.", ",1.", 1))   # changes the first fractional value
    bad = go(base)
    assert bad.returncode == 1 and "FAIL" in bad.stdout, bad.stdout + bad.stderr
