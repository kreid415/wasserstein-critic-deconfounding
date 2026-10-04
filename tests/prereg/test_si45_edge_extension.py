"""SI-45 (code check CR-11, user decision 2026-10-04): when R4 stage a1 leaves a window neighbour of an A2/A3 cell
without an A1 fit (edge_unresolved), the default half is fitted at that lambda as A1 extension rows (posterior mean,
standardisation off, A1 seeds 100-102, same decoder; prereg_rules.extension_rows) and R2 / R3 then run as written on
3-point windows. Synthetic tables only (no fits, no scoring).

1. The case triggers the rows: a cell whose matched point is the lowest A1 point after both R4 extension rounds gets
   edge_unresolved ['lo'] and one extension spec; extension_rows() copies the cell's A1 rows (3 seeds) at that lambda.
   A cell no A2/A3 row uses is not extended. End to end, freeze_matched_lambda.py --stage a1 freezes (exit 0), writes
   the 3 A1 rows to --extension-manifest and records them; decide_a2_a3.py refuses the cell until the rows are passed
   and scored, then reads 3-point windows on both halves.
2. Nothing changes when the case does not occur: no extension spec, no extension manifest, record field 0
   (tests/prereg/test_cli_e2e.py; tests/prereg/test_rules_unchanged.py compares every output with the previous commit).
"""
import json
import os

import pandas as pd
import pytest

import checks
import synth
from synth import P
import decide_a2_a3
import freeze_matched_lambda

TASK, COND, ARM = checks.SI45_CELL
LOW_EXT = checks.SI45_LOW_EXT


def _b(arm, lam):
    return checks.si45_b(arm, lam)


def edge_world(r):
    """immune / conditioned: the edge world (every seed equal); every other task x decoder: synth.world."""
    if r["task"] == TASK and int(r["cond"]) == COND:
        base = "none" if r["arm"] == "none" else P.BASE_ARM[r["arm"]]
        return "ok", _b(base, r["lam"] if r["arm"] != "none" else 1), 0.70
    return synth.world(r)


def _out(include_new_point):
    return checks.si45_out(P, include_new_point)


def test_r4_a1_flags_the_unresolved_neighbour_and_asks_for_the_default_half():
    checks.check_si45_flagged(P)
    r4 = P.r4_stage(_out(False), "a1", checks.r1_record(P), need={f"{TASK}|{COND}|{ARM}"})
    c = r4["cells"][f"{TASK}|{COND}|{ARM}"]
    assert r4["status"] == "frozen" and abs(r4["targets"][f"{TASK}|{COND}"]["bstar"] - 0.435) < 1e-3
    assert any(f.startswith("edge_unresolved_low") for f in c["flags"])
    for k, cell in r4["cells"].items():             # every other cell is interior: nothing else is extended
        assert k == f"{TASK}|{COND}|{ARM}" or "edge_unresolved" not in cell, (k, cell)


def test_a_cell_no_pilot_row_uses_is_not_extended():
    r4 = P.r4_stage(_out(False), "a1", checks.r1_record(P), need={f"{TASK}|{COND}|reference"})
    c = r4["cells"][f"{TASK}|{COND}|{ARM}"]
    assert "edge_unresolved" not in c and any(f.startswith("edge_not_extended") for f in c["flags"])
    assert P.default_half_extensions(r4) == []


def test_default_half_extensions_needs_a_stage_a1_record():
    with pytest.raises(P.PreregError, match="stage-a1"):
        P.default_half_extensions(dict(rule="R4", stage="x1", cells={}))


def test_extension_rows_copy_the_default_half_of_the_cell():
    M = synth.pilot_rows()
    E = P.extension_rows(M, [dict(experiment="A1", task=TASK, cond=COND, arm=ARM, lam=0.003)])
    tmpl = M[(M.experiment == "A1") & (M.task == TASK) & (M.cond == str(COND)) & (M.arm == ARM) & (M.lam == "0.1")]
    assert len(E) == 3 and set(E.seed) == {"100", "101", "102"} and set(E.lam) == {"0.003"}
    assert set(E.adv_input) == {"mean"} and set(E.zstd) == {"0"} and set(E.experiment) == {"A1"}
    same = [c for c in synth.BPM.COLS if c not in ("tag", "lam")]
    assert E[same].sort_values("seed").reset_index(drop=True).equals(tmpl[same].sort_values("seed").reset_index(drop=True))
    assert not set(E.tag) & set(M.tag) and not E.tag.duplicated().any()


def test_r2_reads_three_point_windows_once_the_rows_are_scored():
    r4 = P.r4_stage(_out(False), "a1", checks.r1_record(P), need={f"{TASK}|{COND}|{ARM}"})
    with pytest.raises(P.PreregError, match="SI-45 default-half rows"):
        decide_a2_a3.cells_for(_out(False), "A2", r4)
    (cell,) = decide_a2_a3.cells_for(_out(True), "A2", r4)
    assert cell["window"] == [0.003, 0.01, 0.03]
    assert len(cell["mean"]["curve"]) == 3 and len(cell["sample"]["curve"]) == 3
    assert cell["mean"]["status"] == "bracketed" and cell["sample"]["status"] == "bracketed"


# ---- end to end through the CLIs -----------------------------------------------------------------------------------
def _setup(tmp):
    M = synth.pilot_rows()
    E = P.extension_rows(M, [dict(experiment="A1", task=TASK, cond=COND, arm=ARM, lam=v) for v in LOW_EXT])
    man, ext = synth.write_tsv(M, os.path.join(tmp, "manifest.tsv")), synth.write_tsv(E, os.path.join(tmp, "r4_ext.tsv"))
    A1 = pd.concat([M[M.experiment == "A1"], E], ignore_index=True)
    S, F = synth.scores(A1, edge_world)
    sp, fp = synth.write_scores(S, F, os.path.join(tmp, "scores"), "a1")
    r1 = os.path.join(tmp, "r1.json")
    with open(r1, "w") as f:
        json.dump(checks.r1_record(P), f)
    return man, ext, sp, fp, r1


def _decide(tmp, mans, scores, fp):
    return decide_a2_a3.main(sum((["--manifest", m] for m in mans), []) + [
        "--scores", *scores, "--failures", fp, "--r4a1-record", os.path.join(tmp, "r4a1.json"),
        "--out-json", os.path.join(tmp, "r23.json"), "--out-manifest", os.path.join(tmp, "m.x1.tsv")])


def test_cli_freezes_writes_the_rows_and_r2_r3_use_them(tmp_path):
    tmp = str(tmp_path)
    man, ext, sp, fp, r1 = _setup(tmp)
    si45 = os.path.join(tmp, "a1_si45.tsv")
    code = freeze_matched_lambda.main(["--stage", "a1", "--manifest", man, "--manifest", ext, "--scores", sp,
                                       "--failures", fp, "--r1-record", r1, "--out-json", os.path.join(tmp, "r4a1.json"),
                                       "--out-manifest", os.path.join(tmp, "m.r4a1.tsv"), "--extension-manifest", si45])
    assert code == P.EXIT_FROZEN
    rec = json.load(open(os.path.join(tmp, "r4a1.json")))
    assert rec["n_default_half_extension_rows"] == 3 and rec["default_half_extension_manifest"] == si45
    assert rec["default_half_extensions"] == [dict(experiment="A1", task=TASK, cond=COND, arm=ARM, lam=0.003)]
    E, _ = P.read_manifest(si45)
    assert len(E) == 3 and set(E.lam) == {"0.003"} and set(E.task) == {TASK} and set(E.arm) == {ARM}
    assert set(E.cond) == {str(COND)} and set(E.seed) == {"100", "101", "102"} and set(E.adv_input) == {"mean"}
    M2, _ = P.read_manifest(os.path.join(tmp, "m.r4a1.tsv"))
    pil = M2[M2.experiment.isin(["A2", "A3"]) & (M2.task == TASK) & (M2.cond == str(COND)) & (M2.arm == ARM)]
    assert len(pil) and set(pil.lam) == {"0.003", "0.01", "0.03"}
    S, F = synth.scores(M2[M2.experiment.isin(["A2", "A3"])], edge_world)
    pp, _ = synth.write_scores(S, F, os.path.join(tmp, "scores"), "a2a3")
    # without the SI-45 rows the cell is refused, naming them
    with pytest.raises(P.PreregError, match="SI-45 default-half rows"):
        _decide(tmp, [os.path.join(tmp, "m.r4a1.tsv")], [sp, pp], fp)
    # passed but not scored: the rows are neither scored nor failed
    with pytest.raises(P.PreregError, match="neither scored nor failed"):
        _decide(tmp, [os.path.join(tmp, "m.r4a1.tsv"), si45], [sp, pp], fp)
    S3, F3 = synth.scores(E, edge_world)
    ep, _ = synth.write_scores(S3, F3, os.path.join(tmp, "scores"), "si45")
    assert _decide(tmp, [os.path.join(tmp, "m.r4a1.tsv"), si45], [sp, pp, ep], fp) == P.EXIT_FROZEN
    r23 = json.load(open(os.path.join(tmp, "r23.json")))
    for rule, n in (("R2", 12), ("R3", 20)):
        cells = {f"{c['task']}|{c['cond']}|{c['arm']}": c for c in r23["decisions"][rule]["cells"]}
        assert len(cells) == n
        c = cells[f"{TASK}|{COND}|{ARM}"]
        default, alt = ("mean", "sample") if rule == "R2" else ("off", "on")
        assert c["window"] == [0.003, 0.01, 0.03] and len(c[default]["curve"]) == 3 and len(c[alt]["curve"]) == 3
