"""Planted errors of the 2026-10-15 round (analysis fixes AN-R06 to AN-R09, AN-R13, AN-R14 and SI-83), each an exact text
replacement in scripts/analysis_x1.py. O-series and N11: the independent re-check's planted errors (evidence archive
4fc04d2a-1c2e-481c-bb7e-5157820f755f, scripts/own_mut.py, own_mut2.py) that the tests of 2026-10-14 did not catch; M24: the
first review's grid-size error (mutants_review.py). F-series: the analogues of O07 to O10 for the zero-spread floor (SI-83).
G-series: errors in the code added on 2026-10-15. Each tuple: (id, description, [(old text, new text), ...]).
"""
ACCEPTANCE = [
    ("O02", "grid floor: TAU_FLOOR 1e-15 -> 1e-9", [("TAU_FLOOR = 1e-15", "TAU_FLOOR = 1e-9")]),
    ("O11", "R4 recompute tolerance 1e-9 -> 1e-3",
     [('all(abs(got[k] - want[k]) <= P.TIE_TOL for k in ("b0", "b_common", "bstar"))',
       'all(abs(got[k] - want[k]) <= 1e-3 for k in ("b0", "b_common", "bstar"))')]),
    ("O12", "R4 recompute: b_common not compared", [('for k in ("b0", "b_common", "bstar"))', 'for k in ("b0", "bstar"))')]),
    ("O16", "sort key: rows sorted by lambda instead of tag", [('.sort_values("tag", ignore_index=True)', '.sort_values("lam", ignore_index=True)')]),
    ("O18", "failed-run total: max instead of sum", [("failed_runs=sum(failed.get(arm, [])),", "failed_runs=max(failed.get(arm, []) or [0]),")]),
    ("O20", "gap flag: looks only at the first interior point",
     [("gap=any(min(lams) < lam < max(lams) and lam not in lams for lam in means),",
       "gap=(len(lams) > 1 and sorted(means).index(lams[1]) - sorted(means).index(lams[0]) > 1),")]),
    ("O22", "grid point count 201 -> 151", [("GRID_N = 201", "GRID_N = 151")]),
    ("O23", "sign table: t degrees of freedom a constant 4 (right only when m = 5)",
     [('q = stats.t.ppf(0.975, reading["m"] - 1)', 'q = stats.t.ppf(0.975, 4)')]),
    ("O24", "R4 recompute ignores the extension rows (computed on the six registered points only)",
     [('targets = P.r4_stage(out, "x1", r1)["targets"]',
       'reg = {(a, c): r1["families"][f"{P.FAMILY_OF[a]}|{c}"]["x1_grid"] for a in P.ADV_ARMS for c in (0, 1)}\n'
       '    keep = [r.experiment != "X1" or r.arm not in P.ADV_ARMS or any(math.isclose(r.lamf, g, rel_tol=1e-9) for g in reg[(r.arm, r.cond_i)]) for r in out.itertuples()]\n'
       '    targets = P.r4_stage(out[keep], "x1", r1)["targets"]')]),
    ("M24", "GRID_N 201 -> 101", [("GRID_N = 201", "GRID_N = 101")]),
    ("N11", "input rows not sorted by tag", [('.sort_values("tag", ignore_index=True)', '')]),
    ("F07", "floor: '<' changed to '<=' at the threshold (s = 1e-6 listed as raised)", [("if x < S_MIN]", "if x <= S_MIN]")]),
    ("F08", "floor: threshold 1.5e-6", [("S_MIN = 1e-6 ", "S_MIN = 1.5e-6 ")]),
    ("F09", "floor applied only when s == 0",
     [("max(x, S_MIN) for x in s]", "(S_MIN if x == 0 else x) for x in s]"), ("if x < S_MIN]", "if x == 0]")]),
    ("F10", "floor not applied to the pair's averaged standard error",
     [('            s, pairs = raise_s(units, s, "pair")      # sqrt(s1^2 + s2^2) / 2 can fall below S_MIN even after each s is raised\n'
       '            raised += pairs\n', '')]),
]
EXTRA = [
    ("G01", "n_all_failed_between counts every failed configuration, not only those between the window's ends",
     [("n_all_failed_between=sum(min(lams) < lam < max(lams) for lam in empty))", "n_all_failed_between=len(empty))")]),
    ("G02", "floor: NaN, infinite and negative standard errors not refused",
     [('P._require(all(math.isfinite(x) and x >= 0 for x in s), "standard errors must be finite and nonnegative")', "pass")]),
    ("G03", "refusal message: recorded and recomputed values not printed",
     [('f"{[got[k] for k in four]}, recomputed {[want[k] for k in four]}; if R4 added extension rows, pass "', 'f"x, recomputed y; if R4 added extension rows, pass "')]),
    ("G04", "pairs: the family's standard error averages the datasets' own s, not the raised ones",
     [('s, raised = raise_s(tasks, s, "dataset")', 's0 = list(s)\n    s, raised = raise_s(tasks, s, "dataset")'),
      ("s = [math.sqrt(sum(s[i] ** 2 for i in g)) / len(g) for g in groups]",
       "s = [math.sqrt(sum(s0[i] ** 2 for i in g)) / len(g) for g in groups]")]),
]
