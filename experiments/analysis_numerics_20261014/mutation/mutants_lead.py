"""The planted errors of 2026-10-14 for scripts/analysis_x1.py, retargeted to the code of 2026-10-15: RETARGET (the reviewer's M16, M26 and
M27, whose code text changed) and NEW (the lead's N-series for the 2026-10-14 fixes). N05 (the rule that left a dataset with a very small standard
error out, disabled) is retired: SI-83 removed the rule, and the floor that replaces it is covered by F07-F10 in mutants_r3.py. Each tuple of NEW: (id, description,
[(old text, new text), ...]); the reviewer's 40 are in mutants_review.py."""
RETARGET = {
    "M16": [('q = stats.t.ppf(0.975, reading["m"] - 1)', 'q = 1.0')],
    "M27": [('- (tau ** 2).sum(axis=1) / (2 * tau_scale ** 2)', '- tau.sum(axis=1) / tau_scale')],
    "M26": [("return dict(pool(d, s, levels, scale, scale, margin), units=units, structure=structure, **raised_record(raised))",   # combine also returns `raised`
             "return dict(pool(d, s, levels, scale, 0.1, margin), units=units, structure=structure, **raised_record(raised))")],
}
NEW = [
    ("N01", "revert AN-01: uniform spread grid from 0, no Jacobian",
     [("tau = np.geomspace(TAU_FLOOR, GRID_SPAN * tau_scale, GRID_N)[indices]", "tau = np.linspace(0, GRID_SPAN * tau_scale, GRID_N)[indices]"),
      ("log_weight += np.log(tau).sum(axis=1) - (tau", "log_weight += - (tau")]),
    ("N02", "log-grid Jacobian (weight x tau) dropped", [("log_weight += np.log(tau).sum(axis=1) - (tau", "log_weight += - (tau")]),
    ("N03", "TAU_FLOOR 1e-15 -> 1e-5", [("TAU_FLOOR = 1e-15", "TAU_FLOOR = 1e-5")]),
    ("N04", "Delta posterior from V^-1 again (cancellation-prone)",
     [("    delta_var = delta2 / (1 + delta2 * x[:, :, 0].sum(axis=1))\n    delta_mean, delta_sd = delta_var * x[:, :, 1].sum(axis=1), np.sqrt(delta_var)",
       "    delta_mean = delta2 * vd.sum(axis=1)\n    delta_sd = np.sqrt(delta2 - delta2 ** 2 * inverse.sum(axis=(1, 2)))")]),
    ("N06a", "R4 check compares b0 only", [('for k in ("b0", "b_common", "bstar")', 'for k in ("b0",)')]),
    ("N06b", "R4 check ignores the low-signal flag",
     [('P._require(got["bstar_low_signal"] == want["bstar_low_signal"]\n                   and all(', 'P._require(all(')]),
    ("N07", "X1 seed check removed", [("P._require(sorted(zero_runs(out, task, cond)) == X1_SEEDS,", "P._require(True,")]),
    ("N08", "sign table back to the normal quantile", [('q = stats.t.ppf(0.975, reading["m"] - 1)', 'q = 1.959963984540054')]),
    ("N09", "window gap always False", [("gap=any(min(lams) < lam < max(lams) and lam not in lams for lam in means),", "gap=False,")]),
    ("N10", "best-of-k failed runs not counted",
     [('failed.setdefault(arm, []).append(int((config.status == "failed").sum()))', 'failed.setdefault(arm, []).append(0)')]),
    ("N11", "input rows not sorted by tag (expected equivalent at test tolerance)", [('.sort_values("tag", ignore_index=True)', '')]),
    ("N12", "pool accepts any positive s", [("(s >= S_MIN).all()", "(s > 0).all()")]),
    ("N13", "S_MIN 1e-6 -> 1e-9", [("S_MIN = 1e-6 ", "S_MIN = 1e-9 ")]),
    ("N14", "R4 targets taken from the record instead of recomputed", [('targets = P.r4_stage(out, "x1", r1)["targets"]', 'targets = r4["targets"]')]),
    ("N15", "sign-table t with m instead of m - 1 degrees of freedom", [('stats.t.ppf(0.975, reading["m"] - 1)', 'stats.t.ppf(0.975, reading["m"])')]),
    ("N17", "best-of-k configs = scored configurations", [("configs=len(failed.get(arm, [])),", "configs=len(v),")]),
]
