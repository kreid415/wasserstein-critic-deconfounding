"""Mutation tests: a perturbed copy of scripts/prereg_rules.py must fail the boundary checks that the real
module passes (fail-loud R11: a verifier must be shown to fail). Each mutation edits one line of source;
the test first asserts the target text occurs exactly once, so a mutation can never be silently inert."""
import os
import types

import pytest

import checks
from synth import P

SRC = os.path.abspath(P.__file__)
MUTANTS = {
    "r4_tie_to_larger_lambda": ("if best is None or d < best_d - TIE_TOL:",
                                "if best is None or d <= best_d + TIE_TOL:", checks.check_r4_tie),
    "rope_not_strict": ("beats_rope = bool(ev) and mean_D - ROPE > TIE_TOL",
                        "beats_rope = bool(ev) and mean_D - ROPE >= -TIE_TOL", checks.check_rope_strict),
    "collapse_boundary_open": ('q["dc"] <= -COLLAPSE_BIO + TIE_TOL', 'q["dc"] < -COLLAPSE_BIO - TIE_TOL',
                               checks.check_collapse_boundary),
    "masking_ignored": ("        if masked:\n            choice, reason = safe_setting",
                        "        if False:\n            choice, reason = safe_setting", checks.check_masking),
    "noise_ignored": ('q["del_b"] = max(DELTA_MIN, K_NOISE * sig_b', 'q["del_b"] = max(DELTA_MIN, 0.0 * sig_b',
                      checks.check_noise_threshold),
    # SI-27 thresholds: >= ceil(2n/3) evaluable cells (8 of 12, 14 of 20); sign per task x decoder
    "coverage_floor": ("need = math.ceil(2 * n / 3)", "need = math.floor(2 * n / 3)", checks.check_coverage_threshold),
    "coverage_half": ("need = math.ceil(2 * n / 3)", "need = math.ceil(n / 2)", checks.check_coverage_threshold),
    "sign_per_task_only": ('    return c["group"]\n', '    return c["task"]\n', checks.check_group_sign),
}


def mutant(old, new):
    src = open(SRC).read()
    assert src.count(old) == 1, f"mutation target occurs {src.count(old)} times: {old!r}"
    mod = types.ModuleType("prereg_rules_mutant")
    mod.__file__ = SRC
    exec(compile(src.replace(old, new), SRC + "<mutant>", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_real_rules_pass_the_check(name):
    MUTANTS[name][2](P)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name):
    old, new, check = MUTANTS[name]
    with pytest.raises(AssertionError):
        check(mutant(old, new))
