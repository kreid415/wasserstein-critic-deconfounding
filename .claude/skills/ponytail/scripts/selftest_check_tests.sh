#!/usr/bin/env bash
# Self-test for check_tests.py: planted cases in a throwaway git repo. Prints PASS/FAIL per case; exit 1 on any FAIL.
# Usage: bash selftest_check_tests.sh   (needs git, pytest, pytest-cov)
set -u
CT="$(cd "$(dirname "$0")" && pwd)/check_tests.py"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; cd "$T"
mkdir pkg tests
git init -q -b main && git config user.email t@t && git config user.name t
printf '.check_tests/\n' > .gitignore
printf '[tool.pytest.ini_options]\npythonpath = ["."]\n' > pyproject.toml
touch pkg/__init__.py
printf 'def add(a, b):\n    return a + b\n' > pkg/mathx.py
printf 'from pkg.mathx import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n' > tests/test_m.py
git add -A && git commit -qm base
fails=0
run() { OUTP=$(python "$CT" --base main --src pkg "$@" ${CT_EXTRA:-} 2>&1); CODE=$?; OUTP=$(grep -E '^(FAIL|WARN|check_tests)' <<<"$OUTP"); }
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; echo "$OUTP" | sed 's/^/    /'; echo "    exit=$CODE"; fails=$((fails+1)); fi; }

cat >> pkg/mathx.py <<'EOF'


def clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
EOF
printf 'def unused():\n    return 1\n' > pkg/new.py
cat >> tests/test_m.py <<'EOF'


def test_clip_low():
    from pkg.mathx import clip
    assert clip(-1, 0, 5) == 0


def test_add_again():
    assert add(2, 2) == 4


# test-reason: checks float handling, which test_add does not assert
def test_add_float():
    assert add(0.5, 0.25) == 0.75
EOF
run
check "untested changed lines flagged"      '[[ "$OUTP" == *"mathx.py: changed lines never run"* ]]'
check "untested changed branch flagged"     '[[ "$OUTP" == *"mathx.py: changed branches never taken"* ]]'
check "untested new file flagged"           '[[ "$OUTP" == *"FAIL pkg/new.py: "* ]]'
check "duplicate new test warned"           '[[ "$OUTP" == *"test_add_again: covers no line"* ]]'
check "test-reason comment exempts"         '[[ "$OUTP" != *"test_add_float"* ]]'
check "useful new test not warned"          '[[ "$OUTP" != *"test_clip_low:"* ]]'
check "pre-existing test not warned"        '[[ "$OUTP" != *"::test_add:"* ]]'
check "uncovered change exits 1"            '[[ $CODE -eq 1 ]]'

rm pkg/new.py
python - <<'EOF'
p = 'tests/test_m.py'; s = open(p).read()
s = s.replace("def test_add_again():\n    assert add(2, 2) == 4\n", "")
s = s.replace("def test_clip_low():\n    from pkg.mathx import clip\n    assert clip(-1, 0, 5) == 0\n",
              "def test_clip():\n    from pkg.mathx import clip\n    assert [clip(v, 0, 5) for v in (-1, 3, 9)] == [0, 3, 5]\n")
open(p, 'w').write(s)
EOF
run
check "fully tested change passes"          '[[ $CODE -eq 0 && "$OUTP" == *"check_tests: OK"* ]]'

cat >> tests/test_m.py <<'EOF'


class TestMore:
    def test_dup_in_class(self):
        assert add(3, 3) == 6


@pytest.mark.parametrize("v", [1, 2])
def test_dup_param(v):
    assert add(v, 0) == v
EOF
sed -i '1i import pytest' tests/test_m.py
run
check "warning alone exits 0"               '[[ $CODE -eq 0 ]]'
check "class-method duplicate warned"       '[[ "$OUTP" == *"TestMore::test_dup_in_class: covers no line"* ]]'
check "parametrized duplicate warned"       '[[ "$OUTP" == *"test_dup_param: covers no line"* ]]'
run --strict
check "--strict makes warning exit 1"       '[[ $CODE -eq 1 ]]'

printf '\n\ndef test_broken():\n    assert add(1, 1) == 3\n' >> tests/test_m.py
run
check "failing test exits 2"                '[[ $CODE -eq 2 ]]'
OUTP=$(python "$CT" --base nope 2>&1); CODE=$?
check "missing base ref stops"              '[[ $CODE -ne 0 && "$OUTP" == *"not found"* ]]'

echo "selftest: $fails failure(s)"
exit $((fails > 0))
