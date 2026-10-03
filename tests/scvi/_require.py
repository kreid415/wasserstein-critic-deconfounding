"""Data-dependent tests: skipped (with the reason) when an input env var is missing, but FAILED when
WCD_REQUIRE_DATA=1, so a node check (e.g. on JHPCE) cannot pass vacuously (lead request 2026-10-03)."""
import functools
import importlib
import os

import pytest

REQUIRE_DATA = os.environ.get("WCD_REQUIRE_DATA") == "1"


def needs_env(*names):
    """Decorator: run if every env var in `names` is set; otherwise skip, or fail under WCD_REQUIRE_DATA=1."""
    missing = [n for n in names if not os.environ.get(n)]

    def deco(fn):
        if not missing:
            return fn
        if REQUIRE_DATA:
            @functools.wraps(fn)
            def failing(*args, **kwargs):
                pytest.fail(f"WCD_REQUIRE_DATA=1 but {missing} is not set: this data-dependent test cannot run")
            return failing
        return pytest.mark.skip(reason=f"needs {missing} (prepped scIB h5ad files); WCD_REQUIRE_DATA=1 fails instead")(fn)
    return deco


def import_or_skip(name):
    """pytest.importorskip, except under WCD_REQUIRE_DATA=1, where a missing module is an error."""
    return importlib.import_module(name) if REQUIRE_DATA else pytest.importorskip(name)
