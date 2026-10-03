"""Fit outcomes shared by the training plan (scripts/scvi_adversarial_plan.py), the fitter (scripts/fit_paper_config.py)
and the stage runner (scripts/run_stage.py). docs/PREREG.md section 1, "Fit outcome":

    scored            one score row (scripts/score_scib_native.py)
    diverged          the training loss became non-finite          OUT_DIR/status/<tag>.json, written by the fitter
    nonfinite_latent  the saved posterior mean has a non-finite    OUT_DIR/status/<tag>.json, written by the runner
                      value
    infrastructure errors (out of memory, killed job, I/O) are not outcomes: the fit is rerun and nothing is recorded.

A status file holds exactly one failure outcome of one manifest row and is never overwritten. Standard library only,
so every environment (fit, scoring, a login node) can import it.
"""
import json
import math
import os
import socket
import time

FAILURE_STATUSES = ("diverged", "nonfinite_latent")   # = prereg_rules.FAILURE_STATUSES (checked in tests/stage)
# exit status of fit_paper_config.py after it recorded a divergence; distinct from Python's 1 (uncaught exception),
# argparse's 2 and signal exits (negative in subprocess, 128 + n in a shell)
EXIT_DIVERGED = 23
STATUS_KEYS = ("tag", "status", "detail", "row", "host", "recorded_at")


class NonFiniteLossError(FloatingPointError):
    """A training loss term became non-finite, so the fit diverged (docs/PREREG.md section 1).

    epoch: Lightning's current_epoch; step: 0-based index of the training minibatch over the whole fit;
    terms: {name: value} of the loss terms checked at that step (the non-finite ones are the cause; empty when the
    forward pass itself refused a non-finite distribution parameter); detail: what was checked."""

    def __init__(self, epoch, step, terms, detail=""):
        self.epoch, self.step = int(epoch), int(step)
        self.terms = {str(k): float(v) for k, v in terms.items()}
        self.detail = str(detail)
        bad = {k: v for k, v in self.terms.items() if not math.isfinite(v)}
        msg = f"non-finite training loss at epoch {self.epoch}, step {self.step}"
        if bad:
            msg += ": " + ", ".join(f"{k}={v}" for k, v in bad.items())
        if self.detail:
            msg += f" ({self.detail})"
        super().__init__(msg)


def status_path(out_dir, tag):
    return os.path.join(out_dir, "status", f"{tag}.json")


def _finite_or_text(v):
    """JSON has no inf/NaN: non-finite floats are written as 'inf', '-inf', 'nan'."""
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, dict):
        return {k: _finite_or_text(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite_or_text(x) for x in v]
    return v


def one_line(text, limit=400):
    """A single-line detail for the failures table (the full text goes in the status file's 'error' field)."""
    s = " | ".join(x.strip() for x in str(text).splitlines() if x.strip())
    return s if len(s) <= limit else s[:limit - 3] + "..."


def write_status(out_dir, tag, status, detail, row, **fields):
    """Record the failure outcome of one manifest row. Refuses an unknown status, an empty detail, a row of another
    tag and an existing record (one outcome per tag). Atomic (temporary file + rename)."""
    if status not in FAILURE_STATUSES:
        raise ValueError(f"{tag}: status {status!r} not in {FAILURE_STATUSES} (infrastructure errors are rerun, "
                         f"not recorded)")
    detail = one_line(detail)
    if not detail:
        raise ValueError(f"{tag}: a failure needs a detail")
    if row.get("tag") != tag:
        raise ValueError(f"{tag}: the recorded row belongs to tag {row.get('tag')!r}")
    clash = sorted(set(fields) & set(STATUS_KEYS))
    if clash:
        raise ValueError(f"{tag}: fields {clash} are set by write_status")
    path = status_path(out_dir, tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: {tag} already has a recorded outcome")
    rec = dict(tag=tag, status=status, detail=detail, row=dict(row), host=socket.gethostname(),
               recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_finite_or_text(rec), f, indent=1, allow_nan=False)
    os.replace(tmp, path)
    return path


def read_status(path, tag=None):
    """Load and validate a status file; tag (optional) must match."""
    with open(path) as f:
        rec = json.load(f)
    missing = [k for k in STATUS_KEYS if k not in rec]
    if missing:
        raise ValueError(f"{path}: status record lacks {missing}")
    if rec["status"] not in FAILURE_STATUSES:
        raise ValueError(f"{path}: status {rec['status']!r} not in {FAILURE_STATUSES}")
    if not str(rec["detail"]).strip():
        raise ValueError(f"{path}: empty detail")
    if not isinstance(rec["row"], dict) or rec["row"].get("tag") != rec["tag"]:
        raise ValueError(f"{path}: the recorded row does not belong to tag {rec['tag']!r}")
    if tag is not None and rec["tag"] != tag:
        raise ValueError(f"{path}: record of tag {rec['tag']!r}, expected {tag!r}")
    if os.path.basename(path) != f"{rec['tag']}.json":
        raise ValueError(f"{path}: file name does not match tag {rec['tag']!r}")
    return rec
