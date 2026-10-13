"""Pilot (diagnostic, not a pre-registered stage): one full-length fit with the branch Sinkhorn solver, recording for
every training step the solver's iterations at the final eps (all three solves) and the loss forward time. Question:
does real training ever approach SINKHORN_MAX_ITER (50,000), which raises? Runs scripts/fit_paper_config.py main()
in-process; the only change is a wrapper around the plan's alignment module after it is loaded.
Env: as fit_paper_config.py (MANIFEST, TAG, PREPPED_DIR, OUT_DIR, WCD_SRC). Appends to OUT_DIR/pilot_steps_<TAG>.csv every
200 steps and at exit (a killed run keeps what it logged).
"""
import atexit
import csv
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import torch  # noqa: E402
import scvi_adversarial_plan as plan  # noqa: E402
import fit_paper_config as fit  # noqa: E402

STEPS = []
LOG = os.path.join(os.environ["OUT_DIR"], f"pilot_steps_{os.environ['TAG']}.csv")
_load_heads = plan._load_wcd_heads


def _instrumented(src_root):
    out = _load_heads(src_root)
    al, n = out[2], [0]
    err, sk = al._marginal_err, al.CRITIC_FREE_LOSSES["sinkhorn"]

    def counted(*a):
        n[0] += 1
        return err(*a)

    def timed(z, b, **kw):
        n[0] = 0
        torch.cuda.synchronize()
        t = time.perf_counter()
        v = sk(z, b, **kw)
        torch.cuda.synchronize()
        STEPS.append((_flush.done + len(STEPS), n[0], round((time.perf_counter() - t) * 1e3, 3)))
        if len(STEPS) == 200:
            _flush()
        return v

    al._marginal_err, al.CRITIC_FREE_LOSSES["sinkhorn"] = counted, timed
    return out


def _flush():
    with open(LOG, "a", newline="") as f:
        csv.writer(f).writerows(STEPS)
    _flush.done += len(STEPS)
    STEPS.clear()


_flush.done = 0
with open(LOG, "w", newline="") as f:
    csv.writer(f).writerow(["step", "iterations_all_solves", "loss_forward_ms"])
plan._load_wcd_heads = _instrumented
atexit.register(_flush)

if __name__ == "__main__":
    from fit_outcome import NonFiniteLossError
    try:
        fit.main()
    except NonFiniteLossError as e:
        sys.exit(fit.record_divergence(os.environ["MANIFEST"], os.environ["OUT_DIR"], os.environ["TAG"], e))
