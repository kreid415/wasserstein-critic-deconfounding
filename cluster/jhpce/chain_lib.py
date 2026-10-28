"""Shared readers of the stage runner's outputs for the status board and the fit-to-scoring chain.

A latent or score file is judged by scripts/run_stage.py's own `Outputs` class (the same contract the runner's completion
gate uses), loaded from the checkout given as --stage-repo, so there is one definition of "valid". Results are cached by
file size and modification time (hashing a latent is the slow part), so a second pass only reads changed files.
"""
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_mtime(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).astimezone().isoformat(timespec="seconds")


def load_run_stage(repo):
    scripts = os.path.join(os.path.abspath(repo), "scripts")
    path = os.path.join(scripts, "run_stage.py")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found: --stage-repo must be a checkout that contains scripts/run_stage.py")
    sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("run_stage", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_expected(rs, manifest, manifest_sha256, experiments, tags_files=(), tasks=None):
    """Manifest rows (dicts) of the expected units. Fails if the manifest is not the pinned file. With tags files only those
    tags (every tag must be in the manifest and in `experiments`); without, every row of `experiments` x `tasks`."""
    got = sha256_file(manifest)
    if got != manifest_sha256:
        raise SystemExit(f"{manifest}: SHA-256 {got} != pinned {manifest_sha256}")
    tags = None
    if tags_files:
        tags = []
        for f in tags_files:
            tags += rs.read_tags_file(f)[0]
        dup = sorted({t for t in tags if tags.count(t) > 1})
        if dup:
            raise SystemExit(f"tags listed in more than one tags file: {dup[:5]}")
    return rs.select_rows(rs.load_manifest(manifest), list(experiments), tasks, tags).to_dict("records")


def _sig(*paths):
    out = []
    for p in paths:
        try:
            s = os.stat(p)
            out.append([s.st_size, s.st_mtime_ns])
        except FileNotFoundError:
            out.append(None)
    return out


class Checker:
    """fit(row) and score(row): None (nothing written) or (valid, reason, detail) as run-status-board expects.
    fit: valid = the runner's state is fitted (detail: sha256, finished_at, scorable True) or a recorded final failure
    (diverged / nonfinite_latent: valid, scorable False). score: valid = run_stage's score contract for the latent's SHA-256;
    a row whose fit ended in a recorded failure has no score to wait for (valid, scorable False)."""

    def __init__(self, rs, fit_out, score_out, expect_device, cache_path=None):
        seed = rs.kbet_seed_default()
        self.rs, self.cache_path = rs, cache_path
        # the fit judgement ignores score files (a bad score is a scoring problem, not a fit problem)
        fit_cls = type("FitOutputs", (rs.Outputs,), {"score_path": lambda self, tag: os.path.join(self.out, "no-scores", tag)})
        self.fit_o = fit_cls(fit_out, seed, expect_device, False)
        self.score_o = rs.Outputs(score_out, seed, expect_device, False)
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)

    def _cached(self, key, sig, compute):
        hit = self.cache.get(key)
        if hit and hit["sig"] == sig:
            return hit["res"]
        res = compute()
        self.cache[key] = {"sig": sig, "res": res}
        return res

    def fit(self, row):
        tag, o = row["tag"], self.fit_o
        sig = _sig(o.latent_path(tag), o.model_path(tag), o.status_path(tag))
        return self._cached("fit:" + tag, sig, lambda: self._fit(row))

    def _fit(self, row):
        o, rs = self.fit_o, self.rs
        try:
            state, info = o.state(row, False)
        except rs.InvalidOutput as e:
            return [False, str(e)[:300], {}]
        if state == "needs_fit":
            return None
        if state == "fitted":
            return [True, state, {"sha256": info["latent"]["sha256"], "scorable": True,
                                  "finished_at": iso_mtime(o.latent_path(row["tag"]))}]
        if state in rs.fit_outcome.FAILURE_STATUSES:
            return [True, f"final outcome {state}: no latent to score", {"scorable": False, "outcome": state,
                                                                          "finished_at": iso_mtime(o.status_path(row["tag"]))}]
        return [False, f"state {state}", {}]

    def score(self, row):
        fit = self.fit(row)
        if fit is None or not fit[0]:
            return None                          # no valid fit: nothing to score yet (the fit group shows why)
        if not fit[2]["scorable"]:
            return [True, "no score expected: " + fit[1], {"scorable": False, "finished_at": fit[2]["finished_at"]}]
        sha, p = fit[2]["sha256"], self.score_o.score_path(row["tag"])
        res = self._cached("score2:" + row["tag"], _sig(p) + [sha], lambda: self._score(row, sha, p))
        return res

    def _score(self, row, sha, path):
        if not os.path.exists(path):
            return None
        try:
            self.score_o.score(row, sha)
        except self.rs.InvalidOutput as e:
            return [False, str(e)[:300], {}]
        with open(path, newline="") as f:
            row = next(csv.DictReader(f))
        # who scored it: the JHPCE fixed-seed rows say score_site (SI-115); other rows name the CPU of the machine that scored them
        return [True, "scored", {"scorable": True, "finished_at": iso_mtime(path), "scored_on": row.get("score_site") or row.get("cpu_model", "?")}]

    def save(self):
        if self.cache_path:
            tmp = f"{self.cache_path}.tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(self.cache, f)
            os.replace(tmp, self.cache_path)
