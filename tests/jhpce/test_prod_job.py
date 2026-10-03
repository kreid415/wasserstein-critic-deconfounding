"""Simulation tests of the JHPCE production job (cluster/jhpce/prod_fit_job.sh, prod_helpers.py, harvest_local.py).

No Slurm, GPU or cluster: each test builds a throwaway git repo holding the real job script, helpers and env.sh plus
stubs (run_stage.py, env_versions.py, fingerprint_prepped.py, tests/scvi) and puts fake nvidia-smi / sacct / squeue on
PATH. Checks the refusals (exit codes, runner never started), the stop guard and resume, the claims and concurrency
guards, the harvest pack and the local verify-and-merge, including a tampered part and conflicting files.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = ["taskA", "taskB"]
TAGS = [f"X1_{t}_none_l0_c1_s{s}_h{s}{t[-1]}" for t in TASKS for s in range(3)]

STUB_RUNNER = r'''
import argparse, json, os, signal, sys, time
import numpy as np
ap = argparse.ArgumentParser()
for k in ("--manifest", "--out-dir", "--prepped-dir", "--fit-python", "--wcd-src", "--expect-device", "--tags-file",
          "--fit-lanes", "--fit-threads", "--fit-timeout-s", "--max-attempts"):
    ap.add_argument(k)
ap.add_argument("--experiments", nargs="+"); ap.add_argument("--tasks", nargs="+")
for k in ("--no-score", "--dry-run", "--clear-foreign-claims", "--clear-stale-claims"):
    ap.add_argument(k, action="store_true")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, "stub_calls.jsonl"), "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
tags = [l.split("#")[0].strip() for l in open(a.tags_file) if l.split("#")[0].strip()]
key = "+".join(sorted(a.experiments)) + "__" + "+".join(sorted(a.tasks))
print(f"[stage] {key}: {len(tags)} rows {{'needs_fit': {len(tags)}}}; to run: {len(tags)} fits (no scoring); out {a.out_dir}", flush=True)
if a.dry_run:
    sys.exit(0)
stop = []
signal.signal(signal.SIGTERM, lambda s, f: stop.append(s))
for d in ("latents", "attempts", "ledger", "logs/fit"):
    os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)
done = []
for t in tags:
    if os.path.exists(os.path.join(a.out_dir, "latents", f"{t}.npz")):
        done.append(t); continue
    t_end = time.time() + float(os.environ.get("FAKE_FIT_S", "0"))
    while time.time() < t_end and not stop:
        time.sleep(0.05)
    if stop:
        break
    np.savez(os.path.join(a.out_dir, "latents", f"{t}.npz"), z=np.ones((5, 2)) * len(t))
    os.makedirs(os.path.join(a.out_dir, "models", t), exist_ok=True)
    open(os.path.join(a.out_dir, "models", t, "model.pt"), "w").write("m" + t)
    open(os.path.join(a.out_dir, "attempts", f"{t}.jsonl"), "a").write(json.dumps(dict(event="end", tag=t)) + "\n")
    open(os.path.join(a.out_dir, "logs", "fit", f"{t}.a1.log"), "w").write("fit " + t)
    done.append(t)
counts = {"fitted": len(done), "pending": len(tags) - len(done)}
json.dump(dict(counts=counts, gate=dict(ok=not stop and len(done) == len(tags))),
          open(os.path.join(a.out_dir, "ledger", f"{key}.json"), "w"))
open(os.path.join(a.out_dir, "ledger", f"{key}.csv"), "w").write("tag,task,state\n" + "".join(
    f"{t},{t.split('_')[1]},{'fitted' if t in done else 'pending'}\n" for t in tags))
if stop:
    print("[stage] interrupted; exit 143", flush=True)
    sys.exit(143)
rc = int(os.environ.get("FAKE_RUNNER_RC", "0"))
print(f"[gate] {'PASS' if rc == 0 else 'FAIL'}", flush=True)
sys.exit(rc)
'''

STUB_ENV_VERSIONS = r'''
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--kind"); ap.add_argument("--out"); a = ap.parse_args()
rec = dict(kind=a.kind, python="3.11.15", torch_cuda="12.6", cudnn=91002, cuda_available=True,
           gpu=os.environ.get("FAKE_TORCH_GPU", "NVIDIA L40S"), modules=dict(torch="2.13.0+cu126", scvi="1.4.2"))
json.dump(rec, open(a.out, "w"))
'''

FAKE_SLURM = r'''#!/usr/bin/env python3
import json, os, sys
states = json.load(open(os.environ["FAKE_SLURM_STATES"])) if os.environ.get("FAKE_SLURM_STATES") else {}
args = sys.argv[1:]
if os.path.basename(sys.argv[0]) == "squeue" and "%L" in args:
    print(os.environ.get("FAKE_TIME_LEFT", "23:00:00")); sys.exit(0)
st = states.get(args[args.index("-j") + 1])
if st:
    print(st)
'''


def write(path, text, mode=None):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


GIT_ENV = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")


class Sim:
    def __init__(self, tmp):
        self.tmp, self.repo = tmp, tmp / "repo"
        for rel in ("cluster/jhpce/prod_fit_job.sh", "cluster/jhpce/prod_helpers.py", "cluster/jhpce/env.sh",
                    "cluster/jhpce/harvest_local.py"):
            os.makedirs(os.path.dirname(self.repo / rel), exist_ok=True)
            shutil.copy(os.path.join(ROOT, rel), self.repo / rel)
        write(self.repo / "scripts/run_stage.py", STUB_RUNNER)
        write(self.repo / "cluster/jhpce/env_versions.py", STUB_ENV_VERSIONS)
        write(self.repo / "scripts/fingerprint_prepped.py", "import sys, os\nsys.exit(int(os.environ.get('FAKE_FP_RC', '0')))\n")
        write(self.repo / "docs/prepped_fingerprints_scib.json", "{}")
        write(self.repo / "cluster/jhpce/expected_fit_versions.json", json.dumps(dict(
            python="3.11.15", torch_cuda="12.6", cudnn=91002, gpu_contains="L40S", modules=dict(torch="2.13.0+cu126", scvi="1.4.2"))))
        write(self.repo / "tests/scvi/test_ok.py", "def test_ok():\n    assert True\n")
        rows = ["tag\texperiment\ttask\tarm"] + [f"{t}\tX1\t{t.split('_')[1]}\tnone" for t in TAGS]
        write(self.repo / "manifests/m.tsv", "# test manifest\n" + "\n".join(rows) + "\n")
        write(self.repo / "tags/job.tags", "# job tags\n" + "\n".join(TAGS) + "\n")
        self.commit("sim", init=True)
        self.scratch, home, fakebin = tmp / "scratch", tmp / "home", tmp / "bin"
        fitenv = self.scratch / "conda/envs/wcd-fit"
        os.makedirs(fitenv / "bin")
        os.symlink(sys.executable, fitenv / "bin/python")
        write(fitenv / ".verified", "ok\n")
        os.makedirs(self.scratch / "prepped_scib")
        os.makedirs(home)
        write(fakebin / "nvidia-smi", "#!/bin/sh\nprintf '%b' \"${FAKE_GPUS:-NVIDIA L40S, 555.42.06, 46068 MiB, GPU-x\\n}\"\n", 0o755)
        write(fakebin / "sacct", FAKE_SLURM, 0o755)
        write(fakebin / "squeue", FAKE_SLURM, 0o755)
        self.states = tmp / "slurm_states.json"
        write(self.states, "{}")
        self.env = dict(os.environ, SLURM_JOB_ID="4242", SLURM_CPUS_PER_TASK="10", WCD_SCRATCH=str(self.scratch),
                        HOME=str(home), PATH=f"{fakebin}:{os.environ['PATH']}", FAKE_SLURM_STATES=str(self.states),
                        OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        self.out = self.scratch / "tier12" / "st"

    def commit(self, msg, init=False):
        env = dict(os.environ, **GIT_ENV)
        cmds = ([["git", "init", "-q"]] if init else []) + [["git", "add", "-A"], ["git", "commit", "-q", "-m", msg]]
        for c in cmds:
            subprocess.run(c, cwd=self.repo, check=True, env=env)
        self.head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, check=True, capture_output=True,
                                   text=True).stdout.strip()

    def args(self, **over):
        d = {"--expected-sha": self.head, "--stage": "st", "--tags-file": "tags/job.tags",
             "--tags-sha256": sha(self.repo / "tags/job.tags"), "--manifest": "manifests/m.tsv",
             "--manifest-sha256": sha(self.repo / "manifests/m.tsv"), "--experiments": "X1", "--tasks": " ".join(TASKS),
             "--guard-margin-s": "5", "--time-left-s": "100000"}
        d.update(over)
        return [x for kv in d.items() for x in kv]

    def run(self, extra_env=None, **over):
        return subprocess.run(["bash", str(self.repo / "cluster/jhpce/prod_fit_job.sh")] + self.args(**over),
                              env=dict(self.env, **(extra_env or {})), capture_output=True, text=True, timeout=300)

    def set_states(self, d):
        write(self.states, json.dumps(d))

    def calls(self):
        p = self.out / "stub_calls.jsonl"
        return [json.loads(l) for l in open(p)] if p.exists() else []

    def jobdir(self, jid="4242"):
        return self.scratch / "prod" / f"job_{jid}"

    def harvest(self, jid="4242"):
        return self.scratch / "harvest" / "st" / f"job_{jid}"

    def harvest_local(self, dest, jid="4242"):
        return subprocess.run([sys.executable, str(self.repo / "cluster/jhpce/harvest_local.py"), "--parts-dir",
                               str(self.harvest(jid)), "--dest", str(dest)], capture_output=True, text=True)


@pytest.fixture
def sim(tmp_path):
    return Sim(tmp_path)


def test_happy_path_packs_and_merges(sim, tmp_path):
    r = sim.run()
    assert r.returncode == 0, r.stdout + r.stderr
    real = [c for c in sim.calls() if "--dry-run" not in c]
    assert len(real) == 1 and "--tags-file" in real[0] and real[0][real[0].index("--expect-device") + 1] == "L40S"
    assert "--no-score" in real[0] and "--clear-foreign-claims" not in real[0]
    assert any("--dry-run" in c for c in sim.calls())
    h = json.load(open(sim.jobdir() / "harvest.json"))
    assert h["n_latents"] == len(TAGS) and h["parts"]
    summ = json.load(open(sim.jobdir() / "logs/job_summary.json"))
    assert summ["runner_exit"] == 0 and summ["counts"]["fitted"] == len(TAGS) and summ["stopped_by"] is None
    assert not [d for d in os.listdir(sim.scratch / "prod/slots") if d.startswith("slot")]   # released on exit
    dest = tmp_path / "durable" / "jhpce_tier12" / "st"
    m = sim.harvest_local(dest)
    assert m.returncode == 0, m.stderr
    assert sorted(os.listdir(dest / "latents")) == sorted(f"{t}.npz" for t in TAGS)
    assert (dest / "_harvests" / "st__job4242" / "receipt.json").exists()
    assert not (dest / "ledger").exists()                     # views kept under _harvests, rebuilt locally
    m2 = sim.harvest_local(dest)                              # the same harvest again: everything skipped
    assert m2.returncode == 0 and json.loads(m2.stdout)["skip"] == json.loads(m.stdout)["copy"]


@pytest.mark.parametrize("over,code", [({"--expected-sha": "0" * 40}, 3), ({"--tags-sha256": "f" * 64}, 8),
                                       ({"--tasks": "taskA"}, 8), ({"--expected-sha": "abc"}, 9)])
def test_refusals_before_any_fit(sim, over, code):
    r = sim.run(**over)
    assert r.returncode == code, r.stdout + r.stderr
    assert sim.calls() == []


def test_dirty_tree_refused(sim):
    with open(sim.repo / "scripts/run_stage.py", "a") as f:
        f.write("# local edit\n")
    r = sim.run()
    assert r.returncode == 3 and sim.calls() == []


@pytest.mark.parametrize("gpus", ["NVIDIA A100 80GB PCIe, 555.42.06, 81920 MiB, GPU-a\\n",
                                  "NVIDIA L40S, 555, 1, a\\nNVIDIA L40S, 555, 1, b\\n"])
def test_node_witness_refuses_wrong_gpu(sim, gpus):
    r = sim.run(extra_env={"FAKE_GPUS": gpus})
    assert r.returncode == 2 and sim.calls() == []


def test_failing_node_tests_abort(sim):
    with open(sim.repo / "tests/scvi/test_ok.py", "a") as f:
        f.write("\ndef test_bad():\n    assert False\n")
    sim.commit("bad test")
    r = sim.run()
    assert r.returncode == 2, r.stdout + r.stderr
    assert sim.calls() == []


def test_fit_env_version_mismatch_aborts(sim):
    r = sim.run(extra_env={"FAKE_TORCH_GPU": "NVIDIA A100"})
    assert r.returncode == 2 and sim.calls() == []


def test_prepped_fingerprint_mismatch_aborts(sim):
    r = sim.run(extra_env={"FAKE_FP_RC": "1"})
    assert r.returncode == 8 and sim.calls() == []


def test_environment_refusals(sim):
    assert sim.run(extra_env={"SLURM_JOB_ID": ""}).returncode == 9
    os.remove(sim.scratch / "conda/envs/wcd-fit/.verified")
    assert sim.run().returncode == 9
    assert sim.calls() == []


def test_scratch_under_home_refused(sim):
    r = sim.run(extra_env={"HOME": str(sim.tmp)})
    assert r.returncode == 9 and sim.calls() == []


def test_stop_guard_terminates_runner_harvests_and_resumes(sim):
    # deadline = now + time_left - margin = now + 8 s; each stub fit takes 4 s, so the guard fires mid-stage
    r = sim.run(extra_env={"FAKE_FIT_S": "4"}, **{"--time-left-s": "1808", "--guard-margin-s": "1800", "--min-run-s": "0"})
    assert r.returncode == 143, r.stdout + r.stderr
    summ = json.load(open(sim.jobdir() / "logs/job_summary.json"))
    assert summ["stopped_by"] == "deadline_guard" and summ["runner_exit"] == 143
    assert 0 < summ["counts"]["fitted"] < len(TAGS)
    assert json.load(open(sim.jobdir() / "harvest.json"))["n_latents"] == summ["counts"]["fitted"]
    r2 = sim.run(extra_env={"SLURM_JOB_ID": "4343"})          # resume: fitted rows kept, the rest fitted
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert json.load(open(sim.jobdir("4343") / "logs/job_summary.json"))["counts"]["fitted"] == len(TAGS)


def test_deadline_too_close_refused(sim):
    r = sim.run(**{"--time-left-s": "1000", "--guard-margin-s": "900"})     # < margin + 30 min minimum run
    assert r.returncode == 9 and [c for c in sim.calls() if "--dry-run" not in c] == []


def test_runner_infrastructure_stop_still_harvests(sim):
    r = sim.run(extra_env={"FAKE_RUNNER_RC": "5"})
    assert r.returncode == 5
    assert json.load(open(sim.jobdir() / "logs/job_summary.json"))["runner_exit"] == 5
    assert json.load(open(sim.jobdir() / "harvest.json"))["parts"]


def test_concurrency_slots(sim):
    slots = sim.scratch / "prod/slots"
    for i, (jid, tasks) in enumerate([("111", ["x"]), ("222", ["y"])], 1):
        write(slots / f"slot{i}" / "owner.json", json.dumps(dict(slurm_job_id=jid, tasks=tasks)))
    sim.set_states({"111": "RUNNING", "222": "RUNNING"})
    assert sim.run().returncode == 7 and sim.calls() == []
    sim.set_states({"111": "COMPLETED", "222": "RUNNING"})    # owner job ended: its slot is reclaimed
    r = sim.run()
    assert r.returncode == 0, r.stdout + r.stderr


def test_overlapping_live_job_refused(sim):
    write(sim.scratch / "prod/slots/slot1/owner.json", json.dumps(dict(slurm_job_id="111", tasks=["taskA"])))
    sim.set_states({"111": "RUNNING"})
    assert sim.run().returncode == 7 and sim.calls() == []


def test_unknown_slurm_state_counts_as_live(sim):
    for i in (1, 2):
        write(sim.scratch / "prod/slots" / f"slot{i}" / "owner.json", json.dumps(dict(slurm_job_id=str(900 + i), tasks=["z"])))
    sim.set_states({})                                        # sacct and squeue know nothing: fail safe
    assert sim.run().returncode == 7


@pytest.mark.parametrize("state,code,cleared", [("RUNNING", 7, False), ("COMPLETED", 0, True), ("TIMEOUT", 0, True)])
def test_claims_of_earlier_jobs(sim, state, code, cleared):
    write(sim.out / "claims" / TAGS[0] / "owner.json", json.dumps(dict(slurm_job_id="333", host="compute-171")))
    sim.set_states({"333": state})
    r = sim.run()
    assert r.returncode == code, r.stdout + r.stderr
    real = [c for c in sim.calls() if "--dry-run" not in c]
    if cleared:
        assert "--clear-foreign-claims" in real[0] and "--clear-stale-claims" in real[0]
        assert not any("--dry-run" in c for c in sim.calls())  # dry run skipped while claims are present
    else:
        assert real == []


def test_claim_without_slurm_id_refused(sim):
    write(sim.out / "claims" / TAGS[0] / "owner.json", json.dumps(dict(host="laptop")))
    assert sim.run().returncode == 7 and sim.calls() == []


def test_harvest_local_rejects_tampered_part(sim, tmp_path):
    assert sim.run().returncode == 0
    hv = sim.harvest()
    part = sorted(p for p in os.listdir(hv) if ".part" in p)[0]
    with open(hv / part, "r+b") as f:
        f.seek(100)
        f.write(b"X")
    dest = tmp_path / "durable" / "out"
    r = sim.harvest_local(dest)
    assert r.returncode == 1 and "SHA-256 mismatch" in r.stderr and not dest.exists()


def test_harvest_local_refuses_tier12_and_conflicts(sim, tmp_path):
    assert sim.run().returncode == 0
    r = sim.harvest_local(tmp_path / "proj" / "tier12" / "A1")
    assert r.returncode == 1 and "tier12" in r.stderr
    dest = tmp_path / "durable" / "out"
    write(dest / "latents" / f"{TAGS[0]}.npz", "different bytes")
    r = sim.harvest_local(dest)
    assert r.returncode == 1 and "nothing merged" in r.stderr
    assert sorted(os.listdir(dest / "latents")) == [f"{TAGS[0]}.npz"]      # all-or-nothing


def test_make_tags_committed_files_match_manifest():
    r = subprocess.run([sys.executable, "cluster/jhpce/make_tags.py", "--manifest",
                        "manifests/paper_manifest_stock_pilot_u5_b10.tsv", "--out-dir", "cluster/jhpce/tags", "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout[: r.stdout.index("\n}") + 2])
    assert rec["rows"] == 250 and rec["per_job"] == {"jobA": 100, "jobB": 150}
