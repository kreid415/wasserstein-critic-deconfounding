"""Tests of the JHPCE production job (cluster/jhpce/prod_fit_job.sh, prod_helpers.py, harvest_local.py) with the REAL
stage runner (scripts/run_stage.py, --tags-file interface of branch fix-runner).

Stubbed: only what needs a GPU node or Slurm. Each test clones a throwaway copy of the working tree (tracked files) in
which env_versions.py (CUDA build / GPU name of the node), tests/scvi (GPU-node tests) and fingerprint_prepped.py (the 8
prepped scIB files) are stubs, and puts fake nvidia-smi / sacct / squeue on PATH. The fit interpreter is a dispatcher:
`fit_paper_config.py` and the runner's torch probe go to tests/stage/fake_env.py (the runner's own fake fitter,
device 'NVIDIA L40S'); everything else (the runner, the helpers, pytest) runs in the real interpreter. Run with
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 python -m pytest -q tests/jhpce
"""
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fit_paper_config as fpc  # noqa: E402  (REQUIRED: the manifest columns the runner and fitter read)

TASKS = ["taskA", "taskB"]
TAGS = [f"X1_{t}_none_l0_c1_s{s}" for t in TASKS for s in range(3)]
UNFILTERED_KEY = "X1__" + "+".join(sorted(TASKS))

STUB_ENV_VERSIONS = r'''
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--kind"); ap.add_argument("--out"); a = ap.parse_args()
exp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_fit_versions.json")))
rec = dict(kind=a.kind, python=exp["python"], torch_cuda=exp["torch_cuda"], cudnn=exp["cudnn"], cuda_available=True,
           gpu=os.environ.get("FAKE_TORCH_GPU", "NVIDIA L40S"), modules=dict(exp["modules"]))
rec["modules"].update(json.loads(os.environ.get("FAKE_MODULES", "{}")))
json.dump(rec, open(a.out, "w"))
'''
STUB_FINGERPRINT = "import os, sys\nsys.exit(int(os.environ.get('FAKE_FP_RC', '0')))\n"

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

# fit_paper_config.py -> tests/stage/fake_env.py, except tags in STUBBORN_TAGS: a fit that ignores SIGTERM (forces the
# job's SIGKILL path; dies with the runner through the runner's PR_SET_PDEATHSIG).
DISPATCH = r'''
import os, signal, sys, time
if os.environ.get("TAG") in set(filter(None, os.environ.get("STUBBORN_TAGS", "").split(","))):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(f"stubborn fit {os.environ['TAG']}", flush=True)
    time.sleep(600)
    sys.exit(1)
os.execv(sys.executable, [sys.executable, FAKE_ENV] + sys.argv[1:])
'''


def write(path, text, mode=None):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def rjson(path):
    with open(path) as f:
        return json.load(f)


GIT_ENV = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env=dict(os.environ, **GIT_ENV)).stdout.strip()


def manifest_rows():
    rows = []
    for t in TAGS:
        r = dict(tag=t, experiment="X1", task=t.split("_")[1], counts="scib", arm="none", lam="0", n_critic="1",
                 adv_input="mean", zstd="0", cond="1", decoder="SCVI", n_latent="10", n_layers="1", n_hidden="128",
                 likelihood="zinb", batch_size="128", max_epochs="2", train_size="0.9", seed=t[-1], reference="auto",
                 extra="{}")
        rows.append(r)
    return pd.DataFrame(rows)[fpc.REQUIRED]


@pytest.fixture(scope="session")
def template(tmp_path_factory):
    """The working tree's tracked files with the GPU-node stubs, a 6-row manifest and its tags file, committed."""
    t = tmp_path_factory.mktemp("template") / "repo"
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout.decode()
    for rel in filter(None, files.split("\0")):
        src = os.path.join(ROOT, rel)
        if rel.startswith("tests/scvi/") or not os.path.isfile(src):
            continue
        os.makedirs(os.path.dirname(t / rel), exist_ok=True)
        shutil.copy2(src, t / rel)
    write(t / "tests/scvi/test_ok.py", "def test_ok():\n    assert True\n")
    write(t / "cluster/jhpce/env_versions.py", STUB_ENV_VERSIONS)
    write(t / "scripts/fingerprint_prepped.py", STUB_FINGERPRINT)
    os.makedirs(t / "manifests", exist_ok=True)
    with open(t / "manifests/m.tsv", "w") as f:
        f.write("# test manifest\n")
        manifest_rows().to_csv(f, sep="\t", index=False)
    write(t / "tags/job.tags", "# job tags\n" + "\n".join(TAGS) + "\n")
    git(t, "init", "-q")
    git(t, "add", "-A")
    git(t, "commit", "-q", "-m", "template")
    return t


class Sim:
    def __init__(self, tmp, template):
        self.tmp, self.repo = tmp, tmp / "repo"
        git(tmp, "clone", "-q", str(template), str(self.repo))
        self.head = git(self.repo, "rev-parse", "HEAD")
        self.scratch, home, fakebin = tmp / "scratch", tmp / "home", tmp / "bin"
        fitenv = self.scratch / "conda/envs/wcd-fit"
        fake_env = self.repo / "tests/stage/fake_env.py"
        write(tmp / "dispatch.py", f"FAKE_ENV = {str(fake_env)!r}\n" + DISPATCH)
        write(fitenv / "bin/python", "\n".join([
            "#!/bin/sh",
            'case "$1" in',
            f'  */scripts/fit_paper_config.py) exec "{sys.executable}" "{tmp / "dispatch.py"}" "$@";;',
            f'  -c) case "$2" in *"import json, torch, scvi"*) exec "{sys.executable}" "{fake_env}" "$@";; esac;;',
            "esac",
            f'exec "{sys.executable}" "$@"', ""]), 0o755)
        write(fitenv / ".verified", "ok\n")
        for task in TASKS:                                   # the runner checks existence; the fake fitter reads nothing
            write(self.scratch / "prepped_scib" / f"{task}__scib.h5ad", "")
        os.makedirs(home)
        write(fakebin / "nvidia-smi", "#!/bin/sh\nprintf '%b' \"${FAKE_GPUS:-NVIDIA L40S, 555.42.06, 46068 MiB, GPU-x\\n}\"\n", 0o755)
        write(fakebin / "sacct", FAKE_SLURM, 0o755)
        write(fakebin / "squeue", FAKE_SLURM, 0o755)
        self.states = tmp / "slurm_states.json"
        write(self.states, "{}")
        self.set_plan({})
        os.makedirs(tmp / "fake_state")
        self.env = dict(os.environ, SLURM_JOB_ID="4242", SLURM_CPUS_PER_TASK="10", WCD_SCRATCH=str(self.scratch),
                        HOME=str(home), PATH=f"{fakebin}:{os.environ['PATH']}", FAKE_SLURM_STATES=str(self.states),
                        OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                        FAKE_DEVICE="NVIDIA L40S", FAKE_PLAN=str(tmp / "plan.json"), FAKE_STATE=str(tmp / "fake_state"),
                        FAKE_SLEEP="600")
        self.out = self.scratch / "tier12" / "st"
        self.tags = self.repo / "tags/job.tags"
        self.key = f"{UNFILTERED_KEY}__tags-{sha(self.tags)[:12]}"       # the runner's documented key (checked below)

    def commit(self, msg):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", msg)
        self.head = git(self.repo, "rev-parse", "HEAD")

    def set_plan(self, plan):
        write(self.tmp / "plan.json", json.dumps(plan))

    def args(self, **over):
        d = {"--expected-sha": self.head, "--stage": "st", "--tags-file": "tags/job.tags",
             "--tags-sha256": sha(self.repo / "tags/job.tags"), "--manifest": "manifests/m.tsv",
             "--manifest-sha256": sha(self.repo / "manifests/m.tsv"), "--experiments": "X1", "--tasks": " ".join(TASKS),
             "--guard-margin-s": "5", "--time-left-s": "100000", "--poll-s": "1"}
        d.update(over)
        flags = [k for k, v in d.items() if v is True]
        return [x for kv in d.items() if kv[1] is not True for x in kv] + flags

    def run(self, extra_env=None, **over):
        return subprocess.run(["bash", str(self.repo / "cluster/jhpce/prod_fit_job.sh")] + self.args(**over),
                              env=dict(self.env, **(extra_env or {})), capture_output=True, text=True, timeout=900)

    def start(self, extra_env=None, **over):
        return subprocess.Popen(["bash", str(self.repo / "cluster/jhpce/prod_fit_job.sh")] + self.args(**over),
                                env=dict(self.env, **(extra_env or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)

    def wait_until(self, cond, timeout=600):
        t_end = time.time() + timeout
        while not cond():
            assert time.time() < t_end, "condition not reached"
            time.sleep(0.5)

    def set_states(self, d):
        write(self.states, json.dumps(d))

    def jobdir(self, jid="4242"):
        return self.scratch / "prod" / f"job_{jid}"

    def logs(self, jid="4242"):
        return self.jobdir(jid) / "logs"

    def summary(self, jid="4242"):
        return rjson(self.logs(jid) / "job_summary.json")

    def runner_started(self, jid="4242"):
        return (self.logs(jid) / "dryrun.log").exists() or (self.logs(jid) / "run_stage.log").exists()

    def harvest(self, jid="4242"):
        return self.scratch / "harvest" / "st" / f"job_{jid}"

    def harvest_local(self, dest, jid="4242"):
        return subprocess.run([sys.executable, str(self.repo / "cluster/jhpce/harvest_local.py"), "--parts-dir",
                               str(self.harvest(jid)), "--dest", str(dest)], capture_output=True, text=True)

    def helper(self, *args):
        return subprocess.run([sys.executable, str(self.repo / "cluster/jhpce/prod_helpers.py"), *map(str, args)],
                              capture_output=True, text=True, env=self.env)

    def fit_starts(self, tag):
        p = self.out / "attempts" / f"{tag}.jsonl"
        recs = [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []
        return sum(1 for x in recs if x.get("kind") == "fit" and x.get("event") == "start")


@pytest.fixture
def sim(tmp_path, template):
    return Sim(tmp_path, template)


def test_happy_path_uses_the_runners_filtered_key(sim, tmp_path):
    r = sim.run()
    assert r.returncode == 0, r.stdout + r.stderr
    dry, run = rjson(sim.logs() / "stage_key.dryrun.json"), rjson(sim.logs() / "stage_key.json")
    assert dry["ok"] and run["ok"] and dry["stage_key"] == run["stage_key"] == sim.key
    assert run["n_rows"] == len(TAGS) and run["n_fits"] == len(TAGS) and run["states"] == {"needs_fit": len(TAGS)}
    led = sim.out / "ledger"
    assert sorted(os.listdir(led)) == sorted(f"{sim.key}.{e}" for e in ("csv", "done", "json"))   # own views only
    assert not any(f.startswith(UNFILTERED_KEY + ".") for f in os.listdir(led))
    L = rjson(led / f"{sim.key}.json")
    assert L["stage"] == sim.key and L["tags_file"]["sha256"] == sha(sim.tags) and L["expect_device"] == "L40S"
    assert L["score"] is False and L["devices"] == ["NVIDIA L40S"] and L["runner_git"]["git_sha"] == sim.head
    s = sim.summary()
    assert s["problems"] == [] and s["runner_exit"] == 0 and s["stopped_by"] is None and s["stage_key"] == sim.key
    assert s["views_final"] and s["gate"]["ok"] and s["counts"] == {"fitted": len(TAGS)}
    assert s["done_marker"]["valid"] and not s["views_rebuilt_by_report_only"]
    assert s["latent_git_shas"] == ["fake"] and s["latent_git_shas_match_head"] is False   # the fake fitter's sha
    assert s["state_by_task"] == {f"{t}/fitted": 3 for t in TASKS}
    h = rjson(sim.jobdir() / "harvest.json")
    assert h["n_latents"] == len(TAGS) and h["stage_key"] == sim.key
    assert sorted(h["ledger_files"]) == sorted(f"out/ledger/{sim.key}.{e}" for e in ("csv", "done", "json"))
    assert not [d for d in os.listdir(sim.scratch / "prod/slots") if d.startswith("slot")]   # released on exit
    dest = tmp_path / "durable" / "jhpce_tier12" / "st"
    m = sim.harvest_local(dest)
    assert m.returncode == 0, m.stderr
    rec = json.loads(m.stdout)
    assert rec["stage_key"] == sim.key and rec["ledger"]["final"] and rec["ledger"]["gate_ok"]
    assert sorted(os.listdir(dest / "latents")) == sorted(f"{t}.npz" for t in TAGS)
    assert (dest / "manifests" / f"{sha(sim.repo / 'manifests/m.tsv')[:16]}.tsv").exists()   # the run's snapshot
    keep = dest / "_harvests" / "st__job4242"
    assert (keep / "receipt.json").exists() and (keep / "out" / "ledger" / f"{sim.key}.json").exists()
    assert not (dest / "ledger").exists()                     # views kept under _harvests, rebuilt locally
    m2 = sim.harvest_local(dest)                              # the same harvest again: everything skipped
    assert m2.returncode == 0 and json.loads(m2.stdout)["skip"] == rec["copy"]


def test_dry_run_mode_plans_without_fitting(sim):
    r = sim.run(**{"--dry-run": True})
    assert r.returncode == 0, r.stdout + r.stderr
    s = sim.summary()
    assert s["mode"] == "dry-run" and s["stage_key"] == sim.key and s["problems"] == []
    assert s["planned"]["n_fits"] == len(TAGS) and s["planned"]["ledger"]["json"].endswith(f"ledger/{sim.key}.json")
    assert not (sim.out / "latents").exists() and not (sim.logs() / "run_stage.log").exists()
    assert not (sim.jobdir() / "harvest.json").exists()
    assert not [d for d in os.listdir(sim.scratch / "prod/slots") if d.startswith("slot")]


@pytest.mark.parametrize("over,code", [({"--expected-sha": "0" * 40}, 3), ({"--tags-sha256": "f" * 64}, 8),
                                       ({"--tasks": "taskA"}, 8), ({"--expected-sha": "abc"}, 9),
                                       ({"--poll-s": "x"}, 9)])
def test_refusals_before_any_fit(sim, over, code):
    r = sim.run(**over)
    assert r.returncode == code, r.stdout + r.stderr
    assert not sim.runner_started()


def test_dirty_tree_refused(sim):
    with open(sim.repo / "scripts/run_stage.py", "a") as f:
        f.write("# local edit\n")
    r = sim.run()
    assert r.returncode == 3 and not sim.runner_started()


@pytest.mark.parametrize("gpus", ["NVIDIA A100 80GB PCIe, 555.42.06, 81920 MiB, GPU-a\\n",
                                  "NVIDIA L40S, 555, 1, a\\nNVIDIA L40S, 555, 1, b\\n"])
def test_node_witness_refuses_wrong_gpu(sim, gpus):
    r = sim.run(extra_env={"FAKE_GPUS": gpus})
    assert r.returncode == 2 and not sim.runner_started()


def test_failing_node_tests_abort(sim):
    with open(sim.repo / "tests/scvi/test_ok.py", "a") as f:
        f.write("\ndef test_bad():\n    assert False\n")
    sim.commit("bad test")
    r = sim.run()
    assert r.returncode == 2, r.stdout + r.stderr
    assert not sim.runner_started()


@pytest.mark.parametrize("env", [{"FAKE_TORCH_GPU": "NVIDIA A100"}, {"FAKE_MODULES": json.dumps({"torch": "2.4.1"})}])
def test_fit_env_version_mismatch_aborts(sim, env):
    r = sim.run(extra_env=env)
    assert r.returncode == 2 and not sim.runner_started()


def test_prepped_fingerprint_mismatch_aborts(sim):
    r = sim.run(extra_env={"FAKE_FP_RC": "1"})
    assert r.returncode == 8 and not sim.runner_started()


def test_environment_refusals(sim):
    assert sim.run(extra_env={"SLURM_JOB_ID": ""}).returncode == 9
    os.remove(sim.scratch / "conda/envs/wcd-fit/.verified")
    assert sim.run().returncode == 9
    assert not sim.runner_started()


def test_scratch_under_home_refused(sim):
    r = sim.run(extra_env={"HOME": str(sim.tmp)})
    assert r.returncode == 9 and not sim.runner_started()


def test_stop_guard_stops_runner_rebuilds_views_and_resumes_on_the_same_key(sim, tmp_path):
    sim.set_plan({t: "sleep" for t in TAGS[2:]})             # 2 fits finish, 4 are still running at the deadline
    r = sim.run(**{"--time-left-s": "79", "--guard-margin-s": "4", "--min-run-s": "1"})    # guard ~75 s after start
    assert r.returncode == 143, r.stdout + r.stderr
    s = sim.summary()
    assert s["stopped_by"] == "deadline_guard" and s["runner_exit"] == 143 and s["stage_key"] == sim.key
    assert s["views_rebuilt_by_report_only"] and s["report_only_exit"] == 6 and s["views_final"]
    assert s["problems"] == [] and not s["gate"]["ok"] and s["counts"]["fitted"] == 2 and s["done_marker"] is None
    assert not (sim.out / "ledger" / f"{sim.key}.done").exists()
    assert not list((sim.out / "claims").glob("*/owner.json"))          # the runner released its claims on SIGTERM
    assert rjson(sim.jobdir() / "harvest.json")["n_latents"] == 2
    sim.set_plan({})
    sim.set_states({"4242": "COMPLETED"})
    r2 = sim.run(extra_env={"SLURM_JOB_ID": "4343"})          # resume at the same commit with the same tags file
    assert r2.returncode == 0, r2.stdout + r2.stderr
    s2 = sim.summary("4343")
    assert s2["stage_key"] == sim.key and s2["gate"]["ok"] and s2["done_marker"]["valid"]
    assert s2["counts"] == {"fitted": len(TAGS)}
    assert [sim.fit_starts(t) for t in TAGS] == [1, 1, 2, 2, 2, 2]   # finished rows kept, interrupted rows rerun
    dest = tmp_path / "durable" / "st"
    for jid in ("4242", "4343"):                              # both harvests merge: identical skipped, attempts grown
        m = sim.harvest_local(dest, jid)
        assert m.returncode == 0, m.stderr
    rec = json.loads(m.stdout)
    assert rec["stage_key"] == sim.key and rec["replace"] == 4 and rec["ledger"]["gate_ok"]
    assert sorted(os.listdir(dest / "latents")) == sorted(f"{t}.npz" for t in TAGS)


def test_runner_killed_views_rebuilt_and_stale_claims_cleared_on_resume(sim):
    stubborn = TAGS[3:]                                      # ignore SIGTERM: the job must SIGKILL the runner
    p = sim.start(extra_env={"STUBBORN_TAGS": ",".join(stubborn)}, **{"--stop-wait-s": "2"})
    run_log = sim.logs() / "run_stage.log"
    sim.wait_until(lambda: run_log.exists() and " rows " in run_log.read_text()
                   and all((sim.out / "latents" / f"{t}.npz").exists() for t in TAGS[:3]))
    p.send_signal(signal.SIGTERM)                           # the job forwards it to the runner (signal path)
    out = p.communicate(timeout=600)[0]
    assert p.returncode == 137, out                          # SIGKILL after --stop-wait-s
    s = sim.summary()
    assert s["stopped_by"] == "signal_TERM" and s["problems"] == [], s
    assert s["views_rebuilt_by_report_only"] and s["views_final"] and s["stage_key"] == sim.key
    assert s["counts"]["fitted"] == 3 and not s["gate"]["ok"]
    claims = sorted(os.listdir(sim.out / "claims"))         # the killed runner's claims remain (it still held the
    assert set(stubborn) <= set(claims)                      # finished rows' claims too: released at its end)
    assert {rjson(sim.out / "claims" / t / "owner.json")["slurm_job_id"] for t in claims} == {"4242"}
    assert s["counts"] == {"fitted": 3, "stale_claim": 3}
    sim.set_states({"4242": "CANCELLED"})
    r2 = sim.run(extra_env={"SLURM_JOB_ID": "4343"})
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert rjson(sim.logs("4343") / "claims.json")["flags"] == ["--clear-foreign-claims", "--clear-stale-claims"]
    assert not (sim.logs("4343") / "dryrun.log").exists()     # dry run skipped while claims are present
    assert "cleared stale claim" in open(sim.logs("4343") / "run_stage.log").read()
    s2 = sim.summary("4343")
    assert s2["stage_key"] == sim.key and s2["gate"]["ok"] and s2["counts"] == {"fitted": len(TAGS)}


def test_guard_before_the_runner_planned_still_rebuilds_views(sim):
    # the guard fires ~1 s after the runner starts: usually before it has planned (no '[stage]' line in the run log);
    # if it has planned, its fits are still sleeping. Either way the views are rebuilt under the runner's own key.
    sim.set_plan({t: "sleep" for t in TAGS})
    r = sim.run(**{"--time-left-s": "6", "--guard-margin-s": "4", "--min-run-s": "1"})
    assert r.returncode == 143, r.stdout + r.stderr
    s = sim.summary()
    assert s["problems"] == [] and s["stage_key"] == sim.key and s["views_rebuilt_by_report_only"], s
    assert s["stage_key_from"].endswith("report_only.log") and s["views_final"] and not s["gate"]["ok"]
    assert rjson(sim.jobdir() / "harvest.json")["stage_key"] == sim.key


def test_deadline_too_close_refused(sim):
    r = sim.run(**{"--time-left-s": "1000", "--guard-margin-s": "900"})     # < margin + 30 min minimum run
    assert r.returncode == 9 and not (sim.logs() / "run_stage.log").exists()


def test_runner_infrastructure_stop_still_harvests(sim):
    sim.set_plan({TAGS[0]: ["crash", "crash"]})
    r = sim.run()
    assert r.returncode == 5, r.stdout + r.stderr
    s = sim.summary()
    assert s["runner_exit"] == 5 and s["stop_reason"] and s["views_final"] and not s["views_rebuilt_by_report_only"]
    h = rjson(sim.jobdir() / "harvest.json")
    assert h["parts"] and h["stage_key"] == sim.key
    assert any(f.startswith(f"logs/fit/{TAGS[0]}.a2") for f in
               (x["path"][4:] for x in rjson(sim.harvest() / "st__job4242.harvest_manifest.json")["files"]))


def test_runner_refusal_attributes_no_ledger(sim):
    # an earlier run's views of this key exist; a run refused in its own preflight (exit 4, no '[stage]' line) must
    # not present them as its own
    assert sim.run().returncode == 0
    os.remove(sim.scratch / "prepped_scib" / "taskB__scib.h5ad")            # the runner's preflight refuses this
    r = sim.run(extra_env={"SLURM_JOB_ID": "4343"})
    assert r.returncode == 4 and not (sim.logs("4343") / "run_stage.log").exists()   # caught by the dry run
    write(sim.out / "claims" / TAGS[0] / "owner.json", json.dumps(dict(slurm_job_id="333", host="compute-171")))
    sim.set_states({"333": "COMPLETED"})                     # claims to clear: the dry run is skipped, the run refuses
    r = sim.run(extra_env={"SLURM_JOB_ID": "4444"})
    assert r.returncode == 4, r.stdout + r.stderr
    assert "REFUSED" in open(sim.logs("4444") / "run_stage.log").read()
    st = rjson(sim.logs("4444") / "stage_key.json")
    assert st["ok"] and st["stage_key"] is None
    s = sim.summary("4444")
    assert s["problems"] == [] and s["stage_key"] is None and "ledger" not in s
    h = rjson(sim.jobdir("4444") / "harvest.json")
    assert h["stage_key"] is None and h["ledger_files"] == []
    assert (sim.out / "ledger" / f"{sim.key}.json").exists()               # job 4242's views are still there


def test_concurrency_slots(sim):
    slots = sim.scratch / "prod/slots"
    for i, (jid, tasks) in enumerate([("111", ["x"]), ("222", ["y"])], 1):
        write(slots / f"slot{i}" / "owner.json", json.dumps(dict(slurm_job_id=jid, tasks=tasks)))
    sim.set_states({"111": "RUNNING", "222": "RUNNING"})
    assert sim.run().returncode == 7 and not sim.runner_started()
    sim.set_states({"111": "COMPLETED", "222": "RUNNING"})    # owner job ended: its slot is reclaimed
    r = sim.run()
    assert r.returncode == 0, r.stdout + r.stderr


def test_overlapping_live_job_refused(sim):
    write(sim.scratch / "prod/slots/slot1/owner.json", json.dumps(dict(slurm_job_id="111", tasks=["taskA"])))
    sim.set_states({"111": "RUNNING"})
    assert sim.run().returncode == 7 and not sim.runner_started()


def test_unknown_slurm_state_counts_as_live(sim):
    for i in (1, 2):
        write(sim.scratch / "prod/slots" / f"slot{i}" / "owner.json", json.dumps(dict(slurm_job_id=str(900 + i), tasks=["z"])))
    sim.set_states({})                                        # sacct and squeue know nothing: fail safe
    assert sim.run().returncode == 7


@pytest.mark.parametrize("state,code,cleared", [("RUNNING", 7, False), ("COMPLETED", 0, True), ("TIMEOUT", 0, True)])
def test_claims_of_earlier_jobs(sim, state, code, cleared):
    write(sim.out / "claims" / TAGS[0] / "owner.json", json.dumps(dict(tag=TAGS[0], slurm_job_id="333", host="compute-171",
                                                                       machine_id="m", pid=1, runner_id="r")))
    sim.set_states({"333": state})
    r = sim.run()
    assert r.returncode == code, r.stdout + r.stderr
    if cleared:
        assert not (sim.logs() / "dryrun.log").exists()       # dry run skipped while claims are present
        assert "cleared foreign claim" in open(sim.logs() / "run_stage.log").read()
        assert sim.summary()["gate"]["ok"]
    else:
        assert not sim.runner_started()


def test_claim_without_slurm_id_refused(sim):
    write(sim.out / "claims" / TAGS[0] / "owner.json", json.dumps(dict(host="laptop")))
    assert sim.run().returncode == 7 and not sim.runner_started()


def test_stagekey_reads_the_runner_line_and_refuses_violations(sim, tmp_path):
    assert sim.run(**{"--dry-run": True}).returncode == 0
    line = [x for x in open(sim.logs() / "dryrun.log") if "[stage] " in x and " rows " in x]
    assert len(line) == 1 and sim.key in line[0]             # the real runner's line (format under test)
    good = line[0]
    base = ["stagekey", "--out-dir", sim.out, "--tags", sim.tags, "--tags-sha256", sha(sim.tags)]

    def check(text, *extra):
        p = tmp_path / "log.txt"
        p.write_text(text)
        return sim.helper(*base, "--log", p, *extra)

    ok = check(good)
    assert ok.returncode == 0 and json.loads(ok.stdout)["ledger"]["done"].endswith(f"ledger/{sim.key}.done")
    bad = {"no line": "[12:00:00] [stage] REFUSED: x\n",
           "unfiltered key": good.replace(sim.key, UNFILTERED_KEY),
           "two keys": good + good.replace(sim.key, sim.key[:-1] + "0"),
           "row count": good.replace(f"{len(TAGS)} rows", f"{len(TAGS) - 1} rows"),
           "other out": good.replace(str(sim.out), str(tmp_path / "elsewhere"))}
    for why, text in bad.items():
        r = check(text)
        assert r.returncode == 11, (why, r.stdout)
    assert check(bad["no line"], "--allow-missing").returncode == 0
    want = tmp_path / "want.json"
    want.write_text(json.dumps(dict(stage_key=sim.key[:-1] + "0")))
    assert check(good, "--expect-key-json", want).returncode == 11


def test_summary_refuses_a_ledger_that_disagrees_with_the_job(sim):
    assert sim.run().returncode == 0
    lp = sim.out / "ledger" / f"{sim.key}.json"
    L = rjson(lp)
    L["tags_file"]["sha256"] = "0" * 64
    write(lp, json.dumps(L))
    r = sim.helper("summary", "--mode", "run", "--stage-key-json", sim.logs() / "stage_key.json", "--tags", sim.tags,
                   "--tags-sha256", sha(sim.tags), "--manifest-sha256", sha(sim.repo / "manifests/m.tsv"),
                   "--head-sha", sim.head, "--rc", "0", "--out", sim.tmp / "s.json")
    assert r.returncode == 11 and "tags_file.sha256" in r.stdout


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
    r = sim.harvest_local(tmp_path / "proj" / "tier12_v2" / "A1")          # the restarted A1 tree (RR-07)
    assert r.returncode == 1 and "tier12" in r.stderr and not (tmp_path / "proj" / "tier12_v2").exists()
    dest = tmp_path / "durable" / "out"
    write(dest / "latents" / f"{TAGS[0]}.npz", "different bytes")
    r = sim.harvest_local(dest)
    assert r.returncode == 1 and "nothing merged" in r.stderr
    assert sorted(os.listdir(dest / "latents")) == [f"{TAGS[0]}.npz"]      # all-or-nothing


def test_harvest_local_refuses_views_not_named_after_the_stage_key(sim, tmp_path):
    assert sim.run().returncode == 0
    st = rjson(sim.logs() / "stage_key.json")
    other = sim.out / "ledger" / f"{UNFILTERED_KEY}.csv"     # a view of another stage, packed by mistake
    shutil.copy(st["ledger"]["csv"], other)
    st["ledger"]["csv"] = str(other)
    write(sim.tmp / "st.json", json.dumps(st))
    dest = sim.tmp / "repack"
    r = sim.helper("pack", "--out-dir", sim.out, "--tags", sim.tags, "--job-dir", sim.jobdir(), "--dest", dest,
                   "--stage", "st", "--job", "4242", "--stage-key-json", sim.tmp / "st.json")
    assert r.returncode == 0, r.stderr
    m = subprocess.run([sys.executable, str(sim.repo / "cluster/jhpce/harvest_local.py"), "--parts-dir", str(dest),
                        "--dest", str(tmp_path / "durable")], capture_output=True, text=True)
    assert m.returncode == 1 and "not named after the recorded stage key" in m.stderr


def test_harvest_local_refuses_a_manifest_without_stage_key_fields(sim):
    spec = importlib.util.spec_from_file_location("harvest_local", sim.repo / "cluster/jhpce/harvest_local.py")
    hl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hl)
    with pytest.raises(SystemExit):
        hl.check_views(str(sim.tmp), ["out/ledger/x.csv"], dict(files=[], tags_sha256="0"))


def _prod_manifest():
    """MANIFEST of cluster/jhpce/prod_command.py: the manifest the production jobs submit."""
    spec = importlib.util.spec_from_file_location("prod_command", os.path.join(ROOT, "cluster", "jhpce", "prod_command.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MANIFEST


def test_make_tags_committed_files_match_manifest():
    man = _prod_manifest()
    assert man == "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"      # SI-44 / SI-46 (2026-10-04)
    r = subprocess.run([sys.executable, "cluster/jhpce/make_tags.py", "--manifest", man, "--out-dir", "cluster/jhpce/tags",
                        "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout[: r.stdout.index("\n}") + 2])
    assert rec["rows"] == 250 and rec["per_job"] == {"jobA": 100, "jobB": 150}
