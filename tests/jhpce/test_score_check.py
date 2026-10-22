"""JHPCE score-only job and its match check: cluster/jhpce/score_plan.py (sizes, lists, sbatch files, time estimate),
score_array.sh (one array task, with the scorer stubbed: no scib, R or JHPCE needed) and compare_scores.py (exact
comparison of two score sets). Standard library only; unittest classes, so both
    python -m pytest -q tests/jhpce/test_score_check.py      and      python -m unittest tests.jhpce.test_score_check
run it."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JH = os.path.join(ROOT, "cluster", "jhpce")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(JH, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(fn, *args):
    """(exit code, stdout) of a script's main(); a SystemExit(2) from die() or the parser is code 2."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
    return code or 0, out.getvalue()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def read(path):
    with open(path) as f:
        return f.read()


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class CompareScores(unittest.TestCase):
    COLS = ["tag", "latent_sha256", "experiment", "task", "arm", "lam", "cond", "seed", "NMI_cluster/label", "kBET",
            "hvg_overlap", "traj_root_fallback", "kbet_r_calls", "scorer_git_sha", "scorer_dirty", "score_host",
            "cpu_model", "cpu_simd", "numba_cpu_name", "scorer_versions", "score_seconds"]

    def row(self, d, name, **over):
        r = dict(tag=name, latent_sha256="ab" * 32, experiment="A1", task="immune", arm="mmd", lam="3", cond="0", seed="100",
                 **{"NMI_cluster/label": "0.5"}, kBET="0.1", hvg_overlap="", traj_root_fallback="", kbet_r_calls="6",
                 scorer_git_sha="c88cce3", scorer_dirty="0", score_host="gimli", cpu_model="Ryzen", cpu_simd="avx2",
                 numba_cpu_name="", scorer_versions='{"numba": "0.66.0"}', score_seconds="180.0")
        r.update(over)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, self.COLS + [c for c in r if c not in self.COLS])
            w.writeheader()
            w.writerow(r)

    def case(self, tmp, cand_over=None, tags=("t1", "t2")):
        """ref in tmp/ref, cand in tmp/cand (cand_over: tag -> overrides), list file in tmp/list.tsv."""
        for t in tags:
            self.row(os.path.join(tmp, "ref"), t)
            self.row(os.path.join(tmp, "cand"), t, **(cand_over or {}).get(t, {}))
        write(os.path.join(tmp, "list.tsv"), "tag\tref_score\n" + "".join(
            f"{t}\t{tmp}/ref/{t}.csv\n" for t in tags))
        return ["--list", os.path.join(tmp, "list.tsv"), "--cand", os.path.join(tmp, "cand")]

    def test_pass_rule_and_failures(self):
        cs = load("compare_scores")
        with tempfile.TemporaryDirectory() as tmp:
            # provenance and timing may differ; PASS, also through --ref (the list's ref_score is then ignored)
            args = self.case(tmp, {"t1": dict(score_host="compute-174", cpu_simd="avx512f", scorer_git_sha="10a1ca5",
                                              score_seconds="900.0")})
            code, out = run(cs.main, args)
            self.assertEqual((code, out.splitlines()[-1]), (0, "PASS"))
            self.assertEqual(run(cs.main, args + ["--ref", os.path.join(tmp, "ref")])[0], 0)
        # FAIL: per-metric count and largest difference, NaN against a value, other package versions
        for over, needle in [({"kBET": "0.1000000005", "cpu_simd": "avx512f"}, "kBET"), ({"kBET": ""}, "kBET"),
                             ({"scorer_versions": '{"numba": "0.67.0"}'}, "PROBLEM t2: scorer_versions"),
                             ({"latent_sha256": "cd" * 32}, "PROBLEM t2: latent_sha256")]:
            with self.subTest(over), tempfile.TemporaryDirectory() as tmp:
                code, out = run(cs.main, self.case(tmp, {"t2": over}))
                self.assertEqual((code, out.splitlines()[-1]), (1, "FAIL"))
                self.assertIn(needle, out)
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(cs.main, self.case(tmp, {"t2": {"kBET": "0.1000000005", "cpu_simd": "avx512f"}}))
            line = next(ln for ln in out.splitlines() if ln.startswith("kBET"))
            self.assertEqual(line.split(), ["kBET", "1", "5e-10", "0"])
            self.assertIn("differs: t2: 1 columns, scored on gimli (avx512f)", out)
            self.assertIn("gimli (avx2): 1 scored, 0 differ; gimli (avx512f): 1 scored, 1 differ", out)
            self.assertEqual(next(ln for ln in run(cs.main, self.case(tmp, {"t2": {"kBET": ""}}))[1].splitlines()
                                  if ln.startswith("kBET")).split(), ["kBET", "1", "0", "1"])

    def test_missing_or_malformed_input_is_exit_2(self):
        cs = load("compare_scores")
        for name, over in [("cand file missing", None), ("two rows", None), ("tag differs", {"t2": {"tag": "t1"}}),
                           ("extra column", {"t2": {"extra": "1"}}), ("non-numeric", {"t2": {"kBET": "high"}}),
                           ("empty list", None)]:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                args = self.case(tmp, over)
                if name == "cand file missing":
                    os.remove(os.path.join(tmp, "cand", "t2.csv"))
                elif name == "two rows":
                    p = os.path.join(tmp, "cand", "t2.csv")
                    text = read(p).splitlines()
                    write(p, "\n".join(text + text[1:]) + "\n")
                elif name == "empty list":
                    write(os.path.join(tmp, "list.tsv"), "tag\tref_score\n")
                self.assertEqual(run(cs.main, args)[0], 2)


class ScorePlan(unittest.TestCase):
    MANIFEST = ("# paper manifest\n"
                "tag\texperiment\ttask\tcounts\tarm\tlam\tn_critic\tcond\tseed\n"
                "A1_sim1_mmd_l3_c0_s100_x\tA1\tsim1\tscib\tmmd\t3\t0\t0\t100\n"
                "A1_sim1_none_l0_c1_s100_y\tA1\tsim1\tscib\tnone\t0\t0\t1\t100\n"
                "A1_atac_small_pooled_l10_c1_s101_z\tA1\tatac_small\tscib\tpooled\t10\t5\t1\t101\n")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.man = write(f"{self.tmp}/m.tsv", self.MANIFEST)
        self.tags = ["A1_sim1_mmd_l3_c0_s100_x", "A1_sim1_none_l0_c1_s100_y", "A1_atac_small_pooled_l10_c1_s101_z"]
        for t in self.tags:
            write(f"{self.tmp}/lat/{t}.npz", "latent " + t)
        for t in ("sim1", "atac_small"):
            write(f"{self.tmp}/pre/{t}__scib.h5ad", t)

    def args(self, tags_text, extra=()):
        write(f"{self.tmp}/tags.tsv", tags_text)
        return ["--manifest", self.man, "--tags", f"{self.tmp}/tags.tsv", "--latents-dir", f"{self.tmp}/lat",
                "--prepped-dir", f"{self.tmp}/pre", "--out-dir", f"{self.tmp}/out", *extra]

    def test_lists_sbatch_files_and_estimate(self):
        sp = load("score_plan")
        want = sha(f"{self.tmp}/lat/{self.tags[0]}.npz")
        code, out = run(sp.main, self.args(f"# c\ntag\tsha256\n{self.tags[0]}\t{want}\tx\n{self.tags[1]}\n{self.tags[2]}\n",
                                           ["--run", "r2", "--cpus", "10"]))
        self.assertEqual(code, 0)
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        lines = read(f"{self.tmp}/out/list_sim1.tsv").splitlines()
        tag, npz, pre, meta = lines[0].split("\t")
        self.assertEqual((len(lines), tag, npz, pre), (2, self.tags[0], f"{self.tmp}/lat/{tag}.npz", f"{self.tmp}/pre/sim1__scib.h5ad"))
        self.assertEqual(json.loads(meta), dict(latent_sha256=want, experiment="A1", task="sim1", arm="mmd", lam="3",
                                                cond="0", seed="100"))
        sb = read(f"{self.tmp}/out/score_sim1.sbatch")
        for needle in ("--partition=shared", "--array=1-2%", "--cpus-per-task=1", "--mem=8G", "--time=00:30:00",
                       f"exec bash {ROOT}/cluster/jhpce/score_array.sh {head} {self.tmp}/out/list_sim1.tsv {self.tmp}/out/r2"):
            self.assertIn(needle, sb)
        self.assertIn("--array=1-1%", read(f"{self.tmp}/out/score_atac_small.sbatch"))
        self.assertIn("compute wall at 10 CPUs", out)
        # known answer: 5 datasets x 378 latents on 400 CPUs (hand-computed from docs/scoring_time.csv, the user's 3,000 s
        # at immune_hum_mou's 97,861 cells and configs/dataset_registry.json)
        code, out = run(sp.main, ["--assume", "immune_hum_mou=378", "atac_large=378", "lung=378", "sim2=378", "pancreas=378"])
        idle, loaded = (ln for ln in out.splitlines() if ln.startswith(("idle", "loaded")))
        self.assertIn("1890 latents, 426 CPU-hours, compute wall at 400 CPUs 1.07 h", idle)
        self.assertIn("1890 latents, 807 CPU-hours, compute wall at 400 CPUs 2.02 h", loaded)
        self.assertLessEqual(sum(int(ln.split()[5]) for ln in out.splitlines()[1:6]), 400)
        self.assertIn("RAM: 26.5 GB per running task on average (CPU-time weighted); every array at its throttle holds "
                      "10576 GB at once", out)

    def test_refusals(self):
        sp = load("score_plan")
        os.remove(f"{self.tmp}/lat/{self.tags[2]}.npz")
        for name, text in [("wrong sha", f"{self.tags[0]}\t{'0' * 64}\n"), ("missing latent", f"{self.tags[2]}\n"),
                           ("not in manifest", "A1_nope\n"), ("twice", f"{self.tags[0]}\n{self.tags[0]}\n"),
                           ("no tags", "tag\n")]:
            with self.subTest(name):
                self.assertNotEqual(run(sp.main, self.args(text))[0], 0)
        self.assertEqual(run(sp.main, ["--manifest", self.man])[0], 2)          # arguments missing
        self.assertFalse(os.path.exists(f"{self.tmp}/out/list_sim1.tsv"))        # nothing is written on a refusal


STUB_SCORER = r'''
import json, os, subprocess, sys
mode = os.environ.get("STUB_MODE", "ok")
with open(os.environ["OUT_CSV"] + ".calls", "a") as f:
    f.write(json.dumps({k: v for k, v in os.environ.items()}) + "\n")
if mode == "fail":
    sys.exit(3)
here = os.path.dirname(os.path.abspath(__file__))
git = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
meta = json.loads(os.environ["META"])
tag = "other" if mode == "wrongtag" else os.environ["TAG"]
with open(os.environ["OUT_CSV"], "w") as f:
    f.write("tag,latent_sha256,kBET,scorer_git_sha,scorer_dirty\n%s,%s,0.1,%s,0\n" % (tag, meta["latent_sha256"], git))
'''


class ScoreArrayTask(unittest.TestCase):
    def tree(self):
        """A throwaway git clone-like tree: the real env.sh and score_array.sh, a stub scorer, a verified fake env."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        repo = f"{tmp}/repo"
        for f in ("env.sh", "score_array.sh"):
            os.makedirs(f"{repo}/cluster/jhpce", exist_ok=True)
            shutil.copy(os.path.join(JH, f), f"{repo}/cluster/jhpce/{f}")
        write(f"{repo}/scripts/score_scib_native.py", STUB_SCORER)
        git = ["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
        subprocess.run(git[:3] + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-qm", "x"], check=True)
        sha1 = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        os.makedirs(f"{tmp}/envs/wcd-score/bin")
        os.symlink(sys.executable, f"{tmp}/envs/wcd-score/bin/python")
        write(f"{tmp}/envs/wcd-score/.verified", "ok")
        npz, pre = write(f"{tmp}/lat/a.npz", "latent a"), write(f"{tmp}/pre/sim1__scib.h5ad", "prepped")
        meta = json.dumps(dict(latent_sha256=sha(npz), experiment="A1", task="sim1", arm="mmd", lam="3", cond="0", seed="100"))
        write(f"{tmp}/list.tsv", f"a\t{npz}\t{pre}\t{meta}\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "NUMBA_CPU", "STUB_"))}
        env.update(WCD_SCRATCH=f"{tmp}/scratch", WCD_ENVS=f"{tmp}/envs", SLURM_ARRAY_TASK_ID="1", HOME=os.path.expanduser("~"))
        return dict(tmp=tmp, repo=repo, sha=sha1, env=env, out=f"{tmp}/out", list=f"{tmp}/list.tsv", npz=npz, meta=meta)

    def go(self, t, **over):
        env = dict(t["env"], **{k: v for k, v in over.items() if k.isupper()})
        for k in [k for k, v in over.items() if k.isupper() and v is None]:
            env.pop(k)
        args = [over.get("expected", t["sha"]), t["list"], t["out"]]
        r = subprocess.run(["bash", f"{t['repo']}/cluster/jhpce/score_array.sh", *args], env=env, capture_output=True, text=True)
        return r.returncode

    def test_happy_path_uses_the_workstation_settings_and_skips_valid_files(self):
        t = self.tree()
        self.assertEqual(self.go(t), 0)
        with open(f"{t['out']}/scores/a.csv") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual((row["tag"], row["latent_sha256"], row["scorer_git_sha"]), ("a", sha(t["npz"]), t["sha"]))
        calls = read(f"{t['out']}/scores/a.csv.calls").splitlines()
        env = json.loads(calls[0])
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "KBET_SEED"):
            self.assertEqual(env[k], "1" if k.endswith("THREADS") else "0")
        self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["MKL_THREADING_LAYER"], env["KMP_AFFINITY"], env["META"], env["TAG"]),
                         ("", "SEQUENTIAL", "disabled", t["meta"], "a"))
        self.assertTrue(env["R_HOME"].endswith("envs/wcd-score/lib/R") and env["R_LIBS"].endswith("envs/Rlib_kbet"))
        self.assertNotIn("NUMBA_CPU_NAME", env)
        self.assertEqual(self.go(t), 0)                                     # valid file: skipped, scorer not run again
        self.assertEqual(len(read(f"{t['out']}/scores/a.csv.calls").splitlines()), 1)

    def test_refusals_leave_no_score_file(self):
        def edit(t, f, text):
            write(f"{t['repo']}/{f}", text)
        cases = [("wrong commit", dict(expected="0" * 40), 3),
                 ("dirty checkout", dict(pre=lambda t: edit(t, "scripts/score_scib_native.py", STUB_SCORER + "#")), 3),
                 ("not an array task", dict(SLURM_ARRAY_TASK_ID=None), 9),
                 ("env not verified", dict(pre=lambda t: os.remove(f"{t['tmp']}/envs/wcd-score/.verified")), 9),
                 ("job path under HOME", dict(HOME="TMP"), 9),
                 ("no such list line", dict(SLURM_ARRAY_TASK_ID="2"), 8),
                 ("bad tag", dict(pre=lambda t: write(t["list"], f"a/b\t{t['npz']}\t{t['tmp']}/pre/sim1__scib.h5ad\t{t['meta']}\n")), 8),
                 ("latent missing", dict(pre=lambda t: os.remove(t["npz"])), 8),
                 ("latent changed", dict(pre=lambda t: write(t["npz"], "other bytes")), 8),
                 ("meta without sha", dict(pre=lambda t: write(t["list"], f"a\t{t['npz']}\t{t['tmp']}/pre/sim1__scib.h5ad\t{{}}\n")), 8),
                 ("scorer fails", dict(STUB_MODE="fail"), 10), ("scorer writes another tag", dict(STUB_MODE="wrongtag"), 10)]
        for name, over, want in cases:
            with self.subTest(name):
                t = self.tree()
                if "pre" in over:
                    over.pop("pre")(t)
                if over.get("HOME") == "TMP":
                    over["HOME"] = t["tmp"]
                self.assertEqual(self.go(t, **over), want)
                self.assertFalse(os.path.exists(f"{t['out']}/scores/a.csv"))


if __name__ == "__main__":
    unittest.main()
