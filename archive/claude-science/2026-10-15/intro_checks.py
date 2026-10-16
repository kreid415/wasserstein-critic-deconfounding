"""Read-only checks behind the Introduction review of paper/paper.tex at add2f73.
Usage: python intro_checks.py <repo_dir> <scvi_wheel_dir> <out.json>
repo_dir = clone of kreid415/wasserstein-critic-deconfounding checked out at add2f73 (non-paper files equal 08f1e3c);
scvi_wheel_dir = folder holding scvi_tools-1.4.2-py3-none-any.whl (pip download scvi-tools==1.4.2 --no-deps).
"""
import glob, hashlib, json, sys, zipfile
import pandas as pd

repo, wheel_dir, out = sys.argv[1:4]
res = {}

# manifest of record
mp = f"{repo}/manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv"
res["manifest_sha256_prefix"] = hashlib.sha256(open(mp, "rb").read()).hexdigest()[:16]
m = pd.read_csv(mp, sep="\t", skiprows=1)
res["manifest_rows"] = len(m)
x1 = m[m.experiment == "X1"]
res["x1_rows"] = len(x1)
res["x1_rows_by_arm"] = x1.arm.value_counts().to_dict()
res["x1_n_critic_by_arm"] = {a: sorted(g.n_critic.unique().tolist()) for a, g in x1.groupby("arm")}
sv = x1[x1.arm == "scvi_adv"]
res["scvi_adv_rows"] = len(sv)
res["scvi_adv_adv_input_values"] = sv.adv_input.unique().tolist()
res["scvi_adv_cond_values"] = sorted(sv.cond.unique().tolist())
au = x1[x1.arm.isin(["discriminator", "reference", "pooled", "barycenter", "mmd", "sinkhorn"])]
res["authored_adv_input_values"] = au.adv_input.unique().tolist()
res["authored_zstd_values"] = au.zstd.unique().tolist()
res["authored_cond_values"] = sorted(au.cond.unique().tolist())
ep = x1.groupby("task").max_epochs.agg(lambda s: sorted(set(s)))
res["x1_epochs_by_task"] = ep.to_dict()
res["x1_batch_sizes"] = sorted(x1.batch_size.unique().tolist())
res["x1_n_latent"] = sorted(x1.n_latent.unique().tolist())
chk = {}
for arm in ["sysvi", "scanvi"]:
    g = m[(m.experiment == "X13") & (m.arm == arm)].groupby("task").agg(
        ep=("max_epochs", lambda s: sorted(set(s))), bs=("batch_size", lambda s: sorted(set(s))), nl=("n_latent", lambda s: sorted(set(s))))
    chk[arm] = {t: bool(g.loc[t, "ep"] == ep[t] and g.loc[t, "bs"] == [128] and g.loc[t, "nl"] == [10]) for t in g.index}
res["x13_matches_x1_epochs_batch_latent"] = chk
x6 = m[m.experiment == "X6"]
res["x6_arm_steps_rows"] = {f"{a}[{s}]": int(n) for (a, s), n in x6.groupby(["arm", "n_critic"]).size().items()}
res["x7_arms"] = sorted(m[m.experiment == "X7"].arm.unique().tolist())

# frozen R4 record on A1
rec = json.load(open(f"{repo}/docs/prereg/r4_a1_record.json"))
blocks = {}
for k, t in rec["targets"].items():
    M = t["M"]
    blocks[k] = dict(b0=t["b0"], b_common=t["b_common"], diff=t["b_common"] - t["b0"], setter=min(M, key=M.get),
                     sinkhorn_M=M["sinkhorn"], low_signal=t["bstar_low_signal"])
res["r4_a1_blocks"] = blocks
res["r4_a1_status"], res["r4_a1_extensions"], res["r4_a1_git_sha"] = rec["status"], len(rec["extensions"]), rec["git_sha"]

# code lines cited by the Introduction
pl = open(f"{repo}/scripts/scvi_adversarial_plan.py", encoding="utf-8").read().split("\n")
res["plan_L359"] = pl[358].strip()
res["plan_L292_298"] = [x.strip() for x in pl[291:298]]
res["plan_L348_351"] = [x.strip() for x in pl[347:351]]
res["plan_L8_9"] = [x.strip() for x in pl[7:9]]
whl = zipfile.ZipFile(glob.glob(f"{wheel_dir}/scvi_tools-1.4.2*.whl")[0])
tp = whl.read("scvi/train/_trainingplans.py").decode().split("\n")
bc = whl.read("scvi/nn/_base_components.py").decode().split("\n")
vae = whl.read("scvi/module/_vae.py").decode().split("\n")
res["scvi_tp_L689"] = tp[688].strip()
res["scvi_tp_L674_678"] = [x.strip() for x in tp[673:678]]
res["scvi_bc_L304_305"] = [x.strip() for x in bc[303:305]]
res["scvi_vae_L390"] = vae[389].strip()
res["scvi_vae_L413"] = vae[412].strip()
json.dump(res, open(out, "w"), indent=1, default=str)
print("wrote", out)
