import csv, hashlib, json, os, sys
R="/fastscratch/myscratch/kreid/wcd/replication_20261102"; OUT="/fastscratch/myscratch/kreid/wcd/tier12/sweep_replication"; S="/fastscratch/myscratch/kreid/wcd"
SC=S+"/sweep_replication_scores/scores"; TASK="immune_hum_mou"; EXPECT_TOTAL=125; EXPECT_NEW=40
L=[l for l in open(R+"/sweep_replication_x1.tsv") if not l.startswith("#")]; rr=list(csv.reader(L,delimiter="\t")); h=rr[0]
rows=[dict(zip(h,x)) for x in rr[1:]]; M={d["tag"]:d for d in rows if d["task"]==TASK}
assert len(M)==EXPECT_TOTAL, f"manifest has {len(M)} {TASK} rows, expected {EXPECT_TOTAL}"
prev=set()
for f in ("w1","w2"):
    for l in open(f"{R}/scoring/{f}_{TASK}.tsv"):
        if l.strip(): prev.add(l.split("\t")[0])
assert prev<=set(M), f"{len(prev-set(M))} earlier-list tags not in manifest"
missing_lat=[t for t in M if not os.path.exists(f"{OUT}/latents/{t}.npz")]
assert not missing_lat, f"latents missing: {missing_lat[:3]} ({len(missing_lat)})"
scored={t for t in M if os.path.exists(f"{SC}/{t}.csv")}
assert scored<=set(M)
new=sorted(set(M)-scored)
print("manifest",len(M),"in w1/w2",len(prev),"scored",len(scored),"unscored",len(new),"unscored but in w1/w2:",len(set(new)&prev))
assert len(new)==EXPECT_NEW, f"unscored {len(new)} != {EXPECT_NEW}"
assert not (set(new)&prev), "an unscored tag is already in an earlier list"
out=[]
for t in new:
    d=M[t]; lat=f"{OUT}/latents/{t}.npz"
    sha=hashlib.sha256(open(lat,"rb").read()).hexdigest()
    meta=dict(latent_sha256=sha, experiment=d["experiment"], task=d["task"], arm=d["arm"], lam=d["lam"], cond=d["cond"], seed=d["seed"])
    out.append("\t".join([t, lat, f"{S}/prepped_scib/{d['task']}__scib.h5ad", json.dumps(meta)]))
open(f"{R}/scoring/w3_{TASK}.tsv","w").write("\n".join(out)+"\n"); print("wrote w3", len(out))
