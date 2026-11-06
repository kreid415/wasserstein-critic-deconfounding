import csv, hashlib, json, os
# Scoring lists for the 25 JHPCE replication-extension fits (one list per task); run as a 1-CPU helper job (make_list_ext.py).
R = "/fastscratch/myscratch/kreid/wcd/replication_20261102"; OUT = "/fastscratch/myscratch/kreid/wcd/tier12/sweep_replication"; S = "/fastscratch/myscratch/kreid/wcd"
EXPECT = {"immune_hum_mou": 5, "pancreas": 15, "sim2": 5}


def build_lists(R=R, OUT=OUT, S=S, expect=EXPECT):
    SC = S + "/sweep_replication_scores/scores"
    rr = list(csv.reader([l for l in open(R + "/sweep_replication_ext1.tsv") if not l.startswith("#")], delimiter="\t")); h = rr[0]
    M = {d["tag"]: d for d in (dict(zip(h, x)) for x in rr[1:]) if d["task"] in expect}
    assert len(M) == sum(expect.values()), len(M)
    for t in M:
        assert os.path.exists(f"{OUT}/latents/{t}.npz"), f"latent missing: {t}"
        assert not os.path.exists(f"{SC}/{t}.csv"), f"already scored: {t}"
    for task, n in expect.items():
        out = []
        for t, d in sorted(M.items()):
            if d["task"] != task:
                continue
            lat = f"{OUT}/latents/{t}.npz"; sha = hashlib.sha256(open(lat, "rb").read()).hexdigest()
            meta = dict(latent_sha256=sha, experiment=d["experiment"], task=d["task"], arm=d["arm"], lam=d["lam"], cond=d["cond"], seed=d["seed"])
            out.append("\t".join([t, lat, f"{S}/prepped_scib/{d['task']}__scib.h5ad", json.dumps(meta)]))
        assert len(out) == n, (task, len(out))
        open(f"{R}/scoring/ext_{task}.tsv", "w").write("\n".join(out) + "\n"); print("wrote", task, len(out))


if __name__ == "__main__":  # pragma: no cover  # runs on the JHPCE login/compute node
    build_lists()
