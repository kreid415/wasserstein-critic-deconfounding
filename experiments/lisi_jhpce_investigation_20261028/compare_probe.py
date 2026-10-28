"""Exact-equality comparison of probe_lisi.py outputs. A condition is a set of per-latent JSON files named
<condition>_<latent>.json in one or more directories (JHPCE and workstation outputs are unpacked side by side).
Usage: compare_probe.py OUT.csv STORED_REF.json LATENTS.sha256 LABEL=DIR:CONDITION [LABEL=DIR:CONDITION ...]
 STORED_REF.json: {tag: {iLISI, cLISI, latent_sha256}} read from the workstation's production score files; a stored row is
 used only where its latent_sha256 equals the one in LATENTS.sha256 (3 sinkhorn latents were re-made after their stored score).
Prints, for every pair of labels, how many latents have bit-identical iLISI and cLISI (float.hex equality) and the
helper-output hash equality; exits 1 if an expected-equal pair is not equal, or if the negative control is accepted."""
import csv, glob, json, os, sys
EXPECT_EQUAL = [("jh_final_1", "jh_final_2"), ("jh_final_1", "ws_nat_1"), ("jh_final_2", "ws_nat_2"), ("ws_nat_1", "ws_nat_2"),
                ("ws_final_1", "ws_nat_1"), ("jh_final_1", "ws_final_1")]
EXPECT_DIFFERENT = [("jh_nat_1", "ws_nat_1")]          # the unfixed JHPCE run is the known-bad input: must be rejected

def load(spec):
    label, rest = spec.split("=", 1); d, cond = rest.rsplit(":", 1); out = {}
    for f in glob.glob(f"{d}/{cond}_*.json"):
        r = json.load(open(f))["rows"][0]; out[r["latent"][:-4]] = r
    return label, out

def same(a, b):
    """Per latent: iLISI, cLISI (float.hex strings) and helper output hashes identical."""
    n = {k: 0 for k in ("ilisi", "clisi", "helper_rows")}
    for t in a:
        n["ilisi"] += a[t]["ilisi_hex"] == b[t]["ilisi_hex"]; n["clisi"] += a[t]["clisi_hex"] == b[t]["clisi_hex"]
        n["helper_rows"] += all(a[t][m + "_helper"][k] == b[t][m + "_helper"][k] for m in ("ilisi", "clisi") for k in ("indices_sha", "distances_sha"))
    return n

if __name__ == "__main__":
    out_csv, ref_json, sha_file, specs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    conds = dict(load(s) for s in specs); ref = json.load(open(ref_json))
    want = {l.split()[1][:-4]: l.split()[0] for l in open(sha_file) if l.strip()}
    assert len(want) == 20 and all(len(c) == 20 and set(c) == set(want) for c in conds.values()), {k: len(v) for k, v in conds.items()}
    rows, failed, labels = [], [], list(conds)
    for i, x in enumerate(labels):
        for y in labels[i + 1:]:
            n = same(conds[x], conds[y]); rows.append(dict(a=x, b=y, n_latents=20, **n)); allsame = all(v == 20 for v in n.values())
            if {(x, y), (y, x)} & set(EXPECT_EQUAL) and not allsame: failed.append((x, y, n))
            if {(x, y), (y, x)} & set(EXPECT_DIFFERENT) and allsame: failed.append(("negative control accepted", x, y))
    usable = [t for t in want if t in ref and ref[t]["latent_sha256"] == want[t]]
    for lab, c in conds.items():
        rows.append(dict(a=lab, b="stored_ws_production", n_latents=len(usable),
                         ilisi=sum(float(ref[t]["iLISI"]).hex() == c[t]["ilisi_hex"] for t in usable),
                         clisi=sum(float(ref[t]["cLISI"]).hex() == c[t]["clisi_hex"] for t in usable), helper_rows=""))
    with open(out_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)
    if failed: print("FAILED:", failed); sys.exit(1)
