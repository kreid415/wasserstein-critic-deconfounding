"""Machine checks behind the d170dd5 re-check (read-only). Usage:
   python recheck_checks.py <repo_dir at d170dd5> <july_dir (unzipped Topology_Matters)> <out.json>
Reads: manifest of record, frozen rule-R4 record, fingerprints json, paper.tex, refs.bib, evidence.md, July tex."""
import json, re, sys, os, hashlib
import pandas as pd

repo, july, out = sys.argv[1:4]
R = {}
# 1. manifest of record
mp = os.path.join(repo, "manifests/paper_manifest_stock_pilot_u5_b10_v3.tsv")
R["manifest_sha256_16"] = hashlib.sha256(open(mp, "rb").read()).hexdigest()[:16]
m = pd.read_csv(mp, sep="\t", skiprows=1, low_memory=False)
R["manifest_rows"] = len(m)
x1 = m[m.experiment == "X1"]
R["x1_rows"] = len(x1)
R["x1_n_critic_by_arm"] = {a: sorted(map(int, g.n_critic.unique())) for a, g in x1.groupby("arm")}
sa = x1[x1.arm == "scvi_adv"]
R["x1_scvi_adv_rows"] = len(sa); R["x1_scvi_adv_adv_input"] = sorted(sa.adv_input.astype(str).unique())
x6 = m[m.experiment == "X6"]
R["x6_n_critic_by_arm"] = {a: sorted(map(int, g.n_critic.unique())) for a, g in x6.groupby("arm")}
R["a2_tasks"] = sorted(m[m.experiment == "A2"].task.unique()); R["a3_tasks"] = sorted(m[m.experiment == "A3"].task.unique())
x3 = m[m.experiment == "X3"]
nc = {}
for _, r in x3.iterrows():
    s = json.loads(r["extra"])["subsample"]; nc.setdefault(r["task"], set()).add(s["n_cells"])
R["x3_n_cells_per_task"] = {k: sorted(v) for k, v in nc.items()}
# 2. frozen rule-R4 record: b_common = min over arms of M; flagged blocks
r4 = json.load(open(os.path.join(repo, "docs/prereg/r4_a1_record.json")))
blocks = {}
for k, v in r4["targets"].items():
    setter = min(v["M"], key=v["M"].get)
    blocks[k] = dict(b0=v["b0"], b_common=v["b_common"], diff=v["b_common"] - v["b0"], setter=setter,
                     b_common_is_min_M=abs(v["b_common"] - min(v["M"].values())) < 1e-12, flagged=v["bstar_low_signal"])
R["r4_blocks"] = blocks
R["r4_setter_counts"] = {a: sum(1 for b in blocks.values() if b["setter"] == a) for a in ["barycenter", "mmd", "sinkhorn", "discriminator", "pooled", "reference"]}
R["r4_git_sha"] = r4["git_sha"]; R["r4_status"] = r4["status"]
# 3. fingerprints
fp = json.load(open(os.path.join(repo, "docs/prepped_fingerprints_scib.json")))
R["fingerprint_tasks"] = sorted(k.split("__")[0] for k in fp)
R["atac_source_files"] = {k.split("__")[0]: fp[k]["uns"].get("source_file") for k in fp if k.startswith("atac")}
# 4. cites vs bib; entry and DOI counts
tex = open(os.path.join(repo, "paper/paper.tex")).read()
bib = open(os.path.join(repo, "paper/refs.bib")).read()
ents = re.findall(r"^@\w+\{([^,]+),(.*?)\n\}\n", bib, flags=re.S | re.M)
keys = [k for k, _ in ents]
R["bib_entries"] = len(keys); R["bib_entries_with_doi"] = sum(1 for _, b in ents if re.search(r"\bdoi\s*=", b))
nc_tex = "\n".join(l for l in tex.split("\n") if not l.lstrip().startswith("%"))
cited = {k.strip() for m_ in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", nc_tex) for k in m_.group(1).split(",")}
R["cited_keys"] = len(cited); R["cited_not_in_bib"] = sorted(cited - set(keys)); R["bib_not_cited"] = sorted(set(keys) - cited)
R["corani_cites"] = len(re.findall(r"corani2017statistical", nc_tex))
R["non_ascii_lines"] = {f: sum(1 for l in open(os.path.join(repo, "paper", f), encoding="utf8") if any(ord(c) > 127 for c in l)) for f in ["paper.tex", "refs.bib", "evidence.md"]}
# 5. scCRAFT mentions outside comments, and whether each carries the naming-decision marker
R["sccraft_lines"] = {i + 1: ("naming marker" if re.search(r"names scCRAFT", l) else "no naming marker")
                      for i, l in enumerate(tex.split("\n")) if "scCRAFT" in l and not l.lstrip().startswith("%")}
# 6. July manuscript: arms named MMD or Sinkhorn
jt = open(os.path.join(july, "oup-authoring-template/oup-authoring-template.tex")).read() + open(os.path.join(july, "oup-authoring-template/supplement.tex")).read()
R["july_mmd_sinkhorn_hits"] = len(re.findall(r"MMD|Sinkhorn", jt))
json.dump(R, open(out, "w"), indent=1, default=str)
print("wrote", out)
