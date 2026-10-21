"""Build the 36-row pre-check manifest (PLAN.md and its 2026-10-21 addendum) from the frozen pilot rule-output manifest,
which also holds the pilot extension rows (e.g. strength 0.03): 18 rows of discriminator_sn_critic_opt (copied from the
pilot discriminator rows) and 18 rows of pooled_sn at n_critic 1 (copied from the pilot pooled rows, SI-107). Usage:
python experiments/x6_precheck_20261020/build_manifest.py --source <D>/tier12_v3/rules/r4_pass2/m.r4a1.tsv \
    --r4 <D>/tier12_v3/rules/r4_pass2/r4_a1_record.json --out <D>/tier12_v3/x6_precheck/manifest.tsv"""
import argparse, hashlib, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import build_paper_manifest as B   # noqa: E402  (COLS, finalize: the builder's own tag rule)
import prereg_rules as P            # noqa: E402  (fmt_lam)

EXP, SEED, TASKS = "PC6", "100", ("atac_small", "immune", "sim1")
# (arm, its base arm in the pilot, n_critic of the arm's rows or None to copy the pilot row's value)
ARMS = (("discriminator_sn_critic_opt", "discriminator", None), ("pooled_sn", "pooled", "1"))


def build(source, r4):
    m = pd.read_csv(source, sep="\t", comment="#", dtype=str, keep_default_na=False)
    rec = json.load(open(r4))
    rows = []
    for arm, base, n_critic in ARMS:
        for task in TASKS:
            for cond in ("0", "1"):
                cell = rec["cells"][f"{task}|{cond}|{base}"]
                for lam in (cell["lo"], cell["matched"], cell["hi"]):
                    src = m[(m.experiment == "A1") & (m.arm == base) & (m.task == task) & (m.cond == cond)
                            & (m.seed == SEED) & (pd.to_numeric(m.lam, errors="coerce") == float(lam))]
                    if len(src) != 1:
                        raise SystemExit(f"{task}|{cond} lam {lam}: {len(src)} pilot {base} rows, need 1")
                    r = src.iloc[0].to_dict()
                    r.update(experiment=EXP, arm=arm, lam=P.fmt_lam(lam))
                    if n_critic:
                        r["n_critic"] = n_critic
                    rows.append({c: r[c] for c in B.COLS})
    out = B.finalize(rows)
    return pd.DataFrame([{c: r[c] for c in m.columns} for r in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--r4", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    df = build(a.source, a.r4)
    assert len(df) == 36 and df.tag.is_unique, len(df)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# x6 pre-check manifest (PLAN.md); from source {a.source} sha256 {sha(a.source)} and r4 record sha256 {sha(a.r4)}\n")
        df.to_csv(f, sep="\t", index=False)
    with open(os.path.splitext(a.out)[0] + "_tags.txt", "w") as f:
        f.write("\n".join(df.tag) + "\n")
    print(len(df), "rows ->", a.out)


if __name__ == "__main__":
    main()
