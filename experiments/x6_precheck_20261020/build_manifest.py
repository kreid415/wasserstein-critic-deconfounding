"""Build the 18-row pre-check manifest (PLAN.md). Usage:
python experiments/x6_precheck_20261020/build_manifest.py --v5 manifests/paper_manifest_stock_pilot_u5_b10_v5.tsv \
    --r4 <D>/tier12_v3/rules/r4_pass2/r4_a1_record.json --out <D>/tier12_v3/x6_precheck/manifest.tsv"""
import argparse, hashlib, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import build_paper_manifest as B   # noqa: E402  (COLS, finalize: the builder's own tag rule)
import prereg_rules as P            # noqa: E402  (fmt_lam)

ARM, EXP, SEED, TASKS = "discriminator_sn_critic_opt", "PC6", "100", ("atac_small", "immune", "sim1")


def build(v5, r4):
    m = pd.read_csv(v5, sep="\t", comment="#", dtype=str, keep_default_na=False)
    rec = json.load(open(r4))
    rows = []
    for task in TASKS:
        for cond in ("0", "1"):
            cell = rec["cells"][f"{task}|{cond}|discriminator"]
            for lam in (cell["lo"], cell["matched"], cell["hi"]):
                src = m[(m.experiment == "A1") & (m.arm == "discriminator") & (m.task == task) & (m.cond == cond)
                        & (m.seed == SEED) & (pd.to_numeric(m.lam, errors="coerce") == float(lam))]
                if len(src) != 1:
                    raise SystemExit(f"{task}|{cond} lam {lam}: {len(src)} pilot discriminator rows, need 1")
                r = src.iloc[0].to_dict()
                r.update(experiment=EXP, arm=ARM, lam=P.fmt_lam(lam))
                rows.append({c: r[c] for c in B.COLS})
    out = B.finalize(rows)
    return pd.DataFrame([{c: r[c] for c in m.columns} for r in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True); ap.add_argument("--r4", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    df = build(a.v5, a.r4)
    assert len(df) == 18 and df.tag.is_unique, len(df)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# x6 pre-check manifest (PLAN.md); from v5 sha256 {sha(a.v5)} and r4 record sha256 {sha(a.r4)}\n")
        df.to_csv(f, sep="\t", index=False)
    with open(os.path.splitext(a.out)[0] + "_tags.txt", "w") as f:
        f.write("\n".join(df.tag) + "\n")
    print(len(df), "rows ->", a.out)


if __name__ == "__main__":
    main()
