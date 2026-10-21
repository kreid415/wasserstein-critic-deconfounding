"""Apply the pre-declared rule of PLAN.md. Usage:
python experiments/x6_precheck_20261020/decide.py --manifest M.tsv --scores <out>/scores --failures <out>/failures.csv \
    --r4 r4_a1_record.json --out decision.json"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import prereg_rules as P   # noqa: E402

COUNTED = ("atac_small|0", "atac_small|1", "immune|0", "sim1|0")


def cell_status(pairs, bstar):
    """pairs: [(lam, B or None)]; B None = failed fit."""
    bs = [b for _, b in pairs if b is not None]
    if len(bs) < 2:
        return "not_readable"
    if max(bs) < bstar:
        return "unreached_low"
    if min(bs) > bstar:
        return "unreached_high"
    return "reached"


def decide(statuses):
    """statuses: {cell: status} for the counted cells."""
    n_low = sum(s == "unreached_low" for s in statuses.values())
    n_high = sum(s == "unreached_high" for s in statuses.values())
    return ("shift_up" if n_low >= 3 else "shift_down" if n_high >= 3 else "keep"), n_low, n_high


def main():
    ap = argparse.ArgumentParser()
    for k in ("manifest", "scores", "failures", "r4", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    M, _ = P.read_manifest([a.manifest])
    S, _ = P.read_scores(a.scores)
    F = P.read_failures(a.failures)
    O = P.outcomes(M, S, F, ["PC6"])
    rec = json.load(open(a.r4))
    cells = {}
    for (task, cond), g in O.groupby(["task", "cond"]):
        key = f"{task}|{cond}"
        t = rec["targets"][key]
        pairs = [(float(r.lam), None if r.status == "failed" else float(r.B)) for r in g.sort_values("lamf").itertuples()]
        cells[key] = dict(bstar=t["bstar"], low_signal=bool(t["bstar_low_signal"]), points=pairs,
                          status=cell_status(pairs, t["bstar"]),
                          plain_disc_matched=rec["cells"][key + "|discriminator"]["matched"],
                          plain_disc_matched_B_3seed=rec["cells"][key + "|discriminator"]["matched_B"])
    assert sorted(cells) == sorted(COUNTED + ("immune|1", "sim1|1")), sorted(cells)
    dec, n_low, n_high = decide({k: cells[k]["status"] for k in COUNTED})
    out = dict(decision=dec, n_unreached_low=n_low, n_unreached_high=n_high, counted=list(COUNTED), cells=cells)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    json.dump(out, open(a.out, "w"), indent=1)
    print(dec, n_low, n_high)


if __name__ == "__main__":
    main()
