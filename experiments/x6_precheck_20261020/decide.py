"""Apply the pre-declared rule of PLAN.md to each arm of the pre-check (discriminator_sn_critic_opt, and pooled_sn at
n_critic 1 from the 2026-10-21 addendum, SI-107); each arm gets its own decision. Usage:
python experiments/x6_precheck_20261020/decide.py --manifest M.tsv --scores <out>/scores --failures <out>/failures.csv \
    --r4 r4_a1_record.json --out decision.json
decision.json: {"decisions": {arm: keep | shift_up | shift_down}, "per_arm": {arm: counts, counted cells, six cells}};
freeze_matched_lambda.py --precheck-decision reads the `decisions` key."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import prereg_rules as P   # noqa: E402

COUNTED = ("atac_small|0", "atac_small|1", "immune|0", "sim1|0")
ARMS = ("discriminator_sn_critic_opt", "pooled_sn")   # base arms in the pilot record: P.BASE_ARM (discriminator, pooled)


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


def summarise(O, rec):
    """O: outcomes table (arm, task, cond, lam, lamf, status, B); rec: the pilot record. One decision per arm of ARMS."""
    per_arm = {}
    for arm in ARMS:
        cells = {}
        for (task, cond), g in O[O.arm == arm].groupby(["task", "cond"]):
            key = f"{task}|{cond}"
            t = rec["targets"][key]
            base = rec["cells"][f"{key}|{P.BASE_ARM[arm]}"]
            pairs = [(float(r.lam), None if r.status == "failed" else float(r.B)) for r in g.sort_values("lamf").itertuples()]
            cells[key] = dict(bstar=t["bstar"], low_signal=bool(t["bstar_low_signal"]), points=pairs,
                              status=cell_status(pairs, t["bstar"]),
                              base_arm_matched=base["matched"], base_arm_matched_B_3seed=base["matched_B"])
        assert sorted(cells) == sorted(COUNTED + ("immune|1", "sim1|1")), (arm, sorted(cells))
        dec, n_low, n_high = decide({k: cells[k]["status"] for k in COUNTED})
        per_arm[arm] = dict(decision=dec, n_unreached_low=n_low, n_unreached_high=n_high, counted=list(COUNTED), cells=cells)
    return dict(decisions={arm: r["decision"] for arm, r in per_arm.items()}, per_arm=per_arm)


def main():
    ap = argparse.ArgumentParser()
    for k in ("manifest", "scores", "failures", "r4", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    M, _ = P.read_manifest([a.manifest])
    S, _ = P.read_scores(a.scores)
    F = P.read_failures(a.failures)
    out = summarise(P.outcomes(M, S, F, ["PC6"]), json.load(open(a.r4)))
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    json.dump(out, open(a.out, "w"), indent=1)
    print(out["decisions"])


if __name__ == "__main__":
    main()
