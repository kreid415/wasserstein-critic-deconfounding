#!/usr/bin/env python
"""Map the penalty-strength pilot (experiment A1, three tasks) to the main-benchmark input format of scripts/analysis_x1.py.

The pilot is not blinded (tier12_v3/rules/r4_pass2); this script never reads a main-benchmark (X1) score. Mapping, by column:
  manifest   A1 rows of atac_small, immune, sim1 -> experiment X1; seed 100-102 -> 0-2; tags are kept. Rows kept: plain scVI (arm none)
             and the six penalties at the six strengths of r1_pass2 families.<family>|<cond>.x1_grid. Grid-edge extension rows are
             then added in rounds, as freeze_matched_lambda.py does, from the pilot rows at the strength R4 asks for.
  scores     the pilot's own score files of those tags (never changed); latents: the pilot's latent directories.
  failures   r4_pass2/failures_merged.csv.
  records    R1 = r1_pass2/r1_record.json; R4 stage x1 = prereg_rules.r4_stage on the mapped rows.
STAND-INS (not pilot data): the pilot has no scvi_adv rows and no X13 baselines (scANVI, sysVI, Harmony, Scanorama, PCA) that the
analysis requires for every task. They come from make_synthetic_x1 for the same three tasks, in their own manifest/score
directories (names start with standin_), and are the only synthetic part.

Usage: python scripts/dryrun/build_pilot_inputs.py --data-root <D> --out <dir>
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_synthetic_x1 as SYN  # noqa: E402
P = SYN.P
PILOT_TASKS = ("atac_small", "immune", "sim1")
# build_metric_curves.py column -> pilot score column; dec lin/nl = unconditioned/conditioned decoder (cond 0/1) for this dry run only
LEGACY = {"ilisi": "iLISI", "asw_batch": "ASW_label/batch", "graph_conn": "graph_conn", "pcr": "PCR_batch", "clisi": "cLISI",
          "ari": "ARI_cluster/label", "nmi": "NMI_cluster/label", "asw_celltype": "ASW_label",
          "isolated_asw": "isolated_label_silhouette", "isolated_f1": "isolated_label_F1"}


def main(argv=None):  # pragma: no cover  # needs the pilot data on the workstation (tier12_v3/rules)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rules = os.path.join(a.data_root, "tier12_v3", "rules")
    for d in ("manifests", "scores"):
        os.makedirs(os.path.join(a.out, d))
    r1_path = os.path.join(a.out, "r1_record.json")
    shutil.copy(os.path.join(rules, "r1_pass2", "r1_record.json"), r1_path)
    r1 = P.read_record(r1_path, "R1")
    shutil.copy(os.path.join(rules, "r4_pass2", "failures_merged.csv"), os.path.join(a.out, "failures.csv"))
    F = P.read_failures(os.path.join(a.out, "failures.csv"))
    # pilot rows, X1 format
    pm, _ = P.read_manifest(os.path.join(rules, "r4_pass2", "m.r4a1.tsv"))
    pm = pm[(pm.experiment == "A1") & pm.task.isin(PILOT_TASKS)].copy()
    pm["seed"] = (pm.seed.astype(int) - 100).astype(str)
    pm["experiment"] = "X1"
    lam = pm.lam.map(P._lamf)
    in_grid = [r.arm == "none" or any(math.isclose(l, g, rel_tol=1e-9) for g in r1["families"][f"{P.FAMILY_OF[r.arm]}|{r.cond}"]["x1_grid"])
               for r, l in zip(pm.itertuples(), lam)]
    files = [l.strip() for l in open(os.path.join(rules, "r4_pass2", "scores_list.txt")) if l.strip()]
    by_tag = {os.path.basename(f)[:-4]: f for f in files}
    pilot_scores = pd.concat([P._read_score_file(f) for f in files], ignore_index=True)
    pilot_scores = pilot_scores[pilot_scores.tag.isin(set(pm.tag))]
    legacy = pm[["tag", "task", "arm", "lam", "cond"]].merge(pilot_scores[["tag", *LEGACY.values()]], on="tag", validate="one_to_one")
    legacy = legacy.rename(columns={"task": "dataset", "arm": "adv", **{v: k for k, v in LEGACY.items()}})
    legacy = legacy.assign(dec=legacy.pop("cond").map({"1": "nl", "0": "lin"}), lam=legacy.lam.map(P._lamf))
    legacy.to_csv(os.path.join(a.out, "legacy_scored_pilot.csv"), index=False)
    rounds, scores, unmet = [pm[in_grid]], [], []
    for rnd in range(4):
        name = "x1" if rnd == 0 else f"ext_round{rnd}"
        Mr = rounds[-1]
        os.makedirs(os.path.join(a.out, "scores", name))
        s = pilot_scores[pilot_scores.tag.isin(set(Mr.tag))]
        for tag in Mr.tag:         # the pilot's own score files, linked, not copied
            os.symlink(by_tag[tag], os.path.join(a.out, "scores", name, tag + ".csv"))
        P.write_manifest(Mr, ["# pilot rows in the X1 format (scripts/dryrun/build_pilot_inputs.py)"], os.path.join(a.out, "manifests", name + ".tsv"), f"round {rnd}")
        scores.append(s)
        Mall = pd.concat(rounds, ignore_index=True)
        out = P.outcomes(Mall, pd.concat(scores, ignore_index=True), F, ["X1"])
        r4 = P.r4_stage(out, "x1", r1)
        if r4["status"] == "frozen":
            break
        add = [pm[(pm.task == e["task"]) & (pm.cond.astype(int) == int(e["cond"])) & (pm.arm == e["arm"]) & (lam == e["lam"])]
               for e in r4["extensions"]]
        unmet += [e for e, x in zip(r4["extensions"], add) if x.empty]
        if not any(len(x) for x in add):
            break
        rounds.append(pd.concat([x for x in add if len(x)], ignore_index=True))
    else:
        raise SystemExit("R4 did not freeze in 4 rounds")
    unmet = [dict(t) for t in sorted({tuple(sorted(e.items())) for e in unmet})]
    if unmet:    # the pilot has no fit at a strength R4 asks for: freeze by hand, say so in the record (targets use the rows that exist)
        r4 = dict(r4, status="frozen", unmet_extensions=unmet)
    # stand-ins for scvi_adv and X13 (see the docstring)
    Ms = SYN.build_manifest()
    Ms = Ms[Ms.task.isin(PILOT_TASKS) & ((Ms.experiment == "X13") | (Ms.arm == "scvi_adv"))]
    os.makedirs(os.path.join(a.out, "scores", "standin_x13_scvi_adv"))
    os.makedirs(os.path.join(a.out, "latents_standin"))
    ss, fs = SYN.score_rows(Ms, os.path.join(a.out, "latents_standin"))
    P._require(fs.empty, "stand-in rows must not fail")
    # check_provenance refuses a rule input scored by more than one scorer: the stand-ins carry the pilot scorer's values (flagged by
    # the extra column synthetic_standin; the equivalent commit of docs/EQUIVALENT_COMMITS.json is not used as the value)
    ref = pilot_scores[pilot_scores.scorer_git_sha != json.load(open(P.EQUIVALENT_COMMITS))["commit"]].iloc[0]
    for c in P.PROVENANCE_COLS:
        ss[c] = ref[c]
    ss["synthetic_standin"] = 1
    for tag, row in ss.groupby("tag"):
        row.to_csv(os.path.join(a.out, "scores", "standin_x13_scvi_adv", tag + ".csv"), index=False)
    P.write_manifest(Ms, ["# STAND-IN rows (synthetic), not pilot data"], os.path.join(a.out, "manifests", "standin_x13_scvi_adv.tsv"), "stand-in")
    P.write_record(os.path.join(a.out, "r4_x1_record.json"), r4, {"manifest": [os.path.join(a.out, "manifests", n) for n in sorted(os.listdir(os.path.join(a.out, "manifests")))]})
    # one latent per tag, taken next to the score file that the score row came from (a tag can have an older latent elsewhere,
    # e.g. the superseded Sinkhorn fits of tier12_v2/A1)
    os.makedirs(os.path.join(a.out, "latents"))
    for tag in sorted(set(pm.tag)):
        src = os.path.join(os.path.dirname(os.path.dirname(by_tag[tag])), "latents", tag + ".npz")
        if os.path.exists(src):          # failed fits have none; the analysis asks only for the ones it uses
            os.symlink(src, os.path.join(a.out, "latents", tag + ".npz"))
    json.dump(dict(latent_dirs=[os.path.join(a.out, "latents"), os.path.join(a.out, "latents_standin")], pilot_tasks=PILOT_TASKS), open(os.path.join(a.out, "pilot_meta.json"), "w"))
    print(f"pilot inputs: {len(Mall)} mapped pilot rows in {len(scores)} rounds + {len(Ms)} stand-in rows, R4 frozen with {len(unmet)} unmet extension requests -> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover  # command-line entry
    sys.exit(main())
